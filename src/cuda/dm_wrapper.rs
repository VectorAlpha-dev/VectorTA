#![cfg(feature = "cuda")]

//! CUDA wrapper for Directional Movement (DM): produces +DM and -DM Wilder-smoothed series.
//!
//! Parity goals (aligned with ALMA/CWMA wrappers):
//! - Stream NON_BLOCKING
//! - PTX load via include_str!(concat!(env!("OUT_DIR"), "/dm_kernel.ptx")) with
//!   DetermineTargetFromContext + OptLevel O2, falling back gracefully.
//! - Policy enums for kernel selection; Auto default.
//! - VRAM checks using mem_get_info() with ~64MB headroom.
//! - Warmup/NaN semantics match scalar: write NaN before warm = first + period - 1.

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::dm::{DmBatchRange, DmParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    Plain { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub struct CudaDmPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaDmPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

#[derive(Debug)]
pub enum CudaDmError {
    Cuda(String),
    InvalidInput(String),
}
impl fmt::Display for CudaDmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaDmError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaDmError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}
impl std::error::Error for CudaDmError {}

/// Pair of VRAM-backed arrays (+DM and -DM) produced by the DM kernels.
pub struct DeviceDmPair {
    pub plus: DeviceArrayF32,
    pub minus: DeviceArrayF32,
}
impl DeviceDmPair {
    #[inline]
    pub fn rows(&self) -> usize {
        self.plus.rows
    }
    #[inline]
    pub fn cols(&self) -> usize {
        self.plus.cols
    }
}

pub struct CudaDm {
    module: Module,
    stream: Stream,
    _context: Context,
    policy: CudaDmPolicy,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaDm {
    pub fn new(device_id: usize) -> Result<Self, CudaDmError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaDmError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaDmError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaDmError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/dm_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => Module::from_ptx(ptx, &[]).map_err(|e| CudaDmError::Cuda(e.to_string()))?,
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaDmError::Cuda(e.to_string()))?;
        Ok(Self {
            module,
            stream,
            _context: context,
            policy: CudaDmPolicy::default(),
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn set_policy(&mut self, policy: CudaDmPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaDmPolicy {
        &self.policy
    }
    pub fn synchronize(&self) -> Result<(), CudaDmError> {
        self.stream
            .synchronize()
            .map_err(|e| CudaDmError::Cuda(e.to_string()))
    }

    #[inline]
    fn device_mem_ok(bytes: usize) -> bool {
        match mem_get_info() {
            Ok((free, _)) => bytes.saturating_add(64 * 1024 * 1024) <= free,
            Err(_) => true,
        }
    }

    fn prepare_batch(
        high: &[f32],
        low: &[f32],
        sweep: &DmBatchRange,
    ) -> Result<(Vec<DmParams>, usize, usize), CudaDmError> {
        if high.is_empty() || low.is_empty() || high.len() != low.len() {
            return Err(CudaDmError::InvalidInput(
                "empty or mismatched inputs".into(),
            ));
        }
        let len = high.len();
        let first_valid = (0..len)
            .find(|&i| !high[i].is_nan() && !low[i].is_nan())
            .ok_or_else(|| CudaDmError::InvalidInput("all values are NaN".into()))?;
        let (start, end, step) = sweep.period;
        let periods: Vec<usize> = if step == 0 || start == end {
            vec![start]
        } else {
            (start..=end).step_by(step).collect()
        };
        if periods.is_empty() {
            return Err(CudaDmError::InvalidInput("empty period sweep".into()));
        }
        let combos: Vec<DmParams> = periods
            .iter()
            .map(|&p| DmParams { period: Some(p) })
            .collect();
        let max_p = *periods.iter().max().unwrap();
        if len - first_valid < max_p {
            return Err(CudaDmError::InvalidInput("not enough valid data".into()));
        }
        Ok((combos, first_valid, len))
    }

    pub fn dm_batch_dev(
        &self,
        high: &[f32],
        low: &[f32],
        sweep: &DmBatchRange,
    ) -> Result<(DeviceDmPair, Vec<DmParams>), CudaDmError> {
        let (combos, first_valid, len) = Self::prepare_batch(high, low, sweep)?;
        let rows = combos.len();
        let req = (2 * len + rows + 2 * rows * len) * std::mem::size_of::<f32>();
        if !Self::device_mem_ok(req) {
            return Err(CudaDmError::InvalidInput(
                "insufficient device memory".into(),
            ));
        }

        let d_high = unsafe { DeviceBuffer::from_slice_async(&high[..len], &self.stream) }
            .map_err(|e| CudaDmError::Cuda(e.to_string()))?;
        let d_low = unsafe { DeviceBuffer::from_slice_async(&low[..len], &self.stream) }
            .map_err(|e| CudaDmError::Cuda(e.to_string()))?;
        let periods_host: Vec<i32> = combos.iter().map(|c| c.period.unwrap() as i32).collect();
        let d_periods = DeviceBuffer::from_slice(&periods_host)
            .map_err(|e| CudaDmError::Cuda(e.to_string()))?;
        let mut d_plus: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(rows * len, &self.stream) }
                .map_err(|e| CudaDmError::Cuda(e.to_string()))?;
        let mut d_minus: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(rows * len, &self.stream) }
                .map_err(|e| CudaDmError::Cuda(e.to_string()))?;

        self.launch_batch(
            &d_high,
            &d_low,
            &d_periods,
            len,
            rows,
            first_valid,
            &mut d_plus,
            &mut d_minus,
        )?;
        self.stream
            .synchronize()
            .map_err(|e| CudaDmError::Cuda(e.to_string()))?;

        let pair = DeviceDmPair {
            plus: DeviceArrayF32 {
                buf: d_plus,
                rows,
                cols: len,
            },
            minus: DeviceArrayF32 {
                buf: d_minus,
                rows,
                cols: len,
            },
        };
        Ok((pair, combos))
    }

    fn launch_batch(
        &self,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_plus: &mut DeviceBuffer<f32>,
        d_minus: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaDmError> {
        let func = self
            .module
            .get_function("dm_batch_f32")
            .map_err(|e| CudaDmError::Cuda(e.to_string()))?;
        let block_x = match self.policy.batch {
            BatchKernelPolicy::Auto => 256,
            BatchKernelPolicy::Plain { block_x } => block_x.max(32),
        };
        if cfg!(debug_assertions) || std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if !self.debug_batch_logged {
                eprintln!(
                    "[dm] batch kernel: block_x={} rows={} len={}",
                    block_x, n_combos, series_len
                );
                // mark as logged once
                unsafe {
                    (*(self as *const _ as *mut CudaDm)).debug_batch_logged = true;
                }
            }
        }
        unsafe {
            let grid_x = ((n_combos as u32) + block_x - 1) / block_x;
            let grid: GridSize = (grid_x.max(1), 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            let mut h = d_high.as_device_ptr().as_raw();
            let mut l = d_low.as_device_ptr().as_raw();
            let mut p = d_periods.as_device_ptr().as_raw();
            let mut n = series_len as i32;
            let mut r = n_combos as i32;
            let mut f = first_valid as i32;
            let mut po = d_plus.as_device_ptr().as_raw();
            let mut mo = d_minus.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 8] = [
                &mut h as *mut _ as *mut c_void,
                &mut l as *mut _ as *mut c_void,
                &mut p as *mut _ as *mut c_void,
                &mut n as *mut _ as *mut c_void,
                &mut r as *mut _ as *mut c_void,
                &mut f as *mut _ as *mut c_void,
                &mut po as *mut _ as *mut c_void,
                &mut mo as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, &mut args)
                .map_err(|e| CudaDmError::Cuda(e.to_string()))?;
        }
        Ok(())
    }

    pub fn dm_many_series_one_param_time_major_dev(
        &self,
        high_tm: &[f32],
        low_tm: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
    ) -> Result<DeviceDmPair, CudaDmError> {
        if cols == 0 || rows == 0 {
            return Err(CudaDmError::InvalidInput("empty matrix".into()));
        }
        if high_tm.len() != cols * rows || low_tm.len() != cols * rows {
            return Err(CudaDmError::InvalidInput("matrix shape mismatch".into()));
        }

        // Per-series first_valid detection
        let mut first_valids = vec![rows as i32; cols];
        for s in 0..cols {
            for t in 0..rows {
                let idx = t * cols + s;
                if !high_tm[idx].is_nan() && !low_tm[idx].is_nan() {
                    first_valids[s] = t as i32;
                    break;
                }
            }
        }
        for &fv in &first_valids {
            if (fv as usize) + period - 1 >= rows {
                return Err(CudaDmError::InvalidInput(
                    "not enough valid data for at least one series".into(),
                ));
            }
        }

        let req = (2 * cols * rows + cols + 2 * cols * rows) * std::mem::size_of::<f32>();
        if !Self::device_mem_ok(req) {
            return Err(CudaDmError::InvalidInput(
                "insufficient device memory".into(),
            ));
        }

        let d_high = unsafe { DeviceBuffer::from_slice_async(high_tm, &self.stream) }
            .map_err(|e| CudaDmError::Cuda(e.to_string()))?;
        let d_low = unsafe { DeviceBuffer::from_slice_async(low_tm, &self.stream) }
            .map_err(|e| CudaDmError::Cuda(e.to_string()))?;
        let d_first = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaDmError::Cuda(e.to_string()))?;
        let mut d_plus: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(cols * rows, &self.stream) }
                .map_err(|e| CudaDmError::Cuda(e.to_string()))?;
        let mut d_minus: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(cols * rows, &self.stream) }
                .map_err(|e| CudaDmError::Cuda(e.to_string()))?;

        self.launch_many_series(
            &d_high,
            &d_low,
            cols,
            rows,
            period,
            &d_first,
            &mut d_plus,
            &mut d_minus,
        )?;
        self.stream
            .synchronize()
            .map_err(|e| CudaDmError::Cuda(e.to_string()))?;
        Ok(DeviceDmPair {
            plus: DeviceArrayF32 {
                buf: d_plus,
                rows,
                cols,
            },
            minus: DeviceArrayF32 {
                buf: d_minus,
                rows,
                cols,
            },
        })
    }

    pub fn dm_many_series_one_param_time_major_into_host_f32(
        &self,
        high_tm: &[f32],
        low_tm: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
        plus_tm_out: &mut [f32],
        minus_tm_out: &mut [f32],
    ) -> Result<(), CudaDmError> {
        if plus_tm_out.len() != cols * rows || minus_tm_out.len() != cols * rows {
            return Err(CudaDmError::InvalidInput("out slice wrong length".into()));
        }
        let pair =
            self.dm_many_series_one_param_time_major_dev(high_tm, low_tm, cols, rows, period)?;
        let mut pinned_plus: LockedBuffer<f32> =
            unsafe { LockedBuffer::uninitialized(pair.plus.len()) }
                .map_err(|e| CudaDmError::Cuda(e.to_string()))?;
        let mut pinned_minus: LockedBuffer<f32> =
            unsafe { LockedBuffer::uninitialized(pair.minus.len()) }
                .map_err(|e| CudaDmError::Cuda(e.to_string()))?;
        unsafe {
            pair.plus
                .buf
                .async_copy_to(pinned_plus.as_mut_slice(), &self.stream)
                .map_err(|e| CudaDmError::Cuda(e.to_string()))?;
            pair.minus
                .buf
                .async_copy_to(pinned_minus.as_mut_slice(), &self.stream)
                .map_err(|e| CudaDmError::Cuda(e.to_string()))?;
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaDmError::Cuda(e.to_string()))?;
        plus_tm_out.copy_from_slice(pinned_plus.as_slice());
        minus_tm_out.copy_from_slice(pinned_minus.as_slice());
        Ok(())
    }

    fn launch_many_series(
        &self,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        period: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_plus: &mut DeviceBuffer<f32>,
        d_minus: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaDmError> {
        let func = self
            .module
            .get_function("dm_many_series_one_param_time_major_f32")
            .map_err(|e| CudaDmError::Cuda(e.to_string()))?;
        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => 256,
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(32),
        };
        if cfg!(debug_assertions) || std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if !self.debug_many_logged {
                eprintln!(
                    "[dm] many-series kernel: block_x={} cols={} rows={} period={}",
                    block_x, cols, rows, period
                );
                unsafe {
                    (*(self as *const _ as *mut CudaDm)).debug_many_logged = true;
                }
            }
        }
        unsafe {
            let grid_x = ((cols as u32) + block_x - 1) / block_x;
            let grid: GridSize = (grid_x.max(1), 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();

            let mut h = d_high.as_device_ptr().as_raw();
            let mut l = d_low.as_device_ptr().as_raw();
            let mut c = cols as i32;
            let mut r = rows as i32;
            let mut p = period as i32;
            let mut fv = d_first_valids.as_device_ptr().as_raw();
            let mut po = d_plus.as_device_ptr().as_raw();
            let mut mo = d_minus.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 8] = [
                &mut h as *mut _ as *mut c_void,
                &mut l as *mut _ as *mut c_void,
                &mut c as *mut _ as *mut c_void,
                &mut r as *mut _ as *mut c_void,
                &mut p as *mut _ as *mut c_void,
                &mut fv as *mut _ as *mut c_void,
                &mut po as *mut _ as *mut c_void,
                &mut mo as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, &mut args)
                .map_err(|e| CudaDmError::Cuda(e.to_string()))?;
        }
        Ok(())
    }
}

// ---------- Benches ----------
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const LEN_1M: usize = 1_000_000;
    const COLS_512: usize = 512;
    const ROWS_16K: usize = 16_384;

    fn synth_hl_from_close(close: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut high = close.to_vec();
        let mut low = close.to_vec();
        for i in 0..close.len() {
            let v = close[i];
            if v.is_nan() {
                continue;
            }
            let x = i as f32 * 0.0025;
            let off = (0.002 * x.sin()).abs() + 0.12;
            high[i] = v + off;
            low[i] = v - off;
        }
        (high, low)
    }

    struct BatchState {
        cuda: CudaDm,
        high: Vec<f32>,
        low: Vec<f32>,
        sweep: DmBatchRange,
    }
    impl CudaBenchState for BatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .dm_batch_dev(&self.high, &self.low, &self.sweep)
                .unwrap();
        }
    }

    struct ManySeriesState {
        cuda: CudaDm,
        high_tm: Vec<f32>,
        low_tm: Vec<f32>,
        cols: usize,
        rows: usize,
        period: usize,
    }
    impl CudaBenchState for ManySeriesState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .dm_many_series_one_param_time_major_dev(
                    &self.high_tm,
                    &self.low_tm,
                    self.cols,
                    self.rows,
                    self.period,
                )
                .unwrap();
        }
    }

    fn prep_batch() -> Box<dyn CudaBenchState> {
        let cuda = CudaDm::new(0).expect("cuda dm");
        let close = gen_series(LEN_1M);
        let (high, low) = synth_hl_from_close(&close);
        let sweep = DmBatchRange { period: (8, 96, 8) };
        Box::new(BatchState {
            cuda,
            high,
            low,
            sweep,
        })
    }

    fn prep_many() -> Box<dyn CudaBenchState> {
        let cuda = CudaDm::new(0).expect("cuda dm");
        let cols = COLS_512;
        let rows = ROWS_16K;
        let close_tm = {
            let mut v = vec![f32::NAN; cols * rows];
            for s in 0..cols {
                for t in s..rows {
                    let x = (t as f32) + (s as f32) * 0.2;
                    v[t * cols + s] = (x * 0.002).sin() + 0.0003 * x;
                }
            }
            v
        };
        let (high_tm, low_tm) = synth_hl_from_close(&close_tm);
        let period = 14usize;
        Box::new(ManySeriesState {
            cuda,
            high_tm,
            low_tm,
            cols,
            rows,
            period,
        })
    }

    fn bytes_batch() -> usize {
        (2 * LEN_1M + (LEN_1M / 8) + 2 * (LEN_1M * ((96 - 8) / 8 + 1))) * std::mem::size_of::<f32>()
            + 64 * 1024 * 1024
    }
    fn bytes_many() -> usize {
        (2 * COLS_512 * ROWS_16K + COLS_512 + 2 * COLS_512 * ROWS_16K) * std::mem::size_of::<f32>()
            + 64 * 1024 * 1024
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new("dm", "batch", "dm_cuda_batch", "1m", prep_batch)
                .with_mem_required(bytes_batch()),
            CudaBenchScenario::new(
                "dm",
                "many_series_one_param",
                "dm_cuda_many_series",
                "16k x 512",
                prep_many,
            )
            .with_mem_required(bytes_many()),
        ]
    }
}
