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
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

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

#[derive(thiserror::Error, Debug)]
pub enum CudaDmError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("out of memory: required={required} free={free} headroom={headroom}")]
    OutOfMemory { required: usize, free: usize, headroom: usize },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("device mismatch: buf={buf} current={current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
    NotImplemented,
}

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
    context: Arc<Context>,
    device_id: u32,
    policy: CudaDmPolicy,
    debug_batch_logged: AtomicBool,
    debug_many_logged: AtomicBool,
}

impl CudaDm {
    pub fn new(device_id: usize) -> Result<Self, CudaDmError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/dm_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => {
                if let Ok(m) = Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]) { m } else { Module::from_ptx(ptx, &[])? }
            }
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        Ok(Self {
            module,
            stream,
            context,
            device_id: device_id as u32,
            policy: CudaDmPolicy::default(),
            debug_batch_logged: AtomicBool::new(false),
            debug_many_logged: AtomicBool::new(false),
        })
    }

    pub fn set_policy(&mut self, policy: CudaDmPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaDmPolicy {
        &self.policy
    }
    pub fn synchronize(&self) -> Result<(), CudaDmError> { self.stream.synchronize().map_err(Into::into) }
    pub fn context_arc(&self) -> Arc<Context> { self.context.clone() }
    pub fn device_id(&self) -> u32 { self.device_id }

    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaDmError> {
        if let Ok((free, _)) = mem_get_info() {
            if required_bytes.saturating_add(headroom_bytes) > free {
                return Err(CudaDmError::OutOfMemory { required: required_bytes, free, headroom: headroom_bytes });
            }
        }
        Ok(())
    }

    #[inline]
    fn expand_periods(sweep: &DmBatchRange) -> Result<Vec<usize>, CudaDmError> {
        let (start, end, step) = sweep.period;
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        if start < end {
            let mut v = Vec::new();
            let st = step.max(1);
            let mut x = start;
            while x <= end {
                v.push(x);
                match x.checked_add(st) {
                    Some(next) => x = next,
                    None => break,
                }
            }
            if v.is_empty() {
                return Err(CudaDmError::InvalidInput("empty period sweep".into()));
            }
            return Ok(v);
        }
        // reversed bounds
        let mut v = Vec::new();
        let st = step.max(1) as isize;
        let mut x = start as isize;
        let end_i = end as isize;
        while x >= end_i {
            v.push(x as usize);
            x -= st;
        }
        if v.is_empty() {
            return Err(CudaDmError::InvalidInput("empty period sweep".into()));
        }
        Ok(v)
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
        let periods = Self::expand_periods(sweep)?;
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
        let rows_len = rows.checked_mul(len).ok_or_else(|| CudaDmError::InvalidInput("size overflow (rows*len)".into()))?;
        let mut req = std::mem::size_of::<f32>()
            .checked_mul(2 * len + 2 * rows_len)
            .ok_or_else(|| CudaDmError::InvalidInput("size overflow (bytes)".into()))?;
        req = req
            .checked_add(std::mem::size_of::<i32>() * rows)
            .ok_or_else(|| CudaDmError::InvalidInput("size overflow (periods)".into()))?;
        Self::will_fit(req, 64 * 1024 * 1024)?;

        let d_high = unsafe { DeviceBuffer::from_slice_async(&high[..len], &self.stream) }?;
        let d_low = unsafe { DeviceBuffer::from_slice_async(&low[..len], &self.stream) }?;
        let periods_host: Vec<i32> = combos.iter().map(|c| c.period.unwrap() as i32).collect();
        let d_periods = unsafe { DeviceBuffer::from_slice_async(&periods_host, &self.stream) }?;
        let mut d_plus: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(rows_len, &self.stream) }?;
        let mut d_minus: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(rows_len, &self.stream) }?;

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
        self.stream.synchronize()?;

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
            .map_err(|_| CudaDmError::MissingKernelSymbol { name: "dm_batch_f32" })?;
        let block_x = match self.policy.batch {
            BatchKernelPolicy::Auto => match func.suggested_launch_configuration(0, (1024, 1, 1).into()) {
                Ok((_min_grid, suggested_block)) => suggested_block.clamp(32, 1024),
                Err(_) => 256,
            },
            BatchKernelPolicy::Plain { block_x } => block_x.max(32),
        };
        if cfg!(debug_assertions) || std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if !self.debug_batch_logged.swap(true, Ordering::Relaxed) {
                eprintln!(
                    "[dm] batch kernel: block_x={} rows={} len={}",
                    block_x, n_combos, series_len
                );
            }
        }
        unsafe {
            let grid_x = ((n_combos as u32) + block_x - 1) / block_x;
            let grid: GridSize = (grid_x.max(1), 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            // Validate against device maximum threads per block
            let max_threads = Device::get_attribute(Device::get_device(self.device_id)?, DeviceAttribute::MaxThreadsPerBlock)? as u32;
            if block_x > max_threads {
                return Err(CudaDmError::LaunchConfigTooLarge { gx: grid_x.max(1), gy: 1, gz: 1, bx: block_x, by: 1, bz: 1 });
            }
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
            self.stream.launch(&func, grid, block, 0, &mut args)?;
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
        if cols == 0 || rows == 0 { return Err(CudaDmError::InvalidInput("empty matrix".into())); }
        let elems = cols.checked_mul(rows).ok_or_else(|| CudaDmError::InvalidInput("size overflow (cols*rows)".into()))?;
        if high_tm.len() != elems || low_tm.len() != elems { return Err(CudaDmError::InvalidInput("matrix shape mismatch".into())); }

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

        let req = std::mem::size_of::<f32>()
            .checked_mul(4 * elems)
            .ok_or_else(|| CudaDmError::InvalidInput("size overflow (bytes)".into()))?
            .checked_add(std::mem::size_of::<i32>() * cols)
            .ok_or_else(|| CudaDmError::InvalidInput("size overflow (first_valids)".into()))?;
        Self::will_fit(req, 64 * 1024 * 1024)?;

        let d_high = unsafe { DeviceBuffer::from_slice_async(high_tm, &self.stream) }?;
        let d_low = unsafe { DeviceBuffer::from_slice_async(low_tm, &self.stream) }?;
        let d_first = unsafe { DeviceBuffer::from_slice_async(&first_valids, &self.stream) }?;
        let mut d_plus: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(elems, &self.stream) }?;
        let mut d_minus: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(elems, &self.stream) }?;

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
        self.stream.synchronize()?;
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
        let elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaDmError::InvalidInput("size overflow (cols*rows)".into()))?;
        if plus_tm_out.len() != elems || minus_tm_out.len() != elems {
            return Err(CudaDmError::InvalidInput("out slice wrong length".into()));
        }
        let pair =
            self.dm_many_series_one_param_time_major_dev(high_tm, low_tm, cols, rows, period)?;
        // Single blocking D→H copy per output – no extra pinned stage.
        pair.plus.buf.copy_to(plus_tm_out)?;
        pair.minus.buf.copy_to(minus_tm_out)?;
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
            .map_err(|_| CudaDmError::MissingKernelSymbol { name: "dm_many_series_one_param_time_major_f32" })?;
        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => match func.suggested_launch_configuration(0, (1024, 1, 1).into()) {
                Ok((_min_grid, suggested_block)) => suggested_block.clamp(32, 1024),
                Err(_) => 256,
            },
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(32),
        };
        if cfg!(debug_assertions) || std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if !self.debug_many_logged.swap(true, Ordering::Relaxed) {
                eprintln!(
                    "[dm] many-series kernel: block_x={} cols={} rows={} period={}",
                    block_x, cols, rows, period
                );
            }
        }
        unsafe {
            let grid_x = ((cols as u32) + block_x - 1) / block_x;
            let grid: GridSize = (grid_x.max(1), 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            let max_threads = Device::get_attribute(Device::get_device(self.device_id)?, DeviceAttribute::MaxThreadsPerBlock)? as u32;
            if block_x > max_threads {
                return Err(CudaDmError::LaunchConfigTooLarge { gx: grid_x.max(1), gy: 1, gz: 1, bx: block_x, by: 1, bz: 1 });
            }

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
            self.stream.launch(&func, grid, block, 0, &mut args)?;
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
        d_high: DeviceBuffer<f32>,
        d_low: DeviceBuffer<f32>,
        d_periods: DeviceBuffer<i32>,
        len: usize,
        first_valid: usize,
        n_combos: usize,
        d_plus: DeviceBuffer<f32>,
        d_minus: DeviceBuffer<f32>,
    }
    impl CudaBenchState for BatchState {
        fn launch(&mut self) {
            self.cuda
                .launch_batch(
                    &self.d_high,
                    &self.d_low,
                    &self.d_periods,
                    self.len,
                    self.n_combos,
                    self.first_valid,
                    &mut self.d_plus,
                    &mut self.d_minus,
                )
                .expect("dm batch kernel");
            self.cuda.stream.synchronize().expect("dm sync");
        }
    }

    struct ManySeriesState {
        cuda: CudaDm,
        d_high_tm: DeviceBuffer<f32>,
        d_low_tm: DeviceBuffer<f32>,
        d_first_valids: DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        period: usize,
        block_x: u32,
        grid_x: u32,
        d_plus_tm: DeviceBuffer<f32>,
        d_minus_tm: DeviceBuffer<f32>,
    }
    impl CudaBenchState for ManySeriesState {
        fn launch(&mut self) {
            let func = self
                .cuda
                .module
                .get_function("dm_many_series_one_param_time_major_f32")
                .expect("dm_many_series_one_param_time_major_f32");
            let grid: GridSize = (self.grid_x.max(1), 1, 1).into();
            let block: BlockSize = (self.block_x, 1, 1).into();
            unsafe {
                let mut h = self.d_high_tm.as_device_ptr().as_raw();
                let mut l = self.d_low_tm.as_device_ptr().as_raw();
                let mut c = self.cols as i32;
                let mut r = self.rows as i32;
                let mut p = self.period as i32;
                let mut fv = self.d_first_valids.as_device_ptr().as_raw();
                let mut po = self.d_plus_tm.as_device_ptr().as_raw();
                let mut mo = self.d_minus_tm.as_device_ptr().as_raw();
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
                self.cuda
                    .stream
                    .launch(&func, grid, block, 0, &mut args)
                    .expect("dm many-series launch");
            }
            self.cuda.stream.synchronize().expect("dm sync");
        }
    }

    fn prep_batch() -> Box<dyn CudaBenchState> {
        let cuda = CudaDm::new(0).expect("cuda dm");
        let close = gen_series(LEN_1M);
        let (high, low) = synth_hl_from_close(&close);
        let first_valid = (0..LEN_1M)
            .find(|&i| !high[i].is_nan() && !low[i].is_nan())
            .unwrap_or(LEN_1M);
        let sweep = DmBatchRange { period: (8, 96, 8) };
        let periods_host: Vec<i32> = (sweep.period.0..=sweep.period.1)
            .step_by(sweep.period.2.max(1))
            .map(|p| p as i32)
            .collect();
        let n_combos = periods_host.len();

        let d_high = unsafe { DeviceBuffer::from_slice_async(&high, &cuda.stream) }.expect("d_high");
        let d_low = unsafe { DeviceBuffer::from_slice_async(&low, &cuda.stream) }.expect("d_low");
        let d_periods =
            unsafe { DeviceBuffer::from_slice_async(&periods_host, &cuda.stream) }.expect("d_periods");
        let out_elems = n_combos * LEN_1M;
        let d_plus: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(out_elems, &cuda.stream) }.expect("d_plus");
        let d_minus: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(out_elems, &cuda.stream) }.expect("d_minus");
        cuda.stream.synchronize().expect("sync after prep");
        Box::new(BatchState {
            cuda,
            d_high,
            d_low,
            d_periods,
            len: LEN_1M,
            first_valid,
            n_combos,
            d_plus,
            d_minus,
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
        let mut first_valids: Vec<i32> = vec![-1; cols];
        for s in 0..cols {
            for t in 0..rows {
                let idx = t * cols + s;
                if high_tm[idx].is_finite() && low_tm[idx].is_finite() {
                    first_valids[s] = t as i32;
                    break;
                }
            }
        }
        let d_high_tm = DeviceBuffer::from_slice(&high_tm).expect("d_high_tm");
        let d_low_tm = DeviceBuffer::from_slice(&low_tm).expect("d_low_tm");
        let d_first_valids = DeviceBuffer::from_slice(&first_valids).expect("d_first_valids");
        let elems = cols * rows;
        let d_plus_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems) }.expect("d_plus_tm");
        let d_minus_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems) }.expect("d_minus_tm");

        // Match wrapper block_x selection once in prep.
        let mut func = cuda
            .module
            .get_function("dm_many_series_one_param_time_major_f32")
            .expect("dm_many_series_one_param_time_major_f32");
        let block_x = match cuda.policy.many_series {
            ManySeriesKernelPolicy::Auto => match func
                .suggested_launch_configuration(0, (1024, 1, 1).into())
            {
                Ok((_min_grid, suggested_block)) => suggested_block.clamp(32, 1024),
                Err(_) => 256,
            },
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(32),
        };
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        cuda.stream.synchronize().expect("sync after prep");
        Box::new(ManySeriesState {
            cuda,
            d_high_tm,
            d_low_tm,
            d_first_valids,
            cols,
            rows,
            period,
            block_x,
            grid_x,
            d_plus_tm,
            d_minus_tm,
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
