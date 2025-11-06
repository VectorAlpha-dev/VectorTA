//! CUDA wrapper for RSX (Relative Strength Xtra)
//!
//! Parity with scalar RSX:
//! - Warmup/NaN semantics identical (prefix NaNs up to first_valid + period - 1).
//! - FP32 compute; single-lane sequential scan per parameter (recurrence-bound).
//! - NON_BLOCKING stream; VRAM headroom checks; chunked grid to <= 65_535 blocks.

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32; // shared device array handle
use crate::indicators::rsx::{RsxBatchRange, RsxParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::error::Error;
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaRsxError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaRsxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaRsxError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaRsxError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}
impl Error for CudaRsxError {}

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaRsxPolicy {
    pub batch_block_x: Option<u32>,
    pub many_block_x: Option<u32>,
}

pub struct CudaRsx {
    module: Module,
    stream: Stream,
    _context: Context,
    policy: CudaRsxPolicy,
}

impl CudaRsx {
    pub fn new(device_id: usize) -> Result<Self, CudaRsxError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaRsxError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaRsxError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaRsxError::Cuda(e.to_string()))?;

        let ptx = include_str!(concat!(env!("OUT_DIR"), "/rsx_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O4),
        ];
        let module = Module::from_ptx(ptx, jit_opts)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))
            .map_err(|e| CudaRsxError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaRsxError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            policy: CudaRsxPolicy::default(),
        })
    }

    #[inline]
    pub fn set_policy(&mut self, p: CudaRsxPolicy) { self.policy = p; }

    // ---------- Batch (one series × many params) ----------
    pub fn rsx_batch_dev(
        &self,
        prices_f32: &[f32],
        sweep: &RsxBatchRange,
    ) -> Result<DeviceArrayF32, CudaRsxError> {
        let (combos, first_valid, len) = Self::prepare_batch_inputs(prices_f32, sweep)?;
        let n_combos = combos.len();
        let periods_i32: Vec<i32> = combos
            .iter()
            .map(|p| p.period.unwrap_or(0) as i32)
            .collect();

        // VRAM estimate (best-effort)
        if let Ok((free, _)) = mem_get_info() {
            let in_bytes = prices_f32.len() * std::mem::size_of::<f32>();
            let params_bytes = periods_i32.len() * std::mem::size_of::<i32>();
            let out_bytes = n_combos
                .checked_mul(len)
                .ok_or_else(|| CudaRsxError::InvalidInput("rows*cols overflow".into()))?
                * std::mem::size_of::<f32>();
            let headroom = 64usize * 1024 * 1024;
            let need = in_bytes + params_bytes + out_bytes + headroom;
            if need > free {
                return Err(CudaRsxError::InvalidInput(
                    "estimated device memory exceeds free VRAM".into(),
                ));
            }
        }

        let d_prices = unsafe { to_device_buffer_async(&self.stream, prices_f32)? };
        let d_periods = unsafe { to_device_buffer_async(&self.stream, &periods_i32)? };
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(len * n_combos)
                .map_err(|e| CudaRsxError::Cuda(e.to_string()))?
        };

        self.launch_batch(
            &d_prices,
            &d_periods,
            len,
            first_valid,
            n_combos,
            &mut d_out,
        )?;
        Ok(DeviceArrayF32 { buf: d_out, rows: n_combos, cols: len })
    }

    fn launch_batch(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        len: usize,
        first_valid: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaRsxError> {
        if n_combos == 0 {
            return Ok(());
        }
        if n_combos == 0 {
            return Ok(());
        }
        let func = self
            .module
            .get_function("rsx_batch_f32")
            .map_err(|e| CudaRsxError::Cuda(e.to_string()))?;

        // One thread per combo per the new kernel mapping.
        let block_x = self.policy.batch_block_x.unwrap_or(256);
        let grid_x = ((n_combos as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut series_len_i = len as i32;
            let mut first_i = first_valid as i32;
            let mut combos_i = n_combos as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut first_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaRsxError::Cuda(e.to_string()))?;
        }
        self.stream.synchronize().map_err(|e| CudaRsxError::Cuda(e.to_string()))
    }

    // ---------- Many-series × one-param (time-major) ----------
    pub fn rsx_many_series_one_param_time_major_dev(
        &self,
        prices_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
    ) -> Result<DeviceArrayF32, CudaRsxError> {
        if cols == 0 || rows == 0 {
            return Err(CudaRsxError::InvalidInput("invalid dims".into()));
        }
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaRsxError::InvalidInput("rows*cols overflow".into()))?;
        if prices_tm_f32.len() != expected {
            return Err(CudaRsxError::InvalidInput(
                "time-major length mismatch".into(),
            ));
            return Err(CudaRsxError::InvalidInput(
                "time-major length mismatch".into(),
            ));
        }
        if period == 0 {
            return Err(CudaRsxError::InvalidInput("period must be > 0".into()));
        }

        // Per-series first_valid
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = -1i32;
            for t in 0..rows {
                let v = prices_tm_f32[t * cols + s];
                if !v.is_nan() {
                    fv = t as i32;
                    break;
                }
                if !v.is_nan() {
                    fv = t as i32;
                    break;
                }
            }
            if fv < 0 {
                return Err(CudaRsxError::InvalidInput(format!("series {} all NaN", s)));
            }
            first_valids[s] = fv;
        }

        // VRAM estimate
        if let Ok((free, _)) = mem_get_info() {
            let n = expected;
            let bytes = (2 * n) * std::mem::size_of::<f32>()
                + cols * std::mem::size_of::<i32>()
                + 64 * 1024 * 1024;
            if bytes > free {
                return Err(CudaRsxError::InvalidInput(
                    "estimated device memory exceeds free VRAM".into(),
                ));
            }
        }

        let d_prices = unsafe { to_device_buffer_async(&self.stream, prices_tm_f32)? };
        let d_first = unsafe { to_device_buffer_async(&self.stream, &first_valids)? };
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(expected).map_err(|e| CudaRsxError::Cuda(e.to_string()))?
        };

        self.launch_many_series(&d_prices, &d_first, cols, rows, period, &mut d_out)?;
        Ok(DeviceArrayF32 { buf: d_out, rows, cols })
    }

    fn launch_many_series(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        period: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaRsxError> {
        let func = self
            .module
            .get_function("rsx_many_series_one_param_f32")
            .map_err(|e| CudaRsxError::Cuda(e.to_string()))?;
        let block_x = self.policy.many_block_x.unwrap_or(128);
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut period_i = period as i32;
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaRsxError::Cuda(e.to_string()))?;
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaRsxError::Cuda(e.to_string()))?;
        Ok(())
    }

    // ---------- Helpers ----------
    fn prepare_batch_inputs(
        prices: &[f32],
        sweep: &RsxBatchRange,
    ) -> Result<(Vec<RsxParams>, usize, usize), CudaRsxError> {
        let len = prices.len();
        if len == 0 {
            return Err(CudaRsxError::InvalidInput("empty prices".into()));
        }
        // expand grid
        let (start, end, step) = sweep.period;
        let mut combos = Vec::new();
        if step == 0 || start == end {
            combos.push(RsxParams { period: Some(start) });
        } else {
            let mut v = start;
            while v <= end {
                combos.push(RsxParams { period: Some(v) });
                v = v.saturating_add(step);
            }
        }
        if combos.is_empty() {
            return Err(CudaRsxError::InvalidInput("no period combos".into()));
        }

        let first_valid = (0..len)
            .find(|&i| !prices[i].is_nan())
            .ok_or_else(|| CudaRsxError::InvalidInput("all values NaN".into()))?;
        let max_p = combos
            .iter()
            .map(|c| c.period.unwrap_or(0))
            .max()
            .unwrap_or(0);
        let max_p = combos
            .iter()
            .map(|c| c.period.unwrap_or(0))
            .max()
            .unwrap_or(0);
        if max_p == 0 {
            return Err(CudaRsxError::InvalidInput("period must be > 0".into()));
        }
        let valid = len - first_valid;
        if valid < max_p {
            return Err(CudaRsxError::InvalidInput(format!(
                "not enough valid data: need >= {}, have {}",
                max_p, valid
            )));
        }
        Ok((combos, first_valid, len))
    }
}

// ---------- Benches ----------
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const MANY_COLS: usize = 1024;
    const MANY_ROWS: usize = 8192;
    const PARAM_SWEEP: usize = 200; // ~200 distinct periods

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }
    fn bytes_many_series() -> usize {
        let n = MANY_COLS * MANY_ROWS;
        let in_bytes = n * std::mem::size_of::<f32>();
        let out_bytes = n * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct RsxBatchState {
        cuda: CudaRsx,
        prices: Vec<f32>,
        sweep: RsxBatchRange,
    }
    impl CudaBenchState for RsxBatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .rsx_batch_dev(&self.prices, &self.sweep)
                .expect("rsx batch");
        }
    }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaRsx::new(0).expect("cuda rsx");
        let mut prices = gen_series(ONE_SERIES_LEN);
        
        for i in 0..8 {
            prices[i] = f32::NAN;
        }
        for i in 8..ONE_SERIES_LEN {
            let x = i as f32 * 0.0019;
            prices[i] += 0.0005 * x.sin();
        }
        let sweep = RsxBatchRange {
            period: (2, 1 + PARAM_SWEEP, 1),
        };
        Box::new(RsxBatchState { cuda, prices, sweep })
    }

    struct RsxManyState {
        cuda: CudaRsx,
        prices_tm: Vec<f32>,
    }
    impl CudaBenchState for RsxManyState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .rsx_many_series_one_param_time_major_dev(&self.prices_tm, MANY_COLS, MANY_ROWS, 14)
                .expect("rsx many");
        }
    }
    fn prep_many_series() -> Box<dyn CudaBenchState> {
        let cuda = CudaRsx::new(0).expect("cuda rsx");
        let n = MANY_COLS * MANY_ROWS;
        let mut base = gen_series(n);
        let mut prices = vec![f32::NAN; n];
        for s in 0..MANY_COLS {
            for t in s..MANY_ROWS {
                let idx = t * MANY_COLS + s;
                let x = (t as f32) * 0.002 + (s as f32) * 0.01;
                prices[idx] = base[idx] + 0.05 * x.sin();
            }
        }
        Box::new(RsxManyState { cuda, prices_tm: prices })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "rsx",
                "one_series_many_params",
                "rsx_cuda_batch_dev",
                "1m_x_200",
                prep_one_series_many_params,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "rsx",
                "many_series_one_param",
                "rsx_cuda_many_series_one_param_dev",
                "1024x8192",
                prep_many_series,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_many_series()),
        ]
    }
}

// ---------- Internal helpers ----------
#[inline]
unsafe fn to_device_buffer_async<T: cust::memory::DeviceCopy>(
    stream: &Stream,
    host: &[T],
) -> Result<DeviceBuffer<T>, CudaRsxError> {
    const PIN_BYTES: usize = 4 * 1024 * 1024; // 4 MiB threshold
    let bytes = host.len() * std::mem::size_of::<T>();
    if bytes >= PIN_BYTES {
        let pinned = LockedBuffer::from_slice(host)
            .map_err(|e| CudaRsxError::Cuda(e.to_string()))?;
        DeviceBuffer::from_slice_async(pinned.as_slice(), stream)
            .map_err(|e| CudaRsxError::Cuda(e.to_string()))
    } else {
        DeviceBuffer::from_slice(host).map_err(|e| CudaRsxError::Cuda(e.to_string()))
    }
}

