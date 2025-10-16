#![cfg(feature = "cuda")]

//! CUDA wrapper for TTM Trend (close > SMA(source, period)).
//!
//! Parity with ALMA wrapper policy:
//! - PTX load via include_str!(concat!(env!("OUT_DIR"), "/ttm_trend_kernel.ptx"))
//!   with DetermineTargetFromContext + OptLevel O2, and simpler fallbacks.
//! - NON_BLOCKING stream.
//! - VRAM check with ~64MB headroom and grid.y chunking kept under 65_535.
//! - Public device entry points:
//!   - `ttm_trend_batch_dev(&[f32], &[f32], &TtmTrendBatchRange) -> DeviceArrayF32`
//!   - `ttm_trend_many_series_one_param_time_major_dev(&[f32], &[f32], cols, rows, period)`

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::ttm_trend::TtmTrendBatchRange;
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::launch;
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::error::Error;
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaTtmTrendError {
    Cuda(String),
    InvalidInput(String),
}
impl fmt::Display for CudaTtmTrendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cuda(e) => write!(f, "CUDA error: {}", e),
            Self::InvalidInput(s) => write!(f, "Invalid input: {}", s),
        }
    }
}
impl Error for CudaTtmTrendError {}

#[derive(Clone, Debug)]
struct Combo { period: i32, warm: i32 }

pub struct CudaTtmTrend {
    pub(crate) module: Module,
    pub(crate) stream: Stream,
    _ctx: Context,
}

impl CudaTtmTrend {
    pub fn new(device_id: usize) -> Result<Self, CudaTtmTrendError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaTtmTrendError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaTtmTrendError::Cuda(e.to_string()))?;
        let ctx = Context::new(device).map_err(|e| CudaTtmTrendError::Cuda(e.to_string()))?;
        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/ttm_trend_kernel.ptx"));
        // JIT with context-target + O2, then fallback
        let jit = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, jit)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))
            .map_err(|e| CudaTtmTrendError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaTtmTrendError::Cuda(e.to_string()))?;
        Ok(Self { module, stream, _ctx: ctx })
    }

    #[inline]
    fn will_fit(required_bytes: usize, headroom: usize) -> bool {
        if let Ok((free, _)) = mem_get_info() { required_bytes.saturating_add(headroom) <= free } else { true }
    }

    fn expand_grid(range: &TtmTrendBatchRange) -> Vec<i32> {
        let (start, end, step) = range.period;
        if step == 0 || start == end { vec![start as i32] } else { (start..=end).step_by(step).map(|p| p as i32).collect() }
    }

    fn prepare_batch_inputs(
        source_f32: &[f32],
        close_f32: &[f32],
        sweep: &TtmTrendBatchRange,
    ) -> Result<(Vec<Combo>, usize, usize), CudaTtmTrendError> {
        if source_f32.len() != close_f32.len() {
            return Err(CudaTtmTrendError::InvalidInput("source/close length mismatch".into()));
        }
        let len = source_f32.len();
        if len == 0 { return Err(CudaTtmTrendError::InvalidInput("empty inputs".into())); }
        let first = source_f32
            .iter()
            .zip(close_f32)
            .position(|(&s, &c)| !s.is_nan() && !c.is_nan())
            .ok_or_else(|| CudaTtmTrendError::InvalidInput("all values are NaN".into()))?;
        let periods = Self::expand_grid(sweep);
        if periods.is_empty() { return Err(CudaTtmTrendError::InvalidInput("no parameter combinations".into())); }
        let mut combos = Vec::with_capacity(periods.len());
        for &p in &periods {
            let pu = p as usize;
            if pu == 0 || pu > len { return Err(CudaTtmTrendError::InvalidInput(format!("invalid period {} for len {}", pu, len))); }
            if len - first < pu {
                return Err(CudaTtmTrendError::InvalidInput(format!(
                    "not enough valid data: needed >= {}, valid = {}",
                    pu,
                    len - first
                )));
            }
            let warm = (first + pu - 1) as i32;
            combos.push(Combo { period: p, warm });
        }
        Ok((combos, first, len))
    }

    fn build_prefix_source_f64(source_f32: &[f32], first_valid: usize) -> Vec<f64> {
        let n = source_f32.len();
        let mut pref = vec![0.0f64; n];
        if first_valid < n {
            pref[first_valid] = source_f32[first_valid] as f64;
            for i in (first_valid + 1)..n { pref[i] = pref[i - 1] + (source_f32[i] as f64); }
        }
        pref
    }

    #[inline]
    fn chunk_rows(n_rows: usize) -> usize {
        // Keep grid.y chunks under hardware limit; coarse VRAM heuristic
        let max_grid_y = 65_000usize;
        max_grid_y.min(n_rows).max(1)
    }

    pub fn ttm_trend_batch_dev(
        &self,
        source_f32: &[f32],
        close_f32: &[f32],
        sweep: &TtmTrendBatchRange,
    ) -> Result<DeviceArrayF32, CudaTtmTrendError> {
        let (combos, first, len) = Self::prepare_batch_inputs(source_f32, close_f32, sweep)?;
        let n_combos = combos.len();

        // Host precompute: FP64 prefix of source
        let prefix = Self::build_prefix_source_f64(source_f32, first);

        // Device buffers
        let d_prefix = DeviceBuffer::from_slice(&prefix).map_err(|e| CudaTtmTrendError::Cuda(e.to_string()))?;
        let d_close  = DeviceBuffer::from_slice(close_f32).map_err(|e| CudaTtmTrendError::Cuda(e.to_string()))?;
        let periods: Vec<i32> = combos.iter().map(|c| c.period).collect();
        let warms:   Vec<i32> = combos.iter().map(|c| c.warm).collect();
        let d_periods = DeviceBuffer::from_slice(&periods).map_err(|e| CudaTtmTrendError::Cuda(e.to_string()))?;
        let d_warms   = DeviceBuffer::from_slice(&warms).map_err(|e| CudaTtmTrendError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(n_combos * len) }
            .map_err(|e| CudaTtmTrendError::Cuda(e.to_string()))?;

        // VRAM estimate (best-effort)
        let est_bytes = len * (std::mem::size_of::<f64>() + std::mem::size_of::<f32>())
            + n_combos * (std::mem::size_of::<i32>() * 2)
            + n_combos * len * std::mem::size_of::<f32>();
        if !Self::will_fit(est_bytes, 64usize << 20) {
            return Err(CudaTtmTrendError::InvalidInput("insufficient free VRAM".into()));
        }

        // Launch kernel by chunks of rows (grid.x = rows chunk; blockDim.x = 256)
        let func = self
            .module
            .get_function("ttm_trend_batch_prefix_f64")
            .map_err(|e| CudaTtmTrendError::Cuda(e.to_string()))?;

        let block_x: u32 = 256;
        let rows_per_launch = Self::chunk_rows(n_combos);
        let mut launched = 0usize;
        while launched < n_combos {
            let cur = rows_per_launch.min(n_combos - launched);
            // We map each row to a block in X, so grid = (cur, 1, 1)
            let grid: GridSize = (cur as u32, 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            unsafe {
                let mut pref_ptr = d_prefix.as_device_ptr().as_raw();
                let mut close_ptr = d_close.as_device_ptr().as_raw();
                let mut per_ptr = d_periods.as_device_ptr().as_raw().wrapping_add(launched as u64 * std::mem::size_of::<i32>() as u64);
                let mut warm_ptr = d_warms.as_device_ptr().as_raw().wrapping_add(launched as u64 * std::mem::size_of::<i32>() as u64);
                let mut len_i = len as i32;
                let mut first_i = first as i32;
                let mut ncomb_i = cur as i32;
                let mut out_ptr = d_out
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((launched * len) as u64 * std::mem::size_of::<f32>() as u64);
                let args: &mut [*mut c_void] = &mut [
                    &mut pref_ptr as *mut _ as *mut c_void,
                    &mut close_ptr as *mut _ as *mut c_void,
                    &mut per_ptr as *mut _ as *mut c_void,
                    &mut warm_ptr as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut first_i as *mut _ as *mut c_void,
                    &mut ncomb_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream
                    .launch(&func, grid, block, 0, args)
                    .map_err(|e| CudaTtmTrendError::Cuda(e.to_string()))?;
            }
            launched += cur;
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaTtmTrendError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 { buf: d_out, rows: n_combos, cols: len })
    }

    pub fn ttm_trend_many_series_one_param_time_major_dev(
        &self,
        source_tm_f32: &[f32],
        close_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
    ) -> Result<DeviceArrayF32, CudaTtmTrendError> {
        if cols == 0 || rows == 0 {
            return Err(CudaTtmTrendError::InvalidInput("cols/rows must be > 0".into()));
        }
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaTtmTrendError::InvalidInput("rows*cols overflow".into()))?;
        if source_tm_f32.len() != expected || close_tm_f32.len() != expected {
            return Err(CudaTtmTrendError::InvalidInput(
                "time-major input length mismatch".into(),
            ));
        }
        if period == 0 || period > rows {
            return Err(CudaTtmTrendError::InvalidInput("invalid period".into()));
        }

        // First-valid per series based on both source and close
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = None;
            for t in 0..rows {
                let si = source_tm_f32[t * cols + s];
                let ci = close_tm_f32[t * cols + s];
                if !si.is_nan() && !ci.is_nan() { fv = Some(t as i32); break; }
            }
            let fv = fv.ok_or_else(|| CudaTtmTrendError::InvalidInput(format!("series {} all NaN", s)))?;
            if rows - (fv as usize) < period {
                return Err(CudaTtmTrendError::InvalidInput(format!(
                    "series {} not enough valid data: needed >= {}, valid = {}",
                    s,
                    period,
                    rows - fv as usize
                )));
            }
            first_valids[s] = fv;
        }

        // Device buffers
        let d_src  = DeviceBuffer::from_slice(source_tm_f32).map_err(|e| CudaTtmTrendError::Cuda(e.to_string()))?;
        let d_close = DeviceBuffer::from_slice(close_tm_f32).map_err(|e| CudaTtmTrendError::Cuda(e.to_string()))?;
        let d_fv   = DeviceBuffer::from_slice(&first_valids).map_err(|e| CudaTtmTrendError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(expected) }
            .map_err(|e| CudaTtmTrendError::Cuda(e.to_string()))?;

        let func = self
            .module
            .get_function("ttm_trend_many_series_one_param_time_major_f32")
            .map_err(|e| CudaTtmTrendError::Cuda(e.to_string()))?;
        let block_x: u32 = 256;
        let grid_x: u32 = ((cols as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe {
            let mut src_ptr = d_src.as_device_ptr().as_raw();
            let mut close_ptr = d_close.as_device_ptr().as_raw();
            let mut fv_ptr  = d_fv.as_device_ptr().as_raw();
            let mut cols_i  = cols as i32;
            let mut rows_i  = rows as i32;
            let mut p_i     = period as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut src_ptr as *mut _ as *mut c_void,
                &mut close_ptr as *mut _ as *mut c_void,
                &mut fv_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut p_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaTtmTrendError::Cuda(e.to_string()))?;
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaTtmTrendError::Cuda(e.to_string()))?;
        Ok(DeviceArrayF32 { buf: d_out, rows, cols })
    }

    pub fn synchronize(&self) -> Result<(), CudaTtmTrendError> {
        self.stream
            .synchronize()
            .map_err(|e| CudaTtmTrendError::Cuda(e.to_string()))
    }
}

// ----------------- Benches -----------------
pub mod benches {
    use super::*;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    fn gen_series(n: usize) -> Vec<f32> {
        let mut v = vec![f32::NAN; n];
        for i in 8..n { let x = i as f32; v[i] = (x * 0.00123).sin() + 0.00031 * x; }
        v
    }

    struct TtmBatchState {
        cuda: CudaTtmTrend,
        src: Vec<f32>,
        close: Vec<f32>,
        sweep: TtmTrendBatchRange,
    }
    impl CudaBenchState for TtmBatchState { fn launch(&mut self) { let _ = self.cuda.ttm_trend_batch_dev(&self.src, &self.close, &self.sweep).unwrap(); } }

    fn prep_batch() -> Box<dyn CudaBenchState> {
        let cuda = CudaTtmTrend::new(0).expect("cuda ttm");
        let len = 1_000_000usize; // 1M points
        let src = gen_series(len);
        // Close is a noisy variant of source
        let mut close = vec![f32::NAN; len];
        for i in 8..len { let x = i as f32; close[i] = src[i] + 0.05 * (x * 0.00071).cos(); }
        let sweep = TtmTrendBatchRange { period: (5, 254, 1) };
        Box::new(TtmBatchState { cuda, src, close, sweep })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "ttm_trend",
            "one_series_many_params",
            "ttm_trend_cuda_batch_dev",
            "1m_x_250",
            prep_batch,
        )]
    }
}

