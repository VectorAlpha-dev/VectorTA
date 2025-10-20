#![cfg(feature = "cuda")]

//! CUDA wrapper for Median Absolute Deviation (MEDIUM_AD).
//!
//! Parity with ALMA/CUDA wrappers:
//! - PTX load via include_str!(concat!(env!("OUT_DIR"), "/medium_ad_kernel.ptx")) with JIT
//!   options DetermineTargetFromContext + OptLevel O2, with simpler fallbacks if needed.
//! - NON_BLOCKING stream
//! - VRAM guard and grid.y chunking (<= 65_535)
//! - Public device entry points for batch (one-series × many-params) and
//!   many-series (time-major) × one-param.
//!
//! Semantics:
//! - Identical to scalar path in src/indicators/medium_ad.rs (warmup NaNs,
//!   window NaNs → NaN, period==1 → 0.0 on finite input).

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::medium_ad::MediumAdBatchRange;
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::launch;
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::error::Error;
use std::ffi::c_void;
use std::fmt;

const MEDIUM_AD_MAX_PERIOD: usize = 512; // must match kernels/cuda/medium_ad_kernel.cu

#[derive(Debug)]
pub enum CudaMediumAdError {
    Cuda(String),
    InvalidInput(String),
}
impl fmt::Display for CudaMediumAdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cuda(e) => write!(f, "CUDA error: {}", e),
            Self::InvalidInput(s) => write!(f, "Invalid input: {}", s),
        }
    }
}
impl Error for CudaMediumAdError {}

#[derive(Clone, Debug)]
struct MediumAdCombo {
    period: i32,
}

pub struct CudaMediumAd {
    module: Module,
    stream: Stream,
    _ctx: Context,
}

impl CudaMediumAd {
    pub fn new(device_id: usize) -> Result<Self, CudaMediumAdError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaMediumAdError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaMediumAdError::Cuda(e.to_string()))?;
        let ctx = Context::new(device).map_err(|e| CudaMediumAdError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/medium_ad_kernel.ptx"));
        let jit = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, jit)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))
            .map_err(|e| CudaMediumAdError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaMediumAdError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _ctx: ctx,
        })
    }

    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> bool {
        if let Ok((free, _)) = mem_get_info() {
            required_bytes.saturating_add(headroom_bytes) <= free
        } else {
            true
        }
    }

    fn expand_grid(range: &MediumAdBatchRange) -> Vec<MediumAdCombo> {
        let (start, end, step) = range.period;
        let periods: Vec<usize> = if step == 0 || start == end {
            vec![start]
        } else {
            (start..=end).step_by(step).collect()
        };
        periods
            .into_iter()
            .map(|p| MediumAdCombo { period: p as i32 })
            .collect()
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &MediumAdBatchRange,
    ) -> Result<(Vec<MediumAdCombo>, usize), CudaMediumAdError> {
        if data_f32.is_empty() {
            return Err(CudaMediumAdError::InvalidInput("empty data".into()));
        }
        let len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaMediumAdError::InvalidInput("all NaN".into()))?;
        let combos = Self::expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaMediumAdError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        for c in &combos {
            let p = c.period as usize;
            if p == 0 || p > len {
                return Err(CudaMediumAdError::InvalidInput(format!(
                    "invalid period {} for len {}",
                    p, len
                )));
            }
            if p > MEDIUM_AD_MAX_PERIOD {
                return Err(CudaMediumAdError::InvalidInput(format!(
                    "period {} exceeds MEDIUM_AD_MAX_PERIOD {} for CUDA path",
                    p, MEDIUM_AD_MAX_PERIOD
                )));
            }
            if len - first_valid < p {
                return Err(CudaMediumAdError::InvalidInput(format!(
                    "not enough valid data: needed >= {}, valid = {}",
                    p,
                    len - first_valid
                )));
            }
        }
        Ok((combos, first_valid))
    }

    fn launch_batch_kernel(
        &self,
        d_data: &DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        d_periods: &DeviceBuffer<i32>,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaMediumAdError> {
        let func = self
            .module
            .get_function("medium_ad_batch_f32")
            .map_err(|e| CudaMediumAdError::Cuda(e.to_string()))?;

        let block_x: u32 = 256;
        let grid_x = ((len as u32) + block_x - 1) / block_x;

        // Chunk grid.y to avoid launch limits and large VRAM spikes
        let max_y = 65_000usize;
        let chunk_rows = n_combos.min(max_y).max(1);
        let mut launched = 0usize;
        while launched < n_combos {
            let cur = (n_combos - launched).min(chunk_rows);
            let grid: GridSize = (grid_x.max(1), cur as u32, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();

            unsafe {
                let mut data_ptr = d_data.as_device_ptr().as_raw();
                let mut len_i = len as i32;
                let mut fv_i = first_valid as i32;
                let mut periods_ptr = d_periods
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((launched * std::mem::size_of::<i32>()) as u64);
                let mut ncomb_i = cur as i32;
                let mut out_ptr = d_out
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((launched * len * std::mem::size_of::<f32>()) as u64);

                let args: &mut [*mut c_void] = &mut [
                    &mut data_ptr as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut fv_i as *mut _ as *mut c_void,
                    &mut periods_ptr as *mut _ as *mut c_void,
                    &mut ncomb_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];

                self.stream
                    .launch(&func, grid, block, 0, args)
                    .map_err(|e| CudaMediumAdError::Cuda(e.to_string()))?;
            }

            launched += cur;
        }

        Ok(())
    }

    fn run_batch_kernel(
        &self,
        data_f32: &[f32],
        combos: &[MediumAdCombo],
        first_valid: usize,
    ) -> Result<DeviceArrayF32, CudaMediumAdError> {
        let len = data_f32.len();
        let out_bytes = combos.len() * len * std::mem::size_of::<f32>();
        let in_bytes = len * std::mem::size_of::<f32>();
        let will_fit = Self::will_fit(in_bytes + out_bytes, 64 << 20);
        if !will_fit {
            return Err(CudaMediumAdError::InvalidInput(
                "insufficient VRAM for requested launch".into(),
            ));
        }

        // Prefer async copies
        let h_data = LockedBuffer::from_slice(data_f32)
            .map_err(|e| CudaMediumAdError::Cuda(e.to_string()))?;
        let mut d_data = unsafe { DeviceBuffer::<f32>::uninitialized_async(len, &self.stream) }
            .map_err(|e| CudaMediumAdError::Cuda(e.to_string()))?;
        unsafe { d_data.async_copy_from(&h_data, &self.stream) }
            .map_err(|e| CudaMediumAdError::Cuda(e.to_string()))?;

        let periods: Vec<i32> = combos.iter().map(|c| c.period).collect();
        let d_periods = DeviceBuffer::from_slice(&periods)
            .map_err(|e| CudaMediumAdError::Cuda(e.to_string()))?;

        let mut d_out =
            unsafe { DeviceBuffer::<f32>::uninitialized_async(combos.len() * len, &self.stream) }
                .map_err(|e| CudaMediumAdError::Cuda(e.to_string()))?;

        self.launch_batch_kernel(
            &d_data,
            len,
            first_valid,
            &d_periods,
            combos.len(),
            &mut d_out,
        )?;
        self.stream
            .synchronize()
            .map_err(|e| CudaMediumAdError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: combos.len(),
            cols: len,
        })
    }

    pub fn medium_ad_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &MediumAdBatchRange,
    ) -> Result<DeviceArrayF32, CudaMediumAdError> {
        let (combos, first_valid) = Self::prepare_batch_inputs(data_f32, sweep)?;
        self.run_batch_kernel(data_f32, &combos, first_valid)
    }

    // ---------------- Many-series × one-param (time-major) ----------------

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
    ) -> Result<(Vec<i32>, usize), CudaMediumAdError> {
        if cols == 0 || rows == 0 {
            return Err(CudaMediumAdError::InvalidInput(
                "series dimensions must be positive".into(),
            ));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaMediumAdError::InvalidInput(format!(
                "data length mismatch: expected {}, got {}",
                cols * rows,
                data_tm_f32.len()
            )));
        }
        if period == 0 || period > rows {
            return Err(CudaMediumAdError::InvalidInput(
                "invalid period for many-series".into(),
            ));
        }
        if period > MEDIUM_AD_MAX_PERIOD {
            return Err(CudaMediumAdError::InvalidInput(format!(
                "period {} exceeds MEDIUM_AD_MAX_PERIOD {} for CUDA path",
                period, MEDIUM_AD_MAX_PERIOD
            )));
        }

        let mut first_valids = vec![rows as i32; cols];
        for s in 0..cols {
            let mut fv = rows as i32;
            for t in 0..rows {
                let v = data_tm_f32[t * cols + s];
                if !v.is_nan() {
                    fv = t as i32;
                    break;
                }
            }
            first_valids[s] = fv;
        }
        Ok((first_valids, period))
    }

    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        period: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaMediumAdError> {
        let func = self
            .module
            .get_function("medium_ad_many_series_one_param_f32")
            .map_err(|e| CudaMediumAdError::Cuda(e.to_string()))?;

        let block_x: u32 = 128;
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut data_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut period_i = period as i32;
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut data_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaMediumAdError::Cuda(e.to_string()))?;
        }
        Ok(())
    }

    pub fn medium_ad_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
    ) -> Result<DeviceArrayF32, CudaMediumAdError> {
        let (first_valids, period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, period)?;

        let elems = cols * rows;
        let in_bytes = elems * std::mem::size_of::<f32>();
        let out_bytes = in_bytes;
        if !Self::will_fit(in_bytes + out_bytes, 64 << 20) {
            return Err(CudaMediumAdError::InvalidInput(
                "insufficient VRAM for requested launch".into(),
            ));
        }

        let h_prices = LockedBuffer::from_slice(data_tm_f32)
            .map_err(|e| CudaMediumAdError::Cuda(e.to_string()))?;
        let h_first = LockedBuffer::from_slice(&first_valids)
            .map_err(|e| CudaMediumAdError::Cuda(e.to_string()))?;

        let mut d_prices = unsafe { DeviceBuffer::<f32>::uninitialized_async(elems, &self.stream) }
            .map_err(|e| CudaMediumAdError::Cuda(e.to_string()))?;
        let mut d_first = unsafe { DeviceBuffer::<i32>::uninitialized_async(cols, &self.stream) }
            .map_err(|e| CudaMediumAdError::Cuda(e.to_string()))?;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized_async(elems, &self.stream) }
            .map_err(|e| CudaMediumAdError::Cuda(e.to_string()))?;

        unsafe { d_prices.async_copy_from(&h_prices, &self.stream) }
            .map_err(|e| CudaMediumAdError::Cuda(e.to_string()))?;
        unsafe { d_first.async_copy_from(&h_first, &self.stream) }
            .map_err(|e| CudaMediumAdError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(&d_prices, cols, rows, period, &d_first, &mut d_out)?;

        self.stream
            .synchronize()
            .map_err(|e| CudaMediumAdError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    // Keep sizes modest due to O(p log p) per output cost on GPU
    const ONE_SERIES_LEN: usize = 200_000;
    const PARAM_SWEEP: usize = 64;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + out_bytes + (64 << 20)
    }

    struct MediumAdBatchState {
        cuda: CudaMediumAd,
        price: Vec<f32>,
        sweep: MediumAdBatchRange,
    }
    impl CudaBenchState for MediumAdBatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .medium_ad_batch_dev(&self.price, &self.sweep)
                .expect("medium_ad batch");
        }
    }

    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaMediumAd::new(0).expect("cuda medium_ad");
        let price = gen_series(ONE_SERIES_LEN);
        let start = 5usize;
        let end = start + PARAM_SWEEP - 1;
        let sweep = MediumAdBatchRange {
            period: (start, end, 1),
        };
        Box::new(MediumAdBatchState { cuda, price, sweep })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "medium_ad",
            "one_series_many_params",
            "medium_ad_cuda_batch_dev",
            "200k_x_64",
            prep_one_series_many_params,
        )
        .with_sample_size(10)
        .with_mem_required(bytes_one_series_many_params())]
    }
}
