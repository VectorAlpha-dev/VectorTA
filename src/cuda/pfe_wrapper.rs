#![cfg(feature = "cuda")]

//! CUDA wrapper for Polarized Fractal Efficiency (PFE).
//!
//! Parity items covered:
//! - PTX JIT via include_str!(concat!(env!("OUT_DIR"), "/pfe_kernel.ptx")) with DetermineTargetFromContext
//!   and OptLevel O2, plus simpler fallbacks on failure.
//! - NON_BLOCKING stream
//! - VRAM-aware chunking of parameter rows to keep grid.y within limits and avoid OOMs.
//! - Public device entry points for batch and many-series (time-major) variants.

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::pfe::PfeBatchRange;
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
pub enum CudaPfeError {
    Cuda(String),
    InvalidInput(String),
}
impl fmt::Display for CudaPfeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cuda(e) => write!(f, "CUDA error: {}", e),
            Self::InvalidInput(s) => write!(f, "Invalid input: {}", s),
        }
    }
}
impl Error for CudaPfeError {}

#[derive(Clone, Debug)]
struct PfeCombo {
    period: i32,
    smoothing: i32,
}

pub struct CudaPfe {
    pub(crate) module: Module,
    pub(crate) stream: Stream,
    _ctx: Context,
}

impl CudaPfe {
    pub fn new(device_id: usize) -> Result<Self, CudaPfeError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaPfeError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaPfeError::Cuda(e.to_string()))?;
        let ctx = Context::new(device).map_err(|e| CudaPfeError::Cuda(e.to_string()))?;
        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/pfe_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, jit_opts)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))
            .map_err(|e| CudaPfeError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaPfeError::Cuda(e.to_string()))?;
        Ok(Self {
            module,
            stream,
            _ctx: ctx,
        })
    }

    fn expand_grid(range: &PfeBatchRange) -> Vec<PfeCombo> {
        let axis = |a: (usize, usize, usize)| -> Vec<usize> {
            let (s, e, st) = a;
            if st == 0 || s == e {
                vec![s]
            } else {
                (s..=e).step_by(st).collect()
            }
        };
        let periods = axis(range.period);
        let smoothings = axis(range.smoothing);
        let mut out = Vec::with_capacity(periods.len() * smoothings.len());
        for &p in &periods {
            for &s in &smoothings {
                out.push(PfeCombo {
                    period: p as i32,
                    smoothing: s as i32,
                });
            }
        }
        out
    }

    #[inline]
    fn first_valid(data: &[f32]) -> Option<usize> {
        data.iter().position(|v| !v.is_nan())
    }

    #[inline]
    fn will_fit(total_bytes: usize, headroom: usize) -> bool {
        if let Ok((free, _)) = mem_get_info() {
            total_bytes.saturating_add(headroom) <= free
        } else {
            true
        }
    }

    #[inline]
    fn chunk_rows(n_rows: usize, len: usize) -> usize {
        // Conservative VRAM guard: reserve ~64MB headroom
        let headroom = 64usize << 20;
        let bytes_per_row = len * std::mem::size_of::<f32>(); // outputs only (inputs shared)
        if let Ok((free, _)) = mem_get_info() {
            if free > headroom {
                let cap = (free - headroom) / bytes_per_row;
                return cap.max(1).min(65_000).min(n_rows).max(1);
            }
        }
        n_rows.min(65_000).max(1)
    }

    #[inline]
    fn clone_fill_head_with_first_valid(data: &[f32], first_valid: usize) -> Vec<f32> {
        if first_valid == 0 { return data.to_vec(); }
        let mut v = data.to_vec();
        let seed = v[first_valid];
        for i in 0..=first_valid { v[i] = seed; }
        v
    }

    pub fn pfe_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &PfeBatchRange,
    ) -> Result<DeviceArrayF32, CudaPfeError> {
        if data_f32.is_empty() {
            return Err(CudaPfeError::InvalidInput("empty data".into()));
        }
        let len = data_f32.len();
        let first_valid = Self::first_valid(data_f32)
            .ok_or_else(|| CudaPfeError::InvalidInput("all NaN".into()))?;
        let combos = Self::expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaPfeError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        // Validate parameters
        for c in &combos {
            let p = c.period as usize;
            let s = c.smoothing as usize;
            if p == 0 || p > len {
                return Err(CudaPfeError::InvalidInput("invalid period".into()));
            }
            if s == 0 {
                return Err(CudaPfeError::InvalidInput("invalid smoothing".into()));
            }
            if len - first_valid < p + 1 {
                return Err(CudaPfeError::InvalidInput("not enough valid data".into()));
            }
        }

        // Fast path: device-built steps + dual-FP32 prefix + many-params kernel
        if let (Ok(func_steps), Ok(func_pref), Ok(func_main)) = (
            self.module.get_function("pfe_build_steps_f32"),
            self.module.get_function("pfe_build_prefix_float2_serial"),
            self.module.get_function("pfe_many_params_prefix_f32"),
        ) {
            // Fill the head to avoid NaN in prefix; zeros out head steps in differences.
            let data_filled = Self::clone_fill_head_with_first_valid(data_f32, first_valid);

            // Device inputs
            let d_data = DeviceBuffer::from_slice(&data_filled)
                .map_err(|e| CudaPfeError::Cuda(e.to_string()))?;
            let periods: Vec<i32> = combos.iter().map(|c| c.period).collect();
            let smooths: Vec<i32> = combos.iter().map(|c| c.smoothing).collect();
            let d_periods = DeviceBuffer::from_slice(&periods)
                .map_err(|e| CudaPfeError::Cuda(e.to_string()))?;
            let d_smooths = DeviceBuffer::from_slice(&smooths)
                .map_err(|e| CudaPfeError::Cuda(e.to_string()))?;

            // 1) steps[t] in parallel
            let mut d_steps: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(len) }
                .map_err(|e| CudaPfeError::Cuda(e.to_string()))?;
            let block_x: u32 = 256;
            let grid_x: u32 = (((len as u32) + block_x - 1) / block_x).max(1);
            let grid_1d: GridSize = (grid_x, 1, 1).into();
            let block_1d: BlockSize = (block_x, 1, 1).into();
            unsafe {
                let mut data_ptr = d_data.as_device_ptr().as_raw();
                let mut len_i = len as i32;
                let mut steps_ptr = d_steps.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut data_ptr as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut steps_ptr as *mut _ as *mut c_void,
                ];
                self.stream
                    .launch(&func_steps, grid_1d, block_1d, 0, args)
                    .map_err(|e| CudaPfeError::Cuda(e.to_string()))?;
            }

            // 2) dual-FP32 prefix (serial)
            let mut d_pref_hi: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(len) }
                .map_err(|e| CudaPfeError::Cuda(e.to_string()))?;
            let mut d_pref_lo: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(len) }
                .map_err(|e| CudaPfeError::Cuda(e.to_string()))?;
            unsafe {
                let mut steps_ptr = d_steps.as_device_ptr().as_raw();
                let mut len_i = len as i32;
                let mut hi_ptr = d_pref_hi.as_device_ptr().as_raw();
                let mut lo_ptr = d_pref_lo.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut steps_ptr as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut hi_ptr as *mut _ as *mut c_void,
                    &mut lo_ptr as *mut _ as *mut c_void,
                ];
                let grid_one: GridSize = (1u32, 1u32, 1u32).into();
                let block_one: BlockSize = (1u32, 1u32, 1u32).into();
                self.stream
                    .launch(&func_pref, grid_one, block_one, 0, args)
                    .map_err(|e| CudaPfeError::Cuda(e.to_string()))?;
            }
            drop(d_steps);

            // 3) outputs
            let total_out = combos.len() * len;
            let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total_out) }
                .map_err(|e| CudaPfeError::Cuda(e.to_string()))?;

            // 4) main many-params kernel
            let block_x: u32 = 256;
            let grid_x: u32 = (((combos.len() as u32) + block_x - 1) / block_x).max(1);
            let grid_np: GridSize = (grid_x, 1, 1).into();
            let block_np: BlockSize = (block_x, 1, 1).into();
            unsafe {
                let mut data_ptr = d_data.as_device_ptr().as_raw();
                let mut hi_ptr = d_pref_hi.as_device_ptr().as_raw();
                let mut lo_ptr = d_pref_lo.as_device_ptr().as_raw();
                let mut len_i = len as i32;
                let mut fv_i = first_valid as i32;
                let mut per_ptr = d_periods.as_device_ptr().as_raw();
                let mut sm_ptr = d_smooths.as_device_ptr().as_raw();
                let mut ncomb_i = combos.len() as i32;
                let mut out_ptr = d_out.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut data_ptr as *mut _ as *mut c_void,
                    &mut hi_ptr as *mut _ as *mut c_void,
                    &mut lo_ptr as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut fv_i as *mut _ as *mut c_void,
                    &mut per_ptr as *mut _ as *mut c_void,
                    &mut sm_ptr as *mut _ as *mut c_void,
                    &mut ncomb_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream
                    .launch(&func_main, grid_np, block_np, 0, args)
                    .map_err(|e| CudaPfeError::Cuda(e.to_string()))?;
            }

            self.stream
                .synchronize()
                .map_err(|e| CudaPfeError::Cuda(e.to_string()))?;

            return Ok(DeviceArrayF32 { buf: d_out, rows: combos.len(), cols: len });
        }

        // Fallback: existing prefix/rolling pathways
        // Precompute shared prefix on host for the legacy prefix kernel
        let mut prefix = vec![0.0f64; len];
        for i in 1..len {
            let d = (data_f32[i] as f64) - (data_f32[i - 1] as f64);
            prefix[i] = prefix[i - 1] + (d.mul_add(d, 1.0)).sqrt();
        }

        // Device buffers
        let d_data =
            DeviceBuffer::from_slice(data_f32).map_err(|e| CudaPfeError::Cuda(e.to_string()))?;
        let d_prefix =
            DeviceBuffer::from_slice(&prefix).map_err(|e| CudaPfeError::Cuda(e.to_string()))?;
        let periods: Vec<i32> = combos.iter().map(|c| c.period).collect();
        let smooths: Vec<i32> = combos.iter().map(|c| c.smoothing).collect();
        let d_periods =
            DeviceBuffer::from_slice(&periods).map_err(|e| CudaPfeError::Cuda(e.to_string()))?;
        let d_smooths =
            DeviceBuffer::from_slice(&smooths).map_err(|e| CudaPfeError::Cuda(e.to_string()))?;

        let total_out = combos.len() * len;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total_out) }
            .map_err(|e| CudaPfeError::Cuda(e.to_string()))?;

        if let Ok(func) = self.module.get_function("pfe_batch_prefix_f32") {
            let block_x: u32 = 256;
            let grid_x: u32 = (((combos.len() as u32) + block_x - 1) / block_x).max(1);
            let grid: GridSize = (grid_x, 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            unsafe {
                let mut data_ptr = d_data.as_device_ptr().as_raw();
                let mut pref_ptr = d_prefix.as_device_ptr().as_raw();
                let mut len_i = len as i32;
                let mut fv_i = first_valid as i32;
                let mut per_ptr = d_periods.as_device_ptr().as_raw();
                let mut sm_ptr = d_smooths.as_device_ptr().as_raw();
                let mut ncomb_i = combos.len() as i32;
                let mut out_ptr = d_out.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut data_ptr as *mut _ as *mut c_void,
                    &mut pref_ptr as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut fv_i as *mut _ as *mut c_void,
                    &mut per_ptr as *mut _ as *mut c_void,
                    &mut sm_ptr as *mut _ as *mut c_void,
                    &mut ncomb_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream
                    .launch(&func, grid, block, 0, args)
                    .map_err(|e| CudaPfeError::Cuda(e.to_string()))?;
            }
        } else {
            let func = self
                .module
                .get_function("pfe_batch_f32")
                .map_err(|e| CudaPfeError::Cuda(e.to_string()))?;
            let chunk = Self::chunk_rows(combos.len(), len);
            let mut launched = 0usize;
            while launched < combos.len() {
                let cur = (combos.len() - launched).min(chunk);
                let block_x: u32 = 256;
                let grid_x: u32 = (((cur as u32) + block_x - 1) / block_x).max(1);
                let grid: GridSize = (grid_x, 1, 1).into();
                let block: BlockSize = (block_x, 1, 1).into();
                unsafe {
                    let mut data_ptr = d_data.as_device_ptr().as_raw();
                    let mut len_i = len as i32;
                    let mut fv_i = first_valid as i32;
                    let mut per_ptr = d_periods
                        .as_device_ptr()
                        .as_raw()
                        .wrapping_add((launched * std::mem::size_of::<i32>()) as u64);
                    let mut sm_ptr = d_smooths
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
                        &mut per_ptr as *mut _ as *mut c_void,
                        &mut sm_ptr as *mut _ as *mut c_void,
                        &mut ncomb_i as *mut _ as *mut c_void,
                        &mut out_ptr as *mut _ as *mut c_void,
                    ];
                    self.stream
                        .launch(&func, grid, block, 0, args)
                        .map_err(|e| CudaPfeError::Cuda(e.to_string()))?;
                }
                launched += cur;
            }
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaPfeError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: combos.len(),
            cols: len,
        })
    }

    pub fn pfe_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
        smoothing: usize,
    ) -> Result<DeviceArrayF32, CudaPfeError> {
        if cols == 0 || rows == 0 {
            return Err(CudaPfeError::InvalidInput("cols/rows must be > 0".into()));
        }
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaPfeError::InvalidInput("rows*cols overflow".into()))?;
        if data_tm_f32.len() != expected {
            return Err(CudaPfeError::InvalidInput(
                "time-major input length mismatch".into(),
            ));
        }
        if period == 0 || period > rows {
            return Err(CudaPfeError::InvalidInput("invalid period".into()));
        }
        if smoothing == 0 {
            return Err(CudaPfeError::InvalidInput("invalid smoothing".into()));
        }

        // Build per-series first_valid indices
        let mut fvs = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = 0usize;
            while fv < rows {
                let v = data_tm_f32[fv * cols + s];
                if !v.is_nan() {
                    break;
                }
                fv += 1;
            }
            fvs[s] = fv as i32;
        }

        let d_tm =
            DeviceBuffer::from_slice(data_tm_f32).map_err(|e| CudaPfeError::Cuda(e.to_string()))?;
        let d_fv = DeviceBuffer::from_slice(&fvs).map_err(|e| CudaPfeError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(expected) }
            .map_err(|e| CudaPfeError::Cuda(e.to_string()))?;

        let func = self
            .module
            .get_function("pfe_many_series_one_param_time_major_f32")
            .map_err(|e| CudaPfeError::Cuda(e.to_string()))?;
        let block_x: u32 = 256;
        let grid_x: u32 = (((cols as u32) + block_x - 1) / block_x).max(1);
        let grid: GridSize = (grid_x, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe {
            let mut data_ptr = d_tm.as_device_ptr().as_raw();
            let mut fv_ptr = d_fv.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut p_i = period as i32;
            let mut s_i = smoothing as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut data_ptr as *mut _ as *mut c_void,
                &mut fv_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut p_i as *mut _ as *mut c_void,
                &mut s_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaPfeError::Cuda(e.to_string()))?;
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaPfeError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    pub fn synchronize(&self) -> Result<(), CudaPfeError> {
        self.stream
            .synchronize()
            .map_err(|e| CudaPfeError::Cuda(e.to_string()))
    }
}

pub mod benches {
    use super::*;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "pfe",
                "batch_dev",
                "pfe_cuda_batch_dev",
                "60k_x_grid",
                prep_pfe_batch_box,
            ),
            CudaBenchScenario::new(
                "pfe",
                "many_series_one_param",
                "pfe_cuda_many_series_one_param",
                "250x1m",
                prep_pfe_many_series_box,
            ),
        ]
    }

    struct PfeBatchState {
        cuda: CudaPfe,
        d_data: DeviceBuffer<f32>,
        d_periods: DeviceBuffer<i32>,
        d_smooths: DeviceBuffer<i32>,
        d_out: DeviceBuffer<f32>,
        len: usize,
        n_combos: usize,
        first_valid: usize,
    }
    impl CudaBenchState for PfeBatchState {
        fn launch(&mut self) {
            let func = self
                .cuda
                .module
                .get_function("pfe_batch_f32")
                .expect("func");
            let block_x: u32 = 256;
            let grid_x: u32 = (((self.n_combos as u32) + block_x - 1) / block_x).max(1);
            let grid: GridSize = (grid_x, 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            unsafe {
                let mut data_ptr = self.d_data.as_device_ptr().as_raw();
                let mut len_i = self.len as i32;
                let mut fv_i = self.first_valid as i32;
                let mut per_ptr = self.d_periods.as_device_ptr().as_raw();
                let mut sm_ptr = self.d_smooths.as_device_ptr().as_raw();
                let mut ncomb_i = self.n_combos as i32;
                let mut out_ptr = self.d_out.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut data_ptr as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut fv_i as *mut _ as *mut c_void,
                    &mut per_ptr as *mut _ as *mut c_void,
                    &mut sm_ptr as *mut _ as *mut c_void,
                    &mut ncomb_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.cuda
                    .stream
                    .launch(&func, grid, block, 0, args)
                    .expect("launch");
            }
            self.cuda.synchronize().expect("sync");
        }
    }

    fn prep_pfe_batch() -> PfeBatchState {
        let cuda = CudaPfe::new(0).expect("cuda pfe");
        let len = 60_000usize;
        let mut price = vec![f32::NAN; len];
        for i in 10..len {
            let x = i as f32;
            price[i] = (x * 0.001).sin() + 0.0002 * x;
        }
        let mut periods = Vec::new();
        let mut smooths = Vec::new();
        for p in 5..=49 {
            for s in [3, 5, 7, 9] {
                periods.push(p as i32);
                smooths.push(s as i32);
            }
        }
        let first_valid = price.iter().position(|v| !v.is_nan()).unwrap_or(0);
        let d_data = DeviceBuffer::from_slice(&price).expect("d_data");
        let d_periods = DeviceBuffer::from_slice(&periods).expect("d_periods");
        let d_smooths = DeviceBuffer::from_slice(&smooths).expect("d_smooths");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(periods.len() * len) }.expect("d_out");
        PfeBatchState {
            cuda,
            d_data,
            d_periods,
            d_smooths,
            d_out,
            len,
            n_combos: periods.len(),
            first_valid,
        }
    }
    fn prep_pfe_batch_box() -> Box<dyn CudaBenchState> {
        Box::new(prep_pfe_batch())
    }

    struct PfeManySeriesState {
        cuda: CudaPfe,
        d_tm: DeviceBuffer<f32>,
        d_fv: DeviceBuffer<i32>,
        d_out: DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        period: usize,
        smoothing: usize,
    }
    impl CudaBenchState for PfeManySeriesState {
        fn launch(&mut self) {
            let func = self
                .cuda
                .module
                .get_function("pfe_many_series_one_param_time_major_f32")
                .expect("func");
            let block_x: u32 = 256;
            let grid_x: u32 = (((self.cols as u32) + block_x - 1) / block_x).max(1);
            let grid: GridSize = (grid_x, 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            unsafe {
                let mut data_ptr = self.d_tm.as_device_ptr().as_raw();
                let mut fv_ptr = self.d_fv.as_device_ptr().as_raw();
                let mut cols_i = self.cols as i32;
                let mut rows_i = self.rows as i32;
                let mut p_i = self.period as i32;
                let mut s_i = self.smoothing as i32;
                let mut out_ptr = self.d_out.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut data_ptr as *mut _ as *mut c_void,
                    &mut fv_ptr as *mut _ as *mut c_void,
                    &mut cols_i as *mut _ as *mut c_void,
                    &mut rows_i as *mut _ as *mut c_void,
                    &mut p_i as *mut _ as *mut c_void,
                    &mut s_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.cuda
                    .stream
                    .launch(&func, grid, block, 0, args)
                    .expect("launch");
            }
            self.cuda.synchronize().expect("sync");
        }
    }

    fn prep_pfe_many_series() -> PfeManySeriesState {
        let cuda = CudaPfe::new(0).expect("cuda pfe");
        let cols = 250usize;
        let rows = 1_000_000usize;
        let period = 20usize;
        let smoothing = 5usize;
        let mut tm = vec![f32::NAN; cols * rows];
        for s in 0..cols {
            for t in s..rows {
                let x = (t as f32) + (s as f32) * 0.1;
                tm[t * cols + s] = (x * 0.002).sin() + 0.0002 * x;
            }
        }
        let mut fvs = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = 0usize;
            while fv < rows && tm[fv * cols + s].is_nan() {
                fv += 1;
            }
            fvs[s] = fv as i32;
        }
        let d_tm = DeviceBuffer::from_slice(&tm).expect("d_tm");
        let d_fv = DeviceBuffer::from_slice(&fvs).expect("d_fv");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(cols * rows) }.expect("d_out");
        PfeManySeriesState {
            cuda,
            d_tm,
            d_fv,
            d_out,
            cols,
            rows,
            period,
            smoothing,
        }
    }

    fn prep_pfe_many_series_box() -> Box<dyn CudaBenchState> {
        Box::new(prep_pfe_many_series())
    }
}
