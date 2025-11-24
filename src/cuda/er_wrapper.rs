#![cfg(feature = "cuda")]

//! CUDA wrapper for Kaufman Efficiency Ratio (ER).
//!
//! Parity with ALMA/CUDA wrappers:
//! - PTX load via include_str!(concat!(env!("OUT_DIR"), "/er_kernel.ptx")) with JIT options
//!   DetermineTargetFromContext + OptLevel O2, then simpler fallbacks.
//! - NON_BLOCKING stream
//! - VRAM guard and grid.y chunking (<= 65_535)
//! - Public device entry points for batch and many-series (time-major)

use crate::indicators::er::ErBatchRange;
use cust::context::Context;
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, GridSize};
use cust::launch;
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;
use std::sync::Arc;
use thiserror::Error;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct Float2 { pub x: f32, pub y: f32 }

// Safe because Float2 is plain-old-data with no pointers and #[repr(C)]
unsafe impl cust::memory::DeviceCopy for Float2 {}

#[derive(Debug, Error)]
pub enum CudaErError {
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

#[derive(Clone, Debug)]
struct ErCombo {
    period: i32,
}

/// VRAM-backed array for ER with context guard and device id
pub struct DeviceArrayF32Er {
    pub buf: DeviceBuffer<f32>,
    pub rows: usize,
    pub cols: usize,
    pub ctx: Arc<Context>,
    pub device_id: u32,
}
impl DeviceArrayF32Er {
    #[inline]
    pub fn device_ptr(&self) -> u64 { self.buf.as_device_ptr().as_raw() as u64 }
    #[inline]
    pub fn len(&self) -> usize { self.rows * self.cols }
}

pub struct CudaEr {
    pub(crate) module: Module,
    pub(crate) stream: Stream,
    _ctx: Arc<Context>,
    device_id: u32,
}

impl CudaEr {
    pub fn new(device_id: usize) -> Result<Self, CudaErError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let ctx = Arc::new(Context::new(device)?);
        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/er_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, jit_opts)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        Ok(Self { module, stream, _ctx: ctx, device_id: device_id as u32 })
    }

    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> bool {
        if let Ok((free, _)) = mem_get_info() {
            required_bytes.saturating_add(headroom_bytes) <= free
        } else {
            true
        }
    }

    #[inline]
    fn device_max_grid_xy(&self) -> Result<(u32, u32), CudaErError> {
        let dev = Device::get_device(self.device_id).map_err(CudaErError::Cuda)?;
        let gx = dev.get_attribute(DeviceAttribute::MaxGridDimX).map_err(CudaErError::Cuda)? as u32;
        let gy = dev.get_attribute(DeviceAttribute::MaxGridDimY).map_err(CudaErError::Cuda)? as u32;
        Ok((gx, gy))
    }

    fn expand_grid(range: &ErBatchRange) -> Vec<ErCombo> {
        let (start, end, step) = range.period;
        let periods: Vec<usize> = if step == 0 || start == end {
            vec![start]
        } else if start < end {
            (start..=end).step_by(step.max(1)).collect()
        } else {
            let mut v = Vec::new();
            let mut x = start as isize;
            let end_i = end as isize;
            let st = (step.max(1)) as isize;
            while x >= end_i { v.push(x as usize); x -= st; }
            v
        };
        periods.into_iter().map(|p| ErCombo { period: p as i32 }).collect()
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &ErBatchRange,
    ) -> Result<(Vec<ErCombo>, usize), CudaErError> {
        if data_f32.is_empty() {
            return Err(CudaErError::InvalidInput("empty data".into()));
        }
        let len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaErError::InvalidInput("all NaN".into()))?;
        let first_valid = data_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaErError::InvalidInput("all NaN".into()))?;
        let combos = Self::expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaErError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        if combos.is_empty() {
            return Err(CudaErError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        // Validate periods
        for c in &combos {
            let p = c.period as usize;
            if p == 0 || p > len {
                return Err(CudaErError::InvalidInput(format!(
                    "invalid period {} for len {}",
                    p, len
                )));
            }
            if len - first_valid < p {
                return Err(CudaErError::InvalidInput(format!(
                    "not enough valid data: needed >= {}, valid = {}",
                    p,
                    len - first_valid
                )));
            }
            if p == 0 || p > len {
                return Err(CudaErError::InvalidInput(format!(
                    "invalid period {} for len {}",
                    p, len
                )));
            }
            if len - first_valid < p {
                return Err(CudaErError::InvalidInput(format!(
                    "not enough valid data: needed >= {}, valid = {}",
                    p,
                    len - first_valid
                )));
            }
        }
        Ok((combos, first_valid))
    }

    fn build_prefix_absdiff_dsf(data_f32: &[f32]) -> Vec<Float2> {
        // Build DS prefix of abs diffs: prefix[t] = sum_{k=0..t-1} |x[k+1]-x[k]| in double-single
        let n = data_f32.len();
        let mut pref = vec![Float2 { x: 0.0, y: 0.0 }; n];
        let two_sumf = |a: f32, b: f32| -> (f32, f32) {
            let t = a + b;
            let bp = t - a;
            let e = (a - (t - bp)) + (b - bp);
            (t, e)
        };
        let mut hi: f32 = 0.0;
        let mut lo: f32 = 0.0;
        if let Some(first) = data_f32.iter().position(|v| !v.is_nan()) {
            let mut j = first;
            while j + 1 < n {
                let d = (data_f32[j + 1] - data_f32[j]).abs();
                let (s1, e1) = two_sumf(hi, d);
                let lo1 = lo + e1;
                let (s2, e2) = two_sumf(s1, lo1);
                hi = s2; lo = e2;
                pref[j + 1] = Float2 { x: hi, y: lo };
                j += 1;
            }
        }
        pref
    }

    #[inline]
    fn chunk_rows(n_rows: usize, len: usize) -> usize {
        let max_grid_y = 65_000usize; // keep below hardware limit
        let out_bytes = n_rows
            .saturating_mul(len)
            .saturating_mul(std::mem::size_of::<f32>());
        let out_bytes = n_rows
            .saturating_mul(len)
            .saturating_mul(std::mem::size_of::<f32>());
        if let Ok((free, _)) = mem_get_info() {
            let headroom = 64usize << 20; // ~64MB
            if free > headroom {
                return (free - headroom)
                    .saturating_div(len * std::mem::size_of::<f32>())
                    .max(1)
                    .min(max_grid_y);
            }
        }
        max_grid_y.min(n_rows).max(1)
    }

    pub fn er_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &ErBatchRange,
    ) -> Result<DeviceArrayF32Er, CudaErError> {
        let (combos, first_valid) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let len = data_f32.len();
        let n_combos = combos.len();

        // Build DS prefix on host and prefer prefix kernel by default.
        let prefix = Self::build_prefix_absdiff_dsf(data_f32);

        // Estimate VRAM including prefix; if too tight, fallback to rolling DS kernel.
        let bytes_est = len
            .checked_mul(std::mem::size_of::<f32>())
            .and_then(|b| b.checked_add(n_combos.checked_mul(std::mem::size_of::<i32>())?))
            .and_then(|b| b.checked_add(n_combos.checked_mul(len)?.checked_mul(std::mem::size_of::<f32>())?))
            .and_then(|b| b.checked_add(len.checked_mul(std::mem::size_of::<Float2>())?))
            .ok_or_else(|| CudaErError::InvalidInput("size overflow".into()))?;
        if !Self::will_fit(bytes_est, 64usize << 20) {
            return self.er_batch_dev_fallback_rolling(data_f32, &combos, first_valid);
        }

        // Device buffers
        let d_data = DeviceBuffer::from_slice(data_f32)?;
        let periods: Vec<i32> = combos.iter().map(|c| c.period).collect();
        let d_periods = DeviceBuffer::from_slice(&periods)?;
        let d_prefix = DeviceBuffer::from_slice(&prefix)?;
        let total = n_combos
            .checked_mul(len)
            .ok_or_else(|| CudaErError::InvalidInput("rows*cols overflow".into()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total) }?;

        let func = self
            .module
            .get_function("er_batch_prefix_f32")
            .map_err(|_| CudaErError::MissingKernelSymbol { name: "er_batch_prefix_f32" })?;
        let block_x: u32 = 256;
        let grid_x: u32 = ((len as u32) + block_x - 1) / block_x;
        let chunk = Self::chunk_rows(n_combos, len);
        let mut launched = 0usize;
        let (max_gx, max_gy) = self.device_max_grid_xy()?;
        while launched < n_combos {
            let cur = (n_combos - launched).min(chunk);
            let grid: GridSize = (grid_x.max(1), cur as u32, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            if grid_x > max_gx || (cur as u32) > max_gy {
                return Err(CudaErError::LaunchConfigTooLarge { gx: grid_x, gy: cur as u32, gz: 1, bx: block_x, by: 1, bz: 1 });
            }
            unsafe {
                let mut data_ptr = d_data.as_device_ptr().as_raw();
                let mut pref_ptr = d_prefix.as_device_ptr().as_raw();
                let mut len_i = len as i32;
                let mut fv_i = first_valid as i32;
                let mut per_ptr = d_periods.as_device_ptr().as_raw().wrapping_add((launched * std::mem::size_of::<i32>()) as u64);
                let mut ncomb_i = cur as i32;
                let mut out_ptr = d_out.as_device_ptr().as_raw().wrapping_add((launched * len * std::mem::size_of::<f32>()) as u64);
                let args: &mut [*mut c_void] = &mut [
                    &mut data_ptr as *mut _ as *mut c_void,
                    &mut pref_ptr as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut fv_i as *mut _ as *mut c_void,
                    &mut per_ptr as *mut _ as *mut c_void,
                    &mut ncomb_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream.launch(&func, grid, block, 0, args)?;
            }
            launched += cur;
        }
        self.stream.synchronize()?;
        Ok(DeviceArrayF32Er { buf: d_out, rows: n_combos, cols: len, ctx: Arc::clone(&self._ctx), device_id: self.device_id })
    }

    fn er_batch_dev_fallback_rolling(&self, data_f32: &[f32], combos: &[ErCombo], first_valid: usize) -> Result<DeviceArrayF32Er, CudaErError> {
        let len = data_f32.len();
        let n_combos = combos.len();
        let d_data = DeviceBuffer::from_slice(data_f32)?;
        let periods: Vec<i32> = combos.iter().map(|c| c.period).collect();
        let d_periods = DeviceBuffer::from_slice(&periods)?;
        let total = n_combos
            .checked_mul(len)
            .ok_or_else(|| CudaErError::InvalidInput("rows*cols overflow".into()))?;
        let out_bytes = total
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaErError::InvalidInput("size overflow".into()))?;
        if let Ok((free, _)) = mem_get_info() {
            let headroom = 64usize << 20;
            if out_bytes.saturating_add(headroom) > free {
                return Err(CudaErError::OutOfMemory { required: out_bytes, free, headroom });
            }
        }
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total) }?;

        let func = self
            .module
            .get_function("er_batch_f32")
            .map_err(|_| CudaErError::MissingKernelSymbol { name: "er_batch_f32" })?;
        let block_x: u32 = 256;
        let grid_x: u32 = ((n_combos as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        let (max_gx, _max_gy) = self.device_max_grid_xy()?;
        if grid_x > max_gx {
            return Err(CudaErError::LaunchConfigTooLarge { gx: grid_x, gy: 1, gz: 1, bx: block_x, by: 1, bz: 1 });
        }
        unsafe {
            let mut data_ptr = d_data.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut fv_i = first_valid as i32;
            let mut per_ptr = d_periods.as_device_ptr().as_raw();
            let mut ncomb_i = n_combos as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut data_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut fv_i as *mut _ as *mut c_void,
                &mut per_ptr as *mut _ as *mut c_void,
                &mut ncomb_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        self.stream.synchronize()?;
        Ok(DeviceArrayF32Er { buf: d_out, rows: n_combos, cols: len, ctx: Arc::clone(&self._ctx), device_id: self.device_id })
    }

    pub fn er_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
    ) -> Result<DeviceArrayF32Er, CudaErError> {
        if cols == 0 || rows == 0 {
            return Err(CudaErError::InvalidInput("cols/rows must be > 0".into()));
        }
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaErError::InvalidInput("rows*cols overflow".into()))?;
        if data_tm_f32.len() != expected {
            return Err(CudaErError::InvalidInput(
                "time-major input length mismatch".into(),
            ));
        }
        if period == 0 || period > rows {
            return Err(CudaErError::InvalidInput("invalid period".into()));
        }
        if cols == 0 || rows == 0 {
            return Err(CudaErError::InvalidInput("cols/rows must be > 0".into()));
        }
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaErError::InvalidInput("rows*cols overflow".into()))?;
        if data_tm_f32.len() != expected {
            return Err(CudaErError::InvalidInput(
                "time-major input length mismatch".into(),
            ));
        }
        if period == 0 || period > rows {
            return Err(CudaErError::InvalidInput("invalid period".into()));
        }

        // First-valid per series
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = None;
            for t in 0..rows {
                let v = data_tm_f32[t * cols + s];
                if !v.is_nan() {
                    fv = Some(t as i32);
                    break;
                }
                if !v.is_nan() {
                    fv = Some(t as i32);
                    break;
                }
            }
            let fv =
                fv.ok_or_else(|| CudaErError::InvalidInput(format!("series {} all NaN", s)))?;
            if rows - (fv as usize) < period {
                return Err(CudaErError::InvalidInput(format!(
                    "series {} not enough valid data",
                    s
                )));
            }
            first_valids[s] = fv;
        }

        // Device buffers
        let d_data = DeviceBuffer::from_slice(data_tm_f32)?;
        let d_fv = DeviceBuffer::from_slice(&first_valids)?;
        let out_bytes = expected
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaErError::InvalidInput("size overflow".into()))?;
        if let Ok((free, _)) = mem_get_info() {
            let headroom = 64usize << 20;
            if out_bytes.saturating_add(headroom) > free {
                return Err(CudaErError::OutOfMemory { required: out_bytes, free, headroom });
            }
        }
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(expected) }?;

        let func = self
            .module
            .get_function("er_many_series_one_param_time_major_f32")
            .map_err(|_| CudaErError::MissingKernelSymbol { name: "er_many_series_one_param_time_major_f32" })?;
        let block_x: u32 = 256;
        let grid_x: u32 = ((cols as u32) + block_x - 1) / block_x;
        let (max_gx, _max_gy) = self.device_max_grid_xy()?;
        if grid_x > max_gx {
            return Err(CudaErError::LaunchConfigTooLarge { gx: grid_x, gy: 1, gz: 1, bx: block_x, by: 1, bz: 1 });
        }
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe {
            let mut data_ptr = d_data.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut period_i = period as i32;
            let mut fv_ptr = d_fv.as_device_ptr().as_raw();
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut data_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut fv_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        self.stream.synchronize()?;
        Ok(DeviceArrayF32Er { buf: d_out, rows, cols, ctx: Arc::clone(&self._ctx), device_id: self.device_id })
    }

    pub fn synchronize(&self) -> Result<(), CudaErError> {
        self.stream.synchronize()?;
        Ok(())
    }
}

pub mod benches {
    use super::*;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "er",
                "batch_dev",
                "er_cuda_batch_dev",
                "60k_x_49",
                prep_er_batch_box,
            ),
            CudaBenchScenario::new(
                "er",
                "batch_dev",
                "er_cuda_batch_dev",
                "60k_x_49",
                prep_er_batch_box,
            ),
            CudaBenchScenario::new(
                "er",
                "many_series_one_param",
                "er_cuda_many_series_one_param",
                "250x1m",
                prep_er_many_series_box,
            ),
        ]
    }

    struct ErBatchState {
        cuda: CudaEr,
        d_data: DeviceBuffer<f32>,
        d_periods: DeviceBuffer<i32>,
        d_prefix: DeviceBuffer<super::Float2>,
        d_out: DeviceBuffer<f32>,
        len: usize,
        n_combos: usize,
        first_valid: usize,
    }
    impl CudaBenchState for ErBatchState {
        fn launch(&mut self) {
            // Direct kernel launch path
            let func = self
                .cuda
                .module
                .get_function("er_batch_prefix_f32")
                .expect("func");
            let block_x: u32 = 256;
            let grid_x: u32 = ((self.len as u32) + block_x - 1) / block_x;
            let grid: GridSize = (grid_x.max(1), self.n_combos as u32, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            unsafe {
                let mut data_ptr = self.d_data.as_device_ptr().as_raw();
                let mut pref_ptr = self.d_prefix.as_device_ptr().as_raw();
                let mut len_i = self.len as i32;
                let mut fv_i = self.first_valid as i32;
                let mut per_ptr = self.d_periods.as_device_ptr().as_raw();
                let mut ncomb_i = self.n_combos as i32;
                let mut out_ptr = self.d_out.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut data_ptr as *mut _ as *mut c_void,
                    &mut pref_ptr as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut fv_i as *mut _ as *mut c_void,
                    &mut per_ptr as *mut _ as *mut c_void,
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

    fn prep_er_batch() -> ErBatchState {
        let cuda = CudaEr::new(0).expect("cuda er");
        let len = 60_000usize;
        let mut price = vec![f32::NAN; len];
        for i in 5..len {
            let x = i as f32;
            price[i] = (x * 0.001).sin() + 0.0002 * x;
        }
        for i in 5..len {
            let x = i as f32;
            price[i] = (x * 0.001).sin() + 0.0002 * x;
        }
        let sweep = ErBatchRange { period: (5, 49, 1) };
        let combos: Vec<i32> = (5..=49)
            .collect::<Vec<_>>()
            .into_iter()
            .map(|p| p as i32)
            .collect();
        let combos: Vec<i32> = (5..=49)
            .collect::<Vec<_>>()
            .into_iter()
            .map(|p| p as i32)
            .collect();
        let first_valid = price.iter().position(|v| !v.is_nan()).unwrap_or(0);
        let prefix = super::CudaEr::build_prefix_absdiff_dsf(&price);
        let d_data = DeviceBuffer::from_slice(&price).expect("d_data");
        let d_periods = DeviceBuffer::from_slice(&combos).expect("d_periods");
        let d_prefix = DeviceBuffer::from_slice(&prefix).expect("d_prefix");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(combos.len() * len) }.expect("d_out");
        ErBatchState {
            cuda,
            d_data,
            d_periods,
            d_prefix,
            d_out,
            len,
            n_combos: combos.len(),
            first_valid,
        }
    }

    fn prep_er_batch_box() -> Box<dyn CudaBenchState> {
        Box::new(prep_er_batch())
    }

    struct ErManySeriesState {
        cuda: CudaEr,
        d_tm: DeviceBuffer<f32>,
        d_fv: DeviceBuffer<i32>,
        d_out: DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        period: usize,
    }
    impl CudaBenchState for ErManySeriesState {
        fn launch(&mut self) {
            let func = self
                .cuda
                .module
                .get_function("er_many_series_one_param_time_major_f32")
                .expect("func");
            let block_x: u32 = 256;
            let grid_x: u32 = ((self.cols as u32) + block_x - 1) / block_x;
            let grid: GridSize = (grid_x.max(1), 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            unsafe {
                let mut data_ptr = self.d_tm.as_device_ptr().as_raw();
                let mut cols_i = self.cols as i32;
                let mut rows_i = self.rows as i32;
                let mut per_i = self.period as i32;
                let mut fv_ptr = self.d_fv.as_device_ptr().as_raw();
                let mut out_ptr = self.d_out.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut data_ptr as *mut _ as *mut c_void,
                    &mut cols_i as *mut _ as *mut c_void,
                    &mut rows_i as *mut _ as *mut c_void,
                    &mut per_i as *mut _ as *mut c_void,
                    &mut fv_ptr as *mut _ as *mut c_void,
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

    fn prep_er_many_series() -> ErManySeriesState {
        let cuda = CudaEr::new(0).expect("cuda er");
        let cols = 250usize;
        let rows = 1_000_000usize;
        let period = 20usize;
        let mut tm = vec![f32::NAN; cols * rows];
        for s in 0..cols {
            for t in s..rows {
                let x = (t as f32) + (s as f32) * 0.1;
                tm[t * cols + s] = (x * 0.002).sin() + 0.0002 * x;
            }
        }
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
        ErManySeriesState {
            cuda,
            d_tm,
            d_fv,
            d_out,
            cols,
            rows,
            period,
        }
    }

    fn prep_er_many_series_box() -> Box<dyn CudaBenchState> {
        Box::new(prep_er_many_series())
    }
}
