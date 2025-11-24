//! CUDA wrapper for the 3-pole SuperSmoother filter kernels.
//!
//! Mirrors the ALMA/CWMA-style wrapper: VRAM-first device handles, simple
//! policy selection (plain kernels only), JIT options for PTX loading,
//! VRAM checks, and chunked launches to respect grid limits. Kernels operate
//! in FP32 with FP64 intermediates to match f64 scalar behavior.

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::supersmoother_3_pole::{
    SuperSmoother3PoleBatchRange, SuperSmoother3PoleParams,
};
use cust::context::{CacheConfig, Context};
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, GridSize};
use cust::memory::{
    mem_get_info, AsyncCopyDestination, CopyDestination, DeviceBuffer, LockedBuffer,
};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[inline(always)]
fn div_up(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

#[derive(Debug, thiserror::Error)]
pub enum CudaSuperSmoother3PoleError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("out of memory: required={required}B free={free}B headroom={headroom}B")]
    OutOfMemory { required: usize, free: usize, headroom: usize },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("device mismatch: buffer device={buf} current={current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("not implemented")] 
    NotImplemented,
}

// -------- Kernel policy + introspection (kept minimal; no tiled variants) --------

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
pub struct CudaSupersmoother3PolePolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaSupersmoother3PolePolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected {
    Plain { block_x: u32 },
}
#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}

pub struct CudaSupersmoother3Pole {
    module: Module,
    stream: Stream,
    _context: Arc<Context>,
    device_id: u32,
    policy: CudaSupersmoother3PolePolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaSupersmoother3Pole {
    pub fn new(device_id: usize) -> Result<Self, CudaSuperSmoother3PoleError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Context::new(device)?;
        let ctx = Arc::new(context);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/supersmoother_3_pole_kernel.ptx"));
        // Use context-targeted JIT; default to O4 (most optimized), unless overridden.
        let opt = match std::env::var("CUDA_JIT_OPT").ok().as_deref() {
            Some("O0") => OptLevel::O0,
            Some("O1") => OptLevel::O1,
            Some("O2") => OptLevel::O2,
            Some("O3") => OptLevel::O3,
            _ => OptLevel::O4,
        };
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(opt),
        ];
        let module = Module::from_ptx(ptx, jit_opts)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        Ok(Self {
            module,
            stream,
            _context: ctx,
            device_id: device_id as u32,
            policy: CudaSupersmoother3PolePolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn new_with_policy(
        device_id: usize,
        policy: CudaSupersmoother3PolePolicy,
    ) -> Result<Self, CudaSuperSmoother3PoleError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }
    pub fn set_policy(&mut self, policy: CudaSupersmoother3PolePolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaSupersmoother3PolePolicy {
        &self.policy
    }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch
    }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many
    }
    pub fn synchronize(&self) -> Result<(), CudaSuperSmoother3PoleError> {
        self.stream.synchronize()?;
        Ok(())
    }

    // Expose interop details for Python wrappers
    pub fn stream_handle(&self) -> usize {
        self.stream.as_inner() as usize
    }
    pub fn context_arc(&self) -> Arc<Context> {
        self._context.clone()
    }
    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        }
    }

    fn device_mem_info() -> Option<(usize, usize)> {
        mem_get_info().ok()
    }

    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> bool {
        if !Self::mem_check_enabled() {
            return true;
        }
        if let Some((free, _total)) = Self::device_mem_info() {
            required_bytes.saturating_add(headroom_bytes) <= free
        } else {
            true
        }
    }

    #[inline]
    fn ptr_device_id<T: cust::memory::DeviceCopy>(buf: &DeviceBuffer<T>) -> Result<u32, CudaSuperSmoother3PoleError> {
        unsafe {
            let mut dev_ordinal: i32 = -1;
            let ptr = buf.as_device_ptr().as_raw() as *mut c_void;
            use cust::sys as cu;
            let res = cu::cuPointerGetAttribute(
                &mut dev_ordinal as *mut _ as *mut c_void,
                cu::CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                ptr,
            );
            if res == cu::CUresult::CUDA_SUCCESS && dev_ordinal >= 0 {
                Ok(dev_ordinal as u32)
            } else {
                // Fallback to current context device if attribute not available
                let mut cur_dev: i32 = 0;
                let _ = cu::cuCtxGetDevice(&mut cur_dev as *mut _);
                Ok(cur_dev as u32)
            }
        }
    }

    fn expand_periods(range: &SuperSmoother3PoleBatchRange) -> Vec<SuperSmoother3PoleParams> {
        let (start, end, step) = range.period;
        let periods: Vec<usize> = if step == 0 || start == end {
            vec![start]
        } else if start < end {
            (start..=end).step_by(step).collect::<Vec<_>>()
        } else {
            // Reversed bounds: generate descending sequence
            let mut v = Vec::new();
            let mut cur = start;
            while cur >= end {
                v.push(cur);
                if let Some(next) = cur.checked_sub(step) {
                    if next == cur { break; }
                    cur = next;
                } else {
                    break;
                }
            }
            v
        };
        periods
            .into_iter()
            .map(|p| SuperSmoother3PoleParams { period: Some(p) })
            .collect()
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &SuperSmoother3PoleBatchRange,
    ) -> Result<(Vec<SuperSmoother3PoleParams>, usize, usize), CudaSuperSmoother3PoleError> {
        if data_f32.is_empty() {
            return Err(CudaSuperSmoother3PoleError::InvalidInput(
                "price data is empty".into(),
            ));
        }

        let first_valid = data_f32.iter().position(|v| !v.is_nan()).ok_or_else(|| {
            CudaSuperSmoother3PoleError::InvalidInput("all values are NaN".into())
        })?;

        let combos = Self::expand_periods(sweep);
        if combos.is_empty() {
            return Err(CudaSuperSmoother3PoleError::InvalidInput(
                "no period combinations".into(),
            ));
        }

        let series_len = data_f32.len();
        for prm in &combos {
            let period = prm.period.unwrap_or(0);
            if period == 0 {
                return Err(CudaSuperSmoother3PoleError::InvalidInput(
                    "period must be >= 1".into(),
                ));
            }
            if period > series_len {
                return Err(CudaSuperSmoother3PoleError::InvalidInput(format!(
                    "period {} exceeds series length {}",
                    period, series_len
                )));
            }
            let valid = series_len - first_valid;
            if valid < period {
                return Err(CudaSuperSmoother3PoleError::InvalidInput(format!(
                    "not enough valid data: need >= {}, valid = {}",
                    period, valid
                )));
            }
        }

        Ok((combos, first_valid, series_len))
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSuperSmoother3PoleError> {
        if series_len == 0 || n_combos == 0 {
            return Err(CudaSuperSmoother3PoleError::InvalidInput(
                "series_len and n_combos must be > 0".into(),
            ));
        }

        // Kernel + prefer L1
        let mut func = self
            .module
            .get_function("supersmoother_3_pole_batch_f32")
            .map_err(|_| CudaSuperSmoother3PoleError::MissingKernelSymbol {
                name: "supersmoother_3_pole_batch_f32",
            })?;
        let _ = func.set_cache_config(CacheConfig::PreferL1);

        // Occupancy-based block size for Auto
        let block_x: u32 = match self.policy.batch {
            BatchKernelPolicy::Auto => {
                match func.suggested_launch_configuration(0, (0, 0, 0).into()) {
                    Ok((_min_grid, block)) => block.max(64),
                    Err(_) => 256,
                }
            }
            BatchKernelPolicy::Plain { block_x } => block_x.max(1),
        };
        unsafe {
            (*(self as *const _ as *mut CudaSupersmoother3Pole)).last_batch =
                Some(BatchKernelSelected::Plain { block_x });
        }

        // True device max grid.x
        let device = Device::get_device(self.device_id)?;
        let max_grid_x = device.get_attribute(DeviceAttribute::MaxGridDimX)? as usize;
        let max_block_x = device.get_attribute(DeviceAttribute::MaxBlockDimX)? as u32;
        if block_x > max_block_x {
            return Err(CudaSuperSmoother3PoleError::LaunchConfigTooLarge {
                gx: 1,
                gy: 1,
                gz: 1,
                bx: block_x,
                by: 1,
                bz: 1,
            });
        }

        let tpb = block_x as usize;
        let chunk_capacity = max_grid_x.saturating_mul(tpb);

        let mut launched = 0usize;
        while launched < n_combos {
            let launch_elems = (n_combos - launched).min(chunk_capacity);
            let blocks = (launch_elems + tpb - 1) / tpb;

            let grid: GridSize = (blocks as u32, 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();

            unsafe {
                let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                let mut periods_ptr = d_periods.as_device_ptr().add(launched).as_raw();
                let mut series_len_i = series_len as i32;
                let mut n_elems_i = launch_elems as i32;
                let mut first_valid_i = first_valid as i32;
                let mut out_ptr = d_out.as_device_ptr().add(launched * series_len).as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut prices_ptr as *mut _ as *mut c_void,
                    &mut periods_ptr as *mut _ as *mut c_void,
                    &mut series_len_i as *mut _ as *mut c_void,
                    &mut n_elems_i as *mut _ as *mut c_void,
                    &mut first_valid_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream.launch(&func, grid, block, 0, args)?;
            }

            launched += launch_elems;
        }

        self.maybe_log_batch_debug();
        Ok(())
    }

    fn run_batch_kernel(
        &self,
        data_f32: &[f32],
        combos: &[SuperSmoother3PoleParams],
        first_valid: usize,
        series_len: usize,
    ) -> Result<DeviceArrayF32, CudaSuperSmoother3PoleError> {
        let n_combos = combos.len();
        let total_elems = n_combos
            .checked_mul(series_len)
            .ok_or_else(|| CudaSuperSmoother3PoleError::InvalidInput("rows * cols overflow".into()))?;
        let prices_bytes = series_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaSuperSmoother3PoleError::InvalidInput("series_len * sizeof overflow".into()))?;
        let periods_bytes = n_combos
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaSuperSmoother3PoleError::InvalidInput("n_combos * sizeof overflow".into()))?;
        let out_bytes = total_elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaSuperSmoother3PoleError::InvalidInput("total_elems * sizeof overflow".into()))?;
        let required = prices_bytes + periods_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024; // 64MB cushion
        if !Self::will_fit(required, headroom) {
            let free = Self::device_mem_info().map(|(f, _)| f).unwrap_or(0);
            return Err(CudaSuperSmoother3PoleError::OutOfMemory {
                required,
                free,
                headroom,
            });
        }

        let d_prices = unsafe { DeviceBuffer::from_slice_async(data_f32, &self.stream) }?;
        let periods_i32: Vec<i32> = combos.iter().map(|p| p.period.unwrap() as i32).collect();
        let d_periods = unsafe { DeviceBuffer::from_slice_async(&periods_i32, &self.stream) }?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total_elems) }?;

        self.launch_batch_kernel(
            &d_prices,
            &d_periods,
            series_len,
            n_combos,
            first_valid,
            &mut d_out,
        )?;

        self.stream.synchronize()?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: series_len,
        })
    }

    pub fn supersmoother_3_pole_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &SuperSmoother3PoleBatchRange,
    ) -> Result<DeviceArrayF32, CudaSuperSmoother3PoleError> {
        let (combos, first_valid, series_len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        self.run_batch_kernel(data_f32, &combos, first_valid, series_len)
    }

    pub fn supersmoother_3_pole_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &SuperSmoother3PoleBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<SuperSmoother3PoleParams>), CudaSuperSmoother3PoleError> {
        let (combos, first_valid, series_len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let expected = combos.len() * series_len;
        if out.len() != expected {
            return Err(CudaSuperSmoother3PoleError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out.len(),
                expected
            )));
        }

        let arr = self.run_batch_kernel(data_f32, &combos, first_valid, series_len)?;
        let total = expected;
        let mut pinned = unsafe { LockedBuffer::<f32>::uninitialized(total) }?;
        unsafe {
            arr.buf.async_copy_to(pinned.as_mut_slice(), &self.stream)?;
        }
        self.stream.synchronize()?;
        out.copy_from_slice(pinned.as_slice());
        Ok((arr.rows, arr.cols, combos))
    }

    pub fn supersmoother_3_pole_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSuperSmoother3PoleError> {
        if series_len == 0 || n_combos == 0 {
            return Err(CudaSuperSmoother3PoleError::InvalidInput(
                "series_len and n_combos must be > 0".into(),
            ));
        }
        // Ensure all buffers belong to the same device as this wrapper
        let dev_prices = Self::ptr_device_id(d_prices)?;
        let dev_periods = Self::ptr_device_id(d_periods)?;
        let dev_out = Self::ptr_device_id(d_out)?;
        if dev_prices != self.device_id {
            return Err(CudaSuperSmoother3PoleError::DeviceMismatch { buf: dev_prices, current: self.device_id });
        }
        if dev_periods != self.device_id {
            return Err(CudaSuperSmoother3PoleError::DeviceMismatch { buf: dev_periods, current: self.device_id });
        }
        if dev_out != self.device_id {
            return Err(CudaSuperSmoother3PoleError::DeviceMismatch { buf: dev_out, current: self.device_id });
        }
        self.launch_batch_kernel(
            d_prices,
            d_periods,
            series_len,
            n_combos,
            first_valid,
            d_out,
        )
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &SuperSmoother3PoleParams,
    ) -> Result<(Vec<i32>, usize), CudaSuperSmoother3PoleError> {
        if cols == 0 || rows == 0 {
            return Err(CudaSuperSmoother3PoleError::InvalidInput(
                "num_series or series_len is zero".into(),
            ));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaSuperSmoother3PoleError::InvalidInput(format!(
                "data length {} != cols*rows {}",
                data_tm_f32.len(),
                cols * rows
            )));
        }

        let period = params.period.unwrap_or(0);
        if period == 0 {
            return Err(CudaSuperSmoother3PoleError::InvalidInput(
                "period must be >= 1".into(),
            ));
        }

        let mut first_valids = vec![0i32; cols];
        for series in 0..cols {
            let mut fv = None;
            for t in 0..rows {
                let idx = t * cols + series;
                if !data_tm_f32[idx].is_nan() {
                    fv = Some(t as i32);
                    break;
                }
            }
            let found = fv.ok_or_else(|| {
                CudaSuperSmoother3PoleError::InvalidInput(format!(
                    "series {} contains only NaNs",
                    series
                ))
            })?;
            if (rows as i32 - found) < period as i32 {
                return Err(CudaSuperSmoother3PoleError::InvalidInput(format!(
                    "series {} lacks data: need >= {}, valid = {}",
                    series,
                    period,
                    rows as i32 - found
                )));
            }
            first_valids[series] = found;
        }

        Ok((first_valids, period))
    }

    fn launch_many_series_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        period: usize,
        cols: usize,
        rows: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSuperSmoother3PoleError> {
        if period == 0 {
            return Err(CudaSuperSmoother3PoleError::InvalidInput(
                "period must be >= 1".into(),
            ));
        }

        let mut func = self
            .module
            .get_function("supersmoother_3_pole_many_series_one_param_time_major_f32")
            .map_err(|_| CudaSuperSmoother3PoleError::MissingKernelSymbol {
                name: "supersmoother_3_pole_many_series_one_param_time_major_f32",
            })?;
        let _ = func.set_cache_config(CacheConfig::PreferL1);

        let block_x: u32 = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => {
                match func.suggested_launch_configuration(0, (0, 0, 0).into()) {
                    Ok((_min_grid, block)) => block.max(64),
                    Err(_) => 256,
                }
            }
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(1),
        };
        unsafe {
            (*(self as *const _ as *mut CudaSupersmoother3Pole)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }

        let device = Device::get_device(self.device_id)?;
        let max_grid_x = device.get_attribute(DeviceAttribute::MaxGridDimX)? as usize;
        let max_block_x = device.get_attribute(DeviceAttribute::MaxBlockDimX)? as u32;
        if block_x > max_block_x {
            return Err(CudaSuperSmoother3PoleError::LaunchConfigTooLarge {
                gx: 1,
                gy: 1,
                gz: 1,
                bx: block_x,
                by: 1,
                bz: 1,
            });
        }

        let tpb = block_x as usize;
        let chunk_capacity = max_grid_x.saturating_mul(tpb);

        let mut launched = 0usize;
        while launched < cols {
            let launch_elems = (cols - launched).min(chunk_capacity);
            let blocks = (launch_elems + tpb - 1) / tpb;

            let grid: GridSize = (blocks as u32, 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();

            unsafe {
                let mut prices_ptr = d_prices.as_device_ptr().add(launched).as_raw();
                let mut period_i = period as i32;
                let mut cols_i = cols as i32; // full stride stays constant
                let mut rows_i = rows as i32;
                let mut first_ptr = d_first_valids.as_device_ptr().add(launched).as_raw();
                let mut out_ptr = d_out.as_device_ptr().add(launched).as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut prices_ptr as *mut _ as *mut c_void,
                    &mut period_i as *mut _ as *mut c_void,
                    &mut cols_i as *mut _ as *mut c_void,
                    &mut rows_i as *mut _ as *mut c_void,
                    &mut first_ptr as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream.launch(&func, grid, block, 0, args)?;
            }

            launched += launch_elems;
        }

        self.maybe_log_many_debug();
        Ok(())
    }

    fn run_many_series_kernel(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        first_valids: &[i32],
        period: usize,
    ) -> Result<DeviceArrayF32, CudaSuperSmoother3PoleError> {
        let total_elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaSuperSmoother3PoleError::InvalidInput("rows * cols overflow".into()))?;
        let prices_bytes = total_elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaSuperSmoother3PoleError::InvalidInput("total_elems * sizeof overflow".into()))?;
        let first_valid_bytes = cols
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaSuperSmoother3PoleError::InvalidInput("cols * sizeof overflow".into()))?;
        let out_bytes = total_elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaSuperSmoother3PoleError::InvalidInput("total_elems * sizeof overflow".into()))?;
        let required = prices_bytes + first_valid_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            let free = Self::device_mem_info().map(|(f, _)| f).unwrap_or(0);
            return Err(CudaSuperSmoother3PoleError::OutOfMemory {
                required,
                free,
                headroom,
            });
        }

        let d_prices = unsafe { DeviceBuffer::from_slice_async(data_tm_f32, &self.stream) }?;
        let d_first_valids = unsafe { DeviceBuffer::from_slice_async(first_valids, &self.stream) }?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total_elems) }?;

        self.launch_many_series_kernel(&d_prices, period, cols, rows, &d_first_valids, &mut d_out)?;

        self.stream.synchronize()?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    pub fn supersmoother_3_pole_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &SuperSmoother3PoleParams,
    ) -> Result<DeviceArrayF32, CudaSuperSmoother3PoleError> {
        let (first_valids, period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        self.run_many_series_kernel(data_tm_f32, cols, rows, &first_valids, period)
    }

    pub fn supersmoother_3_pole_many_series_one_param_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &SuperSmoother3PoleParams,
        out: &mut [f32],
    ) -> Result<(), CudaSuperSmoother3PoleError> {
        if out.len() != cols * rows {
            return Err(CudaSuperSmoother3PoleError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out.len(),
                cols * rows
            )));
        }
        let (first_valids, period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        let arr = self.run_many_series_kernel(data_tm_f32, cols, rows, &first_valids, period)?;
        let total = cols * rows;
        let mut pinned = unsafe { LockedBuffer::<f32>::uninitialized(total) }?;
        unsafe {
            arr.buf.async_copy_to(pinned.as_mut_slice(), &self.stream)?;
        }
        self.stream.synchronize()?;
        out.copy_from_slice(pinned.as_slice());
        Ok(())
    }

    pub fn supersmoother_3_pole_many_series_one_param_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        period: usize,
        cols: usize,
        rows: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSuperSmoother3PoleError> {
        if cols == 0 || rows == 0 {
            return Err(CudaSuperSmoother3PoleError::InvalidInput(
                "num_series or series_len is zero".into(),
            ));
        }
        self.launch_many_series_kernel(d_prices, period, cols, rows, d_first_valids, d_out)
    }

    #[inline]
    fn maybe_log_batch_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                let per = std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] SS3P batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaSupersmoother3Pole)).debug_batch_logged = true;
                }
            }
        }
    }

    #[inline]
    fn maybe_log_many_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_many_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                let per = std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] SS3P many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaSupersmoother3Pole)).debug_many_logged = true;
                }
            }
        }
    }
}

// -------------------- Python: CUDA Array Interface v3 + DLPack ----------------------
#[cfg(all(feature = "python", feature = "cuda"))]
use pyo3::prelude::*;
#[cfg(all(feature = "python", feature = "cuda"))]
use pyo3::types::PyDict;

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(module = "ta_indicators.cuda", name = "DeviceArrayF32")]
pub struct DeviceArrayF32Py {
    pub inner: DeviceArrayF32,
    stream_handle: usize,
    _ctx_guard: Arc<Context>,
    _device_id: u32,
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pymethods]
impl DeviceArrayF32Py {
    #[getter]
    fn __cuda_array_interface__<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let itemsize = std::mem::size_of::<f32>();
        let d = PyDict::new(py);
        d.set_item("shape", (self.inner.rows, self.inner.cols))?;
        d.set_item("typestr", "<f4")?;
        d.set_item("strides", (self.inner.cols * itemsize, itemsize))?;
        let size = self.inner.rows.saturating_mul(self.inner.cols);
        let ptr_val: usize = if size == 0 { 0 } else { self.inner.buf.as_device_ptr().as_raw() as usize };
        d.set_item("data", (ptr_val, false))?;
        d.set_item("version", 3)?;
        if self.stream_handle != 0 {
            d.set_item("stream", self.stream_handle)?;
        }
        Ok(d.into())
    }
    fn __dlpack_device__(&self) -> (i32, i32) { (2, self._device_id as i32) }

    #[pyo3(signature = (stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__<'py>(
        &mut self,
        py: Python<'py>,
        stream: Option<PyObject>,
        max_version: Option<(i32, i32)>,
        _dl_device: Option<(i32, i32)>,
        _copy: Option<bool>,
    ) -> PyResult<PyObject> {
        use pyo3::ffi as pyffi;
        use std::os::raw::c_char;
        use std::ptr::null_mut;

        // Synchronize producer stream so no pending work remains.
        unsafe { let _ = cust::sys::cuStreamSynchronize(self.stream_handle as *mut _); }

        #[repr(C)]
        struct DLDataType { code: u8, bits: u8, lanes: u16 }
        #[repr(C)]
        struct DLDevice { device_type: i32, device_id: i32 }
        #[repr(C)]
        struct DLTensor {
            data: *mut std::ffi::c_void,
            device: DLDevice,
            ndim: i32,
            dtype: DLDataType,
            shape: *mut i64,
            strides: *mut i64,
            byte_offset: u64,
        }
        #[repr(C)]
        struct DLPackVersion { major: u32, minor: u32 }
        #[repr(C)]
        struct DLManagedTensorVersioned {
            version: DLPackVersion,
            manager_ctx: *mut std::ffi::c_void,
            deleter: Option<unsafe extern "C" fn(*mut DLManagedTensorVersioned)>,
            flags: u64,
            dl_tensor: DLTensor,
        }
        #[repr(C)]
        struct DLManagedTensor {
            dl_tensor: DLTensor,
            manager_ctx: *mut std::ffi::c_void,
            deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
        }

        struct Manager {
            ctx: cust::sys::CUcontext,
            device_id: i32,
            _buf: DeviceBuffer<f32>,
            _shape: Box<[i64; 2]>,
            _strides: Box<[i64; 2]>,
        }

        unsafe extern "C" fn deleter_v1(p: *mut DLManagedTensorVersioned) {
            if p.is_null() { return; }
            let mgr = (*p).manager_ctx as *mut Manager;
            if !mgr.is_null() {
                let mut old: cust::sys::CUcontext = std::ptr::null_mut();
                let _ = cust::sys::cuCtxPushCurrent((*mgr).ctx);
                let _boxed: Box<Manager> = Box::from_raw(mgr);
                let _ = cust::sys::cuCtxPopCurrent(&mut old as *mut _);
                let _ = cust::sys::cuDevicePrimaryCtxRelease((*p).dl_tensor.device.device_id as i32);
            }
            let _ = Box::from_raw(p);
        }
        unsafe extern "C" fn deleter_legacy(p: *mut DLManagedTensor) {
            if p.is_null() { return; }
            let mgr = (*p).manager_ctx as *mut Manager;
            if !mgr.is_null() {
                let mut old: cust::sys::CUcontext = std::ptr::null_mut();
                let dev = (*mgr).device_id;
                let _ = cust::sys::cuCtxPushCurrent((*mgr).ctx);
                let _boxed: Box<Manager> = Box::from_raw(mgr);
                let _ = cust::sys::cuCtxPopCurrent(&mut old as *mut _);
                let _ = cust::sys::cuDevicePrimaryCtxRelease(dev);
            }
            let _ = Box::from_raw(p);
        }

        unsafe extern "C" fn capsule_dtor(capsule: *mut pyo3::ffi::PyObject) {
            let vname = std::ffi::CString::new("dltensor_versioned").unwrap();
            let legacy = std::ffi::CString::new("dltensor").unwrap();
            let ptr = pyo3::ffi::PyCapsule_GetPointer(capsule, vname.as_ptr());
            if !ptr.is_null() {
                let mt = ptr as *mut DLManagedTensorVersioned;
                if let Some(del) = (*mt).deleter { del(mt) }
                return;
            }
            let ptr2 = pyo3::ffi::PyCapsule_GetPointer(capsule, legacy.as_ptr());
            if !ptr2.is_null() {
                let mt = ptr2 as *mut DLManagedTensor;
                if let Some(del) = (*mt).deleter { del(mt) }
            }
        }

        let rows = self.inner.rows as i64;
        let cols = self.inner.cols as i64;
        let size = (rows as usize).saturating_mul(cols as usize);
        let data_ptr = if size == 0 { null_mut() } else { self.inner.buf.as_device_ptr().as_raw() as *mut _ };
        let mut shape = Box::new([rows, cols]);
        let mut strides = Box::new([cols, 1]);

        // Negotiate version
        let use_versioned = max_version.map(|(maj, _)| maj >= 1).unwrap_or(false);

        // Move buffer ownership into capsule manager
        let dummy = DeviceBuffer::from_slice(&[]).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let inner = std::mem::replace(&mut self.inner, DeviceArrayF32 { buf: dummy, rows: 0, cols: 0 });

        // Retain primary context for allocation device
        let (dev_type, dev_id) = self.__dlpack_device__();
        debug_assert_eq!(dev_type, 2);
        let mut retained: cust::sys::CUcontext = std::ptr::null_mut();
        unsafe { let _ = cust::sys::cuDevicePrimaryCtxRetain(&mut retained as *mut _, dev_id); }

        let mgr = Box::new(Manager { ctx: retained, device_id: dev_id, _buf: inner.buf, _shape: shape, _strides: strides });
        let mgr_ptr = Box::into_raw(mgr) as *mut std::ffi::c_void;

        unsafe {
            if use_versioned {
                let mt = Box::new(DLManagedTensorVersioned {
                    version: DLPackVersion { major: 1, minor: 2 },
                    manager_ctx: mgr_ptr,
                    deleter: Some(deleter_v1),
                    flags: 0,
                    dl_tensor: DLTensor {
                        data: data_ptr,
                        device: DLDevice { device_type: 2, device_id: dev_id },
                        ndim: 2,
                        dtype: DLDataType { code: 2, bits: 32, lanes: 1 },
                        shape: (*((mgr_ptr as *mut Manager)))._shape.as_mut_ptr(),
                        strides: (*((mgr_ptr as *mut Manager)))._strides.as_mut_ptr(),
                        byte_offset: 0,
                    },
                });
                let ptr = Box::into_raw(mt) as *mut std::ffi::c_void;
                let name = std::ffi::CString::new("dltensor_versioned").unwrap();
                let cap = pyffi::PyCapsule_New(ptr, name.as_ptr(), Some(capsule_dtor));
                if cap.is_null() {
                    let _ = Box::from_raw(ptr as *mut DLManagedTensorVersioned);
                    return Err(pyo3::exceptions::PyRuntimeError::new_err("failed to create DLPack v1.x capsule"));
                }
                Ok(pyo3::PyObject::from_owned_ptr(py, cap))
            } else {
                let mt = Box::new(DLManagedTensor {
                    manager_ctx: mgr_ptr,
                    deleter: Some(deleter_legacy),
                    dl_tensor: DLTensor {
                        data: data_ptr,
                        device: DLDevice { device_type: 2, device_id: dev_id },
                        ndim: 2,
                        dtype: DLDataType { code: 2, bits: 32, lanes: 1 },
                        shape: (*((mgr_ptr as *mut Manager)))._shape.as_mut_ptr(),
                        strides: (*((mgr_ptr as *mut Manager)))._strides.as_mut_ptr(),
                        byte_offset: 0,
                    },
                });
                let ptr = Box::into_raw(mt) as *mut std::ffi::c_void;
                let name = std::ffi::CString::new("dltensor").unwrap();
                let cap = pyffi::PyCapsule_New(ptr, name.as_ptr(), Some(capsule_dtor));
                if cap.is_null() {
                    let _ = Box::from_raw(ptr as *mut DLManagedTensor);
                    return Err(pyo3::exceptions::PyRuntimeError::new_err("failed to create DLPack capsule"));
                }
                Ok(pyo3::PyObject::from_owned_ptr(py, cap))
            }
        }
    }
}

#[cfg(all(feature = "python", feature = "cuda"))]
impl DeviceArrayF32Py {
    pub fn new_from_rust(inner: DeviceArrayF32, stream_handle: usize, ctx_guard: Arc<Context>, device_id: u32) -> Self {
        Self { inner, stream_handle, _ctx_guard: ctx_guard, _device_id: device_id }
    }
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::define_ma_period_benches;

    define_ma_period_benches!(
        supersmoother_3_pole_benches,
        CudaSupersmoother3Pole,
        crate::indicators::moving_averages::supersmoother_3_pole::SuperSmoother3PoleBatchRange,
        crate::indicators::moving_averages::supersmoother_3_pole::SuperSmoother3PoleParams,
        supersmoother_3_pole_batch_dev,
        supersmoother_3_pole_many_series_one_param_time_major_dev,
        crate::indicators::moving_averages::supersmoother_3_pole::SuperSmoother3PoleBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1)
        },
        crate::indicators::moving_averages::supersmoother_3_pole::SuperSmoother3PoleParams {
            period: Some(64)
        },
        "supersmoother_3_pole",
        "supersmoother_3_pole"
    );
    pub use supersmoother_3_pole_benches::bench_profiles;
}
