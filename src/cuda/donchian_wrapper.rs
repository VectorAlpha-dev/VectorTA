//! CUDA wrapper for Donchian Channels (upper, middle, lower).
//!
//! Parity with ALMA-style wrappers:
//! - PTX load via include_str!(concat!(env!("OUT_DIR"), "/donchian_kernel.ptx"))
//! - Non-blocking stream; DetermineTargetFromContext + O2 with fallbacks
//! - VRAM checks with ~64MB headroom; simple grid policies
//! - Public device entry points:
//!     - `donchian_batch_dev(&[f32], &[f32], &DonchianBatchRange)` -> (DeviceArrayF32Triplet, Vec<DonchianParams>)
//!     - `donchian_many_series_one_param_time_major_dev(&[f32], cols, rows, &DonchianParams)` -> DeviceArrayF32Triplet

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::cuda::wto_wrapper::DeviceArrayF32Triplet;
use crate::indicators::donchian::{DonchianBatchRange, DonchianParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaDonchianError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaDonchianError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaDonchianError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaDonchianError::InvalidInput(s) => write!(f, "Invalid input: {}", s),
        }
    }
}
impl std::error::Error for CudaDonchianError {}

#[derive(Clone, Copy, Debug, Default)]
pub enum BatchKernelPolicy { #[default] Auto, Plain { block_x: u32 } }

#[derive(Clone, Copy, Debug, Default)]
pub enum ManySeriesKernelPolicy { #[default] Auto, OneD { block_x: u32 } }

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaDonchianPolicy { pub batch: BatchKernelPolicy, pub many_series: ManySeriesKernelPolicy }

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected { Rmq { build_bx: u32, query_bx: u32 } }
#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected { OneD { block_x: u32 } }

pub struct CudaDonchian {
    module: Module,
    stream: Stream,
    _context: Context,
    policy: CudaDonchianPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaDonchian {
    pub fn new(device_id: usize) -> Result<Self, CudaDonchianError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaDonchianError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaDonchianError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaDonchianError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/donchian_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, jit_opts)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))
            .map_err(|e| CudaDonchianError::Cuda(e.to_string()))?;

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaDonchianError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            policy: CudaDonchianPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn set_policy(&mut self, policy: CudaDonchianPolicy) { self.policy = policy; }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> { self.last_batch }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> { self.last_many }

    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") { Ok(v) => v != "0" && v.to_lowercase() != "false", Err(_) => true }
    }
    #[inline]
    fn device_mem_info() -> Option<(usize, usize)> { mem_get_info().ok() }
    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> bool {
        if !Self::mem_check_enabled() { return true; }
        if let Some((free, _)) = Self::device_mem_info() {
            required_bytes.saturating_add(headroom_bytes) <= free
        } else { true }
    }
    #[inline]
    fn maybe_log_batch_debug(&self) {
        static ONCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if self.debug_batch_logged { return; }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                if !ONCE.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    eprintln!("[DEBUG] donchian batch selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut CudaDonchian)).debug_batch_logged = true; }
            }
        }
    }
    #[inline]
    fn maybe_log_many_debug(&self) {
        static ONCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if self.debug_many_logged { return; }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                if !ONCE.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    eprintln!("[DEBUG] donchian many-series selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut CudaDonchian)).debug_many_logged = true; }
            }
        }
    }

    fn expand_grid(range: &DonchianBatchRange) -> Vec<DonchianParams> {
        fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
            if step == 0 || start == end { return vec![start]; }
            (start..=end).step_by(step).collect()
        }
        let periods = axis_usize(range.period);
        periods.into_iter().map(|p| DonchianParams { period: Some(p) }).collect()
    }

    fn prepare_batch_inputs(
        high: &[f32], low: &[f32], sweep: &DonchianBatchRange,
    ) -> Result<(Vec<DonchianParams>, usize, usize), CudaDonchianError> {
        if high.len() != low.len() { return Err(CudaDonchianError::InvalidInput("length mismatch".into())); }
        if high.is_empty() { return Err(CudaDonchianError::InvalidInput("empty input".into())); }
        let len = high.len();
        let first_valid = high.iter().zip(low.iter()).position(|(h,l)| !h.is_nan() && !l.is_nan())
            .ok_or_else(|| CudaDonchianError::InvalidInput("all values are NaN".into()))?;
        let combos = Self::expand_grid(sweep);
        if combos.is_empty() { return Err(CudaDonchianError::InvalidInput("no parameter combinations".into())); }
        for c in &combos {
            let p = c.period.unwrap_or(0);
            if p == 0 { return Err(CudaDonchianError::InvalidInput("period must be > 0".into())); }
            if p > len { return Err(CudaDonchianError::InvalidInput("period exceeds length".into())); }
            if len - first_valid < p { return Err(CudaDonchianError::InvalidInput("not enough valid data".into())); }
        }
        Ok((combos, first_valid, len))
    }

    pub fn donchian_batch_dev(
        &self,
        high_f32: &[f32],
        low_f32: &[f32],
        sweep: &DonchianBatchRange,
    ) -> Result<(DeviceArrayF32Triplet, Vec<DonchianParams>), CudaDonchianError> {
        let (combos, first_valid, len) = Self::prepare_batch_inputs(high_f32, low_f32, sweep)?;
        let max_period = combos.iter().map(|c| c.period.unwrap()).max().unwrap_or(1);
        let levels = rmq_levels_for_max_period(max_period);

        // VRAM estimate: inputs + periods + outputs + RMQ scratch
        let bytes_in = 2 * len * std::mem::size_of::<f32>();
        let bytes_periods = combos.len() * std::mem::size_of::<i32>();
        let bytes_out = 3 * combos.len() * len * std::mem::size_of::<f32>();
        let bytes_rmq = bytes_rmq_tables(len, levels);
        let required = bytes_in + bytes_periods + bytes_out + bytes_rmq;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaDonchianError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                required as f64 / (1024.0 * 1024.0)
            )));
        }

        // Device buffers (common)
        let d_high = unsafe { DeviceBuffer::from_slice_async(high_f32, &self.stream) }
            .map_err(|e| CudaDonchianError::Cuda(e.to_string()))?;
        let d_low = unsafe { DeviceBuffer::from_slice_async(low_f32, &self.stream) }
            .map_err(|e| CudaDonchianError::Cuda(e.to_string()))?;
        let periods_i32: Vec<i32> = combos.iter().map(|c| c.period.unwrap() as i32).collect();
        let d_periods = unsafe { DeviceBuffer::from_slice_async(&periods_i32, &self.stream) }
            .map_err(|e| CudaDonchianError::Cuda(e.to_string()))?;

        let mut d_upper: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(combos.len() * len, &self.stream) }
            .map_err(|e| CudaDonchianError::Cuda(e.to_string()))?;
        let mut d_middle: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(combos.len() * len, &self.stream) }
            .map_err(|e| CudaDonchianError::Cuda(e.to_string()))?;
        let mut d_lower: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(combos.len() * len, &self.stream) }
            .map_err(|e| CudaDonchianError::Cuda(e.to_string()))?;

        // RMQ is always used for batch path
        let stride = len;

        // scratch buffers sized only to required levels
        let mut d_st_high: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(levels * stride, &self.stream) }
            .map_err(|e| CudaDonchianError::Cuda(e.to_string()))?;
        let mut d_st_low: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(levels * stride, &self.stream) }
            .map_err(|e| CudaDonchianError::Cuda(e.to_string()))?;
        let mut d_st_nan: DeviceBuffer<u8> = unsafe { DeviceBuffer::uninitialized_async(levels * stride, &self.stream) }
            .map_err(|e| CudaDonchianError::Cuda(e.to_string()))?;

        let init_lvl0_f32 = self.module.get_function("rmq_init_level0_f32")
            .map_err(|e| CudaDonchianError::Cuda(e.to_string()))?;
        let init_nan_u8 = self.module.get_function("rmq_init_nan_mask_u8")
            .map_err(|e| CudaDonchianError::Cuda(e.to_string()))?;
        let build_max = self.module.get_function("rmq_build_level_max_f32")
            .map_err(|e| CudaDonchianError::Cuda(e.to_string()))?;
        let build_min = self.module.get_function("rmq_build_level_min_f32")
            .map_err(|e| CudaDonchianError::Cuda(e.to_string()))?;
        let build_or  = self.module.get_function("rmq_build_level_or_u8")
            .map_err(|e| CudaDonchianError::Cuda(e.to_string()))?;
        let func_query = self.module.get_function("donchian_batch_from_rmq_f32")
            .map_err(|e| CudaDonchianError::Cuda(e.to_string()))?;

        let build_bx: u32 = 256;
        let query_bx: u32 = match self.policy.batch { BatchKernelPolicy::Auto => 256, BatchKernelPolicy::Plain { block_x } => block_x.max(64) };
        let build_grid_x: u32 = ((len as u32) + build_bx - 1) / build_bx;
        let build_grid: GridSize = (build_grid_x.max(1), 1, 1).into();
        let build_block: BlockSize = (build_bx, 1, 1).into();
        let query_grid_x: u32 = ((combos.len() as u32) + query_bx - 1) / query_bx;
        let query_grid: GridSize = (query_grid_x.max(1), 1, 1).into();
        let query_block: BlockSize = (query_bx, 1, 1).into();
        unsafe { (*(self as *const _ as *mut CudaDonchian)).last_batch = Some(BatchKernelSelected::Rmq { build_bx, query_bx }); }

        unsafe {
            // level 0 copies
            let mut high_in = d_high.as_device_ptr().as_raw();
            let mut low_in  = d_low.as_device_ptr().as_raw();
            let mut out_hi0 = as_raw_offset(&d_st_high, 0);
            let mut out_lo0 = as_raw_offset(&d_st_low, 0);
            let mut N_i = len as i32;
            let mut first_i = first_valid as i32;
            let mut mask0 = as_raw_offset(&d_st_nan, 0);

            let mut args_hi0: &mut [*mut c_void] = &mut [
                &mut high_in as *mut _ as *mut c_void,
                &mut out_hi0 as *mut _ as *mut c_void,
                &mut N_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&init_lvl0_f32, build_grid, build_block, 0, &mut args_hi0)
                .map_err(|e| CudaDonchianError::Cuda(e.to_string()))?;

            let mut args_lo0: &mut [*mut c_void] = &mut [
                &mut low_in as *mut _ as *mut c_void,
                &mut out_lo0 as *mut _ as *mut c_void,
                &mut N_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&init_lvl0_f32, build_grid, build_block, 0, &mut args_lo0)
                .map_err(|e| CudaDonchianError::Cuda(e.to_string()))?;

            let mut args_nm0: &mut [*mut c_void] = &mut [
                &mut high_in as *mut _ as *mut c_void,
                &mut low_in as *mut _ as *mut c_void,
                &mut N_i as *mut _ as *mut c_void,
                &mut first_i as *mut _ as *mut c_void,
                &mut mask0 as *mut _ as *mut c_void,
            ];
            self.stream.launch(&init_nan_u8, build_grid, build_block, 0, &mut args_nm0)
                .map_err(|e| CudaDonchianError::Cuda(e.to_string()))?;
        }

        // Build higher levels only up to `levels`
        for k in 1..levels {
            let offset = 1 << (k - 1);
            unsafe {
                let mut N_i = len as i32;
                let mut off_i = offset as i32;
                let prev_elems = (k - 1) * stride;
                let curr_elems = k * stride;

                // MAX
                let mut prev = as_raw_offset(&d_st_high, prev_elems);
                let mut curr = as_raw_offset(&d_st_high, curr_elems);
                let mut args: &mut [*mut c_void] = &mut [
                    &mut prev as *mut _ as *mut c_void,
                    &mut curr as *mut _ as *mut c_void,
                    &mut N_i as *mut _ as *mut c_void,
                    &mut off_i as *mut _ as *mut c_void,
                ];
                self.stream.launch(&build_max, build_grid, build_block, 0, &mut args)
                    .map_err(|e| CudaDonchianError::Cuda(e.to_string()))?;

                // MIN
                prev = as_raw_offset(&d_st_low, prev_elems);
                curr = as_raw_offset(&d_st_low, curr_elems);
                let mut args2: &mut [*mut c_void] = &mut [
                    &mut prev as *mut _ as *mut c_void,
                    &mut curr as *mut _ as *mut c_void,
                    &mut N_i as *mut _ as *mut c_void,
                    &mut off_i as *mut _ as *mut c_void,
                ];
                self.stream.launch(&build_min, build_grid, build_block, 0, &mut args2)
                    .map_err(|e| CudaDonchianError::Cuda(e.to_string()))?;

                // OR (u8)
                let mut prev_b = as_raw_offset(&d_st_nan, prev_elems);
                let mut curr_b = as_raw_offset(&d_st_nan, curr_elems);
                let mut args3: &mut [*mut c_void] = &mut [
                    &mut prev_b as *mut _ as *mut c_void,
                    &mut curr_b as *mut _ as *mut c_void,
                    &mut N_i as *mut _ as *mut c_void,
                    &mut off_i as *mut _ as *mut c_void,
                ];
                self.stream.launch(&build_or, build_grid, build_block, 0, &mut args3)
                    .map_err(|e| CudaDonchianError::Cuda(e.to_string()))?;
            }
        }

        // Query pass
        unsafe {
            let mut periods  = d_periods.as_device_ptr().as_raw();
            let mut series_len_i = len as i32;
            let mut n_combos_i   = combos.len() as i32;
            let mut first_i      = first_valid as i32;
            let mut st_hi = as_raw_offset(&d_st_high, 0);
            let mut st_lo = as_raw_offset(&d_st_low,  0);
            let mut st_nm = as_raw_offset(&d_st_nan,  0);
            let mut up_ptr = d_upper.as_device_ptr().as_raw();
            let mut mid_ptr = d_middle.as_device_ptr().as_raw();
            let mut lowo_ptr = d_lower.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut periods as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut n_combos_i as *mut _ as *mut c_void,
                &mut first_i as *mut _ as *mut c_void,
                &mut st_hi as *mut _ as *mut c_void,
                &mut st_lo as *mut _ as *mut c_void,
                &mut st_nm as *mut _ as *mut c_void,
                &mut up_ptr as *mut _ as *mut c_void,
                &mut mid_ptr as *mut _ as *mut c_void,
                &mut lowo_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func_query, query_grid, query_block, 0, args)
                .map_err(|e| CudaDonchianError::Cuda(e.to_string()))?;
        }

        self.stream.synchronize().map_err(|e| CudaDonchianError::Cuda(e.to_string()))?;
        self.maybe_log_batch_debug();

        Ok((
            DeviceArrayF32Triplet {
                wt1: DeviceArrayF32 { buf: d_upper, rows: combos.len(), cols: len },
                wt2: DeviceArrayF32 { buf: d_middle, rows: combos.len(), cols: len },
                hist: DeviceArrayF32 { buf: d_lower, rows: combos.len(), cols: len },
            },
            combos,
        ))
    }

    pub fn donchian_many_series_one_param_time_major_dev(
        &self,
        high_tm_f32: &[f32],
        low_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &DonchianParams,
    ) -> Result<DeviceArrayF32Triplet, CudaDonchianError> {
        if high_tm_f32.len() != low_tm_f32.len() { return Err(CudaDonchianError::InvalidInput("length mismatch".into())); }
        if cols == 0 || rows == 0 { return Err(CudaDonchianError::InvalidInput("empty matrix".into())); }
        if high_tm_f32.len() != cols * rows { return Err(CudaDonchianError::InvalidInput("bad shape".into())); }
        let period = params.period.unwrap_or(0);
        if period == 0 || period > rows { return Err(CudaDonchianError::InvalidInput("invalid period".into())); }

        // Build first_valid per series (column) respecting NaN-gating on either input
        let mut first_valids = vec![-1i32; cols];
        for s in 0..cols {
            let mut fv = -1i32;
            for t in 0..rows {
                let h = high_tm_f32[t * cols + s];
                let l = low_tm_f32[t * cols + s];
                if !h.is_nan() && !l.is_nan() { fv = t as i32; break; }
            }
            first_valids[s] = fv;
        }

        let bytes_in = 2 * cols * rows * std::mem::size_of::<f32>();
        let bytes_first = cols * std::mem::size_of::<i32>();
        let bytes_out = 3 * cols * rows * std::mem::size_of::<f32>();
        let required = bytes_in + bytes_first + bytes_out;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaDonchianError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                required as f64 / (1024.0 * 1024.0)
            )));
        }

        let d_high = unsafe { DeviceBuffer::from_slice_async(high_tm_f32, &self.stream) }
            .map_err(|e| CudaDonchianError::Cuda(e.to_string()))?;
        let d_low = unsafe { DeviceBuffer::from_slice_async(low_tm_f32, &self.stream) }
            .map_err(|e| CudaDonchianError::Cuda(e.to_string()))?;
        let d_first = unsafe { DeviceBuffer::from_slice_async(&first_valids, &self.stream) }
            .map_err(|e| CudaDonchianError::Cuda(e.to_string()))?;
        let mut d_upper: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(cols * rows, &self.stream) }
            .map_err(|e| CudaDonchianError::Cuda(e.to_string()))?;
        let mut d_middle: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(cols * rows, &self.stream) }
            .map_err(|e| CudaDonchianError::Cuda(e.to_string()))?;
        let mut d_lower: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(cols * rows, &self.stream) }
            .map_err(|e| CudaDonchianError::Cuda(e.to_string()))?;

        let func = self.module.get_function("donchian_many_series_one_param_f32")
            .map_err(|e| CudaDonchianError::Cuda(e.to_string()))?;
        let block_x: u32 = match self.policy.many_series { ManySeriesKernelPolicy::Auto => 128, ManySeriesKernelPolicy::OneD { block_x } => block_x.max(64) };
        let grid_x: u32 = ((cols as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe { (*(self as *const _ as *mut CudaDonchian)).last_many = Some(ManySeriesKernelSelected::OneD { block_x }); }
        unsafe {
            let mut high_ptr = d_high.as_device_ptr().as_raw();
            let mut low_ptr  = d_low.as_device_ptr().as_raw();
            let mut first_ptr = d_first.as_device_ptr().as_raw();
            let mut num_series_i = cols as i32;
            let mut series_len_i = rows as i32;
            let mut period_i = period as i32;
            let mut up_ptr = d_upper.as_device_ptr().as_raw();
            let mut mid_ptr = d_middle.as_device_ptr().as_raw();
            let mut lowo_ptr = d_lower.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut high_ptr as *mut _ as *mut c_void,
                &mut low_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut up_ptr as *mut _ as *mut c_void,
                &mut mid_ptr as *mut _ as *mut c_void,
                &mut lowo_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)
                .map_err(|e| CudaDonchianError::Cuda(e.to_string()))?;
        }

        self.stream.synchronize().map_err(|e| CudaDonchianError::Cuda(e.to_string()))?;
        self.maybe_log_many_debug();
        Ok(DeviceArrayF32Triplet {
            wt1: DeviceArrayF32 { buf: d_upper, rows, cols },
            wt2: DeviceArrayF32 { buf: d_middle, rows, cols },
            hist: DeviceArrayF32 { buf: d_lower, rows, cols },
        })
    }
}

// ---- Helpers for RMQ levels and sizing ----
#[inline]
fn floor_log2_usize(x: usize) -> u32 {
    debug_assert!(x > 0);
    (usize::BITS - 1 - x.leading_zeros()) as u32
}

#[inline]
fn rmq_levels_for_max_period(max_period: usize) -> usize {
    (floor_log2_usize(max_period) as usize) + 1
}

#[inline]
fn bytes_rmq_tables(len: usize, levels: usize) -> usize {
    levels * len * (2 * std::mem::size_of::<f32>() + std::mem::size_of::<u8>())
}

#[inline]
fn as_raw_offset<T: cust::memory::DeviceCopy>(buf: &DeviceBuffer<T>, elems_offset: usize) -> u64 {
    buf.as_device_ptr().as_raw() + (elems_offset * std::mem::size_of::<T>()) as u64
}

// ------------------------ Benches ------------------------
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};
    use crate::indicators::donchian::DonchianBatchRange;

    const ONE_SERIES_LEN: usize = 200_000; // keep moderate due to naive window scan
    const PARAM_SWEEP: usize = 64;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes  = 2 * ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = 3 * ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        let max_period = 10 + PARAM_SWEEP - 1; // matches prep sweep
        let levels = super::rmq_levels_for_max_period(max_period);
        let rmq_bytes = super::bytes_rmq_tables(ONE_SERIES_LEN, levels);
        in_bytes + out_bytes + rmq_bytes + 64 * 1024 * 1024
    }

    struct DonchianBatchState {
        cuda: CudaDonchian,
        high: Vec<f32>,
        low: Vec<f32>,
        sweep: DonchianBatchRange,
    }
    impl CudaBenchState for DonchianBatchState { fn launch(&mut self) { let _ = self.cuda.donchian_batch_dev(&self.high, &self.low, &self.sweep).expect("donchian batch"); } }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaDonchian::new(0).expect("CudaDonchian");
        let mut high = gen_series(ONE_SERIES_LEN);
        let mut low = vec![0.0f32; ONE_SERIES_LEN];
        for i in 0..ONE_SERIES_LEN { low[i] = 0.7 * high[i] + 0.1 * (i as f32).sin(); }
        // put NaNs at start for warmup semantics
        for i in 0..32 { high[i] = f32::NAN; low[i] = f32::NAN; }
        let sweep = DonchianBatchRange { period: (10, 10 + PARAM_SWEEP - 1, 1) };
        Box::new(DonchianBatchState { cuda, high, low, sweep })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "donchian",
            "one_series_many_params",
            "donchian_cuda_batch_dev",
            "200k_x_64",
            prep_one_series_many_params,
        )
        .with_sample_size(10)
        .with_mem_required(bytes_one_series_many_params())]
    }
}
