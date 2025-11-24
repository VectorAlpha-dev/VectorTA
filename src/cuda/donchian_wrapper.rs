//! CUDA wrapper for Donchian Channels (upper, middle, lower).
//!
//! Notes:
//! - PTX loaded from OUT_DIR with DetermineTargetFromContext + OptLevel O2 fallbacks.
//! - Returns VRAM-backed handles carrying an Arc<Context> and device_id for safe interop.
//! - Typed errors, MissingKernelSymbol mapping, and will_fit() VRAM checks.
//! - Public device entry points:
//!     - `donchian_batch_dev(&[f32], &[f32], &DonchianBatchRange)` -> (DeviceArrayF32Triplet, Vec<DonchianParams>)
//!     - `donchian_many_series_one_param_time_major_dev(&[f32], cols, rows, &DonchianParams)` -> DeviceArrayF32Triplet

#![cfg(feature = "cuda")]

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
use std::sync::Arc;
use thiserror::Error;

/// VRAM-backed array handle for Donchian with context guard and device id
pub struct DeviceArrayF32 {
    pub buf: DeviceBuffer<f32>,
    pub rows: usize,
    pub cols: usize,
    pub ctx: Arc<Context>,
    pub device_id: u32,
}
impl DeviceArrayF32 {
    #[inline]
    pub fn device_ptr(&self) -> u64 { self.buf.as_device_ptr().as_raw() as u64 }
    #[inline]
    pub fn len(&self) -> usize { self.rows * self.cols }
}

pub struct DeviceArrayF32Triplet {
    pub wt1: DeviceArrayF32,
    pub wt2: DeviceArrayF32,
    pub hist: DeviceArrayF32,
}
impl DeviceArrayF32Triplet {
    #[inline] pub fn rows(&self) -> usize { self.wt1.rows }
    #[inline] pub fn cols(&self) -> usize { self.wt1.cols }
}

#[derive(Debug, Error)]
pub enum CudaDonchianError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("out of memory: required={required} free={free} headroom={headroom}")]
    OutOfMemory { required: usize, free: usize, headroom: usize },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("device mismatch: buf={buf} current={current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
    NotImplemented,
}

#[derive(Clone, Copy, Debug, Default)]
pub enum BatchKernelPolicy {
    #[default]
    Auto,
    Plain {
        block_x: u32,
    },
}

#[derive(Clone, Copy, Debug, Default)]
pub enum ManySeriesKernelPolicy {
    #[default]
    Auto,
    OneD {
        block_x: u32,
    },
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaDonchianPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected { Rmq { build_bx: u32, query_bx: u32 } }
#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}

pub struct CudaDonchian {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
    policy: CudaDonchianPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaDonchian {
    pub fn new(device_id: usize) -> Result<Self, CudaDonchianError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/donchian_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, jit_opts)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))?;

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        Ok(Self {
            module,
            stream,
            context,
            device_id: device_id as u32,
            policy: CudaDonchianPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    #[inline]
    fn validate_launch(grid: GridSize, block: BlockSize) -> Result<(), CudaDonchianError> {
        let (gx, gy, gz) = (grid.x, grid.y, grid.z);
        let (bx, by, bz) = (block.x, block.y, block.z);
        let threads = (bx as u64) * (by as u64) * (bz as u64);
        if threads > 1024 {
            return Err(CudaDonchianError::LaunchConfigTooLarge { gx, gy, gz, bx, by, bz });
        }
        Ok(())
    }

    pub fn set_policy(&mut self, policy: CudaDonchianPolicy) {
        self.policy = policy;
    }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch
    }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        }
    }
    #[inline]
    fn device_mem_info() -> Option<(usize, usize)> { mem_get_info().ok() }
    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaDonchianError> {
        if !Self::mem_check_enabled() { return Ok(()); }
        match Self::device_mem_info() {
            Some((free, _)) => {
                if required_bytes.saturating_add(headroom_bytes) <= free {
                    Ok(())
                } else {
                    Err(CudaDonchianError::OutOfMemory { required: required_bytes, free, headroom: headroom_bytes })
                }
            }
            None => Ok(()),
        }
    }
    #[inline]
    fn maybe_log_batch_debug(&self) {
        static ONCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if self.debug_batch_logged {
            return;
        }
        if self.debug_batch_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                if !ONCE.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    eprintln!("[DEBUG] donchian batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaDonchian)).debug_batch_logged = true;
                }
                unsafe {
                    (*(self as *const _ as *mut CudaDonchian)).debug_batch_logged = true;
                }
            }
        }
    }
    #[inline]
    fn maybe_log_many_debug(&self) {
        static ONCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if self.debug_many_logged {
            return;
        }
        if self.debug_many_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                if !ONCE.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    eprintln!("[DEBUG] donchian many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaDonchian)).debug_many_logged = true;
                }
                unsafe {
                    (*(self as *const _ as *mut CudaDonchian)).debug_many_logged = true;
                }
            }
        }
    }

    fn expand_grid(range: &DonchianBatchRange) -> Result<Vec<DonchianParams>, CudaDonchianError> {
        fn axis_usize((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, CudaDonchianError> {
            if step == 0 || start == end { return Ok(vec![start]); }
            if start < end {
                Ok((start..=end).step_by(step).collect())
            } else {
                let mut v = Vec::new();
                let mut cur = start;
                while cur >= end {
                    v.push(cur);
                    if let Some(next) = cur.checked_sub(step) { cur = next; } else { break; }
                    if cur == usize::MAX { break; }
                }
                if v.is_empty() { return Err(CudaDonchianError::InvalidInput(format!("invalid range: start={} end={} step={}", start, end, step))); }
                Ok(v)
            }
        }
        let periods = axis_usize(range.period)?;
        Ok(periods.into_iter().map(|p| DonchianParams { period: Some(p) }).collect())
    }

    fn prepare_batch_inputs(
        high: &[f32],
        low: &[f32],
        sweep: &DonchianBatchRange,
    ) -> Result<(Vec<DonchianParams>, usize, usize), CudaDonchianError> {
        if high.len() != low.len() {
            return Err(CudaDonchianError::InvalidInput("length mismatch".into()));
        }
        if high.is_empty() {
            return Err(CudaDonchianError::InvalidInput("empty input".into()));
        }
        let len = high.len();
        let first_valid = high
            .iter()
            .zip(low.iter())
            .position(|(h, l)| !h.is_nan() && !l.is_nan())
            .ok_or_else(|| CudaDonchianError::InvalidInput("all values are NaN".into()))?;
        let combos = Self::expand_grid(sweep)?;
        if combos.is_empty() {
            return Err(CudaDonchianError::InvalidInput("no parameter combinations".into()));
        }
        for c in &combos {
            let p = c.period.unwrap_or(0);
            if p == 0 {
                return Err(CudaDonchianError::InvalidInput("period must be > 0".into()));
            }
            if p > len {
                return Err(CudaDonchianError::InvalidInput("period exceeds length".into()));
            }
            if len - first_valid < p {
                return Err(CudaDonchianError::InvalidInput("not enough valid data".into()));
            }
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
        let sz_f32 = std::mem::size_of::<f32>();
        let bytes_in = len.checked_mul(2).and_then(|v| v.checked_mul(sz_f32)).ok_or_else(|| CudaDonchianError::InvalidInput("size overflow (inputs)".into()))?;
        let bytes_periods = combos.len().checked_mul(std::mem::size_of::<i32>()).ok_or_else(|| CudaDonchianError::InvalidInput("size overflow (periods)".into()))?;
        let out_elems = combos.len().checked_mul(len).and_then(|v| v.checked_mul(3)).ok_or_else(|| CudaDonchianError::InvalidInput("size overflow (outputs)".into()))?;
        let bytes_out = out_elems.checked_mul(sz_f32).ok_or_else(|| CudaDonchianError::InvalidInput("size overflow (outputs bytes)".into()))?;
        let bytes_rmq = bytes_rmq_tables_checked(len, levels)?;
        let required = bytes_in
            .checked_add(bytes_periods).and_then(|v| v.checked_add(bytes_out)).and_then(|v| v.checked_add(bytes_rmq))
            .ok_or_else(|| CudaDonchianError::InvalidInput("size overflow (total)".into()))?;
        let headroom = 64 * 1024 * 1024;
        Self::will_fit(required, headroom)?;

        // Device buffers (common)
        let d_high = unsafe { DeviceBuffer::from_slice_async(high_f32, &self.stream) }?;
        let d_low = unsafe { DeviceBuffer::from_slice_async(low_f32, &self.stream) }?;
        let periods_i32: Vec<i32> = combos.iter().map(|c| c.period.unwrap() as i32).collect();
        let d_periods = unsafe { DeviceBuffer::from_slice_async(&periods_i32, &self.stream) }?;

        let mut d_upper: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(combos.len() * len, &self.stream) }?;
        let mut d_middle: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(combos.len() * len, &self.stream) }?;
        let mut d_lower: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(combos.len() * len, &self.stream) }?;

        // RMQ is always used for batch path
        let stride = len;

        // scratch buffers sized only to required levels
        let mut d_st_high: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(levels * stride, &self.stream) }?;
        let mut d_st_low: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(levels * stride, &self.stream) }?;
        let mut d_st_nan: DeviceBuffer<u8> = unsafe { DeviceBuffer::uninitialized_async(levels * stride, &self.stream) }?;

        let init_lvl0_f32 = self.module.get_function("rmq_init_level0_f32").map_err(|_| CudaDonchianError::MissingKernelSymbol { name: "rmq_init_level0_f32" })?;
        let init_nan_u8 = self.module.get_function("rmq_init_nan_mask_u8").map_err(|_| CudaDonchianError::MissingKernelSymbol { name: "rmq_init_nan_mask_u8" })?;
        let build_max = self.module.get_function("rmq_build_level_max_f32").map_err(|_| CudaDonchianError::MissingKernelSymbol { name: "rmq_build_level_max_f32" })?;
        let build_min = self.module.get_function("rmq_build_level_min_f32").map_err(|_| CudaDonchianError::MissingKernelSymbol { name: "rmq_build_level_min_f32" })?;
        let build_or  = self.module.get_function("rmq_build_level_or_u8").map_err(|_| CudaDonchianError::MissingKernelSymbol { name: "rmq_build_level_or_u8" })?;
        let func_query = self.module.get_function("donchian_batch_from_rmq_f32").map_err(|_| CudaDonchianError::MissingKernelSymbol { name: "donchian_batch_from_rmq_f32" })?;

        let build_bx: u32 = 256;
        let query_bx: u32 = match self.policy.batch { BatchKernelPolicy::Auto => 256, BatchKernelPolicy::Plain { block_x } => block_x.max(64) };
        let build_grid_x: u32 = ((len as u32) + build_bx - 1) / build_bx;
        let build_grid: GridSize = (build_grid_x.max(1), 1, 1).into();
        let build_block: BlockSize = (build_bx, 1, 1).into();
        Self::validate_launch(build_grid, build_block)?;
        let query_grid_x: u32 = ((combos.len() as u32) + query_bx - 1) / query_bx;
        let query_grid: GridSize = (query_grid_x.max(1), 1, 1).into();
        let query_block: BlockSize = (query_bx, 1, 1).into();
        Self::validate_launch(query_grid, query_block)?;
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
            self.stream.launch(&init_lvl0_f32, build_grid, build_block, 0, &mut args_hi0)?;

            let mut args_lo0: &mut [*mut c_void] = &mut [
                &mut low_in as *mut _ as *mut c_void,
                &mut out_lo0 as *mut _ as *mut c_void,
                &mut N_i as *mut _ as *mut c_void,
            ];
            self.stream.launch(&init_lvl0_f32, build_grid, build_block, 0, &mut args_lo0)?;

            let mut args_nm0: &mut [*mut c_void] = &mut [
                &mut high_in as *mut _ as *mut c_void,
                &mut low_in as *mut _ as *mut c_void,
                &mut N_i as *mut _ as *mut c_void,
                &mut first_i as *mut _ as *mut c_void,
                &mut mask0 as *mut _ as *mut c_void,
            ];
            self.stream.launch(&init_nan_u8, build_grid, build_block, 0, &mut args_nm0)?;
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
                self.stream.launch(&build_max, build_grid, build_block, 0, &mut args)?;

                // MIN
                prev = as_raw_offset(&d_st_low, prev_elems);
                curr = as_raw_offset(&d_st_low, curr_elems);
                let mut args2: &mut [*mut c_void] = &mut [
                    &mut prev as *mut _ as *mut c_void,
                    &mut curr as *mut _ as *mut c_void,
                    &mut N_i as *mut _ as *mut c_void,
                    &mut off_i as *mut _ as *mut c_void,
                ];
                self.stream.launch(&build_min, build_grid, build_block, 0, &mut args2)?;

                // OR (u8)
                let mut prev_b = as_raw_offset(&d_st_nan, prev_elems);
                let mut curr_b = as_raw_offset(&d_st_nan, curr_elems);
                let mut args3: &mut [*mut c_void] = &mut [
                    &mut prev_b as *mut _ as *mut c_void,
                    &mut curr_b as *mut _ as *mut c_void,
                    &mut N_i as *mut _ as *mut c_void,
                    &mut off_i as *mut _ as *mut c_void,
                ];
                self.stream.launch(&build_or, build_grid, build_block, 0, &mut args3)?;
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
            self.stream.launch(&func_query, query_grid, query_block, 0, args)?;
        }

        self.stream.synchronize()?;
        self.stream.synchronize()?;
        self.maybe_log_batch_debug();

        Ok((
            DeviceArrayF32Triplet {
                wt1: DeviceArrayF32 { buf: d_upper, rows: combos.len(), cols: len, ctx: self.context.clone(), device_id: self.device_id },
                wt2: DeviceArrayF32 { buf: d_middle, rows: combos.len(), cols: len, ctx: self.context.clone(), device_id: self.device_id },
                hist: DeviceArrayF32 { buf: d_lower, rows: combos.len(), cols: len, ctx: self.context.clone(), device_id: self.device_id },
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
        if high_tm_f32.len() != low_tm_f32.len() {
            return Err(CudaDonchianError::InvalidInput("length mismatch".into()));
        }
        if cols == 0 || rows == 0 {
            return Err(CudaDonchianError::InvalidInput("empty matrix".into()));
        }
        if high_tm_f32.len() != cols * rows {
            return Err(CudaDonchianError::InvalidInput("bad shape".into()));
        }
        if high_tm_f32.len() != low_tm_f32.len() {
            return Err(CudaDonchianError::InvalidInput("length mismatch".into()));
        }
        if cols == 0 || rows == 0 {
            return Err(CudaDonchianError::InvalidInput("empty matrix".into()));
        }
        if high_tm_f32.len() != cols * rows {
            return Err(CudaDonchianError::InvalidInput("bad shape".into()));
        }
        let period = params.period.unwrap_or(0);
        if period == 0 || period > rows {
            return Err(CudaDonchianError::InvalidInput("invalid period".into()));
        }
        if period == 0 || period > rows {
            return Err(CudaDonchianError::InvalidInput("invalid period".into()));
        }

        // Build first_valid per series (column) respecting NaN-gating on either input
        let mut first_valids = vec![-1i32; cols];
        for s in 0..cols {
            let mut fv = -1i32;
            for t in 0..rows {
                let h = high_tm_f32[t * cols + s];
                let l = low_tm_f32[t * cols + s];
                if !h.is_nan() && !l.is_nan() {
                    fv = t as i32;
                    break;
                }
                if !h.is_nan() && !l.is_nan() {
                    fv = t as i32;
                    break;
                }
            }
            first_valids[s] = fv;
        }

        let elem_f32 = std::mem::size_of::<f32>();
        let elem_i32 = std::mem::size_of::<i32>();
        let bytes_in = cols
            .checked_mul(rows)
            .and_then(|v| v.checked_mul(2))
            .and_then(|v| v.checked_mul(elem_f32))
            .ok_or_else(|| CudaDonchianError::InvalidInput("size overflow (inputs)".into()))?;
        let bytes_first = cols
            .checked_mul(elem_i32)
            .ok_or_else(|| CudaDonchianError::InvalidInput("size overflow (first_valid)".into()))?;
        let bytes_out = cols
            .checked_mul(rows)
            .and_then(|v| v.checked_mul(3))
            .and_then(|v| v.checked_mul(elem_f32))
            .ok_or_else(|| CudaDonchianError::InvalidInput("size overflow (outputs)".into()))?;
        let required = bytes_in
            .checked_add(bytes_first)
            .and_then(|v| v.checked_add(bytes_out))
            .ok_or_else(|| CudaDonchianError::InvalidInput("size overflow (total)".into()))?;
        let headroom = 64 * 1024 * 1024;
        Self::will_fit(required, headroom)?;

        let d_high = unsafe { DeviceBuffer::from_slice_async(high_tm_f32, &self.stream) }?;
        let d_low = unsafe { DeviceBuffer::from_slice_async(low_tm_f32, &self.stream) }?;
        let d_first = unsafe { DeviceBuffer::from_slice_async(&first_valids, &self.stream) }?;
        let mut d_upper: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(cols * rows, &self.stream) }?;
        let mut d_middle: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(cols * rows, &self.stream) }?;
        let mut d_lower: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(cols * rows, &self.stream) }?;

        let func = self
            .module
            .get_function("donchian_many_series_one_param_f32")
            .map_err(|_| CudaDonchianError::MissingKernelSymbol { name: "donchian_many_series_one_param_f32" })?;
        let block_x: u32 = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => 128,
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(64),
        };
        let grid_x: u32 = ((cols as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        Self::validate_launch(grid, block)?;
        unsafe {
            (*(self as *const _ as *mut CudaDonchian)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }
        unsafe {
            (*(self as *const _ as *mut CudaDonchian)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }
        unsafe {
            let mut high_ptr = d_high.as_device_ptr().as_raw();
            let mut low_ptr = d_low.as_device_ptr().as_raw();
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
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        self.stream.synchronize()?;
        self.stream.synchronize()?;
        self.maybe_log_many_debug();
        Ok(DeviceArrayF32Triplet {
            wt1: DeviceArrayF32 { buf: d_upper, rows, cols, ctx: self.context.clone(), device_id: self.device_id },
            wt2: DeviceArrayF32 { buf: d_middle, rows, cols, ctx: self.context.clone(), device_id: self.device_id },
            hist: DeviceArrayF32 { buf: d_lower, rows, cols, ctx: self.context.clone(), device_id: self.device_id },
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
fn bytes_rmq_tables_checked(len: usize, levels: usize) -> Result<usize, CudaDonchianError> {
    let elem = 2 * std::mem::size_of::<f32>() + std::mem::size_of::<u8>();
    levels
        .checked_mul(len)
        .and_then(|v| v.checked_mul(elem))
        .ok_or_else(|| CudaDonchianError::InvalidInput("size overflow (rmq)".into()))
}

#[inline]
fn bytes_rmq_tables(len: usize, levels: usize) -> usize {
    let elem = 2 * std::mem::size_of::<f32>() + std::mem::size_of::<u8>();
    levels.saturating_mul(len).saturating_mul(elem)
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
    impl CudaBenchState for DonchianBatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .donchian_batch_dev(&self.high, &self.low, &self.sweep)
                .expect("donchian batch");
        }
    }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaDonchian::new(0).expect("CudaDonchian");
        let mut high = gen_series(ONE_SERIES_LEN);
        let mut low = vec![0.0f32; ONE_SERIES_LEN];
        
        for i in 0..ONE_SERIES_LEN {
            low[i] = 0.7 * high[i] + 0.1 * (i as f32).sin();
        }
        // put NaNs at start for warmup semantics
        for i in 0..32 {
            high[i] = f32::NAN;
            low[i] = f32::NAN;
        }
        let sweep = DonchianBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1),
        };
        Box::new(DonchianBatchState {
            cuda,
            high,
            low,
            sweep,
        })
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
