//! CUDA scaffolding for the QStick indicator (average of close-open).
//!
//! Parity goals with ALMA wrapper:
//! - Policy enums for batch and many-series selection with light introspection
//! - PTX load via OUT_DIR, JIT DetermineTargetFromContext + OptLevel O2, fallbacks
//! - Stream NON_BLOCKING
//! - VRAM checks + chunking grid.y to <= 65_535 for batch
//! - Host precompute for prefix-sum category; kernels consume prefixes
//
#![cfg(feature = "cuda")]

use crate::indicators::qstick::{QstickBatchRange, QstickParams};
use cust::context::Context;
use cust::device::{Device, DeviceAttribute};
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

// Reuse the common VRAM handle from ALMA
use crate::cuda::moving_averages::alma_wrapper::DeviceArrayF32;

#[derive(Debug, Error)]
pub enum CudaQstickError {
    #[error("CUDA error: {0}")]
    Cuda(#[from] cust::error::CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error(
        "out of memory: required={required}B, free={free}B, headroom={headroom}B"
    )]
    OutOfMemory { required: usize, free: usize, headroom: usize },
    #[error(
        "launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})"
    )]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("device mismatch: buffer on {buf}, current {current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
    NotImplemented,
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    Plain { block_x: u32 },
    Tiled { tile: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 },
    // Tiled2D could be added later if needed
}

#[derive(Clone, Copy, Debug)]
pub struct CudaQstickPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaQstickPolicy {
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
    Tiled1x { tile: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}

pub struct CudaQstick {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
    policy: CudaQstickPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaQstick {
    pub fn new(device_id: usize) -> Result<Self, CudaQstickError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/qstick_kernel.ptx"));
        // Follow ALMA JIT policy: determine target from context + O2, fallback
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(match env::var("QS_JIT_OPT").ok().as_deref() {
                Some("O0") => OptLevel::O0,
                Some("O1") => OptLevel::O1,
                Some("O3") => OptLevel::O3,
                Some("O4") => OptLevel::O4,
                _ => OptLevel::O2,
            }),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => {
                if let Ok(m) = Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]) {
                    m
                } else {
                    Module::from_ptx(ptx, &[])?
                }
            }
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        Ok(Self {
            module,
            stream,
            context,
            device_id: device_id as u32,
            policy: CudaQstickPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn set_policy(&mut self, p: CudaQstickPolicy) {
        self.policy = p;
    }
    pub fn policy(&self) -> &CudaQstickPolicy {
        &self.policy
    }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch
    }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many
    }
    pub fn synchronize(&self) -> Result<(), CudaQstickError> {
        self.stream.synchronize().map_err(CudaQstickError::from)
    }

    #[inline]
    pub fn context_arc(&self) -> Arc<Context> {
        self.context.clone()
    }
    #[inline]
    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    #[inline]
    fn maybe_log_batch_debug(&self) {
        if self.debug_batch_logged {
            return;
        }
        if env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                eprintln!("[DEBUG] QStick batch selected kernel: {:?}", sel);
                unsafe {
                    (*(self as *const _ as *mut CudaQstick)).debug_batch_logged = true;
                }
            }
        }
    }
    #[inline]
    fn maybe_log_many_debug(&self) {
        if self.debug_many_logged {
            return;
        }
        if env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                eprintln!("[DEBUG] QStick many-series selected kernel: {:?}", sel);
                unsafe {
                    (*(self as *const _ as *mut CudaQstick)).debug_many_logged = true;
                }
            }
        }
    }

    // ------------------- Utilities -------------------
    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
            Err(_) => true,
        }
    }
    #[inline]
    fn device_mem_info() -> Option<(usize, usize)> {
        mem_get_info().ok()
    }
    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaQstickError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        if let Some((free, _)) = Self::device_mem_info() {
            if required_bytes.saturating_add(headroom_bytes) <= free {
                Ok(())
            } else {
                Err(CudaQstickError::OutOfMemory {
                    required: required_bytes,
                    free,
                    headroom: headroom_bytes,
                })
            }
        } else {
            Ok(())
        }
    }

    #[inline]
    fn validate_launch(&self, grid: (u32, u32, u32), block: (u32, u32, u32)) -> Result<(), CudaQstickError> {
        let dev = Device::get_device(self.device_id)?;
        let max_bx = dev.get_attribute(DeviceAttribute::MaxBlockDimX)? as u32;
        let max_by = dev.get_attribute(DeviceAttribute::MaxBlockDimY)? as u32;
        let max_bz = dev.get_attribute(DeviceAttribute::MaxBlockDimZ)? as u32;
        let max_gx = dev.get_attribute(DeviceAttribute::MaxGridDimX)? as u32;
        let max_gy = dev.get_attribute(DeviceAttribute::MaxGridDimY)? as u32;
        let max_gz = dev.get_attribute(DeviceAttribute::MaxGridDimZ)? as u32;
        let (gx, gy, gz) = grid;
        let (bx, by, bz) = block;
        if bx == 0
            || by == 0
            || bz == 0
            || gx == 0
            || gy == 0
            || gz == 0
            || bx > max_bx
            || by > max_by
            || bz > max_bz
            || gx > max_gx
            || gy > max_gy
            || gz > max_gz
        {
            return Err(CudaQstickError::LaunchConfigTooLarge { gx, gy, gz, bx, by, bz });
        }
        Ok(())
    }
    #[inline]
    fn pick_tiled_block(&self, len: usize) -> u32 {
        // Prefer 256 for long series; allow override via QS_TILE
        if let Ok(v) = env::var("QS_TILE") {
            if let Ok(b) = v.parse::<u32>() {
                return b;
            }
        }
        if len < 8192 {
            128
        } else {
            256
        }
    }

    // ------------------- Host precompute -------------------
    pub fn build_diff_prefix_f32(open: &[f32], close: &[f32]) -> (Vec<f32>, usize, usize) {
        let len = open.len().min(close.len());
        // first valid: both non-NaN
        let first = (0..len)
            .find(|&i| !open[i].is_nan() && !close[i].is_nan())
            .unwrap_or(0);
        let mut prefix = vec![0.0f32; len + 1];
        let mut acc = 0.0f64;
        for i in 0..len {
            if i < first {
                prefix[i + 1] = acc as f32;
                continue;
            }
            let d = (close[i] as f64) - (open[i] as f64);
            acc += d;
            prefix[i + 1] = acc as f32;
        }
        (prefix, first, len)
    }

    fn prepare_batch_inputs(
        open: &[f32],
        close: &[f32],
        sweep: &QstickBatchRange,
    ) -> Result<(Vec<QstickParams>, usize, usize), CudaQstickError> {
        if open.is_empty() || close.is_empty() {
            return Err(CudaQstickError::InvalidInput("empty inputs".into()));
        }
        let len = open.len().min(close.len());
        let first = (0..len)
            .find(|&i| !open[i].is_nan() && !close[i].is_nan())
            .ok_or_else(|| CudaQstickError::InvalidInput("all values are NaN".into()))?;

        // Expand grid
        let (start, end, step) = sweep.period;
        fn axis_usize(
            (start, end, step): (usize, usize, usize),
        ) -> Result<Vec<usize>, CudaQstickError> {
            if step == 0 || start == end {
                return Ok(vec![start]);
            }
            let mut v = Vec::new();
            if start < end {
                let mut cur = start;
                while cur <= end {
                    v.push(cur);
                    let next = cur.saturating_add(step);
                    if next == cur {
                        break;
                    }
                    cur = next;
                }
            } else {
                let mut cur = start;
                while cur >= end {
                    v.push(cur);
                    let next = cur.saturating_sub(step);
                    if next == cur {
                        break;
                    }
                    cur = next;
                    if cur == 0 && end > 0 {
                        break;
                    }
                }
            }
            if v.is_empty() {
                return Err(CudaQstickError::InvalidInput(
                    "empty period range".into(),
                ));
            }
            Ok(v)
        }
        let periods = axis_usize((start, end, step))?;
        let combos: Vec<QstickParams> = periods
            .into_iter()
            .map(|p| QstickParams { period: Some(p) })
            .collect();
        for c in &combos {
            let p = c.period.unwrap_or(0);
            if p == 0 || p > len {
                return Err(CudaQstickError::InvalidInput(format!(
                    "invalid period {}",
                    p
                )));
            }
            if len - first < p {
                return Err(CudaQstickError::InvalidInput(format!(
                    "not enough valid data: need {}, have {} after first_valid {}",
                    p,
                    len - first,
                    first
                )));
            }
        }
        Ok((combos, first, len))
    }

    fn launch_batch_kernel(
        &self,
        d_prefix: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        len: usize,
        first_valid: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaQstickError> {
        // Decide kernel
        let mut use_tiled = len > 8192;
        let mut block_x: u32 = 256;
        let mut tile_choice: Option<u32> = None;
        match self.policy.batch {
            BatchKernelPolicy::Auto => {}
            BatchKernelPolicy::Plain { block_x: bx } => {
                use_tiled = false;
                block_x = bx;
            }
            BatchKernelPolicy::Tiled { tile } => {
                use_tiled = true;
                tile_choice = Some(tile);
            }
        }

        const MAX_GRID_Y: usize = 65_535;
        if use_tiled {
            let tile = tile_choice.unwrap_or_else(|| self.pick_tiled_block(len));
            let func_name = match tile {
                128 => "qstick_batch_prefix_tiled_f32_tile128",
                _ => "qstick_batch_prefix_tiled_f32_tile256",
            };
            let func = self
                .module
                .get_function(func_name)
                .or_else(|_| self.module.get_function("qstick_batch_prefix_f32"))
                .map_err(|_| CudaQstickError::MissingKernelSymbol { name: "qstick_batch_prefix_f32" })?;
            unsafe {
                (*(self as *const _ as *mut CudaQstick)).last_batch =
                    Some(BatchKernelSelected::Tiled1x { tile });
            }
            self.maybe_log_batch_debug();

            let grid_x = ((len as u32) + tile - 1) / tile;
            let block: BlockSize = (tile, 1, 1).into();
            self.validate_launch((grid_x.max(1), 1, 1), (tile, 1, 1))?;
            let mut start = 0usize;
            while start < n_combos {
                let chunk = (n_combos - start).min(MAX_GRID_Y);
                let grid: GridSize = (grid_x.max(1), chunk as u32, 1).into();
                unsafe {
                    let mut p_ptr = d_prefix.as_device_ptr().as_raw();
                    let mut len_i = len as i32;
                    let mut first_i = first_valid as i32;
                    let mut per_ptr = d_periods
                        .as_device_ptr()
                        .as_raw()
                        .wrapping_add((start * std::mem::size_of::<i32>()) as u64);
                    let mut n_i = chunk as i32;
                    let mut out_ptr = d_out
                        .as_device_ptr()
                        .as_raw()
                        .wrapping_add((start * len * std::mem::size_of::<f32>()) as u64);
                    let args: &mut [*mut c_void] = &mut [
                        &mut p_ptr as *mut _ as *mut c_void,
                        &mut len_i as *mut _ as *mut c_void,
                        &mut first_i as *mut _ as *mut c_void,
                        &mut per_ptr as *mut _ as *mut c_void,
                        &mut n_i as *mut _ as *mut c_void,
                        &mut out_ptr as *mut _ as *mut c_void,
                    ];
            self.stream
                .launch(&func, grid, block, 0, args)
                        .map_err(CudaQstickError::from)?;
                }
                start += chunk;
            }
        } else {
            let func = self
                .module
                .get_function("qstick_batch_prefix_f32")
                .map_err(|_| CudaQstickError::MissingKernelSymbol { name: "qstick_batch_prefix_f32" })?;
            unsafe {
                (*(self as *const _ as *mut CudaQstick)).last_batch =
                    Some(BatchKernelSelected::Plain { block_x });
            }
            self.maybe_log_batch_debug();

            let grid_x = ((len as u32) + block_x - 1) / block_x;
            let block: BlockSize = (block_x, 1, 1).into();
            self.validate_launch((grid_x.max(1), 1, 1), (block_x, 1, 1))?;
            let mut start = 0usize;
            while start < n_combos {
                let chunk = (n_combos - start).min(MAX_GRID_Y);
                let grid: GridSize = (grid_x.max(1), chunk as u32, 1).into();
                unsafe {
                    let mut p_ptr = d_prefix.as_device_ptr().as_raw();
                    let mut len_i = len as i32;
                    let mut first_i = first_valid as i32;
                    let mut per_ptr = d_periods
                        .as_device_ptr()
                        .as_raw()
                        .wrapping_add((start * std::mem::size_of::<i32>()) as u64);
                    let mut n_i = chunk as i32;
                    let mut out_ptr = d_out
                        .as_device_ptr()
                        .as_raw()
                        .wrapping_add((start * len * std::mem::size_of::<f32>()) as u64);
                    let args: &mut [*mut c_void] = &mut [
                        &mut p_ptr as *mut _ as *mut c_void,
                        &mut len_i as *mut _ as *mut c_void,
                        &mut first_i as *mut _ as *mut c_void,
                        &mut per_ptr as *mut _ as *mut c_void,
                        &mut n_i as *mut _ as *mut c_void,
                        &mut out_ptr as *mut _ as *mut c_void,
                    ];
            self.stream
                .launch(&func, grid, block, 0, args)
                        .map_err(CudaQstickError::from)?;
                }
                start += chunk;
            }
        }
        Ok(())
    }

    fn launch_many_series_kernel(
        &self,
        d_prefix_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        period: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaQstickError> {
        // Only OneD for now; grid.y over series, grid.x over time
        let block_x: u32 = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => 256,
            ManySeriesKernelPolicy::OneD { block_x } => block_x,
        };
        let func = self
            .module
            .get_function("qstick_many_series_one_param_f32")
            .map_err(|_| CudaQstickError::MissingKernelSymbol { name: "qstick_many_series_one_param_f32" })?;

        unsafe {
            (*(self as *const _ as *mut CudaQstick)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }
        self.maybe_log_many_debug();

        let grid_x = ((rows as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), cols as u32, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        self.validate_launch((grid_x.max(1), cols as u32, 1), (block_x, 1, 1))?;

        unsafe {
            let mut p_ptr = d_prefix_tm.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut num_series_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut fv_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut p_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut fv_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(CudaQstickError::from)?;
        }
        Ok(())
    }

    // ------------------- Public device entry points -------------------
    pub fn qstick_batch_dev(
        &self,
        open_f32: &[f32],
        close_f32: &[f32],
        sweep: &QstickBatchRange,
    ) -> Result<DeviceArrayF32, CudaQstickError> {
        let (combos, first_valid, len) = Self::prepare_batch_inputs(open_f32, close_f32, sweep)?;

        // VRAM estimate: prefix (len+1) + periods + outputs
        let bytes_prefix = (len + 1)
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaQstickError::InvalidInput("size overflow".into()))?;
        let bytes_periods = combos
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaQstickError::InvalidInput("size overflow".into()))?;
        let bytes_out = combos
            .len()
            .checked_mul(len)
            .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| CudaQstickError::InvalidInput("size overflow".into()))?;
        let bytes_required = bytes_prefix
            .checked_add(bytes_periods)
            .and_then(|v| v.checked_add(bytes_out))
            .ok_or_else(|| CudaQstickError::InvalidInput("size overflow".into()))?;
        let headroom = env::var("CUDA_MEM_HEADROOM")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(64 * 1024 * 1024);
        Self::will_fit(bytes_required, headroom)?;

        // Build prefix on host
        let (prefix, _fv2, _l2) = Self::build_diff_prefix_f32(open_f32, close_f32);
        let d_prefix = DeviceBuffer::from_slice(&prefix)?;
        let periods_i32: Vec<i32> = combos.iter().map(|c| c.period.unwrap() as i32).collect();
        let d_periods = DeviceBuffer::from_slice(&periods_i32)?;
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized_async(combos.len() * len, &self.stream)?
        };

        self.launch_batch_kernel(
            &d_prefix,
            &d_periods,
            len,
            first_valid,
            combos.len(),
            &mut d_out,
        )?;
        self.stream.synchronize()?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: combos.len(),
            cols: len,
        })
    }

    pub fn prepare_many_series_inputs(
        open_tm_f32: &[f32],
        close_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
    ) -> Result<(Vec<f32>, Vec<i32>), CudaQstickError> {
        if cols == 0 || rows == 0 {
            return Err(CudaQstickError::InvalidInput("empty matrix".into()));
        }
        if open_tm_f32.len() != cols * rows || close_tm_f32.len() != cols * rows {
            return Err(CudaQstickError::InvalidInput("shape mismatch".into()));
        }
        if period == 0 || period > rows {
            return Err(CudaQstickError::InvalidInput("invalid period".into()));
        }
        // Build (rows+1) x cols prefix of diffs, time-major
        let mut prefix_tm = vec![0.0f32; (rows + 1) * cols];
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            // find first valid in this series
            let mut fv = 0usize;
            for t in 0..rows {
                let o = open_tm_f32[t * cols + s];
                let c = close_tm_f32[t * cols + s];
                if !o.is_nan() && !c.is_nan() {
                    fv = t;
                    break;
                }
                if t == rows - 1 {
                    return Err(CudaQstickError::InvalidInput(
                        "all values NaN in a series".into(),
                    ));
                }
            }
            first_valids[s] = fv as i32;
            let mut acc = 0.0f64;
            for t in 0..rows {
                if t < fv {
                    prefix_tm[(t + 1) * cols + s] = acc as f32;
                } else {
                    let d =
                        (close_tm_f32[t * cols + s] as f64) - (open_tm_f32[t * cols + s] as f64);
                    acc += d;
                    prefix_tm[(t + 1) * cols + s] = acc as f32;
                }
            }
        }
        Ok((prefix_tm, first_valids))
    }

    pub fn qstick_many_series_one_param_time_major_dev(
        &self,
        open_tm_f32: &[f32],
        close_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
    ) -> Result<DeviceArrayF32, CudaQstickError> {
        // Validate + build host prefixes
        let (prefix_tm, first_valids) =
            Self::prepare_many_series_inputs(open_tm_f32, close_tm_f32, cols, rows, period)?;

        // VRAM estimate: prefix (rows+1)*cols + first_valids + output rows*cols
        let bytes_prefix = (rows + 1)
            .checked_mul(cols)
            .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| CudaQstickError::InvalidInput("size overflow".into()))?;
        let bytes_first = cols
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaQstickError::InvalidInput("size overflow".into()))?;
        let bytes_out = rows
            .checked_mul(cols)
            .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| CudaQstickError::InvalidInput("size overflow".into()))?;
        let bytes_required = bytes_prefix
            .checked_add(bytes_first)
            .and_then(|v| v.checked_add(bytes_out))
            .ok_or_else(|| CudaQstickError::InvalidInput("size overflow".into()))?;
        let headroom = env::var("CUDA_MEM_HEADROOM")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(64 * 1024 * 1024);
        Self::will_fit(bytes_required, headroom)?;

        let d_prefix_tm = DeviceBuffer::from_slice(&prefix_tm)?;
        let d_first_valids = DeviceBuffer::from_slice(&first_valids)?;
        let mut d_out_tm: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized_async(rows * cols, &self.stream)?
        };

        self.launch_many_series_kernel(
            &d_prefix_tm,
            &d_first_valids,
            cols,
            rows,
            period,
            &mut d_out_tm,
        )?;
        self.stream.synchronize()?;
        Ok(DeviceArrayF32 {
            buf: d_out_tm,
            rows,
            cols,
        })
    }
}

// ---------------- Benches registration (for benches/cuda_bench.rs) ----------------
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_time_major_prices;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "qstick",
                "batch_dev",
                "qstick_cuda_batch_dev",
                "60k_x_40periods",
                prep_qstick_batch_box,
            )
            .with_inner_iters(8),
            CudaBenchScenario::new(
                "qstick",
                "many_series_one_param",
                "qstick_cuda_many_series_one_param",
                "250x1m",
                prep_qstick_many_series_box,
            )
            .with_inner_iters(4),
        ]
    }

    struct QsBatchState {
        cuda: CudaQstick,
        d_prefix: DeviceBuffer<f32>,
        d_periods: DeviceBuffer<i32>,
        d_out: DeviceBuffer<f32>,
        len: usize,
        n_combos: usize,
        first_valid: usize,
    }
    impl CudaBenchState for QsBatchState {
        fn launch(&mut self) {
            self.cuda
                .launch_batch_kernel(
                    &self.d_prefix,
                    &self.d_periods,
                    self.len,
                    self.first_valid,
                    self.n_combos,
                    &mut self.d_out,
                )
                .expect("qstick launch");
            self.cuda.synchronize().expect("sync");
        }
    }

    fn prep_qstick_batch() -> QsBatchState {
        let mut cuda = CudaQstick::new(0).expect("cuda qstick");
        cuda.set_policy(CudaQstickPolicy {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        });

        let len = 60_000usize;
        let mut open = vec![f32::NAN; len];
        let mut close = vec![f32::NAN; len];
        for i in 3..len {
            let x = i as f32;
            open[i] = (x * 0.0007).cos() + 0.01 * (x * 0.0001).sin();
            close[i] = open[i] + 0.05 * (x * 0.0017).sin();
        }
        let (prefix, first_valid, _len) = CudaQstick::build_diff_prefix_f32(&open, &close);
        let periods: Vec<i32> = (5..=200).step_by(5).map(|p| p as i32).collect();
        let d_prefix = DeviceBuffer::from_slice(&prefix).expect("d_prefix");
        let d_periods = DeviceBuffer::from_slice(&periods).expect("d_periods");
        let n_combos = periods.len();
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_combos * len) }.expect("d_out");

        QsBatchState {
            cuda,
            d_prefix,
            d_periods,
            d_out,
            len,
            n_combos,
            first_valid,
        }
    }
    fn prep_qstick_batch_box() -> Box<dyn CudaBenchState> {
        Box::new(prep_qstick_batch())
    }

    struct QsManyState {
        cuda: CudaQstick,
        d_prefix_tm: DeviceBuffer<f32>,
        d_first_valids: DeviceBuffer<i32>,
        d_out_tm: DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        period: usize,
    }
    impl CudaBenchState for QsManyState {
        fn launch(&mut self) {
            self.cuda
                .launch_many_series_kernel(
                    &self.d_prefix_tm,
                    &self.d_first_valids,
                    self.cols,
                    self.rows,
                    self.period,
                    &mut self.d_out_tm,
                )
                .expect("qstick many launch");
            self.cuda.synchronize().expect("sync");
        }
    }

    fn prep_qstick_many_series() -> QsManyState {
        let mut cuda = CudaQstick::new(0).expect("cuda qstick");
        cuda.set_policy(CudaQstickPolicy {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        });

        let cols = 250usize;
        let rows = 1_000_000usize;
        // Generate a synthetic open/close TM pair
        let p_tm = gen_time_major_prices(cols, rows);
        let mut o_tm = vec![0f32; cols * rows];
        let mut c_tm = vec![0f32; cols * rows];
        for t in 0..rows {
            for s in 0..cols {
                let idx = t * cols + s;
                o_tm[idx] = p_tm[idx] - 0.05;
                c_tm[idx] = p_tm[idx] + 0.05;
            }
        }

        let period = 21usize;
        let (prefix_tm, first_valids) =
            CudaQstick::prepare_many_series_inputs(&o_tm, &c_tm, cols, rows, period).expect("prep");
        let d_prefix_tm = DeviceBuffer::from_slice(&prefix_tm).expect("d_prefix_tm");
        let d_first_valids = DeviceBuffer::from_slice(&first_valids).expect("d_first_valids");
        let d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(cols * rows) }.expect("d_out_tm");
        QsManyState {
            cuda,
            d_prefix_tm,
            d_first_valids,
            d_out_tm,
            cols,
            rows,
            period,
        }
    }
    fn prep_qstick_many_series_box() -> Box<dyn CudaBenchState> {
        Box::new(prep_qstick_many_series())
    }
}
