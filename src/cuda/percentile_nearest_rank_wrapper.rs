//! CUDA wrapper for Percentile Nearest Rank (PNR).
//!
//! Parity goals with ALMA/CWMA wrappers:
//! - PTX load via include_str!(concat!(env!("OUT_DIR"), "/percentile_nearest_rank_kernel.ptx"))
//! - NON_BLOCKING stream
//! - Simple policy surface with introspection + one-time BENCH_DEBUG logging
//! - VRAM estimate + headroom check; chunking not needed for this baseline
//! - Public device entry points:
//!     - one-series × many-params (batch)
//!     - many-series × one-param (time-major)

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::alma_wrapper::DeviceArrayF32;
use crate::indicators::percentile_nearest_rank::{
    PercentileNearestRankBatchRange, PercentileNearestRankParams,
};
use cust::context::{CacheConfig, Context};
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CudaPnrError {
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

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    OneD { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub struct CudaPnrPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaPnrPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected {
    OneD { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}

pub struct CudaPercentileNearestRank {
    module: Module,
    stream: Stream,
    context: std::sync::Arc<Context>,
    device_id: u32,
    policy: CudaPnrPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaPercentileNearestRank {
    pub fn new(device_id: usize) -> Result<Self, CudaPnrError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = std::sync::Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(
            env!("OUT_DIR"),
            "/percentile_nearest_rank_kernel.ptx"
        ));
        // Stable JIT options with fallbacks
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
            policy: CudaPnrPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn synchronize(&self) -> Result<(), CudaPnrError> {
        self.stream.synchronize()?;
        Ok(())
    }

    pub fn context_arc(&self) -> std::sync::Arc<Context> { self.context.clone() }
    pub fn device_id(&self) -> u32 { self.device_id }

    pub fn set_policy(&mut self, policy: CudaPnrPolicy) { self.policy = policy; }
    pub fn policy(&self) -> &CudaPnrPolicy { &self.policy }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> { self.last_batch }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> { self.last_many }

    #[inline]
    fn maybe_log(
        sel: Option<impl fmt::Debug>,
        which: &str,
        once_flag: &AtomicBool,
        printed: &mut bool,
    ) {
        if *printed {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(s) = sel {
                let per = std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per || !once_flag.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] PNR {} selected kernel: {:?}", which, s);
                }
                *printed = true;
            }
        }
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        match std::env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
            Err(_) => true,
        }
    }
    #[inline]
    fn device_mem_info() -> Option<(usize, usize)> {
        mem_get_info().ok()
    }
    #[inline]
    fn will_fit(required: usize, headroom: usize) -> bool {
        if !Self::mem_check_enabled() {
            return true;
        }
        if let Some((free, _)) = Self::device_mem_info() {
            required.saturating_add(headroom) <= free
        } else {
            true
        }
    }

    // -------- Helpers --------

    fn axis_usize(
        axis: (usize, usize, usize),
    ) -> Result<Vec<usize>, CudaPnrError> {
        let (start, end, step) = axis;
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        if start < end {
            return Ok((start..=end).step_by(step.max(1)).collect());
        }
        let mut out = Vec::new();
        let mut x = start as isize;
        let end_i = end as isize;
        let st = (step as isize).max(1);
        while x >= end_i {
            out.push(x as usize);
            x -= st;
        }
        if out.is_empty() {
            return Err(CudaPnrError::InvalidInput(format!(
                "invalid length range: start={}, end={}, step={}",
                start, end, step
            )));
        }
        Ok(out)
    }

    fn axis_f64(
        axis: (f64, f64, f64),
    ) -> Result<Vec<f64>, CudaPnrError> {
        let (start, end, step) = axis;
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return Ok(vec![start]);
        }
        if start < end {
            let mut out = Vec::new();
            let mut x = start;
            let st = step.abs();
            while x <= end + 1e-12 {
                out.push(x);
                x += st;
            }
            if out.is_empty() {
                return Err(CudaPnrError::InvalidInput(format!(
                    "invalid percentage range: start={}, end={}, step={}",
                    start, end, step
                )));
            }
            return Ok(out);
        }
        let mut out = Vec::new();
        let mut x = start;
        let st = step.abs();
        while x + 1e-12 >= end {
            out.push(x);
            x -= st;
        }
        if out.is_empty() {
            return Err(CudaPnrError::InvalidInput(format!(
                "invalid percentage range: start={}, end={}, step={}",
                start, end, step
            )));
        }
        Ok(out)
    }

    fn expand_grid(
        r: &PercentileNearestRankBatchRange,
    ) -> Result<Vec<PercentileNearestRankParams>, CudaPnrError> {
        let lengths = Self::axis_usize(r.length)?;
        let percentages = Self::axis_f64(r.percentage)?;
        let cap = lengths
            .len()
            .checked_mul(percentages.len())
            .ok_or_else(|| CudaPnrError::InvalidInput("lengths*percentages overflow".into()))?;
        if cap == 0 {
            return Err(CudaPnrError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        let mut combos = Vec::with_capacity(cap);
        for &l in &lengths {
            for &p in &percentages {
                combos.push(PercentileNearestRankParams {
                    length: Some(l),
                    percentage: Some(p),
                });
            }
        }
        Ok(combos)
    }

    #[inline]
    fn next_pow2_u32(x: u32) -> u32 {
        if x <= 1 {
            1
        } else {
            x.next_power_of_two()
        }
    }

    // Dynamic shared bytes for same-length shared kernel
    #[inline]
    fn smem_bytes_for_len(length: usize) -> usize {
        (Self::next_pow2_u32(length as u32) as usize) * core::mem::size_of::<f32>()
    }

    // Simple heuristic for when the shared-sort kernel tends to win
    #[inline]
    fn shared_worth_it(length: usize, group_size: usize) -> bool {
        if group_size < 4 {
            return false;
        }
        let denom = (length as f64).log2().max(1.0);
        group_size >= ((length as f64) / denom).ceil() as usize
    }

    #[inline]
    fn validate_launch(
        &self,
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
    ) -> Result<(), CudaPnrError> {
        let dev = Device::get_device(self.device_id)?;
        let max_threads = dev
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_bx = dev
            .get_attribute(DeviceAttribute::MaxBlockDimX)
            .unwrap_or(1024) as u32;
        let max_by = dev
            .get_attribute(DeviceAttribute::MaxBlockDimY)
            .unwrap_or(1024) as u32;
        let max_bz = dev
            .get_attribute(DeviceAttribute::MaxBlockDimZ)
            .unwrap_or(64) as u32;
        let max_gx = dev
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(2_147_483_647) as u32;
        let max_gy = dev
            .get_attribute(DeviceAttribute::MaxGridDimY)
            .unwrap_or(65_535) as u32;
        let max_gz = dev
            .get_attribute(DeviceAttribute::MaxGridDimZ)
            .unwrap_or(65_535) as u32;

        let threads = bx.saturating_mul(by).saturating_mul(bz);
        if threads > max_threads
            || bx > max_bx
            || by > max_by
            || bz > max_bz
            || gx > max_gx
            || gy > max_gy
            || gz > max_gz
        {
            return Err(CudaPnrError::LaunchConfigTooLarge {
                gx,
                gy,
                gz,
                bx,
                by,
                bz,
            });
        }
        Ok(())
    }

    // -------- Batch: one-series × many params --------
    pub fn pnr_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &PercentileNearestRankBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<PercentileNearestRankParams>), CudaPnrError> {
        if data_f32.is_empty() {
            return Err(CudaPnrError::InvalidInput("empty data".into()));
        }
        let len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaPnrError::InvalidInput("all values are NaN".into()))?;

        // Length-major, percentage-minor pairs
        let combos = Self::expand_grid(sweep)?;
        if combos.is_empty() {
            return Err(CudaPnrError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        let max_len = combos.iter().map(|c| c.length.unwrap_or(15)).max().unwrap();
        if len - first_valid < max_len {
            return Err(CudaPnrError::InvalidInput("not enough valid data".into()));
        }
        
        // Flattened parameter axes
        let periods: Vec<i32> = combos
            .iter()
            .map(|c| c.length.unwrap_or(15) as i32)
            .collect();
        let percs: Vec<f32> = combos
            .iter()
            .map(|c| c.percentage.unwrap_or(50.0) as f32)
            .collect();

        // Group info
        let lengths_axis: Vec<usize> = Self::axis_usize(sweep.length)?;
        let percs_axis: Vec<f64> = Self::axis_f64(sweep.percentage)?;
        let group_rows = percs_axis.len();

        // Memory estimate with per-group scratch (baseline groups only)
        let prices_bytes = len
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or_else(|| CudaPnrError::InvalidInput("series_len bytes overflow".into()))?;
        let percs_bytes = percs
            .len()
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or_else(|| CudaPnrError::InvalidInput("percs bytes overflow".into()))?;
        let periods_bytes = periods
            .len()
            .checked_mul(core::mem::size_of::<i32>())
            .ok_or_else(|| CudaPnrError::InvalidInput("periods bytes overflow".into()))?;
        let out_elems = combos
            .len()
            .checked_mul(len)
            .ok_or_else(|| CudaPnrError::InvalidInput("output elements overflow".into()))?;
        let out_bytes = out_elems
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or_else(|| CudaPnrError::InvalidInput("output bytes overflow".into()))?;

        // Query shared kernel handle and set cache pref
        let mut func_shared = self
            .module
            .get_function("percentile_nearest_rank_one_series_many_params_same_len_f32")
            .map_err(|_| CudaPnrError::MissingKernelSymbol {
                name: "percentile_nearest_rank_one_series_many_params_same_len_f32",
            })?;
        func_shared.set_cache_config(CacheConfig::PreferShared)?;

        // Per-block shared memory limit (conservative)
        let max_smem_per_block = Device::get_device(0)
            .and_then(|d| d.get_attribute(DeviceAttribute::MaxSharedMemoryPerBlock))
            .unwrap_or(48 * 1024) as usize;

        // Decide which length-groups use shared path
        let mut use_shared: Vec<bool> = Vec::with_capacity(lengths_axis.len());
        for &L in &lengths_axis {
            let smem_need = Self::smem_bytes_for_len(L);
            let enough_smem = smem_need <= max_smem_per_block;
            use_shared.push(enough_smem && Self::shared_worth_it(L, group_rows));
        }

        // Sum scratch for baseline groups only
        let mut scratch_bytes = 0usize;
        for (g, &L) in lengths_axis.iter().enumerate() {
            if !use_shared[g] {
                let group = group_rows
                    .checked_mul(L)
                    .and_then(|x| x.checked_mul(core::mem::size_of::<f32>()))
                    .ok_or_else(|| {
                        CudaPnrError::InvalidInput("scratch bytes overflow".into())
                    })?;
                scratch_bytes = scratch_bytes
                    .checked_add(group)
                    .ok_or_else(|| {
                        CudaPnrError::InvalidInput("scratch bytes overflow".into())
                    })?;
            }
        }

        let required = prices_bytes
            .checked_add(percs_bytes)
            .and_then(|x| x.checked_add(periods_bytes))
            .and_then(|x| x.checked_add(out_bytes))
            .and_then(|x| x.checked_add(scratch_bytes))
            .ok_or_else(|| CudaPnrError::InvalidInput("total bytes overflow".into()))?;
        let headroom = 64 * 1024 * 1024; // ~64MB
        if !Self::will_fit(required, headroom) {
            if let Some((free, _)) = Self::device_mem_info() {
                return Err(CudaPnrError::OutOfMemory {
                    required,
                    free,
                    headroom,
                });
            } else {
                return Err(CudaPnrError::InvalidInput(
                    "insufficient device memory".into(),
                ));
            }
        }

        // Device allocations
        let mut d_prices: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(len) }?;
        let mut d_percs: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(percs.len()) }?;
        let mut d_periods: DeviceBuffer<i32> =
            unsafe { DeviceBuffer::uninitialized(periods.len()) }?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(out_elems) }?;

        // Direct H2D copies (avoid extra LockedBuffer host copy)
        unsafe {
            d_prices.async_copy_from(data_f32, &self.stream)?;
            d_percs.async_copy_from(&percs, &self.stream)?;
            d_periods.async_copy_from(&periods, &self.stream)?;
        }

        // Fallback baseline function (per-row maintenance)
        let func_baseline = self
            .module
            .get_function("percentile_nearest_rank_batch_f32")
            .map_err(|_| CudaPnrError::MissingKernelSymbol {
                name: "percentile_nearest_rank_batch_f32",
            })?;
        let block_x_baseline = match self.policy.batch {
            BatchKernelPolicy::Auto => 128,
            BatchKernelPolicy::OneD { block_x } => block_x,
        };

        // Launch per length-group
        let series_len_i = len as i32;
        let first_valid_i = first_valid as i32;
        let mut last_block_x_used: u32 = block_x_baseline;

        for (gi, &L) in lengths_axis.iter().enumerate() {
            let group_start = gi * group_rows;
            let group_size = group_rows;
            let warm = first_valid + L - 1;
            if warm >= len {
                // Degenerate: write NaNs via baseline path (keeps parity)
                let grid_x = ((group_size as u32) + block_x_baseline - 1) / block_x_baseline;
                // small scratch to satisfy baseline signature
                let mut d_scratch: DeviceBuffer<f32> =
                    unsafe { DeviceBuffer::uninitialized(group_size * L) }?;
                unsafe {
                    let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                    let mut periods_ptr = d_periods.as_device_ptr().add(group_start).as_raw();
                    let mut percs_ptr = d_percs.as_device_ptr().add(group_start).as_raw();
                    let mut series_len_i = series_len_i;
                    let mut combos_i = group_size as i32;
                    let mut first_valid_i = first_valid_i;
                    let mut out_ptr = d_out.as_device_ptr().add(group_start * len).as_raw();
                    let mut scratch_ptr = d_scratch.as_device_ptr().as_raw();
                    let mut max_len_i = L as i32;
                    let args: &mut [*mut c_void] = &mut [
                        &mut prices_ptr as *mut _ as *mut c_void,
                        &mut periods_ptr as *mut _ as *mut c_void,
                        &mut percs_ptr as *mut _ as *mut c_void,
                        &mut series_len_i as *mut _ as *mut c_void,
                        &mut combos_i as *mut _ as *mut c_void,
                        &mut first_valid_i as *mut _ as *mut c_void,
                        &mut out_ptr as *mut _ as *mut c_void,
                        &mut scratch_ptr as *mut _ as *mut c_void,
                        &mut max_len_i as *mut _ as *mut c_void,
                    ];
                    let grid: GridSize = (grid_x.max(1), 1, 1).into();
                    let block: BlockSize = (block_x_baseline, 1, 1).into();
                    self.validate_launch(grid_x.max(1), 1, 1, block_x_baseline, 1, 1)?;
                    self.stream.launch(&func_baseline, grid, block, 0, args)?;
                }
                continue;
            }

            if use_shared[gi] {
                // Shared-sort kernel path
                let threads = Self::next_pow2_u32(L as u32).min(256);
                let smem_bytes = Self::smem_bytes_for_len(L);
                let tcount = len - warm;
                let grid_x = (tcount as u32).min(80);
                unsafe {
                    let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                    let mut series_len_i = series_len_i;
                    let mut L_i = L as i32;
                    let mut percs_ptr = d_percs.as_device_ptr().add(group_start).as_raw();
                    let mut n_percs_i = group_size as i32;
                    let mut first_valid_i = first_valid_i;
                    let mut out_ptr = d_out.as_device_ptr().add(group_start * len).as_raw();
                    let args: &mut [*mut c_void] = &mut [
                        &mut prices_ptr as *mut _ as *mut c_void,
                        &mut series_len_i as *mut _ as *mut c_void,
                        &mut L_i as *mut _ as *mut c_void,
                        &mut percs_ptr as *mut _ as *mut c_void,
                        &mut n_percs_i as *mut _ as *mut c_void,
                        &mut first_valid_i as *mut _ as *mut c_void,
                        &mut out_ptr as *mut _ as *mut c_void,
                    ];
                    let grid: GridSize = (grid_x.max(1), 1, 1).into();
                    let block: BlockSize = (threads, 1, 1).into();
                    self.validate_launch(grid_x.max(1), 1, 1, threads, 1, 1)?;
                    self.stream
                        .launch(&func_shared, grid, block, smem_bytes as u32, args)?;
                    last_block_x_used = threads;
                }
            } else {
                // Baseline fallback for this group with minimal scratch
                let grid_x = ((group_size as u32) + block_x_baseline - 1) / block_x_baseline;
                let mut d_scratch: DeviceBuffer<f32> =
                    unsafe { DeviceBuffer::uninitialized(group_size * L) }?;
                unsafe {
                    let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                    let mut periods_ptr = d_periods.as_device_ptr().add(group_start).as_raw();
                    let mut percs_ptr = d_percs.as_device_ptr().add(group_start).as_raw();
                    let mut series_len_i = series_len_i;
                    let mut combos_i = group_size as i32;
                    let mut first_valid_i = first_valid_i;
                    let mut out_ptr = d_out.as_device_ptr().add(group_start * len).as_raw();
                    let mut scratch_ptr = d_scratch.as_device_ptr().as_raw();
                    let mut max_len_i = L as i32;
                    let args: &mut [*mut c_void] = &mut [
                        &mut prices_ptr as *mut _ as *mut c_void,
                        &mut periods_ptr as *mut _ as *mut c_void,
                        &mut percs_ptr as *mut _ as *mut c_void,
                        &mut series_len_i as *mut _ as *mut c_void,
                        &mut combos_i as *mut _ as *mut c_void,
                        &mut first_valid_i as *mut _ as *mut c_void,
                        &mut out_ptr as *mut _ as *mut c_void,
                        &mut scratch_ptr as *mut _ as *mut c_void,
                        &mut max_len_i as *mut _ as *mut c_void,
                    ];
                    let grid: GridSize = (grid_x.max(1), 1, 1).into();
                    let block: BlockSize = (block_x_baseline, 1, 1).into();
                    self.validate_launch(grid_x.max(1), 1, 1, block_x_baseline, 1, 1)?;
                    self.stream.launch(&func_baseline, grid, block, 0, args)?;
                }
            }
        }

        // Record selection (last used block size)
        let sel = BatchKernelSelected::OneD {
            block_x: last_block_x_used,
        };
        unsafe {
            (*(self as *const _ as *mut CudaPercentileNearestRank)).last_batch = Some(sel);
        }
        static ONCE: AtomicBool = AtomicBool::new(false);
        let mut printed = false;
        Self::maybe_log(Some(sel), "batch", &ONCE, &mut printed);

        self.stream.synchronize()?;
        Ok((
            DeviceArrayF32 {
                buf: d_out,
                rows: combos.len(),
                cols: len,
            },
            combos,
        ))
    }

    // -------- Many-series × one-param (time-major) --------
    pub fn pnr_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        length: usize,
        percentage: f64,
    ) -> Result<DeviceArrayF32, CudaPnrError> {
        if cols == 0 || rows == 0 {
            return Err(CudaPnrError::InvalidInput("empty shape".into()));
        }
        let total_elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaPnrError::InvalidInput("cols*rows overflow".into()))?;
        if data_tm_f32.len() != total_elems {
            return Err(CudaPnrError::InvalidInput(
                "time-major input shape mismatch".into(),
            ));
        }
        if length == 0 || length > rows {
            return Err(CudaPnrError::InvalidInput("invalid length".into()));
        }

        // Per-series first_valid (index of first non-NaN)
        let mut firsts = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = -1i32;
            for t in 0..rows {
                let v = data_tm_f32[t * cols + s];
                if !v.is_nan() {
                    fv = t as i32;
                    break;
                }
            }
            if fv < 0 {
                return Err(CudaPnrError::InvalidInput(
                    "all values NaN for a series".into(),
                ));
            }
            if fv < 0 {
                return Err(CudaPnrError::InvalidInput(
                    "all values NaN for a series".into(),
                ));
            }
            if (rows as i32 - fv) < (length as i32) {
                return Err(CudaPnrError::InvalidInput(
                    "not enough valid data for a series".into(),
                ));
                return Err(CudaPnrError::InvalidInput(
                    "not enough valid data for a series".into(),
                ));
            }
            firsts[s] = fv;
        }

        let prices_bytes = total_elems
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or_else(|| CudaPnrError::InvalidInput("prices bytes overflow".into()))?;
        let out_bytes = total_elems
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or_else(|| CudaPnrError::InvalidInput("output bytes overflow".into()))?;
        let firsts_bytes = cols
            .checked_mul(core::mem::size_of::<i32>())
            .ok_or_else(|| CudaPnrError::InvalidInput("firsts bytes overflow".into()))?;
        let scratch_elems = cols
            .checked_mul(length)
            .ok_or_else(|| CudaPnrError::InvalidInput("scratch elements overflow".into()))?;
        let scratch_bytes = scratch_elems
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or_else(|| CudaPnrError::InvalidInput("scratch bytes overflow".into()))?;
        let required = prices_bytes
            .checked_add(out_bytes)
            .and_then(|x| x.checked_add(firsts_bytes))
            .and_then(|x| x.checked_add(scratch_bytes))
            .ok_or_else(|| CudaPnrError::InvalidInput("total bytes overflow".into()))?;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            if let Some((free, _)) = Self::device_mem_info() {
                return Err(CudaPnrError::OutOfMemory {
                    required,
                    free,
                    headroom,
                });
            } else {
                return Err(CudaPnrError::InvalidInput(
                    "insufficient free VRAM for workload".into(),
                ));
            }
        }

        let mut d_prices: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(total_elems) }?;
        let mut d_firsts: DeviceBuffer<i32> =
            unsafe { DeviceBuffer::uninitialized(cols) }?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(total_elems) }?;
        let mut d_scratch: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(scratch_elems) }?;

        unsafe {
            d_prices.async_copy_from(data_tm_f32, &self.stream)?;
            d_firsts.async_copy_from(&firsts, &self.stream)?;
        }

        let func = self
            .module
            .get_function("percentile_nearest_rank_many_series_one_param_time_major_f32")
            .map_err(|_| CudaPnrError::MissingKernelSymbol {
                name: "percentile_nearest_rank_many_series_one_param_time_major_f32",
            })?;

        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => 128,
            ManySeriesKernelPolicy::OneD { block_x } => block_x,
        };
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut length_i = length as i32;
            let mut perc_f = percentage as f32;
            let mut firsts_ptr = d_firsts.as_device_ptr().as_raw();
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let mut scratch_ptr = d_scratch.as_device_ptr().as_raw();
            let mut max_len_i = length as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut length_i as *mut _ as *mut c_void,
                &mut perc_f as *mut _ as *mut c_void,
                &mut firsts_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
                &mut scratch_ptr as *mut _ as *mut c_void,
                &mut max_len_i as *mut _ as *mut c_void,
            ];
            self.validate_launch(grid_x.max(1), 1, 1, block_x, 1, 1)?;
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        let sel = ManySeriesKernelSelected::OneD { block_x };
        unsafe {
            (*(self as *const _ as *mut CudaPercentileNearestRank)).last_many = Some(sel);
        }
        unsafe {
            (*(self as *const _ as *mut CudaPercentileNearestRank)).last_many = Some(sel);
        }
        static ONCE: AtomicBool = AtomicBool::new(false);
        let mut printed = false;
        Self::maybe_log(Some(sel), "many-series", &ONCE, &mut printed);

        self.stream.synchronize()?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }
}

// ---------------- Bench integration ----------------
#[cfg(test)]
mod benches_dummy_compile_only {}

#[cfg(feature = "cuda")]
pub mod benches {
    use super::*;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};

    struct PnrBatchState {
        cuda: CudaPercentileNearestRank,
        d_prices: DeviceBuffer<f32>,
        d_periods: DeviceBuffer<i32>,
        d_percs: DeviceBuffer<f32>,
        d_scratch: DeviceBuffer<f32>,
        d_out: DeviceBuffer<f32>,
        len: usize,
        n_combos: usize,
        first_valid: i32,
        max_len: i32,
        block: BlockSize,
        grid: GridSize,
    }
    impl CudaBenchState for PnrBatchState {
        fn launch(&mut self) {
            unsafe {
                let func = self
                    .cuda
                    .module
                    .get_function("percentile_nearest_rank_batch_f32")
                    .expect("get_function pnr batch");
                let mut prices_ptr = self.d_prices.as_device_ptr().as_raw();
                let mut periods_ptr = self.d_periods.as_device_ptr().as_raw();
                let mut percs_ptr = self.d_percs.as_device_ptr().as_raw();
                let mut series_len_i = self.len as i32;
                let mut combos_i = self.n_combos as i32;
                let mut first_valid_i = self.first_valid;
                let mut out_ptr = self.d_out.as_device_ptr().as_raw();
                let mut scratch_ptr = self.d_scratch.as_device_ptr().as_raw();
                let mut max_len_i = self.max_len;
                let args: &mut [*mut c_void] = &mut [
                    &mut prices_ptr as *mut _ as *mut c_void,
                    &mut periods_ptr as *mut _ as *mut c_void,
                    &mut percs_ptr as *mut _ as *mut c_void,
                    &mut series_len_i as *mut _ as *mut c_void,
                    &mut combos_i as *mut _ as *mut c_void,
                    &mut first_valid_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                    &mut scratch_ptr as *mut _ as *mut c_void,
                    &mut max_len_i as *mut _ as *mut c_void,
                ];
                let _ = self
                    .cuda
                    .stream
                    .launch(&func, self.grid, self.block, 0, args);
                let _ = self.cuda.stream.synchronize();
            }
        }
    }

    struct PnrManyState {
        cuda: CudaPercentileNearestRank,
        d_prices: DeviceBuffer<f32>,
        d_firsts: DeviceBuffer<i32>,
        d_scratch: DeviceBuffer<f32>,
        d_out: DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        length: i32,
        perc: f32,
        block: BlockSize,
        grid: GridSize,
    }
    impl CudaBenchState for PnrManyState {
        fn launch(&mut self) {
            unsafe {
                let func = self
                    .cuda
                    .module
                    .get_function("percentile_nearest_rank_many_series_one_param_time_major_f32")
                    .expect("get_function pnr many");
                let mut prices_ptr = self.d_prices.as_device_ptr().as_raw();
                let mut cols_i = self.cols as i32;
                let mut rows_i = self.rows as i32;
                let mut length_i = self.length;
                let mut perc_f = self.perc;
                let mut firsts_ptr = self.d_firsts.as_device_ptr().as_raw();
                let mut out_ptr = self.d_out.as_device_ptr().as_raw();
                let mut scratch_ptr = self.d_scratch.as_device_ptr().as_raw();
                let mut max_len_i = self.length;
                let args: &mut [*mut c_void] = &mut [
                    &mut prices_ptr as *mut _ as *mut c_void,
                    &mut cols_i as *mut _ as *mut c_void,
                    &mut rows_i as *mut _ as *mut c_void,
                    &mut length_i as *mut _ as *mut c_void,
                    &mut perc_f as *mut _ as *mut c_void,
                    &mut firsts_ptr as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                    &mut scratch_ptr as *mut _ as *mut c_void,
                    &mut max_len_i as *mut _ as *mut c_void,
                ];
                let _ = self
                    .cuda
                    .stream
                    .launch(&func, self.grid, self.block, 0, args);
                let _ = self.cuda.stream.synchronize();
            }
        }
    }

    // Non-capturing prep functions for benches
    fn prep_pnr_batch() -> Box<dyn CudaBenchState> {
        let len = 100_000usize;
        let prices = gen_series(len);
        let periods: Vec<i32> = (10..=50)
            .step_by(10)
            .flat_map(|l| std::iter::repeat(l as i32).take(3))
            .collect();
        let percs: Vec<f32> = [25.0f32, 50.0, 75.0].repeat(5);
        let n_combos = periods.len();
        let first_valid = prices.iter().position(|v| !v.is_nan()).unwrap_or(0) as i32;
        let max_len = *periods.iter().max().unwrap_or(&10);

        let cuda = CudaPercentileNearestRank::new(0).expect("cuda ctx");
        let block_x: u32 = 128;
        let grid_x = ((n_combos as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        let mut d_prices: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(len) }.unwrap();
        let mut d_periods: DeviceBuffer<i32> = unsafe { DeviceBuffer::uninitialized(n_combos) }.unwrap();
        let mut d_percs: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(n_combos) }.unwrap();
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(n_combos * len) }.unwrap();
        let mut d_scratch: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_combos * (max_len as usize)) }.unwrap();

        let hp = LockedBuffer::from_slice(&prices).unwrap();
        let hperiods = LockedBuffer::from_slice(&periods).unwrap();
        let hpercs = LockedBuffer::from_slice(&percs).unwrap();
        unsafe {
            d_prices.async_copy_from(hp.as_slice(), &cuda.stream).unwrap();
            d_periods.async_copy_from(hperiods.as_slice(), &cuda.stream).unwrap();
            d_percs.async_copy_from(hpercs.as_slice(), &cuda.stream).unwrap();
        }
        let _ = cuda.stream.synchronize();

        Box::new(PnrBatchState {
            cuda,
            d_prices,
            d_periods,
            d_percs,
            d_scratch,
            d_out,
            len,
            n_combos,
            first_valid,
            max_len,
            block,
            grid,
        })
    }

    fn prep_pnr_many() -> Box<dyn CudaBenchState> {
        let cols = 128usize;
        let rows = 4096usize;
        let prices = gen_time_major_prices(cols, rows);
        let length = 21i32;
        let perc = 50.0f32;
        let mut firsts = vec![0i32; cols];
        for s in 0..cols {
            firsts[s] = (0..rows)
                .find(|&t| !prices[t * cols + s].is_nan())
                .unwrap_or(0) as i32;
        }
        let cuda = CudaPercentileNearestRank::new(0).expect("cuda ctx");
        let block_x: u32 = 128;
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        let mut d_prices: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }.unwrap();
        let mut d_firsts: DeviceBuffer<i32> = unsafe { DeviceBuffer::uninitialized(cols) }.unwrap();
        let mut d_scratch: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(cols * (length as usize)) }.unwrap();
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }.unwrap();

        let hp = LockedBuffer::from_slice(&prices).unwrap();
        let hf = LockedBuffer::from_slice(&firsts).unwrap();
        unsafe {
            d_prices.async_copy_from(hp.as_slice(), &cuda.stream).unwrap();
            d_firsts.async_copy_from(hf.as_slice(), &cuda.stream).unwrap();
        }
        let _ = cuda.stream.synchronize();

        Box::new(PnrManyState {
            cuda,
            d_prices,
            d_firsts,
            d_scratch,
            d_out,
            cols,
            rows,
            length,
            perc,
            block,
            grid,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        let mut v = Vec::new();

        v.push(
            CudaBenchScenario::new(
                "pnr",
                "one_series_many_params",
                "pnr/batch",
                "pnr_batch/100k",
                prep_pnr_batch,
            )
            .with_mem_required((100_000 * 4) + (15 * 100_000 * 4) + (15 * 64 * 4)),
        );

        v.push(
            CudaBenchScenario::new(
                "pnr",
                "many_series_one_param",
                "pnr/many_series",
                "pnr_many/cols=128,rows=4096",
                prep_pnr_many,
            )
            .with_mem_required((128 * 4096 * 4) + (128 * 4096 * 4) + (128 * 21 * 4)),
        );

        v
    }
}
