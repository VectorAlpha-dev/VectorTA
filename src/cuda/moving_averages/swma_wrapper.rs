//! CUDA wrapper for SWMA (Symmetric Weighted Moving Average) kernels.
//!
//! Parity with ALMA wrapper for API shape, policy, PTX JIT options, VRAM
//! checks, chunking, and stream settings.
//! - Batch (one-series × many-params): kernel computes normalized triangular
//!   weights on device per block using shared memory.
//! - Many-series × one-param (time-major): host precomputes normalized weights
//!   and kernel consumes them from shared memory per block.

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::swma::{SwmaBatchRange, SwmaParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::{c_void, CString};
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug)]
pub enum CudaSwmaError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaSwmaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaSwmaError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaSwmaError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for CudaSwmaError {}

// -------- Kernel selection policy (ALMA-style) --------

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    Plain { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 },
    TwoD { block_x: u32, series_per_block: u32 },
}

#[derive(Clone, Copy, Debug)]
pub struct CudaSwmaPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaSwmaPolicy {
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
    TwoD { block_x: u32, series_per_block: u32 },
}

pub struct CudaSwma {
    module: Module,
    stream: Stream,
    _context: Context,
    device_id: u32,
    policy: CudaSwmaPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
    // Kernel config awareness (kept minimal and defaulted for safety)
    outs_per_thread: u32,
    series_per_block: u32,
    max_period_const: usize,
    has_const_weights: bool,
}

impl CudaSwma {
    #[inline]
    fn parse_env_u32(key: &str, default_: u32) -> u32 {
        std::env::var(key)
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(default_)
    }
    #[inline]
    fn parse_env_usize(key: &str, default_: usize) -> usize {
        std::env::var(key)
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(default_)
    }
    pub fn new(device_id: usize) -> Result<Self, CudaSwmaError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaSwmaError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaSwmaError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaSwmaError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/swma_kernel.ptx"));
        // Prefer context-derived target with a stable opt level; fall back to simpler JIT
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => match Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]) {
                Ok(m) => m,
                Err(_) => {
                    Module::from_ptx(ptx, &[]).map_err(|e| CudaSwmaError::Cuda(e.to_string()))?
                }
            },
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaSwmaError::Cuda(e.to_string()))?;

        // Try to discover the __constant__ symbol via safe cust API.
        // Kernel defaults SWMA_MAX_PERIOD=4096 and SWMA_USE_CONST_WEIGHTS=1.
        // If the symbol is present and sized as expected, enable constant path.
        let (has_const_weights, max_period_const) = {
            const SWMA_MAX_PERIOD_RS: usize = 4096;
            let name = CString::new("c_swma_weights").unwrap();
            match module.get_global::<[f32; SWMA_MAX_PERIOD_RS]>(&name) {
                Ok(_) => (true, SWMA_MAX_PERIOD_RS),
                Err(_) => (false, 0),
            }
        };

        // Defaults must match compiled kernel (see SWMA_* defines in the .cu)
        let outs_per_thread = Self::parse_env_u32("SWMA_OUTS_PER_THREAD", 2);
        let series_per_block = Self::parse_env_u32("SWMA_SERIES_PER_BLOCK", 8);
        let max_period_const_env = Self::parse_env_usize(
            "SWMA_MAX_PERIOD",
            if has_const_weights {
                max_period_const
            } else {
                4096
            },
        );

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaSwmaPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
            outs_per_thread,
            series_per_block,
            max_period_const: max_period_const_env,
            has_const_weights,
        })
    }

    pub fn new_with_policy(
        device_id: usize,
        policy: CudaSwmaPolicy,
    ) -> Result<Self, CudaSwmaError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }
    pub fn set_policy(&mut self, policy: CudaSwmaPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaSwmaPolicy {
        &self.policy
    }

    #[inline]
    fn maybe_log_batch_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                let per_scn =
                    std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scn || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] SWMA batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaSwma)).debug_batch_logged = true;
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
                let per_scn =
                    std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scn || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] SWMA many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaSwma)).debug_many_logged = true;
                }
            }
        }
    }

    // ---------- VRAM helpers ----------
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
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> bool {
        if !Self::mem_check_enabled() {
            return true;
        }
        if let Some((free, _)) = Self::device_mem_info() {
            required_bytes.saturating_add(headroom_bytes) <= free
        } else {
            true
        }
    }

    fn expand_periods(range: &SwmaBatchRange) -> Vec<usize> {
        let (start, end, step) = range.period;
        if step == 0 || start == end {
            return vec![start];
        }
        if start > end {
            return Vec::new();
        }
        (start..=end).step_by(step).collect()
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &SwmaBatchRange,
    ) -> Result<(Vec<usize>, usize, usize, usize), CudaSwmaError> {
        if data_f32.is_empty() {
            return Err(CudaSwmaError::InvalidInput("empty data".into()));
        }

        let first_valid = data_f32
            .iter()
            .position(|x| !x.is_nan())
            .ok_or_else(|| CudaSwmaError::InvalidInput("all values are NaN".into()))?;

        let periods = Self::expand_periods(sweep);
        if periods.is_empty() {
            return Err(CudaSwmaError::InvalidInput("no periods in sweep".into()));
        }

        let len = data_f32.len();
        let mut max_p = 0usize;
        for &period in &periods {
            if period == 0 {
                return Err(CudaSwmaError::InvalidInput("period must be > 0".into()));
            }
            if period > len {
                return Err(CudaSwmaError::InvalidInput(format!(
                    "period {} exceeds data length {}",
                    period, len
                )));
            }
            if len - first_valid < period {
                return Err(CudaSwmaError::InvalidInput(format!(
                    "not enough valid data: needed {}, have {}",
                    period,
                    len - first_valid
                )));
            }
            max_p = max_p.max(period);
        }

        Ok((periods, first_valid, len, max_p))
    }

    #[inline]
    fn upload_const_weights(&self, period: usize, weights: &[f32]) -> Result<(), CudaSwmaError> {
        if !self.has_const_weights {
            return Ok(());
        }
        if period > self.max_period_const {
            return Err(CudaSwmaError::InvalidInput(format!(
                "period {} exceeds SWMA_MAX_PERIOD {} compiled in kernel",
                period, self.max_period_const
            )));
        }
        // Build a fixed-size host array matching the constant symbol, zero-padded beyond period
        const SWMA_MAX_PERIOD_RS: usize = 4096;
        let mut host = [0f32; SWMA_MAX_PERIOD_RS];
        host[..period].copy_from_slice(&weights[..period]);
        let name = CString::new("c_swma_weights").unwrap();
        let mut symbol = self
            .module
            .get_global::<[f32; SWMA_MAX_PERIOD_RS]>(&name)
            .map_err(|e| CudaSwmaError::Cuda(e.to_string()))?;
        symbol
            .copy_from(&host)
            .map_err(|e| CudaSwmaError::Cuda(e.to_string()))?;
        Ok(())
    }

    fn compute_weights(period: usize) -> Vec<f32> {
        let mut weights = vec![0.0f32; period];
        if period == 0 {
            return weights;
        }
        let norm = if period <= 2 {
            period as f32
        } else if period % 2 == 0 {
            let half = (period / 2) as f32;
            half * (half + 1.0f32)
        } else {
            let half_plus = ((period + 1) / 2) as f32;
            half_plus * half_plus
        };
        let inv_norm = 1.0f32 / norm.max(f32::EPSILON);
        for idx in 0..period {
            let left = idx + 1;
            let right = period - idx;
            let w = left.min(right) as f32;
            weights[idx] = w * inv_norm;
        }
        weights
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &SwmaParams,
    ) -> Result<(Vec<i32>, usize), CudaSwmaError> {
        if cols == 0 || rows == 0 {
            return Err(CudaSwmaError::InvalidInput(
                "num_series or series_len is zero".into(),
            ));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaSwmaError::InvalidInput(format!(
                "data length {} != cols*rows {}",
                data_tm_f32.len(),
                cols * rows
            )));
        }

        let period = params.period.unwrap_or(5);
        if period == 0 {
            return Err(CudaSwmaError::InvalidInput("period must be > 0".into()));
        }

        let mut first_valids = vec![0i32; cols];
        for series in 0..cols {
            let mut fv = None;
            for row in 0..rows {
                let idx = row * cols + series;
                let v = data_tm_f32[idx];
                if !v.is_nan() {
                    fv = Some(row);
                    break;
                }
            }
            let fv_row = fv.ok_or_else(|| {
                CudaSwmaError::InvalidInput(format!("series {} contains only NaNs", series))
            })?;
            if rows - fv_row < period {
                return Err(CudaSwmaError::InvalidInput(format!(
                    "series {} lacks enough valid data: needed {}, have {}",
                    series,
                    period,
                    rows - fv_row
                )));
            }
            first_valids[series] = fv_row as i32;
        }

        Ok((first_valids, period))
    }

    fn launch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_warms: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        max_period: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSwmaError> {
        if series_len == 0 {
            return Err(CudaSwmaError::InvalidInput("series_len is zero".into()));
        }
        if n_combos == 0 {
            return Err(CudaSwmaError::InvalidInput("no parameter combos".into()));
        }
        if max_period == 0 {
            return Err(CudaSwmaError::InvalidInput("max_period is zero".into()));
        }
        if series_len > i32::MAX as usize
            || n_combos > i32::MAX as usize
            || max_period > i32::MAX as usize
        {
            return Err(CudaSwmaError::InvalidInput(
                "series_len, n_combos, or max_period exceed i32::MAX".into(),
            ));
        }

        let func = self
            .module
            .get_function("swma_batch_f32")
            .map_err(|e| CudaSwmaError::Cuda(e.to_string()))?;

        // Policy: block size along time dimension
        // Match kernel defaults (see SWMA_BLOCK_X and SWMA_OUTS_PER_THREAD in swma_kernel.cu)
        // OUTS_PER_THREAD is 2 in the kernel by default.
        let block_x: u32 = match self.policy.batch {
            BatchKernelPolicy::Auto => 128,
            BatchKernelPolicy::Plain { block_x } => block_x.max(1),
        };
        // Introspection
        unsafe {
            let this = self as *const _ as *mut CudaSwma;
            (*this).last_batch = Some(BatchKernelSelected::Plain { block_x });
        }
        self.maybe_log_batch_debug();

        // Grid tiles along time with outs_per_thread outputs per thread
        let outs_per_thread = self.outs_per_thread.max(1);
        let tile_out = (block_x as usize) * (outs_per_thread as usize);
        let grid_x = ((series_len + tile_out - 1) / tile_out) as u32;
        let block: BlockSize = (block_x, 1, 1).into();
        // Dynamic shared memory: weights (max_period) + price tile (block_x*outs_per_thread + max_period - 1)
        // Use period = max_period to conservatively size the tile apron.
        let shared_elems = (max_period - 1) + tile_out + max_period;
        let shared_bytes = (shared_elems * std::mem::size_of::<f32>()) as u32;

        // Chunk grid.y to <= 65_535
        const MAX_GRID_Y: usize = 65_535;
        let mut launched = 0usize;
        while launched < n_combos {
            let this_chunk = (n_combos - launched).min(MAX_GRID_Y);
            let grid: GridSize = (grid_x.max(1), this_chunk as u32, 1).into();
            unsafe {
                let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                // Offset params to chunk
                let mut periods_ptr = d_periods.as_device_ptr().add(launched).as_raw();
                let mut warms_ptr = d_warms.as_device_ptr().add(launched).as_raw();
                let mut series_len_i = series_len as i32;
                let mut n_combos_i = this_chunk as i32;
                let mut max_period_i = max_period as i32;
                // Offset output by rows already launched
                let mut out_ptr = d_out.as_device_ptr().add(launched * series_len).as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut prices_ptr as *mut _ as *mut c_void,
                    &mut periods_ptr as *mut _ as *mut c_void,
                    &mut warms_ptr as *mut _ as *mut c_void,
                    &mut series_len_i as *mut _ as *mut c_void,
                    &mut n_combos_i as *mut _ as *mut c_void,
                    &mut max_period_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream
                    .launch(&func, grid, block, shared_bytes, args)
                    .map_err(|e| CudaSwmaError::Cuda(e.to_string()))?;
            }
            launched += this_chunk;
        }

        Ok(())
    }

    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_weights_opt: Option<&DeviceBuffer<f32>>,
        d_first_valids: &DeviceBuffer<i32>,
        period: usize,
        cols: usize,
        rows: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSwmaError> {
        if period == 0 || cols == 0 || rows == 0 {
            return Err(CudaSwmaError::InvalidInput(
                "period, num_series, and series_len must be positive".into(),
            ));
        }
        if period > i32::MAX as usize || cols > i32::MAX as usize || rows > i32::MAX as usize {
            return Err(CudaSwmaError::InvalidInput(
                "period, num_series, or series_len exceed i32::MAX".into(),
            ));
        }

        let func = self
            .module
            .get_function("swma_multi_series_one_param_f32")
            .map_err(|e| CudaSwmaError::Cuda(e.to_string()))?;

        // Match kernel defaults for 2D mapping (see SWMA_BLOCK_X and SWMA_SERIES_PER_BLOCK)
        let (tx, ty, selected): (u32, u32, ManySeriesKernelSelected) = match self.policy.many_series
        {
            ManySeriesKernelPolicy::Auto => (
                128,
                self.series_per_block.max(1),
                ManySeriesKernelSelected::TwoD {
                    block_x: 128,
                    series_per_block: self.series_per_block.max(1),
                },
            ),
            // Preserve legacy OneD knob if explicitly requested
            ManySeriesKernelPolicy::OneD { block_x } => (
                block_x.max(1),
                1,
                ManySeriesKernelSelected::OneD {
                    block_x: block_x.max(1),
                },
            ),
            ManySeriesKernelPolicy::TwoD {
                block_x,
                series_per_block,
            } => (
                block_x.max(1),
                series_per_block.max(1),
                ManySeriesKernelSelected::TwoD {
                    block_x: block_x.max(1),
                    series_per_block: series_per_block.max(1),
                },
            ),
        };
        // Introspection
        unsafe {
            let this = self as *const _ as *mut CudaSwma;
            (*this).last_many = Some(selected);
        }
        self.maybe_log_many_debug();

        let grid_x = ((rows as u32) + tx - 1) / tx;
        let grid_y = ((cols as u32) + ty - 1) / ty;
        let grid: GridSize = (grid_x.max(1), grid_y.max(1), 1).into();
        let block: BlockSize = (tx, ty, 1).into();

        // Dynamic shared memory: per-series tiles and optional shared weight staging
        let per_series_tile = (tx as usize) + period - 1;
        let mut shared_floats = per_series_tile * (ty as usize);
        if !self.has_const_weights {
            shared_floats += period;
        }
        let shared_bytes = (shared_floats * std::mem::size_of::<f32>()) as u32;

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut weights_ptr: u64 = if let Some(w) = d_weights_opt {
                w.as_device_ptr().as_raw()
            } else {
                0u64
            };
            let mut period_i = period as i32;
            let mut num_series_i = cols as i32;
            let mut series_len_i = rows as i32;
            let mut first_valids_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut weights_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut first_valids_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, shared_bytes, args)
                .map_err(|e| CudaSwmaError::Cuda(e.to_string()))?;
        }

        Ok(())
    }

    pub fn swma_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_warms: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        max_period: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSwmaError> {
        self.launch_kernel(
            d_prices, d_periods, d_warms, series_len, n_combos, max_period, d_out,
        )
    }

    pub fn swma_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &SwmaBatchRange,
    ) -> Result<DeviceArrayF32, CudaSwmaError> {
        let (periods, first_valid, series_len, max_period) =
            Self::prepare_batch_inputs(data_f32, sweep)?;
        let n_combos = periods.len();

        // VRAM estimate and check (64MB headroom)
        let prices_bytes = series_len * std::mem::size_of::<f32>();
        let periods_bytes = n_combos * std::mem::size_of::<i32>();
        let warm_bytes = n_combos * std::mem::size_of::<i32>();
        let out_bytes = n_combos * series_len * std::mem::size_of::<f32>();
        let required = prices_bytes + periods_bytes + warm_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaSwmaError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        let periods_i32: Vec<i32> = periods.iter().map(|&p| p as i32).collect();
        let warms_i32: Vec<i32> = periods
            .iter()
            .map(|&p| (first_valid + p - 1) as i32)
            .collect();

        let d_prices =
            DeviceBuffer::from_slice(data_f32).map_err(|e| CudaSwmaError::Cuda(e.to_string()))?;
        let d_periods = DeviceBuffer::from_slice(&periods_i32)
            .map_err(|e| CudaSwmaError::Cuda(e.to_string()))?;
        let d_warms =
            DeviceBuffer::from_slice(&warms_i32).map_err(|e| CudaSwmaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }
                .map_err(|e| CudaSwmaError::Cuda(e.to_string()))?;

        self.launch_kernel(
            &d_prices, &d_periods, &d_warms, series_len, n_combos, max_period, &mut d_out,
        )?;
        self.stream
            .synchronize()
            .map_err(|e| CudaSwmaError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: series_len,
        })
    }

    fn run_many_series_kernel(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        first_valids: &[i32],
        period: usize,
    ) -> Result<DeviceArrayF32, CudaSwmaError> {
        let weights = Self::compute_weights(period);
        // If constant weights are available, upload once per call; otherwise we'll allocate a device buffer
        if self.has_const_weights {
            self.upload_const_weights(period, &weights)?;
        }

        let d_prices = DeviceBuffer::from_slice(data_tm_f32)
            .map_err(|e| CudaSwmaError::Cuda(e.to_string()))?;
        let d_first_valids = DeviceBuffer::from_slice(first_valids)
            .map_err(|e| CudaSwmaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaSwmaError::Cuda(e.to_string()))?;

        // Only pass a device weights pointer when not using constant memory
        let d_weights_opt = if self.has_const_weights {
            None
        } else {
            Some(
                DeviceBuffer::from_slice(&weights)
                    .map_err(|e| CudaSwmaError::Cuda(e.to_string()))?,
            )
        };

        self.launch_many_series_kernel(
            &d_prices,
            d_weights_opt.as_ref(),
            &d_first_valids,
            period,
            cols,
            rows,
            &mut d_out,
        )?;
        self.stream
            .synchronize()
            .map_err(|e| CudaSwmaError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    pub fn swma_multi_series_one_param_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_weights: &DeviceBuffer<f32>,
        period: i32,
        num_series: i32,
        series_len: i32,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSwmaError> {
        if period <= 0 || num_series <= 0 || series_len <= 0 {
            return Err(CudaSwmaError::InvalidInput(
                "period, num_series, and series_len must be positive".into(),
            ));
        }
        // If constant-memory is available, stage device weights to host and upload to constant
        if self.has_const_weights {
            let mut host_w = vec![0f32; period as usize];
            d_weights
                .copy_to(&mut host_w)
                .map_err(|e| CudaSwmaError::Cuda(e.to_string()))?;
            self.upload_const_weights(period as usize, &host_w)?;
        }
        self.launch_many_series_kernel(
            d_prices_tm,
            if self.has_const_weights {
                None
            } else {
                Some(d_weights)
            },
            d_first_valids,
            period as usize,
            num_series as usize,
            series_len as usize,
            d_out_tm,
        )
    }

    pub fn swma_multi_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &SwmaParams,
    ) -> Result<DeviceArrayF32, CudaSwmaError> {
        let (first_valids, period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;

        // VRAM estimate: prices + [weights if not const] + first_valids + out
        let prices_bytes = cols * rows * std::mem::size_of::<f32>();
        let weights_bytes = if self.has_const_weights {
            0
        } else {
            period * std::mem::size_of::<f32>()
        };
        let first_valids_bytes = cols * std::mem::size_of::<i32>();
        let out_bytes = cols * rows * std::mem::size_of::<f32>();
        let required = prices_bytes + weights_bytes + first_valids_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaSwmaError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        self.run_many_series_kernel(data_tm_f32, cols, rows, &first_valids, period)
    }

    pub fn swma_multi_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &SwmaParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaSwmaError> {
        if out_tm.len() != cols * rows {
            return Err(CudaSwmaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out_tm.len(),
                cols * rows
            )));
        }
        let (first_valids, period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        let arr = self.run_many_series_kernel(data_tm_f32, cols, rows, &first_valids, period)?;
        arr.buf
            .copy_to(out_tm)
            .map_err(|e| CudaSwmaError::Cuda(e.to_string()))?;
        Ok(())
    }
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::define_ma_period_benches;

    define_ma_period_benches!(
        swma_benches,
        CudaSwma,
        crate::indicators::moving_averages::swma::SwmaBatchRange,
        crate::indicators::moving_averages::swma::SwmaParams,
        swma_batch_dev,
        swma_multi_series_one_param_time_major_dev,
        crate::indicators::moving_averages::swma::SwmaBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1)
        },
        crate::indicators::moving_averages::swma::SwmaParams { period: Some(64) },
        "swma",
        "swma"
    );
    pub use swma_benches::bench_profiles;
}
