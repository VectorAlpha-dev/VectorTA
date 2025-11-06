//! CUDA wrapper for MAAQ (Moving Average Adaptive Q) kernels.
//!
//! Aligned with the ALMA wrapper design/policy:
//! - Robust PTX loading with context-targeted JIT and opt-level fallbacks
//! - Policy enums for explicit kernel selection (batch/many-series)
//! - VRAM estimation with headroom; fail early when insufficient
//! - Chunking over parameter combinations (<= 65_535 per launch)
//! - NON_BLOCKING stream; async copies and pinned buffers where helpful
//!
//! Kernels expected:
//! - "maaq_batch_f32"                          // one-series × many-params (recurrence)
//! - "maaq_multi_series_one_param_f32"         // many-series × one-param (time-major)
//!
//! Notes:
//! - MAAQ is a recurrence/IIR style indicator. Each thread/block scans time
//!   sequentially per parameter-combo or per series. Tiling only affects launch
//!   geometry; math remains per-thread sequential.

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::maaq::{expand_grid, MaaqBatchRange, MaaqParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

// -------- Kernel selection policy (kept minimal for a recurrence kernel) --------

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    // One thread per combo, sequential over time
    Plain { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    // One thread per series (1D). 2D variants not provided for MAAQ currently.
    OneD { block_x: u32 },
}

impl Default for BatchKernelPolicy {
    fn default() -> Self {
        BatchKernelPolicy::Auto
    }
}

impl Default for ManySeriesKernelPolicy {
    fn default() -> Self {
        ManySeriesKernelPolicy::Auto
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaMaaqPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

#[derive(Debug)]
pub enum CudaMaaqError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaMaaqError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaMaaqError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaMaaqError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for CudaMaaqError {}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected {
    Plain { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}

pub struct CudaMaaq {
    module: Module,
    stream: Stream,
    _context: Context, // keep context alive
    device_id: u32,
    policy: CudaMaaqPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaMaaq {
    pub fn new(device_id: usize) -> Result<Self, CudaMaaqError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaMaaqError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaMaaqError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaMaaqError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/maaq_kernel.ptx"));
        // Prefer context-targeted JIT; allow default optimizer (typically O4)
        let jit_opts = &[ModuleJitOption::DetermineTargetFromContext];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => {
                if let Ok(m) = Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext])
                {
                    m
                } else {
                    Module::from_ptx(ptx, &[]).map_err(|e| CudaMaaqError::Cuda(e.to_string()))?
                }
            }
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaMaaqError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaMaaqPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn new_with_policy(
        device_id: usize,
        policy: CudaMaaqPolicy,
    ) -> Result<Self, CudaMaaqError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }
    pub fn set_policy(&mut self, policy: CudaMaaqPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaMaaqPolicy {
        &self.policy
    }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch
    }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many
    }

    #[inline]
    fn maybe_log_batch_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                let per_scenario =
                    std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scenario || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] MAAQ batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaMaaq)).debug_batch_logged = true;
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
                let per_scenario =
                    std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scenario || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] MAAQ many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaMaaq)).debug_many_logged = true;
                }
            }
        }
    }

    // ---------- Utilities ----------

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
    fn chunk_pairs(total: usize, chunk_max: usize) -> impl Iterator<Item = (usize, usize)> {
        (0..total).step_by(chunk_max).map(move |start| {
            let len = (total - start).min(chunk_max);
            (start, len)
        })
    }

    #[inline]
    fn grid_x_limit(&self) -> usize {
        Device::get_device(self.device_id)
            .ok()
            .and_then(|d| {
                d.get_attribute(cust::device::DeviceAttribute::MaxGridDimX)
                    .ok()
            })
            .map(|v| v as usize)
            .unwrap_or(65_535)
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &MaaqBatchRange,
    ) -> Result<(Vec<MaaqParams>, usize, usize, usize), CudaMaaqError> {
        if data_f32.is_empty() {
            return Err(CudaMaaqError::InvalidInput("empty data".into()));
        }
        let first_valid = data_f32
            .iter()
            .position(|x| !x.is_nan())
            .ok_or_else(|| CudaMaaqError::InvalidInput("all values are NaN".into()))?;

        let combos = expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaMaaqError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        let len = data_f32.len();
        let mut max_period = 0usize;
        for prm in &combos {
            let period = prm.period.unwrap_or(0);
            let fast = prm.fast_period.unwrap_or(0);
            let slow = prm.slow_period.unwrap_or(0);
            if period == 0 || fast == 0 || slow == 0 {
                return Err(CudaMaaqError::InvalidInput(
                    "period, fast_period, and slow_period must be > 0".into(),
                ));
            }
            if period > len {
                return Err(CudaMaaqError::InvalidInput(format!(
                    "period {} exceeds data length {}",
                    period, len
                )));
            }
            if len - first_valid < period {
                return Err(CudaMaaqError::InvalidInput(format!(
                    "not enough valid data: need {}, have {}",
                    period,
                    len - first_valid
                )));
            }
            if period > i32::MAX as usize {
                return Err(CudaMaaqError::InvalidInput(
                    "period exceeds i32::MAX".into(),
                ));
            }
            if max_period < period {
                max_period = period;
            }
        }

        Ok((combos, first_valid, len, max_period))
    }

    fn launch_batch_kernel_plain(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_fast_scs: &DeviceBuffer<f32>,
        d_slow_scs: &DeviceBuffer<f32>,
        first_valid: usize,
        series_len: usize,
        n_combos: usize,
        max_period: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaMaaqError> {
        if series_len == 0 || n_combos == 0 || max_period == 0 {
            return Err(CudaMaaqError::InvalidInput(
                "series_len, n_combos, and max_period must be positive".into(),
            ));
        }
        if series_len > i32::MAX as usize
            || n_combos > i32::MAX as usize
            || max_period > i32::MAX as usize
        {
            return Err(CudaMaaqError::InvalidInput(
                "series_len, n_combos, or max_period exceed i32::MAX".into(),
            ));
        }
        if first_valid > i32::MAX as usize {
            return Err(CudaMaaqError::InvalidInput(
                "first_valid exceeds i32::MAX".into(),
            ));
        }

        let func = self
            .module
            .get_function("maaq_batch_f32")
            .map_err(|e| CudaMaaqError::Cuda(e.to_string()))?;

        // Selection + record
        let block_x = match self.policy.batch {
            BatchKernelPolicy::Auto => 32u32,
            BatchKernelPolicy::Plain { block_x } => block_x.max(1),
        };
        unsafe {
            (*(self as *const _ as *mut CudaMaaq)).last_batch =
                Some(BatchKernelSelected::Plain { block_x });
        }
        self.maybe_log_batch_debug();

        // Launch in chunks based on actual grid.x limit
        let max_chunk = self.grid_x_limit();
        for (start, len) in Self::chunk_pairs(n_combos, max_chunk) {
            let grid: GridSize = (len as u32, 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            let shared_bytes = (max_period * std::mem::size_of::<f32>()) as u32;

            unsafe {
                let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                let mut periods_ptr = d_periods.as_device_ptr().add(start).as_raw();
                let mut fast_ptr = d_fast_scs.as_device_ptr().add(start).as_raw();
                let mut slow_ptr = d_slow_scs.as_device_ptr().add(start).as_raw();
                let mut first_valid_i = first_valid as i32;
                let mut series_len_i = series_len as i32;
                let mut n_combos_i = len as i32;
                let mut max_period_i = max_period as i32;
                // slice output per chunk (row-major: combos × series_len)
                let mut out_ptr = d_out.as_device_ptr().add(start * series_len).as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut prices_ptr as *mut _ as *mut c_void,
                    &mut periods_ptr as *mut _ as *mut c_void,
                    &mut fast_ptr as *mut _ as *mut c_void,
                    &mut slow_ptr as *mut _ as *mut c_void,
                    &mut first_valid_i as *mut _ as *mut c_void,
                    &mut series_len_i as *mut _ as *mut c_void,
                    &mut n_combos_i as *mut _ as *mut c_void,
                    &mut max_period_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream
                    .launch(&func, grid, block, shared_bytes, args)
                    .map_err(|e| CudaMaaqError::Cuda(e.to_string()))?;
            }
        }

        Ok(())
    }

    /// Device-level batch entry: launches `maaq_batch_f32` using preallocated device buffers.
    /// Mirrors ALMA-style device APIs for benches and advanced callers.
    pub fn maaq_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_fast_scs: &DeviceBuffer<f32>,
        d_slow_scs: &DeviceBuffer<f32>,
        first_valid: usize,
        series_len: usize,
        n_combos: usize,
        max_period: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaMaaqError> {
        if series_len == 0 || n_combos == 0 || max_period == 0 {
            return Err(CudaMaaqError::InvalidInput(
                "series_len, n_combos, and max_period must be positive".into(),
            ));
        }
        if first_valid > series_len {
            return Err(CudaMaaqError::InvalidInput(
                "first_valid out of range".into(),
            ));
        }
        self.launch_batch_kernel_plain(
            d_prices,
            d_periods,
            d_fast_scs,
            d_slow_scs,
            first_valid,
            series_len,
            n_combos,
            max_period,
            d_out,
        )
    }

    pub fn maaq_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &MaaqBatchRange,
    ) -> Result<DeviceArrayF32, CudaMaaqError> {
        let (combos, first_valid, len, max_period) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let n_combos = combos.len();

        // VRAM estimate: inputs + params + outputs
        let bytes = len * std::mem::size_of::<f32>() + // prices
            n_combos * (std::mem::size_of::<i32>() + 2 * std::mem::size_of::<f32>()) + // period/fast/slow
            (n_combos * len) * std::mem::size_of::<f32>(); // out
        let headroom = 64 * 1024 * 1024; // ~64MB
        if !Self::will_fit(bytes, headroom) {
            return Err(CudaMaaqError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (bytes as f64) / (1024.0 * 1024.0)
            )));
        }

        let mut periods_i32 = Vec::with_capacity(n_combos);
        let mut fast_scs = Vec::with_capacity(n_combos);
        let mut slow_scs = Vec::with_capacity(n_combos);
        for prm in &combos {
            let period = prm.period.unwrap();
            let fast = prm.fast_period.unwrap();
            let slow = prm.slow_period.unwrap();
            periods_i32.push(period as i32);
            fast_scs.push(2.0f32 / (fast as f32 + 1.0f32));
            slow_scs.push(2.0f32 / (slow as f32 + 1.0f32));
        }

        let d_prices = self.upload_f32_large(data_f32)?;
        let d_periods = DeviceBuffer::from_slice(&periods_i32)
            .map_err(|e| CudaMaaqError::Cuda(e.to_string()))?;
        let d_fast =
            DeviceBuffer::from_slice(&fast_scs).map_err(|e| CudaMaaqError::Cuda(e.to_string()))?;
        let d_slow =
            DeviceBuffer::from_slice(&slow_scs).map_err(|e| CudaMaaqError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(n_combos * len, &self.stream) }
                .map_err(|e| CudaMaaqError::Cuda(e.to_string()))?;

        self.launch_batch_kernel_plain(
            &d_prices,
            &d_periods,
            &d_fast,
            &d_slow,
            first_valid,
            len,
            n_combos,
            max_period,
            &mut d_out,
        )?;
        self.stream
            .synchronize()
            .map_err(|e| CudaMaaqError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: len,
        })
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &MaaqParams,
    ) -> Result<(Vec<i32>, usize, f32, f32), CudaMaaqError> {
        if cols == 0 || rows == 0 {
            return Err(CudaMaaqError::InvalidInput(
                "num_series or series_len is zero".into(),
            ));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaMaaqError::InvalidInput(format!(
                "data length {} != cols*rows {}",
                data_tm_f32.len(),
                cols * rows
            )));
        }

        let period = params.period.unwrap_or(0);
        let fast = params.fast_period.unwrap_or(0);
        let slow = params.slow_period.unwrap_or(0);
        if period == 0 || fast == 0 || slow == 0 {
            return Err(CudaMaaqError::InvalidInput(
                "period, fast_period, and slow_period must be > 0".into(),
            ));
        }
        if period > rows {
            return Err(CudaMaaqError::InvalidInput(format!(
                "period {} exceeds series length {}",
                period, rows
            )));
        }
        if period > i32::MAX as usize {
            return Err(CudaMaaqError::InvalidInput(
                "period exceeds i32::MAX".into(),
            ));
        }

        let mut first_valids = vec![0i32; cols];
        for series in 0..cols {
            let mut found = None;
            for row in 0..rows {
                let idx = row * cols + series;
                let v = data_tm_f32[idx];
                if !v.is_nan() {
                    found = Some(row);
                    break;
                }
            }
            let fv = found.ok_or_else(|| {
                CudaMaaqError::InvalidInput(format!("series {} contains only NaNs", series))
            })?;
            if rows - fv < period {
                return Err(CudaMaaqError::InvalidInput(format!(
                    "series {} lacks enough valid data: need {} have {}",
                    series,
                    period,
                    rows - fv
                )));
            }
            if fv > i32::MAX as usize {
                return Err(CudaMaaqError::InvalidInput(
                    "first_valid exceeds i32::MAX".into(),
                ));
            }
            first_valids[series] = fv as i32;
        }

        let fast_sc = 2.0f32 / (fast as f32 + 1.0f32);
        let slow_sc = 2.0f32 / (slow as f32 + 1.0f32);

        Ok((first_valids, period, fast_sc, slow_sc))
    }

    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        period: usize,
        fast_sc: f32,
        slow_sc: f32,
        num_series: usize,
        series_len: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaMaaqError> {
        if period == 0 || num_series == 0 || series_len == 0 {
            return Err(CudaMaaqError::InvalidInput(
                "period, num_series, and series_len must be positive".into(),
            ));
        }
        if period > i32::MAX as usize
            || num_series > i32::MAX as usize
            || series_len > i32::MAX as usize
        {
            return Err(CudaMaaqError::InvalidInput(
                "period, num_series, or series_len exceed i32::MAX".into(),
            ));
        }

        let func = self
            .module
            .get_function("maaq_multi_series_one_param_f32")
            .map_err(|e| CudaMaaqError::Cuda(e.to_string()))?;

        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => 32u32,
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(1),
        };
        unsafe {
            (*(self as *const _ as *mut CudaMaaq)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }
        self.maybe_log_many_debug();

        // Launch in chunks over series using the actual grid.x limit
        let max_chunk = self.grid_x_limit();
        for (start, len) in Self::chunk_pairs(num_series, max_chunk) {
            let grid: GridSize = (len as u32, 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            let shared_bytes = (period * std::mem::size_of::<f32>()) as u32;

            unsafe {
                // IMPORTANT: base pointers are offset by start; stride remains full num_series
                let mut prices_ptr = d_prices_tm.as_device_ptr().add(start).as_raw();
                let mut period_i = period as i32;
                let mut fast = fast_sc;
                let mut slow = slow_sc;
                let mut num_series_i = num_series as i32; // stride (full column count)
                let mut series_len_i = series_len as i32;
                let mut first_ptr = d_first_valids.as_device_ptr().add(start).as_raw();
                // time-major output; offset by start series
                let mut out_ptr = d_out_tm.as_device_ptr().add(start).as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut prices_ptr as *mut _ as *mut c_void,
                    &mut period_i as *mut _ as *mut c_void,
                    &mut fast as *mut _ as *mut c_void,
                    &mut slow as *mut _ as *mut c_void,
                    &mut num_series_i as *mut _ as *mut c_void,
                    &mut series_len_i as *mut _ as *mut c_void,
                    &mut first_ptr as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream
                    .launch(&func, grid, block, shared_bytes, args)
                    .map_err(|e| CudaMaaqError::Cuda(e.to_string()))?;
            }
        }

        Ok(())
    }

    pub fn maaq_multi_series_one_param_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        period: i32,
        fast_sc: f32,
        slow_sc: f32,
        num_series: i32,
        series_len: i32,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaMaaqError> {
        if period <= 0 || num_series <= 0 || series_len <= 0 {
            return Err(CudaMaaqError::InvalidInput(
                "period, num_series, and series_len must be positive".into(),
            ));
        }
        self.launch_many_series_kernel(
            d_prices_tm,
            period as usize,
            fast_sc,
            slow_sc,
            num_series as usize,
            series_len as usize,
            d_first_valids,
            d_out_tm,
        )
    }

    pub fn maaq_multi_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &MaaqParams,
    ) -> Result<DeviceArrayF32, CudaMaaqError> {
        let (first_valids, period, fast_sc, slow_sc) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;

        let d_prices_tm = self.upload_f32_large(data_tm_f32)?;
        let d_first_valids = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaMaaqError::Cuda(e.to_string()))?;
        let mut d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(cols * rows, &self.stream) }
                .map_err(|e| CudaMaaqError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(
            &d_prices_tm,
            period,
            fast_sc,
            slow_sc,
            cols,
            rows,
            &d_first_valids,
            &mut d_out_tm,
        )?;
        self.stream
            .synchronize()
            .map_err(|e| CudaMaaqError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out_tm,
            rows,
            cols,
        })
    }

    pub fn maaq_multi_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &MaaqParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaMaaqError> {
        if out_tm.len() != cols * rows {
            return Err(CudaMaaqError::InvalidInput(format!(
                "output slice wrong length: got {} expected {}",
                out_tm.len(),
                cols * rows
            )));
        }
        let (first_valids, period, fast_sc, slow_sc) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;

        let d_prices_tm = self.upload_f32_large(data_tm_f32)?;
        let d_first_valids = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaMaaqError::Cuda(e.to_string()))?;
        let mut d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(cols * rows, &self.stream) }
                .map_err(|e| CudaMaaqError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(
            &d_prices_tm,
            period,
            fast_sc,
            slow_sc,
            cols,
            rows,
            &d_first_valids,
            &mut d_out_tm,
        )?;
        // Use pinned host buffer for faster D2H
        self.stream
            .synchronize()
            .map_err(|e| CudaMaaqError::Cuda(e.to_string()))?;
        let mut pinned: LockedBuffer<f32> = unsafe {
            LockedBuffer::uninitialized(cols * rows)
                .map_err(|e| CudaMaaqError::Cuda(e.to_string()))?
        };
        unsafe {
            d_out_tm
                .async_copy_to(pinned.as_mut_slice(), &self.stream)
                .map_err(|e| CudaMaaqError::Cuda(e.to_string()))?;
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaMaaqError::Cuda(e.to_string()))?;
        out_tm.copy_from_slice(pinned.as_slice());
        Ok(())
    }
}

impl CudaMaaq {
    #[inline]
    fn upload_f32_large(&self, src: &[f32]) -> Result<DeviceBuffer<f32>, CudaMaaqError> {
        const PINNED_THRESH_BYTES: usize = 1 << 20; // 1 MiB
        let n = src.len();
        if n * std::mem::size_of::<f32>() >= PINNED_THRESH_BYTES {
            // Stage in pinned memory for true async H2D and higher throughput
            let mut pinned: LockedBuffer<f32> = unsafe { LockedBuffer::uninitialized(n) }
                .map_err(|e| CudaMaaqError::Cuda(e.to_string()))?;
            pinned.as_mut_slice().copy_from_slice(src);

            let mut d = unsafe { DeviceBuffer::uninitialized_async(n, &self.stream) }
                .map_err(|e| CudaMaaqError::Cuda(e.to_string()))?;
            unsafe {
                d.async_copy_from(pinned.as_slice(), &self.stream)
                    .map_err(|e| CudaMaaqError::Cuda(e.to_string()))?;
            }
            Ok(d)
        } else {
            unsafe { DeviceBuffer::from_slice_async(src, &self.stream) }
                .map_err(|e| CudaMaaqError::Cuda(e.to_string()))
        }
    }
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::define_ma_period_benches;

    define_ma_period_benches!(
        maaq_benches,
        CudaMaaq,
        crate::indicators::moving_averages::maaq::MaaqBatchRange,
        crate::indicators::moving_averages::maaq::MaaqParams,
        maaq_batch_dev,
        maaq_multi_series_one_param_time_major_dev,
        crate::indicators::moving_averages::maaq::MaaqBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1),
            fast_period: (2, 2, 0),
            slow_period: (30, 30, 0)
        },
        crate::indicators::moving_averages::maaq::MaaqParams {
            period: Some(64),
            fast_period: Some(2),
            slow_period: Some(30)
        },
        "maaq",
        "maaq"
    );
    pub use maaq_benches::bench_profiles;
}
