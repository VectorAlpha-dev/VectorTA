//! CUDA support for the Wilder's Moving Average (Wilders) indicator.
//!
//! Mirrors the CPU batching API by accepting a single series alongside a sweep
//! of periods. Kernels run entirely in FP32 and reuse precomputed alpha values
//! and warm-up indices to keep the GPU work focused on the recurrence itself.

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::wilders::{WildersBatchRange, WildersParams};
use cust::context::{CacheConfig, Context};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug)]
pub enum CudaWildersError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaWildersError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaWildersError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaWildersError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}

impl std::error::Error for CudaWildersError {}

pub struct CudaWilders {
    module: Module,
    stream: Stream,
    _context: Context,
    policy: CudaWildersPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
    // CHANGE: cache param buffers to avoid re-copying identical sweeps
    param_cache: Option<ParamCache>,
}

// CHANGE: small cache for batch parameter buffers
struct ParamCache {
    hash: u64,
    periods: DeviceBuffer<i32>,
    alphas: DeviceBuffer<f32>,
    warm: DeviceBuffer<i32>,
}

struct PreparedWildersBatch {
    first_valid: usize,
    series_len: usize,
    periods_i32: Vec<i32>,
    alphas_f32: Vec<f32>,
    warm_indices: Vec<i32>,
}

impl CudaWilders {
    pub fn new(device_id: usize) -> Result<Self, CudaWildersError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaWildersError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaWildersError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaWildersError::Cuda(e.to_string()))?;

        // CHANGE: PTX JIT at O4 (most optimized)
        let ptx = include_str!(concat!(env!("OUT_DIR"), "/wilders_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O4),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => match Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]) {
                Ok(m) => m,
                Err(_) => Module::from_ptx(ptx, &[])
                    .map_err(|e| CudaWildersError::Cuda(e.to_string()))?,
            },
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaWildersError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            policy: CudaWildersPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
            param_cache: None,
        })
    }

    pub fn wilders_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &WildersBatchRange,
    ) -> Result<DeviceArrayF32, CudaWildersError> {
        let prepared = Self::prepare_batch_inputs(data_f32, sweep)?;
        let n_combos = prepared.periods_i32.len();

        // VRAM estimate (input + params + output) with headroom
        let prices_bytes = prepared.series_len * std::mem::size_of::<f32>();
        let params_bytes = (prepared.periods_i32.len() * std::mem::size_of::<i32>())
            + (prepared.alphas_f32.len() * std::mem::size_of::<f32>())
            + (prepared.warm_indices.len() * std::mem::size_of::<i32>());
        let out_bytes = prepared.series_len * n_combos * std::mem::size_of::<f32>();
        let required = prices_bytes + params_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024; // 64MB safety
        if !Self::will_fit(required, headroom) {
            return Err(CudaWildersError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        let d_prices = DeviceBuffer::from_slice(data_f32)
            .map_err(|e| CudaWildersError::Cuda(e.to_string()))?;
        // CHANGE: cache parameter device buffers across identical sweeps
        let mut hasher = DefaultHasher::new();
        prepared.periods_i32.hash(&mut hasher);
        for &a in &prepared.alphas_f32 { a.to_bits().hash(&mut hasher); }
        prepared.warm_indices.hash(&mut hasher);
        let params_hash = hasher.finish();

        // SAFETY: we only mutate param_cache field; borrowing controlled
        let (d_periods, d_alphas, d_warm) = unsafe {
            match &mut (*(self as *const _ as *mut CudaWilders)).param_cache {
                Some(cache)
                    if cache.hash == params_hash
                        && cache.periods.len() == prepared.periods_i32.len()
                        && cache.alphas.len() == prepared.alphas_f32.len()
                        && cache.warm.len() == prepared.warm_indices.len() =>
                { (&cache.periods, &cache.alphas, &cache.warm) }
                _ => {
                    let periods = DeviceBuffer::from_slice(&prepared.periods_i32)
                        .map_err(|e| CudaWildersError::Cuda(e.to_string()))?;
                    let alphas = DeviceBuffer::from_slice(&prepared.alphas_f32)
                        .map_err(|e| CudaWildersError::Cuda(e.to_string()))?;
                    let warm = DeviceBuffer::from_slice(&prepared.warm_indices)
                        .map_err(|e| CudaWildersError::Cuda(e.to_string()))?;
                    (*(self as *const _ as *mut CudaWilders)).param_cache =
                        Some(ParamCache { hash: params_hash, periods, alphas, warm });
                    let cache = (*(self as *const _ as *mut CudaWilders))
                        .param_cache
                        .as_ref()
                        .unwrap();
                    (&cache.periods, &cache.alphas, &cache.warm)
                }
            }
        };
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(prepared.series_len * n_combos)
                .map_err(|e| CudaWildersError::Cuda(e.to_string()))?
        };

        self.launch_batch_kernel(
            &d_prices,
            &d_periods,
            &d_alphas,
            &d_warm,
            prepared.series_len,
            prepared.first_valid,
            n_combos,
            &mut d_out,
        )?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: prepared.series_len,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn wilders_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_alphas: &DeviceBuffer<f32>,
        d_warm: &DeviceBuffer<i32>,
        series_len: i32,
        first_valid: i32,
        n_combos: i32,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaWildersError> {
        if series_len <= 0 {
            return Err(CudaWildersError::InvalidInput(
                "series_len must be positive".into(),
            ));
        }
        if first_valid < 0 || first_valid >= series_len {
            return Err(CudaWildersError::InvalidInput(format!(
                "first_valid out of range: {} (len {})",
                first_valid, series_len
            )));
        }
        if n_combos <= 0 {
            return Err(CudaWildersError::InvalidInput(
                "n_combos must be positive".into(),
            ));
        }
        let expected = n_combos as usize;
        if d_periods.len() != expected || d_alphas.len() != expected || d_warm.len() != expected {
            return Err(CudaWildersError::InvalidInput(
                "device buffer length mismatch".into(),
            ));
        }

        self.launch_batch_kernel(
            d_prices,
            d_periods,
            d_alphas,
            d_warm,
            series_len as usize,
            first_valid as usize,
            expected,
            d_out,
        )
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &WildersParams,
    ) -> Result<(Vec<i32>, i32, f32), CudaWildersError> {
        if cols == 0 || rows == 0 {
            return Err(CudaWildersError::InvalidInput(
                "num_series and series_len must be positive".into(),
            ));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaWildersError::InvalidInput(format!(
                "time-major slice length {} != cols*rows {}",
                data_tm_f32.len(),
                cols * rows
            )));
        }
        let period = params.period.unwrap_or(0) as i32;
        if period <= 0 {
            return Err(CudaWildersError::InvalidInput("period must be positive".into()));
        }
        if period as usize > rows {
            return Err(CudaWildersError::InvalidInput(format!(
                "period {} exceeds series length {}",
                period, rows
            )));
        }

        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = None;
            for t in 0..rows {
                let v = data_tm_f32[t * cols + s];
                if v.is_finite() {
                    fv = Some(t as i32);
                    break;
                }
            }
            let fv = fv.ok_or_else(|| {
                CudaWildersError::InvalidInput(format!("series {} contains only NaNs", s))
            })?;
            let remain = rows - fv as usize;
            if remain < period as usize {
                return Err(CudaWildersError::InvalidInput(format!(
                    "series {} lacks enough valid data: need {}, have {}",
                    s, period, remain
                )));
            }
            first_valids[s] = fv;
        }

        let alpha = 1.0f32 / (period as f32);
        Ok((first_valids, period, alpha))
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &WildersBatchRange,
    ) -> Result<PreparedWildersBatch, CudaWildersError> {
        if data_f32.is_empty() {
            return Err(CudaWildersError::InvalidInput("input data is empty".into()));
        }
        let combos = expand_periods(sweep);
        if combos.is_empty() {
            return Err(CudaWildersError::InvalidInput(
                "no period combinations provided".into(),
            ));
        }

        let series_len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaWildersError::InvalidInput("all values are NaN".into()))?;

        let mut periods_i32 = Vec::with_capacity(combos.len());
        let mut alphas_f32 = Vec::with_capacity(combos.len());
        let mut warm_indices = Vec::with_capacity(combos.len());

        for params in &combos {
            let period = params.period.unwrap_or(0);
            if period == 0 {
                return Err(CudaWildersError::InvalidInput(
                    "period must be positive".into(),
                ));
            }
            if series_len - first_valid < period {
                return Err(CudaWildersError::InvalidInput(format!(
                    "not enough valid data: needed >= {}, have {}",
                    period,
                    series_len - first_valid
                )));
            }
            for idx in 0..period {
                let sample = data_f32[first_valid + idx];
                if !sample.is_finite() {
                    return Err(CudaWildersError::InvalidInput(format!(
                        "non-finite value in warm window at offset {}",
                        idx
                    )));
                }
            }
            periods_i32.push(period as i32);
            alphas_f32.push(1.0f32 / (period as f32));
            warm_indices.push((first_valid + period - 1) as i32);
        }

        Ok(PreparedWildersBatch {
            first_valid,
            series_len,
            periods_i32,
            alphas_f32,
            warm_indices,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_alphas: &DeviceBuffer<f32>,
        d_warm: &DeviceBuffer<i32>,
        series_len: usize,
        first_valid: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaWildersError> {
        if n_combos == 0 {
            return Ok(());
        }

        // CHANGE: prefer L1 cache and align block size to warp multiples
        let mut func = self
            .module
            .get_function("wilders_batch_f32")
            .map_err(|e| CudaWildersError::Cuda(e.to_string()))?;
        let _ = func.set_cache_config(CacheConfig::PreferL1);

        let block_x_user = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => block_x.max(32),
            BatchKernelPolicy::Auto => 256,
        };
        let block_threads = ((block_x_user / 32).max(1).min(32)) * 32; // 32..1024
        unsafe {
            (*(self as *const _ as *mut CudaWilders)).last_batch =
                Some(BatchKernelSelected::Plain { block_x: block_threads });
        }
        self.maybe_log_batch_debug();

        let grid: GridSize = (n_combos as u32, 1, 1).into();
        let block: BlockSize = (block_threads, 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut alphas_ptr = d_alphas.as_device_ptr().as_raw();
            let mut warm_ptr = d_warm.as_device_ptr().as_raw();
            let mut series_len_i = series_len as i32;
            let mut first_valid_i = first_valid as i32;
            let mut n_combos_i = n_combos as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut alphas_ptr as *mut _ as *mut c_void,
                &mut warm_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut n_combos_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaWildersError::Cuda(e.to_string()))?;
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        period: i32,
        alpha: f32,
        num_series: usize,
        series_len: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaWildersError> {
        // CHANGE: prefer L1 cache and map one warp per series
        let mut func = self
            .module
            .get_function("wilders_many_series_one_param_f32")
            .map_err(|e| CudaWildersError::Cuda(e.to_string()))?;
        let _ = func.set_cache_config(CacheConfig::PreferL1);

        let block_x_user = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(32),
            ManySeriesKernelPolicy::Auto => {
                if num_series < 64 { 128 } else { 256 }
            }
        };
        let block_threads = ((block_x_user / 32).max(1).min(32)) * 32; // 32..1024
        let warps_per_block: u32 = (block_threads / 32) as u32;
        let grid_x: u32 = ((num_series as u32) + (warps_per_block - 1)) / warps_per_block;
        unsafe {
            (*(self as *const _ as *mut CudaWilders)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x: block_threads });
        }
        self.maybe_log_many_debug();

        let block: BlockSize = (block_threads, 1, 1).into();
        let grid: GridSize = (grid_x, 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut alpha_f = alpha as f32;
            let mut cols_i = num_series as i32;
            let mut rows_i = series_len as i32;
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut alpha_f as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaWildersError::Cuda(e.to_string()))?;
        }

        Ok(())
    }

    pub fn wilders_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &WildersParams,
    ) -> Result<DeviceArrayF32, CudaWildersError> {
        let (first_valids, period, alpha) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;

        // VRAM estimate
        let elems = cols * rows;
        let in_bytes = elems * std::mem::size_of::<f32>();
        let first_bytes = cols * std::mem::size_of::<i32>();
        let out_bytes = elems * std::mem::size_of::<f32>();
        let required = in_bytes + first_bytes + out_bytes + (16 * 1024 * 1024);
        if !Self::will_fit(required, 64 * 1024 * 1024) {
            return Err(CudaWildersError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        let d_prices_tm = DeviceBuffer::from_slice(data_tm_f32)
            .map_err(|e| CudaWildersError::Cuda(e.to_string()))?;
        let d_first = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaWildersError::Cuda(e.to_string()))?;
        let mut d_out_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }
            .map_err(|e| CudaWildersError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(
            &d_prices_tm,
            &d_first,
            period,
            alpha,
            cols,
            rows,
            &mut d_out_tm,
        )?;

        Ok(DeviceArrayF32 { buf: d_out_tm, rows, cols })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn wilders_many_series_one_param_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        period: i32,
        alpha: f32,
        num_series: usize,
        series_len: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaWildersError> {
        if num_series == 0 || series_len == 0 {
            return Err(CudaWildersError::InvalidInput(
                "num_series and series_len must be positive".into(),
            ));
        }
        if d_prices_tm.len() != num_series * series_len || d_out_tm.len() != num_series * series_len {
            return Err(CudaWildersError::InvalidInput(
                "time-major buffer length mismatch".into(),
            ));
        }
        if d_first_valids.len() != num_series {
            return Err(CudaWildersError::InvalidInput(
                "first_valids length mismatch".into(),
            ));
        }
        self.launch_many_series_kernel(
            d_prices_tm,
            d_first_valids,
            period,
            alpha,
            num_series,
            series_len,
            d_out_tm,
        )
    }

    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> bool {
        if let Ok((free, _total)) = mem_get_info() {
            required_bytes.saturating_add(headroom_bytes) <= free
        } else {
            true
        }
    }

    #[inline]
    fn maybe_log_batch_debug(&self) {
        if self.debug_batch_logged { return; }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                static ONCE: AtomicBool = AtomicBool::new(false);
                if !ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] WILDERS batch selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut CudaWilders)).debug_batch_logged = true; }
            }
        }
    }

    #[inline]
    fn maybe_log_many_debug(&self) {
        if self.debug_many_logged { return; }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                static ONCE: AtomicBool = AtomicBool::new(false);
                if !ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] WILDERS many-series selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut CudaWilders)).debug_many_logged = true; }
            }
        }
    }
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::define_ma_period_benches;

    define_ma_period_benches!(
        wilders_benches,
        CudaWilders,
        crate::indicators::moving_averages::wilders::WildersBatchRange,
        crate::indicators::moving_averages::wilders::WildersParams,
        wilders_batch_dev,
        wilders_many_series_one_param_time_major_dev,
        crate::indicators::moving_averages::wilders::WildersBatchRange { period: (10, 10 + PARAM_SWEEP - 1, 1) },
        crate::indicators::moving_averages::wilders::WildersParams { period: Some(64) },
        "wilders",
        "wilders"
    );
    pub use wilders_benches::bench_profiles;
}

fn expand_periods(range: &WildersBatchRange) -> Vec<WildersParams> {
    let (start, end, step) = range.period;
    if step == 0 || start == end {
        return vec![WildersParams {
            period: Some(start),
        }];
    }

    let mut out = Vec::new();
    let mut value = start;
    while value <= end {
        out.push(WildersParams {
            period: Some(value),
        });
        value = value.saturating_add(step);
    }
    out
}

// --- Simple policy types (parity with ALMA/CWMA style) ---

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy { Auto, Plain { block_x: u32 } }
impl Default for BatchKernelPolicy { fn default() -> Self { Self::Auto } }

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy { Auto, OneD { block_x: u32 } }
impl Default for ManySeriesKernelPolicy { fn default() -> Self { Self::Auto } }

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaWildersPolicy { pub batch: BatchKernelPolicy, pub many_series: ManySeriesKernelPolicy }

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected { Plain { block_x: u32 } }

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected { OneD { block_x: u32 } }
