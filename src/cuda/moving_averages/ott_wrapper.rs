//! CUDA wrapper for the OTT (Optimized Trend Tracker) indicator.
//!
//! Design mirrors ALMA/CWMA wrappers:
//! - PTX load with DetermineTargetFromContext + O2, with fallbacks.
//! - NON_BLOCKING stream.
//! - Light policy knobs and one-shot introspection (BENCH_DEBUG=1).
//! - VRAM headroom checks and simple chunking when needed.
//!
//! Math strategy:
//! - Generic path: compute MA on device via `CudaMaSelector` (or wrapper-specific
//!   many-series functions), then apply OTT on device using `ott_apply_single_f32`
//!   or `ott_many_series_one_param_f32`.
//! - Fast path for default `ma_type == "VAR"`: use VAR-integrated kernels
//!   (`ott_from_var_*`) to avoid a separate MA pass.

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
// Reuse CWMA-style enums for parity
use super::{BatchKernelPolicy, ManySeriesKernelPolicy};
use crate::cuda::moving_averages::{
    CudaEma, CudaKama, CudaNama, CudaSma, CudaVpwma, CudaVwma, CudaWilders, CudaZlema,
};
use crate::cuda::moving_averages::{CudaMaData, CudaMaSelector};
use crate::indicators::ott::{OttBatchRange, OttParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, Function, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug)]
pub enum CudaOttError {
    Cuda(String),
    InvalidInput(String),
    Unsupported(String),
}

impl fmt::Display for CudaOttError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaOttError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaOttError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
            CudaOttError::Unsupported(e) => write!(f, "Unsupported: {}", e),
        }
    }
}
impl std::error::Error for CudaOttError {}

#[derive(Clone, Copy, Debug)]
pub struct CudaOttPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

impl Default for CudaOttPolicy {
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

pub struct CudaOtt {
    module: Module,
    stream: Stream,
    _context: Context,
    device_id: u32,
    policy: CudaOttPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaOtt {
    pub fn new(device_id: usize) -> Result<Self, CudaOttError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaOttError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaOttError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaOttError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/ott_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, jit_opts)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))
            .map_err(|e| CudaOttError::Cuda(e.to_string()))?;

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaOttError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaOttPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn synchronize(&self) -> Result<(), CudaOttError> {
        self.stream
            .synchronize()
            .map_err(|e| CudaOttError::Cuda(e.to_string()))
    }

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
        if let Some((free, _total)) = Self::device_mem_info() {
            required_bytes.saturating_add(headroom_bytes) <= free
        } else {
            true
        }
    }

    #[inline]
    fn memset_nan32_async(&self, dst_ptr_raw: u64, n_elems: usize) -> Result<(), CudaOttError> {
        const QNAN_BITS: u32 = 0x7FC0_0000;
        unsafe {
            use cust::sys::cuMemsetD32Async;
            let err = cuMemsetD32Async(
                dst_ptr_raw as cust::sys::CUdeviceptr,
                QNAN_BITS,
                n_elems,
                self.stream.as_inner(),
            );
            if err != cust::sys::CUresult::CUDA_SUCCESS {
                return Err(CudaOttError::Cuda(format!(
                    "cuMemsetD32Async failed: {:?}",
                    err
                )));
            }
        }
        Ok(())
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
                    eprintln!("[DEBUG] OTT batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaOtt)).debug_batch_logged = true;
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
                    eprintln!("[DEBUG] OTT many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaOtt)).debug_many_logged = true;
                }
            }
        }
    }

    // ---------------------- Public API ----------------------

    pub fn ott_batch_dev(
        &self,
        prices_f32: &[f32],
        sweep: &OttBatchRange,
    ) -> Result<DeviceArrayF32, CudaOttError> {
        if prices_f32.is_empty() {
            return Err(CudaOttError::InvalidInput("empty price input".into()));
        }
        let cols = prices_f32.len();
        // Validate there is at least one finite value
        if prices_f32.iter().all(|v| !v.is_finite()) {
            return Err(CudaOttError::InvalidInput("all values are NaN".into()));
        }

        let combos = expand_combos(sweep);
        if combos.is_empty() {
            return Err(CudaOttError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        // VRAM est.
        let rows = combos.len();
        let bytes = cols * std::mem::size_of::<f32>() + rows * cols * std::mem::size_of::<f32>();
        let headroom = env::var("CUDA_MEM_HEADROOM")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(64 * 1024 * 1024);
        if !Self::will_fit(bytes, headroom) {
            return Err(CudaOttError::InvalidInput(
                "insufficient free VRAM for OTT batch".into(),
            ));
        }

        // Stage prices once
        let mut d_prices = unsafe { DeviceBuffer::<f32>::uninitialized(cols) }
            .map_err(|e| CudaOttError::Cuda(e.to_string()))?;
        unsafe { d_prices.async_copy_from(prices_f32, &self.stream) }
            .map_err(|e| CudaOttError::Cuda(e.to_string()))?;

        // Allocate output and prefill qNaN
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(rows * cols) }
            .map_err(|e| CudaOttError::Cuda(e.to_string()))?;
        self.memset_nan32_async(d_out.as_device_ptr().as_raw() as u64, rows * cols)?;

        // Get kernel functions
        let mut f_var: Option<Function> = self.module.get_function("ott_from_var_batch_f32").ok();
        let mut f_apply: Option<Function> = self.module.get_function("ott_apply_single_f32").ok();
        if f_apply.is_none() {
            return Err(CudaOttError::Cuda(
                "missing ott_apply_single_f32 symbol".into(),
            ));
        }

        // Reusable 1-element device buffers for var path
        let mut d_period = unsafe { DeviceBuffer::<i32>::uninitialized(1) }
            .map_err(|e| CudaOttError::Cuda(e.to_string()))?;
        let mut d_percent = unsafe { DeviceBuffer::<f32>::uninitialized(1) }
            .map_err(|e| CudaOttError::Cuda(e.to_string()))?;

        // Host-side MA selector for generic path
        let selector = CudaMaSelector::new(self.device_id as usize);

        for (row_idx, p) in combos.iter().enumerate() {
            let period = p.period.unwrap_or(2);
            let percent = p.percent.unwrap_or(1.4) as f32;
            let ma_type = p.ma_type.as_deref().unwrap_or("VAR");
            let out_row_ptr = unsafe { d_out.as_device_ptr().offset((row_idx * cols) as isize) };

            if ma_type.eq_ignore_ascii_case("VAR") {
                // Prefer integrated VAR path if available
                if let Some(ref mut func) = f_var {
                    // Copy scalars
                    unsafe { d_period.async_copy_from(&[period as i32], &self.stream) }
                        .map_err(|e| CudaOttError::Cuda(e.to_string()))?;
                    unsafe { d_percent.async_copy_from(&[percent], &self.stream) }
                        .map_err(|e| CudaOttError::Cuda(e.to_string()))?;

                    unsafe {
                        let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                        let mut periods_ptr = d_period.as_device_ptr().as_raw();
                        let mut percents_ptr = d_percent.as_device_ptr().as_raw();
                        let mut series_len_i = cols as i32;
                        let mut n_combos_i = 1i32;
                        let mut out_ptr = out_row_ptr.as_raw();
                        let args: &mut [*mut c_void] = &mut [
                            &mut prices_ptr as *mut _ as *mut c_void,
                            &mut periods_ptr as *mut _ as *mut c_void,
                            &mut percents_ptr as *mut _ as *mut c_void,
                            &mut series_len_i as *mut _ as *mut c_void,
                            &mut n_combos_i as *mut _ as *mut c_void,
                            &mut out_ptr as *mut _ as *mut c_void,
                        ];
                        let grid: GridSize = (1, 1, 1).into();
                        let block: BlockSize = (1, 1, 1).into();
                        self.stream
                            .launch(func, grid, block, 0, args)
                            .map_err(|e| CudaOttError::Cuda(e.to_string()))?;
                    }
                } else {
                    // Fallback: compute VAR via selector (not ideal if selector lacks VAR)
                    let dev = selector
                        .ma_to_device("VAR", CudaMaData::SliceF32(prices_f32), period)
                        .map_err(|e| CudaOttError::Cuda(e.to_string()))?;
                    self.launch_apply_single(
                        &mut f_apply.as_mut().unwrap(),
                        &dev.buf,
                        cols,
                        percent,
                        out_row_ptr.as_raw(),
                    )?;
                    // Ensure kernel completion before dropping dev buffer
                    self.stream
                        .synchronize()
                        .map_err(|e| CudaOttError::Cuda(e.to_string()))?;
                }
            } else {
                // Generic path: compute MA then apply OTT
                let dev = selector
                    .ma_to_device(ma_type, CudaMaData::SliceF32(prices_f32), period)
                    .map_err(|e| CudaOttError::Cuda(e.to_string()))?;
                self.launch_apply_single(
                    &mut f_apply.as_mut().unwrap(),
                    &dev.buf,
                    cols,
                    percent,
                    out_row_ptr.as_raw(),
                )?;
                self.stream
                    .synchronize()
                    .map_err(|e| CudaOttError::Cuda(e.to_string()))?;
            }
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaOttError::Cuda(e.to_string()))?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    fn launch_apply_single(
        &self,
        func: &mut Function,
        d_ma: &DeviceBuffer<f32>,
        len: usize,
        percent: f32,
        out_ptr_raw: u64,
    ) -> Result<(), CudaOttError> {
        unsafe {
            let mut ma_ptr = d_ma.as_device_ptr().as_raw();
            let mut series_len_i = len as i32;
            let mut pct = percent;
            let mut out_ptr = out_ptr_raw;
            let args: &mut [*mut c_void] = &mut [
                &mut ma_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut pct as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            let grid: GridSize = (1, 1, 1).into();
            let block: BlockSize = (1, 1, 1).into();
            self.stream
                .launch(func, grid, block, 0, args)
                .map_err(|e| CudaOttError::Cuda(e.to_string()))?;
        }
        Ok(())
    }

    pub fn ott_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &OttParams,
    ) -> Result<DeviceArrayF32, CudaOttError> {
        if cols == 0 || rows == 0 {
            return Err(CudaOttError::InvalidInput("empty input".into()));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaOttError::InvalidInput("shape mismatch".into()));
        }
        let period = params.period.unwrap_or(2);
        let percent = params.percent.unwrap_or(1.4) as f32;
        let ma_type = params.ma_type.as_deref().unwrap_or("VAR");

        // Output buffer
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(cols * rows) }
            .map_err(|e| CudaOttError::Cuda(e.to_string()))?;
        self.memset_nan32_async(d_out.as_device_ptr().as_raw() as u64, cols * rows)?;

        if ma_type.eq_ignore_ascii_case("VAR") {
            // VAR-integrated path
            let mut d_in = unsafe { DeviceBuffer::<f32>::uninitialized(cols * rows) }
                .map_err(|e| CudaOttError::Cuda(e.to_string()))?;
            unsafe { d_in.async_copy_from(data_tm_f32, &self.stream) }
                .map_err(|e| CudaOttError::Cuda(e.to_string()))?;
            let mut func = self
                .module
                .get_function("ott_from_var_many_series_one_param_f32")
                .map_err(|e| CudaOttError::Cuda(e.to_string()))?;
            unsafe {
                let mut in_ptr = d_in.as_device_ptr().as_raw();
                let mut cols_i = cols as i32;
                let mut rows_i = rows as i32;
                let mut period_i = period as i32;
                let mut pct = percent;
                let mut out_ptr = d_out.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut in_ptr as *mut _ as *mut c_void,
                    &mut cols_i as *mut _ as *mut c_void,
                    &mut rows_i as *mut _ as *mut c_void,
                    &mut period_i as *mut _ as *mut c_void,
                    &mut pct as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                let grid: GridSize = ((rows as u32).max(1), 1, 1).into();
                let block: BlockSize = (1, 1, 1).into();
                self.stream
                    .launch(&mut func, grid, block, 0, args)
                    .map_err(|e| CudaOttError::Cuda(e.to_string()))?;
            }
        } else {
            // Compute MA on device with the appropriate wrapper, many-series time-major
            // Fallback support for common MA types
            let ma_dev = if ma_type.eq_ignore_ascii_case("EMA") {
                let p = crate::indicators::moving_averages::ema::EmaParams {
                    period: Some(period),
                };
                CudaEma::new(self.device_id as usize)
                    .map_err(|e| CudaOttError::Cuda(e.to_string()))?
                    .ema_many_series_one_param_time_major_dev(data_tm_f32, cols, rows, &p)
                    .map_err(|e| CudaOttError::Cuda(e.to_string()))?
            } else if ma_type.eq_ignore_ascii_case("SMA") {
                let p = crate::indicators::moving_averages::sma::SmaParams {
                    period: Some(period),
                };
                CudaSma::new(self.device_id as usize)
                    .map_err(|e| CudaOttError::Cuda(e.to_string()))?
                    .sma_multi_series_one_param_time_major_dev(data_tm_f32, cols, rows, &p)
                    .map_err(|e| CudaOttError::Cuda(e.to_string()))?
            } else if ma_type.eq_ignore_ascii_case("ZLEMA") {
                let p = crate::indicators::moving_averages::zlema::ZlemaParams {
                    period: Some(period),
                };
                CudaZlema::new(self.device_id as usize)
                    .map_err(|e| CudaOttError::Cuda(e.to_string()))?
                    .zlema_many_series_one_param_time_major_dev(data_tm_f32, cols, rows, &p)
                    .map_err(|e| CudaOttError::Cuda(e.to_string()))?
            } else if ma_type.eq_ignore_ascii_case("WILDERS")
                || ma_type.eq_ignore_ascii_case("WWMA")
            {
                let p = crate::indicators::moving_averages::wilders::WildersParams {
                    period: Some(period),
                };
                CudaWilders::new(self.device_id as usize)
                    .map_err(|e| CudaOttError::Cuda(e.to_string()))?
                    .wilders_many_series_one_param_time_major_dev(data_tm_f32, cols, rows, &p)
                    .map_err(|e| CudaOttError::Cuda(e.to_string()))?
            } else if ma_type.eq_ignore_ascii_case("KAMA") {
                let p = crate::indicators::moving_averages::kama::KamaParams {
                    period: Some(period),
                };
                CudaKama::new(self.device_id as usize)
                    .map_err(|e| CudaOttError::Cuda(e.to_string()))?
                    .kama_many_series_one_param_time_major_dev(data_tm_f32, cols, rows, &p)
                    .map_err(|e| CudaOttError::Cuda(e.to_string()))?
            } else if ma_type.eq_ignore_ascii_case("VWMA") {
                return Err(CudaOttError::Unsupported(
                    "vwma requires candles+volume; not supported in this path".into(),
                ));
            } else if ma_type.eq_ignore_ascii_case("VPWMA") {
                // Needs power; use default power=0.382 for parity with selector
                let p = crate::indicators::moving_averages::vpwma::VpwmaParams {
                    period: Some(period),
                    power: Some(0.382),
                };
                CudaVpwma::new(self.device_id as usize)
                    .map_err(|e| CudaOttError::Cuda(e.to_string()))?
                    .vpwma_multi_series_one_param_time_major_dev(data_tm_f32, cols, rows, &p)
                    .map_err(|e| CudaOttError::Cuda(e.to_string()))?
            } else if ma_type.eq_ignore_ascii_case("NAMA") {
                let p = crate::indicators::moving_averages::nama::NamaParams {
                    period: Some(period),
                    ..Default::default()
                };
                CudaNama::new(self.device_id as usize)
                    .map_err(|e| CudaOttError::Cuda(e.to_string()))?
                    .nama_many_series_one_param_time_major_dev(data_tm_f32, cols, rows, &p)
                    .map_err(|e| CudaOttError::Cuda(e.to_string()))?
            } else if ma_type.eq_ignore_ascii_case("VWMA") || ma_type.eq_ignore_ascii_case("VWAP") {
                return Err(CudaOttError::Unsupported(
                    "volume/anchor-based MA not supported in ott_many_series path".into(),
                ));
            } else {
                return Err(CudaOttError::Unsupported(format!(
                    "unsupported ma_type '{}' for OTT CUDA many-series",
                    ma_type
                )));
            };

            // Launch apply-many over rows
            let mut func = self
                .module
                .get_function("ott_many_series_one_param_f32")
                .map_err(|e| CudaOttError::Cuda(e.to_string()))?;
            unsafe {
                let mut in_ptr = ma_dev.buf.as_device_ptr().as_raw();
                let mut cols_i = cols as i32;
                let mut rows_i = rows as i32;
                let mut pct = percent;
                let mut out_ptr = d_out.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut in_ptr as *mut _ as *mut c_void,
                    &mut cols_i as *mut _ as *mut c_void,
                    &mut rows_i as *mut _ as *mut c_void,
                    &mut pct as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                let grid: GridSize = ((rows as u32).max(1), 1, 1).into();
                let block: BlockSize = (1, 1, 1).into();
                self.stream
                    .launch(&mut func, grid, block, 0, args)
                    .map_err(|e| CudaOttError::Cuda(e.to_string()))?;
            }
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaOttError::Cuda(e.to_string()))?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }
}

// ---------- Benches ----------
pub mod benches {
    use super::*;
    use crate::define_ma_period_benches;

    define_ma_period_benches!(
        ott_benches,
        CudaOtt,
        crate::indicators::ott::OttBatchRange,
        crate::indicators::ott::OttParams,
        ott_batch_dev,
        ott_many_series_one_param_time_major_dev,
        crate::indicators::ott::OttBatchRange {
            period: (2, 2 + PARAM_SWEEP - 1, 1),
            percent: (0.2, 5.0, (5.0 - 0.2) / (PARAM_SWEEP - 1) as f64),
            ma_types: vec!["VAR".to_string()]
        },
        crate::indicators::ott::OttParams {
            period: Some(16),
            percent: Some(1.4),
            ma_type: Some("VAR".to_string())
        },
        "ott",
        "ott"
    );
    pub use ott_benches::bench_profiles;
}

fn expand_combos(range: &OttBatchRange) -> Vec<OttParams> {
    fn axis_usize(axis: (usize, usize, usize)) -> Vec<usize> {
        let (start, end, step) = axis;
        if step == 0 || start == end {
            return vec![start];
        }
        let mut out = Vec::new();
        let mut v = start;
        while v <= end {
            out.push(v);
            v = v.saturating_add(step);
        }
        out
    }
    fn axis_f64(axis: (f64, f64, f64)) -> Vec<f64> {
        let (start, end, step) = axis;
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return vec![start];
        }
        let mut out = Vec::new();
        let mut v = start;
        while v <= end + 1e-12 {
            out.push(v);
            v += step;
        }
        out
    }
    let periods = axis_usize(range.period);
    let percents = axis_f64(range.percent);
    let types = if range.ma_types.is_empty() {
        vec!["VAR".to_string()]
    } else {
        range.ma_types.clone()
    };
    let mut combos = Vec::with_capacity(periods.len() * percents.len() * types.len());
    for &p in &periods {
        for &q in &percents {
            for t in &types {
                combos.push(OttParams {
                    period: Some(p),
                    percent: Some(q),
                    ma_type: Some(t.clone()),
                });
            }
        }
    }
    combos
}
