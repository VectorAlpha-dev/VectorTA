//! CUDA wrapper for the WCLPRICE (Weighted Close Price) kernels.
//!
//! Parity goals with ALMA wrapper:
//! - PTX load with target-from-context and OptLevel O2, with relaxed fallbacks
//! - NON_BLOCKING stream
//! - VRAM estimates + headroom checks; safe grid sizing
//! - Warmup/NaN semantics match scalar (prefix NaNs until warm)
//! - Minimal policy + introspection for kernel selection (OneD with block_x)
//!
//! Kernels expected:
//! - `wclprice_batch_f32`                                // one-series × many-params (here: 1 row)
//! - `wclprice_many_series_one_param_time_major_f32`     // many-series × one-param (time-major OHLC)
//!
//! Note: WCLPRICE has no tunable parameters. The "batch" entry point is provided for API
//! parity; it returns a single-row output matrix with shape [1, series_len].

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::wclprice::WclpriceBatchRange;
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, CopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug)]
pub enum CudaWclpriceError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaWclpriceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaWclpriceError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaWclpriceError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for CudaWclpriceError {}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    OneD { block_x: u32 },
}

impl Default for BatchKernelPolicy {
    fn default() -> Self {
        BatchKernelPolicy::Auto
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 },
}

impl Default for ManySeriesKernelPolicy {
    fn default() -> Self {
        ManySeriesKernelPolicy::Auto
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaWclpricePolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected {
    OneD { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}

pub struct CudaWclprice {
    module: Module,
    stream: Stream,
    _context: Context,
    device_id: u32,
    policy: CudaWclpricePolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaWclprice {
    pub fn new(device_id: usize) -> Result<Self, CudaWclpriceError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaWclpriceError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaWclpriceError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaWclpriceError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/wclprice_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => match Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]) {
                Ok(m) => m,
                Err(_) => Module::from_ptx(ptx, &[])
                    .map_err(|e| CudaWclpriceError::Cuda(e.to_string()))?,
            },
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaWclpriceError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaWclpricePolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn new_with_policy(device_id: usize, policy: CudaWclpricePolicy) -> Result<Self, CudaWclpriceError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }
    pub fn set_policy(&mut self, policy: CudaWclpricePolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaWclpricePolicy { &self.policy }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> { self.last_batch }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> { self.last_many }
    pub fn synchronize(&self) -> Result<(), CudaWclpriceError> {
        self.stream
            .synchronize()
            .map_err(|e| CudaWclpriceError::Cuda(e.to_string()))
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
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> bool {
        if !Self::mem_check_enabled() { return true; }
        if let Some((free, _)) = Self::device_mem_info() {
            required_bytes.saturating_add(headroom_bytes) <= free
        } else { true }
    }

    #[inline]
    fn choose_block_x(policy_auto_env: &str, default_bx: u32, clamp_min: u32) -> u32 {
        std::env::var(policy_auto_env)
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(default_bx)
            .max(clamp_min)
            .min(1024)
    }

    fn maybe_log_batch_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged { return; }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                let per_s = std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_s || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] WCLPRICE batch selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut CudaWclprice)).debug_batch_logged = true; }
            }
        }
    }

    fn maybe_log_many_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_many_logged { return; }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                let per_s = std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_s || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] WCLPRICE many-series selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut CudaWclprice)).debug_many_logged = true; }
            }
        }
    }

    // ---------------- One-series × many-params (single row) ----------------

    fn prepare_batch_inputs(
        high: &[f32], low: &[f32], close: &[f32], _sweep: &WclpriceBatchRange,
    ) -> Result<(usize, usize), CudaWclpriceError> {
        if high.is_empty() || low.is_empty() || close.is_empty() {
            return Err(CudaWclpriceError::InvalidInput("empty OHLC data".into()));
        }
        if high.len() != low.len() || low.len() != close.len() {
            return Err(CudaWclpriceError::InvalidInput(format!(
                "OHLC length mismatch: h={}, l={}, c={}", high.len(), low.len(), close.len()
            )));
        }
        let series_len = close.len();
        let first_valid = (0..series_len)
            .find(|&i| high[i].is_finite() && low[i].is_finite() && close[i].is_finite())
            .ok_or_else(|| CudaWclpriceError::InvalidInput("all values are NaN".into()))?;
        Ok((series_len, first_valid))
    }

    fn launch_batch_kernel(
        &self,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_close: &DeviceBuffer<f32>,
        series_len: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaWclpriceError> {
        let func = self
            .module
            .get_function("wclprice_batch_f32")
            .map_err(|e| CudaWclpriceError::Cuda(e.to_string()))?;

        let block_x: u32 = match self.policy.batch {
            BatchKernelPolicy::Auto => Self::choose_block_x("WCLPRICE_BLOCK_X", 256, 64),
            BatchKernelPolicy::OneD { block_x } => block_x.max(64).min(1024),
        };
        unsafe {
            (*(self as *const _ as *mut CudaWclprice)).last_batch = Some(BatchKernelSelected::OneD { block_x });
        }
        self.maybe_log_batch_debug();

        // 1 combo (single row), grid-stride along time
        let grid: GridSize = (((series_len as u32 + block_x - 1) / block_x).max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut high_ptr = d_high.as_device_ptr().as_raw();
            let mut low_ptr = d_low.as_device_ptr().as_raw();
            let mut close_ptr = d_close.as_device_ptr().as_raw();
            let mut len_i = series_len as i32;
            let mut first_i = first_valid as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut high_ptr as *mut _ as *mut c_void,
                &mut low_ptr as *mut _ as *mut c_void,
                &mut close_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut first_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaWclpriceError::Cuda(e.to_string()))?;
        }
        Ok(())
    }

    pub fn wclprice_batch_dev(
        &self,
        high: &[f32], low: &[f32], close: &[f32], sweep: &WclpriceBatchRange,
    ) -> Result<DeviceArrayF32, CudaWclpriceError> {
        let (series_len, first_valid) = Self::prepare_batch_inputs(high, low, close, sweep)?;

        // VRAM estimate: 3 inputs + 1 output
        let bytes = series_len * std::mem::size_of::<f32>() * 4;
        if !Self::will_fit(bytes, 64 * 1024 * 1024) {
            return Err(CudaWclpriceError::InvalidInput("insufficient VRAM for WCLPRICE".into()));
        }

        let h_high = LockedBuffer::from_slice(high).map_err(|e| CudaWclpriceError::Cuda(e.to_string()))?;
        let h_low = LockedBuffer::from_slice(low).map_err(|e| CudaWclpriceError::Cuda(e.to_string()))?;
        let h_close = LockedBuffer::from_slice(close).map_err(|e| CudaWclpriceError::Cuda(e.to_string()))?;
        let mut d_high: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(series_len) }
            .map_err(|e| CudaWclpriceError::Cuda(e.to_string()))?;
        let mut d_low: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(series_len) }
            .map_err(|e| CudaWclpriceError::Cuda(e.to_string()))?;
        let mut d_close: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(series_len) }
            .map_err(|e| CudaWclpriceError::Cuda(e.to_string()))?;
        unsafe {
            d_high.copy_from(h_high.as_slice()).map_err(|e| CudaWclpriceError::Cuda(e.to_string()))?;
            d_low.copy_from(h_low.as_slice()).map_err(|e| CudaWclpriceError::Cuda(e.to_string()))?;
            d_close.copy_from(h_close.as_slice()).map_err(|e| CudaWclpriceError::Cuda(e.to_string()))?;
        }

        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(series_len) }
            .map_err(|e| CudaWclpriceError::Cuda(e.to_string()))?;
        self.launch_batch_kernel(&d_high, &d_low, &d_close, series_len, first_valid, &mut d_out)?;
        self.stream.synchronize().map_err(|e| CudaWclpriceError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 { buf: d_out, rows: 1, cols: series_len })
    }

    // ---------------- Many-series × one-param (time-major) ----------------

    fn prepare_many_series_inputs(
        high_tm: &[f32], low_tm: &[f32], close_tm: &[f32], cols: usize, rows: usize,
    ) -> Result<Vec<i32>, CudaWclpriceError> {
        if cols == 0 || rows == 0 {
            return Err(CudaWclpriceError::InvalidInput("invalid dims".into()));
        }
        let expected = cols * rows;
        if high_tm.len() != expected || low_tm.len() != expected || close_tm.len() != expected {
            return Err(CudaWclpriceError::InvalidInput(format!(
                "time-major length mismatch: high={}, low={}, close={}, expected={}",
                high_tm.len(), low_tm.len(), close_tm.len(), expected
            )));
        }
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = None;
            for t in 0..rows {
                let idx = t * cols + s;
                if high_tm[idx].is_finite() && low_tm[idx].is_finite() && close_tm[idx].is_finite() {
                    fv = Some(t as i32);
                    break;
                }
            }
            first_valids[s] = fv.ok_or_else(|| CudaWclpriceError::InvalidInput(format!("series {} all NaN", s)))?;
        }
        Ok(first_valids)
    }

    fn launch_many_series_kernel(
        &self,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_close: &DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaWclpriceError> {
        let func = self
            .module
            .get_function("wclprice_many_series_one_param_time_major_f32")
            .map_err(|e| CudaWclpriceError::Cuda(e.to_string()))?;

        let block_x: u32 = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => Self::choose_block_x("WCLPRICE_MS_BLOCK_X", 256, 64),
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(64).min(1024),
        };
        unsafe { (*(self as *const _ as *mut CudaWclprice)).last_many = Some(ManySeriesKernelSelected::OneD { block_x }); }
        self.maybe_log_many_debug();

        let grid: GridSize = (cols as u32, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe {
            let mut high_ptr = d_high.as_device_ptr().as_raw();
            let mut low_ptr = d_low.as_device_ptr().as_raw();
            let mut close_ptr = d_close.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut high_ptr as *mut _ as *mut c_void,
                &mut low_ptr as *mut _ as *mut c_void,
                &mut close_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaWclpriceError::Cuda(e.to_string()))?;
        }
        Ok(())
    }

    pub fn wclprice_many_series_one_param_time_major_dev(
        &self,
        high_tm: &[f32], low_tm: &[f32], close_tm: &[f32], cols: usize, rows: usize,
    ) -> Result<DeviceArrayF32, CudaWclpriceError> {
        let first_valids = Self::prepare_many_series_inputs(high_tm, low_tm, close_tm, cols, rows)?;

        // VRAM: 3 inputs + first_valids + output
        let bytes = cols * rows * std::mem::size_of::<f32>() * 3
            + cols * std::mem::size_of::<i32>()
            + cols * rows * std::mem::size_of::<f32>();
        if !Self::will_fit(bytes, 64 * 1024 * 1024) {
            return Err(CudaWclpriceError::InvalidInput("insufficient VRAM for WCLPRICE many-series".into()));
        }

        let h_high = LockedBuffer::from_slice(high_tm).map_err(|e| CudaWclpriceError::Cuda(e.to_string()))?;
        let h_low = LockedBuffer::from_slice(low_tm).map_err(|e| CudaWclpriceError::Cuda(e.to_string()))?;
        let h_close = LockedBuffer::from_slice(close_tm).map_err(|e| CudaWclpriceError::Cuda(e.to_string()))?;
        let h_first = LockedBuffer::from_slice(&first_valids).map_err(|e| CudaWclpriceError::Cuda(e.to_string()))?;
        let mut d_high: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }.map_err(|e| CudaWclpriceError::Cuda(e.to_string()))?;
        let mut d_low: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }.map_err(|e| CudaWclpriceError::Cuda(e.to_string()))?;
        let mut d_close: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }.map_err(|e| CudaWclpriceError::Cuda(e.to_string()))?;
        let mut d_first_valids: DeviceBuffer<i32> = unsafe { DeviceBuffer::uninitialized(cols) }.map_err(|e| CudaWclpriceError::Cuda(e.to_string()))?;
        unsafe {
            d_high.copy_from(h_high.as_slice()).map_err(|e| CudaWclpriceError::Cuda(e.to_string()))?;
            d_low.copy_from(h_low.as_slice()).map_err(|e| CudaWclpriceError::Cuda(e.to_string()))?;
            d_close.copy_from(h_close.as_slice()).map_err(|e| CudaWclpriceError::Cuda(e.to_string()))?;
            d_first_valids.copy_from(h_first.as_slice()).map_err(|e| CudaWclpriceError::Cuda(e.to_string()))?;
        }
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }.map_err(|e| CudaWclpriceError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(&d_high, &d_low, &d_close, cols, rows, &d_first_valids, &mut d_out)?;
        self.stream.synchronize().map_err(|e| CudaWclpriceError::Cuda(e.to_string()))?;
        Ok(DeviceArrayF32 { buf: d_out, rows, cols })
    }

    // ---------------- Benches ----------------
    // Provide basic benches to integrate with benches/cuda_bench.rs.
}

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const MANY_SERIES_COLS: usize = 256;
    const MANY_SERIES_LEN: usize = 1_000_000 / 16; // keep VRAM reasonable

    fn bytes_one_series() -> usize {
        let in_bytes = 3 * ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }
    fn bytes_many_series() -> usize {
        let elems = MANY_SERIES_COLS * MANY_SERIES_LEN;
        let in_bytes = 3 * elems * std::mem::size_of::<f32>();
        let first = MANY_SERIES_COLS * std::mem::size_of::<i32>();
        let out_bytes = elems * std::mem::size_of::<f32>();
        in_bytes + first + out_bytes + 64 * 1024 * 1024
    }

    fn synth_hlc_from_close(close: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut high = close.to_vec();
        let mut low = close.to_vec();
        for i in 0..close.len() {
            let v = close[i]; if v.is_nan() { continue; }
            let x = i as f32 * 0.0025; let off = (0.002 * x.sin()).abs() + 0.15;
            high[i] = v + off; low[i] = v - off;
        }
        (high, low)
    }

    fn synth_hlc_time_major_from_close(close_tm: &[f32], cols: usize, rows: usize) -> (Vec<f32>, Vec<f32>) {
        let mut high = close_tm.to_vec();
        let mut low = close_tm.to_vec();
        for t in 0..rows {
            for s in 0..cols {
                let idx = t * cols + s; let v = close_tm[idx]; if v.is_nan() { continue; }
                let x = (t as f32) * 0.0023 + (s as f32) * 0.11; let off = (0.0029 * x.sin()).abs() + 0.1;
                high[idx] = v + off; low[idx] = v - off;
            }
        }
        (high, low)
    }

    struct SeriesState { cuda: CudaWclprice, high: Vec<f32>, low: Vec<f32>, close: Vec<f32> }
    impl CudaBenchState for SeriesState {
        fn launch(&mut self) {
            let _ = self.cuda.wclprice_batch_dev(&self.high, &self.low, &self.close, &WclpriceBatchRange)
                .expect("wclprice batch");
        }
    }
    fn prep_one_series() -> Box<dyn CudaBenchState> {
        let cuda = CudaWclprice::new(0).expect("CudaWclprice");
        let close = gen_series(ONE_SERIES_LEN); let (high, low) = synth_hlc_from_close(&close);
        Box::new(SeriesState { cuda, high, low, close })
    }

    struct ManyState { cuda: CudaWclprice, high_tm: Vec<f32>, low_tm: Vec<f32>, close_tm: Vec<f32>, cols: usize, rows: usize }
    impl CudaBenchState for ManyState {
        fn launch(&mut self) {
            let _ = self.cuda.wclprice_many_series_one_param_time_major_dev(
                &self.high_tm, &self.low_tm, &self.close_tm, self.cols, self.rows
            ).expect("wclprice many");
        }
    }
    fn prep_many_series() -> Box<dyn CudaBenchState> {
        let cuda = CudaWclprice::new(0).expect("CudaWclprice");
        let cols = MANY_SERIES_COLS; let rows = MANY_SERIES_LEN;
        let close_tm = gen_time_major_prices(cols, rows);
        let (high_tm, low_tm) = synth_hlc_time_major_from_close(&close_tm, cols, rows);
        Box::new(ManyState { cuda, high_tm, low_tm, close_tm, cols, rows })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "wclprice",
                "one_series",
                "wclprice_cuda_series",
                "1m",
                prep_one_series,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series()),
            CudaBenchScenario::new(
                "wclprice",
                "many_series_one_param",
                "wclprice_cuda_many",
                "256x62.5k",
                prep_many_series,
            )
            .with_sample_size(5)
            .with_mem_required(bytes_many_series()),
        ]
    }
}

