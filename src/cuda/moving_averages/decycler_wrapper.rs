//! CUDA wrapper for Ehlers Decycler (input minus two‑pole high‑pass).
//!
//! Parity goals with ALMA wrapper:
//! - Stream NON_BLOCKING, PTX loaded from OUT_DIR
//! - Policy enums for kernel selection (1D variants)
//! - VRAM estimation and basic checks, chunking kept simple (grid 1D)
//! - Bench registration via define_ma_period_benches!

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::decycler::{DecyclerBatchRange, DecyclerParams};
use cust::context::{CacheConfig, Context};
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, Function, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug)]
pub enum CudaDecyclerError {
    Cuda(String),
    InvalidInput(String),
}
impl fmt::Display for CudaDecyclerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaDecyclerError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaDecyclerError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}
impl std::error::Error for CudaDecyclerError {}

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
pub struct CudaDecyclerPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaDecyclerPolicy {
    fn default() -> Self {
        Self { batch: BatchKernelPolicy::Auto, many_series: ManySeriesKernelPolicy::Auto }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected { Plain { block_x: u32 } }
#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected { OneD { block_x: u32 } }

pub struct CudaDecycler {
    module: Module,
    stream: Stream,
    _context: Context,
    device_id: u32,
    policy: CudaDecyclerPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaDecycler {
    pub fn new(device_id: usize) -> Result<Self, CudaDecyclerError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaDecyclerError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaDecyclerError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaDecyclerError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/decycler_kernel.ptx"));
        // Simple, robust JIT opts
        let module = Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext])
            .or_else(|_| Module::from_ptx(ptx, &[]))
            .map_err(|e| CudaDecyclerError::Cuda(e.to_string()))?;
        // Recurrence is memory bound; prefer L1
        if let Ok(mut f) = module.get_function("decycler_batch_f32") {
            let _ = f.set_cache_config(CacheConfig::PreferL1);
        }
        if let Ok(mut f) = module.get_function("decycler_many_series_one_param_f32") {
            let _ = f.set_cache_config(CacheConfig::PreferL1);
        }

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaDecyclerError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaDecyclerPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn new_with_policy(device_id: usize, policy: CudaDecyclerPolicy) -> Result<Self, CudaDecyclerError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }
    pub fn set_policy(&mut self, policy: CudaDecyclerPolicy) { self.policy = policy; }
    pub fn policy(&self) -> &CudaDecyclerPolicy { &self.policy }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> { self.last_batch }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> { self.last_many }
    pub fn synchronize(&self) -> Result<(), CudaDecyclerError> {
        self.stream.synchronize().map_err(|e| CudaDecyclerError::Cuda(e.to_string()))
    }

    #[inline]
    fn calc_launch_1d(&self, func: &Function, n_items: usize, override_block: Option<u32>) -> (BlockSize, GridSize) {
        if let Some(bx) = override_block {
            let bx = bx.max(32);
            let gx = ((n_items as u32 + bx - 1) / bx).max(1);
            return ((bx, 1, 1).into(), (gx, 1, 1).into());
        }
        let (min_grid, block_x) = func
            .suggested_launch_configuration(0usize, BlockSize::xyz(0, 0, 0))
            .unwrap_or((1, 256));
        let block_x = block_x.max(128);
        let mut grid_x = ((n_items as u32 + block_x - 1) / block_x).max(min_grid.max(1));
        if let Ok(dev) = Device::get_device(self.device_id) {
            if let Ok(max_gx) = dev.get_attribute(DeviceAttribute::MaxGridDimX) {
                grid_x = grid_x.min(max_gx as u32);
            }
        }
        ((block_x, 1, 1).into(), (grid_x, 1, 1).into())
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        match std::env::var("CUDA_MEM_CHECK") { Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"), Err(_) => true }
    }
    #[inline]
    fn device_mem_info() -> Option<(usize, usize)> { mem_get_info().ok() }
    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> bool {
        if !Self::mem_check_enabled() { return true; }
        if let Some((free, _)) = Self::device_mem_info() { required_bytes.saturating_add(headroom_bytes) <= free } else { true }
    }
    #[inline]
    fn maybe_log_batch_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged { return; }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                let per = std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] DECYCLER batch selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut CudaDecycler)).debug_batch_logged = true; }
            }
        }
    }
    #[inline]
    fn maybe_log_many_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_many_logged { return; }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                let per = std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] DECYCLER many-series selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut CudaDecycler)).debug_many_logged = true; }
            }
        }
    }

    // -------- Batch (one series × many params) --------

    pub fn decycler_batch_dev(&self, data_f32: &[f32], sweep: &DecyclerBatchRange) -> Result<DeviceArrayF32, CudaDecyclerError> {
        let prepared = Self::prepare_batch_inputs(data_f32, sweep)?;
        let n = prepared.combos.len();

        let prices_bytes = prepared.series_len * std::mem::size_of::<f32>();
        let params_bytes = n * (std::mem::size_of::<i32>() + 3 * std::mem::size_of::<f32>());
        let diff_bytes = prepared.series_len * std::mem::size_of::<f32>();
        let out_bytes = n * prepared.series_len * std::mem::size_of::<f32>();
        let required = prices_bytes + params_bytes + diff_bytes + out_bytes;
        if !Self::will_fit(required, 64 * 1024 * 1024) {
            return Err(CudaDecyclerError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        let d_prices = unsafe { DeviceBuffer::from_slice_async(data_f32, &self.stream) }
            .map_err(|e| CudaDecyclerError::Cuda(e.to_string()))?;
        let d_periods = unsafe { DeviceBuffer::from_slice_async(&prepared.periods_i32, &self.stream) }
            .map_err(|e| CudaDecyclerError::Cuda(e.to_string()))?;
        let d_c = unsafe { DeviceBuffer::from_slice_async(&prepared.c_vals, &self.stream) }
            .map_err(|e| CudaDecyclerError::Cuda(e.to_string()))?;
        let d_two = unsafe { DeviceBuffer::from_slice_async(&prepared.two_1m_vals, &self.stream) }
            .map_err(|e| CudaDecyclerError::Cuda(e.to_string()))?;
        let d_neg = unsafe { DeviceBuffer::from_slice_async(&prepared.neg_oma_sq_vals, &self.stream) }
            .map_err(|e| CudaDecyclerError::Cuda(e.to_string()))?;
        let d_diff = unsafe { DeviceBuffer::from_slice_async(&prepared.diff, &self.stream) }
            .map_err(|e| CudaDecyclerError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized_async(prepared.series_len * n, &self.stream)
                .map_err(|e| CudaDecyclerError::Cuda(e.to_string()))?
        };

        self.launch_batch_kernel(
            &d_prices,
            &d_periods,
            &d_c,
            &d_two,
            &d_neg,
            &d_diff,
            prepared.series_len,
            n,
            prepared.first_valid,
            &mut d_out,
        )?;

        Ok(DeviceArrayF32 { buf: d_out, rows: n, cols: prepared.series_len })
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_c: &DeviceBuffer<f32>,
        d_two_1m: &DeviceBuffer<f32>,
        d_neg_oma_sq: &DeviceBuffer<f32>,
        d_diff: &DeviceBuffer<f32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaDecyclerError> {
        let func = self.module.get_function("decycler_batch_f32").map_err(|e| CudaDecyclerError::Cuda(e.to_string()))?;
        let (block, grid) = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => ((block_x, 1, 1).into(), (((n_combos as u32 + block_x - 1) / block_x).max(1), 1, 1).into()),
            BatchKernelPolicy::Auto => self.calc_launch_1d(&func, n_combos, None),
        };
        unsafe {
            let mut p_ptr = d_prices.as_device_ptr().as_raw();
            let mut per_ptr = d_periods.as_device_ptr().as_raw();
            let mut c_ptr = d_c.as_device_ptr().as_raw();
            let mut two_ptr = d_two_1m.as_device_ptr().as_raw();
            let mut neg_ptr = d_neg_oma_sq.as_device_ptr().as_raw();
            let mut diff_ptr = d_diff.as_device_ptr().as_raw();
            let mut len_i = series_len as i32;
            let mut n_i = n_combos as i32;
            let mut first_i = first_valid as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 10] = [
                &mut p_ptr as *mut _ as *mut c_void,
                &mut per_ptr as *mut _ as *mut c_void,
                &mut c_ptr as *mut _ as *mut c_void,
                &mut two_ptr as *mut _ as *mut c_void,
                &mut neg_ptr as *mut _ as *mut c_void,
                &mut diff_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut n_i as *mut _ as *mut c_void,
                &mut first_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, &mut args).map_err(|e| CudaDecyclerError::Cuda(e.to_string()))?;
        }
        unsafe {
            let bx = match self.policy.batch { BatchKernelPolicy::Plain { block_x } => block_x, BatchKernelPolicy::Auto => block.x };
            (*(self as *const _ as *mut CudaDecycler)).last_batch = Some(BatchKernelSelected::Plain { block_x: bx });
        }
        self.maybe_log_batch_debug();
        Ok(())
    }

    // -------- Many-series × one-param (time-major) --------

    pub fn decycler_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &DecyclerParams,
    ) -> Result<DeviceArrayF32, CudaDecyclerError> {
        let prepared = Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;

        let elems = cols * rows;
        let prices_bytes = elems * std::mem::size_of::<f32>();
        let first_bytes = cols * std::mem::size_of::<i32>();
        let out_bytes = elems * std::mem::size_of::<f32>();
        let required = prices_bytes + first_bytes + out_bytes;
        if !Self::will_fit(required, 64 * 1024 * 1024) {
            return Err(CudaDecyclerError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        let d_prices = unsafe { DeviceBuffer::from_slice_async(data_tm_f32, &self.stream) }
            .map_err(|e| CudaDecyclerError::Cuda(e.to_string()))?;
        let d_first = DeviceBuffer::from_slice(&prepared.first_valids)
            .map_err(|e| CudaDecyclerError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized_async(elems, &self.stream)
                .map_err(|e| CudaDecyclerError::Cuda(e.to_string()))?
        };
        self.launch_many_series_kernel(
            &d_prices,
            &d_first,
            prepared.period,
            prepared.c,
            prepared.two_1m,
            prepared.neg_oma_sq,
            cols,
            rows,
            &mut d_out,
        )?;
        Ok(DeviceArrayF32 { buf: d_out, rows, cols })
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        period: i32,
        c: f32,
        two_1m: f32,
        neg_oma_sq: f32,
        num_series: usize,
        series_len: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaDecyclerError> {
        let func = self.module.get_function("decycler_many_series_one_param_f32")
            .map_err(|e| CudaDecyclerError::Cuda(e.to_string()))?;
        let (block, grid) = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => ((block_x, 1, 1).into(), (((num_series as u32 + block_x - 1) / block_x).max(1), 1, 1).into()),
            ManySeriesKernelPolicy::Auto => self.calc_launch_1d(&func, num_series, None),
        };
        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut period_i = period;
            let mut c_val = c;
            let mut two_val = two_1m;
            let mut neg_val = neg_oma_sq;
            let mut cols_i = num_series as i32;
            let mut rows_i = series_len as i32;
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 10] = [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut c_val as *mut _ as *mut c_void,
                &mut two_val as *mut _ as *mut c_void,
                &mut neg_val as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
                std::ptr::null_mut(), // padding (unused)
            ];
            self.stream.launch(&func, grid, block, 0, &mut args).map_err(|e| CudaDecyclerError::Cuda(e.to_string()))?;
        }
        unsafe {
            let bx = match self.policy.many_series { ManySeriesKernelPolicy::OneD { block_x } => block_x, ManySeriesKernelPolicy::Auto => block.x };
            (*(self as *const _ as *mut CudaDecycler)).last_many = Some(ManySeriesKernelSelected::OneD { block_x: bx });
        }
        self.maybe_log_many_debug();
        Ok(())
    }

    // -------- Prep helpers --------

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &DecyclerBatchRange,
    ) -> Result<PreparedDecyclerBatch, CudaDecyclerError> {
        let series_len = data_f32.len();
        if series_len == 0 { return Err(CudaDecyclerError::InvalidInput("empty series".into())); }
        let combos = expand_grid(sweep);
        if combos.is_empty() { return Err(CudaDecyclerError::InvalidInput("empty param grid".into())); }

        // first valid (non-NaN) index
        let mut first_valid: Option<usize> = None;
        for i in 0..series_len { if data_f32[i].is_finite() { first_valid = Some(i); break; } }
        let fv = first_valid.ok_or_else(|| CudaDecyclerError::InvalidInput("all values are NaN".into()))?;
        let max_p = combos.iter().map(|c| c.hp_period.unwrap()).max().unwrap();
        if series_len - fv < max_p {
            return Err(CudaDecyclerError::InvalidInput(format!(
                "not enough valid data: needed >= {}, valid = {}", max_p, series_len - fv
            )));
        }

        // host precompute: second difference reused across rows
        let mut diff = vec![0f32; series_len];
        for i in (fv + 2)..series_len {
            let x  = data_f32[i];
            let x1 = data_f32[i - 1];
            let x2 = data_f32[i - 2];
            diff[i] = x - 2.0 * x1 + x2;
        }

        // per-row coefficients
        let mut periods_i32 = Vec::with_capacity(combos.len());
        let mut c_vals = Vec::with_capacity(combos.len());
        let mut two_1m_vals = Vec::with_capacity(combos.len());
        let mut neg_oma_sq_vals = Vec::with_capacity(combos.len());
        for p in &combos {
            let period = p.hp_period.unwrap();
            let k = p.k.unwrap();
            let coeffs = compute_coefficients(period, k);
            periods_i32.push(period as i32);
            c_vals.push(coeffs.c);
            two_1m_vals.push(coeffs.two_1m);
            neg_oma_sq_vals.push(coeffs.neg_oma_sq);
        }

        Ok(PreparedDecyclerBatch { combos, first_valid: fv, series_len, periods_i32, c_vals, two_1m_vals, neg_oma_sq_vals, diff })
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32], cols: usize, rows: usize, params: &DecyclerParams,
    ) -> Result<PreparedDecyclerMany, CudaDecyclerError> {
        if cols == 0 || rows == 0 { return Err(CudaDecyclerError::InvalidInput("empty matrix".into())); }
        if data_tm_f32.len() != cols * rows { return Err(CudaDecyclerError::InvalidInput("data shape mismatch".into())); }
        let period = params.hp_period.unwrap_or(125);
        let k = params.k.unwrap_or(0.707);
        if period < 2 { return Err(CudaDecyclerError::InvalidInput("hp_period must be >= 2".into())); }
        if !(k.is_finite()) || k <= 0.0 { return Err(CudaDecyclerError::InvalidInput("k must be positive and finite".into())); }

        // per-series first valid
        let needed = period;
        let mut first_valids = Vec::with_capacity(cols);
        for s in 0..cols {
            let mut fv: Option<usize> = None;
            for t in 0..rows {
                let v = data_tm_f32[t * cols + s];
                if v.is_finite() { fv = Some(t); break; }
            }
            let fvu = fv.ok_or_else(|| CudaDecyclerError::InvalidInput(format!("series {} all NaN", s)))?;
            if rows - fvu < needed {
                return Err(CudaDecyclerError::InvalidInput(format!(
                    "series {} not enough valid data: needed >= {}, valid = {}", s, needed, rows - fvu
                )));
            }
            first_valids.push(fvu as i32);
        }

        let coeffs = compute_coefficients(period, k);
        Ok(PreparedDecyclerMany { first_valids, period: period as i32, c: coeffs.c, two_1m: coeffs.two_1m, neg_oma_sq: coeffs.neg_oma_sq })
    }
}

// ---- Prepared inputs ----
struct PreparedDecyclerBatch {
    combos: Vec<DecyclerParams>,
    first_valid: usize,
    series_len: usize,
    periods_i32: Vec<i32>,
    c_vals: Vec<f32>,
    two_1m_vals: Vec<f32>,
    neg_oma_sq_vals: Vec<f32>,
    diff: Vec<f32>,
}
struct PreparedDecyclerMany {
    first_valids: Vec<i32>,
    period: i32,
    c: f32,
    two_1m: f32,
    neg_oma_sq: f32,
}

// ---- Benches ----
pub mod benches {
    use super::*;
    use crate::define_ma_period_benches;
    define_ma_period_benches!(
        decycler_benches,
        CudaDecycler,
        crate::indicators::decycler::DecyclerBatchRange,
        crate::indicators::decycler::DecyclerParams,
        decycler_batch_dev,
        decycler_many_series_one_param_time_major_dev,
        crate::indicators::decycler::DecyclerBatchRange { hp_period: (10, 10 + PARAM_SWEEP - 1, 1), k: (0.5, 0.5, 0.0) },
        crate::indicators::decycler::DecyclerParams { hp_period: Some(64), k: Some(0.5) },
        "decycler",
        "decycler"
    );
    pub use decycler_benches::bench_profiles;
}

// ---- Utilities ----
struct Coefficients { c: f32, two_1m: f32, neg_oma_sq: f32 }
fn compute_coefficients(period: usize, k: f64) -> Coefficients {
    use std::f64::consts::PI;
    let theta = 2.0 * PI * k / period as f64;
    let sin_v = theta.sin();
    let cos_v = theta.cos();
    let alpha = 1.0 + ((sin_v - 1.0) / cos_v);
    let c = (1.0 - 0.5 * alpha).powi(2);
    let oma = 1.0 - alpha;
    let two_1m = 2.0 * oma;
    let neg_oma_sq = -(oma * oma);
    Coefficients { c: c as f32, two_1m: two_1m as f32, neg_oma_sq: neg_oma_sq as f32 }
}

fn expand_grid(range: &DecyclerBatchRange) -> Vec<DecyclerParams> {
    fn axis_usize(a: (usize, usize, usize)) -> Vec<usize> {
        let (s, e, st) = a; if st == 0 || s == e { return vec![s]; } (s..=e).step_by(st).collect()
    }
    fn axis_f64(a: (f64, f64, f64)) -> Vec<f64> {
        let (s, e, st) = a;
        if st.abs() < 1e-12 || (s - e).abs() < 1e-12 { return vec![s]; }
        let mut v = Vec::new();
        let mut cur = s; while cur <= e + 1e-12 { v.push(cur); cur += st; } v
    }
    let ps = axis_usize(range.hp_period);
    let ks = axis_f64(range.k);
    let mut out = Vec::with_capacity(ps.len() * ks.len());
    for &p in &ps { for &k in &ks { out.push(DecyclerParams { hp_period: Some(p), k: Some(k) }); } }
    out
}

