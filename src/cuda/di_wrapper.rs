//! CUDA wrapper for DI (+DI, -DI) kernels.
//!
//! Parity goals with ALMA/CWMA wrappers:
//! - Policy surface for batch and many-series launches (kept simple here).
//! - PTX load via include_str!(concat!(env!("OUT_DIR"), "/di_kernel.ptx")) with stable JIT opts.
//! - Non-blocking stream, VRAM checks, combo chunking when needed.
//! - Warmup/NaN rules identical to scalar: warm = first_valid + period - 1.

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::di::{DiBatchRange, DiParams};
use cust::context::Context;
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug)]
pub enum CudaDiError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaDiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaDiError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaDiError::InvalidInput(s) => write!(f, "Invalid input: {}", s),
        }
    }
}
impl std::error::Error for CudaDiError {}

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

// Introspection enums (parity with ALMA/CWMA style)
#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected { Plain { block_x: u32 } }
#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected { OneD { block_x: u32 } }

#[derive(Clone, Copy, Debug)]
pub struct CudaDiPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaDiPolicy {
    fn default() -> Self {
        Self { batch: BatchKernelPolicy::Auto, many_series: ManySeriesKernelPolicy::Auto }
    }
}

pub struct DeviceArrayF32Pair {
    pub plus: DeviceArrayF32,
    pub minus: DeviceArrayF32,
}
impl DeviceArrayF32Pair {
    #[inline] pub fn rows(&self) -> usize { self.plus.rows }
    #[inline] pub fn cols(&self) -> usize { self.plus.cols }
}

pub struct CudaDi {
    module: Module,
    stream: Stream,
    _context: Context,
    device_id: u32,
    policy: CudaDiPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
    sm_count: u32,
}

impl CudaDi {
    pub fn new(device_id: usize) -> Result<Self, CudaDiError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaDiError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaDiError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaDiError::Cuda(e.to_string()))?;

        // Query SM count for launch heuristics
        let sm_count = device
            .get_attribute(DeviceAttribute::MultiprocessorCount)
            .map_err(|e| CudaDiError::Cuda(e.to_string()))? as u32;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/di_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O4),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => {
                if let Ok(m) = Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext])
                {
                    m
                } else {
                    Module::from_ptx(ptx, &[]).map_err(|e| CudaDiError::Cuda(e.to_string()))?
                }
            }
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaDiError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaDiPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
            sm_count,
        })
    }

    pub fn set_policy(&mut self, policy: CudaDiPolicy) { self.policy = policy; }
    pub fn synchronize(&self) -> Result<(), CudaDiError> { self.stream.synchronize().map_err(|e| CudaDiError::Cuda(e.to_string())) }

    // ---- Helpers ------------------------------------------------------------
    fn device_will_fit(bytes: usize, headroom: usize) -> bool {
        let check = match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        };
        if !check { return true; }
        if let Ok((free, _)) = mem_get_info() { bytes.saturating_add(headroom) <= free } else { true }
    }

    fn first_valid_hlc(high: &[f32], low: &[f32], close: &[f32]) -> Result<usize, CudaDiError> {
        if high.len() == 0 || low.len() == 0 || close.len() == 0 {
            return Err(CudaDiError::InvalidInput("empty input".into()));
        }
        let n = high.len().min(low.len()).min(close.len());
        for i in 0..n { if !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan() { return Ok(i); } }
        Err(CudaDiError::InvalidInput("all values are NaN".into()))
    }

    fn expand_periods(range: &DiBatchRange) -> Vec<usize> {
        let (start, end, step) = range.period;
        if step == 0 || start == end { vec![start] } else { (start..=end).step_by(step).collect() }
    }

    fn chunk_size_for_batch(n_combos: usize, len: usize) -> usize {
        // Inputs: 3×len (up,dn,tr) + params (periods,warm) + outputs: 2×(combos×len)
        let in_bytes = 3 * len * std::mem::size_of::<f32>();
        let params_bytes = n_combos * (2 * std::mem::size_of::<i32>());
        let out_per_combo = 2 * len * std::mem::size_of::<f32>();
        let headroom = 64 * 1024 * 1024;
        let mut chunk = n_combos.max(1);
        while chunk > 1 {
            let need = in_bytes + params_bytes + chunk * out_per_combo + headroom;
            if Self::device_will_fit(need, 0) { break; }
            chunk = (chunk + 1) / 2;
        }
        chunk.max(1)
    }

    #[inline]
    fn maybe_log_batch_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged { return; }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                let per_scenario = std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scenario || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] DI batch selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut CudaDi)).debug_batch_logged = true; }
            }
        }
    }

    #[inline]
    fn maybe_log_many_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_many_logged { return; }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                let per_scenario = std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scenario || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] DI many-series selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut CudaDi)).debug_many_logged = true; }
            }
        }
    }

    // Build up/dn/tr precompute on host (mirrors scalar).
    fn build_up_dn_tr(high: &[f32], low: &[f32], close: &[f32], first: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let n = high.len();
        let mut up = vec![0f32; n];
        let mut dn = vec![0f32; n];
        let mut tr = vec![0f32; n];
        if n == 0 { return (up, dn, tr); }
        let mut prev_h = high[first];
        let mut prev_l = low[first];
        let mut prev_c = close[first];
        let mut i = first + 1;
        while i < n {
            let ch = high[i];
            let cl = low[i];
            let dp = ch - prev_h;
            let dm = prev_l - cl;
            if dp > dm && dp > 0.0 { up[i] = dp; }
            if dm > dp && dm > 0.0 { dn[i] = dm; }
            let mut t = ch - cl;
            let t2 = (ch - prev_c).abs(); if t2 > t { t = t2; }
            let t3 = (cl - prev_c).abs(); if t3 > t { t = t3; }
            tr[i] = t;
            prev_h = ch; prev_l = cl; prev_c = close[i];
            i += 1;
        }
        (up, dn, tr)
    }

    fn launch_batch_from_precomp(
        &self,
        d_up: &DeviceBuffer<f32>,
        d_dn: &DeviceBuffer<f32>,
        d_tr: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_warms: &DeviceBuffer<i32>,
        len: usize,
        first_valid: usize,
        n_combos: usize,
        d_plus: &mut DeviceBuffer<f32>,
        d_minus: &mut DeviceBuffer<f32>,
        row_offset: usize,
        chunk_len: usize,
    ) -> Result<(), CudaDiError> {
        let func = self.module.get_function("di_batch_from_precomputed_f32")
            .map_err(|e| CudaDiError::Cuda(e.to_string()))?;
        let block_x = match self.policy.batch { BatchKernelPolicy::Plain { block_x } => block_x, _ => 256 };
        // Grid-stride over combos: size grid by device parallelism (~8 blocks/SM), cap by chunk size
        let target_blocks = self.sm_count.saturating_mul(8).max(1);
        let grid_x = core::cmp::min(chunk_len as u32, target_blocks).max(1);
        let grid: GridSize = (grid_x, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut up_ptr = d_up.as_device_ptr().as_raw();
            let mut dn_ptr = d_dn.as_device_ptr().as_raw();
            let mut tr_ptr = d_tr.as_device_ptr().as_raw();
            let mut per_ptr = d_periods.as_device_ptr().add(row_offset).as_raw();
            let mut warm_ptr = d_warms.as_device_ptr().add(row_offset).as_raw();
            let mut len_i = len as i32;
            let mut first_i = first_valid as i32;
            let mut comb_i = chunk_len as i32;
            let mut plus_ptr = d_plus.as_device_ptr().add(row_offset * len).as_raw();
            let mut minus_ptr = d_minus.as_device_ptr().add(row_offset * len).as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut up_ptr as *mut _ as *mut c_void,
                &mut dn_ptr as *mut _ as *mut c_void,
                &mut tr_ptr as *mut _ as *mut c_void,
                &mut per_ptr as *mut _ as *mut c_void,
                &mut warm_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut first_i as *mut _ as *mut c_void,
                &mut comb_i as *mut _ as *mut c_void,
                &mut plus_ptr as *mut _ as *mut c_void,
                &mut minus_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)
                .map_err(|e| CudaDiError::Cuda(e.to_string()))?;
        }
        // Record selection for introspection
        unsafe { (*(self as *const _ as *mut CudaDi)).last_batch = Some(BatchKernelSelected::Plain { block_x }); }
        self.maybe_log_batch_debug();
        Ok(())
    }

    pub fn di_batch_dev(
        &self,
        high: &[f32],
        low: &[f32],
        close: &[f32],
        sweep: &DiBatchRange,
    ) -> Result<(DeviceArrayF32, DeviceArrayF32, Vec<DiParams>), CudaDiError> {
        if high.len() != low.len() || low.len() != close.len() {
            return Err(CudaDiError::InvalidInput("length mismatch".into()));
        }
        let len = close.len();
        if len == 0 { return Err(CudaDiError::InvalidInput("empty input".into())); }
        let first_valid = Self::first_valid_hlc(high, low, close)?;

        let periods = Self::expand_periods(sweep);
        if periods.is_empty() { return Err(CudaDiError::InvalidInput("no parameter combos".into())); }
        for &p in &periods {
            if p == 0 || p > len || (len - first_valid) < p {
                return Err(CudaDiError::InvalidInput(format!(
                    "invalid period {} for data length {} (valid after {}: {})",
                    p, len, first_valid, len - first_valid
                )));
            }
        }

        // Host precompute shared across combos
        let (up_h, dn_h, tr_h) = Self::build_up_dn_tr(high, low, close, first_valid);

        // Async uploads to overlap H2D with later work on the same stream
        let d_up: DeviceBuffer<f32> = unsafe { DeviceBuffer::from_slice_async(&up_h, &self.stream) }
            .map_err(|e| CudaDiError::Cuda(e.to_string()))?;
        let d_dn: DeviceBuffer<f32> = unsafe { DeviceBuffer::from_slice_async(&dn_h, &self.stream) }
            .map_err(|e| CudaDiError::Cuda(e.to_string()))?;
        let d_tr: DeviceBuffer<f32> = unsafe { DeviceBuffer::from_slice_async(&tr_h, &self.stream) }
            .map_err(|e| CudaDiError::Cuda(e.to_string()))?;

        let n_combos = periods.len();
        let periods_i32: Vec<i32> = periods.iter().map(|&p| p as i32).collect();
        let warms_i32: Vec<i32> = periods.iter().map(|&p| (first_valid + p - 1) as i32).collect();
        let d_periods: DeviceBuffer<i32> = unsafe { DeviceBuffer::from_slice_async(&periods_i32, &self.stream) }
            .map_err(|e| CudaDiError::Cuda(e.to_string()))?;
        let d_warms: DeviceBuffer<i32> = unsafe { DeviceBuffer::from_slice_async(&warms_i32, &self.stream) }
            .map_err(|e| CudaDiError::Cuda(e.to_string()))?;

        // Allocate outputs (combos x len)
        let elems = n_combos * len;
        let mut d_plus: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(elems, &self.stream) }
            .map_err(|e| CudaDiError::Cuda(e.to_string()))?;
        let mut d_minus: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(elems, &self.stream) }
            .map_err(|e| CudaDiError::Cuda(e.to_string()))?;

        // VRAM-aware chunking across combos if needed
        let chunk = Self::chunk_size_for_batch(n_combos, len);
        let mut processed = 0usize;
        while processed < n_combos {
            let this_chunk = chunk.min(n_combos - processed);
            self.launch_batch_from_precomp(
                &d_up, &d_dn, &d_tr, &d_periods, &d_warms, len, first_valid,
                this_chunk,
                &mut d_plus, &mut d_minus,
                processed,
                this_chunk,
            )?;
            processed += this_chunk;
        }

        self.synchronize()?;

        // Wrap for return
        let plus = DeviceArrayF32 { buf: d_plus, rows: n_combos, cols: len };
        let minus = DeviceArrayF32 { buf: d_minus, rows: n_combos, cols: len };
        let combos: Vec<DiParams> = periods.into_iter().map(|p| DiParams { period: Some(p) }).collect();
        Ok((plus, minus, combos))
    }

    pub fn di_many_series_one_param_time_major_dev(
        &self,
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
    ) -> Result<DeviceArrayF32Pair, CudaDiError> {
        if cols == 0 || rows == 0 { return Err(CudaDiError::InvalidInput("invalid dims".into())); }
        if high_tm.len() != cols * rows || low_tm.len() != cols * rows || close_tm.len() != cols * rows {
            return Err(CudaDiError::InvalidInput("flat input length mismatch".into()));
        }
        if period == 0 { return Err(CudaDiError::InvalidInput("period must be > 0".into())); }

        // Build first_valid per series on host
        let mut first_valids = vec![-1i32; cols];
        for s in 0..cols {
            for t in 0..rows {
                let idx = t * cols + s;
                if !high_tm[idx].is_nan() && !low_tm[idx].is_nan() && !close_tm[idx].is_nan() {
                    first_valids[s] = t as i32; break;
                }
            }
        }

        let d_high: DeviceBuffer<f32> = unsafe { DeviceBuffer::from_slice_async(high_tm, &self.stream) }
            .map_err(|e| CudaDiError::Cuda(e.to_string()))?;
        let d_low:  DeviceBuffer<f32> = unsafe { DeviceBuffer::from_slice_async(low_tm, &self.stream) }
            .map_err(|e| CudaDiError::Cuda(e.to_string()))?;
        let d_close: DeviceBuffer<f32> = unsafe { DeviceBuffer::from_slice_async(close_tm, &self.stream) }
            .map_err(|e| CudaDiError::Cuda(e.to_string()))?;
        let d_first: DeviceBuffer<i32> = unsafe { DeviceBuffer::from_slice_async(&first_valids, &self.stream) }
            .map_err(|e| CudaDiError::Cuda(e.to_string()))?;

        let mut d_plus_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(cols * rows, &self.stream) }
            .map_err(|e| CudaDiError::Cuda(e.to_string()))?;
        let mut d_minus_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(cols * rows, &self.stream) }
            .map_err(|e| CudaDiError::Cuda(e.to_string()))?;

        // Launch config: warp-per-series 1D tiling
        let func = self.module.get_function("di_many_series_one_param_f32")
            .map_err(|e| CudaDiError::Cuda(e.to_string()))?;
        let block_x = match self.policy.many_series { ManySeriesKernelPolicy::OneD { block_x } => block_x, _ => 128 };
        let warps_per_block = (block_x / 32).max(1);
        let grid_x = ((cols as u32) + warps_per_block as u32 - 1) / warps_per_block as u32;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut h_ptr = d_high.as_device_ptr().as_raw();
            let mut l_ptr = d_low.as_device_ptr().as_raw();
            let mut c_ptr = d_close.as_device_ptr().as_raw();
            let mut fv_ptr = d_first.as_device_ptr().as_raw();
            let mut per_i = period as i32;
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut plus_ptr = d_plus_tm.as_device_ptr().as_raw();
            let mut minus_ptr= d_minus_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut h_ptr as *mut _ as *mut c_void,
                &mut l_ptr as *mut _ as *mut c_void,
                &mut c_ptr as *mut _ as *mut c_void,
                &mut fv_ptr as *mut _ as *mut c_void,
                &mut per_i as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut plus_ptr as *mut _ as *mut c_void,
                &mut minus_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)
                .map_err(|e| CudaDiError::Cuda(e.to_string()))?;
        }

        self.synchronize()?;
        // Record selection and optionally log
        unsafe { (*(self as *const _ as *mut CudaDi)).last_many = Some(ManySeriesKernelSelected::OneD { block_x }); }
        self.maybe_log_many_debug();
        Ok(DeviceArrayF32Pair {
            plus: DeviceArrayF32 { buf: d_plus_tm, rows, cols },
            minus: DeviceArrayF32 { buf: d_minus_tm, rows, cols },
        })
    }
}

// ---------- Benches (registered by benches/cuda_bench.rs) -------------------
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_time_major_prices;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "di",
                "batch_dev",
                "di_cuda_batch_dev",
                "60k_x_50",
                prep_di_batch_box,
            ).with_inner_iters(6),
            CudaBenchScenario::new(
                "di",
                "many_series_one_param",
                "di_cuda_many_series_one_param",
                "250x1m",
                prep_di_many_series_box,
            ).with_inner_iters(3),
        ]
    }

    struct DiBatchState {
        cuda: CudaDi,
        d_up: DeviceBuffer<f32>,
        d_dn: DeviceBuffer<f32>,
        d_tr: DeviceBuffer<f32>,
        d_periods: DeviceBuffer<i32>,
        d_warms: DeviceBuffer<i32>,
        d_plus: DeviceBuffer<f32>,
        d_minus: DeviceBuffer<f32>,
        len: usize,
        first: usize,
        combos: usize,
    }
    impl CudaBenchState for DiBatchState { fn launch(&mut self) {
        let _ = self.cuda.launch_batch_from_precomp(
            &self.d_up,&self.d_dn,&self.d_tr,&self.d_periods,&self.d_warms,
            self.len, self.first, self.combos, &mut self.d_plus, &mut self.d_minus,
            0, self.combos
        ).expect("di batch launch");
        self.cuda.synchronize().expect("sync");
    }}

    fn prep_di_batch() -> DiBatchState {
        let mut cuda = CudaDi::new(0).expect("cuda di");
        cuda.set_policy(CudaDiPolicy { batch: BatchKernelPolicy::Auto, many_series: ManySeriesKernelPolicy::Auto });

        let len = 60_000usize;
        // Synthesize H/L/C from a smooth close
        let mut close = vec![f32::NAN; len];
        for i in 5..len { let x = i as f32; close[i] = (x * 0.0013).sin() + 0.00011 * x; }
        let mut high = close.clone();
        let mut low  = close.clone();
        for i in 0..len { if close[i].is_nan() { continue; } let off = 0.12 + (i as f32 * 0.00027).cos().abs() * 0.01; high[i] = close[i] + off; low[i] = close[i] - off; }

        let first = close.iter().position(|v| !v.is_nan()).unwrap_or(0);
        let (up, dn, tr) = CudaDi::build_up_dn_tr(&high, &low, &close, first);
        let d_up = DeviceBuffer::from_slice(&up).expect("up");
        let d_dn = DeviceBuffer::from_slice(&dn).expect("dn");
        let d_tr = DeviceBuffer::from_slice(&tr).expect("tr");
        let periods: Vec<i32> = (5..=254).step_by(5).take(50).map(|p| p as i32).collect();
        let warms: Vec<i32> = periods.iter().map(|&p| first as i32 + p - 1).collect();
        let d_periods = DeviceBuffer::from_slice(&periods).expect("per");
        let d_warms   = DeviceBuffer::from_slice(&warms).expect("warm");
        let combos = periods.len();
        let d_plus: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(combos * len) }.expect("plus");
        let d_minus: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(combos * len) }.expect("minus");

        DiBatchState { cuda, d_up, d_dn, d_tr, d_periods, d_warms, d_plus, d_minus, len, first, combos }
    }
    fn prep_di_batch_box() -> Box<dyn CudaBenchState> { Box::new(prep_di_batch()) }

    struct DiManyState {
        cuda: CudaDi,
        d_high: DeviceBuffer<f32>,
        d_low: DeviceBuffer<f32>,
        d_close: DeviceBuffer<f32>,
        d_first: DeviceBuffer<i32>,
        d_plus_tm: DeviceBuffer<f32>,
        d_minus_tm: DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        period: usize,
    }
    impl CudaBenchState for DiManyState { fn launch(&mut self) {
        // reuse pre-staged buffers via direct kernel launch for deterministic bench
        let func = self.cuda.module.get_function("di_many_series_one_param_f32").unwrap();
        let block_x: u32 = 128; let warps = (block_x/32).max(1); let grid_x = ((self.cols as u32)+warps-1)/warps;
        let grid: GridSize = (grid_x.max(1),1,1).into(); let block: BlockSize = (block_x,1,1).into();
        unsafe {
            let mut h = self.d_high.as_device_ptr().as_raw();
            let mut l = self.d_low.as_device_ptr().as_raw();
            let mut c = self.d_close.as_device_ptr().as_raw();
            let mut fv= self.d_first.as_device_ptr().as_raw();
            let mut p  = self.period as i32; let mut cols = self.cols as i32; let mut rows = self.rows as i32;
            let mut po = self.d_plus_tm.as_device_ptr().as_raw(); let mut mo = self.d_minus_tm.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 9] = [
                &mut h as *mut _ as *mut c_void,
                &mut l as *mut _ as *mut c_void,
                &mut c as *mut _ as *mut c_void,
                &mut fv as *mut _ as *mut c_void,
                &mut p as *mut _ as *mut c_void,
                &mut cols as *mut _ as *mut c_void,
                &mut rows as *mut _ as *mut c_void,
                &mut po as *mut _ as *mut c_void,
                &mut mo as *mut _ as *mut c_void,
            ];
            self.cuda.stream.launch(&func, grid, block, 0, &mut args).unwrap();
        }
        self.cuda.synchronize().unwrap();
    } }

    fn prep_di_many_series() -> DiManyState {
        let mut cuda = CudaDi::new(0).expect("cuda di");
        cuda.set_policy(CudaDiPolicy { batch: BatchKernelPolicy::Auto, many_series: ManySeriesKernelPolicy::Auto });
        let cols = 250usize; let rows = 1_000_000usize; let period = 14usize;
        let close_tm = gen_time_major_prices(cols, rows);
        // Synthesize H/L from close (mirrors patterns used in tests)
        let mut high_tm = close_tm.clone();
        let mut low_tm  = close_tm.clone();
        for s in 0..cols { for t in 0..rows { let idx = t*cols+s; let v = close_tm[idx]; if v.is_nan() { continue; } let off = 0.12f32 + ((t as f32)*0.0029).cos().abs()*0.02; high_tm[idx] = v + off; low_tm[idx] = v - off; } }
        // first_valids on host
        let mut first_valids = vec![-1i32; cols];
        for s in 0..cols { for t in 0..rows { let idx = t*cols+s; if !high_tm[idx].is_nan() && !low_tm[idx].is_nan() && !close_tm[idx].is_nan() { first_valids[s] = t as i32; break; } } }
        let d_high = DeviceBuffer::from_slice(&high_tm).expect("dh");
        let d_low = DeviceBuffer::from_slice(&low_tm).expect("dl");
        let d_close = DeviceBuffer::from_slice(&close_tm).expect("dc");
        let d_first = DeviceBuffer::from_slice(&first_valids).expect("df");
        let d_plus_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols*rows) }.expect("po");
        let d_minus_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols*rows) }.expect("mo");
        DiManyState { cuda, d_high, d_low, d_close, d_first, d_plus_tm, d_minus_tm, cols, rows, period }
    }
    fn prep_di_many_series_box() -> Box<dyn CudaBenchState> { Box::new(prep_di_many_series()) }
}
