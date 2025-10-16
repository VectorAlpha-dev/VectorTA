//! CUDA wrapper for Dynamic Trend Index (DTI).
//!
//! Parity goals with ALMA wrapper:
//! - PTX load via `include_str!(concat!(env!("OUT_DIR"), "/dti_kernel.ptx"))`
//! - Stream NON_BLOCKING
//! - Simple explicit policy types and introspection
//! - VRAM checks + headroom and basic chunking guards
//! - Device entry points returning VRAM-resident `DeviceArrayF32`
//!
//! Category: Recurrence/IIR (triple EMA chains). For the batch path (one series × many params),
//! we precompute the base series x and |x| once on host and reuse across rows, mirroring the
//! scalar batch optimization in `indicators::dti`.

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::alma_wrapper::DeviceArrayF32;
use crate::indicators::dti::{DtiBatchRange, DtiParams};
use cust::context::{CacheConfig, Context};
use cust::device::Device;
use cust::function::{BlockSize, Function, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug)]
pub enum CudaDtiError {
    Cuda(String),
    InvalidInput(String),
}
impl fmt::Display for CudaDtiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaDtiError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaDtiError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}
impl std::error::Error for CudaDtiError {}

// Minimal policies aligned with other osc. wrappers
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
pub struct CudaDtiPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaDtiPolicy {
    fn default() -> Self {
        Self { batch: BatchKernelPolicy::Auto, many_series: ManySeriesKernelPolicy::Auto }
    }
}
#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected { OneD { block_x: u32 } }
#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected { OneD { block_x: u32 } }

pub struct CudaDti {
    module: Module,
    stream: Stream,
    _context: Context,
    policy: CudaDtiPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaDti {
    pub fn new(device_id: usize) -> Result<Self, CudaDtiError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaDtiError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaDtiError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaDtiError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/dti_kernel.ptx"));
        let jit = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, jit)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))
            .map_err(|e| CudaDtiError::Cuda(e.to_string()))?;

        // Favor L1 cache for small working sets (x/ax streams)
        let _ = cust::context::CurrentContext::set_cache_config(CacheConfig::PreferL1);

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaDtiError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            policy: CudaDtiPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn set_policy(&mut self, policy: CudaDtiPolicy) { self.policy = policy; }
    pub fn policy(&self) -> &CudaDtiPolicy { &self.policy }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> { self.last_batch }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> { self.last_many }
    pub fn synchronize(&self) -> Result<(), CudaDtiError> {
        self.stream.synchronize().map_err(|e| CudaDtiError::Cuda(e.to_string()))
    }

    #[inline]
    fn maybe_log_batch_debug(&self) {
        static ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged { return; }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                let per_scen = std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scen || !ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] DTI batch selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut CudaDti)).debug_batch_logged = true; }
            }
        }
    }
    #[inline]
    fn maybe_log_many_debug(&self) {
        static ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_many_logged { return; }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                let per_scen = std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scen || !ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] DTI many-series selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut CudaDti)).debug_many_logged = true; }
            }
        }
    }

    #[inline]
    fn device_mem_info() -> Option<(usize, usize)> { mem_get_info().ok() }
    #[inline]
    fn mem_check_enabled() -> bool {
        match std::env::var("CUDA_MEM_CHECK") { Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"), Err(_) => true }
    }
    #[inline]
    fn will_fit(bytes: usize, headroom: usize) -> bool {
        if !Self::mem_check_enabled() { return true; }
        if let Some((free, _)) = Self::device_mem_info() { bytes.saturating_add(headroom) <= free } else { true }
    }

    // -------------------- Batch: one series × many params --------------------
    fn expand_grid(range: &DtiBatchRange) -> Vec<DtiParams> {
        fn axis_usize(t: (usize, usize, usize)) -> Vec<usize> {
            let (start, end, step) = t;
            if step == 0 || start == end { return vec![start]; }
            (start..=end).step_by(step).collect()
        }
        let rr = axis_usize(range.r);
        let ss = axis_usize(range.s);
        let uu = axis_usize(range.u);
        let mut combos = Vec::with_capacity(rr.len() * ss.len() * uu.len());
        for &r in &rr { for &s in &ss { for &u in &uu {
            combos.push(DtiParams { r: Some(r), s: Some(s), u: Some(u) });
        }}}
        combos
    }

    fn precompute_x_ax(high: &[f32], low: &[f32], start: usize) -> (Vec<f32>, Vec<f32>) {
        let len = high.len();
        let mut x = vec![0f32; len];
        let mut ax = vec![0f32; len];
        if start == 0 || start >= len { return (x, ax); }
        let mut i = start;
        while i < len {
            let dh = high[i] - high[i - 1];
            let dl = low[i] - low[i - 1];
            let x_hmu = if dh > 0.0 { dh } else { 0.0 };
            let x_lmd = if dl < 0.0 { -dl } else { 0.0 };
            let v = x_hmu - x_lmd;
            x[i] = v;
            ax[i] = v.abs();
            i += 1;
        }
        (x, ax)
    }

    fn launch_batch_kernel(
        &self,
        d_x: &DeviceBuffer<f32>,
        d_ax: &DeviceBuffer<f32>,
        d_r: &DeviceBuffer<i32>,
        d_s: &DeviceBuffer<i32>,
        d_u: &DeviceBuffer<i32>,
        len: usize,
        rows: usize,
        start: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaDtiError> {
        let mut func: Function = self
            .module
            .get_function("dti_batch_f32")
            .map_err(|e| CudaDtiError::Cuda(e.to_string()))?;
        let _ = func.set_cache_config(CacheConfig::PreferL1);

        let block_x: u32 = match std::env::var("DTI_BLOCK_X").ok().as_deref() {
            Some(s) => s.parse::<u32>().ok().filter(|&v| v > 0).unwrap_or(128),
            None => {
                let (_min_grid, suggested) = func
                    .suggested_launch_configuration(0, BlockSize::xyz(0, 0, 0))
                    .map_err(|e| CudaDtiError::Cuda(e.to_string()))?;
                suggested
            }
        };
        let grid_x = ((rows as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        // Record selection for debug introspection
        unsafe { (*(self as *const _ as *mut CudaDti)).last_batch = Some(BatchKernelSelected::OneD { block_x }); }

        unsafe {
            let mut px  = d_x.as_device_ptr().as_raw();
            let mut pax = d_ax.as_device_ptr().as_raw();
            let mut pr  = d_r.as_device_ptr().as_raw();
            let mut ps  = d_s.as_device_ptr().as_raw();
            let mut pu  = d_u.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut rows_i = rows as i32;
            let mut start_i = start as i32;
            let mut pout = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut px as *mut _ as *mut c_void,
                &mut pax as *mut _ as *mut c_void,
                &mut pr as *mut _ as *mut c_void,
                &mut ps as *mut _ as *mut c_void,
                &mut pu as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut start_i as *mut _ as *mut c_void,
                &mut pout as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaDtiError::Cuda(e.to_string()))?;
        }
        self.maybe_log_batch_debug();
        Ok(())
    }

    pub fn dti_batch_dev(
        &self,
        high_f32: &[f32],
        low_f32: &[f32],
        sweep: &DtiBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<DtiParams>), CudaDtiError> {
        if high_f32.is_empty() || low_f32.is_empty() || high_f32.len() != low_f32.len() {
            return Err(CudaDtiError::InvalidInput("empty or mismatched inputs".into()));
        }
        let len = high_f32.len();
        let first_valid = high_f32
            .iter()
            .zip(low_f32.iter())
            .position(|(h, l)| !h.is_nan() && !l.is_nan())
            .ok_or_else(|| CudaDtiError::InvalidInput("all values NaN".into()))?;

        let combos = Self::expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaDtiError::InvalidInput("no parameter combinations".into()));
        }
        let max_p = combos
            .iter()
            .map(|c| c.r.unwrap().max(c.s.unwrap()).max(c.u.unwrap()))
            .max()
            .unwrap();
        if len - first_valid < max_p {
            return Err(CudaDtiError::InvalidInput(format!(
                "not enough valid data (needed {}, valid {})",
                max_p,
                len - first_valid
            )));
        }
        let rows = combos.len();
        let start = first_valid + 1;

        // VRAM estimate: x+ax + params + out
        let bytes = (len * 2 * std::mem::size_of::<f32>())
            + (rows * 3 * std::mem::size_of::<i32>())
            + (rows * len * std::mem::size_of::<f32>());
        let headroom = 64 * 1024 * 1024; // 64MB
        if !Self::will_fit(bytes, headroom) {
            return Err(CudaDtiError::InvalidInput("insufficient VRAM for DTI batch".into()));
        }

        // Precompute x and |x| once on host (shares across rows)
        let (x_host, ax_host) = Self::precompute_x_ax(high_f32, low_f32, start);

        // Prepare params (i32)
        let mut r_vec = Vec::with_capacity(rows);
        let mut s_vec = Vec::with_capacity(rows);
        let mut u_vec = Vec::with_capacity(rows);
        for c in &combos { r_vec.push(c.r.unwrap() as i32); s_vec.push(c.s.unwrap() as i32); u_vec.push(c.u.unwrap() as i32); }

        // Async copies using pinned host buffers
        let hx = LockedBuffer::from_slice(&x_host).map_err(|e| CudaDtiError::Cuda(e.to_string()))?;
        let hax = LockedBuffer::from_slice(&ax_host).map_err(|e| CudaDtiError::Cuda(e.to_string()))?;
        let hr = LockedBuffer::from_slice(&r_vec).map_err(|e| CudaDtiError::Cuda(e.to_string()))?;
        let hs = LockedBuffer::from_slice(&s_vec).map_err(|e| CudaDtiError::Cuda(e.to_string()))?;
        let hu = LockedBuffer::from_slice(&u_vec).map_err(|e| CudaDtiError::Cuda(e.to_string()))?;

        let mut d_x  = unsafe { DeviceBuffer::<f32>::uninitialized_async(len, &self.stream) }
            .map_err(|e| CudaDtiError::Cuda(e.to_string()))?;
        let mut d_ax = unsafe { DeviceBuffer::<f32>::uninitialized_async(len, &self.stream) }
            .map_err(|e| CudaDtiError::Cuda(e.to_string()))?;
        let mut d_r  = unsafe { DeviceBuffer::<i32>::uninitialized_async(rows, &self.stream) }
            .map_err(|e| CudaDtiError::Cuda(e.to_string()))?;
        let mut d_s  = unsafe { DeviceBuffer::<i32>::uninitialized_async(rows, &self.stream) }
            .map_err(|e| CudaDtiError::Cuda(e.to_string()))?;
        let mut d_u  = unsafe { DeviceBuffer::<i32>::uninitialized_async(rows, &self.stream) }
            .map_err(|e| CudaDtiError::Cuda(e.to_string()))?;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized_async(rows * len, &self.stream) }
            .map_err(|e| CudaDtiError::Cuda(e.to_string()))?;

        unsafe {
            d_x .async_copy_from(&hx,  &self.stream).map_err(|e| CudaDtiError::Cuda(e.to_string()))?;
            d_ax.async_copy_from(&hax, &self.stream).map_err(|e| CudaDtiError::Cuda(e.to_string()))?;
            d_r .async_copy_from(&hr,  &self.stream).map_err(|e| CudaDtiError::Cuda(e.to_string()))?;
            d_s .async_copy_from(&hs,  &self.stream).map_err(|e| CudaDtiError::Cuda(e.to_string()))?;
            d_u .async_copy_from(&hu,  &self.stream).map_err(|e| CudaDtiError::Cuda(e.to_string()))?;
        }

        self.launch_batch_kernel(&d_x, &d_ax, &d_r, &d_s, &d_u, len, rows, start, &mut d_out)?;
        self.stream.synchronize().map_err(|e| CudaDtiError::Cuda(e.to_string()))?;

        Ok((DeviceArrayF32 { buf: d_out, rows, cols: len }, combos))
    }

    // -------------------- Many series × one param (time-major) --------------------
    fn launch_many_series_kernel(
        &self,
        d_high_tm: &DeviceBuffer<f32>,
        d_low_tm: &DeviceBuffer<f32>,
        d_first: &DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        r: usize,
        s: usize,
        u: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaDtiError> {
        let mut func = self
            .module
            .get_function("dti_many_series_one_param_f32")
            .map_err(|e| CudaDtiError::Cuda(e.to_string()))?;
        let _ = func.set_cache_config(CacheConfig::PreferL1);

        let block_x: u32 = match std::env::var("DTI_MANY_BLOCK_X").ok().as_deref() {
            Some(s) => s.parse::<u32>().ok().filter(|&v| v > 0).unwrap_or(128),
            None => {
                let (_min, suggested) = func
                    .suggested_launch_configuration(0, BlockSize::xyz(0, 0, 0))
                    .map_err(|e| CudaDtiError::Cuda(e.to_string()))?;
                suggested
            }
        };
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe { (*(self as *const _ as *mut CudaDti)).last_many = Some(ManySeriesKernelSelected::OneD { block_x }); }

        unsafe {
            let mut ph  = d_high_tm.as_device_ptr().as_raw();
            let mut pl  = d_low_tm .as_device_ptr().as_raw();
            let mut pfv = d_first  .as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut r_i = r as i32;
            let mut s_i = s as i32;
            let mut u_i = u as i32;
            let mut pout = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut ph as *mut _ as *mut c_void,
                &mut pl as *mut _ as *mut c_void,
                &mut pfv as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut r_i as *mut _ as *mut c_void,
                &mut s_i as *mut _ as *mut c_void,
                &mut u_i as *mut _ as *mut c_void,
                &mut pout as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaDtiError::Cuda(e.to_string()))?;
        }
        self.maybe_log_many_debug();
        Ok(())
    }

    pub fn dti_many_series_one_param_time_major_dev(
        &self,
        high_tm_f32: &[f32],
        low_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &DtiParams,
    ) -> Result<DeviceArrayF32, CudaDtiError> {
        if cols == 0 || rows == 0 { return Err(CudaDtiError::InvalidInput("empty matrix".into())); }
        if high_tm_f32.len() != cols * rows || low_tm_f32.len() != cols * rows {
            return Err(CudaDtiError::InvalidInput("mismatched matrix sizes".into()));
        }
        let r = params.r.unwrap_or(14);
        let s = params.s.unwrap_or(10);
        let u = params.u.unwrap_or(5);

        // Compute per-series first_valid
        let mut first_valids = vec![rows as i32; cols];
        for series in 0..cols {
            let mut fv = rows as i32;
            for t in 0..rows {
                let h = high_tm_f32[t * cols + series];
                let l = low_tm_f32 [t * cols + series];
                if !h.is_nan() && !l.is_nan() { fv = t as i32; break; }
            }
            first_valids[series] = fv;
        }

        // VRAM estimate: input + first_valids + out
        let elems = cols * rows;
        let bytes = (elems * 2 * std::mem::size_of::<f32>())
            + (cols * std::mem::size_of::<i32>())
            + (elems * std::mem::size_of::<f32>());
        if !Self::will_fit(bytes, 64 * 1024 * 1024) {
            return Err(CudaDtiError::InvalidInput("insufficient VRAM for DTI many-series".into()));
        }

        let d_high = DeviceBuffer::from_slice(high_tm_f32).map_err(|e| CudaDtiError::Cuda(e.to_string()))?;
        let d_low  = DeviceBuffer::from_slice(low_tm_f32 ).map_err(|e| CudaDtiError::Cuda(e.to_string()))?;
        let d_first = DeviceBuffer::from_slice(&first_valids).map_err(|e| CudaDtiError::Cuda(e.to_string()))?;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }.map_err(|e| CudaDtiError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(&d_high, &d_low, &d_first, cols, rows, r, s, u, &mut d_out)?;
        self.stream.synchronize().map_err(|e| CudaDtiError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 { buf: d_out, rows, cols })
    }
}

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_time_major_prices;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 180; // r×s×u combos typical order
    const MANY_SERIES_COLS: usize = 192;
    const MANY_SERIES_LEN: usize = 1_000_000;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = ONE_SERIES_LEN * 2 * std::mem::size_of::<f32>(); // high+low (for precompute)
        let pre_bytes = ONE_SERIES_LEN * 2 * std::mem::size_of::<f32>(); // x + ax
        let params = PARAM_SWEEP * 3 * std::mem::size_of::<i32>();
        let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + pre_bytes + params + out_bytes + 64 * 1024 * 1024
    }
    fn bytes_many_series_one_param() -> usize {
        let elems = MANY_SERIES_COLS * MANY_SERIES_LEN;
        (elems * 2 * std::mem::size_of::<f32>()) + (MANY_SERIES_COLS * std::mem::size_of::<i32>())
            + (elems * std::mem::size_of::<f32>()) + 64 * 1024 * 1024
    }

    struct BatchState { cuda: CudaDti, high: Vec<f32>, low: Vec<f32>, sweep: DtiBatchRange }
    impl CudaBenchState for BatchState {
        fn launch(&mut self) {
            let _ = self.cuda.dti_batch_dev(&self.high, &self.low, &self.sweep).expect("dti_batch_dev");
        }
    }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaDti::new(0).expect("cuda");
        let base = crate::cuda::bench::helpers::gen_series(ONE_SERIES_LEN);
        let mut high = vec![f32::NAN; ONE_SERIES_LEN];
        let mut low  = vec![f32::NAN; ONE_SERIES_LEN];
        for i in 1..ONE_SERIES_LEN {
            let x = base[i];
            let prev = base[i - 1];
            high[i] = x.max(prev) + 0.7;
            low[i]  = x.min(prev) - 0.7;
        }
        // modest sweep (cartesian product of 6×5×6 ~= 180)
        let sweep = DtiBatchRange { r: (8, 18, 2), s: (6, 14, 2), u: (3, 13, 2) };
        Box::new(BatchState { cuda, high, low, sweep })
    }

    struct ManyState { cuda: CudaDti, high_tm: Vec<f32>, low_tm: Vec<f32>, cols: usize, rows: usize, params: DtiParams }
    impl CudaBenchState for ManyState {
        fn launch(&mut self) {
            let _ = self.cuda.dti_many_series_one_param_time_major_dev(
                &self.high_tm, &self.low_tm, self.cols, self.rows, &self.params,
            ).expect("dti_many_series_one_param_time_major_dev");
        }
    }
    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaDti::new(0).expect("cuda");
        let cols = MANY_SERIES_COLS; let rows = MANY_SERIES_LEN;
        let mid = gen_time_major_prices(cols, rows);
        let mut high_tm = vec![f32::NAN; cols * rows];
        let mut low_tm  = vec![f32::NAN; cols * rows];
        for t in 0..rows {
            for s in 0..cols {
                let m = mid[t * cols + s];
                if m.is_nan() { continue; }
                // synthesize high/low around mid
                high_tm[t * cols + s] = m + 0.6;
                low_tm [t * cols + s] = m - 0.6;
            }
        }
        let params = DtiParams { r: Some(14), s: Some(10), u: Some(5) };
        Box::new(ManyState { cuda, high_tm, low_tm, cols, rows, params })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "dti",
                "one_series_many_params",
                "dti_cuda_batch_dev",
                "1m_x_180",
                prep_one_series_many_params,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "dti",
                "many_series_one_param",
                "dti_cuda_many_series_one_param",
                "192x1m",
                prep_many_series_one_param,
            )
            .with_sample_size(5)
            .with_mem_required(bytes_many_series_one_param()),
        ]
    }
}
