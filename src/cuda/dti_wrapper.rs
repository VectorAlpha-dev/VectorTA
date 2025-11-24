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

use std::sync::Arc;
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

#[derive(thiserror::Error, Debug)]
pub enum CudaDtiError {
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
pub enum BatchKernelSelected {
    OneD { block_x: u32 },
}
#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}

/// VRAM-backed array handle for DTI with context guard and device id
pub struct DeviceArrayF32Dti {
    pub buf: DeviceBuffer<f32>,
    pub rows: usize,
    pub cols: usize,
    pub ctx: Arc<Context>,
    pub device_id: u32,
}
impl DeviceArrayF32Dti {
    #[inline]
    pub fn device_ptr(&self) -> u64 { self.buf.as_device_ptr().as_raw() as u64 }
    #[inline]
    pub fn len(&self) -> usize { self.rows * self.cols }
}

pub struct CudaDti {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
    policy: CudaDtiPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaDti {
    pub fn new(device_id: usize) -> Result<Self, CudaDtiError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/dti_kernel.ptx"));
        let jit = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, jit)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))?;

        // Favor L1 cache for small working sets (x/ax streams)
        let _ = cust::context::CurrentContext::set_cache_config(CacheConfig::PreferL1);

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        Ok(Self {
            module,
            stream,
            context,
            device_id: device_id as u32,
            policy: CudaDtiPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    #[inline]
    fn device_mem_info() -> Option<(usize, usize)> { mem_get_info().ok() }
    #[inline]
    fn mem_check_enabled() -> bool {
        match std::env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
            Err(_) => true,
        }
    }
    #[inline]
    fn will_fit(bytes: usize, headroom: usize) -> bool {
        if !Self::mem_check_enabled() { return true; }
        if let Some((free, _)) = Self::device_mem_info() {
            bytes.saturating_add(headroom) <= free
        } else { true }
    }

    pub fn set_policy(&mut self, policy: CudaDtiPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaDtiPolicy {
        &self.policy
    }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch
    }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many
    }
    pub fn synchronize(&self) -> Result<(), CudaDtiError> { self.stream.synchronize().map_err(Into::into) }

    #[inline]
    fn validate_launch(&self, gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32) -> Result<(), CudaDtiError> {
        let dev = Device::get_device(self.device_id)?;
        let max_threads = dev
            .get_attribute(cust::device::DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_bx = dev
            .get_attribute(cust::device::DeviceAttribute::MaxBlockDimX)
            .unwrap_or(1024) as u32;
        let max_by = dev
            .get_attribute(cust::device::DeviceAttribute::MaxBlockDimY)
            .unwrap_or(1024) as u32;
        let max_bz = dev
            .get_attribute(cust::device::DeviceAttribute::MaxBlockDimZ)
            .unwrap_or(64) as u32;
        let max_gx = dev
            .get_attribute(cust::device::DeviceAttribute::MaxGridDimX)
            .unwrap_or(2_147_483_647) as u32;
        let max_gy = dev
            .get_attribute(cust::device::DeviceAttribute::MaxGridDimY)
            .unwrap_or(65_535) as u32;
        let max_gz = dev
            .get_attribute(cust::device::DeviceAttribute::MaxGridDimZ)
            .unwrap_or(65_535) as u32;

        let threads = bx.saturating_mul(by).saturating_mul(bz);
        if threads > max_threads || bx > max_bx || by > max_by || bz > max_bz || gx > max_gx || gy > max_gy || gz > max_gz {
            return Err(CudaDtiError::LaunchConfigTooLarge { gx, gy, gz, bx, by, bz });
        }
        Ok(())
    }

    #[inline]
    fn maybe_log_batch_debug(&self) {
        static ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged {
            return;
        }
        if self.debug_batch_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                let per_scen =
                    std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                let per_scen =
                    std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scen || !ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] DTI batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaDti)).debug_batch_logged = true;
                }
                unsafe {
                    (*(self as *const _ as *mut CudaDti)).debug_batch_logged = true;
                }
            }
        }
    }
    #[inline]
    fn maybe_log_many_debug(&self) {
        static ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_many_logged {
            return;
        }
        if self.debug_many_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                let per_scen =
                    std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                let per_scen =
                    std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scen || !ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] DTI many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaDti)).debug_many_logged = true;
                }
                unsafe {
                    (*(self as *const _ as *mut CudaDti)).debug_many_logged = true;
                }
            }
        }
    }

    #[inline]
    // removed duplicate helpers (consolidated below)

    // -------------------- Batch: one series × many params --------------------
    fn expand_grid(range: &DtiBatchRange) -> Vec<DtiParams> {
        fn axis_usize(t: (usize, usize, usize)) -> Vec<usize> {
            let (start, end, step) = t;
            if step == 0 || start == end { return vec![start]; }
            let mut v = Vec::new();
            if start < end {
                let mut cur = start;
                while cur <= end {
                    v.push(cur);
                    match cur.checked_add(step) { Some(n) => { if n > end { break; } cur = n; }, None => break }
                }
            } else {
                let mut cur = start;
                loop {
                    if cur < end { break; }
                    v.push(cur);
                    if cur == end { break; }
                    match cur.checked_sub(step) { Some(n) => { if n < end { break; } cur = n; }, None => break }
                }
            }
            v
        }
        let rr = axis_usize(range.r);
        let ss = axis_usize(range.s);
        let uu = axis_usize(range.u);
        let mut combos = Vec::with_capacity(rr.len() * ss.len() * uu.len());
        for &r in &rr {
            for &s in &ss {
                for &u in &uu {
                    combos.push(DtiParams {
                        r: Some(r),
                        s: Some(s),
                        u: Some(u),
                    });
                }
            }
        }
        combos
    }

    #[inline]
    fn precompute_x_ax_into_locked(
        high: &[f32],
        low: &[f32],
        start: usize,
        x: &mut [f32],
        ax: &mut [f32],
    ) {
        debug_assert_eq!(high.len(), low.len());
        debug_assert_eq!(high.len(), x.len());
        debug_assert_eq!(x.len(), ax.len());
        x.fill(0.0);
        ax.fill(0.0);
        let len = high.len();
        if start == 0 || start >= len { return; }
        for i in start..len {
            let dh = high[i] - high[i - 1];
            let dl = low[i]  - low[i - 1];
            let x_hmu = if dh > 0.0 { dh } else { 0.0 };
            let x_lmd = if dl < 0.0 { -dl } else { 0.0 };
            let v = x_hmu - x_lmd;
            x[i]  = v;
            ax[i] = v.abs();
        }
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
            .map_err(|_| CudaDtiError::MissingKernelSymbol { name: "dti_batch_f32" })?;
        let _ = func.set_cache_config(CacheConfig::PreferL1);

        let block_x: u32 = match std::env::var("DTI_BLOCK_X").ok().as_deref() {
            Some(s) => s.parse::<u32>().ok().filter(|&v| v > 0).unwrap_or(128),
            None => {
                let (_min_grid, suggested) = func
                    .suggested_launch_configuration(0, BlockSize::xyz(0, 0, 0))?;
                suggested
            }
        };
        let grid_x = ((rows as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        self.validate_launch(grid_x.max(1), 1, 1, block_x, 1, 1)?;
        // Record selection for debug introspection
        unsafe {
            (*(self as *const _ as *mut CudaDti)).last_batch =
                Some(BatchKernelSelected::OneD { block_x });
        }
        unsafe {
            (*(self as *const _ as *mut CudaDti)).last_batch =
                Some(BatchKernelSelected::OneD { block_x });
        }

        unsafe {
            let mut px = d_x.as_device_ptr().as_raw();
            let mut px = d_x.as_device_ptr().as_raw();
            let mut pax = d_ax.as_device_ptr().as_raw();
            let mut pr = d_r.as_device_ptr().as_raw();
            let mut ps = d_s.as_device_ptr().as_raw();
            let mut pu = d_u.as_device_ptr().as_raw();
            let mut pr = d_r.as_device_ptr().as_raw();
            let mut ps = d_s.as_device_ptr().as_raw();
            let mut pu = d_u.as_device_ptr().as_raw();
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
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        self.maybe_log_batch_debug();
        Ok(())
    }

    pub fn dti_batch_dev(
        &self,
        high_f32: &[f32],
        low_f32: &[f32],
        sweep: &DtiBatchRange,
    ) -> Result<(DeviceArrayF32Dti, Vec<DtiParams>), CudaDtiError> {
        if high_f32.is_empty() || low_f32.is_empty() || high_f32.len() != low_f32.len() {
            return Err(CudaDtiError::InvalidInput(
                "empty or mismatched inputs".into(),
            ));
        }
        let len = high_f32.len();
        let first_valid = high_f32
            .iter()
            .zip(low_f32.iter())
            .position(|(h, l)| !h.is_nan() && !l.is_nan())
            .ok_or_else(|| CudaDtiError::InvalidInput("all values NaN".into()))?;

        let combos = Self::expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaDtiError::InvalidInput(
                "no parameter combinations".into(),
            ));
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

        // VRAM estimate: x+ax + params + out (checked to avoid usize overflow)
        let inputs_bytes = len
            .checked_mul(2)
            .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| CudaDtiError::InvalidInput("size overflow in dti_batch_dev".into()))?;
        let params_bytes = rows
            .checked_mul(3)
            .and_then(|n| n.checked_mul(std::mem::size_of::<i32>()))
            .ok_or_else(|| CudaDtiError::InvalidInput("size overflow in dti_batch_dev".into()))?;
        let rows_len = rows
            .checked_mul(len)
            .ok_or_else(|| CudaDtiError::InvalidInput("size overflow in dti_batch_dev".into()))?;
        let out_bytes = rows_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaDtiError::InvalidInput("size overflow in dti_batch_dev".into()))?;
        let bytes = inputs_bytes
            .checked_add(params_bytes)
            .and_then(|n| n.checked_add(out_bytes))
            .ok_or_else(|| CudaDtiError::InvalidInput("size overflow in dti_batch_dev".into()))?;
        let headroom = 64 * 1024 * 1024; // 64MB
        if !Self::will_fit(bytes, headroom) {
            if let Some((free, _)) = Self::device_mem_info() {
                return Err(CudaDtiError::OutOfMemory { required: bytes, free, headroom });
            } else {
                return Err(CudaDtiError::OutOfMemory { required: bytes, free: 0, headroom });
            }
        }

        // Precompute x and |x| directly into pinned memory (page-locked)
        let mut hx  = unsafe { LockedBuffer::<f32>::uninitialized(len) }?;
        let mut hax = unsafe { LockedBuffer::<f32>::uninitialized(len) }?;
        Self::precompute_x_ax_into_locked(high_f32, low_f32, start, hx.as_mut_slice(), hax.as_mut_slice());

        // Prepare params (i32) into pinned memory
        let mut r_vec = Vec::with_capacity(rows);
        let mut s_vec = Vec::with_capacity(rows);
        let mut u_vec = Vec::with_capacity(rows);
        for c in &combos { r_vec.push(c.r.unwrap() as i32); s_vec.push(c.s.unwrap() as i32); u_vec.push(c.u.unwrap() as i32); }
        let hr = LockedBuffer::from_slice(&r_vec)?;
        let hs = LockedBuffer::from_slice(&s_vec)?;
        let hu = LockedBuffer::from_slice(&u_vec)?;

        let mut d_x = unsafe { DeviceBuffer::<f32>::uninitialized_async(len, &self.stream) }?;
        let mut d_ax = unsafe { DeviceBuffer::<f32>::uninitialized_async(len, &self.stream) }?;
        let mut d_r = unsafe { DeviceBuffer::<i32>::uninitialized_async(rows, &self.stream) }?;
        let mut d_s = unsafe { DeviceBuffer::<i32>::uninitialized_async(rows, &self.stream) }?;
        let mut d_u = unsafe { DeviceBuffer::<i32>::uninitialized_async(rows, &self.stream) }?;
        let mut d_out =
            unsafe { DeviceBuffer::<f32>::uninitialized_async(rows_len, &self.stream) }?;

        unsafe {
            d_x.async_copy_from(&hx, &self.stream)?;
            d_ax.async_copy_from(&hax, &self.stream)?;
            d_r.async_copy_from(&hr, &self.stream)?;
            d_s.async_copy_from(&hs, &self.stream)?;
            d_u.async_copy_from(&hu, &self.stream)?;
        }

        self.launch_batch_kernel(&d_x, &d_ax, &d_r, &d_s, &d_u, len, rows, start, &mut d_out)?;
        self.stream.synchronize()?;

        Ok((DeviceArrayF32Dti { buf: d_out, rows, cols: len, ctx: Arc::clone(&self.context), device_id: self.device_id }, combos))
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
            .map_err(|_| CudaDtiError::MissingKernelSymbol { name: "dti_many_series_one_param_f32" })?;
        let _ = func.set_cache_config(CacheConfig::PreferL1);

        let block_x: u32 = match std::env::var("DTI_MANY_BLOCK_X").ok().as_deref() {
            Some(s) => s.parse::<u32>().ok().filter(|&v| v > 0).unwrap_or(128),
            None => {
                let (_min, suggested) = func
                    .suggested_launch_configuration(0, BlockSize::xyz(0, 0, 0))?;
                suggested
            }
        };
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        self.validate_launch(grid_x.max(1), 1, 1, block_x, 1, 1)?;
        unsafe {
            (*(self as *const _ as *mut CudaDti)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }

        unsafe {
            let mut ph = d_high_tm.as_device_ptr().as_raw();
            let mut pl = d_low_tm.as_device_ptr().as_raw();
            let mut pfv = d_first.as_device_ptr().as_raw();
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
            self.stream.launch(&func, grid, block, 0, args)?;
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
    ) -> Result<DeviceArrayF32Dti, CudaDtiError> {
        if cols == 0 || rows == 0 {
            return Err(CudaDtiError::InvalidInput("empty matrix".into()));
        }
        if cols == 0 || rows == 0 {
            return Err(CudaDtiError::InvalidInput("empty matrix".into()));
        }
        let elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaDtiError::InvalidInput("matrix size overflow".into()))?;
        if high_tm_f32.len() != elems || low_tm_f32.len() != elems {
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
                let l = low_tm_f32[t * cols + series];
                if !h.is_nan() && !l.is_nan() {
                    fv = t as i32;
                    break;
                }
                let l = low_tm_f32[t * cols + series];
                if !h.is_nan() && !l.is_nan() {
                    fv = t as i32;
                    break;
                }
            }
            first_valids[series] = fv;
        }

        // VRAM estimate: input + first_valids + out (checked)
        let inputs_bytes = elems
            .checked_mul(2)
            .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| CudaDtiError::InvalidInput("size overflow in dti_many_series_one_param_time_major_dev".into()))?;
        let first_bytes = cols
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaDtiError::InvalidInput("size overflow in dti_many_series_one_param_time_major_dev".into()))?;
        let out_bytes = elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaDtiError::InvalidInput("size overflow in dti_many_series_one_param_time_major_dev".into()))?;
        let bytes = inputs_bytes
            .checked_add(first_bytes)
            .and_then(|n| n.checked_add(out_bytes))
            .ok_or_else(|| CudaDtiError::InvalidInput("size overflow in dti_many_series_one_param_time_major_dev".into()))?;
        if !Self::will_fit(bytes, 64 * 1024 * 1024) {
            if let Some((free, _)) = Self::device_mem_info() {
                return Err(CudaDtiError::OutOfMemory { required: bytes, free, headroom: 64 * 1024 * 1024 });
            } else {
                return Err(CudaDtiError::OutOfMemory { required: bytes, free: 0, headroom: 64 * 1024 * 1024 });
            }
        }

        // Pinned host staging
        let h_high  = LockedBuffer::from_slice(high_tm_f32)?;
        let h_low   = LockedBuffer::from_slice(low_tm_f32 )?;
        let h_first = LockedBuffer::from_slice(&first_valids)?;

        // Async device allocations + async copies
        let mut d_high  = unsafe { DeviceBuffer::<f32>::uninitialized_async(elems, &self.stream) }?;
        let mut d_low   = unsafe { DeviceBuffer::<f32>::uninitialized_async(elems, &self.stream) }?;
        let mut d_first = unsafe { DeviceBuffer::<i32>::uninitialized_async(cols, &self.stream) }?;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized_async(elems, &self.stream) }?;

        unsafe {
            d_high .async_copy_from(&h_high,  &self.stream)?;
            d_low  .async_copy_from(&h_low,   &self.stream)?;
            d_first.async_copy_from(&h_first, &self.stream)?;
        }

        self.launch_many_series_kernel(&d_high, &d_low, &d_first, cols, rows, r, s, u, &mut d_out)?;
        self.stream.synchronize()?;

        Ok(DeviceArrayF32Dti { buf: d_out, rows, cols, ctx: Arc::clone(&self.context), device_id: self.device_id })
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
        (elems * 2 * std::mem::size_of::<f32>())
            + (MANY_SERIES_COLS * std::mem::size_of::<i32>())
            + (elems * std::mem::size_of::<f32>())
            + 64 * 1024 * 1024
    }

    struct BatchState {
        cuda: CudaDti,
        high: Vec<f32>,
        low: Vec<f32>,
        sweep: DtiBatchRange,
    }
    impl CudaBenchState for BatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .dti_batch_dev(&self.high, &self.low, &self.sweep)
                .expect("dti_batch_dev");
            let _ = self
                .cuda
                .dti_batch_dev(&self.high, &self.low, &self.sweep)
                .expect("dti_batch_dev");
        }
    }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaDti::new(0).expect("cuda");
        let base = crate::cuda::bench::helpers::gen_series(ONE_SERIES_LEN);
        let mut high = vec![f32::NAN; ONE_SERIES_LEN];
        let mut low = vec![f32::NAN; ONE_SERIES_LEN];
        for i in 1..ONE_SERIES_LEN {
            let x = base[i];
            let prev = base[i - 1];
            high[i] = x.max(prev) + 0.7;
            low[i] = x.min(prev) - 0.7;
        }
        // modest sweep (cartesian product of 6×5×6 ~= 180)
        let sweep = DtiBatchRange {
            r: (8, 18, 2),
            s: (6, 14, 2),
            u: (3, 13, 2),
        };
        Box::new(BatchState { cuda, high, low, sweep })
    }

    struct ManyState {
        cuda: CudaDti,
        high_tm: Vec<f32>,
        low_tm: Vec<f32>,
        cols: usize,
        rows: usize,
        params: DtiParams,
    }
    impl CudaBenchState for ManyState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .dti_many_series_one_param_time_major_dev(
                    &self.high_tm,
                    &self.low_tm,
                    self.cols,
                    self.rows,
                    &self.params,
                )
                .expect("dti_many_series_one_param_time_major_dev");
            let _ = self
                .cuda
                .dti_many_series_one_param_time_major_dev(
                    &self.high_tm,
                    &self.low_tm,
                    self.cols,
                    self.rows,
                    &self.params,
                )
                .expect("dti_many_series_one_param_time_major_dev");
        }
    }
    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaDti::new(0).expect("cuda");
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_LEN;
        let mid = gen_time_major_prices(cols, rows);
        let mut high_tm = vec![f32::NAN; cols * rows];
        let mut low_tm = vec![f32::NAN; cols * rows];
        for t in 0..rows {
            for s in 0..cols {
                let m = mid[t * cols + s];
                if m.is_nan() {
                    continue;
                }
                // synthesize high/low around mid
                high_tm[t * cols + s] = m + 0.6;
                low_tm[t * cols + s] = m - 0.6;
            }
        }
        let params = DtiParams {
            r: Some(14),
            s: Some(10),
            u: Some(5),
        };
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
