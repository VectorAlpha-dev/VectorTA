//! CUDA scaffolding for Damiani Volatmeter (ATR/stddev dual-volatility).
//!
//! Parity goals with ALMA wrapper:
//! - PTX load via DetermineTargetFromContext with OptLevel O2 and simple fallbacks.
//! - NON_BLOCKING stream.
//! - Policy knobs for batch and many-series paths (block size).
//! - VRAM checks and per-series/row first_valid validation.
//! - Outputs stacked as 2 rows per combination/series: row0=vol, row1=anti.

#![cfg(feature = "cuda")]

use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{AsyncCopyDestination, DeviceBuffer, LockedBuffer, DeviceCopy};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;

use crate::cuda::moving_averages::alma_wrapper::DeviceArrayF32;
use crate::indicators::damiani_volatmeter::{DamianiVolatmeterBatchRange, DamianiVolatmeterParams};

// CUDA vector equivalent for float2 (hi, lo) compensated prefix sums
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Float2 { pub x: f32, pub y: f32 }
unsafe impl DeviceCopy for Float2 {}

#[derive(Debug)]
pub enum CudaDamianiError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaDamianiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaDamianiError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaDamianiError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}
impl std::error::Error for CudaDamianiError {}

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
pub enum BatchKernelSelected {
    Plain { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}

pub struct CudaDamianiVolatmeter {
    module: Module,
    stream: Stream,
    _ctx: Context,
    policy_batch: BatchKernelPolicy,
    policy_many: ManySeriesKernelPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaDamianiVolatmeter {
    pub fn new(device_id: usize) -> Result<Self, CudaDamianiError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaDamianiError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaDamianiError::Cuda(e.to_string()))?;
        let ctx = Context::new(device).map_err(|e| CudaDamianiError::Cuda(e.to_string()))?;
        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/damiani_volatmeter_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => {
                if let Ok(m) = Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext])
                {
                    m
                } else {
                    Module::from_ptx(ptx, &[]).map_err(|e| CudaDamianiError::Cuda(e.to_string()))?
                }
            }
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaDamianiError::Cuda(e.to_string()))?;
        Ok(Self {
            module,
            stream,
            _ctx: ctx,
            policy_batch: BatchKernelPolicy::Auto,
            policy_many: ManySeriesKernelPolicy::Auto,
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        match std::env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
            Err(_) => true,
        }
    }
    #[inline]
    fn will_fit(required_bytes: usize, headroom: usize) -> bool {
        if !Self::mem_check_enabled() {
            return true;
        }
        if let Ok((free, _)) = cust::memory::mem_get_info() {
            required_bytes.saturating_add(headroom) <= free
        } else {
            true
        }
    }

    #[inline]
    fn maybe_log_batch_debug(&self) {
        use std::sync::atomic::{AtomicBool, Ordering};
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                let per = std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] Damiani batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaDamianiVolatmeter)).debug_batch_logged = true;
                }
            }
        }
    }
    #[inline]
    fn maybe_log_many_debug(&self) {
        use std::sync::atomic::{AtomicBool, Ordering};
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_many_logged {
            return;
        }
        if self.debug_many_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                let per = std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] Damiani many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaDamianiVolatmeter)).debug_many_logged = true;
                }
                unsafe {
                    (*(self as *const _ as *mut CudaDamianiVolatmeter)).debug_many_logged = true;
                }
            }
        }
    }

    pub fn set_batch_policy(&mut self, p: BatchKernelPolicy) { self.policy_batch = p; }
    pub fn set_many_series_policy(&mut self, p: ManySeriesKernelPolicy) { self.policy_many = p; }
    pub fn batch_policy(&self) -> BatchKernelPolicy { self.policy_batch }
    pub fn many_series_policy(&self) -> ManySeriesKernelPolicy { self.policy_many }

    // ---------- Helpers (host precomputes) ----------
    fn first_valid_close(data: &[f32]) -> Result<usize, CudaDamianiError> {
        if data.is_empty() { return Err(CudaDamianiError::InvalidInput("empty series".into())); }
        if data.is_empty() {
            return Err(CudaDamianiError::InvalidInput("empty series".into()));
        }
        (0..data.len())
            .find(|&i| data[i].is_finite())
            .ok_or_else(|| CudaDamianiError::InvalidInput("all values are NaN".into()))
    }

    fn expand_grid(range: &DamianiVolatmeterBatchRange) -> Vec<DamianiVolatmeterParams> {
        fn axis_usize((s, e, step): (usize, usize, usize)) -> Vec<usize> {
            if step == 0 || s == e {
                return vec![s];
            }
            if step == 0 || s == e {
                return vec![s];
            }
            (s..=e).step_by(step).collect()
        }
        fn axis_f64((s, e, step): (f64, f64, f64)) -> Vec<f64> {
            if step == 0.0 || (s == e) {
                return vec![s];
            }
            if step == 0.0 || (s == e) {
                return vec![s];
            }
            let n = (((e - s) / step).floor() as usize).saturating_add(1);
            (0..n).map(|k| s + (k as f64) * step).collect()
        }
        let a = axis_usize(range.vis_atr);
        let b = axis_usize(range.vis_std);
        let c = axis_usize(range.sed_atr);
        let d = axis_usize(range.sed_std);
        let e = axis_f64(range.threshold);
        let mut out = Vec::with_capacity(a.len() * b.len() * c.len() * d.len() * e.len());
        for &va in &a {
            for &vb in &b {
                for &vc in &c {
                    for &vd in &d {
                        for &ve in &e {
                            out.push(DamianiVolatmeterParams {
                                vis_atr: Some(va),
                                vis_std: Some(vb),
                                sed_atr: Some(vc),
                                sed_std: Some(vd),
                                threshold: Some(ve),
                            });
                        }
                    }
                }
            }
        }
        for &va in &a {
            for &vb in &b {
                for &vc in &c {
                    for &vd in &d {
                        for &ve in &e {
                            out.push(DamianiVolatmeterParams {
                                vis_atr: Some(va),
                                vis_std: Some(vb),
                                sed_atr: Some(vc),
                                sed_std: Some(vd),
                                threshold: Some(ve),
                            });
                        }
                    }
                }
            }
        }
        out
    }

    fn compute_tr_close_only(prices: &[f32], first_valid: usize) -> Vec<f32> {
        let mut tr = vec![0f32; prices.len()];
        let mut prev_close = f32::NAN;
        let mut have_prev = false;
        let mut prev_close = f32::NAN;
        let mut have_prev = false;
        for i in first_valid..prices.len() {
            let c = prices[i];
            let t = if have_prev && c.is_finite() {
                (c - prev_close).abs()
            } else {
                0.0
            };
            let t = if have_prev && c.is_finite() {
                (c - prev_close).abs()
            } else {
                0.0
            };
            tr[i] = t;
            if c.is_finite() {
                prev_close = c;
                have_prev = true;
            }
            if c.is_finite() {
                prev_close = c;
                have_prev = true;
            }
        }
        tr
    }

    fn compute_prefix_sums(prices: &[f32], first_valid: usize) -> (Vec<f64>, Vec<f64>) {
        let mut s = vec![0f64; prices.len()];
        let mut ss = vec![0f64; prices.len()];
        let mut acc = 0f64;
        let mut acc2 = 0f64;
        for i in 0..prices.len() {
            if i >= first_valid {
                let v = if prices[i].is_nan() {
                    0.0
                } else {
                    prices[i] as f64
                };
                acc += v;
                acc2 += v * v;
            }
            s[i] = acc;
            ss[i] = acc2;
        }
        (s, ss)
    }

    fn compute_prefix_sums_time_major(
        close_tm: &[f32],
        cols: usize,
        rows: usize,
        first_valids: &[i32],
    ) -> (Vec<f64>, Vec<f64>) {
        let mut s = vec![0f64; close_tm.len()];
        let mut ss = vec![0f64; close_tm.len()];
        for series in 0..cols {
            let fv = first_valids[series].max(0) as usize;
            let mut acc = 0f64;
            let mut acc2 = 0f64;
            for t in 0..rows {
                let idx = t * cols + series;
                if t >= fv {
                    let v = if close_tm[idx].is_nan() {
                        0.0
                    } else {
                        close_tm[idx] as f64
                    };
                    acc += v;
                    acc2 += v * v;
                }
                s[idx] = acc;
                ss[idx] = acc2;
            }
        }
        (s, ss)
    }

    #[inline]
    fn pack_double_prefix_to_float2(src: &[f64]) -> Vec<Float2> {
        let mut out = Vec::with_capacity(src.len());
        for &d in src {
            let hi = d as f32;
            let lo = (d - hi as f64) as f32;
            out.push(Float2 { x: hi, y: lo });
        }
        out
    }

    #[inline]
    fn pack_double_prefix_to_float2_pinned(src: &[f64]) -> Option<LockedBuffer<Float2>> {
        let mut buf = unsafe { LockedBuffer::<Float2>::uninitialized(src.len()) }.ok()?;
        for (i, &d) in src.iter().enumerate() {
            let hi = d as f32;
            let lo = (d - hi as f64) as f32;
            buf[i] = Float2 { x: hi, y: lo };
        }
        Some(buf)
    }

    #[inline]
    fn compute_tr_close_only_pinned(prices: &[f32], first_valid: usize) -> Option<LockedBuffer<f32>> {
        let mut buf: LockedBuffer<f32> = unsafe { LockedBuffer::uninitialized(prices.len()) }.ok()?;
        for i in 0..first_valid { buf[i] = 0.0; }
        let mut prev_close = f32::NAN; let mut have_prev = false;
        for i in first_valid..prices.len() {
            let c = prices[i];
            buf[i] = if have_prev && c.is_finite() { (c - prev_close).abs() } else { 0.0 };
            if c.is_finite() { prev_close = c; have_prev = true; }
        }
        Some(buf)
    }

    fn launch_batch(&self,
        series_len: usize,
        first_valid: usize,
        d_vis_atr: &DeviceBuffer<i32>,
        d_vis_std: &DeviceBuffer<i32>,
        d_sed_atr: &DeviceBuffer<i32>,
        d_sed_std: &DeviceBuffer<i32>,
        d_threshold: &DeviceBuffer<f32>,
        n_combos: usize,
        d_s: &DeviceBuffer<Float2>,
        d_ss: &DeviceBuffer<Float2>,
        d_tr: &DeviceBuffer<f32>,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaDamianiError> {
        let func = self
            .module
            .get_function("damiani_volatmeter_batch_f32")
            .map_err(|e| CudaDamianiError::Cuda(e.to_string()))?;
        let block_x_env = std::env::var("DAMIANI_BLOCK_X").ok().and_then(|v| v.parse::<u32>().ok());
        // 1 thread per block by default (thread 0 does the scan); overrideable via policy/env
        let block_x = block_x_env
            .or_else(|| match self.policy_batch { BatchKernelPolicy::Plain { block_x } => Some(block_x), _ => None })
            .unwrap_or(1).max(1);
        let grid: GridSize = (n_combos as u32, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe {
            let mut p: u64 = 0; // kernel ignores prices; pass null device ptr
            let mut n = series_len as i32;
            let mut fv = first_valid as i32;
            let mut va = d_vis_atr.as_device_ptr().as_raw();
            let mut vs = d_vis_std.as_device_ptr().as_raw();
            let mut sa = d_sed_atr.as_device_ptr().as_raw();
            let mut ss_ = d_sed_std.as_device_ptr().as_raw();
            let mut th = d_threshold.as_device_ptr().as_raw();
            let mut rows = n_combos as i32;
            let mut s = d_s.as_device_ptr().as_raw();
            let mut ss2 = d_ss.as_device_ptr().as_raw();
            let mut tr = d_tr.as_device_ptr().as_raw();
            let mut o = d_out.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 13] = [
                &mut p as *mut _ as *mut c_void,
                &mut n as *mut _ as *mut c_void,
                &mut fv as *mut _ as *mut c_void,
                &mut va as *mut _ as *mut c_void,
                &mut vs as *mut _ as *mut c_void,
                &mut sa as *mut _ as *mut c_void,
                &mut ss_ as *mut _ as *mut c_void,
                &mut th as *mut _ as *mut c_void,
                &mut rows as *mut _ as *mut c_void,
                &mut s as *mut _ as *mut c_void,
                &mut ss2 as *mut _ as *mut c_void,
                &mut tr as *mut _ as *mut c_void,
                &mut o as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, &mut args)
                .map_err(|e| CudaDamianiError::Cuda(e.to_string()))?;
            (*(self as *const _ as *mut CudaDamianiVolatmeter)).last_batch =
                Some(BatchKernelSelected::Plain { block_x });
        }
        self.maybe_log_batch_debug();
        Ok(())
    }

    pub fn damiani_volatmeter_batch_dev(
        &self,
        prices: &[f32],
        sweep: &DamianiVolatmeterBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<DamianiVolatmeterParams>), CudaDamianiError> {
        let series_len = prices.len();
        if series_len == 0 {
            return Err(CudaDamianiError::InvalidInput("empty series".into()));
        }
        if series_len == 0 {
            return Err(CudaDamianiError::InvalidInput("empty series".into()));
        }
        let combos = Self::expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaDamianiError::InvalidInput("no combinations".into()));
        }
        let first_valid = Self::first_valid_close(prices)?;

        // Validate feasibility
        for prm in &combos {
            let needed = *[
                prm.vis_atr.unwrap(),
                prm.vis_std.unwrap(),
                prm.sed_atr.unwrap(),
                prm.sed_std.unwrap(),
                3,
            ]
            .iter()
            .max()
            .unwrap();
            if series_len - first_valid < needed {
                return Err(CudaDamianiError::InvalidInput(format!(
                    "not enough valid data (need >= {}, have {})",
                    needed,
                    series_len - first_valid
                )));
            }
        }

        // Precomputes (host)
        let tr = Self::compute_tr_close_only(prices, first_valid);
        let (s_prefix_f64, ss_prefix_f64) = Self::compute_prefix_sums(prices, first_valid);
        let s_prefix: Vec<Float2> = Self::pack_double_prefix_to_float2(&s_prefix_f64);
        let ss_prefix: Vec<Float2> = Self::pack_double_prefix_to_float2(&ss_prefix_f64);

        // VRAM: params + prefixes (Float2) + TR + outputs (2 rows per combo)
        let rows = combos.len();
        let req = (rows * (4 * std::mem::size_of::<i32>() + std::mem::size_of::<f32>()))
            + (2 * series_len * std::mem::size_of::<Float2>())
            + (series_len * std::mem::size_of::<f32>()) // TR
            + (2 * rows * series_len * std::mem::size_of::<f32>());
        let headroom = std::env::var("CUDA_MEM_HEADROOM")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(64 * 1024 * 1024);
        let headroom = std::env::var("CUDA_MEM_HEADROOM")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(64 * 1024 * 1024);
        if !Self::will_fit(req, headroom) {
            return Err(CudaDamianiError::InvalidInput(
                "insufficient VRAM for Damiani batch".into(),
            ));
            return Err(CudaDamianiError::InvalidInput(
                "insufficient VRAM for Damiani batch".into(),
            ));
        }

        // Device buffers (prices not uploaded for batch kernel)
        let vis_atr: Vec<i32> = combos.iter().map(|p| p.vis_atr.unwrap() as i32).collect();
        let vis_std: Vec<i32> = combos.iter().map(|p| p.vis_std.unwrap() as i32).collect();
        let sed_atr: Vec<i32> = combos.iter().map(|p| p.sed_atr.unwrap() as i32).collect();
        let sed_std: Vec<i32> = combos.iter().map(|p| p.sed_std.unwrap() as i32).collect();
        let thresh:  Vec<f32> = combos.iter().map(|p| p.threshold.unwrap() as f32).collect();
        let d_va = DeviceBuffer::from_slice(&vis_atr).map_err(|e| CudaDamianiError::Cuda(e.to_string()))?;
        let d_vs = DeviceBuffer::from_slice(&vis_std).map_err(|e| CudaDamianiError::Cuda(e.to_string()))?;
        let d_sa = DeviceBuffer::from_slice(&sed_atr).map_err(|e| CudaDamianiError::Cuda(e.to_string()))?;
        let d_ss_ = DeviceBuffer::from_slice(&sed_std).map_err(|e| CudaDamianiError::Cuda(e.to_string()))?;
        let d_th = DeviceBuffer::from_slice(&thresh).map_err(|e| CudaDamianiError::Cuda(e.to_string()))?;
        // Use pinned upload if available
        let (d_s, d_ss) = match (
            Self::pack_double_prefix_to_float2_pinned(&s_prefix_f64),
            Self::pack_double_prefix_to_float2_pinned(&ss_prefix_f64),
        ) {
            (Some(s_pin), Some(ss_pin)) => {
                let ds = unsafe { DeviceBuffer::from_slice_async(&*s_pin, &self.stream) }
                    .map_err(|e| CudaDamianiError::Cuda(e.to_string()))?;
                let dss = unsafe { DeviceBuffer::from_slice_async(&*ss_pin, &self.stream) }
                    .map_err(|e| CudaDamianiError::Cuda(e.to_string()))?;
                (ds, dss)
            }
            _ => {
                let ds = unsafe { DeviceBuffer::from_slice_async(&s_prefix, &self.stream) }
                    .map_err(|e| CudaDamianiError::Cuda(e.to_string()))?;
                let dss = unsafe { DeviceBuffer::from_slice_async(&ss_prefix, &self.stream) }
                    .map_err(|e| CudaDamianiError::Cuda(e.to_string()))?;
                (ds, dss)
            }
        };
        let d_tr = if let Some(tr_pin) = Self::compute_tr_close_only_pinned(prices, first_valid) {
            unsafe { DeviceBuffer::from_slice_async(&*tr_pin, &self.stream) }
                .map_err(|e| CudaDamianiError::Cuda(e.to_string()))?
        } else {
            unsafe { DeviceBuffer::from_slice_async(&tr, &self.stream) }
                .map_err(|e| CudaDamianiError::Cuda(e.to_string()))?
        };
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(2 * rows * series_len, &self.stream) }
            .map_err(|e| CudaDamianiError::Cuda(e.to_string()))?;

        self.launch_batch(series_len, first_valid, &d_va, &d_vs, &d_sa, &d_ss_, &d_th, rows, &d_s, &d_ss, &d_tr, &mut d_out)?;
        self.stream.synchronize().map_err(|e| CudaDamianiError::Cuda(e.to_string()))?;

        Ok((DeviceArrayF32 { buf: d_out, rows: 2 * rows, cols: series_len }, combos))
    }

    fn launch_many_series(
        &self,
        d_high_tm: &DeviceBuffer<f32>,
        d_low_tm: &DeviceBuffer<f32>,
        d_close_tm: &DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        vis_atr: usize,
        vis_std: usize,
        sed_atr: usize,
        sed_std: usize,
        threshold: f32,
        d_first_valids: &DeviceBuffer<i32>,
        d_s_tm: &DeviceBuffer<Float2>,
        d_ss_tm: &DeviceBuffer<Float2>,
        d_out_tm: &mut DeviceBuffer<f32>) -> Result<(), CudaDamianiError>
    {
        let func = self.module.get_function("damiani_volatmeter_many_series_one_param_time_major_f32")
            .map_err(|e| CudaDamianiError::Cuda(e.to_string()))?;
        let block_x_env = std::env::var("DAMIANI_MANY_BLOCK_X")
            .ok()
            .and_then(|v| v.parse::<u32>().ok());
        let block_x = block_x_env
            .or_else(|| match self.policy_many { ManySeriesKernelPolicy::OneD { block_x } => Some(block_x), _ => None })
            .unwrap_or(1).max(1);
        let grid: GridSize = (cols as u32, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe {
            let mut h = d_high_tm.as_device_ptr().as_raw();
            let mut l = d_low_tm.as_device_ptr().as_raw();
            let mut c = d_close_tm.as_device_ptr().as_raw();
            let mut num_series = cols as i32;
            let mut series_len = rows as i32;
            let mut va = vis_atr as i32;
            let mut vs = vis_std as i32;
            let mut sa = sed_atr as i32;
            let mut ss_ = sed_std as i32;
            let mut th = threshold as f32;
            let mut fv = d_first_valids.as_device_ptr().as_raw();
            let mut s = d_s_tm.as_device_ptr().as_raw();
            let mut s2 = d_ss_tm.as_device_ptr().as_raw();
            let mut o = d_out_tm.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 14] = [
                &mut h as *mut _ as *mut c_void,
                &mut l as *mut _ as *mut c_void,
                &mut c as *mut _ as *mut c_void,
                &mut num_series as *mut _ as *mut c_void,
                &mut series_len as *mut _ as *mut c_void,
                &mut va as *mut _ as *mut c_void,
                &mut vs as *mut _ as *mut c_void,
                &mut sa as *mut _ as *mut c_void,
                &mut ss_ as *mut _ as *mut c_void,
                &mut th as *mut _ as *mut c_void,
                &mut fv as *mut _ as *mut c_void,
                &mut s as *mut _ as *mut c_void,
                &mut s2 as *mut _ as *mut c_void,
                &mut o as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, &mut args)
                .map_err(|e| CudaDamianiError::Cuda(e.to_string()))?;
            (*(self as *const _ as *mut CudaDamianiVolatmeter)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }
        self.maybe_log_many_debug();
        Ok(())
    }

    pub fn damiani_volatmeter_many_series_one_param_time_major_dev(
        &self,
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        cols: usize,
        rows: usize,
        params: &DamianiVolatmeterParams,
    ) -> Result<DeviceArrayF32, CudaDamianiError> {
        if cols == 0 || rows == 0 {
            return Err(CudaDamianiError::InvalidInput("empty matrix".into()));
        }
        if high_tm.len() != low_tm.len() || low_tm.len() != close_tm.len() {
            return Err(CudaDamianiError::InvalidInput(
                "matrix length mismatch".into(),
            ));
        }
        if high_tm.len() != cols * rows {
            return Err(CudaDamianiError::InvalidInput(
                "matrix shape mismatch".into(),
            ));
        }

        let vis_atr = params.vis_atr.unwrap_or(13);
        let vis_std = params.vis_std.unwrap_or(20);
        let sed_atr = params.sed_atr.unwrap_or(40);
        let sed_std = params.sed_std.unwrap_or(100);
        let threshold = params.threshold.unwrap_or(1.4) as f32;

        // Per-series first_valid based on close only
        let mut first_valids = vec![0i32; cols];
        for series in 0..cols {
            let mut fv = None;
            for t in 0..rows {
                let idx = t * cols + series;
                if close_tm[idx].is_finite() {
                    fv = Some(t as i32);
                    break;
                }
            }
            let val = fv.ok_or_else(|| {
                CudaDamianiError::InvalidInput(format!("series {} all NaN", series))
            })?;
            let need = *[vis_atr, vis_std, sed_atr, sed_std, 3]
                .iter()
                .max()
                .unwrap();
            if rows - (val as usize) < need {
                return Err(CudaDamianiError::InvalidInput(format!(
                    "series {} lacks data: need >= {}, valid = {}",
                    series,
                    need,
                    rows - (val as usize)
                )));
            }
            first_valids[series] = val;
        }

        // Precompute stddev prefix (close only) time-major
        let (s_tm_f64, ss_tm_f64) = Self::compute_prefix_sums_time_major(close_tm, cols, rows, &first_valids);
        // Try to build pinned Float2 buffers for faster async H2D; fall back to Vec
        let (s_tm_opt, ss_tm_opt) = (
            Self::pack_double_prefix_to_float2_pinned(&s_tm_f64),
            Self::pack_double_prefix_to_float2_pinned(&ss_tm_f64),
        );
        let (s_tm_vec, ss_tm_vec);
        let (use_pinned_s, use_pinned_ss);
        match (s_tm_opt, ss_tm_opt) {
            (Some(_), Some(_)) => { use_pinned_s = true; use_pinned_ss = true; s_tm_vec = Vec::new(); ss_tm_vec = Vec::new(); }
            _ => {
                use_pinned_s = false; use_pinned_ss = false;
                s_tm_vec = Self::pack_double_prefix_to_float2(&s_tm_f64);
                ss_tm_vec = Self::pack_double_prefix_to_float2(&ss_tm_f64);
            }
        }

        // VRAM estimate: 3 inputs + first_valids + prefixes (Float2) + outputs (2 mats)
        let req = (3 * cols * rows * std::mem::size_of::<f32>())
            + (cols * std::mem::size_of::<i32>())
            + (2 * cols * rows * std::mem::size_of::<Float2>())
            + (2 * cols * rows * std::mem::size_of::<f32>());
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(req, headroom) {
            return Err(CudaDamianiError::InvalidInput(
                "insufficient VRAM for Damiani many-series".into(),
            ));
            return Err(CudaDamianiError::InvalidInput(
                "insufficient VRAM for Damiani many-series".into(),
            ));
        }

        // Device copies
        let d_high = unsafe { DeviceBuffer::from_slice_async(high_tm, &self.stream) }.map_err(|e| CudaDamianiError::Cuda(e.to_string()))?;
        let d_low  = unsafe { DeviceBuffer::from_slice_async(low_tm, &self.stream)  }.map_err(|e| CudaDamianiError::Cuda(e.to_string()))?;
        let d_close= unsafe { DeviceBuffer::from_slice_async(close_tm, &self.stream)}.map_err(|e| CudaDamianiError::Cuda(e.to_string()))?;
        let d_first = DeviceBuffer::from_slice(&first_valids).map_err(|e| CudaDamianiError::Cuda(e.to_string()))?;
        let d_s = if use_pinned_s {
            let pin = Self::pack_double_prefix_to_float2_pinned(&s_tm_f64).unwrap();
            unsafe { DeviceBuffer::from_slice_async(&*pin, &self.stream) }.map_err(|e| CudaDamianiError::Cuda(e.to_string()))?
        } else {
            unsafe { DeviceBuffer::from_slice_async(&s_tm_vec, &self.stream) }.map_err(|e| CudaDamianiError::Cuda(e.to_string()))?
        };
        let d_ss = if use_pinned_ss {
            let pin = Self::pack_double_prefix_to_float2_pinned(&ss_tm_f64).unwrap();
            unsafe { DeviceBuffer::from_slice_async(&*pin, &self.stream) }.map_err(|e| CudaDamianiError::Cuda(e.to_string()))?
        } else {
            unsafe { DeviceBuffer::from_slice_async(&ss_tm_vec, &self.stream) }.map_err(|e| CudaDamianiError::Cuda(e.to_string()))?
        };
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(2 * cols * rows, &self.stream) }
            .map_err(|e| CudaDamianiError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols: 2 * cols,
        })
    }
}

// ---------- Benches ----------
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const COLS_256: usize = 256; // number of series
    const ROWS_8K: usize = 8 * 1024; // timesteps

    fn synth_close(len: usize) -> Vec<f32> { gen_series(len) }

    fn synth_hlc_from_close(close: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut high = close.to_vec();
        let mut low = close.to_vec();
        for i in 0..close.len() {
            let v = close[i];
            if !v.is_finite() { continue; }
            let x = i as f32 * 0.0025;
            let off = (0.002 * x.sin()).abs() + 0.15;
            high[i] = v + off;
            low[i] = v - off;
        }
        (high, low)
    }

    struct BatchState {
        cuda: CudaDamianiVolatmeter,
        close: Vec<f32>,
        sweep: DamianiVolatmeterBatchRange,
    }
    impl CudaBenchState for BatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .damiani_volatmeter_batch_dev(&self.close, &self.sweep)
                .unwrap();
        }
    }

    struct ManySeriesState {
        cuda: CudaDamianiVolatmeter,
        high_tm: Vec<f32>,
        low_tm: Vec<f32>,
        close_tm: Vec<f32>,
        cols: usize,
        rows: usize,
        params: DamianiVolatmeterParams,
    }
    impl CudaBenchState for ManySeriesState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .damiani_volatmeter_many_series_one_param_time_major_dev(
                    &self.high_tm,
                    &self.low_tm,
                    &self.close_tm,
                    self.cols,
                    self.rows,
                    &self.params,
                )
                .unwrap();
        }
    }

    fn prep_batch() -> Box<dyn CudaBenchState> {
        let cuda = CudaDamianiVolatmeter::new(0).expect("cuda damiani");
        let close = synth_close(ONE_SERIES_LEN);
        let sweep = DamianiVolatmeterBatchRange {
            vis_atr: (13, 40, 1),
            vis_std: (20, 40, 1),
            sed_atr: (40, 40, 0),
            sed_std: (100, 100, 0),
            threshold: (1.4, 1.4, 0.0),
        };
        let sweep = DamianiVolatmeterBatchRange {
            vis_atr: (13, 40, 1),
            vis_std: (20, 40, 1),
            sed_atr: (40, 40, 0),
            sed_std: (100, 100, 0),
            threshold: (1.4, 1.4, 0.0),
        };
        Box::new(BatchState { cuda, close, sweep })
    }
    fn prep_many() -> Box<dyn CudaBenchState> {
        let cuda = CudaDamianiVolatmeter::new(0).expect("cuda damiani");
        let cols = COLS_256;
        let rows = ROWS_8K;
        let close_tm = {
            let mut v = vec![f32::NAN; cols * rows];
            for s in 0..cols {
                for t in s..rows {
                    let x = (t as f32) + (s as f32) * 0.2;
                    v[t * cols + s] = (x * 0.002).sin() + 0.0003 * x;
                }
            }
            v
        };
        let (high_tm, low_tm) = synth_hlc_from_close(&close_tm);
        let params = DamianiVolatmeterParams {
            vis_atr: Some(13),
            vis_std: Some(20),
            sed_atr: Some(40),
            sed_std: Some(100),
            threshold: Some(1.4),
        };
        Box::new(ManySeriesState {
            cuda,
            high_tm,
            low_tm,
            close_tm,
            cols,
            rows,
            params,
        })
    }

    fn bytes_batch() -> usize {
        (1 * ONE_SERIES_LEN * std::mem::size_of::<f32>()         // TR only (no prices upload)
            + 2 * ONE_SERIES_LEN * std::mem::size_of::<Float2>() // prefixes as Float2
            + 2 * ONE_SERIES_LEN * std::mem::size_of::<f32>()    // outputs (2 rows)
            + 64 * 1024 * 1024) // headroom
    }
    fn bytes_many() -> usize {
        (3 * COLS_256 * ROWS_8K * std::mem::size_of::<f32>()     // H,L,C
            + COLS_256 * std::mem::size_of::<i32>()
            + 2 * COLS_256 * ROWS_8K * std::mem::size_of::<Float2>() // prefixes as Float2
            + 2 * COLS_256 * ROWS_8K * std::mem::size_of::<f32>()    // outputs
            + 64 * 1024 * 1024)
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "damiani_volatmeter",
                "batch",
                "damiani_cuda_batch",
                "1m",
                prep_batch,
            )
            .with_mem_required(bytes_batch()),
            CudaBenchScenario::new(
                "damiani_volatmeter",
                "many_series_one_param",
                "damiani_cuda_many_series",
                "8k x 256",
                prep_many,
            )
            .with_mem_required(bytes_many()),
            CudaBenchScenario::new(
                "damiani_volatmeter",
                "batch",
                "damiani_cuda_batch",
                "1m",
                prep_batch,
            )
            .with_mem_required(bytes_batch()),
            CudaBenchScenario::new(
                "damiani_volatmeter",
                "many_series_one_param",
                "damiani_cuda_many_series",
                "8k x 256",
                prep_many,
            )
            .with_mem_required(bytes_many()),
        ]
    }
}
