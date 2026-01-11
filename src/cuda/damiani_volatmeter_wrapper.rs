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
use std::sync::Arc;
use thiserror::Error;

use crate::indicators::damiani_volatmeter::{DamianiVolatmeterBatchRange, DamianiVolatmeterParams};


#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Float2 { pub x: f32, pub y: f32 }
unsafe impl DeviceCopy for Float2 {}

#[derive(Error, Debug)]
pub enum CudaDamianiError {
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

/// VRAM-backed array handle for Damiani Volatmeter with context guard and device id
pub struct DeviceArrayF32Damiani {
    pub buf: DeviceBuffer<f32>,
    pub rows: usize,
    pub cols: usize,
    pub ctx: Arc<Context>,
    pub device_id: u32,
}
impl DeviceArrayF32Damiani {
    #[inline]
    pub fn device_ptr(&self) -> u64 { self.buf.as_device_ptr().as_raw() as u64 }
    #[inline]
    pub fn len(&self) -> usize { self.rows.saturating_mul(self.cols) }
}

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
    context: Arc<Context>,
    device_id: u32,
    policy_batch: BatchKernelPolicy,
    policy_many: ManySeriesKernelPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaDamianiVolatmeter {
    pub fn new(device_id: usize) -> Result<Self, CudaDamianiError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let ctx = Arc::new(Context::new(device)?);
        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/damiani_volatmeter_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => {
                if let Ok(m) = Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]) {
                    m
                } else {
                    Module::from_ptx(ptx, &[])?
                }
            }
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        Ok(Self {
            module,
            stream,
            context: ctx,
            device_id: device_id as u32,
            policy_batch: BatchKernelPolicy::Auto,
            policy_many: ManySeriesKernelPolicy::Auto,
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn context_arc(&self) -> Arc<Context> { self.context.clone() }
    pub fn device_id(&self) -> u32 { self.device_id }

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

    
    fn first_valid_close(data: &[f32]) -> Result<usize, CudaDamianiError> {
        if data.is_empty() { return Err(CudaDamianiError::InvalidInput("empty series".into())); }
        if data.is_empty() {
            return Err(CudaDamianiError::InvalidInput("empty series".into()));
        }
        (0..data.len())
            .find(|&i| data[i].is_finite())
            .ok_or_else(|| CudaDamianiError::InvalidInput("all values are NaN".into()))
    }

    fn expand_grid(range: &DamianiVolatmeterBatchRange) -> Result<Vec<DamianiVolatmeterParams>, CudaDamianiError> {
        fn axis_usize((s, e, step): (usize, usize, usize)) -> Result<Vec<usize>, CudaDamianiError> {
            if step == 0 || s == e { return Ok(vec![s]); }
            let mut out = Vec::new();
            if s < e {
                let mut x = s; while x <= e { out.push(x); if let Some(nx) = x.checked_add(step) { x = nx; } else { break; } }
            } else {
                let mut x = s as i64; let step_i = step as i64; while x >= e as i64 { out.push(x as usize); x -= step_i; }
            }
            if out.is_empty() { return Err(CudaDamianiError::InvalidInput("invalid range (usize)".into())); }
            Ok(out)
        }
        fn axis_f64((s, e, step): (f64, f64, f64)) -> Result<Vec<f64>, CudaDamianiError> {
            if step == 0.0 || (s - e).abs() < 1e-12 { return Ok(vec![s]); }
            let mut out = Vec::new(); let eps = 1e-12;
            if s < e {
                if step <= 0.0 { return Err(CudaDamianiError::InvalidInput("invalid range (f64)".into())); }
                let mut x = s; while x <= e + eps { out.push(x); x += step; }
            } else {
                if step <= 0.0 { return Err(CudaDamianiError::InvalidInput("invalid range (f64)".into())); }
                let mut x = s; while x >= e - eps { out.push(x); x -= step; }
            }
            if out.is_empty() { return Err(CudaDamianiError::InvalidInput("invalid range (f64)".into())); }
            Ok(out)
        }
        let a = axis_usize(range.vis_atr)?;
        let b = axis_usize(range.vis_std)?;
        let c = axis_usize(range.sed_atr)?;
        let d = axis_usize(range.sed_std)?;
        let e = axis_f64(range.threshold)?;
        let cap = a
            .len()
            .checked_mul(b.len())
            .and_then(|v| v.checked_mul(c.len()))
            .and_then(|v| v.checked_mul(d.len()))
            .and_then(|v| v.checked_mul(e.len()))
            .ok_or_else(|| CudaDamianiError::InvalidInput("combination count overflow".into()))?;
        if cap == 0 { return Err(CudaDamianiError::InvalidInput("empty combinations".into())); }
        let mut out = Vec::with_capacity(cap);
        for &va in &a { for &vb in &b { for &vc in &c { for &vd in &d { for &ve in &e {
            out.push(DamianiVolatmeterParams { vis_atr: Some(va), vis_std: Some(vb), sed_atr: Some(vc), sed_std: Some(vd), threshold: Some(ve) });
        }}}}}
        Ok(out)
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
            .map_err(|_| CudaDamianiError::MissingKernelSymbol { name: "damiani_volatmeter_batch_f32" })?;
        let block_x_env = std::env::var("DAMIANI_BLOCK_X").ok().and_then(|v| v.parse::<u32>().ok());
        
        
        
        let block_x = block_x_env
            .or_else(|| match self.policy_batch { BatchKernelPolicy::Plain { block_x } => Some(block_x), _ => None })
            .unwrap_or(1).max(1);
        let gx = ((n_combos + (block_x as usize) - 1) / (block_x as usize)).max(1) as u32;
        let grid: GridSize = (gx, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe {
            let mut p: u64 = 0; 
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
                .map_err(CudaDamianiError::Cuda)?;
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
    ) -> Result<(DeviceArrayF32Damiani, Vec<DamianiVolatmeterParams>), CudaDamianiError> {
        let series_len = prices.len();
        if series_len == 0 {
            return Err(CudaDamianiError::InvalidInput("empty series".into()));
        }
        if series_len == 0 {
            return Err(CudaDamianiError::InvalidInput("empty series".into()));
        }
        let combos = Self::expand_grid(sweep)?;
        if combos.is_empty() {
            return Err(CudaDamianiError::InvalidInput("no combinations".into()));
        }
        let first_valid = Self::first_valid_close(prices)?;

        
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

        
        let tr = Self::compute_tr_close_only(prices, first_valid);
        let (s_prefix_f64, ss_prefix_f64) = Self::compute_prefix_sums(prices, first_valid);
        let s_prefix: Vec<Float2> = Self::pack_double_prefix_to_float2(&s_prefix_f64);
        let ss_prefix: Vec<Float2> = Self::pack_double_prefix_to_float2(&ss_prefix_f64);

        
        let rows = combos.len();
        let param_bytes = rows
            .checked_mul(4 * std::mem::size_of::<i32>() + std::mem::size_of::<f32>())
            .ok_or_else(|| CudaDamianiError::InvalidInput("parameter size overflow".into()))?;
        let prefix_bytes = series_len
            .checked_mul(2 * std::mem::size_of::<Float2>())
            .ok_or_else(|| CudaDamianiError::InvalidInput("prefix size overflow".into()))?;
        let tr_bytes = series_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaDamianiError::InvalidInput("TR size overflow".into()))?;
        let out_bytes = rows
            .checked_mul(series_len)
            .and_then(|n| n.checked_mul(2 * std::mem::size_of::<f32>()))
            .ok_or_else(|| CudaDamianiError::InvalidInput("output size overflow".into()))?;
        let req = param_bytes
            .checked_add(prefix_bytes)
            .and_then(|v| v.checked_add(tr_bytes))
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| CudaDamianiError::InvalidInput("total VRAM size overflow".into()))?;
        let headroom = std::env::var("CUDA_MEM_HEADROOM")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(64 * 1024 * 1024);
        if !Self::will_fit(req, headroom) {
            let free = cust::memory::mem_get_info().map(|(f, _)| f).unwrap_or(0);
            return Err(CudaDamianiError::OutOfMemory { required: req, free, headroom });
        }

        
        let vis_atr: Vec<i32> = combos.iter().map(|p| p.vis_atr.unwrap() as i32).collect();
        let vis_std: Vec<i32> = combos.iter().map(|p| p.vis_std.unwrap() as i32).collect();
        let sed_atr: Vec<i32> = combos.iter().map(|p| p.sed_atr.unwrap() as i32).collect();
        let sed_std: Vec<i32> = combos.iter().map(|p| p.sed_std.unwrap() as i32).collect();
        let thresh:  Vec<f32> = combos.iter().map(|p| p.threshold.unwrap() as f32).collect();
        let d_va = DeviceBuffer::from_slice(&vis_atr)?;
        let d_vs = DeviceBuffer::from_slice(&vis_std)?;
        let d_sa = DeviceBuffer::from_slice(&sed_atr)?;
        let d_ss_ = DeviceBuffer::from_slice(&sed_std)?;
        let d_th = DeviceBuffer::from_slice(&thresh)?;
        
        let (d_s, d_ss) = match (
            Self::pack_double_prefix_to_float2_pinned(&s_prefix_f64),
            Self::pack_double_prefix_to_float2_pinned(&ss_prefix_f64),
        ) {
            (Some(s_pin), Some(ss_pin)) => {
                let ds = unsafe { DeviceBuffer::from_slice_async(&*s_pin, &self.stream) }?;
                let dss = unsafe { DeviceBuffer::from_slice_async(&*ss_pin, &self.stream) }?;
                (ds, dss)
            }
            _ => {
                let ds = unsafe { DeviceBuffer::from_slice_async(&s_prefix, &self.stream) }?;
                let dss = unsafe { DeviceBuffer::from_slice_async(&ss_prefix, &self.stream) }?;
                (ds, dss)
            }
        };
        let d_tr = if let Some(tr_pin) = Self::compute_tr_close_only_pinned(prices, first_valid) {
            unsafe { DeviceBuffer::from_slice_async(&*tr_pin, &self.stream) }
                .map_err(CudaDamianiError::Cuda)?
        } else {
            unsafe { DeviceBuffer::from_slice_async(&tr, &self.stream) }
                .map_err(CudaDamianiError::Cuda)?
        };
        let len_out = rows
            .checked_mul(series_len)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| CudaDamianiError::InvalidInput("output element count overflow".into()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(len_out, &self.stream) }
                .map_err(CudaDamianiError::Cuda)?;

        self.launch_batch(
            series_len,
            first_valid,
            &d_va,
            &d_vs,
            &d_sa,
            &d_ss_,
            &d_th,
            rows,
            &d_s,
            &d_ss,
            &d_tr,
            &mut d_out,
        )?;
        self.stream
            .synchronize()
            .map_err(CudaDamianiError::Cuda)?;

        Ok((DeviceArrayF32Damiani { buf: d_out, rows: 2 * rows, cols: series_len, ctx: Arc::clone(&self.context), device_id: self.device_id }, combos))
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
        let func = self.module
            .get_function("damiani_volatmeter_many_series_one_param_time_major_f32")
            .map_err(|_| CudaDamianiError::MissingKernelSymbol { name: "damiani_volatmeter_many_series_one_param_time_major_f32" })?;
        let block_x_env = std::env::var("DAMIANI_MANY_BLOCK_X")
            .ok()
            .and_then(|v| v.parse::<u32>().ok());
        let block_x = block_x_env
            .or_else(|| match self.policy_many { ManySeriesKernelPolicy::OneD { block_x } => Some(block_x), _ => None })
            .unwrap_or(1).max(1);
        let gx = ((cols + (block_x as usize) - 1) / (block_x as usize)).max(1) as u32;
        let grid: GridSize = (gx, 1, 1).into();
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
                .map_err(CudaDamianiError::Cuda)?;
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
    ) -> Result<DeviceArrayF32Damiani, CudaDamianiError> {
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

        
        let (s_tm_f64, ss_tm_f64) = Self::compute_prefix_sums_time_major(close_tm, cols, rows, &first_valids);
        
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

        
        let elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaDamianiError::InvalidInput("matrix size overflow".into()))?;
        let inputs_bytes = 3usize
            .checked_mul(elems)
            .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| CudaDamianiError::InvalidInput("input size overflow".into()))?;
        let first_valid_bytes = cols
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaDamianiError::InvalidInput("first_valids size overflow".into()))?;
        let prefix_bytes = elems
            .checked_mul(2)
            .and_then(|n| n.checked_mul(std::mem::size_of::<Float2>()))
            .ok_or_else(|| CudaDamianiError::InvalidInput("prefix size overflow".into()))?;
        let out_bytes = elems
            .checked_mul(2)
            .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| CudaDamianiError::InvalidInput("output size overflow".into()))?;
        let req = inputs_bytes
            .checked_add(first_valid_bytes)
            .and_then(|v| v.checked_add(prefix_bytes))
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| CudaDamianiError::InvalidInput("total VRAM size overflow".into()))?;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(req, headroom) {
            let free = cust::memory::mem_get_info().map(|(f, _)| f).unwrap_or(0);
            return Err(CudaDamianiError::OutOfMemory { required: req, free, headroom });
        }

        
        let d_high = unsafe { DeviceBuffer::from_slice_async(high_tm, &self.stream) }?;
        let d_low  = unsafe { DeviceBuffer::from_slice_async(low_tm, &self.stream)  }?;
        let d_close= unsafe { DeviceBuffer::from_slice_async(close_tm, &self.stream)}?;
        let d_first = DeviceBuffer::from_slice(&first_valids)?;
        let d_s = if use_pinned_s {
            let pin = Self::pack_double_prefix_to_float2_pinned(&s_tm_f64).unwrap();
            unsafe { DeviceBuffer::from_slice_async(&*pin, &self.stream) }?
        } else {
            unsafe { DeviceBuffer::from_slice_async(&s_tm_vec, &self.stream) }?
        };
        let d_ss = if use_pinned_ss {
            let pin = Self::pack_double_prefix_to_float2_pinned(&ss_tm_f64).unwrap();
            unsafe { DeviceBuffer::from_slice_async(&*pin, &self.stream) }?
        } else {
            unsafe { DeviceBuffer::from_slice_async(&ss_tm_vec, &self.stream) }?
        };
        let len_out = elems
            .checked_mul(2)
            .ok_or_else(|| CudaDamianiError::InvalidInput("output element count overflow".into()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(len_out, &self.stream) }?;

        self.launch_many_series(
            &d_high,
            &d_low,
            &d_close,
            cols,
            rows,
            vis_atr,
            vis_std,
            sed_atr,
            sed_std,
            threshold,
            &d_first,
            &d_s,
            &d_ss,
            &mut d_out,
        )?;
        self.stream
            .synchronize()
            .map_err(CudaDamianiError::Cuda)?;

        Ok(DeviceArrayF32Damiani {
            buf: d_out,
            rows,
            cols: 2 * cols,
            ctx: Arc::clone(&self.context),
            device_id: self.device_id,
        })
    }
}


pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const COLS_256: usize = 256; 
    const ROWS_8K: usize = 8 * 1024; 

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

    struct BatchDevState {
        cuda: CudaDamianiVolatmeter,
        series_len: usize,
        first_valid: usize,
        d_va: DeviceBuffer<i32>,
        d_vs: DeviceBuffer<i32>,
        d_sa: DeviceBuffer<i32>,
        d_ss_: DeviceBuffer<i32>,
        d_th: DeviceBuffer<f32>,
        rows: usize,
        d_s: DeviceBuffer<Float2>,
        d_ss: DeviceBuffer<Float2>,
        d_tr: DeviceBuffer<f32>,
        d_out: DeviceBuffer<f32>,
    }
    impl CudaBenchState for BatchDevState {
        fn launch(&mut self) {
            self.cuda
                .launch_batch(
                    self.series_len,
                    self.first_valid,
                    &self.d_va,
                    &self.d_vs,
                    &self.d_sa,
                    &self.d_ss_,
                    &self.d_th,
                    self.rows,
                    &self.d_s,
                    &self.d_ss,
                    &self.d_tr,
                    &mut self.d_out,
                )
                .expect("damiani batch kernel");
            self.cuda.stream.synchronize().expect("damiani sync");
        }
    }

    struct ManySeriesState {
        cuda: CudaDamianiVolatmeter,
        d_high_tm: DeviceBuffer<f32>,
        d_low_tm: DeviceBuffer<f32>,
        d_close_tm: DeviceBuffer<f32>,
        d_first_valids: DeviceBuffer<i32>,
        d_s_tm: DeviceBuffer<Float2>,
        d_ss_tm: DeviceBuffer<Float2>,
        cols: usize,
        rows: usize,
        vis_atr: usize,
        vis_std: usize,
        sed_atr: usize,
        sed_std: usize,
        threshold: f32,
        d_out_tm: DeviceBuffer<f32>,
    }
    impl CudaBenchState for ManySeriesState {
        fn launch(&mut self) {
            self.cuda
                .launch_many_series(
                    &self.d_high_tm,
                    &self.d_low_tm,
                    &self.d_close_tm,
                    self.cols,
                    self.rows,
                    self.vis_atr,
                    self.vis_std,
                    self.sed_atr,
                    self.sed_std,
                    self.threshold,
                    &self.d_first_valids,
                    &self.d_s_tm,
                    &self.d_ss_tm,
                    &mut self.d_out_tm,
                )
                .expect("damiani many-series kernel");
            self.cuda.stream.synchronize().expect("damiani sync");
        }
    }

    fn prep_batch_dev() -> Box<dyn CudaBenchState> {
        let cuda = CudaDamianiVolatmeter::new(0).expect("cuda damiani");
        let close = synth_close(ONE_SERIES_LEN);
        let sweep = DamianiVolatmeterBatchRange {
            vis_atr: (13, 40, 2),
            vis_std: (20, 40, 2),
            sed_atr: (40, 40, 0),
            sed_std: (100, 100, 0),
            threshold: (1.4, 1.4, 0.0),
        };
        let combos = CudaDamianiVolatmeter::expand_grid(&sweep).expect("damiani expand grid");
        let first_valid = CudaDamianiVolatmeter::first_valid_close(&close).expect("damiani first_valid");

        
        let tr = CudaDamianiVolatmeter::compute_tr_close_only(&close, first_valid);
        let (s_prefix_f64, ss_prefix_f64) = CudaDamianiVolatmeter::compute_prefix_sums(&close, first_valid);
        let s_prefix: Vec<Float2> = CudaDamianiVolatmeter::pack_double_prefix_to_float2(&s_prefix_f64);
        let ss_prefix: Vec<Float2> = CudaDamianiVolatmeter::pack_double_prefix_to_float2(&ss_prefix_f64);

        let rows = combos.len();
        let vis_atr: Vec<i32> = combos.iter().map(|p| p.vis_atr.unwrap() as i32).collect();
        let vis_std: Vec<i32> = combos.iter().map(|p| p.vis_std.unwrap() as i32).collect();
        let sed_atr: Vec<i32> = combos.iter().map(|p| p.sed_atr.unwrap() as i32).collect();
        let sed_std: Vec<i32> = combos.iter().map(|p| p.sed_std.unwrap() as i32).collect();
        let thresh: Vec<f32> = combos.iter().map(|p| p.threshold.unwrap() as f32).collect();

        let d_va = unsafe { DeviceBuffer::from_slice_async(&vis_atr, &cuda.stream) }.expect("d_va");
        let d_vs = unsafe { DeviceBuffer::from_slice_async(&vis_std, &cuda.stream) }.expect("d_vs");
        let d_sa = unsafe { DeviceBuffer::from_slice_async(&sed_atr, &cuda.stream) }.expect("d_sa");
        let d_ss_ = unsafe { DeviceBuffer::from_slice_async(&sed_std, &cuda.stream) }.expect("d_ss");
        let d_th = unsafe { DeviceBuffer::from_slice_async(&thresh, &cuda.stream) }.expect("d_th");
        let d_s = unsafe { DeviceBuffer::from_slice_async(&s_prefix, &cuda.stream) }.expect("d_s");
        let d_ss = unsafe { DeviceBuffer::from_slice_async(&ss_prefix, &cuda.stream) }.expect("d_ss2");
        let d_tr = unsafe { DeviceBuffer::from_slice_async(&tr, &cuda.stream) }.expect("d_tr");

        let len_out = (2usize)
            .checked_mul(rows)
            .and_then(|x| x.checked_mul(ONE_SERIES_LEN))
            .expect("damiani out size overflow");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(len_out, &cuda.stream) }.expect("d_out");
        cuda.stream.synchronize().expect("damiani sync");

        Box::new(BatchDevState {
            cuda,
            series_len: ONE_SERIES_LEN,
            first_valid,
            d_va,
            d_vs,
            d_sa,
            d_ss_,
            d_th,
            rows,
            d_s,
            d_ss,
            d_tr,
            d_out,
        })
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
        let (vis_atr, vis_std, sed_atr, sed_std, threshold) = (13usize, 20usize, 40usize, 100usize, 1.4f32);
        let mut first_valids = vec![-1i32; cols];
        for series in 0..cols {
            let mut fv = None;
            for t in 0..rows {
                let idx = t * cols + series;
                if close_tm[idx].is_finite() {
                    fv = Some(t as i32);
                    break;
                }
            }
            first_valids[series] = fv.expect("series all NaN");
        }
        let (s_tm_f64, ss_tm_f64) =
            CudaDamianiVolatmeter::compute_prefix_sums_time_major(&close_tm, cols, rows, &first_valids);
        let s_tm: Vec<Float2> = CudaDamianiVolatmeter::pack_double_prefix_to_float2(&s_tm_f64);
        let ss_tm: Vec<Float2> = CudaDamianiVolatmeter::pack_double_prefix_to_float2(&ss_tm_f64);

        let d_high_tm = DeviceBuffer::from_slice(&high_tm).expect("d_high_tm");
        let d_low_tm = DeviceBuffer::from_slice(&low_tm).expect("d_low_tm");
        let d_close_tm = DeviceBuffer::from_slice(&close_tm).expect("d_close_tm");
        let d_first_valids = DeviceBuffer::from_slice(&first_valids).expect("d_first_valids");
        let d_s_tm = DeviceBuffer::from_slice(&s_tm).expect("d_s_tm");
        let d_ss_tm = DeviceBuffer::from_slice(&ss_tm).expect("d_ss_tm");
        let elems = cols * rows;
        let d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems * 2) }.expect("d_out_tm");
        cuda.stream.synchronize().expect("sync after prep");
        Box::new(ManySeriesState {
            cuda,
            d_high_tm,
            d_low_tm,
            d_close_tm,
            d_first_valids,
            d_s_tm,
            d_ss_tm,
            cols,
            rows,
            vis_atr,
            vis_std,
            sed_atr,
            sed_std,
            threshold,
            d_out_tm,
        })
    }

    fn bytes_batch() -> usize {
        let combos = 154usize; 
        let param_bytes = combos * (4 * std::mem::size_of::<i32>() + std::mem::size_of::<f32>());
        let prefix_bytes = ONE_SERIES_LEN * (2 * std::mem::size_of::<Float2>());
        let tr_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = combos * ONE_SERIES_LEN * (2 * std::mem::size_of::<f32>());
        param_bytes + prefix_bytes + tr_bytes + out_bytes + 64 * 1024 * 1024
    }
    fn bytes_many() -> usize {
        (3 * COLS_256 * ROWS_8K * std::mem::size_of::<f32>()     
            + COLS_256 * std::mem::size_of::<i32>()
            + 2 * COLS_256 * ROWS_8K * std::mem::size_of::<Float2>() 
            + 2 * COLS_256 * ROWS_8K * std::mem::size_of::<f32>()    
            + 64 * 1024 * 1024)
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "damiani_volatmeter",
                "batch",
                "damiani_cuda_batch_dev",
                "1m_x_154",
                prep_batch_dev,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_batch()),
            CudaBenchScenario::new(
                "damiani_volatmeter",
                "many_series_one_param",
                "damiani_cuda_many_series",
                "8k x 256",
                prep_many,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_many()),
        ]
    }
}
