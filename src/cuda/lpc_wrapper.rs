//! CUDA wrapper for the Low Pass Channel (LPC) indicator.
//!
//! Parity goals (per Agents Guide):
//! - ALMA-style PTX load (DetermineTargetFromContext + O2 fallback), NON_BLOCKING stream
//! - VRAM checks + ~64MB headroom; chunk grid.y if needed
//! - Batch: one-series × many-params; optional host-precomputed dominant cycle reused across rows
//! - Many-series × one-param (time-major): per-series sequential scan (fixed cutoff only for now)

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::cuda::wto_wrapper::DeviceArrayF32Triplet;
use crate::indicators::lpc::{dom_cycle, LpcBatchRange, LpcParams};
use cust::error::CudaError;
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use thiserror::Error;
use std::env;
use std::ffi::c_void;
use std::fmt;
use std::sync::Arc;

#[inline]
fn alpha_from_period_f32(p: i32) -> f32 {
    let p = p.max(1) as f64;
    let omega = 2.0_f64 * std::f64::consts::PI / p;
    let (s, c) = omega.sin_cos();
    ((1.0 - s) / c) as f32
}

#[inline]
fn build_alpha_lut(p_min: i32, p_max: i32) -> (Vec<f32>, i32) {
    debug_assert!(p_max >= p_min && p_min >= 1);
    let mut lut = Vec::with_capacity((p_max - p_min + 1) as usize);
    for p in p_min..=p_max {
        lut.push(alpha_from_period_f32(p));
    }
    (lut, p_min)
}
#[derive(thiserror::Error, Debug)]
pub enum CudaLpcError {
    #[error(transparent)]
    Cuda(#[from] CudaError),
    #[error("out of memory: required={required}B free={free}B headroom={headroom}B")]
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
    #[error("invalid range: start={start} end={end} step={step}")]
    InvalidRange { start: usize, end: usize, step: usize },
    #[error("not implemented")]
    NotImplemented,
}

#[derive(Clone, Copy, Debug, Default)]
pub enum BatchKernelPolicy {
    #[default]
    Auto,
    Plain {
        block_x: u32,
    },
}
#[derive(Clone, Copy, Debug, Default)]
pub enum ManySeriesKernelPolicy {
    #[default]
    Auto,
    OneD {
        block_x: u32,
    },
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaLpcPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected {
    Plain { block_x: u32 },
}
#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}

pub struct CudaLpc {
    module: Module,
    stream: Stream,
    _context: Arc<Context>,
    device_id: u32,
    policy: CudaLpcPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaLpc {
    pub fn new(device_id: usize) -> Result<Self, CudaLpcError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/lpc_kernel.ptx"));
        let module = Module::from_ptx(
            ptx,
            &[
                ModuleJitOption::DetermineTargetFromContext,
                ModuleJitOption::OptLevel(OptLevel::O2),
            ],
        )
        .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
        .or_else(|_| Module::from_ptx(ptx, &[]))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaLpcPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    
    pub fn set_policy(&mut self, p: CudaLpcPolicy) {
        self.policy = p;
    }
    pub fn policy(&self) -> &CudaLpcPolicy {
        &self.policy
    }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch
    }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many
    }

    pub fn context_arc(&self) -> Arc<Context> {
        self._context.clone()
    }
    pub fn device_id(&self) -> u32 {
        self.device_id
    }
    pub fn synchronize(&self) -> Result<(), CudaLpcError> {
        self.stream.synchronize()?;
        Ok(())
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        }
    }
    #[inline]
    fn device_mem_info() -> Option<(usize, usize)> {
        mem_get_info().ok()
    }
    #[inline]
    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaLpcError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        if let Some((free, _)) = Self::device_mem_info() {
            if required.saturating_add(headroom) <= free {
                Ok(())
            } else {
                Err(CudaLpcError::OutOfMemory {
                    required,
                    free,
                    headroom,
                })
            }
        } else {
            Ok(())
        }
    }

    fn expand_grid(range: &LpcBatchRange) -> Result<Vec<LpcParams>, CudaLpcError> {
        fn axis_usize(
            (start, end, step): (usize, usize, usize),
        ) -> Result<Vec<usize>, CudaLpcError> {
            if step == 0 || start == end {
                return Ok(vec![start]);
            }
            let mut vals = Vec::new();
            if start < end {
                let mut v = start;
                while v <= end {
                    vals.push(v);
                    match v.checked_add(step) {
                        Some(next) => {
                            if next == v {
                                break;
                            }
                            v = next;
                        }
                        None => break,
                    }
                }
            } else {
                let mut v = start;
                while v >= end {
                    vals.push(v);
                    if v == 0 {
                        break;
                    }
                    let next = v.saturating_sub(step);
                    if next == v {
                        break;
                    }
                    v = next;
                    if v < end {
                        break;
                    }
                }
            }
            if vals.is_empty() {
                return Err(CudaLpcError::InvalidRange {
                    start,
                    end,
                    step,
                });
            }
            Ok(vals)
        }
        fn axis_f64(
            (start, end, step): (f64, f64, f64),
        ) -> Result<Vec<f64>, CudaLpcError> {
            if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
                return Ok(vec![start]);
            }
            let mut out = Vec::new();
            if start < end {
                let st = if step > 0.0 { step } else { -step };
                let mut x = start;
                while x <= end + 1e-12 {
                    out.push(x);
                    x += st;
                }
            } else {
                let st = if step > 0.0 { -step } else { step };
                if st.abs() < 1e-12 {
                    return Ok(vec![start]);
                }
                let mut x = start;
                while x >= end - 1e-12 {
                    out.push(x);
                    x += st;
                }
            }
            if out.is_empty() {
                return Err(CudaLpcError::InvalidRange {
                    start: start as usize,
                    end: end as usize,
                    step: step as usize,
                });
            }
            Ok(out)
        }
        let ps = axis_usize(range.fixed_period)?;
        let cms = axis_f64(range.cycle_mult)?;
        let tms = axis_f64(range.tr_mult)?;
        let cap = ps
            .len()
            .checked_mul(cms.len())
            .and_then(|v| v.checked_mul(tms.len()))
            .ok_or(CudaLpcError::InvalidRange {
                start: range.fixed_period.0,
                end: range.fixed_period.1,
                step: range.fixed_period.2,
            })?;
        let mut out = Vec::with_capacity(cap);
        for &p in &ps {
            for &cm in &cms {
                for &tm in &tms {
                    out.push(LpcParams {
                        cutoff_type: Some(range.cutoff_type.clone()),
                        fixed_period: Some(p),
                        max_cycle_limit: Some(range.max_cycle_limit),
                        cycle_mult: Some(cm),
                        tr_mult: Some(tm),
                    });
                }
            }
        }
        Ok(out)
    }

    fn first_valid_ohlc4(h: &[f32], l: &[f32], c: &[f32], s: &[f32]) -> Option<usize> {
        (0..s.len())
            .find(|&i| h[i].is_finite() && l[i].is_finite() && c[i].is_finite() && s[i].is_finite())
    }

    pub fn lpc_batch_dev(
        &self,
        high: &[f32],
        low: &[f32],
        close: &[f32],
        src: &[f32],
        range: &LpcBatchRange,
    ) -> Result<(DeviceArrayF32Triplet, Vec<LpcParams>), CudaLpcError> {
        if high.len() != low.len() || high.len() != close.len() || high.len() != src.len() {
            return Err(CudaLpcError::InvalidInput("length mismatch".into()));
        }
        if src.is_empty() {
            return Err(CudaLpcError::InvalidInput("empty input".into()));
        }
        let len = src.len();
        let first = Self::first_valid_ohlc4(high, low, close, src)
            .ok_or_else(|| CudaLpcError::InvalidInput("all values are NaN".into()))?;
        if len.saturating_sub(first) < 2 {
            return Err(CudaLpcError::InvalidInput(
                "not enough valid data after first".into(),
            ));
        }

        // Build combos
        let combos = Self::expand_grid(range)?;
        if combos.is_empty() {
            return Err(CudaLpcError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        for p in &combos {
            let fp = p.fixed_period.unwrap_or(0);
            if fp == 0 || fp > len {
                return Err(CudaLpcError::InvalidInput("invalid fixed_period".into()));
            }
        }

        let n = len;
        let rows = combos.len();

        // VRAM estimate (checked arithmetic)
        let item_bytes = std::mem::size_of::<f32>();
        let inputs_elems = high
            .len()
            .checked_add(low.len())
            .and_then(|v| v.checked_add(close.len()))
            .and_then(|v| v.checked_add(src.len()))
            .ok_or_else(|| CudaLpcError::InvalidInput("input length overflow".into()))?;
        let bytes_inputs = inputs_elems
            .checked_mul(item_bytes)
            .ok_or_else(|| CudaLpcError::InvalidInput("input bytes overflow".into()))?;
        let bytes_params = rows
            .checked_mul(3 * item_bytes)
            .ok_or_else(|| CudaLpcError::InvalidInput("params bytes overflow".into()))?; // approx
        let bytes_outputs = rows
            .checked_mul(n)
            .and_then(|v| v.checked_mul(3 * item_bytes))
            .ok_or_else(|| CudaLpcError::InvalidInput("output bytes overflow".into()))?;
        let mut bytes_dom = 0usize;
        let cutoff_adaptive = range.cutoff_type.eq_ignore_ascii_case("adaptive");
        let dom_host_f32: Option<Vec<f32>> = if cutoff_adaptive {
            // Host-precompute dominant cycle once for the series and upload as FP32
            let src64: Vec<f64> = src.iter().map(|&v| v as f64).collect();
            let dc = dom_cycle(&src64, range.max_cycle_limit);
            let v32: Vec<f32> = dc.iter().map(|&v| v as f32).collect();
            bytes_dom = v32
                .len()
                .checked_mul(item_bytes)
                .ok_or_else(|| CudaLpcError::InvalidInput("dom bytes overflow".into()))?;
            Some(v32)
        } else {
            None
        };
        let required = bytes_inputs
            .checked_add(bytes_params)
            .and_then(|v| v.checked_add(bytes_outputs))
            .and_then(|v| v.checked_add(bytes_dom))
            .ok_or_else(|| CudaLpcError::InvalidInput("total bytes overflow".into()))?;
        Self::will_fit(required, 64 * 1024 * 1024)?;

        // Device buffers
        let d_h = DeviceBuffer::from_slice(high)?;
        let d_l = DeviceBuffer::from_slice(low)?;
        let d_c = DeviceBuffer::from_slice(close)?;
        let d_s = DeviceBuffer::from_slice(src)?;

        // Shared precompute across rows: True Range once for the series
        fn host_true_range_f32(h: &[f32], l: &[f32], c: &[f32]) -> Vec<f32> {
            let n = h.len();
            let mut tr = vec![0f32; n];
            if n == 0 {
                return tr;
            }
            if n == 0 {
                return tr;
            }
            tr[0] = h[0] - l[0];
            for i in 1..n {
                let hl = h[i] - l[i];
                let c_l1 = (c[i] - l[i - 1]).abs();
                let c_h1 = (c[i] - h[i - 1]).abs();
                tr[i] = hl.max(c_l1).max(c_h1);
            }
            tr
        }
        let tr_host = host_true_range_f32(high, low, close);
        let d_tr = DeviceBuffer::from_slice(&tr_host)?;

        // Params
        let periods: Vec<i32> =
            combos.iter().map(|p| p.fixed_period.unwrap() as i32).collect();
        let cms: Vec<f32> = combos.iter().map(|p| p.cycle_mult.unwrap() as f32).collect();
        let tms: Vec<f32> = combos.iter().map(|p| p.tr_mult.unwrap() as f32).collect();
        let d_periods = DeviceBuffer::from_slice(&periods)?;
        let d_cms = DeviceBuffer::from_slice(&cms)?;
        let d_tms = DeviceBuffer::from_slice(&tms)?;
        let d_dom = if let Some(v) = &dom_host_f32 {
            Some(DeviceBuffer::from_slice(v)?)
        } else {
            None
        };

        // Optional alpha LUT for adaptive mode
        let (d_alpha_lut, alpha_lut_len_i32, alpha_lut_pmin_i32) = if cutoff_adaptive {
            let p_min = 3i32;
            let max_fixed = *periods.iter().max().unwrap_or(&p_min);
            let cm_max = cms.iter().copied().fold(0.0f32, f32::max);
            let dom_max = dom_host_f32
                .as_ref()
                .map(|v| v.iter().copied().fold(0.0f32, f32::max))
                .unwrap_or(0.0f32);
            let mut from_dom = (dom_max * cm_max).ceil() as i32;
            let max_cap = if range.max_cycle_limit > 0 { range.max_cycle_limit as i32 } else { i32::MAX };
            if from_dom > max_cap { from_dom = max_cap; }
            let p_max = max_fixed.max(from_dom.max(p_min));
            let (lut, pmin) = build_alpha_lut(p_min, p_max);
            let len_i32 = lut.len() as i32;
            let buf = DeviceBuffer::from_slice(&lut)?;
            (Some(buf), len_i32, pmin)
        } else { (None, 0, 0) };

        // Outputs row-major [combos, len] to preserve API
        let out_elems = rows
            .checked_mul(n)
            .ok_or_else(|| CudaLpcError::InvalidInput("output length overflow".into()))?;
        let mut d_f = unsafe { DeviceBuffer::<f32>::uninitialized(out_elems) }?;
        let mut d_hi = unsafe { DeviceBuffer::<f32>::uninitialized(out_elems) }?;
        let mut d_lo = unsafe { DeviceBuffer::<f32>::uninitialized(out_elems) }?;

        // Launch v2 kernel; keep out_time_major=0 for API compatibility
        let func = self
            .module
            .get_function("lpc_batch_f32_v2")
            .map_err(|_| CudaLpcError::MissingKernelSymbol { name: "lpc_batch_f32_v2" })?;
        let block_x = match self.policy.batch { BatchKernelPolicy::Auto => 256, BatchKernelPolicy::Plain { block_x } => block_x };
        let grid_x = ((rows as u32) + block_x - 1) / block_x;
        if block_x == 0 || grid_x == 0 || grid_x > 65_535 {
            return Err(CudaLpcError::LaunchConfigTooLarge {
                gx: grid_x,
                gy: 1,
                gz: 1,
                bx: block_x,
                by: 1,
                bz: 1,
            });
        }
        unsafe {
            let grid: GridSize = ((grid_x, 1, 1)).into();
            let block: BlockSize = ((block_x, 1, 1)).into();
            let mut h_ptr = d_h.as_device_ptr().as_raw();
            let mut l_ptr = d_l.as_device_ptr().as_raw();
            let mut c_ptr = d_c.as_device_ptr().as_raw();
            let mut s_ptr = d_s.as_device_ptr().as_raw();
            let mut len_i = n as i32;
            let mut tr_ptr = d_tr.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut cms_ptr = d_cms.as_device_ptr().as_raw();
            let mut tms_ptr = d_tms.as_device_ptr().as_raw();
            let mut combos_i = rows as i32;
            let mut first_i = first as i32;
            let mut cutoff_i = if cutoff_adaptive { 1i32 } else { 0i32 };
            let mut maxcl_i = range.max_cycle_limit as i32;
            let mut dom_ptr: *const f32 = if let Some(ref d) = d_dom { d.as_device_ptr().as_raw() as *const f32 } else { std::ptr::null() };
            let mut alpha_ptr: *const f32 = if let Some(ref d) = d_alpha_lut { d.as_device_ptr().as_raw() as *const f32 } else { std::ptr::null() };
            let mut alpha_len = alpha_lut_len_i32;
            let mut alpha_pmin = alpha_lut_pmin_i32;
            let mut out_time_major = 0i32; // keep row-major externally
            let mut out_f_ptr = d_f.as_device_ptr().as_raw();
            let mut out_hi_ptr = d_hi.as_device_ptr().as_raw();
            let mut out_lo_ptr = d_lo.as_device_ptr().as_raw();

            let mut args: [*mut c_void; 21] = [
                &mut h_ptr as *mut _ as *mut c_void,
                &mut l_ptr as *mut _ as *mut c_void,
                &mut c_ptr as *mut _ as *mut c_void,
                &mut s_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut tr_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut cms_ptr as *mut _ as *mut c_void,
                &mut tms_ptr as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut first_i as *mut _ as *mut c_void,
                &mut cutoff_i as *mut _ as *mut c_void,
                &mut maxcl_i as *mut _ as *mut c_void,
                &mut dom_ptr as *mut _ as *mut c_void,
                &mut alpha_ptr as *mut _ as *mut c_void,
                &mut alpha_len as *mut _ as *mut c_void,
                &mut alpha_pmin as *mut _ as *mut c_void,
                &mut out_time_major as *mut _ as *mut c_void,
                &mut out_f_ptr as *mut _ as *mut c_void,
                &mut out_hi_ptr as *mut _ as *mut c_void,
                &mut out_lo_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, &mut args)
                ?;
        }
        self.stream
            .synchronize()?;

        let triplet = DeviceArrayF32Triplet {
            wt1: DeviceArrayF32 { buf: d_f, rows, cols: n },
            wt2: DeviceArrayF32 { buf: d_hi, rows, cols: n },
            hist: DeviceArrayF32 { buf: d_lo, rows, cols: n },
        };
        unsafe {
            (*(self as *const _ as *mut CudaLpc)).last_batch =
                Some(BatchKernelSelected::Plain { block_x });
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") && !self.debug_batch_logged {
            eprintln!("[DEBUG] lpc batch selected kernel: {:?}", self.last_batch);
            unsafe {
                (*(self as *const _ as *mut CudaLpc)).debug_batch_logged = true;
            }
        }
        Ok((triplet, combos))
    }

    pub fn lpc_many_series_one_param_time_major_dev(
        &self,
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        src_tm: &[f32],
        cols: usize,
        rows: usize,
        params: &LpcParams,
    ) -> Result<DeviceArrayF32Triplet, CudaLpcError> {
        if cols == 0 || rows == 0 {
            return Err(CudaLpcError::InvalidInput("empty matrix".into()));
        }
        if [high_tm.len(), low_tm.len(), close_tm.len(), src_tm.len()]
            .iter()
            .copied()
            .any(|n| n != cols * rows)
        {
            return Err(CudaLpcError::InvalidInput("length mismatch".into()));
        }
        let cutoff_type = params
            .cutoff_type
            .clone()
            .unwrap_or_else(|| "adaptive".to_string());
        if !cutoff_type.eq_ignore_ascii_case("fixed") {
            return Err(CudaLpcError::InvalidInput(
                "many-series CUDA supports fixed cutoff only".into(),
            ));
        }
        let fixed_period = params.fixed_period.unwrap_or(20);
        if fixed_period == 0 || fixed_period > rows {
            return Err(CudaLpcError::InvalidInput("invalid period".into()));
        }
        let tr_mult = params.tr_mult.unwrap_or(1.0) as f32;

        // VRAM estimate for many-series path (checked)
        let item_bytes = std::mem::size_of::<f32>();
        let prices_bytes = cols
            .checked_mul(rows)
            .and_then(|v| v.checked_mul(4 * item_bytes))
            .ok_or_else(|| CudaLpcError::InvalidInput("prices bytes overflow".into()))?;
        let first_bytes = cols
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaLpcError::InvalidInput("first bytes overflow".into()))?;
        let out_bytes = cols
            .checked_mul(rows)
            .and_then(|v| v.checked_mul(3 * item_bytes))
            .ok_or_else(|| CudaLpcError::InvalidInput("output bytes overflow".into()))?;
        let required = prices_bytes
            .checked_add(first_bytes)
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| CudaLpcError::InvalidInput("total bytes overflow".into()))?;
        Self::will_fit(required, 64 * 1024 * 1024)?;

        // Per-series first_valid
        let mut firsts = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = rows as i32;
            for t in 0..rows {
                let i = t * cols + s;
                if high_tm[i].is_finite()
                    && low_tm[i].is_finite()
                    && close_tm[i].is_finite()
                    && src_tm[i].is_finite()
                {
                    fv = t as i32;
                    break;
                }
            }
            if fv >= rows as i32 {
                fv = 0;
            }
            firsts[s] = fv;
        }

        // Device buffers
        let d_h = DeviceBuffer::from_slice(high_tm)?;
        let d_l = DeviceBuffer::from_slice(low_tm)?;
        let d_c = DeviceBuffer::from_slice(close_tm)?;
        let d_s = DeviceBuffer::from_slice(src_tm)?;
        let d_firsts = DeviceBuffer::from_slice(&firsts)?;

        let elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaLpcError::InvalidInput("output length overflow".into()))?;
        let mut d_f = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }?;
        let mut d_hi = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }?;
        let mut d_lo = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }?;

        let func = self
            .module
            .get_function("lpc_many_series_one_param_time_major_f32")
            .map_err(|_| CudaLpcError::MissingKernelSymbol {
                name: "lpc_many_series_one_param_time_major_f32",
            })?;
        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => 256,
            ManySeriesKernelPolicy::OneD { block_x } => block_x,
        };
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        if block_x == 0 || grid_x == 0 || grid_x > 65_535 {
            return Err(CudaLpcError::LaunchConfigTooLarge {
                gx: grid_x,
                gy: 1,
                gz: 1,
                bx: block_x,
                by: 1,
                bz: 1,
            });
        }
        unsafe {
            let stream = &self.stream;
            launch!(
                func<<<(grid_x, 1, 1), (block_x, 1, 1), 0, stream>>>(
                    d_h.as_device_ptr(), d_l.as_device_ptr(), d_c.as_device_ptr(), d_s.as_device_ptr(),
                    cols as i32, rows as i32,
                    fixed_period as i32, params.cycle_mult.unwrap_or(1.0) as f32, tr_mult,
                    0i32, params.max_cycle_limit.unwrap_or(60) as i32,
                    d_firsts.as_device_ptr(),
                    d_f.as_device_ptr(), d_hi.as_device_ptr(), d_lo.as_device_ptr()
                )
            )?;
        }
        self.stream
            .synchronize()?;
        let triplet = DeviceArrayF32Triplet {
            wt1: DeviceArrayF32 {
                buf: d_f,
                rows,
                cols,
            },
            wt2: DeviceArrayF32 {
                buf: d_hi,
                rows,
                cols,
            },
            hist: DeviceArrayF32 {
                buf: d_lo,
                rows,
                cols,
            },
        };
        unsafe {
            (*(self as *const _ as *mut CudaLpc)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") && !self.debug_many_logged {
            eprintln!(
                "[DEBUG] lpc many-series selected kernel: {:?}",
                self.last_many
            );
            unsafe {
                (*(self as *const _ as *mut CudaLpc)).debug_many_logged = true;
            }
        }
        Ok(triplet)
    }
}

// ---------------- Benches -----------------
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 100_000;

    struct LpcBatchState {
        cuda: CudaLpc,
        h: Vec<f32>,
        l: Vec<f32>,
        c: Vec<f32>,
        s: Vec<f32>,
        range: LpcBatchRange,
    }
    impl CudaBenchState for LpcBatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .lpc_batch_dev(&self.h, &self.l, &self.c, &self.s, &self.range);
        }
    }

    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let n = ONE_SERIES_LEN;
        let mut s = gen_series(n);
        let mut c = s.clone();
        let mut h = vec![f32::NAN; n];
        let mut l = vec![f32::NAN; n];
        for i in 0..n {
            if s[i].is_finite() {
                h[i] = s[i] + 0.5;
                l[i] = s[i] - 0.5;
            }
        }
        let range = LpcBatchRange {
            fixed_period: (20, 60, 10),
            cycle_mult: (1.0, 1.0, 0.0),
            tr_mult: (1.0, 1.0, 0.0),
            cutoff_type: "fixed".to_string(),
            max_cycle_limit: 60,
        };
        let cuda = CudaLpc::new(0).expect("cuda lpc");
        Box::new(LpcBatchState {
            cuda,
            h,
            l,
            c,
            s,
            range,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "lpc",
            "one_series_many_params",
            "lpc_cuda_batch_dev",
            "100k_fixed",
            prep_one_series_many_params,
        )
        .with_sample_size(15)]
    }
}
