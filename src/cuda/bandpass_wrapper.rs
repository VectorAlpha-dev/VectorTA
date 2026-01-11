//! CUDA wrapper for the Band-Pass indicator.
//!
//! Design mirrors ALMA and WTO wrappers:
//! - VRAM-first API returning device-resident arrays
//! - Non-blocking stream
//! - JIT PTX load with DetermineTargetFromContext and O2 fallback
//! - Simple policy/introspection hooks kept minimal (no tiling variants yet)
//! - VRAM estimation and conservative headroom; grid chunking where needed
//!
//! Math pattern: Recurrence/IIR. We reuse GPU High-Pass precompute across
//! parameter rows in batch by launching the existing `highpass_batch_f32`
//! kernel once for the unique `hp_period`s, then applying the band-pass
//! recurrence per row. For many-series×one-param we reuse the time-major
//! high-pass kernel and run a per-series recurrence.

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::{CudaHighpass, DeviceArrayF32};
use crate::indicators::bandpass::{BandPassBatchRange, BandPassParams};
use cust::context::{CacheConfig, Context};
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::collections::HashMap;
use std::env;
use std::ffi::c_void;
use std::fmt;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaBandpassError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("out of memory: required={required} free={free} headroom={headroom}")]
    OutOfMemory { required: usize, free: usize, headroom: usize },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("launch configuration too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("invalid range: start={start}, end={end}, step={step}")]
    InvalidRange { start: usize, end: usize, step: usize },
    #[error("invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("device mismatch: buffer on device {buf}, current device {current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
    NotImplemented,
}

pub struct DeviceArrayF32Quad {
    pub first: DeviceArrayF32,  
    pub second: DeviceArrayF32, 
    pub third: DeviceArrayF32,  
    pub fourth: DeviceArrayF32, 
}

impl DeviceArrayF32Quad {
    #[inline]
    pub fn rows(&self) -> usize {
        self.first.rows
    }
    #[inline]
    pub fn cols(&self) -> usize {
        self.first.cols
    }
}

pub struct CudaBandpassBatchResult {
    pub outputs: DeviceArrayF32Quad,
    pub combos: Vec<BandPassParams>,
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
pub struct CudaBandpassPolicy {
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

pub struct CudaBandpass {
    module: Module,
    stream: Stream,
    ctx: Arc<Context>,
    device_id: u32,
    policy: CudaBandpassPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaBandpass {
    pub fn new(device_id: usize) -> Result<Self, CudaBandpassError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Context::new(device)?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/bandpass_kernel.ptx"));
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
            ctx: Arc::new(context),
            device_id: device_id as u32,
            policy: CudaBandpassPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    #[inline]
    pub fn device_id(&self) -> u32 { self.device_id }
    #[inline]
    pub fn context_arc(&self) -> Arc<Context> { self.ctx.clone() }
    #[inline]
    pub fn stream_handle_usize(&self) -> usize { self.stream.as_inner() as usize }

    #[inline]
    pub fn stream(&self) -> &Stream { &self.stream }

    #[inline]
    fn maybe_log_batch_debug(&self) {
        use std::sync::atomic::{AtomicBool, Ordering};
        static ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                if !ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] bandpass batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaBandpass)).debug_batch_logged = true;
                }
            }
        }
    }

    #[inline]
    fn maybe_log_many_debug(&self) {
        use std::sync::atomic::{AtomicBool, Ordering};
        static ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_many_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                if !ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] bandpass many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaBandpass)).debug_many_logged = true;
                }
            }
        }
    }

    #[inline]
    fn validate_launch_dims(
        &self,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
    ) -> Result<(), CudaBandpassError> {
        let dev = Device::get_device(self.device_id).map_err(CudaBandpassError::Cuda)?;
        let max_gx = dev
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .map_err(CudaBandpassError::Cuda)? as u32;
        let max_gy = dev
            .get_attribute(DeviceAttribute::MaxGridDimY)
            .map_err(CudaBandpassError::Cuda)? as u32;
        let max_gz = dev
            .get_attribute(DeviceAttribute::MaxGridDimZ)
            .map_err(CudaBandpassError::Cuda)? as u32;
        let max_bx = dev
            .get_attribute(DeviceAttribute::MaxBlockDimX)
            .map_err(CudaBandpassError::Cuda)? as u32;
        let max_by = dev
            .get_attribute(DeviceAttribute::MaxBlockDimY)
            .map_err(CudaBandpassError::Cuda)? as u32;
        let max_bz = dev
            .get_attribute(DeviceAttribute::MaxBlockDimZ)
            .map_err(CudaBandpassError::Cuda)? as u32;
        let (gx, gy, gz) = grid;
        let (bx, by, bz) = block;
        if gx == 0 || gy == 0 || gz == 0 || bx == 0 || by == 0 || bz == 0 {
            return Err(CudaBandpassError::InvalidInput(
                "zero-sized grid or block".into(),
            ));
        }
        if gx > max_gx || gy > max_gy || gz > max_gz || bx > max_bx || by > max_by || bz > max_bz {
            return Err(CudaBandpassError::LaunchConfigTooLarge { gx, gy, gz, bx, by, bz });
        }
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
    fn device_mem_info() -> Option<(usize, usize)> { mem_get_info().ok() }
    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> bool {
        if !Self::mem_check_enabled() {
            return true;
        }
        if let Some((free, _total)) = Self::device_mem_info() {
            required_bytes
                .saturating_add(headroom_bytes)
                <= free
        } else {
            true
        }
    }
    #[inline]
    fn ensure_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaBandpassError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        if let Some((free, _total)) = Self::device_mem_info() {
            if Self::will_fit(required_bytes, headroom_bytes) {
                Ok(())
            } else {
                Err(CudaBandpassError::OutOfMemory { required: required_bytes, free, headroom: headroom_bytes })
            }
        } else {
            Ok(())
        }
    }

    fn expand_grid(range: &BandPassBatchRange) -> Result<Vec<BandPassParams>, CudaBandpassError> {
        fn axis_usize((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, CudaBandpassError> {
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
                loop {
                    vals.push(v);
                    if v == end {
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
                return Err(CudaBandpassError::InvalidRange { start, end, step });
            }
            Ok(vals)
        }
        fn axis_f64((start, end, step): (f64, f64, f64)) -> Result<Vec<f64>, CudaBandpassError> {
            if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
                return Ok(vec![start]);
            }
            let mut vals = Vec::new();
            if start <= end {
                let mut x = start;
                loop {
                    vals.push(x);
                    if x >= end {
                        break;
                    }
                    let next = x + step;
                    if !next.is_finite() || next == x {
                        break;
                    }
                    x = next;
                    if x > end + 1e-12 {
                        break;
                    }
                }
            } else {
                let mut x = start;
                loop {
                    vals.push(x);
                    if x <= end {
                        break;
                    }
                    let next = x - step.abs();
                    if !next.is_finite() || next == x {
                        break;
                    }
                    x = next;
                    if x < end - 1e-12 {
                        break;
                    }
                }
            }
            if vals.is_empty() {
                return Err(CudaBandpassError::InvalidRange {
                    start: start as usize,
                    end: end as usize,
                    step: step.abs() as usize,
                });
            }
            Ok(vals)
        }
        let periods = axis_usize(range.period)?;
        let bands = axis_f64(range.bandwidth)?;
        let cap = periods
            .len()
            .checked_mul(bands.len())
            .ok_or_else(|| CudaBandpassError::InvalidInput("parameter grid too large".into()))?;
        let mut v = Vec::with_capacity(cap);
        for &p in &periods {
            for &b in &bands {
                v.push(BandPassParams {
                    period: Some(p),
                    bandwidth: Some(b),
                });
            }
        }
        Ok(v)
    }

    fn prepare_batch(
        data_f32: &[f32],
        sweep: &BandPassBatchRange,
    ) -> Result<(Vec<BandPassParams>, usize, usize), CudaBandpassError> {
        if data_f32.is_empty() {
            return Err(CudaBandpassError::InvalidInput("empty data".into()));
        }
        let first_valid = data_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaBandpassError::InvalidInput("all values are NaN".into()))?;
        let len = data_f32.len();
        let combos = Self::expand_grid(sweep)?;
        if combos.is_empty() {
            return Err(CudaBandpassError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        for p in &combos {
            let period = p.period.unwrap_or(0);
            let bw = p.bandwidth.unwrap_or(0.0);
            if period < 2 || period > len {
                return Err(CudaBandpassError::InvalidInput(format!(
                    "invalid period {} for len {}",
                    period, len
                )));
            }
            if !(0.0..=1.0).contains(&bw) || !bw.is_finite() || bw == 0.0 {
                return Err(CudaBandpassError::InvalidInput(format!(
                    "invalid bandwidth {}",
                    bw
                )));
            }
            
            let hp_period = (4.0 * period as f64 / bw).round() as usize;
            if len - first_valid < hp_period {
                return Err(CudaBandpassError::InvalidInput(format!(
                    "not enough valid data: need >= {}, have {}",
                    hp_period,
                    len - first_valid
                )));
            }
        }
        Ok((combos, first_valid, len))
    }

    fn host_coeffs(period: usize, bw: f64) -> (f32, f32, i32, usize) {
        use std::f64::consts::PI;
        let beta = (2.0 * PI / period as f64).cos();
        let gamma = (2.0 * PI * bw / period as f64).cos();
        
        let alpha = 1.0 / gamma - ((1.0 / (gamma * gamma)) - 1.0).sqrt();
        let trig = ((period as f64 / bw) / 1.5).round() as i32;
        let hp_period = (4.0 * period as f64 / bw).round() as usize;
        (alpha as f32, beta as f32, trig, hp_period)
    }

    pub fn bandpass_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &BandPassBatchRange,
    ) -> Result<CudaBandpassBatchResult, CudaBandpassError> {
        let (combos, first_valid, len) = Self::prepare_batch(data_f32, sweep)?;
        let n = len;
        let rows = combos.len();

        
        let mut alphas = vec![0f32; rows];
        let mut betas = vec![0f32; rows];
        let mut trig = vec![0i32; rows];
        let mut hp_row_idx = vec![0i32; rows];
        let mut hp_map: HashMap<usize, usize> = HashMap::new();
        let mut hp_unique: Vec<i32> = Vec::new();
        for (i, p) in combos.iter().enumerate() {
            let period = p.period.unwrap();
            let bw = p.bandwidth.unwrap();
            let (a, b, t, hp_p) = Self::host_coeffs(period, bw);
            alphas[i] = a;
            betas[i] = b;
            trig[i] = t;
            let idx = *hp_map.entry(hp_p).or_insert_with(|| {
                hp_unique.push(hp_p as i32);
                hp_unique.len() - 1
            });
            hp_row_idx[i] = idx as i32;
        }

        
        let sz_f32 = std::mem::size_of::<f32>() as u128;
        let sz_i32 = std::mem::size_of::<i32>() as u128;
        let prices_bytes = (n as u128)
            .checked_mul(sz_f32)
            .ok_or_else(|| CudaBandpassError::InvalidInput("size overflow for prices_bytes".into()))?;
        let hp_bytes = (hp_unique.len() as u128)
            .checked_mul(n as u128)
            .and_then(|v| v.checked_mul(sz_f32))
            .ok_or_else(|| CudaBandpassError::InvalidInput("size overflow for hp_bytes".into()))?;
        let params_bytes = (rows as u128)
            .checked_mul((2u128 * sz_f32 + 2u128 * sz_i32))
            .ok_or_else(|| CudaBandpassError::InvalidInput("size overflow for params_bytes".into()))?;
        let outs_bytes = (4u128)
            .checked_mul(rows as u128)
            .and_then(|v| v.checked_mul(n as u128))
            .and_then(|v| v.checked_mul(sz_f32))
            .ok_or_else(|| CudaBandpassError::InvalidInput("size overflow for outs_bytes".into()))?;
        let required_u128 = prices_bytes
            .checked_add(hp_bytes)
            .and_then(|v| v.checked_add(params_bytes))
            .and_then(|v| v.checked_add(outs_bytes))
            .ok_or_else(|| CudaBandpassError::InvalidInput("size overflow for required bytes".into()))?;
        let required = if required_u128 > usize::MAX as u128 {
            return Err(CudaBandpassError::InvalidInput(
                "required VRAM size overflow".into(),
            ));
        } else {
            required_u128 as usize
        };
        let headroom = 64 * 1024 * 1024;
        Self::ensure_fit(required, headroom)?;

        
        
        
        let d_prices = DeviceBuffer::from_slice(data_f32)?;
        
        
        let d_hp_periods = DeviceBuffer::from_slice(&hp_unique)?;
        let hp_len = hp_unique
            .len()
            .checked_mul(n)
            .ok_or_else(|| CudaBandpassError::InvalidInput("hp buffer length overflow".into()))?;
        
        
        let mut d_hp: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(hp_len)? };
        let cuda_hp = CudaHighpass::new(0).map_err(|e| CudaBandpassError::InvalidInput(e.to_string()))?;
        cuda_hp
            .highpass_batch_device(
                &d_prices,
                first_valid as i32,
                &d_hp_periods,
                n as i32,
                hp_unique.len() as i32,
                &mut d_hp,
            ).map_err(|e| CudaBandpassError::InvalidInput(e.to_string()))?;
        
        cuda_hp
            .synchronize()
            .map_err(|e| CudaBandpassError::InvalidInput(e.to_string()))?;

        
        let d_hp_idx = DeviceBuffer::from_slice(&hp_row_idx)?;
        let d_alpha = DeviceBuffer::from_slice(&alphas)?;
        let d_beta = DeviceBuffer::from_slice(&betas)?;
        let d_trig = DeviceBuffer::from_slice(&trig)?;

        
        let total_out = rows
            .checked_mul(n)
            .ok_or_else(|| CudaBandpassError::InvalidInput("output buffer length overflow".into()))?;
        let mut d_bp: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(total_out, &self.stream)? };
        let mut d_bpn: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(total_out, &self.stream)? };
        let mut d_sig: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(total_out, &self.stream)? };
        let mut d_trg: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(total_out, &self.stream)? };

        
        let mut func = self
            .module
            .get_function("bandpass_batch_from_hp_f32")
            .map_err(|_| CudaBandpassError::MissingKernelSymbol { name: "bandpass_batch_from_hp_f32" })?;
        
        let _ = func.set_cache_config(CacheConfig::PreferL1);
        
        let (suggested, _min_grid) =
            func.suggested_launch_configuration(0, BlockSize::xyz(0, 0, 0))?;
        
        let bx = match self.policy.batch {
            BatchKernelPolicy::Auto => suggested.clamp(128, 256),
            BatchKernelPolicy::Plain { block_x } => block_x.clamp(32, 256),
        };
        unsafe {
            (*(self as *const _ as *mut CudaBandpass)).last_batch =
                Some(BatchKernelSelected::Plain { block_x: bx });
        }
        let grid_x = ((rows as u32) + bx - 1) / bx;
        let grid_tuple = (grid_x.max(1), 1, 1);
        let block_tuple = (bx, 1, 1);
        self.validate_launch_dims(grid_tuple, block_tuple)?;
        let block: BlockSize = block_tuple.into();
        let grid: GridSize = grid_tuple.into();

        unsafe {
            let mut hp_ptr = d_hp.as_device_ptr().as_raw();
            let mut hp_rows_i = hp_unique.len() as i32;
            let mut len_i = n as i32;
            let mut hp_idx_ptr = d_hp_idx.as_device_ptr().as_raw();
            let mut alpha_ptr = d_alpha.as_device_ptr().as_raw();
            let mut beta_ptr = d_beta.as_device_ptr().as_raw();
            let mut trig_ptr = d_trig.as_device_ptr().as_raw();
            let mut combos_i = rows as i32;
            let mut bp_ptr = d_bp.as_device_ptr().as_raw();
            let mut bpn_ptr = d_bpn.as_device_ptr().as_raw();
            let mut sig_ptr = d_sig.as_device_ptr().as_raw();
            let mut trg_ptr = d_trg.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut hp_ptr as *mut _ as *mut c_void,
                &mut hp_rows_i as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut hp_idx_ptr as *mut _ as *mut c_void,
                &mut alpha_ptr as *mut _ as *mut c_void,
                &mut beta_ptr as *mut _ as *mut c_void,
                &mut trig_ptr as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut bp_ptr as *mut _ as *mut c_void,
                &mut bpn_ptr as *mut _ as *mut c_void,
                &mut sig_ptr as *mut _ as *mut c_void,
                &mut trg_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        self.maybe_log_batch_debug();

        Ok(CudaBandpassBatchResult {
            outputs: DeviceArrayF32Quad {
                first: DeviceArrayF32 {
                    buf: d_bp,
                    rows,
                    cols: n,
                },
                second: DeviceArrayF32 {
                    buf: d_bpn,
                    rows,
                    cols: n,
                },
                third: DeviceArrayF32 {
                    buf: d_sig,
                    rows,
                    cols: n,
                },
                fourth: DeviceArrayF32 {
                    buf: d_trg,
                    rows,
                    cols: n,
                },
            },
            combos,
        })
    }

    pub fn bandpass_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &BandPassParams,
    ) -> Result<DeviceArrayF32Quad, CudaBandpassError> {
        if cols == 0 || rows == 0 {
            return Err(CudaBandpassError::InvalidInput("invalid cols/rows".into()));
        }
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaBandpassError::InvalidInput("cols*rows overflow".into()))?;
        if data_tm_f32.len() != expected {
            return Err(CudaBandpassError::InvalidInput(format!(
                "time-major input length mismatch (expected {}, got {})",
                expected,
                data_tm_f32.len()
            )));
        }
        let period = params.period.unwrap_or(0);
        let bw = params.bandwidth.unwrap_or(0.0);
        if period < 2 || !(0.0..=1.0).contains(&bw) || bw == 0.0 {
            return Err(CudaBandpassError::InvalidInput("invalid params".into()));
        }
        let (_a, _b, _trig, hp_period) = Self::host_coeffs(period, bw);

        
        let cuda_hp = CudaHighpass::new(0).map_err(|e| CudaBandpassError::InvalidInput(e.to_string()))?;
        let hp_dev = cuda_hp
            .highpass_many_series_one_param_time_major_dev(
                data_tm_f32,
                cols,
                rows,
                &crate::indicators::moving_averages::highpass::HighPassParams {
                    period: Some(hp_period),
                },
            ).map_err(|e| CudaBandpassError::InvalidInput(e.to_string()))?;

        
        let (alpha, beta, trig, _hp) = Self::host_coeffs(period, bw);
        

        
        let total = expected;
        let mut d_bp: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(total, &self.stream)? };
        let mut d_bpn: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(total, &self.stream)? };
        let mut d_sig: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(total, &self.stream)? };
        let mut d_trg_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(total, &self.stream)? };

        
        let mut func = self
            .module
            .get_function("bandpass_many_series_one_param_time_major_from_hp_f32")
            .map_err(|_| CudaBandpassError::MissingKernelSymbol { name: "bandpass_many_series_one_param_time_major_from_hp_f32" })?;
        let _ = func.set_cache_config(CacheConfig::PreferL1);
        let (suggested, _mg) =
            func.suggested_launch_configuration(0, BlockSize::xyz(0, 0, 0))?;
        
        let bx = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => suggested.clamp(128, 256),
            ManySeriesKernelPolicy::OneD { block_x } => block_x.clamp(32, 256),
        };
        unsafe {
            (*(self as *const _ as *mut CudaBandpass)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x: bx });
        }
        let grid_x = ((cols as u32) + bx - 1) / bx;
        let grid_tuple = (grid_x.max(1), 1, 1);
        let block_tuple = (bx, 1, 1);
        self.validate_launch_dims(grid_tuple, block_tuple)?;
        let grid: GridSize = grid_tuple.into();
        let block: BlockSize = block_tuple.into();

        unsafe {
            let mut hp_ptr = hp_dev.buf.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut alpha_f = alpha;
            let mut beta_f = beta;
            let mut trig_i = trig;
            let mut bp_ptr = d_bp.as_device_ptr().as_raw();
            let mut bpn_ptr = d_bpn.as_device_ptr().as_raw();
            let mut sig_ptr = d_sig.as_device_ptr().as_raw();
            let mut trg_ptr = d_trg_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut hp_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut alpha_f as *mut _ as *mut c_void,
                &mut beta_f as *mut _ as *mut c_void,
                &mut trig_i as *mut _ as *mut c_void,
                &mut bp_ptr as *mut _ as *mut c_void,
                &mut bpn_ptr as *mut _ as *mut c_void,
                &mut sig_ptr as *mut _ as *mut c_void,
                &mut trg_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        self.maybe_log_many_debug();

        Ok(DeviceArrayF32Quad {
            first: DeviceArrayF32 {
                buf: d_bp,
                rows,
                cols,
            },
            second: DeviceArrayF32 {
                buf: d_bpn,
                rows,
                cols,
            },
            third: DeviceArrayF32 {
                buf: d_sig,
                rows,
                cols,
            },
            fourth: DeviceArrayF32 {
                buf: d_trg_out,
                rows,
                cols,
            },
        })
    }
}


pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const MANY_SERIES_COLS: usize = 512;
    const MANY_SERIES_ROWS: usize = 8_192;

    fn bytes_one_series(rows: usize, len: usize) -> usize {
        
        let in_b = len * std::mem::size_of::<f32>();
        let hp_b = rows * len * std::mem::size_of::<f32>();
        let out_b = 4 * rows * len * std::mem::size_of::<f32>();
        in_b + hp_b + out_b + 32 * 1024 * 1024
    }

    struct BPBatchState {
        cuda: CudaBandpass,
        hp_rows: usize,
        len: usize,
        n_combos: usize,
        d_hp: DeviceBuffer<f32>,
        d_hp_idx: DeviceBuffer<i32>,
        d_alpha: DeviceBuffer<f32>,
        d_beta: DeviceBuffer<f32>,
        d_trig: DeviceBuffer<i32>,
        d_bp: DeviceBuffer<f32>,
        d_bpn: DeviceBuffer<f32>,
        d_sig: DeviceBuffer<f32>,
        d_trg: DeviceBuffer<f32>,
    }
    impl CudaBenchState for BPBatchState {
        fn launch(&mut self) {
            let mut func = self
                .cuda
                .module
                .get_function("bandpass_batch_from_hp_f32")
                .expect("bandpass kernel");
            let _ = func.set_cache_config(CacheConfig::PreferL1);

            let bx = 128u32;
            let gx = ((self.n_combos as u32) + bx - 1) / bx;
            let grid: GridSize = (gx.max(1), 1, 1).into();
            let block: BlockSize = (bx, 1, 1).into();

            unsafe {
                let mut hp_ptr = self.d_hp.as_device_ptr().as_raw();
                let mut hp_rows_i = self.hp_rows as i32;
                let mut len_i = self.len as i32;
                let mut hp_idx_ptr = self.d_hp_idx.as_device_ptr().as_raw();
                let mut alpha_ptr = self.d_alpha.as_device_ptr().as_raw();
                let mut beta_ptr = self.d_beta.as_device_ptr().as_raw();
                let mut trig_ptr = self.d_trig.as_device_ptr().as_raw();
                let mut combos_i = self.n_combos as i32;
                let mut bp_ptr = self.d_bp.as_device_ptr().as_raw();
                let mut bpn_ptr = self.d_bpn.as_device_ptr().as_raw();
                let mut sig_ptr = self.d_sig.as_device_ptr().as_raw();
                let mut trg_ptr = self.d_trg.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut hp_ptr as *mut _ as *mut c_void,
                    &mut hp_rows_i as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut hp_idx_ptr as *mut _ as *mut c_void,
                    &mut alpha_ptr as *mut _ as *mut c_void,
                    &mut beta_ptr as *mut _ as *mut c_void,
                    &mut trig_ptr as *mut _ as *mut c_void,
                    &mut combos_i as *mut _ as *mut c_void,
                    &mut bp_ptr as *mut _ as *mut c_void,
                    &mut bpn_ptr as *mut _ as *mut c_void,
                    &mut sig_ptr as *mut _ as *mut c_void,
                    &mut trg_ptr as *mut _ as *mut c_void,
                ];
                self.cuda
                    .stream
                    .launch(&func, grid, block, 0, args)
                    .expect("bandpass launch");
            }
            self.cuda.stream.synchronize().expect("bandpass sync");
        }
    }

    fn prep_one_series() -> Box<dyn CudaBenchState> {
        let cuda = CudaBandpass::new(0).expect("cuda bandpass");
        let prices = gen_series(ONE_SERIES_LEN);
        
        let sweep = BandPassBatchRange {
            period: (16, 22, 2),
            bandwidth: (0.2, 0.4, 0.1),
        };

        let (combos, first_valid, len) =
            CudaBandpass::prepare_batch(&prices, &sweep).expect("bandpass prep");
        let rows = combos.len();

        let mut alphas = vec![0f32; rows];
        let mut betas = vec![0f32; rows];
        let mut trig = vec![0i32; rows];
        let mut hp_row_idx = vec![0i32; rows];
        let mut hp_map: HashMap<usize, usize> = HashMap::new();
        let mut hp_unique: Vec<i32> = Vec::new();
        for (i, p) in combos.iter().enumerate() {
            let period = p.period.unwrap();
            let bw = p.bandwidth.unwrap();
            let (a, b, t, hp_p) = CudaBandpass::host_coeffs(period, bw);
            alphas[i] = a;
            betas[i] = b;
            trig[i] = t;
            let idx = *hp_map.entry(hp_p).or_insert_with(|| {
                hp_unique.push(hp_p as i32);
                hp_unique.len() - 1
            });
            hp_row_idx[i] = idx as i32;
        }

        
        let d_prices = DeviceBuffer::from_slice(&prices).expect("d_prices");
        let d_hp_periods = DeviceBuffer::from_slice(&hp_unique).expect("d_hp_periods");
        let hp_rows = hp_unique.len();
        let hp_len = hp_rows * len;
        let mut d_hp: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(hp_len) }.expect("d_hp");
        let cuda_hp = CudaHighpass::new(0).expect("cuda highpass");
        cuda_hp
            .highpass_batch_device(
                &d_prices,
                first_valid as i32,
                &d_hp_periods,
                len as i32,
                hp_rows as i32,
                &mut d_hp,
            )
            .expect("highpass batch device");
        cuda_hp.synchronize().expect("highpass sync");

        let d_hp_idx = DeviceBuffer::from_slice(&hp_row_idx).expect("d_hp_idx");
        let d_alpha = DeviceBuffer::from_slice(&alphas).expect("d_alpha");
        let d_beta = DeviceBuffer::from_slice(&betas).expect("d_beta");
        let d_trig = DeviceBuffer::from_slice(&trig).expect("d_trig");

        let total_out = rows * len;
        let d_bp = unsafe { DeviceBuffer::<f32>::uninitialized(total_out) }.expect("d_bp");
        let d_bpn = unsafe { DeviceBuffer::<f32>::uninitialized(total_out) }.expect("d_bpn");
        let d_sig = unsafe { DeviceBuffer::<f32>::uninitialized(total_out) }.expect("d_sig");
        let d_trg = unsafe { DeviceBuffer::<f32>::uninitialized(total_out) }.expect("d_trg");

        Box::new(BPBatchState {
            cuda,
            hp_rows,
            len,
            n_combos: rows,
            d_hp,
            d_hp_idx,
            d_alpha,
            d_beta,
            d_trig,
            d_bp,
            d_bpn,
            d_sig,
            d_trg,
        })
    }

    struct BPManyState {
        cuda: CudaBandpass,
        d_hp: DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        alpha: f32,
        beta: f32,
        trig: i32,
        grid: GridSize,
        block: BlockSize,
        d_bp: DeviceBuffer<f32>,
        d_bpn: DeviceBuffer<f32>,
        d_sig: DeviceBuffer<f32>,
        d_trg: DeviceBuffer<f32>,
    }
    impl CudaBenchState for BPManyState {
        fn launch(&mut self) {
            let mut func = self
                .cuda
                .module
                .get_function("bandpass_many_series_one_param_time_major_from_hp_f32")
                .expect("bandpass many-series kernel");
            let _ = func.set_cache_config(CacheConfig::PreferL1);

            unsafe {
                let mut hp_ptr = self.d_hp.as_device_ptr().as_raw();
                let mut cols_i = self.cols as i32;
                let mut rows_i = self.rows as i32;
                let mut alpha_f = self.alpha;
                let mut beta_f = self.beta;
                let mut trig_i = self.trig;
                let mut bp_ptr = self.d_bp.as_device_ptr().as_raw();
                let mut bpn_ptr = self.d_bpn.as_device_ptr().as_raw();
                let mut sig_ptr = self.d_sig.as_device_ptr().as_raw();
                let mut trg_ptr = self.d_trg.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut hp_ptr as *mut _ as *mut c_void,
                    &mut cols_i as *mut _ as *mut c_void,
                    &mut rows_i as *mut _ as *mut c_void,
                    &mut alpha_f as *mut _ as *mut c_void,
                    &mut beta_f as *mut _ as *mut c_void,
                    &mut trig_i as *mut _ as *mut c_void,
                    &mut bp_ptr as *mut _ as *mut c_void,
                    &mut bpn_ptr as *mut _ as *mut c_void,
                    &mut sig_ptr as *mut _ as *mut c_void,
                    &mut trg_ptr as *mut _ as *mut c_void,
                ];
                self.cuda
                    .stream
                    .launch(&func, self.grid, self.block, 0, args)
                    .expect("bandpass many-series launch");
            }
            self.cuda.stream.synchronize().expect("bandpass many-series sync");
        }
    }

    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaBandpass::new(0).expect("cuda bandpass");
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_ROWS;
        let data_tm = gen_time_major_prices(cols, rows);
        let params = BandPassParams {
            period: Some(20),
            bandwidth: Some(0.3),
        };
        let period = params.period.unwrap_or(0);
        let bw = params.bandwidth.unwrap_or(0.0);
        let (_a, _b, _trig, hp_period) = CudaBandpass::host_coeffs(period, bw);

        
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv: Option<i32> = None;
            for t in 0..rows {
                let idx = t * cols + s;
                if !data_tm[idx].is_nan() {
                    fv = Some(t as i32);
                    break;
                }
            }
            first_valids[s] = fv.unwrap_or(-1);
        }
        let cuda_hp = CudaHighpass::new(0).expect("cuda highpass");
        let d_prices = DeviceBuffer::from_slice(&data_tm).expect("d_prices_tm");
        let d_first_valids = DeviceBuffer::from_slice(&first_valids).expect("d_first_valids");
        let mut d_hp: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(cols * rows) }.expect("d_hp");
        cuda_hp
            .highpass_many_series_one_param_time_major_device(
                &d_prices,
                &d_first_valids,
                hp_period as i32,
                cols as i32,
                rows as i32,
                &mut d_hp,
            )
            .expect("highpass_many_series_one_param_time_major_device");
        cuda_hp.synchronize().expect("highpass sync");

        let (alpha, beta, trig, _hp) = CudaBandpass::host_coeffs(period, bw);

        let total = cols * rows;
        let d_bp: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total) }.expect("d_bp");
        let d_bpn: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total) }.expect("d_bpn");
        let d_sig: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total) }.expect("d_sig");
        let d_trg: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total) }.expect("d_trg");

        let mut func = cuda
            .module
            .get_function("bandpass_many_series_one_param_time_major_from_hp_f32")
            .expect("bandpass_many_series_one_param_time_major_from_hp_f32");
        let (suggested, _mg) = func
            .suggested_launch_configuration(0, BlockSize::xyz(0, 0, 0))
            .unwrap_or((256, 0));
        let bx = match cuda.policy.many_series {
            ManySeriesKernelPolicy::Auto => suggested.clamp(128, 256),
            ManySeriesKernelPolicy::OneD { block_x } => block_x.clamp(32, 256),
        };
        let grid_x = ((cols as u32) + bx - 1) / bx;
        let grid_tuple = (grid_x.max(1), 1, 1);
        let block_tuple = (bx, 1, 1);
        cuda.validate_launch_dims(grid_tuple, block_tuple)
            .expect("bandpass many-series launch dims");
        let grid: GridSize = grid_tuple.into();
        let block: BlockSize = block_tuple.into();
        cuda.stream.synchronize().expect("bandpass prep sync");
        Box::new(BPManyState {
            cuda,
            d_hp,
            cols,
            rows,
            alpha,
            beta,
            trig,
            grid,
            block,
            d_bp,
            d_bpn,
            d_sig,
            d_trg,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "bandpass",
                "batch_one_series",
                "bandpass_cuda_batch",
                "1m",
                prep_one_series,
            )
            .with_mem_required(bytes_one_series(12, ONE_SERIES_LEN)),
            CudaBenchScenario::new(
                "bandpass",
                "many_series_one_param",
                "bandpass_cuda_many_series_one_param",
                "tm",
                prep_many_series_one_param,
            )
            .with_mem_required({
                let elems = MANY_SERIES_COLS * MANY_SERIES_ROWS;
                
                (6 * elems * std::mem::size_of::<f32>()) + (MANY_SERIES_COLS * std::mem::size_of::<i32>()) + 64 * 1024 * 1024
            }),
        ]
    }
}
