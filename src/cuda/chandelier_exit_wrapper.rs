//! CUDA scaffolding for the Chandelier Exit (CE) indicator.
//!
//! Math category: recurrence/time-scan per parameter (ATR + rolling
//! extremums + trailing logic). We parallelize across rows (batch) or
//! across series (many-series) and scan time within each thread to
//! preserve scalar semantics. No sizable row-shared precompute beyond
//! ATR; use_close is static across a batch sweep.

#![cfg(feature = "cuda")]

use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;
use std::sync::Arc;

use crate::cuda::moving_averages::alma_wrapper::DeviceArrayF32;
use crate::indicators::chandelier_exit::{CeBatchRange, ChandelierExitParams};

#[derive(thiserror::Error, Debug)]
pub enum CudaCeError {
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

pub struct CudaChandelierExit {
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

impl CudaChandelierExit {
    pub fn new(device_id: usize) -> Result<Self, CudaCeError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/chandelier_exit_kernel.ptx"));
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
            context,
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
    pub fn stream(&self) -> &Stream { &self.stream }

    #[inline]
    fn mem_check_enabled() -> bool {
        match std::env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
            Err(_) => true,
        }
    }
    #[inline]
    fn ensure_fit(required_bytes: usize, headroom: usize) -> Result<(), CudaCeError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        if let Ok((free, _)) = mem_get_info() {
            if required_bytes.saturating_add(headroom) <= free {
                Ok(())
            } else {
                Err(CudaCeError::OutOfMemory { required: required_bytes, free, headroom })
            }
        } else {
            Ok(())
        }
    }

    fn expand_grid(range: &CeBatchRange) -> Result<Vec<ChandelierExitParams>, CudaCeError> {
        
        fn axis_usize(t: (usize, usize, usize)) -> Result<Vec<usize>, CudaCeError> {
            if t.2 == 0 || t.0 == t.1 { return Ok(vec![t.0]); }
            let (start, end, step) = (t.0, t.1, t.2);
            let mut v = Vec::new();
            if start <= end {
                let mut x = start;
                while x <= end { v.push(x); x = match x.checked_add(step) { Some(nx)=>nx, None=> return Err(CudaCeError::InvalidInput("period range overflow".into())) }; }
            } else {
                let mut x = start;
                while x >= end { v.push(x); if x < step { break; } x -= step; }
            }
            if v.is_empty() { return Err(CudaCeError::InvalidInput("invalid period range".into())); }
            Ok(v)
        }
        fn axis_f64(t: (f64, f64, f64)) -> Result<Vec<f64>, CudaCeError> {
            if t.2.abs() < 1e-12 || (t.0 - t.1).abs() < 1e-12 { return Ok(vec![t.0]); }
            let (start, end, step) = (t.0, t.1, t.2);
            let s = if step > 0.0 { if start <= end { step } else { -step } } else { step };
            let mut v = Vec::new();
            let mut x = start; let mut it=0usize;
            while it < 1_000_000 { if (s>0.0 && x> end+1e-12) || (s<0.0 && x< end-1e-12) { break; } v.push(x); x += s; it+=1; }
            if v.is_empty() { return Err(CudaCeError::InvalidInput("invalid mult range".into())); }
            Ok(v)
        }
        let periods = axis_usize(range.period)?;
        let mults = axis_f64(range.mult)?;
        let use_close = range.use_close.0;
        let cap = periods.len().checked_mul(mults.len()).ok_or_else(|| CudaCeError::InvalidInput("range size overflow".into()))?;
        let mut out = Vec::with_capacity(cap);
        for &p in &periods {
            for &m in &mults {
                out.push(ChandelierExitParams {
                    period: Some(p),
                    mult: Some(m),
                    use_close: Some(use_close),
                });
            }
        }
        Ok(out)
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
                    eprintln!("[DEBUG] CE batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaChandelierExit)).debug_batch_logged = true;
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
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                let per = std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] CE many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaChandelierExit)).debug_many_logged = true;
                }
            }
        }
    }

    
    pub fn set_batch_policy(&mut self, p: BatchKernelPolicy) {
        self.policy_batch = p;
    }
    pub fn set_many_series_policy(&mut self, p: ManySeriesKernelPolicy) {
        self.policy_many = p;
    }
    pub fn batch_policy(&self) -> BatchKernelPolicy {
        self.policy_batch
    }
    pub fn many_series_policy(&self) -> ManySeriesKernelPolicy {
        self.policy_many
    }

    fn first_valid(
        use_close: bool,
        high: &[f32],
        low: &[f32],
        close: &[f32],
    ) -> Result<usize, CudaCeError> {
        let len = close.len().min(high.len()).min(low.len());
        if len == 0 {
            return Err(CudaCeError::InvalidInput("empty input".into()));
        }
        let fv = if use_close {
            (0..len).find(|&i| !close[i].is_nan())
        } else {
            (0..len).find(|&i| !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan())
        };
        fv.ok_or_else(|| CudaCeError::InvalidInput("all values are NaN".into()))
    }

    fn launch_batch(
        &self,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_close: &DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        d_periods: &DeviceBuffer<i32>,
        d_mults: &DeviceBuffer<f32>,
        n_combos: usize,
        use_close: bool,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaCeError> {
        let func = self
            .module
            .get_function("chandelier_exit_batch_f32")
            .map_err(|_| CudaCeError::MissingKernelSymbol { name: "chandelier_exit_batch_f32" })?;
        
        let block_x_env = std::env::var("CE_BLOCK_X")
            .ok()
            .and_then(|v| v.parse::<u32>().ok());
        let block_x = block_x_env
            .or_else(|| match self.policy_batch {
                BatchKernelPolicy::Plain { block_x } => Some(block_x),
                _ => None,
            })
            .unwrap_or(256)
            .max(32);
        let grid_x = ((n_combos as u32) + block_x - 1) / block_x;
        if grid_x > 65_535 || block_x > 1024 {
            return Err(CudaCeError::LaunchConfigTooLarge { gx: grid_x, gy: 1, gz: 1, bx: block_x, by: 1, bz: 1 });
        }
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe {
            let mut h = d_high.as_device_ptr().as_raw();
            let mut l = d_low.as_device_ptr().as_raw();
            let mut c = d_close.as_device_ptr().as_raw();
            let mut n = len as i32;
            let mut fv = first_valid as i32;
            let mut p = d_periods.as_device_ptr().as_raw();
            let mut m = d_mults.as_device_ptr().as_raw();
            let mut r = n_combos as i32;
            let mut u = if use_close { 1i32 } else { 0i32 };
            let mut o = d_out.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 10] = [
                &mut h as *mut _ as *mut c_void,
                &mut l as *mut _ as *mut c_void,
                &mut c as *mut _ as *mut c_void,
                &mut n as *mut _ as *mut c_void,
                &mut fv as *mut _ as *mut c_void,
                &mut p as *mut _ as *mut c_void,
                &mut m as *mut _ as *mut c_void,
                &mut r as *mut _ as *mut c_void,
                &mut u as *mut _ as *mut c_void,
                &mut o as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, &mut args)?;
            
            (*(self as *const _ as *mut CudaChandelierExit)).last_batch =
                Some(BatchKernelSelected::Plain { block_x });
        }
        self.maybe_log_batch_debug();
        Ok(())
    }

    pub fn chandelier_exit_batch_dev(
        &self,
        high: &[f32],
        low: &[f32],
        close: &[f32],
        sweep: &CeBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<ChandelierExitParams>), CudaCeError> {
        let len = high.len().min(low.len()).min(close.len());
        if len == 0 {
            return Err(CudaCeError::InvalidInput("empty input".into()));
        }
        let combos = Self::expand_grid(sweep)?;
        let use_close = sweep.use_close.0;
        let first_valid = Self::first_valid(use_close, high, low, close)?;
        
        for prm in &combos {
            let p = prm.period.unwrap_or(22);
            if p == 0 {
                return Err(CudaCeError::InvalidInput("period must be >=1".into()));
            }
            if len - first_valid < p {
                return Err(CudaCeError::InvalidInput(format!(
                    "not enough valid data (need >= {}, have {})",
                    p,
                    len - first_valid
                )));
            }
        }

        
        let rows = combos.len();
        let in_bytes = (3usize)
            .checked_mul(len)
            .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| CudaCeError::InvalidInput("size overflow".into()))?;
        let param_bytes = rows
            .checked_mul(std::mem::size_of::<i32>() + std::mem::size_of::<f32>())
            .ok_or_else(|| CudaCeError::InvalidInput("size overflow".into()))?;
        let out_bytes = 2usize
            .checked_mul(rows)
            .and_then(|x| x.checked_mul(len))
            .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| CudaCeError::InvalidInput("size overflow".into()))?;
        let req = in_bytes
            .checked_add(param_bytes)
            .and_then(|x| x.checked_add(out_bytes))
            .ok_or_else(|| CudaCeError::InvalidInput("size overflow".into()))?;
        let headroom = std::env::var("CUDA_MEM_HEADROOM")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(64 * 1024 * 1024);
        Self::ensure_fit(req, headroom)?;

        
        let d_high = unsafe { DeviceBuffer::from_slice_async(&high[..len], &self.stream) }?;
        let d_low = unsafe { DeviceBuffer::from_slice_async(&low[..len], &self.stream) }?;
        let d_close = unsafe { DeviceBuffer::from_slice_async(&close[..len], &self.stream) }?;
        let periods_host: Vec<i32> = combos.iter().map(|c| c.period.unwrap() as i32).collect();
        let mults_host: Vec<f32> = combos.iter().map(|c| c.mult.unwrap() as f32).collect();
        let d_periods = unsafe { DeviceBuffer::from_slice_async(&periods_host, &self.stream) }?;
        let d_mults = unsafe { DeviceBuffer::from_slice_async(&mults_host, &self.stream) }?;
        let elems_out = rows
            .checked_mul(len)
            .and_then(|x| x.checked_mul(2))
            .ok_or_else(|| CudaCeError::InvalidInput("size overflow".into()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(elems_out, &self.stream) }?;

        self.launch_batch(
            &d_high,
            &d_low,
            &d_close,
            len,
            first_valid,
            &d_periods,
            &d_mults,
            rows,
            use_close,
            &mut d_out,
        )?;
        self.stream.synchronize()?;
        Ok((
            DeviceArrayF32 {
                buf: d_out,
                rows: 2 * rows,
                cols: len,
            },
            combos,
        ))
    }

    fn launch_many_series(
        &self,
        d_high_tm: &DeviceBuffer<f32>,
        d_low_tm: &DeviceBuffer<f32>,
        d_close_tm: &DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        period: usize,
        mult: f32,
        d_first_valids: &DeviceBuffer<i32>,
        use_close: bool,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaCeError> {
        let func = self
            .module
            .get_function("chandelier_exit_many_series_one_param_time_major_f32")
            .map_err(|_| CudaCeError::MissingKernelSymbol { name: "chandelier_exit_many_series_one_param_time_major_f32" })?;
        let block_x_env = std::env::var("CE_MANY_BLOCK_X")
            .ok()
            .and_then(|v| v.parse::<u32>().ok());
        let block_x = block_x_env
            .or_else(|| match self.policy_many {
                ManySeriesKernelPolicy::OneD { block_x } => Some(block_x),
                _ => None,
            })
            .unwrap_or(256)
            .max(32);
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        if grid_x > 65_535 || block_x > 1024 {
            return Err(CudaCeError::LaunchConfigTooLarge { gx: grid_x, gy: 1, gz: 1, bx: block_x, by: 1, bz: 1 });
        }
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe {
            let mut h = d_high_tm.as_device_ptr().as_raw();
            let mut l = d_low_tm.as_device_ptr().as_raw();
            let mut c = d_close_tm.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut p = period as i32;
            let mut m = mult as f32;
            let mut fv = d_first_valids.as_device_ptr().as_raw();
            let mut u = if use_close { 1i32 } else { 0i32 };
            let mut o = d_out_tm.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 10] = [
                &mut h as *mut _ as *mut c_void,
                &mut l as *mut _ as *mut c_void,
                &mut c as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut p as *mut _ as *mut c_void,
                &mut m as *mut _ as *mut c_void,
                &mut fv as *mut _ as *mut c_void,
                &mut u as *mut _ as *mut c_void,
                &mut o as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, &mut args)?;
            
            (*(self as *const _ as *mut CudaChandelierExit)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }
        self.maybe_log_many_debug();
        Ok(())
    }

    pub fn chandelier_exit_many_series_one_param_time_major_dev(
        &self,
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
        mult: f32,
        use_close: bool,
    ) -> Result<DeviceArrayF32, CudaCeError> {
        if cols == 0 || rows == 0 { return Err(CudaCeError::InvalidInput("empty matrix".into())); }
        if high_tm.len() != cols * rows
            || low_tm.len() != cols * rows
            || close_tm.len() != cols * rows
        {
            return Err(CudaCeError::InvalidInput("matrix shape mismatch".into()));
        }
        if period == 0 { return Err(CudaCeError::InvalidInput("period must be >=1".into())); }

        
        let mut first_valids = vec![rows as i32; cols];
        for s in 0..cols {
            for t in 0..rows {
                let idx = t * cols + s;
                let ok = if use_close {
                    !close_tm[idx].is_nan()
                } else {
                    !high_tm[idx].is_nan() && !low_tm[idx].is_nan() && !close_tm[idx].is_nan()
                };
                if ok {
                    first_valids[s] = t as i32;
                    break;
                }
                if ok {
                    first_valids[s] = t as i32;
                    break;
                }
            }
            if (rows as i32 - first_valids[s]) < period as i32 {
                return Err(CudaCeError::InvalidInput(
                    "not enough valid data for at least one series".into(),
                ));
            }
        }

        
        let triple = 3usize
            .checked_mul(cols)
            .and_then(|x| x.checked_mul(rows))
            .ok_or_else(|| CudaCeError::InvalidInput("size overflow".into()))?;
        let two_mats = 2usize
            .checked_mul(cols)
            .and_then(|x| x.checked_mul(rows))
            .ok_or_else(|| CudaCeError::InvalidInput("size overflow".into()))?;
        let req = triple
            .checked_add(cols)
            .and_then(|x| x.checked_add(two_mats))
            .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| CudaCeError::InvalidInput("size overflow".into()))?;
        let headroom = std::env::var("CUDA_MEM_HEADROOM")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(64 * 1024 * 1024);
        Self::ensure_fit(req, headroom)?;

        let d_high = unsafe { DeviceBuffer::from_slice_async(high_tm, &self.stream) }?;
        let d_low = unsafe { DeviceBuffer::from_slice_async(low_tm, &self.stream) }?;
        let d_close = unsafe { DeviceBuffer::from_slice_async(close_tm, &self.stream) }?;
        let d_first = unsafe { DeviceBuffer::from_slice_async(&first_valids, &self.stream) }?;
        let elems_out = 2usize
            .checked_mul(cols)
            .and_then(|x| x.checked_mul(rows))
            .ok_or_else(|| CudaCeError::InvalidInput("size overflow".into()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(elems_out, &self.stream) }?;

        self.launch_many_series(
            &d_high, &d_low, &d_close, cols, rows, period, mult, &d_first, use_close, &mut d_out,
        )?;
        self.stream.synchronize()?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: 2 * rows,
            cols,
        })
    }

    
    pub fn chandelier_exit_batch_from_device_dev(
        &self,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_close: &DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        periods: &[i32],
        mults: &[f32],
        use_close: bool,
    ) -> Result<DeviceArrayF32, CudaCeError> {
        if periods.is_empty() || periods.len() != mults.len() {
            return Err(CudaCeError::InvalidInput(
                "periods/mults mismatch or empty".into(),
            ));
        }

        
        let rows = periods.len();
        let param_bytes = rows
            .checked_mul(std::mem::size_of::<i32>() + std::mem::size_of::<f32>())
            .ok_or_else(|| CudaCeError::InvalidInput("size overflow".into()))?;
        let out_bytes = 2usize
            .checked_mul(rows)
            .and_then(|x| x.checked_mul(len))
            .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| CudaCeError::InvalidInput("size overflow".into()))?;
        let req = param_bytes
            .checked_add(out_bytes)
            .ok_or_else(|| CudaCeError::InvalidInput("size overflow".into()))?;
        let headroom = std::env::var("CUDA_MEM_HEADROOM")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(64 * 1024 * 1024);
        Self::ensure_fit(req, headroom)?;

        
        let d_periods: DeviceBuffer<i32> = unsafe { DeviceBuffer::from_slice_async(periods, &self.stream) }?;
        let d_mults: DeviceBuffer<f32> = unsafe { DeviceBuffer::from_slice_async(mults, &self.stream) }?;

        
        let elems_out = rows
            .checked_mul(len)
            .and_then(|x| x.checked_mul(2))
            .ok_or_else(|| CudaCeError::InvalidInput("size overflow".into()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(elems_out, &self.stream) }?;

        
        self.launch_batch(
            d_high,
            d_low,
            d_close,
            len,
            first_valid,
            &d_periods,
            &d_mults,
            rows,
            use_close,
            &mut d_out,
        )?;

        self.stream.synchronize()?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: 2 * rows,
            cols: len,
        })
    }

    /// Device-to-device batch launch (no allocations). Output layout is identical
    /// to `chandelier_exit_batch_dev`: `(2*rows) x len` (row-major).
    pub fn chandelier_exit_batch_device_inplace(
        &self,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_close: &DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        d_periods: &DeviceBuffer<i32>,
        d_mults: &DeviceBuffer<f32>,
        rows: usize,
        use_close: bool,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaCeError> {
        let needed = rows
            .checked_mul(len)
            .and_then(|x| x.checked_mul(2))
            .ok_or_else(|| CudaCeError::InvalidInput("size overflow".into()))?;
        if d_out.len() < needed {
            return Err(CudaCeError::InvalidInput("output buffer too small".into()));
        }
        self.launch_batch(
            d_high,
            d_low,
            d_close,
            len,
            first_valid,
            d_periods,
            d_mults,
            rows,
            use_close,
            d_out,
        )
    }

    
    pub fn chandelier_exit_many_series_one_param_time_major_from_device_dev(
        &self,
        d_high_tm: &DeviceBuffer<f32>,
        d_low_tm: &DeviceBuffer<f32>,
        d_close_tm: &DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        period: usize,
        mult: f32,
        d_first_valids: &DeviceBuffer<i32>,
        use_close: bool,
    ) -> Result<DeviceArrayF32, CudaCeError> {
        if cols == 0 || rows == 0 { return Err(CudaCeError::InvalidInput("empty matrix".into())); }
        if period == 0 { return Err(CudaCeError::InvalidInput("period must be >=1".into())); }

        
        let req = 2usize
            .checked_mul(cols)
            .and_then(|x| x.checked_mul(rows))
            .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| CudaCeError::InvalidInput("size overflow".into()))?;
        let headroom = std::env::var("CUDA_MEM_HEADROOM")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(64 * 1024 * 1024);
        Self::ensure_fit(req, headroom)?;

        let elems_out = 2usize
            .checked_mul(cols)
            .and_then(|x| x.checked_mul(rows))
            .ok_or_else(|| CudaCeError::InvalidInput("size overflow".into()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(elems_out, &self.stream) }?;

        self.launch_many_series(
            d_high_tm,
            d_low_tm,
            d_close_tm,
            cols,
            rows,
            period,
            mult,
            d_first_valids,
            use_close,
            &mut d_out,
        )?;

        self.stream.synchronize()?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: 2 * rows,
            cols,
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

    fn synth_hlc_from_close(close: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut high = close.to_vec();
        let mut low = close.to_vec();
        for i in 0..close.len() {
            let v = close[i];
            if v.is_nan() {
                continue;
            }
            let x = i as f32 * 0.0025;
            let off = (0.002 * x.sin()).abs() + 0.15;
            high[i] = v + off;
            low[i] = v - off;
        }
        (high, low)
    }

    struct BatchDevState {
        cuda: CudaChandelierExit,
        d_high: DeviceBuffer<f32>,
        d_low: DeviceBuffer<f32>,
        d_close: DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        d_periods: DeviceBuffer<i32>,
        d_mults: DeviceBuffer<f32>,
        rows: usize,
        use_close: bool,
        d_out: DeviceBuffer<f32>,
    }
    impl CudaBenchState for BatchDevState {
        fn launch(&mut self) {
            self.cuda
                .chandelier_exit_batch_device_inplace(
                    &self.d_high,
                    &self.d_low,
                    &self.d_close,
                    self.len,
                    self.first_valid,
                    &self.d_periods,
                    &self.d_mults,
                    self.rows,
                    self.use_close,
                    &mut self.d_out,
                )
                .expect("ce batch dev kernel");
            self.cuda.stream().synchronize().expect("ce sync");
        }
    }

    struct ManySeriesState {
        cuda: CudaChandelierExit,
        d_high_tm: DeviceBuffer<f32>,
        d_low_tm: DeviceBuffer<f32>,
        d_close_tm: DeviceBuffer<f32>,
        d_first_valids: DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        period: usize,
        mult: f32,
        use_close: bool,
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
                    self.period,
                    self.mult,
                    &self.d_first_valids,
                    self.use_close,
                    &mut self.d_out_tm,
                )
                .expect("ce many-series kernel");
            self.cuda.stream().synchronize().expect("ce many-series sync");
        }
    }

    fn prep_batch_dev() -> Box<dyn CudaBenchState> {
        let cuda = CudaChandelierExit::new(0).expect("cuda ce");
        let close = gen_series(ONE_SERIES_LEN);
        let (high, low) = synth_hlc_from_close(&close);
        let sweep = CeBatchRange {
            period: (10, 50, 10),
            mult: (2.0, 3.0, 0.5),
            use_close: (true, true, false),
        };
        let combos = CudaChandelierExit::expand_grid(&sweep).expect("ce expand grid");
        let periods_host: Vec<i32> = combos.iter().map(|c| c.period.unwrap() as i32).collect();
        let mults_host: Vec<f32> = combos.iter().map(|c| c.mult.unwrap() as f32).collect();
        let use_close = sweep.use_close.0;
        let first_valid =
            CudaChandelierExit::first_valid(use_close, &high, &low, &close).expect("first_valid");

        let rows = periods_host.len();
        let elems_out = 2usize
            .checked_mul(rows)
            .and_then(|x| x.checked_mul(ONE_SERIES_LEN))
            .expect("size overflow");
        let d_high = unsafe { DeviceBuffer::from_slice_async(&high, cuda.stream()) }.expect("d_high");
        let d_low = unsafe { DeviceBuffer::from_slice_async(&low, cuda.stream()) }.expect("d_low");
        let d_close = unsafe { DeviceBuffer::from_slice_async(&close, cuda.stream()) }.expect("d_close");
        let d_periods = unsafe { DeviceBuffer::from_slice_async(&periods_host, cuda.stream()) }.expect("d_periods");
        let d_mults = unsafe { DeviceBuffer::from_slice_async(&mults_host, cuda.stream()) }.expect("d_mults");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(elems_out, cuda.stream()) }.expect("d_out");
        cuda.stream().synchronize().expect("ce sync");

        Box::new(BatchDevState {
            cuda,
            d_high,
            d_low,
            d_close,
            len: ONE_SERIES_LEN,
            first_valid,
            d_periods,
            d_mults,
            rows,
            use_close,
            d_out,
        })
    }
    fn prep_many() -> Box<dyn CudaBenchState> {
        let cuda = CudaChandelierExit::new(0).expect("cuda ce");
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
        let use_close = true;
        let mut first_valids = vec![rows as i32; cols];
        for s in 0..cols {
            for t in 0..rows {
                let idx = t * cols + s;
                let ok = if use_close {
                    !close_tm[idx].is_nan()
                } else {
                    !high_tm[idx].is_nan() && !low_tm[idx].is_nan() && !close_tm[idx].is_nan()
                };
                if ok {
                    first_valids[s] = t as i32;
                    break;
                }
            }
        }
        let d_high_tm = DeviceBuffer::from_slice(&high_tm).expect("d_high_tm");
        let d_low_tm = DeviceBuffer::from_slice(&low_tm).expect("d_low_tm");
        let d_close_tm = DeviceBuffer::from_slice(&close_tm).expect("d_close_tm");
        let d_first_valids = DeviceBuffer::from_slice(&first_valids).expect("d_first_valids");
        let elems_out = 2usize
            .checked_mul(cols)
            .and_then(|x| x.checked_mul(rows))
            .expect("size overflow");
        let d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems_out) }.expect("d_out_tm");
        cuda.stream().synchronize().expect("ce sync after prep");
        Box::new(ManySeriesState {
            cuda,
            d_high_tm,
            d_low_tm,
            d_close_tm,
            d_first_valids,
            cols,
            rows,
            period: 22,
            mult: 3.0,
            use_close,
            d_out_tm,
        })
    }

    fn bytes_batch() -> usize {
        
        let combos = 15usize; 
        let in_bytes = 3 * ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let param_bytes = combos * (std::mem::size_of::<i32>() + std::mem::size_of::<f32>());
        let out_bytes = 2 * combos * ONE_SERIES_LEN * std::mem::size_of::<f32>();
        in_bytes + param_bytes + out_bytes + 64 * 1024 * 1024
    }
    fn bytes_many() -> usize {
        (3 * COLS_256 * ROWS_8K + COLS_256 + 2 * COLS_256 * ROWS_8K) * std::mem::size_of::<f32>()
            + 64 * 1024 * 1024
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "chandelier_exit",
                "batch",
                "ce_cuda_batch_dev",
                "1m",
                prep_batch_dev,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_batch()),
            CudaBenchScenario::new(
                "chandelier_exit",
                "many_series_one_param",
                "ce_cuda_many_series",
                "8k x 256",
                prep_many,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_many()),
        ]
    }
}
