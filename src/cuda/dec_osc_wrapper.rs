//! CUDA wrapper for Decycler Oscillator (DEC_OSC).
//!
//! Parity goals with ALMA wrapper style:
//! - PTX load from OUT_DIR, DetermineTargetFromContext, OptLevel O2 fallback
//! - NON_BLOCKING stream
//! - Batch (one-series × many-params) and many-series × one-param (time-major)
//! - Warmup/NaN identical to scalar: write NaN for indices < first_valid+2
//! - VRAM estimation + 64MB headroom guard; chunk grid to <= 65_535 rows

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::dec_osc::{DecOscBatchRange, DecOscParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, Function, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaDecOscError {
    #[error("CUDA: {0}")]
    Cuda(#[from] cust::error::CudaError),
    #[error("Out of memory: required={required} bytes, free={free} bytes, headroom={headroom} bytes")]
    OutOfMemory { required: usize, free: usize, headroom: usize },
    #[error("Missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("Invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("Launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("Device mismatch: buffer device {buf}, current {current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("Not implemented")]
    NotImplemented,
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    Plain { block_x: u32 },
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
pub struct CudaDecOscPolicy {
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

pub struct CudaDecOsc {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
    policy: CudaDecOscPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaDecOsc {
    pub fn new(device_id: usize) -> Result<Self, CudaDecOscError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/dec_osc_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, jit_opts)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        Ok(Self {
            module,
            stream,
            context,
            device_id: device_id as u32,
            policy: CudaDecOscPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn context_arc(&self) -> Arc<Context> { self.context.clone() }
    pub fn device_id(&self) -> u32 { self.device_id }

    pub fn set_policy(&mut self, policy: CudaDecOscPolicy) { self.policy = policy; }
    pub fn policy(&self) -> &CudaDecOscPolicy { &self.policy }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> { self.last_batch }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> { self.last_many }

    #[inline]
    fn maybe_log_batch_debug(&self) {
        static ONCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if self.debug_batch_logged { return; }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                if !ONCE.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    eprintln!("[DEBUG] dec_osc batch selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut CudaDecOsc)).debug_batch_logged = true; }
            }
        }
    }
    #[inline]
    fn maybe_log_many_debug(&self) {
        static ONCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if self.debug_many_logged { return; }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                if !ONCE.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    eprintln!("[DEBUG] dec_osc many-series selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut CudaDecOsc)).debug_many_logged = true; }
            }
        }
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
            Err(_) => true,
        }
    }
    #[inline]
    fn will_fit(&self, required: usize, headroom: usize) -> Result<(), CudaDecOscError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        match mem_get_info() {
            Ok((free, _total)) => {
                if required.saturating_add(headroom) > free {
                    Err(CudaDecOscError::OutOfMemory { required, free, headroom })
                } else {
                    Ok(())
                }
            }
            Err(_) => Ok(()),
        }
    }

    #[inline]
    fn ceil_div_u32(n: u32, d: u32) -> u32 { (n + d - 1) / d }
    #[inline]
    fn ceil_div_usize(n: usize, d: usize) -> usize { (n + d - 1) / d }

    fn expand_grid_checked(range: &DecOscBatchRange) -> Result<Vec<DecOscParams>, CudaDecOscError> {
        fn axis_usize(a: (usize, usize, usize)) -> Result<Vec<usize>, CudaDecOscError> {
            let (s, e, st) = a;
            if st == 0 || s == e { return Ok(vec![s]); }
            let mut out = Vec::new();
            if s < e {
                let mut v = s;
                while v <= e {
                    out.push(v);
                    v = match v.checked_add(st) { Some(n) if n != v => n, _ => break };
                }
            } else {
                let mut v = s;
                while v >= e {
                    out.push(v);
                    if v < e + st { break; }
                    v -= st;
                    if v == 0 { break; }
                }
            }
            if out.is_empty() {
                return Err(CudaDecOscError::InvalidInput(format!(
                    "invalid range: start={}, end={}, step={}", s, e, st
                )));
            }
            Ok(out)
        }
        fn axis_f64(a: (f64, f64, f64)) -> Vec<f64> {
            let (s, e, st) = a;
            if st.abs() < 1e-12 || (s - e).abs() < 1e-12 { return vec![s]; }
            let mut v = Vec::new();
            if s <= e { let mut x = s; while x <= e + 1e-12 { v.push(x); x += st; } }
            else { let mut x = s; while x >= e - 1e-12 { v.push(x); x -= st.abs(); } }
            v
        }
        let periods = axis_usize(range.hp_period)?;
        let ks = axis_f64(range.k);
        let cap = periods.len().checked_mul(ks.len()).ok_or_else(|| CudaDecOscError::InvalidInput("rows*cols overflow".into()))?;
        let mut out = Vec::with_capacity(cap);
        for &p in &periods { for &k in &ks { out.push(DecOscParams { hp_period: Some(p), k: Some(k) }); } }
        Ok(out)
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &DecOscBatchRange,
    ) -> Result<(Vec<DecOscParams>, usize, usize), CudaDecOscError> {
        if data_f32.is_empty() {
            return Err(CudaDecOscError::InvalidInput("empty data".into()));
        }
        let len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaDecOscError::InvalidInput("all values are NaN".into()))?;

        let combos = Self::expand_grid_checked(sweep)?;
        for prm in &combos {
            let p = prm.hp_period.unwrap_or(0);
            let k = prm.k.unwrap_or(0.0);
            if p < 2 || p > len {
                return Err(CudaDecOscError::InvalidInput(format!(
                    "invalid hp_period {} for len {}",
                    p, len
                )));
            }
            if k <= 0.0 || !k.is_finite() {
                return Err(CudaDecOscError::InvalidInput(format!("invalid k {}", k)));
            }
            if len - first_valid < 2 {
                return Err(CudaDecOscError::InvalidInput(
                    "not enough valid data".into(),
                ));
            }
        }
        Ok((combos, first_valid, len))
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_ks: &DeviceBuffer<f32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
        periods_off: usize,
        out_off_elems: usize,
    ) -> Result<(), CudaDecOscError> {
        let mut func: Function = self
            .module
            .get_function("dec_osc_batch_f32")
            .map_err(|_| CudaDecOscError::MissingKernelSymbol { name: "dec_osc_batch_f32" })?;

        
        let (suggested_block_x, min_grid) = func
            .suggested_launch_configuration(0, BlockSize::xyz(0, 0, 0))?;
        let block_x = match self.policy.batch {
            BatchKernelPolicy::Auto => suggested_block_x.max(128),
            BatchKernelPolicy::Plain { block_x } => block_x.max(64),
        };
        
        let dev = Device::get_device(self.device_id)?;
        let max_bx = dev.get_attribute(cust::device::DeviceAttribute::MaxBlockDimX)? as u32;
        if block_x > max_bx {
            return Err(CudaDecOscError::LaunchConfigTooLarge { gx: 1, gy: 1, gz: 1, bx: block_x, by: 1, bz: 1 });
        }
        unsafe {
            (*(self as *const _ as *mut CudaDecOsc)).last_batch =
                Some(BatchKernelSelected::Plain { block_x });
        }
        unsafe {
            (*(self as *const _ as *mut CudaDecOsc)).last_batch =
                Some(BatchKernelSelected::Plain { block_x });
        }
        self.maybe_log_batch_debug();
        
        let combos_u32 = n_combos as u32;
        let mut grid_x = Self::ceil_div_u32(combos_u32, block_x);
        grid_x = grid_x.max(min_grid);
        let grid: GridSize = (grid_x, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw()
                + (periods_off * std::mem::size_of::<i32>()) as u64;
            let mut ks_ptr =
                d_ks.as_device_ptr().as_raw() + (periods_off * std::mem::size_of::<f32>()) as u64;
            let mut ks_ptr =
                d_ks.as_device_ptr().as_raw() + (periods_off * std::mem::size_of::<f32>()) as u64;
            let mut len_i = series_len as i32;
            let mut combos_i = n_combos as i32;
            let mut first_i = first_valid as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw()
                + (out_off_elems * std::mem::size_of::<f32>()) as u64;
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut ks_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut first_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        Ok(())
    }

    pub fn dec_osc_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &DecOscBatchRange,
    ) -> Result<DeviceArrayF32, CudaDecOscError> {
        let (combos, first_valid, len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let rows = combos.len();

        let prices_bytes = len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaDecOscError::InvalidInput("prices_bytes overflow".into()))?;
        let params_bytes = rows
            .checked_mul(std::mem::size_of::<i32>() + std::mem::size_of::<f32>())
            .ok_or_else(|| CudaDecOscError::InvalidInput("params_bytes overflow".into()))?;
        let out_elems = rows
            .checked_mul(len)
            .ok_or_else(|| CudaDecOscError::InvalidInput("rows*len overflow".into()))?;
        let out_bytes = out_elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaDecOscError::InvalidInput("out_bytes overflow".into()))?;
        let need = prices_bytes
            .checked_add(params_bytes)
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| CudaDecOscError::InvalidInput("bytes overflow".into()))?;
        let headroom = 64 * 1024 * 1024;
        self.will_fit(need, headroom)?;

        let periods: Vec<i32> = combos.iter().map(|c| c.hp_period.unwrap() as i32).collect();
        let ks: Vec<f32> = combos.iter().map(|c| c.k.unwrap() as f32).collect();

        
        let h_prices = LockedBuffer::from_slice(data_f32)?;
        let h_periods = LockedBuffer::from_slice(&periods)?;
        let h_ks = LockedBuffer::from_slice(&ks)?;

        let mut d_prices =
            unsafe { DeviceBuffer::<f32>::uninitialized_async(len, &self.stream) }?;
        let mut d_periods =
            unsafe { DeviceBuffer::<i32>::uninitialized_async(rows, &self.stream) }?;
        let mut d_ks =
            unsafe { DeviceBuffer::<f32>::uninitialized_async(rows, &self.stream) }?;
        let mut d_out =
            unsafe { DeviceBuffer::<f32>::uninitialized_async(out_elems, &self.stream) }?;
        unsafe {
            d_prices.async_copy_from(&h_prices, &self.stream)?;
            d_periods.async_copy_from(&h_periods, &self.stream)?;
            d_ks.async_copy_from(&h_ks, &self.stream)?;
        }

        
        let func = self
            .module
            .get_function("dec_osc_batch_f32")
            .map_err(|_| CudaDecOscError::MissingKernelSymbol { name: "dec_osc_batch_f32" })?;
        let (suggested_block_x, _min_grid) = func
            .suggested_launch_configuration(0, BlockSize::xyz(0, 0, 0))?;
        let block_x = match self.policy.batch {
            BatchKernelPolicy::Auto => suggested_block_x.max(128),
            BatchKernelPolicy::Plain { block_x } => block_x.max(64),
        } as usize;

        const MAX_GRID_X: usize = 65_535; 
        let max_combos_per_launch = MAX_GRID_X.saturating_mul(block_x);

        let mut launched = 0usize;
        while launched < rows {
            let n = (rows - launched).min(max_combos_per_launch);
            self.launch_batch_kernel(
                &d_prices,
                &d_periods,
                &d_ks,
                len,
                n,
                first_valid,
                &mut d_out,
                launched,
                launched * len,
            )?;
            launched += n;
        }

        self.stream.synchronize()?;

        Ok(DeviceArrayF32 { buf: d_out, rows, cols: len })
    }

    fn prepare_many_series(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &DecOscParams,
    ) -> Result<(Vec<i32>, usize, f32), CudaDecOscError> {
        if cols == 0 || rows == 0 {
            return Err(CudaDecOscError::InvalidInput("cols or rows is zero".into()));
        }
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaDecOscError::InvalidInput("cols*rows overflow".into()))?;
        if data_tm_f32.len() != expected {
            return Err(CudaDecOscError::InvalidInput(
                "time-major shape mismatch".into(),
            ));
        }
        let p = params.hp_period.unwrap_or(0);
        let k = params.k.unwrap_or(0.0);
        if p < 2 || p > rows {
            return Err(CudaDecOscError::InvalidInput("invalid hp_period".into()));
        }
        if k <= 0.0 || !k.is_finite() {
            return Err(CudaDecOscError::InvalidInput("invalid k".into()));
        }
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = None;
            for t in 0..rows {
                let v = data_tm_f32[t * cols + s];
                if !v.is_nan() {
                    fv = Some(t);
                    break;
                }
            }
            let fv =
                fv.ok_or_else(|| CudaDecOscError::InvalidInput(format!("series {} all NaN", s)))?;
            if rows - fv < 2 {
                return Err(CudaDecOscError::InvalidInput(format!(
                    "series {} not enough valid data (need >= 2, got {})",
                    s,
                    rows - fv
                )));
            }
            first_valids[s] = fv as i32;
        }
        Ok((first_valids, p, k as f32))
    }

    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        period: usize,
        k: f32,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaDecOscError> {
        let func: Function = self
            .module
            .get_function("dec_osc_many_series_one_param_time_major_f32")
            .map_err(|_| CudaDecOscError::MissingKernelSymbol { name: "dec_osc_many_series_one_param_time_major_f32" })?;

        
        let (suggested_block_x, _min_grid) = func
            .suggested_launch_configuration(0, BlockSize::xyz(0, 0, 0))?;
        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => suggested_block_x.max(128),
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(64),
        };
        
        let dev = Device::get_device(self.device_id)?;
        let max_bx = dev.get_attribute(cust::device::DeviceAttribute::MaxBlockDimX)? as u32;
        if block_x > max_bx {
            return Err(CudaDecOscError::LaunchConfigTooLarge {
                gx: 1,
                gy: 1,
                gz: 1,
                bx: block_x,
                by: 1,
                bz: 1,
            });
        }
        unsafe {
            (*(self as *const _ as *mut CudaDecOsc)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }
        self.maybe_log_many_debug();
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut period_i = period as i32;
            let mut k_f = k as f32;
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut k_f as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        Ok(())
    }

    pub fn dec_osc_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &DecOscParams,
    ) -> Result<DeviceArrayF32, CudaDecOscError> {
        let (first_valids, period, k_f32) =
            Self::prepare_many_series(data_tm_f32, cols, rows, params)?;

        
        let prices_bytes = data_tm_f32
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaDecOscError::InvalidInput("prices_bytes overflow".into()))?;
        let first_bytes = first_valids
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaDecOscError::InvalidInput("first_bytes overflow".into()))?;
        let out_elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaDecOscError::InvalidInput("rows*cols overflow".into()))?;
        let out_bytes = out_elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaDecOscError::InvalidInput("out_bytes overflow".into()))?;
        let need = prices_bytes
            .checked_add(first_bytes)
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| CudaDecOscError::InvalidInput("bytes overflow".into()))?;
        let headroom     = 64 * 1024 * 1024;
        self.will_fit(need, headroom)?;

        
        let h_prices = LockedBuffer::from_slice(data_tm_f32)?;
        let h_first = LockedBuffer::from_slice(&first_valids)?;

        let mut d_prices =
            unsafe { DeviceBuffer::<f32>::uninitialized_async(out_elems, &self.stream) }?;
        let mut d_first =
            unsafe { DeviceBuffer::<i32>::uninitialized_async(cols, &self.stream) }?;
        let mut d_out =
            unsafe { DeviceBuffer::<f32>::uninitialized_async(out_elems, &self.stream) }?;

        unsafe {
            d_prices
                .async_copy_from(&h_prices, &self.stream)
                ?;
            d_first
                .async_copy_from(&h_first, &self.stream)
                ?;
        }

        self.launch_many_series_kernel(&d_prices, &d_first, cols, rows, period, k_f32, &mut d_out)?;

        self.stream.synchronize()?;

        Ok(DeviceArrayF32 { buf: d_out, rows, cols })
    }
}


pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250;
    const MANY_SERIES_COLS: usize = 250;
    const MANY_SERIES_LEN: usize = 1_000_000;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let params_bytes = PARAM_SWEEP * (std::mem::size_of::<i32>() + std::mem::size_of::<f32>());
        let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + params_bytes + out_bytes + 64 * 1024 * 1024
    }

    fn bytes_many_series_one_param() -> usize {
        let elems = MANY_SERIES_COLS * MANY_SERIES_LEN;
        let in_bytes = elems * std::mem::size_of::<f32>();
        let first_bytes = MANY_SERIES_COLS * std::mem::size_of::<i32>();
        let out_bytes = elems * std::mem::size_of::<f32>();
        in_bytes + first_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct BatchDeviceState {
        cuda: CudaDecOsc,
        d_prices: DeviceBuffer<f32>,
        d_periods: DeviceBuffer<i32>,
        d_ks: DeviceBuffer<f32>,
        d_out: DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        n_combos: usize,
    }
    impl CudaBenchState for BatchDeviceState {
        fn launch(&mut self) {
            self.cuda
                .launch_batch_kernel(
                    &self.d_prices,
                    &self.d_periods,
                    &self.d_ks,
                    self.len,
                    self.n_combos,
                    self.first_valid,
                    &mut self.d_out,
                    0,
                    0,
                )
                .expect("dec_osc launch_batch_kernel");
            self.cuda.stream.synchronize().expect("dec_osc sync");
        }
    }

    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaDecOsc::new(0).expect("cuda dec_osc");
        let price = gen_series(ONE_SERIES_LEN);
        let sweep = DecOscBatchRange {
            hp_period: (50, 50 + PARAM_SWEEP - 1, 1),
            k: (1.0, 1.0, 0.0),
        };
        let (combos, first_valid, len) =
            CudaDecOsc::prepare_batch_inputs(&price, &sweep).expect("prepare_batch_inputs");
        let n_combos = combos.len();
        let periods_i32: Vec<i32> = combos.iter().map(|c| c.hp_period.unwrap() as i32).collect();
        let ks_f32: Vec<f32> = combos.iter().map(|c| c.k.unwrap() as f32).collect();

        let d_prices =
            unsafe { DeviceBuffer::from_slice_async(&price, &cuda.stream) }.expect("d_prices");
        let d_periods = DeviceBuffer::from_slice(&periods_i32).expect("d_periods");
        let d_ks = DeviceBuffer::from_slice(&ks_f32).expect("d_ks");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(n_combos * len, &cuda.stream) }.expect("d_out");
        cuda.stream.synchronize().expect("sync after prep");

        Box::new(BatchDeviceState {
            cuda,
            d_prices,
            d_periods,
            d_ks,
            d_out,
            len,
            first_valid,
            n_combos,
        })
    }

    struct ManySeriesDeviceState {
        cuda: CudaDecOsc,
        d_prices_tm: DeviceBuffer<f32>,
        d_first: DeviceBuffer<i32>,
        d_out_tm: DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        period: usize,
        k: f32,
    }
    impl CudaBenchState for ManySeriesDeviceState {
        fn launch(&mut self) {
            self.cuda
                .launch_many_series_kernel(
                    &self.d_prices_tm,
                    &self.d_first,
                    self.cols,
                    self.rows,
                    self.period,
                    self.k,
                    &mut self.d_out_tm,
                )
                .expect("dec_osc launch_many_series_kernel");
            self.cuda.stream.synchronize().expect("dec_osc sync");
        }
    }

    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaDecOsc::new(0).expect("cuda dec_osc");
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_LEN;
        let data_tm = gen_time_major_prices(cols, rows);
        let params = DecOscParams {
            hp_period: Some(125),
            k: Some(1.0),
        };

        let (first_valids, period, k_f32) =
            CudaDecOsc::prepare_many_series(&data_tm, cols, rows, &params).expect("prepare_many_series");

        let d_prices_tm =
            unsafe { DeviceBuffer::from_slice_async(&data_tm, &cuda.stream) }.expect("d_prices_tm");
        let d_first = DeviceBuffer::from_slice(&first_valids).expect("d_first");
        let d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(cols * rows) }.expect("d_out_tm");
        cuda.stream.synchronize().expect("sync after prep");

        Box::new(ManySeriesDeviceState {
            cuda,
            d_prices_tm,
            d_first,
            d_out_tm,
            cols,
            rows,
            period,
            k: k_f32,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "dec_osc",
                "one_series_many_params",
                "dec_osc_cuda_batch_dev",
                "1m_x_250",
                prep_one_series_many_params,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "dec_osc",
                "many_series_one_param",
                "dec_osc_cuda_many_series_one_param",
                "250x1m",
                prep_many_series_one_param,
            )
            .with_sample_size(5)
            .with_mem_required(bytes_many_series_one_param()),
        ]
    }
}
