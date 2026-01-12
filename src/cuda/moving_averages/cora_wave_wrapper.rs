#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use super::cwma_wrapper::{BatchKernelPolicy, BatchThreadsPerOutput, ManySeriesKernelPolicy};
use crate::indicators::cora_wave::{CoraWaveBatchRange, CoraWaveParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CudaCoraWaveError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("out of memory: required={required} free={free} headroom={headroom}")]
    OutOfMemory {
        required: usize,
        free: usize,
        headroom: usize,
    },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge {
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
    },
    #[error("device mismatch: buf={buf} current={current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
    NotImplemented,
}

#[derive(Clone, Copy, Debug)]
pub struct CudaCoraWavePolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaCoraWavePolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected {
    Plain { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}

pub struct CudaCoraWave {
    module: Module,
    stream: Stream,
    context: std::sync::Arc<Context>,
    device_id: u32,
    policy: CudaCoraWavePolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaCoraWave {
    pub fn new(device_id: usize) -> Result<Self, CudaCoraWaveError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = std::sync::Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/cora_wave_kernel.ptx"));
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
            policy: CudaCoraWavePolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn new_with_policy(
        device_id: usize,
        policy: CudaCoraWavePolicy,
    ) -> Result<Self, CudaCoraWaveError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }
    pub fn set_policy(&mut self, policy: CudaCoraWavePolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaCoraWavePolicy {
        &self.policy
    }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch
    }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many
    }
    pub fn synchronize(&self) -> Result<(), CudaCoraWaveError> {
        self.stream.synchronize().map_err(Into::into)
    }
    pub fn context_arc(&self) -> std::sync::Arc<Context> {
        self.context.clone()
    }
    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    #[inline]
    fn maybe_log_batch_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                let per = std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] CoRa batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaCoraWave)).debug_batch_logged = true;
                }
            }
        }
    }
    #[inline]
    fn maybe_log_many_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_many_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                let per = std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] CoRa many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaCoraWave)).debug_many_logged = true;
                }
            }
        }
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
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> bool {
        if !Self::mem_check_enabled() {
            return true;
        }
        if let Some((free, _)) = Self::device_mem_info() {
            required_bytes.saturating_add(headroom_bytes) <= free
        } else {
            true
        }
    }

    #[inline]
    fn grid_y_chunks(n_combos: usize) -> impl Iterator<Item = (usize, usize)> {
        const MAX_Y: usize = 65_535;
        (0..n_combos)
            .step_by(MAX_Y)
            .map(move |start| (start, (n_combos - start).min(MAX_Y)))
    }

    fn expand_grid(range: &CoraWaveBatchRange) -> Result<Vec<CoraWaveParams>, CudaCoraWaveError> {
        let (ps, pe, pt) = range.period;
        let periods: Vec<usize> = if pt == 0 || ps == pe {
            vec![ps]
        } else if ps <= pe {
            (ps..=pe).step_by(pt).collect()
        } else {
            let mut v = Vec::new();
            let mut x = ps;
            loop {
                v.push(x);
                if x <= pe {
                    break;
                }
                if x < pt {
                    break;
                }
                let next = x - pt;
                if next < pe {
                    break;
                }
                x = next;
            }
            v
        };
        let (ms, me, mt) = range.r_multi;
        let mut mults: Vec<f64> = vec![];
        if mt.abs() < 1e-12 || (ms - me).abs() < 1e-12 {
            mults.push(ms);
        } else if ms <= me {
            let mut x = ms;
            let step = mt.abs();
            while x <= me + 1e-12 {
                mults.push(x);
                x += step;
            }
        } else {
            let mut x = ms;
            let step = mt.abs();
            while x >= me - 1e-12 {
                mults.push(x);
                x -= step;
            }
        }
        if periods.is_empty() || mults.is_empty() {
            return Err(CudaCoraWaveError::InvalidInput(
                "empty parameter expansion".into(),
            ));
        }
        let mut out = Vec::with_capacity(periods.len().saturating_mul(mults.len()));
        for &p in &periods {
            for &m in &mults {
                out.push(CoraWaveParams {
                    period: Some(p),
                    r_multi: Some(m),
                    smooth: Some(range.smooth),
                });
            }
        }
        Ok(out)
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &CoraWaveBatchRange,
    ) -> Result<
        (
            Vec<CoraWaveParams>,
            usize,
            usize,
            usize,
            Vec<i32>,
            Vec<f32>,
            Vec<f32>,
            Vec<i32>,
            Vec<i32>,
        ),
        CudaCoraWaveError,
    > {
        if data_f32.is_empty() {
            return Err(CudaCoraWaveError::InvalidInput("empty data".into()));
        }
        let first_valid = data_f32
            .iter()
            .position(|x| !x.is_nan())
            .ok_or_else(|| CudaCoraWaveError::InvalidInput("all values are NaN".into()))?;
        let series_len = data_f32.len();

        let combos = Self::expand_grid(sweep)?;
        if combos.is_empty() {
            return Err(CudaCoraWaveError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        let mut max_period = 0usize;
        for prm in &combos {
            let p = prm.period.unwrap_or(0);
            if p == 0 || p > series_len {
                return Err(CudaCoraWaveError::InvalidInput(format!(
                    "invalid period {} for series length {}",
                    p, series_len
                )));
            }
            if series_len - first_valid < p {
                return Err(CudaCoraWaveError::InvalidInput(format!(
                    "not enough valid data: needed >= {}, valid = {}",
                    p,
                    series_len - first_valid
                )));
            }
            max_period = max_period.max(p);
        }

        let n = combos.len();
        let mut periods_i32 = vec![0i32; n];
        let mut inv_norms = vec![0f32; n];
        let mut smooth_periods = vec![1i32; n];
        let mut warm0s = vec![0i32; n];
        let flat_len = n.checked_mul(max_period).ok_or_else(|| {
            CudaCoraWaveError::InvalidInput("n_combos*max_period overflow".into())
        })?;
        let mut weights_flat = vec![0f32; flat_len];
        for (row, prm) in combos.iter().enumerate() {
            let p = prm.period.unwrap();
            let r_multi = prm.r_multi.unwrap_or(2.0);

            let start_wt = 0.01f64;
            let end_wt = p as f64;
            let r = (end_wt / start_wt).powf(1.0 / ((p as f64) - 1.0)) - 1.0;
            let base = 1.0 + r * r_multi;
            let mut sum = 0.0f64;
            for j in 0..p {
                let w = start_wt * base.powi((j as i32) + 1);
                weights_flat[row * max_period + j] = w as f32;
                sum += w;
            }
            periods_i32[row] = p as i32;
            inv_norms[row] = (1.0f64 / sum.max(1e-30)) as f32;
            warm0s[row] = (first_valid + p - 1) as i32;
            if sweep.smooth {
                smooth_periods[row] = ((p as f64).sqrt().round() as i32).max(1);
            }
        }
        Ok((
            combos,
            first_valid,
            series_len,
            max_period,
            periods_i32,
            inv_norms,
            weights_flat,
            smooth_periods,
            warm0s,
        ))
    }

    fn pick_block_x(
        &self,
        func: &cust::function::Function,
        shared_bytes: usize,
        default: u32,
    ) -> u32 {
        match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => block_x,
            _ => func
                .suggested_launch_configuration(shared_bytes, BlockSize::xyz(0, 0, 0))
                .map(|(_, bx)| bx)
                .unwrap_or(default)
                .max(64)
                .min(1024),
        }
    }

    pub fn cora_wave_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &CoraWaveBatchRange,
    ) -> Result<DeviceArrayF32, CudaCoraWaveError> {
        let (
            combos,
            first_valid,
            series_len,
            max_period,
            periods_i32,
            inv_norms,
            weights_flat,
            smooth_periods,
            warm0s,
        ) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let n_combos = combos.len();

        let sz_f32 = std::mem::size_of::<f32>();
        let sz_i32 = std::mem::size_of::<i32>();
        let prices_bytes = series_len
            .checked_mul(sz_f32)
            .ok_or_else(|| CudaCoraWaveError::InvalidInput("series_len bytes overflow".into()))?;
        let weights_bytes = n_combos
            .checked_mul(max_period)
            .and_then(|x| x.checked_mul(sz_f32))
            .ok_or_else(|| CudaCoraWaveError::InvalidInput("weights bytes overflow".into()))?;
        let periods_bytes = n_combos
            .checked_mul(sz_i32)
            .ok_or_else(|| CudaCoraWaveError::InvalidInput("periods bytes overflow".into()))?;
        let inv_bytes = n_combos
            .checked_mul(sz_f32)
            .ok_or_else(|| CudaCoraWaveError::InvalidInput("inv bytes overflow".into()))?;
        let smooth_bytes = n_combos
            .checked_mul(sz_i32)
            .and_then(|x| x.checked_mul(2))
            .ok_or_else(|| CudaCoraWaveError::InvalidInput("smooth bytes overflow".into()))?;
        let out_bytes = n_combos
            .checked_mul(series_len)
            .and_then(|x| x.checked_mul(sz_f32))
            .ok_or_else(|| CudaCoraWaveError::InvalidInput("out bytes overflow".into()))?;
        let tmp_bytes = if sweep.smooth { out_bytes } else { 0 };
        let required = prices_bytes
            .checked_add(weights_bytes)
            .and_then(|x| x.checked_add(periods_bytes))
            .and_then(|x| x.checked_add(inv_bytes))
            .and_then(|x| x.checked_add(smooth_bytes))
            .and_then(|x| x.checked_add(out_bytes))
            .and_then(|x| x.checked_add(tmp_bytes))
            .ok_or_else(|| CudaCoraWaveError::InvalidInput("total bytes overflow".into()))?;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            let free = Self::device_mem_info().map(|(f, _)| f).unwrap_or(0);
            return Err(CudaCoraWaveError::OutOfMemory {
                required,
                free,
                headroom,
            });
        }

        let h_prices = LockedBuffer::from_slice(data_f32)?;
        let mut d_prices: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(series_len) }?;
        unsafe {
            d_prices.async_copy_from(&*h_prices, &self.stream)?;
        }

        let d_weights = DeviceBuffer::from_slice(&weights_flat)?;
        let d_periods = DeviceBuffer::from_slice(&periods_i32)?;
        let d_inv = DeviceBuffer::from_slice(&inv_norms)?;
        let n_tmp = n_combos.checked_mul(series_len).ok_or_else(|| {
            CudaCoraWaveError::InvalidInput("n_combos*series_len overflow".into())
        })?;
        let mut d_tmp: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(n_tmp) }?;

        let mut func = self
            .module
            .get_function("cora_wave_batch_f32")
            .map_err(|_| CudaCoraWaveError::MissingKernelSymbol {
                name: "cora_wave_batch_f32",
            })?;
        let shared_bytes = (max_period * std::mem::size_of::<f32>()) as u32;
        let block_x = self.pick_block_x(&func, shared_bytes as usize, 256);
        let grid_x = ((series_len as u32) + block_x - 1) / block_x;
        let block: BlockSize = (block_x, 1, 1).into();

        for (start, len) in Self::grid_y_chunks(n_combos) {
            unsafe {
                let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                let mut weights_ptr = d_weights.as_device_ptr().add(start * max_period).as_raw();
                let mut periods_ptr = d_periods.as_device_ptr().add(start).as_raw();
                let mut inv_ptr = d_inv.as_device_ptr().add(start).as_raw();
                let mut max_p_i = max_period as i32;
                let mut series_i = series_len as i32;
                let mut n_i = len as i32;
                let mut first_i = first_valid as i32;
                let mut out_ptr = d_tmp.as_device_ptr().add(start * series_len).as_raw();
                let grid: GridSize = (grid_x, len as u32, 1).into();
                let args: &mut [*mut c_void] = &mut [
                    &mut prices_ptr as *mut _ as *mut c_void,
                    &mut weights_ptr as *mut _ as *mut c_void,
                    &mut periods_ptr as *mut _ as *mut c_void,
                    &mut inv_ptr as *mut _ as *mut c_void,
                    &mut max_p_i as *mut _ as *mut c_void,
                    &mut series_i as *mut _ as *mut c_void,
                    &mut n_i as *mut _ as *mut c_void,
                    &mut first_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream.launch(&func, grid, block, shared_bytes, args)?;
            }
        }
        unsafe {
            (*(self as *const _ as *mut CudaCoraWave)).last_batch =
                Some(BatchKernelSelected::Plain { block_x });
        }
        self.maybe_log_batch_debug();

        if !sweep.smooth {
            self.stream.synchronize()?;
            return Ok(DeviceArrayF32 {
                buf: d_tmp,
                rows: n_combos,
                cols: series_len,
            });
        }

        let d_smooth = DeviceBuffer::from_slice(&smooth_periods)?;
        let d_warm0s = DeviceBuffer::from_slice(&warm0s)?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(n_tmp) }?;

        let mut func_wma = self
            .module
            .get_function("cora_wave_batch_wma_from_y_f32")
            .map_err(|_| CudaCoraWaveError::MissingKernelSymbol {
                name: "cora_wave_batch_wma_from_y_f32",
            })?;
        let block_x2 = self.pick_block_x(&func_wma, 0, 256);
        let grid_x2 = ((series_len as u32) + block_x2 - 1) / block_x2;
        let block2: BlockSize = (block_x2, 1, 1).into();
        for (start, len) in Self::grid_y_chunks(n_combos) {
            unsafe {
                let mut y_ptr = d_tmp.as_device_ptr().add(start * series_len).as_raw();
                let mut sm_ptr = d_smooth.as_device_ptr().add(start).as_raw();
                let mut w0_ptr = d_warm0s.as_device_ptr().add(start).as_raw();
                let mut series_i = series_len as i32;
                let mut n_i = len as i32;
                let mut out_ptr = d_out.as_device_ptr().add(start * series_len).as_raw();
                let grid: GridSize = (grid_x2, len as u32, 1).into();
                let args: &mut [*mut c_void] = &mut [
                    &mut y_ptr as *mut _ as *mut c_void,
                    &mut sm_ptr as *mut _ as *mut c_void,
                    &mut w0_ptr as *mut _ as *mut c_void,
                    &mut series_i as *mut _ as *mut c_void,
                    &mut n_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream.launch(&func_wma, grid, block2, 0, args)?;
            }
        }
        self.stream.synchronize()?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: series_len,
        })
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &CoraWaveParams,
    ) -> Result<(Vec<i32>, usize, Vec<f32>, f32, usize), CudaCoraWaveError> {
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaCoraWaveError::InvalidInput("cols*rows overflow".into()))?;
        if cols == 0 || rows == 0 || data_tm_f32.len() != expected {
            return Err(CudaCoraWaveError::InvalidInput(
                "invalid dims or data length".into(),
            ));
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
                fv.ok_or_else(|| CudaCoraWaveError::InvalidInput(format!("series {} all NaN", s)))?;
            first_valids[s] = fv as i32;
        }
        let period = params.period.unwrap_or(0);
        if period == 0 || period > rows {
            return Err(CudaCoraWaveError::InvalidInput("invalid period".into()));
        }

        let r_multi = params.r_multi.unwrap_or(2.0);
        let start_wt = 0.01f64;
        let end_wt = period as f64;
        let r = (end_wt / start_wt).powf(1.0 / ((period as f64) - 1.0)) - 1.0;
        let base = 1.0 + r * r_multi;
        let mut weights = vec![0f32; period];
        let mut sum = 0.0f64;
        for j in 0..period {
            let w = start_wt * base.powi((j as i32) + 1);
            weights[j] = w as f32;
            sum += w;
        }
        let inv_norm = (1.0f64 / sum.max(1e-30)) as f32;
        Ok((first_valids, period, weights, inv_norm, rows))
    }

    pub fn cora_wave_multi_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &CoraWaveParams,
    ) -> Result<DeviceArrayF32, CudaCoraWaveError> {
        let (first_valids, period, weights, inv_norm, _rows) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;

        let prices_bytes = cols
            .checked_mul(rows)
            .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| CudaCoraWaveError::InvalidInput("prices bytes overflow".into()))?;
        let weights_bytes = period * std::mem::size_of::<f32>();
        let fv_bytes = cols * std::mem::size_of::<i32>();
        let out_bytes = cols
            .checked_mul(rows)
            .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| CudaCoraWaveError::InvalidInput("out bytes overflow".into()))?;
        let tmp_bytes = if params.smooth.unwrap_or(true) {
            out_bytes
        } else {
            0
        };
        let required = prices_bytes
            .checked_add(weights_bytes)
            .and_then(|x| x.checked_add(fv_bytes))
            .and_then(|x| x.checked_add(out_bytes))
            .and_then(|x| x.checked_add(tmp_bytes))
            .ok_or_else(|| CudaCoraWaveError::InvalidInput("total bytes overflow".into()))?;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            let free = Self::device_mem_info().map(|(f, _)| f).unwrap_or(0);
            return Err(CudaCoraWaveError::OutOfMemory {
                required,
                free,
                headroom,
            });
        }

        let d_prices = DeviceBuffer::from_slice(data_tm_f32)?;
        let d_weights = DeviceBuffer::from_slice(&weights)?;
        let d_first = DeviceBuffer::from_slice(&first_valids)?;
        let n_tmp = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaCoraWaveError::InvalidInput("cols*rows overflow".into()))?;
        let mut d_tmp: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(n_tmp) }?;

        let mut func = self
            .module
            .get_function("cora_wave_multi_series_one_param_time_major_f32")
            .map_err(|_| CudaCoraWaveError::MissingKernelSymbol {
                name: "cora_wave_multi_series_one_param_time_major_f32",
            })?;
        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => block_x,
            _ => func
                .suggested_launch_configuration(0, BlockSize::xyz(0, 0, 0))
                .map(|(_, bx)| bx)
                .unwrap_or(256)
                .max(64)
                .min(1024),
        };
        let grid_x = ((rows as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x, cols as u32, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut w_ptr = d_weights.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut inv = inv_norm as f32;
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut first_ptr = d_first.as_device_ptr().as_raw();
            let mut out_ptr = d_tmp.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut w_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut inv as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        unsafe {
            (*(self as *const _ as *mut CudaCoraWave)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }
        self.maybe_log_many_debug();

        if !params.smooth.unwrap_or(true) {
            self.stream.synchronize()?;
            return Ok(DeviceArrayF32 {
                buf: d_tmp,
                rows,
                cols,
            });
        }

        let wma_m = ((period as f64).sqrt().round() as i32).max(1);
        let mut warm0s = vec![0i32; cols];
        for s in 0..cols {
            warm0s[s] = first_valids[s] + (period as i32) - 1;
        }
        let d_warm0s = DeviceBuffer::from_slice(&warm0s)?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(n_tmp) }?;

        let mut func_wma = self
            .module
            .get_function("cora_wave_ms1p_wma_time_major_f32")
            .map_err(|_| CudaCoraWaveError::MissingKernelSymbol {
                name: "cora_wave_ms1p_wma_time_major_f32",
            })?;
        let block_x2 = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => block_x,
            _ => func_wma
                .suggested_launch_configuration(0, BlockSize::xyz(0, 0, 0))
                .map(|(_, bx)| bx)
                .unwrap_or(256)
                .max(64)
                .min(1024),
        };
        let grid_x2 = ((rows as u32) + block_x2 - 1) / block_x2;
        let grid2: GridSize = (grid_x2, cols as u32, 1).into();
        let block2: BlockSize = (block_x2, 1, 1).into();
        unsafe {
            let mut y_ptr = d_tmp.as_device_ptr().as_raw();
            let mut m_i = wma_m as i32;
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut w0_ptr = d_warm0s.as_device_ptr().as_raw();
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut y_ptr as *mut _ as *mut c_void,
                &mut m_i as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut w0_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func_wma, grid2, block2, 0, args)?;
        }
        self.stream.synchronize()?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }
}

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};
    use crate::indicators::cora_wave::{CoraWaveBatchRange, CoraWaveParams};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250;
    const MANY_SERIES_COLS: usize = 250;
    const MANY_SERIES_LEN: usize = 1_000_000;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }
    fn bytes_many_series_one_param() -> usize {
        let elems = MANY_SERIES_COLS * MANY_SERIES_LEN;
        let in_bytes = elems * std::mem::size_of::<f32>();
        let out_bytes = elems * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct CoraWaveBatchDevState {
        cuda: CudaCoraWave,
        d_prices: DeviceBuffer<f32>,
        d_weights: DeviceBuffer<f32>,
        d_periods: DeviceBuffer<i32>,
        d_inv: DeviceBuffer<f32>,
        d_smooth: DeviceBuffer<i32>,
        d_warm0s: DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        max_period: usize,
        first_valid: usize,
        grid_x: u32,
        block_x: u32,
        shared_bytes: u32,
        grid_x2: u32,
        block_x2: u32,
        d_tmp: DeviceBuffer<f32>,
        d_out: DeviceBuffer<f32>,
    }
    impl CudaBenchState for CoraWaveBatchDevState {
        fn launch(&mut self) {
            let func = self
                .cuda
                .module
                .get_function("cora_wave_batch_f32")
                .expect("cora_wave_batch_f32");
            let block: BlockSize = (self.block_x, 1, 1).into();

            for (start, len) in CudaCoraWave::grid_y_chunks(self.n_combos) {
                unsafe {
                    let mut prices_ptr = self.d_prices.as_device_ptr().as_raw();
                    let mut weights_ptr = self
                        .d_weights
                        .as_device_ptr()
                        .add(start * self.max_period)
                        .as_raw();
                    let mut periods_ptr = self.d_periods.as_device_ptr().add(start).as_raw();
                    let mut inv_ptr = self.d_inv.as_device_ptr().add(start).as_raw();
                    let mut max_p_i = self.max_period as i32;
                    let mut series_i = self.series_len as i32;
                    let mut n_i = len as i32;
                    let mut first_i = self.first_valid as i32;
                    let mut out_ptr = self
                        .d_tmp
                        .as_device_ptr()
                        .add(start * self.series_len)
                        .as_raw();
                    let grid: GridSize = (self.grid_x, len as u32, 1).into();
                    let args: &mut [*mut c_void] = &mut [
                        &mut prices_ptr as *mut _ as *mut c_void,
                        &mut weights_ptr as *mut _ as *mut c_void,
                        &mut periods_ptr as *mut _ as *mut c_void,
                        &mut inv_ptr as *mut _ as *mut c_void,
                        &mut max_p_i as *mut _ as *mut c_void,
                        &mut series_i as *mut _ as *mut c_void,
                        &mut n_i as *mut _ as *mut c_void,
                        &mut first_i as *mut _ as *mut c_void,
                        &mut out_ptr as *mut _ as *mut c_void,
                    ];
                    self.cuda
                        .stream
                        .launch(&func, grid, block, self.shared_bytes, args)
                        .expect("cora batch launch");
                }
            }

            let func_wma = self
                .cuda
                .module
                .get_function("cora_wave_batch_wma_from_y_f32")
                .expect("cora_wave_batch_wma_from_y_f32");
            let block2: BlockSize = (self.block_x2, 1, 1).into();
            for (start, len) in CudaCoraWave::grid_y_chunks(self.n_combos) {
                unsafe {
                    let mut y_ptr = self
                        .d_tmp
                        .as_device_ptr()
                        .add(start * self.series_len)
                        .as_raw();
                    let mut sm_ptr = self.d_smooth.as_device_ptr().add(start).as_raw();
                    let mut w0_ptr = self.d_warm0s.as_device_ptr().add(start).as_raw();
                    let mut series_i = self.series_len as i32;
                    let mut n_i = len as i32;
                    let mut out_ptr = self
                        .d_out
                        .as_device_ptr()
                        .add(start * self.series_len)
                        .as_raw();
                    let grid: GridSize = (self.grid_x2, len as u32, 1).into();
                    let args: &mut [*mut c_void] = &mut [
                        &mut y_ptr as *mut _ as *mut c_void,
                        &mut sm_ptr as *mut _ as *mut c_void,
                        &mut w0_ptr as *mut _ as *mut c_void,
                        &mut series_i as *mut _ as *mut c_void,
                        &mut n_i as *mut _ as *mut c_void,
                        &mut out_ptr as *mut _ as *mut c_void,
                    ];
                    self.cuda
                        .stream
                        .launch(&func_wma, grid, block2, 0, args)
                        .expect("cora smooth launch");
                }
            }

            self.cuda.stream.synchronize().expect("cora sync");
        }
    }

    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaCoraWave::new(0).expect("cuda cora_wave");
        let price = gen_series(ONE_SERIES_LEN);
        let sweep = CoraWaveBatchRange {
            period: (16, 16 + PARAM_SWEEP - 1, 1),
            r_multi: (2.0, 2.0, 0.0),
            smooth: true,
        };

        let (
            _combos,
            first_valid,
            series_len,
            max_period,
            periods_i32,
            inv_norms,
            weights_flat,
            smooth_periods,
            warm0s,
        ) = CudaCoraWave::prepare_batch_inputs(&price, &sweep).expect("cora prep batch inputs");
        let n_combos = periods_i32.len();

        let d_prices = DeviceBuffer::from_slice(&price).expect("d_prices");
        let d_weights = DeviceBuffer::from_slice(&weights_flat).expect("d_weights");
        let d_periods = DeviceBuffer::from_slice(&periods_i32).expect("d_periods");
        let d_inv = DeviceBuffer::from_slice(&inv_norms).expect("d_inv");
        let d_smooth = DeviceBuffer::from_slice(&smooth_periods).expect("d_smooth");
        let d_warm0s = DeviceBuffer::from_slice(&warm0s).expect("d_warm0s");
        let n_tmp = n_combos * series_len;
        let d_tmp: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_tmp) }.expect("d_tmp");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_tmp) }.expect("d_out");

        let shared_bytes = (max_period * std::mem::size_of::<f32>()) as u32;
        let func = cuda
            .module
            .get_function("cora_wave_batch_f32")
            .expect("cora_wave_batch_f32");
        let block_x = cuda.pick_block_x(&func, shared_bytes as usize, 256);
        let grid_x = ((series_len as u32) + block_x - 1) / block_x;

        let func_wma = cuda
            .module
            .get_function("cora_wave_batch_wma_from_y_f32")
            .expect("cora_wave_batch_wma_from_y_f32");
        let block_x2 = cuda.pick_block_x(&func_wma, 0, 256);
        let grid_x2 = ((series_len as u32) + block_x2 - 1) / block_x2;

        cuda.stream.synchronize().expect("sync after prep");
        Box::new(CoraWaveBatchDevState {
            cuda,
            d_prices,
            d_weights,
            d_periods,
            d_inv,
            d_smooth,
            d_warm0s,
            series_len,
            n_combos,
            max_period,
            first_valid,
            grid_x: grid_x.max(1),
            block_x,
            shared_bytes,
            grid_x2: grid_x2.max(1),
            block_x2,
            d_tmp,
            d_out,
        })
    }

    struct CoraWaveManyDevState {
        cuda: CudaCoraWave,
        d_prices_tm: DeviceBuffer<f32>,
        d_weights: DeviceBuffer<f32>,
        d_first_valids: DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        period: usize,
        inv_norm: f32,
        grid_x: u32,
        block_x: u32,
        wma_m: i32,
        d_warm0s: DeviceBuffer<i32>,
        grid_x2: u32,
        block_x2: u32,
        d_tmp: DeviceBuffer<f32>,
        d_out_tm: DeviceBuffer<f32>,
    }
    impl CudaBenchState for CoraWaveManyDevState {
        fn launch(&mut self) {
            let func = self
                .cuda
                .module
                .get_function("cora_wave_multi_series_one_param_time_major_f32")
                .expect("cora_wave_ms1p");
            let grid: GridSize = (self.grid_x, self.cols as u32, 1).into();
            let block: BlockSize = (self.block_x, 1, 1).into();
            unsafe {
                let mut prices_ptr = self.d_prices_tm.as_device_ptr().as_raw();
                let mut w_ptr = self.d_weights.as_device_ptr().as_raw();
                let mut period_i = self.period as i32;
                let mut inv = self.inv_norm as f32;
                let mut cols_i = self.cols as i32;
                let mut rows_i = self.rows as i32;
                let mut first_ptr = self.d_first_valids.as_device_ptr().as_raw();
                let mut out_ptr = self.d_tmp.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut prices_ptr as *mut _ as *mut c_void,
                    &mut w_ptr as *mut _ as *mut c_void,
                    &mut period_i as *mut _ as *mut c_void,
                    &mut inv as *mut _ as *mut c_void,
                    &mut cols_i as *mut _ as *mut c_void,
                    &mut rows_i as *mut _ as *mut c_void,
                    &mut first_ptr as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.cuda
                    .stream
                    .launch(&func, grid, block, 0, args)
                    .expect("cora ms1p launch");
            }

            let func_wma = self
                .cuda
                .module
                .get_function("cora_wave_ms1p_wma_time_major_f32")
                .expect("cora ms1p wma");
            let grid2: GridSize = (self.grid_x2, self.cols as u32, 1).into();
            let block2: BlockSize = (self.block_x2, 1, 1).into();
            unsafe {
                let mut y_ptr = self.d_tmp.as_device_ptr().as_raw();
                let mut m_i = self.wma_m as i32;
                let mut cols_i = self.cols as i32;
                let mut rows_i = self.rows as i32;
                let mut w0_ptr = self.d_warm0s.as_device_ptr().as_raw();
                let mut out_ptr = self.d_out_tm.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut y_ptr as *mut _ as *mut c_void,
                    &mut m_i as *mut _ as *mut c_void,
                    &mut cols_i as *mut _ as *mut c_void,
                    &mut rows_i as *mut _ as *mut c_void,
                    &mut w0_ptr as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.cuda
                    .stream
                    .launch(&func_wma, grid2, block2, 0, args)
                    .expect("cora ms1p smooth launch");
            }

            self.cuda.stream.synchronize().expect("cora sync");
        }
    }

    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaCoraWave::new(0).expect("cuda cora_wave");
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_LEN;
        let data_tm = gen_time_major_prices(cols, rows);
        let params = CoraWaveParams {
            period: Some(64),
            r_multi: Some(2.0),
            smooth: Some(true),
        };
        let (first_valids, period, weights, inv_norm, _rows) =
            CudaCoraWave::prepare_many_series_inputs(&data_tm, cols, rows, &params)
                .expect("cora prep many-series");

        let wma_m = ((period as f64).sqrt().round() as i32).max(1);
        let mut warm0s = vec![0i32; cols];
        for s in 0..cols {
            warm0s[s] = first_valids[s] + (period as i32) - 1;
        }

        let d_prices_tm = DeviceBuffer::from_slice(&data_tm).expect("d_prices_tm");
        let d_weights = DeviceBuffer::from_slice(&weights).expect("d_weights");
        let d_first_valids = DeviceBuffer::from_slice(&first_valids).expect("d_first_valids");
        let d_warm0s = DeviceBuffer::from_slice(&warm0s).expect("d_warm0s");
        let n_tmp = cols * rows;
        let d_tmp: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_tmp) }.expect("d_tmp");
        let d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_tmp) }.expect("d_out_tm");

        let func = cuda
            .module
            .get_function("cora_wave_multi_series_one_param_time_major_f32")
            .expect("cora_wave_multi_series_one_param_time_major_f32");
        let block_x = match cuda.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => block_x,
            _ => func
                .suggested_launch_configuration(0, BlockSize::xyz(0, 0, 0))
                .map(|(_, bx)| bx)
                .unwrap_or(256)
                .max(64)
                .min(1024),
        };
        let grid_x = ((rows as u32) + block_x - 1) / block_x;

        let func_wma = cuda
            .module
            .get_function("cora_wave_ms1p_wma_time_major_f32")
            .expect("cora_wave_ms1p_wma_time_major_f32");
        let block_x2 = match cuda.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => block_x,
            _ => func_wma
                .suggested_launch_configuration(0, BlockSize::xyz(0, 0, 0))
                .map(|(_, bx)| bx)
                .unwrap_or(256)
                .max(64)
                .min(1024),
        };
        let grid_x2 = ((rows as u32) + block_x2 - 1) / block_x2;

        cuda.stream.synchronize().expect("sync after prep");
        Box::new(CoraWaveManyDevState {
            cuda,
            d_prices_tm,
            d_weights,
            d_first_valids,
            cols,
            rows,
            period,
            inv_norm,
            grid_x: grid_x.max(1),
            block_x,
            wma_m,
            d_warm0s,
            grid_x2: grid_x2.max(1),
            block_x2,
            d_tmp,
            d_out_tm,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "cora_wave",
                "one_series_many_params",
                "cora_wave_cuda_batch_dev",
                "1m_x_250",
                prep_one_series_many_params,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "cora_wave",
                "many_series_one_param",
                "cora_wave_cuda_many_series_one_param",
                "250x1m",
                prep_many_series_one_param,
            )
            .with_sample_size(5)
            .with_mem_required(bytes_many_series_one_param()),
        ]
    }
}
