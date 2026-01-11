//! CUDA scaffolding for the Jurik Moving Average (JMA) kernels.
//!
//! Parity goals (mirrors ALMA wrapper):
//! - Policy enums for explicit kernel selection (record last selection)
//! - NON_BLOCKING stream, PTX JIT with DetermineTargetFromContext + O2 fallback
//! - VRAM estimation and early-fail guard; chunk batch grid to avoid limits
//! - Public device entry points that accept host slices or DeviceBuffers
//!
//! Math: JMA is a recurrence (time-marching) — one thread per combo/series
//! performs a sequential scan over time. There are no tiled/2x variants; the
//! policy types exist for API symmetry and debugging.

#![cfg(feature = "cuda")]

use super::DeviceArrayF32;
use std::sync::Arc;
use crate::indicators::moving_averages::jma::{expand_grid_jma, JmaBatchRange, JmaParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaJmaError {
    #[error(transparent)]
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
pub enum BatchThreadsPerOutput {
    One,
    Two,
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    Plain {
        block_x: u32,
    },
    Tiled {
        tile: u32,
        per_thread: BatchThreadsPerOutput,
    }, 
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 },
    Tiled2D { tx: u32, ty: u32 }, 
}

#[derive(Clone, Copy, Debug)]
pub struct CudaJmaPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaJmaPolicy {
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

pub struct CudaJma {
    module: Module,
    stream: Stream,
    _context: Arc<Context>,
    device_id: u32,
    policy: CudaJmaPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

/// VRAM-backed array handle for JMA that carries a context guard
/// VRAM-backed array handle for JMA that carries a context guard
pub struct DeviceArrayF32Jma {
    pub buf: DeviceBuffer<f32>,
    pub rows: usize,
    pub cols: usize,
    pub ctx: Arc<Context>,
    pub device_id: u32,
}
impl DeviceArrayF32Jma {
    #[inline]
    pub fn device_ptr(&self) -> u64 {
        self.buf.as_device_ptr().as_raw() as u64
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

impl CudaJma {
    pub fn new(device_id: usize) -> Result<Self, CudaJmaError> {
        cust::init(CudaFlags::empty())?;

        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/jma_kernel.ptx"));
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
            _context: context,
            device_id: device_id as u32,
            policy: CudaJmaPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn new_with_policy(device_id: usize, policy: CudaJmaPolicy) -> Result<Self, CudaJmaError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }
    pub fn set_policy(&mut self, policy: CudaJmaPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaJmaPolicy {
        &self.policy
    }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch
    }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many
    }

    /// Expose synchronize for benches/tests
    pub fn synchronize(&self) -> Result<(), CudaJmaError> {
        self.stream.synchronize().map_err(CudaJmaError::from)
    }

    
    #[inline]
    fn choose_block_x_auto(n: usize) -> u32 {
        
        
        let hard = 256u32;
        if n >= hard as usize {
            hard
        } else {
            let t = n.next_power_of_two().max(32);
            t.min(hard as usize) as u32
        }
    }

    #[inline]
    fn resolve_batch_block_x(&self, needed: usize) -> u32 {
        match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => block_x.clamp(32, 1024),
            _ => {
                if let Some(v) = std::env::var("JMA_BLOCK_X")
                    .ok()
                    .and_then(|s| s.parse::<u32>().ok())
                {
                    return v.clamp(32, 1024);
                }
                
                
                
                let bx = Self::choose_block_x_auto(needed);
                if needed > 32 && (bx as usize) >= needed {
                    32
                } else {
                    bx
                }
            }
        }
    }

    #[inline]
    fn resolve_many_block_x(&self, needed: usize) -> u32 {
        match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => block_x.clamp(32, 1024),
            _ => {
                if let Some(v) = std::env::var("JMA_BLOCK_X")
                    .ok()
                    .and_then(|s| s.parse::<u32>().ok())
                {
                    return v.clamp(32, 1024);
                }
                Self::choose_block_x_auto(needed)
            }
        }
    }

    #[inline]
    fn ceil_div_u32(n: usize, d: u32) -> u32 {
        ((n as u64 + d as u64 - 1) / d as u64) as u32
    }

    
    #[inline]
    fn grid_chunks_v2(n: usize, block_x: u32) -> impl Iterator<Item = (usize, usize)> {
        const MAX_GRID_X: usize = i32::MAX as usize; 
        let max_elems = MAX_GRID_X.saturating_mul(block_x as usize);
        (0..n).step_by(max_elems).map(move |start| {
            let len = (n - start).min(max_elems);
            (start, len)
        })
    }

    pub fn jma_batch_dev(
        &self,
        prices: &[f32],
        sweep: &JmaBatchRange,
    ) -> Result<DeviceArrayF32Jma, CudaJmaError> {
        let inputs = Self::prepare_batch_inputs(prices, sweep)?;
        self.run_batch_kernel(prices, &inputs)
    }

    pub fn jma_batch_into_host_f32(
        &self,
        prices: &[f32],
        sweep: &JmaBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<JmaParams>), CudaJmaError> {
        let inputs = Self::prepare_batch_inputs(prices, sweep)?;
        let expected = inputs.series_len * inputs.combos.len();
        if out.len() != expected {
            return Err(CudaJmaError::InvalidInput(format!(
                "output slice wrong length: got {}, expected {}",
                out.len(),
                expected
            )));
        }

        let arr = self.run_batch_kernel(prices, &inputs)?;
        arr.buf.copy_to(out)?;
        Ok((arr.rows, arr.cols, inputs.combos))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn jma_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_alphas: &DeviceBuffer<f32>,
        d_one_minus_betas: &DeviceBuffer<f32>,
        d_phase_ratios: &DeviceBuffer<f32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaJmaError> {
        if series_len == 0 || n_combos == 0 {
            return Err(CudaJmaError::InvalidInput(
                "series_len and n_combos must be positive".into(),
            ));
        }
        if series_len > i32::MAX as usize || n_combos > i32::MAX as usize {
            return Err(CudaJmaError::InvalidInput(
                "arguments exceed kernel limits".into(),
            ));
        }

        self.launch_batch_kernel(
            d_prices,
            d_alphas,
            d_one_minus_betas,
            d_phase_ratios,
            series_len,
            n_combos,
            first_valid,
            d_out,
        )
    }

    pub fn jma_many_series_one_param_time_major_dev(
        &self,
        prices_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &JmaParams,
    ) -> Result<DeviceArrayF32Jma, CudaJmaError> {
        let prepared = Self::prepare_many_series_inputs(prices_tm_f32, cols, rows, params)?;
        let consts = Self::compute_params_consts(params)?;
        self.run_many_series_kernel(prices_tm_f32, cols, rows, &prepared, &consts)
    }

    pub fn jma_many_series_one_param_time_major_into_host_f32(
        &self,
        prices_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &JmaParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaJmaError> {
        if out_tm.len() != cols * rows {
            return Err(CudaJmaError::InvalidInput(format!(
                "output slice wrong length: got {}, expected {}",
                out_tm.len(),
                cols * rows
            )));
        }

        let prepared = Self::prepare_many_series_inputs(prices_tm_f32, cols, rows, params)?;
        let consts = Self::compute_params_consts(params)?;
        let arr = self.run_many_series_kernel(prices_tm_f32, cols, rows, &prepared, &consts)?;
        arr.buf
            .copy_to(out_tm)
            .map_err(|e| CudaJmaError::Cuda(e))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn jma_many_series_one_param_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        alpha: f32,
        one_minus_beta: f32,
        phase_ratio: f32,
        num_series: usize,
        series_len: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaJmaError> {
        if num_series == 0 || series_len == 0 {
            return Err(CudaJmaError::InvalidInput(
                "num_series and series_len must be positive".into(),
            ));
        }
        if num_series > i32::MAX as usize || series_len > i32::MAX as usize {
            return Err(CudaJmaError::InvalidInput(
                "arguments exceed kernel limits".into(),
            ));
        }

        let block_x = self.resolve_many_block_x(num_series);
        self.launch_many_series_kernel(
            d_prices_tm,
            alpha,
            one_minus_beta,
            phase_ratio,
            num_series,
            series_len,
            d_first_valids,
            d_out_tm,
            block_x,
        )
    }

    fn run_batch_kernel(
        &self,
        prices: &[f32],
        inputs: &BatchInputs,
    ) -> Result<DeviceArrayF32Jma, CudaJmaError> {
        let n_combos = inputs.combos.len();
        let series_len = inputs.series_len;

        
        let sz_f32 = std::mem::size_of::<f32>();
        let prices_bytes = series_len
            .checked_mul(sz_f32)
            .ok_or_else(|| CudaJmaError::InvalidInput("byte size overflow".into()))?;
        let alpha_bytes = n_combos
            .checked_mul(sz_f32)
            .ok_or_else(|| CudaJmaError::InvalidInput("byte size overflow".into()))?;
        let beta_bytes = n_combos
            .checked_mul(sz_f32)
            .ok_or_else(|| CudaJmaError::InvalidInput("byte size overflow".into()))?;
        let phase_bytes = n_combos
            .checked_mul(sz_f32)
            .ok_or_else(|| CudaJmaError::InvalidInput("byte size overflow".into()))?;
        let out_elems = n_combos
            .checked_mul(series_len)
            .ok_or_else(|| CudaJmaError::InvalidInput("rows * cols overflow".into()))?;
        let out_bytes = out_elems
            .checked_mul(sz_f32)
            .ok_or_else(|| CudaJmaError::InvalidInput("byte size overflow".into()))?;
        let required = prices_bytes
            .checked_add(alpha_bytes)
            .and_then(|v| v.checked_add(beta_bytes))
            .and_then(|v| v.checked_add(phase_bytes))
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| CudaJmaError::InvalidInput("byte size overflow".into()))?;
        let headroom = 64 * 1024 * 1024; 

        if !Self::will_fit(required, headroom) {
            let (free, _) = Self::device_mem_info().unwrap_or((0, 0));
            return Err(CudaJmaError::OutOfMemory { required, free, headroom });
        }

        
        let d_prices = unsafe { DeviceBuffer::from_slice_async(prices, &self.stream) }
            .map_err(|e| CudaJmaError::Cuda(e))?;
        let d_alphas = unsafe { DeviceBuffer::from_slice_async(&inputs.alphas, &self.stream) }
            .map_err(|e| CudaJmaError::Cuda(e))?;
        let d_one_minus_betas =
            unsafe { DeviceBuffer::from_slice_async(&inputs.one_minus_betas, &self.stream) }
                .map_err(|e| CudaJmaError::Cuda(e))?;
        let d_phase_ratios =
            unsafe { DeviceBuffer::from_slice_async(&inputs.phase_ratios, &self.stream) }
                .map_err(|e| CudaJmaError::Cuda(e))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(out_elems, &self.stream) }
                .map_err(|e| CudaJmaError::Cuda(e))?;

        
        let block_x = self.resolve_batch_block_x(n_combos);
        unsafe {
            let this = self as *const _ as *mut CudaJma;
            (*this).last_batch = Some(BatchKernelSelected::Plain { block_x });
        }
        self.maybe_log_batch_debug();

        
        for (start, len) in Self::grid_chunks_v2(n_combos, block_x) {
            self.launch_batch_kernel_chunk(
                &d_prices,
                &d_alphas,
                &d_one_minus_betas,
                &d_phase_ratios,
                series_len,
                start,
                len,
                inputs.first_valid,
                &mut d_out,
                block_x,
            )?;
        }

        self.stream.synchronize().map_err(CudaJmaError::from)?;

        Ok(DeviceArrayF32Jma { buf: d_out, rows: n_combos, cols: series_len, ctx: self._context.clone(), device_id: self.device_id })
    }

    fn run_many_series_kernel(
        &self,
        prices_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        prepared: &ManySeriesInputs,
        consts: &JmaConsts,
    ) -> Result<DeviceArrayF32Jma, CudaJmaError> {
        let num_series = cols;
        let series_len = rows;

        let prices_bytes = prices_tm_f32.len() * std::mem::size_of::<f32>();
        let first_valid_bytes = prepared.first_valids.len() * std::mem::size_of::<i32>();
        let out_bytes = prices_tm_f32.len() * std::mem::size_of::<f32>();
        let required = prices_bytes + first_valid_bytes + out_bytes;
        let headroom = 32 * 1024 * 1024; 

        if !Self::will_fit(required, headroom) {
            let (free, _) = Self::device_mem_info().unwrap_or((0, 0));
            return Err(CudaJmaError::OutOfMemory { required, free, headroom });
        }

        let d_prices_tm = unsafe { DeviceBuffer::from_slice_async(prices_tm_f32, &self.stream) }?;
        let d_first_valids =
            unsafe { DeviceBuffer::from_slice_async(&prepared.first_valids, &self.stream) }?;
        let mut d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(prices_tm_f32.len(), &self.stream) }?;

        
        let block_x = self.resolve_many_block_x(num_series);
        unsafe {
            let this = self as *const _ as *mut CudaJma;
            (*this).last_many = Some(ManySeriesKernelSelected::OneD { block_x });
        }
        self.maybe_log_many_debug();

        self.launch_many_series_kernel(
            &d_prices_tm,
            consts.alpha,
            consts.one_minus_beta,
            consts.phase_ratio,
            num_series,
            series_len,
            &d_first_valids,
            &mut d_out_tm,
            block_x,
        )?;

        self.stream.synchronize()?;

        Ok(DeviceArrayF32Jma { buf: d_out_tm, rows: series_len, cols: num_series, ctx: self._context.clone(), device_id: self.device_id })
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_alphas: &DeviceBuffer<f32>,
        d_one_minus_betas: &DeviceBuffer<f32>,
        d_phase_ratios: &DeviceBuffer<f32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaJmaError> {
        
        let block_x = self.resolve_batch_block_x(n_combos);
        self.launch_batch_kernel_chunk(
            d_prices,
            d_alphas,
            d_one_minus_betas,
            d_phase_ratios,
            series_len,
            0,
            n_combos,
            first_valid,
            d_out,
            block_x,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_batch_kernel_chunk(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_alphas: &DeviceBuffer<f32>,
        d_one_minus_betas: &DeviceBuffer<f32>,
        d_phase_ratios: &DeviceBuffer<f32>,
        series_len: usize,
        start_combo: usize,
        len_combos: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
        block_x: u32,
    ) -> Result<(), CudaJmaError> {
        let func = self
            .module
            .get_function("jma_batch_f32")
            .map_err(|_| CudaJmaError::MissingKernelSymbol { name: "jma_batch_f32" })?;

        let grid_x = Self::ceil_div_u32(len_combos, block_x);
        let grid: GridSize = (grid_x, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        if block_x > 1024 || grid_x == 0 {
            return Err(CudaJmaError::LaunchConfigTooLarge { gx: grid_x, gy: 1, gz: 1, bx: block_x, by: 1, bz: 1 });
        }

        let out_offset = start_combo
            .checked_mul(series_len)
            .ok_or_else(|| CudaJmaError::InvalidInput("output offset overflow".into()))?;

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            
            let mut alphas_ptr = d_alphas.as_device_ptr().add(start_combo).as_raw();
            let mut beta_ptr = d_one_minus_betas.as_device_ptr().add(start_combo).as_raw();
            let mut phase_ptr = d_phase_ratios.as_device_ptr().add(start_combo).as_raw();
            let mut series_len_i = series_len as i32;
            let mut combos_i = len_combos as i32;
            let mut first_valid_i = first_valid as i32;
            
            let mut out_ptr = d_out.as_device_ptr().add(out_offset).as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut alphas_ptr as *mut _ as *mut c_void,
                &mut beta_ptr as *mut _ as *mut c_void,
                &mut phase_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        alpha: f32,
        one_minus_beta: f32,
        phase_ratio: f32,
        num_series: usize,
        series_len: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
        block_x: u32,
    ) -> Result<(), CudaJmaError> {
        let func = self
            .module
            .get_function("jma_many_series_one_param_f32")
            .map_err(|_| CudaJmaError::MissingKernelSymbol { name: "jma_many_series_one_param_f32" })?;

        let grid_x = Self::ceil_div_u32(num_series, block_x);
        let grid: GridSize = (grid_x, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        if block_x > 1024 || grid_x == 0 {
            return Err(CudaJmaError::LaunchConfigTooLarge { gx: grid_x, gy: 1, gz: 1, bx: block_x, by: 1, bz: 1 });
        }

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut alpha_f = alpha;
            let mut one_minus_beta_f = one_minus_beta;
            let mut phase_ratio_f = phase_ratio;
            let mut num_series_i = num_series as i32;
            let mut series_len_i = series_len as i32;
            let mut first_valids_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut alpha_f as *mut _ as *mut c_void,
                &mut one_minus_beta_f as *mut _ as *mut c_void,
                &mut phase_ratio_f as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut first_valids_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        Ok(())
    }

    fn prepare_batch_inputs(
        prices: &[f32],
        sweep: &JmaBatchRange,
    ) -> Result<BatchInputs, CudaJmaError> {
        if prices.is_empty() {
            return Err(CudaJmaError::InvalidInput("empty price series".into()));
        }

        let combos = expand_grid_jma(sweep);
        if combos.is_empty() {
            return Err(CudaJmaError::InvalidInput(
                "no parameter combinations provided".into(),
            ));
        }

        let series_len = prices.len();
        let first_valid = prices
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaJmaError::InvalidInput("all price values are NaN".into()))?;

        let mut alphas = Vec::with_capacity(combos.len());
        let mut one_minus_betas = Vec::with_capacity(combos.len());
        let mut phase_ratios = Vec::with_capacity(combos.len());

        let mut max_period = 0usize;
        for prm in &combos {
            let period = prm.period.unwrap_or(0);
            let phase = prm.phase.unwrap_or(50.0);
            let power = prm.power.unwrap_or(2);
            if period == 0 {
                return Err(CudaJmaError::InvalidInput("period must be positive".into()));
            }
            if period > i32::MAX as usize {
                return Err(CudaJmaError::InvalidInput(
                    "period exceeds kernel limits".into(),
                ));
            }
            if !phase.is_finite() {
                return Err(CudaJmaError::InvalidInput(format!(
                    "phase must be finite (got {})",
                    phase
                )));
            }
            let consts = Self::compute_consts(period, phase, power)?;
            alphas.push(consts.alpha);
            one_minus_betas.push(consts.one_minus_beta);
            phase_ratios.push(consts.phase_ratio);
            max_period = max_period.max(period);
        }

        if series_len - first_valid < max_period {
            return Err(CudaJmaError::InvalidInput(format!(
                "not enough valid data (needed >= {}, valid = {})",
                max_period,
                series_len - first_valid
            )));
        }

        Ok(BatchInputs {
            combos,
            alphas,
            one_minus_betas,
            phase_ratios,
            first_valid,
            series_len,
        })
    }

    fn prepare_many_series_inputs(
        prices_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &JmaParams,
    ) -> Result<ManySeriesInputs, CudaJmaError> {
        if cols == 0 || rows == 0 {
            return Err(CudaJmaError::InvalidInput(
                "matrix dimensions must be positive".into(),
            ));
        }
        if prices_tm_f32.len() != cols * rows {
            return Err(CudaJmaError::InvalidInput("matrix shape mismatch".into()));
        }

        let period = params.period.unwrap_or(0);
        if period == 0 {
            return Err(CudaJmaError::InvalidInput("period must be positive".into()));
        }
        if period > i32::MAX as usize {
            return Err(CudaJmaError::InvalidInput(
                "period exceeds kernel limits".into(),
            ));
        }
        let phase = params.phase.unwrap_or(50.0);
        if !phase.is_finite() {
            return Err(CudaJmaError::InvalidInput(format!(
                "phase must be finite (got {})",
                phase
            )));
        }

        let mut first_valids = vec![0i32; cols];
        for series_idx in 0..cols {
            let mut fv = None;
            for row in 0..rows {
                let idx = row * cols + series_idx;
                if !prices_tm_f32[idx].is_nan() {
                    fv = Some(row);
                    break;
                }
            }
            let first = fv.ok_or_else(|| {
                CudaJmaError::InvalidInput(format!(
                    "series {} contains only NaN values",
                    series_idx
                ))
            })?;
            if rows - first < period {
                return Err(CudaJmaError::InvalidInput(format!(
                    "series {} lacks data: needed >= {}, valid = {}",
                    series_idx,
                    period,
                    rows - first
                )));
            }
            first_valids[series_idx] = first as i32;
        }

        Ok(ManySeriesInputs { first_valids })
    }

    fn compute_params_consts(params: &JmaParams) -> Result<JmaConsts, CudaJmaError> {
        let period = params.period.unwrap_or(0);
        let phase = params.phase.unwrap_or(50.0);
        let power = params.power.unwrap_or(2);
        if period == 0 {
            return Err(CudaJmaError::InvalidInput("period must be positive".into()));
        }
        if !phase.is_finite() {
            return Err(CudaJmaError::InvalidInput(format!(
                "phase must be finite (got {})",
                phase
            )));
        }
        Self::compute_consts(period, phase, power)
    }

    fn compute_consts(period: usize, phase: f64, power: u32) -> Result<JmaConsts, CudaJmaError> {
        let phase_ratio = if phase < -100.0 {
            0.5
        } else if phase > 100.0 {
            2.5
        } else {
            phase / 100.0 + 1.5
        };

        let numerator = 0.45 * (period as f64 - 1.0);
        let denominator = numerator + 2.0;
        if denominator.abs() < f64::EPSILON {
            return Err(CudaJmaError::InvalidInput(
                "invalid period leading to zero denominator in beta".into(),
            ));
        }
        let beta = numerator / denominator;
        let alpha = beta.powi(power as i32);
        let one_minus_beta = 1.0 - beta;

        Ok(JmaConsts {
            alpha: alpha as f32,
            one_minus_beta: one_minus_beta as f32,
            phase_ratio: phase_ratio as f32,
        })
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
        if let Some((free, _total)) = Self::device_mem_info() {
            required_bytes.saturating_add(headroom_bytes) <= free
        } else {
            true
        }
    }

    #[inline]
    fn grid_chunks(n: usize) -> impl Iterator<Item = (usize, usize)> {
        const MAX: usize = 65_535; 
        (0..n).step_by(MAX).map(move |start| {
            let len = (n - start).min(MAX);
            (start, len)
        })
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
                    eprintln!("[DEBUG] JMA batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaJma)).debug_batch_logged = true;
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
                    eprintln!("[DEBUG] JMA many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaJma)).debug_many_logged = true;
                }
            }
        }
    }
}



pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};
    use crate::indicators::moving_averages::jma::{JmaBatchRange, JmaParams};

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

    struct JmaBatchDevState {
        cuda: CudaJma,
        d_prices: DeviceBuffer<f32>,
        d_alphas: DeviceBuffer<f32>,
        d_one_minus_betas: DeviceBuffer<f32>,
        d_phase_ratios: DeviceBuffer<f32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out: DeviceBuffer<f32>,
    }
    impl CudaBenchState for JmaBatchDevState {
        fn launch(&mut self) {
            self.cuda
                .jma_batch_device(
                    &self.d_prices,
                    &self.d_alphas,
                    &self.d_one_minus_betas,
                    &self.d_phase_ratios,
                    self.series_len,
                    self.n_combos,
                    self.first_valid,
                    &mut self.d_out,
                )
                .expect("jma batch kernel");
            self.cuda.stream.synchronize().expect("jma sync");
        }
    }

    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaJma::new(0).expect("cuda jma");
        let price = gen_series(ONE_SERIES_LEN);
        let sweep = JmaBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1),
            phase: (50.0, 50.0, 0.0),
            power: (2, 2, 0),
        };

        let inputs = CudaJma::prepare_batch_inputs(&price, &sweep).expect("jma prepare batch");
        let n_combos = inputs.combos.len();

        let d_prices = DeviceBuffer::from_slice(&price).expect("d_prices");
        let d_alphas = DeviceBuffer::from_slice(&inputs.alphas).expect("d_alphas");
        let d_one_minus_betas =
            DeviceBuffer::from_slice(&inputs.one_minus_betas).expect("d_one_minus_betas");
        let d_phase_ratios = DeviceBuffer::from_slice(&inputs.phase_ratios).expect("d_phase_ratios");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(inputs.series_len * n_combos) }.expect("d_out");

        cuda.stream.synchronize().expect("sync after prep");
        Box::new(JmaBatchDevState {
            cuda,
            d_prices,
            d_alphas,
            d_one_minus_betas,
            d_phase_ratios,
            series_len: inputs.series_len,
            n_combos,
            first_valid: inputs.first_valid,
            d_out,
        })
    }

    struct JmaManyDevState {
        cuda: CudaJma,
        d_prices_tm: DeviceBuffer<f32>,
        d_first_valids: DeviceBuffer<i32>,
        alpha: f32,
        one_minus_beta: f32,
        phase_ratio: f32,
        cols: usize,
        rows: usize,
        d_out_tm: DeviceBuffer<f32>,
    }
    impl CudaBenchState for JmaManyDevState {
        fn launch(&mut self) {
            self.cuda
                .jma_many_series_one_param_device(
                    &self.d_prices_tm,
                    self.alpha,
                    self.one_minus_beta,
                    self.phase_ratio,
                    self.cols,
                    self.rows,
                    &self.d_first_valids,
                    &mut self.d_out_tm,
                )
                .expect("jma many-series kernel");
            self.cuda.stream.synchronize().expect("jma sync");
        }
    }

    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaJma::new(0).expect("cuda jma");
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_LEN;
        let data_tm = gen_time_major_prices(cols, rows);
        let params = JmaParams {
            period: Some(64),
            phase: Some(50.0),
            power: Some(2),
        };

        let prepared = CudaJma::prepare_many_series_inputs(&data_tm, cols, rows, &params)
            .expect("jma prepare many-series");
        let consts = CudaJma::compute_params_consts(&params).expect("jma compute consts");

        let d_prices_tm = DeviceBuffer::from_slice(&data_tm).expect("d_prices_tm");
        let d_first_valids = DeviceBuffer::from_slice(&prepared.first_valids).expect("d_first_valids");
        let d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(cols * rows) }.expect("d_out_tm");

        cuda.stream.synchronize().expect("sync after prep");
        Box::new(JmaManyDevState {
            cuda,
            d_prices_tm,
            d_first_valids,
            alpha: consts.alpha,
            one_minus_beta: consts.one_minus_beta,
            phase_ratio: consts.phase_ratio,
            cols,
            rows,
            d_out_tm,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "jma",
                "one_series_many_params",
                "jma_cuda_batch_dev",
                "1m_x_250",
                prep_one_series_many_params,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "jma",
                "many_series_one_param",
                "jma_cuda_many_series_one_param",
                "250x1m",
                prep_many_series_one_param,
            )
            .with_sample_size(5)
            .with_mem_required(bytes_many_series_one_param()),
        ]
    }
}

struct BatchInputs {
    combos: Vec<JmaParams>,
    alphas: Vec<f32>,
    one_minus_betas: Vec<f32>,
    phase_ratios: Vec<f32>,
    first_valid: usize,
    series_len: usize,
}

struct ManySeriesInputs {
    first_valids: Vec<i32>,
}

struct JmaConsts {
    alpha: f32,
    one_minus_beta: f32,
    phase_ratio: f32,
}
