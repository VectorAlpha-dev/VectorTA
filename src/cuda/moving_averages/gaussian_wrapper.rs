//! CUDA scaffolding for the Gaussian moving average kernels.
//!
//! Parity goals with ALMA wrapper:
//! - PTX JIT with DetermineTargetFromContext and O2 fallback
//! - Non-blocking stream
//! - VRAM estimation with optional mem-check toggle
//! - Policy + selection introspection (Batch/ManySeries) with BENCH_DEBUG logging
//! - Public device + host helpers returning `DeviceArrayF32`
//!
//! Math note: Gaussian here is implemented as a cascaded one‑pole recurrence
//! (time-marching) for numerical parity with the scalar reference. Coefficients
//! are precomputed on host per parameter set and uploaded once.

#![cfg(feature = "cuda")]

use super::DeviceArrayF32;
use crate::indicators::moving_averages::gaussian::{GaussianBatchRange, GaussianParams};
use cust::context::{CacheConfig, Context};
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use cust::sys as cu;
use std::env;
use std::ffi::{c_void, CStr};
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

const COEFF_STRIDE: usize = 5; // coefficient slots per combo (max poles = 4)

use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaGaussianError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("out of memory: required={required} bytes, free={free} bytes, headroom={headroom} bytes")]
    OutOfMemory { required: usize, free: usize, headroom: usize },
    #[error("launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("not implemented")]
    NotImplemented,
}

// -------- Kernel selection policy (mirror ALMA style; single plain path used) --------

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
pub struct CudaGaussianPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

impl Default for CudaGaussianPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

// -------- Introspection (selected kernel) --------

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected {
    Plain { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}

pub struct CudaGaussian {
    module: Module,
    stream: Stream,
    _context: Arc<Context>,
    device_id: u32,
    policy: CudaGaussianPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaGaussian {
    pub fn new(device_id: usize) -> Result<Self, CudaGaussianError> {
        cust::init(CudaFlags::empty())?;

        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/gaussian_kernel.ptx"));
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
            policy: CudaGaussianPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    /// Create with explicit policy (tests/benches).
    pub fn new_with_policy(
        device_id: usize,
        policy: CudaGaussianPolicy,
    ) -> Result<Self, CudaGaussianError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }
    pub fn set_policy(&mut self, policy: CudaGaussianPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaGaussianPolicy {
        &self.policy
    }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch
    }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many
    }
    pub fn synchronize(&self) -> Result<(), CudaGaussianError> { self.stream.synchronize()?; Ok(()) }

    pub fn stream_handle(&self) -> usize { self.stream.as_inner() as usize }
    pub fn context_arc(&self) -> Arc<Context> { self._context.clone() }
    pub fn device_id(&self) -> u32 { self.device_id }

    pub fn gaussian_batch_dev(
        &self,
        prices: &[f32],
        sweep: &GaussianBatchRange,
    ) -> Result<DeviceArrayF32, CudaGaussianError> {
        let inputs = Self::prepare_batch_inputs(prices, sweep)?;
        self.run_batch_kernel(prices, &inputs)
    }

    pub fn gaussian_batch_into_host_f32(
        &self,
        prices: &[f32],
        sweep: &GaussianBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<GaussianParams>), CudaGaussianError> {
        let inputs = Self::prepare_batch_inputs(prices, sweep)?;
        let expected = inputs.series_len * inputs.combos.len();
        if out.len() != expected {
            return Err(CudaGaussianError::InvalidInput(format!(
                "output slice length mismatch: got {}, expected {}",
                out.len(),
                expected
            )));
        }

        let arr = self.run_batch_kernel(prices, &inputs)?;

        // Use pinned host memory + async copy for better D2H throughput.
        let mut pinned: LockedBuffer<f32> = unsafe { LockedBuffer::uninitialized(arr.len())? };
        unsafe { arr.buf.async_copy_to(pinned.as_mut_slice(), &self.stream)?; }
        self.stream.synchronize()?;
        out.copy_from_slice(pinned.as_slice());
        let BatchInputs { combos, .. } = inputs;
        Ok((arr.rows, arr.cols, combos))
    }

    pub fn gaussian_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_poles: &DeviceBuffer<i32>,
        d_coeffs: &DeviceBuffer<f32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaGaussianError> {
        if series_len == 0 || n_combos == 0 {
            return Err(CudaGaussianError::InvalidInput(
                "series_len and n_combos must be positive".into(),
            ));
        }
        if series_len > i32::MAX as usize || n_combos > i32::MAX as usize {
            return Err(CudaGaussianError::InvalidInput(
                "arguments exceed kernel launch limits".into(),
            ));
        }

        self.launch_batch_kernel(
            d_prices,
            d_periods,
            d_poles,
            d_coeffs,
            series_len,
            n_combos,
            first_valid,
            d_out,
        )
    }

    pub fn gaussian_many_series_one_param_time_major_dev(
        &self,
        prices_tm: &[f32],
        cols: usize,
        rows: usize,
        params: &GaussianParams,
    ) -> Result<DeviceArrayF32, CudaGaussianError> {
        let prepared = Self::prepare_many_series_inputs(prices_tm, cols, rows, params)?;
        self.run_many_series_kernel(prices_tm, cols, rows, params, &prepared)
    }

    pub fn gaussian_many_series_one_param_time_major_into_host_f32(
        &self,
        prices_tm: &[f32],
        cols: usize,
        rows: usize,
        params: &GaussianParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaGaussianError> {
        if out_tm.len() != prices_tm.len() {
            return Err(CudaGaussianError::InvalidInput(format!(
                "output slice length mismatch: got {}, expected {}",
                out_tm.len(),
                prices_tm.len()
            )));
        }

        let prepared = Self::prepare_many_series_inputs(prices_tm, cols, rows, params)?;
        let arr = self.run_many_series_kernel(prices_tm, cols, rows, params, &prepared)?;
        arr.buf
            .copy_to(out_tm)
            .map_err(|e| CudaGaussianError::Cuda(e))?;
        Ok(())
    }

    pub fn gaussian_many_series_one_param_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_coeffs: &DeviceBuffer<f32>,
        period: usize,
        poles: usize,
        num_series: usize,
        series_len: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaGaussianError> {
        if num_series == 0 || series_len == 0 {
            return Err(CudaGaussianError::InvalidInput(
                "num_series and series_len must be positive".into(),
            ));
        }
        if period < 2 || !(1..=4).contains(&poles) {
            return Err(CudaGaussianError::InvalidInput(
                "period >= 2 and poles within 1..=4 are required".into(),
            ));
        }
        if num_series > i32::MAX as usize || series_len > i32::MAX as usize {
            return Err(CudaGaussianError::InvalidInput(
                "dimensions exceed kernel launch limits".into(),
            ));
        }

        self.launch_many_series_kernel(
            d_prices_tm,
            d_coeffs,
            period,
            poles,
            num_series,
            series_len,
            d_first_valids,
            d_out_tm,
        )
    }

    fn run_batch_kernel(
        &self,
        prices: &[f32],
        inputs: &BatchInputs,
    ) -> Result<DeviceArrayF32, CudaGaussianError> {
        let n_combos = inputs.combos.len();
        let sz_f32 = std::mem::size_of::<f32>();
        let sz_i32 = std::mem::size_of::<i32>();
        let price_bytes = prices.len().checked_mul(sz_f32).ok_or_else(|| CudaGaussianError::InvalidInput("byte size overflow".into()))?;
        let period_bytes = inputs.periods.len().checked_mul(sz_i32).ok_or_else(|| CudaGaussianError::InvalidInput("byte size overflow".into()))?;
        let pole_bytes = inputs.poles.len().checked_mul(sz_i32).ok_or_else(|| CudaGaussianError::InvalidInput("byte size overflow".into()))?;
        let coeff_bytes = inputs.coeffs.len().checked_mul(sz_f32).ok_or_else(|| CudaGaussianError::InvalidInput("byte size overflow".into()))?;
        let out_elems = inputs.series_len.checked_mul(n_combos).ok_or_else(|| CudaGaussianError::InvalidInput("elem count overflow".into()))?;
        let out_bytes = out_elems.checked_mul(sz_f32).ok_or_else(|| CudaGaussianError::InvalidInput("byte size overflow".into()))?;
        let required = price_bytes
            .checked_add(period_bytes).and_then(|v| v.checked_add(pole_bytes))
            .and_then(|v| v.checked_add(coeff_bytes)).and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| CudaGaussianError::InvalidInput("byte size overflow".into()))?;
        let headroom = 64 * 1024 * 1024; // 64 MiB safety margin
        Self::will_fit_checked(required, headroom)?;

        let d_prices = DeviceBuffer::from_slice(prices)?;
        let d_periods = DeviceBuffer::from_slice(&inputs.periods)?;
        let d_poles = DeviceBuffer::from_slice(&inputs.poles)?;
        let d_coeffs = DeviceBuffer::from_slice(&inputs.coeffs)?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(out_elems) }?;

        self.launch_batch_kernel(
            &d_prices,
            &d_periods,
            &d_poles,
            &d_coeffs,
            inputs.series_len,
            n_combos,
            inputs.first_valid,
            &mut d_out,
        )?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: inputs.series_len,
        })
    }

    fn run_many_series_kernel(
        &self,
        prices_tm: &[f32],
        cols: usize,
        rows: usize,
        params: &GaussianParams,
        prepared: &ManySeriesInputs,
    ) -> Result<DeviceArrayF32, CudaGaussianError> {
        let price_bytes = prices_tm.len() * std::mem::size_of::<f32>();
        let coeff_bytes = prepared.coeffs.len() * std::mem::size_of::<f32>();
        let fv_bytes = prepared.first_valids.len() * std::mem::size_of::<i32>();
        let out_bytes = price_bytes;
        let required = price_bytes + coeff_bytes + fv_bytes + out_bytes;
        let headroom = 32 * 1024 * 1024; // lighter workload, smaller margin
        // VRAM guard with headroom
        Self::will_fit_checked(required, headroom)?;

        let d_prices_tm = DeviceBuffer::from_slice(prices_tm)
            .map_err(|e| CudaGaussianError::Cuda(e))?;
        let d_coeffs = DeviceBuffer::from_slice(&prepared.coeffs)
            .map_err(|e| CudaGaussianError::Cuda(e))?;
        let d_first_valids = DeviceBuffer::from_slice(&prepared.first_valids)
            .map_err(|e| CudaGaussianError::Cuda(e))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(prices_tm.len()) }
            .map_err(|e| CudaGaussianError::Cuda(e))?;

        let period = params.period.unwrap_or(14);
        let poles = params.poles.unwrap_or(4);

        // Optional: populate GAUSS_COEFFS64 constant if present in the module.
        if let Ok(mut sym) = self.module.get_global::<[f64; COEFF_STRIDE]>(unsafe {
            CStr::from_bytes_with_nul_unchecked(b"GAUSS_COEFFS64\0")
        }) {
            let mut coeff64 = [0.0f64; COEFF_STRIDE];
            for i in 0..COEFF_STRIDE {
                coeff64[i] = prepared.coeffs[i] as f64;
            }
            sym.copy_from(&coeff64)
                .map_err(|e| CudaGaussianError::Cuda(e))?;
        }

        self.launch_many_series_kernel(
            &d_prices_tm,
            &d_coeffs,
            period,
            poles,
            cols,
            rows,
            &d_first_valids,
            &mut d_out,
        )?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_poles: &DeviceBuffer<i32>,
        d_coeffs: &DeviceBuffer<f32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaGaussianError> {
        let mut func = self
            .module
            .get_function("gaussian_batch_f32")
            .map_err(|_| CudaGaussianError::MissingKernelSymbol { name: "gaussian_batch_f32" })?;

        // Prefer L1 (harmless hint on arch where fixed)
        func.set_cache_config(CacheConfig::PreferL1)?;

        // Ask CUDA for an occupancy-guided block size suggestion.
        let (_min_grid, suggested_block) = func.suggested_launch_configuration(0, BlockSize::xy(1024, 1))?;

        // Allow explicit policy override (benches); else use suggestion.
        let block_x = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => block_x.max(1),
            BatchKernelPolicy::Auto => suggested_block.max(128),
        };
        let block: BlockSize = BlockSize::xyz(block_x, 1, 1);

        // Grid-stride across combos: single launch covers all work.
        let grid_x = ((n_combos as u32) + block_x - 1) / block_x;
        let grid: GridSize = GridSize::xyz(grid_x.max(1), 1, 1);

        // Introspection
        unsafe {
            let this = self as *const _ as *mut CudaGaussian;
            (*this).last_batch = Some(BatchKernelSelected::Plain { block_x });
        }
        self.maybe_log_batch_debug();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut poles_ptr = d_poles.as_device_ptr().as_raw();
            let mut coeffs_ptr = d_coeffs.as_device_ptr().as_raw();
            let mut coeff_stride_i = COEFF_STRIDE as i32;
            let mut series_len_i = series_len as i32;
            let mut combos_i = n_combos as i32;
            let mut first_valid_i = first_valid as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();

            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut poles_ptr as *mut _ as *mut c_void,
                &mut coeffs_ptr as *mut _ as *mut c_void,
                &mut coeff_stride_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        Ok(())
    }

    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_coeffs: &DeviceBuffer<f32>,
        period: usize,
        poles: usize,
        num_series: usize,
        series_len: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaGaussianError> {
        let mut func = self
            .module
            .get_function("gaussian_many_series_one_param_f32")
            .map_err(|_| CudaGaussianError::MissingKernelSymbol { name: "gaussian_many_series_one_param_f32" })?;

        // Hint: prefer L1
        func.set_cache_config(CacheConfig::PreferL1)?;

        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(1),
            ManySeriesKernelPolicy::Auto => 1,
        };
        let grid: GridSize = (num_series as u32, 1, 1).into();
        let block: BlockSize = (block_x.min(1), 1, 1).into();

        // Introspection
        unsafe {
            let this = self as *const _ as *mut CudaGaussian;
            (*this).last_many = Some(ManySeriesKernelSelected::OneD { block_x: 1 });
        }
        self.maybe_log_many_debug();

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut coeffs_ptr = d_coeffs.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut poles_i = poles as i32;
            let mut num_series_i = num_series as i32;
            let mut series_len_i = series_len as i32;
            let mut first_valids_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut coeffs_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut poles_i as *mut _ as *mut c_void,
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
        sweep: &GaussianBatchRange,
    ) -> Result<BatchInputs, CudaGaussianError> {
        if prices.is_empty() {
            return Err(CudaGaussianError::InvalidInput("empty price series".into()));
        }

        let combos = expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaGaussianError::InvalidInput(
                "Gaussian sweep produced no parameter combinations".into(),
            ));
        }

        let series_len = prices.len();
        let first_valid = prices
            .iter()
            .position(|x| !x.is_nan())
            .ok_or_else(|| CudaGaussianError::InvalidInput("all values are NaN".into()))?;

        let mut periods = Vec::with_capacity(combos.len());
        let mut poles = Vec::with_capacity(combos.len());
        let mut coeffs = Vec::with_capacity(combos.len() * COEFF_STRIDE);

        for prm in &combos {
            let period = prm.period.unwrap_or(14);
            let pole = prm.poles.unwrap_or(4);

            if period < 2 {
                return Err(CudaGaussianError::InvalidInput(format!(
                    "period must be >= 2 (got {})",
                    period
                )));
            }
            if !(1..=4).contains(&pole) {
                return Err(CudaGaussianError::InvalidInput(format!(
                    "poles must be in 1..=4 (got {})",
                    pole
                )));
            }
            if period > i32::MAX as usize {
                return Err(CudaGaussianError::InvalidInput(
                    "period exceeds i32::MAX".into(),
                ));
            }
            if series_len - first_valid < period {
                return Err(CudaGaussianError::InvalidInput(format!(
                    "not enough valid data: needed >= {}, valid = {}",
                    period,
                    series_len - first_valid
                )));
            }

            let coeff = compute_gaussian_coeffs(period, pole)?;
            periods.push(period as i32);
            poles.push(pole as i32);
            coeffs.extend_from_slice(&coeff);
        }

        Ok(BatchInputs {
            combos,
            periods,
            poles,
            coeffs,
            first_valid,
            series_len,
        })
    }

    fn prepare_many_series_inputs(
        prices_tm: &[f32],
        cols: usize,
        rows: usize,
        params: &GaussianParams,
    ) -> Result<ManySeriesInputs, CudaGaussianError> {
        if cols == 0 || rows == 0 {
            return Err(CudaGaussianError::InvalidInput(
                "matrix dimensions must be positive".into(),
            ));
        }
        if prices_tm.len() != cols * rows {
            return Err(CudaGaussianError::InvalidInput(
                "matrix shape mismatch for time-major layout".into(),
            ));
        }

        let period = params.period.unwrap_or(14);
        let poles = params.poles.unwrap_or(4);
        if period < 2 {
            return Err(CudaGaussianError::InvalidInput(format!(
                "period must be >= 2 (got {})",
                period
            )));
        }
        if !(1..=4).contains(&poles) {
            return Err(CudaGaussianError::InvalidInput(format!(
                "poles must be in 1..=4 (got {})",
                poles
            )));
        }

        let mut first_valids = vec![0i32; cols];
        for series_idx in 0..cols {
            let mut fv = None;
            for row in 0..rows {
                let idx = row * cols + series_idx;
                let price = prices_tm[idx];
                if !price.is_nan() {
                    fv = Some(row);
                    break;
                }
            }
            let val = fv.ok_or_else(|| {
                CudaGaussianError::InvalidInput(format!(
                    "series {} has no valid price values",
                    series_idx
                ))
            })?;
            if rows - val < period {
                return Err(CudaGaussianError::InvalidInput(format!(
                    "series {} lacks data: needed >= {}, valid = {}",
                    series_idx,
                    period,
                    rows - val
                )));
            }
            first_valids[series_idx] = val as i32;
        }

        let coeffs = compute_gaussian_coeffs(period, poles)?;
        Ok(ManySeriesInputs {
            first_valids,
            coeffs,
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
    fn will_fit_checked(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaGaussianError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        if let Some((free, _total)) = Self::device_mem_info() {
            let need = required_bytes
                .checked_add(headroom_bytes)
                .ok_or_else(|| CudaGaussianError::InvalidInput("byte size overflow".into()))?;
            if need <= free {
                Ok(())
            } else {
                Err(CudaGaussianError::OutOfMemory { required: required_bytes, free, headroom: headroom_bytes })
            }
        } else {
            Ok(())
        }
    }

    #[inline]
    fn maybe_log_batch_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                let per_scenario =
                    std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scenario || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] Gaussian batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaGaussian)).debug_batch_logged = true;
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
                let per_scenario =
                    std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scenario || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] Gaussian many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaGaussian)).debug_many_logged = true;
                }
            }
        }
    }
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::define_ma_period_benches;

    define_ma_period_benches!(
        gaussian_benches,
        CudaGaussian,
        crate::indicators::moving_averages::gaussian::GaussianBatchRange,
        crate::indicators::moving_averages::gaussian::GaussianParams,
        gaussian_batch_dev,
        gaussian_many_series_one_param_time_major_dev,
        crate::indicators::moving_averages::gaussian::GaussianBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1),
            poles: (4, 4, 0)
        },
        crate::indicators::moving_averages::gaussian::GaussianParams {
            period: Some(64),
            poles: Some(4)
        },
        "gaussian",
        "gaussian"
    );
    pub use gaussian_benches::bench_profiles;
}

struct BatchInputs {
    combos: Vec<GaussianParams>,
    periods: Vec<i32>,
    poles: Vec<i32>,
    coeffs: Vec<f32>,
    first_valid: usize,
    series_len: usize,
}

struct ManySeriesInputs {
    first_valids: Vec<i32>,
    coeffs: [f32; COEFF_STRIDE],
}

fn compute_gaussian_coeffs(
    period: usize,
    poles: usize,
) -> Result<[f32; COEFF_STRIDE], CudaGaussianError> {
    use std::f64::consts::PI;

    if period < 2 {
        return Err(CudaGaussianError::InvalidInput(
            "period must be >= 2 for Gaussian coefficients".into(),
        ));
    }
    if !(1..=4).contains(&poles) {
        return Err(CudaGaussianError::InvalidInput(
            "poles must be within 1..=4 for Gaussian coefficients".into(),
        ));
    }

    let period_f = period as f64;
    let poles_f = poles as f64;

    let beta_num = 1.0 - (2.0 * PI / period_f).cos();
    let beta_den = (2.0f64).powf(1.0 / poles_f) - 1.0;
    if beta_den.abs() < 1e-12 {
        return Err(CudaGaussianError::InvalidInput(
            "beta denominator too small, coefficients unstable".into(),
        ));
    }
    let beta = beta_num / beta_den;
    let discr = beta * beta + 2.0 * beta;
    if discr < 0.0 {
        return Err(CudaGaussianError::InvalidInput(
            "negative discriminant while computing Gaussian alpha".into(),
        ));
    }
    let alpha = -beta + discr.sqrt();
    let one = 1.0 - alpha;

    let mut coeffs = [0f32; COEFF_STRIDE];
    match poles {
        1 => {
            coeffs[0] = alpha as f32;
            coeffs[1] = one as f32;
        }
        2 => {
            let one_sq = one * one;
            coeffs[0] = (alpha * alpha) as f32;
            coeffs[1] = (2.0 * one) as f32;
            coeffs[2] = (-one_sq) as f32;
        }
        3 => {
            let one_sq = one * one;
            coeffs[0] = (alpha * alpha * alpha) as f32;
            coeffs[1] = (3.0 * one) as f32;
            coeffs[2] = (-3.0 * one_sq) as f32;
            coeffs[3] = (one_sq * one) as f32;
        }
        4 => {
            let one_sq = one * one;
            let one_cu = one_sq * one;
            coeffs[0] = (alpha * alpha * alpha * alpha) as f32;
            coeffs[1] = (4.0 * one) as f32;
            coeffs[2] = (-6.0 * one_sq) as f32;
            coeffs[3] = (4.0 * one_cu) as f32;
            coeffs[4] = (-(one_cu * one)) as f32;
        }
        _ => unreachable!(),
    }

    if coeffs.iter().any(|c| !c.is_finite()) {
        return Err(CudaGaussianError::InvalidInput(
            "non-finite Gaussian coefficients produced".into(),
        ));
    }
    Ok(coeffs)
}

fn expand_grid(range: &GaussianBatchRange) -> Vec<GaussianParams> {
    fn axis((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end { return vec![start]; }
        let (lo, hi) = if start <= end { (start, end) } else { (end, start) };
        (lo..=hi).step_by(step).collect()
    }

    let periods = axis(range.period);
    let poles = axis(range.poles);
    let mut combos = Vec::with_capacity(periods.len() * poles.len());
    for &p in &periods {
        for &k in &poles {
            combos.push(GaussianParams {
                period: Some(p),
                poles: Some(k),
            });
        }
    }
    combos
}

// -------------------- Python: CUDA Array Interface v3 + DLPack stubs ----------------------
#[cfg(all(feature = "python", feature = "cuda"))]
use pyo3::prelude::*;
#[cfg(all(feature = "python", feature = "cuda"))]
use pyo3::types::PyDictMethods;
#[cfg(all(feature = "python", feature = "cuda"))]
use pyo3::types::PyDict;

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(module = "ta_indicators.cuda", name = "DeviceArrayF32")]
pub struct DeviceArrayF32Py {
    // Allow one-time move-out on __dlpack__ while keeping shape metadata
    pub inner: Option<DeviceArrayF32>,
    stream_handle: usize,
    _ctx_guard: Arc<Context>,
    _device_id: u32,
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pymethods]
impl DeviceArrayF32Py {
    #[getter]
    fn __cuda_array_interface__<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let itemsize = std::mem::size_of::<f32>();
        let inner = self
            .inner
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("buffer already exported via __dlpack__"))?;
        let d = PyDict::new(py);
        let inner = self
            .inner
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
                "buffer already exported via __dlpack__",
            ))?;
        d.set_item("shape", (inner.rows, inner.cols))?;
        d.set_item("typestr", "<f4")?;
        d.set_item("strides", (inner.cols * itemsize, itemsize))?;
        let size = inner.rows.saturating_mul(inner.cols);
        let ptr_val: usize = if size == 0 { 0 } else { inner.buf.as_device_ptr().as_raw() as usize };
        d.set_item("data", (ptr_val, false))?;
        d.set_item("version", 3)?;
        if self.stream_handle != 0 {
            // Per CAI v3: integer stream sentinel semantics (1 legacy, 2 PTDS, else pointer)
            d.set_item("stream", self.stream_handle)?;
        }
        Ok(d.into())
    }

    fn __dlpack_device__(&self) -> (i32, i32) { (2, self._device_id as i32) }

    #[pyo3(signature = (stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__<'py>(
        &mut self,
        py: Python<'py>,
        stream: Option<PyObject>,
        max_version: Option<(i32, i32)>,
        _dl_device: Option<(i32, i32)>,
        _copy: Option<bool>,
    ) -> PyResult<PyObject> {
        use std::ffi::c_void;
        use cust::memory::DeviceBuffer;

        // Ignore stream parameter because kernels synchronized before handle return.
        let _ = stream;

        // Move ownership of the device buffer into DLManagedTensor manager
        let inner = self
            .inner
            .take()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("__dlpack__ may only be called once"))?;

        // DLPack C structs
        #[repr(C)]
        struct DLDevice { device_type: i32, device_id: i32 }
        #[repr(C)]
        struct DLDataType { code: u8, bits: u8, lanes: u16 }
        #[repr(C)]
        struct DLTensor {
            data: *mut c_void,
            device: DLDevice,
            ndim: i32,
            dtype: DLDataType,
            shape: *mut i64,
            strides: *mut i64,
            byte_offset: u64,
        }
        #[repr(C)]
        struct DLManagedTensor { dl_tensor: DLTensor, manager_ctx: *mut c_void, deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)> }
        #[repr(C)]
        struct DLPackVersion { major: u32, minor: u32 }
        #[repr(C)]
        struct DLManagedTensorVersioned { version: DLPackVersion, manager_ctx: *mut c_void, deleter: Option<unsafe extern "C" fn(*mut DLManagedTensorVersioned)>, flags: u64, dl_tensor: DLTensor }

        // Keep buffer/context and shape/strides alive until consumer deleter runs
        struct Manager {
            _ctx: Arc<Context>,
            _buf: DeviceBuffer<f32>,
            _shape: Box<[i64; 2]>,
            _strides: Box<[i64; 2]>,
        }

        unsafe extern "C" fn deleter_legacy(p: *mut DLManagedTensor) {
            if p.is_null() { return; }
            // Reclaim the manager and managed tensor
            let mt = Box::from_raw(p);
            let _mgr: Box<Manager> = Box::from_raw(mt.manager_ctx as *mut Manager);
            // drops manager (drops buffer + context Arc)
        }
        unsafe extern "C" fn deleter_v1(p: *mut DLManagedTensorVersioned) {
            if p.is_null() { return; }
            let mt = Box::from_raw(p);
            let _mgr: Box<Manager> = Box::from_raw(mt.manager_ctx as *mut Manager);
        }
        unsafe extern "C" fn capsule_destructor(capsule: *mut pyo3::ffi::PyObject) {
            // Only call deleter when capsule name is still the producer one.
            let vname = std::ffi::CString::new("dltensor_versioned").unwrap();
            let lname = std::ffi::CString::new("dltensor").unwrap();
            let p1 = pyo3::ffi::PyCapsule_GetPointer(capsule, vname.as_ptr());
            if !p1.is_null() { let mt = p1 as *mut DLManagedTensorVersioned; if let Some(del) = (*mt).deleter { del(mt) } return; }
            let p2 = pyo3::ffi::PyCapsule_GetPointer(capsule, lname.as_ptr());
            if !p2.is_null() { let mt = p2 as *mut DLManagedTensor; if let Some(del) = (*mt).deleter { del(mt) } }
        }

        let rows = inner.rows as i64;
        let cols = inner.cols as i64;
        let size = (rows as usize).saturating_mul(cols as usize);
        let data_ptr = if size == 0 { std::ptr::null_mut() } else { inner.buf.as_device_ptr().as_raw() as *mut c_void };
        let shape = Box::new([rows, cols]);
        // Element strides (row-major). DLPack expects element units, not bytes.
        let strides = Box::new([cols, 1]);

        let mgr = Box::new(Manager { _ctx: self._ctx_guard.clone(), _buf: inner.buf, _shape: shape, _strides: strides });
        let mgr_ptr = Box::into_raw(mgr);
        let shape_ptr = unsafe { (*mgr_ptr)._shape.as_ptr() as *mut i64 };
        let strides_ptr = unsafe { (*mgr_ptr)._strides.as_ptr() as *mut i64 };

        let use_versioned = max_version.map(|(maj, _)| maj >= 1).unwrap_or(false);
        if use_versioned {
            let mt = Box::new(DLManagedTensorVersioned {
                version: DLPackVersion { major: 1, minor: 2 },
                manager_ctx: mgr_ptr as *mut c_void,
                deleter: Some(deleter_v1),
                flags: 0,
                dl_tensor: DLTensor {
                    data: data_ptr,
                    device: DLDevice { device_type: 2, device_id: self._device_id as i32 },
                    ndim: 2,
                    dtype: DLDataType { code: 2, bits: 32, lanes: 1 },
                    shape: shape_ptr,
                    strides: strides_ptr,
                    byte_offset: 0,
                },
            });
            let raw_capsule = unsafe {
                let name = std::ffi::CString::new("dltensor_versioned").unwrap();
                pyo3::ffi::PyCapsule_New(Box::into_raw(mt) as *mut c_void, name.as_ptr(), Some(capsule_destructor))
            };
            if raw_capsule.is_null() { return Err(pyo3::exceptions::PyValueError::new_err("failed to create DLPack capsule")); }
            Ok(unsafe { PyObject::from_owned_ptr(py, raw_capsule) })
        } else {
            let mt = Box::new(DLManagedTensor {
                dl_tensor: DLTensor {
                    data: data_ptr,
                    device: DLDevice { device_type: 2, device_id: self._device_id as i32 },
                    ndim: 2,
                    dtype: DLDataType { code: 2, bits: 32, lanes: 1 },
                    shape: shape_ptr,
                    strides: strides_ptr,
                    byte_offset: 0,
                },
                manager_ctx: mgr_ptr as *mut c_void,
                deleter: Some(deleter_legacy),
            });
            let raw_capsule = unsafe {
                let name = std::ffi::CString::new("dltensor").unwrap();
                pyo3::ffi::PyCapsule_New(Box::into_raw(mt) as *mut c_void, name.as_ptr(), Some(capsule_destructor))
            };
            if raw_capsule.is_null() { return Err(pyo3::exceptions::PyValueError::new_err("failed to create DLPack capsule")); }
            Ok(unsafe { PyObject::from_owned_ptr(py, raw_capsule) })
        }
    }
}

#[cfg(all(feature = "python", feature = "cuda"))]
impl DeviceArrayF32Py {
    pub fn new_from_rust(inner: DeviceArrayF32, stream_handle: usize, ctx_guard: Arc<Context>, device_id: u32) -> Self {
        Self { inner: Some(inner), stream_handle, _ctx_guard: ctx_guard, _device_id: device_id }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_gaussian_coeffs_matches_known_values() {
        let coeffs = compute_gaussian_coeffs(10, 2).expect("coeffs");
        assert!(coeffs[0].is_finite());
        assert!(coeffs[1].is_finite());
        assert!(coeffs[2].is_finite());
    }
}
