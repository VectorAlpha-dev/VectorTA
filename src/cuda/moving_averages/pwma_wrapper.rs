//! CUDA wrapper for PWMA (Pascal Weighted Moving Average) kernels.
//!
//! Mirrors the ALMA/SWMA scaffold: validate host inputs, upload FP32 data and
//! Pascal weights once, then launch kernels that keep the dot products entirely
//! on device. Supports both the single-series × many-parameter sweep and the
//! many-series × one-parameter path operating on time-major inputs.

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use super::cwma_wrapper::{BatchKernelPolicy, BatchThreadsPerOutput, ManySeriesKernelPolicy};
use crate::indicators::moving_averages::pwma::{expand_grid, PwmaBatchRange, PwmaParams};
use cust::context::Context;
use cust::context::{CacheConfig, SharedMemoryConfig};
use cust::device::Device;
use cust::function::{BlockSize, Function, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::{c_void, CStr};
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

const PWMA_MAX_PERIOD_CONST: usize = 4096; // must match kernel constant
const BATCH_TX: u32 = 128; // must match async tiled kernel

#[derive(Debug)]
pub enum CudaPwmaError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaPwmaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaPwmaError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaPwmaError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for CudaPwmaError {}

pub struct CudaPwma {
    module: Module,
    stream: Stream,
    _context: Context,
    device_id: u32,
    policy: CudaPwmaPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,

    // Optional constant memory weights availability and host scratch buffer
    cmem_available: bool,
    cmem_scratch: [f32; PWMA_MAX_PERIOD_CONST],
}

impl CudaPwma {
    pub fn new(device_id: usize) -> Result<Self, CudaPwmaError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaPwmaError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaPwmaError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaPwmaError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/pwma_kernel.ptx"));
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
                    Module::from_ptx(ptx, &[]).map_err(|e| CudaPwmaError::Cuda(e.to_string()))?
                }
            }
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaPwmaError::Cuda(e.to_string()))?;

        // Resolve optional __constant__ symbol availability for weights
        let name = unsafe { CStr::from_bytes_with_nul_unchecked(b"pwma_const_w\0") };
        let cmem_available = module
            .get_global::<[f32; PWMA_MAX_PERIOD_CONST]>(name)
            .is_ok();

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaPwmaPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
            cmem_available,
            cmem_scratch: [0.0f32; PWMA_MAX_PERIOD_CONST],
        })
    }

    fn pascal_weights_f32(period: usize) -> Result<Vec<f32>, CudaPwmaError> {
        if period == 0 {
            return Err(CudaPwmaError::InvalidInput(
                "period must be greater than zero".into(),
            ));
        }
        let n = period - 1;
        let mut row = Vec::with_capacity(period);
        let mut sum = 0.0f64;
        for r in 0..=n {
            let mut val = 1.0f64;
            for i in 0..r {
                val *= (n - i) as f64;
                val /= (i + 1) as f64;
            }
            row.push(val);
            sum += val;
        }
        if sum == 0.0 {
            return Err(CudaPwmaError::InvalidInput(format!(
                "Pascal weights sum to zero for period {}",
                period
            )));
        }
        let inv = 1.0 / sum;
        Ok(row.into_iter().map(|v| (v * inv) as f32).collect())
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &PwmaBatchRange,
    ) -> Result<(Vec<PwmaParams>, usize, usize, usize, Vec<f32>), CudaPwmaError> {
        if data_f32.is_empty() {
            return Err(CudaPwmaError::InvalidInput("empty data".into()));
        }
        let first_valid = data_f32
            .iter()
            .position(|x| !x.is_nan())
            .ok_or_else(|| CudaPwmaError::InvalidInput("all values are NaN".into()))?;
        let len = data_f32.len();

        let combos = expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaPwmaError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        let mut max_period = 0usize;
        for prm in &combos {
            let period = prm.period.unwrap_or(0);
            if period == 0 {
                return Err(CudaPwmaError::InvalidInput("period must be > 0".into()));
            }
            if period > len {
                return Err(CudaPwmaError::InvalidInput(format!(
                    "period {} exceeds data length {}",
                    period, len
                )));
            }
            if len - first_valid < period {
                return Err(CudaPwmaError::InvalidInput(format!(
                    "not enough valid data: needed {}, have {}",
                    period,
                    len - first_valid
                )));
            }
            if period > max_period {
                max_period = period;
            }
        }

        let n_combos = combos.len();
        let mut weights_flat = vec![0.0f32; n_combos * max_period];
        for (row, prm) in combos.iter().enumerate() {
            let weights = Self::pascal_weights_f32(prm.period.unwrap())?;
            let base = row * max_period;
            for (idx, w) in weights.iter().enumerate() {
                weights_flat[base + idx] = *w;
            }
        }

        Ok((combos, first_valid, len, max_period, weights_flat))
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_weights: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_warms: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        max_period: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaPwmaError> {
        if series_len == 0 || n_combos == 0 || max_period == 0 {
            return Err(CudaPwmaError::InvalidInput(
                "series_len, n_combos, and max_period must be positive".into(),
            ));
        }
        if series_len > i32::MAX as usize
            || n_combos > i32::MAX as usize
            || max_period > i32::MAX as usize
        {
            return Err(CudaPwmaError::InvalidInput(
                "series_len, n_combos, or max_period exceed i32::MAX".into(),
            ));
        }

        // Prefer async tiled when available (unless user forces Plain)
        let mut use_tiled = true;
        match self.policy.batch {
            BatchKernelPolicy::Auto => {}
            BatchKernelPolicy::Plain { .. } => use_tiled = false,
            BatchKernelPolicy::Tiled { .. } => use_tiled = true,
        }

        if use_tiled {
            if let Ok(mut func) = self.module.get_function("pwma_batch_tiled_async_f32") {
                let tile_x: usize = BATCH_TX as usize; // must equal PWMA_TILE_TX
                let align16 = |x: usize| (x + 15) & !15usize;
                let shared_bytes = (align16(max_period * std::mem::size_of::<f32>())
                    + 2 * (tile_x + max_period - 1) * std::mem::size_of::<f32>())
                    as u32;
                self.prefer_shared_and_optin_smem(&mut func, shared_bytes as usize);

                for (start, len) in Self::grid_y_chunks(n_combos) {
                    let grid_x = ((series_len as u32) + tile_x as u32 - 1) / (tile_x as u32);
                    let grid: GridSize = (grid_x.max(1), len as u32, 1).into();
                    let block: BlockSize = (tile_x as u32, 1, 1).into();

                    unsafe {
                        let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                        let mut weights_ptr =
                            d_weights.as_device_ptr().add(start * max_period).as_raw();
                        let mut periods_ptr = d_periods.as_device_ptr().add(start).as_raw();
                        let mut warms_ptr = d_warms.as_device_ptr().add(start).as_raw();
                        let mut series_len_i = series_len as i32;
                        let mut n_combos_i = len as i32;
                        let mut max_period_i = max_period as i32;
                        let mut out_ptr = d_out.as_device_ptr().add(start * series_len).as_raw();
                        let args: &mut [*mut c_void] = &mut [
                            &mut prices_ptr as *mut _ as *mut c_void,
                            &mut weights_ptr as *mut _ as *mut c_void,
                            &mut periods_ptr as *mut _ as *mut c_void,
                            &mut warms_ptr as *mut _ as *mut c_void,
                            &mut series_len_i as *mut _ as *mut c_void,
                            &mut n_combos_i as *mut _ as *mut c_void,
                            &mut max_period_i as *mut _ as *mut c_void,
                            &mut out_ptr as *mut _ as *mut c_void,
                        ];
                        if self
                            .stream
                            .launch(&func, grid, block, shared_bytes, args)
                            .is_err()
                        {
                            use_tiled = false;
                            break;
                        }
                    }
                }

                if use_tiled {
                    unsafe {
                        let this = self as *const _ as *mut CudaPwma;
                        (*this).last_batch = Some(BatchKernelSelected::AsyncTiled { tx: 128 });
                    }
                    self.maybe_log_batch_debug();
                    return Ok(());
                }
            }
        }

        // Plain 1D fallback
        let func = self
            .module
            .get_function("pwma_batch_f32")
            .map_err(|e| CudaPwmaError::Cuda(e.to_string()))?;

        unsafe {
            let this = self as *const _ as *mut CudaPwma;
            (*this).last_batch = Some(BatchKernelSelected::Plain { block_x: 256 });
        }
        self.maybe_log_batch_debug();

        let block_x: u32 = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => block_x,
            _ => 256,
        };
        let shared_bytes = (max_period * std::mem::size_of::<f32>()) as u32;

        for (start, len) in Self::grid_y_chunks(n_combos) {
            let grid_x = ((series_len as u32) + block_x - 1) / block_x;
            let grid: GridSize = (grid_x.max(1), len as u32, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();

            unsafe {
                let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                let mut weights_ptr = d_weights.as_device_ptr().add(start * max_period).as_raw();
                let mut periods_ptr = d_periods.as_device_ptr().add(start).as_raw();
                let mut warms_ptr = d_warms.as_device_ptr().add(start).as_raw();
                let mut series_len_i = series_len as i32;
                let mut n_combos_i = len as i32;
                let mut max_period_i = max_period as i32;
                let mut out_ptr = d_out.as_device_ptr().add(start * series_len).as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut prices_ptr as *mut _ as *mut c_void,
                    &mut weights_ptr as *mut _ as *mut c_void,
                    &mut periods_ptr as *mut _ as *mut c_void,
                    &mut warms_ptr as *mut _ as *mut c_void,
                    &mut series_len_i as *mut _ as *mut c_void,
                    &mut n_combos_i as *mut _ as *mut c_void,
                    &mut max_period_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream
                    .launch(&func, grid, block, shared_bytes, args)
                    .map_err(|e| CudaPwmaError::Cuda(e.to_string()))?;
            }
        }

        Ok(())
    }

    pub fn pwma_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_weights: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_warms: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        max_period: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaPwmaError> {
        self.launch_batch_kernel(
            d_prices, d_weights, d_periods, d_warms, series_len, n_combos, max_period, d_out,
        )
    }

    pub fn pwma_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &PwmaBatchRange,
    ) -> Result<DeviceArrayF32, CudaPwmaError> {
        let (combos, first_valid, series_len, max_period, weights_flat) =
            Self::prepare_batch_inputs(data_f32, sweep)?;
        let n_combos = combos.len();

        let periods_i32: Vec<i32> = combos.iter().map(|p| p.period.unwrap() as i32).collect();
        let warms_i32: Vec<i32> = combos
            .iter()
            .map(|p| (first_valid + p.period.unwrap() - 1) as i32)
            .collect();

        // VRAM estimate and guard (prices + weights + periods + warms + out)
        let prices_bytes = series_len * std::mem::size_of::<f32>();
        let weights_bytes = n_combos * max_period * std::mem::size_of::<f32>();
        let periods_bytes = n_combos * std::mem::size_of::<i32>();
        let warms_bytes = n_combos * std::mem::size_of::<i32>();
        let out_bytes = n_combos * series_len * std::mem::size_of::<f32>();
        let required = prices_bytes + weights_bytes + periods_bytes + warms_bytes + out_bytes;
        if !Self::will_fit(required, 64 * 1024 * 1024) {
            return Err(CudaPwmaError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        let d_prices = unsafe { DeviceBuffer::from_slice_async(data_f32, &self.stream) }
            .map_err(|e| CudaPwmaError::Cuda(e.to_string()))?;
        let d_weights = DeviceBuffer::from_slice(&weights_flat)
            .map_err(|e| CudaPwmaError::Cuda(e.to_string()))?;
        let d_periods = DeviceBuffer::from_slice(&periods_i32)
            .map_err(|e| CudaPwmaError::Cuda(e.to_string()))?;
        let d_warms =
            DeviceBuffer::from_slice(&warms_i32).map_err(|e| CudaPwmaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }
                .map_err(|e| CudaPwmaError::Cuda(e.to_string()))?;

        self.launch_batch_kernel(
            &d_prices, &d_weights, &d_periods, &d_warms, series_len, n_combos, max_period,
            &mut d_out,
        )?;
        self.stream
            .synchronize()
            .map_err(|e| CudaPwmaError::Cuda(e.to_string()))?;

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
        params: &PwmaParams,
    ) -> Result<(Vec<i32>, Vec<f32>, usize), CudaPwmaError> {
        if cols == 0 || rows == 0 {
            return Err(CudaPwmaError::InvalidInput(
                "num_series or series_len is zero".into(),
            ));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaPwmaError::InvalidInput(format!(
                "data length {} != cols*rows {}",
                data_tm_f32.len(),
                cols * rows
            )));
        }
        let period = params.period.unwrap_or(5);
        if period == 0 {
            return Err(CudaPwmaError::InvalidInput("period must be > 0".into()));
        }
        if period > rows {
            return Err(CudaPwmaError::InvalidInput(format!(
                "period {} exceeds series length {}",
                period, rows
            )));
        }

        let weights = Self::pascal_weights_f32(period)?;

        let mut first_valids = vec![0i32; cols];
        for series in 0..cols {
            let mut found = None;
            for row in 0..rows {
                let idx = row * cols + series;
                if !data_tm_f32[idx].is_nan() {
                    found = Some(row);
                    break;
                }
            }
            let fv = found.ok_or_else(|| {
                CudaPwmaError::InvalidInput(format!("series {} contains only NaNs", series))
            })?;
            if rows - fv < period {
                return Err(CudaPwmaError::InvalidInput(format!(
                    "series {} lacks enough valid data: needed {}, have {}",
                    series,
                    period,
                    rows - fv
                )));
            }
            if fv > i32::MAX as usize {
                return Err(CudaPwmaError::InvalidInput(
                    "first_valid exceeds i32::MAX".into(),
                ));
            }
            first_valids[series] = fv as i32;
        }

        Ok((first_valids, weights, period))
    }

    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_weights: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        period: usize,
        num_series: usize,
        series_len: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
        use_const: bool,
    ) -> Result<(), CudaPwmaError> {
        if period == 0 || num_series == 0 || series_len == 0 {
            return Err(CudaPwmaError::InvalidInput(
                "period, num_series, and series_len must be positive".into(),
            ));
        }
        if period > i32::MAX as usize
            || num_series > i32::MAX as usize
            || series_len > i32::MAX as usize
        {
            return Err(CudaPwmaError::InvalidInput(
                "period, num_series, or series_len exceed i32::MAX".into(),
            ));
        }

        // Try 2D tiled variants first
        let try_2d = |tx: u32, ty: u32| -> Option<()> {
            let fname = match (tx, ty) {
                (128, 4) => "pwma_ms1p_tiled_f32_tx128_ty4",
                (128, 2) => "pwma_ms1p_tiled_f32_tx128_ty2",
                _ => return None,
            };
            let mut func = match self.module.get_function(fname) {
                Ok(f) => f,
                Err(_) => return None,
            };
            let wlen = period; // full-period weights
            let align16 = |x: usize| (x + 15) & !15usize;
            let total = tx as usize + wlen - 1;
            // Bank-conflict padding: stride = TY+1 when TY divides 32
            let ty_pad = if (32 % (ty as usize)) == 0 {
                (ty + 1) as usize
            } else {
                ty as usize
            };
            let shared_bytes = (align16(wlen * std::mem::size_of::<f32>())
                + total * ty_pad * std::mem::size_of::<f32>())
                as u32;
            let grid_x = ((series_len as u32) + tx - 1) / tx;
            let grid_y = ((num_series as u32) + ty - 1) / ty;
            let grid: GridSize = (grid_x, grid_y, 1).into();
            let block: BlockSize = (tx, ty, 1).into();

            self.prefer_shared_and_optin_smem(&mut func, shared_bytes as usize);
            unsafe {
                let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
                let mut weights_ptr = d_weights.as_device_ptr().as_raw();
                let mut period_i = period as i32;
                let mut inv_norm = 1.0f32;
                let mut num_series_i = num_series as i32;
                let mut series_len_i = series_len as i32;
                let mut first_valids_ptr = d_first_valids.as_device_ptr().as_raw();
                let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut prices_ptr as *mut _ as *mut c_void,
                    &mut weights_ptr as *mut _ as *mut c_void,
                    &mut period_i as *mut _ as *mut c_void,
                    &mut inv_norm as *mut _ as *mut c_void,
                    &mut num_series_i as *mut _ as *mut c_void,
                    &mut series_len_i as *mut _ as *mut c_void,
                    &mut first_valids_ptr as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream
                    .launch(&func, grid, block, shared_bytes, args)
                    .map_err(|e| CudaPwmaError::Cuda(e.to_string()))
                    .ok()?;
            }
            unsafe {
                let this = self as *const _ as *mut CudaPwma;
                (*this).last_many = Some(ManySeriesKernelSelected::Tiled2D { tx, ty });
            }
            self.maybe_log_many_debug();
            Some(())
        };

        match self.policy.many_series {
            ManySeriesKernelPolicy::Tiled2D { tx, ty } => {
                if try_2d(tx as u32, ty as u32).is_some() {
                    return Ok(());
                }
            }
            ManySeriesKernelPolicy::OneD { .. } => {}
            ManySeriesKernelPolicy::Auto => {
                if num_series >= 128 {
                    if try_2d(128, 4).is_some() {
                        return Ok(());
                    }
                    if try_2d(128, 2).is_some() {
                        return Ok(());
                    }
                } else {
                    if try_2d(128, 2).is_some() {
                        return Ok(());
                    }
                    if try_2d(128, 4).is_some() {
                        return Ok(());
                    }
                }
            }
        }

        // Fallback 1D: prefer constant-memory path when available and requested
        if use_const {
            let name = unsafe { CStr::from_bytes_with_nul_unchecked(b"pwma_const_w\0") };
            if let (Ok(_sym), Ok(func)) = (
                self.module.get_global::<[f32; PWMA_MAX_PERIOD_CONST]>(name),
                self.module.get_function("pwma_ms1p_const_f32"),
            ) {
                if period <= PWMA_MAX_PERIOD_CONST {
                    let block_x: u32 = match self.policy.many_series {
                        ManySeriesKernelPolicy::OneD { block_x } => block_x,
                        _ => 128,
                    };
                    let grid_x = ((series_len as u32) + block_x - 1) / block_x;
                    let grid: GridSize = (grid_x.max(1), num_series as u32, 1).into();
                    let block: BlockSize = (block_x, 1, 1).into();
                    let shared_bytes = 0u32;

                    unsafe {
                        // Constant memory expected to be populated by caller when use_const=true
                        let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
                        let mut period_i = period as i32;
                        let mut num_series_i = num_series as i32;
                        let mut series_len_i = series_len as i32;
                        let mut first_valids_ptr = d_first_valids.as_device_ptr().as_raw();
                        let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
                        let args: &mut [*mut c_void] = &mut [
                            &mut prices_ptr as *mut _ as *mut c_void,
                            &mut period_i as *mut _ as *mut c_void,
                            &mut num_series_i as *mut _ as *mut c_void,
                            &mut series_len_i as *mut _ as *mut c_void,
                            &mut first_valids_ptr as *mut _ as *mut c_void,
                            &mut out_ptr as *mut _ as *mut c_void,
                        ];
                        self.stream
                            .launch(&func, grid, block, shared_bytes, args)
                            .map_err(|e| CudaPwmaError::Cuda(e.to_string()))?;
                    }
                    unsafe {
                        let this = self as *const _ as *mut CudaPwma;
                        (*this).last_many = Some(ManySeriesKernelSelected::Const1D { block_x });
                    }
                    self.maybe_log_many_debug();
                    return Ok(());
                }
            }
        }

        // Non-constant 1D fallback
        let func = self
            .module
            .get_function("pwma_multi_series_one_param_f32")
            .map_err(|e| CudaPwmaError::Cuda(e.to_string()))?;

        let block_x: u32 = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => block_x,
            _ => 128,
        };
        let grid_x = ((series_len as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), num_series as u32, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        let shared_bytes = (period * std::mem::size_of::<f32>()) as u32;

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut weights_ptr = d_weights.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut inv_norm = 1.0f32;
            let mut num_series_i = num_series as i32;
            let mut series_len_i = series_len as i32;
            let mut first_valids_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut weights_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut inv_norm as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut first_valids_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, shared_bytes, args)
                .map_err(|e| CudaPwmaError::Cuda(e.to_string()))?;
        }
        unsafe {
            let this = self as *const _ as *mut CudaPwma;
            (*this).last_many = Some(ManySeriesKernelSelected::OneD { block_x });
        }
        self.maybe_log_many_debug();
        Ok(())
    }

    pub fn pwma_multi_series_one_param_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_weights: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        period: i32,
        num_series: i32,
        series_len: i32,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaPwmaError> {
        if period <= 0 || num_series <= 0 || series_len <= 0 {
            return Err(CudaPwmaError::InvalidInput(
                "period, num_series, and series_len must be positive".into(),
            ));
        }
        self.launch_many_series_kernel(
            d_prices_tm,
            d_weights,
            d_first_valids,
            period as usize,
            num_series as usize,
            series_len as usize,
            d_out_tm,
            false,
        )
    }

    pub fn pwma_multi_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &PwmaParams,
    ) -> Result<DeviceArrayF32, CudaPwmaError> {
        let (first_valids, weights, period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;

        // VRAM estimate: prices + (weights if not const) + first_valids + out
        let prices_bytes = cols * rows * std::mem::size_of::<f32>();
        let use_const = self.cmem_available
            && self.module.get_function("pwma_ms1p_const_f32").is_ok()
            && period <= PWMA_MAX_PERIOD_CONST;
        let weights_bytes = if use_const {
            0
        } else {
            period * std::mem::size_of::<f32>()
        };
        let fv_bytes = cols * std::mem::size_of::<i32>();
        let out_bytes = cols * rows * std::mem::size_of::<f32>();
        let required = prices_bytes + weights_bytes + fv_bytes + out_bytes;
        if !Self::will_fit(required, 64 * 1024 * 1024) {
            return Err(CudaPwmaError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        let d_prices_tm = unsafe { DeviceBuffer::from_slice_async(data_tm_f32, &self.stream) }
            .map_err(|e| CudaPwmaError::Cuda(e.to_string()))?;
        let use_const = self.cmem_available
            && self.module.get_function("pwma_ms1p_const_f32").is_ok()
            && period <= PWMA_MAX_PERIOD_CONST;
        let d_weights = if use_const {
            // no device weights needed when using constant memory
            unsafe { DeviceBuffer::uninitialized(0) }
                .map_err(|e| CudaPwmaError::Cuda(e.to_string()))?
        } else {
            DeviceBuffer::from_slice(&weights).map_err(|e| CudaPwmaError::Cuda(e.to_string()))?
        };
        let d_first_valids = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaPwmaError::Cuda(e.to_string()))?;
        let mut d_out_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaPwmaError::Cuda(e.to_string()))?;
        // Populate constant memory when used
        if use_const {
            let name = unsafe { CStr::from_bytes_with_nul_unchecked(b"pwma_const_w\0") };
            if let Ok(mut sym) = self.module.get_global::<[f32; PWMA_MAX_PERIOD_CONST]>(name) {
                let mut arr = self.cmem_scratch;
                for v in arr.iter_mut() {
                    *v = 0.0;
                }
                arr[..period].copy_from_slice(&weights);
                unsafe { sym.copy_from(&arr) }.map_err(|e| CudaPwmaError::Cuda(e.to_string()))?;
            }
        }

        self.launch_many_series_kernel(
            &d_prices_tm,
            &d_weights,
            &d_first_valids,
            period,
            cols,
            rows,
            &mut d_out_tm,
            use_const,
        )?;
        self.stream
            .synchronize()
            .map_err(|e| CudaPwmaError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out_tm,
            rows,
            cols,
        })
    }

    pub fn pwma_multi_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &PwmaParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaPwmaError> {
        if out_tm.len() != cols * rows {
            return Err(CudaPwmaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out_tm.len(),
                cols * rows
            )));
        }
        let (first_valids, weights, period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;

        let d_prices_tm = unsafe { DeviceBuffer::from_slice_async(data_tm_f32, &self.stream) }
            .map_err(|e| CudaPwmaError::Cuda(e.to_string()))?;
        let use_const = self.cmem_available
            && self.module.get_function("pwma_ms1p_const_f32").is_ok()
            && period <= PWMA_MAX_PERIOD_CONST;
        let d_weights = if use_const {
            unsafe { DeviceBuffer::uninitialized(0) }
                .map_err(|e| CudaPwmaError::Cuda(e.to_string()))?
        } else {
            DeviceBuffer::from_slice(&weights).map_err(|e| CudaPwmaError::Cuda(e.to_string()))?
        };
        let d_first_valids = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaPwmaError::Cuda(e.to_string()))?;
        let mut d_out_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaPwmaError::Cuda(e.to_string()))?;
        if use_const {
            let name = unsafe { CStr::from_bytes_with_nul_unchecked(b"pwma_const_w\0") };
            if let Ok(mut sym) = self.module.get_global::<[f32; PWMA_MAX_PERIOD_CONST]>(name) {
                let mut arr = self.cmem_scratch;
                for v in arr.iter_mut() {
                    *v = 0.0;
                }
                arr[..period].copy_from_slice(&weights);
                unsafe { sym.copy_from(&arr) }.map_err(|e| CudaPwmaError::Cuda(e.to_string()))?;
            }
        }

        self.launch_many_series_kernel(
            &d_prices_tm,
            &d_weights,
            &d_first_valids,
            period,
            cols,
            rows,
            &mut d_out_tm,
            use_const,
        )?;
        self.stream
            .synchronize()
            .map_err(|e| CudaPwmaError::Cuda(e.to_string()))?;
        // Use pinned host memory for the device->host copy to avoid extra staging
        let mut pinned: LockedBuffer<f32> = unsafe { LockedBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaPwmaError::Cuda(e.to_string()))?;
        unsafe {
            d_out_tm
                .async_copy_to(&mut pinned, &self.stream)
                .map_err(|e| CudaPwmaError::Cuda(e.to_string()))?;
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaPwmaError::Cuda(e.to_string()))?;
        out_tm.copy_from_slice(pinned.as_slice());
        Ok(())
    }
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::define_ma_period_benches;

    define_ma_period_benches!(
        pwma_benches,
        CudaPwma,
        crate::indicators::moving_averages::pwma::PwmaBatchRange,
        crate::indicators::moving_averages::pwma::PwmaParams,
        pwma_batch_dev,
        pwma_multi_series_one_param_time_major_dev,
        crate::indicators::moving_averages::pwma::PwmaBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1)
        },
        crate::indicators::moving_averages::pwma::PwmaParams { period: Some(64) },
        "pwma",
        "pwma"
    );
    pub use pwma_benches::bench_profiles;
}

// -------- Policies, introspection and helpers (mirrors ALMA/CWMA) --------

#[derive(Clone, Copy, Debug)]
pub struct CudaPwmaPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

impl Default for CudaPwmaPolicy {
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
    AsyncTiled { tx: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
    Tiled2D { tx: u32, ty: u32 },
    Const1D { block_x: u32 },
}

impl CudaPwma {
    pub fn policy(&self) -> &CudaPwmaPolicy {
        &self.policy
    }
    pub fn set_policy(&mut self, policy: CudaPwmaPolicy) {
        self.policy = policy;
    }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch
    }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many
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
                    eprintln!("[DEBUG] PWMA batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaPwma)).debug_batch_logged = true;
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
                    eprintln!("[DEBUG] PWMA many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaPwma)).debug_many_logged = true;
                }
            }
        }
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        match std::env::var("CUDA_MEM_CHECK") {
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
    fn grid_y_chunks(n: usize) -> impl Iterator<Item = (usize, usize)> {
        const MAX: usize = 65_535;
        (0..n).step_by(MAX).map(move |start| {
            let len = (n - start).min(MAX);
            (start, len)
        })
    }

    #[inline]
    fn prefer_shared_and_optin_smem(&self, func: &mut Function, requested_dynamic_smem: usize) {
        let _ = func.set_cache_config(CacheConfig::PreferShared);
        let _ = func.set_shared_memory_config(SharedMemoryConfig::FourByteBankSize);
        unsafe {
            use cust::sys::{cuFuncSetAttribute, CUfunction_attribute_enum as Attr};
            let raw = func.to_raw();
            let _ = cuFuncSetAttribute(
                raw,
                Attr::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                requested_dynamic_smem as i32,
            );
            let _ = cuFuncSetAttribute(
                raw,
                Attr::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                100,
            );
        }
    }
}
