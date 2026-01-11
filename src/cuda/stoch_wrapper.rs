//! CUDA support for the Stochastic Oscillator (Stoch).
//!
//! Goals (ALMA-parity):
//! - PTX load from OUT_DIR with DetermineTargetFromContext + OptLevel O2 (fallbacks on error)
//! - NON_BLOCKING stream
//! - Batch (one series × many params) and Many-series × one param (time-major)
//! - Warmup/NaN semantics identical to scalar implementation
//! - Where helpful, reuse existing CUDA MA wrappers via the thin `CudaMaSelector`
//!   for the slow %K and %D smoothing stages.
//!
//! Implementation notes:
//! - Batch path computes raw %K once per unique `fastk_period` by precomputing
//!   rolling highest-high/lowest-low on host (O(n)), then launches a light
//!   kernel to convert (close, hh, ll) -> raw %K in FP32. Smoothing is done via
//!   the MA selector per-row to match the scalar `ma` dispatch semantics.
//! - Many-series path operates on time-major OHLC inputs with a shared param
//!   set; raw %K uses an O(period) per-step loop which is sufficient for common
//!   periods (~14). Smoothing uses native SMA/EMA many-series wrappers when
//!   applicable, otherwise falls back to per-series MA selection.

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::ma_selector::{CudaMaData, CudaMaSelector};
use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::stoch::{StochBatchRange, StochParams};
use crate::indicators::utility_functions::{max_rolling, min_rolling};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::mem_get_info;
use cust::memory::{AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Arc;

#[derive(thiserror::Error, Debug)]
pub enum CudaStochError {
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

pub struct CudaStoch {
    module: Module,
    sma_module: Module,
    ema_module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

pub struct CudaStochBatch {
    pub k: DeviceArrayF32,
    pub d: DeviceArrayF32,
    pub combos: Vec<StochParams>,
}

impl CudaStoch {
    pub fn new(device_id: usize) -> Result<Self, CudaStochError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let load = |ptx: &'static str| -> Result<Module, CudaStochError> {
            Module::from_ptx(
                ptx,
                &[
                    ModuleJitOption::DetermineTargetFromContext,
                    ModuleJitOption::OptLevel(OptLevel::O2),
                ],
            )
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))
            .map_err(CudaStochError::Cuda)
        };

        let ptx_stoch: &str = include_str!(concat!(env!("OUT_DIR"), "/stoch_kernel.ptx"));
        let module = load(ptx_stoch)?;

        // Load SMA/EMA modules once for the smoothing stages (avoid per-row JIT/module load).
        let ptx_sma: &str = include_str!(concat!(env!("OUT_DIR"), "/sma_kernel.ptx"));
        let sma_module = load(ptx_sma)?;
        let ptx_ema: &str = include_str!(concat!(env!("OUT_DIR"), "/ema_kernel.ptx"));
        let ema_module = load(ptx_ema)?;

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        Ok(Self {
            module,
            sma_module,
            ema_module,
            stream,
            context,
            device_id: device_id as u32,
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
    fn device_mem_info() -> Option<(usize, usize)> {
        mem_get_info().ok()
    }

    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaStochError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        if let Some((free, _)) = Self::device_mem_info() {
            if required_bytes.saturating_add(headroom_bytes) <= free {
                Ok(())
            } else {
                Err(CudaStochError::OutOfMemory {
                    required: required_bytes,
                    free,
                    headroom: headroom_bytes,
                })
            }
        } else {
            Ok(())
        }
    }

    #[inline]
    pub fn context_arc(&self) -> Arc<Context> {
        self.context.clone()
    }

    #[inline]
    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    // ---------------------- Batch (one series × many params) ----------------------
    pub fn stoch_batch_dev(
        &self,
        high_f32: &[f32],
        low_f32: &[f32],
        close_f32: &[f32],
        sweep: &StochBatchRange,
    ) -> Result<CudaStochBatch, CudaStochError> {
        let len = high_f32.len();
        if len == 0 || low_f32.len() != len || close_f32.len() != len {
            return Err(CudaStochError::InvalidInput(
                "inputs must be non-empty and same length".into(),
            ));
        }

        let first_valid = (0..len)
            .find(|&i| {
                high_f32[i].is_finite() && low_f32[i].is_finite() && close_f32[i].is_finite()
            })
            .ok_or_else(|| CudaStochError::InvalidInput("all values are NaN".into()))?;

        let combos = expand_grid_stoch(sweep)?;
        if combos.is_empty() {
            return Err(CudaStochError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        let max_fkp = combos.iter().map(|c| c.fastk_period.unwrap_or(14)).max().unwrap_or(14);
        if len - first_valid < max_fkp {
            return Err(CudaStochError::InvalidInput(format!(
                "not enough valid data for fastk {} (tail = {})",
                max_fkp, len - first_valid
            )));
        }

        // VRAM estimate: close + hh + ll + kraw + K/D outputs (rows_total × len)
        let rows_total = combos.len();
        let inputs_elems = len
            .checked_mul(4)
            .ok_or_else(|| CudaStochError::InvalidInput("size overflow".into()))?;
        let outputs_elems = rows_total
            .checked_mul(len)
            .and_then(|v| v.checked_mul(2))
            .ok_or_else(|| CudaStochError::InvalidInput("size overflow".into()))?;
        let total_elems = inputs_elems
            .checked_add(outputs_elems)
            .ok_or_else(|| CudaStochError::InvalidInput("size overflow".into()))?;
        let required_bytes = total_elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaStochError::InvalidInput("size overflow".into()))?;
        Self::will_fit(required_bytes, 64 * 1024 * 1024)?;

        // Upload base CLOSE once
        let d_close = DeviceBuffer::from_slice(close_f32).map_err(CudaStochError::Cuda)?;

        // Final outputs on device (row-major: [combos, len])
        let total_out = rows_total
            .checked_mul(len)
            .ok_or_else(|| CudaStochError::InvalidInput("size overflow".into()))?;
        let mut d_k: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(total_out) }.map_err(CudaStochError::Cuda)?;
        let mut d_d: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(total_out) }.map_err(CudaStochError::Cuda)?;

        // Kernel handles
        let func_kraw = self
            .module
            .get_function("stoch_k_raw_from_hhll_f32")
            .map_err(|_| CudaStochError::MissingKernelSymbol {
                name: "stoch_k_raw_from_hhll_f32",
            })?;
        let func_pack = self
            .module
            .get_function("pack_row_broadcast_rowmajor_f32")
            .map_err(|_| CudaStochError::MissingKernelSymbol {
                name: "pack_row_broadcast_rowmajor_f32",
            })?;
        let func_sma = self
            .sma_module
            .get_function("sma_many_series_one_param_f32")
            .map_err(|_| CudaStochError::MissingKernelSymbol {
                name: "sma_many_series_one_param_f32",
            })?;
        let func_ema = self
            .ema_module
            .get_function("ema_many_series_one_param_f32")
            .map_err(|_| CudaStochError::MissingKernelSymbol {
                name: "ema_many_series_one_param_f32",
            })?;
        let func_ema_coalesced = self
            .ema_module
            .get_function("ema_many_series_one_param_f32_coalesced")
            .ok();

        // Fast path: uniform slowK/slowD settings (common case for sweeping fastk).
        // Compute raw %K for all rows on-device, then smooth all rows with SMA/EMA
        // in one pass each, and transpose once per output.
        if rows_total >= 2 {
            let slowk_p0 = combos[0].slowk_period.unwrap_or(3);
            let slowd_p0 = combos[0].slowd_period.unwrap_or(3);
            let slowk_ty0 = combos[0].slowk_ma_type.as_deref().unwrap_or("sma");
            let slowd_ty0 = combos[0].slowd_ma_type.as_deref().unwrap_or("sma");

            let uniform_slow = combos.iter().all(|c| {
                c.slowk_period.unwrap_or(3) == slowk_p0
                    && c.slowd_period.unwrap_or(3) == slowd_p0
                    && c.slowk_ma_type
                        .as_deref()
                        .unwrap_or("sma")
                        .eq_ignore_ascii_case(slowk_ty0)
                    && c.slowd_ma_type
                        .as_deref()
                        .unwrap_or("sma")
                        .eq_ignore_ascii_case(slowd_ty0)
            });

            let slowk_is_sma = slowk_ty0.eq_ignore_ascii_case("sma");
            let slowk_is_ema = slowk_ty0.eq_ignore_ascii_case("ema");
            let slowd_is_sma = slowd_ty0.eq_ignore_ascii_case("sma");
            let slowd_is_ema = slowd_ty0.eq_ignore_ascii_case("ema");

            let all_fastk_pos = combos.iter().all(|c| c.fastk_period.unwrap_or(14) > 0);

            if uniform_slow
                && slowk_p0 > 0
                && slowd_p0 > 0
                && all_fastk_pos
                && (slowk_is_sma || slowk_is_ema)
                && (slowd_is_sma || slowd_is_ema)
            {
                let func_kraw_many = self
                    .module
                    .get_function("stoch_one_series_many_params_f32")
                    .ok();
                let func_transpose = self.module.get_function("transpose_tm_to_rm_f32").ok();

                if let (Some(func_kraw_many), Some(func_transpose)) = (func_kraw_many, func_transpose)
                {
                    // Extra VRAM check for the fast path (time-major temporaries).
                    let tm_elems = rows_total
                        .checked_mul(len)
                        .ok_or_else(|| CudaStochError::InvalidInput("size overflow".into()))?;
                    let in_elems = len
                        .checked_mul(3)
                        .ok_or_else(|| CudaStochError::InvalidInput("size overflow".into()))?;
                    let tm_bufs = tm_elems
                        .checked_mul(2)
                        .ok_or_else(|| CudaStochError::InvalidInput("size overflow".into()))?;
                    let out_elems = rows_total
                        .checked_mul(len)
                        .and_then(|v| v.checked_mul(2))
                        .ok_or_else(|| CudaStochError::InvalidInput("size overflow".into()))?;
                    let total_f32 = in_elems
                        .checked_add(tm_bufs)
                        .and_then(|v| v.checked_add(out_elems))
                        .ok_or_else(|| CudaStochError::InvalidInput("size overflow".into()))?;
                    let required_fast = total_f32
                        .checked_mul(std::mem::size_of::<f32>())
                        .ok_or_else(|| CudaStochError::InvalidInput("size overflow".into()))?;
                    if Self::will_fit(required_fast, 64 * 1024 * 1024).is_ok() {

                    // Upload OHLC needed for raw %K.
                    let d_high = DeviceBuffer::from_slice(high_f32).map_err(CudaStochError::Cuda)?;
                    let d_low = DeviceBuffer::from_slice(low_f32).map_err(CudaStochError::Cuda)?;

                    // Per-row params and stage first-valid indices for smoothing.
                    let mut fastk_periods = Vec::<i32>::with_capacity(rows_total);
                    let mut first_valids = Vec::<i32>::with_capacity(rows_total);
                    let mut first_kraws = Vec::<i32>::with_capacity(rows_total);
                    let mut first_slowks = Vec::<i32>::with_capacity(rows_total);
                    let fv = first_valid as i32;
                    for prm in combos.iter() {
                        let fk = prm.fastk_period.unwrap_or(14);
                        fastk_periods.push(fk as i32);
                        first_valids.push(fv);
                        let first_k = fv + fk as i32 - 1;
                        first_kraws.push(first_k);
                        let first_sk = if slowk_is_sma {
                            first_k + slowk_p0 as i32 - 1
                        } else {
                            first_k
                        };
                        first_slowks.push(first_sk);
                    }

                    let d_fastk = DeviceBuffer::from_slice(&fastk_periods).map_err(CudaStochError::Cuda)?;
                    let d_first = DeviceBuffer::from_slice(&first_valids).map_err(CudaStochError::Cuda)?;
                    let d_first_kraw = DeviceBuffer::from_slice(&first_kraws).map_err(CudaStochError::Cuda)?;
                    let d_first_slowk = DeviceBuffer::from_slice(&first_slowks).map_err(CudaStochError::Cuda)?;

                    // Time-major temporaries: rawK (reused for slowD) and slowK.
                    let tm_total = tm_elems;
                    let mut d_kraw_tm: DeviceBuffer<f32> =
                        unsafe { DeviceBuffer::uninitialized(tm_total) }.map_err(CudaStochError::Cuda)?;
                    let mut d_slowk_tm: DeviceBuffer<f32> =
                        unsafe { DeviceBuffer::uninitialized(tm_total) }.map_err(CudaStochError::Cuda)?;

                    // raw %K for all rows
                    {
                        let block_x: u32 = 256;
                        let grid_x: u32 = ((rows_total as u32) + block_x - 1) / block_x;
                        let grid: GridSize = (grid_x.max(1), 1, 1).into();
                        let block: BlockSize = (block_x, 1, 1).into();
                        unsafe {
                            let mut p_h = d_high.as_device_ptr().as_raw();
                            let mut p_l = d_low.as_device_ptr().as_raw();
                            let mut p_c = d_close.as_device_ptr().as_raw();
                            let mut p_fastk = d_fastk.as_device_ptr().as_raw();
                            let mut p_first = d_first.as_device_ptr().as_raw();
                            let mut p_len = len as i32;
                            let mut p_n = rows_total as i32;
                            let mut p_out = d_kraw_tm.as_device_ptr().as_raw();
                            let args: &mut [*mut c_void] = &mut [
                                &mut p_h as *mut _ as *mut c_void,
                                &mut p_l as *mut _ as *mut c_void,
                                &mut p_c as *mut _ as *mut c_void,
                                &mut p_fastk as *mut _ as *mut c_void,
                                &mut p_first as *mut _ as *mut c_void,
                                &mut p_len as *mut _ as *mut c_void,
                                &mut p_n as *mut _ as *mut c_void,
                                &mut p_out as *mut _ as *mut c_void,
                            ];
                            self.stream
                                .launch(&func_kraw_many, grid, block, 0, args)
                                .map_err(CudaStochError::Cuda)?;
                        }
                    }

                    // slowK for all rows (into d_slowk_tm)
                    if slowk_is_sma {
                        let block_x: u32 = 256;
                        let grid_x: u32 = ((rows_total as u32) + block_x - 1) / block_x;
                        let grid: GridSize = (grid_x.max(1), 1, 1).into();
                        let block: BlockSize = (block_x, 1, 1).into();
                        unsafe {
                            let mut p_prices = d_kraw_tm.as_device_ptr().as_raw();
                            let mut p_first = d_first_kraw.as_device_ptr().as_raw();
                            let mut p_num_series = rows_total as i32;
                            let mut p_len = len as i32;
                            let mut p_period = slowk_p0 as i32;
                            let mut p_out = d_slowk_tm.as_device_ptr().as_raw();
                            let args: &mut [*mut c_void] = &mut [
                                &mut p_prices as *mut _ as *mut c_void,
                                &mut p_first as *mut _ as *mut c_void,
                                &mut p_num_series as *mut _ as *mut c_void,
                                &mut p_len as *mut _ as *mut c_void,
                                &mut p_period as *mut _ as *mut c_void,
                                &mut p_out as *mut _ as *mut c_void,
                            ];
                            self.stream
                                .launch(&func_sma, grid, block, 0, args)
                                .map_err(CudaStochError::Cuda)?;
                        }
                    } else {
                        let alpha: f32 = 2.0f32 / (slowk_p0 as f32 + 1.0f32);
                        let (grid, block): (GridSize, BlockSize) = if func_ema_coalesced.is_some() {
                            let block_x: u32 = 256;
                            let grid_x: u32 = ((rows_total as u32) + block_x - 1) / block_x;
                            (
                                (grid_x.max(1), 1u32, 1u32).into(),
                                (block_x, 1u32, 1u32).into(),
                            )
                        } else {
                            (
                                (rows_total as u32, 1u32, 1u32).into(),
                                (256u32, 1u32, 1u32).into(),
                            )
                        };
                        let f = func_ema_coalesced.as_ref().unwrap_or(&func_ema);
                        unsafe {
                            let mut p_prices = d_kraw_tm.as_device_ptr().as_raw();
                            let mut p_first = d_first_kraw.as_device_ptr().as_raw();
                            let mut p_period = slowk_p0 as i32;
                            let mut p_alpha = alpha;
                            let mut p_num_series = rows_total as i32;
                            let mut p_len = len as i32;
                            let mut p_out = d_slowk_tm.as_device_ptr().as_raw();
                            let args: &mut [*mut c_void] = &mut [
                                &mut p_prices as *mut _ as *mut c_void,
                                &mut p_first as *mut _ as *mut c_void,
                                &mut p_period as *mut _ as *mut c_void,
                                &mut p_alpha as *mut _ as *mut c_void,
                                &mut p_num_series as *mut _ as *mut c_void,
                                &mut p_len as *mut _ as *mut c_void,
                                &mut p_out as *mut _ as *mut c_void,
                            ];
                            self.stream
                                .launch(f, grid, block, 0, args)
                                .map_err(CudaStochError::Cuda)?;
                        }
                    }

                    // slowD for all rows (reuse d_kraw_tm as output)
                    if slowd_is_sma {
                        let block_x: u32 = 256;
                        let grid_x: u32 = ((rows_total as u32) + block_x - 1) / block_x;
                        let grid: GridSize = (grid_x.max(1), 1, 1).into();
                        let block: BlockSize = (block_x, 1, 1).into();
                        unsafe {
                            let mut p_prices = d_slowk_tm.as_device_ptr().as_raw();
                            let mut p_first = d_first_slowk.as_device_ptr().as_raw();
                            let mut p_num_series = rows_total as i32;
                            let mut p_len = len as i32;
                            let mut p_period = slowd_p0 as i32;
                            let mut p_out = d_kraw_tm.as_device_ptr().as_raw();
                            let args: &mut [*mut c_void] = &mut [
                                &mut p_prices as *mut _ as *mut c_void,
                                &mut p_first as *mut _ as *mut c_void,
                                &mut p_num_series as *mut _ as *mut c_void,
                                &mut p_len as *mut _ as *mut c_void,
                                &mut p_period as *mut _ as *mut c_void,
                                &mut p_out as *mut _ as *mut c_void,
                            ];
                            self.stream
                                .launch(&func_sma, grid, block, 0, args)
                                .map_err(CudaStochError::Cuda)?;
                        }
                    } else {
                        let alpha: f32 = 2.0f32 / (slowd_p0 as f32 + 1.0f32);
                        let (grid, block): (GridSize, BlockSize) = if func_ema_coalesced.is_some() {
                            let block_x: u32 = 256;
                            let grid_x: u32 = ((rows_total as u32) + block_x - 1) / block_x;
                            (
                                (grid_x.max(1), 1u32, 1u32).into(),
                                (block_x, 1u32, 1u32).into(),
                            )
                        } else {
                            (
                                (rows_total as u32, 1u32, 1u32).into(),
                                (256u32, 1u32, 1u32).into(),
                            )
                        };
                        let f = func_ema_coalesced.as_ref().unwrap_or(&func_ema);
                        unsafe {
                            let mut p_prices = d_slowk_tm.as_device_ptr().as_raw();
                            let mut p_first = d_first_slowk.as_device_ptr().as_raw();
                            let mut p_period = slowd_p0 as i32;
                            let mut p_alpha = alpha;
                            let mut p_num_series = rows_total as i32;
                            let mut p_len = len as i32;
                            let mut p_out = d_kraw_tm.as_device_ptr().as_raw();
                            let args: &mut [*mut c_void] = &mut [
                                &mut p_prices as *mut _ as *mut c_void,
                                &mut p_first as *mut _ as *mut c_void,
                                &mut p_period as *mut _ as *mut c_void,
                                &mut p_alpha as *mut _ as *mut c_void,
                                &mut p_num_series as *mut _ as *mut c_void,
                                &mut p_len as *mut _ as *mut c_void,
                                &mut p_out as *mut _ as *mut c_void,
                            ];
                            self.stream
                                .launch(f, grid, block, 0, args)
                                .map_err(CudaStochError::Cuda)?;
                        }
                    }

                    // Transpose TM -> RM into final K/D outputs.
                    {
                        let block: BlockSize = (32u32, 8u32, 1u32).into();
                        let grid_x: u32 = ((rows_total as u32) + 32 - 1) / 32;
                        let grid_y: u32 = ((len as u32) + 32 - 1) / 32;
                        let grid: GridSize = (grid_x.max(1), grid_y.max(1), 1u32).into();
                        unsafe {
                            let mut p_in = d_slowk_tm.as_device_ptr().as_raw();
                            let mut p_rows = len as i32;
                            let mut p_cols = rows_total as i32;
                            let mut p_out = d_k.as_device_ptr().as_raw();
                            let args: &mut [*mut c_void] = &mut [
                                &mut p_in as *mut _ as *mut c_void,
                                &mut p_rows as *mut _ as *mut c_void,
                                &mut p_cols as *mut _ as *mut c_void,
                                &mut p_out as *mut _ as *mut c_void,
                            ];
                            self.stream
                                .launch(&func_transpose, grid, block, 0, args)
                                .map_err(CudaStochError::Cuda)?;
                        }
                        unsafe {
                            let mut p_in = d_kraw_tm.as_device_ptr().as_raw();
                            let mut p_rows = len as i32;
                            let mut p_cols = rows_total as i32;
                            let mut p_out = d_d.as_device_ptr().as_raw();
                            let args: &mut [*mut c_void] = &mut [
                                &mut p_in as *mut _ as *mut c_void,
                                &mut p_rows as *mut _ as *mut c_void,
                                &mut p_cols as *mut _ as *mut c_void,
                                &mut p_out as *mut _ as *mut c_void,
                            ];
                            self.stream
                                .launch(&func_transpose, grid, block, 0, args)
                                .map_err(CudaStochError::Cuda)?;
                        }
                    }

                    self.stream.synchronize().map_err(CudaStochError::Cuda)?;
                    return Ok(CudaStochBatch {
                        k: DeviceArrayF32 { buf: d_k, rows: rows_total, cols: len },
                        d: DeviceArrayF32 { buf: d_d, rows: rows_total, cols: len },
                        combos,
                    });
                    }
                }
            }
        }

        // Group parameter rows by fastk period so we reuse Kraw.
        use std::collections::HashMap;
        let mut by_fastk: HashMap<usize, Vec<usize>> = HashMap::new();
        for (row, prm) in combos.iter().enumerate() {
            by_fastk.entry(prm.fastk_period.unwrap_or(14)).or_default().push(row);
        }

        // Reusable device temporaries
        let mut d_hh: Option<DeviceBuffer<f32>> = None;
        let mut d_ll: Option<DeviceBuffer<f32>> = None;
        let mut d_kraw: Option<DeviceBuffer<f32>> = None;

        // Note: For smoothing stages we must pass the first index where the input is finite.
        // This depends on the upstream stage:
        //  - Kraw first valid: first_valid_ohlc + fastk - 1
        //  - slowK first valid: (first_valid_kraw) + slowk_period - 1

        // Helper: 1D launch config
        let launch_1d = |n: usize| -> (GridSize, BlockSize) {
            let block_x: u32 = 256;
            let grid_x: u32 = ((n as u32) + block_x - 1) / block_x;
            ((grid_x.max(1), 1, 1).into(), (block_x, 1, 1).into())
        };

        let norm = |s: &str| s.to_ascii_lowercase();

        // Host-side rolling HH/LL inputs (convert once; reuse across fastk groups).
        let high_f64: Vec<f64> = high_f32.iter().map(|&v| v as f64).collect();
        let low_f64: Vec<f64> = low_f32.iter().map(|&v| v as f64).collect();
        let mut hh_host = vec![f32::NAN; len];
        let mut ll_host = vec![f32::NAN; len];

        for (fkp, rows_in_group) in by_fastk {
            // Build hh/ll on host (f64 helpers → cast to f32)
            hh_host.fill(f32::NAN);
            ll_host.fill(f32::NAN);
            let highs = max_rolling(&high_f64[first_valid..], fkp)
                .map_err(|e| CudaStochError::InvalidInput(e.to_string()))?;
            let lows = min_rolling(&low_f64[first_valid..], fkp)
                .map_err(|e| CudaStochError::InvalidInput(e.to_string()))?;
            for (i, &v) in highs.iter().enumerate() {
                if v.is_finite() {
                    hh_host[first_valid + i] = v as f32;
                }
            }
            for (i, &v) in lows.iter().enumerate() {
                if v.is_finite() {
                    ll_host[first_valid + i] = v as f32;
                }
            }

            // Ensure device buffers sized
            if d_hh.as_ref().map(|b| b.len()).unwrap_or(0) != len {
                d_hh = Some(unsafe { DeviceBuffer::uninitialized(len) }.map_err(CudaStochError::Cuda)?);
            }
            if d_ll.as_ref().map(|b| b.len()).unwrap_or(0) != len {
                d_ll = Some(unsafe { DeviceBuffer::uninitialized(len) }.map_err(CudaStochError::Cuda)?);
            }
            if d_kraw.as_ref().map(|b| b.len()).unwrap_or(0) != len {
                d_kraw = Some(unsafe { DeviceBuffer::uninitialized(len) }.map_err(CudaStochError::Cuda)?);
            }
            let d_hh_ref = d_hh.as_mut().unwrap();
            let d_ll_ref = d_ll.as_mut().unwrap();
            let d_kraw_ref = d_kraw.as_mut().unwrap();

            // Async H2D hh/ll
            unsafe {
                d_hh_ref
                    .async_copy_from(&hh_host, &self.stream)
                    .map_err(CudaStochError::Cuda)?;
                d_ll_ref
                    .async_copy_from(&ll_host, &self.stream)
                    .map_err(CudaStochError::Cuda)?;
            }

            // Launch raw %K (grid‑stride parallel)
            {
                let (grid, block) = launch_1d(len);
                unsafe {
                    let mut p_close = d_close.as_device_ptr().as_raw();
                    let mut p_hh = d_hh_ref.as_device_ptr().as_raw();
                    let mut p_ll = d_ll_ref.as_device_ptr().as_raw();
                    let mut p_len = len as i32;
                    let mut p_fv = first_valid as i32;
                    let mut p_fastk = fkp as i32;
                    let mut p_out = d_kraw_ref.as_device_ptr().as_raw();
                    let args: &mut [*mut c_void] = &mut [
                        &mut p_close as *mut _ as *mut c_void,
                        &mut p_hh as *mut _ as *mut c_void,
                        &mut p_ll as *mut _ as *mut c_void,
                        &mut p_len as *mut _ as *mut c_void,
                        &mut p_fv as *mut _ as *mut c_void,
                        &mut p_fastk as *mut _ as *mut c_void,
                        &mut p_out as *mut _ as *mut c_void,
                    ];
                    self.stream
                        .launch(&func_kraw, grid, block, 0, args)
                        .map_err(CudaStochError::Cuda)?;
                }
            }

            // Partition rows by slowK settings
            #[derive(Hash, Eq, PartialEq, Clone)]
            struct SlowKKey { ty: String, p: usize }
            let mut by_slowk: HashMap<SlowKKey, Vec<usize>> = HashMap::new();
            for &row in &rows_in_group {
                let prm = &combos[row];
                let ty = norm(prm.slowk_ma_type.as_deref().unwrap_or("sma"));
                let p = prm.slowk_period.unwrap_or(3);
                by_slowk.entry(SlowKKey { ty, p }).or_default().push(row);
            }

            for (sk_key, rows_sk) in by_slowk {
                let first_kraw = first_valid + fkp - 1;
                let d_first_kraw =
                    DeviceBuffer::from_slice(&[first_kraw as i32]).map_err(CudaStochError::Cuda)?;
                // Compute slowK once from device d_kraw_ref (1 column time-major)
                let slowk_dev_buf = if sk_key.ty == "sma" {
                    let mut out: DeviceBuffer<f32> =
                        unsafe { DeviceBuffer::uninitialized(len) }.map_err(CudaStochError::Cuda)?;
                    let grid: GridSize = (1u32, 1u32, 1u32).into();
                    let block: BlockSize = (256u32, 1u32, 1u32).into();
                    unsafe {
                        let mut p_prices = d_kraw_ref.as_device_ptr().as_raw();
                        let mut p_first = d_first_kraw.as_device_ptr().as_raw();
                        let mut p_num_series = 1i32;
                        let mut p_len = len as i32;
                        let mut p_period = sk_key.p as i32;
                        let mut p_out = out.as_device_ptr().as_raw();
                        let args: &mut [*mut c_void] = &mut [
                            &mut p_prices as *mut _ as *mut c_void,
                            &mut p_first as *mut _ as *mut c_void,
                            &mut p_num_series as *mut _ as *mut c_void,
                            &mut p_len as *mut _ as *mut c_void,
                            &mut p_period as *mut _ as *mut c_void,
                            &mut p_out as *mut _ as *mut c_void,
                        ];
                        self.stream
                            .launch(&func_sma, grid, block, 0, args)
                            .map_err(CudaStochError::Cuda)?;
                    }
                    out
                } else if sk_key.ty == "ema" {
                    let mut out: DeviceBuffer<f32> =
                        unsafe { DeviceBuffer::uninitialized(len) }.map_err(CudaStochError::Cuda)?;
                    let alpha: f32 = 2.0f32 / (sk_key.p as f32 + 1.0f32);
                    let grid: GridSize = (1u32, 1u32, 1u32).into();
                    let block: BlockSize = (256u32, 1u32, 1u32).into();
                    unsafe {
                        let mut p_prices = d_kraw_ref.as_device_ptr().as_raw();
                        let mut p_first = d_first_kraw.as_device_ptr().as_raw();
                        let mut p_period = sk_key.p as i32;
                        let mut p_alpha = alpha;
                        let mut p_num_series = 1i32;
                        let mut p_len = len as i32;
                        let mut p_out = out.as_device_ptr().as_raw();
                        let args: &mut [*mut c_void] = &mut [
                            &mut p_prices as *mut _ as *mut c_void,
                            &mut p_first as *mut _ as *mut c_void,
                            &mut p_period as *mut _ as *mut c_void,
                            &mut p_alpha as *mut _ as *mut c_void,
                            &mut p_num_series as *mut _ as *mut c_void,
                            &mut p_len as *mut _ as *mut c_void,
                            &mut p_out as *mut _ as *mut c_void,
                        ];
                        self.stream
                            .launch(&func_ema, grid, block, 0, args)
                            .map_err(CudaStochError::Cuda)?;
                    }
                    out
                } else {
                    // Generic MA via selector: copy once to host then run selector
                        let selector = CudaMaSelector::new(0);
                        let mut kraw_host: LockedBuffer<f32> = unsafe {
                            LockedBuffer::uninitialized(len).map_err(CudaStochError::Cuda)?
                        };
                        unsafe {
                            d_kraw_ref
                                .async_copy_to(kraw_host.as_mut_slice(), &self.stream)
                                .map_err(CudaStochError::Cuda)?;
                        }
                        self.stream
                            .synchronize()
                            .map_err(CudaStochError::Cuda)?;
                    let dev = selector
                        .ma_to_device(&sk_key.ty, CudaMaData::SliceF32(kraw_host.as_slice()), sk_key.p)
                        .map_err(|e| CudaStochError::InvalidInput(format!("slowK: {}", e)))?;
                    dev.buf
                };

                // Broadcast slowK to all K rows in this subgroup
                {
                    let idx_i32: Vec<i32> = rows_sk.iter().map(|&r| r as i32).collect();
                    let d_rows = DeviceBuffer::from_slice(&idx_i32)
                        .map_err(CudaStochError::Cuda)?;
                    let (grid, block) = launch_1d(len);
                    unsafe {
                        let mut p_src = slowk_dev_buf.as_device_ptr().as_raw();
                        let mut p_len = len as i32;
                        let mut p_rows = d_rows.as_device_ptr().as_raw();
                        let mut p_nrows = rows_sk.len() as i32;
                        let mut p_dst = d_k.as_device_ptr().as_raw();
                        let mut p_stride = len as i32;
                        let args: &mut [*mut c_void] = &mut [
                            &mut p_src as *mut _ as *mut c_void,
                            &mut p_len as *mut _ as *mut c_void,
                            &mut p_rows as *mut _ as *mut c_void,
                            &mut p_nrows as *mut _ as *mut c_void,
                            &mut p_dst as *mut _ as *mut c_void,
                            &mut p_stride as *mut _ as *mut c_void,
                        ];
                        self.stream
                            .launch(&func_pack, grid, block, 0, args)
                            .map_err(CudaStochError::Cuda)?;
                    }
                }

                // Partition by slowD settings
                #[derive(Hash, Eq, PartialEq, Clone)]
                struct SlowDKey { ty: String, p: usize }
                let mut by_slowd: HashMap<SlowDKey, Vec<usize>> = HashMap::new();
                for &row in &rows_sk {
                    let prm = &combos[row];
                    let ty = norm(prm.slowd_ma_type.as_deref().unwrap_or("sma"));
                    let p = prm.slowd_period.unwrap_or(3);
                    by_slowd.entry(SlowDKey { ty, p }).or_default().push(row);
                }

                for (sd_key, rows_sd) in by_slowd {
                    let first_slowk = first_valid + fkp - 1 + sk_key.p - 1;
                    let d_first_slowk = DeviceBuffer::from_slice(&[first_slowk as i32])
                        .map_err(CudaStochError::Cuda)?;
                    // slowD once from device slowK
                    let slowd_dev_buf = if sd_key.ty == "sma" {
                        let mut out: DeviceBuffer<f32> =
                            unsafe { DeviceBuffer::uninitialized(len) }.map_err(CudaStochError::Cuda)?;
                        let grid: GridSize = (1u32, 1u32, 1u32).into();
                        let block: BlockSize = (256u32, 1u32, 1u32).into();
                        unsafe {
                            let mut p_prices = slowk_dev_buf.as_device_ptr().as_raw();
                            let mut p_first = d_first_slowk.as_device_ptr().as_raw();
                            let mut p_num_series = 1i32;
                            let mut p_len = len as i32;
                            let mut p_period = sd_key.p as i32;
                            let mut p_out = out.as_device_ptr().as_raw();
                            let args: &mut [*mut c_void] = &mut [
                                &mut p_prices as *mut _ as *mut c_void,
                                &mut p_first as *mut _ as *mut c_void,
                                &mut p_num_series as *mut _ as *mut c_void,
                                &mut p_len as *mut _ as *mut c_void,
                                &mut p_period as *mut _ as *mut c_void,
                                &mut p_out as *mut _ as *mut c_void,
                            ];
                            self.stream
                                .launch(&func_sma, grid, block, 0, args)
                                .map_err(CudaStochError::Cuda)?;
                        }
                        out
                    } else if sd_key.ty == "ema" {
                        let mut out: DeviceBuffer<f32> =
                            unsafe { DeviceBuffer::uninitialized(len) }.map_err(CudaStochError::Cuda)?;
                        let alpha: f32 = 2.0f32 / (sd_key.p as f32 + 1.0f32);
                        let grid: GridSize = (1u32, 1u32, 1u32).into();
                        let block: BlockSize = (256u32, 1u32, 1u32).into();
                        unsafe {
                            let mut p_prices = slowk_dev_buf.as_device_ptr().as_raw();
                            let mut p_first = d_first_slowk.as_device_ptr().as_raw();
                            let mut p_period = sd_key.p as i32;
                            let mut p_alpha = alpha;
                            let mut p_num_series = 1i32;
                            let mut p_len = len as i32;
                            let mut p_out = out.as_device_ptr().as_raw();
                            let args: &mut [*mut c_void] = &mut [
                                &mut p_prices as *mut _ as *mut c_void,
                                &mut p_first as *mut _ as *mut c_void,
                                &mut p_period as *mut _ as *mut c_void,
                                &mut p_alpha as *mut _ as *mut c_void,
                                &mut p_num_series as *mut _ as *mut c_void,
                                &mut p_len as *mut _ as *mut c_void,
                                &mut p_out as *mut _ as *mut c_void,
                            ];
                            self.stream
                                .launch(&func_ema, grid, block, 0, args)
                                .map_err(CudaStochError::Cuda)?;
                        }
                        out
                    } else {
                        let selector = CudaMaSelector::new(0);
                        let mut slowk_host: LockedBuffer<f32> = unsafe {
                            LockedBuffer::uninitialized(len).map_err(CudaStochError::Cuda)?
                        };
                        unsafe {
                            slowk_dev_buf
                                .async_copy_to(slowk_host.as_mut_slice(), &self.stream)
                                .map_err(CudaStochError::Cuda)?;
                        }
                        self.stream
                            .synchronize()
                            .map_err(CudaStochError::Cuda)?;
                        let dev = selector
                            .ma_to_device(&sd_key.ty, CudaMaData::SliceF32(slowk_host.as_slice()), sd_key.p)
                            .map_err(|e| CudaStochError::InvalidInput(format!("slowD: {}", e)))?;
                        dev.buf
                    };

                    // Broadcast slowD into D matrix rows
                    let idx_i32: Vec<i32> = rows_sd.iter().map(|&r| r as i32).collect();
                    let d_rows = DeviceBuffer::from_slice(&idx_i32)
                        .map_err(CudaStochError::Cuda)?;
                    let (grid, block) = launch_1d(len);
                    unsafe {
                        let mut p_src = slowd_dev_buf.as_device_ptr().as_raw();
                        let mut p_len = len as i32;
                        let mut p_rows = d_rows.as_device_ptr().as_raw();
                        let mut p_nrows = rows_sd.len() as i32;
                        let mut p_dst = d_d.as_device_ptr().as_raw();
                        let mut p_stride = len as i32;
                        let args: &mut [*mut c_void] = &mut [
                            &mut p_src as *mut _ as *mut c_void,
                            &mut p_len as *mut _ as *mut c_void,
                            &mut p_rows as *mut _ as *mut c_void,
                            &mut p_nrows as *mut _ as *mut c_void,
                            &mut p_dst as *mut _ as *mut c_void,
                            &mut p_stride as *mut _ as *mut c_void,
                        ];
                        self.stream
                            .launch(&func_pack, grid, block, 0, args)
                            .map_err(CudaStochError::Cuda)?;
                    }
                }
            }
        }

        // Ensure completion
        self.stream.synchronize().map_err(CudaStochError::Cuda)?;

        Ok(CudaStochBatch {
            k: DeviceArrayF32 { buf: d_k, rows: rows_total, cols: len },
            d: DeviceArrayF32 { buf: d_d, rows: rows_total, cols: len },
            combos,
        })
    }

    // ---------------- Many-series × one param (time-major inputs) ----------------
    pub fn stoch_many_series_one_param_time_major_dev(
        &self,
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        cols: usize,
        rows: usize,
        params: &StochParams,
    ) -> Result<(DeviceArrayF32, DeviceArrayF32), CudaStochError> {
        if cols == 0 || rows == 0 {
            return Err(CudaStochError::InvalidInput(
                "series dims must be positive".into(),
            ));
        }
        let total = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaStochError::InvalidInput("size overflow in rows*cols".into()))?;
        if high_tm.len() != total || low_tm.len() != total || close_tm.len() != total {
            return Err(CudaStochError::InvalidInput(
                "time-major inputs must all be rows*cols".into(),
            ));
        }
        // First-valid per series (column) in time-major
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = None;
            for r in 0..rows {
                let idx = r * cols + s;
                if high_tm[idx].is_finite() && low_tm[idx].is_finite() && close_tm[idx].is_finite() {
                    fv = Some(r as i32);
                    break;
                }
            }
            first_valids[s] =
                fv.ok_or_else(|| CudaStochError::InvalidInput(format!("series {} all NaN", s)))?;
        }

        let fastk = params.fastk_period.unwrap_or(14);
        let slowk_p = params.slowk_period.unwrap_or(3);
        let slowd_p = params.slowd_period.unwrap_or(3);
        let slowk_ty = params
            .slowk_ma_type
            .as_ref()
            .map(|s| s.as_str())
            .unwrap_or("sma");
        let slowd_ty = params
            .slowd_ma_type
            .as_ref()
            .map(|s| s.as_str())
            .unwrap_or("sma");

        if fastk == 0 || fastk > rows {
            return Err(CudaStochError::InvalidInput("invalid fastk period".into()));
        }

        // Approximate VRAM check: 3×inputs + K/D outputs (time-major)
        let elems_inputs = total
            .checked_mul(3)
            .ok_or_else(|| CudaStochError::InvalidInput("size overflow".into()))?;
        let elems_outputs = total
            .checked_mul(2)
            .ok_or_else(|| CudaStochError::InvalidInput("size overflow".into()))?;
        let total_elems = elems_inputs
            .checked_add(elems_outputs)
            .ok_or_else(|| CudaStochError::InvalidInput("size overflow".into()))?;
        let required_bytes = total_elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaStochError::InvalidInput("size overflow".into()))?;
        Self::will_fit(required_bytes, 64 * 1024 * 1024)?;

        // H2D
        let d_high = DeviceBuffer::from_slice(high_tm).map_err(CudaStochError::Cuda)?;
        let d_low = DeviceBuffer::from_slice(low_tm).map_err(CudaStochError::Cuda)?;
        let d_close = DeviceBuffer::from_slice(close_tm).map_err(CudaStochError::Cuda)?;
        let d_high = DeviceBuffer::from_slice(high_tm).map_err(CudaStochError::Cuda)?;
        let d_low = DeviceBuffer::from_slice(low_tm).map_err(CudaStochError::Cuda)?;
        let d_close = DeviceBuffer::from_slice(close_tm).map_err(CudaStochError::Cuda)?;
        let d_first = DeviceBuffer::from_slice(&first_valids).map_err(CudaStochError::Cuda)?;
        let mut d_k_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(total) }.map_err(CudaStochError::Cuda)?;

        // Kernel: raw %K time-major
        let func = self
            .module
            .get_function("stoch_many_series_one_param_f32")
            .map_err(|_| CudaStochError::MissingKernelSymbol {
                name: "stoch_many_series_one_param_f32",
            })?;
        let block_x: u32 = 256;
        let grid_x: u32 = ((cols as u32) + block_x - 1) / block_x;
        let block: BlockSize = (block_x, 1, 1).into();
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        unsafe {
            let mut p_h = d_high.as_device_ptr().as_raw();
            let mut p_l = d_low.as_device_ptr().as_raw();
            let mut p_c = d_close.as_device_ptr().as_raw();
            let mut p_first = d_first.as_device_ptr().as_raw();
            let mut p_cols = cols as i32;
            let mut p_rows = rows as i32;
            let mut p_fastk = fastk as i32;
            let mut p_out = d_k_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut p_h as *mut _ as *mut c_void,
                &mut p_l as *mut _ as *mut c_void,
                &mut p_c as *mut _ as *mut c_void,
                &mut p_first as *mut _ as *mut c_void,
                &mut p_cols as *mut _ as *mut c_void,
                &mut p_rows as *mut _ as *mut c_void,
                &mut p_fastk as *mut _ as *mut c_void,
                &mut p_out as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(CudaStochError::Cuda)?;
        }

        // Smoothing (native SMA/EMA many-series path if available; otherwise per-series)
        // K smoothing
        let k_tm: DeviceBuffer<f32> = if slowk_ty.eq_ignore_ascii_case("sma") {
            use crate::cuda::moving_averages::sma_wrapper::CudaSma;
            use crate::indicators::moving_averages::sma::SmaParams as SParams;
            let sma = CudaSma::new(0)
                .map_err(|e| CudaStochError::InvalidInput(e.to_string()))?;
            let params = SParams { period: Some(slowk_p) };
            let dev = sma
                .sma_multi_series_one_param_time_major_dev_from_device(
                    &d_k_tm, &d_first, cols, rows, slowk_p,
                )
                .map_err(|e| CudaStochError::InvalidInput(e.to_string()))?;
            dev.buf
        } else if slowk_ty.eq_ignore_ascii_case("ema") {
            use crate::cuda::moving_averages::ema_wrapper::CudaEma;
            let ema = CudaEma::new(0)
                .map_err(|e| CudaStochError::InvalidInput(e.to_string()))?;
            // Compute EMA directly from device buffers (no host staging)
            let mut d_k_sm: DeviceBuffer<f32> =
                unsafe { DeviceBuffer::uninitialized(total) }.map_err(CudaStochError::Cuda)?;
            let alpha = 2.0f32 / (slowk_p as f32 + 1.0);
            ema
                .ema_many_series_one_param_device(
                    &d_k_tm,
                    &d_first,
                    slowk_p as i32,
                    alpha,
                    cols,
                    rows,
                    &mut d_k_sm,
                )
                .map_err(|e| CudaStochError::InvalidInput(e.to_string()))?;
            d_k_sm
        } else {
            // Fallback: per-series via selector
            let selector = CudaMaSelector::new(0);
            // Stage d_k_tm to host
            let mut k_tm_host = vec![0f32; total];
            d_k_tm
                .copy_to(&mut k_tm_host)
                .map_err(CudaStochError::Cuda)?;
            let mut out_tm = vec![f32::NAN; total];
            for s in 0..cols {
                let mut series = vec![f32::NAN; rows];
                for r in 0..rows {
                    series[r] = k_tm_host[r * cols + s];
                }
                let dev = selector
                    .ma_to_device(slowk_ty, CudaMaData::SliceF32(&series), slowk_p)
                    .map_err(|e| CudaStochError::InvalidInput(format!("slowK many-series: {}", e)))?;
                let mut host_row = vec![0f32; rows];
                dev.buf
                    .copy_to(&mut host_row)
                    .map_err(CudaStochError::Cuda)?;
                for r in 0..rows {
                    out_tm[r * cols + s] = host_row[r];
                }
            }
            let mut tmp: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total) }
                .map_err(CudaStochError::Cuda)?;
            tmp.copy_from(&out_tm)
                .map_err(CudaStochError::Cuda)?;
            tmp
        };

        // D smoothing over K_slow
        let d_tm: DeviceBuffer<f32> = if slowd_ty.eq_ignore_ascii_case("sma") {
            use crate::cuda::moving_averages::sma_wrapper::CudaSma;
            let sma = CudaSma::new(0)
                .map_err(|e| CudaStochError::InvalidInput(e.to_string()))?;
            let dev = sma
                .sma_multi_series_one_param_time_major_dev_from_device(
                    &k_tm, &d_first, cols, rows, slowd_p,
                )
                .map_err(|e| CudaStochError::InvalidInput(e.to_string()))?;
            dev.buf
        } else if slowd_ty.eq_ignore_ascii_case("ema") {
            use crate::cuda::moving_averages::ema_wrapper::CudaEma;
            let ema = CudaEma::new(0)
                .map_err(|e| CudaStochError::InvalidInput(e.to_string()))?;
            // Compute EMA directly from device buffers (no host staging)
            let mut d_d_sm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total) }
                .map_err(CudaStochError::Cuda)?;
            let alpha = 2.0f32 / (slowd_p as f32 + 1.0);
            ema
                .ema_many_series_one_param_device(
                    &k_tm,
                    &d_first,
                    slowd_p as i32,
                    alpha,
                    cols,
                    rows,
                    &mut d_d_sm,
                )
                .map_err(|e| CudaStochError::InvalidInput(e.to_string()))?;
            d_d_sm
        } else {
            // Fallback: per-series via selector again
            let selector = CudaMaSelector::new(0);
            let mut k_tm_host = vec![0f32; total];
            k_tm.copy_to(&mut k_tm_host)
                .map_err(CudaStochError::Cuda)?;
            let mut out_tm = vec![f32::NAN; total];
            for s in 0..cols {
                let mut series = vec![f32::NAN; rows];
                for r in 0..rows {
                    series[r] = k_tm_host[r * cols + s];
                }
                let dev = selector
                    .ma_to_device(slowd_ty, CudaMaData::SliceF32(&series), slowd_p)
                    .map_err(|e| CudaStochError::InvalidInput(format!("slowD many-series: {}", e)))?;
                let mut host_row = vec![0f32; rows];
                dev.buf
                    .copy_to(&mut host_row)
                    .map_err(CudaStochError::Cuda)?;
                for r in 0..rows {
                    out_tm[r * cols + s] = host_row[r];
                }
            }
            let mut tmp: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total) }
                .map_err(CudaStochError::Cuda)?;
            tmp.copy_from(&out_tm)
                .map_err(CudaStochError::Cuda)?;
            tmp
        };

        // Ensure producing work is complete so Python interop can omit 'stream'.
        self.stream.synchronize().map_err(CudaStochError::Cuda)?;

        Ok((
            DeviceArrayF32 {
                buf: k_tm,
                rows,
                cols,
            },
            DeviceArrayF32 {
                buf: d_tm,
                rows,
                cols,
            },
        ))
    }
}

// -------- helpers --------
fn axis_usize((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, CudaStochError> {
    if step == 0 || start == end {
        return Ok(vec![start]);
    }

    let mut v = Vec::new();
    if start < end {
        let mut x = start;
        loop {
            v.push(x);
            match x.checked_add(step) {
                Some(next) if next <= end => x = next,
                Some(_) | None => break,
            }
        }
    } else {
        let mut x = start;
        loop {
            v.push(x);
            match x.checked_sub(step) {
                Some(next) if next >= end => x = next,
                Some(_) | None => break,
            }
        }
    }

    if v.is_empty() {
        Err(CudaStochError::InvalidInput(format!(
            "invalid range: start={} end={} step={}",
            start, end, step
        )))
    } else {
        Ok(v)
    }
}

fn expand_grid_stoch(r: &StochBatchRange) -> Result<Vec<StochParams>, CudaStochError> {
    let fastk = axis_usize(r.fastk_period)?;
    let slowk = axis_usize(r.slowk_period)?;
    let slowd = axis_usize(r.slowd_period)?;

    let combos_len = fastk
        .len()
        .checked_mul(slowk.len())
        .and_then(|v| v.checked_mul(slowd.len()))
        .ok_or_else(|| CudaStochError::InvalidInput("size overflow in expand_grid_stoch".into()))?;

    let mut out = Vec::with_capacity(combos_len);
    for fk in &fastk {
        for sk in &slowk {
            for sd in &slowd {
                out.push(StochParams {
                    fastk_period: Some(*fk),
                    slowk_period: Some(*sk),
                    slowk_ma_type: Some(r.slowk_ma_type.0.clone()),
                    slowd_period: Some(*sd),
                    slowd_ma_type: Some(r.slowd_ma_type.0.clone()),
                });
            }
        }
    }
    Ok(out)
}

// ---------- Benches ----------
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 100_000;
    const PARAM_SWEEP: usize = 128;

    fn bytes_one_series_many_params() -> usize {
        let rows = PARAM_SWEEP;
        let in_bytes = 3 * ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let params_bytes = rows * 4 * std::mem::size_of::<i32>(); // fastk + first + first_kraw + first_slowk
        let tm_bytes = 2 * rows * ONE_SERIES_LEN * std::mem::size_of::<f32>(); // kraw_tm + slowk_tm
        let out_bytes = 2 * rows * ONE_SERIES_LEN * std::mem::size_of::<f32>(); // K + D
        in_bytes + params_bytes + tm_bytes + out_bytes + 64 * 1024 * 1024
    }

    fn synth_hlc_from_close(close: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut high = close.to_vec();
        let mut low = close.to_vec();
        for i in 0..close.len() {
            let v = close[i];
            if !v.is_finite() {
                continue;
            }
            let x = i as f32 * 0.0037;
            let off = (0.0041 * x.sin()).abs() + 0.15;
            high[i] = v + off;
            low[i] = v - off;
        }
        (high, low)
    }

    struct StochBatchDeviceState {
        cuda: CudaStoch,
        func_kraw: Function<'static>,
        func_sma: Function<'static>,
        func_transpose: Function<'static>,
        d_high: DeviceBuffer<f32>,
        d_low: DeviceBuffer<f32>,
        d_close: DeviceBuffer<f32>,
        d_fastk: DeviceBuffer<i32>,
        d_first: DeviceBuffer<i32>,
        d_first_kraw: DeviceBuffer<i32>,
        d_first_slowk: DeviceBuffer<i32>,
        d_kraw_tm: DeviceBuffer<f32>,
        d_slowk_tm: DeviceBuffer<f32>,
        d_k: DeviceBuffer<f32>,
        d_d: DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        rows: usize,
        slowk_p: i32,
        slowd_p: i32,
        block_x_row: u32,
    }
    impl CudaBenchState for StochBatchDeviceState {
        fn launch(&mut self) {
            // rawK for all rows (time-major)
            {
                let grid_x: u32 = ((self.rows as u32) + self.block_x_row - 1) / self.block_x_row;
                let grid: GridSize = (grid_x.max(1), 1, 1).into();
                let block: BlockSize = (self.block_x_row, 1, 1).into();
                unsafe {
                    let mut p_h = self.d_high.as_device_ptr().as_raw();
                    let mut p_l = self.d_low.as_device_ptr().as_raw();
                    let mut p_c = self.d_close.as_device_ptr().as_raw();
                    let mut p_fastk = self.d_fastk.as_device_ptr().as_raw();
                    let mut p_first = self.d_first.as_device_ptr().as_raw();
                    let mut p_len = self.len as i32;
                    let mut p_n = self.rows as i32;
                    let mut p_out = self.d_kraw_tm.as_device_ptr().as_raw();
                    let args: &mut [*mut c_void] = &mut [
                        &mut p_h as *mut _ as *mut c_void,
                        &mut p_l as *mut _ as *mut c_void,
                        &mut p_c as *mut _ as *mut c_void,
                        &mut p_fastk as *mut _ as *mut c_void,
                        &mut p_first as *mut _ as *mut c_void,
                        &mut p_len as *mut _ as *mut c_void,
                        &mut p_n as *mut _ as *mut c_void,
                        &mut p_out as *mut _ as *mut c_void,
                    ];
                    self.cuda
                        .stream
                        .launch(&self.func_kraw, grid, block, 0, args)
                        .expect("stoch rawK launch");
                }
            }

            // slowK for all rows (SMA -> time-major)
            {
                let grid_x: u32 = ((self.rows as u32) + self.block_x_row - 1) / self.block_x_row;
                let grid: GridSize = (grid_x.max(1), 1, 1).into();
                let block: BlockSize = (self.block_x_row, 1, 1).into();
                unsafe {
                    let mut p_prices = self.d_kraw_tm.as_device_ptr().as_raw();
                    let mut p_first = self.d_first_kraw.as_device_ptr().as_raw();
                    let mut p_num_series = self.rows as i32;
                    let mut p_len = self.len as i32;
                    let mut p_period = self.slowk_p;
                    let mut p_out = self.d_slowk_tm.as_device_ptr().as_raw();
                    let args: &mut [*mut c_void] = &mut [
                        &mut p_prices as *mut _ as *mut c_void,
                        &mut p_first as *mut _ as *mut c_void,
                        &mut p_num_series as *mut _ as *mut c_void,
                        &mut p_len as *mut _ as *mut c_void,
                        &mut p_period as *mut _ as *mut c_void,
                        &mut p_out as *mut _ as *mut c_void,
                    ];
                    self.cuda
                        .stream
                        .launch(&self.func_sma, grid, block, 0, args)
                        .expect("stoch slowK sma launch");
                }
            }

            // slowD for all rows (SMA -> reuse kraw_tm as output)
            {
                let grid_x: u32 = ((self.rows as u32) + self.block_x_row - 1) / self.block_x_row;
                let grid: GridSize = (grid_x.max(1), 1, 1).into();
                let block: BlockSize = (self.block_x_row, 1, 1).into();
                unsafe {
                    let mut p_prices = self.d_slowk_tm.as_device_ptr().as_raw();
                    let mut p_first = self.d_first_slowk.as_device_ptr().as_raw();
                    let mut p_num_series = self.rows as i32;
                    let mut p_len = self.len as i32;
                    let mut p_period = self.slowd_p;
                    let mut p_out = self.d_kraw_tm.as_device_ptr().as_raw();
                    let args: &mut [*mut c_void] = &mut [
                        &mut p_prices as *mut _ as *mut c_void,
                        &mut p_first as *mut _ as *mut c_void,
                        &mut p_num_series as *mut _ as *mut c_void,
                        &mut p_len as *mut _ as *mut c_void,
                        &mut p_period as *mut _ as *mut c_void,
                        &mut p_out as *mut _ as *mut c_void,
                    ];
                    self.cuda
                        .stream
                        .launch(&self.func_sma, grid, block, 0, args)
                        .expect("stoch slowD sma launch");
                }
            }

            // Transpose TM -> RM into final K/D outputs.
            {
                let block: BlockSize = (32u32, 8u32, 1u32).into();
                let grid_x: u32 = ((self.rows as u32) + 32 - 1) / 32;
                let grid_y: u32 = ((self.len as u32) + 32 - 1) / 32;
                let grid: GridSize = (grid_x.max(1), grid_y.max(1), 1u32).into();
                unsafe {
                    let mut p_in = self.d_slowk_tm.as_device_ptr().as_raw();
                    let mut p_rows = self.len as i32;
                    let mut p_cols = self.rows as i32;
                    let mut p_out = self.d_k.as_device_ptr().as_raw();
                    let args: &mut [*mut c_void] = &mut [
                        &mut p_in as *mut _ as *mut c_void,
                        &mut p_rows as *mut _ as *mut c_void,
                        &mut p_cols as *mut _ as *mut c_void,
                        &mut p_out as *mut _ as *mut c_void,
                    ];
                    self.cuda
                        .stream
                        .launch(&self.func_transpose, grid, block, 0, args)
                        .expect("stoch transpose K");
                }
                unsafe {
                    let mut p_in = self.d_kraw_tm.as_device_ptr().as_raw();
                    let mut p_rows = self.len as i32;
                    let mut p_cols = self.rows as i32;
                    let mut p_out = self.d_d.as_device_ptr().as_raw();
                    let args: &mut [*mut c_void] = &mut [
                        &mut p_in as *mut _ as *mut c_void,
                        &mut p_rows as *mut _ as *mut c_void,
                        &mut p_cols as *mut _ as *mut c_void,
                        &mut p_out as *mut _ as *mut c_void,
                    ];
                    self.cuda
                        .stream
                        .launch(&self.func_transpose, grid, block, 0, args)
                        .expect("stoch transpose D");
                }
            }

            self.cuda.stream.synchronize().expect("stoch sync");
        }
    }

    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaStoch::new(0).expect("cuda stoch");
        let close = gen_series(ONE_SERIES_LEN);
        let (high, low) = synth_hlc_from_close(&close);

        let first_valid = (0..ONE_SERIES_LEN)
            .find(|&i| high[i].is_finite() && low[i].is_finite() && close[i].is_finite())
            .unwrap_or(0);

        let slowk_p = 3i32;
        let slowd_p = 3i32;
        let rows = PARAM_SWEEP;

        // Per-row params and stage first-valid indices for smoothing.
        let mut fastk_periods = Vec::<i32>::with_capacity(rows);
        let mut first_valids = Vec::<i32>::with_capacity(rows);
        let mut first_kraws = Vec::<i32>::with_capacity(rows);
        let mut first_slowks = Vec::<i32>::with_capacity(rows);
        let fv = first_valid as i32;
        for fk in 14..=(14 + (PARAM_SWEEP as i32) - 1) {
            fastk_periods.push(fk);
            first_valids.push(fv);
            let first_k = fv + fk - 1;
            first_kraws.push(first_k);
            let first_sk = first_k + slowk_p - 1;
            first_slowks.push(first_sk);
        }

        // Upload OHLC needed for raw %K.
        let d_high = DeviceBuffer::from_slice(&high).expect("d_high");
        let d_low = DeviceBuffer::from_slice(&low).expect("d_low");
        let d_close = DeviceBuffer::from_slice(&close).expect("d_close");

        // Upload per-row params
        let d_fastk = DeviceBuffer::from_slice(&fastk_periods).expect("d_fastk");
        let d_first = DeviceBuffer::from_slice(&first_valids).expect("d_first");
        let d_first_kraw = DeviceBuffer::from_slice(&first_kraws).expect("d_first_kraw");
        let d_first_slowk = DeviceBuffer::from_slice(&first_slowks).expect("d_first_slowk");

        // Time-major temporaries: rawK (reused for slowD) and slowK.
        let tm_total = rows * ONE_SERIES_LEN;
        let d_kraw_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(tm_total) }.expect("d_kraw_tm");
        let d_slowk_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(tm_total) }.expect("d_slowk_tm");

        // Final outputs on device (row-major: [rows, len])
        let d_k: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(tm_total) }.expect("d_k");
        let d_d: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(tm_total) }.expect("d_d");

        let func_kraw = cuda
            .module
            .get_function("stoch_one_series_many_params_f32")
            .expect("stoch_one_series_many_params_f32");
        let func_kraw: Function<'static> = unsafe { std::mem::transmute(func_kraw) };
        let func_transpose = cuda
            .module
            .get_function("transpose_tm_to_rm_f32")
            .expect("transpose_tm_to_rm_f32");
        let func_transpose: Function<'static> = unsafe { std::mem::transmute(func_transpose) };
        let func_sma = cuda
            .sma_module
            .get_function("sma_many_series_one_param_f32")
            .expect("sma_many_series_one_param_f32");
        let func_sma: Function<'static> = unsafe { std::mem::transmute(func_sma) };

        cuda.stream.synchronize().expect("sync after prep");
        Box::new(StochBatchDeviceState {
            cuda,
            func_kraw,
            func_sma,
            func_transpose,
            d_high,
            d_low,
            d_close,
            d_fastk,
            d_first,
            d_first_kraw,
            d_first_slowk,
            d_kraw_tm,
            d_slowk_tm,
            d_k,
            d_d,
            len: ONE_SERIES_LEN,
            first_valid,
            rows,
            slowk_p,
            slowd_p,
            block_x_row: 256,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "stoch",
            "one_series_many_params",
            "stoch_cuda_batch_dev",
            "100k_x_128",
            prep_one_series_many_params,
        )
        .with_sample_size(10)
        .with_mem_required(bytes_one_series_many_params())]
    }
}
