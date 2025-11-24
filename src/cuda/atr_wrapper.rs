//! CUDA wrapper for ATR (Average True Range) kernels.
//!
//! Parity goals:
//! - ALMA-style device buffer API returning `DeviceArrayF32`.
//! - Batch (one-series × many-params) and Many-series × one-param (time-major).
//! - NaN warmup identical to scalar: warm = first_valid + period - 1.
//! - VRAM check + simple chunking for large combo counts.

#![cfg(feature = "cuda")]

use crate::indicators::atr::AtrBatchRange;
use cust::context::Context;
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaAtrError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("Out of memory: required={required} bytes, free={free} bytes, headroom={headroom} bytes")]
    OutOfMemory { required: usize, free: usize, headroom: usize },
    #[error("Missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("Invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("Launch configuration too large: grid=({gx},{gy},{gz}), block=({bx},{by},{bz})")]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("Device mismatch: buffer on {buf}, current device {current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("Not implemented")]
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
pub struct CudaAtrPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaAtrPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SeedPlan {
    Prefix2,
    TrOnly,
    OnTheFly,
}

pub struct CudaAtr {
    module: Module,
    stream: Stream,
    _context: Arc<Context>,
    device_id: u32,
    policy: CudaAtrPolicy,
}

/// VRAM-backed array handle for ATR with context guard and device id
pub struct DeviceArrayF32Atr {
    pub buf: DeviceBuffer<f32>,
    pub rows: usize,
    pub cols: usize,
    pub ctx: Arc<Context>,
    pub device_id: u32,
}
impl DeviceArrayF32Atr {
    #[inline]
    pub fn device_ptr(&self) -> u64 { self.buf.as_device_ptr().as_raw() as u64 }
    #[inline]
    pub fn len(&self) -> usize { self.rows * self.cols }
}

impl CudaAtr {
    pub fn new(device_id: usize) -> Result<Self, CudaAtrError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/atr_kernel.ptx"));
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
            _context: context,
            device_id: device_id as u32,
            policy: CudaAtrPolicy::default(),
        })
    }


    fn first_valid_hlc(high: &[f32], low: &[f32], close: &[f32]) -> Result<usize, CudaAtrError> {
        if high.len() == 0 || low.len() == 0 || close.len() == 0 {
            return Err(CudaAtrError::InvalidInput("empty input".into()));
        }
        let len = high.len().min(low.len()).min(close.len());
        for i in 0..len {
            if !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan() {
                return Ok(i);
            }
        }
        Err(CudaAtrError::InvalidInput("all values are NaN".into()))
    }

    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        }
    }

    fn device_will_fit(bytes: usize, headroom: usize) -> bool {
        if !Self::mem_check_enabled() {
            return true;
        }
        if let Ok((free, _)) = mem_get_info() {
            return bytes.saturating_add(headroom) <= free;
        }
        true
    }

    fn will_fit(bytes: usize, headroom: usize) -> Result<(), CudaAtrError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        if let Ok((free, _)) = mem_get_info() {
            if bytes.saturating_add(headroom) > free {
                return Err(CudaAtrError::OutOfMemory {
                    required: bytes,
                    free,
                    headroom,
                });
            }
        }
        Ok(())
    }

    fn validate_launch(
        &self,
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
    ) -> Result<(), CudaAtrError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)?
            .max(1) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)?
            .max(1) as u32;
        let max_grid_y = device
            .get_attribute(DeviceAttribute::MaxGridDimY)?
            .max(1) as u32;
        let max_grid_z = device
            .get_attribute(DeviceAttribute::MaxGridDimZ)?
            .max(1) as u32;

        let threads_per_block = bx
            .saturating_mul(by)
            .saturating_mul(bz);
        if threads_per_block > max_threads
            || gx > max_grid_x
            || gy > max_grid_y
            || gz > max_grid_z
        {
            return Err(CudaAtrError::LaunchConfigTooLarge {
                gx,
                gy,
                gz,
                bx,
                by,
                bz,
            });
        }
        Ok(())
    }

    fn chunk_size_for_batch(n_combos: usize, len: usize) -> usize {
        // Inputs: 3×len f32; params per combo (periods i32, alphas f32, warms i32); outputs: combos×len f32.
        let input_bytes = 3usize
            .saturating_mul(len)
            .saturating_mul(std::mem::size_of::<f32>());
        let params_bytes = n_combos
            .saturating_mul(std::mem::size_of::<i32>() * 2 + std::mem::size_of::<f32>());
        let out_per_combo = len.saturating_mul(std::mem::size_of::<f32>());
        let headroom = 64 * 1024 * 1024; // ~64MB
                                         // Start from all combos and shrink until it fits.
                                         // Start from all combos and shrink until it fits.
        let mut chunk = n_combos.max(1);
        while chunk > 1 {
            let need = input_bytes
                .saturating_add(params_bytes)
                .saturating_add(chunk.saturating_mul(out_per_combo))
                .saturating_add(headroom);
            if Self::device_will_fit(need, 0) {
                break;
            }
            chunk = (chunk + 1) / 2;
        }
        chunk.max(1)
    }

    #[inline]
    fn choose_seed_plan(periods: &[usize], _len: usize) -> SeedPlan {
        // Heuristic: prefer high-accuracy float2 prefix for many combos or longer windows,
        // else prefer TR-only reduction; small/short cases use on-the-fly HLC.
        let n = periods.len().max(1);
        let sum_p: u64 = periods.iter().copied().map(|p| p as u64).sum();
        let avg_p = (sum_p as f32) / (n as f32);
        if n >= 8 || avg_p >= 64.0 {
            SeedPlan::Prefix2
        } else if n >= 3 || avg_p >= 24.0 {
            SeedPlan::TrOnly
        } else {
            SeedPlan::OnTheFly
        }
    }

    #[inline]
    fn chunk_size_for_batch_with_inputs(
        &self,
        n_combos: usize,
        len: usize,
        fixed_input_bytes: usize,
    ) -> usize {
        // Params per combo: period i32, alpha f32, warm i32
        let params_bytes = n_combos
            .saturating_mul(std::mem::size_of::<i32>() * 2 + std::mem::size_of::<f32>());
        let out_per_combo = len.saturating_mul(std::mem::size_of::<f32>());
        let headroom = 64 * 1024 * 1024; // ~64MB
        let mut chunk = n_combos.max(1);
        while chunk > 1 {
            let need = fixed_input_bytes
                .saturating_add(params_bytes)
                .saturating_add(chunk.saturating_mul(out_per_combo))
                .saturating_add(headroom);
            if Self::device_will_fit(need, 0) {
                break;
            }
            chunk = (chunk + 1) / 2;
        }
        chunk.max(1)
    }

    pub fn atr_batch_dev(
        &self,
        high: &[f32],
        low: &[f32],
        close: &[f32],
        sweep: &AtrBatchRange,
    ) -> Result<DeviceArrayF32Atr, CudaAtrError> {
        if high.len() != low.len() || low.len() != close.len() {
            return Err(CudaAtrError::InvalidInput("input length mismatch".into()));
        }
        let len = close.len();
        if len == 0 {
            return Err(CudaAtrError::InvalidInput("empty input".into()));
        }
        let first_valid = Self::first_valid_hlc(high, low, close)?;

        // Expand parameter combos (length axis only)
        let (start, end, step) = sweep.length;
        if start == 0 {
            return Err(CudaAtrError::InvalidInput("period must be > 0".into()));
        }
        let periods: Vec<usize> = if step == 0 || start == end {
            vec![start]
        } else if start < end {
            (start..=end).step_by(step).collect()
        } else {
            let mut v: Vec<usize> = (end..=start).step_by(step).collect();
            v.reverse();
            v
        };
        if periods.is_empty() {
            return Err(CudaAtrError::InvalidInput("no parameter combos".into()));
        }
        for &p in &periods {
            if p == 0 || p > len || (len - first_valid) < p {
                return Err(CudaAtrError::InvalidInput(format!(
                    "invalid period {} for data length {} (valid after {}: {})",
                    p,
                    len,
                    first_valid,
                    len - first_valid
                )));
            }
        }

        let n_combos = periods.len();

        // Device params (shared across chunks)
        let h_periods_i32: Vec<i32> = periods.iter().map(|&p| p as i32).collect();
        let h_alphas: Vec<f32> = periods.iter().map(|&p| 1.0f32 / (p as f32)).collect();
        let h_warms: Vec<i32> = periods
            .iter()
            .map(|&p| (first_valid + p - 1) as i32)
            .collect();

        let params_bytes = h_periods_i32.len()
            .checked_mul(std::mem::size_of::<i32>() * 2 + std::mem::size_of::<f32>())
            .ok_or_else(|| CudaAtrError::InvalidInput("params size overflow".into()))?;
        Self::will_fit(params_bytes, 8 * 1024 * 1024)?;

        let d_periods = DeviceBuffer::from_slice(&h_periods_i32)?;
        let d_alphas = DeviceBuffer::from_slice(&h_alphas)?;
        let d_warms = DeviceBuffer::from_slice(&h_warms)?;

        // Heuristic seed plan (no flags)
        let plan = Self::choose_seed_plan(&periods, len);

        // Upload inputs; may be dropped after TR/prefix build.
        let input_elems = len
            .checked_mul(3)
            .ok_or_else(|| CudaAtrError::InvalidInput("input size overflow".into()))?;
        let input_bytes = input_elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaAtrError::InvalidInput("input size overflow".into()))?;
        Self::will_fit(input_bytes, 8 * 1024 * 1024)?;

        let mut d_high: Option<DeviceBuffer<f32>> =
            Some(DeviceBuffer::from_slice(high).map_err(CudaAtrError::Cuda)?);
        let mut d_low: Option<DeviceBuffer<f32>> =
            Some(DeviceBuffer::from_slice(low).map_err(CudaAtrError::Cuda)?);
        let mut d_close: Option<DeviceBuffer<f32>> =
            Some(DeviceBuffer::from_slice(close).map_err(CudaAtrError::Cuda)?);

        // Precompute on device as needed
        let mut d_tr: Option<DeviceBuffer<f32>> = None;
        let mut d_prefix2: Option<DeviceBuffer<[f32; 2]>> = None;

        // Device functions
        let k_batch = match self.module.get_function("atr_batch_unified_f32") {
            Ok(f) => f,
            Err(_e) => {
                // Fallback to legacy kernels if unified symbol is absent
                // Keep behavior compatible with older PTX
                // Try prefix kernel first (legacy), then plain
                return self.atr_batch_dev_legacy(
                    &d_periods,
                    &d_alphas,
                    &d_warms,
                    len,
                    first_valid,
                    n_combos,
                    &mut d_high,
                    &mut d_low,
                    &mut d_close,
                );
            }
        };

        // Try to grab helper kernels; if not found, we can still run on-the-fly path
        let k_tr = self.module.get_function("tr_from_hlc_f32").ok();
        let k_prefix = self
            .module
            .get_function("exclusive_prefix_float2_from_tr")
            .ok();

        if matches!(plan, SeedPlan::Prefix2 | SeedPlan::TrOnly) {
            // Require TR kernel for these plans; if missing, fall back to on-the-fly
            if let Some(k_tr_f) = k_tr {
                // Allocate TR
                let tr_bytes = len
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| CudaAtrError::InvalidInput("tr buffer size overflow".into()))?;
                Self::will_fit(tr_bytes, 8 * 1024 * 1024)?;
                let mut db_tr: DeviceBuffer<f32> =
                    unsafe { DeviceBuffer::uninitialized(len) }?;

                // Launch tr_from_hlc_f32
                let block_tr: BlockSize = (256, 1, 1).into();
                let grid_tr_x = ((len as u32) + 256 - 1) / 256;
                let grid_tr: GridSize = (grid_tr_x.max(1), 1, 1).into();

                unsafe {
                    let mut high_ptr = d_high.as_ref().unwrap().as_device_ptr().as_raw();
                    let mut low_ptr = d_low.as_ref().unwrap().as_device_ptr().as_raw();
                    let mut close_ptr = d_close.as_ref().unwrap().as_device_ptr().as_raw();
                    let mut len_i = len as i32;
                    let mut first_i = first_valid as i32;
                    let mut tr_ptr = db_tr.as_device_ptr().as_raw();
                    let args: &mut [*mut c_void] = &mut [
                        &mut high_ptr as *mut _ as *mut c_void,
                        &mut low_ptr as *mut _ as *mut c_void,
                        &mut close_ptr as *mut _ as *mut c_void,
                        &mut len_i as *mut _ as *mut c_void,
                        &mut first_i as *mut _ as *mut c_void,
                        &mut tr_ptr as *mut _ as *mut c_void,
                    ];
                    self.validate_launch(grid_tr_x.max(1), 1, 1, 256, 1, 1)?;
                    self.stream.launch(&k_tr_f, grid_tr, block_tr, 0, args)?;
                }

                if matches!(plan, SeedPlan::Prefix2) {
                    if let Some(k_pf) = k_prefix {
                        // Exclusive prefix (float2) length len+1
                        let pfx_elems = len
                            .checked_add(1)
                            .ok_or_else(|| {
                                CudaAtrError::InvalidInput("prefix buffer size overflow".into())
                            })?;
                        let pfx_bytes = pfx_elems
                            .checked_mul(std::mem::size_of::<[f32; 2]>())
                            .ok_or_else(|| {
                                CudaAtrError::InvalidInput("prefix buffer size overflow".into())
                            })?;
                        Self::will_fit(pfx_bytes, 8 * 1024 * 1024)?;
                        let mut db_pfx: DeviceBuffer<[f32; 2]> =
                            unsafe { DeviceBuffer::uninitialized(len + 1) }?;
                        let block_pf: BlockSize = (1, 1, 1).into();
                        let grid_pf: GridSize = (1, 1, 1).into();
                        unsafe {
                            let mut tr_ptr = db_tr.as_device_ptr().as_raw();
                            let mut len_i = len as i32;
                            let mut prefix_ptr = db_pfx.as_device_ptr().as_raw();
                            let args: &mut [*mut c_void] = &mut [
                                &mut tr_ptr as *mut _ as *mut c_void,
                                &mut len_i as *mut _ as *mut c_void,
                                &mut prefix_ptr as *mut _ as *mut c_void,
                            ];
                            self.validate_launch(1, 1, 1, 1, 1, 1)?;
                            self.stream.launch(&k_pf, grid_pf, block_pf, 0, args)?;
                        }
                        // We can drop H/L/C now to free VRAM
                        self.synchronize()?;
                        d_tr = Some(db_tr);
                        d_prefix2 = Some(db_pfx);
                        d_high = None;
                        d_low = None;
                        d_close = None;
                    } else {
                        // Prefix kernel missing; degrade to TR-only
                        self.synchronize()?;
                        d_tr = Some(db_tr);
                        d_high = None;
                        d_low = None;
                        d_close = None;
                    }
                } else {
                    // TR-only
                    self.synchronize()?;
                    d_tr = Some(db_tr);
                    d_high = None;
                    d_low = None;
                    d_close = None;
                }
            }
        }

        // Fixed input bytes for chunking
        let fixed_input_bytes = match (&d_tr, &d_prefix2) {
            (Some(_), Some(_)) => {
                len * std::mem::size_of::<f32>() + (len + 1) * std::mem::size_of::<[f32; 2]>()
            }
            (Some(_), None) => len * std::mem::size_of::<f32>(),
            (None, None) => 3 * len * std::mem::size_of::<f32>(),
            _ => unreachable!(),
        };

        // Chunk combos given current fixed inputs
        let chunk = self.chunk_size_for_batch_with_inputs(n_combos, len, fixed_input_bytes);

        // Launch unified batch kernel (pointers decide seed path)
        let block_x = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => block_x.max(32),
            BatchKernelPolicy::Auto => 64,
        };
        let block: BlockSize = (block_x, 1, 1).into();

        // Output buffer (n_combos x len)
        let out_elems = n_combos
            .checked_mul(len)
            .ok_or_else(|| CudaAtrError::InvalidInput("n_combos*len overflow".into()))?;
        let out_bytes = out_elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaAtrError::InvalidInput("output size overflow".into()))?;
        Self::will_fit(out_bytes, 16 * 1024 * 1024)?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(out_elems) }?;

        let mut launched = 0usize;
        while launched < n_combos {
            let cur = (n_combos - launched).min(chunk);
            let grid_x = cur as u32;
            let grid: GridSize = (grid_x, 1, 1).into();

            unsafe {
                let mut high_ptr = if d_tr.is_some() {
                    0u64
                } else {
                    d_high.as_ref().unwrap().as_device_ptr().as_raw()
                };
                let mut low_ptr = if d_tr.is_some() {
                    0u64
                } else {
                    d_low.as_ref().unwrap().as_device_ptr().as_raw()
                };
                let mut close_ptr = if d_tr.is_some() {
                    0u64
                } else {
                    d_close.as_ref().unwrap().as_device_ptr().as_raw()
                };
                let mut tr_ptr = d_tr
                    .as_ref()
                    .map(|b| b.as_device_ptr().as_raw())
                    .unwrap_or(0u64);
                let mut pfx_ptr = d_prefix2
                    .as_ref()
                    .map(|b| b.as_device_ptr().as_raw())
                    .unwrap_or(0u64);

                let mut periods_ptr = d_periods
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((launched * std::mem::size_of::<i32>()) as u64);
                let mut alphas_ptr = d_alphas
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((launched * std::mem::size_of::<f32>()) as u64);
                let mut warms_ptr = d_warms
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((launched * std::mem::size_of::<i32>()) as u64);

                let mut len_i = len as i32;
                let mut first_i = first_valid as i32;
                let mut cur_i = cur as i32;
                let mut out_ptr = d_out
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((launched * len * std::mem::size_of::<f32>()) as u64);

                let args: &mut [*mut c_void] = &mut [
                    &mut high_ptr as *mut _ as *mut c_void,
                    &mut low_ptr as *mut _ as *mut c_void,
                    &mut close_ptr as *mut _ as *mut c_void,
                    &mut tr_ptr as *mut _ as *mut c_void,
                    &mut pfx_ptr as *mut _ as *mut c_void,
                    &mut periods_ptr as *mut _ as *mut c_void,
                    &mut alphas_ptr as *mut _ as *mut c_void,
                    &mut warms_ptr as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut first_i as *mut _ as *mut c_void,
                    &mut cur_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.validate_launch(grid_x, 1, 1, block_x, 1, 1)?;
                self.stream.launch(&k_batch, grid, block, 0, args)?;
            }

            launched += cur;
        }

        Ok(DeviceArrayF32Atr { buf: d_out, rows: n_combos, cols: len, ctx: Arc::clone(&self._context), device_id: self.device_id })
    }

    // Legacy fallback if unified symbol missing; keep API stable
    fn atr_batch_dev_legacy(
        &self,
        d_periods: &DeviceBuffer<i32>,
        d_alphas: &DeviceBuffer<f32>,
        d_warms: &DeviceBuffer<i32>,
        len: usize,
        first_valid: usize,
        n_combos: usize,
        d_high: &mut Option<DeviceBuffer<f32>>,
        d_low: &mut Option<DeviceBuffer<f32>>,
        d_close: &mut Option<DeviceBuffer<f32>>,
    ) -> Result<DeviceArrayF32Atr, CudaAtrError> {
        // Prefer old prefix kernel if available; else plain
        if let Ok(func) = self.module.get_function("atr_batch_from_tr_prefix_f32") {
            // Build TR+prefix on host would be required, but we avoid FP64: fall back to plain kernel
            // to keep code simple here when unified is missing.
            drop(func);
        }
        let func = self
            .module
            .get_function("atr_batch_f32")
            .map_err(|_| CudaAtrError::MissingKernelSymbol { name: "atr_batch_f32" })?;

        let n_combos = n_combos;
        let chunk = Self::chunk_size_for_batch(n_combos, len);
        let block_x = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => block_x,
            BatchKernelPolicy::Auto => 256,
        };
        let block: BlockSize = (block_x, 1, 1).into();
        let out_elems = n_combos
            .checked_mul(len)
            .ok_or_else(|| CudaAtrError::InvalidInput("n_combos*len overflow".into()))?;
        let out_bytes = out_elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaAtrError::InvalidInput("output size overflow".into()))?;
        Self::will_fit(out_bytes, 16 * 1024 * 1024)?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(out_elems) }?;

        let mut launched = 0usize;
        while launched < n_combos {
            let cur = (n_combos - launched).min(chunk);
            let grid_x = cur as u32;
            let grid: GridSize = (grid_x, 1, 1).into();
            unsafe {
                let mut high_ptr = d_high.as_mut().unwrap().as_device_ptr().as_raw();
                let mut low_ptr = d_low.as_mut().unwrap().as_device_ptr().as_raw();
                let mut close_ptr = d_close.as_mut().unwrap().as_device_ptr().as_raw();
                let mut periods_ptr = d_periods
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((launched * std::mem::size_of::<i32>()) as u64);
                let mut alphas_ptr = d_alphas
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((launched * std::mem::size_of::<f32>()) as u64);
                let mut warms_ptr = d_warms
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((launched * std::mem::size_of::<i32>()) as u64);
                let mut len_i = len as i32;
                let mut first_i = first_valid as i32;
                let mut cur_i = cur as i32;
                let mut out_ptr = d_out
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((launched * len * std::mem::size_of::<f32>()) as u64);
                let args: &mut [*mut c_void] = &mut [
                    &mut high_ptr as *mut _ as *mut c_void,
                    &mut low_ptr as *mut _ as *mut c_void,
                    &mut close_ptr as *mut _ as *mut c_void,
                    &mut periods_ptr as *mut _ as *mut c_void,
                    &mut alphas_ptr as *mut _ as *mut c_void,
                    &mut warms_ptr as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut first_i as *mut _ as *mut c_void,
                    &mut cur_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.validate_launch(grid_x, 1, 1, block_x, 1, 1)?;
                self.stream.launch(&func, grid, block, 0, args)?;
            }
            launched += cur;
        }
        Ok(DeviceArrayF32Atr { buf: d_out, rows: n_combos, cols: len, ctx: Arc::clone(&self._context), device_id: self.device_id })
    }

    fn first_valids_time_major(
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        cols: usize,
        rows: usize,
    ) -> Result<Vec<i32>, CudaAtrError> {
        let n = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaAtrError::InvalidInput("rows*cols overflow".into()))?;
        if high_tm.len() != n || low_tm.len() != n || close_tm.len() != n {
            return Err(CudaAtrError::InvalidInput(
                "time-major input length mismatch".into(),
            ));
        }
        let mut out = vec![-1i32; cols];
        for s in 0..cols {
            for t in 0..rows {
                let idx = t * cols + s;
                let h = high_tm[idx];
                let l = low_tm[idx];
                let c = close_tm[idx];
                if !h.is_nan() && !l.is_nan() && !c.is_nan() {
                    out[s] = t as i32;
                    break;
                }
            }
        }
        Ok(out)
    }

    pub fn atr_many_series_one_param_time_major_dev(
        &self,
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
    ) -> Result<DeviceArrayF32Atr, CudaAtrError> {
        if period == 0 {
            return Err(CudaAtrError::InvalidInput("period must be > 0".into()));
        }
        if period == 0 {
            return Err(CudaAtrError::InvalidInput("period must be > 0".into()));
        }
        let first_valids = Self::first_valids_time_major(high_tm, low_tm, close_tm, cols, rows)?;
        if rows < period {
            return Err(CudaAtrError::InvalidInput(
                "not enough rows for period".into(),
            ));
        }
        if rows < period {
            return Err(CudaAtrError::InvalidInput(
                "not enough rows for period".into(),
            ));
        }

        let elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaAtrError::InvalidInput("rows*cols overflow".into()))?;
        let inputs_bytes = elems
            .checked_mul(3)
            .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| CudaAtrError::InvalidInput("input size overflow".into()))?;
        let fv_bytes = cols
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaAtrError::InvalidInput("first_valids size overflow".into()))?;
        let out_bytes = elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaAtrError::InvalidInput("output size overflow".into()))?;
        let total_bytes = inputs_bytes
            .checked_add(fv_bytes)
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| CudaAtrError::InvalidInput("total size overflow".into()))?;
        Self::will_fit(total_bytes, 16 * 1024 * 1024)?;

        let mut d_high = DeviceBuffer::from_slice(high_tm)?;
        let mut d_low = DeviceBuffer::from_slice(low_tm)?;
        let mut d_close = DeviceBuffer::from_slice(close_tm)?;
        let d_fv = DeviceBuffer::from_slice(&first_valids)?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems) }?;

        // Prefer coalesced time-major kernel name if present, else fallback to legacy symbol.
        let func = match self.module.get_function("atr_many_series_one_param_f32_tm_coalesced") {
            Ok(f) => f,
            Err(_) => self
                .module
                .get_function("atr_many_series_one_param_f32")
                .map_err(|_| CudaAtrError::MissingKernelSymbol { name: "atr_many_series_one_param_f32" })?,
        };
        // Launch config: warp tiles of 32 series; each warp walks time in lockstep.
        let mut block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => block_x,
            ManySeriesKernelPolicy::Auto => 256,
        };
        block_x = (block_x / 32).max(1) * 32; // align to multiples of warps
        let warps_per_block = (block_x / 32) as usize;
        let series_tiles = (cols + 31) / 32;
        let grid_x = ((series_tiles + warps_per_block - 1) / warps_per_block).max(1) as u32;
        let grid: GridSize = (grid_x, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut high_ptr = d_high.as_device_ptr().as_raw();
            let mut low_ptr = d_low.as_device_ptr().as_raw();
            let mut close_ptr = d_close.as_device_ptr().as_raw();
            let mut fv_ptr = d_fv.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut alpha = 1.0f32 / (period as f32);
            let mut num_series_i = cols as i32;
            let mut series_len_i = rows as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut high_ptr as *mut _ as *mut c_void,
                &mut low_ptr as *mut _ as *mut c_void,
                &mut close_ptr as *mut _ as *mut c_void,
                &mut fv_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut alpha as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.validate_launch(grid_x, 1, 1, block_x, 1, 1)?;
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(DeviceArrayF32Atr { buf: d_out, rows, cols, ctx: Arc::clone(&self._context), device_id: self.device_id })
    }

    #[inline]
    pub fn synchronize(&self) -> Result<(), CudaAtrError> { Ok(self.stream.synchronize()?) }

// ---------------- Bench profiles ----------------
// Exclude from test builds to avoid compiling heavy bench prep when running unit tests.

}

#[cfg(not(test))]
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;

    fn bytes_one_series(n_combos: usize) -> usize {
        let in_bytes = 3 * ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = n_combos * ONE_SERIES_LEN * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    fn synth_hlc_from_close(close: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut high = close.to_vec();
        let mut low = close.to_vec();
        for i in 0..close.len() {
            let v = close[i];
            if v.is_nan() {
                continue;
            }
            let x = i as f32 * 0.002f32;
            let off = (0.004 * x.sin()).abs() + 0.12;
            high[i] = v + off;
            low[i] = v - off;
        }
        (high, low)
    }

    struct AtrBatchState {
        cuda: CudaAtr,
        high: Vec<f32>,
        low: Vec<f32>,
        close: Vec<f32>,
        sweep: AtrBatchRange,
    }
    impl CudaBenchState for AtrBatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .atr_batch_dev(&self.high, &self.low, &self.close, &self.sweep)
                .unwrap();
        }
    }

    struct AtrManyState {
        cuda: CudaAtr,
        high_tm: Vec<f32>,
        low_tm: Vec<f32>,
        close_tm: Vec<f32>,
        cols: usize,
        rows: usize,
        period: usize,
    }
    impl CudaBenchState for AtrManyState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .atr_many_series_one_param_time_major_dev(
                    &self.high_tm,
                    &self.low_tm,
                    &self.close_tm,
                    self.cols,
                    self.rows,
                    self.period,
                )
                .unwrap();
        }
    }

    struct BatchPrepCfg;
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let len = ONE_SERIES_LEN;
        let pstart = 5usize;
        let pend = 64usize;
        let pstep = 5usize;
        let close = gen_series(len);
        let (high, low) = synth_hlc_from_close(&close);
        Box::new(AtrBatchState {
            cuda: CudaAtr::new(0).unwrap(),
            high,
            low,
            close,
            sweep: AtrBatchRange {
                length: (pstart, pend, pstep),
            },
        })
    }

    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let (cols, rows, period) = (256usize, 262_144usize, 14usize);
        let mut close_tm = vec![f32::NAN; cols * rows];
        for s in 0..cols {
            for t in s..rows {
                let x = (t as f32) + (s as f32) * 0.2;
                close_tm[t * cols + s] = (x * 0.0017).sin() + 0.00015 * x;
            }
        }
        let (mut high_tm, mut low_tm) = (close_tm.clone(), close_tm.clone());
        for s in 0..cols {
            for t in 0..rows {
                let v = close_tm[t * cols + s];
                if v.is_nan() {
                    continue;
                }
                let x = (t as f32) * 0.002;
                let off = (0.004 * x.cos()).abs() + 0.11;
                high_tm[t * cols + s] = v + off;
                low_tm[t * cols + s] = v - off;
            }
        }
        Box::new(AtrManyState {
            cuda: CudaAtr::new(0).unwrap(),
            high_tm,
            low_tm,
            close_tm,
            cols,
            rows,
            period,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        let pstart = 5usize;
        let pend = 64usize;
        let pstep = 5usize;
        let n_combos = ((pend - pstart) / pstep + 1).max(1);
        let scen_batch = CudaBenchScenario::new(
            "atr",
            "one_series_many_params",
            "atr_cuda_batch_dev",
            "1m_x_params",
            prep_one_series_many_params,
        )
        .with_mem_required(bytes_one_series(n_combos));

        let (cols, rows) = (256usize, 262_144usize);
        let scen_many = CudaBenchScenario::new(
            "atr",
            "many_series_one_param",
            "atr_cuda_many_series_one_param_dev",
            "256x262k",
            prep_many_series_one_param,
        )
        .with_mem_required(
            (3 * cols * rows + cols * rows) * std::mem::size_of::<f32>() + 64 * 1024 * 1024,
        );

        vec![scen_batch, scen_many]
    }
}
