//! CUDA support for Parabolic SAR (Stop and Reverse).
//!
//! Parity with ALMA/CWMA wrappers:
//! - PTX load via include_str!(.../sar_kernel.ptx) with DetermineTargetFromContext + O2
//! - NON_BLOCKING stream
//! - Policy enums and simple selection
//! - VRAM checks with ~64MB headroom and grid.y chunking (<= 65_535)
//! - Public device entry points:
//!     - one-series × many-params (batch)
//!     - many-series × one-param (time‑major)
//! - Numerics & warmup/NaN semantics match src/indicators/sar.rs

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::sar::{SarBatchRange, SarParams};
use cust::context::{CacheConfig, Context};
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaSarError {
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
impl Default for BatchKernelPolicy {
    fn default() -> Self {
        BatchKernelPolicy::Auto
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32, block_y: u32 },
}
impl Default for ManySeriesKernelPolicy {
    fn default() -> Self {
        ManySeriesKernelPolicy::Auto
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaSarPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

pub struct CudaSar {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
    policy: CudaSarPolicy,
    debug_logged: std::sync::atomic::AtomicBool,
}

impl CudaSar {
    pub fn new(device_id: usize) -> Result<Self, CudaSarError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/sar_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => {
                if let Ok(m) =
                    Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext])
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
            policy: CudaSarPolicy::default(),
            debug_logged: std::sync::atomic::AtomicBool::new(false),
        })
    }

    pub fn set_policy(&mut self, p: CudaSarPolicy) {
        self.policy = p;
    }

    #[inline]
    pub fn context_arc(&self) -> Arc<Context> { self.context.clone() }

    #[inline]
    pub fn device_id(&self) -> u32 { self.device_id }

    #[inline]
    pub fn synchronize(&self) -> Result<(), CudaSarError> {
        self.stream.synchronize()?;
        Ok(())
    }

    #[inline]
    fn headroom_bytes() -> usize {
        env::var("CUDA_MEM_HEADROOM")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(64 * 1024 * 1024)
    }
    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
            Err(_) => true,
        }
    }
    #[inline]
    fn device_mem_info() -> Option<(usize, usize)> {
        mem_get_info().ok()
    }
    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaSarError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        if let Some((free, _)) = Self::device_mem_info() {
            if required_bytes.saturating_add(headroom_bytes) <= free {
                Ok(())
            } else {
                Err(CudaSarError::OutOfMemory {
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
    fn validate_launch(
        &self,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
    ) -> Result<(), CudaSarError> {
        let dev = Device::get_device(self.device_id)?;
        let max_bx = dev
            .get_attribute(DeviceAttribute::MaxBlockDimX)?
            as u32;
        let max_by = dev
            .get_attribute(DeviceAttribute::MaxBlockDimY)?
            as u32;
        let max_gx = dev
            .get_attribute(DeviceAttribute::MaxGridDimX)?
            as u32;
        let max_gy = dev
            .get_attribute(DeviceAttribute::MaxGridDimY)?
            as u32;
        if block.0 == 0
            || block.0 > max_bx
            || block.1 == 0
            || block.1 > max_by
            || grid.0 == 0
            || grid.0 > max_gx
            || grid.1 == 0
            || grid.1 > max_gy
        {
            return Err(CudaSarError::LaunchConfigTooLarge {
                gx: grid.0,
                gy: grid.1,
                gz: grid.2,
                bx: block.0,
                by: block.1,
                bz: block.2,
            });
        }
        Ok(())
    }

    
    pub fn sar_batch_dev(
        &self,
        high: &[f32],
        low: &[f32],
        sweep: &SarBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<SarParams>), CudaSarError> {
        if high.is_empty() || low.is_empty() || high.len() != low.len() {
            return Err(CudaSarError::InvalidInput(
                "inputs must be non-empty and same length".into(),
            ));
        }
        let len = high.len();
        let first = first_valid_hl(high, low)
            .ok_or_else(|| CudaSarError::InvalidInput("all values are NaN".into()))?;
        if len - first < 2 {
            return Err(CudaSarError::InvalidInput(
                "not enough valid data (need >= 2 after first)".into(),
            ));
        }

        let combos = expand_grid(sweep)?;

        
        let mut accs = Vec::with_capacity(combos.len());
        let mut maxs = Vec::with_capacity(combos.len());
        for p in &combos {
            let a = p.acceleration.unwrap_or(0.02);
            let m = p.maximum.unwrap_or(0.2);
            if !(a > 0.0) || !(m > 0.0) {
                return Err(CudaSarError::InvalidInput(
                    "invalid acceleration/maximum".into(),
                ));
            }
            accs.push(a as f32);
            maxs.push(m as f32);
        }

        
        let elem_size = std::mem::size_of::<f32>();
        let in_bytes = len
            .checked_mul(2)
            .and_then(|v| v.checked_mul(elem_size))
            .ok_or_else(|| CudaSarError::InvalidInput("size overflow".into()))?;
        let param_bytes = combos
            .len()
            .checked_mul(2)
            .and_then(|v| v.checked_mul(elem_size))
            .ok_or_else(|| CudaSarError::InvalidInput("size overflow".into()))?;
        let out_elems = combos
            .len()
            .checked_mul(len)
            .ok_or_else(|| CudaSarError::InvalidInput("size overflow".into()))?;
        let out_bytes = out_elems
            .checked_mul(elem_size)
            .ok_or_else(|| CudaSarError::InvalidInput("size overflow".into()))?;
        let required = in_bytes
            .checked_add(param_bytes)
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| CudaSarError::InvalidInput("size overflow".into()))?;
        Self::will_fit(required, Self::headroom_bytes())?;

        
        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_accs = DeviceBuffer::from_slice(&accs)?;
        let d_maxs = DeviceBuffer::from_slice(&maxs)?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(out_elems) }?;

        let len_i32 = len
            .try_into()
            .map_err(|_| CudaSarError::InvalidInput("len exceeds i32".into()))?;
        let first_i32 = first
            .try_into()
            .map_err(|_| CudaSarError::InvalidInput("first_valid exceeds i32".into()))?;
        let rows_i32 = combos
            .len()
            .try_into()
            .map_err(|_| CudaSarError::InvalidInput("n_rows exceeds i32".into()))?;

        self.launch_batch_kernel(
            &d_high,
            &d_low,
            len_i32,
            first_i32,
            &d_accs,
            &d_maxs,
            rows_i32,
            &mut d_out,
        )?;

        self.stream.synchronize()?;

        Ok((DeviceArrayF32 { buf: d_out, rows: combos.len(), cols: len }, combos))
    }

    /// Same as `sar_batch_dev`, but reuses device-resident inputs to avoid re-copying.
    pub fn sar_batch_dev_from_device(
        &self,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        sweep: &SarBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<SarParams>), CudaSarError> {
        if len == 0 {
            return Err(CudaSarError::InvalidInput("empty inputs".into()));
        }
        if d_high.len() != len || d_low.len() != len {
            return Err(CudaSarError::InvalidInput(
                "device inputs must be length 'len'".into(),
            ));
        }
        if len - first_valid < 2 {
            return Err(CudaSarError::InvalidInput(
                "not enough valid data (need >= 2 after first)".into(),
            ));
        }

        let combos = expand_grid(sweep)?;

        let mut accs = Vec::with_capacity(combos.len());
        let mut maxs = Vec::with_capacity(combos.len());
        for p in &combos {
            let a = p.acceleration.unwrap_or(0.02);
            let m = p.maximum.unwrap_or(0.2);
            if !(a > 0.0) || !(m > 0.0) {
                return Err(CudaSarError::InvalidInput(
                    "invalid acceleration/maximum".into(),
                ));
            }
            accs.push(a as f32);
            maxs.push(m as f32);
        }

        
        let elem_size = std::mem::size_of::<f32>();
        let param_bytes = combos
            .len()
            .checked_mul(2)
            .and_then(|v| v.checked_mul(elem_size))
            .ok_or_else(|| CudaSarError::InvalidInput("size overflow".into()))?;
        let out_elems = combos
            .len()
            .checked_mul(len)
            .ok_or_else(|| CudaSarError::InvalidInput("size overflow".into()))?;
        let out_bytes = out_elems
            .checked_mul(elem_size)
            .ok_or_else(|| CudaSarError::InvalidInput("size overflow".into()))?;
        let required = param_bytes
            .checked_add(out_bytes)
            .ok_or_else(|| CudaSarError::InvalidInput("size overflow".into()))?;
        Self::will_fit(required, Self::headroom_bytes())?;

        let d_accs = DeviceBuffer::from_slice(&accs)?;
        let d_maxs = DeviceBuffer::from_slice(&maxs)?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(out_elems) }?;

        let len_i32 = len
            .try_into()
            .map_err(|_| CudaSarError::InvalidInput("len exceeds i32".into()))?;
        let first_i32 = first_valid
            .try_into()
            .map_err(|_| CudaSarError::InvalidInput("first_valid exceeds i32".into()))?;
        let rows_i32 = combos
            .len()
            .try_into()
            .map_err(|_| CudaSarError::InvalidInput("n_rows exceeds i32".into()))?;

        self.launch_batch_kernel(
            d_high,
            d_low,
            len_i32,
            first_i32,
            &d_accs,
            &d_maxs,
            rows_i32,
            &mut d_out,
        )?;

        self.stream.synchronize()?;

        Ok((DeviceArrayF32 { buf: d_out, rows: combos.len(), cols: len }, combos))
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_batch_kernel(
        &self,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        len: i32,
        first_valid: i32,
        d_accs: &DeviceBuffer<f32>,
        d_maxs: &DeviceBuffer<f32>,
        n_rows: i32,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSarError> {
        if len <= 0 || n_rows <= 0 {
            return Ok(());
        }
        let mut func = self
            .module
            .get_function("sar_batch_f32")
            .map_err(|_| CudaSarError::MissingKernelSymbol { name: "sar_batch_f32" })?;
        let _ = func.set_cache_config(CacheConfig::PreferL1);
        
        let block_x = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } if block_x > 0 => block_x,
            _ => {
                match func.suggested_launch_configuration(0, BlockSize::xyz(0, 0, 0)) {
                    Ok((_, bs)) => {
                        let x: u32 = bs.into();
                        x.max(32).min(1024)
                    }
                    Err(_) => 256,
                }
            }
        };
        let grid_x = ((n_rows as u32) + block_x - 1) / block_x;
        let grid_tuple = (grid_x.max(1), 1u32, 1u32);
        let block_tuple = (block_x, 1u32, 1u32);
        self.validate_launch(grid_tuple, block_tuple)?;
        let grid: GridSize = grid_tuple.into();
        let block: BlockSize = block_tuple.into();
        unsafe {
            let mut p_high = d_high.as_device_ptr().as_raw();
            let mut p_low = d_low.as_device_ptr().as_raw();
            let mut p_len = len;
            let mut p_first = first_valid;
            let mut p_accs = d_accs.as_device_ptr().as_raw();
            let mut p_maxs = d_maxs.as_device_ptr().as_raw();
            let mut p_n = n_rows;
            let mut p_out = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut p_high as *mut _ as *mut c_void,
                &mut p_low as *mut _ as *mut c_void,
                &mut p_len as *mut _ as *mut c_void,
                &mut p_first as *mut _ as *mut c_void,
                &mut p_accs as *mut _ as *mut c_void,
                &mut p_maxs as *mut _ as *mut c_void,
                &mut p_n as *mut _ as *mut c_void,
                &mut p_out as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)?;
        }
        Ok(())
    }

    
    pub fn sar_many_series_one_param_time_major_dev(
        &self,
        high_tm: &[f32],
        low_tm: &[f32],
        cols: usize,
        rows: usize,
        params: &SarParams,
    ) -> Result<DeviceArrayF32, CudaSarError> {
        if cols == 0 || rows == 0 {
            return Err(CudaSarError::InvalidInput("empty matrix".into()));
        }
        let elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaSarError::InvalidInput("size overflow".into()))?;
        if high_tm.len() != elems || low_tm.len() != elems {
            return Err(CudaSarError::InvalidInput(
                "inputs must be time‑major and equal size".into(),
            ));
        }
        let acceleration = params.acceleration.unwrap_or(0.02);
        let maximum = params.maximum.unwrap_or(0.2);
        if !(acceleration > 0.0) || !(maximum > 0.0) {
            return Err(CudaSarError::InvalidInput(
                "invalid acceleration/maximum".into(),
            ));
        }

        let first_valids = first_valids_time_major(high_tm, low_tm, cols, rows)?;

        
        let elem_f32 = std::mem::size_of::<f32>();
        let elem_i32 = std::mem::size_of::<i32>();
        let in_bytes = elems
            .checked_mul(2)
            .and_then(|v| v.checked_mul(elem_f32))
            .ok_or_else(|| CudaSarError::InvalidInput("size overflow".into()))?;
        let fv_bytes = cols
            .checked_mul(elem_i32)
            .ok_or_else(|| CudaSarError::InvalidInput("size overflow".into()))?;
        let out_bytes = elems
            .checked_mul(elem_f32)
            .ok_or_else(|| CudaSarError::InvalidInput("size overflow".into()))?;
        let required = in_bytes
            .checked_add(fv_bytes)
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| CudaSarError::InvalidInput("size overflow".into()))?;
        Self::will_fit(required, Self::headroom_bytes())?;

        
        let d_high = DeviceBuffer::from_slice(high_tm)?;
        let d_low = DeviceBuffer::from_slice(low_tm)?;
        let d_fv = DeviceBuffer::from_slice(&first_valids)?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems) }?;

        let cols_i32 = cols
            .try_into()
            .map_err(|_| CudaSarError::InvalidInput("cols exceeds i32".into()))?;
        let rows_i32 = rows
            .try_into()
            .map_err(|_| CudaSarError::InvalidInput("rows exceeds i32".into()))?;

        self.launch_many_series_kernel(
            &d_high,
            &d_low,
            &d_fv,
            cols_i32,
            rows_i32,
            acceleration as f32,
            maximum as f32,
            &mut d_out,
        )?;

        self.stream.synchronize()?;

        Ok(DeviceArrayF32 { buf: d_out, rows, cols })
    }

    /// Many-series × one-param with device-resident inputs to avoid re-copying.
    pub fn sar_many_series_one_param_time_major_dev_from_device(
        &self,
        d_high_tm: &DeviceBuffer<f32>,
        d_low_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        params: &SarParams,
    ) -> Result<DeviceArrayF32, CudaSarError> {
        if cols == 0 || rows == 0 {
            return Err(CudaSarError::InvalidInput("empty matrix".into()));
        }
        let elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaSarError::InvalidInput("size overflow".into()))?;
        if d_high_tm.len() != elems || d_low_tm.len() != elems || d_first_valids.len() != cols {
            return Err(CudaSarError::InvalidInput(
                "device inputs must match time-major sizes".into(),
            ));
        }

        let acceleration = params.acceleration.unwrap_or(0.02);
        let maximum = params.maximum.unwrap_or(0.2);
        if !(acceleration > 0.0) || !(maximum > 0.0) {
            return Err(CudaSarError::InvalidInput(
                "invalid acceleration/maximum".into(),
            ));
        }

        let elem_f32 = std::mem::size_of::<f32>();
        let required = elems
            .checked_mul(elem_f32)
            .ok_or_else(|| CudaSarError::InvalidInput("size overflow".into()))?;
        Self::will_fit(required, Self::headroom_bytes())?;

        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems) }?;

        let cols_i32 = cols
            .try_into()
            .map_err(|_| CudaSarError::InvalidInput("cols exceeds i32".into()))?;
        let rows_i32 = rows
            .try_into()
            .map_err(|_| CudaSarError::InvalidInput("rows exceeds i32".into()))?;

        self.launch_many_series_kernel(
            d_high_tm,
            d_low_tm,
            d_first_valids,
            cols_i32,
            rows_i32,
            acceleration as f32,
            maximum as f32,
            &mut d_out,
        )?;

        self.stream.synchronize()?;

        Ok(DeviceArrayF32 { buf: d_out, rows, cols })
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_many_series_kernel(
        &self,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_fv: &DeviceBuffer<i32>,
        cols: i32,
        rows: i32,
        acceleration: f32,
        maximum: f32,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSarError> {
        if cols <= 0 || rows <= 0 {
            return Ok(());
        }
        let mut func = self
            .module
            .get_function("sar_many_series_one_param_time_major_f32")
            .map_err(|_| CudaSarError::MissingKernelSymbol {
                name: "sar_many_series_one_param_time_major_f32",
            })?;
        let _ = func.set_cache_config(CacheConfig::PreferL1);
        let (block_x, block_y) = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x, block_y } if block_x > 0 && block_y > 0 => {
                (block_x, block_y)
            }
            _ => (128, 4),
        };
        let grid_y = ((cols as u32) + block_y - 1) / block_y;
        let grid_tuple = (1u32, grid_y.max(1), 1u32);
        let block_tuple = (block_x, block_y, 1u32);
        self.validate_launch(grid_tuple, block_tuple)?;
        let grid: GridSize = grid_tuple.into();
        let block: BlockSize = block_tuple.into();
        unsafe {
            let mut p_high = d_high.as_device_ptr().as_raw();
            let mut p_low = d_low.as_device_ptr().as_raw();
            let mut p_fv = d_fv.as_device_ptr().as_raw();
            let mut p_cols = cols;
            let mut p_rows = rows;
            let mut p_acc = acceleration;
            let mut p_max = maximum;
            let mut p_out = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut p_high as *mut _ as *mut c_void,
                &mut p_low as *mut _ as *mut c_void,
                &mut p_fv as *mut _ as *mut c_void,
                &mut p_cols as *mut _ as *mut c_void,
                &mut p_rows as *mut _ as *mut c_void,
                &mut p_acc as *mut _ as *mut c_void,
                &mut p_max as *mut _ as *mut c_void,
                &mut p_out as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)?;
        }
        Ok(())
    }
}


fn first_valid_hl(high: &[f32], low: &[f32]) -> Option<usize> {
    high.iter()
        .zip(low.iter())
        .position(|(&h, &l)| h.is_finite() && l.is_finite())
}

fn first_valids_time_major(
    high_tm: &[f32],
    low_tm: &[f32],
    cols: usize,
    rows: usize,
) -> Result<Vec<i32>, CudaSarError> {
    let mut out = vec![-1i32; cols];
    for s in 0..cols {
        for t in s..rows {
            
            let idx = t
                .checked_mul(cols)
                .and_then(|v| v.checked_add(s))
                .ok_or_else(|| {
                    CudaSarError::InvalidInput(
                        "size overflow in first_valids_time_major".into(),
                    )
                })?;
            let h = high_tm[idx];
            let l = low_tm[idx];
            if h.is_finite() && l.is_finite() {
                out[s] = t as i32;
                break;
            }
        }
    }
    Ok(out)
}

fn axis_f64(axis: (f64, f64, f64)) -> Result<Vec<f64>, CudaSarError> {
    let (start, end, step) = axis;
    if !step.is_finite() {
        return Err(CudaSarError::InvalidInput(format!(
            "invalid parameter range: start={start}, end={end}, step={step}"
        )));
    }
    if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
        return Ok(vec![start]);
    }

    let mut v = Vec::new();
    let tol = step.abs() * 1e-12;

    if step > 0.0 {
        if start <= end {
            let mut x = start;
            while x <= end + tol {
                v.push(x);
                x += step;
            }
        } else {
            
            let mut x = start;
            while x >= end - tol {
                v.push(x);
                x -= step;
            }
        }
    } else {
        
        if start >= end {
            let mut x = start;
            while x >= end - tol {
                v.push(x);
                x += step; 
            }
        } else {
            return Err(CudaSarError::InvalidInput(format!(
                "invalid parameter range (negative step with start < end): start={start}, end={end}, step={step}"
            )));
        }
    }

    if v.is_empty() {
        Err(CudaSarError::InvalidInput(format!(
            "parameter range produced no values: start={start}, end={end}, step={step}"
        )))
    } else {
        Ok(v)
    }
}

fn expand_grid(r: &SarBatchRange) -> Result<Vec<SarParams>, CudaSarError> {
    let accs = axis_f64(r.acceleration)?;
    let maxs = axis_f64(r.maximum)?;
    let capacity = accs
        .len()
        .checked_mul(maxs.len())
        .ok_or_else(|| CudaSarError::InvalidInput("parameter grid too large".into()))?;

    let mut out = Vec::with_capacity(capacity);
    for &a in &accs {
        for &m in &maxs {
            out.push(SarParams {
                acceleration: Some(a),
                maximum: Some(m),
            });
        }
    }
    Ok(out)
}


pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const MANY_SERIES_COLS: usize = 256;
    const MANY_SERIES_ROWS: usize = 1_000_000;

    fn bytes_one_series_many_params(n_rows: usize) -> usize {
        let in_bytes = 2 * ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let param_bytes = 2 * n_rows * std::mem::size_of::<f32>();
        let out_bytes = n_rows * ONE_SERIES_LEN * std::mem::size_of::<f32>();
        in_bytes + param_bytes + out_bytes + 64 * 1024 * 1024
    }
    fn bytes_many_series_one_param() -> usize {
        let elems = MANY_SERIES_COLS * MANY_SERIES_ROWS;
        let in_bytes = 2 * elems * std::mem::size_of::<f32>();
        let fv_bytes = MANY_SERIES_COLS * std::mem::size_of::<i32>();
        let out_bytes = elems * std::mem::size_of::<f32>();
        in_bytes + fv_bytes + out_bytes + 64 * 1024 * 1024
    }

    fn synth_high_low_from_price(price: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut high = price.to_vec();
        let mut low = price.to_vec();
        for i in 0..price.len() {
            let p = price[i];
            if !p.is_finite() {
                continue;
            }
            let x = i as f32 * 0.0021;
            let off = (0.0087 * x.cos()).abs() + 0.1;
            high[i] = p + off;
            low[i] = p - off;
        }
        (high, low)
    }

    struct SarBatchDeviceState {
        cuda: CudaSar,
        d_high: DeviceBuffer<f32>,
        d_low: DeviceBuffer<f32>,
        d_accs: DeviceBuffer<f32>,
        d_maxs: DeviceBuffer<f32>,
        d_out: DeviceBuffer<f32>,
        len_i32: i32,
        first_i32: i32,
        rows_i32: i32,
    }
    impl CudaBenchState for SarBatchDeviceState {
        fn launch(&mut self) {
            self.cuda
                .launch_batch_kernel(
                    &self.d_high,
                    &self.d_low,
                    self.len_i32,
                    self.first_i32,
                    &self.d_accs,
                    &self.d_maxs,
                    self.rows_i32,
                    &mut self.d_out,
                )
                .expect("sar batch");
            self.cuda.stream.synchronize().expect("sar batch sync");
        }
    }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaSar::new(0).expect("cuda sar");
        let price = gen_series(ONE_SERIES_LEN);
        let (high, low) = synth_high_low_from_price(&price);
        let sweep = SarBatchRange {
            acceleration: (0.01, 0.1, 0.01),
            maximum: (0.1, 0.3, 0.05),
        };
        let first = high
            .iter()
            .zip(low.iter())
            .position(|(&h, &l)| h.is_finite() && l.is_finite())
            .unwrap_or(0);
        let combos = expand_grid(&sweep).expect("expand_grid");
        let mut accs: Vec<f32> = Vec::with_capacity(combos.len());
        let mut maxs: Vec<f32> = Vec::with_capacity(combos.len());
        for p in &combos {
            accs.push(p.acceleration.unwrap_or(0.02) as f32);
            maxs.push(p.maximum.unwrap_or(0.2) as f32);
        }
        let len = ONE_SERIES_LEN;
        let rows = combos.len();
        let d_high = DeviceBuffer::from_slice(&high).expect("d_high");
        let d_low = DeviceBuffer::from_slice(&low).expect("d_low");
        let d_accs = DeviceBuffer::from_slice(&accs).expect("d_accs");
        let d_maxs = DeviceBuffer::from_slice(&maxs).expect("d_maxs");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(rows * len) }.expect("d_out");
        cuda.stream.synchronize().expect("sync after prep");
        Box::new(SarBatchDeviceState {
            cuda,
            d_high,
            d_low,
            d_accs,
            d_maxs,
            d_out,
            len_i32: len as i32,
            first_i32: first as i32,
            rows_i32: rows as i32,
        })
    }

    struct SarManyDeviceState {
        cuda: CudaSar,
        d_high_tm: DeviceBuffer<f32>,
        d_low_tm: DeviceBuffer<f32>,
        d_fv: DeviceBuffer<i32>,
        d_out: DeviceBuffer<f32>,
        cols: i32,
        rows: i32,
        acceleration: f32,
        maximum: f32,
    }
    impl CudaBenchState for SarManyDeviceState {
        fn launch(&mut self) {
            self.cuda
                .launch_many_series_kernel(
                    &self.d_high_tm,
                    &self.d_low_tm,
                    &self.d_fv,
                    self.cols,
                    self.rows,
                    self.acceleration,
                    self.maximum,
                    &mut self.d_out,
                )
                .expect("sar many-series");
            self.cuda.stream.synchronize().expect("sar many sync");
        }
    }
    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaSar::new(0).expect("cuda sar");
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_ROWS;
        
        let mut high_tm = vec![f32::NAN; cols * rows];
        let mut low_tm = vec![f32::NAN; cols * rows];
        for s in 0..cols {
            let price = gen_series(rows);
            let (h, l) = synth_high_low_from_price(&price);
            for t in s..rows {
                let idx = t * cols + s;
                high_tm[idx] = h[t];
                low_tm[idx] = l[t];
            }
        }
        let params = SarParams {
            acceleration: Some(0.02),
            maximum: Some(0.2),
        };
        let mut first_valids = vec![rows as i32; cols];
        for s in 0..cols {
            for t in 0..rows {
                let idx = t * cols + s;
                if high_tm[idx].is_finite() && low_tm[idx].is_finite() {
                    first_valids[s] = t as i32;
                    break;
                }
            }
        }
        let d_high_tm = DeviceBuffer::from_slice(&high_tm).expect("d_high_tm");
        let d_low_tm = DeviceBuffer::from_slice(&low_tm).expect("d_low_tm");
        let d_fv = DeviceBuffer::from_slice(&first_valids).expect("d_fv");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(cols * rows) }.expect("d_out");
        cuda.stream.synchronize().expect("sync after prep");
        Box::new(SarManyDeviceState {
            cuda,
            d_high_tm,
            d_low_tm,
            d_fv,
            d_out,
            cols: cols as i32,
            rows: rows as i32,
            acceleration: params.acceleration.unwrap_or(0.02) as f32,
            maximum: params.maximum.unwrap_or(0.2) as f32,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        let combos = (((0.1 - 0.01) / 0.01) as usize + 1) * (((0.3 - 0.1) / 0.05) as usize + 1);
        vec![
            CudaBenchScenario::new(
                "sar",
                "one_series_many_params",
                "sar_batch",
                "sar_batch/rowsxcols",
                prep_one_series_many_params,
            )
            .with_mem_required(bytes_one_series_many_params(combos)),
            CudaBenchScenario::new(
                "sar",
                "many_series_one_param",
                "sar_many_series",
                "sar_many/colsxrows",
                prep_many_series_one_param,
            )
            .with_mem_required(bytes_many_series_one_param()),
        ]
    }
}
