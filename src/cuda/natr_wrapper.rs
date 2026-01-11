//! CUDA support for the NATR (Normalized Average True Range) indicator.
//!
//! Parity goals (mirrors ALMA/CWMA wrappers):
//! - PTX load via DetermineTargetFromContext + OptLevel O2 with simple fallback
//! - NON_BLOCKING stream
//! - VRAM checks with optional headroom and grid chunking
//! - Public entry points:
//!     - one-series × many-params (batch)
//!     - many-series × one-param (time‑major)
//! - Numerics: warmup/NaN identical to scalar; warmup ATR by mean(TR) and
//!   sequential Wilder recurrence; outputs in f32.

#![cfg(feature = "cuda")]

use crate::indicators::natr::{NatrBatchRange, NatrParams};
use cust::context::Context;
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
pub enum CudaNatrError {
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
    #[error(
        "Launch configuration too large: grid=({gx},{gy},{gz}), block=({bx},{by},{bz})"
    )]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("Device mismatch: buffer on {buf}, current device {current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("Not implemented")]
    NotImplemented,
}

/// VRAM-backed array handle for NATR with context guard and device id
pub struct DeviceArrayF32Natr {
    pub buf: DeviceBuffer<f32>,
    pub rows: usize,
    pub cols: usize,
    pub ctx: Arc<Context>,
    pub device_id: u32,
}
impl DeviceArrayF32Natr {
    #[inline]
    pub fn device_ptr(&self) -> u64 {
        self.buf.as_device_ptr().as_raw() as u64
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
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
pub struct CudaNatrPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

pub struct CudaNatr {
    module: Module,
    stream: Stream,
    _context: Arc<Context>,
    device_id: u32,
    policy: CudaNatrPolicy,
    debug_logged: bool,
}

impl CudaNatr {
    pub fn new(device_id: usize) -> Result<Self, CudaNatrError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/natr_kernel.ptx"));
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
            policy: CudaNatrPolicy::default(),
            debug_logged: false,
        })
    }

    pub fn set_policy(&mut self, policy: CudaNatrPolicy) {
        self.policy = policy;
    }

    #[inline]
    pub fn context_arc(&self) -> Arc<Context> { Arc::clone(&self._context) }

    #[inline]
    pub fn device_id(&self) -> u32 { self.device_id }

    #[inline]
    fn headroom_bytes() -> usize {
        env::var("CUDA_MEM_HEADROOM")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
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
    fn will_fit(bytes: usize, headroom: usize) -> Result<(), CudaNatrError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        if let Ok((free, _)) = mem_get_info() {
            if bytes.saturating_add(headroom) > free {
                return Err(CudaNatrError::OutOfMemory {
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
    ) -> Result<(), CudaNatrError> {
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
            return Err(CudaNatrError::LaunchConfigTooLarge {
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

    
    fn first_valid_hlc(high: &[f32], low: &[f32], close: &[f32]) -> Option<usize> {
        let n = high.len().min(low.len()).min(close.len());
        for i in 0..n {
            
            if high[i].is_finite() && low[i].is_finite() && close[i].is_finite() {
                return Some(i);
            }
        }
        None
    }

    fn build_tr_one_series(
        high: &[f32],
        low: &[f32],
        close: &[f32],
    ) -> Result<(Vec<f32>, usize), CudaNatrError> {
        if high.len() != low.len() || high.len() != close.len() || high.is_empty() {
            return Err(CudaNatrError::InvalidInput(
                "mismatched or empty inputs".into(),
            ));
        }
        let len = high.len();
        let first = Self::first_valid_hlc(high, low, close)
            .ok_or_else(|| CudaNatrError::InvalidInput("all values are NaN".into()))?;
        let mut tr = vec![0f32; len];
        if first < len {
            tr[first] = high[first] - low[first];
            for i in (first + 1)..len {
                let h = high[i];
                let l = low[i];
                let pc = close[i - 1];
                let hl = h - l;
                let hc = (h - pc).abs();
                let lc = (l - pc).abs();
                tr[i] = hl.max(hc.max(lc));
            }
        }
        Ok((tr, first))
    }

    fn first_valids_time_major(
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        cols: usize,
        rows: usize,
    ) -> Result<Vec<i32>, CudaNatrError> {
        if cols == 0 || rows == 0 {
            return Err(CudaNatrError::InvalidInput("cols/rows zero".into()));
        }
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaNatrError::InvalidInput("rows*cols overflow".into()))?;
        if high_tm.len() != expected
            || low_tm.len() != expected
            || close_tm.len() != expected
        {
            return Err(CudaNatrError::InvalidInput(
                "time-major inputs wrong length".into(),
            ));
        }
        let mut fv = vec![0i32; cols];
        for s in 0..cols {
            let mut first: i32 = rows as i32; 
            for t in 0..rows {
                let idx = match t.checked_mul(cols).and_then(|v| v.checked_add(s)) {
                    Some(i) => i,
                    None => {
                        return Err(CudaNatrError::InvalidInput(
                            "index overflow in first_valids_time_major".into(),
                        ))
                    }
                };
                if high_tm[idx].is_finite() && low_tm[idx].is_finite() && close_tm[idx].is_finite()
                {
                    first = t as i32;
                    break;
                }
            }
            fv[s] = first;
        }
        Ok(fv)
    }

    
    pub fn natr_batch_dev(
        &mut self,
        high: &[f32],
        low: &[f32],
        close: &[f32],
        sweep: &NatrBatchRange,
    ) -> Result<DeviceArrayF32Natr, CudaNatrError> {
        let len = high.len();
        if len == 0 || low.len() != len || close.len() != len {
            return Err(CudaNatrError::InvalidInput(
                "mismatched or empty inputs".into(),
            ));
        }

        
        fn axis_usize((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, CudaNatrError> {
            if step == 0 || start == end {
                return Ok(vec![start]);
            }

            let mut values = Vec::new();
            let step_u = step;

            if start <= end {
                let mut v = start;
                loop {
                    if v > end {
                        break;
                    }
                    values.push(v);
                    match v.checked_add(step_u) {
                        Some(next) => v = next,
                        None => break,
                    }
                }
            } else {
                let mut v = start;
                loop {
                    if v < end {
                        break;
                    }
                    values.push(v);
                    match v.checked_sub(step_u) {
                        Some(next) => v = next,
                        None => break,
                    }
                }
            }

            if values.is_empty() {
                return Err(CudaNatrError::InvalidInput("empty period range".into()));
            }
            Ok(values)
        }
        let periods_v = axis_usize(sweep.period)?;
        let (tr, first_valid) = Self::build_tr_one_series(high, low, close)?;

        let max_p = *periods_v.iter().max().unwrap();
        if len - first_valid < max_p {
            return Err(CudaNatrError::InvalidInput(format!(
                "not enough valid data (needed >= {}, valid = {})",
                max_p,
                len - first_valid
            )));
        }
        let periods_i32: Vec<i32> = periods_v.iter().map(|&p| p as i32).collect();
        let rows = periods_v.len();

        
        let min_period = *periods_v.iter().min().unwrap();
        let warm_needed = first_valid + min_period - 1;
        let active_len = if len > warm_needed { len - warm_needed } else { 0 };
        let use_precompute = rows >= 4
            || rows.saturating_mul(active_len) >= 1_000_000;

        
        let elem_out = rows
            .checked_mul(len)
            .ok_or_else(|| CudaNatrError::InvalidInput("rows*len overflow".into()))?;
        let out_bytes = elem_out
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaNatrError::InvalidInput("output bytes overflow".into()))?;
        let tr_bytes = tr
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaNatrError::InvalidInput("tr bytes overflow".into()))?;
        let close_bytes = close
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaNatrError::InvalidInput("close bytes overflow".into()))?;
        let periods_bytes = periods_i32
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaNatrError::InvalidInput("periods bytes overflow".into()))?;
        let in_bytes = tr_bytes
            .checked_add(close_bytes)
            .and_then(|b| b.checked_add(periods_bytes))
            .ok_or_else(|| CudaNatrError::InvalidInput("input bytes overflow".into()))?;
        let head = Self::headroom_bytes();

        
        let base_bytes = out_bytes
            .checked_add(in_bytes)
            .ok_or_else(|| CudaNatrError::InvalidInput("VRAM size overflow".into()))?;
        Self::will_fit(base_bytes, head)?;

        
        let inv_bytes = len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaNatrError::InvalidInput("inv bytes overflow".into()))?;
        let allow_inv = if use_precompute {
            let total_with_inv = base_bytes
                .checked_add(inv_bytes)
                .ok_or_else(|| CudaNatrError::InvalidInput("VRAM size overflow".into()))?;
            Self::will_fit(total_with_inv, head).is_ok()
        } else {
            false
        };

        
        let d_tr: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::from_slice_async(&tr, &self.stream)?
        };
        let d_close: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::from_slice_async(close, &self.stream)?
        };
        let d_periods: DeviceBuffer<i32> = unsafe {
            DeviceBuffer::from_slice_async(&periods_i32, &self.stream)?
        };
        let mut d_out: DeviceBuffer<f32> =
            match unsafe { DeviceBuffer::uninitialized_async(elem_out, &self.stream) } {
                Ok(buf) => buf,
                Err(_) => unsafe { DeviceBuffer::uninitialized(elem_out)? },
            };

        
        let mut d_inv: Option<DeviceBuffer<f32>> = None;
        if allow_inv {
            let mut d = match unsafe { DeviceBuffer::<f32>::uninitialized_async(len, &self.stream) }
            {
                Ok(buf) => buf,
                Err(_) => unsafe { DeviceBuffer::<f32>::uninitialized(len)? },
            };
            let make_fn = self
                .module
                .get_function("natr_make_inv_close100")
                .map_err(|_| CudaNatrError::MissingKernelSymbol { name: "natr_make_inv_close100" })?;
            unsafe {
                let mut c_ptr = d_close.as_device_ptr().as_raw();
                let mut len_i = len as i32;
                let mut inv_ptr = d.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut c_ptr as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut inv_ptr as *mut _ as *mut c_void,
                ];
                let grid_x = ((len as u32) + 255) / 256;
                let grid: GridSize = (grid_x, 1, 1).into();
                let block: BlockSize = (256u32, 1, 1).into();
                self.validate_launch(grid_x, 1, 1, 256, 1, 1)?;
                self.stream.launch(&make_fn, grid, block, 0, args)?;
            }
            d_inv = Some(d);
        }

        let warp_io_enabled = std::env::var("NATR_BATCH_WARP_IO")
            .map(|v| v != "0")
            .unwrap_or(true);

        let block_x = match self.policy.batch {
            BatchKernelPolicy::Auto => {
                if warp_io_enabled {
                    32u32
                } else {
                    256u32
                }
            }
            BatchKernelPolicy::Plain { block_x } => block_x.max(32).min(1024),
        };

        let use_warp_io = warp_io_enabled && block_x == 32;

        
        let func = match (use_warp_io, d_inv.is_some()) {
            (true, true) => self
                .module
                .get_function("natr_batch_warp_io_f32_with_inv")
                .map_err(|_| CudaNatrError::MissingKernelSymbol { name: "natr_batch_warp_io_f32_with_inv" })?,
            (true, false) => self
                .module
                .get_function("natr_batch_warp_io_f32")
                .map_err(|_| CudaNatrError::MissingKernelSymbol { name: "natr_batch_warp_io_f32" })?,
            (false, true) => self
                .module
                .get_function("natr_batch_f32_with_inv")
                .map_err(|_| CudaNatrError::MissingKernelSymbol { name: "natr_batch_f32_with_inv" })?,
            (false, false) => self
                .module
                .get_function("natr_batch_f32")
                .map_err(|_| CudaNatrError::MissingKernelSymbol { name: "natr_batch_f32" })?,
        };
        let grid_x = rows as u32; 
        let grid: GridSize = (grid_x, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        self.validate_launch(grid_x, 1, 1, block_x, 1, 1)?;

        unsafe {
            if let Some(mut d_inv) = d_inv {
                let mut tr_ptr = d_tr.as_device_ptr().as_raw();
                let mut inv_ptr = d_inv.as_device_ptr().as_raw();
                let mut periods_ptr = d_periods.as_device_ptr().as_raw();
                let mut len_i = len as i32;
                let mut first_i = first_valid as i32;
                let mut rows_i = rows as i32;
                let mut out_ptr = d_out.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut tr_ptr as *mut _ as *mut c_void,
                    &mut inv_ptr as *mut _ as *mut c_void,
                    &mut periods_ptr as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut first_i as *mut _ as *mut c_void,
                    &mut rows_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream.launch(&func, grid, block, 0, args)?;
            } else {
                let mut tr_ptr = d_tr.as_device_ptr().as_raw();
                let mut close_ptr = d_close.as_device_ptr().as_raw();
                let mut periods_ptr = d_periods.as_device_ptr().as_raw();
                let mut len_i = len as i32;
                let mut first_i = first_valid as i32;
                let mut rows_i = rows as i32;
                let mut out_ptr = d_out.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut tr_ptr as *mut _ as *mut c_void,
                    &mut close_ptr as *mut _ as *mut c_void,
                    &mut periods_ptr as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut first_i as *mut _ as *mut c_void,
                    &mut rows_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream.launch(&func, grid, block, 0, args)?;
            }
        }

        self.stream
            .synchronize()
            ?;

        Ok(DeviceArrayF32Natr {
            buf: d_out,
            rows,
            cols: len,
            ctx: Arc::clone(&self._context),
            device_id: self.device_id,
        })
    }

    pub fn natr_many_series_one_param_time_major_dev(
        &mut self,
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
    ) -> Result<DeviceArrayF32Natr, CudaNatrError> {
        if cols == 0 || rows == 0 {
            return Err(CudaNatrError::InvalidInput("cols/rows zero".into()));
        }
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaNatrError::InvalidInput("rows*cols overflow".into()))?;
        if high_tm.len() != expected
            || low_tm.len() != expected
            || close_tm.len() != expected
        {
            return Err(CudaNatrError::InvalidInput(
                "time-major inputs wrong length".into(),
            ));
        }
        if period == 0 {
            return Err(CudaNatrError::InvalidInput("period must be > 0".into()));
        }

        let first_valids = Self::first_valids_time_major(high_tm, low_tm, close_tm, cols, rows)?;

        
        let elem_out = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaNatrError::InvalidInput("rows*cols overflow".into()))?;
        let out_bytes = elem_out
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaNatrError::InvalidInput("output bytes overflow".into()))?;
        let in_elems = high_tm
            .len()
            .checked_add(low_tm.len())
            .and_then(|v| v.checked_add(close_tm.len()))
            .ok_or_else(|| CudaNatrError::InvalidInput("input size overflow".into()))?;
        let in_bytes_main = in_elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaNatrError::InvalidInput("input bytes overflow".into()))?;
        let fv_bytes = first_valids
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaNatrError::InvalidInput("fv bytes overflow".into()))?;
        let in_bytes = in_bytes_main
            .checked_add(fv_bytes)
            .ok_or_else(|| CudaNatrError::InvalidInput("input bytes overflow".into()))?;
        let head = Self::headroom_bytes();
        let total = out_bytes
            .checked_add(in_bytes)
            .ok_or_else(|| CudaNatrError::InvalidInput("VRAM size overflow".into()))?;
        Self::will_fit(total, head)?;

        
        let d_high: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::from_slice_async(high_tm, &self.stream)?
        };
        let d_low: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::from_slice_async(low_tm, &self.stream)?
        };
        let d_close: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::from_slice_async(close_tm, &self.stream)?
        };
        let d_fv: DeviceBuffer<i32> = unsafe {
            DeviceBuffer::from_slice_async(&first_valids, &self.stream)?
        };
        let mut d_out: DeviceBuffer<f32> =
            match unsafe { DeviceBuffer::uninitialized_async(elem_out, &self.stream) } {
                Ok(buf) => buf,
                Err(_) => unsafe { DeviceBuffer::uninitialized(elem_out)? },
            };

        
        let func = self
            .module
            .get_function("natr_many_series_one_param_f32")
            .map_err(|_| CudaNatrError::MissingKernelSymbol { name: "natr_many_series_one_param_f32" })?;

        
        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => 128u32,
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(32).min(1024),
        };
        let warps_per_block = (block_x / 32).max(1);
        let grid_x = ((cols as u32) + warps_per_block - 1) / warps_per_block;
        let grid: GridSize = (grid_x, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        self.validate_launch(grid_x, 1, 1, block_x, 1, 1)?;

        unsafe {
            let mut h_ptr = d_high.as_device_ptr().as_raw();
            let mut l_ptr = d_low.as_device_ptr().as_raw();
            let mut c_ptr = d_close.as_device_ptr().as_raw();
            let mut per_i = period as i32;
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut fv_ptr = d_fv.as_device_ptr().as_raw();
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut h_ptr as *mut _ as *mut c_void,
                &mut l_ptr as *mut _ as *mut c_void,
                &mut c_ptr as *mut _ as *mut c_void,
                &mut per_i as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut fv_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        self.stream.synchronize()?;

        Ok(DeviceArrayF32Natr {
            buf: d_out,
            rows,
            cols,
            ctx: Arc::clone(&self._context),
            device_id: self.device_id,
        })
    }
}


pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;

    fn bytes_one_series(n_combos: usize) -> usize {
        
        let in_bytes = 3 * ONE_SERIES_LEN * std::mem::size_of::<f32>()
            + ONE_SERIES_LEN * std::mem::size_of::<f32>()
            + n_combos * std::mem::size_of::<i32>();
        let out_bytes = n_combos * ONE_SERIES_LEN * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    fn synth_hlc_from_close(close: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut high = close.to_vec();
        let mut low = close.to_vec();
        for i in 0..close.len() {
            let v = close[i];
            if !v.is_finite() {
                continue;
            }
            let x = i as f32 * 0.0031;
            let off = (0.002 * x.sin()).abs() + 0.5;
            high[i] = v + off;
            low[i] = v - off;
        }
        (high, low)
    }

    struct NatrBatchState {
        cuda: CudaNatr,
        d_tr: DeviceBuffer<f32>,
        d_close: DeviceBuffer<f32>,
        d_periods: DeviceBuffer<i32>,
        len: usize,
        first_valid: usize,
        rows: usize,
        block_x: u32,
        d_out: DeviceBuffer<f32>,
    }
    impl CudaBenchState for NatrBatchState {
        fn launch(&mut self) {
            let func = self
                .cuda
                .module
                .get_function("natr_batch_warp_io_f32")
                .or_else(|_| self.cuda.module.get_function("natr_batch_f32"))
                .expect("natr batch kernel");
            let grid: GridSize = (self.rows as u32, 1, 1).into();
            let block: BlockSize = (self.block_x, 1, 1).into();
            unsafe {
                let mut tr_ptr = self.d_tr.as_device_ptr().as_raw();
                let mut close_ptr = self.d_close.as_device_ptr().as_raw();
                let mut periods_ptr = self.d_periods.as_device_ptr().as_raw();
                let mut len_i = self.len as i32;
                let mut first_i = self.first_valid as i32;
                let mut rows_i = self.rows as i32;
                let mut out_ptr = self.d_out.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut tr_ptr as *mut _ as *mut c_void,
                    &mut close_ptr as *mut _ as *mut c_void,
                    &mut periods_ptr as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut first_i as *mut _ as *mut c_void,
                    &mut rows_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.cuda
                    .stream
                    .launch(&func, grid, block, 0, args)
                    .expect("natr launch");
            }
            self.cuda.stream.synchronize().expect("natr sync");
        }
    }

    fn prep_one_series() -> Box<dyn CudaBenchState> {
        let cuda = CudaNatr::new(0).expect("cuda natr");
        let close = gen_series(ONE_SERIES_LEN);
        let (high, low) = synth_hlc_from_close(&close);
        let sweep = NatrBatchRange { period: (7, 64, 3) };
        let periods: Vec<usize> = (sweep.period.0..=sweep.period.1)
            .step_by(sweep.period.2.max(1))
            .collect();
        let rows = periods.len();
        let periods_i32: Vec<i32> = periods.iter().map(|&p| p as i32).collect();

        let (tr, first_valid) = CudaNatr::build_tr_one_series(&high, &low, &close).expect("tr");
        let d_tr = unsafe { DeviceBuffer::from_slice_async(&tr, &cuda.stream) }.expect("d_tr");
        let d_close = unsafe { DeviceBuffer::from_slice_async(&close, &cuda.stream) }.expect("d_close");
        let d_periods =
            unsafe { DeviceBuffer::from_slice_async(&periods_i32, &cuda.stream) }.expect("d_periods");
        let out_elems = rows * ONE_SERIES_LEN;
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(out_elems, &cuda.stream) }.expect("d_out");

        cuda.stream.synchronize().expect("sync after prep");
        Box::new(NatrBatchState {
            cuda,
            d_tr,
            d_close,
            d_periods,
            len: ONE_SERIES_LEN,
            first_valid,
            rows,
            block_x: 32,
            d_out,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "natr",
            "one_series_many_params",
            "natr_cuda_batch",
            "1m",
            prep_one_series,
        )
        .with_sample_size(10)
        .with_mem_required(bytes_one_series(20))]
    }
}
