//! CUDA scaffolding for the Simple Moving Average (SMA).
//!
//! Mirrors the VRAM-first integrations provided for ALMA/CWMA:
//! - Host-side validation expands parameter sweeps and checks warmup (NaN) semantics.
//! - Kernels execute in FP32 and return VRAM-resident `DeviceArrayF32` handles, with
//!   host-copy helpers layered on top for convenience.
//! - PTX is JITed with conservative options (DetermineTargetFromContext + O2) and
//!   falls back progressively for driver stability.
//! - Adds light VRAM checks to guard against accidental OOM for large sweeps; users
//!   can override headroom via `CUDA_MEM_HEADROOM`.

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::sma::{expand_grid_sma, SmaBatchRange, SmaParams};
use cust::context::{CacheConfig, Context};
use cust::device::{Device, DeviceAttribute};
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
pub enum CudaSmaError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("out of memory: required={required}B free={free}B headroom={headroom}B")]
    OutOfMemory { required: usize, free: usize, headroom: usize },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("device mismatch: buffer on {buf}, current {current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
    NotImplemented,
    #[error("invalid policy: {0}")]
    InvalidPolicy(&'static str),
}

pub struct CudaSma {
    module: Module,
    stream: Stream,
    _context: Arc<Context>,
    device_id: u32,
}

impl CudaSma {
    pub fn new(device_id: usize) -> Result<Self, CudaSmaError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/sma_kernel.ptx"));
        // Align with ALMA/CWMA: JIT target from context + OptLevel (default O2 here).
        let opt = match env::var("SMA_JIT_OPT").ok().as_deref() {
            Some("O0") => OptLevel::O0,
            Some("O1") => OptLevel::O1,
            Some("O2") => OptLevel::O2,
            Some("O3") => OptLevel::O3,
            Some("O4") => OptLevel::O4,
            _ => OptLevel::O2,
        };
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(opt),
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
        })
    }

    #[inline]
    pub fn synchronize(&self) -> Result<(), CudaSmaError> {
        self.stream.synchronize()?;
        Ok(())
    }

    #[inline]
    pub fn context_arc_clone(&self) -> Arc<Context> { self._context.clone() }

    #[inline]
    pub fn device_id(&self) -> u32 { self.device_id }

    #[inline]
    fn use_async_transfers() -> bool {
        // Default ON; set SMA_ASYNC=0 to disable.
        match env::var("SMA_ASYNC") {
            Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
            Err(_) => true,
        }
    }

    #[inline]
    fn parse_cache_pref() -> Option<CacheConfig> {
        match env::var("SMA_CACHE").ok().as_deref() {
            Some("prefer_l1") => Some(CacheConfig::PreferL1),
            Some("prefer_shared") => Some(CacheConfig::PreferShared),
            _ => Some(CacheConfig::PreferL1), // default to PreferL1 for read-mostly kernels
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
    fn device_mem_info() -> Option<(usize, usize)> { mem_get_info().ok() }

    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaSmaError> {
        if !Self::mem_check_enabled() { return Ok(()); }
        if let Some((free, _total)) = Self::device_mem_info() {
            if required_bytes.saturating_add(headroom_bytes) <= free {
                Ok(())
            } else {
                Err(CudaSmaError::OutOfMemory { required: required_bytes, free, headroom: headroom_bytes })
            }
        } else {
            Ok(())
        }
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &SmaBatchRange,
    ) -> Result<(Vec<SmaParams>, usize, usize), CudaSmaError> {
        if data_f32.is_empty() {
            return Err(CudaSmaError::InvalidInput("empty data".into()));
        }

        let len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaSmaError::InvalidInput("all values are NaN".into()))?;

        let combos = expand_grid_sma(sweep)
            .map_err(|e| CudaSmaError::InvalidInput(e.to_string()))?;
        if combos.is_empty() {
            return Err(CudaSmaError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        for combo in &combos {
            let period = combo.period.unwrap_or(0);
            if period == 0 {
                return Err(CudaSmaError::InvalidInput(
                    "period must be at least 1".into(),
                ));
            }
            if period > len {
                return Err(CudaSmaError::InvalidInput(format!(
                    "period {} exceeds data length {}",
                    period, len
                )));
            }
            if len - first_valid < period {
                return Err(CudaSmaError::InvalidInput(format!(
                    "not enough valid data for period {} (have {} after first valid)",
                    period,
                    len - first_valid
                )));
            }
        }

        Ok((combos, first_valid, len))
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSmaError> {
        let mut func: Function = self
            .module
            .get_function("sma_batch_f32")
            .map_err(|_| CudaSmaError::MissingKernelSymbol { name: "sma_batch_f32" })?;

        if let Some(cfg) = Self::parse_cache_pref() {
            let _ = func.set_cache_config(cfg);
        }

        // Default to CUDA-suggested block size; allow numeric override via SMA_BLOCK_X
        let block_x: u32 = match env::var("SMA_BLOCK_X").ok().as_deref() {
            Some(s) if s.eq_ignore_ascii_case("auto") => {
                let (_min_grid, suggested) = func
                    .suggested_launch_configuration(0, BlockSize::xyz(0, 0, 0))?;
                suggested
            }
            Some(s) => s.parse::<u32>().ok().filter(|&v| v > 0).unwrap_or(128),
            None => {
                let (_min_grid, suggested) = func
                    .suggested_launch_configuration(0, BlockSize::xyz(0, 0, 0))?;
                suggested
            }
        };
        let grid_x = ((n_combos as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        // Optional validation against device limits
        let dev = Device::get_device(self.device_id)?;
        let max_threads = dev.get_attribute(DeviceAttribute::MaxThreadsPerBlock)? as u32;
        if block_x > max_threads {
            return Err(CudaSmaError::LaunchConfigTooLarge { gx: grid_x, gy: 1, gz: 1, bx: block_x, by: 1, bz: 1 });
        }
        let max_grid_x = dev.get_attribute(DeviceAttribute::MaxGridDimX)? as u32;
        if grid_x > max_grid_x {
            return Err(CudaSmaError::LaunchConfigTooLarge { gx: grid_x, gy: 1, gz: 1, bx: block_x, by: 1, bz: 1 });
        }

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut series_len_i = series_len as i32;
            let mut combos_i = n_combos as i32;
            let mut first_valid_i = first_valid as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();

            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];

            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn run_batch_kernel(
        &self,
        data_f32: &[f32],
        combos: &[SmaParams],
        first_valid: usize,
        len: usize,
    ) -> Result<DeviceArrayF32, CudaSmaError> {
        // Optional VRAM check (rough estimate) like ALMA/CWMA wrappers
        let rows = combos.len();
        // Checked arithmetic: rows * cols and byte sizes
        let elems = rows
            .checked_mul(len)
            .ok_or_else(|| CudaSmaError::InvalidInput("rows*cols overflow".into()))?;
        let b_prices = len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaSmaError::InvalidInput("byte size overflow".into()))?;
        let b_periods = rows
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaSmaError::InvalidInput("byte size overflow".into()))?;
        let b_outputs = elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaSmaError::InvalidInput("byte size overflow".into()))?;
        let bytes_required = b_prices
            .checked_add(b_periods)
            .and_then(|x| x.checked_add(b_outputs))
            .ok_or_else(|| CudaSmaError::InvalidInput("byte size overflow".into()))?;
        let headroom = env::var("CUDA_MEM_HEADROOM")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(64 * 1024 * 1024);
        Self::will_fit(bytes_required, headroom)?;

        let periods: Vec<i32> = combos.iter().map(|c| c.period.unwrap() as i32).collect();

        if Self::use_async_transfers() {
            // Async path: pinned host buffers + stream-ordered alloc + async copies
            let h_prices = LockedBuffer::from_slice(data_f32)?;
            let h_periods = LockedBuffer::from_slice(&periods)?;

            let mut d_prices = unsafe { DeviceBuffer::<f32>::uninitialized_async(len, &self.stream) }?;
            let mut d_periods = unsafe { DeviceBuffer::<i32>::uninitialized_async(combos.len(), &self.stream) }?;
            let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized_async(elems, &self.stream) }?;

            unsafe {
                d_prices.async_copy_from(&h_prices, &self.stream)?;
                d_periods.async_copy_from(&h_periods, &self.stream)?;
            }

            self.launch_batch_kernel(
                &d_prices,
                &d_periods,
                len,
                combos.len(),
                first_valid,
                &mut d_out,
            )?;

            // Ensure DMA + kernel completed before returning device buffers
            self.stream.synchronize()?;

            Ok(DeviceArrayF32 {
                buf: d_out,
                rows: combos.len(),
                cols: len,
            })
        } else {
            // Fallback synchronous path
            let d_prices = DeviceBuffer::from_slice(data_f32)?;
            let d_periods = DeviceBuffer::from_slice(&periods)?;

            let elems = elems;
            let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }?;

            self.launch_batch_kernel(
                &d_prices,
                &d_periods,
                len,
                combos.len(),
                first_valid,
                &mut d_out,
            )?;

            Ok(DeviceArrayF32 {
                buf: d_out,
                rows: combos.len(),
                cols: len,
            })
        }
    }

    pub fn sma_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &SmaBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<SmaParams>), CudaSmaError> {
        let (combos, first_valid, len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let dev = self.run_batch_kernel(data_f32, &combos, first_valid, len)?;
        Ok((dev, combos))
    }

    pub fn sma_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &SmaBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<SmaParams>), CudaSmaError> {
        let (combos, first_valid, len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let expected = combos.len() * len;
        if out.len() != expected {
            return Err(CudaSmaError::InvalidInput(format!(
                "output length mismatch: expected {}, got {}",
                expected,
                out.len()
            )));
        }

        let dev = self.run_batch_kernel(data_f32, &combos, first_valid, len)?;
        dev.buf.copy_to(out)?;
        Ok((combos.len(), len, combos))
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &SmaParams,
    ) -> Result<(Vec<i32>, usize), CudaSmaError> {
        if cols == 0 || rows == 0 {
            return Err(CudaSmaError::InvalidInput(
                "series dimensions must be positive".into(),
            ));
        }
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaSmaError::InvalidInput("rows*cols overflow".into()))?;
        if data_tm_f32.len() != expected {
            return Err(CudaSmaError::InvalidInput(format!(
                "data length mismatch: expected {}, got {}",
                expected,
                data_tm_f32.len()
            )));
        }

        let period = params.period.unwrap_or(0);
        if period == 0 {
            return Err(CudaSmaError::InvalidInput(
                "period must be at least 1".into(),
            ));
        }
        if period > rows {
            return Err(CudaSmaError::InvalidInput(format!(
                "period {} exceeds series length {}",
                period, rows
            )));
        }

        let mut first_valids = vec![0i32; cols];
        for series in 0..cols {
            let mut fv = None;
            for row in 0..rows {
                let idx = row * cols + series;
                if !data_tm_f32[idx].is_nan() {
                    fv = Some(row);
                    break;
                }
            }
            let fv =
                fv.ok_or_else(|| CudaSmaError::InvalidInput(format!("series {} all NaN", series)))?;
            if rows - fv < period {
                return Err(CudaSmaError::InvalidInput(format!(
                    "series {} insufficient data for period {} (tail = {})",
                    series,
                    period,
                    rows - fv
                )));
            }
            first_valids[series] = fv as i32;
        }

        Ok((first_valids, period))
    }

    fn launch_many_series_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        period: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSmaError> {
        let mut func: Function = self
            .module
            .get_function("sma_many_series_one_param_f32")
            .map_err(|_| CudaSmaError::MissingKernelSymbol { name: "sma_many_series_one_param_f32" })?;

        if let Some(cfg) = Self::parse_cache_pref() {
            let _ = func.set_cache_config(cfg);
        }

        // Default to CUDA-suggested block size; allow numeric override via SMA_MS_BLOCK_X
        let block_x: u32 = match env::var("SMA_MS_BLOCK_X").ok().as_deref() {
            Some(s) if s.eq_ignore_ascii_case("auto") => {
                let (_min_grid, suggested) = func
                    .suggested_launch_configuration(0, BlockSize::xyz(0, 0, 0))?;
                suggested
            }
            Some(s) => s.parse::<u32>().ok().filter(|&v| v > 0).unwrap_or(128),
            None => {
                let (_min_grid, suggested) = func
                    .suggested_launch_configuration(0, BlockSize::xyz(0, 0, 0))?;
                suggested
            }
        };
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut num_series_i = cols as i32;
            let mut series_len_i = rows as i32;
            let mut period_i = period as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();

            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];

            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn run_many_series_kernel(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        first_valids: &[i32],
        period: usize,
    ) -> Result<DeviceArrayF32, CudaSmaError> {
        if Self::use_async_transfers() {
            // Async path: pinned I/O and stream-ordered allocs
            let h_prices = LockedBuffer::from_slice(data_tm_f32)?;
            let h_first = LockedBuffer::from_slice(first_valids)?;

            let mut d_prices = unsafe { DeviceBuffer::<f32>::uninitialized_async(cols * rows, &self.stream) }?;
            let mut d_first = unsafe { DeviceBuffer::<i32>::uninitialized_async(cols, &self.stream) }?;
            let elems = cols * rows;
            let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized_async(elems, &self.stream) }?;

            unsafe {
                d_prices.async_copy_from(&h_prices, &self.stream)?;
                d_first.async_copy_from(&h_first, &self.stream)?;
            }

            self.launch_many_series_kernel(&d_prices, &d_first, cols, rows, period, &mut d_out)?;

            self.stream.synchronize()?;

            Ok(DeviceArrayF32 {
                buf: d_out,
                rows,
                cols,
            })
        } else {
            // Synchronous fallback
            let d_prices = DeviceBuffer::from_slice(data_tm_f32)?;
            let d_first = DeviceBuffer::from_slice(first_valids)?;
            let elems = cols * rows;
            let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }?;

            self.launch_many_series_kernel(&d_prices, &d_first, cols, rows, period, &mut d_out)?;

            Ok(DeviceArrayF32 {
                buf: d_out,
                rows,
                cols,
            })
        }
    }

    pub fn sma_multi_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &SmaParams,
    ) -> Result<DeviceArrayF32, CudaSmaError> {
        let (first_valids, period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        self.run_many_series_kernel(data_tm_f32, cols, rows, &first_valids, period)
    }

    pub fn sma_multi_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &SmaParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaSmaError> {
        if out_tm.len() != cols * rows {
            return Err(CudaSmaError::InvalidInput(format!(
                "output length mismatch: expected {}, got {}",
                cols * rows,
                out_tm.len()
            )));
        }
        let (first_valids, period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        let dev = self.run_many_series_kernel(data_tm_f32, cols, rows, &first_valids, period)?;
        dev.buf
            .copy_to(out_tm)
            .map_err(CudaSmaError::Cuda)
    }

    // -------- Optional pinned output helpers (no extra host memcpy) --------
    /// Copy SMA batch output directly into a page-locked host buffer using async D2H.
    pub fn sma_batch_into_host_pinned_f32(
        &self,
        data_f32: &[f32],
        sweep: &SmaBatchRange,
        out_pinned: &mut LockedBuffer<f32>,
    ) -> Result<(usize, usize, Vec<SmaParams>), CudaSmaError> {
        let (combos, first_valid, len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let expected = combos.len() * len;
        if out_pinned.len() != expected {
            return Err(CudaSmaError::InvalidInput(format!(
                "output length mismatch: expected {}, got {}",
                expected,
                out_pinned.len()
            )));
        }
        let dev = self.run_batch_kernel(data_f32, &combos, first_valid, len)?;
        unsafe { dev.buf.async_copy_to(out_pinned, &self.stream)?; }
        self.stream.synchronize()?;
        Ok((combos.len(), len, combos))
    }

    /// Copy many-series, one-param output directly into a page-locked host buffer.
    pub fn sma_multi_series_one_param_time_major_into_host_pinned_f32(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &SmaParams,
        out_tm_pinned: &mut LockedBuffer<f32>,
    ) -> Result<(), CudaSmaError> {
        if out_tm_pinned.len() != cols * rows {
            return Err(CudaSmaError::InvalidInput(format!(
                "output length mismatch: expected {}, got {}",
                cols * rows,
                out_tm_pinned.len()
            )));
        }
        let (first_valids, period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        let dev = self.run_many_series_kernel(data_tm_f32, cols, rows, &first_valids, period)?;
        unsafe { dev.buf.async_copy_to(out_tm_pinned, &self.stream)?; }
        Ok(self.stream.synchronize()?)
    }

    // -------- Device-resident input variants to avoid re-uploading --------
    pub fn sma_batch_dev_from_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        len: usize,
        combos: &[SmaParams],
        first_valid: usize,
    ) -> Result<DeviceArrayF32, CudaSmaError> {
        let periods: Vec<i32> = combos.iter().map(|c| c.period.unwrap() as i32).collect();
        let d_periods = DeviceBuffer::from_slice(&periods)?;
        let elems = combos.len() * len;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }?;
        self.launch_batch_kernel(
            d_prices,
            &d_periods,
            len,
            combos.len(),
            first_valid,
            &mut d_out,
        )?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: combos.len(),
            cols: len,
        })
    }

    pub fn sma_multi_series_one_param_time_major_dev_from_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        period: usize,
    ) -> Result<DeviceArrayF32, CudaSmaError> {
        let elems = cols * rows;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }?;
        self.launch_many_series_kernel(
            d_prices_tm,
            d_first_valids,
            cols,
            rows,
            period,
            &mut d_out,
        )?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::define_ma_period_benches;

    define_ma_period_benches!(
        sma_benches,
        CudaSma,
        crate::indicators::moving_averages::sma::SmaBatchRange,
        crate::indicators::moving_averages::sma::SmaParams,
        sma_batch_dev,
        sma_multi_series_one_param_time_major_dev,
        crate::indicators::moving_averages::sma::SmaBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1)
        },
        crate::indicators::moving_averages::sma::SmaParams { period: Some(64) },
        "sma",
        "sma"
    );
    pub use sma_benches::bench_profiles;
}
