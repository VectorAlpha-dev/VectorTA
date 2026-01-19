#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::zlema::{expand_grid_zlema, ZlemaBatchRange, ZlemaParams};
use cust::context::{CacheConfig, Context};
use cust::device::Device;
use cust::device::DeviceAttribute as DevAttr;
use cust::function::{BlockSize, GridSize};
use cust::memory::{AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaZlemaError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("out of memory: required={required} free={free} headroom={headroom}")]
    OutOfMemory {
        required: usize,
        free: usize,
        headroom: usize,
    },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
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
    #[error("device mismatch: buffer device {buf}, current {current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
    NotImplemented,
}

pub struct CudaZlema {
    module: Module,
    stream: Stream,
    _context: Arc<Context>,
    device_id: u32,
}

impl CudaZlema {
    #[inline]
    fn env_flag(name: &str, default: bool) -> bool {
        match env::var(name) {
            Ok(v) => {
                let v = v.to_ascii_lowercase();
                matches!(v.as_str(), "1" | "true" | "yes" | "on")
            }
            Err(_) => default,
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
    fn device_mem_info() -> Option<(usize, usize)> {
        cust::memory::mem_get_info().ok()
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
    fn ensure_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaZlemaError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        if let Some((free, _total)) = Self::device_mem_info() {
            if required_bytes.saturating_add(headroom_bytes) > free {
                return Err(CudaZlemaError::OutOfMemory {
                    required: required_bytes,
                    free,
                    headroom: headroom_bytes,
                });
            }
        }
        Ok(())
    }
    #[inline]
    fn validate_launch(&self, grid: GridSize, block: BlockSize) -> Result<(), CudaZlemaError> {
        let dev = Device::get_device(self.device_id)?;
        let max_threads_per_block = dev.get_attribute(DevAttr::MaxThreadsPerBlock)? as u32;
        let bx = block.x * block.y * block.z;
        if bx > max_threads_per_block {
            return Err(CudaZlemaError::LaunchConfigTooLarge {
                gx: grid.x,
                gy: grid.y,
                gz: grid.z,
                bx: block.x,
                by: block.y,
                bz: block.z,
            });
        }

        let max_grid_x = dev.get_attribute(DevAttr::MaxGridDimX)? as u32;
        if grid.x > max_grid_x {
            return Err(CudaZlemaError::LaunchConfigTooLarge {
                gx: grid.x,
                gy: grid.y,
                gz: grid.z,
                bx: block.x,
                by: block.y,
                bz: block.z,
            });
        }
        Ok(())
    }
    pub fn new(device_id: usize) -> Result<Self, CudaZlemaError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/zlema_kernel.ptx"));

        let jit_opts = &[ModuleJitOption::DetermineTargetFromContext];
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
            _context: context,
            device_id: device_id as u32,
        })
    }

    #[inline]
    pub fn context_arc(&self) -> Arc<Context> {
        Arc::clone(&self._context)
    }

    #[inline]
    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    #[inline]
    pub fn stream_handle(&self) -> usize {
        self.stream.as_inner() as usize
    }

    #[inline]
    pub fn synchronize(&self) -> Result<(), CudaZlemaError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &ZlemaBatchRange,
    ) -> Result<(Vec<ZlemaParams>, usize, usize), CudaZlemaError> {
        if data_f32.is_empty() {
            return Err(CudaZlemaError::InvalidInput("empty data".into()));
        }

        let len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaZlemaError::InvalidInput("all values are NaN".into()))?;

        let combos = expand_grid_zlema(sweep);
        if combos.is_empty() {
            return Err(CudaZlemaError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        let mut max_period = 0usize;
        for combo in &combos {
            let period = combo.period.unwrap_or(0);
            if period == 0 {
                return Err(CudaZlemaError::InvalidInput(
                    "period must be at least 1 in CUDA path".into(),
                ));
            }
            if period > len {
                return Err(CudaZlemaError::InvalidInput(format!(
                    "period {} exceeds data length {}",
                    period, len
                )));
            }
            max_period = max_period.max(period);
        }

        if len - first_valid < max_period {
            return Err(CudaZlemaError::InvalidInput(format!(
                "not enough valid data (need >= {}, have {} after first valid)",
                max_period,
                len - first_valid
            )));
        }

        Ok((combos, first_valid, len))
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_lags: &DeviceBuffer<i32>,
        d_alphas: &DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaZlemaError> {
        let mut func = self.module.get_function("zlema_batch_f32").map_err(|_| {
            CudaZlemaError::MissingKernelSymbol {
                name: "zlema_batch_f32",
            }
        })?;

        if Self::env_flag("ZLEMA_PREFER_L1", true) {
            let _ = func.set_cache_config(CacheConfig::PreferL1);
        }

        let block_x_override = env::var("ZLEMA_BATCH_BLOCK_X")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .filter(|&v| v > 0);

        let (grid, block): (GridSize, BlockSize) = if let Some(bx) = block_x_override {
            let grid_x = ((n_combos as u32) + bx - 1) / bx;
            ((grid_x.max(1), 1, 1).into(), (bx, 1, 1).into())
        } else {
            let (min_grid, block_size) =
                func.suggested_launch_configuration(0, (0, 0, 0).into())?;
            let bx = block_size.clamp(64, 1024);
            let grid_x = ((n_combos as u32) + bx - 1) / bx;
            let gx = grid_x.max(min_grid);
            ((gx.max(1), 1, 1).into(), (bx, 1, 1).into())
        };

        self.validate_launch(grid, block)?;

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut lags_ptr = d_lags.as_device_ptr().as_raw();
            let mut alphas_ptr = d_alphas.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut combos_i = n_combos as i32;
            let mut first_valid_i = first_valid as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();

            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut lags_ptr as *mut _ as *mut c_void,
                &mut alphas_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];

            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn launch_batch_kernel_tiled(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_lags: &DeviceBuffer<i32>,
        d_alphas: &DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        n_combos: usize,
        max_lag: i32,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaZlemaError> {
        let mut func = self
            .module
            .get_function("zlema_batch_f32_tiled_f32")
            .map_err(|_| CudaZlemaError::MissingKernelSymbol {
                name: "zlema_batch_f32_tiled_f32",
            })?;

        if Self::env_flag("ZLEMA_PREFER_L1", true) {
            let _ = func.set_cache_config(CacheConfig::PreferL1);
        }

        let tile: usize = env::var("ZLEMA_BATCH_TILE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(1024);
        let shmem_bytes = (tile + (max_lag as usize)) * std::mem::size_of::<f32>();

        let block_x_override = env::var("ZLEMA_BATCH_BLOCK_X")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .filter(|&v| v > 0);
        let (grid, block): (GridSize, BlockSize) = if let Some(bx) = block_x_override {
            let grid_x = ((n_combos as u32) + bx - 1) / bx;
            ((grid_x.max(1), 1, 1).into(), (bx, 1, 1).into())
        } else {
            let (min_grid, block_size) =
                func.suggested_launch_configuration(shmem_bytes, (0, 0, 0).into())?;
            let bx = block_size.clamp(64, 1024);
            let grid_x = ((n_combos as u32) + bx - 1) / bx;
            let gx = grid_x.max(min_grid);
            ((gx.max(1), 1, 1).into(), (bx, 1, 1).into())
        };

        self.validate_launch(grid, block)?;

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut lags_ptr = d_lags.as_device_ptr().as_raw();
            let mut alphas_ptr = d_alphas.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut combos_i = n_combos as i32;
            let mut first_valid_i = first_valid as i32;
            let mut max_lag_i = max_lag as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();

            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut lags_ptr as *mut _ as *mut c_void,
                &mut alphas_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut max_lag_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];

            self.stream
                .launch(&func, grid, block, shmem_bytes as u32, args)?;
        }

        Ok(())
    }

    fn launch_batch_kernel_warp_scan(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        len: usize,
        first_valid: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaZlemaError> {
        let mut func = self
            .module
            .get_function("zlema_batch_warp_scan_f32")
            .map_err(|_| CudaZlemaError::MissingKernelSymbol {
                name: "zlema_batch_warp_scan_f32",
            })?;

        if Self::env_flag("ZLEMA_PREFER_L1", true) {
            let _ = func.set_cache_config(CacheConfig::PreferL1);
        }

        let block_x_override = env::var("ZLEMA_BATCH_BLOCK_X")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .filter(|&v| v > 0);

        let mut block_x: u32 = block_x_override.unwrap_or(128);
        if block_x < 32 {
            block_x = 32;
        }

        block_x = ((block_x + 31) / 32) * 32;
        if block_x > 1024 {
            block_x = 1024;
        }

        let warps_per_block = (block_x / 32).max(1);
        let grid_x = (((n_combos as u32) + warps_per_block - 1) / warps_per_block).max(1);
        let grid: GridSize = (grid_x, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        self.validate_launch(grid, block)?;

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut combos_i = n_combos as i32;
            let mut first_valid_i = first_valid as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();

            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
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
        combos: &[ZlemaParams],
        first_valid: usize,
        len: usize,
    ) -> Result<DeviceArrayF32, CudaZlemaError> {
        let rows = combos.len();
        let sz_f32 = std::mem::size_of::<f32>();
        let sz_i32 = std::mem::size_of::<i32>();
        let prices_b = len
            .checked_mul(sz_f32)
            .ok_or_else(|| CudaZlemaError::InvalidInput("byte size overflow".into()))?;
        let periods_b = rows
            .checked_mul(sz_i32)
            .ok_or_else(|| CudaZlemaError::InvalidInput("byte size overflow".into()))?;
        let out_b = rows
            .checked_mul(len)
            .and_then(|v| v.checked_mul(sz_f32))
            .ok_or_else(|| CudaZlemaError::InvalidInput("byte size overflow".into()))?;

        let n_combos = combos.len();
        let has_warp_scan = self
            .module
            .get_function("zlema_batch_warp_scan_f32")
            .is_ok();
        let use_warp_scan = has_warp_scan && Self::env_flag("ZLEMA_BATCH_WARP_SCAN", true);

        let lags_b = if use_warp_scan {
            0
        } else {
            rows.checked_mul(sz_i32)
                .ok_or_else(|| CudaZlemaError::InvalidInput("byte size overflow".into()))?
        };
        let alphas_b = if use_warp_scan {
            0
        } else {
            rows.checked_mul(sz_f32)
                .ok_or_else(|| CudaZlemaError::InvalidInput("byte size overflow".into()))?
        };

        let bytes_required = prices_b
            .checked_add(periods_b)
            .and_then(|v| v.checked_add(lags_b))
            .and_then(|v| v.checked_add(alphas_b))
            .and_then(|v| v.checked_add(out_b))
            .ok_or_else(|| CudaZlemaError::InvalidInput("byte size overflow".into()))?;
        let headroom = env::var("CUDA_MEM_HEADROOM")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(64 * 1024 * 1024);
        Self::ensure_fit(bytes_required, headroom)?;
        let d_prices = DeviceBuffer::from_slice(data_f32)?;

        let periods: Vec<i32> = combos.iter().map(|c| c.period.unwrap() as i32).collect();
        let d_periods = DeviceBuffer::from_slice(&periods)?;

        let elems = combos
            .len()
            .checked_mul(len)
            .ok_or_else(|| CudaZlemaError::InvalidInput("rows*cols overflow".into()))?;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }?;

        if use_warp_scan {
            self.launch_batch_kernel_warp_scan(
                &d_prices,
                &d_periods,
                len,
                first_valid,
                n_combos,
                &mut d_out,
            )?;
        } else {
            let lags: Vec<i32> = combos
                .iter()
                .map(|c| ((c.period.unwrap() - 1) / 2) as i32)
                .collect();
            let alphas: Vec<f32> = combos
                .iter()
                .map(|c| 2.0f32 / (c.period.unwrap() as f32 + 1.0f32))
                .collect();
            let d_lags = DeviceBuffer::from_slice(&lags)?;
            let d_alphas = DeviceBuffer::from_slice(&alphas)?;

            let max_lag = *lags.iter().max().unwrap_or(&0);
            let use_tiled = (n_combos >= 64) && (len >= 4096);

            if use_tiled {
                self.launch_batch_kernel_tiled(
                    &d_prices,
                    &d_periods,
                    &d_lags,
                    &d_alphas,
                    len,
                    first_valid,
                    n_combos,
                    max_lag,
                    &mut d_out,
                )?;
            } else {
                self.launch_batch_kernel(
                    &d_prices,
                    &d_periods,
                    &d_lags,
                    &d_alphas,
                    len,
                    first_valid,
                    n_combos,
                    &mut d_out,
                )?;
            }
        }

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: combos.len(),
            cols: len,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn zlema_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_lags: &DeviceBuffer<i32>,
        d_alphas: &DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        n_combos: usize,
        max_lag: i32,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaZlemaError> {
        if len == 0 || n_combos == 0 {
            return Err(CudaZlemaError::InvalidInput(
                "len and n_combos must be positive".into(),
            ));
        }
        if first_valid >= len {
            return Err(CudaZlemaError::InvalidInput(format!(
                "first_valid {} out of range for len {}",
                first_valid, len
            )));
        }
        if d_prices.len() != len {
            return Err(CudaZlemaError::InvalidInput(
                "device prices length mismatch".into(),
            ));
        }
        if d_periods.len() != n_combos {
            return Err(CudaZlemaError::InvalidInput(
                "device periods length mismatch".into(),
            ));
        }
        if d_out.len() != n_combos * len {
            return Err(CudaZlemaError::InvalidInput(
                "device output length mismatch".into(),
            ));
        }

        let has_warp_scan = self
            .module
            .get_function("zlema_batch_warp_scan_f32")
            .is_ok();
        let use_warp_scan = has_warp_scan && Self::env_flag("ZLEMA_BATCH_WARP_SCAN", true);

        if use_warp_scan {
            self.launch_batch_kernel_warp_scan(d_prices, d_periods, len, first_valid, n_combos, d_out)?;
            return Ok(());
        }

        if d_lags.len() != n_combos || d_alphas.len() != n_combos {
            return Err(CudaZlemaError::InvalidInput(
                "device lags/alphas length mismatch".into(),
            ));
        }

        let use_tiled = (n_combos >= 64) && (len >= 4096);
        if use_tiled {
            self.launch_batch_kernel_tiled(
                d_prices,
                d_periods,
                d_lags,
                d_alphas,
                len,
                first_valid,
                n_combos,
                max_lag,
                d_out,
            )?;
        } else {
            self.launch_batch_kernel(
                d_prices,
                d_periods,
                d_lags,
                d_alphas,
                len,
                first_valid,
                n_combos,
                d_out,
            )?;
        }

        Ok(())
    }

    pub fn zlema_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &ZlemaBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<ZlemaParams>), CudaZlemaError> {
        let (combos, first_valid, len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let dev = self.run_batch_kernel(data_f32, &combos, first_valid, len)?;
        Ok((dev, combos))
    }

    pub fn zlema_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &ZlemaBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<ZlemaParams>), CudaZlemaError> {
        let (combos, first_valid, len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let expected = combos.len() * len;
        if out.len() != expected {
            return Err(CudaZlemaError::InvalidInput(format!(
                "output slice length mismatch: expected {}, got {}",
                expected,
                out.len()
            )));
        }
        let dev = self.run_batch_kernel(data_f32, &combos, first_valid, len)?;

        let mut pinned =
            unsafe { LockedBuffer::<f32>::uninitialized(expected).map_err(CudaZlemaError::Cuda)? };
        unsafe {
            dev.buf
                .async_copy_to(&mut pinned.as_mut_slice(), &self.stream)
                .map_err(CudaZlemaError::Cuda)?;
        }
        self.stream.synchronize().map_err(CudaZlemaError::Cuda)?;
        out.copy_from_slice(pinned.as_slice());
        Ok((combos.len(), len, combos))
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &ZlemaParams,
    ) -> Result<(Vec<i32>, usize, f32), CudaZlemaError> {
        if cols == 0 || rows == 0 {
            return Err(CudaZlemaError::InvalidInput(
                "series dimensions must be positive".into(),
            ));
        }
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaZlemaError::InvalidInput("rows*cols overflow".into()))?;
        if data_tm_f32.len() != expected {
            return Err(CudaZlemaError::InvalidInput(format!(
                "data length mismatch: expected {}, got {}",
                expected,
                data_tm_f32.len()
            )));
        }

        let period = params.period.unwrap_or(0);
        if period == 0 {
            return Err(CudaZlemaError::InvalidInput(
                "period must be at least 1".into(),
            ));
        }
        if period > rows {
            return Err(CudaZlemaError::InvalidInput(format!(
                "period {} exceeds series length {}",
                period, rows
            )));
        }

        let mut first_valids = vec![0i32; cols];
        for series in 0..cols {
            let mut fv: Option<usize> = None;
            for row in 0..rows {
                let idx = row * cols + series;
                if !data_tm_f32[idx].is_nan() {
                    fv = Some(row);
                    break;
                }
            }
            let fvu = fv.ok_or_else(|| {
                CudaZlemaError::InvalidInput(format!("series {} all NaN", series))
            })?;
            if rows - fvu < period {
                return Err(CudaZlemaError::InvalidInput(format!(
                    "series {} insufficient data for period {} (tail = {})",
                    series,
                    period,
                    rows - fvu
                )));
            }
            first_valids[series] = fvu as i32;
        }

        let alpha = 2.0f32 / (period as f32 + 1.0f32);
        Ok((first_valids, period, alpha))
    }

    fn launch_many_series_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        period: usize,
        alpha: f32,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaZlemaError> {
        let mut func = self
            .module
            .get_function("zlema_many_series_one_param_f32")
            .map_err(|_| CudaZlemaError::MissingKernelSymbol {
                name: "zlema_many_series_one_param_f32",
            })?;

        if Self::env_flag("ZLEMA_PREFER_L1", true) {
            let _ = func.set_cache_config(CacheConfig::PreferL1);
        }

        let block_x_override = env::var("ZLEMA_MS_BLOCK_X")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .filter(|&v| v > 0);
        let (grid, block): (GridSize, BlockSize) = if let Some(bx) = block_x_override {
            let grid_x = ((cols as u32) + bx - 1) / bx;
            ((grid_x.max(1), 1, 1).into(), (bx, 1, 1).into())
        } else {
            let (min_grid, block_size) = func
                .suggested_launch_configuration(0, (0, 0, 0).into())
                .map_err(CudaZlemaError::Cuda)?;
            let bx = block_size.clamp(64, 1024);
            let grid_x = ((cols as u32) + bx - 1) / bx;
            let gx = grid_x.max(min_grid);
            ((gx.max(1), 1, 1).into(), (bx, 1, 1).into())
        };

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut num_series_i = cols as i32;
            let mut series_len_i = rows as i32;
            let mut period_i = period as i32;
            let mut alpha_f = alpha as f32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();

            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut alpha_f as *mut _ as *mut c_void,
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
        alpha: f32,
    ) -> Result<DeviceArrayF32, CudaZlemaError> {
        let elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaZlemaError::InvalidInput("rows*cols overflow".into()))?;
        let sz_f32 = std::mem::size_of::<f32>();
        let sz_i32 = std::mem::size_of::<i32>();
        let prices_b = elems
            .checked_mul(sz_f32)
            .ok_or_else(|| CudaZlemaError::InvalidInput("byte size overflow".into()))?;
        let first_b = cols
            .checked_mul(sz_i32)
            .ok_or_else(|| CudaZlemaError::InvalidInput("byte size overflow".into()))?;
        let out_b = elems
            .checked_mul(sz_f32)
            .ok_or_else(|| CudaZlemaError::InvalidInput("byte size overflow".into()))?;
        let bytes_required = prices_b
            .checked_add(first_b)
            .and_then(|v| v.checked_add(out_b))
            .ok_or_else(|| CudaZlemaError::InvalidInput("byte size overflow".into()))?;
        let headroom = env::var("CUDA_MEM_HEADROOM")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(64 * 1024 * 1024);
        if !Self::will_fit(bytes_required, headroom) {
            return Err(CudaZlemaError::InvalidInput(format!(
                "insufficient VRAM: need ~{} MB (incl headroom)",
                (bytes_required + headroom) / (1024 * 1024)
            )));
        }
        let d_prices = DeviceBuffer::from_slice(data_tm_f32)?;
        let d_first = DeviceBuffer::from_slice(first_valids)?;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }?;

        self.launch_many_series_kernel(&d_prices, &d_first, cols, rows, period, alpha, &mut d_out)?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    pub fn zlema_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &ZlemaParams,
    ) -> Result<DeviceArrayF32, CudaZlemaError> {
        let (first_valids, p, alpha) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        self.run_many_series_kernel(data_tm_f32, cols, rows, &first_valids, p, alpha)
    }

    pub fn zlema_many_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &ZlemaParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaZlemaError> {
        if out_tm.len() != cols * rows {
            return Err(CudaZlemaError::InvalidInput(format!(
                "output length mismatch: expected {}, got {}",
                cols * rows,
                out_tm.len()
            )));
        }
        let (first_valids, p, alpha) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        let dev = self.run_many_series_kernel(data_tm_f32, cols, rows, &first_valids, p, alpha)?;

        let mut pinned = unsafe { LockedBuffer::<f32>::uninitialized(cols * rows)? };
        unsafe {
            dev.buf
                .async_copy_to(&mut pinned.as_mut_slice(), &self.stream)?;
        }
        self.stream.synchronize()?;
        out_tm.copy_from_slice(pinned.as_slice());
        Ok(())
    }
}

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};
    use crate::indicators::moving_averages::zlema::ZlemaParams;

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250;
    const MANY_SERIES_COLS: usize = 250;
    const MANY_SERIES_LEN: usize = 1_000_000;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let param_bytes = PARAM_SWEEP * (std::mem::size_of::<i32>() + std::mem::size_of::<f32>());
        let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + param_bytes + out_bytes + 64 * 1024 * 1024
    }
    fn bytes_many_series_one_param() -> usize {
        let elems = MANY_SERIES_COLS * MANY_SERIES_LEN;
        let in_bytes = elems * std::mem::size_of::<f32>();
        let out_bytes = elems * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct ZlemaBatchDevState {
        cuda: CudaZlema,
        d_prices: DeviceBuffer<f32>,
        d_periods: DeviceBuffer<i32>,
        d_lags: Option<DeviceBuffer<i32>>,
        d_alphas: Option<DeviceBuffer<f32>>,
        len: usize,
        first_valid: usize,
        n_combos: usize,
        max_lag: i32,
        use_warp_scan: bool,
        d_out: DeviceBuffer<f32>,
    }
    impl CudaBenchState for ZlemaBatchDevState {
        fn launch(&mut self) {
            if self.use_warp_scan {
                self.cuda
                    .launch_batch_kernel_warp_scan(
                        &self.d_prices,
                        &self.d_periods,
                        self.len,
                        self.first_valid,
                        self.n_combos,
                        &mut self.d_out,
                    )
                    .expect("zlema batch warp-scan kernel");
            } else {
                let d_lags = self.d_lags.as_ref().expect("d_lags");
                let d_alphas = self.d_alphas.as_ref().expect("d_alphas");
                self.cuda
                    .launch_batch_kernel_tiled(
                        &self.d_prices,
                        &self.d_periods,
                        d_lags,
                        d_alphas,
                        self.len,
                        self.first_valid,
                        self.n_combos,
                        self.max_lag,
                        &mut self.d_out,
                    )
                    .expect("zlema batch tiled kernel");
            }
            self.cuda.stream.synchronize().expect("zlema sync");
        }
    }

    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaZlema::new(0).expect("cuda zlema");
        let price = gen_series(ONE_SERIES_LEN);
        let sweep = ZlemaBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1),
        };
        let (combos, first_valid, len) =
            CudaZlema::prepare_batch_inputs(&price, &sweep).expect("zlema prepare batch inputs");
        let n_combos = combos.len();

        let periods: Vec<i32> = combos.iter().map(|c| c.period.unwrap() as i32).collect();
        let d_prices = DeviceBuffer::from_slice(&price).expect("d_prices");
        let d_periods = DeviceBuffer::from_slice(&periods).expect("d_periods");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_combos * len) }.expect("d_out");

        let has_warp_scan = cuda
            .module
            .get_function("zlema_batch_warp_scan_f32")
            .is_ok();
        let use_warp_scan = has_warp_scan && CudaZlema::env_flag("ZLEMA_BATCH_WARP_SCAN", true);

        let (d_lags, d_alphas, max_lag) = if use_warp_scan {
            (None, None, 0)
        } else {
            let lags: Vec<i32> = combos
                .iter()
                .map(|c| ((c.period.unwrap() - 1) / 2) as i32)
                .collect();
            let alphas: Vec<f32> = combos
                .iter()
                .map(|c| 2.0f32 / (c.period.unwrap() as f32 + 1.0f32))
                .collect();
            let max_lag = *lags.iter().max().unwrap_or(&0);
            (
                Some(DeviceBuffer::from_slice(&lags).expect("d_lags")),
                Some(DeviceBuffer::from_slice(&alphas).expect("d_alphas")),
                max_lag,
            )
        };

        cuda.stream.synchronize().expect("sync after prep");

        Box::new(ZlemaBatchDevState {
            cuda,
            d_prices,
            d_periods,
            d_lags,
            d_alphas,
            len,
            first_valid,
            n_combos,
            max_lag,
            use_warp_scan,
            d_out,
        })
    }

    struct ZlemaManyDevState {
        cuda: CudaZlema,
        d_prices_tm: DeviceBuffer<f32>,
        d_first_valids: DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        period: usize,
        alpha: f32,
        d_out_tm: DeviceBuffer<f32>,
    }
    impl CudaBenchState for ZlemaManyDevState {
        fn launch(&mut self) {
            self.cuda
                .launch_many_series_kernel(
                    &self.d_prices_tm,
                    &self.d_first_valids,
                    self.cols,
                    self.rows,
                    self.period,
                    self.alpha,
                    &mut self.d_out_tm,
                )
                .expect("zlema many-series kernel");
            self.cuda.stream.synchronize().expect("zlema sync");
        }
    }

    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaZlema::new(0).expect("cuda zlema");
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_LEN;
        let data_tm = gen_time_major_prices(cols, rows);
        let params = ZlemaParams { period: Some(64) };
        let (first_valids, period, alpha) =
            CudaZlema::prepare_many_series_inputs(&data_tm, cols, rows, &params)
                .expect("zlema prepare many-series");

        let d_prices_tm = DeviceBuffer::from_slice(&data_tm).expect("d_prices_tm");
        let d_first_valids = DeviceBuffer::from_slice(&first_valids).expect("d_first_valids");
        let d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(cols * rows) }.expect("d_out_tm");
        cuda.stream.synchronize().expect("sync after prep");

        Box::new(ZlemaManyDevState {
            cuda,
            d_prices_tm,
            d_first_valids,
            cols,
            rows,
            period,
            alpha,
            d_out_tm,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "zlema",
                "one_series_many_params",
                "zlema_cuda_batch_dev",
                "1m_x_250",
                prep_one_series_many_params,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "zlema",
                "many_series_one_param",
                "zlema_cuda_many_series_one_param",
                "250x1m",
                prep_many_series_one_param,
            )
            .with_sample_size(5)
            .with_mem_required(bytes_many_series_one_param()),
        ]
    }
}
