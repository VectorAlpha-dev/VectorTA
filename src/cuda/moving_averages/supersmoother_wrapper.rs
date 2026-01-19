#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::supersmoother::{
    expand_grid_supersmoother, SuperSmootherBatchRange, SuperSmootherParams,
};
use cust::context::{CacheConfig, Context};
use cust::device::Device;
use cust::function::{BlockSize, Function, GridSize};
use cust::memory::{
    mem_get_info, AsyncCopyDestination, CopyDestination, DeviceBuffer, LockedBuffer,
};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::mem::size_of;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaSuperSmootherError {
    #[error("CUDA error: {0}")]
    Cuda(#[from] cust::error::CudaError),
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("CUDA supersmoother not implemented")]
    NotImplemented,
    #[error("out of device memory: required={required} free={free} headroom={headroom}")]
    OutOfMemory {
        required: usize,
        free: usize,
        headroom: usize,
    },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("output slice length mismatch: expected={expected}, got={got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("launch config too large: grid=({gx}, {gy}, {gz}), block=({bx}, {by}, {bz})")]
    LaunchConfigTooLarge {
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
    },
    #[error("device mismatch: buffer on device {buf}, current device {current}")]
    DeviceMismatch { buf: u32, current: u32 },
}

pub struct CudaSuperSmoother {
    module: Module,
    stream: Stream,
    ctx: Arc<Context>,
    device_id: u32,
}

impl CudaSuperSmoother {
    pub fn new(device_id: usize) -> Result<Self, CudaSuperSmootherError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Context::new(device)?;
        let ctx = Arc::new(context);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/supersmoother_kernel.ptx"));

        let mut jit_opts: Vec<ModuleJitOption> = vec![
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O4),
        ];
        if let Ok(v) = std::env::var("SS_MAXREG") {
            if let Ok(cap) = v.parse::<u32>() {
                jit_opts.push(ModuleJitOption::MaxRegisters(cap));
            }
        }

        let module = match Module::from_ptx(ptx, &jit_opts) {
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
            ctx,
            device_id: device_id as u32,
        })
    }

    #[inline]
    pub fn device_id(&self) -> u32 {
        self.device_id
    }
    #[inline]
    pub fn context_arc(&self) -> Arc<Context> {
        self.ctx.clone()
    }

    #[inline]
    pub fn synchronize(&self) -> Result<(), CudaSuperSmootherError> {
        self.stream.synchronize()?;
        Ok(())
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

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &SuperSmootherBatchRange,
    ) -> Result<(Vec<SuperSmootherParams>, usize, usize), CudaSuperSmootherError> {
        if data_f32.is_empty() {
            return Err(CudaSuperSmootherError::InvalidInput("empty data".into()));
        }
        let first_valid = data_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaSuperSmootherError::InvalidInput("all values are NaN".into()))?;

        let combos = expand_grid_supersmoother(sweep);
        if combos.is_empty() {
            return Err(CudaSuperSmootherError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        let len = data_f32.len();
        let tail_len = len - first_valid;
        for combo in &combos {
            let period = combo.period.unwrap_or(0);
            if period == 0 {
                return Err(CudaSuperSmootherError::InvalidInput(
                    "period must be at least 1".into(),
                ));
            }
            if period > len {
                return Err(CudaSuperSmootherError::InvalidInput(format!(
                    "period {} exceeds data length {}",
                    period, len
                )));
            }
            if tail_len < period {
                return Err(CudaSuperSmootherError::InvalidInput(format!(
                    "not enough valid data for period {} (tail = {})",
                    period, tail_len
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
    ) -> Result<(), CudaSuperSmootherError> {
        if let Ok(mut func) = self
            .module
            .get_function("supersmoother_batch_warp_scan_f32")
        {
            let _ = func.set_cache_config(CacheConfig::PreferL1);

            const MAX_GRID_X: usize = 65_535;
            let block: BlockSize = (32u32, 1, 1).into();

            let mut launched = 0usize;
            while launched < n_combos {
                let rows = (n_combos - launched).min(MAX_GRID_X);
                let grid: GridSize = (rows as u32, 1, 1).into();

                unsafe {
                    let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                    let mut periods_ptr = d_periods.as_device_ptr().add(launched).as_raw();
                    let mut series_len_i = series_len as i32;
                    let mut combos_i = rows as i32;
                    let mut first_valid_i = first_valid as i32;
                    let mut out_ptr = d_out.as_device_ptr().add(launched * series_len).as_raw();

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

                launched += rows;
            }

            return Ok(());
        }

        let mut func: Function = self
            .module
            .get_function("supersmoother_batch_f32")
            .map_err(|_| CudaSuperSmootherError::MissingKernelSymbol {
                name: "supersmoother_batch_f32",
            })?;

        let _ = func.set_cache_config(CacheConfig::PreferL1);

        let (_min_grid, block_x) = func
            .suggested_launch_configuration(0, BlockSize::xyz(0, 0, 0))
            .unwrap_or((0, 256));
        let block_x: u32 = block_x.max(64);
        let grid_x = ((n_combos as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        let dev = Device::get_device(self.device_id)?;
        let max_threads =
            dev.get_attribute(cust::device::DeviceAttribute::MaxThreadsPerBlock)? as u32;
        let max_grid_x = dev.get_attribute(cust::device::DeviceAttribute::MaxGridDimX)? as u32;
        if block_x > max_threads || grid_x > max_grid_x {
            return Err(CudaSuperSmootherError::LaunchConfigTooLarge {
                gx: grid_x,
                gy: 1,
                gz: 1,
                bx: block_x,
                by: 1,
                bz: 1,
            });
        }

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut len_i = series_len as i32;
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
        combos: &[SuperSmootherParams],
        first_valid: usize,
        len: usize,
    ) -> Result<DeviceArrayF32, CudaSuperSmootherError> {
        let d_prices = DeviceBuffer::from_slice(data_f32)?;
        let periods: Vec<i32> = combos.iter().map(|c| c.period.unwrap() as i32).collect();
        let d_periods = DeviceBuffer::from_slice(&periods)?;

        let elems = combos.len().checked_mul(len).ok_or_else(|| {
            CudaSuperSmootherError::InvalidInput("size overflow (rows*cols)".into())
        })?;
        let needed = elems
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| CudaSuperSmootherError::InvalidInput("size overflow (bytes)".into()))?;
        if !Self::will_fit(needed, 4 * 1024 * 1024) {
            let (free, _) = mem_get_info().unwrap_or((0, 0));
            return Err(CudaSuperSmootherError::OutOfMemory {
                required: needed,
                free,
                headroom: 4 * 1024 * 1024,
            });
        }

        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized_async(elems, &self.stream) }?;

        self.launch_batch_kernel(
            &d_prices,
            &d_periods,
            len,
            combos.len(),
            first_valid,
            &mut d_out,
        )?;

        self.stream.synchronize()?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: combos.len(),
            cols: len,
        })
    }

    pub fn supersmoother_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &SuperSmootherBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<SuperSmootherParams>), CudaSuperSmootherError> {
        let (combos, first_valid, len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let dev = self.run_batch_kernel(data_f32, &combos, first_valid, len)?;
        Ok((dev, combos))
    }

    pub fn supersmoother_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &SuperSmootherBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<SuperSmootherParams>), CudaSuperSmootherError> {
        let (combos, first_valid, len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let expected = combos.len().checked_mul(len).ok_or_else(|| {
            CudaSuperSmootherError::InvalidInput("size overflow (rows*cols)".into())
        })?;
        if out.len() != expected {
            return Err(CudaSuperSmootherError::OutputLengthMismatch {
                expected,
                got: out.len(),
            });
        }
        let dev = self.run_batch_kernel(data_f32, &combos, first_valid, len)?;
        dev.buf.copy_to(out)?;
        Ok((combos.len(), len, combos))
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &SuperSmootherParams,
    ) -> Result<(Vec<i32>, usize), CudaSuperSmootherError> {
        if cols == 0 || rows == 0 {
            return Err(CudaSuperSmootherError::InvalidInput(
                "series dimensions must be positive".into(),
            ));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaSuperSmootherError::InvalidInput(format!(
                "data length mismatch: expected {}, got {}",
                cols * rows,
                data_tm_f32.len()
            )));
        }
        let period = params.period.unwrap_or(0);
        if period == 0 {
            return Err(CudaSuperSmootherError::InvalidInput(
                "period must be at least 1".into(),
            ));
        }
        if period > rows {
            return Err(CudaSuperSmootherError::InvalidInput(format!(
                "period {} exceeds series length {}",
                period, rows
            )));
        }

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
                CudaSuperSmootherError::InvalidInput(format!(
                    "series {} contains only NaNs",
                    series
                ))
            })?;
            let tail = rows - fv;
            if tail < period {
                return Err(CudaSuperSmootherError::InvalidInput(format!(
                    "series {} insufficient data for period {} (tail = {})",
                    series, period, tail
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
    ) -> Result<(), CudaSuperSmootherError> {
        let mut func: Function = self
            .module
            .get_function("supersmoother_many_series_one_param_f32")
            .map_err(|_| CudaSuperSmootherError::MissingKernelSymbol {
                name: "supersmoother_many_series_one_param_f32",
            })?;
        let _ = func.set_cache_config(CacheConfig::PreferL1);

        let (_min_grid, block_x) = func
            .suggested_launch_configuration(0, BlockSize::xyz(0, 0, 0))
            .unwrap_or((0, 256));
        let block_x: u32 = block_x.max(64);
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        let dev = Device::get_device(self.device_id).map_err(CudaSuperSmootherError::Cuda)?;
        let max_threads = dev
            .get_attribute(cust::device::DeviceAttribute::MaxThreadsPerBlock)
            .map_err(CudaSuperSmootherError::Cuda)? as u32;
        let max_grid_x = dev
            .get_attribute(cust::device::DeviceAttribute::MaxGridDimX)
            .map_err(CudaSuperSmootherError::Cuda)? as u32;
        if block_x > max_threads || grid_x > max_grid_x {
            return Err(CudaSuperSmootherError::LaunchConfigTooLarge {
                gx: grid_x,
                gy: 1,
                gz: 1,
                bx: block_x,
                by: 1,
                bz: 1,
            });
        }

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

            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(CudaSuperSmootherError::Cuda)?;
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
    ) -> Result<DeviceArrayF32, CudaSuperSmootherError> {
        let d_prices = DeviceBuffer::from_slice(data_tm_f32)?;
        let d_first_valids = DeviceBuffer::from_slice(first_valids)?;
        let elems = cols.checked_mul(rows).ok_or_else(|| {
            CudaSuperSmootherError::InvalidInput("size overflow (rows*cols)".into())
        })?;
        let needed = elems
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| CudaSuperSmootherError::InvalidInput("size overflow (bytes)".into()))?;
        if !Self::will_fit(needed, 4 * 1024 * 1024) {
            let (free, _) = mem_get_info().unwrap_or((0, 0));
            return Err(CudaSuperSmootherError::OutOfMemory {
                required: needed,
                free,
                headroom: 4 * 1024 * 1024,
            });
        }
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized_async(elems, &self.stream) }?;

        self.launch_many_series_kernel(&d_prices, &d_first_valids, cols, rows, period, &mut d_out)?;

        self.stream.synchronize()?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    pub fn supersmoother_multi_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &SuperSmootherParams,
    ) -> Result<DeviceArrayF32, CudaSuperSmootherError> {
        let (first_valids, period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        self.run_many_series_kernel(data_tm_f32, cols, rows, &first_valids, period)
    }

    pub fn supersmoother_multi_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &SuperSmootherParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaSuperSmootherError> {
        if out_tm.len() != cols * rows {
            return Err(CudaSuperSmootherError::InvalidInput(format!(
                "output slice mismatch: expected {}, got {}",
                cols * rows,
                out_tm.len()
            )));
        }
        let (first_valids, period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        let dev = self.run_many_series_kernel(data_tm_f32, cols, rows, &first_valids, period)?;
        Ok(dev.buf.copy_to(out_tm)?)
    }

    pub fn supersmoother_batch_from_device_prices(
        &self,
        d_prices: &DeviceBuffer<f32>,
        first_valid: usize,
        len: usize,
        combos: &[SuperSmootherParams],
    ) -> Result<DeviceArrayF32, CudaSuperSmootherError> {
        if d_prices.len() != len {
            return Err(CudaSuperSmootherError::InvalidInput(format!(
                "device prices length mismatch: expected {}, got {}",
                len,
                d_prices.len()
            )));
        }
        if combos.is_empty() {
            return Err(CudaSuperSmootherError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        let periods: Vec<i32> = combos.iter().map(|c| c.period.unwrap() as i32).collect();
        let d_periods = DeviceBuffer::from_slice(&periods)?;
        let elems = combos.len() * len;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized_async(elems, &self.stream) }?;
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

    pub fn supersmoother_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSuperSmootherError> {
        if series_len == 0 || n_combos == 0 {
            return Err(CudaSuperSmootherError::InvalidInput(
                "series_len and n_combos must be positive".into(),
            ));
        }
        if d_prices.len() != series_len {
            return Err(CudaSuperSmootherError::InvalidInput(
                "prices buffer length mismatch".into(),
            ));
        }
        if d_periods.len() != n_combos {
            return Err(CudaSuperSmootherError::InvalidInput(
                "periods buffer length mismatch".into(),
            ));
        }
        if d_out.len() != n_combos * series_len {
            return Err(CudaSuperSmootherError::InvalidInput(
                "output buffer length mismatch".into(),
            ));
        }

        self.launch_batch_kernel(d_prices, d_periods, series_len, n_combos, first_valid, d_out)
    }

    pub fn supersmoother_batch_into_pinned_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &SuperSmootherBatchRange,
        out_pinned: &mut LockedBuffer<f32>,
    ) -> Result<(usize, usize, Vec<SuperSmootherParams>), CudaSuperSmootherError> {
        let (combos, first_valid, len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let expected = combos.len() * len;
        if out_pinned.len() != expected {
            return Err(CudaSuperSmootherError::InvalidInput(format!(
                "output pinned buffer length mismatch: expected {}, got {}",
                expected,
                out_pinned.len()
            )));
        }
        let dev = self.run_batch_kernel(data_f32, &combos, first_valid, len)?;
        unsafe {
            dev.buf
                .async_copy_to(out_pinned.as_mut_slice(), &self.stream)
                .map_err(CudaSuperSmootherError::Cuda)?;
        }
        self.stream
            .synchronize()
            .map_err(CudaSuperSmootherError::Cuda)?;
        Ok((combos.len(), len, combos))
    }

    pub fn supersmoother_multi_series_one_param_time_major_into_host_pinned_f32(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &SuperSmootherParams,
        out_tm_pinned: &mut LockedBuffer<f32>,
    ) -> Result<(), CudaSuperSmootherError> {
        if out_tm_pinned.len() != cols * rows {
            return Err(CudaSuperSmootherError::InvalidInput(format!(
                "output pinned buffer length mismatch: expected {}, got {}",
                cols * rows,
                out_tm_pinned.len()
            )));
        }
        let (first_valids, period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        let dev = self.run_many_series_kernel(data_tm_f32, cols, rows, &first_valids, period)?;
        unsafe {
            dev.buf
                .async_copy_to(out_tm_pinned.as_mut_slice(), &self.stream)
                .map_err(CudaSuperSmootherError::Cuda)?;
        }
        self.stream
            .synchronize()
            .map_err(CudaSuperSmootherError::Cuda)
    }
}

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};
    use crate::indicators::moving_averages::supersmoother::SuperSmootherParams;

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

    struct BatchDevState {
        cuda: CudaSuperSmoother,
        d_prices: DeviceBuffer<f32>,
        d_periods: DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out: DeviceBuffer<f32>,
    }
    impl CudaBenchState for BatchDevState {
        fn launch(&mut self) {
            self.cuda
                .launch_batch_kernel(
                    &self.d_prices,
                    &self.d_periods,
                    self.series_len,
                    self.n_combos,
                    self.first_valid,
                    &mut self.d_out,
                )
                .expect("supersmoother batch kernel");
            self.cuda.stream.synchronize().expect("supersmoother sync");
        }
    }

    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaSuperSmoother::new(0).expect("cuda supersmoother");
        let price = gen_series(ONE_SERIES_LEN);
        let sweep = crate::indicators::moving_averages::supersmoother::SuperSmootherBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1),
        };
        let (combos, first_valid, series_len) =
            CudaSuperSmoother::prepare_batch_inputs(&price, &sweep)
                .expect("supersmoother prepare batch");
        let periods_i32: Vec<i32> = combos
            .iter()
            .map(|p| p.period.unwrap_or(0) as i32)
            .collect();
        let n_combos = periods_i32.len();

        let d_prices = DeviceBuffer::from_slice(&price).expect("d_prices");
        let d_periods = DeviceBuffer::from_slice(&periods_i32).expect("d_periods");
        let d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(series_len.checked_mul(n_combos).expect("out size"))
        }
        .expect("d_out");
        cuda.stream.synchronize().expect("sync after prep");

        Box::new(BatchDevState {
            cuda,
            d_prices,
            d_periods,
            series_len,
            n_combos,
            first_valid,
            d_out,
        })
    }

    struct ManyDevState {
        cuda: CudaSuperSmoother,
        d_prices_tm: DeviceBuffer<f32>,
        d_first_valids: DeviceBuffer<i32>,
        period: usize,
        cols: usize,
        rows: usize,
        d_out_tm: DeviceBuffer<f32>,
    }
    impl CudaBenchState for ManyDevState {
        fn launch(&mut self) {
            self.cuda
                .launch_many_series_kernel(
                    &self.d_prices_tm,
                    &self.d_first_valids,
                    self.cols,
                    self.rows,
                    self.period,
                    &mut self.d_out_tm,
                )
                .expect("supersmoother many-series kernel");
            self.cuda.stream.synchronize().expect("supersmoother sync");
        }
    }

    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaSuperSmoother::new(0).expect("cuda supersmoother");
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_LEN;
        let data_tm = gen_time_major_prices(cols, rows);
        let params = SuperSmootherParams { period: Some(64) };
        let (first_valids, period) =
            CudaSuperSmoother::prepare_many_series_inputs(&data_tm, cols, rows, &params)
                .expect("supersmoother prepare many");

        let d_prices_tm = DeviceBuffer::from_slice(&data_tm).expect("d_prices_tm");
        let d_first_valids = DeviceBuffer::from_slice(&first_valids).expect("d_first_valids");
        let d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(cols.checked_mul(rows).expect("out size")) }
                .expect("d_out_tm");
        cuda.stream.synchronize().expect("sync after prep");

        Box::new(ManyDevState {
            cuda,
            d_prices_tm,
            d_first_valids,
            period,
            cols,
            rows,
            d_out_tm,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "supersmoother",
                "one_series_many_params",
                "supersmoother_cuda_batch_dev",
                "1m_x_250",
                prep_one_series_many_params,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "supersmoother",
                "many_series_one_param",
                "supersmoother_cuda_many_series_one_param",
                "250x1m",
                prep_many_series_one_param,
            )
            .with_sample_size(5)
            .with_mem_required(bytes_many_series_one_param()),
        ]
    }
}
