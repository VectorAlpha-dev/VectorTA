//! CUDA scaffolding for the Ultimate Moving Average (UMA).
//!
//! The GPU path mirrors the scalar UMA implementation by evaluating each
//! parameter combination sequentially on device while keeping the series data
//! resident. Sliding sums produce the max-length mean/std window and the
//! adaptive power weights plus optional smoothing are performed entirely on the
//! GPU. The wrapper exposes convenience helpers that align with the existing
//! ALMA/DMA/VWMA APIs.

#![cfg(feature = "cuda")]

use super::DeviceArrayF32;
use crate::indicators::moving_averages::uma::{expand_grid_uma, UmaBatchRange, UmaParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::DeviceBuffer;
use cust::module::Module;
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use cust::sys as cu;
use std::env;
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaUmaError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaUmaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaUmaError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaUmaError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}

impl std::error::Error for CudaUmaError {}

pub struct CudaUma {
    module: Module,
    stream: Stream,
    _context: Context,
}

impl CudaUma {
    pub fn new(device_id: usize) -> Result<Self, CudaUmaError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaUmaError::Cuda(e.to_string()))?;

        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaUmaError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaUmaError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/uma_kernel.ptx"));
        let module = Module::from_ptx(ptx, &[]).map_err(|e| CudaUmaError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaUmaError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
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
        unsafe {
            let mut free: usize = 0;
            let mut total: usize = 0;
            let res = cu::cuMemGetInfo_v2(&mut free as *mut usize, &mut total as *mut usize);
            if res == cu::CUresult::CUDA_SUCCESS {
                Some((free, total))
            } else {
                None
            }
        }
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

    pub fn uma_batch_dev(
        &self,
        prices: &[f32],
        volumes: Option<&[f32]>,
        sweep: &UmaBatchRange,
    ) -> Result<DeviceArrayF32, CudaUmaError> {
        let inputs = Self::prepare_batch_inputs(prices, volumes, sweep)?;
        self.run_batch_kernel(prices, volumes, &inputs)
    }

    pub fn uma_batch_into_host_f32(
        &self,
        prices: &[f32],
        volumes: Option<&[f32]>,
        sweep: &UmaBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<UmaParams>), CudaUmaError> {
        let inputs = Self::prepare_batch_inputs(prices, volumes, sweep)?;
        let expected = inputs.series_len * inputs.combos.len();
        if out.len() != expected {
            return Err(CudaUmaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out.len(),
                expected
            )));
        }

        let arr = self.run_batch_kernel(prices, volumes, &inputs)?;
        arr.buf
            .copy_to(out)
            .map_err(|e| CudaUmaError::Cuda(e.to_string()))?;
        let BatchInputs { combos, .. } = inputs;
        Ok((arr.rows, arr.cols, combos))
    }

    pub fn uma_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_volumes: Option<&DeviceBuffer<f32>>,
        d_accelerators: &DeviceBuffer<f32>,
        d_min_lengths: &DeviceBuffer<i32>,
        d_max_lengths: &DeviceBuffer<i32>,
        d_smooth_lengths: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        has_volume: bool,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaUmaError> {
        let mut d_raw: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(series_len * n_combos) }
                .map_err(|e| CudaUmaError::Cuda(e.to_string()))?;

        self.launch_batch_kernel(
            d_prices,
            d_volumes,
            d_accelerators,
            d_min_lengths,
            d_max_lengths,
            d_smooth_lengths,
            series_len,
            n_combos,
            first_valid,
            has_volume,
            &mut d_raw,
            d_out,
        )?;

        self.stream
            .synchronize()
            .map_err(|e| CudaUmaError::Cuda(e.to_string()))
    }

    pub fn uma_many_series_one_param_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_volumes_tm: Option<&DeviceBuffer<f32>>,
        accelerator: f32,
        min_length: i32,
        max_length: i32,
        smooth_length: i32,
        num_series: usize,
        series_len: usize,
        d_first_valids: &DeviceBuffer<i32>,
        has_volume: bool,
        d_raw_tm: &mut DeviceBuffer<f32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaUmaError> {
        if accelerator < 1.0 {
            return Err(CudaUmaError::InvalidInput(format!(
                "accelerator must be >= 1.0 (got {})",
                accelerator
            )));
        }
        if min_length <= 0 || max_length <= 0 || smooth_length <= 0 {
            return Err(CudaUmaError::InvalidInput(
                "min_length, max_length, and smooth_length must be positive".into(),
            ));
        }
        if min_length > max_length {
            return Err(CudaUmaError::InvalidInput(format!(
                "min_length {} greater than max_length {}",
                min_length, max_length
            )));
        }
        if num_series == 0 || series_len == 0 {
            return Err(CudaUmaError::InvalidInput(
                "num_series and series_len must be positive".into(),
            ));
        }
        if num_series > i32::MAX as usize || series_len > i32::MAX as usize {
            return Err(CudaUmaError::InvalidInput(
                "arguments exceed kernel limits".into(),
            ));
        }

        self.launch_many_series_kernel(
            d_prices_tm,
            d_volumes_tm,
            accelerator,
            min_length,
            max_length,
            smooth_length,
            num_series,
            series_len,
            d_first_valids,
            has_volume,
            d_raw_tm,
            d_out_tm,
        )?;

        self.stream
            .synchronize()
            .map_err(|e| CudaUmaError::Cuda(e.to_string()))
    }

    pub fn uma_many_series_one_param_time_major_dev(
        &self,
        prices_tm_f32: &[f32],
        volumes_tm_f32: Option<&[f32]>,
        cols: usize,
        rows: usize,
        params: &UmaParams,
    ) -> Result<DeviceArrayF32, CudaUmaError> {
        let inputs =
            Self::prepare_many_series_inputs(prices_tm_f32, volumes_tm_f32, cols, rows, params)?;
        self.run_many_series_kernel(prices_tm_f32, volumes_tm_f32, &inputs)
    }

    pub fn uma_many_series_one_param_time_major_into_host_f32(
        &self,
        prices_tm_f32: &[f32],
        volumes_tm_f32: Option<&[f32]>,
        cols: usize,
        rows: usize,
        params: &UmaParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaUmaError> {
        if out_tm.len() != cols * rows {
            return Err(CudaUmaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out_tm.len(),
                cols * rows
            )));
        }
        let inputs =
            Self::prepare_many_series_inputs(prices_tm_f32, volumes_tm_f32, cols, rows, params)?;
        let arr = self.run_many_series_kernel(prices_tm_f32, volumes_tm_f32, &inputs)?;
        arr.buf
            .copy_to(out_tm)
            .map_err(|e| CudaUmaError::Cuda(e.to_string()))?;
        Ok(())
    }

    fn run_batch_kernel(
        &self,
        prices: &[f32],
        volumes: Option<&[f32]>,
        inputs: &BatchInputs,
    ) -> Result<DeviceArrayF32, CudaUmaError> {
        let n_combos = inputs.combos.len();
        let series_len = inputs.series_len;

        let price_bytes = prices.len() * std::mem::size_of::<f32>();
        let volume_bytes = if inputs.has_volume {
            series_len * std::mem::size_of::<f32>()
        } else {
            0
        };
        let accel_bytes = n_combos * std::mem::size_of::<f32>();
        let len_bytes = n_combos * std::mem::size_of::<i32>() * 3;
        let out_bytes = n_combos * series_len * std::mem::size_of::<f32>();
        let raw_bytes = out_bytes;
        let required = price_bytes + volume_bytes + accel_bytes + len_bytes + out_bytes + raw_bytes;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaUmaError::InvalidInput(
                "not enough device memory for UMA batch".into(),
            ));
        }

        let d_prices =
            DeviceBuffer::from_slice(prices).map_err(|e| CudaUmaError::Cuda(e.to_string()))?;
        let d_volumes = if let Some(v) = volumes {
            Some(DeviceBuffer::from_slice(v).map_err(|e| CudaUmaError::Cuda(e.to_string()))?)
        } else {
            None
        };
        let d_accels = DeviceBuffer::from_slice(&inputs.accelerators)
            .map_err(|e| CudaUmaError::Cuda(e.to_string()))?;
        let d_min = DeviceBuffer::from_slice(&inputs.min_lengths)
            .map_err(|e| CudaUmaError::Cuda(e.to_string()))?;
        let d_max = DeviceBuffer::from_slice(&inputs.max_lengths)
            .map_err(|e| CudaUmaError::Cuda(e.to_string()))?;
        let d_smooth = DeviceBuffer::from_slice(&inputs.smooth_lengths)
            .map_err(|e| CudaUmaError::Cuda(e.to_string()))?;

        let mut d_raw: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(series_len * n_combos) }
                .map_err(|e| CudaUmaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(series_len * n_combos) }
                .map_err(|e| CudaUmaError::Cuda(e.to_string()))?;

        self.launch_batch_kernel(
            &d_prices,
            d_volumes.as_ref(),
            &d_accels,
            &d_min,
            &d_max,
            &d_smooth,
            series_len,
            n_combos,
            inputs.first_valid,
            inputs.has_volume,
            &mut d_raw,
            &mut d_out,
        )?;

        self.stream
            .synchronize()
            .map_err(|e| CudaUmaError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: series_len,
        })
    }

    fn run_many_series_kernel(
        &self,
        prices_tm_f32: &[f32],
        volumes_tm_f32: Option<&[f32]>,
        inputs: &ManySeriesInputs,
    ) -> Result<DeviceArrayF32, CudaUmaError> {
        let prices_bytes = prices_tm_f32.len() * std::mem::size_of::<f32>();
        let volume_bytes = if inputs.has_volume {
            inputs.num_series * inputs.series_len * std::mem::size_of::<f32>()
        } else {
            0
        };
        let first_valid_bytes = inputs.num_series * std::mem::size_of::<i32>();
        let out_bytes = inputs.num_series * inputs.series_len * std::mem::size_of::<f32>();
        let raw_bytes = out_bytes;
        let required = prices_bytes + volume_bytes + first_valid_bytes + out_bytes + raw_bytes;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaUmaError::InvalidInput(
                "not enough device memory for UMA many-series".into(),
            ));
        }

        let d_prices_tm = DeviceBuffer::from_slice(prices_tm_f32)
            .map_err(|e| CudaUmaError::Cuda(e.to_string()))?;
        let d_volumes_tm = if inputs.has_volume {
            let slice = volumes_tm_f32.ok_or_else(|| {
                CudaUmaError::InvalidInput("volume matrix missing despite has_volume".into())
            })?;
            Some(DeviceBuffer::from_slice(slice).map_err(|e| CudaUmaError::Cuda(e.to_string()))?)
        } else {
            None
        };
        let d_first_valids = DeviceBuffer::from_slice(&inputs.first_valids)
            .map_err(|e| CudaUmaError::Cuda(e.to_string()))?;
        let mut d_raw_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(inputs.num_series * inputs.series_len) }
                .map_err(|e| CudaUmaError::Cuda(e.to_string()))?;
        let mut d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(inputs.num_series * inputs.series_len) }
                .map_err(|e| CudaUmaError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(
            &d_prices_tm,
            d_volumes_tm.as_ref(),
            inputs.accelerator,
            inputs.min_length,
            inputs.max_length,
            inputs.smooth_length,
            inputs.num_series,
            inputs.series_len,
            &d_first_valids,
            inputs.has_volume,
            &mut d_raw_tm,
            &mut d_out_tm,
        )?;

        self.stream
            .synchronize()
            .map_err(|e| CudaUmaError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out_tm,
            rows: inputs.series_len,
            cols: inputs.num_series,
        })
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_volumes: Option<&DeviceBuffer<f32>>,
        d_accelerators: &DeviceBuffer<f32>,
        d_min_lengths: &DeviceBuffer<i32>,
        d_max_lengths: &DeviceBuffer<i32>,
        d_smooth_lengths: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        has_volume: bool,
        d_raw: &mut DeviceBuffer<f32>,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaUmaError> {
        if series_len > i32::MAX as usize {
            return Err(CudaUmaError::InvalidInput(
                "series too long for kernel argument width".into(),
            ));
        }
        if n_combos > i32::MAX as usize {
            return Err(CudaUmaError::InvalidInput(
                "too many parameter combinations".into(),
            ));
        }

        let func = self
            .module
            .get_function("uma_batch_f32")
            .map_err(|e| CudaUmaError::Cuda(e.to_string()))?;

        let grid: GridSize = (n_combos as u32, 1, 1).into();
        let block: BlockSize = (1, 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut volumes_ptr = d_volumes
                .map(|buf| buf.as_device_ptr().as_raw())
                .unwrap_or(0);
            let mut has_volume_i = if has_volume { 1i32 } else { 0i32 };
            let mut accel_ptr = d_accelerators.as_device_ptr().as_raw();
            let mut min_ptr = d_min_lengths.as_device_ptr().as_raw();
            let mut max_ptr = d_max_lengths.as_device_ptr().as_raw();
            let mut smooth_ptr = d_smooth_lengths.as_device_ptr().as_raw();
            let mut series_len_i = series_len as i32;
            let mut combos_i = n_combos as i32;
            let mut first_valid_i = first_valid as i32;
            let mut raw_ptr = d_raw.as_device_ptr().as_raw();
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut volumes_ptr as *mut _ as *mut c_void,
                &mut has_volume_i as *mut _ as *mut c_void,
                &mut accel_ptr as *mut _ as *mut c_void,
                &mut min_ptr as *mut _ as *mut c_void,
                &mut max_ptr as *mut _ as *mut c_void,
                &mut smooth_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut raw_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaUmaError::Cuda(e.to_string()))?
        }
        Ok(())
    }

    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_volumes_tm: Option<&DeviceBuffer<f32>>,
        accelerator: f32,
        min_length: i32,
        max_length: i32,
        smooth_length: i32,
        num_series: usize,
        series_len: usize,
        d_first_valids: &DeviceBuffer<i32>,
        has_volume: bool,
        d_raw_tm: &mut DeviceBuffer<f32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaUmaError> {
        if num_series > i32::MAX as usize || series_len > i32::MAX as usize {
            return Err(CudaUmaError::InvalidInput(
                "series dimensions exceed kernel limits".into(),
            ));
        }

        let func = self
            .module
            .get_function("uma_many_series_one_param_f32")
            .map_err(|e| CudaUmaError::Cuda(e.to_string()))?;

        let grid: GridSize = (num_series as u32, 1, 1).into();
        let block: BlockSize = (1, 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut volumes_ptr = d_volumes_tm
                .map(|buf| buf.as_device_ptr().as_raw())
                .unwrap_or(0);
            let mut has_volume_i = if has_volume { 1i32 } else { 0i32 };
            let mut accel = accelerator;
            let mut min_i = min_length;
            let mut max_i = max_length;
            let mut smooth_i = smooth_length;
            let mut num_series_i = num_series as i32;
            let mut series_len_i = series_len as i32;
            let mut first_valids_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut raw_ptr = d_raw_tm.as_device_ptr().as_raw();
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut volumes_ptr as *mut _ as *mut c_void,
                &mut has_volume_i as *mut _ as *mut c_void,
                &mut accel as *mut _ as *mut c_void,
                &mut min_i as *mut _ as *mut c_void,
                &mut max_i as *mut _ as *mut c_void,
                &mut smooth_i as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut first_valids_ptr as *mut _ as *mut c_void,
                &mut raw_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaUmaError::Cuda(e.to_string()))?
        }
        Ok(())
    }

    fn prepare_batch_inputs(
        prices: &[f32],
        volumes: Option<&[f32]>,
        sweep: &UmaBatchRange,
    ) -> Result<BatchInputs, CudaUmaError> {
        if prices.is_empty() {
            return Err(CudaUmaError::InvalidInput("empty price series".into()));
        }
        if let Some(v) = volumes {
            if v.len() != prices.len() {
                return Err(CudaUmaError::InvalidInput(format!(
                    "price/volume length mismatch: {} vs {}",
                    prices.len(),
                    v.len()
                )));
            }
        }

        let combos = expand_grid_uma(sweep);
        if combos.is_empty() {
            return Err(CudaUmaError::InvalidInput(
                "no UMA parameter combinations".into(),
            ));
        }

        let series_len = prices.len();
        let first_valid = prices
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaUmaError::InvalidInput("all price values are NaN".into()))?;

        let mut accelerators = Vec::with_capacity(combos.len());
        let mut min_lengths = Vec::with_capacity(combos.len());
        let mut max_lengths = Vec::with_capacity(combos.len());
        let mut smooth_lengths = Vec::with_capacity(combos.len());

        for prm in &combos {
            let accel = prm.accelerator.unwrap_or(1.0);
            let min_len = prm.min_length.unwrap_or(5);
            let max_len = prm.max_length.unwrap_or(50);
            let smooth_len = prm.smooth_length.unwrap_or(4);

            if accel < 1.0 {
                return Err(CudaUmaError::InvalidInput(format!(
                    "accelerator must be >= 1.0 (got {})",
                    accel
                )));
            }
            if min_len == 0 {
                return Err(CudaUmaError::InvalidInput(
                    "min_length must be positive".into(),
                ));
            }
            if max_len == 0 {
                return Err(CudaUmaError::InvalidInput(
                    "max_length must be positive".into(),
                ));
            }
            if min_len > max_len {
                return Err(CudaUmaError::InvalidInput(format!(
                    "min_length {} greater than max_length {}",
                    min_len, max_len
                )));
            }
            if smooth_len == 0 {
                return Err(CudaUmaError::InvalidInput(
                    "smooth_length must be positive".into(),
                ));
            }
            if max_len > i32::MAX as usize
                || min_len > i32::MAX as usize
                || smooth_len > i32::MAX as usize
            {
                return Err(CudaUmaError::InvalidInput(
                    "parameters exceed kernel limits".into(),
                ));
            }
            let valid_available = series_len - first_valid;
            if valid_available < max_len {
                return Err(CudaUmaError::InvalidInput(format!(
                    "not enough valid data: need >= {}, have {}",
                    max_len, valid_available
                )));
            }

            accelerators.push(accel as f32);
            min_lengths.push(min_len as i32);
            max_lengths.push(max_len as i32);
            smooth_lengths.push(smooth_len as i32);
        }

        Ok(BatchInputs {
            combos,
            accelerators,
            min_lengths,
            max_lengths,
            smooth_lengths,
            first_valid,
            series_len,
            has_volume: volumes.map_or(false, |v| !v.is_empty()),
        })
    }

    fn prepare_many_series_inputs(
        prices_tm_f32: &[f32],
        volumes_tm_f32: Option<&[f32]>,
        cols: usize,
        rows: usize,
        params: &UmaParams,
    ) -> Result<ManySeriesInputs, CudaUmaError> {
        if cols == 0 || rows == 0 {
            return Err(CudaUmaError::InvalidInput(
                "num_series or series_len is zero".into(),
            ));
        }
        if prices_tm_f32.len() != cols * rows {
            return Err(CudaUmaError::InvalidInput(format!(
                "price matrix length {} != cols*rows {}",
                prices_tm_f32.len(),
                cols * rows
            )));
        }
        if let Some(v) = volumes_tm_f32 {
            if v.len() != cols * rows {
                return Err(CudaUmaError::InvalidInput(format!(
                    "volume matrix length {} != cols*rows {}",
                    v.len(),
                    cols * rows
                )));
            }
        }

        let accelerator = params.accelerator.unwrap_or(1.0);
        let min_length = params.min_length.unwrap_or(5);
        let max_length = params.max_length.unwrap_or(50);
        let smooth_length = params.smooth_length.unwrap_or(4);

        if accelerator < 1.0 {
            return Err(CudaUmaError::InvalidInput(format!(
                "accelerator must be >= 1.0 (got {})",
                accelerator
            )));
        }
        if min_length == 0 {
            return Err(CudaUmaError::InvalidInput(
                "min_length must be positive".into(),
            ));
        }
        if max_length == 0 {
            return Err(CudaUmaError::InvalidInput(
                "max_length must be positive".into(),
            ));
        }
        if smooth_length == 0 {
            return Err(CudaUmaError::InvalidInput(
                "smooth_length must be positive".into(),
            ));
        }
        if min_length > max_length {
            return Err(CudaUmaError::InvalidInput(format!(
                "min_length {} greater than max_length {}",
                min_length, max_length
            )));
        }
        if min_length > i32::MAX as usize
            || max_length > i32::MAX as usize
            || smooth_length > i32::MAX as usize
        {
            return Err(CudaUmaError::InvalidInput(
                "parameters exceed kernel limits".into(),
            ));
        }

        let mut first_valids = vec![0i32; cols];
        for series in 0..cols {
            let mut fv = None;
            for t in 0..rows {
                let val = prices_tm_f32[t * cols + series];
                if !val.is_nan() {
                    fv = Some(t);
                    break;
                }
            }
            let fv = fv.ok_or_else(|| {
                CudaUmaError::InvalidInput(format!("series {} consists entirely of NaNs", series))
            })?;
            if rows - fv < max_length {
                return Err(CudaUmaError::InvalidInput(format!(
                    "series {} not enough valid data (needed >= {}, valid = {})",
                    series,
                    max_length,
                    rows - fv
                )));
            }
            first_valids[series] = fv as i32;
        }

        Ok(ManySeriesInputs {
            first_valids,
            accelerator: accelerator as f32,
            min_length: min_length as i32,
            max_length: max_length as i32,
            smooth_length: smooth_length as i32,
            num_series: cols,
            series_len: rows,
            has_volume: volumes_tm_f32.map_or(false, |v| !v.is_empty()),
        })
    }
}

// ---------- Bench profiles (custom; UMA needs volume option) ----------

pub mod benches {
    use super::*;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};

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

    struct UmaBatchState {
        cuda: CudaUma,
        price: Vec<f32>,
        sweep: UmaBatchRange,
    }
    impl CudaBenchState for UmaBatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .uma_batch_dev(&self.price, None, &self.sweep)
                .expect("launch uma batch");
        }
    }
    fn prep_uma_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaUma::new(0).expect("cuda uma");
        let price = gen_series(ONE_SERIES_LEN);
        // Build 250-combo sweep by varying max_length, holding others constant.
        let sweep = UmaBatchRange {
            accelerator: (1.0, 1.0, 0.0),
            min_length: (5, 5, 0),
            max_length: (16, 16 + PARAM_SWEEP - 1, 1),
            smooth_length: (4, 4, 0),
        };
        Box::new(UmaBatchState { cuda, price, sweep })
    }

    struct UmaManyState {
        cuda: CudaUma,
        data_tm: Vec<f32>,
        cols: usize,
        rows: usize,
        params: UmaParams,
    }
    impl CudaBenchState for UmaManyState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .uma_many_series_one_param_time_major_dev(
                    &self.data_tm,
                    None,
                    self.cols,
                    self.rows,
                    &self.params,
                )
                .expect("launch uma many-series");
        }
    }
    fn prep_uma_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaUma::new(0).expect("cuda uma");
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_LEN;
        let data_tm = gen_time_major_prices(cols, rows);
        let params = UmaParams {
            accelerator: Some(1.0),
            min_length: Some(5),
            max_length: Some(64),
            smooth_length: Some(4),
        };
        Box::new(UmaManyState {
            cuda,
            data_tm,
            cols,
            rows,
            params,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "uma",
                "one_series_many_params",
                "uma_cuda_batch_dev",
                "1m_x_250",
                prep_uma_one_series_many_params,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "uma",
                "many_series_one_param",
                "uma_cuda_many_series_one_param",
                "250x1m",
                prep_uma_many_series_one_param,
            )
            .with_sample_size(5)
            .with_mem_required(bytes_many_series_one_param()),
        ]
    }
}

struct BatchInputs {
    combos: Vec<UmaParams>,
    accelerators: Vec<f32>,
    min_lengths: Vec<i32>,
    max_lengths: Vec<i32>,
    smooth_lengths: Vec<i32>,
    first_valid: usize,
    series_len: usize,
    has_volume: bool,
}

struct ManySeriesInputs {
    first_valids: Vec<i32>,
    accelerator: f32,
    min_length: i32,
    max_length: i32,
    smooth_length: i32,
    num_series: usize,
    series_len: usize,
    has_volume: bool,
}
