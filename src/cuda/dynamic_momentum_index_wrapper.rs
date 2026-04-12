#![cfg(feature = "cuda")]

use crate::indicators::dynamic_momentum_index::{
    expand_grid_dynamic_momentum_index, DynamicMomentumIndexBatchRange, DynamicMomentumIndexParams,
};
use cust::context::Context;
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, GridSize};
use cust::launch;
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::Module;
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::sync::Arc;
use thiserror::Error;

const DYNAMIC_MOMENTUM_INDEX_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaDynamicMomentumIndexError {
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
    #[error("launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge {
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
    },
}

pub struct DynamicMomentumIndexDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl DynamicMomentumIndexDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct CudaDynamicMomentumIndexBatchResult {
    pub outputs: DynamicMomentumIndexDeviceArrayF64,
    pub combos: Vec<DynamicMomentumIndexParams>,
}

pub struct CudaDynamicMomentumIndex {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaDynamicMomentumIndex {
    pub fn new(device_id: usize) -> Result<Self, CudaDynamicMomentumIndexError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("dynamic_momentum_index_kernel")?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        Ok(Self {
            module,
            stream,
            context,
            device_id: device_id as u32,
        })
    }

    #[inline]
    pub fn context_arc(&self) -> Arc<Context> {
        self.context.clone()
    }

    #[inline]
    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    pub fn synchronize(&self) -> Result<(), CudaDynamicMomentumIndexError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn longest_valid_run(data: &[f64]) -> usize {
        let mut best = 0usize;
        let mut cur = 0usize;
        for &value in data {
            if value.is_finite() {
                cur += 1;
                best = best.max(cur);
            } else {
                cur = 0;
            }
        }
        best
    }

    fn warmup_bars(
        volatility_period: usize,
        volatility_sma_period: usize,
        lower_limit: usize,
    ) -> usize {
        (volatility_period + volatility_sma_period - 2).max(lower_limit)
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaDynamicMomentumIndexError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaDynamicMomentumIndexError::OutOfMemory {
                    required,
                    free,
                    headroom,
                });
            }
        }
        Ok(())
    }

    fn validate_launch(
        &self,
        grid: GridSize,
        block: BlockSize,
    ) -> Result<(), CudaDynamicMomentumIndexError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaDynamicMomentumIndexError::LaunchConfigTooLarge {
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

    pub fn batch_dev(
        &self,
        data: &[f64],
        sweep: &DynamicMomentumIndexBatchRange,
    ) -> Result<CudaDynamicMomentumIndexBatchResult, CudaDynamicMomentumIndexError> {
        if data.is_empty() {
            return Err(CudaDynamicMomentumIndexError::InvalidInput(
                "empty input".into(),
            ));
        }
        let longest = Self::longest_valid_run(data);
        if longest == 0 {
            return Err(CudaDynamicMomentumIndexError::InvalidInput(
                "all values are NaN".into(),
            ));
        }

        let combos = expand_grid_dynamic_momentum_index(sweep);
        if combos.is_empty() {
            return Err(CudaDynamicMomentumIndexError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let rows = combos.len();
        let cols = data.len();
        let mut max_volatility_period = 0usize;
        let mut max_volatility_sma_period = 0usize;
        let mut max_upper_limit = 0usize;
        let mut max_needed = 0usize;
        let mut rsi_periods = Vec::with_capacity(rows);
        let mut volatility_periods = Vec::with_capacity(rows);
        let mut volatility_sma_periods = Vec::with_capacity(rows);
        let mut upper_limits = Vec::with_capacity(rows);
        let mut lower_limits = Vec::with_capacity(rows);

        for combo in &combos {
            let rsi_period = combo.rsi_period.unwrap_or(14);
            let volatility_period = combo.volatility_period.unwrap_or(5);
            let volatility_sma_period = combo.volatility_sma_period.unwrap_or(10);
            let upper_limit = combo.upper_limit.unwrap_or(30);
            let lower_limit = combo.lower_limit.unwrap_or(5);
            if rsi_period == 0
                || volatility_period == 0
                || volatility_sma_period == 0
                || upper_limit == 0
                || lower_limit == 0
                || lower_limit > upper_limit
            {
                return Err(CudaDynamicMomentumIndexError::InvalidInput(
                    "invalid parameters".into(),
                ));
            }
            max_volatility_period = max_volatility_period.max(volatility_period);
            max_volatility_sma_period = max_volatility_sma_period.max(volatility_sma_period);
            max_upper_limit = max_upper_limit.max(upper_limit);
            max_needed = max_needed
                .max(Self::warmup_bars(volatility_period, volatility_sma_period, lower_limit) + 1);
            rsi_periods.push(rsi_period as i32);
            volatility_periods.push(volatility_period as i32);
            volatility_sma_periods.push(volatility_sma_period as i32);
            upper_limits.push(upper_limit as i32);
            lower_limits.push(lower_limit as i32);
        }

        if longest < max_needed {
            return Err(CudaDynamicMomentumIndexError::InvalidInput(format!(
                "not enough valid data: needed={max_needed}, valid={longest}"
            )));
        }

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaDynamicMomentumIndexError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = rsi_periods
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| {
                volatility_periods
                    .len()
                    .checked_mul(std::mem::size_of::<i32>())
                    .and_then(|other| value.checked_add(other))
            })
            .and_then(|value| {
                volatility_sma_periods
                    .len()
                    .checked_mul(std::mem::size_of::<i32>())
                    .and_then(|other| value.checked_add(other))
            })
            .and_then(|value| {
                upper_limits
                    .len()
                    .checked_mul(std::mem::size_of::<i32>())
                    .and_then(|other| value.checked_add(other))
            })
            .and_then(|value| {
                lower_limits
                    .len()
                    .checked_mul(std::mem::size_of::<i32>())
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| {
                CudaDynamicMomentumIndexError::InvalidInput("params bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaDynamicMomentumIndexError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaDynamicMomentumIndexError::InvalidInput("output bytes overflow".into())
            })?;
        let close_scratch = rows.checked_mul(max_volatility_period).ok_or_else(|| {
            CudaDynamicMomentumIndexError::InvalidInput("close scratch overflow".into())
        })?;
        let std_scratch = rows.checked_mul(max_volatility_sma_period).ok_or_else(|| {
            CudaDynamicMomentumIndexError::InvalidInput("std scratch overflow".into())
        })?;
        let gain_scratch = rows.checked_mul(max_upper_limit).ok_or_else(|| {
            CudaDynamicMomentumIndexError::InvalidInput("gain scratch overflow".into())
        })?;
        let loss_scratch = rows.checked_mul(max_upper_limit).ok_or_else(|| {
            CudaDynamicMomentumIndexError::InvalidInput("loss scratch overflow".into())
        })?;
        let scratch_bytes = close_scratch
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| {
                std_scratch
                    .checked_mul(std::mem::size_of::<f64>())
                    .and_then(|other| value.checked_add(other))
            })
            .and_then(|value| {
                gain_scratch
                    .checked_mul(std::mem::size_of::<f64>())
                    .and_then(|other| value.checked_add(other))
            })
            .and_then(|value| {
                loss_scratch
                    .checked_mul(std::mem::size_of::<f64>())
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| {
                CudaDynamicMomentumIndexError::InvalidInput("scratch bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .and_then(|value| value.checked_add(scratch_bytes))
            .ok_or_else(|| {
                CudaDynamicMomentumIndexError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_data = DeviceBuffer::from_slice(data)?;
        let d_rsi_periods = DeviceBuffer::from_slice(&rsi_periods)?;
        let d_volatility_periods = DeviceBuffer::from_slice(&volatility_periods)?;
        let d_volatility_sma_periods = DeviceBuffer::from_slice(&volatility_sma_periods)?;
        let d_upper_limits = DeviceBuffer::from_slice(&upper_limits)?;
        let d_lower_limits = DeviceBuffer::from_slice(&lower_limits)?;
        let d_close_buffer = unsafe { DeviceBuffer::<f64>::uninitialized(close_scratch)? };
        let d_std_buffer = unsafe { DeviceBuffer::<f64>::uninitialized(std_scratch)? };
        let d_gain_buffer = unsafe { DeviceBuffer::<f64>::uninitialized(gain_scratch)? };
        let d_loss_buffer = unsafe { DeviceBuffer::<f64>::uninitialized(loss_scratch)? };
        let d_out = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("dynamic_momentum_index_batch_f64")
            .map_err(|_| CudaDynamicMomentumIndexError::MissingKernelSymbol {
                name: "dynamic_momentum_index_batch_f64",
            })?;
        let grid_x =
            ((rows as u32) + DYNAMIC_MOMENTUM_INDEX_BLOCK_X - 1) / DYNAMIC_MOMENTUM_INDEX_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(DYNAMIC_MOMENTUM_INDEX_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_data.as_device_ptr(),
                cols as i32,
                d_rsi_periods.as_device_ptr(),
                d_volatility_periods.as_device_ptr(),
                d_volatility_sma_periods.as_device_ptr(),
                d_upper_limits.as_device_ptr(),
                d_lower_limits.as_device_ptr(),
                rows as i32,
                max_volatility_period as i32,
                max_volatility_sma_period as i32,
                max_upper_limit as i32,
                d_close_buffer.as_device_ptr(),
                d_std_buffer.as_device_ptr(),
                d_gain_buffer.as_device_ptr(),
                d_loss_buffer.as_device_ptr(),
                d_out.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaDynamicMomentumIndexBatchResult {
            outputs: DynamicMomentumIndexDeviceArrayF64 {
                buf: d_out,
                rows,
                cols,
            },
            combos,
        })
    }
}
