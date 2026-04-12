#![cfg(feature = "cuda")]

use crate::indicators::volatility_ratio_adaptive_rsx::{
    expand_grid_volatility_ratio_adaptive_rsx, VolatilityRatioAdaptiveRsxBatchRange,
    VolatilityRatioAdaptiveRsxParams,
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

const VOLATILITY_RATIO_ADAPTIVE_RSX_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const DEFAULT_PERIOD: usize = 14;
const DEFAULT_SPEED: f64 = 0.5;

#[derive(Debug, Error)]
pub enum CudaVolatilityRatioAdaptiveRsxError {
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

pub struct VolatilityRatioAdaptiveRsxDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl VolatilityRatioAdaptiveRsxDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct VolatilityRatioAdaptiveRsxDeviceArrayF64Pair {
    pub line: VolatilityRatioAdaptiveRsxDeviceArrayF64,
    pub signal: VolatilityRatioAdaptiveRsxDeviceArrayF64,
}

impl VolatilityRatioAdaptiveRsxDeviceArrayF64Pair {
    #[inline]
    pub fn rows(&self) -> usize {
        self.line.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.line.cols
    }
}

pub struct CudaVolatilityRatioAdaptiveRsxBatchResult {
    pub outputs: VolatilityRatioAdaptiveRsxDeviceArrayF64Pair,
    pub combos: Vec<VolatilityRatioAdaptiveRsxParams>,
}

pub struct CudaVolatilityRatioAdaptiveRsx {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaVolatilityRatioAdaptiveRsx {
    pub fn new(device_id: usize) -> Result<Self, CudaVolatilityRatioAdaptiveRsxError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("volatility_ratio_adaptive_rsx_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaVolatilityRatioAdaptiveRsxError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn first_valid_value(data: &[f64]) -> Option<usize> {
        data.iter().position(|value| value.is_finite())
    }

    fn will_fit(
        required: usize,
        headroom: usize,
    ) -> Result<(), CudaVolatilityRatioAdaptiveRsxError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaVolatilityRatioAdaptiveRsxError::OutOfMemory {
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
    ) -> Result<(), CudaVolatilityRatioAdaptiveRsxError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaVolatilityRatioAdaptiveRsxError::LaunchConfigTooLarge {
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
        sweep: &VolatilityRatioAdaptiveRsxBatchRange,
    ) -> Result<CudaVolatilityRatioAdaptiveRsxBatchResult, CudaVolatilityRatioAdaptiveRsxError>
    {
        if data.is_empty() {
            return Err(CudaVolatilityRatioAdaptiveRsxError::InvalidInput(
                "empty input".into(),
            ));
        }
        let first = Self::first_valid_value(data).ok_or_else(|| {
            CudaVolatilityRatioAdaptiveRsxError::InvalidInput("all values are NaN".into())
        })?;

        let combos = expand_grid_volatility_ratio_adaptive_rsx(sweep)
            .map_err(|err| CudaVolatilityRatioAdaptiveRsxError::InvalidInput(err.to_string()))?;
        if combos.is_empty() {
            return Err(CudaVolatilityRatioAdaptiveRsxError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let rows = combos.len();
        let cols = data.len();
        let valid = cols - first;
        let mut periods = Vec::with_capacity(rows);
        let mut speeds = Vec::with_capacity(rows);

        for combo in &combos {
            let period = combo.period.unwrap_or(DEFAULT_PERIOD);
            let speed = combo.speed.unwrap_or(DEFAULT_SPEED);
            if period == 0 || period > cols {
                return Err(CudaVolatilityRatioAdaptiveRsxError::InvalidInput(format!(
                    "invalid period: {period}"
                )));
            }
            if !speed.is_finite() || !(0.0..=1.0).contains(&speed) {
                return Err(CudaVolatilityRatioAdaptiveRsxError::InvalidInput(format!(
                    "invalid speed: {speed}"
                )));
            }
            let needed = (2 * period).saturating_sub(1);
            if valid < needed {
                return Err(CudaVolatilityRatioAdaptiveRsxError::InvalidInput(format!(
                    "not enough valid data: needed={needed}, valid={valid}"
                )));
            }
            periods.push(period as i32);
            speeds.push(speed);
        }

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaVolatilityRatioAdaptiveRsxError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = rows
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| value.checked_add(rows.checked_mul(std::mem::size_of::<f64>())?))
            .ok_or_else(|| {
                CudaVolatilityRatioAdaptiveRsxError::InvalidInput("params bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaVolatilityRatioAdaptiveRsxError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                CudaVolatilityRatioAdaptiveRsxError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaVolatilityRatioAdaptiveRsxError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_data = DeviceBuffer::from_slice(data)?;
        let d_periods = DeviceBuffer::from_slice(&periods)?;
        let d_speeds = DeviceBuffer::from_slice(&speeds)?;
        let d_out_line = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_signal = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("volatility_ratio_adaptive_rsx_batch_f64")
            .map_err(
                |_| CudaVolatilityRatioAdaptiveRsxError::MissingKernelSymbol {
                    name: "volatility_ratio_adaptive_rsx_batch_f64",
                },
            )?;
        let grid_x = ((rows as u32) + VOLATILITY_RATIO_ADAPTIVE_RSX_BLOCK_X - 1)
            / VOLATILITY_RATIO_ADAPTIVE_RSX_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(VOLATILITY_RATIO_ADAPTIVE_RSX_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_data.as_device_ptr(),
                cols as i32,
                d_periods.as_device_ptr(),
                d_speeds.as_device_ptr(),
                rows as i32,
                d_out_line.as_device_ptr(),
                d_out_signal.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaVolatilityRatioAdaptiveRsxBatchResult {
            outputs: VolatilityRatioAdaptiveRsxDeviceArrayF64Pair {
                line: VolatilityRatioAdaptiveRsxDeviceArrayF64 {
                    buf: d_out_line,
                    rows,
                    cols,
                },
                signal: VolatilityRatioAdaptiveRsxDeviceArrayF64 {
                    buf: d_out_signal,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
