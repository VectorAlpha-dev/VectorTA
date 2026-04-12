#![cfg(feature = "cuda")]

use crate::indicators::ewma_volatility::{EwmaVolatilityBatchRange, EwmaVolatilityParams};
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

const EWMA_VOLATILITY_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaEwmaVolatilityError {
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

pub struct DeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl DeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct CudaEwmaVolatilityBatchResult {
    pub outputs: DeviceArrayF64,
    pub combos: Vec<EwmaVolatilityParams>,
}

pub struct CudaEwmaVolatility {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaEwmaVolatility {
    pub fn new(device_id: usize) -> Result<Self, CudaEwmaVolatilityError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("ewma_volatility_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaEwmaVolatilityError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn axis_f64((start, end, step): (f64, f64, f64)) -> Result<Vec<f64>, CudaEwmaVolatilityError> {
        if !start.is_finite() || !end.is_finite() || !step.is_finite() {
            return Err(CudaEwmaVolatilityError::InvalidInput(format!(
                "invalid lambda range: start={start}, end={end}, step={step}"
            )));
        }
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return Ok(vec![start]);
        }

        let mut out = Vec::new();
        let delta = step.abs();
        if start < end {
            let mut value = start;
            while value <= end + 1e-12 {
                out.push(value);
                value += delta;
            }
        } else {
            let mut value = start;
            while value >= end - 1e-12 {
                out.push(value);
                value -= delta;
            }
        }

        if out.is_empty() {
            return Err(CudaEwmaVolatilityError::InvalidInput(format!(
                "invalid lambda range: start={start}, end={end}, step={step}"
            )));
        }
        Ok(out)
    }

    fn expand_grid(
        sweep: &EwmaVolatilityBatchRange,
    ) -> Result<Vec<EwmaVolatilityParams>, CudaEwmaVolatilityError> {
        Ok(Self::axis_f64(sweep.lambda)?
            .into_iter()
            .map(|lambda| EwmaVolatilityParams {
                lambda: Some(lambda),
            })
            .collect())
    }

    fn period_from_lambda(lambda: f64) -> Result<usize, CudaEwmaVolatilityError> {
        if !lambda.is_finite() || !(0.0..1.0).contains(&lambda) {
            return Err(CudaEwmaVolatilityError::InvalidInput(format!(
                "invalid lambda: {lambda}"
            )));
        }
        Ok(((2.0 / (1.0 - lambda)) - 1.0).round().max(1.0) as usize)
    }

    fn alpha_from_period(period: usize) -> f64 {
        2.0 / (period as f64 + 1.0)
    }

    fn count_valid_returns(data: &[f64]) -> usize {
        let mut count = 0usize;
        for i in 1..data.len() {
            let prev = data[i - 1];
            let curr = data[i];
            if prev.is_finite() && curr.is_finite() && prev > 0.0 && curr > 0.0 {
                count += 1;
            }
        }
        count
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaEwmaVolatilityError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaEwmaVolatilityError::OutOfMemory {
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
    ) -> Result<(), CudaEwmaVolatilityError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaEwmaVolatilityError::LaunchConfigTooLarge {
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
        sweep: &EwmaVolatilityBatchRange,
    ) -> Result<CudaEwmaVolatilityBatchResult, CudaEwmaVolatilityError> {
        if data.is_empty() {
            return Err(CudaEwmaVolatilityError::InvalidInput("empty data".into()));
        }
        if !data.iter().any(|value| !value.is_nan()) {
            return Err(CudaEwmaVolatilityError::InvalidInput(
                "all values are NaN".into(),
            ));
        }

        let combos = Self::expand_grid(sweep)?;
        let periods: Vec<i32> = combos
            .iter()
            .map(|combo| Self::period_from_lambda(combo.lambda.unwrap_or(0.94)))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .map(|period| period as i32)
            .collect();
        let alphas: Vec<f64> = periods
            .iter()
            .map(|&period| Self::alpha_from_period(period as usize))
            .collect();

        let valid_returns = Self::count_valid_returns(data);
        let max_period = periods.iter().copied().max().unwrap_or(0) as usize;
        if valid_returns < max_period {
            return Err(CudaEwmaVolatilityError::InvalidInput(format!(
                "not enough valid data: needed={max_period}, valid={valid_returns}"
            )));
        }

        let rows = combos.len();
        let cols = data.len();
        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| CudaEwmaVolatilityError::InvalidInput("input bytes overflow".into()))?;
        let period_bytes = periods
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaEwmaVolatilityError::InvalidInput("period bytes overflow".into()))?;
        let alpha_bytes = alphas
            .len()
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| CudaEwmaVolatilityError::InvalidInput("alpha bytes overflow".into()))?;
        let output_elems = rows
            .checked_mul(cols)
            .ok_or_else(|| CudaEwmaVolatilityError::InvalidInput("rows*cols overflow".into()))?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| CudaEwmaVolatilityError::InvalidInput("output bytes overflow".into()))?;
        let required = input_bytes
            .checked_add(period_bytes)
            .and_then(|v| v.checked_add(alpha_bytes))
            .and_then(|v| v.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaEwmaVolatilityError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_data = DeviceBuffer::from_slice(data)?;
        let d_periods = DeviceBuffer::from_slice(&periods)?;
        let d_alphas = DeviceBuffer::from_slice(&alphas)?;
        let mut d_out = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("ewma_volatility_batch_f64")
            .map_err(|_| CudaEwmaVolatilityError::MissingKernelSymbol {
                name: "ewma_volatility_batch_f64",
            })?;
        let grid_x = ((rows as u32) + EWMA_VOLATILITY_BLOCK_X - 1) / EWMA_VOLATILITY_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(EWMA_VOLATILITY_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_data.as_device_ptr(),
                cols as i32,
                d_periods.as_device_ptr(),
                d_alphas.as_device_ptr(),
                rows as i32,
                d_out.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaEwmaVolatilityBatchResult {
            outputs: DeviceArrayF64 {
                buf: d_out,
                rows,
                cols,
            },
            combos,
        })
    }
}
