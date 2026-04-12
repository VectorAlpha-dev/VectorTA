#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::historical_volatility::{
    HistoricalVolatilityBatchRange, HistoricalVolatilityParams,
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

const HISTORICAL_VOLATILITY_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaHistoricalVolatilityError {
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

pub struct CudaHistoricalVolatilityBatchResult {
    pub outputs: DeviceArrayF32,
    pub combos: Vec<HistoricalVolatilityParams>,
}

pub struct CudaHistoricalVolatility {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaHistoricalVolatility {
    pub fn new(device_id: usize) -> Result<Self, CudaHistoricalVolatilityError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("historical_volatility_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaHistoricalVolatilityError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn axis_usize(
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, CudaHistoricalVolatilityError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }

        let mut out = Vec::new();
        if start < end {
            let mut x = start;
            while x <= end {
                out.push(x);
                let next = x.saturating_add(step);
                if next == x {
                    break;
                }
                x = next;
            }
        } else {
            let mut x = start;
            loop {
                out.push(x);
                if x == end {
                    break;
                }
                let next = x.saturating_sub(step);
                if next == x || next < end {
                    break;
                }
                x = next;
            }
        }

        if out.is_empty() {
            return Err(CudaHistoricalVolatilityError::InvalidInput(format!(
                "invalid lookback range: start={start}, end={end}, step={step}"
            )));
        }
        Ok(out)
    }

    fn axis_f32(
        (start, end, step): (f64, f64, f64),
    ) -> Result<Vec<f32>, CudaHistoricalVolatilityError> {
        if !start.is_finite() || !end.is_finite() || !step.is_finite() {
            return Err(CudaHistoricalVolatilityError::InvalidInput(format!(
                "invalid annualization range: start={start}, end={end}, step={step}"
            )));
        }
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return Ok(vec![start as f32]);
        }

        let mut out = Vec::new();
        if start < end {
            let delta = step.abs();
            let mut x = start;
            while x <= end + 1e-12 {
                out.push(x as f32);
                x += delta;
            }
        } else {
            let delta = -step.abs();
            let mut x = start;
            while x >= end - 1e-12 {
                out.push(x as f32);
                x += delta;
            }
        }

        if out.is_empty() {
            return Err(CudaHistoricalVolatilityError::InvalidInput(format!(
                "invalid annualization range: start={start}, end={end}, step={step}"
            )));
        }
        Ok(out)
    }

    fn expand_grid(
        sweep: &HistoricalVolatilityBatchRange,
    ) -> Result<Vec<HistoricalVolatilityParams>, CudaHistoricalVolatilityError> {
        let lookbacks = Self::axis_usize(sweep.lookback)?;
        let annualization_days = Self::axis_f32(sweep.annualization_days)?;
        let mut combos =
            Vec::with_capacity(lookbacks.len().saturating_mul(annualization_days.len()));
        for lookback in lookbacks {
            if lookback == 0 {
                return Err(CudaHistoricalVolatilityError::InvalidInput(
                    "lookback must be > 0".into(),
                ));
            }
            for annualization in &annualization_days {
                if !annualization.is_finite() || *annualization <= 0.0 {
                    return Err(CudaHistoricalVolatilityError::InvalidInput(format!(
                        "invalid annualization_days {annualization}"
                    )));
                }
                combos.push(HistoricalVolatilityParams {
                    lookback: Some(lookback),
                    annualization_days: Some(*annualization as f64),
                });
            }
        }
        Ok(combos)
    }

    fn count_valid_returns(data: &[f32]) -> usize {
        let mut count = 0usize;
        for i in 1..data.len() {
            let prev = data[i - 1];
            let curr = data[i];
            if prev.is_finite() && curr.is_finite() && prev != 0.0 {
                count += 1;
            }
        }
        count
    }

    fn first_valid_return(data: &[f32]) -> Option<usize> {
        (1..data.len()).find(|&i| {
            let prev = data[i - 1];
            let curr = data[i];
            prev.is_finite() && curr.is_finite() && prev != 0.0
        })
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaHistoricalVolatilityError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaHistoricalVolatilityError::OutOfMemory {
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
    ) -> Result<(), CudaHistoricalVolatilityError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaHistoricalVolatilityError::LaunchConfigTooLarge {
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
        data_f32: &[f32],
        sweep: &HistoricalVolatilityBatchRange,
    ) -> Result<CudaHistoricalVolatilityBatchResult, CudaHistoricalVolatilityError> {
        if data_f32.is_empty() {
            return Err(CudaHistoricalVolatilityError::InvalidInput(
                "empty data".into(),
            ));
        }
        Self::first_valid_return(data_f32).ok_or_else(|| {
            CudaHistoricalVolatilityError::InvalidInput("all values are NaN or invalid".into())
        })?;

        let combos = Self::expand_grid(sweep)?;
        let max_lookback = combos
            .iter()
            .map(|combo| combo.lookback.unwrap_or(20))
            .max()
            .unwrap_or(0);
        let valid = Self::count_valid_returns(data_f32);
        if max_lookback == 0 || valid < max_lookback {
            return Err(CudaHistoricalVolatilityError::InvalidInput(format!(
                "not enough valid returns: needed={max_lookback}, valid={valid}"
            )));
        }

        let rows = combos.len();
        let cols = data_f32.len();
        let lookbacks: Vec<i32> = combos
            .iter()
            .map(|combo| combo.lookback.unwrap_or(20) as i32)
            .collect();
        let annualization_scales: Vec<f32> = combos
            .iter()
            .map(|combo| combo.annualization_days.unwrap_or(250.0).sqrt() as f32)
            .collect();

        let input_bytes = data_f32
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                CudaHistoricalVolatilityError::InvalidInput("input bytes overflow".into())
            })?;
        let lookback_bytes = lookbacks
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| {
                CudaHistoricalVolatilityError::InvalidInput("lookback bytes overflow".into())
            })?;
        let annualization_bytes = annualization_scales
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                CudaHistoricalVolatilityError::InvalidInput("annualization bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaHistoricalVolatilityError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                CudaHistoricalVolatilityError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(lookback_bytes)
            .and_then(|v| v.checked_add(annualization_bytes))
            .and_then(|v| v.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaHistoricalVolatilityError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_data = DeviceBuffer::from_slice(data_f32)?;
        let d_lookbacks = DeviceBuffer::from_slice(&lookbacks)?;
        let d_annualization_scales = DeviceBuffer::from_slice(&annualization_scales)?;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("historical_volatility_batch_f32")
            .map_err(|_| CudaHistoricalVolatilityError::MissingKernelSymbol {
                name: "historical_volatility_batch_f32",
            })?;
        let grid_x =
            ((rows as u32) + HISTORICAL_VOLATILITY_BLOCK_X - 1) / HISTORICAL_VOLATILITY_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(HISTORICAL_VOLATILITY_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_data.as_device_ptr(),
                cols as i32,
                d_lookbacks.as_device_ptr(),
                d_annualization_scales.as_device_ptr(),
                rows as i32,
                d_out.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaHistoricalVolatilityBatchResult {
            outputs: DeviceArrayF32 {
                buf: d_out,
                rows,
                cols,
            },
            combos,
        })
    }
}
