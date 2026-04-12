#![cfg(feature = "cuda")]

use crate::indicators::historical_volatility_percentile::{
    HistoricalVolatilityPercentileBatchRange, HistoricalVolatilityPercentileParams,
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

const HISTORICAL_VOLATILITY_PERCENTILE_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaHistoricalVolatilityPercentileError {
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

pub struct HistoricalVolatilityPercentileDeviceArrayF64Pair {
    pub hvp: DeviceArrayF64,
    pub hvp_sma: DeviceArrayF64,
}

impl HistoricalVolatilityPercentileDeviceArrayF64Pair {
    #[inline]
    pub fn rows(&self) -> usize {
        self.hvp.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.hvp.cols
    }
}

pub struct CudaHistoricalVolatilityPercentileBatchResult {
    pub outputs: HistoricalVolatilityPercentileDeviceArrayF64Pair,
    pub combos: Vec<HistoricalVolatilityPercentileParams>,
}

pub struct CudaHistoricalVolatilityPercentile {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaHistoricalVolatilityPercentile {
    pub fn new(device_id: usize) -> Result<Self, CudaHistoricalVolatilityPercentileError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("historical_volatility_percentile_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaHistoricalVolatilityPercentileError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn axis(
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, CudaHistoricalVolatilityPercentileError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }

        let mut out = Vec::new();
        if start <= end {
            let mut value = start;
            while value <= end {
                out.push(value);
                match value.checked_add(step.max(1)) {
                    Some(next) if next > value => value = next,
                    _ => break,
                }
            }
        } else {
            let mut value = start;
            loop {
                out.push(value);
                if value == end {
                    break;
                }
                let next = value.saturating_sub(step.max(1));
                if next == value || next < end {
                    break;
                }
                value = next;
            }
        }

        if out.is_empty() {
            return Err(CudaHistoricalVolatilityPercentileError::InvalidInput(
                format!("invalid range: start={start}, end={end}, step={step}"),
            ));
        }
        Ok(out)
    }

    fn expand_grid(
        sweep: &HistoricalVolatilityPercentileBatchRange,
    ) -> Result<Vec<HistoricalVolatilityPercentileParams>, CudaHistoricalVolatilityPercentileError>
    {
        let lengths = Self::axis(sweep.length)?;
        let annual_lengths = Self::axis(sweep.annual_length)?;
        let mut combos = Vec::with_capacity(lengths.len().saturating_mul(annual_lengths.len()));
        for length in lengths {
            if length < 2 {
                return Err(CudaHistoricalVolatilityPercentileError::InvalidInput(
                    "length must be >= 2".into(),
                ));
            }
            for annual_length in &annual_lengths {
                if *annual_length == 0 {
                    return Err(CudaHistoricalVolatilityPercentileError::InvalidInput(
                        "annual_length must be > 0".into(),
                    ));
                }
                combos.push(HistoricalVolatilityPercentileParams {
                    length: Some(length),
                    annual_length: Some(*annual_length),
                });
            }
        }
        Ok(combos)
    }

    fn first_valid_source(data: &[f64]) -> Option<usize> {
        data.iter()
            .position(|&value| value.is_finite() && value > 0.0)
    }

    fn will_fit(
        required: usize,
        headroom: usize,
    ) -> Result<(), CudaHistoricalVolatilityPercentileError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaHistoricalVolatilityPercentileError::OutOfMemory {
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
    ) -> Result<(), CudaHistoricalVolatilityPercentileError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(
                CudaHistoricalVolatilityPercentileError::LaunchConfigTooLarge {
                    gx: grid.x,
                    gy: grid.y,
                    gz: grid.z,
                    bx: block.x,
                    by: block.y,
                    bz: block.z,
                },
            );
        }
        Ok(())
    }

    pub fn batch_dev(
        &self,
        data: &[f64],
        sweep: &HistoricalVolatilityPercentileBatchRange,
    ) -> Result<
        CudaHistoricalVolatilityPercentileBatchResult,
        CudaHistoricalVolatilityPercentileError,
    > {
        if data.is_empty() {
            return Err(CudaHistoricalVolatilityPercentileError::InvalidInput(
                "empty data".into(),
            ));
        }

        let first = Self::first_valid_source(data).ok_or_else(|| {
            CudaHistoricalVolatilityPercentileError::InvalidInput("all values are invalid".into())
        })?;
        let combos = Self::expand_grid(sweep)?;
        let max_needed = combos
            .iter()
            .map(|combo| combo.length.unwrap_or(21) + combo.annual_length.unwrap_or(252) - 1)
            .max()
            .unwrap_or(0);
        let valid = data.len().saturating_sub(first);
        if valid < max_needed {
            return Err(CudaHistoricalVolatilityPercentileError::InvalidInput(
                format!("not enough valid data: needed={max_needed}, valid={valid}"),
            ));
        }

        let rows = combos.len();
        let cols = data.len();
        let lengths: Vec<i32> = combos
            .iter()
            .map(|combo| combo.length.unwrap_or(21) as i32)
            .collect();
        let annual_lengths: Vec<i32> = combos
            .iter()
            .map(|combo| combo.annual_length.unwrap_or(252) as i32)
            .collect();

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaHistoricalVolatilityPercentileError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = lengths
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|v| {
                annual_lengths
                    .len()
                    .checked_mul(std::mem::size_of::<i32>())
                    .and_then(|w| v.checked_add(w))
            })
            .ok_or_else(|| {
                CudaHistoricalVolatilityPercentileError::InvalidInput(
                    "params bytes overflow".into(),
                )
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaHistoricalVolatilityPercentileError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|v| v.checked_mul(2))
            .ok_or_else(|| {
                CudaHistoricalVolatilityPercentileError::InvalidInput(
                    "output bytes overflow".into(),
                )
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|v| v.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaHistoricalVolatilityPercentileError::InvalidInput(
                    "required bytes overflow".into(),
                )
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_data = DeviceBuffer::from_slice(data)?;
        let d_lengths = DeviceBuffer::from_slice(&lengths)?;
        let d_annual_lengths = DeviceBuffer::from_slice(&annual_lengths)?;
        let mut d_hvp = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_hvp_sma = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("historical_volatility_percentile_batch_f64")
            .map_err(
                |_| CudaHistoricalVolatilityPercentileError::MissingKernelSymbol {
                    name: "historical_volatility_percentile_batch_f64",
                },
            )?;
        let grid_x = ((rows as u32) + HISTORICAL_VOLATILITY_PERCENTILE_BLOCK_X - 1)
            / HISTORICAL_VOLATILITY_PERCENTILE_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(HISTORICAL_VOLATILITY_PERCENTILE_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_data.as_device_ptr(),
                cols as i32,
                d_lengths.as_device_ptr(),
                d_annual_lengths.as_device_ptr(),
                rows as i32,
                d_hvp.as_device_ptr(),
                d_hvp_sma.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaHistoricalVolatilityPercentileBatchResult {
            outputs: HistoricalVolatilityPercentileDeviceArrayF64Pair {
                hvp: DeviceArrayF64 {
                    buf: d_hvp,
                    rows,
                    cols,
                },
                hvp_sma: DeviceArrayF64 {
                    buf: d_hvp_sma,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
