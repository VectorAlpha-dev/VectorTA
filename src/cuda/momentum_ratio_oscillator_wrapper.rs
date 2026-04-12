#![cfg(feature = "cuda")]

use crate::indicators::momentum_ratio_oscillator::{
    MomentumRatioOscillatorBatchRange, MomentumRatioOscillatorParams,
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

const MOMENTUM_RATIO_OSCILLATOR_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaMomentumRatioOscillatorError {
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

pub struct MomentumRatioOscillatorDeviceArrayF64Pair {
    pub line: DeviceArrayF64,
    pub signal: DeviceArrayF64,
}

impl MomentumRatioOscillatorDeviceArrayF64Pair {
    #[inline]
    pub fn rows(&self) -> usize {
        self.line.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.line.cols
    }
}

pub struct CudaMomentumRatioOscillatorBatchResult {
    pub outputs: MomentumRatioOscillatorDeviceArrayF64Pair,
    pub combos: Vec<MomentumRatioOscillatorParams>,
}

pub struct CudaMomentumRatioOscillator {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaMomentumRatioOscillator {
    pub fn new(device_id: usize) -> Result<Self, CudaMomentumRatioOscillatorError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("momentum_ratio_oscillator_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaMomentumRatioOscillatorError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn axis_usize(
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, CudaMomentumRatioOscillatorError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }

        let mut out = Vec::new();
        if start < end {
            let mut value = start;
            while value <= end {
                out.push(value);
                match value.checked_add(step) {
                    Some(next) if next > value => value = next,
                    _ => break,
                }
            }
        } else {
            let mut value = start;
            while value >= end {
                out.push(value);
                if value < end + step {
                    break;
                }
                value = value.saturating_sub(step);
                if value == 0 {
                    break;
                }
            }
        }

        if out.is_empty() {
            return Err(CudaMomentumRatioOscillatorError::InvalidInput(format!(
                "invalid period range: start={start}, end={end}, step={step}"
            )));
        }
        Ok(out)
    }

    fn expand_grid(
        sweep: &MomentumRatioOscillatorBatchRange,
    ) -> Result<Vec<MomentumRatioOscillatorParams>, CudaMomentumRatioOscillatorError> {
        Ok(Self::axis_usize(sweep.period)?
            .into_iter()
            .map(|period| MomentumRatioOscillatorParams {
                period: Some(period),
            })
            .collect())
    }

    fn first_valid_source(data: &[f64]) -> Option<usize> {
        data.iter().position(|value| value.is_finite())
    }

    fn count_valid_from(data: &[f64], start: usize) -> usize {
        data[start..]
            .iter()
            .filter(|value| value.is_finite())
            .count()
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaMomentumRatioOscillatorError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaMomentumRatioOscillatorError::OutOfMemory {
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
    ) -> Result<(), CudaMomentumRatioOscillatorError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaMomentumRatioOscillatorError::LaunchConfigTooLarge {
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
        sweep: &MomentumRatioOscillatorBatchRange,
    ) -> Result<CudaMomentumRatioOscillatorBatchResult, CudaMomentumRatioOscillatorError> {
        if data.is_empty() {
            return Err(CudaMomentumRatioOscillatorError::InvalidInput(
                "empty data".into(),
            ));
        }

        let first = Self::first_valid_source(data).ok_or_else(|| {
            CudaMomentumRatioOscillatorError::InvalidInput("all values are NaN".into())
        })?;
        let valid = Self::count_valid_from(data, first);
        if valid < 2 {
            return Err(CudaMomentumRatioOscillatorError::InvalidInput(format!(
                "not enough valid data: needed=2, valid={valid}"
            )));
        }

        let combos = Self::expand_grid(sweep)?;
        let max_period = combos
            .iter()
            .map(|params| params.period.unwrap_or(50))
            .max()
            .unwrap_or(0);
        if max_period == 0 || max_period > data.len() {
            return Err(CudaMomentumRatioOscillatorError::InvalidInput(format!(
                "invalid period: period={max_period}, data_len={}",
                data.len()
            )));
        }

        let rows = combos.len();
        let cols = data.len();
        let periods: Vec<i32> = combos
            .iter()
            .map(|params| params.period.unwrap_or(50) as i32)
            .collect();

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaMomentumRatioOscillatorError::InvalidInput("input bytes overflow".into())
            })?;
        let period_bytes = periods
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| {
                CudaMomentumRatioOscillatorError::InvalidInput("period bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaMomentumRatioOscillatorError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|v| v.checked_mul(2))
            .ok_or_else(|| {
                CudaMomentumRatioOscillatorError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(period_bytes)
            .and_then(|v| v.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaMomentumRatioOscillatorError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_data = DeviceBuffer::from_slice(data)?;
        let d_periods = DeviceBuffer::from_slice(&periods)?;
        let mut d_line = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_signal = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("momentum_ratio_oscillator_batch_f64")
            .map_err(|_| CudaMomentumRatioOscillatorError::MissingKernelSymbol {
                name: "momentum_ratio_oscillator_batch_f64",
            })?;
        let grid_x = ((rows as u32) + MOMENTUM_RATIO_OSCILLATOR_BLOCK_X - 1)
            / MOMENTUM_RATIO_OSCILLATOR_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(MOMENTUM_RATIO_OSCILLATOR_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_data.as_device_ptr(),
                cols as i32,
                d_periods.as_device_ptr(),
                rows as i32,
                d_line.as_device_ptr(),
                d_signal.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaMomentumRatioOscillatorBatchResult {
            outputs: MomentumRatioOscillatorDeviceArrayF64Pair {
                line: DeviceArrayF64 {
                    buf: d_line,
                    rows,
                    cols,
                },
                signal: DeviceArrayF64 {
                    buf: d_signal,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
