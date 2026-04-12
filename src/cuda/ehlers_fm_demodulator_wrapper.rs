#![cfg(feature = "cuda")]

use crate::indicators::ehlers_fm_demodulator::{
    EhlersFmDemodulatorBatchRange, EhlersFmDemodulatorParams,
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

const EHLERS_FM_DEMODULATOR_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaEhlersFmDemodulatorError {
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

pub struct EhlersFmDemodulatorDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl EhlersFmDemodulatorDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct CudaEhlersFmDemodulatorBatchResult {
    pub outputs: EhlersFmDemodulatorDeviceArrayF64,
    pub combos: Vec<EhlersFmDemodulatorParams>,
}

pub struct CudaEhlersFmDemodulator {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaEhlersFmDemodulator {
    pub fn new(device_id: usize) -> Result<Self, CudaEhlersFmDemodulatorError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("ehlers_fm_demodulator_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaEhlersFmDemodulatorError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn minimum_valid_length(period: usize) -> usize {
        period.saturating_sub(2).max(1)
    }

    fn axis(
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, CudaEhlersFmDemodulatorError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }

        let mut values = Vec::new();
        if start < end {
            let mut cur = start;
            while cur <= end {
                values.push(cur);
                let next = cur.saturating_add(step);
                if next == cur {
                    break;
                }
                cur = next;
            }
        } else {
            let mut cur = start;
            while cur >= end {
                values.push(cur);
                let next = cur.saturating_sub(step);
                if next == cur {
                    break;
                }
                cur = next;
                if cur == 0 && end > 0 {
                    break;
                }
            }
        }

        if values.is_empty() {
            return Err(CudaEhlersFmDemodulatorError::InvalidInput(format!(
                "invalid range: start={start}, end={end}, step={step}"
            )));
        }
        Ok(values)
    }

    fn expand_grid(
        sweep: &EhlersFmDemodulatorBatchRange,
    ) -> Result<Vec<EhlersFmDemodulatorParams>, CudaEhlersFmDemodulatorError> {
        Ok(Self::axis(sweep.period)?
            .into_iter()
            .map(|period| EhlersFmDemodulatorParams {
                period: Some(period),
            })
            .collect())
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaEhlersFmDemodulatorError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaEhlersFmDemodulatorError::OutOfMemory {
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
    ) -> Result<(), CudaEhlersFmDemodulatorError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaEhlersFmDemodulatorError::LaunchConfigTooLarge {
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
        open: &[f64],
        close: &[f64],
        sweep: &EhlersFmDemodulatorBatchRange,
    ) -> Result<CudaEhlersFmDemodulatorBatchResult, CudaEhlersFmDemodulatorError> {
        if open.is_empty() || close.is_empty() {
            return Err(CudaEhlersFmDemodulatorError::InvalidInput(
                "empty input".into(),
            ));
        }
        if open.len() != close.len() {
            return Err(CudaEhlersFmDemodulatorError::InvalidInput(format!(
                "input length mismatch: open={}, close={}",
                open.len(),
                close.len()
            )));
        }

        let combos = Self::expand_grid(sweep)?;
        let len = open.len();
        let first_valid = (0..len)
            .find(|&i| !open[i].is_nan() && !close[i].is_nan())
            .ok_or_else(|| {
                CudaEhlersFmDemodulatorError::InvalidInput("all values are NaN".into())
            })?;
        let valid = len - first_valid;
        let max_period = combos
            .iter()
            .map(|combo| combo.period.unwrap_or(30))
            .max()
            .unwrap_or(30);
        if max_period == 0 || max_period > len {
            return Err(CudaEhlersFmDemodulatorError::InvalidInput(format!(
                "invalid period: period={max_period}, data_len={len}"
            )));
        }
        let needed = Self::minimum_valid_length(max_period);
        if valid < needed {
            return Err(CudaEhlersFmDemodulatorError::InvalidInput(format!(
                "not enough valid data: needed={needed}, valid={valid}"
            )));
        }

        let rows = combos.len();
        let cols = len;
        let periods: Vec<i32> = combos
            .iter()
            .map(|combo| combo.period.unwrap_or(30) as i32)
            .collect();
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaEhlersFmDemodulatorError::InvalidInput("rows*cols overflow".into())
        })?;
        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                CudaEhlersFmDemodulatorError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = periods
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| {
                CudaEhlersFmDemodulatorError::InvalidInput("params bytes overflow".into())
            })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaEhlersFmDemodulatorError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaEhlersFmDemodulatorError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_open = DeviceBuffer::from_slice(open)?;
        let d_close = DeviceBuffer::from_slice(close)?;
        let d_periods = DeviceBuffer::from_slice(&periods)?;
        let d_out = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("ehlers_fm_demodulator_batch_f64")
            .map_err(|_| CudaEhlersFmDemodulatorError::MissingKernelSymbol {
                name: "ehlers_fm_demodulator_batch_f64",
            })?;
        let grid_x =
            ((rows as u32) + EHLERS_FM_DEMODULATOR_BLOCK_X - 1) / EHLERS_FM_DEMODULATOR_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(EHLERS_FM_DEMODULATOR_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_open.as_device_ptr(),
                d_close.as_device_ptr(),
                cols as i32,
                d_periods.as_device_ptr(),
                rows as i32,
                d_out.as_device_ptr()
            ))?;
        }

        Ok(CudaEhlersFmDemodulatorBatchResult {
            outputs: EhlersFmDemodulatorDeviceArrayF64 {
                buf: d_out,
                rows,
                cols,
            },
            combos,
        })
    }
}
