#![cfg(feature = "cuda")]

use crate::indicators::absolute_strength_index_oscillator::{
    AbsoluteStrengthIndexOscillatorBatchRange, AbsoluteStrengthIndexOscillatorParams,
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

const ABSOLUTE_STRENGTH_INDEX_OSCILLATOR_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const DEFAULT_EMA_LENGTH: usize = 21;
const DEFAULT_SIGNAL_LENGTH: usize = 34;

#[derive(Debug, Error)]
pub enum CudaAbsoluteStrengthIndexOscillatorError {
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

pub struct AbsoluteStrengthIndexOscillatorDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl AbsoluteStrengthIndexOscillatorDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct AbsoluteStrengthIndexOscillatorDeviceArrayF64Triple {
    pub oscillator: AbsoluteStrengthIndexOscillatorDeviceArrayF64,
    pub signal: AbsoluteStrengthIndexOscillatorDeviceArrayF64,
    pub histogram: AbsoluteStrengthIndexOscillatorDeviceArrayF64,
}

impl AbsoluteStrengthIndexOscillatorDeviceArrayF64Triple {
    #[inline]
    pub fn rows(&self) -> usize {
        self.oscillator.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.oscillator.cols
    }
}

pub struct CudaAbsoluteStrengthIndexOscillatorBatchResult {
    pub outputs: AbsoluteStrengthIndexOscillatorDeviceArrayF64Triple,
    pub combos: Vec<AbsoluteStrengthIndexOscillatorParams>,
}

pub struct CudaAbsoluteStrengthIndexOscillator {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaAbsoluteStrengthIndexOscillator {
    pub fn new(device_id: usize) -> Result<Self, CudaAbsoluteStrengthIndexOscillatorError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module =
            crate::load_cuda_embedded_module!("absolute_strength_index_oscillator_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaAbsoluteStrengthIndexOscillatorError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn expand_axis(
        start: usize,
        end: usize,
        step: usize,
    ) -> Result<Vec<usize>, CudaAbsoluteStrengthIndexOscillatorError> {
        if start == end {
            return Ok(vec![start]);
        }
        if step == 0 {
            return Err(CudaAbsoluteStrengthIndexOscillatorError::InvalidInput(
                format!("invalid range: start={start}, end={end}, step={step}"),
            ));
        }

        let mut out = Vec::new();
        if start < end {
            let mut current = start;
            while current <= end {
                out.push(current);
                let next = current.saturating_add(step);
                if next == current {
                    break;
                }
                current = next;
            }
        } else {
            let mut current = start;
            while current >= end {
                out.push(current);
                let next = current.saturating_sub(step);
                if next == current {
                    break;
                }
                current = next;
            }
        }

        if out.is_empty() {
            return Err(CudaAbsoluteStrengthIndexOscillatorError::InvalidInput(
                format!("invalid range: start={start}, end={end}, step={step}"),
            ));
        }
        Ok(out)
    }

    fn expand_grid(
        sweep: &AbsoluteStrengthIndexOscillatorBatchRange,
    ) -> Result<Vec<AbsoluteStrengthIndexOscillatorParams>, CudaAbsoluteStrengthIndexOscillatorError>
    {
        let ema_lengths =
            Self::expand_axis(sweep.ema_length.0, sweep.ema_length.1, sweep.ema_length.2)?;
        let signal_lengths = Self::expand_axis(
            sweep.signal_length.0,
            sweep.signal_length.1,
            sweep.signal_length.2,
        )?;

        let mut combos = Vec::with_capacity(ema_lengths.len().saturating_mul(signal_lengths.len()));
        for ema_length in ema_lengths {
            if ema_length == 0 {
                return Err(CudaAbsoluteStrengthIndexOscillatorError::InvalidInput(
                    format!("invalid ema_length: {ema_length}"),
                ));
            }
            for &signal_length in &signal_lengths {
                if signal_length <= 1 {
                    return Err(CudaAbsoluteStrengthIndexOscillatorError::InvalidInput(
                        format!("invalid signal_length: {signal_length}"),
                    ));
                }
                combos.push(AbsoluteStrengthIndexOscillatorParams {
                    ema_length: Some(ema_length),
                    signal_length: Some(signal_length),
                });
            }
        }
        Ok(combos)
    }

    fn will_fit(
        required: usize,
        headroom: usize,
    ) -> Result<(), CudaAbsoluteStrengthIndexOscillatorError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaAbsoluteStrengthIndexOscillatorError::OutOfMemory {
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
    ) -> Result<(), CudaAbsoluteStrengthIndexOscillatorError> {
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
                CudaAbsoluteStrengthIndexOscillatorError::LaunchConfigTooLarge {
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
        sweep: &AbsoluteStrengthIndexOscillatorBatchRange,
    ) -> Result<
        CudaAbsoluteStrengthIndexOscillatorBatchResult,
        CudaAbsoluteStrengthIndexOscillatorError,
    > {
        if data.is_empty() {
            return Err(CudaAbsoluteStrengthIndexOscillatorError::InvalidInput(
                "empty input".into(),
            ));
        }
        data.iter()
            .position(|value| value.is_finite())
            .ok_or_else(|| {
                CudaAbsoluteStrengthIndexOscillatorError::InvalidInput("all values are NaN".into())
            })?;

        let combos = Self::expand_grid(sweep)?;
        let rows = combos.len();
        let cols = data.len();
        let ema_lengths: Vec<i32> = combos
            .iter()
            .map(|combo| combo.ema_length.unwrap_or(DEFAULT_EMA_LENGTH) as i32)
            .collect();
        let signal_lengths: Vec<i32> = combos
            .iter()
            .map(|combo| combo.signal_length.unwrap_or(DEFAULT_SIGNAL_LENGTH) as i32)
            .collect();

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaAbsoluteStrengthIndexOscillatorError::InvalidInput(
                    "input bytes overflow".into(),
                )
            })?;
        let params_bytes = ema_lengths
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| {
                signal_lengths
                    .len()
                    .checked_mul(std::mem::size_of::<i32>())
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| {
                CudaAbsoluteStrengthIndexOscillatorError::InvalidInput(
                    "params bytes overflow".into(),
                )
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaAbsoluteStrengthIndexOscillatorError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(3))
            .ok_or_else(|| {
                CudaAbsoluteStrengthIndexOscillatorError::InvalidInput(
                    "output bytes overflow".into(),
                )
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaAbsoluteStrengthIndexOscillatorError::InvalidInput(
                    "required bytes overflow".into(),
                )
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_data = DeviceBuffer::from_slice(data)?;
        let d_ema_lengths = DeviceBuffer::from_slice(&ema_lengths)?;
        let d_signal_lengths = DeviceBuffer::from_slice(&signal_lengths)?;
        let d_out_oscillator = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_signal = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_histogram = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("absolute_strength_index_oscillator_batch_f64")
            .map_err(
                |_| CudaAbsoluteStrengthIndexOscillatorError::MissingKernelSymbol {
                    name: "absolute_strength_index_oscillator_batch_f64",
                },
            )?;
        let grid_x = ((rows as u32) + ABSOLUTE_STRENGTH_INDEX_OSCILLATOR_BLOCK_X - 1)
            / ABSOLUTE_STRENGTH_INDEX_OSCILLATOR_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(ABSOLUTE_STRENGTH_INDEX_OSCILLATOR_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_data.as_device_ptr(),
                cols as i32,
                d_ema_lengths.as_device_ptr(),
                d_signal_lengths.as_device_ptr(),
                rows as i32,
                d_out_oscillator.as_device_ptr(),
                d_out_signal.as_device_ptr(),
                d_out_histogram.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaAbsoluteStrengthIndexOscillatorBatchResult {
            outputs: AbsoluteStrengthIndexOscillatorDeviceArrayF64Triple {
                oscillator: AbsoluteStrengthIndexOscillatorDeviceArrayF64 {
                    buf: d_out_oscillator,
                    rows,
                    cols,
                },
                signal: AbsoluteStrengthIndexOscillatorDeviceArrayF64 {
                    buf: d_out_signal,
                    rows,
                    cols,
                },
                histogram: AbsoluteStrengthIndexOscillatorDeviceArrayF64 {
                    buf: d_out_histogram,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
