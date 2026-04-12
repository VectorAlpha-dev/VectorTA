#![cfg(feature = "cuda")]

use crate::indicators::on_balance_volume_oscillator::{
    OnBalanceVolumeOscillatorBatchRange, OnBalanceVolumeOscillatorParams,
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

const ON_BALANCE_VOLUME_OSCILLATOR_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaOnBalanceVolumeOscillatorError {
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

pub struct OnBalanceVolumeOscillatorDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl OnBalanceVolumeOscillatorDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct OnBalanceVolumeOscillatorDeviceArrayF64Pair {
    pub line: OnBalanceVolumeOscillatorDeviceArrayF64,
    pub signal: OnBalanceVolumeOscillatorDeviceArrayF64,
}

impl OnBalanceVolumeOscillatorDeviceArrayF64Pair {
    #[inline]
    pub fn rows(&self) -> usize {
        self.line.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.line.cols
    }
}

pub struct CudaOnBalanceVolumeOscillatorBatchResult {
    pub outputs: OnBalanceVolumeOscillatorDeviceArrayF64Pair,
    pub combos: Vec<OnBalanceVolumeOscillatorParams>,
}

pub struct CudaOnBalanceVolumeOscillator {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaOnBalanceVolumeOscillator {
    pub fn new(device_id: usize) -> Result<Self, CudaOnBalanceVolumeOscillatorError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("on_balance_volume_oscillator_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaOnBalanceVolumeOscillatorError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn axis_usize(
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, CudaOnBalanceVolumeOscillatorError> {
        if start == 0 || end == 0 {
            return Err(CudaOnBalanceVolumeOscillatorError::InvalidInput(format!(
                "invalid range: start={start}, end={end}, step={step}"
            )));
        }
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
                if value < end.saturating_add(step) {
                    break;
                }
                value = value.saturating_sub(step);
                if value == 0 {
                    break;
                }
            }
        }

        if out.is_empty() {
            return Err(CudaOnBalanceVolumeOscillatorError::InvalidInput(format!(
                "invalid range: start={start}, end={end}, step={step}"
            )));
        }
        Ok(out)
    }

    fn expand_grid(
        sweep: &OnBalanceVolumeOscillatorBatchRange,
    ) -> Result<Vec<OnBalanceVolumeOscillatorParams>, CudaOnBalanceVolumeOscillatorError> {
        let obv_lengths = Self::axis_usize(sweep.obv_length)?;
        let ema_lengths = Self::axis_usize(sweep.ema_length)?;
        let mut combos = Vec::with_capacity(obv_lengths.len().saturating_mul(ema_lengths.len()));
        for obv_length in obv_lengths {
            for ema_length in ema_lengths.iter().copied() {
                combos.push(OnBalanceVolumeOscillatorParams {
                    obv_length: Some(obv_length),
                    ema_length: Some(ema_length),
                });
            }
        }
        Ok(combos)
    }

    fn first_valid_bar(source: &[f64], volume: &[f64]) -> Option<usize> {
        (0..source.len()).find(|&i| source[i].is_finite() && volume[i].is_finite())
    }

    fn max_valid_run_length(source: &[f64], volume: &[f64]) -> usize {
        let mut best = 0usize;
        let mut run = 0usize;
        for i in 0..source.len() {
            if source[i].is_finite() && volume[i].is_finite() {
                run += 1;
                if run > best {
                    best = run;
                }
            } else {
                run = 0;
            }
        }
        best
    }

    fn will_fit(
        required: usize,
        headroom: usize,
    ) -> Result<(), CudaOnBalanceVolumeOscillatorError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaOnBalanceVolumeOscillatorError::OutOfMemory {
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
    ) -> Result<(), CudaOnBalanceVolumeOscillatorError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaOnBalanceVolumeOscillatorError::LaunchConfigTooLarge {
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
        source: &[f64],
        volume: &[f64],
        sweep: &OnBalanceVolumeOscillatorBatchRange,
    ) -> Result<CudaOnBalanceVolumeOscillatorBatchResult, CudaOnBalanceVolumeOscillatorError> {
        if source.is_empty() {
            return Err(CudaOnBalanceVolumeOscillatorError::InvalidInput(
                "empty data".into(),
            ));
        }
        if source.len() != volume.len() {
            return Err(CudaOnBalanceVolumeOscillatorError::InvalidInput(format!(
                "data length mismatch: source_len={}, volume_len={}",
                source.len(),
                volume.len()
            )));
        }

        let combos = Self::expand_grid(sweep)?;
        let max_obv_length = combos
            .iter()
            .map(|combo| combo.obv_length.unwrap_or(20))
            .max()
            .unwrap_or(0);
        let max_ema_length = combos
            .iter()
            .map(|combo| combo.ema_length.unwrap_or(9))
            .max()
            .unwrap_or(0);
        if max_obv_length == 0 || max_obv_length > source.len() {
            return Err(CudaOnBalanceVolumeOscillatorError::InvalidInput(format!(
                "invalid OBV length: obv_length={max_obv_length}, data_len={}",
                source.len()
            )));
        }
        if max_ema_length == 0 {
            return Err(CudaOnBalanceVolumeOscillatorError::InvalidInput(
                "invalid EMA length: ema_length=0".into(),
            ));
        }
        Self::first_valid_bar(source, volume).ok_or_else(|| {
            CudaOnBalanceVolumeOscillatorError::InvalidInput("all values are NaN".into())
        })?;
        let valid = Self::max_valid_run_length(source, volume);
        if valid < max_obv_length {
            return Err(CudaOnBalanceVolumeOscillatorError::InvalidInput(format!(
                "not enough valid data: needed={max_obv_length}, valid={valid}"
            )));
        }

        let rows = combos.len();
        let cols = source.len();
        let obv_lengths: Vec<i32> = combos
            .iter()
            .map(|combo| combo.obv_length.unwrap_or(20) as i32)
            .collect();
        let ema_lengths: Vec<i32> = combos
            .iter()
            .map(|combo| combo.ema_length.unwrap_or(9) as i32)
            .collect();

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|v| v.checked_mul(2))
            .ok_or_else(|| {
                CudaOnBalanceVolumeOscillatorError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = obv_lengths
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|v| {
                ema_lengths
                    .len()
                    .checked_mul(std::mem::size_of::<i32>())
                    .and_then(|w| v.checked_add(w))
            })
            .ok_or_else(|| {
                CudaOnBalanceVolumeOscillatorError::InvalidInput("params bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaOnBalanceVolumeOscillatorError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|v| v.checked_mul(2))
            .ok_or_else(|| {
                CudaOnBalanceVolumeOscillatorError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|v| v.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaOnBalanceVolumeOscillatorError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_source = DeviceBuffer::from_slice(source)?;
        let d_volume = DeviceBuffer::from_slice(volume)?;
        let d_obv_lengths = DeviceBuffer::from_slice(&obv_lengths)?;
        let d_ema_lengths = DeviceBuffer::from_slice(&ema_lengths)?;
        let mut d_line = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_signal = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("on_balance_volume_oscillator_batch_f64")
            .map_err(
                |_| CudaOnBalanceVolumeOscillatorError::MissingKernelSymbol {
                    name: "on_balance_volume_oscillator_batch_f64",
                },
            )?;
        let grid_x = ((rows as u32) + ON_BALANCE_VOLUME_OSCILLATOR_BLOCK_X - 1)
            / ON_BALANCE_VOLUME_OSCILLATOR_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(ON_BALANCE_VOLUME_OSCILLATOR_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_source.as_device_ptr(),
                d_volume.as_device_ptr(),
                cols as i32,
                d_obv_lengths.as_device_ptr(),
                d_ema_lengths.as_device_ptr(),
                rows as i32,
                d_line.as_device_ptr(),
                d_signal.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaOnBalanceVolumeOscillatorBatchResult {
            outputs: OnBalanceVolumeOscillatorDeviceArrayF64Pair {
                line: OnBalanceVolumeOscillatorDeviceArrayF64 {
                    buf: d_line,
                    rows,
                    cols,
                },
                signal: OnBalanceVolumeOscillatorDeviceArrayF64 {
                    buf: d_signal,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
