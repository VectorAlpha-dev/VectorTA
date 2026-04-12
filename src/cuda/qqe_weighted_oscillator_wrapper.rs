#![cfg(feature = "cuda")]

use crate::indicators::qqe_weighted_oscillator::{
    QqeWeightedOscillatorBatchRange, QqeWeightedOscillatorParams,
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

const QQE_WEIGHTED_OSCILLATOR_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const DEFAULT_LENGTH: usize = 14;
const DEFAULT_FACTOR: f64 = 4.236;
const DEFAULT_SMOOTH: usize = 5;
const DEFAULT_WEIGHT: f64 = 2.0;

#[derive(Debug, Error)]
pub enum CudaQqeWeightedOscillatorError {
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

pub struct QqeWeightedOscillatorDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl QqeWeightedOscillatorDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct QqeWeightedOscillatorDeviceArrayF64Pair {
    pub rsi: QqeWeightedOscillatorDeviceArrayF64,
    pub trailing_stop: QqeWeightedOscillatorDeviceArrayF64,
}

impl QqeWeightedOscillatorDeviceArrayF64Pair {
    #[inline]
    pub fn rows(&self) -> usize {
        self.rsi.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.rsi.cols
    }
}

pub struct CudaQqeWeightedOscillatorBatchResult {
    pub outputs: QqeWeightedOscillatorDeviceArrayF64Pair,
    pub combos: Vec<QqeWeightedOscillatorParams>,
}

pub struct CudaQqeWeightedOscillator {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

fn expand_axis_usize(
    start: usize,
    end: usize,
    step: usize,
) -> Result<Vec<usize>, CudaQqeWeightedOscillatorError> {
    if step == 0 || start == end {
        return Ok(vec![start]);
    }

    let mut out = Vec::new();
    if start <= end {
        let mut current = start;
        while current <= end {
            out.push(current);
            match current.checked_add(step) {
                Some(next) => current = next,
                None => break,
            }
        }
    } else {
        let mut current = start;
        while current >= end {
            out.push(current);
            match current.checked_sub(step) {
                Some(next) => current = next,
                None => break,
            }
            if current < end {
                break;
            }
        }
    }

    if out.is_empty() {
        return Err(CudaQqeWeightedOscillatorError::InvalidInput(format!(
            "invalid range: start={start}, end={end}, step={step}"
        )));
    }
    Ok(out)
}

fn expand_axis_f64(
    start: f64,
    end: f64,
    step: f64,
) -> Result<Vec<f64>, CudaQqeWeightedOscillatorError> {
    let eps = 1e-12;
    if !start.is_finite() || !end.is_finite() || !step.is_finite() {
        return Err(CudaQqeWeightedOscillatorError::InvalidInput(format!(
            "invalid range: start={start}, end={end}, step={step}"
        )));
    }
    if step.abs() < eps || (start - end).abs() < eps {
        return Ok(vec![start]);
    }

    let mut out = Vec::new();
    let dir = if end >= start { 1.0 } else { -1.0 };
    let step_eff = dir * step.abs();
    let mut current = start;
    if dir > 0.0 {
        while current <= end + eps {
            out.push(current);
            current += step_eff;
        }
    } else {
        while current >= end - eps {
            out.push(current);
            current += step_eff;
        }
    }

    if out.is_empty() {
        return Err(CudaQqeWeightedOscillatorError::InvalidInput(format!(
            "invalid range: start={start}, end={end}, step={step}"
        )));
    }
    Ok(out)
}

fn expand_grid_qqe_weighted_oscillator(
    range: &QqeWeightedOscillatorBatchRange,
) -> Result<Vec<QqeWeightedOscillatorParams>, CudaQqeWeightedOscillatorError> {
    let lengths = expand_axis_usize(range.length.0, range.length.1, range.length.2)?;
    let factors = expand_axis_f64(range.factor.0, range.factor.1, range.factor.2)?;
    let smooths = expand_axis_usize(range.smooth.0, range.smooth.1, range.smooth.2)?;
    let weights = expand_axis_f64(range.weight.0, range.weight.1, range.weight.2)?;

    let mut combos = Vec::with_capacity(
        lengths
            .len()
            .saturating_mul(factors.len())
            .saturating_mul(smooths.len())
            .saturating_mul(weights.len()),
    );
    for &length in &lengths {
        for &factor in &factors {
            for &smooth in &smooths {
                for &weight in &weights {
                    combos.push(QqeWeightedOscillatorParams {
                        length: Some(length),
                        factor: Some(factor),
                        smooth: Some(smooth),
                        weight: Some(weight),
                    });
                }
            }
        }
    }
    Ok(combos)
}

impl CudaQqeWeightedOscillator {
    pub fn new(device_id: usize) -> Result<Self, CudaQqeWeightedOscillatorError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("qqe_weighted_oscillator_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaQqeWeightedOscillatorError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn count_valid_values(data: &[f64], first: usize) -> usize {
        data[first..]
            .iter()
            .filter(|value| value.is_finite())
            .count()
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaQqeWeightedOscillatorError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaQqeWeightedOscillatorError::OutOfMemory {
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
    ) -> Result<(), CudaQqeWeightedOscillatorError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaQqeWeightedOscillatorError::LaunchConfigTooLarge {
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
        sweep: &QqeWeightedOscillatorBatchRange,
    ) -> Result<CudaQqeWeightedOscillatorBatchResult, CudaQqeWeightedOscillatorError> {
        if data.is_empty() {
            return Err(CudaQqeWeightedOscillatorError::InvalidInput(
                "empty input".into(),
            ));
        }
        let first = data
            .iter()
            .position(|value| value.is_finite())
            .ok_or_else(|| {
                CudaQqeWeightedOscillatorError::InvalidInput("all values are NaN".into())
            })?;
        let valid = Self::count_valid_values(data, first);

        let combos = expand_grid_qqe_weighted_oscillator(sweep)?;
        if combos.is_empty() {
            return Err(CudaQqeWeightedOscillatorError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let rows = combos.len();
        let cols = data.len();
        let mut lengths = Vec::with_capacity(rows);
        let mut factors = Vec::with_capacity(rows);
        let mut smooths = Vec::with_capacity(rows);
        let mut weights = Vec::with_capacity(rows);

        for combo in &combos {
            let length = combo.length.unwrap_or(DEFAULT_LENGTH);
            let factor = combo.factor.unwrap_or(DEFAULT_FACTOR);
            let smooth = combo.smooth.unwrap_or(DEFAULT_SMOOTH);
            let weight = combo.weight.unwrap_or(DEFAULT_WEIGHT);
            if length == 0 || length > cols {
                return Err(CudaQqeWeightedOscillatorError::InvalidInput(format!(
                    "invalid length: length={length}, data_len={cols}"
                )));
            }
            if smooth == 0 {
                return Err(CudaQqeWeightedOscillatorError::InvalidInput(format!(
                    "invalid smooth: smooth={smooth}, data_len={cols}"
                )));
            }
            if !factor.is_finite() || factor < 0.0 {
                return Err(CudaQqeWeightedOscillatorError::InvalidInput(format!(
                    "invalid factor: {factor}"
                )));
            }
            if !weight.is_finite() {
                return Err(CudaQqeWeightedOscillatorError::InvalidInput(format!(
                    "invalid weight: {weight}"
                )));
            }
            let needed = length.checked_add(1).ok_or_else(|| {
                CudaQqeWeightedOscillatorError::InvalidInput("needed bars overflow".into())
            })?;
            if valid < needed {
                return Err(CudaQqeWeightedOscillatorError::InvalidInput(format!(
                    "not enough valid data: needed={needed}, valid={valid}"
                )));
            }
            lengths.push(length as i32);
            factors.push(factor);
            smooths.push(smooth as i32);
            weights.push(weight);
        }

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaQqeWeightedOscillatorError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = rows
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| value.checked_mul(2))
            .and_then(|value| {
                rows.checked_mul(std::mem::size_of::<f64>())
                    .and_then(|other| other.checked_mul(2))
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| {
                CudaQqeWeightedOscillatorError::InvalidInput("params bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaQqeWeightedOscillatorError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                CudaQqeWeightedOscillatorError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaQqeWeightedOscillatorError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_data = DeviceBuffer::from_slice(data)?;
        let d_lengths = DeviceBuffer::from_slice(&lengths)?;
        let d_factors = DeviceBuffer::from_slice(&factors)?;
        let d_smooths = DeviceBuffer::from_slice(&smooths)?;
        let d_weights = DeviceBuffer::from_slice(&weights)?;
        let d_out_rsi = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_ts = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("qqe_weighted_oscillator_batch_f64")
            .map_err(|_| CudaQqeWeightedOscillatorError::MissingKernelSymbol {
                name: "qqe_weighted_oscillator_batch_f64",
            })?;
        let grid_x =
            ((rows as u32) + QQE_WEIGHTED_OSCILLATOR_BLOCK_X - 1) / QQE_WEIGHTED_OSCILLATOR_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(QQE_WEIGHTED_OSCILLATOR_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_data.as_device_ptr(),
                cols as i32,
                d_lengths.as_device_ptr(),
                d_factors.as_device_ptr(),
                d_smooths.as_device_ptr(),
                d_weights.as_device_ptr(),
                rows as i32,
                d_out_rsi.as_device_ptr(),
                d_out_ts.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaQqeWeightedOscillatorBatchResult {
            outputs: QqeWeightedOscillatorDeviceArrayF64Pair {
                rsi: QqeWeightedOscillatorDeviceArrayF64 {
                    buf: d_out_rsi,
                    rows,
                    cols,
                },
                trailing_stop: QqeWeightedOscillatorDeviceArrayF64 {
                    buf: d_out_ts,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
