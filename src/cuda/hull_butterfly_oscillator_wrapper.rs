#![cfg(feature = "cuda")]

use crate::indicators::hull_butterfly_oscillator::{
    HullButterflyOscillatorBatchRange, HullButterflyOscillatorParams,
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

const HULL_BUTTERFLY_OSCILLATOR_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const DEFAULT_LENGTH: usize = 14;
const DEFAULT_MULT: f64 = 2.0;
const FLOAT_TOL: f64 = 1e-12;

#[derive(Debug, Error)]
pub enum CudaHullButterflyOscillatorError {
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

pub struct HullButterflyOscillatorDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl HullButterflyOscillatorDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct HullButterflyOscillatorDeviceArrayF64Triple {
    pub oscillator: HullButterflyOscillatorDeviceArrayF64,
    pub cumulative_mean: HullButterflyOscillatorDeviceArrayF64,
    pub signal: HullButterflyOscillatorDeviceArrayF64,
}

impl HullButterflyOscillatorDeviceArrayF64Triple {
    #[inline]
    pub fn rows(&self) -> usize {
        self.oscillator.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.oscillator.cols
    }
}

pub struct CudaHullButterflyOscillatorBatchResult {
    pub outputs: HullButterflyOscillatorDeviceArrayF64Triple,
    pub combos: Vec<HullButterflyOscillatorParams>,
}

pub struct CudaHullButterflyOscillator {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

fn first_valid_value(data: &[f64]) -> Option<usize> {
    data.iter().position(|value| value.is_finite())
}

fn max_consecutive_valid_values(data: &[f64]) -> usize {
    let mut best = 0usize;
    let mut run = 0usize;
    for &value in data {
        if value.is_finite() {
            run += 1;
            best = best.max(run);
        } else {
            run = 0;
        }
    }
    best
}

fn compute_hull_coeffs(length: usize) -> Vec<f64> {
    let short_len = length / 2;
    let hull_len = ((length as f64).sqrt().floor() as usize).max(1);
    let den1 = (short_len * (short_len + 1) / 2) as f64;
    let den2 = (length * (length + 1) / 2) as f64;
    let den3 = (hull_len * (hull_len + 1) / 2) as f64;

    let mut lcwa_coeffs = vec![0.0; hull_len];
    for i in 0..length {
        let sum1 = short_len.saturating_sub(i) as f64;
        let sum2 = (length - i) as f64;
        lcwa_coeffs.insert(0, 2.0 * (sum1 / den1) - (sum2 / den2));
    }
    for _ in 0..hull_len.saturating_sub(1) {
        lcwa_coeffs.insert(0, 0.0);
    }

    let size = lcwa_coeffs.len();
    let mut hull_coeffs = Vec::with_capacity(size.saturating_sub(hull_len));
    for i in hull_len..size {
        let mut sum3 = 0.0;
        for j in (i - hull_len)..i {
            sum3 += lcwa_coeffs[j] * (i - j) as f64;
        }
        hull_coeffs.insert(0, sum3 / den3);
    }
    hull_coeffs
}

fn expand_axis_usize(
    start: usize,
    end: usize,
    step: usize,
) -> Result<Vec<usize>, CudaHullButterflyOscillatorError> {
    if step == 0 || start == end {
        return Ok(vec![start]);
    }

    let mut out = Vec::new();
    if start < end {
        let mut value = start;
        while value <= end {
            out.push(value);
            let next = value.saturating_add(step);
            if next == value {
                break;
            }
            value = next;
        }
    } else {
        let mut value = start;
        loop {
            out.push(value);
            if value == end {
                break;
            }
            let next = value.saturating_sub(step);
            if next == value || next < end {
                break;
            }
            value = next;
        }
    }

    if out.is_empty() {
        return Err(CudaHullButterflyOscillatorError::InvalidInput(format!(
            "invalid range: start={start}, end={end}, step={step}"
        )));
    }
    Ok(out)
}

fn expand_axis_f64(
    start: f64,
    end: f64,
    step: f64,
) -> Result<Vec<f64>, CudaHullButterflyOscillatorError> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() || start > end {
        return Err(CudaHullButterflyOscillatorError::InvalidInput(format!(
            "invalid range: start={start}, end={end}, step={step}"
        )));
    }
    if (start - end).abs() < FLOAT_TOL {
        if step.abs() > FLOAT_TOL {
            return Err(CudaHullButterflyOscillatorError::InvalidInput(format!(
                "invalid range: start={start}, end={end}, step={step}"
            )));
        }
        return Ok(vec![start]);
    }
    if step <= 0.0 {
        return Err(CudaHullButterflyOscillatorError::InvalidInput(format!(
            "invalid range: start={start}, end={end}, step={step}"
        )));
    }

    let mut values = Vec::new();
    let mut value = start;
    while value <= end + FLOAT_TOL {
        values.push(value.min(end));
        value += step;
    }
    if (values.last().copied().unwrap_or(start) - end).abs() > 1e-9 {
        return Err(CudaHullButterflyOscillatorError::InvalidInput(format!(
            "invalid range: start={start}, end={end}, step={step}"
        )));
    }
    Ok(values)
}

fn expand_grid_hull_butterfly_oscillator(
    sweep: &HullButterflyOscillatorBatchRange,
) -> Result<Vec<HullButterflyOscillatorParams>, CudaHullButterflyOscillatorError> {
    let lengths = expand_axis_usize(sweep.length.0, sweep.length.1, sweep.length.2)?;
    let mults = expand_axis_f64(sweep.mult.0, sweep.mult.1, sweep.mult.2)?;
    let mut combos = Vec::with_capacity(lengths.len().saturating_mul(mults.len()));
    for length in lengths {
        if length < 2 {
            return Err(CudaHullButterflyOscillatorError::InvalidInput(format!(
                "invalid length: length={length}"
            )));
        }
        for &mult in &mults {
            if !mult.is_finite() {
                return Err(CudaHullButterflyOscillatorError::InvalidInput(format!(
                    "invalid multiplier: {mult}"
                )));
            }
            combos.push(HullButterflyOscillatorParams {
                length: Some(length),
                mult: Some(mult),
            });
        }
    }
    Ok(combos)
}

impl CudaHullButterflyOscillator {
    pub fn new(device_id: usize) -> Result<Self, CudaHullButterflyOscillatorError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("hull_butterfly_oscillator_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaHullButterflyOscillatorError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaHullButterflyOscillatorError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaHullButterflyOscillatorError::OutOfMemory {
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
    ) -> Result<(), CudaHullButterflyOscillatorError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaHullButterflyOscillatorError::LaunchConfigTooLarge {
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
        sweep: &HullButterflyOscillatorBatchRange,
    ) -> Result<CudaHullButterflyOscillatorBatchResult, CudaHullButterflyOscillatorError> {
        if data.is_empty() {
            return Err(CudaHullButterflyOscillatorError::InvalidInput(
                "empty input".into(),
            ));
        }
        first_valid_value(data).ok_or_else(|| {
            CudaHullButterflyOscillatorError::InvalidInput("all values are NaN".into())
        })?;

        let combos = expand_grid_hull_butterfly_oscillator(sweep)?;
        if combos.is_empty() {
            return Err(CudaHullButterflyOscillatorError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let valid = max_consecutive_valid_values(data);
        let rows = combos.len();
        let cols = data.len();
        let mut coeff_lens = Vec::with_capacity(rows);
        let mut mults = Vec::with_capacity(rows);
        let mut coeff_rows = Vec::with_capacity(rows);
        let mut max_coeff_len = 0usize;
        for combo in &combos {
            let length = combo.length.unwrap_or(DEFAULT_LENGTH);
            let mult = combo.mult.unwrap_or(DEFAULT_MULT);
            if length < 2 || length > cols {
                return Err(CudaHullButterflyOscillatorError::InvalidInput(format!(
                    "invalid length: length={length}, data_len={cols}"
                )));
            }
            if !mult.is_finite() {
                return Err(CudaHullButterflyOscillatorError::InvalidInput(format!(
                    "invalid multiplier: {mult}"
                )));
            }

            let coeffs = compute_hull_coeffs(length);
            let needed = coeffs.len();
            if valid < needed {
                return Err(CudaHullButterflyOscillatorError::InvalidInput(format!(
                    "not enough valid data: needed={needed}, valid={valid}"
                )));
            }
            max_coeff_len = max_coeff_len.max(coeffs.len());
            coeff_rows.push(coeffs);
            coeff_lens.push(i32::try_from(needed).map_err(|_| {
                CudaHullButterflyOscillatorError::InvalidInput(format!(
                    "coefficient length out of range: {needed}"
                ))
            })?);
            mults.push(mult);
        }

        let coeff_elems = rows.checked_mul(max_coeff_len).ok_or_else(|| {
            CudaHullButterflyOscillatorError::InvalidInput("coeff rows*cols overflow".into())
        })?;
        let mut coeff_matrix = vec![0.0f64; coeff_elems];
        for (row, coeffs) in coeff_rows.iter().enumerate() {
            let start = row * max_coeff_len;
            coeff_matrix[start..start + coeffs.len()].copy_from_slice(coeffs);
        }

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaHullButterflyOscillatorError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = rows
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| {
                rows.checked_mul(std::mem::size_of::<f64>())
                    .and_then(|other| value.checked_add(other))
            })
            .and_then(|value| {
                coeff_elems
                    .checked_mul(std::mem::size_of::<f64>())
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| {
                CudaHullButterflyOscillatorError::InvalidInput("params bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaHullButterflyOscillatorError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(3))
            .ok_or_else(|| {
                CudaHullButterflyOscillatorError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaHullButterflyOscillatorError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_data = DeviceBuffer::from_slice(data)?;
        let d_coeff_lens = DeviceBuffer::from_slice(&coeff_lens)?;
        let d_mults = DeviceBuffer::from_slice(&mults)?;
        let d_coeffs = DeviceBuffer::from_slice(&coeff_matrix)?;
        let d_out_oscillator = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_cumulative_mean = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_signal = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("hull_butterfly_oscillator_batch_f64")
            .map_err(|_| CudaHullButterflyOscillatorError::MissingKernelSymbol {
                name: "hull_butterfly_oscillator_batch_f64",
            })?;
        let grid_x = ((rows as u32) + HULL_BUTTERFLY_OSCILLATOR_BLOCK_X - 1)
            / HULL_BUTTERFLY_OSCILLATOR_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(HULL_BUTTERFLY_OSCILLATOR_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_data.as_device_ptr(),
                cols as i32,
                d_coeff_lens.as_device_ptr(),
                d_mults.as_device_ptr(),
                d_coeffs.as_device_ptr(),
                max_coeff_len as i32,
                rows as i32,
                d_out_oscillator.as_device_ptr(),
                d_out_cumulative_mean.as_device_ptr(),
                d_out_signal.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaHullButterflyOscillatorBatchResult {
            outputs: HullButterflyOscillatorDeviceArrayF64Triple {
                oscillator: HullButterflyOscillatorDeviceArrayF64 {
                    buf: d_out_oscillator,
                    rows,
                    cols,
                },
                cumulative_mean: HullButterflyOscillatorDeviceArrayF64 {
                    buf: d_out_cumulative_mean,
                    rows,
                    cols,
                },
                signal: HullButterflyOscillatorDeviceArrayF64 {
                    buf: d_out_signal,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
