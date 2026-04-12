#![cfg(feature = "cuda")]

use crate::indicators::normalized_resonator::{
    NormalizedResonatorBatchRange, NormalizedResonatorParams,
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

const NORMALIZED_RESONATOR_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const DEFAULT_PERIOD: usize = 100;
const DEFAULT_DELTA: f64 = 0.5;
const DEFAULT_LOOKBACK_MULT: f64 = 1.0;
const DEFAULT_SIGNAL_LENGTH: usize = 9;
const MIN_VALID_SAMPLES: usize = 3;
const FLOAT_TOL: f64 = 1e-12;

#[derive(Debug, Error)]
pub enum CudaNormalizedResonatorError {
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

pub struct NormalizedResonatorDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl NormalizedResonatorDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct NormalizedResonatorDeviceArrayF64Pair {
    pub oscillator: NormalizedResonatorDeviceArrayF64,
    pub signal: NormalizedResonatorDeviceArrayF64,
}

impl NormalizedResonatorDeviceArrayF64Pair {
    #[inline]
    pub fn rows(&self) -> usize {
        self.oscillator.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.oscillator.cols
    }
}

pub struct CudaNormalizedResonatorBatchResult {
    pub outputs: NormalizedResonatorDeviceArrayF64Pair,
    pub combos: Vec<NormalizedResonatorParams>,
}

pub struct CudaNormalizedResonator {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

fn expand_axis_usize(
    start: usize,
    end: usize,
    step: usize,
) -> Result<Vec<usize>, CudaNormalizedResonatorError> {
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
        return Err(CudaNormalizedResonatorError::InvalidInput(format!(
            "invalid range: start={start}, end={end}, step={step}"
        )));
    }
    Ok(out)
}

fn expand_axis_f64(
    start: f64,
    end: f64,
    step: f64,
) -> Result<Vec<f64>, CudaNormalizedResonatorError> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() || start > end {
        return Err(CudaNormalizedResonatorError::InvalidInput(format!(
            "invalid range: start={start}, end={end}, step={step}"
        )));
    }
    if (start - end).abs() < FLOAT_TOL {
        if step.abs() > FLOAT_TOL {
            return Err(CudaNormalizedResonatorError::InvalidInput(format!(
                "invalid range: start={start}, end={end}, step={step}"
            )));
        }
        return Ok(vec![start]);
    }
    if step <= 0.0 {
        return Err(CudaNormalizedResonatorError::InvalidInput(format!(
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
        return Err(CudaNormalizedResonatorError::InvalidInput(format!(
            "invalid range: start={start}, end={end}, step={step}"
        )));
    }
    Ok(values)
}

fn expand_grid_normalized_resonator(
    sweep: &NormalizedResonatorBatchRange,
) -> Result<Vec<NormalizedResonatorParams>, CudaNormalizedResonatorError> {
    let periods = expand_axis_usize(sweep.period.0, sweep.period.1, sweep.period.2)?;
    let deltas = expand_axis_f64(sweep.delta.0, sweep.delta.1, sweep.delta.2)?;
    let lookback_mults = expand_axis_f64(
        sweep.lookback_mult.0,
        sweep.lookback_mult.1,
        sweep.lookback_mult.2,
    )?;
    let signal_lengths = expand_axis_usize(
        sweep.signal_length.0,
        sweep.signal_length.1,
        sweep.signal_length.2,
    )?;

    let mut combos = Vec::with_capacity(
        periods
            .len()
            .saturating_mul(deltas.len())
            .saturating_mul(lookback_mults.len())
            .saturating_mul(signal_lengths.len()),
    );
    for period in periods {
        for &delta in &deltas {
            for &lookback_mult in &lookback_mults {
                for signal_length in signal_lengths.iter().copied() {
                    if period < 2 {
                        return Err(CudaNormalizedResonatorError::InvalidInput(format!(
                            "invalid period: {period}"
                        )));
                    }
                    if !delta.is_finite() || delta <= 0.0 || delta > 1.0 {
                        return Err(CudaNormalizedResonatorError::InvalidInput(format!(
                            "invalid delta: {delta}"
                        )));
                    }
                    if !lookback_mult.is_finite() || lookback_mult <= 0.0 {
                        return Err(CudaNormalizedResonatorError::InvalidInput(format!(
                            "invalid lookback_mult: {lookback_mult}"
                        )));
                    }
                    if signal_length == 0 {
                        return Err(CudaNormalizedResonatorError::InvalidInput(format!(
                            "invalid signal_length: {signal_length}"
                        )));
                    }
                    let alpha = (std::f64::consts::PI * delta / period as f64).tan();
                    if !alpha.is_finite() {
                        return Err(CudaNormalizedResonatorError::InvalidInput(format!(
                            "invalid delta: {delta}"
                        )));
                    }
                    let peak_lookback_raw = period as f64 * lookback_mult;
                    if !peak_lookback_raw.is_finite() || peak_lookback_raw > usize::MAX as f64 {
                        return Err(CudaNormalizedResonatorError::InvalidInput(format!(
                            "invalid lookback_mult: {lookback_mult}"
                        )));
                    }
                    combos.push(NormalizedResonatorParams {
                        period: Some(period),
                        delta: Some(delta),
                        lookback_mult: Some(lookback_mult),
                        signal_length: Some(signal_length),
                    });
                }
            }
        }
    }
    Ok(combos)
}

impl CudaNormalizedResonator {
    pub fn new(device_id: usize) -> Result<Self, CudaNormalizedResonatorError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("normalized_resonator_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaNormalizedResonatorError> {
        self.stream.synchronize()?;
        Ok(())
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
                if run > best {
                    best = run;
                }
            } else {
                run = 0;
            }
        }
        best
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaNormalizedResonatorError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaNormalizedResonatorError::OutOfMemory {
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
    ) -> Result<(), CudaNormalizedResonatorError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaNormalizedResonatorError::LaunchConfigTooLarge {
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
        sweep: &NormalizedResonatorBatchRange,
    ) -> Result<CudaNormalizedResonatorBatchResult, CudaNormalizedResonatorError> {
        if data.is_empty() {
            return Err(CudaNormalizedResonatorError::InvalidInput(
                "empty input".into(),
            ));
        }
        Self::first_valid_value(data).ok_or_else(|| {
            CudaNormalizedResonatorError::InvalidInput("all values are NaN".into())
        })?;
        let valid = Self::max_consecutive_valid_values(data);
        if valid < MIN_VALID_SAMPLES {
            return Err(CudaNormalizedResonatorError::InvalidInput(format!(
                "not enough valid data: needed={MIN_VALID_SAMPLES}, valid={valid}"
            )));
        }

        let combos = expand_grid_normalized_resonator(sweep)?;
        if combos.is_empty() {
            return Err(CudaNormalizedResonatorError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let rows = combos.len();
        let cols = data.len();
        let mut periods = Vec::with_capacity(rows);
        let mut deltas = Vec::with_capacity(rows);
        let mut lookback_mults = Vec::with_capacity(rows);
        let mut signal_lengths = Vec::with_capacity(rows);

        for combo in &combos {
            let period = combo.period.unwrap_or(DEFAULT_PERIOD);
            let delta = combo.delta.unwrap_or(DEFAULT_DELTA);
            let lookback_mult = combo.lookback_mult.unwrap_or(DEFAULT_LOOKBACK_MULT);
            let signal_length = combo.signal_length.unwrap_or(DEFAULT_SIGNAL_LENGTH);
            periods.push(i32::try_from(period).map_err(|_| {
                CudaNormalizedResonatorError::InvalidInput(format!("period out of range: {period}"))
            })?);
            deltas.push(delta);
            lookback_mults.push(lookback_mult);
            signal_lengths.push(i32::try_from(signal_length).map_err(|_| {
                CudaNormalizedResonatorError::InvalidInput(format!(
                    "signal_length out of range: {signal_length}"
                ))
            })?);
        }

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaNormalizedResonatorError::InvalidInput("input bytes overflow".into())
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
                CudaNormalizedResonatorError::InvalidInput("params bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaNormalizedResonatorError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                CudaNormalizedResonatorError::InvalidInput("output bytes overflow".into())
            })?;
        let scratch_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaNormalizedResonatorError::InvalidInput("scratch bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .and_then(|value| value.checked_add(scratch_bytes))
            .ok_or_else(|| {
                CudaNormalizedResonatorError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_data = DeviceBuffer::from_slice(data)?;
        let d_periods = DeviceBuffer::from_slice(&periods)?;
        let d_deltas = DeviceBuffer::from_slice(&deltas)?;
        let d_lookback_mults = DeviceBuffer::from_slice(&lookback_mults)?;
        let d_signal_lengths = DeviceBuffer::from_slice(&signal_lengths)?;
        let d_out_oscillator = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_signal = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_bp_history = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("normalized_resonator_batch_f64")
            .map_err(|_| CudaNormalizedResonatorError::MissingKernelSymbol {
                name: "normalized_resonator_batch_f64",
            })?;
        let grid_x =
            ((rows as u32) + NORMALIZED_RESONATOR_BLOCK_X - 1) / NORMALIZED_RESONATOR_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(NORMALIZED_RESONATOR_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_data.as_device_ptr(),
                cols as i32,
                d_periods.as_device_ptr(),
                d_deltas.as_device_ptr(),
                d_lookback_mults.as_device_ptr(),
                d_signal_lengths.as_device_ptr(),
                rows as i32,
                d_out_oscillator.as_device_ptr(),
                d_out_signal.as_device_ptr(),
                d_bp_history.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaNormalizedResonatorBatchResult {
            outputs: NormalizedResonatorDeviceArrayF64Pair {
                oscillator: NormalizedResonatorDeviceArrayF64 {
                    buf: d_out_oscillator,
                    rows,
                    cols,
                },
                signal: NormalizedResonatorDeviceArrayF64 {
                    buf: d_out_signal,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
