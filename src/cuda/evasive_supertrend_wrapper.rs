#![cfg(feature = "cuda")]

use crate::indicators::evasive_supertrend::{EvasiveSuperTrendBatchRange, EvasiveSuperTrendParams};
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

const EVASIVE_SUPERTREND_BLOCK_X: u32 = 64;
const DEFAULT_ATR_LENGTH: usize = 10;
const DEFAULT_BASE_MULTIPLIER: f64 = 3.0;
const DEFAULT_NOISE_THRESHOLD: f64 = 1.0;
const DEFAULT_EXPANSION_ALPHA: f64 = 0.5;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaEvasiveSuperTrendError {
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

pub struct EvasiveSuperTrendDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl EvasiveSuperTrendDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct EvasiveSuperTrendDeviceOutputs {
    pub band: EvasiveSuperTrendDeviceArrayF64,
    pub state: EvasiveSuperTrendDeviceArrayF64,
    pub noisy: EvasiveSuperTrendDeviceArrayF64,
    pub changed: EvasiveSuperTrendDeviceArrayF64,
}

impl EvasiveSuperTrendDeviceOutputs {
    #[inline]
    pub fn rows(&self) -> usize {
        self.band.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.band.cols
    }
}

pub struct CudaEvasiveSuperTrendBatchResult {
    pub outputs: EvasiveSuperTrendDeviceOutputs,
    pub combos: Vec<EvasiveSuperTrendParams>,
}

pub struct CudaEvasiveSuperTrend {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

#[inline]
fn expand_usize_range(
    start: usize,
    end: usize,
    step: usize,
) -> Result<Vec<usize>, CudaEvasiveSuperTrendError> {
    if start == 0 || end == 0 {
        return Err(CudaEvasiveSuperTrendError::InvalidInput(format!(
            "invalid range: start={start}, end={end}, step={step}"
        )));
    }
    if step == 0 {
        return Ok(vec![start]);
    }
    if start > end {
        return Err(CudaEvasiveSuperTrendError::InvalidInput(format!(
            "invalid range: start={start}, end={end}, step={step}"
        )));
    }
    let mut out = Vec::new();
    let mut current = start;
    loop {
        out.push(current);
        if current >= end {
            break;
        }
        let next = current.saturating_add(step);
        if next <= current {
            return Err(CudaEvasiveSuperTrendError::InvalidInput(format!(
                "invalid range: start={start}, end={end}, step={step}"
            )));
        }
        current = next.min(end);
        if current == *out.last().unwrap_or(&0) {
            break;
        }
    }
    Ok(out)
}

#[inline]
fn expand_f64_range(
    start: f64,
    end: f64,
    step: f64,
) -> Result<Vec<f64>, CudaEvasiveSuperTrendError> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() {
        return Err(CudaEvasiveSuperTrendError::InvalidInput(format!(
            "invalid range: start={start}, end={end}, step={step}"
        )));
    }
    if step == 0.0 {
        return Ok(vec![start]);
    }
    if start > end || step < 0.0 {
        return Err(CudaEvasiveSuperTrendError::InvalidInput(format!(
            "invalid range: start={start}, end={end}, step={step}"
        )));
    }
    let mut out = Vec::new();
    let mut current = start;
    loop {
        out.push(current);
        if current >= end || (end - current).abs() <= 1e-12 {
            break;
        }
        let next = current + step;
        if next <= current {
            return Err(CudaEvasiveSuperTrendError::InvalidInput(format!(
                "invalid range: start={start}, end={end}, step={step}"
            )));
        }
        current = if next > end { end } else { next };
    }
    Ok(out)
}

#[inline]
fn expand_grid_checked(
    range: &EvasiveSuperTrendBatchRange,
) -> Result<Vec<EvasiveSuperTrendParams>, CudaEvasiveSuperTrendError> {
    let atr_lengths =
        expand_usize_range(range.atr_length.0, range.atr_length.1, range.atr_length.2)?;
    let base_multipliers = expand_f64_range(
        range.base_multiplier.0,
        range.base_multiplier.1,
        range.base_multiplier.2,
    )?;
    let noise_thresholds = expand_f64_range(
        range.noise_threshold.0,
        range.noise_threshold.1,
        range.noise_threshold.2,
    )?;
    let expansion_alphas = expand_f64_range(
        range.expansion_alpha.0,
        range.expansion_alpha.1,
        range.expansion_alpha.2,
    )?;
    let mut out = Vec::new();
    for &atr_length in &atr_lengths {
        for &base_multiplier in &base_multipliers {
            for &noise_threshold in &noise_thresholds {
                for &expansion_alpha in &expansion_alphas {
                    out.push(EvasiveSuperTrendParams {
                        atr_length: Some(atr_length),
                        base_multiplier: Some(base_multiplier),
                        noise_threshold: Some(noise_threshold),
                        expansion_alpha: Some(expansion_alpha),
                    });
                }
            }
        }
    }
    Ok(out)
}

#[inline]
fn is_valid_ohlc(open: f64, high: f64, low: f64, close: f64) -> bool {
    open.is_finite() && high.is_finite() && low.is_finite() && close.is_finite()
}

#[inline]
fn longest_valid_run(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> usize {
    let mut best = 0usize;
    let mut cur = 0usize;
    for (((open, high), low), close) in open
        .iter()
        .zip(high.iter())
        .zip(low.iter())
        .zip(close.iter())
    {
        if is_valid_ohlc(*open, *high, *low, *close) {
            cur += 1;
            best = best.max(cur);
        } else {
            cur = 0;
        }
    }
    best
}

impl CudaEvasiveSuperTrend {
    pub fn new(device_id: usize) -> Result<Self, CudaEvasiveSuperTrendError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("evasive_supertrend_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaEvasiveSuperTrendError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaEvasiveSuperTrendError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaEvasiveSuperTrendError::OutOfMemory {
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
    ) -> Result<(), CudaEvasiveSuperTrendError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaEvasiveSuperTrendError::LaunchConfigTooLarge {
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
        high: &[f64],
        low: &[f64],
        close: &[f64],
        sweep: &EvasiveSuperTrendBatchRange,
    ) -> Result<CudaEvasiveSuperTrendBatchResult, CudaEvasiveSuperTrendError> {
        if open.is_empty() || high.is_empty() || low.is_empty() || close.is_empty() {
            return Err(CudaEvasiveSuperTrendError::InvalidInput(
                "empty input".into(),
            ));
        }
        if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
            return Err(CudaEvasiveSuperTrendError::InvalidInput(format!(
                "input length mismatch: open={}, high={}, low={}, close={}",
                open.len(),
                high.len(),
                low.len(),
                close.len()
            )));
        }

        let combos = expand_grid_checked(sweep)?;
        let max_atr_length = combos
            .iter()
            .map(|params| params.atr_length.unwrap_or(DEFAULT_ATR_LENGTH))
            .max()
            .unwrap_or(0);
        let max_base_multiplier = combos
            .iter()
            .map(|params| params.base_multiplier.unwrap_or(DEFAULT_BASE_MULTIPLIER))
            .fold(0.0_f64, f64::max);
        let max_noise_threshold = combos
            .iter()
            .map(|params| params.noise_threshold.unwrap_or(DEFAULT_NOISE_THRESHOLD))
            .fold(0.0_f64, f64::max);
        let max_expansion_alpha = combos
            .iter()
            .map(|params| params.expansion_alpha.unwrap_or(DEFAULT_EXPANSION_ALPHA))
            .fold(0.0_f64, f64::max);

        if max_atr_length == 0
            || !max_base_multiplier.is_finite()
            || max_base_multiplier < 0.1
            || !max_noise_threshold.is_finite()
            || max_noise_threshold < 0.1
            || !max_expansion_alpha.is_finite()
            || max_expansion_alpha < 0.0
        {
            return Err(CudaEvasiveSuperTrendError::InvalidInput(
                "invalid parameters".into(),
            ));
        }

        let longest = longest_valid_run(open, high, low, close);
        if longest == 0 {
            return Err(CudaEvasiveSuperTrendError::InvalidInput(
                "all values are NaN".into(),
            ));
        }
        if longest < max_atr_length {
            return Err(CudaEvasiveSuperTrendError::InvalidInput(format!(
                "not enough valid data: needed={max_atr_length}, valid={longest}"
            )));
        }

        let rows = combos.len();
        let cols = close.len();
        let atr_lengths: Vec<i32> = combos
            .iter()
            .map(|params| params.atr_length.unwrap_or(DEFAULT_ATR_LENGTH) as i32)
            .collect();
        let base_multipliers: Vec<f64> = combos
            .iter()
            .map(|params| params.base_multiplier.unwrap_or(DEFAULT_BASE_MULTIPLIER))
            .collect();
        let noise_thresholds: Vec<f64> = combos
            .iter()
            .map(|params| params.noise_threshold.unwrap_or(DEFAULT_NOISE_THRESHOLD))
            .collect();
        let expansion_alphas: Vec<f64> = combos
            .iter()
            .map(|params| params.expansion_alpha.unwrap_or(DEFAULT_EXPANSION_ALPHA))
            .collect();

        let output_elems = rows
            .checked_mul(cols)
            .ok_or_else(|| CudaEvasiveSuperTrendError::InvalidInput("rows*cols overflow".into()))?;
        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| {
                CudaEvasiveSuperTrendError::InvalidInput("input bytes overflow".into())
            })?;
        let param_bytes = rows
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(3))
            .and_then(|value| value.checked_add(rows * std::mem::size_of::<i32>()))
            .ok_or_else(|| {
                CudaEvasiveSuperTrendError::InvalidInput("param bytes overflow".into())
            })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| {
                CudaEvasiveSuperTrendError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(param_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaEvasiveSuperTrendError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_open = DeviceBuffer::from_slice(open)?;
        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_close = DeviceBuffer::from_slice(close)?;
        let d_atr_lengths = DeviceBuffer::from_slice(&atr_lengths)?;
        let d_base_multipliers = DeviceBuffer::from_slice(&base_multipliers)?;
        let d_noise_thresholds = DeviceBuffer::from_slice(&noise_thresholds)?;
        let d_expansion_alphas = DeviceBuffer::from_slice(&expansion_alphas)?;
        let mut d_band = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_state = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_noisy = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_changed = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("evasive_supertrend_batch_f64")
            .map_err(|_| CudaEvasiveSuperTrendError::MissingKernelSymbol {
                name: "evasive_supertrend_batch_f64",
            })?;
        let grid_x = ((rows as u32) + EVASIVE_SUPERTREND_BLOCK_X - 1) / EVASIVE_SUPERTREND_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(EVASIVE_SUPERTREND_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_open.as_device_ptr(),
                d_high.as_device_ptr(),
                d_low.as_device_ptr(),
                d_close.as_device_ptr(),
                cols as i32,
                d_atr_lengths.as_device_ptr(),
                d_base_multipliers.as_device_ptr(),
                d_noise_thresholds.as_device_ptr(),
                d_expansion_alphas.as_device_ptr(),
                rows as i32,
                d_band.as_device_ptr(),
                d_state.as_device_ptr(),
                d_noisy.as_device_ptr(),
                d_changed.as_device_ptr()
            ))?;
        }
        self.stream.synchronize()?;

        Ok(CudaEvasiveSuperTrendBatchResult {
            outputs: EvasiveSuperTrendDeviceOutputs {
                band: EvasiveSuperTrendDeviceArrayF64 {
                    buf: d_band,
                    rows,
                    cols,
                },
                state: EvasiveSuperTrendDeviceArrayF64 {
                    buf: d_state,
                    rows,
                    cols,
                },
                noisy: EvasiveSuperTrendDeviceArrayF64 {
                    buf: d_noisy,
                    rows,
                    cols,
                },
                changed: EvasiveSuperTrendDeviceArrayF64 {
                    buf: d_changed,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
