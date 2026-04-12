#![cfg(feature = "cuda")]

use crate::indicators::adaptive_schaff_trend_cycle::{
    AdaptiveSchaffTrendCycleBatchRange, AdaptiveSchaffTrendCycleParams,
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

const ADAPTIVE_SCHAFF_TREND_CYCLE_BLOCK_X: u32 = 64;
const DEFAULT_ADAPTIVE_LENGTH: usize = 55;
const DEFAULT_STC_LENGTH: usize = 12;
const DEFAULT_SMOOTHING_FACTOR: f64 = 0.45;
const DEFAULT_FAST_LENGTH: usize = 26;
const DEFAULT_SLOW_LENGTH: usize = 50;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const EPS: f64 = 1.0e-12;

#[derive(Debug, Error)]
pub enum CudaAdaptiveSchaffTrendCycleError {
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

pub struct AdaptiveSchaffTrendCycleDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl AdaptiveSchaffTrendCycleDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct AdaptiveSchaffTrendCycleDeviceOutputs {
    pub stc: AdaptiveSchaffTrendCycleDeviceArrayF64,
    pub histogram: AdaptiveSchaffTrendCycleDeviceArrayF64,
}

impl AdaptiveSchaffTrendCycleDeviceOutputs {
    #[inline]
    pub fn rows(&self) -> usize {
        self.stc.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.stc.cols
    }
}

pub struct CudaAdaptiveSchaffTrendCycleBatchResult {
    pub outputs: AdaptiveSchaffTrendCycleDeviceOutputs,
    pub combos: Vec<AdaptiveSchaffTrendCycleParams>,
}

pub struct CudaAdaptiveSchaffTrendCycle {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

#[inline]
fn axis_usize(
    (start, end, step): (usize, usize, usize),
) -> Result<Vec<usize>, CudaAdaptiveSchaffTrendCycleError> {
    if step == 0 || start == end {
        return Ok(vec![start]);
    }

    let mut out = Vec::new();
    if start <= end {
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
        while x >= end {
            out.push(x);
            let next = x.saturating_sub(step);
            if next == x {
                break;
            }
            x = next;
            if x < end {
                break;
            }
        }
    }

    if out.is_empty() {
        return Err(CudaAdaptiveSchaffTrendCycleError::InvalidInput(format!(
            "invalid range: start={start}, end={end}, step={step}"
        )));
    }
    Ok(out)
}

#[inline]
fn axis_f64(
    (start, end, step): (f64, f64, f64),
) -> Result<Vec<f64>, CudaAdaptiveSchaffTrendCycleError> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() {
        return Err(CudaAdaptiveSchaffTrendCycleError::InvalidInput(format!(
            "invalid float range: start={start}, end={end}, step={step}"
        )));
    }
    if step.abs() < EPS || (start - end).abs() < EPS {
        return Ok(vec![start]);
    }

    let step = step.abs();
    let mut out = Vec::new();
    if start <= end {
        let mut x = start;
        while x <= end + EPS {
            out.push(x);
            x += step;
        }
    } else {
        let mut x = start;
        while x + EPS >= end {
            out.push(x);
            x -= step;
        }
    }

    if out.is_empty() {
        return Err(CudaAdaptiveSchaffTrendCycleError::InvalidInput(format!(
            "invalid float range: start={start}, end={end}, step={step}"
        )));
    }
    Ok(out)
}

#[inline]
fn expand_grid_checked(
    range: &AdaptiveSchaffTrendCycleBatchRange,
) -> Result<Vec<AdaptiveSchaffTrendCycleParams>, CudaAdaptiveSchaffTrendCycleError> {
    let adaptive_lengths = axis_usize(range.adaptive_length)?;
    let stc_lengths = axis_usize(range.stc_length)?;
    let smoothing_factors = axis_f64(range.smoothing_factor)?;
    let fast_lengths = axis_usize(range.fast_length)?;
    let slow_lengths = axis_usize(range.slow_length)?;

    let cap = adaptive_lengths
        .len()
        .checked_mul(stc_lengths.len())
        .and_then(|value| value.checked_mul(smoothing_factors.len()))
        .and_then(|value| value.checked_mul(fast_lengths.len()))
        .and_then(|value| value.checked_mul(slow_lengths.len()))
        .ok_or_else(|| {
            CudaAdaptiveSchaffTrendCycleError::InvalidInput("parameter grid overflow".into())
        })?;

    let mut out = Vec::with_capacity(cap);
    for &adaptive_length in &adaptive_lengths {
        for &stc_length in &stc_lengths {
            for &smoothing_factor in &smoothing_factors {
                for &fast_length in &fast_lengths {
                    for &slow_length in &slow_lengths {
                        out.push(AdaptiveSchaffTrendCycleParams {
                            adaptive_length: Some(adaptive_length),
                            stc_length: Some(stc_length),
                            smoothing_factor: Some(smoothing_factor),
                            fast_length: Some(fast_length),
                            slow_length: Some(slow_length),
                        });
                    }
                }
            }
        }
    }
    Ok(out)
}

#[inline]
fn valid_bar(high: f64, low: f64, close: f64) -> bool {
    high.is_finite() && low.is_finite() && close.is_finite() && high >= low
}

impl CudaAdaptiveSchaffTrendCycle {
    pub fn new(device_id: usize) -> Result<Self, CudaAdaptiveSchaffTrendCycleError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("adaptive_schaff_trend_cycle_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaAdaptiveSchaffTrendCycleError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaAdaptiveSchaffTrendCycleError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaAdaptiveSchaffTrendCycleError::OutOfMemory {
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
    ) -> Result<(), CudaAdaptiveSchaffTrendCycleError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaAdaptiveSchaffTrendCycleError::LaunchConfigTooLarge {
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
        high: &[f64],
        low: &[f64],
        close: &[f64],
        sweep: &AdaptiveSchaffTrendCycleBatchRange,
    ) -> Result<CudaAdaptiveSchaffTrendCycleBatchResult, CudaAdaptiveSchaffTrendCycleError> {
        if high.is_empty() || low.is_empty() || close.is_empty() {
            return Err(CudaAdaptiveSchaffTrendCycleError::InvalidInput(
                "empty input".into(),
            ));
        }
        if high.len() != low.len() || low.len() != close.len() {
            return Err(CudaAdaptiveSchaffTrendCycleError::InvalidInput(format!(
                "input length mismatch: high={}, low={}, close={}",
                high.len(),
                low.len(),
                close.len()
            )));
        }

        let combos = expand_grid_checked(sweep)?;
        if !high
            .iter()
            .zip(low.iter())
            .zip(close.iter())
            .any(|((high, low), close)| valid_bar(*high, *low, *close))
        {
            return Err(CudaAdaptiveSchaffTrendCycleError::InvalidInput(
                "all values are NaN".into(),
            ));
        }

        let rows = combos.len();
        let cols = close.len();
        let adaptive_cap = combos
            .iter()
            .map(|params| params.adaptive_length.unwrap_or(DEFAULT_ADAPTIVE_LENGTH))
            .max()
            .unwrap_or(0);
        let stc_cap = combos
            .iter()
            .map(|params| params.stc_length.unwrap_or(DEFAULT_STC_LENGTH))
            .max()
            .unwrap_or(0);
        let queue_cap = stc_cap.saturating_add(1);
        if adaptive_cap == 0 || stc_cap == 0 {
            return Err(CudaAdaptiveSchaffTrendCycleError::InvalidInput(
                "invalid parameter grid".into(),
            ));
        }

        let adaptive_lengths: Vec<i32> = combos
            .iter()
            .map(|params| params.adaptive_length.unwrap_or(DEFAULT_ADAPTIVE_LENGTH) as i32)
            .collect();
        let stc_lengths: Vec<i32> = combos
            .iter()
            .map(|params| params.stc_length.unwrap_or(DEFAULT_STC_LENGTH) as i32)
            .collect();
        let smoothing_factors: Vec<f64> = combos
            .iter()
            .map(|params| params.smoothing_factor.unwrap_or(DEFAULT_SMOOTHING_FACTOR))
            .collect();
        let fast_lengths: Vec<i32> = combos
            .iter()
            .map(|params| params.fast_length.unwrap_or(DEFAULT_FAST_LENGTH) as i32)
            .collect();
        let slow_lengths: Vec<i32> = combos
            .iter()
            .map(|params| params.slow_length.unwrap_or(DEFAULT_SLOW_LENGTH) as i32)
            .collect();

        for params in &combos {
            let adaptive_length = params.adaptive_length.unwrap_or(DEFAULT_ADAPTIVE_LENGTH);
            let stc_length = params.stc_length.unwrap_or(DEFAULT_STC_LENGTH);
            let smoothing_factor = params.smoothing_factor.unwrap_or(DEFAULT_SMOOTHING_FACTOR);
            let fast_length = params.fast_length.unwrap_or(DEFAULT_FAST_LENGTH);
            let slow_length = params.slow_length.unwrap_or(DEFAULT_SLOW_LENGTH);
            if adaptive_length == 0 || adaptive_length > cols {
                return Err(CudaAdaptiveSchaffTrendCycleError::InvalidInput(format!(
                    "invalid adaptive_length: {adaptive_length}"
                )));
            }
            if stc_length == 0 || stc_length > cols {
                return Err(CudaAdaptiveSchaffTrendCycleError::InvalidInput(format!(
                    "invalid stc_length: {stc_length}"
                )));
            }
            if !smoothing_factor.is_finite()
                || !(0.0..=1.0).contains(&smoothing_factor)
                || smoothing_factor <= 0.0
            {
                return Err(CudaAdaptiveSchaffTrendCycleError::InvalidInput(format!(
                    "invalid smoothing_factor: {smoothing_factor}"
                )));
            }
            if fast_length == 0 || slow_length == 0 {
                return Err(CudaAdaptiveSchaffTrendCycleError::InvalidInput(
                    "invalid EMA lengths".into(),
                ));
            }
        }

        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaAdaptiveSchaffTrendCycleError::InvalidInput("rows*cols overflow".into())
        })?;
        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(3))
            .ok_or_else(|| {
                CudaAdaptiveSchaffTrendCycleError::InvalidInput("input bytes overflow".into())
            })?;
        let param_bytes = rows
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_add(rows * std::mem::size_of::<f64>()))
            .and_then(|value| value.checked_add(rows * std::mem::size_of::<i32>() * 4))
            .ok_or_else(|| {
                CudaAdaptiveSchaffTrendCycleError::InvalidInput("param bytes overflow".into())
            })?;
        let scratch_values_bytes = rows
            .checked_mul(adaptive_cap)
            .and_then(|value| value.checked_mul(std::mem::size_of::<f64>()))
            .ok_or_else(|| {
                CudaAdaptiveSchaffTrendCycleError::InvalidInput("scratch values overflow".into())
            })?;
        let scratch_idx_bytes = rows
            .checked_mul(queue_cap)
            .and_then(|value| value.checked_mul(4))
            .and_then(|value| value.checked_mul(std::mem::size_of::<i32>()))
            .ok_or_else(|| {
                CudaAdaptiveSchaffTrendCycleError::InvalidInput("scratch idx overflow".into())
            })?;
        let scratch_val_bytes = rows
            .checked_mul(queue_cap)
            .and_then(|value| value.checked_mul(4))
            .and_then(|value| value.checked_mul(std::mem::size_of::<f64>()))
            .ok_or_else(|| {
                CudaAdaptiveSchaffTrendCycleError::InvalidInput("scratch val overflow".into())
            })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                CudaAdaptiveSchaffTrendCycleError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(param_bytes)
            .and_then(|value| value.checked_add(scratch_values_bytes))
            .and_then(|value| value.checked_add(scratch_idx_bytes))
            .and_then(|value| value.checked_add(scratch_val_bytes))
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaAdaptiveSchaffTrendCycleError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_close = DeviceBuffer::from_slice(close)?;
        let d_adaptive_lengths = DeviceBuffer::from_slice(&adaptive_lengths)?;
        let d_stc_lengths = DeviceBuffer::from_slice(&stc_lengths)?;
        let d_smoothing_factors = DeviceBuffer::from_slice(&smoothing_factors)?;
        let d_fast_lengths = DeviceBuffer::from_slice(&fast_lengths)?;
        let d_slow_lengths = DeviceBuffer::from_slice(&slow_lengths)?;
        let scratch_values_elems = rows * adaptive_cap;
        let scratch_idx_elems = rows * queue_cap * 4;
        let scratch_val_elems = rows * queue_cap * 4;
        let mut d_corr_values =
            unsafe { DeviceBuffer::<f64>::uninitialized(scratch_values_elems)? };
        let mut d_queue_idx = unsafe { DeviceBuffer::<i32>::uninitialized(scratch_idx_elems)? };
        let mut d_queue_val = unsafe { DeviceBuffer::<f64>::uninitialized(scratch_val_elems)? };
        let mut d_stc = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_histogram = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("adaptive_schaff_trend_cycle_batch_f64")
            .map_err(|_| CudaAdaptiveSchaffTrendCycleError::MissingKernelSymbol {
                name: "adaptive_schaff_trend_cycle_batch_f64",
            })?;
        let grid_x = ((rows as u32) + ADAPTIVE_SCHAFF_TREND_CYCLE_BLOCK_X - 1)
            / ADAPTIVE_SCHAFF_TREND_CYCLE_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(ADAPTIVE_SCHAFF_TREND_CYCLE_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_high.as_device_ptr(),
                d_low.as_device_ptr(),
                d_close.as_device_ptr(),
                cols as i32,
                d_adaptive_lengths.as_device_ptr(),
                d_stc_lengths.as_device_ptr(),
                d_smoothing_factors.as_device_ptr(),
                d_fast_lengths.as_device_ptr(),
                d_slow_lengths.as_device_ptr(),
                rows as i32,
                adaptive_cap as i32,
                stc_cap as i32,
                queue_cap as i32,
                d_corr_values.as_device_ptr(),
                d_queue_idx.as_device_ptr(),
                d_queue_val.as_device_ptr(),
                d_stc.as_device_ptr(),
                d_histogram.as_device_ptr()
            ))?;
        }
        self.stream.synchronize()?;

        Ok(CudaAdaptiveSchaffTrendCycleBatchResult {
            outputs: AdaptiveSchaffTrendCycleDeviceOutputs {
                stc: AdaptiveSchaffTrendCycleDeviceArrayF64 {
                    buf: d_stc,
                    rows,
                    cols,
                },
                histogram: AdaptiveSchaffTrendCycleDeviceArrayF64 {
                    buf: d_histogram,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
