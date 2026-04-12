#![cfg(feature = "cuda")]

use crate::indicators::ehlers_autocorrelation_periodogram::{
    EhlersAutocorrelationPeriodogramBatchRange, EhlersAutocorrelationPeriodogramParams,
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

const EHLERS_AUTOCORRELATION_PERIODOGRAM_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const DEFAULT_MIN_PERIOD: usize = 8;
const DEFAULT_MAX_PERIOD: usize = 48;
const DEFAULT_AVG_LENGTH: usize = 3;

#[derive(Debug, Error)]
pub enum CudaEhlersAutocorrelationPeriodogramError {
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

pub struct EhlersAutocorrelationPeriodogramDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl EhlersAutocorrelationPeriodogramDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct EhlersAutocorrelationPeriodogramDeviceArrayF64Pair {
    pub dominant_cycle: EhlersAutocorrelationPeriodogramDeviceArrayF64,
    pub normalized_power: EhlersAutocorrelationPeriodogramDeviceArrayF64,
}

impl EhlersAutocorrelationPeriodogramDeviceArrayF64Pair {
    #[inline]
    pub fn rows(&self) -> usize {
        self.dominant_cycle.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.dominant_cycle.cols
    }
}

pub struct CudaEhlersAutocorrelationPeriodogramBatchResult {
    pub outputs: EhlersAutocorrelationPeriodogramDeviceArrayF64Pair,
    pub combos: Vec<EhlersAutocorrelationPeriodogramParams>,
}

pub struct CudaEhlersAutocorrelationPeriodogram {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

#[inline]
fn axis_usize(
    (start, end, step): (usize, usize, usize),
) -> Result<Vec<usize>, CudaEhlersAutocorrelationPeriodogramError> {
    if step == 0 || start == end {
        return Ok(vec![start]);
    }
    let step = step.max(1);
    if start < end {
        let mut out = Vec::new();
        let mut value = start;
        while value <= end {
            out.push(value);
            match value.checked_add(step) {
                Some(next) if next != value => value = next,
                _ => break,
            }
        }
        if out.is_empty() {
            return Err(CudaEhlersAutocorrelationPeriodogramError::InvalidInput(
                format!("invalid range: start={start}, end={end}, step={step}"),
            ));
        }
        Ok(out)
    } else {
        let mut out = Vec::new();
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
        if out.is_empty() {
            return Err(CudaEhlersAutocorrelationPeriodogramError::InvalidInput(
                format!("invalid range: start={start}, end={end}, step={step}"),
            ));
        }
        Ok(out)
    }
}

#[inline]
fn corr_window(avg_length: usize, lag: usize) -> usize {
    if avg_length == 0 {
        lag.max(2)
    } else {
        avg_length.max(2)
    }
}

#[inline]
fn warmup_period(max_period: usize, avg_length: usize) -> usize {
    max_period + corr_window(avg_length, max_period) - 1
}

#[inline]
fn expand_grid(
    range: &EhlersAutocorrelationPeriodogramBatchRange,
) -> Result<Vec<EhlersAutocorrelationPeriodogramParams>, CudaEhlersAutocorrelationPeriodogramError>
{
    let mins = axis_usize(range.min_period)?;
    let maxes = axis_usize(range.max_period)?;
    let avgs = axis_usize(range.avg_length)?;
    let mut out = Vec::with_capacity(mins.len() * maxes.len() * avgs.len());
    for &min_period in &mins {
        for &max_period in &maxes {
            for &avg_length in &avgs {
                out.push(EhlersAutocorrelationPeriodogramParams {
                    min_period: Some(min_period),
                    max_period: Some(max_period),
                    avg_length: Some(avg_length),
                    enhance: Some(range.enhance),
                });
            }
        }
    }
    Ok(out)
}

impl CudaEhlersAutocorrelationPeriodogram {
    pub fn new(device_id: usize) -> Result<Self, CudaEhlersAutocorrelationPeriodogramError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module =
            crate::load_cuda_embedded_module!("ehlers_autocorrelation_periodogram_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaEhlersAutocorrelationPeriodogramError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn validate_launch(
        &self,
        grid: GridSize,
        block: BlockSize,
    ) -> Result<(), CudaEhlersAutocorrelationPeriodogramError> {
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
                CudaEhlersAutocorrelationPeriodogramError::LaunchConfigTooLarge {
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

    fn will_fit(
        required: usize,
        headroom: usize,
    ) -> Result<(), CudaEhlersAutocorrelationPeriodogramError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaEhlersAutocorrelationPeriodogramError::OutOfMemory {
                    required,
                    free,
                    headroom,
                });
            }
        }
        Ok(())
    }

    pub fn batch_dev(
        &self,
        data: &[f64],
        sweep: &EhlersAutocorrelationPeriodogramBatchRange,
    ) -> Result<
        CudaEhlersAutocorrelationPeriodogramBatchResult,
        CudaEhlersAutocorrelationPeriodogramError,
    > {
        if data.is_empty() {
            return Err(CudaEhlersAutocorrelationPeriodogramError::InvalidInput(
                "empty input".into(),
            ));
        }

        let first = data
            .iter()
            .position(|value| value.is_finite())
            .ok_or_else(|| {
                CudaEhlersAutocorrelationPeriodogramError::InvalidInput("all values are NaN".into())
            })?;
        let valid = data.iter().filter(|value| value.is_finite()).count();
        let combos = expand_grid(sweep)?;
        if combos.is_empty() {
            return Err(CudaEhlersAutocorrelationPeriodogramError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let rows = combos.len();
        let cols = data.len();
        let mut min_periods = Vec::with_capacity(rows);
        let mut max_periods = Vec::with_capacity(rows);
        let mut avg_lengths = Vec::with_capacity(rows);
        let mut enhances = Vec::with_capacity(rows);
        let mut max_history_cap = 0usize;
        let mut max_period_cap = 0usize;

        for params in &combos {
            let min_period = params.min_period.unwrap_or(DEFAULT_MIN_PERIOD);
            let max_period = params.max_period.unwrap_or(DEFAULT_MAX_PERIOD);
            let avg_length = params.avg_length.unwrap_or(DEFAULT_AVG_LENGTH);
            if min_period < 3 {
                return Err(CudaEhlersAutocorrelationPeriodogramError::InvalidInput(
                    format!("invalid min_period: {min_period}"),
                ));
            }
            if max_period <= min_period {
                return Err(CudaEhlersAutocorrelationPeriodogramError::InvalidInput(
                    format!(
                        "invalid period order: min_period={min_period}, max_period={max_period}"
                    ),
                ));
            }
            if max_period > cols {
                return Err(CudaEhlersAutocorrelationPeriodogramError::InvalidInput(
                    format!("invalid max_period: max_period={max_period}, data_len={cols}"),
                ));
            }

            let needed = warmup_period(max_period, avg_length) + 1;
            if valid < needed {
                return Err(CudaEhlersAutocorrelationPeriodogramError::InvalidInput(
                    format!("not enough valid data: needed={needed}, valid={valid}"),
                ));
            }

            let history_cap = max_period + corr_window(avg_length, max_period);
            max_history_cap = max_history_cap.max(history_cap);
            max_period_cap = max_period_cap.max(max_period + 1);
            min_periods.push(min_period as i32);
            max_periods.push(max_period as i32);
            avg_lengths.push(avg_length as i32);
            enhances.push(i32::from(params.enhance.unwrap_or(true)));
        }

        let _ = first;
        let scratch_cap = max_history_cap
            .checked_add(max_period_cap.checked_mul(3).ok_or_else(|| {
                CudaEhlersAutocorrelationPeriodogramError::InvalidInput(
                    "scratch capacity overflow".into(),
                )
            })?)
            .ok_or_else(|| {
                CudaEhlersAutocorrelationPeriodogramError::InvalidInput(
                    "scratch capacity overflow".into(),
                )
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaEhlersAutocorrelationPeriodogramError::InvalidInput("rows*cols overflow".into())
        })?;

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaEhlersAutocorrelationPeriodogramError::InvalidInput(
                    "input bytes overflow".into(),
                )
            })?;
        let param_bytes = rows
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| {
                CudaEhlersAutocorrelationPeriodogramError::InvalidInput(
                    "param bytes overflow".into(),
                )
            })?;
        let scratch_bytes = rows
            .checked_mul(scratch_cap)
            .and_then(|value| value.checked_mul(std::mem::size_of::<f64>()))
            .ok_or_else(|| {
                CudaEhlersAutocorrelationPeriodogramError::InvalidInput(
                    "scratch bytes overflow".into(),
                )
            })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                CudaEhlersAutocorrelationPeriodogramError::InvalidInput(
                    "output bytes overflow".into(),
                )
            })?;
        let required = input_bytes
            .checked_add(param_bytes)
            .and_then(|value| value.checked_add(scratch_bytes))
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaEhlersAutocorrelationPeriodogramError::InvalidInput(
                    "required bytes overflow".into(),
                )
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_data = DeviceBuffer::from_slice(data)?;
        let d_min_periods = DeviceBuffer::from_slice(&min_periods)?;
        let d_max_periods = DeviceBuffer::from_slice(&max_periods)?;
        let d_avg_lengths = DeviceBuffer::from_slice(&avg_lengths)?;
        let d_enhances = DeviceBuffer::from_slice(&enhances)?;
        let mut d_scratch = unsafe { DeviceBuffer::<f64>::uninitialized(rows * scratch_cap)? };
        let mut d_dominant_cycle = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_normalized_power = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("ehlers_autocorrelation_periodogram_batch_f64")
            .map_err(
                |_| CudaEhlersAutocorrelationPeriodogramError::MissingKernelSymbol {
                    name: "ehlers_autocorrelation_periodogram_batch_f64",
                },
            )?;

        let grid_x = ((rows as u32) + EHLERS_AUTOCORRELATION_PERIODOGRAM_BLOCK_X - 1)
            / EHLERS_AUTOCORRELATION_PERIODOGRAM_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(EHLERS_AUTOCORRELATION_PERIODOGRAM_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_data.as_device_ptr(),
                cols as i32,
                d_min_periods.as_device_ptr(),
                d_max_periods.as_device_ptr(),
                d_avg_lengths.as_device_ptr(),
                d_enhances.as_device_ptr(),
                rows as i32,
                scratch_cap as i32,
                d_scratch.as_device_ptr(),
                d_dominant_cycle.as_device_ptr(),
                d_normalized_power.as_device_ptr()
            ))?;
        }
        self.stream.synchronize()?;

        Ok(CudaEhlersAutocorrelationPeriodogramBatchResult {
            outputs: EhlersAutocorrelationPeriodogramDeviceArrayF64Pair {
                dominant_cycle: EhlersAutocorrelationPeriodogramDeviceArrayF64 {
                    buf: d_dominant_cycle,
                    rows,
                    cols,
                },
                normalized_power: EhlersAutocorrelationPeriodogramDeviceArrayF64 {
                    buf: d_normalized_power,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
