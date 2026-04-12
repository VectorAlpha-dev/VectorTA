#![cfg(feature = "cuda")]

use crate::indicators::stochastic_distance::{
    StochasticDistanceBatchRange, StochasticDistanceParams,
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

const STOCHASTIC_DISTANCE_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaStochasticDistanceError {
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

pub struct StochasticDistanceDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl StochasticDistanceDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct StochasticDistanceDeviceArrayF64Pair {
    pub oscillator: StochasticDistanceDeviceArrayF64,
    pub signal: StochasticDistanceDeviceArrayF64,
}

impl StochasticDistanceDeviceArrayF64Pair {
    #[inline]
    pub fn rows(&self) -> usize {
        self.oscillator.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.oscillator.cols
    }
}

pub struct CudaStochasticDistanceBatchResult {
    pub outputs: StochasticDistanceDeviceArrayF64Pair,
    pub combos: Vec<StochasticDistanceParams>,
}

pub struct CudaStochasticDistance {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaStochasticDistance {
    pub fn new(device_id: usize) -> Result<Self, CudaStochasticDistanceError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("stochastic_distance_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaStochasticDistanceError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn count_valid_values(data: &[f64]) -> usize {
        data.iter().filter(|v| v.is_finite()).count()
    }

    fn first_valid_value(data: &[f64]) -> Option<usize> {
        data.iter().position(|value| value.is_finite())
    }

    fn expand_axis_usize(
        start: usize,
        end: usize,
        step: usize,
    ) -> Result<Vec<usize>, CudaStochasticDistanceError> {
        if start > end {
            return Err(CudaStochasticDistanceError::InvalidInput(format!(
                "invalid range: start={start}, end={end}, step={step}"
            )));
        }
        if start == end {
            if step != 0 {
                return Err(CudaStochasticDistanceError::InvalidInput(format!(
                    "invalid range: start={start}, end={end}, step={step}"
                )));
            }
            return Ok(vec![start]);
        }
        if step == 0 {
            return Err(CudaStochasticDistanceError::InvalidInput(format!(
                "invalid range: start={start}, end={end}, step={step}"
            )));
        }
        let mut out = Vec::new();
        let mut value = start;
        while value <= end {
            out.push(value);
            match value.checked_add(step) {
                Some(next) => value = next,
                None => break,
            }
        }
        if *out.last().unwrap_or(&start) != end {
            return Err(CudaStochasticDistanceError::InvalidInput(format!(
                "invalid range: start={start}, end={end}, step={step}"
            )));
        }
        Ok(out)
    }

    fn expand_axis_i32(
        start: i32,
        end: i32,
        step: i32,
    ) -> Result<Vec<i32>, CudaStochasticDistanceError> {
        if start > end {
            return Err(CudaStochasticDistanceError::InvalidInput(format!(
                "invalid range: start={start}, end={end}, step={step}"
            )));
        }
        if start == end {
            if step != 0 {
                return Err(CudaStochasticDistanceError::InvalidInput(format!(
                    "invalid range: start={start}, end={end}, step={step}"
                )));
            }
            return Ok(vec![start]);
        }
        if step <= 0 {
            return Err(CudaStochasticDistanceError::InvalidInput(format!(
                "invalid range: start={start}, end={end}, step={step}"
            )));
        }
        let mut out = Vec::new();
        let mut value = start;
        while value <= end {
            out.push(value);
            match value.checked_add(step) {
                Some(next) => value = next,
                None => break,
            }
        }
        if *out.last().unwrap_or(&start) != end {
            return Err(CudaStochasticDistanceError::InvalidInput(format!(
                "invalid range: start={start}, end={end}, step={step}"
            )));
        }
        Ok(out)
    }

    fn expand_grid(
        range: &StochasticDistanceBatchRange,
    ) -> Result<Vec<StochasticDistanceParams>, CudaStochasticDistanceError> {
        let lookbacks = Self::expand_axis_usize(
            range.lookback_length.0,
            range.lookback_length.1,
            range.lookback_length.2,
        )?;
        let length1s = Self::expand_axis_usize(range.length1.0, range.length1.1, range.length1.2)?;
        let length2s = Self::expand_axis_usize(range.length2.0, range.length2.1, range.length2.2)?;
        let ob_levels =
            Self::expand_axis_i32(range.ob_level.0, range.ob_level.1, range.ob_level.2)?;
        let os_levels =
            Self::expand_axis_i32(range.os_level.0, range.os_level.1, range.os_level.2)?;

        let mut combos = Vec::with_capacity(
            lookbacks.len() * length1s.len() * length2s.len() * ob_levels.len() * os_levels.len(),
        );
        for &lookback_length in &lookbacks {
            for &length1 in &length1s {
                for &length2 in &length2s {
                    for &ob_level in &ob_levels {
                        for &os_level in &os_levels {
                            if lookback_length == 0 || length1 == 0 || length2 == 0 {
                                return Err(CudaStochasticDistanceError::InvalidInput(
                                    "invalid length parameters".into(),
                                ));
                            }
                            if !(0..=100).contains(&ob_level) {
                                return Err(CudaStochasticDistanceError::InvalidInput(format!(
                                    "invalid ob_level: {ob_level}"
                                )));
                            }
                            if !(-100..=0).contains(&os_level) {
                                return Err(CudaStochasticDistanceError::InvalidInput(format!(
                                    "invalid os_level: {os_level}"
                                )));
                            }
                            if os_level >= ob_level {
                                return Err(CudaStochasticDistanceError::InvalidInput(format!(
                                    "invalid thresholds: os_level ({os_level}) must be less than ob_level ({ob_level})"
                                )));
                            }
                            combos.push(StochasticDistanceParams {
                                lookback_length: Some(lookback_length),
                                length1: Some(length1),
                                length2: Some(length2),
                                ob_level: Some(ob_level),
                                os_level: Some(os_level),
                            });
                        }
                    }
                }
            }
        }
        Ok(combos)
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaStochasticDistanceError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaStochasticDistanceError::OutOfMemory {
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
    ) -> Result<(), CudaStochasticDistanceError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if (threads > max_threads) || (grid.x > max_grid_x) {
            return Err(CudaStochasticDistanceError::LaunchConfigTooLarge {
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
        sweep: &StochasticDistanceBatchRange,
    ) -> Result<CudaStochasticDistanceBatchResult, CudaStochasticDistanceError> {
        if data.is_empty() {
            return Err(CudaStochasticDistanceError::InvalidInput(
                "empty input".into(),
            ));
        }
        let _first = Self::first_valid_value(data).ok_or_else(|| {
            CudaStochasticDistanceError::InvalidInput("all values are NaN".into())
        })?;
        let valid = Self::count_valid_values(data);

        let combos = Self::expand_grid(sweep)?;
        let rows = combos.len();
        let cols = data.len();
        let mut max_lookback = 0usize;
        let mut max_length1 = 0usize;
        let mut max_needed = 0usize;
        let mut lookbacks = Vec::with_capacity(rows);
        let mut length1s = Vec::with_capacity(rows);
        let mut length2s = Vec::with_capacity(rows);
        let mut ob_levels = Vec::with_capacity(rows);
        let mut os_levels = Vec::with_capacity(rows);

        for combo in &combos {
            let lookback = combo.lookback_length.unwrap_or(200);
            let length1 = combo.length1.unwrap_or(12);
            let length2 = combo.length2.unwrap_or(3);
            let ob_level = combo.ob_level.unwrap_or(40);
            let os_level = combo.os_level.unwrap_or(-40);
            if lookback > data.len() || length1 > data.len() {
                return Err(CudaStochasticDistanceError::InvalidInput(
                    "invalid lengths for data size".into(),
                ));
            }
            max_lookback = max_lookback.max(lookback);
            max_length1 = max_length1.max(length1);
            max_needed = max_needed.max(lookback + length1);
            lookbacks.push(lookback as i32);
            length1s.push(length1 as i32);
            length2s.push(length2 as i32);
            ob_levels.push(ob_level);
            os_levels.push(os_level);
        }

        if valid < max_needed {
            return Err(CudaStochasticDistanceError::InvalidInput(format!(
                "not enough valid data: needed={max_needed}, valid={valid}"
            )));
        }

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaStochasticDistanceError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = lookbacks
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| {
                length1s
                    .len()
                    .checked_mul(std::mem::size_of::<i32>())
                    .and_then(|other| value.checked_add(other))
            })
            .and_then(|value| {
                length2s
                    .len()
                    .checked_mul(std::mem::size_of::<i32>())
                    .and_then(|other| value.checked_add(other))
            })
            .and_then(|value| {
                ob_levels
                    .len()
                    .checked_mul(std::mem::size_of::<i32>())
                    .and_then(|other| value.checked_add(other))
            })
            .and_then(|value| {
                os_levels
                    .len()
                    .checked_mul(std::mem::size_of::<i32>())
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| {
                CudaStochasticDistanceError::InvalidInput("params bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaStochasticDistanceError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                CudaStochasticDistanceError::InvalidInput("output bytes overflow".into())
            })?;
        let close_scratch = rows.checked_mul(max_length1).ok_or_else(|| {
            CudaStochasticDistanceError::InvalidInput("close scratch overflow".into())
        })?;
        let dist_scratch = rows.checked_mul(max_lookback).ok_or_else(|| {
            CudaStochasticDistanceError::InvalidInput("distance scratch overflow".into())
        })?;
        let scratch_bytes = close_scratch
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| {
                dist_scratch
                    .checked_mul(std::mem::size_of::<f64>())
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| {
                CudaStochasticDistanceError::InvalidInput("scratch bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .and_then(|value| value.checked_add(scratch_bytes))
            .ok_or_else(|| {
                CudaStochasticDistanceError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_data = DeviceBuffer::from_slice(data)?;
        let d_lookbacks = DeviceBuffer::from_slice(&lookbacks)?;
        let d_length1s = DeviceBuffer::from_slice(&length1s)?;
        let d_length2s = DeviceBuffer::from_slice(&length2s)?;
        let d_ob_levels = DeviceBuffer::from_slice(&ob_levels)?;
        let d_os_levels = DeviceBuffer::from_slice(&os_levels)?;
        let d_close_buffer = unsafe { DeviceBuffer::<f64>::uninitialized(close_scratch)? };
        let d_distance_buffer = unsafe { DeviceBuffer::<f64>::uninitialized(dist_scratch)? };
        let d_out_oscillator = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_signal = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("stochastic_distance_batch_f64")
            .map_err(|_| CudaStochasticDistanceError::MissingKernelSymbol {
                name: "stochastic_distance_batch_f64",
            })?;
        let grid_x =
            ((rows as u32) + STOCHASTIC_DISTANCE_BLOCK_X - 1) / STOCHASTIC_DISTANCE_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(STOCHASTIC_DISTANCE_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_data.as_device_ptr(),
                cols as i32,
                d_lookbacks.as_device_ptr(),
                d_length1s.as_device_ptr(),
                d_length2s.as_device_ptr(),
                d_ob_levels.as_device_ptr(),
                d_os_levels.as_device_ptr(),
                rows as i32,
                max_lookback as i32,
                max_length1 as i32,
                d_close_buffer.as_device_ptr(),
                d_distance_buffer.as_device_ptr(),
                d_out_oscillator.as_device_ptr(),
                d_out_signal.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaStochasticDistanceBatchResult {
            outputs: StochasticDistanceDeviceArrayF64Pair {
                oscillator: StochasticDistanceDeviceArrayF64 {
                    buf: d_out_oscillator,
                    rows,
                    cols,
                },
                signal: StochasticDistanceDeviceArrayF64 {
                    buf: d_out_signal,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
