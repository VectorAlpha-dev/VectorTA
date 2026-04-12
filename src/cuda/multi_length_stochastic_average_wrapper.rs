#![cfg(feature = "cuda")]

use crate::indicators::multi_length_stochastic_average::{
    MultiLengthStochasticAverageBatchRange, MultiLengthStochasticAverageParams,
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

const MULTI_LENGTH_STOCHASTIC_AVERAGE_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const DEFAULT_LENGTH: usize = 14;
const DEFAULT_PRESMOOTH: usize = 10;
const DEFAULT_POSTSMOOTH: usize = 10;
const DEFAULT_SMOOTHING_METHOD: &str = "sma";
const MIN_STOCH_LENGTH: usize = 4;
const METHOD_NONE: i32 = 0;
const METHOD_SMA: i32 = 1;
const METHOD_TMA: i32 = 2;
const METHOD_LSMA: i32 = 3;

#[derive(Debug, Error)]
pub enum CudaMultiLengthStochasticAverageError {
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

pub struct MultiLengthStochasticAverageDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl MultiLengthStochasticAverageDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct CudaMultiLengthStochasticAverageBatchResult {
    pub outputs: MultiLengthStochasticAverageDeviceArrayF64,
    pub combos: Vec<MultiLengthStochasticAverageParams>,
}

pub struct CudaMultiLengthStochasticAverage {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

#[inline]
fn canonical_method_name(name: Option<&str>) -> String {
    name.unwrap_or(DEFAULT_SMOOTHING_METHOD)
        .to_ascii_lowercase()
}

#[inline]
fn parse_method_code(name: &str) -> Option<i32> {
    if name.eq_ignore_ascii_case("none") {
        Some(METHOD_NONE)
    } else if name.eq_ignore_ascii_case("sma") {
        Some(METHOD_SMA)
    } else if name.eq_ignore_ascii_case("tma") {
        Some(METHOD_TMA)
    } else if name.eq_ignore_ascii_case("lsma") {
        Some(METHOD_LSMA)
    } else {
        None
    }
}

#[inline]
fn smoothing_warmup(method: i32, length: usize) -> usize {
    match method {
        METHOD_NONE => 0,
        METHOD_SMA | METHOD_LSMA => length.saturating_sub(1),
        METHOD_TMA => length.saturating_sub(1).saturating_mul(2),
        _ => 0,
    }
}

#[inline]
fn total_warmup(
    length: usize,
    presmooth: usize,
    premethod: i32,
    postsmooth: usize,
    postmethod: i32,
) -> usize {
    smoothing_warmup(premethod, presmooth)
        + length.saturating_sub(1)
        + smoothing_warmup(postmethod, postsmooth)
}

#[inline]
fn smoothing_scratch_capacity(method: i32, length: usize) -> usize {
    match method {
        METHOD_NONE => 0,
        METHOD_SMA | METHOD_LSMA => length,
        METHOD_TMA => length.saturating_mul(2),
        _ => 0,
    }
}

fn expand_axis_usize(
    start: usize,
    end: usize,
    step: usize,
) -> Result<Vec<usize>, CudaMultiLengthStochasticAverageError> {
    if start == end {
        return Ok(vec![start]);
    }
    if step == 0 {
        return Err(CudaMultiLengthStochasticAverageError::InvalidInput(
            format!("invalid range: start={start}, end={end}, step={step}"),
        ));
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
        while value >= end {
            out.push(value);
            let next = value.saturating_sub(step);
            if next == value {
                break;
            }
            value = next;
        }
    }

    if out.is_empty() {
        return Err(CudaMultiLengthStochasticAverageError::InvalidInput(
            format!("invalid range: start={start}, end={end}, step={step}"),
        ));
    }

    Ok(out)
}

fn expand_grid(
    range: &MultiLengthStochasticAverageBatchRange,
) -> Result<Vec<MultiLengthStochasticAverageParams>, CudaMultiLengthStochasticAverageError> {
    let lengths = expand_axis_usize(range.length.0, range.length.1, range.length.2)?;
    let presmooths = expand_axis_usize(range.presmooth.0, range.presmooth.1, range.presmooth.2)?;
    let postsmooths =
        expand_axis_usize(range.postsmooth.0, range.postsmooth.1, range.postsmooth.2)?;
    let premethod = canonical_method_name(range.premethod.as_deref());
    let postmethod = canonical_method_name(range.postmethod.as_deref());

    let mut combos = Vec::with_capacity(
        lengths
            .len()
            .saturating_mul(presmooths.len())
            .saturating_mul(postsmooths.len()),
    );

    for &length in &lengths {
        for &presmooth in &presmooths {
            for &postsmooth in &postsmooths {
                combos.push(MultiLengthStochasticAverageParams {
                    length: Some(length),
                    presmooth: Some(presmooth),
                    premethod: Some(premethod.clone()),
                    postsmooth: Some(postsmooth),
                    postmethod: Some(postmethod.clone()),
                });
            }
        }
    }

    if combos.is_empty() {
        return Err(CudaMultiLengthStochasticAverageError::InvalidInput(
            "empty parameter grid".into(),
        ));
    }

    Ok(combos)
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

impl CudaMultiLengthStochasticAverage {
    pub fn new(device_id: usize) -> Result<Self, CudaMultiLengthStochasticAverageError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("multi_length_stochastic_average_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaMultiLengthStochasticAverageError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn will_fit(
        required: usize,
        headroom: usize,
    ) -> Result<(), CudaMultiLengthStochasticAverageError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaMultiLengthStochasticAverageError::OutOfMemory {
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
    ) -> Result<(), CudaMultiLengthStochasticAverageError> {
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
                CudaMultiLengthStochasticAverageError::LaunchConfigTooLarge {
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
        sweep: &MultiLengthStochasticAverageBatchRange,
    ) -> Result<CudaMultiLengthStochasticAverageBatchResult, CudaMultiLengthStochasticAverageError>
    {
        if data.is_empty() {
            return Err(CudaMultiLengthStochasticAverageError::InvalidInput(
                "empty input".into(),
            ));
        }
        if !data.iter().any(|value| value.is_finite()) {
            return Err(CudaMultiLengthStochasticAverageError::InvalidInput(
                "all values are NaN".into(),
            ));
        }

        let combos = expand_grid(sweep)?;
        let rows = combos.len();
        let cols = data.len();
        let max_valid = max_consecutive_valid_values(data);

        let mut lengths = Vec::with_capacity(rows);
        let mut presmooths = Vec::with_capacity(rows);
        let mut postsmooths = Vec::with_capacity(rows);
        let mut premethods = Vec::with_capacity(rows);
        let mut postmethods = Vec::with_capacity(rows);
        let mut scratch_cap = 1usize;

        for combo in &combos {
            let length = combo.length.unwrap_or(DEFAULT_LENGTH);
            let presmooth = combo.presmooth.unwrap_or(DEFAULT_PRESMOOTH);
            let postsmooth = combo.postsmooth.unwrap_or(DEFAULT_POSTSMOOTH);
            let premethod_name = canonical_method_name(combo.premethod.as_deref());
            let postmethod_name = canonical_method_name(combo.postmethod.as_deref());
            let premethod = parse_method_code(&premethod_name).ok_or_else(|| {
                CudaMultiLengthStochasticAverageError::InvalidInput(format!(
                    "invalid premethod: {premethod_name}"
                ))
            })?;
            let postmethod = parse_method_code(&postmethod_name).ok_or_else(|| {
                CudaMultiLengthStochasticAverageError::InvalidInput(format!(
                    "invalid postmethod: {postmethod_name}"
                ))
            })?;

            if length < MIN_STOCH_LENGTH || length > cols {
                return Err(CudaMultiLengthStochasticAverageError::InvalidInput(
                    format!("invalid length: length={length}, data_len={cols}"),
                ));
            }
            if presmooth == 0 {
                return Err(CudaMultiLengthStochasticAverageError::InvalidInput(
                    "invalid presmooth: 0".into(),
                ));
            }
            if postsmooth == 0 {
                return Err(CudaMultiLengthStochasticAverageError::InvalidInput(
                    "invalid postsmooth: 0".into(),
                ));
            }

            let needed = total_warmup(length, presmooth, premethod, postsmooth, postmethod) + 1;
            if max_valid < needed {
                return Err(CudaMultiLengthStochasticAverageError::InvalidInput(
                    format!("not enough valid data: needed={needed}, valid={max_valid}"),
                ));
            }

            scratch_cap = scratch_cap.max(
                smoothing_scratch_capacity(premethod, presmooth)
                    .saturating_add(smoothing_scratch_capacity(postmethod, postsmooth))
                    .saturating_add(length),
            );

            lengths.push(i32::try_from(length).map_err(|_| {
                CudaMultiLengthStochasticAverageError::InvalidInput(format!(
                    "length out of range: {length}"
                ))
            })?);
            presmooths.push(i32::try_from(presmooth).map_err(|_| {
                CudaMultiLengthStochasticAverageError::InvalidInput(format!(
                    "presmooth out of range: {presmooth}"
                ))
            })?);
            postsmooths.push(i32::try_from(postsmooth).map_err(|_| {
                CudaMultiLengthStochasticAverageError::InvalidInput(format!(
                    "postsmooth out of range: {postsmooth}"
                ))
            })?);
            premethods.push(premethod);
            postmethods.push(postmethod);
        }

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaMultiLengthStochasticAverageError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = rows
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| value.checked_mul(5))
            .ok_or_else(|| {
                CudaMultiLengthStochasticAverageError::InvalidInput("params bytes overflow".into())
            })?;
        let scratch_elems = rows.checked_mul(scratch_cap).ok_or_else(|| {
            CudaMultiLengthStochasticAverageError::InvalidInput("scratch elems overflow".into())
        })?;
        let scratch_bytes = scratch_elems
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaMultiLengthStochasticAverageError::InvalidInput("scratch bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaMultiLengthStochasticAverageError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaMultiLengthStochasticAverageError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(scratch_bytes))
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaMultiLengthStochasticAverageError::InvalidInput(
                    "required bytes overflow".into(),
                )
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_data = DeviceBuffer::from_slice(data)?;
        let d_lengths = DeviceBuffer::from_slice(&lengths)?;
        let d_presmooths = DeviceBuffer::from_slice(&presmooths)?;
        let d_postsmooths = DeviceBuffer::from_slice(&postsmooths)?;
        let d_premethods = DeviceBuffer::from_slice(&premethods)?;
        let d_postmethods = DeviceBuffer::from_slice(&postmethods)?;
        let d_scratch = unsafe { DeviceBuffer::<f64>::uninitialized(scratch_elems)? };
        let d_out_values = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("multi_length_stochastic_average_batch_f64")
            .map_err(
                |_| CudaMultiLengthStochasticAverageError::MissingKernelSymbol {
                    name: "multi_length_stochastic_average_batch_f64",
                },
            )?;
        let grid_x = ((rows as u32) + MULTI_LENGTH_STOCHASTIC_AVERAGE_BLOCK_X - 1)
            / MULTI_LENGTH_STOCHASTIC_AVERAGE_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(MULTI_LENGTH_STOCHASTIC_AVERAGE_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_data.as_device_ptr(),
                cols as i32,
                d_lengths.as_device_ptr(),
                d_presmooths.as_device_ptr(),
                d_postsmooths.as_device_ptr(),
                d_premethods.as_device_ptr(),
                d_postmethods.as_device_ptr(),
                rows as i32,
                scratch_cap as i32,
                d_scratch.as_device_ptr(),
                d_out_values.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaMultiLengthStochasticAverageBatchResult {
            outputs: MultiLengthStochasticAverageDeviceArrayF64 {
                buf: d_out_values,
                rows,
                cols,
            },
            combos,
        })
    }
}
