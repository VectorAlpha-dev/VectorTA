#![cfg(feature = "cuda")]

use crate::indicators::hypertrend::{HyperTrendBatchRange, HyperTrendParams};
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

const HYPERTREND_BLOCK_X: u32 = 64;
const DEFAULT_FACTOR: f64 = 5.0;
const DEFAULT_SLOPE: f64 = 14.0;
const DEFAULT_WIDTH_PERCENT: f64 = 80.0;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaHyperTrendError {
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

pub struct HyperTrendDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl HyperTrendDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct HyperTrendDeviceArrayF64Quint {
    pub upper: HyperTrendDeviceArrayF64,
    pub average: HyperTrendDeviceArrayF64,
    pub lower: HyperTrendDeviceArrayF64,
    pub trend: HyperTrendDeviceArrayF64,
    pub changed: HyperTrendDeviceArrayF64,
}

impl HyperTrendDeviceArrayF64Quint {
    #[inline]
    pub fn rows(&self) -> usize {
        self.upper.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.upper.cols
    }
}

pub struct CudaHyperTrendBatchResult {
    pub outputs: HyperTrendDeviceArrayF64Quint,
    pub combos: Vec<HyperTrendParams>,
}

pub struct CudaHyperTrend {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

#[inline]
fn axis_f64((start, end, step): (f64, f64, f64)) -> Result<Vec<f64>, CudaHyperTrendError> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() {
        return Err(CudaHyperTrendError::InvalidInput(format!(
            "invalid float range: start={start}, end={end}, step={step}"
        )));
    }
    if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
        return Ok(vec![start]);
    }

    let step = step.abs();
    let mut out = Vec::new();
    if start <= end {
        let mut x = start;
        while x <= end + 1e-12 {
            out.push(x);
            x += step;
        }
    } else {
        let mut x = start;
        while x + 1e-12 >= end {
            out.push(x);
            x -= step;
        }
    }

    if out.is_empty() {
        return Err(CudaHyperTrendError::InvalidInput(format!(
            "invalid float range: start={start}, end={end}, step={step}"
        )));
    }
    Ok(out)
}

#[inline]
fn expand_grid_checked(
    range: &HyperTrendBatchRange,
) -> Result<Vec<HyperTrendParams>, CudaHyperTrendError> {
    let factors = axis_f64(range.factor)?;
    let slopes = axis_f64(range.slope)?;
    let widths = axis_f64(range.width_percent)?;
    let total = factors
        .len()
        .checked_mul(slopes.len())
        .and_then(|value| value.checked_mul(widths.len()))
        .ok_or_else(|| CudaHyperTrendError::InvalidInput("parameter grid overflow".into()))?;
    let mut out = Vec::with_capacity(total);
    for &factor in &factors {
        for &slope in &slopes {
            for &width_percent in &widths {
                out.push(HyperTrendParams {
                    factor: Some(factor),
                    slope: Some(slope),
                    width_percent: Some(width_percent),
                });
            }
        }
    }
    Ok(out)
}

#[inline]
fn valid_bar(high: f64, low: f64, source: f64) -> bool {
    high.is_finite() && low.is_finite() && source.is_finite() && high >= low
}

#[inline]
fn validate_params(factor: f64, slope: f64, width_percent: f64) -> Result<(), CudaHyperTrendError> {
    if !factor.is_finite() || factor <= 0.0 {
        return Err(CudaHyperTrendError::InvalidInput(format!(
            "invalid factor: {factor}"
        )));
    }
    if !slope.is_finite() || slope <= 0.0 {
        return Err(CudaHyperTrendError::InvalidInput(format!(
            "invalid slope: {slope}"
        )));
    }
    if !width_percent.is_finite() || !(0.0..=100.0).contains(&width_percent) {
        return Err(CudaHyperTrendError::InvalidInput(format!(
            "invalid width_percent: {width_percent}"
        )));
    }
    Ok(())
}

impl CudaHyperTrend {
    pub fn new(device_id: usize) -> Result<Self, CudaHyperTrendError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("hypertrend_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaHyperTrendError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaHyperTrendError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaHyperTrendError::OutOfMemory {
                    required,
                    free,
                    headroom,
                });
            }
        }
        Ok(())
    }

    fn validate_launch(&self, grid: GridSize, block: BlockSize) -> Result<(), CudaHyperTrendError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaHyperTrendError::LaunchConfigTooLarge {
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
        source: &[f64],
        sweep: &HyperTrendBatchRange,
    ) -> Result<CudaHyperTrendBatchResult, CudaHyperTrendError> {
        if high.is_empty() || low.is_empty() || source.is_empty() {
            return Err(CudaHyperTrendError::InvalidInput("empty input".into()));
        }
        if high.len() != low.len() || low.len() != source.len() {
            return Err(CudaHyperTrendError::InvalidInput(format!(
                "input length mismatch: high={}, low={}, source={}",
                high.len(),
                low.len(),
                source.len()
            )));
        }

        let combos = expand_grid_checked(sweep)?;
        for params in &combos {
            validate_params(
                params.factor.unwrap_or(DEFAULT_FACTOR),
                params.slope.unwrap_or(DEFAULT_SLOPE),
                params.width_percent.unwrap_or(DEFAULT_WIDTH_PERCENT),
            )?;
        }
        if !high
            .iter()
            .zip(low.iter())
            .zip(source.iter())
            .any(|((high, low), source)| valid_bar(*high, *low, *source))
        {
            return Err(CudaHyperTrendError::InvalidInput(
                "all values are NaN".into(),
            ));
        }

        let rows = combos.len();
        let cols = source.len();
        let factors: Vec<f64> = combos
            .iter()
            .map(|params| params.factor.unwrap_or(DEFAULT_FACTOR))
            .collect();
        let slopes: Vec<f64> = combos
            .iter()
            .map(|params| params.slope.unwrap_or(DEFAULT_SLOPE))
            .collect();
        let width_ratios: Vec<f64> = combos
            .iter()
            .map(|params| params.width_percent.unwrap_or(DEFAULT_WIDTH_PERCENT) * 0.01)
            .collect();

        let output_elems = rows
            .checked_mul(cols)
            .ok_or_else(|| CudaHyperTrendError::InvalidInput("rows*cols overflow".into()))?;
        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(3))
            .ok_or_else(|| CudaHyperTrendError::InvalidInput("input bytes overflow".into()))?;
        let param_bytes = rows
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(3))
            .ok_or_else(|| CudaHyperTrendError::InvalidInput("param bytes overflow".into()))?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(5))
            .ok_or_else(|| CudaHyperTrendError::InvalidInput("output bytes overflow".into()))?;
        let required = input_bytes
            .checked_add(param_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| CudaHyperTrendError::InvalidInput("required bytes overflow".into()))?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_source = DeviceBuffer::from_slice(source)?;
        let d_factors = DeviceBuffer::from_slice(&factors)?;
        let d_slopes = DeviceBuffer::from_slice(&slopes)?;
        let d_width_ratios = DeviceBuffer::from_slice(&width_ratios)?;
        let mut d_upper = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_average = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_lower = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_trend = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_changed = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("hypertrend_batch_f64")
            .map_err(|_| CudaHyperTrendError::MissingKernelSymbol {
                name: "hypertrend_batch_f64",
            })?;
        let grid_x = ((rows as u32) + HYPERTREND_BLOCK_X - 1) / HYPERTREND_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(HYPERTREND_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_high.as_device_ptr(),
                d_low.as_device_ptr(),
                d_source.as_device_ptr(),
                cols as i32,
                d_factors.as_device_ptr(),
                d_slopes.as_device_ptr(),
                d_width_ratios.as_device_ptr(),
                rows as i32,
                d_upper.as_device_ptr(),
                d_average.as_device_ptr(),
                d_lower.as_device_ptr(),
                d_trend.as_device_ptr(),
                d_changed.as_device_ptr()
            ))?;
        }
        self.stream.synchronize()?;

        Ok(CudaHyperTrendBatchResult {
            outputs: HyperTrendDeviceArrayF64Quint {
                upper: HyperTrendDeviceArrayF64 {
                    buf: d_upper,
                    rows,
                    cols,
                },
                average: HyperTrendDeviceArrayF64 {
                    buf: d_average,
                    rows,
                    cols,
                },
                lower: HyperTrendDeviceArrayF64 {
                    buf: d_lower,
                    rows,
                    cols,
                },
                trend: HyperTrendDeviceArrayF64 {
                    buf: d_trend,
                    rows,
                    cols,
                },
                changed: HyperTrendDeviceArrayF64 {
                    buf: d_changed,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
