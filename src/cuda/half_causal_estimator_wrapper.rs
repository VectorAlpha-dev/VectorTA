#![cfg(feature = "cuda")]

use crate::indicators::half_causal_estimator::{
    HalfCausalEstimatorBatchRange, HalfCausalEstimatorConfidenceAdjust,
    HalfCausalEstimatorKernelType, HalfCausalEstimatorParams,
};
use cust::context::Context;
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, GridSize};
use cust::launch;
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::Module;
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::f64::consts::PI;
use std::sync::Arc;
use thiserror::Error;

const HALF_CAUSAL_ESTIMATOR_BLOCK_X: u32 = 64;
const DEFAULT_DATA_PERIOD: usize = 5;
const DEFAULT_FILTER_LENGTH: usize = 20;
const DEFAULT_KERNEL_WIDTH: f64 = 20.0;
const DEFAULT_MAXIMUM_CONFIDENCE_ADJUST: f64 = 100.0;
const DEFAULT_ENABLE_EXPECTED_VALUE: bool = false;
const DEFAULT_EXTRA_SMOOTHING: usize = 0;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaHalfCausalEstimatorError {
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

pub struct HalfCausalEstimatorDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl HalfCausalEstimatorDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct HalfCausalEstimatorDeviceOutputs {
    pub estimate: HalfCausalEstimatorDeviceArrayF64,
    pub expected_value: HalfCausalEstimatorDeviceArrayF64,
}

impl HalfCausalEstimatorDeviceOutputs {
    #[inline]
    pub fn rows(&self) -> usize {
        self.estimate.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.estimate.cols
    }
}

pub struct CudaHalfCausalEstimatorBatchResult {
    pub outputs: HalfCausalEstimatorDeviceOutputs,
    pub combos: Vec<HalfCausalEstimatorParams>,
}

pub struct CudaHalfCausalEstimator {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

#[derive(Clone, Copy)]
struct ResolvedHalfCausalParams {
    slots_per_day: usize,
    data_period: usize,
    filter_length: usize,
    real_filter_length: usize,
    window_size: usize,
    kernel_width: f64,
    kernel_type: HalfCausalEstimatorKernelType,
    confidence_adjust: HalfCausalEstimatorConfidenceAdjust,
    maximum_confidence_adjust_factor: f64,
    enable_expected_value: bool,
    extra_smoothing: usize,
}

#[inline]
fn axis_usize(
    (start, end, step): (usize, usize, usize),
) -> Result<Vec<usize>, CudaHalfCausalEstimatorError> {
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
        return Err(CudaHalfCausalEstimatorError::InvalidInput(format!(
            "invalid range: start={start}, end={end}, step={step}"
        )));
    }
    Ok(out)
}

#[inline]
fn axis_f64((start, end, step): (f64, f64, f64)) -> Result<Vec<f64>, CudaHalfCausalEstimatorError> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() {
        return Err(CudaHalfCausalEstimatorError::InvalidInput(format!(
            "invalid range: start={start}, end={end}, step={step}"
        )));
    }
    if step == 0.0 || (start - end).abs() <= f64::EPSILON {
        return Ok(vec![start]);
    }
    let mut out = Vec::new();
    if start < end {
        if step <= 0.0 {
            return Err(CudaHalfCausalEstimatorError::InvalidInput(format!(
                "invalid range: start={start}, end={end}, step={step}"
            )));
        }
        let mut value = start;
        while value <= end + 1e-12 {
            out.push(value);
            value += step;
        }
    } else {
        let step = step.abs();
        let mut value = start;
        while value >= end - 1e-12 {
            out.push(value);
            value -= step;
        }
    }
    if out.is_empty() {
        return Err(CudaHalfCausalEstimatorError::InvalidInput(format!(
            "invalid range: start={start}, end={end}, step={step}"
        )));
    }
    Ok(out)
}

#[inline]
fn expand_grid_checked(
    sweep: &HalfCausalEstimatorBatchRange,
) -> Result<Vec<HalfCausalEstimatorParams>, CudaHalfCausalEstimatorError> {
    let data_periods = axis_usize(sweep.data_period)?;
    let filter_lengths = axis_usize(sweep.filter_length)?;
    let kernel_widths = axis_f64(sweep.kernel_width)?;
    let maximum_confidence_adjusts = axis_f64(sweep.maximum_confidence_adjust)?;
    let extra_smoothings = axis_usize(sweep.extra_smoothing)?;

    let mut combos = Vec::new();
    for data_period in data_periods {
        for &filter_length in &filter_lengths {
            for &kernel_width in &kernel_widths {
                for &maximum_confidence_adjust in &maximum_confidence_adjusts {
                    for &extra_smoothing in &extra_smoothings {
                        combos.push(HalfCausalEstimatorParams {
                            slots_per_day: sweep.slots_per_day,
                            data_period: Some(data_period),
                            filter_length: Some(filter_length),
                            kernel_width: Some(kernel_width),
                            kernel_type: Some(sweep.kernel_type),
                            confidence_adjust: Some(sweep.confidence_adjust),
                            maximum_confidence_adjust: Some(maximum_confidence_adjust),
                            enable_expected_value: Some(sweep.enable_expected_value),
                            extra_smoothing: Some(extra_smoothing),
                        });
                    }
                }
            }
        }
    }
    Ok(combos)
}

#[inline]
fn resolve_params(
    params: &HalfCausalEstimatorParams,
    slots_per_day: usize,
) -> Result<ResolvedHalfCausalParams, CudaHalfCausalEstimatorError> {
    if slots_per_day < 2 {
        return Err(CudaHalfCausalEstimatorError::InvalidInput(format!(
            "invalid slots_per_day: {slots_per_day}"
        )));
    }

    let data_period = params.data_period.unwrap_or(DEFAULT_DATA_PERIOD);
    if data_period == usize::MAX {
        return Err(CudaHalfCausalEstimatorError::InvalidInput(format!(
            "invalid data_period: {data_period}"
        )));
    }

    let filter_length = params.filter_length.unwrap_or(DEFAULT_FILTER_LENGTH);
    if filter_length < 2 {
        return Err(CudaHalfCausalEstimatorError::InvalidInput(format!(
            "invalid filter_length: {filter_length}"
        )));
    }

    let kernel_width = params.kernel_width.unwrap_or(DEFAULT_KERNEL_WIDTH);
    if !kernel_width.is_finite() || kernel_width <= 0.0 {
        return Err(CudaHalfCausalEstimatorError::InvalidInput(format!(
            "invalid kernel_width: {kernel_width}"
        )));
    }

    let maximum_confidence_adjust = params
        .maximum_confidence_adjust
        .unwrap_or(DEFAULT_MAXIMUM_CONFIDENCE_ADJUST);
    if !maximum_confidence_adjust.is_finite() || !(0.0..=100.0).contains(&maximum_confidence_adjust)
    {
        return Err(CudaHalfCausalEstimatorError::InvalidInput(format!(
            "invalid maximum_confidence_adjust: {maximum_confidence_adjust}"
        )));
    }

    let kernel_type = params.kernel_type.unwrap_or_default();
    let confidence_adjust = params.confidence_adjust.unwrap_or_default();
    let extra_smoothing = params.extra_smoothing.unwrap_or(DEFAULT_EXTRA_SMOOTHING);
    let real_filter_length = if matches!(kernel_type, HalfCausalEstimatorKernelType::Sinc) {
        filter_length.saturating_mul(2)
    } else {
        filter_length
    };

    Ok(ResolvedHalfCausalParams {
        slots_per_day,
        data_period,
        filter_length,
        real_filter_length,
        window_size: real_filter_length.saturating_mul(2).saturating_sub(1),
        kernel_width,
        kernel_type,
        confidence_adjust,
        maximum_confidence_adjust_factor: maximum_confidence_adjust * 0.01,
        enable_expected_value: params
            .enable_expected_value
            .unwrap_or(DEFAULT_ENABLE_EXPECTED_VALUE),
        extra_smoothing,
    })
}

#[inline]
fn gaussian_kernel(centered_index: f64, bandwidth: f64) -> f64 {
    let ratio = centered_index / bandwidth;
    (-ratio * ratio * 0.25).exp() / (2.0 * PI).sqrt()
}

#[inline]
fn epanechnikov_kernel(centered_index: f64, bandwidth: f64) -> f64 {
    let ratio = centered_index / bandwidth;
    if ratio.abs() <= 1.0 {
        0.75 * (1.0 - ratio * ratio)
    } else {
        0.0
    }
}

#[inline]
fn triangular_kernel(centered_index: f64, bandwidth: f64) -> f64 {
    let ratio = centered_index / bandwidth;
    if ratio.abs() <= 1.0 {
        1.0 - ratio.abs()
    } else {
        0.0
    }
}

#[inline]
fn blackman(index: f64, length: f64) -> f64 {
    0.42 - 0.5 * ((2.0 * PI * index) / (length - 1.0)).cos()
        + 0.08 * ((4.0 * PI * index) / (length - 1.0)).cos()
}

#[inline]
fn sinc(centered_index: f64, width: f64) -> f64 {
    let fc = 0.5 / width;
    if centered_index.abs() <= f64::EPSILON {
        1.0
    } else {
        let x = PI * fc * centered_index;
        x.sin() / x
    }
}

#[inline]
fn build_kernel(params: ResolvedHalfCausalParams) -> Vec<f64> {
    let mut kernel = Vec::with_capacity(params.window_size);
    let center = (params.window_size - 1) as f64 * 0.5;
    let length = params.window_size as f64;
    let mut normalization = 0.0;

    for i in 0..params.window_size {
        let index = i as f64;
        let centered = index - center;
        let weight = match params.kernel_type {
            HalfCausalEstimatorKernelType::Gaussian => {
                gaussian_kernel(centered, params.kernel_width)
            }
            HalfCausalEstimatorKernelType::Epanechnikov => {
                epanechnikov_kernel(centered, params.kernel_width)
            }
            HalfCausalEstimatorKernelType::Triangular => {
                triangular_kernel(centered, params.kernel_width)
            }
            HalfCausalEstimatorKernelType::Sinc => {
                sinc(centered, params.kernel_width) * blackman(index, length)
            }
        };
        normalization += weight;
        kernel.push(weight);
    }

    if normalization != 0.0 {
        for weight in &mut kernel {
            *weight /= normalization;
        }
    }
    kernel
}

#[inline]
fn kernel_type_id(value: HalfCausalEstimatorKernelType) -> i32 {
    match value {
        HalfCausalEstimatorKernelType::Gaussian => 0,
        HalfCausalEstimatorKernelType::Epanechnikov => 1,
        HalfCausalEstimatorKernelType::Triangular => 2,
        HalfCausalEstimatorKernelType::Sinc => 3,
    }
}

#[inline]
fn confidence_adjust_id(value: HalfCausalEstimatorConfidenceAdjust) -> i32 {
    match value {
        HalfCausalEstimatorConfidenceAdjust::Symmetric => 0,
        HalfCausalEstimatorConfidenceAdjust::Linear => 1,
        HalfCausalEstimatorConfidenceAdjust::None => 2,
    }
}

#[inline]
fn first_finite(values: &[f64]) -> usize {
    values
        .iter()
        .position(|value| value.is_finite())
        .unwrap_or(values.len())
}

impl CudaHalfCausalEstimator {
    pub fn new(device_id: usize) -> Result<Self, CudaHalfCausalEstimatorError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("half_causal_estimator_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaHalfCausalEstimatorError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaHalfCausalEstimatorError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaHalfCausalEstimatorError::OutOfMemory {
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
    ) -> Result<(), CudaHalfCausalEstimatorError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaHalfCausalEstimatorError::LaunchConfigTooLarge {
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
        sweep: &HalfCausalEstimatorBatchRange,
    ) -> Result<CudaHalfCausalEstimatorBatchResult, CudaHalfCausalEstimatorError> {
        if data.is_empty() {
            return Err(CudaHalfCausalEstimatorError::InvalidInput(
                "empty input".into(),
            ));
        }
        if first_finite(data) >= data.len() {
            return Err(CudaHalfCausalEstimatorError::InvalidInput(
                "all values are NaN".into(),
            ));
        }

        let slots_per_day = sweep.slots_per_day.ok_or_else(|| {
            CudaHalfCausalEstimatorError::InvalidInput("missing slots_per_day".into())
        })?;
        let combos = expand_grid_checked(sweep)?;
        let rows = combos.len();
        let cols = data.len();

        let mut resolved = Vec::with_capacity(rows);
        let mut kernel_matrix = Vec::new();
        let mut max_window_size = 0usize;
        let mut max_real_filter_length = 0usize;
        let mut max_wma_length = 0usize;
        for combo in &combos {
            let params = resolve_params(combo, combo.slots_per_day.unwrap_or(slots_per_day))?;
            max_window_size = max_window_size.max(params.window_size);
            max_real_filter_length = max_real_filter_length.max(params.real_filter_length);
            max_wma_length = max_wma_length.max(params.extra_smoothing.saturating_add(1));
            resolved.push(params);
        }

        kernel_matrix.resize(rows * max_window_size, 0.0);
        for (row, &params) in resolved.iter().enumerate() {
            let weights = build_kernel(params);
            let start = row * max_window_size;
            kernel_matrix[start..start + weights.len()].copy_from_slice(&weights);
        }

        let slots_per_days: Vec<i32> = resolved.iter().map(|p| p.slots_per_day as i32).collect();
        let data_periods: Vec<i32> = resolved.iter().map(|p| p.data_period as i32).collect();
        let filter_lengths: Vec<i32> = resolved.iter().map(|p| p.filter_length as i32).collect();
        let real_filter_lengths: Vec<i32> = resolved
            .iter()
            .map(|p| p.real_filter_length as i32)
            .collect();
        let window_sizes: Vec<i32> = resolved.iter().map(|p| p.window_size as i32).collect();
        let maximum_confidence_adjust_factors: Vec<f64> = resolved
            .iter()
            .map(|p| p.maximum_confidence_adjust_factor)
            .collect();
        let enable_expected_values: Vec<i32> = resolved
            .iter()
            .map(|p| i32::from(p.enable_expected_value))
            .collect();
        let confidence_adjusts: Vec<i32> = resolved
            .iter()
            .map(|p| confidence_adjust_id(p.confidence_adjust))
            .collect();
        let wma_lengths: Vec<i32> = resolved
            .iter()
            .map(|p| p.extra_smoothing.saturating_add(1) as i32)
            .collect();

        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaHalfCausalEstimatorError::InvalidInput("rows*cols overflow".into())
        })?;
        let future_cap = max_real_filter_length.saturating_sub(1).max(1);
        let window_cap = max_window_size.max(1);
        let wma_cap = max_wma_length.max(1);
        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaHalfCausalEstimatorError::InvalidInput("input bytes overflow".into())
            })?;
        let param_bytes = rows
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|v| v.checked_mul(8))
            .and_then(|v| {
                rows.checked_mul(std::mem::size_of::<f64>())
                    .and_then(|extra| extra.checked_mul(1))
                    .and_then(|extra| v.checked_add(extra))
            })
            .ok_or_else(|| {
                CudaHalfCausalEstimatorError::InvalidInput("param bytes overflow".into())
            })?;
        let kernel_bytes = rows
            .checked_mul(window_cap)
            .and_then(|v| v.checked_mul(std::mem::size_of::<f64>()))
            .ok_or_else(|| {
                CudaHalfCausalEstimatorError::InvalidInput("kernel bytes overflow".into())
            })?;
        let scratch_bytes = rows
            .checked_mul(future_cap)
            .and_then(|v| v.checked_mul(std::mem::size_of::<f64>()))
            .and_then(|v| {
                rows.checked_mul(future_cap)
                    .and_then(|n| n.checked_mul(std::mem::size_of::<f64>()))
                    .and_then(|extra| v.checked_add(extra))
            })
            .and_then(|v| {
                rows.checked_mul(wma_cap)
                    .and_then(|n| n.checked_mul(std::mem::size_of::<f64>()))
                    .and_then(|extra| v.checked_add(extra))
            })
            .ok_or_else(|| {
                CudaHalfCausalEstimatorError::InvalidInput("scratch bytes overflow".into())
            })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|v| v.checked_mul(2))
            .ok_or_else(|| {
                CudaHalfCausalEstimatorError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(param_bytes)
            .and_then(|v| v.checked_add(kernel_bytes))
            .and_then(|v| v.checked_add(scratch_bytes))
            .and_then(|v| v.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaHalfCausalEstimatorError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_data = DeviceBuffer::from_slice(data)?;
        let d_slots_per_days = DeviceBuffer::from_slice(&slots_per_days)?;
        let d_data_periods = DeviceBuffer::from_slice(&data_periods)?;
        let d_filter_lengths = DeviceBuffer::from_slice(&filter_lengths)?;
        let d_real_filter_lengths = DeviceBuffer::from_slice(&real_filter_lengths)?;
        let d_window_sizes = DeviceBuffer::from_slice(&window_sizes)?;
        let d_maximum_confidence_adjust_factors =
            DeviceBuffer::from_slice(&maximum_confidence_adjust_factors)?;
        let d_enable_expected_values = DeviceBuffer::from_slice(&enable_expected_values)?;
        let d_confidence_adjusts = DeviceBuffer::from_slice(&confidence_adjusts)?;
        let d_wma_lengths = DeviceBuffer::from_slice(&wma_lengths)?;
        let d_kernel_matrix = DeviceBuffer::from_slice(&kernel_matrix)?;
        let mut d_future_values = unsafe { DeviceBuffer::<f64>::uninitialized(rows * future_cap)? };
        let mut d_future_weights =
            unsafe { DeviceBuffer::<f64>::uninitialized(rows * future_cap)? };
        let mut d_wma_history = unsafe { DeviceBuffer::<f64>::uninitialized(rows * wma_cap)? };
        let mut d_estimate = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_expected_value = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("half_causal_estimator_batch_f64")
            .map_err(|_| CudaHalfCausalEstimatorError::MissingKernelSymbol {
                name: "half_causal_estimator_batch_f64",
            })?;
        let grid_x =
            ((rows as u32) + HALF_CAUSAL_ESTIMATOR_BLOCK_X - 1) / HALF_CAUSAL_ESTIMATOR_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(HALF_CAUSAL_ESTIMATOR_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_data.as_device_ptr(),
                cols as i32,
                d_slots_per_days.as_device_ptr(),
                d_data_periods.as_device_ptr(),
                d_filter_lengths.as_device_ptr(),
                d_real_filter_lengths.as_device_ptr(),
                d_window_sizes.as_device_ptr(),
                d_maximum_confidence_adjust_factors.as_device_ptr(),
                d_enable_expected_values.as_device_ptr(),
                d_confidence_adjusts.as_device_ptr(),
                d_wma_lengths.as_device_ptr(),
                rows as i32,
                future_cap as i32,
                window_cap as i32,
                wma_cap as i32,
                d_kernel_matrix.as_device_ptr(),
                d_future_values.as_device_ptr(),
                d_future_weights.as_device_ptr(),
                d_wma_history.as_device_ptr(),
                d_estimate.as_device_ptr(),
                d_expected_value.as_device_ptr()
            ))?;
        }
        self.stream.synchronize()?;

        Ok(CudaHalfCausalEstimatorBatchResult {
            outputs: HalfCausalEstimatorDeviceOutputs {
                estimate: HalfCausalEstimatorDeviceArrayF64 {
                    buf: d_estimate,
                    rows,
                    cols,
                },
                expected_value: HalfCausalEstimatorDeviceArrayF64 {
                    buf: d_expected_value,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
