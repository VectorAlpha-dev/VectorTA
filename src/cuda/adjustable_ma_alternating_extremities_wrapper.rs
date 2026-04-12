#![cfg(feature = "cuda")]

use crate::indicators::adjustable_ma_alternating_extremities::{
    AdjustableMaAlternatingExtremitiesBatchRange, AdjustableMaAlternatingExtremitiesParams,
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

const ADJUSTABLE_MA_ALTERNATING_EXTREMITIES_BLOCK_X: u32 = 64;
const DEFAULT_LENGTH: usize = 50;
const DEFAULT_MULT: f64 = 2.0;
const DEFAULT_ALPHA: f64 = 1.0;
const DEFAULT_BETA: f64 = 0.5;
const TWO_PI: f64 = core::f64::consts::PI * 2.0;
const WEIGHT_SUM_EPS: f64 = 1e-12;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaAdjustableMaAlternatingExtremitiesError {
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

pub struct AdjustableMaAlternatingExtremitiesDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl AdjustableMaAlternatingExtremitiesDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct AdjustableMaAlternatingExtremitiesDeviceOutputs {
    pub ma: AdjustableMaAlternatingExtremitiesDeviceArrayF64,
    pub upper: AdjustableMaAlternatingExtremitiesDeviceArrayF64,
    pub lower: AdjustableMaAlternatingExtremitiesDeviceArrayF64,
    pub extremity: AdjustableMaAlternatingExtremitiesDeviceArrayF64,
    pub state: AdjustableMaAlternatingExtremitiesDeviceArrayF64,
    pub changed: AdjustableMaAlternatingExtremitiesDeviceArrayF64,
    pub smoothed_open: AdjustableMaAlternatingExtremitiesDeviceArrayF64,
    pub smoothed_high: AdjustableMaAlternatingExtremitiesDeviceArrayF64,
    pub smoothed_low: AdjustableMaAlternatingExtremitiesDeviceArrayF64,
    pub smoothed_close: AdjustableMaAlternatingExtremitiesDeviceArrayF64,
}

impl AdjustableMaAlternatingExtremitiesDeviceOutputs {
    #[inline]
    pub fn rows(&self) -> usize {
        self.ma.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.ma.cols
    }
}

pub struct CudaAdjustableMaAlternatingExtremitiesBatchResult {
    pub outputs: AdjustableMaAlternatingExtremitiesDeviceOutputs,
    pub combos: Vec<AdjustableMaAlternatingExtremitiesParams>,
}

pub struct CudaAdjustableMaAlternatingExtremities {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

#[inline]
fn axis_usize(
    (start, end, step): (usize, usize, usize),
) -> Result<Vec<usize>, CudaAdjustableMaAlternatingExtremitiesError> {
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
        return Err(CudaAdjustableMaAlternatingExtremitiesError::InvalidInput(
            format!("invalid range: start={start}, end={end}, step={step}"),
        ));
    }
    Ok(out)
}

#[inline]
fn axis_f64(
    (start, end, step): (f64, f64, f64),
) -> Result<Vec<f64>, CudaAdjustableMaAlternatingExtremitiesError> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() {
        return Err(CudaAdjustableMaAlternatingExtremitiesError::InvalidInput(
            format!("invalid float range: start={start}, end={end}, step={step}"),
        ));
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
        return Err(CudaAdjustableMaAlternatingExtremitiesError::InvalidInput(
            format!("invalid float range: start={start}, end={end}, step={step}"),
        ));
    }
    Ok(out)
}

#[inline]
fn build_weights_sum(length: usize, alpha: f64, beta: f64) -> f64 {
    let denom = (length - 1) as f64;
    let mut sum = 0.0;
    for i in 0..length {
        let x = i as f64 / denom;
        sum += (TWO_PI * x.powf(alpha)).sin() * (1.0 - x.powf(beta));
    }
    sum
}

#[inline]
fn expand_grid_checked(
    range: &AdjustableMaAlternatingExtremitiesBatchRange,
) -> Result<
    Vec<AdjustableMaAlternatingExtremitiesParams>,
    CudaAdjustableMaAlternatingExtremitiesError,
> {
    let lengths = axis_usize(range.length)?;
    let mults = axis_f64(range.mult)?;
    let alphas = axis_f64(range.alpha)?;
    let betas = axis_f64(range.beta)?;
    let total = lengths
        .len()
        .checked_mul(mults.len())
        .and_then(|v| v.checked_mul(alphas.len()))
        .and_then(|v| v.checked_mul(betas.len()))
        .ok_or_else(|| {
            CudaAdjustableMaAlternatingExtremitiesError::InvalidInput(
                "parameter grid overflow".into(),
            )
        })?;
    let mut out = Vec::with_capacity(total);
    for &length in &lengths {
        for &mult in &mults {
            for &alpha in &alphas {
                for &beta in &betas {
                    out.push(AdjustableMaAlternatingExtremitiesParams {
                        length: Some(length),
                        mult: Some(mult),
                        alpha: Some(alpha),
                        beta: Some(beta),
                    });
                }
            }
        }
    }
    Ok(out)
}

impl CudaAdjustableMaAlternatingExtremities {
    pub fn new(device_id: usize) -> Result<Self, CudaAdjustableMaAlternatingExtremitiesError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module =
            crate::load_cuda_embedded_module!("adjustable_ma_alternating_extremities_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaAdjustableMaAlternatingExtremitiesError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn will_fit(
        required: usize,
        headroom: usize,
    ) -> Result<(), CudaAdjustableMaAlternatingExtremitiesError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaAdjustableMaAlternatingExtremitiesError::OutOfMemory {
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
    ) -> Result<(), CudaAdjustableMaAlternatingExtremitiesError> {
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
                CudaAdjustableMaAlternatingExtremitiesError::LaunchConfigTooLarge {
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
        high: &[f64],
        low: &[f64],
        close: &[f64],
        sweep: &AdjustableMaAlternatingExtremitiesBatchRange,
    ) -> Result<
        CudaAdjustableMaAlternatingExtremitiesBatchResult,
        CudaAdjustableMaAlternatingExtremitiesError,
    > {
        if high.is_empty() || low.is_empty() || close.is_empty() {
            return Err(CudaAdjustableMaAlternatingExtremitiesError::InvalidInput(
                "empty input".into(),
            ));
        }
        if high.len() != low.len() || high.len() != close.len() {
            return Err(CudaAdjustableMaAlternatingExtremitiesError::InvalidInput(
                format!(
                    "input length mismatch: high={}, low={}, close={}",
                    high.len(),
                    low.len(),
                    close.len()
                ),
            ));
        }

        let combos = expand_grid_checked(sweep)?;
        let cols = close.len();
        let first = (0..cols)
            .find(|&i| high[i].is_finite() && low[i].is_finite() && close[i].is_finite())
            .ok_or_else(|| {
                CudaAdjustableMaAlternatingExtremitiesError::InvalidInput(
                    "all values are NaN".into(),
                )
            })?;

        for params in &combos {
            let length = params.length.unwrap_or(DEFAULT_LENGTH);
            let mult = params.mult.unwrap_or(DEFAULT_MULT);
            let alpha = params.alpha.unwrap_or(DEFAULT_ALPHA);
            let beta = params.beta.unwrap_or(DEFAULT_BETA);
            if length < 2 || length > cols {
                return Err(CudaAdjustableMaAlternatingExtremitiesError::InvalidInput(
                    format!("invalid length: length={length}, data_len={cols}"),
                ));
            }
            if !mult.is_finite() || mult < 1.0 {
                return Err(CudaAdjustableMaAlternatingExtremitiesError::InvalidInput(
                    format!("invalid mult: {mult}"),
                ));
            }
            if !alpha.is_finite() || alpha < 0.0 {
                return Err(CudaAdjustableMaAlternatingExtremitiesError::InvalidInput(
                    format!("invalid alpha: {alpha}"),
                ));
            }
            if !beta.is_finite() || beta < 0.0 {
                return Err(CudaAdjustableMaAlternatingExtremitiesError::InvalidInput(
                    format!("invalid beta: {beta}"),
                ));
            }
            let needed = (length * 2) - 1;
            if cols - first < needed {
                return Err(CudaAdjustableMaAlternatingExtremitiesError::InvalidInput(
                    format!(
                        "not enough valid data: needed={needed}, valid={}",
                        cols - first
                    ),
                ));
            }
            let sum = build_weights_sum(length, alpha, beta);
            if !sum.is_finite() || sum.abs() <= WEIGHT_SUM_EPS {
                return Err(CudaAdjustableMaAlternatingExtremitiesError::InvalidInput(
                    format!("degenerate kernel weights for alpha={alpha}, beta={beta}"),
                ));
            }
        }

        let rows = combos.len();
        let lengths: Vec<i32> = combos
            .iter()
            .map(|params| params.length.unwrap_or(DEFAULT_LENGTH) as i32)
            .collect();
        let mults: Vec<f64> = combos
            .iter()
            .map(|params| params.mult.unwrap_or(DEFAULT_MULT))
            .collect();
        let alphas: Vec<f64> = combos
            .iter()
            .map(|params| params.alpha.unwrap_or(DEFAULT_ALPHA))
            .collect();
        let betas: Vec<f64> = combos
            .iter()
            .map(|params| params.beta.unwrap_or(DEFAULT_BETA))
            .collect();

        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaAdjustableMaAlternatingExtremitiesError::InvalidInput("rows*cols overflow".into())
        })?;
        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(3))
            .ok_or_else(|| {
                CudaAdjustableMaAlternatingExtremitiesError::InvalidInput(
                    "input bytes overflow".into(),
                )
            })?;
        let param_bytes = rows
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(3))
            .and_then(|value| value.checked_add(rows * std::mem::size_of::<i32>()))
            .ok_or_else(|| {
                CudaAdjustableMaAlternatingExtremitiesError::InvalidInput(
                    "param bytes overflow".into(),
                )
            })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(10))
            .ok_or_else(|| {
                CudaAdjustableMaAlternatingExtremitiesError::InvalidInput(
                    "output bytes overflow".into(),
                )
            })?;
        let required = input_bytes
            .checked_add(param_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaAdjustableMaAlternatingExtremitiesError::InvalidInput(
                    "required bytes overflow".into(),
                )
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_close = DeviceBuffer::from_slice(close)?;
        let d_lengths = DeviceBuffer::from_slice(&lengths)?;
        let d_mults = DeviceBuffer::from_slice(&mults)?;
        let d_alphas = DeviceBuffer::from_slice(&alphas)?;
        let d_betas = DeviceBuffer::from_slice(&betas)?;
        let mut d_ma = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_upper = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_lower = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_extremity = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_state = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_changed = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_smoothed_open = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_smoothed_high = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_smoothed_low = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_smoothed_close = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("adjustable_ma_alternating_extremities_batch_f64")
            .map_err(
                |_| CudaAdjustableMaAlternatingExtremitiesError::MissingKernelSymbol {
                    name: "adjustable_ma_alternating_extremities_batch_f64",
                },
            )?;
        let grid_x = ((rows as u32) + ADJUSTABLE_MA_ALTERNATING_EXTREMITIES_BLOCK_X - 1)
            / ADJUSTABLE_MA_ALTERNATING_EXTREMITIES_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(ADJUSTABLE_MA_ALTERNATING_EXTREMITIES_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_high.as_device_ptr(),
                d_low.as_device_ptr(),
                d_close.as_device_ptr(),
                cols as i32,
                d_lengths.as_device_ptr(),
                d_mults.as_device_ptr(),
                d_alphas.as_device_ptr(),
                d_betas.as_device_ptr(),
                rows as i32,
                d_ma.as_device_ptr(),
                d_upper.as_device_ptr(),
                d_lower.as_device_ptr(),
                d_extremity.as_device_ptr(),
                d_state.as_device_ptr(),
                d_changed.as_device_ptr(),
                d_smoothed_open.as_device_ptr(),
                d_smoothed_high.as_device_ptr(),
                d_smoothed_low.as_device_ptr(),
                d_smoothed_close.as_device_ptr()
            ))?;
        }
        self.stream.synchronize()?;

        Ok(CudaAdjustableMaAlternatingExtremitiesBatchResult {
            outputs: AdjustableMaAlternatingExtremitiesDeviceOutputs {
                ma: AdjustableMaAlternatingExtremitiesDeviceArrayF64 {
                    buf: d_ma,
                    rows,
                    cols,
                },
                upper: AdjustableMaAlternatingExtremitiesDeviceArrayF64 {
                    buf: d_upper,
                    rows,
                    cols,
                },
                lower: AdjustableMaAlternatingExtremitiesDeviceArrayF64 {
                    buf: d_lower,
                    rows,
                    cols,
                },
                extremity: AdjustableMaAlternatingExtremitiesDeviceArrayF64 {
                    buf: d_extremity,
                    rows,
                    cols,
                },
                state: AdjustableMaAlternatingExtremitiesDeviceArrayF64 {
                    buf: d_state,
                    rows,
                    cols,
                },
                changed: AdjustableMaAlternatingExtremitiesDeviceArrayF64 {
                    buf: d_changed,
                    rows,
                    cols,
                },
                smoothed_open: AdjustableMaAlternatingExtremitiesDeviceArrayF64 {
                    buf: d_smoothed_open,
                    rows,
                    cols,
                },
                smoothed_high: AdjustableMaAlternatingExtremitiesDeviceArrayF64 {
                    buf: d_smoothed_high,
                    rows,
                    cols,
                },
                smoothed_low: AdjustableMaAlternatingExtremitiesDeviceArrayF64 {
                    buf: d_smoothed_low,
                    rows,
                    cols,
                },
                smoothed_close: AdjustableMaAlternatingExtremitiesDeviceArrayF64 {
                    buf: d_smoothed_close,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
