#![cfg(feature = "cuda")]

use crate::indicators::vwap_zscore_with_signals::{
    VwapZscoreWithSignalsBatchRange, VwapZscoreWithSignalsParams,
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

const VWAP_ZSCORE_WITH_SIGNALS_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const DEFAULT_LENGTH: usize = 20;
const DEFAULT_UPPER_BOTTOM: f64 = 2.5;
const DEFAULT_LOWER_BOTTOM: f64 = -2.5;

#[derive(Debug, Error)]
pub enum CudaVwapZscoreWithSignalsError {
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

pub struct VwapZscoreWithSignalsDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl VwapZscoreWithSignalsDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct VwapZscoreWithSignalsDeviceArrayF64Triple {
    pub zvwap: VwapZscoreWithSignalsDeviceArrayF64,
    pub support_signal: VwapZscoreWithSignalsDeviceArrayF64,
    pub resistance_signal: VwapZscoreWithSignalsDeviceArrayF64,
}

impl VwapZscoreWithSignalsDeviceArrayF64Triple {
    #[inline]
    pub fn rows(&self) -> usize {
        self.zvwap.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.zvwap.cols
    }
}

pub struct CudaVwapZscoreWithSignalsBatchResult {
    pub outputs: VwapZscoreWithSignalsDeviceArrayF64Triple,
    pub combos: Vec<VwapZscoreWithSignalsParams>,
}

pub struct CudaVwapZscoreWithSignals {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

fn expand_axis_usize(
    start: usize,
    end: usize,
    step: usize,
) -> Result<Vec<usize>, CudaVwapZscoreWithSignalsError> {
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
        return Err(CudaVwapZscoreWithSignalsError::InvalidInput(format!(
            "invalid range: start={start}, end={end}, step={step}"
        )));
    }
    Ok(out)
}

fn expand_axis_f64(
    start: f64,
    end: f64,
    step: f64,
) -> Result<Vec<f64>, CudaVwapZscoreWithSignalsError> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() {
        return Err(CudaVwapZscoreWithSignalsError::InvalidInput(format!(
            "invalid range: start={start}, end={end}, step={step}"
        )));
    }
    if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
        return Ok(vec![start]);
    }
    let mut out = Vec::new();
    if start < end {
        let step_abs = step.abs();
        let mut value = start;
        while value <= end + 1e-12 {
            out.push(value);
            value += step_abs;
        }
    } else {
        let step_abs = -step.abs();
        let mut value = start;
        while value >= end - 1e-12 {
            out.push(value);
            value += step_abs;
        }
    }
    if out.is_empty() {
        return Err(CudaVwapZscoreWithSignalsError::InvalidInput(format!(
            "invalid range: start={start}, end={end}, step={step}"
        )));
    }
    Ok(out)
}

fn expand_grid_vwap_zscore_with_signals(
    range: &VwapZscoreWithSignalsBatchRange,
) -> Result<Vec<VwapZscoreWithSignalsParams>, CudaVwapZscoreWithSignalsError> {
    let lengths = expand_axis_usize(range.length.0, range.length.1, range.length.2)?;
    let upper_bottoms = expand_axis_f64(
        range.upper_bottom.0,
        range.upper_bottom.1,
        range.upper_bottom.2,
    )?;
    let lower_bottoms = expand_axis_f64(
        range.lower_bottom.0,
        range.lower_bottom.1,
        range.lower_bottom.2,
    )?;

    let mut out = Vec::with_capacity(lengths.len() * upper_bottoms.len() * lower_bottoms.len());
    for length in lengths {
        if length == 0 {
            return Err(CudaVwapZscoreWithSignalsError::InvalidInput(
                "invalid length: 0".into(),
            ));
        }
        for &upper_bottom in &upper_bottoms {
            if !upper_bottom.is_finite() {
                return Err(CudaVwapZscoreWithSignalsError::InvalidInput(format!(
                    "invalid upper_bottom: {upper_bottom}"
                )));
            }
            for &lower_bottom in &lower_bottoms {
                if !lower_bottom.is_finite() {
                    return Err(CudaVwapZscoreWithSignalsError::InvalidInput(format!(
                        "invalid lower_bottom: {lower_bottom}"
                    )));
                }
                out.push(VwapZscoreWithSignalsParams {
                    length: Some(length),
                    upper_bottom: Some(upper_bottom),
                    lower_bottom: Some(lower_bottom),
                });
            }
        }
    }
    Ok(out)
}

impl CudaVwapZscoreWithSignals {
    pub fn new(device_id: usize) -> Result<Self, CudaVwapZscoreWithSignalsError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("vwap_zscore_with_signals_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaVwapZscoreWithSignalsError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn valid_close_volume_bar(close: f64, volume: f64) -> bool {
        close.is_finite() && volume.is_finite() && volume >= 0.0
    }

    fn first_valid_close_volume(close: &[f64], volume: &[f64]) -> Option<usize> {
        (0..close.len()).find(|&i| Self::valid_close_volume_bar(close[i], volume[i]))
    }

    fn count_valid_close_volume(close: &[f64], volume: &[f64]) -> usize {
        (0..close.len())
            .filter(|&i| Self::valid_close_volume_bar(close[i], volume[i]))
            .count()
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaVwapZscoreWithSignalsError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaVwapZscoreWithSignalsError::OutOfMemory {
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
    ) -> Result<(), CudaVwapZscoreWithSignalsError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaVwapZscoreWithSignalsError::LaunchConfigTooLarge {
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
        close: &[f64],
        volume: &[f64],
        sweep: &VwapZscoreWithSignalsBatchRange,
    ) -> Result<CudaVwapZscoreWithSignalsBatchResult, CudaVwapZscoreWithSignalsError> {
        if close.is_empty() || volume.is_empty() {
            return Err(CudaVwapZscoreWithSignalsError::InvalidInput(
                "empty input".into(),
            ));
        }
        if close.len() != volume.len() {
            return Err(CudaVwapZscoreWithSignalsError::InvalidInput(format!(
                "input length mismatch: close={}, volume={}",
                close.len(),
                volume.len()
            )));
        }
        Self::first_valid_close_volume(close, volume).ok_or_else(|| {
            CudaVwapZscoreWithSignalsError::InvalidInput("all values are NaN".into())
        })?;
        let valid = Self::count_valid_close_volume(close, volume);

        let combos = expand_grid_vwap_zscore_with_signals(sweep)?;
        if combos.is_empty() {
            return Err(CudaVwapZscoreWithSignalsError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let rows = combos.len();
        let cols = close.len();
        let mut lengths = Vec::with_capacity(rows);
        let mut upper_bottoms = Vec::with_capacity(rows);
        let mut lower_bottoms = Vec::with_capacity(rows);
        let mut max_length = 0usize;
        for combo in &combos {
            let length = combo.length.unwrap_or(DEFAULT_LENGTH);
            let upper_bottom = combo.upper_bottom.unwrap_or(DEFAULT_UPPER_BOTTOM);
            let lower_bottom = combo.lower_bottom.unwrap_or(DEFAULT_LOWER_BOTTOM);
            if length == 0 || length > cols {
                return Err(CudaVwapZscoreWithSignalsError::InvalidInput(format!(
                    "invalid length: length={length}, data_len={cols}"
                )));
            }
            if !upper_bottom.is_finite() {
                return Err(CudaVwapZscoreWithSignalsError::InvalidInput(format!(
                    "invalid upper_bottom: {upper_bottom}"
                )));
            }
            if !lower_bottom.is_finite() {
                return Err(CudaVwapZscoreWithSignalsError::InvalidInput(format!(
                    "invalid lower_bottom: {lower_bottom}"
                )));
            }
            let needed = length
                .checked_mul(2)
                .and_then(|value| value.checked_sub(1))
                .ok_or_else(|| {
                    CudaVwapZscoreWithSignalsError::InvalidInput("needed overflow".into())
                })?;
            if valid < needed {
                return Err(CudaVwapZscoreWithSignalsError::InvalidInput(format!(
                    "not enough valid data: needed={needed}, valid={valid}"
                )));
            }
            max_length = max_length.max(length);
            lengths.push(i32::try_from(length).map_err(|_| {
                CudaVwapZscoreWithSignalsError::InvalidInput(format!(
                    "length out of range: {length}"
                ))
            })?);
            upper_bottoms.push(upper_bottom);
            lower_bottoms.push(lower_bottom);
        }

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                CudaVwapZscoreWithSignalsError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = rows
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| {
                rows.checked_mul(std::mem::size_of::<f64>())
                    .and_then(|other| other.checked_mul(2))
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| {
                CudaVwapZscoreWithSignalsError::InvalidInput("params bytes overflow".into())
            })?;
        let scratch_elems = rows.checked_mul(max_length).ok_or_else(|| {
            CudaVwapZscoreWithSignalsError::InvalidInput("scratch rows*cols overflow".into())
        })?;
        let scratch_bytes = scratch_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| {
                scratch_elems
                    .checked_mul(std::mem::size_of::<f64>())
                    .and_then(|other| value.checked_add(other))
            })
            .and_then(|value| {
                scratch_elems
                    .checked_mul(std::mem::size_of::<i32>())
                    .and_then(|other| value.checked_add(other))
            })
            .and_then(|value| {
                scratch_elems
                    .checked_mul(std::mem::size_of::<f64>())
                    .and_then(|other| value.checked_add(other))
            })
            .and_then(|value| {
                scratch_elems
                    .checked_mul(std::mem::size_of::<i32>())
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| {
                CudaVwapZscoreWithSignalsError::InvalidInput("scratch bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaVwapZscoreWithSignalsError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(3))
            .ok_or_else(|| {
                CudaVwapZscoreWithSignalsError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(scratch_bytes))
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaVwapZscoreWithSignalsError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_close = DeviceBuffer::from_slice(close)?;
        let d_volume = DeviceBuffer::from_slice(volume)?;
        let d_lengths = DeviceBuffer::from_slice(&lengths)?;
        let d_upper_bottoms = DeviceBuffer::from_slice(&upper_bottoms)?;
        let d_lower_bottoms = DeviceBuffer::from_slice(&lower_bottoms)?;
        let d_pv_values = unsafe { DeviceBuffer::<f64>::uninitialized(scratch_elems)? };
        let d_vol_values = unsafe { DeviceBuffer::<f64>::uninitialized(scratch_elems)? };
        let d_pv_valid = unsafe { DeviceBuffer::<i32>::uninitialized(scratch_elems)? };
        let d_dev_values = unsafe { DeviceBuffer::<f64>::uninitialized(scratch_elems)? };
        let d_dev_valid = unsafe { DeviceBuffer::<i32>::uninitialized(scratch_elems)? };
        let d_out_zvwap = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_support = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_resistance = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("vwap_zscore_with_signals_batch_f64")
            .map_err(|_| CudaVwapZscoreWithSignalsError::MissingKernelSymbol {
                name: "vwap_zscore_with_signals_batch_f64",
            })?;
        let grid_x = ((rows as u32) + VWAP_ZSCORE_WITH_SIGNALS_BLOCK_X - 1)
            / VWAP_ZSCORE_WITH_SIGNALS_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(VWAP_ZSCORE_WITH_SIGNALS_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_close.as_device_ptr(),
                d_volume.as_device_ptr(),
                cols as i32,
                d_lengths.as_device_ptr(),
                d_upper_bottoms.as_device_ptr(),
                d_lower_bottoms.as_device_ptr(),
                rows as i32,
                max_length as i32,
                d_pv_values.as_device_ptr(),
                d_vol_values.as_device_ptr(),
                d_pv_valid.as_device_ptr(),
                d_dev_values.as_device_ptr(),
                d_dev_valid.as_device_ptr(),
                d_out_zvwap.as_device_ptr(),
                d_out_support.as_device_ptr(),
                d_out_resistance.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaVwapZscoreWithSignalsBatchResult {
            outputs: VwapZscoreWithSignalsDeviceArrayF64Triple {
                zvwap: VwapZscoreWithSignalsDeviceArrayF64 {
                    buf: d_out_zvwap,
                    rows,
                    cols,
                },
                support_signal: VwapZscoreWithSignalsDeviceArrayF64 {
                    buf: d_out_support,
                    rows,
                    cols,
                },
                resistance_signal: VwapZscoreWithSignalsDeviceArrayF64 {
                    buf: d_out_resistance,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
