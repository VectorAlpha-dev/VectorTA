#![cfg(feature = "cuda")]

use crate::indicators::volume_energy_reservoirs::{
    VolumeEnergyReservoirsBatchRange, VolumeEnergyReservoirsParams,
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

const VOLUME_ENERGY_RESERVOIRS_BLOCK_X: u32 = 64;
const DEFAULT_LENGTH: usize = 20;
const DEFAULT_SENSITIVITY: f64 = 1.5;
const MIN_LENGTH: usize = 5;
const FLOAT_TOL: f64 = 1.0e-12;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaVolumeEnergyReservoirsError {
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

pub struct VolumeEnergyReservoirsDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl VolumeEnergyReservoirsDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct VolumeEnergyReservoirsDeviceOutputs {
    pub momentum: VolumeEnergyReservoirsDeviceArrayF64,
    pub reservoir: VolumeEnergyReservoirsDeviceArrayF64,
    pub squeeze_active: VolumeEnergyReservoirsDeviceArrayF64,
    pub squeeze_start: VolumeEnergyReservoirsDeviceArrayF64,
    pub range_high: VolumeEnergyReservoirsDeviceArrayF64,
    pub range_low: VolumeEnergyReservoirsDeviceArrayF64,
}

impl VolumeEnergyReservoirsDeviceOutputs {
    #[inline]
    pub fn rows(&self) -> usize {
        self.momentum.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.momentum.cols
    }
}

pub struct CudaVolumeEnergyReservoirsBatchResult {
    pub outputs: VolumeEnergyReservoirsDeviceOutputs,
    pub combos: Vec<VolumeEnergyReservoirsParams>,
}

pub struct CudaVolumeEnergyReservoirs {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

#[inline]
fn first_valid_ohlcv(high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> usize {
    let len = high.len();
    let mut i = 0usize;
    while i < len {
        if high[i].is_finite()
            && low[i].is_finite()
            && close[i].is_finite()
            && volume[i].is_finite()
        {
            return i;
        }
        i += 1;
    }
    len
}

#[inline]
fn expand_axis_usize(
    (start, end, step): (usize, usize, usize),
) -> Result<Vec<usize>, CudaVolumeEnergyReservoirsError> {
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
        return Err(CudaVolumeEnergyReservoirsError::InvalidInput(format!(
            "invalid range: start={start}, end={end}, step={step}"
        )));
    }
    Ok(out)
}

#[inline]
fn expand_axis_f64(
    start: f64,
    end: f64,
    step: f64,
) -> Result<Vec<f64>, CudaVolumeEnergyReservoirsError> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() || start > end {
        return Err(CudaVolumeEnergyReservoirsError::InvalidInput(format!(
            "invalid range: start={start}, end={end}, step={step}"
        )));
    }
    if (start - end).abs() < FLOAT_TOL {
        if step.abs() > FLOAT_TOL {
            return Err(CudaVolumeEnergyReservoirsError::InvalidInput(format!(
                "invalid range: start={start}, end={end}, step={step}"
            )));
        }
        return Ok(vec![start]);
    }
    if step <= 0.0 {
        return Err(CudaVolumeEnergyReservoirsError::InvalidInput(format!(
            "invalid range: start={start}, end={end}, step={step}"
        )));
    }

    let mut out = Vec::new();
    let mut value = start;
    while value <= end + FLOAT_TOL {
        out.push(value.min(end));
        value += step;
    }
    if (out.last().copied().unwrap_or(start) - end).abs() > 1.0e-9 {
        return Err(CudaVolumeEnergyReservoirsError::InvalidInput(format!(
            "invalid range: start={start}, end={end}, step={step}"
        )));
    }
    Ok(out)
}

#[inline]
fn expand_grid_checked(
    sweep: &VolumeEnergyReservoirsBatchRange,
) -> Result<Vec<VolumeEnergyReservoirsParams>, CudaVolumeEnergyReservoirsError> {
    let lengths = expand_axis_usize(sweep.length)?;
    let sensitivities = expand_axis_f64(
        sweep.sensitivity.0,
        sweep.sensitivity.1,
        sweep.sensitivity.2,
    )?;
    let mut combos = Vec::with_capacity(lengths.len().saturating_mul(sensitivities.len()));
    for length in lengths {
        if length < MIN_LENGTH {
            return Err(CudaVolumeEnergyReservoirsError::InvalidInput(format!(
                "invalid length: {length}"
            )));
        }
        for &sensitivity in &sensitivities {
            if !sensitivity.is_finite() || sensitivity < 0.5 {
                return Err(CudaVolumeEnergyReservoirsError::InvalidInput(format!(
                    "invalid sensitivity: {sensitivity}"
                )));
            }
            combos.push(VolumeEnergyReservoirsParams {
                length: Some(length),
                sensitivity: Some(sensitivity),
            });
        }
    }
    Ok(combos)
}

impl CudaVolumeEnergyReservoirs {
    pub fn new(device_id: usize) -> Result<Self, CudaVolumeEnergyReservoirsError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("volume_energy_reservoirs_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaVolumeEnergyReservoirsError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaVolumeEnergyReservoirsError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaVolumeEnergyReservoirsError::OutOfMemory {
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
    ) -> Result<(), CudaVolumeEnergyReservoirsError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaVolumeEnergyReservoirsError::LaunchConfigTooLarge {
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
        volume: &[f64],
        sweep: &VolumeEnergyReservoirsBatchRange,
    ) -> Result<CudaVolumeEnergyReservoirsBatchResult, CudaVolumeEnergyReservoirsError> {
        if high.is_empty() || low.is_empty() || close.is_empty() || volume.is_empty() {
            return Err(CudaVolumeEnergyReservoirsError::InvalidInput(
                "empty input".into(),
            ));
        }
        if high.len() != low.len() || high.len() != close.len() || high.len() != volume.len() {
            return Err(CudaVolumeEnergyReservoirsError::InvalidInput(format!(
                "input length mismatch: high={}, low={}, close={}, volume={}",
                high.len(),
                low.len(),
                close.len(),
                volume.len()
            )));
        }
        if first_valid_ohlcv(high, low, close, volume) >= high.len() {
            return Err(CudaVolumeEnergyReservoirsError::InvalidInput(
                "all values are NaN".into(),
            ));
        }

        let combos = expand_grid_checked(sweep)?;
        let rows = combos.len();
        let cols = close.len();
        let max_length = combos
            .iter()
            .map(|params| params.length.unwrap_or(DEFAULT_LENGTH))
            .max()
            .unwrap_or(DEFAULT_LENGTH);

        let lengths: Vec<i32> = combos
            .iter()
            .map(|params| params.length.unwrap_or(DEFAULT_LENGTH) as i32)
            .collect();
        let sensitivities: Vec<f64> = combos
            .iter()
            .map(|params| params.sensitivity.unwrap_or(DEFAULT_SENSITIVITY))
            .collect();

        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaVolumeEnergyReservoirsError::InvalidInput("rows*cols overflow".into())
        })?;
        let window_cap = max_length.saturating_add(1).max(1);
        let scratch_elems = rows.checked_mul(window_cap).ok_or_else(|| {
            CudaVolumeEnergyReservoirsError::InvalidInput("scratch elements overflow".into())
        })?;

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| {
                CudaVolumeEnergyReservoirsError::InvalidInput("input bytes overflow".into())
            })?;
        let param_bytes = rows
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| {
                rows.checked_mul(std::mem::size_of::<f64>())
                    .and_then(|extra| value.checked_add(extra))
            })
            .ok_or_else(|| {
                CudaVolumeEnergyReservoirsError::InvalidInput("param bytes overflow".into())
            })?;
        let scratch_idx_bytes = scratch_elems
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                CudaVolumeEnergyReservoirsError::InvalidInput("scratch idx overflow".into())
            })?;
        let scratch_val_bytes = scratch_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                CudaVolumeEnergyReservoirsError::InvalidInput("scratch value overflow".into())
            })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(6))
            .ok_or_else(|| {
                CudaVolumeEnergyReservoirsError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(param_bytes)
            .and_then(|value| value.checked_add(scratch_idx_bytes))
            .and_then(|value| value.checked_add(scratch_val_bytes))
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaVolumeEnergyReservoirsError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_close = DeviceBuffer::from_slice(close)?;
        let d_volume = DeviceBuffer::from_slice(volume)?;
        let d_lengths = DeviceBuffer::from_slice(&lengths)?;
        let d_sensitivities = DeviceBuffer::from_slice(&sensitivities)?;
        let mut d_high_idx = unsafe { DeviceBuffer::<i32>::uninitialized(scratch_elems)? };
        let mut d_high_val = unsafe { DeviceBuffer::<f64>::uninitialized(scratch_elems)? };
        let mut d_low_idx = unsafe { DeviceBuffer::<i32>::uninitialized(scratch_elems)? };
        let mut d_low_val = unsafe { DeviceBuffer::<f64>::uninitialized(scratch_elems)? };
        let mut d_momentum = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_reservoir = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_squeeze_active = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_squeeze_start = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_range_high = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_range_low = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("volume_energy_reservoirs_batch_f64")
            .map_err(|_| CudaVolumeEnergyReservoirsError::MissingKernelSymbol {
                name: "volume_energy_reservoirs_batch_f64",
            })?;
        let grid_x = ((rows as u32) + VOLUME_ENERGY_RESERVOIRS_BLOCK_X - 1)
            / VOLUME_ENERGY_RESERVOIRS_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(VOLUME_ENERGY_RESERVOIRS_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_high.as_device_ptr(),
                d_low.as_device_ptr(),
                d_close.as_device_ptr(),
                d_volume.as_device_ptr(),
                cols as i32,
                d_lengths.as_device_ptr(),
                d_sensitivities.as_device_ptr(),
                rows as i32,
                window_cap as i32,
                d_high_idx.as_device_ptr(),
                d_high_val.as_device_ptr(),
                d_low_idx.as_device_ptr(),
                d_low_val.as_device_ptr(),
                d_momentum.as_device_ptr(),
                d_reservoir.as_device_ptr(),
                d_squeeze_active.as_device_ptr(),
                d_squeeze_start.as_device_ptr(),
                d_range_high.as_device_ptr(),
                d_range_low.as_device_ptr()
            ))?;
        }
        self.stream.synchronize()?;

        Ok(CudaVolumeEnergyReservoirsBatchResult {
            outputs: VolumeEnergyReservoirsDeviceOutputs {
                momentum: VolumeEnergyReservoirsDeviceArrayF64 {
                    buf: d_momentum,
                    rows,
                    cols,
                },
                reservoir: VolumeEnergyReservoirsDeviceArrayF64 {
                    buf: d_reservoir,
                    rows,
                    cols,
                },
                squeeze_active: VolumeEnergyReservoirsDeviceArrayF64 {
                    buf: d_squeeze_active,
                    rows,
                    cols,
                },
                squeeze_start: VolumeEnergyReservoirsDeviceArrayF64 {
                    buf: d_squeeze_start,
                    rows,
                    cols,
                },
                range_high: VolumeEnergyReservoirsDeviceArrayF64 {
                    buf: d_range_high,
                    rows,
                    cols,
                },
                range_low: VolumeEnergyReservoirsDeviceArrayF64 {
                    buf: d_range_low,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
