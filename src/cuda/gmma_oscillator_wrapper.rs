#![cfg(feature = "cuda")]

use crate::indicators::gmma_oscillator::{
    expand_grid_gmma_oscillator, GmmaOscillatorBatchRange, GmmaOscillatorParams,
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

const GMMA_OSCILLATOR_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaGmmaOscillatorError {
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

pub struct GmmaOscillatorDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl GmmaOscillatorDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct GmmaOscillatorDeviceArrayF64Pair {
    pub oscillator: GmmaOscillatorDeviceArrayF64,
    pub signal: GmmaOscillatorDeviceArrayF64,
}

impl GmmaOscillatorDeviceArrayF64Pair {
    #[inline]
    pub fn rows(&self) -> usize {
        self.oscillator.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.oscillator.cols
    }
}

pub struct CudaGmmaOscillatorBatchResult {
    pub outputs: GmmaOscillatorDeviceArrayF64Pair,
    pub combos: Vec<GmmaOscillatorParams>,
}

pub struct CudaGmmaOscillator {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

fn longest_valid_run(data: &[f64]) -> usize {
    let mut best = 0usize;
    let mut current = 0usize;
    for &value in data {
        if value.is_finite() {
            current += 1;
            best = best.max(current);
        } else {
            current = 0;
        }
    }
    best
}

fn resolve_mode_flag(gmma_type: Option<&str>) -> Result<i32, CudaGmmaOscillatorError> {
    let gmma_type = gmma_type.unwrap_or("guppy");
    if gmma_type.eq_ignore_ascii_case("guppy") {
        return Ok(0);
    }
    if gmma_type.eq_ignore_ascii_case("super_guppy")
        || gmma_type.eq_ignore_ascii_case("superguppy")
        || gmma_type.eq_ignore_ascii_case("super-guppy")
    {
        return Ok(1);
    }
    Err(CudaGmmaOscillatorError::InvalidInput(format!(
        "invalid GMMA type: {gmma_type}"
    )))
}

fn resolve_multiplier(
    anchor_minutes: usize,
    interval_minutes: Option<usize>,
) -> Result<usize, CudaGmmaOscillatorError> {
    if anchor_minutes == 0 {
        return Ok(1);
    }
    let interval_minutes = interval_minutes.ok_or_else(|| {
        CudaGmmaOscillatorError::InvalidInput("anchor_minutes requires interval_minutes".into())
    })?;
    if interval_minutes == 0 {
        return Err(CudaGmmaOscillatorError::InvalidInput(
            "invalid interval_minutes: 0".into(),
        ));
    }
    if interval_minutes >= anchor_minutes {
        return Ok(1);
    }
    let ratio = (anchor_minutes as f64 / interval_minutes as f64).round();
    Ok(ratio.max(1.0) as usize)
}

impl CudaGmmaOscillator {
    pub fn new(device_id: usize) -> Result<Self, CudaGmmaOscillatorError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("gmma_oscillator_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaGmmaOscillatorError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaGmmaOscillatorError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaGmmaOscillatorError::OutOfMemory {
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
    ) -> Result<(), CudaGmmaOscillatorError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaGmmaOscillatorError::LaunchConfigTooLarge {
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
        sweep: &GmmaOscillatorBatchRange,
        fixed: &GmmaOscillatorParams,
    ) -> Result<CudaGmmaOscillatorBatchResult, CudaGmmaOscillatorError> {
        if data.is_empty() {
            return Err(CudaGmmaOscillatorError::InvalidInput("empty input".into()));
        }
        if longest_valid_run(data) == 0 {
            return Err(CudaGmmaOscillatorError::InvalidInput(
                "all values are NaN".into(),
            ));
        }

        let mode_flag = resolve_mode_flag(fixed.gmma_type.as_deref())?;
        let anchor_minutes = fixed.anchor_minutes.unwrap_or(0);
        if anchor_minutes > 1440 {
            return Err(CudaGmmaOscillatorError::InvalidInput(format!(
                "invalid anchor_minutes: {anchor_minutes}"
            )));
        }
        if fixed.interval_minutes == Some(0) {
            return Err(CudaGmmaOscillatorError::InvalidInput(
                "invalid interval_minutes: 0".into(),
            ));
        }
        let multiplier = resolve_multiplier(anchor_minutes, fixed.interval_minutes)?;
        let combos = expand_grid_gmma_oscillator(sweep, fixed)
            .map_err(|err| CudaGmmaOscillatorError::InvalidInput(err.to_string()))?;
        if combos.is_empty() {
            return Err(CudaGmmaOscillatorError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let rows = combos.len();
        let cols = data.len();
        let mut smooth_lengths = Vec::with_capacity(rows);
        let mut signal_lengths = Vec::with_capacity(rows);
        let mut max_smooth_length = 0usize;
        for combo in &combos {
            let smooth_length = combo.smooth_length.unwrap_or(1);
            let signal_length = combo.signal_length.unwrap_or(13);
            if smooth_length == 0 {
                return Err(CudaGmmaOscillatorError::InvalidInput(format!(
                    "invalid smooth_length: {smooth_length}"
                )));
            }
            if signal_length == 0 {
                return Err(CudaGmmaOscillatorError::InvalidInput(format!(
                    "invalid signal_length: {signal_length}"
                )));
            }
            max_smooth_length = max_smooth_length.max(smooth_length);
            smooth_lengths.push(i32::try_from(smooth_length).map_err(|_| {
                CudaGmmaOscillatorError::InvalidInput(format!(
                    "smooth_length out of range: {smooth_length}"
                ))
            })?);
            signal_lengths.push(i32::try_from(signal_length).map_err(|_| {
                CudaGmmaOscillatorError::InvalidInput(format!(
                    "signal_length out of range: {signal_length}"
                ))
            })?);
        }

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| CudaGmmaOscillatorError::InvalidInput("input bytes overflow".into()))?;
        let params_bytes = rows
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| {
                rows.checked_mul(std::mem::size_of::<i32>())
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| CudaGmmaOscillatorError::InvalidInput("params bytes overflow".into()))?;
        let output_elems = rows
            .checked_mul(cols)
            .ok_or_else(|| CudaGmmaOscillatorError::InvalidInput("rows*cols overflow".into()))?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| CudaGmmaOscillatorError::InvalidInput("output bytes overflow".into()))?;
        let scratch_elems = rows.checked_mul(max_smooth_length).ok_or_else(|| {
            CudaGmmaOscillatorError::InvalidInput("scratch rows*cols overflow".into())
        })?;
        let scratch_bytes = scratch_elems
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaGmmaOscillatorError::InvalidInput("scratch bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .and_then(|value| value.checked_add(scratch_bytes))
            .ok_or_else(|| {
                CudaGmmaOscillatorError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_data = DeviceBuffer::from_slice(data)?;
        let d_smooth_lengths = DeviceBuffer::from_slice(&smooth_lengths)?;
        let d_signal_lengths = DeviceBuffer::from_slice(&signal_lengths)?;
        let d_raw_windows = unsafe { DeviceBuffer::<f64>::uninitialized(scratch_elems)? };
        let d_out_oscillator = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_signal = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("gmma_oscillator_batch_f64")
            .map_err(|_| CudaGmmaOscillatorError::MissingKernelSymbol {
                name: "gmma_oscillator_batch_f64",
            })?;
        let grid_x = ((rows as u32) + GMMA_OSCILLATOR_BLOCK_X - 1) / GMMA_OSCILLATOR_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(GMMA_OSCILLATOR_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_data.as_device_ptr(),
                cols as i32,
                mode_flag,
                multiplier as i32,
                d_smooth_lengths.as_device_ptr(),
                d_signal_lengths.as_device_ptr(),
                rows as i32,
                max_smooth_length as i32,
                d_raw_windows.as_device_ptr(),
                d_out_oscillator.as_device_ptr(),
                d_out_signal.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaGmmaOscillatorBatchResult {
            outputs: GmmaOscillatorDeviceArrayF64Pair {
                oscillator: GmmaOscillatorDeviceArrayF64 {
                    buf: d_out_oscillator,
                    rows,
                    cols,
                },
                signal: GmmaOscillatorDeviceArrayF64 {
                    buf: d_out_signal,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
