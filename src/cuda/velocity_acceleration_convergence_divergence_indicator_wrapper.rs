#![cfg(feature = "cuda")]

use crate::indicators::velocity_acceleration_convergence_divergence_indicator::{
    VelocityAccelerationConvergenceDivergenceIndicatorBatchRange,
    VelocityAccelerationConvergenceDivergenceIndicatorParams,
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

const VELOCITY_ACCELERATION_CONVERGENCE_DIVERGENCE_INDICATOR_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const DEFAULT_LENGTH: usize = 21;
const DEFAULT_SMOOTH_LENGTH: usize = 5;

#[derive(Debug, Error)]
pub enum CudaVelocityAccelerationConvergenceDivergenceIndicatorError {
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

pub struct VelocityAccelerationConvergenceDivergenceIndicatorDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl VelocityAccelerationConvergenceDivergenceIndicatorDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct VelocityAccelerationConvergenceDivergenceIndicatorDeviceArrayF64Pair {
    pub vacd: VelocityAccelerationConvergenceDivergenceIndicatorDeviceArrayF64,
    pub signal: VelocityAccelerationConvergenceDivergenceIndicatorDeviceArrayF64,
}

impl VelocityAccelerationConvergenceDivergenceIndicatorDeviceArrayF64Pair {
    #[inline]
    pub fn rows(&self) -> usize {
        self.vacd.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.vacd.cols
    }
}

pub struct CudaVelocityAccelerationConvergenceDivergenceIndicatorBatchResult {
    pub outputs: VelocityAccelerationConvergenceDivergenceIndicatorDeviceArrayF64Pair,
    pub combos: Vec<VelocityAccelerationConvergenceDivergenceIndicatorParams>,
}

pub struct CudaVelocityAccelerationConvergenceDivergenceIndicator {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

fn expand_axis(
    range: (usize, usize, usize),
    is_smooth: bool,
) -> Result<Vec<usize>, CudaVelocityAccelerationConvergenceDivergenceIndicatorError> {
    let (start, end, step) = range;
    if is_smooth {
        if start == 0 {
            return Err(
                CudaVelocityAccelerationConvergenceDivergenceIndicatorError::InvalidInput(format!(
                    "invalid smooth_length: {start}"
                )),
            );
        }
    } else if start < 2 {
        return Err(
            CudaVelocityAccelerationConvergenceDivergenceIndicatorError::InvalidInput(format!(
                "invalid length: {start}"
            )),
        );
    }
    if step == 0 {
        return Ok(vec![start]);
    }
    if start > end {
        return Err(
            CudaVelocityAccelerationConvergenceDivergenceIndicatorError::InvalidInput(format!(
                "invalid range: start={start}, end={end}, step={step}"
            )),
        );
    }
    let mut values = Vec::new();
    let mut current = start;
    loop {
        values.push(current);
        if current >= end {
            break;
        }
        let next = current.checked_add(step).ok_or_else(|| {
            CudaVelocityAccelerationConvergenceDivergenceIndicatorError::InvalidInput(
                "range step overflow".into(),
            )
        })?;
        if next <= current {
            return Err(
                CudaVelocityAccelerationConvergenceDivergenceIndicatorError::InvalidInput(format!(
                    "invalid range: start={start}, end={end}, step={step}"
                )),
            );
        }
        current = next.min(end);
    }
    Ok(values)
}

fn expand_grid_checked(
    range: &VelocityAccelerationConvergenceDivergenceIndicatorBatchRange,
) -> Result<
    Vec<VelocityAccelerationConvergenceDivergenceIndicatorParams>,
    CudaVelocityAccelerationConvergenceDivergenceIndicatorError,
> {
    let lengths = expand_axis(range.length, false)?;
    let smooth_lengths = expand_axis(range.smooth_length, true)?;
    let mut combos = Vec::with_capacity(lengths.len().saturating_mul(smooth_lengths.len()));
    for &length in &lengths {
        for &smooth_length in &smooth_lengths {
            combos.push(VelocityAccelerationConvergenceDivergenceIndicatorParams {
                length: Some(length),
                smooth_length: Some(smooth_length),
            });
        }
    }
    Ok(combos)
}

impl CudaVelocityAccelerationConvergenceDivergenceIndicator {
    pub fn new(
        device_id: usize,
    ) -> Result<Self, CudaVelocityAccelerationConvergenceDivergenceIndicatorError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!(
            "velocity_acceleration_convergence_divergence_indicator_kernel"
        )?;
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

    pub fn synchronize(
        &self,
    ) -> Result<(), CudaVelocityAccelerationConvergenceDivergenceIndicatorError> {
        self.stream.synchronize()?;
        Ok(())
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

    fn will_fit(
        required: usize,
        headroom: usize,
    ) -> Result<(), CudaVelocityAccelerationConvergenceDivergenceIndicatorError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(
                    CudaVelocityAccelerationConvergenceDivergenceIndicatorError::OutOfMemory {
                        required,
                        free,
                        headroom,
                    },
                );
            }
        }
        Ok(())
    }

    fn validate_launch(
        &self,
        grid: GridSize,
        block: BlockSize,
    ) -> Result<(), CudaVelocityAccelerationConvergenceDivergenceIndicatorError> {
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
                CudaVelocityAccelerationConvergenceDivergenceIndicatorError::LaunchConfigTooLarge {
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
        sweep: &VelocityAccelerationConvergenceDivergenceIndicatorBatchRange,
    ) -> Result<
        CudaVelocityAccelerationConvergenceDivergenceIndicatorBatchResult,
        CudaVelocityAccelerationConvergenceDivergenceIndicatorError,
    > {
        if data.is_empty() {
            return Err(
                CudaVelocityAccelerationConvergenceDivergenceIndicatorError::InvalidInput(
                    "empty input".into(),
                ),
            );
        }
        let valid = Self::longest_valid_run(data);
        if valid == 0 {
            return Err(
                CudaVelocityAccelerationConvergenceDivergenceIndicatorError::InvalidInput(
                    "all values are NaN".into(),
                ),
            );
        }

        let combos = expand_grid_checked(sweep)?;
        if combos.is_empty() {
            return Err(
                CudaVelocityAccelerationConvergenceDivergenceIndicatorError::InvalidInput(
                    "empty parameter grid".into(),
                ),
            );
        }

        let rows = combos.len();
        let cols = data.len();
        let mut lengths = Vec::with_capacity(rows);
        let mut smooth_lengths = Vec::with_capacity(rows);
        let mut max_length = 0usize;
        let mut max_smooth_length = 0usize;

        for combo in &combos {
            let length = combo.length.unwrap_or(DEFAULT_LENGTH);
            let smooth_length = combo.smooth_length.unwrap_or(DEFAULT_SMOOTH_LENGTH);
            if length < 2 {
                return Err(
                    CudaVelocityAccelerationConvergenceDivergenceIndicatorError::InvalidInput(
                        format!("invalid length: {length}"),
                    ),
                );
            }
            if smooth_length == 0 {
                return Err(
                    CudaVelocityAccelerationConvergenceDivergenceIndicatorError::InvalidInput(
                        format!("invalid smooth_length: {smooth_length}"),
                    ),
                );
            }
            if valid < smooth_length {
                return Err(
                    CudaVelocityAccelerationConvergenceDivergenceIndicatorError::InvalidInput(
                        format!("not enough valid data: needed={smooth_length}, valid={valid}"),
                    ),
                );
            }
            max_length = max_length.max(length);
            max_smooth_length = max_smooth_length.max(smooth_length);
            lengths.push(i32::try_from(length).map_err(|_| {
                CudaVelocityAccelerationConvergenceDivergenceIndicatorError::InvalidInput(format!(
                    "length out of range: {length}"
                ))
            })?);
            smooth_lengths.push(i32::try_from(smooth_length).map_err(|_| {
                CudaVelocityAccelerationConvergenceDivergenceIndicatorError::InvalidInput(format!(
                    "smooth_length out of range: {smooth_length}"
                ))
            })?);
        }

        let source_scratch_elems = rows.checked_mul(max_length).ok_or_else(|| {
            CudaVelocityAccelerationConvergenceDivergenceIndicatorError::InvalidInput(
                "source scratch overflow".into(),
            )
        })?;
        let raw_scratch_elems = rows.checked_mul(max_smooth_length).ok_or_else(|| {
            CudaVelocityAccelerationConvergenceDivergenceIndicatorError::InvalidInput(
                "raw scratch overflow".into(),
            )
        })?;
        let velocity_avg_scratch_elems = rows.checked_mul(max_length).ok_or_else(|| {
            CudaVelocityAccelerationConvergenceDivergenceIndicatorError::InvalidInput(
                "velocity scratch overflow".into(),
            )
        })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaVelocityAccelerationConvergenceDivergenceIndicatorError::InvalidInput(
                "rows*cols overflow".into(),
            )
        })?;
        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaVelocityAccelerationConvergenceDivergenceIndicatorError::InvalidInput(
                    "input bytes overflow".into(),
                )
            })?;
        let params_bytes = rows
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                CudaVelocityAccelerationConvergenceDivergenceIndicatorError::InvalidInput(
                    "params bytes overflow".into(),
                )
            })?;
        let scratch_bytes = source_scratch_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| {
                raw_scratch_elems
                    .checked_mul(std::mem::size_of::<f64>())
                    .and_then(|other| value.checked_add(other))
            })
            .and_then(|value| {
                velocity_avg_scratch_elems
                    .checked_mul(std::mem::size_of::<f64>())
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| {
                CudaVelocityAccelerationConvergenceDivergenceIndicatorError::InvalidInput(
                    "scratch bytes overflow".into(),
                )
            })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                CudaVelocityAccelerationConvergenceDivergenceIndicatorError::InvalidInput(
                    "output bytes overflow".into(),
                )
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(scratch_bytes))
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaVelocityAccelerationConvergenceDivergenceIndicatorError::InvalidInput(
                    "required bytes overflow".into(),
                )
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_data = DeviceBuffer::from_slice(data)?;
        let d_lengths = DeviceBuffer::from_slice(&lengths)?;
        let d_smooth_lengths = DeviceBuffer::from_slice(&smooth_lengths)?;
        let d_source_history = unsafe { DeviceBuffer::<f64>::uninitialized(source_scratch_elems)? };
        let d_raw_velocity_history =
            unsafe { DeviceBuffer::<f64>::uninitialized(raw_scratch_elems)? };
        let d_velocity_avg_history =
            unsafe { DeviceBuffer::<f64>::uninitialized(velocity_avg_scratch_elems)? };
        let d_out_vacd = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_signal = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("velocity_acceleration_convergence_divergence_indicator_batch_f64")
            .map_err(|_| {
                CudaVelocityAccelerationConvergenceDivergenceIndicatorError::MissingKernelSymbol {
                    name: "velocity_acceleration_convergence_divergence_indicator_batch_f64",
                }
            })?;
        let grid_x =
            ((rows as u32) + VELOCITY_ACCELERATION_CONVERGENCE_DIVERGENCE_INDICATOR_BLOCK_X - 1)
                / VELOCITY_ACCELERATION_CONVERGENCE_DIVERGENCE_INDICATOR_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(VELOCITY_ACCELERATION_CONVERGENCE_DIVERGENCE_INDICATOR_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_data.as_device_ptr(),
                cols as i32,
                d_lengths.as_device_ptr(),
                d_smooth_lengths.as_device_ptr(),
                rows as i32,
                max_length as i32,
                max_smooth_length as i32,
                d_source_history.as_device_ptr(),
                d_raw_velocity_history.as_device_ptr(),
                d_velocity_avg_history.as_device_ptr(),
                d_out_vacd.as_device_ptr(),
                d_out_signal.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(
            CudaVelocityAccelerationConvergenceDivergenceIndicatorBatchResult {
                outputs: VelocityAccelerationConvergenceDivergenceIndicatorDeviceArrayF64Pair {
                    vacd: VelocityAccelerationConvergenceDivergenceIndicatorDeviceArrayF64 {
                        buf: d_out_vacd,
                        rows,
                        cols,
                    },
                    signal: VelocityAccelerationConvergenceDivergenceIndicatorDeviceArrayF64 {
                        buf: d_out_signal,
                        rows,
                        cols,
                    },
                },
                combos,
            },
        )
    }
}
