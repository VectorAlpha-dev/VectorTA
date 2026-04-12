#![cfg(feature = "cuda")]

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

use crate::indicators::adaptive_bandpass_trigger_oscillator::{
    AdaptiveBandpassTriggerOscillatorBatchRange, AdaptiveBandpassTriggerOscillatorParams,
};

const ADAPTIVE_BANDPASS_TRIGGER_OSCILLATOR_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const DEFAULT_DELTA: f64 = 0.1;
const DEFAULT_ALPHA: f64 = 0.07;
const FLOAT_TOL: f64 = 1e-12;
const MIN_VALID_SAMPLES: usize = 12;

#[derive(Debug, Error)]
pub enum CudaAdaptiveBandpassTriggerOscillatorError {
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

pub struct AdaptiveBandpassTriggerOscillatorDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl AdaptiveBandpassTriggerOscillatorDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct AdaptiveBandpassTriggerOscillatorDeviceArrayF64Pair {
    pub in_phase: AdaptiveBandpassTriggerOscillatorDeviceArrayF64,
    pub lead: AdaptiveBandpassTriggerOscillatorDeviceArrayF64,
}

impl AdaptiveBandpassTriggerOscillatorDeviceArrayF64Pair {
    #[inline]
    pub fn rows(&self) -> usize {
        self.in_phase.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.in_phase.cols
    }
}

pub struct CudaAdaptiveBandpassTriggerOscillatorBatchResult {
    pub outputs: AdaptiveBandpassTriggerOscillatorDeviceArrayF64Pair,
    pub combos: Vec<AdaptiveBandpassTriggerOscillatorParams>,
}

pub struct CudaAdaptiveBandpassTriggerOscillator {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaAdaptiveBandpassTriggerOscillator {
    pub fn new(device_id: usize) -> Result<Self, CudaAdaptiveBandpassTriggerOscillatorError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module =
            crate::load_cuda_embedded_module!("adaptive_bandpass_trigger_oscillator_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaAdaptiveBandpassTriggerOscillatorError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn first_valid_value(data: &[f64]) -> Option<usize> {
        data.iter().position(|value| value.is_finite())
    }

    fn count_valid_values(data: &[f64]) -> usize {
        data.iter().filter(|value| value.is_finite()).count()
    }

    fn expand_axis_f64(
        start: f64,
        end: f64,
        step: f64,
    ) -> Result<Vec<f64>, CudaAdaptiveBandpassTriggerOscillatorError> {
        if !start.is_finite() || !end.is_finite() || !step.is_finite() || start > end {
            return Err(CudaAdaptiveBandpassTriggerOscillatorError::InvalidInput(
                format!("invalid range: start={start}, end={end}, step={step}"),
            ));
        }
        if (start - end).abs() < FLOAT_TOL {
            if step.abs() > FLOAT_TOL {
                return Err(CudaAdaptiveBandpassTriggerOscillatorError::InvalidInput(
                    format!("invalid range: start={start}, end={end}, step={step}"),
                ));
            }
            return Ok(vec![start]);
        }
        if step <= 0.0 {
            return Err(CudaAdaptiveBandpassTriggerOscillatorError::InvalidInput(
                format!("invalid range: start={start}, end={end}, step={step}"),
            ));
        }

        let mut values = Vec::new();
        let mut value = start;
        while value <= end + FLOAT_TOL {
            values.push(value.min(end));
            value += step;
        }
        if (values.last().copied().unwrap_or(start) - end).abs() > 1e-9 {
            return Err(CudaAdaptiveBandpassTriggerOscillatorError::InvalidInput(
                format!("invalid range: start={start}, end={end}, step={step}"),
            ));
        }
        Ok(values)
    }

    fn expand_grid(
        sweep: &AdaptiveBandpassTriggerOscillatorBatchRange,
    ) -> Result<
        Vec<AdaptiveBandpassTriggerOscillatorParams>,
        CudaAdaptiveBandpassTriggerOscillatorError,
    > {
        let deltas = Self::expand_axis_f64(sweep.delta.0, sweep.delta.1, sweep.delta.2)?;
        let alphas = Self::expand_axis_f64(sweep.alpha.0, sweep.alpha.1, sweep.alpha.2)?;
        let mut combos = Vec::with_capacity(deltas.len().saturating_mul(alphas.len()));
        for &delta in &deltas {
            for &alpha in &alphas {
                if !delta.is_finite() || delta <= 0.0 || delta >= 1.0 {
                    return Err(CudaAdaptiveBandpassTriggerOscillatorError::InvalidInput(
                        format!("invalid delta: {delta}"),
                    ));
                }
                if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
                    return Err(CudaAdaptiveBandpassTriggerOscillatorError::InvalidInput(
                        format!("invalid alpha: {alpha}"),
                    ));
                }
                combos.push(AdaptiveBandpassTriggerOscillatorParams {
                    delta: Some(delta),
                    alpha: Some(alpha),
                });
            }
        }
        Ok(combos)
    }

    fn will_fit(
        required: usize,
        headroom: usize,
    ) -> Result<(), CudaAdaptiveBandpassTriggerOscillatorError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaAdaptiveBandpassTriggerOscillatorError::OutOfMemory {
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
    ) -> Result<(), CudaAdaptiveBandpassTriggerOscillatorError> {
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
                CudaAdaptiveBandpassTriggerOscillatorError::LaunchConfigTooLarge {
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
        sweep: &AdaptiveBandpassTriggerOscillatorBatchRange,
    ) -> Result<
        CudaAdaptiveBandpassTriggerOscillatorBatchResult,
        CudaAdaptiveBandpassTriggerOscillatorError,
    > {
        if data.is_empty() {
            return Err(CudaAdaptiveBandpassTriggerOscillatorError::InvalidInput(
                "empty input".into(),
            ));
        }
        Self::first_valid_value(data).ok_or_else(|| {
            CudaAdaptiveBandpassTriggerOscillatorError::InvalidInput("all values are NaN".into())
        })?;
        let valid = Self::count_valid_values(data);
        if valid < MIN_VALID_SAMPLES {
            return Err(CudaAdaptiveBandpassTriggerOscillatorError::InvalidInput(
                format!("not enough valid data: needed={MIN_VALID_SAMPLES}, valid={valid}"),
            ));
        }

        let combos = Self::expand_grid(sweep)?;
        if combos.is_empty() {
            return Err(CudaAdaptiveBandpassTriggerOscillatorError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let rows = combos.len();
        let cols = data.len();
        let mut deltas = Vec::with_capacity(rows);
        let mut alphas = Vec::with_capacity(rows);
        for combo in &combos {
            deltas.push(combo.delta.unwrap_or(DEFAULT_DELTA));
            alphas.push(combo.alpha.unwrap_or(DEFAULT_ALPHA));
        }

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaAdaptiveBandpassTriggerOscillatorError::InvalidInput(
                    "input bytes overflow".into(),
                )
            })?;
        let params_bytes = rows
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| {
                rows.checked_mul(std::mem::size_of::<f64>())
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| {
                CudaAdaptiveBandpassTriggerOscillatorError::InvalidInput(
                    "params bytes overflow".into(),
                )
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaAdaptiveBandpassTriggerOscillatorError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                CudaAdaptiveBandpassTriggerOscillatorError::InvalidInput(
                    "output bytes overflow".into(),
                )
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaAdaptiveBandpassTriggerOscillatorError::InvalidInput(
                    "required bytes overflow".into(),
                )
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_data = DeviceBuffer::from_slice(data)?;
        let d_deltas = DeviceBuffer::from_slice(&deltas)?;
        let d_alphas = DeviceBuffer::from_slice(&alphas)?;
        let d_out_in_phase = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_lead = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("adaptive_bandpass_trigger_oscillator_batch_f64")
            .map_err(
                |_| CudaAdaptiveBandpassTriggerOscillatorError::MissingKernelSymbol {
                    name: "adaptive_bandpass_trigger_oscillator_batch_f64",
                },
            )?;
        let grid_x = ((rows as u32) + ADAPTIVE_BANDPASS_TRIGGER_OSCILLATOR_BLOCK_X - 1)
            / ADAPTIVE_BANDPASS_TRIGGER_OSCILLATOR_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(ADAPTIVE_BANDPASS_TRIGGER_OSCILLATOR_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_data.as_device_ptr(),
                cols as i32,
                d_deltas.as_device_ptr(),
                d_alphas.as_device_ptr(),
                rows as i32,
                d_out_in_phase.as_device_ptr(),
                d_out_lead.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaAdaptiveBandpassTriggerOscillatorBatchResult {
            outputs: AdaptiveBandpassTriggerOscillatorDeviceArrayF64Pair {
                in_phase: AdaptiveBandpassTriggerOscillatorDeviceArrayF64 {
                    buf: d_out_in_phase,
                    rows,
                    cols,
                },
                lead: AdaptiveBandpassTriggerOscillatorDeviceArrayF64 {
                    buf: d_out_lead,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
