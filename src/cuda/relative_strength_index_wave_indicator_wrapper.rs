#![cfg(feature = "cuda")]

use crate::indicators::relative_strength_index_wave_indicator::{
    expand_grid, RelativeStrengthIndexWaveIndicatorBatchRange,
    RelativeStrengthIndexWaveIndicatorParams,
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

const RELATIVE_STRENGTH_INDEX_WAVE_INDICATOR_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const DEFAULT_RSI_LENGTH: usize = 14;
const DEFAULT_LENGTH1: usize = 2;
const DEFAULT_LENGTH2: usize = 5;
const DEFAULT_LENGTH3: usize = 9;
const DEFAULT_LENGTH4: usize = 13;

#[derive(Debug, Error)]
pub enum CudaRelativeStrengthIndexWaveIndicatorError {
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

pub struct RelativeStrengthIndexWaveIndicatorDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl RelativeStrengthIndexWaveIndicatorDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct RelativeStrengthIndexWaveIndicatorDeviceArrayF64Quint {
    pub rsi_ma1: RelativeStrengthIndexWaveIndicatorDeviceArrayF64,
    pub rsi_ma2: RelativeStrengthIndexWaveIndicatorDeviceArrayF64,
    pub rsi_ma3: RelativeStrengthIndexWaveIndicatorDeviceArrayF64,
    pub rsi_ma4: RelativeStrengthIndexWaveIndicatorDeviceArrayF64,
    pub state: RelativeStrengthIndexWaveIndicatorDeviceArrayF64,
}

impl RelativeStrengthIndexWaveIndicatorDeviceArrayF64Quint {
    #[inline]
    pub fn rows(&self) -> usize {
        self.rsi_ma1.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.rsi_ma1.cols
    }
}

pub struct CudaRelativeStrengthIndexWaveIndicatorBatchResult {
    pub outputs: RelativeStrengthIndexWaveIndicatorDeviceArrayF64Quint,
    pub combos: Vec<RelativeStrengthIndexWaveIndicatorParams>,
}

pub struct CudaRelativeStrengthIndexWaveIndicator {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaRelativeStrengthIndexWaveIndicator {
    pub fn new(device_id: usize) -> Result<Self, CudaRelativeStrengthIndexWaveIndicatorError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module =
            crate::load_cuda_embedded_module!("relative_strength_index_wave_indicator_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaRelativeStrengthIndexWaveIndicatorError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn first_valid_triplet(source: &[f64], high: &[f64], low: &[f64]) -> Option<usize> {
        (0..source.len())
            .find(|&i| source[i].is_finite() && high[i].is_finite() && low[i].is_finite())
    }

    fn will_fit(
        required: usize,
        headroom: usize,
    ) -> Result<(), CudaRelativeStrengthIndexWaveIndicatorError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaRelativeStrengthIndexWaveIndicatorError::OutOfMemory {
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
    ) -> Result<(), CudaRelativeStrengthIndexWaveIndicatorError> {
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
                CudaRelativeStrengthIndexWaveIndicatorError::LaunchConfigTooLarge {
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
        source: &[f64],
        high: &[f64],
        low: &[f64],
        sweep: &RelativeStrengthIndexWaveIndicatorBatchRange,
    ) -> Result<
        CudaRelativeStrengthIndexWaveIndicatorBatchResult,
        CudaRelativeStrengthIndexWaveIndicatorError,
    > {
        if source.is_empty() || high.is_empty() || low.is_empty() {
            return Err(CudaRelativeStrengthIndexWaveIndicatorError::InvalidInput(
                "empty input".into(),
            ));
        }
        if source.len() != high.len() || source.len() != low.len() {
            return Err(CudaRelativeStrengthIndexWaveIndicatorError::InvalidInput(
                format!(
                    "input length mismatch: source={}, high={}, low={}",
                    source.len(),
                    high.len(),
                    low.len()
                ),
            ));
        }
        Self::first_valid_triplet(source, high, low).ok_or_else(|| {
            CudaRelativeStrengthIndexWaveIndicatorError::InvalidInput("all values are NaN".into())
        })?;

        let combos = expand_grid(sweep).map_err(|err| {
            CudaRelativeStrengthIndexWaveIndicatorError::InvalidInput(err.to_string())
        })?;
        if combos.is_empty() {
            return Err(CudaRelativeStrengthIndexWaveIndicatorError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let rows = combos.len();
        let cols = source.len();
        let mut rsi_lengths = Vec::with_capacity(rows);
        let mut length1s = Vec::with_capacity(rows);
        let mut length2s = Vec::with_capacity(rows);
        let mut length3s = Vec::with_capacity(rows);
        let mut length4s = Vec::with_capacity(rows);

        for combo in &combos {
            let rsi_length = combo.rsi_length.unwrap_or(DEFAULT_RSI_LENGTH);
            let length1 = combo.length1.unwrap_or(DEFAULT_LENGTH1);
            let length2 = combo.length2.unwrap_or(DEFAULT_LENGTH2);
            let length3 = combo.length3.unwrap_or(DEFAULT_LENGTH3);
            let length4 = combo.length4.unwrap_or(DEFAULT_LENGTH4);
            if rsi_length == 0 || length1 == 0 || length2 == 0 || length3 == 0 || length4 == 0 {
                return Err(CudaRelativeStrengthIndexWaveIndicatorError::InvalidInput(
                    "invalid parameters".into(),
                ));
            }
            rsi_lengths.push(rsi_length as i32);
            length1s.push(length1 as i32);
            length2s.push(length2 as i32);
            length3s.push(length3 as i32);
            length4s.push(length4 as i32);
        }

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(3))
            .ok_or_else(|| {
                CudaRelativeStrengthIndexWaveIndicatorError::InvalidInput(
                    "input bytes overflow".into(),
                )
            })?;
        let params_bytes = rows
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| value.checked_mul(5))
            .ok_or_else(|| {
                CudaRelativeStrengthIndexWaveIndicatorError::InvalidInput(
                    "params bytes overflow".into(),
                )
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaRelativeStrengthIndexWaveIndicatorError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(5))
            .ok_or_else(|| {
                CudaRelativeStrengthIndexWaveIndicatorError::InvalidInput(
                    "output bytes overflow".into(),
                )
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaRelativeStrengthIndexWaveIndicatorError::InvalidInput(
                    "required bytes overflow".into(),
                )
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_source = DeviceBuffer::from_slice(source)?;
        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_rsi_lengths = DeviceBuffer::from_slice(&rsi_lengths)?;
        let d_length1s = DeviceBuffer::from_slice(&length1s)?;
        let d_length2s = DeviceBuffer::from_slice(&length2s)?;
        let d_length3s = DeviceBuffer::from_slice(&length3s)?;
        let d_length4s = DeviceBuffer::from_slice(&length4s)?;
        let d_out_ma1 = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_ma2 = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_ma3 = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_ma4 = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_state = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("relative_strength_index_wave_indicator_batch_f64")
            .map_err(
                |_| CudaRelativeStrengthIndexWaveIndicatorError::MissingKernelSymbol {
                    name: "relative_strength_index_wave_indicator_batch_f64",
                },
            )?;
        let grid_x = ((rows as u32) + RELATIVE_STRENGTH_INDEX_WAVE_INDICATOR_BLOCK_X - 1)
            / RELATIVE_STRENGTH_INDEX_WAVE_INDICATOR_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(RELATIVE_STRENGTH_INDEX_WAVE_INDICATOR_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_source.as_device_ptr(),
                d_high.as_device_ptr(),
                d_low.as_device_ptr(),
                cols as i32,
                d_rsi_lengths.as_device_ptr(),
                d_length1s.as_device_ptr(),
                d_length2s.as_device_ptr(),
                d_length3s.as_device_ptr(),
                d_length4s.as_device_ptr(),
                rows as i32,
                d_out_ma1.as_device_ptr(),
                d_out_ma2.as_device_ptr(),
                d_out_ma3.as_device_ptr(),
                d_out_ma4.as_device_ptr(),
                d_out_state.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaRelativeStrengthIndexWaveIndicatorBatchResult {
            outputs: RelativeStrengthIndexWaveIndicatorDeviceArrayF64Quint {
                rsi_ma1: RelativeStrengthIndexWaveIndicatorDeviceArrayF64 {
                    buf: d_out_ma1,
                    rows,
                    cols,
                },
                rsi_ma2: RelativeStrengthIndexWaveIndicatorDeviceArrayF64 {
                    buf: d_out_ma2,
                    rows,
                    cols,
                },
                rsi_ma3: RelativeStrengthIndexWaveIndicatorDeviceArrayF64 {
                    buf: d_out_ma3,
                    rows,
                    cols,
                },
                rsi_ma4: RelativeStrengthIndexWaveIndicatorDeviceArrayF64 {
                    buf: d_out_ma4,
                    rows,
                    cols,
                },
                state: RelativeStrengthIndexWaveIndicatorDeviceArrayF64 {
                    buf: d_out_state,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
