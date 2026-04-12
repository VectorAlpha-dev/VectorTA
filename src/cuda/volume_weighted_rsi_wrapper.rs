#![cfg(feature = "cuda")]

use crate::indicators::volume_weighted_rsi::{
    VolumeWeightedRsiBatchRange, VolumeWeightedRsiParams,
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

const VOLUME_WEIGHTED_RSI_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaVolumeWeightedRsiError {
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

pub struct VolumeWeightedRsiDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl VolumeWeightedRsiDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct CudaVolumeWeightedRsiBatchResult {
    pub outputs: VolumeWeightedRsiDeviceArrayF64,
    pub combos: Vec<VolumeWeightedRsiParams>,
}

pub struct CudaVolumeWeightedRsi {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaVolumeWeightedRsi {
    pub fn new(device_id: usize) -> Result<Self, CudaVolumeWeightedRsiError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("volume_weighted_rsi_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaVolumeWeightedRsiError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn expand_grid(
        range: &VolumeWeightedRsiBatchRange,
    ) -> Result<Vec<VolumeWeightedRsiParams>, CudaVolumeWeightedRsiError> {
        let (start, end, step) = range.period;
        if start == 0 || end == 0 {
            return Err(CudaVolumeWeightedRsiError::InvalidInput(format!(
                "invalid range: start={start}, end={end}, step={step}"
            )));
        }
        if step == 0 {
            return Ok(vec![VolumeWeightedRsiParams {
                period: Some(start),
            }]);
        }
        if start > end {
            return Err(CudaVolumeWeightedRsiError::InvalidInput(format!(
                "invalid range: start={start}, end={end}, step={step}"
            )));
        }

        let mut out = Vec::new();
        let mut cur = start;
        loop {
            out.push(VolumeWeightedRsiParams { period: Some(cur) });
            if cur >= end {
                break;
            }
            let next = cur.saturating_add(step);
            if next <= cur {
                return Err(CudaVolumeWeightedRsiError::InvalidInput(format!(
                    "invalid range: start={start}, end={end}, step={step}"
                )));
            }
            cur = next.min(end);
            if cur == out.last().and_then(|p| p.period).unwrap_or(cur) {
                break;
            }
        }
        Ok(out)
    }

    fn is_valid_pair(close: f64, volume: f64) -> bool {
        close.is_finite() && volume.is_finite()
    }

    fn longest_valid_pair_run(close: &[f64], volume: &[f64]) -> usize {
        let mut best = 0usize;
        let mut cur = 0usize;
        for (&c, &v) in close.iter().zip(volume.iter()) {
            if Self::is_valid_pair(c, v) {
                cur += 1;
                if cur > best {
                    best = cur;
                }
            } else {
                cur = 0;
            }
        }
        best
    }

    fn validate_common(
        close: &[f64],
        volume: &[f64],
        period: usize,
    ) -> Result<(), CudaVolumeWeightedRsiError> {
        if close.is_empty() || volume.is_empty() {
            return Err(CudaVolumeWeightedRsiError::InvalidInput(
                "empty input".into(),
            ));
        }
        if close.len() != volume.len() {
            return Err(CudaVolumeWeightedRsiError::InvalidInput(format!(
                "input length mismatch: close={}, volume={}",
                close.len(),
                volume.len()
            )));
        }
        if period == 0 || period > close.len() {
            return Err(CudaVolumeWeightedRsiError::InvalidInput(format!(
                "invalid period: period={period}, data_len={}",
                close.len()
            )));
        }

        let max_run = Self::longest_valid_pair_run(close, volume);
        if max_run == 0 {
            return Err(CudaVolumeWeightedRsiError::InvalidInput(
                "all values are NaN".into(),
            ));
        }
        if max_run < period {
            return Err(CudaVolumeWeightedRsiError::InvalidInput(format!(
                "not enough valid data: needed={period}, valid={max_run}"
            )));
        }
        Ok(())
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaVolumeWeightedRsiError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaVolumeWeightedRsiError::OutOfMemory {
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
    ) -> Result<(), CudaVolumeWeightedRsiError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaVolumeWeightedRsiError::LaunchConfigTooLarge {
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
        sweep: &VolumeWeightedRsiBatchRange,
    ) -> Result<CudaVolumeWeightedRsiBatchResult, CudaVolumeWeightedRsiError> {
        Self::validate_common(close, volume, 1)?;
        let combos = Self::expand_grid(sweep)?;
        let max_period = combos
            .iter()
            .map(|params| params.period.unwrap_or(14))
            .max()
            .unwrap_or(0);
        Self::validate_common(close, volume, max_period)?;

        let rows = combos.len();
        let cols = close.len();
        let periods: Vec<i32> = combos
            .iter()
            .map(|params| params.period.unwrap_or(14) as i32)
            .collect();

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|v| v.checked_mul(2))
            .ok_or_else(|| {
                CudaVolumeWeightedRsiError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = periods
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| {
                CudaVolumeWeightedRsiError::InvalidInput("params bytes overflow".into())
            })?;
        let output_elems = rows
            .checked_mul(cols)
            .ok_or_else(|| CudaVolumeWeightedRsiError::InvalidInput("rows*cols overflow".into()))?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaVolumeWeightedRsiError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|v| v.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaVolumeWeightedRsiError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_close = DeviceBuffer::from_slice(close)?;
        let d_volume = DeviceBuffer::from_slice(volume)?;
        let d_periods = DeviceBuffer::from_slice(&periods)?;
        let mut d_out = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("volume_weighted_rsi_batch_f64")
            .map_err(|_| CudaVolumeWeightedRsiError::MissingKernelSymbol {
                name: "volume_weighted_rsi_batch_f64",
            })?;
        let grid_x =
            ((rows as u32) + VOLUME_WEIGHTED_RSI_BLOCK_X - 1) / VOLUME_WEIGHTED_RSI_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(VOLUME_WEIGHTED_RSI_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_close.as_device_ptr(),
                d_volume.as_device_ptr(),
                cols as i32,
                d_periods.as_device_ptr(),
                rows as i32,
                d_out.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaVolumeWeightedRsiBatchResult {
            outputs: VolumeWeightedRsiDeviceArrayF64 {
                buf: d_out,
                rows,
                cols,
            },
            combos,
        })
    }
}
