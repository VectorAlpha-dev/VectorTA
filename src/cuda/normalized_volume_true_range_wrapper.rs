#![cfg(feature = "cuda")]

use crate::indicators::normalized_volume_true_range::{
    expand_grid_normalized_volume_true_range, NormalizedVolumeTrueRangeBatchRange,
    NormalizedVolumeTrueRangeParams, NormalizedVolumeTrueRangeStyle,
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

const NORMALIZED_VOLUME_TRUE_RANGE_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaNormalizedVolumeTrueRangeError {
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

pub struct NormalizedVolumeTrueRangeDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl NormalizedVolumeTrueRangeDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct NormalizedVolumeTrueRangeDeviceArrayF64Quint {
    pub normalized_volume: NormalizedVolumeTrueRangeDeviceArrayF64,
    pub normalized_true_range: NormalizedVolumeTrueRangeDeviceArrayF64,
    pub baseline: NormalizedVolumeTrueRangeDeviceArrayF64,
    pub atr: NormalizedVolumeTrueRangeDeviceArrayF64,
    pub average_volume: NormalizedVolumeTrueRangeDeviceArrayF64,
}

impl NormalizedVolumeTrueRangeDeviceArrayF64Quint {
    #[inline]
    pub fn rows(&self) -> usize {
        self.normalized_volume.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.normalized_volume.cols
    }
}

pub struct CudaNormalizedVolumeTrueRangeBatchResult {
    pub outputs: NormalizedVolumeTrueRangeDeviceArrayF64Quint,
    pub combos: Vec<NormalizedVolumeTrueRangeParams>,
}

pub struct CudaNormalizedVolumeTrueRange {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaNormalizedVolumeTrueRange {
    pub fn new(device_id: usize) -> Result<Self, CudaNormalizedVolumeTrueRangeError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("normalized_volume_true_range_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaNormalizedVolumeTrueRangeError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn style_id(style: NormalizedVolumeTrueRangeStyle) -> i32 {
        match style {
            NormalizedVolumeTrueRangeStyle::Body => 0,
            NormalizedVolumeTrueRangeStyle::Hl => 1,
            NormalizedVolumeTrueRangeStyle::Delta => 2,
        }
    }

    fn any_valid(
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
        style: NormalizedVolumeTrueRangeStyle,
    ) -> bool {
        (0..close.len()).any(|idx| match style {
            NormalizedVolumeTrueRangeStyle::Body => {
                open[idx].is_finite() && close[idx].is_finite() && volume[idx].is_finite()
            }
            NormalizedVolumeTrueRangeStyle::Hl => {
                high[idx].is_finite() && low[idx].is_finite() && volume[idx].is_finite()
            }
            NormalizedVolumeTrueRangeStyle::Delta => {
                close[idx].is_finite() && volume[idx].is_finite()
            }
        })
    }

    fn will_fit(
        required: usize,
        headroom: usize,
    ) -> Result<(), CudaNormalizedVolumeTrueRangeError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaNormalizedVolumeTrueRangeError::OutOfMemory {
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
    ) -> Result<(), CudaNormalizedVolumeTrueRangeError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaNormalizedVolumeTrueRangeError::LaunchConfigTooLarge {
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
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
        sweep: &NormalizedVolumeTrueRangeBatchRange,
    ) -> Result<CudaNormalizedVolumeTrueRangeBatchResult, CudaNormalizedVolumeTrueRangeError> {
        if open.is_empty() {
            return Err(CudaNormalizedVolumeTrueRangeError::InvalidInput(
                "empty input".into(),
            ));
        }
        if open.len() != high.len()
            || high.len() != low.len()
            || low.len() != close.len()
            || close.len() != volume.len()
        {
            return Err(CudaNormalizedVolumeTrueRangeError::InvalidInput(format!(
                "length mismatch: open={}, high={}, low={}, close={}, volume={}",
                open.len(),
                high.len(),
                low.len(),
                close.len(),
                volume.len()
            )));
        }

        let combos = expand_grid_normalized_volume_true_range(sweep)
            .map_err(|err| CudaNormalizedVolumeTrueRangeError::InvalidInput(err.to_string()))?;
        if combos.is_empty() {
            return Err(CudaNormalizedVolumeTrueRangeError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let style = sweep.true_range_style.unwrap_or_default();
        if !Self::any_valid(open, high, low, close, volume, style) {
            return Err(CudaNormalizedVolumeTrueRangeError::InvalidInput(
                "all values are NaN".into(),
            ));
        }

        let rows = combos.len();
        let cols = close.len();
        let outlier_ranges: Vec<f64> = combos
            .iter()
            .map(|params| params.outlier_range.unwrap_or(5.0))
            .collect();
        let atr_lengths: Vec<i32> = combos
            .iter()
            .map(|params| params.atr_length.unwrap_or(14) as i32)
            .collect();
        let volume_lengths: Vec<i32> = combos
            .iter()
            .map(|params| params.volume_length.unwrap_or(14) as i32)
            .collect();
        let styles: Vec<i32> = combos
            .iter()
            .map(|params| Self::style_id(params.true_range_style.unwrap_or_default()))
            .collect();

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(5))
            .ok_or_else(|| {
                CudaNormalizedVolumeTrueRangeError::InvalidInput("input bytes overflow".into())
            })?;
        let param_bytes = rows
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| {
                rows.checked_mul(std::mem::size_of::<i32>())
                    .and_then(|other| value.checked_add(other.checked_mul(3)?))
            })
            .ok_or_else(|| {
                CudaNormalizedVolumeTrueRangeError::InvalidInput("parameter bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaNormalizedVolumeTrueRangeError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(5))
            .ok_or_else(|| {
                CudaNormalizedVolumeTrueRangeError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(param_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaNormalizedVolumeTrueRangeError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_open = DeviceBuffer::from_slice(open)?;
        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_close = DeviceBuffer::from_slice(close)?;
        let d_volume = DeviceBuffer::from_slice(volume)?;
        let d_outlier_ranges = DeviceBuffer::from_slice(&outlier_ranges)?;
        let d_atr_lengths = DeviceBuffer::from_slice(&atr_lengths)?;
        let d_volume_lengths = DeviceBuffer::from_slice(&volume_lengths)?;
        let d_styles = DeviceBuffer::from_slice(&styles)?;
        let d_out_nv = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_ntr = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_baseline = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_atr = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_avg_vol = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("normalized_volume_true_range_batch_f64")
            .map_err(
                |_| CudaNormalizedVolumeTrueRangeError::MissingKernelSymbol {
                    name: "normalized_volume_true_range_batch_f64",
                },
            )?;
        let grid_x = ((rows as u32) + NORMALIZED_VOLUME_TRUE_RANGE_BLOCK_X - 1)
            / NORMALIZED_VOLUME_TRUE_RANGE_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(NORMALIZED_VOLUME_TRUE_RANGE_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_open.as_device_ptr(),
                d_high.as_device_ptr(),
                d_low.as_device_ptr(),
                d_close.as_device_ptr(),
                d_volume.as_device_ptr(),
                cols as i32,
                d_outlier_ranges.as_device_ptr(),
                d_atr_lengths.as_device_ptr(),
                d_volume_lengths.as_device_ptr(),
                d_styles.as_device_ptr(),
                rows as i32,
                d_out_nv.as_device_ptr(),
                d_out_ntr.as_device_ptr(),
                d_out_baseline.as_device_ptr(),
                d_out_atr.as_device_ptr(),
                d_out_avg_vol.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaNormalizedVolumeTrueRangeBatchResult {
            outputs: NormalizedVolumeTrueRangeDeviceArrayF64Quint {
                normalized_volume: NormalizedVolumeTrueRangeDeviceArrayF64 {
                    buf: d_out_nv,
                    rows,
                    cols,
                },
                normalized_true_range: NormalizedVolumeTrueRangeDeviceArrayF64 {
                    buf: d_out_ntr,
                    rows,
                    cols,
                },
                baseline: NormalizedVolumeTrueRangeDeviceArrayF64 {
                    buf: d_out_baseline,
                    rows,
                    cols,
                },
                atr: NormalizedVolumeTrueRangeDeviceArrayF64 {
                    buf: d_out_atr,
                    rows,
                    cols,
                },
                average_volume: NormalizedVolumeTrueRangeDeviceArrayF64 {
                    buf: d_out_avg_vol,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
