#![cfg(feature = "cuda")]

use crate::indicators::volatility_quality_index::{
    VolatilityQualityIndexBatchRange, VolatilityQualityIndexParams,
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

const VOLATILITY_QUALITY_INDEX_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaVolatilityQualityIndexError {
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

pub struct VolatilityQualityIndexDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl VolatilityQualityIndexDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct VolatilityQualityIndexDeviceArrayF64Triplet {
    pub vqi_sum: VolatilityQualityIndexDeviceArrayF64,
    pub fast_sma: VolatilityQualityIndexDeviceArrayF64,
    pub slow_sma: VolatilityQualityIndexDeviceArrayF64,
}

impl VolatilityQualityIndexDeviceArrayF64Triplet {
    #[inline]
    pub fn rows(&self) -> usize {
        self.vqi_sum.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.vqi_sum.cols
    }
}

pub struct CudaVolatilityQualityIndexBatchResult {
    pub outputs: VolatilityQualityIndexDeviceArrayF64Triplet,
    pub combos: Vec<VolatilityQualityIndexParams>,
}

pub struct CudaVolatilityQualityIndex {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaVolatilityQualityIndex {
    pub fn new(device_id: usize) -> Result<Self, CudaVolatilityQualityIndexError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("volatility_quality_index_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaVolatilityQualityIndexError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn axis(
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, CudaVolatilityQualityIndexError> {
        if start == end {
            return Ok(vec![start]);
        }
        if step == 0 {
            return Err(CudaVolatilityQualityIndexError::InvalidInput(format!(
                "invalid range: start={start}, end={end}, step={step}"
            )));
        }

        let mut out = Vec::new();
        if start < end {
            let mut value = start;
            while value <= end {
                out.push(value);
                match value.checked_add(step) {
                    Some(next) if next > value => value = next,
                    _ => break,
                }
            }
        } else {
            let mut value = start;
            while value >= end {
                out.push(value);
                match value.checked_sub(step) {
                    Some(next) if next < value => value = next,
                    _ => break,
                }
            }
        }

        if out.is_empty() {
            return Err(CudaVolatilityQualityIndexError::InvalidInput(format!(
                "invalid range: start={start}, end={end}, step={step}"
            )));
        }
        Ok(out)
    }

    fn expand_grid(
        sweep: &VolatilityQualityIndexBatchRange,
    ) -> Result<Vec<VolatilityQualityIndexParams>, CudaVolatilityQualityIndexError> {
        let fast_values = Self::axis(sweep.fast_length)?;
        let slow_values = Self::axis(sweep.slow_length)?;
        let mut combos = Vec::with_capacity(fast_values.len().saturating_mul(slow_values.len()));
        for fast_length in fast_values {
            if fast_length == 0 {
                return Err(CudaVolatilityQualityIndexError::InvalidInput(
                    "fast_length must be > 0".into(),
                ));
            }
            for slow_length in &slow_values {
                if *slow_length == 0 {
                    return Err(CudaVolatilityQualityIndexError::InvalidInput(
                        "slow_length must be > 0".into(),
                    ));
                }
                combos.push(VolatilityQualityIndexParams {
                    fast_length: Some(fast_length),
                    slow_length: Some(*slow_length),
                });
            }
        }
        Ok(combos)
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaVolatilityQualityIndexError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaVolatilityQualityIndexError::OutOfMemory {
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
    ) -> Result<(), CudaVolatilityQualityIndexError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaVolatilityQualityIndexError::LaunchConfigTooLarge {
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
        sweep: &VolatilityQualityIndexBatchRange,
    ) -> Result<CudaVolatilityQualityIndexBatchResult, CudaVolatilityQualityIndexError> {
        let len = close.len();
        if open.is_empty() || high.is_empty() || low.is_empty() || close.is_empty() {
            return Err(CudaVolatilityQualityIndexError::InvalidInput(
                "empty input".into(),
            ));
        }
        if open.len() != len || high.len() != len || low.len() != len {
            return Err(CudaVolatilityQualityIndexError::InvalidInput(format!(
                "inconsistent slice lengths: open={}, high={}, low={}, close={}",
                open.len(),
                high.len(),
                low.len(),
                close.len()
            )));
        }
        if !open
            .iter()
            .zip(high.iter())
            .zip(low.iter())
            .zip(close.iter())
            .any(|(((o, h), l), c)| {
                o.is_finite() || h.is_finite() || l.is_finite() || c.is_finite()
            })
        {
            return Err(CudaVolatilityQualityIndexError::InvalidInput(
                "all values are NaN".into(),
            ));
        }

        let combos = Self::expand_grid(sweep)?;
        let rows = combos.len();
        let cols = len;
        let fast_lengths: Vec<i32> = combos
            .iter()
            .map(|combo| combo.fast_length.unwrap_or(9) as i32)
            .collect();
        let slow_lengths: Vec<i32> = combos
            .iter()
            .map(|combo| combo.slow_length.unwrap_or(200) as i32)
            .collect();

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|v| v.checked_mul(4))
            .ok_or_else(|| {
                CudaVolatilityQualityIndexError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = fast_lengths
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|v| {
                slow_lengths
                    .len()
                    .checked_mul(std::mem::size_of::<i32>())
                    .and_then(|w| v.checked_add(w))
            })
            .ok_or_else(|| {
                CudaVolatilityQualityIndexError::InvalidInput("params bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaVolatilityQualityIndexError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|v| v.checked_mul(3))
            .ok_or_else(|| {
                CudaVolatilityQualityIndexError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|v| v.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaVolatilityQualityIndexError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_open = DeviceBuffer::from_slice(open)?;
        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_close = DeviceBuffer::from_slice(close)?;
        let d_fast_lengths = DeviceBuffer::from_slice(&fast_lengths)?;
        let d_slow_lengths = DeviceBuffer::from_slice(&slow_lengths)?;
        let mut d_vqi_sum = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_fast_sma = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_slow_sma = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("volatility_quality_index_batch_f64")
            .map_err(|_| CudaVolatilityQualityIndexError::MissingKernelSymbol {
                name: "volatility_quality_index_batch_f64",
            })?;
        let grid_x = ((rows as u32) + VOLATILITY_QUALITY_INDEX_BLOCK_X - 1)
            / VOLATILITY_QUALITY_INDEX_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(VOLATILITY_QUALITY_INDEX_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_open.as_device_ptr(),
                d_high.as_device_ptr(),
                d_low.as_device_ptr(),
                d_close.as_device_ptr(),
                cols as i32,
                d_fast_lengths.as_device_ptr(),
                d_slow_lengths.as_device_ptr(),
                rows as i32,
                d_vqi_sum.as_device_ptr(),
                d_fast_sma.as_device_ptr(),
                d_slow_sma.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaVolatilityQualityIndexBatchResult {
            outputs: VolatilityQualityIndexDeviceArrayF64Triplet {
                vqi_sum: VolatilityQualityIndexDeviceArrayF64 {
                    buf: d_vqi_sum,
                    rows,
                    cols,
                },
                fast_sma: VolatilityQualityIndexDeviceArrayF64 {
                    buf: d_fast_sma,
                    rows,
                    cols,
                },
                slow_sma: VolatilityQualityIndexDeviceArrayF64 {
                    buf: d_slow_sma,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
