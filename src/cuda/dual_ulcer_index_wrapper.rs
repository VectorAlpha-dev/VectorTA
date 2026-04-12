#![cfg(feature = "cuda")]

use crate::indicators::dual_ulcer_index::{DualUlcerIndexBatchRange, DualUlcerIndexParams};
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

const DUAL_ULCER_INDEX_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaDualUlcerIndexError {
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

pub struct DeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl DeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct DualUlcerIndexDeviceArrayF64Triplet {
    pub long_ulcer: DeviceArrayF64,
    pub short_ulcer: DeviceArrayF64,
    pub threshold: DeviceArrayF64,
}

impl DualUlcerIndexDeviceArrayF64Triplet {
    #[inline]
    pub fn rows(&self) -> usize {
        self.long_ulcer.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.long_ulcer.cols
    }
}

pub struct CudaDualUlcerIndexBatchResult {
    pub outputs: DualUlcerIndexDeviceArrayF64Triplet,
    pub combos: Vec<DualUlcerIndexParams>,
}

pub struct CudaDualUlcerIndex {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaDualUlcerIndex {
    pub fn new(device_id: usize) -> Result<Self, CudaDualUlcerIndexError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("dual_ulcer_index_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaDualUlcerIndexError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn axis_usize(
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, CudaDualUlcerIndexError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }

        let mut out = Vec::new();
        if start < end {
            let mut cur = start;
            while cur <= end {
                out.push(cur);
                let next = cur.saturating_add(step.max(1));
                if next == cur {
                    break;
                }
                cur = next;
            }
        } else {
            let mut cur = start;
            loop {
                out.push(cur);
                if cur == end {
                    break;
                }
                let next = cur.saturating_sub(step.max(1));
                if next == cur || next < end {
                    break;
                }
                cur = next;
            }
        }

        if out.is_empty() {
            return Err(CudaDualUlcerIndexError::InvalidInput(format!(
                "invalid usize range: start={start}, end={end}, step={step}"
            )));
        }
        Ok(out)
    }

    fn axis_f64((start, end, step): (f64, f64, f64)) -> Result<Vec<f64>, CudaDualUlcerIndexError> {
        if !start.is_finite() || !end.is_finite() || !step.is_finite() {
            return Err(CudaDualUlcerIndexError::InvalidInput(format!(
                "invalid float range: start={start}, end={end}, step={step}"
            )));
        }
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return Ok(vec![start]);
        }

        let mut out = Vec::new();
        if start < end {
            let delta = step.abs();
            let mut cur = start;
            while cur <= end + 1e-12 {
                out.push(cur);
                cur += delta;
            }
        } else {
            let delta = step.abs();
            let mut cur = start;
            while cur >= end - 1e-12 {
                out.push(cur);
                cur -= delta;
            }
        }

        if out.is_empty() {
            return Err(CudaDualUlcerIndexError::InvalidInput(format!(
                "invalid float range: start={start}, end={end}, step={step}"
            )));
        }
        Ok(out)
    }

    fn expand_grid(
        sweep: &DualUlcerIndexBatchRange,
    ) -> Result<Vec<DualUlcerIndexParams>, CudaDualUlcerIndexError> {
        let periods = Self::axis_usize(sweep.period)?;
        let thresholds = Self::axis_f64(sweep.threshold)?;
        let mut combos = Vec::with_capacity(periods.len().saturating_mul(thresholds.len()));

        for period in periods {
            if period == 0 {
                return Err(CudaDualUlcerIndexError::InvalidInput(
                    "period must be > 0".into(),
                ));
            }
            for threshold in &thresholds {
                if !threshold.is_finite() || *threshold < 0.0 {
                    return Err(CudaDualUlcerIndexError::InvalidInput(format!(
                        "invalid threshold {threshold}"
                    )));
                }
                combos.push(DualUlcerIndexParams {
                    period: Some(period),
                    auto_threshold: Some(sweep.auto_threshold),
                    threshold: Some(*threshold),
                });
            }
        }

        Ok(combos)
    }

    fn longest_valid_run(data: &[f64]) -> usize {
        let mut best = 0usize;
        let mut cur = 0usize;
        for &value in data {
            if value.is_finite() && value > 0.0 {
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

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaDualUlcerIndexError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaDualUlcerIndexError::OutOfMemory {
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
    ) -> Result<(), CudaDualUlcerIndexError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaDualUlcerIndexError::LaunchConfigTooLarge {
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
        sweep: &DualUlcerIndexBatchRange,
    ) -> Result<CudaDualUlcerIndexBatchResult, CudaDualUlcerIndexError> {
        if data.is_empty() {
            return Err(CudaDualUlcerIndexError::InvalidInput("empty data".into()));
        }

        let combos = Self::expand_grid(sweep)?;
        let max_period = combos
            .iter()
            .map(|combo| combo.period.unwrap_or(5))
            .max()
            .unwrap_or(0);
        if max_period > data.len() {
            return Err(CudaDualUlcerIndexError::InvalidInput(format!(
                "invalid period: period={max_period}, data_len={}",
                data.len()
            )));
        }
        let longest_run = Self::longest_valid_run(data);
        if longest_run == 0 {
            return Err(CudaDualUlcerIndexError::InvalidInput(
                "all values are NaN or non-positive".into(),
            ));
        }
        let needed = max_period
            .checked_mul(2)
            .and_then(|v| v.checked_sub(1))
            .ok_or_else(|| CudaDualUlcerIndexError::InvalidInput("period overflow".into()))?;
        if longest_run < needed {
            return Err(CudaDualUlcerIndexError::InvalidInput(format!(
                "not enough valid data: needed={needed}, valid={longest_run}"
            )));
        }

        let rows = combos.len();
        let cols = data.len();
        let periods: Vec<i32> = combos
            .iter()
            .map(|combo| combo.period.unwrap_or(5) as i32)
            .collect();
        let thresholds: Vec<f64> = combos
            .iter()
            .map(|combo| combo.threshold.unwrap_or(0.1))
            .collect();

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| CudaDualUlcerIndexError::InvalidInput("input bytes overflow".into()))?;
        let params_bytes = periods
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|v| {
                thresholds
                    .len()
                    .checked_mul(std::mem::size_of::<f64>())
                    .and_then(|w| v.checked_add(w))
            })
            .ok_or_else(|| CudaDualUlcerIndexError::InvalidInput("params bytes overflow".into()))?;
        let output_elems = rows
            .checked_mul(cols)
            .ok_or_else(|| CudaDualUlcerIndexError::InvalidInput("rows*cols overflow".into()))?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|v| v.checked_mul(3))
            .ok_or_else(|| CudaDualUlcerIndexError::InvalidInput("output bytes overflow".into()))?;
        let temp_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|v| v.checked_mul(2))
            .ok_or_else(|| CudaDualUlcerIndexError::InvalidInput("temp bytes overflow".into()))?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|v| v.checked_add(temp_bytes))
            .and_then(|v| v.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaDualUlcerIndexError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_data = DeviceBuffer::from_slice(data)?;
        let d_periods = DeviceBuffer::from_slice(&periods)?;
        let d_thresholds = DeviceBuffer::from_slice(&thresholds)?;
        let mut d_long_sq = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_short_sq = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_long = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_short = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_threshold = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let build_func = self
            .module
            .get_function("dual_ulcer_index_build_squares_f64")
            .map_err(|_| CudaDualUlcerIndexError::MissingKernelSymbol {
                name: "dual_ulcer_index_build_squares_f64",
            })?;
        let finalize_func = self
            .module
            .get_function("dual_ulcer_index_finalize_f64")
            .map_err(|_| CudaDualUlcerIndexError::MissingKernelSymbol {
                name: "dual_ulcer_index_finalize_f64",
            })?;
        let grid_x = ((rows as u32) + DUAL_ULCER_INDEX_BLOCK_X - 1) / DUAL_ULCER_INDEX_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(DUAL_ULCER_INDEX_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(build_func<<<grid, block, 0, stream>>>(
                d_data.as_device_ptr(),
                cols as i32,
                d_periods.as_device_ptr(),
                rows as i32,
                d_long_sq.as_device_ptr(),
                d_short_sq.as_device_ptr()
            ))?;
            launch!(finalize_func<<<grid, block, 0, stream>>>(
                d_long_sq.as_device_ptr(),
                d_short_sq.as_device_ptr(),
                cols as i32,
                d_periods.as_device_ptr(),
                d_thresholds.as_device_ptr(),
                if sweep.auto_threshold { 1 } else { 0 },
                rows as i32,
                d_long.as_device_ptr(),
                d_short.as_device_ptr(),
                d_threshold.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaDualUlcerIndexBatchResult {
            outputs: DualUlcerIndexDeviceArrayF64Triplet {
                long_ulcer: DeviceArrayF64 {
                    buf: d_long,
                    rows,
                    cols,
                },
                short_ulcer: DeviceArrayF64 {
                    buf: d_short,
                    rows,
                    cols,
                },
                threshold: DeviceArrayF64 {
                    buf: d_threshold,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
