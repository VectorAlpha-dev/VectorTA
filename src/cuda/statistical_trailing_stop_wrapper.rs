#![cfg(feature = "cuda")]

use crate::indicators::statistical_trailing_stop::{
    expand_grid, StatisticalTrailingStopBatchRange, StatisticalTrailingStopParams,
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

const STATISTICAL_TRAILING_STOP_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const DEFAULT_DATA_LENGTH: usize = 10;
const DEFAULT_NORMALIZATION_LENGTH: usize = 100;
const MIN_DATA_LENGTH: usize = 1;
const MIN_NORMALIZATION_LENGTH: usize = 10;

#[derive(Debug, Error)]
pub enum CudaStatisticalTrailingStopError {
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

pub struct StatisticalTrailingStopDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl StatisticalTrailingStopDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct StatisticalTrailingStopDeviceArrayF64Quad {
    pub level: StatisticalTrailingStopDeviceArrayF64,
    pub anchor: StatisticalTrailingStopDeviceArrayF64,
    pub bias: StatisticalTrailingStopDeviceArrayF64,
    pub changed: StatisticalTrailingStopDeviceArrayF64,
}

impl StatisticalTrailingStopDeviceArrayF64Quad {
    #[inline]
    pub fn rows(&self) -> usize {
        self.level.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.level.cols
    }
}

pub struct CudaStatisticalTrailingStopBatchResult {
    pub outputs: StatisticalTrailingStopDeviceArrayF64Quad,
    pub combos: Vec<StatisticalTrailingStopParams>,
}

pub struct CudaStatisticalTrailingStop {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

fn parse_base_level(value: &str) -> Result<i32, CudaStatisticalTrailingStopError> {
    let normalized = value
        .trim()
        .to_ascii_lowercase()
        .replace([' ', '_', '-'], "");
    match normalized.as_str() {
        "level0" | "0" => Ok(0),
        "level1" | "1" => Ok(1),
        "level2" | "2" => Ok(2),
        "level3" | "3" => Ok(3),
        _ => Err(CudaStatisticalTrailingStopError::InvalidInput(format!(
            "invalid base_level: {value}"
        ))),
    }
}

impl CudaStatisticalTrailingStop {
    pub fn new(device_id: usize) -> Result<Self, CudaStatisticalTrailingStopError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("statistical_trailing_stop_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaStatisticalTrailingStopError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn max_valid_run(high: &[f64], low: &[f64], close: &[f64]) -> usize {
        let mut best = 0usize;
        let mut run = 0usize;
        for i in 0..close.len() {
            if high[i].is_finite() && low[i].is_finite() && close[i].is_finite() {
                run += 1;
                best = best.max(run);
            } else {
                run = 0;
            }
        }
        best
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaStatisticalTrailingStopError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaStatisticalTrailingStopError::OutOfMemory {
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
    ) -> Result<(), CudaStatisticalTrailingStopError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaStatisticalTrailingStopError::LaunchConfigTooLarge {
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
        sweep: &StatisticalTrailingStopBatchRange,
    ) -> Result<CudaStatisticalTrailingStopBatchResult, CudaStatisticalTrailingStopError> {
        if high.is_empty() || low.is_empty() || close.is_empty() {
            return Err(CudaStatisticalTrailingStopError::InvalidInput(
                "empty input".into(),
            ));
        }
        if high.len() != low.len() || high.len() != close.len() {
            return Err(CudaStatisticalTrailingStopError::InvalidInput(format!(
                "input length mismatch: high={}, low={}, close={}",
                high.len(),
                low.len(),
                close.len()
            )));
        }

        let max_run = Self::max_valid_run(high, low, close);
        if max_run == 0 {
            return Err(CudaStatisticalTrailingStopError::InvalidInput(
                "all values are NaN".into(),
            ));
        }

        let combos = expand_grid(sweep)
            .map_err(|err| CudaStatisticalTrailingStopError::InvalidInput(err.to_string()))?;
        if combos.is_empty() {
            return Err(CudaStatisticalTrailingStopError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let rows = combos.len();
        let cols = close.len();
        let mut data_lengths = Vec::with_capacity(rows);
        let mut normalization_lengths = Vec::with_capacity(rows);
        let mut base_level_indices = Vec::with_capacity(rows);
        let mut max_data_length = 0usize;
        let mut max_normalization_length = 0usize;

        for combo in &combos {
            let data_length = combo.data_length.unwrap_or(DEFAULT_DATA_LENGTH);
            let normalization_length = combo
                .normalization_length
                .unwrap_or(DEFAULT_NORMALIZATION_LENGTH);
            let base_level_index =
                parse_base_level(combo.base_level.as_deref().unwrap_or("level2"))?;

            if data_length < MIN_DATA_LENGTH
                || normalization_length < MIN_NORMALIZATION_LENGTH
                || data_length + normalization_length + 1 > cols
            {
                return Err(CudaStatisticalTrailingStopError::InvalidInput(format!(
                    "invalid periods: data_length={data_length}, normalization_length={normalization_length}, data_len={cols}"
                )));
            }
            let needed = data_length + normalization_length + 1;
            if max_run < needed {
                return Err(CudaStatisticalTrailingStopError::InvalidInput(format!(
                    "not enough valid data: needed={needed}, valid={max_run}"
                )));
            }

            max_data_length = max_data_length.max(data_length);
            max_normalization_length = max_normalization_length.max(normalization_length);
            data_lengths.push(i32::try_from(data_length).map_err(|_| {
                CudaStatisticalTrailingStopError::InvalidInput(format!(
                    "data_length out of range: {data_length}"
                ))
            })?);
            normalization_lengths.push(i32::try_from(normalization_length).map_err(|_| {
                CudaStatisticalTrailingStopError::InvalidInput(format!(
                    "normalization_length out of range: {normalization_length}"
                ))
            })?);
            base_level_indices.push(base_level_index);
        }

        let rows_i32 = i32::try_from(rows).map_err(|_| {
            CudaStatisticalTrailingStopError::InvalidInput("rows out of range".into())
        })?;
        let cols_i32 = i32::try_from(cols).map_err(|_| {
            CudaStatisticalTrailingStopError::InvalidInput("cols out of range".into())
        })?;
        let deque_cap = max_data_length.checked_add(2).ok_or_else(|| {
            CudaStatisticalTrailingStopError::InvalidInput("deque cap overflow".into())
        })?;
        let deque_cap_i32 = i32::try_from(deque_cap).map_err(|_| {
            CudaStatisticalTrailingStopError::InvalidInput("deque cap out of range".into())
        })?;
        let stats_cap_i32 = i32::try_from(max_normalization_length).map_err(|_| {
            CudaStatisticalTrailingStopError::InvalidInput("stats cap out of range".into())
        })?;

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(3))
            .ok_or_else(|| {
                CudaStatisticalTrailingStopError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = rows
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| value.checked_mul(3))
            .ok_or_else(|| {
                CudaStatisticalTrailingStopError::InvalidInput("params bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaStatisticalTrailingStopError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| {
                CudaStatisticalTrailingStopError::InvalidInput("output bytes overflow".into())
            })?;
        let deque_elems = rows.checked_mul(deque_cap).ok_or_else(|| {
            CudaStatisticalTrailingStopError::InvalidInput("deque elems overflow".into())
        })?;
        let stats_elems = rows.checked_mul(max_normalization_length).ok_or_else(|| {
            CudaStatisticalTrailingStopError::InvalidInput("stats elems overflow".into())
        })?;
        let scratch_bytes = deque_elems
            .checked_mul(std::mem::size_of::<i32>() * 2 + std::mem::size_of::<f64>() * 3)
            .and_then(|value| {
                stats_elems
                    .checked_mul(std::mem::size_of::<f64>())
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| {
                CudaStatisticalTrailingStopError::InvalidInput("scratch bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .and_then(|value| value.checked_add(scratch_bytes))
            .ok_or_else(|| {
                CudaStatisticalTrailingStopError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_close = DeviceBuffer::from_slice(close)?;
        let d_data_lengths = DeviceBuffer::from_slice(&data_lengths)?;
        let d_normalization_lengths = DeviceBuffer::from_slice(&normalization_lengths)?;
        let d_base_level_indices = DeviceBuffer::from_slice(&base_level_indices)?;
        let d_out_level = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_anchor = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_bias = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_changed = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_max_high_idx = unsafe { DeviceBuffer::<i32>::uninitialized(deque_elems)? };
        let d_max_high_vals = unsafe { DeviceBuffer::<f64>::uninitialized(deque_elems)? };
        let d_min_low_idx = unsafe { DeviceBuffer::<i32>::uninitialized(deque_elems)? };
        let d_min_low_vals = unsafe { DeviceBuffer::<f64>::uninitialized(deque_elems)? };
        let d_close_history = unsafe { DeviceBuffer::<f64>::uninitialized(deque_elems)? };
        let d_stats_ring = unsafe { DeviceBuffer::<f64>::uninitialized(stats_elems)? };

        let func = self
            .module
            .get_function("statistical_trailing_stop_batch_f64")
            .map_err(|_| CudaStatisticalTrailingStopError::MissingKernelSymbol {
                name: "statistical_trailing_stop_batch_f64",
            })?;
        let grid_x = ((rows as u32) + STATISTICAL_TRAILING_STOP_BLOCK_X - 1)
            / STATISTICAL_TRAILING_STOP_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(STATISTICAL_TRAILING_STOP_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_high.as_device_ptr(),
                d_low.as_device_ptr(),
                d_close.as_device_ptr(),
                cols_i32,
                d_data_lengths.as_device_ptr(),
                d_normalization_lengths.as_device_ptr(),
                d_base_level_indices.as_device_ptr(),
                rows_i32,
                deque_cap_i32,
                stats_cap_i32,
                d_out_level.as_device_ptr(),
                d_out_anchor.as_device_ptr(),
                d_out_bias.as_device_ptr(),
                d_out_changed.as_device_ptr(),
                d_max_high_idx.as_device_ptr(),
                d_max_high_vals.as_device_ptr(),
                d_min_low_idx.as_device_ptr(),
                d_min_low_vals.as_device_ptr(),
                d_close_history.as_device_ptr(),
                d_stats_ring.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaStatisticalTrailingStopBatchResult {
            outputs: StatisticalTrailingStopDeviceArrayF64Quad {
                level: StatisticalTrailingStopDeviceArrayF64 {
                    buf: d_out_level,
                    rows,
                    cols,
                },
                anchor: StatisticalTrailingStopDeviceArrayF64 {
                    buf: d_out_anchor,
                    rows,
                    cols,
                },
                bias: StatisticalTrailingStopDeviceArrayF64 {
                    buf: d_out_bias,
                    rows,
                    cols,
                },
                changed: StatisticalTrailingStopDeviceArrayF64 {
                    buf: d_out_changed,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
