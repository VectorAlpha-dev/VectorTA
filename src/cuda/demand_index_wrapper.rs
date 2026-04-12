#![cfg(feature = "cuda")]

use crate::indicators::demand_index::{DemandIndexBatchRange, DemandIndexParams};
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

const DEMAND_INDEX_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const MA_EMA: i32 = 0;
const MA_SMA: i32 = 1;
const MA_WMA: i32 = 2;
const MA_RMA: i32 = 3;

#[derive(Debug, Error)]
pub enum CudaDemandIndexError {
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

pub struct DemandIndexDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl DemandIndexDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct DemandIndexDeviceArrayF64Pair {
    pub demand_index: DemandIndexDeviceArrayF64,
    pub signal: DemandIndexDeviceArrayF64,
}

impl DemandIndexDeviceArrayF64Pair {
    #[inline]
    pub fn rows(&self) -> usize {
        self.demand_index.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.demand_index.cols
    }
}

pub struct CudaDemandIndexBatchResult {
    pub outputs: DemandIndexDeviceArrayF64Pair,
    pub combos: Vec<DemandIndexParams>,
}

pub struct CudaDemandIndex {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

#[inline]
fn parse_ma_type(value: &str) -> Result<i32, CudaDemandIndexError> {
    if value.eq_ignore_ascii_case("ema") {
        Ok(MA_EMA)
    } else if value.eq_ignore_ascii_case("sma") {
        Ok(MA_SMA)
    } else if value.eq_ignore_ascii_case("wma") {
        Ok(MA_WMA)
    } else if value.eq_ignore_ascii_case("rma") || value.eq_ignore_ascii_case("wilders") {
        Ok(MA_RMA)
    } else {
        Err(CudaDemandIndexError::InvalidInput(format!(
            "invalid ma_type: {value}"
        )))
    }
}

#[inline]
fn expand_axis_usize(
    (start, end, step): (usize, usize, usize),
) -> Result<Vec<usize>, CudaDemandIndexError> {
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
        return Err(CudaDemandIndexError::InvalidInput(format!(
            "invalid range: start={start}, end={end}, step={step}"
        )));
    }
    Ok(out)
}

#[inline]
fn expand_grid(
    range: &DemandIndexBatchRange,
) -> Result<Vec<DemandIndexParams>, CudaDemandIndexError> {
    let len_bs_values = expand_axis_usize(range.len_bs)?;
    let len_bs_ma_values = expand_axis_usize(range.len_bs_ma)?;
    let len_di_ma_values = expand_axis_usize(range.len_di_ma)?;
    let ma_type = range.ma_type.clone().unwrap_or_else(|| "ema".to_string());
    let _ = parse_ma_type(&ma_type)?;

    let mut out =
        Vec::with_capacity(len_bs_values.len() * len_bs_ma_values.len() * len_di_ma_values.len());
    for &len_bs in &len_bs_values {
        for &len_bs_ma in &len_bs_ma_values {
            for &len_di_ma in &len_di_ma_values {
                if len_bs == 0 || len_bs_ma == 0 || len_di_ma == 0 {
                    return Err(CudaDemandIndexError::InvalidInput(
                        "invalid lengths in parameter grid".into(),
                    ));
                }
                out.push(DemandIndexParams {
                    len_bs: Some(len_bs),
                    len_bs_ma: Some(len_bs_ma),
                    len_di_ma: Some(len_di_ma),
                    ma_type: Some(ma_type.clone()),
                });
            }
        }
    }
    Ok(out)
}

#[inline]
fn valid_ohlcv(high: f64, low: f64, close: f64, volume: f64) -> bool {
    high.is_finite() && low.is_finite() && close.is_finite() && volume.is_finite()
}

#[inline]
fn first_valid_ohlcv(high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Option<usize> {
    (0..high.len()).find(|&i| valid_ohlcv(high[i], low[i], close[i], volume[i]))
}

#[inline]
fn count_valid_ohlcv(high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> usize {
    (0..high.len())
        .filter(|&i| valid_ohlcv(high[i], low[i], close[i], volume[i]))
        .count()
}

#[inline]
fn di_warmup(ma_code: i32, len_bs: usize, len_bs_ma: usize) -> usize {
    match ma_code {
        MA_EMA | MA_RMA => 1,
        _ => len_bs.saturating_sub(1).max(1) + len_bs_ma.saturating_sub(1),
    }
}

#[inline]
fn signal_warmup(ma_code: i32, len_bs: usize, len_bs_ma: usize, len_di_ma: usize) -> usize {
    di_warmup(ma_code, len_bs, len_bs_ma) + len_di_ma.saturating_sub(1)
}

#[inline]
fn needed_valid_bars(ma_code: i32, len_bs: usize, len_bs_ma: usize, len_di_ma: usize) -> usize {
    signal_warmup(ma_code, len_bs, len_bs_ma, len_di_ma) + 1
}

impl CudaDemandIndex {
    pub fn new(device_id: usize) -> Result<Self, CudaDemandIndexError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("demand_index_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaDemandIndexError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaDemandIndexError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaDemandIndexError::OutOfMemory {
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
    ) -> Result<(), CudaDemandIndexError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaDemandIndexError::LaunchConfigTooLarge {
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
        sweep: &DemandIndexBatchRange,
    ) -> Result<CudaDemandIndexBatchResult, CudaDemandIndexError> {
        if high.is_empty() || low.is_empty() || close.is_empty() || volume.is_empty() {
            return Err(CudaDemandIndexError::InvalidInput("empty input".into()));
        }
        if high.len() != low.len() || high.len() != close.len() || high.len() != volume.len() {
            return Err(CudaDemandIndexError::InvalidInput(format!(
                "input length mismatch: high={}, low={}, close={}, volume={}",
                high.len(),
                low.len(),
                close.len(),
                volume.len()
            )));
        }

        let first = first_valid_ohlcv(high, low, close, volume)
            .ok_or_else(|| CudaDemandIndexError::InvalidInput("all values are NaN".into()))?;
        let valid = count_valid_ohlcv(high, low, close, volume);
        let combos = expand_grid(sweep)?;
        if combos.is_empty() {
            return Err(CudaDemandIndexError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let rows = combos.len();
        let cols = high.len();
        let mut len_bs_values = Vec::with_capacity(rows);
        let mut len_bs_ma_values = Vec::with_capacity(rows);
        let mut len_di_ma_values = Vec::with_capacity(rows);
        let mut ma_codes = Vec::with_capacity(rows);
        let mut max_scratch_cap = 0usize;

        for combo in &combos {
            let len_bs = combo.len_bs.unwrap_or(19);
            let len_bs_ma = combo.len_bs_ma.unwrap_or(19);
            let len_di_ma = combo.len_di_ma.unwrap_or(19);
            if len_bs == 0 || len_bs_ma == 0 || len_di_ma == 0 {
                return Err(CudaDemandIndexError::InvalidInput("invalid lengths".into()));
            }
            if len_bs > cols || len_bs_ma > cols || len_di_ma > cols {
                return Err(CudaDemandIndexError::InvalidInput(format!(
                    "invalid lengths: len_bs={len_bs}, len_bs_ma={len_bs_ma}, len_di_ma={len_di_ma}, data_len={cols}"
                )));
            }
            let ma_code = parse_ma_type(combo.ma_type.as_deref().unwrap_or("ema"))?;
            let needed = needed_valid_bars(ma_code, len_bs, len_bs_ma, len_di_ma);
            if valid < needed {
                return Err(CudaDemandIndexError::InvalidInput(format!(
                    "not enough valid data: needed={needed}, valid={valid}"
                )));
            }
            let _ = first;
            let scratch_cap = len_bs * 2 + len_bs_ma * 4 + len_di_ma * 2;
            max_scratch_cap = max_scratch_cap.max(scratch_cap);
            len_bs_values.push(len_bs as i32);
            len_bs_ma_values.push(len_bs_ma as i32);
            len_di_ma_values.push(len_di_ma as i32);
            ma_codes.push(ma_code);
        }

        let output_elems = rows
            .checked_mul(cols)
            .ok_or_else(|| CudaDemandIndexError::InvalidInput("rows*cols overflow".into()))?;
        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| CudaDemandIndexError::InvalidInput("input bytes overflow".into()))?;
        let param_bytes = rows
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| CudaDemandIndexError::InvalidInput("param bytes overflow".into()))?;
        let scratch_bytes = rows
            .checked_mul(max_scratch_cap)
            .and_then(|value| value.checked_mul(std::mem::size_of::<f64>()))
            .ok_or_else(|| CudaDemandIndexError::InvalidInput("scratch bytes overflow".into()))?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| CudaDemandIndexError::InvalidInput("output bytes overflow".into()))?;
        let required = input_bytes
            .checked_add(param_bytes)
            .and_then(|value| value.checked_add(scratch_bytes))
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| CudaDemandIndexError::InvalidInput("required bytes overflow".into()))?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_close = DeviceBuffer::from_slice(close)?;
        let d_volume = DeviceBuffer::from_slice(volume)?;
        let d_len_bs_values = DeviceBuffer::from_slice(&len_bs_values)?;
        let d_len_bs_ma_values = DeviceBuffer::from_slice(&len_bs_ma_values)?;
        let d_len_di_ma_values = DeviceBuffer::from_slice(&len_di_ma_values)?;
        let d_ma_codes = DeviceBuffer::from_slice(&ma_codes)?;
        let mut d_scratch = unsafe { DeviceBuffer::<f64>::uninitialized(rows * max_scratch_cap)? };
        let mut d_demand_index = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_signal = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("demand_index_batch_f64")
            .map_err(|_| CudaDemandIndexError::MissingKernelSymbol {
                name: "demand_index_batch_f64",
            })?;
        let grid_x = ((rows as u32) + DEMAND_INDEX_BLOCK_X - 1) / DEMAND_INDEX_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(DEMAND_INDEX_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_high.as_device_ptr(),
                d_low.as_device_ptr(),
                d_close.as_device_ptr(),
                d_volume.as_device_ptr(),
                cols as i32,
                d_len_bs_values.as_device_ptr(),
                d_len_bs_ma_values.as_device_ptr(),
                d_len_di_ma_values.as_device_ptr(),
                d_ma_codes.as_device_ptr(),
                rows as i32,
                max_scratch_cap as i32,
                d_scratch.as_device_ptr(),
                d_demand_index.as_device_ptr(),
                d_signal.as_device_ptr()
            ))?;
        }
        self.stream.synchronize()?;

        Ok(CudaDemandIndexBatchResult {
            outputs: DemandIndexDeviceArrayF64Pair {
                demand_index: DemandIndexDeviceArrayF64 {
                    buf: d_demand_index,
                    rows,
                    cols,
                },
                signal: DemandIndexDeviceArrayF64 {
                    buf: d_signal,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
