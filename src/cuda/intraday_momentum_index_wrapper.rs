#![cfg(feature = "cuda")]

use crate::indicators::intraday_momentum_index::{
    IntradayMomentumIndexBatchRange, IntradayMomentumIndexParams,
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

const INTRADAY_MOMENTUM_INDEX_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const DEFAULT_LENGTH: usize = 14;
const DEFAULT_LENGTH_MA: usize = 6;
const DEFAULT_MULT: f64 = 2.0;
const DEFAULT_LENGTH_BB: usize = 20;
const DEFAULT_APPLY_SMOOTHING: bool = false;
const DEFAULT_LOW_BAND: usize = 10;

#[derive(Debug, Error)]
pub enum CudaIntradayMomentumIndexError {
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

pub struct IntradayMomentumIndexDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl IntradayMomentumIndexDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct IntradayMomentumIndexDeviceArrayF64Quad {
    pub imi: IntradayMomentumIndexDeviceArrayF64,
    pub upper_hit: IntradayMomentumIndexDeviceArrayF64,
    pub lower_hit: IntradayMomentumIndexDeviceArrayF64,
    pub signal: IntradayMomentumIndexDeviceArrayF64,
}

impl IntradayMomentumIndexDeviceArrayF64Quad {
    #[inline]
    pub fn rows(&self) -> usize {
        self.imi.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.imi.cols
    }
}

pub struct CudaIntradayMomentumIndexBatchResult {
    pub outputs: IntradayMomentumIndexDeviceArrayF64Quad,
    pub combos: Vec<IntradayMomentumIndexParams>,
}

pub struct CudaIntradayMomentumIndex {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

fn expand_usize_range(
    start: usize,
    end: usize,
    step: usize,
) -> Result<Vec<usize>, CudaIntradayMomentumIndexError> {
    if start > end {
        return Err(CudaIntradayMomentumIndexError::InvalidInput(format!(
            "invalid range: start={start}, end={end}, step={step}"
        )));
    }
    if step == 0 {
        return Ok(vec![start]);
    }

    let mut out = Vec::new();
    let mut cur = start;
    while cur <= end {
        out.push(cur);
        match cur.checked_add(step) {
            Some(next) if next > cur => cur = next,
            _ => break,
        }
    }
    Ok(out)
}

fn expand_f64_range(
    start: f64,
    end: f64,
    step: f64,
) -> Result<Vec<f64>, CudaIntradayMomentumIndexError> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() || start > end || step < 0.0 {
        return Err(CudaIntradayMomentumIndexError::InvalidInput(format!(
            "invalid range: start={start}, end={end}, step={step}"
        )));
    }
    if step == 0.0 {
        return Ok(vec![start]);
    }

    let mut out = Vec::new();
    let mut cur = start;
    while cur <= end + 1e-12 {
        out.push(cur);
        cur += step;
    }
    Ok(out)
}

fn expand_grid_intraday_momentum_index(
    sweep: &IntradayMomentumIndexBatchRange,
) -> Result<Vec<IntradayMomentumIndexParams>, CudaIntradayMomentumIndexError> {
    let lengths = expand_usize_range(sweep.length.0, sweep.length.1, sweep.length.2)?;
    let length_mas = expand_usize_range(sweep.length_ma.0, sweep.length_ma.1, sweep.length_ma.2)?;
    let mults = expand_f64_range(sweep.mult.0, sweep.mult.1, sweep.mult.2)?;
    let length_bbs = expand_usize_range(sweep.length_bb.0, sweep.length_bb.1, sweep.length_bb.2)?;
    let low_bands = expand_usize_range(sweep.low_band.0, sweep.low_band.1, sweep.low_band.2)?;
    let apply_smoothing = sweep.apply_smoothing.unwrap_or(DEFAULT_APPLY_SMOOTHING);

    let mut combos = Vec::with_capacity(
        lengths
            .len()
            .saturating_mul(length_mas.len())
            .saturating_mul(mults.len())
            .saturating_mul(length_bbs.len())
            .saturating_mul(low_bands.len()),
    );
    for &length in &lengths {
        for &length_ma in &length_mas {
            for &mult in &mults {
                for &length_bb in &length_bbs {
                    for &low_band in &low_bands {
                        combos.push(IntradayMomentumIndexParams {
                            length: Some(length),
                            length_ma: Some(length_ma),
                            mult: Some(mult),
                            length_bb: Some(length_bb),
                            apply_smoothing: Some(apply_smoothing),
                            low_band: Some(low_band),
                        });
                    }
                }
            }
        }
    }
    Ok(combos)
}

impl CudaIntradayMomentumIndex {
    pub fn new(device_id: usize) -> Result<Self, CudaIntradayMomentumIndexError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("intraday_momentum_index_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaIntradayMomentumIndexError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn first_valid_open_close(open: &[f64], close: &[f64]) -> Option<usize> {
        open.iter()
            .zip(close.iter())
            .position(|(&o, &c)| o.is_finite() && c.is_finite())
    }

    fn count_valid_open_close(open: &[f64], close: &[f64]) -> usize {
        open.iter()
            .zip(close.iter())
            .filter(|&(o, c)| o.is_finite() && c.is_finite())
            .count()
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaIntradayMomentumIndexError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaIntradayMomentumIndexError::OutOfMemory {
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
    ) -> Result<(), CudaIntradayMomentumIndexError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaIntradayMomentumIndexError::LaunchConfigTooLarge {
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
        close: &[f64],
        sweep: &IntradayMomentumIndexBatchRange,
    ) -> Result<CudaIntradayMomentumIndexBatchResult, CudaIntradayMomentumIndexError> {
        if open.is_empty() || close.is_empty() {
            return Err(CudaIntradayMomentumIndexError::InvalidInput(
                "empty input".into(),
            ));
        }
        if open.len() != close.len() {
            return Err(CudaIntradayMomentumIndexError::InvalidInput(format!(
                "input length mismatch: open={}, close={}",
                open.len(),
                close.len()
            )));
        }
        Self::first_valid_open_close(open, close).ok_or_else(|| {
            CudaIntradayMomentumIndexError::InvalidInput("all values are NaN".into())
        })?;
        let valid = Self::count_valid_open_close(open, close);

        let combos = expand_grid_intraday_momentum_index(sweep)?;
        if combos.is_empty() {
            return Err(CudaIntradayMomentumIndexError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let rows = combos.len();
        let cols = open.len();
        let mut lengths = Vec::with_capacity(rows);
        let mut length_mas = Vec::with_capacity(rows);
        let mut mults = Vec::with_capacity(rows);
        let mut length_bbs = Vec::with_capacity(rows);
        let mut apply_smoothings = Vec::with_capacity(rows);
        let mut low_bands = Vec::with_capacity(rows);

        for combo in &combos {
            let length = combo.length.unwrap_or(DEFAULT_LENGTH);
            let length_ma = combo.length_ma.unwrap_or(DEFAULT_LENGTH_MA);
            let mult = combo.mult.unwrap_or(DEFAULT_MULT);
            let length_bb = combo.length_bb.unwrap_or(DEFAULT_LENGTH_BB);
            let apply_smoothing = combo.apply_smoothing.unwrap_or(DEFAULT_APPLY_SMOOTHING);
            let low_band = combo.low_band.unwrap_or(DEFAULT_LOW_BAND);

            if length == 0 || length > cols {
                return Err(CudaIntradayMomentumIndexError::InvalidInput(format!(
                    "invalid length: length={length}, data_len={cols}"
                )));
            }
            if length_ma == 0 || length_ma > cols {
                return Err(CudaIntradayMomentumIndexError::InvalidInput(format!(
                    "invalid length_ma: length_ma={length_ma}, data_len={cols}"
                )));
            }
            if length_bb == 0 || length_bb > cols {
                return Err(CudaIntradayMomentumIndexError::InvalidInput(format!(
                    "invalid length_bb: length_bb={length_bb}, data_len={cols}"
                )));
            }
            if !mult.is_finite() || mult < 0.0 {
                return Err(CudaIntradayMomentumIndexError::InvalidInput(format!(
                    "invalid mult: {mult}"
                )));
            }
            if apply_smoothing && low_band == 0 {
                return Err(CudaIntradayMomentumIndexError::InvalidInput(format!(
                    "invalid low_band: {low_band}"
                )));
            }
            if valid < length {
                return Err(CudaIntradayMomentumIndexError::InvalidInput(format!(
                    "not enough valid data: needed={length}, valid={valid}"
                )));
            }

            lengths.push(length as i32);
            length_mas.push(length_ma as i32);
            mults.push(mult);
            length_bbs.push(length_bb as i32);
            apply_smoothings.push(if apply_smoothing { 1 } else { 0 });
            low_bands.push(low_band as i32);
        }

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                CudaIntradayMomentumIndexError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = rows
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| value.checked_mul(5))
            .and_then(|value| {
                rows.checked_mul(std::mem::size_of::<f64>())
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| {
                CudaIntradayMomentumIndexError::InvalidInput("params bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaIntradayMomentumIndexError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| {
                CudaIntradayMomentumIndexError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaIntradayMomentumIndexError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_open = DeviceBuffer::from_slice(open)?;
        let d_close = DeviceBuffer::from_slice(close)?;
        let d_lengths = DeviceBuffer::from_slice(&lengths)?;
        let d_length_mas = DeviceBuffer::from_slice(&length_mas)?;
        let d_mults = DeviceBuffer::from_slice(&mults)?;
        let d_length_bbs = DeviceBuffer::from_slice(&length_bbs)?;
        let d_apply_smoothings = DeviceBuffer::from_slice(&apply_smoothings)?;
        let d_low_bands = DeviceBuffer::from_slice(&low_bands)?;
        let d_out_imi = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_upper = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_lower = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_signal = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("intraday_momentum_index_batch_f64")
            .map_err(|_| CudaIntradayMomentumIndexError::MissingKernelSymbol {
                name: "intraday_momentum_index_batch_f64",
            })?;
        let grid_x =
            ((rows as u32) + INTRADAY_MOMENTUM_INDEX_BLOCK_X - 1) / INTRADAY_MOMENTUM_INDEX_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(INTRADAY_MOMENTUM_INDEX_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_open.as_device_ptr(),
                d_close.as_device_ptr(),
                cols as i32,
                d_lengths.as_device_ptr(),
                d_length_mas.as_device_ptr(),
                d_mults.as_device_ptr(),
                d_length_bbs.as_device_ptr(),
                d_apply_smoothings.as_device_ptr(),
                d_low_bands.as_device_ptr(),
                rows as i32,
                d_out_imi.as_device_ptr(),
                d_out_upper.as_device_ptr(),
                d_out_lower.as_device_ptr(),
                d_out_signal.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaIntradayMomentumIndexBatchResult {
            outputs: IntradayMomentumIndexDeviceArrayF64Quad {
                imi: IntradayMomentumIndexDeviceArrayF64 {
                    buf: d_out_imi,
                    rows,
                    cols,
                },
                upper_hit: IntradayMomentumIndexDeviceArrayF64 {
                    buf: d_out_upper,
                    rows,
                    cols,
                },
                lower_hit: IntradayMomentumIndexDeviceArrayF64 {
                    buf: d_out_lower,
                    rows,
                    cols,
                },
                signal: IntradayMomentumIndexDeviceArrayF64 {
                    buf: d_out_signal,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
