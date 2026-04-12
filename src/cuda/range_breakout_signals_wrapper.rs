#![cfg(feature = "cuda")]

use crate::indicators::range_breakout_signals::{
    expand_grid_range_breakout_signals, RangeBreakoutSignalsBatchRange, RangeBreakoutSignalsParams,
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

const RANGE_BREAKOUT_SIGNALS_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const DEFAULT_RANGE_LENGTH: usize = 20;
const DEFAULT_CONFIRMATION_LENGTH: usize = 5;
const ATR_LENGTH: usize = 14;

#[derive(Debug, Error)]
pub enum CudaRangeBreakoutSignalsError {
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

pub struct RangeBreakoutSignalsDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl RangeBreakoutSignalsDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct RangeBreakoutSignalsDeviceArrayF64Sextet {
    pub range_top: RangeBreakoutSignalsDeviceArrayF64,
    pub range_bottom: RangeBreakoutSignalsDeviceArrayF64,
    pub bullish: RangeBreakoutSignalsDeviceArrayF64,
    pub extra_bullish: RangeBreakoutSignalsDeviceArrayF64,
    pub bearish: RangeBreakoutSignalsDeviceArrayF64,
    pub extra_bearish: RangeBreakoutSignalsDeviceArrayF64,
}

impl RangeBreakoutSignalsDeviceArrayF64Sextet {
    #[inline]
    pub fn rows(&self) -> usize {
        self.range_top.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.range_top.cols
    }
}

pub struct CudaRangeBreakoutSignalsBatchResult {
    pub outputs: RangeBreakoutSignalsDeviceArrayF64Sextet,
    pub combos: Vec<RangeBreakoutSignalsParams>,
}

pub struct CudaRangeBreakoutSignals {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaRangeBreakoutSignals {
    pub fn new(device_id: usize) -> Result<Self, CudaRangeBreakoutSignalsError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("range_breakout_signals_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaRangeBreakoutSignalsError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn max_valid_run(
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> usize {
        let mut best = 0usize;
        let mut run = 0usize;
        for i in 0..close.len() {
            if open[i].is_finite()
                && high[i].is_finite()
                && low[i].is_finite()
                && close[i].is_finite()
                && volume[i].is_finite()
            {
                run += 1;
                best = best.max(run);
            } else {
                run = 0;
            }
        }
        best
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaRangeBreakoutSignalsError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaRangeBreakoutSignalsError::OutOfMemory {
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
    ) -> Result<(), CudaRangeBreakoutSignalsError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaRangeBreakoutSignalsError::LaunchConfigTooLarge {
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
        sweep: &RangeBreakoutSignalsBatchRange,
    ) -> Result<CudaRangeBreakoutSignalsBatchResult, CudaRangeBreakoutSignalsError> {
        if open.is_empty()
            || high.is_empty()
            || low.is_empty()
            || close.is_empty()
            || volume.is_empty()
        {
            return Err(CudaRangeBreakoutSignalsError::InvalidInput(
                "empty input".into(),
            ));
        }
        if open.len() != high.len()
            || open.len() != low.len()
            || open.len() != close.len()
            || open.len() != volume.len()
        {
            return Err(CudaRangeBreakoutSignalsError::InvalidInput(format!(
                "input length mismatch: open={}, high={}, low={}, close={}, volume={}",
                open.len(),
                high.len(),
                low.len(),
                close.len(),
                volume.len()
            )));
        }

        let max_run = Self::max_valid_run(open, high, low, close, volume);
        if max_run == 0 {
            return Err(CudaRangeBreakoutSignalsError::InvalidInput(
                "all values are NaN".into(),
            ));
        }

        let combos = expand_grid_range_breakout_signals(sweep)
            .map_err(|err| CudaRangeBreakoutSignalsError::InvalidInput(err.to_string()))?;
        if combos.is_empty() {
            return Err(CudaRangeBreakoutSignalsError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let rows = combos.len();
        let cols = close.len();
        let mut range_lengths = Vec::with_capacity(rows);
        let mut confirmation_lengths = Vec::with_capacity(rows);
        let mut max_range_length = 0usize;
        let mut max_confirmation_window = 0usize;

        for combo in &combos {
            let range_length = combo.range_length.unwrap_or(DEFAULT_RANGE_LENGTH);
            let confirmation_length = combo
                .confirmation_length
                .unwrap_or(DEFAULT_CONFIRMATION_LENGTH);

            if range_length == 0 || range_length > cols {
                return Err(CudaRangeBreakoutSignalsError::InvalidInput(format!(
                    "invalid range_length: range_length={range_length}, data_len={cols}"
                )));
            }
            if confirmation_length == 0 {
                return Err(CudaRangeBreakoutSignalsError::InvalidInput(format!(
                    "invalid confirmation_length: {confirmation_length}"
                )));
            }

            let needed = (range_length + 1)
                .max(ATR_LENGTH)
                .max(confirmation_length + 1);
            if max_run < needed {
                return Err(CudaRangeBreakoutSignalsError::InvalidInput(format!(
                    "not enough valid data: needed={needed}, valid={max_run}"
                )));
            }

            max_range_length = max_range_length.max(range_length);
            max_confirmation_window = max_confirmation_window.max(confirmation_length + 1);

            range_lengths.push(i32::try_from(range_length).map_err(|_| {
                CudaRangeBreakoutSignalsError::InvalidInput(format!(
                    "range_length out of range: {range_length}"
                ))
            })?);
            confirmation_lengths.push(i32::try_from(confirmation_length).map_err(|_| {
                CudaRangeBreakoutSignalsError::InvalidInput(format!(
                    "confirmation_length out of range: {confirmation_length}"
                ))
            })?);
        }

        let rows_i32 = i32::try_from(rows)
            .map_err(|_| CudaRangeBreakoutSignalsError::InvalidInput("rows out of range".into()))?;
        let cols_i32 = i32::try_from(cols)
            .map_err(|_| CudaRangeBreakoutSignalsError::InvalidInput("cols out of range".into()))?;
        let max_range_length_i32 = i32::try_from(max_range_length).map_err(|_| {
            CudaRangeBreakoutSignalsError::InvalidInput("max_range_length out of range".into())
        })?;
        let max_confirmation_window_i32 = i32::try_from(max_confirmation_window).map_err(|_| {
            CudaRangeBreakoutSignalsError::InvalidInput(
                "max_confirmation_window out of range".into(),
            )
        })?;

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(5))
            .ok_or_else(|| {
                CudaRangeBreakoutSignalsError::InvalidInput("input bytes overflow".into())
            })?;
        let param_bytes = rows
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                CudaRangeBreakoutSignalsError::InvalidInput("param bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaRangeBreakoutSignalsError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(6))
            .ok_or_else(|| {
                CudaRangeBreakoutSignalsError::InvalidInput("output bytes overflow".into())
            })?;
        let scratch_bytes = rows
            .checked_mul(max_range_length)
            .and_then(|value| value.checked_mul(std::mem::size_of::<f64>() * 2))
            .and_then(|value| {
                rows.checked_mul(max_confirmation_window)
                    .and_then(|other| other.checked_mul(std::mem::size_of::<f64>() * 2 + 1))
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| {
                CudaRangeBreakoutSignalsError::InvalidInput("scratch bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(param_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .and_then(|value| value.checked_add(scratch_bytes))
            .ok_or_else(|| {
                CudaRangeBreakoutSignalsError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_open = DeviceBuffer::from_slice(open)?;
        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_close = DeviceBuffer::from_slice(close)?;
        let d_volume = DeviceBuffer::from_slice(volume)?;
        let d_range_lengths = DeviceBuffer::from_slice(&range_lengths)?;
        let d_confirmation_lengths = DeviceBuffer::from_slice(&confirmation_lengths)?;
        let d_out_range_top = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_range_bottom = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_bullish = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_extra_bullish = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_bearish = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_extra_bearish = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_dist_ring = unsafe { DeviceBuffer::<f64>::uninitialized(rows * max_range_length)? };
        let d_dist_sorted = unsafe { DeviceBuffer::<f64>::uninitialized(rows * max_range_length)? };
        let d_up_volume =
            unsafe { DeviceBuffer::<f64>::uninitialized(rows * max_confirmation_window)? };
        let d_down_volume =
            unsafe { DeviceBuffer::<f64>::uninitialized(rows * max_confirmation_window)? };
        let d_under = unsafe { DeviceBuffer::<u8>::uninitialized(rows * max_confirmation_window)? };

        let func = self
            .module
            .get_function("range_breakout_signals_batch_f64")
            .map_err(|_| CudaRangeBreakoutSignalsError::MissingKernelSymbol {
                name: "range_breakout_signals_batch_f64",
            })?;
        let grid_x =
            ((rows as u32) + RANGE_BREAKOUT_SIGNALS_BLOCK_X - 1) / RANGE_BREAKOUT_SIGNALS_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(RANGE_BREAKOUT_SIGNALS_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_open.as_device_ptr(),
                d_high.as_device_ptr(),
                d_low.as_device_ptr(),
                d_close.as_device_ptr(),
                d_volume.as_device_ptr(),
                cols_i32,
                d_range_lengths.as_device_ptr(),
                d_confirmation_lengths.as_device_ptr(),
                rows_i32,
                max_range_length_i32,
                max_confirmation_window_i32,
                d_out_range_top.as_device_ptr(),
                d_out_range_bottom.as_device_ptr(),
                d_out_bullish.as_device_ptr(),
                d_out_extra_bullish.as_device_ptr(),
                d_out_bearish.as_device_ptr(),
                d_out_extra_bearish.as_device_ptr(),
                d_dist_ring.as_device_ptr(),
                d_dist_sorted.as_device_ptr(),
                d_up_volume.as_device_ptr(),
                d_down_volume.as_device_ptr(),
                d_under.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaRangeBreakoutSignalsBatchResult {
            outputs: RangeBreakoutSignalsDeviceArrayF64Sextet {
                range_top: RangeBreakoutSignalsDeviceArrayF64 {
                    buf: d_out_range_top,
                    rows,
                    cols,
                },
                range_bottom: RangeBreakoutSignalsDeviceArrayF64 {
                    buf: d_out_range_bottom,
                    rows,
                    cols,
                },
                bullish: RangeBreakoutSignalsDeviceArrayF64 {
                    buf: d_out_bullish,
                    rows,
                    cols,
                },
                extra_bullish: RangeBreakoutSignalsDeviceArrayF64 {
                    buf: d_out_extra_bullish,
                    rows,
                    cols,
                },
                bearish: RangeBreakoutSignalsDeviceArrayF64 {
                    buf: d_out_bearish,
                    rows,
                    cols,
                },
                extra_bearish: RangeBreakoutSignalsDeviceArrayF64 {
                    buf: d_out_extra_bearish,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
