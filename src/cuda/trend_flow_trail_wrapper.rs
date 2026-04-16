#![cfg(feature = "cuda")]

use crate::indicators::trend_flow_trail::{
    expand_grid_trend_flow_trail, TrendFlowTrailBatchRange, TrendFlowTrailParams,
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

const TREND_FLOW_TRAIL_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const DEFAULT_ALPHA_LENGTH: usize = 33;
const DEFAULT_ALPHA_MULTIPLIER: f64 = 3.3;
const DEFAULT_MFI_LENGTH: usize = 14;
const MFI_HMA_LENGTH: usize = 7;
const MFI_HMA_HALF: usize = 3;
const MFI_HMA_SQRT: usize = 2;

#[derive(Debug, Error)]
pub enum CudaTrendFlowTrailError {
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

pub struct TrendFlowTrailDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl TrendFlowTrailDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct TrendFlowTrailDeviceOutputs {
    pub alpha_trail: TrendFlowTrailDeviceArrayF64,
    pub alpha_trail_bullish: TrendFlowTrailDeviceArrayF64,
    pub alpha_trail_bearish: TrendFlowTrailDeviceArrayF64,
    pub alpha_dir: TrendFlowTrailDeviceArrayF64,
    pub mfi: TrendFlowTrailDeviceArrayF64,
    pub tp_upper: TrendFlowTrailDeviceArrayF64,
    pub tp_lower: TrendFlowTrailDeviceArrayF64,
    pub alpha_trail_bullish_switch: TrendFlowTrailDeviceArrayF64,
    pub alpha_trail_bearish_switch: TrendFlowTrailDeviceArrayF64,
    pub mfi_overbought: TrendFlowTrailDeviceArrayF64,
    pub mfi_oversold: TrendFlowTrailDeviceArrayF64,
    pub mfi_cross_up_mid: TrendFlowTrailDeviceArrayF64,
    pub mfi_cross_down_mid: TrendFlowTrailDeviceArrayF64,
    pub price_cross_alpha_trail_up: TrendFlowTrailDeviceArrayF64,
    pub price_cross_alpha_trail_down: TrendFlowTrailDeviceArrayF64,
    pub mfi_above_90: TrendFlowTrailDeviceArrayF64,
    pub mfi_below_10: TrendFlowTrailDeviceArrayF64,
}

impl TrendFlowTrailDeviceOutputs {
    #[inline]
    pub fn rows(&self) -> usize {
        self.alpha_trail.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.alpha_trail.cols
    }
}

pub struct CudaTrendFlowTrailBatchResult {
    pub outputs: TrendFlowTrailDeviceOutputs,
    pub combos: Vec<TrendFlowTrailParams>,
}

pub struct CudaTrendFlowTrail {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

#[inline]
fn alpha_required_bars(alpha_length: usize) -> usize {
    if alpha_length <= 1 {
        1
    } else {
        alpha_length + (alpha_length as f64).sqrt().floor() as usize - 1
    }
}

#[inline]
fn required_valid_bars(alpha_length: usize, mfi_length: usize) -> usize {
    alpha_required_bars(alpha_length).max(mfi_length + MFI_HMA_LENGTH + MFI_HMA_SQRT - 1)
}

#[inline]
fn longest_valid_run(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
) -> usize {
    let mut best = 0usize;
    let mut current = 0usize;
    for ((((&o, &h), &l), &c), &v) in open
        .iter()
        .zip(high.iter())
        .zip(low.iter())
        .zip(close.iter())
        .zip(volume.iter())
    {
        if o.is_finite() && h.is_finite() && l.is_finite() && c.is_finite() && v.is_finite() {
            current += 1;
            best = best.max(current);
        } else {
            current = 0;
        }
    }
    best
}

impl CudaTrendFlowTrail {
    pub fn new(device_id: usize) -> Result<Self, CudaTrendFlowTrailError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("trend_flow_trail_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaTrendFlowTrailError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaTrendFlowTrailError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaTrendFlowTrailError::OutOfMemory {
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
    ) -> Result<(), CudaTrendFlowTrailError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaTrendFlowTrailError::LaunchConfigTooLarge {
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
        sweep: &TrendFlowTrailBatchRange,
    ) -> Result<CudaTrendFlowTrailBatchResult, CudaTrendFlowTrailError> {
        if open.is_empty()
            || high.is_empty()
            || low.is_empty()
            || close.is_empty()
            || volume.is_empty()
        {
            return Err(CudaTrendFlowTrailError::InvalidInput("empty input".into()));
        }
        if open.len() != high.len()
            || open.len() != low.len()
            || open.len() != close.len()
            || open.len() != volume.len()
        {
            return Err(CudaTrendFlowTrailError::InvalidInput(format!(
                "input length mismatch: open={}, high={}, low={}, close={}, volume={}",
                open.len(),
                high.len(),
                low.len(),
                close.len(),
                volume.len()
            )));
        }

        let max_run = longest_valid_run(open, high, low, close, volume);
        if max_run == 0 {
            return Err(CudaTrendFlowTrailError::InvalidInput(
                "all values are NaN".into(),
            ));
        }

        let combos = expand_grid_trend_flow_trail(sweep)
            .map_err(|err| CudaTrendFlowTrailError::InvalidInput(err.to_string()))?;
        if combos.is_empty() {
            return Err(CudaTrendFlowTrailError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let rows = combos.len();
        let cols = close.len();
        let mut alpha_lengths = Vec::with_capacity(rows);
        let mut alpha_multipliers = Vec::with_capacity(rows);
        let mut mfi_lengths = Vec::with_capacity(rows);
        let mut max_alpha_length = 1usize;
        let mut max_alpha_half = 1usize;
        let mut max_alpha_sqrt = 1usize;
        let mut max_mfi_length = 1usize;

        for combo in &combos {
            let alpha_length = combo.alpha_length.unwrap_or(DEFAULT_ALPHA_LENGTH);
            let alpha_multiplier = combo.alpha_multiplier.unwrap_or(DEFAULT_ALPHA_MULTIPLIER);
            let mfi_length = combo.mfi_length.unwrap_or(DEFAULT_MFI_LENGTH);

            if alpha_length == 0 {
                return Err(CudaTrendFlowTrailError::InvalidInput(format!(
                    "invalid alpha_length: {alpha_length}"
                )));
            }
            if !alpha_multiplier.is_finite() || alpha_multiplier < 0.1 {
                return Err(CudaTrendFlowTrailError::InvalidInput(format!(
                    "invalid alpha_multiplier: {alpha_multiplier}"
                )));
            }
            if mfi_length == 0 {
                return Err(CudaTrendFlowTrailError::InvalidInput(format!(
                    "invalid mfi_length: {mfi_length}"
                )));
            }
            let needed = required_valid_bars(alpha_length, mfi_length);
            if max_run < needed {
                return Err(CudaTrendFlowTrailError::InvalidInput(format!(
                    "not enough valid data: needed={needed}, valid={max_run}"
                )));
            }

            max_alpha_length = max_alpha_length.max(alpha_length);
            max_alpha_half = max_alpha_half.max((alpha_length / 2).max(1));
            max_alpha_sqrt = max_alpha_sqrt
                .max((alpha_length as f64).sqrt().floor() as usize)
                .max(1);
            max_mfi_length = max_mfi_length.max(mfi_length);

            alpha_lengths.push(i32::try_from(alpha_length).map_err(|_| {
                CudaTrendFlowTrailError::InvalidInput("alpha_length out of range".into())
            })?);
            alpha_multipliers.push(alpha_multiplier);
            mfi_lengths.push(i32::try_from(mfi_length).map_err(|_| {
                CudaTrendFlowTrailError::InvalidInput("mfi_length out of range".into())
            })?);
        }

        let output_elems = rows
            .checked_mul(cols)
            .ok_or_else(|| CudaTrendFlowTrailError::InvalidInput("rows*cols overflow".into()))?;
        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(5))
            .ok_or_else(|| CudaTrendFlowTrailError::InvalidInput("input bytes overflow".into()))?;
        let param_bytes = rows
            .checked_mul(std::mem::size_of::<i32>() * 2 + std::mem::size_of::<f64>())
            .ok_or_else(|| CudaTrendFlowTrailError::InvalidInput("param bytes overflow".into()))?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(17))
            .ok_or_else(|| CudaTrendFlowTrailError::InvalidInput("output bytes overflow".into()))?;
        let scratch_elems = rows
            .checked_mul(max_alpha_length)
            .and_then(|value| value.checked_add(rows * max_alpha_half))
            .and_then(|value| value.checked_add(rows * max_alpha_sqrt))
            .and_then(|value| value.checked_add(rows * max_mfi_length * 2))
            .and_then(|value| value.checked_add(rows * MFI_HMA_LENGTH))
            .and_then(|value| value.checked_add(rows * MFI_HMA_HALF))
            .and_then(|value| value.checked_add(rows * MFI_HMA_SQRT))
            .ok_or_else(|| {
                CudaTrendFlowTrailError::InvalidInput("scratch elements overflow".into())
            })?;
        let scratch_bytes = scratch_elems
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaTrendFlowTrailError::InvalidInput("scratch bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(param_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .and_then(|value| value.checked_add(scratch_bytes))
            .ok_or_else(|| {
                CudaTrendFlowTrailError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_open = DeviceBuffer::from_slice(open)?;
        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_close = DeviceBuffer::from_slice(close)?;
        let d_volume = DeviceBuffer::from_slice(volume)?;
        let d_alpha_lengths = DeviceBuffer::from_slice(&alpha_lengths)?;
        let d_alpha_multipliers = DeviceBuffer::from_slice(&alpha_multipliers)?;
        let d_mfi_lengths = DeviceBuffer::from_slice(&mfi_lengths)?;
        let d_alpha_full = unsafe { DeviceBuffer::<f64>::uninitialized(rows * max_alpha_length)? };
        let d_alpha_half = unsafe { DeviceBuffer::<f64>::uninitialized(rows * max_alpha_half)? };
        let d_alpha_sqrt = unsafe { DeviceBuffer::<f64>::uninitialized(rows * max_alpha_sqrt)? };
        let d_mfi_pos = unsafe { DeviceBuffer::<f64>::uninitialized(rows * max_mfi_length)? };
        let d_mfi_neg = unsafe { DeviceBuffer::<f64>::uninitialized(rows * max_mfi_length)? };
        let d_mfi_full = unsafe { DeviceBuffer::<f64>::uninitialized(rows * MFI_HMA_LENGTH)? };
        let d_mfi_half = unsafe { DeviceBuffer::<f64>::uninitialized(rows * MFI_HMA_HALF)? };
        let d_mfi_sqrt = unsafe { DeviceBuffer::<f64>::uninitialized(rows * MFI_HMA_SQRT)? };

        let d_alpha_trail = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_alpha_trail_bullish = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_alpha_trail_bearish = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_alpha_dir = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_mfi = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_tp_upper = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_tp_lower = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_alpha_trail_bullish_switch =
            unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_alpha_trail_bearish_switch =
            unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_mfi_overbought = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_mfi_oversold = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_mfi_cross_up_mid = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_mfi_cross_down_mid = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_price_cross_alpha_trail_up =
            unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_price_cross_alpha_trail_down =
            unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_mfi_above_90 = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_mfi_below_10 = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("trend_flow_trail_batch_f64")
            .map_err(|_| CudaTrendFlowTrailError::MissingKernelSymbol {
                name: "trend_flow_trail_batch_f64",
            })?;
        let grid_x = ((rows as u32) + TREND_FLOW_TRAIL_BLOCK_X - 1) / TREND_FLOW_TRAIL_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(TREND_FLOW_TRAIL_BLOCK_X);
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
                d_alpha_lengths.as_device_ptr(),
                d_alpha_multipliers.as_device_ptr(),
                d_mfi_lengths.as_device_ptr(),
                rows as i32,
                max_alpha_length as i32,
                max_alpha_half as i32,
                max_alpha_sqrt as i32,
                max_mfi_length as i32,
                d_alpha_full.as_device_ptr(),
                d_alpha_half.as_device_ptr(),
                d_alpha_sqrt.as_device_ptr(),
                d_mfi_pos.as_device_ptr(),
                d_mfi_neg.as_device_ptr(),
                d_mfi_full.as_device_ptr(),
                d_mfi_half.as_device_ptr(),
                d_mfi_sqrt.as_device_ptr(),
                d_alpha_trail.as_device_ptr(),
                d_alpha_trail_bullish.as_device_ptr(),
                d_alpha_trail_bearish.as_device_ptr(),
                d_alpha_dir.as_device_ptr(),
                d_mfi.as_device_ptr(),
                d_tp_upper.as_device_ptr(),
                d_tp_lower.as_device_ptr(),
                d_alpha_trail_bullish_switch.as_device_ptr(),
                d_alpha_trail_bearish_switch.as_device_ptr(),
                d_mfi_overbought.as_device_ptr(),
                d_mfi_oversold.as_device_ptr(),
                d_mfi_cross_up_mid.as_device_ptr(),
                d_mfi_cross_down_mid.as_device_ptr(),
                d_price_cross_alpha_trail_up.as_device_ptr(),
                d_price_cross_alpha_trail_down.as_device_ptr(),
                d_mfi_above_90.as_device_ptr(),
                d_mfi_below_10.as_device_ptr()
            ))?;
        }
        self.stream.synchronize()?;

        Ok(CudaTrendFlowTrailBatchResult {
            outputs: TrendFlowTrailDeviceOutputs {
                alpha_trail: TrendFlowTrailDeviceArrayF64 {
                    buf: d_alpha_trail,
                    rows,
                    cols,
                },
                alpha_trail_bullish: TrendFlowTrailDeviceArrayF64 {
                    buf: d_alpha_trail_bullish,
                    rows,
                    cols,
                },
                alpha_trail_bearish: TrendFlowTrailDeviceArrayF64 {
                    buf: d_alpha_trail_bearish,
                    rows,
                    cols,
                },
                alpha_dir: TrendFlowTrailDeviceArrayF64 {
                    buf: d_alpha_dir,
                    rows,
                    cols,
                },
                mfi: TrendFlowTrailDeviceArrayF64 {
                    buf: d_mfi,
                    rows,
                    cols,
                },
                tp_upper: TrendFlowTrailDeviceArrayF64 {
                    buf: d_tp_upper,
                    rows,
                    cols,
                },
                tp_lower: TrendFlowTrailDeviceArrayF64 {
                    buf: d_tp_lower,
                    rows,
                    cols,
                },
                alpha_trail_bullish_switch: TrendFlowTrailDeviceArrayF64 {
                    buf: d_alpha_trail_bullish_switch,
                    rows,
                    cols,
                },
                alpha_trail_bearish_switch: TrendFlowTrailDeviceArrayF64 {
                    buf: d_alpha_trail_bearish_switch,
                    rows,
                    cols,
                },
                mfi_overbought: TrendFlowTrailDeviceArrayF64 {
                    buf: d_mfi_overbought,
                    rows,
                    cols,
                },
                mfi_oversold: TrendFlowTrailDeviceArrayF64 {
                    buf: d_mfi_oversold,
                    rows,
                    cols,
                },
                mfi_cross_up_mid: TrendFlowTrailDeviceArrayF64 {
                    buf: d_mfi_cross_up_mid,
                    rows,
                    cols,
                },
                mfi_cross_down_mid: TrendFlowTrailDeviceArrayF64 {
                    buf: d_mfi_cross_down_mid,
                    rows,
                    cols,
                },
                price_cross_alpha_trail_up: TrendFlowTrailDeviceArrayF64 {
                    buf: d_price_cross_alpha_trail_up,
                    rows,
                    cols,
                },
                price_cross_alpha_trail_down: TrendFlowTrailDeviceArrayF64 {
                    buf: d_price_cross_alpha_trail_down,
                    rows,
                    cols,
                },
                mfi_above_90: TrendFlowTrailDeviceArrayF64 {
                    buf: d_mfi_above_90,
                    rows,
                    cols,
                },
                mfi_below_10: TrendFlowTrailDeviceArrayF64 {
                    buf: d_mfi_below_10,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
