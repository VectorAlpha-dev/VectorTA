#![cfg(feature = "cuda")]

use crate::indicators::price_moving_average_ratio_percentile::{
    expand_grid, PriceMovingAverageRatioPercentileBatchRange,
    PriceMovingAverageRatioPercentileLineMode, PriceMovingAverageRatioPercentileMaType,
    PriceMovingAverageRatioPercentileParams,
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

const PRICE_MOVING_AVERAGE_RATIO_PERCENTILE_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const DEFAULT_MA_LENGTH: usize = 20;
const DEFAULT_PMARP_LOOKBACK: usize = 350;
const DEFAULT_SIGNAL_MA_LENGTH: usize = 20;
const MA_SMA: i32 = 0;
const MA_EMA: i32 = 1;
const MA_HMA: i32 = 2;
const MA_RMA: i32 = 3;
const MA_VWMA: i32 = 4;
const LINE_PMAR: i32 = 0;
const LINE_PMARP: i32 = 1;

#[derive(Debug, Error)]
pub enum CudaPriceMovingAverageRatioPercentileError {
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

pub struct PriceMovingAverageRatioPercentileDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl PriceMovingAverageRatioPercentileDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct PriceMovingAverageRatioPercentileDeviceArrayF64Seven {
    pub pmar: PriceMovingAverageRatioPercentileDeviceArrayF64,
    pub pmarp: PriceMovingAverageRatioPercentileDeviceArrayF64,
    pub plotline: PriceMovingAverageRatioPercentileDeviceArrayF64,
    pub signal: PriceMovingAverageRatioPercentileDeviceArrayF64,
    pub pmar_high: PriceMovingAverageRatioPercentileDeviceArrayF64,
    pub pmar_low: PriceMovingAverageRatioPercentileDeviceArrayF64,
    pub scaled_pmar: PriceMovingAverageRatioPercentileDeviceArrayF64,
}

impl PriceMovingAverageRatioPercentileDeviceArrayF64Seven {
    #[inline]
    pub fn rows(&self) -> usize {
        self.pmar.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.pmar.cols
    }
}

pub struct CudaPriceMovingAverageRatioPercentileBatchResult {
    pub outputs: PriceMovingAverageRatioPercentileDeviceArrayF64Seven,
    pub combos: Vec<PriceMovingAverageRatioPercentileParams>,
}

pub struct CudaPriceMovingAverageRatioPercentile {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

#[inline]
fn parse_ma_type(value: PriceMovingAverageRatioPercentileMaType) -> i32 {
    match value {
        PriceMovingAverageRatioPercentileMaType::Sma => MA_SMA,
        PriceMovingAverageRatioPercentileMaType::Ema => MA_EMA,
        PriceMovingAverageRatioPercentileMaType::Hma => MA_HMA,
        PriceMovingAverageRatioPercentileMaType::Rma => MA_RMA,
        PriceMovingAverageRatioPercentileMaType::Vwma => MA_VWMA,
    }
}

#[inline]
fn parse_line_mode(value: PriceMovingAverageRatioPercentileLineMode) -> i32 {
    match value {
        PriceMovingAverageRatioPercentileLineMode::Pmar => LINE_PMAR,
        PriceMovingAverageRatioPercentileLineMode::Pmarp => LINE_PMARP,
    }
}

impl CudaPriceMovingAverageRatioPercentile {
    pub fn new(device_id: usize) -> Result<Self, CudaPriceMovingAverageRatioPercentileError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module =
            crate::load_cuda_embedded_module!("price_moving_average_ratio_percentile_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaPriceMovingAverageRatioPercentileError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn will_fit(
        required: usize,
        headroom: usize,
    ) -> Result<(), CudaPriceMovingAverageRatioPercentileError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaPriceMovingAverageRatioPercentileError::OutOfMemory {
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
    ) -> Result<(), CudaPriceMovingAverageRatioPercentileError> {
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
                CudaPriceMovingAverageRatioPercentileError::LaunchConfigTooLarge {
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
        price: &[f64],
        volume: &[f64],
        sweep: &PriceMovingAverageRatioPercentileBatchRange,
    ) -> Result<
        CudaPriceMovingAverageRatioPercentileBatchResult,
        CudaPriceMovingAverageRatioPercentileError,
    > {
        if price.is_empty() || volume.is_empty() {
            return Err(CudaPriceMovingAverageRatioPercentileError::InvalidInput(
                "empty input".into(),
            ));
        }
        if price.len() != volume.len() {
            return Err(CudaPriceMovingAverageRatioPercentileError::InvalidInput(
                "price and volume length mismatch".into(),
            ));
        }
        if !price
            .iter()
            .zip(volume.iter())
            .any(|(&p, &v)| p.is_finite() && v.is_finite())
        {
            return Err(CudaPriceMovingAverageRatioPercentileError::InvalidInput(
                "all values are NaN".into(),
            ));
        }

        let combos = expand_grid(sweep)
            .map_err(|e| CudaPriceMovingAverageRatioPercentileError::InvalidInput(e.to_string()))?;
        if combos.is_empty() {
            return Err(CudaPriceMovingAverageRatioPercentileError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let rows = combos.len();
        let cols = price.len();
        let mut ma_lengths = Vec::with_capacity(rows);
        let mut pmarp_lookbacks = Vec::with_capacity(rows);
        let mut signal_ma_lengths = Vec::with_capacity(rows);
        let mut ma_codes = Vec::with_capacity(rows);
        let mut signal_ma_codes = Vec::with_capacity(rows);
        let mut line_modes = Vec::with_capacity(rows);
        let mut scratch_cap = 1usize;

        for combo in &combos {
            let ma_length = combo.ma_length.unwrap_or(DEFAULT_MA_LENGTH);
            let pmarp_lookback = combo.pmarp_lookback.unwrap_or(DEFAULT_PMARP_LOOKBACK);
            let signal_ma_length = combo.signal_ma_length.unwrap_or(DEFAULT_SIGNAL_MA_LENGTH);
            let ma_type = combo
                .ma_type
                .unwrap_or(PriceMovingAverageRatioPercentileMaType::Vwma);
            let signal_ma_type = combo
                .signal_ma_type
                .unwrap_or(PriceMovingAverageRatioPercentileMaType::Sma);
            let line_mode = combo
                .line_mode
                .unwrap_or(PriceMovingAverageRatioPercentileLineMode::Pmarp);

            if ma_length == 0 {
                return Err(CudaPriceMovingAverageRatioPercentileError::InvalidInput(
                    "invalid ma_length: 0".into(),
                ));
            }
            if pmarp_lookback == 0 {
                return Err(CudaPriceMovingAverageRatioPercentileError::InvalidInput(
                    "invalid pmarp_lookback: 0".into(),
                ));
            }
            if signal_ma_length == 0 {
                return Err(CudaPriceMovingAverageRatioPercentileError::InvalidInput(
                    "invalid signal_ma_length: 0".into(),
                ));
            }
            if ma_length > cols {
                return Err(CudaPriceMovingAverageRatioPercentileError::InvalidInput(
                    format!("invalid ma_length: ma_length={ma_length}, data_len={cols}"),
                ));
            }
            if signal_ma_length > cols {
                return Err(CudaPriceMovingAverageRatioPercentileError::InvalidInput(
                    format!(
                    "invalid signal_ma_length: signal_ma_length={signal_ma_length}, data_len={cols}"
                ),
                ));
            }

            scratch_cap = scratch_cap.max(ma_length).max(signal_ma_length);
            ma_lengths.push(i32::try_from(ma_length).map_err(|_| {
                CudaPriceMovingAverageRatioPercentileError::InvalidInput(format!(
                    "ma_length out of range: {ma_length}"
                ))
            })?);
            pmarp_lookbacks.push(i32::try_from(pmarp_lookback).map_err(|_| {
                CudaPriceMovingAverageRatioPercentileError::InvalidInput(format!(
                    "pmarp_lookback out of range: {pmarp_lookback}"
                ))
            })?);
            signal_ma_lengths.push(i32::try_from(signal_ma_length).map_err(|_| {
                CudaPriceMovingAverageRatioPercentileError::InvalidInput(format!(
                    "signal_ma_length out of range: {signal_ma_length}"
                ))
            })?);
            ma_codes.push(parse_ma_type(ma_type));
            signal_ma_codes.push(parse_ma_type(signal_ma_type));
            line_modes.push(parse_line_mode(line_mode));
        }

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                CudaPriceMovingAverageRatioPercentileError::InvalidInput(
                    "input bytes overflow".into(),
                )
            })?;
        let params_bytes = rows
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| value.checked_mul(6))
            .ok_or_else(|| {
                CudaPriceMovingAverageRatioPercentileError::InvalidInput(
                    "params bytes overflow".into(),
                )
            })?;
        let scratch_elems = rows
            .checked_mul(scratch_cap)
            .and_then(|value| value.checked_mul(10))
            .ok_or_else(|| {
                CudaPriceMovingAverageRatioPercentileError::InvalidInput(
                    "scratch size overflow".into(),
                )
            })?;
        let scratch_bytes = scratch_elems
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaPriceMovingAverageRatioPercentileError::InvalidInput(
                    "scratch bytes overflow".into(),
                )
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaPriceMovingAverageRatioPercentileError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(7))
            .ok_or_else(|| {
                CudaPriceMovingAverageRatioPercentileError::InvalidInput(
                    "output bytes overflow".into(),
                )
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(scratch_bytes))
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaPriceMovingAverageRatioPercentileError::InvalidInput(
                    "required bytes overflow".into(),
                )
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_price = DeviceBuffer::from_slice(price)?;
        let d_volume = DeviceBuffer::from_slice(volume)?;
        let d_ma_lengths = DeviceBuffer::from_slice(&ma_lengths)?;
        let d_pmarp_lookbacks = DeviceBuffer::from_slice(&pmarp_lookbacks)?;
        let d_signal_ma_lengths = DeviceBuffer::from_slice(&signal_ma_lengths)?;
        let d_ma_codes = DeviceBuffer::from_slice(&ma_codes)?;
        let d_signal_ma_codes = DeviceBuffer::from_slice(&signal_ma_codes)?;
        let d_line_modes = DeviceBuffer::from_slice(&line_modes)?;
        let d_scratch = unsafe { DeviceBuffer::<f64>::uninitialized(scratch_elems)? };
        let d_out_pmar = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_pmarp = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_plotline = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_signal = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_pmar_high = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_pmar_low = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_scaled_pmar = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("price_moving_average_ratio_percentile_batch_f64")
            .map_err(
                |_| CudaPriceMovingAverageRatioPercentileError::MissingKernelSymbol {
                    name: "price_moving_average_ratio_percentile_batch_f64",
                },
            )?;
        let grid_x = ((rows as u32) + PRICE_MOVING_AVERAGE_RATIO_PERCENTILE_BLOCK_X - 1)
            / PRICE_MOVING_AVERAGE_RATIO_PERCENTILE_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(PRICE_MOVING_AVERAGE_RATIO_PERCENTILE_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_price.as_device_ptr(),
                d_volume.as_device_ptr(),
                cols as i32,
                d_ma_lengths.as_device_ptr(),
                d_pmarp_lookbacks.as_device_ptr(),
                d_signal_ma_lengths.as_device_ptr(),
                d_ma_codes.as_device_ptr(),
                d_signal_ma_codes.as_device_ptr(),
                d_line_modes.as_device_ptr(),
                rows as i32,
                scratch_cap as i32,
                d_scratch.as_device_ptr(),
                d_out_pmar.as_device_ptr(),
                d_out_pmarp.as_device_ptr(),
                d_out_plotline.as_device_ptr(),
                d_out_signal.as_device_ptr(),
                d_out_pmar_high.as_device_ptr(),
                d_out_pmar_low.as_device_ptr(),
                d_out_scaled_pmar.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaPriceMovingAverageRatioPercentileBatchResult {
            outputs: PriceMovingAverageRatioPercentileDeviceArrayF64Seven {
                pmar: PriceMovingAverageRatioPercentileDeviceArrayF64 {
                    buf: d_out_pmar,
                    rows,
                    cols,
                },
                pmarp: PriceMovingAverageRatioPercentileDeviceArrayF64 {
                    buf: d_out_pmarp,
                    rows,
                    cols,
                },
                plotline: PriceMovingAverageRatioPercentileDeviceArrayF64 {
                    buf: d_out_plotline,
                    rows,
                    cols,
                },
                signal: PriceMovingAverageRatioPercentileDeviceArrayF64 {
                    buf: d_out_signal,
                    rows,
                    cols,
                },
                pmar_high: PriceMovingAverageRatioPercentileDeviceArrayF64 {
                    buf: d_out_pmar_high,
                    rows,
                    cols,
                },
                pmar_low: PriceMovingAverageRatioPercentileDeviceArrayF64 {
                    buf: d_out_pmar_low,
                    rows,
                    cols,
                },
                scaled_pmar: PriceMovingAverageRatioPercentileDeviceArrayF64 {
                    buf: d_out_scaled_pmar,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
