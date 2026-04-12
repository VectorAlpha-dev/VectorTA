#![cfg(feature = "cuda")]

use crate::indicators::keltner_channel_width_oscillator::{
    expand_grid_keltner_channel_width_oscillator, KeltnerChannelWidthOscillatorBatchRange,
    KeltnerChannelWidthOscillatorParams,
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

const KELTNER_CHANNEL_WIDTH_OSCILLATOR_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const DEFAULT_LENGTH: usize = 20;
const DEFAULT_MULTIPLIER: f64 = 2.0;
const DEFAULT_ATR_LENGTH: usize = 10;

const BANDS_STYLE_ATR: i32 = 0;
const BANDS_STYLE_TR: i32 = 1;
const BANDS_STYLE_RANGE: i32 = 2;

#[derive(Debug, Error)]
pub enum CudaKeltnerChannelWidthOscillatorError {
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

pub struct KeltnerChannelWidthOscillatorDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl KeltnerChannelWidthOscillatorDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct KeltnerChannelWidthOscillatorDeviceArrayF64Pair {
    pub kbw: KeltnerChannelWidthOscillatorDeviceArrayF64,
    pub kbw_sma: KeltnerChannelWidthOscillatorDeviceArrayF64,
}

impl KeltnerChannelWidthOscillatorDeviceArrayF64Pair {
    #[inline]
    pub fn rows(&self) -> usize {
        self.kbw.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.kbw.cols
    }
}

pub struct CudaKeltnerChannelWidthOscillatorBatchResult {
    pub outputs: KeltnerChannelWidthOscillatorDeviceArrayF64Pair,
    pub combos: Vec<KeltnerChannelWidthOscillatorParams>,
}

pub struct CudaKeltnerChannelWidthOscillator {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

fn parse_bands_style(value: &str) -> Result<i32, CudaKeltnerChannelWidthOscillatorError> {
    let normalized = value.trim().to_ascii_uppercase();
    match normalized.as_str() {
        "AVERAGE TRUE RANGE" | "ATR" => Ok(BANDS_STYLE_ATR),
        "TRUE RANGE" | "TR" => Ok(BANDS_STYLE_TR),
        "RANGE" => Ok(BANDS_STYLE_RANGE),
        _ => Err(CudaKeltnerChannelWidthOscillatorError::InvalidInput(
            format!("invalid bands style: {value}"),
        )),
    }
}

fn is_valid_bar(high: f64, low: f64, close: f64, source: f64) -> bool {
    high.is_finite() && low.is_finite() && close.is_finite() && source.is_finite() && high >= low
}

fn first_valid_bar(high: &[f64], low: &[f64], close: &[f64], source: &[f64]) -> Option<usize> {
    (0..source.len()).find(|&i| is_valid_bar(high[i], low[i], close[i], source[i]))
}

fn width_needed_bars(length: usize, atr_length: usize, bands_style: i32) -> usize {
    match bands_style {
        BANDS_STYLE_ATR => length.max(atr_length),
        BANDS_STYLE_TR | BANDS_STYLE_RANGE => length,
        _ => length,
    }
}

impl CudaKeltnerChannelWidthOscillator {
    pub fn new(device_id: usize) -> Result<Self, CudaKeltnerChannelWidthOscillatorError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("keltner_channel_width_oscillator_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaKeltnerChannelWidthOscillatorError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn will_fit(
        required: usize,
        headroom: usize,
    ) -> Result<(), CudaKeltnerChannelWidthOscillatorError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaKeltnerChannelWidthOscillatorError::OutOfMemory {
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
    ) -> Result<(), CudaKeltnerChannelWidthOscillatorError> {
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
                CudaKeltnerChannelWidthOscillatorError::LaunchConfigTooLarge {
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
        high: &[f64],
        low: &[f64],
        close: &[f64],
        source: &[f64],
        sweep: &KeltnerChannelWidthOscillatorBatchRange,
    ) -> Result<CudaKeltnerChannelWidthOscillatorBatchResult, CudaKeltnerChannelWidthOscillatorError>
    {
        if source.is_empty() {
            return Err(CudaKeltnerChannelWidthOscillatorError::InvalidInput(
                "empty input".into(),
            ));
        }
        if high.len() != source.len() || low.len() != source.len() || close.len() != source.len() {
            return Err(CudaKeltnerChannelWidthOscillatorError::InvalidInput(
                "data length mismatch".into(),
            ));
        }

        let first = first_valid_bar(high, low, close, source).ok_or_else(|| {
            CudaKeltnerChannelWidthOscillatorError::InvalidInput("all values are NaN".into())
        })?;
        let combos = expand_grid_keltner_channel_width_oscillator(sweep)
            .map_err(|err| CudaKeltnerChannelWidthOscillatorError::InvalidInput(err.to_string()))?;
        if combos.is_empty() {
            return Err(CudaKeltnerChannelWidthOscillatorError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let rows = combos.len();
        let cols = source.len();
        let valid = cols - first;
        let mut lengths = Vec::with_capacity(rows);
        let mut multipliers = Vec::with_capacity(rows);
        let mut use_exponentials = Vec::with_capacity(rows);
        let mut bands_styles = Vec::with_capacity(rows);
        let mut atr_lengths = Vec::with_capacity(rows);
        let mut max_length = 0usize;

        for combo in &combos {
            let length = combo.length.unwrap_or(DEFAULT_LENGTH);
            let multiplier = combo.multiplier.unwrap_or(DEFAULT_MULTIPLIER);
            let use_exponential = combo.use_exponential.unwrap_or(true);
            let bands_style =
                parse_bands_style(combo.bands_style.as_deref().unwrap_or("Average True Range"))?;
            let atr_length = combo.atr_length.unwrap_or(DEFAULT_ATR_LENGTH);

            if length == 0 || length > cols {
                return Err(CudaKeltnerChannelWidthOscillatorError::InvalidInput(
                    format!("invalid length: length={length}, data_len={cols}"),
                ));
            }
            if !multiplier.is_finite() || multiplier < 0.0 {
                return Err(CudaKeltnerChannelWidthOscillatorError::InvalidInput(
                    format!("invalid multiplier: {multiplier}"),
                ));
            }
            if atr_length == 0 || (bands_style == BANDS_STYLE_ATR && atr_length > cols) {
                return Err(CudaKeltnerChannelWidthOscillatorError::InvalidInput(
                    format!("invalid atr_length: atr_length={atr_length}, data_len={cols}"),
                ));
            }

            let needed = width_needed_bars(length, atr_length, bands_style);
            if valid < needed {
                return Err(CudaKeltnerChannelWidthOscillatorError::InvalidInput(
                    format!("not enough valid data: needed={needed}, valid={valid}"),
                ));
            }

            lengths.push(i32::try_from(length).map_err(|_| {
                CudaKeltnerChannelWidthOscillatorError::InvalidInput(format!(
                    "length out of range: {length}"
                ))
            })?);
            multipliers.push(multiplier);
            use_exponentials.push(if use_exponential { 1 } else { 0 });
            bands_styles.push(bands_style);
            atr_lengths.push(i32::try_from(atr_length).map_err(|_| {
                CudaKeltnerChannelWidthOscillatorError::InvalidInput(format!(
                    "atr_length out of range: {atr_length}"
                ))
            })?);
            max_length = max_length.max(length);
        }

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| {
                CudaKeltnerChannelWidthOscillatorError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = rows
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| value.checked_mul(4))
            .and_then(|value| {
                rows.checked_mul(std::mem::size_of::<f64>())
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| {
                CudaKeltnerChannelWidthOscillatorError::InvalidInput("params bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaKeltnerChannelWidthOscillatorError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                CudaKeltnerChannelWidthOscillatorError::InvalidInput("output bytes overflow".into())
            })?;
        let scratch_elems = rows
            .checked_mul(max_length)
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                CudaKeltnerChannelWidthOscillatorError::InvalidInput(
                    "scratch elements overflow".into(),
                )
            })?;
        let scratch_bytes = scratch_elems
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaKeltnerChannelWidthOscillatorError::InvalidInput(
                    "scratch bytes overflow".into(),
                )
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .and_then(|value| value.checked_add(scratch_bytes))
            .ok_or_else(|| {
                CudaKeltnerChannelWidthOscillatorError::InvalidInput(
                    "required bytes overflow".into(),
                )
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_close = DeviceBuffer::from_slice(close)?;
        let d_source = DeviceBuffer::from_slice(source)?;
        let d_lengths = DeviceBuffer::from_slice(&lengths)?;
        let d_multipliers = DeviceBuffer::from_slice(&multipliers)?;
        let d_use_exponentials = DeviceBuffer::from_slice(&use_exponentials)?;
        let d_bands_styles = DeviceBuffer::from_slice(&bands_styles)?;
        let d_atr_lengths = DeviceBuffer::from_slice(&atr_lengths)?;
        let d_out_kbw = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_kbw_sma = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let scratch_count = rows * max_length;
        let d_center_sma_buffers = unsafe { DeviceBuffer::<f64>::uninitialized(scratch_count)? };
        let d_width_sma_buffers = unsafe { DeviceBuffer::<f64>::uninitialized(scratch_count)? };

        let func = self
            .module
            .get_function("keltner_channel_width_oscillator_batch_f64")
            .map_err(
                |_| CudaKeltnerChannelWidthOscillatorError::MissingKernelSymbol {
                    name: "keltner_channel_width_oscillator_batch_f64",
                },
            )?;
        let grid_x = ((rows as u32) + KELTNER_CHANNEL_WIDTH_OSCILLATOR_BLOCK_X - 1)
            / KELTNER_CHANNEL_WIDTH_OSCILLATOR_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(KELTNER_CHANNEL_WIDTH_OSCILLATOR_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_high.as_device_ptr(),
                d_low.as_device_ptr(),
                d_close.as_device_ptr(),
                d_source.as_device_ptr(),
                i32::try_from(cols).map_err(|_| {
                    CudaKeltnerChannelWidthOscillatorError::InvalidInput("cols out of range".into())
                })?,
                d_lengths.as_device_ptr(),
                d_multipliers.as_device_ptr(),
                d_use_exponentials.as_device_ptr(),
                d_bands_styles.as_device_ptr(),
                d_atr_lengths.as_device_ptr(),
                i32::try_from(rows).map_err(|_| {
                    CudaKeltnerChannelWidthOscillatorError::InvalidInput("rows out of range".into())
                })?,
                i32::try_from(max_length).map_err(|_| {
                    CudaKeltnerChannelWidthOscillatorError::InvalidInput(
                        "max_length out of range".into(),
                    )
                })?,
                d_out_kbw.as_device_ptr(),
                d_out_kbw_sma.as_device_ptr(),
                d_center_sma_buffers.as_device_ptr(),
                d_width_sma_buffers.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaKeltnerChannelWidthOscillatorBatchResult {
            outputs: KeltnerChannelWidthOscillatorDeviceArrayF64Pair {
                kbw: KeltnerChannelWidthOscillatorDeviceArrayF64 {
                    buf: d_out_kbw,
                    rows,
                    cols,
                },
                kbw_sma: KeltnerChannelWidthOscillatorDeviceArrayF64 {
                    buf: d_out_kbw_sma,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
