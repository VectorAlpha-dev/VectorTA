#![cfg(feature = "cuda")]

use crate::indicators::ema::{ema_with_kernel, EmaData, EmaInput, EmaParams};
use crate::indicators::emd_trend::{expand_grid_emd_trend, EmdTrendBatchRange, EmdTrendParams};
use crate::indicators::moving_averages::ma::{ma_with_kernel, MaData};
use crate::utilities::enums::Kernel;
use cust::context::Context;
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, GridSize};
use cust::launch;
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::Module;
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::error::Error;
use std::sync::Arc;
use thiserror::Error;

const EMD_TREND_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const DEFAULT_SOURCE: &str = "close";
const DEFAULT_AVG_TYPE: &str = "SMA";
const DEFAULT_LENGTH: usize = 28;
const DEFAULT_MULT: f64 = 1.0;

#[derive(Clone, Copy, Debug)]
enum SourceKind {
    Open,
    High,
    Low,
    Close,
    Oc2,
    Hl2,
    Occ3,
    Hlc3,
    Ohlc4,
    Hlcc4,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AvgKind {
    Sma,
    Ema,
    Hma,
    Dema,
    Tema,
    Rma,
    Frama,
}

#[derive(Debug, Error)]
pub enum CudaEmdTrendError {
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

pub struct EmdTrendDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl EmdTrendDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct EmdTrendDeviceOutputs {
    pub direction: EmdTrendDeviceArrayF64,
    pub average: EmdTrendDeviceArrayF64,
    pub upper: EmdTrendDeviceArrayF64,
    pub lower: EmdTrendDeviceArrayF64,
}

impl EmdTrendDeviceOutputs {
    #[inline]
    pub fn rows(&self) -> usize {
        self.average.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.average.cols
    }
}

pub struct CudaEmdTrendBatchResult {
    pub outputs: EmdTrendDeviceOutputs,
    pub combos: Vec<EmdTrendParams>,
}

pub struct CudaEmdTrend {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

#[inline]
fn parse_source_kind(value: &str) -> Result<SourceKind, CudaEmdTrendError> {
    if value.eq_ignore_ascii_case("open") {
        Ok(SourceKind::Open)
    } else if value.eq_ignore_ascii_case("high") {
        Ok(SourceKind::High)
    } else if value.eq_ignore_ascii_case("low") {
        Ok(SourceKind::Low)
    } else if value.eq_ignore_ascii_case("close") {
        Ok(SourceKind::Close)
    } else if value.eq_ignore_ascii_case("oc2") {
        Ok(SourceKind::Oc2)
    } else if value.eq_ignore_ascii_case("hl2") {
        Ok(SourceKind::Hl2)
    } else if value.eq_ignore_ascii_case("occ3") {
        Ok(SourceKind::Occ3)
    } else if value.eq_ignore_ascii_case("hlc3") {
        Ok(SourceKind::Hlc3)
    } else if value.eq_ignore_ascii_case("ohlc4") {
        Ok(SourceKind::Ohlc4)
    } else if value.eq_ignore_ascii_case("hlcc4") {
        Ok(SourceKind::Hlcc4)
    } else {
        Err(CudaEmdTrendError::InvalidInput(format!(
            "invalid source: {value}"
        )))
    }
}

#[inline]
fn parse_avg_kind(value: &str) -> Result<AvgKind, CudaEmdTrendError> {
    if value.eq_ignore_ascii_case("SMA") {
        Ok(AvgKind::Sma)
    } else if value.eq_ignore_ascii_case("EMA") {
        Ok(AvgKind::Ema)
    } else if value.eq_ignore_ascii_case("HMA") {
        Ok(AvgKind::Hma)
    } else if value.eq_ignore_ascii_case("DEMA") {
        Ok(AvgKind::Dema)
    } else if value.eq_ignore_ascii_case("TEMA") {
        Ok(AvgKind::Tema)
    } else if value.eq_ignore_ascii_case("RMA") {
        Ok(AvgKind::Rma)
    } else if value.eq_ignore_ascii_case("FRAMA") {
        Ok(AvgKind::Frama)
    } else {
        Err(CudaEmdTrendError::InvalidInput(format!(
            "invalid avg_type: {value}"
        )))
    }
}

#[inline]
fn avg_id(kind: AvgKind) -> &'static str {
    match kind {
        AvgKind::Sma => "sma",
        AvgKind::Ema => "ema",
        AvgKind::Hma => "hma",
        AvgKind::Dema => "dema",
        AvgKind::Tema => "tema",
        AvgKind::Rma => "wilders",
        AvgKind::Frama => "frama",
    }
}

#[inline]
fn source_value(kind: SourceKind, open: f64, high: f64, low: f64, close: f64) -> f64 {
    match kind {
        SourceKind::Open => open,
        SourceKind::High => high,
        SourceKind::Low => low,
        SourceKind::Close => close,
        SourceKind::Oc2 => (open + close) * 0.5,
        SourceKind::Hl2 => (high + low) * 0.5,
        SourceKind::Occ3 => (open + close + close) / 3.0,
        SourceKind::Hlc3 => (high + low + close) / 3.0,
        SourceKind::Ohlc4 => (open + high + low + close) * 0.25,
        SourceKind::Hlcc4 => (high + low + close + close) * 0.25,
    }
}

#[inline]
fn longest_valid_run(values: &[f64]) -> usize {
    let mut best = 0usize;
    let mut current = 0usize;
    for &value in values {
        if value.is_finite() {
            current += 1;
            best = best.max(current);
        } else {
            current = 0;
        }
    }
    best
}

#[inline]
fn parse_number_after(segment: &str) -> Option<usize> {
    let eq = segment.find('=')?;
    let number: String = segment[eq + 1..]
        .chars()
        .skip_while(|c| c.is_whitespace())
        .take_while(|c| c.is_ascii_digit())
        .collect();
    number.parse().ok()
}

#[inline]
fn parse_needed_valid(msg: &str) -> Option<(usize, usize)> {
    let needed_key = "needed";
    let valid_key = "valid";
    let needed_idx = msg.find(needed_key)?;
    let valid_idx = msg.find(valid_key)?;
    let needed = parse_number_after(&msg[needed_idx + needed_key.len()..])?;
    let valid = parse_number_after(&msg[valid_idx + valid_key.len()..])?;
    Some((needed, valid))
}

#[inline]
fn map_ma_error(err: Box<dyn Error>) -> CudaEmdTrendError {
    let msg = err.to_string();
    if let Some((needed, valid)) = parse_needed_valid(&msg) {
        return CudaEmdTrendError::InvalidInput(format!(
            "not enough valid data: needed={needed}, valid={valid}"
        ));
    }
    CudaEmdTrendError::InvalidInput(msg)
}

#[inline]
fn compute_average_series(
    src: &[f64],
    avg_kind: AvgKind,
    length: usize,
) -> Result<Vec<f64>, CudaEmdTrendError> {
    ma_with_kernel(avg_id(avg_kind), MaData::Slice(src), length, Kernel::Scalar)
        .map_err(map_ma_error)
}

#[inline]
fn compute_deviation_ema(
    average: &[f64],
    src: &[f64],
    length: usize,
) -> Result<Vec<f64>, CudaEmdTrendError> {
    let mut abs_dev = vec![f64::NAN; src.len()];
    for i in 0..src.len() {
        if average[i].is_finite() && src[i].is_finite() {
            abs_dev[i] = (src[i] - average[i]).abs();
        }
    }
    let input = EmaInput {
        data: EmaData::Slice(&abs_dev),
        params: EmaParams {
            period: Some(length),
        },
    };
    ema_with_kernel(&input, Kernel::Scalar)
        .map(|out| out.values)
        .map_err(|err| CudaEmdTrendError::InvalidInput(err.to_string()))
}

impl CudaEmdTrend {
    pub fn new(device_id: usize) -> Result<Self, CudaEmdTrendError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("emd_trend_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaEmdTrendError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaEmdTrendError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaEmdTrendError::OutOfMemory {
                    required,
                    free,
                    headroom,
                });
            }
        }
        Ok(())
    }

    fn validate_launch(&self, grid: GridSize, block: BlockSize) -> Result<(), CudaEmdTrendError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaEmdTrendError::LaunchConfigTooLarge {
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
        sweep: &EmdTrendBatchRange,
        source: &str,
        avg_type: &str,
    ) -> Result<CudaEmdTrendBatchResult, CudaEmdTrendError> {
        if open.is_empty() || high.is_empty() || low.is_empty() || close.is_empty() {
            return Err(CudaEmdTrendError::InvalidInput("empty input".into()));
        }
        if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
            return Err(CudaEmdTrendError::InvalidInput(format!(
                "input length mismatch: open={}, high={}, low={}, close={}",
                open.len(),
                high.len(),
                low.len(),
                close.len()
            )));
        }

        let source_kind = parse_source_kind(if source.is_empty() {
            DEFAULT_SOURCE
        } else {
            source
        })?;
        let avg_kind = parse_avg_kind(if avg_type.is_empty() {
            DEFAULT_AVG_TYPE
        } else {
            avg_type
        })?;
        let combos = expand_grid_emd_trend(sweep, source, avg_type)
            .map_err(|err| CudaEmdTrendError::InvalidInput(err.to_string()))?;
        if combos.is_empty() {
            return Err(CudaEmdTrendError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let mut src = Vec::with_capacity(close.len());
        for i in 0..close.len() {
            src.push(source_value(
                source_kind,
                open[i],
                high[i],
                low[i],
                close[i],
            ));
        }
        let max_run = longest_valid_run(&src);
        if max_run == 0 {
            return Err(CudaEmdTrendError::InvalidInput("all values are NaN".into()));
        }

        let rows = combos.len();
        let cols = src.len();
        let output_elems = rows
            .checked_mul(cols)
            .ok_or_else(|| CudaEmdTrendError::InvalidInput("rows*cols overflow".into()))?;
        let mut mults = Vec::with_capacity(rows);
        let mut average_flat = vec![f64::NAN; output_elems];
        let mut deviation_flat = vec![f64::NAN; output_elems];

        for (row, combo) in combos.iter().enumerate() {
            let length = combo.length.unwrap_or(DEFAULT_LENGTH);
            let mult = combo.mult.unwrap_or(DEFAULT_MULT);
            if length == 0 {
                return Err(CudaEmdTrendError::InvalidInput(format!(
                    "invalid length: {length}"
                )));
            }
            if !mult.is_finite() || mult < 0.05 {
                return Err(CudaEmdTrendError::InvalidInput(format!(
                    "invalid mult: {mult}"
                )));
            }
            let needed = if avg_kind == AvgKind::Frama && length % 2 == 1 {
                length + 1
            } else {
                length
            };
            if max_run < needed {
                return Err(CudaEmdTrendError::InvalidInput(format!(
                    "not enough valid data: needed={needed}, valid={max_run}"
                )));
            }

            let average = compute_average_series(&src, avg_kind, length)?;
            let deviation = compute_deviation_ema(&average, &src, length)?;
            let start = row * cols;
            average_flat[start..start + cols].copy_from_slice(&average);
            deviation_flat[start..start + cols].copy_from_slice(&deviation);
            mults.push(mult);
        }

        let src_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| CudaEmdTrendError::InvalidInput("src bytes overflow".into()))?;
        let mult_bytes = rows
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| CudaEmdTrendError::InvalidInput("mult bytes overflow".into()))?;
        let matrix_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(5))
            .ok_or_else(|| CudaEmdTrendError::InvalidInput("matrix bytes overflow".into()))?;
        let required = src_bytes
            .checked_add(mult_bytes)
            .and_then(|value| value.checked_add(matrix_bytes))
            .ok_or_else(|| CudaEmdTrendError::InvalidInput("required bytes overflow".into()))?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_src = DeviceBuffer::from_slice(&src)?;
        let d_mults = DeviceBuffer::from_slice(&mults)?;
        let d_average = DeviceBuffer::from_slice(&average_flat)?;
        let d_deviation = DeviceBuffer::from_slice(&deviation_flat)?;
        let d_direction = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_upper = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_lower = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("emd_trend_batch_f64")
            .map_err(|_| CudaEmdTrendError::MissingKernelSymbol {
                name: "emd_trend_batch_f64",
            })?;
        let grid_x = ((rows as u32) + EMD_TREND_BLOCK_X - 1) / EMD_TREND_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(EMD_TREND_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_src.as_device_ptr(),
                cols as i32,
                d_mults.as_device_ptr(),
                d_average.as_device_ptr(),
                d_deviation.as_device_ptr(),
                rows as i32,
                d_direction.as_device_ptr(),
                d_upper.as_device_ptr(),
                d_lower.as_device_ptr()
            ))?;
        }
        self.stream.synchronize()?;

        Ok(CudaEmdTrendBatchResult {
            outputs: EmdTrendDeviceOutputs {
                direction: EmdTrendDeviceArrayF64 {
                    buf: d_direction,
                    rows,
                    cols,
                },
                average: EmdTrendDeviceArrayF64 {
                    buf: d_average,
                    rows,
                    cols,
                },
                upper: EmdTrendDeviceArrayF64 {
                    buf: d_upper,
                    rows,
                    cols,
                },
                lower: EmdTrendDeviceArrayF64 {
                    buf: d_lower,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
