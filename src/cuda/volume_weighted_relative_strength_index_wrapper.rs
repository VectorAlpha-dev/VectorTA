#![cfg(feature = "cuda")]

use crate::indicators::volume_weighted_relative_strength_index::{
    VolumeWeightedRelativeStrengthIndexBatchRange, VolumeWeightedRelativeStrengthIndexParams,
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

const VOLUME_WEIGHTED_RELATIVE_STRENGTH_INDEX_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const DEFAULT_RSI_LENGTH: usize = 14;
const DEFAULT_RANGE_LENGTH: usize = 10;
const DEFAULT_MA_LENGTH: usize = 14;
const DEFAULT_MA_TYPE: &str = "EMA";
const MA_EMA: i32 = 0;
const MA_SMA: i32 = 1;
const MA_HMA: i32 = 2;
const MA_RMA: i32 = 3;
const MA_WMA: i32 = 4;
const MA_VWMA: i32 = 5;

#[derive(Debug, Error)]
pub enum CudaVolumeWeightedRelativeStrengthIndexError {
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

pub struct VolumeWeightedRelativeStrengthIndexDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl VolumeWeightedRelativeStrengthIndexDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct VolumeWeightedRelativeStrengthIndexDeviceArrayF64Five {
    pub rsi: VolumeWeightedRelativeStrengthIndexDeviceArrayF64,
    pub consolidation_strength: VolumeWeightedRelativeStrengthIndexDeviceArrayF64,
    pub rsi_ma: VolumeWeightedRelativeStrengthIndexDeviceArrayF64,
    pub bearish_tp: VolumeWeightedRelativeStrengthIndexDeviceArrayF64,
    pub bullish_tp: VolumeWeightedRelativeStrengthIndexDeviceArrayF64,
}

impl VolumeWeightedRelativeStrengthIndexDeviceArrayF64Five {
    #[inline]
    pub fn rows(&self) -> usize {
        self.rsi.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.rsi.cols
    }
}

pub struct CudaVolumeWeightedRelativeStrengthIndexBatchResult {
    pub outputs: VolumeWeightedRelativeStrengthIndexDeviceArrayF64Five,
    pub combos: Vec<VolumeWeightedRelativeStrengthIndexParams>,
}

pub struct CudaVolumeWeightedRelativeStrengthIndex {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

fn parse_ma_type(value: &str) -> Result<i32, CudaVolumeWeightedRelativeStrengthIndexError> {
    let normalized = value.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "ema" => Ok(MA_EMA),
        "sma" => Ok(MA_SMA),
        "hma" => Ok(MA_HMA),
        "smma (rma)" | "smma" | "rma" => Ok(MA_RMA),
        "wma" => Ok(MA_WMA),
        "vwma" => Ok(MA_VWMA),
        _ => Err(CudaVolumeWeightedRelativeStrengthIndexError::InvalidInput(
            format!("invalid ma_type: {value}"),
        )),
    }
}

fn axis_usize(
    (start, end, step): (usize, usize, usize),
) -> Result<Vec<usize>, CudaVolumeWeightedRelativeStrengthIndexError> {
    if step == 0 || start == end {
        return Ok(vec![start]);
    }
    let mut out = Vec::new();
    if start <= end {
        let mut current = start;
        while current <= end {
            out.push(current);
            match current.checked_add(step) {
                Some(next) => current = next,
                None => break,
            }
        }
    } else {
        let mut current = start;
        while current >= end {
            out.push(current);
            match current.checked_sub(step) {
                Some(next) => current = next,
                None => break,
            }
            if current < end {
                break;
            }
        }
    }
    if out.is_empty() {
        return Err(CudaVolumeWeightedRelativeStrengthIndexError::InvalidInput(
            format!("invalid range: start={start}, end={end}, step={step}"),
        ));
    }
    Ok(out)
}

fn expand_grid(
    range: &VolumeWeightedRelativeStrengthIndexBatchRange,
) -> Result<
    Vec<VolumeWeightedRelativeStrengthIndexParams>,
    CudaVolumeWeightedRelativeStrengthIndexError,
> {
    let _ = parse_ma_type(&range.ma_type)?;
    let rsi_lengths = axis_usize(range.rsi_length)?;
    let range_lengths = axis_usize(range.range_length)?;
    let ma_lengths = axis_usize(range.ma_length)?;
    let total = rsi_lengths
        .len()
        .checked_mul(range_lengths.len())
        .and_then(|value| value.checked_mul(ma_lengths.len()))
        .ok_or_else(|| {
            CudaVolumeWeightedRelativeStrengthIndexError::InvalidInput(
                "parameter grid overflow".into(),
            )
        })?;

    let mut combos = Vec::with_capacity(total);
    for &rsi_length in &rsi_lengths {
        for &range_length in &range_lengths {
            for &ma_length in &ma_lengths {
                combos.push(VolumeWeightedRelativeStrengthIndexParams {
                    rsi_length: Some(rsi_length),
                    range_length: Some(range_length),
                    ma_length: Some(ma_length),
                    ma_type: Some(range.ma_type.clone()),
                });
            }
        }
    }
    Ok(combos)
}

impl CudaVolumeWeightedRelativeStrengthIndex {
    pub fn new(device_id: usize) -> Result<Self, CudaVolumeWeightedRelativeStrengthIndexError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module =
            crate::load_cuda_embedded_module!("volume_weighted_relative_strength_index_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaVolumeWeightedRelativeStrengthIndexError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn first_valid_pair(source: &[f64], volume: &[f64]) -> Option<usize> {
        source
            .iter()
            .zip(volume.iter())
            .position(|(&src, &vol)| src.is_finite() && vol.is_finite())
    }

    fn will_fit(
        required: usize,
        headroom: usize,
    ) -> Result<(), CudaVolumeWeightedRelativeStrengthIndexError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaVolumeWeightedRelativeStrengthIndexError::OutOfMemory {
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
    ) -> Result<(), CudaVolumeWeightedRelativeStrengthIndexError> {
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
                CudaVolumeWeightedRelativeStrengthIndexError::LaunchConfigTooLarge {
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
        source: &[f64],
        volume: &[f64],
        sweep: &VolumeWeightedRelativeStrengthIndexBatchRange,
    ) -> Result<
        CudaVolumeWeightedRelativeStrengthIndexBatchResult,
        CudaVolumeWeightedRelativeStrengthIndexError,
    > {
        if source.is_empty() || volume.is_empty() {
            return Err(CudaVolumeWeightedRelativeStrengthIndexError::InvalidInput(
                "empty input".into(),
            ));
        }
        if source.len() != volume.len() {
            return Err(CudaVolumeWeightedRelativeStrengthIndexError::InvalidInput(
                "source and volume length mismatch".into(),
            ));
        }

        let first = Self::first_valid_pair(source, volume).ok_or_else(|| {
            CudaVolumeWeightedRelativeStrengthIndexError::InvalidInput("all values are NaN".into())
        })?;
        let valid = source.len() - first;

        let combos = expand_grid(sweep)?;
        if combos.is_empty() {
            return Err(CudaVolumeWeightedRelativeStrengthIndexError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let rows = combos.len();
        let cols = source.len();
        let mut rsi_lengths = Vec::with_capacity(rows);
        let mut range_lengths = Vec::with_capacity(rows);
        let mut ma_lengths = Vec::with_capacity(rows);
        let mut ma_codes = Vec::with_capacity(rows);
        let mut max_ma_length = 0usize;
        let mut max_range_length = 0usize;

        for combo in &combos {
            let rsi_length = combo.rsi_length.unwrap_or(DEFAULT_RSI_LENGTH);
            let range_length = combo.range_length.unwrap_or(DEFAULT_RANGE_LENGTH);
            let ma_length = combo.ma_length.unwrap_or(DEFAULT_MA_LENGTH);
            let ma_code = parse_ma_type(combo.ma_type.as_deref().unwrap_or(DEFAULT_MA_TYPE))?;

            if rsi_length == 0 || rsi_length > cols {
                return Err(CudaVolumeWeightedRelativeStrengthIndexError::InvalidInput(
                    format!("invalid rsi_length: rsi_length={rsi_length}, data_len={cols}"),
                ));
            }
            if range_length == 0 || range_length > cols {
                return Err(CudaVolumeWeightedRelativeStrengthIndexError::InvalidInput(
                    format!("invalid range_length: range_length={range_length}, data_len={cols}"),
                ));
            }
            if ma_length == 0 || ma_length > cols {
                return Err(CudaVolumeWeightedRelativeStrengthIndexError::InvalidInput(
                    format!("invalid ma_length: ma_length={ma_length}, data_len={cols}"),
                ));
            }
            let needed = rsi_length + 1;
            if valid < needed {
                return Err(CudaVolumeWeightedRelativeStrengthIndexError::InvalidInput(
                    format!("not enough valid data: needed={needed}, valid={valid}"),
                ));
            }

            max_ma_length = max_ma_length.max(ma_length);
            max_range_length = max_range_length.max(range_length);
            rsi_lengths.push(i32::try_from(rsi_length).map_err(|_| {
                CudaVolumeWeightedRelativeStrengthIndexError::InvalidInput(format!(
                    "rsi_length out of range: {rsi_length}"
                ))
            })?);
            range_lengths.push(i32::try_from(range_length).map_err(|_| {
                CudaVolumeWeightedRelativeStrengthIndexError::InvalidInput(format!(
                    "range_length out of range: {range_length}"
                ))
            })?);
            ma_lengths.push(i32::try_from(ma_length).map_err(|_| {
                CudaVolumeWeightedRelativeStrengthIndexError::InvalidInput(format!(
                    "ma_length out of range: {ma_length}"
                ))
            })?);
            ma_codes.push(ma_code);
        }

        let scratch_cap = max_ma_length.max(max_range_length.saturating_mul(2)).max(1);

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                CudaVolumeWeightedRelativeStrengthIndexError::InvalidInput(
                    "input bytes overflow".into(),
                )
            })?;
        let params_bytes = rows
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| {
                CudaVolumeWeightedRelativeStrengthIndexError::InvalidInput(
                    "params bytes overflow".into(),
                )
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaVolumeWeightedRelativeStrengthIndexError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(5))
            .ok_or_else(|| {
                CudaVolumeWeightedRelativeStrengthIndexError::InvalidInput(
                    "output bytes overflow".into(),
                )
            })?;
        let scratch_elems = rows
            .checked_mul(scratch_cap)
            .and_then(|value| value.checked_mul(10))
            .ok_or_else(|| {
                CudaVolumeWeightedRelativeStrengthIndexError::InvalidInput(
                    "scratch elements overflow".into(),
                )
            })?;
        let scratch_bytes = scratch_elems
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaVolumeWeightedRelativeStrengthIndexError::InvalidInput(
                    "scratch bytes overflow".into(),
                )
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .and_then(|value| value.checked_add(scratch_bytes))
            .ok_or_else(|| {
                CudaVolumeWeightedRelativeStrengthIndexError::InvalidInput(
                    "required bytes overflow".into(),
                )
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_source = DeviceBuffer::from_slice(source)?;
        let d_volume = DeviceBuffer::from_slice(volume)?;
        let d_rsi_lengths = DeviceBuffer::from_slice(&rsi_lengths)?;
        let d_range_lengths = DeviceBuffer::from_slice(&range_lengths)?;
        let d_ma_lengths = DeviceBuffer::from_slice(&ma_lengths)?;
        let d_ma_codes = DeviceBuffer::from_slice(&ma_codes)?;
        let d_scratch = unsafe { DeviceBuffer::<f64>::uninitialized(scratch_elems)? };
        let d_out_rsi = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_consolidation = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_rsi_ma = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_bearish = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_bullish = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("volume_weighted_relative_strength_index_batch_f64")
            .map_err(
                |_| CudaVolumeWeightedRelativeStrengthIndexError::MissingKernelSymbol {
                    name: "volume_weighted_relative_strength_index_batch_f64",
                },
            )?;
        let grid_x = ((rows as u32) + VOLUME_WEIGHTED_RELATIVE_STRENGTH_INDEX_BLOCK_X - 1)
            / VOLUME_WEIGHTED_RELATIVE_STRENGTH_INDEX_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(VOLUME_WEIGHTED_RELATIVE_STRENGTH_INDEX_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_source.as_device_ptr(),
                d_volume.as_device_ptr(),
                cols as i32,
                d_rsi_lengths.as_device_ptr(),
                d_range_lengths.as_device_ptr(),
                d_ma_lengths.as_device_ptr(),
                d_ma_codes.as_device_ptr(),
                rows as i32,
                scratch_cap as i32,
                d_scratch.as_device_ptr(),
                d_out_rsi.as_device_ptr(),
                d_out_consolidation.as_device_ptr(),
                d_out_rsi_ma.as_device_ptr(),
                d_out_bearish.as_device_ptr(),
                d_out_bullish.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaVolumeWeightedRelativeStrengthIndexBatchResult {
            outputs: VolumeWeightedRelativeStrengthIndexDeviceArrayF64Five {
                rsi: VolumeWeightedRelativeStrengthIndexDeviceArrayF64 {
                    buf: d_out_rsi,
                    rows,
                    cols,
                },
                consolidation_strength: VolumeWeightedRelativeStrengthIndexDeviceArrayF64 {
                    buf: d_out_consolidation,
                    rows,
                    cols,
                },
                rsi_ma: VolumeWeightedRelativeStrengthIndexDeviceArrayF64 {
                    buf: d_out_rsi_ma,
                    rows,
                    cols,
                },
                bearish_tp: VolumeWeightedRelativeStrengthIndexDeviceArrayF64 {
                    buf: d_out_bearish,
                    rows,
                    cols,
                },
                bullish_tp: VolumeWeightedRelativeStrengthIndexDeviceArrayF64 {
                    buf: d_out_bullish,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
