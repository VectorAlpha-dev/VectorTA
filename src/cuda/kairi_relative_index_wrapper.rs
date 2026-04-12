#![cfg(feature = "cuda")]

use crate::indicators::kairi_relative_index::{
    expand_grid_kairi_relative_index, KairiRelativeIndexBatchRange, KairiRelativeIndexParams,
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

const KAIRI_RELATIVE_INDEX_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const DEFAULT_LENGTH: usize = 50;
const MA_SMA: i32 = 0;
const MA_EMA: i32 = 1;
const MA_WMA: i32 = 2;
const MA_TMA: i32 = 3;
const MA_VIDYA: i32 = 4;
const MA_WWMA: i32 = 5;
const MA_ZLEMA: i32 = 6;
const MA_TSF: i32 = 7;
const MA_HMA: i32 = 8;
const MA_VWMA: i32 = 9;

#[derive(Debug, Error)]
pub enum CudaKairiRelativeIndexError {
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

pub struct KairiRelativeIndexDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl KairiRelativeIndexDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct CudaKairiRelativeIndexBatchResult {
    pub outputs: KairiRelativeIndexDeviceArrayF64,
    pub combos: Vec<KairiRelativeIndexParams>,
}

pub struct CudaKairiRelativeIndex {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaKairiRelativeIndex {
    pub fn new(device_id: usize) -> Result<Self, CudaKairiRelativeIndexError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("kairi_relative_index_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaKairiRelativeIndexError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn parse_ma_type(value: &str) -> Result<i32, CudaKairiRelativeIndexError> {
        match value.trim().to_ascii_uppercase().as_str() {
            "SMA" => Ok(MA_SMA),
            "EMA" => Ok(MA_EMA),
            "WMA" => Ok(MA_WMA),
            "TMA" => Ok(MA_TMA),
            "VIDYA" => Ok(MA_VIDYA),
            "WWMA" => Ok(MA_WWMA),
            "ZLEMA" => Ok(MA_ZLEMA),
            "TSF" => Ok(MA_TSF),
            "HMA" | "HULL" => Ok(MA_HMA),
            "VWMA" => Ok(MA_VWMA),
            _ => Err(CudaKairiRelativeIndexError::InvalidInput(format!(
                "invalid ma_type: {value}"
            ))),
        }
    }

    fn needs_volume(ma_code: i32) -> bool {
        ma_code == MA_VWMA
    }

    fn required_samples(ma_code: i32, length: usize) -> usize {
        match ma_code {
            MA_WWMA | MA_VIDYA => 1,
            MA_HMA => length + (length as f64).sqrt().floor() as usize - 1,
            _ => length,
        }
    }

    fn longest_valid_run(source: &[f64], volume: &[f64], needs_volume: bool) -> usize {
        let mut best = 0usize;
        let mut cur = 0usize;
        for (&src, &vol) in source.iter().zip(volume.iter()) {
            let valid = src.is_finite() && (!needs_volume || vol.is_finite());
            if valid {
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

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaKairiRelativeIndexError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaKairiRelativeIndexError::OutOfMemory {
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
    ) -> Result<(), CudaKairiRelativeIndexError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaKairiRelativeIndexError::LaunchConfigTooLarge {
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
        source: &[f64],
        volume: &[f64],
        sweep: &KairiRelativeIndexBatchRange,
    ) -> Result<CudaKairiRelativeIndexBatchResult, CudaKairiRelativeIndexError> {
        if source.is_empty() || volume.is_empty() {
            return Err(CudaKairiRelativeIndexError::InvalidInput(
                "empty input".into(),
            ));
        }
        if source.len() != volume.len() {
            return Err(CudaKairiRelativeIndexError::InvalidInput(format!(
                "input length mismatch: source={}, volume={}",
                source.len(),
                volume.len()
            )));
        }

        let combos = expand_grid_kairi_relative_index(sweep)
            .map_err(|err| CudaKairiRelativeIndexError::InvalidInput(err.to_string()))?;
        if combos.is_empty() {
            return Err(CudaKairiRelativeIndexError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        if !source.iter().any(|value| value.is_finite()) {
            return Err(CudaKairiRelativeIndexError::InvalidInput(
                "all values are NaN".into(),
            ));
        }

        let rows = combos.len();
        let cols = source.len();
        let mut lengths = Vec::with_capacity(rows);
        let mut ma_codes = Vec::with_capacity(rows);

        for combo in &combos {
            let length = combo.length.unwrap_or(DEFAULT_LENGTH);
            if length < 2 || length > cols {
                return Err(CudaKairiRelativeIndexError::InvalidInput(format!(
                    "invalid length: length={length}, data_len={cols}"
                )));
            }
            let ma_code = Self::parse_ma_type(combo.ma_type.as_deref().unwrap_or("SMA"))?;
            let valid = Self::longest_valid_run(source, volume, Self::needs_volume(ma_code));
            let needed = Self::required_samples(ma_code, length);
            if valid < needed {
                return Err(CudaKairiRelativeIndexError::InvalidInput(format!(
                    "not enough valid data: needed={needed}, valid={valid}"
                )));
            }
            lengths.push(length as i32);
            ma_codes.push(ma_code);
        }

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                CudaKairiRelativeIndexError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = rows
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                CudaKairiRelativeIndexError::InvalidInput("params bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaKairiRelativeIndexError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaKairiRelativeIndexError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaKairiRelativeIndexError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_source = DeviceBuffer::from_slice(source)?;
        let d_volume = DeviceBuffer::from_slice(volume)?;
        let d_lengths = DeviceBuffer::from_slice(&lengths)?;
        let d_ma_codes = DeviceBuffer::from_slice(&ma_codes)?;
        let d_out = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("kairi_relative_index_batch_f64")
            .map_err(|_| CudaKairiRelativeIndexError::MissingKernelSymbol {
                name: "kairi_relative_index_batch_f64",
            })?;
        let grid_x =
            ((rows as u32) + KAIRI_RELATIVE_INDEX_BLOCK_X - 1) / KAIRI_RELATIVE_INDEX_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(KAIRI_RELATIVE_INDEX_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_source.as_device_ptr(),
                d_volume.as_device_ptr(),
                cols as i32,
                d_lengths.as_device_ptr(),
                d_ma_codes.as_device_ptr(),
                rows as i32,
                d_out.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaKairiRelativeIndexBatchResult {
            outputs: KairiRelativeIndexDeviceArrayF64 {
                buf: d_out,
                rows,
                cols,
            },
            combos,
        })
    }
}
