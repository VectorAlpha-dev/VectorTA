#![cfg(feature = "cuda")]

use crate::indicators::volume_weighted_stochastic_rsi::{
    expand_grid_volume_weighted_stochastic_rsi, VolumeWeightedStochasticRsiBatchRange,
    VolumeWeightedStochasticRsiParams,
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

const VOLUME_WEIGHTED_STOCHASTIC_RSI_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const DEFAULT_RSI_LENGTH: usize = 14;
const DEFAULT_STOCH_LENGTH: usize = 14;
const DEFAULT_K_LENGTH: usize = 3;
const DEFAULT_D_LENGTH: usize = 3;
const MA_WSMA: i32 = 0;
const MA_SMA: i32 = 1;
const MA_EMA: i32 = 2;
const MA_WMA: i32 = 3;
const MA_VWMA: i32 = 4;

#[derive(Debug, Error)]
pub enum CudaVolumeWeightedStochasticRsiError {
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

pub struct VolumeWeightedStochasticRsiDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl VolumeWeightedStochasticRsiDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct VolumeWeightedStochasticRsiDeviceArrayF64Pair {
    pub k: VolumeWeightedStochasticRsiDeviceArrayF64,
    pub d: VolumeWeightedStochasticRsiDeviceArrayF64,
}

impl VolumeWeightedStochasticRsiDeviceArrayF64Pair {
    #[inline]
    pub fn rows(&self) -> usize {
        self.k.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.k.cols
    }
}

pub struct CudaVolumeWeightedStochasticRsiBatchResult {
    pub outputs: VolumeWeightedStochasticRsiDeviceArrayF64Pair,
    pub combos: Vec<VolumeWeightedStochasticRsiParams>,
}

pub struct CudaVolumeWeightedStochasticRsi {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaVolumeWeightedStochasticRsi {
    pub fn new(device_id: usize) -> Result<Self, CudaVolumeWeightedStochasticRsiError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("volume_weighted_stochastic_rsi_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaVolumeWeightedStochasticRsiError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn parse_ma_type(value: &str) -> Result<i32, CudaVolumeWeightedStochasticRsiError> {
        match value.trim().to_ascii_uppercase().as_str() {
            "WSMA" | "SMMA" | "RMA" | "WILDERS" | "WILDER" => Ok(MA_WSMA),
            "SMA" => Ok(MA_SMA),
            "EMA" => Ok(MA_EMA),
            "WMA" => Ok(MA_WMA),
            "VWMA" => Ok(MA_VWMA),
            _ => Err(CudaVolumeWeightedStochasticRsiError::InvalidInput(format!(
                "invalid ma_type: {value}"
            ))),
        }
    }

    fn ma_extra_bars(ma_code: i32, period: usize) -> usize {
        match ma_code {
            MA_EMA => 0,
            _ => period.saturating_sub(1),
        }
    }

    fn needed_bars(
        ma_code: i32,
        rsi_length: usize,
        stoch_length: usize,
        k_length: usize,
        d_length: usize,
    ) -> usize {
        rsi_length
            + stoch_length
            + Self::ma_extra_bars(ma_code, k_length)
            + Self::ma_extra_bars(ma_code, d_length)
    }

    fn is_valid_pair(source: f64, volume: f64) -> bool {
        source.is_finite() && volume.is_finite()
    }

    fn first_valid_pair(source: &[f64], volume: &[f64]) -> Option<usize> {
        source
            .iter()
            .zip(volume.iter())
            .position(|(&src, &vol)| Self::is_valid_pair(src, vol))
    }

    fn will_fit(
        required: usize,
        headroom: usize,
    ) -> Result<(), CudaVolumeWeightedStochasticRsiError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaVolumeWeightedStochasticRsiError::OutOfMemory {
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
    ) -> Result<(), CudaVolumeWeightedStochasticRsiError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaVolumeWeightedStochasticRsiError::LaunchConfigTooLarge {
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
        sweep: &VolumeWeightedStochasticRsiBatchRange,
    ) -> Result<CudaVolumeWeightedStochasticRsiBatchResult, CudaVolumeWeightedStochasticRsiError>
    {
        if source.is_empty() || volume.is_empty() {
            return Err(CudaVolumeWeightedStochasticRsiError::InvalidInput(
                "empty input".into(),
            ));
        }
        if source.len() != volume.len() {
            return Err(CudaVolumeWeightedStochasticRsiError::InvalidInput(
                "source and volume length mismatch".into(),
            ));
        }

        let first = Self::first_valid_pair(source, volume).ok_or_else(|| {
            CudaVolumeWeightedStochasticRsiError::InvalidInput("all values are NaN".into())
        })?;
        let valid = source.len() - first;

        let combos = expand_grid_volume_weighted_stochastic_rsi(sweep)
            .map_err(|err| CudaVolumeWeightedStochasticRsiError::InvalidInput(err.to_string()))?;
        if combos.is_empty() {
            return Err(CudaVolumeWeightedStochasticRsiError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let rows = combos.len();
        let cols = source.len();
        let mut rsi_lengths = Vec::with_capacity(rows);
        let mut stoch_lengths = Vec::with_capacity(rows);
        let mut k_lengths = Vec::with_capacity(rows);
        let mut d_lengths = Vec::with_capacity(rows);
        let mut ma_codes = Vec::with_capacity(rows);

        for combo in &combos {
            let rsi_length = combo.rsi_length.unwrap_or(DEFAULT_RSI_LENGTH);
            let stoch_length = combo.stoch_length.unwrap_or(DEFAULT_STOCH_LENGTH);
            let k_length = combo.k_length.unwrap_or(DEFAULT_K_LENGTH);
            let d_length = combo.d_length.unwrap_or(DEFAULT_D_LENGTH);
            if rsi_length == 0 || rsi_length > cols {
                return Err(CudaVolumeWeightedStochasticRsiError::InvalidInput(format!(
                    "invalid rsi_length: rsi_length={rsi_length}, data_len={cols}"
                )));
            }
            if stoch_length == 0 || stoch_length > cols {
                return Err(CudaVolumeWeightedStochasticRsiError::InvalidInput(format!(
                    "invalid stoch_length: stoch_length={stoch_length}, data_len={cols}"
                )));
            }
            if k_length == 0 || k_length > cols {
                return Err(CudaVolumeWeightedStochasticRsiError::InvalidInput(format!(
                    "invalid k_length: k_length={k_length}, data_len={cols}"
                )));
            }
            if d_length == 0 || d_length > cols {
                return Err(CudaVolumeWeightedStochasticRsiError::InvalidInput(format!(
                    "invalid d_length: d_length={d_length}, data_len={cols}"
                )));
            }

            let ma_code = Self::parse_ma_type(combo.ma_type.as_deref().unwrap_or("WSMA"))?;
            let needed = Self::needed_bars(ma_code, rsi_length, stoch_length, k_length, d_length);
            if valid < needed {
                return Err(CudaVolumeWeightedStochasticRsiError::InvalidInput(format!(
                    "not enough valid data: needed={needed}, valid={valid}"
                )));
            }

            rsi_lengths.push(rsi_length as i32);
            stoch_lengths.push(stoch_length as i32);
            k_lengths.push(k_length as i32);
            d_lengths.push(d_length as i32);
            ma_codes.push(ma_code);
        }

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                CudaVolumeWeightedStochasticRsiError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = rows
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| value.checked_mul(5))
            .ok_or_else(|| {
                CudaVolumeWeightedStochasticRsiError::InvalidInput("params bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaVolumeWeightedStochasticRsiError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                CudaVolumeWeightedStochasticRsiError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaVolumeWeightedStochasticRsiError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_source = DeviceBuffer::from_slice(source)?;
        let d_volume = DeviceBuffer::from_slice(volume)?;
        let d_rsi_lengths = DeviceBuffer::from_slice(&rsi_lengths)?;
        let d_stoch_lengths = DeviceBuffer::from_slice(&stoch_lengths)?;
        let d_k_lengths = DeviceBuffer::from_slice(&k_lengths)?;
        let d_d_lengths = DeviceBuffer::from_slice(&d_lengths)?;
        let d_ma_codes = DeviceBuffer::from_slice(&ma_codes)?;
        let d_out_k = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_d = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("volume_weighted_stochastic_rsi_batch_f64")
            .map_err(
                |_| CudaVolumeWeightedStochasticRsiError::MissingKernelSymbol {
                    name: "volume_weighted_stochastic_rsi_batch_f64",
                },
            )?;
        let grid_x = ((rows as u32) + VOLUME_WEIGHTED_STOCHASTIC_RSI_BLOCK_X - 1)
            / VOLUME_WEIGHTED_STOCHASTIC_RSI_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(VOLUME_WEIGHTED_STOCHASTIC_RSI_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_source.as_device_ptr(),
                d_volume.as_device_ptr(),
                cols as i32,
                d_rsi_lengths.as_device_ptr(),
                d_stoch_lengths.as_device_ptr(),
                d_k_lengths.as_device_ptr(),
                d_d_lengths.as_device_ptr(),
                d_ma_codes.as_device_ptr(),
                rows as i32,
                d_out_k.as_device_ptr(),
                d_out_d.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaVolumeWeightedStochasticRsiBatchResult {
            outputs: VolumeWeightedStochasticRsiDeviceArrayF64Pair {
                k: VolumeWeightedStochasticRsiDeviceArrayF64 {
                    buf: d_out_k,
                    rows,
                    cols,
                },
                d: VolumeWeightedStochasticRsiDeviceArrayF64 {
                    buf: d_out_d,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
