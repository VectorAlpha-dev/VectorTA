#![cfg(feature = "cuda")]

use crate::indicators::twiggs_money_flow::{
    expand_grid_twiggs_money_flow, TwiggsMoneyFlowBatchRange, TwiggsMoneyFlowParams,
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

const TWIGGS_MONEY_FLOW_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const DEFAULT_LENGTH: usize = 21;
const DEFAULT_SMOOTHING_LENGTH: usize = 14;
const DEFAULT_MA_TYPE: &str = "WMA";
const MA_SMA: i32 = 0;
const MA_EMA: i32 = 1;
const MA_WMA: i32 = 2;
const MA_VWMA: i32 = 3;

#[derive(Debug, Error)]
pub enum CudaTwiggsMoneyFlowError {
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

pub struct TwiggsMoneyFlowDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl TwiggsMoneyFlowDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct TwiggsMoneyFlowDeviceArrayF64Pair {
    pub tmf: TwiggsMoneyFlowDeviceArrayF64,
    pub smoothed: TwiggsMoneyFlowDeviceArrayF64,
}

impl TwiggsMoneyFlowDeviceArrayF64Pair {
    #[inline]
    pub fn rows(&self) -> usize {
        self.tmf.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.tmf.cols
    }
}

pub struct CudaTwiggsMoneyFlowBatchResult {
    pub outputs: TwiggsMoneyFlowDeviceArrayF64Pair,
    pub combos: Vec<TwiggsMoneyFlowParams>,
}

pub struct CudaTwiggsMoneyFlow {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaTwiggsMoneyFlow {
    pub fn new(device_id: usize) -> Result<Self, CudaTwiggsMoneyFlowError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("twiggs_money_flow_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaTwiggsMoneyFlowError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn is_valid_bar(high: f64, low: f64, close: f64, volume: f64) -> bool {
        high.is_finite()
            && low.is_finite()
            && close.is_finite()
            && volume.is_finite()
            && high >= low
    }

    fn first_valid_adv(high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Option<usize> {
        if close.len() < 2 {
            return None;
        }
        (1..close.len()).find(|&i| {
            Self::is_valid_bar(high[i], low[i], close[i], volume[i]) && close[i - 1].is_finite()
        })
    }

    fn parse_ma_type(value: &str) -> Result<i32, CudaTwiggsMoneyFlowError> {
        match value.trim().to_ascii_uppercase().as_str() {
            "SMA" => Ok(MA_SMA),
            "EMA" => Ok(MA_EMA),
            "WMA" => Ok(MA_WMA),
            "VWMA" => Ok(MA_VWMA),
            _ => Err(CudaTwiggsMoneyFlowError::InvalidInput(format!(
                "invalid ma_type: {value}"
            ))),
        }
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaTwiggsMoneyFlowError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaTwiggsMoneyFlowError::OutOfMemory {
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
    ) -> Result<(), CudaTwiggsMoneyFlowError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaTwiggsMoneyFlowError::LaunchConfigTooLarge {
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
        sweep: &TwiggsMoneyFlowBatchRange,
    ) -> Result<CudaTwiggsMoneyFlowBatchResult, CudaTwiggsMoneyFlowError> {
        if high.is_empty() || low.is_empty() || close.is_empty() || volume.is_empty() {
            return Err(CudaTwiggsMoneyFlowError::InvalidInput("empty input".into()));
        }
        if high.len() != low.len() || high.len() != close.len() || high.len() != volume.len() {
            return Err(CudaTwiggsMoneyFlowError::InvalidInput(format!(
                "input length mismatch: high={}, low={}, close={}, volume={}",
                high.len(),
                low.len(),
                close.len(),
                volume.len()
            )));
        }

        let first_adv = Self::first_valid_adv(high, low, close, volume)
            .ok_or_else(|| CudaTwiggsMoneyFlowError::InvalidInput("all values are NaN".into()))?;

        let combos = expand_grid_twiggs_money_flow(sweep)
            .map_err(|err| CudaTwiggsMoneyFlowError::InvalidInput(err.to_string()))?;
        if combos.is_empty() {
            return Err(CudaTwiggsMoneyFlowError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let rows = combos.len();
        let cols = close.len();
        let valid = cols - first_adv;
        let mut lengths = Vec::with_capacity(rows);
        let mut smoothing_lengths = Vec::with_capacity(rows);
        let mut ma_codes = Vec::with_capacity(rows);

        for combo in &combos {
            let length = combo.length.unwrap_or(DEFAULT_LENGTH);
            let smoothing_length = combo.smoothing_length.unwrap_or(DEFAULT_SMOOTHING_LENGTH);
            let ma_code = Self::parse_ma_type(combo.ma_type.as_deref().unwrap_or(DEFAULT_MA_TYPE))?;
            if length == 0 || length > cols {
                return Err(CudaTwiggsMoneyFlowError::InvalidInput(
                    "invalid length".into(),
                ));
            }
            if smoothing_length > cols {
                return Err(CudaTwiggsMoneyFlowError::InvalidInput(
                    "invalid smoothing_length".into(),
                ));
            }
            let needed = if ma_code == MA_EMA { 1 } else { length };
            if valid < needed {
                return Err(CudaTwiggsMoneyFlowError::InvalidInput(format!(
                    "not enough valid data: needed={needed}, valid={valid}"
                )));
            }
            lengths.push(length as i32);
            smoothing_lengths.push(smoothing_length as i32);
            ma_codes.push(ma_code);
        }

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| CudaTwiggsMoneyFlowError::InvalidInput("input bytes overflow".into()))?;
        let params_bytes = rows
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| value.checked_mul(3))
            .ok_or_else(|| {
                CudaTwiggsMoneyFlowError::InvalidInput("params bytes overflow".into())
            })?;
        let output_elems = rows
            .checked_mul(cols)
            .ok_or_else(|| CudaTwiggsMoneyFlowError::InvalidInput("rows*cols overflow".into()))?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                CudaTwiggsMoneyFlowError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaTwiggsMoneyFlowError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_close = DeviceBuffer::from_slice(close)?;
        let d_volume = DeviceBuffer::from_slice(volume)?;
        let d_lengths = DeviceBuffer::from_slice(&lengths)?;
        let d_smoothing_lengths = DeviceBuffer::from_slice(&smoothing_lengths)?;
        let d_ma_codes = DeviceBuffer::from_slice(&ma_codes)?;
        let d_out_tmf = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_smoothed = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("twiggs_money_flow_batch_f64")
            .map_err(|_| CudaTwiggsMoneyFlowError::MissingKernelSymbol {
                name: "twiggs_money_flow_batch_f64",
            })?;
        let grid_x = ((rows as u32) + TWIGGS_MONEY_FLOW_BLOCK_X - 1) / TWIGGS_MONEY_FLOW_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(TWIGGS_MONEY_FLOW_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_high.as_device_ptr(),
                d_low.as_device_ptr(),
                d_close.as_device_ptr(),
                d_volume.as_device_ptr(),
                cols as i32,
                d_lengths.as_device_ptr(),
                d_smoothing_lengths.as_device_ptr(),
                d_ma_codes.as_device_ptr(),
                rows as i32,
                d_out_tmf.as_device_ptr(),
                d_out_smoothed.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaTwiggsMoneyFlowBatchResult {
            outputs: TwiggsMoneyFlowDeviceArrayF64Pair {
                tmf: TwiggsMoneyFlowDeviceArrayF64 {
                    buf: d_out_tmf,
                    rows,
                    cols,
                },
                smoothed: TwiggsMoneyFlowDeviceArrayF64 {
                    buf: d_out_smoothed,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
