#![cfg(feature = "cuda")]

use crate::indicators::stochastic_connors_rsi::{
    expand_grid_stochastic_connors_rsi, StochasticConnorsRsiBatchRange, StochasticConnorsRsiParams,
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

const STOCHASTIC_CONNORS_RSI_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const DEFAULT_STOCH_LENGTH: usize = 3;
const DEFAULT_SMOOTH_K: usize = 3;
const DEFAULT_SMOOTH_D: usize = 3;
const DEFAULT_RSI_LENGTH: usize = 3;
const DEFAULT_UPDOWN_LENGTH: usize = 2;
const DEFAULT_ROC_LENGTH: usize = 100;

#[derive(Debug, Error)]
pub enum CudaStochasticConnorsRsiError {
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

pub struct StochasticConnorsRsiDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl StochasticConnorsRsiDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct StochasticConnorsRsiDeviceArrayF64Pair {
    pub k: StochasticConnorsRsiDeviceArrayF64,
    pub d: StochasticConnorsRsiDeviceArrayF64,
}

impl StochasticConnorsRsiDeviceArrayF64Pair {
    #[inline]
    pub fn rows(&self) -> usize {
        self.k.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.k.cols
    }
}

pub struct CudaStochasticConnorsRsiBatchResult {
    pub outputs: StochasticConnorsRsiDeviceArrayF64Pair,
    pub combos: Vec<StochasticConnorsRsiParams>,
}

pub struct CudaStochasticConnorsRsi {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaStochasticConnorsRsi {
    pub fn new(device_id: usize) -> Result<Self, CudaStochasticConnorsRsiError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("stochastic_connors_rsi_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaStochasticConnorsRsiError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn first_valid_value(data: &[f64]) -> Option<usize> {
        data.iter().position(|value| value.is_finite())
    }

    fn longest_valid_run(data: &[f64]) -> usize {
        let mut best = 0usize;
        let mut current = 0usize;
        for &value in data {
            if value.is_finite() {
                current += 1;
                if current > best {
                    best = current;
                }
            } else {
                current = 0;
            }
        }
        best
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaStochasticConnorsRsiError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaStochasticConnorsRsiError::OutOfMemory {
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
    ) -> Result<(), CudaStochasticConnorsRsiError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaStochasticConnorsRsiError::LaunchConfigTooLarge {
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
        data: &[f64],
        sweep: &StochasticConnorsRsiBatchRange,
    ) -> Result<CudaStochasticConnorsRsiBatchResult, CudaStochasticConnorsRsiError> {
        if data.is_empty() {
            return Err(CudaStochasticConnorsRsiError::InvalidInput(
                "empty input".into(),
            ));
        }
        let first = Self::first_valid_value(data).ok_or_else(|| {
            CudaStochasticConnorsRsiError::InvalidInput("all values are NaN".into())
        })?;

        let combos = expand_grid_stochastic_connors_rsi(sweep)
            .map_err(|err| CudaStochasticConnorsRsiError::InvalidInput(err.to_string()))?;
        if combos.is_empty() {
            return Err(CudaStochasticConnorsRsiError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let rows = combos.len();
        let cols = data.len();
        let valid = Self::longest_valid_run(&data[first..]);
        let mut stoch_lengths = Vec::with_capacity(rows);
        let mut smooth_ks = Vec::with_capacity(rows);
        let mut smooth_ds = Vec::with_capacity(rows);
        let mut rsi_lengths = Vec::with_capacity(rows);
        let mut updown_lengths = Vec::with_capacity(rows);
        let mut roc_lengths = Vec::with_capacity(rows);

        for combo in &combos {
            let stoch_length = combo.stoch_length.unwrap_or(DEFAULT_STOCH_LENGTH);
            let smooth_k = combo.smooth_k.unwrap_or(DEFAULT_SMOOTH_K);
            let smooth_d = combo.smooth_d.unwrap_or(DEFAULT_SMOOTH_D);
            let rsi_length = combo.rsi_length.unwrap_or(DEFAULT_RSI_LENGTH);
            let updown_length = combo.updown_length.unwrap_or(DEFAULT_UPDOWN_LENGTH);
            let roc_length = combo.roc_length.unwrap_or(DEFAULT_ROC_LENGTH);

            if stoch_length == 0
                || stoch_length > cols
                || smooth_k == 0
                || smooth_k > cols
                || smooth_d == 0
                || smooth_d > cols
                || rsi_length == 0
                || rsi_length > cols
                || updown_length == 0
                || updown_length > cols
                || roc_length == 0
                || roc_length > cols
            {
                return Err(CudaStochasticConnorsRsiError::InvalidInput(
                    "invalid parameters".into(),
                ));
            }

            let max_component_len = rsi_length.max(updown_length).max(roc_length);
            let needed = max_component_len
                .checked_add(stoch_length)
                .and_then(|value| value.checked_add(smooth_k))
                .and_then(|value| value.checked_add(smooth_d))
                .and_then(|value| value.checked_sub(1))
                .ok_or_else(|| {
                    CudaStochasticConnorsRsiError::InvalidInput("needed bars overflow".into())
                })?;
            if valid < needed {
                return Err(CudaStochasticConnorsRsiError::InvalidInput(format!(
                    "not enough valid data: needed={needed}, valid={valid}"
                )));
            }

            stoch_lengths.push(stoch_length as i32);
            smooth_ks.push(smooth_k as i32);
            smooth_ds.push(smooth_d as i32);
            rsi_lengths.push(rsi_length as i32);
            updown_lengths.push(updown_length as i32);
            roc_lengths.push(roc_length as i32);
        }

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaStochasticConnorsRsiError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = rows
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| value.checked_mul(6))
            .ok_or_else(|| {
                CudaStochasticConnorsRsiError::InvalidInput("params bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaStochasticConnorsRsiError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                CudaStochasticConnorsRsiError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaStochasticConnorsRsiError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_data = DeviceBuffer::from_slice(data)?;
        let d_stoch_lengths = DeviceBuffer::from_slice(&stoch_lengths)?;
        let d_smooth_ks = DeviceBuffer::from_slice(&smooth_ks)?;
        let d_smooth_ds = DeviceBuffer::from_slice(&smooth_ds)?;
        let d_rsi_lengths = DeviceBuffer::from_slice(&rsi_lengths)?;
        let d_updown_lengths = DeviceBuffer::from_slice(&updown_lengths)?;
        let d_roc_lengths = DeviceBuffer::from_slice(&roc_lengths)?;
        let d_out_k = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_d = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("stochastic_connors_rsi_batch_f64")
            .map_err(|_| CudaStochasticConnorsRsiError::MissingKernelSymbol {
                name: "stochastic_connors_rsi_batch_f64",
            })?;
        let grid_x =
            ((rows as u32) + STOCHASTIC_CONNORS_RSI_BLOCK_X - 1) / STOCHASTIC_CONNORS_RSI_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(STOCHASTIC_CONNORS_RSI_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_data.as_device_ptr(),
                cols as i32,
                d_stoch_lengths.as_device_ptr(),
                d_smooth_ks.as_device_ptr(),
                d_smooth_ds.as_device_ptr(),
                d_rsi_lengths.as_device_ptr(),
                d_updown_lengths.as_device_ptr(),
                d_roc_lengths.as_device_ptr(),
                rows as i32,
                d_out_k.as_device_ptr(),
                d_out_d.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaStochasticConnorsRsiBatchResult {
            outputs: StochasticConnorsRsiDeviceArrayF64Pair {
                k: StochasticConnorsRsiDeviceArrayF64 {
                    buf: d_out_k,
                    rows,
                    cols,
                },
                d: StochasticConnorsRsiDeviceArrayF64 {
                    buf: d_out_d,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
