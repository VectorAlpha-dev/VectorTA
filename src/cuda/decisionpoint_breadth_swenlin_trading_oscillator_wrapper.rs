#![cfg(feature = "cuda")]

use crate::indicators::decisionpoint_breadth_swenlin_trading_oscillator::{
    DecisionPointBreadthSwenlinTradingOscillatorBatchRange,
    DecisionPointBreadthSwenlinTradingOscillatorParams,
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

const DECISIONPOINT_BREADTH_SWENLIN_TRADING_OSCILLATOR_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const SMA_LENGTH: usize = 5;
const EPSILON: f64 = 1e-12;

#[derive(Debug, Error)]
pub enum CudaDecisionPointBreadthSwenlinTradingOscillatorError {
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

pub struct DecisionPointBreadthSwenlinTradingOscillatorDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl DecisionPointBreadthSwenlinTradingOscillatorDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct CudaDecisionPointBreadthSwenlinTradingOscillatorBatchResult {
    pub outputs: DecisionPointBreadthSwenlinTradingOscillatorDeviceArrayF64,
    pub combos: Vec<DecisionPointBreadthSwenlinTradingOscillatorParams>,
}

pub struct CudaDecisionPointBreadthSwenlinTradingOscillator {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaDecisionPointBreadthSwenlinTradingOscillator {
    pub fn new(
        device_id: usize,
    ) -> Result<Self, CudaDecisionPointBreadthSwenlinTradingOscillatorError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!(
            "decisionpoint_breadth_swenlin_trading_oscillator_kernel"
        )?;
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

    pub fn synchronize(&self) -> Result<(), CudaDecisionPointBreadthSwenlinTradingOscillatorError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn valid_breadth_pair(advancing: f64, declining: f64) -> bool {
        if !advancing.is_finite() || !declining.is_finite() {
            return false;
        }
        let total = advancing + declining;
        total.is_finite() && total.abs() > EPSILON
    }

    fn first_valid_pair(advancing: &[f64], declining: &[f64]) -> Option<usize> {
        (0..advancing.len()).find(|&i| Self::valid_breadth_pair(advancing[i], declining[i]))
    }

    fn count_valid_pairs(advancing: &[f64], declining: &[f64]) -> usize {
        (0..advancing.len())
            .filter(|&i| Self::valid_breadth_pair(advancing[i], declining[i]))
            .count()
    }

    fn expand_grid(
        _sweep: &DecisionPointBreadthSwenlinTradingOscillatorBatchRange,
    ) -> Vec<DecisionPointBreadthSwenlinTradingOscillatorParams> {
        vec![DecisionPointBreadthSwenlinTradingOscillatorParams]
    }

    fn will_fit(
        required: usize,
        headroom: usize,
    ) -> Result<(), CudaDecisionPointBreadthSwenlinTradingOscillatorError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(
                    CudaDecisionPointBreadthSwenlinTradingOscillatorError::OutOfMemory {
                        required,
                        free,
                        headroom,
                    },
                );
            }
        }
        Ok(())
    }

    fn validate_launch(
        &self,
        grid: GridSize,
        block: BlockSize,
    ) -> Result<(), CudaDecisionPointBreadthSwenlinTradingOscillatorError> {
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
                CudaDecisionPointBreadthSwenlinTradingOscillatorError::LaunchConfigTooLarge {
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
        advancing: &[f64],
        declining: &[f64],
        sweep: &DecisionPointBreadthSwenlinTradingOscillatorBatchRange,
    ) -> Result<
        CudaDecisionPointBreadthSwenlinTradingOscillatorBatchResult,
        CudaDecisionPointBreadthSwenlinTradingOscillatorError,
    > {
        if advancing.len() != declining.len() {
            return Err(
                CudaDecisionPointBreadthSwenlinTradingOscillatorError::InvalidInput(format!(
                    "input length mismatch: advancing={}, declining={}",
                    advancing.len(),
                    declining.len()
                )),
            );
        }
        if advancing.is_empty() {
            return Err(
                CudaDecisionPointBreadthSwenlinTradingOscillatorError::InvalidInput(
                    "empty input".into(),
                ),
            );
        }

        Self::first_valid_pair(advancing, declining).ok_or_else(|| {
            CudaDecisionPointBreadthSwenlinTradingOscillatorError::InvalidInput(
                "all values are NaN".into(),
            )
        })?;
        let valid = Self::count_valid_pairs(advancing, declining);
        if valid < SMA_LENGTH {
            return Err(
                CudaDecisionPointBreadthSwenlinTradingOscillatorError::InvalidInput(format!(
                    "not enough valid data: needed={SMA_LENGTH}, valid={valid}"
                )),
            );
        }

        let combos = Self::expand_grid(sweep);
        let rows = combos.len();
        let cols = advancing.len();
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaDecisionPointBreadthSwenlinTradingOscillatorError::InvalidInput(
                "rows*cols overflow".into(),
            )
        })?;
        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                CudaDecisionPointBreadthSwenlinTradingOscillatorError::InvalidInput(
                    "input bytes overflow".into(),
                )
            })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaDecisionPointBreadthSwenlinTradingOscillatorError::InvalidInput(
                    "output bytes overflow".into(),
                )
            })?;
        let required = input_bytes.checked_add(output_bytes).ok_or_else(|| {
            CudaDecisionPointBreadthSwenlinTradingOscillatorError::InvalidInput(
                "required bytes overflow".into(),
            )
        })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_advancing = DeviceBuffer::from_slice(advancing)?;
        let d_declining = DeviceBuffer::from_slice(declining)?;
        let d_out = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("decisionpoint_breadth_swenlin_trading_oscillator_batch_f64")
            .map_err(|_| {
                CudaDecisionPointBreadthSwenlinTradingOscillatorError::MissingKernelSymbol {
                    name: "decisionpoint_breadth_swenlin_trading_oscillator_batch_f64",
                }
            })?;
        let grid_x = ((rows as u32) + DECISIONPOINT_BREADTH_SWENLIN_TRADING_OSCILLATOR_BLOCK_X - 1)
            / DECISIONPOINT_BREADTH_SWENLIN_TRADING_OSCILLATOR_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(DECISIONPOINT_BREADTH_SWENLIN_TRADING_OSCILLATOR_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_advancing.as_device_ptr(),
                d_declining.as_device_ptr(),
                cols as i32,
                rows as i32,
                d_out.as_device_ptr()
            ))?;
        }

        Ok(
            CudaDecisionPointBreadthSwenlinTradingOscillatorBatchResult {
                outputs: DecisionPointBreadthSwenlinTradingOscillatorDeviceArrayF64 {
                    buf: d_out,
                    rows,
                    cols,
                },
                combos,
            },
        )
    }
}
