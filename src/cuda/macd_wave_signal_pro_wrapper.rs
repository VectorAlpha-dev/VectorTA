#![cfg(feature = "cuda")]

use crate::indicators::macd_wave_signal_pro::{
    MacdWaveSignalProBatchRange, MacdWaveSignalProParams,
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

const MACD_WAVE_SIGNAL_PRO_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const DIFF_SLOW_PERIOD: usize = 26;
const DEA_PERIOD: usize = 9;
const LINE_LONG_PERIOD: usize = 40;

#[derive(Debug, Error)]
pub enum CudaMacdWaveSignalProError {
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

pub struct MacdWaveSignalProDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl MacdWaveSignalProDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct MacdWaveSignalProDeviceArrayF64Hex {
    pub diff: MacdWaveSignalProDeviceArrayF64,
    pub dea: MacdWaveSignalProDeviceArrayF64,
    pub macd_histogram: MacdWaveSignalProDeviceArrayF64,
    pub line_convergence: MacdWaveSignalProDeviceArrayF64,
    pub buy_signal: MacdWaveSignalProDeviceArrayF64,
    pub sell_signal: MacdWaveSignalProDeviceArrayF64,
}

impl MacdWaveSignalProDeviceArrayF64Hex {
    #[inline]
    pub fn rows(&self) -> usize {
        self.diff.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.diff.cols
    }
}

pub struct CudaMacdWaveSignalProBatchResult {
    pub outputs: MacdWaveSignalProDeviceArrayF64Hex,
    pub combos: Vec<MacdWaveSignalProParams>,
}

pub struct CudaMacdWaveSignalPro {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaMacdWaveSignalPro {
    pub fn new(device_id: usize) -> Result<Self, CudaMacdWaveSignalProError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("macd_wave_signal_pro_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaMacdWaveSignalProError> {
        self.stream.synchronize()?;
        Ok(())
    }

    #[inline]
    fn valid_ohlc(open: f64, high: f64, low: f64, close: f64) -> bool {
        open.is_finite() && high.is_finite() && low.is_finite() && close.is_finite()
    }

    fn first_valid_ohlc(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Option<usize> {
        (0..close.len()).find(|&i| Self::valid_ohlc(open[i], high[i], low[i], close[i]))
    }

    fn count_valid_ohlc(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> usize {
        (0..close.len())
            .filter(|&i| Self::valid_ohlc(open[i], high[i], low[i], close[i]))
            .count()
    }

    fn max_required_valid() -> usize {
        LINE_LONG_PERIOD.max(DIFF_SLOW_PERIOD + DEA_PERIOD - 1)
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaMacdWaveSignalProError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaMacdWaveSignalProError::OutOfMemory {
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
    ) -> Result<(), CudaMacdWaveSignalProError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaMacdWaveSignalProError::LaunchConfigTooLarge {
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
        _sweep: &MacdWaveSignalProBatchRange,
    ) -> Result<CudaMacdWaveSignalProBatchResult, CudaMacdWaveSignalProError> {
        if open.is_empty() || high.is_empty() || low.is_empty() || close.is_empty() {
            return Err(CudaMacdWaveSignalProError::InvalidInput(
                "empty input".into(),
            ));
        }
        if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
            return Err(CudaMacdWaveSignalProError::InvalidInput(format!(
                "input length mismatch: open={}, high={}, low={}, close={}",
                open.len(),
                high.len(),
                low.len(),
                close.len()
            )));
        }

        let _first = Self::first_valid_ohlc(open, high, low, close)
            .ok_or_else(|| CudaMacdWaveSignalProError::InvalidInput("all values are NaN".into()))?;
        let valid = Self::count_valid_ohlc(open, high, low, close);
        let needed = Self::max_required_valid();
        if valid < needed {
            return Err(CudaMacdWaveSignalProError::InvalidInput(format!(
                "not enough valid data: needed={needed}, valid={valid}"
            )));
        }

        let rows = 1usize;
        let cols = close.len();
        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| {
                CudaMacdWaveSignalProError::InvalidInput("input bytes overflow".into())
            })?;
        let output_elems = rows
            .checked_mul(cols)
            .ok_or_else(|| CudaMacdWaveSignalProError::InvalidInput("rows*cols overflow".into()))?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(6))
            .ok_or_else(|| {
                CudaMacdWaveSignalProError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes.checked_add(output_bytes).ok_or_else(|| {
            CudaMacdWaveSignalProError::InvalidInput("required bytes overflow".into())
        })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_open = DeviceBuffer::from_slice(open)?;
        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_close = DeviceBuffer::from_slice(close)?;
        let mut d_diff = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_dea = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_macd_histogram = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_line_convergence = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_buy_signal = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_sell_signal = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("macd_wave_signal_pro_batch_f64")
            .map_err(|_| CudaMacdWaveSignalProError::MissingKernelSymbol {
                name: "macd_wave_signal_pro_batch_f64",
            })?;
        let grid_x =
            ((rows as u32) + MACD_WAVE_SIGNAL_PRO_BLOCK_X - 1) / MACD_WAVE_SIGNAL_PRO_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(MACD_WAVE_SIGNAL_PRO_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_open.as_device_ptr(),
                d_high.as_device_ptr(),
                d_low.as_device_ptr(),
                d_close.as_device_ptr(),
                cols as i32,
                rows as i32,
                d_diff.as_device_ptr(),
                d_dea.as_device_ptr(),
                d_macd_histogram.as_device_ptr(),
                d_line_convergence.as_device_ptr(),
                d_buy_signal.as_device_ptr(),
                d_sell_signal.as_device_ptr()
            ))?;
        }
        self.stream.synchronize()?;

        Ok(CudaMacdWaveSignalProBatchResult {
            outputs: MacdWaveSignalProDeviceArrayF64Hex {
                diff: MacdWaveSignalProDeviceArrayF64 {
                    buf: d_diff,
                    rows,
                    cols,
                },
                dea: MacdWaveSignalProDeviceArrayF64 {
                    buf: d_dea,
                    rows,
                    cols,
                },
                macd_histogram: MacdWaveSignalProDeviceArrayF64 {
                    buf: d_macd_histogram,
                    rows,
                    cols,
                },
                line_convergence: MacdWaveSignalProDeviceArrayF64 {
                    buf: d_line_convergence,
                    rows,
                    cols,
                },
                buy_signal: MacdWaveSignalProDeviceArrayF64 {
                    buf: d_buy_signal,
                    rows,
                    cols,
                },
                sell_signal: MacdWaveSignalProDeviceArrayF64 {
                    buf: d_sell_signal,
                    rows,
                    cols,
                },
            },
            combos: vec![MacdWaveSignalProParams],
        })
    }
}
