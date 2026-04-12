#![cfg(feature = "cuda")]

use crate::indicators::bull_power_vs_bear_power::{
    BullPowerVsBearPowerBatchRange, BullPowerVsBearPowerParams,
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

const BULL_POWER_VS_BEAR_POWER_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaBullPowerVsBearPowerError {
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

pub struct BullPowerVsBearPowerDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl BullPowerVsBearPowerDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct CudaBullPowerVsBearPowerBatchResult {
    pub outputs: BullPowerVsBearPowerDeviceArrayF64,
    pub combos: Vec<BullPowerVsBearPowerParams>,
}

pub struct CudaBullPowerVsBearPower {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaBullPowerVsBearPower {
    pub fn new(device_id: usize) -> Result<Self, CudaBullPowerVsBearPowerError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("bull_power_vs_bear_power_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaBullPowerVsBearPowerError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn expand_grid(
        range: &BullPowerVsBearPowerBatchRange,
    ) -> Result<Vec<BullPowerVsBearPowerParams>, CudaBullPowerVsBearPowerError> {
        let (start, end, step) = range.period;
        let periods = if step == 0 || start == end {
            vec![start]
        } else {
            let step = step.max(1);
            if start < end {
                let mut out = Vec::new();
                let mut x = start;
                while x <= end {
                    out.push(x);
                    match x.checked_add(step) {
                        Some(next) if next != x => x = next,
                        _ => break,
                    }
                }
                out
            } else {
                let mut out = Vec::new();
                let mut x = start;
                loop {
                    out.push(x);
                    if x == end {
                        break;
                    }
                    let next = x.saturating_sub(step);
                    if next == x || next < end {
                        break;
                    }
                    x = next;
                }
                out
            }
        };

        if periods.is_empty() {
            return Err(CudaBullPowerVsBearPowerError::InvalidInput(format!(
                "invalid period range: start={start}, end={end}, step={step}"
            )));
        }
        if let Some(&period) = periods.iter().find(|&&period| period == 0) {
            return Err(CudaBullPowerVsBearPowerError::InvalidInput(format!(
                "invalid period: period={period}"
            )));
        }

        Ok(periods
            .into_iter()
            .map(|period| BullPowerVsBearPowerParams {
                period: Some(period),
            })
            .collect())
    }

    fn valid_ohlc_bar(open: f64, high: f64, low: f64, close: f64) -> bool {
        open.is_finite() && high.is_finite() && low.is_finite() && close.is_finite() && close != 0.0
    }

    fn first_valid_ohlc(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Option<usize> {
        (0..close.len()).find(|&i| Self::valid_ohlc_bar(open[i], high[i], low[i], close[i]))
    }

    fn count_valid_ohlc(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> usize {
        let mut count = 0usize;
        for i in 0..close.len() {
            if Self::valid_ohlc_bar(open[i], high[i], low[i], close[i]) {
                count += 1;
            }
        }
        count
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaBullPowerVsBearPowerError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaBullPowerVsBearPowerError::OutOfMemory {
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
    ) -> Result<(), CudaBullPowerVsBearPowerError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaBullPowerVsBearPowerError::LaunchConfigTooLarge {
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
        sweep: &BullPowerVsBearPowerBatchRange,
    ) -> Result<CudaBullPowerVsBearPowerBatchResult, CudaBullPowerVsBearPowerError> {
        let len = close.len();
        if len == 0 {
            return Err(CudaBullPowerVsBearPowerError::InvalidInput(
                "empty input".into(),
            ));
        }
        if open.len() != len || high.len() != len || low.len() != len {
            return Err(CudaBullPowerVsBearPowerError::InvalidInput(format!(
                "inconsistent slice lengths: open={}, high={}, low={}, close={}",
                open.len(),
                high.len(),
                low.len(),
                close.len()
            )));
        }

        let combos = Self::expand_grid(sweep)?;
        Self::first_valid_ohlc(open, high, low, close).ok_or_else(|| {
            CudaBullPowerVsBearPowerError::InvalidInput(
                "all values are NaN or have zero close".into(),
            )
        })?;
        let max_period = combos
            .iter()
            .map(|combo| combo.period.unwrap_or(5))
            .max()
            .unwrap_or(0);
        if max_period == 0 || max_period > len {
            return Err(CudaBullPowerVsBearPowerError::InvalidInput(format!(
                "invalid period: period={max_period}, data_len={len}"
            )));
        }
        let valid = Self::count_valid_ohlc(open, high, low, close);
        if valid < max_period {
            return Err(CudaBullPowerVsBearPowerError::InvalidInput(format!(
                "not enough valid data: needed={max_period}, valid={valid}"
            )));
        }

        let rows = combos.len();
        let cols = len;
        let periods: Vec<i32> = combos
            .iter()
            .map(|combo| combo.period.unwrap_or(5) as i32)
            .collect();

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|v| v.checked_mul(4))
            .ok_or_else(|| {
                CudaBullPowerVsBearPowerError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = periods
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| {
                CudaBullPowerVsBearPowerError::InvalidInput("params bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaBullPowerVsBearPowerError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaBullPowerVsBearPowerError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|v| v.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaBullPowerVsBearPowerError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_open = DeviceBuffer::from_slice(open)?;
        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_close = DeviceBuffer::from_slice(close)?;
        let d_periods = DeviceBuffer::from_slice(&periods)?;
        let mut d_out = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("bull_power_vs_bear_power_batch_f64")
            .map_err(|_| CudaBullPowerVsBearPowerError::MissingKernelSymbol {
                name: "bull_power_vs_bear_power_batch_f64",
            })?;
        let grid_x = ((rows as u32) + BULL_POWER_VS_BEAR_POWER_BLOCK_X - 1)
            / BULL_POWER_VS_BEAR_POWER_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(BULL_POWER_VS_BEAR_POWER_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_open.as_device_ptr(),
                d_high.as_device_ptr(),
                d_low.as_device_ptr(),
                d_close.as_device_ptr(),
                cols as i32,
                d_periods.as_device_ptr(),
                rows as i32,
                d_out.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaBullPowerVsBearPowerBatchResult {
            outputs: BullPowerVsBearPowerDeviceArrayF64 {
                buf: d_out,
                rows,
                cols,
            },
            combos,
        })
    }
}
