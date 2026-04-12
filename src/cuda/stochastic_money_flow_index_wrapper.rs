#![cfg(feature = "cuda")]

use crate::indicators::stochastic_money_flow_index::{
    StochasticMoneyFlowIndexBatchRange, StochasticMoneyFlowIndexParams,
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

const STOCHASTIC_MONEY_FLOW_INDEX_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaStochasticMoneyFlowIndexError {
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

pub struct StochasticMoneyFlowIndexDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl StochasticMoneyFlowIndexDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct StochasticMoneyFlowIndexDeviceArrayF64Pair {
    pub k: StochasticMoneyFlowIndexDeviceArrayF64,
    pub d: StochasticMoneyFlowIndexDeviceArrayF64,
}

impl StochasticMoneyFlowIndexDeviceArrayF64Pair {
    #[inline]
    pub fn rows(&self) -> usize {
        self.k.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.k.cols
    }
}

pub struct CudaStochasticMoneyFlowIndexBatchResult {
    pub outputs: StochasticMoneyFlowIndexDeviceArrayF64Pair,
    pub combos: Vec<StochasticMoneyFlowIndexParams>,
}

pub struct CudaStochasticMoneyFlowIndex {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaStochasticMoneyFlowIndex {
    pub fn new(device_id: usize) -> Result<Self, CudaStochasticMoneyFlowIndexError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("stochastic_money_flow_index_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaStochasticMoneyFlowIndexError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn longest_valid_run(source: &[f64], volume: &[f64]) -> usize {
        let mut best = 0usize;
        let mut cur = 0usize;
        for (&src, &vol) in source.iter().zip(volume.iter()) {
            if src.is_finite() && vol.is_finite() {
                cur += 1;
                best = best.max(cur);
            } else {
                cur = 0;
            }
        }
        best
    }

    fn required_bars_for_k(
        mfi_length: usize,
        stoch_k_length: usize,
        stoch_k_smooth: usize,
    ) -> usize {
        mfi_length
            .saturating_add(stoch_k_length)
            .saturating_add(stoch_k_smooth)
            .saturating_sub(2)
    }

    fn expand_axis(
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, CudaStochasticMoneyFlowIndexError> {
        if start == 0 {
            return Err(CudaStochasticMoneyFlowIndexError::InvalidInput(format!(
                "invalid range: start={start}, end={end}, step={step}"
            )));
        }
        if step == 0 {
            return Ok(vec![start]);
        }
        if start > end {
            return Err(CudaStochasticMoneyFlowIndexError::InvalidInput(format!(
                "invalid range: start={start}, end={end}, step={step}"
            )));
        }
        let mut values = Vec::new();
        let mut cur = start;
        loop {
            values.push(cur);
            if cur >= end {
                break;
            }
            let next = cur.saturating_add(step);
            if next <= cur {
                return Err(CudaStochasticMoneyFlowIndexError::InvalidInput(format!(
                    "invalid range: start={start}, end={end}, step={step}"
                )));
            }
            cur = next.min(end);
        }
        Ok(values)
    }

    fn expand_grid(
        range: &StochasticMoneyFlowIndexBatchRange,
    ) -> Result<Vec<StochasticMoneyFlowIndexParams>, CudaStochasticMoneyFlowIndexError> {
        let stoch_k_lengths = Self::expand_axis(range.stoch_k_length)?;
        let stoch_k_smooths = Self::expand_axis(range.stoch_k_smooth)?;
        let stoch_d_smooths = Self::expand_axis(range.stoch_d_smooth)?;
        let mfi_lengths = Self::expand_axis(range.mfi_length)?;

        let mut combos = Vec::new();
        for stoch_k_length in stoch_k_lengths {
            for &stoch_k_smooth in &stoch_k_smooths {
                for &stoch_d_smooth in &stoch_d_smooths {
                    for &mfi_length in &mfi_lengths {
                        combos.push(StochasticMoneyFlowIndexParams {
                            stoch_k_length: Some(stoch_k_length),
                            stoch_k_smooth: Some(stoch_k_smooth),
                            stoch_d_smooth: Some(stoch_d_smooth),
                            mfi_length: Some(mfi_length),
                        });
                    }
                }
            }
        }
        Ok(combos)
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaStochasticMoneyFlowIndexError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaStochasticMoneyFlowIndexError::OutOfMemory {
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
    ) -> Result<(), CudaStochasticMoneyFlowIndexError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaStochasticMoneyFlowIndexError::LaunchConfigTooLarge {
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
        sweep: &StochasticMoneyFlowIndexBatchRange,
    ) -> Result<CudaStochasticMoneyFlowIndexBatchResult, CudaStochasticMoneyFlowIndexError> {
        let len = source.len();
        if len == 0 || volume.is_empty() {
            return Err(CudaStochasticMoneyFlowIndexError::InvalidInput(
                "empty input".into(),
            ));
        }
        if len != volume.len() {
            return Err(CudaStochasticMoneyFlowIndexError::InvalidInput(format!(
                "input length mismatch: source={len}, volume={}",
                volume.len()
            )));
        }

        let combos = Self::expand_grid(sweep)?;
        let max_run = Self::longest_valid_run(source, volume);
        if max_run == 0 {
            return Err(CudaStochasticMoneyFlowIndexError::InvalidInput(
                "all values are NaN".into(),
            ));
        }

        let mut max_needed = 0usize;
        let mut max_flow_len = 0usize;
        let mut max_stoch_k_length = 0usize;
        let mut max_k_smooth = 0usize;
        let mut max_d_smooth = 0usize;
        for combo in &combos {
            let stoch_k_length = combo.stoch_k_length.unwrap_or(14);
            let stoch_k_smooth = combo.stoch_k_smooth.unwrap_or(3);
            let stoch_d_smooth = combo.stoch_d_smooth.unwrap_or(3);
            let mfi_length = combo.mfi_length.unwrap_or(14);
            max_needed = max_needed.max(Self::required_bars_for_k(
                mfi_length,
                stoch_k_length,
                stoch_k_smooth,
            ));
            max_flow_len = max_flow_len.max(mfi_length.saturating_sub(1));
            max_stoch_k_length = max_stoch_k_length.max(stoch_k_length);
            max_k_smooth = max_k_smooth.max(stoch_k_smooth);
            max_d_smooth = max_d_smooth.max(stoch_d_smooth);
        }
        if max_run < max_needed {
            return Err(CudaStochasticMoneyFlowIndexError::InvalidInput(format!(
                "not enough valid data: needed={max_needed}, valid={max_run}"
            )));
        }

        let rows = combos.len();
        let cols = len;
        let stoch_k_lengths: Vec<i32> = combos
            .iter()
            .map(|combo| combo.stoch_k_length.unwrap_or(14) as i32)
            .collect();
        let stoch_k_smooths: Vec<i32> = combos
            .iter()
            .map(|combo| combo.stoch_k_smooth.unwrap_or(3) as i32)
            .collect();
        let stoch_d_smooths: Vec<i32> = combos
            .iter()
            .map(|combo| combo.stoch_d_smooth.unwrap_or(3) as i32)
            .collect();
        let mfi_lengths: Vec<i32> = combos
            .iter()
            .map(|combo| combo.mfi_length.unwrap_or(14) as i32)
            .collect();

        let flow_cap = max_flow_len.max(1);
        let stoch_k_cap = max_stoch_k_length.max(1);
        let k_smooth_cap = max_k_smooth.max(1);
        let d_smooth_cap = max_d_smooth.max(1);

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                CudaStochasticMoneyFlowIndexError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = stoch_k_lengths
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| {
                stoch_k_smooths
                    .len()
                    .checked_mul(std::mem::size_of::<i32>())
                    .and_then(|other| value.checked_add(other))
            })
            .and_then(|value| {
                stoch_d_smooths
                    .len()
                    .checked_mul(std::mem::size_of::<i32>())
                    .and_then(|other| value.checked_add(other))
            })
            .and_then(|value| {
                mfi_lengths
                    .len()
                    .checked_mul(std::mem::size_of::<i32>())
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| {
                CudaStochasticMoneyFlowIndexError::InvalidInput("params bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaStochasticMoneyFlowIndexError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                CudaStochasticMoneyFlowIndexError::InvalidInput("output bytes overflow".into())
            })?;
        let flow_scratch = rows.checked_mul(flow_cap).ok_or_else(|| {
            CudaStochasticMoneyFlowIndexError::InvalidInput("rows*flow_cap overflow".into())
        })?;
        let stoch_scratch = rows.checked_mul(stoch_k_cap).ok_or_else(|| {
            CudaStochasticMoneyFlowIndexError::InvalidInput("rows*stoch_k_cap overflow".into())
        })?;
        let k_scratch = rows.checked_mul(k_smooth_cap).ok_or_else(|| {
            CudaStochasticMoneyFlowIndexError::InvalidInput("rows*k_smooth_cap overflow".into())
        })?;
        let d_scratch = rows.checked_mul(d_smooth_cap).ok_or_else(|| {
            CudaStochasticMoneyFlowIndexError::InvalidInput("rows*d_smooth_cap overflow".into())
        })?;
        let scratch_bytes = flow_scratch
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(2))
            .and_then(|value| {
                stoch_scratch
                    .checked_mul(std::mem::size_of::<i32>())
                    .and_then(|other| other.checked_mul(2))
                    .and_then(|other| value.checked_add(other))
            })
            .and_then(|value| {
                stoch_scratch
                    .checked_mul(std::mem::size_of::<f64>())
                    .and_then(|other| other.checked_mul(2))
                    .and_then(|other| value.checked_add(other))
            })
            .and_then(|value| {
                k_scratch
                    .checked_mul(std::mem::size_of::<f64>())
                    .and_then(|other| value.checked_add(other))
            })
            .and_then(|value| {
                d_scratch
                    .checked_mul(std::mem::size_of::<f64>())
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| {
                CudaStochasticMoneyFlowIndexError::InvalidInput("scratch bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .and_then(|value| value.checked_add(scratch_bytes))
            .ok_or_else(|| {
                CudaStochasticMoneyFlowIndexError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_source = DeviceBuffer::from_slice(source)?;
        let d_volume = DeviceBuffer::from_slice(volume)?;
        let d_stoch_k_lengths = DeviceBuffer::from_slice(&stoch_k_lengths)?;
        let d_stoch_k_smooths = DeviceBuffer::from_slice(&stoch_k_smooths)?;
        let d_stoch_d_smooths = DeviceBuffer::from_slice(&stoch_d_smooths)?;
        let d_mfi_lengths = DeviceBuffer::from_slice(&mfi_lengths)?;
        let d_pos_buf = unsafe { DeviceBuffer::<f64>::uninitialized(flow_scratch)? };
        let d_neg_buf = unsafe { DeviceBuffer::<f64>::uninitialized(flow_scratch)? };
        let d_maxdq_idx = unsafe { DeviceBuffer::<i32>::uninitialized(stoch_scratch)? };
        let d_maxdq_val = unsafe { DeviceBuffer::<f64>::uninitialized(stoch_scratch)? };
        let d_mindq_idx = unsafe { DeviceBuffer::<i32>::uninitialized(stoch_scratch)? };
        let d_mindq_val = unsafe { DeviceBuffer::<f64>::uninitialized(stoch_scratch)? };
        let d_k_buf = unsafe { DeviceBuffer::<f64>::uninitialized(k_scratch)? };
        let d_d_buf = unsafe { DeviceBuffer::<f64>::uninitialized(d_scratch)? };
        let d_out_k = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_d = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("stochastic_money_flow_index_batch_f64")
            .map_err(|_| CudaStochasticMoneyFlowIndexError::MissingKernelSymbol {
                name: "stochastic_money_flow_index_batch_f64",
            })?;
        let grid_x = ((rows as u32) + STOCHASTIC_MONEY_FLOW_INDEX_BLOCK_X - 1)
            / STOCHASTIC_MONEY_FLOW_INDEX_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(STOCHASTIC_MONEY_FLOW_INDEX_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_source.as_device_ptr(),
                d_volume.as_device_ptr(),
                cols as i32,
                d_stoch_k_lengths.as_device_ptr(),
                d_stoch_k_smooths.as_device_ptr(),
                d_stoch_d_smooths.as_device_ptr(),
                d_mfi_lengths.as_device_ptr(),
                rows as i32,
                flow_cap as i32,
                stoch_k_cap as i32,
                k_smooth_cap as i32,
                d_smooth_cap as i32,
                d_pos_buf.as_device_ptr(),
                d_neg_buf.as_device_ptr(),
                d_maxdq_idx.as_device_ptr(),
                d_maxdq_val.as_device_ptr(),
                d_mindq_idx.as_device_ptr(),
                d_mindq_val.as_device_ptr(),
                d_k_buf.as_device_ptr(),
                d_d_buf.as_device_ptr(),
                d_out_k.as_device_ptr(),
                d_out_d.as_device_ptr()
            ))?;
        }

        Ok(CudaStochasticMoneyFlowIndexBatchResult {
            outputs: StochasticMoneyFlowIndexDeviceArrayF64Pair {
                k: StochasticMoneyFlowIndexDeviceArrayF64 {
                    buf: d_out_k,
                    rows,
                    cols,
                },
                d: StochasticMoneyFlowIndexDeviceArrayF64 {
                    buf: d_out_d,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
