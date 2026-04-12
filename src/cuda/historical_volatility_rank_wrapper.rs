#![cfg(feature = "cuda")]

use crate::indicators::historical_volatility_rank::{
    HistoricalVolatilityRankBatchRange, HistoricalVolatilityRankParams,
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

const HISTORICAL_VOLATILITY_RANK_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaHistoricalVolatilityRankError {
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

pub struct DeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl DeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct HistoricalVolatilityRankDeviceArrayF64Pair {
    pub hvr: DeviceArrayF64,
    pub hv: DeviceArrayF64,
}

impl HistoricalVolatilityRankDeviceArrayF64Pair {
    #[inline]
    pub fn rows(&self) -> usize {
        self.hvr.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.hvr.cols
    }
}

pub struct CudaHistoricalVolatilityRankBatchResult {
    pub outputs: HistoricalVolatilityRankDeviceArrayF64Pair,
    pub combos: Vec<HistoricalVolatilityRankParams>,
}

pub struct CudaHistoricalVolatilityRank {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaHistoricalVolatilityRank {
    pub fn new(device_id: usize) -> Result<Self, CudaHistoricalVolatilityRankError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("historical_volatility_rank_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaHistoricalVolatilityRankError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn axis_usize(
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, CudaHistoricalVolatilityRankError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }

        let mut out = Vec::new();
        if start < end {
            let mut cur = start;
            while cur <= end {
                out.push(cur);
                let next = cur.saturating_add(step.max(1));
                if next == cur {
                    break;
                }
                cur = next;
            }
        } else {
            let mut cur = start;
            loop {
                out.push(cur);
                if cur == end {
                    break;
                }
                let next = cur.saturating_sub(step.max(1));
                if next == cur || next < end {
                    break;
                }
                cur = next;
            }
        }

        if out.is_empty() {
            return Err(CudaHistoricalVolatilityRankError::InvalidInput(format!(
                "invalid usize range: start={start}, end={end}, step={step}"
            )));
        }
        Ok(out)
    }

    fn axis_f64(
        (start, end, step): (f64, f64, f64),
    ) -> Result<Vec<f64>, CudaHistoricalVolatilityRankError> {
        if !start.is_finite() || !end.is_finite() || !step.is_finite() {
            return Err(CudaHistoricalVolatilityRankError::InvalidInput(format!(
                "invalid float range: start={start}, end={end}, step={step}"
            )));
        }
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return Ok(vec![start]);
        }

        let mut out = Vec::new();
        if start < end {
            let delta = step.abs();
            let mut cur = start;
            while cur <= end + 1e-12 {
                out.push(cur);
                cur += delta;
            }
        } else {
            let delta = step.abs();
            let mut cur = start;
            while cur >= end - 1e-12 {
                out.push(cur);
                cur -= delta;
            }
        }

        if out.is_empty() {
            return Err(CudaHistoricalVolatilityRankError::InvalidInput(format!(
                "invalid float range: start={start}, end={end}, step={step}"
            )));
        }
        Ok(out)
    }

    fn expand_grid(
        sweep: &HistoricalVolatilityRankBatchRange,
    ) -> Result<Vec<HistoricalVolatilityRankParams>, CudaHistoricalVolatilityRankError> {
        let hv_lengths = Self::axis_usize(sweep.hv_length)?;
        let rank_lengths = Self::axis_usize(sweep.rank_length)?;
        let annualization_days = Self::axis_f64(sweep.annualization_days)?;
        let bar_days = Self::axis_f64(sweep.bar_days)?;

        let mut combos = Vec::with_capacity(
            hv_lengths
                .len()
                .saturating_mul(rank_lengths.len())
                .saturating_mul(annualization_days.len())
                .saturating_mul(bar_days.len()),
        );

        for hv_length in hv_lengths {
            if hv_length == 0 {
                return Err(CudaHistoricalVolatilityRankError::InvalidInput(
                    "hv_length must be > 0".into(),
                ));
            }
            for rank_length in &rank_lengths {
                if *rank_length == 0 {
                    return Err(CudaHistoricalVolatilityRankError::InvalidInput(
                        "rank_length must be > 0".into(),
                    ));
                }
                for annualization_day in &annualization_days {
                    if !annualization_day.is_finite() || *annualization_day <= 0.0 {
                        return Err(CudaHistoricalVolatilityRankError::InvalidInput(format!(
                            "invalid annualization_days {annualization_day}"
                        )));
                    }
                    for bar_day in &bar_days {
                        if !bar_day.is_finite() || *bar_day <= 0.0 {
                            return Err(CudaHistoricalVolatilityRankError::InvalidInput(format!(
                                "invalid bar_days {bar_day}"
                            )));
                        }
                        combos.push(HistoricalVolatilityRankParams {
                            hv_length: Some(hv_length),
                            rank_length: Some(*rank_length),
                            annualization_days: Some(*annualization_day),
                            bar_days: Some(*bar_day),
                        });
                    }
                }
            }
        }

        Ok(combos)
    }

    fn longest_valid_run(data: &[f64]) -> usize {
        let mut best = 0usize;
        let mut cur = 0usize;
        for &value in data {
            if value.is_finite() && value > 0.0 {
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

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaHistoricalVolatilityRankError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaHistoricalVolatilityRankError::OutOfMemory {
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
    ) -> Result<(), CudaHistoricalVolatilityRankError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaHistoricalVolatilityRankError::LaunchConfigTooLarge {
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
        sweep: &HistoricalVolatilityRankBatchRange,
    ) -> Result<CudaHistoricalVolatilityRankBatchResult, CudaHistoricalVolatilityRankError> {
        if data.is_empty() {
            return Err(CudaHistoricalVolatilityRankError::InvalidInput(
                "empty data".into(),
            ));
        }

        let combos = Self::expand_grid(sweep)?;
        let max_hv_length = combos
            .iter()
            .map(|combo| combo.hv_length.unwrap_or(10))
            .max()
            .unwrap_or(0);
        if max_hv_length >= data.len() {
            return Err(CudaHistoricalVolatilityRankError::InvalidInput(format!(
                "invalid hv_length: hv_length={max_hv_length}, data_len={}",
                data.len()
            )));
        }

        let longest_run = Self::longest_valid_run(data);
        if longest_run == 0 {
            return Err(CudaHistoricalVolatilityRankError::InvalidInput(
                "all values are NaN or non-positive".into(),
            ));
        }
        if longest_run <= max_hv_length {
            return Err(CudaHistoricalVolatilityRankError::InvalidInput(format!(
                "not enough valid data: needed={}, valid={longest_run}",
                max_hv_length + 1
            )));
        }

        let rows = combos.len();
        let cols = data.len();
        let hv_lengths: Vec<i32> = combos
            .iter()
            .map(|combo| combo.hv_length.unwrap_or(10) as i32)
            .collect();
        let rank_lengths: Vec<i32> = combos
            .iter()
            .map(|combo| combo.rank_length.unwrap_or(52 * 7) as i32)
            .collect();
        let annualization_scales: Vec<f64> = combos
            .iter()
            .map(|combo| {
                let annualization = combo.annualization_days.unwrap_or(365.0);
                let bar_days = combo.bar_days.unwrap_or(1.0);
                (annualization / bar_days).sqrt()
            })
            .collect();

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaHistoricalVolatilityRankError::InvalidInput("input bytes overflow".into())
            })?;
        let hv_length_bytes = hv_lengths
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| {
                CudaHistoricalVolatilityRankError::InvalidInput("hv_length bytes overflow".into())
            })?;
        let rank_length_bytes = rank_lengths
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| {
                CudaHistoricalVolatilityRankError::InvalidInput("rank_length bytes overflow".into())
            })?;
        let scale_bytes = annualization_scales
            .len()
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaHistoricalVolatilityRankError::InvalidInput("scale bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaHistoricalVolatilityRankError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|v| v.checked_mul(2))
            .ok_or_else(|| {
                CudaHistoricalVolatilityRankError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(hv_length_bytes)
            .and_then(|v| v.checked_add(rank_length_bytes))
            .and_then(|v| v.checked_add(scale_bytes))
            .and_then(|v| v.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaHistoricalVolatilityRankError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_data = DeviceBuffer::from_slice(data)?;
        let d_hv_lengths = DeviceBuffer::from_slice(&hv_lengths)?;
        let d_rank_lengths = DeviceBuffer::from_slice(&rank_lengths)?;
        let d_annualization_scales = DeviceBuffer::from_slice(&annualization_scales)?;
        let mut d_hvr = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_hv = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("historical_volatility_rank_batch_f64")
            .map_err(|_| CudaHistoricalVolatilityRankError::MissingKernelSymbol {
                name: "historical_volatility_rank_batch_f64",
            })?;
        let grid_x = ((rows as u32) + HISTORICAL_VOLATILITY_RANK_BLOCK_X - 1)
            / HISTORICAL_VOLATILITY_RANK_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(HISTORICAL_VOLATILITY_RANK_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_data.as_device_ptr(),
                cols as i32,
                d_hv_lengths.as_device_ptr(),
                d_rank_lengths.as_device_ptr(),
                d_annualization_scales.as_device_ptr(),
                rows as i32,
                d_hvr.as_device_ptr(),
                d_hv.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaHistoricalVolatilityRankBatchResult {
            outputs: HistoricalVolatilityRankDeviceArrayF64Pair {
                hvr: DeviceArrayF64 {
                    buf: d_hvr,
                    rows,
                    cols,
                },
                hv: DeviceArrayF64 {
                    buf: d_hv,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
