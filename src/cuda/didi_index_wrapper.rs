#![cfg(feature = "cuda")]

use crate::indicators::didi_index::{DidiIndexBatchRange, DidiIndexParams};
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

const DIDI_INDEX_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaDidiIndexError {
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

pub struct DidiIndexDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl DidiIndexDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct DidiIndexDeviceArrayF64Quad {
    pub short: DidiIndexDeviceArrayF64,
    pub long: DidiIndexDeviceArrayF64,
    pub crossover: DidiIndexDeviceArrayF64,
    pub crossunder: DidiIndexDeviceArrayF64,
}

impl DidiIndexDeviceArrayF64Quad {
    #[inline]
    pub fn rows(&self) -> usize {
        self.short.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.short.cols
    }
}

pub struct CudaDidiIndexBatchResult {
    pub outputs: DidiIndexDeviceArrayF64Quad,
    pub combos: Vec<DidiIndexParams>,
}

pub struct CudaDidiIndex {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaDidiIndex {
    pub fn new(device_id: usize) -> Result<Self, CudaDidiIndexError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("didi_index_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaDidiIndexError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn axis_usize(
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, CudaDidiIndexError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
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
            if out.is_empty() {
                return Err(CudaDidiIndexError::InvalidInput(format!(
                    "invalid range: start={start}, end={end}, step={step}"
                )));
            }
            Ok(out)
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
            if out.is_empty() {
                return Err(CudaDidiIndexError::InvalidInput(format!(
                    "invalid range: start={start}, end={end}, step={step}"
                )));
            }
            Ok(out)
        }
    }

    fn expand_grid(
        sweep: &DidiIndexBatchRange,
    ) -> Result<Vec<DidiIndexParams>, CudaDidiIndexError> {
        let shorts = Self::axis_usize(sweep.short_length)?;
        let mediums = Self::axis_usize(sweep.medium_length)?;
        let longs = Self::axis_usize(sweep.long_length)?;

        let mut out = Vec::with_capacity(shorts.len() * mediums.len() * longs.len());
        for &short_length in &shorts {
            for &medium_length in &mediums {
                for &long_length in &longs {
                    out.push(DidiIndexParams {
                        short_length: Some(short_length),
                        medium_length: Some(medium_length),
                        long_length: Some(long_length),
                    });
                }
            }
        }
        Ok(out)
    }

    fn first_valid_value(data: &[f64]) -> Option<usize> {
        data.iter().position(|value| value.is_finite())
    }

    fn count_valid_values(data: &[f64]) -> usize {
        data.iter().filter(|value| value.is_finite()).count()
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaDidiIndexError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaDidiIndexError::OutOfMemory {
                    required,
                    free,
                    headroom,
                });
            }
        }
        Ok(())
    }

    fn validate_launch(&self, grid: GridSize, block: BlockSize) -> Result<(), CudaDidiIndexError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaDidiIndexError::LaunchConfigTooLarge {
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
        sweep: &DidiIndexBatchRange,
    ) -> Result<CudaDidiIndexBatchResult, CudaDidiIndexError> {
        if data.is_empty() {
            return Err(CudaDidiIndexError::InvalidInput("empty input".into()));
        }

        let combos = Self::expand_grid(sweep)?;
        if combos.is_empty() {
            return Err(CudaDidiIndexError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let _ = Self::first_valid_value(data)
            .ok_or_else(|| CudaDidiIndexError::InvalidInput("all values are NaN".into()))?;
        let valid = Self::count_valid_values(data);

        let rows = combos.len();
        let cols = data.len();
        let mut short_lengths = Vec::with_capacity(rows);
        let mut medium_lengths = Vec::with_capacity(rows);
        let mut long_lengths = Vec::with_capacity(rows);
        for combo in &combos {
            let short_length = combo.short_length.unwrap_or(3);
            let medium_length = combo.medium_length.unwrap_or(8);
            let long_length = combo.long_length.unwrap_or(20);
            if short_length == 0 || short_length > cols {
                return Err(CudaDidiIndexError::InvalidInput(format!(
                    "invalid short_length: short_length={short_length}, data_len={cols}"
                )));
            }
            if medium_length == 0 || medium_length > cols {
                return Err(CudaDidiIndexError::InvalidInput(format!(
                    "invalid medium_length: medium_length={medium_length}, data_len={cols}"
                )));
            }
            if long_length == 0 || long_length > cols {
                return Err(CudaDidiIndexError::InvalidInput(format!(
                    "invalid long_length: long_length={long_length}, data_len={cols}"
                )));
            }
            let needed = short_length.max(medium_length).max(long_length);
            if valid < needed {
                return Err(CudaDidiIndexError::InvalidInput(format!(
                    "not enough valid data: needed={needed}, valid={valid}"
                )));
            }
            short_lengths.push(short_length as i32);
            medium_lengths.push(medium_length as i32);
            long_lengths.push(long_length as i32);
        }

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| CudaDidiIndexError::InvalidInput("input bytes overflow".into()))?;
        let params_bytes = short_lengths
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| {
                medium_lengths
                    .len()
                    .checked_mul(std::mem::size_of::<i32>())
                    .and_then(|other| value.checked_add(other))
            })
            .and_then(|value| {
                long_lengths
                    .len()
                    .checked_mul(std::mem::size_of::<i32>())
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| CudaDidiIndexError::InvalidInput("params bytes overflow".into()))?;
        let output_elems = rows
            .checked_mul(cols)
            .ok_or_else(|| CudaDidiIndexError::InvalidInput("rows*cols overflow".into()))?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| CudaDidiIndexError::InvalidInput("output bytes overflow".into()))?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| CudaDidiIndexError::InvalidInput("required bytes overflow".into()))?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_data = DeviceBuffer::from_slice(data)?;
        let d_short_lengths = DeviceBuffer::from_slice(&short_lengths)?;
        let d_medium_lengths = DeviceBuffer::from_slice(&medium_lengths)?;
        let d_long_lengths = DeviceBuffer::from_slice(&long_lengths)?;
        let d_out_short = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_long = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_crossover = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_crossunder = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("didi_index_batch_f64")
            .map_err(|_| CudaDidiIndexError::MissingKernelSymbol {
                name: "didi_index_batch_f64",
            })?;
        let grid_x = ((rows as u32) + DIDI_INDEX_BLOCK_X - 1) / DIDI_INDEX_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(DIDI_INDEX_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_data.as_device_ptr(),
                cols as i32,
                d_short_lengths.as_device_ptr(),
                d_medium_lengths.as_device_ptr(),
                d_long_lengths.as_device_ptr(),
                rows as i32,
                d_out_short.as_device_ptr(),
                d_out_long.as_device_ptr(),
                d_out_crossover.as_device_ptr(),
                d_out_crossunder.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaDidiIndexBatchResult {
            outputs: DidiIndexDeviceArrayF64Quad {
                short: DidiIndexDeviceArrayF64 {
                    buf: d_out_short,
                    rows,
                    cols,
                },
                long: DidiIndexDeviceArrayF64 {
                    buf: d_out_long,
                    rows,
                    cols,
                },
                crossover: DidiIndexDeviceArrayF64 {
                    buf: d_out_crossover,
                    rows,
                    cols,
                },
                crossunder: DidiIndexDeviceArrayF64 {
                    buf: d_out_crossunder,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
