#![cfg(feature = "cuda")]

use crate::indicators::mesa_stochastic_multi_length::{
    expand_grid, MesaStochasticMultiLengthBatchRange, MesaStochasticMultiLengthParams,
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

const MESA_STOCHASTIC_MULTI_LENGTH_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaMesaStochasticMultiLengthError {
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

pub struct MesaStochasticMultiLengthDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl MesaStochasticMultiLengthDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct MesaStochasticMultiLengthDeviceArrayF64Octa {
    pub mesa_1: MesaStochasticMultiLengthDeviceArrayF64,
    pub mesa_2: MesaStochasticMultiLengthDeviceArrayF64,
    pub mesa_3: MesaStochasticMultiLengthDeviceArrayF64,
    pub mesa_4: MesaStochasticMultiLengthDeviceArrayF64,
    pub trigger_1: MesaStochasticMultiLengthDeviceArrayF64,
    pub trigger_2: MesaStochasticMultiLengthDeviceArrayF64,
    pub trigger_3: MesaStochasticMultiLengthDeviceArrayF64,
    pub trigger_4: MesaStochasticMultiLengthDeviceArrayF64,
}

impl MesaStochasticMultiLengthDeviceArrayF64Octa {
    #[inline]
    pub fn rows(&self) -> usize {
        self.mesa_1.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.mesa_1.cols
    }
}

pub struct CudaMesaStochasticMultiLengthBatchResult {
    pub outputs: MesaStochasticMultiLengthDeviceArrayF64Octa,
    pub combos: Vec<MesaStochasticMultiLengthParams>,
}

pub struct CudaMesaStochasticMultiLength {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaMesaStochasticMultiLength {
    pub fn new(device_id: usize) -> Result<Self, CudaMesaStochasticMultiLengthError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("mesa_stochastic_multi_length_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaMesaStochasticMultiLengthError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn will_fit(
        required: usize,
        headroom: usize,
    ) -> Result<(), CudaMesaStochasticMultiLengthError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaMesaStochasticMultiLengthError::OutOfMemory {
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
    ) -> Result<(), CudaMesaStochasticMultiLengthError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaMesaStochasticMultiLengthError::LaunchConfigTooLarge {
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
        sweep: &MesaStochasticMultiLengthBatchRange,
    ) -> Result<CudaMesaStochasticMultiLengthBatchResult, CudaMesaStochasticMultiLengthError> {
        if source.is_empty() {
            return Err(CudaMesaStochasticMultiLengthError::InvalidInput(
                "empty input".into(),
            ));
        }
        if !source.iter().any(|value| value.is_finite()) {
            return Err(CudaMesaStochasticMultiLengthError::InvalidInput(
                "all values are NaN".into(),
            ));
        }

        let combos = expand_grid(sweep)
            .map_err(|err| CudaMesaStochasticMultiLengthError::InvalidInput(err.to_string()))?;
        if combos.is_empty() {
            return Err(CudaMesaStochasticMultiLengthError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let rows = combos.len();
        let cols = source.len();
        let mut length_1s = Vec::with_capacity(rows);
        let mut length_2s = Vec::with_capacity(rows);
        let mut length_3s = Vec::with_capacity(rows);
        let mut length_4s = Vec::with_capacity(rows);
        let mut trigger_lengths = Vec::with_capacity(rows);
        let mut max_length = 0usize;
        let mut max_trigger_length = 0usize;

        for combo in &combos {
            let length_1 = combo.length_1.unwrap_or(48);
            let length_2 = combo.length_2.unwrap_or(21);
            let length_3 = combo.length_3.unwrap_or(9);
            let length_4 = combo.length_4.unwrap_or(6);
            let trigger_length = combo.trigger_length.unwrap_or(2);
            if length_1 == 0
                || length_2 == 0
                || length_3 == 0
                || length_4 == 0
                || trigger_length == 0
            {
                return Err(CudaMesaStochasticMultiLengthError::InvalidInput(
                    "invalid zero period in parameter grid".into(),
                ));
            }
            max_length = max_length
                .max(length_1)
                .max(length_2)
                .max(length_3)
                .max(length_4);
            max_trigger_length = max_trigger_length.max(trigger_length);
            length_1s.push(i32::try_from(length_1).map_err(|_| {
                CudaMesaStochasticMultiLengthError::InvalidInput(format!(
                    "length_1 out of range: {length_1}"
                ))
            })?);
            length_2s.push(i32::try_from(length_2).map_err(|_| {
                CudaMesaStochasticMultiLengthError::InvalidInput(format!(
                    "length_2 out of range: {length_2}"
                ))
            })?);
            length_3s.push(i32::try_from(length_3).map_err(|_| {
                CudaMesaStochasticMultiLengthError::InvalidInput(format!(
                    "length_3 out of range: {length_3}"
                ))
            })?);
            length_4s.push(i32::try_from(length_4).map_err(|_| {
                CudaMesaStochasticMultiLengthError::InvalidInput(format!(
                    "length_4 out of range: {length_4}"
                ))
            })?);
            trigger_lengths.push(i32::try_from(trigger_length).map_err(|_| {
                CudaMesaStochasticMultiLengthError::InvalidInput(format!(
                    "trigger_length out of range: {trigger_length}"
                ))
            })?);
        }

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaMesaStochasticMultiLengthError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = rows
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| value.checked_mul(5))
            .ok_or_else(|| {
                CudaMesaStochasticMultiLengthError::InvalidInput("params bytes overflow".into())
            })?;
        let scratch_elems = rows
            .checked_mul(max_length)
            .and_then(|value| value.checked_mul(4))
            .and_then(|value| {
                rows.checked_mul(max_trigger_length)
                    .and_then(|other| other.checked_mul(4))
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| {
                CudaMesaStochasticMultiLengthError::InvalidInput("scratch elems overflow".into())
            })?;
        let scratch_bytes = scratch_elems
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaMesaStochasticMultiLengthError::InvalidInput("scratch bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaMesaStochasticMultiLengthError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(8))
            .ok_or_else(|| {
                CudaMesaStochasticMultiLengthError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(scratch_bytes))
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaMesaStochasticMultiLengthError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_source = DeviceBuffer::from_slice(source)?;
        let d_length_1s = DeviceBuffer::from_slice(&length_1s)?;
        let d_length_2s = DeviceBuffer::from_slice(&length_2s)?;
        let d_length_3s = DeviceBuffer::from_slice(&length_3s)?;
        let d_length_4s = DeviceBuffer::from_slice(&length_4s)?;
        let d_trigger_lengths = DeviceBuffer::from_slice(&trigger_lengths)?;
        let d_mesa_ring = unsafe { DeviceBuffer::<f64>::uninitialized(rows * max_length * 4)? };
        let d_trigger_ring =
            unsafe { DeviceBuffer::<f64>::uninitialized(rows * max_trigger_length * 4)? };
        let d_out_mesa_1 = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_mesa_2 = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_mesa_3 = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_mesa_4 = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_trigger_1 = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_trigger_2 = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_trigger_3 = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_trigger_4 = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("mesa_stochastic_multi_length_batch_f64")
            .map_err(
                |_| CudaMesaStochasticMultiLengthError::MissingKernelSymbol {
                    name: "mesa_stochastic_multi_length_batch_f64",
                },
            )?;
        let grid_x = ((rows as u32) + MESA_STOCHASTIC_MULTI_LENGTH_BLOCK_X - 1)
            / MESA_STOCHASTIC_MULTI_LENGTH_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(MESA_STOCHASTIC_MULTI_LENGTH_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_source.as_device_ptr(),
                cols as i32,
                d_length_1s.as_device_ptr(),
                d_length_2s.as_device_ptr(),
                d_length_3s.as_device_ptr(),
                d_length_4s.as_device_ptr(),
                d_trigger_lengths.as_device_ptr(),
                rows as i32,
                max_length as i32,
                max_trigger_length as i32,
                d_mesa_ring.as_device_ptr(),
                d_trigger_ring.as_device_ptr(),
                d_out_mesa_1.as_device_ptr(),
                d_out_mesa_2.as_device_ptr(),
                d_out_mesa_3.as_device_ptr(),
                d_out_mesa_4.as_device_ptr(),
                d_out_trigger_1.as_device_ptr(),
                d_out_trigger_2.as_device_ptr(),
                d_out_trigger_3.as_device_ptr(),
                d_out_trigger_4.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaMesaStochasticMultiLengthBatchResult {
            outputs: MesaStochasticMultiLengthDeviceArrayF64Octa {
                mesa_1: MesaStochasticMultiLengthDeviceArrayF64 {
                    buf: d_out_mesa_1,
                    rows,
                    cols,
                },
                mesa_2: MesaStochasticMultiLengthDeviceArrayF64 {
                    buf: d_out_mesa_2,
                    rows,
                    cols,
                },
                mesa_3: MesaStochasticMultiLengthDeviceArrayF64 {
                    buf: d_out_mesa_3,
                    rows,
                    cols,
                },
                mesa_4: MesaStochasticMultiLengthDeviceArrayF64 {
                    buf: d_out_mesa_4,
                    rows,
                    cols,
                },
                trigger_1: MesaStochasticMultiLengthDeviceArrayF64 {
                    buf: d_out_trigger_1,
                    rows,
                    cols,
                },
                trigger_2: MesaStochasticMultiLengthDeviceArrayF64 {
                    buf: d_out_trigger_2,
                    rows,
                    cols,
                },
                trigger_3: MesaStochasticMultiLengthDeviceArrayF64 {
                    buf: d_out_trigger_3,
                    rows,
                    cols,
                },
                trigger_4: MesaStochasticMultiLengthDeviceArrayF64 {
                    buf: d_out_trigger_4,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
