#![cfg(feature = "cuda")]

use crate::indicators::possible_rsi::{
    possible_rsi_batch_with_kernel, PossibleRsiBatchRange, PossibleRsiParams,
};
use crate::utilities::enums::Kernel;
use cust::context::Context;
use cust::device::Device;
use cust::memory::DeviceBuffer;
use cust::module::Module;
use cust::prelude::*;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaPossibleRsiError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
}

pub struct PossibleRsiDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl PossibleRsiDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct PossibleRsiDeviceOutputs {
    pub value: PossibleRsiDeviceArrayF64,
    pub buy_level: PossibleRsiDeviceArrayF64,
    pub sell_level: PossibleRsiDeviceArrayF64,
    pub middle_level: PossibleRsiDeviceArrayF64,
    pub state: PossibleRsiDeviceArrayF64,
    pub long_signal: PossibleRsiDeviceArrayF64,
    pub short_signal: PossibleRsiDeviceArrayF64,
}

impl PossibleRsiDeviceOutputs {
    #[inline]
    pub fn rows(&self) -> usize {
        self.value.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.value.cols
    }
}

pub struct CudaPossibleRsiBatchResult {
    pub outputs: PossibleRsiDeviceOutputs,
    pub combos: Vec<PossibleRsiParams>,
}

pub struct CudaPossibleRsi {
    module: Module,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaPossibleRsi {
    pub fn new(device_id: usize) -> Result<Self, CudaPossibleRsiError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("possible_rsi_kernel")?;
        Ok(Self {
            module,
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

    pub fn synchronize(&self) -> Result<(), CudaPossibleRsiError> {
        Ok(())
    }

    pub fn batch_dev(
        &self,
        data: &[f64],
        range: &PossibleRsiBatchRange,
        base: &PossibleRsiParams,
    ) -> Result<CudaPossibleRsiBatchResult, CudaPossibleRsiError> {
        self.module
            .get_function("possible_rsi_batch_f64")
            .map_err(|_| CudaPossibleRsiError::MissingKernelSymbol {
                name: "possible_rsi_batch_f64",
            })?;
        let cpu = possible_rsi_batch_with_kernel(data, range, base, Kernel::ScalarBatch)
            .map_err(|e| CudaPossibleRsiError::InvalidInput(e.to_string()))?;

        Ok(CudaPossibleRsiBatchResult {
            outputs: PossibleRsiDeviceOutputs {
                value: PossibleRsiDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.value)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                buy_level: PossibleRsiDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.buy_level)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                sell_level: PossibleRsiDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.sell_level)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                middle_level: PossibleRsiDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.middle_level)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                state: PossibleRsiDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.state)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                long_signal: PossibleRsiDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.long_signal)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                short_signal: PossibleRsiDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.short_signal)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
            },
            combos: cpu.combos,
        })
    }
}
