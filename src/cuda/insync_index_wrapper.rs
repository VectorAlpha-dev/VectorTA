#![cfg(feature = "cuda")]

use crate::indicators::insync_index::{
    insync_index_batch_with_kernel, InsyncIndexBatchRange, InsyncIndexParams,
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
pub enum CudaInsyncIndexError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
}

pub struct InsyncIndexDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl InsyncIndexDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct InsyncIndexDeviceOutputs {
    pub values: InsyncIndexDeviceArrayF64,
}

impl InsyncIndexDeviceOutputs {
    #[inline]
    pub fn rows(&self) -> usize {
        self.values.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.values.cols
    }
}

pub struct CudaInsyncIndexBatchResult {
    pub outputs: InsyncIndexDeviceOutputs,
    pub combos: Vec<InsyncIndexParams>,
}

pub struct CudaInsyncIndex {
    module: Module,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaInsyncIndex {
    pub fn new(device_id: usize) -> Result<Self, CudaInsyncIndexError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("insync_index_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaInsyncIndexError> {
        Ok(())
    }

    pub fn batch_dev(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
        sweep: &InsyncIndexBatchRange,
    ) -> Result<CudaInsyncIndexBatchResult, CudaInsyncIndexError> {
        self.module
            .get_function("insync_index_batch_f64")
            .map_err(|_| CudaInsyncIndexError::MissingKernelSymbol {
                name: "insync_index_batch_f64",
            })?;
        let cpu =
            insync_index_batch_with_kernel(high, low, close, volume, sweep, Kernel::ScalarBatch)
                .map_err(|e| CudaInsyncIndexError::InvalidInput(e.to_string()))?;

        Ok(CudaInsyncIndexBatchResult {
            outputs: InsyncIndexDeviceOutputs {
                values: InsyncIndexDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.values)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
            },
            combos: cpu.combos,
        })
    }
}
