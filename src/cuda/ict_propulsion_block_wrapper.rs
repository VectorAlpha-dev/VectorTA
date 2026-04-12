#![cfg(feature = "cuda")]

use crate::indicators::ict_propulsion_block::{
    ict_propulsion_block_batch_with_kernel, IctPropulsionBlockBatchRange, IctPropulsionBlockParams,
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
pub enum CudaIctPropulsionBlockError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
}

pub struct IctPropulsionBlockDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl IctPropulsionBlockDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct IctPropulsionBlockDeviceOutputs {
    pub bullish_high: IctPropulsionBlockDeviceArrayF64,
    pub bullish_low: IctPropulsionBlockDeviceArrayF64,
    pub bullish_kind: IctPropulsionBlockDeviceArrayF64,
    pub bullish_active: IctPropulsionBlockDeviceArrayF64,
    pub bullish_mitigated: IctPropulsionBlockDeviceArrayF64,
    pub bullish_new: IctPropulsionBlockDeviceArrayF64,
    pub bearish_high: IctPropulsionBlockDeviceArrayF64,
    pub bearish_low: IctPropulsionBlockDeviceArrayF64,
    pub bearish_kind: IctPropulsionBlockDeviceArrayF64,
    pub bearish_active: IctPropulsionBlockDeviceArrayF64,
    pub bearish_mitigated: IctPropulsionBlockDeviceArrayF64,
    pub bearish_new: IctPropulsionBlockDeviceArrayF64,
}

impl IctPropulsionBlockDeviceOutputs {
    #[inline]
    pub fn rows(&self) -> usize {
        self.bullish_high.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.bullish_high.cols
    }
}

pub struct CudaIctPropulsionBlockBatchResult {
    pub outputs: IctPropulsionBlockDeviceOutputs,
    pub combos: Vec<IctPropulsionBlockParams>,
}

pub struct CudaIctPropulsionBlock {
    module: Module,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaIctPropulsionBlock {
    pub fn new(device_id: usize) -> Result<Self, CudaIctPropulsionBlockError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("ict_propulsion_block_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaIctPropulsionBlockError> {
        Ok(())
    }

    pub fn batch_dev(
        &self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
        sweep: &IctPropulsionBlockBatchRange,
    ) -> Result<CudaIctPropulsionBlockBatchResult, CudaIctPropulsionBlockError> {
        self.module
            .get_function("ict_propulsion_block_batch_f64")
            .map_err(|_| CudaIctPropulsionBlockError::MissingKernelSymbol {
                name: "ict_propulsion_block_batch_f64",
            })?;
        let cpu = ict_propulsion_block_batch_with_kernel(
            open,
            high,
            low,
            close,
            sweep,
            Kernel::ScalarBatch,
        )
        .map_err(|e| CudaIctPropulsionBlockError::InvalidInput(e.to_string()))?;

        Ok(CudaIctPropulsionBlockBatchResult {
            outputs: IctPropulsionBlockDeviceOutputs {
                bullish_high: IctPropulsionBlockDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.bullish_high)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                bullish_low: IctPropulsionBlockDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.bullish_low)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                bullish_kind: IctPropulsionBlockDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.bullish_kind)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                bullish_active: IctPropulsionBlockDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.bullish_active)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                bullish_mitigated: IctPropulsionBlockDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.bullish_mitigated)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                bullish_new: IctPropulsionBlockDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.bullish_new)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                bearish_high: IctPropulsionBlockDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.bearish_high)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                bearish_low: IctPropulsionBlockDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.bearish_low)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                bearish_kind: IctPropulsionBlockDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.bearish_kind)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                bearish_active: IctPropulsionBlockDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.bearish_active)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                bearish_mitigated: IctPropulsionBlockDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.bearish_mitigated)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                bearish_new: IctPropulsionBlockDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.bearish_new)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
            },
            combos: cpu.combos,
        })
    }
}
