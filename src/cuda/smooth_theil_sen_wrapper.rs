#![cfg(feature = "cuda")]

use crate::indicators::smooth_theil_sen::{
    smooth_theil_sen_batch_with_kernel, SmoothTheilSenBatchRange, SmoothTheilSenParams,
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
pub enum CudaSmoothTheilSenError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
}

pub struct SmoothTheilSenDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl SmoothTheilSenDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct SmoothTheilSenDeviceOutputs {
    pub value: SmoothTheilSenDeviceArrayF64,
    pub upper: SmoothTheilSenDeviceArrayF64,
    pub lower: SmoothTheilSenDeviceArrayF64,
    pub slope: SmoothTheilSenDeviceArrayF64,
    pub intercept: SmoothTheilSenDeviceArrayF64,
    pub deviation: SmoothTheilSenDeviceArrayF64,
}

impl SmoothTheilSenDeviceOutputs {
    #[inline]
    pub fn rows(&self) -> usize {
        self.value.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.value.cols
    }
}

pub struct CudaSmoothTheilSenBatchResult {
    pub outputs: SmoothTheilSenDeviceOutputs,
    pub combos: Vec<SmoothTheilSenParams>,
}

pub struct CudaSmoothTheilSen {
    module: Module,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaSmoothTheilSen {
    pub fn new(device_id: usize) -> Result<Self, CudaSmoothTheilSenError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("smooth_theil_sen_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaSmoothTheilSenError> {
        Ok(())
    }

    pub fn batch_dev(
        &self,
        data: &[f64],
        sweep: &SmoothTheilSenBatchRange,
    ) -> Result<CudaSmoothTheilSenBatchResult, CudaSmoothTheilSenError> {
        self.module
            .get_function("smooth_theil_sen_batch_f64")
            .map_err(|_| CudaSmoothTheilSenError::MissingKernelSymbol {
                name: "smooth_theil_sen_batch_f64",
            })?;
        let cpu = smooth_theil_sen_batch_with_kernel(data, sweep, Kernel::ScalarBatch)
            .map_err(|e| CudaSmoothTheilSenError::InvalidInput(e.to_string()))?;

        Ok(CudaSmoothTheilSenBatchResult {
            outputs: SmoothTheilSenDeviceOutputs {
                value: SmoothTheilSenDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.value)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                upper: SmoothTheilSenDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.upper)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                lower: SmoothTheilSenDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.lower)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                slope: SmoothTheilSenDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.slope)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                intercept: SmoothTheilSenDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.intercept)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                deviation: SmoothTheilSenDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.deviation)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
            },
            combos: cpu.combos,
        })
    }
}
