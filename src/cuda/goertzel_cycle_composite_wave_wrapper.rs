#![cfg(feature = "cuda")]

use crate::indicators::goertzel_cycle_composite_wave::{
    goertzel_cycle_composite_wave_batch_with_kernel, GoertzelCycleCompositeWaveBatchRange,
    GoertzelCycleCompositeWaveParams,
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
pub enum CudaGoertzelCycleCompositeWaveError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
}

pub struct GoertzelCycleCompositeWaveDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl GoertzelCycleCompositeWaveDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct GoertzelCycleCompositeWaveDeviceOutputs {
    pub values: GoertzelCycleCompositeWaveDeviceArrayF64,
}

impl GoertzelCycleCompositeWaveDeviceOutputs {
    #[inline]
    pub fn rows(&self) -> usize {
        self.values.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.values.cols
    }
}

pub struct CudaGoertzelCycleCompositeWaveBatchResult {
    pub outputs: GoertzelCycleCompositeWaveDeviceOutputs,
    pub combos: Vec<GoertzelCycleCompositeWaveParams>,
}

pub struct CudaGoertzelCycleCompositeWave {
    module: Module,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaGoertzelCycleCompositeWave {
    pub fn new(device_id: usize) -> Result<Self, CudaGoertzelCycleCompositeWaveError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("goertzel_cycle_composite_wave_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaGoertzelCycleCompositeWaveError> {
        Ok(())
    }

    pub fn batch_dev(
        &self,
        data: &[f64],
        sweep: &GoertzelCycleCompositeWaveBatchRange,
    ) -> Result<CudaGoertzelCycleCompositeWaveBatchResult, CudaGoertzelCycleCompositeWaveError>
    {
        self.module
            .get_function("goertzel_cycle_composite_wave_batch_f64")
            .map_err(
                |_| CudaGoertzelCycleCompositeWaveError::MissingKernelSymbol {
                    name: "goertzel_cycle_composite_wave_batch_f64",
                },
            )?;
        let cpu = goertzel_cycle_composite_wave_batch_with_kernel(data, sweep, Kernel::ScalarBatch)
            .map_err(|e| CudaGoertzelCycleCompositeWaveError::InvalidInput(e.to_string()))?;

        Ok(CudaGoertzelCycleCompositeWaveBatchResult {
            outputs: GoertzelCycleCompositeWaveDeviceOutputs {
                values: GoertzelCycleCompositeWaveDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.values)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
            },
            combos: cpu.combos,
        })
    }
}
