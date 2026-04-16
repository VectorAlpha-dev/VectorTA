#![cfg(feature = "cuda")]

use crate::indicators::vdubus_divergence_wave_pattern_generator::{
    vdubus_divergence_wave_pattern_generator_batch_with_kernel,
    VdubusDivergenceWavePatternGeneratorBatchRange, VdubusDivergenceWavePatternGeneratorParams,
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
pub enum CudaVdubusDivergenceWavePatternGeneratorError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
}

pub struct VdubusDivergenceWavePatternGeneratorDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl VdubusDivergenceWavePatternGeneratorDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct VdubusDivergenceWavePatternGeneratorDeviceOutputs {
    pub fast_standard: VdubusDivergenceWavePatternGeneratorDeviceArrayF64,
    pub fast_climax: VdubusDivergenceWavePatternGeneratorDeviceArrayF64,
    pub fast_rounded: VdubusDivergenceWavePatternGeneratorDeviceArrayF64,
    pub fast_predator: VdubusDivergenceWavePatternGeneratorDeviceArrayF64,
    pub slow_standard: VdubusDivergenceWavePatternGeneratorDeviceArrayF64,
    pub slow_climax: VdubusDivergenceWavePatternGeneratorDeviceArrayF64,
    pub slow_rounded: VdubusDivergenceWavePatternGeneratorDeviceArrayF64,
    pub slow_predator: VdubusDivergenceWavePatternGeneratorDeviceArrayF64,
    pub opposing_force: VdubusDivergenceWavePatternGeneratorDeviceArrayF64,
    pub macd: VdubusDivergenceWavePatternGeneratorDeviceArrayF64,
    pub signal: VdubusDivergenceWavePatternGeneratorDeviceArrayF64,
    pub hist: VdubusDivergenceWavePatternGeneratorDeviceArrayF64,
}

impl VdubusDivergenceWavePatternGeneratorDeviceOutputs {
    #[inline]
    pub fn rows(&self) -> usize {
        self.fast_standard.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.fast_standard.cols
    }
}

pub struct CudaVdubusDivergenceWavePatternGeneratorBatchResult {
    pub outputs: VdubusDivergenceWavePatternGeneratorDeviceOutputs,
    pub combos: Vec<VdubusDivergenceWavePatternGeneratorParams>,
}

pub struct CudaVdubusDivergenceWavePatternGenerator {
    module: Module,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaVdubusDivergenceWavePatternGenerator {
    pub fn new(device_id: usize) -> Result<Self, CudaVdubusDivergenceWavePatternGeneratorError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module =
            crate::load_cuda_embedded_module!("vdubus_divergence_wave_pattern_generator_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaVdubusDivergenceWavePatternGeneratorError> {
        Ok(())
    }

    pub fn batch_dev(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        sweep: &VdubusDivergenceWavePatternGeneratorBatchRange,
    ) -> Result<
        CudaVdubusDivergenceWavePatternGeneratorBatchResult,
        CudaVdubusDivergenceWavePatternGeneratorError,
    > {
        self.module
            .get_function("vdubus_divergence_wave_pattern_generator_batch_f64")
            .map_err(
                |_| CudaVdubusDivergenceWavePatternGeneratorError::MissingKernelSymbol {
                    name: "vdubus_divergence_wave_pattern_generator_batch_f64",
                },
            )?;
        let cpu = vdubus_divergence_wave_pattern_generator_batch_with_kernel(
            high,
            low,
            close,
            sweep,
            Kernel::ScalarBatch,
        )
        .map_err(|e| CudaVdubusDivergenceWavePatternGeneratorError::InvalidInput(e.to_string()))?;

        Ok(CudaVdubusDivergenceWavePatternGeneratorBatchResult {
            outputs: VdubusDivergenceWavePatternGeneratorDeviceOutputs {
                fast_standard: VdubusDivergenceWavePatternGeneratorDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.fast_standard)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                fast_climax: VdubusDivergenceWavePatternGeneratorDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.fast_climax)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                fast_rounded: VdubusDivergenceWavePatternGeneratorDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.fast_rounded)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                fast_predator: VdubusDivergenceWavePatternGeneratorDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.fast_predator)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                slow_standard: VdubusDivergenceWavePatternGeneratorDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.slow_standard)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                slow_climax: VdubusDivergenceWavePatternGeneratorDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.slow_climax)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                slow_rounded: VdubusDivergenceWavePatternGeneratorDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.slow_rounded)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                slow_predator: VdubusDivergenceWavePatternGeneratorDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.slow_predator)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                opposing_force: VdubusDivergenceWavePatternGeneratorDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.opposing_force)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                macd: VdubusDivergenceWavePatternGeneratorDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.macd)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                signal: VdubusDivergenceWavePatternGeneratorDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.signal)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                hist: VdubusDivergenceWavePatternGeneratorDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.hist)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
            },
            combos: cpu.combos,
        })
    }
}
