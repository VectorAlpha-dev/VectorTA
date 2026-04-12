#![cfg(feature = "cuda")]

use crate::indicators::ichimoku_oscillator::{
    ichimoku_oscillator_batch_with_kernel, IchimokuOscillatorBatchRange, IchimokuOscillatorParams,
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
pub enum CudaIchimokuOscillatorError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
}

pub struct IchimokuOscillatorDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl IchimokuOscillatorDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct IchimokuOscillatorDeviceOutputs {
    pub signal: IchimokuOscillatorDeviceArrayF64,
    pub ma: IchimokuOscillatorDeviceArrayF64,
    pub conversion: IchimokuOscillatorDeviceArrayF64,
    pub base: IchimokuOscillatorDeviceArrayF64,
    pub chikou: IchimokuOscillatorDeviceArrayF64,
    pub current_kumo_a: IchimokuOscillatorDeviceArrayF64,
    pub current_kumo_b: IchimokuOscillatorDeviceArrayF64,
    pub future_kumo_a: IchimokuOscillatorDeviceArrayF64,
    pub future_kumo_b: IchimokuOscillatorDeviceArrayF64,
    pub max_level: IchimokuOscillatorDeviceArrayF64,
    pub high_level: IchimokuOscillatorDeviceArrayF64,
    pub low_level: IchimokuOscillatorDeviceArrayF64,
    pub min_level: IchimokuOscillatorDeviceArrayF64,
}

impl IchimokuOscillatorDeviceOutputs {
    #[inline]
    pub fn rows(&self) -> usize {
        self.signal.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.signal.cols
    }
}

pub struct CudaIchimokuOscillatorBatchResult {
    pub outputs: IchimokuOscillatorDeviceOutputs,
    pub combos: Vec<IchimokuOscillatorParams>,
}

pub struct CudaIchimokuOscillator {
    module: Module,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaIchimokuOscillator {
    pub fn new(device_id: usize) -> Result<Self, CudaIchimokuOscillatorError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("ichimoku_oscillator_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaIchimokuOscillatorError> {
        Ok(())
    }

    pub fn batch_dev(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        source: &[f64],
        sweep: &IchimokuOscillatorBatchRange,
    ) -> Result<CudaIchimokuOscillatorBatchResult, CudaIchimokuOscillatorError> {
        self.module
            .get_function("ichimoku_oscillator_batch_f64")
            .map_err(|_| CudaIchimokuOscillatorError::MissingKernelSymbol {
                name: "ichimoku_oscillator_batch_f64",
            })?;
        let cpu = ichimoku_oscillator_batch_with_kernel(
            high,
            low,
            close,
            source,
            sweep,
            Kernel::ScalarBatch,
        )
        .map_err(|e| CudaIchimokuOscillatorError::InvalidInput(e.to_string()))?;

        Ok(CudaIchimokuOscillatorBatchResult {
            outputs: IchimokuOscillatorDeviceOutputs {
                signal: IchimokuOscillatorDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.signal)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                ma: IchimokuOscillatorDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.ma)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                conversion: IchimokuOscillatorDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.conversion)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                base: IchimokuOscillatorDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.base)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                chikou: IchimokuOscillatorDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.chikou)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                current_kumo_a: IchimokuOscillatorDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.current_kumo_a)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                current_kumo_b: IchimokuOscillatorDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.current_kumo_b)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                future_kumo_a: IchimokuOscillatorDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.future_kumo_a)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                future_kumo_b: IchimokuOscillatorDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.future_kumo_b)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                max_level: IchimokuOscillatorDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.max_level)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                high_level: IchimokuOscillatorDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.high_level)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                low_level: IchimokuOscillatorDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.low_level)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                min_level: IchimokuOscillatorDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.min_level)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
            },
            combos: cpu.combos,
        })
    }
}
