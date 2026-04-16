#![cfg(feature = "cuda")]

use crate::indicators::kase_peak_oscillator_with_divergences::{
    kase_peak_oscillator_with_divergences_batch_with_kernel,
    KasePeakOscillatorWithDivergencesBatchRange,
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
pub enum CudaKasePeakOscillatorWithDivergencesError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
}

pub struct KasePeakOscillatorWithDivergencesDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl KasePeakOscillatorWithDivergencesDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct KasePeakOscillatorWithDivergencesDeviceOutputs {
    pub oscillator: KasePeakOscillatorWithDivergencesDeviceArrayF64,
    pub histogram: KasePeakOscillatorWithDivergencesDeviceArrayF64,
    pub max_peak_value: KasePeakOscillatorWithDivergencesDeviceArrayF64,
    pub min_peak_value: KasePeakOscillatorWithDivergencesDeviceArrayF64,
    pub market_extreme: KasePeakOscillatorWithDivergencesDeviceArrayF64,
    pub regular_bullish: KasePeakOscillatorWithDivergencesDeviceArrayF64,
    pub hidden_bullish: KasePeakOscillatorWithDivergencesDeviceArrayF64,
    pub regular_bearish: KasePeakOscillatorWithDivergencesDeviceArrayF64,
    pub hidden_bearish: KasePeakOscillatorWithDivergencesDeviceArrayF64,
    pub go_long: KasePeakOscillatorWithDivergencesDeviceArrayF64,
    pub go_short: KasePeakOscillatorWithDivergencesDeviceArrayF64,
}

impl KasePeakOscillatorWithDivergencesDeviceOutputs {
    #[inline]
    pub fn rows(&self) -> usize {
        self.oscillator.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.oscillator.cols
    }
}

pub struct CudaKasePeakOscillatorWithDivergencesBatchResult {
    pub outputs: KasePeakOscillatorWithDivergencesDeviceOutputs,
    pub deviations: Vec<f64>,
    pub short_cycles: Vec<usize>,
    pub long_cycles: Vec<usize>,
    pub sensitivities: Vec<f64>,
}

pub struct CudaKasePeakOscillatorWithDivergences {
    module: Module,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaKasePeakOscillatorWithDivergences {
    pub fn new(device_id: usize) -> Result<Self, CudaKasePeakOscillatorWithDivergencesError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module =
            crate::load_cuda_embedded_module!("kase_peak_oscillator_with_divergences_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaKasePeakOscillatorWithDivergencesError> {
        Ok(())
    }

    pub fn batch_dev(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        sweep: &KasePeakOscillatorWithDivergencesBatchRange,
    ) -> Result<
        CudaKasePeakOscillatorWithDivergencesBatchResult,
        CudaKasePeakOscillatorWithDivergencesError,
    > {
        self.module
            .get_function("kase_peak_oscillator_with_divergences_batch_f64")
            .map_err(
                |_| CudaKasePeakOscillatorWithDivergencesError::MissingKernelSymbol {
                    name: "kase_peak_oscillator_with_divergences_batch_f64",
                },
            )?;
        let cpu = kase_peak_oscillator_with_divergences_batch_with_kernel(
            high,
            low,
            close,
            sweep,
            Kernel::ScalarBatch,
        )
        .map_err(|e| CudaKasePeakOscillatorWithDivergencesError::InvalidInput(e.to_string()))?;

        Ok(CudaKasePeakOscillatorWithDivergencesBatchResult {
            outputs: KasePeakOscillatorWithDivergencesDeviceOutputs {
                oscillator: KasePeakOscillatorWithDivergencesDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.oscillator)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                histogram: KasePeakOscillatorWithDivergencesDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.histogram)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                max_peak_value: KasePeakOscillatorWithDivergencesDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.max_peak_value)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                min_peak_value: KasePeakOscillatorWithDivergencesDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.min_peak_value)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                market_extreme: KasePeakOscillatorWithDivergencesDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.market_extreme)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                regular_bullish: KasePeakOscillatorWithDivergencesDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.regular_bullish)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                hidden_bullish: KasePeakOscillatorWithDivergencesDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.hidden_bullish)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                regular_bearish: KasePeakOscillatorWithDivergencesDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.regular_bearish)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                hidden_bearish: KasePeakOscillatorWithDivergencesDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.hidden_bearish)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                go_long: KasePeakOscillatorWithDivergencesDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.go_long)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                go_short: KasePeakOscillatorWithDivergencesDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.go_short)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
            },
            deviations: cpu.deviations,
            short_cycles: cpu.short_cycles,
            long_cycles: cpu.long_cycles,
            sensitivities: cpu.sensitivities,
        })
    }
}
