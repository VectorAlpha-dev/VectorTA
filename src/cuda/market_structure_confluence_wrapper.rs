#![cfg(feature = "cuda")]

use crate::indicators::market_structure_confluence::{
    market_structure_confluence_batch_with_kernel, MarketStructureConfluenceBatchRange,
    MarketStructureConfluenceParams,
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
pub enum CudaMarketStructureConfluenceError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
}

pub struct MarketStructureConfluenceDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl MarketStructureConfluenceDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct MarketStructureConfluenceDeviceOutputs {
    pub basis: MarketStructureConfluenceDeviceArrayF64,
    pub upper_band: MarketStructureConfluenceDeviceArrayF64,
    pub lower_band: MarketStructureConfluenceDeviceArrayF64,
    pub structure_direction: MarketStructureConfluenceDeviceArrayF64,
    pub bullish_arrow: MarketStructureConfluenceDeviceArrayF64,
    pub bearish_arrow: MarketStructureConfluenceDeviceArrayF64,
    pub bullish_change: MarketStructureConfluenceDeviceArrayF64,
    pub bearish_change: MarketStructureConfluenceDeviceArrayF64,
    pub hh: MarketStructureConfluenceDeviceArrayF64,
    pub lh: MarketStructureConfluenceDeviceArrayF64,
    pub hl: MarketStructureConfluenceDeviceArrayF64,
    pub ll: MarketStructureConfluenceDeviceArrayF64,
    pub bullish_bos: MarketStructureConfluenceDeviceArrayF64,
    pub bullish_choch: MarketStructureConfluenceDeviceArrayF64,
    pub bearish_bos: MarketStructureConfluenceDeviceArrayF64,
    pub bearish_choch: MarketStructureConfluenceDeviceArrayF64,
}

impl MarketStructureConfluenceDeviceOutputs {
    #[inline]
    pub fn rows(&self) -> usize {
        self.basis.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.basis.cols
    }
}

pub struct CudaMarketStructureConfluenceBatchResult {
    pub outputs: MarketStructureConfluenceDeviceOutputs,
    pub combos: Vec<MarketStructureConfluenceParams>,
}

pub struct CudaMarketStructureConfluence {
    module: Module,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaMarketStructureConfluence {
    pub fn new(device_id: usize) -> Result<Self, CudaMarketStructureConfluenceError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("market_structure_confluence_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaMarketStructureConfluenceError> {
        Ok(())
    }

    pub fn batch_dev(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        range: &MarketStructureConfluenceBatchRange,
    ) -> Result<CudaMarketStructureConfluenceBatchResult, CudaMarketStructureConfluenceError> {
        self.module
            .get_function("market_structure_confluence_batch_f64")
            .map_err(
                |_| CudaMarketStructureConfluenceError::MissingKernelSymbol {
                    name: "market_structure_confluence_batch_f64",
                },
            )?;
        let cpu = market_structure_confluence_batch_with_kernel(
            high,
            low,
            close,
            range,
            Kernel::ScalarBatch,
        )
        .map_err(|e| CudaMarketStructureConfluenceError::InvalidInput(e.to_string()))?;

        Ok(CudaMarketStructureConfluenceBatchResult {
            outputs: MarketStructureConfluenceDeviceOutputs {
                basis: MarketStructureConfluenceDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.basis)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                upper_band: MarketStructureConfluenceDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.upper_band)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                lower_band: MarketStructureConfluenceDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.lower_band)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                structure_direction: MarketStructureConfluenceDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.structure_direction)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                bullish_arrow: MarketStructureConfluenceDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.bullish_arrow)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                bearish_arrow: MarketStructureConfluenceDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.bearish_arrow)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                bullish_change: MarketStructureConfluenceDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.bullish_change)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                bearish_change: MarketStructureConfluenceDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.bearish_change)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                hh: MarketStructureConfluenceDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.hh)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                lh: MarketStructureConfluenceDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.lh)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                hl: MarketStructureConfluenceDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.hl)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                ll: MarketStructureConfluenceDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.ll)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                bullish_bos: MarketStructureConfluenceDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.bullish_bos)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                bullish_choch: MarketStructureConfluenceDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.bullish_choch)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                bearish_bos: MarketStructureConfluenceDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.bearish_bos)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
                bearish_choch: MarketStructureConfluenceDeviceArrayF64 {
                    buf: DeviceBuffer::from_slice(&cpu.bearish_choch)?,
                    rows: cpu.rows,
                    cols: cpu.cols,
                },
            },
            combos: cpu.combos,
        })
    }
}
