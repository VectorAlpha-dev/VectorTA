//! Thin CUDA MA selector that mirrors the shape of `indicators::moving_averages::ma`.
//!
//! Dispatches to the existing CUDA wrappers by `ma_type` for the common
//! price-only moving averages where a single `period` is the only explicit
//! argument. Indicators that require additional inputs (e.g., OHLC, volume)
//! or return multiple outputs are reported as unsupported here — call their
//! dedicated CUDA wrappers directly.
//!
//! Decision: Selector remains a thin dispatcher (no new kernels). For Python,
//! device handles carry an Arc<Context> and device_id to keep the CUDA primary
//! context alive for correct VRAM frees; numerical outputs unchanged.

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::cuda::moving_averages::*;
use crate::utilities::data_loader::{source_type, Candles};
// Faster D2H with pinned buffers; enable async copies when possible
use cust::memory::{mem_get_info, AsyncCopyDestination, CopyDestination, LockedBuffer};
use cust::stream::{Stream, StreamFlags};
use cust::context::Context;
use thiserror::Error;
use std::sync::Arc;

/// Unified error type for the CUDA MA selector.
#[derive(Debug, Error)]
pub enum CudaMaSelectorError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("backend error: {0}")]
    Backend(String),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("unsupported: {0}")]
    Unsupported(String),
    #[error("invalid range: start={start} end={end} step={step}")]
    InvalidRange { start: usize, end: usize, step: usize },
    #[error("out of memory: required={required}B free={free}B headroom={headroom}B")]
    OutOfMemory { required: usize, free: usize, headroom: usize },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("device mismatch: buf={buf} current={current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
    NotImplemented,
}

#[inline]
fn mem_check_enabled() -> bool {
    match std::env::var("CUDA_MEM_CHECK") {
        Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
        Err(_) => true,
    }
}

#[inline]
fn device_mem_info() -> Option<(usize, usize)> { mem_get_info().ok() }

#[inline]
fn will_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaMaSelectorError> {
    if !mem_check_enabled() { return Ok(()); }
    if let Some((free, _total)) = device_mem_info() {
        if required_bytes.saturating_add(headroom_bytes) <= free {
            Ok(())
        } else {
            Err(CudaMaSelectorError::OutOfMemory { required: required_bytes, free, headroom: headroom_bytes })
        }
    } else {
        Ok(())
    }
}

/// Input data for the selector (price-only).
#[derive(Debug, Clone, Copy)]
pub enum CudaMaData<'a> {
    /// Use a candle field as the price source (e.g., "close").
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    /// Use a raw price slice.
    Slice(&'a [f64]),
    /// NEW: raw f32 slice (skips f64→f32 conversion cost)
    SliceF32(&'a [f32]),
}

impl<'a> CudaMaData<'a> {
    #[inline]
    fn as_prices_f64(self) -> &'a [f64] {
        match self {
            CudaMaData::Slice(s) => s,
            CudaMaData::Candles { candles, source } => source_type(candles, source),
            CudaMaData::SliceF32(_) => panic!("as_prices_f64 called for f32 slice"),
        }
    }

    #[inline]
    fn prices_len(self) -> usize {
        match self {
            CudaMaData::Slice(s) => s.len(),
            CudaMaData::SliceF32(s) => s.len(),
            CudaMaData::Candles { candles, source } => source_type(candles, source).len(),
        }
    }

    #[inline]
    fn to_prices_f32(self) -> Vec<f32> {
        match self {
            CudaMaData::SliceF32(s) => s.to_vec(),
            CudaMaData::Slice(s) => s.iter().map(|&v| v as f32).collect(),
            CudaMaData::Candles { candles, source } => {
                let src = source_type(candles, source);
                src.iter().map(|&v| v as f32).collect()
            }
        }
    }
}

/// Minimal, host-side dispatcher that launches the correct CUDA kernel for a
/// price-only moving average. For indicators with additional parameters, this
/// dispatcher sets sensible defaults that match the scalar path.
pub struct CudaMaSelector {
    device_id: usize,
    // Reusable non-blocking stream for async D2H copies
    stream: Stream,
}

impl CudaMaSelector {
    /// Create a new selector bound to a specific CUDA device.
    pub fn new(device_id: usize) -> Self {
        let stream =
            Stream::new(StreamFlags::NON_BLOCKING, None).expect("failed to create CUDA stream");
        Self { device_id, stream }
    }

    /// Compute the requested MA on device and return a device buffer handle
    /// (1 row × N cols). The row-major buffer can be staged back by the caller.
    pub fn ma_to_device(
        &self,
        ma_type: &str,
        data: CudaMaData,
        period: usize,
    ) -> Result<DeviceArrayF32, CudaMaSelectorError> {
        // Validate early without forcing a conversion
        let n = data.prices_len();
        if n == 0 {
            return Err(CudaMaSelectorError::InvalidInput(
                "empty price input".into(),
            ));
        }
        if period == 0 || period > n {
            return Err(CudaMaSelectorError::InvalidInput(format!(
                "invalid period: {} for length {}",
                period, n
            )));
        }

        // Case-insensitive helper without allocating
        let is = |s: &str| ma_type.eq_ignore_ascii_case(s);

        // Coverage additions that require volume/ohlc
        if is("vwma") {
            if let CudaMaData::Candles { candles, source } = data {
                let prices = source_type(candles, source);
                let prices_f32: Vec<f32> = prices.iter().map(|&v| v as f32).collect();
                let volumes_f32: Vec<f32> = candles.volume.iter().map(|&v| v as f32).collect();
                let sweep = crate::indicators::moving_averages::vwma::VwmaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaVwma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                return cuda
                    .vwma_batch_dev(&prices_f32, &volumes_f32, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()));
            } else {
                return Err(CudaMaSelectorError::Unsupported(
                    "vwma requires candles with volume; pass CudaMaData::Candles".into(),
                ));
            }
        }

        if is("vpwma") {
            if let CudaMaData::Candles { candles, source } = data {
                let prices = source_type(candles, source);
                let prices_f32: Vec<f32> = prices.iter().map(|&v| v as f32).collect();
                let sweep = crate::indicators::moving_averages::vpwma::VpwmaBatchRange {
                    period: (period, period, 0),
                    power: (0.382, 0.382, 0.0), // match CPU default
                };
                let cuda = CudaVpwma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                let (dev, _combos) = cuda
                    .vpwma_batch_dev(&prices_f32, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                return Ok(super::DeviceArrayF32 { buf: dev.buf, rows: dev.rows, cols: dev.cols });
            } else {
                return Err(CudaMaSelectorError::Unsupported(
                    "vpwma requires candles with volume; pass CudaMaData::Candles".into(),
                ));
            }
        }

        if is("vwap") {
            if let CudaMaData::Candles { candles, .. } = data {
                let sweep = crate::indicators::moving_averages::vwap::VwapBatchRange {
                    anchor: ("1d".to_string(), "1d".to_string(), 0),
                };
                // Use HLC3 to mirror CPU default
                let prices = &candles.hlc3;
                let cuda = CudaVwap::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                return cuda
                    .vwap_batch_dev(&candles.timestamp, &candles.volume, prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()));
            } else {
                return Err(CudaMaSelectorError::Unsupported(
                    "vwap requires OHLC + volume; pass CudaMaData::Candles".into(),
                ));
            }
        }

        // Lazy f64→f32 conversion only when we actually need price-only slices
        let mut prices_f32_cache: Option<Vec<f32>> = None;
        macro_rules! ensure_prices {
            () => {{
                if prices_f32_cache.is_none() {
                    prices_f32_cache = Some(data.to_prices_f32());
                }
                prices_f32_cache.as_ref().unwrap().as_slice()
            }};
        }

        match ma_type.to_ascii_lowercase().as_str() {
            // --- Price-only, period-only (direct mapping) ---
            "sma" => {
                let sweep = crate::indicators::moving_averages::sma::SmaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaSma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                let cuda = CudaSma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                let (dev, _combos) = cuda
                    .sma_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                Ok(dev)
            }
            "ema" => {
                let sweep = crate::indicators::moving_averages::ema::EmaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaEma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.ema_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "dema" => {
                let sweep = crate::indicators::moving_averages::dema::DemaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaDema::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.dema_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "wma" => {
                let sweep = crate::indicators::moving_averages::wma::WmaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaWma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.wma_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "zlema" => {
                let sweep = crate::indicators::moving_averages::zlema::ZlemaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaZlema::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                let cuda = CudaZlema::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                let (dev, _combos) = cuda
                    .zlema_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                Ok(dev)
            }
            "smma" => {
                let sweep = crate::indicators::moving_averages::smma::SmmaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaSmma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda
                    .smma_batch_dev(ensure_prices!(), &sweep)
                    .map(|h| super::alma_wrapper::DeviceArrayF32 { buf: h.buf, rows: h.rows, cols: h.cols })
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "trima" => {
                let sweep = crate::indicators::moving_averages::trima::TrimaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaTrima::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda
                    .trima_batch_dev(ensure_prices!(), &sweep)
                    .map(|h| super::alma_wrapper::DeviceArrayF32 { buf: h.buf, rows: h.rows, cols: h.cols })
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "tema" => {
                let sweep = crate::indicators::moving_averages::tema::TemaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaTema::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.tema_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "tilson" => {
                let sweep = crate::indicators::moving_averages::tilson::TilsonBatchRange {
                    period: (period, period, 0),
                    volume_factor: (0.0, 0.0, 0.0),
                };
                let cuda = CudaTilson::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda
                    .tilson_batch_dev(ensure_prices!(), &sweep)
                    .map(|h| super::alma_wrapper::DeviceArrayF32 { buf: h.buf, rows: h.rows, cols: h.cols })
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "wilders" => {
                let sweep = crate::indicators::moving_averages::wilders::WildersBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaWilders::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda
                    .wilders_batch_dev(ensure_prices!(), &sweep)
                    .map(|h| super::alma_wrapper::DeviceArrayF32 { buf: h.buf, rows: h.rows, cols: h.cols })
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "cwma" => {
                let sweep = crate::indicators::moving_averages::cwma::CwmaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaCwma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.cwma_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "jsa" => {
                let sweep = crate::indicators::moving_averages::jsa::JsaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaJsa::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.jsa_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "fwma" => {
                let sweep = crate::indicators::moving_averages::fwma::FwmaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaFwma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.fwma_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "hma" => {
                let sweep = crate::indicators::moving_averages::hma::HmaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaHma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                let cuda = CudaHma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                let (dev, _combos) = cuda
                    .hma_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                Ok(dev)
            }
            "srwma" => {
                let sweep = crate::indicators::moving_averages::srwma::SrwmaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaSrwma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda
                    .srwma_batch_dev(ensure_prices!(), &sweep)
                    .map(|h| super::alma_wrapper::DeviceArrayF32 { buf: h.buf, rows: h.rows, cols: h.cols })
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "sinwma" => {
                let sweep = crate::indicators::moving_averages::sinwma::SinWmaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaSinwma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.sinwma_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "sqwma" => {
                let sweep = crate::indicators::moving_averages::sqwma::SqwmaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaSqwma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.sqwma_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "swma" => {
                let sweep = crate::indicators::moving_averages::swma::SwmaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaSwma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.swma_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "linreg" => {
                let sweep = crate::indicators::moving_averages::linreg::LinRegBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaLinreg::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                let cuda = CudaLinreg::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                let (dev, _combos) = cuda
                    .linreg_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                Ok(dev)
            }
            "hwma" => {
                // Uses default smoothing params (na, nb, nc). Period is ignored.
                let sweep = crate::indicators::moving_averages::hwma::HwmaBatchRange::default();
                let cuda = CudaHwma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.hwma_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "edcf" => {
                let sweep = crate::indicators::moving_averages::edcf::EdcfBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaEdcf::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.edcf_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "dma" => {
                let sweep = crate::indicators::moving_averages::dma::DmaBatchRange {
                    hull_length: (period, period, 0),
                    ema_length: (20, 20, 0),
                    ema_gain_limit: (50, 50, 0),
                    hull_ma_type: "WMA".to_string(),
                };
                let cuda = CudaDma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.dma_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "highpass" => {
                let sweep = crate::indicators::moving_averages::highpass::HighPassBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaHighpass::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda
                    .highpass_batch_dev(ensure_prices!(), &sweep)
                    .map(|h| super::alma_wrapper::DeviceArrayF32 { buf: h.buf, rows: h.rows, cols: h.cols })
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "highpass2" | "highpass_2_pole" => {
                let sweep =
                    crate::indicators::moving_averages::highpass_2_pole::HighPass2BatchRange {
                        period: (period, period, 0),
                        k: (0.707, 0.707, 0.0),
                    };
                let sweep =
                    crate::indicators::moving_averages::highpass_2_pole::HighPass2BatchRange {
                        period: (period, period, 0),
                        k: (0.707, 0.707, 0.0),
                    };
                let cuda = CudaHighPass2::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.highpass2_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }

            // --- Price-only with extra params (use scalar defaults) ---
            "alma" => {
                let sweep = crate::indicators::moving_averages::alma::AlmaBatchRange {
                    period: (period, period, 0),
                    // Defaults: offset=0.85, sigma=6.0 (single-combo)
                    offset: (0.85, 0.85, 0.0),
                    sigma: (6.0, 6.0, 0.0),
                };
                let cuda = CudaAlma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.alma_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "epma" => {
                let sweep = crate::indicators::moving_averages::epma::EpmaBatchRange {
                    period: (period, period, 0),
                    // Default offset=4
                    offset: (4, 4, 0),
                };
                let cuda = CudaEpma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.epma_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "gaussian" => {
                let sweep = crate::indicators::moving_averages::gaussian::GaussianBatchRange {
                    period: (period, period, 0),
                    // Default poles=4
                    poles: (4, 4, 0),
                };
                let cuda = CudaGaussian::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.gaussian_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "jma" => {
                let sweep = crate::indicators::moving_averages::jma::JmaBatchRange {
                    period: (period, period, 0),
                    // Defaults: phase=50.0, power=2
                    phase: (50.0, 50.0, 0.0),
                    power: (2, 2, 0),
                };
                let cuda = CudaJma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda
                    .jma_batch_dev(ensure_prices!(), &sweep)
                    .map(|h| super::DeviceArrayF32 { buf: h.buf, rows: h.rows, cols: h.cols })
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "ehma" => {
                let sweep = crate::indicators::moving_averages::ehma::EhmaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaEhma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.ehma_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "supersmoother" => {
                let sweep =
                    crate::indicators::moving_averages::supersmoother::SuperSmootherBatchRange {
                        period: (period, period, 0),
                    };
                let sweep =
                    crate::indicators::moving_averages::supersmoother::SuperSmootherBatchRange {
                        period: (period, period, 0),
                    };
                let cuda = CudaSuperSmoother::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                let (dev, _combos) = cuda
                    .supersmoother_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                Ok(dev)
            }
            "supersmoother_3_pole" => {
                let sweep = crate::indicators::moving_averages::supersmoother_3_pole::SuperSmoother3PoleBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaSupersmoother3Pole::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.supersmoother_3_pole_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "kama" => {
                let sweep = crate::indicators::moving_averages::kama::KamaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaKama::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda
                    .kama_batch_dev(ensure_prices!(), &sweep)
                    .map(|h| super::DeviceArrayF32 { buf: h.buf, rows: h.rows, cols: h.cols })
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "sama" => {
                // Map period -> length; other params use defaults (maj=14, min=6)
                let sweep = crate::indicators::moving_averages::sama::SamaBatchRange {
                    length: (period, period, 0),
                    ..Default::default()
                };
                let cuda = CudaSama::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda
                    .sama_batch_dev(ensure_prices!(), &sweep)
                    .map(|h| super::alma_wrapper::DeviceArrayF32 { buf: h.buf, rows: h.rows, cols: h.cols })
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "ehlers_kama" => {
                let sweep = crate::indicators::moving_averages::ehlers_kama::EhlersKamaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaEhlersKama::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.ehlers_kama_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "ehlers_itrend" => {
                let sweep =
                    crate::indicators::moving_averages::ehlers_itrend::EhlersITrendBatchRange {
                        // Match ma.rs convention
                        warmup_bars: (20, 20, 0),
                        max_dc_period: (period, period, 0),
                    };
                let sweep =
                    crate::indicators::moving_averages::ehlers_itrend::EhlersITrendBatchRange {
                        // Match ma.rs convention
                        warmup_bars: (20, 20, 0),
                        max_dc_period: (period, period, 0),
                    };
                let cuda = CudaEhlersITrend::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.ehlers_itrend_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "ehlers_ecema" => {
                // Defaults: length=20, gain_limit=50, pine_compatible=false, confirmed_only=false
                let sweep =
                    crate::indicators::moving_averages::ehlers_ecema::EhlersEcemaBatchRange {
                        length: (period, period, 0),
                        gain_limit: (50, 50, 0),
                    };
                let params =
                    crate::indicators::moving_averages::ehlers_ecema::EhlersEcemaParams::default();
                let sweep =
                    crate::indicators::moving_averages::ehlers_ecema::EhlersEcemaBatchRange {
                        length: (period, period, 0),
                        gain_limit: (50, 50, 0),
                    };
                let params =
                    crate::indicators::moving_averages::ehlers_ecema::EhlersEcemaParams::default();
                let cuda = CudaEhlersEcema::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.ehlers_ecema_batch_dev(ensure_prices!(), &sweep, &params)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "nama" => {
                // Price-only NAMA path
                let sweep = crate::indicators::moving_averages::nama::NamaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaNama::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.nama_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "nma" => {
                let sweep = crate::indicators::moving_averages::nma::NmaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaNma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                let cuda = CudaNma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                let (dev, _combos) = cuda
                    .nma_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                Ok(dev)
            }
            "pwma" => {
                let sweep = crate::indicators::moving_averages::pwma::PwmaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaPwma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda
                    .pwma_batch_dev(ensure_prices!(), &sweep)
                    .map(|h| super::alma_wrapper::DeviceArrayF32 { buf: h.buf, rows: h.rows, cols: h.cols })
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "maaq" => {
                let sweep = crate::indicators::moving_averages::maaq::MaaqBatchRange {
                    period: (period, period, 0),
                    fast_period: (2, 2, 0),
                    slow_period: (30, 30, 0),
                };
                let cuda = CudaMaaq::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.maaq_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "mwdx" => {
                let sweep = crate::indicators::moving_averages::mwdx::MwdxBatchRange {
                    factor: (0.2, 0.2, 0.0),
                };
                let cuda = CudaMwdx::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda
                    .mwdx_batch_dev(ensure_prices!(), &sweep)
                    .map(|h| super::alma_wrapper::DeviceArrayF32 { buf: h.buf, rows: h.rows, cols: h.cols })
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "reflex" => {
                let sweep = crate::indicators::moving_averages::reflex::ReflexBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaReflex::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.reflex_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "volatility_adjusted_ma" | "vama" => {
                // Map base_period to `period` and keep a reasonable default vol_period (51)
                let sweep =
                    crate::indicators::moving_averages::volatility_adjusted_ma::VamaBatchRange {
                        base_period: (period, period, 0),
                        vol_period: (51, 51, 0),
                    };
                let cuda = CudaVama::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda
                    .vama_batch_dev(ensure_prices!(), &sweep)
                    .map(|h| super::alma_wrapper::DeviceArrayF32 { buf: h.buf, rows: h.rows, cols: h.cols })
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "trendflex" => {
                let sweep = crate::indicators::moving_averages::trendflex::TrendFlexBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaTrendflex::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                let (dev, _combos) = cuda
                    .trendflex_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                Ok(dev)
            }

            // --- Not supported in this thin selector ---
            // Require OHLC or volume, or produce multiple outputs.
            "frama" => Err(CudaMaSelectorError::Unsupported(
                "frama requires high/low/close; use CudaFrama directly".into(),
            )),
            // These are now handled earlier for Candles inputs
            "vwap" | "vwma" | "vpwma" => Err(CudaMaSelectorError::Unsupported(
                "requires candles; pass CudaMaData::Candles for VWAP/VWMA/VPWMA".into(),
            )),
            "volume_adjusted_ma" => Err(CudaMaSelectorError::Unsupported(
                "volume_adjusted_ma requires volume; use CudaVolumeAdjustedMa".into(),
            )),
            "tradjema" => Err(CudaMaSelectorError::Unsupported(
                "tradjema requires high/low/close; use CudaTradjema directly".into(),
            )),
            "uma" => Err(CudaMaSelectorError::Unsupported(
                "uma requires volume; use CudaUma directly".into(),
            )),
            "mama" => Err(CudaMaSelectorError::Unsupported(
                "mama returns dual outputs; use CudaMama and pick the series".into(),
            )),
            "ehlers_pma" => Err(CudaMaSelectorError::Unsupported(
                "ehlers_pma returns dual outputs; use CudaEhlersPma".into(),
            )),

            other => Err(CudaMaSelectorError::InvalidInput(format!(
                "unknown moving average type: {}",
                other
            ))),
        }
    }

    /// Copy back as f32 using pinned host memory + async copy on the selector's stream.
    pub fn ma_to_host_f32(
        &self,
        ma_type: &str,
        data: CudaMaData,
        period: usize,
    ) -> Result<Vec<f32>, CudaMaSelectorError> {
        let dev = self.ma_to_device(ma_type, data, period)?;
        debug_assert_eq!(dev.rows, 1);
        // Checked arithmetic to avoid overflow on large inputs
        let total = dev
            .rows
            .checked_mul(dev.cols)
            .ok_or_else(|| CudaMaSelectorError::InvalidInput("rows*cols overflow".into()))?;
        let mut pinned: LockedBuffer<f32> = unsafe { LockedBuffer::uninitialized(total) }?;
        // Prefer async D2H if the underlying context/stream permits; otherwise falls back
        unsafe {
            dev.buf
                .async_copy_to(pinned.as_mut_slice(), &self.stream)
                .map_err(CudaMaSelectorError::Cuda)?;
        }
        self.stream
            .synchronize()
            .map_err(CudaMaSelectorError::Cuda)?;
        Ok(pinned.to_vec())
    }

    /// Compute on device and stage results back to host as `Vec<f64>`.
    /// Output length matches input; warmup NaNs and semantics follow the
    /// underlying kernels.
    pub fn ma_to_host_f64(
        &self,
        ma_type: &str,
        data: CudaMaData,
        period: usize,
    ) -> Result<Vec<f64>, CudaMaSelectorError> {
        let out32 = self.ma_to_host_f32(ma_type, data, period)?;
        Ok(out32.into_iter().map(|v| v as f64).collect())
    }

    /// Optional: sweep many periods in one batched launch for price-only MAs.
    pub fn ma_sweep_to_device(
        &self,
        ma_type: &str,
        data: CudaMaData,
        start: usize,
        end: usize,
        step: usize,
    ) -> Result<DeviceArrayF32, CudaMaSelectorError> {
        // Hardened expansion semantics:
        // - step == 0 => treat as static (single value = start)
        // - support reversed bounds (start > end) by walking backwards
        // - error on empty expansion
        let periods: Vec<usize> = {
            if step == 0 || start == end {
                vec![start]
            } else if start < end {
                let s = step.max(1);
                (start..=end).step_by(s).collect()
            } else {
                // reversed
                let s = step.max(1);
                let mut v = Vec::new();
                let mut cur = start;
                while cur >= end {
                    v.push(cur);
                    if cur < s { break; }
                    cur = cur.saturating_sub(s);
                    if cur == usize::MAX { break; }
                }
                v
            }
        };
        if periods.is_empty() {
            return Err(CudaMaSelectorError::InvalidRange { start, end, step });
        }
        let is = |s: &str| ma_type.eq_ignore_ascii_case(s);
        let prices = data.to_prices_f32();
        // Optional VRAM preflight similar to other wrappers
        let rows = periods.len();
        let cols = prices.len();
        let elems = rows
            .checked_mul(cols)
            .ok_or_else(|| CudaMaSelectorError::InvalidInput("rows*cols overflow".into()))?;
        let bytes_out = elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaMaSelectorError::InvalidInput("byte size overflow".into()))?;
        let headroom = std::env::var("CUDA_MEM_HEADROOM").ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(64 * 1024 * 1024);
        will_fit(bytes_out, headroom)?;
        if is("ema") {
            let sweep = crate::indicators::moving_averages::ema::EmaBatchRange {
                // pass through original triple; per-indicator expansion is consistent
                period: (start, end, step),
            };
            let cuda = CudaEma::new(self.device_id)
                .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
            return cuda
                .ema_batch_dev(&prices, &sweep)
                .map_err(|e| CudaMaSelectorError::Backend(e.to_string()));
        }
        if is("sma") {
            let sweep = crate::indicators::moving_averages::sma::SmaBatchRange {
                period: (start, end, step),
            };
            let cuda = CudaSma::new(self.device_id)
                .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
            let (dev, _c) = cuda
                .sma_batch_dev(&prices, &sweep)
                .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
            return Ok(dev);
        }
        Err(CudaMaSelectorError::InvalidInput(format!(
            "ma_sweep_to_device unsupported for {}",
            ma_type
        )))
    }
}

// ---------------- Python interop (CAI v3 + DLPack v1.x) ----------------
#[cfg(all(feature = "python", feature = "cuda"))]
use pyo3::prelude::*;
#[cfg(all(feature = "python", feature = "cuda"))]
use pyo3::types::PyDict;
#[cfg(all(feature = "python", feature = "cuda"))]
use pyo3::exceptions::PyValueError;
#[cfg(all(feature = "python", feature = "cuda"))]
use numpy::PyReadonlyArray1;

#[cfg(all(feature = "python", feature = "cuda"))]
pub struct DeviceArrayF32Sel {
    pub buf: cust::memory::DeviceBuffer<f32>,
    pub rows: usize,
    pub cols: usize,
    pub ctx: Arc<Context>,
    pub device_id: u32,
}

#[cfg(all(feature = "python", feature = "cuda"))]
impl DeviceArrayF32Sel {
    #[inline]
    pub fn device_ptr(&self) -> u64 { self.buf.as_device_ptr().as_raw() as u64 }
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(module = "ta_indicators.cuda", unsendable)]
pub struct DeviceArrayF32PySel { inner: DeviceArrayF32Sel }

#[cfg(all(feature = "python", feature = "cuda"))]
#[pymethods]
impl DeviceArrayF32PySel {
    #[getter]
    fn __cuda_array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("shape", (self.inner.rows, self.inner.cols))?;
        d.set_item("typestr", "<f4")?;
        // Byte strides; even for contiguous arrays (CAI v3)
        let row_stride = self
            .inner
            .cols
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| PyValueError::new_err("byte stride overflow"))?;
        d.set_item("strides", (row_stride, std::mem::size_of::<f32>()))?;
        d.set_item("data", (self.inner.device_ptr() as usize, false))?;
        // Producer work is synchronized before return; omit stream
        d.set_item("version", 3)?;
        Ok(d)
    }

    fn __dlpack_device__(&self) -> (i32, i32) { (2, self.inner.device_id as i32) }

    #[pyo3(signature = (stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__<'py>(
        slf: pyo3::PyRef<'py, Self>,
        py: Python<'py>,
        _stream: Option<isize>,
        max_version: Option<(u32, u32)>,
        _dl_device: Option<(i32, i32)>,
        _copy: Option<bool>,
    ) -> PyResult<PyObject> {
        use pyo3::ffi as pyffi;
        use std::ffi::{c_void, CString};

        #[repr(C)]
        struct DLDevice { device_type: i32, device_id: i32 }
        #[repr(C)]
        struct DLDataType { code: u8, bits: u8, lanes: u16 }
        #[repr(C)]
        struct DLTensor {
            data: *mut c_void,
            device: DLDevice,
            ndim: i32,
            dtype: DLDataType,
            shape: *mut i64,
            strides: *mut i64,
            byte_offset: u64,
        }
        #[repr(C)]
        struct DLManagedTensor { dl_tensor: DLTensor, manager_ctx: *mut c_void, deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)> }
        #[repr(C)]
        struct DLManagedTensorVersioned { manager: *mut DLManagedTensor, version: u32 }

        #[repr(C)]
        struct ManagerCtx { shape: *mut i64, strides: *mut i64, _shape: Box<[i64; 2]>, _strides: Box<[i64; 2]>, _ctx_guard: Arc<Context>, _self_ref: pyo3::PyObject }

        unsafe extern "C" fn deleter(p: *mut DLManagedTensor) {
            if p.is_null() { return; }
            let mt = Box::from_raw(p);
            let ctx = mt.manager_ctx as *mut ManagerCtx;
            if !ctx.is_null() { let _ = Box::from_raw(ctx); }
        }

        unsafe extern "C" fn capsule_destructor(capsule: *mut pyffi::PyObject) {
            let name = b"dltensor\0";
            let ptr = pyffi::PyCapsule_GetPointer(capsule, name.as_ptr() as *const _) as *mut DLManagedTensor;
            if !ptr.is_null() {
                if let Some(del) = (*ptr).deleter { del(ptr); }
                let used = b"used_dltensor\0";
                let _ = pyffi::PyCapsule_SetName(capsule, used.as_ptr() as *const _);
            }
        }

        let rows = slf.inner.rows as i64;
        let cols = slf.inner.cols as i64;
        let mut shape = Box::new([rows, cols]);
        let mut strides = Box::new([cols, 1]); // elements per v1.2+
        let shape_ptr = shape.as_mut_ptr();
        let strides_ptr = strides.as_mut_ptr();
        let self_ref = unsafe { pyo3::PyObject::from_borrowed_ptr(py, slf.as_ptr()) };
        let mgr = Box::new(ManagerCtx { shape: shape_ptr, strides: strides_ptr, _shape: shape, _strides: strides, _ctx_guard: slf.inner.ctx.clone(), _self_ref: self_ref });
        let mgr_ptr = Box::into_raw(mgr) as *mut c_void;

        let data_ptr: *mut c_void = if rows == 0 || cols == 0 { std::ptr::null_mut() } else { slf.inner.device_ptr() as usize as *mut c_void };
        let tensor = DLTensor {
            data: data_ptr,
            device: DLDevice { device_type: 2, device_id: slf.inner.device_id as i32 },
            ndim: 2,
            dtype: DLDataType { code: 2, bits: 32, lanes: 1 },
            shape: shape_ptr,
            strides: strides_ptr,
            byte_offset: 0,
        };
        let mt = Box::new(DLManagedTensor { dl_tensor: tensor, manager_ctx: mgr_ptr, deleter: Some(deleter) });
        let want_versioned = max_version.map(|(maj, _)| maj >= 1).unwrap_or(false);
        unsafe {
            if want_versioned {
                let wrapped = Box::new(DLManagedTensorVersioned { manager: Box::into_raw(mt), version: 1 });
                let ptr = Box::into_raw(wrapped) as *mut c_void;
                let name = CString::new("dltensor_versioned").unwrap();
                let cap = pyffi::PyCapsule_New(ptr, name.as_ptr(), None);
                if cap.is_null() { let _ = Box::from_raw(ptr as *mut DLManagedTensorVersioned); return Err(pyo3::exceptions::PyRuntimeError::new_err("failed to create versioned DLPack capsule")); }
                Ok(pyo3::PyObject::from_owned_ptr(py, cap))
            } else {
                let ptr = Box::into_raw(mt) as *mut c_void;
                let name = CString::new("dltensor").unwrap();
                let cap = pyffi::PyCapsule_New(ptr, name.as_ptr(), Some(capsule_destructor));
                if cap.is_null() { let _ = Box::from_raw(ptr as *mut DLManagedTensor); return Err(pyo3::exceptions::PyRuntimeError::new_err("failed to create DLPack capsule")); }
                Ok(pyo3::PyObject::from_owned_ptr(py, cap))
            }
        }
    }
}

#[cfg(all(feature = "python", feature = "cuda"))]
fn not_empty_f32<'a>(arr: PyReadonlyArray1<'a, f32>) -> PyResult<&'a [f32]> {
    let s = arr.as_slice()?;
    if s.is_empty() { Err(PyValueError::new_err("empty data")) } else { Ok(s) }
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "ma_selector_cuda_to_device")] 
#[pyo3(signature = (ma_type, data, period, device_id=0))]
pub fn ma_selector_cuda_to_device_py(
    py: Python<'_>,
    ma_type: &str,
    data: PyReadonlyArray1<'_, f32>,
    period: usize,
    device_id: usize,
) -> PyResult<DeviceArrayF32PySel> {
    let prices = not_empty_f32(data)?;
    let is = |s: &str| ma_type.eq_ignore_ascii_case(s);
    let inner = py.allow_threads(|| -> Result<DeviceArrayF32Sel, String> {
        if is("sma") {
            let sweep = crate::indicators::moving_averages::sma::SmaBatchRange { period: (period, period, 0) };
            let cuda = CudaSma::new(device_id).map_err(|e| e.to_string())?;
            let ctx = cuda.context_arc_clone();
            let dev_id = cuda.device_id();
            let (dev, _c) = cuda.sma_batch_dev(prices, &sweep).map_err(|e| e.to_string())?;
            return Ok(DeviceArrayF32Sel { buf: dev.buf, rows: dev.rows, cols: dev.cols, ctx, device_id: dev_id });
        }
        if is("ema") {
            let sweep = crate::indicators::moving_averages::ema::EmaBatchRange { period: (period, period, 0) };
            let cuda = CudaEma::new(device_id).map_err(|e| e.to_string())?;
            let ctx = cuda.context_arc();
            let dev = cuda.ema_batch_dev(prices, &sweep).map_err(|e| e.to_string())?;
            let dev_id = device_id as u32;
            return Ok(DeviceArrayF32Sel { buf: dev.buf, rows: dev.rows, cols: dev.cols, ctx, device_id: dev_id });
        }
        Err(format!("unsupported MA type: {}", ma_type))
    }).map_err(PyValueError::new_err)?;
    Ok(DeviceArrayF32PySel { inner })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "ma_selector_cuda_sweep_to_device")]
#[pyo3(signature = (ma_type, data, period_range, device_id=0))]
pub fn ma_selector_cuda_sweep_to_device_py(
    py: Python<'_>,
    ma_type: &str,
    data: PyReadonlyArray1<'_, f32>,
    period_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<DeviceArrayF32PySel> {
    let prices = not_empty_f32(data)?;
    let is = |s: &str| ma_type.eq_ignore_ascii_case(s);
    let inner = py.allow_threads(|| -> Result<DeviceArrayF32Sel, String> {
        if is("sma") {
            let sweep = crate::indicators::moving_averages::sma::SmaBatchRange { period: period_range };
            let cuda = CudaSma::new(device_id).map_err(|e| e.to_string())?;
            let ctx = cuda.context_arc_clone();
            let dev_id = cuda.device_id();
            let (dev, _c) = cuda.sma_batch_dev(prices, &sweep).map_err(|e| e.to_string())?;
            return Ok(DeviceArrayF32Sel { buf: dev.buf, rows: dev.rows, cols: dev.cols, ctx, device_id: dev_id });
        }
        if is("ema") {
            let sweep = crate::indicators::moving_averages::ema::EmaBatchRange { period: period_range };
            let cuda = CudaEma::new(device_id).map_err(|e| e.to_string())?;
            let ctx = cuda.context_arc();
            let dev = cuda.ema_batch_dev(prices, &sweep).map_err(|e| e.to_string())?;
            let dev_id = device_id as u32;
            return Ok(DeviceArrayF32Sel { buf: dev.buf, rows: dev.rows, cols: dev.cols, ctx, device_id: dev_id });
        }
        Err(format!("ma_sweep_to_device unsupported for {}", ma_type))
    }).map_err(PyValueError::new_err)?;
    Ok(DeviceArrayF32PySel { inner })
}
