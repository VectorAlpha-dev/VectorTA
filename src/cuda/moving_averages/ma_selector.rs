//! Thin CUDA MA selector that mirrors the shape of `indicators::moving_averages::ma`.
//!
//! Dispatches to the existing CUDA wrappers by `ma_type` for the common
//! price-only moving averages where a single `period` is the only explicit
//! argument. Indicators that require additional inputs (e.g., OHLC, volume)
//! or return multiple outputs are reported as unsupported here — call their
//! dedicated CUDA wrappers directly.

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::cuda::moving_averages::*;
use crate::utilities::data_loader::{source_type, Candles};
// Faster D2H with pinned buffers; enable async copies when possible
use cust::memory::{AsyncCopyDestination, CopyDestination, LockedBuffer};
use cust::stream::{Stream, StreamFlags};

/// Unified error type for the CUDA MA selector.
#[derive(Debug)]
pub enum CudaMaSelectorError {
    /// Underlying CUDA wrapper error (stringified).
    Cuda(String),
    /// Invalid or unsupported inputs for this thin selector.
    InvalidInput(String),
    /// The requested MA requires extra inputs (e.g., OHLC/volume) or returns
    /// multiple outputs; use the indicator-specific CUDA wrapper instead.
    Unsupported(String),
}

impl std::fmt::Display for CudaMaSelectorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CudaMaSelectorError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaMaSelectorError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
            CudaMaSelectorError::Unsupported(e) => write!(f, "Unsupported: {}", e),
        }
    }
}

impl std::error::Error for CudaMaSelectorError {}

/// Input data for the selector (price-only).
#[derive(Debug, Clone, Copy)]
pub enum CudaMaData<'a> {
    /// Use a candle field as the price source (e.g., "close").
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
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
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .expect("failed to create CUDA stream");
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
            return Err(CudaMaSelectorError::InvalidInput("empty price input".into()));
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
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                return cuda
                    .vwma_batch_dev(&prices_f32, &volumes_f32, &sweep)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()));
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
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                let (dev, _combos) = cuda
                    .vpwma_batch_dev(&prices_f32, &sweep)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                return Ok(dev);
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
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                return cuda
                    .vwap_batch_dev(&candles.timestamp, &candles.volume, prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()));
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
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                let cuda = CudaSma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                let (dev, _combos) = cuda
                    .sma_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                Ok(dev)
            }
            "ema" => {
                let sweep = crate::indicators::moving_averages::ema::EmaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaEma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                cuda.ema_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))
            }
            "dema" => {
                let sweep = crate::indicators::moving_averages::dema::DemaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaDema::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                cuda.dema_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))
            }
            "wma" => {
                let sweep = crate::indicators::moving_averages::wma::WmaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaWma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                cuda.wma_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))
            }
            "zlema" => {
                let sweep = crate::indicators::moving_averages::zlema::ZlemaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaZlema::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                let cuda = CudaZlema::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                let (dev, _combos) = cuda
                    .zlema_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                Ok(dev)
            }
            "smma" => {
                let sweep = crate::indicators::moving_averages::smma::SmmaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaSmma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                cuda.smma_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))
            }
            "trima" => {
                let sweep = crate::indicators::moving_averages::trima::TrimaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaTrima::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                cuda.trima_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))
            }
            "tema" => {
                let sweep = crate::indicators::moving_averages::tema::TemaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaTema::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                cuda.tema_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))
            }
            "tilson" => {
                let sweep = crate::indicators::moving_averages::tilson::TilsonBatchRange {
                    period: (period, period, 0),
                    volume_factor: (0.0, 0.0, 0.0),
                };
                let cuda = CudaTilson::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                cuda.tilson_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))
            }
            "wilders" => {
                let sweep = crate::indicators::moving_averages::wilders::WildersBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaWilders::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                cuda.wilders_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))
            }
            "cwma" => {
                let sweep = crate::indicators::moving_averages::cwma::CwmaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaCwma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                cuda.cwma_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))
            }
            "jsa" => {
                let sweep = crate::indicators::moving_averages::jsa::JsaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaJsa::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                cuda.jsa_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))
            }
            "fwma" => {
                let sweep = crate::indicators::moving_averages::fwma::FwmaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaFwma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                cuda.fwma_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))
            }
            "hma" => {
                let sweep = crate::indicators::moving_averages::hma::HmaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaHma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                let cuda = CudaHma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                let (dev, _combos) = cuda
                    .hma_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                Ok(dev)
            }
            "srwma" => {
                let sweep = crate::indicators::moving_averages::srwma::SrwmaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaSrwma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                cuda.srwma_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))
            }
            "sinwma" => {
                let sweep = crate::indicators::moving_averages::sinwma::SinWmaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaSinwma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                cuda.sinwma_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))
            }
            "sqwma" => {
                let sweep = crate::indicators::moving_averages::sqwma::SqwmaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaSqwma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                cuda.sqwma_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))
            }
            "swma" => {
                let sweep = crate::indicators::moving_averages::swma::SwmaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaSwma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                cuda.swma_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))
            }
            "linreg" => {
                let sweep = crate::indicators::moving_averages::linreg::LinRegBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaLinreg::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                let cuda = CudaLinreg::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                let (dev, _combos) = cuda
                    .linreg_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                Ok(dev)
            }
            "hwma" => {
                // Uses default smoothing params (na, nb, nc). Period is ignored.
                let sweep = crate::indicators::moving_averages::hwma::HwmaBatchRange::default();
                let cuda = CudaHwma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                cuda.hwma_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))
            }
            "edcf" => {
                let sweep = crate::indicators::moving_averages::edcf::EdcfBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaEdcf::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                cuda.edcf_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))
            }
            "dma" => {
                let sweep = crate::indicators::moving_averages::dma::DmaBatchRange {
                    hull_length: (period, period, 0),
                    ema_length: (20, 20, 0),
                    ema_gain_limit: (50, 50, 0),
                    hull_ma_type: "WMA".to_string(),
                };
                let cuda = CudaDma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                cuda.dma_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))
            }
            "highpass" => {
                let sweep = crate::indicators::moving_averages::highpass::HighPassBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaHighpass::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                cuda.highpass_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))
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
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                cuda.highpass2_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))
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
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                cuda.alma_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))
            }
            "epma" => {
                let sweep = crate::indicators::moving_averages::epma::EpmaBatchRange {
                    period: (period, period, 0),
                    // Default offset=4
                    offset: (4, 4, 0),
                };
                let cuda = CudaEpma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                cuda.epma_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))
            }
            "gaussian" => {
                let sweep = crate::indicators::moving_averages::gaussian::GaussianBatchRange {
                    period: (period, period, 0),
                    // Default poles=4
                    poles: (4, 4, 0),
                };
                let cuda = CudaGaussian::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                cuda.gaussian_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))
            }
            "jma" => {
                let sweep = crate::indicators::moving_averages::jma::JmaBatchRange {
                    period: (period, period, 0),
                    // Defaults: phase=50.0, power=2
                    phase: (50.0, 50.0, 0.0),
                    power: (2, 2, 0),
                };
                let cuda = CudaJma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                cuda.jma_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))
            }
            "ehma" => {
                let sweep = crate::indicators::moving_averages::ehma::EhmaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaEhma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                cuda.ehma_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))
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
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                let (dev, _combos) = cuda
                    .supersmoother_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                Ok(dev)
            }
            "supersmoother_3_pole" => {
                let sweep = crate::indicators::moving_averages::supersmoother_3_pole::SuperSmoother3PoleBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaSupersmoother3Pole::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                cuda.supersmoother_3_pole_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))
            }
            "kama" => {
                let sweep = crate::indicators::moving_averages::kama::KamaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaKama::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                cuda.kama_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))
            }
            "sama" => {
                // Map period -> length; other params use defaults (maj=14, min=6)
                let sweep = crate::indicators::moving_averages::sama::SamaBatchRange {
                    length: (period, period, 0),
                    ..Default::default()
                };
                let cuda = CudaSama::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                cuda.sama_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))
            }
            "ehlers_kama" => {
                let sweep = crate::indicators::moving_averages::ehlers_kama::EhlersKamaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaEhlersKama::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                cuda.ehlers_kama_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))
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
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                cuda.ehlers_itrend_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))
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
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                cuda.ehlers_ecema_batch_dev(ensure_prices!(), &sweep, &params)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))
            }
            "nama" => {
                // Price-only NAMA path
                let sweep = crate::indicators::moving_averages::nama::NamaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaNama::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                cuda.nama_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))
            }
            "nma" => {
                let sweep = crate::indicators::moving_averages::nma::NmaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaNma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                let cuda = CudaNma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                let (dev, _combos) = cuda
                    .nma_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                Ok(dev)
            }
            "pwma" => {
                let sweep = crate::indicators::moving_averages::pwma::PwmaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaPwma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                cuda.pwma_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))
            }
            "maaq" => {
                let sweep = crate::indicators::moving_averages::maaq::MaaqBatchRange {
                    period: (period, period, 0),
                    fast_period: (2, 2, 0),
                    slow_period: (30, 30, 0),
                };
                let cuda = CudaMaaq::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                cuda.maaq_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))
            }
            "mwdx" => {
                let sweep = crate::indicators::moving_averages::mwdx::MwdxBatchRange {
                    factor: (0.2, 0.2, 0.0),
                };
                let cuda = CudaMwdx::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                cuda.mwdx_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))
            }
            "reflex" => {
                let sweep = crate::indicators::moving_averages::reflex::ReflexBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaReflex::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                cuda.reflex_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))
            }
            "volatility_adjusted_ma" | "vama" => {
                // Map base_period to `period` and keep a reasonable default vol_period (51)
                let sweep =
                    crate::indicators::moving_averages::volatility_adjusted_ma::VamaBatchRange {
                        base_period: (period, period, 0),
                        vol_period: (51, 51, 0),
                    };
                let cuda = CudaVama::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                cuda.vama_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))
            }
            "trendflex" => {
                let sweep = crate::indicators::moving_averages::trendflex::TrendFlexBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaTrendflex::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
                let (dev, _combos) = cuda
                    .trendflex_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
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
        let total = dev.rows * dev.cols;
        let mut pinned: LockedBuffer<f32> =
            unsafe { LockedBuffer::uninitialized(total) }
                .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
        // Prefer async D2H if the underlying context/stream permits; otherwise falls back
        unsafe {
            dev.buf
                .async_copy_to(pinned.as_mut_slice(), &self.stream)
                .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
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
        if start == 0 || start > end || step == 0 {
            return Err(CudaMaSelectorError::InvalidInput(
                "invalid (start,end,step)".into(),
            ));
        }
        let is = |s: &str| ma_type.eq_ignore_ascii_case(s);
        let prices = data.to_prices_f32();
        if is("ema") {
            let sweep = crate::indicators::moving_averages::ema::EmaBatchRange {
                period: (start, end, step),
            };
            let cuda = CudaEma::new(self.device_id)
                .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
            return cuda
                .ema_batch_dev(&prices, &sweep)
                .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()));
        }
        if is("sma") {
            let sweep = crate::indicators::moving_averages::sma::SmaBatchRange {
                period: (start, end, step),
            };
            let cuda = CudaSma::new(self.device_id)
                .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
            let (dev, _c) = cuda
                .sma_batch_dev(&prices, &sweep)
                .map_err(|e| CudaMaSelectorError::Cuda(e.to_string()))?;
            return Ok(dev);
        }
        Err(CudaMaSelectorError::InvalidInput(format!(
            "ma_sweep_to_device unsupported for {}",
            ma_type
        )))
    }
}
