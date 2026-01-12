#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::cuda::moving_averages::*;
use crate::utilities::data_loader::{source_type, Candles};

use cust::context::Context;
use cust::memory::{mem_get_info, AsyncCopyDestination, CopyDestination, LockedBuffer};
use cust::stream::{Stream, StreamFlags};
use std::sync::Arc;
use thiserror::Error;

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
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("out of memory: required={required}B free={free}B headroom={headroom}B")]
    OutOfMemory {
        required: usize,
        free: usize,
        headroom: usize,
    },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge {
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
    },
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
fn device_mem_info() -> Option<(usize, usize)> {
    mem_get_info().ok()
}

#[inline]
fn will_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaMaSelectorError> {
    if !mem_check_enabled() {
        return Ok(());
    }
    if let Some((free, _total)) = device_mem_info() {
        if required_bytes.saturating_add(headroom_bytes) <= free {
            Ok(())
        } else {
            Err(CudaMaSelectorError::OutOfMemory {
                required: required_bytes,
                free,
                headroom: headroom_bytes,
            })
        }
    } else {
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub enum CudaMaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },

    Slice(&'a [f64]),

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

pub struct CudaMaSelector {
    device_id: usize,

    stream: Stream,
}

impl CudaMaSelector {
    pub fn new(device_id: usize) -> Self {
        let stream =
            Stream::new(StreamFlags::NON_BLOCKING, None).expect("failed to create CUDA stream");
        Self { device_id, stream }
    }

    pub fn ma_to_device(
        &self,
        ma_type: &str,
        data: CudaMaData,
        period: usize,
    ) -> Result<DeviceArrayF32, CudaMaSelectorError> {
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

        let is = |s: &str| ma_type.eq_ignore_ascii_case(s);

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
                    power: (0.382, 0.382, 0.0),
                };
                let cuda = CudaVpwma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                let (dev, _combos) = cuda
                    .vpwma_batch_dev(&prices_f32, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                return Ok(super::DeviceArrayF32 {
                    buf: dev.buf,
                    rows: dev.rows,
                    cols: dev.cols,
                });
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
                cuda.smma_batch_dev(ensure_prices!(), &sweep)
                    .map(|h| super::alma_wrapper::DeviceArrayF32 {
                        buf: h.buf,
                        rows: h.rows,
                        cols: h.cols,
                    })
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "trima" => {
                let sweep = crate::indicators::moving_averages::trima::TrimaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaTrima::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.trima_batch_dev(ensure_prices!(), &sweep)
                    .map(|h| super::alma_wrapper::DeviceArrayF32 {
                        buf: h.buf,
                        rows: h.rows,
                        cols: h.cols,
                    })
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
                cuda.tilson_batch_dev(ensure_prices!(), &sweep)
                    .map(|h| super::alma_wrapper::DeviceArrayF32 {
                        buf: h.buf,
                        rows: h.rows,
                        cols: h.cols,
                    })
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "wilders" => {
                let sweep = crate::indicators::moving_averages::wilders::WildersBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaWilders::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.wilders_batch_dev(ensure_prices!(), &sweep)
                    .map(|h| super::alma_wrapper::DeviceArrayF32 {
                        buf: h.buf,
                        rows: h.rows,
                        cols: h.cols,
                    })
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
                cuda.srwma_batch_dev(ensure_prices!(), &sweep)
                    .map(|h| super::alma_wrapper::DeviceArrayF32 {
                        buf: h.buf,
                        rows: h.rows,
                        cols: h.cols,
                    })
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
                cuda.highpass_batch_dev(ensure_prices!(), &sweep)
                    .map(|h| super::alma_wrapper::DeviceArrayF32 {
                        buf: h.buf,
                        rows: h.rows,
                        cols: h.cols,
                    })
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

            "alma" => {
                let sweep = crate::indicators::moving_averages::alma::AlmaBatchRange {
                    period: (period, period, 0),

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

                    phase: (50.0, 50.0, 0.0),
                    power: (2, 2, 0),
                };
                let cuda = CudaJma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.jma_batch_dev(ensure_prices!(), &sweep)
                    .map(|h| super::DeviceArrayF32 {
                        buf: h.buf,
                        rows: h.rows,
                        cols: h.cols,
                    })
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
                cuda.kama_batch_dev(ensure_prices!(), &sweep)
                    .map(|h| super::DeviceArrayF32 {
                        buf: h.buf,
                        rows: h.rows,
                        cols: h.cols,
                    })
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "sama" => {
                let sweep = crate::indicators::moving_averages::sama::SamaBatchRange {
                    length: (period, period, 0),
                    ..Default::default()
                };
                let cuda = CudaSama::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.sama_batch_dev(ensure_prices!(), &sweep)
                    .map(|h| super::alma_wrapper::DeviceArrayF32 {
                        buf: h.buf,
                        rows: h.rows,
                        cols: h.cols,
                    })
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
                        warmup_bars: (20, 20, 0),
                        max_dc_period: (period, period, 0),
                    };
                let sweep =
                    crate::indicators::moving_averages::ehlers_itrend::EhlersITrendBatchRange {
                        warmup_bars: (20, 20, 0),
                        max_dc_period: (period, period, 0),
                    };
                let cuda = CudaEhlersITrend::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.ehlers_itrend_batch_dev(ensure_prices!(), &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "ehlers_ecema" => {
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
                cuda.pwma_batch_dev(ensure_prices!(), &sweep)
                    .map(|h| super::alma_wrapper::DeviceArrayF32 {
                        buf: h.buf,
                        rows: h.rows,
                        cols: h.cols,
                    })
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
                cuda.mwdx_batch_dev(ensure_prices!(), &sweep)
                    .map(|h| super::alma_wrapper::DeviceArrayF32 {
                        buf: h.buf,
                        rows: h.rows,
                        cols: h.cols,
                    })
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
                let sweep =
                    crate::indicators::moving_averages::volatility_adjusted_ma::VamaBatchRange {
                        base_period: (period, period, 0),
                        vol_period: (51, 51, 0),
                    };
                let cuda = CudaVama::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.vama_batch_dev(ensure_prices!(), &sweep)
                    .map(|h| super::alma_wrapper::DeviceArrayF32 {
                        buf: h.buf,
                        rows: h.rows,
                        cols: h.cols,
                    })
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

            "frama" => Err(CudaMaSelectorError::Unsupported(
                "frama requires high/low/close; use CudaFrama directly".into(),
            )),

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

    pub fn ma_to_host_f32(
        &self,
        ma_type: &str,
        data: CudaMaData,
        period: usize,
    ) -> Result<Vec<f32>, CudaMaSelectorError> {
        let dev = self.ma_to_device(ma_type, data, period)?;
        debug_assert_eq!(dev.rows, 1);

        let total = dev
            .rows
            .checked_mul(dev.cols)
            .ok_or_else(|| CudaMaSelectorError::InvalidInput("rows*cols overflow".into()))?;
        let mut pinned: LockedBuffer<f32> = unsafe { LockedBuffer::uninitialized(total) }?;

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

    pub fn ma_to_host_f64(
        &self,
        ma_type: &str,
        data: CudaMaData,
        period: usize,
    ) -> Result<Vec<f64>, CudaMaSelectorError> {
        let out32 = self.ma_to_host_f32(ma_type, data, period)?;
        Ok(out32.into_iter().map(|v| v as f64).collect())
    }

    pub fn ma_sweep_to_device(
        &self,
        ma_type: &str,
        data: CudaMaData,
        start: usize,
        end: usize,
        step: usize,
    ) -> Result<DeviceArrayF32, CudaMaSelectorError> {
        let periods: Vec<usize> = if step == 0 || start == end {
            vec![start]
        } else if start < end {
            let s = step.max(1);
            (start..=end).step_by(s).collect()
        } else {
            let s = step.max(1);
            let mut v = Vec::new();
            let mut cur = start;
            while cur >= end {
                v.push(cur);
                if cur == 0 {
                    break;
                }
                let next = cur.saturating_sub(s);
                if next == cur {
                    break;
                }
                cur = next;
                if cur < end {
                    break;
                }
            }
            v
        };
        if periods.is_empty() {
            return Err(CudaMaSelectorError::InvalidRange { start, end, step });
        }

        let ma_lc = ma_type.to_ascii_lowercase();

        let prices = data.to_prices_f32();
        if prices.is_empty() {
            return Err(CudaMaSelectorError::InvalidInput(
                "empty price input".into(),
            ));
        }
        let period_range = (start, end, step);

        let rows = periods.len();
        let cols = prices.len();
        let elems = rows
            .checked_mul(cols)
            .ok_or_else(|| CudaMaSelectorError::InvalidInput("rows*cols overflow".into()))?;
        let bytes_out = elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaMaSelectorError::InvalidInput("byte size overflow".into()))?;
        let headroom = std::env::var("CUDA_MEM_HEADROOM")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(64 * 1024 * 1024);
        will_fit(bytes_out, headroom)?;

        match ma_lc.as_str() {
            "vwma" => {
                let candles = match data {
                    CudaMaData::Candles { candles, .. } => candles,
                    _ => {
                        return Err(CudaMaSelectorError::Unsupported(
                            "vwma requires candles with volume; pass CudaMaData::Candles".into(),
                        ));
                    }
                };
                let volumes_f32: Vec<f32> = candles.volume.iter().map(|&v| v as f32).collect();
                let sweep = crate::indicators::moving_averages::vwma::VwmaBatchRange {
                    period: period_range,
                };
                let cuda = CudaVwma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.vwma_batch_dev(&prices, &volumes_f32, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "vpwma" => {
                match data {
                    CudaMaData::Candles { .. } => {}
                    _ => {
                        return Err(CudaMaSelectorError::Unsupported(
                            "vpwma requires candles; pass CudaMaData::Candles".into(),
                        ));
                    }
                }
                let sweep = crate::indicators::moving_averages::vpwma::VpwmaBatchRange {
                    period: period_range,
                    power: (0.382, 0.382, 0.0),
                };
                let cuda = CudaVpwma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                let (dev, _combos) = cuda
                    .vpwma_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                Ok(DeviceArrayF32 {
                    buf: dev.buf,
                    rows: dev.rows,
                    cols: dev.cols,
                })
            }

            "sma" => {
                let sweep = crate::indicators::moving_averages::sma::SmaBatchRange {
                    period: period_range,
                };
                let cuda = CudaSma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                let (dev, _combos) = cuda
                    .sma_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                Ok(dev)
            }
            "ema" => {
                let sweep = crate::indicators::moving_averages::ema::EmaBatchRange {
                    period: period_range,
                };
                let cuda = CudaEma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.ema_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "dema" => {
                let sweep = crate::indicators::moving_averages::dema::DemaBatchRange {
                    period: period_range,
                };
                let cuda = CudaDema::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.dema_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "wma" => {
                let sweep = crate::indicators::moving_averages::wma::WmaBatchRange {
                    period: period_range,
                };
                let cuda = CudaWma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.wma_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "zlema" => {
                let sweep = crate::indicators::moving_averages::zlema::ZlemaBatchRange {
                    period: period_range,
                };
                let cuda = CudaZlema::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                let (dev, _combos) = cuda
                    .zlema_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                Ok(dev)
            }
            "smma" => {
                let sweep = crate::indicators::moving_averages::smma::SmmaBatchRange {
                    period: period_range,
                };
                let cuda = CudaSmma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.smma_batch_dev(&prices, &sweep)
                    .map(|h| DeviceArrayF32 {
                        buf: h.buf,
                        rows: h.rows,
                        cols: h.cols,
                    })
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "trima" => {
                let sweep = crate::indicators::moving_averages::trima::TrimaBatchRange {
                    period: period_range,
                };
                let cuda = CudaTrima::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.trima_batch_dev(&prices, &sweep)
                    .map(|h| DeviceArrayF32 {
                        buf: h.buf,
                        rows: h.rows,
                        cols: h.cols,
                    })
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "tema" => {
                let sweep = crate::indicators::moving_averages::tema::TemaBatchRange {
                    period: period_range,
                };
                let cuda = CudaTema::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.tema_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "tilson" => {
                let sweep = crate::indicators::moving_averages::tilson::TilsonBatchRange {
                    period: period_range,
                    volume_factor: (0.0, 0.0, 0.0),
                };
                let cuda = CudaTilson::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.tilson_batch_dev(&prices, &sweep)
                    .map(|h| DeviceArrayF32 {
                        buf: h.buf,
                        rows: h.rows,
                        cols: h.cols,
                    })
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "wilders" => {
                let sweep = crate::indicators::moving_averages::wilders::WildersBatchRange {
                    period: period_range,
                };
                let cuda = CudaWilders::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.wilders_batch_dev(&prices, &sweep)
                    .map(|h| DeviceArrayF32 {
                        buf: h.buf,
                        rows: h.rows,
                        cols: h.cols,
                    })
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "cwma" => {
                let sweep = crate::indicators::moving_averages::cwma::CwmaBatchRange {
                    period: period_range,
                };
                let cuda = CudaCwma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.cwma_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "jsa" => {
                let sweep = crate::indicators::moving_averages::jsa::JsaBatchRange {
                    period: period_range,
                };
                let cuda = CudaJsa::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.jsa_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "fwma" => {
                let sweep = crate::indicators::moving_averages::fwma::FwmaBatchRange {
                    period: period_range,
                };
                let cuda = CudaFwma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.fwma_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "hma" => {
                let sweep = crate::indicators::moving_averages::hma::HmaBatchRange {
                    period: period_range,
                };
                let cuda = CudaHma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                let (dev, _combos) = cuda
                    .hma_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                Ok(dev)
            }
            "srwma" => {
                let sweep = crate::indicators::moving_averages::srwma::SrwmaBatchRange {
                    period: period_range,
                };
                let cuda = CudaSrwma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.srwma_batch_dev(&prices, &sweep)
                    .map(|h| DeviceArrayF32 {
                        buf: h.buf,
                        rows: h.rows,
                        cols: h.cols,
                    })
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "sinwma" => {
                let sweep = crate::indicators::moving_averages::sinwma::SinWmaBatchRange {
                    period: period_range,
                };
                let cuda = CudaSinwma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.sinwma_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "sqwma" => {
                let sweep = crate::indicators::moving_averages::sqwma::SqwmaBatchRange {
                    period: period_range,
                };
                let cuda = CudaSqwma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.sqwma_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "swma" => {
                let sweep = crate::indicators::moving_averages::swma::SwmaBatchRange {
                    period: period_range,
                };
                let cuda = CudaSwma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.swma_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "linreg" => {
                let sweep = crate::indicators::moving_averages::linreg::LinRegBatchRange {
                    period: period_range,
                };
                let cuda = CudaLinreg::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                let (dev, _combos) = cuda
                    .linreg_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                Ok(dev)
            }
            "edcf" => {
                let sweep = crate::indicators::moving_averages::edcf::EdcfBatchRange {
                    period: period_range,
                };
                let cuda = CudaEdcf::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.edcf_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "dma" => {
                let sweep = crate::indicators::moving_averages::dma::DmaBatchRange {
                    hull_length: period_range,
                    ema_length: (20, 20, 0),
                    ema_gain_limit: (50, 50, 0),
                    hull_ma_type: "WMA".to_string(),
                };
                let cuda = CudaDma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.dma_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "highpass" => {
                let sweep = crate::indicators::moving_averages::highpass::HighPassBatchRange {
                    period: period_range,
                };
                let cuda = CudaHighpass::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.highpass_batch_dev(&prices, &sweep)
                    .map(|h| DeviceArrayF32 {
                        buf: h.buf,
                        rows: h.rows,
                        cols: h.cols,
                    })
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "highpass2" | "highpass_2_pole" => {
                let sweep =
                    crate::indicators::moving_averages::highpass_2_pole::HighPass2BatchRange {
                        period: period_range,
                        k: (0.707, 0.707, 0.0),
                    };
                let cuda = CudaHighPass2::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.highpass2_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "alma" => {
                let sweep = crate::indicators::moving_averages::alma::AlmaBatchRange {
                    period: period_range,
                    offset: (0.85, 0.85, 0.0),
                    sigma: (6.0, 6.0, 0.0),
                };
                let cuda = CudaAlma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.alma_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "epma" => {
                let sweep = crate::indicators::moving_averages::epma::EpmaBatchRange {
                    period: period_range,
                    offset: (4, 4, 0),
                };
                let cuda = CudaEpma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.epma_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "gaussian" => {
                let sweep = crate::indicators::moving_averages::gaussian::GaussianBatchRange {
                    period: period_range,
                    poles: (4, 4, 0),
                };
                let cuda = CudaGaussian::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.gaussian_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "jma" => {
                let sweep = crate::indicators::moving_averages::jma::JmaBatchRange {
                    period: period_range,
                    phase: (50.0, 50.0, 0.0),
                    power: (2, 2, 0),
                };
                let cuda = CudaJma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.jma_batch_dev(&prices, &sweep)
                    .map(|h| DeviceArrayF32 {
                        buf: h.buf,
                        rows: h.rows,
                        cols: h.cols,
                    })
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "ehma" => {
                let sweep = crate::indicators::moving_averages::ehma::EhmaBatchRange {
                    period: period_range,
                };
                let cuda = CudaEhma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.ehma_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "supersmoother" => {
                let sweep =
                    crate::indicators::moving_averages::supersmoother::SuperSmootherBatchRange {
                        period: period_range,
                    };
                let cuda = CudaSuperSmoother::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                let (dev, _combos) = cuda
                    .supersmoother_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                Ok(dev)
            }
            "supersmoother_3_pole" => {
                let sweep = crate::indicators::moving_averages::supersmoother_3_pole::SuperSmoother3PoleBatchRange { period: period_range };
                let cuda = CudaSupersmoother3Pole::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.supersmoother_3_pole_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "kama" => {
                let sweep = crate::indicators::moving_averages::kama::KamaBatchRange {
                    period: period_range,
                };
                let cuda = CudaKama::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.kama_batch_dev(&prices, &sweep)
                    .map(|h| DeviceArrayF32 {
                        buf: h.buf,
                        rows: h.rows,
                        cols: h.cols,
                    })
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "sama" => {
                let sweep = crate::indicators::moving_averages::sama::SamaBatchRange {
                    length: period_range,
                    ..Default::default()
                };
                let cuda = CudaSama::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.sama_batch_dev(&prices, &sweep)
                    .map(|h| DeviceArrayF32 {
                        buf: h.buf,
                        rows: h.rows,
                        cols: h.cols,
                    })
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "ehlers_kama" => {
                let sweep = crate::indicators::moving_averages::ehlers_kama::EhlersKamaBatchRange {
                    period: period_range,
                };
                let cuda = CudaEhlersKama::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.ehlers_kama_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "ehlers_itrend" => {
                let sweep =
                    crate::indicators::moving_averages::ehlers_itrend::EhlersITrendBatchRange {
                        warmup_bars: (20, 20, 0),
                        max_dc_period: period_range,
                    };
                let cuda = CudaEhlersITrend::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.ehlers_itrend_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "ehlers_ecema" => {
                let sweep =
                    crate::indicators::moving_averages::ehlers_ecema::EhlersEcemaBatchRange {
                        length: period_range,
                        gain_limit: (50, 50, 0),
                    };
                let params =
                    crate::indicators::moving_averages::ehlers_ecema::EhlersEcemaParams::default();
                let cuda = CudaEhlersEcema::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.ehlers_ecema_batch_dev(&prices, &sweep, &params)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "nama" => {
                let sweep = crate::indicators::moving_averages::nama::NamaBatchRange {
                    period: period_range,
                };
                let cuda = CudaNama::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.nama_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "nma" => {
                let sweep = crate::indicators::moving_averages::nma::NmaBatchRange {
                    period: period_range,
                };
                let cuda = CudaNma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                let (dev, _combos) = cuda
                    .nma_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                Ok(dev)
            }
            "pwma" => {
                let sweep = crate::indicators::moving_averages::pwma::PwmaBatchRange {
                    period: period_range,
                };
                let cuda = CudaPwma::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.pwma_batch_dev(&prices, &sweep)
                    .map(|h| DeviceArrayF32 {
                        buf: h.buf,
                        rows: h.rows,
                        cols: h.cols,
                    })
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "maaq" => {
                let sweep = crate::indicators::moving_averages::maaq::MaaqBatchRange {
                    period: period_range,
                    fast_period: (2, 2, 0),
                    slow_period: (30, 30, 0),
                };
                let cuda = CudaMaaq::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.maaq_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "reflex" => {
                let sweep = crate::indicators::moving_averages::reflex::ReflexBatchRange {
                    period: period_range,
                };
                let cuda = CudaReflex::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.reflex_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "volatility_adjusted_ma" | "vama" => {
                let sweep =
                    crate::indicators::moving_averages::volatility_adjusted_ma::VamaBatchRange {
                        base_period: period_range,
                        vol_period: (51, 51, 0),
                    };
                let cuda = CudaVama::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                cuda.vama_batch_dev(&prices, &sweep)
                    .map(|h| DeviceArrayF32 {
                        buf: h.buf,
                        rows: h.rows,
                        cols: h.cols,
                    })
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))
            }
            "trendflex" => {
                let sweep = crate::indicators::moving_averages::trendflex::TrendFlexBatchRange {
                    period: period_range,
                };
                let cuda = CudaTrendflex::new(self.device_id)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                let (dev, _combos) = cuda
                    .trendflex_batch_dev(&prices, &sweep)
                    .map_err(|e| CudaMaSelectorError::Backend(e.to_string()))?;
                Ok(dev)
            }

            "vwap" | "hwma" | "mwdx" | "mama" => Err(CudaMaSelectorError::Unsupported(format!(
                "{ma_type} does not support period-sweep batching"
            ))),

            "frama" => Err(CudaMaSelectorError::Unsupported(
                "frama requires high/low/close; use CudaFrama directly".into(),
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
            "ehlers_pma" => Err(CudaMaSelectorError::Unsupported(
                "ehlers_pma returns dual outputs; use CudaEhlersPma".into(),
            )),

            other => Err(CudaMaSelectorError::InvalidInput(format!(
                "ma_sweep_to_device unsupported for {}",
                other
            ))),
        }
    }

    pub fn ma_sweep_to_host_f32(
        &self,
        ma_type: &str,
        data: CudaMaData,
        start: usize,
        end: usize,
        step: usize,
    ) -> Result<(Vec<f32>, usize, usize), CudaMaSelectorError> {
        let dev = self.ma_sweep_to_device(ma_type, data, start, end, step)?;
        let total = dev
            .rows
            .checked_mul(dev.cols)
            .ok_or_else(|| CudaMaSelectorError::InvalidInput("rows*cols overflow".into()))?;
        let mut pinned: LockedBuffer<f32> = unsafe { LockedBuffer::uninitialized(total) }?;
        unsafe {
            dev.buf
                .async_copy_to(pinned.as_mut_slice(), &self.stream)
                .map_err(CudaMaSelectorError::Cuda)?;
        }
        self.stream
            .synchronize()
            .map_err(CudaMaSelectorError::Cuda)?;
        Ok((pinned.to_vec(), dev.rows, dev.cols))
    }

    pub fn ma_sweep_to_host_f64(
        &self,
        ma_type: &str,
        data: CudaMaData,
        start: usize,
        end: usize,
        step: usize,
    ) -> Result<(Vec<f64>, usize, usize), CudaMaSelectorError> {
        let (out32, rows, cols) = self.ma_sweep_to_host_f32(ma_type, data, start, end, step)?;
        Ok((out32.into_iter().map(|v| v as f64).collect(), rows, cols))
    }
}

#[cfg(all(feature = "python", feature = "cuda"))]
use numpy::PyReadonlyArray1;
#[cfg(all(feature = "python", feature = "cuda"))]
use pyo3::exceptions::PyValueError;
#[cfg(all(feature = "python", feature = "cuda"))]
use pyo3::prelude::*;
#[cfg(all(feature = "python", feature = "cuda"))]
use pyo3::types::PyDict;

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
    pub fn device_ptr(&self) -> u64 {
        self.buf.as_device_ptr().as_raw() as u64
    }
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(module = "ta_indicators.cuda", unsendable)]
pub struct DeviceArrayF32PySel {
    inner: Option<DeviceArrayF32Sel>,
    device_id: u32,
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pymethods]
impl DeviceArrayF32PySel {
    #[getter]
    fn __cuda_array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let inner = self
            .inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("buffer already exported via __dlpack__"))?;
        let d = PyDict::new(py);
        d.set_item("shape", (inner.rows, inner.cols))?;
        d.set_item("typestr", "<f4")?;

        let row_stride = inner
            .cols
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| PyValueError::new_err("byte stride overflow"))?;
        d.set_item("strides", (row_stride, std::mem::size_of::<f32>()))?;
        d.set_item("data", (inner.device_ptr() as usize, false))?;

        d.set_item("version", 3)?;
        Ok(d)
    }

    fn __dlpack_device__(&self) -> (i32, i32) {
        (2, self.device_id as i32)
    }

    #[pyo3(signature = (stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__<'py>(
        &mut self,
        py: Python<'py>,
        stream: Option<PyObject>,
        max_version: Option<PyObject>,
        dl_device: Option<PyObject>,
        copy: Option<PyObject>,
    ) -> PyResult<PyObject> {
        use crate::utilities::dlpack_cuda::export_f32_cuda_dlpack_2d;

        let (kdl, alloc_dev) = self.__dlpack_device__();
        if let Some(dev_obj) = dl_device.as_ref() {
            if let Ok((dev_ty, dev_id)) = dev_obj.extract::<(i32, i32)>(py) {
                if dev_ty != kdl || dev_id != alloc_dev {
                    let wants_copy = copy
                        .as_ref()
                        .and_then(|c| c.extract::<bool>(py).ok())
                        .unwrap_or(false);
                    if wants_copy {
                        return Err(PyValueError::new_err(
                            "device copy not implemented for __dlpack__",
                        ));
                    } else {
                        return Err(PyValueError::new_err("dl_device mismatch for __dlpack__"));
                    }
                }
            }
        }

        let _ = stream;

        let inner = self
            .inner
            .take()
            .ok_or_else(|| PyValueError::new_err("__dlpack__ may only be called once"))?;

        let rows = inner.rows;
        let cols = inner.cols;
        let buf = inner.buf;

        let max_version_bound = max_version.map(|obj| obj.into_bound(py));

        export_f32_cuda_dlpack_2d(py, buf, rows, cols, alloc_dev, max_version_bound)
    }
}

#[cfg(all(feature = "python", feature = "cuda"))]
fn not_empty_f32(arr: PyReadonlyArray1<'_, f32>) -> PyResult<Vec<f32>> {
    let s = arr.as_slice()?;
    if s.is_empty() {
        Err(PyValueError::new_err("empty data"))
    } else {
        Ok(s.to_vec())
    }
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
    let inner = py
        .allow_threads(|| -> Result<DeviceArrayF32Sel, String> {
            if is("sma") {
                let sweep = crate::indicators::moving_averages::sma::SmaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaSma::new(device_id).map_err(|e| e.to_string())?;
                let ctx = cuda.context_arc_clone();
                let dev_id = cuda.device_id();
                let (dev, _c) = cuda
                    .sma_batch_dev(&prices, &sweep)
                    .map_err(|e| e.to_string())?;
                return Ok(DeviceArrayF32Sel {
                    buf: dev.buf,
                    rows: dev.rows,
                    cols: dev.cols,
                    ctx,
                    device_id: dev_id,
                });
            }
            if is("ema") {
                let sweep = crate::indicators::moving_averages::ema::EmaBatchRange {
                    period: (period, period, 0),
                };
                let cuda = CudaEma::new(device_id).map_err(|e| e.to_string())?;
                let ctx = cuda.context_arc();
                let dev = cuda
                    .ema_batch_dev(&prices, &sweep)
                    .map_err(|e| e.to_string())?;
                let dev_id = device_id as u32;
                return Ok(DeviceArrayF32Sel {
                    buf: dev.buf,
                    rows: dev.rows,
                    cols: dev.cols,
                    ctx,
                    device_id: dev_id,
                });
            }
            Err(format!("unsupported MA type: {}", ma_type))
        })
        .map_err(PyValueError::new_err)?;
    let device_id = inner.device_id;
    Ok(DeviceArrayF32PySel {
        inner: Some(inner),
        device_id,
    })
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
    let inner = py
        .allow_threads(|| -> Result<DeviceArrayF32Sel, String> {
            if is("sma") {
                let sweep = crate::indicators::moving_averages::sma::SmaBatchRange {
                    period: period_range,
                };
                let cuda = CudaSma::new(device_id).map_err(|e| e.to_string())?;
                let ctx = cuda.context_arc_clone();
                let dev_id = cuda.device_id();
                let (dev, _c) = cuda
                    .sma_batch_dev(&prices, &sweep)
                    .map_err(|e| e.to_string())?;
                return Ok(DeviceArrayF32Sel {
                    buf: dev.buf,
                    rows: dev.rows,
                    cols: dev.cols,
                    ctx,
                    device_id: dev_id,
                });
            }
            if is("ema") {
                let sweep = crate::indicators::moving_averages::ema::EmaBatchRange {
                    period: period_range,
                };
                let cuda = CudaEma::new(device_id).map_err(|e| e.to_string())?;
                let ctx = cuda.context_arc();
                let dev = cuda
                    .ema_batch_dev(&prices, &sweep)
                    .map_err(|e| e.to_string())?;
                let dev_id = device_id as u32;
                return Ok(DeviceArrayF32Sel {
                    buf: dev.buf,
                    rows: dev.rows,
                    cols: dev.cols,
                    ctx,
                    device_id: dev_id,
                });
            }
            Err(format!("ma_sweep_to_device unsupported for {}", ma_type))
        })
        .map_err(PyValueError::new_err)?;
    let device_id = inner.device_id;
    Ok(DeviceArrayF32PySel {
        inner: Some(inner),
        device_id,
    })
}
