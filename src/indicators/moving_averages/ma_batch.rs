//! # Moving Average Batch Dispatcher (CPU)
//!
//! A uniform, runtime-selected entry point for period-sweep batch kernels.
//! This mirrors the single-series dispatcher in `ma.rs`, but calls each
//! indicator's `*_batch_with_kernel` CPU implementation to produce a
//! row-major result matrix (rows = parameter combos, cols = input length).
//!
//! Scope: This dispatcher is intentionally **period-sweep oriented**. Moving
//! averages that don't use a `period` (or require non-period parameters) are
//! reported as unsupported here - call their dedicated batch APIs directly.

use super::ma::MaData;
use crate::utilities::data_loader::source_type;
use crate::utilities::enums::Kernel;
use std::error::Error;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MaBatchDispatchError {
    #[error("Unknown moving average type: {ma_type}")]
    UnknownType { ma_type: String },
    #[error("{indicator} does not support period-sweep batch dispatch; use the indicator directly")]
    NotPeriodBased { indicator: &'static str },
    #[error("{indicator} requires candles (timestamp/volume/OHLC); pass MaData::Candles")]
    RequiresCandles { indicator: &'static str },
    #[error("invalid kernel for batch path: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Clone, Debug)]
pub struct MaBatchOutput {
    pub values: Vec<f64>,
    pub periods: Vec<usize>,
    pub rows: usize,
    pub cols: usize,
}

impl MaBatchOutput {
    pub fn row_for_period(&self, period: usize) -> Option<usize> {
        self.periods.iter().position(|&p| p == period)
    }

    pub fn values_for_period(&self, period: usize) -> Option<&[f64]> {
        self.row_for_period(period).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline]
fn to_batch_kernel(k: Kernel) -> Result<Kernel, MaBatchDispatchError> {
    let out = match k {
        Kernel::Auto => Kernel::Auto,
        Kernel::Scalar => Kernel::ScalarBatch,
        Kernel::Avx2 => Kernel::Avx2Batch,
        Kernel::Avx512 => Kernel::Avx512Batch,
        other if other.is_batch() => other,
        other => return Err(MaBatchDispatchError::InvalidKernelForBatch(other)),
    };
    Ok(out)
}

#[inline]
fn map_periods<T>(combos: &[T], get_period: impl Fn(&T) -> usize) -> Vec<usize> {
    combos.iter().map(get_period).collect()
}

#[inline]
pub fn ma_batch<'a>(
    ma_type: &str,
    data: MaData<'a>,
    period_range: (usize, usize, usize),
) -> Result<MaBatchOutput, Box<dyn Error>> {
    ma_batch_with_kernel(ma_type, data, period_range, Kernel::Auto)
}

pub fn ma_batch_with_kernel<'a>(
    ma_type: &str,
    data: MaData<'a>,
    period_range: (usize, usize, usize),
    kernel: Kernel,
) -> Result<MaBatchOutput, Box<dyn Error>> {
    let kernel = to_batch_kernel(kernel)?;
    let (prices, candles) = match data {
        MaData::Slice(s) => (s, None),
        MaData::Candles { candles, source } => (source_type(candles, source), Some(candles)),
    };

    match ma_type.to_ascii_lowercase().as_str() {
        "sma" => {
            let sweep = super::sma::SmaBatchRange { period: period_range };
            let out = super::sma::sma_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(9)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "ema" => {
            let sweep = super::ema::EmaBatchRange { period: period_range };
            let out = super::ema::ema_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(9)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "dema" => {
            let sweep = super::dema::DemaBatchRange { period: period_range };
            let out = super::dema::dema_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(9)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "tema" => {
            let sweep = super::tema::TemaBatchRange { period: period_range };
            let out = super::tema::tema_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(9)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "smma" => {
            let sweep = super::smma::SmmaBatchRange { period: period_range };
            let out = super::smma::smma_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(9)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "zlema" => {
            let sweep = super::zlema::ZlemaBatchRange { period: period_range };
            let out = super::zlema::zlema_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(9)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "wma" => {
            let sweep = super::wma::WmaBatchRange { period: period_range };
            let out = super::wma::wma_with_kernel_batch(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(9)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "alma" => {
            let mut sweep = super::alma::AlmaBatchRange::default();
            sweep.period = period_range;
            let out = super::alma::alma_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(9)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "cwma" => {
            let sweep = super::cwma::CwmaBatchRange { period: period_range };
            let out = super::cwma::cwma_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(9)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "edcf" => {
            let sweep = super::edcf::EdcfBatchRange { period: period_range };
            let out = super::edcf::edcf_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(9)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "fwma" => {
            let sweep = super::fwma::FwmaBatchRange { period: period_range };
            let out = super::fwma::fwma_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(9)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "gaussian" => {
            let mut sweep = super::gaussian::GaussianBatchRange::default();
            sweep.period = period_range;
            let out = super::gaussian::gaussian_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(9)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "highpass" => {
            let sweep = super::highpass::HighPassBatchRange { period: period_range };
            let out = super::highpass::highpass_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(48)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "highpass2" | "highpass_2_pole" => {
            let mut sweep = super::highpass_2_pole::HighPass2BatchRange::default();
            sweep.period = period_range;
            let out = super::highpass_2_pole::highpass_2_pole_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(48)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "hma" => {
            let sweep = super::hma::HmaBatchRange { period: period_range };
            let out = super::hma::hma_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(9)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "jma" => {
            let mut sweep = super::jma::JmaBatchRange::default();
            sweep.period = period_range;
            let out = super::jma::jma_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(9)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "jsa" => {
            let sweep = super::jsa::JsaBatchRange { period: period_range };
            let out = super::jsa::jsa_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(9)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "linreg" => {
            let sweep = super::linreg::LinRegBatchRange { period: period_range };
            let out = super::linreg::linreg_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(9)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "kama" => {
            let sweep = super::kama::KamaBatchRange { period: period_range };
            let out = super::kama::kama_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(9)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "ehlers_kama" => {
            let sweep = super::ehlers_kama::EhlersKamaBatchRange { period: period_range };
            let out = super::ehlers_kama::ehlers_kama_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(9)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "ehlers_itrend" => {
            let sweep = super::ehlers_itrend::EhlersITrendBatchRange {
                warmup_bars: (20, 20, 0),
                max_dc_period: period_range,
            };
            let out = super::ehlers_itrend::ehlers_itrend_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.max_dc_period.unwrap_or(48)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "ehlers_ecema" => {
            let sweep = super::ehlers_ecema::EhlersEcemaBatchRange {
                length: period_range,
                gain_limit: (50, 50, 0),
            };
            let out = super::ehlers_ecema::ehlers_ecema_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.length.unwrap_or(20)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "ehma" => {
            let sweep = super::ehma::EhmaBatchRange { period: period_range };
            let out = super::ehma::ehma_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(9)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "nama" => {
            let sweep = super::nama::NamaBatchRange { period: period_range };
            let out = super::nama::nama_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(9)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "nma" => {
            let sweep = super::nma::NmaBatchRange { period: period_range };
            let out = super::nma::nma_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(9)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "pwma" => {
            let sweep = super::pwma::PwmaBatchRange { period: period_range };
            let out = super::pwma::pwma_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(9)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "reflex" => {
            let sweep = super::reflex::ReflexBatchRange { period: period_range };
            let out = super::reflex::reflex_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(48)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "sinwma" => {
            let sweep = super::sinwma::SinWmaBatchRange { period: period_range };
            let out = super::sinwma::sinwma_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(9)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "sqwma" => {
            let sweep = super::sqwma::SqwmaBatchRange { period: period_range };
            let out = super::sqwma::sqwma_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(9)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "srwma" => {
            let sweep = super::srwma::SrwmaBatchRange { period: period_range };
            let out = super::srwma::srwma_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(9)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "swma" => {
            let sweep = super::swma::SwmaBatchRange { period: period_range };
            let out = super::swma::swma_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(9)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "supersmoother" => {
            let sweep = super::supersmoother::SuperSmootherBatchRange { period: period_range };
            let out = super::supersmoother::supersmoother_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(48)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "supersmoother_3_pole" => {
            let sweep = super::supersmoother_3_pole::SuperSmoother3PoleBatchRange { period: period_range };
            let out = super::supersmoother_3_pole::supersmoother_3_pole_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(48)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "tilson" => {
            let mut sweep = super::tilson::TilsonBatchRange::default();
            sweep.period = period_range;
            let out = super::tilson::tilson_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(9)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "trendflex" => {
            let sweep = super::trendflex::TrendFlexBatchRange { period: period_range };
            let out = super::trendflex::trendflex_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(48)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "trima" => {
            let sweep = super::trima::TrimaBatchRange { period: period_range };
            let out = super::trima::trima_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(9)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "wilders" => {
            let sweep = super::wilders::WildersBatchRange { period: period_range };
            let out = super::wilders::wilders_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(14)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "vpwma" => {
            let sweep = super::vpwma::VpwmaBatchRange {
                period: period_range,
                power: (0.382, 0.382, 0.0),
            };
            let out = super::vpwma::vpwma_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(14)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "vwma" => {
            let candles = candles.ok_or(MaBatchDispatchError::RequiresCandles { indicator: "vwma" })?;
            let sweep = super::vwma::VwmaBatchRange { period: period_range };
            let out = super::vwma::vwma_batch_with_kernel(prices, &candles.volume, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(20)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "tradjema" => Err(MaBatchDispatchError::NotPeriodBased { indicator: "tradjema" }.into()),
        "uma" => Err(MaBatchDispatchError::NotPeriodBased { indicator: "uma" }.into()),
        "volume_adjusted_ma" => Err(MaBatchDispatchError::NotPeriodBased { indicator: "volume_adjusted_ma" }.into()),
        "hwma" => Err(MaBatchDispatchError::NotPeriodBased { indicator: "hwma" }.into()),
        "mama" => Err(MaBatchDispatchError::NotPeriodBased { indicator: "mama" }.into()),
        "mwdx" => Err(MaBatchDispatchError::NotPeriodBased { indicator: "mwdx" }.into()),
        "vwap" => Err(MaBatchDispatchError::NotPeriodBased { indicator: "vwap" }.into()),
        "dma" => {
            let sweep = super::dma::DmaBatchRange {
                hull_length: period_range,
                ema_length: (20, 20, 0),
                ema_gain_limit: (50, 50, 0),
                hull_ma_type: "WMA".to_string(),
            };
            let out = super::dma::dma_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.hull_length.unwrap_or(7)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "epma" => {
            let sweep = super::epma::EpmaBatchRange {
                period: period_range,
                offset: (4, 4, 0),
            };
            let out = super::epma::epma_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(9)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "sama" => {
            let mut sweep = super::sama::SamaBatchRange::default();
            sweep.length = period_range;
            let out = super::sama::sama_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.length.unwrap_or(10)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "volatility_adjusted_ma" | "vama" => {
            let sweep = super::volatility_adjusted_ma::VamaBatchRange {
                base_period: period_range,
                vol_period: (51, 51, 0),
            };
            let out = super::volatility_adjusted_ma::vama_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.base_period.unwrap_or(10)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "maaq" => {
            let mut sweep = super::maaq::MaaqBatchRange::default();
            sweep.period = period_range;
            let out = super::maaq::maaq_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(9)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        other => Err(MaBatchDispatchError::UnknownType { ma_type: other.to_string() }.into()),
    }
}
