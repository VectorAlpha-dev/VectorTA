use super::ma::MaData;
use crate::utilities::data_loader::source_type;
use crate::utilities::enums::Kernel;
use std::error::Error;
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MaBatchDispatchError {
    #[error("Unknown moving average type: {ma_type}")]
    UnknownType { ma_type: String },
    #[error(
        "{indicator} does not support period-sweep batch dispatch; use the indicator directly"
    )]
    NotPeriodBased { indicator: &'static str },
    #[error("{indicator} requires candles (timestamp/volume/OHLC); pass MaData::Candles")]
    RequiresCandles { indicator: &'static str },
    #[error("invalid param '{key}' for {indicator}: value={value} ({reason})")]
    InvalidParam {
        indicator: &'static str,
        key: &'static str,
        value: f64,
        reason: &'static str,
    },
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
    ma_batch_with_kernel_and_params(ma_type, data, period_range, kernel, None)
}

#[inline]
pub fn ma_batch_with_params<'a>(
    ma_type: &str,
    data: MaData<'a>,
    period_range: (usize, usize, usize),
    params: &HashMap<String, f64>,
) -> Result<MaBatchOutput, Box<dyn Error>> {
    ma_batch_with_kernel_and_params(ma_type, data, period_range, Kernel::Auto, Some(params))
}

pub fn ma_batch_with_kernel_and_params<'a>(
    ma_type: &str,
    data: MaData<'a>,
    period_range: (usize, usize, usize),
    kernel: Kernel,
    params: Option<&HashMap<String, f64>>,
) -> Result<MaBatchOutput, Box<dyn Error>> {
    let kernel = to_batch_kernel(kernel)?;
    let (prices, candles) = match data {
        MaData::Slice(s) => (s, None),
        MaData::Candles { candles, source } => (source_type(candles, source), Some(candles)),
    };

    #[inline]
    fn get_f64(
        params: Option<&HashMap<String, f64>>,
        indicator: &'static str,
        key: &'static str,
    ) -> Result<Option<f64>, MaBatchDispatchError> {
        match params.and_then(|m| m.get(key).copied()) {
            None => Ok(None),
            Some(v) if v.is_finite() => Ok(Some(v)),
            Some(v) => Err(MaBatchDispatchError::InvalidParam {
                indicator,
                key,
                value: v,
                reason: "expected finite number",
            }),
        }
    }

    #[inline]
    fn get_usize(
        params: Option<&HashMap<String, f64>>,
        indicator: &'static str,
        key: &'static str,
    ) -> Result<Option<usize>, MaBatchDispatchError> {
        let Some(v) = get_f64(params, indicator, key)? else {
            return Ok(None);
        };
        if v < 0.0 {
            return Err(MaBatchDispatchError::InvalidParam {
                indicator,
                key,
                value: v,
                reason: "expected >= 0",
            });
        }
        let r = v.round();
        if (v - r).abs() > 1e-9 {
            return Err(MaBatchDispatchError::InvalidParam {
                indicator,
                key,
                value: v,
                reason: "expected integer",
            });
        }
        if r > (usize::MAX as f64) {
            return Err(MaBatchDispatchError::InvalidParam {
                indicator,
                key,
                value: v,
                reason: "too large for usize",
            });
        }
        Ok(Some(r as usize))
    }

    #[inline]
    fn get_u32(
        params: Option<&HashMap<String, f64>>,
        indicator: &'static str,
        key: &'static str,
    ) -> Result<Option<u32>, MaBatchDispatchError> {
        let Some(v) = get_usize(params, indicator, key)? else {
            return Ok(None);
        };
        if v > (u32::MAX as usize) {
            return Err(MaBatchDispatchError::InvalidParam {
                indicator,
                key,
                value: v as f64,
                reason: "too large for u32",
            });
        }
        Ok(Some(v as u32))
    }

    match ma_type.to_ascii_lowercase().as_str() {
        "sma" => {
            let sweep = super::sma::SmaBatchRange {
                period: period_range,
            };
            let out = super::sma::sma_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(9)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "ema" => {
            let sweep = super::ema::EmaBatchRange {
                period: period_range,
            };
            let out = super::ema::ema_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(9)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "dema" => {
            let sweep = super::dema::DemaBatchRange {
                period: period_range,
            };
            let out = super::dema::dema_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(9)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "tema" => {
            let sweep = super::tema::TemaBatchRange {
                period: period_range,
            };
            let out = super::tema::tema_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(9)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "smma" => {
            let sweep = super::smma::SmmaBatchRange {
                period: period_range,
            };
            let out = super::smma::smma_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(9)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "zlema" => {
            let sweep = super::zlema::ZlemaBatchRange {
                period: period_range,
            };
            let out = super::zlema::zlema_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(9)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "wma" => {
            let sweep = super::wma::WmaBatchRange {
                period: period_range,
            };
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
            if let Some(v) = get_f64(params, "alma", "offset")? {
                sweep.offset = (v, v, 0.0);
            }
            if let Some(v) = get_f64(params, "alma", "sigma")? {
                sweep.sigma = (v, v, 0.0);
            }
            let out = super::alma::alma_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(9)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "cwma" => {
            let sweep = super::cwma::CwmaBatchRange {
                period: period_range,
            };
            let out = super::cwma::cwma_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(9)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "cora_wave" => {
            let mut sweep = crate::indicators::cora_wave::CoraWaveBatchRange {
                period: period_range,
                r_multi: (2.0, 2.0, 0.0),
                smooth: true,
            };
            if let Some(v) = get_f64(params, "cora_wave", "r_multi")? {
                if v < 0.0 {
                    return Err(MaBatchDispatchError::InvalidParam {
                        indicator: "cora_wave",
                        key: "r_multi",
                        value: v,
                        reason: "expected >= 0",
                    }
                    .into());
                }
                sweep.r_multi = (v, v, 0.0);
            }
            if let Some(v) = get_usize(params, "cora_wave", "smooth")? {
                sweep.smooth = match v {
                    0 => false,
                    1 => true,
                    other => {
                        return Err(MaBatchDispatchError::InvalidParam {
                            indicator: "cora_wave",
                            key: "smooth",
                            value: other as f64,
                            reason: "expected 0 or 1",
                        }
                        .into());
                    }
                };
            }
            let out = crate::indicators::cora_wave::cora_wave_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(20)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "edcf" => {
            let sweep = super::edcf::EdcfBatchRange {
                period: period_range,
            };
            let out = super::edcf::edcf_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(9)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "fwma" => {
            let sweep = super::fwma::FwmaBatchRange {
                period: period_range,
            };
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
            if let Some(v) = get_usize(params, "gaussian", "poles")? {
                sweep.poles = (v, v, 0);
            }
            let out = super::gaussian::gaussian_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(9)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "highpass" => {
            let sweep = super::highpass::HighPassBatchRange {
                period: period_range,
            };
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
            if let Some(v) = get_f64(params, "highpass_2_pole", "k")? {
                sweep.k = (v, v, 0.0);
            }
            let out =
                super::highpass_2_pole::highpass_2_pole_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(48)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "hma" => {
            let sweep = super::hma::HmaBatchRange {
                period: period_range,
            };
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
            if let Some(v) = get_f64(params, "jma", "phase")? {
                sweep.phase = (v, v, 0.0);
            }
            if let Some(v) = get_u32(params, "jma", "power")? {
                sweep.power = (v, v, 0);
            }
            let out = super::jma::jma_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(9)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "jsa" => {
            let sweep = super::jsa::JsaBatchRange {
                period: period_range,
            };
            let out = super::jsa::jsa_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(9)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "linreg" => {
            let sweep = super::linreg::LinRegBatchRange {
                period: period_range,
            };
            let out = super::linreg::linreg_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(9)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "kama" => {
            let sweep = super::kama::KamaBatchRange {
                period: period_range,
            };
            let out = super::kama::kama_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(9)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "ehlers_kama" => {
            let sweep = super::ehlers_kama::EhlersKamaBatchRange {
                period: period_range,
            };
            let out = super::ehlers_kama::ehlers_kama_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(9)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "ehlers_itrend" => {
            let warmup = get_usize(params, "ehlers_itrend", "warmup_bars")?.unwrap_or(20);
            let sweep = super::ehlers_itrend::EhlersITrendBatchRange {
                warmup_bars: (warmup, warmup, 0),
                max_dc_period: period_range,
            };
            let out =
                super::ehlers_itrend::ehlers_itrend_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.max_dc_period.unwrap_or(48)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "ehlers_ecema" => {
            let gain_limit = get_usize(params, "ehlers_ecema", "gain_limit")?.unwrap_or(50);
            let sweep = super::ehlers_ecema::EhlersEcemaBatchRange {
                length: period_range,
                gain_limit: (gain_limit, gain_limit, 0),
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
            let sweep = super::ehma::EhmaBatchRange {
                period: period_range,
            };
            let out = super::ehma::ehma_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(9)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "nama" => {
            let sweep = super::nama::NamaBatchRange {
                period: period_range,
            };
            let out = super::nama::nama_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(9)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "nma" => {
            let sweep = super::nma::NmaBatchRange {
                period: period_range,
            };
            let out = super::nma::nma_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(9)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "pwma" => {
            let sweep = super::pwma::PwmaBatchRange {
                period: period_range,
            };
            let out = super::pwma::pwma_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(9)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "reflex" => {
            let sweep = super::reflex::ReflexBatchRange {
                period: period_range,
            };
            let out = super::reflex::reflex_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(48)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "sinwma" => {
            let sweep = super::sinwma::SinWmaBatchRange {
                period: period_range,
            };
            let out = super::sinwma::sinwma_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(9)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "sqwma" => {
            let sweep = super::sqwma::SqwmaBatchRange {
                period: period_range,
            };
            let out = super::sqwma::sqwma_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(9)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "srwma" => {
            let sweep = super::srwma::SrwmaBatchRange {
                period: period_range,
            };
            let out = super::srwma::srwma_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(9)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "swma" => {
            let sweep = super::swma::SwmaBatchRange {
                period: period_range,
            };
            let out = super::swma::swma_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(9)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "supersmoother" => {
            let sweep = super::supersmoother::SuperSmootherBatchRange {
                period: period_range,
            };
            let out =
                super::supersmoother::supersmoother_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(48)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "supersmoother_3_pole" => {
            let sweep = super::supersmoother_3_pole::SuperSmoother3PoleBatchRange {
                period: period_range,
            };
            let out = super::supersmoother_3_pole::supersmoother_3_pole_batch_with_kernel(
                prices, &sweep, kernel,
            )?;
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
            if let Some(v) = get_f64(params, "tilson", "volume_factor")? {
                sweep.volume_factor = (v, v, 0.0);
            }
            let out = super::tilson::tilson_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(9)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "trendflex" => {
            let sweep = super::trendflex::TrendFlexBatchRange {
                period: period_range,
            };
            let out = super::trendflex::trendflex_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(48)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "trima" => {
            let sweep = super::trima::TrimaBatchRange {
                period: period_range,
            };
            let out = super::trima::trima_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(9)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "wilders" => {
            let sweep = super::wilders::WildersBatchRange {
                period: period_range,
            };
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
                power: {
                    let v = get_f64(params, "vpwma", "power")?.unwrap_or(0.382);
                    (v, v, 0.0)
                },
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
            let candles =
                candles.ok_or(MaBatchDispatchError::RequiresCandles { indicator: "vwma" })?;
            let sweep = super::vwma::VwmaBatchRange {
                period: period_range,
            };
            let out = super::vwma::vwma_batch_with_kernel(prices, &candles.volume, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(20)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "tradjema" => Err(MaBatchDispatchError::NotPeriodBased {
            indicator: "tradjema",
        }
        .into()),
        "uma" => Err(MaBatchDispatchError::NotPeriodBased { indicator: "uma" }.into()),
        "volume_adjusted_ma" => Err(MaBatchDispatchError::NotPeriodBased {
            indicator: "volume_adjusted_ma",
        }
        .into()),
        "hwma" => Err(MaBatchDispatchError::NotPeriodBased { indicator: "hwma" }.into()),
        "mama" => Err(MaBatchDispatchError::NotPeriodBased { indicator: "mama" }.into()),
        "mwdx" => Err(MaBatchDispatchError::NotPeriodBased { indicator: "mwdx" }.into()),
        "vwap" => Err(MaBatchDispatchError::NotPeriodBased { indicator: "vwap" }.into()),
        "dma" => {
            let ema_length = get_usize(params, "dma", "ema_length")?.unwrap_or(20);
            let ema_gain_limit = get_usize(params, "dma", "ema_gain_limit")?.unwrap_or(50);
            let sweep = super::dma::DmaBatchRange {
                hull_length: period_range,
                ema_length: (ema_length, ema_length, 0),
                ema_gain_limit: (ema_gain_limit, ema_gain_limit, 0),
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
            let offset = get_usize(params, "epma", "offset")?.unwrap_or(4);
            let sweep = super::epma::EpmaBatchRange {
                period: period_range,
                offset: (offset, offset, 0),
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
            if let Some(v) = get_usize(params, "sama", "maj_length")? {
                sweep.maj_length = (v, v, 0);
            }
            if let Some(v) = get_usize(params, "sama", "min_length")? {
                sweep.min_length = (v, v, 0);
            }
            let out = super::sama::sama_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.length.unwrap_or(10)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "volatility_adjusted_ma" | "vama" => {
            let vol_period = get_usize(params, "vama", "vol_period")?.unwrap_or(51);
            let sweep = super::volatility_adjusted_ma::VamaBatchRange {
                base_period: period_range,
                vol_period: (vol_period, vol_period, 0),
            };
            let out =
                super::volatility_adjusted_ma::vama_batch_with_kernel(prices, &sweep, kernel)?;
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
            if let Some(v) = get_usize(params, "maaq", "fast_period")? {
                sweep.fast_period = (v, v, 0);
            }
            if let Some(v) = get_usize(params, "maaq", "slow_period")? {
                sweep.slow_period = (v, v, 0);
            }
            let out = super::maaq::maaq_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(9)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "frama" => {
            let sc = get_usize(params, "frama", "sc")?.unwrap_or(300);
            let fc = get_usize(params, "frama", "fc")?.unwrap_or(1);
            let (high, low, close) = match candles {
                Some(c) => (&c.high[..], &c.low[..], &c.close[..]),
                None => (prices, prices, prices),
            };
            let sweep = super::frama::FramaBatchRange {
                window: period_range,
                sc: (sc, sc, 0),
                fc: (fc, fc, 0),
            };
            let out = super::frama::frama_batch_with_kernel(
                high,
                low,
                close,
                &sweep,
                kernel,
            )?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.window.unwrap_or(10)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        other => Err(MaBatchDispatchError::UnknownType {
            ma_type: other.to_string(),
        }
        .into()),
    }
}
