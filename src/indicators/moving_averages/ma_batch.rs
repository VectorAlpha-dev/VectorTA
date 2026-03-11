use super::ma::MaData;
use crate::utilities::data_loader::source_type;
use crate::utilities::enums::Kernel;
use std::collections::HashMap;
use std::error::Error;
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

#[derive(Clone, Debug)]
pub enum MaBatchParamValue<'a> {
    Int(i64),
    Float(f64),
    Bool(bool),
    EnumString(&'a str),
}

#[derive(Clone, Debug)]
pub struct MaBatchParamKV<'a> {
    pub key: &'a str,
    pub value: MaBatchParamValue<'a>,
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
fn expand_period_axis(range: (usize, usize, usize)) -> Result<Vec<usize>, MaBatchDispatchError> {
    let (start, end, step) = range;
    let periods = if step == 0 || start == end {
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
        return Err(MaBatchDispatchError::InvalidParam {
            indicator: "period_range",
            key: "step",
            value: step as f64,
            reason: "invalid period range",
        });
    }
    Ok(periods)
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

pub fn ma_batch_with_kernel_and_typed_params<'a>(
    ma_type: &str,
    data: MaData<'a>,
    period_range: (usize, usize, usize),
    kernel: Kernel,
    params: &[MaBatchParamKV<'_>],
) -> Result<MaBatchOutput, Box<dyn Error>> {
    let mut numeric: HashMap<String, f64> = HashMap::with_capacity(params.len());
    let mut text: HashMap<String, String> = HashMap::new();

    for p in params {
        match p.value {
            MaBatchParamValue::Int(v) => {
                numeric.insert(p.key.to_string(), v as f64);
            }
            MaBatchParamValue::Float(v) => {
                if !v.is_finite() {
                    return Err(MaBatchDispatchError::InvalidParam {
                        indicator: "typed_params",
                        key: "float",
                        value: v,
                        reason: "expected finite number",
                    }
                    .into());
                }
                numeric.insert(p.key.to_string(), v);
            }
            MaBatchParamValue::Bool(v) => {
                numeric.insert(p.key.to_string(), if v { 1.0 } else { 0.0 });
            }
            MaBatchParamValue::EnumString(v) => {
                text.insert(p.key.to_string(), v.to_string());
            }
        }
    }

    if ma_type.eq_ignore_ascii_case("dma") && text.contains_key("hull_ma_type") {
        let kernel = to_batch_kernel(kernel)?;
        let (prices, _) = match data {
            MaData::Slice(s) => (s, None),
            MaData::Candles { candles, source } => (source_type(candles, source), Some(candles)),
        };

        let get_u = |key: &'static str, default_v: usize| -> Result<usize, MaBatchDispatchError> {
            let Some(v) = numeric.get(key).copied() else {
                return Ok(default_v);
            };
            if v < 0.0 {
                return Err(MaBatchDispatchError::InvalidParam {
                    indicator: "dma",
                    key,
                    value: v,
                    reason: "expected >= 0",
                });
            }
            let r = v.round();
            if (v - r).abs() > 1e-9 {
                return Err(MaBatchDispatchError::InvalidParam {
                    indicator: "dma",
                    key,
                    value: v,
                    reason: "expected integer",
                });
            }
            Ok(r as usize)
        };

        let ema_length = get_u("ema_length", 20)?;
        let ema_gain_limit = get_u("ema_gain_limit", 50)?;
        let hull_ma_type = text
            .get("hull_ma_type")
            .cloned()
            .unwrap_or_else(|| "WMA".to_string());
        let sweep = super::dma::DmaBatchRange {
            hull_length: period_range,
            ema_length: (ema_length, ema_length, 0),
            ema_gain_limit: (ema_gain_limit, ema_gain_limit, 0),
            hull_ma_type,
        };
        let out = super::dma::dma_batch_with_kernel(prices, &sweep, kernel)?;
        return Ok(MaBatchOutput {
            periods: map_periods(&out.combos, |p| p.hull_length.unwrap_or(7)),
            values: out.values,
            rows: out.rows,
            cols: out.cols,
        });
    }

    if ma_type.eq_ignore_ascii_case("vwap")
        && (text.contains_key("anchor")
            || text.contains_key("anchor_start")
            || text.contains_key("anchor_end"))
    {
        let kernel = to_batch_kernel(kernel)?;
        let (prices, candles) = match data {
            MaData::Slice(s) => (s, None),
            MaData::Candles { candles, source } => (source_type(candles, source), Some(candles)),
        };
        let candles = candles.ok_or(MaBatchDispatchError::RequiresCandles { indicator: "vwap" })?;

        let single_anchor = text.get("anchor").cloned();
        let anchor_start = text
            .get("anchor_start")
            .cloned()
            .or_else(|| single_anchor.clone())
            .unwrap_or_else(|| "1d".to_string());
        let anchor_end = text
            .get("anchor_end")
            .cloned()
            .or_else(|| single_anchor.clone())
            .unwrap_or_else(|| anchor_start.clone());
        let anchor_step = numeric
            .get("anchor_step")
            .copied()
            .map(|v| {
                if v < 0.0 {
                    return Err(MaBatchDispatchError::InvalidParam {
                        indicator: "vwap",
                        key: "anchor_step",
                        value: v,
                        reason: "expected >= 0",
                    });
                }
                let r = v.round();
                if (v - r).abs() > 1e-9 {
                    return Err(MaBatchDispatchError::InvalidParam {
                        indicator: "vwap",
                        key: "anchor_step",
                        value: v,
                        reason: "expected integer",
                    });
                }
                Ok(r as u32)
            })
            .transpose()?
            .unwrap_or_else(|| if anchor_start == anchor_end { 0 } else { 1 });

        let sweep = super::vwap::VwapBatchRange {
            anchor: (anchor_start, anchor_end, anchor_step),
        };
        let out = super::vwap::vwap_batch_with_kernel(
            &candles.timestamp,
            &candles.volume,
            prices,
            &sweep,
            kernel,
        )?;
        let periods = out
            .combos
            .iter()
            .enumerate()
            .map(|(i, p)| {
                p.anchor
                    .as_deref()
                    .and_then(|a| super::vwap::parse_anchor(a).ok().map(|(n, _)| n as usize))
                    .unwrap_or(i + 1)
            })
            .collect();
        return Ok(MaBatchOutput {
            periods,
            values: out.values,
            rows: out.rows,
            cols: out.cols,
        });
    }

    if ma_type.eq_ignore_ascii_case("mama") {
        let kernel = to_batch_kernel(kernel)?;
        let (prices, _) = match data {
            MaData::Slice(s) => (s, None),
            MaData::Candles { candles, source } => (source_type(candles, source), Some(candles)),
        };
        let mut sweep = super::mama::MamaBatchRange::default();
        if let Some(v) = numeric.get("fast_limit").copied() {
            sweep.fast_limit = (v, v, 0.0);
        } else {
            if let Some(v) = numeric.get("fast_limit_start").copied() {
                sweep.fast_limit.0 = v;
            }
            if let Some(v) = numeric.get("fast_limit_end").copied() {
                sweep.fast_limit.1 = v;
            }
            if let Some(v) = numeric.get("fast_limit_step").copied() {
                sweep.fast_limit.2 = v;
            }
        }
        if let Some(v) = numeric.get("slow_limit").copied() {
            sweep.slow_limit = (v, v, 0.0);
        } else {
            if let Some(v) = numeric.get("slow_limit_start").copied() {
                sweep.slow_limit.0 = v;
            }
            if let Some(v) = numeric.get("slow_limit_end").copied() {
                sweep.slow_limit.1 = v;
            }
            if let Some(v) = numeric.get("slow_limit_step").copied() {
                sweep.slow_limit.2 = v;
            }
        }
        let out = super::mama::mama_batch_with_kernel(prices, &sweep, kernel)?;
        let output = text
            .get("output")
            .map(String::as_str)
            .unwrap_or("mama")
            .to_ascii_lowercase();
        let values = match output.as_str() {
            "mama" => out.mama_values,
            "fama" => out.fama_values,
            _ => {
                return Err(MaBatchDispatchError::InvalidParam {
                    indicator: "mama",
                    key: "output",
                    value: f64::NAN,
                    reason: "expected 'mama' or 'fama'",
                }
                .into())
            }
        };
        return Ok(MaBatchOutput {
            periods: (1..=out.rows).collect(),
            values,
            rows: out.rows,
            cols: out.cols,
        });
    }

    if ma_type.eq_ignore_ascii_case("ehlers_pma") {
        let kernel = to_batch_kernel(kernel)?;
        let periods = expand_period_axis(period_range)?;
        let rows = periods.len();
        let (prices, _) = match data {
            MaData::Slice(s) => (s, None),
            MaData::Candles { candles, source } => (source_type(candles, source), Some(candles)),
        };
        let input = super::ehlers_pma::EhlersPmaInput::from_slice(
            prices,
            super::ehlers_pma::EhlersPmaParams::default(),
        );
        let out = super::ehlers_pma::ehlers_pma_with_kernel(&input, kernel)?;
        let output = text
            .get("output")
            .map(String::as_str)
            .unwrap_or("predict")
            .to_ascii_lowercase();
        let series = match output.as_str() {
            "predict" => &out.predict,
            "trigger" => &out.trigger,
            _ => {
                return Err(MaBatchDispatchError::InvalidParam {
                    indicator: "ehlers_pma",
                    key: "output",
                    value: f64::NAN,
                    reason: "expected 'predict' or 'trigger'",
                }
                .into())
            }
        };
        let cols = series.len();
        let mut values = Vec::with_capacity(rows.saturating_mul(cols));
        for _ in 0..rows {
            values.extend_from_slice(series);
        }
        return Ok(MaBatchOutput {
            periods,
            values,
            rows,
            cols,
        });
    }

    if ma_type.eq_ignore_ascii_case("ema_deviation_corrected_t3") {
        let kernel = to_batch_kernel(kernel)?;
        let (prices, _) = match data {
            MaData::Slice(s) => (s, None),
            MaData::Candles { candles, source } => (source_type(candles, source), Some(candles)),
        };
        let sweep = super::ema_deviation_corrected_t3::EmaDeviationCorrectedT3BatchRange {
            period: period_range,
            hot: {
                let v = numeric.get("hot").copied().unwrap_or(0.7);
                (v, v, 0.0)
            },
            t3_mode: {
                let v = numeric.get("t3_mode").copied().unwrap_or(0.0);
                if v < 0.0 || (v - v.round()).abs() > 1e-9 {
                    return Err(MaBatchDispatchError::InvalidParam {
                        indicator: "ema_deviation_corrected_t3",
                        key: "t3_mode",
                        value: v,
                        reason: "expected integer >= 0",
                    }
                    .into());
                }
                let v = v.round() as usize;
                (v, v, 0)
            },
        };
        let out = super::ema_deviation_corrected_t3::ema_deviation_corrected_t3_batch_with_kernel(
            prices, &sweep, kernel,
        )?;
        let output = text
            .get("output")
            .map(String::as_str)
            .unwrap_or("corrected")
            .to_ascii_lowercase();
        let periods = map_periods(&out.combos, |p| p.period.unwrap_or(10));
        let rows = out.rows;
        let cols = out.cols;
        let series = match output.as_str() {
            "corrected" | "value" => out.corrected,
            "t3" => out.t3,
            _ => {
                return Err(MaBatchDispatchError::InvalidParam {
                    indicator: "ema_deviation_corrected_t3",
                    key: "output",
                    value: f64::NAN,
                    reason: "expected 'corrected' or 't3'",
                }
                .into())
            }
        };
        return Ok(MaBatchOutput {
            periods,
            values: series,
            rows,
            cols,
        });
    }

    if ma_type.eq_ignore_ascii_case("ehlers_undersampled_double_moving_average") {
        let kernel = to_batch_kernel(kernel)?;
        let (prices, _) = match data {
            MaData::Slice(s) => (s, None),
            MaData::Candles { candles, source } => (source_type(candles, source), Some(candles)),
        };

        let get_u = |key: &'static str, default_v: usize| -> Result<usize, MaBatchDispatchError> {
            let Some(v) = numeric.get(key).copied() else {
                return Ok(default_v);
            };
            if v < 0.0 {
                return Err(MaBatchDispatchError::InvalidParam {
                    indicator: "ehlers_undersampled_double_moving_average",
                    key,
                    value: v,
                    reason: "expected >= 0",
                });
            }
            let r = v.round();
            if (v - r).abs() > 1e-9 {
                return Err(MaBatchDispatchError::InvalidParam {
                    indicator: "ehlers_undersampled_double_moving_average",
                    key,
                    value: v,
                    reason: "expected integer",
                });
            }
            Ok(r as usize)
        };

        let mut sweep =
            super::ehlers_undersampled_double_moving_average::EhlersUndersampledDoubleMovingAverageBatchRange::default();
        sweep.fast_length = (
            get_u("fast_length_start", get_u("fast_length", 6)?)?,
            get_u("fast_length_end", get_u("fast_length", 6)?)?,
            get_u("fast_length_step", 0)?,
        );
        sweep.slow_length = (
            get_u("slow_length_start", get_u("slow_length", 12)?)?,
            get_u("slow_length_end", get_u("slow_length", 12)?)?,
            get_u("slow_length_step", 0)?,
        );
        sweep.sample_length = (
            get_u("sample_length_start", get_u("sample_length", 5)?)?,
            get_u("sample_length_end", get_u("sample_length", 5)?)?,
            get_u("sample_length_step", 0)?,
        );

        let out =
            super::ehlers_undersampled_double_moving_average::ehlers_undersampled_double_moving_average_batch_with_kernel(
                prices, &sweep, kernel,
            )?;
        let output = text
            .get("output")
            .map(String::as_str)
            .unwrap_or("fast")
            .to_ascii_lowercase();
        let values = match output.as_str() {
            "fast" => out.fast_values,
            "slow" => out.slow_values,
            _ => {
                return Err(MaBatchDispatchError::InvalidParam {
                    indicator: "ehlers_undersampled_double_moving_average",
                    key: "output",
                    value: f64::NAN,
                    reason: "expected 'fast' or 'slow'",
                }
                .into())
            }
        };

        return Ok(MaBatchOutput {
            periods: (1..=out.rows).collect(),
            values,
            rows: out.rows,
            cols: out.cols,
        });
    }

    if ma_type.eq_ignore_ascii_case("buff_averages") {
        let kernel = to_batch_kernel(kernel)?;
        let (prices, candles) = match data {
            MaData::Slice(s) => (s, None),
            MaData::Candles { candles, source } => (source_type(candles, source), Some(candles)),
        };
        let candles = candles.ok_or(MaBatchDispatchError::RequiresCandles {
            indicator: "buff_averages",
        })?;

        let get_u = |key: &'static str| -> Result<Option<usize>, MaBatchDispatchError> {
            let Some(v) = numeric.get(key).copied() else {
                return Ok(None);
            };
            if v < 0.0 {
                return Err(MaBatchDispatchError::InvalidParam {
                    indicator: "buff_averages",
                    key,
                    value: v,
                    reason: "expected >= 0",
                });
            }
            let r = v.round();
            if (v - r).abs() > 1e-9 {
                return Err(MaBatchDispatchError::InvalidParam {
                    indicator: "buff_averages",
                    key,
                    value: v,
                    reason: "expected integer",
                });
            }
            Ok(Some(r as usize))
        };

        let mut sweep = super::buff_averages::BuffAveragesBatchRange::default();
        sweep.slow_period = period_range;

        if let Some(v) = get_u("fast_period")? {
            sweep.fast_period = (v, v, 0);
        }
        if let Some(v) = get_u("slow_period")? {
            sweep.slow_period = (v, v, 0);
        }
        if let Some(v) = get_u("fast_period_start")? {
            sweep.fast_period.0 = v;
        }
        if let Some(v) = get_u("fast_period_end")? {
            sweep.fast_period.1 = v;
        }
        if let Some(v) = get_u("fast_period_step")? {
            sweep.fast_period.2 = v;
        }
        if let Some(v) = get_u("slow_period_start")? {
            sweep.slow_period.0 = v;
        }
        if let Some(v) = get_u("slow_period_end")? {
            sweep.slow_period.1 = v;
        }
        if let Some(v) = get_u("slow_period_step")? {
            sweep.slow_period.2 = v;
        }

        let out = super::buff_averages::buff_averages_batch_with_kernel(
            prices,
            &candles.volume,
            &sweep,
            kernel,
        )?;

        let output = text
            .get("output")
            .map(String::as_str)
            .unwrap_or("fast")
            .to_ascii_lowercase();
        let values = match output.as_str() {
            "fast" | "fast_buff" => out.fast,
            "slow" | "slow_buff" => out.slow,
            _ => {
                return Err(MaBatchDispatchError::InvalidParam {
                    indicator: "buff_averages",
                    key: "output",
                    value: f64::NAN,
                    reason: "expected 'fast' or 'slow'",
                }
                .into())
            }
        };

        let all_fast_same = out
            .combos
            .first()
            .map(|c| out.combos.iter().all(|x| x.0 == c.0))
            .unwrap_or(true);
        let all_slow_same = out
            .combos
            .first()
            .map(|c| out.combos.iter().all(|x| x.1 == c.1))
            .unwrap_or(true);
        let periods = if all_fast_same {
            out.combos.iter().map(|c| c.1).collect()
        } else if all_slow_same {
            out.combos.iter().map(|c| c.0).collect()
        } else {
            (1..=out.rows).collect()
        };

        return Ok(MaBatchOutput {
            periods,
            values,
            rows: out.rows,
            cols: out.cols,
        });
    }

    if ma_type.eq_ignore_ascii_case("n_order_ema") {
        let kernel = to_batch_kernel(kernel)?;
        let (prices, _) = match data {
            MaData::Slice(s) => (s, None),
            MaData::Candles { candles, source } => (source_type(candles, source), Some(candles)),
        };

        let order = match numeric.get("order").copied() {
            Some(v) => {
                if v <= 0.0 {
                    return Err(MaBatchDispatchError::InvalidParam {
                        indicator: "n_order_ema",
                        key: "order",
                        value: v,
                        reason: "expected integer > 0",
                    }
                    .into());
                }
                let r = v.round();
                if (v - r).abs() > 1.0e-9 {
                    return Err(MaBatchDispatchError::InvalidParam {
                        indicator: "n_order_ema",
                        key: "order",
                        value: v,
                        reason: "expected integer",
                    }
                    .into());
                }
                r as usize
            }
            None => 1usize,
        };

        let ema_style = text
            .get("ema_style")
            .cloned()
            .unwrap_or_else(|| "ema".to_string());
        let iir_style = text
            .get("iir_style")
            .cloned()
            .unwrap_or_else(|| "impulse_matched".to_string());

        let out = super::n_order_ema::n_order_ema_batch_with_kernel(
            prices,
            &super::n_order_ema::NOrderEmaBatchRange {
                period: (
                    period_range.0 as f64,
                    period_range.1 as f64,
                    period_range.2 as f64,
                ),
                order: (order, order, 0),
            },
            &super::n_order_ema::NOrderEmaParams {
                period: None,
                order: None,
                ema_style: Some(ema_style),
                iir_style: Some(iir_style),
            },
            kernel,
        )?;
        return Ok(MaBatchOutput {
            periods: map_periods(&out.combos, |p| p.period.unwrap_or(9.0).round() as usize),
            values: out.values,
            rows: out.rows,
            cols: out.cols,
        });
    }

    ma_batch_with_kernel_and_params(ma_type, data, period_range, kernel, Some(&numeric))
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
        "corrected_moving_average" | "cma" => {
            let sweep = super::corrected_moving_average::CorrectedMovingAverageBatchRange {
                period: period_range,
            };
            let out = super::corrected_moving_average::corrected_moving_average_batch_with_kernel(
                prices, &sweep, kernel,
            )?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(35)),
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
            let out =
                crate::indicators::cora_wave::cora_wave_batch_with_kernel(prices, &sweep, kernel)?;
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
        "n_order_ema" => {
            let out = super::n_order_ema::n_order_ema_batch_with_kernel(
                prices,
                &super::n_order_ema::NOrderEmaBatchRange {
                    period: (
                        period_range.0 as f64,
                        period_range.1 as f64,
                        period_range.2 as f64,
                    ),
                    order: (1, 1, 0),
                },
                &super::n_order_ema::NOrderEmaParams {
                    period: None,
                    order: None,
                    ema_style: Some("ema".to_string()),
                    iir_style: Some("impulse_matched".to_string()),
                },
                kernel,
            )?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(9.0).round() as usize),
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
        "sgf" => {
            let poly_order = get_usize(params, "sgf", "poly_order")?.unwrap_or(2);
            let sweep = super::sgf::SgfBatchRange {
                period: period_range,
                poly_order: (poly_order, poly_order, 0),
            };
            let out = super::sgf::sgf_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(21)),
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
        "corrected_moving_average" => {
            let sweep = super::corrected_moving_average::CorrectedMovingAverageBatchRange {
                period: period_range,
            };
            let out = super::corrected_moving_average::corrected_moving_average_batch_with_kernel(
                prices, &sweep, kernel,
            )?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(35)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "ema_deviation_corrected_t3" => {
            let sweep = super::ema_deviation_corrected_t3::EmaDeviationCorrectedT3BatchRange {
                period: period_range,
                hot: {
                    let v = get_f64(params, "ema_deviation_corrected_t3", "hot")?.unwrap_or(0.7);
                    (v, v, 0.0)
                },
                t3_mode: {
                    let v =
                        get_usize(params, "ema_deviation_corrected_t3", "t3_mode")?.unwrap_or(0);
                    (v, v, 0)
                },
            };
            let out =
                super::ema_deviation_corrected_t3::ema_deviation_corrected_t3_batch_with_kernel(
                    prices, &sweep, kernel,
                )?;
            let periods = map_periods(&out.combos, |p| p.period.unwrap_or(10));
            let rows = out.rows;
            let cols = out.cols;
            Ok(MaBatchOutput {
                periods,
                values: out.corrected,
                rows,
                cols,
            })
        }
        "wave_smoother" => {
            let sweep = super::wave_smoother::WaveSmootherBatchRange {
                period: period_range,
                phase: {
                    let v = get_f64(params, "wave_smoother", "phase")?.unwrap_or(70.0);
                    (v, v, 0.0)
                },
            };
            let out =
                super::wave_smoother::wave_smoother_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.period.unwrap_or(20)),
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
        "elastic_volume_weighted_moving_average" => {
            let candles = candles.ok_or(MaBatchDispatchError::RequiresCandles {
                indicator: "elastic_volume_weighted_moving_average",
            })?;
            let mut sweep =
                super::elastic_volume_weighted_moving_average::ElasticVolumeWeightedMovingAverageBatchRange::default();
            sweep.length = period_range;
            if let Some(v) = get_f64(
                params,
                "elastic_volume_weighted_moving_average",
                "absolute_volume_millions",
            )? {
                sweep.absolute_volume_millions = Some(v);
            }
            if let Some(v) = get_usize(params, "elastic_volume_weighted_moving_average", "length")?
            {
                sweep.length = (v, v, 0);
            }
            if let Some(v) = get_usize(
                params,
                "elastic_volume_weighted_moving_average",
                "use_volume_sum",
            )? {
                sweep.use_volume_sum = Some(match v {
                    0 => false,
                    1 => true,
                    other => {
                        return Err(MaBatchDispatchError::InvalidParam {
                            indicator: "elastic_volume_weighted_moving_average",
                            key: "use_volume_sum",
                            value: other as f64,
                            reason: "expected 0 or 1",
                        }
                        .into());
                    }
                });
            } else {
                sweep.use_volume_sum = Some(true);
            }
            let out = super::elastic_volume_weighted_moving_average::elastic_volume_weighted_moving_average_batch_with_kernel(
                prices,
                &candles.volume,
                &sweep,
                kernel,
            )?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.length.unwrap_or(30)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "tradjema" => {
            let candles = candles.ok_or(MaBatchDispatchError::RequiresCandles {
                indicator: "tradjema",
            })?;
            let mut sweep = super::tradjema::TradjemaBatchRange::default();
            sweep.length = period_range;
            if let Some(v) = get_f64(params, "tradjema", "mult")? {
                sweep.mult = (v, v, 0.0);
            }
            let out = super::tradjema::tradjema_batch_with_kernel(
                &candles.high,
                &candles.low,
                &candles.close,
                &sweep,
                kernel,
            )?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.length.unwrap_or(40)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "uma" => {
            let mut sweep = super::uma::UmaBatchRange::default();
            sweep.max_length = period_range;
            if let Some(v) = get_f64(params, "uma", "accelerator")? {
                sweep.accelerator = (v, v, 0.0);
            }
            if let Some(v) = get_usize(params, "uma", "min_length")? {
                sweep.min_length = (v, v, 0);
            }
            if let Some(v) = get_usize(params, "uma", "max_length")? {
                sweep.max_length = (v, v, 0);
            }
            if let Some(v) = get_usize(params, "uma", "smooth_length")? {
                sweep.smooth_length = (v, v, 0);
            }
            let volumes = candles.map(|c| c.volume.as_slice());
            let out = super::uma::uma_batch_with_kernel(prices, volumes, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.max_length.unwrap_or(50)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "volume_adjusted_ma" => {
            let candles = candles.ok_or(MaBatchDispatchError::RequiresCandles {
                indicator: "volume_adjusted_ma",
            })?;
            let mut sweep = super::volume_adjusted_ma::VolumeAdjustedMaBatchRange::default();
            sweep.length = period_range;
            if let Some(v) = get_f64(params, "volume_adjusted_ma", "vi_factor")? {
                sweep.vi_factor = (v, v, 0.0);
            }
            if let Some(v) = get_usize(params, "volume_adjusted_ma", "sample_period")? {
                sweep.sample_period = (v, v, 0);
            }
            if let Some(v) = get_usize(params, "volume_adjusted_ma", "strict")? {
                sweep.strict = Some(match v {
                    0 => false,
                    1 => true,
                    other => {
                        return Err(MaBatchDispatchError::InvalidParam {
                            indicator: "volume_adjusted_ma",
                            key: "strict",
                            value: other as f64,
                            reason: "expected 0 or 1",
                        }
                        .into());
                    }
                });
            }
            let out = super::volume_adjusted_ma::VolumeAdjustedMa_batch_with_kernel(
                prices,
                &candles.volume,
                &sweep,
                kernel,
            )?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.length.unwrap_or(13)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "hwma" => {
            let mut sweep = super::hwma::HwmaBatchRange::default();
            if let Some(v) = get_f64(params, "hwma", "na")? {
                sweep.na = (v, v, 0.0);
            }
            if let Some(v) = get_f64(params, "hwma", "nb")? {
                sweep.nb = (v, v, 0.0);
            }
            if let Some(v) = get_f64(params, "hwma", "nc")? {
                sweep.nc = (v, v, 0.0);
            }
            let out = super::hwma::hwma_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: (1..=out.rows).collect(),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "mama" => {
            let mut sweep = super::mama::MamaBatchRange::default();
            if let Some(v) = get_f64(params, "mama", "fast_limit")? {
                sweep.fast_limit = (v, v, 0.0);
            } else {
                if let Some(v) = get_f64(params, "mama", "fast_limit_start")? {
                    sweep.fast_limit.0 = v;
                }
                if let Some(v) = get_f64(params, "mama", "fast_limit_end")? {
                    sweep.fast_limit.1 = v;
                }
                if let Some(v) = get_f64(params, "mama", "fast_limit_step")? {
                    sweep.fast_limit.2 = v;
                }
            }
            if let Some(v) = get_f64(params, "mama", "slow_limit")? {
                sweep.slow_limit = (v, v, 0.0);
            } else {
                if let Some(v) = get_f64(params, "mama", "slow_limit_start")? {
                    sweep.slow_limit.0 = v;
                }
                if let Some(v) = get_f64(params, "mama", "slow_limit_end")? {
                    sweep.slow_limit.1 = v;
                }
                if let Some(v) = get_f64(params, "mama", "slow_limit_step")? {
                    sweep.slow_limit.2 = v;
                }
            }
            let out = super::mama::mama_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: (1..=out.rows).collect(),
                values: out.mama_values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "ehlers_pma" => {
            let periods = expand_period_axis(period_range)?;
            let rows = periods.len();
            let input = super::ehlers_pma::EhlersPmaInput::from_slice(
                prices,
                super::ehlers_pma::EhlersPmaParams::default(),
            );
            let out = super::ehlers_pma::ehlers_pma_with_kernel(&input, kernel)?;
            let cols = out.predict.len();
            let mut values = Vec::with_capacity(rows.saturating_mul(cols));
            for _ in 0..rows {
                values.extend_from_slice(&out.predict);
            }
            Ok(MaBatchOutput {
                periods,
                values,
                rows,
                cols,
            })
        }
        "ehlers_undersampled_double_moving_average" => {
            let mut sweep =
                super::ehlers_undersampled_double_moving_average::EhlersUndersampledDoubleMovingAverageBatchRange::default();
            if let Some(v) = get_usize(
                params,
                "ehlers_undersampled_double_moving_average",
                "fast_length",
            )? {
                sweep.fast_length = (v, v, 0);
            } else {
                if let Some(v) = get_usize(
                    params,
                    "ehlers_undersampled_double_moving_average",
                    "fast_length_start",
                )? {
                    sweep.fast_length.0 = v;
                }
                if let Some(v) = get_usize(
                    params,
                    "ehlers_undersampled_double_moving_average",
                    "fast_length_end",
                )? {
                    sweep.fast_length.1 = v;
                }
                if let Some(v) = get_usize(
                    params,
                    "ehlers_undersampled_double_moving_average",
                    "fast_length_step",
                )? {
                    sweep.fast_length.2 = v;
                }
            }
            if let Some(v) = get_usize(
                params,
                "ehlers_undersampled_double_moving_average",
                "slow_length",
            )? {
                sweep.slow_length = (v, v, 0);
            } else {
                if let Some(v) = get_usize(
                    params,
                    "ehlers_undersampled_double_moving_average",
                    "slow_length_start",
                )? {
                    sweep.slow_length.0 = v;
                }
                if let Some(v) = get_usize(
                    params,
                    "ehlers_undersampled_double_moving_average",
                    "slow_length_end",
                )? {
                    sweep.slow_length.1 = v;
                }
                if let Some(v) = get_usize(
                    params,
                    "ehlers_undersampled_double_moving_average",
                    "slow_length_step",
                )? {
                    sweep.slow_length.2 = v;
                }
            }
            if let Some(v) = get_usize(
                params,
                "ehlers_undersampled_double_moving_average",
                "sample_length",
            )? {
                sweep.sample_length = (v, v, 0);
            } else {
                if let Some(v) = get_usize(
                    params,
                    "ehlers_undersampled_double_moving_average",
                    "sample_length_start",
                )? {
                    sweep.sample_length.0 = v;
                }
                if let Some(v) = get_usize(
                    params,
                    "ehlers_undersampled_double_moving_average",
                    "sample_length_end",
                )? {
                    sweep.sample_length.1 = v;
                }
                if let Some(v) = get_usize(
                    params,
                    "ehlers_undersampled_double_moving_average",
                    "sample_length_step",
                )? {
                    sweep.sample_length.2 = v;
                }
            }
            let out =
                super::ehlers_undersampled_double_moving_average::ehlers_undersampled_double_moving_average_batch_with_kernel(
                    prices, &sweep, kernel,
                )?;
            Ok(MaBatchOutput {
                periods: (1..=out.rows).collect(),
                values: out.fast_values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "mwdx" => {
            let mut sweep = super::mwdx::MwdxBatchRange::default();
            if let Some(v) = get_f64(params, "mwdx", "factor")? {
                sweep.factor = (v, v, 0.0);
            } else {
                let fac_start = 2.0 / (period_range.0 as f64 + 1.0);
                let fac_end = 2.0 / (period_range.1 as f64 + 1.0);
                let next_period = if period_range.2 == 0 || period_range.0 == period_range.1 {
                    period_range.0
                } else if period_range.0 < period_range.1 {
                    period_range.0.saturating_add(period_range.2)
                } else {
                    period_range.0.saturating_sub(period_range.2)
                };
                let fac_next = 2.0 / (next_period as f64 + 1.0);
                let fac_step = (fac_next - fac_start).abs();
                sweep.factor = (fac_start, fac_end, fac_step);
            }
            let out = super::mwdx::mwdx_batch_with_kernel(prices, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| {
                    let f = p.factor.unwrap_or(0.2);
                    if f > 0.0 {
                        ((2.0 / f) - 1.0).round().max(1.0) as usize
                    } else {
                        1
                    }
                }),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "vwap" => {
            let candles =
                candles.ok_or(MaBatchDispatchError::RequiresCandles { indicator: "vwap" })?;
            let sweep = super::vwap::VwapBatchRange {
                anchor: ("1d".to_string(), "1d".to_string(), 0),
            };
            let out = super::vwap::vwap_batch_with_kernel(
                &candles.timestamp,
                &candles.volume,
                prices,
                &sweep,
                kernel,
            )?;
            let periods = out
                .combos
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    p.anchor
                        .as_deref()
                        .and_then(|a| super::vwap::parse_anchor(a).ok().map(|(n, _)| n as usize))
                        .unwrap_or(i + 1)
                })
                .collect();
            Ok(MaBatchOutput {
                periods,
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
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
            let out = super::frama::frama_batch_with_kernel(high, low, close, &sweep, kernel)?;
            Ok(MaBatchOutput {
                periods: map_periods(&out.combos, |p| p.window.unwrap_or(10)),
                values: out.values,
                rows: out.rows,
                cols: out.cols,
            })
        }
        "buff_averages" => {
            let candles = candles.ok_or(MaBatchDispatchError::RequiresCandles {
                indicator: "buff_averages",
            })?;
            let mut sweep = super::buff_averages::BuffAveragesBatchRange::default();
            sweep.slow_period = period_range;

            if let Some(v) = get_usize(params, "buff_averages", "fast_period")? {
                sweep.fast_period = (v, v, 0);
            }
            if let Some(v) = get_usize(params, "buff_averages", "slow_period")? {
                sweep.slow_period = (v, v, 0);
            }
            if let Some(v) = get_usize(params, "buff_averages", "fast_period_start")? {
                sweep.fast_period.0 = v;
            }
            if let Some(v) = get_usize(params, "buff_averages", "fast_period_end")? {
                sweep.fast_period.1 = v;
            }
            if let Some(v) = get_usize(params, "buff_averages", "fast_period_step")? {
                sweep.fast_period.2 = v;
            }
            if let Some(v) = get_usize(params, "buff_averages", "slow_period_start")? {
                sweep.slow_period.0 = v;
            }
            if let Some(v) = get_usize(params, "buff_averages", "slow_period_end")? {
                sweep.slow_period.1 = v;
            }
            if let Some(v) = get_usize(params, "buff_averages", "slow_period_step")? {
                sweep.slow_period.2 = v;
            }

            let out = super::buff_averages::buff_averages_batch_with_kernel(
                prices,
                &candles.volume,
                &sweep,
                kernel,
            )?;

            let all_fast_same = out
                .combos
                .first()
                .map(|c| out.combos.iter().all(|x| x.0 == c.0))
                .unwrap_or(true);
            let all_slow_same = out
                .combos
                .first()
                .map(|c| out.combos.iter().all(|x| x.1 == c.1))
                .unwrap_or(true);
            let periods = if all_fast_same {
                out.combos.iter().map(|c| c.1).collect()
            } else if all_slow_same {
                out.combos.iter().map(|c| c.0).collect()
            } else {
                (1..=out.rows).collect()
            };

            Ok(MaBatchOutput {
                periods,
                values: out.fast,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::moving_averages::ma::ma_with_kernel;
    use crate::utilities::data_loader::Candles;
    use crate::utilities::enums::Kernel;

    fn sample_prices(len: usize) -> Vec<f64> {
        (0..len)
            .map(|i| ((i as f64) * 0.1).sin() + (i as f64) * 0.001 + 100.0)
            .collect()
    }

    fn sample_candles(len: usize) -> Candles {
        let timestamp: Vec<i64> = (0..len)
            .map(|i| 1_700_000_000_000_i64 + (i as i64) * 60_000)
            .collect();
        let close = sample_prices(len);
        let open: Vec<f64> = close.iter().map(|v| v - 0.1).collect();
        let high: Vec<f64> = close
            .iter()
            .enumerate()
            .map(|(i, v)| v + 0.35 + ((i as f64) * 0.01).sin().abs())
            .collect();
        let low: Vec<f64> = close
            .iter()
            .enumerate()
            .map(|(i, v)| v - 0.35 - ((i as f64) * 0.01).sin().abs())
            .collect();
        let volume: Vec<f64> = (0..len)
            .map(|i| 1000.0 + ((i % 31) as f64) * 7.0 + (i as f64) * 0.1)
            .collect();
        Candles::new(timestamp, open, high, low, close, volume)
    }

    fn assert_series_eq(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len());
        for (i, (&av, &bv)) in a.iter().zip(b.iter()).enumerate() {
            if av.is_nan() && bv.is_nan() {
                continue;
            }
            let d = (av - bv).abs();
            assert!(
                d <= tol,
                "series mismatch at index {i}: left={av}, right={bv}, abs_diff={d}"
            );
        }
    }

    fn assert_series_eq_ctx(a: &[f64], b: &[f64], tol: f64, ctx: &str) {
        assert_eq!(a.len(), b.len(), "length mismatch for {ctx}");
        for (i, (&av, &bv)) in a.iter().zip(b.iter()).enumerate() {
            if av.is_nan() && bv.is_nan() {
                continue;
            }
            let d = (av - bv).abs();
            assert!(
                d <= tol,
                "series mismatch for {ctx} at index {i}: left={av}, right={bv}, abs_diff={d}"
            );
        }
    }

    #[test]
    fn period_based_batch_matches_single_direct_for_many_ids() {
        let prices = sample_prices(320);
        let candles = sample_candles(320);
        let period_range = (18, 22, 2);
        let expected_periods = vec![18, 20, 22];

        let slice_cases = [
            "sma",
            "ema",
            "dema",
            "tema",
            "smma",
            "zlema",
            "wma",
            "alma",
            "cwma",
            "corrected_moving_average",
            "cora_wave",
            "edcf",
            "fwma",
            "gaussian",
            "highpass",
            "highpass_2_pole",
            "hma",
            "jma",
            "jsa",
            "linreg",
            "kama",
            "ehlers_kama",
            "ehlers_ecema",
            "ehma",
            "nama",
            "nma",
            "pwma",
            "reflex",
            "sinwma",
            "sqwma",
            "srwma",
            "sgf",
            "swma",
            "supersmoother",
            "supersmoother_3_pole",
            "tilson",
            "trendflex",
            "corrected_moving_average",
            "ema_deviation_corrected_t3",
            "wave_smoother",
            "trima",
            "wilders",
            "epma",
            "sama",
        ];

        for ma_type in slice_cases {
            let batch =
                ma_batch_with_kernel(ma_type, MaData::Slice(&prices), period_range, Kernel::Auto)
                    .unwrap();

            assert_eq!(batch.periods, expected_periods);
            assert_eq!(batch.rows, expected_periods.len());
            assert_eq!(batch.cols, prices.len());

            for (row, period) in expected_periods.iter().copied().enumerate() {
                let direct =
                    ma_with_kernel(ma_type, MaData::Slice(&prices), period, Kernel::Auto).unwrap();
                let start = row * batch.cols;
                let end = start + batch.cols;
                let ctx = format!("{ma_type} slice period={period}");
                assert_series_eq_ctx(&batch.values[start..end], &direct, 1e-10, &ctx);
            }
        }

        let candle_cases = ["vpwma", "vwma", "frama"];
        for ma_type in candle_cases {
            let batch = ma_batch_with_kernel(
                ma_type,
                MaData::Candles {
                    candles: &candles,
                    source: "close",
                },
                period_range,
                Kernel::Auto,
            )
            .unwrap();

            assert_eq!(batch.periods, expected_periods);
            assert_eq!(batch.rows, expected_periods.len());
            assert_eq!(batch.cols, candles.close.len());

            for (row, period) in expected_periods.iter().copied().enumerate() {
                let direct = ma_with_kernel(
                    ma_type,
                    MaData::Candles {
                        candles: &candles,
                        source: "close",
                    },
                    period,
                    Kernel::Auto,
                )
                .unwrap();
                let start = row * batch.cols;
                let end = start + batch.cols;
                let ctx = format!("{ma_type} candles period={period}");
                assert_series_eq_ctx(&batch.values[start..end], &direct, 1e-10, &ctx);
            }
        }
    }

    #[test]
    fn mama_typed_output_selection_matches_direct() {
        let prices = sample_prices(256);
        let data = MaData::Slice(&prices);
        let params = [
            MaBatchParamKV {
                key: "fast_limit",
                value: MaBatchParamValue::Float(0.35),
            },
            MaBatchParamKV {
                key: "slow_limit",
                value: MaBatchParamValue::Float(0.06),
            },
            MaBatchParamKV {
                key: "output",
                value: MaBatchParamValue::EnumString("fama"),
            },
        ];

        let got =
            ma_batch_with_kernel_and_typed_params("mama", data, (10, 10, 0), Kernel::Auto, &params)
                .unwrap();

        let direct = crate::indicators::moving_averages::mama::mama_batch_with_kernel(
            &prices,
            &crate::indicators::moving_averages::mama::MamaBatchRange {
                fast_limit: (0.35, 0.35, 0.0),
                slow_limit: (0.06, 0.06, 0.0),
            },
            Kernel::Auto,
        )
        .unwrap();

        assert_eq!(got.rows, direct.rows);
        assert_eq!(got.cols, direct.cols);
        assert_eq!(got.periods, (1..=direct.rows).collect::<Vec<_>>());
        assert_series_eq(&got.values, &direct.fama_values, 1e-12);
    }

    #[test]
    fn ehlers_pma_typed_output_selection_matches_direct() {
        let prices = sample_prices(300);
        let data = MaData::Slice(&prices);
        let params = [MaBatchParamKV {
            key: "output",
            value: MaBatchParamValue::EnumString("trigger"),
        }];

        let got = ma_batch_with_kernel_and_typed_params(
            "ehlers_pma",
            data,
            (8, 10, 1),
            Kernel::Auto,
            &params,
        )
        .unwrap();

        let input = crate::indicators::moving_averages::ehlers_pma::EhlersPmaInput::from_slice(
            &prices,
            crate::indicators::moving_averages::ehlers_pma::EhlersPmaParams::default(),
        );
        let direct = crate::indicators::moving_averages::ehlers_pma::ehlers_pma_with_kernel(
            &input,
            Kernel::Auto,
        )
        .unwrap();

        assert_eq!(got.rows, 3);
        assert_eq!(got.cols, prices.len());
        assert_eq!(got.periods, vec![8, 9, 10]);
        for row in 0..got.rows {
            let start = row * got.cols;
            let end = start + got.cols;
            assert_series_eq(&got.values[start..end], &direct.trigger, 1e-12);
        }
    }

    #[test]
    fn invalid_output_selection_returns_error() {
        let prices = sample_prices(256);
        let data = MaData::Slice(&prices);
        let params = [MaBatchParamKV {
            key: "output",
            value: MaBatchParamValue::EnumString("bad_line"),
        }];

        let err =
            ma_batch_with_kernel_and_typed_params("mama", data, (10, 10, 0), Kernel::Auto, &params)
                .unwrap_err()
                .to_string();

        assert!(err.contains("expected 'mama' or 'fama'"));
    }

    #[test]
    fn mama_numeric_path_defaults_to_primary_output() {
        let prices = sample_prices(256);
        let mut params = HashMap::new();
        params.insert("fast_limit".to_string(), 0.4);
        params.insert("slow_limit".to_string(), 0.07);

        let got = ma_batch_with_kernel_and_params(
            "mama",
            MaData::Slice(&prices),
            (12, 12, 0),
            Kernel::Auto,
            Some(&params),
        )
        .unwrap();

        let direct = crate::indicators::moving_averages::mama::mama_batch_with_kernel(
            &prices,
            &crate::indicators::moving_averages::mama::MamaBatchRange {
                fast_limit: (0.4, 0.4, 0.0),
                slow_limit: (0.07, 0.07, 0.0),
            },
            Kernel::Auto,
        )
        .unwrap();

        assert_eq!(got.rows, direct.rows);
        assert_eq!(got.cols, direct.cols);
        assert_series_eq(&got.values, &direct.mama_values, 1e-12);
    }

    #[test]
    fn ehlers_pma_numeric_path_defaults_and_descending_periods() {
        let prices = sample_prices(300);
        let got = ma_batch_with_kernel_and_params(
            "ehlers_pma",
            MaData::Slice(&prices),
            (10, 8, 1),
            Kernel::Auto,
            None,
        )
        .unwrap();

        let input = crate::indicators::moving_averages::ehlers_pma::EhlersPmaInput::from_slice(
            &prices,
            crate::indicators::moving_averages::ehlers_pma::EhlersPmaParams::default(),
        );
        let direct = crate::indicators::moving_averages::ehlers_pma::ehlers_pma_with_kernel(
            &input,
            Kernel::Auto,
        )
        .unwrap();

        assert_eq!(got.periods, vec![10, 9, 8]);
        assert_eq!(got.rows, 3);
        assert_eq!(got.cols, prices.len());
        for row in 0..got.rows {
            let start = row * got.cols;
            let end = start + got.cols;
            assert_series_eq(&got.values[start..end], &direct.predict, 1e-12);
        }
    }

    #[test]
    fn hwma_typed_params_match_direct() {
        let prices = sample_prices(256);
        let params = [
            MaBatchParamKV {
                key: "na",
                value: MaBatchParamValue::Float(0.23),
            },
            MaBatchParamKV {
                key: "nb",
                value: MaBatchParamValue::Float(0.11),
            },
            MaBatchParamKV {
                key: "nc",
                value: MaBatchParamValue::Float(0.17),
            },
        ];
        let got = ma_batch_with_kernel_and_typed_params(
            "hwma",
            MaData::Slice(&prices),
            (10, 10, 0),
            Kernel::Auto,
            &params,
        )
        .unwrap();
        let direct = crate::indicators::moving_averages::hwma::hwma_batch_with_kernel(
            &prices,
            &crate::indicators::moving_averages::hwma::HwmaBatchRange {
                na: (0.23, 0.23, 0.0),
                nb: (0.11, 0.11, 0.0),
                nc: (0.17, 0.17, 0.0),
            },
            Kernel::Auto,
        )
        .unwrap();
        assert_eq!(got.rows, direct.rows);
        assert_eq!(got.cols, direct.cols);
        assert_series_eq(&got.values, &direct.values, 1e-12);
    }

    #[test]
    fn mwdx_typed_factor_matches_direct() {
        let prices = sample_prices(256);
        let params = [MaBatchParamKV {
            key: "factor",
            value: MaBatchParamValue::Float(2.0 / 11.0),
        }];
        let got = ma_batch_with_kernel_and_typed_params(
            "mwdx",
            MaData::Slice(&prices),
            (10, 10, 0),
            Kernel::Auto,
            &params,
        )
        .unwrap();
        let direct = crate::indicators::moving_averages::mwdx::mwdx_batch_with_kernel(
            &prices,
            &crate::indicators::moving_averages::mwdx::MwdxBatchRange {
                factor: (2.0 / 11.0, 2.0 / 11.0, 0.0),
            },
            Kernel::Auto,
        )
        .unwrap();
        assert_eq!(got.rows, direct.rows);
        assert_eq!(got.cols, direct.cols);
        assert_series_eq(&got.values, &direct.values, 1e-12);
    }

    #[test]
    fn uma_typed_params_match_direct() {
        let prices = sample_prices(256);
        let params = [
            MaBatchParamKV {
                key: "accelerator",
                value: MaBatchParamValue::Float(1.0),
            },
            MaBatchParamKV {
                key: "min_length",
                value: MaBatchParamValue::Int(5),
            },
            MaBatchParamKV {
                key: "max_length",
                value: MaBatchParamValue::Int(35),
            },
            MaBatchParamKV {
                key: "smooth_length",
                value: MaBatchParamValue::Int(4),
            },
        ];
        let got = ma_batch_with_kernel_and_typed_params(
            "uma",
            MaData::Slice(&prices),
            (35, 35, 0),
            Kernel::Auto,
            &params,
        )
        .unwrap();
        let direct = crate::indicators::moving_averages::uma::uma_batch_with_kernel(
            &prices,
            None,
            &crate::indicators::moving_averages::uma::UmaBatchRange {
                accelerator: (1.0, 1.0, 0.0),
                min_length: (5, 5, 0),
                max_length: (35, 35, 0),
                smooth_length: (4, 4, 0),
            },
            Kernel::Auto,
        )
        .unwrap();
        assert_eq!(got.rows, direct.rows);
        assert_eq!(got.cols, direct.cols);
        assert_series_eq(&got.values, &direct.values, 1e-12);
    }

    #[test]
    fn tradjema_typed_params_match_direct() {
        let candles = sample_candles(300);
        let params = [MaBatchParamKV {
            key: "mult",
            value: MaBatchParamValue::Float(2.3),
        }];
        let got = ma_batch_with_kernel_and_typed_params(
            "tradjema",
            MaData::Candles {
                candles: &candles,
                source: "close",
            },
            (40, 40, 0),
            Kernel::Auto,
            &params,
        )
        .unwrap();
        let direct = crate::indicators::moving_averages::tradjema::tradjema_batch_with_kernel(
            &candles.high,
            &candles.low,
            &candles.close,
            &crate::indicators::moving_averages::tradjema::TradjemaBatchRange {
                length: (40, 40, 0),
                mult: (2.3, 2.3, 0.0),
            },
            Kernel::Auto,
        )
        .unwrap();
        assert_eq!(got.rows, direct.rows);
        assert_eq!(got.cols, direct.cols);
        assert_series_eq(&got.values, &direct.values, 1e-12);
    }

    #[test]
    fn volume_adjusted_ma_typed_params_match_direct() {
        let candles = sample_candles(300);
        let params = [
            MaBatchParamKV {
                key: "vi_factor",
                value: MaBatchParamValue::Float(2.0),
            },
            MaBatchParamKV {
                key: "sample_period",
                value: MaBatchParamValue::Int(30),
            },
            MaBatchParamKV {
                key: "strict",
                value: MaBatchParamValue::Bool(true),
            },
        ];
        let got = ma_batch_with_kernel_and_typed_params(
            "volume_adjusted_ma",
            MaData::Candles {
                candles: &candles,
                source: "close",
            },
            (20, 20, 0),
            Kernel::Auto,
            &params,
        )
        .unwrap();
        let direct =
            crate::indicators::moving_averages::volume_adjusted_ma::VolumeAdjustedMa_batch_with_kernel(
                &candles.close,
                &candles.volume,
                &crate::indicators::moving_averages::volume_adjusted_ma::VolumeAdjustedMaBatchRange {
                    length: (20, 20, 0),
                    vi_factor: (2.0, 2.0, 0.0),
                    sample_period: (30, 30, 0),
                    strict: Some(true),
                },
                Kernel::Auto,
            )
            .unwrap();
        assert_eq!(got.rows, direct.rows);
        assert_eq!(got.cols, direct.cols);
        assert_series_eq(&got.values, &direct.values, 1e-12);
    }

    #[test]
    fn vwap_typed_anchor_matches_direct() {
        let candles = sample_candles(300);
        let params = [MaBatchParamKV {
            key: "anchor",
            value: MaBatchParamValue::EnumString("1d"),
        }];
        let got = ma_batch_with_kernel_and_typed_params(
            "vwap",
            MaData::Candles {
                candles: &candles,
                source: "close",
            },
            (10, 10, 0),
            Kernel::Auto,
            &params,
        )
        .unwrap();
        let direct = crate::indicators::moving_averages::vwap::vwap_batch_with_kernel(
            &candles.timestamp,
            &candles.volume,
            &candles.close,
            &crate::indicators::moving_averages::vwap::VwapBatchRange {
                anchor: ("1d".to_string(), "1d".to_string(), 0),
            },
            Kernel::Auto,
        )
        .unwrap();
        assert_eq!(got.rows, direct.rows);
        assert_eq!(got.cols, direct.cols);
        assert_series_eq(&got.values, &direct.values, 1e-12);
    }

    #[test]
    fn dma_typed_hull_ma_type_matches_direct() {
        let prices = sample_prices(256);
        let params = [
            MaBatchParamKV {
                key: "ema_length",
                value: MaBatchParamValue::Int(20),
            },
            MaBatchParamKV {
                key: "ema_gain_limit",
                value: MaBatchParamValue::Int(50),
            },
            MaBatchParamKV {
                key: "hull_ma_type",
                value: MaBatchParamValue::EnumString("EMA"),
            },
        ];
        let got = ma_batch_with_kernel_and_typed_params(
            "dma",
            MaData::Slice(&prices),
            (14, 14, 0),
            Kernel::Auto,
            &params,
        )
        .unwrap();
        let direct = crate::indicators::moving_averages::dma::dma_batch_with_kernel(
            &prices,
            &crate::indicators::moving_averages::dma::DmaBatchRange {
                hull_length: (14, 14, 0),
                ema_length: (20, 20, 0),
                ema_gain_limit: (50, 50, 0),
                hull_ma_type: "EMA".to_string(),
            },
            Kernel::Auto,
        )
        .unwrap();
        assert_eq!(got.rows, direct.rows);
        assert_eq!(got.cols, direct.cols);
        assert_series_eq(&got.values, &direct.values, 1e-12);
    }

    #[test]
    fn ehlers_itrend_typed_params_match_direct() {
        let prices = sample_prices(320);
        let params = [MaBatchParamKV {
            key: "warmup_bars",
            value: MaBatchParamValue::Int(30),
        }];
        let got = ma_batch_with_kernel_and_typed_params(
            "ehlers_itrend",
            MaData::Slice(&prices),
            (48, 48, 0),
            Kernel::Auto,
            &params,
        )
        .unwrap();
        let direct =
            crate::indicators::moving_averages::ehlers_itrend::ehlers_itrend_batch_with_kernel(
                &prices,
                &crate::indicators::moving_averages::ehlers_itrend::EhlersITrendBatchRange {
                    warmup_bars: (30, 30, 0),
                    max_dc_period: (48, 48, 0),
                },
                Kernel::Auto,
            )
            .unwrap();
        assert_eq!(got.rows, direct.rows);
        assert_eq!(got.cols, direct.cols);
        assert_series_eq(&got.values, &direct.values, 1e-12);
    }

    #[test]
    fn vama_typed_params_match_direct() {
        let prices = sample_prices(320);
        let params = [MaBatchParamKV {
            key: "vol_period",
            value: MaBatchParamValue::Int(51),
        }];
        let got = ma_batch_with_kernel_and_typed_params(
            "vama",
            MaData::Slice(&prices),
            (18, 22, 2),
            Kernel::Auto,
            &params,
        )
        .unwrap();
        let direct =
            crate::indicators::moving_averages::volatility_adjusted_ma::vama_batch_with_kernel(
                &prices,
                &crate::indicators::moving_averages::volatility_adjusted_ma::VamaBatchRange {
                    base_period: (18, 22, 2),
                    vol_period: (51, 51, 0),
                },
                Kernel::Auto,
            )
            .unwrap();
        assert_eq!(got.rows, direct.rows);
        assert_eq!(got.cols, direct.cols);
        assert_series_eq(&got.values, &direct.values, 1e-12);
    }

    #[test]
    fn maaq_typed_params_match_direct() {
        let prices = sample_prices(320);
        let params = [
            MaBatchParamKV {
                key: "fast_period",
                value: MaBatchParamValue::Int(2),
            },
            MaBatchParamKV {
                key: "slow_period",
                value: MaBatchParamValue::Int(30),
            },
        ];
        let got = ma_batch_with_kernel_and_typed_params(
            "maaq",
            MaData::Slice(&prices),
            (18, 22, 2),
            Kernel::Auto,
            &params,
        )
        .unwrap();
        let direct = crate::indicators::moving_averages::maaq::maaq_batch_with_kernel(
            &prices,
            &crate::indicators::moving_averages::maaq::MaaqBatchRange {
                period: (18, 22, 2),
                fast_period: (2, 2, 0),
                slow_period: (30, 30, 0),
            },
            Kernel::Auto,
        )
        .unwrap();
        assert_eq!(got.rows, direct.rows);
        assert_eq!(got.cols, direct.cols);
        assert_series_eq(&got.values, &direct.values, 1e-12);
    }

    #[test]
    fn tradjema_requires_candles_error() {
        let prices = sample_prices(256);
        let err = ma_batch_with_kernel_and_typed_params(
            "tradjema",
            MaData::Slice(&prices),
            (40, 40, 0),
            Kernel::Auto,
            &[],
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("requires candles"));
    }

    #[test]
    fn vwap_requires_candles_error() {
        let prices = sample_prices(256);
        let err = ma_batch_with_kernel_and_typed_params(
            "vwap",
            MaData::Slice(&prices),
            (10, 10, 0),
            Kernel::Auto,
            &[],
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("requires candles"));
    }

    #[test]
    fn volume_adjusted_ma_requires_candles_error() {
        let prices = sample_prices(256);
        let err = ma_batch_with_kernel_and_typed_params(
            "volume_adjusted_ma",
            MaData::Slice(&prices),
            (20, 20, 0),
            Kernel::Auto,
            &[],
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("requires candles"));
    }

    #[test]
    fn volume_adjusted_ma_invalid_strict_numeric_error() {
        let candles = sample_candles(300);
        let mut params = HashMap::new();
        params.insert("strict".to_string(), 2.0);
        let err = ma_batch_with_kernel_and_params(
            "volume_adjusted_ma",
            MaData::Candles {
                candles: &candles,
                source: "close",
            },
            (20, 20, 0),
            Kernel::Auto,
            Some(&params),
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("expected 0 or 1"));
    }

    #[test]
    fn vwap_invalid_anchor_step_error() {
        let candles = sample_candles(300);
        let params = [
            MaBatchParamKV {
                key: "anchor",
                value: MaBatchParamValue::EnumString("1d"),
            },
            MaBatchParamKV {
                key: "anchor_step",
                value: MaBatchParamValue::Float(-1.0),
            },
        ];
        let err = ma_batch_with_kernel_and_typed_params(
            "vwap",
            MaData::Candles {
                candles: &candles,
                source: "close",
            },
            (10, 10, 0),
            Kernel::Auto,
            &params,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("expected >= 0"));
    }

    #[test]
    fn ehlers_pma_invalid_output_selection_returns_error() {
        let prices = sample_prices(256);
        let params = [MaBatchParamKV {
            key: "output",
            value: MaBatchParamValue::EnumString("bad_line"),
        }];
        let err = ma_batch_with_kernel_and_typed_params(
            "ehlers_pma",
            MaData::Slice(&prices),
            (10, 10, 0),
            Kernel::Auto,
            &params,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("expected 'predict' or 'trigger'"));
    }

    #[test]
    fn buff_averages_typed_output_selection_matches_direct() {
        let candles = sample_candles(300);
        let params = [
            MaBatchParamKV {
                key: "fast_period",
                value: MaBatchParamValue::Int(5),
            },
            MaBatchParamKV {
                key: "output",
                value: MaBatchParamValue::EnumString("slow"),
            },
        ];
        let got = ma_batch_with_kernel_and_typed_params(
            "buff_averages",
            MaData::Candles {
                candles: &candles,
                source: "close",
            },
            (20, 20, 0),
            Kernel::Auto,
            &params,
        )
        .unwrap();
        let direct =
            crate::indicators::moving_averages::buff_averages::buff_averages_batch_with_kernel(
                &candles.close,
                &candles.volume,
                &crate::indicators::moving_averages::buff_averages::BuffAveragesBatchRange {
                    fast_period: (5, 5, 0),
                    slow_period: (20, 20, 0),
                },
                Kernel::Auto,
            )
            .unwrap();
        assert_eq!(got.rows, direct.rows);
        assert_eq!(got.cols, direct.cols);
        assert_series_eq(&got.values, &direct.slow, 1e-12);
    }

    #[test]
    fn buff_averages_requires_candles_error() {
        let prices = sample_prices(256);
        let err = ma_batch_with_kernel_and_typed_params(
            "buff_averages",
            MaData::Slice(&prices),
            (20, 20, 0),
            Kernel::Auto,
            &[],
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("requires candles"));
    }

    #[test]
    fn buff_averages_invalid_output_selection_returns_error() {
        let candles = sample_candles(300);
        let params = [MaBatchParamKV {
            key: "output",
            value: MaBatchParamValue::EnumString("bad_line"),
        }];
        let err = ma_batch_with_kernel_and_typed_params(
            "buff_averages",
            MaData::Candles {
                candles: &candles,
                source: "close",
            },
            (20, 20, 0),
            Kernel::Auto,
            &params,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("expected 'fast' or 'slow'"));
    }

    #[test]
    fn buff_averages_numeric_params_match_direct_fast() {
        let candles = sample_candles(300);
        let mut params = HashMap::new();
        params.insert("fast_period_start".to_string(), 5.0);
        params.insert("fast_period_end".to_string(), 5.0);
        params.insert("fast_period_step".to_string(), 0.0);
        params.insert("slow_period_start".to_string(), 20.0);
        params.insert("slow_period_end".to_string(), 22.0);
        params.insert("slow_period_step".to_string(), 1.0);

        let got = ma_batch_with_kernel_and_params(
            "buff_averages",
            MaData::Candles {
                candles: &candles,
                source: "close",
            },
            (20, 22, 1),
            Kernel::Auto,
            Some(&params),
        )
        .unwrap();

        let direct =
            crate::indicators::moving_averages::buff_averages::buff_averages_batch_with_kernel(
                &candles.close,
                &candles.volume,
                &crate::indicators::moving_averages::buff_averages::BuffAveragesBatchRange {
                    fast_period: (5, 5, 0),
                    slow_period: (20, 22, 1),
                },
                Kernel::Auto,
            )
            .unwrap();

        assert_eq!(got.periods, vec![20, 21, 22]);
        assert_eq!(got.rows, direct.rows);
        assert_eq!(got.cols, direct.cols);
        assert_series_eq(&got.values, &direct.fast, 1e-12);
    }

    #[test]
    fn typed_non_finite_float_rejected() {
        let prices = sample_prices(256);
        let params = [MaBatchParamKV {
            key: "offset",
            value: MaBatchParamValue::Float(f64::NAN),
        }];
        let err = ma_batch_with_kernel_and_typed_params(
            "alma",
            MaData::Slice(&prices),
            (20, 20, 0),
            Kernel::Auto,
            &params,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("expected finite number"));
    }

    #[test]
    fn uma_typed_integer_param_rejects_fractional_value() {
        let prices = sample_prices(256);
        let params = [MaBatchParamKV {
            key: "min_length",
            value: MaBatchParamValue::Float(7.25),
        }];
        let err = ma_batch_with_kernel_and_typed_params(
            "uma",
            MaData::Slice(&prices),
            (35, 35, 0),
            Kernel::Auto,
            &params,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("expected integer"));
    }

    #[test]
    fn buff_averages_typed_integer_param_rejects_fractional_value() {
        let candles = sample_candles(300);
        let params = [MaBatchParamKV {
            key: "fast_period",
            value: MaBatchParamValue::Float(5.5),
        }];
        let err = ma_batch_with_kernel_and_typed_params(
            "buff_averages",
            MaData::Candles {
                candles: &candles,
                source: "close",
            },
            (20, 20, 0),
            Kernel::Auto,
            &params,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("expected integer"));
    }

    #[test]
    fn vwap_typed_anchor_step_rejects_fractional_value() {
        let candles = sample_candles(300);
        let params = [
            MaBatchParamKV {
                key: "anchor",
                value: MaBatchParamValue::EnumString("1d"),
            },
            MaBatchParamKV {
                key: "anchor_step",
                value: MaBatchParamValue::Float(1.5),
            },
        ];
        let err = ma_batch_with_kernel_and_typed_params(
            "vwap",
            MaData::Candles {
                candles: &candles,
                source: "close",
            },
            (10, 10, 0),
            Kernel::Auto,
            &params,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("expected integer"));
    }

    #[test]
    fn mwdx_typed_non_finite_factor_rejected() {
        let prices = sample_prices(256);
        let params = [MaBatchParamKV {
            key: "factor",
            value: MaBatchParamValue::Float(f64::INFINITY),
        }];
        let err = ma_batch_with_kernel_and_typed_params(
            "mwdx",
            MaData::Slice(&prices),
            (10, 10, 0),
            Kernel::Auto,
            &params,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("expected finite number"));
    }

    #[test]
    fn hwma_typed_non_finite_param_rejected() {
        let prices = sample_prices(256);
        let params = [MaBatchParamKV {
            key: "na",
            value: MaBatchParamValue::Float(f64::NAN),
        }];
        let err = ma_batch_with_kernel_and_typed_params(
            "hwma",
            MaData::Slice(&prices),
            (10, 10, 0),
            Kernel::Auto,
            &params,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("expected finite number"));
    }

    #[test]
    fn dma_typed_fractional_ema_length_rejected() {
        let prices = sample_prices(256);
        let params = [
            MaBatchParamKV {
                key: "ema_length",
                value: MaBatchParamValue::Float(20.5),
            },
            MaBatchParamKV {
                key: "hull_ma_type",
                value: MaBatchParamValue::EnumString("EMA"),
            },
        ];
        let err = ma_batch_with_kernel_and_typed_params(
            "dma",
            MaData::Slice(&prices),
            (14, 14, 0),
            Kernel::Auto,
            &params,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("expected integer"));
    }

    #[test]
    fn tradjema_typed_non_finite_mult_rejected() {
        let candles = sample_candles(300);
        let params = [MaBatchParamKV {
            key: "mult",
            value: MaBatchParamValue::Float(f64::NAN),
        }];
        let err = ma_batch_with_kernel_and_typed_params(
            "tradjema",
            MaData::Candles {
                candles: &candles,
                source: "close",
            },
            (40, 40, 0),
            Kernel::Auto,
            &params,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("expected finite number"));
    }

    #[test]
    fn uma_typed_negative_min_length_rejected() {
        let prices = sample_prices(256);
        let params = [MaBatchParamKV {
            key: "min_length",
            value: MaBatchParamValue::Int(-1),
        }];
        let err = ma_batch_with_kernel_and_typed_params(
            "uma",
            MaData::Slice(&prices),
            (35, 35, 0),
            Kernel::Auto,
            &params,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("expected >= 0"));
    }

    #[test]
    fn volume_adjusted_ma_typed_fractional_sample_period_rejected() {
        let candles = sample_candles(300);
        let params = [MaBatchParamKV {
            key: "sample_period",
            value: MaBatchParamValue::Float(30.5),
        }];
        let err = ma_batch_with_kernel_and_typed_params(
            "volume_adjusted_ma",
            MaData::Candles {
                candles: &candles,
                source: "close",
            },
            (20, 20, 0),
            Kernel::Auto,
            &params,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("expected integer"));
    }
}
