use super::{
    compute_cpu_batch, IndicatorBatchOutput, IndicatorBatchRequest, IndicatorComputeOutput,
    IndicatorComputeRequest, IndicatorDataRef, IndicatorDispatchError, IndicatorParamSet,
    IndicatorSeries,
};
use crate::indicators::pattern_recognition::{
    pattern_recognition_with_kernel, PatternRecognitionData, PatternRecognitionError,
    PatternRecognitionInput,
};
use crate::indicators::registry::{get_indicator, IndicatorInfo, IndicatorInputKind};

pub fn compute_cpu(
    req: IndicatorComputeRequest<'_>,
) -> Result<IndicatorComputeOutput, IndicatorDispatchError> {
    let info = get_indicator(req.indicator_id);

    if let Some(info) = info {
        if info.id.eq_ignore_ascii_case("pattern_recognition") {
            return compute_pattern_recognition(req, info);
        }
    }

    let indicator_id = info.map_or(req.indicator_id, |info| info.id);

    let combos = [IndicatorParamSet { params: req.params }];
    let out = match compute_cpu_batch(IndicatorBatchRequest {
        indicator_id,
        output_id: req.output_id,
        data: req.data,
        combos: &combos,
        kernel: req.kernel,
    }) {
        Err(IndicatorDispatchError::UnsupportedCapability { .. }) if info.is_none() => {
            return Err(IndicatorDispatchError::UnknownIndicator {
                id: req.indicator_id.to_string(),
            });
        }
        other => other?,
    };
    map_batch_output_to_compute(indicator_id, out)
}

fn compute_pattern_recognition(
    req: IndicatorComputeRequest<'_>,
    info: &IndicatorInfo,
) -> Result<IndicatorComputeOutput, IndicatorDispatchError> {
    if !info.capabilities.supports_cpu_single {
        return Err(IndicatorDispatchError::UnsupportedCapability {
            indicator: info.id.to_string(),
            capability: "cpu_single",
        });
    }

    if let Some(param) = req.params.first() {
        return Err(IndicatorDispatchError::InvalidParam {
            indicator: info.id.to_string(),
            key: param.key.to_string(),
            reason: "pattern_recognition does not accept parameters".to_string(),
        });
    }

    let output_id = resolve_output_id(info, req.output_id)?;
    let input = match req.data {
        IndicatorDataRef::Candles { candles, .. } => PatternRecognitionInput::from_candles(
            candles,
            crate::indicators::pattern_recognition::PatternRecognitionParams::default(),
        ),
        IndicatorDataRef::Ohlc {
            open,
            high,
            low,
            close,
        } => PatternRecognitionInput::from_slices(
            open,
            high,
            low,
            close,
            crate::indicators::pattern_recognition::PatternRecognitionParams::default(),
        ),
        IndicatorDataRef::Ohlcv {
            open,
            high,
            low,
            close,
            ..
        } => PatternRecognitionInput::from_slices(
            open,
            high,
            low,
            close,
            crate::indicators::pattern_recognition::PatternRecognitionParams::default(),
        ),
        _ => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: info.id.to_string(),
                input: IndicatorInputKind::Ohlc,
            });
        }
    };

    let out = pattern_recognition_with_kernel(&input, req.kernel)
        .map_err(|e| map_pattern_error(info.id, e))?;

    Ok(IndicatorComputeOutput {
        output_id: output_id.to_string(),
        series: IndicatorSeries::Bool(out.values_u8.into_iter().map(|v| v != 0).collect()),
        warmup: out.warmup,
        rows: out.rows,
        cols: out.cols,
        pattern_ids: Some(
            out.pattern_ids
                .into_iter()
                .map(|id| id.to_string())
                .collect(),
        ),
    })
}

fn normalize_output_token(value: &str) -> String {
    let mut normalized = String::with_capacity(value.len());
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() {
            normalized.push(ch.to_ascii_lowercase());
        }
    }
    if normalized == "values" {
        "value".to_string()
    } else {
        normalized
    }
}

fn output_id_matches(candidate: &str, requested: &str) -> bool {
    candidate.eq_ignore_ascii_case(requested)
        || normalize_output_token(candidate) == normalize_output_token(requested)
}

fn resolve_output_id<'a>(
    info: &'a IndicatorInfo,
    requested: Option<&str>,
) -> Result<&'a str, IndicatorDispatchError> {
    if info.outputs.is_empty() {
        return Err(IndicatorDispatchError::ComputeFailed {
            indicator: info.id.to_string(),
            details: "indicator has no registered outputs".to_string(),
        });
    }

    if info.outputs.len() == 1 {
        let only = info.outputs[0].id;
        if let Some(req) = requested {
            if !output_id_matches(only, req) {
                return Err(IndicatorDispatchError::UnknownOutput {
                    indicator: info.id.to_string(),
                    output: req.to_string(),
                });
            }
        }
        return Ok(only);
    }

    let req = requested.ok_or_else(|| IndicatorDispatchError::InvalidParam {
        indicator: info.id.to_string(),
        key: "output_id".to_string(),
        reason: "output_id is required for multi-output indicators".to_string(),
    })?;

    info.outputs
        .iter()
        .find(|o| output_id_matches(o.id, req))
        .map(|o| o.id)
        .ok_or_else(|| IndicatorDispatchError::UnknownOutput {
            indicator: info.id.to_string(),
            output: req.to_string(),
        })
}

fn map_batch_output_to_compute(
    indicator: &str,
    out: IndicatorBatchOutput,
) -> Result<IndicatorComputeOutput, IndicatorDispatchError> {
    let series = if let Some(values) = out.values_f64 {
        IndicatorSeries::F64(values)
    } else if let Some(values) = out.values_i32 {
        IndicatorSeries::I32(values)
    } else if let Some(values) = out.values_bool {
        IndicatorSeries::Bool(values)
    } else {
        return Err(IndicatorDispatchError::ComputeFailed {
            indicator: indicator.to_string(),
            details: "dispatcher returned no output series".to_string(),
        });
    };

    Ok(IndicatorComputeOutput {
        output_id: out.output_id,
        series,
        warmup: None,
        rows: out.rows,
        cols: out.cols,
        pattern_ids: None,
    })
}

fn map_pattern_error(indicator: &str, err: PatternRecognitionError) -> IndicatorDispatchError {
    match err {
        PatternRecognitionError::DataLengthMismatch {
            open,
            high,
            low,
            close,
        } => IndicatorDispatchError::DataLengthMismatch {
            details: format!("open={} high={} low={} close={}", open, high, low, close),
        },
        PatternRecognitionError::OutputLengthMismatch {
            pattern_id,
            expected,
            got,
        } => IndicatorDispatchError::ComputeFailed {
            indicator: indicator.to_string(),
            details: format!(
                "pattern output mismatch for {}: expected {}, got {}",
                pattern_id, expected, got
            ),
        },
        PatternRecognitionError::Pattern(e) => IndicatorDispatchError::ComputeFailed {
            indicator: indicator.to_string(),
            details: e.to_string(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::dispatch::{compute_cpu_batch, ParamKV, ParamValue};
    use crate::indicators::pattern_recognition::list_patterns;
    use crate::utilities::enums::Kernel;

    fn sample_series(len: usize) -> Vec<f64> {
        (0..len)
            .map(|i| 100.0 + ((i as f64) * 0.01).sin() + ((i as f64) * 0.0005).cos())
            .collect()
    }

    fn sample_ohlc(len: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let open = sample_series(len);
        let high: Vec<f64> = open.iter().map(|v| v + 1.0).collect();
        let low: Vec<f64> = open.iter().map(|v| v - 1.0).collect();
        let close: Vec<f64> = open.iter().map(|v| v + 0.25).collect();
        (open, high, low, close)
    }

    #[test]
    fn compute_cpu_pattern_recognition_returns_matrix() {
        let (open, high, low, close) = sample_ohlc(192);
        let req = IndicatorComputeRequest {
            indicator_id: "pattern_recognition",
            output_id: Some("matrix"),
            data: IndicatorDataRef::Ohlc {
                open: &open,
                high: &high,
                low: &low,
                close: &close,
            },
            params: &[],
            kernel: Kernel::Auto,
        };
        let out = compute_cpu(req).unwrap();
        assert_eq!(out.output_id, "matrix");
        assert_eq!(out.rows, list_patterns().len());
        assert_eq!(out.cols, close.len());
        match out.series {
            IndicatorSeries::Bool(v) => assert_eq!(v.len(), out.rows * out.cols),
            other => panic!("expected Bool matrix series, got {:?}", other),
        }
        let ids = out.pattern_ids.unwrap();
        assert_eq!(ids.len(), out.rows);
    }

    #[test]
    fn compute_cpu_pattern_recognition_rejects_missing_input_shape() {
        let series = sample_series(64);
        let req = IndicatorComputeRequest {
            indicator_id: "pattern_recognition",
            output_id: Some("matrix"),
            data: IndicatorDataRef::Slice { values: &series },
            params: &[],
            kernel: Kernel::Auto,
        };
        let err = compute_cpu(req).unwrap_err();
        match err {
            IndicatorDispatchError::MissingRequiredInput { indicator, input } => {
                assert_eq!(indicator, "pattern_recognition");
                assert_eq!(input, IndicatorInputKind::Ohlc);
            }
            other => panic!("expected MissingRequiredInput, got {:?}", other),
        }
    }

    #[test]
    fn compute_cpu_pattern_recognition_rejects_unknown_output() {
        let (open, high, low, close) = sample_ohlc(64);
        let req = IndicatorComputeRequest {
            indicator_id: "pattern_recognition",
            output_id: Some("value"),
            data: IndicatorDataRef::Ohlc {
                open: &open,
                high: &high,
                low: &low,
                close: &close,
            },
            params: &[],
            kernel: Kernel::Auto,
        };
        let err = compute_cpu(req).unwrap_err();
        match err {
            IndicatorDispatchError::UnknownOutput { indicator, output } => {
                assert_eq!(indicator, "pattern_recognition");
                assert_eq!(output, "value");
            }
            other => panic!("expected UnknownOutput, got {:?}", other),
        }
    }

    #[test]
    fn compute_cpu_pattern_recognition_rejects_params() {
        let (open, high, low, close) = sample_ohlc(64);
        let params = [ParamKV {
            key: "period",
            value: ParamValue::Int(14),
        }];
        let req = IndicatorComputeRequest {
            indicator_id: "pattern_recognition",
            output_id: Some("matrix"),
            data: IndicatorDataRef::Ohlc {
                open: &open,
                high: &high,
                low: &low,
                close: &close,
            },
            params: &params,
            kernel: Kernel::Auto,
        };
        let err = compute_cpu(req).unwrap_err();
        match err {
            IndicatorDispatchError::InvalidParam {
                indicator,
                key,
                reason,
            } => {
                assert_eq!(indicator, "pattern_recognition");
                assert_eq!(key, "period");
                assert!(reason.contains("does not accept parameters"));
            }
            other => panic!("expected InvalidParam, got {:?}", other),
        }
    }

    #[test]
    fn pattern_recognition_batch_mode_is_explicitly_unsupported() {
        let (open, high, low, close) = sample_ohlc(96);
        let combos = [IndicatorParamSet { params: &[] }];
        let err = compute_cpu_batch(IndicatorBatchRequest {
            indicator_id: "pattern_recognition",
            output_id: Some("matrix"),
            data: IndicatorDataRef::Ohlc {
                open: &open,
                high: &high,
                low: &low,
                close: &close,
            },
            combos: &combos,
            kernel: Kernel::Auto,
        })
        .unwrap_err();
        match err {
            IndicatorDispatchError::UnsupportedCapability {
                indicator,
                capability,
            } => {
                assert_eq!(indicator, "pattern_recognition");
                assert_eq!(capability, "cpu_batch");
            }
            other => panic!("expected UnsupportedCapability, got {:?}", other),
        }
    }

    #[test]
    fn compute_cpu_for_sma_delegates_to_batch_dispatch() {
        let series = sample_series(200);
        let params = [ParamKV {
            key: "period",
            value: ParamValue::Int(14),
        }];
        let req = IndicatorComputeRequest {
            indicator_id: "sma",
            output_id: Some("value"),
            data: IndicatorDataRef::Slice { values: &series },
            params: &params,
            kernel: Kernel::Auto,
        };
        let out = compute_cpu(req).unwrap();
        assert_eq!(out.output_id, "value");
        assert_eq!(out.rows, 1);
        assert_eq!(out.cols, series.len());
        match out.series {
            IndicatorSeries::F64(v) => assert_eq!(v.len(), series.len()),
            other => panic!("expected F64 series, got {:?}", other),
        }
    }
}
