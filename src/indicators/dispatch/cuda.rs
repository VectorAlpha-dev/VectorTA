use super::{
    CudaOutputTarget, DeviceMatrixF32, IndicatorCudaDataRef, IndicatorCudaOutput,
    IndicatorCudaRequest, IndicatorCudaSeries, IndicatorDispatchError, ParamKV, ParamValue,
};
use crate::cuda::moving_averages::ma_selector::{CudaMaParamKV, CudaMaParamValue};
use crate::cuda::moving_averages::{CudaMaData, CudaMaSelector, CudaMaSelectorError};
use crate::cuda::pattern_recognition_wrapper::CudaPatternRecognition;
use crate::indicators::moving_averages::registry::list_moving_averages;
use crate::indicators::registry::{get_indicator, IndicatorInfo, IndicatorInputKind};
use cust::memory::CopyDestination;

pub fn compute_cuda(
    req: IndicatorCudaRequest<'_>,
) -> Result<IndicatorCudaOutput, IndicatorDispatchError> {
    let normalized_id = normalize_cuda_dispatch_id(req.indicator_id);
    let normalized_req = IndicatorCudaRequest {
        indicator_id: normalized_id.as_str(),
        ..req
    };
    let info = get_indicator(normalized_id.as_str());

    if normalized_id.eq_ignore_ascii_case("pattern_recognition") {
        let info = info.ok_or_else(|| IndicatorDispatchError::UnknownIndicator {
            id: req.indicator_id.to_string(),
        })?;
        return compute_pattern_recognition_cuda(normalized_req, info);
    }

    let device_id = resolve_device_id(normalized_id.as_str(), normalized_req.params)? as usize;
    if let Some(out) = super::cuda_non_ma_generated::try_dispatch_non_ma_cuda(
        normalized_id.as_str(),
        info,
        normalized_req,
        device_id,
    ) {
        return out;
    }

    let info = info.ok_or_else(|| IndicatorDispatchError::UnknownIndicator {
        id: req.indicator_id.to_string(),
    })?;

    if !info.capabilities.supports_cuda_batch {
        return Err(IndicatorDispatchError::UnsupportedCapability {
            indicator: info.id.to_string(),
            capability: "cuda_batch",
        });
    }

    if !is_moving_average(info.id) {
        return Err(IndicatorDispatchError::UnsupportedCapability {
            indicator: info.id.to_string(),
            capability: "cuda_batch",
        });
    }

    let output_id = resolve_output_id(info, normalized_req.output_id)?;
    let data = cuda_data_from_req(info, normalized_req.data)?;
    let period_based = is_period_based_ma(info.id);
    let (start, end, step) = resolve_period_range(info.id, period_based, normalized_req.params)?;
    let mut typed_params = to_cuda_typed_params(info.id, normalized_req.params)?;

    if info.id.eq_ignore_ascii_case("buff_averages") {
        if !has_key(normalized_req.params, "fast_period") {
            typed_params.push(CudaMaParamKV {
                key: "fast_period",
                value: CudaMaParamValue::Int(5),
            });
        }
        if !has_key(normalized_req.params, "slow_period") {
            typed_params.push(CudaMaParamKV {
                key: "slow_period",
                value: CudaMaParamValue::Int(20),
            });
        }
    }

    if info.outputs.len() > 1 && !has_key(normalized_req.params, "output") {
        typed_params.push(CudaMaParamKV {
            key: "output",
            value: CudaMaParamValue::EnumString(output_id),
        });
    }

    let selector = CudaMaSelector::new(device_id);
    let dev = selector
        .ma_sweep_to_device_with_typed_params(info.id, data, start, end, step, &typed_params)
        .map_err(|e| map_cuda_error(info.id, e))?;
    let warmup = if period_based {
        Some(start.min(end).saturating_sub(1))
    } else {
        None
    };
    let out_rows = dev.rows;
    let out_cols = dev.cols;

    match normalized_req.target {
        CudaOutputTarget::DeviceF32 => Ok(IndicatorCudaOutput {
            output_id: output_id.to_string(),
            series: IndicatorCudaSeries::DeviceF32(DeviceMatrixF32::from_owned(
                dev,
                device_id as u32,
            )),
            warmup,
            rows: out_rows,
            cols: out_cols,
            pattern_ids: None,
        }),
        CudaOutputTarget::HostF32 => {
            let mut host = vec![0.0f32; out_rows.saturating_mul(out_cols)];
            dev.buf.copy_to(host.as_mut_slice()).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            Ok(IndicatorCudaOutput {
                output_id: output_id.to_string(),
                series: IndicatorCudaSeries::HostF32(host),
                warmup,
                rows: out_rows,
                cols: out_cols,
                pattern_ids: None,
            })
        }
    }
}

fn compute_pattern_recognition_cuda(
    req: IndicatorCudaRequest<'_>,
    info: &IndicatorInfo,
) -> Result<IndicatorCudaOutput, IndicatorDispatchError> {
    if !info.capabilities.supports_cuda_single {
        return Err(IndicatorDispatchError::UnsupportedCapability {
            indicator: info.id.to_string(),
            capability: "cuda_single",
        });
    }

    let output_id = resolve_output_id(info, req.output_id)?;
    validate_pattern_params(info.id, req.params)?;
    let device_id = resolve_device_id(info.id, req.params)? as usize;
    let (open, high, low, close) = pattern_ohlc_from_req(info.id, req.data)?;
    if close.is_empty() {
        return Err(IndicatorDispatchError::DataLengthMismatch {
            details: "pattern_recognition: empty OHLC input".to_string(),
        });
    }

    let cuda = CudaPatternRecognition::new(device_id).map_err(|e| {
        IndicatorDispatchError::KernelUnavailable {
            details: e.to_string(),
        }
    })?;
    let features = cuda
        .compute_features_device(open, high, low, close)
        .map_err(|e| IndicatorDispatchError::KernelUnavailable {
            details: e.to_string(),
        })?;
    let native_ids = CudaPatternRecognition::native_supported_pattern_ids();
    let rows = native_ids.len();
    let cols = close.len();
    let row_map: Vec<(&str, usize)> = native_ids
        .iter()
        .enumerate()
        .map(|(row, id)| (*id, row))
        .collect();
    let d_u8 = cuda
        .compute_native_matrix_device(&features, rows, cols, row_map.as_slice())
        .map_err(|e| IndicatorDispatchError::KernelUnavailable {
            details: e.to_string(),
        })?;
    let pattern_ids: Vec<String> = native_ids.iter().map(|id| id.to_string()).collect();

    match req.target {
        CudaOutputTarget::HostF32 => {
            cuda.synchronize()
                .map_err(|e| IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                })?;
            let mut host_u8 = vec![0u8; rows.saturating_mul(cols)];
            d_u8.copy_to(host_u8.as_mut_slice()).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let host = host_u8
                .into_iter()
                .map(|v| if v == 0 { 0.0 } else { 1.0 })
                .collect();
            Ok(IndicatorCudaOutput {
                output_id: output_id.to_string(),
                series: IndicatorCudaSeries::HostF32(host),
                warmup: None,
                rows,
                cols,
                pattern_ids: Some(pattern_ids),
            })
        }
        CudaOutputTarget::DeviceF32 => {
            let dev = cuda
                .matrix_u8_to_f32_device(&d_u8, rows, cols)
                .map_err(|e| IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                })?;
            Ok(IndicatorCudaOutput {
                output_id: output_id.to_string(),
                series: IndicatorCudaSeries::DeviceF32(DeviceMatrixF32::from_owned(
                    dev,
                    device_id as u32,
                )),
                warmup: None,
                rows,
                cols,
                pattern_ids: Some(pattern_ids),
            })
        }
    }
}

fn pattern_ohlc_from_req<'a>(
    indicator: &str,
    data: IndicatorCudaDataRef<'a>,
) -> Result<(&'a [f32], &'a [f32], &'a [f32], &'a [f32]), IndicatorDispatchError> {
    match data {
        IndicatorCudaDataRef::Ohlc {
            open,
            high,
            low,
            close,
            ..
        } => Ok((open, high, low, close)),
        IndicatorCudaDataRef::Ohlcv {
            open,
            high,
            low,
            close,
            ..
        } => Ok((open, high, low, close)),
        _ => Err(IndicatorDispatchError::MissingRequiredInput {
            indicator: indicator.to_string(),
            input: IndicatorInputKind::Ohlc,
        }),
    }
}

fn validate_pattern_params(
    indicator: &str,
    params: &[ParamKV<'_>],
) -> Result<(), IndicatorDispatchError> {
    for param in params {
        if param.key.eq_ignore_ascii_case("device_id") {
            continue;
        }
        return Err(IndicatorDispatchError::InvalidParam {
            indicator: indicator.to_string(),
            key: param.key.to_string(),
            reason: "pattern_recognition does not accept this parameter".to_string(),
        });
    }
    Ok(())
}

fn normalize_cuda_dispatch_id(id: &str) -> String {
    let lower = id.to_ascii_lowercase();
    match lower.as_str() {
        "damiani" => "damiani_volatmeter".to_string(),
        "vama" => "volatility_adjusted_ma".to_string(),
        "bbw" => "bollinger_bands_width".to_string(),
        "fvg_ts" => "fvg_trailing_stop".to_string(),
        "nwe" => "nadaraya_watson_envelope".to_string(),
        "pnr" => "percentile_nearest_rank".to_string(),
        _ => lower,
    }
}

fn is_moving_average(id: &str) -> bool {
    list_moving_averages()
        .iter()
        .any(|ma| ma.id.eq_ignore_ascii_case(id))
}

fn is_period_based_ma(id: &str) -> bool {
    list_moving_averages()
        .iter()
        .find(|ma| ma.id.eq_ignore_ascii_case(id))
        .map(|ma| ma.period_based)
        .unwrap_or(true)
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
            if !req.eq_ignore_ascii_case(only) {
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
        .find(|o| o.id.eq_ignore_ascii_case(req))
        .map(|o| o.id)
        .ok_or_else(|| IndicatorDispatchError::UnknownOutput {
            indicator: info.id.to_string(),
            output: req.to_string(),
        })
}

fn cuda_data_from_req<'a>(
    info: &IndicatorInfo,
    data: IndicatorCudaDataRef<'a>,
) -> Result<CudaMaData<'a>, IndicatorDispatchError> {
    match (info.input_kind, data) {
        (
            IndicatorInputKind::Candles,
            IndicatorCudaDataRef::Ohlcv {
                timestamp,
                open,
                high,
                low,
                close,
                volume,
                source,
            },
        ) => Ok(CudaMaData::OhlcvF32 {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
            source,
        }),
        (
            IndicatorInputKind::Candles,
            IndicatorCudaDataRef::Ohlc {
                open,
                high,
                low,
                close,
                source,
            },
        ) => Ok(CudaMaData::OhlcF32 {
            open,
            high,
            low,
            close,
            source,
        }),
        (IndicatorInputKind::Candles, _) => Err(IndicatorDispatchError::MissingRequiredInput {
            indicator: info.id.to_string(),
            input: IndicatorInputKind::Candles,
        }),
        (_, IndicatorCudaDataRef::Slice { values }) => Ok(CudaMaData::SliceF32(values)),
        (
            _,
            IndicatorCudaDataRef::Ohlc {
                open,
                high,
                low,
                close,
                source,
            },
        ) => Ok(CudaMaData::OhlcF32 {
            open,
            high,
            low,
            close,
            source,
        }),
        (
            _,
            IndicatorCudaDataRef::Ohlcv {
                timestamp,
                open,
                high,
                low,
                close,
                volume,
                source,
            },
        ) => Ok(CudaMaData::OhlcvF32 {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
            source,
        }),
        (_, IndicatorCudaDataRef::CloseVolume { close, volume }) => Ok(CudaMaData::OhlcvF32 {
            timestamp: None,
            open: close,
            high: close,
            low: close,
            close,
            volume,
            source: None,
        }),
        _ => Err(IndicatorDispatchError::MissingRequiredInput {
            indicator: info.id.to_string(),
            input: IndicatorInputKind::Slice,
        }),
    }
}

fn resolve_device_id(
    indicator: &str,
    params: &[ParamKV<'_>],
) -> Result<u32, IndicatorDispatchError> {
    match get_usize_param(params, "device_id", indicator)? {
        Some(v) => u32::try_from(v).map_err(|_| IndicatorDispatchError::InvalidParam {
            indicator: indicator.to_string(),
            key: "device_id".to_string(),
            reason: format!("value {} exceeds u32::MAX", v),
        }),
        None => Ok(0),
    }
}

fn resolve_period_range(
    indicator: &str,
    period_based: bool,
    params: &[ParamKV<'_>],
) -> Result<(usize, usize, usize), IndicatorDispatchError> {
    let period = get_usize_param(params, "period", indicator)?;
    let start = get_usize_param(params, "period_start", indicator)?
        .or(get_usize_param(params, "start", indicator)?);
    let end = get_usize_param(params, "period_end", indicator)?
        .or(get_usize_param(params, "end", indicator)?);
    let step = get_usize_param(params, "period_step", indicator)?
        .or(get_usize_param(params, "step", indicator)?);

    if period_based {
        let s = start.or(period).unwrap_or(14);
        let e = end.or(period).unwrap_or(s);
        let st = step.unwrap_or(if s == e { 0 } else { 1 });
        if s == 0 || e == 0 {
            return Err(IndicatorDispatchError::InvalidParam {
                indicator: indicator.to_string(),
                key: "period".to_string(),
                reason: "period values must be >= 1".to_string(),
            });
        }
        return Ok((s, e, st));
    }

    let s = start.or(period).unwrap_or(1);
    let e = end.or(period).unwrap_or(s);
    let st = step.unwrap_or(if s == e { 0 } else { 1 });
    Ok((s, e, st))
}

fn to_cuda_typed_params<'a>(
    indicator: &str,
    params: &'a [ParamKV<'a>],
) -> Result<Vec<CudaMaParamKV<'a>>, IndicatorDispatchError> {
    let mut out = Vec::with_capacity(params.len());
    for p in params {
        if is_internal_key(p.key) {
            continue;
        }
        let value = match p.value {
            ParamValue::Int(v) => CudaMaParamValue::Int(v),
            ParamValue::Float(v) => {
                if !v.is_finite() {
                    return Err(IndicatorDispatchError::InvalidParam {
                        indicator: indicator.to_string(),
                        key: p.key.to_string(),
                        reason: "expected finite float".to_string(),
                    });
                }
                CudaMaParamValue::Float(v)
            }
            ParamValue::Bool(v) => CudaMaParamValue::Bool(v),
            ParamValue::EnumString(v) => CudaMaParamValue::EnumString(v),
        };
        out.push(CudaMaParamKV { key: p.key, value });
    }
    Ok(out)
}

fn is_internal_key(key: &str) -> bool {
    key.eq_ignore_ascii_case("period")
        || key.eq_ignore_ascii_case("period_start")
        || key.eq_ignore_ascii_case("period_end")
        || key.eq_ignore_ascii_case("period_step")
        || key.eq_ignore_ascii_case("start")
        || key.eq_ignore_ascii_case("end")
        || key.eq_ignore_ascii_case("step")
        || key.eq_ignore_ascii_case("device_id")
}

fn has_key(params: &[ParamKV<'_>], key: &str) -> bool {
    params.iter().any(|kv| kv.key.eq_ignore_ascii_case(key))
}

fn get_usize_param(
    params: &[ParamKV<'_>],
    key: &str,
    indicator: &str,
) -> Result<Option<usize>, IndicatorDispatchError> {
    let item = params.iter().find(|kv| kv.key.eq_ignore_ascii_case(key));
    let Some(item) = item else {
        return Ok(None);
    };
    match item.value {
        ParamValue::Int(v) => {
            if v < 0 {
                return Err(IndicatorDispatchError::InvalidParam {
                    indicator: indicator.to_string(),
                    key: key.to_string(),
                    reason: format!("expected non-negative integer, got {}", v),
                });
            }
            Ok(Some(v as usize))
        }
        ParamValue::Float(v) => {
            if !v.is_finite() || v < 0.0 {
                return Err(IndicatorDispatchError::InvalidParam {
                    indicator: indicator.to_string(),
                    key: key.to_string(),
                    reason: format!("expected non-negative finite number, got {}", v),
                });
            }
            let rounded = v.round();
            if (v - rounded).abs() > 1e-9 {
                return Err(IndicatorDispatchError::InvalidParam {
                    indicator: indicator.to_string(),
                    key: key.to_string(),
                    reason: format!("expected integer-compatible value, got {}", v),
                });
            }
            Ok(Some(rounded as usize))
        }
        ParamValue::Bool(v) => Ok(Some(if v { 1 } else { 0 })),
        ParamValue::EnumString(v) => Err(IndicatorDispatchError::InvalidParam {
            indicator: indicator.to_string(),
            key: key.to_string(),
            reason: format!("expected integer value, got enum string '{}'", v),
        }),
    }
}

fn map_cuda_error(indicator: &str, err: CudaMaSelectorError) -> IndicatorDispatchError {
    match err {
        CudaMaSelectorError::Unsupported(_) => IndicatorDispatchError::UnsupportedCapability {
            indicator: indicator.to_string(),
            capability: "cuda_batch",
        },
        CudaMaSelectorError::InvalidInput(reason) => IndicatorDispatchError::InvalidParam {
            indicator: indicator.to_string(),
            key: "params".to_string(),
            reason,
        },
        CudaMaSelectorError::InvalidRange { start, end, step } => {
            IndicatorDispatchError::InvalidParam {
                indicator: indicator.to_string(),
                key: "period_range".to_string(),
                reason: format!("start={} end={} step={}", start, end, step),
            }
        }
        CudaMaSelectorError::Cuda(e) => IndicatorDispatchError::KernelUnavailable {
            details: e.to_string(),
        },
        other => IndicatorDispatchError::ComputeFailed {
            indicator: indicator.to_string(),
            details: other.to_string(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::dispatch::{
        compute_cpu, compute_cpu_batch, IndicatorBatchRequest, IndicatorComputeRequest,
        IndicatorDataRef, IndicatorParamSet, IndicatorSeries,
    };
    use crate::indicators::registry::{IndicatorParamKind, ParamValueStatic};
    use crate::utilities::enums::Kernel;
    use std::panic::{catch_unwind, AssertUnwindSafe};

    fn sample_series() -> Vec<f32> {
        (1..=128).map(|v| v as f32).collect()
    }

    fn sample_ohlc(len: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let open: Vec<f32> = (0..len)
            .map(|i| 100.0f32 + (i as f32 * 0.1) + ((i as f32) * 0.03).sin())
            .collect();
        let high: Vec<f32> = open
            .iter()
            .enumerate()
            .map(|(i, v)| v + 0.8 + ((i as f32) * 0.02).cos().abs() * 0.3)
            .collect();
        let low: Vec<f32> = open
            .iter()
            .enumerate()
            .map(|(i, v)| v - 0.8 - ((i as f32) * 0.02).sin().abs() * 0.3)
            .collect();
        let close: Vec<f32> = open
            .iter()
            .enumerate()
            .map(|(i, v)| v + ((i as f32) * 0.05).sin() * 0.4)
            .collect();
        (open, high, low, close)
    }

    fn to_f64(values: &[f32]) -> Vec<f64> {
        values.iter().map(|&v| v as f64).collect()
    }

    #[test]
    fn unknown_indicator_is_rejected() {
        let data = sample_series();
        let req = IndicatorCudaRequest {
            indicator_id: "does_not_exist",
            output_id: Some("value"),
            data: IndicatorCudaDataRef::Slice { values: &data },
            params: &[],
            kernel: Kernel::Auto,
            target: CudaOutputTarget::HostF32,
        };
        let err = compute_cuda(req).unwrap_err();
        match err {
            IndicatorDispatchError::UnknownIndicator { id } => {
                assert_eq!(id, "does_not_exist");
            }
            other => panic!("expected UnknownIndicator, got {other:?}"),
        }
    }

    #[test]
    fn unsupported_indicator_is_rejected_without_fallback() {
        let data = sample_series();
        let req = IndicatorCudaRequest {
            indicator_id: "ad",
            output_id: Some("value"),
            data: IndicatorCudaDataRef::Slice { values: &data },
            params: &[],
            kernel: Kernel::Auto,
            target: CudaOutputTarget::HostF32,
        };
        let err = compute_cuda(req).unwrap_err();
        match err {
            IndicatorDispatchError::UnsupportedCapability {
                indicator,
                capability,
            } => {
                assert_eq!(indicator, "ad");
                assert_eq!(capability, "cuda_batch");
            }
            other => panic!("expected UnsupportedCapability, got {other:?}"),
        }
    }

    #[test]
    fn output_id_is_validated() {
        let data = sample_series();
        let req = IndicatorCudaRequest {
            indicator_id: "sma",
            output_id: Some("hist"),
            data: IndicatorCudaDataRef::Slice { values: &data },
            params: &[ParamKV {
                key: "period",
                value: ParamValue::Int(14),
            }],
            kernel: Kernel::Auto,
            target: CudaOutputTarget::HostF32,
        };
        let err = compute_cuda(req).unwrap_err();
        match err {
            IndicatorDispatchError::UnknownOutput { indicator, output } => {
                assert_eq!(indicator, "sma");
                assert_eq!(output, "hist");
            }
            other => panic!("expected UnknownOutput, got {other:?}"),
        }
    }

    #[test]
    fn host_output_matches_cpu_for_sma_when_cuda_available() {
        if !crate::cuda::cuda_available() {
            return;
        }
        let data = sample_series();
        let params = [ParamKV {
            key: "period",
            value: ParamValue::Int(14),
        }];
        let req_cuda = IndicatorCudaRequest {
            indicator_id: "sma",
            output_id: Some("value"),
            data: IndicatorCudaDataRef::Slice { values: &data },
            params: &params,
            kernel: Kernel::Auto,
            target: CudaOutputTarget::HostF32,
        };
        let out_cuda = compute_cuda(req_cuda).unwrap();
        assert_eq!(out_cuda.rows, 1);
        assert_eq!(out_cuda.cols, data.len());
        assert!(out_cuda.pattern_ids.is_none());
        let host = match out_cuda.series {
            IndicatorCudaSeries::HostF32(v) => v,
            _ => panic!("expected HostF32"),
        };

        let combo = [ParamKV {
            key: "period",
            value: ParamValue::Int(14),
        }];
        let combos = [IndicatorParamSet { params: &combo }];
        let data_f64 = to_f64(&data);
        let req_cpu = IndicatorBatchRequest {
            indicator_id: "sma",
            output_id: Some("value"),
            data: IndicatorDataRef::Slice { values: &data_f64 },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out_cpu = compute_cpu_batch(req_cpu).unwrap();
        let cpu = out_cpu.values_f64.unwrap();
        assert_eq!(host.len(), cpu.len());
        for i in 0..host.len() {
            let a = host[i] as f64;
            let b = cpu[i];
            if a.is_nan() && b.is_nan() {
                continue;
            }
            assert!((a - b).abs() <= 1e-3, "mismatch at index {i}: {a} vs {b}");
        }
    }

    #[test]
    fn host_output_matches_cpu_for_adx_when_cuda_available() {
        if !crate::cuda::cuda_available() {
            return;
        }

        let (_open, high, low, close) = sample_ohlc(192);
        let params = [ParamKV {
            key: "period",
            value: ParamValue::Int(14),
        }];

        let req_cuda = IndicatorCudaRequest {
            indicator_id: "adx",
            output_id: Some("value"),
            data: IndicatorCudaDataRef::Ohlc {
                open: &close,
                high: &high,
                low: &low,
                close: &close,
                source: None,
            },
            params: &params,
            kernel: Kernel::Auto,
            target: CudaOutputTarget::HostF32,
        };

        let out_cuda = compute_cuda(req_cuda).unwrap();
        assert_eq!(out_cuda.rows, 1);
        assert_eq!(out_cuda.cols, close.len());
        let cuda = match out_cuda.series {
            IndicatorCudaSeries::HostF32(v) => v,
            _ => panic!("expected HostF32"),
        };

        let combo = [ParamKV {
            key: "period",
            value: ParamValue::Int(14),
        }];
        let combos = [IndicatorParamSet { params: &combo }];
        let close_f64 = to_f64(&close);
        let high_f64 = to_f64(&high);
        let low_f64 = to_f64(&low);
        let req_cpu = IndicatorBatchRequest {
            indicator_id: "adx",
            output_id: Some("value"),
            data: IndicatorDataRef::Ohlc {
                open: &close_f64,
                high: &high_f64,
                low: &low_f64,
                close: &close_f64,
            },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out_cpu = compute_cpu_batch(req_cpu).unwrap();
        let cpu = out_cpu.values_f64.unwrap();
        assert_eq!(cuda.len(), cpu.len());
        for i in 0..cuda.len() {
            let a = cuda[i] as f64;
            let b = cpu[i];
            if a.is_nan() && b.is_nan() {
                continue;
            }
            assert!((a - b).abs() <= 1e-3, "mismatch at index {i}: {a} vs {b}");
        }
    }

    #[test]
    fn host_output_matches_cpu_for_yang_zhang_when_cuda_available() {
        if !crate::cuda::cuda_available() {
            return;
        }

        let (open, high, low, close) = sample_ohlc(192);
        let params = [
            ParamKV {
                key: "lookback",
                value: ParamValue::Int(21),
            },
            ParamKV {
                key: "k_override",
                value: ParamValue::Bool(true),
            },
            ParamKV {
                key: "k",
                value: ParamValue::Float(0.28),
            },
        ];
        let req_cuda = IndicatorCudaRequest {
            indicator_id: "yang_zhang_volatility",
            output_id: Some("rs"),
            data: IndicatorCudaDataRef::Ohlc {
                open: &open,
                high: &high,
                low: &low,
                close: &close,
                source: None,
            },
            params: &params,
            kernel: Kernel::Auto,
            target: CudaOutputTarget::HostF32,
        };

        let out_cuda = compute_cuda(req_cuda).unwrap();
        assert_eq!(out_cuda.rows, 1);
        assert_eq!(out_cuda.cols, close.len());
        let cuda = match out_cuda.series {
            IndicatorCudaSeries::HostF32(v) => v,
            _ => panic!("expected HostF32"),
        };

        let open_f64 = to_f64(&open);
        let high_f64 = to_f64(&high);
        let low_f64 = to_f64(&low);
        let close_f64 = to_f64(&close);
        let combos = [IndicatorParamSet { params: &params }];
        let req_cpu = IndicatorBatchRequest {
            indicator_id: "yang_zhang_volatility",
            output_id: Some("rs"),
            data: IndicatorDataRef::Ohlc {
                open: &open_f64,
                high: &high_f64,
                low: &low_f64,
                close: &close_f64,
            },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out_cpu = compute_cpu_batch(req_cpu).unwrap();
        let cpu = out_cpu.values_f64.unwrap();
        assert_eq!(cuda.len(), cpu.len());
        for i in 0..cuda.len() {
            let a = cuda[i] as f64;
            let b = cpu[i];
            if a.is_nan() && b.is_nan() {
                continue;
            }
            assert!((a - b).abs() <= 1e-3, "mismatch at index {i}: {a} vs {b}");
        }
    }

    #[test]
    fn device_output_exposes_non_null_handle_when_cuda_available() {
        if !crate::cuda::cuda_available() {
            return;
        }
        let data = sample_series();
        let params = [
            ParamKV {
                key: "period_start",
                value: ParamValue::Int(5),
            },
            ParamKV {
                key: "period_end",
                value: ParamValue::Int(7),
            },
            ParamKV {
                key: "period_step",
                value: ParamValue::Int(1),
            },
        ];
        let req = IndicatorCudaRequest {
            indicator_id: "sma",
            output_id: Some("value"),
            data: IndicatorCudaDataRef::Slice { values: &data },
            params: &params,
            kernel: Kernel::Auto,
            target: CudaOutputTarget::DeviceF32,
        };
        let out = compute_cuda(req).unwrap();
        assert_eq!(out.rows, 3);
        assert_eq!(out.cols, data.len());
        assert!(out.pattern_ids.is_none());
        match out.series {
            IndicatorCudaSeries::DeviceF32(dev) => {
                assert_ne!(dev.device_ptr, 0);
                assert_eq!(dev.rows, 3);
                assert_eq!(dev.cols, data.len());
                assert!(!dev.is_empty());
                assert_eq!(dev.len(), 3 * data.len());
                assert_eq!(dev.owner().rows, dev.rows);
                assert_eq!(dev.owner().cols, dev.cols);
            }
            _ => panic!("expected DeviceF32"),
        }
    }

    #[test]
    fn pattern_output_id_is_validated() {
        let (open, high, low, close) = sample_ohlc(128);
        let req = IndicatorCudaRequest {
            indicator_id: "pattern_recognition",
            output_id: Some("value"),
            data: IndicatorCudaDataRef::Ohlc {
                open: &open,
                high: &high,
                low: &low,
                close: &close,
                source: None,
            },
            params: &[],
            kernel: Kernel::Auto,
            target: CudaOutputTarget::HostF32,
        };
        let err = compute_cuda(req).unwrap_err();
        match err {
            IndicatorDispatchError::UnknownOutput { indicator, output } => {
                assert_eq!(indicator, "pattern_recognition");
                assert_eq!(output, "value");
            }
            other => panic!("expected UnknownOutput, got {other:?}"),
        }
    }

    #[test]
    fn pattern_requires_ohlc_shape() {
        let data = sample_series();
        let req = IndicatorCudaRequest {
            indicator_id: "pattern_recognition",
            output_id: Some("matrix"),
            data: IndicatorCudaDataRef::Slice { values: &data },
            params: &[],
            kernel: Kernel::Auto,
            target: CudaOutputTarget::HostF32,
        };
        let err = compute_cuda(req).unwrap_err();
        match err {
            IndicatorDispatchError::MissingRequiredInput { indicator, input } => {
                assert_eq!(indicator, "pattern_recognition");
                assert_eq!(input, IndicatorInputKind::Ohlc);
            }
            other => panic!("expected MissingRequiredInput, got {other:?}"),
        }
    }

    #[test]
    fn pattern_rejects_unknown_param_key() {
        let (open, high, low, close) = sample_ohlc(128);
        let params = [ParamKV {
            key: "period",
            value: ParamValue::Int(14),
        }];
        let req = IndicatorCudaRequest {
            indicator_id: "pattern_recognition",
            output_id: Some("matrix"),
            data: IndicatorCudaDataRef::Ohlc {
                open: &open,
                high: &high,
                low: &low,
                close: &close,
                source: None,
            },
            params: &params,
            kernel: Kernel::Auto,
            target: CudaOutputTarget::HostF32,
        };
        let err = compute_cuda(req).unwrap_err();
        match err {
            IndicatorDispatchError::InvalidParam { indicator, key, .. } => {
                assert_eq!(indicator, "pattern_recognition");
                assert_eq!(key, "period");
            }
            other => panic!("expected InvalidParam, got {other:?}"),
        }
    }

    #[test]
    fn pattern_host_output_matches_cpu_when_cuda_available() {
        if !crate::cuda::cuda_available() {
            return;
        }

        let (open, high, low, close) = sample_ohlc(192);
        let req_cuda = IndicatorCudaRequest {
            indicator_id: "pattern_recognition",
            output_id: Some("matrix"),
            data: IndicatorCudaDataRef::Ohlc {
                open: &open,
                high: &high,
                low: &low,
                close: &close,
                source: None,
            },
            params: &[],
            kernel: Kernel::Auto,
            target: CudaOutputTarget::HostF32,
        };
        let out_cuda = compute_cuda(req_cuda).unwrap();
        let open_f64 = to_f64(&open);
        let high_f64 = to_f64(&high);
        let low_f64 = to_f64(&low);
        let close_f64 = to_f64(&close);
        assert_eq!(
            out_cuda.rows,
            out_cpu_rows(&open_f64, &high_f64, &low_f64, &close_f64)
        );
        assert_eq!(out_cuda.cols, close.len());
        let ids = out_cuda.pattern_ids.clone().unwrap();
        assert_eq!(ids.len(), out_cuda.rows);
        let cuda_values = match out_cuda.series {
            IndicatorCudaSeries::HostF32(v) => v,
            _ => panic!("expected HostF32"),
        };

        let req_cpu = IndicatorComputeRequest {
            indicator_id: "pattern_recognition",
            output_id: Some("matrix"),
            data: IndicatorDataRef::Ohlc {
                open: &open_f64,
                high: &high_f64,
                low: &low_f64,
                close: &close_f64,
            },
            params: &[],
            kernel: Kernel::Auto,
        };
        let out_cpu = compute_cpu(req_cpu).unwrap();
        assert_eq!(out_cuda.rows, out_cpu.rows);
        assert_eq!(out_cuda.cols, out_cpu.cols);
        let cpu_ids = out_cpu.pattern_ids.clone().unwrap();
        assert_eq!(ids, cpu_ids);
        let cpu_values = match out_cpu.series {
            IndicatorSeries::Bool(v) => v,
            _ => panic!("expected Bool output"),
        };

        assert_eq!(cuda_values.len(), cpu_values.len());
        let mut mismatches = 0usize;
        for i in 0..cuda_values.len() {
            let expected = if cpu_values[i] { 1.0 } else { 0.0 };
            if cuda_values[i] != expected {
                mismatches += 1;
            }
        }
        let mismatch_ratio = mismatches as f64 / cuda_values.len() as f64;
        assert!(
            mismatch_ratio <= 0.01,
            "CUDA pattern mismatch ratio too high: mismatches={} total={} ratio={:.6}",
            mismatches,
            cuda_values.len(),
            mismatch_ratio
        );
    }

    #[test]
    fn pattern_device_output_exposes_non_null_handle_when_cuda_available() {
        if !crate::cuda::cuda_available() {
            return;
        }

        let (open, high, low, close) = sample_ohlc(160);
        let req = IndicatorCudaRequest {
            indicator_id: "pattern_recognition",
            output_id: Some("matrix"),
            data: IndicatorCudaDataRef::Ohlc {
                open: &open,
                high: &high,
                low: &low,
                close: &close,
                source: None,
            },
            params: &[],
            kernel: Kernel::Auto,
            target: CudaOutputTarget::DeviceF32,
        };
        let out = compute_cuda(req).unwrap();
        assert_eq!(out.cols, close.len());
        let ids = out.pattern_ids.clone().unwrap();
        assert_eq!(ids.len(), out.rows);
        match out.series {
            IndicatorCudaSeries::DeviceF32(dev) => {
                assert_ne!(dev.device_ptr, 0);
                assert_eq!(dev.cols, close.len());
                assert!(dev.rows > 0);
                assert_eq!(dev.len(), dev.rows * dev.cols);
                assert_eq!(dev.rows, out.rows);
                assert_eq!(dev.cols, out.cols);
            }
            _ => panic!("expected DeviceF32"),
        }
    }

    struct ProbeInputData {
        timestamp: Vec<i64>,
        open: Vec<f32>,
        high: Vec<f32>,
        low: Vec<f32>,
        close: Vec<f32>,
        volume: Vec<f32>,
    }

    impl ProbeInputData {
        fn new(len: usize) -> Self {
            let mut timestamp = Vec::with_capacity(len);
            let mut open = Vec::with_capacity(len);
            let mut high = Vec::with_capacity(len);
            let mut low = Vec::with_capacity(len);
            let mut close = Vec::with_capacity(len);
            let mut volume = Vec::with_capacity(len);
            let mut base = 100.0f32;
            for i in 0..len {
                let t = i as f32;
                let drift = 0.0009 * t;
                let wave = (t * 0.017).sin() * 0.85 + (t * 0.004).cos() * 0.42;
                let o = base + drift + wave;
                let h = o + 0.7 + (t * 0.019).sin().abs() * 0.33;
                let l = o - 0.7 - (t * 0.013).cos().abs() * 0.31;
                let c = o + (t * 0.029).sin() * 0.25;
                let v = 1000.0 + (t * 0.37).sin() * 70.0 + (i % 89) as f32;
                timestamp.push(1_700_000_000_000i64 + (i as i64) * 60_000i64);
                open.push(o);
                high.push(h);
                low.push(l);
                close.push(c);
                volume.push(v.max(1.0));
                base = c;
            }
            Self {
                timestamp,
                open,
                high,
                low,
                close,
                volume,
            }
        }
    }

    fn probe_data_for_indicator<'a>(
        kind: IndicatorInputKind,
        data: &'a ProbeInputData,
    ) -> IndicatorCudaDataRef<'a> {
        match kind {
            IndicatorInputKind::Slice => IndicatorCudaDataRef::Slice {
                values: &data.close,
            },
            IndicatorInputKind::Ohlc => IndicatorCudaDataRef::Ohlc {
                open: &data.open,
                high: &data.high,
                low: &data.low,
                close: &data.close,
                source: Some(&data.close),
            },
            IndicatorInputKind::Ohlcv | IndicatorInputKind::Candles => {
                IndicatorCudaDataRef::Ohlcv {
                    timestamp: Some(&data.timestamp),
                    open: &data.open,
                    high: &data.high,
                    low: &data.low,
                    close: &data.close,
                    volume: &data.volume,
                    source: Some(&data.close),
                }
            }
            IndicatorInputKind::HighLow => IndicatorCudaDataRef::HighLow {
                high: &data.high,
                low: &data.low,
            },
            IndicatorInputKind::CloseVolume => IndicatorCudaDataRef::CloseVolume {
                close: &data.close,
                volume: &data.volume,
            },
        }
    }

    fn probe_default_params(info: &IndicatorInfo) -> Vec<ParamKV<'static>> {
        let mut out = Vec::with_capacity(info.params.len());
        for p in info.params.iter() {
            if p.key.eq_ignore_ascii_case("source")
                || p.key.eq_ignore_ascii_case("output")
                || p.key.eq_ignore_ascii_case("device_id")
            {
                continue;
            }
            if !p.required && p.default.is_none() {
                continue;
            }
            let value = match (p.kind, p.default) {
                (_, Some(ParamValueStatic::Int(v))) => ParamValue::Int(v),
                (_, Some(ParamValueStatic::Float(v))) => ParamValue::Float(v),
                (_, Some(ParamValueStatic::Bool(v))) => ParamValue::Bool(v),
                (_, Some(ParamValueStatic::EnumString(v))) => ParamValue::EnumString(v),
                (IndicatorParamKind::Int, None) => {
                    let v = p.min.unwrap_or(1.0).round() as i64;
                    ParamValue::Int(if v <= 0 { 1 } else { v })
                }
                (IndicatorParamKind::Float, None) => {
                    let v = p.min.unwrap_or(1.0);
                    ParamValue::Float(if v.is_finite() { v } else { 1.0 })
                }
                (IndicatorParamKind::Bool, None) => ParamValue::Bool(false),
                (IndicatorParamKind::EnumString, None) => {
                    let v = p.enum_values.first().copied().unwrap_or("sma");
                    ParamValue::EnumString(v)
                }
            };
            out.push(ParamKV { key: p.key, value });
        }

        if info.id.eq_ignore_ascii_case("buff_averages") {
            for kv in out.iter_mut() {
                if kv.key.eq_ignore_ascii_case("fast_period") {
                    kv.value = ParamValue::Int(5);
                } else if kv.key.eq_ignore_ascii_case("slow_period") {
                    kv.value = ParamValue::Int(20);
                }
            }
            if !out
                .iter()
                .any(|kv| kv.key.eq_ignore_ascii_case("fast_period"))
            {
                out.push(ParamKV {
                    key: "fast_period",
                    value: ParamValue::Int(5),
                });
            }
            if !out
                .iter()
                .any(|kv| kv.key.eq_ignore_ascii_case("slow_period"))
            {
                out.push(ParamKV {
                    key: "slow_period",
                    value: ParamValue::Int(20),
                });
            }
        }

        out
    }

    fn run_probe_once(
        indicator_id: &str,
        target: CudaOutputTarget,
    ) -> Result<IndicatorCudaOutput, String> {
        let info = get_indicator(indicator_id)
            .ok_or_else(|| format!("unknown indicator '{}'", indicator_id))?;
        let params = probe_default_params(info);
        let data = ProbeInputData::new(4096);
        let req = IndicatorCudaRequest {
            indicator_id: info.id,
            output_id: info.outputs.first().map(|o| o.id).or(Some("value")),
            data: probe_data_for_indicator(info.input_kind, &data),
            params: params.as_slice(),
            kernel: Kernel::Auto,
            target,
        };

        match catch_unwind(AssertUnwindSafe(|| compute_cuda(req))) {
            Ok(Ok(v)) => Ok(v),
            Ok(Err(e)) => Err(format!("error: {}", e)),
            Err(p) => {
                if let Some(s) = p.downcast_ref::<&str>() {
                    return Err(format!("panic: {}", s));
                }
                if let Some(s) = p.downcast_ref::<String>() {
                    return Err(format!("panic: {}", s));
                }
                Err("panic".to_string())
            }
        }
    }

    fn probe_indicator_stability(indicator_id: &str) -> Result<(), String> {
        let dev = run_probe_once(indicator_id, CudaOutputTarget::DeviceF32)
            .map_err(|e| format!("{} device target {}", indicator_id, e))?;
        match dev.series {
            IndicatorCudaSeries::DeviceF32(_) => {}
            _ => {
                return Err(format!(
                    "{}: expected DeviceF32 result for device target",
                    indicator_id
                ));
            }
        }
        let host = run_probe_once(indicator_id, CudaOutputTarget::HostF32)
            .map_err(|e| format!("{} host target {}", indicator_id, e))?;
        match host.series {
            IndicatorCudaSeries::HostF32(_) => {}
            _ => {
                return Err(format!(
                    "{}: expected HostF32 result for host target",
                    indicator_id
                ));
            }
        }
        Ok(())
    }

    macro_rules! stability_probe_test {
        ($name:ident, $id:literal) => {
            #[test]
            #[ignore]
            fn $name() {
                if !crate::cuda::cuda_available() {
                    return;
                }
                if let Err(e) = probe_indicator_stability($id) {
                    panic!("{}", e);
                }
            }
        };
    }

    stability_probe_test!(stability_probe_buff_averages, "buff_averages");
    stability_probe_test!(stability_probe_mab, "mab");
    stability_probe_test!(stability_probe_tsf, "tsf");
    stability_probe_test!(stability_probe_aso, "aso");
    stability_probe_test!(stability_probe_stoch, "stoch");
    stability_probe_test!(stability_probe_hwma, "hwma");
    stability_probe_test!(stability_probe_ehlers_itrend, "ehlers_itrend");
    stability_probe_test!(stability_probe_hma, "hma");
    stability_probe_test!(stability_probe_gaussian, "gaussian");
    stability_probe_test!(stability_probe_sma, "sma");
    stability_probe_test!(stability_probe_mean_ad, "mean_ad");
    stability_probe_test!(stability_probe_supersmoother_3_pole, "supersmoother_3_pole");

    fn out_cpu_rows(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> usize {
        let req_cpu = IndicatorComputeRequest {
            indicator_id: "pattern_recognition",
            output_id: Some("matrix"),
            data: IndicatorDataRef::Ohlc {
                open,
                high,
                low,
                close,
            },
            params: &[],
            kernel: Kernel::Auto,
        };
        let out = compute_cpu(req_cpu).unwrap();
        out.rows
    }
}
