use super::{
    compute_cpu_batch, IndicatorBatchOutput, IndicatorBatchRequest, IndicatorDataRef,
    IndicatorDispatchError, IndicatorParamSet, ParamKV, ParamValue,
};
use crate::indicators::dx::{dx_batch_with_kernel, DxBatchRange};
use crate::indicators::mfi::{mfi_batch_with_kernel, MfiBatchRange};
use crate::indicators::moving_averages::sma::{sma_batch_with_kernel, SmaBatchRange};
use crate::indicators::registry::get_indicator;
use crate::utilities::data_loader::source_type;
use crate::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use super::{
    compute_cuda, CudaOutputTarget, IndicatorCudaDataRef, IndicatorCudaOutput, IndicatorCudaRequest,
};

#[derive(Debug, Clone, PartialEq)]
enum OwnedParamValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    EnumString(String),
}

#[derive(Debug, Clone, PartialEq)]
struct OwnedParamKV {
    key: String,
    value: OwnedParamValue,
}

#[derive(Debug, Clone, PartialEq)]
enum CpuCompiledPlan {
    Generic,
    SmaPeriod { period: usize },
    MfiPeriod { period: usize },
    DxPeriod { period: usize },
}

#[derive(Debug, Clone, PartialEq)]
pub struct CompiledIndicatorCall {
    indicator_id: String,
    output_id: Option<String>,
    params: Vec<OwnedParamKV>,
    cpu_plan: CpuCompiledPlan,
    prefer_cuda: bool,
}

impl CompiledIndicatorCall {
    pub fn indicator_id(&self) -> &str {
        &self.indicator_id
    }

    pub fn output_id(&self) -> Option<&str> {
        self.output_id.as_deref()
    }

    pub fn prefer_cuda(&self) -> bool {
        self.prefer_cuda
    }

    fn as_param_kv(&self) -> Vec<ParamKV<'_>> {
        let mut out = Vec::with_capacity(self.params.len());
        for p in &self.params {
            let value = match &p.value {
                OwnedParamValue::Int(v) => ParamValue::Int(*v),
                OwnedParamValue::Float(v) => ParamValue::Float(*v),
                OwnedParamValue::Bool(v) => ParamValue::Bool(*v),
                OwnedParamValue::EnumString(v) => ParamValue::EnumString(v.as_str()),
            };
            out.push(ParamKV {
                key: p.key.as_str(),
                value,
            });
        }
        out
    }
}

fn parse_usize_param_value(
    indicator: &str,
    key: &str,
    value: ParamValue<'_>,
) -> Result<usize, IndicatorDispatchError> {
    match value {
        ParamValue::Int(v) => {
            if v < 0 {
                return Err(IndicatorDispatchError::InvalidParam {
                    indicator: indicator.to_string(),
                    key: key.to_string(),
                    reason: "expected integer >= 0".to_string(),
                });
            }
            Ok(v as usize)
        }
        ParamValue::Float(v) => {
            if !v.is_finite() {
                return Err(IndicatorDispatchError::InvalidParam {
                    indicator: indicator.to_string(),
                    key: key.to_string(),
                    reason: "expected finite number".to_string(),
                });
            }
            if v < 0.0 {
                return Err(IndicatorDispatchError::InvalidParam {
                    indicator: indicator.to_string(),
                    key: key.to_string(),
                    reason: "expected number >= 0".to_string(),
                });
            }
            let rounded = v.round();
            if (v - rounded).abs() > 1e-9 {
                return Err(IndicatorDispatchError::InvalidParam {
                    indicator: indicator.to_string(),
                    key: key.to_string(),
                    reason: "expected whole number".to_string(),
                });
            }
            Ok(rounded as usize)
        }
        _ => Err(IndicatorDispatchError::InvalidParam {
            indicator: indicator.to_string(),
            key: key.to_string(),
            reason: "expected Int or Float".to_string(),
        }),
    }
}

fn compile_period_only_plan(
    indicator: &str,
    selected_output: Option<&str>,
    params: &[ParamKV<'_>],
) -> Result<CpuCompiledPlan, IndicatorDispatchError> {
    let supports_fast_route = indicator.eq_ignore_ascii_case("sma")
        || indicator.eq_ignore_ascii_case("mfi")
        || indicator.eq_ignore_ascii_case("dx");
    if !supports_fast_route {
        return Ok(CpuCompiledPlan::Generic);
    }

    let is_value = selected_output
        .map(|out| out.eq_ignore_ascii_case("value"))
        .unwrap_or(true);
    if !is_value {
        return Ok(CpuCompiledPlan::Generic);
    }

    let mut period: Option<usize> = None;
    for p in params {
        if p.key.eq_ignore_ascii_case("period") {
            period = Some(parse_usize_param_value(indicator, "period", p.value)?);
        } else {
            return Ok(CpuCompiledPlan::Generic);
        }
    }
    let period = period.unwrap_or(14);

    if indicator.eq_ignore_ascii_case("sma") {
        return Ok(CpuCompiledPlan::SmaPeriod { period });
    }
    if indicator.eq_ignore_ascii_case("mfi") {
        return Ok(CpuCompiledPlan::MfiPeriod { period });
    }
    if indicator.eq_ignore_ascii_case("dx") {
        return Ok(CpuCompiledPlan::DxPeriod { period });
    }
    Ok(CpuCompiledPlan::Generic)
}

pub fn compile_call(
    indicator_id: &str,
    output_id: Option<&str>,
    params: &[ParamKV<'_>],
    prefer_cuda: bool,
) -> Result<CompiledIndicatorCall, IndicatorDispatchError> {
    let info =
        get_indicator(indicator_id).ok_or_else(|| IndicatorDispatchError::UnknownIndicator {
            id: indicator_id.to_string(),
        })?;

    if info.outputs.len() > 1 && output_id.is_none() {
        return Err(IndicatorDispatchError::InvalidParam {
            indicator: info.id.to_string(),
            key: "output_id".to_string(),
            reason: "output_id is required for multi-output indicators".to_string(),
        });
    }

    if let Some(out_id) = output_id {
        let exists = info
            .outputs
            .iter()
            .any(|out| out.id.eq_ignore_ascii_case(out_id));
        if !exists {
            return Err(IndicatorDispatchError::UnknownOutput {
                indicator: info.id.to_string(),
                output: out_id.to_string(),
            });
        }
    }

    if prefer_cuda && !info.capabilities.supports_cuda_batch {
        return Err(IndicatorDispatchError::UnsupportedCapability {
            indicator: info.id.to_string(),
            capability: "cuda_batch",
        });
    }

    let mut owned_params = Vec::with_capacity(params.len());
    for param in params {
        let value = match param.value {
            ParamValue::Int(v) => OwnedParamValue::Int(v),
            ParamValue::Float(v) => OwnedParamValue::Float(v),
            ParamValue::Bool(v) => OwnedParamValue::Bool(v),
            ParamValue::EnumString(v) => OwnedParamValue::EnumString(v.to_string()),
        };
        owned_params.push(OwnedParamKV {
            key: param.key.to_string(),
            value,
        });
    }

    let selected_output = output_id.or_else(|| {
        if info.outputs.len() == 1 {
            Some(info.outputs[0].id)
        } else {
            None
        }
    });
    let cpu_plan = compile_period_only_plan(info.id, selected_output, params)?;

    Ok(CompiledIndicatorCall {
        indicator_id: info.id.to_string(),
        output_id: output_id.map(str::to_string),
        params: owned_params,
        cpu_plan,
        prefer_cuda,
    })
}

pub fn run_compiled_cpu(
    call: &CompiledIndicatorCall,
    data: IndicatorDataRef<'_>,
    kernel: Kernel,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    match call.cpu_plan {
        CpuCompiledPlan::SmaPeriod { period } => {
            let series = match data {
                IndicatorDataRef::Slice { values } => values,
                IndicatorDataRef::Candles { candles, source } => {
                    source_type(candles, source.unwrap_or("close"))
                }
                IndicatorDataRef::Ohlc { close, .. } => close,
                IndicatorDataRef::Ohlcv { close, .. } => close,
                IndicatorDataRef::CloseVolume { close, .. } => close,
                IndicatorDataRef::HighLow { .. } => {
                    return Err(IndicatorDispatchError::MissingRequiredInput {
                        indicator: "sma".to_string(),
                        input: crate::indicators::registry::IndicatorInputKind::Slice,
                    });
                }
            };
            let out = sma_batch_with_kernel(
                series,
                &SmaBatchRange {
                    period: (period, period, 0),
                },
                to_batch_kernel(kernel),
            )
            .map_err(|e| IndicatorDispatchError::ComputeFailed {
                indicator: "sma".to_string(),
                details: e.to_string(),
            })?;
            return Ok(f64_output(
                call.output_id.as_deref().unwrap_or("value"),
                out.rows,
                out.cols,
                out.values,
            ));
        }
        CpuCompiledPlan::MfiPeriod { period } => {
            let mut derived_typical_price: Option<Vec<f64>> = None;
            let (typical_price, volume): (&[f64], &[f64]) = match data {
                IndicatorDataRef::Candles { candles, source } => (
                    source_type(candles, source.unwrap_or("hlc3")),
                    candles.volume.as_slice(),
                ),
                IndicatorDataRef::Ohlcv {
                    open,
                    high,
                    low,
                    close,
                    volume,
                } => {
                    ensure_same_len_5(
                        "mfi",
                        open.len(),
                        high.len(),
                        low.len(),
                        close.len(),
                        volume.len(),
                    )?;
                    derived_typical_price = Some(
                        high.iter()
                            .zip(low)
                            .zip(close)
                            .map(|((h, l), c)| (h + l + c) / 3.0)
                            .collect(),
                    );
                    (derived_typical_price.as_deref().unwrap_or(close), volume)
                }
                IndicatorDataRef::CloseVolume { close, volume } => {
                    ensure_same_len_2("mfi", close.len(), volume.len())?;
                    (close, volume)
                }
                _ => {
                    return Err(IndicatorDispatchError::MissingRequiredInput {
                        indicator: "mfi".to_string(),
                        input: crate::indicators::registry::IndicatorInputKind::CloseVolume,
                    });
                }
            };
            let out = mfi_batch_with_kernel(
                typical_price,
                volume,
                &MfiBatchRange {
                    period: (period, period, 0),
                },
                to_batch_kernel(kernel),
            )
            .map_err(|e| IndicatorDispatchError::ComputeFailed {
                indicator: "mfi".to_string(),
                details: e.to_string(),
            })?;
            return Ok(f64_output(
                call.output_id.as_deref().unwrap_or("value"),
                out.rows,
                out.cols,
                out.values,
            ));
        }
        CpuCompiledPlan::DxPeriod { period } => {
            let (high, low, close): (&[f64], &[f64], &[f64]) = match data {
                IndicatorDataRef::Candles { candles, .. } => (
                    candles.high.as_slice(),
                    candles.low.as_slice(),
                    candles.close.as_slice(),
                ),
                IndicatorDataRef::Ohlc {
                    open,
                    high,
                    low,
                    close,
                } => {
                    ensure_same_len_4("dx", open.len(), high.len(), low.len(), close.len())?;
                    (high, low, close)
                }
                IndicatorDataRef::Ohlcv {
                    open,
                    high,
                    low,
                    close,
                    volume,
                } => {
                    ensure_same_len_5(
                        "dx",
                        open.len(),
                        high.len(),
                        low.len(),
                        close.len(),
                        volume.len(),
                    )?;
                    (high, low, close)
                }
                _ => {
                    return Err(IndicatorDispatchError::MissingRequiredInput {
                        indicator: "dx".to_string(),
                        input: crate::indicators::registry::IndicatorInputKind::Ohlc,
                    });
                }
            };
            let out = dx_batch_with_kernel(
                high,
                low,
                close,
                &DxBatchRange {
                    period: (period, period, 0),
                },
                to_batch_kernel(kernel),
            )
            .map_err(|e| IndicatorDispatchError::ComputeFailed {
                indicator: "dx".to_string(),
                details: e.to_string(),
            })?;
            return Ok(f64_output(
                call.output_id.as_deref().unwrap_or("value"),
                out.rows,
                out.cols,
                out.values,
            ));
        }
        CpuCompiledPlan::Generic => {}
    }

    let params = call.as_param_kv();
    let combos = [IndicatorParamSet {
        params: params.as_slice(),
    }];
    compute_cpu_batch(IndicatorBatchRequest {
        indicator_id: call.indicator_id.as_str(),
        output_id: call.output_id.as_deref(),
        data,
        combos: &combos,
        kernel,
    })
}

fn to_batch_kernel(kernel: Kernel) -> Kernel {
    match kernel {
        Kernel::Auto => Kernel::Auto,
        Kernel::Scalar => Kernel::ScalarBatch,
        Kernel::Avx2 => Kernel::Avx2Batch,
        Kernel::Avx512 => Kernel::Avx512Batch,
        other => other,
    }
}

fn ensure_same_len_2(indicator: &str, a: usize, b: usize) -> Result<(), IndicatorDispatchError> {
    if a == b {
        return Ok(());
    }
    Err(IndicatorDispatchError::DataLengthMismatch {
        details: format!("{indicator}: expected equal lengths, got {a} and {b}"),
    })
}

fn ensure_same_len_4(
    indicator: &str,
    a: usize,
    b: usize,
    c: usize,
    d: usize,
) -> Result<(), IndicatorDispatchError> {
    if a == b && b == c && c == d {
        return Ok(());
    }
    Err(IndicatorDispatchError::DataLengthMismatch {
        details: format!("{indicator}: expected equal lengths, got {a}, {b}, {c}, {d}"),
    })
}

fn ensure_same_len_5(
    indicator: &str,
    a: usize,
    b: usize,
    c: usize,
    d: usize,
    e: usize,
) -> Result<(), IndicatorDispatchError> {
    if a == b && b == c && c == d && d == e {
        return Ok(());
    }
    Err(IndicatorDispatchError::DataLengthMismatch {
        details: format!("{indicator}: expected equal lengths, got {a}, {b}, {c}, {d}, {e}"),
    })
}

fn f64_output(output_id: &str, rows: usize, cols: usize, values: Vec<f64>) -> IndicatorBatchOutput {
    IndicatorBatchOutput {
        output_id: output_id.to_string(),
        rows,
        cols,
        values_f64: Some(values),
        values_i32: None,
        values_bool: None,
    }
}

#[cfg(feature = "cuda")]
pub fn run_compiled_cuda(
    call: &CompiledIndicatorCall,
    data: IndicatorCudaDataRef<'_>,
    kernel: Kernel,
    target: CudaOutputTarget,
) -> Result<IndicatorCudaOutput, IndicatorDispatchError> {
    let params = call.as_param_kv();
    compute_cuda(IndicatorCudaRequest {
        indicator_id: call.indicator_id.as_str(),
        output_id: call.output_id.as_deref(),
        data,
        params: params.as_slice(),
        kernel,
        target,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::dispatch::{compute_cpu_batch, IndicatorBatchRequest};

    fn sample_series() -> Vec<f64> {
        (1..=128).map(|v| v as f64).collect()
    }

    fn sample_ohlc() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let open: Vec<f64> = (0..128).map(|i| 100.0 + (i as f64 * 0.1)).collect();
        let high: Vec<f64> = open.iter().map(|v| v + 1.0).collect();
        let low: Vec<f64> = open.iter().map(|v| v - 1.0).collect();
        let close: Vec<f64> = open.iter().map(|v| v + 0.25).collect();
        (open, high, low, close)
    }

    #[test]
    fn compile_rejects_unknown_indicator() {
        let err = compile_call("does_not_exist", Some("value"), &[], false).unwrap_err();
        match err {
            IndicatorDispatchError::UnknownIndicator { id } => assert_eq!(id, "does_not_exist"),
            other => panic!("expected UnknownIndicator, got {other:?}"),
        }
    }

    #[test]
    fn compile_validates_output_id() {
        let err = compile_call("sma", Some("hist"), &[], false).unwrap_err();
        match err {
            IndicatorDispatchError::UnknownOutput { indicator, output } => {
                assert_eq!(indicator, "sma");
                assert_eq!(output, "hist");
            }
            other => panic!("expected UnknownOutput, got {other:?}"),
        }
    }

    #[test]
    fn run_compiled_cpu_matches_direct_dispatch() {
        let data = sample_series();
        let params = [ParamKV {
            key: "period",
            value: ParamValue::Int(14),
        }];
        let call = compile_call("sma", Some("value"), &params, false).unwrap();
        let compiled = run_compiled_cpu(
            &call,
            IndicatorDataRef::Slice { values: &data },
            Kernel::Auto,
        )
        .unwrap();

        let combos = [IndicatorParamSet { params: &params }];
        let direct = compute_cpu_batch(IndicatorBatchRequest {
            indicator_id: "sma",
            output_id: Some("value"),
            data: IndicatorDataRef::Slice { values: &data },
            combos: &combos,
            kernel: Kernel::Auto,
        })
        .unwrap();
        assert_eq!(compiled.output_id, direct.output_id);
        assert_eq!(compiled.rows, direct.rows);
        assert_eq!(compiled.cols, direct.cols);
        let compiled_values = compiled.values_f64.unwrap();
        let direct_values = direct.values_f64.unwrap();
        assert_eq!(compiled_values.len(), direct_values.len());
        for i in 0..compiled_values.len() {
            let a = compiled_values[i];
            let b = direct_values[i];
            if a.is_nan() && b.is_nan() {
                continue;
            }
            assert!((a - b).abs() <= 1e-12, "mismatch at index {i}: {a} vs {b}");
        }
    }

    #[test]
    fn compile_pre_resolves_sma_period_plan() {
        let params = [ParamKV {
            key: "period",
            value: ParamValue::Int(9),
        }];
        let call = compile_call("sma", Some("value"), &params, false).unwrap();
        assert!(matches!(
            call.cpu_plan,
            CpuCompiledPlan::SmaPeriod { period: 9 }
        ));
    }

    #[test]
    fn compile_falls_back_to_generic_when_params_are_not_period_only() {
        let params = [
            ParamKV {
                key: "period",
                value: ParamValue::Int(9),
            },
            ParamKV {
                key: "unused",
                value: ParamValue::Float(1.0),
            },
        ];
        let call = compile_call("mfi", Some("value"), &params, false).unwrap();
        assert!(matches!(call.cpu_plan, CpuCompiledPlan::Generic));
    }

    #[test]
    fn run_compiled_mfi_fast_plan_matches_dispatch() {
        let (open, high, low, close) = sample_ohlc();
        let volume: Vec<f64> = (0..close.len()).map(|i| 1000.0 + (i as f64)).collect();
        let params = [ParamKV {
            key: "period",
            value: ParamValue::Int(14),
        }];
        let call = compile_call("mfi", Some("value"), &params, false).unwrap();
        let compiled = run_compiled_cpu(
            &call,
            IndicatorDataRef::Ohlcv {
                open: &open,
                high: &high,
                low: &low,
                close: &close,
                volume: &volume,
            },
            Kernel::Auto,
        )
        .unwrap();
        let combos = [IndicatorParamSet { params: &params }];
        let direct = compute_cpu_batch(IndicatorBatchRequest {
            indicator_id: "mfi",
            output_id: Some("value"),
            data: IndicatorDataRef::Ohlcv {
                open: &open,
                high: &high,
                low: &low,
                close: &close,
                volume: &volume,
            },
            combos: &combos,
            kernel: Kernel::Auto,
        })
        .unwrap();
        assert_eq!(compiled.rows, direct.rows);
        assert_eq!(compiled.cols, direct.cols);
        let a = compiled.values_f64.unwrap();
        let b = direct.values_f64.unwrap();
        assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            let x = a[i];
            let y = b[i];
            if x.is_nan() && y.is_nan() {
                continue;
            }
            assert!((x - y).abs() <= 1e-12, "mismatch at index {i}: {x} vs {y}");
        }
    }

    #[test]
    fn run_compiled_dx_fast_plan_matches_dispatch() {
        let (open, high, low, close) = sample_ohlc();
        let params = [ParamKV {
            key: "period",
            value: ParamValue::Int(14),
        }];
        let call = compile_call("dx", Some("value"), &params, false).unwrap();
        let compiled = run_compiled_cpu(
            &call,
            IndicatorDataRef::Ohlc {
                open: &open,
                high: &high,
                low: &low,
                close: &close,
            },
            Kernel::Auto,
        )
        .unwrap();
        let combos = [IndicatorParamSet { params: &params }];
        let direct = compute_cpu_batch(IndicatorBatchRequest {
            indicator_id: "dx",
            output_id: Some("value"),
            data: IndicatorDataRef::Ohlc {
                open: &open,
                high: &high,
                low: &low,
                close: &close,
            },
            combos: &combos,
            kernel: Kernel::Auto,
        })
        .unwrap();
        assert_eq!(compiled.rows, direct.rows);
        assert_eq!(compiled.cols, direct.cols);
        let a = compiled.values_f64.unwrap();
        let b = direct.values_f64.unwrap();
        assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            let x = a[i];
            let y = b[i];
            if x.is_nan() && y.is_nan() {
                continue;
            }
            assert!((x - y).abs() <= 1e-12, "mismatch at index {i}: {x} vs {y}");
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn compile_prefer_cuda_rejects_non_cuda_indicator() {
        let err = compile_call("historical_volatility", Some("value"), &[], true).unwrap_err();
        match err {
            IndicatorDispatchError::UnsupportedCapability {
                indicator,
                capability,
            } => {
                assert_eq!(indicator, "historical_volatility");
                assert_eq!(capability, "cuda_batch");
            }
            other => panic!("expected UnsupportedCapability, got {other:?}"),
        }
    }
}
