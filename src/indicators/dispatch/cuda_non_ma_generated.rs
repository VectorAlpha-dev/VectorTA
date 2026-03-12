use super::{
    CudaOutputTarget, DeviceMatrixF32, IndicatorCudaDataRef, IndicatorCudaOutput,
    IndicatorCudaRequest, IndicatorCudaSeries, IndicatorDispatchError, ParamKV, ParamValue,
};
use crate::indicators::registry::{IndicatorInfo, IndicatorInputKind};
use cust::memory::CopyDestination;

pub(super) fn try_dispatch_non_ma_cuda(
    indicator: &str,
    info: Option<&IndicatorInfo>,
    req: IndicatorCudaRequest<'_>,
    device_id: usize,
) -> Option<Result<IndicatorCudaOutput, IndicatorDispatchError>> {
    match indicator {
        "acosc" => Some((|| {
            let indicator = "acosc";
            let fallback_outputs: &[&str] = &["osc", "change"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let high_f32 = required_high_f32(req.data, indicator)?;
            let low_f32 = required_low_f32(req.data, indicator)?;
            let mut cuda = crate::cuda::oscillators::CudaAcosc::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .acosc_batch_dev(high_f32.as_slice(), low_f32.as_slice())
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.osc.buf,
                    rows: result.osc.rows,
                    cols: result.osc.cols,
                },
                1 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.change.buf,
                    rows: result.change.rows,
                    cols: result.change.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "adosc" => Some((|| {
            let indicator = "adosc";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let high_f32 = required_high_f32(req.data, indicator)?;
            let low_f32 = required_low_f32(req.data, indicator)?;
            let close_f32 = required_close_f32(req.data, indicator)?;
            let volume_f32 = required_volume_f32(req.data, indicator)?;
            let mut sweep: crate::indicators::adosc::AdoscBatchRange = Default::default();
            sweep.short_period = resolve_usize_range_param(
                req.params,
                "short_period",
                sweep.short_period,
                indicator,
            )?;
            sweep.long_period =
                resolve_usize_range_param(req.params, "long_period", sweep.long_period, indicator)?;
            let mut cuda = crate::cuda::oscillators::CudaAdosc::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .adosc_batch_dev(
                    high_f32.as_slice(),
                    low_f32.as_slice(),
                    close_f32.as_slice(),
                    volume_f32.as_slice(),
                    &sweep,
                )
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.buf,
                    rows: result.rows,
                    cols: result.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "adx" => Some((|| {
            let indicator = "adx";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let high_f32 = required_high_f32(req.data, indicator)?;
            let low_f32 = required_low_f32(req.data, indicator)?;
            let close_f32 = required_close_f32(req.data, indicator)?;
            let mut sweep: crate::indicators::adx::AdxBatchRange = Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            let mut cuda = crate::cuda::CudaAdx::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .adx_batch_dev(
                    high_f32.as_slice(),
                    low_f32.as_slice(),
                    close_f32.as_slice(),
                    &sweep,
                )
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.buf,
                    rows: result.0.rows,
                    cols: result.0.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "adxr" => Some((|| {
            let indicator = "adxr";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let high_f32 = required_high_f32(req.data, indicator)?;
            let low_f32 = required_low_f32(req.data, indicator)?;
            let close_f32 = required_close_f32(req.data, indicator)?;
            let mut sweep: crate::indicators::adxr::AdxrBatchRange = Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            let mut cuda = crate::cuda::CudaAdxr::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .adxr_batch_dev(
                    high_f32.as_slice(),
                    low_f32.as_slice(),
                    close_f32.as_slice(),
                    &sweep,
                )
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.buf,
                    rows: result.0.rows,
                    cols: result.0.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "alligator" => Some((|| {
            let indicator = "alligator";
            let fallback_outputs: &[&str] = &["jaw", "teeth", "lips"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::alligator::AlligatorBatchRange = Default::default();
            sweep.jaw_period =
                resolve_usize_range_param(req.params, "jaw_period", sweep.jaw_period, indicator)?;
            sweep.jaw_offset =
                resolve_usize_range_param(req.params, "jaw_offset", sweep.jaw_offset, indicator)?;
            sweep.teeth_period = resolve_usize_range_param(
                req.params,
                "teeth_period",
                sweep.teeth_period,
                indicator,
            )?;
            sweep.teeth_offset = resolve_usize_range_param(
                req.params,
                "teeth_offset",
                sweep.teeth_offset,
                indicator,
            )?;
            sweep.lips_period =
                resolve_usize_range_param(req.params, "lips_period", sweep.lips_period, indicator)?;
            sweep.lips_offset =
                resolve_usize_range_param(req.params, "lips_offset", sweep.lips_offset, indicator)?;
            let mut cuda = crate::cuda::CudaAlligator::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .alligator_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.outputs.jaw.buf,
                    rows: result.outputs.jaw.rows,
                    cols: result.outputs.jaw.cols,
                },
                1 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.outputs.teeth.buf,
                    rows: result.outputs.teeth.rows,
                    cols: result.outputs.teeth.cols,
                },
                2 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.outputs.lips.buf,
                    rows: result.outputs.lips.rows,
                    cols: result.outputs.lips.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "alphatrend" => Some((|| {
            let indicator = "alphatrend";
            let fallback_outputs: &[&str] = &["k1", "k2"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let high_f32 = required_high_f32(req.data, indicator)?;
            let low_f32 = required_low_f32(req.data, indicator)?;
            let close_f32 = required_close_f32(req.data, indicator)?;
            let volume_f32 = required_volume_f32(req.data, indicator)?;
            let mut sweep: crate::indicators::alphatrend::AlphaTrendBatchRange = Default::default();
            sweep.coeff = resolve_f64_range_param(req.params, "coeff", sweep.coeff, indicator)?;
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            if let Some(v) = get_bool_param(req.params, "no_volume", indicator)? {
                sweep.no_volume = v;
            }
            let mut cuda = crate::cuda::CudaAlphaTrend::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .alphatrend_batch_dev(
                    high_f32.as_slice(),
                    low_f32.as_slice(),
                    close_f32.as_slice(),
                    volume_f32.as_slice(),
                    &sweep,
                )
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.k1.buf,
                    rows: result.k1.rows,
                    cols: result.k1.cols,
                },
                1 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.k2.buf,
                    rows: result.k2.rows,
                    cols: result.k2.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "ao" => Some((|| {
            let indicator = "ao";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::ao::AoBatchRange = Default::default();
            sweep.short_period = resolve_usize_range_param(
                req.params,
                "short_period",
                sweep.short_period,
                indicator,
            )?;
            sweep.long_period =
                resolve_usize_range_param(req.params, "long_period", sweep.long_period, indicator)?;
            let mut cuda = crate::cuda::oscillators::CudaAo::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .ao_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.buf,
                    rows: result.rows,
                    cols: result.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "apo" => Some((|| {
            let indicator = "apo";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::apo::ApoBatchRange = Default::default();
            sweep.short = resolve_usize_range_param(req.params, "short", sweep.short, indicator)?;
            sweep.long = resolve_usize_range_param(req.params, "long", sweep.long, indicator)?;
            let mut cuda = crate::cuda::moving_averages::CudaApo::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .apo_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.buf,
                    rows: result.rows,
                    cols: result.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "aroon" => Some((|| {
            let indicator = "aroon";
            let fallback_outputs: &[&str] = &["first", "second"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let high_f32 = required_high_f32(req.data, indicator)?;
            let low_f32 = required_low_f32(req.data, indicator)?;
            let mut sweep: crate::indicators::aroon::AroonBatchRange = Default::default();
            sweep.length =
                resolve_usize_range_param(req.params, "length", sweep.length, indicator)?;
            let mut cuda = crate::cuda::CudaAroon::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .aroon_batch_dev(high_f32.as_slice(), low_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.outputs.first.buf,
                    rows: result.outputs.first.rows,
                    cols: result.outputs.first.cols,
                },
                1 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.outputs.second.buf,
                    rows: result.outputs.second.rows,
                    cols: result.outputs.second.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "aroonosc" => Some((|| {
            let indicator = "aroonosc";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let high_f32 = required_high_f32(req.data, indicator)?;
            let low_f32 = required_low_f32(req.data, indicator)?;
            let mut sweep: crate::indicators::aroonosc::AroonOscBatchRange = Default::default();
            sweep.length =
                resolve_usize_range_param(req.params, "length", sweep.length, indicator)?;
            let mut cuda = crate::cuda::oscillators::CudaAroonOsc::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .aroonosc_batch_dev(high_f32.as_slice(), low_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.buf,
                    rows: result.rows,
                    cols: result.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "aso" => Some((|| {
            let indicator = "aso";
            let fallback_outputs: &[&str] = &["output_0", "output_1"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let open_f32 = required_open_f32(req.data, indicator)?;
            let high_f32 = required_high_f32(req.data, indicator)?;
            let low_f32 = required_low_f32(req.data, indicator)?;
            let close_f32 = required_close_f32(req.data, indicator)?;
            let mut sweep: crate::indicators::aso::AsoBatchRange = Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            sweep.mode = resolve_usize_range_param(req.params, "mode", sweep.mode, indicator)?;
            let mut cuda = crate::cuda::oscillators::CudaAso::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .aso_batch_dev(
                    open_f32.as_slice(),
                    high_f32.as_slice(),
                    low_f32.as_slice(),
                    close_f32.as_slice(),
                    &sweep,
                )
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.buf,
                    rows: result.0.rows,
                    cols: result.0.cols,
                },
                1 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.1.buf,
                    rows: result.1.rows,
                    cols: result.1.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "atr" => Some((|| {
            let indicator = "atr";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let high_f32 = required_high_f32(req.data, indicator)?;
            let low_f32 = required_low_f32(req.data, indicator)?;
            let close_f32 = required_close_f32(req.data, indicator)?;
            let mut sweep: crate::indicators::atr::AtrBatchRange = Default::default();
            sweep.length =
                resolve_usize_range_param(req.params, "length", sweep.length, indicator)?;
            let mut cuda = crate::cuda::CudaAtr::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .atr_batch_dev(
                    high_f32.as_slice(),
                    low_f32.as_slice(),
                    close_f32.as_slice(),
                    &sweep,
                )
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.buf,
                    rows: result.rows,
                    cols: result.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "avsl" => Some((|| {
            let indicator = "avsl";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let low_f32 = required_low_f32(req.data, indicator)?;
            let close_f32 = required_close_f32(req.data, indicator)?;
            let volume_f32 = required_volume_f32(req.data, indicator)?;
            let mut sweep: crate::indicators::avsl::AvslBatchRange = Default::default();
            sweep.fast_period =
                resolve_usize_range_param(req.params, "fast_period", sweep.fast_period, indicator)?;
            sweep.slow_period =
                resolve_usize_range_param(req.params, "slow_period", sweep.slow_period, indicator)?;
            sweep.multiplier =
                resolve_f64_range_param(req.params, "multiplier", sweep.multiplier, indicator)?;
            let mut cuda = crate::cuda::CudaAvsl::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .avsl_batch_dev(
                    close_f32.as_slice(),
                    low_f32.as_slice(),
                    volume_f32.as_slice(),
                    &sweep,
                )
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.buf,
                    rows: result.0.rows,
                    cols: result.0.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "bandpass" => Some((|| {
            let indicator = "bandpass";
            let fallback_outputs: &[&str] = &["first", "second", "third", "fourth"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::bandpass::BandPassBatchRange = Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            sweep.bandwidth =
                resolve_f64_range_param(req.params, "bandwidth", sweep.bandwidth, indicator)?;
            let mut cuda = crate::cuda::CudaBandpass::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .bandpass_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.outputs.first.buf,
                    rows: result.outputs.first.rows,
                    cols: result.outputs.first.cols,
                },
                1 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.outputs.second.buf,
                    rows: result.outputs.second.rows,
                    cols: result.outputs.second.cols,
                },
                2 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.outputs.third.buf,
                    rows: result.outputs.third.rows,
                    cols: result.outputs.third.cols,
                },
                3 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.outputs.fourth.buf,
                    rows: result.outputs.fourth.rows,
                    cols: result.outputs.fourth.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "bollinger_bands" => Some((|| {
            let indicator = "bollinger_bands";
            let fallback_outputs: &[&str] = &["output_0", "output_1", "output_2"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::bollinger_bands::BollingerBandsBatchRange =
                Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            sweep.devup = resolve_f64_range_param(req.params, "devup", sweep.devup, indicator)?;
            sweep.devdn = resolve_f64_range_param(req.params, "devdn", sweep.devdn, indicator)?;
            sweep.matype =
                resolve_string_range_param(req.params, "matype", sweep.matype, indicator)?;
            sweep.devtype =
                resolve_usize_range_param(req.params, "devtype", sweep.devtype, indicator)?;
            let mut cuda = crate::cuda::CudaBollingerBands::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .bollinger_bands_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.buf,
                    rows: result.0.rows,
                    cols: result.0.cols,
                },
                1 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.1.buf,
                    rows: result.1.rows,
                    cols: result.1.cols,
                },
                2 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.2.buf,
                    rows: result.2.rows,
                    cols: result.2.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "bollinger_bands_width" => Some((|| {
            let indicator = "bollinger_bands_width";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::bollinger_bands_width::BollingerBandsWidthBatchRange =
                Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            sweep.devup = resolve_f64_range_param(req.params, "devup", sweep.devup, indicator)?;
            sweep.devdn = resolve_f64_range_param(req.params, "devdn", sweep.devdn, indicator)?;
            let mut cuda = crate::cuda::CudaBbw::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .bbw_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.buf,
                    rows: result.0.rows,
                    cols: result.0.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "bop" => Some((|| {
            let indicator = "bop";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let open_f32 = required_open_f32(req.data, indicator)?;
            let high_f32 = required_high_f32(req.data, indicator)?;
            let low_f32 = required_low_f32(req.data, indicator)?;
            let close_f32 = required_close_f32(req.data, indicator)?;
            let mut cuda = crate::cuda::oscillators::CudaBop::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .bop_batch_dev(
                    open_f32.as_slice(),
                    high_f32.as_slice(),
                    low_f32.as_slice(),
                    close_f32.as_slice(),
                )
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.buf,
                    rows: result.rows,
                    cols: result.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "cci" => Some((|| {
            let indicator = "cci";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::cci::CciBatchRange = Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            let mut cuda = crate::cuda::oscillators::CudaCci::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .cci_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.buf,
                    rows: result.rows,
                    cols: result.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "cci_cycle" => Some((|| {
            let indicator = "cci_cycle";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::cci_cycle::CciCycleBatchRange = Default::default();
            sweep.length =
                resolve_usize_range_param(req.params, "length", sweep.length, indicator)?;
            sweep.factor = resolve_f64_range_param(req.params, "factor", sweep.factor, indicator)?;
            let mut cuda = crate::cuda::oscillators::CudaCciCycle::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .cci_cycle_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.buf,
                    rows: result.rows,
                    cols: result.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "cfo" => Some((|| {
            let indicator = "cfo";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::cfo::CfoBatchRange = Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            sweep.scalar = resolve_f64_range_param(req.params, "scalar", sweep.scalar, indicator)?;
            let mut cuda = crate::cuda::oscillators::CudaCfo::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .cfo_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.buf,
                    rows: result.rows,
                    cols: result.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "cg" => Some((|| {
            let indicator = "cg";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::cg::CgBatchRange = Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            let mut cuda = crate::cuda::oscillators::CudaCg::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .cg_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.buf,
                    rows: result.rows,
                    cols: result.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "chande" => Some((|| {
            let indicator = "chande";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let high_f32 = required_high_f32(req.data, indicator)?;
            let low_f32 = required_low_f32(req.data, indicator)?;
            let close_f32 = required_close_f32(req.data, indicator)?;
            let direction = get_string_param(req.params, "direction")
                .unwrap_or("long")
                .to_string();
            let mut sweep: crate::indicators::chande::ChandeBatchRange = Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            sweep.mult = resolve_f64_range_param(req.params, "mult", sweep.mult, indicator)?;
            let mut cuda = crate::cuda::CudaChande::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .chande_batch_dev(
                    high_f32.as_slice(),
                    low_f32.as_slice(),
                    close_f32.as_slice(),
                    &sweep,
                    direction.as_str(),
                )
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.buf,
                    rows: result.rows,
                    cols: result.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "chandelier_exit" => Some((|| {
            let indicator = "chandelier_exit";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let high_f32 = required_high_f32(req.data, indicator)?;
            let low_f32 = required_low_f32(req.data, indicator)?;
            let close_f32 = required_close_f32(req.data, indicator)?;
            let mut sweep: crate::indicators::chandelier_exit::CeBatchRange = Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            sweep.mult = resolve_f64_range_param(req.params, "mult", sweep.mult, indicator)?;
            sweep.use_close =
                resolve_bool_range_param(req.params, "use_close", sweep.use_close, indicator)?;
            let mut cuda = crate::cuda::CudaChandelierExit::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .chandelier_exit_batch_dev(
                    high_f32.as_slice(),
                    low_f32.as_slice(),
                    close_f32.as_slice(),
                    &sweep,
                )
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.buf,
                    rows: result.0.rows,
                    cols: result.0.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "chop" => Some((|| {
            let indicator = "chop";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let high_f32 = required_high_f32(req.data, indicator)?;
            let low_f32 = required_low_f32(req.data, indicator)?;
            let close_f32 = required_close_f32(req.data, indicator)?;
            let mut sweep: crate::indicators::chop::ChopBatchRange = Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            sweep.scalar = resolve_f64_range_param(req.params, "scalar", sweep.scalar, indicator)?;
            sweep.drift = resolve_usize_range_param(req.params, "drift", sweep.drift, indicator)?;
            let mut cuda = crate::cuda::oscillators::CudaChop::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .chop_batch_dev(
                    high_f32.as_slice(),
                    low_f32.as_slice(),
                    close_f32.as_slice(),
                    &sweep,
                )
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.buf,
                    rows: result.0.rows,
                    cols: result.0.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "cksp" => Some((|| {
            let indicator = "cksp";
            let fallback_outputs: &[&str] = &["long", "short"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let high_f32 = required_high_f32(req.data, indicator)?;
            let low_f32 = required_low_f32(req.data, indicator)?;
            let close_f32 = required_close_f32(req.data, indicator)?;
            let mut sweep: crate::indicators::cksp::CkspBatchRange = Default::default();
            sweep.p = resolve_usize_range_param(req.params, "p", sweep.p, indicator)?;
            sweep.x = resolve_f64_range_param(req.params, "x", sweep.x, indicator)?;
            sweep.q = resolve_usize_range_param(req.params, "q", sweep.q, indicator)?;
            let mut cuda = crate::cuda::CudaCksp::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .cksp_batch_dev(
                    high_f32.as_slice(),
                    low_f32.as_slice(),
                    close_f32.as_slice(),
                    &sweep,
                )
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.long.buf,
                    rows: result.0.long.rows,
                    cols: result.0.long.cols,
                },
                1 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.short.buf,
                    rows: result.0.short.rows,
                    cols: result.0.short.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "cmo" => Some((|| {
            let indicator = "cmo";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::cmo::CmoBatchRange = Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            let mut cuda = crate::cuda::oscillators::CudaCmo::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .cmo_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.buf,
                    rows: result.rows,
                    cols: result.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "coppock" => Some((|| {
            let indicator = "coppock";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::coppock::CoppockBatchRange = Default::default();
            sweep.short = resolve_usize_range_param(req.params, "short", sweep.short, indicator)?;
            sweep.long = resolve_usize_range_param(req.params, "long", sweep.long, indicator)?;
            sweep.ma = resolve_usize_range_param(req.params, "ma", sweep.ma, indicator)?;
            let mut cuda = crate::cuda::oscillators::CudaCoppock::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .coppock_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.buf,
                    rows: result.rows,
                    cols: result.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "correl_hl" => Some((|| {
            let indicator = "correl_hl";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let high_f32 = required_high_f32(req.data, indicator)?;
            let low_f32 = required_low_f32(req.data, indicator)?;
            let mut sweep: crate::indicators::correl_hl::CorrelHlBatchRange = Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            let mut cuda = crate::cuda::CudaCorrelHl::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .correl_hl_batch_dev(high_f32.as_slice(), low_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.buf,
                    rows: result.0.rows,
                    cols: result.0.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "parkinson_volatility" => Some((|| {
            let indicator = "parkinson_volatility";
            let fallback_outputs: &[&str] = &["volatility", "variance"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let high_f32 = required_high_f32(req.data, indicator)?;
            let low_f32 = required_low_f32(req.data, indicator)?;
            let mut sweep: crate::indicators::parkinson_volatility::ParkinsonVolatilityBatchRange =
                Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            let mut cuda = crate::cuda::CudaParkinsonVolatility::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .parkinson_volatility_batch_dev(high_f32.as_slice(), low_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.outputs.volatility.buf,
                    rows: result.outputs.volatility.rows,
                    cols: result.outputs.volatility.cols,
                },
                1 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.outputs.variance.buf,
                    rows: result.outputs.variance.rows,
                    cols: result.outputs.variance.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "correlation_cycle" => Some((|| {
            let indicator = "correlation_cycle";
            let fallback_outputs: &[&str] = &["real", "imag", "angle", "state"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::correlation_cycle::CorrelationCycleBatchRange =
                Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            sweep.threshold =
                resolve_f64_range_param(req.params, "threshold", sweep.threshold, indicator)?;
            let mut cuda = crate::cuda::moving_averages::CudaCorrelationCycle::new(device_id)
                .map_err(|e| IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                })?;
            let result = cuda
                .correlation_cycle_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.real.buf,
                    rows: result.real.rows,
                    cols: result.real.cols,
                },
                1 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.imag.buf,
                    rows: result.imag.rows,
                    cols: result.imag.cols,
                },
                2 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.angle.buf,
                    rows: result.angle.rows,
                    cols: result.angle.cols,
                },
                3 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.state.buf,
                    rows: result.state.rows,
                    cols: result.state.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "cvi" => Some((|| {
            let indicator = "cvi";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let high_f32 = required_high_f32(req.data, indicator)?;
            let low_f32 = required_low_f32(req.data, indicator)?;
            let mut sweep: crate::indicators::cvi::CviBatchRange = Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            let mut cuda = crate::cuda::CudaCvi::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .cvi_batch_dev(high_f32.as_slice(), low_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.buf,
                    rows: result.rows,
                    cols: result.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "damiani_volatmeter" => Some((|| {
            let indicator = "damiani_volatmeter";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::damiani_volatmeter::DamianiVolatmeterBatchRange =
                Default::default();
            sweep.vis_atr =
                resolve_usize_range_param(req.params, "vis_atr", sweep.vis_atr, indicator)?;
            sweep.vis_std =
                resolve_usize_range_param(req.params, "vis_std", sweep.vis_std, indicator)?;
            sweep.sed_atr =
                resolve_usize_range_param(req.params, "sed_atr", sweep.sed_atr, indicator)?;
            sweep.sed_std =
                resolve_usize_range_param(req.params, "sed_std", sweep.sed_std, indicator)?;
            sweep.threshold =
                resolve_f64_range_param(req.params, "threshold", sweep.threshold, indicator)?;
            let mut cuda = crate::cuda::CudaDamianiVolatmeter::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .damiani_volatmeter_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.buf,
                    rows: result.0.rows,
                    cols: result.0.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "dec_osc" => Some((|| {
            let indicator = "dec_osc";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::dec_osc::DecOscBatchRange = Default::default();
            sweep.hp_period =
                resolve_usize_range_param(req.params, "hp_period", sweep.hp_period, indicator)?;
            sweep.k = resolve_f64_range_param(req.params, "k", sweep.k, indicator)?;
            let mut cuda = crate::cuda::oscillators::CudaDecOsc::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .dec_osc_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.buf,
                    rows: result.rows,
                    cols: result.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "decycler" => Some((|| {
            let indicator = "decycler";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let first_valid = primary_f32
                .as_slice()
                .iter()
                .position(|value| value.is_finite())
                .ok_or_else(|| IndicatorDispatchError::InvalidParam {
                    indicator: indicator.to_string(),
                    key: "data".to_string(),
                    reason: "all values are NaN".to_string(),
                })?;
            let mut sweep: crate::indicators::decycler::DecyclerBatchRange = Default::default();
            sweep.hp_period =
                resolve_usize_range_param(req.params, "hp_period", sweep.hp_period, indicator)?;
            sweep.k = resolve_f64_range_param(req.params, "k", sweep.k, indicator)?;
            let mut cuda =
                crate::cuda::moving_averages::CudaDecycler::new(device_id).map_err(|e| {
                    IndicatorDispatchError::KernelUnavailable {
                        details: e.to_string(),
                    }
                })?;
            let result = cuda
                .decycler_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.buf,
                    rows: result.rows,
                    cols: result.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(
                output_id,
                owner,
                req.target,
                device_id as u32,
                Some(first_valid + 2),
            )
        })()),
        "deviation" => Some((|| {
            let indicator = "deviation";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::deviation::DeviationBatchRange = Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            sweep.devtype =
                resolve_usize_range_param(req.params, "devtype", sweep.devtype, indicator)?;
            let mut cuda = crate::cuda::CudaDeviation::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .deviation_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.buf,
                    rows: result.0.rows,
                    cols: result.0.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "devstop" => Some((|| {
            let indicator = "devstop";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let high_f32 = required_high_f32(req.data, indicator)?;
            let low_f32 = required_low_f32(req.data, indicator)?;
            let is_long = resolve_is_long(req.params, indicator)?;
            let mut sweep: crate::indicators::devstop::DevStopBatchRange = Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            sweep.mult = resolve_f64_range_param(req.params, "mult", sweep.mult, indicator)?;
            sweep.devtype =
                resolve_usize_range_param(req.params, "devtype", sweep.devtype, indicator)?;
            let mut cuda = crate::cuda::CudaDevStop::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .devstop_batch_dev(high_f32.as_slice(), low_f32.as_slice(), &sweep, is_long)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.buf,
                    rows: result.0.rows,
                    cols: result.0.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "di" => Some((|| {
            let indicator = "di";
            let fallback_outputs: &[&str] = &["output_0", "output_1"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let high_f32 = required_high_f32(req.data, indicator)?;
            let low_f32 = required_low_f32(req.data, indicator)?;
            let close_f32 = required_close_f32(req.data, indicator)?;
            let mut sweep: crate::indicators::di::DiBatchRange = Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            let mut cuda = crate::cuda::CudaDi::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .di_batch_dev(
                    high_f32.as_slice(),
                    low_f32.as_slice(),
                    close_f32.as_slice(),
                    &sweep,
                )
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.buf,
                    rows: result.0.rows,
                    cols: result.0.cols,
                },
                1 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.1.buf,
                    rows: result.1.rows,
                    cols: result.1.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "dm" => Some((|| {
            let indicator = "dm";
            let fallback_outputs: &[&str] = &["plus", "minus"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let high_f32 = required_high_f32(req.data, indicator)?;
            let low_f32 = required_low_f32(req.data, indicator)?;
            let mut sweep: crate::indicators::dm::DmBatchRange = Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            let mut cuda = crate::cuda::CudaDm::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .dm_batch_dev(high_f32.as_slice(), low_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.plus.buf,
                    rows: result.0.plus.rows,
                    cols: result.0.plus.cols,
                },
                1 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.minus.buf,
                    rows: result.0.minus.rows,
                    cols: result.0.minus.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "donchian" => Some((|| {
            let indicator = "donchian";
            let fallback_outputs: &[&str] = &["wt1", "wt2", "hist"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let high_f32 = required_high_f32(req.data, indicator)?;
            let low_f32 = required_low_f32(req.data, indicator)?;
            let mut sweep: crate::indicators::donchian::DonchianBatchRange = Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            let mut cuda = crate::cuda::CudaDonchian::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .donchian_batch_dev(high_f32.as_slice(), low_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.wt1.buf,
                    rows: result.0.wt1.rows,
                    cols: result.0.wt1.cols,
                },
                1 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.wt2.buf,
                    rows: result.0.wt2.rows,
                    cols: result.0.wt2.cols,
                },
                2 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.hist.buf,
                    rows: result.0.hist.rows,
                    cols: result.0.hist.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "dpo" => Some((|| {
            let indicator = "dpo";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::dpo::DpoBatchRange = Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            let mut cuda = crate::cuda::oscillators::CudaDpo::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .dpo_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.buf,
                    rows: result.rows,
                    cols: result.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "dti" => Some((|| {
            let indicator = "dti";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let high_f32 = required_high_f32(req.data, indicator)?;
            let low_f32 = required_low_f32(req.data, indicator)?;
            let mut sweep: crate::indicators::dti::DtiBatchRange = Default::default();
            sweep.r = resolve_usize_range_param(req.params, "r", sweep.r, indicator)?;
            sweep.s = resolve_usize_range_param(req.params, "s", sweep.s, indicator)?;
            sweep.u = resolve_usize_range_param(req.params, "u", sweep.u, indicator)?;
            let mut cuda = crate::cuda::oscillators::CudaDti::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .dti_batch_dev(high_f32.as_slice(), low_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.buf,
                    rows: result.0.rows,
                    cols: result.0.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "dvdiqqe" => Some((|| {
            let indicator = "dvdiqqe";
            let fallback_outputs: &[&str] = &["dvdi", "fast", "slow", "center"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let open_f32 = required_open_f32(req.data, indicator)?;
            let close_f32 = required_close_f32(req.data, indicator)?;
            let volume_f32_opt = optional_volume_f32(req.data);
            let volume_type = get_string_param(req.params, "volume_type")
                .unwrap_or("default")
                .to_string();
            let center_type = get_string_param(req.params, "center_type")
                .unwrap_or("dynamic")
                .to_string();
            let tick_size = get_f32_param(req.params, "tick_size", indicator)?.unwrap_or(0.01f32);
            let mut sweep: crate::indicators::dvdiqqe::DvdiqqeBatchRange = Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            sweep.smoothing_period = resolve_usize_range_param(
                req.params,
                "smoothing_period",
                sweep.smoothing_period,
                indicator,
            )?;
            sweep.fast_multiplier = resolve_f64_range_param(
                req.params,
                "fast_multiplier",
                sweep.fast_multiplier,
                indicator,
            )?;
            sweep.slow_multiplier = resolve_f64_range_param(
                req.params,
                "slow_multiplier",
                sweep.slow_multiplier,
                indicator,
            )?;
            let mut cuda = crate::cuda::CudaDvdiqqe::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .dvdiqqe_batch_dev(
                    open_f32.as_slice(),
                    close_f32.as_slice(),
                    volume_f32_opt.as_ref().map(F32Input::as_slice),
                    &sweep,
                    volume_type.as_str(),
                    center_type.as_str(),
                    tick_size,
                )
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.dvdi.buf,
                    rows: result.dvdi.rows,
                    cols: result.dvdi.cols,
                },
                1 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.fast.buf,
                    rows: result.fast.rows,
                    cols: result.fast.cols,
                },
                2 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.slow.buf,
                    rows: result.slow.rows,
                    cols: result.slow.cols,
                },
                3 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.center.buf,
                    rows: result.center.rows,
                    cols: result.center.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "dx" => Some((|| {
            let indicator = "dx";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let high_f32 = required_high_f32(req.data, indicator)?;
            let low_f32 = required_low_f32(req.data, indicator)?;
            let close_f32 = required_close_f32(req.data, indicator)?;
            let mut sweep: crate::indicators::dx::DxBatchRange = Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            let mut cuda = crate::cuda::CudaDx::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .dx_batch_dev(
                    high_f32.as_slice(),
                    low_f32.as_slice(),
                    close_f32.as_slice(),
                    &sweep,
                )
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.buf,
                    rows: result.0.rows,
                    cols: result.0.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "efi" => Some((|| {
            let indicator = "efi";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let volume_f32 = required_volume_f32(req.data, indicator)?;
            let mut sweep: crate::indicators::efi::EfiBatchRange = Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            let mut cuda = crate::cuda::CudaEfi::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .efi_batch_dev(primary_f32.as_slice(), volume_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.buf,
                    rows: result.rows,
                    cols: result.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "emd" => Some((|| {
            let indicator = "emd";
            let fallback_outputs: &[&str] = &["upper", "middle", "lower"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let high_f32 = required_high_f32(req.data, indicator)?;
            let low_f32 = required_low_f32(req.data, indicator)?;
            let mut sweep: crate::indicators::emd::EmdBatchRange = Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            sweep.delta = resolve_f64_range_param(req.params, "delta", sweep.delta, indicator)?;
            sweep.fraction =
                resolve_f64_range_param(req.params, "fraction", sweep.fraction, indicator)?;
            let mut cuda = crate::cuda::CudaEmd::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .emd_batch_dev(high_f32.as_slice(), low_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.outputs.upper.buf,
                    rows: result.outputs.upper.rows,
                    cols: result.outputs.upper.cols,
                },
                1 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.outputs.middle.buf,
                    rows: result.outputs.middle.rows,
                    cols: result.outputs.middle.cols,
                },
                2 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.outputs.lower.buf,
                    rows: result.outputs.lower.rows,
                    cols: result.outputs.lower.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "emv" => Some((|| {
            let indicator = "emv";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let high_f32 = required_high_f32(req.data, indicator)?;
            let low_f32 = required_low_f32(req.data, indicator)?;
            let volume_f32 = required_volume_f32(req.data, indicator)?;
            let mut cuda = crate::cuda::oscillators::CudaEmv::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .emv_batch_dev(
                    high_f32.as_slice(),
                    low_f32.as_slice(),
                    volume_f32.as_slice(),
                )
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.buf,
                    rows: result.rows,
                    cols: result.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "er" => Some((|| {
            let indicator = "er";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::er::ErBatchRange = Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            let mut cuda = crate::cuda::CudaEr::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .er_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.buf,
                    rows: result.rows,
                    cols: result.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "eri" => Some((|| {
            let indicator = "eri";
            let fallback_outputs: &[&str] = &["output_0", "output_1"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let high_f32 = required_high_f32(req.data, indicator)?;
            let low_f32 = required_low_f32(req.data, indicator)?;
            let mut sweep: crate::indicators::eri::EriBatchRange = Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            let mut cuda = crate::cuda::CudaEri::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .eri_batch_dev(
                    high_f32.as_slice(),
                    low_f32.as_slice(),
                    primary_f32.as_slice(),
                    &sweep,
                )
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0 .0.buf,
                    rows: result.0 .0.rows,
                    cols: result.0 .0.cols,
                },
                1 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0 .1.buf,
                    rows: result.0 .1.rows,
                    cols: result.0 .1.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "fisher" => Some((|| {
            let indicator = "fisher";
            let fallback_outputs: &[&str] = &["fisher", "signal"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let high_f32 = required_high_f32(req.data, indicator)?;
            let low_f32 = required_low_f32(req.data, indicator)?;
            let mut sweep: crate::indicators::fisher::FisherBatchRange = Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            let mut cuda = crate::cuda::oscillators::CudaFisher::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .fisher_batch_dev(high_f32.as_slice(), low_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.fisher.buf,
                    rows: result.0.fisher.rows,
                    cols: result.0.fisher.cols,
                },
                1 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.signal.buf,
                    rows: result.0.signal.rows,
                    cols: result.0.signal.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "fosc" => Some((|| {
            let indicator = "fosc";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::fosc::FoscBatchRange = Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            let mut cuda = crate::cuda::oscillators::CudaFosc::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .fosc_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.buf,
                    rows: result.rows,
                    cols: result.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "fvg_trailing_stop" => Some((|| {
            let indicator = "fvg_trailing_stop";
            let fallback_outputs: &[&str] = &["upper", "lower", "upper_ts", "lower_ts"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let high_f32 = required_high_f32(req.data, indicator)?;
            let low_f32 = required_low_f32(req.data, indicator)?;
            let close_f32 = required_close_f32(req.data, indicator)?;
            let mut sweep: crate::indicators::fvg_trailing_stop::FvgTsBatchRange =
                Default::default();
            sweep.lookback =
                resolve_usize_range_param(req.params, "lookback", sweep.lookback, indicator)?;
            sweep.smoothing =
                resolve_usize_range_param(req.params, "smoothing", sweep.smoothing, indicator)?;
            let mut cuda = crate::cuda::CudaFvgTs::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .fvg_ts_batch_dev(
                    high_f32.as_slice(),
                    low_f32.as_slice(),
                    close_f32.as_slice(),
                    &sweep,
                )
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.upper.buf,
                    rows: result.upper.rows,
                    cols: result.upper.cols,
                },
                1 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.lower.buf,
                    rows: result.lower.rows,
                    cols: result.lower.cols,
                },
                2 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.upper_ts.buf,
                    rows: result.upper_ts.rows,
                    cols: result.upper_ts.cols,
                },
                3 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.lower_ts.buf,
                    rows: result.lower_ts.rows,
                    cols: result.lower_ts.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "gatorosc" => Some((|| {
            let indicator = "gatorosc";
            let fallback_outputs: &[&str] = &["upper", "lower", "upper_change", "lower_change"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::gatorosc::GatorOscBatchRange = Default::default();
            sweep.jaws_length =
                resolve_usize_range_param(req.params, "jaws_length", sweep.jaws_length, indicator)?;
            sweep.jaws_shift =
                resolve_usize_range_param(req.params, "jaws_shift", sweep.jaws_shift, indicator)?;
            sweep.teeth_length = resolve_usize_range_param(
                req.params,
                "teeth_length",
                sweep.teeth_length,
                indicator,
            )?;
            sweep.teeth_shift =
                resolve_usize_range_param(req.params, "teeth_shift", sweep.teeth_shift, indicator)?;
            sweep.lips_length =
                resolve_usize_range_param(req.params, "lips_length", sweep.lips_length, indicator)?;
            sweep.lips_shift =
                resolve_usize_range_param(req.params, "lips_shift", sweep.lips_shift, indicator)?;
            let mut cuda = crate::cuda::oscillators::CudaGatorOsc::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .gatorosc_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.upper.buf,
                    rows: result.upper.rows,
                    cols: result.upper.cols,
                },
                1 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.lower.buf,
                    rows: result.lower.rows,
                    cols: result.lower.cols,
                },
                2 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.upper_change.buf,
                    rows: result.upper_change.rows,
                    cols: result.upper_change.cols,
                },
                3 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.lower_change.buf,
                    rows: result.lower_change.rows,
                    cols: result.lower_change.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "halftrend" => Some((|| {
            let indicator = "halftrend";
            let fallback_outputs: &[&str] =
                &["halftrend", "trend", "atr_high", "atr_low", "buy", "sell"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let high_f32 = required_high_f32(req.data, indicator)?;
            let low_f32 = required_low_f32(req.data, indicator)?;
            let close_f32 = required_close_f32(req.data, indicator)?;
            let mut sweep: crate::indicators::halftrend::HalfTrendBatchRange = Default::default();
            sweep.amplitude =
                resolve_usize_range_param(req.params, "amplitude", sweep.amplitude, indicator)?;
            sweep.channel_deviation = resolve_f64_range_param(
                req.params,
                "channel_deviation",
                sweep.channel_deviation,
                indicator,
            )?;
            sweep.atr_period =
                resolve_usize_range_param(req.params, "atr_period", sweep.atr_period, indicator)?;
            let mut cuda = crate::cuda::CudaHalftrend::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .halftrend_batch_dev(
                    high_f32.as_slice(),
                    low_f32.as_slice(),
                    close_f32.as_slice(),
                    &sweep,
                )
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.halftrend.buf,
                    rows: result.halftrend.rows,
                    cols: result.halftrend.cols,
                },
                1 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.trend.buf,
                    rows: result.trend.rows,
                    cols: result.trend.cols,
                },
                2 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.atr_high.buf,
                    rows: result.atr_high.rows,
                    cols: result.atr_high.cols,
                },
                3 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.atr_low.buf,
                    rows: result.atr_low.rows,
                    cols: result.atr_low.cols,
                },
                4 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.buy.buf,
                    rows: result.buy.rows,
                    cols: result.buy.cols,
                },
                5 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.sell.buf,
                    rows: result.sell.rows,
                    cols: result.sell.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "ift_rsi" => Some((|| {
            let indicator = "ift_rsi";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::ift_rsi::IftRsiBatchRange = Default::default();
            sweep.rsi_period =
                resolve_usize_range_param(req.params, "rsi_period", sweep.rsi_period, indicator)?;
            sweep.wma_period =
                resolve_usize_range_param(req.params, "wma_period", sweep.wma_period, indicator)?;
            let mut cuda = crate::cuda::oscillators::CudaIftRsi::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .ift_rsi_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.buf,
                    rows: result.0.rows,
                    cols: result.0.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "kaufmanstop" => Some((|| {
            let indicator = "kaufmanstop";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let high_f32 = required_high_f32(req.data, indicator)?;
            let low_f32 = required_low_f32(req.data, indicator)?;
            let mut sweep: crate::indicators::kaufmanstop::KaufmanstopBatchRange =
                Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            sweep.mult = resolve_f64_range_param(req.params, "mult", sweep.mult, indicator)?;
            let mut cuda = crate::cuda::CudaKaufmanstop::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .kaufmanstop_batch_dev(high_f32.as_slice(), low_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.buf,
                    rows: result.0.rows,
                    cols: result.0.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "kdj" => Some((|| {
            let indicator = "kdj";
            let fallback_outputs: &[&str] = &["output_0", "output_1", "output_2"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let high_f32 = required_high_f32(req.data, indicator)?;
            let low_f32 = required_low_f32(req.data, indicator)?;
            let close_f32 = required_close_f32(req.data, indicator)?;
            let mut sweep: crate::indicators::kdj::KdjBatchRange = Default::default();
            sweep.fast_k_period = resolve_usize_range_param(
                req.params,
                "fast_k_period",
                sweep.fast_k_period,
                indicator,
            )?;
            sweep.slow_k_period = resolve_usize_range_param(
                req.params,
                "slow_k_period",
                sweep.slow_k_period,
                indicator,
            )?;
            sweep.slow_d_period = resolve_usize_range_param(
                req.params,
                "slow_d_period",
                sweep.slow_d_period,
                indicator,
            )?;
            let mut cuda = crate::cuda::oscillators::CudaKdj::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .kdj_batch_dev(
                    high_f32.as_slice(),
                    low_f32.as_slice(),
                    close_f32.as_slice(),
                    &sweep,
                )
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.buf,
                    rows: result.0.rows,
                    cols: result.0.cols,
                },
                1 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.1.buf,
                    rows: result.1.rows,
                    cols: result.1.cols,
                },
                2 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.2.buf,
                    rows: result.2.rows,
                    cols: result.2.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "keltner" => Some((|| {
            let indicator = "keltner";
            let fallback_outputs: &[&str] = &["upper", "middle", "lower"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let high_f32 = required_high_f32(req.data, indicator)?;
            let low_f32 = required_low_f32(req.data, indicator)?;
            let close_f32 = if primary_source_is_close(req.data) {
                None
            } else {
                Some(required_close_f32(req.data, indicator)?)
            };
            let close_input = close_f32
                .as_ref()
                .map(F32Input::as_slice)
                .unwrap_or(primary_f32.as_slice());
            let ma_type = get_string_param(req.params, "ma_type")
                .unwrap_or("ema")
                .to_string();
            let mut sweep: crate::indicators::keltner::KeltnerBatchRange = Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            sweep.multiplier =
                resolve_f64_range_param(req.params, "multiplier", sweep.multiplier, indicator)?;
            let mut cuda = crate::cuda::CudaKeltner::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .keltner_batch_dev(
                    high_f32.as_slice(),
                    low_f32.as_slice(),
                    close_input,
                    primary_f32.as_slice(),
                    &sweep,
                    ma_type.as_str(),
                )
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.outputs.upper.buf,
                    rows: result.outputs.upper.rows,
                    cols: result.outputs.upper.cols,
                },
                1 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.outputs.middle.buf,
                    rows: result.outputs.middle.rows,
                    cols: result.outputs.middle.cols,
                },
                2 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.outputs.lower.buf,
                    rows: result.outputs.lower.rows,
                    cols: result.outputs.lower.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "kst" => Some((|| {
            let indicator = "kst";
            let fallback_outputs: &[&str] = &["line", "signal"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::kst::KstBatchRange = Default::default();
            sweep.sma_period1 =
                resolve_usize_range_param(req.params, "sma_period1", sweep.sma_period1, indicator)?;
            sweep.sma_period2 =
                resolve_usize_range_param(req.params, "sma_period2", sweep.sma_period2, indicator)?;
            sweep.sma_period3 =
                resolve_usize_range_param(req.params, "sma_period3", sweep.sma_period3, indicator)?;
            sweep.sma_period4 =
                resolve_usize_range_param(req.params, "sma_period4", sweep.sma_period4, indicator)?;
            sweep.roc_period1 =
                resolve_usize_range_param(req.params, "roc_period1", sweep.roc_period1, indicator)?;
            sweep.roc_period2 =
                resolve_usize_range_param(req.params, "roc_period2", sweep.roc_period2, indicator)?;
            sweep.roc_period3 =
                resolve_usize_range_param(req.params, "roc_period3", sweep.roc_period3, indicator)?;
            sweep.roc_period4 =
                resolve_usize_range_param(req.params, "roc_period4", sweep.roc_period4, indicator)?;
            sweep.signal_period = resolve_usize_range_param(
                req.params,
                "signal_period",
                sweep.signal_period,
                indicator,
            )?;
            let mut cuda = crate::cuda::oscillators::CudaKst::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .kst_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.line.buf,
                    rows: result.0.line.rows,
                    cols: result.0.line.cols,
                },
                1 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.signal.buf,
                    rows: result.0.signal.rows,
                    cols: result.0.signal.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "kurtosis" => Some((|| {
            let indicator = "kurtosis";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::kurtosis::KurtosisBatchRange = Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            let mut cuda = crate::cuda::CudaKurtosis::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .kurtosis_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.buf,
                    rows: result.0.rows,
                    cols: result.0.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "kvo" => Some((|| {
            let indicator = "kvo";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let high_f32 = required_high_f32(req.data, indicator)?;
            let low_f32 = required_low_f32(req.data, indicator)?;
            let close_f32 = required_close_f32(req.data, indicator)?;
            let volume_f32 = required_volume_f32(req.data, indicator)?;
            let mut sweep: crate::indicators::kvo::KvoBatchRange = Default::default();
            sweep.short_period = resolve_usize_range_param(
                req.params,
                "short_period",
                sweep.short_period,
                indicator,
            )?;
            sweep.long_period =
                resolve_usize_range_param(req.params, "long_period", sweep.long_period, indicator)?;
            let mut cuda = crate::cuda::oscillators::CudaKvo::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .kvo_batch_dev(
                    high_f32.as_slice(),
                    low_f32.as_slice(),
                    close_f32.as_slice(),
                    volume_f32.as_slice(),
                    &sweep,
                )
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.buf,
                    rows: result.0.rows,
                    cols: result.0.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "linearreg_angle" => Some((|| {
            let indicator = "linearreg_angle";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::linearreg_angle::Linearreg_angleBatchRange =
                Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            let mut cuda = crate::cuda::CudaLinearregAngle::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .linearreg_angle_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.buf,
                    rows: result.rows,
                    cols: result.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "linearreg_intercept" => Some((|| {
            let indicator = "linearreg_intercept";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::linearreg_intercept::LinearRegInterceptBatchRange =
                Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            let mut cuda = crate::cuda::moving_averages::CudaLinregIntercept::new(device_id)
                .map_err(|e| IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                })?;
            let result = cuda
                .linearreg_intercept_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.buf,
                    rows: result.0.rows,
                    cols: result.0.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "linearreg_slope" => Some((|| {
            let indicator = "linearreg_slope";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::linearreg_slope::LinearRegSlopeBatchRange =
                Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            let mut cuda = crate::cuda::moving_averages::CudaLinearregSlope::new(device_id)
                .map_err(|e| IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                })?;
            let result = cuda
                .linearreg_slope_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.buf,
                    rows: result.0.rows,
                    cols: result.0.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "lpc" => Some((|| {
            let indicator = "lpc";
            let fallback_outputs: &[&str] = &["wt1", "wt2", "hist"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let high_f32 = required_high_f32(req.data, indicator)?;
            let low_f32 = required_low_f32(req.data, indicator)?;
            let close_f32 = if primary_source_is_close(req.data) {
                None
            } else {
                Some(required_close_f32(req.data, indicator)?)
            };
            let close_input = close_f32
                .as_ref()
                .map(F32Input::as_slice)
                .unwrap_or(primary_f32.as_slice());
            let mut sweep: crate::indicators::lpc::LpcBatchRange = Default::default();
            sweep.fixed_period = resolve_usize_range_param(
                req.params,
                "fixed_period",
                sweep.fixed_period,
                indicator,
            )?;
            sweep.cycle_mult =
                resolve_f64_range_param(req.params, "cycle_mult", sweep.cycle_mult, indicator)?;
            sweep.tr_mult =
                resolve_f64_range_param(req.params, "tr_mult", sweep.tr_mult, indicator)?;
            let mut cuda = crate::cuda::CudaLpc::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .lpc_batch_dev(
                    high_f32.as_slice(),
                    low_f32.as_slice(),
                    close_input,
                    primary_f32.as_slice(),
                    &sweep,
                )
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.wt1.buf,
                    rows: result.0.wt1.rows,
                    cols: result.0.wt1.cols,
                },
                1 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.wt2.buf,
                    rows: result.0.wt2.rows,
                    cols: result.0.wt2.cols,
                },
                2 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.hist.buf,
                    rows: result.0.hist.rows,
                    cols: result.0.hist.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "lrsi" => Some((|| {
            let indicator = "lrsi";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let high_f32 = required_high_f32(req.data, indicator)?;
            let low_f32 = required_low_f32(req.data, indicator)?;
            let mut sweep: crate::indicators::lrsi::LrsiBatchRange = Default::default();
            sweep.alpha = resolve_f64_range_param(req.params, "alpha", sweep.alpha, indicator)?;
            let mut cuda = crate::cuda::oscillators::CudaLrsi::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .lrsi_batch_dev(high_f32.as_slice(), low_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.buf,
                    rows: result.rows,
                    cols: result.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "mab" => Some((|| {
            let indicator = "mab";
            let fallback_outputs: &[&str] = &["upper", "middle", "lower"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::mab::MabBatchRange = Default::default();
            sweep.fast_period =
                resolve_usize_range_param(req.params, "fast_period", sweep.fast_period, indicator)?;
            sweep.slow_period =
                resolve_usize_range_param(req.params, "slow_period", sweep.slow_period, indicator)?;
            sweep.devup = resolve_f64_range_param(req.params, "devup", sweep.devup, indicator)?;
            sweep.devdn = resolve_f64_range_param(req.params, "devdn", sweep.devdn, indicator)?;
            let mut cuda = crate::cuda::moving_averages::CudaMab::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .mab_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.upper.buf,
                    rows: result.0.upper.rows,
                    cols: result.0.upper.cols,
                },
                1 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.middle.buf,
                    rows: result.0.middle.rows,
                    cols: result.0.middle.cols,
                },
                2 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.lower.buf,
                    rows: result.0.lower.rows,
                    cols: result.0.lower.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "macd" => Some((|| {
            let indicator = "macd";
            let fallback_outputs: &[&str] = &["macd", "signal", "hist"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::macd::MacdBatchRange = Default::default();
            sweep.fast_period =
                resolve_usize_range_param(req.params, "fast_period", sweep.fast_period, indicator)?;
            sweep.slow_period =
                resolve_usize_range_param(req.params, "slow_period", sweep.slow_period, indicator)?;
            sweep.signal_period = resolve_usize_range_param(
                req.params,
                "signal_period",
                sweep.signal_period,
                indicator,
            )?;
            let mut cuda = crate::cuda::oscillators::CudaMacd::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .macd_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.macd.buf,
                    rows: result.0.macd.rows,
                    cols: result.0.macd.cols,
                },
                1 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.signal.buf,
                    rows: result.0.signal.rows,
                    cols: result.0.signal.cols,
                },
                2 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.hist.buf,
                    rows: result.0.hist.rows,
                    cols: result.0.hist.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "macz" => Some((|| {
            let indicator = "macz";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let volume_f32_opt = optional_volume_f32(req.data);
            let mut sweep: crate::indicators::macz::MaczBatchRange = Default::default();
            sweep.fast_length =
                resolve_usize_range_param(req.params, "fast_length", sweep.fast_length, indicator)?;
            sweep.slow_length =
                resolve_usize_range_param(req.params, "slow_length", sweep.slow_length, indicator)?;
            sweep.signal_length = resolve_usize_range_param(
                req.params,
                "signal_length",
                sweep.signal_length,
                indicator,
            )?;
            sweep.lengthz =
                resolve_usize_range_param(req.params, "lengthz", sweep.lengthz, indicator)?;
            sweep.length_stdev = resolve_usize_range_param(
                req.params,
                "length_stdev",
                sweep.length_stdev,
                indicator,
            )?;
            sweep.a = resolve_f64_range_param(req.params, "a", sweep.a, indicator)?;
            sweep.b = resolve_f64_range_param(req.params, "b", sweep.b, indicator)?;
            let mut cuda = crate::cuda::moving_averages::CudaMacz::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .macz_batch_dev(
                    primary_f32.as_slice(),
                    volume_f32_opt.as_ref().map(F32Input::as_slice),
                    &sweep,
                )
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.buf,
                    rows: result.0.rows,
                    cols: result.0.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "marketefi" => Some((|| {
            let indicator = "marketefi";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let high_f32 = required_high_f32(req.data, indicator)?;
            let low_f32 = required_low_f32(req.data, indicator)?;
            let volume_f32 = required_volume_f32(req.data, indicator)?;
            let mut cuda = crate::cuda::CudaMarketefi::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .marketefi_batch_dev(
                    high_f32.as_slice(),
                    low_f32.as_slice(),
                    volume_f32.as_slice(),
                )
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.buf,
                    rows: result.rows,
                    cols: result.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "mass" => Some((|| {
            let indicator = "mass";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let high_f32 = required_high_f32(req.data, indicator)?;
            let low_f32 = required_low_f32(req.data, indicator)?;
            let mut sweep: crate::indicators::mass::MassBatchRange = Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            let mut cuda = crate::cuda::CudaMass::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .mass_batch_dev(high_f32.as_slice(), low_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.buf,
                    rows: result.0.rows,
                    cols: result.0.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "mean_ad" => Some((|| {
            let indicator = "mean_ad";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::mean_ad::MeanAdBatchRange = Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            let mut cuda = crate::cuda::CudaMeanAd::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .mean_ad_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.buf,
                    rows: result.rows,
                    cols: result.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "medium_ad" => Some((|| {
            let indicator = "medium_ad";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::medium_ad::MediumAdBatchRange = Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            let mut cuda = crate::cuda::CudaMediumAd::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .medium_ad_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.buf,
                    rows: result.rows,
                    cols: result.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "medprice" => Some((|| {
            let indicator = "medprice";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let high_f32 = required_high_f32(req.data, indicator)?;
            let low_f32 = required_low_f32(req.data, indicator)?;
            let mut cuda = crate::cuda::CudaMedprice::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .medprice_batch_dev(high_f32.as_slice(), low_f32.as_slice())
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.buf,
                    rows: result.rows,
                    cols: result.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "mfi" => Some((|| {
            let indicator = "mfi";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let volume_f32 = required_volume_f32(req.data, indicator)?;
            let mut sweep: crate::indicators::mfi::MfiBatchRange = Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            let mut cuda = crate::cuda::oscillators::CudaMfi::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .mfi_batch_dev(primary_f32.as_slice(), volume_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.buf,
                    rows: result.0.rows,
                    cols: result.0.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "minmax" => Some((|| {
            let indicator = "minmax";
            let fallback_outputs: &[&str] = &["is_min", "is_max", "last_min", "last_max"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let high_f32 = required_high_f32(req.data, indicator)?;
            let low_f32 = required_low_f32(req.data, indicator)?;
            let mut sweep: crate::indicators::minmax::MinmaxBatchRange = Default::default();
            sweep.order = resolve_usize_range_param(req.params, "order", sweep.order, indicator)?;
            let mut cuda = crate::cuda::CudaMinmax::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .minmax_batch_dev(high_f32.as_slice(), low_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.is_min,
                    rows: result.0.rows,
                    cols: result.0.cols,
                },
                1 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.is_max,
                    rows: result.0.rows,
                    cols: result.0.cols,
                },
                2 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.last_min,
                    rows: result.0.rows,
                    cols: result.0.cols,
                },
                3 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.last_max,
                    rows: result.0.rows,
                    cols: result.0.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "mod_god_mode" => Some((|| {
            let indicator = "mod_god_mode";
            let fallback_outputs: &[&str] = &["wt1", "wt2", "hist"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let high_f32 = required_high_f32(req.data, indicator)?;
            let low_f32 = required_low_f32(req.data, indicator)?;
            let close_f32 = required_close_f32(req.data, indicator)?;
            let volume_f32_opt = optional_volume_f32(req.data);
            let mut sweep: crate::indicators::mod_god_mode::ModGodModeBatchRange =
                Default::default();
            sweep.n1 = resolve_usize_range_param(req.params, "n1", sweep.n1, indicator)?;
            sweep.n2 = resolve_usize_range_param(req.params, "n2", sweep.n2, indicator)?;
            sweep.n3 = resolve_usize_range_param(req.params, "n3", sweep.n3, indicator)?;
            let mut cuda = crate::cuda::CudaModGodMode::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .mod_god_mode_batch_dev(
                    high_f32.as_slice(),
                    low_f32.as_slice(),
                    close_f32.as_slice(),
                    volume_f32_opt.as_ref().map(F32Input::as_slice),
                    &sweep,
                )
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.outputs.wt1.buf,
                    rows: result.outputs.wt1.rows,
                    cols: result.outputs.wt1.cols,
                },
                1 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.outputs.wt2.buf,
                    rows: result.outputs.wt2.rows,
                    cols: result.outputs.wt2.cols,
                },
                2 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.outputs.hist.buf,
                    rows: result.outputs.hist.rows,
                    cols: result.outputs.hist.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "mom" => Some((|| {
            let indicator = "mom";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::mom::MomBatchRange = Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            let mut cuda = crate::cuda::oscillators::CudaMom::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .mom_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.buf,
                    rows: result.rows,
                    cols: result.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "msw" => Some((|| {
            let indicator = "msw";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::msw::MswBatchRange = Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            let mut cuda = crate::cuda::oscillators::CudaMsw::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .msw_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.buf,
                    rows: result.0.rows,
                    cols: result.0.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "nadaraya_watson_envelope" => Some((|| {
            let indicator = "nadaraya_watson_envelope";
            let fallback_outputs: &[&str] = &["upper", "lower"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::nadaraya_watson_envelope::NweBatchRange =
                Default::default();
            sweep.bandwidth =
                resolve_f64_range_param(req.params, "bandwidth", sweep.bandwidth, indicator)?;
            sweep.multiplier =
                resolve_f64_range_param(req.params, "multiplier", sweep.multiplier, indicator)?;
            sweep.lookback =
                resolve_usize_range_param(req.params, "lookback", sweep.lookback, indicator)?;
            let mut cuda = crate::cuda::CudaNwe::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .nwe_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.upper.buf,
                    rows: result.0.upper.rows,
                    cols: result.0.upper.cols,
                },
                1 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.lower.buf,
                    rows: result.0.lower.rows,
                    cols: result.0.lower.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "natr" => Some((|| {
            let indicator = "natr";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let high_f32 = required_high_f32(req.data, indicator)?;
            let low_f32 = required_low_f32(req.data, indicator)?;
            let close_f32 = required_close_f32(req.data, indicator)?;
            let mut sweep: crate::indicators::natr::NatrBatchRange = Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            let mut cuda = crate::cuda::CudaNatr::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .natr_batch_dev(
                    high_f32.as_slice(),
                    low_f32.as_slice(),
                    close_f32.as_slice(),
                    &sweep,
                )
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.buf,
                    rows: result.rows,
                    cols: result.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "net_myrsi" => Some((|| {
            let indicator = "net_myrsi";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::net_myrsi::NetMyrsiBatchRange = Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            let mut cuda = crate::cuda::CudaNetMyrsi::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .net_myrsi_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.buf,
                    rows: result.0.rows,
                    cols: result.0.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "nvi" => Some((|| {
            let indicator = "nvi";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let close_f32 = required_close_f32(req.data, indicator)?;
            let volume_f32 = required_volume_f32(req.data, indicator)?;
            let mut cuda = crate::cuda::CudaNvi::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .nvi_batch_dev(close_f32.as_slice(), volume_f32.as_slice())
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.buf,
                    rows: result.rows,
                    cols: result.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "obv" => Some((|| {
            let indicator = "obv";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let close_f32 = required_close_f32(req.data, indicator)?;
            let volume_f32 = required_volume_f32(req.data, indicator)?;
            let mut cuda = crate::cuda::CudaObv::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .obv_batch_dev(close_f32.as_slice(), volume_f32.as_slice())
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.buf,
                    rows: result.rows,
                    cols: result.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "ott" => Some((|| {
            let indicator = "ott";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::ott::OttBatchRange = Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            sweep.percent =
                resolve_f64_range_param(req.params, "percent", sweep.percent, indicator)?;
            let mut cuda = crate::cuda::moving_averages::CudaOtt::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .ott_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.buf,
                    rows: result.rows,
                    cols: result.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "otto" => Some((|| {
            let indicator = "otto";
            let fallback_outputs: &[&str] = &["output_0", "output_1"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::otto::OttoBatchRange = Default::default();
            sweep.ott_period =
                resolve_usize_range_param(req.params, "ott_period", sweep.ott_period, indicator)?;
            sweep.ott_percent =
                resolve_f64_range_param(req.params, "ott_percent", sweep.ott_percent, indicator)?;
            sweep.fast_vidya =
                resolve_usize_range_param(req.params, "fast_vidya", sweep.fast_vidya, indicator)?;
            sweep.slow_vidya =
                resolve_usize_range_param(req.params, "slow_vidya", sweep.slow_vidya, indicator)?;
            sweep.correcting_constant = resolve_f64_range_param(
                req.params,
                "correcting_constant",
                sweep.correcting_constant,
                indicator,
            )?;
            let mut cuda = crate::cuda::moving_averages::CudaOtto::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .otto_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.buf,
                    rows: result.0.rows,
                    cols: result.0.cols,
                },
                1 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.1.buf,
                    rows: result.1.rows,
                    cols: result.1.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "percentile_nearest_rank" => Some((|| {
            let indicator = "percentile_nearest_rank";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::percentile_nearest_rank::PercentileNearestRankBatchRange = Default::default();
            sweep.length =
                resolve_usize_range_param(req.params, "length", sweep.length, indicator)?;
            sweep.percentage =
                resolve_f64_range_param(req.params, "percentage", sweep.percentage, indicator)?;
            let mut cuda = crate::cuda::CudaPercentileNearestRank::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .pnr_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.buf,
                    rows: result.0.rows,
                    cols: result.0.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "pfe" => Some((|| {
            let indicator = "pfe";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::pfe::PfeBatchRange = Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            sweep.smoothing =
                resolve_usize_range_param(req.params, "smoothing", sweep.smoothing, indicator)?;
            let mut cuda = crate::cuda::CudaPfe::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .pfe_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.buf,
                    rows: result.rows,
                    cols: result.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "pivot" => Some((|| {
            let indicator = "pivot";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let open_f32 = required_open_f32(req.data, indicator)?;
            let high_f32 = required_high_f32(req.data, indicator)?;
            let low_f32 = required_low_f32(req.data, indicator)?;
            let close_f32 = required_close_f32(req.data, indicator)?;
            let mut sweep: crate::indicators::pivot::PivotBatchRange = Default::default();
            sweep.mode = resolve_usize_range_param(req.params, "mode", sweep.mode, indicator)?;
            let mut cuda = crate::cuda::CudaPivot::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .pivot_batch_dev(
                    high_f32.as_slice(),
                    low_f32.as_slice(),
                    close_f32.as_slice(),
                    open_f32.as_slice(),
                    &sweep,
                )
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.buf,
                    rows: result.0.rows,
                    cols: result.0.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "pma" => Some((|| {
            let indicator = "pma";
            let fallback_outputs: &[&str] = &["predict", "trigger"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::pma::PmaBatchRange = Default::default();
            sweep.dummy = resolve_usize_range_param(req.params, "dummy", sweep.dummy, indicator)?;
            let mut cuda = crate::cuda::moving_averages::CudaPma::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .pma_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.predict.buf,
                    rows: result.predict.rows,
                    cols: result.predict.cols,
                },
                1 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.trigger.buf,
                    rows: result.trigger.rows,
                    cols: result.trigger.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "ppo" => Some((|| {
            let indicator = "ppo";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::ppo::PpoBatchRange = Default::default();
            sweep.fast_period =
                resolve_usize_range_param(req.params, "fast_period", sweep.fast_period, indicator)?;
            sweep.slow_period =
                resolve_usize_range_param(req.params, "slow_period", sweep.slow_period, indicator)?;
            let mut cuda = crate::cuda::oscillators::CudaPpo::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .ppo_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.buf,
                    rows: result.0.rows,
                    cols: result.0.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "prb" => Some((|| {
            let indicator = "prb";
            let fallback_outputs: &[&str] = &["output_0", "output_1", "output_2"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let smooth_data = get_bool_param(req.params, "smooth_data", indicator)?.unwrap_or(true);
            let mut sweep: crate::indicators::prb::PrbBatchRange = Default::default();
            sweep.smooth_period = resolve_usize_range_param(
                req.params,
                "smooth_period",
                sweep.smooth_period,
                indicator,
            )?;
            sweep.regression_period = resolve_usize_range_param(
                req.params,
                "regression_period",
                sweep.regression_period,
                indicator,
            )?;
            sweep.polynomial_order = resolve_usize_range_param(
                req.params,
                "polynomial_order",
                sweep.polynomial_order,
                indicator,
            )?;
            let mut cuda = crate::cuda::CudaPrb::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .prb_batch_dev(primary_f32.as_slice(), &sweep, smooth_data)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.buf,
                    rows: result.0.rows,
                    cols: result.0.cols,
                },
                1 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.1.buf,
                    rows: result.1.rows,
                    cols: result.1.cols,
                },
                2 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.2.buf,
                    rows: result.2.rows,
                    cols: result.2.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "pvi" => Some((|| {
            let indicator = "pvi";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let close_f32 = if primary_source_is_close(req.data) {
                None
            } else {
                Some(required_close_f32(req.data, indicator)?)
            };
            let close_input = close_f32
                .as_ref()
                .map(F32Input::as_slice)
                .unwrap_or(primary_f32.as_slice());
            let volume_f32 = required_volume_f32(req.data, indicator)?;
            let mut cuda = crate::cuda::CudaPvi::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .pvi_batch_dev(close_input, volume_f32.as_slice(), primary_f32.as_slice())
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.buf,
                    rows: result.rows,
                    cols: result.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "qqe" => Some((|| {
            let indicator = "qqe";
            let fallback_outputs: &[&str] = &["fast", "slow"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::qqe::QqeBatchRange = Default::default();
            sweep.rsi_period =
                resolve_usize_range_param(req.params, "rsi_period", sweep.rsi_period, indicator)?;
            sweep.smoothing_factor = resolve_usize_range_param(
                req.params,
                "smoothing_factor",
                sweep.smoothing_factor,
                indicator,
            )?;
            sweep.fast_factor =
                resolve_f64_range_param(req.params, "fast_factor", sweep.fast_factor, indicator)?;
            let mut cuda = crate::cuda::oscillators::CudaQqe::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let first_valid = primary_f32
                .as_slice()
                .iter()
                .position(|value| value.is_finite())
                .ok_or_else(|| IndicatorDispatchError::InvalidParam {
                    indicator: indicator.to_string(),
                    key: "data".to_string(),
                    reason: "all values are NaN".to_string(),
                })?;
            let (result, _) = cuda
                .qqe_batch_output_dev(primary_f32.as_slice(), &sweep, output_index)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = crate::cuda::moving_averages::DeviceArrayF32 {
                buf: result.buf,
                rows: result.rows,
                cols: result.cols,
            };
            let warmup = Some(
                first_valid
                    + sweep.rsi_period.0.min(sweep.rsi_period.1)
                    + sweep.smoothing_factor.0.min(sweep.smoothing_factor.1)
                    - 2,
            );
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, warmup)
        })()),
        "qstick" => Some((|| {
            let indicator = "qstick";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let open_f32 = required_open_f32(req.data, indicator)?;
            let close_f32 = required_close_f32(req.data, indicator)?;
            let mut sweep: crate::indicators::qstick::QstickBatchRange = Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            let mut cuda = crate::cuda::CudaQstick::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .qstick_batch_dev(open_f32.as_slice(), close_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.buf,
                    rows: result.rows,
                    cols: result.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "range_filter" => Some((|| {
            let indicator = "range_filter";
            let fallback_outputs: &[&str] = &["filter", "high", "low"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::range_filter::RangeFilterBatchRange =
                Default::default();
            sweep.range_size =
                resolve_f64_range_param(req.params, "range_size", sweep.range_size, indicator)?;
            sweep.range_period = resolve_usize_range_param(
                req.params,
                "range_period",
                sweep.range_period,
                indicator,
            )?;
            let mut cuda = crate::cuda::CudaRangeFilter::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .range_filter_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.filter,
                    rows: result.0.rows,
                    cols: result.0.cols,
                },
                1 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.high,
                    rows: result.0.rows,
                    cols: result.0.cols,
                },
                2 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.low,
                    rows: result.0.rows,
                    cols: result.0.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "reverse_rsi" => Some((|| {
            let indicator = "reverse_rsi";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::reverse_rsi::ReverseRsiBatchRange =
                Default::default();
            sweep.rsi_length_range = resolve_usize_range_param(
                req.params,
                "rsi_length_range",
                sweep.rsi_length_range,
                indicator,
            )?;
            sweep.rsi_level_range = resolve_f64_range_param(
                req.params,
                "rsi_level_range",
                sweep.rsi_level_range,
                indicator,
            )?;
            let mut cuda =
                crate::cuda::oscillators::CudaReverseRsi::new(device_id).map_err(|e| {
                    IndicatorDispatchError::KernelUnavailable {
                        details: e.to_string(),
                    }
                })?;
            let result = cuda
                .reverse_rsi_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.buf,
                    rows: result.0.rows,
                    cols: result.0.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "roc" => Some((|| {
            let indicator = "roc";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::roc::RocBatchRange = Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            let mut cuda = crate::cuda::oscillators::CudaRoc::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .roc_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.buf,
                    rows: result.rows,
                    cols: result.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "rocp" => Some((|| {
            let indicator = "rocp";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::rocp::RocpBatchRange = Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            let mut cuda = crate::cuda::oscillators::CudaRocp::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .rocp_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.buf,
                    rows: result.0.rows,
                    cols: result.0.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "rocr" => Some((|| {
            let indicator = "rocr";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::rocr::RocrBatchRange = Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            let mut cuda = crate::cuda::CudaRocr::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .rocr_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.buf,
                    rows: result.rows,
                    cols: result.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "rsi" => Some((|| {
            let indicator = "rsi";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::rsi::RsiBatchRange = Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            let mut cuda = crate::cuda::oscillators::CudaRsi::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .rsi_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.buf,
                    rows: result.rows,
                    cols: result.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "rsmk" => Some((|| {
            let indicator = "rsmk";
            let fallback_outputs: &[&str] = &["a", "b"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::rsmk::RsmkBatchRange = Default::default();
            sweep.lookback =
                resolve_usize_range_param(req.params, "lookback", sweep.lookback, indicator)?;
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            sweep.signal_period = resolve_usize_range_param(
                req.params,
                "signal_period",
                sweep.signal_period,
                indicator,
            )?;
            let mut cuda = crate::cuda::moving_averages::CudaRsmk::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .rsmk_batch_dev(primary_f32.as_slice(), primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.a.buf,
                    rows: result.0.a.rows,
                    cols: result.0.a.cols,
                },
                1 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.b.buf,
                    rows: result.0.b.rows,
                    cols: result.0.b.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "rsx" => Some((|| {
            let indicator = "rsx";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::rsx::RsxBatchRange = Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            let mut cuda = crate::cuda::oscillators::CudaRsx::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .rsx_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.buf,
                    rows: result.rows,
                    cols: result.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "rvi" => Some((|| {
            let indicator = "rvi";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::rvi::RviBatchRange = Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            sweep.ma_len =
                resolve_usize_range_param(req.params, "ma_len", sweep.ma_len, indicator)?;
            sweep.matype =
                resolve_usize_range_param(req.params, "matype", sweep.matype, indicator)?;
            sweep.devtype =
                resolve_usize_range_param(req.params, "devtype", sweep.devtype, indicator)?;
            let mut cuda = crate::cuda::oscillators::CudaRvi::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .rvi_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.buf,
                    rows: result.0.rows,
                    cols: result.0.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "safezonestop" => Some((|| {
            let indicator = "safezonestop";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let high_f32 = required_high_f32(req.data, indicator)?;
            let low_f32 = required_low_f32(req.data, indicator)?;
            let direction = get_string_param(req.params, "direction")
                .unwrap_or("long")
                .to_string();
            let mut sweep: crate::indicators::safezonestop::SafeZoneStopBatchRange =
                Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            sweep.mult = resolve_f64_range_param(req.params, "mult", sweep.mult, indicator)?;
            sweep.max_lookback = resolve_usize_range_param(
                req.params,
                "max_lookback",
                sweep.max_lookback,
                indicator,
            )?;
            let mut cuda = crate::cuda::CudaSafeZoneStop::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .safezonestop_batch_dev(
                    high_f32.as_slice(),
                    low_f32.as_slice(),
                    direction.as_str(),
                    &sweep,
                )
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.buf,
                    rows: result.0.rows,
                    cols: result.0.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "sar" => Some((|| {
            let indicator = "sar";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let high_f32 = required_high_f32(req.data, indicator)?;
            let low_f32 = required_low_f32(req.data, indicator)?;
            let mut sweep: crate::indicators::sar::SarBatchRange = Default::default();
            sweep.acceleration =
                resolve_f64_range_param(req.params, "acceleration", sweep.acceleration, indicator)?;
            sweep.maximum =
                resolve_f64_range_param(req.params, "maximum", sweep.maximum, indicator)?;
            let mut cuda = crate::cuda::CudaSar::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .sar_batch_dev(high_f32.as_slice(), low_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.buf,
                    rows: result.0.rows,
                    cols: result.0.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "squeeze_momentum" => {
            Some((|| {
                let indicator = "squeeze_momentum";
                let fallback_outputs: &[&str] = &["output_0", "output_1", "output_2"];
                let (output_id, output_index) =
                    resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
                let high_f32 = required_high_f32(req.data, indicator)?;
                let low_f32 = required_low_f32(req.data, indicator)?;
                let close_f32 = required_close_f32(req.data, indicator)?;
                let mut sweep: crate::indicators::squeeze_momentum::SqueezeMomentumBatchRange =
                    Default::default();
                sweep.length_bb =
                    resolve_usize_range_param(req.params, "length_bb", sweep.length_bb, indicator)?;
                sweep.mult_bb =
                    resolve_f64_range_param(req.params, "mult_bb", sweep.mult_bb, indicator)?;
                sweep.length_kc =
                    resolve_usize_range_param(req.params, "length_kc", sweep.length_kc, indicator)?;
                sweep.mult_kc =
                    resolve_f64_range_param(req.params, "mult_kc", sweep.mult_kc, indicator)?;
                let mut cuda = crate::cuda::oscillators::CudaSqueezeMomentum::new(device_id)
                    .map_err(|e| IndicatorDispatchError::KernelUnavailable {
                        details: e.to_string(),
                    })?;
                let result = cuda
                    .squeeze_momentum_batch_dev(
                        high_f32.as_slice(),
                        low_f32.as_slice(),
                        close_f32.as_slice(),
                        &sweep,
                    )
                    .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
                let owner = match output_index {
                    0 => crate::cuda::moving_averages::DeviceArrayF32 {
                        buf: result.1.buf,
                        rows: result.1.rows,
                        cols: result.1.cols,
                    },
                    1 => crate::cuda::moving_averages::DeviceArrayF32 {
                        buf: result.0.buf,
                        rows: result.0.rows,
                        cols: result.0.cols,
                    },
                    2 => crate::cuda::moving_averages::DeviceArrayF32 {
                        buf: result.2.buf,
                        rows: result.2.rows,
                        cols: result.2.cols,
                    },
                    _ => {
                        return Err(IndicatorDispatchError::UnknownOutput {
                            indicator: indicator.to_string(),
                            output: output_id.clone(),
                        });
                    }
                };
                finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
            })())
        }
        "srsi" => Some((|| {
            let indicator = "srsi";
            let fallback_outputs: &[&str] = &["k", "d"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::srsi::SrsiBatchRange = Default::default();
            sweep.rsi_period =
                resolve_usize_range_param(req.params, "rsi_period", sweep.rsi_period, indicator)?;
            sweep.stoch_period = resolve_usize_range_param(
                req.params,
                "stoch_period",
                sweep.stoch_period,
                indicator,
            )?;
            sweep.k = resolve_usize_range_param(req.params, "k", sweep.k, indicator)?;
            sweep.d = resolve_usize_range_param(req.params, "d", sweep.d, indicator)?;
            let mut cuda = crate::cuda::oscillators::CudaSrsi::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .srsi_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.k.buf,
                    rows: result.0.k.rows,
                    cols: result.0.k.cols,
                },
                1 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.d.buf,
                    rows: result.0.d.rows,
                    cols: result.0.d.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "stc" => Some((|| {
            let indicator = "stc";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::stc::StcBatchRange = Default::default();
            sweep.fast_period =
                resolve_usize_range_param(req.params, "fast_period", sweep.fast_period, indicator)?;
            sweep.slow_period =
                resolve_usize_range_param(req.params, "slow_period", sweep.slow_period, indicator)?;
            sweep.k_period =
                resolve_usize_range_param(req.params, "k_period", sweep.k_period, indicator)?;
            sweep.d_period =
                resolve_usize_range_param(req.params, "d_period", sweep.d_period, indicator)?;
            let mut cuda = crate::cuda::oscillators::CudaStc::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .stc_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.buf,
                    rows: result.0.rows,
                    cols: result.0.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "stddev" => Some((|| {
            let indicator = "stddev";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::stddev::StdDevBatchRange = Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            sweep.nbdev = resolve_f64_range_param(req.params, "nbdev", sweep.nbdev, indicator)?;
            let mut cuda = crate::cuda::CudaStddev::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .stddev_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.buf,
                    rows: result.0.rows,
                    cols: result.0.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "stoch" => Some((|| {
            let indicator = "stoch";
            let fallback_outputs: &[&str] = &["k", "d"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let high_f32 = required_high_f32(req.data, indicator)?;
            let low_f32 = required_low_f32(req.data, indicator)?;
            let close_f32 = required_close_f32(req.data, indicator)?;
            let mut sweep: crate::indicators::stoch::StochBatchRange = Default::default();
            sweep.fastk_period = resolve_usize_range_param(
                req.params,
                "fastk_period",
                sweep.fastk_period,
                indicator,
            )?;
            sweep.slowk_period = resolve_usize_range_param(
                req.params,
                "slowk_period",
                sweep.slowk_period,
                indicator,
            )?;
            sweep.slowd_period = resolve_usize_range_param(
                req.params,
                "slowd_period",
                sweep.slowd_period,
                indicator,
            )?;
            let mut cuda = crate::cuda::oscillators::CudaStoch::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .stoch_batch_dev(
                    high_f32.as_slice(),
                    low_f32.as_slice(),
                    close_f32.as_slice(),
                    &sweep,
                )
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.k.buf,
                    rows: result.k.rows,
                    cols: result.k.cols,
                },
                1 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.d.buf,
                    rows: result.d.rows,
                    cols: result.d.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "stochf" => Some((|| {
            let indicator = "stochf";
            let fallback_outputs: &[&str] = &["a", "b"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let high_f32 = required_high_f32(req.data, indicator)?;
            let low_f32 = required_low_f32(req.data, indicator)?;
            let close_f32 = required_close_f32(req.data, indicator)?;
            let mut sweep: crate::indicators::stochf::StochfBatchRange = Default::default();
            sweep.fastk_period = resolve_usize_range_param(
                req.params,
                "fastk_period",
                sweep.fastk_period,
                indicator,
            )?;
            sweep.fastd_period = resolve_usize_range_param(
                req.params,
                "fastd_period",
                sweep.fastd_period,
                indicator,
            )?;
            let mut cuda = crate::cuda::oscillators::CudaStochf::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .stochf_batch_dev(
                    high_f32.as_slice(),
                    low_f32.as_slice(),
                    close_f32.as_slice(),
                    &sweep,
                )
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.a.buf,
                    rows: result.0.a.rows,
                    cols: result.0.a.cols,
                },
                1 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.b.buf,
                    rows: result.0.b.rows,
                    cols: result.0.b.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "supertrend" => Some((|| {
            let indicator = "supertrend";
            let fallback_outputs: &[&str] = &["output_0", "output_1"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let high_f32 = required_high_f32(req.data, indicator)?;
            let low_f32 = required_low_f32(req.data, indicator)?;
            let close_f32 = required_close_f32(req.data, indicator)?;
            let mut sweep: crate::indicators::supertrend::SuperTrendBatchRange = Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            sweep.factor = resolve_f64_range_param(req.params, "factor", sweep.factor, indicator)?;
            let mut cuda = crate::cuda::CudaSupertrend::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .supertrend_batch_dev(
                    high_f32.as_slice(),
                    low_f32.as_slice(),
                    close_f32.as_slice(),
                    &sweep,
                )
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.buf,
                    rows: result.0.rows,
                    cols: result.0.cols,
                },
                1 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.1.buf,
                    rows: result.1.rows,
                    cols: result.1.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "trix" => Some((|| {
            let indicator = "trix";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::trix::TrixBatchRange = Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            let mut cuda = crate::cuda::moving_averages::CudaTrix::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .trix_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.buf,
                    rows: result.rows,
                    cols: result.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "tsf" => Some((|| {
            let indicator = "tsf";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::tsf::TsfBatchRange = Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            let mut cuda = crate::cuda::moving_averages::CudaTsf::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .tsf_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.buf,
                    rows: result.0.rows,
                    cols: result.0.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "tsi" => Some((|| {
            let indicator = "tsi";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::tsi::TsiBatchRange = Default::default();
            sweep.long_period =
                resolve_usize_range_param(req.params, "long_period", sweep.long_period, indicator)?;
            sweep.short_period = resolve_usize_range_param(
                req.params,
                "short_period",
                sweep.short_period,
                indicator,
            )?;
            let mut cuda = crate::cuda::oscillators::CudaTsi::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .tsi_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.buf,
                    rows: result.0.rows,
                    cols: result.0.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "ttm_squeeze" => Some((|| {
            let indicator = "ttm_squeeze";
            let fallback_outputs: &[&str] = &["output_0", "output_1"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let high_f32 = required_high_f32(req.data, indicator)?;
            let low_f32 = required_low_f32(req.data, indicator)?;
            let close_f32 = required_close_f32(req.data, indicator)?;
            let mut sweep: crate::indicators::ttm_squeeze::TtmSqueezeBatchRange =
                Default::default();
            sweep.length =
                resolve_usize_range_param(req.params, "length", sweep.length, indicator)?;
            sweep.bb_mult =
                resolve_f64_range_param(req.params, "bb_mult", sweep.bb_mult, indicator)?;
            sweep.kc_high =
                resolve_f64_range_param(req.params, "kc_high", sweep.kc_high, indicator)?;
            sweep.kc_mid = resolve_f64_range_param(req.params, "kc_mid", sweep.kc_mid, indicator)?;
            sweep.kc_low = resolve_f64_range_param(req.params, "kc_low", sweep.kc_low, indicator)?;
            let mut cuda =
                crate::cuda::oscillators::CudaTtmSqueeze::new(device_id).map_err(|e| {
                    IndicatorDispatchError::KernelUnavailable {
                        details: e.to_string(),
                    }
                })?;
            let result = cuda
                .ttm_squeeze_batch_dev(
                    high_f32.as_slice(),
                    low_f32.as_slice(),
                    close_f32.as_slice(),
                    &sweep,
                )
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.buf,
                    rows: result.0.rows,
                    cols: result.0.cols,
                },
                1 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.1.buf,
                    rows: result.1.rows,
                    cols: result.1.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "ttm_trend" => Some((|| {
            let indicator = "ttm_trend";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let close_f32 = if primary_source_is_close(req.data) {
                None
            } else {
                Some(required_close_f32(req.data, indicator)?)
            };
            let close_input = close_f32
                .as_ref()
                .map(F32Input::as_slice)
                .unwrap_or(primary_f32.as_slice());
            let mut sweep: crate::indicators::ttm_trend::TtmTrendBatchRange = Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            let mut cuda = crate::cuda::CudaTtmTrend::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .ttm_trend_batch_dev(primary_f32.as_slice(), close_input, &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.buf,
                    rows: result.rows,
                    cols: result.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "ui" => Some((|| {
            let indicator = "ui";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::ui::UiBatchRange = Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            sweep.scalar = resolve_f64_range_param(req.params, "scalar", sweep.scalar, indicator)?;
            let mut cuda = crate::cuda::CudaUi::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .ui_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.buf,
                    rows: result.0.rows,
                    cols: result.0.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "ultosc" => Some((|| {
            let indicator = "ultosc";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let high_f32 = required_high_f32(req.data, indicator)?;
            let low_f32 = required_low_f32(req.data, indicator)?;
            let close_f32 = required_close_f32(req.data, indicator)?;
            let mut sweep: crate::indicators::ultosc::UltOscBatchRange = Default::default();
            sweep.timeperiod1 =
                resolve_usize_range_param(req.params, "timeperiod1", sweep.timeperiod1, indicator)?;
            sweep.timeperiod2 =
                resolve_usize_range_param(req.params, "timeperiod2", sweep.timeperiod2, indicator)?;
            sweep.timeperiod3 =
                resolve_usize_range_param(req.params, "timeperiod3", sweep.timeperiod3, indicator)?;
            let mut cuda = crate::cuda::oscillators::CudaUltosc::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .ultosc_batch_dev(
                    high_f32.as_slice(),
                    low_f32.as_slice(),
                    close_f32.as_slice(),
                    &sweep,
                )
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.buf,
                    rows: result.rows,
                    cols: result.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "var" => Some((|| {
            let indicator = "var";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::var::VarBatchRange = Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            sweep.nbdev = resolve_f64_range_param(req.params, "nbdev", sweep.nbdev, indicator)?;
            let mut cuda = crate::cuda::CudaVar::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .var_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.buf,
                    rows: result.0.rows,
                    cols: result.0.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "vi" => Some((|| {
            let indicator = "vi";
            let fallback_outputs: &[&str] = &["a", "b"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let high_f32 = required_high_f32(req.data, indicator)?;
            let low_f32 = required_low_f32(req.data, indicator)?;
            let close_f32 = required_close_f32(req.data, indicator)?;
            let mut sweep: crate::indicators::vi::ViBatchRange = Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            let mut cuda = crate::cuda::CudaVi::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .vi_batch_dev(
                    high_f32.as_slice(),
                    low_f32.as_slice(),
                    close_f32.as_slice(),
                    &sweep,
                )
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.a.buf,
                    rows: result.0.a.rows,
                    cols: result.0.a.cols,
                },
                1 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.b.buf,
                    rows: result.0.b.rows,
                    cols: result.0.b.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "vidya" => Some((|| {
            let indicator = "vidya";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::vidya::VidyaBatchRange = Default::default();
            sweep.short_period = resolve_usize_range_param(
                req.params,
                "short_period",
                sweep.short_period,
                indicator,
            )?;
            sweep.long_period =
                resolve_usize_range_param(req.params, "long_period", sweep.long_period, indicator)?;
            sweep.alpha = resolve_f64_range_param(req.params, "alpha", sweep.alpha, indicator)?;
            let mut cuda =
                crate::cuda::moving_averages::CudaVidya::new(device_id).map_err(|e| {
                    IndicatorDispatchError::KernelUnavailable {
                        details: e.to_string(),
                    }
                })?;
            let result = cuda
                .vidya_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.buf,
                    rows: result.rows,
                    cols: result.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "vlma" => Some((|| {
            let indicator = "vlma";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::vlma::VlmaBatchRange = Default::default();
            sweep.min_period =
                resolve_usize_range_param(req.params, "min_period", sweep.min_period, indicator)?;
            sweep.max_period =
                resolve_usize_range_param(req.params, "max_period", sweep.max_period, indicator)?;
            sweep.devtype =
                resolve_usize_range_param(req.params, "devtype", sweep.devtype, indicator)?;
            let mut cuda = crate::cuda::moving_averages::CudaVlma::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .vlma_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.buf,
                    rows: result.rows,
                    cols: result.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "vosc" => Some((|| {
            let indicator = "vosc";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::vosc::VoscBatchRange = Default::default();
            sweep.short_period = resolve_usize_range_param(
                req.params,
                "short_period",
                sweep.short_period,
                indicator,
            )?;
            sweep.long_period =
                resolve_usize_range_param(req.params, "long_period", sweep.long_period, indicator)?;
            let mut cuda = crate::cuda::CudaVosc::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .vosc_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.buf,
                    rows: result.0.rows,
                    cols: result.0.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "voss" => Some((|| {
            let indicator = "voss";
            let fallback_outputs: &[&str] = &["output_0", "output_1"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::voss::VossBatchRange = Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            sweep.predict =
                resolve_usize_range_param(req.params, "predict", sweep.predict, indicator)?;
            sweep.bandwidth =
                resolve_f64_range_param(req.params, "bandwidth", sweep.bandwidth, indicator)?;
            let mut cuda = crate::cuda::CudaVoss::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .voss_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.buf,
                    rows: result.0.rows,
                    cols: result.0.cols,
                },
                1 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.1.buf,
                    rows: result.1.rows,
                    cols: result.1.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "vpci" => Some((|| {
            let indicator = "vpci";
            let fallback_outputs: &[&str] = &["a", "b"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let close_f32 = required_close_f32(req.data, indicator)?;
            let volume_f32 = required_volume_f32(req.data, indicator)?;
            let mut sweep: crate::indicators::vpci::VpciBatchRange = Default::default();
            sweep.short_range =
                resolve_usize_range_param(req.params, "short_range", sweep.short_range, indicator)?;
            sweep.long_range =
                resolve_usize_range_param(req.params, "long_range", sweep.long_range, indicator)?;
            let mut cuda = crate::cuda::CudaVpci::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .vpci_batch_dev(close_f32.as_slice(), volume_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.a.buf,
                    rows: result.0.a.rows,
                    cols: result.0.a.cols,
                },
                1 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.b.buf,
                    rows: result.0.b.rows,
                    cols: result.0.b.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "vpt" => Some((|| {
            let indicator = "vpt";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let volume_f32 = required_volume_f32(req.data, indicator)?;
            let mut cuda = crate::cuda::CudaVpt::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .vpt_batch_dev(primary_f32.as_slice(), volume_f32.as_slice())
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.buf,
                    rows: result.rows,
                    cols: result.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "vwmacd" => Some((|| {
            let indicator = "vwmacd";
            let fallback_outputs: &[&str] = &["macd", "signal", "hist"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let volume_f32 = required_volume_f32(req.data, indicator)?;
            let mut sweep: crate::indicators::vwmacd::VwmacdBatchRange = Default::default();
            sweep.fast = resolve_usize_range_param(req.params, "fast", sweep.fast, indicator)?;
            sweep.slow = resolve_usize_range_param(req.params, "slow", sweep.slow, indicator)?;
            sweep.signal =
                resolve_usize_range_param(req.params, "signal", sweep.signal, indicator)?;
            let mut cuda = crate::cuda::CudaVwmacd::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .vwmacd_batch_dev(primary_f32.as_slice(), volume_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.macd.buf,
                    rows: result.0.macd.rows,
                    cols: result.0.macd.cols,
                },
                1 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.signal.buf,
                    rows: result.0.signal.rows,
                    cols: result.0.signal.cols,
                },
                2 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.hist.buf,
                    rows: result.0.hist.rows,
                    cols: result.0.hist.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "wad" => Some((|| {
            let indicator = "wad";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let high_f32 = required_high_f32(req.data, indicator)?;
            let low_f32 = required_low_f32(req.data, indicator)?;
            let close_f32 = required_close_f32(req.data, indicator)?;
            let mut cuda = crate::cuda::CudaWad::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .wad_batch_dev(
                    high_f32.as_slice(),
                    low_f32.as_slice(),
                    close_f32.as_slice(),
                )
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.buf,
                    rows: result.rows,
                    cols: result.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "wavetrend" => Some((|| {
            let indicator = "wavetrend";
            let fallback_outputs: &[&str] = &["wt1", "wt2", "wt_diff"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::wavetrend::WavetrendBatchRange = Default::default();
            sweep.channel_length = resolve_usize_range_param(
                req.params,
                "channel_length",
                sweep.channel_length,
                indicator,
            )?;
            sweep.average_length = resolve_usize_range_param(
                req.params,
                "average_length",
                sweep.average_length,
                indicator,
            )?;
            sweep.ma_length =
                resolve_usize_range_param(req.params, "ma_length", sweep.ma_length, indicator)?;
            sweep.factor = resolve_f64_range_param(req.params, "factor", sweep.factor, indicator)?;
            let mut cuda = crate::cuda::wavetrend::CudaWavetrend::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .wavetrend_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.wt1.buf,
                    rows: result.wt1.rows,
                    cols: result.wt1.cols,
                },
                1 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.wt2.buf,
                    rows: result.wt2.rows,
                    cols: result.wt2.cols,
                },
                2 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.wt_diff.buf,
                    rows: result.wt_diff.rows,
                    cols: result.wt_diff.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "wclprice" => Some((|| {
            let indicator = "wclprice";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let high_f32 = required_high_f32(req.data, indicator)?;
            let low_f32 = required_low_f32(req.data, indicator)?;
            let close_f32 = required_close_f32(req.data, indicator)?;
            let mut sweep: crate::indicators::wclprice::WclpriceBatchRange = Default::default();
            let mut cuda =
                crate::cuda::moving_averages::CudaWclprice::new(device_id).map_err(|e| {
                    IndicatorDispatchError::KernelUnavailable {
                        details: e.to_string(),
                    }
                })?;
            let result = cuda
                .wclprice_batch_dev(
                    high_f32.as_slice(),
                    low_f32.as_slice(),
                    close_f32.as_slice(),
                    &sweep,
                )
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.buf,
                    rows: result.rows,
                    cols: result.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "willr" => Some((|| {
            let indicator = "willr";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let high_f32 = required_high_f32(req.data, indicator)?;
            let low_f32 = required_low_f32(req.data, indicator)?;
            let close_f32 = required_close_f32(req.data, indicator)?;
            let mut sweep: crate::indicators::willr::WillrBatchRange = Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            let mut cuda = crate::cuda::oscillators::CudaWillr::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .willr_batch_dev(
                    high_f32.as_slice(),
                    low_f32.as_slice(),
                    close_f32.as_slice(),
                    &sweep,
                )
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.buf,
                    rows: result.rows,
                    cols: result.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "wto" => Some((|| {
            let indicator = "wto";
            let fallback_outputs: &[&str] = &["wt1", "wt2", "hist"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::wto::WtoBatchRange = Default::default();
            sweep.channel =
                resolve_usize_range_param(req.params, "channel", sweep.channel, indicator)?;
            sweep.average =
                resolve_usize_range_param(req.params, "average", sweep.average, indicator)?;
            let mut cuda = crate::cuda::CudaWto::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .wto_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.outputs.wt1.buf,
                    rows: result.outputs.wt1.rows,
                    cols: result.outputs.wt1.cols,
                },
                1 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.outputs.wt2.buf,
                    rows: result.outputs.wt2.rows,
                    cols: result.outputs.wt2.cols,
                },
                2 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.outputs.hist.buf,
                    rows: result.outputs.hist.rows,
                    cols: result.outputs.hist.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "yang_zhang_volatility" => Some((|| {
            let indicator = "yang_zhang_volatility";
            let fallback_outputs: &[&str] = &["yz", "rs"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let open_f32 = required_open_f32(req.data, indicator)?;
            let high_f32 = required_high_f32(req.data, indicator)?;
            let low_f32 = required_low_f32(req.data, indicator)?;
            let close_f32 = required_close_f32(req.data, indicator)?;
            let mut sweep: crate::indicators::yang_zhang_volatility::YangZhangVolatilityBatchRange =
                Default::default();
            sweep.lookback =
                resolve_usize_range_param(req.params, "lookback", sweep.lookback, indicator)?;
            if let Some(v) = get_bool_param(req.params, "k_override", indicator)? {
                sweep.k_override = v;
            }
            sweep.k = resolve_f64_range_param(req.params, "k", sweep.k, indicator)?;
            let mut cuda = crate::cuda::CudaYangZhangVolatility::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .yang_zhang_volatility_batch_dev(
                    open_f32.as_slice(),
                    high_f32.as_slice(),
                    low_f32.as_slice(),
                    close_f32.as_slice(),
                    &sweep,
                )
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.outputs.yz.buf,
                    rows: result.outputs.yz.rows,
                    cols: result.outputs.yz.cols,
                },
                1 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.outputs.rs.buf,
                    rows: result.outputs.rs.rows,
                    cols: result.outputs.rs.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "garman_klass_volatility" => Some((|| {
            let indicator = "garman_klass_volatility";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let open_f32 = required_open_f32(req.data, indicator)?;
            let high_f32 = required_high_f32(req.data, indicator)?;
            let low_f32 = required_low_f32(req.data, indicator)?;
            let close_f32 = required_close_f32(req.data, indicator)?;
            let mut sweep: crate::indicators::garman_klass_volatility::GarmanKlassVolatilityBatchRange =
                Default::default();
            sweep.lookback =
                resolve_usize_range_param(req.params, "lookback", sweep.lookback, indicator)?;
            let cuda = crate::cuda::CudaGarmanKlassVolatility::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .garman_klass_volatility_batch_dev(
                    open_f32.as_slice(),
                    high_f32.as_slice(),
                    low_f32.as_slice(),
                    close_f32.as_slice(),
                    &sweep,
                )
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.outputs.buf,
                    rows: result.outputs.rows,
                    cols: result.outputs.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        "zscore" => Some((|| {
            let indicator = "zscore";
            let fallback_outputs: &[&str] = &["value"];
            let (output_id, output_index) =
                resolve_output_with_fallback(indicator, info, req.output_id, fallback_outputs)?;
            let primary_f32 = primary_f32_from_data(req.data, indicator)?;
            let mut sweep: crate::indicators::zscore::ZscoreBatchRange = Default::default();
            sweep.period =
                resolve_usize_range_param(req.params, "period", sweep.period, indicator)?;
            sweep.nbdev = resolve_f64_range_param(req.params, "nbdev", sweep.nbdev, indicator)?;
            sweep.devtype =
                resolve_usize_range_param(req.params, "devtype", sweep.devtype, indicator)?;
            let mut cuda = crate::cuda::CudaZscore::new(device_id).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            let result = cuda
                .zscore_batch_dev(primary_f32.as_slice(), &sweep)
                .map_err(|e| map_non_ma_compute_error(indicator, e.to_string()))?;
            let owner = match output_index {
                0 => crate::cuda::moving_averages::DeviceArrayF32 {
                    buf: result.0.buf,
                    rows: result.0.rows,
                    cols: result.0.cols,
                },
                _ => {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: indicator.to_string(),
                        output: output_id.clone(),
                    });
                }
            };
            finalize_cuda_matrix_output(output_id, owner, req.target, device_id as u32, None)
        })()),
        _ => None,
    }
}

fn finalize_cuda_matrix_output(
    output_id: String,
    matrix: crate::cuda::moving_averages::DeviceArrayF32,
    target: CudaOutputTarget,
    device_id: u32,
    warmup: Option<usize>,
) -> Result<IndicatorCudaOutput, IndicatorDispatchError> {
    let rows = matrix.rows;
    let cols = matrix.cols;
    match target {
        CudaOutputTarget::DeviceF32 => Ok(IndicatorCudaOutput {
            output_id,
            series: IndicatorCudaSeries::DeviceF32(DeviceMatrixF32::from_owned(matrix, device_id)),
            warmup,
            rows,
            cols,
            pattern_ids: None,
        }),
        CudaOutputTarget::HostF32 => {
            let mut host = vec![0.0f32; rows.saturating_mul(cols)];
            matrix.buf.copy_to(host.as_mut_slice()).map_err(|e| {
                IndicatorDispatchError::KernelUnavailable {
                    details: e.to_string(),
                }
            })?;
            Ok(IndicatorCudaOutput {
                output_id,
                series: IndicatorCudaSeries::HostF32(host),
                warmup,
                rows,
                cols,
                pattern_ids: None,
            })
        }
    }
}

fn map_non_ma_compute_error(indicator: &str, details: String) -> IndicatorDispatchError {
    let lower = details.to_ascii_lowercase();
    if lower.contains("invalid input") || lower.contains("invalid param") {
        return IndicatorDispatchError::InvalidParam {
            indicator: indicator.to_string(),
            key: "params".to_string(),
            reason: details,
        };
    }
    if lower.contains("out of memory") || lower.contains("cuda") || lower.contains("kernel") {
        return IndicatorDispatchError::KernelUnavailable { details };
    }
    IndicatorDispatchError::ComputeFailed {
        indicator: indicator.to_string(),
        details,
    }
}

fn resolve_output_with_fallback(
    indicator: &str,
    info: Option<&IndicatorInfo>,
    requested: Option<&str>,
    fallback: &[&str],
) -> Result<(String, usize), IndicatorDispatchError> {
    if let Some(info) = info {
        if info.outputs.is_empty() {
            return Err(IndicatorDispatchError::ComputeFailed {
                indicator: info.id.to_string(),
                details: "indicator has no registered outputs".to_string(),
            });
        }

        let resolved = if info.outputs.len() == 1 {
            let only = info.outputs[0].id;
            if let Some(req) = requested {
                if !req.eq_ignore_ascii_case(only) {
                    return Err(IndicatorDispatchError::UnknownOutput {
                        indicator: info.id.to_string(),
                        output: req.to_string(),
                    });
                }
            }
            only
        } else {
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
                })?
        };

        let idx = info
            .outputs
            .iter()
            .position(|o| o.id.eq_ignore_ascii_case(resolved))
            .ok_or_else(|| IndicatorDispatchError::UnknownOutput {
                indicator: info.id.to_string(),
                output: resolved.to_string(),
            })?;
        return Ok((resolved.to_string(), idx));
    }

    if fallback.is_empty() {
        return Err(IndicatorDispatchError::ComputeFailed {
            indicator: indicator.to_string(),
            details: "indicator has no output mapping".to_string(),
        });
    }

    if fallback.len() == 1 {
        let only = fallback[0];
        if let Some(req) = requested {
            if !req.eq_ignore_ascii_case(only) {
                return Err(IndicatorDispatchError::UnknownOutput {
                    indicator: indicator.to_string(),
                    output: req.to_string(),
                });
            }
        }
        return Ok((only.to_string(), 0));
    }

    let req = requested.ok_or_else(|| IndicatorDispatchError::InvalidParam {
        indicator: indicator.to_string(),
        key: "output_id".to_string(),
        reason: "output_id is required for multi-output indicators".to_string(),
    })?;

    let idx = fallback
        .iter()
        .position(|id| id.eq_ignore_ascii_case(req))
        .ok_or_else(|| IndicatorDispatchError::UnknownOutput {
            indicator: indicator.to_string(),
            output: req.to_string(),
        })?;
    Ok((fallback[idx].to_string(), idx))
}

enum F32Input<'a> {
    Borrowed(&'a [f32]),
    Owned(Vec<f32>),
}

impl<'a> F32Input<'a> {
    fn as_slice(&self) -> &[f32] {
        match self {
            Self::Borrowed(v) => v,
            Self::Owned(v) => v.as_slice(),
        }
    }
}

fn primary_source_is_close(data: IndicatorCudaDataRef<'_>) -> bool {
    match data {
        IndicatorCudaDataRef::Slice { .. } => true,
        IndicatorCudaDataRef::Ohlc { close, source, .. } => source
            .map(|s| s.len() == close.len() && std::ptr::eq(s.as_ptr(), close.as_ptr()))
            .unwrap_or(true),
        IndicatorCudaDataRef::Ohlcv { close, source, .. } => source
            .map(|s| s.len() == close.len() && std::ptr::eq(s.as_ptr(), close.as_ptr()))
            .unwrap_or(true),
        IndicatorCudaDataRef::CloseVolume { .. } => true,
        IndicatorCudaDataRef::HighLow { .. } => false,
    }
}

fn primary_f32_from_data<'a>(
    data: IndicatorCudaDataRef<'a>,
    _indicator: &str,
) -> Result<F32Input<'a>, IndicatorDispatchError> {
    match data {
        IndicatorCudaDataRef::Slice { values } => Ok(F32Input::Borrowed(values)),
        IndicatorCudaDataRef::Ohlc { close, source, .. } => {
            Ok(F32Input::Borrowed(source.unwrap_or(close)))
        }
        IndicatorCudaDataRef::Ohlcv { close, source, .. } => {
            Ok(F32Input::Borrowed(source.unwrap_or(close)))
        }
        IndicatorCudaDataRef::CloseVolume { close, .. } => Ok(F32Input::Borrowed(close)),
        IndicatorCudaDataRef::HighLow { high, low } => {
            if high.len() != low.len() {
                return Err(IndicatorDispatchError::DataLengthMismatch {
                    details: format!("high={} low={}", high.len(), low.len()),
                });
            }
            Ok(F32Input::Owned(
                high.iter()
                    .zip(low.iter())
                    .map(|(&h, &l)| (h + l) * 0.5)
                    .collect(),
            ))
        }
    }
}

fn required_open_slice<'a>(
    data: IndicatorCudaDataRef<'a>,
    indicator: &str,
) -> Result<&'a [f32], IndicatorDispatchError> {
    match data {
        IndicatorCudaDataRef::Ohlc { open, .. } => Ok(open),
        IndicatorCudaDataRef::Ohlcv { open, .. } => Ok(open),
        _ => Err(IndicatorDispatchError::MissingRequiredInput {
            indicator: indicator.to_string(),
            input: IndicatorInputKind::Ohlc,
        }),
    }
}

fn required_high_slice<'a>(
    data: IndicatorCudaDataRef<'a>,
    indicator: &str,
) -> Result<&'a [f32], IndicatorDispatchError> {
    match data {
        IndicatorCudaDataRef::Ohlc { high, .. } => Ok(high),
        IndicatorCudaDataRef::Ohlcv { high, .. } => Ok(high),
        IndicatorCudaDataRef::HighLow { high, .. } => Ok(high),
        _ => Err(IndicatorDispatchError::MissingRequiredInput {
            indicator: indicator.to_string(),
            input: IndicatorInputKind::HighLow,
        }),
    }
}

fn required_low_slice<'a>(
    data: IndicatorCudaDataRef<'a>,
    indicator: &str,
) -> Result<&'a [f32], IndicatorDispatchError> {
    match data {
        IndicatorCudaDataRef::Ohlc { low, .. } => Ok(low),
        IndicatorCudaDataRef::Ohlcv { low, .. } => Ok(low),
        IndicatorCudaDataRef::HighLow { low, .. } => Ok(low),
        _ => Err(IndicatorDispatchError::MissingRequiredInput {
            indicator: indicator.to_string(),
            input: IndicatorInputKind::HighLow,
        }),
    }
}

fn required_close_slice<'a>(
    data: IndicatorCudaDataRef<'a>,
    indicator: &str,
) -> Result<&'a [f32], IndicatorDispatchError> {
    match data {
        IndicatorCudaDataRef::Slice { values } => Ok(values),
        IndicatorCudaDataRef::Ohlc { close, .. } => Ok(close),
        IndicatorCudaDataRef::Ohlcv { close, .. } => Ok(close),
        IndicatorCudaDataRef::CloseVolume { close, .. } => Ok(close),
        _ => Err(IndicatorDispatchError::MissingRequiredInput {
            indicator: indicator.to_string(),
            input: IndicatorInputKind::Slice,
        }),
    }
}

fn required_volume_slice<'a>(
    data: IndicatorCudaDataRef<'a>,
    indicator: &str,
) -> Result<&'a [f32], IndicatorDispatchError> {
    match data {
        IndicatorCudaDataRef::Ohlcv { volume, .. } => Ok(volume),
        IndicatorCudaDataRef::CloseVolume { volume, .. } => Ok(volume),
        _ => Err(IndicatorDispatchError::MissingRequiredInput {
            indicator: indicator.to_string(),
            input: IndicatorInputKind::CloseVolume,
        }),
    }
}

fn required_open_f32<'a>(
    data: IndicatorCudaDataRef<'a>,
    indicator: &str,
) -> Result<F32Input<'a>, IndicatorDispatchError> {
    Ok(F32Input::Borrowed(required_open_slice(data, indicator)?))
}

fn required_high_f32<'a>(
    data: IndicatorCudaDataRef<'a>,
    indicator: &str,
) -> Result<F32Input<'a>, IndicatorDispatchError> {
    Ok(F32Input::Borrowed(required_high_slice(data, indicator)?))
}

fn required_low_f32<'a>(
    data: IndicatorCudaDataRef<'a>,
    indicator: &str,
) -> Result<F32Input<'a>, IndicatorDispatchError> {
    Ok(F32Input::Borrowed(required_low_slice(data, indicator)?))
}

fn required_close_f32<'a>(
    data: IndicatorCudaDataRef<'a>,
    indicator: &str,
) -> Result<F32Input<'a>, IndicatorDispatchError> {
    Ok(F32Input::Borrowed(required_close_slice(data, indicator)?))
}

fn required_volume_f32<'a>(
    data: IndicatorCudaDataRef<'a>,
    indicator: &str,
) -> Result<F32Input<'a>, IndicatorDispatchError> {
    Ok(F32Input::Borrowed(required_volume_slice(data, indicator)?))
}

fn optional_volume_f32(data: IndicatorCudaDataRef<'_>) -> Option<F32Input<'_>> {
    match data {
        IndicatorCudaDataRef::Ohlcv { volume, .. } => Some(F32Input::Borrowed(volume)),
        IndicatorCudaDataRef::CloseVolume { volume, .. } => Some(F32Input::Borrowed(volume)),
        _ => None,
    }
}

fn get_param<'a>(params: &'a [ParamKV<'a>], key: &str) -> Option<&'a ParamKV<'a>> {
    params.iter().find(|p| p.key.eq_ignore_ascii_case(key))
}

fn get_usize_param(
    params: &[ParamKV<'_>],
    key: &str,
    indicator: &str,
) -> Result<Option<usize>, IndicatorDispatchError> {
    let Some(item) = get_param(params, key) else {
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
            let r = v.round();
            if (v - r).abs() > 1e-9 {
                return Err(IndicatorDispatchError::InvalidParam {
                    indicator: indicator.to_string(),
                    key: key.to_string(),
                    reason: format!("expected whole number, got {}", v),
                });
            }
            Ok(Some(r as usize))
        }
        ParamValue::Bool(v) => Ok(Some(if v { 1 } else { 0 })),
        ParamValue::EnumString(v) => Err(IndicatorDispatchError::InvalidParam {
            indicator: indicator.to_string(),
            key: key.to_string(),
            reason: format!("expected integer-compatible value, got '{}'", v),
        }),
    }
}

fn get_f64_param(
    params: &[ParamKV<'_>],
    key: &str,
    indicator: &str,
) -> Result<Option<f64>, IndicatorDispatchError> {
    let Some(item) = get_param(params, key) else {
        return Ok(None);
    };
    match item.value {
        ParamValue::Int(v) => Ok(Some(v as f64)),
        ParamValue::Float(v) => {
            if !v.is_finite() {
                return Err(IndicatorDispatchError::InvalidParam {
                    indicator: indicator.to_string(),
                    key: key.to_string(),
                    reason: "expected finite float".to_string(),
                });
            }
            Ok(Some(v))
        }
        ParamValue::Bool(v) => Ok(Some(if v { 1.0 } else { 0.0 })),
        ParamValue::EnumString(v) => Err(IndicatorDispatchError::InvalidParam {
            indicator: indicator.to_string(),
            key: key.to_string(),
            reason: format!("expected numeric value, got '{}'", v),
        }),
    }
}

fn get_f32_param(
    params: &[ParamKV<'_>],
    key: &str,
    indicator: &str,
) -> Result<Option<f32>, IndicatorDispatchError> {
    Ok(get_f64_param(params, key, indicator)?.map(|v| v as f32))
}

fn get_bool_param(
    params: &[ParamKV<'_>],
    key: &str,
    indicator: &str,
) -> Result<Option<bool>, IndicatorDispatchError> {
    let Some(item) = get_param(params, key) else {
        return Ok(None);
    };
    match item.value {
        ParamValue::Bool(v) => Ok(Some(v)),
        ParamValue::Int(v) => Ok(Some(v != 0)),
        ParamValue::Float(v) => {
            if !v.is_finite() {
                return Err(IndicatorDispatchError::InvalidParam {
                    indicator: indicator.to_string(),
                    key: key.to_string(),
                    reason: "expected finite bool-compatible value".to_string(),
                });
            }
            Ok(Some(v != 0.0))
        }
        ParamValue::EnumString(v) => {
            if v.eq_ignore_ascii_case("true") {
                return Ok(Some(true));
            }
            if v.eq_ignore_ascii_case("false") {
                return Ok(Some(false));
            }
            Err(IndicatorDispatchError::InvalidParam {
                indicator: indicator.to_string(),
                key: key.to_string(),
                reason: format!("expected bool value, got '{}'", v),
            })
        }
    }
}

fn get_string_param<'a>(params: &'a [ParamKV<'a>], key: &str) -> Option<&'a str> {
    let item = get_param(params, key)?;
    match item.value {
        ParamValue::EnumString(v) => Some(v),
        _ => None,
    }
}

fn resolve_is_long(
    params: &[ParamKV<'_>],
    indicator: &str,
) -> Result<bool, IndicatorDispatchError> {
    if let Some(v) = get_bool_param(params, "is_long", indicator)? {
        return Ok(v);
    }
    if let Some(v) = get_string_param(params, "direction") {
        return Ok(v.eq_ignore_ascii_case("long"));
    }
    Ok(true)
}

fn resolve_usize_range_param(
    params: &[ParamKV<'_>],
    key: &str,
    default: (usize, usize, usize),
    indicator: &str,
) -> Result<(usize, usize, usize), IndicatorDispatchError> {
    let mut out = default;
    if let Some(v) = get_usize_param(params, key, indicator)? {
        out = (v, v, 0);
    }
    let k_start = format!("{}_start", key);
    let k_end = format!("{}_end", key);
    let k_step = format!("{}_step", key);
    if let Some(v) = get_usize_param(params, k_start.as_str(), indicator)? {
        out.0 = v;
    }
    if let Some(v) = get_usize_param(params, k_end.as_str(), indicator)? {
        out.1 = v;
    }
    if let Some(v) = get_usize_param(params, k_step.as_str(), indicator)? {
        out.2 = v;
    }
    Ok(out)
}

fn resolve_f64_range_param(
    params: &[ParamKV<'_>],
    key: &str,
    default: (f64, f64, f64),
    indicator: &str,
) -> Result<(f64, f64, f64), IndicatorDispatchError> {
    let mut out = default;
    if let Some(v) = get_f64_param(params, key, indicator)? {
        out = (v, v, 0.0);
    }
    let k_start = format!("{}_start", key);
    let k_end = format!("{}_end", key);
    let k_step = format!("{}_step", key);
    if let Some(v) = get_f64_param(params, k_start.as_str(), indicator)? {
        out.0 = v;
    }
    if let Some(v) = get_f64_param(params, k_end.as_str(), indicator)? {
        out.1 = v;
    }
    if let Some(v) = get_f64_param(params, k_step.as_str(), indicator)? {
        out.2 = v;
    }
    Ok(out)
}

fn resolve_bool_range_param(
    params: &[ParamKV<'_>],
    key: &str,
    default: (bool, bool, bool),
    indicator: &str,
) -> Result<(bool, bool, bool), IndicatorDispatchError> {
    let mut out = default;
    if let Some(v) = get_bool_param(params, key, indicator)? {
        out = (v, v, false);
    }
    let k_start = format!("{}_start", key);
    let k_end = format!("{}_end", key);
    let k_step = format!("{}_step", key);
    if let Some(v) = get_bool_param(params, k_start.as_str(), indicator)? {
        out.0 = v;
    }
    if let Some(v) = get_bool_param(params, k_end.as_str(), indicator)? {
        out.1 = v;
    }
    if let Some(v) = get_bool_param(params, k_step.as_str(), indicator)? {
        out.2 = v;
    }
    Ok(out)
}

fn resolve_string_range_param(
    params: &[ParamKV<'_>],
    key: &str,
    default: (String, String, usize),
    indicator: &str,
) -> Result<(String, String, usize), IndicatorDispatchError> {
    let mut out = default;
    if let Some(v) = get_string_param(params, key) {
        out = (v.to_string(), v.to_string(), out.2.max(1));
    }
    let k_start = format!("{}_start", key);
    let k_end = format!("{}_end", key);
    let k_step = format!("{}_step", key);
    if let Some(v) = get_string_param(params, k_start.as_str()) {
        out.0 = v.to_string();
    }
    if let Some(v) = get_string_param(params, k_end.as_str()) {
        out.1 = v.to_string();
    }
    if let Some(v) = get_usize_param(params, k_step.as_str(), indicator)? {
        out.2 = v;
    }
    if out.2 == 0 {
        out.2 = 1;
    }
    Ok(out)
}
