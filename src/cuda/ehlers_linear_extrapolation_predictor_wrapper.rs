#![cfg(feature = "cuda")]

use crate::indicators::ehlers_linear_extrapolation_predictor::{
    EhlersLinearExtrapolationPredictorBatchRange, EhlersLinearExtrapolationPredictorParams,
};
use cust::context::Context;
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, GridSize};
use cust::launch;
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::Module;
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::sync::Arc;
use thiserror::Error;

const EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const DEFAULT_HIGH_PASS_LENGTH: usize = 125;
const DEFAULT_LOW_PASS_LENGTH: usize = 12;
const DEFAULT_GAIN: f64 = 0.7;
const DEFAULT_BARS_FORWARD: usize = 5;
const DEFAULT_SIGNAL_MODE: &str = "predict_filter_crosses";
const MAX_BARS_FORWARD: usize = 10;
const FLOAT_TOL: f64 = 1e-12;

const SIGNAL_MODE_PREDICT_FILTER_CROSSES: i32 = 0;
const SIGNAL_MODE_PREDICT_MIDDLE_CROSSES: i32 = 1;
const SIGNAL_MODE_FILTER_MIDDLE_CROSSES: i32 = 2;

#[derive(Debug, Error)]
pub enum CudaEhlersLinearExtrapolationPredictorError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("out of memory: required={required} free={free} headroom={headroom}")]
    OutOfMemory {
        required: usize,
        free: usize,
        headroom: usize,
    },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge {
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
    },
}

pub struct EhlersLinearExtrapolationPredictorDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl EhlersLinearExtrapolationPredictorDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct EhlersLinearExtrapolationPredictorDeviceArrayF64Quint {
    pub prediction: EhlersLinearExtrapolationPredictorDeviceArrayF64,
    pub filter: EhlersLinearExtrapolationPredictorDeviceArrayF64,
    pub state: EhlersLinearExtrapolationPredictorDeviceArrayF64,
    pub go_long: EhlersLinearExtrapolationPredictorDeviceArrayF64,
    pub go_short: EhlersLinearExtrapolationPredictorDeviceArrayF64,
}

impl EhlersLinearExtrapolationPredictorDeviceArrayF64Quint {
    #[inline]
    pub fn rows(&self) -> usize {
        self.prediction.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.prediction.cols
    }
}

pub struct CudaEhlersLinearExtrapolationPredictorBatchResult {
    pub outputs: EhlersLinearExtrapolationPredictorDeviceArrayF64Quint,
    pub combos: Vec<EhlersLinearExtrapolationPredictorParams>,
}

pub struct CudaEhlersLinearExtrapolationPredictor {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

fn normalize_signal_mode_name(value: Option<&str>) -> String {
    value
        .unwrap_or(DEFAULT_SIGNAL_MODE)
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_lowercase())
        .collect::<String>()
}

fn parse_signal_mode(
    value: Option<&str>,
) -> Result<(i32, String), CudaEhlersLinearExtrapolationPredictorError> {
    let normalized = normalize_signal_mode_name(value);
    match normalized.as_str() {
        "predictfiltercrosses" | "sm02" => Ok((
            SIGNAL_MODE_PREDICT_FILTER_CROSSES,
            "predict_filter_crosses".to_string(),
        )),
        "predictmiddlecrosses" | "sm03" => Ok((
            SIGNAL_MODE_PREDICT_MIDDLE_CROSSES,
            "predict_middle_crosses".to_string(),
        )),
        "filtermiddlecrosses" | "sm04" => Ok((
            SIGNAL_MODE_FILTER_MIDDLE_CROSSES,
            "filter_middle_crosses".to_string(),
        )),
        _ => Err(CudaEhlersLinearExtrapolationPredictorError::InvalidInput(
            format!(
                "invalid signal_mode: {}",
                value.unwrap_or(DEFAULT_SIGNAL_MODE)
            ),
        )),
    }
}

fn count_valid_values(data: &[f64]) -> usize {
    data.iter().filter(|value| value.is_finite()).count()
}

fn warmup_period(low_pass_length: usize) -> usize {
    low_pass_length + 11
}

fn expand_axis_usize(
    start: usize,
    end: usize,
    step: usize,
) -> Result<Vec<usize>, CudaEhlersLinearExtrapolationPredictorError> {
    if start == end {
        return Ok(vec![start]);
    }
    if step == 0 {
        return Err(CudaEhlersLinearExtrapolationPredictorError::InvalidInput(
            format!("invalid range: start={start}, end={end}, step={step}"),
        ));
    }

    let mut out = Vec::new();
    if start < end {
        let mut value = start;
        loop {
            out.push(value);
            if value == end {
                break;
            }
            let next = value.saturating_add(step);
            if next == value || next > end {
                break;
            }
            value = next;
        }
    } else {
        let mut value = start;
        loop {
            out.push(value);
            if value == end {
                break;
            }
            let next = value.saturating_sub(step);
            if next == value || next < end {
                break;
            }
            value = next;
        }
    }

    if out.is_empty() {
        return Err(CudaEhlersLinearExtrapolationPredictorError::InvalidInput(
            format!("invalid range: start={start}, end={end}, step={step}"),
        ));
    }
    Ok(out)
}

fn expand_axis_f64(
    start: f64,
    end: f64,
    step: f64,
) -> Result<Vec<f64>, CudaEhlersLinearExtrapolationPredictorError> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() {
        return Err(CudaEhlersLinearExtrapolationPredictorError::InvalidInput(
            format!("invalid range: start={start}, end={end}, step={step}"),
        ));
    }
    if (start - end).abs() < FLOAT_TOL || step.abs() < FLOAT_TOL {
        return Ok(vec![start]);
    }

    let mut out = Vec::new();
    if start < end {
        let st = step.abs();
        let mut value = start;
        while value <= end + FLOAT_TOL {
            out.push(value);
            value += st;
        }
    } else {
        let st = -step.abs();
        let mut value = start;
        while value >= end - FLOAT_TOL {
            out.push(value);
            value += st;
        }
    }

    if out.is_empty() {
        return Err(CudaEhlersLinearExtrapolationPredictorError::InvalidInput(
            format!("invalid range: start={start}, end={end}, step={step}"),
        ));
    }
    Ok(out)
}

fn expand_grid_ehlers_linear_extrapolation_predictor(
    sweep: &EhlersLinearExtrapolationPredictorBatchRange,
) -> Result<
    Vec<EhlersLinearExtrapolationPredictorParams>,
    CudaEhlersLinearExtrapolationPredictorError,
> {
    let high_pass_lengths = expand_axis_usize(
        sweep.high_pass_length.0,
        sweep.high_pass_length.1,
        sweep.high_pass_length.2,
    )?;
    let low_pass_lengths = expand_axis_usize(
        sweep.low_pass_length.0,
        sweep.low_pass_length.1,
        sweep.low_pass_length.2,
    )?;
    let gains = expand_axis_f64(sweep.gain.0, sweep.gain.1, sweep.gain.2)?;
    let bars_forwards = expand_axis_usize(
        sweep.bars_forward.0,
        sweep.bars_forward.1,
        sweep.bars_forward.2,
    )?;
    let (_, signal_mode_name) = parse_signal_mode(sweep.signal_mode.as_deref())?;

    let mut combos = Vec::with_capacity(
        high_pass_lengths
            .len()
            .saturating_mul(low_pass_lengths.len())
            .saturating_mul(gains.len())
            .saturating_mul(bars_forwards.len()),
    );

    for high_pass_length in high_pass_lengths {
        for &low_pass_length in &low_pass_lengths {
            for &gain in &gains {
                for bars_forward in bars_forwards.iter().copied() {
                    combos.push(EhlersLinearExtrapolationPredictorParams {
                        high_pass_length: Some(high_pass_length),
                        low_pass_length: Some(low_pass_length),
                        gain: Some(gain),
                        bars_forward: Some(bars_forward),
                        signal_mode: Some(signal_mode_name.clone()),
                    });
                }
            }
        }
    }

    Ok(combos)
}

impl CudaEhlersLinearExtrapolationPredictor {
    pub fn new(device_id: usize) -> Result<Self, CudaEhlersLinearExtrapolationPredictorError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module =
            crate::load_cuda_embedded_module!("ehlers_linear_extrapolation_predictor_kernel")?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        Ok(Self {
            module,
            stream,
            context,
            device_id: device_id as u32,
        })
    }

    #[inline]
    pub fn context_arc(&self) -> Arc<Context> {
        self.context.clone()
    }

    #[inline]
    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    pub fn synchronize(&self) -> Result<(), CudaEhlersLinearExtrapolationPredictorError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn will_fit(
        required: usize,
        headroom: usize,
    ) -> Result<(), CudaEhlersLinearExtrapolationPredictorError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaEhlersLinearExtrapolationPredictorError::OutOfMemory {
                    required,
                    free,
                    headroom,
                });
            }
        }
        Ok(())
    }

    fn validate_launch(
        &self,
        grid: GridSize,
        block: BlockSize,
    ) -> Result<(), CudaEhlersLinearExtrapolationPredictorError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(
                CudaEhlersLinearExtrapolationPredictorError::LaunchConfigTooLarge {
                    gx: grid.x,
                    gy: grid.y,
                    gz: grid.z,
                    bx: block.x,
                    by: block.y,
                    bz: block.z,
                },
            );
        }
        Ok(())
    }

    pub fn batch_dev(
        &self,
        data: &[f64],
        sweep: &EhlersLinearExtrapolationPredictorBatchRange,
    ) -> Result<
        CudaEhlersLinearExtrapolationPredictorBatchResult,
        CudaEhlersLinearExtrapolationPredictorError,
    > {
        if data.is_empty() {
            return Err(CudaEhlersLinearExtrapolationPredictorError::InvalidInput(
                "empty input".into(),
            ));
        }
        if !data.iter().any(|value| value.is_finite()) {
            return Err(CudaEhlersLinearExtrapolationPredictorError::InvalidInput(
                "all values are NaN".into(),
            ));
        }

        let combos = expand_grid_ehlers_linear_extrapolation_predictor(sweep)?;
        if combos.is_empty() {
            return Err(CudaEhlersLinearExtrapolationPredictorError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let valid = count_valid_values(data);
        let rows = combos.len();
        let cols = data.len();
        let mut high_pass_lengths = Vec::with_capacity(rows);
        let mut low_pass_lengths = Vec::with_capacity(rows);
        let mut gains = Vec::with_capacity(rows);
        let mut bars_forwards = Vec::with_capacity(rows);
        let mut signal_modes = Vec::with_capacity(rows);
        let mut max_low_pass_length = 0usize;
        let mut max_needed = 0usize;

        for combo in &combos {
            let high_pass_length = combo.high_pass_length.unwrap_or(DEFAULT_HIGH_PASS_LENGTH);
            let low_pass_length = combo.low_pass_length.unwrap_or(DEFAULT_LOW_PASS_LENGTH);
            let gain = combo.gain.unwrap_or(DEFAULT_GAIN);
            let bars_forward = combo.bars_forward.unwrap_or(DEFAULT_BARS_FORWARD);
            let (signal_mode, _) = parse_signal_mode(combo.signal_mode.as_deref())?;

            if high_pass_length == 0 {
                return Err(CudaEhlersLinearExtrapolationPredictorError::InvalidInput(
                    format!("invalid high_pass_length: {high_pass_length}"),
                ));
            }
            if low_pass_length == 0 {
                return Err(CudaEhlersLinearExtrapolationPredictorError::InvalidInput(
                    format!("invalid low_pass_length: {low_pass_length}"),
                ));
            }
            if !gain.is_finite() {
                return Err(CudaEhlersLinearExtrapolationPredictorError::InvalidInput(
                    format!("invalid gain: {gain}"),
                ));
            }
            if bars_forward > MAX_BARS_FORWARD {
                return Err(CudaEhlersLinearExtrapolationPredictorError::InvalidInput(
                    format!("invalid bars_forward: {bars_forward}"),
                ));
            }

            high_pass_lengths.push(i32::try_from(high_pass_length).map_err(|_| {
                CudaEhlersLinearExtrapolationPredictorError::InvalidInput(format!(
                    "high_pass_length out of range: {high_pass_length}"
                ))
            })?);
            low_pass_lengths.push(i32::try_from(low_pass_length).map_err(|_| {
                CudaEhlersLinearExtrapolationPredictorError::InvalidInput(format!(
                    "low_pass_length out of range: {low_pass_length}"
                ))
            })?);
            gains.push(gain);
            bars_forwards.push(i32::try_from(bars_forward).map_err(|_| {
                CudaEhlersLinearExtrapolationPredictorError::InvalidInput(format!(
                    "bars_forward out of range: {bars_forward}"
                ))
            })?);
            signal_modes.push(signal_mode);
            max_low_pass_length = max_low_pass_length.max(low_pass_length);
            max_needed = max_needed.max(warmup_period(low_pass_length) + 1);
        }

        if valid < max_needed {
            return Err(CudaEhlersLinearExtrapolationPredictorError::InvalidInput(
                format!("not enough valid data: needed={max_needed}, valid={valid}"),
            ));
        }

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaEhlersLinearExtrapolationPredictorError::InvalidInput(
                    "input bytes overflow".into(),
                )
            })?;
        let params_bytes = rows
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| value.checked_mul(4))
            .and_then(|value| {
                rows.checked_mul(std::mem::size_of::<f64>())
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| {
                CudaEhlersLinearExtrapolationPredictorError::InvalidInput(
                    "params bytes overflow".into(),
                )
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaEhlersLinearExtrapolationPredictorError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(5))
            .ok_or_else(|| {
                CudaEhlersLinearExtrapolationPredictorError::InvalidInput(
                    "output bytes overflow".into(),
                )
            })?;
        let scratch_elems = rows.checked_mul(max_low_pass_length).ok_or_else(|| {
            CudaEhlersLinearExtrapolationPredictorError::InvalidInput(
                "scratch elements overflow".into(),
            )
        })?;
        let scratch_bytes = scratch_elems
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaEhlersLinearExtrapolationPredictorError::InvalidInput(
                    "scratch bytes overflow".into(),
                )
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .and_then(|value| value.checked_add(scratch_bytes))
            .ok_or_else(|| {
                CudaEhlersLinearExtrapolationPredictorError::InvalidInput(
                    "required bytes overflow".into(),
                )
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_data = DeviceBuffer::from_slice(data)?;
        let d_high_pass_lengths = DeviceBuffer::from_slice(&high_pass_lengths)?;
        let d_low_pass_lengths = DeviceBuffer::from_slice(&low_pass_lengths)?;
        let d_gains = DeviceBuffer::from_slice(&gains)?;
        let d_bars_forwards = DeviceBuffer::from_slice(&bars_forwards)?;
        let d_signal_modes = DeviceBuffer::from_slice(&signal_modes)?;
        let d_out_prediction = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_filter = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_state = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_go_long = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_go_short = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_hp_history = unsafe { DeviceBuffer::<f64>::uninitialized(scratch_elems)? };

        let func = self
            .module
            .get_function("ehlers_linear_extrapolation_predictor_batch_f64")
            .map_err(
                |_| CudaEhlersLinearExtrapolationPredictorError::MissingKernelSymbol {
                    name: "ehlers_linear_extrapolation_predictor_batch_f64",
                },
            )?;
        let grid_x = ((rows as u32) + EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_BLOCK_X - 1)
            / EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_data.as_device_ptr(),
                i32::try_from(cols).map_err(|_| {
                    CudaEhlersLinearExtrapolationPredictorError::InvalidInput(
                        "cols out of range".into(),
                    )
                })?,
                d_high_pass_lengths.as_device_ptr(),
                d_low_pass_lengths.as_device_ptr(),
                d_gains.as_device_ptr(),
                d_bars_forwards.as_device_ptr(),
                d_signal_modes.as_device_ptr(),
                i32::try_from(rows).map_err(|_| {
                    CudaEhlersLinearExtrapolationPredictorError::InvalidInput(
                        "rows out of range".into(),
                    )
                })?,
                i32::try_from(max_low_pass_length).map_err(|_| {
                    CudaEhlersLinearExtrapolationPredictorError::InvalidInput(
                        "max_low_pass_length out of range".into(),
                    )
                })?,
                d_out_prediction.as_device_ptr(),
                d_out_filter.as_device_ptr(),
                d_out_state.as_device_ptr(),
                d_out_go_long.as_device_ptr(),
                d_out_go_short.as_device_ptr(),
                d_hp_history.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaEhlersLinearExtrapolationPredictorBatchResult {
            outputs: EhlersLinearExtrapolationPredictorDeviceArrayF64Quint {
                prediction: EhlersLinearExtrapolationPredictorDeviceArrayF64 {
                    buf: d_out_prediction,
                    rows,
                    cols,
                },
                filter: EhlersLinearExtrapolationPredictorDeviceArrayF64 {
                    buf: d_out_filter,
                    rows,
                    cols,
                },
                state: EhlersLinearExtrapolationPredictorDeviceArrayF64 {
                    buf: d_out_state,
                    rows,
                    cols,
                },
                go_long: EhlersLinearExtrapolationPredictorDeviceArrayF64 {
                    buf: d_out_go_long,
                    rows,
                    cols,
                },
                go_short: EhlersLinearExtrapolationPredictorDeviceArrayF64 {
                    buf: d_out_go_short,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
