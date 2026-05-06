#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use serde::{Deserialize, Serialize};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen::prelude::*;

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_uninit_f64, detect_best_batch_kernel, init_matrix_prefixes, make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::error::Error;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum VolumeWeightedRsiData<'a> {
    Candles {
        candles: &'a Candles,
        close_source: &'a str,
    },
    Slices {
        close: &'a [f64],
        volume: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct VolumeWeightedRsiOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct VolumeWeightedRsiParams {
    pub period: Option<usize>,
}

impl Default for VolumeWeightedRsiParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct VolumeWeightedRsiInput<'a> {
    pub data: VolumeWeightedRsiData<'a>,
    pub params: VolumeWeightedRsiParams,
}

impl<'a> VolumeWeightedRsiInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        close_source: &'a str,
        params: VolumeWeightedRsiParams,
    ) -> Self {
        Self {
            data: VolumeWeightedRsiData::Candles {
                candles,
                close_source,
            },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        close: &'a [f64],
        volume: &'a [f64],
        params: VolumeWeightedRsiParams,
    ) -> Self {
        Self {
            data: VolumeWeightedRsiData::Slices { close, volume },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "close", VolumeWeightedRsiParams::default())
    }

    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(14)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct VolumeWeightedRsiBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for VolumeWeightedRsiBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl VolumeWeightedRsiBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn period(mut self, value: usize) -> Self {
        self.period = Some(value);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline(always)]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<VolumeWeightedRsiOutput, VolumeWeightedRsiError> {
        let params = VolumeWeightedRsiParams {
            period: self.period,
        };
        volume_weighted_rsi_with_kernel(
            &VolumeWeightedRsiInput::from_candles(candles, "close", params),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        close: &[f64],
        volume: &[f64],
    ) -> Result<VolumeWeightedRsiOutput, VolumeWeightedRsiError> {
        let params = VolumeWeightedRsiParams {
            period: self.period,
        };
        volume_weighted_rsi_with_kernel(
            &VolumeWeightedRsiInput::from_slices(close, volume, params),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<VolumeWeightedRsiStream, VolumeWeightedRsiError> {
        VolumeWeightedRsiStream::try_new(VolumeWeightedRsiParams {
            period: self.period,
        })
    }
}

#[derive(Debug, Error)]
pub enum VolumeWeightedRsiError {
    #[error("volume_weighted_rsi: Input data slice is empty.")]
    EmptyInputData,
    #[error(
        "volume_weighted_rsi: Input length mismatch: close = {close_len}, volume = {volume_len}"
    )]
    InputLengthMismatch { close_len: usize, volume_len: usize },
    #[error("volume_weighted_rsi: All values are NaN.")]
    AllValuesNaN,
    #[error("volume_weighted_rsi: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("volume_weighted_rsi: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("volume_weighted_rsi: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("volume_weighted_rsi: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("volume_weighted_rsi: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error(
        "volume_weighted_rsi: Output length mismatch: dst = {dst_len}, expected = {expected_len}"
    )]
    MismatchedOutputLen { dst_len: usize, expected_len: usize },
    #[error("volume_weighted_rsi: Invalid input: {msg}")]
    InvalidInput { msg: String },
}

#[derive(Debug, Clone)]
pub struct VolumeWeightedRsiStream {
    period: usize,
    inv_period: f64,
    beta: f64,
    prev_close: f64,
    has_prev: bool,
    seeded: usize,
    sum_up: f64,
    sum_down: f64,
    avg_up: f64,
    avg_down: f64,
}

impl VolumeWeightedRsiStream {
    #[inline(always)]
    pub fn try_new(params: VolumeWeightedRsiParams) -> Result<Self, VolumeWeightedRsiError> {
        let period = params.period.unwrap_or(14);
        if period == 0 {
            return Err(VolumeWeightedRsiError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        let inv_period = 1.0 / period as f64;
        Ok(Self {
            period,
            inv_period,
            beta: 1.0 - inv_period,
            prev_close: f64::NAN,
            has_prev: false,
            seeded: 0,
            sum_up: 0.0,
            sum_down: 0.0,
            avg_up: 0.0,
            avg_down: 0.0,
        })
    }

    #[inline(always)]
    pub fn reset(&mut self) {
        self.prev_close = f64::NAN;
        self.has_prev = false;
        self.seeded = 0;
        self.sum_up = 0.0;
        self.sum_down = 0.0;
        self.avg_up = 0.0;
        self.avg_down = 0.0;
    }

    #[inline(always)]
    pub fn update(&mut self, close: f64, volume: f64) -> Option<f64> {
        if !is_valid_pair(close, volume) {
            self.reset();
            return None;
        }

        let (up, down) = if self.has_prev {
            if close > self.prev_close {
                (volume, 0.0)
            } else if close < self.prev_close {
                (0.0, volume)
            } else {
                (0.0, 0.0)
            }
        } else {
            (0.0, 0.0)
        };

        self.prev_close = close;
        self.has_prev = true;

        if self.seeded < self.period {
            self.sum_up += up;
            self.sum_down += down;
            self.seeded += 1;
            if self.seeded < self.period {
                return None;
            }
            self.avg_up = self.sum_up * self.inv_period;
            self.avg_down = self.sum_down * self.inv_period;
            return Some(rsi_from_components(self.avg_up, self.avg_down));
        }

        self.avg_up = self.avg_up.mul_add(self.beta, self.inv_period * up);
        self.avg_down = self.avg_down.mul_add(self.beta, self.inv_period * down);
        Some(rsi_from_components(self.avg_up, self.avg_down))
    }

    #[inline(always)]
    pub fn get_warmup_period(&self) -> usize {
        self.period.saturating_sub(1)
    }
}

#[inline(always)]
fn is_valid_pair(close: f64, volume: f64) -> bool {
    close.is_finite() && volume.is_finite()
}

#[inline(always)]
fn rsi_from_components(avg_up: f64, avg_down: f64) -> f64 {
    let denom = avg_up + avg_down;
    if denom == 0.0 {
        50.0
    } else {
        100.0 * avg_up / denom
    }
}

#[inline(always)]
fn longest_valid_pair_run(close: &[f64], volume: &[f64]) -> usize {
    let mut best = 0usize;
    let mut cur = 0usize;
    for (&c, &v) in close.iter().zip(volume.iter()) {
        if is_valid_pair(c, v) {
            cur += 1;
            if cur > best {
                best = cur;
            }
        } else {
            cur = 0;
        }
    }
    best
}

#[inline(always)]
fn input_slices<'a>(
    input: &'a VolumeWeightedRsiInput<'a>,
) -> Result<(&'a [f64], &'a [f64]), VolumeWeightedRsiError> {
    match &input.data {
        VolumeWeightedRsiData::Candles {
            candles,
            close_source,
        } => Ok((
            if close_source.eq_ignore_ascii_case("close") {
                candles.close.as_slice()
            } else {
                source_type(candles, close_source)
            },
            candles.volume.as_slice(),
        )),
        VolumeWeightedRsiData::Slices { close, volume } => Ok((*close, *volume)),
    }
}

#[inline(always)]
fn validate_common(
    close: &[f64],
    volume: &[f64],
    period: usize,
) -> Result<(), VolumeWeightedRsiError> {
    if close.is_empty() || volume.is_empty() {
        return Err(VolumeWeightedRsiError::EmptyInputData);
    }
    if close.len() != volume.len() {
        return Err(VolumeWeightedRsiError::InputLengthMismatch {
            close_len: close.len(),
            volume_len: volume.len(),
        });
    }
    if period == 0 || period > close.len() {
        return Err(VolumeWeightedRsiError::InvalidPeriod {
            period,
            data_len: close.len(),
        });
    }

    let max_run = longest_valid_pair_run(close, volume);
    if max_run == 0 {
        return Err(VolumeWeightedRsiError::AllValuesNaN);
    }
    if max_run < period {
        return Err(VolumeWeightedRsiError::NotEnoughValidData {
            needed: period,
            valid: max_run,
        });
    }
    Ok(())
}

#[inline(always)]
fn compute_row(close: &[f64], volume: &[f64], period: usize, out: &mut [f64]) {
    let inv_period = 1.0 / period as f64;
    let beta = 1.0 - inv_period;

    let len = close.len();
    let mut i = 0usize;
    let mut prev_close = f64::NAN;
    let mut has_prev = false;
    let mut seeded = 0usize;
    let mut sum_up = 0.0f64;
    let mut sum_down = 0.0f64;
    let mut avg_up = 0.0f64;
    let mut avg_down = 0.0f64;

    while i < len {
        let c = close[i];
        let vol = volume[i];
        if !is_valid_pair(c, vol) {
            prev_close = f64::NAN;
            has_prev = false;
            seeded = 0;
            sum_up = 0.0;
            sum_down = 0.0;
            avg_up = 0.0;
            avg_down = 0.0;
            out[i] = f64::NAN;
            i += 1;
            continue;
        }

        let (up, down) = if has_prev {
            if c > prev_close {
                (vol, 0.0)
            } else if c < prev_close {
                (0.0, vol)
            } else {
                (0.0, 0.0)
            }
        } else {
            (0.0, 0.0)
        };

        prev_close = c;
        has_prev = true;

        if seeded < period {
            sum_up += up;
            sum_down += down;
            seeded += 1;
            if seeded < period {
                out[i] = f64::NAN;
                i += 1;
                continue;
            }
            avg_up = sum_up * inv_period;
            avg_down = sum_down * inv_period;
            out[i] = rsi_from_components(avg_up, avg_down);
            i += 1;
            continue;
        }

        avg_up = avg_up.mul_add(beta, inv_period * up);
        avg_down = avg_down.mul_add(beta, inv_period * down);
        out[i] = rsi_from_components(avg_up, avg_down);
        i += 1;
    }
}

#[inline]
pub fn volume_weighted_rsi(
    input: &VolumeWeightedRsiInput,
) -> Result<VolumeWeightedRsiOutput, VolumeWeightedRsiError> {
    volume_weighted_rsi_with_kernel(input, Kernel::Auto)
}

pub fn volume_weighted_rsi_with_kernel(
    input: &VolumeWeightedRsiInput,
    kernel: Kernel,
) -> Result<VolumeWeightedRsiOutput, VolumeWeightedRsiError> {
    let (close, volume) = input_slices(input)?;
    let period = input.get_period();
    validate_common(close, volume, period)?;

    let _ = kernel;

    let mut out = alloc_uninit_f64(close.len());
    compute_row(close, volume, period, &mut out);
    Ok(VolumeWeightedRsiOutput { values: out })
}

pub fn volume_weighted_rsi_into_slice(
    dst: &mut [f64],
    input: &VolumeWeightedRsiInput,
    kernel: Kernel,
) -> Result<(), VolumeWeightedRsiError> {
    let (close, volume) = input_slices(input)?;
    let period = input.get_period();
    validate_common(close, volume, period)?;

    if dst.len() != close.len() {
        return Err(VolumeWeightedRsiError::OutputLengthMismatch {
            expected: close.len(),
            got: dst.len(),
        });
    }

    let _ = kernel;

    compute_row(close, volume, period, dst);
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub fn volume_weighted_rsi_into(
    input: &VolumeWeightedRsiInput,
    out: &mut [f64],
) -> Result<(), VolumeWeightedRsiError> {
    volume_weighted_rsi_into_slice(out, input, Kernel::Auto)
}

#[derive(Debug, Clone, Copy)]
pub struct VolumeWeightedRsiBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for VolumeWeightedRsiBatchRange {
    fn default() -> Self {
        Self {
            period: (14, 14, 0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VolumeWeightedRsiBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<VolumeWeightedRsiParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct VolumeWeightedRsiBatchBuilder {
    range: VolumeWeightedRsiBatchRange,
    kernel: Kernel,
}

impl Default for VolumeWeightedRsiBatchBuilder {
    fn default() -> Self {
        Self {
            range: VolumeWeightedRsiBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl VolumeWeightedRsiBatchBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline(always)]
    pub fn period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.period = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn period_static(mut self, value: usize) -> Self {
        self.range.period = (value, value, 0);
        self
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        close: &[f64],
        volume: &[f64],
    ) -> Result<VolumeWeightedRsiBatchOutput, VolumeWeightedRsiError> {
        volume_weighted_rsi_batch_with_kernel(close, volume, &self.range, self.kernel)
    }

    #[inline(always)]
    pub fn apply_candles(
        self,
        candles: &Candles,
    ) -> Result<VolumeWeightedRsiBatchOutput, VolumeWeightedRsiError> {
        volume_weighted_rsi_batch_with_kernel(
            candles.close.as_slice(),
            candles.volume.as_slice(),
            &self.range,
            self.kernel,
        )
    }
}

#[inline(always)]
fn expand_grid_checked(
    range: &VolumeWeightedRsiBatchRange,
) -> Result<Vec<VolumeWeightedRsiParams>, VolumeWeightedRsiError> {
    let (start, end, step) = range.period;
    if start == 0 || end == 0 {
        return Err(VolumeWeightedRsiError::InvalidRange { start, end, step });
    }
    if step == 0 {
        return Ok(vec![VolumeWeightedRsiParams {
            period: Some(start),
        }]);
    }
    if start > end {
        return Err(VolumeWeightedRsiError::InvalidRange { start, end, step });
    }

    let mut out = Vec::new();
    let mut cur = start;
    loop {
        out.push(VolumeWeightedRsiParams { period: Some(cur) });
        if cur >= end {
            break;
        }
        let next = cur.saturating_add(step);
        if next <= cur {
            return Err(VolumeWeightedRsiError::InvalidRange { start, end, step });
        }
        cur = next.min(end);
        if cur == *out.last().and_then(|p| p.period.as_ref()).unwrap() {
            break;
        }
    }
    Ok(out)
}

#[inline(always)]
pub fn expand_grid_volume_weighted_rsi(
    range: &VolumeWeightedRsiBatchRange,
) -> Vec<VolumeWeightedRsiParams> {
    expand_grid_checked(range).unwrap_or_default()
}

pub fn volume_weighted_rsi_batch_with_kernel(
    close: &[f64],
    volume: &[f64],
    sweep: &VolumeWeightedRsiBatchRange,
    kernel: Kernel,
) -> Result<VolumeWeightedRsiBatchOutput, VolumeWeightedRsiError> {
    match kernel {
        Kernel::Auto
        | Kernel::Scalar
        | Kernel::ScalarBatch
        | Kernel::Avx2
        | Kernel::Avx2Batch
        | Kernel::Avx512
        | Kernel::Avx512Batch => {}
        other => return Err(VolumeWeightedRsiError::InvalidKernelForBatch(other)),
    }

    validate_common(close, volume, 1)?;
    let combos = expand_grid_checked(sweep)?;
    let max_period = combos
        .iter()
        .map(|params| params.period.unwrap_or(14))
        .max()
        .unwrap_or(0);
    validate_common(close, volume, max_period)?;

    let rows = combos.len();
    let cols = close.len();
    let mut values_mu = make_uninit_matrix(rows, cols);
    let warmups: Vec<usize> = combos
        .iter()
        .map(|params| params.period.unwrap_or(14).saturating_sub(1))
        .collect();
    init_matrix_prefixes(&mut values_mu, cols, &warmups);
    let mut values = unsafe {
        Vec::from_raw_parts(
            values_mu.as_mut_ptr() as *mut f64,
            values_mu.len(),
            values_mu.capacity(),
        )
    };
    std::mem::forget(values_mu);

    volume_weighted_rsi_batch_inner_into(close, volume, sweep, kernel, true, &mut values)?;

    Ok(VolumeWeightedRsiBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

pub fn volume_weighted_rsi_batch_slice(
    close: &[f64],
    volume: &[f64],
    sweep: &VolumeWeightedRsiBatchRange,
    kernel: Kernel,
) -> Result<VolumeWeightedRsiBatchOutput, VolumeWeightedRsiError> {
    volume_weighted_rsi_batch_inner(close, volume, sweep, kernel, false)
}

pub fn volume_weighted_rsi_batch_par_slice(
    close: &[f64],
    volume: &[f64],
    sweep: &VolumeWeightedRsiBatchRange,
    kernel: Kernel,
) -> Result<VolumeWeightedRsiBatchOutput, VolumeWeightedRsiError> {
    volume_weighted_rsi_batch_inner(close, volume, sweep, kernel, true)
}

fn volume_weighted_rsi_batch_inner(
    close: &[f64],
    volume: &[f64],
    sweep: &VolumeWeightedRsiBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<VolumeWeightedRsiBatchOutput, VolumeWeightedRsiError> {
    let combos = expand_grid_checked(sweep)?;
    let rows = combos.len();
    let cols = close.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| VolumeWeightedRsiError::InvalidInput {
            msg: "volume_weighted_rsi: rows*cols overflow in batch".to_string(),
        })?;

    let mut values_mu = make_uninit_matrix(rows, cols);
    let warmups: Vec<usize> = combos
        .iter()
        .map(|params| params.period.unwrap_or(14).saturating_sub(1))
        .collect();
    init_matrix_prefixes(&mut values_mu, cols, &warmups);
    let mut values = unsafe {
        Vec::from_raw_parts(
            values_mu.as_mut_ptr() as *mut f64,
            values_mu.len(),
            values_mu.capacity(),
        )
    };
    std::mem::forget(values_mu);

    debug_assert_eq!(values.len(), total);

    volume_weighted_rsi_batch_inner_into(close, volume, sweep, kernel, parallel, &mut values)?;

    Ok(VolumeWeightedRsiBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

fn volume_weighted_rsi_batch_inner_into(
    close: &[f64],
    volume: &[f64],
    sweep: &VolumeWeightedRsiBatchRange,
    kernel: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<VolumeWeightedRsiParams>, VolumeWeightedRsiError> {
    match kernel {
        Kernel::Auto
        | Kernel::Scalar
        | Kernel::ScalarBatch
        | Kernel::Avx2
        | Kernel::Avx2Batch
        | Kernel::Avx512
        | Kernel::Avx512Batch => {}
        other => return Err(VolumeWeightedRsiError::InvalidKernelForBatch(other)),
    }

    let combos = expand_grid_checked(sweep)?;
    let len = close.len();
    if len == 0 || volume.is_empty() {
        return Err(VolumeWeightedRsiError::EmptyInputData);
    }
    if len != volume.len() {
        return Err(VolumeWeightedRsiError::InputLengthMismatch {
            close_len: len,
            volume_len: volume.len(),
        });
    }

    let total =
        combos
            .len()
            .checked_mul(len)
            .ok_or_else(|| VolumeWeightedRsiError::InvalidInput {
                msg: "volume_weighted_rsi: rows*cols overflow in batch_into".to_string(),
            })?;
    if out.len() != total {
        return Err(VolumeWeightedRsiError::MismatchedOutputLen {
            dst_len: out.len(),
            expected_len: total,
        });
    }

    let max_period = combos
        .iter()
        .map(|params| params.period.unwrap_or(14))
        .max()
        .unwrap_or(0);
    validate_common(close, volume, max_period)?;

    let _ = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other => other,
    };

    let worker = |row: usize, dst: &mut [f64]| {
        let period = combos[row].period.unwrap_or(14);
        compute_row(close, volume, period, dst);
    };

    #[cfg(not(target_arch = "wasm32"))]
    if parallel {
        out.par_chunks_mut(len)
            .enumerate()
            .for_each(|(row, dst)| worker(row, dst));
    } else {
        for (row, dst) in out.chunks_mut(len).enumerate() {
            worker(row, dst);
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        let _ = parallel;
        for (row, dst) in out.chunks_mut(len).enumerate() {
            worker(row, dst);
        }
    }

    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "volume_weighted_rsi")]
#[pyo3(signature = (close, volume, period=14, kernel=None))]
pub fn volume_weighted_rsi_py<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let close = close.as_slice()?;
    let volume = volume.as_slice()?;
    let kern = validate_kernel(kernel, false)?;
    let input = VolumeWeightedRsiInput::from_slices(
        close,
        volume,
        VolumeWeightedRsiParams {
            period: Some(period),
        },
    );
    let out = py
        .allow_threads(|| volume_weighted_rsi_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(out.values.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "VolumeWeightedRsiStream")]
pub struct VolumeWeightedRsiStreamPy {
    stream: VolumeWeightedRsiStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl VolumeWeightedRsiStreamPy {
    #[new]
    fn new(period: usize) -> PyResult<Self> {
        let stream = VolumeWeightedRsiStream::try_new(VolumeWeightedRsiParams {
            period: Some(period),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, close: f64, volume: f64) -> Option<f64> {
        self.stream.update(close, volume)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "volume_weighted_rsi_batch")]
#[pyo3(signature = (close, volume, period_range=(14,14,0), kernel=None))]
pub fn volume_weighted_rsi_batch_py<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let close = close.as_slice()?;
    let volume = volume.as_slice()?;
    let kern = validate_kernel(kernel, true)?;

    let output = py
        .allow_threads(|| {
            volume_weighted_rsi_batch_with_kernel(
                close,
                volume,
                &VolumeWeightedRsiBatchRange {
                    period: period_range,
                },
                kern,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let rows = output.rows;
    let cols = output.cols;
    let dict = PyDict::new(py);
    dict.set_item(
        "values",
        output.values.into_pyarray(py).reshape((rows, cols))?,
    )?;
    dict.set_item(
        "periods",
        output
            .combos
            .iter()
            .map(|params| params.period.unwrap_or(14) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_volume_weighted_rsi_module(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(volume_weighted_rsi_py, m)?)?;
    m.add_function(wrap_pyfunction!(volume_weighted_rsi_batch_py, m)?)?;
    m.add_class::<VolumeWeightedRsiStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeWeightedRsiBatchConfig {
    pub period_range: Vec<usize>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = volume_weighted_rsi_js)]
pub fn volume_weighted_rsi_js(
    close: &[f64],
    volume: &[f64],
    period: usize,
) -> Result<JsValue, JsValue> {
    let input = VolumeWeightedRsiInput::from_slices(
        close,
        volume,
        VolumeWeightedRsiParams {
            period: Some(period),
        },
    );
    let out = volume_weighted_rsi_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&out.values).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = volume_weighted_rsi_batch_js)]
pub fn volume_weighted_rsi_batch_js(
    close: &[f64],
    volume: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: VolumeWeightedRsiBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    if config.period_range.len() != 3 {
        return Err(JsValue::from_str(
            "Invalid config: period_range must have exactly 3 elements [start, end, step]",
        ));
    }

    let out = volume_weighted_rsi_batch_with_kernel(
        close,
        volume,
        &VolumeWeightedRsiBatchRange {
            period: (
                config.period_range[0],
                config.period_range[1],
                config.period_range[2],
            ),
        },
        Kernel::Auto,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("values"),
        &serde_wasm_bindgen::to_value(&out.values).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("rows"),
        &JsValue::from_f64(out.rows as f64),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("cols"),
        &JsValue::from_f64(out.cols as f64),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("combos"),
        &serde_wasm_bindgen::to_value(&out.combos).unwrap(),
    )?;
    Ok(obj.into())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn volume_weighted_rsi_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn volume_weighted_rsi_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn volume_weighted_rsi_into(
    close_ptr: *const f64,
    volume_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
) -> Result<(), JsValue> {
    if close_ptr.is_null() || volume_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to volume_weighted_rsi_into",
        ));
    }

    unsafe {
        let close = std::slice::from_raw_parts(close_ptr, len);
        let volume = std::slice::from_raw_parts(volume_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, len);
        let input = VolumeWeightedRsiInput::from_slices(
            close,
            volume,
            VolumeWeightedRsiParams {
                period: Some(period),
            },
        );
        volume_weighted_rsi_into_slice(out, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn volume_weighted_rsi_batch_into(
    close_ptr: *const f64,
    volume_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<usize, JsValue> {
    if close_ptr.is_null() || volume_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to volume_weighted_rsi_batch_into",
        ));
    }

    let sweep = VolumeWeightedRsiBatchRange {
        period: (period_start, period_end, period_step),
    };
    let combos = expand_grid_checked(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();
    let total = rows
        .checked_mul(len)
        .ok_or_else(|| JsValue::from_str("rows*cols overflow in volume_weighted_rsi_batch_into"))?;

    unsafe {
        let close = std::slice::from_raw_parts(close_ptr, len);
        let volume = std::slice::from_raw_parts(volume_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, total);
        volume_weighted_rsi_batch_inner_into(close, volume, &sweep, Kernel::Auto, false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }

    Ok(rows)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn volume_weighted_rsi_output_into_js(
    close: &[f64],
    volume: &[f64],
    period: usize,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = volume_weighted_rsi_js(close, volume, period)?;
    crate::write_wasm_object_f64_outputs("volume_weighted_rsi_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn volume_weighted_rsi_batch_output_into_js(
    close: &[f64],
    volume: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = volume_weighted_rsi_batch_js(close, volume, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "volume_weighted_rsi_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::dispatch::{
        compute_cpu, IndicatorComputeRequest, IndicatorDataRef, ParamKV, ParamValue,
    };

    fn sample_close_volume(len: usize) -> (Vec<f64>, Vec<f64>) {
        let close: Vec<f64> = (0..len)
            .map(|i| 100.0 + ((i as f64) * 0.13).sin() * 4.0 + (i as f64) * 0.02)
            .collect();
        let volume: Vec<f64> = (0..len)
            .map(|i| 1000.0 + ((i as f64) * 0.17).cos().abs() * 250.0 + (i % 11) as f64 * 7.0)
            .collect();
        (close, volume)
    }

    fn naive_volume_weighted_rsi(close: &[f64], volume: &[f64], period: usize) -> Vec<f64> {
        let mut out = vec![f64::NAN; close.len()];
        compute_row(close, volume, period, &mut out);
        out
    }

    #[test]
    fn volume_weighted_rsi_matches_naive() -> Result<(), Box<dyn Error>> {
        let (close, volume) = sample_close_volume(256);
        let input = VolumeWeightedRsiInput::from_slices(
            &close,
            &volume,
            VolumeWeightedRsiParams { period: Some(14) },
        );
        let out = volume_weighted_rsi(&input)?;
        let expected = naive_volume_weighted_rsi(&close, &volume, 14);
        for (a, b) in out.values.iter().zip(expected.iter()) {
            if a.is_nan() || b.is_nan() {
                assert!(a.is_nan() && b.is_nan());
            } else {
                assert!((a - b).abs() < 1e-12);
            }
        }
        Ok(())
    }

    #[test]
    fn volume_weighted_rsi_into_matches_api() -> Result<(), Box<dyn Error>> {
        let (close, volume) = sample_close_volume(200);
        let input = VolumeWeightedRsiInput::from_slices(
            &close,
            &volume,
            VolumeWeightedRsiParams { period: Some(10) },
        );
        let base = volume_weighted_rsi(&input)?;
        let mut out = vec![0.0; close.len()];
        volume_weighted_rsi_into_slice(&mut out, &input, Kernel::Auto)?;
        for (a, b) in out.iter().zip(base.values.iter()) {
            if a.is_nan() || b.is_nan() {
                assert!(a.is_nan() && b.is_nan());
            } else {
                assert!((a - b).abs() < 1e-12);
            }
        }
        Ok(())
    }

    #[test]
    fn volume_weighted_rsi_stream_matches_batch() -> Result<(), Box<dyn Error>> {
        let (close, volume) = sample_close_volume(220);
        let period = 12;
        let input = VolumeWeightedRsiInput::from_slices(
            &close,
            &volume,
            VolumeWeightedRsiParams {
                period: Some(period),
            },
        );
        let batch = volume_weighted_rsi(&input)?;
        let mut stream = VolumeWeightedRsiStream::try_new(VolumeWeightedRsiParams {
            period: Some(period),
        })?;
        let mut values = Vec::with_capacity(close.len());
        for (&c, &v) in close.iter().zip(volume.iter()) {
            values.push(stream.update(c, v).unwrap_or(f64::NAN));
        }
        for (a, b) in values.iter().zip(batch.values.iter()) {
            if a.is_nan() || b.is_nan() {
                assert!(a.is_nan() && b.is_nan());
            } else {
                assert!((a - b).abs() < 1e-12);
            }
        }
        Ok(())
    }

    #[test]
    fn volume_weighted_rsi_batch_single_matches_single() -> Result<(), Box<dyn Error>> {
        let (close, volume) = sample_close_volume(128);
        let single = volume_weighted_rsi(&VolumeWeightedRsiInput::from_slices(
            &close,
            &volume,
            VolumeWeightedRsiParams { period: Some(14) },
        ))?;
        let batch = volume_weighted_rsi_batch_with_kernel(
            &close,
            &volume,
            &VolumeWeightedRsiBatchRange {
                period: (14, 14, 0),
            },
            Kernel::Auto,
        )?;
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, close.len());
        for (a, b) in batch.values.iter().zip(single.values.iter()) {
            if a.is_nan() || b.is_nan() {
                assert!(a.is_nan() && b.is_nan());
            } else {
                assert!((a - b).abs() < 1e-12);
            }
        }
        Ok(())
    }

    #[test]
    fn volume_weighted_rsi_rejects_invalid_params() {
        let (close, volume) = sample_close_volume(16);
        let err = volume_weighted_rsi(&VolumeWeightedRsiInput::from_slices(
            &close,
            &volume,
            VolumeWeightedRsiParams { period: Some(0) },
        ))
        .unwrap_err();
        assert!(matches!(err, VolumeWeightedRsiError::InvalidPeriod { .. }));
    }

    #[test]
    fn volume_weighted_rsi_dispatch_compute_returns_value() {
        let (close, volume) = sample_close_volume(128);
        let params = [ParamKV {
            key: "period",
            value: ParamValue::Int(14),
        }];
        let out = compute_cpu(IndicatorComputeRequest {
            indicator_id: "volume_weighted_rsi",
            output_id: Some("value"),
            data: IndicatorDataRef::CloseVolume {
                close: &close,
                volume: &volume,
            },
            params: &params,
            kernel: Kernel::Auto,
        })
        .unwrap();
        assert_eq!(out.output_id, "value");
        assert_eq!(out.cols, close.len());
    }
}
