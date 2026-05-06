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

use crate::utilities::data_loader::Candles;
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{alloc_with_nan_prefix, init_matrix_prefixes, make_uninit_matrix};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::collections::VecDeque;
use std::error::Error;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum DonchianChannelWidthData<'a> {
    Candles { candles: &'a Candles },
    Slices { high: &'a [f64], low: &'a [f64] },
}

#[derive(Debug, Clone)]
pub struct DonchianChannelWidthOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct DonchianChannelWidthParams {
    pub period: Option<usize>,
}

impl Default for DonchianChannelWidthParams {
    fn default() -> Self {
        Self { period: Some(20) }
    }
}

#[derive(Debug, Clone)]
pub struct DonchianChannelWidthInput<'a> {
    pub data: DonchianChannelWidthData<'a>,
    pub params: DonchianChannelWidthParams,
}

impl<'a> DonchianChannelWidthInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: DonchianChannelWidthParams) -> Self {
        Self {
            data: DonchianChannelWidthData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        params: DonchianChannelWidthParams,
    ) -> Self {
        Self {
            data: DonchianChannelWidthData::Slices { high, low },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, DonchianChannelWidthParams::default())
    }

    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(20)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct DonchianChannelWidthBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for DonchianChannelWidthBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl DonchianChannelWidthBuilder {
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
    ) -> Result<DonchianChannelWidthOutput, DonchianChannelWidthError> {
        let params = DonchianChannelWidthParams {
            period: self.period,
        };
        donchian_channel_width_with_kernel(
            &DonchianChannelWidthInput::from_candles(candles, params),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
    ) -> Result<DonchianChannelWidthOutput, DonchianChannelWidthError> {
        let params = DonchianChannelWidthParams {
            period: self.period,
        };
        donchian_channel_width_with_kernel(
            &DonchianChannelWidthInput::from_slices(high, low, params),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<DonchianChannelWidthStream, DonchianChannelWidthError> {
        DonchianChannelWidthStream::try_new(DonchianChannelWidthParams {
            period: self.period,
        })
    }
}

#[derive(Debug, Error)]
pub enum DonchianChannelWidthError {
    #[error("donchian_channel_width: Input data slice is empty.")]
    EmptyInputData,
    #[error("donchian_channel_width: Input length mismatch: high = {high_len}, low = {low_len}")]
    InputLengthMismatch { high_len: usize, low_len: usize },
    #[error("donchian_channel_width: All values are NaN.")]
    AllValuesNaN,
    #[error("donchian_channel_width: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("donchian_channel_width: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("donchian_channel_width: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("donchian_channel_width: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("donchian_channel_width: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error(
        "donchian_channel_width: Output length mismatch: dst = {dst_len}, expected = {expected_len}"
    )]
    MismatchedOutputLen { dst_len: usize, expected_len: usize },
    #[error("donchian_channel_width: Invalid input: {msg}")]
    InvalidInput { msg: String },
}

#[derive(Debug, Clone)]
pub struct DonchianChannelWidthStream {
    period: usize,
    next_index: usize,
    max_deque: VecDeque<(usize, f64)>,
    min_deque: VecDeque<(usize, f64)>,
}

impl DonchianChannelWidthStream {
    #[inline(always)]
    pub fn try_new(params: DonchianChannelWidthParams) -> Result<Self, DonchianChannelWidthError> {
        let period = params.period.unwrap_or(20);
        if period == 0 {
            return Err(DonchianChannelWidthError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        Ok(Self {
            period,
            next_index: 0,
            max_deque: VecDeque::with_capacity(period.max(1)),
            min_deque: VecDeque::with_capacity(period.max(1)),
        })
    }

    #[inline(always)]
    pub fn reset(&mut self) {
        self.next_index = 0;
        self.max_deque.clear();
        self.min_deque.clear();
    }

    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64) -> Option<f64> {
        if !is_valid_pair(high, low) {
            self.reset();
            return None;
        }

        let idx = self.next_index;
        self.next_index += 1;

        while let Some((_, v)) = self.max_deque.back() {
            if *v <= high {
                self.max_deque.pop_back();
            } else {
                break;
            }
        }
        self.max_deque.push_back((idx, high));

        while let Some((_, v)) = self.min_deque.back() {
            if *v >= low {
                self.min_deque.pop_back();
            } else {
                break;
            }
        }
        self.min_deque.push_back((idx, low));

        let window_start = idx.saturating_add(1).saturating_sub(self.period);
        while let Some((front_idx, _)) = self.max_deque.front() {
            if *front_idx < window_start {
                self.max_deque.pop_front();
            } else {
                break;
            }
        }
        while let Some((front_idx, _)) = self.min_deque.front() {
            if *front_idx < window_start {
                self.min_deque.pop_front();
            } else {
                break;
            }
        }

        if idx + 1 < self.period {
            None
        } else {
            let upper = self.max_deque.front().map(|(_, v)| *v).unwrap_or(f64::NAN);
            let lower = self.min_deque.front().map(|(_, v)| *v).unwrap_or(f64::NAN);
            Some(upper - lower)
        }
    }

    #[inline(always)]
    pub fn get_warmup_period(&self) -> usize {
        self.period.saturating_sub(1)
    }
}

#[inline(always)]
fn is_valid_pair(high: f64, low: f64) -> bool {
    high.is_finite() && low.is_finite()
}

#[inline(always)]
fn valid_pair_run_state(high: &[f64], low: &[f64]) -> (usize, bool) {
    let mut best = 0usize;
    let mut cur = 0usize;
    let mut all_valid = true;
    for (&h, &l) in high.iter().zip(low.iter()) {
        if is_valid_pair(h, l) {
            cur += 1;
            if cur > best {
                best = cur;
            }
        } else {
            all_valid = false;
            cur = 0;
        }
    }
    (best, all_valid)
}

#[inline(always)]
fn input_slices<'a>(
    input: &'a DonchianChannelWidthInput<'a>,
) -> Result<(&'a [f64], &'a [f64]), DonchianChannelWidthError> {
    match &input.data {
        DonchianChannelWidthData::Candles { candles } => {
            Ok((candles.high.as_slice(), candles.low.as_slice()))
        }
        DonchianChannelWidthData::Slices { high, low } => Ok((*high, *low)),
    }
}

#[inline(always)]
fn validate_common(
    high: &[f64],
    low: &[f64],
    period: usize,
) -> Result<bool, DonchianChannelWidthError> {
    if high.is_empty() || low.is_empty() {
        return Err(DonchianChannelWidthError::EmptyInputData);
    }
    if high.len() != low.len() {
        return Err(DonchianChannelWidthError::InputLengthMismatch {
            high_len: high.len(),
            low_len: low.len(),
        });
    }
    if period == 0 || period > high.len() {
        return Err(DonchianChannelWidthError::InvalidPeriod {
            period,
            data_len: high.len(),
        });
    }

    let (max_run, all_valid) = valid_pair_run_state(high, low);
    if max_run == 0 {
        return Err(DonchianChannelWidthError::AllValuesNaN);
    }
    if max_run < period {
        return Err(DonchianChannelWidthError::NotEnoughValidData {
            needed: period,
            valid: max_run,
        });
    }
    Ok(all_valid)
}

#[inline(always)]
fn compute_row(high: &[f64], low: &[f64], period: usize, out: &mut [f64]) {
    let cap = period + 1;
    let mut max_queue = vec![0usize; cap];
    let mut min_queue = vec![0usize; cap];
    let mut max_head = 0usize;
    let mut max_tail = 0usize;
    let mut min_head = 0usize;
    let mut min_tail = 0usize;
    let mut seg_start = 0usize;
    let mut in_segment = false;

    for i in 0..high.len() {
        let h = high[i];
        let l = low[i];
        if !is_valid_pair(h, l) {
            out[i] = f64::NAN;
            max_head = 0;
            max_tail = 0;
            min_head = 0;
            min_tail = 0;
            in_segment = false;
            continue;
        }

        if !in_segment {
            seg_start = i;
            in_segment = true;
        }

        let raw_start = i.saturating_add(1).saturating_sub(period);
        let window_start = raw_start.max(seg_start);

        while max_head != max_tail && max_queue[max_head] < window_start {
            max_head += 1;
            if max_head == cap {
                max_head = 0;
            }
        }
        while min_head != min_tail && min_queue[min_head] < window_start {
            min_head += 1;
            if min_head == cap {
                min_head = 0;
            }
        }

        while max_head != max_tail {
            let back = if max_tail == 0 { cap - 1 } else { max_tail - 1 };
            if high[max_queue[back]] <= h {
                max_tail = back;
            } else {
                break;
            }
        }
        max_queue[max_tail] = i;
        max_tail += 1;
        if max_tail == cap {
            max_tail = 0;
        }

        while min_head != min_tail {
            let back = if min_tail == 0 { cap - 1 } else { min_tail - 1 };
            if low[min_queue[back]] >= l {
                min_tail = back;
            } else {
                break;
            }
        }
        min_queue[min_tail] = i;
        min_tail += 1;
        if min_tail == cap {
            min_tail = 0;
        }

        if i + 1 >= seg_start + period {
            let upper = high[max_queue[max_head]];
            let lower = low[min_queue[min_head]];
            out[i] = upper - lower;
        } else {
            out[i] = f64::NAN;
        }
    }
}

#[inline(always)]
fn compute_row_no_nan(high: &[f64], low: &[f64], period: usize, out: &mut [f64]) {
    let cap = period + 1;
    let mut max_queue = vec![0usize; cap];
    let mut min_queue = vec![0usize; cap];
    let mut max_head = 0usize;
    let mut max_tail = 0usize;
    let mut min_head = 0usize;
    let mut min_tail = 0usize;

    for i in 0..high.len() {
        let h = high[i];
        let l = low[i];
        let window_start = i.saturating_add(1).saturating_sub(period);

        while max_head != max_tail && max_queue[max_head] < window_start {
            max_head += 1;
            if max_head == cap {
                max_head = 0;
            }
        }
        while min_head != min_tail && min_queue[min_head] < window_start {
            min_head += 1;
            if min_head == cap {
                min_head = 0;
            }
        }

        while max_head != max_tail {
            let back = if max_tail == 0 { cap - 1 } else { max_tail - 1 };
            if high[max_queue[back]] <= h {
                max_tail = back;
            } else {
                break;
            }
        }
        max_queue[max_tail] = i;
        max_tail += 1;
        if max_tail == cap {
            max_tail = 0;
        }

        while min_head != min_tail {
            let back = if min_tail == 0 { cap - 1 } else { min_tail - 1 };
            if low[min_queue[back]] >= l {
                min_tail = back;
            } else {
                break;
            }
        }
        min_queue[min_tail] = i;
        min_tail += 1;
        if min_tail == cap {
            min_tail = 0;
        }

        if i + 1 >= period {
            out[i] = high[max_queue[max_head]] - low[min_queue[min_head]];
        } else {
            out[i] = f64::NAN;
        }
    }
}

#[inline]
pub fn donchian_channel_width(
    input: &DonchianChannelWidthInput,
) -> Result<DonchianChannelWidthOutput, DonchianChannelWidthError> {
    donchian_channel_width_with_kernel(input, Kernel::Auto)
}

pub fn donchian_channel_width_with_kernel(
    input: &DonchianChannelWidthInput,
    kernel: Kernel,
) -> Result<DonchianChannelWidthOutput, DonchianChannelWidthError> {
    let (high, low) = input_slices(input)?;
    let period = input.get_period();
    let all_valid = validate_common(high, low, period)?;
    let _ = kernel;

    let mut out = alloc_with_nan_prefix(high.len(), 0);
    if all_valid {
        compute_row_no_nan(high, low, period, &mut out);
    } else {
        compute_row(high, low, period, &mut out);
    }
    Ok(DonchianChannelWidthOutput { values: out })
}

pub fn donchian_channel_width_into_slice(
    dst: &mut [f64],
    input: &DonchianChannelWidthInput,
    kernel: Kernel,
) -> Result<(), DonchianChannelWidthError> {
    let (high, low) = input_slices(input)?;
    let period = input.get_period();
    let all_valid = validate_common(high, low, period)?;

    if dst.len() != high.len() {
        return Err(DonchianChannelWidthError::OutputLengthMismatch {
            expected: high.len(),
            got: dst.len(),
        });
    }

    let _ = kernel;

    if all_valid {
        compute_row_no_nan(high, low, period, dst);
    } else {
        compute_row(high, low, period, dst);
    }
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub fn donchian_channel_width_into(
    input: &DonchianChannelWidthInput,
    out: &mut [f64],
) -> Result<(), DonchianChannelWidthError> {
    donchian_channel_width_into_slice(out, input, Kernel::Auto)
}

#[derive(Debug, Clone, Copy)]
pub struct DonchianChannelWidthBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for DonchianChannelWidthBatchRange {
    fn default() -> Self {
        Self {
            period: (20, 20, 0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DonchianChannelWidthBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<DonchianChannelWidthParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct DonchianChannelWidthBatchBuilder {
    range: DonchianChannelWidthBatchRange,
    kernel: Kernel,
}

impl Default for DonchianChannelWidthBatchBuilder {
    fn default() -> Self {
        Self {
            range: DonchianChannelWidthBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl DonchianChannelWidthBatchBuilder {
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
        high: &[f64],
        low: &[f64],
    ) -> Result<DonchianChannelWidthBatchOutput, DonchianChannelWidthError> {
        donchian_channel_width_batch_with_kernel(high, low, &self.range, self.kernel)
    }

    #[inline(always)]
    pub fn apply_candles(
        self,
        candles: &Candles,
    ) -> Result<DonchianChannelWidthBatchOutput, DonchianChannelWidthError> {
        donchian_channel_width_batch_with_kernel(
            candles.high.as_slice(),
            candles.low.as_slice(),
            &self.range,
            self.kernel,
        )
    }
}

#[inline(always)]
fn expand_grid_checked(
    range: &DonchianChannelWidthBatchRange,
) -> Result<Vec<DonchianChannelWidthParams>, DonchianChannelWidthError> {
    let (start, end, step) = range.period;
    if start == 0 || end == 0 {
        return Err(DonchianChannelWidthError::InvalidRange { start, end, step });
    }
    if step == 0 {
        return Ok(vec![DonchianChannelWidthParams {
            period: Some(start),
        }]);
    }
    if start > end {
        return Err(DonchianChannelWidthError::InvalidRange { start, end, step });
    }

    let mut out = Vec::new();
    let mut cur = start;
    loop {
        out.push(DonchianChannelWidthParams { period: Some(cur) });
        if cur >= end {
            break;
        }
        let next = cur.saturating_add(step);
        if next <= cur {
            return Err(DonchianChannelWidthError::InvalidRange { start, end, step });
        }
        cur = next.min(end);
        if cur == *out.last().and_then(|p| p.period.as_ref()).unwrap() {
            break;
        }
    }
    Ok(out)
}

#[inline(always)]
pub fn expand_grid_donchian_channel_width(
    range: &DonchianChannelWidthBatchRange,
) -> Vec<DonchianChannelWidthParams> {
    expand_grid_checked(range).unwrap_or_default()
}

pub fn donchian_channel_width_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    sweep: &DonchianChannelWidthBatchRange,
    kernel: Kernel,
) -> Result<DonchianChannelWidthBatchOutput, DonchianChannelWidthError> {
    match kernel {
        Kernel::Auto
        | Kernel::Scalar
        | Kernel::ScalarBatch
        | Kernel::Avx2
        | Kernel::Avx2Batch
        | Kernel::Avx512
        | Kernel::Avx512Batch => {}
        other => return Err(DonchianChannelWidthError::InvalidKernelForBatch(other)),
    }

    let combos = expand_grid_checked(sweep)?;
    let max_period = combos
        .iter()
        .map(|params| params.period.unwrap_or(20))
        .max()
        .unwrap_or(0);
    validate_common(high, low, max_period)?;

    let rows = combos.len();
    let cols = high.len();
    let mut values_mu = make_uninit_matrix(rows, cols);
    let warmups: Vec<usize> = combos
        .iter()
        .map(|params| params.period.unwrap_or(20).saturating_sub(1))
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

    donchian_channel_width_batch_inner_into(high, low, sweep, kernel, true, &mut values)?;

    Ok(DonchianChannelWidthBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

pub fn donchian_channel_width_batch_slice(
    high: &[f64],
    low: &[f64],
    sweep: &DonchianChannelWidthBatchRange,
    kernel: Kernel,
) -> Result<DonchianChannelWidthBatchOutput, DonchianChannelWidthError> {
    donchian_channel_width_batch_inner(high, low, sweep, kernel, false)
}

pub fn donchian_channel_width_batch_par_slice(
    high: &[f64],
    low: &[f64],
    sweep: &DonchianChannelWidthBatchRange,
    kernel: Kernel,
) -> Result<DonchianChannelWidthBatchOutput, DonchianChannelWidthError> {
    donchian_channel_width_batch_inner(high, low, sweep, kernel, true)
}

fn donchian_channel_width_batch_inner(
    high: &[f64],
    low: &[f64],
    sweep: &DonchianChannelWidthBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<DonchianChannelWidthBatchOutput, DonchianChannelWidthError> {
    let combos = expand_grid_checked(sweep)?;
    let rows = combos.len();
    let cols = high.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| DonchianChannelWidthError::InvalidInput {
            msg: "donchian_channel_width: rows*cols overflow in batch".to_string(),
        })?;

    let mut values_mu = make_uninit_matrix(rows, cols);
    let warmups: Vec<usize> = combos
        .iter()
        .map(|params| params.period.unwrap_or(20).saturating_sub(1))
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

    donchian_channel_width_batch_inner_into(high, low, sweep, kernel, parallel, &mut values)?;

    Ok(DonchianChannelWidthBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

fn donchian_channel_width_batch_inner_into(
    high: &[f64],
    low: &[f64],
    sweep: &DonchianChannelWidthBatchRange,
    kernel: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<DonchianChannelWidthParams>, DonchianChannelWidthError> {
    match kernel {
        Kernel::Auto
        | Kernel::Scalar
        | Kernel::ScalarBatch
        | Kernel::Avx2
        | Kernel::Avx2Batch
        | Kernel::Avx512
        | Kernel::Avx512Batch => {}
        other => return Err(DonchianChannelWidthError::InvalidKernelForBatch(other)),
    }

    let combos = expand_grid_checked(sweep)?;
    let len = high.len();
    if len == 0 || low.is_empty() {
        return Err(DonchianChannelWidthError::EmptyInputData);
    }
    if len != low.len() {
        return Err(DonchianChannelWidthError::InputLengthMismatch {
            high_len: len,
            low_len: low.len(),
        });
    }

    let total =
        combos
            .len()
            .checked_mul(len)
            .ok_or_else(|| DonchianChannelWidthError::InvalidInput {
                msg: "donchian_channel_width: rows*cols overflow in batch_into".to_string(),
            })?;
    if out.len() != total {
        return Err(DonchianChannelWidthError::MismatchedOutputLen {
            dst_len: out.len(),
            expected_len: total,
        });
    }

    let max_period = combos
        .iter()
        .map(|params| params.period.unwrap_or(20))
        .max()
        .unwrap_or(0);
    let all_valid = validate_common(high, low, max_period)?;
    let _ = kernel;

    let worker = |row: usize, dst: &mut [f64]| {
        let period = combos[row].period.unwrap_or(20);
        if all_valid {
            compute_row_no_nan(high, low, period, dst);
        } else {
            compute_row(high, low, period, dst);
        }
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
#[pyfunction(name = "donchian_channel_width")]
#[pyo3(signature = (high, low, period=20, kernel=None))]
pub fn donchian_channel_width_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    let input = DonchianChannelWidthInput::from_slices(
        high,
        low,
        DonchianChannelWidthParams {
            period: Some(period),
        },
    );
    let out = py
        .allow_threads(|| donchian_channel_width_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(out.values.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "DonchianChannelWidthStream")]
pub struct DonchianChannelWidthStreamPy {
    stream: DonchianChannelWidthStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl DonchianChannelWidthStreamPy {
    #[new]
    fn new(period: usize) -> PyResult<Self> {
        let stream = DonchianChannelWidthStream::try_new(DonchianChannelWidthParams {
            period: Some(period),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, high: f64, low: f64) -> Option<f64> {
        self.stream.update(high, low)
    }

    fn reset(&mut self) {
        self.stream.reset();
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "donchian_channel_width_batch")]
#[pyo3(signature = (high, low, period_range=(20,20,0), kernel=None))]
pub fn donchian_channel_width_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let kern = validate_kernel(kernel, true)?;

    let output = py
        .allow_threads(|| {
            donchian_channel_width_batch_with_kernel(
                high,
                low,
                &DonchianChannelWidthBatchRange {
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
            .map(|params| params.period.unwrap_or(20) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_donchian_channel_width_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(donchian_channel_width_py, m)?)?;
    m.add_function(wrap_pyfunction!(donchian_channel_width_batch_py, m)?)?;
    m.add_class::<DonchianChannelWidthStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DonchianChannelWidthBatchConfig {
    pub period_range: Vec<usize>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = donchian_channel_width_js)]
pub fn donchian_channel_width_js(
    high: &[f64],
    low: &[f64],
    period: usize,
) -> Result<JsValue, JsValue> {
    let input = DonchianChannelWidthInput::from_slices(
        high,
        low,
        DonchianChannelWidthParams {
            period: Some(period),
        },
    );
    let out = donchian_channel_width_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&out.values).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = donchian_channel_width_batch_js)]
pub fn donchian_channel_width_batch_js(
    high: &[f64],
    low: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: DonchianChannelWidthBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    if config.period_range.len() != 3 {
        return Err(JsValue::from_str(
            "Invalid config: period_range must have exactly 3 elements [start, end, step]",
        ));
    }
    let out = donchian_channel_width_batch_with_kernel(
        high,
        low,
        &DonchianChannelWidthBatchRange {
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
pub fn donchian_channel_width_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn donchian_channel_width_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn donchian_channel_width_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
) -> Result<(), JsValue> {
    if high_ptr.is_null() || low_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to donchian_channel_width_into",
        ));
    }
    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, len);
        let input = DonchianChannelWidthInput::from_slices(
            high,
            low,
            DonchianChannelWidthParams {
                period: Some(period),
            },
        );
        donchian_channel_width_into_slice(out, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn donchian_channel_width_batch_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<usize, JsValue> {
    if high_ptr.is_null() || low_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to donchian_channel_width_batch_into",
        ));
    }
    let sweep = DonchianChannelWidthBatchRange {
        period: (period_start, period_end, period_step),
    };
    let combos = expand_grid_checked(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();
    let total = rows.checked_mul(len).ok_or_else(|| {
        JsValue::from_str("rows*cols overflow in donchian_channel_width_batch_into")
    })?;
    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, total);
        donchian_channel_width_batch_inner_into(high, low, &sweep, Kernel::Auto, false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }
    Ok(rows)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn donchian_channel_width_output_into_js(
    high: &[f64],
    low: &[f64],
    period: usize,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = donchian_channel_width_js(high, low, period)?;
    crate::write_wasm_object_f64_outputs("donchian_channel_width_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn donchian_channel_width_batch_output_into_js(
    high: &[f64],
    low: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = donchian_channel_width_batch_js(high, low, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "donchian_channel_width_batch_output_into_js",
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

    fn sample_high_low(len: usize) -> (Vec<f64>, Vec<f64>) {
        let high: Vec<f64> = (0..len)
            .map(|i| {
                let x = i as f64;
                100.0 + x * 0.03 + (x * 0.11).sin() * 1.2 + (x * 0.017).cos() * 0.4
            })
            .collect();
        let low: Vec<f64> = high
            .iter()
            .enumerate()
            .map(|(i, &h)| h - 1.1 - ((i as f64) * 0.09).cos().abs() * 0.35)
            .collect();
        (high, low)
    }

    fn naive_width(high: &[f64], low: &[f64], period: usize) -> Vec<f64> {
        let mut out = vec![f64::NAN; high.len()];
        let mut start = 0usize;
        while start < high.len() {
            while start < high.len() && !is_valid_pair(high[start], low[start]) {
                start += 1;
            }
            if start >= high.len() {
                break;
            }
            let mut end = start;
            while end < high.len() && is_valid_pair(high[end], low[end]) {
                end += 1;
            }
            if end - start >= period {
                for i in (start + period - 1)..end {
                    let mut upper = f64::NEG_INFINITY;
                    let mut lower = f64::INFINITY;
                    for j in (i + 1 - period)..=i {
                        upper = upper.max(high[j]);
                        lower = lower.min(low[j]);
                    }
                    out[i] = upper - lower;
                }
            }
            start = end;
        }
        out
    }

    fn assert_series_close(left: &[f64], right: &[f64], tol: f64) {
        assert_eq!(left.len(), right.len());
        for (a, b) in left.iter().zip(right.iter()) {
            if a.is_nan() || b.is_nan() {
                assert!(a.is_nan() && b.is_nan());
            } else {
                assert!((a - b).abs() <= tol, "left={a} right={b}");
            }
        }
    }

    #[test]
    fn donchian_channel_width_matches_naive() -> Result<(), Box<dyn Error>> {
        let (high, low) = sample_high_low(256);
        let input = DonchianChannelWidthInput::from_slices(
            &high,
            &low,
            DonchianChannelWidthParams::default(),
        );
        let out = donchian_channel_width_with_kernel(&input, Kernel::Scalar)?;
        let expected = naive_width(&high, &low, 20);
        assert_series_close(&out.values, &expected, 1e-12);
        Ok(())
    }

    #[test]
    fn donchian_channel_width_into_matches_api() -> Result<(), Box<dyn Error>> {
        let (high, low) = sample_high_low(200);
        let input = DonchianChannelWidthInput::from_slices(
            &high,
            &low,
            DonchianChannelWidthParams { period: Some(20) },
        );
        let base = donchian_channel_width(&input)?;
        let mut out = vec![f64::NAN; high.len()];
        donchian_channel_width_into_slice(&mut out, &input, Kernel::Auto)?;
        assert_series_close(&base.values, &out, 1e-12);
        Ok(())
    }

    #[test]
    fn donchian_channel_width_stream_matches_batch() -> Result<(), Box<dyn Error>> {
        let (high, low) = sample_high_low(256);
        let batch = donchian_channel_width(&DonchianChannelWidthInput::from_slices(
            &high,
            &low,
            DonchianChannelWidthParams { period: Some(20) },
        ))?;

        let mut stream =
            DonchianChannelWidthStream::try_new(DonchianChannelWidthParams { period: Some(20) })?;
        let mut streamed = Vec::with_capacity(high.len());
        for (&h, &l) in high.iter().zip(low.iter()) {
            streamed.push(stream.update(h, l).unwrap_or(f64::NAN));
        }
        assert_series_close(&batch.values, &streamed, 1e-12);
        Ok(())
    }

    #[test]
    fn donchian_channel_width_batch_single_matches_single() -> Result<(), Box<dyn Error>> {
        let (high, low) = sample_high_low(256);
        let single = donchian_channel_width(&DonchianChannelWidthInput::from_slices(
            &high,
            &low,
            DonchianChannelWidthParams { period: Some(20) },
        ))?;
        let batch = donchian_channel_width_batch_with_kernel(
            &high,
            &low,
            &DonchianChannelWidthBatchRange::default(),
            Kernel::Auto,
        )?;
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, high.len());
        assert_series_close(&single.values, &batch.values, 1e-12);
        Ok(())
    }

    #[test]
    fn donchian_channel_width_rejects_invalid_params() {
        let (high, low) = sample_high_low(32);
        let err = donchian_channel_width(&DonchianChannelWidthInput::from_slices(
            &high,
            &low,
            DonchianChannelWidthParams { period: Some(0) },
        ))
        .unwrap_err();
        assert!(matches!(
            err,
            DonchianChannelWidthError::InvalidPeriod { .. }
        ));

        let err = donchian_channel_width_batch_with_kernel(
            &high,
            &low,
            &DonchianChannelWidthBatchRange { period: (10, 5, 1) },
            Kernel::Auto,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            DonchianChannelWidthError::InvalidRange { .. }
        ));
    }

    #[test]
    fn donchian_channel_width_dispatch_compute_returns_value() -> Result<(), Box<dyn Error>> {
        let (high, low) = sample_high_low(192);
        let req = IndicatorComputeRequest {
            indicator_id: "donchian_channel_width",
            output_id: Some("value"),
            data: IndicatorDataRef::HighLow {
                high: &high,
                low: &low,
            },
            params: &[ParamKV {
                key: "period",
                value: ParamValue::Int(20),
            }],
            kernel: Kernel::Auto,
        };
        let out = compute_cpu(req)?;
        assert_eq!(out.output_id, "value");
        assert_eq!(out.rows, 1);
        assert_eq!(out.cols, high.len());
        Ok(())
    }
}
