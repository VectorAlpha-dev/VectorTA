#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;
#[cfg(feature = "python")]
use pyo3::wrap_pyfunction;

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use serde::{Deserialize, Serialize};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen::prelude::*;

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::mem::ManuallyDrop;
use thiserror::Error;

const DEFAULT_SHORT_LENGTH: usize = 3;
const DEFAULT_MEDIUM_LENGTH: usize = 8;
const DEFAULT_LONG_LENGTH: usize = 20;

impl<'a> AsRef<[f64]> for DidiIndexInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            DidiIndexData::Slice(slice) => slice,
            DidiIndexData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum DidiIndexData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct DidiIndexOutput {
    pub short: Vec<f64>,
    pub long: Vec<f64>,
    pub crossover: Vec<f64>,
    pub crossunder: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DidiIndexOutputField {
    Short,
    Long,
    Crossover,
    Crossunder,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct DidiIndexParams {
    pub short_length: Option<usize>,
    pub medium_length: Option<usize>,
    pub long_length: Option<usize>,
}

impl Default for DidiIndexParams {
    fn default() -> Self {
        Self {
            short_length: Some(DEFAULT_SHORT_LENGTH),
            medium_length: Some(DEFAULT_MEDIUM_LENGTH),
            long_length: Some(DEFAULT_LONG_LENGTH),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DidiIndexInput<'a> {
    pub data: DidiIndexData<'a>,
    pub params: DidiIndexParams,
}

impl<'a> DidiIndexInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: DidiIndexParams) -> Self {
        Self {
            data: DidiIndexData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(slice: &'a [f64], params: DidiIndexParams) -> Self {
        Self {
            data: DidiIndexData::Slice(slice),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "close", DidiIndexParams::default())
    }

    #[inline]
    pub fn get_short_length(&self) -> usize {
        self.params.short_length.unwrap_or(DEFAULT_SHORT_LENGTH)
    }

    #[inline]
    pub fn get_medium_length(&self) -> usize {
        self.params.medium_length.unwrap_or(DEFAULT_MEDIUM_LENGTH)
    }

    #[inline]
    pub fn get_long_length(&self) -> usize {
        self.params.long_length.unwrap_or(DEFAULT_LONG_LENGTH)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct DidiIndexBuilder {
    short_length: Option<usize>,
    medium_length: Option<usize>,
    long_length: Option<usize>,
    kernel: Kernel,
}

impl Default for DidiIndexBuilder {
    fn default() -> Self {
        Self {
            short_length: None,
            medium_length: None,
            long_length: None,
            kernel: Kernel::Auto,
        }
    }
}

impl DidiIndexBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn short_length(mut self, short_length: usize) -> Self {
        self.short_length = Some(short_length);
        self
    }

    #[inline]
    pub fn medium_length(mut self, medium_length: usize) -> Self {
        self.medium_length = Some(medium_length);
        self
    }

    #[inline]
    pub fn long_length(mut self, long_length: usize) -> Self {
        self.long_length = Some(long_length);
        self
    }

    #[inline]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline]
    pub fn apply(self, candles: &Candles, source: &str) -> Result<DidiIndexOutput, DidiIndexError> {
        let input = DidiIndexInput::from_candles(
            candles,
            source,
            DidiIndexParams {
                short_length: self.short_length,
                medium_length: self.medium_length,
                long_length: self.long_length,
            },
        );
        didi_index_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn apply_slice(self, data: &[f64]) -> Result<DidiIndexOutput, DidiIndexError> {
        let input = DidiIndexInput::from_slice(
            data,
            DidiIndexParams {
                short_length: self.short_length,
                medium_length: self.medium_length,
                long_length: self.long_length,
            },
        );
        didi_index_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn into_stream(self) -> Result<DidiIndexStream, DidiIndexError> {
        DidiIndexStream::try_new(DidiIndexParams {
            short_length: self.short_length,
            medium_length: self.medium_length,
            long_length: self.long_length,
        })
    }
}

#[derive(Debug, Error)]
pub enum DidiIndexError {
    #[error("didi_index: Input data slice is empty.")]
    EmptyInputData,
    #[error("didi_index: All values are NaN.")]
    AllValuesNaN,
    #[error(
        "didi_index: Invalid short_length: short_length = {short_length}, data length = {data_len}"
    )]
    InvalidShortLength {
        short_length: usize,
        data_len: usize,
    },
    #[error("didi_index: Invalid medium_length: medium_length = {medium_length}, data length = {data_len}")]
    InvalidMediumLength {
        medium_length: usize,
        data_len: usize,
    },
    #[error(
        "didi_index: Invalid long_length: long_length = {long_length}, data length = {data_len}"
    )]
    InvalidLongLength { long_length: usize, data_len: usize },
    #[error("didi_index: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("didi_index: Output length mismatch: expected = {expected}, short = {short_got}, long = {long_got}, crossover = {crossover_got}, crossunder = {crossunder_got}")]
    OutputLengthMismatch {
        expected: usize,
        short_got: usize,
        long_got: usize,
        crossover_got: usize,
        crossunder_got: usize,
    },
    #[error("didi_index: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("didi_index: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Debug, Clone)]
struct SmaWindow {
    period: usize,
    values: Vec<f64>,
    idx: usize,
    count: usize,
    sum: f64,
}

impl SmaWindow {
    #[inline]
    fn new(period: usize) -> Self {
        Self {
            period,
            values: vec![0.0; period.max(1)],
            idx: 0,
            count: 0,
            sum: 0.0,
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.idx = 0;
        self.count = 0;
        self.sum = 0.0;
    }

    #[inline]
    fn update(&mut self, value: f64) -> Option<f64> {
        if self.count < self.period {
            self.values[self.idx] = value;
            self.sum += value;
            self.count += 1;
            self.idx += 1;
            if self.idx == self.period {
                self.idx = 0;
            }
            if self.count == self.period {
                Some(self.sum / self.period as f64)
            } else {
                None
            }
        } else {
            let old = self.values[self.idx];
            self.values[self.idx] = value;
            self.sum += value - old;
            self.idx += 1;
            if self.idx == self.period {
                self.idx = 0;
            }
            Some(self.sum / self.period as f64)
        }
    }
}

#[derive(Debug, Clone)]
pub struct DidiIndexStream {
    short: SmaWindow,
    medium: SmaWindow,
    long: SmaWindow,
    prev_short: f64,
    prev_long: f64,
    have_prev: bool,
    warmup: usize,
}

impl DidiIndexStream {
    pub fn try_new(params: DidiIndexParams) -> Result<Self, DidiIndexError> {
        let short_length = params.short_length.unwrap_or(DEFAULT_SHORT_LENGTH);
        if short_length == 0 {
            return Err(DidiIndexError::InvalidShortLength {
                short_length,
                data_len: 0,
            });
        }
        let medium_length = params.medium_length.unwrap_or(DEFAULT_MEDIUM_LENGTH);
        if medium_length == 0 {
            return Err(DidiIndexError::InvalidMediumLength {
                medium_length,
                data_len: 0,
            });
        }
        let long_length = params.long_length.unwrap_or(DEFAULT_LONG_LENGTH);
        if long_length == 0 {
            return Err(DidiIndexError::InvalidLongLength {
                long_length,
                data_len: 0,
            });
        }
        Ok(Self {
            short: SmaWindow::new(short_length),
            medium: SmaWindow::new(medium_length),
            long: SmaWindow::new(long_length),
            prev_short: f64::NAN,
            prev_long: f64::NAN,
            have_prev: false,
            warmup: short_length.max(medium_length).max(long_length) - 1,
        })
    }

    #[inline]
    fn reset(&mut self) {
        self.short.reset();
        self.medium.reset();
        self.long.reset();
        self.prev_short = f64::NAN;
        self.prev_long = f64::NAN;
        self.have_prev = false;
    }

    #[inline]
    pub fn update(&mut self, value: f64) -> Option<(f64, f64, f64, f64)> {
        if !valid_value(value) {
            self.reset();
            return None;
        }

        let short_ma = self.short.update(value);
        let medium_ma = self.medium.update(value);
        let long_ma = self.long.update(value);
        if short_ma.is_none() || medium_ma.is_none() || long_ma.is_none() {
            self.have_prev = false;
            return None;
        }

        let medium_ma = medium_ma.unwrap_or(f64::NAN);
        if !medium_ma.is_finite() || medium_ma == 0.0 {
            self.have_prev = false;
            return Some((f64::NAN, f64::NAN, f64::NAN, f64::NAN));
        }

        let short = short_ma.unwrap_or(f64::NAN) / medium_ma;
        let long = long_ma.unwrap_or(f64::NAN) / medium_ma;
        if !short.is_finite() || !long.is_finite() {
            self.have_prev = false;
            return Some((f64::NAN, f64::NAN, f64::NAN, f64::NAN));
        }

        let crossover = if self.have_prev && short > long && self.prev_short <= self.prev_long {
            1.0
        } else {
            0.0
        };
        let crossunder = if self.have_prev && short < long && self.prev_short >= self.prev_long {
            1.0
        } else {
            0.0
        };
        self.prev_short = short;
        self.prev_long = long;
        self.have_prev = true;
        Some((short, long, crossover, crossunder))
    }

    #[inline]
    pub fn get_warmup_period(&self) -> usize {
        self.warmup
    }
}

#[inline]
pub fn didi_index(input: &DidiIndexInput) -> Result<DidiIndexOutput, DidiIndexError> {
    didi_index_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn valid_value(value: f64) -> bool {
    value.is_finite()
}

#[inline(always)]
fn first_valid_value(data: &[f64]) -> usize {
    let mut i = 0usize;
    while i < data.len() {
        if valid_value(data[i]) {
            break;
        }
        i += 1;
    }
    i.min(data.len())
}

#[inline(always)]
fn count_valid_values(data: &[f64]) -> usize {
    data.iter().filter(|v| valid_value(**v)).count()
}

#[inline(always)]
fn didi_index_row_from_slice(
    data: &[f64],
    params: &DidiIndexParams,
    short_out: &mut [f64],
    long_out: &mut [f64],
    crossover_out: &mut [f64],
    crossunder_out: &mut [f64],
) -> Result<(), DidiIndexError> {
    let mut stream = DidiIndexStream::try_new(params.clone())?;
    for i in 0..data.len() {
        match stream.update(data[i]) {
            Some((short, long, crossover, crossunder)) => {
                short_out[i] = short;
                long_out[i] = long;
                crossover_out[i] = crossover;
                crossunder_out[i] = crossunder;
            }
            None => {
                short_out[i] = f64::NAN;
                long_out[i] = f64::NAN;
                crossover_out[i] = f64::NAN;
                crossunder_out[i] = f64::NAN;
            }
        }
    }
    Ok(())
}

#[inline(always)]
fn didi_index_selected_row_from_slice(
    data: &[f64],
    params: &DidiIndexParams,
    field: DidiIndexOutputField,
    out: &mut [f64],
) -> Result<(), DidiIndexError> {
    let mut stream = DidiIndexStream::try_new(params.clone())?;
    match field {
        DidiIndexOutputField::Short => {
            for i in 0..data.len() {
                out[i] = match stream.update(data[i]) {
                    Some((short, _, _, _)) => short,
                    None => f64::NAN,
                };
            }
        }
        DidiIndexOutputField::Long => {
            for i in 0..data.len() {
                out[i] = match stream.update(data[i]) {
                    Some((_, long, _, _)) => long,
                    None => f64::NAN,
                };
            }
        }
        DidiIndexOutputField::Crossover => {
            for i in 0..data.len() {
                out[i] = match stream.update(data[i]) {
                    Some((_, _, crossover, _)) => crossover,
                    None => f64::NAN,
                };
            }
        }
        DidiIndexOutputField::Crossunder => {
            for i in 0..data.len() {
                out[i] = match stream.update(data[i]) {
                    Some((_, _, _, crossunder)) => crossunder,
                    None => f64::NAN,
                };
            }
        }
    }
    Ok(())
}

#[inline(always)]
fn didi_index_prepare<'a>(
    input: &'a DidiIndexInput,
    kernel: Kernel,
) -> Result<(&'a [f64], usize, DidiIndexParams, Kernel), DidiIndexError> {
    let data = input.as_ref();
    if data.is_empty() {
        return Err(DidiIndexError::EmptyInputData);
    }

    let first = first_valid_value(data);
    if first >= data.len() {
        return Err(DidiIndexError::AllValuesNaN);
    }

    let params = input.params.clone();
    let short_length = params.short_length.unwrap_or(DEFAULT_SHORT_LENGTH);
    let medium_length = params.medium_length.unwrap_or(DEFAULT_MEDIUM_LENGTH);
    let long_length = params.long_length.unwrap_or(DEFAULT_LONG_LENGTH);
    let len = data.len();
    if short_length == 0 || short_length > len {
        return Err(DidiIndexError::InvalidShortLength {
            short_length,
            data_len: len,
        });
    }
    if medium_length == 0 || medium_length > len {
        return Err(DidiIndexError::InvalidMediumLength {
            medium_length,
            data_len: len,
        });
    }
    if long_length == 0 || long_length > len {
        return Err(DidiIndexError::InvalidLongLength {
            long_length,
            data_len: len,
        });
    }

    let needed = short_length.max(medium_length).max(long_length);
    let valid = count_valid_values(data);
    if valid < needed {
        return Err(DidiIndexError::NotEnoughValidData { needed, valid });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other.to_non_batch(),
    };
    Ok((data, first, params, chosen))
}

#[inline]
pub fn didi_index_with_kernel(
    input: &DidiIndexInput,
    kernel: Kernel,
) -> Result<DidiIndexOutput, DidiIndexError> {
    let (data, first, params, _chosen) = didi_index_prepare(input, kernel)?;
    let mut short = alloc_with_nan_prefix(data.len(), first);
    let mut long = alloc_with_nan_prefix(data.len(), first);
    let mut crossover = alloc_with_nan_prefix(data.len(), first);
    let mut crossunder = alloc_with_nan_prefix(data.len(), first);
    didi_index_row_from_slice(
        data,
        &params,
        &mut short,
        &mut long,
        &mut crossover,
        &mut crossunder,
    )?;
    Ok(DidiIndexOutput {
        short,
        long,
        crossover,
        crossunder,
    })
}

#[inline]
pub fn didi_index_into_slices(
    short_out: &mut [f64],
    long_out: &mut [f64],
    crossover_out: &mut [f64],
    crossunder_out: &mut [f64],
    input: &DidiIndexInput,
    kernel: Kernel,
) -> Result<(), DidiIndexError> {
    let (data, _first, params, _chosen) = didi_index_prepare(input, kernel)?;
    if short_out.len() != data.len()
        || long_out.len() != data.len()
        || crossover_out.len() != data.len()
        || crossunder_out.len() != data.len()
    {
        return Err(DidiIndexError::OutputLengthMismatch {
            expected: data.len(),
            short_got: short_out.len(),
            long_got: long_out.len(),
            crossover_got: crossover_out.len(),
            crossunder_got: crossunder_out.len(),
        });
    }
    didi_index_row_from_slice(
        data,
        &params,
        short_out,
        long_out,
        crossover_out,
        crossunder_out,
    )
}

#[inline]
pub fn didi_index_output_into_slice(
    out: &mut [f64],
    input: &DidiIndexInput,
    kernel: Kernel,
    field: DidiIndexOutputField,
) -> Result<(), DidiIndexError> {
    let (data, _first, params, _chosen) = didi_index_prepare(input, kernel)?;
    if out.len() != data.len() {
        return Err(DidiIndexError::OutputLengthMismatch {
            expected: data.len(),
            short_got: out.len(),
            long_got: out.len(),
            crossover_got: out.len(),
            crossunder_got: out.len(),
        });
    }
    didi_index_selected_row_from_slice(data, &params, field, out)
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn didi_index_into(
    input: &DidiIndexInput,
    short_out: &mut [f64],
    long_out: &mut [f64],
    crossover_out: &mut [f64],
    crossunder_out: &mut [f64],
) -> Result<(), DidiIndexError> {
    didi_index_into_slices(
        short_out,
        long_out,
        crossover_out,
        crossunder_out,
        input,
        Kernel::Auto,
    )
}

#[derive(Clone, Debug)]
pub struct DidiIndexBatchRange {
    pub short_length: (usize, usize, usize),
    pub medium_length: (usize, usize, usize),
    pub long_length: (usize, usize, usize),
}

impl Default for DidiIndexBatchRange {
    fn default() -> Self {
        Self {
            short_length: (DEFAULT_SHORT_LENGTH, DEFAULT_SHORT_LENGTH, 0),
            medium_length: (DEFAULT_MEDIUM_LENGTH, DEFAULT_MEDIUM_LENGTH, 0),
            long_length: (DEFAULT_LONG_LENGTH, DEFAULT_LONG_LENGTH, 0),
        }
    }
}

#[derive(Clone, Debug)]
pub struct DidiIndexBatchBuilder {
    range: DidiIndexBatchRange,
    kernel: Kernel,
}

impl Default for DidiIndexBatchBuilder {
    fn default() -> Self {
        Self {
            range: DidiIndexBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl DidiIndexBatchBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn short_length_range(mut self, range: (usize, usize, usize)) -> Self {
        self.range.short_length = range;
        self
    }

    #[inline]
    pub fn medium_length_range(mut self, range: (usize, usize, usize)) -> Self {
        self.range.medium_length = range;
        self
    }

    #[inline]
    pub fn long_length_range(mut self, range: (usize, usize, usize)) -> Self {
        self.range.long_length = range;
        self
    }

    #[inline]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline]
    pub fn apply_slice(self, data: &[f64]) -> Result<DidiIndexBatchOutput, DidiIndexError> {
        didi_index_batch_with_kernel(data, &self.range, self.kernel)
    }

    #[inline]
    pub fn apply_candles(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<DidiIndexBatchOutput, DidiIndexError> {
        self.apply_slice(source_type(candles, source))
    }

    #[inline]
    pub fn with_default_candles(candles: &Candles) -> Result<DidiIndexBatchOutput, DidiIndexError> {
        DidiIndexBatchBuilder::new().apply_candles(candles, "close")
    }
}

#[derive(Clone, Debug)]
pub struct DidiIndexBatchOutput {
    pub short: Vec<f64>,
    pub long: Vec<f64>,
    pub crossover: Vec<f64>,
    pub crossunder: Vec<f64>,
    pub combos: Vec<DidiIndexParams>,
    pub rows: usize,
    pub cols: usize,
}

impl DidiIndexBatchOutput {
    pub fn row_for_params(&self, params: &DidiIndexParams) -> Option<usize> {
        self.combos.iter().position(|combo| combo == params)
    }

    pub fn short_for(&self, params: &DidiIndexParams) -> Option<&[f64]> {
        self.row_for_params(params).and_then(|row| {
            row.checked_mul(self.cols)
                .and_then(|start| self.short.get(start..start + self.cols))
        })
    }

    pub fn long_for(&self, params: &DidiIndexParams) -> Option<&[f64]> {
        self.row_for_params(params).and_then(|row| {
            row.checked_mul(self.cols)
                .and_then(|start| self.long.get(start..start + self.cols))
        })
    }
}

#[inline(always)]
fn axis_usize((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, DidiIndexError> {
    if step == 0 || start == end {
        return Ok(vec![start]);
    }
    let step = step.max(1);
    if start < end {
        let mut out = Vec::new();
        let mut x = start;
        while x <= end {
            out.push(x);
            match x.checked_add(step) {
                Some(next) if next != x => x = next,
                _ => break,
            }
        }
        if out.is_empty() {
            return Err(DidiIndexError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        Ok(out)
    } else {
        let mut out = Vec::new();
        let mut x = start;
        loop {
            out.push(x);
            if x == end {
                break;
            }
            let next = x.saturating_sub(step);
            if next == x || next < end {
                break;
            }
            x = next;
        }
        if out.is_empty() {
            return Err(DidiIndexError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        Ok(out)
    }
}

#[inline(always)]
fn expand_grid_didi_index(
    range: &DidiIndexBatchRange,
) -> Result<Vec<DidiIndexParams>, DidiIndexError> {
    let shorts = axis_usize(range.short_length)?;
    let mediums = axis_usize(range.medium_length)?;
    let longs = axis_usize(range.long_length)?;

    if let Some(&short_length) = shorts.iter().find(|&&value| value == 0) {
        return Err(DidiIndexError::InvalidShortLength {
            short_length,
            data_len: 0,
        });
    }
    if let Some(&medium_length) = mediums.iter().find(|&&value| value == 0) {
        return Err(DidiIndexError::InvalidMediumLength {
            medium_length,
            data_len: 0,
        });
    }
    if let Some(&long_length) = longs.iter().find(|&&value| value == 0) {
        return Err(DidiIndexError::InvalidLongLength {
            long_length,
            data_len: 0,
        });
    }

    let mut out = Vec::with_capacity(shorts.len() * mediums.len() * longs.len());
    for &short_length in &shorts {
        for &medium_length in &mediums {
            for &long_length in &longs {
                out.push(DidiIndexParams {
                    short_length: Some(short_length),
                    medium_length: Some(medium_length),
                    long_length: Some(long_length),
                });
            }
        }
    }
    Ok(out)
}

pub fn didi_index_batch_with_kernel(
    data: &[f64],
    sweep: &DidiIndexBatchRange,
    kernel: Kernel,
) -> Result<DidiIndexBatchOutput, DidiIndexError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(DidiIndexError::InvalidKernelForBatch(other)),
    };
    didi_index_batch_inner(data, sweep, batch_kernel.to_non_batch(), false)
}

#[inline]
pub fn didi_index_batch_slice(
    data: &[f64],
    sweep: &DidiIndexBatchRange,
) -> Result<DidiIndexBatchOutput, DidiIndexError> {
    didi_index_batch_with_kernel(data, sweep, Kernel::Auto)
}

#[inline]
pub fn didi_index_batch_par_slice(
    data: &[f64],
    sweep: &DidiIndexBatchRange,
) -> Result<DidiIndexBatchOutput, DidiIndexError> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let kernel = detect_best_batch_kernel().to_non_batch();
        return didi_index_batch_inner(data, sweep, kernel, true);
    }
    #[cfg(target_arch = "wasm32")]
    {
        didi_index_batch_inner(data, sweep, detect_best_kernel(), false)
    }
}

pub fn didi_index_batch_inner(
    data: &[f64],
    sweep: &DidiIndexBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<DidiIndexBatchOutput, DidiIndexError> {
    if data.is_empty() {
        return Err(DidiIndexError::EmptyInputData);
    }
    let first = first_valid_value(data);
    if first >= data.len() {
        return Err(DidiIndexError::AllValuesNaN);
    }

    let combos = expand_grid_didi_index(sweep)?;
    let rows = combos.len();
    let cols = data.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| DidiIndexError::OutputLengthMismatch {
            expected: usize::MAX,
            short_got: 0,
            long_got: 0,
            crossover_got: 0,
            crossunder_got: 0,
        })?;

    let valid = count_valid_values(data);
    let mut warms = Vec::with_capacity(rows);
    for combo in &combos {
        let short_length = combo.short_length.unwrap_or(DEFAULT_SHORT_LENGTH);
        let medium_length = combo.medium_length.unwrap_or(DEFAULT_MEDIUM_LENGTH);
        let long_length = combo.long_length.unwrap_or(DEFAULT_LONG_LENGTH);
        let needed = short_length.max(medium_length).max(long_length);
        if short_length > cols {
            return Err(DidiIndexError::InvalidShortLength {
                short_length,
                data_len: cols,
            });
        }
        if medium_length > cols {
            return Err(DidiIndexError::InvalidMediumLength {
                medium_length,
                data_len: cols,
            });
        }
        if long_length > cols {
            return Err(DidiIndexError::InvalidLongLength {
                long_length,
                data_len: cols,
            });
        }
        if valid < needed {
            return Err(DidiIndexError::NotEnoughValidData { needed, valid });
        }
        warms.push((first + needed - 1).min(cols));
    }

    let mut short_mu = make_uninit_matrix(rows, cols);
    let mut long_mu = make_uninit_matrix(rows, cols);
    let mut crossover_mu = make_uninit_matrix(rows, cols);
    let mut crossunder_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut short_mu, cols, &warms);
    init_matrix_prefixes(&mut long_mu, cols, &warms);
    init_matrix_prefixes(&mut crossover_mu, cols, &warms);
    init_matrix_prefixes(&mut crossunder_mu, cols, &warms);

    let mut short_guard = ManuallyDrop::new(short_mu);
    let mut long_guard = ManuallyDrop::new(long_mu);
    let mut crossover_guard = ManuallyDrop::new(crossover_mu);
    let mut crossunder_guard = ManuallyDrop::new(crossunder_mu);

    let short_out =
        unsafe { std::slice::from_raw_parts_mut(short_guard.as_mut_ptr() as *mut f64, total) };
    let long_out =
        unsafe { std::slice::from_raw_parts_mut(long_guard.as_mut_ptr() as *mut f64, total) };
    let crossover_out =
        unsafe { std::slice::from_raw_parts_mut(crossover_guard.as_mut_ptr() as *mut f64, total) };
    let crossunder_out =
        unsafe { std::slice::from_raw_parts_mut(crossunder_guard.as_mut_ptr() as *mut f64, total) };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            short_out
                .par_chunks_mut(cols)
                .zip(long_out.par_chunks_mut(cols))
                .zip(crossover_out.par_chunks_mut(cols))
                .zip(crossunder_out.par_chunks_mut(cols))
                .zip(combos.par_iter())
                .for_each(
                    |((((dst_short, dst_long), dst_crossover), dst_crossunder), combo)| {
                        let _ = didi_index_row_from_slice(
                            data,
                            combo,
                            dst_short,
                            dst_long,
                            dst_crossover,
                            dst_crossunder,
                        );
                    },
                );
        }
    } else {
        let _ = kernel;
        for (row, combo) in combos.iter().enumerate() {
            let start = row * cols;
            let end = start + cols;
            didi_index_row_from_slice(
                data,
                combo,
                &mut short_out[start..end],
                &mut long_out[start..end],
                &mut crossover_out[start..end],
                &mut crossunder_out[start..end],
            )?;
        }
    }

    let short = unsafe {
        Vec::from_raw_parts(
            short_guard.as_mut_ptr() as *mut f64,
            short_guard.len(),
            short_guard.capacity(),
        )
    };
    let long = unsafe {
        Vec::from_raw_parts(
            long_guard.as_mut_ptr() as *mut f64,
            long_guard.len(),
            long_guard.capacity(),
        )
    };
    let crossover = unsafe {
        Vec::from_raw_parts(
            crossover_guard.as_mut_ptr() as *mut f64,
            crossover_guard.len(),
            crossover_guard.capacity(),
        )
    };
    let crossunder = unsafe {
        Vec::from_raw_parts(
            crossunder_guard.as_mut_ptr() as *mut f64,
            crossunder_guard.len(),
            crossunder_guard.capacity(),
        )
    };
    core::mem::forget(short_guard);
    core::mem::forget(long_guard);
    core::mem::forget(crossover_guard);
    core::mem::forget(crossunder_guard);

    Ok(DidiIndexBatchOutput {
        short,
        long,
        crossover,
        crossunder,
        combos,
        rows,
        cols,
    })
}

pub fn didi_index_batch_inner_into(
    data: &[f64],
    sweep: &DidiIndexBatchRange,
    kernel: Kernel,
    short_out: &mut [f64],
    long_out: &mut [f64],
    crossover_out: &mut [f64],
    crossunder_out: &mut [f64],
) -> Result<Vec<DidiIndexParams>, DidiIndexError> {
    let out = didi_index_batch_inner(data, sweep, kernel, false)?;
    let total = out.rows * out.cols;
    if short_out.len() != total
        || long_out.len() != total
        || crossover_out.len() != total
        || crossunder_out.len() != total
    {
        return Err(DidiIndexError::OutputLengthMismatch {
            expected: total,
            short_got: short_out.len(),
            long_got: long_out.len(),
            crossover_got: crossover_out.len(),
            crossunder_got: crossunder_out.len(),
        });
    }
    short_out.copy_from_slice(&out.short);
    long_out.copy_from_slice(&out.long);
    crossover_out.copy_from_slice(&out.crossover);
    crossunder_out.copy_from_slice(&out.crossunder);
    Ok(out.combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "didi_index")]
#[pyo3(signature = (data, short_length=None, medium_length=None, long_length=None, kernel=None))]
pub fn didi_index_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    short_length: Option<usize>,
    medium_length: Option<usize>,
    long_length: Option<usize>,
    kernel: Option<&str>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let data = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;
    let input = DidiIndexInput::from_slice(
        data,
        DidiIndexParams {
            short_length,
            medium_length,
            long_length,
        },
    );
    let out = py
        .allow_threads(|| didi_index_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        out.short.into_pyarray(py),
        out.long.into_pyarray(py),
        out.crossover.into_pyarray(py),
        out.crossunder.into_pyarray(py),
    ))
}

#[cfg(feature = "python")]
#[pyclass(name = "DidiIndexStream")]
pub struct DidiIndexStreamPy {
    inner: DidiIndexStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl DidiIndexStreamPy {
    #[new]
    #[pyo3(signature = (short_length=DEFAULT_SHORT_LENGTH, medium_length=DEFAULT_MEDIUM_LENGTH, long_length=DEFAULT_LONG_LENGTH))]
    fn new(short_length: usize, medium_length: usize, long_length: usize) -> PyResult<Self> {
        let inner = DidiIndexStream::try_new(DidiIndexParams {
            short_length: Some(short_length),
            medium_length: Some(medium_length),
            long_length: Some(long_length),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn update(&mut self, value: f64) -> Option<(f64, f64, f64, f64)> {
        self.inner.update(value)
    }

    #[getter]
    fn warmup_period(&self) -> usize {
        self.inner.get_warmup_period()
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "didi_index_batch")]
#[pyo3(signature = (data, short_length_range=(DEFAULT_SHORT_LENGTH, DEFAULT_SHORT_LENGTH, 0), medium_length_range=(DEFAULT_MEDIUM_LENGTH, DEFAULT_MEDIUM_LENGTH, 0), long_length_range=(DEFAULT_LONG_LENGTH, DEFAULT_LONG_LENGTH, 0), kernel=None))]
pub fn didi_index_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    short_length_range: (usize, usize, usize),
    medium_length_range: (usize, usize, usize),
    long_length_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let data = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    let sweep = DidiIndexBatchRange {
        short_length: short_length_range,
        medium_length: medium_length_range,
        long_length: long_length_range,
    };
    let combos =
        expand_grid_didi_index(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = data.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;

    let short_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let long_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let crossover_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let crossunder_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let short_slice = unsafe { short_arr.as_slice_mut()? };
    let long_slice = unsafe { long_arr.as_slice_mut()? };
    let crossover_slice = unsafe { crossover_arr.as_slice_mut()? };
    let crossunder_slice = unsafe { crossunder_arr.as_slice_mut()? };

    let combos = py
        .allow_threads(|| {
            let batch = match kern {
                Kernel::Auto => detect_best_batch_kernel(),
                other => other,
            };
            didi_index_batch_inner_into(
                data,
                &sweep,
                batch.to_non_batch(),
                short_slice,
                long_slice,
                crossover_slice,
                crossunder_slice,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("short", short_arr.reshape((rows, cols))?)?;
    dict.set_item("long", long_arr.reshape((rows, cols))?)?;
    dict.set_item("crossover", crossover_arr.reshape((rows, cols))?)?;
    dict.set_item("crossunder", crossunder_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "short_lengths",
        combos
            .iter()
            .map(|p| p.short_length.unwrap_or(DEFAULT_SHORT_LENGTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "medium_lengths",
        combos
            .iter()
            .map(|p| p.medium_length.unwrap_or(DEFAULT_MEDIUM_LENGTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "long_lengths",
        combos
            .iter()
            .map(|p| p.long_length.unwrap_or(DEFAULT_LONG_LENGTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_didi_index_module(module: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(didi_index_py, module)?)?;
    module.add_function(wrap_pyfunction!(didi_index_batch_py, module)?)?;
    module.add_class::<DidiIndexStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "didi_index_js")]
pub fn didi_index_js(
    data: &[f64],
    short_length: usize,
    medium_length: usize,
    long_length: usize,
) -> Result<JsValue, JsValue> {
    let input = DidiIndexInput::from_slice(
        data,
        DidiIndexParams {
            short_length: Some(short_length),
            medium_length: Some(medium_length),
            long_length: Some(long_length),
        },
    );
    let out = didi_index(&input).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let result = js_sys::Object::new();

    let short = js_sys::Float64Array::new_with_length(out.short.len() as u32);
    short.copy_from(&out.short);
    js_sys::Reflect::set(&result, &JsValue::from_str("short"), &short)?;

    let long = js_sys::Float64Array::new_with_length(out.long.len() as u32);
    long.copy_from(&out.long);
    js_sys::Reflect::set(&result, &JsValue::from_str("long"), &long)?;

    let crossover = js_sys::Float64Array::new_with_length(out.crossover.len() as u32);
    crossover.copy_from(&out.crossover);
    js_sys::Reflect::set(&result, &JsValue::from_str("crossover"), &crossover)?;

    let crossunder = js_sys::Float64Array::new_with_length(out.crossunder.len() as u32);
    crossunder.copy_from(&out.crossunder);
    js_sys::Reflect::set(&result, &JsValue::from_str("crossunder"), &crossunder)?;

    Ok(result.into())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn didi_index_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn didi_index_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn didi_index_into(
    data_ptr: *const f64,
    short_ptr: *mut f64,
    long_ptr: *mut f64,
    crossover_ptr: *mut f64,
    crossunder_ptr: *mut f64,
    len: usize,
    short_length: usize,
    medium_length: usize,
    long_length: usize,
) -> Result<(), JsValue> {
    if data_ptr.is_null()
        || short_ptr.is_null()
        || long_ptr.is_null()
        || crossover_ptr.is_null()
        || crossunder_ptr.is_null()
    {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(data_ptr, len);
        let input = DidiIndexInput::from_slice(
            data,
            DidiIndexParams {
                short_length: Some(short_length),
                medium_length: Some(medium_length),
                long_length: Some(long_length),
            },
        );
        let alias = data_ptr == short_ptr
            || data_ptr == long_ptr
            || data_ptr == crossover_ptr
            || data_ptr == crossunder_ptr;
        if alias {
            let mut short_tmp = vec![0.0; len];
            let mut long_tmp = vec![0.0; len];
            let mut crossover_tmp = vec![0.0; len];
            let mut crossunder_tmp = vec![0.0; len];
            didi_index_into_slices(
                &mut short_tmp,
                &mut long_tmp,
                &mut crossover_tmp,
                &mut crossunder_tmp,
                &input,
                Kernel::Auto,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
            std::slice::from_raw_parts_mut(short_ptr, len).copy_from_slice(&short_tmp);
            std::slice::from_raw_parts_mut(long_ptr, len).copy_from_slice(&long_tmp);
            std::slice::from_raw_parts_mut(crossover_ptr, len).copy_from_slice(&crossover_tmp);
            std::slice::from_raw_parts_mut(crossunder_ptr, len).copy_from_slice(&crossunder_tmp);
        } else {
            didi_index_into_slices(
                std::slice::from_raw_parts_mut(short_ptr, len),
                std::slice::from_raw_parts_mut(long_ptr, len),
                std::slice::from_raw_parts_mut(crossover_ptr, len),
                std::slice::from_raw_parts_mut(crossunder_ptr, len),
                &input,
                Kernel::Auto,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct DidiIndexBatchConfig {
    pub short_length_range: (usize, usize, usize),
    pub medium_length_range: Option<(usize, usize, usize)>,
    pub long_length_range: Option<(usize, usize, usize)>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct DidiIndexBatchJsOutput {
    pub short: Vec<f64>,
    pub long: Vec<f64>,
    pub crossover: Vec<f64>,
    pub crossunder: Vec<f64>,
    pub combos: Vec<DidiIndexParams>,
    pub short_lengths: Vec<usize>,
    pub medium_lengths: Vec<usize>,
    pub long_lengths: Vec<usize>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "didi_index_batch_js")]
pub fn didi_index_batch_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: DidiIndexBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    let sweep = DidiIndexBatchRange {
        short_length: config.short_length_range,
        medium_length: config.medium_length_range.unwrap_or((
            DEFAULT_MEDIUM_LENGTH,
            DEFAULT_MEDIUM_LENGTH,
            0,
        )),
        long_length: config.long_length_range.unwrap_or((
            DEFAULT_LONG_LENGTH,
            DEFAULT_LONG_LENGTH,
            0,
        )),
    };
    let out = didi_index_batch_inner(
        data,
        &sweep,
        detect_best_batch_kernel().to_non_batch(),
        false,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&DidiIndexBatchJsOutput {
        short_lengths: out
            .combos
            .iter()
            .map(|p| p.short_length.unwrap_or(DEFAULT_SHORT_LENGTH))
            .collect(),
        medium_lengths: out
            .combos
            .iter()
            .map(|p| p.medium_length.unwrap_or(DEFAULT_MEDIUM_LENGTH))
            .collect(),
        long_lengths: out
            .combos
            .iter()
            .map(|p| p.long_length.unwrap_or(DEFAULT_LONG_LENGTH))
            .collect(),
        short: out.short,
        long: out.long,
        crossover: out.crossover,
        crossunder: out.crossunder,
        combos: out.combos,
        rows: out.rows,
        cols: out.cols,
    })
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn didi_index_batch_into(
    data_ptr: *const f64,
    short_ptr: *mut f64,
    long_ptr: *mut f64,
    crossover_ptr: *mut f64,
    crossunder_ptr: *mut f64,
    len: usize,
    short_start: usize,
    short_end: usize,
    short_step: usize,
    medium_start: usize,
    medium_end: usize,
    medium_step: usize,
    long_start: usize,
    long_end: usize,
    long_step: usize,
) -> Result<usize, JsValue> {
    if data_ptr.is_null()
        || short_ptr.is_null()
        || long_ptr.is_null()
        || crossover_ptr.is_null()
        || crossunder_ptr.is_null()
    {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    let sweep = DidiIndexBatchRange {
        short_length: (short_start, short_end, short_step),
        medium_length: (medium_start, medium_end, medium_step),
        long_length: (long_start, long_end, long_step),
    };
    let combos = expand_grid_didi_index(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();
    let total = rows
        .checked_mul(len)
        .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;

    unsafe {
        let data = std::slice::from_raw_parts(data_ptr, len);
        didi_index_batch_inner_into(
            data,
            &sweep,
            detect_best_batch_kernel().to_non_batch(),
            std::slice::from_raw_parts_mut(short_ptr, total),
            std::slice::from_raw_parts_mut(long_ptr, total),
            std::slice::from_raw_parts_mut(crossover_ptr, total),
            std::slice::from_raw_parts_mut(crossunder_ptr, total),
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }
    Ok(rows)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn didi_index_output_into_js(
    data: &[f64],
    short_length: usize,
    medium_length: usize,
    long_length: usize,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = didi_index_js(data, short_length, medium_length, long_length)?;
    crate::write_wasm_object_f64_outputs("didi_index_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn didi_index_batch_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = didi_index_batch_js(data, config)?;
    crate::write_wasm_selected_object_f64_outputs("didi_index_batch_output_into_js", &value, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-12
    }

    fn approx_eq_or_nan(a: f64, b: f64) -> bool {
        (a.is_nan() && b.is_nan()) || approx_eq(a, b)
    }

    #[test]
    fn didi_index_matches_manual_ratios() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = DidiIndexInput::from_slice(
            &data,
            DidiIndexParams {
                short_length: Some(2),
                medium_length: Some(3),
                long_length: Some(4),
            },
        );
        let out = didi_index(&input).unwrap();

        assert!(out.short[..3].iter().all(|v| v.is_nan()));
        assert!(out.long[..3].iter().all(|v| v.is_nan()));
        assert!(approx_eq(out.short[3], 3.5 / 3.0));
        assert!(approx_eq(out.long[3], 2.5 / 3.0));
        assert!(approx_eq(out.short[4], 4.5 / 4.0));
        assert!(approx_eq(out.long[4], 3.5 / 4.0));
        assert!(approx_eq(out.crossover[3], 0.0));
        assert!(approx_eq(out.crossunder[3], 0.0));
    }

    #[test]
    fn didi_index_detects_crossover_and_crossunder() {
        let cross_up = [5.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let up_input = DidiIndexInput::from_slice(
            &cross_up,
            DidiIndexParams {
                short_length: Some(2),
                medium_length: Some(3),
                long_length: Some(4),
            },
        );
        let up_out = didi_index(&up_input).unwrap();
        assert!(approx_eq(up_out.crossover[6], 1.0));
        assert!(approx_eq(up_out.crossunder[6], 0.0));

        let cross_down = [1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let down_input = DidiIndexInput::from_slice(
            &cross_down,
            DidiIndexParams {
                short_length: Some(2),
                medium_length: Some(3),
                long_length: Some(4),
            },
        );
        let down_out = didi_index(&down_input).unwrap();
        assert!(approx_eq(down_out.crossunder[6], 1.0));
        assert!(approx_eq(down_out.crossover[6], 0.0));
    }

    #[test]
    fn didi_index_stream_matches_batch_with_reset() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, f64::NAN, 3.0, 4.0, 5.0, 6.0];
        let params = DidiIndexParams {
            short_length: Some(2),
            medium_length: Some(3),
            long_length: Some(4),
        };
        let input = DidiIndexInput::from_slice(&data, params.clone());
        let batch = didi_index(&input).unwrap();
        let mut stream = DidiIndexStream::try_new(params).unwrap();

        let mut short = Vec::new();
        let mut long = Vec::new();
        let mut crossover = Vec::new();
        let mut crossunder = Vec::new();
        for &value in &data {
            match stream.update(value) {
                Some((s, l, co, cu)) => {
                    short.push(s);
                    long.push(l);
                    crossover.push(co);
                    crossunder.push(cu);
                }
                None => {
                    short.push(f64::NAN);
                    long.push(f64::NAN);
                    crossover.push(f64::NAN);
                    crossunder.push(f64::NAN);
                }
            }
        }

        assert_eq!(stream.get_warmup_period(), 3);
        for i in 0..data.len() {
            assert!(approx_eq_or_nan(batch.short[i], short[i]));
            assert!(approx_eq_or_nan(batch.long[i], long[i]));
            assert!(approx_eq_or_nan(batch.crossover[i], crossover[i]));
            assert!(approx_eq_or_nan(batch.crossunder[i], crossunder[i]));
        }
        assert!(batch.short[5].is_nan());
        assert!(batch.short[8].is_nan());
        assert!(batch.short[9].is_finite());
    }

    #[test]
    fn didi_index_batch_default_row_matches_single() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let batch = didi_index_batch_slice(
            &data,
            &DidiIndexBatchRange {
                short_length: (2, 2, 0),
                medium_length: (3, 3, 0),
                long_length: (4, 4, 0),
            },
        )
        .unwrap();
        let single = didi_index(&DidiIndexInput::from_slice(
            &data,
            DidiIndexParams {
                short_length: Some(2),
                medium_length: Some(3),
                long_length: Some(4),
            },
        ))
        .unwrap();

        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, data.len());
        assert_eq!(batch.short.len(), data.len());
        for i in 0..data.len() {
            assert!(approx_eq_or_nan(batch.short[i], single.short[i]));
            assert!(approx_eq_or_nan(batch.long[i], single.long[i]));
        }
    }

    #[test]
    fn didi_index_rejects_invalid_lengths() {
        let data = [1.0, 2.0, 3.0];
        let err = didi_index(&DidiIndexInput::from_slice(
            &data,
            DidiIndexParams {
                short_length: Some(0),
                medium_length: Some(2),
                long_length: Some(3),
            },
        ))
        .unwrap_err();
        assert!(matches!(err, DidiIndexError::InvalidShortLength { .. }));
    }
}
