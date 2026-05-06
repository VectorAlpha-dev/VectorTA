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
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::mem::MaybeUninit;
use thiserror::Error;

const DEFAULT_LENGTH: usize = 30;
const DEFAULT_ABSOLUTE_VOLUME_MILLIONS: f64 = 134.0;
const DEFAULT_USE_VOLUME_SUM: bool = false;
const MAX_LENGTH: usize = 4096;

impl<'a> AsRef<[f64]> for ElasticVolumeWeightedMovingAverageInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            ElasticVolumeWeightedMovingAverageData::Candles { candles, source } => {
                source_type(candles, source)
            }
            ElasticVolumeWeightedMovingAverageData::Slice { prices, .. } => prices,
        }
    }
}

#[derive(Debug, Clone)]
pub enum ElasticVolumeWeightedMovingAverageData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice {
        prices: &'a [f64],
        volumes: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct ElasticVolumeWeightedMovingAverageOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct ElasticVolumeWeightedMovingAverageParams {
    pub length: Option<usize>,
    pub absolute_volume_millions: Option<f64>,
    pub use_volume_sum: Option<bool>,
}

impl Default for ElasticVolumeWeightedMovingAverageParams {
    fn default() -> Self {
        Self {
            length: Some(DEFAULT_LENGTH),
            absolute_volume_millions: Some(DEFAULT_ABSOLUTE_VOLUME_MILLIONS),
            use_volume_sum: Some(DEFAULT_USE_VOLUME_SUM),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ElasticVolumeWeightedMovingAverageInput<'a> {
    pub data: ElasticVolumeWeightedMovingAverageData<'a>,
    pub params: ElasticVolumeWeightedMovingAverageParams,
}

impl<'a> ElasticVolumeWeightedMovingAverageInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: ElasticVolumeWeightedMovingAverageParams,
    ) -> Self {
        Self {
            data: ElasticVolumeWeightedMovingAverageData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(
        prices: &'a [f64],
        volumes: &'a [f64],
        params: ElasticVolumeWeightedMovingAverageParams,
    ) -> Self {
        Self {
            data: ElasticVolumeWeightedMovingAverageData::Slice { prices, volumes },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(
            candles,
            "hlcc4",
            ElasticVolumeWeightedMovingAverageParams::default(),
        )
    }

    #[inline]
    pub fn get_length(&self) -> usize {
        self.params.length.unwrap_or(DEFAULT_LENGTH)
    }

    #[inline]
    pub fn get_absolute_volume_millions(&self) -> f64 {
        self.params
            .absolute_volume_millions
            .unwrap_or(DEFAULT_ABSOLUTE_VOLUME_MILLIONS)
    }

    #[inline]
    pub fn get_use_volume_sum(&self) -> bool {
        self.params.use_volume_sum.unwrap_or(DEFAULT_USE_VOLUME_SUM)
    }

    #[inline]
    pub fn get_volume(&self) -> &[f64] {
        match &self.data {
            ElasticVolumeWeightedMovingAverageData::Candles { candles, .. } => &candles.volume,
            ElasticVolumeWeightedMovingAverageData::Slice { volumes, .. } => volumes,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct ElasticVolumeWeightedMovingAverageBuilder {
    length: Option<usize>,
    absolute_volume_millions: Option<f64>,
    use_volume_sum: Option<bool>,
    kernel: Kernel,
}

impl Default for ElasticVolumeWeightedMovingAverageBuilder {
    fn default() -> Self {
        Self {
            length: None,
            absolute_volume_millions: None,
            use_volume_sum: None,
            kernel: Kernel::Auto,
        }
    }
}

impl ElasticVolumeWeightedMovingAverageBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn length(mut self, length: usize) -> Self {
        self.length = Some(length);
        self
    }

    #[inline(always)]
    pub fn absolute_volume_millions(mut self, absolute_volume_millions: f64) -> Self {
        self.absolute_volume_millions = Some(absolute_volume_millions);
        self
    }

    #[inline(always)]
    pub fn use_volume_sum(mut self, use_volume_sum: bool) -> Self {
        self.use_volume_sum = Some(use_volume_sum);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline(always)]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<ElasticVolumeWeightedMovingAverageOutput, ElasticVolumeWeightedMovingAverageError>
    {
        let input = ElasticVolumeWeightedMovingAverageInput::from_candles(
            candles,
            "hlcc4",
            ElasticVolumeWeightedMovingAverageParams {
                length: self.length,
                absolute_volume_millions: self.absolute_volume_millions,
                use_volume_sum: self.use_volume_sum,
            },
        );
        elastic_volume_weighted_moving_average_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        prices: &[f64],
        volumes: &[f64],
    ) -> Result<ElasticVolumeWeightedMovingAverageOutput, ElasticVolumeWeightedMovingAverageError>
    {
        let input = ElasticVolumeWeightedMovingAverageInput::from_slice(
            prices,
            volumes,
            ElasticVolumeWeightedMovingAverageParams {
                length: self.length,
                absolute_volume_millions: self.absolute_volume_millions,
                use_volume_sum: self.use_volume_sum,
            },
        );
        elastic_volume_weighted_moving_average_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(
        self,
    ) -> Result<ElasticVolumeWeightedMovingAverageStream, ElasticVolumeWeightedMovingAverageError>
    {
        ElasticVolumeWeightedMovingAverageStream::try_new(
            ElasticVolumeWeightedMovingAverageParams {
                length: self.length,
                absolute_volume_millions: self.absolute_volume_millions,
                use_volume_sum: self.use_volume_sum,
            },
        )
    }
}

#[derive(Debug, Error)]
pub enum ElasticVolumeWeightedMovingAverageError {
    #[error("elastic_volume_weighted_moving_average: input data slice is empty.")]
    EmptyInputData,
    #[error("elastic_volume_weighted_moving_average: all values are NaN.")]
    AllValuesNaN,
    #[error("elastic_volume_weighted_moving_average: price and volume length mismatch: price = {price_len}, volume = {volume_len}")]
    DataLengthMismatch { price_len: usize, volume_len: usize },
    #[error("elastic_volume_weighted_moving_average: invalid length: {length}. Expected 1..={MAX_LENGTH}.")]
    InvalidLength { length: usize },
    #[error("elastic_volume_weighted_moving_average: invalid absolute_volume_millions: {absolute_volume_millions}")]
    InvalidAbsoluteVolumeMillions { absolute_volume_millions: f64 },
    #[error("elastic_volume_weighted_moving_average: output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("elastic_volume_weighted_moving_average: invalid length range: start={start}, end={end}, step={step}")]
    InvalidLengthRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("elastic_volume_weighted_moving_average: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Copy, Clone, Debug)]
struct PreparedEvwma<'a> {
    prices: &'a [f64],
    volumes: &'a [f64],
    first_valid: usize,
    length: usize,
    absolute_volume: f64,
    use_volume_sum: bool,
}

#[inline(always)]
fn validate_length(length: usize) -> Result<(), ElasticVolumeWeightedMovingAverageError> {
    if length == 0 || length > MAX_LENGTH {
        return Err(ElasticVolumeWeightedMovingAverageError::InvalidLength { length });
    }
    Ok(())
}

#[inline(always)]
fn validate_absolute_volume(
    absolute_volume_millions: f64,
) -> Result<(), ElasticVolumeWeightedMovingAverageError> {
    if !absolute_volume_millions.is_finite() || absolute_volume_millions <= 0.0 {
        return Err(
            ElasticVolumeWeightedMovingAverageError::InvalidAbsoluteVolumeMillions {
                absolute_volume_millions,
            },
        );
    }
    Ok(())
}

#[inline(always)]
fn find_first_valid(
    prices: &[f64],
    volumes: &[f64],
) -> Result<usize, ElasticVolumeWeightedMovingAverageError> {
    prices
        .iter()
        .zip(volumes.iter())
        .position(|(&price, &volume)| price.is_finite() && volume.is_finite())
        .ok_or(ElasticVolumeWeightedMovingAverageError::AllValuesNaN)
}

#[inline(always)]
fn prepare<'a>(
    input: &'a ElasticVolumeWeightedMovingAverageInput,
    _kernel: Kernel,
) -> Result<PreparedEvwma<'a>, ElasticVolumeWeightedMovingAverageError> {
    let prices = input.as_ref();
    let volumes = input.get_volume();
    if prices.is_empty() {
        return Err(ElasticVolumeWeightedMovingAverageError::EmptyInputData);
    }
    if prices.len() != volumes.len() {
        return Err(
            ElasticVolumeWeightedMovingAverageError::DataLengthMismatch {
                price_len: prices.len(),
                volume_len: volumes.len(),
            },
        );
    }

    let length = input.get_length();
    validate_length(length)?;
    let use_volume_sum = input.get_use_volume_sum();
    let absolute_volume_millions = input.get_absolute_volume_millions();
    if !use_volume_sum {
        validate_absolute_volume(absolute_volume_millions)?;
    }

    Ok(PreparedEvwma {
        prices,
        volumes,
        first_valid: find_first_valid(prices, volumes)?,
        length,
        absolute_volume: absolute_volume_millions * 1_000_000.0,
        use_volume_sum,
    })
}

#[inline(always)]
fn compute_absolute_into(
    prices: &[f64],
    volumes: &[f64],
    first_valid: usize,
    absolute_volume: f64,
    out: &mut [f64],
) {
    let mut prev = f64::NAN;
    let inv_absolute_volume = 1.0 / absolute_volume;
    for index in first_valid..prices.len() {
        let price = prices[index];
        let volume = volumes[index];
        if !price.is_finite() || !volume.is_finite() {
            out[index] = f64::NAN;
            prev = f64::NAN;
            continue;
        }
        let base = if prev.is_finite() { prev } else { price };
        let value = base + volume * inv_absolute_volume * (price - base);
        out[index] = value;
        prev = value;
    }
}

#[inline(always)]
fn compute_volume_sum_into(
    prices: &[f64],
    volumes: &[f64],
    first_valid: usize,
    length: usize,
    out: &mut [f64],
) {
    let mut ring = vec![0.0; length];
    let mut rolling_sum = 0.0;
    let mut count = 0usize;
    let mut head = 0usize;
    let mut prev = f64::NAN;

    for index in first_valid..prices.len() {
        let price = prices[index];
        let volume = volumes[index];
        let volume_value = if volume.is_finite() { volume } else { 0.0 };

        if count < length {
            ring[count] = volume_value;
            rolling_sum += volume_value;
            count += 1;
        } else {
            rolling_sum += volume_value - ring[head];
            ring[head] = volume_value;
            head += 1;
            if head == length {
                head = 0;
            }
        }

        if !price.is_finite()
            || !volume.is_finite()
            || !rolling_sum.is_finite()
            || rolling_sum == 0.0
        {
            out[index] = f64::NAN;
            prev = f64::NAN;
            continue;
        }

        let base = if prev.is_finite() { prev } else { price };
        let value = ((rolling_sum - volume) * base + volume * price) / rolling_sum;
        out[index] = value;
        prev = value;
    }
}

#[inline(always)]
fn compute_into(prepared: PreparedEvwma<'_>, out: &mut [f64]) {
    if prepared.use_volume_sum {
        compute_volume_sum_into(
            prepared.prices,
            prepared.volumes,
            prepared.first_valid,
            prepared.length,
            out,
        );
    } else {
        compute_absolute_into(
            prepared.prices,
            prepared.volumes,
            prepared.first_valid,
            prepared.absolute_volume,
            out,
        );
    }
}

#[inline]
pub fn elastic_volume_weighted_moving_average(
    input: &ElasticVolumeWeightedMovingAverageInput,
) -> Result<ElasticVolumeWeightedMovingAverageOutput, ElasticVolumeWeightedMovingAverageError> {
    elastic_volume_weighted_moving_average_with_kernel(input, Kernel::Auto)
}

#[inline]
pub fn elastic_volume_weighted_moving_average_with_kernel(
    input: &ElasticVolumeWeightedMovingAverageInput,
    kernel: Kernel,
) -> Result<ElasticVolumeWeightedMovingAverageOutput, ElasticVolumeWeightedMovingAverageError> {
    let prepared = prepare(input, kernel)?;
    let mut out = alloc_with_nan_prefix(prepared.prices.len(), prepared.first_valid);
    compute_into(prepared, &mut out);
    Ok(ElasticVolumeWeightedMovingAverageOutput { values: out })
}

#[inline]
pub fn elastic_volume_weighted_moving_average_into_slice(
    out: &mut [f64],
    input: &ElasticVolumeWeightedMovingAverageInput,
    kernel: Kernel,
) -> Result<(), ElasticVolumeWeightedMovingAverageError> {
    let prepared = prepare(input, kernel)?;
    if out.len() != prepared.prices.len() {
        return Err(
            ElasticVolumeWeightedMovingAverageError::OutputLengthMismatch {
                expected: prepared.prices.len(),
                got: out.len(),
            },
        );
    }
    out[..prepared.first_valid].fill(f64::NAN);
    compute_into(prepared, out);
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn elastic_volume_weighted_moving_average_into(
    input: &ElasticVolumeWeightedMovingAverageInput,
    out: &mut [f64],
) -> Result<(), ElasticVolumeWeightedMovingAverageError> {
    elastic_volume_weighted_moving_average_into_slice(out, input, Kernel::Auto)
}

#[derive(Clone, Debug)]
pub struct ElasticVolumeWeightedMovingAverageStream {
    length: usize,
    absolute_volume: f64,
    use_volume_sum: bool,
    volume_ring: Vec<f64>,
    rolling_sum: f64,
    count: usize,
    head: usize,
    prev: f64,
}

impl ElasticVolumeWeightedMovingAverageStream {
    pub fn try_new(
        params: ElasticVolumeWeightedMovingAverageParams,
    ) -> Result<Self, ElasticVolumeWeightedMovingAverageError> {
        let length = params.length.unwrap_or(DEFAULT_LENGTH);
        validate_length(length)?;
        let use_volume_sum = params.use_volume_sum.unwrap_or(DEFAULT_USE_VOLUME_SUM);
        let absolute_volume_millions = params
            .absolute_volume_millions
            .unwrap_or(DEFAULT_ABSOLUTE_VOLUME_MILLIONS);
        if !use_volume_sum {
            validate_absolute_volume(absolute_volume_millions)?;
        }
        Ok(Self {
            length,
            absolute_volume: absolute_volume_millions * 1_000_000.0,
            use_volume_sum,
            volume_ring: vec![0.0; length],
            rolling_sum: 0.0,
            count: 0,
            head: 0,
            prev: f64::NAN,
        })
    }

    #[inline]
    pub fn reset(&mut self) {
        self.volume_ring.fill(0.0);
        self.rolling_sum = 0.0;
        self.count = 0;
        self.head = 0;
        self.prev = f64::NAN;
    }

    #[inline]
    pub fn update(&mut self, price: f64, volume: f64) -> Option<f64> {
        let volume_period = if self.use_volume_sum {
            let volume_value = if volume.is_finite() { volume } else { 0.0 };
            if self.count < self.length {
                self.volume_ring[self.count] = volume_value;
                self.rolling_sum += volume_value;
                self.count += 1;
            } else {
                self.rolling_sum += volume_value - self.volume_ring[self.head];
                self.volume_ring[self.head] = volume_value;
                self.head += 1;
                if self.head == self.length {
                    self.head = 0;
                }
            }
            self.rolling_sum
        } else {
            self.absolute_volume
        };

        if !price.is_finite()
            || !volume.is_finite()
            || !volume_period.is_finite()
            || volume_period == 0.0
        {
            self.prev = f64::NAN;
            return None;
        }

        let base = if self.prev.is_finite() {
            self.prev
        } else {
            price
        };
        let value = ((volume_period - volume) * base + volume * price) / volume_period;
        self.prev = value;
        value.is_finite().then_some(value)
    }
}

#[derive(Clone, Debug)]
pub struct ElasticVolumeWeightedMovingAverageBatchRange {
    pub length: (usize, usize, usize),
    pub absolute_volume_millions: Option<f64>,
    pub use_volume_sum: Option<bool>,
}

impl Default for ElasticVolumeWeightedMovingAverageBatchRange {
    fn default() -> Self {
        Self {
            length: (DEFAULT_LENGTH, DEFAULT_LENGTH, 0),
            absolute_volume_millions: Some(DEFAULT_ABSOLUTE_VOLUME_MILLIONS),
            use_volume_sum: Some(DEFAULT_USE_VOLUME_SUM),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ElasticVolumeWeightedMovingAverageBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<ElasticVolumeWeightedMovingAverageParams>,
    pub rows: usize,
    pub cols: usize,
}

impl ElasticVolumeWeightedMovingAverageBatchOutput {
    pub fn values_for(&self, params: &ElasticVolumeWeightedMovingAverageParams) -> Option<&[f64]> {
        self.combos
            .iter()
            .position(|candidate| {
                candidate.length.unwrap_or(DEFAULT_LENGTH)
                    == params.length.unwrap_or(DEFAULT_LENGTH)
                    && candidate.use_volume_sum.unwrap_or(DEFAULT_USE_VOLUME_SUM)
                        == params.use_volume_sum.unwrap_or(DEFAULT_USE_VOLUME_SUM)
                    && (candidate
                        .absolute_volume_millions
                        .unwrap_or(DEFAULT_ABSOLUTE_VOLUME_MILLIONS)
                        - params
                            .absolute_volume_millions
                            .unwrap_or(DEFAULT_ABSOLUTE_VOLUME_MILLIONS))
                    .abs()
                        <= 1e-12
            })
            .map(|row| {
                let start = row * self.cols;
                &self.values[start..start + self.cols]
            })
    }
}

#[inline]
pub fn expand_grid_elastic_volume_weighted_moving_average(
    range: &ElasticVolumeWeightedMovingAverageBatchRange,
) -> Result<Vec<ElasticVolumeWeightedMovingAverageParams>, ElasticVolumeWeightedMovingAverageError>
{
    let (start, end, step) = range.length;
    if start == 0 || end == 0 {
        return Err(
            ElasticVolumeWeightedMovingAverageError::InvalidLengthRange { start, end, step },
        );
    }
    let absolute_volume_millions = range
        .absolute_volume_millions
        .unwrap_or(DEFAULT_ABSOLUTE_VOLUME_MILLIONS);
    let use_volume_sum = range.use_volume_sum.unwrap_or(DEFAULT_USE_VOLUME_SUM);
    if !use_volume_sum {
        validate_absolute_volume(absolute_volume_millions)?;
    }

    let mut combos = Vec::new();
    if step == 0 || start == end {
        combos.push(ElasticVolumeWeightedMovingAverageParams {
            length: Some(start),
            absolute_volume_millions: Some(absolute_volume_millions),
            use_volume_sum: Some(use_volume_sum),
        });
        return Ok(combos);
    }

    if start < end {
        for length in (start..=end).step_by(step) {
            combos.push(ElasticVolumeWeightedMovingAverageParams {
                length: Some(length),
                absolute_volume_millions: Some(absolute_volume_millions),
                use_volume_sum: Some(use_volume_sum),
            });
        }
    } else {
        let mut current = start;
        loop {
            combos.push(ElasticVolumeWeightedMovingAverageParams {
                length: Some(current),
                absolute_volume_millions: Some(absolute_volume_millions),
                use_volume_sum: Some(use_volume_sum),
            });
            if current <= end || current - end < step {
                break;
            }
            current -= step;
        }
    }

    if combos.is_empty() {
        return Err(
            ElasticVolumeWeightedMovingAverageError::InvalidLengthRange { start, end, step },
        );
    }

    Ok(combos)
}

#[inline(always)]
fn normalize_batch_kernel(
    kernel: Kernel,
) -> Result<Kernel, ElasticVolumeWeightedMovingAverageError> {
    Ok(match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => {
            return Err(ElasticVolumeWeightedMovingAverageError::InvalidKernelForBatch(other));
        }
    })
}

#[inline(always)]
fn batch_simd_kernel(kernel: Kernel) -> Kernel {
    match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => Kernel::Scalar,
    }
}

#[inline]
pub fn elastic_volume_weighted_moving_average_batch_inner_into(
    prices: &[f64],
    volumes: &[f64],
    range: &ElasticVolumeWeightedMovingAverageBatchRange,
    _kernel: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<ElasticVolumeWeightedMovingAverageParams>, ElasticVolumeWeightedMovingAverageError>
{
    if prices.is_empty() {
        return Err(ElasticVolumeWeightedMovingAverageError::EmptyInputData);
    }
    if prices.len() != volumes.len() {
        return Err(
            ElasticVolumeWeightedMovingAverageError::DataLengthMismatch {
                price_len: prices.len(),
                volume_len: volumes.len(),
            },
        );
    }

    let combos = expand_grid_elastic_volume_weighted_moving_average(range)?;
    let rows = combos.len();
    let cols = prices.len();
    let expected = rows.checked_mul(cols).ok_or(
        ElasticVolumeWeightedMovingAverageError::InvalidLengthRange {
            start: range.length.0,
            end: range.length.1,
            step: range.length.2,
        },
    )?;
    if out.len() != expected {
        return Err(
            ElasticVolumeWeightedMovingAverageError::OutputLengthMismatch {
                expected,
                got: out.len(),
            },
        );
    }

    let first_valid = find_first_valid(prices, volumes)?;
    let out_mu: &mut [MaybeUninit<f64>] = unsafe {
        core::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };
    let warm_prefixes = vec![first_valid; rows];
    init_matrix_prefixes(out_mu, cols, &warm_prefixes);

    let do_row = |row: usize, row_mu: &mut [MaybeUninit<f64>]| unsafe {
        let row_out =
            core::slice::from_raw_parts_mut(row_mu.as_mut_ptr() as *mut f64, row_mu.len());
        let params = &combos[row];
        compute_into(
            PreparedEvwma {
                prices,
                volumes,
                first_valid,
                length: params.length.unwrap_or(DEFAULT_LENGTH),
                absolute_volume: params
                    .absolute_volume_millions
                    .unwrap_or(DEFAULT_ABSOLUTE_VOLUME_MILLIONS)
                    * 1_000_000.0,
                use_volume_sum: params.use_volume_sum.unwrap_or(DEFAULT_USE_VOLUME_SUM),
            },
            row_out,
        );
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out_mu
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, chunk)| do_row(row, chunk));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, chunk) in out_mu.chunks_mut(cols).enumerate() {
                do_row(row, chunk);
            }
        }
    } else {
        for (row, chunk) in out_mu.chunks_mut(cols).enumerate() {
            do_row(row, chunk);
        }
    }

    Ok(combos)
}

#[inline]
fn elastic_volume_weighted_moving_average_batch_inner(
    prices: &[f64],
    volumes: &[f64],
    range: &ElasticVolumeWeightedMovingAverageBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<ElasticVolumeWeightedMovingAverageBatchOutput, ElasticVolumeWeightedMovingAverageError>
{
    let combos = expand_grid_elastic_volume_weighted_moving_average(range)?;
    let rows = combos.len();
    let cols = prices.len();
    let mut raw = make_uninit_matrix(rows, cols);
    let first_valid = find_first_valid(prices, volumes)?;
    let warm_prefixes = vec![first_valid; rows];
    unsafe { init_matrix_prefixes(&mut raw, cols, &warm_prefixes) };

    let do_row = |row: usize, row_mu: &mut [MaybeUninit<f64>]| unsafe {
        let row_out =
            core::slice::from_raw_parts_mut(row_mu.as_mut_ptr() as *mut f64, row_mu.len());
        let params = &combos[row];
        compute_into(
            PreparedEvwma {
                prices,
                volumes,
                first_valid,
                length: params.length.unwrap_or(DEFAULT_LENGTH),
                absolute_volume: params
                    .absolute_volume_millions
                    .unwrap_or(DEFAULT_ABSOLUTE_VOLUME_MILLIONS)
                    * 1_000_000.0,
                use_volume_sum: params.use_volume_sum.unwrap_or(DEFAULT_USE_VOLUME_SUM),
            },
            row_out,
        );
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            raw.par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, chunk)| do_row(row, chunk));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, chunk) in raw.chunks_mut(cols).enumerate() {
                do_row(row, chunk);
            }
        }
    } else {
        for (row, chunk) in raw.chunks_mut(cols).enumerate() {
            do_row(row, chunk);
        }
    }

    let _ = kernel;
    let values: Vec<f64> = unsafe { std::mem::transmute(raw) };
    Ok(ElasticVolumeWeightedMovingAverageBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline]
pub fn elastic_volume_weighted_moving_average_batch_slice(
    prices: &[f64],
    volumes: &[f64],
    range: &ElasticVolumeWeightedMovingAverageBatchRange,
    kernel: Kernel,
) -> Result<ElasticVolumeWeightedMovingAverageBatchOutput, ElasticVolumeWeightedMovingAverageError>
{
    elastic_volume_weighted_moving_average_batch_inner(prices, volumes, range, kernel, false)
}

#[inline]
pub fn elastic_volume_weighted_moving_average_batch_par_slice(
    prices: &[f64],
    volumes: &[f64],
    range: &ElasticVolumeWeightedMovingAverageBatchRange,
    kernel: Kernel,
) -> Result<ElasticVolumeWeightedMovingAverageBatchOutput, ElasticVolumeWeightedMovingAverageError>
{
    elastic_volume_weighted_moving_average_batch_inner(prices, volumes, range, kernel, true)
}

#[inline]
pub fn elastic_volume_weighted_moving_average_batch_with_kernel(
    prices: &[f64],
    volumes: &[f64],
    range: &ElasticVolumeWeightedMovingAverageBatchRange,
    kernel: Kernel,
) -> Result<ElasticVolumeWeightedMovingAverageBatchOutput, ElasticVolumeWeightedMovingAverageError>
{
    let batch_kernel = normalize_batch_kernel(kernel)?;
    elastic_volume_weighted_moving_average_batch_par_slice(
        prices,
        volumes,
        range,
        batch_simd_kernel(batch_kernel),
    )
}

#[derive(Clone, Debug)]
pub struct ElasticVolumeWeightedMovingAverageBatchBuilder {
    range: ElasticVolumeWeightedMovingAverageBatchRange,
    kernel: Kernel,
}

impl Default for ElasticVolumeWeightedMovingAverageBatchBuilder {
    fn default() -> Self {
        Self {
            range: ElasticVolumeWeightedMovingAverageBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl ElasticVolumeWeightedMovingAverageBatchBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline(always)]
    pub fn length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.length = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn length_static(mut self, length: usize) -> Self {
        self.range.length = (length, length, 0);
        self
    }

    #[inline(always)]
    pub fn absolute_volume_millions(mut self, absolute_volume_millions: f64) -> Self {
        self.range.absolute_volume_millions = Some(absolute_volume_millions);
        self
    }

    #[inline(always)]
    pub fn use_volume_sum(mut self, use_volume_sum: bool) -> Self {
        self.range.use_volume_sum = Some(use_volume_sum);
        self
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        prices: &[f64],
        volumes: &[f64],
    ) -> Result<
        ElasticVolumeWeightedMovingAverageBatchOutput,
        ElasticVolumeWeightedMovingAverageError,
    > {
        elastic_volume_weighted_moving_average_batch_with_kernel(
            prices,
            volumes,
            &self.range,
            self.kernel,
        )
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "elastic_volume_weighted_moving_average")]
#[pyo3(signature = (prices, volumes, length=None, absolute_volume_millions=None, use_volume_sum=None, kernel=None))]
pub fn elastic_volume_weighted_moving_average_py<'py>(
    py: Python<'py>,
    prices: PyReadonlyArray1<'py, f64>,
    volumes: PyReadonlyArray1<'py, f64>,
    length: Option<usize>,
    absolute_volume_millions: Option<f64>,
    use_volume_sum: Option<bool>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let prices = prices.as_slice()?;
    let volumes = volumes.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let input = ElasticVolumeWeightedMovingAverageInput::from_slice(
        prices,
        volumes,
        ElasticVolumeWeightedMovingAverageParams {
            length,
            absolute_volume_millions,
            use_volume_sum,
        },
    );
    let out = py
        .allow_threads(|| elastic_volume_weighted_moving_average_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(out.values.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "ElasticVolumeWeightedMovingAverageStream")]
pub struct ElasticVolumeWeightedMovingAverageStreamPy {
    stream: ElasticVolumeWeightedMovingAverageStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl ElasticVolumeWeightedMovingAverageStreamPy {
    #[new]
    #[pyo3(signature = (length=None, absolute_volume_millions=None, use_volume_sum=None))]
    fn new(
        length: Option<usize>,
        absolute_volume_millions: Option<f64>,
        use_volume_sum: Option<bool>,
    ) -> PyResult<Self> {
        let stream = ElasticVolumeWeightedMovingAverageStream::try_new(
            ElasticVolumeWeightedMovingAverageParams {
                length,
                absolute_volume_millions,
                use_volume_sum,
            },
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, price: f64, volume: f64) -> Option<f64> {
        self.stream.update(price, volume)
    }

    fn reset(&mut self) {
        self.stream.reset();
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "elastic_volume_weighted_moving_average_batch")]
#[pyo3(signature = (prices, volumes, length_range=(DEFAULT_LENGTH, DEFAULT_LENGTH, 0), absolute_volume_millions=None, use_volume_sum=None, kernel=None))]
pub fn elastic_volume_weighted_moving_average_batch_py<'py>(
    py: Python<'py>,
    prices: PyReadonlyArray1<'py, f64>,
    volumes: PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    absolute_volume_millions: Option<f64>,
    use_volume_sum: Option<bool>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let prices = prices.as_slice()?;
    let volumes = volumes.as_slice()?;
    let range = ElasticVolumeWeightedMovingAverageBatchRange {
        length: length_range,
        absolute_volume_millions,
        use_volume_sum,
    };
    let combos = expand_grid_elastic_volume_weighted_moving_average(&range)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = prices.len();
    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let out_slice = unsafe { out_arr.as_slice_mut()? };
    let kernel = validate_kernel(kernel, true)?;
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other => other,
    };
    let combos = py
        .allow_threads(|| {
            elastic_volume_weighted_moving_average_batch_inner_into(
                prices,
                volumes,
                &range,
                batch_simd_kernel(batch_kernel),
                true,
                out_slice,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "lengths",
        combos
            .iter()
            .map(|combo| combo.length.unwrap_or(DEFAULT_LENGTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "absolute_volume_millions",
        combos
            .iter()
            .map(|combo| {
                combo
                    .absolute_volume_millions
                    .unwrap_or(DEFAULT_ABSOLUTE_VOLUME_MILLIONS)
            })
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "use_volume_sum",
        combos
            .iter()
            .map(|combo| combo.use_volume_sum.unwrap_or(DEFAULT_USE_VOLUME_SUM))
            .collect::<Vec<_>>(),
    )?;
    Ok(dict.into())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct ElasticVolumeWeightedMovingAverageBatchConfig {
    pub length_range: (usize, usize, usize),
    pub absolute_volume_millions: Option<f64>,
    pub use_volume_sum: Option<bool>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct ElasticVolumeWeightedMovingAverageBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<ElasticVolumeWeightedMovingAverageParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = elastic_volume_weighted_moving_average)]
pub fn elastic_volume_weighted_moving_average_js(
    prices: &[f64],
    volumes: &[f64],
    length: Option<usize>,
    absolute_volume_millions: Option<f64>,
    use_volume_sum: Option<bool>,
) -> Result<Vec<f64>, JsValue> {
    let input = ElasticVolumeWeightedMovingAverageInput::from_slice(
        prices,
        volumes,
        ElasticVolumeWeightedMovingAverageParams {
            length,
            absolute_volume_millions,
            use_volume_sum,
        },
    );
    let mut out = vec![0.0; prices.len()];
    elastic_volume_weighted_moving_average_into_slice(&mut out, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = elastic_volume_weighted_moving_average_batch)]
pub fn elastic_volume_weighted_moving_average_batch_js(
    prices: &[f64],
    volumes: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: ElasticVolumeWeightedMovingAverageBatchConfig =
        serde_wasm_bindgen::from_value(config)
            .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    let range = ElasticVolumeWeightedMovingAverageBatchRange {
        length: config.length_range,
        absolute_volume_millions: config.absolute_volume_millions,
        use_volume_sum: config.use_volume_sum,
    };
    let out = elastic_volume_weighted_moving_average_batch_with_kernel(
        prices,
        volumes,
        &range,
        detect_best_batch_kernel(),
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&ElasticVolumeWeightedMovingAverageBatchJsOutput {
        values: out.values,
        combos: out.combos,
        rows: out.rows,
        cols: out.cols,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn elastic_volume_weighted_moving_average_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn elastic_volume_weighted_moving_average_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = elastic_volume_weighted_moving_average_into)]
pub fn elastic_volume_weighted_moving_average_into_js(
    price_ptr: *const f64,
    volume_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length: Option<usize>,
    absolute_volume_millions: Option<f64>,
    use_volume_sum: Option<bool>,
) -> Result<(), JsValue> {
    if price_ptr.is_null() || volume_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to elastic_volume_weighted_moving_average_into",
        ));
    }
    unsafe {
        let prices = std::slice::from_raw_parts(price_ptr, len);
        let volumes = std::slice::from_raw_parts(volume_ptr, len);
        let input = ElasticVolumeWeightedMovingAverageInput::from_slice(
            prices,
            volumes,
            ElasticVolumeWeightedMovingAverageParams {
                length,
                absolute_volume_millions,
                use_volume_sum,
            },
        );
        if out_ptr == price_ptr as *mut f64 || out_ptr == volume_ptr as *mut f64 {
            let mut temp = vec![0.0; len];
            elastic_volume_weighted_moving_average_into_slice(&mut temp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            elastic_volume_weighted_moving_average_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = elastic_volume_weighted_moving_average_batch_into)]
pub fn elastic_volume_weighted_moving_average_batch_into_js(
    price_ptr: *const f64,
    volume_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    config: JsValue,
) -> Result<usize, JsValue> {
    if price_ptr.is_null() || volume_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to elastic_volume_weighted_moving_average_batch_into",
        ));
    }
    let config: ElasticVolumeWeightedMovingAverageBatchConfig =
        serde_wasm_bindgen::from_value(config)
            .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    let range = ElasticVolumeWeightedMovingAverageBatchRange {
        length: config.length_range,
        absolute_volume_millions: config.absolute_volume_millions,
        use_volume_sum: config.use_volume_sum,
    };
    let combos = expand_grid_elastic_volume_weighted_moving_average(&range)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();
    unsafe {
        let prices = std::slice::from_raw_parts(price_ptr, len);
        let volumes = std::slice::from_raw_parts(volume_ptr, len);
        let simd = batch_simd_kernel(detect_best_batch_kernel());
        if out_ptr == price_ptr as *mut f64 || out_ptr == volume_ptr as *mut f64 {
            let mut temp = vec![0.0; rows * len];
            elastic_volume_weighted_moving_average_batch_inner_into(
                prices, volumes, &range, simd, false, &mut temp,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, rows * len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, rows * len);
            elastic_volume_weighted_moving_average_batch_inner_into(
                prices, volumes, &range, simd, false, out,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }
    Ok(rows)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub struct ElasticVolumeWeightedMovingAverageStreamWasm {
    inner: ElasticVolumeWeightedMovingAverageStream,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
impl ElasticVolumeWeightedMovingAverageStreamWasm {
    #[wasm_bindgen(constructor)]
    pub fn new(
        length: Option<usize>,
        absolute_volume_millions: Option<f64>,
        use_volume_sum: Option<bool>,
    ) -> Result<ElasticVolumeWeightedMovingAverageStreamWasm, JsValue> {
        let inner = ElasticVolumeWeightedMovingAverageStream::try_new(
            ElasticVolumeWeightedMovingAverageParams {
                length,
                absolute_volume_millions,
                use_volume_sum,
            },
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Self { inner })
    }

    pub fn update(&mut self, price: f64, volume: f64) -> Option<f64> {
        self.inner.update(price, volume)
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn elastic_volume_weighted_moving_average_output_into_js(
    prices: &[f64],
    volumes: &[f64],
    length: Option<usize>,
    absolute_volume_millions: Option<f64>,
    use_volume_sum: Option<bool>,
    out: &js_sys::Float64Array,
) -> Result<usize, JsValue> {
    let values = elastic_volume_weighted_moving_average_js(
        prices,
        volumes,
        length,
        absolute_volume_millions,
        use_volume_sum,
    )?;
    crate::write_wasm_f64_output(
        "elastic_volume_weighted_moving_average_output_into_js",
        &values,
        out,
    )
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn elastic_volume_weighted_moving_average_batch_output_into_js(
    prices: &[f64],
    volumes: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = elastic_volume_weighted_moving_average_batch_js(prices, volumes, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "elastic_volume_weighted_moving_average_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::moving_averages::ma::MaData;
    use crate::indicators::moving_averages::ma_batch::{
        ma_batch_with_kernel_and_typed_params, MaBatchParamKV, MaBatchParamValue,
    };

    fn sample_prices(len: usize) -> Vec<f64> {
        (0..len)
            .map(|i| 100.0 + (i as f64) * 0.23 + ((i as f64) * 0.17).sin())
            .collect()
    }

    fn sample_volumes(len: usize) -> Vec<f64> {
        (0..len)
            .map(|i| 1_000.0 + ((i % 9) as f64) * 57.0 + (i as f64) * 2.0)
            .collect()
    }

    fn sample_candles(len: usize) -> Candles {
        let timestamp: Vec<i64> = (0..len)
            .map(|i| 1_700_000_000_000_i64 + (i as i64) * 60_000)
            .collect();
        let close = sample_prices(len);
        let open: Vec<f64> = close.iter().map(|v| v - 0.2).collect();
        let high: Vec<f64> = close.iter().map(|v| v + 0.4).collect();
        let low: Vec<f64> = close.iter().map(|v| v - 0.6).collect();
        let volume = sample_volumes(len);
        Candles::new(timestamp, open, high, low, close, volume)
    }

    fn naive_evwma(
        prices: &[f64],
        volumes: &[f64],
        length: usize,
        absolute_volume_millions: f64,
        use_volume_sum: bool,
    ) -> Vec<f64> {
        let mut out = vec![f64::NAN; prices.len()];
        let first = prices
            .iter()
            .zip(volumes.iter())
            .position(|(&price, &volume)| price.is_finite() && volume.is_finite())
            .unwrap();
        let absolute_volume = absolute_volume_millions * 1_000_000.0;
        let mut prev = f64::NAN;
        let mut ring = vec![0.0; length];
        let mut rolling_sum = 0.0;
        let mut count = 0usize;
        let mut head = 0usize;

        for index in first..prices.len() {
            let price = prices[index];
            let volume = volumes[index];
            if use_volume_sum {
                let volume_value = if volume.is_finite() { volume } else { 0.0 };
                if count < length {
                    ring[count] = volume_value;
                    rolling_sum += volume_value;
                    count += 1;
                } else {
                    rolling_sum += volume_value - ring[head];
                    ring[head] = volume_value;
                    head += 1;
                    if head == length {
                        head = 0;
                    }
                }
            }
            let volume_period = if use_volume_sum {
                rolling_sum
            } else {
                absolute_volume
            };
            if !price.is_finite()
                || !volume.is_finite()
                || !volume_period.is_finite()
                || volume_period == 0.0
            {
                out[index] = f64::NAN;
                prev = f64::NAN;
                continue;
            }
            let base = if prev.is_finite() { prev } else { price };
            let value = ((volume_period - volume) * base + volume * price) / volume_period;
            out[index] = value;
            prev = value;
        }

        out
    }

    fn assert_series_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual_value, &expected_value)) in
            actual.iter().zip(expected.iter()).enumerate()
        {
            if actual_value.is_nan() && expected_value.is_nan() {
                continue;
            }
            let diff = (actual_value - expected_value).abs();
            assert!(
                diff <= 1e-10,
                "series mismatch at index {index}: actual={actual_value}, expected={expected_value}, diff={diff}"
            );
        }
    }

    #[test]
    fn evwma_matches_naive_absolute_mode() {
        let prices = sample_prices(96);
        let volumes = sample_volumes(96);
        let input = ElasticVolumeWeightedMovingAverageInput::from_slice(
            &prices,
            &volumes,
            ElasticVolumeWeightedMovingAverageParams::default(),
        );
        let output = elastic_volume_weighted_moving_average(&input).unwrap();
        let expected = naive_evwma(
            &prices,
            &volumes,
            DEFAULT_LENGTH,
            DEFAULT_ABSOLUTE_VOLUME_MILLIONS,
            false,
        );
        assert_series_close(&output.values, &expected);
    }

    #[test]
    fn evwma_matches_naive_volume_sum_mode() {
        let prices = sample_prices(96);
        let volumes = sample_volumes(96);
        let input = ElasticVolumeWeightedMovingAverageInput::from_slice(
            &prices,
            &volumes,
            ElasticVolumeWeightedMovingAverageParams {
                length: Some(7),
                absolute_volume_millions: Some(200.0),
                use_volume_sum: Some(true),
            },
        );
        let output = elastic_volume_weighted_moving_average(&input).unwrap();
        let expected = naive_evwma(&prices, &volumes, 7, 200.0, true);
        assert_series_close(&output.values, &expected);
    }

    #[test]
    fn evwma_into_matches_api() {
        let prices = sample_prices(72);
        let volumes = sample_volumes(72);
        let input = ElasticVolumeWeightedMovingAverageInput::from_slice(
            &prices,
            &volumes,
            ElasticVolumeWeightedMovingAverageParams {
                length: Some(8),
                absolute_volume_millions: Some(175.0),
                use_volume_sum: Some(true),
            },
        );
        let output = elastic_volume_weighted_moving_average(&input).unwrap();
        let mut into = vec![0.0; prices.len()];
        elastic_volume_weighted_moving_average_into_slice(&mut into, &input, Kernel::Auto).unwrap();
        assert_series_close(&into, &output.values);
    }

    #[test]
    fn evwma_stream_matches_batch_and_reset() {
        let prices = sample_prices(80);
        let volumes = sample_volumes(80);
        let params = ElasticVolumeWeightedMovingAverageParams {
            length: Some(9),
            absolute_volume_millions: Some(220.0),
            use_volume_sum: Some(true),
        };
        let input =
            ElasticVolumeWeightedMovingAverageInput::from_slice(&prices, &volumes, params.clone());
        let batch = elastic_volume_weighted_moving_average(&input).unwrap();
        let mut stream = ElasticVolumeWeightedMovingAverageStream::try_new(params).unwrap();

        let mut streamed = Vec::with_capacity(prices.len());
        for (&price, &volume) in prices.iter().zip(volumes.iter()) {
            streamed.push(stream.update(price, volume).unwrap_or(f64::NAN));
        }
        assert_series_close(&streamed, &batch.values);

        stream.reset();
        let mut streamed_again = Vec::with_capacity(prices.len());
        for (&price, &volume) in prices.iter().zip(volumes.iter()) {
            streamed_again.push(stream.update(price, volume).unwrap_or(f64::NAN));
        }
        assert_series_close(&streamed_again, &batch.values);
    }

    #[test]
    fn evwma_batch_matches_single() {
        let prices = sample_prices(84);
        let volumes = sample_volumes(84);
        let range = ElasticVolumeWeightedMovingAverageBatchRange {
            length: (5, 9, 2),
            absolute_volume_millions: Some(180.0),
            use_volume_sum: Some(true),
        };
        let batch = elastic_volume_weighted_moving_average_batch_with_kernel(
            &prices,
            &volumes,
            &range,
            Kernel::Auto,
        )
        .unwrap();
        assert_eq!(batch.rows, 3);
        assert_eq!(batch.cols, prices.len());

        for params in &batch.combos {
            let input = ElasticVolumeWeightedMovingAverageInput::from_slice(
                &prices,
                &volumes,
                params.clone(),
            );
            let single = elastic_volume_weighted_moving_average(&input).unwrap();
            assert_series_close(batch.values_for(params).unwrap(), &single.values);
        }
    }

    #[test]
    fn evwma_ma_batch_typed_params_match_direct() {
        let candles = sample_candles(96);
        let params = [
            MaBatchParamKV {
                key: "absolute_volume_millions",
                value: MaBatchParamValue::Float(180.0),
            },
            MaBatchParamKV {
                key: "use_volume_sum",
                value: MaBatchParamValue::Bool(true),
            },
        ];
        let batch = ma_batch_with_kernel_and_typed_params(
            "elastic_volume_weighted_moving_average",
            MaData::Candles {
                candles: &candles,
                source: "hlcc4",
            },
            (6, 10, 2),
            Kernel::ScalarBatch,
            &params,
        )
        .unwrap();

        for (row, length) in batch.periods.iter().enumerate() {
            let input = ElasticVolumeWeightedMovingAverageInput::from_candles(
                &candles,
                "hlcc4",
                ElasticVolumeWeightedMovingAverageParams {
                    length: Some(*length),
                    absolute_volume_millions: Some(180.0),
                    use_volume_sum: Some(true),
                },
            );
            let single = elastic_volume_weighted_moving_average(&input).unwrap();
            let start = row * batch.cols;
            let end = start + batch.cols;
            assert_series_close(&batch.values[start..end], &single.values);
        }
    }
}
