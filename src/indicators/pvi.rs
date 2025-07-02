//! # Positive Volume Index (PVI)
//!
//! The Positive Volume Index (PVI) tracks price changes only when volume increases, starting from an initial value.
//! Like ALMA, this indicator provides builder, batch, and streaming APIs, kernel stubs, and parameter expansion.
//!
//! ## Parameters
//! - **initial_value**: Starting PVI value. Default is 1000.0.
//!
//! ## Errors
//! - **AllValuesNaN**: All input values are NaN.
//! - **MismatchedLength**: Close and volume arrays have different lengths.
//! - **EmptyData**: Provided slices are empty.
//! - **NotEnoughValidData**: Less than 2 valid points after the first valid index.
//!
//! ## Returns
//! - **`Ok(PviOutput)`** on success, with a Vec<f64> of PVI values matching the input length.
//! - **`Err(PviError)`** otherwise.
//!
use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use thiserror::Error;
use std::convert::AsRef;

impl<'a> AsRef<[f64]> for PviInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            PviData::Slices { close, .. } => close,
            PviData::Candles { candles, close_source, .. } => source_type(candles, close_source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum PviData<'a> {
    Candles {
        candles: &'a Candles,
        close_source: &'a str,
        volume_source: &'a str,
    },
    Slices {
        close: &'a [f64],
        volume: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct PviOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct PviParams {
    pub initial_value: Option<f64>,
}

impl Default for PviParams {
    fn default() -> Self {
        Self {
            initial_value: Some(1000.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PviInput<'a> {
    pub data: PviData<'a>,
    pub params: PviParams,
}

impl<'a> PviInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        close_source: &'a str,
        volume_source: &'a str,
        params: PviParams,
    ) -> Self {
        Self {
            data: PviData::Candles {
                candles,
                close_source,
                volume_source,
            },
            params,
        }
    }
    #[inline]
    pub fn from_slices(close: &'a [f64], volume: &'a [f64], params: PviParams) -> Self {
        Self {
            data: PviData::Slices { close, volume },
            params,
        }
    }
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "close", "volume", PviParams::default())
    }
    #[inline]
    pub fn get_initial_value(&self) -> f64 {
        self.params.initial_value.unwrap_or(1000.0)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct PviBuilder {
    initial_value: Option<f64>,
    kernel: Kernel,
}

impl Default for PviBuilder {
    fn default() -> Self {
        Self {
            initial_value: None,
            kernel: Kernel::Auto,
        }
    }
}

impl PviBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn initial_value(mut self, v: f64) -> Self {
        self.initial_value = Some(v);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<PviOutput, PviError> {
        let p = PviParams {
            initial_value: self.initial_value,
        };
        let i = PviInput::from_candles(c, "close", "volume", p);
        pvi_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, close: &[f64], volume: &[f64]) -> Result<PviOutput, PviError> {
        let p = PviParams {
            initial_value: self.initial_value,
        };
        let i = PviInput::from_slices(close, volume, p);
        pvi_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<PviStream, PviError> {
        let p = PviParams {
            initial_value: self.initial_value,
        };
        PviStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum PviError {
    #[error("pvi: Empty data provided.")]
    EmptyData,
    #[error("pvi: All values are NaN.")]
    AllValuesNaN,
    #[error("pvi: Close and volume data have different lengths.")]
    MismatchedLength,
    #[error("pvi: Not enough valid data: needed at least 2 valid data points.")]
    NotEnoughValidData,
}

#[inline]
pub fn pvi(input: &PviInput) -> Result<PviOutput, PviError> {
    pvi_with_kernel(input, Kernel::Auto)
}

pub fn pvi_with_kernel(input: &PviInput, kernel: Kernel) -> Result<PviOutput, PviError> {
    let (close, volume) = match &input.data {
        PviData::Candles { candles, close_source, volume_source } => {
            let c = source_type(candles, close_source);
            let v = source_type(candles, volume_source);
            (c, v)
        }
        PviData::Slices { close, volume } => (*close, *volume),
    };

    if close.is_empty() || volume.is_empty() {
        return Err(PviError::EmptyData);
    }
    if close.len() != volume.len() {
        return Err(PviError::MismatchedLength);
    }
    let first_valid_idx = close.iter()
        .zip(volume.iter())
        .position(|(&c, &v)| !c.is_nan() && !v.is_nan())
        .ok_or(PviError::AllValuesNaN)?;
    if (close.len() - first_valid_idx) < 2 {
        return Err(PviError::NotEnoughValidData);
    }

    let mut out = vec![f64::NAN; close.len()];
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                pvi_scalar(close, volume, first_valid_idx, input.get_initial_value(), &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                pvi_avx2(close, volume, first_valid_idx, input.get_initial_value(), &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                pvi_avx512(close, volume, first_valid_idx, input.get_initial_value(), &mut out)
            }
            _ => unreachable!(),
        }
    }
    Ok(PviOutput { values: out })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn pvi_avx512(
    close: &[f64],
    volume: &[f64],
    first_valid: usize,
    initial: f64,
    out: &mut [f64],
) {
    unsafe {
        if close.len() <= 32 {
            pvi_avx512_short(close, volume, first_valid, initial, out)
        
            } else {
            pvi_avx512_long(close, volume, first_valid, initial, out)
        }
    }
}

#[inline]
pub fn pvi_scalar(
    close: &[f64],
    volume: &[f64],
    first_valid: usize,
    initial: f64,
    out: &mut [f64],
) {
    let mut pvi_current = initial;
    out[first_valid] = pvi_current;
    for i in (first_valid + 1)..close.len() {
        if !close[i].is_nan() && !volume[i].is_nan() && !close[i - 1].is_nan() && !volume[i - 1].is_nan() {
            if volume[i] > volume[i - 1] {
                pvi_current += ((close[i] - close[i - 1]) / close[i - 1]) * pvi_current;
            }
            out[i] = pvi_current;
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn pvi_avx2(
    close: &[f64],
    volume: &[f64],
    first_valid: usize,
    initial: f64,
    out: &mut [f64],
) {
    // Forward to scalar, AVX2 stub
    pvi_scalar(close, volume, first_valid, initial, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn pvi_avx512_short(
    close: &[f64],
    volume: &[f64],
    first_valid: usize,
    initial: f64,
    out: &mut [f64],
) {
    // Forward to scalar, AVX512 short stub
    pvi_scalar(close, volume, first_valid, initial, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn pvi_avx512_long(
    close: &[f64],
    volume: &[f64],
    first_valid: usize,
    initial: f64,
    out: &mut [f64],
) {
    // Forward to scalar, AVX512 long stub
    pvi_scalar(close, volume, first_valid, initial, out)
}

#[derive(Debug, Clone)]
pub struct PviStream {
    initial_value: f64,
    last_close: f64,
    last_volume: f64,
    curr: f64,
    state: StreamState,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StreamState {
    Init,
    Valid,
}

impl PviStream {
    pub fn try_new(params: PviParams) -> Result<Self, PviError> {
        let initial = params.initial_value.unwrap_or(1000.0);
        Ok(Self {
            initial_value: initial,
            last_close: f64::NAN,
            last_volume: f64::NAN,
            curr: initial,
            state: StreamState::Init,
        })
    }
    #[inline(always)]
    pub fn update(&mut self, close: f64, volume: f64) -> Option<f64> {
        match self.state {
            StreamState::Init => {
                if close.is_nan() || volume.is_nan() {
                    None
                
                    } else {
                    self.last_close = close;
                    self.last_volume = volume;
                    self.curr = self.initial_value;
                    self.state = StreamState::Valid;
                    Some(self.curr)
                }
            }
            StreamState::Valid => {
                if close.is_nan() || volume.is_nan() || self.last_close.is_nan() || self.last_volume.is_nan() {
                    None
                
                    } else {
                    if volume > self.last_volume {
                        self.curr += ((close - self.last_close) / self.last_close) * self.curr;
                    }
                    self.last_close = close;
                    self.last_volume = volume;
                    Some(self.curr)
                }
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct PviBatchRange {
    pub initial_value: (f64, f64, f64),
}

impl Default for PviBatchRange {
    fn default() -> Self {
        Self {
            initial_value: (1000.0, 1000.0, 0.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct PviBatchBuilder {
    range: PviBatchRange,
    kernel: Kernel,
}

impl PviBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline]
    pub fn initial_value_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.initial_value = (start, end, step);
        self
    }
    #[inline]
    pub fn initial_value_static(mut self, v: f64) -> Self {
        self.range.initial_value = (v, v, 0.0);
        self
    }
    pub fn apply_slices(self, close: &[f64], volume: &[f64]) -> Result<PviBatchOutput, PviError> {
        pvi_batch_with_kernel(close, volume, &self.range, self.kernel)
    }
    pub fn with_default_slices(close: &[f64], volume: &[f64], k: Kernel) -> Result<PviBatchOutput, PviError> {
        PviBatchBuilder::new().kernel(k).apply_slices(close, volume)
    }
    pub fn apply_candles(self, c: &Candles, close_src: &str, vol_src: &str) -> Result<PviBatchOutput, PviError> {
        let close = source_type(c, close_src);
        let vol = source_type(c, vol_src);
        self.apply_slices(close, vol)
    }
    pub fn with_default_candles(c: &Candles) -> Result<PviBatchOutput, PviError> {
        PviBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "close", "volume")
    }
}

pub fn pvi_batch_with_kernel(
    close: &[f64],
    volume: &[f64],
    sweep: &PviBatchRange,
    k: Kernel,
) -> Result<PviBatchOutput, PviError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(PviError::EmptyData),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    pvi_batch_par_slice(close, volume, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct PviBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<PviParams>,
    pub rows: usize,
    pub cols: usize,
}
impl PviBatchOutput {
    pub fn row_for_params(&self, p: &PviParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            (c.initial_value.unwrap_or(1000.0) - p.initial_value.unwrap_or(1000.0)).abs() < 1e-12
        })
    }
    pub fn values_for(&self, p: &PviParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &PviBatchRange) -> Vec<PviParams> {
    fn axis_f64((start, end, step): (f64, f64, f64)) -> Vec<f64> {
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return vec![start];
        }
        let mut v = Vec::new();
        let mut x = start;
        while x <= end + 1e-12 {
            v.push(x);
            x += step;
        }
        v
    }
    let initials = axis_f64(r.initial_value);
    let mut out = Vec::with_capacity(initials.len());
    for &v in &initials {
        out.push(PviParams { initial_value: Some(v) });
    }
    out
}

#[inline(always)]
pub fn pvi_batch_slice(
    close: &[f64],
    volume: &[f64],
    sweep: &PviBatchRange,
    kern: Kernel,
) -> Result<PviBatchOutput, PviError> {
    pvi_batch_inner(close, volume, sweep, kern, false)
}

#[inline(always)]
pub fn pvi_batch_par_slice(
    close: &[f64],
    volume: &[f64],
    sweep: &PviBatchRange,
    kern: Kernel,
) -> Result<PviBatchOutput, PviError> {
    pvi_batch_inner(close, volume, sweep, kern, true)
}

#[inline(always)]
fn pvi_batch_inner(
    close: &[f64],
    volume: &[f64],
    sweep: &PviBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<PviBatchOutput, PviError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(PviError::EmptyData);
    }
    if close.is_empty() || volume.is_empty() {
        return Err(PviError::EmptyData);
    }
    if close.len() != volume.len() {
        return Err(PviError::MismatchedLength);
    }
    let first_valid_idx = close.iter()
        .zip(volume.iter())
        .position(|(&c, &v)| !c.is_nan() && !v.is_nan())
        .ok_or(PviError::AllValuesNaN)?;
    if (close.len() - first_valid_idx) < 2 {
        return Err(PviError::NotEnoughValidData);
    }

    let rows = combos.len();
    let cols = close.len();
    let mut values = vec![f64::NAN; rows * cols];

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let iv = combos[row].initial_value.unwrap_or(1000.0);
        match kern {
            Kernel::Scalar => pvi_row_scalar(close, volume, first_valid_idx, iv, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => pvi_row_avx2(close, volume, first_valid_idx, iv, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => pvi_row_avx512(close, volume, first_valid_idx, iv, out_row),
            _ => unreachable!(),
        }
    };
    if parallel {

        #[cfg(not(target_arch = "wasm32"))] {

        values

                    .par_chunks_mut(cols)

                    .enumerate()

                    .for_each(|(row, slice)| do_row(row, slice));

        }

        #[cfg(target_arch = "wasm32")] {

        for (row, slice) in values.chunks_mut(cols).enumerate() {

                    do_row(row, slice);

        }

    } else {
        for (row, slice) in values.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }
    Ok(PviBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
unsafe fn pvi_row_scalar(
    close: &[f64],
    volume: &[f64],
    first: usize,
    initial: f64,
    out: &mut [f64],
) {
    pvi_scalar(close, volume, first, initial, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn pvi_row_avx2(
    close: &[f64],
    volume: &[f64],
    first: usize,
    initial: f64,
    out: &mut [f64],
) {
    pvi_scalar(close, volume, first, initial, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn pvi_row_avx512(
    close: &[f64],
    volume: &[f64],
    first: usize,
    initial: f64,
    out: &mut [f64],
) {
    if close.len() <= 32 {
        pvi_row_avx512_short(close, volume, first, initial, out);
    
        } else {
        pvi_row_avx512_long(close, volume, first, initial, out);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn pvi_row_avx512_short(
    close: &[f64],
    volume: &[f64],
    first: usize,
    initial: f64,
    out: &mut [f64],
) {
    pvi_scalar(close, volume, first, initial, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn pvi_row_avx512_long(
    close: &[f64],
    volume: &[f64],
    first: usize,
    initial: f64,
    out: &mut [f64],
) {
    pvi_scalar(close, volume, first, initial, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_pvi_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = PviParams { initial_value: None };
        let input = PviInput::from_candles(&candles, "close", "volume", default_params);
        let output = pvi_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_pvi_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let close_data = [100.0, 102.0, 101.0, 103.0, 103.0, 105.0];
        let volume_data = [500.0, 600.0, 500.0, 700.0, 680.0, 900.0];
        let params = PviParams { initial_value: Some(1000.0) };
        let input = PviInput::from_slices(&close_data, &volume_data, params);
        let output = pvi_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), close_data.len());
        assert!((output.values[0] - 1000.0).abs() < 1e-6);
        Ok(())
    }

    fn check_pvi_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = PviInput::with_default_candles(&candles);
        let output = pvi_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_pvi_empty_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let close_data = [];
        let volume_data = [];
        let params = PviParams::default();
        let input = PviInput::from_slices(&close_data, &volume_data, params);
        let result = pvi_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_pvi_mismatched_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let close_data = [100.0, 101.0];
        let volume_data = [500.0];
        let params = PviParams::default();
        let input = PviInput::from_slices(&close_data, &volume_data, params);
        let result = pvi_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_pvi_all_values_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let close_data = [f64::NAN, f64::NAN, f64::NAN];
        let volume_data = [f64::NAN, f64::NAN, f64::NAN];
        let params = PviParams::default();
        let input = PviInput::from_slices(&close_data, &volume_data, params);
        let result = pvi_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_pvi_not_enough_valid_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let close_data = [f64::NAN, 100.0];
        let volume_data = [f64::NAN, 500.0];
        let params = PviParams::default();
        let input = PviInput::from_slices(&close_data, &volume_data, params);
        let result = pvi_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_pvi_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let close_data = [100.0, 102.0, 101.0, 103.0, 103.0, 105.0];
        let volume_data = [500.0, 600.0, 500.0, 700.0, 680.0, 900.0];
        let params = PviParams { initial_value: Some(1000.0) };
        let input = PviInput::from_slices(&close_data, &volume_data, params.clone());
        let batch_output = pvi_with_kernel(&input, kernel)?.values;

        let mut stream = PviStream::try_new(params)?;
        let mut stream_values = Vec::with_capacity(close_data.len());
        for (&close, &vol) in close_data.iter().zip(volume_data.iter()) {
            match stream.update(close, vol) {
                Some(val) => stream_values.push(val),
                None => stream_values.push(f64::NAN),
            }
        }
        assert_eq!(batch_output.len(), stream_values.len());
        for (i, (&b, &s)) in batch_output.iter().zip(stream_values.iter()).enumerate() {
            if b.is_nan() && s.is_nan() { continue; }
            let diff = (b - s).abs();
            assert!(diff < 1e-9, "[{}] PVI streaming mismatch at idx {}: batch={}, stream={}, diff={}", test_name, i, b, s, diff);
        }
        Ok(())
    }

    macro_rules! generate_all_pvi_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $(
                    #[test]
                    fn [<$test_fn _scalar_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar);
                    }
                )*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $(
                    #[test]
                    fn [<$test_fn _avx2_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2);
                    }
                    #[test]
                    fn [<$test_fn _avx512_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512);
                    }
                )*
            }
        }
    }

    generate_all_pvi_tests!(
        check_pvi_partial_params,
        check_pvi_accuracy,
        check_pvi_default_candles,
        check_pvi_empty_data,
        check_pvi_mismatched_length,
        check_pvi_all_values_nan,
        check_pvi_not_enough_valid_data,
        check_pvi_streaming
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let close_data = [100.0, 102.0, 101.0, 103.0, 103.0, 105.0];
        let volume_data = [500.0, 600.0, 500.0, 700.0, 680.0, 900.0];
        let output = PviBatchBuilder::new().kernel(kernel).apply_slices(&close_data, &volume_data)?;
        let def = PviParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), close_data.len());
        Ok(())
    }

    macro_rules! gen_batch_tests {
        ($fn_name:ident) => {
            paste::paste! {
                #[test] fn [<$fn_name _scalar>]()      {
                    let _ = $fn_name(stringify!([<$fn_name _scalar>]), Kernel::ScalarBatch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx2>]()        {
                    let _ = $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx512>]()      {
                    let _ = $fn_name(stringify!([<$fn_name _avx512>]), Kernel::Avx512Batch);
                }
                #[test] fn [<$fn_name _auto_detect>]() {
                    let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto);
                }
            }
        };
    }
    gen_batch_tests!(check_batch_default_row);
}
