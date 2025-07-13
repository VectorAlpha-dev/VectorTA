//! # Volume Weighted Moving Average (VWMA)
//!
//! VWMA weights each price by its volume over a moving window. This captures the price action with regard to trading activity.
//!
//! ## Parameters
//! - **period**: Number of bars to use for weighting (default 20). Must be ≥ 1 and ≤ data length.
//!
//! ## Errors
//! - **AllValuesNaN**: vwma: All price-volume pairs are NaN.
//! - **InvalidPeriod**: vwma: Period is zero or exceeds data length.
//! - **PriceVolumeMismatch**: vwma: Price and volume lengths do not match.
//! - **NotEnoughValidData**: vwma: Not enough valid price-volume pairs for the requested period.
//!
//! ## Returns
//! - **Ok(VwmaOutput)** on success, containing the VWMA as a Vec<f64>.
//! - **Err(VwmaError)** on error.

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_kernel, detect_best_batch_kernel, alloc_with_nan_prefix, init_matrix_prefixes, make_uninit_matrix};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::error::Error;
use thiserror::Error;
use std::convert::AsRef;
use std::mem::MaybeUninit;

#[derive(Debug, Clone)]
pub enum VwmaData<'a> {
    Candles { candles: &'a Candles, source: &'a str },
    CandlesPlusPrices { candles: &'a Candles, prices: &'a [f64] },
    Slice { prices: &'a [f64], volumes: &'a [f64] },
}

#[derive(Debug, Clone)]
pub struct VwmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct VwmaParams {
    pub period: Option<usize>,
}

impl Default for VwmaParams {
    fn default() -> Self {
        Self { period: Some(20) }
    }
}

#[derive(Debug, Clone)]
pub struct VwmaInput<'a> {
    pub data: VwmaData<'a>,
    pub params: VwmaParams,
}

impl<'a> VwmaInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: VwmaParams) -> Self {
        Self {
            data: VwmaData::Candles { candles, source },
            params,
        }
    }

    pub fn from_candles_plus_prices(
        candles: &'a Candles,
        prices: &'a [f64],
        params: VwmaParams,
    ) -> Self {
        Self {
            data: VwmaData::CandlesPlusPrices { candles, prices },
            params,
        }
    }

    pub fn from_slice(prices: &'a [f64], volumes: &'a [f64], params: VwmaParams) -> Self {
        Self {
            data: VwmaData::Slice { prices, volumes },
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: VwmaData::Candles {
                candles,
                source: "close",
            },
            params: VwmaParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(20)
    }
}

impl<'a> AsRef<[f64]> for VwmaInput<'a> {
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            VwmaData::Candles { candles, source } => source_type(candles, source),
            VwmaData::CandlesPlusPrices { prices, .. } => prices,
            VwmaData::Slice { prices, .. } => prices,
        }
    }
}

#[derive(Debug, Error)]
pub enum VwmaError {
    #[error("vwma: All values are NaN.")]
    AllValuesNaN,
    #[error("vwma: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("vwma: Price and volume mismatch: price length = {price_len}, volume length = {volume_len}")]
    PriceVolumeMismatch { price_len: usize, volume_len: usize },
    #[error("vwma: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

#[derive(Copy, Clone, Debug)]
pub struct VwmaBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for VwmaBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl VwmaBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn period(mut self, n: usize) -> Self {
        self.period = Some(n);
        self
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    pub fn apply(self, c: &Candles) -> Result<VwmaOutput, VwmaError> {
        let p = VwmaParams { period: self.period };
        let i = VwmaInput::from_candles(c, "close", p);
        vwma_with_kernel(&i, self.kernel)
    }
    pub fn apply_slice(self, prices: &[f64], volumes: &[f64]) -> Result<VwmaOutput, VwmaError> {
        let p = VwmaParams { period: self.period };
        let i = VwmaInput::from_slice(prices, volumes, p);
        vwma_with_kernel(&i, self.kernel)
    }
    pub fn into_stream(self) -> Result<VwmaStream, VwmaError> {
        let p = VwmaParams { period: self.period };
        VwmaStream::try_new(p)
    }
}

#[inline]
pub fn vwma(input: &VwmaInput) -> Result<VwmaOutput, VwmaError> {
    vwma_with_kernel(input, Kernel::Auto)
}

pub fn vwma_with_kernel(input: &VwmaInput, kernel: Kernel) -> Result<VwmaOutput, VwmaError> {
    let (price, volume): (&[f64], &[f64]) = match &input.data {
        VwmaData::Candles { candles, source } => (
            source_type(candles, source),
            source_type(candles, "volume"),
        ),
        VwmaData::CandlesPlusPrices { candles, prices } => (
            prices,
            source_type(candles, "volume"),
        ),
        VwmaData::Slice { prices, volumes } => (prices, volumes),
    };
    let len = price.len();
    let period = input.get_period();

    if period == 0 || period > len {
        return Err(VwmaError::InvalidPeriod { period, data_len: len });
    }
    if volume.len() != len {
        return Err(VwmaError::PriceVolumeMismatch {
            price_len: len,
            volume_len: volume.len(),
        });
    }
    let first = price
        .iter()
        .zip(volume.iter())
        .position(|(&p, &v)| !p.is_nan() && !v.is_nan())
        .ok_or(VwmaError::AllValuesNaN)?;

    if (len - first) < period {
        return Err(VwmaError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let warm   = first + period;
    let mut out = alloc_with_nan_prefix(len, warm);

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                vwma_scalar(price, volume, period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                vwma_avx2(price, volume, period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                vwma_avx512(price, volume, period, first, &mut out)
            }
            _ => unreachable!(),
        }
    }

    Ok(VwmaOutput { values: out })
}

#[inline]
pub fn vwma_scalar(
    price: &[f64],
    volume: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    let len = price.len();
    if len < period { return; }
    let mut sum = 0.0;
    let mut vsum = 0.0;
    for i in 0..period {
        let idx = first + i;
        sum += price[idx] * volume[idx];
        vsum += volume[idx];
    }
    let first_idx = first + period - 1;
    out[first_idx] = sum / vsum;
    for i in (first_idx + 1)..len {
        sum += price[i] * volume[i];
        vsum += volume[i];
        let old_idx = i - period;
        sum -= price[old_idx] * volume[old_idx];
        vsum -= volume[old_idx];
        out[i] = sum / vsum;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn vwma_avx512(
    price: &[f64],
    volume: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    // stub: fallback to scalar
    vwma_scalar(price, volume, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn vwma_avx2(
    price: &[f64],
    volume: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    // stub: fallback to scalar
    vwma_scalar(price, volume, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vwma_avx512_short(
    price: &[f64],
    volume: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    vwma_scalar(price, volume, period, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vwma_avx512_long(
    price: &[f64],
    volume: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    vwma_scalar(price, volume, period, first, out)
}

#[inline(always)]
pub fn vwma_batch_with_kernel(
    price: &[f64],
    volume: &[f64],
    sweep: &VwmaBatchRange,
    kernel: Kernel,
) -> Result<VwmaBatchOutput, VwmaError> {
    let chosen = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(VwmaError::InvalidPeriod {
                period: 0,
                data_len: 0,
            })
        }
    };
    let simd = match chosen {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    vwma_batch_par_slice(price, volume, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct VwmaBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for VwmaBatchRange {
    fn default() -> Self {
        Self {
            period: (20, 50, 1),
        }
    }
}

#[derive(Clone, Debug)]
pub struct VwmaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<VwmaParams>,
    pub rows: usize,
    pub cols: usize,
}
impl VwmaBatchOutput {
    pub fn row_for_params(&self, p: &VwmaParams) -> Option<usize> {
        self.combos.iter().position(|c| c.period.unwrap_or(20) == p.period.unwrap_or(20))
    }
    pub fn values_for(&self, p: &VwmaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

fn expand_grid_vwma(r: &VwmaBatchRange) -> Vec<VwmaParams> {
    let (start, end, step) = r.period;
    if step == 0 || start == end {
        return vec![VwmaParams { period: Some(start) }];
    }
    (start..=end)
        .step_by(step)
        .map(|p| VwmaParams { period: Some(p) })
        .collect()
}

#[inline(always)]
pub fn vwma_batch_slice(
    price: &[f64],
    volume: &[f64],
    sweep: &VwmaBatchRange,
    kern: Kernel,
) -> Result<VwmaBatchOutput, VwmaError> {
    vwma_batch_inner(price, volume, sweep, kern, false)
}

#[inline(always)]
pub fn vwma_batch_par_slice(
    price: &[f64],
    volume: &[f64],
    sweep: &VwmaBatchRange,
    kern: Kernel,
) -> Result<VwmaBatchOutput, VwmaError> {
    vwma_batch_inner(price, volume, sweep, kern, true)
}

#[inline]
fn vwma_batch_inner(
    price: &[f64],
    volume: &[f64],
    sweep: &VwmaBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<VwmaBatchOutput, VwmaError> {
    // ---------- 0. parameter checks ----------
    let combos = expand_grid_vwma(sweep);
    if combos.is_empty() {
        return Err(VwmaError::InvalidPeriod { period: 0, data_len: 0 });
    }

    let len   = price.len();
    let first = price
        .iter()
        .zip(volume.iter())
        .position(|(&p, &v)| !p.is_nan() && !v.is_nan())
        .ok_or(VwmaError::AllValuesNaN)?;

    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if len - first < max_p {
        return Err(VwmaError::NotEnoughValidData {
            needed: max_p,
            valid : len - first,
        });
    }

    // ---------- 1. allocate matrix as MaybeUninit<f64> + write NaN prefixes ----------
    let rows = combos.len();
    let cols = len;

    let warm_prefixes: Vec<usize> =
        combos.iter().map(|c| first + c.period.unwrap()).collect();

    let mut raw = make_uninit_matrix(rows, cols);
    unsafe { init_matrix_prefixes(&mut raw, cols, &warm_prefixes) };

    // ---------- 2. row worker (fills one row) ----------
    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period  = combos[row].period.unwrap();

        // Cast the `MaybeUninit` slice for this row to a plain `f64` slice.
        let out_row = core::slice::from_raw_parts_mut(
            dst_mu.as_mut_ptr() as *mut f64,
            dst_mu.len(),
        );

        match kern {
            Kernel::Scalar => vwma_row_scalar (price, volume, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2   => vwma_row_avx2   (price, volume, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => vwma_row_avx512 (price, volume, first, period, out_row),
            _ => unreachable!(),
        }
    };

    // ---------- 3. run every row ----------
    if parallel {

        #[cfg(not(target_arch = "wasm32"))] {

        raw.par_chunks_mut(cols)

                   .enumerate()

                   .for_each(|(row, slice)| do_row(row, slice));

        }

        #[cfg(target_arch = "wasm32")] {

        for (row, slice) in raw.chunks_mut(cols).enumerate() {

                    do_row(row, slice);

        }
        }
    } else {
        for (row, slice) in raw.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    // ---------- 4. transmute to Vec<f64> & return ----------
    let values: Vec<f64> = unsafe { std::mem::transmute(raw) };
    Ok(VwmaBatchOutput { values, combos, rows, cols })
}

#[inline(always)]
pub unsafe fn vwma_row_scalar(
    price: &[f64],
    volume: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    vwma_scalar(price, volume, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vwma_row_avx2(
    price: &[f64],
    volume: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    vwma_scalar(price, volume, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vwma_row_avx512(
    price: &[f64],
    volume: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    if period <= 32 {
        vwma_row_avx512_short(price, volume, first, period, out);
    
        } else {
        vwma_row_avx512_long(price, volume, first, period, out);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vwma_row_avx512_short(
    price: &[f64],
    volume: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    vwma_scalar(price, volume, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vwma_row_avx512_long(
    price: &[f64],
    volume: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    vwma_scalar(price, volume, period, first, out)
}

#[derive(Debug, Clone)]
pub struct VwmaStream {
    period: usize,
    prices: Vec<f64>,
    volumes: Vec<f64>,
    sum: f64,
    vsum: f64,
    head: usize,
    filled: bool,
}

impl VwmaStream {
    pub fn try_new(params: VwmaParams) -> Result<Self, VwmaError> {
        let period = params.period.unwrap_or(20);
        if period == 0 {
            return Err(VwmaError::InvalidPeriod { period, data_len: 0 });
        }
        Ok(Self {
            period,
            prices: vec![f64::NAN; period],
            volumes: vec![f64::NAN; period],
            sum: 0.0,
            vsum: 0.0,
            head: 0,
            filled: false,
        })
    }
    pub fn update(&mut self, price: f64, volume: f64) -> Option<f64> {
        let idx = self.head;
        if !self.filled {
            self.sum += price * volume;
            self.vsum += volume;
            self.prices[idx] = price;
            self.volumes[idx] = volume;
            self.head += 1;
            if self.head == self.period {
                self.head = 0;
                self.filled = true;
            }
            if !self.filled {
                return None;
            }
        } else {
            let old_p = self.prices[idx];
            let old_v = self.volumes[idx];
            self.sum += price * volume - old_p * old_v;
            self.vsum += volume - old_v;
            self.prices[idx] = price;
            self.volumes[idx] = volume;
            self.head = (self.head + 1) % self.period;
        }
        Some(self.sum / self.vsum)
    }
}

#[derive(Clone, Debug)]
pub struct VwmaBatchBuilder {
    range: VwmaBatchRange,
    kernel: Kernel,
}

impl Default for VwmaBatchBuilder {
    fn default() -> Self {
        Self {
            range: VwmaBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl VwmaBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    pub fn period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.period = (start, end, step);
        self
    }
    pub fn period_static(mut self, p: usize) -> Self {
        self.range.period = (p, p, 0);
        self
    }
    pub fn apply_slice(self, prices: &[f64], volumes: &[f64]) -> Result<VwmaBatchOutput, VwmaError> {
        vwma_batch_with_kernel(prices, volumes, &self.range, self.kernel)
    }
}

#[inline(always)]
fn expand_grid(_r: &VwmaBatchRange) -> Vec<VwmaParams> {
    expand_grid_vwma(_r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;
    fn check_vwma_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = VwmaParams { period: None };
        let input_default = VwmaInput::from_candles(&candles, "close", default_params);
        let output_default = vwma_with_kernel(&input_default, kernel)?;
        assert_eq!(output_default.values.len(), candles.close.len());
        let custom_params = VwmaParams { period: Some(10) };
        let input_custom = VwmaInput::from_candles(&candles, "hlc3", custom_params);
        let output_custom = vwma_with_kernel(&input_custom, kernel)?;
        assert_eq!(output_custom.values.len(), candles.close.len());
        Ok(())
    }
    fn check_vwma_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let close_prices = candles.select_candle_field("close")?;
        let params = VwmaParams { period: Some(20) };
        let input = VwmaInput::from_candles(&candles, "close", params);
        let vwma_result = vwma_with_kernel(&input, kernel)?;
        assert_eq!(vwma_result.values.len(), close_prices.len());
        let expected_last_five_vwma = [
            59201.87047121331,
            59217.157390630266,
            59195.74526905522,
            59196.261392450084,
            59151.22059588594,
        ];
        let start_index = vwma_result.values.len() - 5;
        let result_last_five_vwma = &vwma_result.values[start_index..];
        for (i, &val) in result_last_five_vwma.iter().enumerate() {
            let exp = expected_last_five_vwma[i];
            assert!(
                (val - exp).abs() < 1e-3,
                "[{}] VWMA mismatch at index {}: expected {}, got {}",
                test_name,
                i,
                exp,
                val
            );
        }
        Ok(())
    }
    fn check_vwma_input_with_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = VwmaInput::with_default_candles(&candles);
        match input.data {
            VwmaData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected VwmaData::Candles"),
        }
        Ok(())
    }
    fn check_vwma_candles_plus_prices(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let custom_prices = candles.close.iter().map(|v| v * 1.001).collect::<Vec<f64>>();
        let params = VwmaParams { period: Some(20) };
        let input = VwmaInput::from_candles_plus_prices(&candles, &custom_prices, params);
        let result = vwma_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), custom_prices.len());
        Ok(())
    }
    fn check_vwma_slice_data_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params_first = VwmaParams { period: Some(20) };
        let input_first = VwmaInput::from_candles(&candles, "close", params_first);
        let result_first = vwma_with_kernel(&input_first, kernel)?;
        assert_eq!(result_first.values.len(), candles.close.len());
        let params_second = VwmaParams { period: Some(10) };
        let input_second = VwmaInput::from_slice(&result_first.values, &candles.volume, params_second);
        let result_second = vwma_with_kernel(&input_second, kernel)?;
        assert_eq!(result_second.values.len(), result_first.values.len());
        let start = input_first.get_period() + input_second.get_period() - 2;
        for i in start..result_second.values.len() {
            assert!(!result_second.values[i].is_nan());
        }
        Ok(())
    }

    macro_rules! generate_all_vwma_tests {
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

    generate_all_vwma_tests!(
        check_vwma_partial_params,
        check_vwma_accuracy,
        check_vwma_input_with_default_candles,
        check_vwma_candles_plus_prices,
        check_vwma_slice_data_reinput
    );
    #[cfg(test)]
mod batch_tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = VwmaBatchBuilder::new()
            .kernel(kernel)
            .apply_slice(&c.close, &c.volume)?;

        let def = VwmaParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        // For illustration, use the known expected values from the previous test.
        // You may update these with precise reference values if needed.
        let expected = [
            59201.87047121331,
            59217.157390630266,
            59195.74526905522,
            59196.261392450084,
            59151.22059588594,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-3,
                "[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
            );
        }
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

}

#[cfg(feature = "python")]
#[pyfunction(name = "vwma")]
#[pyo3(signature = (prices, volumes, period, kernel=None))]
/// Compute the Volume Weighted Moving Average (VWMA) of the input data.
///
/// VWMA weights each price by its volume over a moving window to capture
/// price action with regard to trading activity.
///
/// Parameters:
/// -----------
/// prices : np.ndarray
///     Price data array (float64).
/// volumes : np.ndarray
///     Volume data array (float64), must be same length as prices.
/// period : int
///     Number of bars in the moving average window.
/// kernel : str, optional
///     Computation kernel to use: 'auto', 'scalar', 'avx2', 'avx512'.
///     Default is 'auto' which auto-detects the best available.
///
/// Returns:
/// --------
/// np.ndarray
///     Array of VWMA values, same length as input.
///
/// Raises:
/// -------
/// ValueError
///     If inputs are invalid (mismatched lengths, invalid period, etc).
pub fn vwma_py<'py>(
    py: Python<'py>,
    prices: numpy::PyReadonlyArray1<'py, f64>,
    volumes: numpy::PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use numpy::{PyArray1, PyArrayMethods};

    let prices_slice = prices.as_slice()?;
    let volumes_slice = volumes.as_slice()?;

    // Parse kernel string to enum
    let kern = match kernel {
        None | Some("auto") => Kernel::Auto,
        Some("scalar") => Kernel::Scalar,
        Some("avx2") => Kernel::Avx2,
        Some("avx512") => Kernel::Avx512,
        Some(k) => return Err(PyValueError::new_err(format!("Unknown kernel: {}", k))),
    };

    // Build input struct
    let params = VwmaParams { period: Some(period) };
    let vwma_in = VwmaInput::from_slice(prices_slice, volumes_slice, params);

    // Allocate NumPy output buffer
    let out_arr = unsafe { PyArray1::<f64>::new(py, [prices_slice.len()], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    // Heavy lifting without the GIL
    py.allow_threads(|| -> Result<(), VwmaError> {
        let (price, volume): (&[f64], &[f64]) = match &vwma_in.data {
            VwmaData::Slice { prices, volumes } => (prices, volumes),
            _ => unreachable!(),
        };
        
        let len = price.len();
        let period = vwma_in.get_period();

        if period == 0 || period > len {
            return Err(VwmaError::InvalidPeriod { period, data_len: len });
        }
        if volume.len() != len {
            return Err(VwmaError::PriceVolumeMismatch {
                price_len: len,
                volume_len: volume.len(),
            });
        }

        let first = price
            .iter()
            .zip(volume.iter())
            .position(|(&p, &v)| !p.is_nan() && !v.is_nan())
            .ok_or(VwmaError::AllValuesNaN)?;

        if (len - first) < period {
            return Err(VwmaError::NotEnoughValidData {
                needed: period,
                valid: len - first,
            });
        }

        let warm = first + period;
        
        // Initialize prefix with NaN
        slice_out[..warm - 1].fill(f64::NAN);

        let chosen = match kern {
            Kernel::Auto => detect_best_kernel(),
            other => other,
        };

        unsafe {
            match chosen {
                Kernel::Scalar | Kernel::ScalarBatch => {
                    vwma_scalar(price, volume, period, first, slice_out)
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx2 | Kernel::Avx2Batch => {
                    vwma_avx2(price, volume, period, first, slice_out)
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx512 | Kernel::Avx512Batch => {
                    vwma_avx512(price, volume, period, first, slice_out)
                }
                _ => unreachable!(),
            }
        }

        Ok(())
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(out_arr.into())
}

#[cfg(feature = "python")]
#[pyclass(name = "VwmaStream")]
pub struct VwmaStreamPy {
    stream: VwmaStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl VwmaStreamPy {
    #[new]
    fn new(period: usize) -> PyResult<Self> {
        let params = VwmaParams { period: Some(period) };
        let stream = VwmaStream::try_new(params)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(VwmaStreamPy { stream })
    }

    /// Updates the stream with new price and volume values and returns the calculated VWMA value.
    /// Returns `None` if the buffer is not yet full.
    fn update(&mut self, price: f64, volume: f64) -> Option<f64> {
        self.stream.update(price, volume)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "vwma_batch")]
#[pyo3(signature = (prices, volumes, period_range, kernel=None))]
/// Compute VWMA for multiple period values in a single pass.
///
/// Parameters:
/// -----------
/// prices : np.ndarray
///     Price data array (float64).
/// volumes : np.ndarray
///     Volume data array (float64), must be same length as prices.
/// period_range : tuple
///     (start, end, step) for period values to compute.
/// kernel : str, optional
///     Computation kernel: 'auto', 'scalar', 'avx2', 'avx512'.
///     Default is 'auto' which auto-detects the best available.
///
/// Returns:
/// --------
/// dict
///     Dictionary with 'values' (2D array) and 'periods' array.
pub fn vwma_batch_py<'py>(
    py: Python<'py>,
    prices: numpy::PyReadonlyArray1<'py, f64>,
    volumes: numpy::PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let prices_slice = prices.as_slice()?;
    let volumes_slice = volumes.as_slice()?;

    let sweep = VwmaBatchRange { period: period_range };

    // Expand grid to know rows*cols
    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = prices_slice.len();

    // Pre-allocate NumPy array
    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    // Parse kernel string to enum
    let kern = match kernel {
        None | Some("auto") => Kernel::Auto,
        Some("scalar") => Kernel::ScalarBatch,
        Some("avx2") => Kernel::Avx2Batch,
        Some("avx512") => Kernel::Avx512Batch,
        Some(k) => return Err(PyValueError::new_err(format!("Unknown kernel: {}", k))),
    };

    // Heavy work without the GIL
    py.allow_threads(|| {
        let kernel = match kern {
            Kernel::Auto => detect_best_batch_kernel(),
            k => k,
        };
        let simd = match kernel {
            Kernel::Avx512Batch => Kernel::Avx512,
            Kernel::Avx2Batch => Kernel::Avx2,
            Kernel::ScalarBatch => Kernel::Scalar,
            _ => unreachable!(),
        };
        
        // Initialize output with NaN and compute
        let len = prices_slice.len();
        let first = prices_slice
            .iter()
            .zip(volumes_slice.iter())
            .position(|(&p, &v)| !p.is_nan() && !v.is_nan())
            .ok_or(VwmaError::AllValuesNaN)?;

        let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
        if len - first < max_p {
            return Err(VwmaError::NotEnoughValidData {
                needed: max_p,
                valid: len - first,
            });
        }

        // Initialize matrix prefixes with NaN
        let warm_prefixes: Vec<usize> = combos.iter().map(|c| first + c.period.unwrap()).collect();
        
        // Convert slice_out to MaybeUninit for init_matrix_prefixes
        let out_uninit = unsafe {
            std::slice::from_raw_parts_mut(
                slice_out.as_mut_ptr() as *mut MaybeUninit<f64>,
                slice_out.len()
            )
        };
        
        unsafe { init_matrix_prefixes(out_uninit, cols, &warm_prefixes) };

        // Process each row
        for (row, combo) in combos.iter().enumerate() {
            let period = combo.period.unwrap();
            let row_start = row * cols;
            let row_slice = &mut slice_out[row_start..row_start + cols];
            
            unsafe {
                match simd {
                    Kernel::Scalar => vwma_row_scalar(prices_slice, volumes_slice, first, period, row_slice),
                    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                    Kernel::Avx2 => vwma_row_avx2(prices_slice, volumes_slice, first, period, row_slice),
                    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                    Kernel::Avx512 => vwma_row_avx512(prices_slice, volumes_slice, first, period, row_slice),
                    _ => unreachable!(),
                }
            }
        }
        
        Ok::<_, VwmaError>(())
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Build dict with the GIL
    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "periods",
        combos
            .iter()
            .map(|p| p.period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;

    Ok(dict.into())
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vwma_js(prices: &[f64], volumes: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
    let params = VwmaParams { period: Some(period) };
    let input = VwmaInput::from_slice(prices, volumes, params);

    vwma_with_kernel(&input, Kernel::Scalar)
        .map(|o| o.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vwma_batch_js(
    prices: &[f64],
    volumes: &[f64],
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = VwmaBatchRange {
        period: (period_start, period_end, period_step),
    };

    // Use the existing batch function with parallel=false for WASM
    vwma_batch_inner(prices, volumes, &sweep, Kernel::Scalar, false)
        .map(|output| output.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vwma_batch_metadata_js(
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = VwmaBatchRange {
        period: (period_start, period_end, period_step),
    };

    let combos = expand_grid(&sweep);
    let mut metadata = Vec::with_capacity(combos.len());

    for combo in combos {
        metadata.push(combo.period.unwrap() as f64);
    }

    Ok(metadata)
}
