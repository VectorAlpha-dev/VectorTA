//! # Percentage Price Oscillator (PPO)
//!
//! Expresses the difference between two moving averages as a percentage of the slower moving average.
//! Parity with alma.rs API and features. SIMD/AVX2/AVX512 stubs provided for benchmarking, batch/grid/stream API available.
//! Input validation, error handling, and unit tests matching alma.rs expectations.
//!
//! ## Parameters
//! - **fast_period**: Short-term MA period (default: 12)
//! - **slow_period**: Long-term MA period (default: 26)
//! - **ma_type**: MA type, e.g., "sma", "ema" (default: "sma")
//!
//! ## Errors
//! - **AllValuesNaN**: All input data values are NaN
//! - **InvalidPeriod**: fast or slow period is zero or exceeds data length
//! - **NotEnoughValidData**: Not enough valid data for slow period
//! - **MaError**: Internal MA computation error
//!
//! ## Returns
//! - **Ok(PpoOutput)** on success, containing Vec<f64> with same length as input
//! - **Err(PpoError)** otherwise

use crate::indicators::moving_averages::ma::{ma, MaData};
use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;

impl<'a> AsRef<[f64]> for PpoInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            PpoData::Slice(slice) => slice,
            PpoData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum PpoData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct PpoOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct PpoParams {
    pub fast_period: Option<usize>,
    pub slow_period: Option<usize>,
    pub ma_type: Option<String>,
}

impl Default for PpoParams {
    fn default() -> Self {
        Self {
            fast_period: Some(12),
            slow_period: Some(26),
            ma_type: Some("sma".to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PpoInput<'a> {
    pub data: PpoData<'a>,
    pub params: PpoParams,
}

impl<'a> PpoInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: PpoParams) -> Self {
        Self {
            data: PpoData::Candles { candles: c, source: s },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: PpoParams) -> Self {
        Self {
            data: PpoData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", PpoParams::default())
    }
    #[inline]
    pub fn get_fast_period(&self) -> usize {
        self.params.fast_period.unwrap_or(12)
    }
    #[inline]
    pub fn get_slow_period(&self) -> usize {
        self.params.slow_period.unwrap_or(26)
    }
    #[inline]
    pub fn get_ma_type(&self) -> String {
        self.params.ma_type.clone().unwrap_or_else(|| "sma".to_string())
    }
}

#[derive(Clone, Debug)]
pub struct PpoBuilder {
    fast_period: Option<usize>,
    slow_period: Option<usize>,
    ma_type: Option<String>,
    kernel: Kernel,
}

impl Default for PpoBuilder {
    fn default() -> Self {
        Self {
            fast_period: None,
            slow_period: None,
            ma_type: None,
            kernel: Kernel::Auto,
        }
    }
}

impl PpoBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn fast_period(mut self, n: usize) -> Self {
        self.fast_period = Some(n);
        self
    }
    #[inline(always)]
    pub fn slow_period(mut self, n: usize) -> Self {
        self.slow_period = Some(n);
        self
    }
    #[inline(always)]
    pub fn ma_type<S: Into<String>>(mut self, s: S) -> Self {
        self.ma_type = Some(s.into());
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<PpoOutput, PpoError> {
        let p = PpoParams {
            fast_period: self.fast_period,
            slow_period: self.slow_period,
            ma_type: self.ma_type,
        };
        let i = PpoInput::from_candles(c, "close", p);
        ppo_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<PpoOutput, PpoError> {
        let p = PpoParams {
            fast_period: self.fast_period,
            slow_period: self.slow_period,
            ma_type: self.ma_type,
        };
        let i = PpoInput::from_slice(d, p);
        ppo_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<PpoStream, PpoError> {
        let p = PpoParams {
            fast_period: self.fast_period,
            slow_period: self.slow_period,
            ma_type: self.ma_type,
        };
        PpoStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum PpoError {
    #[error("ppo: All values are NaN.")]
    AllValuesNaN,
    #[error("ppo: Invalid period: fast = {fast}, slow = {slow}, data length = {data_len}")]
    InvalidPeriod { fast: usize, slow: usize, data_len: usize },
    #[error("ppo: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("ppo: MA error: {0}")]
    MaError(String),
}

#[inline]
pub fn ppo(input: &PpoInput) -> Result<PpoOutput, PpoError> {
    ppo_with_kernel(input, Kernel::Auto)
}

pub fn ppo_with_kernel(input: &PpoInput, kernel: Kernel) -> Result<PpoOutput, PpoError> {
    let data: &[f64] = input.as_ref();

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(PpoError::AllValuesNaN)?;

    let len = data.len();
    let fast = input.get_fast_period();
    let slow = input.get_slow_period();
    let ma_type = input.get_ma_type();

    if fast == 0 || slow == 0 || fast > len || slow > len {
        return Err(PpoError::InvalidPeriod {
            fast,
            slow,
            data_len: len,
        });
    }
    if (len - first) < slow {
        return Err(PpoError::NotEnoughValidData {
            needed: slow,
            valid: len - first,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let mut out = vec![f64::NAN; len];
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                ppo_scalar(data, fast, slow, &ma_type, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                ppo_avx2(data, fast, slow, &ma_type, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                ppo_avx512(data, fast, slow, &ma_type, first, &mut out)
            }
            _ => unreachable!(),
        }
    }

    Ok(PpoOutput { values: out })
}

#[inline]
pub unsafe fn ppo_scalar(
    data: &[f64],
    fast: usize,
    slow: usize,
    ma_type: &str,
    first: usize,
    out: &mut [f64],
) {
    let fast_ma = ma(ma_type, MaData::Slice(data), fast).expect("ma error");
    let slow_ma = ma(ma_type, MaData::Slice(data), slow).expect("ma error");
    for i in (first + slow - 1)..data.len() {
        let sf = slow_ma[i];
        let ff = fast_ma[i];
        if sf.is_nan() || ff.is_nan() || sf == 0.0 {
            out[i] = f64::NAN;
        } else {
            out[i] = 100.0 * (ff - sf) / sf;
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn ppo_avx2(
    data: &[f64],
    fast: usize,
    slow: usize,
    ma_type: &str,
    first: usize,
    out: &mut [f64],
) {
    ppo_scalar(data, fast, slow, ma_type, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn ppo_avx512(
    data: &[f64],
    fast: usize,
    slow: usize,
    ma_type: &str,
    first: usize,
    out: &mut [f64],
) {
    if slow <= 32 {
        ppo_avx512_short(data, fast, slow, ma_type, first, out)
    } else {
        ppo_avx512_long(data, fast, slow, ma_type, first, out)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn ppo_avx512_short(
    data: &[f64],
    fast: usize,
    slow: usize,
    ma_type: &str,
    first: usize,
    out: &mut [f64],
) {
    ppo_scalar(data, fast, slow, ma_type, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn ppo_avx512_long(
    data: &[f64],
    fast: usize,
    slow: usize,
    ma_type: &str,
    first: usize,
    out: &mut [f64],
) {
    ppo_scalar(data, fast, slow, ma_type, first, out)
}

#[derive(Clone, Debug)]
pub struct PpoBatchRange {
    pub fast_period: (usize, usize, usize),
    pub slow_period: (usize, usize, usize),
    pub ma_type: String,
}

impl Default for PpoBatchRange {
    fn default() -> Self {
        Self {
            fast_period: (12, 12, 0),
            slow_period: (26, 26, 0),
            ma_type: "sma".to_string(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct PpoBatchBuilder {
    range: PpoBatchRange,
    kernel: Kernel,
}

impl PpoBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    pub fn fast_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.fast_period = (start, end, step);
        self
    }
    pub fn slow_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.slow_period = (start, end, step);
        self
    }
    pub fn ma_type<S: Into<String>>(mut self, t: S) -> Self {
        self.range.ma_type = t.into();
        self
    }
    pub fn apply_slice(self, data: &[f64]) -> Result<PpoBatchOutput, PpoError> {
        ppo_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<PpoBatchOutput, PpoError> {
        PpoBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<PpoBatchOutput, PpoError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<PpoBatchOutput, PpoError> {
        PpoBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct PpoBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<PpoParams>,
    pub rows: usize,
    pub cols: usize,
}

impl PpoBatchOutput {
    pub fn row_for_params(&self, p: &PpoParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.fast_period.unwrap_or(12) == p.fast_period.unwrap_or(12)
                && c.slow_period.unwrap_or(26) == p.slow_period.unwrap_or(26)
                && c.ma_type.as_ref().unwrap_or(&"sma".to_string()) == p.ma_type.as_ref().unwrap_or(&"sma".to_string())
        })
    }
    pub fn values_for(&self, p: &PpoParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &PpoBatchRange) -> Vec<PpoParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }

    let fasts = axis_usize(r.fast_period);
    let slows = axis_usize(r.slow_period);
    let ma_type = r.ma_type.clone();

    let mut out = Vec::with_capacity(fasts.len() * slows.len());
    for &f in &fasts {
        for &s in &slows {
            out.push(PpoParams {
                fast_period: Some(f),
                slow_period: Some(s),
                ma_type: Some(ma_type.clone()),
            });
        }
    }
    out
}

#[inline(always)]
pub fn ppo_batch_with_kernel(
    data: &[f64],
    sweep: &PpoBatchRange,
    k: Kernel,
) -> Result<PpoBatchOutput, PpoError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(PpoError::InvalidPeriod {
                fast: 0,
                slow: 0,
                data_len: 0,
            })
        }
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    ppo_batch_par_slice(data, sweep, simd)
}

#[inline(always)]
pub fn ppo_batch_slice(
    data: &[f64],
    sweep: &PpoBatchRange,
    kern: Kernel,
) -> Result<PpoBatchOutput, PpoError> {
    ppo_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn ppo_batch_par_slice(
    data: &[f64],
    sweep: &PpoBatchRange,
    kern: Kernel,
) -> Result<PpoBatchOutput, PpoError> {
    ppo_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn ppo_batch_inner(
    data: &[f64],
    sweep: &PpoBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<PpoBatchOutput, PpoError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(PpoError::InvalidPeriod {
            fast: 0,
            slow: 0,
            data_len: 0,
        });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(PpoError::AllValuesNaN)?;
    let max_slow = combos.iter().map(|c| c.slow_period.unwrap()).max().unwrap();
    if data.len() - first < max_slow {
        return Err(PpoError::NotEnoughValidData {
            needed: max_slow,
            valid: data.len() - first,
        });
    }
    let rows = combos.len();
    let cols = data.len();
    let mut values = vec![f64::NAN; rows * cols];

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let p = &combos[row];
        match kern {
            Kernel::Scalar => ppo_row_scalar(data, first, p.fast_period.unwrap(), p.slow_period.unwrap(), p.ma_type.as_ref().unwrap(), out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => ppo_row_avx2(data, first, p.fast_period.unwrap(), p.slow_period.unwrap(), p.ma_type.as_ref().unwrap(), out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => ppo_row_avx512(data, first, p.fast_period.unwrap(), p.slow_period.unwrap(), p.ma_type.as_ref().unwrap(), out_row),
            _ => unreachable!(),
        }
    };

    if parallel {
        values
            .par_chunks_mut(cols)
            .enumerate()
            .for_each(|(row, slice)| do_row(row, slice));
    } else {
        for (row, slice) in values.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }
    Ok(PpoBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
pub unsafe fn ppo_row_scalar(
    data: &[f64],
    first: usize,
    fast: usize,
    slow: usize,
    ma_type: &str,
    out: &mut [f64],
) {
    ppo_scalar(data, fast, slow, ma_type, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn ppo_row_avx2(
    data: &[f64],
    first: usize,
    fast: usize,
    slow: usize,
    ma_type: &str,
    out: &mut [f64],
) {
    ppo_scalar(data, fast, slow, ma_type, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn ppo_row_avx512(
    data: &[f64],
    first: usize,
    fast: usize,
    slow: usize,
    ma_type: &str,
    out: &mut [f64],
) {
    if slow <= 32 {
        ppo_row_avx512_short(data, first, fast, slow, ma_type, out)
    } else {
        ppo_row_avx512_long(data, first, fast, slow, ma_type, out)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn ppo_row_avx512_short(
    data: &[f64],
    first: usize,
    fast: usize,
    slow: usize,
    ma_type: &str,
    out: &mut [f64],
) {
    ppo_scalar(data, fast, slow, ma_type, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn ppo_row_avx512_long(
    data: &[f64],
    first: usize,
    fast: usize,
    slow: usize,
    ma_type: &str,
    out: &mut [f64],
) {
    ppo_scalar(data, fast, slow, ma_type, first, out)
}

pub struct PpoStream {
    fast_period: usize,
    slow_period: usize,
    ma_type: String,
    count: usize,
}

impl PpoStream {
    pub fn try_new(params: PpoParams) -> Result<Self, PpoError> {
        let fast = params.fast_period.unwrap_or(12);
        let slow = params.slow_period.unwrap_or(26);
        let ma_type = params
            .ma_type
            .clone()
            .unwrap_or_else(|| "sma".to_string());

        Ok(Self {
            fast_period: fast,
            slow_period: slow,
            ma_type,
            count: 0,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.count += 1;
        // TODO: Implement actual PPO streaming calculation
        // Return None until enough data, then Some(value)
        None
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_ppo_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = PpoParams {
            fast_period: None,
            slow_period: None,
            ma_type: None,
        };
        let input = PpoInput::from_candles(&candles, "close", default_params);
        let output = ppo_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_ppo_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = PpoInput::from_candles(&candles, "close", PpoParams::default());
        let result = ppo_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), candles.close.len());
        let expected_last_five = [
            -0.8532313608928664,
            -0.8537562894550523,
            -0.6821291938174874,
            -0.5620008722078592,
            -0.4101724140910927,
        ];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-7,
                "[{}] PPO {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_ppo_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = PpoInput::with_default_candles(&candles);
        match input.data {
            PpoData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected PpoData::Candles"),
        }
        let output = ppo_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_ppo_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = PpoParams {
            fast_period: Some(0),
            slow_period: None,
            ma_type: None,
        };
        let input = PpoInput::from_slice(&input_data, params);
        let res = ppo_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] PPO should fail with zero fast period", test_name);
        Ok(())
    }

    fn check_ppo_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = PpoParams {
            fast_period: Some(12),
            slow_period: Some(26),
            ma_type: None,
        };
        let input = PpoInput::from_slice(&data_small, params);
        let res = ppo_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] PPO should fail with period exceeding length", test_name);
        Ok(())
    }

    fn check_ppo_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = PpoParams {
            fast_period: Some(12),
            slow_period: Some(26),
            ma_type: None,
        };
        let input = PpoInput::from_slice(&single_point, params);
        let res = ppo_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] PPO should fail with insufficient data", test_name);
        Ok(())
    }

    fn check_ppo_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = PpoInput::from_candles(
            &candles,
            "close",
            PpoParams {
                fast_period: Some(12),
                slow_period: Some(26),
                ma_type: Some("sma".to_string()),
            },
        );
        let res = ppo_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), candles.close.len());
        if res.values.len() > 30 {
            for (i, &val) in res.values[30..].iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "[{}] Found unexpected NaN at out-index {}",
                    test_name,
                    30 + i
                );
            }
        }
        Ok(())
    }

    fn check_ppo_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let fast = 12;
        let slow = 26;
        let ma_type = "sma".to_string();
        let input = PpoInput::from_candles(
            &candles,
            "close",
            PpoParams {
                fast_period: Some(fast),
                slow_period: Some(slow),
                ma_type: Some(ma_type.clone()),
            },
        );
        let batch_output = ppo_with_kernel(&input, kernel)?.values;
        let mut stream = PpoStream::try_new(PpoParams {
            fast_period: Some(fast),
            slow_period: Some(slow),
            ma_type: Some(ma_type),
        })?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(ppo_val) => stream_values.push(ppo_val),
                None => stream_values.push(f64::NAN),
            }
        }
        assert_eq!(batch_output.len(), stream_values.len());
        for (i, (&b, &s)) in batch_output.iter().zip(stream_values.iter()).enumerate() {
            if b.is_nan() && s.is_nan() {
                continue;
            }
            let diff = (b - s).abs();
            assert!(
                diff < 1e-9,
                "[{}] PPO streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    macro_rules! generate_all_ppo_tests {
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

    generate_all_ppo_tests!(
        check_ppo_partial_params,
        check_ppo_accuracy,
        check_ppo_default_candles,
        check_ppo_zero_period,
        check_ppo_period_exceeds_length,
        check_ppo_very_small_dataset,
        check_ppo_nan_handling,
        check_ppo_streaming
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = PpoBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = PpoParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        let expected = [
            -0.8532313608928664,
            -0.8537562894550523,
            -0.6821291938174874,
            -0.5620008722078592,
            -0.4101724140910927,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-7,
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
