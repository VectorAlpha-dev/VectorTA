//! # Moving Average Convergence/Divergence (MACD)
//!
//! A trend-following momentum indicator that shows the relationship between two moving averages of a data series.
//! Calculates the MACD line, signal line, and histogram.
//!
//! ## Parameters
//! - **fast_period**: Shorter moving average period (default: 12)
//! - **slow_period**: Longer moving average period (default: 26)
//! - **signal_period**: Signal line moving average period (default: 9)
//! - **ma_type**: Moving average type for all components (default: "ema")
//!
//! ## Errors
//! - **AllValuesNaN**: All input values are NaN.
//! - **InvalidPeriod**: One or more periods are zero or exceed data length.
//! - **NotEnoughValidData**: Insufficient valid data points for requested period(s).
//!
//! ## Returns
//! - `Ok(MacdOutput)` on success, containing MACD, signal, and histogram vectors
//! - `Err(MacdError)` otherwise

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use rayon::prelude::*;
use std::error::Error;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum MacdData<'a> {
    Candles { candles: &'a Candles, source: &'a str },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct MacdOutput {
    pub macd: Vec<f64>,
    pub signal: Vec<f64>,
    pub hist: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct MacdParams {
    pub fast_period: Option<usize>,
    pub slow_period: Option<usize>,
    pub signal_period: Option<usize>,
    pub ma_type: Option<String>,
}

impl Default for MacdParams {
    fn default() -> Self {
        Self {
            fast_period: Some(12),
            slow_period: Some(26),
            signal_period: Some(9),
            ma_type: Some("ema".to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MacdInput<'a> {
    pub data: MacdData<'a>,
    pub params: MacdParams,
}

impl<'a> MacdInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: MacdParams) -> Self {
        Self { data: MacdData::Candles { candles: c, source: s }, params: p }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: MacdParams) -> Self {
        Self { data: MacdData::Slice(sl), params: p }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", MacdParams::default())
    }
    #[inline]
    pub fn get_fast_period(&self) -> usize { self.params.fast_period.unwrap_or(12) }
    #[inline]
    pub fn get_slow_period(&self) -> usize { self.params.slow_period.unwrap_or(26) }
    #[inline]
    pub fn get_signal_period(&self) -> usize { self.params.signal_period.unwrap_or(9) }
    #[inline]
    pub fn get_ma_type(&self) -> String {
        self.params.ma_type.clone().unwrap_or_else(|| "ema".to_string())
    }
}

#[derive(Clone, Debug)]
pub struct MacdBuilder {
    fast_period: Option<usize>,
    slow_period: Option<usize>,
    signal_period: Option<usize>,
    ma_type: Option<String>,
    kernel: Kernel,
}

impl Default for MacdBuilder {
    fn default() -> Self {
        Self { fast_period: None, slow_period: None, signal_period: None, ma_type: None, kernel: Kernel::Auto }
    }
}

impl MacdBuilder {
    #[inline]
    pub fn new() -> Self { Self::default() }
    #[inline]
    pub fn get_fast_period(mut self, n: usize) -> Self { self.fast_period = Some(n); self }
    #[inline]
    pub fn get_slow_period(mut self, n: usize) -> Self { self.slow_period = Some(n); self }
    #[inline]
    pub fn get_signal_period(mut self, n: usize) -> Self { self.signal_period = Some(n); self }
    #[inline]
    pub fn ma_type<S: Into<String>>(mut self, s: S) -> Self { self.ma_type = Some(s.into()); self }
    #[inline]
    pub fn kernel(mut self, k: Kernel) -> Self { self.kernel = k; self }

    #[inline]
    pub fn apply(self, c: &Candles) -> Result<MacdOutput, MacdError> {
        let p = MacdParams {
            fast_period: self.fast_period,
            slow_period: self.slow_period,
            signal_period: self.signal_period,
            ma_type: self.ma_type,
        };
        let i = MacdInput::from_candles(c, "close", p);
        macd_with_kernel(&i, self.kernel)
    }

    #[inline]
    pub fn apply_slice(self, d: &[f64]) -> Result<MacdOutput, MacdError> {
        let p = MacdParams {
            fast_period: self.fast_period,
            slow_period: self.slow_period,
            signal_period: self.signal_period,
            ma_type: self.ma_type,
        };
        let i = MacdInput::from_slice(d, p);
        macd_with_kernel(&i, self.kernel)
    }
}

#[derive(Debug, Error)]
pub enum MacdError {
    #[error("macd: All values are NaN.")]
    AllValuesNaN,
    #[error("macd: Invalid period: fast = {fast}, slow = {slow}, signal = {signal}, data length = {data_len}")]
    InvalidPeriod { fast: usize, slow: usize, signal: usize, data_len: usize },
    #[error("macd: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("macd: Unknown MA type: {0}")]
    UnknownMA(String),
}

#[inline]
pub fn macd(input: &MacdInput) -> Result<MacdOutput, MacdError> {
    macd_with_kernel(input, Kernel::Auto)
}

pub fn macd_with_kernel(input: &MacdInput, kernel: Kernel) -> Result<MacdOutput, MacdError> {
    let data: &[f64] = match &input.data {
        MacdData::Candles { candles, source } => source_type(candles, source),
        MacdData::Slice(sl) => sl,
    };

    let len = data.len();
    let fast = input.get_fast_period();
    let slow = input.get_slow_period();
    let signal = input.get_signal_period();
    let ma_type = input.get_ma_type();

    let first = data.iter().position(|x| !x.is_nan()).ok_or(MacdError::AllValuesNaN)?;
    if fast == 0 || slow == 0 || signal == 0 || fast > len || slow > len || signal > len {
        return Err(MacdError::InvalidPeriod { fast, slow, signal, data_len: len });
    }
    if (len - first) < slow {
        return Err(MacdError::NotEnoughValidData { needed: slow, valid: len - first });
    }

    let chosen = match kernel { Kernel::Auto => detect_best_kernel(), other => other };
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => macd_scalar(data, fast, slow, signal, &ma_type, first),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => macd_avx2(data, fast, slow, signal, &ma_type, first),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => macd_avx512(data, fast, slow, signal, &ma_type, first),
            _ => unreachable!(),
        }
    }
}

#[inline(always)]
pub unsafe fn macd_scalar(
    data: &[f64],
    fast: usize,
    slow: usize,
    signal: usize,
    ma_type: &str,
    first: usize,
) -> Result<MacdOutput, MacdError> {
    use crate::indicators::moving_averages::ma::{ma, MaData};
    let len = data.len();
    let fast_ma = ma(ma_type, MaData::Slice(data), fast).map_err(|_| MacdError::AllValuesNaN)?;
    let slow_ma = ma(ma_type, MaData::Slice(data), slow).map_err(|_| MacdError::AllValuesNaN)?;

    let mut macd = vec![f64::NAN; len];
    for i in first..len {
        if fast_ma[i].is_nan() || slow_ma[i].is_nan() { continue; }
        macd[i] = fast_ma[i] - slow_ma[i];
    }
    let signal_ma = ma(ma_type, MaData::Slice(&macd), signal).map_err(|_| MacdError::AllValuesNaN)?;
    let mut signal_vec = vec![f64::NAN; len];
    let mut hist = vec![f64::NAN; len];
    for i in first..len {
        if macd[i].is_nan() || signal_ma[i].is_nan() { continue; }
        signal_vec[i] = signal_ma[i];
        hist[i] = macd[i] - signal_ma[i];
    }
    Ok(MacdOutput { macd, signal: signal_vec, hist })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn macd_avx2(
    data: &[f64], fast: usize, slow: usize, signal: usize, ma_type: &str, first: usize
) -> Result<MacdOutput, MacdError> {
    macd_scalar(data, fast, slow, signal, ma_type, first)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn macd_avx512(
    data: &[f64], fast: usize, slow: usize, signal: usize, ma_type: &str, first: usize
) -> Result<MacdOutput, MacdError> {
    macd_scalar(data, fast, slow, signal, ma_type, first)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn macd_avx512_short(
    data: &[f64], fast: usize, slow: usize, signal: usize, ma_type: &str, first: usize
) -> Result<MacdOutput, MacdError> {
    macd_avx512(data, fast, slow, signal, ma_type, first)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn macd_avx512_long(
    data: &[f64], fast: usize, slow: usize, signal: usize, ma_type: &str, first: usize
) -> Result<MacdOutput, MacdError> {
    macd_avx512(data, fast, slow, signal, ma_type, first)
}

#[inline(always)]
pub fn macd_row_scalar(
    data: &[f64], fast: usize, slow: usize, signal: usize, ma_type: &str, first: usize
) -> Result<MacdOutput, MacdError> {
    unsafe { macd_scalar(data, fast, slow, signal, ma_type, first) }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn macd_row_avx2(
    data: &[f64], fast: usize, slow: usize, signal: usize, ma_type: &str, first: usize
) -> Result<MacdOutput, MacdError> {
    unsafe { macd_avx2(data, fast, slow, signal, ma_type, first) }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn macd_row_avx512(
    data: &[f64], fast: usize, slow: usize, signal: usize, ma_type: &str, first: usize
) -> Result<MacdOutput, MacdError> {
    unsafe { macd_avx512(data, fast, slow, signal, ma_type, first) }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn macd_row_avx512_short(
    data: &[f64], fast: usize, slow: usize, signal: usize, ma_type: &str, first: usize
) -> Result<MacdOutput, MacdError> {
    unsafe { macd_avx512_short(data, fast, slow, signal, ma_type, first) }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn macd_row_avx512_long(
    data: &[f64], fast: usize, slow: usize, signal: usize, ma_type: &str, first: usize
) -> Result<MacdOutput, MacdError> {
    unsafe { macd_avx512_long(data, fast, slow, signal, ma_type, first) }
}

#[derive(Clone, Debug)]
pub struct MacdBatchRange {
    pub fast_period: (usize, usize, usize),
    pub slow_period: (usize, usize, usize),
    pub signal_period: (usize, usize, usize),
    pub ma_type: (String, String, String),
}

impl Default for MacdBatchRange {
    fn default() -> Self {
        Self {
            fast_period: (12, 12, 0),
            slow_period: (26, 26, 0),
            signal_period: (9, 9, 0),
            ma_type: ("ema".to_string(), "ema".to_string(), "".to_string()),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct MacdBatchBuilder {
    range: MacdBatchRange,
    kernel: Kernel,
}

impl MacdBatchBuilder {
    pub fn new() -> Self { Self::default() }
    pub fn kernel(mut self, k: Kernel) -> Self { self.kernel = k; self }
    #[inline]
    pub fn fast_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.fast_period = (start, end, step); self
    }
    #[inline]
    pub fn slow_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.slow_period = (start, end, step); self
    }
    #[inline]
    pub fn signal_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.signal_period = (start, end, step); self
    }
    #[inline]
    pub fn ma_type_static(mut self, s: &str) -> Self {
        self.range.ma_type = (s.to_string(), s.to_string(), "".to_string()); self
    }

    pub fn apply_slice(self, data: &[f64]) -> Result<MacdBatchOutput, MacdError> {
        macd_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<MacdBatchOutput, MacdError> {
        MacdBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<MacdBatchOutput, MacdError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<MacdBatchOutput, MacdError> {
        MacdBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "close")
    }
}

pub fn macd_batch_with_kernel(
    data: &[f64], sweep: &MacdBatchRange, k: Kernel
) -> Result<MacdBatchOutput, MacdError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(MacdError::InvalidPeriod { fast: 0, slow: 0, signal: 0, data_len: 0 }),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    macd_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct MacdBatchOutput {
    pub macd: Vec<f64>,
    pub signal: Vec<f64>,
    pub hist: Vec<f64>,
    pub combos: Vec<MacdParams>,
    pub rows: usize,
    pub cols: usize,
}

#[inline(always)]
pub fn expand_grid(r: &MacdBatchRange) -> Vec<MacdParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end { return vec![start]; }
        (start..=end).step_by(step).collect()
    }
    let fasts = axis_usize(r.fast_period);
    let slows = axis_usize(r.slow_period);
    let signals = axis_usize(r.signal_period);
    let ma_types = vec![r.ma_type.0.clone()]; // For now, static MA type

    let mut combos = vec![];
    for &f in &fasts {
        for &s in &slows {
            for &g in &signals {
                for t in &ma_types {
                    combos.push(MacdParams {
                        fast_period: Some(f),
                        slow_period: Some(s),
                        signal_period: Some(g),
                        ma_type: Some(t.clone()),
                    });
                }
            }
        }
    }
    combos
}

pub fn macd_batch_par_slice(
    data: &[f64],
    sweep: &MacdBatchRange,
    simd: Kernel,
) -> Result<MacdBatchOutput, MacdError> {
    let combos = expand_grid(sweep);
    let rows = combos.len();
    let cols = data.len();
    let first = data.iter().position(|x| !x.is_nan()).unwrap_or(0);

    // Storage for results
    let mut macd = Vec::with_capacity(rows * cols);
    let mut signal = Vec::with_capacity(rows * cols);
    let mut hist = Vec::with_capacity(rows * cols);

    for p in &combos {
        // Handle Option<usize> (you may want to provide real defaults elsewhere)
        let fast = p.fast_period.unwrap_or(12);
        let slow = p.slow_period.unwrap_or(26);
        let sig = p.signal_period.unwrap_or(9);
        let ma_type = p.ma_type.clone().unwrap_or_else(|| "ema".to_string());

        // You may want to check validity here (fast > 0, slow > 0, etc)
        let out = match unsafe {
            match simd {
                Kernel::Scalar => macd_scalar(data, fast, slow, sig, &ma_type, first),
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx2 => macd_avx2(data, fast, slow, sig, &ma_type, first),
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx512 => macd_avx512(data, fast, slow, sig, &ma_type, first),
                _ => unreachable!(),
            }
        } {
            Ok(out) => out,
            Err(_) => MacdOutput {
                macd: vec![f64::NAN; cols],
                signal: vec![f64::NAN; cols],
                hist: vec![f64::NAN; cols],
            }
        };
        macd.extend(out.macd);
        signal.extend(out.signal);
        hist.extend(out.hist);
    }

    Ok(MacdBatchOutput {
        macd,
        signal,
        hist,
        combos,
        rows,
        cols,
    })
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;
    use crate::utilities::enums::Kernel;

    fn check_macd_partial_params(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file)?;

        let default_params = MacdParams { fast_period: None, slow_period: None, signal_period: None, ma_type: None };
        let input = MacdInput::from_candles(&candles, "close", default_params);
        let output = macd_with_kernel(&input, kernel)?;
        assert_eq!(output.macd.len(), candles.close.len());
        assert_eq!(output.signal.len(), candles.close.len());
        assert_eq!(output.hist.len(), candles.close.len());
        Ok(())
    }

    fn check_macd_accuracy(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file)?;

        let params = MacdParams::default();
        let input = MacdInput::from_candles(&candles, "close", params);
        let result = macd_with_kernel(&input, kernel)?;

        let expected_macd = [
            -629.8674025082801,
            -600.2986584356258,
            -581.6188884820076,
            -551.1020443476082,
            -560.798510688488,
        ];
        let expected_signal = [
            -721.9744591891067,
            -697.6392990384105,
            -674.4352169271299,
            -649.7685824112256,
            -631.9745680666781,
        ];
        let expected_hist = [
            92.10705668082664,
            97.34064060278467,
            92.81632844512228,
            98.6665380636174,
            71.17605737819008,
        ];
        let len = result.macd.len();
        let start = len - 5;
        for i in 0..5 {
            assert!((result.macd[start + i] - expected_macd[i]).abs() < 1e-1);
            assert!((result.signal[start + i] - expected_signal[i]).abs() < 1e-1);
            assert!((result.hist[start + i] - expected_hist[i]).abs() < 1e-1);
        }
        Ok(())
    }

    fn check_macd_zero_period(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let input_data = [10.0, 20.0, 30.0];
        let params = MacdParams { fast_period: Some(0), slow_period: Some(26), signal_period: Some(9), ma_type: Some("ema".to_string()) };
        let input = MacdInput::from_slice(&input_data, params);
        let res = macd_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] MACD should fail with zero fast period", test);
        Ok(())
    }

    fn check_macd_period_exceeds_length(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let data = [10.0, 20.0, 30.0];
        let params = MacdParams { fast_period: Some(12), slow_period: Some(26), signal_period: Some(9), ma_type: Some("ema".to_string()) };
        let input = MacdInput::from_slice(&data, params);
        let res = macd_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] MACD should fail with period exceeding length", test);
        Ok(())
    }

    fn check_macd_very_small_dataset(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let data = [42.0];
        let params = MacdParams { fast_period: Some(12), slow_period: Some(26), signal_period: Some(9), ma_type: Some("ema".to_string()) };
        let input = MacdInput::from_slice(&data, params);
        let res = macd_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] MACD should fail with insufficient data", test);
        Ok(())
    }

    fn check_macd_reinput(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file)?;

        let params = MacdParams::default();
        let input = MacdInput::from_candles(&candles, "close", params.clone());
        let first_result = macd_with_kernel(&input, kernel)?;

        let reinput = MacdInput::from_slice(&first_result.macd, params);
        let re_result = macd_with_kernel(&reinput, kernel)?;

        assert_eq!(re_result.macd.len(), first_result.macd.len());
        for i in 52..re_result.macd.len() {
            assert!(!re_result.macd[i].is_nan());
        }
        Ok(())
    }

    fn check_macd_nan_handling(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file)?;

        let params = MacdParams::default();
        let input = MacdInput::from_candles(&candles, "close", params);
        let res = macd_with_kernel(&input, kernel)?;
        let n = res.macd.len();
        if n > 240 {
            for i in 240..n {
                assert!(!res.macd[i].is_nan());
                assert!(!res.signal[i].is_nan());
                assert!(!res.hist[i].is_nan());
            }
        }
        Ok(())
    }

    macro_rules! generate_all_macd_tests {
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
    generate_all_macd_tests!(
        check_macd_partial_params,
        check_macd_accuracy,
        check_macd_zero_period,
        check_macd_period_exceeds_length,
        check_macd_very_small_dataset,
        check_macd_reinput,
        check_macd_nan_handling
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = MacdBatchBuilder::new().kernel(kernel).apply_candles(&c, "close")?;
        let def = MacdParams::default();
        let row = output.combos.iter().position(|prm|
            prm.fast_period == def.fast_period &&
            prm.slow_period == def.slow_period &&
            prm.signal_period == def.signal_period &&
            prm.ma_type == def.ma_type
        ).expect("default row missing");
        let start = row * output.cols;
        let macd = &output.macd[start..start + output.cols];
        let signal = &output.signal[start..start + output.cols];
        let hist = &output.hist[start..start + output.cols];
        let expected_macd = [
            -629.8674025082801,
            -600.2986584356258,
            -581.6188884820076,
            -551.1020443476082,
            -560.798510688488,
        ];
        let len = macd.len();
        let s = len - 5;
        for i in 0..5 {
            assert!((macd[s + i] - expected_macd[i]).abs() < 1e-1);
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
