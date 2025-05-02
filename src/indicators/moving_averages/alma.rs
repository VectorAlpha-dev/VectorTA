//! # Arnaud Legoux Moving Average (ALMA)
//!
//! A smooth yet responsive moving average that uses Gaussian weighting. Its parameters
//! (`period`, `offset`, `sigma`) control the window size, the weighting center, and
//! the Gaussian smoothness. ALMA can also be re-applied to its own output, allowing
//! iterative smoothing on previously computed results.
//!
//! ## Parameters
//! - **period**: Window size (number of data points).
//! - **offset**: Shift in [0.0, 1.0] for the Gaussian center (defaults to 0.85).
//! - **sigma**: Controls the Gaussian curve’s width (defaults to 6.0).
//!
//! ## Errors
//! - **AllValuesNaN**: alma: All input data values are `NaN`.
//! - **InvalidPeriod**: alma: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: alma: Not enough valid data points for the requested `period`.
//! - **InvalidSigma**: alma: `sigma` ≤ 0.0.
//! - **InvalidOffset**: alma: `offset` is `NaN` or infinite.
//!
//! ## Returns
//! - **`Ok(AlmaOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(AlmaError)`** otherwise.
//!
use crate::utilities::data_loader::{source_type, Candles};
use core::arch::x86_64::*;
use std::error::Error;
use crate::utilities::enums::Kernel;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum AlmaData<'a> {
    Candles { candles: &'a Candles, source: &'a str },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct AlmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct AlmaParams {
    pub period: Option<usize>,
    pub offset: Option<f64>,
    pub sigma : Option<f64>,
}

impl Default for AlmaParams {
    fn default() -> Self {
        Self {
            period: Some(9),
            offset: Some(0.85),
            sigma:  Some(6.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AlmaInput<'a> {
    pub data:   AlmaData<'a>,
    pub params: AlmaParams,
}

impl<'a> AlmaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: AlmaParams) -> Self {
        Self {
            data: AlmaData::Candles { candles: c, source: s },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: AlmaParams) -> Self {
        Self {
            data: AlmaData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", AlmaParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(9)
    }
    #[inline]
    pub fn get_offset(&self) -> f64 {
        self.params.offset.unwrap_or(0.85)
    }
    #[inline]
    pub fn get_sigma(&self) -> f64 {
        self.params.sigma.unwrap_or(6.0)
    }
}


#[derive(Copy, Clone, Debug)]
pub struct AlmaBuilder {
    period: Option<usize>,
    offset: Option<f64>,
    sigma : Option<f64>,
    kernel: Kernel,
}

impl Default for AlmaBuilder {
    fn default() -> Self {
        Self {
            period: None,
            offset: None,
            sigma: None,
            kernel: Kernel::Auto,
        }
    }
}

impl AlmaBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn period(mut self, n: usize) -> Self {
        self.period = Some(n);
        self
    }
    #[inline(always)]
    pub fn offset(mut self, x: f64) -> Self {
        self.offset = Some(x);
        self
    }
    #[inline(always)]
    pub fn sigma(mut self, s: f64) -> Self {
        self.sigma = Some(s);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<AlmaOutput, AlmaError> {
        let p = AlmaParams {
            period: self.period,
            offset: self.offset,
            sigma:  self.sigma,
        };
        let i = AlmaInput::from_candles(c, "close", p);
        alma_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<AlmaOutput, AlmaError> {
        let p = AlmaParams {
            period: self.period,
            offset: self.offset,
            sigma:  self.sigma,
        };
        let i = AlmaInput::from_slice(d, p);
        alma_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<AlmaStream, AlmaError> {
        let p = AlmaParams {
            period: self.period,
            offset: self.offset,
            sigma:  self.sigma,
        };
        AlmaStream::try_new(p)
    }
}


#[derive(Debug, Error)]
pub enum AlmaError {
    #[error("alma: All values are NaN.")]
    AllValuesNaN,

    #[error("alma: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },

    #[error("alma: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },

    #[error("alma: Invalid sigma: {sigma}")]
    InvalidSigma { sigma: f64 },

    #[error("alma: Invalid offset: {offset}")]
    InvalidOffset { offset: f64 },
}



#[inline]
pub fn alma(input: &AlmaInput) -> Result<AlmaOutput, AlmaError> {
    alma_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn detect_best_kernel() -> Kernel {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("fma") {
            return Kernel::Avx512;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return Kernel::Avx2;
        }   
    }
    Kernel::Scalar
}



pub fn alma_with_kernel(input: &AlmaInput, kernel: Kernel) -> Result<AlmaOutput, AlmaError> {
    let data: &[f64] = match &input.data {
        AlmaData::Candles { candles, source } => source_type(candles, source),
        AlmaData::Slice(sl) => sl,
    };

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(AlmaError::AllValuesNaN)?;

    let len = data.len();
    let period = input.get_period();
    let offset = input.get_offset();
    let sigma  = input.get_sigma();

    if period == 0 || period > len {
        return Err(AlmaError::InvalidPeriod { period, data_len: len });
    }
    if (len - first) < period {
        return Err(AlmaError::NotEnoughValidData {
            needed: period,
            valid:  len - first,
        });
    }
    if sigma <= 0.0 {
        return Err(AlmaError::InvalidSigma { sigma });
    }
    if !(0.0..=1.0).contains(&offset) || offset.is_nan() || offset.is_infinite() {
           return Err(AlmaError::InvalidOffset { offset });
       }

    let m   = offset * (period - 1) as f64;
    let s   = period as f64 / sigma;
    let s2  = 2.0 * s * s;

    let mut weights = Vec::with_capacity(period);
    let mut norm    = 0.0;
    for i in 0..period {
        let w = (-(i as f64 - m).powi(2) / s2).exp();
        weights.push(w);
        norm += w;
    }
    let inv_norm = 1.0 / norm;

    let mut out = vec![f64::NAN; len];

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other        => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar => alma_scalar(data, &weights, period, first, inv_norm, &mut out),
            Kernel::Avx2   => alma_avx2(data, &weights, period, first, inv_norm, &mut out),
            Kernel::Avx512 => alma_avx512(data, &weights, period, first, inv_norm, &mut out),
            Kernel::Auto => unreachable!(),
        }
    }

    Ok(AlmaOutput { values: out })
}

#[inline(always)]
pub fn alma_scalar(
    data:      &[f64],
    weights:   &[f64],
    period:    usize,
    first_val: usize,
    inv_norm:  f64,
    out:       &mut [f64],
) {
    assert_eq!(weights.len(), period, "weights.len() must equal `period`");
    assert!(
        out.len() >= data.len(),
        "`out` must be at least as long as `data`"
    );

    let p4 = period & !3;

    for i in (first_val + period - 1)..data.len() {
        let start  = i + 1 - period;
        let window = &data[start .. start + period];

        let mut sum = 0.0;
        for (d4, w4) in window[..p4]
            .chunks_exact(4)
            .zip(weights[..p4].chunks_exact(4))
        {
            sum += d4[0] * w4[0]
                 + d4[1] * w4[1]
                 + d4[2] * w4[2]
                 + d4[3] * w4[3];
        }

        for (d, w) in window[p4..].iter().zip(&weights[p4..]) {
            sum += d * w;
        }

        out[i] = sum * inv_norm;
    }
}


#[target_feature(enable = "avx2,fma")]
unsafe fn alma_avx2(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first_valid: usize,
    inv_norm: f64,
    out: &mut [f64],
) {

    const STEP: usize = 4;
    let chunks = period / STEP;
    let tail   = period % STEP;


    let tail_mask = match tail {
        0 => _mm256_setzero_si256(),
        1 => _mm256_setr_epi64x(-1, 0, 0, 0),
        2 => _mm256_setr_epi64x(-1, -1, 0, 0),
        3 => _mm256_setr_epi64x(-1, -1, -1, 0),
        _ => unreachable!(),
    };

    for i in (first_valid + period - 1)..data.len() {
        let start = i + 1 - period;
        let mut acc = _mm256_setzero_pd();


        for blk in 0..chunks {
            let idx = blk * STEP;
            let w   = _mm256_loadu_pd(weights.as_ptr().add(idx));
            let d   = _mm256_loadu_pd(data.as_ptr().add(start + idx));
            acc     = _mm256_fmadd_pd(d, w, acc);
        }


        if tail != 0 {
            let w_tail = _mm256_maskload_pd(
                weights.as_ptr().add(chunks * STEP),
                tail_mask,
            );
            let d_tail = _mm256_maskload_pd(
                data.as_ptr().add(start + chunks * STEP),
                tail_mask,
            );
            acc = _mm256_fmadd_pd(d_tail, w_tail, acc);
        }


        let hi  = _mm256_extractf128_pd(acc, 1);
        let lo  = _mm256_castpd256_pd128(acc);
        let sum2 = _mm_add_pd(hi, lo);
        let sum1 = _mm_add_pd(sum2, _mm_unpackhi_pd(sum2, sum2));
        let sum  = _mm_cvtsd_f64(sum1);

        *out.get_unchecked_mut(i) = sum * inv_norm;
    }
}

#[target_feature(enable = "avx512f,fma")]
unsafe fn alma_avx512(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first_valid: usize,
    inv_norm: f64,
    out: &mut [f64],
) {

    const STEP: usize = 8;
    let chunks = period / STEP;
    let tail   = period % STEP;

    let tail_mask: __mmask8 = (1u8 << tail).wrapping_sub(1);

    for i in (first_valid + period - 1)..data.len() {
        let start = i + 1 - period;

        let mut acc0 = _mm512_setzero_pd();
        let mut acc1 = _mm512_setzero_pd();

        let paired = chunks & !1;
        for blk in (0..paired).step_by(2) {
            let idx0 = blk * STEP;
            let idx1 = idx0 + STEP;

            let w0 = _mm512_loadu_pd(weights.as_ptr().add(idx0));
            let w1 = _mm512_loadu_pd(weights.as_ptr().add(idx1));
            let d0 = _mm512_loadu_pd(data.as_ptr().add(start + idx0));
            let d1 = _mm512_loadu_pd(data.as_ptr().add(start + idx1));

            acc0 = _mm512_fmadd_pd(d0, w0, acc0);
            acc1 = _mm512_fmadd_pd(d1, w1, acc1);
        }

        if chunks & 1 != 0 {
            let idx = (chunks - 1) * STEP;
            let w   = _mm512_loadu_pd(weights.as_ptr().add(idx));
            let d   = _mm512_loadu_pd(data.as_ptr().add(start + idx));
            acc0    = _mm512_fmadd_pd(d, w, acc0);
        }

        if tail != 0 {
            let w_tail = _mm512_maskz_loadu_pd(tail_mask, weights.as_ptr().add(chunks * STEP));
            let d_tail = _mm512_maskz_loadu_pd(tail_mask, data.as_ptr().add(start + chunks * STEP));
            acc0       = _mm512_fmadd_pd(d_tail, w_tail, acc0);
        }

        acc0 = _mm512_add_pd(acc0, acc1);
        let sum = _mm512_reduce_add_pd(acc0);

        *out.get_unchecked_mut(i) = sum * inv_norm;
    }
}

#[derive(Debug, Clone)]
pub struct AlmaStream {
    period:   usize,
    weights:  Vec<f64>,
    inv_norm: f64,
    buffer:   Vec<f64>,
    head:     usize,
    filled:   bool,
}

impl AlmaStream {
    pub fn try_new(params: AlmaParams) -> Result<Self, AlmaError> {
        let period = params.period.unwrap_or(9);
        if period == 0 {
            return Err(AlmaError::InvalidPeriod { period, data_len: 0 });
        }
        let offset = params.offset.unwrap_or(0.85);
        if !(0.0..=1.0).contains(&offset) || offset.is_nan() || offset.is_infinite() {
            return Err(AlmaError::InvalidOffset { offset });
       }
        let sigma = params.sigma.unwrap_or(6.0);
        if sigma <= 0.0 {
            return Err(AlmaError::InvalidSigma { sigma });
        }

        let m  = offset * (period - 1) as f64;
        let s  = period as f64 / sigma;
        let s2 = 2.0 * s * s;

        let mut weights = Vec::with_capacity(period);
        let mut norm    = 0.0;
        for i in 0..period {
            let diff = i as f64 - m;
            let w    = (-(diff * diff) / s2).exp();
            weights.push(w);
            norm += w;
        }
        let inv_norm = 1.0 / norm;

        Ok(Self {
            period,
            weights,
            inv_norm,
            buffer: vec![f64::NAN; period],
            head: 0,
            filled: false,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.buffer[self.head] = value;
        self.head = (self.head + 1) % self.period;

        if !self.filled && self.head == 0 {
            self.filled = true;
        }
        if !self.filled {
            return None;
        }
        Some(self.dot_ring())
    }

    #[inline(always)]
    fn dot_ring(&self) -> f64 {
        let mut sum = 0.0;
        let mut idx = self.head;
        for &w in &self.weights {
            sum += w * self.buffer[idx];
            idx = (idx + 1) % self.period;
        }
        sum * self.inv_norm
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn skip_if_unsupported(kernel: Kernel, test_name: &str) {
        match kernel {
            Kernel::Avx2 => {
                if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
                    eprintln!("[{}] Skipping AVX2 test on non-AVX2 CPU", test_name);
                    std::process::exit(0);
                }
            }
            Kernel::Avx512 => {
                if !(is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("fma")) {
                    eprintln!("[{}] Skipping AVX512 test on non-AVX512 CPU", test_name);
                    std::process::exit(0);
                }
            }
            Kernel::Scalar => {}
            Kernel::Auto => {}
        }
    }

    fn check_alma_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = AlmaParams {
            period: None,
            offset: None,
            sigma:  None,
        };
        let input = AlmaInput::from_candles(&candles, "close", default_params);
        let output = alma_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_alma_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = AlmaInput::from_candles(&candles, "close", AlmaParams::default());
        let result = alma_with_kernel(&input, kernel)?;
        let expected_last_five = [
            59286.72216704,
            59273.53428138,
            59204.37290721,
            59155.93381742,
            59026.92526112,
        ];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-8,
                "[{}] ALMA {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_alma_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = AlmaInput::with_default_candles(&candles);
        match input.data {
            AlmaData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected AlmaData::Candles"),
        }
        let output = alma_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_alma_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = AlmaParams {
            period: Some(0),
            offset: None,
            sigma:  None,
        };
        let input = AlmaInput::from_slice(&input_data, params);
        let res = alma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] ALMA should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_alma_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = AlmaParams {
            period: Some(10),
            offset: None,
            sigma:  None,
        };
        let input = AlmaInput::from_slice(&data_small, params);
        let res = alma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] ALMA should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_alma_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported(kernel, test_name);
        let single_point = [42.0];
        let params = AlmaParams {
            period: Some(9),
            offset: None,
            sigma:  None,
        };
        let input = AlmaInput::from_slice(&single_point, params);
        let res = alma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] ALMA should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_alma_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let first_params = AlmaParams {
            period: Some(9),
            offset: None,
            sigma:  None,
        };
        let first_input = AlmaInput::from_candles(&candles, "close", first_params);
        let first_result = alma_with_kernel(&first_input, kernel)?;

        let second_params = AlmaParams {
            period: Some(9),
            offset: None,
            sigma:  None,
        };
        let second_input = AlmaInput::from_slice(&first_result.values, second_params);
        let second_result = alma_with_kernel(&second_input, kernel)?;

        assert_eq!(second_result.values.len(), first_result.values.len());
        let expected_last_five = [
            59140.73195170,
            59211.58090986,
            59238.16030697,
            59222.63528822,
            59165.14427332,
        ];
        let start = second_result.values.len().saturating_sub(5);
        for (i, &val) in second_result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-8,
                "[{}] ALMA Slice Reinput {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_alma_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = AlmaInput::from_candles(
            &candles,
            "close",
            AlmaParams {
                period: Some(9),
                offset: None,
                sigma: None,
            },
        );
        let res = alma_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), candles.close.len());
        if res.values.len() > 240 {
            for (i, &val) in res.values[240..].iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "[{}] Found unexpected NaN at out-index {}",
                    test_name,
                    240 + i
                );
            }
        }
        Ok(())
    }

    fn check_alma_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let period = 9;
        let offset = 0.85;
        let sigma  = 6.0;

        let input = AlmaInput::from_candles(
            &candles,
            "close",
            AlmaParams {
                period: Some(period),
                offset: Some(offset),
                sigma:  Some(sigma),
            },
        );
        let batch_output = alma_with_kernel(&input, kernel)?.values;

        let mut stream = AlmaStream::try_new(AlmaParams {
            period: Some(period),
            offset: Some(offset),
            sigma:  Some(sigma),
        })?;

        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(alma_val) => stream_values.push(alma_val),
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
                "[{}] ALMA streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    macro_rules! generate_all_alma_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $(
                    #[test]
                    fn [<$test_fn _scalar_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar);
                    }

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

    generate_all_alma_tests!(
        check_alma_partial_params,
        check_alma_accuracy,
        check_alma_default_candles,
        check_alma_zero_period,
        check_alma_period_exceeds_length,
        check_alma_very_small_dataset,
        check_alma_reinput,
        check_alma_nan_handling,
        check_alma_streaming
    );
}
