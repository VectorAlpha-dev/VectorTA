//! # Cubic Weighted Moving Average (CWMA)
//!
//! A polynomial-weighted moving average that emphasises recent observations by
//! applying a **cubic** weight curve.  The newest sample receives the weight
//! `period³`, the second-newest `(period-1)³`, …, while the oldest sample in
//! the window is weighted `1³`.
//!
//! ## Parameters
//! * **period** – window length in data-points (defaults to 14).
//!
//! ## Errors
//! * **AllValuesNaN**      – every element in the input slice is `NaN`.
//! * **InvalidPeriod**     – `period == 0` or `period > data.len()`.
//! * **NotEnoughValidData**– fewer than *period* non-`NaN` values exist after
//!   the first valid entry.
//!
//! ## Returns
//! * **`Ok(CwmaOutput)`** on success (vector length == input length, leading
//!   values are `NaN` until the window is filled).
//! * **`Err(CwmaError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::detect_best_kernel;

use core::arch::x86_64::*;
use thiserror::Error;
use std::convert::AsRef;

impl<'a> AsRef<[f64]> for CwmaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            CwmaData::Slice(slice) => slice,
            CwmaData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum CwmaData<'a> {
    Candles { candles: &'a Candles, source: &'a str },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct CwmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct CwmaParams {
    pub period: Option<usize>,
}
impl Default for CwmaParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct CwmaInput<'a> {
    pub data:   CwmaData<'a>,
    pub params: CwmaParams,
}

impl<'a> CwmaInput<'a> {
    #[inline(always)] pub fn from_candles(c: &'a Candles, s: &'a str, p: CwmaParams) -> Self {
        Self { data: CwmaData::Candles { candles: c, source: s }, params: p }
    }
    #[inline(always)] pub fn from_slice(sl: &'a [f64], p: CwmaParams) -> Self {
        Self { data: CwmaData::Slice(sl), params: p }
    }
    #[inline(always)] pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", CwmaParams::default())
    }

    #[inline(always)] pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(14)
    }
}


#[derive(Copy, Clone, Debug)]
pub struct CwmaBuilder {
    period: Option<usize>,
    kernel: Kernel,
}
impl Default for CwmaBuilder {
        fn default() -> Self {
            Self {
                period: None,
                kernel: Kernel::Auto,
            }
        }
    }
    
    impl CwmaBuilder {
    #[inline(always)] pub fn new() -> Self { Self::default() }
    #[inline(always)] pub fn period(mut self, n: usize) -> Self { self.period = Some(n); self }
    #[inline(always)] pub fn kernel(mut self, k: Kernel) -> Self { self.kernel = k; self }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<CwmaOutput, CwmaError> {
        let p = CwmaParams { period: self.period };
        let i = CwmaInput::from_candles(c, "close", p);
        cwma_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<CwmaOutput, CwmaError> {
        let p = CwmaParams { period: self.period };
        let i = CwmaInput::from_slice(d, p);
        cwma_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<CwmaStream, CwmaError> {
        let p = CwmaParams { period: self.period };
        CwmaStream::try_new(p)
    }
}


#[derive(Debug, Error)]
pub enum CwmaError {
    #[error("cwma: All values are NaN.")]
    AllValuesNaN,
    #[error("cwma: Invalid period specified for CWMA calculation: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("cwma: Not enough valid data points to compute CWMA: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}


#[inline(always)] pub fn cwma(input: &CwmaInput) -> Result<CwmaOutput, CwmaError> {
    cwma_with_kernel(input, Kernel::Auto)
}

pub fn cwma_with_kernel(input: &CwmaInput, kernel: Kernel) -> Result<CwmaOutput, CwmaError> {
    let data: &[f64] = match &input.data {
        CwmaData::Candles { candles, source } => source_type(candles, source),
        CwmaData::Slice(sl)                   => sl,
    };
    let first = data.iter().position(|x| !x.is_nan())
        .ok_or(CwmaError::AllValuesNaN)?;

    let len    = data.len();
    let period = input.get_period();

    if period == 0 || period > len {
        return Err(CwmaError::InvalidPeriod { period, data_len: len });
    }
    if (len - first) < period {
        return Err(CwmaError::NotEnoughValidData {
            needed: period, valid: len - first });
    }

    let mut weights = Vec::with_capacity(period - 1);
    let mut norm    = 0.0;
    for i in 0..period - 1 {
        let w = ((period - i) as f64).powi(3);
        weights.push(w);
        norm += w;
    }
    let inv_norm = 1.0 / norm;

    let mut out = vec![f64::NAN; len];

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k            => k,
    };

unsafe {
    match chosen {
        Kernel::Scalar  |
        Kernel::ScalarBatch => cwma_scalar (data, &weights, period, first, inv_norm, &mut out),

        Kernel::Avx2    |
        Kernel::Avx2Batch   => cwma_avx2   (data, &weights, period, first, inv_norm, &mut out),

        Kernel::Avx512  |
        Kernel::Avx512Batch => cwma_avx512 (data, &weights, period, first, inv_norm, &mut out),

        Kernel::Auto => unreachable!(),
    }
}

    Ok(CwmaOutput { values: out })
}


#[inline(always)]
fn cwma_scalar(
    data:      &[f64],
    weights:   &[f64],
    _period:   usize,
    first_val: usize,
    inv_norm:  f64,
    out:       &mut [f64],
) {
    let wlen = weights.len();

    for i in (first_val + wlen + 1)..data.len() {

        let mut acc = 0.0;
        for (k, &w) in weights.iter().enumerate() {
            acc = data[i - k].mul_add(w, acc);
        }
        out[i] = acc * inv_norm;
    }
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn cwma_avx2(
    data:        &[f64],
    weights:     &[f64],
    _period:     usize,
    first_valid: usize,
    inv_norm:    f64,
    out:         &mut [f64],
) {
    const STEP: usize = 4;

    let wlen   = weights.len();
    let chunks = wlen / STEP;
    let tail   = wlen % STEP;
    let first_out = first_valid + wlen + 1;

    for i in first_out..data.len() {
        let mut acc = _mm256_setzero_pd();

        for blk in 0..chunks {
            let idx = blk * STEP;
            let w   = _mm256_loadu_pd(weights.as_ptr().add(idx));

            let base = i - idx - (STEP - 1);
            let mut d = _mm256_loadu_pd(data.as_ptr().add(base));
            d = _mm256_permute4x64_pd(d, 0b00011011); 

            acc = _mm256_fmadd_pd(d, w, acc);
        }

        let mut tail_sum = 0.0;
        if tail != 0 {
            let base = chunks * STEP;
            for k in 0..tail {
                let w = *weights.get_unchecked(base + k);
                let d = *data   .get_unchecked(i - (base + k));
                tail_sum = d.mul_add(w, tail_sum);
            }
        }

        let hi   = _mm256_extractf128_pd(acc, 1);
        let lo   = _mm256_castpd256_pd128(acc);
        let sum2 = _mm_add_pd(hi, lo);
        let sum1 = _mm_add_pd(sum2, _mm_unpackhi_pd(sum2, sum2));
        let mut sum = _mm_cvtsd_f64(sum1);

        sum += tail_sum;
        *out.get_unchecked_mut(i) = sum * inv_norm;
    }
}
#[target_feature(enable = "avx512f,fma")]
fn make_weight_blocks(weights: &[f64]) -> Vec<core::arch::x86_64::__m512d> {
    use core::arch::x86_64::{_mm512_loadu_pd, _mm512_permutexvar_pd, _mm512_set_epi64};
    const STEP: usize = 8;
    let lanes = _mm512_set_epi64(0,1,2,3,4,5,6,7);
    weights.chunks_exact(STEP)
           .map(|chunk| unsafe {
               let v   = _mm512_loadu_pd(chunk.as_ptr());
               _mm512_permutexvar_pd(lanes, v)
           })
           .collect()
}

#[target_feature(enable = "avx512f,fma")]
pub unsafe fn cwma_avx512(
    data:        &[f64],
    weights:     &[f64],
    period:      usize,
    first_valid: usize,
    inv_norm:    f64,
    out:         &mut [f64],
) {
    if weights.len() < 24 {
        return cwma_avx2(data, weights, period, first_valid, inv_norm, out);
    }

    const STEP: usize = 8;
    let wlen        = weights.len();
    let chunks      = wlen / STEP;
    let tail        = wlen % STEP;
    let first_out   = first_valid + wlen + 1;

    let wblocks = make_weight_blocks(weights);

    for i in first_out .. data.len() {
        let mut acc0 = _mm512_setzero_pd();
        let mut acc1 = _mm512_setzero_pd();

        let paired = chunks & !1;
        let mut blk = 0;
        while blk < paired {
            let d0 = _mm512_loadu_pd(data.as_ptr().add(i - blk*STEP       - 7));
            let d1 = _mm512_loadu_pd(data.as_ptr().add(i - (blk+1)*STEP   - 7));

            acc0 = _mm512_fmadd_pd(d0, *wblocks.get_unchecked(blk    ), acc0);
            acc1 = _mm512_fmadd_pd(d1, *wblocks.get_unchecked(blk + 1), acc1);

            blk += 2;
        }

        if blk < chunks {
            let d = _mm512_loadu_pd(data.as_ptr().add(i - blk*STEP - 7));
            acc0  = _mm512_fmadd_pd(d, *wblocks.get_unchecked(blk), acc0);
        }

        let mut tail_sum = 0.0;
        if tail != 0 {
            let base = chunks * STEP;
            for k in 0..tail {
                let w = *weights.get_unchecked(base + k);
                let d = *data   .get_unchecked(i - (base + k));
                tail_sum = d.mul_add(w, tail_sum);
            }
        }

        let mut acc = _mm512_add_pd(acc0, acc1);
        let mut sum = _mm512_reduce_add_pd(acc);
        sum += tail_sum;
        *out.get_unchecked_mut(i) = sum * inv_norm;
    }
}

#[derive(Debug, Clone)]
pub struct CwmaStream {
    period:      usize,
    weights:     Vec<f64>,
    inv_norm:    f64,
    ring:        Vec<f64>,
    head:        usize,
    total_count: usize,
    found_first: bool,
    first_idx:   usize,
}

impl CwmaStream {
    pub fn try_new(params: CwmaParams) -> Result<Self, CwmaError> {
        let period = params.period.unwrap_or(14);
        if period == 0 {
            return Err(CwmaError::InvalidPeriod { period, data_len: 0 });
        }

        let mut weights = Vec::with_capacity(period - 1);
        for i in 0..period - 1 {
            weights.push(((period - i) as f64).powi(3));
        }
        let inv_norm = 1.0 / weights.iter().sum::<f64>();

        Ok(Self {
            period,
            weights,
            inv_norm,
            ring: vec![f64::NAN; period],
            head: 0,
            total_count: 0,
            found_first: false,
            first_idx: 0,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        let idx = self.total_count;
        self.total_count += 1;

        if !self.found_first && !value.is_nan() {
            self.found_first = true;
            self.first_idx = idx;
        }

        // overwrite the oldest slot, then move the head to the new oldest
        self.ring[self.head] = value;
        self.head = (self.head + 1) % self.period;

        // --- emit only when batch would: first_idx + period  *** changed ***
        if !self.found_first || idx < self.first_idx + self.period {          // ***
            return None;
        }

        // dot-product: newest sample gets the largest weight (period³)
        let mut sum = 0.0;
        for k in 0..self.period - 1 {
            let ring_idx = (self.head + self.period - 1 - k) % self.period;
            sum += self.ring[ring_idx] * self.weights[k];
        }

        Some(sum * self.inv_norm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::utilities::helpers::skip_if_unsupported;

    fn check_cwma_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = CwmaParams { period: None };
        let input_def = CwmaInput::from_candles(&candles, "close", default_params);
        let output_def = cwma_with_kernel(&input_def, kernel)?;
        assert_eq!(output_def.values.len(), candles.close.len());

        let params_14 = CwmaParams { period: Some(14) };
        let input_14 = CwmaInput::from_candles(&candles, "hl2", params_14);
        let output_14 = cwma_with_kernel(&input_14, kernel)?;
        assert_eq!(output_14.values.len(), candles.close.len());

        let params_custom = CwmaParams { period: Some(20) };
        let input_custom = CwmaInput::from_candles(&candles, "hlc3", params_custom);
        let output_custom = cwma_with_kernel(&input_custom, kernel)?;
        assert_eq!(output_custom.values.len(), candles.close.len());

        Ok(())
    }

    fn check_cwma_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = CwmaInput::with_default_candles(&candles);
        let result = cwma_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), candles.close.len());

        let expected_last_five = [
            59224.641237300435,
            59213.64831277214,
            59171.21190130624,
            59167.01279027576,
            59039.413552249636,
        ];

        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-9,
                "[{}] CWMA {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_cwma_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = CwmaInput::with_default_candles(&candles);
        match input.data {
            CwmaData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected CwmaData::Candles"),
        }
        let output = cwma_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_cwma_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = CwmaParams { period: Some(0) };
        let input = CwmaInput::from_slice(&input_data, params);
        let res = cwma_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] Should fail with zero period", test_name);
        Ok(())
    }

    fn check_cwma_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = CwmaParams { period: Some(10) };
        let input = CwmaInput::from_slice(&data_small, params);
        let res = cwma_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] Should fail with period exceeding length", test_name);
        Ok(())
    }

    fn check_cwma_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported(kernel, test_name);
        let single_point = [42.0];
        let params = CwmaParams { period: Some(9) };
        let input = CwmaInput::from_slice(&single_point, params);
        let res = cwma_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] Should fail with insufficient data", test_name);
        Ok(())
    }

    fn check_cwma_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let first_params = CwmaParams { period: Some(80) };
        let first_input = CwmaInput::from_candles(&candles, "close", first_params);
        let first_result = cwma_with_kernel(&first_input, kernel)?;

        let second_params = CwmaParams { period: Some(60) };
        let second_input = CwmaInput::from_slice(&first_result.values, second_params);
        let second_result = cwma_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());

        if second_result.values.len() > 240 {
            for i in 240..second_result.values.len() {
                assert!(
                    !second_result.values[i].is_nan(),
                    "[{}] Found unexpected NaN at index {}",
                    test_name,
                    i
                );
            }
        }
        Ok(())
    }

    fn check_cwma_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = CwmaInput::from_candles(&candles, "close", CwmaParams { period: Some(9) });
        let res = cwma_with_kernel(&input, kernel)?;
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

    fn check_cwma_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let period = 9;
        let input = CwmaInput::from_candles(&candles, "close", CwmaParams { period: Some(period) });
        let batch_output = cwma_with_kernel(&input, kernel)?.values;

        let mut stream = CwmaStream::try_new(CwmaParams { period: Some(period) })?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(val) => stream_values.push(val),
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
                "[{}] CWMA streaming mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    macro_rules! generate_all_cwma_tests {
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

    generate_all_cwma_tests!(
        check_cwma_partial_params,
        check_cwma_accuracy,
        check_cwma_default_candles,
        check_cwma_zero_period,
        check_cwma_period_exceeds_length,
        check_cwma_very_small_dataset,
        check_cwma_reinput,
        check_cwma_nan_handling,
        check_cwma_streaming
    );
}