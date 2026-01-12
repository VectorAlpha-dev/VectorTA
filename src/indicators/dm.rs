#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::dm_wrapper::CudaDm;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::alma::DeviceArrayF32Py;
use crate::utilities::data_loader::Candles;
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::mem::MaybeUninit;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum DmData<'a> {
    Candles { candles: &'a Candles },
    Slices { high: &'a [f64], low: &'a [f64] },
}

#[derive(Debug, Clone)]
pub struct DmOutput {
    pub plus: Vec<f64>,
    pub minus: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct DmParams {
    pub period: Option<usize>,
}

impl Default for DmParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct DmInput<'a> {
    pub data: DmData<'a>,
    pub params: DmParams,
}

impl<'a> DmInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: DmParams) -> Self {
        Self {
            data: DmData::Candles { candles },
            params,
        }
    }
    #[inline]
    pub fn from_slices(high: &'a [f64], low: &'a [f64], params: DmParams) -> Self {
        Self {
            data: DmData::Slices { high, low },
            params,
        }
    }
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: DmData::Candles { candles },
            params: DmParams::default(),
        }
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| DmParams::default().period.unwrap())
    }
}

#[derive(Copy, Clone, Debug)]
pub struct DmBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for DmBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl DmBuilder {
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
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, candles: &Candles) -> Result<DmOutput, DmError> {
        let p = DmParams {
            period: self.period,
        };
        let i = DmInput::from_candles(candles, p);
        dm_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<DmOutput, DmError> {
        let p = DmParams {
            period: self.period,
        };
        let i = DmInput::from_slices(high, low, p);
        dm_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<DmStream, DmError> {
        let p = DmParams {
            period: self.period,
        };
        DmStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum DmError {
    #[error("dm: Empty data provided (or high/low length mismatch).")]
    EmptyInputData,
    #[error("dm: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("dm: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("dm: All values are NaN.")]
    AllValuesNaN,
    #[error("dm: output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("dm: invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("dm: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[inline]
pub fn dm(input: &DmInput) -> Result<DmOutput, DmError> {
    dm_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn dm_prepare<'a>(
    input: &'a DmInput,
    kernel: Kernel,
) -> Result<(&'a [f64], &'a [f64], usize, usize, Kernel), DmError> {
    let (high, low) = match &input.data {
        DmData::Candles { candles } => {
            let h = candles
                .select_candle_field("high")
                .map_err(|_| DmError::EmptyInputData)?;
            let l = candles
                .select_candle_field("low")
                .map_err(|_| DmError::EmptyInputData)?;
            (h, l)
        }
        DmData::Slices { high, low } => (*high, *low),
    };

    if high.is_empty() || low.is_empty() || high.len() != low.len() {
        return Err(DmError::EmptyInputData);
    }

    let period = input.get_period();
    if period == 0 || period > high.len() {
        return Err(DmError::InvalidPeriod {
            period,
            data_len: high.len(),
        });
    }

    let first = high
        .iter()
        .zip(low.iter())
        .position(|(&h, &l)| !h.is_nan() && !l.is_nan())
        .ok_or(DmError::AllValuesNaN)?;

    if high.len() - first < period {
        return Err(DmError::NotEnoughValidData {
            needed: period,
            valid: high.len() - first,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        k => k,
    };
    Ok((high, low, period, first, chosen))
}

#[inline(always)]
fn dm_compute_into_scalar(
    high: &[f64],
    low: &[f64],
    period: usize,
    first: usize,
    plus_out: &mut [f64],
    minus_out: &mut [f64],
) {
    debug_assert_eq!(high.len(), low.len());
    let n = high.len();
    if n == 0 {
        return;
    }

    let end_init = first + period - 1;

    unsafe {
        let mut sum_plus = 0.0f64;
        let mut sum_minus = 0.0f64;

        let mut i = first + 1;
        let warm_stop = end_init + 1;

        let mut prev_high = *high.get_unchecked(first);
        let mut prev_low = *low.get_unchecked(first);

        while i < warm_stop {
            let hi = *high.get_unchecked(i);
            let lo = *low.get_unchecked(i);
            let diff_p = hi - prev_high;
            let diff_m = prev_low - lo;
            prev_high = hi;
            prev_low = lo;

            if diff_p > 0.0 && diff_p > diff_m {
                sum_plus += diff_p;
            } else if diff_m > 0.0 && diff_m > diff_p {
                sum_minus += diff_m;
            }
            i += 1;
        }

        *plus_out.get_unchecked_mut(end_init) = sum_plus;
        *minus_out.get_unchecked_mut(end_init) = sum_minus;

        if end_init + 1 >= n {
            return;
        }
        let inv_p = 1.0 / (period as f64);

        let mut j = end_init + 1;
        while j < n {
            let hi = *high.get_unchecked(j);
            let lo = *low.get_unchecked(j);
            let diff_p = hi - prev_high;
            let diff_m = prev_low - lo;
            prev_high = hi;
            prev_low = lo;

            let (p, m) = if diff_p > 0.0 && diff_p > diff_m {
                (diff_p, 0.0)
            } else if diff_m > 0.0 && diff_m > diff_p {
                (0.0, diff_m)
            } else {
                (0.0, 0.0)
            };

            #[cfg(target_feature = "fma")]
            {
                sum_plus = (-inv_p).mul_add(sum_plus, sum_plus + p);
                sum_minus = (-inv_p).mul_add(sum_minus, sum_minus + m);
            }
            #[cfg(not(target_feature = "fma"))]
            {
                sum_plus = sum_plus - (sum_plus * inv_p) + p;
                sum_minus = sum_minus - (sum_minus * inv_p) + m;
            }

            *plus_out.get_unchecked_mut(j) = sum_plus;
            *minus_out.get_unchecked_mut(j) = sum_minus;
            j += 1;
        }
    }
}

#[inline(always)]
fn dm_compute_into(
    high: &[f64],
    low: &[f64],
    period: usize,
    first: usize,
    kernel: Kernel,
    plus_out: &mut [f64],
    minus_out: &mut [f64],
) {
    match kernel {
        Kernel::Scalar | Kernel::ScalarBatch => {
            dm_compute_into_scalar(high, low, period, first, plus_out, minus_out)
        }
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2 | Kernel::Avx2Batch => unsafe {
            dm_compute_into_avx2(high, low, period, first, plus_out, minus_out)
        },
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512 | Kernel::Avx512Batch => unsafe {
            dm_compute_into_avx512(high, low, period, first, plus_out, minus_out)
        },
        _ => unreachable!(),
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn dm_compute_into_avx2(
    high: &[f64],
    low: &[f64],
    period: usize,
    first: usize,
    plus_out: &mut [f64],
    minus_out: &mut [f64],
) {
    use core::arch::x86_64::*;
    debug_assert_eq!(high.len(), low.len());
    let n = high.len();
    if n == 0 {
        return;
    }

    let end_init = first + period - 1;
    let inv_p = 1.0 / (period as f64);
    let zero = _mm256_setzero_pd();

    let mut sum_plus = 0.0f64;
    let mut sum_minus = 0.0f64;
    let mut i = first + 1;
    let warm_stop = end_init + 1;
    while i + 4 <= warm_stop {
        let hc = _mm256_loadu_pd(high.as_ptr().add(i));
        let hp = _mm256_loadu_pd(high.as_ptr().add(i - 1));
        let dp = _mm256_sub_pd(hc, hp);

        let lp = _mm256_loadu_pd(low.as_ptr().add(i - 1));
        let lc = _mm256_loadu_pd(low.as_ptr().add(i));
        let dm = _mm256_sub_pd(lp, lc);

        let dp_pos = _mm256_max_pd(dp, zero);
        let dm_pos = _mm256_max_pd(dm, zero);

        let p_mask = _mm256_cmp_pd(dp_pos, dm_pos, _CMP_GT_OQ);
        let m_mask = _mm256_cmp_pd(dm_pos, dp_pos, _CMP_GT_OQ);
        let p_vec = _mm256_and_pd(dp_pos, p_mask);
        let m_vec = _mm256_and_pd(dm_pos, m_mask);

        let mut p_buf = [0.0f64; 4];
        let mut m_buf = [0.0f64; 4];
        _mm256_storeu_pd(p_buf.as_mut_ptr(), p_vec);
        _mm256_storeu_pd(m_buf.as_mut_ptr(), m_vec);
        sum_plus += p_buf.iter().sum::<f64>();
        sum_minus += m_buf.iter().sum::<f64>();
        i += 4;
    }
    while i < warm_stop {
        let dp = *high.get_unchecked(i) - *high.get_unchecked(i - 1);
        let dm = *low.get_unchecked(i - 1) - *low.get_unchecked(i);
        if dp > 0.0 && dp > dm {
            sum_plus += dp;
        } else if dm > 0.0 && dm > dp {
            sum_minus += dm;
        }
        i += 1;
    }

    *plus_out.get_unchecked_mut(end_init) = sum_plus;
    *minus_out.get_unchecked_mut(end_init) = sum_minus;

    if end_init + 1 >= n {
        return;
    }

    let mut j = end_init + 1;
    while j + 4 <= n {
        let hc = _mm256_loadu_pd(high.as_ptr().add(j));
        let hp = _mm256_loadu_pd(high.as_ptr().add(j - 1));
        let dp = _mm256_sub_pd(hc, hp);

        let lp = _mm256_loadu_pd(low.as_ptr().add(j - 1));
        let lc = _mm256_loadu_pd(low.as_ptr().add(j));
        let dm = _mm256_sub_pd(lp, lc);

        let dp_pos = _mm256_max_pd(dp, zero);
        let dm_pos = _mm256_max_pd(dm, zero);

        let p_mask = _mm256_cmp_pd(dp_pos, dm_pos, _CMP_GT_OQ);
        let m_mask = _mm256_cmp_pd(dm_pos, dp_pos, _CMP_GT_OQ);
        let p_vec = _mm256_and_pd(dp_pos, p_mask);
        let m_vec = _mm256_and_pd(dm_pos, m_mask);

        let mut p_buf = [0.0f64; 4];
        let mut m_buf = [0.0f64; 4];
        _mm256_storeu_pd(p_buf.as_mut_ptr(), p_vec);
        _mm256_storeu_pd(m_buf.as_mut_ptr(), m_vec);

        #[cfg(target_feature = "fma")]
        {
            sum_plus = (-inv_p).mul_add(sum_plus, sum_plus + p_buf[0]);
            sum_minus = (-inv_p).mul_add(sum_minus, sum_minus + m_buf[0]);
            *plus_out.get_unchecked_mut(j) = sum_plus;
            *minus_out.get_unchecked_mut(j) = sum_minus;

            sum_plus = (-inv_p).mul_add(sum_plus, sum_plus + p_buf[1]);
            sum_minus = (-inv_p).mul_add(sum_minus, sum_minus + m_buf[1]);
            *plus_out.get_unchecked_mut(j + 1) = sum_plus;
            *minus_out.get_unchecked_mut(j + 1) = sum_minus;

            sum_plus = (-inv_p).mul_add(sum_plus, sum_plus + p_buf[2]);
            sum_minus = (-inv_p).mul_add(sum_minus, sum_minus + m_buf[2]);
            *plus_out.get_unchecked_mut(j + 2) = sum_plus;
            *minus_out.get_unchecked_mut(j + 2) = sum_minus;

            sum_plus = (-inv_p).mul_add(sum_plus, sum_plus + p_buf[3]);
            sum_minus = (-inv_p).mul_add(sum_minus, sum_minus + m_buf[3]);
            *plus_out.get_unchecked_mut(j + 3) = sum_plus;
            *minus_out.get_unchecked_mut(j + 3) = sum_minus;
        }
        #[cfg(not(target_feature = "fma"))]
        {
            sum_plus = sum_plus - (sum_plus * inv_p) + p_buf[0];
            sum_minus = sum_minus - (sum_minus * inv_p) + m_buf[0];
            *plus_out.get_unchecked_mut(j) = sum_plus;
            *minus_out.get_unchecked_mut(j) = sum_minus;

            sum_plus = sum_plus - (sum_plus * inv_p) + p_buf[1];
            sum_minus = sum_minus - (sum_minus * inv_p) + m_buf[1];
            *plus_out.get_unchecked_mut(j + 1) = sum_plus;
            *minus_out.get_unchecked_mut(j + 1) = sum_minus;

            sum_plus = sum_plus - (sum_plus * inv_p) + p_buf[2];
            sum_minus = sum_minus - (sum_minus * inv_p) + m_buf[2];
            *plus_out.get_unchecked_mut(j + 2) = sum_plus;
            *minus_out.get_unchecked_mut(j + 2) = sum_minus;

            sum_plus = sum_plus - (sum_plus * inv_p) + p_buf[3];
            sum_minus = sum_minus - (sum_minus * inv_p) + m_buf[3];
            *plus_out.get_unchecked_mut(j + 3) = sum_plus;
            *minus_out.get_unchecked_mut(j + 3) = sum_minus;
        }
        j += 4;
    }

    while j < n {
        let dp = *high.get_unchecked(j) - *high.get_unchecked(j - 1);
        let dm = *low.get_unchecked(j - 1) - *low.get_unchecked(j);

        let (p, m) = if dp > 0.0 && dp > dm {
            (dp, 0.0)
        } else if dm > 0.0 && dm > dp {
            (0.0, dm)
        } else {
            (0.0, 0.0)
        };

        #[cfg(target_feature = "fma")]
        {
            sum_plus = (-inv_p).mul_add(sum_plus, sum_plus + p);
            sum_minus = (-inv_p).mul_add(sum_minus, sum_minus + m);
        }
        #[cfg(not(target_feature = "fma"))]
        {
            sum_plus = sum_plus - (sum_plus * inv_p) + p;
            sum_minus = sum_minus - (sum_minus * inv_p) + m;
        }
        *plus_out.get_unchecked_mut(j) = sum_plus;
        *minus_out.get_unchecked_mut(j) = sum_minus;
        j += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn dm_compute_into_avx512(
    high: &[f64],
    low: &[f64],
    period: usize,
    first: usize,
    plus_out: &mut [f64],
    minus_out: &mut [f64],
) {
    use core::arch::x86_64::*;
    debug_assert_eq!(high.len(), low.len());
    let n = high.len();
    if n == 0 {
        return;
    }

    let end_init = first + period - 1;
    let inv_p = 1.0 / (period as f64);
    let zero = _mm512_set1_pd(0.0);

    let mut sum_plus = 0.0f64;
    let mut sum_minus = 0.0f64;
    let mut i = first + 1;
    let warm_stop = end_init + 1;
    while i + 8 <= warm_stop {
        let hc = _mm512_loadu_pd(high.as_ptr().add(i));
        let hp = _mm512_loadu_pd(high.as_ptr().add(i - 1));
        let dp = _mm512_sub_pd(hc, hp);

        let lp = _mm512_loadu_pd(low.as_ptr().add(i - 1));
        let lc = _mm512_loadu_pd(low.as_ptr().add(i));
        let dm = _mm512_sub_pd(lp, lc);

        let dp_pos = _mm512_max_pd(dp, zero);
        let dm_pos = _mm512_max_pd(dm, zero);

        let p_mask = _mm512_cmp_pd_mask(dp_pos, dm_pos, _CMP_GT_OQ);
        let m_mask = _mm512_cmp_pd_mask(dm_pos, dp_pos, _CMP_GT_OQ);
        let p_vec = _mm512_maskz_mov_pd(p_mask, dp_pos);
        let m_vec = _mm512_maskz_mov_pd(m_mask, dm_pos);

        let mut p_buf = [0.0f64; 8];
        let mut m_buf = [0.0f64; 8];
        _mm512_storeu_pd(p_buf.as_mut_ptr(), p_vec);
        _mm512_storeu_pd(m_buf.as_mut_ptr(), m_vec);
        for k in 0..8 {
            sum_plus += p_buf[k];
            sum_minus += m_buf[k];
        }
        i += 8;
    }
    while i < warm_stop {
        let dp = *high.get_unchecked(i) - *high.get_unchecked(i - 1);
        let dm = *low.get_unchecked(i - 1) - *low.get_unchecked(i);
        if dp > 0.0 && dp > dm {
            sum_plus += dp;
        } else if dm > 0.0 && dm > dp {
            sum_minus += dm;
        }
        i += 1;
    }
    *plus_out.get_unchecked_mut(end_init) = sum_plus;
    *minus_out.get_unchecked_mut(end_init) = sum_minus;

    if end_init + 1 >= n {
        return;
    }

    let mut j = end_init + 1;
    while j + 8 <= n {
        let hc = _mm512_loadu_pd(high.as_ptr().add(j));
        let hp = _mm512_loadu_pd(high.as_ptr().add(j - 1));
        let dp = _mm512_sub_pd(hc, hp);

        let lp = _mm512_loadu_pd(low.as_ptr().add(j - 1));
        let lc = _mm512_loadu_pd(low.as_ptr().add(j));
        let dm = _mm512_sub_pd(lp, lc);

        let dp_pos = _mm512_max_pd(dp, zero);
        let dm_pos = _mm512_max_pd(dm, zero);

        let p_mask = _mm512_cmp_pd_mask(dp_pos, dm_pos, _CMP_GT_OQ);
        let m_mask = _mm512_cmp_pd_mask(dm_pos, dp_pos, _CMP_GT_OQ);
        let p_vec = _mm512_maskz_mov_pd(p_mask, dp_pos);
        let m_vec = _mm512_maskz_mov_pd(m_mask, dm_pos);

        let mut p_buf = [0.0f64; 8];
        let mut m_buf = [0.0f64; 8];
        _mm512_storeu_pd(p_buf.as_mut_ptr(), p_vec);
        _mm512_storeu_pd(m_buf.as_mut_ptr(), m_vec);

        #[cfg(target_feature = "fma")]
        {
            for t in 0..8 {
                sum_plus = (-inv_p).mul_add(sum_plus, sum_plus + p_buf[t]);
                sum_minus = (-inv_p).mul_add(sum_minus, sum_minus + m_buf[t]);
                *plus_out.get_unchecked_mut(j + t) = sum_plus;
                *minus_out.get_unchecked_mut(j + t) = sum_minus;
            }
        }
        #[cfg(not(target_feature = "fma"))]
        {
            for t in 0..8 {
                sum_plus = sum_plus - (sum_plus * inv_p) + p_buf[t];
                sum_minus = sum_minus - (sum_minus * inv_p) + m_buf[t];
                *plus_out.get_unchecked_mut(j + t) = sum_plus;
                *minus_out.get_unchecked_mut(j + t) = sum_minus;
            }
        }
        j += 8;
    }
    while j < n {
        let dp = *high.get_unchecked(j) - *high.get_unchecked(j - 1);
        let dm = *low.get_unchecked(j - 1) - *low.get_unchecked(j);

        let (p, m) = if dp > 0.0 && dp > dm {
            (dp, 0.0)
        } else if dm > 0.0 && dm > dp {
            (0.0, dm)
        } else {
            (0.0, 0.0)
        };

        #[cfg(target_feature = "fma")]
        {
            sum_plus = (-inv_p).mul_add(sum_plus, sum_plus + p);
            sum_minus = (-inv_p).mul_add(sum_minus, sum_minus + m);
        }
        #[cfg(not(target_feature = "fma"))]
        {
            sum_plus = sum_plus - (sum_plus * inv_p) + p;
            sum_minus = sum_minus - (sum_minus * inv_p) + m;
        }
        *plus_out.get_unchecked_mut(j) = sum_plus;
        *minus_out.get_unchecked_mut(j) = sum_minus;
        j += 1;
    }
}

pub fn dm_with_kernel(input: &DmInput, kernel: Kernel) -> Result<DmOutput, DmError> {
    let (high, low, period, first, chosen) = dm_prepare(input, kernel)?;
    let warm = first + period - 1;

    let mut plus = alloc_with_nan_prefix(high.len(), warm);
    let mut minus = alloc_with_nan_prefix(high.len(), warm);

    dm_compute_into(high, low, period, first, chosen, &mut plus, &mut minus);
    Ok(DmOutput { plus, minus })
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn dm_into(
    input: &DmInput,
    plus_out: &mut [f64],
    minus_out: &mut [f64],
) -> Result<(), DmError> {
    let (high, low, period, first, chosen) = dm_prepare(input, Kernel::Auto)?;

    if plus_out.len() != high.len() {
        return Err(DmError::OutputLengthMismatch {
            expected: high.len(),
            got: plus_out.len(),
        });
    }
    if minus_out.len() != high.len() {
        return Err(DmError::OutputLengthMismatch {
            expected: high.len(),
            got: minus_out.len(),
        });
    }

    let warm = first + period - 1;
    let qnan = f64::from_bits(0x7ff8_0000_0000_0000);
    let warm_end = warm.min(high.len());
    for v in &mut plus_out[..warm_end] {
        *v = qnan;
    }
    for v in &mut minus_out[..warm_end] {
        *v = qnan;
    }

    dm_compute_into(high, low, period, first, chosen, plus_out, minus_out);
    Ok(())
}

#[inline]
pub fn dm_into_slice(
    plus_dst: &mut [f64],
    minus_dst: &mut [f64],
    input: &DmInput,
    kernel: Kernel,
) -> Result<(), DmError> {
    let (high, low, period, first, chosen) = dm_prepare(input, kernel)?;
    if plus_dst.len() != high.len() {
        return Err(DmError::OutputLengthMismatch {
            expected: high.len(),
            got: plus_dst.len(),
        });
    }
    if minus_dst.len() != high.len() {
        return Err(DmError::OutputLengthMismatch {
            expected: high.len(),
            got: minus_dst.len(),
        });
    }

    dm_compute_into(high, low, period, first, chosen, plus_dst, minus_dst);

    let warm = first + period - 1;
    for v in &mut plus_dst[..warm] {
        *v = f64::NAN;
    }
    for v in &mut minus_dst[..warm] {
        *v = f64::NAN;
    }
    Ok(())
}

#[inline]
pub unsafe fn dm_scalar(
    high: &[f64],
    low: &[f64],
    period: usize,
    first_valid_idx: usize,
) -> Result<DmOutput, DmError> {
    let warm = first_valid_idx + period - 1;
    let mut plus_dm = alloc_with_nan_prefix(high.len(), warm);
    let mut minus_dm = alloc_with_nan_prefix(high.len(), warm);

    dm_compute_into_scalar(
        high,
        low,
        period,
        first_valid_idx,
        &mut plus_dm,
        &mut minus_dm,
    );

    Ok(DmOutput {
        plus: plus_dm,
        minus: minus_dm,
    })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn dm_avx2(
    high: &[f64],
    low: &[f64],
    period: usize,
    first_valid_idx: usize,
) -> Result<DmOutput, DmError> {
    let warm = first_valid_idx + period - 1;
    let mut plus_dm = alloc_with_nan_prefix(high.len(), warm);
    let mut minus_dm = alloc_with_nan_prefix(high.len(), warm);
    dm_compute_into_avx2(
        high,
        low,
        period,
        first_valid_idx,
        &mut plus_dm,
        &mut minus_dm,
    );
    Ok(DmOutput {
        plus: plus_dm,
        minus: minus_dm,
    })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn dm_avx512(
    high: &[f64],
    low: &[f64],
    period: usize,
    first_valid_idx: usize,
) -> Result<DmOutput, DmError> {
    let warm = first_valid_idx + period - 1;
    let mut plus_dm = alloc_with_nan_prefix(high.len(), warm);
    let mut minus_dm = alloc_with_nan_prefix(high.len(), warm);
    dm_compute_into_avx512(
        high,
        low,
        period,
        first_valid_idx,
        &mut plus_dm,
        &mut minus_dm,
    );
    Ok(DmOutput {
        plus: plus_dm,
        minus: minus_dm,
    })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn dm_avx512_short(
    high: &[f64],
    low: &[f64],
    period: usize,
    first_valid_idx: usize,
) -> Result<DmOutput, DmError> {
    dm_avx512(high, low, period, first_valid_idx)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn dm_avx512_long(
    high: &[f64],
    low: &[f64],
    period: usize,
    first_valid_idx: usize,
) -> Result<DmOutput, DmError> {
    dm_avx512(high, low, period, first_valid_idx)
}

#[derive(Debug, Clone)]
pub struct DmStream {
    period: usize,
    inv_period: f64,
    sum_plus: f64,
    sum_minus: f64,
    prev_high: f64,
    prev_low: f64,
    count: usize,
}

impl DmStream {
    pub fn try_new(params: DmParams) -> Result<Self, DmError> {
        let period = params.period.unwrap_or(14);
        if period == 0 {
            return Err(DmError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        let inv = 1.0 / (period as f64);
        Ok(Self {
            period,
            inv_period: inv,
            sum_plus: 0.0,
            sum_minus: 0.0,
            prev_high: f64::NAN,
            prev_low: f64::NAN,
            count: 0,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64) -> Option<(f64, f64)> {
        if self.count == 0 {
            self.prev_high = high;
            self.prev_low = low;
        }

        let dp = high - self.prev_high;
        let dm = self.prev_low - low;

        self.prev_high = high;
        self.prev_low = low;

        let dp_pos = dp.max(0.0);
        let dm_pos = dm.max(0.0);

        let plus_val = if dp_pos > dm_pos { dp_pos } else { 0.0 };
        let minus_val = if dm_pos > dp_pos { dm_pos } else { 0.0 };

        if self.count < self.period - 1 {
            self.sum_plus += plus_val;
            self.sum_minus += minus_val;
            self.count += 1;
            return None;
        } else if self.count == self.period - 1 {
            self.sum_plus += plus_val;
            self.sum_minus += minus_val;
            self.count += 1;
            return Some((self.sum_plus, self.sum_minus));
        }

        #[cfg(target_feature = "fma")]
        {
            self.sum_plus = (-self.inv_period).mul_add(self.sum_plus, self.sum_plus + plus_val);
            self.sum_minus = (-self.inv_period).mul_add(self.sum_minus, self.sum_minus + minus_val);
        }
        #[cfg(not(target_feature = "fma"))]
        {
            self.sum_plus = self.sum_plus - (self.sum_plus * self.inv_period) + plus_val;
            self.sum_minus = self.sum_minus - (self.sum_minus * self.inv_period) + minus_val;
        }

        Some((self.sum_plus, self.sum_minus))
    }
}

#[derive(Clone, Debug)]
pub struct DmBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for DmBatchRange {
    fn default() -> Self {
        Self {
            period: (14, 263, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct DmBatchBuilder {
    range: DmBatchRange,
    kernel: Kernel,
}

impl DmBatchBuilder {
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
    pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<DmBatchOutput, DmError> {
        dm_batch_with_kernel(high, low, &self.range, self.kernel)
    }
    pub fn apply_candles(self, c: &Candles) -> Result<DmBatchOutput, DmError> {
        let high = c
            .select_candle_field("high")
            .map_err(|_| DmError::EmptyInputData)?;
        let low = c
            .select_candle_field("low")
            .map_err(|_| DmError::EmptyInputData)?;
        self.apply_slices(high, low)
    }
    pub fn with_default_candles(c: &Candles) -> Result<DmBatchOutput, DmError> {
        DmBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c)
    }
}

#[derive(Clone, Debug)]
pub struct DmBatchOutput {
    pub plus: Vec<f64>,
    pub minus: Vec<f64>,
    pub combos: Vec<DmParams>,
    pub rows: usize,
    pub cols: usize,
}
impl DmBatchOutput {
    pub fn row_for_params(&self, p: &DmParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(14) == p.period.unwrap_or(14))
    }
    pub fn values_for(&self, p: &DmParams) -> Option<(&[f64], &[f64])> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            (
                &self.plus[start..start + self.cols],
                &self.minus[start..start + self.cols],
            )
        })
    }
}

#[inline(always)]
fn expand_grid(r: &DmBatchRange) -> Result<Vec<DmParams>, DmError> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, DmError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        if start < end {
            let mut v = Vec::new();
            let st = step.max(1);
            let mut x = start;
            while x <= end {
                v.push(x);
                match x.checked_add(st) {
                    Some(next) => x = next,
                    None => break,
                }
            }
            if v.is_empty() {
                return Err(DmError::InvalidRange { start, end, step });
            }
            return Ok(v);
        }

        let mut v = Vec::new();
        let st = step.max(1) as isize;
        let mut x = start as isize;
        let end_i = end as isize;
        while x >= end_i {
            v.push(x as usize);
            x -= st;
        }
        if v.is_empty() {
            return Err(DmError::InvalidRange { start, end, step });
        }
        Ok(v)
    }

    let periods = axis_usize(r.period)?;
    let mut out = Vec::with_capacity(periods.len());
    for p in periods {
        out.push(DmParams { period: Some(p) });
    }
    Ok(out)
}

pub fn dm_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    sweep: &DmBatchRange,
    k: Kernel,
) -> Result<DmBatchOutput, DmError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(DmError::InvalidKernelForBatch(k)),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    dm_batch_par_slice(high, low, sweep, simd)
}

#[inline(always)]
pub fn dm_batch_slice(
    high: &[f64],
    low: &[f64],
    sweep: &DmBatchRange,
    kern: Kernel,
) -> Result<DmBatchOutput, DmError> {
    dm_batch_inner(high, low, sweep, kern, false)
}

#[inline(always)]
pub fn dm_batch_par_slice(
    high: &[f64],
    low: &[f64],
    sweep: &DmBatchRange,
    kern: Kernel,
) -> Result<DmBatchOutput, DmError> {
    dm_batch_inner(high, low, sweep, kern, true)
}

#[inline(always)]
fn dm_batch_inner_into(
    high: &[f64],
    low: &[f64],
    sweep: &DmBatchRange,
    kern: Kernel,
    parallel: bool,
    first: usize,
    plus_out: &mut [f64],
    minus_out: &mut [f64],
) -> Result<Vec<DmParams>, DmError> {
    let combos = expand_grid(sweep)?;

    let rows = combos.len();
    let cols = high.len();

    let _total = rows.checked_mul(cols).ok_or(DmError::InvalidRange {
        start: sweep.period.0,
        end: sweep.period.1,
        step: sweep.period.2,
    })?;
    let chosen = match kern {
        Kernel::Auto => detect_best_batch_kernel(),
        k => k,
    };

    let do_row = |row: usize, plus_row: &mut [f64], minus_row: &mut [f64]| {
        let p = combos[row].period.unwrap();
        dm_compute_into(
            high,
            low,
            p,
            first,
            match chosen {
                Kernel::Avx512Batch => Kernel::Avx512,
                Kernel::Avx2Batch => Kernel::Avx2,
                Kernel::ScalarBatch => Kernel::Scalar,
                k => k,
            },
            plus_row,
            minus_row,
        );
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;
            plus_out
                .par_chunks_mut(cols)
                .zip(minus_out.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(r, (pr, mr))| do_row(r, pr, mr));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (r, (pr, mr)) in plus_out
                .chunks_mut(cols)
                .zip(minus_out.chunks_mut(cols))
                .enumerate()
            {
                do_row(r, pr, mr);
            }
        }
    } else {
        for (r, (pr, mr)) in plus_out
            .chunks_mut(cols)
            .zip(minus_out.chunks_mut(cols))
            .enumerate()
        {
            do_row(r, pr, mr);
        }
    }

    Ok(combos)
}

#[inline(always)]
fn dm_batch_inner(
    high: &[f64],
    low: &[f64],
    sweep: &DmBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<DmBatchOutput, DmError> {
    let combos = expand_grid(sweep)?;

    let first = high
        .iter()
        .zip(low.iter())
        .position(|(&h, &l)| !h.is_nan() && !l.is_nan())
        .ok_or(DmError::AllValuesNaN)?;

    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if high.len() - first < max_p {
        return Err(DmError::NotEnoughValidData {
            needed: max_p,
            valid: high.len() - first,
        });
    }

    let rows = combos.len();
    let cols = high.len();

    let _total = rows.checked_mul(cols).ok_or(DmError::InvalidRange {
        start: sweep.period.0,
        end: sweep.period.1,
        step: sweep.period.2,
    })?;

    let mut plus_mu = make_uninit_matrix(rows, cols);
    let mut minus_mu = make_uninit_matrix(rows, cols);

    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();
    init_matrix_prefixes(&mut plus_mu, cols, &warm);
    init_matrix_prefixes(&mut minus_mu, cols, &warm);

    let mut plus_guard = core::mem::ManuallyDrop::new(plus_mu);
    let mut minus_guard = core::mem::ManuallyDrop::new(minus_mu);
    let plus_out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(plus_guard.as_mut_ptr() as *mut f64, plus_guard.len())
    };
    let minus_out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(minus_guard.as_mut_ptr() as *mut f64, minus_guard.len())
    };

    let combos = dm_batch_inner_into(high, low, sweep, kern, parallel, first, plus_out, minus_out)?;

    let plus = unsafe {
        Vec::from_raw_parts(
            plus_guard.as_mut_ptr() as *mut f64,
            plus_guard.len(),
            plus_guard.capacity(),
        )
    };
    let minus = unsafe {
        Vec::from_raw_parts(
            minus_guard.as_mut_ptr() as *mut f64,
            minus_guard.len(),
            minus_guard.capacity(),
        )
    };

    Ok(DmBatchOutput {
        plus,
        minus,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
unsafe fn dm_row_scalar(
    high: &[f64],
    low: &[f64],
    first: usize,
    period: usize,
    plus: &mut [f64],
    minus: &mut [f64],
) {
    let mut prev_high = high[first];
    let mut prev_low = low[first];
    let mut sum_plus = 0.0;
    let mut sum_minus = 0.0;

    let end_init = first + period - 1;
    for i in (first + 1)..=end_init {
        let diff_p = high[i] - prev_high;
        let diff_m = prev_low - low[i];
        prev_high = high[i];
        prev_low = low[i];

        let plus_val = if diff_p > 0.0 && diff_p > diff_m {
            diff_p
        } else {
            0.0
        };
        let minus_val = if diff_m > 0.0 && diff_m > diff_p {
            diff_m
        } else {
            0.0
        };

        sum_plus += plus_val;
        sum_minus += minus_val;
    }

    plus[end_init] = sum_plus;
    minus[end_init] = sum_minus;

    let inv_period = 1.0 / (period as f64);

    for i in (end_init + 1)..high.len() {
        let diff_p = high[i] - prev_high;
        let diff_m = prev_low - low[i];
        prev_high = high[i];
        prev_low = low[i];

        let plus_val = if diff_p > 0.0 && diff_p > diff_m {
            diff_p
        } else {
            0.0
        };
        let minus_val = if diff_m > 0.0 && diff_m > diff_p {
            diff_m
        } else {
            0.0
        };

        sum_plus = sum_plus - (sum_plus * inv_period) + plus_val;
        sum_minus = sum_minus - (sum_minus * inv_period) + minus_val;

        plus[i] = sum_plus;
        minus[i] = sum_minus;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn dm_row_avx2(
    high: &[f64],
    low: &[f64],
    first: usize,
    period: usize,
    plus: &mut [f64],
    minus: &mut [f64],
) {
    dm_row_scalar(high, low, first, period, plus, minus)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn dm_row_avx512(
    high: &[f64],
    low: &[f64],
    first: usize,
    period: usize,
    plus: &mut [f64],
    minus: &mut [f64],
) {
    dm_row_scalar(high, low, first, period, plus, minus)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn dm_row_avx512_short(
    high: &[f64],
    low: &[f64],
    first: usize,
    period: usize,
    plus: &mut [f64],
    minus: &mut [f64],
) {
    dm_row_avx512(high, low, first, period, plus, minus)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn dm_row_avx512_long(
    high: &[f64],
    low: &[f64],
    first: usize,
    period: usize,
    plus: &mut [f64],
    minus: &mut [f64],
) {
    dm_row_avx512(high, low, first, period, plus, minus)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn check_dm_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = DmParams { period: None };
        let input_default = DmInput::from_candles(&candles, default_params);
        let output_default = dm_with_kernel(&input_default, kernel)?;
        assert_eq!(output_default.plus.len(), candles.high.len());
        assert_eq!(output_default.minus.len(), candles.high.len());

        let params_custom = DmParams { period: Some(10) };
        let input_custom = DmInput::from_candles(&candles, params_custom);
        let output_custom = dm_with_kernel(&input_custom, kernel)?;
        assert_eq!(output_custom.plus.len(), candles.high.len());
        assert_eq!(output_custom.minus.len(), candles.high.len());
        Ok(())
    }

    fn check_dm_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = DmInput::with_default_candles(&candles);
        let result = dm_with_kernel(&input, kernel)?;
        assert_eq!(result.plus.len(), candles.high.len());
        assert_eq!(result.minus.len(), candles.high.len());
        Ok(())
    }

    fn check_dm_with_slice_data(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high_values = [8000.0, 8050.0, 8100.0, 8075.0, 8110.0, 8050.0];
        let low_values = [7800.0, 7900.0, 7950.0, 7950.0, 8000.0, 7950.0];
        let params = DmParams { period: Some(3) };
        let input = DmInput::from_slices(&high_values, &low_values, params);
        let result = dm_with_kernel(&input, kernel)?;
        assert_eq!(result.plus.len(), 6);
        assert_eq!(result.minus.len(), 6);

        for i in 0..2 {
            assert!(result.plus[i].is_nan());
            assert!(result.minus[i].is_nan());
        }
        Ok(())
    }

    fn check_dm_zero_period(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high_values = [100.0, 110.0, 120.0];
        let low_values = [90.0, 100.0, 110.0];
        let params = DmParams { period: Some(0) };
        let input = DmInput::from_slices(&high_values, &low_values, params);
        let result = dm_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_dm_period_exceeds_data_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high_values = [100.0, 110.0, 120.0];
        let low_values = [90.0, 100.0, 110.0];
        let params = DmParams { period: Some(10) };
        let input = DmInput::from_slices(&high_values, &low_values, params);
        let result = dm_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_dm_not_enough_valid_data(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high_values = [f64::NAN, f64::NAN, 100.0, 101.0, 102.0];
        let low_values = [f64::NAN, f64::NAN, 90.0, 89.0, 88.0];
        let params = DmParams { period: Some(5) };
        let input = DmInput::from_slices(&high_values, &low_values, params);
        let result = dm_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_dm_all_values_nan(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high_values = [f64::NAN, f64::NAN, f64::NAN];
        let low_values = [f64::NAN, f64::NAN, f64::NAN];
        let params = DmParams { period: Some(3) };
        let input = DmInput::from_slices(&high_values, &low_values, params);
        let result = dm_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_dm_with_slice_reinput(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high_values = [9000.0, 9100.0, 9050.0, 9200.0, 9150.0, 9300.0];
        let low_values = [8900.0, 9000.0, 8950.0, 9000.0, 9050.0, 9100.0];
        let params = DmParams { period: Some(2) };
        let input_first = DmInput::from_slices(&high_values, &low_values, params.clone());
        let result_first = dm_with_kernel(&input_first, kernel)?;
        let input_second = DmInput::from_slices(&result_first.plus, &result_first.minus, params);
        let result_second = dm_with_kernel(&input_second, kernel)?;
        assert_eq!(result_second.plus.len(), high_values.len());
        assert_eq!(result_second.minus.len(), high_values.len());
        Ok(())
    }

    fn check_dm_known_values(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = DmParams { period: Some(14) };
        let input = DmInput::from_candles(&candles, params);
        let output = dm_with_kernel(&input, kernel)?;

        let slice_size = 5;
        let last_plus_slice = &output.plus[output.plus.len() - slice_size..];
        let last_minus_slice = &output.minus[output.minus.len() - slice_size..];

        let expected_plus = [
            1410.819956368491,
            1384.04710234217,
            1285.186595032015,
            1199.3875525297283,
            1113.7170130633192,
        ];
        let expected_minus = [
            3602.8631384045057,
            3345.5157713756125,
            3258.5503591344973,
            3025.796762053462,
            3493.668421906786,
        ];

        for i in 0..slice_size {
            let diff_plus = (last_plus_slice[i] - expected_plus[i]).abs();
            let diff_minus = (last_minus_slice[i] - expected_minus[i]).abs();
            assert!(diff_plus < 1e-6);
            assert!(diff_minus < 1e-6);
        }
        Ok(())
    }

    macro_rules! generate_all_dm_tests {
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

    #[cfg(debug_assertions)]
    fn check_dm_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let test_params = vec![
            DmParams::default(),
            DmParams { period: Some(2) },
            DmParams { period: Some(3) },
            DmParams { period: Some(5) },
            DmParams { period: Some(7) },
            DmParams { period: Some(10) },
            DmParams { period: Some(14) },
            DmParams { period: Some(20) },
            DmParams { period: Some(30) },
            DmParams { period: Some(50) },
            DmParams { period: Some(100) },
            DmParams { period: Some(200) },
            DmParams { period: Some(25) },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = DmInput::from_candles(&candles, params.clone());
            let output = dm_with_kernel(&input, kernel)?;

            for (i, &val) in output.plus.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} in plus array \
						 with params: period={} (param set {})",
						test_name, val, bits, i,
						params.period.unwrap_or(14), param_idx
					);
                }

                if bits == 0x22222222_22222222 {
                    panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} in plus array \
						 with params: period={} (param set {})",
						test_name, val, bits, i,
						params.period.unwrap_or(14), param_idx
					);
                }

                if bits == 0x33333333_33333333 {
                    panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} in plus array \
						 with params: period={} (param set {})",
						test_name, val, bits, i,
						params.period.unwrap_or(14), param_idx
					);
                }
            }

            for (i, &val) in output.minus.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} in minus array \
						 with params: period={} (param set {})",
						test_name, val, bits, i,
						params.period.unwrap_or(14), param_idx
					);
                }

                if bits == 0x22222222_22222222 {
                    panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} in minus array \
						 with params: period={} (param set {})",
						test_name, val, bits, i,
						params.period.unwrap_or(14), param_idx
					);
                }

                if bits == 0x33333333_33333333 {
                    panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} in minus array \
						 with params: period={} (param set {})",
						test_name, val, bits, i,
						params.period.unwrap_or(14), param_idx
					);
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_dm_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_dm_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let strat = (2usize..=50).prop_flat_map(|period| {
            (
                (100f64..10000f64, 0.01f64..0.05f64, period + 10..400)
                    .prop_flat_map(move |(base_price, volatility, data_len)| {
                        (
                            Just(base_price),
                            Just(volatility),
                            Just(data_len),
                            prop::collection::vec((-1f64..1f64), data_len),
                            prop::collection::vec((0f64..2f64), data_len),
                        )
                    })
                    .prop_map(
                        move |(base_price, volatility, data_len, changes, spreads)| {
                            let mut high = Vec::with_capacity(data_len);
                            let mut low = Vec::with_capacity(data_len);
                            let mut current_price = base_price;

                            for i in 0..data_len {
                                let change = changes[i] * volatility * current_price;
                                current_price = (current_price + change).max(10.0);

                                let spread = current_price * 0.01 * spreads[i];
                                let daily_high = current_price + spread;
                                let daily_low = current_price - spread;

                                high.push(daily_high);
                                low.push(daily_low.max(1.0));
                            }

                            (high, low)
                        },
                    ),
                Just(period),
            )
        });

        proptest::test_runner::TestRunner::default().run(&strat, |((high, low), period)| {
            let params = DmParams {
                period: Some(period),
            };
            let input = DmInput::from_slices(&high, &low, params);

            let DmOutput {
                plus: out_plus,
                minus: out_minus,
            } = dm_with_kernel(&input, kernel)?;

            let DmOutput {
                plus: ref_plus,
                minus: ref_minus,
            } = dm_with_kernel(&input, Kernel::Scalar)?;

            prop_assert_eq!(out_plus.len(), high.len());
            prop_assert_eq!(out_minus.len(), high.len());

            let warmup_period = period - 1;
            for i in 0..warmup_period {
                prop_assert!(
                    out_plus[i].is_nan(),
                    "Plus value at index {} should be NaN during warmup",
                    i
                );
                prop_assert!(
                    out_minus[i].is_nan(),
                    "Minus value at index {} should be NaN during warmup",
                    i
                );
            }

            for i in warmup_period..high.len() {
                if !out_plus[i].is_nan() {
                    prop_assert!(
                        out_plus[i] >= -1e-9,
                        "Plus DM at index {} is negative: {}",
                        i,
                        out_plus[i]
                    );
                }
                if !out_minus[i].is_nan() {
                    prop_assert!(
                        out_minus[i] >= -1e-9,
                        "Minus DM at index {} is negative: {}",
                        i,
                        out_minus[i]
                    );
                }
            }

            const MAX_ULP: i64 = 3;
            for i in 0..high.len() {
                let plus_y = out_plus[i];
                let plus_r = ref_plus[i];
                let minus_y = out_minus[i];
                let minus_r = ref_minus[i];

                if plus_y.is_nan() {
                    prop_assert!(
                        plus_r.is_nan(),
                        "Plus kernel mismatch at {}: {} vs NaN",
                        i,
                        plus_r
                    );
                } else {
                    let plus_y_bits = plus_y.to_bits();
                    let plus_r_bits = plus_r.to_bits();
                    let plus_ulp_diff = (plus_y_bits as i64).wrapping_sub(plus_r_bits as i64).abs();

                    prop_assert!(
                        plus_ulp_diff <= MAX_ULP,
                        "Plus kernel mismatch at {}: {} vs {} (ULP diff: {})",
                        i,
                        plus_y,
                        plus_r,
                        plus_ulp_diff
                    );
                }

                if minus_y.is_nan() {
                    prop_assert!(
                        minus_r.is_nan(),
                        "Minus kernel mismatch at {}: {} vs NaN",
                        i,
                        minus_r
                    );
                } else {
                    let minus_y_bits = minus_y.to_bits();
                    let minus_r_bits = minus_r.to_bits();
                    let minus_ulp_diff = (minus_y_bits as i64)
                        .wrapping_sub(minus_r_bits as i64)
                        .abs();

                    prop_assert!(
                        minus_ulp_diff <= MAX_ULP,
                        "Minus kernel mismatch at {}: {} vs {} (ULP diff: {})",
                        i,
                        minus_y,
                        minus_r,
                        minus_ulp_diff
                    );
                }
            }

            let all_high_equal = high.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10);
            let all_low_equal = low.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10);

            if all_high_equal && all_low_equal {
                for i in (period * 2).min(high.len() - 1)..high.len() {
                    if !out_plus[i].is_nan() {
                        prop_assert!(
                            out_plus[i].abs() < 1e-6,
                            "Plus DM should be near zero for constant data at {}: {}",
                            i,
                            out_plus[i]
                        );
                    }
                    if !out_minus[i].is_nan() {
                        prop_assert!(
                            out_minus[i].abs() < 1e-6,
                            "Minus DM should be near zero for constant data at {}: {}",
                            i,
                            out_minus[i]
                        );
                    }
                }
            }

            Ok(())
        })?;

        Ok(())
    }

    generate_all_dm_tests!(
        check_dm_partial_params,
        check_dm_default_candles,
        check_dm_with_slice_data,
        check_dm_zero_period,
        check_dm_period_exceeds_data_length,
        check_dm_not_enough_valid_data,
        check_dm_all_values_nan,
        check_dm_with_slice_reinput,
        check_dm_known_values,
        check_dm_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_dm_tests!(check_dm_property);

    #[test]
    fn test_dm_into_matches_api() -> Result<(), Box<dyn std::error::Error>> {
        let n = 256usize;
        let mut high = Vec::with_capacity(n);
        let mut low = Vec::with_capacity(n);
        let mut price = 100.0f64;
        for i in 0..n {
            let drift = ((i % 7) as i32 - 3) as f64 * 0.3;
            price = (price + drift).max(1.0);
            let spread = 0.5 + 0.1 * ((i % 5) as f64);
            high.push(price + spread);
            low.push((price - spread).max(0.01));
        }

        let input = DmInput::from_slices(&high, &low, DmParams::default());

        let base = dm(&input)?;

        let mut plus = vec![0.0; n];
        let mut minus = vec![0.0; n];
        #[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
        dm_into(&input, &mut plus, &mut minus)?;

        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            a == b || (a.is_nan() && b.is_nan())
        }

        assert_eq!(base.plus.len(), n);
        assert_eq!(base.minus.len(), n);
        for i in 0..n {
            assert!(
                eq_or_both_nan(base.plus[i], plus[i]),
                "plus mismatch at {}: base={} into={}",
                i,
                base.plus[i],
                plus[i]
            );
            assert!(
                eq_or_both_nan(base.minus[i], minus[i]),
                "minus mismatch at {}: base={} into={}",
                i,
                base.minus[i],
                minus[i]
            );
        }
        Ok(())
    }

    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = DmBatchBuilder::new().kernel(kernel).apply_candles(&c)?;

        let def = DmParams::default();
        let (row_plus, row_minus) = output.values_for(&def).expect("default row missing");

        assert_eq!(row_plus.len(), c.high.len());
        assert_eq!(row_minus.len(), c.high.len());

        let expected_plus = [
            1410.819956368491,
            1384.04710234217,
            1285.186595032015,
            1199.3875525297283,
            1113.7170130633192,
        ];
        let expected_minus = [
            3602.8631384045057,
            3345.5157713756125,
            3258.5503591344973,
            3025.796762053462,
            3493.668421906786,
        ];
        let start = row_plus.len() - 5;
        for (i, &v) in row_plus[start..].iter().enumerate() {
            assert!((v - expected_plus[i]).abs() < 1e-6);
        }
        for (i, &v) in row_minus[start..].iter().enumerate() {
            assert!((v - expected_minus[i]).abs() < 1e-6);
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let test_configs = vec![
            (2, 10, 2),
            (5, 25, 5),
            (30, 60, 15),
            (2, 5, 1),
            (14, 14, 0),
            (10, 100, 10),
            (100, 200, 50),
        ];

        for (cfg_idx, &(p_start, p_end, p_step)) in test_configs.iter().enumerate() {
            let output = DmBatchBuilder::new()
                .kernel(kernel)
                .period_range(p_start, p_end, p_step)
                .apply_candles(&c)?;

            for (idx, &val) in output.plus.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                let row = idx / output.cols;
                let col = idx % output.cols;
                let combo = &output.combos[row];

                if bits == 0x11111111_11111111 {
                    panic!(
						"[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) in plus \
						 at row {} col {} (flat index {}) with params: period={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.period.unwrap_or(14)
					);
                }

                if bits == 0x22222222_22222222 {
                    panic!(
						"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) in plus \
						 at row {} col {} (flat index {}) with params: period={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.period.unwrap_or(14)
					);
                }

                if bits == 0x33333333_33333333 {
                    panic!(
						"[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) in plus \
						 at row {} col {} (flat index {}) with params: period={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.period.unwrap_or(14)
					);
                }
            }

            for (idx, &val) in output.minus.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                let row = idx / output.cols;
                let col = idx % output.cols;
                let combo = &output.combos[row];

                if bits == 0x11111111_11111111 {
                    panic!(
						"[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) in minus \
						 at row {} col {} (flat index {}) with params: period={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.period.unwrap_or(14)
					);
                }

                if bits == 0x22222222_22222222 {
                    panic!(
						"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) in minus \
						 at row {} col {} (flat index {}) with params: period={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.period.unwrap_or(14)
					);
                }

                if bits == 0x33333333_33333333 {
                    panic!(
						"[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) in minus \
						 at row {} col {} (flat index {}) with params: period={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.period.unwrap_or(14)
					);
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(
        _test: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
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
    gen_batch_tests!(check_batch_no_poison);
}

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;

#[cfg(feature = "python")]
#[pyfunction(name = "dm")]
#[pyo3(signature = (high, low, period, kernel=None))]
pub fn dm_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let h = high.as_slice()?;
    let l = low.as_slice()?;
    if h.len() != l.len() {
        return Err(PyValueError::new_err("high/low length mismatch"));
    }

    let params = DmParams {
        period: Some(period),
    };
    let input = DmInput::from_slices(h, l, params);
    let kern = validate_kernel(kernel, false)?;

    let out_plus = unsafe { PyArray1::<f64>::new(py, [h.len()], false) };
    let out_minus = unsafe { PyArray1::<f64>::new(py, [h.len()], false) };
    let plus_slice = unsafe { out_plus.as_slice_mut()? };
    let minus_slice = unsafe { out_minus.as_slice_mut()? };

    py.allow_threads(|| dm_into_slice(plus_slice, minus_slice, &input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok((out_plus, out_minus))
}

#[cfg(feature = "python")]
#[pyfunction(name = "dm_batch")]
#[pyo3(signature = (high, low, period_range, kernel=None))]
pub fn dm_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let h = high.as_slice()?;
    let l = low.as_slice()?;
    if h.len() != l.len() {
        return Err(PyValueError::new_err("high/low length mismatch"));
    }

    let sweep = DmBatchRange {
        period: period_range,
    };
    let kern = validate_kernel(kernel, true)?;

    let output = py
        .allow_threads(|| dm_batch_with_kernel(h, l, &sweep, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let plus = unsafe { PyArray1::from_vec(py, output.plus).reshape((output.rows, output.cols))? };
    let minus =
        unsafe { PyArray1::from_vec(py, output.minus).reshape((output.rows, output.cols))? };

    let dict = PyDict::new(py);
    dict.set_item("plus", plus)?;
    dict.set_item("minus", minus)?;
    dict.set_item(
        "periods",
        output
            .combos
            .iter()
            .map(|p| p.period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    Ok(dict)
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "dm_cuda_batch_dev")]
#[pyo3(signature = (high_f32, low_f32, period_range, device_id=0))]
pub fn dm_cuda_batch_dev_py(
    py: Python<'_>,
    high_f32: numpy::PyReadonlyArray1<'_, f32>,
    low_f32: numpy::PyReadonlyArray1<'_, f32>,
    period_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<(DeviceArrayF32Py, DeviceArrayF32Py)> {
    use crate::cuda::cuda_available;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let h = high_f32.as_slice()?;
    let l = low_f32.as_slice()?;
    let sweep = DmBatchRange {
        period: period_range,
    };
    let (pair, ctx, dev) = py.allow_threads(|| {
        let cuda = CudaDm::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx = cuda.context_arc();
        let dev = cuda.device_id();
        cuda.dm_batch_dev(h, l, &sweep)
            .map(|(pair, _)| (pair, ctx, dev))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    Ok((
        DeviceArrayF32Py {
            inner: pair.plus,
            _ctx: Some(ctx.clone()),
            device_id: Some(dev),
        },
        DeviceArrayF32Py {
            inner: pair.minus,
            _ctx: Some(ctx),
            device_id: Some(dev),
        },
    ))
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "dm_cuda_many_series_one_param_dev")]
#[pyo3(signature = (high_tm_f32, low_tm_f32, cols, rows, period, device_id=0))]
pub fn dm_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    high_tm_f32: numpy::PyReadonlyArray1<'_, f32>,
    low_tm_f32: numpy::PyReadonlyArray1<'_, f32>,
    cols: usize,
    rows: usize,
    period: usize,
    device_id: usize,
) -> PyResult<(DeviceArrayF32Py, DeviceArrayF32Py)> {
    use crate::cuda::cuda_available;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let h = high_tm_f32.as_slice()?;
    let l = low_tm_f32.as_slice()?;
    let (pair, ctx, dev) = py.allow_threads(|| {
        let cuda = CudaDm::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx = cuda.context_arc();
        let dev = cuda.device_id();
        cuda.dm_many_series_one_param_time_major_dev(h, l, cols, rows, period)
            .map(|pair| (pair, ctx, dev))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    Ok((
        DeviceArrayF32Py {
            inner: pair.plus,
            _ctx: Some(ctx.clone()),
            device_id: Some(dev),
        },
        DeviceArrayF32Py {
            inner: pair.minus,
            _ctx: Some(ctx),
            device_id: Some(dev),
        },
    ))
}

#[cfg(feature = "python")]
#[pyclass(name = "DmStream")]
pub struct DmStreamPy {
    stream: DmStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl DmStreamPy {
    #[new]
    fn new(period: usize) -> PyResult<Self> {
        let s = DmStream::try_new(DmParams {
            period: Some(period),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream: s })
    }
    fn update(&mut self, high: f64, low: f64) -> Option<(f64, f64)> {
        self.stream.update(high, low)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use serde::{Deserialize, Serialize};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen::prelude::*;

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct DmJsOutput {
    pub values: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = dm)]
pub fn dm_js(high: &[f64], low: &[f64], period: usize) -> Result<JsValue, JsValue> {
    if high.len() != low.len() {
        return Err(JsValue::from_str("length mismatch"));
    }
    let input = DmInput::from_slices(
        high,
        low,
        DmParams {
            period: Some(period),
        },
    );

    let mut plus = vec![0.0; high.len()];
    let mut minus = vec![0.0; high.len()];
    dm_into_slice(&mut plus, &mut minus, &input, detect_best_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let mut values = plus;
    values.extend_from_slice(&minus);

    let output = DmJsOutput {
        values,
        rows: 2,
        cols: high.len(),
    };
    serde_wasm_bindgen::to_value(&output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct DmBatchConfig {
    pub period_range: (usize, usize, usize),
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct DmBatchJsOutput {
    pub values: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
    pub periods: Vec<usize>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = dm_batch)]
pub fn dm_batch_unified_js(high: &[f64], low: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    if high.len() != low.len() {
        return Err(JsValue::from_str("length mismatch"));
    }
    let cfg: DmBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = DmBatchRange {
        period: cfg.period_range,
    };
    let out = dm_batch_inner(high, low, &sweep, detect_best_kernel(), false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let mut values = Vec::with_capacity(out.plus.len() + out.minus.len());
    values.extend_from_slice(&out.plus);
    values.extend_from_slice(&out.minus);

    let periods = out
        .combos
        .iter()
        .map(|p| p.period.unwrap())
        .collect::<Vec<_>>();

    let js = DmBatchJsOutput {
        values,
        rows: out.rows * 2,
        cols: out.cols,
        periods,
    };
    serde_wasm_bindgen::to_value(&js)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn dm_alloc(len: usize) -> *mut f64 {
    let mut v = Vec::<f64>::with_capacity(len);
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn dm_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = dm_into)]
pub fn dm_into_js(
    high_ptr: *const f64,
    low_ptr: *const f64,
    plus_ptr: *mut f64,
    minus_ptr: *mut f64,
    len: usize,
    period: usize,
) -> Result<(), JsValue> {
    if high_ptr.is_null() || low_ptr.is_null() || plus_ptr.is_null() || minus_ptr.is_null() {
        return Err(JsValue::from_str("null pointer"));
    }
    unsafe {
        let h = std::slice::from_raw_parts(high_ptr, len);
        let l = std::slice::from_raw_parts(low_ptr, len);
        let input = DmInput::from_slices(
            h,
            l,
            DmParams {
                period: Some(period),
            },
        );
        let plus = std::slice::from_raw_parts_mut(plus_ptr, len);
        let minus = std::slice::from_raw_parts_mut(minus_ptr, len);
        dm_into_slice(plus, minus, &input, detect_best_kernel())
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}
