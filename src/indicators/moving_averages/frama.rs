//! # Fractal Adaptive Moving Average (FRAMA)
//!
//! An adaptive moving average that adjusts its smoothing factor using fractal dimension analysis,
//! calculated over a window using high, low, and close price data. Designed to match alma.rs in
//! interface, kernel handling, batch sweep, builder/stream API, and test coverage. SIMD kernels
//! (AVX2/AVX512) are stubbed for API parity only.
//!
//! ## Parameters
//! - **window**: Lookback window (even, default 10).
//! - **sc**: Slow constant (default 300).
//! - **fc**: Fast constant (default 1).
//!
//! ## Errors
//! - **AllValuesNaN**: frama: All input data values are `NaN`.
//! - **InvalidWindow**: frama: `window` is zero or exceeds the data length.
//! - **NotEnoughValidData**: frama: Not enough valid data points for the requested `window`.
//!
//! ## Returns
//! - **`Ok(FramaOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(FramaError)`** otherwise.

use crate::utilities::aligned_vector::AlignedVec;
use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use std::hint::unlikely;
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;

impl<'a> AsRef<[f64]> for FramaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            FramaData::Candles { candles } => candles.select_candle_field("close").unwrap(),
            FramaData::Slices { close, .. } => close,
        }
    }
}

#[inline(always)]
unsafe fn seed_sma(close: &[f64], first: usize, win: usize, out: &mut [f64]) {
    let mut sum = 0.0;
    for k in 0..win {
        sum += *close.get_unchecked(first + k);
    }
    *out.get_unchecked_mut(first + win - 1) = sum / win as f64;
}

#[derive(Debug, Clone)]
pub enum FramaData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct FramaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct FramaParams {
    pub window: Option<usize>,
    pub sc: Option<usize>,
    pub fc: Option<usize>,
}

impl Default for FramaParams {
    fn default() -> Self {
        Self {
            window: Some(10),
            sc: Some(300),
            fc: Some(1),
        }
    }
}

#[derive(Debug, Clone)]
pub struct FramaInput<'a> {
    pub data: FramaData<'a>,
    pub params: FramaParams,
}

impl<'a> FramaInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: FramaParams) -> Self {
        Self {
            data: FramaData::Candles { candles },
            params,
        }
    }
    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: FramaParams,
    ) -> Self {
        Self {
            data: FramaData::Slices { high, low, close },
            params,
        }
    }
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, FramaParams::default())
    }
    #[inline]
    pub fn get_window(&self) -> usize {
        self.params.window.unwrap_or(10)
    }
    #[inline]
    pub fn get_sc(&self) -> usize {
        self.params.sc.unwrap_or(300)
    }
    #[inline]
    pub fn get_fc(&self) -> usize {
        self.params.fc.unwrap_or(1)
    }

    #[inline]
    pub fn slices(&self) -> (&'a [f64], &'a [f64], &'a [f64]) {
        match &self.data {
            FramaData::Candles { candles } => (
                candles.select_candle_field("high").unwrap(),
                candles.select_candle_field("low").unwrap(),
                candles.select_candle_field("close").unwrap(),
            ),
            FramaData::Slices { high, low, close } => (*high, *low, *close),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct FramaBuilder {
    window: Option<usize>,
    sc: Option<usize>,
    fc: Option<usize>,
    kernel: Kernel,
}

impl Default for FramaBuilder {
    fn default() -> Self {
        Self {
            window: None,
            sc: None,
            fc: None,
            kernel: Kernel::Auto,
        }
    }
}
impl FramaBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn window(mut self, n: usize) -> Self {
        self.window = Some(n);
        self
    }
    #[inline(always)]
    pub fn sc(mut self, x: usize) -> Self {
        self.sc = Some(x);
        self
    }
    #[inline(always)]
    pub fn fc(mut self, x: usize) -> Self {
        self.fc = Some(x);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<FramaOutput, FramaError> {
        let p = FramaParams {
            window: self.window,
            sc: self.sc,
            fc: self.fc,
        };
        let i = FramaInput::from_candles(c, p);
        frama_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<FramaOutput, FramaError> {
        let p = FramaParams {
            window: self.window,
            sc: self.sc,
            fc: self.fc,
        };
        let i = FramaInput::from_slices(high, low, close, p);
        frama_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<FramaStream, FramaError> {
        let p = FramaParams {
            window: self.window,
            sc: self.sc,
            fc: self.fc,
        };
        FramaStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum FramaError {
    #[error("frama: All values are NaN.")]
    AllValuesNaN,
    #[error("frama: Invalid window: window = {window}, data length = {data_len}")]
    InvalidWindow { window: usize, data_len: usize },
    #[error("frama: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

#[inline]
pub fn frama(input: &FramaInput) -> Result<FramaOutput, FramaError> {
    frama_with_kernel(input, Kernel::Auto)
}

pub fn frama_with_kernel(input: &FramaInput, kernel: Kernel) -> Result<FramaOutput, FramaError> {
    let (high, low, close) = input.slices();
    let len = high.len();
    if len == 0 || low.len() != len || close.len() != len {
        return Err(FramaError::InvalidWindow {
            window: input.get_window(),
            data_len: len,
        });
    }
    let window = input.get_window();
    let sc = input.get_sc();
    let fc = input.get_fc();
    let first = (0..len)
        .find(|&i| !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan())
        .ok_or(FramaError::AllValuesNaN)?;
    if window == 0 || window > len {
        return Err(FramaError::InvalidWindow {
            window,
            data_len: len,
        });
    }
    if (len - first) < window {
        return Err(FramaError::NotEnoughValidData {
            needed: window,
            valid: len - first,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };
    match chosen {
        Kernel::Scalar | Kernel::ScalarBatch => {
            frama_scalar(high, low, close, window, sc, fc, first, len)
        }

        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2 | Kernel::Avx2Batch => unsafe {
            frama_avx2(high, low, close, window, sc, fc, first, len)
        },

        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512 | Kernel::Avx512Batch => unsafe {
            frama_avx512(high, low, close, window, sc, fc, first, len)
        },

        _ => unreachable!("`Auto` must be resolved above"),
    }
}

use core::mem::swap;
use core::mem::MaybeUninit;

#[derive(Copy, Clone)]
struct MonoDeque<const CAP: usize> {
    buf: [usize; CAP],
    head: usize,
    tail: usize,
}
impl<const CAP: usize> MonoDeque<CAP> {
    #[inline(always)]
    const fn new() -> Self {
        Self {
            buf: [0; CAP],
            head: 0,
            tail: 0,
        }
    }
    #[inline(always)]
    fn clear(&mut self) {
        self.head = 0;
        self.tail = 0;
    }
    #[inline(always)]
    fn is_empty(&self) -> bool {
        self.head == self.tail
    }

    #[inline(always)]
    unsafe fn front(&self) -> usize {
        *self.buf.get_unchecked(self.head)
    }

    #[inline(always)]
    fn expire(&mut self, idx_out: usize) {
        if !self.is_empty() && unsafe { self.front() } == idx_out {
            self.head = (self.head + 1) % CAP;
        }
    }

    #[inline(always)]
    unsafe fn push_max(&mut self, idx: usize, data: &[f64]) {
        while !self.is_empty() {
            let last = self.buf[(self.tail + CAP - 1) % CAP];
            if *data.get_unchecked(last) >= *data.get_unchecked(idx) {
                break;
            }
            self.tail = (self.tail + CAP - 1) % CAP;
        }
        self.buf[self.tail] = idx;
        self.tail = (self.tail + 1) % CAP;
    }

    #[inline(always)]
    unsafe fn push_min(&mut self, idx: usize, data: &[f64]) {
        while !self.is_empty() {
            let last = self.buf[(self.tail + CAP - 1) % CAP];
            if *data.get_unchecked(last) <= *data.get_unchecked(idx) {
                break;
            }
            self.tail = (self.tail + CAP - 1) % CAP;
        }
        self.buf[self.tail] = idx;
        self.tail = (self.tail + 1) % CAP;
    }
}

#[inline(always)]
fn frama_scalar_deque(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    mut window: usize,
    sc: usize,
    fc: usize,
    first: usize,
    len: usize,
    out: &mut [f64],
) -> Result<(), FramaError> {
    if window & 1 == 1 {
        window += 1;
    }
    let half = window / 2;
    const MAX_W: usize = 1024;
    assert!(window <= MAX_W, "window bigger than CAP");

    let mut d_full_max: MonoDeque<MAX_W> = MonoDeque::new();
    let mut d_full_min: MonoDeque<MAX_W> = MonoDeque::new();
    let mut d_left_max: MonoDeque<MAX_W> = MonoDeque::new();
    let mut d_left_min: MonoDeque<MAX_W> = MonoDeque::new();
    let mut d_right_max: MonoDeque<MAX_W> = MonoDeque::new();
    let mut d_right_min: MonoDeque<MAX_W> = MonoDeque::new();

    unsafe {
        for idx in first..(first + window) {
            if !high[idx].is_nan() && !low[idx].is_nan() {
                d_full_max.push_max(idx, high);
                d_full_min.push_min(idx, low);
                if idx < first + half {
                    d_left_max.push_max(idx, high);
                    d_left_min.push_min(idx, low);
                } else {
                    d_right_max.push_max(idx, high);
                    d_right_min.push_min(idx, low);
                }
            }
        }
    }

    let w_ln = (2.0 / (sc as f64 + 1.0)).ln();
    let sc_lim = 2.0 / (sc as f64 + 1.0);
    let mut d_prev = 1.0;

    let mut pm1 = f64::NAN;
    let mut pm2 = f64::NAN;
    let mut pm3 = f64::NAN;
    let mut pn1 = f64::NAN;
    let mut pn2 = f64::NAN;
    let mut pn3 = f64::NAN;

    let mut half_progress = 0usize;

    for i in (first + window)..len {
        let idx_out = i - window;
        d_full_max.expire(idx_out);
        d_full_min.expire(idx_out);
        d_left_max.expire(idx_out);
        d_left_min.expire(idx_out);
        d_right_max.expire(idx_out + half);
        d_right_min.expire(idx_out + half);

        let newest = i - 1;
        if !high[newest].is_nan() && !low[newest].is_nan() {
            unsafe {
                d_full_max.push_max(newest, high);
                d_full_min.push_min(newest, low);

                if newest < (idx_out + half) {
                    d_left_max.push_max(newest, high);
                    d_left_min.push_min(newest, low);
                } else {
                    d_right_max.push_max(newest, high);
                    d_right_min.push_min(newest, low);
                }
            }
        }
        fn front_or(
            dq_max: &MonoDeque<MAX_W>,
            dq_min: &MonoDeque<MAX_W>,
            prev_max: &mut f64,
            prev_min: &mut f64,
            high: &[f64],
            low: &[f64],
        ) -> (f64, f64) {
            let maxv = if !dq_max.is_empty() {
                high[unsafe { dq_max.front() }]
            } else {
                *prev_max
            };
            let minv = if !dq_min.is_empty() {
                low[unsafe { dq_min.front() }]
            } else {
                *prev_min
            };
            *prev_max = maxv;
            *prev_min = minv;
            (maxv, minv)
        }
        let (max1, min1) = front_or(&d_right_max, &d_right_min, &mut pm1, &mut pn1, high, low);
        let (max2, min2) = front_or(&d_left_max, &d_left_min, &mut pm2, &mut pn2, high, low);
        let (max3, min3) = front_or(&d_full_max, &d_full_min, &mut pm3, &mut pn3, high, low);

        if !(high[i].is_nan() || low[i].is_nan() || close[i].is_nan()) {
            let n1 = (max1 - min1) / (half as f64);
            let n2 = (max2 - min2) / (half as f64);
            let n3 = (max3 - min3) / (window as f64);

            let d_cur = if n1 > 0.0 && n2 > 0.0 && n3 > 0.0 {
                ((n1 + n2).ln() - n3.ln()) / std::f64::consts::LN_2
            } else {
                d_prev
            };
            d_prev = d_cur;

            let mut alpha0 = (w_ln * (d_cur - 1.0)).exp();
            if alpha0 < 0.1 {
                alpha0 = 0.1;
            }
            if alpha0 > 1.0 {
                alpha0 = 1.0;
            }
            let old_n = (2.0 - alpha0) / alpha0;
            let new_n = (sc - fc) as f64 * ((old_n - 1.0) / (sc as f64 - 1.0)) + fc as f64;
            let mut alpha = 2.0 / (new_n + 1.0);
            if alpha < sc_lim {
                alpha = sc_lim;
            }
            if alpha > 1.0 {
                alpha = 1.0;
            }

            out[i] = alpha * close[i] + (1.0 - alpha) * out[i - 1];
        } else {
            out[i] = out[i - 1];
        }

        half_progress += 1;
        if half_progress == half {
            swap(&mut d_left_max, &mut d_right_max);
            swap(&mut d_left_min, &mut d_right_min);
            d_right_max.clear();
            d_right_min.clear();
            half_progress = 0;
        }
    }

    Ok(())
}

#[inline(always)]
pub fn frama_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    window: usize,
    sc: usize,
    fc: usize,
    first: usize,
    len: usize,
) -> Result<FramaOutput, FramaError> {
    let mut out = vec![f64::NAN; len];

    let mut win = window;
    if win & 1 == 1 {
        win += 1;
    }
    let seed = close[first..first + win].iter().sum::<f64>() / win as f64;
    out[first + win - 1] = seed;

    if win <= 32 {
        unsafe {
            frama_small_scan(high, low, close, win, sc, fc, first, len, &mut out)?;
        }
    } else {
        frama_scalar_deque(high, low, close, win, sc, fc, first, len, &mut out)?;
    }
    Ok(FramaOutput { values: out })
}

#[inline(always)]
unsafe fn frama_small_scan(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    win: usize,
    sc: usize,
    fc: usize,
    first: usize,
    len: usize,
    out: &mut [f64],
) -> Result<(), FramaError> {
    let half = win >> 1;
    let win_f64 = win as f64;
    let half_f64 = half as f64;
    let w_ln = (2.0 / (sc as f64 + 1.0)).ln();
    let sc_floor = 2.0 / (sc as f64 + 1.0);
    let mut d_prev = 1.0_f64;

    for i in (first + win)..len {
        let seg_start = i - win;
        let mid = seg_start + half;

        let mut max1 = f64::MIN;
        let mut min1 = f64::MAX;
        let mut max2 = f64::MIN;
        let mut min2 = f64::MAX;

        let mut j = seg_start;
        while j + 1 < mid {
            let h0 = *high.get_unchecked(j);
            let h1 = *high.get_unchecked(j + 1);
            let l0 = *low.get_unchecked(j);
            let l1 = *low.get_unchecked(j + 1);
            max2 = f64::max(max2, f64::max(h0, h1));
            min2 = f64::min(min2, f64::min(l0, l1));
            j += 2;
        }
        if j < mid {
            max2 = f64::max(max2, *high.get_unchecked(j));
            min2 = f64::min(min2, *low.get_unchecked(j));
        }

        j = mid;
        while j + 1 < i {
            let h0 = *high.get_unchecked(j);
            let h1 = *high.get_unchecked(j + 1);
            let l0 = *low.get_unchecked(j);
            let l1 = *low.get_unchecked(j + 1);
            max1 = f64::max(max1, f64::max(h0, h1));
            min1 = f64::min(min1, f64::min(l0, l1));
            j += 2;
        }
        if j < i {
            max1 = f64::max(max1, *high.get_unchecked(j));
            min1 = f64::min(min1, *low.get_unchecked(j));
        }

        let max3 = f64::max(max1, max2);
        let min3 = f64::min(min1, min2);

        let n1 = (max1 - min1) / half_f64;
        let n2 = (max2 - min2) / half_f64;
        let n3 = (max3 - min3) / win_f64;

        let d_cur = if n1 > 0.0 && n2 > 0.0 && n3 > 0.0 {
            ((n1 + n2).ln() - n3.ln()) / std::f64::consts::LN_2
        } else {
            d_prev
        };
        d_prev = d_cur;

        let mut alpha0 = (w_ln * (d_cur - 1.0)).exp().clamp(0.1, 1.0);
        let old_n = (2.0 - alpha0) / alpha0;
        let new_n = (sc - fc) as f64 * ((old_n - 1.0) / (sc as f64 - 1.0)) + fc as f64;
        let alpha = (2.0 / (new_n + 1.0)).clamp(sc_floor, 1.0);

        out[i] = (*close.get_unchecked(i)).mul_add(alpha, (1.0 - alpha) * out[i - 1]);
    }
    Ok(())
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn hmax_pd256(v: __m256d) -> f64 {
    let hi = _mm256_extractf128_pd::<1>(v);
    let lo = _mm256_castpd256_pd128(v);
    let m = _mm_max_pd(hi, lo);
    let m = _mm_max_pd(m, _mm_permute_pd::<0b01>(m));
    _mm_cvtsd_f64(m)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn hmin_pd256(v: __m256d) -> f64 {
    let hi = _mm256_extractf128_pd::<1>(v);
    let lo = _mm256_castpd256_pd128(v);
    let m = _mm_min_pd(hi, lo);
    let m = _mm_min_pd(m, _mm_permute_pd::<0b01>(m));
    _mm_cvtsd_f64(m)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn frama_avx2_small<const WIN: usize>(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sc: usize,
    fc: usize,
    first: usize,
    len: usize,
    out: &mut [f64],
) {
    const LANES: usize = 4;
    const LN2: f64 = std::f64::consts::LN_2;

    let half = WIN / 2;
    let win_f64 = WIN as f64;
    let half_f64 = half as f64;
    let w_ln = (2.0 / (sc as f64 + 1.0)).ln();
    let sc_floor = 2.0 / (sc as f64 + 1.0);
    let mut d_prev = 1.0;

    for i in (first + WIN)..len {
        if i + 1 < len {
            _mm_prefetch(high.as_ptr().add(i + 1 - WIN) as *const i8, _MM_HINT_T0);
            _mm_prefetch(low.as_ptr().add(i + 1 - WIN) as *const i8, _MM_HINT_T0);
            _mm_prefetch(close.as_ptr().add(i + 1) as *const i8, _MM_HINT_T0);
        }

        if unlikely(
            (*high.get_unchecked(i)).is_nan()
                || (*low.get_unchecked(i)).is_nan()
                || (*close.get_unchecked(i)).is_nan(),
        ) {
            *out.get_unchecked_mut(i) = *out.get_unchecked(i - 1);
            continue;
        }

        let mut v_max_l = _mm256_set1_pd(f64::MIN);
        let mut v_min_l = _mm256_set1_pd(f64::MAX);
        let mut idx_l = i - WIN;

        for _ in 0..(half / LANES) {
            let h = _mm256_loadu_pd(high.as_ptr().add(idx_l));
            let l = _mm256_loadu_pd(low.as_ptr().add(idx_l));
            v_max_l = _mm256_max_pd(v_max_l, h);
            v_min_l = _mm256_min_pd(v_min_l, l);
            idx_l += LANES;
        }

        let mut max_l = hmax_pd256(v_max_l);
        let mut min_l = hmin_pd256(v_min_l);

        for j in idx_l..(i - half) {
            let h = *high.get_unchecked(j);
            let l = *low.get_unchecked(j);
            max_l = max_l.max(h);
            min_l = min_l.min(l);
        }

        let mut v_max_r = _mm256_set1_pd(f64::MIN);
        let mut v_min_r = _mm256_set1_pd(f64::MAX);
        let mut idx_r = i - half;

        for _ in 0..(half / LANES) {
            let h = _mm256_loadu_pd(high.as_ptr().add(idx_r));
            let l = _mm256_loadu_pd(low.as_ptr().add(idx_r));
            v_max_r = _mm256_max_pd(v_max_r, h);
            v_min_r = _mm256_min_pd(v_min_r, l);
            idx_r += LANES;
        }

        let mut max_r = hmax_pd256(v_max_r);
        let mut min_r = hmin_pd256(v_min_r);

        for j in idx_r..i {
            let h = *high.get_unchecked(j);
            let l = *low.get_unchecked(j);
            max_r = max_r.max(h);
            min_r = min_r.min(l);
        }

        let max_w = max_l.max(max_r);
        let min_w = min_l.min(min_r);

        let n1 = (max_r - min_r) / half_f64;
        let n2 = (max_l - min_l) / half_f64;
        let n3 = (max_w - min_w) / win_f64;

        let d = if n1 > 0.0 && n2 > 0.0 && n3 > 0.0 {
            ((n1 + n2).ln() - n3.ln()) / LN2
        } else {
            d_prev
        };
        d_prev = d;

        let mut a0 = (w_ln * (d - 1.0)).exp().clamp(0.1, 1.0);
        let old_n = (2.0 - a0) / a0;
        let new_n = (sc - fc) as f64 * ((old_n - 1.0) / (sc as f64 - 1.0)) + fc as f64;
        let alpha = (2.0 / (new_n + 1.0)).clamp(sc_floor, 1.0);

        *out.get_unchecked_mut(i) =
            (*close.get_unchecked(i)).mul_add(alpha, (1.0 - alpha) * *out.get_unchecked(i - 1));
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn frama_avx512_small<const WIN: usize>(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sc: usize,
    fc: usize,
    first: usize,
    len: usize,
    out: &mut [f64],
) {
    const LANES: usize = 8;
    const LN2: f64 = std::f64::consts::LN_2;

    let half = WIN / 2;
    let vec_cnt = half / LANES;
    let tail = (half & (LANES - 1)) as i32;
    let mask = (1u8 << tail) - 1;

    let w_ln = (2.0 / (sc as f64 + 1.0)).ln();
    let sc_floor = 2.0 / (sc as f64 + 1.0);
    let win_f64 = WIN as f64;
    let half_f64 = half as f64;

    let v_min_init = _mm512_set1_pd(f64::MIN);
    let v_max_init = _mm512_set1_pd(f64::MAX);

    let mut d_prev = 1.0;

    for i in (first + WIN)..len {
        if i + 1 < len {
            _mm_prefetch(high.as_ptr().add(i + 1 - WIN) as *const i8, _MM_HINT_T0);
            _mm_prefetch(low.as_ptr().add(i + 1 - WIN) as *const i8, _MM_HINT_T0);
            _mm_prefetch(close.as_ptr().add(i + 1) as *const i8, _MM_HINT_T0);
        }

        if unlikely(
            (*high.get_unchecked(i)).is_nan()
                || (*low.get_unchecked(i)).is_nan()
                || (*close.get_unchecked(i)).is_nan(),
        ) {
            *out.get_unchecked_mut(i) = *out.get_unchecked(i - 1);
            continue;
        }

        let mut v_max_l = v_min_init;
        let mut v_min_l = v_max_init;
        let base_l = i - WIN;

        for k in 0..vec_cnt {
            let off = base_l + k * LANES;
            let h = _mm512_loadu_pd(high.as_ptr().add(off));
            let l = _mm512_loadu_pd(low.as_ptr().add(off));
            v_max_l = _mm512_max_pd(v_max_l, h);
            v_min_l = _mm512_min_pd(v_min_l, l);
        }

        if tail != 0 {
            let off = base_l + vec_cnt * LANES;
            let h_tail =
                _mm512_mask_loadu_pd(_mm512_set1_pd(f64::MIN), mask, high.as_ptr().add(off));
            let l_tail =
                _mm512_mask_loadu_pd(_mm512_set1_pd(f64::MAX), mask, low.as_ptr().add(off));
            v_max_l = _mm512_max_pd(v_max_l, h_tail);
            v_min_l = _mm512_min_pd(v_min_l, l_tail);
        }

        let max_l = _mm512_reduce_max_pd(v_max_l);
        let min_l = _mm512_reduce_min_pd(v_min_l);

        let mut v_max_r = v_min_init;
        let mut v_min_r = v_max_init;
        let base_r = i - half;

        for k in 0..vec_cnt {
            let off = base_r + k * LANES;
            let h = _mm512_loadu_pd(high.as_ptr().add(off));
            let l = _mm512_loadu_pd(low.as_ptr().add(off));
            v_max_r = _mm512_max_pd(v_max_r, h);
            v_min_r = _mm512_min_pd(v_min_r, l);
        }

        if tail != 0 {
            let off = base_r + vec_cnt * LANES;
            let h_tail =
                _mm512_mask_loadu_pd(_mm512_set1_pd(f64::MIN), mask, high.as_ptr().add(off));
            let l_tail =
                _mm512_mask_loadu_pd(_mm512_set1_pd(f64::MAX), mask, low.as_ptr().add(off));
            v_max_r = _mm512_max_pd(v_max_r, h_tail);
            v_min_r = _mm512_min_pd(v_min_r, l_tail);
        }

        let max_r = _mm512_reduce_max_pd(v_max_r);
        let min_r = _mm512_reduce_min_pd(v_min_r);

        let max_w = max_l.max(max_r);
        let min_w = min_l.min(min_r);

        let n1 = (max_r - min_r) / half_f64;
        let n2 = (max_l - min_l) / half_f64;
        let n3 = (max_w - min_w) / win_f64;

        let d = if n1 > 0.0 && n2 > 0.0 && n3 > 0.0 {
            ((n1 + n2).ln() - n3.ln()) / LN2 // Keep exact original formula
        } else {
            d_prev
        };
        d_prev = d;

        let mut a0 = (w_ln * (d - 1.0)).exp().clamp(0.1, 1.0);
        let old_n = (2.0 - a0) / a0;
        let new_n = (sc - fc) as f64 * ((old_n - 1.0) / (sc as f64 - 1.0)) + fc as f64;
        let alpha = (2.0 / (new_n + 1.0)).clamp(sc_floor, 1.0);

        *out.get_unchecked_mut(i) =
            (*close.get_unchecked(i)).mul_add(alpha, (1.0 - alpha) * *out.get_unchecked(i - 1));
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn frama_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    window: usize,
    sc: usize,
    fc: usize,
    first: usize,
    len: usize,
) -> Result<FramaOutput, FramaError> {
    if window <= 32 && window & 1 == 0 {
        let mut out = vec![f64::NAN; len];
        unsafe {
            seed_sma(close, first, window, &mut out);
            match window {
                10 => frama_avx2_small::<10>(high, low, close, sc, fc, first, len, &mut out),
                14 => frama_avx2_small::<14>(high, low, close, sc, fc, first, len, &mut out),
                20 => frama_avx2_small::<20>(high, low, close, sc, fc, first, len, &mut out),
                32 => frama_avx2_small::<32>(high, low, close, sc, fc, first, len, &mut out),
                _ => return frama_scalar(high, low, close, window, sc, fc, first, len),
            }
        }
        Ok(FramaOutput { values: out })
    } else {
        frama_scalar(high, low, close, window, sc, fc, first, len)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn frama_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    window: usize,
    sc: usize,
    fc: usize,
    first: usize,
    len: usize,
) -> Result<FramaOutput, FramaError> {
    if window <= 32 && window & 1 == 0 {
        let mut out = vec![f64::NAN; len];
        unsafe {
            seed_sma(close, first, window, &mut out);
            match window {
                10 => frama_avx512_small::<10>(high, low, close, sc, fc, first, len, &mut out),
                14 => frama_avx512_small::<14>(high, low, close, sc, fc, first, len, &mut out),
                20 => frama_avx512_small::<20>(high, low, close, sc, fc, first, len, &mut out),
                32 => frama_avx512_small::<32>(high, low, close, sc, fc, first, len, &mut out),
                _ => return frama_scalar(high, low, close, window, sc, fc, first, len),
            }
        }
        Ok(FramaOutput { values: out })
    } else {
        frama_scalar(high, low, close, window, sc, fc, first, len)
    }
}

#[derive(Clone, Debug)]
pub struct FramaBatchRange {
    pub window: (usize, usize, usize),
    pub sc: (usize, usize, usize),
    pub fc: (usize, usize, usize),
}
impl Default for FramaBatchRange {
    fn default() -> Self {
        Self {
            window: (10, 32, 0),
            sc: (300, 300, 0),
            fc: (1, 1, 0),
        }
    }
}
#[derive(Clone, Debug, Default)]
pub struct FramaBatchBuilder {
    range: FramaBatchRange,
    kernel: Kernel,
}
impl FramaBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline]
    pub fn window_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.window = (start, end, step);
        self
    }
    #[inline]
    pub fn sc_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.sc = (start, end, step);
        self
    }
    #[inline]
    pub fn fc_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.fc = (start, end, step);
        self
    }
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<FramaBatchOutput, FramaError> {
        frama_batch_with_kernel(high, low, close, &self.range, self.kernel)
    }
    pub fn apply_slice(self, slice: &[f64]) -> Result<FramaBatchOutput, FramaError> {
        self.apply_slices(slice, slice, slice)
    }
    pub fn with_default_slices(
        high: &[f64],
        low: &[f64],
        close: &[f64],
        k: Kernel,
    ) -> Result<FramaBatchOutput, FramaError> {
        FramaBatchBuilder::new()
            .kernel(k)
            .apply_slices(high, low, close)
    }
    pub fn apply_candles(self, c: &Candles) -> Result<FramaBatchOutput, FramaError> {
        let h = c.select_candle_field("high").unwrap();
        let l = c.select_candle_field("low").unwrap();
        let o = c.select_candle_field("close").unwrap();
        self.apply_slices(h, l, o)
    }
    pub fn with_default_candles(c: &Candles) -> Result<FramaBatchOutput, FramaError> {
        FramaBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c)
    }
}
#[derive(Clone, Debug)]
pub struct FramaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<FramaParams>,
    pub rows: usize,
    pub cols: usize,
}
impl FramaBatchOutput {
    pub fn row_for_params(&self, p: &FramaParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.window.unwrap_or(10) == p.window.unwrap_or(10)
                && c.sc.unwrap_or(300) == p.sc.unwrap_or(300)
                && c.fc.unwrap_or(1) == p.fc.unwrap_or(1)
        })
    }
    pub fn values_for(&self, p: &FramaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}
#[inline(always)]
fn expand_grid(r: &FramaBatchRange) -> Vec<FramaParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let windows = axis_usize(r.window);
    let scs = axis_usize(r.sc);
    let fcs = axis_usize(r.fc);
    let mut out = Vec::with_capacity(windows.len() * scs.len() * fcs.len());
    for &w in &windows {
        for &s in &scs {
            for &f in &fcs {
                out.push(FramaParams {
                    window: Some(w),
                    sc: Some(s),
                    fc: Some(f),
                });
            }
        }
    }
    out
}

pub fn frama_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &FramaBatchRange,
    k: Kernel,
) -> Result<FramaBatchOutput, FramaError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(FramaError::InvalidWindow {
                window: 0,
                data_len: 0,
            })
        }
    };
    let simd = match kernel {
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512Batch => Kernel::Avx512,
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    frama_batch_inner(high, low, close, sweep, simd, true)
}

#[inline(always)]
pub fn frama_batch_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &FramaBatchRange,
    kern: Kernel,
) -> Result<FramaBatchOutput, FramaError> {
    frama_batch_inner(high, low, close, sweep, kern, false)
}
#[inline(always)]
pub fn frama_batch_par_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &FramaBatchRange,
    kern: Kernel,
) -> Result<FramaBatchOutput, FramaError> {
    frama_batch_inner(high, low, close, sweep, kern, true)
}
#[inline(always)]
fn frama_batch_inner(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &FramaBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<FramaBatchOutput, FramaError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(FramaError::InvalidWindow {
            window: 0,
            data_len: 0,
        });
    }
    let len = high.len();
    let first = (0..len)
        .find(|&i| !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan())
        .ok_or(FramaError::AllValuesNaN)?;
    let max_w = combos.iter().map(|c| c.window.unwrap()).max().unwrap();
    if len - first < max_w {
        return Err(FramaError::NotEnoughValidData {
            needed: max_w,
            valid: len - first,
        });
    }
    let rows = combos.len();
    let cols = len;
    let mut values = vec![f64::NAN; rows * cols];
    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let p = &combos[row];
        let window = p.window.unwrap();
        let sc = p.sc.unwrap();
        let fc = p.fc.unwrap();

        match kern {
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => frama_row_avx512(high, low, close, first, window, out_row, sc, fc),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => frama_row_avx2(high, low, close, first, window, out_row, sc, fc),
            _ => frama_row_scalar(high, low, close, first, window, out_row, sc, fc),
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
    Ok(FramaBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[derive(Debug, Clone)]
pub struct FramaStream {
    window: usize,
    sc: usize,
    fc: usize,
    n: usize,
    w: f64,
    buffer: Vec<(f64, f64, f64)>,
    head: usize,
    filled: bool,
    last_val: f64,
    d_prev: f64,
    alpha_prev: f64,
}
impl FramaStream {
    pub fn try_new(params: FramaParams) -> Result<Self, FramaError> {
        let window = params.window.unwrap_or(10);
        let sc = params.sc.unwrap_or(300);
        let fc = params.fc.unwrap_or(1);
        if window == 0 {
            return Err(FramaError::InvalidWindow {
                window,
                data_len: 0,
            });
        }
        let mut n = window;
        if n % 2 == 1 {
            n += 1;
        }
        Ok(Self {
            window,
            sc,
            fc,
            n,
            w: (2.0 / (sc as f64 + 1.0)).ln(),
            buffer: vec![(f64::NAN, f64::NAN, f64::NAN); n],
            head: 0,
            filled: false,
            last_val: f64::NAN,
            d_prev: 1.0,
            alpha_prev: 2.0 / (sc as f64 + 1.0),
        })
    }
    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        if !self.filled {
            self.buffer[self.head] = (high, low, close);
            self.head += 1;
            if self.head == self.n {
                self.head = 0;
                self.filled = true;

                let sum: f64 = self.buffer.iter().map(|&(_, _, c)| c).sum();
                self.last_val = sum / self.n as f64;
                return Some(self.last_val);
            }
            return None;
        }

        let half = self.n / 2;
        let mut max1 = f64::MIN;
        let mut min1 = f64::MAX;
        let mut max2 = f64::MIN;
        let mut min2 = f64::MAX;
        let mut max3 = f64::MIN;
        let mut min3 = f64::MAX;

        for j in 0..self.n {
            let (h, l, _) = self.buffer[(self.head + j) % self.n];
            if j < half {
                if h > max2 {
                    max2 = h;
                }
                if l < min2 {
                    min2 = l;
                }
            } else {
                if h > max1 {
                    max1 = h;
                }
                if l < min1 {
                    min1 = l;
                }
            }
            if h > max3 {
                max3 = h;
            }
            if l < min3 {
                min3 = l;
            }
        }

        let n1 = (max1 - min1) / half as f64;
        let n2 = (max2 - min2) / half as f64;
        let n3 = (max3 - min3) / self.n as f64;

        let d = if n1 > 0.0 && n2 > 0.0 && n3 > 0.0 {
            ((n1 + n2).ln() - n3.ln()) / std::f64::consts::LN_2
        } else {
            self.d_prev
        };
        self.d_prev = d;

        let mut old_alpha = (self.w * (d - 1.0)).exp().clamp(0.1, 1.0);
        let old_n = (2.0 - old_alpha) / old_alpha;
        let new_n = ((self.sc as f64 - self.fc as f64) * ((old_n - 1.0) / (self.sc as f64 - 1.0)))
            + self.fc as f64;
        let mut alpha_ = (2.0 / (new_n + 1.0)).clamp(2.0 / (self.sc as f64 + 1.0), 1.0);
        self.alpha_prev = alpha_;

        let out = alpha_ * close + (1.0 - alpha_) * self.last_val;

        self.buffer[self.head] = (high, low, close);
        self.head = (self.head + 1) % self.n;
        self.last_val = out;

        Some(out)
    }
}

#[inline(always)]
pub unsafe fn frama_row_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    window: usize,
    out: &mut [f64],
    sc: usize,
    fc: usize,
) {
    let len = high.len();
    let tmp = frama_scalar(high, low, close, window, sc, fc, first, len).unwrap();
    out.copy_from_slice(&tmp.values);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn frama_row_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    window: usize,
    out: &mut [f64],
    sc: usize,
    fc: usize,
) {
    if window <= 32 && window & 1 == 0 {
        seed_sma(close, first, window, out);
        match window {
            10 => frama_avx2_small::<10>(high, low, close, sc, fc, first, high.len(), out),
            14 => frama_avx2_small::<14>(high, low, close, sc, fc, first, high.len(), out),
            20 => frama_avx2_small::<20>(high, low, close, sc, fc, first, high.len(), out),
            32 => frama_avx2_small::<32>(high, low, close, sc, fc, first, high.len(), out),
            _ => frama_row_scalar(high, low, close, first, window, out, sc, fc),
        }
    } else {
        frama_row_scalar(high, low, close, first, window, out, sc, fc)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn frama_row_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    window: usize,
    out: &mut [f64],
    sc: usize,
    fc: usize,
) {
    if window <= 32 && window & 1 == 0 {
        seed_sma(close, first, window, out);
        match window {
            10 => frama_avx512_small::<10>(high, low, close, sc, fc, first, high.len(), out),
            14 => frama_avx512_small::<14>(high, low, close, sc, fc, first, high.len(), out),
            20 => frama_avx512_small::<20>(high, low, close, sc, fc, first, high.len(), out),
            32 => frama_avx512_small::<32>(high, low, close, sc, fc, first, high.len(), out),
            _ => frama_row_scalar(high, low, close, first, window, out, sc, fc),
        }
    } else {
        frama_row_scalar(high, low, close, first, window, out, sc, fc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::utilities::enums::Kernel;
    use crate::skip_if_unsupported;
    use paste::paste;

    fn check_frama_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = FramaParams {
            window: None,
            sc: None,
            fc: None,
        };
        let input = FramaInput::from_candles(&candles, default_params);
        let output = frama_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }
    fn check_frama_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = FramaInput::from_candles(&candles, FramaParams::default());
        let result = frama_with_kernel(&input, kernel)?;
        let expected_last_five = [
            59337.23056930512,
            59321.607512374605,
            59286.677929994796,
            59268.00202402624,
            59160.03888720062,
        ];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-1,
                "[{}] FRAMA {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }
    fn check_frama_zero_window(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = FramaParams {
            window: Some(0),
            sc: None,
            fc: None,
        };
        let input = FramaInput::from_slices(&input_data, &input_data, &input_data, params);
        let res = frama_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] FRAMA should fail with zero window",
            test_name
        );
        Ok(())
    }
    fn check_frama_window_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = FramaParams {
            window: Some(10),
            sc: None,
            fc: None,
        };
        let input = FramaInput::from_slices(&data_small, &data_small, &data_small, params);
        let res = frama_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] FRAMA should fail with window exceeding length",
            test_name
        );
        Ok(())
    }
    fn check_frama_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = FramaParams {
            window: Some(9),
            sc: None,
            fc: None,
        };
        let input = FramaInput::from_slices(&single_point, &single_point, &single_point, params);
        let res = frama_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] FRAMA should fail with insufficient data",
            test_name
        );
        Ok(())
    }
    fn check_frama_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let nan_data = [f64::NAN, f64::NAN, f64::NAN];
        let params = FramaParams::default();
        let input = FramaInput::from_slices(&nan_data, &nan_data, &nan_data, params);
        let res = frama_with_kernel(&input, kernel);
        assert!(res.is_err());
        Ok(())
    }
    fn check_frama_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let high = candles.select_candle_field("high").unwrap();
        let low = candles.select_candle_field("low").unwrap();
        let close = candles.select_candle_field("close").unwrap();
        let period = 10;
        let sc = 300;
        let fc = 1;
        let input = FramaInput::from_slices(
            high,
            low,
            close,
            FramaParams {
                window: Some(period),
                sc: Some(sc),
                fc: Some(fc),
            },
        );
        let batch_output = frama_with_kernel(&input, kernel)?.values;
        let mut stream = FramaStream::try_new(FramaParams {
            window: Some(period),
            sc: Some(sc),
            fc: Some(fc),
        })?;
        let mut stream_values = Vec::with_capacity(close.len());
        for ((&h, &l), &c) in high.iter().zip(low.iter()).zip(close.iter()) {
            match stream.update(h, l, c) {
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
                diff < 1e-7,
                "[{}] FRAMA streaming mismatch at idx {}: batch={}, stream={}",
                test_name,
                i,
                b,
                s
            );
        }
        Ok(())
    }
    fn check_frama_default_candles(test: &str, k: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(k, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let input = FramaInput::with_default_candles(&c);
        match input.data {
            FramaData::Candles { .. } => {}
            _ => panic!("Expected FramaData::Candles"),
        }
        let out = frama_with_kernel(&input, k)?;
        assert_eq!(out.values.len(), c.close.len());
        Ok(())
    }

    macro_rules! generate_all_frama_tests {
        ($($test_fn:ident),*) => {
            paste! {
                $(
                    #[test]
                    fn [<$test_fn _scalar_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar);
                    }
                )*
                $(
                    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                    #[test]
                    fn [<$test_fn _avx2_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2);
                    }
                )*
                $(
                    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                    #[test]
                    fn [<$test_fn _avx512_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512);
                    }
                )*
            }
        }
    }
    generate_all_frama_tests!(
        check_frama_partial_params,
        check_frama_accuracy,
        check_frama_zero_window,
        check_frama_window_exceeds_length,
        check_frama_very_small_dataset,
        check_frama_all_nan,
        check_frama_streaming,
        check_frama_default_candles
    );
    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = FramaBatchBuilder::new().kernel(kernel).apply_candles(&c)?;
        let def = FramaParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        let expected = [
            59337.23056930512,
            59321.607512374605,
            59286.677929994796,
            59268.00202402624,
            59160.03888720062,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-1,
                "[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
            );
        }
        Ok(())
    }
    macro_rules! gen_batch_tests {
        ($fn_name:ident) => {
            paste! {
                #[test]
                fn [<$fn_name _scalar>]() {
                    let _ = $fn_name(stringify!([<$fn_name _scalar>]), Kernel::ScalarBatch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test]
                fn [<$fn_name _avx2>]() {
                    let _ = $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test]
                fn [<$fn_name _avx512>]() {
                    let _ = $fn_name(stringify!([<$fn_name _avx512>]), Kernel::Avx512Batch);
                }
                #[test]
                fn [<$fn_name _auto_detect>]() {
                    let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto);
                }
            }
        };
    }
}