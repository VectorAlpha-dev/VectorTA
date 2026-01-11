//! # Gator Oscillator (GATOR)
//!
//! The Gator Oscillator is based on Bill Williams' Alligator indicator. It calculates three exponentially smoothed averages (Jaws, Teeth, and Lips) of a given source, then produces two lines: `upper = abs(Jaws - Teeth)` and `lower = -abs(Teeth - Lips)`. Their 1-period momentum changes are also reported.
//!
//! ## Parameters
//! - **jaws_length**: EMA length for Jaws (default: 13)
//! - **jaws_shift**: Shift Jaws forward (default: 8)
//! - **teeth_length**: EMA length for Teeth (default: 8)
//! - **teeth_shift**: Shift Teeth forward (default: 5)
//! - **lips_length**: EMA length for Lips (default: 5)
//! - **lips_shift**: Shift Lips forward (default: 3)
//!
//! ## Inputs
//! - Single data slice (typically close prices)
//!
//! ## Returns
//! - `Ok(GatorOscOutput)` containing:
//!   - `upper`: Upper oscillator values (Vec<f64>)
//!   - `lower`: Lower oscillator values (Vec<f64>)
//!   - `upper_change`: Upper momentum changes (Vec<f64>)
//!   - `lower_change`: Lower momentum changes (Vec<f64>)
//! - Output lengths match input data length with NaN padding for warmup period
//!
//! ## Developer Notes
//! - **SIMD status**: AVX2/AVX512 update the three EMA states in packed lanes per step (broadcast price); modest wins (>5% at 100k) since the time-axis is sequential. Runtime detects and falls back to scalar when unsupported.
//! - **Batch status**: Row-specific optimizations not added; batch executes per-row single-series kernels (benefits from SIMD when enabled). Little cross-row reuse available beyond identical lengths with different shifts.
//! - **Streaming**: O(1) per update via small EMA rings; exact and matches batch warmups.
//! - **Memory optimization**: ✅ Uses alloc_with_nan_prefix (zero-copy) for all four outputs (lines 280-283)
//! - **Batch operations**: ✅ Implemented with parallel processing support
//! - **Decision log**: SIMD enabled (AVX2/AVX512); CUDA kernels present for batch + many-series; GPU path used when available but scalar remains the reference for correctness.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum GatorOscData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

impl<'a> AsRef<[f64]> for GatorOscInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            GatorOscData::Slice(slice) => slice,
            GatorOscData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GatorOscOutput {
    pub upper: Vec<f64>,
    pub lower: Vec<f64>,
    pub upper_change: Vec<f64>,
    pub lower_change: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct GatorOscParams {
    pub jaws_length: Option<usize>,
    pub jaws_shift: Option<usize>,
    pub teeth_length: Option<usize>,
    pub teeth_shift: Option<usize>,
    pub lips_length: Option<usize>,
    pub lips_shift: Option<usize>,
}

impl Default for GatorOscParams {
    fn default() -> Self {
        Self {
            jaws_length: Some(13),
            jaws_shift: Some(8),
            teeth_length: Some(8),
            teeth_shift: Some(5),
            lips_length: Some(5),
            lips_shift: Some(3),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GatorOscInput<'a> {
    pub data: GatorOscData<'a>,
    pub params: GatorOscParams,
}

impl<'a> GatorOscInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: GatorOscParams) -> Self {
        Self {
            data: GatorOscData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: GatorOscParams) -> Self {
        Self {
            data: GatorOscData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", GatorOscParams::default())
    }
    #[inline]
    pub fn get_jaws_length(&self) -> usize {
        self.params.jaws_length.unwrap_or(13)
    }
    #[inline]
    pub fn get_jaws_shift(&self) -> usize {
        self.params.jaws_shift.unwrap_or(8)
    }
    #[inline]
    pub fn get_teeth_length(&self) -> usize {
        self.params.teeth_length.unwrap_or(8)
    }
    #[inline]
    pub fn get_teeth_shift(&self) -> usize {
        self.params.teeth_shift.unwrap_or(5)
    }
    #[inline]
    pub fn get_lips_length(&self) -> usize {
        self.params.lips_length.unwrap_or(5)
    }
    #[inline]
    pub fn get_lips_shift(&self) -> usize {
        self.params.lips_shift.unwrap_or(3)
    }
}

#[derive(Clone, Debug)]
pub struct GatorOscBuilder {
    jaws_length: Option<usize>,
    jaws_shift: Option<usize>,
    teeth_length: Option<usize>,
    teeth_shift: Option<usize>,
    lips_length: Option<usize>,
    lips_shift: Option<usize>,
    kernel: Kernel,
}

impl Default for GatorOscBuilder {
    fn default() -> Self {
        Self {
            jaws_length: None,
            jaws_shift: None,
            teeth_length: None,
            teeth_shift: None,
            lips_length: None,
            lips_shift: None,
            kernel: Kernel::Auto,
        }
    }
}

impl GatorOscBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn jaws_length(mut self, n: usize) -> Self {
        self.jaws_length = Some(n);
        self
    }
    #[inline(always)]
    pub fn jaws_shift(mut self, x: usize) -> Self {
        self.jaws_shift = Some(x);
        self
    }
    #[inline(always)]
    pub fn teeth_length(mut self, n: usize) -> Self {
        self.teeth_length = Some(n);
        self
    }
    #[inline(always)]
    pub fn teeth_shift(mut self, x: usize) -> Self {
        self.teeth_shift = Some(x);
        self
    }
    #[inline(always)]
    pub fn lips_length(mut self, n: usize) -> Self {
        self.lips_length = Some(n);
        self
    }
    #[inline(always)]
    pub fn lips_shift(mut self, x: usize) -> Self {
        self.lips_shift = Some(x);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<GatorOscOutput, GatorOscError> {
        let p = GatorOscParams {
            jaws_length: self.jaws_length,
            jaws_shift: self.jaws_shift,
            teeth_length: self.teeth_length,
            teeth_shift: self.teeth_shift,
            lips_length: self.lips_length,
            lips_shift: self.lips_shift,
        };
        let i = GatorOscInput::from_candles(c, "close", p);
        gatorosc_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<GatorOscOutput, GatorOscError> {
        let p = GatorOscParams {
            jaws_length: self.jaws_length,
            jaws_shift: self.jaws_shift,
            teeth_length: self.teeth_length,
            teeth_shift: self.teeth_shift,
            lips_length: self.lips_length,
            lips_shift: self.lips_shift,
        };
        let i = GatorOscInput::from_slice(d, p);
        gatorosc_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<GatorOscStream, GatorOscError> {
        let p = GatorOscParams {
            jaws_length: self.jaws_length,
            jaws_shift: self.jaws_shift,
            teeth_length: self.teeth_length,
            teeth_shift: self.teeth_shift,
            lips_length: self.lips_length,
            lips_shift: self.lips_shift,
        };
        GatorOscStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum GatorOscError {
    #[error("gatorosc: Input data slice is empty.")]
    EmptyInputData,
    #[error("gatorosc: All values are NaN.")]
    AllValuesNaN,
    #[error("gatorosc: Invalid period: period={period} data_len={data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("gatorosc: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("gatorosc: output length mismatch: expected={expected} got={got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("gatorosc: invalid range: start={start} end={end} step={step}")]
    InvalidRange { start: usize, end: usize, step: usize },
    #[error("gatorosc: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(crate::utilities::enums::Kernel),
    #[error("gatorosc: invalid input: {0}")]
    InvalidInput(String),
}

#[inline(always)]
fn gator_warmups(
    first: usize,
    jl: usize,
    js: usize,
    tl: usize,
    ts: usize,
    ll: usize,
    ls: usize,
) -> (usize, usize, usize, usize) {
    // bars required for EMA pairs + their display shifts
    let upper_needed = jl.max(tl) + js.max(ts);
    let lower_needed = tl.max(ll) + ts.max(ls);

    let upper_warmup = first + upper_needed.saturating_sub(1);
    let lower_warmup = first + lower_needed.saturating_sub(1);
    let upper_change_warmup = upper_warmup + 1;
    let lower_change_warmup = lower_warmup + 1;

    (
        upper_warmup,
        lower_warmup,
        upper_change_warmup,
        lower_change_warmup,
    )
}

#[inline]
pub fn gatorosc(input: &GatorOscInput) -> Result<GatorOscOutput, GatorOscError> {
    gatorosc_with_kernel(input, Kernel::Auto)
}

pub fn gatorosc_with_kernel(
    input: &GatorOscInput,
    kernel: Kernel,
) -> Result<GatorOscOutput, GatorOscError> {
    let (
        data,
        jaws_length,
        jaws_shift,
        teeth_length,
        teeth_shift,
        lips_length,
        lips_shift,
        first,
        chosen,
    ) = gatorosc_prepare(input, kernel)?;

    let (upper_warmup, lower_warmup, upper_change_warmup, lower_change_warmup) = gator_warmups(
        first,
        jaws_length,
        jaws_shift,
        teeth_length,
        teeth_shift,
        lips_length,
        lips_shift,
    );

    let mut upper = alloc_with_nan_prefix(data.len(), upper_warmup);
    let mut lower = alloc_with_nan_prefix(data.len(), lower_warmup);
    let mut upper_change = alloc_with_nan_prefix(data.len(), upper_change_warmup);
    let mut lower_change = alloc_with_nan_prefix(data.len(), lower_change_warmup);

    gatorosc_compute_into(
        data,
        jaws_length,
        jaws_shift,
        teeth_length,
        teeth_shift,
        lips_length,
        lips_shift,
        first,
        chosen,
        &mut upper,
        &mut lower,
        &mut upper_change,
        &mut lower_change,
    );

    // Re-mask prefixes to ensure NaN consistency
    for v in &mut upper[..upper_warmup] {
        *v = f64::NAN;
    }
    for v in &mut lower[..lower_warmup] {
        *v = f64::NAN;
    }
    for v in &mut upper_change[..upper_change_warmup] {
        *v = f64::NAN;
    }
    for v in &mut lower_change[..lower_change_warmup] {
        *v = f64::NAN;
    }

    Ok(GatorOscOutput {
        upper,
        lower,
        upper_change,
        lower_change,
    })
}

#[inline(always)]
pub unsafe fn gatorosc_scalar(
    data: &[f64],
    jaws_length: usize,
    jaws_shift: usize,
    teeth_length: usize,
    teeth_shift: usize,
    lips_length: usize,
    lips_shift: usize,
    first_valid: usize,
    upper: &mut [f64],
    lower: &mut [f64],
    upper_change: &mut [f64],
    lower_change: &mut [f64],
) {
    let n = data.len();
    if first_valid >= n {
        return;
    }

    // Precompute EMA coefficients: y = (1-a)*y_prev + a*x
    let ja = 2.0 / (jaws_length as f64 + 1.0);
    let ta = 2.0 / (teeth_length as f64 + 1.0);
    let la = 2.0 / (lips_length as f64 + 1.0);
    let jma = 1.0 - ja;
    let tma = 1.0 - ta;
    let lma = 1.0 - la;

    // Warmup gates for outputs/changes
    let (uw, lw, ucw, lcw) = gator_warmups(
        first_valid,
        jaws_length,
        jaws_shift,
        teeth_length,
        teeth_shift,
        lips_length,
        lips_shift,
    );

    // Initialize EMA states
    let mut jema = if data[first_valid].is_nan() {
        0.0
    } else {
        data[first_valid]
    };
    let mut tema = jema;
    let mut lema = jema;

    // Small ring buffers for shifted reads; avoid modulo inside hot loop
    let max_shift = jaws_shift.max(teeth_shift).max(lips_shift);
    let buf_len = max_shift + 1;

    let mut jring: AVec<f64> = AVec::with_capacity(CACHELINE_ALIGN, buf_len);
    let mut tring: AVec<f64> = AVec::with_capacity(CACHELINE_ALIGN, buf_len);
    let mut lring: AVec<f64> = AVec::with_capacity(CACHELINE_ALIGN, buf_len);
    jring.resize(buf_len, jema);
    tring.resize(buf_len, tema);
    lring.resize(buf_len, lema);

    let mut rpos: usize = 0;

    // For change outputs on-the-fly
    let mut u_prev = 0.0;
    let mut l_prev = 0.0;
    let mut have_u = false;
    let mut have_l = false;

    let mut i = first_valid;
    while i < n {
        let x = {
            let xi = *data.get_unchecked(i);
            if xi.is_nan() {
                jema
            } else {
                xi
            }
        };

        // EMA updates with fused multiply-add when available
        jema = jma.mul_add(jema, ja * x);
        tema = tma.mul_add(tema, ta * x);
        lema = lma.mul_add(lema, la * x);

        // Push current EMAs into rings at rpos
        *jring.get_unchecked_mut(rpos) = jema;
        *tring.get_unchecked_mut(rpos) = tema;
        *lring.get_unchecked_mut(rpos) = lema;

        // Wrapped indices without %
        let mut jj = rpos + buf_len - jaws_shift;
        if jj >= buf_len {
            jj -= buf_len;
        }
        let mut tt = rpos + buf_len - teeth_shift;
        if tt >= buf_len {
            tt -= buf_len;
        }
        let mut ll = rpos + buf_len - lips_shift;
        if ll >= buf_len {
            ll -= buf_len;
        }

        if i >= uw {
            let u = (*jring.get_unchecked(jj) - *tring.get_unchecked(tt)).abs();
            *upper.get_unchecked_mut(i) = u;
            // Prime previous at the first finite upper
            if i == uw {
                u_prev = u;
                have_u = true;
            } else if i >= ucw && have_u {
                *upper_change.get_unchecked_mut(i) = u - u_prev;
                u_prev = u;
            }
        }

        if i >= lw {
            let l = -(*tring.get_unchecked(tt) - *lring.get_unchecked(ll)).abs();
            *lower.get_unchecked_mut(i) = l;
            if i == lw {
                l_prev = l;
                have_l = true;
            } else if i >= lcw && have_l {
                *lower_change.get_unchecked_mut(i) = -(l - l_prev);
                l_prev = l;
            }
        }

        // Advance ring position with branchless wrap
        rpos += 1;
        if rpos == buf_len {
            rpos = 0;
        }

        i += 1;
    }
}

#[cfg(all(target_feature = "simd128", target_arch = "wasm32"))]
#[inline(always)]
pub unsafe fn gatorosc_simd128(
    data: &[f64],
    jaws_length: usize,
    jaws_shift: usize,
    teeth_length: usize,
    teeth_shift: usize,
    lips_length: usize,
    lips_shift: usize,
    first_valid: usize,
    upper: &mut [f64],
    lower: &mut [f64],
    upper_change: &mut [f64],
    lower_change: &mut [f64],
) {
    // SIMD128 implementation uses scalar since the complex EMA and ring buffer logic
    // doesn't benefit significantly from SIMD128's limited 128-bit vectors
    gatorosc_scalar(
        data,
        jaws_length,
        jaws_shift,
        teeth_length,
        teeth_shift,
        lips_length,
        lips_shift,
        first_valid,
        upper,
        lower,
        upper_change,
        lower_change,
    );
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn gatorosc_avx2(
    data: &[f64],
    jaws_length: usize,
    jaws_shift: usize,
    teeth_length: usize,
    teeth_shift: usize,
    lips_length: usize,
    lips_shift: usize,
    first_valid: usize,
    upper: &mut [f64],
    lower: &mut [f64],
    upper_change: &mut [f64],
    lower_change: &mut [f64],
) {
    use core::arch::x86_64::*;

    let n = data.len();
    if first_valid >= n {
        return;
    }

    let ja = 2.0 / (jaws_length as f64 + 1.0);
    let ta = 2.0 / (teeth_length as f64 + 1.0);
    let la = 2.0 / (lips_length as f64 + 1.0);
    // lanes: [j, t, l, 0]
    let a = _mm256_set_pd(0.0, la, ta, ja);
    let one = _mm256_set1_pd(1.0);
    let oma = _mm256_sub_pd(one, a);

    let (uw, lw, ucw, lcw) = gator_warmups(
        first_valid,
        jaws_length,
        jaws_shift,
        teeth_length,
        teeth_shift,
        lips_length,
        lips_shift,
    );

    let mut jema = if data.get_unchecked(first_valid).is_nan() {
        0.0
    } else {
        *data.get_unchecked(first_valid)
    };
    let mut tema = jema;
    let mut lema = jema;

    // e = [j, t, l, 0]
    let mut e = _mm256_set_pd(0.0, lema, tema, jema);

    let max_shift = jaws_shift.max(teeth_shift).max(lips_shift);
    let buf_len = max_shift + 1;
    let mut jring: AVec<f64> = AVec::with_capacity(CACHELINE_ALIGN, buf_len);
    let mut tring: AVec<f64> = AVec::with_capacity(CACHELINE_ALIGN, buf_len);
    let mut lring: AVec<f64> = AVec::with_capacity(CACHELINE_ALIGN, buf_len);
    jring.resize(buf_len, jema);
    tring.resize(buf_len, tema);
    lring.resize(buf_len, lema);

    let mut rpos: usize = 0;
    let mut u_prev = 0.0;
    let mut l_prev = 0.0;
    let mut have_u = false;
    let mut have_l = false;
    let mut lanes: [f64; 4] = core::mem::zeroed();

    let mut i = first_valid;
    while i < n {
        let x = {
            let xi = *data.get_unchecked(i);
            if xi.is_nan() {
                jema
            } else {
                xi
            }
        };
        let vx = _mm256_set1_pd(x);
        let oma_e = _mm256_mul_pd(oma, e);
        let a_vx = _mm256_mul_pd(a, vx);
        e = _mm256_add_pd(oma_e, a_vx);

        _mm256_storeu_pd(lanes.as_mut_ptr(), e);
        jema = lanes[0];
        tema = lanes[1];
        lema = lanes[2];

        *jring.get_unchecked_mut(rpos) = jema;
        *tring.get_unchecked_mut(rpos) = tema;
        *lring.get_unchecked_mut(rpos) = lema;

        let mut jj = rpos + buf_len - jaws_shift;
        if jj >= buf_len {
            jj -= buf_len;
        }
        let mut tt = rpos + buf_len - teeth_shift;
        if tt >= buf_len {
            tt -= buf_len;
        }
        let mut ll = rpos + buf_len - lips_shift;
        if ll >= buf_len {
            ll -= buf_len;
        }

        if i >= uw {
            let u = (*jring.get_unchecked(jj) - *tring.get_unchecked(tt)).abs();
            *upper.get_unchecked_mut(i) = u;
            if i == uw {
                u_prev = u;
                have_u = true;
            } else if i >= ucw && have_u {
                *upper_change.get_unchecked_mut(i) = u - u_prev;
                u_prev = u;
            }
        }

        if i >= lw {
            let l = -(*tring.get_unchecked(tt) - *lring.get_unchecked(ll)).abs();
            *lower.get_unchecked_mut(i) = l;
            if i == lw {
                l_prev = l;
                have_l = true;
            } else if i >= lcw && have_l {
                *lower_change.get_unchecked_mut(i) = -(l - l_prev);
                l_prev = l;
            }
        }

        rpos += 1;
        if rpos == buf_len {
            rpos = 0;
        }
        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn gatorosc_avx512(
    data: &[f64],
    jaws_length: usize,
    jaws_shift: usize,
    teeth_length: usize,
    teeth_shift: usize,
    lips_length: usize,
    lips_shift: usize,
    first_valid: usize,
    upper: &mut [f64],
    lower: &mut [f64],
    upper_change: &mut [f64],
    lower_change: &mut [f64],
) {
    if jaws_length <= 32 && teeth_length <= 32 && lips_length <= 32 {
        gatorosc_avx512_short(
            data,
            jaws_length,
            jaws_shift,
            teeth_length,
            teeth_shift,
            lips_length,
            lips_shift,
            first_valid,
            upper,
            lower,
            upper_change,
            lower_change,
        );
    } else {
        gatorosc_avx512_long(
            data,
            jaws_length,
            jaws_shift,
            teeth_length,
            teeth_shift,
            lips_length,
            lips_shift,
            first_valid,
            upper,
            lower,
            upper_change,
            lower_change,
        );
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn gatorosc_avx512_short(
    data: &[f64],
    jaws_length: usize,
    jaws_shift: usize,
    teeth_length: usize,
    teeth_shift: usize,
    lips_length: usize,
    lips_shift: usize,
    first_valid: usize,
    upper: &mut [f64],
    lower: &mut [f64],
    upper_change: &mut [f64],
    lower_change: &mut [f64],
) {
    use core::arch::x86_64::*;

    let n = data.len();
    if first_valid >= n {
        return;
    }

    let ja = 2.0 / (jaws_length as f64 + 1.0);
    let ta = 2.0 / (teeth_length as f64 + 1.0);
    let la = 2.0 / (lips_length as f64 + 1.0);
    // lanes: [j, t, l, 0, 0, 0, 0, 0]
    let a = _mm512_set_pd(0.0, 0.0, 0.0, 0.0, 0.0, la, ta, ja);
    let one = _mm512_set1_pd(1.0);
    let oma = _mm512_sub_pd(one, a);

    let (uw, lw, ucw, lcw) = gator_warmups(
        first_valid,
        jaws_length,
        jaws_shift,
        teeth_length,
        teeth_shift,
        lips_length,
        lips_shift,
    );

    let mut jema = if data.get_unchecked(first_valid).is_nan() {
        0.0
    } else {
        *data.get_unchecked(first_valid)
    };
    let mut tema = jema;
    let mut lema = jema;

    let mut e = _mm512_set_pd(0.0, 0.0, 0.0, 0.0, 0.0, lema, tema, jema);

    let max_shift = jaws_shift.max(teeth_shift).max(lips_shift);
    let buf_len = max_shift + 1;
    let mut jring: AVec<f64> = AVec::with_capacity(CACHELINE_ALIGN, buf_len);
    let mut tring: AVec<f64> = AVec::with_capacity(CACHELINE_ALIGN, buf_len);
    let mut lring: AVec<f64> = AVec::with_capacity(CACHELINE_ALIGN, buf_len);
    jring.resize(buf_len, jema);
    tring.resize(buf_len, tema);
    lring.resize(buf_len, lema);

    let mut rpos: usize = 0;
    let mut u_prev = 0.0;
    let mut l_prev = 0.0;
    let mut have_u = false;
    let mut have_l = false;
    let mut lanes: [f64; 8] = core::mem::zeroed();

    let mut i = first_valid;
    while i < n {
        let x = {
            let xi = *data.get_unchecked(i);
            if xi.is_nan() {
                jema
            } else {
                xi
            }
        };
        let vx = _mm512_set1_pd(x);
        let oma_e = _mm512_mul_pd(oma, e);
        let a_vx = _mm512_mul_pd(a, vx);
        e = _mm512_add_pd(oma_e, a_vx);

        _mm512_storeu_pd(lanes.as_mut_ptr(), e);
        // Lane order after store: [jema, tema, lema, 0, 0, 0, 0, 0]
        jema = lanes[0];
        tema = lanes[1];
        lema = lanes[2];

        *jring.get_unchecked_mut(rpos) = jema;
        *tring.get_unchecked_mut(rpos) = tema;
        *lring.get_unchecked_mut(rpos) = lema;

        let mut jj = rpos + buf_len - jaws_shift;
        if jj >= buf_len {
            jj -= buf_len;
        }
        let mut tt = rpos + buf_len - teeth_shift;
        if tt >= buf_len {
            tt -= buf_len;
        }
        let mut ll = rpos + buf_len - lips_shift;
        if ll >= buf_len {
            ll -= buf_len;
        }

        if i >= uw {
            let u = (*jring.get_unchecked(jj) - *tring.get_unchecked(tt)).abs();
            *upper.get_unchecked_mut(i) = u;
            if i == uw {
                u_prev = u;
                have_u = true;
            } else if i >= ucw && have_u {
                *upper_change.get_unchecked_mut(i) = u - u_prev;
                u_prev = u;
            }
        }

        if i >= lw {
            let l = -(*tring.get_unchecked(tt) - *lring.get_unchecked(ll)).abs();
            *lower.get_unchecked_mut(i) = l;
            if i == lw {
                l_prev = l;
                have_l = true;
            } else if i >= lcw && have_l {
                *lower_change.get_unchecked_mut(i) = -(l - l_prev);
                l_prev = l;
            }
        }

        rpos += 1;
        if rpos == buf_len {
            rpos = 0;
        }
        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn gatorosc_avx512_long(
    data: &[f64],
    jaws_length: usize,
    jaws_shift: usize,
    teeth_length: usize,
    teeth_shift: usize,
    lips_length: usize,
    lips_shift: usize,
    first_valid: usize,
    upper: &mut [f64],
    lower: &mut [f64],
    upper_change: &mut [f64],
    lower_change: &mut [f64],
) {
    // Reuse the same packed-EMA approach
    gatorosc_avx512_short(
        data,
        jaws_length,
        jaws_shift,
        teeth_length,
        teeth_shift,
        lips_length,
        lips_shift,
        first_valid,
        upper,
        lower,
        upper_change,
        lower_change,
    );
}

// Helper function for validation and preparation
#[inline]
fn gatorosc_prepare<'a>(
    input: &'a GatorOscInput<'a>,
    kernel: Kernel,
) -> Result<
    (
        &'a [f64],
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        Kernel,
    ),
    GatorOscError,
> {
    let data: &[f64] = input.as_ref();

    // Check for empty data first
    if data.is_empty() {
        return Err(GatorOscError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(GatorOscError::AllValuesNaN)?;

    let jaws_length = input.get_jaws_length();
    let jaws_shift = input.get_jaws_shift();
    let teeth_length = input.get_teeth_length();
    let teeth_shift = input.get_teeth_shift();
    let lips_length = input.get_lips_length();
    let lips_shift = input.get_lips_shift();

    if jaws_length == 0 {
        return Err(GatorOscError::InvalidPeriod { period: jaws_length, data_len: data.len() });
    }
    if teeth_length == 0 {
        return Err(GatorOscError::InvalidPeriod { period: teeth_length, data_len: data.len() });
    }
    if lips_length == 0 {
        return Err(GatorOscError::InvalidPeriod { period: lips_length, data_len: data.len() });
    }

    let needed = jaws_length.max(teeth_length).max(lips_length)
        + jaws_shift.max(teeth_shift).max(lips_shift);
    if data.len() - first < needed {
        return Err(GatorOscError::NotEnoughValidData {
            needed,
            valid: data.len() - first,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    Ok((
        data,
        jaws_length,
        jaws_shift,
        teeth_length,
        teeth_shift,
        lips_length,
        lips_shift,
        first,
        chosen,
    ))
}

// Zero-allocation compute function
#[inline]
fn gatorosc_compute_into(
    data: &[f64],
    jaws_length: usize,
    jaws_shift: usize,
    teeth_length: usize,
    teeth_shift: usize,
    lips_length: usize,
    lips_shift: usize,
    first: usize,
    kernel: Kernel,
    upper: &mut [f64],
    lower: &mut [f64],
    upper_change: &mut [f64],
    lower_change: &mut [f64],
) {
    unsafe {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            if matches!(kernel, Kernel::Scalar | Kernel::ScalarBatch) {
                gatorosc_simd128(
                    data,
                    jaws_length,
                    jaws_shift,
                    teeth_length,
                    teeth_shift,
                    lips_length,
                    lips_shift,
                    first,
                    upper,
                    lower,
                    upper_change,
                    lower_change,
                );
                return;
            }
        }
        match kernel {
            Kernel::Scalar | Kernel::ScalarBatch => gatorosc_scalar(
                data,
                jaws_length,
                jaws_shift,
                teeth_length,
                teeth_shift,
                lips_length,
                lips_shift,
                first,
                upper,
                lower,
                upper_change,
                lower_change,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => gatorosc_avx2(
                data,
                jaws_length,
                jaws_shift,
                teeth_length,
                teeth_shift,
                lips_length,
                lips_shift,
                first,
                upper,
                lower,
                upper_change,
                lower_change,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => gatorosc_avx512(
                data,
                jaws_length,
                jaws_shift,
                teeth_length,
                teeth_shift,
                lips_length,
                lips_shift,
                first,
                upper,
                lower,
                upper_change,
                lower_change,
            ),
            _ => unreachable!(),
        }
    }
}

// Into slice wrapper for external use
#[inline]
pub fn gatorosc_into_slice(
    upper_dst: &mut [f64],
    lower_dst: &mut [f64],
    upper_change_dst: &mut [f64],
    lower_change_dst: &mut [f64],
    input: &GatorOscInput,
    kernel: Kernel,
) -> Result<(), GatorOscError> {
    let (
        data,
        jaws_length,
        jaws_shift,
        teeth_length,
        teeth_shift,
        lips_length,
        lips_shift,
        first,
        chosen,
    ) = gatorosc_prepare(input, kernel)?;

    let expected = data.len();
    if upper_dst.len() != expected {
        return Err(GatorOscError::OutputLengthMismatch { expected, got: upper_dst.len() });
    }
    if lower_dst.len() != expected {
        return Err(GatorOscError::OutputLengthMismatch { expected, got: lower_dst.len() });
    }
    if upper_change_dst.len() != expected {
        return Err(GatorOscError::OutputLengthMismatch { expected, got: upper_change_dst.len() });
    }
    if lower_change_dst.len() != expected {
        return Err(GatorOscError::OutputLengthMismatch { expected, got: lower_change_dst.len() });
    }

    gatorosc_compute_into(
        data,
        jaws_length,
        jaws_shift,
        teeth_length,
        teeth_shift,
        lips_length,
        lips_shift,
        first,
        chosen,
        upper_dst,
        lower_dst,
        upper_change_dst,
        lower_change_dst,
    );

    let (upper_warmup, lower_warmup, upper_change_warmup, lower_change_warmup) = gator_warmups(
        first,
        jaws_length,
        jaws_shift,
        teeth_length,
        teeth_shift,
        lips_length,
        lips_shift,
    );

    for v in &mut upper_dst[..upper_warmup] {
        *v = f64::NAN;
    }
    for v in &mut lower_dst[..lower_warmup] {
        *v = f64::NAN;
    }
    for v in &mut upper_change_dst[..upper_change_warmup] {
        *v = f64::NAN;
    }
    for v in &mut lower_change_dst[..lower_change_warmup] {
        *v = f64::NAN;
    }

    Ok(())
}

/// Writes Gator Oscillator results into caller-provided buffers without allocating.
///
/// - Preserves NaN warmups exactly like the Vec-returning API.
/// - All output slices must have the same length as the input series.
/// - Equivalent to calling `gatorosc_with_kernel(input, Kernel::Auto)` and copying results.
#[cfg(not(feature = "wasm"))]
#[inline]
pub fn gatorosc_into(
    input: &GatorOscInput,
    upper: &mut [f64],
    lower: &mut [f64],
    upper_change: &mut [f64],
    lower_change: &mut [f64],
) -> Result<(), GatorOscError> {
    gatorosc_into_slice(upper, lower, upper_change, lower_change, input, Kernel::Auto)
}

// --- Streaming kernel (O(1) per update) -------------------------------------

#[derive(Debug, Clone)]
pub struct GatorOscStream {
    // EMA coefficients (alpha) and state
    ja: f64,
    ta: f64,
    la: f64,
    jema: f64,
    tema: f64,
    lema: f64,
    initialized: bool,

    // Shifts (display offsets)
    jaws_shift: usize,
    teeth_shift: usize,
    lips_shift: usize,

    // Small fixed-size rings of EMA values for shift lookbacks
    jring: AVec<f64>,
    tring: AVec<f64>,
    lring: AVec<f64>,
    rpos: usize, // write position in rings

    // Index of next update (acts like "i" in vectorized kernels)
    idx: usize,

    // Warmup bookkeeping relative to first valid index
    first_valid: Option<usize>,
    upper_needed: usize,
    lower_needed: usize,
    warmup_upper: usize, // absolute index where upper first valid
    warmup_lower: usize, // absolute index where lower first valid
    warmup_uc: usize,    // absolute index where upper_change first valid
    warmup_lc: usize,    // absolute index where lower_change first valid

    // For 1-bar momentum
    prev_u: f64,
    prev_l: f64,
    have_prev_u: bool,
    have_prev_l: bool,
}

// --- fast tiny helpers -------------------------------------------------------

#[inline(always)]
fn ema_update(prev: f64, x: f64, a: f64) -> f64 {
    // y_t = y_{t-1} + a * (x_t - y_{t-1}); fused if FMA is available
    (x - prev).mul_add(a, prev)
}

#[inline(always)]
fn fast_abs_f64(x: f64) -> f64 {
    // bitwise abs; identical to x.abs() for all finite values & NaN-preserving
    f64::from_bits(x.to_bits() & 0x7FFF_FFFF_FFFF_FFFF)
}

#[inline(always)]
fn wrap_back(pos: usize, len: usize, back: usize) -> usize {
    let mut idx = pos + len - back;
    if idx >= len {
        idx -= len;
    }
    idx
}

impl GatorOscStream {
    pub fn try_new(params: GatorOscParams) -> Result<Self, GatorOscError> {
        let jaws_length = params.jaws_length.unwrap_or(13);
        let jaws_shift = params.jaws_shift.unwrap_or(8);
        let teeth_length = params.teeth_length.unwrap_or(8);
        let teeth_shift = params.teeth_shift.unwrap_or(5);
        let lips_length = params.lips_length.unwrap_or(5);
        let lips_shift = params.lips_shift.unwrap_or(3);

        if jaws_length == 0 {
            return Err(GatorOscError::InvalidPeriod { period: jaws_length, data_len: 0 });
        }
        if teeth_length == 0 {
            return Err(GatorOscError::InvalidPeriod { period: teeth_length, data_len: 0 });
        }
        if lips_length == 0 {
            return Err(GatorOscError::InvalidPeriod { period: lips_length, data_len: 0 });
        }

        // EMA alphas
        let ja = 2.0 / (jaws_length as f64 + 1.0);
        let ta = 2.0 / (teeth_length as f64 + 1.0);
        let la = 2.0 / (lips_length as f64 + 1.0);

        // Rings sized to max shift + 1 (match offline kernels)
        let buf_len = jaws_shift.max(teeth_shift).max(lips_shift) + 1;
        let mut jring: AVec<f64> = AVec::with_capacity(CACHELINE_ALIGN, buf_len);
        let mut tring: AVec<f64> = AVec::with_capacity(CACHELINE_ALIGN, buf_len);
        let mut lring: AVec<f64> = AVec::with_capacity(CACHELINE_ALIGN, buf_len);
        jring.resize(buf_len, 0.0);
        tring.resize(buf_len, 0.0);
        lring.resize(buf_len, 0.0);

        // Bars required for each output (relative, not absolute)
        let upper_needed = jaws_length.max(teeth_length) + jaws_shift.max(teeth_shift);
        let lower_needed = teeth_length.max(lips_length) + teeth_shift.max(lips_shift);

        Ok(Self {
            ja,
            ta,
            la,
            jema: 0.0,
            tema: 0.0,
            lema: 0.0,
            initialized: false,

            jaws_shift,
            teeth_shift,
            lips_shift,

            jring,
            tring,
            lring,
            rpos: 0,
            idx: 0,

            first_valid: None,
            upper_needed,
            lower_needed,
            warmup_upper: usize::MAX,
            warmup_lower: usize::MAX,
            warmup_uc: usize::MAX,
            warmup_lc: usize::MAX,

            prev_u: 0.0,
            prev_l: 0.0,
            have_prev_u: false,
            have_prev_l: false,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<(f64, f64, f64, f64)> {
        let i = self.idx; // current bar index
        self.idx = i + 1;

        // Delay initialization until first finite value (like offline first-valid)
        if !self.initialized {
            if !value.is_finite() {
                return None; // still warming up on NaNs
            }
            self.jema = value;
            self.tema = value;
            self.lema = value;
            self.initialized = true;
            self.first_valid = Some(i);

            // Convert relative warmups to absolute indices
            self.warmup_upper = i + self.upper_needed.saturating_sub(1);
            self.warmup_lower = i + self.lower_needed.saturating_sub(1);
            self.warmup_uc = self.warmup_upper + 1;
            self.warmup_lc = self.warmup_lower + 1;
        } else {
            let x = if value.is_nan() { self.jema } else { value };
            self.jema = ema_update(self.jema, x, self.ja);
            self.tema = ema_update(self.tema, x, self.ta);
            self.lema = ema_update(self.lema, x, self.la);
        }

        // Push current EMA states into the rings
        let r = self.rpos;
        self.jring[r] = self.jema;
        self.tring[r] = self.tema;
        self.lring[r] = self.lema;

        // Compute shifted indices
        let len = self.jring.len();
        let jj = wrap_back(r, len, self.jaws_shift);
        let tt = wrap_back(r, len, self.teeth_shift);
        let ll = wrap_back(r, len, self.lips_shift);

        // Advance ring position
        let mut next = r + 1;
        if next == len {
            next = 0;
        }
        self.rpos = next;

        // Prime prev values when each output first becomes defined
        if i == self.warmup_upper {
            let u0 = fast_abs_f64(self.jring[jj] - self.tring[tt]);
            self.prev_u = u0;
            self.have_prev_u = true;
        }
        if i == self.warmup_lower {
            let l0 = -fast_abs_f64(self.tring[tt] - self.lring[ll]);
            self.prev_l = l0;
            self.have_prev_l = true;
        }

        // Respect original API: emit only after lower_change warmup is satisfied
        if i < self.warmup_lc {
            return None;
        }

        // Compute outputs for this bar
        let u = fast_abs_f64(self.jring[jj] - self.tring[tt]);
        let l = -fast_abs_f64(self.tring[tt] - self.lring[ll]);

        // 1-bar momentum
        let uc = if i >= self.warmup_uc && self.have_prev_u {
            let d = u - self.prev_u;
            self.prev_u = u;
            d
        } else {
            f64::NAN
        };
        let lc = if i >= self.warmup_lc && self.have_prev_l {
            let d = self.prev_l - l; // == -(l - prev_l)
            self.prev_l = l;
            d
        } else {
            f64::NAN
        };

        Some((u, l, uc, lc))
    }
}

#[derive(Clone, Debug)]
pub struct GatorOscBatchRange {
    pub jaws_length: (usize, usize, usize),
    pub jaws_shift: (usize, usize, usize),
    pub teeth_length: (usize, usize, usize),
    pub teeth_shift: (usize, usize, usize),
    pub lips_length: (usize, usize, usize),
    pub lips_shift: (usize, usize, usize),
}

impl Default for GatorOscBatchRange {
    fn default() -> Self {
        Self {
            jaws_length: (13, 262, 1),
            jaws_shift: (8, 8, 0),
            teeth_length: (8, 8, 0),
            teeth_shift: (5, 5, 0),
            lips_length: (5, 5, 0),
            lips_shift: (3, 3, 0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct GatorOscBatchBuilder {
    range: GatorOscBatchRange,
    kernel: Kernel,
}

impl GatorOscBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    pub fn jaws_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.jaws_length = (start, end, step);
        self
    }
    pub fn jaws_shift_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.jaws_shift = (start, end, step);
        self
    }
    pub fn teeth_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.teeth_length = (start, end, step);
        self
    }
    pub fn teeth_shift_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.teeth_shift = (start, end, step);
        self
    }
    pub fn lips_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.lips_length = (start, end, step);
        self
    }
    pub fn lips_shift_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.lips_shift = (start, end, step);
        self
    }
    pub fn apply_slice(self, data: &[f64]) -> Result<GatorOscBatchOutput, GatorOscError> {
        gatorosc_batch_with_kernel(data, &self.range, self.kernel)
    }
}

#[derive(Clone, Debug)]
pub struct GatorOscBatchOutput {
    pub upper: Vec<f64>,
    pub lower: Vec<f64>,
    pub upper_change: Vec<f64>,
    pub lower_change: Vec<f64>,
    pub combos: Vec<GatorOscParams>,
    pub rows: usize,
    pub cols: usize,
}

pub fn gatorosc_batch_with_kernel(
    data: &[f64],
    sweep: &GatorOscBatchRange,
    k: Kernel,
) -> Result<GatorOscBatchOutput, GatorOscError> {
    let combos = expand_grid_gatorosc(sweep)?;
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(GatorOscError::InvalidKernelForBatch(k)),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => kernel,
    };
    gatorosc_batch_inner(data, &combos, simd)
}

fn expand_grid_gatorosc(r: &GatorOscBatchRange) -> Result<Vec<GatorOscParams>, GatorOscError> {
    fn axis((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        if start < end {
            // ascending, inclusive upper bound
            return (start..=end).step_by(step.max(1)).collect();
        }
        // descending bounds supported
        let mut v = Vec::new();
        let mut cur = start;
        let s = step.max(1);
        while cur >= end {
            v.push(cur);
            if cur < end + s {
                break;
            }
            cur = cur.saturating_sub(s);
            if cur == usize::MAX {
                break;
            }
        }
        v
    }

    let jaws_lengths = axis(r.jaws_length);
    if jaws_lengths.is_empty() {
        return Err(GatorOscError::InvalidRange {
            start: r.jaws_length.0,
            end: r.jaws_length.1,
            step: r.jaws_length.2,
        });
    }
    let jaws_shifts = axis(r.jaws_shift);
    if jaws_shifts.is_empty() {
        return Err(GatorOscError::InvalidRange {
            start: r.jaws_shift.0,
            end: r.jaws_shift.1,
            step: r.jaws_shift.2,
        });
    }
    let teeth_lengths = axis(r.teeth_length);
    if teeth_lengths.is_empty() {
        return Err(GatorOscError::InvalidRange {
            start: r.teeth_length.0,
            end: r.teeth_length.1,
            step: r.teeth_length.2,
        });
    }
    let teeth_shifts = axis(r.teeth_shift);
    if teeth_shifts.is_empty() {
        return Err(GatorOscError::InvalidRange {
            start: r.teeth_shift.0,
            end: r.teeth_shift.1,
            step: r.teeth_shift.2,
        });
    }
    let lips_lengths = axis(r.lips_length);
    if lips_lengths.is_empty() {
        return Err(GatorOscError::InvalidRange {
            start: r.lips_length.0,
            end: r.lips_length.1,
            step: r.lips_length.2,
        });
    }
    let lips_shifts = axis(r.lips_shift);
    if lips_shifts.is_empty() {
        return Err(GatorOscError::InvalidRange {
            start: r.lips_shift.0,
            end: r.lips_shift.1,
            step: r.lips_shift.2,
        });
    }

    let cap = jaws_lengths
        .len()
        .checked_mul(jaws_shifts.len())
        .and_then(|v| v.checked_mul(teeth_lengths.len()))
        .and_then(|v| v.checked_mul(teeth_shifts.len()))
        .and_then(|v| v.checked_mul(lips_lengths.len()))
        .and_then(|v| v.checked_mul(lips_shifts.len()))
        .ok_or_else(|| GatorOscError::InvalidInput("batch sweep size overflow".into()))?;

    let mut out = Vec::with_capacity(cap);
    for &jl in &jaws_lengths {
        for &js in &jaws_shifts {
            for &tl in &teeth_lengths {
                for &ts in &teeth_shifts {
                    for &ll in &lips_lengths {
                        for &ls in &lips_shifts {
                            out.push(GatorOscParams {
                                jaws_length: Some(jl),
                                jaws_shift: Some(js),
                                teeth_length: Some(tl),
                                teeth_shift: Some(ts),
                                lips_length: Some(ll),
                                lips_shift: Some(ls),
                            });
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

fn gatorosc_batch_inner(
    data: &[f64],
    combos: &[GatorOscParams],
    kern: Kernel,
) -> Result<GatorOscBatchOutput, GatorOscError> {
    // Check for empty data first
    if data.is_empty() {
        return Err(GatorOscError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(GatorOscError::AllValuesNaN)?;
    let max_jl = combos.iter().map(|c| c.jaws_length.unwrap()).max().unwrap();
    let max_js = combos.iter().map(|c| c.jaws_shift.unwrap()).max().unwrap();
    let max_tl = combos
        .iter()
        .map(|c| c.teeth_length.unwrap())
        .max()
        .unwrap();
    let max_ts = combos.iter().map(|c| c.teeth_shift.unwrap()).max().unwrap();
    let max_ll = combos.iter().map(|c| c.lips_length.unwrap()).max().unwrap();
    let max_ls = combos.iter().map(|c| c.lips_shift.unwrap()).max().unwrap();
    let needed = max_jl.max(max_tl).max(max_ll) + max_js.max(max_ts).max(max_ls);
    if data.len() - first < needed {
        return Err(GatorOscError::NotEnoughValidData {
            needed,
            valid: data.len() - first,
        });
    }
    let rows = combos.len();
    let cols = data.len();

    // Use make_uninit_matrix for efficient allocation
    let mut upper_mu = make_uninit_matrix(rows, cols);
    let mut lower_mu = make_uninit_matrix(rows, cols);
    let mut upper_change_mu = make_uninit_matrix(rows, cols);
    let mut lower_change_mu = make_uninit_matrix(rows, cols);

    // Initialize prefixes with NaN based on warmup periods using gator_warmups
    let warm_upper: Vec<usize> = combos
        .iter()
        .map(|c| {
            let (uw, _, _, _) = gator_warmups(
                first,
                c.jaws_length.unwrap(),
                c.jaws_shift.unwrap(),
                c.teeth_length.unwrap(),
                c.teeth_shift.unwrap(),
                c.lips_length.unwrap(),
                c.lips_shift.unwrap(),
            );
            uw
        })
        .collect();

    let warm_lower: Vec<usize> = combos
        .iter()
        .map(|c| {
            let (_, lw, _, _) = gator_warmups(
                first,
                c.jaws_length.unwrap(),
                c.jaws_shift.unwrap(),
                c.teeth_length.unwrap(),
                c.teeth_shift.unwrap(),
                c.lips_length.unwrap(),
                c.lips_shift.unwrap(),
            );
            lw
        })
        .collect();

    let warm_uc: Vec<usize> = combos
        .iter()
        .map(|c| {
            let (_, _, ucw, _) = gator_warmups(
                first,
                c.jaws_length.unwrap(),
                c.jaws_shift.unwrap(),
                c.teeth_length.unwrap(),
                c.teeth_shift.unwrap(),
                c.lips_length.unwrap(),
                c.lips_shift.unwrap(),
            );
            ucw
        })
        .collect();

    let warm_lc: Vec<usize> = combos
        .iter()
        .map(|c| {
            let (_, _, _, lcw) = gator_warmups(
                first,
                c.jaws_length.unwrap(),
                c.jaws_shift.unwrap(),
                c.teeth_length.unwrap(),
                c.teeth_shift.unwrap(),
                c.lips_length.unwrap(),
                c.lips_shift.unwrap(),
            );
            lcw
        })
        .collect();

    init_matrix_prefixes(&mut upper_mu, cols, &warm_upper);
    init_matrix_prefixes(&mut lower_mu, cols, &warm_lower);
    init_matrix_prefixes(&mut upper_change_mu, cols, &warm_uc);
    init_matrix_prefixes(&mut lower_change_mu, cols, &warm_lc);

    // Zero-copy convert using ManuallyDrop
    let mut u_guard = core::mem::ManuallyDrop::new(upper_mu);
    let mut l_guard = core::mem::ManuallyDrop::new(lower_mu);
    let mut uc_guard = core::mem::ManuallyDrop::new(upper_change_mu);
    let mut lc_guard = core::mem::ManuallyDrop::new(lower_change_mu);

    let upper: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(u_guard.as_mut_ptr() as *mut f64, u_guard.len()) };
    let lower: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(l_guard.as_mut_ptr() as *mut f64, l_guard.len()) };
    let upper_change: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(uc_guard.as_mut_ptr() as *mut f64, uc_guard.len())
    };
    let lower_change: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(lc_guard.as_mut_ptr() as *mut f64, lc_guard.len())
    };

    let do_row = |row: usize, u: &mut [f64], l: &mut [f64], uc: &mut [f64], lc: &mut [f64]| {
        let prm = &combos[row];
        gatorosc_compute_into(
            data,
            prm.jaws_length.unwrap(),
            prm.jaws_shift.unwrap(),
            prm.teeth_length.unwrap(),
            prm.teeth_shift.unwrap(),
            prm.lips_length.unwrap(),
            prm.lips_shift.unwrap(),
            first,
            kern,
            u,
            l,
            uc,
            lc,
        );
    };

    #[cfg(not(target_arch = "wasm32"))]
    {
        upper
            .par_chunks_mut(cols)
            .zip(lower.par_chunks_mut(cols))
            .zip(upper_change.par_chunks_mut(cols))
            .zip(lower_change.par_chunks_mut(cols))
            .enumerate()
            .for_each(|(row, (((u, l), uc), lc))| {
                do_row(row, u, l, uc, lc);
            });
    }
    #[cfg(target_arch = "wasm32")]
    {
        for row in 0..rows {
            let start = row * cols;
            let end = start + cols;
            do_row(
                row,
                &mut upper[start..end],
                &mut lower[start..end],
                &mut upper_change[start..end],
                &mut lower_change[start..end],
            );
        }
    }

    // Convert back to owned vectors without copying
    let upper = unsafe {
        Vec::from_raw_parts(
            u_guard.as_mut_ptr() as *mut f64,
            u_guard.len(),
            u_guard.capacity(),
        )
    };
    let lower = unsafe {
        Vec::from_raw_parts(
            l_guard.as_mut_ptr() as *mut f64,
            l_guard.len(),
            l_guard.capacity(),
        )
    };
    let upper_change = unsafe {
        Vec::from_raw_parts(
            uc_guard.as_mut_ptr() as *mut f64,
            uc_guard.len(),
            uc_guard.capacity(),
        )
    };
    let lower_change = unsafe {
        Vec::from_raw_parts(
            lc_guard.as_mut_ptr() as *mut f64,
            lc_guard.len(),
            lc_guard.capacity(),
        )
    };

    Ok(GatorOscBatchOutput {
        upper,
        lower,
        upper_change,
        lower_change,
        combos: combos.to_vec(),
        rows,
        cols,
    })
}

// Zero-allocation batch function that writes directly to output slices
#[inline]
pub fn gatorosc_batch_inner_into(
    data: &[f64],
    sweep: &GatorOscBatchRange,
    kernel: Kernel,
    parallel: bool,
    upper_out: &mut [f64],
    lower_out: &mut [f64],
    upper_change_out: &mut [f64],
    lower_change_out: &mut [f64],
) -> Result<Vec<GatorOscParams>, GatorOscError> {
    let combos = expand_grid_gatorosc(sweep)?;

    // Check for empty data first
    if data.is_empty() {
        return Err(GatorOscError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(GatorOscError::AllValuesNaN)?;

    let rows = combos.len();
    let cols = data.len();
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| GatorOscError::InvalidInput("rows*cols overflow".into()))?;
    if upper_out.len() != expected {
        return Err(GatorOscError::OutputLengthMismatch { expected, got: upper_out.len() });
    }
    if lower_out.len() != expected {
        return Err(GatorOscError::OutputLengthMismatch { expected, got: lower_out.len() });
    }
    if upper_change_out.len() != expected {
        return Err(GatorOscError::OutputLengthMismatch { expected, got: upper_change_out.len() });
    }
    if lower_change_out.len() != expected {
        return Err(GatorOscError::OutputLengthMismatch { expected, got: lower_change_out.len() });
    }

    // Initialize NaN prefixes for each row based on warmup periods
    // This is critical for externally-provided buffers from Python/WASM
    for (row, combo) in combos.iter().enumerate() {
        // Calculate warmup periods for each output using gator_warmups
        let (upper_warmup, lower_warmup, upper_change_warmup, lower_change_warmup) = gator_warmups(
            first,
            combo.jaws_length.unwrap(),
            combo.jaws_shift.unwrap(),
            combo.teeth_length.unwrap(),
            combo.teeth_shift.unwrap(),
            combo.lips_length.unwrap(),
            combo.lips_shift.unwrap(),
        );

        let row_start = row * cols;

        // Initialize upper buffer with NaN
        for i in 0..upper_warmup.min(cols) {
            upper_out[row_start + i] = f64::NAN;
        }

        // Initialize lower buffer with NaN
        for i in 0..lower_warmup.min(cols) {
            lower_out[row_start + i] = f64::NAN;
        }

        // Initialize upper_change buffer with NaN
        for i in 0..upper_change_warmup.min(cols) {
            upper_change_out[row_start + i] = f64::NAN;
        }

        // Initialize lower_change buffer with NaN
        for i in 0..lower_change_warmup.min(cols) {
            lower_change_out[row_start + i] = f64::NAN;
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    if parallel {
        use rayon::prelude::*;

        // Split the slices into chunks for parallel processing
        let chunk_size = cols;
        let upper_chunks = upper_out.chunks_mut(chunk_size);
        let lower_chunks = lower_out.chunks_mut(chunk_size);
        let upper_change_chunks = upper_change_out.chunks_mut(chunk_size);
        let lower_change_chunks = lower_change_out.chunks_mut(chunk_size);

        // Zip all chunks together and process in parallel
        upper_chunks
            .zip(lower_chunks)
            .zip(upper_change_chunks)
            .zip(lower_change_chunks)
            .enumerate()
            .par_bridge()
            .for_each(|(row, (((upper, lower), upper_change), lower_change))| {
                let prm = &combos[row];

                gatorosc_compute_into(
                    data,
                    prm.jaws_length.unwrap(),
                    prm.jaws_shift.unwrap(),
                    prm.teeth_length.unwrap(),
                    prm.teeth_shift.unwrap(),
                    prm.lips_length.unwrap(),
                    prm.lips_shift.unwrap(),
                    first,
                    kernel,
                    upper,
                    lower,
                    upper_change,
                    lower_change,
                );
            });
    } else {
        for row in 0..rows {
            let prm = &combos[row];
            let start = row * cols;
            let end = start + cols;

            gatorosc_compute_into(
                data,
                prm.jaws_length.unwrap(),
                prm.jaws_shift.unwrap(),
                prm.teeth_length.unwrap(),
                prm.teeth_shift.unwrap(),
                prm.lips_length.unwrap(),
                prm.lips_shift.unwrap(),
                first,
                kernel,
                &mut upper_out[start..end],
                &mut lower_out[start..end],
                &mut upper_change_out[start..end],
                &mut lower_change_out[start..end],
            );
        }
    }

    Ok(combos)
}

#[inline(always)]
unsafe fn gatorosc_row_scalar(
    data: &[f64],
    first: usize,
    jaws_length: usize,
    jaws_shift: usize,
    teeth_length: usize,
    teeth_shift: usize,
    lips_length: usize,
    lips_shift: usize,
    upper: &mut [f64],
    lower: &mut [f64],
    upper_change: &mut [f64],
    lower_change: &mut [f64],
) {
    gatorosc_scalar(
        data,
        jaws_length,
        jaws_shift,
        teeth_length,
        teeth_shift,
        lips_length,
        lips_shift,
        first,
        upper,
        lower,
        upper_change,
        lower_change,
    );
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn gatorosc_row_avx2(
    data: &[f64],
    first: usize,
    jaws_length: usize,
    jaws_shift: usize,
    teeth_length: usize,
    teeth_shift: usize,
    lips_length: usize,
    lips_shift: usize,
    upper: &mut [f64],
    lower: &mut [f64],
    upper_change: &mut [f64],
    lower_change: &mut [f64],
) {
    gatorosc_row_scalar(
        data,
        first,
        jaws_length,
        jaws_shift,
        teeth_length,
        teeth_shift,
        lips_length,
        lips_shift,
        upper,
        lower,
        upper_change,
        lower_change,
    );
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn gatorosc_row_avx512(
    data: &[f64],
    first: usize,
    jaws_length: usize,
    jaws_shift: usize,
    teeth_length: usize,
    teeth_shift: usize,
    lips_length: usize,
    lips_shift: usize,
    upper: &mut [f64],
    lower: &mut [f64],
    upper_change: &mut [f64],
    lower_change: &mut [f64],
) {
    gatorosc_row_scalar(
        data,
        first,
        jaws_length,
        jaws_shift,
        teeth_length,
        teeth_shift,
        lips_length,
        lips_shift,
        upper,
        lower,
        upper_change,
        lower_change,
    );
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn gatorosc_row_avx512_short(
    data: &[f64],
    first: usize,
    jaws_length: usize,
    jaws_shift: usize,
    teeth_length: usize,
    teeth_shift: usize,
    lips_length: usize,
    lips_shift: usize,
    upper: &mut [f64],
    lower: &mut [f64],
    upper_change: &mut [f64],
    lower_change: &mut [f64],
) {
    gatorosc_row_scalar(
        data,
        first,
        jaws_length,
        jaws_shift,
        teeth_length,
        teeth_shift,
        lips_length,
        lips_shift,
        upper,
        lower,
        upper_change,
        lower_change,
    );
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn gatorosc_row_avx512_long(
    data: &[f64],
    first: usize,
    jaws_length: usize,
    jaws_shift: usize,
    teeth_length: usize,
    teeth_shift: usize,
    lips_length: usize,
    lips_shift: usize,
    upper: &mut [f64],
    lower: &mut [f64],
    upper_change: &mut [f64],
    lower_change: &mut [f64],
) {
    gatorosc_row_scalar(
        data,
        first,
        jaws_length,
        jaws_shift,
        teeth_length,
        teeth_shift,
        lips_length,
        lips_shift,
        upper,
        lower,
        upper_change,
        lower_change,
    );
}

#[inline(always)]
pub fn gatorosc_batch_slice(
    data: &[f64],
    sweep: &GatorOscBatchRange,
    kern: Kernel,
) -> Result<GatorOscBatchOutput, GatorOscError> {
    let combos = expand_grid_gatorosc(sweep)?;
    gatorosc_batch_inner(data, &combos, kern)
}

#[inline(always)]
pub fn gatorosc_batch_par_slice(
    data: &[f64],
    sweep: &GatorOscBatchRange,
    kern: Kernel,
) -> Result<GatorOscBatchOutput, GatorOscError> {
    let combos = expand_grid_gatorosc(sweep)?;

    // Check for empty data first
    if data.is_empty() {
        return Err(GatorOscError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(GatorOscError::AllValuesNaN)?;
    let rows = combos.len();
    let cols = data.len();

    // Use helper functions for zero-allocation
    let mut upper_mu = make_uninit_matrix(rows, cols);
    let mut lower_mu = make_uninit_matrix(rows, cols);
    let mut upper_change_mu = make_uninit_matrix(rows, cols);
    let mut lower_change_mu = make_uninit_matrix(rows, cols);

    // Calculate warmup periods using gator_warmups helper
    let warm_upper: Vec<usize> = combos
        .iter()
        .map(|c| {
            let (uw, _, _, _) = gator_warmups(
                first,
                c.jaws_length.unwrap(),
                c.jaws_shift.unwrap(),
                c.teeth_length.unwrap(),
                c.teeth_shift.unwrap(),
                c.lips_length.unwrap(),
                c.lips_shift.unwrap(),
            );
            uw
        })
        .collect();

    let warm_lower: Vec<usize> = combos
        .iter()
        .map(|c| {
            let (_, lw, _, _) = gator_warmups(
                first,
                c.jaws_length.unwrap(),
                c.jaws_shift.unwrap(),
                c.teeth_length.unwrap(),
                c.teeth_shift.unwrap(),
                c.lips_length.unwrap(),
                c.lips_shift.unwrap(),
            );
            lw
        })
        .collect();

    let warm_uc: Vec<usize> = combos
        .iter()
        .map(|c| {
            let (_, _, ucw, _) = gator_warmups(
                first,
                c.jaws_length.unwrap(),
                c.jaws_shift.unwrap(),
                c.teeth_length.unwrap(),
                c.teeth_shift.unwrap(),
                c.lips_length.unwrap(),
                c.lips_shift.unwrap(),
            );
            ucw
        })
        .collect();

    let warm_lc: Vec<usize> = combos
        .iter()
        .map(|c| {
            let (_, _, _, lcw) = gator_warmups(
                first,
                c.jaws_length.unwrap(),
                c.jaws_shift.unwrap(),
                c.teeth_length.unwrap(),
                c.teeth_shift.unwrap(),
                c.lips_length.unwrap(),
                c.lips_shift.unwrap(),
            );
            lcw
        })
        .collect();

    // Initialize with NaN prefixes
    init_matrix_prefixes(&mut upper_mu, cols, &warm_upper);
    init_matrix_prefixes(&mut lower_mu, cols, &warm_lower);
    init_matrix_prefixes(&mut upper_change_mu, cols, &warm_uc);
    init_matrix_prefixes(&mut lower_change_mu, cols, &warm_lc);

    // Zero-copy convert using ManuallyDrop
    let mut u_guard = core::mem::ManuallyDrop::new(upper_mu);
    let mut l_guard = core::mem::ManuallyDrop::new(lower_mu);
    let mut uc_guard = core::mem::ManuallyDrop::new(upper_change_mu);
    let mut lc_guard = core::mem::ManuallyDrop::new(lower_change_mu);

    let upper: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(u_guard.as_mut_ptr() as *mut f64, u_guard.len()) };
    let lower: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(l_guard.as_mut_ptr() as *mut f64, l_guard.len()) };
    let upper_change: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(uc_guard.as_mut_ptr() as *mut f64, uc_guard.len())
    };
    let lower_change: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(lc_guard.as_mut_ptr() as *mut f64, lc_guard.len())
    };
    #[cfg(not(target_arch = "wasm32"))]
    use rayon::prelude::*;

    #[cfg(not(target_arch = "wasm32"))]
    {
        upper
            .par_chunks_mut(cols)
            .zip(lower.par_chunks_mut(cols))
            .zip(upper_change.par_chunks_mut(cols))
            .zip(lower_change.par_chunks_mut(cols))
            .enumerate()
            .for_each(|(row, (((u, l), uc), lc))| {
                let prm = &combos[row];
                unsafe {
                    gatorosc_row_scalar(
                        data,
                        first,
                        prm.jaws_length.unwrap(),
                        prm.jaws_shift.unwrap(),
                        prm.teeth_length.unwrap(),
                        prm.teeth_shift.unwrap(),
                        prm.lips_length.unwrap(),
                        prm.lips_shift.unwrap(),
                        u,
                        l,
                        uc,
                        lc,
                    );
                }
            });
    }
    #[cfg(target_arch = "wasm32")]
    {
        for row in 0..rows {
            let start = row * cols;
            let end = start + cols;
            let prm = &combos[row];
            unsafe {
                gatorosc_row_scalar(
                    data,
                    first,
                    prm.jaws_length.unwrap(),
                    prm.jaws_shift.unwrap(),
                    prm.teeth_length.unwrap(),
                    prm.teeth_shift.unwrap(),
                    prm.lips_length.unwrap(),
                    prm.lips_shift.unwrap(),
                    &mut upper[start..end],
                    &mut lower[start..end],
                    &mut upper_change[start..end],
                    &mut lower_change[start..end],
                );
            }
        }
    }

    // Convert back to owned vectors without copying
    let upper = unsafe {
        Vec::from_raw_parts(
            u_guard.as_mut_ptr() as *mut f64,
            u_guard.len(),
            u_guard.capacity(),
        )
    };
    let lower = unsafe {
        Vec::from_raw_parts(
            l_guard.as_mut_ptr() as *mut f64,
            l_guard.len(),
            l_guard.capacity(),
        )
    };
    let upper_change = unsafe {
        Vec::from_raw_parts(
            uc_guard.as_mut_ptr() as *mut f64,
            uc_guard.len(),
            uc_guard.capacity(),
        )
    };
    let lower_change = unsafe {
        Vec::from_raw_parts(
            lc_guard.as_mut_ptr() as *mut f64,
            lc_guard.len(),
            lc_guard.capacity(),
        )
    };

    Ok(GatorOscBatchOutput {
        upper,
        lower,
        upper_change,
        lower_change,
        combos,
        rows,
        cols,
    })
}

//======= WASM Bindings =======

#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct GatorOscJsOutput {
    pub values: Vec<f64>, // Flattened [upper..., lower..., upper_change..., lower_change...]
    pub rows: usize,      // 4 for gatorosc
    pub cols: usize,      // data length
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn gatorosc_js(
    data: &[f64],
    jaws_length: usize,
    jaws_shift: usize,
    teeth_length: usize,
    teeth_shift: usize,
    lips_length: usize,
    lips_shift: usize,
) -> Result<JsValue, JsValue> {
    let params = GatorOscParams {
        jaws_length: Some(jaws_length),
        jaws_shift: Some(jaws_shift),
        teeth_length: Some(teeth_length),
        teeth_shift: Some(teeth_shift),
        lips_length: Some(lips_length),
        lips_shift: Some(lips_shift),
    };
    let input = GatorOscInput::from_slice(data, params);

    // Single allocation for all outputs (flattened)
    let len = data.len();
    let mut values = vec![0.0; 4 * len];

    // Split into mutable slices for each output
    let (upper_part, rest) = values.split_at_mut(len);
    let (lower_part, rest) = rest.split_at_mut(len);
    let (upper_change_part, lower_change_part) = rest.split_at_mut(len);

    // Compute using zero-allocation helper
    gatorosc_into_slice(
        upper_part,
        lower_part,
        upper_change_part,
        lower_change_part,
        &input,
        Kernel::Auto,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let output = GatorOscJsOutput {
        values,
        rows: 4,
        cols: len,
    };

    serde_wasm_bindgen::to_value(&output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn gatorosc_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn gatorosc_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn gatorosc_into(
    in_ptr: *const f64,
    upper_ptr: *mut f64,
    lower_ptr: *mut f64,
    upper_change_ptr: *mut f64,
    lower_change_ptr: *mut f64,
    len: usize,
    jaws_length: usize,
    jaws_shift: usize,
    teeth_length: usize,
    teeth_shift: usize,
    lips_length: usize,
    lips_shift: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null()
        || upper_ptr.is_null()
        || lower_ptr.is_null()
        || upper_change_ptr.is_null()
        || lower_change_ptr.is_null()
    {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let params = GatorOscParams {
            jaws_length: Some(jaws_length),
            jaws_shift: Some(jaws_shift),
            teeth_length: Some(teeth_length),
            teeth_shift: Some(teeth_shift),
            lips_length: Some(lips_length),
            lips_shift: Some(lips_shift),
        };
        let input = GatorOscInput::from_slice(data, params);

        // Check for aliasing - if any output pointer equals input pointer
        let needs_temp = in_ptr == upper_ptr as *const f64
            || in_ptr == lower_ptr as *const f64
            || in_ptr == upper_change_ptr as *const f64
            || in_ptr == lower_change_ptr as *const f64;

        if needs_temp {
            // Use single temporary buffer for all outputs
            let mut temp = vec![0.0; 4 * len];

            // Split into slices for computation
            let (temp_upper, rest) = temp.split_at_mut(len);
            let (temp_lower, rest) = rest.split_at_mut(len);
            let (temp_upper_change, temp_lower_change) = rest.split_at_mut(len);

            gatorosc_into_slice(
                temp_upper,
                temp_lower,
                temp_upper_change,
                temp_lower_change,
                &input,
                Kernel::Auto,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

            // Copy results to output pointers
            let upper_out = std::slice::from_raw_parts_mut(upper_ptr, len);
            let lower_out = std::slice::from_raw_parts_mut(lower_ptr, len);
            let upper_change_out = std::slice::from_raw_parts_mut(upper_change_ptr, len);
            let lower_change_out = std::slice::from_raw_parts_mut(lower_change_ptr, len);

            upper_out.copy_from_slice(temp_upper);
            lower_out.copy_from_slice(temp_lower);
            upper_change_out.copy_from_slice(temp_upper_change);
            lower_change_out.copy_from_slice(temp_lower_change);
        } else {
            // Direct computation into output buffers
            let upper_out = std::slice::from_raw_parts_mut(upper_ptr, len);
            let lower_out = std::slice::from_raw_parts_mut(lower_ptr, len);
            let upper_change_out = std::slice::from_raw_parts_mut(upper_change_ptr, len);
            let lower_change_out = std::slice::from_raw_parts_mut(lower_change_ptr, len);

            gatorosc_into_slice(
                upper_out,
                lower_out,
                upper_change_out,
                lower_change_out,
                &input,
                Kernel::Auto,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct GatorOscBatchConfig {
    pub jaws_length_range: (usize, usize, usize),
    pub jaws_shift_range: (usize, usize, usize),
    pub teeth_length_range: (usize, usize, usize),
    pub teeth_shift_range: (usize, usize, usize),
    pub lips_length_range: (usize, usize, usize),
    pub lips_shift_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct GatorOscBatchJsOutput {
    pub values: Vec<f64>, // Flattened [upper..., lower..., upper_change..., lower_change...]
    pub combos: Vec<GatorOscParams>,
    pub rows: usize,    // Number of parameter combinations
    pub cols: usize,    // Data length
    pub outputs: usize, // 4 for gatorosc
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = gatorosc_batch)]
pub fn gatorosc_batch_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: GatorOscBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = GatorOscBatchRange {
        jaws_length: config.jaws_length_range,
        jaws_shift: config.jaws_shift_range,
        teeth_length: config.teeth_length_range,
        teeth_shift: config.teeth_shift_range,
        lips_length: config.lips_length_range,
        lips_shift: config.lips_shift_range,
    };

    // Calculate total combinations (checked)
    let combos = expand_grid_gatorosc(&sweep)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let n_combos = combos.len();
    let len = data.len();

    // Single allocation for all outputs
    let total_size = n_combos
        .checked_mul(len)
        .ok_or_else(|| JsValue::from_str("gatorosc_batch_js: rows*cols overflow"))?;
    let slots = total_size
        .checked_mul(4)
        .ok_or_else(|| JsValue::from_str("gatorosc_batch_js: output size overflow"))?;
    let mut values = vec![0.0; slots];

    // Split into mutable slices for each output
    let (upper_part, rest) = values.split_at_mut(total_size);
    let (lower_part, rest) = rest.split_at_mut(total_size);
    let (upper_change_part, lower_change_part) = rest.split_at_mut(total_size);

    // Use zero-allocation batch function
    gatorosc_batch_inner_into(
        data,
        &sweep,
        Kernel::Auto,
        false, // No parallel in WASM
        upper_part,
        lower_part,
        upper_change_part,
        lower_change_part,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = GatorOscBatchJsOutput {
        values,
        combos,
        rows: n_combos,
        cols: len,
        outputs: 4,
    };

    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn gatorosc_batch_into(
    in_ptr: *const f64,
    upper_ptr: *mut f64,
    lower_ptr: *mut f64,
    upper_change_ptr: *mut f64,
    lower_change_ptr: *mut f64,
    len: usize,
    jaws_length_start: usize,
    jaws_length_end: usize,
    jaws_length_step: usize,
    jaws_shift_start: usize,
    jaws_shift_end: usize,
    jaws_shift_step: usize,
    teeth_length_start: usize,
    teeth_length_end: usize,
    teeth_length_step: usize,
    teeth_shift_start: usize,
    teeth_shift_end: usize,
    teeth_shift_step: usize,
    lips_length_start: usize,
    lips_length_end: usize,
    lips_length_step: usize,
    lips_shift_start: usize,
    lips_shift_end: usize,
    lips_shift_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null()
        || upper_ptr.is_null()
        || lower_ptr.is_null()
        || upper_change_ptr.is_null()
        || lower_change_ptr.is_null()
    {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let sweep = GatorOscBatchRange {
            jaws_length: (jaws_length_start, jaws_length_end, jaws_length_step),
            jaws_shift: (jaws_shift_start, jaws_shift_end, jaws_shift_step),
            teeth_length: (teeth_length_start, teeth_length_end, teeth_length_step),
            teeth_shift: (teeth_shift_start, teeth_shift_end, teeth_shift_step),
            lips_length: (lips_length_start, lips_length_end, lips_length_step),
            lips_shift: (lips_shift_start, lips_shift_end, lips_shift_step),
        };

        // Calculate number of combinations (checked)
        let combos = expand_grid_gatorosc(&sweep)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let n_combos = combos.len();
        let total_size = n_combos
            .checked_mul(len)
            .ok_or_else(|| JsValue::from_str("gatorosc_batch_into: rows*cols overflow"))?;

        // Create output slices
        let upper_out = std::slice::from_raw_parts_mut(upper_ptr, total_size);
        let lower_out = std::slice::from_raw_parts_mut(lower_ptr, total_size);
        let upper_change_out = std::slice::from_raw_parts_mut(upper_change_ptr, total_size);
        let lower_change_out = std::slice::from_raw_parts_mut(lower_change_ptr, total_size);

        // Use zero-allocation batch function
        gatorosc_batch_inner_into(
            data,
            &sweep,
            Kernel::Auto,
            false, // No parallel in WASM
            upper_out,
            lower_out,
            upper_change_out,
            lower_change_out,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(n_combos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    #[cfg(feature = "proptest")]
    use proptest::prelude::*;

    fn check_gatorosc_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = GatorOscParams::default();
        let input = GatorOscInput::from_candles(&candles, "close", default_params);
        let output = gatorosc_with_kernel(&input, kernel)?;
        assert_eq!(output.upper.len(), candles.close.len());
        Ok(())
    }

    #[test]
    #[cfg(not(feature = "wasm"))]
    fn test_gatorosc_into_matches_api() -> Result<(), Box<dyn std::error::Error>> {
        // Build a small but non-trivial input (with a few NaNs at the front)
        let len = 256;
        let mut data = vec![0.0_f64; len];
        for i in 0..len {
            data[i] = (i as f64).sin() * 10.0 + ((i % 7) as f64) * 0.25;
        }
        // Introduce some NaNs at the start to exercise warmup handling with first-valid > 0
        if len >= 3 {
            data[0] = f64::NAN;
            data[1] = f64::NAN;
            data[2] = f64::NAN;
        }

        let input = GatorOscInput::from_slice(&data, GatorOscParams::default());

        // Baseline via Vec-returning API
        let baseline = gatorosc(&input)?;

        // Preallocate destination buffers
        let mut up = vec![0.0; len];
        let mut lo = vec![0.0; len];
        let mut upc = vec![0.0; len];
        let mut loc = vec![0.0; len];

        // Zero-allocation into API
        gatorosc_into(&input, &mut up, &mut lo, &mut upc, &mut loc)?;

        assert_eq!(baseline.upper.len(), len);
        assert_eq!(baseline.lower.len(), len);
        assert_eq!(baseline.upper_change.len(), len);
        assert_eq!(baseline.lower_change.len(), len);

        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b)
        }

        for i in 0..len {
            assert!(
                eq_or_both_nan(baseline.upper[i], up[i]),
                "upper mismatch at {}: {:?} vs {:?}",
                i,
                baseline.upper[i],
                up[i]
            );
            assert!(
                eq_or_both_nan(baseline.lower[i], lo[i]),
                "lower mismatch at {}: {:?} vs {:?}",
                i,
                baseline.lower[i],
                lo[i]
            );
            assert!(
                eq_or_both_nan(baseline.upper_change[i], upc[i]),
                "upper_change mismatch at {}: {:?} vs {:?}",
                i,
                baseline.upper_change[i],
                upc[i]
            );
            assert!(
                eq_or_both_nan(baseline.lower_change[i], loc[i]),
                "lower_change mismatch at {}: {:?} vs {:?}",
                i,
                baseline.lower_change[i],
                loc[i]
            );
        }

        Ok(())
    }

    fn check_gatorosc_nan_handling(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = GatorOscInput::from_candles(&candles, "close", GatorOscParams::default());
        let output = gatorosc_with_kernel(&input, kernel)?;
        assert_eq!(output.upper.len(), candles.close.len());
        if output.upper.len() > 24 {
            for &val in &output.upper[24..] {
                assert!(!val.is_nan(), "Found unexpected NaN in upper");
            }
        }
        Ok(())
    }

    fn check_gatorosc_zero_setting(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [10.0, 20.0, 30.0];
        let params = GatorOscParams {
            jaws_length: Some(0),
            ..Default::default()
        };
        let input = GatorOscInput::from_slice(&data, params);
        let res = gatorosc_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] GatorOsc should fail with zero setting",
            test_name
        );
        Ok(())
    }

    fn check_gatorosc_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single = [42.0];
        let params = GatorOscParams::default();
        let input = GatorOscInput::from_slice(&single, params);
        let res = gatorosc_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] GatorOsc should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_gatorosc_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = GatorOscInput::with_default_candles(&candles);
        let output = gatorosc_with_kernel(&input, kernel)?;
        assert_eq!(output.upper.len(), candles.close.len());
        Ok(())
    }

    fn check_gatorosc_batch_default_row(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = GatorOscBatchBuilder::new()
            .kernel(kernel)
            .apply_slice(&c.close)?;
        let def = GatorOscParams::default();
        let row = output
            .combos
            .iter()
            .position(|p| {
                p.jaws_length.unwrap_or(13) == def.jaws_length.unwrap_or(13)
                    && p.jaws_shift.unwrap_or(8) == def.jaws_shift.unwrap_or(8)
                    && p.teeth_length.unwrap_or(8) == def.teeth_length.unwrap_or(8)
                    && p.teeth_shift.unwrap_or(5) == def.teeth_shift.unwrap_or(5)
                    && p.lips_length.unwrap_or(5) == def.lips_length.unwrap_or(5)
                    && p.lips_shift.unwrap_or(3) == def.lips_shift.unwrap_or(3)
            })
            .expect("default row missing");
        let u = &output.upper[row * output.cols..][..output.cols];
        assert_eq!(u.len(), c.close.len());
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_gatorosc_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Define comprehensive parameter combinations
        let test_params = vec![
            GatorOscParams::default(), // Default values
            // Minimum values
            GatorOscParams {
                jaws_length: Some(2),
                jaws_shift: Some(0),
                teeth_length: Some(2),
                teeth_shift: Some(0),
                lips_length: Some(2),
                lips_shift: Some(0),
            },
            // Small values
            GatorOscParams {
                jaws_length: Some(5),
                jaws_shift: Some(2),
                teeth_length: Some(4),
                teeth_shift: Some(1),
                lips_length: Some(3),
                lips_shift: Some(1),
            },
            // Medium values
            GatorOscParams {
                jaws_length: Some(20),
                jaws_shift: Some(10),
                teeth_length: Some(15),
                teeth_shift: Some(8),
                lips_length: Some(10),
                lips_shift: Some(5),
            },
            // Large values
            GatorOscParams {
                jaws_length: Some(50),
                jaws_shift: Some(20),
                teeth_length: Some(30),
                teeth_shift: Some(15),
                lips_length: Some(20),
                lips_shift: Some(10),
            },
            // Edge case: jaws < teeth < lips (unusual)
            GatorOscParams {
                jaws_length: Some(5),
                jaws_shift: Some(3),
                teeth_length: Some(8),
                teeth_shift: Some(5),
                lips_length: Some(13),
                lips_shift: Some(8),
            },
            // Edge case: all same length
            GatorOscParams {
                jaws_length: Some(10),
                jaws_shift: Some(5),
                teeth_length: Some(10),
                teeth_shift: Some(5),
                lips_length: Some(10),
                lips_shift: Some(5),
            },
            // Edge case: no shifts
            GatorOscParams {
                jaws_length: Some(13),
                jaws_shift: Some(0),
                teeth_length: Some(8),
                teeth_shift: Some(0),
                lips_length: Some(5),
                lips_shift: Some(0),
            },
            // Edge case: large shifts
            GatorOscParams {
                jaws_length: Some(10),
                jaws_shift: Some(20),
                teeth_length: Some(8),
                teeth_shift: Some(15),
                lips_length: Some(5),
                lips_shift: Some(10),
            },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = GatorOscInput::from_candles(&candles, "close", params.clone());
            let output = gatorosc_with_kernel(&input, kernel)?;

            // Check upper values
            for (i, &val) in output.upper.iter().enumerate() {
                if val.is_nan() {
                    continue; // NaN values are expected during warmup
                }

                let bits = val.to_bits();

                // Check all three poison patterns
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 in upper output with params: {:?} (param set {})",
                        test_name, val, bits, i, params, param_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 in upper output with params: {:?} (param set {})",
                        test_name, val, bits, i, params, param_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 in upper output with params: {:?} (param set {})",
                        test_name, val, bits, i, params, param_idx
                    );
                }
            }

            // Check lower values
            for (i, &val) in output.lower.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 in lower output with params: {:?} (param set {})",
                        test_name, val, bits, i, params, param_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 in lower output with params: {:?} (param set {})",
                        test_name, val, bits, i, params, param_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 in lower output with params: {:?} (param set {})",
                        test_name, val, bits, i, params, param_idx
                    );
                }
            }

            // Check upper_change values
            for (i, &val) in output.upper_change.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 in upper_change output with params: {:?} (param set {})",
                        test_name, val, bits, i, params, param_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 in upper_change output with params: {:?} (param set {})",
                        test_name, val, bits, i, params, param_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 in upper_change output with params: {:?} (param set {})",
                        test_name, val, bits, i, params, param_idx
                    );
                }
            }

            // Check lower_change values
            for (i, &val) in output.lower_change.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 in lower_change output with params: {:?} (param set {})",
                        test_name, val, bits, i, params, param_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 in lower_change output with params: {:?} (param set {})",
                        test_name, val, bits, i, params, param_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 in lower_change output with params: {:?} (param set {})",
                        test_name, val, bits, i, params, param_idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_gatorosc_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(()) // No-op in release builds
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_gatorosc_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Generate realistic market data with various parameter combinations
        let strat = (
            // Jaws parameters (longest EMA)
            (5usize..=50), // jaws_length
            (1usize..=10), // jaws_shift (must be at least 1)
            // Teeth parameters (medium EMA)
            (3usize..=30), // teeth_length
            (1usize..=8),  // teeth_shift (must be at least 1)
            // Lips parameters (shortest EMA)
            (2usize..=20), // lips_length
            (1usize..=5),  // lips_shift (must be at least 1)
        )
            .prop_flat_map(
                |(jaws_len, jaws_shift, teeth_len, teeth_shift, lips_len, lips_shift)| {
                    // Calculate minimum data needed
                    let min_data_len = jaws_len.max(teeth_len).max(lips_len)
                        + jaws_shift.max(teeth_shift).max(lips_shift)
                        + 10;
                    (
                        // Data generation: realistic price data with enough points
                        prop::collection::vec(
                            (10.0f64..100000.0f64).prop_filter("finite", |x| x.is_finite()),
                            min_data_len..400,
                        ),
                        Just(jaws_len),
                        Just(jaws_shift),
                        Just(teeth_len),
                        Just(teeth_shift),
                        Just(lips_len),
                        Just(lips_shift),
                    )
                },
            );

        proptest::test_runner::TestRunner::default()
            .run(
                &strat,
                |(
                    data,
                    jaws_length,
                    jaws_shift,
                    teeth_length,
                    teeth_shift,
                    lips_length,
                    lips_shift,
                )| {
                    let params = GatorOscParams {
                        jaws_length: Some(jaws_length),
                        jaws_shift: Some(jaws_shift),
                        teeth_length: Some(teeth_length),
                        teeth_shift: Some(teeth_shift),
                        lips_length: Some(lips_length),
                        lips_shift: Some(lips_shift),
                    };
                    let input = GatorOscInput::from_slice(&data, params);

                    // Get outputs from test kernel and reference (scalar)
                    let test_output = gatorosc_with_kernel(&input, kernel).unwrap();
                    let ref_output = gatorosc_with_kernel(&input, Kernel::Scalar).unwrap();

                    let GatorOscOutput {
                        upper,
                        lower,
                        upper_change,
                        lower_change,
                    } = test_output;
                    let GatorOscOutput {
                        upper: ref_upper,
                        lower: ref_lower,
                        upper_change: ref_upper_change,
                        lower_change: ref_lower_change,
                    } = ref_output;

                    // Property 1: Warmup period validation
                    // Check which values are actually NaN vs finite
                    let mut first_finite_upper = None;
                    let mut first_finite_lower = None;

                    for i in 0..upper.len() {
                        if upper[i].is_finite() && first_finite_upper.is_none() {
                            first_finite_upper = Some(i);
                        }
                        if lower[i].is_finite() && first_finite_lower.is_none() {
                            first_finite_lower = Some(i);
                        }
                    }

                    // At least some warmup period should exist
                    if let Some(idx) = first_finite_upper {
                        prop_assert!(
						idx > 0,
						"Upper should have at least some warmup period, but first finite value is at index {}",
						idx
					);
                    }

                    if let Some(idx) = first_finite_lower {
                        prop_assert!(
						idx > 0,
						"Lower should have at least some warmup period, but first finite value is at index {}",
						idx
					);
                    }

                    // Property 2: Output finiteness after initial warmup
                    // After sufficient data points, outputs should be finite
                    let safe_start = (jaws_length.max(teeth_length).max(lips_length)
                        + jaws_shift.max(teeth_shift).max(lips_shift))
                    .min(data.len() - 1);

                    for i in safe_start..upper.len() {
                        prop_assert!(
                            upper[i].is_finite() || upper[i].is_nan(),
                            "Upper should be finite or NaN at index {}: got {}",
                            i,
                            upper[i]
                        );
                    }

                    for i in safe_start..lower.len() {
                        prop_assert!(
                            lower[i].is_finite() || lower[i].is_nan(),
                            "Lower should be finite or NaN at index {}: got {}",
                            i,
                            lower[i]
                        );
                    }

                    // Property 3: Kernel consistency (compare with scalar reference)
                    for i in 0..data.len() {
                        // Check upper consistency
                        if upper[i].is_finite() && ref_upper[i].is_finite() {
                            let ulp_diff = upper[i].to_bits().abs_diff(ref_upper[i].to_bits());
                            prop_assert!(
                                (upper[i] - ref_upper[i]).abs() <= 1e-9 || ulp_diff <= 4,
                                "Upper mismatch at {}: {} vs {} (ULP={})",
                                i,
                                upper[i],
                                ref_upper[i],
                                ulp_diff
                            );
                        } else {
                            prop_assert_eq!(
                                upper[i].is_nan(),
                                ref_upper[i].is_nan(),
                                "Upper NaN mismatch at {}",
                                i
                            );
                        }

                        // Check lower consistency
                        if lower[i].is_finite() && ref_lower[i].is_finite() {
                            let ulp_diff = lower[i].to_bits().abs_diff(ref_lower[i].to_bits());
                            prop_assert!(
                                (lower[i] - ref_lower[i]).abs() <= 1e-9 || ulp_diff <= 4,
                                "Lower mismatch at {}: {} vs {} (ULP={})",
                                i,
                                lower[i],
                                ref_lower[i],
                                ulp_diff
                            );
                        } else {
                            prop_assert_eq!(
                                lower[i].is_nan(),
                                ref_lower[i].is_nan(),
                                "Lower NaN mismatch at {}",
                                i
                            );
                        }

                        // Check upper_change consistency
                        if upper_change[i].is_finite() && ref_upper_change[i].is_finite() {
                            let ulp_diff = upper_change[i]
                                .to_bits()
                                .abs_diff(ref_upper_change[i].to_bits());
                            prop_assert!(
                                (upper_change[i] - ref_upper_change[i]).abs() <= 1e-9
                                    || ulp_diff <= 4,
                                "Upper change mismatch at {}: {} vs {} (ULP={})",
                                i,
                                upper_change[i],
                                ref_upper_change[i],
                                ulp_diff
                            );
                        } else {
                            prop_assert_eq!(
                                upper_change[i].is_nan(),
                                ref_upper_change[i].is_nan(),
                                "Upper change NaN mismatch at {}",
                                i
                            );
                        }

                        // Check lower_change consistency
                        if lower_change[i].is_finite() && ref_lower_change[i].is_finite() {
                            let ulp_diff = lower_change[i]
                                .to_bits()
                                .abs_diff(ref_lower_change[i].to_bits());
                            prop_assert!(
                                (lower_change[i] - ref_lower_change[i]).abs() <= 1e-9
                                    || ulp_diff <= 4,
                                "Lower change mismatch at {}: {} vs {} (ULP={})",
                                i,
                                lower_change[i],
                                ref_lower_change[i],
                                ulp_diff
                            );
                        } else {
                            prop_assert_eq!(
                                lower_change[i].is_nan(),
                                ref_lower_change[i].is_nan(),
                                "Lower change NaN mismatch at {}",
                                i
                            );
                        }
                    }

                    // Property 4: Upper always positive/zero, lower always negative/zero
                    for i in safe_start..upper.len() {
                        prop_assert!(
                            upper[i] >= -1e-10, // Allow tiny numerical errors
                            "Upper should be non-negative at {}: got {}",
                            i,
                            upper[i]
                        );
                    }

                    for i in safe_start..lower.len() {
                        prop_assert!(
                            lower[i] <= 1e-10, // Allow tiny numerical errors
                            "Lower should be non-positive at {}: got {}",
                            i,
                            lower[i]
                        );
                    }

                    // Property 5: Change values are correct 1-period differences
                    for i in 1..data.len() {
                        if !upper[i].is_nan() && !upper[i - 1].is_nan() {
                            let expected_change = upper[i] - upper[i - 1];
                            if upper_change[i].is_finite() {
                                prop_assert!(
                                    (upper_change[i] - expected_change).abs() <= 1e-9,
                                    "Upper change incorrect at {}: got {}, expected {}",
                                    i,
                                    upper_change[i],
                                    expected_change
                                );
                            }
                        }

                        if !lower[i].is_nan() && !lower[i - 1].is_nan() {
                            // Note: lower_change is negated in the implementation
                            let expected_change = -(lower[i] - lower[i - 1]);
                            if lower_change[i].is_finite() {
                                prop_assert!(
                                    (lower_change[i] - expected_change).abs() <= 1e-9,
                                    "Lower change incorrect at {}: got {}, expected {}",
                                    i,
                                    lower_change[i],
                                    expected_change
                                );
                            }
                        }
                    }

                    // Property 6: EMA bounds - outputs influenced by input range
                    let min_price = data.iter().cloned().fold(f64::INFINITY, f64::min);
                    let max_price = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let price_range = max_price - min_price;

                    for i in safe_start..upper.len() {
                        // Upper is abs(Jaws - Teeth), bounded by price range
                        prop_assert!(
                            upper[i] <= price_range + 1e-9,
                            "Upper exceeds price range at {}: {} > {}",
                            i,
                            upper[i],
                            price_range
                        );
                    }

                    for i in safe_start..lower.len() {
                        // Lower is -abs(Teeth - Lips), bounded by negative price range
                        prop_assert!(
                            lower[i] >= -(price_range + 1e-9),
                            "Lower exceeds negative price range at {}: {} < {}",
                            i,
                            lower[i],
                            -price_range
                        );
                    }

                    // Property 7: Constant data produces minimal oscillation
                    if data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-9) {
                        // With constant data, all EMAs converge to same value
                        // So upper and lower should approach zero
                        for i in (data.len() * 3 / 4)..data.len() {
                            if upper[i].is_finite() {
                                prop_assert!(
                                    upper[i].abs() <= 1e-6,
                                    "Upper should be near zero with constant data at {}: {}",
                                    i,
                                    upper[i]
                                );
                            }
                            if lower[i].is_finite() {
                                prop_assert!(
                                    lower[i].abs() <= 1e-6,
                                    "Lower should be near zero with constant data at {}: {}",
                                    i,
                                    lower[i]
                                );
                            }
                        }
                    }

                    // Property 8: No poison values in output
                    for i in 0..data.len() {
                        if upper[i].is_finite() {
                            prop_assert_ne!(upper[i].to_bits(), 0x11111111_11111111u64);
                            prop_assert_ne!(upper[i].to_bits(), 0x22222222_22222222u64);
                            prop_assert_ne!(upper[i].to_bits(), 0x33333333_33333333u64);
                        }
                        if lower[i].is_finite() {
                            prop_assert_ne!(lower[i].to_bits(), 0x11111111_11111111u64);
                            prop_assert_ne!(lower[i].to_bits(), 0x22222222_22222222u64);
                            prop_assert_ne!(lower[i].to_bits(), 0x33333333_33333333u64);
                        }
                        if upper_change[i].is_finite() {
                            prop_assert_ne!(upper_change[i].to_bits(), 0x11111111_11111111u64);
                            prop_assert_ne!(upper_change[i].to_bits(), 0x22222222_22222222u64);
                            prop_assert_ne!(upper_change[i].to_bits(), 0x33333333_33333333u64);
                        }
                        if lower_change[i].is_finite() {
                            prop_assert_ne!(lower_change[i].to_bits(), 0x11111111_11111111u64);
                            prop_assert_ne!(lower_change[i].to_bits(), 0x22222222_22222222u64);
                            prop_assert_ne!(lower_change[i].to_bits(), 0x33333333_33333333u64);
                        }
                    }

                    Ok(())
                },
            )
            .unwrap();

        Ok(())
    }

    macro_rules! generate_all_gatorosc_tests {
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

    generate_all_gatorosc_tests!(
        check_gatorosc_partial_params,
        check_gatorosc_nan_handling,
        check_gatorosc_zero_setting,
        check_gatorosc_small_dataset,
        check_gatorosc_default_candles,
        check_gatorosc_batch_default_row,
        check_gatorosc_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_gatorosc_tests!(check_gatorosc_property);
    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = GatorOscBatchBuilder::new()
            .kernel(kernel)
            .apply_slice(&c.close)?;

        let def = GatorOscParams::default();
        let row = output
            .combos
            .iter()
            .position(|p| {
                p.jaws_length == def.jaws_length
                    && p.jaws_shift == def.jaws_shift
                    && p.teeth_length == def.teeth_length
                    && p.teeth_shift == def.teeth_shift
                    && p.lips_length == def.lips_length
                    && p.lips_shift == def.lips_shift
            })
            .expect("default row missing");

        let upper = &output.upper[row * output.cols..][..output.cols];
        let lower = &output.lower[row * output.cols..][..output.cols];

        assert_eq!(upper.len(), c.close.len());
        assert_eq!(lower.len(), c.close.len());
        Ok(())
    }

    fn check_batch_multi_param_sweep(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let builder = GatorOscBatchBuilder::new()
            .kernel(kernel)
            .jaws_length_range(8, 14, 3)
            .jaws_shift_range(5, 8, 3)
            .teeth_length_range(5, 8, 3)
            .teeth_shift_range(3, 5, 2)
            .lips_length_range(3, 5, 2)
            .lips_shift_range(2, 3, 1);

        let output = builder.apply_slice(&c.close)?;
        // Shape checks
        assert!(output.rows > 1, "Should have multiple param sweeps");
        assert_eq!(output.cols, c.close.len());

        // Check at least one output row is not all-NaN (assuming input is valid)
        let some_upper = output
            .upper
            .chunks(output.cols)
            .any(|row| row.iter().any(|&x| !x.is_nan()));
        assert!(some_upper);

        Ok(())
    }

    fn check_batch_not_enough_data(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let short = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut sweep = GatorOscBatchRange::default();
        sweep.jaws_length = (6, 6, 0);

        let res = gatorosc_batch_with_kernel(&short, &sweep, kernel);
        assert!(res.is_err());
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let test_configs = vec![
            // Single parameter sweeps
            (5, 20, 5, 8, 8, 0, 8, 8, 0, 5, 5, 0, 5, 5, 0, 3, 3, 0), // Vary jaws_length
            (13, 13, 0, 3, 10, 2, 8, 8, 0, 5, 5, 0, 5, 5, 0, 3, 3, 0), // Vary jaws_shift
            (13, 13, 0, 8, 8, 0, 3, 10, 2, 5, 5, 0, 5, 5, 0, 3, 3, 0), // Vary teeth_length
            (13, 13, 0, 8, 8, 0, 8, 8, 0, 2, 8, 2, 5, 5, 0, 3, 3, 0), // Vary teeth_shift
            (13, 13, 0, 8, 8, 0, 8, 8, 0, 5, 5, 0, 2, 8, 2, 3, 3, 0), // Vary lips_length
            (13, 13, 0, 8, 8, 0, 8, 8, 0, 5, 5, 0, 5, 5, 0, 1, 5, 1), // Vary lips_shift
            // Multi-parameter sweeps
            (8, 14, 3, 5, 8, 3, 5, 8, 3, 3, 5, 2, 3, 5, 2, 2, 3, 1), // Vary all parameters
            // Static configurations
            (10, 10, 0, 5, 5, 0, 8, 8, 0, 4, 4, 0, 5, 5, 0, 2, 2, 0), // All static
            (13, 13, 0, 8, 8, 0, 8, 8, 0, 5, 5, 0, 5, 5, 0, 3, 3, 0), // Default static
            // Edge cases
            (2, 5, 1, 0, 3, 1, 2, 5, 1, 0, 3, 1, 2, 5, 1, 0, 3, 1), // Small values
            (
                30, 50, 10, 10, 20, 5, 20, 30, 5, 8, 15, 3, 10, 20, 5, 5, 10, 2,
            ), // Large values
        ];

        for (
            cfg_idx,
            &(
                jl_s,
                jl_e,
                jl_st,
                js_s,
                js_e,
                js_st,
                tl_s,
                tl_e,
                tl_st,
                ts_s,
                ts_e,
                ts_st,
                ll_s,
                ll_e,
                ll_st,
                ls_s,
                ls_e,
                ls_st,
            ),
        ) in test_configs.iter().enumerate()
        {
            let output = GatorOscBatchBuilder::new()
                .kernel(kernel)
                .jaws_length_range(jl_s, jl_e, jl_st)
                .jaws_shift_range(js_s, js_e, js_st)
                .teeth_length_range(tl_s, tl_e, tl_st)
                .teeth_shift_range(ts_s, ts_e, ts_st)
                .lips_length_range(ll_s, ll_e, ll_st)
                .lips_shift_range(ls_s, ls_e, ls_st)
                .apply_slice(&c.close)?;

            // Helper function to check poison in a matrix
            let check_poison = |matrix: &[f64], matrix_name: &str| {
                for (idx, &val) in matrix.iter().enumerate() {
                    if val.is_nan() {
                        continue;
                    }

                    let bits = val.to_bits();
                    let row = idx / output.cols;
                    let col = idx % output.cols;
                    let combo = &output.combos[row];

                    if bits == 0x11111111_11111111 {
                        panic!(
							"[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
							at row {} col {} (flat index {}) in {} output with params: jl={}, js={}, tl={}, ts={}, ll={}, ls={}",
							test, cfg_idx, val, bits, row, col, idx, matrix_name,
							combo.jaws_length.unwrap_or(13), combo.jaws_shift.unwrap_or(8),
							combo.teeth_length.unwrap_or(8), combo.teeth_shift.unwrap_or(5),
							combo.lips_length.unwrap_or(5), combo.lips_shift.unwrap_or(3)
						);
                    }

                    if bits == 0x22222222_22222222 {
                        panic!(
							"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
							at row {} col {} (flat index {}) in {} output with params: jl={}, js={}, tl={}, ts={}, ll={}, ls={}",
							test, cfg_idx, val, bits, row, col, idx, matrix_name,
							combo.jaws_length.unwrap_or(13), combo.jaws_shift.unwrap_or(8),
							combo.teeth_length.unwrap_or(8), combo.teeth_shift.unwrap_or(5),
							combo.lips_length.unwrap_or(5), combo.lips_shift.unwrap_or(3)
						);
                    }

                    if bits == 0x33333333_33333333 {
                        panic!(
							"[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
							at row {} col {} (flat index {}) in {} output with params: jl={}, js={}, tl={}, ts={}, ll={}, ls={}",
							test, cfg_idx, val, bits, row, col, idx, matrix_name,
							combo.jaws_length.unwrap_or(13), combo.jaws_shift.unwrap_or(8),
							combo.teeth_length.unwrap_or(8), combo.teeth_shift.unwrap_or(5),
							combo.lips_length.unwrap_or(5), combo.lips_shift.unwrap_or(3)
						);
                    }
                }
            };

            // Check all four output matrices
            check_poison(&output.upper, "upper");
            check_poison(&output.lower, "lower");
            check_poison(&output.upper_change, "upper_change");
            check_poison(&output.lower_change, "lower_change");
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
                #[cfg(all(target_feature = "simd128", target_arch = "wasm32"))]
                #[test] fn [<$fn_name _simd128>]()     {
                    let _ = $fn_name(stringify!([<$fn_name _simd128>]), Kernel::Simd128Batch);
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
    gen_batch_tests!(check_batch_multi_param_sweep);
    gen_batch_tests!(check_batch_not_enough_data);
    gen_batch_tests!(check_batch_no_poison);
}

// ============================================================================
//                               PYTHON BINDINGS
// ============================================================================

#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;

#[cfg(feature = "python")]
#[pyfunction(name = "gatorosc")]
#[pyo3(signature = (data, jaws_length=13, jaws_shift=8, teeth_length=8, teeth_shift=5, lips_length=5, lips_shift=3, kernel=None))]
pub fn gatorosc_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    jaws_length: usize,
    jaws_shift: usize,
    teeth_length: usize,
    teeth_shift: usize,
    lips_length: usize,
    lips_shift: usize,
    kernel: Option<&str>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let params = GatorOscParams {
        jaws_length: Some(jaws_length),
        jaws_shift: Some(jaws_shift),
        teeth_length: Some(teeth_length),
        teeth_shift: Some(teeth_shift),
        lips_length: Some(lips_length),
        lips_shift: Some(lips_shift),
    };
    let input = GatorOscInput::from_slice(slice_in, params);

    let (upper_vec, lower_vec, upper_change_vec, lower_change_vec) = py
        .allow_threads(|| {
            gatorosc_with_kernel(&input, kern)
                .map(|o| (o.upper, o.lower, o.upper_change, o.lower_change))
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok((
        upper_vec.into_pyarray(py),
        lower_vec.into_pyarray(py),
        upper_change_vec.into_pyarray(py),
        lower_change_vec.into_pyarray(py),
    ))
}

#[cfg(feature = "python")]
#[pyclass(name = "GatorOscStream")]
pub struct GatorOscStreamPy {
    stream: GatorOscStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl GatorOscStreamPy {
    #[new]
    #[pyo3(signature = (jaws_length=13, jaws_shift=8, teeth_length=8, teeth_shift=5, lips_length=5, lips_shift=3))]
    fn new(
        jaws_length: usize,
        jaws_shift: usize,
        teeth_length: usize,
        teeth_shift: usize,
        lips_length: usize,
        lips_shift: usize,
    ) -> PyResult<Self> {
        let params = GatorOscParams {
            jaws_length: Some(jaws_length),
            jaws_shift: Some(jaws_shift),
            teeth_length: Some(teeth_length),
            teeth_shift: Some(teeth_shift),
            lips_length: Some(lips_length),
            lips_shift: Some(lips_shift),
        };
        let stream =
            GatorOscStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(GatorOscStreamPy { stream })
    }

    fn update(&mut self, value: f64) -> Option<(f64, f64, f64, f64)> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "gatorosc_batch")]
#[pyo3(signature = (data, jaws_length_range=(13, 13, 0), jaws_shift_range=(8, 8, 0), teeth_length_range=(8, 8, 0), teeth_shift_range=(5, 5, 0), lips_length_range=(5, 5, 0), lips_shift_range=(3, 3, 0), kernel=None))]
pub fn gatorosc_batch_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    jaws_length_range: (usize, usize, usize),
    jaws_shift_range: (usize, usize, usize),
    teeth_length_range: (usize, usize, usize),
    teeth_shift_range: (usize, usize, usize),
    lips_length_range: (usize, usize, usize),
    lips_shift_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;

    let sweep = GatorOscBatchRange {
        jaws_length: jaws_length_range,
        jaws_shift: jaws_shift_range,
        teeth_length: teeth_length_range,
        teeth_shift: teeth_shift_range,
        lips_length: lips_length_range,
        lips_shift: lips_shift_range,
    };

    let combos = expand_grid_gatorosc(&sweep)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = slice_in.len();

    // Pre-allocate output arrays for batch operations
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("gatorosc_batch_py: rows*cols overflow"))?;
    let upper_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let lower_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let upper_change_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let lower_change_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };

    let slice_upper = unsafe { upper_arr.as_slice_mut()? };
    let slice_lower = unsafe { lower_arr.as_slice_mut()? };
    let slice_upper_change = unsafe { upper_change_arr.as_slice_mut()? };
    let slice_lower_change = unsafe { lower_change_arr.as_slice_mut()? };

    let combos = py
        .allow_threads(|| {
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

            // Use zero-allocation batch function
            gatorosc_batch_inner_into(
                slice_in,
                &sweep,
                simd,
                true,
                slice_upper,
                slice_lower,
                slice_upper_change,
                slice_lower_change,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("upper", upper_arr.reshape((rows, cols))?)?;
    dict.set_item("lower", lower_arr.reshape((rows, cols))?)?;
    dict.set_item("upper_change", upper_change_arr.reshape((rows, cols))?)?;
    dict.set_item("lower_change", lower_change_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "jaws_lengths",
        combos
            .iter()
            .map(|p| p.jaws_length.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "jaws_shifts",
        combos
            .iter()
            .map(|p| p.jaws_shift.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "teeth_lengths",
        combos
            .iter()
            .map(|p| p.teeth_length.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "teeth_shifts",
        combos
            .iter()
            .map(|p| p.teeth_shift.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "lips_lengths",
        combos
            .iter()
            .map(|p| p.lips_length.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "lips_shifts",
        combos
            .iter()
            .map(|p| p.lips_shift.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;

    Ok(dict)
}

// ---------------- CUDA Python bindings ----------------
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::cuda_available;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::oscillators::gatorosc_wrapper::CudaGatorOsc;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::utilities::dlpack_cuda::export_f32_cuda_dlpack_2d;
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::context::Context;
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::memory::DeviceBuffer;
#[cfg(all(feature = "python", feature = "cuda"))]
use std::sync::Arc;

// GATOR-specific Python device handle with CAI v3 and DLPack v1.x negotiation
#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(module = "ta_indicators.cuda", name = "GatorDeviceArrayF32", unsendable)]
pub struct DeviceArrayF32GatorPy {
    pub(crate) inner: crate::cuda::moving_averages::DeviceArrayF32,
    _ctx_guard: Arc<Context>,
    _device_id: u32,
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pymethods]
impl DeviceArrayF32GatorPy {
    #[getter]
    fn __cuda_array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("shape", (self.inner.rows, self.inner.cols))?;
        d.set_item("typestr", "<f4")?;
        d.set_item(
            "strides",
            (
                self.inner.cols * std::mem::size_of::<f32>(),
                std::mem::size_of::<f32>(),
            ),
        )?;
        let ptr_val: usize = if self.inner.rows == 0 || self.inner.cols == 0 {
            0
        } else {
            self.inner.buf.as_device_ptr().as_raw() as usize
        };
        d.set_item("data", (ptr_val, false))?;
        d.set_item("version", 3)?;
        Ok(d)
    }

    fn __dlpack_device__(&self) -> (i32, i32) { (2, self._device_id as i32) }

    #[pyo3(signature = (stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__<'py>(
        &mut self,
        py: Python<'py>,
        stream: Option<pyo3::PyObject>,
        max_version: Option<pyo3::PyObject>,
        dl_device: Option<pyo3::PyObject>,
        copy: Option<pyo3::PyObject>,
    ) -> PyResult<pyo3::PyObject> {
        // Compute target device id and validate `dl_device` hint if provided.
        let (kdl, alloc_dev) = self.__dlpack_device__();
        if let Some(dev_obj) = dl_device.as_ref() {
            if let Ok((dev_ty, dev_id)) = dev_obj.extract::<(i32, i32)>(py) {
                if dev_ty != kdl || dev_id != alloc_dev {
                    let wants_copy = copy
                        .as_ref()
                        .and_then(|c| c.extract::<bool>(py).ok())
                        .unwrap_or(false);
                    if wants_copy {
                        return Err(PyValueError::new_err(
                            "__dlpack__(copy=True) not implemented for Gator CUDA handle",
                        ));
                    } else {
                        return Err(PyValueError::new_err(
                            "dl_device mismatch for Gator DLPack tensor",
                        ));
                    }
                }
            }
        }
        let _ = stream;
        // copy=True is not yet supported (no cross-device copy path here).
        if let Some(copy_obj) = copy.as_ref() {
            let do_copy: bool = copy_obj.extract(py)?;
            if do_copy {
                return Err(PyValueError::new_err(
                    "__dlpack__(copy=True) not implemented for Gator CUDA handle",
                ));
            }
        }

        // Move VRAM handle out of this wrapper; the DLPack capsule owns it afterwards.
        let dummy = DeviceBuffer::from_slice(&[])
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let rows = self.inner.rows;
        let cols = self.inner.cols;
        let inner = std::mem::replace(
            &mut self.inner,
            crate::cuda::moving_averages::DeviceArrayF32 {
                buf: dummy,
                rows: 0,
                cols: 0,
            },
        );

        let max_version_bound = max_version.map(|obj| obj.into_bound(py));

        export_f32_cuda_dlpack_2d(py, inner.buf, rows, cols, alloc_dev, max_version_bound)
    }
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "gatorosc_cuda_batch_dev")]
#[pyo3(signature = (data_f32, jaws_length_range=(13,13,0), jaws_shift_range=(8,8,0), teeth_length_range=(8,8,0), teeth_shift_range=(5,5,0), lips_length_range=(5,5,0), lips_shift_range=(3,3,0), device_id=0))]
pub fn gatorosc_cuda_batch_dev_py(
    py: Python<'_>,
    data_f32: numpy::PyReadonlyArray1<'_, f32>,
    jaws_length_range: (usize, usize, usize),
    jaws_shift_range: (usize, usize, usize),
    teeth_length_range: (usize, usize, usize),
    teeth_shift_range: (usize, usize, usize),
    lips_length_range: (usize, usize, usize),
    lips_shift_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<(
    DeviceArrayF32GatorPy,
    DeviceArrayF32GatorPy,
    DeviceArrayF32GatorPy,
    DeviceArrayF32GatorPy,
)> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let data = data_f32.as_slice()?;
    let sweep = GatorOscBatchRange {
        jaws_length: jaws_length_range,
        jaws_shift: jaws_shift_range,
        teeth_length: teeth_length_range,
        teeth_shift: teeth_shift_range,
        lips_length: lips_length_range,
        lips_shift: lips_shift_range,
    };
    let (upper, lower, upper_change, lower_change) = py.allow_threads(|| {
        let cuda = CudaGatorOsc::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let dev_id = cuda.device_id();
        let ctx = cuda.ctx();
        let quad = cuda
            .gatorosc_batch_dev(data, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok::<_, PyErr>(
            (
                DeviceArrayF32GatorPy { inner: quad.upper, _ctx_guard: ctx.clone(), _device_id: dev_id },
                DeviceArrayF32GatorPy { inner: quad.lower, _ctx_guard: ctx.clone(), _device_id: dev_id },
                DeviceArrayF32GatorPy { inner: quad.upper_change, _ctx_guard: ctx.clone(), _device_id: dev_id },
                DeviceArrayF32GatorPy { inner: quad.lower_change, _ctx_guard: ctx, _device_id: dev_id },
            )
        )
    })?;
    Ok((upper, lower, upper_change, lower_change))
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "gatorosc_cuda_many_series_one_param_dev")]
#[pyo3(signature = (prices_tm_f32, cols, rows, jaws_length=13, jaws_shift=8, teeth_length=8, teeth_shift=5, lips_length=5, lips_shift=3, device_id=0))]
pub fn gatorosc_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    prices_tm_f32: numpy::PyReadonlyArray1<'_, f32>,
    cols: usize,
    rows: usize,
    jaws_length: usize,
    jaws_shift: usize,
    teeth_length: usize,
    teeth_shift: usize,
    lips_length: usize,
    lips_shift: usize,
    device_id: usize,
) -> PyResult<(
    DeviceArrayF32GatorPy,
    DeviceArrayF32GatorPy,
    DeviceArrayF32GatorPy,
    DeviceArrayF32GatorPy,
)> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let prices = prices_tm_f32.as_slice()?;
    let expected = cols
        .checked_mul(rows)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;
    if prices.len() != expected {
        return Err(PyValueError::new_err("time-major input length mismatch"));
    }
    let (upper, lower, upper_change, lower_change) = py.allow_threads(|| {
        let cuda = CudaGatorOsc::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let dev_id = cuda.device_id();
        let ctx = cuda.ctx();
        let quad = cuda
            .gatorosc_many_series_one_param_time_major_dev(
                prices,
                cols,
                rows,
                jaws_length,
                jaws_shift,
                teeth_length,
                teeth_shift,
                lips_length,
                lips_shift,
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok::<_, PyErr>(
            (
                DeviceArrayF32GatorPy { inner: quad.upper, _ctx_guard: ctx.clone(), _device_id: dev_id },
                DeviceArrayF32GatorPy { inner: quad.lower, _ctx_guard: ctx.clone(), _device_id: dev_id },
                DeviceArrayF32GatorPy { inner: quad.upper_change, _ctx_guard: ctx.clone(), _device_id: dev_id },
                DeviceArrayF32GatorPy { inner: quad.lower_change, _ctx_guard: ctx, _device_id: dev_id },
            )
        )
    })?;
    Ok((upper, lower, upper_change, lower_change))
}
