//! # Ehlers Instantaneous Trend (EIT)
//!
//! John Ehlers' digital signal processing trendline using adaptive MESA-based spectral analysis. Smooths price with minimal lag and adapts dynamically to dominant cycle period.
//!
//! ## Parameters
//! - **warmup_bars**: Initial bars for filter state (default: 12)
//! - **max_dc_period**: Maximum cycle period (default: 50)
//!
//! ## Errors
//! - **EmptyInputData**: Input data is empty
//! - **AllValuesNaN**: All input values are NaN
//! - **NotEnoughDataForWarmup**: Not enough data for warmup bars
//!
//! ## Returns
//! - **Ok(EhlersITrendOutput)**: Vec<f64> matching input length
//! - **Err(EhlersITrendError)**: On error
//!
//! ## SIMD note
//! **SIMD acceleration is not implemented for this indicator. SIMD (AVX2/AVX512) is ineffective for memory-bound or highly sequential filters like Ehlers ITrend. Scalar path is always used.**
use crate::utilities::aligned_vector::AlignedVec;
use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel, alloc_with_nan_prefix, init_matrix_prefixes, make_uninit_matrix};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use rayon::prelude::*;
use std::error::Error;
use std::f64::consts::PI;
use thiserror::Error;

impl<'a> AsRef<[f64]> for EhlersITrendInput<'a> {
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            EhlersITrendData::Candles { candles, source } => source_type(candles, source),
            EhlersITrendData::Slice(slice) => slice,
        }
    }
}

#[derive(Debug, Clone)]
pub enum EhlersITrendData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct EhlersITrendOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct EhlersITrendParams {
    pub warmup_bars: Option<usize>,
    pub max_dc_period: Option<usize>,
}
impl Default for EhlersITrendParams {
    fn default() -> Self {
        Self {
            warmup_bars: Some(12),
            max_dc_period: Some(50),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EhlersITrendInput<'a> {
    pub data: EhlersITrendData<'a>,
    pub params: EhlersITrendParams,
}
impl<'a> EhlersITrendInput<'a> {
    #[inline(always)]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: EhlersITrendParams) -> Self {
        Self {
            data: EhlersITrendData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline(always)]
    pub fn from_slice(sl: &'a [f64], p: EhlersITrendParams) -> Self {
        Self {
            data: EhlersITrendData::Slice(sl),
            params: p,
        }
    }
    #[inline(always)]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", EhlersITrendParams::default())
    }
    #[inline(always)]
    pub fn get_warmup_bars(&self) -> usize {
        self.params.warmup_bars.unwrap_or(12)
    }
    #[inline(always)]
    pub fn get_max_dc_period(&self) -> usize {
        self.params.max_dc_period.unwrap_or(50)
    }
}

#[derive(Debug, Error)]
pub enum EhlersITrendError {
    #[error("ehlers_itrend: Input data is empty.")]
    EmptyInputData,
    #[error("ehlers_itrend: All values are NaN.")]
    AllValuesNaN,
    #[error("ehlers_itrend: Not enough data for warmup. warmup_bars={warmup_bars} but data length={length}")]
    NotEnoughDataForWarmup { warmup_bars: usize, length: usize },
}

#[inline]
pub fn ehlers_itrend(input: &EhlersITrendInput) -> Result<EhlersITrendOutput, EhlersITrendError> {
    ehlers_itrend_with_kernel(input, Kernel::Auto)
}

pub fn ehlers_itrend_with_kernel(
    input: &EhlersITrendInput,
    kernel: Kernel,
) -> Result<EhlersITrendOutput, EhlersITrendError> {
    let data: &[f64] = match &input.data {
        EhlersITrendData::Candles { candles, source } => source_type(candles, source),
        EhlersITrendData::Slice(sl) => sl,
    };

    let len = data.len();
    if len == 0 {
        return Err(EhlersITrendError::EmptyInputData);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(EhlersITrendError::AllValuesNaN)?;
    let warmup_bars = input.get_warmup_bars();
    let max_dc = input.get_max_dc_period().max(1);

    if warmup_bars >= len {
        return Err(EhlersITrendError::NotEnoughDataForWarmup {
            warmup_bars,
            length: len,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let warm = first + warmup_bars;          
    let mut out = alloc_with_nan_prefix(len, warm);

    match chosen {
        Kernel::Scalar | Kernel::ScalarBatch => unsafe {
            ehlers_itrend_scalar(data, warmup_bars, max_dc, first, &mut out)
        },
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2 | Kernel::Avx2Batch => {
            ehlers_itrend_avx2(data, warmup_bars, max_dc, first, &mut out);
        }
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512 | Kernel::Avx512Batch => {
            ehlers_itrend_avx512(data, warmup_bars, max_dc, first, &mut out);
        }
        _ => unreachable!(),
    }

    Ok(EhlersITrendOutput { values: out })
}

#[inline]
pub fn ehlers_itrend_scalar(
    src: &[f64],
    warmup_bars: usize,
    max_dc: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    debug_assert_eq!(src.len(), out.len());
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("fma") {
            unsafe { ehlers_itrend_unsafe_scalar(src, warmup_bars, max_dc, first_valid, out) }
        } else {
            ehlers_itrend_safe_scalar(src, warmup_bars, max_dc, first_valid, out)
        }
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        ehlers_itrend_safe_scalar(src, warmup_bars, max_dc, first_valid, out)
    }
}

#[inline(always)]
fn ehlers_itrend_safe_scalar(
    src: &[f64],
    warmup_bars: usize,
    max_dc: usize,
    first: usize,
    out: &mut [f64],
) {
    let length = src.len();
    let mut fir_buf = [0.0; 7];
    let mut det_buf = [0.0; 7];
    let mut i1_buf = [0.0; 7];
    let mut q1_buf = [0.0; 7];
    let mut prev_i2 = 0.0;
    let mut prev_q2 = 0.0;
    let mut prev_re = 0.0;
    let mut prev_im = 0.0;
    let mut prev_mesa = 0.0;
    let mut prev_smooth = 0.0;
    let mut sum_ring = vec![0.0; max_dc];
    let mut sum_idx = 0_usize;
    let mut prev_it1 = 0.0;
    let mut prev_it2 = 0.0;
    let mut prev_it3 = 0.0;
    let mut ring_ptr = 0_usize;
    for i in 0..length {
        let x0 = src[i];
        let x1 = if i >= 1 { src[i - 1] } else { 0.0 };
        let x2 = if i >= 2 { src[i - 2] } else { 0.0 };
        let x3 = if i >= 3 { src[i - 3] } else { 0.0 };
        let fir_val = (4.0 * x0 + 3.0 * x1 + 2.0 * x2 + x3) / 10.0;
        fir_buf[ring_ptr] = fir_val;

        #[inline(always)]
        fn get_ring(buf: &[f64; 7], center: usize, offset: usize) -> f64 {
            buf[(7 + center - offset) % 7]
        }
        let fir_0 = get_ring(&fir_buf, ring_ptr, 0);
        let fir_2 = get_ring(&fir_buf, ring_ptr, 2);
        let fir_4 = get_ring(&fir_buf, ring_ptr, 4);
        let fir_6 = get_ring(&fir_buf, ring_ptr, 6);

        let h_in = 0.0962 * fir_0 + 0.5769 * fir_2 - 0.5769 * fir_4 - 0.0962 * fir_6;
        let period_mult = 0.075 * prev_mesa + 0.54;
        let det_val = h_in * period_mult;
        det_buf[ring_ptr] = det_val;

        let i1_val = get_ring(&det_buf, ring_ptr, 3);
        i1_buf[ring_ptr] = i1_val;

        let det_0 = get_ring(&det_buf, ring_ptr, 0);
        let det_2 = get_ring(&det_buf, ring_ptr, 2);
        let det_4 = get_ring(&det_buf, ring_ptr, 4);
        let det_6 = get_ring(&det_buf, ring_ptr, 6);
        let h_in_q1 = 0.0962 * det_0 + 0.5769 * det_2 - 0.5769 * det_4 - 0.0962 * det_6;
        let q1_val = h_in_q1 * period_mult;
        q1_buf[ring_ptr] = q1_val;

        let i1_0 = get_ring(&i1_buf, ring_ptr, 0);
        let i1_2 = get_ring(&i1_buf, ring_ptr, 2);
        let i1_4 = get_ring(&i1_buf, ring_ptr, 4);
        let i1_6 = get_ring(&i1_buf, ring_ptr, 6);
        let j_i_val = (0.0962 * i1_0 + 0.5769 * i1_2 - 0.5769 * i1_4 - 0.0962 * i1_6) * period_mult;

        let q1_0 = get_ring(&q1_buf, ring_ptr, 0);
        let q1_2 = get_ring(&q1_buf, ring_ptr, 2);
        let q1_4 = get_ring(&q1_buf, ring_ptr, 4);
        let q1_6 = get_ring(&q1_buf, ring_ptr, 6);
        let j_q_val = (0.0962 * q1_0 + 0.5769 * q1_2 - 0.5769 * q1_4 - 0.0962 * q1_6) * period_mult;

        let mut i2_cur = i1_val - j_q_val;
        let mut q2_cur = q1_val + j_i_val;
        i2_cur = 0.2 * i2_cur + 0.8 * prev_i2;
        q2_cur = 0.2 * q2_cur + 0.8 * prev_q2;
        prev_i2 = i2_cur;
        prev_q2 = q2_cur;

        let re_val = i2_cur * prev_i2 + q2_cur * prev_q2;
        let im_val = i2_cur * prev_q2 - q2_cur * prev_i2;

        let re_smooth = 0.2 * re_val + 0.8 * prev_re;
        let im_smooth = 0.2 * im_val + 0.8 * prev_im;
        prev_re = re_smooth;
        prev_im = im_smooth;

        let mut new_mesa = 0.0;
        if re_smooth != 0.0 && im_smooth != 0.0 {
            new_mesa = 2.0 * PI / (im_smooth / re_smooth).atan();
        }
        let up_lim = 1.5 * prev_mesa;
        if new_mesa > up_lim {
            new_mesa = up_lim;
        }
        let low_lim = 0.67 * prev_mesa;
        if new_mesa < low_lim {
            new_mesa = low_lim;
        }
        new_mesa = new_mesa.clamp(6.0, 50.0);
        let final_mesa = 0.2 * new_mesa + 0.8 * prev_mesa;
        prev_mesa = final_mesa;
        let sp_val = 0.33 * final_mesa + 0.67 * prev_smooth;
        prev_smooth = sp_val;
        let mut dcp = (sp_val + 0.5).floor() as i32;
        if dcp < 1 {
            dcp = 1;
        }
        if dcp as usize > max_dc {
            dcp = max_dc as i32;
        }

        sum_ring[sum_idx] = x0;
        sum_idx = (sum_idx + 1) % max_dc;
        let mut sum_src = 0.0;
        let mut idx2 = sum_idx;
        for _ in 0..dcp {
            idx2 = if idx2 == 0 { max_dc - 1 } else { idx2 - 1 };
            sum_src += sum_ring[idx2];
        }
        let it_val = sum_src / dcp as f64;

        let eit_val = if i < warmup_bars {
            x0
        } else {
            (4.0 * it_val + 3.0 * prev_it1 + 2.0 * prev_it2 + prev_it3) / 10.0
        };
        prev_it3 = prev_it2;
        prev_it2 = prev_it1;
        prev_it1 = it_val;

        out[i] = eit_val;

        ring_ptr = (ring_ptr + 1) % 7;
    }
}

#[inline(always)]
unsafe fn r7(buf: &[f64; 7], p: usize, off: usize) -> f64 {
    let idx = if p >= off { p - off } else { p + 7 - off };
    *buf.get_unchecked(idx)
}

#[inline(always)]
pub unsafe fn ehlers_itrend_unsafe_scalar(
    src: &[f64],
    warmup: usize,
    max_dc: usize,
    _first_valid: usize,
    out: &mut [f64],
) {
    debug_assert_eq!(src.len(), out.len());
    let len = src.len();

    let mut fir = [0.0; 7];
    let mut det = [0.0; 7];
    let mut i1 = [0.0; 7];
    let mut q1 = [0.0; 7];
    let mut sum = [0.0; 64];

    let (mut i2p, mut q2p, mut rep, mut imp) = (0.0, 0.0, 0.0, 0.0);
    let (mut mesa_p, mut sm_p) = (0.0, 0.0);
    let (mut it1p, mut it2p, mut it3p) = (0.0, 0.0, 0.0);

    const C0: f64 = 0.0962;
    const C1: f64 = 0.5769;
    const DIV10: f64 = 0.1;
    const TWO_PI: f64 = core::f64::consts::PI * 2.0;

    let mut rp = 0;
    let mut sp = 0;

    for (idx, &x0) in src.iter().enumerate() {
        let fir_val = (4.0 * x0
            + 3.0 * src.get_unchecked(idx.saturating_sub(1))
            + 2.0 * src.get_unchecked(idx.saturating_sub(2))
            + src.get_unchecked(idx.saturating_sub(3)))
            * DIV10;
        *fir.get_unchecked_mut(rp) = fir_val;

        let hp =
            C0 * (r7(&fir, rp, 0) - r7(&fir, rp, 6)) + C1 * (r7(&fir, rp, 2) - r7(&fir, rp, 4));
        let period_mult = 0.075 * mesa_p + 0.54;
        let det_val = hp * period_mult;
        *det.get_unchecked_mut(rp) = det_val;

        let i1v = r7(&det, rp, 3);
        let q1v = (C0 * (r7(&det, rp, 0) - r7(&det, rp, 6))
            + C1 * (r7(&det, rp, 2) - r7(&det, rp, 4)))
            * period_mult;
        *i1.get_unchecked_mut(rp) = i1v;
        *q1.get_unchecked_mut(rp) = q1v;

        let j_i = (C0 * (r7(&i1, rp, 0) - r7(&i1, rp, 6)) + C1 * (r7(&i1, rp, 2) - r7(&i1, rp, 4)))
            * period_mult;
        let j_q = (C0 * (r7(&q1, rp, 0) - r7(&q1, rp, 6)) + C1 * (r7(&q1, rp, 2) - r7(&q1, rp, 4)))
            * period_mult;

        let mut i2 = 0.2 * (i1v - j_q) + 0.8 * i2p;
        let mut q2 = 0.2 * (q1v + j_i) + 0.8 * q2p;
        i2p = i2;
        q2p = q2;

        let re = 0.2 * (i2 * i2p + q2 * q2p) + 0.8 * rep;
        let im = 0.2 * (i2 * q2p - q2 * i2p) + 0.8 * imp;
        rep = re;
        imp = im;

        let mut mesa = if re != 0.0 && im != 0.0 {
            TWO_PI / im.atan2(re)
        } else {
            0.0
        };
        mesa = mesa.clamp(0.67 * mesa_p, 1.5 * mesa_p).clamp(6.0, 50.0);
        let mesa_f = 0.2 * mesa + 0.8 * mesa_p;
        mesa_p = mesa_f;

        let sp_v = 0.33 * mesa_f + 0.67 * sm_p;
        sm_p = sp_v;

        let dcp = sp_v.round().clamp(1.0, max_dc as f64) as usize;

        sum[sp] = x0;
        sp += 1;
        if sp == max_dc {
            sp = 0;
        }

        let mut acc = 0.0;
        let mut j = sp;
        for _ in 0..dcp {
            j = if j == 0 { max_dc - 1 } else { j - 1 };
            acc += sum[j];
        }
        let it = acc / dcp as f64;

        out[idx] = if idx < warmup {
            x0
        } else {
            (4.0 * it + 3.0 * it1p + 2.0 * it2p + it3p) * DIV10
        };

        it3p = it2p;
        it2p = it1p;
        it1p = it;

        rp += 1;
        if rp == 7 {
            rp = 0;
        }
    }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn ehlers_itrend_avx2(
    data: &[f64],
    warmup_bars: usize,
    max_dc: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    ehlers_itrend_scalar(data, warmup_bars, max_dc, first_valid, out);
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn ehlers_itrend_avx512(
    data: &[f64],
    warmup_bars: usize,
    max_dc: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    ehlers_itrend_scalar(data, warmup_bars, max_dc, first_valid, out);
}

#[derive(Copy, Clone, Debug)]
pub struct EhlersITrendBuilder {
    warmup_bars: Option<usize>,
    max_dc_period: Option<usize>,
    kernel: Kernel,
}
impl Default for EhlersITrendBuilder {
    fn default() -> Self {
        Self {
            warmup_bars: None,
            max_dc_period: None,
            kernel: Kernel::Auto,
        }
    }
}
impl EhlersITrendBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn warmup_bars(mut self, n: usize) -> Self {
        self.warmup_bars = Some(n);
        self
    }
    #[inline(always)]
    pub fn max_dc_period(mut self, n: usize) -> Self {
        self.max_dc_period = Some(n);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<EhlersITrendOutput, EhlersITrendError> {
        let p = EhlersITrendParams {
            warmup_bars: self.warmup_bars,
            max_dc_period: self.max_dc_period,
        };
        let i = EhlersITrendInput::from_candles(c, "close", p);
        ehlers_itrend_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<EhlersITrendOutput, EhlersITrendError> {
        let p = EhlersITrendParams {
            warmup_bars: self.warmup_bars,
            max_dc_period: self.max_dc_period,
        };
        let i = EhlersITrendInput::from_slice(d, p);
        ehlers_itrend_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<EhlersITrendStream, EhlersITrendError> {
        let p = EhlersITrendParams {
            warmup_bars: self.warmup_bars,
            max_dc_period: self.max_dc_period,
        };
        EhlersITrendStream::try_new(p)
    }
}

#[derive(Debug, Clone)]
pub struct EhlersITrendStream {
    warmup_bars: usize,
    max_dc: usize,
    fir_buf: [f64; 7],
    det_buf: [f64; 7],
    i1_buf: [f64; 7],
    q1_buf: [f64; 7],
    prev_i2: f64,
    prev_q2: f64,
    prev_re: f64,
    prev_im: f64,
    prev_mesa: f64,
    prev_smooth: f64,
    sum_ring: Vec<f64>,
    sum_idx: usize,
    prev_it1: f64,
    prev_it2: f64,
    prev_it3: f64,
    ring_ptr: usize,
    bar: usize,
}
impl EhlersITrendStream {
    pub fn try_new(params: EhlersITrendParams) -> Result<Self, EhlersITrendError> {
        let warmup_bars = params.warmup_bars.unwrap_or(12);
        let max_dc = params.max_dc_period.unwrap_or(50).max(1);
        Ok(Self {
            warmup_bars,
            max_dc,
            fir_buf: [0.0; 7],
            det_buf: [0.0; 7],
            i1_buf: [0.0; 7],
            q1_buf: [0.0; 7],
            prev_i2: 0.0,
            prev_q2: 0.0,
            prev_re: 0.0,
            prev_im: 0.0,
            prev_mesa: 0.0,
            prev_smooth: 0.0,
            sum_ring: vec![0.0; max_dc],
            sum_idx: 0,
            prev_it1: 0.0,
            prev_it2: 0.0,
            prev_it3: 0.0,
            ring_ptr: 0,
            bar: 0,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, x0: f64) -> Option<f64> {
        let i = self.bar;
        let x1 = if i >= 1 {
            self.sum_ring[(self.sum_idx + self.max_dc - 1) % self.max_dc]
        } else {
            0.0
        };
        let x2 = if i >= 2 {
            self.sum_ring[(self.sum_idx + self.max_dc - 2) % self.max_dc]
        } else {
            0.0
        };
        let x3 = if i >= 3 {
            self.sum_ring[(self.sum_idx + self.max_dc - 3) % self.max_dc]
        } else {
            0.0
        };
        let fir_val = (4.0 * x0 + 3.0 * x1 + 2.0 * x2 + x3) / 10.0;
        self.fir_buf[self.ring_ptr] = fir_val;

        #[inline(always)]
        fn get_ring(buf: &[f64; 7], center: usize, offset: usize) -> f64 {
            buf[(7 + center - offset) % 7]
        }
        let fir_0 = get_ring(&self.fir_buf, self.ring_ptr, 0);
        let fir_2 = get_ring(&self.fir_buf, self.ring_ptr, 2);
        let fir_4 = get_ring(&self.fir_buf, self.ring_ptr, 4);
        let fir_6 = get_ring(&self.fir_buf, self.ring_ptr, 6);

        let h_in = 0.0962 * fir_0 + 0.5769 * fir_2 - 0.5769 * fir_4 - 0.0962 * fir_6;
        let period_mult = 0.075 * self.prev_mesa + 0.54;
        let det_val = h_in * period_mult;
        self.det_buf[self.ring_ptr] = det_val;

        let i1_val = get_ring(&self.det_buf, self.ring_ptr, 3);
        self.i1_buf[self.ring_ptr] = i1_val;

        let det_0 = get_ring(&self.det_buf, self.ring_ptr, 0);
        let det_2 = get_ring(&self.det_buf, self.ring_ptr, 2);
        let det_4 = get_ring(&self.det_buf, self.ring_ptr, 4);
        let det_6 = get_ring(&self.det_buf, self.ring_ptr, 6);
        let h_in_q1 = 0.0962 * det_0 + 0.5769 * det_2 - 0.5769 * det_4 - 0.0962 * det_6;
        let q1_val = h_in_q1 * period_mult;
        self.q1_buf[self.ring_ptr] = q1_val;

        let i1_0 = get_ring(&self.i1_buf, self.ring_ptr, 0);
        let i1_2 = get_ring(&self.i1_buf, self.ring_ptr, 2);
        let i1_4 = get_ring(&self.i1_buf, self.ring_ptr, 4);
        let i1_6 = get_ring(&self.i1_buf, self.ring_ptr, 6);
        let j_i_val = (0.0962 * i1_0 + 0.5769 * i1_2 - 0.5769 * i1_4 - 0.0962 * i1_6) * period_mult;

        let q1_0 = get_ring(&self.q1_buf, self.ring_ptr, 0);
        let q1_2 = get_ring(&self.q1_buf, self.ring_ptr, 2);
        let q1_4 = get_ring(&self.q1_buf, self.ring_ptr, 4);
        let q1_6 = get_ring(&self.q1_buf, self.ring_ptr, 6);
        let j_q_val = (0.0962 * q1_0 + 0.5769 * q1_2 - 0.5769 * q1_4 - 0.0962 * q1_6) * period_mult;

        let mut i2_cur = i1_val - j_q_val;
        let mut q2_cur = q1_val + j_i_val;
        i2_cur = 0.2 * i2_cur + 0.8 * self.prev_i2;
        q2_cur = 0.2 * q2_cur + 0.8 * self.prev_q2;
        self.prev_i2 = i2_cur;
        self.prev_q2 = q2_cur;

        let re_val = i2_cur * self.prev_i2 + q2_cur * self.prev_q2;
        let im_val = i2_cur * self.prev_q2 - q2_cur * self.prev_i2;

        let re_smooth = 0.2 * re_val + 0.8 * self.prev_re;
        let im_smooth = 0.2 * im_val + 0.8 * self.prev_im;
        self.prev_re = re_smooth;
        self.prev_im = im_smooth;

        let mut new_mesa = 0.0;
        if re_smooth != 0.0 && im_smooth != 0.0 {
            new_mesa = 2.0 * PI / (im_smooth / re_smooth).atan();
        }
        let up_lim = 1.5 * self.prev_mesa;
        if new_mesa > up_lim {
            new_mesa = up_lim;
        }
        let low_lim = 0.67 * self.prev_mesa;
        if new_mesa < low_lim {
            new_mesa = low_lim;
        }
        new_mesa = new_mesa.clamp(6.0, 50.0);
        let final_mesa = 0.2 * new_mesa + 0.8 * self.prev_mesa;
        self.prev_mesa = final_mesa;
        let sp_val = 0.33 * final_mesa + 0.67 * self.prev_smooth;
        self.prev_smooth = sp_val;
        let mut dcp = (sp_val + 0.5).floor() as i32;
        if dcp < 1 {
            dcp = 1;
        }
        if dcp as usize > self.max_dc {
            dcp = self.max_dc as i32;
        }

        self.sum_ring[self.sum_idx] = x0;
        self.sum_idx = (self.sum_idx + 1) % self.max_dc;
        let mut sum_src = 0.0;
        let mut idx2 = self.sum_idx;
        for _ in 0..dcp {
            idx2 = if idx2 == 0 { self.max_dc - 1 } else { idx2 - 1 };
            sum_src += self.sum_ring[idx2];
        }
        let it_val = sum_src / dcp as f64;
        let eit_val = if self.bar < self.warmup_bars {
            x0
        } else {
            (4.0 * it_val + 3.0 * self.prev_it1 + 2.0 * self.prev_it2 + self.prev_it3) / 10.0
        };
        self.prev_it3 = self.prev_it2;
        self.prev_it2 = self.prev_it1;
        self.prev_it1 = it_val;
        self.ring_ptr = (self.ring_ptr + 1) % 7;
        self.bar += 1;
        Some(eit_val)
    }
}

#[derive(Clone, Debug)]
pub struct EhlersITrendBatchRange {
    pub warmup_bars: (usize, usize, usize),
    pub max_dc_period: (usize, usize, usize),
}
impl Default for EhlersITrendBatchRange {
    fn default() -> Self {
        Self {
            warmup_bars: (12, 12, 0),
            max_dc_period: (50, 50, 0),
        }
    }
}
#[derive(Clone, Debug, Default)]
pub struct EhlersITrendBatchBuilder {
    range: EhlersITrendBatchRange,
    kernel: Kernel,
}
impl EhlersITrendBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline]
    pub fn warmup_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.warmup_bars = (start, end, step);
        self
    }
    #[inline]
    pub fn warmup_static(mut self, w: usize) -> Self {
        self.range.warmup_bars = (w, w, 0);
        self
    }
    #[inline]
    pub fn max_dc_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.max_dc_period = (start, end, step);
        self
    }
    #[inline]
    pub fn max_dc_static(mut self, m: usize) -> Self {
        self.range.max_dc_period = (m, m, 0);
        self
    }
    pub fn apply_slice(self, data: &[f64]) -> Result<EhlersITrendBatchOutput, EhlersITrendError> {
        ehlers_itrend_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn apply_candles(
        self,
        c: &Candles,
        src: &str,
    ) -> Result<EhlersITrendBatchOutput, EhlersITrendError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<EhlersITrendBatchOutput, EhlersITrendError> {
        Self::new().kernel(Kernel::Auto).apply_candles(c, "close")
    }
}

pub fn ehlers_itrend_batch_with_kernel(
    data: &[f64],
    sweep: &EhlersITrendBatchRange,
    kernel: Kernel,
) -> Result<EhlersITrendBatchOutput, EhlersITrendError> {
    let kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(EhlersITrendError::EmptyInputData),
    };

    let simd = match kernel {
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512Batch => Kernel::Avx512,
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };


    ehlers_itrend_batch_par_slice(data, sweep, simd)
}

#[inline(always)]
fn ehlers_itrend_batch_inner(
    data: &[f64],
    sweep: &EhlersITrendBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<EhlersITrendBatchOutput, EhlersITrendError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(EhlersITrendError::EmptyInputData);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(EhlersITrendError::AllValuesNaN)?;
    let max_warmup = combos.iter().map(|c| c.warmup_bars.unwrap()).max().unwrap();
    if data.len() - first < max_warmup {
        return Err(EhlersITrendError::NotEnoughDataForWarmup {
            warmup_bars: max_warmup,
            length: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();
        let mut raw = make_uninit_matrix(rows, cols);          // step ❶

        // --- prepare a per-row warm-up prefix ------------------------------------
        let warm: Vec<usize> = combos                      // step ❷
            .iter()
            .map(|c| first + c.warmup_bars.unwrap())
            .collect();

        // SAFETY:  we’re filling only the NaN prefixes row-by-row.
        unsafe { init_matrix_prefixes(&mut raw, cols, &warm) };   // step ❸

        // --- turn the uninit buffer into a Vec<f64> -------------------------------
        let mut values: Vec<f64> = unsafe {                // step ❹
            let ptr = raw.as_mut_ptr() as *mut f64;
            let cap = raw.capacity();
            std::mem::forget(raw);
            Vec::from_raw_parts(ptr, rows * cols, cap)
        };

    let do_row = |row: usize, out_row: &mut [f64]| {
        let p = &combos[row];
        let warmup_bars = p.warmup_bars.unwrap();
        let max_dc = p.max_dc_period.unwrap();
        match kern {
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => ehlers_itrend_row_avx512(data, warmup_bars, max_dc, first, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => ehlers_itrend_row_avx2(data, warmup_bars, max_dc, first, out_row),
            _ => ehlers_itrend_row_scalar(data, warmup_bars, max_dc, first, out_row),
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

    Ok(EhlersITrendBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[derive(Clone, Debug)]
pub struct EhlersITrendBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<EhlersITrendParams>,
    pub rows: usize,
    pub cols: usize,
}
impl EhlersITrendBatchOutput {
    pub fn row_for_params(&self, p: &EhlersITrendParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.warmup_bars.unwrap_or(12) == p.warmup_bars.unwrap_or(12)
                && c.max_dc_period.unwrap_or(50) == p.max_dc_period.unwrap_or(50)
        })
    }
    pub fn values_for(&self, p: &EhlersITrendParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &EhlersITrendBatchRange) -> Vec<EhlersITrendParams> {
    fn axis((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let warmups = axis(r.warmup_bars);
    let max_dcs = axis(r.max_dc_period);
    let mut out = Vec::with_capacity(warmups.len() * max_dcs.len());
    for &w in &warmups {
        for &m in &max_dcs {
            out.push(EhlersITrendParams {
                warmup_bars: Some(w),
                max_dc_period: Some(m),
            });
        }
    }
    out
}

#[inline(always)]
pub fn ehlers_itrend_per_slice(
    data: &[f64],
    sweep: &EhlersITrendBatchRange,
    kern: Kernel,
) -> Result<EhlersITrendBatchOutput, EhlersITrendError> {
    ehlers_itrend_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn ehlers_itrend_batch_par_slice(
    data: &[f64],
    sweep: &EhlersITrendBatchRange,
    kern: Kernel,
) -> Result<EhlersITrendBatchOutput, EhlersITrendError> {
    ehlers_itrend_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
pub fn ehlers_itrend_row_scalar(
    data: &[f64],
    warmup_bars: usize,
    max_dc: usize,
    first: usize,
    out: &mut [f64],
) {
    ehlers_itrend_scalar(data, warmup_bars, max_dc, first, out);
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn ehlers_itrend_row_avx2(
    data: &[f64],
    warmup_bars: usize,
    max_dc: usize,
    first: usize,
    out: &mut [f64],
) {
    ehlers_itrend_scalar(data, warmup_bars, max_dc, first, out);
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn ehlers_itrend_row_avx512(
    data: &[f64],
    warmup_bars: usize,
    max_dc: usize,
    first: usize,
    out: &mut [f64],
) {
    ehlers_itrend_scalar(data, warmup_bars, max_dc, first, out);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_itrend_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = EhlersITrendParams {
            warmup_bars: None,
            max_dc_period: None,
        };
        let input = EhlersITrendInput::from_candles(&candles, "close", default_params);
        let output = ehlers_itrend_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_itrend_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = EhlersITrendInput::with_default_candles(&candles);
        let result = ehlers_itrend_with_kernel(&input, kernel)?;
        let expected_last_five = [59097.88, 59145.9, 59191.96, 59217.26, 59179.68];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-1,
                "[{}] EIT {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_itrend_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = EhlersITrendInput::with_default_candles(&candles);
        match input.data {
            EhlersITrendData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected EhlersITrendData::Candles"),
        }
        let output = ehlers_itrend_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_itrend_no_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [];
        let params = EhlersITrendParams {
            warmup_bars: Some(12),
            max_dc_period: Some(50),
        };
        let input = EhlersITrendInput::from_slice(&input_data, params);
        let res = ehlers_itrend_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] EIT should fail with empty data",
            test_name
        );
        Ok(())
    }

    fn check_itrend_all_nan_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [f64::NAN, f64::NAN, f64::NAN];
        let params = EhlersITrendParams {
            warmup_bars: Some(12),
            max_dc_period: Some(50),
        };
        let input = EhlersITrendInput::from_slice(&data, params);
        let res = ehlers_itrend_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] EIT should fail with all-NaN data",
            test_name
        );
        Ok(())
    }

    fn check_itrend_small_data_for_warmup(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [42.0; 5];
        let params = EhlersITrendParams {
            warmup_bars: Some(12),
            max_dc_period: Some(50),
        };
        let input = EhlersITrendInput::from_slice(&data, params);
        let res = ehlers_itrend_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] EIT should fail if warmup_bars >= data length",
            test_name
        );
        Ok(())
    }

    fn check_itrend_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [42.0];
        let params = EhlersITrendParams {
            warmup_bars: Some(0),
            max_dc_period: Some(50),
        };
        let input = EhlersITrendInput::from_slice(&data, params);
        let result = ehlers_itrend_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), data.len());
        Ok(())
    }

    fn check_itrend_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = EhlersITrendParams {
            warmup_bars: Some(12),
            max_dc_period: Some(50),
        };
        let first_input = EhlersITrendInput::from_candles(&candles, "close", first_params);
        let first_result = ehlers_itrend_with_kernel(&first_input, kernel)?;
        let second_params = EhlersITrendParams {
            warmup_bars: Some(6),
            max_dc_period: Some(25),
        };
        let second_input = EhlersITrendInput::from_slice(&first_result.values, second_params);
        let second_result = ehlers_itrend_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        if second_result.values.len() > 240 {
            for i in 240..second_result.values.len() {
                assert!(
                    !second_result.values[i].is_nan(),
                    "[{}] NaN found at index {} in EIT result",
                    test_name,
                    i
                );
            }
        }
        Ok(())
    }

    fn check_itrend_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = EhlersITrendInput::from_candles(
            &candles,
            "close",
            EhlersITrendParams {
                warmup_bars: Some(12),
                max_dc_period: Some(50),
            },
        );
        let result = ehlers_itrend_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), candles.close.len());
        if result.values.len() > 240 {
            for i in 240..result.values.len() {
                assert!(
                    !result.values[i].is_nan(),
                    "[{}] NaN found at index {} in EIT result",
                    test_name,
                    i
                );
            }
        }
        Ok(())
    }

    fn check_itrend_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let warmup_bars = 12;
        let max_dc = 50;
        let input = EhlersITrendInput::from_candles(
            &candles,
            "close",
            EhlersITrendParams {
                warmup_bars: Some(warmup_bars),
                max_dc_period: Some(max_dc),
            },
        );
        let batch_output = ehlers_itrend_with_kernel(&input, kernel)?.values;
        let mut stream = EhlersITrendStream::try_new(EhlersITrendParams {
            warmup_bars: Some(warmup_bars),
            max_dc_period: Some(max_dc),
        })?;
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
                "[{}] EIT streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    macro_rules! generate_all_itrend_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $(
                    #[test]
                    fn [<$test_fn _scalar_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar);
                    }
                    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                    #[test]
                    fn [<$test_fn _avx2_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2);
                    }
                    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                    #[test]
                    fn [<$test_fn _avx512_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512);
                    }
                )*
            }
        }
    }

    generate_all_itrend_tests!(
        check_itrend_partial_params,
        check_itrend_accuracy,
        check_itrend_default_candles,
        check_itrend_no_data,
        check_itrend_all_nan_data,
        check_itrend_small_data_for_warmup,
        check_itrend_very_small_dataset,
        check_itrend_reinput,
        check_itrend_nan_handling,
        check_itrend_streaming
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = EhlersITrendBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = EhlersITrendParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        let expected = [59097.88, 59145.9, 59191.96, 59217.26, 59179.68];
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
            paste::paste! {
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
    gen_batch_tests!(check_batch_default_row);
}
