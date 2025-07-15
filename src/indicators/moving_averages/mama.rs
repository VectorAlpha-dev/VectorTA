//! # MESA Adaptive Moving Average (MAMA)
//!
//! The MESA Adaptive Moving Average (MAMA) adapts its smoothing factor based on the phase and amplitude
//! of the underlying data, offering low lag and dynamic adaptation. Two series are output: MAMA and FAMA.
//!
//! ## Parameters
//! - **fast_limit**: Upper bound for the adaptive smoothing factor (defaults to 0.5)
//! - **slow_limit**: Lower bound for the adaptive smoothing factor (defaults to 0.05)
//!
//! ## Errors
//! - **NotEnoughData**: mama: Fewer than 10 data points provided.
//! - **InvalidFastLimit**: mama: Invalid fast limit (≤ 0.0, `NaN`, or infinite).
//! - **InvalidSlowLimit**: mama: Invalid slow limit (≤ 0.0, `NaN`, or infinite).
//!
//! ## Returns
//! - **`Ok(MamaOutput)`** on success, containing two `Vec<f64>`: `mama_values` and `fama_values`.
//! - **`Err(MamaError)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_kernel, detect_best_batch_kernel, make_uninit_matrix, alloc_with_nan_prefix, init_matrix_prefixes};
use crate::utilities::math_functions::atan_fast;
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use thiserror::Error;
use std::convert::AsRef;
use std::error::Error;
use std::f64::consts::PI;
use std::mem::MaybeUninit;
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;


#[derive(Debug, Clone)]
pub enum MamaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct MamaOutput {
    pub mama_values: Vec<f64>,
    pub fama_values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct MamaParams {
    pub fast_limit: Option<f64>,
    pub slow_limit: Option<f64>,
}

impl Default for MamaParams {
    fn default() -> Self {
        Self {
            fast_limit: Some(0.5),
            slow_limit: Some(0.05),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MamaInput<'a> {
    pub data: MamaData<'a>,
    pub params: MamaParams,
}

impl<'a> AsRef<[f64]> for MamaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            MamaData::Slice(slice) => slice,
            MamaData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

impl<'a> MamaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: MamaParams) -> Self {
        Self {
            data: MamaData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: MamaParams) -> Self {
        Self {
            data: MamaData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", MamaParams::default())
    }
    #[inline]
    pub fn get_fast_limit(&self) -> f64 {
        self.params.fast_limit.unwrap_or(0.5)
    }
    #[inline]
    pub fn get_slow_limit(&self) -> f64 {
        self.params.slow_limit.unwrap_or(0.05)
    }
}

// Builder struct

#[derive(Copy, Clone, Debug)]
pub struct MamaBuilder {
    fast_limit: Option<f64>,
    slow_limit: Option<f64>,
    kernel: Kernel,
}

impl Default for MamaBuilder {
    fn default() -> Self {
        Self {
            fast_limit: None,
            slow_limit: None,
            kernel: Kernel::Auto,
        }
    }
}

impl MamaBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn fast_limit(mut self, n: f64) -> Self {
        self.fast_limit = Some(n);
        self
    }
    #[inline(always)]
    pub fn slow_limit(mut self, x: f64) -> Self {
        self.slow_limit = Some(x);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<MamaOutput, MamaError> {
        let p = MamaParams {
            fast_limit: self.fast_limit,
            slow_limit: self.slow_limit,
        };
        let i = MamaInput::from_candles(c, "close", p);
        mama_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<MamaOutput, MamaError> {
        let p = MamaParams {
            fast_limit: self.fast_limit,
            slow_limit: self.slow_limit,
        };
        let i = MamaInput::from_slice(d, p);
        mama_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<MamaStream, MamaError> {
        let p = MamaParams {
            fast_limit: self.fast_limit,
            slow_limit: self.slow_limit,
        };
        MamaStream::try_new(p)
    }
}

// Error type

#[derive(Debug, Error)]
pub enum MamaError {
    #[error("mama: Not enough data: needed at least {needed}, found {found}")]
    NotEnoughData { needed: usize, found: usize },
    #[error("mama: Invalid fast limit: {fast_limit}")]
    InvalidFastLimit { fast_limit: f64 },
    #[error("mama: Invalid slow limit: {slow_limit}")]
    InvalidSlowLimit { slow_limit: f64 },
}

// Indicator API

#[inline]
pub fn mama(input: &MamaInput) -> Result<MamaOutput, MamaError> {
    mama_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn mama_prepare<'a>(
    input: &'a MamaInput,
    kernel: Kernel,
) -> Result<
    (
        /*data*/ &'a [f64],
        /*fast_limit*/ f64,
        /*slow_limit*/ f64,
        /*chosen*/ Kernel,
    ),
    MamaError,
> {
    /* ---------- 0. validate ---------------------------------------- */
    let data = input.as_ref();
    let len = data.len();
    if len < 10 {
        return Err(MamaError::NotEnoughData { needed: 10, found: len });
    }

    let fast_limit = input.get_fast_limit();
    let slow_limit = input.get_slow_limit();
    if fast_limit <= 0.0 || fast_limit.is_nan() || fast_limit.is_infinite() {
        return Err(MamaError::InvalidFastLimit { fast_limit });
    }
    if slow_limit <= 0.0 || slow_limit.is_nan() || slow_limit.is_infinite() {
        return Err(MamaError::InvalidSlowLimit { slow_limit });
    }

    /* ---------- kernel auto-detection only once ---------- */
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };

    Ok((data, fast_limit, slow_limit, chosen))
}

pub fn mama_with_kernel(
    input:  &MamaInput,
    kernel: Kernel,
) -> Result<MamaOutput, MamaError> {
    let (data, fast_limit, slow_limit, chosen) = mama_prepare(input, kernel)?;

    /* ---------- allocate outputs -------------------------------- */
    let mut mama_values = vec![0.0; data.len()];
    let mut fama_values = vec![0.0; data.len()];

    /* ---------- run kernel in-place ----------------- */
    unsafe {
        match chosen {
            /* ---- scalar (one-row) ---------------------------------- */
            Kernel::Scalar | Kernel::ScalarBatch => {
                mama_scalar_inplace(
                    data, fast_limit, slow_limit,
                    &mut mama_values, &mut fama_values,
                );
            }

            /* ---- AVX2 --------------------------------------------- */
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                mama_avx2_inplace(
                    data, fast_limit, slow_limit,
                    &mut mama_values, &mut fama_values,
                );
            }

            /* ---- AVX-512 ------------------------------------------ */
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                mama_avx2_inplace(
                    data, fast_limit, slow_limit,
                    &mut mama_values, &mut fama_values,
                );
            }

            _ => unreachable!("unsupported kernel variant"),
        }
    }

    Ok(MamaOutput { mama_values, fama_values })
}

/// Compute MAMA directly into pre-allocated output slices (zero-copy)
pub fn mama_compute_into(
    input: &MamaInput,
    kernel: Kernel,
    out_mama: &mut [f64],
    out_fama: &mut [f64],
) -> Result<(), MamaError> {
    let (data, fast_limit, slow_limit, chosen) = mama_prepare(input, kernel)?;
    
    /* ---------- validate output buffer sizes ----------------------- */
    if out_mama.len() != data.len() || out_fama.len() != data.len() {
        return Err(MamaError::NotEnoughData { needed: data.len(), found: out_mama.len() });
    }

    // MAMA produces values from the first data point
    // No NaN warmup period needed as the algorithm starts immediately

    /* ---------- run kernel in-place -------------------------------- */
    unsafe {
        match chosen {
            /* ---- scalar (one-row) ---------------------------------- */
            Kernel::Scalar | Kernel::ScalarBatch => {
                mama_scalar_inplace(data, fast_limit, slow_limit, out_mama, out_fama);
            }

            /* ---- AVX2 --------------------------------------------- */
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                mama_avx2_inplace(data, fast_limit, slow_limit, out_mama, out_fama);
            }

            /* ---- AVX-512 ------------------------------------------ */
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                mama_avx2_inplace(data, fast_limit, slow_limit, out_mama, out_fama);
            }

            _ => unreachable!("unsupported kernel variant"),
        }
    }

    Ok(())
}

#[inline(always)]
pub fn mama_scalar(data: &[f64], fast_limit: f64, slow_limit: f64, out_mama: &mut [f64], out_fama: &mut [f64]) -> Result<(), MamaError> {
    mama_scalar_inplace(data, fast_limit, slow_limit, out_mama, out_fama);
    Ok(())
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn mama_avx2(data: &[f64], fast_limit: f64, slow_limit: f64, out_mama: &mut [f64], out_fama: &mut [f64]) -> Result<(), MamaError> {
    mama_avx2_inplace(data, fast_limit, slow_limit, out_mama, out_fama);
    Ok(())
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512dq,fma")]
#[inline]
unsafe fn hilbert4_avx512(x0: f64, x2: f64, x4: f64, x6: f64) -> f64 {
    // X = [x6, x4, x2, x0, 0,0,0,0]  (upper 4 lanes zeroed so they don't
    // influence the horizontal reduction)
    let v_x = _mm512_set_pd(0.0, 0.0, 0.0, 0.0, x6, x4, x2, x0);

    // H = [-0.0962, -0.5769, 0.5769, 0.0962, 0,0,0,0]
    const H3: f64 = -0.096_2;
    const H2: f64 = -0.576_9;
    const H1: f64 =  0.576_9;
    const H0: f64 =  0.096_2;
    let v_h = _mm512_set_pd(0.0, 0.0, 0.0, 0.0, H3, H2, H1, H0);

    // 1) component‑wise multiply, 2) horizontal add of all 8 lanes
    let v_mul = _mm512_mul_pd(v_x, v_h);           // 1 µ‑op
    _mm512_reduce_add_pd(v_mul)                    // 1 µ‑op (ICE: latency → 4 cy)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512dq,fma")]
#[inline]
pub unsafe fn mama_avx512_inplace(
    data:       &[f64],
    fast_limit: f64,
    slow_limit: f64,
    out_mama:   &mut [f64],
    out_fama:   &mut [f64],
) {
    debug_assert_eq!(data.len(), out_mama.len());
    debug_assert_eq!(data.len(), out_fama.len());

    /* ---- identical constants & ring‑buffer setup to your AVX2 version ---- */
    const LEN: usize = 8;          // power of two
    const MASK: usize = LEN - 1;

    // align on 64 B so every load hits one line
    #[repr(align(64))] struct A([f64; LEN]);
    let first = data[0];
    let mut smooth    = A([first; LEN]).0;
    let mut detrender = A([first; LEN]).0;
    let mut i1_buf    = A([first; LEN]).0;
    let mut q1_buf    = A([first; LEN]).0;

    const DEG_PER_RAD: f64 = 180.0 / std::f64::consts::PI;

    /* ---- state ---- */
    let (mut idx, mut prev_mesa, mut prev_phase) = (0usize, 0.0, 0.0);
    let (mut prev_mama, mut prev_fama) = (first, first);
    let (mut prev_i2,  mut prev_q2)    = (0.0, 0.0);
    let (mut prev_re,  mut prev_im)    = (0.0, 0.0);

    #[inline(always)]
    fn lag(buf: &[f64; LEN], p: usize, k: usize) -> f64 {
        unsafe { *buf.get_unchecked((p.wrapping_sub(k)) & MASK) }
    }

    /* ---- main loop ---- */
    for (i, &price) in data.iter().enumerate() {
        /* 1. 4‑3‑2‑1 smoother (scalar FMA) */
        let s1 = if i >= 1 { data[i - 1] } else { price };
        let s2 = if i >= 2 { data[i - 2] } else { price };
        let s3 = if i >= 3 { data[i - 3] } else { price };
        let smooth_val = 0.1 * (4.0_f64.mul_add(price, 3.0_f64.mul_add(s1, 2.0_f64.mul_add(s2, s3))));
        smooth[idx] = smooth_val;

        /* 2. Hilbert detrender (AVX‑512) */
        let dt_val = hilbert4_avx512(
            smooth[idx],
            lag(&smooth, idx, 2),
            lag(&smooth, idx, 4),
            lag(&smooth, idx, 6),
        );
        detrender[idx] = dt_val;

        /* 3. In‑phase / quadrature */
        let i1 = lag(&detrender, idx, 3);
        i1_buf[idx] = i1;

        let q1 = hilbert4_avx512(
            detrender[idx],
            lag(&detrender, idx, 2),
            lag(&detrender, idx, 4),
            lag(&detrender, idx, 6),
        );
        q1_buf[idx] = q1;

        /* 4. 90° leads */
        let j_i = hilbert4_avx512(
            i1_buf[idx],
            lag(&i1_buf, idx, 2),
            lag(&i1_buf, idx, 4),
            lag(&i1_buf, idx, 6),
        );
        let j_q = hilbert4_avx512(
            q1_buf[idx],
            lag(&q1_buf, idx, 2),
            lag(&q1_buf, idx, 4),
            lag(&q1_buf, idx, 6),
        );

        /* 5. Homodyne discriminator (unchanged) */
        let i2  = i1 - j_q;
        let q2  = q1 + j_i;
        let i2s = 0.2_f64.mul_add(i2, 0.8 * prev_i2);
        let q2s = 0.2_f64.mul_add(q2, 0.8 * prev_q2);
        prev_i2 = i2s;
        prev_q2 = q2s;

        let re  = 0.2_f64.mul_add(i2s * prev_i2 + q2s * prev_q2, 0.8 * prev_re);
        let im  = 0.2_f64.mul_add(i2s * prev_q2 - q2s * prev_i2, 0.8 * prev_im);
        prev_re = re;
        prev_im = im;

        /* 6. Dominant cycle period */
        let mut mesa = if re != 0.0 && im != 0.0 {
            2.0 * std::f64::consts::PI / atan_fast(im / re)
        } else { prev_mesa };
        mesa = mesa.min(1.5 * prev_mesa).max(0.67 * prev_mesa).max(6.0).min(50.0);
        mesa = 0.2_f64.mul_add(mesa, 0.8 * prev_mesa);
        prev_mesa = mesa;

        /* 7. Adaptive alpha */
        let phase = if i1 != 0.0 { atan_fast(q1 / i1) * DEG_PER_RAD } else { prev_phase };
        let mut dp = prev_phase - phase;
        if dp < 1.0 { dp = 1.0; }
        prev_phase = phase;

        let mut alpha = fast_limit / dp;
        alpha = alpha.clamp(slow_limit, fast_limit);

        /* 8. MAMA / FAMA */
        let cur_mama = alpha.mul_add(price, (1.0 - alpha) * prev_mama);
        let cur_fama = (0.5 * alpha).mul_add(cur_mama, (1.0 - 0.5 * alpha) * prev_fama);
        prev_mama = cur_mama;
        prev_fama = cur_fama;

        out_mama[i] = cur_mama;
        out_fama[i] = cur_fama;

        idx = (idx + 1) & MASK;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn hilbert4_avx2(x0: f64, x2: f64, x4: f64, x6: f64) -> f64 {
    // X = [x6,x4,x2,x0]  – order chosen so that vhadd folds without shuffle
    let v_x = _mm256_set_pd(x6, x4, x2, x0);
    // H = [‑.0962,‑.5769,.5769,.0962]
    const H3: f64 = -0.096_2; const H2: f64 = -0.576_9;
    const H1: f64 =  0.576_9; const H0: f64 =  0.096_2;
    let v_h = _mm256_set_pd(H3, H2, H1, H0);

    // multiply & first stage add
    let v_mul = _mm256_mul_pd(v_x, v_h);          // 1 µ‑op
    let v_sum = _mm256_hadd_pd(v_mul, v_mul);     // horizontal add
    // fold lanes 0‑1 with 2‑3
    let v_fold = _mm256_permute2f128_pd(v_sum, v_sum, 0x1);
    let v_res  = _mm256_add_pd(v_sum, v_fold);    // 1 µ‑op
    _mm256_cvtsd_f64(v_res)                       // lane 0 → scalar
}


#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn mama_avx2_inplace(
    data:       &[f64],
    fast_limit: f64,
    slow_limit: f64,
    out_mama:   &mut [f64],
    out_fama:   &mut [f64],
) {
    debug_assert_eq!(data.len(), out_mama.len());
    debug_assert_eq!(data.len(), out_fama.len());

    /* ----------------------------------------------------------
     * 1. Constants – weights & helpers
     * -------------------------------------------------------- */
    const RING_LEN : usize = 8;               // power‑of‑two → & (LEN‑1) instead of %
    const MASK     : usize = RING_LEN - 1;

    // 4‑3‑2‑1 smoother coefficients (sum = 10).
    const W0: f64 = 4.0;
    const W1: f64 = 3.0;
    const W2: f64 = 2.0;
    const W3: f64 = 1.0;

    // Hilbert transform taps (John Ehlers)
    const H0: f64 =  0.096_2;
    const H1: f64 =  0.576_9;
    const H2: f64 = -0.576_9;
    const H3: f64 = -0.096_2;

    const DEG_PER_RAD: f64 = 180.0 / std::f64::consts::PI;

    /* ----------------------------------------------------------
     * 3. Local ring buffers (filled with first sample)
     * -------------------------------------------------------- */
    let first = data[0];
    let mut smooth    = [first; RING_LEN];
    let mut detrender = [first; RING_LEN];
    let mut i1_buf    = [first; RING_LEN];
    let mut q1_buf    = [first; RING_LEN];

    /* ----------------------------------------------------------
     * 4. Rolling state variables
     * -------------------------------------------------------- */
    let mut idx         = 0usize;
    let mut prev_mesa   = 0.0;
    let mut prev_phase  = 0.0;
    let mut prev_mama   = first;
    let mut prev_fama   = first;
    let mut prev_i2     = 0.0;
    let mut prev_q2     = 0.0;
    let mut prev_re     = 0.0;
    let mut prev_im     = 0.0;

    // lag helper – branch‑free via masking
    #[inline(always)]
    fn lag(buf: &[f64; RING_LEN], p: usize, k: usize) -> f64 {
        buf[(p.wrapping_sub(k)) & MASK]
    }

    /* ----------------------------------------------------------
     * 5. Main processing loop
     * -------------------------------------------------------- */
    for (i, &price) in data.iter().enumerate() {
        /* ---------- 5.1 4‑3‑2‑1 smoother ------------------- */
        let s1  = if i >= 1 { data[i - 1] } else { price };
        let s2  = if i >= 2 { data[i - 2] } else { price };
        let s3  = if i >= 3 { data[i - 3] } else { price };

        // fused  (4p + 3s1 + 2s2 + s3) / 10
        let smooth_val =
            W0.mul_add(price,
            W1.mul_add(s1,
            W2.mul_add(s2, s3))) * 0.1;
        smooth[idx] = smooth_val;

        /* ---------- 5.2 Hilbert detrender ------------------ */
        let dt_val = hilbert4_avx2(
            smooth[idx], lag(&smooth, idx, 2),
            lag(&smooth, idx, 4), lag(&smooth, idx, 6)
        );

        /* ---------- 5.3 In‑phase & quadrature -------------- */
        let i1 = lag(&detrender, idx, 3);        // 3‑bar lag
        i1_buf[idx] = i1;

        let q1 =
            H0.mul_add(detrender[idx],
            H1.mul_add(lag(&detrender, idx, 2),
            H2.mul_add(lag(&detrender, idx, 4),
                       H3 * lag(&detrender, idx, 6))));
        q1_buf[idx] = q1;

        /* ---------- 5.4 90° leads -------------------------- */
        let j_i =
            H0.mul_add(i1_buf[idx],
            H1.mul_add(lag(&i1_buf, idx, 2),
            H2.mul_add(lag(&i1_buf, idx, 4),
                       H3 * lag(&i1_buf, idx, 6))));
        let j_q =
            H0.mul_add(q1_buf[idx],
            H1.mul_add(lag(&q1_buf, idx, 2),
            H2.mul_add(lag(&q1_buf, idx, 4),
                       H3 * lag(&q1_buf, idx, 6))));

        /* ---------- 5.5 Homodyne discriminator ------------- */
        let i2  = i1 - j_q;
        let q2  = q1 + j_i;
        let i2s = 0.2_f64.mul_add(i2, 0.8 * prev_i2);
        let q2s = 0.2_f64.mul_add(q2, 0.8 * prev_q2);
        prev_i2 = i2s;
        prev_q2 = q2s;

        let re  = 0.2_f64.mul_add(i2s * prev_i2 + q2s * prev_q2, 0.8 * prev_re);
        let im  = 0.2_f64.mul_add(i2s * prev_q2 - q2s * prev_i2, 0.8 * prev_im);
        prev_re = re;
        prev_im = im;

        /* ---------- 5.6 Dominant cycle period -------------- */
        let mut mesa = if re != 0.0 && im != 0.0 {
            2.0 * std::f64::consts::PI
                / atan_fast(im / re)
        } else {
            prev_mesa
        };

        mesa = mesa
            .min(1.5 * prev_mesa)
            .max(0.67 * prev_mesa)
            .max(6.0)
            .min(50.0);
        mesa = 0.2_f64.mul_add(mesa, 0.8 * prev_mesa);
        prev_mesa = mesa;

        /* ---------- 5.7 Adaptive alpha --------------------- */
        let phase = if i1 != 0.0 {
            atan_fast(q1 / i1)
                * DEG_PER_RAD
        } else {
            prev_phase
        };
        let mut dp = prev_phase - phase;
        if dp < 1.0 { dp = 1.0; }
        prev_phase = phase;

        let mut alpha = fast_limit / dp;
        alpha = alpha.clamp(slow_limit, fast_limit);

        /* ---------- 5.8 MAMA & FAMA ------------------------ */
        let cur_mama = alpha.mul_add(price, (1.0 - alpha) * prev_mama);
        let cur_fama = (0.5 * alpha).mul_add(cur_mama, (1.0 - 0.5 * alpha) * prev_fama);
        prev_mama = cur_mama;
        prev_fama = cur_fama;

        out_mama[i] = cur_mama;
        out_fama[i] = cur_fama;

        /* ---------- 5.9 Advance ring index ----------------- */
        idx = (idx + 1) & MASK;     // branch‑free, cheaper than %
    }
}
#[inline(always)]
fn hilbert(x0: f64, x2: f64, x4: f64, x6: f64) -> f64 {
    0.0962 * x0 + 0.5769 * x2 - 0.5769 * x4 - 0.0962 * x6
}

#[inline]
pub fn mama_scalar_inplace(
    data:      &[f64],
    fast_limit: f64,
    slow_limit: f64,
    out_mama:  &mut [f64],
    out_fama:  &mut [f64],
) {
    debug_assert_eq!(data.len(), out_mama.len());
    debug_assert_eq!(data.len(), out_fama.len());
    let len   = data.len();
    let warm  = 10;

    /* ---- ring buffers & rolling state ---- */
    let mut smooth_buf    = [data[0]; 7];
    let mut detrender_buf = [data[0]; 7];
    let mut i1_buf        = [data[0]; 7];
    let mut q1_buf        = [data[0]; 7];

    let mut prev_mesa_period = 0.0;
    let mut prev_mama        = data[0];
    let mut prev_fama        = data[0];
    let mut prev_i2_sm       = 0.0;
    let mut prev_q2_sm       = 0.0;
    let mut prev_re          = 0.0;
    let mut prev_im          = 0.0;
    let mut prev_phase       = 0.0;

    #[inline(always)]
    fn hilbert(x0: f64, x2: f64, x4: f64, x6: f64) -> f64 {
        0.0962 * x0 + 0.5769 * x2 - 0.5769 * x4 - 0.0962 * x6
    }

    for i in 0..len {
        let price = data[i];

        /* --- 4-3-2-1 smoother ------------------------------------------ */
        let s1 = if i >= 1 { data[i - 1] } else { price };
        let s2 = if i >= 2 { data[i - 2] } else { price };
        let s3 = if i >= 3 { data[i - 3] } else { price };
        let smooth_val = (4.0 * price + 3.0 * s1 + 2.0 * s2 + s3) / 10.0;

        let idx = i % 7;
        smooth_buf[idx] = smooth_val;

        /* --- Hilbert transform (detrender) ----------------------------- */
        let x0 = smooth_buf[idx];
        let x2 = smooth_buf[(idx + 5) % 7];
        let x4 = smooth_buf[(idx + 3) % 7];
        let x6 = smooth_buf[(idx + 1) % 7];

        // empirical Mesa multiplier
        let mesa_mult = 0.075 * prev_mesa_period + 0.54;
        let dt_val    = hilbert(x0, x2, x4, x6) * mesa_mult;
        detrender_buf[idx] = dt_val;

        /* --- in-phase & quadrature ------------------------------------ */
        let i1_val = if i >= 3 {
            detrender_buf[(idx + 4) % 7]   // lag 3
        
            } else {
            dt_val
        };
        i1_buf[idx] = i1_val;

        let d0 = detrender_buf[idx];
        let d2 = detrender_buf[(idx + 5) % 7];
        let d4 = detrender_buf[(idx + 3) % 7];
        let d6 = detrender_buf[(idx + 1) % 7];
        let q1_val = hilbert(d0, d2, d4, d6) * mesa_mult;
        q1_buf[idx] = q1_val;

        /* --- 90° leads (J components) ---------------------------------- */
        let j_i = {
            let i0 = i1_buf[idx];
            let i2 = i1_buf[(idx + 5) % 7];
            let i4 = i1_buf[(idx + 3) % 7];
            let i6 = i1_buf[(idx + 1) % 7];
            hilbert(i0, i2, i4, i6) * mesa_mult
        };
        let j_q = {
            let q0 = q1_buf[idx];
            let q2 = q1_buf[(idx + 5) % 7];
            let q4 = q1_buf[(idx + 3) % 7];
            let q6 = q1_buf[(idx + 1) % 7];
            hilbert(q0, q2, q4, q6) * mesa_mult
        };

        /* --- homodyne discriminator ----------------------------------- */
        let i2      = i1_val - j_q;
        let q2      = q1_val + j_i;
        let i2_sm   = 0.2 * i2 + 0.8 * prev_i2_sm;
        let q2_sm   = 0.2 * q2 + 0.8 * prev_q2_sm;
        let re      = 0.2 * (i2_sm * prev_i2_sm + q2_sm * prev_q2_sm) + 0.8 * prev_re;
        let im      = 0.2 * (i2_sm * prev_q2_sm - q2_sm * prev_i2_sm) + 0.8 * prev_im;
        prev_i2_sm  = i2_sm;
        prev_q2_sm  = q2_sm;
        prev_re     = re;
        prev_im     = im;

        /* --- dominant cycle period ------------------------------------ */
        let mut mesa_period = if re != 0.0 && im != 0.0 {
            2.0 * std::f64::consts::PI / (im / re).atan()
        
            } else {
            prev_mesa_period
        };

        // apply the traditional Mesa constraints
        mesa_period = mesa_period
            .min(1.5 * prev_mesa_period)
            .max(0.67 * prev_mesa_period)
            .max(6.0)
            .min(50.0);
        mesa_period = 0.2 * mesa_period + 0.8 * prev_mesa_period;
        prev_mesa_period = mesa_period;

        /* --- adaptive alpha ------------------------------------------- */
        let phase     = if i1_val != 0.0 { (q1_val / i1_val).atan() * 180.0 / std::f64::consts::PI }
                        else { prev_phase };
        let mut dp    = prev_phase - phase;
        if dp < 1.0 { dp = 1.0; }
        prev_phase    = phase;

        let mut alpha = fast_limit / dp;
        alpha = alpha.clamp(slow_limit, fast_limit);

        /* --- MAMA & FAMA ---------------------------------------------- */
        let cur_mama = alpha * price + (1.0 - alpha) * prev_mama;
        let cur_fama = 0.5 * alpha * cur_mama + (1.0 - 0.5 * alpha) * prev_fama;
        prev_mama    = cur_mama;
        prev_fama    = cur_fama;

        /* --- store ----------------------------------------------------- */
        out_mama[i] = cur_mama;
        out_fama[i] = cur_fama;
    }
}

// Stream (online) MAMA

/* ---------------------------------------------------------------
 *  Streaming (online) MAMA – in-place version
 * ------------------------------------------------------------- */

#[derive(Debug, Clone)]
pub struct MamaStream {
    fast_limit: f64,
    slow_limit: f64,

    /* last 10 prices in a ring-buffer */
    buffer: [f64; 10],
    pos:    usize,
    filled: bool,

    /* workspaces that receive the kernel’s output each tick */
    mama_out: [f64; 10],
    fama_out: [f64; 10],
}

impl MamaStream {
    /* ---------- constructor ----------------------------------- */
    pub fn try_new(params: MamaParams) -> Result<Self, MamaError> {
        let fast_limit = params.fast_limit.unwrap_or(0.5);
        let slow_limit = params.slow_limit.unwrap_or(0.05);

        if fast_limit <= 0.0 || fast_limit.is_nan() || fast_limit.is_infinite() {
            return Err(MamaError::InvalidFastLimit { fast_limit });
        }
        if slow_limit <= 0.0 || slow_limit.is_nan() || slow_limit.is_infinite() {
            return Err(MamaError::InvalidSlowLimit { slow_limit });
        }

        Ok(Self {
            fast_limit,
            slow_limit,
            buffer:   [f64::NAN; 10],
            pos:      0,
            filled:   false,
            mama_out: [f64::NAN; 10],   // already NaN-prefilled
            fama_out: [f64::NAN; 10],
        })
    }

    /* ---------- push one new price ---------------------------- */
    pub fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        /* 1. write into the 10-slot ring */
        self.buffer[self.pos] = value;
        self.pos = (self.pos + 1) % 10;

        /* 2. mark ‘filled’ once we have 10 distinct samples */
        if !self.filled && self.pos == 0 {
            self.filled = true;
        }
        if !self.filled {
            return None;                   // still warming up
        }

        /* 3. build a contiguous slice view of the ring ---------- */
        let slice_storage;                      // lives only in this branch
        let slice: &[f64] = if self.pos == 0 {
            &self.buffer[..]                   // already contiguous
        
            } else {
            // copy into a local array to linearise [pos..] ∪ [..pos]
            slice_storage = {
                let mut tmp = [0.0_f64; 10];   // stack-allocated
                let (head, tail) = self.buffer.split_at(self.pos);
                tmp[..10 - self.pos].copy_from_slice(tail);
                tmp[10 - self.pos..].copy_from_slice(head);
                tmp
            };
            &slice_storage[..]
        };

        /* 4. run the in-place scalar kernel --------------------- */
        unsafe {
            mama_scalar_inplace(
                slice,
                self.fast_limit,
                self.slow_limit,
                &mut self.mama_out,
                &mut self.fama_out,
            );
        }

        /* 5. pick the most recent value (index 9) --------------- */
        let last = 9;                          // slice.len() – 1
        Some((self.mama_out[last], self.fama_out[last]))
    }
}


// Batch types, grid expansion

#[derive(Clone, Debug)]
pub struct MamaBatchRange {
    pub fast_limit: (f64, f64, f64),
    pub slow_limit: (f64, f64, f64),
}

impl Default for MamaBatchRange {
    fn default() -> Self {
        Self {
            fast_limit: (0.5, 0.5, 0.0),
            slow_limit: (0.05, 0.05, 0.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct MamaBatchBuilder {
    range: MamaBatchRange,
    kernel: Kernel,
}

impl MamaBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline]
    pub fn fast_limit_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.fast_limit = (start, end, step);
        self
    }
    #[inline]
    pub fn fast_limit_static(mut self, x: f64) -> Self {
        self.range.fast_limit = (x, x, 0.0);
        self
    }
    #[inline]
    pub fn slow_limit_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.slow_limit = (start, end, step);
        self
    }
    #[inline]
    pub fn slow_limit_static(mut self, x: f64) -> Self {
        self.range.slow_limit = (x, x, 0.0);
        self
    }
    pub fn apply_slice(self, data: &[f64]) -> Result<MamaBatchOutput, MamaError> {
        mama_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<MamaBatchOutput, MamaError> {
        MamaBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<MamaBatchOutput, MamaError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<MamaBatchOutput, MamaError> {
        MamaBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct MamaBatchOutput {
    pub mama_values: Vec<f64>,
    pub fama_values: Vec<f64>,
    pub combos: Vec<MamaParams>,
    pub rows: usize,
    pub cols: usize,
}

impl MamaBatchOutput {
    pub fn row_for_params(&self, p: &MamaParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            (c.fast_limit.unwrap_or(0.5) - p.fast_limit.unwrap_or(0.5)).abs() < 1e-12
                && (c.slow_limit.unwrap_or(0.05) - p.slow_limit.unwrap_or(0.05)).abs() < 1e-12
        })
    }
    pub fn mama_for(&self, p: &MamaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.mama_values[start..start + self.cols]
        })
    }
    pub fn fama_for(&self, p: &MamaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.fama_values[start..start + self.cols]
        })
    }
}

#[inline(always)]
pub fn expand_grid(r: &MamaBatchRange) -> Vec<MamaParams> {
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
    let fast_limits = axis_f64(r.fast_limit);
    let slow_limits = axis_f64(r.slow_limit);
    let mut out = Vec::with_capacity(fast_limits.len() * slow_limits.len());
    for &f in &fast_limits {
        for &s in &slow_limits {
            out.push(MamaParams {
                fast_limit: Some(f),
                slow_limit: Some(s),
            });
        }
    }
    out
}

pub fn mama_batch_with_kernel(
    data: &[f64],
    sweep: &MamaBatchRange,
    k: Kernel,
) -> Result<MamaBatchOutput, MamaError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(MamaError::NotEnoughData { needed: 10, found: 0 }),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    mama_batch_par_slice(data, sweep, simd)
}

#[inline(always)]
pub fn mama_batch_slice(
    data: &[f64],
    sweep: &MamaBatchRange,
    kern: Kernel,
) -> Result<MamaBatchOutput, MamaError> {
    mama_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn mama_batch_par_slice(
    data: &[f64],
    sweep: &MamaBatchRange,
    kern: Kernel,
) -> Result<MamaBatchOutput, MamaError> {
    mama_batch_inner(data, sweep, kern, true)
}

fn mama_batch_inner(
    data: &[f64],
    sweep: &MamaBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<MamaBatchOutput, MamaError> {
    // ---------- 0. prelim checks ----------
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(MamaError::NotEnoughData { needed: 10, found: 0 });
    }
    if data.len() < 10 {
        return Err(MamaError::NotEnoughData {
            needed: 10,
            found: data.len(),
        });
    }
    
    // Validate all parameter combinations
    for combo in &combos {
        let fast_limit = combo.fast_limit.unwrap_or(0.5);
        let slow_limit = combo.slow_limit.unwrap_or(0.05);
        
        if fast_limit <= 0.0 || fast_limit.is_nan() || fast_limit.is_infinite() {
            return Err(MamaError::InvalidFastLimit { fast_limit });
        }
        if slow_limit <= 0.0 || slow_limit.is_nan() || slow_limit.is_infinite() {
            return Err(MamaError::InvalidSlowLimit { slow_limit });
        }
    }

    // ---------- 1. matrix allocation ----------
    let rows = combos.len();
    let cols = data.len();

    // uninitialised backing buffers
    let mut raw_mama = make_uninit_matrix(rows, cols);
    let mut raw_fama = make_uninit_matrix(rows, cols);

    // write quiet-NaN prefixes so the first 10 values line up with streaming MAMA
    let warm_prefixes = vec![10; rows];
    unsafe {
        init_matrix_prefixes(&mut raw_mama, cols, &warm_prefixes);
        init_matrix_prefixes(&mut raw_fama, cols, &warm_prefixes);
    }

    // ---------- 2. per-row worker ----------
    let do_row = |row: usize,
                  dst_m: &mut [MaybeUninit<f64>],
                  dst_f: &mut [MaybeUninit<f64>]| unsafe {
        let prm  = &combos[row];
        let fast = prm.fast_limit.unwrap_or(0.5);
        let slow = prm.slow_limit.unwrap_or(0.05);

        // cast each row to `&mut [f64]` once and let the kernel write directly
        let out_m = core::slice::from_raw_parts_mut(
            dst_m.as_mut_ptr() as *mut f64,
            dst_m.len(),
        );
        let out_f = core::slice::from_raw_parts_mut(
            dst_f.as_mut_ptr() as *mut f64,
            dst_f.len(),
        );

        match kern {
            Kernel::Scalar => mama_row_scalar (data, fast, slow, out_m, out_f),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2   => mama_row_avx2   (data, fast, slow, out_m, out_f),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => mama_row_avx512 (data, fast, slow, out_m, out_f),
            _ => unreachable!(),
        }
    };

    // ---------- 3. run over every row ----------
    if parallel {

        #[cfg(not(target_arch = "wasm32"))] {

        raw_mama.par_chunks_mut(cols)

                        .zip(raw_fama.par_chunks_mut(cols))

                        .enumerate()

                        .for_each(|(row, (m_row, f_row))| do_row(row, m_row, f_row));

        }

        #[cfg(target_arch = "wasm32")] {

        for (row, (m_row, f_row)) in raw_mama.chunks_mut(cols)

                                                     .zip(raw_fama.chunks_mut(cols))

                                                     .enumerate()

                {

                    do_row(row, m_row, f_row);

        }
        }
    } else {
        for (row, (m_row, f_row)) in raw_mama.chunks_mut(cols)
                                             .zip(raw_fama.chunks_mut(cols))
                                             .enumerate()
        {
            do_row(row, m_row, f_row);
        }
    }

    // ---------- 4. transmute to Vec<f64> ----------
    let mama_values: Vec<f64> = unsafe { std::mem::transmute(raw_mama) };
    let fama_values: Vec<f64> = unsafe { std::mem::transmute(raw_fama) };

    // ---------- 5. package result ----------
    Ok(MamaBatchOutput {
        mama_values,
        fama_values,
        combos,
        rows,
        cols,
    })
}

/// Batch compute MAMA directly into pre-allocated output slices (zero-copy)
pub fn mama_batch_inner_into(
    data: &[f64],
    sweep: &MamaBatchRange,
    kern: Kernel,
    parallel: bool,
    out_mama: &mut [f64],
    out_fama: &mut [f64],
) -> Result<Vec<MamaParams>, MamaError> {
    // ---------- 0. prelim checks ----------
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(MamaError::NotEnoughData { needed: 10, found: 0 });
    }
    if data.len() < 10 {
        return Err(MamaError::NotEnoughData {
            needed: 10,
            found: data.len(),
        });
    }
    
    // Validate all parameter combinations
    for combo in &combos {
        let fast_limit = combo.fast_limit.unwrap_or(0.5);
        let slow_limit = combo.slow_limit.unwrap_or(0.05);
        
        if fast_limit <= 0.0 || fast_limit.is_nan() || fast_limit.is_infinite() {
            return Err(MamaError::InvalidFastLimit { fast_limit });
        }
        if slow_limit <= 0.0 || slow_limit.is_nan() || slow_limit.is_infinite() {
            return Err(MamaError::InvalidSlowLimit { slow_limit });
        }
    }

    let rows = combos.len();
    let cols = data.len();
    
    // Validate output slice sizes
    if out_mama.len() != rows * cols || out_fama.len() != rows * cols {
        return Err(MamaError::NotEnoughData { 
            needed: rows * cols, 
            found: out_mama.len().min(out_fama.len()) 
        });
    }

    // ---------- 1. cast output slices to MaybeUninit for init_matrix_prefixes ----------
    let out_mama_uninit = unsafe {
        std::slice::from_raw_parts_mut(
            out_mama.as_mut_ptr() as *mut MaybeUninit<f64>,
            out_mama.len()
        )
    };
    let out_fama_uninit = unsafe {
        std::slice::from_raw_parts_mut(
            out_fama.as_mut_ptr() as *mut MaybeUninit<f64>,
            out_fama.len()
        )
    };

    // write quiet-NaN prefixes so the first 10 values line up with streaming MAMA
    let warm_prefixes = vec![10; rows];
    unsafe {
        init_matrix_prefixes(out_mama_uninit, cols, &warm_prefixes);
        init_matrix_prefixes(out_fama_uninit, cols, &warm_prefixes);
    }

    // ---------- 2. per-row worker ----------
    let do_row = |row: usize,
                  dst_m: &mut [MaybeUninit<f64>],
                  dst_f: &mut [MaybeUninit<f64>]| unsafe {
        let prm  = &combos[row];
        let fast = prm.fast_limit.unwrap_or(0.5);
        let slow = prm.slow_limit.unwrap_or(0.05);

        // cast each row to `&mut [f64]` once and let the kernel write directly
        let out_m = core::slice::from_raw_parts_mut(
            dst_m.as_mut_ptr() as *mut f64,
            dst_m.len(),
        );
        let out_f = core::slice::from_raw_parts_mut(
            dst_f.as_mut_ptr() as *mut f64,
            dst_f.len(),
        );

        match kern {
            Kernel::Scalar => mama_row_scalar (data, fast, slow, out_m, out_f),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2   => mama_row_avx2   (data, fast, slow, out_m, out_f),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => mama_row_avx512 (data, fast, slow, out_m, out_f),
            _ => unreachable!(),
        }
    };

    // ---------- 3. run over every row ----------
    if parallel {
        #[cfg(not(target_arch = "wasm32"))] {
            out_mama_uninit.par_chunks_mut(cols)
                .zip(out_fama_uninit.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(row, (m_row, f_row))| do_row(row, m_row, f_row));
        }

        #[cfg(target_arch = "wasm32")] {
            for (row, (m_row, f_row)) in out_mama_uninit.chunks_mut(cols)
                .zip(out_fama_uninit.chunks_mut(cols))
                .enumerate()
            {
                do_row(row, m_row, f_row);
            }
        }
    } else {
        for (row, (m_row, f_row)) in out_mama_uninit.chunks_mut(cols)
            .zip(out_fama_uninit.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, m_row, f_row);
        }
    }

    Ok(combos)
}

// Row API (for batch)

#[inline(always)]
pub unsafe fn mama_row_scalar(
    data: &[f64],
    fast_limit: f64,
    slow_limit: f64,
    out_mama: &mut [f64],
    out_fama: &mut [f64],
) {
    mama_scalar_inplace(data, fast_limit, slow_limit, out_mama, out_fama);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn mama_row_avx2(
    data: &[f64],
    fast_limit: f64,
    slow_limit: f64,
    out_mama: &mut [f64],
    out_fama: &mut [f64],
) {
    mama_avx2_inplace(data, fast_limit, slow_limit, out_mama, out_fama);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn mama_row_avx512(
    data: &[f64],
    fast_limit: f64,
    slow_limit: f64,
    out_mama: &mut [f64],
    out_fama: &mut [f64],
) {
    mama_row_avx2(data, fast_limit, slow_limit, out_mama, out_fama);
}

// Tests (see ALMA-style harness for parity; copy/adapt as needed)

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;
    use paste::paste;

    fn check_mama_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = MamaParams { fast_limit: None, slow_limit: None };
        let input = MamaInput::from_candles(&candles, "close", default_params);
        let output = mama_with_kernel(&input, kernel)?;
        assert_eq!(output.mama_values.len(), candles.close.len());
        assert_eq!(output.fama_values.len(), candles.close.len());
        Ok(())
    }

    fn check_mama_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = MamaInput::from_candles(&candles, "close", MamaParams::default());
        let result = mama_with_kernel(&input, kernel)?;
        assert_eq!(result.mama_values.len(), candles.close.len());
        assert_eq!(result.fama_values.len(), candles.close.len());
        Ok(())
    }

    fn check_mama_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = MamaInput::with_default_candles(&candles);
        match input.data {
            MamaData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected MamaData::Candles"),
        }
        let output = mama_with_kernel(&input, kernel)?;
        assert_eq!(output.mama_values.len(), candles.close.len());
        assert_eq!(output.fama_values.len(), candles.close.len());
        Ok(())
    }

    fn check_mama_with_insufficient_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [100.0; 9];
        let params = MamaParams::default();
        let input = MamaInput::from_slice(&input_data, params);
        let res = mama_with_kernel(&input, kernel);
        assert!(res.is_err());
        Ok(())
    }

    fn check_mama_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [42.0; 10];
        let params = MamaParams::default();
        let input = MamaInput::from_slice(&input_data, params);
        let result = mama_with_kernel(&input, kernel)?;
        assert_eq!(result.mama_values.len(), input_data.len());
        assert_eq!(result.fama_values.len(), input_data.len());
        Ok(())
    }

    fn check_mama_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = MamaParams::default();
        let first_input = MamaInput::from_candles(&candles, "close", first_params);
        let first_result = mama_with_kernel(&first_input, kernel)?;
        let second_params = MamaParams { fast_limit: Some(0.7), slow_limit: Some(0.1) };
        let second_input = MamaInput::from_slice(&first_result.mama_values, second_params);
        let second_result = mama_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.mama_values.len(), first_result.mama_values.len());
        assert_eq!(second_result.fama_values.len(), first_result.mama_values.len());
        Ok(())
    }

    fn check_mama_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = MamaParams::default();
        let input = MamaInput::from_candles(&candles, "close", params);
        let result = mama_with_kernel(&input, kernel)?;
        for (i, &val) in result.mama_values.iter().enumerate() {
            if i > 20 {
                assert!(val.is_finite());
            }
        }
        for (i, &val) in result.fama_values.iter().enumerate() {
            if i > 20 {
                assert!(val.is_finite());
            }
        }
        Ok(())
    }

    macro_rules! generate_all_mama_tests {
        ($($test_fn:ident),*) => {
            paste! {
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

    // Check for poison values in single output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_mama_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Test with default parameters
        let input = MamaInput::from_candles(&candles, "close", MamaParams::default());
        let output = mama_with_kernel(&input, kernel)?;

        // Check every value for poison patterns in mama_values
        for (i, &val) in output.mama_values.iter().enumerate() {
            // Skip NaN values as they're expected in the warmup period
            if val.is_nan() {
                continue;
            }

            let bits = val.to_bits();

            // Check for alloc_with_nan_prefix poison (0x11111111_11111111)
            if bits == 0x11111111_11111111 {
                panic!(
                    "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} in mama_values",
                    test_name, val, bits, i
                );
            }

            // Check for init_matrix_prefixes poison (0x22222222_22222222)
            if bits == 0x22222222_22222222 {
                panic!(
                    "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} in mama_values",
                    test_name, val, bits, i
                );
            }

            // Check for make_uninit_matrix poison (0x33333333_33333333)
            if bits == 0x33333333_33333333 {
                panic!(
                    "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} in mama_values",
                    test_name, val, bits, i
                );
            }
        }

        // Check every value for poison patterns in fama_values
        for (i, &val) in output.fama_values.iter().enumerate() {
            // Skip NaN values as they're expected in the warmup period
            if val.is_nan() {
                continue;
            }

            let bits = val.to_bits();

            // Check for alloc_with_nan_prefix poison (0x11111111_11111111)
            if bits == 0x11111111_11111111 {
                panic!(
                    "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} in fama_values",
                    test_name, val, bits, i
                );
            }

            // Check for init_matrix_prefixes poison (0x22222222_22222222)
            if bits == 0x22222222_22222222 {
                panic!(
                    "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} in fama_values",
                    test_name, val, bits, i
                );
            }

            // Check for make_uninit_matrix poison (0x33333333_33333333)
            if bits == 0x33333333_33333333 {
                panic!(
                    "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} in fama_values",
                    test_name, val, bits, i
                );
            }
        }

        Ok(())
    }

    // Release mode stub - does nothing
    #[cfg(not(debug_assertions))]
    fn check_mama_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    generate_all_mama_tests!(
        check_mama_partial_params,
        check_mama_accuracy,
        check_mama_default_candles,
        check_mama_with_insufficient_data,
        check_mama_very_small_dataset,
        check_mama_reinput,
        check_mama_nan_handling,
        check_mama_no_poison
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = MamaBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = MamaParams::default();
        let mama_row = output.mama_for(&def).expect("default row missing");
        assert_eq!(mama_row.len(), c.close.len());
        Ok(())
    }

    macro_rules! gen_batch_tests {
        ($fn_name:ident) => {
            paste! {
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

    // Check for poison values in batch output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test batch with multiple parameter combinations
        let output = MamaBatchBuilder::new()
            .kernel(kernel)
            .fast_limit_range(0.3, 0.7, 0.2)
            .slow_limit_range(0.03, 0.07, 0.02)
            .apply_candles(&c, "close")?;

        // Check every value in the entire batch matrix for poison patterns in mama_values
        for (idx, &val) in output.mama_values.iter().enumerate() {
            // Skip NaN values as they're expected in warmup periods
            if val.is_nan() {
                continue;
            }

            let bits = val.to_bits();
            let row = idx / output.cols;
            let col = idx % output.cols;

            // Check for alloc_with_nan_prefix poison (0x11111111_11111111)
            if bits == 0x11111111_11111111 {
                panic!(
                    "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (flat index {}) in mama_values",
                    test, val, bits, row, col, idx
                );
            }

            // Check for init_matrix_prefixes poison (0x22222222_22222222)
            if bits == 0x22222222_22222222 {
                panic!(
                    "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (flat index {}) in mama_values",
                    test, val, bits, row, col, idx
                );
            }

            // Check for make_uninit_matrix poison (0x33333333_33333333)
            if bits == 0x33333333_33333333 {
                panic!(
                    "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (flat index {}) in mama_values",
                    test, val, bits, row, col, idx
                );
            }
        }

        // Check every value in the entire batch matrix for poison patterns in fama_values
        for (idx, &val) in output.fama_values.iter().enumerate() {
            // Skip NaN values as they're expected in warmup periods
            if val.is_nan() {
                continue;
            }

            let bits = val.to_bits();
            let row = idx / output.cols;
            let col = idx % output.cols;

            // Check for alloc_with_nan_prefix poison (0x11111111_11111111)
            if bits == 0x11111111_11111111 {
                panic!(
                    "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (flat index {}) in fama_values",
                    test, val, bits, row, col, idx
                );
            }

            // Check for init_matrix_prefixes poison (0x22222222_22222222)
            if bits == 0x22222222_22222222 {
                panic!(
                    "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (flat index {}) in fama_values",
                    test, val, bits, row, col, idx
                );
            }

            // Check for make_uninit_matrix poison (0x33333333_33333333)
            if bits == 0x33333333_33333333 {
                panic!(
                    "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (flat index {}) in fama_values",
                    test, val, bits, row, col, idx
                );
            }
        }

        Ok(())
    }

    // Release mode stub - does nothing
    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_no_poison);
}

// Python bindings
#[cfg(feature = "python")]
mod python_bindings {
    use super::*;
    use pyo3::prelude::*;
    use pyo3::exceptions::PyValueError;
    use pyo3::types::PyDict;
    use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyArrayMethods, IntoPyArray};
    use std::collections::HashMap;
    
    #[pyfunction]
    #[pyo3(name = "mama")]
    pub fn mama_py<'py>(
        py: Python<'py>,
        data: PyReadonlyArray1<'py, f64>,
        fast_limit: f64,
        slow_limit: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let slice_in = data.as_slice()?;  // zero-copy read
        let params = MamaParams {
            fast_limit: Some(fast_limit),
            slow_limit: Some(slow_limit),
        };
        let input = MamaInput::from_slice(slice_in, params);
        
        // Pre-allocate NumPy output arrays
        let mama_arr = unsafe { PyArray1::<f64>::new(py, [slice_in.len()], false) };
        let fama_arr = unsafe { PyArray1::<f64>::new(py, [slice_in.len()], false) };
        let mama_slice = unsafe { mama_arr.as_slice_mut()? };
        let fama_slice = unsafe { fama_arr.as_slice_mut()? };
        
        // Compute directly into pre-allocated arrays (zero-copy)
        py.allow_threads(|| -> Result<(), MamaError> {
            let (data, fast_limit, slow_limit, chosen) = mama_prepare(&input, Kernel::Auto)?;
            // MAMA produces values from the first data point
            // The algorithm uses adaptive techniques that start immediately
            unsafe {
                match chosen {
                    Kernel::Scalar | Kernel::ScalarBatch => {
                        mama_scalar_inplace(data, fast_limit, slow_limit, mama_slice, fama_slice);
                    }
                    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                    Kernel::Avx2 | Kernel::Avx2Batch => {
                        mama_avx2_inplace(data, fast_limit, slow_limit, mama_slice, fama_slice);
                    }
                    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                    Kernel::Avx512 | Kernel::Avx512Batch => {
                        mama_avx2_inplace(data, fast_limit, slow_limit, mama_slice, fama_slice);
                    }
                    _ => unreachable!(),
                }
            }
            Ok(())
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        // Return dict with "mama" and "fama" keys
        let dict = PyDict::new(py);
        dict.set_item("mama", mama_arr)?;
        dict.set_item("fama", fama_arr)?;
        Ok(dict)
    }
    
    #[pyfunction]
    #[pyo3(name = "mama_batch")]
    pub fn mama_batch_py<'py>(
        py: Python<'py>,
        data: PyReadonlyArray1<'py, f64>,
        fast_limit_range: (f64, f64, f64),
        slow_limit_range: (f64, f64, f64),
    ) -> PyResult<Bound<'py, PyDict>> {
        let slice_in = data.as_slice()?;
        let sweep = MamaBatchRange {
            fast_limit: fast_limit_range,
            slow_limit: slow_limit_range,
        };
        
        // 1. Expand grid once to know rows*cols
        let combos = expand_grid(&sweep);
        let rows = combos.len();
        let cols = slice_in.len();
        
        // 2. Pre-allocate NumPy arrays
        let mama_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
        let fama_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
        let mama_slice = unsafe { mama_arr.as_slice_mut()? };
        let fama_slice = unsafe { fama_arr.as_slice_mut()? };
        
        // 3. Heavy work without the GIL
        let combos = py.allow_threads(|| -> Result<Vec<MamaParams>, MamaError> {
            let kernel = match Kernel::Auto {
                Kernel::Auto => detect_best_batch_kernel(),
                k => k,
            };
            let simd = match kernel {
                Kernel::Avx512Batch => Kernel::Avx512,
                Kernel::Avx2Batch => Kernel::Avx2,
                Kernel::ScalarBatch => Kernel::Scalar,
                _ => unreachable!(),
            };
            // Use the _into variant that writes directly to our pre-allocated buffers
            mama_batch_inner_into(slice_in, &sweep, simd, true, mama_slice, fama_slice)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        // 4. Build dict with the GIL
        let dict = PyDict::new(py);
        dict.set_item("mama_values", mama_arr.reshape((rows, cols))?)?;
        dict.set_item("fama_values", fama_arr.reshape((rows, cols))?)?;
        dict.set_item(
            "fast_limits",
            combos
                .iter()
                .map(|p| p.fast_limit.unwrap_or(0.5))
                .collect::<Vec<_>>()
                .into_pyarray(py),
        )?;
        dict.set_item(
            "slow_limits",
            combos
                .iter()
                .map(|p| p.slow_limit.unwrap_or(0.05))
                .collect::<Vec<_>>()
                .into_pyarray(py),
        )?;
        
        Ok(dict)
    }
    
    #[pyclass]
    #[pyo3(name = "MamaStream")]
    pub struct MamaStreamPy {
        inner: MamaStream,
    }
    
    #[pymethods]
    impl MamaStreamPy {
        #[new]
        pub fn new(fast_limit: f64, slow_limit: f64) -> PyResult<Self> {
            let params = MamaParams {
                fast_limit: Some(fast_limit),
                slow_limit: Some(slow_limit),
            };
            let stream = MamaStream::try_new(params)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(Self { inner: stream })
        }
        
        pub fn update(&mut self, value: f64) -> Option<(f64, f64)> {
            self.inner.update(value)
        }
    }
}

// Re-export Python bindings at module level
#[cfg(feature = "python")]
pub use python_bindings::{mama_py, mama_batch_py, MamaStreamPy};

// WASM bindings
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn mama_js(data: &[f64], fast_limit: f64, slow_limit: f64) -> Result<Vec<f64>, JsValue> {
    let params = MamaParams {
        fast_limit: Some(fast_limit),
        slow_limit: Some(slow_limit),
    };
    let input = MamaInput::from_slice(data, params);
    
    match mama(&input) {
        Ok(output) => {
            // Return both arrays concatenated: first mama_values, then fama_values
            let mut result = Vec::with_capacity(output.mama_values.len() * 2);
            result.extend_from_slice(&output.mama_values);
            result.extend_from_slice(&output.fama_values);
            Ok(result)
        }
        Err(e) => Err(JsValue::from_str(&e.to_string()))
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn mama_batch_js(
    data: &[f64],
    fast_limit_start: f64,
    fast_limit_end: f64,
    fast_limit_step: f64,
    slow_limit_start: f64,
    slow_limit_end: f64,
    slow_limit_step: f64,
) -> Result<Vec<f64>, JsValue> {
    let range = MamaBatchRange {
        fast_limit: (fast_limit_start, fast_limit_end, fast_limit_step),
        slow_limit: (slow_limit_start, slow_limit_end, slow_limit_step),
    };
    
    match mama_batch_with_kernel(data, &range, Kernel::Auto) {
        Ok(batch_output) => {
            // Return both arrays concatenated: first all mama_values, then all fama_values
            let mut result = Vec::with_capacity(batch_output.mama_values.len() + batch_output.fama_values.len());
            result.extend_from_slice(&batch_output.mama_values);
            result.extend_from_slice(&batch_output.fama_values);
            Ok(result)
        }
        Err(e) => Err(JsValue::from_str(&e.to_string()))
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn mama_batch_metadata_js(
    fast_limit_start: f64,
    fast_limit_end: f64,
    fast_limit_step: f64,
    slow_limit_start: f64,
    slow_limit_end: f64,
    slow_limit_step: f64,
) -> Vec<f64> {
    let range = MamaBatchRange {
        fast_limit: (fast_limit_start, fast_limit_end, fast_limit_step),
        slow_limit: (slow_limit_start, slow_limit_end, slow_limit_step),
    };
    
    let combos = expand_grid(&range);
    let mut metadata = Vec::with_capacity(combos.len() * 2);
    
    for combo in combos {
        metadata.push(combo.fast_limit.unwrap_or(0.5));
        metadata.push(combo.slow_limit.unwrap_or(0.05));
    }
    
    metadata
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn mama_batch_rows_cols_js(
    fast_limit_start: f64,
    fast_limit_end: f64,
    fast_limit_step: f64,
    slow_limit_start: f64,
    slow_limit_end: f64,
    slow_limit_step: f64,
    data_len: usize,
) -> Vec<usize> {
    let range = MamaBatchRange {
        fast_limit: (fast_limit_start, fast_limit_end, fast_limit_step),
        slow_limit: (slow_limit_start, slow_limit_end, slow_limit_step),
    };
    
    let combos = expand_grid(&range);
    vec![combos.len(), data_len]
}
