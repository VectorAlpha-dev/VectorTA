/// # Arnaud Legoux Moving Average (ALMA)
///
/// A smooth yet responsive moving average that uses Gaussian weighting. Its parameters
/// (`period`, `offset`, `sigma`) control the window size, the weighting center, and
/// the Gaussian smoothness. ALMA can also be re-applied to its own output, allowing
/// iterative smoothing on previously computed results.
///
/// ## Parameters
/// - **period**: Window size (number of data points).
/// - **offset**: Shift in [0.0, 1.0] for the Gaussian center (defaults to 0.85).
/// - **sigma**: Controls the Gaussian curve’s width (defaults to 6.0).
///
/// ## Errors
/// - **AllValuesNaN**: alma: All input data values are `NaN`.
/// - **InvalidPeriod**: alma: `period` is zero or exceeds the data length.
/// - **NotEnoughValidData**: alma: Not enough valid data points for the requested `period`.
/// - **InvalidSigma**: alma: `sigma` ≤ 0.0.
/// - **InvalidOffset**: alma: `offset` is `NaN` or infinite.
///
/// ## Returns
/// - **`Ok(AlmaOutput)`** on success, containing a `Vec<f64>` of length matching the input.
/// - **`Err(AlmaError)`** otherwise.
///
use crate::utilities::data_loader::{source_type, source_type_f32, Candles};
use core::arch::x86_64::*;
use std::error::Error;
use crate::utilities::enums::Kernel;	

macro_rules! impl_alma_indicator {
    (
        // the concrete float you want (f64 | f32)
        $Float:ty,
        // a short suffix used in the generated identifiers (F64 | F32)
        $suffix:ident,
        // stem of the “engine” function that actually does the math
        //   e.g.  alma_with_kernel            , alma_f32_with_kernel
        $algo_fn:ident,
        // stem of the streaming type      AlmaStream , AlmaStreamF32
        $stream_ty:ident
    ) => {
        paste::paste! {

            /*────────────  ① Data / Params / Output  ────────────*/

            #[derive(Debug, Clone)]
            pub enum [<AlmaData $suffix>]<'a> {
                Candles { candles: &'a Candles, source: &'a str },
                Slice(&'a [$Float]),
            }

            #[derive(Debug, Clone)]
            pub struct [<AlmaOutput $suffix>] { pub values: Vec<$Float> }

            #[derive(Debug, Clone)]
            pub struct [<AlmaParams $suffix>] {
                pub period: Option<usize>,
                pub offset: Option<$Float>,
                pub sigma : Option<$Float>,
            }

            impl Default for [<AlmaParams $suffix>] {
                fn default() -> Self {
                    Self { period: Some(9),
                           offset: Some(0.85 as $Float),
                           sigma : Some(6.0 as $Float) }
                }
            }

            /*──────────── ② Input helper  ────────────*/

            #[derive(Debug, Clone)]
            pub struct [<AlmaInput $suffix>]<'a> {
                pub data  : [<AlmaData $suffix>]<'a>,
                pub params: [<AlmaParams $suffix>],
            }

            impl<'a> [<AlmaInput $suffix>]<'a> {
                #[inline] pub fn from_candles(c:&'a Candles,
                                              s:&'a str,
                                              p:[<AlmaParams $suffix>]) -> Self {
                    Self { data:[<AlmaData $suffix>]::Candles{candles:c,source:s}, params:p }
                }
                #[inline] pub fn from_slice(sl:&'a [$Float],
                                            p:[<AlmaParams $suffix>]) -> Self {
                    Self { data:[<AlmaData $suffix>]::Slice(sl), params:p }
                }
                #[inline] pub fn with_default_candles(c:&'a Candles) -> Self {
                    Self::from_candles(c,"close",[<AlmaParams $suffix>]::default())
                }
                #[inline] pub fn get_period(&self)->usize  { self.params.period.unwrap_or(9) }
                #[inline] pub fn get_offset(&self)->$Float { self.params.offset.unwrap_or(0.85 as $Float) }
                #[inline] pub fn get_sigma (&self)->$Float { self.params.sigma .unwrap_or(6.0  as $Float) }
            }

            /*──────────── ③ Builder façade  ────────────*/

            #[derive(Copy, Clone, Debug)]
            pub struct [<AlmaBuilder $suffix>] {
                period: Option<usize>,
                offset: Option<$Float>,
                sigma : Option<$Float>,
                kernel: Kernel,
            }

            impl Default for [<AlmaBuilder $suffix>] {
                fn default() -> Self {
                    Self { period:None, offset:None, sigma:None, kernel:Kernel::Auto }
                }
            }

            impl [<AlmaBuilder $suffix>] {
                #[inline(always)] pub fn new()            -> Self { Self::default() }
                #[inline(always)] pub fn period(mut self,n:usize) -> Self { self.period=Some(n); self }
                #[inline(always)] pub fn offset(mut self,x:$Float)-> Self { self.offset=Some(x); self }
                #[inline(always)] pub fn sigma (mut self,s:$Float)-> Self { self.sigma =Some(s); self }
                #[inline(always)] pub fn kernel(mut self,k:Kernel)-> Self { self.kernel=k; self }

                #[inline(always)]
                pub fn apply(self,c:&Candles) -> Result<[<AlmaOutput $suffix>],AlmaError>{
                    let p=[<AlmaParams $suffix>]{period:self.period,
                                                offset:self.offset,
                                                sigma :self.sigma};
                    let i=[<AlmaInput $suffix>]::from_candles(c,"close",p);
                    [<$algo_fn _with_kernel>](&i,self.kernel)
                }

                #[inline(always)]
                pub fn apply_slice(self,d:&[$Float]) -> Result<[<AlmaOutput $suffix>],AlmaError>{
                    let p=[<AlmaParams $suffix>]{period:self.period,offset:self.offset,sigma:self.sigma};
                    let i=[<AlmaInput $suffix>]::from_slice(d,p);
                    [<$algo_fn _with_kernel>](&i,self.kernel)
                }

                #[inline(always)]
                pub fn into_stream(self) -> Result<[< $stream_ty $suffix>],AlmaError>{
                    let p=[<AlmaParams $suffix>]{period:self.period,offset:self.offset,sigma:self.sigma};
                    [< $stream_ty $suffix>]::try_new(p)
                }
            }
        } // paste!
    };
}

impl_alma_indicator!(f64, F64, alma,       AlmaStream);
impl_alma_indicator!(f32, F32, alma_f32,   AlmaStream);
pub type  AlmaData<'a>   = AlmaDataF64<'a>;
pub type  AlmaInput<'a>  = AlmaInputF64<'a>;
pub type  AlmaParams     = AlmaParamsF64;
pub type  AlmaOutput     = AlmaOutputF64;
pub type  AlmaBuilder    = AlmaBuilderF64;
pub type  AlmaStream     = AlmaStreamF64;

use thiserror::Error;

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

// macros.rs — add right after impl_alma_indicator! / impl_alma_stream!
#[macro_export]
macro_rules! impl_alma_kernels {
    ($Float:ty,       // f64 | f32
     $suffix:ident,   // F64 | F32
     $weights_exp:expr,          // lambda: |i,m,s2| -> $Float  (exp expression)
     $src_fn:ident,              // source_type  | source_type_f32
     $core_scalar:ident,          // alma_core_scalar    | alma_core_scalar_f32
     $core_avx2:ident,         // NEW
     $core_avx512:ident        // NEW
    ) => {
        paste::paste! {

            /*─────────────────────────────────────────
             *  Top-level dispatcher
             *────────────────────────────────────────*/
            #[inline]
            pub fn [<alma_with_kernel $suffix:lower>](
                input : &[<AlmaInput $suffix>],
                kernel: Kernel
            ) -> Result<[<AlmaOutput $suffix>], AlmaError>
            {
                // 1.  Borrow the raw series
                let data: &[$Float] = match &input.data {
                    [<AlmaData $suffix>]::Candles { candles, source } =>
                        $src_fn(candles, source),
                    [<AlmaData $suffix>]::Slice(sl) => sl,
                };

                // 2.  Validate + extract parameters  (identical for both widths)
                let first = data.iter().position(|x| !x.is_nan())
                                        .ok_or(AlmaError::AllValuesNaN)?;

                let len      = data.len();
                let period   = input.get_period();
                let offset   = input.get_offset();
                let sigma    = input.get_sigma();

                if period == 0 || period > len {
                    return Err(AlmaError::InvalidPeriod { period, data_len: len });
                }
                if (len - first) < period {
                    return Err(AlmaError::NotEnoughValidData {
                        needed: period, valid: len - first
                    });
                }
                if sigma <= 0.0 {
                    return Err(AlmaError::InvalidSigma { sigma: sigma as f64 });
                }
                if offset.is_nan() || offset.is_infinite() {
                    return Err(AlmaError::InvalidOffset { offset: offset as f64 });
                }

                // 3.  Pre-compute Gaussian weights
                let m   = offset * (period - 1) as $Float;
                let s   = period as $Float / sigma;
                let s2  = 2.0 as $Float * s * s;

                let mut weights = Vec::with_capacity(period);
                let mut norm    = 0.0 as $Float;
                for i in 0..period {
                    let w = $weights_exp(i, m, s2);
                    weights.push(w);
                    norm += w;
                }
                let inv_norm = 1.0 as $Float / norm;

                // 4.  Allocate output
                let mut out = vec![<$Float>::NAN; len];

                // 5.  Choose kernel & run
                let chosen = match kernel { Kernel::Auto => detect_best_kernel(),
                                            other             => other };

                unsafe {
                    match chosen {
                        Kernel::Scalar  => $core_scalar (
                            data, &weights, period, first, inv_norm, &mut out),
                        Kernel::Avx2    => $core_avx2   (
                            data, &weights, period, first, inv_norm, &mut out),
                        Kernel::Avx512  => $core_avx512 (
                            data, &weights, period, first, inv_norm, &mut out),
                        Kernel::Auto    => unreachable!(),
                    }
                }

                Ok([<AlmaOutput $suffix>] { values: out })
            }

            #[inline]
            pub fn [<alma $suffix:lower>](input: &[<AlmaInput $suffix>])
                -> Result<[<AlmaOutput $suffix>], AlmaError>
            {
                [<alma_with_kernel $suffix:lower>](input, Kernel::Auto)
            }


            /*─────────────────────────────────────────
             *  Scalar dot-product kernel
             *────────────────────────────────────────*/
             #[inline(always)]
             unsafe fn $core_scalar(
                 data      : &[$Float],
                 weights   : &[$Float],
                 period    : usize,
                 first_val : usize,
                 inv_norm  : $Float,
                 out       : &mut [$Float],
             ) {
                 debug_assert_eq!(weights.len(), period);
                 debug_assert!(out.len() >= data.len());
 
                 let p4 = period & !3;
                 for i in (first_val + period - 1)..data.len() {
                     let start = i + 1 - period;
                     let mut acc0 = 0.0 as $Float;
                     let mut acc1 = 0.0 as $Float;
                     let mut k    = 0;
 
                     // unroll ×8
                     while k + 8 <= p4 {
                         acc0 += data.get_unchecked(start + k    ) *
                                 weights.get_unchecked(k        );
                         acc0 += data.get_unchecked(start + k + 1) *
                                 weights.get_unchecked(k + 1    );
                         acc0 += data.get_unchecked(start + k + 2) *
                                 weights.get_unchecked(k + 2    );
                         acc0 += data.get_unchecked(start + k + 3) *
                                 weights.get_unchecked(k + 3    );
 
                         acc1 += data.get_unchecked(start + k + 4) *
                                 weights.get_unchecked(k + 4    );
                         acc1 += data.get_unchecked(start + k + 5) *
                                 weights.get_unchecked(k + 5    );
                         acc1 += data.get_unchecked(start + k + 6) *
                                 weights.get_unchecked(k + 6    );
                         acc1 += data.get_unchecked(start + k + 7) *
                                 weights.get_unchecked(k + 7    );
                         k += 8;
                     }
                     // unroll ×4
                     while k + 4 <= p4 {
                         acc0 += data.get_unchecked(start + k    ) *
                                 weights.get_unchecked(k        );
                         acc0 += data.get_unchecked(start + k + 1) *
                                 weights.get_unchecked(k + 1    );
                         acc0 += data.get_unchecked(start + k + 2) *
                                 weights.get_unchecked(k + 2    );
                         acc0 += data.get_unchecked(start + k + 3) *
                                 weights.get_unchecked(k + 3    );
                         k += 4;
                     }
                     let mut sum = acc0 + acc1;
                     while k < period {
                         sum += data.get_unchecked(start + k) *
                                weights.get_unchecked(k);
                         k += 1;
                     }
                     *out.get_unchecked_mut(i) = sum * inv_norm;
                 }
             }
        } // paste
    };
}

impl_alma_kernels!(
    /* -------------- f64 -------------- */
    f64, F64,
    /* weights lambda ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ */
    |i: usize, m: f64, s2: f64| -> f64 {
        (-(i as f64 - m).powi(2) / s2).exp()
    },
    source_type,
    alma_core_scalar,
    alma_core_avx2,
    alma_core_avx512
);

impl_alma_kernels!(
    /* -------------- f32 -------------- */
    f32, F32,
    /* weights lambda ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ */
    |i: usize, m: f32, s2: f32| -> f32 {
        let diff = i as f32 - m;
        (-(diff * diff) / s2).exp()
    },
    source_type_f32,
    alma_core_scalar_f32,
    alma_core_avx2_f32,
    alma_core_avx512_f32
);

pub use alma_with_kernelf64 as alma_with_kernel;      // f64 is the default
pub use alma_with_kernelf32 as alma_f32_with_kernel;
pub use almaf32            as alma_f32;


#[target_feature(enable = "avx2,fma")]
unsafe fn alma_core_avx2(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first_valid: usize,
    inv_norm: f64,
    out: &mut [f64],
) {

    debug_assert_eq!(weights.len(), period);
    debug_assert!(out.len() >= data.len());

    const STEP: usize = 4;                  // 4 × f64 per YMM
    let chunks = period / STEP;             // full 256-bit blocks
    let tail   = period % STEP;             // 0-to-3 element remainder

    /* Build a mask for the tail once.
       Each 64-bit lane is 0xFFFF_FFFF_FFFF_FFFF if to be loaded, 0 otherwise. */
    let tail_mask = match tail {
        0 => _mm256_setzero_si256(),
        1 => _mm256_setr_epi64x(-1, 0, 0, 0),
        2 => _mm256_setr_epi64x(-1, -1, 0, 0),
        3 => _mm256_setr_epi64x(-1, -1, -1, 0),
        _ => unreachable!(),
    };

    for i in (first_valid + period - 1)..data.len() {
        let start = i + 1 - period;

        /* Two accumulators → two independent FMA streams → both execution
           pipelines stay busy (1 × FMA / pipe / cycle on all AVX2 cores).   */
        let mut acc0 = _mm256_setzero_pd();
        let mut acc1 = _mm256_setzero_pd();

        /* -------- main body: 2 × STEP per iteration -------- */
        let paired = chunks & !1;           // even part we can process in pairs
        let mut blk = 0;
        while blk < paired {
            let idx0 = blk * STEP;
            let idx1 = idx0 + STEP;

            let w0 = _mm256_loadu_pd(weights.as_ptr().add(idx0));
            let d0 = _mm256_loadu_pd(data   .as_ptr().add(start + idx0));
            acc0   = _mm256_fmadd_pd(d0, w0, acc0);

            let w1 = _mm256_loadu_pd(weights.as_ptr().add(idx1));
            let d1 = _mm256_loadu_pd(data   .as_ptr().add(start + idx1));
            acc1   = _mm256_fmadd_pd(d1, w1, acc1);

            blk += 2;
        }

        /* -------- odd leftover block (if chunks is odd) ----- */
        if chunks & 1 != 0 {
            let idx = (chunks - 1) * STEP;
            let w   = _mm256_loadu_pd(weights.as_ptr().add(idx));
            let d   = _mm256_loadu_pd(data   .as_ptr().add(start + idx));
            acc0    = _mm256_fmadd_pd(d, w, acc0);
        }

        /* ---------------- tail ≤ 3 doubles ------------------ */
        if tail != 0 {
            let w_tail = _mm256_maskload_pd(weights.as_ptr().add(chunks * STEP), tail_mask);
            let d_tail = _mm256_maskload_pd(data   .as_ptr().add(start  + chunks * STEP), tail_mask);
            acc0       = _mm256_fmadd_pd(d_tail, w_tail, acc0);
        }

        /* -------------- horizontal reduction ---------------- */
        acc0 = _mm256_add_pd(acc0, acc1);
        let hi   = _mm256_extractf128_pd(acc0, 1);
        let lo   = _mm256_castpd256_pd128(acc0);
        let sum2 = _mm_add_pd(hi, lo);                  // 2 × f64
        let sum1 = _mm_add_pd(sum2, _mm_unpackhi_pd(sum2, sum2));
        let sum  = _mm_cvtsd_f64(sum1);

        *out.get_unchecked_mut(i) = sum * inv_norm;
    }
}
use core::arch::x86_64::*;

#[target_feature(enable = "avx512f,fma")]
unsafe fn alma_core_avx512(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first_valid: usize,
    inv_norm: f64,
    out: &mut [f64],
) {


    const STEP: usize = 8;                // 8 × f64 per ZMM
    let chunks = period / STEP;           // full 512-bit blocks
    let tail   = period % STEP;           // 0-to-7 element remainder
    let tail_mask: __mmask8 = (1u8 << tail).wrapping_sub(1);

    for i in (first_valid + period - 1)..data.len() {
        let start = i + 1 - period;

        // two independent accumulators (keeps both FMA ports busy)
        let mut acc0 = _mm512_setzero_pd();
        let mut acc1 = _mm512_setzero_pd();

        /* -------- main body: two 512-bit FMAs per iteration -------- */
        let paired = chunks & !1;                     // even part
        for blk in (0..paired).step_by(2) {
            let idx0 = blk * STEP;
            let idx1 = idx0 + STEP;

            // unaligned loads are safe & just as fast on SKX/Zen4
            let w0 = _mm512_loadu_pd(weights.as_ptr().add(idx0));
            let w1 = _mm512_loadu_pd(weights.as_ptr().add(idx1));
            let d0 = _mm512_loadu_pd(data.as_ptr().add(start + idx0));
            let d1 = _mm512_loadu_pd(data.as_ptr().add(start + idx1));

            acc0 = _mm512_fmadd_pd(d0, w0, acc0);
            acc1 = _mm512_fmadd_pd(d1, w1, acc1);
        }

        /* -------- odd leftover block (if period/8 is odd) ---------- */
        if chunks & 1 != 0 {
            let idx = (chunks - 1) * STEP;
            let w   = _mm512_loadu_pd(weights.as_ptr().add(idx));
            let d   = _mm512_loadu_pd(data.as_ptr().add(start + idx));
            acc0    = _mm512_fmadd_pd(d, w, acc0);
        }

        /* --------------------- tail ≤ 7 doubles --------------------- */
        if tail != 0 {
            let w_tail = _mm512_maskz_loadu_pd(tail_mask, weights.as_ptr().add(chunks * STEP));
            let d_tail = _mm512_maskz_loadu_pd(tail_mask, data   .as_ptr().add(start  + chunks * STEP));
            acc0       = _mm512_fmadd_pd(d_tail, w_tail, acc0);
        }

        /* ---------------- horizontal reduction --------------------- */
        acc0 = _mm512_add_pd(acc0, acc1);
        let sum = _mm512_reduce_add_pd(acc0);

        *out.get_unchecked_mut(i) = sum * inv_norm;
    }
}


#[target_feature(enable = "avx512f,fma")]
unsafe fn alma_core_avx512_f32(
    data: &[f32],
    weights: &[f32],
    period: usize,
    first_valid: usize,
    inv_norm: f32,
    out: &mut [f32],
) {
    debug_assert_eq!(weights.len(), period);
    debug_assert!(out.len() >= data.len());

    let step = 16;
    let chunk = step * 6;
    let p16 = period & !15;

    for i in (first_valid + period - 1)..data.len() {
        let start = i + 1 - period;

        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();
        let mut acc2 = _mm512_setzero_ps();
        let mut acc3 = _mm512_setzero_ps();
        let mut acc4 = _mm512_setzero_ps();
        let mut acc5 = _mm512_setzero_ps();

        let mut k = 0;
        while k + chunk <= p16 {
            let v0 = _mm512_loadu_ps(data.as_ptr().add(start + k + step * 0));
            let w0 = _mm512_loadu_ps(weights.as_ptr().add(k + step * 0));
            acc0 = _mm512_fmadd_ps(v0, w0, acc0);

            let v1 = _mm512_loadu_ps(data.as_ptr().add(start + k + step * 1));
            let w1 = _mm512_loadu_ps(weights.as_ptr().add(k + step * 1));
            acc1 = _mm512_fmadd_ps(v1, w1, acc1);

            let v2 = _mm512_loadu_ps(data.as_ptr().add(start + k + step * 2));
            let w2 = _mm512_loadu_ps(weights.as_ptr().add(k + step * 2));
            acc2 = _mm512_fmadd_ps(v2, w2, acc2);

            let v3 = _mm512_loadu_ps(data.as_ptr().add(start + k + step * 3));
            let w3 = _mm512_loadu_ps(weights.as_ptr().add(k + step * 3));
            acc3 = _mm512_fmadd_ps(v3, w3, acc3);

            let v4 = _mm512_loadu_ps(data.as_ptr().add(start + k + step * 4));
            let w4 = _mm512_loadu_ps(weights.as_ptr().add(k + step * 4));
            acc4 = _mm512_fmadd_ps(v4, w4, acc4);

            let v5 = _mm512_loadu_ps(data.as_ptr().add(start + k + step * 5));
            let w5 = _mm512_loadu_ps(weights.as_ptr().add(k + step * 5));
            acc5 = _mm512_fmadd_ps(v5, w5, acc5);

            k += chunk;
        }

        acc0 = _mm512_add_ps(acc0, acc1);
        acc2 = _mm512_add_ps(acc2, acc3);
        acc4 = _mm512_add_ps(acc4, acc5);

        acc0 = _mm512_add_ps(acc0, acc2);
        acc0 = _mm512_add_ps(acc0, acc4);

        while k + step <= p16 {
            let v = _mm512_loadu_ps(data.as_ptr().add(start + k));
            let wt = _mm512_loadu_ps(weights.as_ptr().add(k));
            acc0 = _mm512_fmadd_ps(v, wt, acc0);
            k += step;
        }

        let mut acc_scalar = _mm512_reduce_add_ps(acc0);

        while k < period {
            acc_scalar += *data.get_unchecked(start + k) * *weights.get_unchecked(k);
            k += 1;
        }

        *out.get_unchecked_mut(i) = acc_scalar * inv_norm;
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn alma_core_avx2_f32(
    data: &[f32],
    weights: &[f32],
    period: usize,
    first_valid: usize,
    inv_norm: f32,
    out: &mut [f32],
) {
    use core::arch::x86_64::*;

    debug_assert_eq!(weights.len(), period);
    debug_assert!(out.len() >= data.len());

    let step = 8;
    let chunk = step * 4;
    let p8 = period & !7;

    for i in (first_valid + period - 1)..data.len() {
        let start = i + 1 - period;

        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();

        let mut k = 0;

        while k + chunk <= p8 {
            let v0 = _mm256_loadu_ps(data.as_ptr().add(start + k + step * 0));
            let w0 = _mm256_loadu_ps(weights.as_ptr().add(k + step * 0));
            acc0 = _mm256_fmadd_ps(v0, w0, acc0);

            let v1 = _mm256_loadu_ps(data.as_ptr().add(start + k + step * 1));
            let w1 = _mm256_loadu_ps(weights.as_ptr().add(k + step * 1));
            acc1 = _mm256_fmadd_ps(v1, w1, acc1);

            let v2 = _mm256_loadu_ps(data.as_ptr().add(start + k + step * 2));
            let w2 = _mm256_loadu_ps(weights.as_ptr().add(k + step * 2));
            acc2 = _mm256_fmadd_ps(v2, w2, acc2);

            let v3 = _mm256_loadu_ps(data.as_ptr().add(start + k + step * 3));
            let w3 = _mm256_loadu_ps(weights.as_ptr().add(k + step * 3));
            acc3 = _mm256_fmadd_ps(v3, w3, acc3);

            k += chunk;
        }

        acc0 = _mm256_add_ps(acc0, acc1);
        acc2 = _mm256_add_ps(acc2, acc3);
        acc0 = _mm256_add_ps(acc0, acc2);

        while k + step <= p8 {
            let vd = _mm256_loadu_ps(data.as_ptr().add(start + k));
            let wt = _mm256_loadu_ps(weights.as_ptr().add(k));
            acc0 = _mm256_fmadd_ps(vd, wt, acc0);
            k += step;
        }

        let hi = _mm256_extractf128_ps(acc0, 1);
        let lo = _mm256_castps256_ps128(acc0);
        let sum128 = _mm_add_ps(hi, lo); // lanes 0-3 + 4-7
        let sum64 = _mm_hadd_ps(sum128, sum128);
        let sum32 = _mm_hadd_ps(sum64, sum64);
        let mut acc_scalar = _mm_cvtss_f32(sum32);

        while k < period {
            acc_scalar += *data.get_unchecked(start + k) * *weights.get_unchecked(k);
            k += 1;
        }

        *out.get_unchecked_mut(i) = acc_scalar * inv_norm;
    }
}


#[macro_export]
macro_rules! impl_alma_stream {
    ($Float:ty, $suffix:ident) => {
        paste::paste! {
            #[derive(Debug, Clone)]
            pub struct [<AlmaStream $suffix>] {
                period:   usize,
                weights:  Vec<$Float>,
                inv_norm: $Float,
                buffer:   Vec<$Float>,
                head:     usize,
                filled:   bool,
            }

            impl [<AlmaStream $suffix>] {
                pub fn try_new(
                    params: [<AlmaParams $suffix>]
                ) -> Result<Self, AlmaError> {
                    // ── validate input ─────────────────────────────
                    let period = params.period.unwrap_or(9);
                    if period == 0 {
                        return Err(AlmaError::InvalidPeriod { period, data_len: 0 });
                    }
                    let offset = params.offset.unwrap_or(0.85 as $Float);
                    if offset.is_nan() || offset.is_infinite() {
                        // AlmaError keeps f64 fields, so cast if we’re in f32-land
                        return Err(AlmaError::InvalidOffset { offset: offset as f64 });
                    }
                    let sigma = params.sigma.unwrap_or(6.0 as $Float);
                    if sigma <= 0.0 {
                        return Err(AlmaError::InvalidSigma { sigma: sigma as f64 });
                    }

                    // ── pre-compute Gaussian weights ───────────────
                    let m   = offset * (period - 1) as $Float;
                    let s   = period as $Float / sigma;
                    let s2  = 2.0 as $Float * s * s;

                    let mut weights = Vec::with_capacity(period);
                    let mut norm    = 0.0 as $Float;
                    for i in 0..period {
                        let diff = i as $Float - m;
                        let w    = (-(diff * diff) / s2).exp();
                        weights.push(w);
                        norm += w;
                    }
                    let inv_norm = 1.0 as $Float / norm;

                    Ok(Self {
                        period,
                        weights,
                        inv_norm,
                        buffer: vec![<$Float>::NAN; period],
                        head:   0,
                        filled: false,
                    })
                }

                #[inline(always)]
                pub fn update(&mut self, value: $Float) -> Option<$Float> {
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
                fn dot_ring(&self) -> $Float {
                    let mut sum = 0.0 as $Float;
                    let mut idx = self.head;
                    for &w in &self.weights {
                        sum += w * self.buffer[idx];
                        idx = (idx + 1) % self.period;
                    }
                    sum * self.inv_norm
                }
            }
        } // paste
    };
}

impl_alma_stream!(f64, F64);
impl_alma_stream!(f32, F32);

use super::*;
use crate::utilities::data_loader::read_candles_from_csv;

fn check_alma_streaming(
    test_name: &str,
    kernel: Kernel,
    is_f32: bool,
) -> Result<(), Box<dyn Error>> {
    if kernel == Kernel::Avx2
        && !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"))
    {
        eprintln!("[{}] Skipping AVX2 test on non-AVX2 CPU", test_name);
        return Ok(());
    }
    if kernel == Kernel::Avx512
        && !(is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("fma"))
    {
        eprintln!("[{}] Skipping AVX512 test on non-AVX512 CPU", test_name);
        return Ok(());
    }

    let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
    let candles = read_candles_from_csv(file_path)?;

    let period = 9;
    let offset = 0.85;
    let sigma  = 6.0;

    if !is_f32 {
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

        let mut stream =
            AlmaStreamF64::try_new(AlmaParams {
                period: Some(period),
                offset: Some(offset),
                sigma:  Some(sigma),
            })
            .expect("Failed to create AlmaStreamF64");

        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(alma_val) => stream_values.push(alma_val),
                None => stream_values.push(f64::NAN),
            }
        }

        assert_eq!(batch_output.len(), stream_values.len());
        for (i, (&b, &s)) in batch_output.iter().zip(stream_values.iter()).enumerate() {
            // If both are NaN, skip
            if b.is_nan() && s.is_nan() {
                continue;
            }
            let diff = (b - s).abs();
            assert!(
                diff < 1e-9,
                "[{}] ALMA streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name, i, b, s, diff
            );
        }
    } else {
        let closes_f32: Vec<f32> = candles.close.iter().map(|&x| x as f32).collect();

        let input_f32 = AlmaInputF32::from_slice(
            &closes_f32,
            AlmaParamsF32 {
                period: Some(period),
                offset: Some(offset as f32),
                sigma:  Some(sigma as f32),
            },
        );
        let batch_output_f32 = alma_f32_with_kernel(&input_f32, kernel)?.values;
        let mut stream_f32 =
            AlmaStreamF32::try_new(AlmaParamsF32 {
                period: Some(period),
                offset: Some(offset as f32),
                sigma:  Some(sigma as f32),
            })
            .expect("Failed to create AlmaStreamF32");

        let mut stream_values_f32 = Vec::with_capacity(closes_f32.len());
        for &price in &closes_f32 {
            match stream_f32.update(price) {
                Some(alma_val) => stream_values_f32.push(alma_val),
                None => stream_values_f32.push(f32::NAN),
            }
        }

        assert_eq!(batch_output_f32.len(), stream_values_f32.len());
        for (i, (&b, &s)) in batch_output_f32
            .iter()
            .zip(stream_values_f32.iter())
            .enumerate()
        {
            if b.is_nan() && s.is_nan() {
                continue;
            }
            let diff = (b - s).abs();
            assert!(
                diff < 1e-1,
                "[{}] ALMA streaming f32 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name, i, b, s, diff
            );
        }
    }

    Ok(())
}

fn check_alma_partial_params(
    test_name: &str,
    kernel: Kernel,
    is_f32: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if kernel == Kernel::Avx2
        && !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"))
    {
        eprintln!("[{}] Skipping AVX2 test on non-AVX2 CPU", test_name);
        return Ok(());
    }
    if kernel == Kernel::Avx512
        && !(is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("fma"))
    {
        eprintln!("[{}] Skipping AVX512 test on non-AVX512 CPU", test_name);
        return Ok(());
    }

    let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
    let candles = read_candles_from_csv(file_path)?;

    if !is_f32 {
        let default_params = AlmaParams {
            period: None,
            offset: None,
            sigma: None,
        };
        let input = AlmaInput::from_candles(&candles, "close", default_params);
        let output = alma_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
    } else {
        let default_params = AlmaParamsF32 {
            period: None,
            offset: None,
            sigma: None,
        };
        let input_f32 = AlmaInputF32::from_candles(&candles, "close", default_params);
        let output_f32 = alma_f32_with_kernel(&input_f32, kernel)?;
        assert_eq!(output_f32.values.len(), candles.close.len());
    }

    Ok(())
}

fn check_alma_accuracy(
    test_name: &str,
    kernel: Kernel,
    is_f32: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if kernel == Kernel::Avx2
        && !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"))
    {
        eprintln!("[{}] Skipping AVX2 test on non-AVX2 CPU", test_name);
        return Ok(());
    }
    if kernel == Kernel::Avx512
        && !(is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("fma"))
    {
        eprintln!("[{}] Skipping AVX512 test on non-AVX512 CPU", test_name);
        return Ok(());
    }

    let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
    let candles = read_candles_from_csv(file_path)?;

    if !is_f32 {
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
            assert!(
                (val - expected_last_five[i]).abs() < 1e-8,
                "[{}] ALMA {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
    } else {
        let input_f32 = AlmaInputF32::from_candles(&candles, "close", AlmaParamsF32::default());
        let result_f32 = alma_f32_with_kernel(&input_f32, kernel)?;
        let expected_last_five = [
            59286.72216704_f32,
            59273.53428138_f32,
            59204.37290721_f32,
            59155.93381742_f32,
            59026.92526112_f32,
        ];
        let start = result_f32.values.len().saturating_sub(5);
        for (i, &val) in result_f32.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-2,
                "[{}] ALMA-f32 {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
    }

    Ok(())
}

fn check_alma_default_candles(
    test_name: &str,
    kernel: Kernel,
    is_f32: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if kernel == Kernel::Avx2
        && !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"))
    {
        eprintln!("[{}] Skipping AVX2 test on non-AVX2 CPU", test_name);
        return Ok(());
    }
    if kernel == Kernel::Avx512
        && !(is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("fma"))
    {
        eprintln!("[{}] Skipping AVX512 test on non-AVX512 CPU", test_name);
        return Ok(());
    }

    let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
    let candles = read_candles_from_csv(file_path)?;

    if !is_f32 {
        let input = AlmaInput::with_default_candles(&candles);
        match input.data {
            AlmaData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected AlmaData::Candles"),
        }
        let output = alma_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
    } else {
        let input_f32 = AlmaInputF32::with_default_candles(&candles);
        match input_f32.data {
            AlmaDataF32::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected AlmaDataF32::Candles"),
        }
        let output_f32 = alma_f32_with_kernel(&input_f32, kernel)?;
        assert_eq!(output_f32.values.len(), candles.close.len());
    }

    Ok(())
}

fn check_alma_zero_period(
    test_name: &str,
    kernel: Kernel,
    is_f32: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let input_data = [10.0, 20.0, 30.0];
    if !is_f32 {
        let params = AlmaParams {
            period: Some(0),
            offset: None,
            sigma: None,
        };
        let input = AlmaInput::from_slice(&input_data, params);
        let res = alma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] ALMA should fail with zero period",
            test_name
        );
    } else {
        let data_f32 = [10.0_f32, 20.0, 30.0];
        let params_f32 = AlmaParamsF32 {
            period: Some(0),
            offset: None,
            sigma: None,
        };
        let input_f32 = AlmaInputF32::from_slice(&data_f32, params_f32);
        let res = alma_f32_with_kernel(&input_f32, kernel);
        assert!(
            res.is_err(),
            "[{}] ALMA f32 should fail with zero period",
            test_name
        );
    }
    Ok(())
}

fn check_alma_period_exceeds_length(
    test_name: &str,
    kernel: Kernel,
    is_f32: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let data_small = [10.0, 20.0, 30.0];
    if !is_f32 {
        let params = AlmaParams {
            period: Some(10),
            offset: None,
            sigma: None,
        };
        let input = AlmaInput::from_slice(&data_small, params);
        let res = alma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] ALMA should fail with period exceeding length",
            test_name
        );
    } else {
        let data_f32 = [10.0_f32, 20.0, 30.0];
        let params_f32 = AlmaParamsF32 {
            period: Some(10),
            offset: None,
            sigma: None,
        };
        let input_f32 = AlmaInputF32::from_slice(&data_f32, params_f32);
        let res = alma_f32_with_kernel(&input_f32, kernel);
        assert!(
            res.is_err(),
            "[{}] ALMA-f32 should fail with period exceeding length",
            test_name
        );
    }
    Ok(())
}

fn check_alma_very_small_dataset(
    test_name: &str,
    kernel: Kernel,
    is_f32: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let single_point = [42.0];
    if !is_f32 {
        let params = AlmaParams {
            period: Some(9),
            offset: None,
            sigma: None,
        };
        let input = AlmaInput::from_slice(&single_point, params);
        let res = alma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] ALMA should fail with insufficient data",
            test_name
        );
    } else {
        let single_f32 = [42.0_f32];
        let params_f32 = AlmaParamsF32 {
            period: Some(9),
            offset: None,
            sigma: None,
        };
        let input_f32 = AlmaInputF32::from_slice(&single_f32, params_f32);
        let res = alma_f32_with_kernel(&input_f32, kernel);
        assert!(
            res.is_err(),
            "[{}] ALMA-f32 should fail with insufficient data",
            test_name
        );
    }
    Ok(())
}

fn check_alma_reinput(
    test_name: &str,
    kernel: Kernel,
    is_f32: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
    let candles = read_candles_from_csv(file_path)?;

    if !is_f32 {
        let first_params = AlmaParams {
            period: Some(9),
            offset: None,
            sigma: None,
        };
        let first_input = AlmaInput::from_candles(&candles, "close", first_params);
        let first_result = alma_with_kernel(&first_input, kernel)?;
        let second_params = AlmaParams {
            period: Some(9),
            offset: None,
            sigma: None,
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
            assert!(
                (val - expected_last_five[i]).abs() < 1e-8,
                "[{}] ALMA Slice Reinput {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
    } else {
        let first_params_f32 = AlmaParamsF32 {
            period: Some(9),
            offset: None,
            sigma: None,
        };
        let first_input_f32 = AlmaInputF32::from_candles(&candles, "close", first_params_f32);
        let first_result_f32 = alma_f32_with_kernel(&first_input_f32, kernel)?;
        let second_params_f32 = AlmaParamsF32 {
            period: Some(9),
            offset: None,
            sigma: None,
        };
        let second_input_f32 =
            AlmaInputF32::from_slice(&first_result_f32.values, second_params_f32);
        let second_result_f32 = alma_f32_with_kernel(&second_input_f32, kernel)?;
        let expected_last_five = [
            59140.73195170_f32,
            59211.58090986_f32,
            59238.16030697_f32,
            59222.63528822_f32,
            59165.14427332_f32,
        ];
        let start = second_result_f32.values.len().saturating_sub(5);
        for (i, &val) in second_result_f32.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-2,
                "[{}] ALMA-f32 Slice Reinput {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        assert_eq!(
            second_result_f32.values.len(),
            first_result_f32.values.len()
        );
    }
    Ok(())
}

fn check_alma_nan_handling(
    test_name: &str,
    kernel: Kernel,
    is_f32: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
    let candles = read_candles_from_csv(file_path)?;

    if !is_f32 {
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
    } else {
        let input_f32 = AlmaInputF32::from_candles(
            &candles,
            "close",
            AlmaParamsF32 {
                period: Some(9),
                offset: None,
                sigma: None,
            },
        );
        let res_f32 = alma_f32_with_kernel(&input_f32, kernel)?;
        assert_eq!(res_f32.values.len(), candles.close.len());
        if res_f32.values.len() > 240 {
            for (i, &val) in res_f32.values[240..].iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "[{}] Found unexpected f32 NaN at out-index {}",
                    test_name,
                    240 + i
                );
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use paste::paste;

    macro_rules! generate_all_alma_tests {
        ($($test_fn:ident),*) => {
            paste! {
                $(
                    #[test]
                    fn [<$test_fn _scalar_f64>]() {
                        let _ = $test_fn(
                            stringify!([<$test_fn _scalar_f64>]),
                            Kernel::Scalar,
                            false
                        );
                    }
                    #[test]
                    fn [<$test_fn _scalar_f32>]() {
                        let _ = $test_fn(
                            stringify!([<$test_fn _scalar_f32>]),
                            Kernel::Scalar,
                            true
                        );
                    }
                    #[test]
                    fn [<$test_fn _avx2_f64>]() {
                        let _ = $test_fn(
                            stringify!([<$test_fn _avx2_f64>]),
                            Kernel::Avx2,
                            false
                        );
                    }
                    #[test]
                    fn [<$test_fn _avx2_f32>]() {
                        let _ = $test_fn(
                            stringify!([<$test_fn _avx2_f32>]),
                            Kernel::Avx2,
                            true
                        );
                    }
                    #[test]
                    fn [<$test_fn _avx512_f64>]() {
                        let _ = $test_fn(
                            stringify!([<$test_fn _avx512_f64>]),
                            Kernel::Avx512,
                            false
                        );
                    }
                    #[test]
                    fn [<$test_fn _avx512_f32>]() {
                        let _ = $test_fn(
                            stringify!([<$test_fn _avx512_f32>]),
                            Kernel::Avx512,
                            true
                        );
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
