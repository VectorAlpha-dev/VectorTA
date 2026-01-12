#[cfg(all(feature = "python", feature = "cuda"))]
use numpy::PyUntypedArrayMethods;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use js_sys;
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use serde::{Deserialize, Serialize};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen::prelude::*;

#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::{cuda_available, CudaAroon};
use crate::utilities::data_loader::{source_type, Candles};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::utilities::dlpack_cuda::export_f32_cuda_dlpack_2d;
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
use std::collections::VecDeque;
use std::convert::AsRef;
use std::mem::{ManuallyDrop, MaybeUninit};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum AroonData<'a> {
    Candles { candles: &'a Candles },
    SlicesHL { high: &'a [f64], low: &'a [f64] },
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct AroonParams {
    pub length: Option<usize>,
}

impl Default for AroonParams {
    fn default() -> Self {
        Self { length: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct AroonInput<'a> {
    pub data: AroonData<'a>,
    pub params: AroonParams,
}

impl<'a> AroonInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, p: AroonParams) -> Self {
        Self {
            data: AroonData::Candles { candles: c },
            params: p,
        }
    }
    #[inline]
    pub fn from_slices_hl(high: &'a [f64], low: &'a [f64], p: AroonParams) -> Self {
        Self {
            data: AroonData::SlicesHL { high, low },
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, AroonParams::default())
    }
    #[inline]
    pub fn get_length(&self) -> usize {
        self.params.length.unwrap_or(14)
    }
}

impl<'a> AsRef<[f64]> for AroonInput<'a> {
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            AroonData::Candles { candles } => &candles.high,
            AroonData::SlicesHL { high, .. } => high,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AroonOutput {
    pub aroon_up: Vec<f64>,
    pub aroon_down: Vec<f64>,
}

#[derive(Copy, Clone, Debug)]
pub struct AroonBuilder {
    length: Option<usize>,
    kernel: Kernel,
}

impl Default for AroonBuilder {
    fn default() -> Self {
        Self {
            length: None,
            kernel: Kernel::Auto,
        }
    }
}
impl AroonBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn length(mut self, n: usize) -> Self {
        self.length = Some(n);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<AroonOutput, AroonError> {
        let p = AroonParams {
            length: self.length,
        };
        let i = AroonInput::from_candles(c, p);
        aroon_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<AroonOutput, AroonError> {
        let p = AroonParams {
            length: self.length,
        };
        let i = AroonInput::from_slices_hl(high, low, p);
        aroon_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<AroonStream, AroonError> {
        let p = AroonParams {
            length: self.length,
        };
        AroonStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum AroonError {
    #[error("aroon: All values are NaN.")]
    AllValuesNaN,
    #[error("aroon: Input data slice is empty.")]
    EmptyInputData,
    #[error("aroon: Invalid length: length = {length}, data length = {data_len}")]
    InvalidLength { length: usize, data_len: usize },
    #[error("aroon: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("aroon: Mismatch in high/low slice length: high_len={high_len}, low_len={low_len}")]
    MismatchSliceLength { high_len: usize, low_len: usize },
    #[error("aroon: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("aroon: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("aroon: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[inline(always)]
fn first_valid_pair(high: &[f64], low: &[f64]) -> Option<usize> {
    high.iter()
        .zip(low.iter())
        .position(|(h, l)| h.is_finite() && l.is_finite())
}

#[inline]
pub fn aroon(input: &AroonInput) -> Result<AroonOutput, AroonError> {
    aroon_with_kernel(input, Kernel::Auto)
}

pub fn aroon_with_kernel(input: &AroonInput, kernel: Kernel) -> Result<AroonOutput, AroonError> {
    let (high, low): (&[f64], &[f64]) = match &input.data {
        AroonData::Candles { candles } => {
            (source_type(candles, "high"), source_type(candles, "low"))
        }
        AroonData::SlicesHL { high, low } => (*high, *low),
    };
    if high.is_empty() || low.is_empty() {
        return Err(AroonError::EmptyInputData);
    }
    if high.len() != low.len() {
        return Err(AroonError::MismatchSliceLength {
            high_len: high.len(),
            low_len: low.len(),
        });
    }
    let len = high.len();
    let length = input.get_length();

    if length == 0 || length > len {
        return Err(AroonError::InvalidLength {
            length,
            data_len: len,
        });
    }
    if len < length {
        return Err(AroonError::NotEnoughValidData {
            needed: length,
            valid: len,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    let first = first_valid_pair(high, low).ok_or(AroonError::AllValuesNaN)?;
    let warmup_period = first + length;
    let mut up = alloc_with_nan_prefix(len, warmup_period);
    let mut down = alloc_with_nan_prefix(len, warmup_period);

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                aroon_scalar(high, low, length, &mut up, &mut down)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => aroon_avx2(high, low, length, &mut up, &mut down),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                aroon_avx512(high, low, length, &mut up, &mut down)
            }
            _ => unreachable!(),
        }
    }

    let warm = warmup_period.min(len);
    for v in &mut up[..warm] {
        *v = f64::NAN;
    }
    for v in &mut down[..warm] {
        *v = f64::NAN;
    }

    Ok(AroonOutput {
        aroon_up: up,
        aroon_down: down,
    })
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn aroon_into(
    input: &AroonInput,
    out_up: &mut [f64],
    out_down: &mut [f64],
) -> Result<(), AroonError> {
    aroon_into_slice(out_up, out_down, input, Kernel::Auto)
}

#[inline]
pub fn aroon_scalar(high: &[f64], low: &[f64], length: usize, up: &mut [f64], down: &mut [f64]) {
    let len = high.len();
    assert!(
        length >= 1 && length <= len,
        "Invalid length: {} for data of size {}",
        length,
        len
    );
    assert!(
        low.len() == len && up.len() == len && down.len() == len,
        "Slice lengths must match"
    );

    let scale_100 = 100.0 / (length as f64);

    #[inline(always)]
    fn pair_is_finite(h: f64, l: f64) -> bool {
        const EXP_MASK: u64 = 0x7ff0_0000_0000_0000;
        (h.to_bits() & EXP_MASK) != EXP_MASK && (l.to_bits() & EXP_MASK) != EXP_MASK
    }

    #[inline(always)]
    fn aroon_percent(dist: usize, scale_100: f64) -> f64 {
        let v = 100.0 - (dist as f64) * scale_100;
        v.max(0.0)
    }

    let mut all_finite = true;
    unsafe {
        let hp = high.as_ptr();
        let lp = low.as_ptr();
        let mut i = 0usize;
        while i < len {
            if !pair_is_finite(*hp.add(i), *lp.add(i)) {
                all_finite = false;
                break;
            }
            i += 1;
        }
    }

    if all_finite {
        unsafe {
            let hp = high.as_ptr();
            let lp = low.as_ptr();
            let up_ptr = up.as_mut_ptr();
            let dn_ptr = down.as_mut_ptr();

            if length < len {
                let i0 = length;
                let mut maxi = 0usize;
                let mut mini = 0usize;
                let mut max = *hp;
                let mut min = *lp;

                let mut j = 1usize;
                while j <= i0 {
                    let hv = *hp.add(j);
                    if hv > max {
                        max = hv;
                        maxi = j;
                    }
                    let lv = *lp.add(j);
                    if lv < min {
                        min = lv;
                        mini = j;
                    }
                    j += 1;
                }

                *up_ptr.add(i0) = aroon_percent(i0 - maxi, scale_100);
                *dn_ptr.add(i0) = aroon_percent(i0 - mini, scale_100);

                let mut i = i0 + 1;
                while i < len {
                    let start = i - length;
                    let h = *hp.add(i);
                    let l = *lp.add(i);

                    if maxi < start {
                        maxi = start;
                        max = *hp.add(maxi);
                        let mut j = start + 1;
                        while j <= i {
                            let hv = *hp.add(j);
                            if hv > max {
                                max = hv;
                                maxi = j;
                            }
                            j += 1;
                        }
                    } else if h > max {
                        maxi = i;
                        max = h;
                    }

                    if mini < start {
                        mini = start;
                        min = *lp.add(mini);
                        let mut j = start + 1;
                        while j <= i {
                            let lv = *lp.add(j);
                            if lv < min {
                                min = lv;
                                mini = j;
                            }
                            j += 1;
                        }
                    } else if l < min {
                        mini = i;
                        min = l;
                    }

                    *up_ptr.add(i) = aroon_percent(i - maxi, scale_100);
                    *dn_ptr.add(i) = aroon_percent(i - mini, scale_100);

                    i += 1;
                }
            }
        }

        return;
    }

    let window = length + 1;
    let mut invalid_count: usize = 0;

    let mut have_extremes = false;
    let mut maxi: usize = 0;
    let mut mini: usize = 0;
    let mut max = 0.0f64;
    let mut min = 0.0f64;

    for i in 0..len {
        let h = high[i];
        let l = low[i];
        if !pair_is_finite(h, l) {
            invalid_count += 1;
        }
        if i >= window {
            let leave = i - window;
            if !pair_is_finite(high[leave], low[leave]) {
                invalid_count -= 1;
            }
        }

        if i < length {
            continue;
        }

        if invalid_count != 0 {
            up[i] = f64::NAN;
            down[i] = f64::NAN;
            have_extremes = false;
            continue;
        }

        let start = i - length;

        if !have_extremes {
            maxi = start;
            mini = start;
            max = high[start];
            min = low[start];
            for j in (start + 1)..=i {
                let hv = high[j];
                if hv > max {
                    max = hv;
                    maxi = j;
                }
                let lv = low[j];
                if lv < min {
                    min = lv;
                    mini = j;
                }
            }
            have_extremes = true;
        } else {
            if maxi < start {
                maxi = start;
                max = high[maxi];
                for j in (start + 1)..=i {
                    let hv = high[j];
                    if hv > max {
                        max = hv;
                        maxi = j;
                    }
                }
            } else if h > max {
                maxi = i;
                max = h;
            }

            if mini < start {
                mini = start;
                min = low[mini];
                for j in (start + 1)..=i {
                    let lv = low[j];
                    if lv < min {
                        min = lv;
                        mini = j;
                    }
                }
            } else if l < min {
                mini = i;
                min = l;
            }
        }

        let dist_hi = i - maxi;
        let dist_lo = i - mini;
        up[i] = aroon_percent(dist_hi, scale_100);
        down[i] = aroon_percent(dist_lo, scale_100);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn aroon_avx512(high: &[f64], low: &[f64], length: usize, up: &mut [f64], down: &mut [f64]) {
    unsafe {
        use core::arch::x86_64::*;

        let len = high.len();
        debug_assert_eq!(low.len(), len);
        debug_assert_eq!(up.len(), len);
        debug_assert_eq!(down.len(), len);
        if length == 0 || length > len {
            return;
        }

        let hi_ptr = high.as_ptr();
        let lo_ptr = low.as_ptr();
        let up_ptr = up.as_mut_ptr();
        let dn_ptr = down.as_mut_ptr();

        let scale = 100.0 / (length as f64);
        let window = length + 1;

        let sign_mask = _mm512_set1_pd(-0.0);
        let max_finite = _mm512_set1_pd(f64::MAX);

        #[inline(always)]
        unsafe fn lanes_all_finite_512(
            h: __m512d,
            l: __m512d,
            sign_mask: __m512d,
            max_finite: __m512d,
        ) -> bool {
            let h_abs = _mm512_andnot_pd(sign_mask, h);
            let l_abs = _mm512_andnot_pd(sign_mask, l);
            let ok_h: __mmask8 = _mm512_cmp_pd_mask(h_abs, max_finite, _CMP_LE_OQ);
            let ok_l: __mmask8 = _mm512_cmp_pd_mask(l_abs, max_finite, _CMP_LE_OQ);
            (ok_h & ok_l) == 0xFF
        }

        for i in length..len {
            let start = i - length;
            let base_h = hi_ptr.add(start);
            let base_l = lo_ptr.add(start);

            let mut best_h = core::f64::NEG_INFINITY;
            let mut best_l = core::f64::INFINITY;
            let mut best_h_off = 0usize;
            let mut best_l_off = 0usize;

            let mut j = 0usize;
            let mut invalid = false;

            while j + 8 <= window {
                let h8 = _mm512_loadu_pd(base_h.add(j));
                let l8 = _mm512_loadu_pd(base_l.add(j));

                if !lanes_all_finite_512(h8, l8, sign_mask, max_finite) {
                    invalid = true;
                    break;
                }

                let mut hv = [0.0f64; 8];
                let mut lv = [0.0f64; 8];
                _mm512_storeu_pd(hv.as_mut_ptr(), h8);
                _mm512_storeu_pd(lv.as_mut_ptr(), l8);

                if hv[0] > best_h {
                    best_h = hv[0];
                    best_h_off = j;
                }
                if lv[0] < best_l {
                    best_l = lv[0];
                    best_l_off = j;
                }
                if hv[1] > best_h {
                    best_h = hv[1];
                    best_h_off = j + 1;
                }
                if lv[1] < best_l {
                    best_l = lv[1];
                    best_l_off = j + 1;
                }
                if hv[2] > best_h {
                    best_h = hv[2];
                    best_h_off = j + 2;
                }
                if lv[2] < best_l {
                    best_l = lv[2];
                    best_l_off = j + 2;
                }
                if hv[3] > best_h {
                    best_h = hv[3];
                    best_h_off = j + 3;
                }
                if lv[3] < best_l {
                    best_l = lv[3];
                    best_l_off = j + 3;
                }
                if hv[4] > best_h {
                    best_h = hv[4];
                    best_h_off = j + 4;
                }
                if lv[4] < best_l {
                    best_l = lv[4];
                    best_l_off = j + 4;
                }
                if hv[5] > best_h {
                    best_h = hv[5];
                    best_h_off = j + 5;
                }
                if lv[5] < best_l {
                    best_l = lv[5];
                    best_l_off = j + 5;
                }
                if hv[6] > best_h {
                    best_h = hv[6];
                    best_h_off = j + 6;
                }
                if lv[6] < best_l {
                    best_l = lv[6];
                    best_l_off = j + 6;
                }
                if hv[7] > best_h {
                    best_h = hv[7];
                    best_h_off = j + 7;
                }
                if lv[7] < best_l {
                    best_l = lv[7];
                    best_l_off = j + 7;
                }

                j += 8;
            }

            if !invalid {
                while j < window {
                    let h = *base_h.add(j);
                    let l = *base_l.add(j);
                    const EXP_MASK: u64 = 0x7ff0_0000_0000_0000;
                    let hb = h.to_bits();
                    let lb = l.to_bits();
                    if (hb & EXP_MASK) == EXP_MASK || (lb & EXP_MASK) == EXP_MASK {
                        invalid = true;
                        break;
                    }
                    if h > best_h {
                        best_h = h;
                        best_h_off = j;
                    }
                    if l < best_l {
                        best_l = l;
                        best_l_off = j;
                    }
                    j += 1;
                }
            }

            if invalid {
                *up_ptr.add(i) = f64::NAN;
                *dn_ptr.add(i) = f64::NAN;
            } else {
                let dist_hi = length - best_h_off;
                let dist_lo = length - best_l_off;
                let up_val = (-(dist_hi as f64)).mul_add(scale, 100.0);
                let dn_val = (-(dist_lo as f64)).mul_add(scale, 100.0);
                *up_ptr.add(i) = if dist_hi == 0 {
                    100.0
                } else if dist_hi >= length {
                    0.0
                } else {
                    up_val
                };
                *dn_ptr.add(i) = if dist_lo == 0 {
                    100.0
                } else if dist_lo >= length {
                    0.0
                } else {
                    dn_val
                };
            }
        }
    }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn aroon_avx2(high: &[f64], low: &[f64], length: usize, up: &mut [f64], down: &mut [f64]) {
    unsafe {
        use core::arch::x86_64::*;

        let len = high.len();
        debug_assert_eq!(low.len(), len);
        debug_assert_eq!(up.len(), len);
        debug_assert_eq!(down.len(), len);
        if length == 0 || length > len {
            return;
        }

        let hi_ptr = high.as_ptr();
        let lo_ptr = low.as_ptr();
        let up_ptr = up.as_mut_ptr();
        let dn_ptr = down.as_mut_ptr();

        let scale = 100.0 / (length as f64);
        let window = length + 1;

        let sign_mask = _mm256_set1_pd(-0.0);
        let max_finite = _mm256_set1_pd(f64::MAX);

        #[inline(always)]
        unsafe fn lanes_all_finite(
            h: __m256d,
            l: __m256d,
            sign_mask: __m256d,
            max_finite: __m256d,
        ) -> bool {
            let h_abs = _mm256_andnot_pd(sign_mask, h);
            let l_abs = _mm256_andnot_pd(sign_mask, l);
            let ok_h = _mm256_cmp_pd(h_abs, max_finite, _CMP_LE_OQ);
            let ok_l = _mm256_cmp_pd(l_abs, max_finite, _CMP_LE_OQ);
            let ok = _mm256_and_pd(ok_h, ok_l);
            _mm256_movemask_pd(ok) == 0b1111
        }

        for i in length..len {
            let start = i - length;
            let base_h = hi_ptr.add(start);
            let base_l = lo_ptr.add(start);

            let mut best_h = core::f64::NEG_INFINITY;
            let mut best_l = core::f64::INFINITY;
            let mut best_h_off = 0usize;
            let mut best_l_off = 0usize;

            let mut j = 0usize;
            let mut invalid = false;

            while j + 4 <= window {
                let h4 = _mm256_loadu_pd(base_h.add(j));
                let l4 = _mm256_loadu_pd(base_l.add(j));

                if !lanes_all_finite(h4, l4, sign_mask, max_finite) {
                    invalid = true;
                    break;
                }

                let mut hv = [0.0f64; 4];
                let mut lv = [0.0f64; 4];
                _mm256_storeu_pd(hv.as_mut_ptr(), h4);
                _mm256_storeu_pd(lv.as_mut_ptr(), l4);

                if hv[0] > best_h {
                    best_h = hv[0];
                    best_h_off = j;
                }
                if lv[0] < best_l {
                    best_l = lv[0];
                    best_l_off = j;
                }
                if hv[1] > best_h {
                    best_h = hv[1];
                    best_h_off = j + 1;
                }
                if lv[1] < best_l {
                    best_l = lv[1];
                    best_l_off = j + 1;
                }
                if hv[2] > best_h {
                    best_h = hv[2];
                    best_h_off = j + 2;
                }
                if lv[2] < best_l {
                    best_l = lv[2];
                    best_l_off = j + 2;
                }
                if hv[3] > best_h {
                    best_h = hv[3];
                    best_h_off = j + 3;
                }
                if lv[3] < best_l {
                    best_l = lv[3];
                    best_l_off = j + 3;
                }

                j += 4;
            }

            if !invalid {
                while j < window {
                    let h = *base_h.add(j);
                    let l = *base_l.add(j);
                    const EXP_MASK: u64 = 0x7ff0_0000_0000_0000;
                    let hb = h.to_bits();
                    let lb = l.to_bits();
                    if (hb & EXP_MASK) == EXP_MASK || (lb & EXP_MASK) == EXP_MASK {
                        invalid = true;
                        break;
                    }
                    if h > best_h {
                        best_h = h;
                        best_h_off = j;
                    }
                    if l < best_l {
                        best_l = l;
                        best_l_off = j;
                    }
                    j += 1;
                }
            }

            if invalid {
                *up_ptr.add(i) = f64::NAN;
                *dn_ptr.add(i) = f64::NAN;
            } else {
                let dist_hi = length - best_h_off;
                let dist_lo = length - best_l_off;
                let up_val = (-(dist_hi as f64)).mul_add(scale, 100.0);
                let dn_val = (-(dist_lo as f64)).mul_add(scale, 100.0);
                *up_ptr.add(i) = if dist_hi == 0 {
                    100.0
                } else if dist_hi >= length {
                    0.0
                } else {
                    up_val
                };
                *dn_ptr.add(i) = if dist_lo == 0 {
                    100.0
                } else if dist_lo >= length {
                    0.0
                } else {
                    dn_val
                };
            }
        }
    }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn aroon_avx512_short(
    high: &[f64],
    low: &[f64],
    length: usize,
    up: &mut [f64],
    down: &mut [f64],
) {
    aroon_avx512(high, low, length, up, down)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn aroon_avx512_long(
    high: &[f64],
    low: &[f64],
    length: usize,
    up: &mut [f64],
    down: &mut [f64],
) {
    aroon_avx512(high, low, length, up, down)
}

#[derive(Debug)]
pub struct AroonStream {
    length: usize,
    buf_size: usize,
    head: usize,
    count: usize,
    t: usize,
    scale_100: f64,

    flags: Vec<u8>,
    invalid_count: usize,

    maxq: VecDeque<(f64, usize)>,
    minq: VecDeque<(f64, usize)>,
}

impl AroonStream {
    #[inline]
    pub fn try_new(params: AroonParams) -> Result<Self, AroonError> {
        let length = params.length.unwrap_or(14);
        if length == 0 {
            return Err(AroonError::InvalidLength {
                length: 0,
                data_len: 0,
            });
        }
        let buf_size = length + 1;
        Ok(AroonStream {
            length,
            buf_size,
            head: 0,
            count: 0,
            t: 0,
            scale_100: 100.0 / (length as f64),
            flags: vec![0u8; buf_size],
            invalid_count: 0,
            maxq: VecDeque::with_capacity(buf_size),
            minq: VecDeque::with_capacity(buf_size),
        })
    }

    #[inline(always)]
    fn pct_from_distance(&self, dist: usize) -> f64 {
        if dist == 0 {
            100.0
        } else if dist >= self.length {
            0.0
        } else {
            (-(dist as f64)).mul_add(self.scale_100, 100.0)
        }
    }

    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64) -> Option<(f64, f64)> {
        let i = self.t;

        if self.count == self.buf_size {
            let old = self.flags[self.head] as usize;
            self.invalid_count -= old;
        } else {
            self.count += 1;
        }

        let invalid = !(high.is_finite() && low.is_finite());
        let new_flag = invalid as u8;

        self.flags[self.head] = new_flag;
        self.invalid_count += new_flag as usize;
        self.head += 1;
        if self.head == self.buf_size {
            self.head = 0;
        }

        let earliest = i.saturating_sub(self.length);
        while let Some(&(_, idx)) = self.maxq.front() {
            if idx < earliest {
                self.maxq.pop_front();
            } else {
                break;
            }
        }
        while let Some(&(_, idx)) = self.minq.front() {
            if idx < earliest {
                self.minq.pop_front();
            } else {
                break;
            }
        }

        if !invalid {
            while let Some(&(v, _)) = self.maxq.back() {
                if high > v {
                    self.maxq.pop_back();
                } else {
                    break;
                }
            }
            self.maxq.push_back((high, i));

            while let Some(&(v, _)) = self.minq.back() {
                if low < v {
                    self.minq.pop_back();
                } else {
                    break;
                }
            }
            self.minq.push_back((low, i));
        }

        let out = if self.count == self.buf_size && self.invalid_count == 0 {
            debug_assert!(self.maxq.front().is_some() && self.minq.front().is_some());
            let max_idx = self.maxq.front().unwrap().1;
            let min_idx = self.minq.front().unwrap().1;

            let dist_hi = i - max_idx;
            let dist_lo = i - min_idx;

            let up = self.pct_from_distance(dist_hi);
            let down = self.pct_from_distance(dist_lo);
            Some((up, down))
        } else {
            None
        };

        self.t = i + 1;
        out
    }
}

#[derive(Clone, Debug)]
pub struct AroonBatchRange {
    pub length: (usize, usize, usize),
}
impl Default for AroonBatchRange {
    fn default() -> Self {
        Self {
            length: (14, 263, 1),
        }
    }
}

#[inline(always)]
fn expand_grid_aroon(r: &AroonBatchRange) -> Result<Vec<AroonParams>, AroonError> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, AroonError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        let mut vals = Vec::new();
        if start < end {
            let mut v = start;
            while v <= end {
                vals.push(v);
                match v.checked_add(step) {
                    Some(next) => {
                        if next == v {
                            break;
                        }
                        v = next;
                    }
                    None => break,
                }
            }
        } else {
            let mut v = start;
            loop {
                vals.push(v);
                if v == end {
                    break;
                }
                let next = v.saturating_sub(step);
                if next == v {
                    break;
                }
                v = next;
                if v < end {
                    break;
                }
            }
        }
        if vals.is_empty() {
            return Err(AroonError::InvalidRange { start, end, step });
        }
        Ok(vals)
    }
    let lengths = axis_usize(r.length)?;
    let mut out = Vec::with_capacity(lengths.len());
    for &l in &lengths {
        out.push(AroonParams { length: Some(l) });
    }
    Ok(out)
}

#[derive(Clone, Debug, Default)]
pub struct AroonBatchBuilder {
    range: AroonBatchRange,
    kernel: Kernel,
}
impl AroonBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline]
    pub fn length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.length = (start, end, step);
        self
    }
    #[inline]
    pub fn length_static(mut self, x: usize) -> Self {
        self.range.length = (x, x, 0);
        self
    }
    pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<AroonBatchOutput, AroonError> {
        aroon_batch_with_kernel(high, low, &self.range, self.kernel)
    }
    pub fn with_default_slices(
        high: &[f64],
        low: &[f64],
        k: Kernel,
    ) -> Result<AroonBatchOutput, AroonError> {
        AroonBatchBuilder::new().kernel(k).apply_slices(high, low)
    }
    pub fn apply_candles(self, c: &Candles) -> Result<AroonBatchOutput, AroonError> {
        self.apply_slices(source_type(c, "high"), source_type(c, "low"))
    }
    pub fn with_default_candles(c: &Candles) -> Result<AroonBatchOutput, AroonError> {
        AroonBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c)
    }
}

pub struct AroonBatchOutput {
    pub up: Vec<f64>,
    pub down: Vec<f64>,
    pub combos: Vec<AroonParams>,
    pub rows: usize,
    pub cols: usize,
}
impl AroonBatchOutput {
    pub fn row_for_params(&self, p: &AroonParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.length.unwrap_or(14) == p.length.unwrap_or(14))
    }
    pub fn up_for(&self, p: &AroonParams) -> Option<&[f64]> {
        self.row_for_params(p).and_then(|row| {
            row.checked_mul(self.cols)
                .and_then(|start| self.up.get(start..start + self.cols))
        })
    }
    pub fn down_for(&self, p: &AroonParams) -> Option<&[f64]> {
        self.row_for_params(p).and_then(|row| {
            row.checked_mul(self.cols)
                .and_then(|start| self.down.get(start..start + self.cols))
        })
    }
}

#[inline(always)]
fn expand_grid(r: &AroonBatchRange) -> Vec<AroonParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let lengths = axis_usize(r.length);
    let mut out = Vec::with_capacity(lengths.len());
    for &l in &lengths {
        out.push(AroonParams { length: Some(l) });
    }
    out
}

pub fn aroon_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    sweep: &AroonBatchRange,
    k: Kernel,
) -> Result<AroonBatchOutput, AroonError> {
    let kernel = match k {
        Kernel::Auto => Kernel::ScalarBatch,
        other if other.is_batch() => other,
        other => return Err(AroonError::InvalidKernelForBatch(other)),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    aroon_batch_par_slice(high, low, sweep, simd)
}

#[inline(always)]
pub fn aroon_batch_slice(
    high: &[f64],
    low: &[f64],
    sweep: &AroonBatchRange,
    kern: Kernel,
) -> Result<AroonBatchOutput, AroonError> {
    aroon_batch_inner(high, low, sweep, kern, false)
}
#[inline(always)]
pub fn aroon_batch_par_slice(
    high: &[f64],
    low: &[f64],
    sweep: &AroonBatchRange,
    kern: Kernel,
) -> Result<AroonBatchOutput, AroonError> {
    aroon_batch_inner(high, low, sweep, kern, true)
}
#[inline(always)]
fn aroon_batch_inner(
    high: &[f64],
    low: &[f64],
    sweep: &AroonBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<AroonBatchOutput, AroonError> {
    let combos = expand_grid_aroon(sweep)?;
    if high.len() != low.len() {
        return Err(AroonError::MismatchSliceLength {
            high_len: high.len(),
            low_len: low.len(),
        });
    }
    let len = high.len();
    let first = first_valid_pair(high, low).ok_or(AroonError::AllValuesNaN)?;
    let max_l = combos.iter().map(|c| c.length.unwrap()).max().unwrap();
    if len.saturating_sub(first) < max_l {
        return Err(AroonError::NotEnoughValidData {
            needed: max_l,
            valid: len.saturating_sub(first),
        });
    }
    let rows = combos.len();
    let cols = len;

    let mut buf_up_mu = make_uninit_matrix(rows, cols);
    let mut buf_down_mu = make_uninit_matrix(rows, cols);

    let warmup_periods: Vec<usize> = combos
        .iter()
        .map(|c| first.saturating_add(c.length.unwrap()))
        .collect();

    init_matrix_prefixes(&mut buf_up_mu, cols, &warmup_periods);
    init_matrix_prefixes(&mut buf_down_mu, cols, &warmup_periods);

    let mut buf_up_guard = ManuallyDrop::new(buf_up_mu);
    let mut buf_down_guard = ManuallyDrop::new(buf_down_mu);
    let up: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(buf_up_guard.as_mut_ptr() as *mut f64, buf_up_guard.len())
    };
    let down: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(
            buf_down_guard.as_mut_ptr() as *mut f64,
            buf_down_guard.len(),
        )
    };

    aroon_batch_inner_into(high, low, sweep, kern, parallel, up, down)?;

    for (row, &warmup) in warmup_periods.iter().enumerate() {
        let row_start = row * cols;
        let warm_end = (row_start + warmup).min(row_start + cols);
        for i in row_start..warm_end {
            up[i] = f64::NAN;
            down[i] = f64::NAN;
        }
    }

    let up_values = unsafe {
        Vec::from_raw_parts(
            buf_up_guard.as_mut_ptr() as *mut f64,
            buf_up_guard.len(),
            buf_up_guard.capacity(),
        )
    };
    let down_values = unsafe {
        Vec::from_raw_parts(
            buf_down_guard.as_mut_ptr() as *mut f64,
            buf_down_guard.len(),
            buf_down_guard.capacity(),
        )
    };

    Ok(AroonBatchOutput {
        up: up_values,
        down: down_values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn aroon_batch_inner_into(
    high: &[f64],
    low: &[f64],
    sweep: &AroonBatchRange,
    kern: Kernel,
    parallel: bool,
    out_up: &mut [f64],
    out_down: &mut [f64],
) -> Result<Vec<AroonParams>, AroonError> {
    let combos = expand_grid_aroon(sweep)?;
    if high.len() != low.len() {
        return Err(AroonError::MismatchSliceLength {
            high_len: high.len(),
            low_len: low.len(),
        });
    }
    let len = high.len();
    let first = first_valid_pair(high, low).ok_or(AroonError::AllValuesNaN)?;
    let max_l = combos.iter().map(|c| c.length.unwrap()).max().unwrap();
    if len.saturating_sub(first) < max_l {
        return Err(AroonError::NotEnoughValidData {
            needed: max_l,
            valid: len.saturating_sub(first),
        });
    }

    let rows = combos.len();
    let cols = len;
    let expected = rows.checked_mul(cols).ok_or(AroonError::InvalidRange {
        start: sweep.length.0,
        end: sweep.length.1,
        step: sweep.length.2,
    })?;

    if out_up.len() != expected || out_down.len() != expected {
        return Err(AroonError::OutputLengthMismatch {
            expected,
            got: out_up.len().max(out_down.len()),
        });
    }

    let first = first_valid_pair(high, low).ok_or(AroonError::AllValuesNaN)?;

    let warmup_periods: Vec<usize> = combos.iter().map(|c| first + c.length.unwrap()).collect();

    for (row, &warmup) in warmup_periods.iter().enumerate() {
        let row_start = row * cols;
        for i in 0..warmup.min(cols) {
            out_up[row_start + i] = f64::NAN;
            out_down[row_start + i] = f64::NAN;
        }
    }

    let do_row = |row: usize, out_up: &mut [f64], out_down: &mut [f64]| unsafe {
        let length = combos[row].length.unwrap();
        match kern {
            Kernel::Scalar => aroon_row_scalar(high, low, length, out_up, out_down),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => aroon_row_avx2(high, low, length, out_up, out_down),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => aroon_row_avx512(high, low, length, out_up, out_down),
            _ => unreachable!(),
        }
    };
    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out_up
                .par_chunks_mut(cols)
                .zip(out_down.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(row, (u, d))| do_row(row, u, d));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, (u, d)) in out_up
                .chunks_mut(cols)
                .zip(out_down.chunks_mut(cols))
                .enumerate()
            {
                do_row(row, u, d);
            }
        }
    } else {
        for (row, (u, d)) in out_up
            .chunks_mut(cols)
            .zip(out_down.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, u, d);
        }
    }

    for (row, &warmup) in warmup_periods.iter().enumerate() {
        let row_start = row * cols;
        let warm_end = (row_start + warmup).min(row_start + cols);
        for i in row_start..warm_end {
            out_up[i] = f64::NAN;
            out_down[i] = f64::NAN;
        }
    }

    Ok(combos)
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "aroon_cuda_batch_dev")]
#[pyo3(signature = (high_f32, low_f32, length_range, device_id=0))]
pub fn aroon_cuda_batch_dev_py<'py>(
    py: Python<'py>,
    high_f32: numpy::PyReadonlyArray1<'py, f32>,
    low_f32: numpy::PyReadonlyArray1<'py, f32>,
    length_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<(AroonDeviceArrayF32Py, AroonDeviceArrayF32Py)> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let h = high_f32.as_slice()?;
    let l = low_f32.as_slice()?;
    let sweep = AroonBatchRange {
        length: length_range,
    };
    let (up_dev, dn_dev, ctx_arc, dev_id) = py.allow_threads(|| {
        let cuda = CudaAroon::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let res = cuda
            .aroon_batch_dev(h, l, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok::<_, PyErr>((
            res.outputs.first,
            res.outputs.second,
            cuda.context_arc_clone(),
            cuda.device_id(),
        ))
    })?;
    Ok((
        AroonDeviceArrayF32Py {
            inner: up_dev,
            _ctx: ctx_arc.clone(),
            device_id: dev_id,
        },
        AroonDeviceArrayF32Py {
            inner: dn_dev,
            _ctx: ctx_arc,
            device_id: dev_id,
        },
    ))
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "aroon_cuda_many_series_one_param_dev")]
#[pyo3(signature = (high_tm_f32, low_tm_f32, length, device_id=0))]
pub fn aroon_cuda_many_series_one_param_dev_py<'py>(
    py: Python<'py>,
    high_tm_f32: numpy::PyReadonlyArray2<'py, f32>,
    low_tm_f32: numpy::PyReadonlyArray2<'py, f32>,
    length: usize,
    device_id: usize,
) -> PyResult<(AroonDeviceArrayF32Py, AroonDeviceArrayF32Py)> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let shape = high_tm_f32.shape();
    if shape.len() != 2 || low_tm_f32.shape() != shape {
        return Err(PyValueError::new_err("expected two matching 2D arrays"));
    }
    let rows = shape[0];
    let cols = shape[1];
    let h = high_tm_f32.as_slice()?;
    let l = low_tm_f32.as_slice()?;
    let (up_dev, dn_dev, ctx_arc, dev_id) = py.allow_threads(|| {
        let cuda = CudaAroon::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let pair = cuda
            .aroon_many_series_one_param_time_major_dev(h, l, cols, rows, length)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok::<_, PyErr>((
            pair.first,
            pair.second,
            cuda.context_arc_clone(),
            cuda.device_id(),
        ))
    })?;
    Ok((
        AroonDeviceArrayF32Py {
            inner: up_dev,
            _ctx: ctx_arc.clone(),
            device_id: dev_id,
        },
        AroonDeviceArrayF32Py {
            inner: dn_dev,
            _ctx: ctx_arc,
            device_id: dev_id,
        },
    ))
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(
    module = "ta_indicators.cuda",
    name = "AroonDeviceArrayF32",
    unsendable
)]
pub struct AroonDeviceArrayF32Py {
    pub(crate) inner: crate::cuda::moving_averages::alma_wrapper::DeviceArrayF32,
    pub(crate) _ctx: std::sync::Arc<cust::context::Context>,
    pub(crate) device_id: u32,
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pymethods]
impl AroonDeviceArrayF32Py {
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
        d.set_item("data", (self.inner.device_ptr() as usize, false))?;

        d.set_item("version", 3)?;
        Ok(d)
    }

    fn __dlpack_device__(&self) -> (i32, i32) {
        (2, self.device_id as i32)
    }

    #[pyo3(signature=(stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__<'py>(
        &mut self,
        py: Python<'py>,
        stream: Option<pyo3::PyObject>,
        max_version: Option<pyo3::PyObject>,
        dl_device: Option<pyo3::PyObject>,
        copy: Option<pyo3::PyObject>,
    ) -> PyResult<PyObject> {
        use cust::memory::DeviceBuffer;
        use pyo3::types::PyAny;
        use pyo3::Bound;

        let (dev_ty, alloc_dev) = self.__dlpack_device__();
        if let Some(dev_obj) = dl_device.as_ref() {
            if let Ok((want_ty, want_dev)) = dev_obj.extract::<(i32, i32)>(py) {
                if want_ty != dev_ty || want_dev != alloc_dev {
                    let wants_copy = copy
                        .as_ref()
                        .and_then(|c| c.extract::<bool>(py).ok())
                        .unwrap_or(false);
                    if wants_copy {
                        return Err(PyValueError::new_err(
                            "device copy not implemented for __dlpack__",
                        ));
                    } else {
                        return Err(PyValueError::new_err("dl_device mismatch for __dlpack__"));
                    }
                }
            }
        }
        let _ = stream;

        let dummy =
            DeviceBuffer::from_slice(&[]).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let inner = std::mem::replace(
            &mut self.inner,
            crate::cuda::moving_averages::alma_wrapper::DeviceArrayF32 {
                buf: dummy,
                rows: 0,
                cols: 0,
            },
        );
        let rows = inner.rows;
        let cols = inner.cols;
        let buf = inner.buf;

        let max_version_bound: Option<Bound<'py, PyAny>> =
            max_version.map(|obj| obj.into_bound(py));

        export_f32_cuda_dlpack_2d(py, buf, rows, cols, alloc_dev, max_version_bound)
    }
}

#[inline(always)]
pub unsafe fn aroon_row_scalar(
    high: &[f64],
    low: &[f64],
    length: usize,
    out_up: &mut [f64],
    out_down: &mut [f64],
) {
    aroon_scalar(high, low, length, out_up, out_down)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn aroon_row_avx2(
    high: &[f64],
    low: &[f64],
    length: usize,
    out_up: &mut [f64],
    out_down: &mut [f64],
) {
    aroon_avx2(high, low, length, out_up, out_down)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn aroon_row_avx512(
    high: &[f64],
    low: &[f64],
    length: usize,
    out_up: &mut [f64],
    out_down: &mut [f64],
) {
    if length <= 32 {
        aroon_row_avx512_short(high, low, length, out_up, out_down)
    } else {
        aroon_row_avx512_long(high, low, length, out_up, out_down)
    }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn aroon_row_avx512_short(
    high: &[f64],
    low: &[f64],
    length: usize,
    out_up: &mut [f64],
    out_down: &mut [f64],
) {
    aroon_avx512(high, low, length, out_up, out_down)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn aroon_row_avx512_long(
    high: &[f64],
    low: &[f64],
    length: usize,
    out_up: &mut [f64],
    out_down: &mut [f64],
) {
    aroon_avx512(high, low, length, out_up, out_down)
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::utilities::enums::Kernel;

    fn check_aroon_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let partial_params = AroonParams { length: None };
        let input = AroonInput::from_candles(&candles, partial_params);
        let result = aroon_with_kernel(&input, kernel)?;
        assert_eq!(result.aroon_up.len(), candles.close.len());
        assert_eq!(result.aroon_down.len(), candles.close.len());
        Ok(())
    }

    fn check_aroon_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = AroonInput::with_default_candles(&candles);
        let result = aroon_with_kernel(&input, kernel)?;

        let expected_up_last_five = [21.43, 14.29, 7.14, 0.0, 0.0];
        let expected_down_last_five = [71.43, 64.29, 57.14, 50.0, 42.86];

        assert!(
            result.aroon_up.len() >= 5 && result.aroon_down.len() >= 5,
            "Not enough Aroon values"
        );

        let start_index = result.aroon_up.len().saturating_sub(5);

        let up_last_five = &result.aroon_up[start_index..];
        let down_last_five = &result.aroon_down[start_index..];

        for (i, &value) in up_last_five.iter().enumerate() {
            assert!(
                (value - expected_up_last_five[i]).abs() < 1e-2,
                "Aroon Up mismatch at index {}: expected {}, got {}",
                i,
                expected_up_last_five[i],
                value
            );
        }

        for (i, &value) in down_last_five.iter().enumerate() {
            assert!(
                (value - expected_down_last_five[i]).abs() < 1e-2,
                "Aroon Down mismatch at index {}: expected {}, got {}",
                i,
                expected_down_last_five[i],
                value
            );
        }

        Ok(())
    }

    fn check_aroon_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = AroonInput::with_default_candles(&candles);
        match input.data {
            AroonData::Candles { .. } => {}
            _ => panic!("Expected AroonData::Candles variant"),
        }
        let result = aroon_with_kernel(&input, kernel)?;
        assert_eq!(result.aroon_up.len(), candles.close.len());
        assert_eq!(result.aroon_down.len(), candles.close.len());
        Ok(())
    }

    fn check_aroon_zero_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 11.0, 12.0];
        let low = [9.0, 10.0, 11.0];
        let params = AroonParams { length: Some(0) };
        let input = AroonInput::from_slices_hl(&high, &low, params);
        let result = aroon_with_kernel(&input, kernel);
        assert!(result.is_err(), "Expected error for zero length");
        Ok(())
    }

    fn check_aroon_length_exceeds_data(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 11.0, 12.0];
        let low = [9.0, 10.0, 11.0];
        let params = AroonParams { length: Some(14) };
        let input = AroonInput::from_slices_hl(&high, &low, params);
        let result = aroon_with_kernel(&input, kernel);
        assert!(result.is_err(), "Expected error for length > data.len()");
        Ok(())
    }

    fn check_aroon_very_small_data_set(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [100.0];
        let low = [99.5];
        let params = AroonParams { length: Some(14) };
        let input = AroonInput::from_slices_hl(&high, &low, params);
        let result = aroon_with_kernel(&input, kernel);
        assert!(
            result.is_err(),
            "Expected error for data smaller than length"
        );
        Ok(())
    }

    fn check_aroon_reinput(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = AroonParams { length: Some(14) };
        let first_input = AroonInput::from_candles(&candles, first_params);
        let first_result = aroon_with_kernel(&first_input, kernel)?;
        assert_eq!(first_result.aroon_up.len(), candles.close.len());
        assert_eq!(first_result.aroon_down.len(), candles.close.len());
        let second_params = AroonParams { length: Some(5) };
        let second_input = AroonInput::from_slices_hl(&candles.high, &candles.low, second_params);
        let second_result = aroon_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.aroon_up.len(), candles.close.len());
        assert_eq!(second_result.aroon_down.len(), candles.close.len());
        Ok(())
    }

    fn check_aroon_nan_handling(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = AroonParams { length: Some(14) };
        let input = AroonInput::from_candles(&candles, params);
        let result = aroon_with_kernel(&input, kernel)?;
        assert_eq!(result.aroon_up.len(), candles.close.len());
        assert_eq!(result.aroon_down.len(), candles.close.len());
        if result.aroon_up.len() > 240 {
            for i in 240..result.aroon_up.len() {
                assert!(
                    !result.aroon_up[i].is_nan(),
                    "Found NaN in aroon_up at {}",
                    i
                );
                assert!(
                    !result.aroon_down[i].is_nan(),
                    "Found NaN in aroon_down at {}",
                    i
                );
            }
        }
        Ok(())
    }

    fn check_aroon_streaming(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let length = 14;

        let input = AroonInput::from_candles(
            &candles,
            AroonParams {
                length: Some(length),
            },
        );
        let batch_output = aroon_with_kernel(&input, kernel)?;

        let mut stream = AroonStream::try_new(AroonParams {
            length: Some(length),
        })?;
        let mut stream_up = Vec::with_capacity(candles.close.len());
        let mut stream_down = Vec::with_capacity(candles.close.len());
        for (&h, &l) in candles.high.iter().zip(&candles.low) {
            match stream.update(h, l) {
                Some((up, down)) => {
                    stream_up.push(up);
                    stream_down.push(down);
                }
                None => {
                    stream_up.push(f64::NAN);
                    stream_down.push(f64::NAN);
                }
            }
        }
        assert_eq!(batch_output.aroon_up.len(), stream_up.len());
        assert_eq!(batch_output.aroon_down.len(), stream_down.len());
        for (i, (&b, &s)) in batch_output.aroon_up.iter().zip(&stream_up).enumerate() {
            if b.is_nan() && s.is_nan() {
                continue;
            }
            let diff = (b - s).abs();
            assert!(
                diff < 1e-8,
                "[{}] Aroon streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        for (i, (&b, &s)) in batch_output.aroon_down.iter().zip(&stream_down).enumerate() {
            if b.is_nan() && s.is_nan() {
                continue;
            }
            let diff = (b - s).abs();
            assert!(
                diff < 1e-8,
                "[{}] Aroon streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_aroon_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let test_params = vec![
            AroonParams::default(),
            AroonParams { length: Some(1) },
            AroonParams { length: Some(2) },
            AroonParams { length: Some(5) },
            AroonParams { length: Some(10) },
            AroonParams { length: Some(20) },
            AroonParams { length: Some(50) },
            AroonParams { length: Some(100) },
            AroonParams { length: Some(200) },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = AroonInput::from_candles(&candles, params.clone());
            let output = aroon_with_kernel(&input, kernel)?;

            for (i, &val) in output.aroon_up.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 in aroon_up output with params: length={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.length.unwrap_or(14),
                        param_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 in aroon_up output with params: length={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.length.unwrap_or(14),
                        param_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 in aroon_up output with params: length={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.length.unwrap_or(14),
                        param_idx
                    );
                }
            }

            for (i, &val) in output.aroon_down.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 in aroon_down output with params: length={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.length.unwrap_or(14),
                        param_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 in aroon_down output with params: length={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.length.unwrap_or(14),
                        param_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 in aroon_down output with params: length={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.length.unwrap_or(14),
                        param_idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_aroon_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_aroon_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let strat = (1usize..=100).prop_flat_map(|length| {
            (
                prop::collection::vec(
                    (-1e6f64..1e6f64)
                        .prop_filter("finite", |x| x.is_finite())
                        .prop_flat_map(|base| {
                            (0.0f64..0.3f64).prop_map(move |volatility| {
                                let range = base.abs() * volatility + 0.01;
                                let mid = base;
                                let high = mid + range;
                                let low = mid - range;
                                (high, low)
                            })
                        }),
                    length..400,
                ),
                Just(length),
            )
        });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(bars, length)| {
                let (highs, lows): (Vec<f64>, Vec<f64>) = bars.into_iter().unzip();

                let params = AroonParams {
                    length: Some(length),
                };
                let input = AroonInput::from_slices_hl(&highs, &lows, params.clone());

                let AroonOutput {
                    aroon_up: out_up,
                    aroon_down: out_down,
                } = aroon_with_kernel(&input, kernel).unwrap();
                let AroonOutput {
                    aroon_up: ref_up,
                    aroon_down: ref_down,
                } = aroon_with_kernel(&input, Kernel::Scalar).unwrap();

                prop_assert_eq!(out_up.len(), highs.len());
                prop_assert_eq!(out_down.len(), lows.len());

                for i in 0..length.min(out_up.len()) {
                    prop_assert!(out_up[i].is_nan());
                    prop_assert!(out_down[i].is_nan());
                }

                for i in length..out_up.len() {
                    prop_assert!(!out_up[i].is_nan());
                    prop_assert!(!out_down[i].is_nan());
                }

                for i in length..out_up.len() {
                    prop_assert!(
                        out_up[i] >= 0.0 && out_up[i] <= 100.0,
                        "Aroon up at {} = {}, outside [0,100]",
                        i,
                        out_up[i]
                    );
                    prop_assert!(
                        out_down[i] >= 0.0 && out_down[i] <= 100.0,
                        "Aroon down at {} = {}, outside [0,100]",
                        i,
                        out_down[i]
                    );
                }

                for i in length..out_up.len().min(length + 5) {
                    let window_start = i - length;
                    let mut max_val = highs[window_start];
                    let mut max_idx = window_start;
                    let mut min_val = lows[window_start];
                    let mut min_idx = window_start;

                    for j in (window_start + 1)..=i {
                        if highs[j] > max_val {
                            max_val = highs[j];
                            max_idx = j;
                        }
                        if lows[j] < min_val {
                            min_val = lows[j];
                            min_idx = j;
                        }
                    }

                    let periods_since_high = i - max_idx;
                    let periods_since_low = i - min_idx;
                    let expected_up =
                        ((length as f64 - periods_since_high as f64) / length as f64) * 100.0;
                    let expected_down =
                        ((length as f64 - periods_since_low as f64) / length as f64) * 100.0;

                    prop_assert!(
                        (out_up[i] - expected_up).abs() < 1e-9,
                        "Formula mismatch for aroon_up at {}: expected {}, got {}",
                        i,
                        expected_up,
                        out_up[i]
                    );
                    prop_assert!(
                        (out_down[i] - expected_down).abs() < 1e-9,
                        "Formula mismatch for aroon_down at {}: expected {}, got {}",
                        i,
                        expected_down,
                        out_down[i]
                    );
                }

                if length == 1 {
                    for i in 1..out_up.len().min(10) {
                        prop_assert!(
                            out_up[i] == 0.0 || out_up[i] == 100.0,
                            "With length=1, aroon_up must be exactly 0 or 100, got {} at {}",
                            out_up[i],
                            i
                        );
                        prop_assert!(
                            out_down[i] == 0.0 || out_down[i] == 100.0,
                            "With length=1, aroon_down must be exactly 0 or 100, got {} at {}",
                            out_down[i],
                            i
                        );

                        if i > 0 && i < highs.len() {
                            if highs[i] > highs[i - 1] {
                                prop_assert_eq!(
                                    out_up[i],
                                    100.0,
                                    "When high[{}]={} > high[{}]={}, aroon_up should be 100",
                                    i,
                                    highs[i],
                                    i - 1,
                                    highs[i - 1]
                                );
                            }

                            if lows[i] < lows[i - 1] {
                                prop_assert_eq!(
                                    out_down[i],
                                    100.0,
                                    "When low[{}]={} < low[{}]={}, aroon_down should be 100",
                                    i,
                                    lows[i],
                                    i - 1,
                                    lows[i - 1]
                                );
                            }
                        }
                    }
                }

                let is_constant = highs.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10)
                    && lows.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10);

                if is_constant && length > 1 {
                    for i in (length * 2).min(out_up.len())..(length * 3).min(out_up.len()) {
                        prop_assert!(
                            out_up[i] <= 100.0 / length as f64 + 1e-9,
                            "With constant prices, aroon_up should approach 0, got {} at {}",
                            out_up[i],
                            i
                        );
                        prop_assert!(
                            out_down[i] <= 100.0 / length as f64 + 1e-9,
                            "With constant prices, aroon_down should approach 0, got {} at {}",
                            out_down[i],
                            i
                        );
                    }
                }

                prop_assert_eq!(out_up.len(), ref_up.len());
                prop_assert_eq!(out_down.len(), ref_down.len());

                for i in 0..out_up.len() {
                    let y_up = out_up[i];
                    let r_up = ref_up[i];
                    let y_down = out_down[i];
                    let r_down = ref_down[i];

                    if !y_up.is_finite() || !r_up.is_finite() {
                        prop_assert_eq!(y_up.to_bits(), r_up.to_bits());
                    } else {
                        let ulp_diff = y_up.to_bits().abs_diff(r_up.to_bits());
                        prop_assert!(
                            (y_up - r_up).abs() <= 1e-9 || ulp_diff <= 4,
                            "Kernel mismatch for aroon_up at {}: {} vs {} (ULP={})",
                            i,
                            y_up,
                            r_up,
                            ulp_diff
                        );
                    }

                    if !y_down.is_finite() || !r_down.is_finite() {
                        prop_assert_eq!(y_down.to_bits(), r_down.to_bits());
                    } else {
                        let ulp_diff = y_down.to_bits().abs_diff(r_down.to_bits());
                        prop_assert!(
                            (y_down - r_down).abs() <= 1e-9 || ulp_diff <= 4,
                            "Kernel mismatch for aroon_down at {}: {} vs {} (ULP={})",
                            i,
                            y_down,
                            r_down,
                            ulp_diff
                        );
                    }
                }

                for i in (length + 10)..(out_up.len().min(length + 15)) {
                    let window_start = i - length;

                    let mut max_idx = window_start;
                    for j in (window_start + 1)..=i {
                        if highs[j] > highs[max_idx] {
                            max_idx = j;
                        }
                    }

                    if i + 1 < out_up.len() && max_idx < i {
                        let next_window_start = i + 1 - length;
                        let mut next_max_idx = next_window_start;
                        for j in (next_window_start + 1)..=i + 1 {
                            if j < highs.len() && highs[j] > highs[next_max_idx] {
                                next_max_idx = j;
                            }
                        }

                        if next_max_idx == max_idx {
                            prop_assert!(
                                out_up[i + 1] <= out_up[i] + 1e-9,
                                "Monotonicity: Aroon up should decrease as extreme ages: {} -> {}",
                                out_up[i],
                                out_up[i + 1]
                            );
                        }
                    }
                }

                for i in 0..highs.len() {
                    prop_assert!(
                        highs[i] >= lows[i],
                        "Data integrity: High {} < Low {} at index {}",
                        highs[i],
                        lows[i],
                        i
                    );
                }

                #[cfg(debug_assertions)]
                {
                    for (i, &val) in out_up.iter().enumerate() {
                        if val.is_finite() {
                            let bits = val.to_bits();
                            prop_assert!(
                                bits != 0x11111111_11111111
                                    && bits != 0x22222222_22222222
                                    && bits != 0x33333333_33333333,
                                "Found poison value {} (0x{:016X}) at {} in aroon_up",
                                val,
                                bits,
                                i
                            );
                        }
                    }
                    for (i, &val) in out_down.iter().enumerate() {
                        if val.is_finite() {
                            let bits = val.to_bits();
                            prop_assert!(
                                bits != 0x11111111_11111111
                                    && bits != 0x22222222_22222222
                                    && bits != 0x33333333_33333333,
                                "Found poison value {} (0x{:016X}) at {} in aroon_down",
                                val,
                                bits,
                                i
                            );
                        }
                    }
                }

                Ok(())
            })
            .unwrap();

        Ok(())
    }

    macro_rules! generate_all_aroon_tests {
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

    generate_all_aroon_tests!(
        check_aroon_partial_params,
        check_aroon_accuracy,
        check_aroon_default_candles,
        check_aroon_zero_length,
        check_aroon_length_exceeds_data,
        check_aroon_very_small_data_set,
        check_aroon_reinput,
        check_aroon_nan_handling,
        check_aroon_streaming,
        check_aroon_no_poison,
        check_aroon_all_nan_error,
        check_aroon_leading_nan_warmup,
        check_aroon_nan_in_window,
        check_aroon_streaming_nan_window
    );

    #[cfg(feature = "proptest")]
    generate_all_aroon_tests!(check_aroon_property);

    fn check_aroon_all_nan_error(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let high = vec![f64::NAN; 20];
        let low = vec![f64::NAN; 20];
        let params = AroonParams { length: Some(5) };
        let input = AroonInput::from_slices_hl(&high, &low, params);

        let result = aroon_with_kernel(&input, kernel);
        assert!(
            matches!(result, Err(AroonError::AllValuesNaN)),
            "Expected AllValuesNaN error, got: {:?}",
            result
        );

        Ok(())
    }

    fn check_aroon_leading_nan_warmup(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let mut high = vec![f64::NAN; 5];
        let mut low = vec![f64::NAN; 5];
        high.extend_from_slice(&[
            100.0, 110.0, 105.0, 115.0, 112.0, 120.0, 118.0, 125.0, 122.0, 130.0,
        ]);
        low.extend_from_slice(&[
            90.0, 95.0, 92.0, 98.0, 96.0, 100.0, 99.0, 105.0, 103.0, 108.0,
        ]);

        let params = AroonParams { length: Some(3) };
        let input = AroonInput::from_slices_hl(&high, &low, params);
        let result = aroon_with_kernel(&input, kernel)?;

        for i in 0..8 {
            assert!(
                result.aroon_up[i].is_nan(),
                "Expected NaN at index {} for aroon_up during warmup, got {}",
                i,
                result.aroon_up[i]
            );
            assert!(
                result.aroon_down[i].is_nan(),
                "Expected NaN at index {} for aroon_down during warmup, got {}",
                i,
                result.aroon_down[i]
            );
        }

        for i in 8..high.len() {
            assert!(
                !result.aroon_up[i].is_nan(),
                "Unexpected NaN at index {} for aroon_up after warmup",
                i
            );
            assert!(
                !result.aroon_down[i].is_nan(),
                "Unexpected NaN at index {} for aroon_down after warmup",
                i
            );
        }

        Ok(())
    }

    fn check_aroon_nan_in_window(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let mut high = vec![
            100.0,
            110.0,
            105.0,
            115.0,
            112.0,
            f64::NAN,
            118.0,
            125.0,
            122.0,
            130.0,
        ];
        let low = vec![
            90.0, 95.0, 92.0, 98.0, 96.0, 100.0, 99.0, 105.0, 103.0, 108.0,
        ];

        let params = AroonParams { length: Some(3) };
        let input = AroonInput::from_slices_hl(&high, &low, params);
        let result = aroon_with_kernel(&input, kernel)?;

        for i in 5..=8 {
            if i < result.aroon_up.len() {
                assert!(
                    result.aroon_up[i].is_nan(),
                    "Expected NaN at index {} for aroon_up when NaN is in window, got {}",
                    i,
                    result.aroon_up[i]
                );
                assert!(
                    result.aroon_down[i].is_nan(),
                    "Expected NaN at index {} for aroon_down when NaN is in window, got {}",
                    i,
                    result.aroon_down[i]
                );
            }
        }

        if result.aroon_up.len() > 9 {
            assert!(
                !result.aroon_up[9].is_nan(),
                "Unexpected NaN at index 9 for aroon_up after NaN exits window"
            );
            assert!(
                !result.aroon_down[9].is_nan(),
                "Unexpected NaN at index 9 for aroon_down after NaN exits window"
            );
        }

        Ok(())
    }

    fn check_aroon_streaming_nan_window(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let mut stream = AroonStream::try_new(AroonParams { length: Some(3) })?;

        assert_eq!(stream.update(100.0, 90.0), None);
        assert_eq!(stream.update(110.0, 95.0), None);
        assert_eq!(stream.update(105.0, 92.0), None);

        let result = stream.update(115.0, 98.0);
        assert!(result.is_some(), "Expected Some result after 4 values");

        let result_with_nan = stream.update(f64::NAN, 100.0);
        assert_eq!(result_with_nan, None, "Expected None when NaN is in window");

        assert_eq!(stream.update(120.0, 105.0), None);
        assert_eq!(stream.update(125.0, 108.0), None);
        assert_eq!(stream.update(130.0, 110.0), None);

        let result_after_nan = stream.update(135.0, 112.0);
        assert!(
            result_after_nan.is_some(),
            "Expected Some result after NaN exits window"
        );

        Ok(())
    }

    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = AroonBatchBuilder::new().kernel(kernel).apply_candles(&c)?;

        let def = AroonParams::default();
        let row = output.up_for(&def).expect("default up row missing");
        assert_eq!(row.len(), c.close.len());

        let expected = [21.43, 14.29, 7.14, 0.0, 0.0];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-2,
                "[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
            );
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let test_configs = vec![
            (1, 10, 1),
            (2, 20, 2),
            (5, 50, 5),
            (10, 100, 10),
            (14, 14, 0),
            (50, 200, 50),
            (1, 5, 1),
            (100, 200, 50),
            (3, 30, 3),
        ];

        for (cfg_idx, &(l_start, l_end, l_step)) in test_configs.iter().enumerate() {
            let output = AroonBatchBuilder::new()
                .kernel(kernel)
                .length_range(l_start, l_end, l_step)
                .apply_candles(&c)?;

            for (idx, &val) in output.up.iter().enumerate() {
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
						 at row {} col {} (flat index {}) in aroon_up output with params: length={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.length.unwrap_or(14)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in aroon_up output with params: length={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.length.unwrap_or(14)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in aroon_up output with params: length={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.length.unwrap_or(14)
                    );
                }
            }

            for (idx, &val) in output.down.iter().enumerate() {
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
						 at row {} col {} (flat index {}) in aroon_down output with params: length={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.length.unwrap_or(14)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in aroon_down output with params: length={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.length.unwrap_or(14)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in aroon_down output with params: length={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.length.unwrap_or(14)
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

    #[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
    #[test]
    fn test_aroon_into_matches_api() -> Result<(), Box<dyn std::error::Error>> {
        let len = 256usize;
        let mut high = Vec::with_capacity(len);
        let mut low = Vec::with_capacity(len);

        for _ in 0..8 {
            high.push(f64::NAN);
            low.push(f64::NAN);
        }

        for i in 8..len {
            let base = 100.0 + (i as f64) * 0.017;
            high.push(base + 0.75 + (i as f64).sin() * 0.01);
            low.push(base - 0.80 + (i as f64).cos() * 0.01);
        }

        let input = AroonInput::from_slices_hl(&high, &low, AroonParams::default());

        let baseline = aroon(&input)?;

        let mut up = vec![0.0; len];
        let mut down = vec![0.0; len];
        aroon_into(&input, &mut up, &mut down)?;

        assert_eq!(baseline.aroon_up.len(), up.len());
        assert_eq!(baseline.aroon_down.len(), down.len());

        #[inline]
        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b)
        }

        for i in 0..len {
            assert!(
                eq_or_both_nan(baseline.aroon_up[i], up[i]),
                "Mismatch at index {} (up): baseline={}, into={}",
                i,
                baseline.aroon_up[i],
                up[i]
            );
            assert!(
                eq_or_both_nan(baseline.aroon_down[i], down[i]),
                "Mismatch at index {} (down): baseline={}, into={}",
                i,
                baseline.aroon_down[i],
                down[i]
            );
        }

        Ok(())
    }
}

#[inline]
pub fn aroon_into_slice(
    dst_up: &mut [f64],
    dst_down: &mut [f64],
    input: &AroonInput,
    kern: Kernel,
) -> Result<(), AroonError> {
    let (high, low): (&[f64], &[f64]) = match &input.data {
        AroonData::Candles { candles } => {
            (source_type(candles, "high"), source_type(candles, "low"))
        }
        AroonData::SlicesHL { high, low } => (*high, *low),
    };
    if high.is_empty() || low.is_empty() {
        return Err(AroonError::EmptyInputData);
    }
    if high.len() != low.len() {
        return Err(AroonError::MismatchSliceLength {
            high_len: high.len(),
            low_len: low.len(),
        });
    }
    let len = high.len();
    let length = input.get_length();
    if length == 0 || length > len {
        return Err(AroonError::InvalidLength {
            length,
            data_len: len,
        });
    }
    if dst_up.len() != len || dst_down.len() != len {
        return Err(AroonError::OutputLengthMismatch {
            expected: len,
            got: dst_up.len().max(dst_down.len()),
        });
    }

    let first = first_valid_pair(high, low).ok_or(AroonError::AllValuesNaN)?;
    let warm = first + length;

    let chosen = match kern {
        Kernel::Auto => Kernel::Scalar,
        k => k,
    };
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                aroon_scalar(high, low, length, dst_up, dst_down)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => aroon_avx2(high, low, length, dst_up, dst_down),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                aroon_avx512(high, low, length, dst_up, dst_down)
            }
            _ => unreachable!(),
        }
    }
    for v in &mut dst_up[..warm.min(len)] {
        *v = f64::NAN;
    }
    for v in &mut dst_down[..warm.min(len)] {
        *v = f64::NAN;
    }
    Ok(())
}

#[cfg(feature = "python")]
#[pyfunction(name = "aroon")]
#[pyo3(signature = (high, low, length, kernel=None))]
pub fn aroon_py<'py>(
    py: Python<'py>,
    high: numpy::PyReadonlyArray1<'py, f64>,
    low: numpy::PyReadonlyArray1<'py, f64>,
    length: usize,
    kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};

    let h = high.as_slice()?;
    let l = low.as_slice()?;
    if h.len() != l.len() {
        return Err(PyValueError::new_err(format!(
            "High/low length mismatch: {} vs {}",
            h.len(),
            l.len()
        )));
    }

    let kern = validate_kernel(kernel, false)?;
    let params = AroonParams {
        length: Some(length),
    };
    let input = AroonInput::from_slices_hl(h, l, params);

    let out = py
        .allow_threads(|| aroon_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok((
        out.aroon_up.into_pyarray(py),
        out.aroon_down.into_pyarray(py),
    ))
}

#[cfg(feature = "python")]
#[pyclass(name = "AroonStream")]
pub struct AroonStreamPy {
    stream: AroonStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl AroonStreamPy {
    #[new]
    fn new(length: usize) -> PyResult<Self> {
        let params = AroonParams {
            length: Some(length),
        };
        let stream =
            AroonStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(AroonStreamPy { stream })
    }

    fn update(&mut self, high: f64, low: f64) -> Option<(f64, f64)> {
        self.stream.update(high, low)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "aroon_batch")]
#[pyo3(signature = (high, low, length_range, kernel=None))]
pub fn aroon_batch_py<'py>(
    py: Python<'py>,
    high: numpy::PyReadonlyArray1<'py, f64>,
    low: numpy::PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let h = high.as_slice()?;
    let l = low.as_slice()?;
    if h.len() != l.len() {
        return Err(PyValueError::new_err(format!(
            "High/low length mismatch: {} vs {}",
            h.len(),
            l.len()
        )));
    }

    let sweep = AroonBatchRange {
        length: length_range,
    };
    let combos = expand_grid_aroon(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = h.len();

    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows * cols overflow"))?;
    let up_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let down_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let up_slice = unsafe { up_arr.as_slice_mut()? };
    let down_slice = unsafe { down_arr.as_slice_mut()? };

    let kern = validate_kernel(kernel, true)?;
    py.allow_threads(|| {
        let batch = match kern {
            Kernel::Auto => detect_best_batch_kernel(),
            k => k,
        };
        let simd = match batch {
            Kernel::Avx512Batch => Kernel::Avx512,
            Kernel::Avx2Batch => Kernel::Avx2,
            Kernel::ScalarBatch => Kernel::Scalar,
            _ => unreachable!(),
        };
        aroon_batch_inner_into(h, l, &sweep, simd, true, up_slice, down_slice)
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("up", up_arr.reshape((rows, cols))?)?;
    dict.set_item("down", down_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "lengths",
        combos
            .iter()
            .map(|p| p.length.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct AroonJsOutput {
    pub values: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "aroon_js")]
pub fn aroon_js(high: &[f64], low: &[f64], length: usize) -> Result<JsValue, JsValue> {
    let params = AroonParams {
        length: Some(length),
    };
    let input = AroonInput::from_slices_hl(high, low, params);

    let mut up = vec![0.0; high.len()];
    let mut down = vec![0.0; high.len()];

    aroon_into_slice(&mut up, &mut down, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("up"),
        &serde_wasm_bindgen::to_value(&up).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("down"),
        &serde_wasm_bindgen::to_value(&down).unwrap(),
    )?;

    Ok(obj.into())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct AroonBatchJsOutput {
    pub values: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
    pub combos: Vec<AroonParams>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "aroon_batch_js")]
pub fn aroon_batch_unified_js(
    high: &[f64],
    low: &[f64],
    length_start: usize,
    length_end: usize,
    length_step: usize,
) -> Result<JsValue, JsValue> {
    let sweep = AroonBatchRange {
        length: (length_start, length_end, length_step),
    };
    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = high.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| JsValue::from_str("rows * cols overflow"))?;

    let mut up = vec![0.0; total];
    let mut down = vec![0.0; total];

    aroon_batch_inner_into(
        high,
        low,
        &sweep,
        detect_best_kernel(),
        false,
        &mut up,
        &mut down,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("up"),
        &serde_wasm_bindgen::to_value(&up).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("down"),
        &serde_wasm_bindgen::to_value(&down).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("rows"),
        &JsValue::from_f64(rows as f64),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("cols"),
        &JsValue::from_f64(cols as f64),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("combos"),
        &serde_wasm_bindgen::to_value(&combos).unwrap(),
    )?;

    Ok(obj.into())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "aroon_batch_metadata_js")]
pub fn aroon_batch_metadata_js(
    length_start: usize,
    length_end: usize,
    length_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = AroonBatchRange {
        length: (length_start, length_end, length_step),
    };

    let combos = expand_grid(&sweep);
    let metadata: Vec<f64> = combos
        .iter()
        .map(|c| c.length.unwrap_or(14) as f64)
        .collect();

    Ok(metadata)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct AroonBatchConfig {
    pub length_range: Vec<usize>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "aroon_batch")]
pub fn aroon_batch_config_js(
    high: &[f64],
    low: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: AroonBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    if config.length_range.len() != 3 {
        return Err(JsValue::from_str(
            "Invalid config: length_range must have exactly 3 elements [start, end, step]",
        ));
    }

    let sweep = AroonBatchRange {
        length: (
            config.length_range[0],
            config.length_range[1],
            config.length_range[2],
        ),
    };

    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = high.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| JsValue::from_str("rows * cols overflow"))?;

    let mut up = vec![0.0; total];
    let mut down = vec![0.0; total];

    aroon_batch_inner_into(
        high,
        low,
        &sweep,
        detect_best_kernel(),
        false,
        &mut up,
        &mut down,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("up"),
        &serde_wasm_bindgen::to_value(&up).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("down"),
        &serde_wasm_bindgen::to_value(&down).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("rows"),
        &JsValue::from_f64(rows as f64),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("cols"),
        &JsValue::from_f64(cols as f64),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("combos"),
        &serde_wasm_bindgen::to_value(&combos).unwrap(),
    )?;

    Ok(obj.into())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn aroon_alloc(len: usize) -> *mut f64 {
    let mut v = Vec::<f64>::with_capacity(2 * len);
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn aroon_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, 2 * len, 2 * len);
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn aroon_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length: usize,
) -> Result<(), JsValue> {
    if high_ptr.is_null() || low_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to aroon_into"));
    }
    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, 2 * len);

        let params = AroonParams {
            length: Some(length),
        };
        let input = AroonInput::from_slices_hl(high, low, params);

        let (up, down) = out.split_at_mut(len);
        aroon_into_slice(up, down, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}
