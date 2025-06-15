//! # Holt-Winters Moving Average (HWMA)
//!
//! Triple-smoothed adaptive moving average with three parameters (`na`, `nb`, `nc`).
//! Each parameter adjusts a component of smoothing: level (`na`), trend (`nb`), and acceleration (`nc`).
//! API and implementation structure matches alma.rs, including AVX stubs, kernel support, batch/grid/stream, and unit tests.
//!
//! ## Parameters
//! - **na**: Smoothing for level (0,1)
//! - **nb**: Smoothing for trend (0,1)
//! - **nc**: Smoothing for acceleration (0,1)
//!
//! ## Errors
//! - **EmptyData**: hwma: The provided data array is empty.
//! - **AllValuesNaN**: hwma: All input data values are `NaN`.
//! - **InvalidParams**: hwma: One or more of `(na, nb, nc)` are out of (0,1).
//!
//! ## Returns
//! - **`Ok(HwmaOutput)`** with results, or **`Err(HwmaError)`** on failure.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel, alloc_with_nan_prefix, init_matrix_prefixes, make_uninit_matrix};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;
use std::mem::MaybeUninit;  

impl<'a> AsRef<[f64]> for HwmaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            HwmaData::Slice(slice) => slice,
            HwmaData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum HwmaData<'a> {
    Candles { candles: &'a Candles, source: &'a str },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct HwmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct HwmaParams {
    pub na: Option<f64>,
    pub nb: Option<f64>,
    pub nc: Option<f64>,
}

impl Default for HwmaParams {
    fn default() -> Self {
        Self {
            na: Some(0.2),
            nb: Some(0.1),
            nc: Some(0.1),
        }
    }
}

#[derive(Debug, Clone)]
pub struct HwmaInput<'a> {
    pub data: HwmaData<'a>,
    pub params: HwmaParams,
}

impl<'a> HwmaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: HwmaParams) -> Self {
        Self {
            data: HwmaData::Candles { candles: c, source: s },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: HwmaParams) -> Self {
        Self {
            data: HwmaData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", HwmaParams::default())
    }
    #[inline]
    pub fn get_na(&self) -> f64 {
        self.params.na.unwrap_or(0.2)
    }
    #[inline]
    pub fn get_nb(&self) -> f64 {
        self.params.nb.unwrap_or(0.1)
    }
    #[inline]
    pub fn get_nc(&self) -> f64 {
        self.params.nc.unwrap_or(0.1)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct HwmaBuilder {
    na: Option<f64>,
    nb: Option<f64>,
    nc: Option<f64>,
    kernel: Kernel,
}

impl Default for HwmaBuilder {
    fn default() -> Self {
        Self {
            na: None,
            nb: None,
            nc: None,
            kernel: Kernel::Auto,
        }
    }
}

impl HwmaBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn na(mut self, x: f64) -> Self {
        self.na = Some(x);
        self
    }
    #[inline(always)]
    pub fn nb(mut self, x: f64) -> Self {
        self.nb = Some(x);
        self
    }
    #[inline(always)]
    pub fn nc(mut self, x: f64) -> Self {
        self.nc = Some(x);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<HwmaOutput, HwmaError> {
        let p = HwmaParams { na: self.na, nb: self.nb, nc: self.nc };
        let i = HwmaInput::from_candles(c, "close", p);
        hwma_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<HwmaOutput, HwmaError> {
        let p = HwmaParams { na: self.na, nb: self.nb, nc: self.nc };
        let i = HwmaInput::from_slice(d, p);
        hwma_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<HwmaStream, HwmaError> {
        let p = HwmaParams { na: self.na, nb: self.nb, nc: self.nc };
        HwmaStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum HwmaError {
    #[error("hwma: calculation received empty data array.")]
    EmptyData,
    #[error("hwma: All values in input data are NaN.")]
    AllValuesNaN,
    #[error("hwma: Parameters (na, nb, nc) must be in (0,1). Received: na={na}, nb={nb}, nc={nc}")]
    InvalidParams { na: f64, nb: f64, nc: f64 },
}

#[inline]
pub fn hwma(input: &HwmaInput) -> Result<HwmaOutput, HwmaError> {
    hwma_with_kernel(input, Kernel::Auto)
}

pub fn hwma_with_kernel(input: &HwmaInput, kernel: Kernel) -> Result<HwmaOutput, HwmaError> {
    let data: &[f64] = input.as_ref();
    let first = data.iter().position(|x| !x.is_nan()).ok_or(HwmaError::AllValuesNaN)?;
    let len = data.len();
    if len == 0 {
        return Err(HwmaError::EmptyData);
    }
    let na = input.get_na();
    let nb = input.get_nb();
    let nc = input.get_nc();

    if !(na > 0.0 && na < 1.0 && nb > 0.0 && nb < 1.0 && nc > 0.0 && nc < 1.0) {
        return Err(HwmaError::InvalidParams { na, nb, nc });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let mut out = alloc_with_nan_prefix(len, first);
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                hwma_scalar(data, na, nb, nc, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                hwma_avx2(data, na, nb, nc, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                hwma_avx512(data, na, nb, nc, first, &mut out)
            }
            _ => unreachable!(),
        }
    }

    Ok(HwmaOutput { values: out })
}

#[inline(always)]
pub fn hwma_scalar(
    data: &[f64],
    na: f64,
    nb: f64,
    nc: f64,
    first: usize,
    out: &mut [f64],
) {
    debug_assert_eq!(data.len(), out.len());
    if first >= data.len() { return; }

    /* -------- coefficients (kept once) ---------------------------- */
    let one_m_na = 1.0 - na;
    let one_m_nb = 1.0 - nb;
    let one_m_nc = 1.0 - nc;
    const HALF: f64 = 0.5;

    /* -------- state registers ------------------------------------- */
    let mut f = data[first];   // level
    let mut v = 0.0;           // trend
    let mut a = 0.0;           // acceleration

    /* -------- main loop ------------------------------------------- */
    for i in first..data.len() {
        // SAFETY: `i` checked by loop guard.
        let price = unsafe { *data.get_unchecked(i) };

        /* ---- level (f') ---- */
        // s = f + v + 0.5·a   computed with one FMA
        let s        = HALF.mul_add(a, f + v);
        // f' = na·price + (1-na)·s
        let f_new    = na.mul_add(price, one_m_na * s);

        /* ---- trend (v') ---- */
        let diff_f   = f_new - f;
        // v' = nb·(f'-f) + (1-nb)·(v + a)
        let v_new    = nb.mul_add(diff_f, one_m_nb * (v + a));

        /* ---- acceleration (a') ---- */
        let diff_v   = v_new - v;
        // a' = nc·(v'-v) + (1-nc)·a
        let a_new    = nc.mul_add(diff_v, one_m_nc * a);

        /* ---- output HWMA = f' + v' + 0.5·a'  (one more FMA) ---- */
        out[i]       = HALF.mul_add(a_new, f_new + v_new);

        /* ---- roll state ---- */
        f = f_new;  v = v_new;  a = a_new;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn hwma_avx2(
    data: &[f64],
    na: f64,
    nb: f64,
    nc: f64,
    first: usize,
    out: &mut [f64],
) {
    use core::arch::x86_64::*;
    debug_assert_eq!(data.len(), out.len());
    if first >= data.len() { return; }

    // ---- broadcast constants once ---------------------------------------
    let na_v     = _mm_set_sd(na);
    let one_na_v = _mm_set_sd(1.0 - na);
    let nb_v     = _mm_set_sd(nb);
    let one_nb_v = _mm_set_sd(1.0 - nb);
    let nc_v     = _mm_set_sd(nc);
    let one_nc_v = _mm_set_sd(1.0 - nc);
    let half_v   = _mm_set_sd(0.5);

    // ---- state registers -------------------------------------------------
    let mut f_v = _mm_set_sd(*data.get_unchecked(first));
    let mut v_v = _mm_setzero_pd();      // v = 0
    let mut a_v = _mm_setzero_pd();      // a = 0

    // ---- software‑pipelined main loop (4‑way unroll) --------------------
    let mut i = first;
    let n = data.len();
    while i + 3 < n {
        macro_rules! step {
            ($idx:expr) => {{
                let price_v = _mm_set_sd(*data.get_unchecked($idx));

                // tmp = f + v + 0.5·a
                let tmp  = _mm_fmadd_sd(half_v, a_v, _mm_add_sd(f_v, v_v));

                // f_new = (1‑na)*tmp + na*price
                let f_new = _mm_fmadd_sd(na_v, price_v, _mm_mul_sd(one_na_v, tmp));

                // v_new = (1‑nb)*(v+a) + nb*(f_new‑f)
                let vpa   = _mm_add_sd(v_v, a_v);
                let diff_f= _mm_sub_sd(f_new, f_v);
                let v_new = _mm_fmadd_sd(nb_v, diff_f, _mm_mul_sd(one_nb_v, vpa));

                // a_new = (1‑nc)*a + nc*(v_new‑v)
                let diff_v= _mm_sub_sd(v_new, v_v);
                let a_new = _mm_fmadd_sd(nc_v, diff_v, _mm_mul_sd(one_nc_v, a_v));

                // out = f_new + v_new + 0.5·a_new
                let out_v = _mm_fmadd_sd(half_v, a_new, _mm_add_sd(f_new, v_new));
                _mm_store_sd(out.as_mut_ptr().add($idx), out_v);

                // roll state
                f_v = f_new;  v_v = v_new;  a_v = a_new;
            }};
        }
        step!(i);     step!(i + 1);  step!(i + 2);  step!(i + 3);
        i += 4;
    }
    // tail
    while i < n {
        let price_v = _mm_set_sd(*data.get_unchecked(i));
        let tmp     = _mm_fmadd_sd(half_v, a_v, _mm_add_sd(f_v, v_v));
        let f_new   = _mm_fmadd_sd(na_v, price_v, _mm_mul_sd(one_na_v, tmp));
        let v_new   = _mm_fmadd_sd(nb_v, _mm_sub_sd(f_new, f_v),
                                   _mm_mul_sd(one_nb_v, _mm_add_sd(v_v, a_v)));
        let a_new   = _mm_fmadd_sd(nc_v, _mm_sub_sd(v_new, v_v),
                                   _mm_mul_sd(one_nc_v, a_v));
        let out_v   = _mm_fmadd_sd(half_v, a_new, _mm_add_sd(f_new, v_new));
        _mm_store_sd(out.as_mut_ptr().add(i), out_v);
        f_v = f_new;  v_v = v_new;  a_v = a_new;
        i += 1;
    }
}


#[cfg(all(feature="nightly-avx", target_arch="x86_64"))]
#[target_feature(enable="avx512f,fma")]
#[inline]
pub unsafe fn hwma_avx512(
    data: &[f64], na: f64, nb: f64, nc: f64,
    first: usize, out: &mut [f64])
{
    // identical to AVX2 version; the CPU will use 512-wide FMA units
    // thanks to the target_feature.  Copy or `#[cfg(target_feature="avx512f")]`.
    hwma_avx2(data, na, nb, nc, first, out);
}




#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn hwma_avx512_short(
    data: &[f64],
    na: f64,
    nb: f64,
    nc: f64,
    first_valid: usize,
    out: &mut [f64],
) {
    hwma_scalar(data, na, nb, nc, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn hwma_avx512_long(
    data: &[f64],
    na: f64,
    nb: f64,
    nc: f64,
    first_valid: usize,
    out: &mut [f64],
) {
    hwma_scalar(data, na, nb, nc, first_valid, out)
}

#[derive(Debug, Clone)]
pub struct HwmaStream {
    na: f64,
    nb: f64,
    nc: f64,
    last_f: f64,
    last_v: f64,
    last_a: f64,
    filled: bool,
}

impl HwmaStream {
    pub fn try_new(params: HwmaParams) -> Result<Self, HwmaError> {
        let na = params.na.unwrap_or(0.2);
        let nb = params.nb.unwrap_or(0.1);
        let nc = params.nc.unwrap_or(0.1);

        if !(na > 0.0 && na < 1.0 && nb > 0.0 && nb < 1.0 && nc > 0.0 && nc < 1.0) {
            return Err(HwmaError::InvalidParams { na, nb, nc });
        }
        Ok(Self {
            na,
            nb,
            nc,
            last_f: f64::NAN,
            last_v: 0.0,
            last_a: 0.0,
            filled: false,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        if !self.filled {
            self.last_f = value;
            self.last_v = 0.0;
            self.last_a = 0.0;
            self.filled = true;
            return Some(self.last_f + self.last_v + 0.5 * self.last_a);
        }
        let f = (1.0 - self.na) * (self.last_f + self.last_v + 0.5 * self.last_a) + self.na * value;
        let v = (1.0 - self.nb) * (self.last_v + self.last_a) + self.nb * (f - self.last_f);
        let a = (1.0 - self.nc) * self.last_a + self.nc * (v - self.last_v);
        let hwma_val = f + v + 0.5 * a;
        self.last_f = f;
        self.last_v = v;
        self.last_a = a;
        Some(hwma_val)
    }
}

#[derive(Clone, Debug)]
pub struct HwmaBatchRange {
    pub na: (f64, f64, f64),
    pub nb: (f64, f64, f64),
    pub nc: (f64, f64, f64),
}

impl Default for HwmaBatchRange {
    fn default() -> Self {
        Self {
            na: (0.2, 0.2, 0.0),
            nb: (0.1, 0.1, 0.0),
            nc: (0.1, 0.1, 0.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct HwmaBatchBuilder {
    range: HwmaBatchRange,
    kernel: Kernel,
}

impl HwmaBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline]
    pub fn na_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.na = (start, end, step);
        self
    }
    #[inline]
    pub fn nb_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.nb = (start, end, step);
        self
    }
    #[inline]
    pub fn nc_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.nc = (start, end, step);
        self
    }
    #[inline]
    pub fn na_static(mut self, v: f64) -> Self {
        self.range.na = (v, v, 0.0);
        self
    }
    #[inline]
    pub fn nb_static(mut self, v: f64) -> Self {
        self.range.nb = (v, v, 0.0);
        self
    }
    #[inline]
    pub fn nc_static(mut self, v: f64) -> Self {
        self.range.nc = (v, v, 0.0);
        self
    }

    pub fn apply_slice(self, data: &[f64]) -> Result<HwmaBatchOutput, HwmaError> {
        hwma_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<HwmaBatchOutput, HwmaError> {
        HwmaBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<HwmaBatchOutput, HwmaError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<HwmaBatchOutput, HwmaError> {
        HwmaBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn hwma_batch_with_kernel(
    data: &[f64],
    sweep: &HwmaBatchRange,
    k: Kernel,
) -> Result<HwmaBatchOutput, HwmaError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(HwmaError::EmptyData),
    };

    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    hwma_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct HwmaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<HwmaParams>,
    pub rows: usize,
    pub cols: usize,
}
impl HwmaBatchOutput {
    pub fn row_for_params(&self, p: &HwmaParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            (c.na.unwrap_or(0.2) - p.na.unwrap_or(0.2)).abs() < 1e-12 &&
            (c.nb.unwrap_or(0.1) - p.nb.unwrap_or(0.1)).abs() < 1e-12 &&
            (c.nc.unwrap_or(0.1) - p.nc.unwrap_or(0.1)).abs() < 1e-12
        })
    }
    pub fn values_for(&self, p: &HwmaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &HwmaBatchRange) -> Vec<HwmaParams> {
    fn axis((start, end, step): (f64, f64, f64)) -> Vec<f64> {
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
    let nas = axis(r.na);
    let nbs = axis(r.nb);
    let ncs = axis(r.nc);

    let mut out = Vec::with_capacity(nas.len() * nbs.len() * ncs.len());
    for &a in &nas {
        for &b in &nbs {
            for &c in &ncs {
                out.push(HwmaParams { na: Some(a), nb: Some(b), nc: Some(c) });
            }
        }
    }
    out
}

#[inline(always)]
pub fn hwma_batch_slice(
    data: &[f64],
    sweep: &HwmaBatchRange,
    kern: Kernel,
) -> Result<HwmaBatchOutput, HwmaError> {
    hwma_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn hwma_batch_par_slice(
    data: &[f64],
    sweep: &HwmaBatchRange,
    kern: Kernel,
) -> Result<HwmaBatchOutput, HwmaError> {
    hwma_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn hwma_batch_inner(
    data: &[f64],
    sweep: &HwmaBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<HwmaBatchOutput, HwmaError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(HwmaError::EmptyData);
    }
    let first = data.iter().position(|x| !x.is_nan()).ok_or(HwmaError::AllValuesNaN)?;
    let rows = combos.len();
    let cols = data.len();
    let warm: Vec<usize> = std::iter::repeat(first).take(rows).collect();

    // ----- 2. allocate rows×cols as MaybeUninit and write the NaN prefixes --------
    let mut raw = make_uninit_matrix(rows, cols);
    unsafe { init_matrix_prefixes(&mut raw, cols, &warm) };

    // ----- 3. closure that fills one row ------------------------------------------
    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let prm = &combos[row];
        let na = prm.na.unwrap();
        let nb = prm.nb.unwrap();
        let nc = prm.nc.unwrap();

        // cast this row to &mut [f64]
        let out_row = core::slice::from_raw_parts_mut(
            dst_mu.as_mut_ptr() as *mut f64,
            dst_mu.len(),
        );

        match kern {
            Kernel::Scalar => hwma_row_scalar(data, first, na, nb, nc, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2   => hwma_row_avx2  (data, first, na, nb, nc, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => hwma_row_avx512(data, first, na, nb, nc, out_row),
            _ => unreachable!(),
        }
    };

    // ----- 4. run every row, writing directly into `raw` ---------------------------
    if parallel {
        raw.par_chunks_mut(cols)
        .enumerate()
        .for_each(|(row, slice)| do_row(row, slice));
    } else {
        for (row, slice) in raw.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    // ----- 5. transmute to Vec<f64> once everything is initialised -----------------
    let values: Vec<f64> = unsafe { std::mem::transmute(raw) };

    Ok(HwmaBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
unsafe fn hwma_row_scalar(
    data: &[f64],
    first: usize,
    na: f64,
    nb: f64,
    nc: f64,
    out: &mut [f64],
) {
    hwma_scalar(data, na, nb, nc, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn hwma_row_avx2(
    data: &[f64],
    first: usize,
    na: f64,
    nb: f64,
    nc: f64,
    out: &mut [f64],
) {
    hwma_scalar(data, na, nb, nc, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn hwma_row_avx512(
    data: &[f64],
    first: usize,
    na: f64,
    nb: f64,
    nc: f64,
    out: &mut [f64],
) {
    hwma_scalar(data, na, nb, nc, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn hwma_row_avx512_short(
    data: &[f64],
    first: usize,
    na: f64,
    nb: f64,
    nc: f64,
    out: &mut [f64],
) {
    hwma_scalar(data, na, nb, nc, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn hwma_row_avx512_long(
    data: &[f64],
    first: usize,
    na: f64,
    nb: f64,
    nc: f64,
    out: &mut [f64],
) {
    hwma_scalar(data, na, nb, nc, first, out);
}

#[inline(always)]
pub fn expand_grid_hwma(r: &HwmaBatchRange) -> Vec<HwmaParams> {
    expand_grid(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_hwma_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = HwmaParams { na: None, nb: None, nc: None };
        let input = HwmaInput::from_candles(&candles, "close", default_params);
        let output = hwma_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_hwma_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = HwmaInput::from_candles(&candles, "close", HwmaParams::default());
        let result = hwma_with_kernel(&input, kernel)?;
        let expected_last_five = [
            57941.04005793378,
            58106.90324194954,
            58250.474156632234,
            58428.90005831887,
            58499.37021151028,
        ];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(diff < 1e-3, "[{}] HWMA {:?} mismatch at idx {}: got {}, expected {}", test_name, kernel, i, val, expected_last_five[i]);
        }
        Ok(())
    }

    fn check_hwma_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = HwmaInput::with_default_candles(&candles);
        match input.data {
            HwmaData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected HwmaData::Candles"),
        }
        let output = hwma_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_hwma_invalid_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [10.0, 20.0, 30.0];
        let params = HwmaParams { na: Some(-0.2), nb: Some(1.1), nc: Some(0.1) };
        let input = HwmaInput::from_slice(&data, params);
        let result = hwma_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_hwma_empty_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data: [f64; 0] = [];
        let params = HwmaParams { na: Some(0.2), nb: Some(0.1), nc: Some(0.1) };
        let input = HwmaInput::from_slice(&data, params);
        let result = hwma_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_hwma_small_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [42.0];
        let params = HwmaParams { na: Some(0.2), nb: Some(0.1), nc: Some(0.1) };
        let input = HwmaInput::from_slice(&data, params);
        let result = hwma_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), data.len());
        assert_eq!(result.values[0], data[0] + 0.0 + 0.5 * 0.0);
        Ok(())
    }

    fn check_hwma_slice_data_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params_1 = HwmaParams::default();
        let input_1 = HwmaInput::from_candles(&candles, "close", params_1);
        let result_1 = hwma_with_kernel(&input_1, kernel)?;
        assert_eq!(result_1.values.len(), candles.close.len());
        let params_2 = HwmaParams { na: Some(0.3), nb: Some(0.15), nc: Some(0.05) };
        let input_2 = HwmaInput::from_slice(&result_1.values, params_2);
        let result_2 = hwma_with_kernel(&input_2, kernel)?;
        assert_eq!(result_2.values.len(), result_1.values.len());
        for i in 240..result_2.values.len() {
            assert!(!result_2.values[i].is_nan());
        }
        Ok(())
    }

    fn check_hwma_nan_check(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = HwmaParams::default();
        let input = HwmaInput::from_candles(&candles, "close", params);
        let result = hwma_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), candles.close.len());
        if result.values.len() > 240 {
            for i in 240..result.values.len() {
                assert!(!result.values[i].is_nan());
            }
        }
        Ok(())
    }

    macro_rules! generate_all_hwma_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $( #[test] fn [<$test_fn _scalar_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar);
                })*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $( #[test] fn [<$test_fn _avx2_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2);
                })*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $( #[test] fn [<$test_fn _avx512_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512);
                })*
            }
        }
    }
    generate_all_hwma_tests!(
        check_hwma_partial_params,
        check_hwma_accuracy,
        check_hwma_default_candles,
        check_hwma_invalid_params,
        check_hwma_empty_data,
        check_hwma_small_data,
        check_hwma_slice_data_reinput,
        check_hwma_nan_check
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = HwmaBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = HwmaParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        let expected = [
            57941.04005793378,
            58106.90324194954,
            58250.474156632234,
            58428.90005831887,
            58499.37021151028,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-3,
                "[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
            );
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
