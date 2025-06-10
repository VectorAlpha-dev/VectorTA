//! # Gaussian Filter
//!
//! A parametric smoothing technique that approximates a Gaussian response using
//! a cascade of discrete poles. Its parameters (`period`, `poles`) control the
//! filter's length and the number of cascaded stages that shape the overall
//! filter response.
//!
//! ## Parameters
//! - **period**: Window size (number of data points).
//! - **poles**: The number of poles (1..4) to use for the filter.
//!
//! ## Errors
//! - **NoData**: gaussian: No data provided.
//! - **InvalidPoles**: gaussian: `poles` is out of range (expected 1..4).
//! - **PeriodLongerThanData**: gaussian: The `period` is longer than the data length.
//!
//! ## Returns
//! - **`Ok(GaussianOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(GaussianError)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    detect_best_batch_kernel, detect_best_kernel, alloc_with_nan_prefix, init_matrix_prefixes, make_uninit_matrix
};
use rayon::prelude::*;
use std::f64::consts::PI;
use thiserror::Error;
use std::mem::MaybeUninit;  
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use std::arch::x86_64::*;

const LANES_AVX512: usize = 8;
const LANES_AVX2  : usize = 4;  

#[inline(always)]
fn alpha_from(period: usize, poles: usize) -> f64 {
    let beta = {
        let numerator = 1.0 - (2.0 * PI / period as f64).cos();
        let denominator = (2.0_f64).powf(1.0 / poles as f64) - 1.0;
        numerator / denominator
    };
    let tmp = beta * beta + 2.0 * beta;
    -beta + tmp.sqrt()
}

impl<'a> AsRef<[f64]> for GaussianInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            GaussianData::Slice(slice) => slice,
            GaussianData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
unsafe fn gaussian_rows8_avx512(
    data:   &[f64],
    params: &[GaussianParams],
    out_rows: &mut [f64],
    cols:  usize,
) {
    debug_assert_eq!(params.len(), LANES_AVX512);
    debug_assert_eq!(out_rows.len(), LANES_AVX512 * cols);

    // ---- per-lane α and pole-count ----------------------------------------
    let mut alpha_arr = [0.0f64; LANES_AVX512];
    let mut pole_arr  = [0u32 ; LANES_AVX512];
    for (lane, prm) in params.iter().enumerate() {
        let p = prm.period.unwrap_or(14);
        let k = prm.poles .unwrap_or(4);
        alpha_arr[lane] = alpha_from(p, k);
        pole_arr [lane] = k as u32;
    }

    let alpha_v     = _mm512_loadu_pd(alpha_arr.as_ptr());
    let one_minus_v = _mm512_sub_pd(_mm512_set1_pd(1.0), alpha_v);

    // stage-masks: lane bit = 1 when that stage is **present**
    let mask_for = |stage: u32| -> __mmask8 {
        let mut m: u8 = 0;
        for lane in 0..LANES_AVX512 {
            if pole_arr[lane] > stage { m |= 1 << lane; }
        }
        m as __mmask8
    };
    let m0 = mask_for(0);
    let m1 = mask_for(1);
    let m2 = mask_for(2);
    let m3 = mask_for(3);

    // state per pole
    let mut st0 = _mm512_setzero_pd();
    let mut st1 = _mm512_setzero_pd();
    let mut st2 = _mm512_setzero_pd();
    let mut st3 = _mm512_setzero_pd();

    // ---- main time loop ----------------------------------------------------
    for (t, &x_n) in data.iter().enumerate() {
        let x_vec = _mm512_set1_pd(x_n);

        // stage 0  : y = α x + (1-α) st
        let y0  = _mm512_fmadd_pd(alpha_v, x_vec, _mm512_mul_pd(one_minus_v, st0));
        st0     = _mm512_mask_mov_pd(st0, m0, y0);

        // stage 1
        let y1  = _mm512_fmadd_pd(alpha_v, st0,  _mm512_mul_pd(one_minus_v, st1));
        st1     = _mm512_mask_mov_pd(st1, m1, y1);

        // stage 2
        let y2  = _mm512_fmadd_pd(alpha_v, st1,  _mm512_mul_pd(one_minus_v, st2));
        st2     = _mm512_mask_mov_pd(st2, m2, y2);

        // stage 3
        let y3  = _mm512_fmadd_pd(alpha_v, st2,  _mm512_mul_pd(one_minus_v, st3));
        st3     = _mm512_mask_mov_pd(st3, m3, y3);

        let mut y = st0;
        y = _mm512_mask_mov_pd(y, m1, st1);
        y = _mm512_mask_mov_pd(y, m2, st2);
        y = _mm512_mask_mov_pd(y, m3, st3);

        // scatter → row-major
        let mut tmp = [0.0f64; LANES_AVX512];
        _mm512_storeu_pd(tmp.as_mut_ptr(), y);
        for lane in 0..LANES_AVX512 {
            out_rows[lane * cols + t] = tmp[lane];
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
unsafe fn gaussian_batch_tile_avx2(
    data:    &[f64],
    combos:  &[GaussianParams],
    out_mu:  &mut [core::mem::MaybeUninit<f64>],
    cols:    usize,
) {
    // view as &[f64] for the SIMD helpers
    let out = core::slice::from_raw_parts_mut(out_mu.as_mut_ptr() as *mut f64,
                                              out_mu.len());

    let mut row = 0;
    while row + LANES_AVX2 <= combos.len() {
        gaussian_rows4_avx2(
            data,
            &combos[row..row + LANES_AVX2],
            &mut out[row * cols..(row + LANES_AVX2) * cols],
            cols,
        );
        row += LANES_AVX2;
    }
    for r in row..combos.len() {
        gaussian_row_scalar(
            data,
            &combos[r],
            &mut out[r * cols..(r + 1) * cols],
        );
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
unsafe fn gaussian_rows4_avx2(
    data:    &[f64],
    params:  &[GaussianParams],
    out_rows:&mut [f64],
    cols:    usize,
) {
    debug_assert_eq!(params.len(), LANES_AVX2);
    debug_assert_eq!(out_rows.len(), LANES_AVX2 * cols);

    // lane-specific α and pole-count
    let mut alpha_arr = [0.0f64; LANES_AVX2];
    let mut pole_arr  = [0u32 ; LANES_AVX2];
    for (l, prm) in params.iter().enumerate() {
        alpha_arr[l] = alpha_from(prm.period.unwrap_or(14),
                                  prm.poles .unwrap_or(4));
        pole_arr [l] = prm.poles.unwrap_or(4) as u32;
    }
    let alpha_v     = _mm256_loadu_pd(alpha_arr.as_ptr());
    let one_minus_v = _mm256_sub_pd(_mm256_set1_pd(1.0), alpha_v);

    // state per pole
    let mut st0 = _mm256_setzero_pd();
    let mut st1 = _mm256_setzero_pd();
    let mut st2 = _mm256_setzero_pd();
    let mut st3 = _mm256_setzero_pd();

    // scratch arrays to pick lane-wise final value
    let mut y0a = [0.0; LANES_AVX2];
    let mut y1a = [0.0; LANES_AVX2];
    let mut y2a = [0.0; LANES_AVX2];
    let mut y3a = [0.0; LANES_AVX2];

    for (t, &x_n) in data.iter().enumerate() {
        let x_v = _mm256_set1_pd(x_n);

        // stage-0 … stage-3
        let y0_v = _mm256_fmadd_pd(alpha_v, x_v,  _mm256_mul_pd(one_minus_v, st0));
        st0      = y0_v;

        let y1_v = _mm256_fmadd_pd(alpha_v, st0,  _mm256_mul_pd(one_minus_v, st1));
        st1      = y1_v;

        let y2_v = _mm256_fmadd_pd(alpha_v, st1,  _mm256_mul_pd(one_minus_v, st2));
        st2      = y2_v;

        let y3_v = _mm256_fmadd_pd(alpha_v, st2,  _mm256_mul_pd(one_minus_v, st3));
        st3      = y3_v;

        // store to scratch ­- (compilers fold these back-to-back stores nicely)
        _mm256_storeu_pd(y0a.as_mut_ptr(), y0_v);
        _mm256_storeu_pd(y1a.as_mut_ptr(), y1_v);
        _mm256_storeu_pd(y2a.as_mut_ptr(), y2_v);
        _mm256_storeu_pd(y3a.as_mut_ptr(), y3_v);

        // choose per-lane final output and scatter
        for lane in 0..LANES_AVX2 {
            let final_y = match pole_arr[lane] {
                1 => y0a[lane],
                2 => y1a[lane],
                3 => y2a[lane],
                _ => y3a[lane],          // 4
            };
            out_rows[lane * cols + t] = final_y;
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
unsafe fn gaussian_batch_tile_avx512(
    data:   &[f64],
    combos: &[GaussianParams],
    out_mu: &mut [core::mem::MaybeUninit<f64>],   // <-- changed
    cols:   usize,
) {
    // temporary view as &mut [f64]
    let out = core::slice::from_raw_parts_mut(out_mu.as_mut_ptr() as *mut f64,
                                              out_mu.len());

    let mut row = 0;
    while row + LANES_AVX512 <= combos.len() {
        gaussian_rows8_avx512(
            data,
            &combos[row..row + LANES_AVX512],
            &mut out[row * cols..(row + LANES_AVX512) * cols],
            cols,
        );
        row += LANES_AVX512;
    }
    // tail rows (≤7) – scalar
    for r in row..combos.len() {
        gaussian_row_scalar(
            data,
            &combos[r],
            &mut out[r * cols..(r + 1) * cols],
        );
    }
}

#[derive(Debug, Clone)]
pub enum GaussianData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct GaussianOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, Copy)]
pub struct GaussianParams {
    pub period: Option<usize>,
    pub poles: Option<usize>,
}

impl Default for GaussianParams {
    fn default() -> Self {
        Self {
            period: Some(14),
            poles: Some(4),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GaussianInput<'a> {
    pub data: GaussianData<'a>,
    pub params: GaussianParams,
}

impl<'a> GaussianInput<'a> {
    pub fn from_candles(c: &'a Candles, s: &'a str, p: GaussianParams) -> Self {
        Self {
            data: GaussianData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    pub fn from_slice(sl: &'a [f64], p: GaussianParams) -> Self {
        Self {
            data: GaussianData::Slice(sl),
            params: p,
        }
    }
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", GaussianParams::default())
    }
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(14)
    }
    pub fn get_poles(&self) -> usize {
        self.params.poles.unwrap_or(4)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct GaussianBuilder {
    period: Option<usize>,
    poles: Option<usize>,
    kernel: Kernel,
}

impl Default for GaussianBuilder {
    fn default() -> Self {
        Self {
            period: None,
            poles: None,
            kernel: Kernel::Auto,
        }
    }
}

impl GaussianBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn period(mut self, n: usize) -> Self {
        self.period = Some(n);
        self
    }
    pub fn poles(mut self, k: usize) -> Self {
        self.poles = Some(k);
        self
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    pub fn apply(self, c: &Candles) -> Result<GaussianOutput, GaussianError> {
        let p = GaussianParams {
            period: self.period,
            poles: self.poles,
        };
        let i = GaussianInput::from_candles(c, "close", p);
        gaussian_with_kernel(&i, self.kernel)
    }
    pub fn apply_slice(self, d: &[f64]) -> Result<GaussianOutput, GaussianError> {
        let p = GaussianParams {
            period: self.period,
            poles: self.poles,
        };
        let i = GaussianInput::from_slice(d, p);
        gaussian_with_kernel(&i, self.kernel)
    }
    pub fn into_stream(self) -> Result<GaussianStream, GaussianError> {
        let p = GaussianParams {
            period: self.period,
            poles: self.poles,
        };
        GaussianStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum GaussianError {
    #[error("gaussian: No data provided to Gaussian filter.")]
    NoData,
    #[error("gaussian: Invalid number of poles: expected 1..4, got {poles}")]
    InvalidPoles { poles: usize },
    #[error(
        "Gaussian filter period is longer than the data. period={period}, data_len={data_len}"
    )]
    PeriodLongerThanData { period: usize, data_len: usize },
}

#[inline]
pub fn gaussian(input: &GaussianInput) -> Result<GaussianOutput, GaussianError> {
    gaussian_with_kernel(input, Kernel::Auto)
}

pub fn gaussian_with_kernel(
    input: &GaussianInput,
    kernel: Kernel,
) -> Result<GaussianOutput, GaussianError> {
    let data: &[f64] = match &input.data {
        GaussianData::Candles { candles, source } => source_type(candles, source),
        GaussianData::Slice(sl) => sl,
    };

    let len = data.len();
    let period = input.get_period();
    let poles = input.get_poles();

    if len == 0 {
        return Err(GaussianError::NoData);
    }
    if !(1..=4).contains(&poles) {
        return Err(GaussianError::InvalidPoles { poles });
    }
    if len < period {
        return Err(GaussianError::PeriodLongerThanData {
            period,
            data_len: len,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let first_valid = data.iter().position(|x| !x.is_nan()).unwrap_or(len);
    let warm        = first_valid + period;

    let mut out = alloc_with_nan_prefix(len, warm);

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => gaussian_scalar(data, period, poles, &mut out),
            Kernel::Avx2 | Kernel::Avx2Batch => gaussian_avx2(data, period, poles, &mut out),
            Kernel::Avx512 | Kernel::Avx512Batch => gaussian_avx512(data, period, poles, &mut out),
            Kernel::Auto => unreachable!(),
        }
    }
    Ok(GaussianOutput { values: out })
}

#[inline(always)]
pub fn gaussian_scalar(data: &[f64], period: usize, poles: usize, out: &mut [f64]) {
    let n = data.len();
    if n == 0 {
        return;
    }
    let beta = {
        let numerator = 1.0 - (2.0 * PI / period as f64).cos();
        let denominator = (2.0_f64).powf(1.0 / poles as f64) - 1.0;
        numerator / denominator
    };
    let alpha = {
        let tmp = beta * beta + 2.0 * beta;
        -beta + tmp.sqrt()
    };
    let vals = match poles {
        1 => gaussian_poles1(data, n, alpha),
        2 => gaussian_poles2(data, n, alpha),
        3 => gaussian_poles3(data, n, alpha),
        4 => gaussian_poles4(data, n, alpha),
        _ => unreachable!(),
    };
    out.copy_from_slice(&vals);
}


#[inline(always)]
#[allow(unused_variables)]
pub unsafe fn gaussian_avx2(data: &[f64], period: usize, poles: usize, out: &mut [f64]) {
    gaussian_scalar(data, period, poles, out);
}

#[inline(always)]
#[allow(unused_variables)]
pub unsafe fn gaussian_avx512(data: &[f64], period: usize, poles: usize, out: &mut [f64]) {
    gaussian_scalar(data, period, poles, out);
}

#[inline(always)]
fn gaussian_poles1(data: &[f64], n: usize, alpha: f64) -> Vec<f64> {
    let c0 = alpha;
    let c1 = 1.0 - alpha;
    let mut fil = vec![0.0; 1 + n];
    for i in 0..n {
        fil[i + 1] = c0 * data[i] + c1 * fil[i];
    }
    fil[1..1 + n].to_vec()
}
#[inline(always)]
fn gaussian_poles2(data: &[f64], n: usize, alpha: f64) -> Vec<f64> {
    let a2 = alpha * alpha;
    let one_a = 1.0 - alpha;
    let c0 = a2;
    let c1 = 2.0 * one_a;
    let c2 = -(one_a * one_a);
    let mut fil = vec![0.0; 2 + n];
    for i in 0..n {
        fil[i + 2] = c0 * data[i] + c1 * fil[i + 1] + c2 * fil[i];
    }
    fil[2..2 + n].to_vec()
}
#[inline(always)]
fn gaussian_poles3(data: &[f64], n: usize, alpha: f64) -> Vec<f64> {
    let a3 = alpha * alpha * alpha;
    let one_a = 1.0 - alpha;
    let one_a2 = one_a * one_a;
    let c0 = a3;
    let c1 = 3.0 * one_a;
    let c2 = -3.0 * one_a2;
    let c3 = one_a2 * one_a;
    let mut fil = vec![0.0; 3 + n];
    for i in 0..n {
        fil[i + 3] = c0 * data[i] + c1 * fil[i + 2] + c2 * fil[i + 1] + c3 * fil[i];
    }
    fil[3..3 + n].to_vec()
}
#[inline(always)]
fn gaussian_poles4(data: &[f64], n: usize, alpha: f64) -> Vec<f64> {
    let a4 = alpha * alpha * alpha * alpha;
    let one_a = 1.0 - alpha;
    let one_a2 = one_a * one_a;
    let one_a3 = one_a2 * one_a;
    let c0 = a4;
    let c1 = 4.0 * one_a;
    let c2 = -6.0 * one_a2;
    let c3 = 4.0 * one_a3;
    let c4 = -(one_a3 * one_a);
    let mut fil = vec![0.0; 4 + n];
    for i in 0..n {
        fil[i + 4] =
            c0 * data[i] + c1 * fil[i + 3] + c2 * fil[i + 2] + c3 * fil[i + 1] + c4 * fil[i];
    }
    fil[4..4 + n].to_vec()
}

#[derive(Debug, Clone)]
pub struct GaussianStream {
    period: usize,
    poles: usize,
    alpha: f64,
    state: Vec<f64>,
    idx: usize,
    init: bool,
}

impl GaussianStream {
    pub fn try_new(params: GaussianParams) -> Result<Self, GaussianError> {
        let period = params.period.unwrap_or(14);
        let poles = params.poles.unwrap_or(4);
        if !(1..=4).contains(&poles) {
            return Err(GaussianError::InvalidPoles { poles });
        }
        let beta = {
            let numerator = 1.0 - (2.0 * PI / period as f64).cos();
            let denominator = (2.0_f64).powf(1.0 / poles as f64) - 1.0;
            numerator / denominator
        };
        let alpha = {
            let tmp = beta * beta + 2.0 * beta;
            -beta + tmp.sqrt()
        };
        Ok(Self {
            period,
            poles,
            alpha,
            state: vec![0.0; poles],
            idx: 0,
            init: false,
        })
    }
    pub fn update(&mut self, value: f64) -> f64 {
        let p = self.poles;
        let a = self.alpha;
        let mut prev = value;
        for s in 0..p {
            let last = self.state[s];
            let next = a * prev + (1.0 - a) * last;
            self.state[s] = next;
            prev = next;
        }
        prev
    }
}

// Batch

#[derive(Clone, Debug)]
pub struct GaussianBatchRange {
    pub period: (usize, usize, usize),
    pub poles: (usize, usize, usize),
}

impl Default for GaussianBatchRange {
    fn default() -> Self {
        Self {
            period: (14, 60, 1),
            poles: (1, 4, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct GaussianBatchBuilder {
    range: GaussianBatchRange,
    kernel: Kernel,
}

impl GaussianBatchBuilder {
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
    pub fn poles_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.poles = (start, end, step);
        self
    }
    pub fn poles_static(mut self, p: usize) -> Self {
        self.range.poles = (p, p, 0);
        self
    }
    pub fn apply_slice(self, data: &[f64]) -> Result<GaussianBatchOutput, GaussianError> {
        gaussian_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn apply_candles(
        self,
        c: &Candles,
        src: &str,
    ) -> Result<GaussianBatchOutput, GaussianError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_slice(
        data: &[f64],
        k: Kernel,
    ) -> Result<GaussianBatchOutput, GaussianError> {
        GaussianBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn with_default_candles(c: &Candles) -> Result<GaussianBatchOutput, GaussianError> {
        GaussianBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct GaussianBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<GaussianParams>,
    pub rows: usize,
    pub cols: usize,
}

impl GaussianBatchOutput {
    pub fn row_for_params(&self, p: &GaussianParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(14) == p.period.unwrap_or(14)
                && c.poles.unwrap_or(4) == p.poles.unwrap_or(4)
        })
    }
    pub fn values_for(&self, p: &GaussianParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &GaussianBatchRange) -> Vec<GaussianParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let periods = axis_usize(r.period);
    let poles = axis_usize(r.poles);
    let mut out = Vec::with_capacity(periods.len() * poles.len());
    for &p in &periods {
        for &k in &poles {
            out.push(GaussianParams {
                period: Some(p),
                poles: Some(k),
            });
        }
    }
    out
}

pub fn gaussian_batch_with_kernel(
    data: &[f64],
    sweep: &GaussianBatchRange,
    k: Kernel,
) -> Result<GaussianBatchOutput, GaussianError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(GaussianError::NoData),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    gaussian_batch_par_slice(data, sweep, simd)
}

#[inline(always)]
pub fn gaussian_batch_slice(
    data: &[f64],
    sweep: &GaussianBatchRange,
    kern: Kernel,
) -> Result<GaussianBatchOutput, GaussianError> {
    gaussian_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn gaussian_batch_par_slice(
    data: &[f64],
    sweep: &GaussianBatchRange,
    kern: Kernel,
) -> Result<GaussianBatchOutput, GaussianError> {
    gaussian_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn gaussian_batch_inner(
    data: &[f64],
    sweep: &GaussianBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<GaussianBatchOutput, GaussianError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(GaussianError::NoData);
    }
    let len  = data.len();
    let rows = combos.len();
    let cols = len;

    // ---------------------------------------------------------------------------
    // 1.  Determine warm-up prefix for every row
    // ---------------------------------------------------------------------------
    let first_valid = data.iter().position(|x| !x.is_nan())
        .ok_or(GaussianError::NoData)?;
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first_valid + c.period.unwrap_or(14))
        .collect();

    // ---------------------------------------------------------------------------
    // 2.  Allocate an un-initialised matrix and write the NaN prefixes
    // ---------------------------------------------------------------------------
    let mut raw = make_uninit_matrix(rows, cols);
    unsafe { init_matrix_prefixes(&mut raw, cols, &warm) };

    // ---------------------------------------------------------------------------
    // 3.  Closure that fills ONE row  (gets &mut [MaybeUninit<f64>])
    // ---------------------------------------------------------------------------
    let do_row = |row: usize, dst_mu: &mut [core::mem::MaybeUninit<f64>]| unsafe {
        let prm     = &combos[row];
        let out_row = core::slice::from_raw_parts_mut(
            dst_mu.as_mut_ptr() as *mut f64,
            dst_mu.len(),
        );
        match kern {
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => gaussian_row_avx512(data, prm, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2   => gaussian_row_avx2  (data, prm, out_row),
            _              => gaussian_row_scalar(data, prm, out_row),
        }
    };

    // ---------------------------------------------------------------------------
    // 4.  Run every row (parallel or serial) directly into `raw`
    // ---------------------------------------------------------------------------
    if parallel {
        match kern {
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 if is_x86_feature_detected!("avx512f") => {
                // Rayon over 8-row tiles keeps the work-chunks large
                let chunks = raw.par_chunks_mut(cols * LANES_AVX512)
                    .zip(combos.par_chunks(LANES_AVX512));
                chunks.for_each(|(out_blk, prm_blk)| unsafe {
                    gaussian_batch_tile_avx512(data, prm_blk, out_blk, cols)
                });
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2   if is_x86_feature_detected!("avx2")     => {
            raw.par_chunks_mut(cols * LANES_AVX2)
               .zip(combos.par_chunks(LANES_AVX2))
               .for_each(|(out_blk, prm_blk)| unsafe {
                   gaussian_batch_tile_avx2(data, prm_blk, out_blk, cols)
               });
            }
            _ => {
                raw.par_chunks_mut(cols)
                      .enumerate()
                      .for_each(|(row, slice)| do_row(row, slice));
            }
        }
    } else {
        match kern {
          #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 if is_x86_feature_detected!("avx512f") => unsafe {
                gaussian_batch_tile_avx512(data, &combos, &mut raw, cols);
            },
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 if is_x86_feature_detected!("avx2") => unsafe {
                gaussian_batch_tile_avx2(data, &combos, &mut raw, cols);
          },
            _ => {
                for (row, slice) in raw.chunks_mut(cols).enumerate() {
                    do_row(row, slice);
                }
            }
        }
    }

    // ---------------------------------------------------------------------------
    // 5.  All elements are now initialised – transmute to Vec<f64>
    // ---------------------------------------------------------------------------
    let values: Vec<f64> = unsafe { std::mem::transmute(raw) };
    Ok(GaussianBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
pub unsafe fn gaussian_row_scalar(data: &[f64], prm: &GaussianParams, out: &mut [f64]) {
    gaussian_scalar(data, prm.period.unwrap_or(14), prm.poles.unwrap_or(4), out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn gaussian_row_avx2(
    data: &[f64],
    prm:  &GaussianParams,
    out:  &mut [f64],
) {
    let mut combos = [prm.clone(); LANES_AVX2];
    let mut buf    = vec![0.0f64; LANES_AVX2 * data.len()];
    gaussian_rows4_avx2(data, &combos, &mut buf, data.len());
    out.copy_from_slice(&buf[..data.len()]);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn gaussian_row_avx512(
    data: &[f64],
    prm: &GaussianParams,
    out: &mut [f64],
) {
    // fast path: 1-row “batch” using the SIMD core
    let mut combos = [prm.clone(); LANES_AVX512];
    let mut buf    = vec![0.0f64; LANES_AVX512 * data.len()];
    gaussian_rows8_avx512(data, &combos, &mut buf, data.len());
    out.copy_from_slice(&buf[..data.len()]);
}

// =================== Macro-Based Unit Tests ====================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_gaussian_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = GaussianParams {
            period: None,
            poles: None,
        };
        let input = GaussianInput::from_candles(&candles, "close", default_params);
        let output = gaussian_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_gaussian_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = GaussianParams {
            period: Some(14),
            poles: Some(4),
        };
        let input = GaussianInput::from_candles(&candles, "close", params);
        let result = gaussian_with_kernel(&input, kernel)?;
        let expected_last_five = [
            59221.90637814869,
            59236.15215167245,
            59207.10087088464,
            59178.48276885589,
            59085.36983209433,
        ];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-4,
                "[{}] Gaussian {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_gaussian_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = GaussianInput::with_default_candles(&candles);
        match input.data {
            GaussianData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected GaussianData::Candles"),
        }
        let output = gaussian_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_gaussian_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = GaussianParams {
            period: Some(10),
            poles: None,
        };
        let input = GaussianInput::from_slice(&data_small, params);
        let res = gaussian_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Gaussian should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_gaussian_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = GaussianParams {
            period: Some(14),
            poles: None,
        };
        let input = GaussianInput::from_slice(&single_point, params);
        let res = gaussian_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Gaussian should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_gaussian_reinput(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = GaussianParams {
            period: Some(14),
            poles: Some(4),
        };
        let first_input = GaussianInput::from_candles(&candles, "close", first_params);
        let first_result = gaussian_with_kernel(&first_input, kernel)?;
        assert_eq!(first_result.values.len(), candles.close.len());
        let second_params = GaussianParams {
            period: Some(7),
            poles: Some(2),
        };
        let second_input = GaussianInput::from_slice(&first_result.values, second_params);
        let second_result = gaussian_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        for i in 10..second_result.values.len() {
            assert!(
                !second_result.values[i].is_nan(),
                "NaN found at index {}",
                i
            );
        }
        Ok(())
    }

    fn check_gaussian_nan_handling(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = GaussianInput::from_candles(&candles, "close", GaussianParams::default());
        let res = gaussian_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), candles.close.len());
        let skip = input.params.poles.unwrap_or(4);
        for val in res.values.iter().skip(skip) {
            assert!(
                val.is_finite(),
                "[{}] Gaussian output should be finite once settled.",
                test_name
            );
        }
        Ok(())
    }

    fn check_gaussian_streaming(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let period = 14;
        let poles = 4;
        let input = GaussianInput::from_candles(
            &candles,
            "close",
            GaussianParams {
                period: Some(period),
                poles: Some(poles),
            },
        );
        let batch_output = gaussian_with_kernel(&input, kernel)?.values;
        let mut stream = GaussianStream::try_new(GaussianParams {
            period: Some(period),
            poles: Some(poles),
        })?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            stream_values.push(stream.update(price));
        }
        assert_eq!(batch_output.len(), stream_values.len());
        let skip = poles;
        for (i, (&b, &s)) in batch_output
            .iter()
            .zip(stream_values.iter())
            .enumerate()
            .skip(skip)
        {
            if b.is_nan() && s.is_nan() {
                continue;
            }
            let diff = (b - s).abs();
            assert!(
                diff < 1e-9,
                "[{}] Gaussian streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    macro_rules! generate_all_gaussian_tests {
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

    generate_all_gaussian_tests!(
        check_gaussian_partial_params,
        check_gaussian_accuracy,
        check_gaussian_default_candles,
        check_gaussian_period_exceeds_length,
        check_gaussian_very_small_dataset,
        check_gaussian_reinput,
        check_gaussian_nan_handling,
        check_gaussian_streaming
    );

    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = GaussianBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = GaussianParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        let expected = [
            59221.90637814869,
            59236.15215167245,
            59207.10087088464,
            59178.48276885589,
            59085.36983209433,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-4,
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
                #[test] fn [<$fn_name _avx2>]()        {
                    let _ = $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch);
                }
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
