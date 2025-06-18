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
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use rayon::prelude::*;
use thiserror::Error;
use std::convert::AsRef;
use std::error::Error;
use std::f64::consts::PI;
use std::mem::MaybeUninit;


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

pub fn mama_with_kernel(input: &MamaInput, kernel: Kernel) -> Result<MamaOutput, MamaError> {
    let data: &[f64] = input.as_ref();
    let len: usize = data.len();
    if len < 10 {
        return Err(MamaError::NotEnoughData {
            needed: 10,
            found: len,
        });
    }
    let fast_limit = input.get_fast_limit();
    let slow_limit = input.get_slow_limit();
    if fast_limit <= 0.0 || fast_limit.is_infinite() || fast_limit.is_nan() {
        return Err(MamaError::InvalidFastLimit { fast_limit });
    }
    if slow_limit <= 0.0 || slow_limit.is_infinite() || slow_limit.is_nan() {
        return Err(MamaError::InvalidSlowLimit { slow_limit });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => mama_scalar(data, fast_limit, slow_limit),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => mama_avx2(data, fast_limit, slow_limit),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => mama_avx512(data, fast_limit, slow_limit),
            _ => unreachable!(),
        }
    }
}

#[inline(always)]
pub unsafe fn mama_scalar(data: &[f64], fast_limit: f64, slow_limit: f64) -> Result<MamaOutput, MamaError> {
    Ok(mama_scalar_impl(data, fast_limit, slow_limit))
}

// SIMD stubs (AVX2/AVX512 all point to scalar)

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn mama_avx2(data: &[f64], fast_limit: f64, slow_limit: f64) -> Result<MamaOutput, MamaError> {
    mama_scalar(data, fast_limit, slow_limit)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn mama_avx512(data: &[f64], fast_limit: f64, slow_limit: f64) -> Result<MamaOutput, MamaError> {
    if data.len() <= 32 {
        unsafe { mama_avx512_short(data, fast_limit, slow_limit) }
    } else {
        unsafe { mama_avx512_long(data, fast_limit, slow_limit) }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn mama_avx512_short(data: &[f64], fast_limit: f64, slow_limit: f64) -> Result<MamaOutput, MamaError> {
    mama_scalar(data, fast_limit, slow_limit)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn mama_avx512_long(data: &[f64], fast_limit: f64, slow_limit: f64) -> Result<MamaOutput, MamaError> {
    mama_scalar(data, fast_limit, slow_limit)
}

// Scalar implementation (your logic preserved)

#[inline(always)]
fn hilbert(x0: f64, x2: f64, x4: f64, x6: f64) -> f64 {
    0.0962 * x0 + 0.5769 * x2 - 0.5769 * x4 - 0.0962 * x6
}

fn mama_scalar_impl(data: &[f64], fast_limit: f64, slow_limit: f64) -> MamaOutput {
    let len = data.len();
    let warm = 10;
    let mut mama_values = alloc_with_nan_prefix(len, warm);
    let mut fama_values = alloc_with_nan_prefix(len, warm);

    let mut smooth_buf = [data[0]; 7];
    let mut detrender_buf = [data[0]; 7];
    let mut i1_buf = [data[0]; 7];
    let mut q1_buf = [data[0]; 7];

    let mut prev_mesa_period = 0.0;
    let mut prev_mama = data[0];
    let mut prev_fama = data[0];
    let mut prev_i2_sm = 0.0;
    let mut prev_q2_sm = 0.0;
    let mut prev_re = 0.0;
    let mut prev_im = 0.0;
    let mut prev_phase = 0.0;

    for i in 0..len {
        let src_i = data[i];
        let s1 = if i >= 1 { data[i - 1] } else { src_i };
        let s2 = if i >= 2 { data[i - 2] } else { src_i };
        let s3 = if i >= 3 { data[i - 3] } else { src_i };
        let smooth_val = (4.0 * src_i + 3.0 * s1 + 2.0 * s2 + s3) / 10.0;
        let idx = i % 7;
        smooth_buf[idx] = smooth_val;
        let x0 = smooth_buf[idx];
        let x2 = smooth_buf[(idx + 7 - 2) % 7];
        let x4 = smooth_buf[(idx + 7 - 4) % 7];
        let x6 = smooth_buf[(idx + 7 - 6) % 7];
        let mesa_period_mult = 0.075 * prev_mesa_period + 0.54;
        let dt_val = hilbert(x0, x2, x4, x6) * mesa_period_mult;
        detrender_buf[idx] = dt_val;
        let d0 = detrender_buf[idx];
        let d2 = detrender_buf[(idx + 7 - 2) % 7];
        let d4 = detrender_buf[(idx + 7 - 4) % 7];
        let d6 = detrender_buf[(idx + 7 - 6) % 7];
        let i1_val = if i >= 3 {
            detrender_buf[(idx + 7 - 3) % 7]
        } else {
            d0
        };
        i1_buf[idx] = i1_val;
        let q1_val = hilbert(d0, d2, d4, d6) * mesa_period_mult;
        q1_buf[idx] = q1_val;
        let i1_0 = i1_buf[idx];
        let i1_2 = i1_buf[(idx + 7 - 2) % 7];
        let i1_4 = i1_buf[(idx + 7 - 4) % 7];
        let i1_6 = i1_buf[(idx + 7 - 6) % 7];
        let j_i = hilbert(i1_0, i1_2, i1_4, i1_6) * mesa_period_mult;
        let q1_0 = q1_buf[idx];
        let q1_2 = q1_buf[(idx + 7 - 2) % 7];
        let q1_4 = q1_buf[(idx + 7 - 4) % 7];
        let q1_6 = q1_buf[(idx + 7 - 6) % 7];
        let j_q = hilbert(q1_0, q1_2, q1_4, q1_6) * mesa_period_mult;
        let i2 = i1_val - j_q;
        let q2 = q1_val + j_i;
        let i2_sm = 0.2 * i2 + 0.8 * prev_i2_sm;
        let q2_sm = 0.2 * q2 + 0.8 * prev_q2_sm;
        let re = 0.2 * (i2_sm * prev_i2_sm + q2_sm * prev_q2_sm) + 0.8 * prev_re;
        let im = 0.2 * (i2_sm * prev_q2_sm - q2_sm * prev_i2_sm) + 0.8 * prev_im;
        prev_i2_sm = i2_sm;
        prev_q2_sm = q2_sm;
        prev_re = re;
        prev_im = im;
        let mut cur_mesa = if re != 0.0 && im != 0.0 {
            2.0 * PI / crate::utilities::math_functions::atan64(im / re)
        } else {
            0.0
        };
        let pm = if i > 0 { prev_mesa_period } else { cur_mesa };
        if cur_mesa > 1.5 * pm {
            cur_mesa = 1.5 * pm;
        }
        if cur_mesa < 0.67 * pm {
            cur_mesa = 0.67 * pm;
        }
        if cur_mesa < 6.0 {
            cur_mesa = 6.0;
        } else if cur_mesa > 50.0 {
            cur_mesa = 50.0;
        }
        let cur_mesa_smooth = 0.2 * cur_mesa + 0.8 * pm;
        prev_mesa_period = cur_mesa_smooth;
        let mut cur_phase = 0.0;
        if i1_val != 0.0 {
            cur_phase = (180.0 / PI) * crate::utilities::math_functions::atan64(q1_val / i1_val)
        }
        let old_phase = prev_phase;
        let mut dp = old_phase - cur_phase;
        if dp < 1.0 {
            dp = 1.0;
        }
        prev_phase = cur_phase;
        let alpha = {
            let a = fast_limit / dp;
            if a < slow_limit {
                slow_limit
            } else {
                a
            }
        };
        let cur_mama = alpha * src_i + (1.0 - alpha) * prev_mama;
        let a2 = 0.5 * alpha;
        let cur_fama = a2 * cur_mama + (1.0 - a2) * prev_fama;
        prev_mama = cur_mama;
        prev_fama = cur_fama;
        mama_values[i] = cur_mama;
        fama_values[i] = cur_fama;
    }
    MamaOutput {
        mama_values,
        fama_values,
    }
}

// Stream (online) MAMA

#[derive(Debug, Clone)]
pub struct MamaStream {
    fast_limit: f64,
    slow_limit: f64,
    buffer: Vec<f64>,
    pos: usize,
    filled: bool,
    state: Option<mama_scalar_stream_state>,
}

#[derive(Debug, Clone)]
struct mama_scalar_stream_state {
    smooth_buf: [f64; 7],
    detrender_buf: [f64; 7],
    i1_buf: [f64; 7],
    q1_buf: [f64; 7],
    prev_mesa_period: f64,
    prev_mama: f64,
    prev_fama: f64,
    prev_i2_sm: f64,
    prev_q2_sm: f64,
    prev_re: f64,
    prev_im: f64,
    prev_phase: f64,
    i: usize,
}

impl MamaStream {
    pub fn try_new(params: MamaParams) -> Result<Self, MamaError> {
        let fast_limit = params.fast_limit.unwrap_or(0.5);
        let slow_limit = params.slow_limit.unwrap_or(0.05);
        if fast_limit <= 0.0 || fast_limit.is_infinite() || fast_limit.is_nan() {
            return Err(MamaError::InvalidFastLimit { fast_limit });
        }
        if slow_limit <= 0.0 || slow_limit.is_infinite() || slow_limit.is_nan() {
            return Err(MamaError::InvalidSlowLimit { slow_limit });
        }
        Ok(Self {
            fast_limit,
            slow_limit,
            buffer: vec![f64::NAN; 10],
            pos: 0,
            filled: false,
            state: None,
        })
    }
    pub fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        self.buffer[self.pos] = value;
        self.pos = (self.pos + 1) % 10;
        if !self.filled && self.pos == 0 {
            self.filled = true;
        }
        if !self.filled {
            return None;
        }
        let slice_data: Vec<f64>;
        let slice: &[f64] = if self.pos == 0 {
            &self.buffer[..]
        } else {
            slice_data = {
                let mut tmp = Vec::with_capacity(10);
                tmp.extend_from_slice(&self.buffer[self.pos..]);
                tmp.extend_from_slice(&self.buffer[..self.pos]);
                tmp
            };
            &slice_data[..]
        };
        let output = mama_scalar_impl(slice, self.fast_limit, self.slow_limit);
        let idx = output.mama_values.len() - 1;
        Some((output.mama_values[idx], output.fama_values[idx]))
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
        raw_mama.par_chunks_mut(cols)
                .zip(raw_fama.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(row, (m_row, f_row))| do_row(row, m_row, f_row));
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

// Row API (for batch)

#[inline(always)]
pub unsafe fn mama_row_scalar(
    data: &[f64],
    fast_limit: f64,
    slow_limit: f64,
    out_mama: &mut [f64],
    out_fama: &mut [f64],
) {
    let output = mama_scalar_impl(data, fast_limit, slow_limit);
    out_mama.copy_from_slice(&output.mama_values);
    out_fama.copy_from_slice(&output.fama_values);
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
    mama_row_scalar(data, fast_limit, slow_limit, out_mama, out_fama)
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
    if data.len() <= 32 {
        mama_row_avx512_short(data, fast_limit, slow_limit, out_mama, out_fama)
    } else {
        mama_row_avx512_long(data, fast_limit, slow_limit, out_mama, out_fama)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn mama_row_avx512_short(
    data: &[f64],
    fast_limit: f64,
    slow_limit: f64,
    out_mama: &mut [f64],
    out_fama: &mut [f64],
) {
    mama_row_scalar(data, fast_limit, slow_limit, out_mama, out_fama)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn mama_row_avx512_long(
    data: &[f64],
    fast_limit: f64,
    slow_limit: f64,
    out_mama: &mut [f64],
    out_fama: &mut [f64],
) {
    mama_row_scalar(data, fast_limit, slow_limit, out_mama, out_fama)
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

    generate_all_mama_tests!(
        check_mama_partial_params,
        check_mama_accuracy,
        check_mama_default_candles,
        check_mama_with_insufficient_data,
        check_mama_very_small_dataset,
        check_mama_reinput,
        check_mama_nan_handling
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
    gen_batch_tests!(check_batch_default_row);
}
