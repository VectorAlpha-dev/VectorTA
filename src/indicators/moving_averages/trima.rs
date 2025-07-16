//! # Triangular Moving Average (TRIMA)
//!
//! A moving average computed by averaging an underlying Simple Moving Average (SMA) over
//! the specified `period`, resulting in a smoother output than a single SMA.
//! TRIMA supports different compute kernels and batch processing via builder APIs.
//!
//! ## Parameters
//! - **period**: Window size (must be > 3).
//!
//! ## Errors
//! - **AllValuesNaN**: trima: All input data values are `NaN`.
//! - **InvalidPeriod**: trima: `period` is zero, ≤ 3, or exceeds the data length.
//! - **NotEnoughValidData**: trima: Not enough valid data points for the requested `period`.
//! - **NoData**: trima: No data provided.
//!
//! ## Returns
//! - **`Ok(TrimaOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(TrimaError)`** otherwise.
use crate::indicators::sma::{sma, SmaData, SmaInput, SmaParams};
use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes, make_uninit_matrix, alloc_with_nan_prefix};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use thiserror::Error;
use paste::paste;
use std::mem::MaybeUninit;

impl<'a> AsRef<[f64]> for TrimaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            TrimaData::Slice(slice) => slice,
            TrimaData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum TrimaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct TrimaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct TrimaParams {
    pub period: Option<usize>,
}

impl Default for TrimaParams {
    fn default() -> Self {
        Self { period: Some(30) }
    }
}

#[derive(Debug, Clone)]
pub struct TrimaInput<'a> {
    pub data: TrimaData<'a>,
    pub params: TrimaParams,
}

impl<'a> TrimaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: TrimaParams) -> Self {
        Self {
            data: TrimaData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: TrimaParams) -> Self {
        Self {
            data: TrimaData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", TrimaParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(14)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct TrimaBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for TrimaBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl TrimaBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<TrimaOutput, TrimaError> {
        let p = TrimaParams {
            period: self.period,
        };
        let i = TrimaInput::from_candles(c, "close", p);
        trima_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<TrimaOutput, TrimaError> {
        let p = TrimaParams {
            period: self.period,
        };
        let i = TrimaInput::from_slice(d, p);
        trima_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<TrimaStream, TrimaError> {
        let p = TrimaParams {
            period: self.period,
        };
        TrimaStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum TrimaError {
    #[error("trima: All values are NaN.")]
    AllValuesNaN,

    #[error("trima: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },

    #[error("trima: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },

    #[error("trima: Period too small: {period}")]
    PeriodTooSmall { period: usize },

    #[error("trima: No data provided.")]
    NoData,
}

#[inline]
pub fn trima(input: &TrimaInput) -> Result<TrimaOutput, TrimaError> {
    trima_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn trima_prepare<'a>(
    input: &'a TrimaInput,
    kernel: Kernel,
) -> Result<
    (
        /*data*/ &'a [f64],
        /*period*/ usize,
        /*m1*/ usize,
        /*m2*/ usize,
        /*first*/ usize,
        /*chosen*/ Kernel,
    ),
    TrimaError,
> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(TrimaError::NoData);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(TrimaError::AllValuesNaN)?;
    let period = input.get_period();
    
    if period == 0 || period > len {
        return Err(TrimaError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if period <= 3 {
        return Err(TrimaError::PeriodTooSmall { period });
    }
    if (len - first) < period {
        return Err(TrimaError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }
    
    let m1 = (period + 1) / 2;
    let m2 = period - m1 + 1;
    
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };
    
    Ok((data, period, m1, m2, first, chosen))
}

#[inline(always)]
fn trima_compute_into(
    data: &[f64],
    period: usize,
    m1: usize,
    m2: usize,
    first: usize,
    kernel: Kernel,
    out: &mut [f64],
) {
    unsafe {
        match kernel {
            Kernel::Scalar | Kernel::ScalarBatch => {
                trima_scalar_optimized(data, period, m1, m2, first, out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => trima_scalar_optimized(data, period, m1, m2, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                trima_scalar_optimized(data, period, m1, m2, first, out)
            }
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                // Fallback to scalar when AVX is not available
                trima_scalar_optimized(data, period, m1, m2, first, out)
            }
            _ => unreachable!(),
        }
    }
}

#[inline(always)]
unsafe fn trima_scalar_optimized(
    data: &[f64],
    _period: usize,
    m1: usize,
    m2: usize,
    first: usize,
    out: &mut [f64],
) {
    // For now, fall back to the original implementation using two SMA passes
    // This ensures correctness while maintaining the zero-copy API
    trima_scalar(data, m1 + m2 - 1, first, out);
}

pub fn trima_with_kernel(input: &TrimaInput, kernel: Kernel) -> Result<TrimaOutput, TrimaError> {
    let (data, period, m1, m2, first, chosen) = trima_prepare(input, kernel)?;
    let len = data.len();
    let warm = first + m1 + m2 - 1;
    let mut out = alloc_with_nan_prefix(len, warm);
    trima_compute_into(data, period, m1, m2, first, chosen, &mut out);
    Ok(TrimaOutput { values: out })
}

#[inline]
pub fn trima_scalar(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    // ─── TWO-PASS SMA APPROACH ───
    //
    // Let m1 = (period+1)/2,  m2 = period − m1 + 1.
    // First pass: compute the m1-period SMA of `data`.
    // Second pass: compute the m2-period SMA of that first-pass result.
    // That is exactly the standard definition of a “Triangular MA.”

    let n = data.len();
    let m1 = (period + 1) / 2;
    let m2 = period - m1 + 1;

    // FIRST PASS:  length-m1 SMA on `data`.
    let sma1_in = SmaInput {
        data: SmaData::Slice(data),
        params: SmaParams { period: Some(m1) },
    };
    // We know this cannot error, because `period <= n` and m1 > 0.
    let pass1 = sma(&sma1_in).unwrap();

    // SECOND PASS: length-m2 SMA on the first-pass values.
    let sma2_in = SmaInput {
        data: SmaData::Slice(&pass1.values),
        params: SmaParams { period: Some(m2) },
    };
    let pass2 = sma(&sma2_in).unwrap();

    // Copy the final “pass2” values straight into `out`.
    // pass2.values is already length = n, with the appropriate NaN prefix.
    out.copy_from_slice(&pass2.values);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn trima_avx512(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    if period <= 32 {
        unsafe { trima_avx512_short(data, period, first, out) }
    } else {
        unsafe { trima_avx512_long(data, period, first, out) }
    }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn trima_avx2(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    trima_scalar(data, period, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn trima_avx512_short(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    trima_scalar(data, period, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn trima_avx512_long(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    trima_scalar(data, period, first, out)
}

#[inline(always)]
pub fn trima_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    _w_ptr: *const f64,
    _inv_n: f64,
    out: &mut [f64],
) {
    trima_scalar(data, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn trima_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    _w_ptr: *const f64,
    _inv_n: f64,
    out: &mut [f64],
) {
    trima_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn trima_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    _w_ptr: *const f64,
    _inv_n: f64,
    out: &mut [f64],
) {
    trima_avx512(data, period, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn trima_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    _w_ptr: *const f64,
    _inv_n: f64,
    out: &mut [f64],
) {
    trima_avx512_short(data, period, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn trima_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    _w_ptr: *const f64,
    _inv_n: f64,
    out: &mut [f64],
) {
    trima_avx512_long(data, period, first, out)
}

#[derive(Debug, Clone)]
pub struct TrimaStream {
    // the overall TRIMA period
    period: usize,
    // the two intermediate SMA window‐sizes
    m1: usize,
    m2: usize,

    // first SMA window
    buffer1: Vec<f64>,
    sum1:    f64,
    head1:   usize,
    filled1: bool,

    // second SMA window
    buffer2: Vec<f64>,
    sum2:    f64,
    head2:   usize,
    filled2: bool,
}

impl TrimaStream {
    pub fn try_new(params: TrimaParams) -> Result<Self, TrimaError> {
        let period = params.period.unwrap_or(30);
        if period == 0 || period <= 3 {
            return Err(TrimaError::PeriodTooSmall { period });
        }
        // compute m₁ and m₂ exactly as in the “two‐pass” formula:
        //   m₁ = (period+1)/2,    m₂ = period−m₁+1
        let m1 = (period + 1) / 2;
        let m2 = period - m1 + 1;

        Ok(Self {
            period,
            m1,
            m2,
            buffer1: vec![f64::NAN; m1],
            sum1: 0.0,
            head1: 0,
            filled1: false,
            buffer2: vec![f64::NAN; m2],
            sum2: 0.0,
            head2: 0,
            filled2: false,
        })
    }

    /// Feed a single new raw price into the TRIMA‐stream.
    /// Returns `Some(trima_value)` only once enough data has been seen for both sub‐windows;
    /// otherwise returns `None` (which the test harness will compare as NaN).
    #[inline(always)]
    pub fn update(&mut self, x: f64) -> Option<f64> {
        // ──  STEP 1:  Update the m₁‐window (compute first‐stage SMA)  ──
        let old1 = self.buffer1[self.head1];
        self.buffer1[self.head1] = x;
        self.head1 = (self.head1 + 1) % self.m1;
        if !self.filled1 && self.head1 == 0 {
            self.filled1 = true;
        }
        // Adjust sum1, always ignoring NaNs:
        if !old1.is_nan() {
            self.sum1 -= old1;
        }
        if !x.is_nan() {
            self.sum1 += x;
        }
        // Once filled1 is true, we can compute SMA₁ = sum1 / m₁:
        let sma1 = if self.filled1 {
            Some(self.sum1 / (self.m1 as f64))
        
            } else {
            None
        };

        // ──  STEP 2:  Once we have an SMA₁, feed it into the m₂‐window (second pass)  ──
        if let Some(s1) = sma1 {
            let old2 = self.buffer2[self.head2];
            self.buffer2[self.head2] = s1;
            self.head2 = (self.head2 + 1) % self.m2;
            if !self.filled2 && self.head2 == 0 {
                self.filled2 = true;
            }
            if !old2.is_nan() {
                self.sum2 -= old2;
            }
            if !s1.is_nan() {
                self.sum2 += s1;
            }
            // Once filled2 == true, we can output TRIMA = sum2 / m₂
            if self.filled2 {
                return Some(self.sum2 / (self.m2 as f64));
            }
        }

        None
    }
}



#[derive(Clone, Debug)]
pub struct TrimaBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for TrimaBatchRange {
    fn default() -> Self {
        Self {
            period: (14, 100, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct TrimaBatchBuilder {
    range: TrimaBatchRange,
    kernel: Kernel,
}

impl TrimaBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline]
    pub fn period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.period = (start, end, step);
        self
    }
    #[inline]
    pub fn period_static(mut self, p: usize) -> Self {
        self.range.period = (p, p, 0);
        self
    }

    pub fn apply_slice(self, data: &[f64]) -> Result<TrimaBatchOutput, TrimaError> {
        trima_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<TrimaBatchOutput, TrimaError> {
        TrimaBatchBuilder::new().kernel(k).apply_slice(data)
    }

    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<TrimaBatchOutput, TrimaError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }

    pub fn with_default_candles(c: &Candles) -> Result<TrimaBatchOutput, TrimaError> {
        TrimaBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn trima_batch_with_kernel(
    data: &[f64],
    sweep: &TrimaBatchRange,
    k: Kernel,
) -> Result<TrimaBatchOutput, TrimaError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(TrimaError::InvalidPeriod {
                period: 0,
                data_len: 0,
            })
        }
    };

    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    trima_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct TrimaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<TrimaParams>,
    pub rows: usize,
    pub cols: usize,
}
impl TrimaBatchOutput {
    pub fn row_for_params(&self, p: &TrimaParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(14) == p.period.unwrap_or(14)
        })
    }

    pub fn values_for(&self, p: &TrimaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &TrimaBatchRange) -> Vec<TrimaParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }

    let periods = axis_usize(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(TrimaParams { period: Some(p) });
    }
    out
}

#[inline(always)]
pub fn trima_batch_slice(
    data: &[f64],
    sweep: &TrimaBatchRange,
    kern: Kernel,
) -> Result<TrimaBatchOutput, TrimaError> {
    trima_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn trima_batch_par_slice(
    data: &[f64],
    sweep: &TrimaBatchRange,
    kern: Kernel,
) -> Result<TrimaBatchOutput, TrimaError> {
    trima_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn trima_batch_inner(
    data: &[f64],
    sweep: &TrimaBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<TrimaBatchOutput, TrimaError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(TrimaError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(TrimaError::AllValuesNaN)?;
    let max_p = combos
        .iter()
        .map(|c| c.period.unwrap())
        .max()
        .unwrap();
    if data.len() - first < max_p {
        return Err(TrimaError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();
    let warm: Vec<usize> = combos.iter()
        .map(|c| first + c.period.unwrap())
        .collect();

    // ---------- 2. allocate rows×cols buffer and stamp NaN prefixes ----------
    let mut raw = make_uninit_matrix(rows, cols);
    unsafe { init_matrix_prefixes(&mut raw, cols, &warm) };

    // ---------- 3. closure that writes one row in-place ----------
    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period  = combos[row].period.unwrap();

        // cast *just this row* to &mut [f64]
        let out_row = core::slice::from_raw_parts_mut(
            dst_mu.as_mut_ptr() as *mut f64,
            dst_mu.len(),
        );

        match kern {
            Kernel::Scalar => trima_row_scalar(data, first, period, 0, core::ptr::null(), 1.0, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2   => trima_row_avx2  (data, first, period, 0, core::ptr::null(), 1.0, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => trima_row_avx512(data, first, period, 0, core::ptr::null(), 1.0, out_row),
            _ => unreachable!(),
        }
    };

    // ---------- 4. run every row (parallel or serial) ----------
    if parallel {

        #[cfg(not(target_arch = "wasm32"))] {

        raw.par_chunks_mut(cols)

                .enumerate()

                .for_each(|(row, slice)| do_row(row, slice));

        }

        #[cfg(target_arch = "wasm32")] {

        for (row, slice) in raw.chunks_mut(cols).enumerate() {

                    do_row(row, slice);

        }
        }
    } else {
        for (row, slice) in raw.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    // ---------- 5. convert Vec<MaybeUninit<f64>> → Vec<f64> ----------
    let values: Vec<f64> = unsafe { core::mem::transmute(raw) };

    Ok(TrimaBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

// Python bindings
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;

#[cfg(feature = "python")]
#[pyfunction(name = "trima")]
#[pyo3(signature = (data, period, kernel=None))]
/// Compute the Triangular Moving Average (TRIMA) of the input data.
///
/// TRIMA is a double-smoothed simple moving average that places more weight
/// on the middle portion of the smoothing period.
///
/// Parameters:
/// -----------
/// data : np.ndarray
///     Input data array (float64).
/// period : int
///     Window size for the moving average (must be > 3).
/// kernel : str, optional
///     Computation kernel to use: 'auto', 'scalar', 'avx2', 'avx512'.
///     Default is 'auto' which auto-detects the best available.
///
/// Returns:
/// --------
/// np.ndarray
///     Array of TRIMA values, same length as input.
///
/// Raises:
/// -------
/// ValueError
///     If inputs are invalid (period <= 3, period > data length, etc).
pub fn trima_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use numpy::{PyArray1, PyArrayMethods};

    let slice_in = data.as_slice()?; // zero-copy, read-only view

    // Parse kernel string to enum
    let kern = match kernel {
        None | Some("auto") => Kernel::Auto,
        Some("scalar") => Kernel::Scalar,
        Some("avx2") => Kernel::Avx2,
        Some("avx512") => Kernel::Avx512,
        Some(k) => return Err(PyValueError::new_err(format!("Unknown kernel: {}", k))),
    };

    // Build input struct
    let params = TrimaParams {
        period: Some(period),
    };
    let trima_in = TrimaInput::from_slice(slice_in, params);

    // Allocate NumPy output buffer
    let out_arr = unsafe { PyArray1::<f64>::new(py, [slice_in.len()], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? }; // safe: we own the array

    // Heavy lifting without the GIL
    py.allow_threads(|| -> Result<(), TrimaError> {
        let (data, period, m1, m2, first, chosen) = trima_prepare(&trima_in, kern)?;
        // prefix initialise exactly once
        let warmup = first + m1 + m2 - 1;
        slice_out[..warmup].fill(f64::NAN);
        trima_compute_into(data, period, m1, m2, first, chosen, slice_out);
        Ok(())
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(out_arr.into())
}

#[cfg(feature = "python")]
#[pyclass(name = "TrimaStream")]
pub struct TrimaStreamPy {
    stream: TrimaStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl TrimaStreamPy {
    #[new]
    fn new(period: Option<usize>) -> PyResult<Self> {
        let params = TrimaParams { period };
        let stream =
            TrimaStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(TrimaStreamPy { stream })
    }

    /// Updates the stream with a new value and returns the calculated TRIMA value.
    /// Returns `None` if the buffer is not yet full.
    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "trima_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]
/// Compute TRIMA for multiple period values in a single pass.
///
/// Parameters:
/// -----------
/// data : np.ndarray
///     Input data array (float64).
/// period_range : tuple
///     (start, end, step) for period values to compute.
/// kernel : str, optional
///     Computation kernel: 'auto', 'scalar', 'avx2', 'avx512'.
///     Default is 'auto' which auto-detects the best available.
///
/// Returns:
/// --------
/// dict
///     Dictionary with 'values' (2D array) and 'periods' array.
pub fn trima_batch_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let slice_in = data.as_slice()?;

    let sweep = TrimaBatchRange {
        period: period_range,
    };

    // Parse kernel string to enum
    let kern = match kernel {
        None | Some("auto") => Kernel::Auto,
        Some("scalar") => Kernel::ScalarBatch,
        Some("avx2") => Kernel::Avx2Batch,
        Some("avx512") => Kernel::Avx512Batch,
        Some(k) => return Err(PyValueError::new_err(format!("Unknown kernel: {}", k))),
    };

    // Heavy work without the GIL
    let output = py.allow_threads(|| -> Result<TrimaBatchOutput, TrimaError> {
        trima_batch_with_kernel(slice_in, &sweep, kern)
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Build output dict
    let dict = PyDict::new(py);
    
    // Convert values to NumPy array and reshape
    dict.set_item("values", output.values.into_pyarray(py).reshape((output.rows, output.cols))?)?;
    
    // Extract periods from combos
    dict.set_item(
        "periods",
        output.combos
            .iter()
            .map(|p| p.period.unwrap_or(30) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;

    Ok(dict)
}

// WASM bindings
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "wasm")]
#[wasm_bindgen]
/// Compute the Triangular Moving Average (TRIMA) of the input data.
/// 
/// # Arguments
/// * `data` - Input data array
/// * `period` - Window size for the moving average (must be > 3)
/// 
/// # Returns
/// Array of TRIMA values, same length as input
pub fn trima_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
    let params = TrimaParams { period: Some(period) };
    let input = TrimaInput::from_slice(data, params);

    trima_with_kernel(&input, Kernel::Scalar)
        .map(|o| o.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
/// Compute TRIMA for multiple period values in a single pass.
/// 
/// # Arguments
/// * `data` - Input data array
/// * `period_start`, `period_end`, `period_step` - Period range parameters
/// 
/// # Returns
/// Flattened array of values (row-major order)
pub fn trima_batch_js(
    data: &[f64],
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = TrimaBatchRange {
        period: (period_start, period_end, period_step),
    };

    // Use the existing batch function with parallel=false for WASM
    trima_batch_inner(data, &sweep, Kernel::Scalar, false)
        .map(|output| output.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
/// Get metadata about batch computation.
/// 
/// # Arguments
/// * Period range parameters (same as trima_batch_js)
/// 
/// # Returns
/// Array containing period values
pub fn trima_batch_metadata_js(
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = TrimaBatchRange {
        period: (period_start, period_end, period_step),
    };

    let combos = expand_grid(&sweep);
    let metadata: Vec<f64> = combos
        .iter()
        .map(|combo| combo.period.unwrap_or(30) as f64)
        .collect();

    Ok(metadata)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;
    use crate::utilities::enums::Kernel;

    fn check_trima_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = TrimaParams { period: None };
        let input = TrimaInput::from_candles(&candles, "close", default_params);
        let output = trima_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        let params_period_10 = TrimaParams { period: Some(10) };
        let input2 = TrimaInput::from_candles(&candles, "hl2", params_period_10);
        let output2 = trima_with_kernel(&input2, kernel)?;
        assert_eq!(output2.values.len(), candles.close.len());

        let params_custom = TrimaParams { period: Some(14) };
        let input3 = TrimaInput::from_candles(&candles, "hlc3", params_custom);
        let output3 = trima_with_kernel(&input3, kernel)?;
        assert_eq!(output3.values.len(), candles.close.len());

        Ok(())
    }

    fn check_trima_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let close_prices = &candles.close;
        let params = TrimaParams { period: Some(30) };
        let input = TrimaInput::from_candles(&candles, "close", params);
        let trima_result = trima_with_kernel(&input, kernel)?;

        assert_eq!(trima_result.values.len(), close_prices.len(), "TRIMA output length should match input data length");
        let expected_last_five_trima = [
            59957.916666666664,
            59846.770833333336,
            59750.620833333334,
            59665.2125,
            59581.612499999996,
        ];
        assert!(trima_result.values.len() >= 5, "Not enough TRIMA values for the test");
        let start_index = trima_result.values.len() - 5;
        let result_last_five_trima = &trima_result.values[start_index..];
        for (i, &value) in result_last_five_trima.iter().enumerate() {
            let expected_value = expected_last_five_trima[i];
            assert!(
                (value - expected_value).abs() < 1e-6,
                "[{}] TRIMA value mismatch at index {}: expected {}, got {}",
                test_name,
                i,
                expected_value,
                value
            );
        }
        let period = input.params.period.unwrap_or(14);
        for i in 0..(period - 1) {
            assert!(
                trima_result.values[i].is_nan(),
                "[{}] Expected NaN at early index {} for TRIMA, got {}",
                test_name,
                i,
                trima_result.values[i]
            );
        }
        Ok(())
    }

    fn check_trima_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = TrimaInput::with_default_candles(&candles);
        match input.data {
            TrimaData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected TrimaData::Candles"),
        }
        let output = trima_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_trima_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = TrimaParams { period: Some(0) };
        let input = TrimaInput::from_slice(&input_data, params);
        let res = trima_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] TRIMA should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_trima_period_too_small(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0, 40.0];
        let params = TrimaParams { period: Some(3) };
        let input = TrimaInput::from_slice(&input_data, params);
        let res = trima_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] TRIMA should fail with period <= 3",
            test_name
        );
        Ok(())
    }

    fn check_trima_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = TrimaParams { period: Some(10) };
        let input = TrimaInput::from_slice(&data_small, params);
        let res = trima_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] TRIMA should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_trima_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = TrimaParams { period: Some(14) };
        let input = TrimaInput::from_slice(&single_point, params);
        let res = trima_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] TRIMA should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_trima_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let first_params = TrimaParams { period: Some(14) };
        let first_input = TrimaInput::from_candles(&candles, "close", first_params);
        let first_result = trima_with_kernel(&first_input, kernel)?;

        let second_params = TrimaParams { period: Some(10) };
        let second_input = TrimaInput::from_slice(&first_result.values, second_params);
        let second_result = trima_with_kernel(&second_input, kernel)?;

        assert_eq!(second_result.values.len(), first_result.values.len());
        for val in &second_result.values[240..] {
            assert!(val.is_finite());
        }
        Ok(())
    }

    fn check_trima_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = TrimaInput::from_candles(
            &candles,
            "close",
            TrimaParams { period: Some(14) },
        );
        let res = trima_with_kernel(&input, kernel)?;
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
        Ok(())
    }

    fn check_trima_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let period = 14;

        let input = TrimaInput::from_candles(
            &candles,
            "close",
            TrimaParams { period: Some(period) },
        );
        let batch_output = trima_with_kernel(&input, kernel)?.values;

        let mut stream = TrimaStream::try_new(TrimaParams { period: Some(period) })?;

        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(trima_val) => stream_values.push(trima_val),
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
                diff < 1e-8,
                "[{}] TRIMA streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    macro_rules! generate_all_trima_tests {
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
    fn check_trima_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Test with multiple parameter combinations to increase coverage
        let test_periods = vec![4, 10, 14, 30, 50, 100];
        let test_sources = vec!["close", "open", "high", "low", "hl2", "hlc3", "ohlc4"];

        for period in test_periods {
            for source in &test_sources {
                let params = TrimaParams { period: Some(period) };
                let input = TrimaInput::from_candles(&candles, source, params);
                let output = trima_with_kernel(&input, kernel)?;

                // Check every value for poison patterns
                for (i, &val) in output.values.iter().enumerate() {
                    // Skip NaN values as they're expected in the warmup period
                    if val.is_nan() {
                        continue;
                    }

                    let bits = val.to_bits();

                    // Check for alloc_with_nan_prefix poison (0x11111111_11111111)
                    if bits == 0x11111111_11111111 {
                        panic!(
                            "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} (period={}, source={})",
                            test_name, val, bits, i, period, source
                        );
                    }

                    // Check for init_matrix_prefixes poison (0x22222222_22222222)
                    if bits == 0x22222222_22222222 {
                        panic!(
                            "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} (period={}, source={})",
                            test_name, val, bits, i, period, source
                        );
                    }

                    // Check for make_uninit_matrix poison (0x33333333_33333333)
                    if bits == 0x33333333_33333333 {
                        panic!(
                            "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} (period={}, source={})",
                            test_name, val, bits, i, period, source
                        );
                    }
                }
            }
        }

        Ok(())
    }

    // Release mode stub - does nothing
    #[cfg(not(debug_assertions))]
    fn check_trima_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    generate_all_trima_tests!(
        check_trima_partial_params,
        check_trima_accuracy,
        check_trima_default_candles,
        check_trima_zero_period,
        check_trima_period_exceeds_length,
        check_trima_period_too_small,
        check_trima_very_small_dataset,
        check_trima_reinput,
        check_trima_nan_handling,
        check_trima_streaming,
        check_trima_no_poison
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = TrimaBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = TrimaParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        // You can use expected values as appropriate for TRIMA.
        let expected = [
            59957.916666666664,
            59846.770833333336,
            59750.620833333334,
            59665.2125,
            59581.612499999996,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-6,
                "[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
            );
        }
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
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test batch with multiple parameter range combinations
        let period_ranges = vec![
            (4, 20, 4),    // Small periods
            (20, 50, 10),  // Medium periods
            (50, 100, 25), // Large periods
            (5, 15, 1),    // Dense small range
        ];

        let test_sources = vec!["close", "open", "high", "low", "hl2", "hlc3", "ohlc4"];

        for (start, end, step) in period_ranges {
            for source in &test_sources {
                let output = TrimaBatchBuilder::new()
                    .kernel(kernel)
                    .period_range(start, end, step)
                    .apply_candles(&c, source)?;

                // Check every value in the entire batch matrix for poison patterns
                for (idx, &val) in output.values.iter().enumerate() {
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
                            "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (flat index {}) with period_range({},{},{}) source={}",
                            test, val, bits, row, col, idx, start, end, step, source
                        );
                    }

                    // Check for init_matrix_prefixes poison (0x22222222_22222222)
                    if bits == 0x22222222_22222222 {
                        panic!(
                            "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (flat index {}) with period_range({},{},{}) source={}",
                            test, val, bits, row, col, idx, start, end, step, source
                        );
                    }

                    // Check for make_uninit_matrix poison (0x33333333_33333333)
                    if bits == 0x33333333_33333333 {
                        panic!(
                            "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (flat index {}) with period_range({},{},{}) source={}",
                            test, val, bits, row, col, idx, start, end, step, source
                        );
                    }
                }
            }
        }

        Ok(())
    }

    // Release mode stub - does nothing
    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_no_poison);
}
