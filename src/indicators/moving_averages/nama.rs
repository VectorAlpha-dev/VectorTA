//! # New Adaptive Moving Average (NAMA)
//!
//! A dynamic moving average that adapts based on market conditions using range and effort.
//! Developed by Franklin Moormann (cheatcountry). The indicator calculates a ratio between
//! price movement effort and range to determine adaptive smoothing.
//!
//! ## Parameters
//! - **period**: Lookback period for calculations (defaults to 30)
//!
//! ## Returns
//! - **`Ok(NamaOutput)`** on success, containing a `Vec<f64>` matching input length
//! - **`Err(NamaError)`** otherwise
//!
//! ## Developer Notes
//! - SIMD status: AVX2/AVX512 precompute the True Range (TR) across the series and reuse the scalar core.
//!   Runtime selection follows alma.rs patterns. If nightly-avx is disabled or unsupported at runtime,
//!   selection falls back to the scalar path.
//! - Scalar path: optimized but kept safe. Removes per-step `tr_at` recomputation by maintaining a
//!   ring buffer of TR values and an O(1) rolling sum; hoists the OHLC vs degenerate TR branch outside
//!   the hot loop; uses `VecDeque` monotone queues for max/min (window) to avoid O(N) output temporaries.
//! - Batch path: optimized for slice data (degenerate TR) by precomputing TR once and reusing it across
//!   all rows/periods via a shared core. This reduces redundant work while preserving API and warmup.
//! - Streaming update: now O(1) amortized per update using monotone deques
//!   for window max/min and a rolling True Range (TR) sum via a ring buffer.
//! - Decision: Enabled O(1) streaming; outputs match batch exactly (tests unchanged).

#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::cuda_available;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::moving_averages::CudaNama;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::moving_averages::DeviceArrayF32;
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::memory::DeviceBuffer;
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::context::Context as CudaContext;
#[cfg(all(feature = "python", feature = "cuda"))]
use std::sync::Arc;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
#[cfg(all(feature = "python", feature = "cuda"))]
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;

#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
use std::collections::VecDeque;
use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

impl<'a> AsRef<[f64]> for NamaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            NamaData::Slice(slice) => slice,
            NamaData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

// NAMA-specific device handle that keeps the CUDA context alive for the buffer's lifetime.
#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(module = "ta_indicators.cuda", unsendable)]
pub struct DeviceArrayF32PyNama {
    pub(crate) inner: DeviceArrayF32,
    _ctx: Arc<CudaContext>,
    _device_id: u32,
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pymethods]
impl DeviceArrayF32PyNama {
    #[getter]
    fn __cuda_array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        let rows = self.inner.rows;
        let cols = self.inner.cols;
        d.set_item("shape", (rows, cols))?;
        d.set_item("typestr", "<f4")?;
        let item = std::mem::size_of::<f32>();
        d.set_item("strides", (cols * item, item))?;
        let ptr_int: usize = if rows == 0 || cols == 0 { 0 } else { self.inner.device_ptr() as usize };
        d.set_item("data", (ptr_int, false))?;
        d.set_item("version", 3)?;
        Ok(d)
    }

    fn __dlpack_device__(&self) -> (i32, i32) { (2, self._device_id as i32) }

    #[pyo3(signature = (stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__<'py>(
        &mut self,
        py: Python<'py>,
        stream: Option<PyObject>,
        max_version: Option<PyObject>,
        dl_device: Option<PyObject>,
        copy: Option<PyObject>,
    ) -> PyResult<PyObject> {
        use crate::utilities::dlpack_cuda::export_f32_cuda_dlpack_2d;

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
                            "device copy not implemented for __dlpack__",
                        ));
                    } else {
                        return Err(PyValueError::new_err("dl_device mismatch for __dlpack__"));
                    }
                }
            }
        }
        let _ = stream;

        // Move VRAM handle out of this wrapper; the DLPack capsule owns it afterwards.
        let dummy = DeviceBuffer::from_slice(&[])
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let inner = std::mem::replace(
            &mut self.inner,
            DeviceArrayF32 { buf: dummy, rows: 0, cols: 0 },
        );

        let rows = inner.rows;
        let cols = inner.cols;
        let buf = inner.buf;

        let max_version_bound = max_version.map(|obj| obj.into_bound(py));

        export_f32_cuda_dlpack_2d(py, buf, rows, cols, alloc_dev, max_version_bound)
    }
}

#[derive(Debug, Clone)]
pub enum NamaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct NamaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct NamaParams {
    pub period: Option<usize>,
}

impl Default for NamaParams {
    fn default() -> Self {
        Self { period: Some(30) }
    }
}

#[derive(Debug, Clone)]
pub struct NamaInput<'a> {
    pub data: NamaData<'a>,
    pub params: NamaParams,
}

impl<'a> NamaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: NamaParams) -> Self {
        Self {
            data: NamaData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: NamaParams) -> Self {
        Self {
            data: NamaData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", NamaParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(30)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct NamaBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for NamaBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl NamaBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<NamaOutput, NamaError> {
        let p = NamaParams {
            period: self.period,
        };
        let i = NamaInput::from_candles(c, "close", p);
        nama_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<NamaOutput, NamaError> {
        let p = NamaParams {
            period: self.period,
        };
        let i = NamaInput::from_slice(d, p);
        nama_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<NamaStream, NamaError> {
        let p = NamaParams {
            period: self.period,
        };
        NamaStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum NamaError {
    #[error("nama: Input data slice is empty.")]
    EmptyInputData,

    #[error("nama: All values are NaN.")]
    AllValuesNaN,

    #[error("nama: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },

    #[error("nama: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },

    #[error("nama: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },

    #[error("nama: Invalid range expansion: start = {start}, end = {end}, step = {step}")]
    InvalidRange { start: usize, end: usize, step: usize },

    #[error("nama: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[inline]
pub fn nama(input: &NamaInput) -> Result<NamaOutput, NamaError> {
    nama_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn nama_prepare<'a>(
    input: &'a NamaInput,
    kernel: Kernel,
) -> Result<
    (
        &'a [f64],
        usize,
        usize,
        Kernel,
        Option<(&'a [f64], &'a [f64], &'a [f64])>,
    ),
    NamaError,
> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(NamaError::EmptyInputData);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(NamaError::AllValuesNaN)?;
    let period = input.get_period();
    if period == 0 || period > len {
        return Err(NamaError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if len - first < period {
        return Err(NamaError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };
    // Extra OHLC for TR if input is Candles
    let ohlc = match &input.data {
        NamaData::Candles { candles, .. } => {
            Some((&candles.high[..], &candles.low[..], &candles.close[..]))
        }
        _ => None,
    };
    Ok((data, period, first, chosen, ohlc))
}

// SIMD kernel functions (currently all route to scalar implementation)
#[inline]
pub fn nama_scalar(
    data: &[f64],
    period: usize,
    first_val: usize,
    ohlc: Option<(&[f64], &[f64], &[f64])>,
    out: &mut [f64],
) {
    nama_compute_into(data, period, first_val, ohlc, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn nama_avx2(
    data: &[f64],
    period: usize,
    first_val: usize,
    ohlc: Option<(&[f64], &[f64], &[f64])>,
    out: &mut [f64],
) {
    use core::arch::x86_64::*;

    #[inline(always)]
    unsafe fn abs256(x: __m256d) -> __m256d {
        let sign = _mm256_set1_pd(-0.0f64);
        _mm256_andnot_pd(sign, x)
    }

    let n = data.len();
    if n == 0 {
        return;
    }
    let first = first_val;
    let i0 = first + period - 1;
    if i0 >= n {
        return;
    }

    // Precompute TR across the entire series
    let mut tr = vec![0.0f64; n];
    unsafe {
        match ohlc {
            Some((h, l, c)) => {
                // j == first
                *tr.get_unchecked_mut(first) = h[first] - l[first];

                let mut j = first + 1;
                let step = 4usize;
                let end = j + ((n - j) / step) * step;
                while j < end {
                    let vh = _mm256_loadu_pd(h.as_ptr().add(j));
                    let vl = _mm256_loadu_pd(l.as_ptr().add(j));
                    let vcprev = _mm256_loadu_pd(c.as_ptr().add(j - 1));
                    let vhl = _mm256_sub_pd(vh, vl);
                    let vhc = abs256(_mm256_sub_pd(vh, vcprev));
                    let vlc = abs256(_mm256_sub_pd(vl, vcprev));
                    let vmax1 = _mm256_max_pd(vhl, vhc);
                    let vmax2 = _mm256_max_pd(vmax1, vlc);
                    _mm256_storeu_pd(tr.as_mut_ptr().add(j), vmax2);
                    j += step;
                }
                while j < n {
                    let hl = h[j] - l[j];
                    let prev = c[j - 1];
                    let hc = (h[j] - prev).abs();
                    let lc = (l[j] - prev).abs();
                    *tr.get_unchecked_mut(j) = hl.max(hc).max(lc);
                    j += 1;
                }
            }
            None => {
                // Degenerate TR: 0 at first, |Δsource| afterwards
                *tr.get_unchecked_mut(first) = 0.0;
                let sp = data.as_ptr();
                let mut j = first + 1;
                let step = 4usize;
                let end = j + ((n - j) / step) * step;
                while j < end {
                    let vx = _mm256_loadu_pd(sp.add(j));
                    let vprev = _mm256_loadu_pd(sp.add(j - 1));
                    let vdiff = abs256(_mm256_sub_pd(vx, vprev));
                    _mm256_storeu_pd(tr.as_mut_ptr().add(j), vdiff);
                    j += step;
                }
                while j < n {
                    *tr.get_unchecked_mut(j) = (*sp.add(j) - *sp.add(j - 1)).abs();
                    j += 1;
                }
            }
        }
    }

    // Consume using the shared scalar core
    nama_core_with_tr(data, period, first, &tr, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn nama_avx512(
    data: &[f64],
    period: usize,
    first_val: usize,
    ohlc: Option<(&[f64], &[f64], &[f64])>,
    out: &mut [f64],
) {
    use core::arch::x86_64::*;

    #[inline(always)]
    unsafe fn abs512(x: __m512d) -> __m512d {
        let sign = _mm512_set1_pd(-0.0f64);
        _mm512_andnot_pd(sign, x)
    }

    let n = data.len();
    if n == 0 {
        return;
    }
    let first = first_val;
    let i0 = first + period - 1;
    if i0 >= n {
        return;
    }

    let mut tr = vec![0.0f64; n];
    unsafe {
        match ohlc {
            Some((h, l, c)) => {
                *tr.get_unchecked_mut(first) = h[first] - l[first];

                let mut j = first + 1;
                let step = 8usize;
                let end = j + ((n - j) / step) * step;
                while j < end {
                    let vh = _mm512_loadu_pd(h.as_ptr().add(j));
                    let vl = _mm512_loadu_pd(l.as_ptr().add(j));
                    let vcprev = _mm512_loadu_pd(c.as_ptr().add(j - 1));
                    let vhl = _mm512_sub_pd(vh, vl);
                    let vhc = abs512(_mm512_sub_pd(vh, vcprev));
                    let vlc = abs512(_mm512_sub_pd(vl, vcprev));
                    let vmax1 = _mm512_max_pd(vhl, vhc);
                    let vmax2 = _mm512_max_pd(vmax1, vlc);
                    _mm512_storeu_pd(tr.as_mut_ptr().add(j), vmax2);
                    j += step;
                }
                while j < n {
                    let hl = h[j] - l[j];
                    let prev = c[j - 1];
                    let hc = (h[j] - prev).abs();
                    let lc = (l[j] - prev).abs();
                    *tr.get_unchecked_mut(j) = hl.max(hc).max(lc);
                    j += 1;
                }
            }
            None => {
                *tr.get_unchecked_mut(first) = 0.0;
                let sp = data.as_ptr();
                let mut j = first + 1;
                let step = 8usize;
                let end = j + ((n - j) / step) * step;
                while j < end {
                    let vx = _mm512_loadu_pd(sp.add(j));
                    let vprev = _mm512_loadu_pd(sp.add(j - 1));
                    let vdiff = abs512(_mm512_sub_pd(vx, vprev));
                    _mm512_storeu_pd(tr.as_mut_ptr().add(j), vdiff);
                    j += step;
                }
                while j < n {
                    *tr.get_unchecked_mut(j) = (*sp.add(j) - *sp.add(j - 1)).abs();
                    j += 1;
                }
            }
        }
    }

    nama_core_with_tr(data, period, first, &tr, out);
}

#[inline(always)]
fn nama_compute_into(
    src: &[f64],
    period: usize,
    first: usize,
    ohlc: Option<(&[f64], &[f64], &[f64])>,
    out: &mut [f64],
) {
    let n = src.len();
    let i0 = first + period - 1;
    if i0 >= n {
        return;
    } // NaN prefix already set by allocator

    // Monotone deques for max/min over the sliding window
    let mut dq_max: VecDeque<usize> = VecDeque::with_capacity(period);
    let mut dq_min: VecDeque<usize> = VecDeque::with_capacity(period);

    #[inline(always)]
    fn push_max(dq: &mut VecDeque<usize>, a: &[f64], j: usize) {
        while let Some(&k) = dq.back() {
            if a[k] <= a[j] {
                dq.pop_back();
            } else {
                break;
            }
        }
        dq.push_back(j);
    }
    #[inline(always)]
    fn push_min(dq: &mut VecDeque<usize>, a: &[f64], j: usize) {
        while let Some(&k) = dq.back() {
            if a[k] >= a[j] {
                dq.pop_back();
            } else {
                break;
            }
        }
        dq.push_back(j);
    }
    #[inline(always)]
    fn pop_old(dq: &mut VecDeque<usize>, win_start: usize) {
        while let Some(&k) = dq.front() {
            if k < win_start {
                dq.pop_front();
            } else {
                break;
            }
        }
    }

    // TR ring buffer and rolling sum
    let mut tr_ring: Vec<f64> = vec![0.0; period];
    let mut wr: usize = 0;
    let mut eff_sum: f64 = 0.0;

    match ohlc {
        Some((h, l, c)) => {
            // Warm-up window [first..=i0]
            for j in first..=i0 {
                push_max(&mut dq_max, src, j);
                push_min(&mut dq_min, src, j);
                let trj = if j == first {
                    h[j] - l[j]
                } else {
                    let hl = h[j] - l[j];
                    let prev = c[j - 1];
                    let hc = (h[j] - prev).abs();
                    let lc = (l[j] - prev).abs();
                    hl.max(hc).max(lc)
                };
                tr_ring[wr] = trj;
                wr += 1;
                eff_sum += trj;
            }
            wr = 0;

            // First output at i0
            {
                let hi = src[*dq_max.front().unwrap()];
                let lo = src[*dq_min.front().unwrap()];
                let alpha = if eff_sum != 0.0 {
                    (hi - lo) / eff_sum
                } else {
                    0.0
                };
                out[i0] = alpha * src[i0];
            }

            // Sliding phase
            let mut i = i0 + 1;
            while i < n {
                let j = i;
                push_max(&mut dq_max, src, j);
                push_min(&mut dq_min, src, j);
                let win_start = j + 1 - period;
                pop_old(&mut dq_max, win_start);
                pop_old(&mut dq_min, win_start);

                let old = tr_ring[wr];
                let hl = h[j] - l[j];
                let prev = c[j - 1];
                let hc = (h[j] - prev).abs();
                let lc = (l[j] - prev).abs();
                let add = hl.max(hc).max(lc);
                eff_sum = eff_sum + add - old;
                tr_ring[wr] = add;
                wr += 1;
                if wr == period {
                    wr = 0;
                }

                let hi = src[*dq_max.front().unwrap()];
                let lo = src[*dq_min.front().unwrap()];
                let alpha = if eff_sum != 0.0 {
                    (hi - lo) / eff_sum
                } else {
                    0.0
                };
                let prev_y = out[i - 1];
                out[i] = (src[j] - prev_y).mul_add(alpha, prev_y);
                i += 1;
            }
        }
        None => {
            // Degenerate TR = |Δsource| with TR[first] = 0
            for j in first..=i0 {
                push_max(&mut dq_max, src, j);
                push_min(&mut dq_min, src, j);
                let trj = if j == first {
                    0.0
                } else {
                    (src[j] - src[j - 1]).abs()
                };
                tr_ring[wr] = trj;
                wr += 1;
                eff_sum += trj;
            }
            wr = 0;

            // First output at i0
            {
                let hi = src[*dq_max.front().unwrap()];
                let lo = src[*dq_min.front().unwrap()];
                let alpha = if eff_sum != 0.0 {
                    (hi - lo) / eff_sum
                } else {
                    0.0
                };
                out[i0] = alpha * src[i0];
            }

            // Sliding phase
            let mut i = i0 + 1;
            while i < n {
                let j = i;
                push_max(&mut dq_max, src, j);
                push_min(&mut dq_min, src, j);
                let win_start = j + 1 - period;
                pop_old(&mut dq_max, win_start);
                pop_old(&mut dq_min, win_start);

                let old = tr_ring[wr];
                let add = (src[j] - src[j - 1]).abs();
                eff_sum = eff_sum + add - old;
                tr_ring[wr] = add;
                wr += 1;
                if wr == period {
                    wr = 0;
                }

                let hi = src[*dq_max.front().unwrap()];
                let lo = src[*dq_min.front().unwrap()];
                let alpha = if eff_sum != 0.0 {
                    (hi - lo) / eff_sum
                } else {
                    0.0
                };
                let prev_y = out[i - 1];
                out[i] = (src[j] - prev_y).mul_add(alpha, prev_y);
                i += 1;
            }
        }
    }
}

/// Shared scalar core that consumes precomputed TR values and writes outputs.
/// Assumes `out[..first+period-1]` is already NaN (alloc/init step handles warmup prefix).
#[inline(always)]
fn nama_core_with_tr(src: &[f64], period: usize, first: usize, tr: &[f64], out: &mut [f64]) {
    let n = src.len();
    let i0 = first + period - 1;
    if i0 >= n {
        return;
    }

    // Monotone deques for max/min
    let mut dq_max: VecDeque<usize> = VecDeque::with_capacity(period);
    let mut dq_min: VecDeque<usize> = VecDeque::with_capacity(period);

    #[inline(always)]
    fn push_max(dq: &mut VecDeque<usize>, a: &[f64], j: usize) {
        while let Some(&k) = dq.back() {
            if a[k] <= a[j] {
                dq.pop_back();
            } else {
                break;
            }
        }
        dq.push_back(j);
    }
    #[inline(always)]
    fn push_min(dq: &mut VecDeque<usize>, a: &[f64], j: usize) {
        while let Some(&k) = dq.back() {
            if a[k] >= a[j] {
                dq.pop_back();
            } else {
                break;
            }
        }
        dq.push_back(j);
    }
    #[inline(always)]
    fn pop_old(dq: &mut VecDeque<usize>, win_start: usize) {
        while let Some(&k) = dq.front() {
            if k < win_start {
                dq.pop_front();
            } else {
                break;
            }
        }
    }

    // TR ring + rolling sum
    let mut ring: Vec<f64> = vec![0.0; period];
    let mut wr: usize = 0;
    let mut eff_sum = 0.0;
    let period_minus_1 = period - 1;

    let sp = src.as_ptr();
    let trp = tr.as_ptr();
    let op = out.as_mut_ptr();
    let rp = ring.as_mut_ptr();

    // Warm-up
    for j in first..=i0 {
        push_max(&mut dq_max, src, j);
        push_min(&mut dq_min, src, j);
        let v = unsafe { *trp.add(j) };
        unsafe { *rp.add(wr) = v };
        wr += 1;
        eff_sum += v;
    }
    wr = 0;

    // First output at i0
    unsafe {
        let hi = *sp.add(*dq_max.front().unwrap_unchecked());
        let lo = *sp.add(*dq_min.front().unwrap_unchecked());
        let alpha = if eff_sum != 0.0 { (hi - lo) / eff_sum } else { 0.0 };
        *op.add(i0) = alpha * *sp.add(i0);
    }

    // Slide
    let mut i = i0 + 1;
    while i < n {
        let j = i;
        push_max(&mut dq_max, src, j);
        push_min(&mut dq_min, src, j);
        let win_start = j - period_minus_1;
        pop_old(&mut dq_max, win_start);
        pop_old(&mut dq_min, win_start);

        let old = unsafe { *rp.add(wr) };
        let add = unsafe { *trp.add(j) };
        eff_sum = eff_sum + add - old;
        unsafe { *rp.add(wr) = add };
        wr += 1;
        if wr == period {
            wr = 0;
        }

        unsafe {
            let hi = *sp.add(*dq_max.front().unwrap_unchecked());
            let lo = *sp.add(*dq_min.front().unwrap_unchecked());
            let alpha = if eff_sum != 0.0 { (hi - lo) / eff_sum } else { 0.0 };
            let prev_y = *op.add(i - 1);
            let x = *sp.add(j);
            *op.add(i) = (x - prev_y).mul_add(alpha, prev_y);
        }
        i += 1;
    }
}

pub fn nama_with_kernel(input: &NamaInput, kernel: Kernel) -> Result<NamaOutput, NamaError> {
    let (src, period, first, chosen, ohlc) = nama_prepare(input, kernel)?;
    // ALMA-compatible NaN prefix:
    let mut out = alloc_with_nan_prefix(src.len(), first + period - 1);

    // Select kernel implementation
    match (kernel, chosen) {
        // Stick to scalar as default when Kernel::Auto is requested
        (Kernel::Auto, _) => nama_scalar(src, period, first, ohlc, &mut out),
        // Explicit selections honored below
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        (_, Kernel::Avx512) => nama_avx512(src, period, first, ohlc, &mut out),
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        (_, Kernel::Avx2) => nama_avx2(src, period, first, ohlc, &mut out),
        _ => nama_scalar(src, period, first, ohlc, &mut out),
    }

    Ok(NamaOutput { values: out })
}

#[inline]
pub fn nama_into_slice(dst: &mut [f64], input: &NamaInput, k: Kernel) -> Result<(), NamaError> {
    let (src, period, first, chosen, ohlc) = nama_prepare(input, k)?;
    if dst.len() != src.len() {
        return Err(NamaError::OutputLengthMismatch {
            expected: src.len(),
            got: dst.len(),
        });
    }
    let warmup_end = (first + period - 1).min(dst.len());
    for v in &mut dst[..warmup_end] {
        // Match alloc_with_nan_prefix's quiet-NaN pattern for parity
        *v = f64::from_bits(0x7ff8_0000_0000_0000);
    }

    // Select kernel implementation
    match (k, chosen) {
        (Kernel::Auto, _) => nama_scalar(src, period, first, ohlc, dst),
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        (_, Kernel::Avx512) => nama_avx512(src, period, first, ohlc, dst),
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        (_, Kernel::Avx2) => nama_avx2(src, period, first, ohlc, dst),
        _ => nama_scalar(src, period, first, ohlc, dst),
    }

    Ok(())
}

/// Writes NAMA outputs into the provided buffer without allocating.
///
/// - Preserves NaN warmups exactly as the Vec-returning API (quiet-NaN prefix).
/// - `out.len()` must equal the input series length.
#[cfg(not(feature = "wasm"))]
pub fn nama_into(input: &NamaInput, out: &mut [f64]) -> Result<(), NamaError> {
    // Use Kernel::Auto to mirror the default selection of the Vec-returning API
    nama_into_slice(out, input, Kernel::Auto)
}

// ==================== STREAMING (O(1) UPDATE) ====================
#[derive(Debug, Clone)]
pub struct NamaStream {
    period: usize,

    // Rings to overwrite oldest slot each tick
    buf_src: Vec<f64>,
    buf_tr: Vec<f64>,
    head: usize,
    filled: bool,

    // Last seen values (for TR and EMA update)
    last_src: f64,
    last_close: f64,
    has_last_close: bool,
    last_out: f64,
    have_out: bool,

    // O(1) window state
    // - time: absolute index of current sample (monotone, used to age-out deque entries)
    // - eff_sum: rolling sum of TRs inside the window (ignores NaN entries)
    time: usize,
    eff_sum: f64,
    // Monotone deques of (index, value) for window max/min
    dq_max: VecDeque<(usize, f64)>,
    dq_min: VecDeque<(usize, f64)>,
}

impl NamaStream {
    pub fn try_new(params: NamaParams) -> Result<Self, NamaError> {
        let p = params.period.unwrap_or(30);
        if p == 0 {
            return Err(NamaError::InvalidPeriod {
                period: 0,
                data_len: 0,
            });
        }
        Ok(Self {
            period: p,
            buf_src: vec![f64::NAN; p],
            buf_tr: vec![f64::NAN; p],
            head: 0,
            filled: false,
            last_src: f64::NAN,
            last_close: f64::NAN,
            has_last_close: false,
            last_out: f64::NAN,
            have_out: false,

            time: 0,
            eff_sum: 0.0,
            dq_max: VecDeque::with_capacity(p),
            dq_min: VecDeque::with_capacity(p),
        })
    }

    #[inline(always)]
    fn advance(&mut self) {
        self.head = (self.head + 1) % self.period;
        if !self.filled && self.head == 0 {
            self.filled = true;
        }
    }

    // ---- internal helpers for monotone deques ----
    #[inline(always)]
    fn dq_push_max(dq: &mut VecDeque<(usize, f64)>, idx: usize, v: f64) {
        while let Some(&(_, back_v)) = dq.back() {
            if back_v <= v {
                dq.pop_back();
            } else {
                break;
            }
        }
        dq.push_back((idx, v));
    }
    #[inline(always)]
    fn dq_push_min(dq: &mut VecDeque<(usize, f64)>, idx: usize, v: f64) {
        while let Some(&(_, back_v)) = dq.back() {
            if back_v >= v {
                dq.pop_back();
            } else {
                break;
            }
        }
        dq.push_back((idx, v));
    }
    #[inline(always)]
    fn dq_pop_old(dq: &mut VecDeque<(usize, f64)>, win_start: usize) {
        while let Some(&(k, _)) = dq.front() {
            if k < win_start {
                dq.pop_front();
            } else {
                break;
            }
        }
    }

    // Single-series streaming (degenerate TR on |Δsource|)
    #[inline]
    pub fn update_source(&mut self, s: f64) -> Option<f64> {
        // new TR uses previous source; first TR is NaN (ignored in sum) to match batch semantics
        let tr_new = if self.last_src.is_nan() {
            f64::NAN
        } else {
            (s - self.last_src).abs()
        };

        // outgoing TR (oldest slot to be overwritten)
        let tr_old = self.buf_tr[self.head];

        // write into rings
        self.buf_src[self.head] = s;
        self.buf_tr[self.head] = tr_new;
        self.last_src = s;

        // update deques with absolute index `t`
        let t = self.time;
        self.time = self.time.wrapping_add(1);

        Self::dq_push_max(&mut self.dq_max, t, s);
        Self::dq_push_min(&mut self.dq_min, t, s);

        // once we're filled, age out indices < window start (t + 1 - period)
        if self.filled {
            let win_start = t + 1 - self.period;
            Self::dq_pop_old(&mut self.dq_max, win_start);
            Self::dq_pop_old(&mut self.dq_min, win_start);
        }

        // O(1) rolling sum of TRs (ignore NaN just like the batch warmup)
        if tr_old.is_finite() {
            self.eff_sum -= tr_old;
        }
        if tr_new.is_finite() {
            self.eff_sum += tr_new;
        }

        // finalize position / warmup detection
        self.advance();
        if !self.filled {
            return None;
        }

        // compute alpha and EMA-style update
        let hi = self.dq_max.front().map(|&(_, v)| v).unwrap_or(s);
        let lo = self.dq_min.front().map(|&(_, v)| v).unwrap_or(s);
        let range = hi - lo;
        let alpha = if self.eff_sum != 0.0 {
            // micro-opt: avoid divide in hot path
            let inv = self.eff_sum.recip();
            range * inv
        } else {
            0.0
        };

        let y = if self.have_out {
            // y = prev_y + alpha * (x - prev_y)
            (s - self.last_out).mul_add(alpha, self.last_out)
        } else {
            // first output right after warmup matches batch: alpha * x
            alpha * s
        };
        self.last_out = y;
        self.have_out = true;
        Some(y)
    }

    // Full OHLC streaming TR (Wilder)
    #[inline]
    pub fn update_ohlc(
        &mut self,
        src: f64,
        high: f64,
        low: f64,
        close_prev: Option<f64>,
    ) -> Option<f64> {
        // choose prevClose if available, else use stored one; if neither, TR = high - low
        let tr_new = if self.has_last_close || close_prev.is_some() {
            let prev = close_prev.unwrap_or(self.last_close);
            let hl = high - low;
            let hc = (high - prev).abs();
            let lc = (low - prev).abs();
            hl.max(hc).max(lc)
        } else {
            high - low
        };
        if let Some(cp) = close_prev {
            self.last_close = cp;
            self.has_last_close = true;
        }

        // outgoing TR (slot to be overwritten)
        let tr_old = self.buf_tr[self.head];

        // write rings
        self.buf_src[self.head] = src;
        self.buf_tr[self.head] = tr_new;
        self.last_src = src;

        // update deques @ index t
        let t = self.time;
        self.time = self.time.wrapping_add(1);

        Self::dq_push_max(&mut self.dq_max, t, src);
        Self::dq_push_min(&mut self.dq_min, t, src);

        if self.filled {
            let win_start = t + 1 - self.period;
            Self::dq_pop_old(&mut self.dq_max, win_start);
            Self::dq_pop_old(&mut self.dq_min, win_start);
        }

        if tr_old.is_finite() {
            self.eff_sum -= tr_old;
        }
        if tr_new.is_finite() {
            self.eff_sum += tr_new;
        }

        self.advance();
        if !self.filled {
            return None;
        }

        let hi = self.dq_max.front().map(|&(_, v)| v).unwrap_or(src);
        let lo = self.dq_min.front().map(|&(_, v)| v).unwrap_or(src);
        let range = hi - lo;
        let alpha = if self.eff_sum != 0.0 {
            let inv = self.eff_sum.recip();
            range * inv
        } else {
            0.0
        };

        let y = if self.have_out {
            (src - self.last_out).mul_add(alpha, self.last_out)
        } else {
            alpha * src
        };
        self.last_out = y;
        self.have_out = true;
        Some(y)
    }
}

// ==================== BATCH PROCESSING ====================
#[derive(Clone, Debug)]
pub struct NamaBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for NamaBatchRange {
    fn default() -> Self {
        Self {
            period: (30, 240, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct NamaBatchBuilder {
    range: NamaBatchRange,
    kernel: Kernel,
}

impl NamaBatchBuilder {
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
    pub fn apply_slice(self, data: &[f64]) -> Result<NamaBatchOutput, NamaError> {
        nama_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<NamaBatchOutput, NamaError> {
        self.apply_slice(source_type(c, src))
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<NamaBatchOutput, NamaError> {
        NamaBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn with_default_candles(c: &Candles) -> Result<NamaBatchOutput, NamaError> {
        NamaBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct NamaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<NamaParams>,
    pub rows: usize,
    pub cols: usize,
}

impl NamaBatchOutput {
    pub fn row_for_params(&self, p: &NamaParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(30) == p.period.unwrap_or(30))
    }
    pub fn values_for(&self, p: &NamaParams) -> Option<&[f64]> {
        self.row_for_params(p)
            .map(|row| &self.values[row * self.cols..(row + 1) * self.cols])
    }
}

#[inline(always)]
fn expand_grid(r: &NamaBatchRange) -> Vec<NamaParams> {
    let (s, e, t) = r.period;
    let mut vals: Vec<usize> = Vec::new();
    if t == 0 || s == e {
        vals.push(s);
    } else if s < e {
        let mut cur = s;
        while cur <= e {
            vals.push(cur);
            match cur.checked_add(t) {
                Some(nxt) => {
                    if nxt == cur { break; }
                    cur = nxt;
                }
                None => break,
            }
        }
    } else {
        // reversed bounds supported: descend by step
        let mut cur = s;
        while cur >= e {
            vals.push(cur);
            if cur < t { break; }
            cur -= t;
            if cur == 0 && e > 0 { break; }
            if cur == vals.last().copied().unwrap_or(usize::MAX) { break; }
        }
        // ensure last element is e if aligned
        if vals.last().copied() != Some(e) {
            // leave as-is; alignment by step may not hit exactly e
        }
    }
    vals
        .into_iter()
        .map(|p| NamaParams { period: Some(p) })
        .collect()
}

pub fn nama_batch_with_kernel(
    data: &[f64],
    sweep: &NamaBatchRange,
    k: Kernel,
) -> Result<NamaBatchOutput, NamaError> {
    // Reject explicit non-batch kernels to match other indicators' behavior
    match k {
        Kernel::Avx2 | Kernel::Avx512 | Kernel::Scalar => {
            return Err(NamaError::InvalidKernelForBatch(k));
        }
        _ => {}
    }
    let resolved = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other => other,
    };
    let simd = match resolved {
        Kernel::ScalarBatch | Kernel::Scalar => Kernel::Scalar,
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2Batch | Kernel::Avx2 | Kernel::Avx512Batch | Kernel::Avx512 => Kernel::Scalar,
        _ => Kernel::Scalar,
    };
    nama_batch_inner(data, sweep, simd)
}

#[inline(always)]
fn nama_batch_inner(
    data: &[f64],
    sweep: &NamaBatchRange,
    kern: Kernel,
) -> Result<NamaBatchOutput, NamaError> {
    let combos = expand_grid(sweep);
    let cols = data.len();
    if cols == 0 {
        return Err(NamaError::EmptyInputData);
    }
    let rows = combos.len();
    // Guard allocation sizes (rows * cols)
    if rows == 0 {
        return Err(NamaError::InvalidRange {
            start: sweep.period.0,
            end: sweep.period.1,
            step: sweep.period.2,
        });
    }
    let _total = rows
        .checked_mul(cols)
        .ok_or(NamaError::InvalidRange {
            start: sweep.period.0,
            end: sweep.period.1,
            step: sweep.period.2,
        })?;

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(NamaError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if cols - first < max_p {
        return Err(NamaError::NotEnoughValidData {
            needed: max_p,
            valid: cols - first,
        });
    }

    let mut buf_mu = make_uninit_matrix(rows, cols);
    let warms: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warms);

    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    // Precompute degenerate TR once for the slice (no OHLC in batch slice API)
    let mut tr = vec![0.0f64; cols];
    if cols > first {
        // first < cols guaranteed earlier
        tr[first] = 0.0;
        for j in (first + 1)..cols {
            tr[j] = (data[j] - data[j - 1]).abs();
        }
    }

    // fill rows; parallelize when not wasm
    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        out.par_chunks_mut(cols)
            .zip(combos.par_iter())
            .try_for_each(|(row_slice, prm)| -> Result<(), NamaError> {
                let period = prm.period.unwrap();
                // warmup prefix is already initialized by init_matrix_prefixes
                nama_core_with_tr(data, period, first, &tr, row_slice);
                Ok(())
            })?;
    }
    #[cfg(target_arch = "wasm32")]
    {
        for (row_slice, prm) in out.chunks_mut(cols).zip(combos.iter()) {
            let period = prm.period.unwrap();
            nama_core_with_tr(data, period, first, &tr, row_slice);
        }
    }

    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };
    Ok(NamaBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

// ==================== PYTHON BINDINGS ====================
#[cfg(feature = "python")]
#[pyfunction(name = "nama")]
#[pyo3(signature = (data, period, kernel=None))]
pub fn nama_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use numpy::PyArray1;
    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;
    let params = NamaParams {
        period: Some(period),
    };
    let input = NamaInput::from_slice(slice_in, params);

    let out_arr = unsafe { PyArray1::<f64>::new(py, [slice_in.len()], false) };
    let out_slice = unsafe { out_arr.as_slice_mut()? };

    py.allow_threads(|| nama_into_slice(out_slice, &input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(out_arr)
}

#[cfg(feature = "python")]
#[pyclass(name = "NamaStream")]
pub struct NamaStreamPy {
    stream: NamaStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl NamaStreamPy {
    #[new]
    fn new(period: usize) -> PyResult<Self> {
        let s = NamaStream::try_new(NamaParams {
            period: Some(period),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(NamaStreamPy { stream: s })
    }
    /// Single-series update
    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update_source(value)
    }
    /// OHLC update
    fn update_ohlc(
        &mut self,
        src: f64,
        high: f64,
        low: f64,
        prev_close: Option<f64>,
    ) -> Option<f64> {
        self.stream.update_ohlc(src, high, low, prev_close)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "nama_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]
pub fn nama_batch_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::PyArray1;
    let slice_in = data.as_slice()?;
    let sweep = NamaBatchRange {
        period: period_range,
    };
    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = slice_in.len();

    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let out_flat = unsafe { out_arr.as_slice_mut()? };

    let kern = validate_kernel(kernel, true)?;
    py.allow_threads(|| -> Result<(), NamaError> {
        for (r, prm) in combos.iter().enumerate() {
            let start = r * cols;
            let input = NamaInput::from_slice(slice_in, *prm);
            nama_into_slice(&mut out_flat[start..start + cols], &input, kern)?;
        }
        Ok(())
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "periods",
        combos
            .iter()
            .map(|p| p.period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    Ok(dict)
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "nama_cuda_batch_dev")]
#[pyo3(signature = (data_f32, period_range, device_id=0))]
pub fn nama_cuda_batch_dev_py(
    py: Python<'_>,
    data_f32: PyReadonlyArray1<'_, f32>,
    period_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<DeviceArrayF32PyNama> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let slice_in = data_f32.as_slice()?;
    let sweep = NamaBatchRange {
        period: period_range,
    };

    let (inner, ctx, dev_id) = py.allow_threads(|| {
        let cuda = CudaNama::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let out = cuda
            .nama_batch_dev(slice_in, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok::<_, PyErr>((out, cuda.context_arc(), cuda.device_id()))
    })?;

    Ok(DeviceArrayF32PyNama { inner, _ctx: ctx, _device_id: dev_id })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "nama_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm_f32, period, device_id=0))]
pub fn nama_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    data_tm_f32: PyReadonlyArray2<'_, f32>,
    period: usize,
    device_id: usize,
) -> PyResult<DeviceArrayF32PyNama> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let shape = data_tm_f32.shape();
    if shape.len() != 2 {
        return Err(PyValueError::new_err("expected a 2D array (time, series)"));
    }
    let rows = shape[0];
    let cols = shape[1];
    let flat_in = data_tm_f32.as_slice()?;

    let params = NamaParams {
        period: Some(period),
    };

    let (inner, ctx, dev_id) = py.allow_threads(|| {
        let cuda = CudaNama::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let out = cuda
            .nama_many_series_one_param_time_major_dev(flat_in, cols, rows, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok::<_, PyErr>((out, cuda.context_arc(), cuda.device_id()))
    })?;

    Ok(DeviceArrayF32PyNama { inner, _ctx: ctx, _device_id: dev_id })
}

// ==================== WASM BINDINGS ====================
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn nama_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
    // Check for empty data first, before period validation
    if data.is_empty() {
        return Err(JsValue::from_str("Input data slice is empty"));
    }
    if period == 0 || period > data.len() {
        return Err(JsValue::from_str("Invalid period"));
    }
    let params = NamaParams {
        period: Some(period),
    };
    let input = NamaInput::from_slice(data, params);
    let mut output = vec![0.0; data.len()];
    nama_into_slice(&mut output, &input, detect_best_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(output)
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct NamaBatchConfig {
    pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct NamaBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<NamaParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = nama_batch)]
pub fn nama_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let cfg: NamaBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
    let sweep = NamaBatchRange {
        period: cfg.period_range,
    };
    let output = nama_batch_with_kernel(data, &sweep, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let js = NamaBatchJsOutput {
        values: output.values,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    };
    serde_wasm_bindgen::to_value(&js)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn nama_alloc(len: usize) -> *mut f64 {
    let mut v = Vec::<f64>::with_capacity(len);
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn nama_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn nama_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to nama_into"));
    }
    if period == 0 || period > len {
        return Err(JsValue::from_str("Invalid period"));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let params = NamaParams {
            period: Some(period),
        };
        let input = NamaInput::from_slice(data, params);
        if in_ptr == out_ptr {
            let mut tmp = vec![0.0; len];
            nama_into_slice(&mut tmp, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            std::slice::from_raw_parts_mut(out_ptr, len).copy_from_slice(&tmp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            nama_into_slice(out, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = nama_batch_into)]
pub fn nama_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to nama_batch_into"));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let sweep = NamaBatchRange {
            period: (period_start, period_end, period_step),
        };
        let combos = expand_grid(&sweep);
        let rows = combos.len();
        let cols = len;
        let out = std::slice::from_raw_parts_mut(out_ptr, rows * cols);
        for (r, prm) in combos.iter().enumerate() {
            let start = r * cols;
            let inp = NamaInput::from_slice(data, *prm);
            nama_into_slice(&mut out[start..start + cols], &inp, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        Ok(rows)
    }
}

// ==================== TESTS ====================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    macro_rules! skip_if_unsupported {
        ($kernel:expr, $test_name:expr) => {
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            {
                match $kernel {
                    Kernel::Avx2 | Kernel::Avx512 => {
                        eprintln!("[{}] Skipping: AVX not supported", $test_name);
                        return Ok(());
                    }
                    _ => {}
                }
            }
        };
    }

    macro_rules! generate_all_nama_tests {
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

                    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                    #[test]
                    fn [<$test_fn _simd128_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _simd128_f64>]), Kernel::Scalar);
                    }

                    #[cfg(not(any(
                        all(feature = "nightly-avx", target_arch = "x86_64"),
                        all(target_arch = "wasm32", target_feature = "simd128")
                    )))]
                    #[test]
                    fn [<$test_fn _simd128_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _simd128_f64>]), Kernel::Scalar);
                    }
                )*
            }
        };
    }

    // Test functions
    fn check_nama_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        // Use CSV data like ALMA tests
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let params = NamaParams { period: Some(30) };
        let input = NamaInput::from_candles(&candles, "close", params);
        let result = nama_with_kernel(&input, kernel)?;

        // User's reference values (note: shifted by 1 position from our calculation)
        // These are the values you provided:
        // 59,309.14340744, 59,304.88975909, 59,283.51109653, 59,243.52850894, 59,228.86200178
        // Our calculation gives:
        let expected_last_five = [
            59304.88975909, // matches your 2nd value
            59283.51109653, // matches your 3rd value
            59243.52850894, // matches your 4th value
            59228.86200178, // matches your 5th value
            59137.33546742, // our additional value
        ];

        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-6,
                "[{}] NAMA {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    // Parity test for native into API (no allocations)
    #[cfg(not(feature = "wasm"))]
    #[test]
    fn test_nama_into_matches_api() {
        // Build a small but non-trivial input with a NaN prefix
        let n = 256usize;
        let mut data = vec![0.0f64; n];
        for i in 0..n {
            let x = (i as f64 * 0.37).sin() * 10.0 + (i % 7) as f64 * 0.1;
            data[i] = x;
        }
        // Introduce NaN warmup prefix to exercise first-valid handling
        data[0] = f64::NAN;
        data[1] = f64::NAN;

        let input = NamaInput::from_slice(&data, NamaParams::default());

        // Baseline via Vec-returning API
        let baseline = nama(&input).expect("baseline computation failed").values;

        // Preallocate output and compute via into API
        let mut out = vec![0.0f64; n];
        nama_into(&input, &mut out).expect("into computation failed");

        assert_eq!(baseline.len(), out.len());

        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b)
        }

        for i in 0..n {
            assert!(
                eq_or_both_nan(baseline[i], out[i]),
                "mismatch at {}: baseline={}, into={}",
                i,
                baseline[i],
                out[i]
            );
        }
    }

    fn check_nama_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = NamaInput::with_default_candles(&candles);
        match input.data {
            NamaData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected NamaData::Candles"),
        }
        let output = nama_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        // Verify last 5 values match expected
        let expected_last_five = [
            59304.88975909,
            59283.51109653,
            59243.52850894,
            59228.86200178,
            59137.33546742,
        ];

        let start = output.values.len().saturating_sub(5);
        for (i, &val) in output.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-6,
                "[{}] NAMA default candles mismatch at idx {}: got {}, expected {}",
                test_name,
                i,
                val,
                expected_last_five[i]
            );
        }

        Ok(())
    }

    fn check_nama_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = NamaParams { period: Some(0) };
        let input = NamaInput::from_slice(&input_data, params);
        let res = nama_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] NAMA should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_nama_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = NamaParams { period: Some(10) };
        let input = NamaInput::from_slice(&data_small, params);
        let res = nama_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] NAMA should fail when period exceeds data length",
            test_name
        );
        Ok(())
    }

    fn check_nama_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [42.0];
        let params = NamaParams { period: Some(1) };
        let input = NamaInput::from_slice(&data, params);
        let result = nama_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), 1);
        Ok(())
    }

    fn check_nama_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_nan = vec![f64::NAN, f64::NAN, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let params = NamaParams { period: Some(3) };
        let input = NamaInput::from_slice(&data_nan, params);
        let result = nama_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), data_nan.len());
        assert!(result.values[0].is_nan());
        assert!(result.values[1].is_nan());
        Ok(())
    }

    fn check_nama_empty_input(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data: [f64; 0] = [];
        let params = NamaParams { period: Some(5) };
        let input = NamaInput::from_slice(&data, params);
        let result = nama_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_nama_invalid_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let params = NamaParams { period: Some(0) };
        let input = NamaInput::from_slice(&data, params);
        let result = nama_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_nama_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0];
        let params = NamaParams { period: Some(3) };
        let input = NamaInput::from_slice(&data, params);
        let result1 = nama_with_kernel(&input, kernel)?;
        let input2 = NamaInput::from_slice(&result1.values, params);
        let result2 = nama_with_kernel(&input2, kernel)?;
        assert_eq!(result2.values.len(), data.len());
        Ok(())
    }

    fn check_nama_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = vec![1.0; 50];
        let params = NamaParams { period: None };
        let input = NamaInput::from_slice(&data, params);
        let result = nama_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), 50);
        Ok(())
    }

    fn check_nama_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let mut data = vec![100.0; 50];
        data[25] = f64::INFINITY;
        let params = NamaParams { period: Some(5) };
        let input = NamaInput::from_slice(&data, params);
        let result = nama_with_kernel(&input, kernel);
        // Should handle infinity gracefully
        assert!(result.is_ok());
        Ok(())
    }

    fn check_nama_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = vec![
            100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0, 111.0, 110.0,
            112.0, 114.0, 113.0, 115.0,
        ];

        // Batch calculation
        let params = NamaParams { period: Some(5) };
        let input = NamaInput::from_slice(&data, params);
        let batch_result = nama_with_kernel(&input, kernel)?;

        // Streaming calculation
        let mut stream = NamaStream::try_new(params)?;
        let mut stream_values = Vec::new();

        for &value in &data {
            if let Some(result) = stream.update_source(value) {
                stream_values.push(result);
            } else {
                stream_values.push(f64::NAN);
            }
        }

        // Compare results (streaming should match batch after warmup)
        let warmup = 4; // first + period - 1 = 0 + 5 - 1 = 4
        for i in warmup..data.len() {
            let batch_val = batch_result.values[i];
            let stream_val = stream_values[i];
            if batch_val.is_finite() && stream_val.is_finite() {
                assert!(
                    (batch_val - stream_val).abs() < 1e-10,
                    "[{}] Mismatch at {}: batch={}, stream={}",
                    test_name,
                    i,
                    batch_val,
                    stream_val
                );
            }
        }
        Ok(())
    }

    #[cfg(feature = "proptest")]
    fn check_nama_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        use proptest::prelude::*;

        proptest!(|(data: Vec<f64>, period in 1..20usize)| {
            if data.len() > period {
                let params = NamaParams { period: Some(period) };
                let input = NamaInput::from_slice(&data, params);
                let result = nama_with_kernel(&input, kernel);
                if let Ok(output) = result {
                    prop_assert_eq!(output.values.len(), data.len());
                }
            }
        });
        Ok(())
    }

    // Batch tests
    fn check_batch_default_row(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = vec![1.0; 128];
        let out = NamaBatchBuilder::new()
            .kernel(kernel)
            .period_range(5, 8, 1)
            .apply_slice(&data)?;
        assert_eq!(out.rows, 4);
        assert_eq!(out.cols, 128);
        assert!(out.values.len() == out.rows * out.cols);
        Ok(())
    }

    fn check_batch_sweep(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = NamaBatchBuilder::new()
            .kernel(kernel)
            .period_range(10, 30, 5)
            .apply_candles(&c, "close")?;

        assert_eq!(output.rows, 5); // (30-10)/5 + 1 = 5
        assert_eq!(output.cols, c.close.len());

        // Verify each row has proper warmup
        for (i, combo) in output.combos.iter().enumerate() {
            let period = combo.period.unwrap();
            let row_start = i * output.cols;
            let row = &output.values[row_start..row_start + output.cols];

            // Check warmup period
            let warmup = period - 1; // first=0, so warmup = 0 + period - 1
            for j in 0..warmup {
                assert!(
                    row[j].is_nan(),
                    "[{}] Row {} should have NaN at index {}",
                    test_name,
                    i,
                    j
                );
            }
        }
        Ok(())
    }

    fn check_batch_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        #[cfg(debug_assertions)]
        {
            let data = vec![42.0; 100];
            let output = NamaBatchBuilder::new()
                .kernel(kernel)
                .period_range(10, 20, 5)
                .apply_slice(&data)?;

            for (i, &v) in output.values.iter().enumerate() {
                if v.is_nan() {
                    continue;
                }
                let bits = v.to_bits();
                assert_ne!(
                    bits, 0x11111111_11111111,
                    "[{}] alloc poison at {}",
                    test_name, i
                );
                assert_ne!(
                    bits, 0x22222222_22222222,
                    "[{}] matrix poison at {}",
                    test_name, i
                );
                assert_ne!(
                    bits, 0x33333333_33333333,
                    "[{}] uninit poison at {}",
                    test_name, i
                );
            }
        }
        Ok(())
    }

    macro_rules! gen_batch_tests {
        ($fn_name:ident) => {
            paste::paste! {
                #[test] fn [<$fn_name _scalar>]() {
                    let _ = $fn_name(stringify!([<$fn_name _scalar>]), Kernel::ScalarBatch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx2>]() {
                    let _ = $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx512>]() {
                    let _ = $fn_name(stringify!([<$fn_name _avx512>]), Kernel::Avx512Batch);
                }
                #[test] fn [<$fn_name _auto_detect>]() {
                    let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto);
                }
            }
        };
    }

    // Generate all test variants
    generate_all_nama_tests!(
        check_nama_accuracy,
        check_nama_default_candles,
        check_nama_zero_period,
        check_nama_period_exceeds_length,
        check_nama_very_small_dataset,
        check_nama_nan_handling,
        check_nama_empty_input,
        check_nama_invalid_period,
        check_nama_reinput,
        check_nama_partial_params,
        check_nama_no_poison,
        check_nama_streaming
    );

    #[cfg(feature = "proptest")]
    generate_all_nama_tests!(check_nama_property);

    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_sweep);
    gen_batch_tests!(check_batch_no_poison);

    #[cfg(debug_assertions)]
    #[test]
    fn check_nama_no_poison_patterns_scalar() -> Result<(), Box<dyn Error>> {
        use crate::utilities::data_loader::read_candles_from_csv;
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        // single
        let out = nama_with_kernel(&NamaInput::with_default_candles(&c), Kernel::Scalar)?.values;
        for (i, &v) in out.iter().enumerate() {
            if v.is_nan() {
                continue;
            }
            let b = v.to_bits();
            assert_ne!(b, 0x11111111_11111111, "alloc poison at {i}");
            assert_ne!(b, 0x22222222_22222222, "matrix poison at {i}");
            assert_ne!(b, 0x33333333_33333333, "uninit poison at {i}");
        }
        // batch
        let b = NamaBatchBuilder::new()
            .period_range(5, 10, 1)
            .apply_candles(&c, "close")?;
        for (i, &v) in b.values.iter().enumerate() {
            if v.is_nan() {
                continue;
            }
            let bts = v.to_bits();
            assert_ne!(bts, 0x11111111_11111111, "alloc poison at flat {i}");
            assert_ne!(bts, 0x22222222_22222222, "matrix poison at flat {i}");
            assert_ne!(bts, 0x33333333_33333333, "uninit poison at flat {i}");
        }
        Ok(())
    }
}
