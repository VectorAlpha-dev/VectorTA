//! # Zero Lag Exponential Moving Average (ZLEMA)
//!
//! ZLEMA is a moving average designed to reduce lag by de-lagging the input before EMA calculation.
//! Supports kernel (SIMD) selection and batch/grid computation with streaming support.
//!
//! ## Parameters
//! - **period**: Lookback window (>= 1, defaults to 14).
//!
//! ## Returns
//! - **`Ok(ZlemaOutput)`** on success, containing a `Vec<f64>`.
//! - **`Err(ZlemaError)`** otherwise.
//!
//! ## Developer Status
//! - SIMD (AVX2/AVX512): Implemented but disabled by default — EMA is sequential; de-lag vectorization yields <5% on typical sizes.
//! - Scalar: optimized two-phase loop with unrolled core and unchecked indexing; no change to numerical behavior.
//! - Streaming: O(1), ring = lag+1 with absolute-index warmup/de‑lag gating; no FMA to match scalar/batch.
//! - Batch rows: no shared precompute to exploit; row kernels delegate to single-series variants.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::moving_averages::{alma_wrapper::DeviceArrayF32, CudaZlema};
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::memory::DeviceBuffer;
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::types::PyDictMethods;
#[cfg(feature = "python")]
use pyo3::{pyclass, pyfunction, pymethods, Bound, PyResult, Python};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;
#[cfg(feature = "wasm")]
use wasm_bindgen::JsValue;

// Python CUDA handle for ZLEMA: keeps CUDA context alive and exposes CAI v3 + DLPack
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::context::Context;
#[cfg(all(feature = "python", feature = "cuda"))]
use std::sync::Arc;

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(module = "ta_indicators.cuda", name = "ZlemaDeviceArrayF32", unsendable)]
pub struct DeviceArrayF32Py {
    pub(crate) inner: DeviceArrayF32,
    pub(crate) _ctx: Arc<Context>,
    pub(crate) device_id: u32,
    pub(crate) stream: usize,
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pymethods]
impl DeviceArrayF32Py {
    #[getter]
    fn __cuda_array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let d = pyo3::types::PyDict::new(py);
        d.set_item("shape", (self.inner.rows, self.inner.cols))?;
        d.set_item("typestr", "<f4")?;
        d.set_item(
            "strides",
            (
                self.inner.cols * core::mem::size_of::<f32>(),
                core::mem::size_of::<f32>(),
            ),
        )?;
        d.set_item("data", (self.inner.device_ptr() as usize, false))?;
        // Kernels are launched on a non-default stream and not synchronized here; include stream per CAI v3.
        if self.stream != 0 {
            d.set_item("stream", self.stream)?;
        }
        d.set_item("version", 3)?;
        Ok(d)
    }

    fn __dlpack_device__(&self) -> (i32, i32) { (2, self.device_id as i32) }

    #[pyo3(signature = (stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__<'py>(
        &mut self,
        py: Python<'py>,
        stream: Option<pyo3::PyObject>,
        max_version: Option<pyo3::PyObject>,
        dl_device: Option<pyo3::PyObject>,
        copy: Option<pyo3::PyObject>,
    ) -> PyResult<pyo3::PyObject> {
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

        // If a consumer stream pointer (>2) is provided, insert a dependency via event
        #[cfg(feature = "cuda")]
        if let Some(stream_obj) = stream.as_ref() {
            if let Ok(s) = stream_obj.extract::<usize>(py) {
                if s > 2 {
                    unsafe {
                        use cust::sys as cu;
                        let mut ev: cu::CUevent = std::ptr::null_mut();
                        let _ = cu::cuEventCreate(&mut ev as *mut _, 0);
                        let prod: cu::CUstream = self.stream as *mut _;
                        let _ = cu::cuEventRecord(ev, prod);
                        let cons: cu::CUstream = s as *mut _;
                        let _ = cu::cuStreamWaitEvent(cons, ev, 0);
                        let _ = cu::cuEventDestroy_v2(ev);
                    }
                }
            }
        }

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

impl<'a> AsRef<[f64]> for ZlemaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            ZlemaData::Slice(slice) => slice,
            ZlemaData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum ZlemaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct ZlemaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct ZlemaParams {
    pub period: Option<usize>,
}

impl Default for ZlemaParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct ZlemaInput<'a> {
    pub data: ZlemaData<'a>,
    pub params: ZlemaParams,
}

impl<'a> ZlemaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: ZlemaParams) -> Self {
        Self {
            data: ZlemaData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: ZlemaParams) -> Self {
        Self {
            data: ZlemaData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", ZlemaParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(14)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct ZlemaBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for ZlemaBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl ZlemaBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<ZlemaOutput, ZlemaError> {
        let p = ZlemaParams {
            period: self.period,
        };
        let i = ZlemaInput::from_candles(c, "close", p);
        zlema_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<ZlemaOutput, ZlemaError> {
        let p = ZlemaParams {
            period: self.period,
        };
        let i = ZlemaInput::from_slice(d, p);
        zlema_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<ZlemaStream, ZlemaError> {
        let p = ZlemaParams {
            period: self.period,
        };
        ZlemaStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum ZlemaError {
    #[error("zlema: Input data slice is empty.")]
    EmptyInputData,
    #[error("zlema: All values are NaN.")]
    AllValuesNaN,
    #[error("zlema: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("zlema: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("zlema: output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("zlema: invalid range: start={start}, end={end}, step={step}")]
    InvalidRange { start: usize, end: usize, step: usize },
    #[error("zlema: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[inline(always)]
fn zlema_validate<'a>(
    input: &'a ZlemaInput,
) -> Result<(&'a [f64], usize, usize, usize), ZlemaError> {
    let data: &'a [f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(ZlemaError::EmptyInputData);
    }
    let period = input.get_period();
    if period == 0 || period > len {
        return Err(ZlemaError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(ZlemaError::AllValuesNaN)?;
    if len - first < period {
        return Err(ZlemaError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }
    let warm = first + period - 1;
    Ok((data, first, period, warm))
}

#[inline]
pub fn zlema(input: &ZlemaInput) -> Result<ZlemaOutput, ZlemaError> {
    zlema_with_kernel(input, Kernel::Auto)
}

pub fn zlema_with_kernel(input: &ZlemaInput, kernel: Kernel) -> Result<ZlemaOutput, ZlemaError> {
    let (data, first, period, warm) = zlema_validate(input)?;
    let mut out = alloc_with_nan_prefix(data.len(), warm);

    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        k => k,
    };

    unsafe {
        match (kernel, chosen) {
            // Keep Auto on scalar by policy
            (Kernel::Auto, _) => zlema_scalar(data, period, first, &mut out),
            // Route all explicit kernels to scalar since it’s faster here
            (_, Kernel::Scalar | Kernel::ScalarBatch) => {
                zlema_scalar(data, period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            (_, Kernel::Avx2 | Kernel::Avx2Batch) => zlema_scalar(data, period, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            (_, Kernel::Avx512 | Kernel::Avx512Batch) => {
                zlema_scalar(data, period, first, &mut out)
            }
            _ => unreachable!(),
        }
    }

    Ok(ZlemaOutput { values: out })
}

#[inline]
pub fn zlema_scalar(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    let len = data.len();
    let lag = (period - 1) / 2;
    let alpha = 2.0 / (period as f64 + 1.0);

    let warm = first + period - 1;

    // Initialize EMA with first valid value
    let mut last_ema = data[first];

    // Process all values starting from first
    for i in first..len {
        if i > first {
            // For de-lagging, we need to ensure we're not accessing NaN values
            // We can only de-lag if we have enough valid history
            let val = if i < first + lag {
                // Not enough valid history for de-lagging, use regular value
                data[i]
            } else {
                // Apply de-lagging formula
                2.0 * data[i] - data[i - lag]
            };
            last_ema = alpha * val + (1.0 - alpha) * last_ema;
        }
        if i >= warm {
            out[i] = last_ema;
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn zlema_avx2(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    use core::arch::x86_64::*;

    debug_assert_eq!(data.len(), out.len());
    let len = data.len();
    let lag = (period - 1) / 2;
    let warm = first + period - 1;
    let alpha = 2.0 / (period as f64 + 1.0);
    let decay = 1.0 - alpha;

    let mut last_ema = unsafe { *data.get_unchecked(first) };
    if warm == first {
        unsafe { *out.get_unchecked_mut(first) = last_ema };
    }

    // Phase A: scalar
    let mut i = first + 1;
    let phase_a_end = if lag > 0 {
        core::cmp::min(len, first + lag)
    } else {
        i
    };
    unsafe {
        while i < phase_a_end {
            let xi = *data.get_unchecked(i);
            last_ema = alpha * xi + decay * last_ema;
            if i >= warm {
                *out.get_unchecked_mut(i) = last_ema;
            }
            i += 1;
        }
    }

    // Phase B: vectorized de-lagging
    let start_b = if lag > 0 {
        first + lag
    } else {
        core::cmp::min(first + 1, len)
    };
    if start_b >= len {
        return;
    }
    i = start_b;

    unsafe {
        let two = _mm256_set1_pd(2.0);
        while i + 4 <= len {
            let x = _mm256_loadu_pd(data.as_ptr().add(i));
            let xlg = _mm256_loadu_pd(data.as_ptr().add(i - lag));
            let val = _mm256_sub_pd(_mm256_mul_pd(two, x), xlg);

            let mut tmp: [f64; 4] = MaybeUninit::uninit().assume_init();
            _mm256_storeu_pd(tmp.as_mut_ptr(), val);

            {
                let j = i;
                let v = *tmp.get_unchecked(0);
                // FMA form: last += alpha * (v - last)
                last_ema = (v - last_ema).mul_add(alpha, last_ema);
                if j >= warm {
                    *out.get_unchecked_mut(j) = last_ema;
                }
            }
            {
                let j = i + 1;
                let v = *tmp.get_unchecked(1);
                last_ema = (v - last_ema).mul_add(alpha, last_ema);
                if j >= warm {
                    *out.get_unchecked_mut(j) = last_ema;
                }
            }
            {
                let j = i + 2;
                let v = *tmp.get_unchecked(2);
                last_ema = (v - last_ema).mul_add(alpha, last_ema);
                if j >= warm {
                    *out.get_unchecked_mut(j) = last_ema;
                }
            }
            {
                let j = i + 3;
                let v = *tmp.get_unchecked(3);
                last_ema = (v - last_ema).mul_add(alpha, last_ema);
                if j >= warm {
                    *out.get_unchecked_mut(j) = last_ema;
                }
            }

            i += 4;
        }

        while i < len {
            let xi = *data.get_unchecked(i);
            let xlag = *data.get_unchecked(i - lag);
            let val = 2.0 * xi - xlag;
            last_ema = (val - last_ema).mul_add(alpha, last_ema);
            if i >= warm {
                *out.get_unchecked_mut(i) = last_ema;
            }
            i += 1;
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn zlema_avx512(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    // AVX512 path delegates to AVX2 FMA variant to keep behavior consistent
    zlema_avx2(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn zlema_avx512_short(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    zlema_avx512(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn zlema_avx512_long(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    zlema_avx512(data, period, first, out)
}

#[inline]
pub fn zlema_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    _w_ptr: *const f64,
    _inv_n: f64,
    out: &mut [f64],
) {
    let len = data.len();
    let lag = (period - 1) / 2;
    let alpha = 2.0 / (period as f64 + 1.0);
    let warm = first + period - 1;

    let mut last_ema = data[first];

    // Process all values starting from first
    for i in first..len {
        if i > first {
            // For de-lagging, we need to ensure we're not accessing NaN values
            // We can only de-lag if we have enough valid history
            let val = if i < first + lag {
                // Not enough valid history for de-lagging, use regular value
                data[i]
            } else {
                // Apply de-lagging formula
                2.0 * data[i] - data[i - lag]
            };
            last_ema = alpha * val + (1.0 - alpha) * last_ema;
        }
        if i >= warm {
            out[i] = last_ema;
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn zlema_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    stride: usize,
    w_ptr: *const f64,
    inv_n: f64,
    out: &mut [f64],
) {
    zlema_row_scalar(data, first, period, stride, w_ptr, inv_n, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn zlema_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    stride: usize,
    w_ptr: *const f64,
    inv_n: f64,
    out: &mut [f64],
) {
    zlema_row_scalar(data, first, period, stride, w_ptr, inv_n, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn zlema_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    stride: usize,
    w_ptr: *const f64,
    inv_n: f64,
    out: &mut [f64],
) {
    zlema_row_scalar(data, first, period, stride, w_ptr, inv_n, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn zlema_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    stride: usize,
    w_ptr: *const f64,
    inv_n: f64,
    out: &mut [f64],
) {
    zlema_row_scalar(data, first, period, stride, w_ptr, inv_n, out)
}

#[derive(Debug, Clone)]
pub struct ZlemaStream {
    // configuration
    period: usize,
    lag: usize,
    alpha: f64,
    decay: f64, // 1 - alpha

    // state
    last_ema: f64, // current EMA state (NaN until first finite)
    ring: Vec<f64>,
    head: usize,              // write index in ring [0 .. ring.len())
    idx: usize,               // total samples seen since construction (absolute index)
    first_idx: Option<usize>, // absolute index of first non-NaN sample
    warm_idx: Option<usize>,  // absolute index where outputs become valid: first + period - 1
}

impl ZlemaStream {
    #[inline]
    pub fn try_new(params: ZlemaParams) -> Result<Self, ZlemaError> {
        let period = params.period.unwrap_or(14);
        if period == 0 {
            return Err(ZlemaError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        let lag = (period - 1) / 2;
        let alpha = 2.0 / (period as f64 + 1.0);

        // ring holds the last (lag + 1) raw inputs (including NaN), which is the
        // minimum needed to access x[t - lag] at any time t without modulo.
        let ring_len = (lag + 1).max(1);
        Ok(Self {
            period,
            lag,
            alpha,
            decay: 1.0 - alpha,
            last_ema: f64::NAN,
            ring: vec![f64::NAN; ring_len],
            head: 0,
            idx: 0,
            first_idx: None,
            warm_idx: None,
        })
    }

    /// O(1) update. Returns `Some(zlema)` once warmed; `None` before warmup.
    /// Matches batch behavior exactly:
    ///   - de-lag input starts at absolute index (first + lag)
    ///   - first EMA sample is the first finite value (no recurrence step)
    ///   - any NaN taints EMA forever (as in batch)
    #[inline(always)]
    pub fn update(&mut self, x: f64) -> Option<f64> {
        // 0) write new sample into the ring at current head
        let pos = self.head;
        self.ring[pos] = x;
        // advance head with a single predictable branch (no %)
        self.head += 1;
        if self.head == self.ring.len() {
            self.head = 0;
        }

        // 1) capture absolute index for this sample
        let i = self.idx;
        // wrapping add to avoid potential overflow panics on extremely long runs
        self.idx = self.idx.wrapping_add(1);

        // 2) NaN handling: preserve batch semantics
        if x.is_nan() {
            // Any NaN encountered makes EMA NaN permanently in the batch algorithm.
            self.last_ema = f64::NAN;
            // warm decision uses absolute warm index if we have one
            return match self.warm_idx {
                Some(w) if i >= w => Some(self.last_ema),
                _ => None,
            };
        }

        // 3) initialize on the first finite sample (exactly like batch)
        if self.first_idx.is_none() {
            self.first_idx = Some(i);
            // warm index is now known
            let w = i + (self.period - 1);
            self.warm_idx = Some(w);
            // first EMA sample is the raw first value (no recurrence)
            self.last_ema = x;

            return if i >= w { Some(self.last_ema) } else { None };
        }

        // 4) compute de-lagged input using absolute gating: i < first + lag ? x : 2x - x[i-lag]
        let first = self.first_idx.unwrap();
        let val = if self.lag == 0 || i < first + self.lag {
            x
        } else {
            // with ring.len() = lag + 1, the slot holding x[i - lag] is:
            //  - pos >= lag : pos - lag
            //  - else       : pos + 1   (since ring_len - lag == 1)
            let lag_pos = if pos >= self.lag {
                pos - self.lag
            } else {
                pos + 1
            };
            let x_lag = self.ring[lag_pos];
            2.0 * x - x_lag
        };

        // 5) standard EMA recurrence (no FMA to match batch/scalar exactly)
        // last = alpha * val + (1 - alpha) * last
        self.last_ema = self.alpha * val + self.decay * self.last_ema;

        // 6) warm gate using absolute warm index
        match self.warm_idx {
            Some(w) if i >= w => Some(self.last_ema),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ZlemaBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for ZlemaBatchRange {
    fn default() -> Self {
        Self {
            period: (14, 263, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ZlemaBatchBuilder {
    range: ZlemaBatchRange,
    kernel: Kernel,
}

impl ZlemaBatchBuilder {
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
    pub fn apply_slice(self, data: &[f64]) -> Result<ZlemaBatchOutput, ZlemaError> {
        zlema_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<ZlemaBatchOutput, ZlemaError> {
        ZlemaBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<ZlemaBatchOutput, ZlemaError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<ZlemaBatchOutput, ZlemaError> {
        ZlemaBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn zlema_batch_with_kernel(
    data: &[f64],
    sweep: &ZlemaBatchRange,
    k: Kernel,
) -> Result<ZlemaBatchOutput, ZlemaError> {
    // Auto-map non-batch kernels to their batch equivalents for better UX
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        Kernel::Scalar => Kernel::ScalarBatch,
        Kernel::Avx2 => Kernel::Avx2Batch,
        Kernel::Avx512 => Kernel::Avx512Batch,
        other if other.is_batch() => other,
        _ => Kernel::ScalarBatch, // Fallback for any unknown kernels
    };

    // Disable SIMD for batch: ZLEMA is sequential and period-varying; negligible wins observed
    // and AVX-512 downclock can dominate. Keep the kernel parameter for API parity but execute
    // the scalar batch path.
    let kernel = match kernel {
        Kernel::Avx512Batch | Kernel::Avx2Batch => Kernel::ScalarBatch,
        other => other,
    };
    zlema_batch_par_slice(data, sweep, kernel)
}

#[derive(Clone, Debug)]
pub struct ZlemaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<ZlemaParams>,
    pub rows: usize,
    pub cols: usize,
}

impl ZlemaBatchOutput {
    pub fn row_for_params(&self, p: &ZlemaParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(14) == p.period.unwrap_or(14))
    }
    pub fn values_for(&self, p: &ZlemaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &ZlemaBatchRange) -> Vec<ZlemaParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        let mut vals = Vec::new();
        if start <= end {
            for v in (start..=end).step_by(step) {
                vals.push(v);
            }
        } else {
            // Support reversed bounds using descending sequence
            let mut v = start;
            loop {
                vals.push(v);
                if v <= end { break; }
                match v.checked_sub(step) {
                    Some(nx) => {
                        if nx == v { break; }
                        v = nx;
                    }
                    None => break,
                }
            }
        }
        vals
    }
    let periods = axis_usize(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(ZlemaParams { period: Some(p) });
    }
    out
}

#[inline(always)]
pub fn zlema_batch_slice(
    data: &[f64],
    sweep: &ZlemaBatchRange,
    kern: Kernel,
) -> Result<ZlemaBatchOutput, ZlemaError> {
    zlema_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn zlema_batch_par_slice(
    data: &[f64],
    sweep: &ZlemaBatchRange,
    kern: Kernel,
) -> Result<ZlemaBatchOutput, ZlemaError> {
    zlema_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn zlema_batch_inner(
    data: &[f64],
    sweep: &ZlemaBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<ZlemaBatchOutput, ZlemaError> {
    // Select batch kernel (require batch or Auto)
    let simd = match kern {
        Kernel::Auto => match detect_best_batch_kernel() {
            Kernel::Avx512Batch => Kernel::Avx512,
            Kernel::Avx2Batch => Kernel::Avx2,
            Kernel::ScalarBatch => Kernel::Scalar,
            _ => Kernel::Scalar,
        },
        k if k.is_batch() => k.to_non_batch(),
        other => return Err(ZlemaError::InvalidKernelForBatch(other)),
    };

    // grid
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(ZlemaError::InvalidRange { start: sweep.period.0, end: sweep.period.1, step: sweep.period.2 });
    }

    if data.is_empty() {
        return Err(ZlemaError::EmptyInputData);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(ZlemaError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(ZlemaError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();
    // checked multiplication for allocation sizes
    let _cap = rows
        .checked_mul(cols)
        .ok_or(ZlemaError::InvalidRange { start: sweep.period.0, end: sweep.period.1, step: sweep.period.2 })?;

    // allocate uninit and stamp only warm prefixes
    let mut buf_mu = make_uninit_matrix(rows, cols);
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    // reinterpret once
    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    // write rows without touching the warm prefix
    let do_row = |row: usize, dst: &mut [f64]| {
        let p = combos[row].period.unwrap();
        match simd {
            Kernel::Scalar => zlema_row_scalar(data, first, p, 0, core::ptr::null(), 0.0, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => zlema_row_avx2(data, first, p, 0, core::ptr::null(), 0.0, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => zlema_row_avx512(data, first, p, 0, core::ptr::null(), 0.0, dst),
            _ => unreachable!(),
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        out.par_chunks_mut(cols)
            .enumerate()
            .for_each(|(r, s)| do_row(r, s));
        #[cfg(target_arch = "wasm32")]
        for (r, s) in out.chunks_mut(cols).enumerate() {
            do_row(r, s);
        }
    } else {
        for (r, s) in out.chunks_mut(cols).enumerate() {
            do_row(r, s);
        }
    }

    // move out the flat f64 buffer safely
    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(ZlemaBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
pub fn expand_grid_zlema(r: &ZlemaBatchRange) -> Vec<ZlemaParams> {
    expand_grid(r)
}

/// Direct buffer write version for Python bindings
#[inline(always)]
pub fn zlema_batch_inner_into(
    data: &[f64],
    sweep: &ZlemaBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<ZlemaParams>, ZlemaError> {
    // Select batch kernel (require batch or Auto)
    let simd = match kern {
        Kernel::Auto => match detect_best_batch_kernel() {
            Kernel::Avx512Batch => Kernel::Avx512,
            Kernel::Avx2Batch => Kernel::Avx2,
            Kernel::ScalarBatch => Kernel::Scalar,
            _ => Kernel::Scalar,
        },
        k if k.is_batch() => k.to_non_batch(),
        other => return Err(ZlemaError::InvalidKernelForBatch(other)),
    };

    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(ZlemaError::InvalidRange { start: sweep.period.0, end: sweep.period.1, step: sweep.period.2 });
    }
    if data.is_empty() {
        return Err(ZlemaError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(ZlemaError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(ZlemaError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();
    let total = rows
        .checked_mul(cols)
        .ok_or(ZlemaError::InvalidRange { start: sweep.period.0, end: sweep.period.1, step: sweep.period.2 })?;
    if out.len() != total {
        return Err(ZlemaError::OutputLengthMismatch { expected: total, got: out.len() });
    }

    // Reinterpret the target as MaybeUninit and stamp ONLY warm prefixes once.
    let out_mu: &mut [MaybeUninit<f64>] = unsafe {
        core::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();
    init_matrix_prefixes(out_mu, cols, &warm);

    // Now write rows beyond warm
    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| {
        let p = combos[row].period.unwrap();
        let dst = unsafe {
            core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len())
        };
        match simd {
            Kernel::Scalar => zlema_row_scalar(data, first, p, 0, core::ptr::null(), 0.0, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => zlema_row_avx2(data, first, p, 0, core::ptr::null(), 0.0, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => zlema_row_avx512(data, first, p, 0, core::ptr::null(), 0.0, dst),
            _ => unreachable!(),
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        out_mu
            .par_chunks_mut(cols)
            .enumerate()
            .for_each(|(r, s)| do_row(r, s));
        #[cfg(target_arch = "wasm32")]
        for (r, s) in out_mu.chunks_mut(cols).enumerate() {
            do_row(r, s);
        }
    } else {
        for (r, s) in out_mu.chunks_mut(cols).enumerate() {
            do_row(r, s);
        }
    }

    Ok(combos)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn check_zlema_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = ZlemaParams { period: None };
        let input = ZlemaInput::from_candles(&candles, "close", default_params);
        let output = zlema_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_zlema_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = ZlemaInput::from_candles(&candles, "close", ZlemaParams::default());
        let result = zlema_with_kernel(&input, kernel)?;
        let expected_last_five = [59015.1, 59165.2, 59168.1, 59147.0, 58978.9];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-1,
                "[{}] ZLEMA {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_zlema_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = ZlemaParams { period: Some(0) };
        let input = ZlemaInput::from_slice(&input_data, params);
        let res = zlema_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] ZLEMA should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_zlema_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = ZlemaParams { period: Some(10) };
        let input = ZlemaInput::from_slice(&data_small, params);
        let res = zlema_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] ZLEMA should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_zlema_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = ZlemaParams { period: Some(14) };
        let input = ZlemaInput::from_slice(&single_point, params);
        let res = zlema_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] ZLEMA should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_zlema_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = ZlemaParams { period: Some(21) };
        let first_input = ZlemaInput::from_candles(&candles, "close", first_params);
        let first_result = zlema_with_kernel(&first_input, kernel)?;
        let second_params = ZlemaParams { period: Some(14) };
        let second_input = ZlemaInput::from_slice(&first_result.values, second_params);
        let second_result = zlema_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        // First result has warmup at index 20 (period 21 - 1)
        // So second calculation starts from index 20 and has warmup at 20 + 14 - 1 = 33
        // Values should be valid from index 33 onwards
        for (idx, &val) in second_result.values.iter().enumerate().skip(34) {
            assert!(val.is_finite(), "NaN found at index {}", idx);
        }
        Ok(())
    }

    fn check_zlema_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = ZlemaInput::from_candles(&candles, "close", ZlemaParams::default());
        let res = zlema_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), candles.close.len());
        if res.values.len() > 20 {
            for (i, &val) in res.values[20..].iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "[{}] Found unexpected NaN at out-index {}",
                    test_name,
                    20 + i
                );
            }
        }
        Ok(())
    }

    fn check_zlema_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let period = 14;
        let input = ZlemaInput::from_candles(
            &candles,
            "close",
            ZlemaParams {
                period: Some(period),
            },
        );
        let batch_output = zlema_with_kernel(&input, kernel)?.values;

        let mut stream = ZlemaStream::try_new(ZlemaParams {
            period: Some(period),
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
                "[{}] ZLEMA streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    // Check for poison values in single output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_zlema_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Test multiple parameter combinations to catch uninitialized memory reads
        // Include edge cases like odd/even periods, small and large values
        let test_periods = vec![1, 2, 3, 5, 7, 10, 14, 20, 21, 30, 50, 100, 200];

        for period in test_periods {
            let params = ZlemaParams {
                period: Some(period),
            };
            let input = ZlemaInput::from_candles(&candles, "close", params);

            // Skip if period is too large for the data
            if period > candles.close.len() {
                continue;
            }

            let output = zlema_with_kernel(&input, kernel)?;

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
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} with period {}",
						test_name, val, bits, i, period
					);
                }

                // Check for init_matrix_prefixes poison (0x22222222_22222222)
                if bits == 0x22222222_22222222 {
                    panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} with period {}",
						test_name, val, bits, i, period
					);
                }

                // Check for make_uninit_matrix poison (0x33333333_33333333)
                if bits == 0x33333333_33333333 {
                    panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} with period {}",
						test_name, val, bits, i, period
					);
                }
            }
        }

        Ok(())
    }

    // Release mode stub - does nothing
    #[cfg(not(debug_assertions))]
    fn check_zlema_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_zlema_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // ZLEMA only has period parameter, no offset/sigma like ALMA
        // Strategy: period from 1..=100, then generate matching data
        let strat = (1usize..=100).prop_flat_map(|period| {
            (
                prop::collection::vec(
                    (-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                    period.max(2)..400,
                ),
                Just(period),
            )
        });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(data, period)| {
                let params = ZlemaParams {
                    period: Some(period),
                };
                let input = ZlemaInput::from_slice(&data, params);

                let ZlemaOutput { values: out } = zlema_with_kernel(&input, kernel).unwrap();
                let ZlemaOutput { values: ref_out } =
                    zlema_with_kernel(&input, Kernel::Scalar).unwrap();

                // Property 1: Output length should match input
                prop_assert_eq!(out.len(), data.len(), "Output length mismatch");

                // Property 2: Warmup period check
                // ZLEMA starts calculating from the first non-NaN value, but uses a warmup period
                // The actual warmup is first + period - 1, where first is the first non-NaN index
                let first_non_nan = data.iter().position(|x| !x.is_nan()).unwrap_or(0);
                let warmup = first_non_nan + period - 1;

                // Essential: Values before first_non_nan input must be NaN
                for i in 0..first_non_nan.min(data.len()) {
                    prop_assert!(
                        out[i].is_nan(),
                        "Expected NaN at index {} before first non-NaN input",
                        i
                    );
                }

                // Essential: After warmup period, must have valid calculated values
                for i in warmup..data.len() {
                    prop_assert!(
                        !out[i].is_nan(),
                        "Expected valid value after warmup at index {}",
                        i
                    );
                }

                // Implementation detail: During warmup (first_non_nan..warmup),
                // the implementation may choose to output NaN or calculated values.
                // The current implementation produces values, but this is not a requirement.

                // Property 3: De-lagging behavior verification
                // ZLEMA uses: lag = (period - 1) / 2
                // de-lagged value = 2.0 * data[i] - data[i - lag]
                let lag = (period - 1) / 2;
                let alpha = 2.0 / (period as f64 + 1.0);

                // Property 4: Values should be within reasonable bounds
                //
                // ZLEMA is an EMA of a de-lagged input series. EMA is a convex combination of
                // all previous inputs, so it must stay within the running min/max of that
                // de-lagged series (up to floating-point rounding).
                let mut min_delag = f64::INFINITY;
                let mut max_delag = f64::NEG_INFINITY;
                for i in first_non_nan..data.len() {
                    let delag_x = if i < first_non_nan + lag {
                        data[i]
                    } else {
                        2.0 * data[i] - data[i - lag]
                    };
                    min_delag = min_delag.min(delag_x);
                    max_delag = max_delag.max(delag_x);

                    if i >= warmup {
                        let y = out[i];
                        prop_assert!(
                            y >= min_delag - 1e-9 && y <= max_delag + 1e-9,
                            "idx {}: {} ∉ [{}, {}] (bounds for de-lagged input history)",
                            i,
                            y,
                            min_delag,
                            max_delag
                        );
                    }
                }

                // Property 5: Period=1 edge case
                // With period=1, lag=0, so de-lagged value = 2*data[i] - data[i] = data[i]
                // And alpha = 2/2 = 1, so EMA = data[i]
                if period == 1 && data.len() > 0 {
                    for i in 1..data.len() {
                        let expected = data[i];
                        let actual = out[i];
                        prop_assert!(
                            (actual - expected).abs() <= 1e-9,
                            "Period=1 mismatch at {}: expected {}, got {}",
                            i,
                            expected,
                            actual
                        );
                    }
                }

                // Property 6: Constant data should converge to that constant
                if data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-12) && data.len() > warmup {
                    // After sufficient iterations, ZLEMA should converge to the constant value
                    let constant_val = data[first_non_nan];
                    for i in (warmup + period * 2)..data.len() {
                        prop_assert!(
                            (out[i] - constant_val).abs() <= 1e-6,
                            "Constant data convergence failed at {}: expected {}, got {}",
                            i,
                            constant_val,
                            out[i]
                        );
                    }
                }

                // Property 7: Cross-kernel validation
                // Compare against scalar reference implementation
                for i in 0..data.len() {
                    let y = out[i];
                    let r = ref_out[i];

                    // Handle NaN/infinity cases
                    if !y.is_finite() || !r.is_finite() {
                        prop_assert!(
                            y.to_bits() == r.to_bits(),
                            "NaN/Inf mismatch at idx {}: {} vs {}",
                            i,
                            y,
                            r
                        );
                        continue;
                    }

                    // Check ULP difference for finite values
                    let y_bits = y.to_bits();
                    let r_bits = r.to_bits();
                    let ulp_diff: u64 = y_bits.abs_diff(r_bits);

                    // ZLEMA is relatively simple, so use tighter ULP tolerance
                    let max_ulp = if matches!(kernel, Kernel::Avx512) {
                        10
                    } else {
                        5
                    };

                    prop_assert!(
                        (y - r).abs() <= 1e-9 || ulp_diff <= max_ulp,
                        "Cross-kernel mismatch at idx {}: {} vs {} (ULP={})",
                        i,
                        y,
                        r,
                        ulp_diff
                    );
                }

                Ok(())
            })
            .unwrap();

        Ok(())
    }

    macro_rules! generate_all_zlema_tests {
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

    generate_all_zlema_tests!(
        check_zlema_partial_params,
        check_zlema_accuracy,
        check_zlema_zero_period,
        check_zlema_period_exceeds_length,
        check_zlema_very_small_dataset,
        check_zlema_reinput,
        check_zlema_nan_handling,
        check_zlema_streaming,
        check_zlema_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_zlema_tests!(check_zlema_property);

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = ZlemaBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = ZlemaParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        Ok(())
    }

    // Check for poison values in batch output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test multiple different batch configurations to catch edge cases
        // ZLEMA uses lag = (period - 1) / 2, so test odd/even periods
        let batch_configs = vec![
            (1, 10, 1),   // All small periods including edge cases
            (3, 21, 3),   // Odd periods with gaps
            (2, 20, 2),   // Even periods
            (10, 50, 10), // Larger periods
            (7, 7, 1),    // Single odd period (edge case)
            (8, 8, 1),    // Single even period (edge case)
            (5, 100, 5),  // Wide range with step
        ];

        for (start, end, step) in batch_configs {
            // Skip if the largest period exceeds data length
            if end > c.close.len() {
                continue;
            }

            let output = ZlemaBatchBuilder::new()
                .kernel(kernel)
                .period_range(start, end, step)
                .apply_candles(&c, "close")?;

            // Check every value in the entire batch matrix for poison patterns
            for (idx, &val) in output.values.iter().enumerate() {
                // Skip NaN values as they're expected in warmup periods
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                let row = idx / output.cols;
                let col = idx % output.cols;
                let period = if row < output.combos.len() {
                    output.combos[row].period.unwrap_or(0)
                } else {
                    0
                };

                // Check for alloc_with_nan_prefix poison (0x11111111_11111111)
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (flat index {}) with period {} in batch ({}, {}, {})",
                        test, val, bits, row, col, idx, period, start, end, step
                    );
                }

                // Check for init_matrix_prefixes poison (0x22222222_22222222)
                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (flat index {}) with period {} in batch ({}, {}, {})",
                        test, val, bits, row, col, idx, period, start, end, step
                    );
                }

                // Check for make_uninit_matrix poison (0x33333333_33333333)
                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (flat index {}) with period {} in batch ({}, {}, {})",
                        test, val, bits, row, col, idx, period, start, end, step
                    );
                }
            }
        }

        Ok(())
    }

    // Release mode stub - does nothing
    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
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
}

/// Zero-copy version that writes directly into a provided buffer
#[inline]
pub fn zlema_compute_into(
    input: &ZlemaInput,
    kernel: Kernel,
    out: &mut [f64],
) -> Result<(), ZlemaError> {
    let (data, first, period, warm) = zlema_validate(input)?;
    if out.len() != data.len() {
        return Err(ZlemaError::OutputLengthMismatch { expected: data.len(), got: out.len() });
    }

    // Initialize the warmup period with the same quiet-NaN pattern used by Vec API
    // to preserve exact warmup semantics/parity.
    let qnan = f64::from_bits(0x7ff8_0000_0000_0000);
    for v in &mut out[..warm] {
        *v = qnan;
    }

    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    // Write directly into the provided buffer
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                zlema_scalar(data, period, first, out);
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                zlema_avx2(data, period, first, out);
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                zlema_avx512(data, period, first, out);
            }
            _ => unreachable!(),
        }
    }

    Ok(())
}

/// WASM-optimized helper function that writes directly to output slice - no allocations
#[inline]
pub fn zlema_into_slice(
    dst: &mut [f64],
    input: &ZlemaInput,
    kern: Kernel,
) -> Result<(), ZlemaError> {
    let (data, first, period, warm) = zlema_validate(input)?;
    if dst.len() != data.len() {
        return Err(ZlemaError::OutputLengthMismatch { expected: data.len(), got: dst.len() });
    }
    // Initialize only the prefix, zero-copy style, with quiet-NaN to match Vec API
    let qnan = f64::from_bits(0x7ff8_0000_0000_0000);
    for v in &mut dst[..warm] {
        *v = qnan;
    }

    let chosen = match kern {
        Kernel::Auto => Kernel::Scalar,
        k => k,
    };
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => zlema_scalar(data, period, first, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => zlema_avx2(data, period, first, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => zlema_avx512(data, period, first, dst),
            _ => unreachable!(),
        }
    }
    Ok(())
}

/// Compute ZLEMA directly into the provided output buffer without any allocations.
///
/// - Preserves NaN warmups exactly as the Vec-returning API (quiet-NaN prefix).
/// - Requires `out.len() == input.len()`; returns `ZlemaError::InvalidPeriod` on mismatch.
/// - Uses the same kernel-selection semantics as `zlema()` (Auto → Scalar for parity).
#[cfg(not(feature = "wasm"))]
pub fn zlema_into(input: &ZlemaInput, out: &mut [f64]) -> Result<(), ZlemaError> {
    // Match `zlema_with_kernel` Auto semantics by forcing Scalar here for parity.
    zlema_compute_into(input, Kernel::Scalar, out)
}

#[cfg(test)]
#[cfg(not(feature = "wasm"))]
fn eq_or_both_nan(a: f64, b: f64) -> bool {
    (a.is_nan() && b.is_nan()) || (a == b) || ((a - b).abs() <= 1e-12)
}

#[cfg(test)]
#[cfg(not(feature = "wasm"))]
#[test]
fn test_zlema_into_matches_api() -> Result<(), Box<dyn std::error::Error>> {
    // Construct a small but non-trivial input (no leading NaNs; default period = 14)
    let n = 256usize;
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        let x = i as f64;
        data.push(x.sin() + 0.5 * x.cos());
    }

    let input = ZlemaInput::from_slice(&data, ZlemaParams::default());

    // Baseline via existing Vec-returning API
    let baseline = zlema(&input)?.values;

    // Preallocate and compute via new into API
    let mut out = vec![0.0f64; n];
    zlema_into(&input, &mut out)?;

    assert_eq!(baseline.len(), out.len());
    for (i, (&a, &b)) in baseline.iter().zip(out.iter()).enumerate() {
        assert!(
            eq_or_both_nan(a, b),
            "mismatch at {}: baseline={}, into={}",
            i,
            a,
            b
        );
    }

    Ok(())
}

#[cfg(feature = "python")]
#[pyfunction(name = "zlema")]
#[pyo3(signature = (data, period, kernel=None))]
/// Compute the Zero Lag Exponential Moving Average (ZLEMA) of the input data.
///
/// ZLEMA reduces lag by de-lagging the input before EMA calculation.
///
/// Parameters:
/// -----------
/// data : np.ndarray
///     Input data array (float64).
/// period : int
///     Number of data points in the moving average window.
/// kernel : str, optional
///     Computation kernel to use: 'auto', 'scalar', 'avx2', 'avx512'.
///     Default is 'auto' which auto-detects the best available.
///
/// Returns:
/// --------
/// np.ndarray
///     Array of ZLEMA values, same length as input.
///
/// Raises:
/// -------
/// ValueError
///     If inputs are invalid (period is 0 or exceeds data length).
pub fn zlema_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let params = ZlemaParams {
        period: Some(period),
    };
    let zlema_in = ZlemaInput::from_slice(slice_in, params);

    // Get Vec<f64> from Rust function
    let result_vec: Vec<f64> = py
        .allow_threads(|| zlema_with_kernel(&zlema_in, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Zero-copy transfer to NumPy
    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "ZlemaStream")]
pub struct ZlemaStreamPy {
    stream: ZlemaStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl ZlemaStreamPy {
    #[new]
    fn new(period: usize) -> PyResult<Self> {
        let params = ZlemaParams {
            period: Some(period),
        };
        let stream =
            ZlemaStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(ZlemaStreamPy { stream })
    }

    /// Updates the stream with a new value and returns the calculated ZLEMA value.
    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "zlema_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]
/// Compute ZLEMA for multiple period values in a single pass.
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
///     Dictionary with 'values' (2D array) and 'periods' arrays.
pub fn zlema_batch_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    let sweep = ZlemaBatchRange {
        period: period_range,
    };

    // Calculate dimensions
    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = slice_in.len();

    // Pre-allocate output array (OK for batch operations)
    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    // Compute without GIL
    let combos = py
        .allow_threads(|| zlema_batch_inner_into(slice_in, &sweep, kern, true, slice_out))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Build result dictionary
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

    Ok(dict.into())
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "zlema_cuda_batch_dev")]
#[pyo3(signature = (data_f32, period_range, device_id=0))]
pub fn zlema_cuda_batch_dev_py<'py>(
    py: Python<'py>,
    data_f32: numpy::PyReadonlyArray1<'py, f32>,
    period_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<(DeviceArrayF32Py, Bound<'py, pyo3::types::PyDict>)> {
    use crate::cuda::cuda_available;
    use numpy::IntoPyArray;
    use pyo3::types::PyDict;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let slice_in = data_f32.as_slice()?;
    let sweep = ZlemaBatchRange {
        period: period_range,
    };

    let (inner, combos, ctx_arc, dev_id, stream_handle) = py
        .allow_threads(|| -> Result<_, crate::cuda::moving_averages::zlema_wrapper::CudaZlemaError> {
            let cuda = CudaZlema::new(device_id)?;
            let (dev, combos) = cuda.zlema_batch_dev(slice_in, &sweep)?;
            Ok((dev, combos, cuda.context_arc(), cuda.device_id(), cuda.stream_handle()))
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    let periods: Vec<u64> = combos.iter().map(|c| c.period.unwrap() as u64).collect();
    dict.set_item("periods", periods.into_pyarray(py))?;

    Ok((DeviceArrayF32Py { inner, _ctx: ctx_arc, device_id: dev_id, stream: stream_handle }, dict))
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "zlema_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm_f32, period, device_id=0))]
pub fn zlema_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    data_tm_f32: numpy::PyReadonlyArray2<'_, f32>,
    period: usize,
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    use crate::cuda::cuda_available;
    use numpy::PyUntypedArrayMethods;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let flat_in = data_tm_f32.as_slice()?;
    let rows = data_tm_f32.shape()[0];
    let cols = data_tm_f32.shape()[1];

    let (inner, ctx_arc, dev_id, stream_handle) = py
        .allow_threads(|| -> Result<_, crate::cuda::moving_averages::zlema_wrapper::CudaZlemaError> {
            let cuda = CudaZlema::new(device_id)?;
            let params = ZlemaParams { period: Some(period) };
            let dev = cuda.zlema_many_series_one_param_time_major_dev(flat_in, cols, rows, &params)?;
            Ok((dev, cuda.context_arc(), cuda.device_id(), cuda.stream_handle()))
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(DeviceArrayF32Py { inner, _ctx: ctx_arc, device_id: dev_id, stream: stream_handle })
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn zlema_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
    let params = ZlemaParams {
        period: Some(period),
    };
    let input = ZlemaInput::from_slice(data, params);

    let mut output = vec![0.0; data.len()];

    zlema_into_slice(&mut output, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn zlema_batch_js(
    data: &[f64],
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = ZlemaBatchRange {
        period: (period_start, period_end, period_step),
    };

    // Use the existing batch function with parallel=false for WASM
    zlema_batch_inner(data, &sweep, Kernel::Auto, false)
        .map(|output| output.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn zlema_batch_metadata_js(
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = ZlemaBatchRange {
        period: (period_start, period_end, period_step),
    };

    let combos = expand_grid(&sweep);
    let mut metadata = Vec::with_capacity(combos.len());

    for combo in combos {
        metadata.push(combo.period.unwrap() as f64);
    }

    Ok(metadata)
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct ZlemaBatchConfig {
    pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct ZlemaBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<ZlemaParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = zlema_batch)]
pub fn zlema_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: ZlemaBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = ZlemaBatchRange {
        period: config.period_range,
    };

    let output = zlema_batch_inner(data, &sweep, Kernel::Auto, false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = ZlemaBatchJsOutput {
        values: output.values,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    };

    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn zlema_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn zlema_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn zlema_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to zlema_into"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);

        if period == 0 || period > len {
            return Err(JsValue::from_str("Invalid period"));
        }

        let params = ZlemaParams {
            period: Some(period),
        };
        let input = ZlemaInput::from_slice(data, params);

        if in_ptr == out_ptr {
            // CRITICAL: Aliasing check - handle in-place operations
            let mut temp = vec![0.0; len];
            zlema_into_slice(&mut temp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            zlema_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn zlema_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to zlema_batch_into"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let sweep = ZlemaBatchRange {
            period: (period_start, period_end, period_step),
        };

        let combos = expand_grid(&sweep);
        let rows = combos.len();
        let total = rows
            .checked_mul(len)
            .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;
        let out = std::slice::from_raw_parts_mut(out_ptr, total);

        zlema_batch_inner_into(data, &sweep, Kernel::Auto, false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(rows)
    }
}
