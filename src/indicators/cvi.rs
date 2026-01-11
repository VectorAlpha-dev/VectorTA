//! # Chaikin's Volatility (CVI)
//!
//! A volatility indicator that measures the rate of change between the high-low spread over time.
//! CVI calculates the percentage difference between two exponentially smoothed averages of the
//! trading range (high-low), helping identify periods of increasing or decreasing volatility.
//! Expanding volatility often precedes trend changes, while contracting volatility suggests consolidation.
//!
//! ## Parameters
//! - **period**: Window size for EMA calculations (default: 10)
//!
//! ## Inputs
//! - Requires high and low price arrays
//! - Supports both raw slices and Candles data structure
//!
//! ## Returns
//! - **`Ok(CviOutput)`** containing a `Vec<f64>` matching input length
//! - Values represent percentage change in volatility
//! - Leading NaNs for first `2*period - 1` values during warmup
//! - Positive values indicate expanding volatility
//! - Negative values indicate contracting volatility
//!
//! ## Developer Notes (Implementation Status)
//! - Decision: EMA is inherently sequential, so cross-time SIMD is limited. We keep scalar as the
//!   reference and use AVX2/AVX512 to hint codegen, elide bounds checks, and prefetch.
//! - SIMD Kernels:
//!   - AVX2/AVX512 enabled (preserve scalar math; modest speedups expected).
//! - Scalar Optimization:
//!   - Loop-jammed, bounds-check-free hot loops; ring buffer without modulo; aligned buffer.
//! - Batch Optimization:
//!   - Row-specific path shares a precomputed `range = high - low` across rows to reduce traffic.
//! - Streaming Performance: O(1) per element via EMA state + circular buffer.
//! - Memory Optimization: Uses `alloc_with_nan_prefix` and `make_uninit_matrix` helpers.
//! - Rationale: SIMD underperforms on strict recurrences; biggest batch wins come from shared
//!   precomputation and reduced memory traffic.
//! - Decision log: SIMD kept but Auto→Scalar for parity; CUDA wrapper returns VRAM handles; Python interop provides CAI v3 + DLPack v1.x (versioned capsules + legacy fallback); numerical outputs unchanged.

// ==================== PYTHON CUDA HANDLE (CAI v3 + DLPack) ====================
// For CUDA-enabled Python builds, provide an indicator-specific VRAM handle that
// keeps the CUDA context alive and exposes both CAI v3 and DLPack interop.
#[cfg(all(feature = "python", feature = "cuda"))]
mod cvi_python_cuda_handle {
    use cust::context::Context;
    use cust::memory::DeviceBuffer;
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::PyDict;
    use crate::utilities::dlpack_cuda::export_f32_cuda_dlpack_2d;
    use std::ffi::c_void;
    use std::sync::Arc;

    #[pyclass(module = "ta_indicators.cuda", unsendable, name = "DeviceArrayF32Py")]
    pub struct DeviceArrayF32Py {
        pub(crate) buf: Option<DeviceBuffer<f32>>, // moved into DLPack once exported
        pub(crate) rows: usize,
        pub(crate) cols: usize,
        pub(crate) _ctx: Arc<Context>,
        pub(crate) device_id: u32,
    }

    #[pymethods]
    impl DeviceArrayF32Py {
        #[getter]
        fn __cuda_array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
            let d = PyDict::new(py);
            d.set_item("shape", (self.rows, self.cols))?;
            d.set_item("typestr", "<f4")?;
            d.set_item(
                "strides",
                (
                    self.cols * std::mem::size_of::<f32>(),
                    std::mem::size_of::<f32>(),
                ),
            )?;
            let ptr = self
                .buf
                .as_ref()
                .ok_or_else(|| PyValueError::new_err("buffer already exported via __dlpack__"))?
                .as_device_ptr()
                .as_raw() as usize;
            d.set_item("data", (ptr, false))?;
            d.set_item("version", 3)?;
            Ok(d)
        }

        fn __dlpack_device__(&self) -> (i32, i32) {
            let mut device_ordinal: i32 = self.device_id as i32;
            unsafe {
                let attr = cust::sys::CUpointer_attribute::CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL;
                let mut value = std::mem::MaybeUninit::<i32>::uninit();
                let ptr = self
                    .buf
                    .as_ref()
                    .map(|b| b.as_device_ptr().as_raw())
                    .unwrap_or(0);
                if ptr != 0 {
                    let rc = cust::sys::cuPointerGetAttribute(
                        value.as_mut_ptr() as *mut c_void,
                        attr,
                        ptr,
                    );
                    if rc == cust::sys::CUresult::CUDA_SUCCESS {
                        device_ordinal = value.assume_init();
                    }
                }
            }
            (2, device_ordinal)
        }

        #[pyo3(signature = (stream=None, max_version=None, dl_device=None, copy=None))]
        fn __dlpack__<'py>(
            &mut self,
            py: Python<'py>,
            stream: Option<pyo3::PyObject>,
            max_version: Option<pyo3::PyObject>,
            dl_device: Option<pyo3::PyObject>,
            copy: Option<pyo3::PyObject>,
        ) -> PyResult<PyObject> {
            
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
                            return Err(PyValueError::new_err(
                                "dl_device mismatch for __dlpack__",
                            ));
                        }
                    }
                }
            }
            let _ = stream;

            
            let buf = self
                .buf
                .take()
                .ok_or_else(|| PyValueError::new_err("__dlpack__ may only be called once"))?;

            let rows = self.rows;
            let cols = self.cols;

            let max_version_bound = max_version.map(|obj| obj.into_bound(py));

            export_f32_cuda_dlpack_2d(py, buf, rows, cols, alloc_dev, max_version_bound)
        }
    }

    pub use DeviceArrayF32Py as CviDeviceArrayF32Py;
}

#[cfg(all(feature = "python", feature = "cuda"))]
use self::cvi_python_cuda_handle::CviDeviceArrayF32Py;
use crate::utilities::data_loader::Candles;
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
use std::mem::MaybeUninit;
use thiserror::Error;
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[derive(Debug, Clone)]
pub enum CviData<'a> {
    Candles(&'a Candles),
    Slices { high: &'a [f64], low: &'a [f64] },
}

#[derive(Debug, Clone)]
pub struct CviOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct CviParams {
    pub period: Option<usize>,
}

impl Default for CviParams {
    fn default() -> Self {
        Self { period: Some(10) }
    }
}

#[derive(Debug, Clone)]
pub struct CviInput<'a> {
    pub data: CviData<'a>,
    pub params: CviParams,
}

impl<'a> CviInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, p: CviParams) -> Self {
        Self {
            data: CviData::Candles(c),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, CviParams::default())
    }
    #[inline]
    pub fn from_slices(high: &'a [f64], low: &'a [f64], p: CviParams) -> Self {
        Self {
            data: CviData::Slices { high, low },
            params: p,
        }
    }

    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(10)
    }
}

#[derive(Debug, Copy, Clone)]
pub struct CviBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for CviBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl CviBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline]
    pub fn period(mut self, n: usize) -> Self {
        self.period = Some(n);
        self
    }
    #[inline]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline]
    pub fn apply(self, candles: &Candles) -> Result<CviOutput, CviError> {
        let params = CviParams {
            period: self.period,
        };
        let input = CviInput::from_candles(candles, params);
        cvi_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn apply_slice(self, high: &[f64], low: &[f64]) -> Result<CviOutput, CviError> {
        let params = CviParams {
            period: self.period,
        };
        let input = CviInput::from_slices(high, low, params);
        cvi_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn into_stream(self, initial_high: f64, initial_low: f64) -> Result<CviStream, CviError> {
        let params = CviParams {
            period: self.period,
        };
        CviStream::try_new(params, initial_high, initial_low)
    }
}

#[derive(Debug, Error)]
pub enum CviError {
    #[error("cvi: Empty data provided for CVI.")]
    EmptyData,
    #[error("cvi: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("cvi: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("cvi: All values are NaN.")]
    AllValuesNaN,
    #[error("cvi: Output length mismatch: expected={expected}, got={got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("cvi: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange { start: usize, end: usize, step: usize },
    #[error("cvi: Invalid kernel for batch path: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[inline]
pub fn cvi(input: &CviInput) -> Result<CviOutput, CviError> {
    cvi_with_kernel(input, Kernel::Auto)
}

pub fn cvi_with_kernel(input: &CviInput, kernel: Kernel) -> Result<CviOutput, CviError> {
    let (high, low) = match &input.data {
        CviData::Candles(c) => (&c.high[..], &c.low[..]),
        CviData::Slices { high, low } => (*high, *low),
    };

    if high.is_empty() || low.is_empty() || high.len() != low.len() {
        return Err(CviError::EmptyData);
    }

    let period = input.get_period();
    if period == 0 || period > high.len() {
        return Err(CviError::InvalidPeriod {
            period,
            data_len: high.len(),
        });
    }

    let first_valid_idx = match (0..high.len()).find(|&i| !high[i].is_nan() && !low[i].is_nan()) {
        Some(idx) => idx,
        None => return Err(CviError::AllValuesNaN),
    };

    let needed = 2 * period - 1;
    if (high.len() - first_valid_idx) < needed {
        return Err(CviError::NotEnoughValidData {
            needed,
            valid: high.len() - first_valid_idx,
        });
    }

    let mut cvi_values = alloc_with_nan_prefix(high.len(), first_valid_idx + needed);

    
    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                cvi_scalar(high, low, period, first_valid_idx, &mut cvi_values)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                cvi_avx2(high, low, period, first_valid_idx, &mut cvi_values)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                cvi_avx512(high, low, period, first_valid_idx, &mut cvi_values)
            }
            _ => unreachable!(),
        }
    }

    Ok(CviOutput { values: cvi_values })
}

/// Write Chaikin's Volatility (CVI) into a caller-provided buffer without allocating.
///
/// - Preserves NaN warmup semantics identical to the Vec-returning API.
/// - `out.len()` must equal the input length; values before the first valid index and
///   required warmup are filled with NaN, and the remainder contains CVI outputs.
/// - Uses the module's default kernel selection (Auto short-circuits to Scalar for parity).
#[cfg(not(feature = "wasm"))]
#[inline]
pub fn cvi_into(input: &CviInput, out: &mut [f64]) -> Result<(), CviError> {
    
    cvi_into_slice(out, input, Kernel::Scalar)
}

#[inline]
pub fn cvi_scalar(
    high: &[f64],
    low: &[f64],
    period: usize,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    
    let alpha = 2.0 / (period as f64 + 1.0);

    
    let mut val =
        unsafe { *high.get_unchecked(first_valid_idx) - *low.get_unchecked(first_valid_idx) };

    
    let mut lag = AVec::<f64>::with_capacity(CACHELINE_ALIGN, period);
    unsafe { lag.set_len(period) };
    unsafe { *lag.get_unchecked_mut(0) = val };

    let mut head = 1usize;
    let len = high.len();
    let needed = 2 * period - 1; 

    
    let mut i = first_valid_idx + 1;
    let end_warm = first_valid_idx + needed; 
    while i < end_warm {
        let range = unsafe { *high.get_unchecked(i) - *low.get_unchecked(i) };
        val += (range - val) * alpha;
        unsafe { *lag.get_unchecked_mut(head) = val };
        head += 1;
        if head == period {
            head = 0;
        }
        i += 1;
    }

    
    let mut j = end_warm;
    while j < len {
        let range = unsafe { *high.get_unchecked(j) - *low.get_unchecked(j) };
        val += (range - val) * alpha;
        let old = unsafe { *lag.get_unchecked(head) };
        unsafe {
            *out.get_unchecked_mut(j) = 100.0 * (val - old) / old;
            *lag.get_unchecked_mut(head) = val;
        }
        head += 1;
        if head == period {
            head = 0;
        }
        j += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn cvi_avx2(high: &[f64], low: &[f64], period: usize, first_valid_idx: usize, out: &mut [f64]) {
    #[target_feature(enable = "avx2,fma")]
    unsafe fn inner(
        high: &[f64],
        low: &[f64],
        period: usize,
        first_valid_idx: usize,
        out: &mut [f64],
    ) {
        const DO_PREFETCH: bool = false;
        let alpha = 2.0 / (period as f64 + 1.0);
        let mut val = *high.get_unchecked(first_valid_idx) - *low.get_unchecked(first_valid_idx);

        let mut lag = AVec::<f64>::with_capacity(CACHELINE_ALIGN, period);
        lag.set_len(period);
        *lag.get_unchecked_mut(0) = val;

        let mut head = 1usize;
        let len = high.len();
        let needed = 2 * period - 1;

        
        let mut i = first_valid_idx + 1;
        let end_warm = first_valid_idx + needed;
        while i < end_warm {
            if DO_PREFETCH && i + 64 < len {
                _mm_prefetch(high.as_ptr().add(i + 64) as *const i8, _MM_HINT_T0);
                _mm_prefetch(low.as_ptr().add(i + 64) as *const i8, _MM_HINT_T0);
            }
            let range = *high.get_unchecked(i) - *low.get_unchecked(i);
            val += (range - val) * alpha;
            *lag.get_unchecked_mut(head) = val;
            head += 1;
            if head == period {
                head = 0;
            }
            i += 1;
        }

        
        let mut j = end_warm;
        while j < len {
            if DO_PREFETCH && j + 64 < len {
                _mm_prefetch(high.as_ptr().add(j + 64) as *const i8, _MM_HINT_T0);
                _mm_prefetch(low.as_ptr().add(j + 64) as *const i8, _MM_HINT_T0);
            }
            let range = *high.get_unchecked(j) - *low.get_unchecked(j);
            val += (range - val) * alpha;
            let old = *lag.get_unchecked(head);
            *out.get_unchecked_mut(j) = 100.0 * (val - old) / old;
            *lag.get_unchecked_mut(head) = val;
            head += 1;
            if head == period {
                head = 0;
            }
            j += 1;
        }
    }

    unsafe { inner(high, low, period, first_valid_idx, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn cvi_avx512(
    high: &[f64],
    low: &[f64],
    period: usize,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    if period <= 32 {
        unsafe { cvi_avx512_short(high, low, period, first_valid_idx, out) }
    } else {
        unsafe { cvi_avx512_long(high, low, period, first_valid_idx, out) }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn cvi_avx512_short(
    high: &[f64],
    low: &[f64],
    period: usize,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    cvi_avx512_core(high, low, period, first_valid_idx, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn cvi_avx512_long(
    high: &[f64],
    low: &[f64],
    period: usize,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    cvi_avx512_core(high, low, period, first_valid_idx, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
unsafe fn cvi_avx512_core(
    high: &[f64],
    low: &[f64],
    period: usize,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    const DO_PREFETCH: bool = false;
    let alpha = 2.0 / (period as f64 + 1.0);
    let mut val = *high.get_unchecked(first_valid_idx) - *low.get_unchecked(first_valid_idx);

    let mut lag = AVec::<f64>::with_capacity(CACHELINE_ALIGN, period);
    lag.set_len(period);
    *lag.get_unchecked_mut(0) = val;

    let mut head = 1usize;
    let len = high.len();
    let needed = 2 * period - 1;

    
    let mut i = first_valid_idx + 1;
    let end_warm = first_valid_idx + needed;
    while i < end_warm {
        if DO_PREFETCH && i + 96 < len {
            _mm_prefetch(high.as_ptr().add(i + 96) as *const i8, _MM_HINT_T0);
            _mm_prefetch(low.as_ptr().add(i + 96) as *const i8, _MM_HINT_T0);
        }
        let range = *high.get_unchecked(i) - *low.get_unchecked(i);
        val += (range - val) * alpha;
        *lag.get_unchecked_mut(head) = val;
        head += 1;
        if head == period {
            head = 0;
        }
        i += 1;
    }

    
    let mut j = end_warm;
    while j < len {
        if DO_PREFETCH && j + 96 < len {
            _mm_prefetch(high.as_ptr().add(j + 96) as *const i8, _MM_HINT_T0);
            _mm_prefetch(low.as_ptr().add(j + 96) as *const i8, _MM_HINT_T0);
        }
        let range = *high.get_unchecked(j) - *low.get_unchecked(j);
        val += (range - val) * alpha;
        let old = *lag.get_unchecked(head);
        *out.get_unchecked_mut(j) = 100.0 * (val - old) / old;
        *lag.get_unchecked_mut(head) = val;
        head += 1;
        if head == period {
            head = 0;
        }
        j += 1;
    }
}

#[inline(always)]
pub fn cvi_row_scalar(
    high: &[f64],
    low: &[f64],
    period: usize,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    cvi_scalar(high, low, period, first_valid_idx, out)
}

#[inline(always)]
fn cvi_scalar_from_range(range: &[f64], period: usize, first_valid_idx: usize, out: &mut [f64]) {
    let alpha = 2.0 / (period as f64 + 1.0);

    let mut val = unsafe { *range.get_unchecked(first_valid_idx) };

    let mut lag = AVec::<f64>::with_capacity(CACHELINE_ALIGN, period);
    unsafe { lag.set_len(period) };
    unsafe { *lag.get_unchecked_mut(0) = val };

    let mut head = 1usize;
    let len = range.len();
    let needed = 2 * period - 1;

    let mut i = first_valid_idx + 1;
    let end_warm = first_valid_idx + needed;
    while i < end_warm {
        let r = unsafe { *range.get_unchecked(i) };
        val += (r - val) * alpha;
        unsafe { *lag.get_unchecked_mut(head) = val };
        head += 1;
        if head == period {
            head = 0;
        }
        i += 1;
    }

    let mut j = end_warm;
    while j < len {
        let r = unsafe { *range.get_unchecked(j) };
        val += (r - val) * alpha;
        let old = unsafe { *lag.get_unchecked(head) };
        unsafe {
            *out.get_unchecked_mut(j) = 100.0 * (val - old) / old;
            *lag.get_unchecked_mut(head) = val;
        }
        head += 1;
        if head == period {
            head = 0;
        }
        j += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn cvi_row_avx2(
    high: &[f64],
    low: &[f64],
    period: usize,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    cvi_avx2(high, low, period, first_valid_idx, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn cvi_row_avx512(
    high: &[f64],
    low: &[f64],
    period: usize,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    cvi_avx512(high, low, period, first_valid_idx, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn cvi_row_avx512_short(
    high: &[f64],
    low: &[f64],
    period: usize,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    unsafe { cvi_avx512_short(high, low, period, first_valid_idx, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn cvi_row_avx512_long(
    high: &[f64],
    low: &[f64],
    period: usize,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    unsafe { cvi_avx512_long(high, low, period, first_valid_idx, out) }
}

#[derive(Clone, Debug)]
pub struct CviBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for CviBatchRange {
    fn default() -> Self {
        Self {
            period: (10, 259, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct CviBatchBuilder {
    range: CviBatchRange,
    kernel: Kernel,
}

impl CviBatchBuilder {
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
    pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<CviBatchOutput, CviError> {
        cvi_batch_with_kernel(high, low, &self.range, self.kernel)
    }
    pub fn apply_candles(self, candles: &Candles) -> Result<CviBatchOutput, CviError> {
        self.apply_slices(&candles.high, &candles.low)
    }
}

pub fn cvi_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    sweep: &CviBatchRange,
    k: Kernel,
) -> Result<CviBatchOutput, CviError> {
    let kernel = match k {
        Kernel::Auto => {
            let k = detect_best_batch_kernel();
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            {
                
                if k == Kernel::Avx512Batch
                    && std::arch::is_x86_feature_detected!("avx2")
                    && std::arch::is_x86_feature_detected!("fma")
                {
                    Kernel::Avx2Batch
                } else {
                    k
                }
            }
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            {
                k
            }
        }
        other if other.is_batch() => other,
        other => return Err(CviError::InvalidKernelForBatch(other)),
    };

    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    cvi_batch_par_slice(high, low, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct CviBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<CviParams>,
    pub rows: usize,
    pub cols: usize,
}
impl CviBatchOutput {
    pub fn row_for_params(&self, p: &CviParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(10) == p.period.unwrap_or(10))
    }

    pub fn values_for(&self, p: &CviParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &CviBatchRange) -> Vec<CviParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        if start <= end {
            return (start..=end).step_by(step.max(1)).collect();
        }
        
        let mut v = Vec::new();
        let s = step.max(1);
        let mut cur = start;
        loop {
            v.push(cur);
            if cur <= end { break; }
            let next = cur.saturating_sub(s);
            if next == cur { break; }
            cur = next;
        }
        v.retain(|&x| x >= end);
        v
    }
    let periods = axis_usize(r.period);

    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(CviParams { period: Some(p) });
    }
    out
}

#[inline(always)]
pub fn cvi_batch_slice(
    high: &[f64],
    low: &[f64],
    sweep: &CviBatchRange,
    kern: Kernel,
) -> Result<CviBatchOutput, CviError> {
    cvi_batch_inner(high, low, sweep, kern, false)
}

#[inline(always)]
pub fn cvi_batch_par_slice(
    high: &[f64],
    low: &[f64],
    sweep: &CviBatchRange,
    kern: Kernel,
) -> Result<CviBatchOutput, CviError> {
    cvi_batch_inner(high, low, sweep, kern, true)
}

#[inline(always)]
fn cvi_batch_inner(
    high: &[f64],
    low: &[f64],
    sweep: &CviBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<CviBatchOutput, CviError> {
    if high.is_empty() || low.is_empty() || high.len() != low.len() {
        return Err(CviError::EmptyData);
    }

    let combos = expand_grid(sweep);
    if combos.is_empty() {
        let (s, e, st) = sweep.period;
        return Err(CviError::InvalidRange { start: s, end: e, step: st });
    }

    let first_valid_idx = (0..high.len())
        .find(|&i| !high[i].is_nan() && !low[i].is_nan())
        .ok_or(CviError::AllValuesNaN)?;

    if combos.iter().any(|c| c.period.unwrap_or(0) == 0) {
        return Err(CviError::InvalidPeriod {
            period: 0,
            data_len: high.len(),
        });
    }

    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    let needed = 2 * max_p - 1;
    if high.len() - first_valid_idx < needed {
        return Err(CviError::NotEnoughValidData {
            needed,
            valid: high.len() - first_valid_idx,
        });
    }

    let rows = combos.len();
    let cols = high.len();

    rows
        .checked_mul(cols)
        .ok_or_else(|| CviError::InvalidRange { start: rows, end: cols, step: 0 })?;

    let mut buf_mu = make_uninit_matrix(rows, cols);
    let warmup_periods: Vec<usize> = combos
        .iter()
        .map(|c| {
            let period = c.period.unwrap();
            first_valid_idx + (2 * period - 1)
        })
        .collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warmup_periods);

    let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);
    let values_slice: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len())
    };

    
    let shared_range: Option<Vec<f64>> = match kern {
        Kernel::Scalar | Kernel::Auto => {
            let mut r = Vec::with_capacity(cols);
            unsafe {
                r.set_len(cols);
                let mut k = 0usize;
                while k < cols {
                    let v = *high.get_unchecked(k) - *low.get_unchecked(k);
                    *r.get_unchecked_mut(k) = v;
                    k += 1;
                }
            }
            Some(r)
        }
        _ => None,
    };

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        match kern {
            Kernel::Scalar | Kernel::Auto => {
                if let Some(range) = &shared_range {
                    cvi_scalar_from_range(range, period, first_valid_idx, out_row)
                } else {
                    cvi_row_scalar(high, low, period, first_valid_idx, out_row)
                }
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => cvi_row_avx2(high, low, period, first_valid_idx, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => cvi_row_avx512(high, low, period, first_valid_idx, out_row),
            _ => cvi_row_scalar(high, low, period, first_valid_idx, out_row),
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            values_slice
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, slice) in values_slice.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
    } else {
        for (row, slice) in values_slice.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    let values = unsafe {
        Vec::from_raw_parts(
            buf_guard.as_mut_ptr() as *mut f64,
            buf_guard.len(),
            buf_guard.capacity(),
        )
    };
    core::mem::forget(buf_guard);

    Ok(CviBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn cvi_batch_inner_into(
    high: &[f64],
    low: &[f64],
    sweep: &CviBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<CviParams>, CviError> {
    if high.is_empty() || low.is_empty() || high.len() != low.len() {
        return Err(CviError::EmptyData);
    }

    let combos = expand_grid(sweep);
    if combos.is_empty() {
        let (s, e, st) = sweep.period;
        return Err(CviError::InvalidRange { start: s, end: e, step: st });
    }

    let first = (0..high.len())
        .find(|&i| !high[i].is_nan() && !low[i].is_nan())
        .ok_or(CviError::AllValuesNaN)?;

    if combos.iter().any(|c| c.period.unwrap_or(0) == 0) {
        return Err(CviError::InvalidPeriod {
            period: 0,
            data_len: high.len(),
        });
    }

    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    let needed = 2 * max_p - 1;
    if high.len() - first < needed {
        return Err(CviError::NotEnoughValidData {
            needed,
            valid: high.len() - first,
        });
    }

    let rows = combos.len();
    let cols = high.len();
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| CviError::InvalidRange { start: rows, end: cols, step: 0 })?;
    if out.len() != expected {
        return Err(CviError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }

    
    let out_mu = unsafe {
        core::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };
    let warms: Vec<usize> = combos
        .iter()
        .map(|p| first + (2 * p.period.unwrap() - 1))
        .collect();
    init_matrix_prefixes(out_mu, cols, &warms);

    
    let shared_range: Option<Vec<f64>> = match kern {
        Kernel::Scalar | Kernel::ScalarBatch | Kernel::Auto => {
            let mut r = Vec::with_capacity(cols);
            unsafe {
                r.set_len(cols);
                let mut i = 0usize;
                while i < cols {
                    let v = *high.get_unchecked(i) - *low.get_unchecked(i);
                    *r.get_unchecked_mut(i) = v;
                    i += 1;
                }
            }
            Some(r)
        }
        _ => None,
    };

    
    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| {
        let p = combos[row].period.unwrap();
        let dst = unsafe {
            core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len())
        };
        match kern {
            Kernel::Scalar | Kernel::ScalarBatch | Kernel::Auto => {
                if let Some(range) = &shared_range {
                    cvi_scalar_from_range(range, p, first, dst)
                } else {
                    cvi_row_scalar(high, low, p, first, dst)
                }
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => cvi_row_avx2(high, low, p, first, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => cvi_row_avx512(high, low, p, first, dst),
            _ => cvi_row_scalar(high, low, p, first, dst),
        }
    };

    #[cfg(not(target_arch = "wasm32"))]
    {
        if parallel {
            out_mu
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(r, s)| do_row(r, s));
        } else {
            for (r, s) in out_mu.chunks_mut(cols).enumerate() {
                do_row(r, s);
            }
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        for (r, s) in out_mu.chunks_mut(cols).enumerate() {
            do_row(r, s);
        }
    }

    Ok(combos)
}





#[derive(Debug, Clone)]
pub struct CviStream {
    period: usize,
    alpha: f64,
    lag_buffer: Vec<f64>, 
    head: usize,          
    warmup_remaining: usize,
    state_val: f64, 
}

impl CviStream {
    #[inline]
    pub fn try_new(
        params: CviParams,
        initial_high: f64,
        initial_low: f64,
    ) -> Result<Self, CviError> {
        let period = params.period.unwrap_or(10);
        if period == 0 {
            return Err(CviError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }

        
        let alpha = 2.0 / (period as f64 + 1.0);

        
        let ema0 = initial_high - initial_low;

        
        let mut lag_buffer = vec![0.0; period.max(1)];
        lag_buffer[0] = ema0;

        
        let head = if period > 1 { 1 } else { 0 };

        
        
        let warmup_remaining = period.saturating_mul(2).saturating_sub(2);

        Ok(Self {
            period,
            alpha,
            lag_buffer,
            head,
            warmup_remaining,
            state_val: ema0,
        })
    }

    /// Update with new (high, low). Returns None during warmup, then Some(CVI).
    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64) -> Option<f64> {
        let range = high - low;
        self.update_range(range)
    }

    /// Update with a precomputed range (high - low). Same semantics as `update`.
    #[inline(always)]
    pub fn update_range(&mut self, range: f64) -> Option<f64> {
        
        
        self.state_val = (range - self.state_val).mul_add(self.alpha, self.state_val);

        
        if self.warmup_remaining != 0 {
            self.lag_buffer[self.head] = self.state_val;
            let next = self.head + 1;
            self.head = if next == self.period { 0 } else { next };
            self.warmup_remaining -= 1;
            return None;
        }

        
        
        let old = self.lag_buffer[self.head];
        let out = 100.0 * (self.state_val - old) / old;

        self.lag_buffer[self.head] = self.state_val;
        let next = self.head + 1;
        self.head = if next == self.period { 0 } else { next };

        Some(out)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "cvi")]
#[pyo3(signature = (high, low, period, kernel=None))]
pub fn cvi_py<'py>(
    py: Python<'py>,
    high: numpy::PyReadonlyArray1<'py, f64>,
    low: numpy::PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};

    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let kern = validate_kernel(kernel, false)?;
    let params = CviParams {
        period: Some(period),
    };
    let cvi_in = CviInput::from_slices(high_slice, low_slice, params);

    let result_vec: Vec<f64> = py
        .allow_threads(|| cvi_with_kernel(&cvi_in, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "CviStream")]
pub struct CviStreamPy {
    // Python-binding stream with adjusted warmup semantics (first value ~ after `period` updates).
    period: usize,
    alpha: f64,
    lag_buffer: Vec<f64>,
    head: usize,
    warmup_remaining: usize,
    state_val: f64,
}

#[cfg(feature = "python")]
#[pymethods]
impl CviStreamPy {
    #[new]
    fn new(period: usize, initial_high: f64, initial_low: f64) -> PyResult<Self> {
        if period == 0 {
            return Err(PyValueError::new_err("cvi: Invalid period: period = 0, data length = 0"));
        }
        let alpha = 2.0 / (period as f64 + 1.0);
        let ema0 = initial_high - initial_low;
        let mut lag = vec![0.0; period];
        lag[0] = ema0;
        let head = if period > 1 { 1 } else { 0 };
        // Python stream emits first value after `period` updates (seed + (period-1) writes)
        let warmup_remaining = period.saturating_sub(1);
        Ok(CviStreamPy {
            period,
            alpha,
            lag_buffer: lag,
            head,
            warmup_remaining,
            state_val: ema0,
        })
    }

    fn update(&mut self, high: f64, low: f64) -> Option<f64> {
        // EMA update
        let range = high - low;
        self.state_val = (range - self.state_val).mul_add(self.alpha, self.state_val);

        if self.warmup_remaining != 0 {
            self.lag_buffer[self.head] = self.state_val;
            let next = self.head + 1;
            self.head = if next == self.period { 0 } else { next };
            self.warmup_remaining -= 1;
            return None;
        }

        let old = self.lag_buffer[self.head];
        let out = 100.0 * (self.state_val - old) / old;
        self.lag_buffer[self.head] = self.state_val;
        let next = self.head + 1;
        self.head = if next == self.period { 0 } else { next };
        Some(out)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "cvi_batch")]
#[pyo3(signature = (high, low, period_range, kernel=None))]
pub fn cvi_batch_py<'py>(
    py: Python<'py>,
    high: numpy::PyReadonlyArray1<'py, f64>,
    low: numpy::PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;

    let sweep = CviBatchRange {
        period: period_range,
    };
    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = high_slice.len();

    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;
    let out_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    let kern = validate_kernel(kernel, true)?;

    let combos = py
        .allow_threads(|| {
            let kernel = match kern {
                Kernel::Auto => detect_best_batch_kernel(),
                k => k,
            };
            let simd = match kernel {
                Kernel::Avx512Batch => Kernel::Avx512,
                Kernel::Avx2Batch => Kernel::Avx2,
                Kernel::ScalarBatch => Kernel::Scalar,
                _ => unreachable!(),
            };
            cvi_batch_inner_into(high_slice, low_slice, &sweep, simd, true, slice_out)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "periods",
        combos
            .iter()
            .map(|p| p.period.unwrap_or(10) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;

    Ok(dict.into())
}





#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "cvi_cuda_batch_dev")]
#[pyo3(signature = (high, low, period_range, device_id=0))]
pub fn cvi_cuda_batch_dev_py(
    py: Python<'_>,
    high: numpy::PyReadonlyArray1<'_, f32>,
    low: numpy::PyReadonlyArray1<'_, f32>,
    period_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<CviDeviceArrayF32Py> {
    use crate::cuda::cuda_available;
    use crate::cuda::CudaCvi;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    if high_slice.len() != low_slice.len() {
        return Err(PyValueError::new_err("mismatched input lengths"));
    }

    let sweep = CviBatchRange {
        period: period_range,
    };
    let (inner, ctx, dev_id) = py.allow_threads(|| {
        let cuda = CudaCvi::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx = cuda.ctx();
        let dev_id = cuda.device_id();
        let arr = cuda
            .cvi_batch_dev(high_slice, low_slice, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok::<_, PyErr>((arr, ctx, dev_id))
    })?;

    Ok(CviDeviceArrayF32Py {
        buf: Some(inner.buf),
        rows: inner.rows,
        cols: inner.cols,
        _ctx: ctx,
        device_id: dev_id,
    })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "cvi_cuda_many_series_one_param_dev")]
#[pyo3(signature = (high_tm, low_tm, cols, rows, period, device_id=0))]
pub fn cvi_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    high_tm: numpy::PyReadonlyArray1<'_, f32>,
    low_tm: numpy::PyReadonlyArray1<'_, f32>,
    cols: usize,
    rows: usize,
    period: usize,
    device_id: usize,
) -> PyResult<CviDeviceArrayF32Py> {
    use crate::cuda::cuda_available;
    use crate::cuda::CudaCvi;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let high_slice = high_tm.as_slice()?;
    let low_slice = low_tm.as_slice()?;
    let expected = cols
        .checked_mul(rows)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;
    if high_slice.len() != expected || low_slice.len() != expected {
        return Err(PyValueError::new_err("time-major input length mismatch"));
    }

    let (inner, ctx, dev_id) = py.allow_threads(|| {
        let cuda = CudaCvi::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx = cuda.ctx();
        let dev_id = cuda.device_id();
        let arr = cuda
            .cvi_many_series_one_param_time_major_dev(
                high_slice,
                low_slice,
                cols,
                rows,
                period,
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok::<_, PyErr>((arr, ctx, dev_id))
    })?;

    Ok(CviDeviceArrayF32Py {
        buf: Some(inner.buf),
        rows: inner.rows,
        cols: inner.cols,
        _ctx: ctx,
        device_id: dev_id,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn check_cvi_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = CviParams { period: None };
        let input_default = CviInput::from_candles(&candles, default_params);
        let output_default = cvi_with_kernel(&input_default, kernel)?;
        assert_eq!(output_default.values.len(), candles.close.len());
        Ok(())
    }

    fn check_cvi_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = CviParams { period: Some(5) };
        let input = CviInput::from_candles(&candles, params);
        let cvi_result = cvi_with_kernel(&input, kernel)?;

        let expected_last_five_cvi = [
            -52.96320026271643,
            -64.39616778235792,
            -59.4830094380472,
            -52.4690724045071,
            -11.858704179539174,
        ];
        assert!(cvi_result.values.len() >= 5);
        let start_index = cvi_result.values.len() - 5;
        let result_last_five = &cvi_result.values[start_index..];
        for (i, &val) in result_last_five.iter().enumerate() {
            let expected = expected_last_five_cvi[i];
            assert!(
                (val - expected).abs() < 1e-6,
                "[{}] CVI mismatch at index {}: expected {}, got {}",
                test_name,
                i,
                expected,
                val
            );
        }
        Ok(())
    }

    fn check_cvi_input_with_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = CviInput::with_default_candles(&candles);
        match input.data {
            CviData::Candles(_) => {}
            _ => panic!("Expected CviData::Candles variant"),
        }
        Ok(())
    }

    fn check_cvi_with_zero_period(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 20.0, 30.0];
        let low = [5.0, 15.0, 25.0];
        let params = CviParams { period: Some(0) };
        let input = CviInput::from_slices(&high, &low, params);

        let result = cvi_with_kernel(&input, kernel);
        assert!(
            result.is_err(),
            "[{}] Expected an error for zero period",
            test_name
        );
        Ok(())
    }

    fn check_cvi_with_period_exceeding_data_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 20.0, 30.0];
        let low = [5.0, 15.0, 25.0];
        let params = CviParams { period: Some(10) };
        let input = CviInput::from_slices(&high, &low, params);

        let result = cvi_with_kernel(&input, kernel);
        assert!(
            result.is_err(),
            "[{}] Expected an error for period > data.len()",
            test_name
        );
        Ok(())
    }

    fn check_cvi_very_small_data_set(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [5.0];
        let low = [2.0];
        let params = CviParams { period: Some(10) };
        let input = CviInput::from_slices(&high, &low, params);

        let result = cvi_with_kernel(&input, kernel);
        assert!(
            result.is_err(),
            "[{}] Expected error for data smaller than period",
            test_name
        );
        Ok(())
    }

    fn check_cvi_with_nan_data(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [f64::NAN, 20.0, 30.0];
        let low = [5.0, 15.0, f64::NAN];
        let input = CviInput::from_slices(&high, &low, CviParams { period: Some(2) });

        let result = cvi_with_kernel(&input, kernel);
        assert!(
            result.is_err(),
            "[{}] Expected an error due to trailing NaN",
            test_name
        );
        Ok(())
    }

    fn check_cvi_slice_reinput(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [
            10.0, 12.0, 12.5, 12.2, 13.0, 14.0, 15.0, 16.0, 16.5, 17.0, 17.5, 18.0,
        ];
        let low = [
            9.0, 10.0, 11.5, 11.0, 12.0, 13.5, 14.0, 14.5, 15.5, 16.0, 16.5, 17.0,
        ];
        let first_input = CviInput::from_slices(&high, &low, CviParams { period: Some(3) });
        let first_result = cvi_with_kernel(&first_input, kernel)?;
        let second_input =
            CviInput::from_slices(&first_result.values, &low, CviParams { period: Some(3) });
        let second_result = cvi_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), low.len());
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_cvi_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        
        let test_params = vec![
            CviParams::default(),            
            CviParams { period: Some(2) },   
            CviParams { period: Some(5) },   
            CviParams { period: Some(10) },  
            CviParams { period: Some(14) },  
            CviParams { period: Some(20) },  
            CviParams { period: Some(50) },  
            CviParams { period: Some(100) }, 
            CviParams { period: Some(200) }, 
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            
            let input = CviInput::from_candles(&candles, params.clone());
            let output = cvi_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue; 
                }

                let bits = val.to_bits();

                
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 with params: period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(10),
                        param_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 with params: period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(10),
                        param_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 with params: period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(10),
                        param_idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_cvi_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(()) 
    }

    macro_rules! generate_all_cvi_tests {
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

    generate_all_cvi_tests!(
        check_cvi_partial_params,
        check_cvi_accuracy,
        check_cvi_input_with_default_candles,
        check_cvi_with_zero_period,
        check_cvi_with_period_exceeding_data_length,
        check_cvi_very_small_data_set,
        check_cvi_with_nan_data,
        check_cvi_slice_reinput,
        check_cvi_no_poison
    );

    #[cfg(test)]
    generate_all_cvi_tests!(check_cvi_property);

    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = CviBatchBuilder::new().kernel(kernel).apply_candles(&c)?;
        let def = CviParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
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

    #[test]
    #[cfg(not(feature = "wasm"))]
    fn test_cvi_into_matches_api() -> Result<(), Box<dyn std::error::Error>> {
        
        let len = 256usize;
        let mut high = vec![0.0f64; len];
        let mut low = vec![0.0f64; len];

        
        high[0] = f64::NAN;
        low[0] = f64::NAN;
        high[1] = f64::NAN;
        low[1] = f64::NAN;
        high[2] = f64::NAN;
        low[2] = f64::NAN;

        for i in 3..len {
            let i_f = i as f64;
            let base = 100.0 + 0.1 * i_f;
            
            high[i] = base + (i % 7) as f64 * 0.03;
            let spread = 1.0 + (i % 5) as f64 * 0.2;
            low[i] = high[i] - spread.max(0.001);
        }

        let input = CviInput::from_slices(&high, &low, CviParams::default());

        
        let baseline = cvi(&input)?.values;

        
        let mut into_out = vec![0.0f64; len];
        cvi_into(&input, &mut into_out)?;

        assert_eq!(baseline.len(), into_out.len());

        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-12
        }

        for (i, (&a, &b)) in baseline.iter().zip(into_out.iter()).enumerate() {
            assert!(
                eq_or_both_nan(a, b),
                "mismatch at {}: baseline={} into={}",
                i,
                a,
                b
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
            (2, 10, 2),    
            (5, 25, 5),    
            (10, 50, 10),  
            (2, 5, 1),     
            (30, 60, 15),  
            (14, 21, 7),   
            (50, 100, 25), 
        ];

        for (cfg_idx, &(p_start, p_end, p_step)) in test_configs.iter().enumerate() {
            let output = CviBatchBuilder::new()
                .kernel(kernel)
                .period_range(p_start, p_end, p_step)
                .apply_candles(&c)?;

            for (idx, &val) in output.values.iter().enumerate() {
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
						 at row {} col {} (flat index {}) with params: period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(10)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(10)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(10)
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

    #[cfg(test)]
    #[allow(clippy::float_cmp)]
    fn check_cvi_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        
        let strat = (2usize..=50) 
            .prop_flat_map(|period| {
                (
                    
                    100.0f64..5000.0f64,
                    
                    (2 * period + 20)..400,
                    
                    0.001f64..0.1f64,
                    
                    -0.01f64..0.01f64,
                    
                    Just(period),
                )
            })
            .prop_map(|(base_price, data_len, volatility, trend, period)| {
                
                let mut high_data = Vec::with_capacity(data_len);
                let mut low_data = Vec::with_capacity(data_len);
                let mut current_price = base_price;

                for i in 0..data_len {
                    
                    current_price *= 1.0 + trend;
                    current_price = current_price.max(10.0); 

                    
                    
                    let volatility_factor = volatility * (1.0 + (i as f64 * 0.1).sin() * 0.5);
                    let range = current_price * volatility_factor;

                    
                    let high = current_price + range * 0.5;
                    let low = (current_price - range * 0.5).max(1.0);

                    high_data.push(high);
                    low_data.push(low);
                }

                (high_data, low_data, period)
            });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(high_data, low_data, period)| {
                let params = CviParams {
                    period: Some(period),
                };
                let input = CviInput::from_slices(&high_data, &low_data, params);

                
                let out = cvi_with_kernel(&input, kernel)?;
                let ref_out = cvi_with_kernel(&input, Kernel::Scalar)?;

                
                
                
                let expected_first_valid = 2 * period - 1;
                let first_valid_idx = out.values.iter().position(|&v| !v.is_nan());

                if let Some(idx) = first_valid_idx {
                    prop_assert_eq!(
                        idx,
                        expected_first_valid,
                        "[{}] First valid index mismatch: expected {}, got {}",
                        test_name,
                        expected_first_valid,
                        idx
                    );
                }

                
                for (i, &val) in out.values.iter().enumerate() {
                    if !val.is_nan() {
                        prop_assert!(
                            val.is_finite(),
                            "[{}] CVI value {} at index {} is not finite",
                            test_name,
                            val,
                            i
                        );

                        
                        
                        prop_assert!(
							val > -100.0 && val < 1000.0,
							"[{}] CVI value {} at index {} outside reasonable bounds [-100%, 1000%]",
							test_name, val, i
						);
                    }
                }

                
                prop_assert_eq!(
                    out.values.len(),
                    high_data.len(),
                    "[{}] Output length mismatch",
                    test_name
                );

                
                prop_assert_eq!(
                    out.values.len(),
                    ref_out.values.len(),
                    "[{}] Output length mismatch between kernels",
                    test_name
                );

                for (i, (&y, &r)) in out.values.iter().zip(ref_out.values.iter()).enumerate() {
                    if !y.is_finite() || !r.is_finite() {
                        prop_assert_eq!(
                            y.to_bits(),
                            r.to_bits(),
                            "[{}] NaN/Inf mismatch at index {}: {} vs {}",
                            test_name,
                            i,
                            y,
                            r
                        );
                        continue;
                    }

                    
                    let ulp_diff = y.to_bits().abs_diff(r.to_bits());
                    prop_assert!(
                        (y - r).abs() <= 1e-9 || ulp_diff <= 4,
                        "[{}] Value mismatch at index {}: {} vs {} (ULP={})",
                        test_name,
                        i,
                        y,
                        r,
                        ulp_diff
                    );
                }

                
                if period == 2 {
                    
                    let warmup_count = out.values.iter().take_while(|&&v| v.is_nan()).count();
                    prop_assert_eq!(
                        warmup_count,
                        3,
                        "[{}] Period=2 should have warmup count of 3, got {}",
                        test_name,
                        warmup_count
                    );
                }

                
                
                let ranges: Vec<f64> = high_data
                    .iter()
                    .zip(low_data.iter())
                    .map(|(&h, &l)| h - l)
                    .collect();

                
                let valid_cvi: Vec<f64> = out
                    .values
                    .iter()
                    .filter(|&&v| !v.is_nan())
                    .cloned()
                    .collect();

                if valid_cvi.len() > 10 {
                    
                    let min_cvi = valid_cvi.iter().cloned().fold(f64::INFINITY, f64::min);
                    let max_cvi = valid_cvi.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

                    
                    let range_variance = ranges
                        .windows(period)
                        .map(|w| {
                            let mean = w.iter().sum::<f64>() / w.len() as f64;
                            w.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / w.len() as f64
                        })
                        .fold(0.0f64, f64::max);

                    if range_variance > 1e-10 {
                        
                        prop_assert!(
							(max_cvi - min_cvi).abs() > 1.0,
							"[{}] CVI should show meaningful variation (>1%) when volatility changes, got range: {}",
							test_name, (max_cvi - min_cvi).abs()
						);
                    }
                }

                
                let min_range = ranges.iter().cloned().fold(f64::INFINITY, f64::min);
                if min_range < 1e-10 {
                    for &val in &out.values {
                        if !val.is_nan() {
                            prop_assert!(
                                val.is_finite(),
                                "[{}] CVI should remain finite even with tiny ranges",
                                test_name
                            );
                        }
                    }
                }

                
                
                if ranges.len() > period * 3 {
                    let mean_range = ranges.iter().sum::<f64>() / ranges.len() as f64;
                    let all_similar = ranges
                        .iter()
                        .all(|&r| (r - mean_range).abs() < mean_range * 0.01); 

                    if all_similar && mean_range > 1e-10 {
                        
                        let last_quarter_start = valid_cvi.len() * 3 / 4;
                        if last_quarter_start < valid_cvi.len() {
                            let last_quarter: Vec<f64> = valid_cvi[last_quarter_start..].to_vec();
                            if !last_quarter.is_empty() {
                                let avg_abs_cvi = last_quarter.iter().map(|v| v.abs()).sum::<f64>()
                                    / last_quarter.len() as f64;

                                prop_assert!(
									avg_abs_cvi < 5.0,
									"[{}] CVI should converge near 0 for constant volatility, but average |CVI| = {}",
									test_name, avg_abs_cvi
								);
                            }
                        }
                    }
                }

                
                
                if ranges.len() > period * 2 {
                    
                    let mut max_volatility_change = 1.0;
                    let mut max_change_idx = 0;

                    for i in period..ranges.len() - period {
                        let prev_avg = ranges[i - period..i].iter().sum::<f64>() / period as f64;
                        let next_avg = ranges[i..i + period].iter().sum::<f64>() / period as f64;

                        if prev_avg > 1e-10 && next_avg > 1e-10 {
                            let ratio = (next_avg / prev_avg).max(prev_avg / next_avg);
                            if ratio > max_volatility_change {
                                max_volatility_change = ratio;
                                max_change_idx = i;
                            }
                        }
                    }

                    
                    if max_volatility_change > 5.0 {
                        
                        let spike_start =
                            (max_change_idx + expected_first_valid).saturating_sub(period);
                        let spike_end = (max_change_idx + expected_first_valid + period * 2)
                            .min(out.values.len());

                        if spike_end > spike_start {
                            let spike_region: Vec<f64> = out.values[spike_start..spike_end]
                                .iter()
                                .filter(|&&v| !v.is_nan())
                                .cloned()
                                .collect();

                            if !spike_region.is_empty() {
                                let max_abs_cvi =
                                    spike_region.iter().map(|v| v.abs()).fold(0.0f64, f64::max);

                                prop_assert!(
									max_abs_cvi > 10.0,
									"[{}] CVI should spike (>10%) for {}x volatility change, but max |CVI| = {}",
									test_name, max_volatility_change, max_abs_cvi
								);
                            }
                        }
                    }
                }

                Ok(())
            })
            .unwrap();

        Ok(())
    }
}

/// Helper function to find the first valid index where both high and low are not NaN
#[inline]
fn find_first_valid_idx(high: &[f64], low: &[f64]) -> Option<usize> {
    (0..high.len()).find(|&i| !high[i].is_nan() && !low[i].is_nan())
}

/// Core helper function that writes directly to an output slice.
/// This is used by WASM bindings to enable zero-copy operations.
#[inline(always)]
pub fn cvi_into_slice(
    output: &mut [f64],
    input: &CviInput,
    kernel: Kernel,
) -> Result<(), CviError> {
    let (high, low) = match &input.data {
        CviData::Candles(c) => (&c.high[..], &c.low[..]),
        CviData::Slices { high, low } => (*high, *low),
    };

    let period = input.params.period.unwrap_or(10);

    
    if high.is_empty() || low.is_empty() {
        return Err(CviError::EmptyData);
    }
    if period == 0 || period > high.len() {
        return Err(CviError::InvalidPeriod {
            period,
            data_len: high.len(),
        });
    }
    if high.len() != low.len() {
        return Err(CviError::EmptyData);
    }
    if output.len() != high.len() {
        return Err(CviError::OutputLengthMismatch { expected: high.len(), got: output.len() });
    }

    
    let first_valid_idx = match find_first_valid_idx(high, low) {
        Some(idx) => idx,
        None => return Err(CviError::AllValuesNaN),
    };

    let warmup = period - 1;
    let min_data_points = warmup + period;

    
    if high.len() - first_valid_idx < min_data_points {
        return Err(CviError::NotEnoughValidData {
            needed: min_data_points,
            valid: high.len() - first_valid_idx,
        });
    }

    
    let out_start = first_valid_idx + 2 * period - 1;
    for i in 0..out_start {
        output[i] = f64::NAN;
    }

    
    match kernel {
        Kernel::Scalar => cvi_scalar(high, low, period, first_valid_idx, output),
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2 => unsafe { cvi_avx2(high, low, period, first_valid_idx, output) },
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512 => unsafe { cvi_avx512(high, low, period, first_valid_idx, output) },
        Kernel::Auto => cvi_scalar(high, low, period, first_valid_idx, output),
        _ => return Err(CviError::InvalidKernelForBatch(kernel)),
    }

    Ok(())
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn cvi_js(high: &[f64], low: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
    let params = CviParams {
        period: Some(period),
    };
    let input = CviInput::from_slices(high, low, params);

    
    let output = cvi(&input).map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output.values)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn cvi_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn cvi_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn cvi_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
) -> Result<(), JsValue> {
    if high_ptr.is_null() || low_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to cvi_into"));
    }

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);

        if period == 0 || period > len {
            return Err(JsValue::from_str("Invalid period"));
        }

        let params = CviParams {
            period: Some(period),
        };
        let input = CviInput::from_slices(high, low, params);

        
        let aliased = high_ptr == out_ptr || low_ptr == out_ptr;

        if aliased {
            
            let result = cvi(&input).map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&result.values);
        } else {
            
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            cvi_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct CviBatchConfig {
    pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct CviBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<CviParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = cvi_batch)]
pub fn cvi_batch_unified_js(
    high: &[f64],
    low: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: CviBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = CviBatchRange {
        period: config.period_range,
    };

    let kernel = detect_best_kernel();
    let output = cvi_batch_inner(high, low, &sweep, kernel, false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = CviBatchJsOutput {
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
pub fn cvi_batch_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<usize, JsValue> {
    if high_ptr.is_null() || low_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to cvi_batch_into"));
    }

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);

        let sweep = CviBatchRange {
            period: (period_start, period_end, period_step),
        };

        let combos = expand_grid(&sweep);
        let rows = combos.len();
        let cols = len;

        let total = rows
            .checked_mul(cols)
            .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;
        let out = std::slice::from_raw_parts_mut(out_ptr, total);

        cvi_batch_inner_into(high, low, &sweep, Kernel::Auto, false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(rows)
    }
}
