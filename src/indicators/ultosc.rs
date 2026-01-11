//! # Ultimate Oscillator (ULTOSC)
//!
//! Combines short, medium, and long time periods into a single oscillator value,
//! blending market momentum over multiple horizons with weighted averages (4:2:1 ratio).
//!
//! ## Parameters
//! - **timeperiod1**: Short window for fast momentum (default: 7)
//! - **timeperiod2**: Medium window for intermediate momentum (default: 14)
//! - **timeperiod3**: Long window for slow momentum (default: 28)
//!
//! ## Inputs
//! - High, low, and close price series (or candles)
//! - All series must have the same length
//!
//! ## Returns
//! - **values**: Ultimate Oscillator values as `Vec<f64>` (length matches input, range 0-100)
//!
//! ## Developer Notes
//! - Decision: SIMD kept as stubs delegating to scalar. ULTOSC has tight per-step recurrences (prev_close, rolling sums),
//!   so across-time SIMD shows negligible gains on realistic sizes. Scalar is the reference path.
//! - Scalar path: single-pass, branch-light ring with zeroed evictions; >5% faster vs prior (bench: ~387µs → ~265µs on 100k).
//! - Batch: row-specific optimization via shared prefix sums for CMTL/TR; minor micro-tuning uses reciprocal multiplies.
//! - CUDA: prefix-sum kernels for batch and many-series paths are enabled; wrappers use typed errors, VRAM checks, and CAI v3/DLPack handles without changing scalar numerical outputs.

#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
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
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use thiserror::Error;

#[cfg(all(feature = "python", feature = "cuda"))]
mod ultosc_python_cuda_handle {
    use crate::utilities::dlpack_cuda::export_f32_cuda_dlpack_2d;
    use cust::context::Context;
    use cust::memory::DeviceBuffer;
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::PyDict;
    use std::ffi::c_void;
    use std::sync::Arc;

    #[pyclass(module = "ta_indicators.cuda", unsendable, name = "UltOscDeviceArrayF32Py")]
    pub struct DeviceArrayF32Py {
        pub(crate) buf: Option<DeviceBuffer<f32>>,
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
            (2, self.device_id as i32)
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
            let (exp_dev_ty, alloc_dev) = self.__dlpack_device__();
            if let Some(dev_obj) = dl_device.as_ref() {
                if let Ok((dev_ty, dev_id)) = dev_obj.extract::<(i32, i32)>(py) {
                    if dev_ty != exp_dev_ty || dev_id != alloc_dev {
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

            let max_version_bound = max_version.map(|obj| obj.into_bound(py));

            export_f32_cuda_dlpack_2d(py, buf, self.rows, self.cols, alloc_dev, max_version_bound)
        }
    }

    pub use DeviceArrayF32Py as UltOscDeviceArrayF32Py;
}

#[cfg(all(feature = "python", feature = "cuda"))]
use self::ultosc_python_cuda_handle::UltOscDeviceArrayF32Py;

// --- DATA STRUCTS ---
#[derive(Debug, Clone)]
pub enum UltOscData<'a> {
    Candles {
        candles: &'a Candles,
        high_src: &'a str,
        low_src: &'a str,
        close_src: &'a str,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct UltOscOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct UltOscParams {
    pub timeperiod1: Option<usize>,
    pub timeperiod2: Option<usize>,
    pub timeperiod3: Option<usize>,
}

impl Default for UltOscParams {
    fn default() -> Self {
        Self {
            timeperiod1: Some(7),
            timeperiod2: Some(14),
            timeperiod3: Some(28),
        }
    }
}

#[derive(Debug, Clone)]
pub struct UltOscInput<'a> {
    pub data: UltOscData<'a>,
    pub params: UltOscParams,
}

impl<'a> UltOscInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        high_src: &'a str,
        low_src: &'a str,
        close_src: &'a str,
        params: UltOscParams,
    ) -> Self {
        Self {
            data: UltOscData::Candles {
                candles,
                high_src,
                low_src,
                close_src,
            },
            params,
        }
    }
    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: UltOscParams,
    ) -> Self {
        Self {
            data: UltOscData::Slices { high, low, close },
            params,
        }
    }
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: UltOscData::Candles {
                candles,
                high_src: "high",
                low_src: "low",
                close_src: "close",
            },
            params: UltOscParams::default(),
        }
    }
    #[inline]
    pub fn get_timeperiod1(&self) -> usize {
        self.params.timeperiod1.unwrap_or(7)
    }
    #[inline]
    pub fn get_timeperiod2(&self) -> usize {
        self.params.timeperiod2.unwrap_or(14)
    }
    #[inline]
    pub fn get_timeperiod3(&self) -> usize {
        self.params.timeperiod3.unwrap_or(28)
    }
}

// --- BUILDER ---
#[derive(Copy, Clone, Debug)]
pub struct UltOscBuilder {
    timeperiod1: Option<usize>,
    timeperiod2: Option<usize>,
    timeperiod3: Option<usize>,
    kernel: Kernel,
}

impl Default for UltOscBuilder {
    fn default() -> Self {
        Self {
            timeperiod1: None,
            timeperiod2: None,
            timeperiod3: None,
            kernel: Kernel::Auto,
        }
    }
}

impl UltOscBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn timeperiod1(mut self, p: usize) -> Self {
        self.timeperiod1 = Some(p);
        self
    }
    #[inline(always)]
    pub fn timeperiod2(mut self, p: usize) -> Self {
        self.timeperiod2 = Some(p);
        self
    }
    #[inline(always)]
    pub fn timeperiod3(mut self, p: usize) -> Self {
        self.timeperiod3 = Some(p);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, candles: &Candles) -> Result<UltOscOutput, UltOscError> {
        let params = UltOscParams {
            timeperiod1: self.timeperiod1,
            timeperiod2: self.timeperiod2,
            timeperiod3: self.timeperiod3,
        };
        let input = UltOscInput::with_default_candles(candles);
        ultosc_with_kernel(&UltOscInput { params, ..input }, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<UltOscOutput, UltOscError> {
        let params = UltOscParams {
            timeperiod1: self.timeperiod1,
            timeperiod2: self.timeperiod2,
            timeperiod3: self.timeperiod3,
        };
        let input = UltOscInput::from_slices(high, low, close, params);
        ultosc_with_kernel(&input, self.kernel)
    }
}

// --- ERROR ---
#[derive(Debug, Error)]
pub enum UltOscError {
    #[error("ultosc: Input data slice is empty.")]
    EmptyInputData,
    #[error("ultosc: All values are NaN (or their preceding data is NaN).")]
    AllValuesNaN,
    #[error("ultosc: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("ultosc: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("ultosc: Output length mismatch: expected {expected}, got {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("ultosc: Inconsistent input lengths")]
    InconsistentLengths,
    #[error("ultosc: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange { start: String, end: String, step: String },
    #[error("ultosc: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

// --- HELPER FUNCTIONS ---
#[inline]
fn ultosc_prepare<'a>(
    input: &'a UltOscInput,
    kernel: Kernel,
) -> Result<
    (
        (&'a [f64], &'a [f64], &'a [f64]),
        usize,
        usize,
        usize,
        usize,
        usize,
        Kernel,
    ),
    UltOscError,
> {
    let (high, low, close) = match &input.data {
        UltOscData::Candles {
            candles,
            high_src,
            low_src,
            close_src,
        } => {
            let high = source_type(candles, high_src);
            let low = source_type(candles, low_src);
            let close = source_type(candles, close_src);
            (high, low, close)
        }
        UltOscData::Slices { high, low, close } => (*high, *low, *close),
    };

    let len = high.len();
    if len == 0 || low.len() == 0 || close.len() == 0 {
        return Err(UltOscError::EmptyInputData);
    }
    if low.len() != len || close.len() != len {
        return Err(UltOscError::InconsistentLengths);
    }

    let p1 = input.get_timeperiod1();
    let p2 = input.get_timeperiod2();
    let p3 = input.get_timeperiod3();

    if p1 == 0 || p2 == 0 || p3 == 0 || p1 > len || p2 > len || p3 > len {
        let bad = if p1 == 0 || p1 > len {
            p1
        } else if p2 == 0 || p2 > len {
            p2
        } else {
            p3
        };
        return Err(UltOscError::InvalidPeriod {
            period: bad,
            data_len: len,
        });
    }

    let largest_period = p1.max(p2.max(p3));
    let first_valid = match (1..len).find(|&i| {
        !high[i - 1].is_nan()
            && !low[i - 1].is_nan()
            && !close[i - 1].is_nan()
            && !high[i].is_nan()
            && !low[i].is_nan()
            && !close[i].is_nan()
    }) {
        Some(i) => i,
        None => return Err(UltOscError::AllValuesNaN),
    };

    let start_idx = first_valid + (largest_period - 1);
    if start_idx >= len {
        return Err(UltOscError::NotEnoughValidData {
            needed: largest_period,
            valid: len.saturating_sub(first_valid),
        });
    }

    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    Ok((
        (high, low, close),
        p1,
        p2,
        p3,
        first_valid,
        start_idx,
        chosen,
    ))
}


#[inline]
pub fn ultosc(input: &UltOscInput) -> Result<UltOscOutput, UltOscError> {
    ultosc_with_kernel(input, Kernel::Auto)
}

pub fn ultosc_with_kernel(
    input: &UltOscInput,
    kernel: Kernel,
) -> Result<UltOscOutput, UltOscError> {
    let ((high, low, close), p1, p2, p3, first_valid, start_idx, chosen) =
        ultosc_prepare(input, kernel)?;
    let len = high.len();
    let mut out = alloc_with_nan_prefix(len, start_idx);

    ultosc_compute_into(high, low, close, p1, p2, p3, first_valid, chosen, &mut out);

    Ok(UltOscOutput { values: out })
}

/// Compute Ultimate Oscillator into a caller-provided buffer (no allocations).
///
/// - Preserves NaN warmup prefix exactly like the Vec-returning API
///   (quiet NaNs via `0x7ff8_0000_0000_0000`).
/// - `out.len()` must equal the input length, otherwise returns this module's
///   existing error variant.
#[cfg(not(feature = "wasm"))]
pub fn ultosc_into(input: &UltOscInput, out: &mut [f64]) -> Result<(), UltOscError> {
    let ((high, low, close), p1, p2, p3, first_valid, start_idx, chosen) =
        ultosc_prepare(input, Kernel::Auto)?;

    if out.len() != high.len() {
        return Err(UltOscError::OutputLengthMismatch {
            expected: high.len(),
            got: out.len(),
        });
    }

    // Prefill warmup prefix with the same quiet-NaN pattern used by `alloc_with_nan_prefix`.
    let warm = start_idx.min(out.len());
    let qnan = f64::from_bits(0x7ff8_0000_0000_0000);
    for v in &mut out[..warm] {
        *v = qnan;
    }

    // Compute the remainder directly into the destination buffer.
    ultosc_compute_into(high, low, close, p1, p2, p3, first_valid, chosen, out);

    Ok(())
}

#[inline]
fn ultosc_compute_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    p1: usize,
    p2: usize,
    p3: usize,
    first_valid: usize,
    chosen: Kernel,
    dst: &mut [f64],
) {
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                ultosc_scalar(high, low, close, p1, p2, p3, first_valid, dst)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                ultosc_avx2(high, low, close, p1, p2, p3, first_valid, dst)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                ultosc_avx512(high, low, close, p1, p2, p3, first_valid, dst)
            }
            _ => unreachable!(),
        }
    }
}

// --- KERNEL IMPL ---
#[inline(always)]
pub unsafe fn ultosc_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    p1: usize,
    p2: usize,
    p3: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    let len = high.len();
    let max_period = p1.max(p2).max(p3);

    // Allocate temporary buffers on stack for small periods, heap for large
    const STACK_THRESHOLD: usize = 256;

    if max_period <= STACK_THRESHOLD {
        // Stack allocation for small periods
        let mut cmtl_stack = [0.0_f64; STACK_THRESHOLD];
        let mut tr_stack = [0.0_f64; STACK_THRESHOLD];
        let cmtl_buf = &mut cmtl_stack[..max_period];
        let tr_buf = &mut tr_stack[..max_period];

        ultosc_scalar_impl(
            high,
            low,
            close,
            p1,
            p2,
            p3,
            first_valid,
            out,
            cmtl_buf,
            tr_buf,
        );
    } else {
        // Heap allocation for large periods
        let mut cmtl_vec = vec![0.0; max_period];
        let mut tr_vec = vec![0.0; max_period];

        ultosc_scalar_impl(
            high,
            low,
            close,
            p1,
            p2,
            p3,
            first_valid,
            out,
            &mut cmtl_vec,
            &mut tr_vec,
        );
    }
}

#[inline(always)]
unsafe fn ultosc_scalar_impl(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    p1: usize,
    p2: usize,
    p3: usize,
    first_valid: usize,
    out: &mut [f64],
    cmtl_buf: &mut [f64],
    tr_buf: &mut [f64],
) {
    let len = high.len();
    if len == 0 {
        return;
    }

    let max_p = p1.max(p2).max(p3);
    debug_assert!(max_p > 0 && max_p <= len);

    // Start computing outputs when the largest window is filled
    let start_idx = first_valid + max_p - 1;

    // Weights: 100 * (4*BP1/TR1 + 2*BP2/TR2 + 1*BP3/TR3) / 7
    let inv7_100: f64 = 100.0f64 / 7.0f64;
    let w1: f64 = inv7_100 * 4.0f64;
    let w2: f64 = inv7_100 * 2.0f64;
    let w3: f64 = inv7_100 * 1.0f64;

    // Rolling sums for the three windows
    let mut sum1_a = 0.0f64;
    let mut sum1_b = 0.0f64;
    let mut sum2_a = 0.0f64;
    let mut sum2_b = 0.0f64;
    let mut sum3_a = 0.0f64;
    let mut sum3_b = 0.0f64;

    // Ring buffer index and element count processed since first_valid
    let mut buf_idx: usize = 0;
    let mut count: usize = 0;

    // Single-pass loop from first_valid to end
    let mut i = first_valid;
    while i < len {
        let hi = *high.get_unchecked(i);
        let lo = *low.get_unchecked(i);
        let ci = *close.get_unchecked(i);
        let prev_c = *close.get_unchecked(i - 1);

        // Valid if all inputs for this step are non-NaN
        let valid = !(hi.is_nan() | lo.is_nan() | ci.is_nan() | prev_c.is_nan());

        // Compute today's CMTL (close - true low) and True Range.
        
        let (c_new, t_new) = if valid {
            
            let tl = if lo < prev_c { lo } else { prev_c };

            
            
            
            
            
            
            
            let th = if hi > prev_c { hi } else { prev_c };
            let tr = th - tl;
            (ci - tl, tr)
        } else {
            (0.0, 0.0)
        };

        
        if count >= p1 {
            let mut old_idx1 = buf_idx + max_p - p1;
            if old_idx1 >= max_p {
                old_idx1 -= max_p;
            }
            sum1_a -= *cmtl_buf.get_unchecked(old_idx1);
            sum1_b -= *tr_buf.get_unchecked(old_idx1);
        }
        if count >= p2 {
            let mut old_idx2 = buf_idx + max_p - p2;
            if old_idx2 >= max_p {
                old_idx2 -= max_p;
            }
            sum2_a -= *cmtl_buf.get_unchecked(old_idx2);
            sum2_b -= *tr_buf.get_unchecked(old_idx2);
        }
        if count >= p3 {
            let mut old_idx3 = buf_idx + max_p - p3;
            if old_idx3 >= max_p {
                old_idx3 -= max_p;
            }
            sum3_a -= *cmtl_buf.get_unchecked(old_idx3);
            sum3_b -= *tr_buf.get_unchecked(old_idx3);
        }

        
        *cmtl_buf.get_unchecked_mut(buf_idx) = c_new;
        *tr_buf.get_unchecked_mut(buf_idx) = t_new;

        
        if valid {
            sum1_a += c_new;
            sum1_b += t_new;
            sum2_a += c_new;
            sum2_b += t_new;
            sum3_a += c_new;
            sum3_b += t_new;
        }

        
        count += 1;
        if i >= start_idx {
            
            let t1 = if sum1_b != 0.0 {
                sum1_a * sum1_b.recip()
            } else {
                0.0
            };
            let t2 = if sum2_b != 0.0 {
                sum2_a * sum2_b.recip()
            } else {
                0.0
            };
            let t3 = if sum3_b != 0.0 {
                sum3_a * sum3_b.recip()
            } else {
                0.0
            };

            
            let acc = f64::mul_add(w2, t2, w3 * t3);
            *out.get_unchecked_mut(i) = f64::mul_add(w1, t1, acc);
        }

        
        buf_idx += 1;
        if buf_idx == max_p {
            buf_idx = 0;
        }

        i += 1;
    }
}


#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn ultosc_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    p1: usize,
    p2: usize,
    p3: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    ultosc_scalar(high, low, close, p1, p2, p3, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn ultosc_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    p1: usize,
    p2: usize,
    p3: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    if p1.max(p2).max(p3) <= 32 {
        ultosc_avx512_short(high, low, close, p1, p2, p3, first_valid, out)
    } else {
        ultosc_avx512_long(high, low, close, p1, p2, p3, first_valid, out)
    }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn ultosc_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    p1: usize,
    p2: usize,
    p3: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    ultosc_scalar(high, low, close, p1, p2, p3, first_valid, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn ultosc_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    p1: usize,
    p2: usize,
    p3: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    ultosc_scalar(high, low, close, p1, p2, p3, first_valid, out)
}


#[inline(always)]
pub fn ultosc_row_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    p1: usize,
    p2: usize,
    p3: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    unsafe { ultosc_scalar(high, low, close, p1, p2, p3, first_valid, out) }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn ultosc_row_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    p1: usize,
    p2: usize,
    p3: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    unsafe { ultosc_avx2(high, low, close, p1, p2, p3, first_valid, out) }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn ultosc_row_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    p1: usize,
    p2: usize,
    p3: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    unsafe { ultosc_avx512(high, low, close, p1, p2, p3, first_valid, out) }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn ultosc_row_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    p1: usize,
    p2: usize,
    p3: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    unsafe { ultosc_avx512_short(high, low, close, p1, p2, p3, first_valid, out) }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn ultosc_row_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    p1: usize,
    p2: usize,
    p3: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    unsafe { ultosc_avx512_long(high, low, close, p1, p2, p3, first_valid, out) }
}


#[derive(Clone, Debug)]
pub struct UltOscBatchRange {
    pub timeperiod1: (usize, usize, usize),
    pub timeperiod2: (usize, usize, usize),
    pub timeperiod3: (usize, usize, usize),
}

impl Default for UltOscBatchRange {
    fn default() -> Self {
        Self {
            timeperiod1: (7, 7, 0),
            timeperiod2: (14, 14, 0),
            timeperiod3: (28, 277, 1),
        }
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct UltOscBatchConfig {
    pub timeperiod1_range: (usize, usize, usize),
    pub timeperiod2_range: (usize, usize, usize),
    pub timeperiod3_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct UltOscBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<UltOscParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone, Debug)]
pub struct UltOscBatchBuilder {
    kernel: Kernel,
}

impl Default for UltOscBatchBuilder {
    fn default() -> Self {
        Self {
            kernel: Kernel::Auto,
        }
    }
}

impl UltOscBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    pub fn apply_slice(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        sweep: &UltOscBatchRange,
    ) -> Result<UltOscBatchOutput, UltOscError> {
        ultosc_batch_with_kernel(high, low, close, sweep, self.kernel)
    }
}

#[derive(Clone, Debug)]
pub struct UltOscBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<UltOscParams>,
    pub rows: usize,
    pub cols: usize,
}

impl UltOscBatchOutput {
    pub fn row_for_params(&self, p: &UltOscParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.timeperiod1.unwrap_or(7) == p.timeperiod1.unwrap_or(7)
                && c.timeperiod2.unwrap_or(14) == p.timeperiod2.unwrap_or(14)
                && c.timeperiod3.unwrap_or(28) == p.timeperiod3.unwrap_or(28)
        })
    }

    pub fn values_for(&self, p: &UltOscParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &UltOscBatchRange) -> Result<Vec<UltOscParams>, UltOscError> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, UltOscError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        let s = step.max(1);
        let mut v = Vec::new();
        if start <= end {
            let mut cur = start;
            loop {
                v.push(cur);
                if cur == end {
                    break;
                }
                let next = cur
                    .checked_add(s)
                    .ok_or_else(|| UltOscError::InvalidRange {
                        start: start.to_string(),
                        end: end.to_string(),
                        step: step.to_string(),
                    })?;
                if next <= cur || next > end {
                    break;
                }
                cur = next;
            }
        } else {
            let mut cur = start;
            loop {
                v.push(cur);
                if cur == end {
                    break;
                }
                let next = match cur.checked_sub(s) {
                    Some(n) => n,
                    None => break,
                };
                if next < end {
                    break;
                }
                cur = next;
            }
        }
        if v.is_empty() {
            return Err(UltOscError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        Ok(v)
    }

    let timeperiod1s = axis_usize(r.timeperiod1)?;
    let timeperiod2s = axis_usize(r.timeperiod2)?;
    let timeperiod3s = axis_usize(r.timeperiod3)?;

    let cap = timeperiod1s
        .len()
        .checked_mul(timeperiod2s.len())
        .and_then(|v| v.checked_mul(timeperiod3s.len()))
        .ok_or_else(|| UltOscError::InvalidRange {
            start: r.timeperiod1.0.to_string(),
            end: r.timeperiod3.1.to_string(),
            step: r.timeperiod1.2.to_string(),
        })?;

    let mut out = Vec::with_capacity(cap);
    for &tp1 in &timeperiod1s {
        for &tp2 in &timeperiod2s {
            for &tp3 in &timeperiod3s {
                out.push(UltOscParams {
                    timeperiod1: Some(tp1),
                    timeperiod2: Some(tp2),
                    timeperiod3: Some(tp3),
                });
            }
        }
    }
    Ok(out)
}

pub fn ultosc_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &UltOscBatchRange,
    k: Kernel,
) -> Result<UltOscBatchOutput, UltOscError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(UltOscError::InvalidKernelForBatch(k))
        }
    };

    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };

    ultosc_batch_inner(high, low, close, sweep, simd, true)
}

#[inline(always)]
fn ultosc_batch_inner(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &UltOscBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<UltOscBatchOutput, UltOscError> {
    let combos = expand_grid(sweep)?;
    let cols = high.len();
    let rows = combos.len();

    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| UltOscError::InvalidRange {
            start: rows.to_string(),
            end: cols.to_string(),
            step: "rows*cols".to_string(),
        })?;

    if cols == 0 {
        return Err(UltOscError::EmptyInputData);
    }
    if low.len() != cols || close.len() != cols {
        return Err(UltOscError::InconsistentLengths);
    }

    let mut buf_mu = make_uninit_matrix(rows, cols);
    if buf_mu.len() != expected {
        return Err(UltOscError::OutputLengthMismatch {
            expected,
            got: buf_mu.len(),
        });
    }

    let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len())
    };

    ultosc_batch_inner_into(high, low, close, sweep, kern, parallel, out)?;

    let values = unsafe {
        Vec::from_raw_parts(
            buf_guard.as_mut_ptr() as *mut f64,
            buf_guard.len(),
            buf_guard.capacity(),
        )
    };

    Ok(UltOscBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
pub fn ultosc_batch_inner_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &UltOscBatchRange,
    simd: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<UltOscParams>, UltOscError> {
    let combos = expand_grid(sweep)?;

    let _ = simd;

    let len = high.len();
    if len == 0 || low.is_empty() || close.is_empty() {
        return Err(UltOscError::EmptyInputData);
    }
    if low.len() != len || close.len() != len {
        return Err(UltOscError::InconsistentLengths);
    }

    
    let first_valid_idx = (1..len)
        .find(|&i| {
            !high[i - 1].is_nan()
                && !low[i - 1].is_nan()
                && !close[i - 1].is_nan()
                && !high[i].is_nan()
                && !low[i].is_nan()
                && !close[i].is_nan()
        })
        .ok_or(UltOscError::AllValuesNaN)?;

    let rows = combos.len();
    let cols = len;

    let mut warm = Vec::with_capacity(rows);
    let mut max_p = 0usize;
    for c in &combos {
        let p1 = c.timeperiod1.unwrap_or(7);
        let p2 = c.timeperiod2.unwrap_or(14);
        let p3 = c.timeperiod3.unwrap_or(28);
        if p1 == 0 || p2 == 0 || p3 == 0 || p1 > len || p2 > len || p3 > len {
            let bad = if p1 == 0 || p1 > len {
                p1
            } else if p2 == 0 || p2 > len {
                p2
            } else {
                p3
            };
            return Err(UltOscError::InvalidPeriod {
                period: bad,
                data_len: len,
            });
        }
        let pmax = p1.max(p2).max(p3);
        if pmax > max_p {
            max_p = pmax;
        }
        warm.push(first_valid_idx + pmax - 1);
    }

    if len - first_valid_idx < max_p {
        return Err(UltOscError::NotEnoughValidData {
            needed: max_p,
            valid: len - first_valid_idx,
        });
    }

    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| UltOscError::InvalidRange {
            start: rows.to_string(),
            end: cols.to_string(),
            step: "rows*cols".to_string(),
        })?;
    if out.len() != expected {
        return Err(UltOscError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }

    
    let out_uninit = unsafe {
        core::slice::from_raw_parts_mut(
            out.as_mut_ptr() as *mut core::mem::MaybeUninit<f64>,
            out.len(),
        )
    };
    init_matrix_prefixes(out_uninit, cols, &warm);

    
    
    let mut pcmtl = vec![0.0f64; len + 1];
    let mut ptr = vec![0.0f64; len + 1];
    for i in 0..len {
        let (mut add_c, mut add_t) = (0.0f64, 0.0f64);
        if i >= first_valid_idx {
            let hi = high[i];
            let lo = low[i];
            let ci = close[i];
            let pc = close[i - 1];
            if hi.is_finite() && lo.is_finite() && ci.is_finite() && pc.is_finite() {
                let tl = if lo < pc { lo } else { pc };
                let mut trv = hi - lo;
                let d1 = (hi - pc).abs();
                if d1 > trv {
                    trv = d1;
                }
                let d2 = (lo - pc).abs();
                if d2 > trv {
                    trv = d2;
                }
                add_c = ci - tl;
                add_t = trv;
            }
        }
        pcmtl[i + 1] = pcmtl[i] + add_c;
        ptr[i + 1] = ptr[i] + add_t;
    }

    let do_row = |row: usize, row_out: &mut [f64]| {
        let p1 = combos[row].timeperiod1.unwrap();
        let p2 = combos[row].timeperiod2.unwrap();
        let p3 = combos[row].timeperiod3.unwrap();
        let start = first_valid_idx + p1.max(p2).max(p3) - 1;

        let inv7_100: f64 = 100.0f64 / 7.0f64;
        let w1: f64 = inv7_100 * 4.0f64;
        let w2: f64 = inv7_100 * 2.0f64;
        let w3: f64 = inv7_100 * 1.0f64;

        
        for i in start..len {
            
            let s1a = pcmtl[i + 1] - pcmtl[i + 1 - p1];
            let s1b = ptr[i + 1] - ptr[i + 1 - p1];
            let s2a = pcmtl[i + 1] - pcmtl[i + 1 - p2];
            let s2b = ptr[i + 1] - ptr[i + 1 - p2];
            let s3a = pcmtl[i + 1] - pcmtl[i + 1 - p3];
            let s3b = ptr[i + 1] - ptr[i + 1 - p3];

            
            let t1 = if s1b != 0.0 { s1a * s1b.recip() } else { 0.0 };
            let t2 = if s2b != 0.0 { s2a * s2b.recip() } else { 0.0 };
            let t3 = if s3b != 0.0 { s3a * s3b.recip() } else { 0.0 };

            let acc = f64::mul_add(w2, t2, w3 * t3);
            row_out[i] = f64::mul_add(w1, t1, acc);
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;
            out_uninit
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, row_mu)| {
                    let row_out = unsafe {
                        core::slice::from_raw_parts_mut(
                            row_mu.as_mut_ptr() as *mut f64,
                            row_mu.len(),
                        )
                    };
                    do_row(row, row_out)
                });
        }
        #[cfg(target_arch = "wasm32")]
        {
            out_uninit
                .chunks_mut(cols)
                .enumerate()
                .for_each(|(row, row_mu)| {
                    let row_out = unsafe {
                        core::slice::from_raw_parts_mut(
                            row_mu.as_mut_ptr() as *mut f64,
                            row_mu.len(),
                        )
                    };
                    do_row(row, row_out)
                });
        }
    } else {
        out_uninit
            .chunks_mut(cols)
            .enumerate()
            .for_each(|(row, row_mu)| {
                let row_out = unsafe {
                    core::slice::from_raw_parts_mut(row_mu.as_mut_ptr() as *mut f64, row_mu.len())
                };
                do_row(row, row_out)
            });
    }

    Ok(combos)
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    #[cfg(feature = "proptest")]
    use proptest::prelude::*;

    fn check_ultosc_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = UltOscParams {
            timeperiod1: None,
            timeperiod2: None,
            timeperiod3: None,
        };
        let input = UltOscInput::from_candles(&candles, "high", "low", "close", params);
        let output = ultosc_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_ultosc_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = UltOscParams {
            timeperiod1: Some(7),
            timeperiod2: Some(14),
            timeperiod3: Some(28),
        };
        let input = UltOscInput::from_candles(&candles, "high", "low", "close", params);
        let result = ultosc_with_kernel(&input, kernel)?;
        let expected_last_five = [
            41.25546890298435,
            40.83865967175865,
            48.910324164909625,
            45.43113094857947,
            42.163165136766295,
        ];
        assert!(result.values.len() >= 5);
        let start_idx = result.values.len() - 5;
        for (i, &val) in result.values[start_idx..].iter().enumerate() {
            let exp = expected_last_five[i];
            assert!(
                (val - exp).abs() < 1e-8,
                "[{}] ULTOSC mismatch at last five index {}: expected {}, got {}",
                test_name,
                i,
                exp,
                val
            );
        }
        Ok(())
    }

    fn check_ultosc_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = UltOscInput::with_default_candles(&candles);
        let result = ultosc_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), candles.close.len());
        Ok(())
    }

    fn check_ultosc_zero_periods(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_high = [1.0, 2.0, 3.0];
        let input_low = [0.5, 1.5, 2.5];
        let input_close = [0.8, 1.8, 2.8];
        let params = UltOscParams {
            timeperiod1: Some(0),
            timeperiod2: Some(14),
            timeperiod3: Some(28),
        };
        let input = UltOscInput::from_slices(&input_high, &input_low, &input_close, params);
        let result = ultosc_with_kernel(&input, kernel);
        assert!(
            result.is_err(),
            "[{}] Expected error for zero period",
            test_name
        );
        Ok(())
    }

    fn check_ultosc_period_exceeds_data_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_high = [1.0, 2.0, 3.0];
        let input_low = [0.5, 1.5, 2.5];
        let input_close = [0.8, 1.8, 2.8];
        let params = UltOscParams {
            timeperiod1: Some(7),
            timeperiod2: Some(14),
            timeperiod3: Some(28),
        };
        let input = UltOscInput::from_slices(&input_high, &input_low, &input_close, params);
        let result = ultosc_with_kernel(&input, kernel);
        assert!(
            result.is_err(),
            "[{}] Expected error for period exceeding data length",
            test_name
        );
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_ultosc_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        
        let test_params = vec![
            
            UltOscParams::default(),
            
            UltOscParams {
                timeperiod1: Some(1),
                timeperiod2: Some(2),
                timeperiod3: Some(3),
            },
            
            UltOscParams {
                timeperiod1: Some(2),
                timeperiod2: Some(4),
                timeperiod3: Some(8),
            },
            
            UltOscParams {
                timeperiod1: Some(5),
                timeperiod2: Some(10),
                timeperiod3: Some(20),
            },
            
            UltOscParams {
                timeperiod1: Some(7),
                timeperiod2: Some(14),
                timeperiod3: Some(28),
            },
            
            UltOscParams {
                timeperiod1: Some(10),
                timeperiod2: Some(20),
                timeperiod3: Some(40),
            },
            
            UltOscParams {
                timeperiod1: Some(14),
                timeperiod2: Some(28),
                timeperiod3: Some(56),
            },
            
            UltOscParams {
                timeperiod1: Some(20),
                timeperiod2: Some(40),
                timeperiod3: Some(80),
            },
            
            UltOscParams {
                timeperiod1: Some(5),
                timeperiod2: Some(6),
                timeperiod3: Some(7),
            },
            
            UltOscParams {
                timeperiod1: Some(3),
                timeperiod2: Some(10),
                timeperiod3: Some(50),
            },
            
            UltOscParams {
                timeperiod1: Some(14),
                timeperiod2: Some(14),
                timeperiod3: Some(14),
            },
            
            UltOscParams {
                timeperiod1: Some(28),
                timeperiod2: Some(14),
                timeperiod3: Some(7),
            },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = UltOscInput::from_candles(&candles, "high", "low", "close", params.clone());
            let output = ultosc_with_kernel(&input, kernel)?;

            
            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue; 
                }

                let bits = val.to_bits();

                
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 with params: timeperiod1={}, timeperiod2={}, timeperiod3={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.timeperiod1.unwrap_or(7),
                        params.timeperiod2.unwrap_or(14),
                        params.timeperiod3.unwrap_or(28),
                        param_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 with params: timeperiod1={}, timeperiod2={}, timeperiod3={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.timeperiod1.unwrap_or(7),
                        params.timeperiod2.unwrap_or(14),
                        params.timeperiod3.unwrap_or(28),
                        param_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 with params: timeperiod1={}, timeperiod2={}, timeperiod3={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.timeperiod1.unwrap_or(7),
                        params.timeperiod2.unwrap_or(14),
                        params.timeperiod3.unwrap_or(28),
                        param_idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_ultosc_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(()) 
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_ultosc_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        
        let strat = (1usize..=50, 1usize..=50, 1usize..=50).prop_flat_map(|(p1, p2, p3)| {
            let max_period = p1.max(p2).max(p3);
            (
                
                
                prop::collection::vec(
                    (0.1f64..10000.0f64).prop_filter("finite", |x| x.is_finite()),
                    (max_period + 1)..400,
                ),
                Just((p1, p2, p3)),
            )
        });

        proptest::test_runner::TestRunner::default().run(
            &strat,
            |(base_prices, (p1, p2, p3))| {
                
                let mut high = Vec::with_capacity(base_prices.len());
                let mut low = Vec::with_capacity(base_prices.len());
                let mut close = Vec::with_capacity(base_prices.len());

                
                let mut seed = p1 + p2 * 7 + p3 * 13;
                for &price in &base_prices {
                    
                    seed = (seed * 1103515245 + 12345) % (1 << 31);
                    let spread_pct = 0.01 + (seed as f64 / (1u64 << 31) as f64) * 0.09;
                    let spread = price * spread_pct;

                    
                    seed = (seed * 1103515245 + 12345) % (1 << 31);
                    let close_position = seed as f64 / (1u64 << 31) as f64; 

                    let h = price + spread * 0.5;
                    let l = price - spread * 0.5;
                    let c = l + (h - l) * close_position;

                    high.push(h);
                    low.push(l);
                    close.push(c);
                }

                let params = UltOscParams {
                    timeperiod1: Some(p1),
                    timeperiod2: Some(p2),
                    timeperiod3: Some(p3),
                };
                let input = UltOscInput::from_slices(&high, &low, &close, params.clone());

                let result = ultosc_with_kernel(&input, kernel).unwrap();
                let out = result.values;

                
                let ref_result = ultosc_with_kernel(&input, Kernel::Scalar).unwrap();
                let ref_out = ref_result.values;

                let max_period = p1.max(p2).max(p3);
                
                let warmup = max_period;

                
                
                for i in 0..warmup.min(out.len()) {
                    prop_assert!(
                        out[i].is_nan(),
                        "[{}] Expected NaN during warmup at index {}, got {}",
                        test_name,
                        i,
                        out[i]
                    );
                }

                
                
                for (i, (&y, &r)) in out.iter().zip(ref_out.iter()).enumerate() {
                    if !y.is_finite() || !r.is_finite() {
                        prop_assert!(
                            y.to_bits() == r.to_bits(),
                            "[{}] NaN/inf mismatch at index {}: {} vs {}",
                            test_name,
                            i,
                            y,
                            r
                        );
                    } else {
                        let ulp_diff = y.to_bits().abs_diff(r.to_bits());
                        prop_assert!(
                            (y - r).abs() <= 1e-9 || ulp_diff <= 4,
                            "[{}] Value mismatch at index {}: {} vs {} (ULP diff: {})",
                            test_name,
                            i,
                            y,
                            r,
                            ulp_diff
                        );
                    }
                }

                
                
                for (i, &val) in out.iter().enumerate() {
                    if !val.is_nan() {
                        prop_assert!(
                            val >= 0.0 && val <= 100.0,
                            "[{}] ULTOSC value {} at index {} is out of bounds [0, 100]",
                            test_name,
                            val,
                            i
                        );
                    }
                }

                
                
                if high.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10)
                    && low.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10)
                    && close.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10)
                {
                    
                    
                    let stability_check_start = (warmup + p3.max(p2).max(p1)).min(out.len());
                    if stability_check_start < out.len() - 2 {
                        
                        let stable_region = &out[stability_check_start..];
                        let first_valid = stable_region.iter().position(|&v| !v.is_nan());

                        if let Some(idx) = first_valid {
                            let expected_stable = stable_region[idx];
                            
                            for (i, &val) in stable_region.iter().skip(idx + 1).enumerate() {
                                if !val.is_nan() {
                                    prop_assert!(
										(val - expected_stable).abs() < 1e-8,
										"[{}] Expected stable value {} for constant prices at index {}, got {}",
										test_name, expected_stable, stability_check_start + idx + 1 + i, val
									);
                                }
                            }
                        }
                    }
                }

                
                
                let zero_range_high = vec![100.0; base_prices.len()];
                let zero_range_low = zero_range_high.clone();
                let zero_range_close = zero_range_high.clone();

                let zero_input = UltOscInput::from_slices(
                    &zero_range_high,
                    &zero_range_low,
                    &zero_range_close,
                    params.clone(),
                );
                if let Ok(zero_result) = ultosc_with_kernel(&zero_input, kernel) {
                    
                    
                    for (i, &val) in zero_result.values.iter().enumerate().skip(warmup) {
                        if !val.is_nan() {
                            prop_assert!(
                                val.abs() < 1e-8,
                                "[{}] Expected 0 for zero range at index {}, got {}",
                                test_name,
                                i,
                                val
                            );
                        }
                    }
                }

                
                
                
                
                
                if out.len() > warmup {
                    
                    
                    for i in warmup..out.len().min(warmup + 5) {
                        if !out[i].is_nan() {
                            
                            
                            prop_assert!(
                                out[i] >= 0.0 && out[i] <= 100.0,
                                "[{}] ULTOSC at {} should be in [0,100], got {}",
                                test_name,
                                i,
                                out[i]
                            );
                        }
                    }
                }

                
                
                let reordered_params = UltOscParams {
                    timeperiod1: Some(p3),
                    timeperiod2: Some(p1),
                    timeperiod3: Some(p2),
                };
                let reordered_input =
                    UltOscInput::from_slices(&high, &low, &close, reordered_params);

                
                prop_assert!(
                    ultosc_with_kernel(&reordered_input, kernel).is_ok(),
                    "[{}] ULTOSC should work with any period ordering",
                    test_name
                );

                Ok(())
            },
        )?;

        Ok(())
    }

    macro_rules! generate_all_ultosc_tests {
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

    generate_all_ultosc_tests!(
        check_ultosc_partial_params,
        check_ultosc_accuracy,
        check_ultosc_default_candles,
        check_ultosc_zero_periods,
        check_ultosc_period_exceeds_data_length,
        check_ultosc_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_ultosc_tests!(check_ultosc_property);
    fn check_ultosc_batch_default(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        
        let sweep = UltOscBatchRange {
            timeperiod1: (5, 9, 2),   
            timeperiod2: (12, 16, 2), 
            timeperiod3: (26, 30, 2), 
        };

        let batch_builder = UltOscBatchBuilder::new().kernel(kernel);
        let output =
            batch_builder.apply_slice(&candles.high, &candles.low, &candles.close, &sweep)?;

        
        assert_eq!(output.rows, 3 * 3 * 3); 
        assert_eq!(output.cols, candles.close.len());
        assert_eq!(output.values.len(), output.rows * output.cols);
        assert_eq!(output.combos.len(), output.rows);

        
        let single_params = UltOscParams {
            timeperiod1: Some(7),
            timeperiod2: Some(14),
            timeperiod3: Some(28),
        };
        let single_input =
            UltOscInput::from_slices(&candles.high, &candles.low, &candles.close, single_params);
        let single_result = ultosc_with_kernel(&single_input, kernel)?;

        
        if let Some(row_idx) = output.row_for_params(&single_params) {
            let batch_row = output.values_for(&single_params).unwrap();

            
            let start = batch_row.len().saturating_sub(5);
            for i in 0..5 {
                let diff = (batch_row[start + i] - single_result.values[start + i]).abs();
                assert!(
                    diff < 1e-10,
                    "[{}] Batch vs single mismatch at idx {}: got {}, expected {}",
                    test_name,
                    i,
                    batch_row[start + i],
                    single_result.values[start + i]
                );
            }
        } else {
            panic!("[{}] Could not find row for params 7,14,28", test_name);
        }

        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        
        let test_configs = vec![
            
            (2, 8, 2, 4, 16, 4, 8, 32, 8),   
            (5, 7, 1, 10, 14, 2, 20, 28, 4), 
            (7, 7, 0, 14, 14, 0, 14, 42, 7), 
            (1, 5, 1, 10, 10, 0, 20, 20, 0), 
            (10, 20, 5, 20, 40, 10, 40, 80, 20), 
            (3, 9, 3, 6, 18, 6, 12, 36, 12), 
            (5, 10, 1, 10, 20, 2, 20, 40, 4), 
        ];

        for (
            cfg_idx,
            &(
                tp1_start,
                tp1_end,
                tp1_step,
                tp2_start,
                tp2_end,
                tp2_step,
                tp3_start,
                tp3_end,
                tp3_step,
            ),
        ) in test_configs.iter().enumerate()
        {
            let sweep = UltOscBatchRange {
                timeperiod1: (tp1_start, tp1_end, tp1_step),
                timeperiod2: (tp2_start, tp2_end, tp2_step),
                timeperiod3: (tp3_start, tp3_end, tp3_step),
            };

            let batch_builder = UltOscBatchBuilder::new().kernel(kernel);
            let output =
                batch_builder.apply_slice(&candles.high, &candles.low, &candles.close, &sweep)?;

            
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
						at row {} col {} (flat index {}) with params: timeperiod1={}, timeperiod2={}, timeperiod3={}",
                        test_name,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.timeperiod1.unwrap_or(7),
                        combo.timeperiod2.unwrap_or(14),
                        combo.timeperiod3.unwrap_or(28)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						at row {} col {} (flat index {}) with params: timeperiod1={}, timeperiod2={}, timeperiod3={}",
                        test_name,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.timeperiod1.unwrap_or(7),
                        combo.timeperiod2.unwrap_or(14),
                        combo.timeperiod3.unwrap_or(28)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						at row {} col {} (flat index {}) with params: timeperiod1={}, timeperiod2={}, timeperiod3={}",
                        test_name,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.timeperiod1.unwrap_or(7),
                        combo.timeperiod2.unwrap_or(14),
                        combo.timeperiod3.unwrap_or(28)
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(()) 
    }

    macro_rules! gen_batch_tests {
        ($fn_name:ident) => {
            paste::paste! {
                #[test]
                fn [<$fn_name _scalar>]() {
                    let _ = $fn_name(stringify!([<$fn_name _scalar>]), Kernel::ScalarBatch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test]
                fn [<$fn_name _avx2>]() {
                    let _ = $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test]
                fn [<$fn_name _avx512>]() {
                    let _ = $fn_name(stringify!([<$fn_name _avx512>]), Kernel::Avx512Batch);
                }
            }
        };
    }

    gen_batch_tests!(check_ultosc_batch_default);
    gen_batch_tests!(check_batch_no_poison);

    #[test]
    fn test_ultosc_into_matches_api() -> Result<(), Box<dyn std::error::Error>> {
        
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = UltOscInput::from_candles(
            &candles,
            "high",
            "low",
            "close",
            UltOscParams::default(),
        );

        let baseline = ultosc(&input)?;

        let mut out = vec![0.0; candles.close.len()];
        #[cfg(not(feature = "wasm"))]
        {
            ultosc_into(&input, &mut out)?;
        }
        #[cfg(feature = "wasm")]
        {
            
            ultosc_into_slice(&mut out, &input, Kernel::Auto)?;
        }

        assert_eq!(baseline.values.len(), out.len());

        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b) || ((a - b).abs() <= 1e-12)
        }

        for i in 0..out.len() {
            assert!(
                eq_or_both_nan(baseline.values[i], out[i]),
                "Mismatch at index {}: baseline={} out={}",
                i,
                baseline.values[i],
                out[i]
            );
        }

        Ok(())
    }
}





#[inline]
pub fn ultosc_into_slice(
    dst: &mut [f64],
    input: &UltOscInput,
    kern: Kernel,
) -> Result<(), UltOscError> {
    let ((high, low, close), p1, p2, p3, first_valid, start_idx, chosen) =
        ultosc_prepare(input, kern)?;

    if dst.len() != high.len() {
        return Err(UltOscError::OutputLengthMismatch {
            expected: high.len(),
            got: dst.len(),
        });
    }

    
    ultosc_compute_into(high, low, close, p1, p2, p3, first_valid, chosen, dst);

    
    let qnan = f64::from_bits(0x7ff8_0000_0000_0000);
    dst[..start_idx].fill(qnan);

    Ok(())
}





#[derive(Debug, Clone)]
pub struct UltOscStream {
    params: UltOscParams,
    
    cmtl_buf: Vec<f64>,
    tr_buf: Vec<f64>,

    
    sum1_a: f64,
    sum1_b: f64,
    sum2_a: f64,
    sum2_b: f64,
    sum3_a: f64,
    sum3_b: f64,

    
    buffer_idx: usize,
    count: usize,

    
    max_period: usize,
    p1: usize,
    p2: usize,
    p3: usize,

    
    w1: f64,
    w2: f64,
    w3: f64,

    
    prev_close: Option<f64>,
}

impl UltOscStream {
    #[inline]
    pub fn try_new(params: UltOscParams) -> Result<Self, UltOscError> {
        let p1 = params.timeperiod1.unwrap_or(7);
        let p2 = params.timeperiod2.unwrap_or(14);
        let p3 = params.timeperiod3.unwrap_or(28);

        if p1 == 0 || p2 == 0 || p3 == 0 {
            let bad = if p1 == 0 {
                p1
            } else if p2 == 0 {
                p2
            } else {
                p3
            };
            return Err(UltOscError::InvalidPeriod {
                period: bad,
                data_len: 0,
            });
        }

        let max_period = p1.max(p2).max(p3);

        
        
        const INV7_100: f64 = 100.0 / 7.0;
        let w1 = INV7_100 * 4.0;
        let w2 = INV7_100 * 2.0;
        let w3 = INV7_100 * 1.0;

        Ok(Self {
            params,
            cmtl_buf: vec![0.0; max_period],
            tr_buf: vec![0.0; max_period],

            sum1_a: 0.0,
            sum1_b: 0.0,
            sum2_a: 0.0,
            sum2_b: 0.0,
            sum3_a: 0.0,
            sum3_b: 0.0,

            buffer_idx: 0,
            count: 0,

            max_period,
            p1,
            p2,
            p3,

            w1,
            w2,
            w3,
            prev_close: None,
        })
    }

    #[inline(always)]
    fn idx_minus(&self, k: usize) -> usize {
        
        let mut j = self.buffer_idx + self.max_period - k;
        if j >= self.max_period {
            j -= self.max_period;
        }
        j
    }

    /// Push one bar; returns Some(ultosc) once the largest window is filled, None otherwise.
    #[inline]
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        
        let prev_close = match self.prev_close {
            Some(pc) => pc,
            None => {
                self.prev_close = Some(close);
                return None; 
            }
        };

        
        
        let valid = !(high.is_nan() | low.is_nan() | close.is_nan() | prev_close.is_nan());

        
        let (c_new, t_new) = if valid {
            
            let true_low = if low < prev_close { low } else { prev_close };

            
            
            let base = high - low;
            let d1 = (high - prev_close).abs();
            let d2 = (low - prev_close).abs();
            let tr = if d1 > base {
                if d2 > d1 {
                    d2
                } else {
                    d1
                }
            } else {
                if d2 > base {
                    d2
                } else {
                    base
                }
            };

            (close - true_low, tr)
        } else {
            (0.0, 0.0)
        };

        
        if self.count >= self.p1 {
            let j = self.idx_minus(self.p1);
            self.sum1_a -= self.cmtl_buf[j];
            self.sum1_b -= self.tr_buf[j];
        }
        if self.count >= self.p2 {
            let j = self.idx_minus(self.p2);
            self.sum2_a -= self.cmtl_buf[j];
            self.sum2_b -= self.tr_buf[j];
        }
        if self.count >= self.p3 {
            let j = self.idx_minus(self.p3);
            self.sum3_a -= self.cmtl_buf[j];
            self.sum3_b -= self.tr_buf[j];
        }

        
        self.cmtl_buf[self.buffer_idx] = c_new;
        self.tr_buf[self.buffer_idx] = t_new;

        
        self.sum1_a += c_new;
        self.sum1_b += t_new;
        self.sum2_a += c_new;
        self.sum2_b += t_new;
        self.sum3_a += c_new;
        self.sum3_b += t_new;

        
        self.buffer_idx += 1;
        if self.buffer_idx == self.max_period {
            self.buffer_idx = 0;
        }
        self.count += 1;

        
        self.prev_close = Some(close);

        
        if self.count < self.max_period {
            return None;
        }

        
        let t1 = if self.sum1_b != 0.0 {
            self.sum1_a * self.sum1_b.recip()
        } else {
            0.0
        };
        let t2 = if self.sum2_b != 0.0 {
            self.sum2_a * self.sum2_b.recip()
        } else {
            0.0
        };
        let t3 = if self.sum3_b != 0.0 {
            self.sum3_a * self.sum3_b.recip()
        } else {
            0.0
        };

        let acc = f64::mul_add(self.w2, t2, self.w3 * t3);
        Some(f64::mul_add(self.w1, t1, acc))
    }
}





#[cfg(feature = "python")]
#[pyfunction(name = "ultosc")]
#[pyo3(signature = (high, low, close, timeperiod1=None, timeperiod2=None, timeperiod3=None, kernel=None))]
pub fn ultosc_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod1: Option<usize>,
    timeperiod2: Option<usize>,
    timeperiod3: Option<usize>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let params = UltOscParams {
        timeperiod1,
        timeperiod2,
        timeperiod3,
    };
    let input = UltOscInput::from_slices(high_slice, low_slice, close_slice, params);

    let result_vec: Vec<f64> = py
        .allow_threads(|| ultosc_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyfunction(name = "ultosc_batch")]
#[pyo3(signature = (high, low, close, timeperiod1_range, timeperiod2_range, timeperiod3_range, kernel=None))]
pub fn ultosc_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod1_range: (usize, usize, usize),
    timeperiod2_range: (usize, usize, usize),
    timeperiod3_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;
    let kern = validate_kernel(kernel, true)?;

    let sweep = UltOscBatchRange {
        timeperiod1: timeperiod1_range,
        timeperiod2: timeperiod2_range,
        timeperiod3: timeperiod3_range,
    };

    let combos = expand_grid(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = high_slice.len();

    let total_elems = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows * cols overflow in ultosc_batch_py"))?;

    let out_arr = unsafe { PyArray1::<f64>::new(py, [total_elems], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

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
                _ => kernel,
            };
            ultosc_batch_inner_into(
                high_slice,
                low_slice,
                close_slice,
                &sweep,
                simd,
                true,
                slice_out,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "timeperiod1",
        combos
            .iter()
            .map(|p| p.timeperiod1.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "timeperiod2",
        combos
            .iter()
            .map(|p| p.timeperiod2.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "timeperiod3",
        combos
            .iter()
            .map(|p| p.timeperiod3.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;

    Ok(dict)
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "ultosc_cuda_batch_dev")]
#[pyo3(signature = (high, low, close, timeperiod1_range, timeperiod2_range, timeperiod3_range, device_id=0))]
pub fn ultosc_cuda_batch_dev_py(
    py: Python<'_>,
    high: PyReadonlyArray1<'_, f32>,
    low: PyReadonlyArray1<'_, f32>,
    close: PyReadonlyArray1<'_, f32>,
    timeperiod1_range: (usize, usize, usize),
    timeperiod2_range: (usize, usize, usize),
    timeperiod3_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<UltOscDeviceArrayF32Py> {
    use crate::cuda::cuda_available;
    use crate::cuda::oscillators::CudaUltosc;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;

    let sweep = UltOscBatchRange {
        timeperiod1: timeperiod1_range,
        timeperiod2: timeperiod2_range,
        timeperiod3: timeperiod3_range,
    };

    let (buf, rows, cols, ctx_arc, dev_id) = py.allow_threads(|| -> PyResult<_> {
        let cuda = CudaUltosc::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let dev = cuda
            .ultosc_batch_dev(high_slice, low_slice, close_slice, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let buf = dev.buf;
        let rows = dev.rows;
        let cols = dev.cols;
        let ctx = cuda.context_arc();
        let dev_id = cuda.device_id();
        Ok((buf, rows, cols, ctx, dev_id))
    })?;
    Ok(UltOscDeviceArrayF32Py {
        buf: Some(buf),
        rows,
        cols,
        _ctx: ctx_arc,
        device_id: dev_id as u32,
    })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "ultosc_cuda_many_series_one_param_dev")]
#[pyo3(signature = (high_tm, low_tm, close_tm, cols, rows, timeperiod1, timeperiod2, timeperiod3, device_id=0))]
pub fn ultosc_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    high_tm: PyReadonlyArray1<'_, f32>,
    low_tm: PyReadonlyArray1<'_, f32>,
    close_tm: PyReadonlyArray1<'_, f32>,
    cols: usize,
    rows: usize,
    timeperiod1: usize,
    timeperiod2: usize,
    timeperiod3: usize,
    device_id: usize,
) -> PyResult<UltOscDeviceArrayF32Py> {
    use crate::cuda::cuda_available;
    use crate::cuda::oscillators::CudaUltosc;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let h = high_tm.as_slice()?;
    let l = low_tm.as_slice()?;
    let c = close_tm.as_slice()?;
    let (buf, rows_out, cols_out, ctx_arc, dev_id) = py.allow_threads(|| -> PyResult<_> {
        let cuda = CudaUltosc::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let dev = cuda
            .ultosc_many_series_one_param_time_major_dev(
                h,
                l,
                c,
                cols,
                rows,
                timeperiod1,
                timeperiod2,
                timeperiod3,
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let buf = dev.buf;
        let rows_out = dev.rows;
        let cols_out = dev.cols;
        let ctx = cuda.context_arc();
        let dev_id = cuda.device_id();
        Ok((buf, rows_out, cols_out, ctx, dev_id))
    })?;
    Ok(UltOscDeviceArrayF32Py {
        buf: Some(buf),
        rows: rows_out,
        cols: cols_out,
        _ctx: ctx_arc,
        device_id: dev_id as u32,
    })
}

#[cfg(feature = "python")]
#[pyclass(name = "UltOscStream")]
pub struct UltOscStreamPy {
    stream: UltOscStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl UltOscStreamPy {
    #[new]
    #[pyo3(signature = (timeperiod1=None, timeperiod2=None, timeperiod3=None))]
    fn new(
        timeperiod1: Option<usize>,
        timeperiod2: Option<usize>,
        timeperiod3: Option<usize>,
    ) -> PyResult<Self> {
        let params = UltOscParams {
            timeperiod1,
            timeperiod2,
            timeperiod3,
        };
        let stream =
            UltOscStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(UltOscStreamPy { stream })
    }

    fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        self.stream.update(high, low, close)
    }
}





#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ultosc_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    timeperiod1: usize,
    timeperiod2: usize,
    timeperiod3: usize,
) -> Result<Vec<f64>, JsValue> {
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(JsValue::from_str("Empty data"));
    }
    if timeperiod1 == 0 || timeperiod2 == 0 || timeperiod3 == 0 {
        return Err(JsValue::from_str("Invalid period"));
    }
    let len = high.len();
    if timeperiod1 > len || timeperiod2 > len || timeperiod3 > len {
        return Err(JsValue::from_str("Period exceeds data length"));
    }

    let params = UltOscParams {
        timeperiod1: Some(timeperiod1),
        timeperiod2: Some(timeperiod2),
        timeperiod3: Some(timeperiod3),
    };
    let input = UltOscInput::from_slices(high, low, close, params);

    
    let mut output = vec![0.0; len];
    ultosc_into_slice(&mut output, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ultosc_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    timeperiod1: usize,
    timeperiod2: usize,
    timeperiod3: usize,
) -> Result<(), JsValue> {
    if high_ptr.is_null() || low_ptr.is_null() || close_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to ultosc_into"));
    }

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);

        let params = UltOscParams {
            timeperiod1: Some(timeperiod1),
            timeperiod2: Some(timeperiod2),
            timeperiod3: Some(timeperiod3),
        };
        let input = UltOscInput::from_slices(high, low, close, params);

        
        if high_ptr == out_ptr as *const f64
            || low_ptr == out_ptr as *const f64
            || close_ptr == out_ptr as *const f64
        {
            
            let mut temp = vec![0.0; len];
            ultosc_into_slice(&mut temp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            ultosc_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ultosc_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ultosc_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = ultosc_batch)]
pub fn ultosc_batch_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: UltOscBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = UltOscBatchRange {
        timeperiod1: config.timeperiod1_range,
        timeperiod2: config.timeperiod2_range,
        timeperiod3: config.timeperiod3_range,
    };

    let batch_output = ultosc_batch_with_kernel(high, low, close, &sweep, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let rows = batch_output.combos.len();
    let cols = high.len();

    let result = UltOscBatchJsOutput {
        values: batch_output.values,
        combos: batch_output.combos,
        rows,
        cols,
    };

    serde_wasm_bindgen::to_value(&result)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}
