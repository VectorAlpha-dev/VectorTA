//! # Median Price (MEDPRICE)
//!
//! Calculates the median price as `(high + low) / 2.0` for each period.
//!
//! ## Parameters
//! - **high**: High price data
//! - **low**: Low price data
//!
//! ## Returns
//! - `Vec<f64>` - Median price values matching input length
//!
//! ## Developer Status
//! **AVX2**: Stub (calls scalar)
//! **AVX512**: Stub (calls scalar)
//! **Streaming**: O(1) - Simple calculation
//! **Memory**: Good - Uses `alloc_with_nan_prefix` and `make_uninit_matrix`

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1};
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
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::mem::MaybeUninit;
use thiserror::Error;

#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::{cuda_available, CudaMedprice};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::alma::DeviceArrayF32Py;

/// Source data for medprice indicator.
#[derive(Debug, Clone)]
pub enum MedpriceData<'a> {
    Candles {
        candles: &'a Candles,
        high_source: &'a str,
        low_source: &'a str,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct MedpriceOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct MedpriceParams;

#[derive(Debug, Clone)]
pub struct MedpriceInput<'a> {
    pub data: MedpriceData<'a>,
    pub params: MedpriceParams,
}

impl<'a> MedpriceInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        high_source: &'a str,
        low_source: &'a str,
        params: MedpriceParams,
    ) -> Self {
        Self {
            data: MedpriceData::Candles {
                candles,
                high_source,
                low_source,
            },
            params,
        }
    }

    #[inline]
    pub fn from_slices(high: &'a [f64], low: &'a [f64], params: MedpriceParams) -> Self {
        Self {
            data: MedpriceData::Slices { high, low },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "high", "low", MedpriceParams::default())
    }

    #[inline]
    pub fn get_high_low(&self) -> (&[f64], &[f64]) {
        match &self.data {
            MedpriceData::Candles {
                candles,
                high_source,
                low_source,
            } => (
                source_type(candles, high_source),
                source_type(candles, low_source),
            ),
            MedpriceData::Slices { high, low } => (high, low),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct MedpriceBuilder {
    kernel: Kernel,
}

impl Default for MedpriceBuilder {
    fn default() -> Self {
        Self {
            kernel: Kernel::Auto,
        }
    }
}

impl MedpriceBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline]
    pub fn apply(self, candles: &Candles) -> Result<MedpriceOutput, MedpriceError> {
        let input = MedpriceInput::with_default_candles(candles);
        medprice_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<MedpriceOutput, MedpriceError> {
        let input = MedpriceInput::from_slices(high, low, MedpriceParams::default());
        medprice_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn into_stream(self) -> Result<MedpriceStream, MedpriceError> {
        MedpriceStream::try_new()
    }
}

#[derive(Debug, Error)]
pub enum MedpriceError {
    #[error("medprice: Empty data provided.")]
    EmptyData,
    #[error("medprice: Different lengths for high ({high_len}) and low ({low_len}).")]
    DifferentLength { high_len: usize, low_len: usize },
    #[error("medprice: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn medprice(input: &MedpriceInput) -> Result<MedpriceOutput, MedpriceError> {
    medprice_with_kernel(input, Kernel::Auto)
}

pub fn medprice_with_kernel(
    input: &MedpriceInput,
    kernel: Kernel,
) -> Result<MedpriceOutput, MedpriceError> {
    let (high, low) = input.get_high_low();

    if high.is_empty() || low.is_empty() {
        return Err(MedpriceError::EmptyData);
    }
    if high.len() != low.len() {
        return Err(MedpriceError::DifferentLength {
            high_len: high.len(),
            low_len: low.len(),
        });
    }

    let first_valid_idx = match (0..high.len()).find(|&i| !high[i].is_nan() && !low[i].is_nan()) {
        Some(idx) => idx,
        None => return Err(MedpriceError::AllValuesNaN),
    };

    let mut out = alloc_with_nan_prefix(high.len(), first_valid_idx);

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                medprice_scalar(high, low, first_valid_idx, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => medprice_avx2(high, low, first_valid_idx, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                medprice_avx512(high, low, first_valid_idx, &mut out)
            }
            _ => unreachable!(),
        }
    }

    Ok(MedpriceOutput { values: out })
}

/// Writes MEDPRICE results into a caller-provided buffer without allocating.
///
/// - Preserves NaN warmups exactly as `medprice()`/`medprice_with_kernel()`.
/// - `out.len()` must equal the input series length; returns an error on mismatch.
#[cfg(not(feature = "wasm"))]
#[inline]
pub fn medprice_into(input: &MedpriceInput, out: &mut [f64]) -> Result<(), MedpriceError> {
    // Delegate to the existing slice-based writer using Kernel::Auto.
    medprice_into_slice(out, input, Kernel::Auto)
}

#[inline(always)]
pub fn medprice_compute_into(
    high: &[f64],
    low: &[f64],
    kernel: Kernel,
    out: &mut [f64],
) -> Result<(), MedpriceError> {
    if high.is_empty() || low.is_empty() {
        return Err(MedpriceError::EmptyData);
    }
    if high.len() != low.len() {
        return Err(MedpriceError::DifferentLength {
            high_len: high.len(),
            low_len: low.len(),
        });
    }
    if out.len() != high.len() {
        return Err(MedpriceError::DifferentLength {
            high_len: high.len(),
            low_len: out.len(),
        });
    }

    let first = (0..high.len())
        .find(|&i| !high[i].is_nan() && !low[i].is_nan())
        .ok_or(MedpriceError::AllValuesNaN)?;

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => medprice_scalar(high, low, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => medprice_avx2(high, low, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => medprice_avx512(high, low, first, out),
            _ => unreachable!(),
        }
    }

    // Warm prefix NaNs after compute, like ALMA
    out[..first].fill(f64::NAN);
    Ok(())
}

#[inline]
pub fn medprice_scalar(high: &[f64], low: &[f64], first: usize, out: &mut [f64]) {
    // Write every index >= first. NaN input => NaN output.
    for i in first..high.len() {
        let h = high[i];
        let l = low[i];
        out[i] = if h.is_nan() || l.is_nan() {
            f64::NAN
        } else {
            (h + l) * 0.5
        };
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn medprice_avx2(high: &[f64], low: &[f64], first: usize, out: &mut [f64]) {
    // AVX2 stub, just call scalar
    medprice_scalar(high, low, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn medprice_avx512(high: &[f64], low: &[f64], first: usize, out: &mut [f64]) {
    // AVX512 stub, just call scalar
    medprice_scalar(high, low, first, out)
}

// Row functions
#[inline(always)]
pub unsafe fn medprice_row_scalar(high: &[f64], low: &[f64], first: usize, out: &mut [f64]) {
    medprice_scalar(high, low, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn medprice_row_avx2(high: &[f64], low: &[f64], first: usize, out: &mut [f64]) {
    medprice_avx2(high, low, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn medprice_row_avx512(high: &[f64], low: &[f64], first: usize, out: &mut [f64]) {
    medprice_avx512(high, low, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn medprice_row_avx512_short(high: &[f64], low: &[f64], first: usize, out: &mut [f64]) {
    medprice_avx512(high, low, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn medprice_row_avx512_long(high: &[f64], low: &[f64], first: usize, out: &mut [f64]) {
    medprice_avx512(high, low, first, out)
}

// Streaming (single-point) stateful
#[derive(Debug, Clone)]
pub struct MedpriceStream {
    started: bool,
}

impl MedpriceStream {
    pub fn try_new() -> Result<Self, MedpriceError> {
        Ok(Self { started: false })
    }

    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64) -> Option<f64> {
        if high.is_nan() || low.is_nan() {
            return None;
        }
        Some((high + low) * 0.5)
    }
}

// Batch/grid sweep for "expand_grid" compatibility (for future-proof API parity)
#[derive(Clone, Debug)]
pub struct MedpriceBatchRange {
    pub dummy: (usize, usize, usize), // for compatibility
}
impl Default for MedpriceBatchRange {
    fn default() -> Self {
        Self { dummy: (0, 0, 0) }
    }
}

#[derive(Clone, Debug, Default)]
pub struct MedpriceBatchBuilder {
    kernel: Kernel,
    range: MedpriceBatchRange,
}

impl MedpriceBatchBuilder {
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
    ) -> Result<MedpriceBatchOutput, MedpriceError> {
        medprice_batch_with_kernel(high, low, self.kernel)
    }
    pub fn apply_candles(
        self,
        c: &Candles,
        high_src: &str,
        low_src: &str,
    ) -> Result<MedpriceBatchOutput, MedpriceError> {
        let high = source_type(c, high_src);
        let low = source_type(c, low_src);
        self.apply_slice(high, low)
    }
}

pub fn medprice_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    k: Kernel,
) -> Result<MedpriceBatchOutput, MedpriceError> {
    let simd = match k {
        Kernel::Auto => match detect_best_batch_kernel() {
            Kernel::Avx512Batch => Kernel::Avx512,
            Kernel::Avx2Batch => Kernel::Avx2,
            Kernel::ScalarBatch => Kernel::Scalar,
            _ => unreachable!(),
        },
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        other => other, // allow direct simd or scalar for tests
    };
    medprice_batch_par_slice(high, low, simd)
}

#[derive(Clone, Debug)]
pub struct MedpriceBatchOutput {
    pub values: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

#[inline(always)]
fn expand_grid(_r: &MedpriceBatchRange) -> Vec<MedpriceParams> {
    vec![MedpriceParams::default()]
}

#[inline(always)]
pub fn medprice_batch_slice(
    high: &[f64],
    low: &[f64],
    kern: Kernel,
) -> Result<MedpriceBatchOutput, MedpriceError> {
    medprice_batch_inner(high, low, kern, false)
}

#[inline(always)]
pub fn medprice_batch_par_slice(
    high: &[f64],
    low: &[f64],
    kern: Kernel,
) -> Result<MedpriceBatchOutput, MedpriceError> {
    medprice_batch_inner(high, low, kern, true)
}

#[inline(always)]
fn medprice_batch_inner(
    high: &[f64],
    low: &[f64],
    kern: Kernel,
    _parallel: bool,
) -> Result<MedpriceBatchOutput, MedpriceError> {
    if high.is_empty() || low.is_empty() {
        return Err(MedpriceError::EmptyData);
    }
    if high.len() != low.len() {
        return Err(MedpriceError::DifferentLength {
            high_len: high.len(),
            low_len: low.len(),
        });
    }

    let first = match (0..high.len()).find(|&i| !high[i].is_nan() && !low[i].is_nan()) {
        Some(idx) => idx,
        None => return Err(MedpriceError::AllValuesNaN),
    };

    let rows = 1;
    let cols = high.len();

    let mut buf_mu = make_uninit_matrix(rows, cols);
    let warmup_periods = vec![first];
    init_matrix_prefixes(&mut buf_mu, cols, &warmup_periods);

    let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len())
    };

    let chosen = match kern {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    medprice_batch_inner_into(high, low, chosen, _parallel, out)?;

    let values = unsafe {
        Vec::from_raw_parts(
            buf_guard.as_mut_ptr() as *mut f64,
            buf_guard.len(),
            buf_guard.capacity(),
        )
    };

    Ok(MedpriceBatchOutput { values, rows, cols })
}

#[inline(always)]
fn medprice_batch_inner_into(
    high: &[f64],
    low: &[f64],
    kern: Kernel,
    _parallel: bool,
    out: &mut [f64],
) -> Result<Vec<MedpriceParams>, MedpriceError> {
    // For medprice, we only have one "parameter set" since it has no parameters
    let combos = vec![MedpriceParams::default()];

    if high.is_empty() || low.is_empty() {
        return Err(MedpriceError::EmptyData);
    }
    if high.len() != low.len() {
        return Err(MedpriceError::DifferentLength {
            high_len: high.len(),
            low_len: low.len(),
        });
    }

    let first = match (0..high.len()).find(|&i| !high[i].is_nan() && !low[i].is_nan()) {
        Some(idx) => idx,
        None => return Err(MedpriceError::AllValuesNaN),
    };

    // Since we only have one row, we can use the kernel directly
    unsafe {
        match kern {
            Kernel::Scalar | Kernel::ScalarBatch => medprice_scalar(high, low, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => medprice_avx2(high, low, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => medprice_avx512(high, low, first, out),
            _ => unreachable!(),
        }
    }

    Ok(combos)
}

// =============================================================================
// WASM helper functions
// =============================================================================

#[inline]
pub fn medprice_into_slice_raw(
    dst: &mut [f64],
    high: &[f64],
    low: &[f64],
    kern: Kernel,
) -> Result<(), MedpriceError> {
    medprice_compute_into(high, low, kern, dst)
}

/// Write medprice results directly into pre-allocated slice.
/// This version takes MedpriceInput and follows the ALMA pattern.
#[inline]
pub fn medprice_into_slice(
    dst: &mut [f64],
    input: &MedpriceInput,
    kern: Kernel,
) -> Result<(), MedpriceError> {
    let (high, low) = input.get_high_low();

    if high.is_empty() || low.is_empty() {
        return Err(MedpriceError::EmptyData);
    }
    if high.len() != low.len() {
        return Err(MedpriceError::DifferentLength {
            high_len: high.len(),
            low_len: low.len(),
        });
    }
    if dst.len() != high.len() {
        return Err(MedpriceError::DifferentLength {
            high_len: high.len(),
            low_len: dst.len(),
        });
    }

    let first_valid_idx = match (0..high.len()).find(|&i| !high[i].is_nan() && !low[i].is_nan()) {
        Some(idx) => idx,
        None => return Err(MedpriceError::AllValuesNaN),
    };

    let chosen = match kern {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                medprice_scalar(high, low, first_valid_idx, dst)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => medprice_avx2(high, low, first_valid_idx, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                medprice_avx512(high, low, first_valid_idx, dst)
            }
            _ => unreachable!(),
        }
    }

    // Fill warmup period with NaN
    for v in &mut dst[..first_valid_idx] {
        *v = f64::NAN;
    }

    Ok(())
}

// =============================================================================
// Python bindings
// =============================================================================

#[cfg(feature = "python")]
#[pyfunction(name = "medprice")]
#[pyo3(signature = (high, low, kernel=None))]
pub fn medprice_py<'py>(
    py: Python<'py>,
    high: numpy::PyReadonlyArray1<'py, f64>,
    low: numpy::PyReadonlyArray1<'py, f64>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let input = MedpriceInput::from_slices(high_slice, low_slice, MedpriceParams::default());

    let result_vec: Vec<f64> = py
        .allow_threads(|| medprice_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "MedpriceStream")]
pub struct MedpriceStreamPy {
    stream: MedpriceStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl MedpriceStreamPy {
    #[new]
    fn new() -> PyResult<Self> {
        let stream = MedpriceStream::try_new().map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(MedpriceStreamPy { stream })
    }

    fn update(&mut self, high: f64, low: f64) -> Option<f64> {
        self.stream.update(high, low)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "medprice_batch")]
#[pyo3(signature = (high, low, dummy_range=None, kernel=None))]
pub fn medprice_batch_py<'py>(
    py: Python<'py>,
    high: numpy::PyReadonlyArray1<'py, f64>,
    low: numpy::PyReadonlyArray1<'py, f64>,
    dummy_range: Option<(usize, usize, usize)>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;

    // Since medprice has no parameters, we just use a dummy range
    let _range = dummy_range.unwrap_or((0, 0, 0));

    // Always 1 row for medprice
    let rows = 1;
    let cols = high_slice.len();

    // Pre-allocate output array for batch operations
    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    let kern = validate_kernel(kernel, true)?;

    let _combos = py
        .allow_threads(|| {
            let kernel = match kern {
                Kernel::Auto => detect_best_batch_kernel(),
                k => k,
            };
            medprice_batch_inner_into(high_slice, low_slice, kernel, true, slice_out)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    // For medprice, we don't have any parameters to return
    dict.set_item("params", Vec::<u64>::new().into_pyarray(py))?;

    Ok(dict)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::utilities::enums::Kernel;

    fn check_medprice_with_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = MedpriceInput::with_default_candles(&candles);
        let output = medprice_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_medprice_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = MedpriceInput::from_candles(&candles, "high", "low", MedpriceParams);
        let result = medprice_with_kernel(&input, kernel)?;
        assert_eq!(
            result.values.len(),
            candles.close.len(),
            "Output length mismatch"
        );
        let expected_last_five = [59166.0, 59244.5, 59118.0, 59146.5, 58767.5];
        assert!(result.values.len() >= 5, "Not enough data for comparison");
        let start_index = result.values.len() - 5;
        let actual_last_five = &result.values[start_index..];
        for (i, &val) in actual_last_five.iter().enumerate() {
            let expected = expected_last_five[i];
            assert!(
                (val - expected).abs() < 1e-1,
                "Mismatch at last five index {}: expected {}, got {}",
                i,
                expected,
                val
            );
        }
        Ok(())
    }

    fn check_medprice_empty_data(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [];
        let low = [];
        let input = MedpriceInput::from_slices(&high, &low, MedpriceParams);
        let result = medprice_with_kernel(&input, kernel);
        assert!(result.is_err(), "Expected error for empty data");
        Ok(())
    }

    fn check_medprice_different_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 20.0, 30.0];
        let low = [5.0, 15.0];
        let input = MedpriceInput::from_slices(&high, &low, MedpriceParams);
        let result = medprice_with_kernel(&input, kernel);
        assert!(
            result.is_err(),
            "Expected error for different slice lengths"
        );
        Ok(())
    }

    fn check_medprice_all_values_nan(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [f64::NAN, f64::NAN, f64::NAN];
        let low = [f64::NAN, f64::NAN, f64::NAN];
        let input = MedpriceInput::from_slices(&high, &low, MedpriceParams);
        let result = medprice_with_kernel(&input, kernel);
        assert!(result.is_err(), "Expected error for all NaN data");
        Ok(())
    }

    fn check_medprice_nan_handling(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [f64::NAN, 100.0, 110.0];
        let low = [f64::NAN, 80.0, 90.0];
        let input = MedpriceInput::from_slices(&high, &low, MedpriceParams);
        let result = medprice_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), 3);
        assert!(result.values[0].is_nan());
        assert_eq!(result.values[1], 90.0);
        assert_eq!(result.values[2], 100.0);
        Ok(())
    }

    fn check_medprice_late_nan_handling(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [100.0, 110.0, f64::NAN];
        let low = [80.0, 90.0, f64::NAN];
        let input = MedpriceInput::from_slices(&high, &low, MedpriceParams);
        let result = medprice_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), 3);
        assert_eq!(result.values[0], 90.0);
        assert_eq!(result.values[1], 100.0);
        assert!(result.values[2].is_nan());
        Ok(())
    }

    fn check_medprice_streaming(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [100.0, 110.0, 120.0];
        let low = [80.0, 90.0, 100.0];
        let mut stream = MedpriceStream::try_new()?;
        let mut values = Vec::with_capacity(high.len());
        for (&h, &l) in high.iter().zip(low.iter()) {
            values.push(stream.update(h, l));
        }
        assert_eq!(values[0], Some(90.0));
        assert_eq!(values[1], Some(100.0));
        assert_eq!(values[2], Some(110.0));
        Ok(())
    }

    fn check_medprice_batch(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [100.0, 110.0, 120.0];
        let low = [80.0, 90.0, 100.0];
        let builder = MedpriceBatchBuilder::new().kernel(kernel);
        let batch = builder.apply_slice(&high, &low)?;
        assert_eq!(batch.values.len(), high.len());
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, 3);
        assert_eq!(batch.values, vec![90.0, 100.0, 110.0]);
        Ok(())
    }

    macro_rules! generate_all_medprice_tests {
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

    generate_all_medprice_tests!(
        check_medprice_with_default_candles,
        check_medprice_accuracy,
        check_medprice_empty_data,
        check_medprice_different_length,
        check_medprice_all_values_nan,
        check_medprice_nan_handling,
        check_medprice_late_nan_handling,
        check_medprice_streaming,
        check_medprice_batch
    );
    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let high = source_type(&c, "high");
        let low = source_type(&c, "low");

        let output = MedpriceBatchBuilder::new()
            .kernel(kernel)
            .apply_slice(high, low)?;

        assert_eq!(output.rows, 1, "[{test}] batch output should have one row");
        assert_eq!(output.cols, high.len(), "[{test}] batch cols mismatch");
        assert_eq!(
            output.values.len(),
            output.cols,
            "[{test}] values shape mismatch"
        );

        let last_expected = [59166.0, 59244.5, 59118.0, 59146.5, 58767.5];
        let start = output.values.len().saturating_sub(5);
        for (i, &val) in output.values[start..].iter().enumerate() {
            assert!(
                (val - last_expected[i]).abs() < 1e-1,
                "[{test}] batch last-five mismatch idx {i}: got {val}, expected {}",
                last_expected[i]
            );
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

    gen_batch_tests!(check_batch_default_row);

    #[test]
    fn test_medprice_into_matches_api() {
        // Build a small-but-nontrivial OHLCV series with NaN warmup
        let n = 256usize;
        let mut ts: Vec<i64> = (0..n as i64).collect();
        let mut open = vec![0.0; n];
        let mut high = vec![0.0; n];
        let mut low = vec![0.0; n];
        let mut close = vec![0.0; n];
        let mut vol = vec![1.0; n];

        for i in 0..n {
            if i < 3 {
                // Warmup NaNs in high/low
                high[i] = f64::NAN;
                low[i] = f64::NAN;
                open[i] = f64::NAN;
                close[i] = f64::NAN;
            } else {
                // Varying prices
                let x = i as f64;
                low[i] = 95.0 + (x.sin() * 2.0);
                high[i] = low[i] + 10.0 + (x.cos());
                open[i] = low[i] + 3.0;
                close[i] = high[i] - 4.0;
            }
        }

        let candles = crate::utilities::data_loader::Candles::new(ts.clone(), open, high.clone(), low.clone(), close, vol);
        let input = MedpriceInput::with_default_candles(&candles);

        // Baseline via Vec-returning API
        let baseline = medprice(&input).expect("baseline medprice failed").values;

        // Preallocate destination and compute via into-API
        let mut out = vec![0.0; baseline.len()];
        #[cfg(not(feature = "wasm"))]
        {
            medprice_into(&input, &mut out).expect("medprice_into failed");
        }

        assert_eq!(baseline.len(), out.len());

        // Equality: NaN == NaN, else exact equality (identical path)
        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b)
        }

        for i in 0..out.len() {
            assert!(
                eq_or_both_nan(baseline[i], out[i]),
                "value mismatch at {}: baseline={:?}, into={:?}",
                i,
                baseline[i],
                out[i]
            );
        }
    }
}

// =============================================================================
// WASM bindings
// =============================================================================

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn medprice_js(high: &[f64], low: &[f64]) -> Result<Vec<f64>, JsValue> {
    let mut output = vec![0.0; high.len()];

    medprice_into_slice_raw(&mut output, high, low, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn medprice_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn medprice_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn medprice_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
) -> Result<(), JsValue> {
    if high_ptr.is_null() || low_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to medprice_into"));
    }

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);

        // Check if any input pointer equals output pointer (aliasing)
        if high_ptr == out_ptr || low_ptr == out_ptr {
            let mut temp = vec![0.0; len];
            medprice_into_slice_raw(&mut temp, high, low, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            medprice_into_slice_raw(out, high, low, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct MedpriceBatchConfig {
    pub dummy_range: (usize, usize, usize), // For API consistency
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct MedpriceBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<MedpriceParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = medprice_batch)]
pub fn medprice_batch_js(high: &[f64], low: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    // Since medprice has no parameters, we ignore config and just compute once
    let _config: Option<MedpriceBatchConfig> = if config.is_object() {
        serde_wasm_bindgen::from_value(config).ok()
    } else {
        None
    };

    let mut output = vec![0.0; high.len()];
    medprice_into_slice_raw(&mut output, high, low, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = MedpriceBatchJsOutput {
        values: output,
        combos: vec![MedpriceParams::default()], // Single empty params
        rows: 1,
        cols: high.len(),
    };

    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "python")]
pub fn register_medprice_module(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(medprice_py, m)?)?;
    m.add_function(wrap_pyfunction!(medprice_batch_py, m)?)?;
    Ok(())
}

// =============================================================================
// Python CUDA bindings (device-backed arrays)
// =============================================================================

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "medprice_cuda_dev")]
#[pyo3(signature = (high, low, device_id=0))]
pub fn medprice_cuda_dev_py(
    py: Python<'_>,
    high: numpy::PyReadonlyArray1<'_, f32>,
    low: numpy::PyReadonlyArray1<'_, f32>,
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let hs = high.as_slice()?;
    let ls = low.as_slice()?;

    let inner = py.allow_threads(|| {
        let cuda =
            CudaMedprice::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.medprice_dev(hs, ls)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    Ok(DeviceArrayF32Py { inner })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "medprice_cuda_batch_dev")]
#[pyo3(signature = (high, low, device_id=0))]
pub fn medprice_cuda_batch_dev_py(
    py: Python<'_>,
    high: numpy::PyReadonlyArray1<'_, f32>,
    low: numpy::PyReadonlyArray1<'_, f32>,
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let hs = high.as_slice()?;
    let ls = low.as_slice()?;
    let inner = py.allow_threads(|| {
        let cuda =
            CudaMedprice::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.medprice_batch_dev(hs, ls)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    Ok(DeviceArrayF32Py { inner })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "medprice_cuda_many_series_one_param_dev")]
#[pyo3(signature = (high_tm, low_tm, cols, rows, device_id=0))]
pub fn medprice_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    high_tm: numpy::PyReadonlyArray1<'_, f32>,
    low_tm: numpy::PyReadonlyArray1<'_, f32>,
    cols: usize,
    rows: usize,
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let hs = high_tm.as_slice()?;
    let ls = low_tm.as_slice()?;
    let inner = py.allow_threads(|| {
        let cuda =
            CudaMedprice::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.medprice_many_series_one_param_time_major_dev(hs, ls, cols, rows)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    Ok(DeviceArrayF32Py { inner })
}
