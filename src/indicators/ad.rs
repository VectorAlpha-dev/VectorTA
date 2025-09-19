//! # Chaikin Accumulation/Distribution (AD)
//!
//! Volume-based cumulative money flow indicator using high, low, close, and volume.
//!
//! ## Parameters
//! - No adjustable parameters beyond input data.
//!
//! ## Returns
//! - **Ok(AdOutput)** on success, with AD values.
//! - **Err(AdError)** otherwise.
//!
//! ## Developer Notes
//! - **AVX2/AVX512 kernels**: Stubs (all call scalar implementation)
//! - **Streaming update**: O(1) - simple cumulative sum calculation
//! - **Memory optimization**: Uses zero-copy helpers (alloc_with_nan_prefix)
//! - **Optimization needed**: Implement actual SIMD kernels for batch processing

use crate::utilities::data_loader::Candles;
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, make_uninit_matrix,
};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList, PyListMethods};
#[cfg(feature = "python")]
use pyo3::{pyfunction, Bound, PyResult, Python};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum AdData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        volume: &'a [f64],
    },
}

#[derive(Debug, Clone, Default)]
pub struct AdParams {}

#[derive(Debug, Clone)]
pub struct AdInput<'a> {
    pub data: AdData<'a>,
    pub params: AdParams,
}

impl<'a> AdInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: AdParams) -> Self {
        Self {
            data: AdData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        volume: &'a [f64],
        params: AdParams,
    ) -> Self {
        Self {
            data: AdData::Slices {
                high,
                low,
                close,
                volume,
            },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, AdParams::default())
    }
}

#[derive(Debug, Clone)]
pub struct AdOutput {
    pub values: Vec<f64>,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct AdBuilder {
    kernel: Kernel,
}

impl AdBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            kernel: Kernel::Auto,
        }
    }

    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<AdOutput, AdError> {
        let input = AdInput::from_candles(c, AdParams::default());
        ad_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> Result<AdOutput, AdError> {
        let input = AdInput::from_slices(high, low, close, volume, AdParams::default());
        ad_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<AdStream, AdError> {
        AdStream::try_new()
    }
}

#[derive(Debug, Error)]
pub enum AdError {
    #[error("ad: Candle field error: {0}")]
    CandleFieldError(String),
    #[error("ad: Data length mismatch for AD calculation: high={high_len}, low={low_len}, close={close_len}, volume={volume_len}")]
    DataLengthMismatch {
        high_len: usize,
        low_len: usize,
        close_len: usize,
        volume_len: usize,
    },
    #[error("ad: Not enough data points to calculate AD. Length={len}")]
    NotEnoughData { len: usize },
}

#[inline]
pub fn ad(input: &AdInput) -> Result<AdOutput, AdError> {
    ad_with_kernel(input, Kernel::Auto)
}

pub fn ad_with_kernel(input: &AdInput, kernel: Kernel) -> Result<AdOutput, AdError> {
    let (high, low, close, volume): (&[f64], &[f64], &[f64], &[f64]) = match &input.data {
        AdData::Candles { candles } => {
            let high = candles
                .select_candle_field("high")
                .map_err(|e| AdError::CandleFieldError(e.to_string()))?;
            let low = candles
                .select_candle_field("low")
                .map_err(|e| AdError::CandleFieldError(e.to_string()))?;
            let close = candles
                .select_candle_field("close")
                .map_err(|e| AdError::CandleFieldError(e.to_string()))?;
            let volume = candles
                .select_candle_field("volume")
                .map_err(|e| AdError::CandleFieldError(e.to_string()))?;
            (high, low, close, volume)
        }
        AdData::Slices {
            high,
            low,
            close,
            volume,
        } => (*high, *low, *close, *volume),
    };

    if high.len() != low.len() || high.len() != close.len() || high.len() != volume.len() {
        return Err(AdError::DataLengthMismatch {
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
            volume_len: volume.len(),
        });
    }

    let size = high.len();
    if size < 1 {
        return Err(AdError::NotEnoughData { len: size });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };

    // For AD, warmup period is 0 since it's a cumulative indicator starting from 0
    // We use alloc_with_nan_prefix with warmup=0 to get an uninitialized vector
    let mut out = alloc_with_nan_prefix(size, 0);

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => ad_scalar(high, low, close, volume, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => ad_avx2(high, low, close, volume, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => ad_avx512(high, low, close, volume, &mut out),
            _ => unreachable!(),
        }
    }
    Ok(AdOutput { values: out })
}

/// Write AD values directly to output slice - no allocations
pub fn ad_into_slice(dst: &mut [f64], input: &AdInput, kern: Kernel) -> Result<(), AdError> {
    let (high, low, close, volume) = match &input.data {
        AdData::Candles { candles, .. } => (
            &candles.high[..],
            &candles.low[..],
            &candles.close[..],
            &candles.volume[..],
        ),
        AdData::Slices {
            high,
            low,
            close,
            volume,
        } => (*high, *low, *close, *volume),
    };

    // Check for empty input
    if high.is_empty() {
        return Err(AdError::NotEnoughData { len: 0 });
    }

    // Validate array lengths
    if high.len() != low.len() || high.len() != close.len() || high.len() != volume.len() {
        return Err(AdError::DataLengthMismatch {
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
            volume_len: volume.len(),
        });
    }

    if dst.len() != high.len() {
        return Err(AdError::DataLengthMismatch {
            high_len: high.len(),
            low_len: dst.len(),
            close_len: dst.len(),
            volume_len: dst.len(),
        });
    }

    // Compute AD values directly into dst
    match kern {
        Kernel::Auto => {
            let k = detect_best_kernel();
            match k {
                Kernel::Scalar => ad_scalar(high, low, close, volume, dst),
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx2 => ad_avx2(high, low, close, volume, dst),
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx512 => ad_avx512(high, low, close, volume, dst),
                _ => ad_scalar(high, low, close, volume, dst),
            }
        }
        Kernel::Scalar => ad_scalar(high, low, close, volume, dst),
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2 => ad_avx2(high, low, close, volume, dst),
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512 => ad_avx512(high, low, close, volume, dst),
        _ => ad_scalar(high, low, close, volume, dst),
    }

    // AD has no warmup period, so no NaN filling needed
    Ok(())
}

#[inline]
pub fn ad_scalar(high: &[f64], low: &[f64], close: &[f64], volume: &[f64], out: &mut [f64]) {
    let size = high.len();
    let mut sum = 0.0;
    for i in 0..size {
        let hl = high[i] - low[i];
        if hl != 0.0 {
            let mfm = ((close[i] - low[i]) - (high[i] - close[i])) / hl;
            let mfv = mfm * volume[i];
            sum += mfv;
        }
        out[i] = sum;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn ad_avx2(high: &[f64], low: &[f64], close: &[f64], volume: &[f64], out: &mut [f64]) {
    ad_scalar(high, low, close, volume, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn ad_avx512(high: &[f64], low: &[f64], close: &[f64], volume: &[f64], out: &mut [f64]) {
    ad_scalar(high, low, close, volume, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn ad_avx512_short(high: &[f64], low: &[f64], close: &[f64], volume: &[f64], out: &mut [f64]) {
    ad_scalar(high, low, close, volume, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn ad_avx512_long(high: &[f64], low: &[f64], close: &[f64], volume: &[f64], out: &mut [f64]) {
    ad_scalar(high, low, close, volume, out)
}

#[inline]
pub fn ad_batch_with_kernel(data: &AdBatchInput, k: Kernel) -> Result<AdBatchOutput, AdError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => Kernel::ScalarBatch,
    };

    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    ad_batch_par_slice(data, simd)
}

#[derive(Clone, Debug)]
pub struct AdBatchInput<'a> {
    pub highs: &'a [&'a [f64]],
    pub lows: &'a [&'a [f64]],
    pub closes: &'a [&'a [f64]],
    pub volumes: &'a [&'a [f64]],
}

#[derive(Clone, Debug)]
pub struct AdBatchOutput {
    pub values: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

#[inline(always)]
pub fn ad_batch_slice(data: &AdBatchInput, kern: Kernel) -> Result<AdBatchOutput, AdError> {
    ad_batch_inner(data, kern, false)
}

#[inline(always)]
pub fn ad_batch_par_slice(data: &AdBatchInput, kern: Kernel) -> Result<AdBatchOutput, AdError> {
    ad_batch_inner(data, kern, true)
}

fn ad_batch_inner(
    data: &AdBatchInput,
    kern: Kernel,
    parallel: bool,
) -> Result<AdBatchOutput, AdError> {
    let rows = data.highs.len();
    let cols = if rows > 0 { data.highs[0].len() } else { 0 };

    // Use make_uninit_matrix for better poison detection during debugging
    // AD has no warmup period, so we don't need init_matrix_prefixes
    let mut buf_mu = make_uninit_matrix(rows, cols);
    let values = unsafe {
        let ptr = buf_mu.as_mut_ptr() as *mut f64;
        let slice = std::slice::from_raw_parts_mut(ptr, rows * cols);

        // Compute into the buffer
        ad_batch_inner_into(data, kern, parallel, slice)?;

        // Convert to Vec for output
        Vec::from_raw_parts(ptr, rows * cols, rows * cols)
    };
    std::mem::forget(buf_mu);

    Ok(AdBatchOutput { values, rows, cols })
}

fn ad_batch_inner_into(
    data: &AdBatchInput,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<(), AdError> {
    let rows = data.highs.len();
    let cols = if rows > 0 { data.highs[0].len() } else { 0 };

    // Validate that all input arrays have the same number of rows
    if data.lows.len() != rows || data.closes.len() != rows || data.volumes.len() != rows {
        return Err(AdError::DataLengthMismatch {
            high_len: data.highs.len(),
            low_len: data.lows.len(),
            close_len: data.closes.len(),
            volume_len: data.volumes.len(),
        });
    }

    // Validate that each row has consistent length across all OHLCV arrays and matches cols
    for row in 0..rows {
        let h_len = data.highs[row].len();
        let l_len = data.lows[row].len();
        let c_len = data.closes[row].len();
        let v_len = data.volumes[row].len();

        if h_len != cols || l_len != cols || c_len != cols || v_len != cols {
            return Err(AdError::DataLengthMismatch {
                high_len: h_len,
                low_len: l_len,
                close_len: c_len,
                volume_len: v_len,
            });
        }
    }

    if out.len() != rows * cols {
        return Err(AdError::DataLengthMismatch {
            high_len: rows,
            low_len: rows,
            close_len: rows,
            volume_len: out.len(),
        });
    }

    // Resolve actual kernel for row computation
    let actual = match kern {
        Kernel::Auto => detect_best_batch_kernel(),
        k => k,
    };

    let do_row = |row: usize, dst: &mut [f64]| {
        // All row variants call scalar for now (SIMD ignored by request)
        unsafe {
            match actual {
                Kernel::Scalar | Kernel::ScalarBatch => ad_row_scalar(
                    data.highs[row],
                    data.lows[row],
                    data.closes[row],
                    data.volumes[row],
                    dst,
                ),
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx2 | Kernel::Avx2Batch => ad_row_scalar(
                    data.highs[row],
                    data.lows[row],
                    data.closes[row],
                    data.volumes[row],
                    dst,
                ),
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx512 | Kernel::Avx512Batch => ad_row_scalar(
                    data.highs[row],
                    data.lows[row],
                    data.closes[row],
                    data.volumes[row],
                    dst,
                ),
                _ => ad_row_scalar(
                    data.highs[row],
                    data.lows[row],
                    data.closes[row],
                    data.volumes[row],
                    dst,
                ),
            }
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;
            out.par_chunks_mut(cols)
                .enumerate()
                .for_each(|(r, s)| do_row(r, s));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (r, s) in out.chunks_mut(cols).enumerate() {
                do_row(r, s);
            }
        }
    } else {
        for (r, s) in out.chunks_mut(cols).enumerate() {
            do_row(r, s);
        }
    }

    Ok(())
}

#[inline(always)]
pub unsafe fn ad_row_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    out: &mut [f64],
) {
    ad_scalar(high, low, close, volume, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn ad_row_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    out: &mut [f64],
) {
    ad_row_scalar(high, low, close, volume, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn ad_row_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    out: &mut [f64],
) {
    ad_row_scalar(high, low, close, volume, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn ad_row_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    out: &mut [f64],
) {
    ad_row_scalar(high, low, close, volume, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn ad_row_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    out: &mut [f64],
) {
    ad_row_scalar(high, low, close, volume, out)
}

#[derive(Debug, Clone)]
pub struct AdStream {
    sum: f64,
}

impl AdStream {
    pub fn try_new() -> Result<Self, AdError> {
        Ok(Self { sum: 0.0 })
    }

    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64, close: f64, volume: f64) -> f64 {
        let hl = high - low;
        if hl != 0.0 {
            let mfm = ((close - low) - (high - close)) / hl;
            let mfv = mfm * volume;
            self.sum += mfv;
        }
        self.sum
    }
}

// Batch Builder for parity with Alma
#[derive(Clone, Debug, Default)]
pub struct AdBatchBuilder {
    pub kernel: Kernel,
}

impl AdBatchBuilder {
    pub fn new() -> Self {
        Self {
            kernel: Kernel::Auto,
        }
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    pub fn apply_slices(
        self,
        highs: &[&[f64]],
        lows: &[&[f64]],
        closes: &[&[f64]],
        volumes: &[&[f64]],
    ) -> Result<AdBatchOutput, AdError> {
        let batch = AdBatchInput {
            highs,
            lows,
            closes,
            volumes,
        };
        ad_batch_with_kernel(&batch, self.kernel)
    }
}

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1};
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "python")]
#[pyfunction(name = "ad")]
#[pyo3(signature = (high, low, close, volume, kernel=None))]
/// Compute the Chaikin Accumulation/Distribution (AD) indicator.
///
/// AD is a volume-based cumulative money flow indicator that uses the relationship
/// between close, high, and low prices to determine the Money Flow Multiplier,
/// which is then multiplied by volume to create Money Flow Volume.
///
/// Parameters:
/// -----------
/// high : np.ndarray
///     High prices array (float64).
/// low : np.ndarray
///     Low prices array (float64).
/// close : np.ndarray
///     Close prices array (float64).
/// volume : np.ndarray
///     Volume array (float64).
/// kernel : str, optional
///     Computation kernel to use: 'auto', 'scalar', 'avx2', 'avx512'.
///     Default is 'auto' which auto-detects the best available.
///
/// Returns:
/// --------
/// np.ndarray
///     Array of AD values, same length as input.
///
/// Raises:
/// -------
/// ValueError
///     If input arrays have different lengths or are empty.
pub fn ad_py<'py>(
    py: Python<'py>,
    high: numpy::PyReadonlyArray1<'py, f64>,
    low: numpy::PyReadonlyArray1<'py, f64>,
    close: numpy::PyReadonlyArray1<'py, f64>,
    volume: numpy::PyReadonlyArray1<'py, f64>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    // Zero-copy views of input arrays
    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;
    let volume_slice = volume.as_slice()?;

    // Parse and validate kernel
    let kern = crate::utilities::kernel_validation::validate_kernel(kernel, false)?;

    // Create input struct
    let input = AdInput::from_slices(
        high_slice,
        low_slice,
        close_slice,
        volume_slice,
        AdParams::default(),
    );

    // Compute without GIL and get Vec<f64> result
    let result_vec: Vec<f64> = py
        .allow_threads(|| ad_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Zero-copy transfer to NumPy
    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "AdStream")]
pub struct AdStreamPy {
    stream: AdStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl AdStreamPy {
    #[new]
    fn new() -> PyResult<Self> {
        let stream = AdStream::try_new().map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(AdStreamPy { stream })
    }

    /// Updates the stream with new OHLCV data and returns the calculated AD value.
    fn update(&mut self, high: f64, low: f64, close: f64, volume: f64) -> f64 {
        self.stream.update(high, low, close, volume)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "ad_batch")]
#[pyo3(signature = (highs, lows, closes, volumes, kernel=None))]
/// Compute AD for multiple securities in a single pass.
///
/// This function processes multiple securities (rows) efficiently in parallel.
/// Each row represents a different security's time series data.
///
/// Parameters:
/// -----------
/// highs : List[np.ndarray]
///     List of high price arrays, one per security.
/// lows : List[np.ndarray]
///     List of low price arrays, one per security.
/// closes : List[np.ndarray]
///     List of close price arrays, one per security.
/// volumes : List[np.ndarray]
///     List of volume arrays, one per security.
/// kernel : str, optional
///     Computation kernel: 'auto', 'scalar', 'avx2', 'avx512'.
///     Default is 'auto' which auto-detects the best available.
///
/// Returns:
/// --------
/// dict
///     Dictionary with 'values' (2D array where each row is a security), 'rows', and 'cols'.
pub fn ad_batch_py<'py>(
    py: Python<'py>,
    highs: &Bound<'py, PyList>,
    lows: &Bound<'py, PyList>,
    closes: &Bound<'py, PyList>,
    volumes: &Bound<'py, PyList>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
    use pyo3::types::PyDict;

    let rows = highs.len();
    if lows.len() != rows || closes.len() != rows || volumes.len() != rows {
        return Err(PyValueError::new_err(
            "All input lists must have the same length",
        ));
    }

    // Hold PyReadonlyArray objects to keep them alive
    let mut high_arrays: Vec<PyReadonlyArray1<f64>> = Vec::with_capacity(rows);
    let mut low_arrays: Vec<PyReadonlyArray1<f64>> = Vec::with_capacity(rows);
    let mut close_arrays: Vec<PyReadonlyArray1<f64>> = Vec::with_capacity(rows);
    let mut volume_arrays: Vec<PyReadonlyArray1<f64>> = Vec::with_capacity(rows);

    for i in 0..rows {
        let h = highs.get_item(i)?.extract::<PyReadonlyArray1<f64>>()?;
        let l = lows.get_item(i)?.extract::<PyReadonlyArray1<f64>>()?;
        let c = closes.get_item(i)?.extract::<PyReadonlyArray1<f64>>()?;
        let v = volumes.get_item(i)?.extract::<PyReadonlyArray1<f64>>()?;
        // Validate equal lengths per row
        let n = h.len()?;
        if l.len()? != n || c.len()? != n || v.len()? != n {
            return Err(PyValueError::new_err(
                "Rows must have equal lengths across OHLCV arrays",
            ));
        }
        high_arrays.push(h);
        low_arrays.push(l);
        close_arrays.push(c);
        volume_arrays.push(v);
    }

    // Now borrow slices from the arrays (zero-copy)
    let high_slices: Vec<&[f64]> = high_arrays.iter().map(|a| a.as_slice().unwrap()).collect();
    let low_slices: Vec<&[f64]> = low_arrays.iter().map(|a| a.as_slice().unwrap()).collect();
    let close_slices: Vec<&[f64]> = close_arrays.iter().map(|a| a.as_slice().unwrap()).collect();
    let volume_slices: Vec<&[f64]> = volume_arrays
        .iter()
        .map(|a| a.as_slice().unwrap())
        .collect();

    let cols = if rows > 0 { high_slices[0].len() } else { 0 };
    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let out_slice = unsafe { out_arr.as_slice_mut()? };

    let kern = crate::utilities::kernel_validation::validate_kernel(kernel, true)?;

    py.allow_threads(|| -> Result<(), AdError> {
        let batch_input = AdBatchInput {
            highs: &high_slices,
            lows: &low_slices,
            closes: &close_slices,
            volumes: &volume_slices,
        };

        let actual = match kern {
            Kernel::Auto => detect_best_batch_kernel(),
            k => k,
        };
        ad_batch_inner_into(&batch_input, actual, true, out_slice)
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item("rows", rows)?; // parity metadata
    dict.set_item("cols", cols)?; // parity metadata
    Ok(dict)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ad_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
) -> Result<Vec<f64>, JsValue> {
    let input = AdInput::from_slices(high, low, close, volume, AdParams::default());

    let mut output = vec![0.0; high.len()]; // Single allocation
    ad_into_slice(&mut output, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ad_batch_js(
    highs_flat: &[f64],
    lows_flat: &[f64],
    closes_flat: &[f64],
    volumes_flat: &[f64],
    rows: usize,
) -> Result<Vec<f64>, JsValue> {
    if highs_flat.is_empty() || rows == 0 {
        return Err(JsValue::from_str("Empty input data"));
    }

    let cols = highs_flat.len() / rows;
    if highs_flat.len() != rows * cols
        || lows_flat.len() != rows * cols
        || closes_flat.len() != rows * cols
        || volumes_flat.len() != rows * cols
    {
        return Err(JsValue::from_str(
            "Input arrays must have rows*cols elements",
        ));
    }

    // Convert flattened arrays to slices
    let mut high_slices = Vec::with_capacity(rows);
    let mut low_slices = Vec::with_capacity(rows);
    let mut close_slices = Vec::with_capacity(rows);
    let mut volume_slices = Vec::with_capacity(rows);

    for i in 0..rows {
        let start = i * cols;
        let end = start + cols;
        high_slices.push(&highs_flat[start..end]);
        low_slices.push(&lows_flat[start..end]);
        close_slices.push(&closes_flat[start..end]);
        volume_slices.push(&volumes_flat[start..end]);
    }

    let batch_input = AdBatchInput {
        highs: &high_slices,
        lows: &low_slices,
        closes: &close_slices,
        volumes: &volume_slices,
    };

    ad_batch_with_kernel(&batch_input, Kernel::Scalar)
        .map(|o| o.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ad_batch_metadata_js(rows: usize, cols: usize) -> Vec<f64> {
    // For AD, we just return the dimensions since there are no parameters
    vec![rows as f64, cols as f64]
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ad_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ad_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ad_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    volume_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
) -> Result<(), JsValue> {
    if high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || volume_ptr.is_null()
        || out_ptr.is_null()
    {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let volume = std::slice::from_raw_parts(volume_ptr, len);

        let input = AdInput::from_slices(high, low, close, volume, AdParams::default());

        // Check for aliasing - if any input pointer equals output pointer
        if high_ptr as *const f64 == out_ptr
            || low_ptr as *const f64 == out_ptr
            || close_ptr as *const f64 == out_ptr
            || volume_ptr as *const f64 == out_ptr
        {
            // Handle aliasing case with temp buffer
            let mut temp = vec![0.0; len];
            ad_into_slice(&mut temp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            // No aliasing - compute directly into output
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            ad_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct AdBatchJsOutput {
    pub values: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "ad_batch")]
pub fn ad_batch_unified_js(
    highs_flat: &[f64],
    lows_flat: &[f64],
    closes_flat: &[f64],
    volumes_flat: &[f64],
    rows: usize,
) -> Result<JsValue, JsValue> {
    if rows == 0 {
        return Err(JsValue::from_str("rows must be > 0"));
    }
    if highs_flat.is_empty() {
        return Err(JsValue::from_str("empty inputs"));
    }
    let cols = highs_flat.len() / rows;
    let check = rows * cols;
    if lows_flat.len() != check || closes_flat.len() != check || volumes_flat.len() != check {
        return Err(JsValue::from_str(
            "Input arrays must have rows*cols elements",
        ));
    }

    // Build row slices (zero-copy views over caller's memory)
    let mut highs = Vec::with_capacity(rows);
    let mut lows = Vec::with_capacity(rows);
    let mut closes = Vec::with_capacity(rows);
    let mut volumes = Vec::with_capacity(rows);
    for r in 0..rows {
        let s = r * cols;
        let e = s + cols;
        highs.push(&highs_flat[s..e]);
        lows.push(&lows_flat[s..e]);
        closes.push(&closes_flat[s..e]);
        volumes.push(&volumes_flat[s..e]);
    }

    let batch = AdBatchInput {
        highs: &highs,
        lows: &lows,
        closes: &closes,
        volumes: &volumes,
    };
    let out = ad_batch_with_kernel(&batch, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let packed = AdBatchJsOutput {
        values: out.values,
        rows: out.rows,
        cols: out.cols,
    };
    serde_wasm_bindgen::to_value(&packed)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ad_batch_into(
    highs_ptr: *const f64,
    lows_ptr: *const f64,
    closes_ptr: *const f64,
    volumes_ptr: *const f64,
    out_ptr: *mut f64,
    rows: usize,
    cols: usize,
) -> Result<(), JsValue> {
    if highs_ptr.is_null()
        || lows_ptr.is_null()
        || closes_ptr.is_null()
        || volumes_ptr.is_null()
        || out_ptr.is_null()
    {
        return Err(JsValue::from_str("null pointer"));
    }
    unsafe {
        let check = rows
            .checked_mul(cols)
            .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;
        let highs_flat = std::slice::from_raw_parts(highs_ptr, check);
        let lows_flat = std::slice::from_raw_parts(lows_ptr, check);
        let closes_flat = std::slice::from_raw_parts(closes_ptr, check);
        let volumes_flat = std::slice::from_raw_parts(volumes_ptr, check);
        let out = std::slice::from_raw_parts_mut(out_ptr, check);

        let mut highs = Vec::with_capacity(rows);
        let mut lows = Vec::with_capacity(rows);
        let mut closes = Vec::with_capacity(rows);
        let mut volumes = Vec::with_capacity(rows);
        for r in 0..rows {
            let s = r * cols;
            let e = s + cols;
            highs.push(&highs_flat[s..e]);
            lows.push(&lows_flat[s..e]);
            closes.push(&closes_flat[s..e]);
            volumes.push(&volumes_flat[s..e]);
        }
        let batch = AdBatchInput {
            highs: &highs,
            lows: &lows,
            closes: &closes,
            volumes: &volumes,
        };

        // Compute directly into caller buffer
        ad_batch_inner_into(&batch, detect_best_batch_kernel(), false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::utilities::enums::Kernel;

    fn check_ad_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = AdParams::default();
        let input = AdInput::from_candles(&candles, default_params);
        let output = ad_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_ad_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = AdInput::with_default_candles(&candles);
        let ad_result = ad_with_kernel(&input, kernel)?;
        assert_eq!(ad_result.values.len(), candles.close.len());
        let expected_last_five = [1645918.16, 1645876.11, 1645824.27, 1645828.87, 1645728.78];
        let start = ad_result.values.len() - 5;
        let actual = &ad_result.values[start..];
        for (i, &val) in actual.iter().enumerate() {
            assert!(
                (val - expected_last_five[i]).abs() < 1e-1,
                "[{}] AD mismatch at idx {}: got {}, expected {}",
                test_name,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_ad_with_slice_data_reinput(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_input = AdInput::with_default_candles(&candles);
        let first_result = ad_with_kernel(&first_input, kernel)?;
        let second_input = AdInput::from_slices(
            &first_result.values,
            &first_result.values,
            &first_result.values,
            &first_result.values,
            AdParams::default(),
        );
        let second_result = ad_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        for i in 50..second_result.values.len() {
            assert!(!second_result.values[i].is_nan());
        }
        Ok(())
    }

    fn check_ad_input_with_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = AdInput::with_default_candles(&candles);
        match input.data {
            AdData::Candles { .. } => {}
            _ => panic!("Expected AdData::Candles variant"),
        }
        Ok(())
    }

    fn check_ad_accuracy_nan_check(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = AdInput::with_default_candles(&candles);
        let ad_result = ad_with_kernel(&input, kernel)?;
        assert_eq!(ad_result.values.len(), candles.close.len());
        if ad_result.values.len() > 50 {
            for i in 50..ad_result.values.len() {
                assert!(
                    !ad_result.values[i].is_nan(),
                    "[{}] Expected no NaN after index 50, but found NaN at index {}",
                    test_name,
                    i
                );
            }
        }
        Ok(())
    }

    fn check_ad_streaming(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = AdInput::with_default_candles(&candles);
        let batch = ad_with_kernel(&input, kernel)?.values;
        let mut stream = AdStream::try_new()?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for i in 0..candles.close.len() {
            let val = stream.update(
                candles.high[i],
                candles.low[i],
                candles.close[i],
                candles.volume[i],
            );
            stream_values.push(val);
        }
        assert_eq!(batch.len(), stream_values.len());
        for (b, s) in batch.iter().zip(stream_values.iter()) {
            if b.is_nan() && s.is_nan() {
                continue;
            }
            assert!(
                (b - s).abs() < 1e-9,
                "[{}] AD streaming mismatch",
                test_name
            );
        }
        Ok(())
    }

    // Check for poison values in single output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_ad_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Since AD doesn't have parameters, we test with default input
        // but we can test with different data slices to increase coverage
        let input = AdInput::with_default_candles(&candles);
        let output = ad_with_kernel(&input, kernel)?;

        // Check every value for poison patterns
        for (i, &val) in output.values.iter().enumerate() {
            // Skip NaN values as they're expected in some cases
            if val.is_nan() {
                continue;
            }

            let bits = val.to_bits();

            // Check for alloc_with_nan_prefix poison (0x11111111_11111111)
            if bits == 0x11111111_11111111 {
                panic!(
                    "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {}",
                    test_name, val, bits, i
                );
            }

            // Check for init_matrix_prefixes poison (0x22222222_22222222)
            if bits == 0x22222222_22222222 {
                panic!(
                    "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {}",
                    test_name, val, bits, i
                );
            }

            // Check for make_uninit_matrix poison (0x33333333_33333333)
            if bits == 0x33333333_33333333 {
                panic!(
                    "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {}",
                    test_name, val, bits, i
                );
            }
        }

        // Also test with slice data to increase coverage
        let slice_input = AdInput::from_slices(
            &candles.high,
            &candles.low,
            &candles.close,
            &candles.volume,
            AdParams::default(),
        );
        let slice_output = ad_with_kernel(&slice_input, kernel)?;

        for (i, &val) in slice_output.values.iter().enumerate() {
            if val.is_nan() {
                continue;
            }

            let bits = val.to_bits();

            if bits == 0x11111111_11111111 {
                panic!(
					"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} (slice test)",
					test_name, val, bits, i
				);
            }

            if bits == 0x22222222_22222222 {
                panic!(
					"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} (slice test)",
					test_name, val, bits, i
				);
            }

            if bits == 0x33333333_33333333 {
                panic!(
					"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} (slice test)",
					test_name, val, bits, i
				);
            }
        }

        Ok(())
    }

    // Release mode stub - does nothing
    #[cfg(not(debug_assertions))]
    fn check_ad_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    macro_rules! generate_all_ad_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $(#[test] fn [<$test_fn _scalar_f64>]() {
                    let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar);
                })*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $(#[test] fn [<$test_fn _avx2_f64>]() {
                    let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2);
                })*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $(#[test] fn [<$test_fn _avx512_f64>]() {
                    let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512);
                })*
            }
        }
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_ad_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Generate test data with appropriate ranges for AD (OHLCV data)
        // Data length: 10-400 to test various sizes
        let strat = (10usize..400).prop_flat_map(|len| {
            // Generate OHLCV data with proper constraints
            prop::collection::vec(
                // Generate (low, high_delta, close_ratio, volume) tuples
                (
                    1.0f64..1000.0f64,
                    0.0f64..500.0f64,
                    0.0f64..1.0f64,
                    0.0f64..1e6f64,
                )
                    .prop_filter("finite values", |(l, hd, cr, v)| {
                        l.is_finite()
                            && hd.is_finite()
                            && cr.is_finite()
                            && v.is_finite()
                            && *v >= 0.0
                    })
                    .prop_map(|(low, high_delta, close_ratio, volume)| {
                        let high = low + high_delta;
                        let close = if high_delta == 0.0 {
                            low
                        } else {
                            low + high_delta * close_ratio
                        };
                        (high, low, close, volume)
                    }),
                len,
            )
            .prop_map(|data| {
                let (highs, lows, closes, volumes): (Vec<_>, Vec<_>, Vec<_>, Vec<_>) =
                    data.into_iter().map(|(h, l, c, v)| (h, l, c, v)).unzip4();
                (highs, lows, closes, volumes)
            })
        });

        // Helper trait for unzip4
        trait Unzip4<A, B, C, D> {
            fn unzip4(self) -> (Vec<A>, Vec<B>, Vec<C>, Vec<D>);
        }

        impl<I, A, B, C, D> Unzip4<A, B, C, D> for I
        where
            I: Iterator<Item = (A, B, C, D)>,
        {
            fn unzip4(self) -> (Vec<A>, Vec<B>, Vec<C>, Vec<D>) {
                let (mut a, mut b, mut c, mut d) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
                for (av, bv, cv, dv) in self {
                    a.push(av);
                    b.push(bv);
                    c.push(cv);
                    d.push(dv);
                }
                (a, b, c, d)
            }
        }

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(highs, lows, closes, volumes)| {
                let input =
                    AdInput::from_slices(&highs, &lows, &closes, &volumes, AdParams::default());

                // Get output from the kernel being tested
                let AdOutput { values: out } = ad_with_kernel(&input, kernel).unwrap();

                // Get reference output from scalar kernel
                let AdOutput { values: ref_out } = ad_with_kernel(&input, Kernel::Scalar).unwrap();

                // Property 1: Output length should match input length
                prop_assert_eq!(out.len(), highs.len(), "Output length mismatch");

                // Property 2: No NaN values (AD has no warmup period)
                for (i, &val) in out.iter().enumerate() {
                    prop_assert!(
                        !val.is_nan(),
                        "Unexpected NaN at index {}: AD should not have NaN values",
                        i
                    );
                }

                // Property 3: SIMD kernel consistency
                for i in 0..out.len() {
                    let y = out[i];
                    let r = ref_out[i];

                    let y_bits = y.to_bits();
                    let r_bits = r.to_bits();

                    // Check for exact bit equality for special values
                    if !y.is_finite() || !r.is_finite() {
                        prop_assert_eq!(
                            y_bits,
                            r_bits,
                            "Special value mismatch at idx {}: {} vs {}",
                            i,
                            y,
                            r
                        );
                    } else {
                        // For finite values, allow small tolerance
                        let ulp_diff: u64 = y_bits.abs_diff(r_bits);
                        prop_assert!(
                            (y - r).abs() <= 1e-9 || ulp_diff <= 4,
                            "Value mismatch at idx {}: {} vs {} (ULP={})",
                            i,
                            y,
                            r,
                            ulp_diff
                        );
                    }
                }

                // Property 4: Zero volume handling
                // When volume is 0, AD should remain unchanged from previous value
                for i in 1..volumes.len() {
                    if volumes[i] == 0.0 {
                        prop_assert!(
                            (out[i] - out[i - 1]).abs() < 1e-10,
                            "AD should not change when volume is 0 at index {}",
                            i
                        );
                    }
                }

                // Property 5: High = Low edge case
                // When high equals low, MFM calculation should handle division by zero gracefully
                for i in 0..highs.len() {
                    if (highs[i] - lows[i]).abs() < 1e-10 {
                        if i == 0 {
                            prop_assert!(
                                out[i].abs() < 1e-10,
                                "When high=low, first AD value should be 0, got {}",
                                out[i]
                            );
                        } else {
                            prop_assert!(
                                (out[i] - out[i - 1]).abs() < 1e-10,
                                "When high=low at index {}, AD should remain unchanged",
                                i
                            );
                        }
                    }
                }

                // Property 6: Cumulative property
                // AD is cumulative, so we can verify by recalculating
                let mut expected_ad = 0.0;
                for i in 0..highs.len() {
                    let hl = highs[i] - lows[i];
                    if hl != 0.0 {
                        let mfm = ((closes[i] - lows[i]) - (highs[i] - closes[i])) / hl;
                        let mfv = mfm * volumes[i];
                        expected_ad += mfv;
                    }
                    prop_assert!(
                        (out[i] - expected_ad).abs() < 1e-9,
                        "Cumulative property violation at index {}: expected {}, got {}",
                        i,
                        expected_ad,
                        out[i]
                    );
                }

                // Property 7: First value calculation
                // The first AD value should equal the first MFV
                if !highs.is_empty() {
                    let hl = highs[0] - lows[0];
                    let expected_first = if hl != 0.0 {
                        ((closes[0] - lows[0]) - (highs[0] - closes[0])) / hl * volumes[0]
                    } else {
                        0.0
                    };
                    prop_assert!(
                        (out[0] - expected_first).abs() < 1e-10,
                        "First value mismatch: expected {}, got {}",
                        expected_first,
                        out[0]
                    );
                }

                // Property 8: Price relationship constraints
                // Verify that input data maintains low <= close <= high
                for i in 0..highs.len() {
                    prop_assert!(
                        lows[i] <= closes[i] + 1e-10 && closes[i] <= highs[i] + 1e-10,
                        "Price constraint violation at index {}: low={}, close={}, high={}",
                        i,
                        lows[i],
                        closes[i],
                        highs[i]
                    );
                }

                // Property 9: Special case - all equal prices
                // If high = low = close for all data points, AD should remain at 0
                let all_equal = highs
                    .iter()
                    .zip(lows.iter())
                    .zip(closes.iter())
                    .all(|((&h, &l), &c)| (h - l).abs() < 1e-10 && (l - c).abs() < 1e-10);

                if all_equal {
                    for (i, &val) in out.iter().enumerate() {
                        prop_assert!(
                            val.abs() < 1e-10,
                            "When all prices are equal, AD should be 0 at index {}, got {}",
                            i,
                            val
                        );
                    }
                }

                Ok(())
            })
            .unwrap();

        Ok(())
    }

    generate_all_ad_tests!(
        check_ad_partial_params,
        check_ad_accuracy,
        check_ad_input_with_default_candles,
        check_ad_with_slice_data_reinput,
        check_ad_accuracy_nan_check,
        check_ad_streaming,
        check_ad_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_ad_tests!(check_ad_property);

    fn check_batch_single_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        // Prepare slices
        let highs: Vec<&[f64]> = vec![&candles.high];
        let lows: Vec<&[f64]> = vec![&candles.low];
        let closes: Vec<&[f64]> = vec![&candles.close];
        let volumes: Vec<&[f64]> = vec![&candles.volume];

        // Individual calculation
        let single = ad_with_kernel(
            &AdInput::from_candles(&candles, AdParams::default()),
            kernel,
        )?
        .values;

        // Batch calculation
        let batch = AdBatchBuilder::new()
            .kernel(kernel)
            .apply_slices(&highs, &lows, &closes, &volumes)?;

        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, candles.close.len());
        assert_eq!(batch.values.len(), candles.close.len());

        for (i, (a, b)) in single.iter().zip(&batch.values).enumerate() {
            assert!(
                (a - b).abs() < 1e-8,
                "[{}] AD batch single row mismatch at {}: {} vs {}",
                test,
                i,
                a,
                b
            );
        }
        Ok(())
    }

    fn check_batch_multi_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        // Multi-row: repeat candle data 3 times as separate rows (for test)
        let highs: Vec<&[f64]> = vec![&candles.high, &candles.high, &candles.high];
        let lows: Vec<&[f64]> = vec![&candles.low, &candles.low, &candles.low];
        let closes: Vec<&[f64]> = vec![&candles.close, &candles.close, &candles.close];
        let volumes: Vec<&[f64]> = vec![&candles.volume, &candles.volume, &candles.volume];

        // Individual calculation (should match every row)
        let single = ad_with_kernel(
            &AdInput::from_candles(&candles, AdParams::default()),
            kernel,
        )?
        .values;

        let batch = AdBatchBuilder::new()
            .kernel(kernel)
            .apply_slices(&highs, &lows, &closes, &volumes)?;

        assert_eq!(batch.rows, 3);
        assert_eq!(batch.cols, candles.close.len());
        assert_eq!(batch.values.len(), 3 * candles.close.len());

        for row in 0..3 {
            let row_slice = &batch.values[row * batch.cols..(row + 1) * batch.cols];
            for (i, (a, b)) in single.iter().zip(row_slice.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-8,
                    "[{}] AD batch multi row mismatch row {} idx {}: {} vs {}",
                    test,
                    row,
                    i,
                    a,
                    b
                );
            }
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

    // Check for poison values in batch output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Since AD doesn't have parameters, we test with different data configurations
        // Create multiple rows with slight variations to increase test coverage
        let mut highs: Vec<&[f64]> = vec![];
        let mut lows: Vec<&[f64]> = vec![];
        let mut closes: Vec<&[f64]> = vec![];
        let mut volumes: Vec<&[f64]> = vec![];

        // Test with original data
        highs.push(&c.high);
        lows.push(&c.low);
        closes.push(&c.close);
        volumes.push(&c.volume);

        // Test with reversed data (to create different patterns)
        let high_rev: Vec<f64> = c.high.iter().rev().copied().collect();
        let low_rev: Vec<f64> = c.low.iter().rev().copied().collect();
        let close_rev: Vec<f64> = c.close.iter().rev().copied().collect();
        let volume_rev: Vec<f64> = c.volume.iter().rev().copied().collect();

        highs.push(&high_rev);
        lows.push(&low_rev);
        closes.push(&close_rev);
        volumes.push(&volume_rev);

        // Test with shifted data
        if c.high.len() > 100 {
            highs.push(&c.high[50..]);
            lows.push(&c.low[50..]);
            closes.push(&c.close[50..]);
            volumes.push(&c.volume[50..]);
        }

        let batch = AdBatchBuilder::new()
            .kernel(kernel)
            .apply_slices(&highs, &lows, &closes, &volumes)?;

        // Check every value in the entire batch matrix for poison patterns
        for (idx, &val) in batch.values.iter().enumerate() {
            // Skip NaN values as they're expected in some cases
            if val.is_nan() {
                continue;
            }

            let bits = val.to_bits();
            let row = idx / batch.cols;
            let col = idx % batch.cols;

            // Check for alloc_with_nan_prefix poison (0x11111111_11111111)
            if bits == 0x11111111_11111111 {
                panic!(
					"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (flat index {})",
					test, val, bits, row, col, idx
				);
            }

            // Check for init_matrix_prefixes poison (0x22222222_22222222)
            if bits == 0x22222222_22222222 {
                panic!(
					"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (flat index {})",
					test, val, bits, row, col, idx
				);
            }

            // Check for make_uninit_matrix poison (0x33333333_33333333)
            if bits == 0x33333333_33333333 {
                panic!(
					"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (flat index {})",
					test, val, bits, row, col, idx
				);
            }
        }

        Ok(())
    }

    // Release mode stub - does nothing
    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(
        _test: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    gen_batch_tests!(check_batch_single_row);
    gen_batch_tests!(check_batch_multi_row);
    gen_batch_tests!(check_batch_no_poison);
}
