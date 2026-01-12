use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::cuda_available;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::moving_averages::DeviceArrayF32;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::CudaEmv;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::utilities::dlpack_cuda::export_f32_cuda_dlpack_2d;
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::context::Context;
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::memory::DeviceBuffer;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;
#[cfg(all(feature = "python", feature = "cuda"))]
use std::sync::Arc;

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use serde::{Deserialize, Serialize};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen::prelude::*;

#[derive(Debug, Clone)]
pub enum EmvData<'a> {
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

#[derive(Debug, Clone)]
pub struct EmvOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, Default)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct EmvParams;

#[derive(Debug, Clone)]
pub struct EmvInput<'a> {
    pub data: EmvData<'a>,
    pub params: EmvParams,
}

impl<'a> EmvInput<'a> {
    #[inline(always)]
    pub fn from_candles(candles: &'a Candles) -> Self {
        Self {
            data: EmvData::Candles { candles },
            params: EmvParams,
        }
    }

    #[inline(always)]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        volume: &'a [f64],
    ) -> Self {
        Self {
            data: EmvData::Slices {
                high,
                low,
                close,
                volume,
            },
            params: EmvParams,
        }
    }

    #[inline(always)]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles)
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct EmvBuilder {
    kernel: Kernel,
}

impl EmvBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<EmvOutput, EmvError> {
        let input = EmvInput::from_candles(c);
        emv_with_kernel(&input, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> Result<EmvOutput, EmvError> {
        let input = EmvInput::from_slices(high, low, close, volume);
        emv_with_kernel(&input, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<EmvStream, EmvError> {
        EmvStream::try_new()
    }
}

#[derive(Debug, Error)]
pub enum EmvError {
    #[error("emv: input data slice is empty")]
    EmptyInputData,
    #[error("emv: All values are NaN")]
    AllValuesNaN,
    #[error("emv: invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("emv: not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("emv: output length mismatch: expected {expected}, got {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("emv: invalid range expansion: start={start} end={end} step={step}")]
    InvalidRange {
        start: isize,
        end: isize,
        step: isize,
    },
    #[error("emv: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("emv: invalid input: {0}")]
    InvalidInput(&'static str),
}

#[inline]
pub fn emv(input: &EmvInput) -> Result<EmvOutput, EmvError> {
    emv_with_kernel(input, Kernel::Auto)
}

pub fn emv_with_kernel(input: &EmvInput, kernel: Kernel) -> Result<EmvOutput, EmvError> {
    let (high, low, _close, volume) = match &input.data {
        EmvData::Candles { candles } => {
            let high = source_type(candles, "high");
            let low = source_type(candles, "low");
            let close = source_type(candles, "close");
            let volume = source_type(candles, "volume");
            (high, low, close, volume)
        }
        EmvData::Slices {
            high,
            low,
            close,
            volume,
        } => (*high, *low, *close, *volume),
    };

    if high.is_empty() || low.is_empty() || volume.is_empty() {
        return Err(EmvError::EmptyInputData);
    }
    let len = high.len().min(low.len()).min(volume.len());
    if len == 0 {
        return Err(EmvError::EmptyInputData);
    }

    let first = (0..len).find(|&i| !(high[i].is_nan() || low[i].is_nan() || volume[i].is_nan()));
    let first = match first {
        Some(idx) => idx,
        None => return Err(EmvError::AllValuesNaN),
    };

    let has_second = (first + 1..len)
        .find(|&i| !(high[i].is_nan() || low[i].is_nan() || volume[i].is_nan()))
        .is_some();
    if !has_second {
        return Err(EmvError::NotEnoughValidData {
            needed: 2,
            valid: 1,
        });
    }

    let mut out = alloc_with_nan_prefix(len, first + 1);
    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => emv_scalar(high, low, volume, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => emv_avx2(high, low, volume, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => emv_avx512(high, low, volume, first, &mut out),
            _ => unreachable!(),
        }
    }
    Ok(EmvOutput { values: out })
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn emv_into(input: &EmvInput, out: &mut [f64]) -> Result<(), EmvError> {
    emv_into_slice(out, input, Kernel::Auto)
}

#[inline]
pub fn emv_into_slice(dst: &mut [f64], input: &EmvInput, kern: Kernel) -> Result<(), EmvError> {
    let (high, low, _close, volume) = match &input.data {
        EmvData::Candles { candles } => {
            let high = source_type(candles, "high");
            let low = source_type(candles, "low");
            let close = source_type(candles, "close");
            let volume = source_type(candles, "volume");
            (high, low, close, volume)
        }
        EmvData::Slices {
            high,
            low,
            close,
            volume,
        } => (*high, *low, *close, *volume),
    };

    if high.is_empty() || low.is_empty() || volume.is_empty() {
        return Err(EmvError::EmptyInputData);
    }
    let len = high.len().min(low.len()).min(volume.len());
    if len == 0 {
        return Err(EmvError::EmptyInputData);
    }

    if dst.len() != len {
        return Err(EmvError::OutputLengthMismatch {
            expected: len,
            got: dst.len(),
        });
    }

    let first = (0..len).find(|&i| !(high[i].is_nan() || low[i].is_nan() || volume[i].is_nan()));
    let first = match first {
        Some(idx) => idx,
        None => return Err(EmvError::AllValuesNaN),
    };

    let has_second = (first + 1..len)
        .find(|&i| !(high[i].is_nan() || low[i].is_nan() || volume[i].is_nan()))
        .is_some();
    if !has_second {
        return Err(EmvError::NotEnoughValidData {
            needed: 2,
            valid: 1,
        });
    }

    let warm = first + 1;
    let qnan = f64::from_bits(0x7ff8_0000_0000_0000);
    for v in &mut dst[..warm] {
        *v = qnan;
    }

    let chosen = match kern {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => emv_scalar(high, low, volume, first, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => emv_avx2(high, low, volume, first, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => emv_avx512(high, low, volume, first, dst),
            _ => unreachable!(),
        }
    }
    Ok(())
}

#[inline]
pub fn emv_scalar(high: &[f64], low: &[f64], volume: &[f64], first: usize, out: &mut [f64]) {
    let len = high.len().min(low.len()).min(volume.len());
    let mut last_mid = 0.5 * (high[first] + low[first]);

    unsafe {
        let h_ptr = high.as_ptr();
        let l_ptr = low.as_ptr();
        let v_ptr = volume.as_ptr();
        let o_ptr = out.as_mut_ptr();

        let mut i = first + 1;
        while i < len {
            let h = *h_ptr.add(i);
            let l = *l_ptr.add(i);
            let v = *v_ptr.add(i);

            if h.is_nan() || l.is_nan() || v.is_nan() {
                *o_ptr.add(i) = f64::NAN;
                i += 1;
                continue;
            }

            let current_mid = 0.5 * (h + l);
            let range = h - l;
            if range == 0.0 {
                *o_ptr.add(i) = f64::NAN;
                last_mid = current_mid;
                i += 1;
                continue;
            }

            let br = v / 10000.0 / range;
            let dmid = current_mid - last_mid;
            *o_ptr.add(i) = dmid / br;
            last_mid = current_mid;

            i += 1;
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn emv_avx512(high: &[f64], low: &[f64], volume: &[f64], first: usize, out: &mut [f64]) {
    emv_avx2(high, low, volume, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn emv_avx2(high: &[f64], low: &[f64], volume: &[f64], first: usize, out: &mut [f64]) {
    let len = high.len().min(low.len()).min(volume.len());
    let mut last_mid = 0.5 * (high[first] + low[first]);
    unsafe {
        let h_ptr = high.as_ptr();
        let l_ptr = low.as_ptr();
        let v_ptr = volume.as_ptr();
        let o_ptr = out.as_mut_ptr();

        let mut i = first + 1;
        while i < len {
            let h = *h_ptr.add(i);
            let l = *l_ptr.add(i);
            let v = *v_ptr.add(i);

            if !(h.is_nan() || l.is_nan() || v.is_nan()) {
                let range = h - l;
                let current_mid = 0.5 * (h + l);

                if range == 0.0 {
                    *o_ptr.add(i) = f64::NAN;
                    last_mid = current_mid;
                } else {
                    let br = (v / 10000.0) / range;
                    let dmid = current_mid - last_mid;
                    *o_ptr.add(i) = dmid / br;
                    last_mid = current_mid;
                }
            } else {
                *o_ptr.add(i) = f64::NAN;
            }

            i += 1;
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn emv_avx512_short(
    high: &[f64],
    low: &[f64],
    volume: &[f64],
    first: usize,
    out: &mut [f64],
) {
    emv_avx2(high, low, volume, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn emv_avx512_long(
    high: &[f64],
    low: &[f64],
    volume: &[f64],
    first: usize,
    out: &mut [f64],
) {
    emv_avx2(high, low, volume, first, out);
}

#[derive(Debug, Clone)]
pub struct EmvStream {
    last_mid: Option<f64>,
}

impl EmvStream {
    pub fn try_new() -> Result<Self, EmvError> {
        Ok(Self { last_mid: None })
    }

    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64, volume: f64) -> Option<f64> {
        if high.is_nan() || low.is_nan() || volume.is_nan() {
            return None;
        }
        let current_mid = 0.5 * (high + low);
        if self.last_mid.is_none() {
            self.last_mid = Some(current_mid);
            return None;
        }
        let last_mid = self.last_mid.unwrap();
        let range = high - low;
        if range == 0.0 {
            self.last_mid = Some(current_mid);
            return None;
        }
        let br = volume / 10000.0 / range;
        let out = (current_mid - last_mid) / br;
        self.last_mid = Some(current_mid);
        Some(out)
    }

    #[inline(always)]
    pub fn update_fast(&mut self, high: f64, low: f64, volume: f64) -> Option<f64> {
        if high.is_nan() || low.is_nan() || volume.is_nan() {
            return None;
        }
        let current_mid = 0.5 * (high + low);
        if self.last_mid.is_none() {
            self.last_mid = Some(current_mid);
            return None;
        }
        let last_mid = self.last_mid.unwrap();
        let range = high - low;
        if range == 0.0 {
            self.last_mid = Some(current_mid);
            return None;
        }

        let inv_v = fast_recip_f64(volume);
        let out = (current_mid - last_mid) * range * 10_000.0 * inv_v;
        self.last_mid = Some(current_mid);
        Some(out)
    }
}

#[inline(always)]
fn newton_refine_recip(y0: f64, x: f64) -> f64 {
    let t = 2.0_f64 - x.mul_add(y0, 0.0);
    y0 * t
}

#[inline(always)]
fn fast_recip_f64(x: f64) -> f64 {
    #[cfg(all(
        feature = "nightly-avx",
        target_arch = "x86_64",
        target_feature = "avx512f"
    ))]
    unsafe {
        use core::arch::x86_64::*;
        let vx = _mm512_set1_pd(x);
        let rcp = _mm512_rcp14_pd(vx);
        let lo = _mm512_castpd512_pd128(rcp);
        let y0 = _mm_cvtsd_f64(lo);
        let y1 = newton_refine_recip(y0, x);
        let y2 = newton_refine_recip(y1, x);
        return y2;
    }
    1.0 / x
}

#[derive(Clone, Debug)]
pub struct EmvBatchRange {}

impl Default for EmvBatchRange {
    fn default() -> Self {
        Self {}
    }
}

#[derive(Clone, Debug, Default)]
pub struct EmvBatchBuilder {
    kernel: Kernel,
    _range: EmvBatchRange,
}

impl EmvBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> Result<EmvBatchOutput, EmvError> {
        emv_batch_with_kernel(high, low, close, volume, self.kernel)
    }

    pub fn with_default_slices(
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
        k: Kernel,
    ) -> Result<EmvBatchOutput, EmvError> {
        EmvBatchBuilder::new()
            .kernel(k)
            .apply_slices(high, low, close, volume)
    }

    pub fn apply_candles(self, c: &Candles) -> Result<EmvBatchOutput, EmvError> {
        let high = source_type(c, "high");
        let low = source_type(c, "low");
        let close = source_type(c, "close");
        let volume = source_type(c, "volume");
        self.apply_slices(high, low, close, volume)
    }

    pub fn with_default_candles(c: &Candles, k: Kernel) -> Result<EmvBatchOutput, EmvError> {
        EmvBatchBuilder::new().kernel(k).apply_candles(c)
    }
}

pub fn emv_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    _close: &[f64],
    volume: &[f64],
    kernel: Kernel,
) -> Result<EmvBatchOutput, EmvError> {
    let simd = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(EmvError::InvalidKernelForBatch(other)),
    };
    emv_batch_par_slice(high, low, volume, simd)
}

#[derive(Clone, Debug)]
pub struct EmvBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<EmvParams>,
    pub rows: usize,
    pub cols: usize,
}

impl EmvBatchOutput {
    #[inline]
    pub fn single_row(&self) -> &[f64] {
        debug_assert_eq!(self.rows, 1);
        &self.values[..self.cols]
    }
}

#[inline(always)]
fn expand_grid(_r: &EmvBatchRange) -> Vec<()> {
    vec![()]
}

#[inline(always)]
pub fn emv_batch_slice(
    high: &[f64],
    low: &[f64],
    volume: &[f64],
    kern: Kernel,
) -> Result<EmvBatchOutput, EmvError> {
    emv_batch_inner(high, low, volume, kern, false)
}

#[inline(always)]
pub fn emv_batch_par_slice(
    high: &[f64],
    low: &[f64],
    volume: &[f64],
    kern: Kernel,
) -> Result<EmvBatchOutput, EmvError> {
    emv_batch_inner(high, low, volume, kern, true)
}

fn emv_batch_inner(
    high: &[f64],
    low: &[f64],
    volume: &[f64],
    kern: Kernel,
    _parallel: bool,
) -> Result<EmvBatchOutput, EmvError> {
    let len = high.len().min(low.len()).min(volume.len());
    if len == 0 {
        return Err(EmvError::EmptyInputData);
    }

    let first = (0..len)
        .find(|&i| !(high[i].is_nan() || low[i].is_nan() || volume[i].is_nan()))
        .ok_or(EmvError::AllValuesNaN)?;

    let valid = (first..len)
        .filter(|&i| !(high[i].is_nan() || low[i].is_nan() || volume[i].is_nan()))
        .count();
    if valid < 2 {
        return Err(EmvError::NotEnoughValidData { needed: 2, valid });
    }

    let rows = 1usize;
    let cols = len;
    let _ = rows
        .checked_mul(cols)
        .ok_or(EmvError::InvalidInput("rows*cols overflow"))?;

    let mut buf_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut buf_mu, cols, &[first + 1]);

    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    unsafe {
        match kern {
            Kernel::Scalar | Kernel::ScalarBatch => emv_scalar(high, low, volume, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => emv_avx2(high, low, volume, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => emv_avx512(high, low, volume, first, out),
            _ => emv_scalar(high, low, volume, first, out),
        }
    }

    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(EmvBatchOutput {
        values,
        combos: vec![EmvParams],
        rows,
        cols,
    })
}

#[inline(always)]
pub fn emv_row_scalar(
    high: &[f64],
    low: &[f64],
    volume: &[f64],
    first: usize,
    _stride: usize,
    _w_ptr: *const f64,
    _inv_n: f64,
    out: &mut [f64],
) {
    emv_scalar(high, low, volume, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn emv_row_avx2(
    high: &[f64],
    low: &[f64],
    volume: &[f64],
    first: usize,
    _stride: usize,
    _w_ptr: *const f64,
    _inv_n: f64,
    out: &mut [f64],
) {
    emv_scalar(high, low, volume, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn emv_row_avx512(
    high: &[f64],
    low: &[f64],
    volume: &[f64],
    first: usize,
    _stride: usize,
    _w_ptr: *const f64,
    _inv_n: f64,
    out: &mut [f64],
) {
    emv_avx512(high, low, volume, first, out);
}

#[inline(always)]
fn expand_grid_emv(_r: &EmvBatchRange) -> Vec<()> {
    vec![()]
}

#[cfg(feature = "python")]
#[pyfunction(name = "emv")]
#[pyo3(signature = (high, low, close, volume, kernel=None))]
pub fn emv_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;
    let volume_slice = volume.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let data = EmvData::Slices {
        high: high_slice,
        low: low_slice,
        close: close_slice,
        volume: volume_slice,
    };
    let input = EmvInput {
        data,
        params: EmvParams,
    };

    let result_vec: Vec<f64> = py
        .allow_threads(|| emv_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "EmvStream")]
pub struct EmvStreamPy {
    stream: EmvStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl EmvStreamPy {
    #[new]
    fn new() -> PyResult<Self> {
        let stream = EmvStream::try_new().map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(EmvStreamPy { stream })
    }

    fn update(&mut self, high: f64, low: f64, close: f64, volume: f64) -> Option<f64> {
        self.stream.update(high, low, volume)
    }
}

#[cfg(feature = "python")]
fn emv_batch_inner_into(
    high: &[f64],
    low: &[f64],
    _close: &[f64],
    volume: &[f64],
    _range: &EmvBatchRange,
    kern: Kernel,
    _parallel: bool,
    out: &mut [f64],
) -> Result<Vec<EmvParams>, EmvError> {
    let len = high.len().min(low.len()).min(volume.len());
    if len == 0 {
        return Err(EmvError::EmptyInputData);
    }

    if out.len() != len {
        return Err(EmvError::OutputLengthMismatch {
            expected: len,
            got: out.len(),
        });
    }

    let out_mu: &mut [MaybeUninit<f64>] = unsafe {
        core::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };

    let first = (0..len)
        .find(|&i| !(high[i].is_nan() || low[i].is_nan() || volume[i].is_nan()))
        .ok_or(EmvError::AllValuesNaN)?;

    let valid = (first..len)
        .filter(|&i| !(high[i].is_nan() || low[i].is_nan() || volume[i].is_nan()))
        .count();
    if valid < 2 {
        return Err(EmvError::NotEnoughValidData { needed: 2, valid });
    }

    init_matrix_prefixes(out_mu, len, &[first + 1]);

    let out_f: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(out_mu.as_mut_ptr() as *mut f64, out_mu.len()) };

    unsafe {
        match kern {
            Kernel::Scalar | Kernel::ScalarBatch => emv_scalar(high, low, volume, first, out_f),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => emv_avx2(high, low, volume, first, out_f),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => emv_avx512(high, low, volume, first, out_f),
            _ => emv_scalar(high, low, volume, first, out_f),
        }
    }

    Ok(vec![EmvParams])
}

#[cfg(feature = "python")]
#[pyfunction(name = "emv_batch")]
#[pyo3(signature = (high, low, close, volume, kernel=None))]
pub fn emv_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};

    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;
    let volume_slice = volume.as_slice()?;
    let kern = validate_kernel(kernel, true)?;

    let sweep = EmvBatchRange {};
    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = high_slice
        .len()
        .min(low_slice.len())
        .min(volume_slice.len());

    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    let _params = py
        .allow_threads(|| {
            let kernel = match kern {
                Kernel::Auto => detect_best_batch_kernel(),
                k => k,
            };
            emv_batch_inner_into(
                high_slice,
                low_slice,
                close_slice,
                volume_slice,
                &sweep,
                kernel,
                true,
                slice_out,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;

    Ok(dict)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn emv_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
) -> Result<Vec<f64>, JsValue> {
    let input = EmvInput::from_slices(high, low, close, volume);

    let mut output = vec![0.0; high.len().min(low.len()).min(close.len()).min(volume.len())];

    emv_into_slice(&mut output, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn emv_into(
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
        return Err(JsValue::from_str("null pointer passed to emv_into"));
    }

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let volume = std::slice::from_raw_parts(volume_ptr, len);

        let input = EmvInput::from_slices(high, low, close, volume);

        if out_ptr == high_ptr as *mut f64
            || out_ptr == low_ptr as *mut f64
            || out_ptr == close_ptr as *mut f64
            || out_ptr == volume_ptr as *mut f64
        {
            let mut temp = vec![0.0; len];
            emv_into_slice(&mut temp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            emv_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn emv_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn emv_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct EmvBatchConfig {}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct EmvBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<EmvParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = emv_batch)]
pub fn emv_batch_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    _config: JsValue,
) -> Result<JsValue, JsValue> {
    let input = EmvInput::from_slices(high, low, close, volume);
    let len = high.len().min(low.len()).min(close.len()).min(volume.len());

    let mut output = vec![0.0; len];

    let kernel = detect_best_kernel();

    emv_into_slice(&mut output, &input, kernel).map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = EmvBatchJsOutput {
        values: output,
        combos: vec![EmvParams],
        rows: 1,
        cols: len,
    };

    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn emv_batch_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    volume_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
) -> Result<usize, JsValue> {
    if high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || volume_ptr.is_null()
        || out_ptr.is_null()
    {
        return Err(JsValue::from_str("null pointer passed to emv_batch_into"));
    }

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let volume = std::slice::from_raw_parts(volume_ptr, len);

        let input = EmvInput::from_slices(high, low, close, volume);

        let kernel = detect_best_kernel();

        let out = std::slice::from_raw_parts_mut(out_ptr, len);
        emv_into_slice(out, &input, kernel).map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn check_emv_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = EmvInput::from_candles(&candles);
        let output = emv_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        let expected_last_five_emv = [
            -6488905.579799851,
            2371436.7401001123,
            -3855069.958128531,
            1051939.877943717,
            -8519287.22257077,
        ];
        let start = output.values.len().saturating_sub(5);
        for (i, &val) in output.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five_emv[i]).abs();
            let tol = expected_last_five_emv[i].abs() * 0.0001;
            assert!(
                diff <= tol,
                "[{}] EMV {:?} mismatch at idx {}: got {}, expected {}, diff={}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five_emv[i],
                diff
            );
        }
        Ok(())
    }

    fn check_emv_with_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = EmvInput::with_default_candles(&candles);
        let output = emv_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_emv_empty_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let empty: [f64; 0] = [];
        let input = EmvInput::from_slices(&empty, &empty, &empty, &empty);
        let result = emv_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_emv_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let nan_arr = [f64::NAN, f64::NAN];
        let input = EmvInput::from_slices(&nan_arr, &nan_arr, &nan_arr, &nan_arr);
        let result = emv_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_emv_not_enough_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10000.0, f64::NAN];
        let low = [9990.0, f64::NAN];
        let close = [9995.0, f64::NAN];
        let volume = [1_000_000.0, f64::NAN];
        let input = EmvInput::from_slices(&high, &low, &close, &volume);
        let result = emv_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_emv_basic_calculation(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 12.0, 13.0, 15.0];
        let low = [5.0, 7.0, 8.0, 10.0];
        let close = [7.5, 9.0, 10.5, 12.5];
        let volume = [10000.0, 20000.0, 25000.0, 30000.0];
        let input = EmvInput::from_slices(&high, &low, &close, &volume);
        let output = emv_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), 4);
        assert!(output.values[0].is_nan());
        for &val in &output.values[1..] {
            assert!(!val.is_nan());
        }
        Ok(())
    }

    fn check_emv_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let high = source_type(&candles, "high");
        let low = source_type(&candles, "low");
        let volume = source_type(&candles, "volume");

        let output = emv_with_kernel(&EmvInput::from_candles(&candles), kernel)?.values;

        let mut stream = EmvStream::try_new()?;
        let mut stream_values = Vec::with_capacity(high.len());
        for i in 0..high.len() {
            match stream.update(high[i], low[i], volume[i]) {
                Some(val) => stream_values.push(val),
                None => stream_values.push(f64::NAN),
            }
        }
        assert_eq!(output.len(), stream_values.len());
        for (b, s) in output.iter().zip(stream_values.iter()) {
            if b.is_nan() && s.is_nan() {
                continue;
            }
            let diff = (b - s).abs();
            assert!(
                diff < 1e-9,
                "[{}] EMV streaming f64 mismatch: batch={}, stream={}, diff={}",
                test_name,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_emv_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input1 = EmvInput::from_candles(&candles);
        let output1 = emv_with_kernel(&input1, kernel)?;

        let high = source_type(&candles, "high");
        let low = source_type(&candles, "low");
        let close = source_type(&candles, "close");
        let volume = source_type(&candles, "volume");
        let input2 = EmvInput::from_slices(high, low, close, volume);
        let output2 = emv_with_kernel(&input2, kernel)?;

        let input3 = EmvInput::with_default_candles(&candles);
        let output3 = emv_with_kernel(&input3, kernel)?;

        let outputs = [
            ("from_candles", &output1.values),
            ("from_slices", &output2.values),
            ("with_default_candles", &output3.values),
        ];

        for (method_name, values) in &outputs {
            for (i, &val) in values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 using method: {}",
                        test_name, val, bits, i, method_name
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 using method: {}",
                        test_name, val, bits, i, method_name
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 using method: {}",
                        test_name, val, bits, i, method_name
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_emv_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_emv_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let strat = prop::collection::vec(
            (10.0f64..100000.0f64, 0.5f64..0.999f64, 1000.0f64..1e9f64),
            2..400,
        )
        .prop_map(|data| {
            let high: Vec<f64> = data.iter().map(|(h, _, _)| *h).collect();
            let low: Vec<f64> = data
                .iter()
                .zip(&high)
                .map(|((_, l_pct, _), h)| h * l_pct)
                .collect();
            let volume: Vec<f64> = data.iter().map(|(_, _, v)| *v).collect();

            let close = high.clone();
            (high, low, close, volume)
        });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(high, low, close, volume)| {
                let input = EmvInput::from_slices(&high, &low, &close, &volume);

                let EmvOutput { values: out } = emv_with_kernel(&input, kernel).unwrap();

                let EmvOutput { values: ref_out } =
                    emv_with_kernel(&input, Kernel::Scalar).unwrap();

                prop_assert!(
                    out[0].is_nan(),
                    "First EMV value should always be NaN (warmup period)"
                );

                for i in 1..out.len() {
                    if high[i].is_finite() && low[i].is_finite() && volume[i].is_finite() {
                        let range = high[i] - low[i];
                        if range != 0.0 {
                            prop_assert!(
								out[i].is_finite(),
								"EMV at index {} should be finite when inputs are finite and range != 0",
								i
							);
                        }
                    }
                }

                for i in 0..out.len() {
                    let y = out[i];
                    let r = ref_out[i];

                    if !y.is_finite() || !r.is_finite() {
                        prop_assert!(
                            y.to_bits() == r.to_bits(),
                            "Non-finite mismatch at index {}: {} vs {}",
                            i,
                            y,
                            r
                        );
                    } else {
                        let y_bits = y.to_bits();
                        let r_bits = r.to_bits();
                        let ulp_diff = y_bits.abs_diff(r_bits);

                        prop_assert!(
                            ulp_diff <= 3,
                            "ULP difference too large at index {}: {} vs {} (ULP={})",
                            i,
                            y,
                            r,
                            ulp_diff
                        );
                    }
                }

                let mut last_mid = 0.5 * (high[0] + low[0]);
                for i in 1..out.len() {
                    let current_mid = 0.5 * (high[i] + low[i]);
                    let range = high[i] - low[i];

                    if range == 0.0 {
                        prop_assert!(
                            out[i].is_nan(),
                            "EMV at index {} should be NaN when range is zero",
                            i
                        );
                    } else {
                        let expected_emv = (current_mid - last_mid) / (volume[i] / 10000.0 / range);

                        if out[i].is_finite() && expected_emv.is_finite() {
                            let diff = (out[i] - expected_emv).abs();
                            let tolerance = 1e-9;
                            prop_assert!(
                                diff <= tolerance,
                                "EMV formula mismatch at index {}: got {}, expected {}, diff={}",
                                i,
                                out[i],
                                expected_emv,
                                diff
                            );
                        }
                    }

                    last_mid = current_mid;
                }

                for i in 1..out.len() {
                    if out[i].is_finite() {
                        let price_change =
                            (high[i] + low[i]) / 2.0 - (high[i - 1] + low[i - 1]) / 2.0;
                        let max_reasonable = price_change.abs() * 1e8;

                        prop_assert!(
                            out[i].abs() <= max_reasonable,
                            "EMV at index {} seems unreasonably large: {} (price change: {})",
                            i,
                            out[i],
                            price_change
                        );
                    }
                }

                if high.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10)
                    && low.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10)
                    && high.iter().zip(&low).all(|(h, l)| h > l)
                {
                    for i in 1..out.len() {
                        if out[i].is_finite() {
                            prop_assert!(
                                out[i].abs() < 1e-9,
                                "EMV should be ~0 for constant prices, got {} at index {}",
                                out[i],
                                i
                            );
                        }
                    }
                }

                for (i, &val) in out.iter().enumerate() {
                    if !val.is_nan() {
                        let bits = val.to_bits();
                        prop_assert!(
                            bits != 0x11111111_11111111
                                && bits != 0x22222222_22222222
                                && bits != 0x33333333_33333333,
                            "Found poison value at index {}: {} (0x{:016X})",
                            i,
                            val,
                            bits
                        );
                    }
                }

                Ok(())
            })
            .unwrap();

        Ok(())
    }

    macro_rules! generate_all_emv_tests {
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

    generate_all_emv_tests!(
        check_emv_accuracy,
        check_emv_with_default_candles,
        check_emv_empty_data,
        check_emv_all_nan,
        check_emv_not_enough_data,
        check_emv_basic_calculation,
        check_emv_streaming,
        check_emv_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_emv_tests!(check_emv_property);

    fn check_batch_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = EmvBatchBuilder::new().kernel(kernel).apply_candles(&c)?;
        assert_eq!(output.values.len(), c.close.len());
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
    gen_batch_tests!(check_batch_row);
    gen_batch_tests!(check_batch_no_poison);

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = EmvBatchBuilder::new().kernel(kernel).apply_candles(&c)?;

        for (idx, &val) in output.values.iter().enumerate() {
            if val.is_nan() {
                continue;
            }

            let bits = val.to_bits();
            let row = idx / output.cols;
            let col = idx % output.cols;

            if bits == 0x11111111_11111111 {
                panic!(
                    "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
					 at row {} col {} (flat index {})",
                    test, val, bits, row, col, idx
                );
            }

            if bits == 0x22222222_22222222 {
                panic!(
                    "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) \
					 at row {} col {} (flat index {})",
                    test, val, bits, row, col, idx
                );
            }

            if bits == 0x33333333_33333333 {
                panic!(
                    "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) \
					 at row {} col {} (flat index {})",
                    test, val, bits, row, col, idx
                );
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

    #[test]
    fn test_emv_into_matches_api() -> Result<(), Box<dyn Error>> {
        let n = 256usize;
        let mut high = Vec::with_capacity(n);
        let mut low = Vec::with_capacity(n);
        let mut close = Vec::with_capacity(n);
        let mut volume = Vec::with_capacity(n);
        for i in 0..n {
            let base = 100.0 + (i as f64) * 0.1;
            let spread = 1.0 + ((i % 5) as f64) * 0.2;
            let h = base + spread * 0.6;
            let l = base - spread * 0.4;
            high.push(h);
            low.push(l);
            close.push(0.5 * (h + l));
            volume.push(10_000.0 + ((i * 37) % 1000) as f64 * 100.0);
        }

        let input = EmvInput::from_slices(&high, &low, &close, &volume);
        let baseline = emv(&input)?.values;

        let mut into_out = vec![0.0; baseline.len()];

        #[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
        {
            emv_into(&input, &mut into_out)?;
        }
        #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
        {
            emv_into_slice(&mut into_out, &input, Kernel::Auto)?;
        }

        assert_eq!(baseline.len(), into_out.len());
        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b) || (a - b).abs() <= 1e-12
        }
        for (i, (a, b)) in baseline.iter().zip(into_out.iter()).enumerate() {
            assert!(
                eq_or_both_nan(*a, *b),
                "divergence at idx {}: api={}, into={}",
                i,
                a,
                b
            );
        }
        Ok(())
    }
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(module = "ta_indicators.cuda", name = "EmvDeviceArrayF32", unsendable)]
pub struct EmvDeviceArrayF32Py {
    pub inner: DeviceArrayF32,
    _ctx_guard: Arc<Context>,
    device_id: i32,
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pymethods]
impl EmvDeviceArrayF32Py {
    #[getter]
    fn __cuda_array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        let inner = &self.inner;
        let itemsize = std::mem::size_of::<f32>();
        d.set_item("shape", (inner.rows, inner.cols))?;
        d.set_item("typestr", "<f4")?;
        d.set_item("strides", (inner.cols * itemsize, itemsize))?;
        let ptr_val = inner.buf.as_device_ptr().as_raw() as usize;
        d.set_item("data", (ptr_val, false))?;

        d.set_item("version", 3)?;
        Ok(d)
    }

    fn __dlpack_device__(&self) -> (i32, i32) {
        (2, self.device_id)
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
                        return Err(PyValueError::new_err("dl_device mismatch for __dlpack__"));
                    }
                }
            }
        }
        let _ = stream;

        let dummy =
            DeviceBuffer::from_slice(&[]).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let inner = std::mem::replace(
            &mut self.inner,
            DeviceArrayF32 {
                buf: dummy,
                rows: 0,
                cols: 0,
            },
        );

        let rows = inner.rows;
        let cols = inner.cols;
        let buf = inner.buf;

        let max_version_bound = max_version.map(|obj| obj.into_bound(py));

        export_f32_cuda_dlpack_2d(py, buf, rows, cols, alloc_dev, max_version_bound)
    }
}

#[cfg(all(feature = "python", feature = "cuda"))]
impl EmvDeviceArrayF32Py {
    fn new_from_cuda(inner: DeviceArrayF32, ctx_guard: Arc<Context>, device_id: u32) -> Self {
        Self {
            inner,
            _ctx_guard: ctx_guard,
            device_id: device_id as i32,
        }
    }
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "emv_cuda_batch_dev")]
#[pyo3(signature = (high_f32, low_f32, volume_f32, device_id=0))]
pub fn emv_cuda_batch_dev_py<'py>(
    py: Python<'py>,
    high_f32: numpy::PyReadonlyArray1<'py, f32>,
    low_f32: numpy::PyReadonlyArray1<'py, f32>,
    volume_f32: numpy::PyReadonlyArray1<'py, f32>,
    device_id: usize,
) -> PyResult<EmvDeviceArrayF32Py> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let h = high_f32.as_slice()?;
    let l = low_f32.as_slice()?;
    let v = volume_f32.as_slice()?;
    let (inner, ctx, dev_id) = py.allow_threads(|| -> PyResult<_> {
        let cuda = CudaEmv::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx = cuda.context_arc();
        let dev_id = cuda.device_id();
        let buf = cuda
            .emv_batch_dev(h, l, v)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((buf, ctx, dev_id))
    })?;
    Ok(EmvDeviceArrayF32Py::new_from_cuda(inner, ctx, dev_id))
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "emv_cuda_many_series_one_param_dev")]
#[pyo3(signature = (high_tm_f32, low_tm_f32, volume_tm_f32, device_id=0))]
pub fn emv_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    high_tm_f32: numpy::PyReadonlyArray2<'_, f32>,
    low_tm_f32: numpy::PyReadonlyArray2<'_, f32>,
    volume_tm_f32: numpy::PyReadonlyArray2<'_, f32>,
    device_id: usize,
) -> PyResult<EmvDeviceArrayF32Py> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    use numpy::PyUntypedArrayMethods;
    let h_flat = high_tm_f32.as_slice()?;
    let l_flat = low_tm_f32.as_slice()?;
    let v_flat = volume_tm_f32.as_slice()?;
    let rows = high_tm_f32.shape()[0];
    let cols = high_tm_f32.shape()[1];
    if low_tm_f32.shape() != [rows, cols] || volume_tm_f32.shape() != [rows, cols] {
        return Err(PyValueError::new_err("high/low/volume shapes mismatch"));
    }
    let (inner, ctx, dev_id) = py.allow_threads(|| -> PyResult<_> {
        let cuda = CudaEmv::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx = cuda.context_arc();
        let dev_id = cuda.device_id();
        let buf = cuda
            .emv_many_series_one_param_time_major_dev(h_flat, l_flat, v_flat, cols, rows)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((buf, ctx, dev_id))
    })?;
    Ok(EmvDeviceArrayF32Py::new_from_cuda(inner, ctx, dev_id))
}
