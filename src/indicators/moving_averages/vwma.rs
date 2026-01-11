//! # Volume Weighted Moving Average (VWMA)
//!
//! VWMA weights each price by its volume over a moving window. This captures the price action with regard to trading activity.
//!
//! ## Parameters
//! - **period**: Number of bars to use for weighting (default 20). Must be ≥ 1 and ≤ data length.
//!
//! ## Errors
//! - **AllValuesNaN**: vwma: All price-volume pairs are NaN.
//! - **InvalidPeriod**: vwma: Period is zero or exceeds data length.
//! - **PriceVolumeMismatch**: vwma: Price and volume lengths do not match.
//! - **NotEnoughValidData**: vwma: Not enough valid price-volume pairs for the requested period.
//!
//! ## Returns
//! - **Ok(VwmaOutput)** on success, containing the VWMA as a Vec<f64>.
//! - **Err(VwmaError)** on error.
//!
//! ## Developer Notes
//! - SIMD: AVX2/AVX512 implementations exist but are disabled by default.
//!   Rationale: strict non-finite bit-equality property tests and minimal wins for short periods.
//!   Runtime selection short-circuits to scalar; SIMD kept for future experimentation.
//! - CUDA: FP32 wrappers present; kernels loaded from embedded PTX. Wrapper validates memory
//!   headroom and launch shape and returns VRAM handles usable from Python via CAI v3 + DLPack.
//! - Streaming: O(1) ring-buffer update with running sums; hot path avoids modulo.
//!   Exact IEEE math retained (no fast-math) to preserve test reference values.
//! - Memory: uses `alloc_with_nan_prefix` for zero-copy allocation (see alma.rs for patterns).

#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::cuda_available;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::moving_averages::CudaVwma;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::utilities::dlpack_cuda::export_f32_cuda_dlpack_2d;

#[cfg(all(feature = "python", feature = "cuda"))]
use cust::context::Context;
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::memory::DeviceBuffer;
#[cfg(all(feature = "python", feature = "cuda"))]
use std::sync::Arc;
#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(module = "ta_indicators.cuda", unsendable)]
pub struct VwmaDeviceArrayF32Py {
    pub(crate) buf: Option<DeviceBuffer<f32>>, 
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) _ctx: Arc<Context>,
    pub(crate) device_id: u32,
}
#[cfg(all(feature = "python", feature = "cuda"))]
#[pymethods]
impl VwmaDeviceArrayF32Py {
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
        // Producer stream is synchronized in the CUDA wrapper; omit "stream" per CAI v3.
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
    ) -> PyResult<PyObject> {
        // Compute target device id and validate `dl_device` hint if provided.
        let (kdl, alloc_dev) = self.__dlpack_device__(); // (2, device_id)
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

        // Move `DeviceBuffer` into capsule-owned manager (one-shot).
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
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};
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
use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum VwmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    CandlesPlusPrices {
        candles: &'a Candles,
        prices: &'a [f64],
    },
    Slice {
        prices: &'a [f64],
        volumes: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct VwmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(serde::Serialize, serde::Deserialize))]
pub struct VwmaParams {
    pub period: Option<usize>,
}

impl Default for VwmaParams {
    fn default() -> Self {
        Self { period: Some(20) }
    }
}

#[derive(Debug, Clone)]
pub struct VwmaInput<'a> {
    pub data: VwmaData<'a>,
    pub params: VwmaParams,
}

impl<'a> VwmaInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: VwmaParams) -> Self {
        Self {
            data: VwmaData::Candles { candles, source },
            params,
        }
    }

    pub fn from_candles_plus_prices(
        candles: &'a Candles,
        prices: &'a [f64],
        params: VwmaParams,
    ) -> Self {
        Self {
            data: VwmaData::CandlesPlusPrices { candles, prices },
            params,
        }
    }

    pub fn from_slice(prices: &'a [f64], volumes: &'a [f64], params: VwmaParams) -> Self {
        Self {
            data: VwmaData::Slice { prices, volumes },
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: VwmaData::Candles {
                candles,
                source: "close",
            },
            params: VwmaParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(20)
    }
}

impl<'a> AsRef<[f64]> for VwmaInput<'a> {
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            VwmaData::Candles { candles, source } => source_type(candles, source),
            VwmaData::CandlesPlusPrices { prices, .. } => prices,
            VwmaData::Slice { prices, .. } => prices,
        }
    }
}

#[derive(Debug, Error)]
pub enum VwmaError {
    #[error("vwma: All values are NaN.")]
    AllValuesNaN,
    #[error("vwma: empty input data")]
    EmptyInputData,
    #[error("vwma: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error(
        "vwma: Price and volume mismatch: price length = {price_len}, volume length = {volume_len}"
    )]
    PriceVolumeMismatch { price_len: usize, volume_len: usize },
    #[error("vwma: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("vwma: output length mismatch: expected {expected}, got {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("vwma: invalid range: start={start}, end={end}, step={step}")]
    InvalidRange { start: usize, end: usize, step: usize },
    #[error("vwma: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("vwma: arithmetic overflow while computing {context}")]
    ArithmeticOverflow { context: &'static str },
}

#[derive(Copy, Clone, Debug)]
pub struct VwmaBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for VwmaBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl VwmaBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn period(mut self, n: usize) -> Self {
        self.period = Some(n);
        self
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    pub fn apply(self, c: &Candles) -> Result<VwmaOutput, VwmaError> {
        let p = VwmaParams {
            period: self.period,
        };
        let i = VwmaInput::from_candles(c, "close", p);
        vwma_with_kernel(&i, self.kernel)
    }
    pub fn apply_slice(self, prices: &[f64], volumes: &[f64]) -> Result<VwmaOutput, VwmaError> {
        let p = VwmaParams {
            period: self.period,
        };
        let i = VwmaInput::from_slice(prices, volumes, p);
        vwma_with_kernel(&i, self.kernel)
    }
    pub fn into_stream(self) -> Result<VwmaStream, VwmaError> {
        let p = VwmaParams {
            period: self.period,
        };
        VwmaStream::try_new(p)
    }
}

#[inline]
pub fn vwma(input: &VwmaInput) -> Result<VwmaOutput, VwmaError> {
    vwma_with_kernel(input, Kernel::Auto)
}

pub fn vwma_with_kernel(input: &VwmaInput, kernel: Kernel) -> Result<VwmaOutput, VwmaError> {
    let (price, volume): (&[f64], &[f64]) = match &input.data {
        VwmaData::Candles { candles, source } => {
            (source_type(candles, source), source_type(candles, "volume"))
        }
        VwmaData::CandlesPlusPrices { candles, prices } => (prices, source_type(candles, "volume")),
        VwmaData::Slice { prices, volumes } => (prices, volumes),
    };
    let len = price.len();
    if len == 0 {
        return Err(VwmaError::EmptyInputData);
    }
    let period = input.get_period();

    if period == 0 || period > len {
        return Err(VwmaError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if volume.len() != len {
        return Err(VwmaError::PriceVolumeMismatch {
            price_len: len,
            volume_len: volume.len(),
        });
    }
    let first = price
        .iter()
        .zip(volume.iter())
        .position(|(&p, &v)| !p.is_nan() && !v.is_nan())
        .ok_or(VwmaError::AllValuesNaN)?;

    if (len - first) < period {
        return Err(VwmaError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let warm = first
        .checked_add(period)
        .and_then(|x| x.checked_sub(1))
        .ok_or(VwmaError::ArithmeticOverflow { context: "warmup prefix index" })?;
    let mut out = alloc_with_nan_prefix(len, warm);

    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                vwma_scalar(price, volume, period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => vwma_avx2(price, volume, period, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                vwma_avx512(price, volume, period, first, &mut out)
            }
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                vwma_scalar(price, volume, period, first, &mut out)
            }
            _ => unreachable!(),
        }
    }

    Ok(VwmaOutput { values: out })
}

#[inline]
pub fn vwma_scalar(price: &[f64], volume: &[f64], period: usize, first: usize, out: &mut [f64]) {
    let len = price.len();
    if len < period {
        return;
    }

    unsafe {
        let p_ptr = price.as_ptr();
        let v_ptr = volume.as_ptr();
        let out_ptr = out.as_mut_ptr();

        
        let base = first;
        let mut sum = 0.0f64;
        let mut vsum = 0.0f64;
        for i in 0..period {
            let p = *p_ptr.add(base + i);
            let v = *v_ptr.add(base + i);
            sum += p * v;
            vsum += v;
        }

        *out_ptr.add(base + period - 1) = sum / vsum;

        
        let mut new_idx = base + period;
        let mut old_idx = base;
        while new_idx + 3 < len {
            
            let pn0 = *p_ptr.add(new_idx);
            let vn0 = *v_ptr.add(new_idx);
            let po0 = *p_ptr.add(old_idx);
            let vo0 = *v_ptr.add(old_idx);
            sum += pn0 * vn0;
            sum -= po0 * vo0;
            vsum += vn0 - vo0;
            *out_ptr.add(new_idx) = sum / vsum;

            
            let pn1 = *p_ptr.add(new_idx + 1);
            let vn1 = *v_ptr.add(new_idx + 1);
            let po1 = *p_ptr.add(old_idx + 1);
            let vo1 = *v_ptr.add(old_idx + 1);
            sum += pn1 * vn1;
            sum -= po1 * vo1;
            vsum += vn1 - vo1;
            *out_ptr.add(new_idx + 1) = sum / vsum;

            
            let pn2 = *p_ptr.add(new_idx + 2);
            let vn2 = *v_ptr.add(new_idx + 2);
            let po2 = *p_ptr.add(old_idx + 2);
            let vo2 = *v_ptr.add(old_idx + 2);
            sum += pn2 * vn2;
            sum -= po2 * vo2;
            vsum += vn2 - vo2;
            *out_ptr.add(new_idx + 2) = sum / vsum;

            
            let pn3 = *p_ptr.add(new_idx + 3);
            let vn3 = *v_ptr.add(new_idx + 3);
            let po3 = *p_ptr.add(old_idx + 3);
            let vo3 = *v_ptr.add(old_idx + 3);
            sum += pn3 * vn3;
            sum -= po3 * vo3;
            vsum += vn3 - vo3;
            *out_ptr.add(new_idx + 3) = sum / vsum;

            new_idx += 4;
            old_idx += 4;
        }

        while new_idx < len {
            let pn = *p_ptr.add(new_idx);
            let vn = *v_ptr.add(new_idx);
            let po = *p_ptr.add(old_idx);
            let vo = *v_ptr.add(old_idx);
            sum += pn * vn;
            sum -= po * vo;
            vsum += vn - vo;
            *out_ptr.add(new_idx) = sum / vsum;
            new_idx += 1;
            old_idx += 1;
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn vwma_avx2(price: &[f64], volume: &[f64], period: usize, first: usize, out: &mut [f64]) {
    
    
    vwma_scalar(price, volume, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn vwma_avx2_impl(
    price: &[f64],
    volume: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    use core::arch::x86_64::*;

    #[inline(always)]
    unsafe fn hsum256(a: __m256d) -> f64 {
        let hi = _mm256_extractf128_pd(a, 1);
        let lo = _mm256_castpd256_pd128(a);
        let sum = _mm_add_pd(lo, hi);
        let sh = _mm_unpackhi_pd(sum, sum);
        let sum = _mm_add_sd(sum, sh);
        _mm_cvtsd_f64(sum)
    }

    let len = price.len();
    if len < period {
        return;
    }
    let p_ptr = price.as_ptr();
    let v_ptr = volume.as_ptr();

    
    let base = first;
    let mut sum = 0.0f64;
    let mut vsum = 0.0f64;
    for i in 0..period {
        let p = *p_ptr.add(base + i);
        let v = *v_ptr.add(base + i);
        sum += p * v;
        vsum += v;
    }

    let mut out_idx = base + period - 1;
    *out.get_unchecked_mut(out_idx) = sum / vsum;

    
    let mut new_idx = out_idx + 1;
    let mut old_idx = base;
    while new_idx + 3 < len {
        
        let pn0 = *p_ptr.add(new_idx);
        let vn0 = *v_ptr.add(new_idx);
        let po0 = *p_ptr.add(old_idx);
        let vo0 = *v_ptr.add(old_idx);
        sum += pn0 * vn0;
        sum -= po0 * vo0;
        vsum += vn0 - vo0;
        *out.get_unchecked_mut(new_idx) = sum / vsum;

        
        let pn1 = *p_ptr.add(new_idx + 1);
        let vn1 = *v_ptr.add(new_idx + 1);
        let po1 = *p_ptr.add(old_idx + 1);
        let vo1 = *v_ptr.add(old_idx + 1);
        sum += pn1 * vn1;
        sum -= po1 * vo1;
        vsum += vn1 - vo1;
        *out.get_unchecked_mut(new_idx + 1) = sum / vsum;

        
        let pn2 = *p_ptr.add(new_idx + 2);
        let vn2 = *v_ptr.add(new_idx + 2);
        let po2 = *p_ptr.add(old_idx + 2);
        let vo2 = *v_ptr.add(old_idx + 2);
        sum += pn2 * vn2;
        sum -= po2 * vo2;
        vsum += vn2 - vo2;
        *out.get_unchecked_mut(new_idx + 2) = sum / vsum;

        
        let pn3 = *p_ptr.add(new_idx + 3);
        let vn3 = *v_ptr.add(new_idx + 3);
        let po3 = *p_ptr.add(old_idx + 3);
        let vo3 = *v_ptr.add(old_idx + 3);
        sum += pn3 * vn3;
        sum -= po3 * vo3;
        vsum += vn3 - vo3;
        *out.get_unchecked_mut(new_idx + 3) = sum / vsum;

        new_idx += 4;
        old_idx += 4;
    }
    while new_idx < len {
        let pn = *p_ptr.add(new_idx);
        let vn = *v_ptr.add(new_idx);
        let po = *p_ptr.add(old_idx);
        let vo = *v_ptr.add(old_idx);
        sum += pn * vn;
        sum -= po * vo;
        vsum += vn - vo;
        *out.get_unchecked_mut(new_idx) = sum / vsum;
        new_idx += 1;
        old_idx += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn vwma_avx512(price: &[f64], volume: &[f64], period: usize, first: usize, out: &mut [f64]) {
    
    vwma_scalar(price, volume, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn vwma_avx512_impl(
    price: &[f64],
    volume: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    use core::arch::x86_64::*;

    #[inline(always)]
    unsafe fn hsum512(a: __m512d) -> f64 {
        let lo256 = _mm512_castpd512_pd256(a);
        let hi256 = _mm512_extractf64x4_pd(a, 1);
        let sum256 = _mm256_add_pd(lo256, hi256);
        let hi = _mm256_extractf128_pd(sum256, 1);
        let lo = _mm256_castpd256_pd128(sum256);
        let s2 = _mm_add_pd(lo, hi);
        let sh = _mm_unpackhi_pd(s2, s2);
        let s1 = _mm_add_sd(s2, sh);
        _mm_cvtsd_f64(s1)
    }

    let len = price.len();
    if len < period {
        return;
    }
    let p_ptr = price.as_ptr();
    let v_ptr = volume.as_ptr();

    
    let base = first;
    let mut sum = 0.0f64;
    let mut vsum = 0.0f64;
    for i in 0..period {
        let p = *p_ptr.add(base + i);
        let v = *v_ptr.add(base + i);
        sum += p * v;
        vsum += v;
    }

    let mut out_idx = base + period - 1;
    *out.get_unchecked_mut(out_idx) = sum / vsum;

    
    let mut new_idx = out_idx + 1;
    let mut old_idx = base;
    while new_idx + 3 < len {
        
        let pn0 = *p_ptr.add(new_idx);
        let vn0 = *v_ptr.add(new_idx);
        let po0 = *p_ptr.add(old_idx);
        let vo0 = *v_ptr.add(old_idx);
        sum += pn0 * vn0;
        sum -= po0 * vo0;
        vsum += vn0 - vo0;
        *out.get_unchecked_mut(new_idx) = sum / vsum;

        
        let pn1 = *p_ptr.add(new_idx + 1);
        let vn1 = *v_ptr.add(new_idx + 1);
        let po1 = *p_ptr.add(old_idx + 1);
        let vo1 = *v_ptr.add(old_idx + 1);
        sum += pn1 * vn1;
        sum -= po1 * vo1;
        vsum += vn1 - vo1;
        *out.get_unchecked_mut(new_idx + 1) = sum / vsum;

        
        let pn2 = *p_ptr.add(new_idx + 2);
        let vn2 = *v_ptr.add(new_idx + 2);
        let po2 = *p_ptr.add(old_idx + 2);
        let vo2 = *v_ptr.add(old_idx + 2);
        sum += pn2 * vn2;
        sum -= po2 * vo2;
        vsum += vn2 - vo2;
        *out.get_unchecked_mut(new_idx + 2) = sum / vsum;

        
        let pn3 = *p_ptr.add(new_idx + 3);
        let vn3 = *v_ptr.add(new_idx + 3);
        let po3 = *p_ptr.add(old_idx + 3);
        let vo3 = *v_ptr.add(old_idx + 3);
        sum += pn3 * vn3;
        sum -= po3 * vo3;
        vsum += vn3 - vo3;
        *out.get_unchecked_mut(new_idx + 3) = sum / vsum;

        new_idx += 4;
        old_idx += 4;
    }
    while new_idx < len {
        let pn = *p_ptr.add(new_idx);
        let vn = *v_ptr.add(new_idx);
        let po = *p_ptr.add(old_idx);
        let vo = *v_ptr.add(old_idx);
        sum += pn * vn;
        sum -= po * vo;
        vsum += vn - vo;
        *out.get_unchecked_mut(new_idx) = sum / vsum;
        new_idx += 1;
        old_idx += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vwma_avx512_short(
    price: &[f64],
    volume: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    vwma_scalar(price, volume, period, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vwma_avx512_long(
    price: &[f64],
    volume: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    vwma_scalar(price, volume, period, first, out)
}

#[inline(always)]
pub fn vwma_batch_with_kernel(
    price: &[f64],
    volume: &[f64],
    sweep: &VwmaBatchRange,
    kernel: Kernel,
) -> Result<VwmaBatchOutput, VwmaError> {
    let chosen = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(VwmaError::InvalidKernelForBatch(other)),
    };
    let simd = match chosen {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        
        _ => Kernel::Scalar,
    };
    vwma_batch_par_slice(price, volume, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct VwmaBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for VwmaBatchRange {
    fn default() -> Self {
        Self {
            period: (20, 269, 1),
        }
    }
}

#[derive(Clone, Debug)]
pub struct VwmaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<VwmaParams>,
    pub rows: usize,
    pub cols: usize,
}
impl VwmaBatchOutput {
    pub fn row_for_params(&self, p: &VwmaParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(20) == p.period.unwrap_or(20))
    }
    pub fn values_for(&self, p: &VwmaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

fn expand_grid_vwma(r: &VwmaBatchRange) -> Vec<VwmaParams> {
    let (start, end, step) = r.period;
    if step == 0 || start == end {
        return vec![VwmaParams { period: Some(start) }];
    }
    if start < end {
        (start..=end)
            .step_by(step)
            .map(|p| VwmaParams { period: Some(p) })
            .collect()
    } else {
        
        let mut v = Vec::new();
        let mut p = start;
        while p >= end {
            v.push(VwmaParams { period: Some(p) });
            if p - end < step { break; }
            p -= step;
        }
        v
    }
}

#[inline(always)]
pub fn vwma_batch_slice(
    price: &[f64],
    volume: &[f64],
    sweep: &VwmaBatchRange,
    kern: Kernel,
) -> Result<VwmaBatchOutput, VwmaError> {
    vwma_batch_inner(price, volume, sweep, kern, false)
}

#[inline(always)]
pub fn vwma_batch_par_slice(
    price: &[f64],
    volume: &[f64],
    sweep: &VwmaBatchRange,
    kern: Kernel,
) -> Result<VwmaBatchOutput, VwmaError> {
    vwma_batch_inner(price, volume, sweep, kern, true)
}

#[inline]
fn vwma_batch_inner(
    price: &[f64],
    volume: &[f64],
    sweep: &VwmaBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<VwmaBatchOutput, VwmaError> {
    
    let combos = expand_grid_vwma(sweep);
    if combos.is_empty() {
        let (s, e, st) = sweep.period;
        return Err(VwmaError::InvalidRange {
            start: s,
            end: e,
            step: st,
        });
    }

    let len = price.len();
    if len == 0 {
        return Err(VwmaError::EmptyInputData);
    }
    if volume.len() != len {
        return Err(VwmaError::PriceVolumeMismatch {
            price_len: len,
            volume_len: volume.len(),
        });
    }
    let first = price
        .iter()
        .zip(volume.iter())
        .position(|(&p, &v)| !p.is_nan() && !v.is_nan())
        .ok_or(VwmaError::AllValuesNaN)?;

    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if len - first < max_p {
        return Err(VwmaError::NotEnoughValidData {
            needed: max_p,
            valid: len - first,
        });
    }

    
    let rows = combos.len();
    let cols = len;

    let mut warm_prefixes: Vec<usize> = Vec::with_capacity(combos.len());
    for c in &combos {
        let p = c.period.unwrap();
        let warm = first
            .checked_add(p)
            .and_then(|x| x.checked_sub(1))
            .ok_or(VwmaError::ArithmeticOverflow { context: "warmup prefix per-row" })?;
        warm_prefixes.push(warm);
    }

    let _ = rows
        .checked_mul(cols)
        .ok_or(VwmaError::InvalidRange {
            start: sweep.period.0,
            end: sweep.period.1,
            step: sweep.period.2,
        })?;
    let mut raw = make_uninit_matrix(rows, cols);
    unsafe { init_matrix_prefixes(&mut raw, cols, &warm_prefixes) };

    
    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();

        
        let out_row =
            core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

        match kern {
            Kernel::Scalar => vwma_row_scalar(price, volume, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => vwma_row_avx2(price, volume, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => vwma_row_avx512(price, volume, first, period, out_row),
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx512 => vwma_row_scalar(price, volume, first, period, out_row),
            
            _ => vwma_row_scalar(price, volume, first, period, out_row),
        }
    };

    
    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            raw.par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, slice) in raw.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
    } else {
        for (row, slice) in raw.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    
    let values: Vec<f64> = unsafe { std::mem::transmute(raw) };
    Ok(VwmaBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
pub unsafe fn vwma_row_scalar(
    price: &[f64],
    volume: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    vwma_scalar(price, volume, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vwma_row_avx2(
    price: &[f64],
    volume: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    
    vwma_scalar(price, volume, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vwma_row_avx512(
    price: &[f64],
    volume: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    if period <= 32 {
        vwma_row_avx512_short(price, volume, first, period, out);
    } else {
        vwma_row_avx512_long(price, volume, first, period, out);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vwma_row_avx512_short(
    price: &[f64],
    volume: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    
    vwma_scalar(price, volume, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vwma_row_avx512_long(
    price: &[f64],
    volume: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    
    vwma_scalar(price, volume, period, first, out)
}

#[derive(Debug, Clone)]
pub struct VwmaStream {
    period: usize,
    prices: Vec<f64>,
    volumes: Vec<f64>,
    sum: f64,
    vsum: f64,
    head: usize,
    filled: bool,
}

impl VwmaStream {
    pub fn try_new(params: VwmaParams) -> Result<Self, VwmaError> {
        let period = params.period.unwrap_or(20);
        if period == 0 {
            return Err(VwmaError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        Ok(Self {
            period,
            prices: vec![f64::NAN; period],
            volumes: vec![f64::NAN; period],
            sum: 0.0,
            vsum: 0.0,
            head: 0,
            filled: false,
        })
    }
    pub fn update(&mut self, price: f64, volume: f64) -> Option<f64> {
        
        let idx = self.head;
        let new_w = price * volume;

        if !self.filled {
            
            self.sum += new_w;
            self.vsum += volume;

            self.prices[idx] = price;
            self.volumes[idx] = volume;

            
            let next = idx + 1;
            if next == self.period {
                self.head = 0;
                self.filled = true;
                
                return Some(self.sum / self.vsum);
            } else {
                self.head = next;
                return None;
            }
        } else {
            
            let old_p = self.prices[idx];
            let old_v = self.volumes[idx];
            let old_w = old_p * old_v;

            self.sum += new_w - old_w;
            self.vsum += volume - old_v;

            self.prices[idx] = price;
            self.volumes[idx] = volume;

            
            let next = idx + 1;
            self.head = if next == self.period { 0 } else { next };

            Some(self.sum / self.vsum)
        }
    }
}

#[derive(Clone, Debug)]
pub struct VwmaBatchBuilder {
    range: VwmaBatchRange,
    kernel: Kernel,
}

impl Default for VwmaBatchBuilder {
    fn default() -> Self {
        Self {
            range: VwmaBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl VwmaBatchBuilder {
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
    pub fn apply_slice(
        self,
        prices: &[f64],
        volumes: &[f64],
    ) -> Result<VwmaBatchOutput, VwmaError> {
        vwma_batch_with_kernel(prices, volumes, &self.range, self.kernel)
    }
}

#[inline(always)]
pub fn vwma_batch_inner_into(
    price: &[f64],
    volume: &[f64],
    sweep: &VwmaBatchRange,
    kern: Kernel, 
    parallel: bool,
    out: &mut [f64], 
) -> Result<Vec<VwmaParams>, VwmaError> {
    
    let combos = expand_grid_vwma(sweep);
    if combos.is_empty() {
        let (s, e, st) = sweep.period;
        return Err(VwmaError::InvalidRange {
            start: s,
            end: e,
            step: st,
        });
    }
    let len = price.len();
    if volume.len() != len {
        return Err(VwmaError::PriceVolumeMismatch {
            price_len: len,
            volume_len: volume.len(),
        });
    }
    let first = price
        .iter()
        .zip(volume.iter())
        .position(|(&p, &v)| !p.is_nan() && !v.is_nan())
        .ok_or(VwmaError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if len - first < max_p {
        return Err(VwmaError::NotEnoughValidData {
            needed: max_p,
            valid: len - first,
        });
    }

    
    let rows = combos.len();
    let cols = len;
    let expected = rows
        .checked_mul(cols)
        .ok_or(VwmaError::InvalidRange {
            start: sweep.period.0,
            end: sweep.period.1,
            step: sweep.period.2,
        })?;
    if out.len() != expected {
        return Err(VwmaError::OutputLengthMismatch { expected, got: out.len() });
    }
    let out_mu: &mut [MaybeUninit<f64>] = unsafe {
        core::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };

    
    let mut warm_prefixes: Vec<usize> = Vec::with_capacity(combos.len());
    for c in &combos {
        let p = c.period.unwrap();
        let warm = first
            .checked_add(p)
            .and_then(|x| x.checked_sub(1))
            .ok_or(VwmaError::ArithmeticOverflow { context: "warmup prefix per-row" })?;
        warm_prefixes.push(warm);
    }
    init_matrix_prefixes(out_mu, cols, &warm_prefixes);

    
    let do_row = |row: usize, row_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();
        let row_out: &mut [f64] =
            core::slice::from_raw_parts_mut(row_mu.as_mut_ptr() as *mut f64, row_mu.len());
        match kern {
            Kernel::Scalar => vwma_row_scalar(price, volume, first, period, row_out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => vwma_row_avx2(price, volume, first, period, row_out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => vwma_row_avx512(price, volume, first, period, row_out),
            _ => vwma_row_scalar(price, volume, first, period, row_out),
        }
    };

    
    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out_mu
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(r, chunk)| do_row(r, chunk));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (r, chunk) in out_mu.chunks_mut(cols).enumerate() {
                do_row(r, chunk);
            }
        }
    } else {
        for (r, chunk) in out_mu.chunks_mut(cols).enumerate() {
            do_row(r, chunk);
        }
    }

    Ok(combos)
}

#[inline(always)]
pub fn vwma_batch_into_slice(
    dst: &mut [f64],
    price: &[f64],
    volume: &[f64],
    sweep: &VwmaBatchRange,
    k: Kernel,
) -> Result<Vec<VwmaParams>, VwmaError> {
    let simd = match if matches!(k, Kernel::Auto) {
        detect_best_batch_kernel()
    } else {
        k
    } {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => Kernel::Scalar,
    };
    vwma_batch_inner_into(price, volume, sweep, simd, true, dst)
}

#[inline(always)]
fn expand_grid(_r: &VwmaBatchRange) -> Vec<VwmaParams> {
    expand_grid_vwma(_r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    fn check_vwma_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = VwmaParams { period: None };
        let input_default = VwmaInput::from_candles(&candles, "close", default_params);
        let output_default = vwma_with_kernel(&input_default, kernel)?;
        assert_eq!(output_default.values.len(), candles.close.len());
        let custom_params = VwmaParams { period: Some(10) };
        let input_custom = VwmaInput::from_candles(&candles, "hlc3", custom_params);
        let output_custom = vwma_with_kernel(&input_custom, kernel)?;
        assert_eq!(output_custom.values.len(), candles.close.len());
        Ok(())
    }

    #[test]
    fn test_vwma_into_matches_api() -> Result<(), Box<dyn std::error::Error>> {
        
        let n = 256usize;
        let mut prices = Vec::with_capacity(n);
        let mut volumes = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f64;
            prices.push(100.0 + (t * 0.05).sin() * 2.0 + (t * 0.01).cos());
            volumes.push(((i * 3) % 50 + 1) as f64);
        }

        let params = VwmaParams { period: Some(20) };
        let input = VwmaInput::from_slice(&prices, &volumes, params);

        
        let baseline = vwma(&input)?.values;

        
        let mut out = vec![0.0; n];
        #[cfg(not(feature = "wasm"))]
        {
            vwma_into(&input, &mut out)?;
        }
        #[cfg(feature = "wasm")]
        {
            
            vwma_into_slice(&mut out, &input, Kernel::Auto)?;
        }

        assert_eq!(baseline.len(), out.len());
        for (a, b) in baseline.iter().zip(out.iter()) {
            let equal = (a.is_nan() && b.is_nan()) || (a == b);
            assert!(equal, "VWMA into parity failed: expected {}, got {}", a, b);
        }

        Ok(())
    }
    fn check_vwma_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let close_prices = candles.select_candle_field("close")?;
        let params = VwmaParams { period: Some(20) };
        let input = VwmaInput::from_candles(&candles, "close", params);
        let vwma_result = vwma_with_kernel(&input, kernel)?;
        assert_eq!(vwma_result.values.len(), close_prices.len());
        let expected_last_five_vwma = [
            59201.87047121331,
            59217.157390630266,
            59195.74526905522,
            59196.261392450084,
            59151.22059588594,
        ];
        let start_index = vwma_result.values.len() - 5;
        let result_last_five_vwma = &vwma_result.values[start_index..];
        for (i, &val) in result_last_five_vwma.iter().enumerate() {
            let exp = expected_last_five_vwma[i];
            assert!(
                (val - exp).abs() < 1e-3,
                "[{}] VWMA mismatch at index {}: expected {}, got {}",
                test_name,
                i,
                exp,
                val
            );
        }
        Ok(())
    }
    fn check_vwma_input_with_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = VwmaInput::with_default_candles(&candles);
        match input.data {
            VwmaData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected VwmaData::Candles"),
        }
        Ok(())
    }
    fn check_vwma_candles_plus_prices(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let custom_prices = candles
            .close
            .iter()
            .map(|v| v * 1.001)
            .collect::<Vec<f64>>();
        let params = VwmaParams { period: Some(20) };
        let input = VwmaInput::from_candles_plus_prices(&candles, &custom_prices, params);
        let result = vwma_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), custom_prices.len());
        Ok(())
    }
    fn check_vwma_slice_data_reinput(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params_first = VwmaParams { period: Some(20) };
        let input_first = VwmaInput::from_candles(&candles, "close", params_first);
        let result_first = vwma_with_kernel(&input_first, kernel)?;
        assert_eq!(result_first.values.len(), candles.close.len());
        let params_second = VwmaParams { period: Some(10) };
        let input_second =
            VwmaInput::from_slice(&result_first.values, &candles.volume, params_second);
        let result_second = vwma_with_kernel(&input_second, kernel)?;
        assert_eq!(result_second.values.len(), result_first.values.len());
        let start = input_first.get_period() + input_second.get_period() - 2;
        for i in start..result_second.values.len() {
            assert!(!result_second.values[i].is_nan());
        }
        Ok(())
    }

    macro_rules! generate_all_vwma_tests {
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

    
    #[cfg(debug_assertions)]
    fn check_vwma_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        
        let test_periods = vec![
            1,   
            5,   
            10,  
            20,  
            50,  
            100, 
            200, 
        ];

        for &period in &test_periods {
            
            if period > candles.close.len() {
                continue;
            }

            let input = VwmaInput::from_candles(
                &candles,
                "close",
                VwmaParams {
                    period: Some(period),
                },
            );
            let output = vwma_with_kernel(&input, kernel)?;

            
            for (i, &val) in output.values.iter().enumerate() {
                
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                
                if bits == 0x11111111_11111111 {
                    panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} with period {}",
						test_name, val, bits, i, period
					);
                }

                
                if bits == 0x22222222_22222222 {
                    panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} with period {}",
						test_name, val, bits, i, period
					);
                }

                
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

    
    #[cfg(not(debug_assertions))]
    fn check_vwma_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_vwma_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        
        
        
        let strat = (1usize..=50).prop_flat_map(|period| {
            (period..400).prop_flat_map(move |len| {
                (
                    prop::collection::vec(
                        (-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                        len,
                    ),
                    prop::collection::vec(
                        (0.0f64..1e6f64)
                            .prop_filter("non-negative finite", |x| x.is_finite() && *x >= 0.0),
                        len,
                    ),
                    Just(period),
                )
            })
        });

        proptest::test_runner::TestRunner::default()
			.run(&strat, |(prices, volumes, period)| {
				let params = VwmaParams { period: Some(period) };
				let input = VwmaInput::from_slice(&prices, &volumes, params);

				
				let VwmaOutput { values: out } = vwma_with_kernel(&input, kernel).unwrap();
				
				let VwmaOutput { values: ref_out } = vwma_with_kernel(&input, Kernel::Scalar).unwrap();

				
				prop_assert_eq!(out.len(), prices.len(), "Output length mismatch");
				prop_assert_eq!(out.len(), volumes.len(), "Output/volume length mismatch");

				
				let first_valid = 0;

				
				let warmup_end = first_valid + period - 1;

				
				for i in 0..warmup_end.min(out.len()) {
					prop_assert!(
						out[i].is_nan(),
						"Expected NaN during warmup at index {}, got {}",
						i,
						out[i]
					);
				}

				
				let is_constant_price = prices.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-12);
				let is_constant_volume = volumes.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-12);

				
				
				for i in warmup_end..prices.len() {
					let y = out[i];
					let r = ref_out[i];

					
					
					let window_start = if i >= period - 1 { i + 1 - period } else { 0 };
					let window_prices = &prices[window_start..=i];
					let window_volumes = &volumes[window_start..=i];

					
					let price_min = window_prices.iter().cloned().fold(f64::INFINITY, f64::min);
					let price_max = window_prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

					
					let volume_sum: f64 = window_volumes.iter().sum();
					let has_valid_volume = volume_sum > 0.0 && volume_sum.is_finite();

					
					if y.is_finite() && has_valid_volume {
						
						let tolerance = 1e-6 * price_max.abs().max(price_min.abs()).max(1.0);
						prop_assert!(
							y >= price_min - tolerance && y <= price_max + tolerance,
							"VWMA at idx {} out of bounds: {} not in [{}, {}]",
							i, y, price_min, price_max
						);
					} else if !has_valid_volume {
						
						
						

						
						let numerator: f64 = window_prices.iter()
							.zip(window_volumes.iter())
							.map(|(p, v)| p * v)
							.sum();

						if numerator == 0.0 || !numerator.is_finite() {
							
							prop_assert!(
								!y.is_finite() || y == 0.0 || y == -0.0,
								"Expected NaN, 0, or -0 for 0/0 case at idx {}, got {}",
								i, y
							);
						} else {
							
							
							
							
							
							if y.is_finite() {
								prop_assert!(
									y >= price_min - 1e-6 && y <= price_max + 1e-6,
									"VWMA with zero volume sum but non-zero numerator at idx {} out of bounds: {} not in [{}, {}]",
									i, y, price_min, price_max
								);
							}
						}
					}

					
					if y.is_finite() && r.is_finite() {
						let y_bits = y.to_bits();
						let r_bits = r.to_bits();
						let ulp_diff = y_bits.abs_diff(r_bits);

						prop_assert!(
							(y - r).abs() <= 1e-9 || ulp_diff <= 4,
							"SIMD mismatch at idx {}: {} vs {} (ULP={})",
							i, y, r, ulp_diff
						);
					} else {
						
						prop_assert_eq!(
							y.to_bits(),
							r.to_bits(),
							"Non-finite value mismatch at index {}",
							i
						);
					}

					
					if is_constant_price && i >= warmup_end + period {
						let const_price = prices[first_valid];
						prop_assert!(
							(y - const_price).abs() <= 1e-9,
							"Constant price property failed at idx {}: expected {}, got {}",
							i, const_price, y
						);
					}

					
					if period == 1 && y.is_finite() {
						
						let expected_price = prices[i];
						if expected_price.is_finite() && volumes[i] > 0.0 {
							
							let tolerance = (expected_price.abs() * 1e-10).max(1e-9);
							prop_assert!(
								(y - expected_price).abs() <= tolerance,
								"Period=1 property failed at idx {}: expected {}, got {}",
								i, expected_price, y
							);
						}
					}

					
					if is_constant_volume && volumes[first_valid] > 0.0 && y.is_finite() && has_valid_volume {
						
						let sma: f64 = window_prices.iter().sum::<f64>() / period as f64;
						prop_assert!(
							(y - sma).abs() <= 1e-9,
							"Constant volume property failed at idx {}: VWMA={}, SMA={}",
							i, y, sma
						);
					}
				}

				
				for (i, &v) in volumes.iter().enumerate() {
					if v.is_finite() {
						prop_assert!(
							v >= 0.0,
							"Volume at index {} is negative: {}",
							i, v
						);
					}
				}

				
				
				if volumes.iter().all(|&v| v > 0.0 && v.is_finite()) {
					let scaled_volumes: Vec<f64> = volumes.iter().map(|&v| v * 2.0).collect();
					let scaled_params = VwmaParams { period: Some(period) };
					let scaled_input = VwmaInput::from_slice(&prices, &scaled_volumes, scaled_params);
					if let Ok(VwmaOutput { values: scaled_out }) = vwma_with_kernel(&scaled_input, kernel) {
						for i in warmup_end..prices.len() {
							if out[i].is_finite() && scaled_out[i].is_finite() {
								prop_assert!(
									(out[i] - scaled_out[i]).abs() <= 1e-9,
									"Volume scaling invariance failed at idx {}: {} vs {}",
									i, out[i], scaled_out[i]
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

    generate_all_vwma_tests!(
        check_vwma_partial_params,
        check_vwma_accuracy,
        check_vwma_input_with_default_candles,
        check_vwma_candles_plus_prices,
        check_vwma_slice_data_reinput,
        check_vwma_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_vwma_tests!(check_vwma_property);
    #[cfg(test)]
    mod batch_tests {
        use super::*;
        use crate::skip_if_unsupported;
        use crate::utilities::data_loader::read_candles_from_csv;

        fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
            skip_if_unsupported!(kernel, test);

            let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
            let c = read_candles_from_csv(file)?;

            let output = VwmaBatchBuilder::new()
                .kernel(kernel)
                .apply_slice(&c.close, &c.volume)?;

            let def = VwmaParams::default();
            let row = output.values_for(&def).expect("default row missing");

            assert_eq!(row.len(), c.close.len());

            
            
            let expected = [
                59201.87047121331,
                59217.157390630266,
                59195.74526905522,
                59196.261392450084,
                59151.22059588594,
            ];
            let start = row.len() - 5;
            for (i, &v) in row[start..].iter().enumerate() {
                assert!(
                    (v - expected[i]).abs() < 1e-3,
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

        
        #[cfg(debug_assertions)]
        fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
            skip_if_unsupported!(kernel, test);

            let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
            let c = read_candles_from_csv(file)?;

            
            let batch_configs = vec![
                (1, 10, 1),    
                (5, 25, 5),    
                (10, 30, 10),  
                (20, 100, 10), 
                (50, 200, 50), 
                (1, 5, 1),     
            ];

            for (start, end, step) in batch_configs {
                
                if start > c.close.len() {
                    continue;
                }

                let output = VwmaBatchBuilder::new()
                    .kernel(kernel)
                    .period_range(start, end, step)
                    .apply_slice(&c.close, &c.volume)?;

                
                for (idx, &val) in output.values.iter().enumerate() {
                    
                    if val.is_nan() {
                        continue;
                    }

                    let bits = val.to_bits();
                    let row = idx / output.cols;
                    let col = idx % output.cols;
                    let period = output.combos[row].period.unwrap_or(0);

                    
                    if bits == 0x11111111_11111111 {
                        panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (flat index {}) for period {} in range ({}, {}, {})",
                        test, val, bits, row, col, idx, period, start, end, step
                    );
                    }

                    
                    if bits == 0x22222222_22222222 {
                        panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (flat index {}) for period {} in range ({}, {}, {})",
                        test, val, bits, row, col, idx, period, start, end, step
                    );
                    }

                    
                    if bits == 0x33333333_33333333 {
                        panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (flat index {}) for period {} in range ({}, {}, {})",
                        test, val, bits, row, col, idx, period, start, end, step
                    );
                    }
                }
            }

            Ok(())
        }

        
        #[cfg(not(debug_assertions))]
        fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
            Ok(())
        }

        gen_batch_tests!(check_batch_default_row);
        gen_batch_tests!(check_batch_no_poison);
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "vwma")]
#[pyo3(signature = (prices, volumes, period, kernel=None))]
/// Compute the Volume Weighted Moving Average (VWMA) of the input data.
///
/// VWMA weights each price by its volume over a moving window to capture
/// price action with regard to trading activity.
///
/// Parameters:
/// -----------
/// prices : np.ndarray
///     Price data array (float64).
/// volumes : np.ndarray
///     Volume data array (float64), must be same length as prices.
/// period : int
///     Number of bars in the moving average window.
/// kernel : str, optional
///     Computation kernel to use: 'auto', 'scalar', 'avx2', 'avx512'.
///     Default is 'auto' which auto-detects the best available.
///
/// Returns:
/// --------
/// np.ndarray
///     Array of VWMA values, same length as input.
///
/// Raises:
/// -------
/// ValueError
///     If inputs are invalid (mismatched lengths, invalid period, etc).
pub fn vwma_py<'py>(
    py: Python<'py>,
    prices: numpy::PyReadonlyArray1<'py, f64>,
    volumes: numpy::PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let prices_slice = prices.as_slice()?;
    let volumes_slice = volumes.as_slice()?;

    // Parse and validate kernel before allow_threads
    let kern = validate_kernel(kernel, false)?;

    // Build input struct
    let params = VwmaParams {
        period: Some(period),
    };
    let vwma_in = VwmaInput::from_slice(prices_slice, volumes_slice, params);

    // Get Vec<f64> from Rust function - zero-copy optimization
    let result_vec: Vec<f64> = py
        .allow_threads(|| vwma_with_kernel(&vwma_in, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Zero-copy transfer to NumPy
    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "VwmaStream")]
pub struct VwmaStreamPy {
    stream: VwmaStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl VwmaStreamPy {
    #[new]
    fn new(period: usize) -> PyResult<Self> {
        let params = VwmaParams {
            period: Some(period),
        };
        let stream =
            VwmaStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(VwmaStreamPy { stream })
    }

    /// Updates the stream with new price and volume values and returns the calculated VWMA value.
    /// Returns `None` if the buffer is not yet full.
    fn update(&mut self, price: f64, volume: f64) -> Option<f64> {
        self.stream.update(price, volume)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "vwma_batch")]
#[pyo3(signature = (prices, volumes, period_range, kernel=None))]
/// Compute VWMA for multiple period values in a single pass.
///
/// Parameters:
/// -----------
/// prices : np.ndarray
///     Price data array (float64).
/// volumes : np.ndarray
///     Volume data array (float64), must be same length as prices.
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
pub fn vwma_batch_py<'py>(
    py: Python<'py>,
    prices: numpy::PyReadonlyArray1<'py, f64>,
    volumes: numpy::PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let p = prices.as_slice()?;
    let v = volumes.as_slice()?;
    let sweep = VwmaBatchRange {
        period: period_range,
    };

    
    let combos0 = expand_grid_vwma(&sweep);
    let rows = combos0.len();
    let cols = p.len();

    
    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let out_slice = unsafe { out_arr.as_slice_mut()? };

    let kern = validate_kernel(kernel, true)?;
    let combos = py
        .allow_threads(|| {
            let batch = match kern {
                Kernel::Auto => detect_best_batch_kernel(),
                k => k,
            };
            let simd = match batch {
                Kernel::Avx512Batch => Kernel::Avx512,
                Kernel::Avx2Batch => Kernel::Avx2,
                Kernel::ScalarBatch => Kernel::Scalar,
                _ => Kernel::Scalar,
            };
            vwma_batch_inner_into(p, v, &sweep, simd, true, out_slice)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "periods",
        combos
            .iter()
            .map(|c| c.period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    Ok(dict.into())
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "vwma_cuda_batch_dev")]
#[pyo3(signature = (prices_f32, volumes_f32, period_range, device_id=0))]
pub fn vwma_cuda_batch_dev_py(
    py: Python<'_>,
    prices_f32: numpy::PyReadonlyArray1<'_, f32>,
    volumes_f32: numpy::PyReadonlyArray1<'_, f32>,
    period_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<VwmaDeviceArrayF32Py> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let price_slice = prices_f32.as_slice()?;
    let volume_slice = volumes_f32.as_slice()?;
    let sweep = VwmaBatchRange {
        period: period_range,
    };

    let (buf, rows, cols, ctx, dev) = py.allow_threads(|| {
        let cuda = CudaVwma::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let arr = cuda
            .vwma_batch_dev(price_slice, volume_slice, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx = cuda.context_arc();
        let dev = cuda.device_id();
        Ok::<_, pyo3::PyErr>((arr.buf, arr.rows, arr.cols, ctx, dev))
    })?;

    Ok(VwmaDeviceArrayF32Py { buf: Some(buf), rows, cols, _ctx: ctx, device_id: dev })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "vwma_cuda_many_series_one_param_dev")]
#[pyo3(signature = (prices_tm_f32, volumes_tm_f32, period, device_id=0))]
pub fn vwma_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    prices_tm_f32: numpy::PyReadonlyArray2<'_, f32>,
    volumes_tm_f32: numpy::PyReadonlyArray2<'_, f32>,
    period: usize,
    device_id: usize,
) -> PyResult<VwmaDeviceArrayF32Py> {
    use numpy::PyUntypedArrayMethods;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let price_shape = prices_tm_f32.shape();
    let volume_shape = volumes_tm_f32.shape();
    if price_shape != volume_shape {
        return Err(PyValueError::new_err(
            "price and volume matrices must share shape",
        ));
    }

    let rows = price_shape[0];
    let cols = price_shape[1];

    let prices_flat = prices_tm_f32.as_slice()?;
    let volumes_flat = volumes_tm_f32.as_slice()?;

    let (buf, rows_o, cols_o, ctx, dev) = py.allow_threads(|| {
        let cuda = CudaVwma::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let arr = cuda
            .vwma_many_series_one_param_time_major_dev(
                prices_flat,
                volumes_flat,
                cols,
                rows,
                period,
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx = cuda.context_arc();
        let dev = cuda.device_id();
        Ok::<_, pyo3::PyErr>((arr.buf, arr.rows, arr.cols, ctx, dev))
    })?;

    Ok(VwmaDeviceArrayF32Py { buf: Some(buf), rows: rows_o, cols: cols_o, _ctx: ctx, device_id: dev })
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vwma_js(prices: &[f64], volumes: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
    let params = VwmaParams {
        period: Some(period),
    };
    let input = VwmaInput::from_slice(prices, volumes, params);

    let mut output = vec![0.0; prices.len()];
    vwma_into_slice(&mut output, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vwma_batch_js(
    prices: &[f64],
    volumes: &[f64],
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = VwmaBatchRange {
        period: (period_start, period_end, period_step),
    };

    
    let kernel = detect_best_batch_kernel();
    let simd_kernel = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        
        _ => Kernel::Scalar,
    };

    
    vwma_batch_inner(prices, volumes, &sweep, simd_kernel, false)
        .map(|output| output.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vwma_batch_metadata_js(
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = VwmaBatchRange {
        period: (period_start, period_end, period_step),
    };

    let combos = expand_grid(&sweep);
    let mut metadata = Vec::with_capacity(combos.len());

    for combo in combos {
        metadata.push(combo.period.unwrap() as f64);
    }

    Ok(metadata)
}

#[inline]
pub fn vwma_into_slice(dst: &mut [f64], input: &VwmaInput, kern: Kernel) -> Result<(), VwmaError> {
    let (price, volume): (&[f64], &[f64]) = match &input.data {
        VwmaData::Candles { candles, source } => {
            (source_type(candles, source), source_type(candles, "volume"))
        }
        VwmaData::CandlesPlusPrices { candles, prices } => (prices, source_type(candles, "volume")),
        VwmaData::Slice { prices, volumes } => (prices, volumes),
    };
    let len = price.len();
    let period = input.get_period();

    if period == 0 || period > len {
        return Err(VwmaError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if volume.len() != len {
        return Err(VwmaError::PriceVolumeMismatch {
            price_len: len,
            volume_len: volume.len(),
        });
    }
    if dst.len() != len {
        return Err(VwmaError::OutputLengthMismatch { expected: len, got: dst.len() });
    }

    let first = price
        .iter()
        .zip(volume.iter())
        .position(|(&p, &v)| !p.is_nan() && !v.is_nan())
        .ok_or(VwmaError::AllValuesNaN)?;

    if (len - first) < period {
        return Err(VwmaError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let chosen = match kern {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => vwma_scalar(price, volume, period, first, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => vwma_avx2(price, volume, period, first, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => vwma_avx512(price, volume, period, first, dst),
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                vwma_scalar(price, volume, period, first, dst)
            }
            
            _ => vwma_scalar(price, volume, period, first, dst),
        }
    }

    
    let warmup_end = first
        .checked_add(period)
        .and_then(|x| x.checked_sub(1))
        .ok_or(VwmaError::ArithmeticOverflow { context: "warmup prefix index" })?;
    for v in &mut dst[..warmup_end] {
        *v = f64::from_bits(0x7ff8_0000_0000_0000);
    }

    Ok(())
}

/// Writes VWMA into a caller-provided buffer without allocating.
///
/// Preserves NaN warmups identical to the Vec-returning API (quiet-NaN pattern)
/// and requires `out.len() == input length`.
#[cfg(not(feature = "wasm"))]
#[inline]
pub fn vwma_into(input: &VwmaInput, out: &mut [f64]) -> Result<(), VwmaError> {
    vwma_into_slice(out, input, Kernel::Auto)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vwma_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vwma_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vwma_into(
    price_ptr: *const f64,
    volume_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
) -> Result<(), JsValue> {
    if price_ptr.is_null() || volume_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let prices = std::slice::from_raw_parts(price_ptr, len);
        let volumes = std::slice::from_raw_parts(volume_ptr, len);
        let params = VwmaParams {
            period: Some(period),
        };
        let input = VwmaInput::from_slice(prices, volumes, params);

        
        if price_ptr == out_ptr || volume_ptr == out_ptr {
            
            let mut temp = vec![0.0; len];
            vwma_into_slice(&mut temp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            vwma_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct VwmaBatchConfig {
    pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct VwmaBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<VwmaParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = vwma_batch)]
pub fn vwma_batch_unified_js(
    prices: &[f64],
    volumes: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: VwmaBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = VwmaBatchRange {
        period: config.period_range,
    };

    
    let kernel = detect_best_batch_kernel();
    let simd_kernel = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        
        _ => Kernel::Scalar,
    };
    let output = vwma_batch_inner(prices, volumes, &sweep, simd_kernel, false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = VwmaBatchJsOutput {
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
pub fn vwma_batch_into(
    price_ptr: *const f64,
    volume_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<usize, JsValue> {
    if price_ptr.is_null() || volume_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }
    unsafe {
        let prices = std::slice::from_raw_parts(price_ptr, len);
        let volumes = std::slice::from_raw_parts(volume_ptr, len);
        let sweep = VwmaBatchRange {
            period: (period_start, period_end, period_step),
        };
        let combos = expand_grid_vwma(&sweep);
        let rows = combos.len();

        
        let simd = match detect_best_batch_kernel() {
            Kernel::Avx512Batch => Kernel::Avx512,
            Kernel::Avx2Batch => Kernel::Avx2,
            Kernel::ScalarBatch => Kernel::Scalar,
            _ => Kernel::Scalar,
        };

        if (out_ptr as *const f64) == price_ptr || (out_ptr as *const f64) == volume_ptr {
            
            let mut tmp = vec![0f64; rows * len];
            vwma_batch_inner_into(prices, volumes, &sweep, simd, false, &mut tmp)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, rows * len);
            out.copy_from_slice(&tmp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, rows * len);
            vwma_batch_inner_into(prices, volumes, &sweep, simd, false, out)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        Ok(rows)
    }
}
