#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::cuda_available;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::moving_averages::CudaBuffAverages;
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
use pyo3::types::{PyDict, PyList};
#[cfg(all(feature = "python", feature = "cuda"))]
use std::sync::Arc;

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use serde::{Deserialize, Serialize};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen::prelude::*;

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

use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(module = "ta_indicators.cuda", unsendable)]
pub struct BuffAveragesDeviceArrayF32Py {
    pub(crate) buf: Option<DeviceBuffer<f32>>,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) _ctx: Arc<Context>,
    pub(crate) device_id: u32,
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pymethods]
impl BuffAveragesDeviceArrayF32Py {
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

    #[pyo3(signature=(stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__<'py>(
        &mut self,
        py: Python<'py>,
        stream: Option<pyo3::PyObject>,
        max_version: Option<pyo3::PyObject>,
        dl_device: Option<pyo3::PyObject>,
        copy: Option<pyo3::PyObject>,
    ) -> PyResult<PyObject> {
        use crate::utilities::dlpack_cuda::export_f32_cuda_dlpack_2d;

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
                        return Err(PyValueError::new_err("device mismatch for __dlpack__"));
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

impl<'a> AsRef<[f64]> for BuffAveragesInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            BuffAveragesData::Slice(slice) => slice,
            BuffAveragesData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum BuffAveragesData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct BuffAveragesOutput {
    pub fast_buff: Vec<f64>,
    pub slow_buff: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct BuffAveragesParams {
    pub fast_period: Option<usize>,
    pub slow_period: Option<usize>,
}

impl Default for BuffAveragesParams {
    fn default() -> Self {
        Self {
            fast_period: Some(5),
            slow_period: Some(20),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BuffAveragesInput<'a> {
    pub data: BuffAveragesData<'a>,
    pub volume: Option<&'a [f64]>,
    pub params: BuffAveragesParams,
}

impl<'a> BuffAveragesInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: BuffAveragesParams) -> Self {
        Self {
            data: BuffAveragesData::Candles {
                candles: c,
                source: s,
            },
            volume: Some(&c.volume),
            params: p,
        }
    }

    #[inline]
    pub fn from_slices(price: &'a [f64], volume: &'a [f64], p: BuffAveragesParams) -> Self {
        Self {
            data: BuffAveragesData::Slice(price),
            volume: Some(volume),
            params: p,
        }
    }

    #[inline]
    pub fn from_slice(sl: &'a [f64], p: BuffAveragesParams) -> Self {
        Self {
            data: BuffAveragesData::Slice(sl),
            volume: None,
            params: p,
        }
    }

    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", BuffAveragesParams::default())
    }

    #[inline]
    pub fn get_fast_period(&self) -> usize {
        self.params.fast_period.unwrap_or(5)
    }

    #[inline]
    pub fn get_slow_period(&self) -> usize {
        self.params.slow_period.unwrap_or(20)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct BuffAveragesBuilder {
    fast_period: Option<usize>,
    slow_period: Option<usize>,
    kernel: Kernel,
}

impl Default for BuffAveragesBuilder {
    fn default() -> Self {
        Self {
            fast_period: None,
            slow_period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl BuffAveragesBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn fast_period(mut self, val: usize) -> Self {
        self.fast_period = Some(val);
        self
    }

    #[inline(always)]
    pub fn slow_period(mut self, val: usize) -> Self {
        self.slow_period = Some(val);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<BuffAveragesOutput, BuffAveragesError> {
        let p = BuffAveragesParams {
            fast_period: self.fast_period,
            slow_period: self.slow_period,
        };
        let i = BuffAveragesInput::from_candles(c, "close", p);
        buff_averages_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        price: &[f64],
        volume: &[f64],
    ) -> Result<BuffAveragesOutput, BuffAveragesError> {
        let p = BuffAveragesParams {
            fast_period: self.fast_period,
            slow_period: self.slow_period,
        };
        let i = BuffAveragesInput::from_slices(price, volume, p);
        buff_averages_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<BuffAveragesStream, BuffAveragesError> {
        let p = BuffAveragesParams {
            fast_period: self.fast_period,
            slow_period: self.slow_period,
        };
        BuffAveragesStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum BuffAveragesError {
    #[error("buff_averages: Input data slice is empty.")]
    EmptyInputData,

    #[error("buff_averages: All values are NaN.")]
    AllValuesNaN,

    #[error("buff_averages: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },

    #[error("buff_averages: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },

    #[error("buff_averages: Price and volume arrays have different lengths: price = {price_len}, volume = {volume_len}")]
    MismatchedDataLength { price_len: usize, volume_len: usize },

    #[error("buff_averages: Volume data is required for this indicator")]
    MissingVolumeData,

    #[error("buff_averages: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },

    #[error("buff_averages: Invalid range: start = {start}, end = {end}, step = {step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },

    #[error("buff_averages: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),

    #[error("buff_averages: size overflow for rows = {rows}, cols = {cols}")]
    SizeOverflow { rows: usize, cols: usize },
}

#[inline]
pub fn buff_averages(input: &BuffAveragesInput) -> Result<BuffAveragesOutput, BuffAveragesError> {
    buff_averages_with_kernel(input, Kernel::Auto)
}

pub fn buff_averages_with_kernel(
    input: &BuffAveragesInput,
    kernel: Kernel,
) -> Result<BuffAveragesOutput, BuffAveragesError> {
    let (price, volume, fast_period, slow_period, first, chosen) =
        buff_averages_prepare(input, kernel)?;

    let warm = first + slow_period - 1;

    let mut fast_buff = alloc_with_nan_prefix(price.len(), warm);
    let mut slow_buff = alloc_with_nan_prefix(price.len(), warm);

    buff_averages_compute_into(
        price,
        volume,
        fast_period,
        slow_period,
        first,
        chosen,
        &mut fast_buff,
        &mut slow_buff,
    );

    Ok(BuffAveragesOutput {
        fast_buff,
        slow_buff,
    })
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn buff_averages_into(
    input: &BuffAveragesInput,
    fast_out: &mut [f64],
    slow_out: &mut [f64],
) -> Result<(), BuffAveragesError> {
    let (price, volume, fast_period, slow_period, first, chosen) =
        buff_averages_prepare(input, Kernel::Auto)?;

    if fast_out.len() != price.len() || slow_out.len() != price.len() {
        return Err(BuffAveragesError::OutputLengthMismatch {
            expected: price.len(),
            got: core::cmp::min(fast_out.len(), slow_out.len()),
        });
    }

    let warm = first + slow_period - 1;
    let nan = f64::from_bits(0x7ff8_0000_0000_0000);
    let warmup_len = warm.min(price.len());
    for v in &mut fast_out[..warmup_len] {
        *v = nan;
    }
    for v in &mut slow_out[..warmup_len] {
        *v = nan;
    }

    buff_averages_compute_into(
        price,
        volume,
        fast_period,
        slow_period,
        first,
        chosen,
        fast_out,
        slow_out,
    );

    Ok(())
}

#[inline]
pub fn buff_averages_into_slices(
    fast_dst: &mut [f64],
    slow_dst: &mut [f64],
    input: &BuffAveragesInput,
    kern: Kernel,
) -> Result<(), BuffAveragesError> {
    let (price, volume, fast_p, slow_p, first, chosen) = buff_averages_prepare(input, kern)?;

    if fast_dst.len() != price.len() || slow_dst.len() != price.len() {
        return Err(BuffAveragesError::OutputLengthMismatch {
            expected: price.len(),
            got: core::cmp::min(fast_dst.len(), slow_dst.len()),
        });
    }

    buff_averages_compute_into(
        price, volume, fast_p, slow_p, first, chosen, fast_dst, slow_dst,
    );

    let warm = first + slow_p - 1;
    for x in &mut fast_dst[..warm] {
        *x = f64::NAN;
    }
    for x in &mut slow_dst[..warm] {
        *x = f64::NAN;
    }

    Ok(())
}

#[inline(always)]
fn buff_averages_prepare<'a>(
    input: &'a BuffAveragesInput,
    kernel: Kernel,
) -> Result<(&'a [f64], &'a [f64], usize, usize, usize, Kernel), BuffAveragesError> {
    let price: &[f64] = input.as_ref();
    let len = price.len();

    if len == 0 {
        return Err(BuffAveragesError::EmptyInputData);
    }

    let volume = match &input.data {
        BuffAveragesData::Candles { candles, .. } => &candles.volume,
        BuffAveragesData::Slice(_) => input.volume.ok_or(BuffAveragesError::MissingVolumeData)?,
    };

    if price.len() != volume.len() {
        return Err(BuffAveragesError::MismatchedDataLength {
            price_len: price.len(),
            volume_len: volume.len(),
        });
    }

    let first = price
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(BuffAveragesError::AllValuesNaN)?;

    let fast_period = input.get_fast_period();
    let slow_period = input.get_slow_period();

    if fast_period == 0 || fast_period > len {
        return Err(BuffAveragesError::InvalidPeriod {
            period: fast_period,
            data_len: len,
        });
    }

    if slow_period == 0 || slow_period > len {
        return Err(BuffAveragesError::InvalidPeriod {
            period: slow_period,
            data_len: len,
        });
    }

    if len - first < slow_period {
        return Err(BuffAveragesError::NotEnoughValidData {
            needed: slow_period,
            valid: len - first,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        k => k,
    };

    Ok((price, volume, fast_period, slow_period, first, chosen))
}

#[inline(always)]
fn buff_averages_compute_into(
    price: &[f64],
    volume: &[f64],
    fast_period: usize,
    slow_period: usize,
    first: usize,
    kernel: Kernel,
    fast_out: &mut [f64],
    slow_out: &mut [f64],
) {
    unsafe {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            if matches!(kernel, Kernel::Scalar | Kernel::ScalarBatch) {
                buff_averages_simd128(
                    price,
                    volume,
                    fast_period,
                    slow_period,
                    first,
                    fast_out,
                    slow_out,
                );
                return;
            }
        }

        match kernel {
            Kernel::Scalar | Kernel::ScalarBatch => buff_averages_scalar(
                price,
                volume,
                fast_period,
                slow_period,
                first,
                fast_out,
                slow_out,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => buff_averages_avx2(
                price,
                volume,
                fast_period,
                slow_period,
                first,
                fast_out,
                slow_out,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => buff_averages_avx512(
                price,
                volume,
                fast_period,
                slow_period,
                first,
                fast_out,
                slow_out,
            ),
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                buff_averages_scalar(
                    price,
                    volume,
                    fast_period,
                    slow_period,
                    first,
                    fast_out,
                    slow_out,
                )
            }
            _ => unreachable!(),
        }
    }
}

#[inline]
pub fn buff_averages_scalar(
    price: &[f64],
    volume: &[f64],
    fast_period: usize,
    slow_period: usize,
    first: usize,
    fast_out: &mut [f64],
    slow_out: &mut [f64],
) {
    let len = price.len();
    if len == 0 {
        return;
    }

    let warm = first + slow_period - 1;
    if warm >= len {
        return;
    }

    let mut slow_numerator = 0.0;
    let mut slow_denominator = 0.0;
    let slow_start = warm + 1 - slow_period;
    for i in slow_start..=warm {
        let p = price[i];
        let v = volume[i];
        if !p.is_nan() && !v.is_nan() {
            slow_numerator += p * v;
            slow_denominator += v;
        }
    }

    let mut fast_numerator = 0.0;
    let mut fast_denominator = 0.0;
    let fast_start = warm + 1 - fast_period;
    for i in fast_start..=warm {
        let p = price[i];
        let v = volume[i];
        if !p.is_nan() && !v.is_nan() {
            fast_numerator += p * v;
            fast_denominator += v;
        }
    }

    if slow_denominator != 0.0 {
        slow_out[warm] = slow_numerator / slow_denominator;
    } else {
        slow_out[warm] = 0.0;
    }

    if fast_denominator != 0.0 {
        fast_out[warm] = fast_numerator / fast_denominator;
    } else {
        fast_out[warm] = 0.0;
    }

    for i in (warm + 1)..len {
        let old_slow = i - slow_period;
        let new_p = price[i];
        let new_v = volume[i];
        let old_p = price[old_slow];
        let old_v = volume[old_slow];

        if !old_p.is_nan() && !old_v.is_nan() {
            slow_numerator -= old_p * old_v;
            slow_denominator -= old_v;
        }
        if !new_p.is_nan() && !new_v.is_nan() {
            slow_numerator += new_p * new_v;
            slow_denominator += new_v;
        }

        slow_out[i] = if slow_denominator != 0.0 {
            slow_numerator / slow_denominator
        } else {
            0.0
        };

        let old_fast = i - fast_period;
        let old_pf = price[old_fast];
        let old_vf = volume[old_fast];

        if !old_pf.is_nan() && !old_vf.is_nan() {
            fast_numerator -= old_pf * old_vf;
            fast_denominator -= old_vf;
        }
        if !new_p.is_nan() && !new_v.is_nan() {
            fast_numerator += new_p * new_v;
            fast_denominator += new_v;
        }

        fast_out[i] = if fast_denominator != 0.0 {
            fast_numerator / fast_denominator
        } else {
            0.0
        };
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
unsafe fn buff_averages_simd128(
    price: &[f64],
    volume: &[f64],
    fast_period: usize,
    slow_period: usize,
    first: usize,
    fast_out: &mut [f64],
    slow_out: &mut [f64],
) {
    buff_averages_scalar(
        price,
        volume,
        fast_period,
        slow_period,
        first,
        fast_out,
        slow_out,
    );
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn hsum256_pd(v: __m256d) -> f64 {
    let hi: __m128d = _mm256_extractf128_pd::<1>(v);
    let lo: __m128d = _mm256_castpd256_pd128(v);
    let sum2: __m128d = _mm_add_pd(lo, hi);
    let hi64: __m128d = _mm_unpackhi_pd(sum2, sum2);
    let sum: __m128d = _mm_add_sd(sum2, hi64);
    _mm_cvtsd_f64(sum)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn buff_averages_avx2(
    price: &[f64],
    volume: &[f64],
    fast_period: usize,
    slow_period: usize,
    first: usize,
    fast_out: &mut [f64],
    slow_out: &mut [f64],
) {
    let len = price.len();
    if len == 0 {
        return;
    }
    let warm = first + slow_period - 1;
    if warm >= len {
        return;
    }

    let slow_start = warm + 1 - slow_period;
    let mut i = slow_start;
    let end = warm + 1;
    let mut slow_num_v = _mm256_setzero_pd();
    let mut slow_den_v = _mm256_setzero_pd();

    while i + 4 <= end {
        let p = _mm256_loadu_pd(price.as_ptr().add(i));
        let v = _mm256_loadu_pd(volume.as_ptr().add(i));

        let mp = _mm256_cmp_pd(p, p, _CMP_ORD_Q);
        let mv = _mm256_cmp_pd(v, v, _CMP_ORD_Q);
        let m = _mm256_and_pd(mp, mv);

        let pz = _mm256_and_pd(p, m);
        let vz = _mm256_and_pd(v, m);
        slow_num_v = _mm256_fmadd_pd(pz, vz, slow_num_v);
        slow_den_v = _mm256_add_pd(slow_den_v, vz);
        i += 4;
    }

    let mut slow_numerator = hsum256_pd(slow_num_v);
    let mut slow_denominator = hsum256_pd(slow_den_v);

    while i < end {
        let p = *price.get_unchecked(i);
        let v = *volume.get_unchecked(i);
        if !p.is_nan() && !v.is_nan() {
            slow_numerator += p * v;
            slow_denominator += v;
        }
        i += 1;
    }

    let fast_start = warm + 1 - fast_period;
    let mut j = fast_start;
    let mut fast_num_v = _mm256_setzero_pd();
    let mut fast_den_v = _mm256_setzero_pd();

    while j + 4 <= end {
        let p = _mm256_loadu_pd(price.as_ptr().add(j));
        let v = _mm256_loadu_pd(volume.as_ptr().add(j));
        let mp = _mm256_cmp_pd(p, p, _CMP_ORD_Q);
        let mv = _mm256_cmp_pd(v, v, _CMP_ORD_Q);
        let m = _mm256_and_pd(mp, mv);
        let pz = _mm256_and_pd(p, m);
        let vz = _mm256_and_pd(v, m);
        fast_num_v = _mm256_fmadd_pd(pz, vz, fast_num_v);
        fast_den_v = _mm256_add_pd(fast_den_v, vz);
        j += 4;
    }

    let mut fast_numerator = hsum256_pd(fast_num_v);
    let mut fast_denominator = hsum256_pd(fast_den_v);

    while j < end {
        let p = *price.get_unchecked(j);
        let v = *volume.get_unchecked(j);
        if !p.is_nan() && !v.is_nan() {
            fast_numerator += p * v;
            fast_denominator += v;
        }
        j += 1;
    }

    slow_out[warm] = if slow_denominator != 0.0 {
        slow_numerator / slow_denominator
    } else {
        0.0
    };
    fast_out[warm] = if fast_denominator != 0.0 {
        fast_numerator / fast_denominator
    } else {
        0.0
    };

    for k in (warm + 1)..len {
        let old_slow = k - slow_period;
        let new_p = *price.get_unchecked(k);
        let new_v = *volume.get_unchecked(k);
        let old_p = *price.get_unchecked(old_slow);
        let old_v = *volume.get_unchecked(old_slow);

        if !old_p.is_nan() && !old_v.is_nan() {
            slow_numerator -= old_p * old_v;
            slow_denominator -= old_v;
        }
        if !new_p.is_nan() && !new_v.is_nan() {
            slow_numerator += new_p * new_v;
            slow_denominator += new_v;
        }
        slow_out[k] = if slow_denominator != 0.0 {
            slow_numerator / slow_denominator
        } else {
            0.0
        };

        let old_fast = k - fast_period;
        let old_pf = *price.get_unchecked(old_fast);
        let old_vf = *volume.get_unchecked(old_fast);
        if !old_pf.is_nan() && !old_vf.is_nan() {
            fast_numerator -= old_pf * old_vf;
            fast_denominator -= old_vf;
        }
        if !new_p.is_nan() && !new_v.is_nan() {
            fast_numerator += new_p * new_v;
            fast_denominator += new_v;
        }
        fast_out[k] = if fast_denominator != 0.0 {
            fast_numerator / fast_denominator
        } else {
            0.0
        };
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
unsafe fn buff_averages_avx512(
    price: &[f64],
    volume: &[f64],
    fast_period: usize,
    slow_period: usize,
    first: usize,
    fast_out: &mut [f64],
    slow_out: &mut [f64],
) {
    let len = price.len();
    if len == 0 {
        return;
    }
    let warm = first + slow_period - 1;
    if warm >= len {
        return;
    }

    let slow_start = warm + 1 - slow_period;
    let mut i = slow_start;
    let end = warm + 1;
    let mut slow_num_v = _mm512_setzero_pd();
    let mut slow_den_v = _mm512_setzero_pd();

    while i + 8 <= end {
        let p = _mm512_loadu_pd(price.as_ptr().add(i));
        let v = _mm512_loadu_pd(volume.as_ptr().add(i));

        let mp: __mmask8 = _mm512_cmp_pd_mask(p, p, _CMP_ORD_Q);
        let mv: __mmask8 = _mm512_cmp_pd_mask(v, v, _CMP_ORD_Q);
        let m: __mmask8 = mp & mv;

        let pz = _mm512_maskz_mov_pd(m, p);
        let vz = _mm512_maskz_mov_pd(m, v);
        slow_num_v = _mm512_fmadd_pd(pz, vz, slow_num_v);
        slow_den_v = _mm512_add_pd(slow_den_v, vz);

        i += 8;
    }

    let mut slow_numerator = _mm512_reduce_add_pd(slow_num_v);
    let mut slow_denominator = _mm512_reduce_add_pd(slow_den_v);

    while i < end {
        let p = *price.get_unchecked(i);
        let v = *volume.get_unchecked(i);
        if !p.is_nan() && !v.is_nan() {
            slow_numerator += p * v;
            slow_denominator += v;
        }
        i += 1;
    }

    let fast_start = warm + 1 - fast_period;
    let mut j = fast_start;
    let mut fast_num_v = _mm512_setzero_pd();
    let mut fast_den_v = _mm512_setzero_pd();

    while j + 8 <= end {
        let p = _mm512_loadu_pd(price.as_ptr().add(j));
        let v = _mm512_loadu_pd(volume.as_ptr().add(j));
        let mp: __mmask8 = _mm512_cmp_pd_mask(p, p, _CMP_ORD_Q);
        let mv: __mmask8 = _mm512_cmp_pd_mask(v, v, _CMP_ORD_Q);
        let m: __mmask8 = mp & mv;

        let pz = _mm512_maskz_mov_pd(m, p);
        let vz = _mm512_maskz_mov_pd(m, v);
        fast_num_v = _mm512_fmadd_pd(pz, vz, fast_num_v);
        fast_den_v = _mm512_add_pd(fast_den_v, vz);

        j += 8;
    }

    let mut fast_numerator = _mm512_reduce_add_pd(fast_num_v);
    let mut fast_denominator = _mm512_reduce_add_pd(fast_den_v);

    while j < end {
        let p = *price.get_unchecked(j);
        let v = *volume.get_unchecked(j);
        if !p.is_nan() && !v.is_nan() {
            fast_numerator += p * v;
            fast_denominator += v;
        }
        j += 1;
    }

    slow_out[warm] = if slow_denominator != 0.0 {
        slow_numerator / slow_denominator
    } else {
        0.0
    };
    fast_out[warm] = if fast_denominator != 0.0 {
        fast_numerator / fast_denominator
    } else {
        0.0
    };

    for k in (warm + 1)..len {
        let old_slow = k - slow_period;
        let new_p = *price.get_unchecked(k);
        let new_v = *volume.get_unchecked(k);
        let old_p = *price.get_unchecked(old_slow);
        let old_v = *volume.get_unchecked(old_slow);

        if !old_p.is_nan() && !old_v.is_nan() {
            slow_numerator -= old_p * old_v;
            slow_denominator -= old_v;
        }
        if !new_p.is_nan() && !new_v.is_nan() {
            slow_numerator += new_p * new_v;
            slow_denominator += new_v;
        }
        slow_out[k] = if slow_denominator != 0.0 {
            slow_numerator / slow_denominator
        } else {
            0.0
        };

        let old_fast = k - fast_period;
        let old_pf = *price.get_unchecked(old_fast);
        let old_vf = *volume.get_unchecked(old_fast);
        if !old_pf.is_nan() && !old_vf.is_nan() {
            fast_numerator -= old_pf * old_vf;
            fast_denominator -= old_vf;
        }
        if !new_p.is_nan() && !new_v.is_nan() {
            fast_numerator += new_p * new_v;
            fast_denominator += new_v;
        }
        fast_out[k] = if fast_denominator != 0.0 {
            fast_numerator / fast_denominator
        } else {
            0.0
        };
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn build_masked_pv_v_avx2(
    price: &[f64],
    volume: &[f64],
    pv: &mut [f64],
    vv: &mut [f64],
) {
    debug_assert_eq!(price.len(), volume.len());
    debug_assert_eq!(pv.len(), price.len());
    debug_assert_eq!(vv.len(), price.len());

    let n = price.len();
    let mut i = 0usize;

    while i + 4 <= n {
        let p = _mm256_loadu_pd(price.as_ptr().add(i));
        let v = _mm256_loadu_pd(volume.as_ptr().add(i));

        let mp = _mm256_cmp_pd(p, p, _CMP_ORD_Q);
        let mv = _mm256_cmp_pd(v, v, _CMP_ORD_Q);
        let m = _mm256_and_pd(mp, mv);

        let pvv = _mm256_mul_pd(p, v);
        let pv_masked = _mm256_and_pd(pvv, m);
        let vv_masked = _mm256_and_pd(v, m);

        _mm256_storeu_pd(pv.as_mut_ptr().add(i), pv_masked);
        _mm256_storeu_pd(vv.as_mut_ptr().add(i), vv_masked);

        i += 4;
    }

    while i < n {
        let p = *price.get_unchecked(i);
        let v = *volume.get_unchecked(i);
        if !p.is_nan() && !v.is_nan() {
            *pv.get_unchecked_mut(i) = p * v;
            *vv.get_unchecked_mut(i) = v;
        } else {
            *pv.get_unchecked_mut(i) = 0.0;
            *vv.get_unchecked_mut(i) = 0.0;
        }
        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
pub unsafe fn build_masked_pv_v_avx512(
    price: &[f64],
    volume: &[f64],
    pv: &mut [f64],
    vv: &mut [f64],
) {
    debug_assert_eq!(price.len(), volume.len());
    debug_assert_eq!(pv.len(), price.len());
    debug_assert_eq!(vv.len(), price.len());

    let n = price.len();
    let mut i = 0usize;

    while i + 8 <= n {
        let p = _mm512_loadu_pd(price.as_ptr().add(i));
        let v = _mm512_loadu_pd(volume.as_ptr().add(i));

        let mp: __mmask8 = _mm512_cmp_pd_mask(p, p, _CMP_ORD_Q);
        let mv: __mmask8 = _mm512_cmp_pd_mask(v, v, _CMP_ORD_Q);
        let m: __mmask8 = mp & mv;

        let pvv = _mm512_mul_pd(p, v);
        let pv_masked = _mm512_maskz_mov_pd(m, pvv);
        let vv_masked = _mm512_maskz_mov_pd(m, v);

        _mm512_storeu_pd(pv.as_mut_ptr().add(i), pv_masked);
        _mm512_storeu_pd(vv.as_mut_ptr().add(i), vv_masked);

        i += 8;
    }

    while i < n {
        let p = *price.get_unchecked(i);
        let v = *volume.get_unchecked(i);
        if !p.is_nan() && !v.is_nan() {
            *pv.get_unchecked_mut(i) = p * v;
            *vv.get_unchecked_mut(i) = v;
        } else {
            *pv.get_unchecked_mut(i) = 0.0;
            *vv.get_unchecked_mut(i) = 0.0;
        }
        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn buff_averages_row_avx2_from_masked(
    pv: &[f64],
    vv: &[f64],
    fast_period: usize,
    slow_period: usize,
    first: usize,
    fast_out: &mut [f64],
    slow_out: &mut [f64],
) {
    let len = pv.len();
    if len == 0 {
        return;
    }

    let warm = first + slow_period - 1;
    if warm >= len {
        return;
    }

    let end = warm + 1;

    let mut i = end - slow_period;
    let mut s_num_v = _mm256_setzero_pd();
    let mut s_den_v = _mm256_setzero_pd();
    while i + 4 <= end {
        let pv4 = _mm256_loadu_pd(pv.as_ptr().add(i));
        let v4 = _mm256_loadu_pd(vv.as_ptr().add(i));
        s_num_v = _mm256_add_pd(s_num_v, pv4);
        s_den_v = _mm256_add_pd(s_den_v, v4);
        i += 4;
    }
    let mut slow_numerator = hsum256_pd(s_num_v);
    let mut slow_denominator = hsum256_pd(s_den_v);
    while i < end {
        slow_numerator += *pv.get_unchecked(i);
        slow_denominator += *vv.get_unchecked(i);
        i += 1;
    }

    let mut j = end - fast_period;
    let mut f_num_v = _mm256_setzero_pd();
    let mut f_den_v = _mm256_setzero_pd();
    while j + 4 <= end {
        let pv4 = _mm256_loadu_pd(pv.as_ptr().add(j));
        let v4 = _mm256_loadu_pd(vv.as_ptr().add(j));
        f_num_v = _mm256_add_pd(f_num_v, pv4);
        f_den_v = _mm256_add_pd(f_den_v, v4);
        j += 4;
    }
    let mut fast_numerator = hsum256_pd(f_num_v);
    let mut fast_denominator = hsum256_pd(f_den_v);
    while j < end {
        fast_numerator += *pv.get_unchecked(j);
        fast_denominator += *vv.get_unchecked(j);
        j += 1;
    }

    slow_out[warm] = if slow_denominator != 0.0 {
        slow_numerator / slow_denominator
    } else {
        0.0
    };
    fast_out[warm] = if fast_denominator != 0.0 {
        fast_numerator / fast_denominator
    } else {
        0.0
    };

    for k in (warm + 1)..len {
        let old_s = k - slow_period;
        let new_pv = *pv.get_unchecked(k);
        let new_vv = *vv.get_unchecked(k);
        let old_pv = *pv.get_unchecked(old_s);
        let old_vv = *vv.get_unchecked(old_s);
        slow_numerator -= old_pv;
        slow_denominator -= old_vv;
        slow_numerator += new_pv;
        slow_denominator += new_vv;
        slow_out[k] = if slow_denominator != 0.0 {
            slow_numerator / slow_denominator
        } else {
            0.0
        };

        let old_f = k - fast_period;
        let old_fp = *pv.get_unchecked(old_f);
        let old_fv = *vv.get_unchecked(old_f);
        fast_numerator -= old_fp;
        fast_denominator -= old_fv;
        fast_numerator += new_pv;
        fast_denominator += new_vv;
        fast_out[k] = if fast_denominator != 0.0 {
            fast_numerator / fast_denominator
        } else {
            0.0
        };
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
pub unsafe fn buff_averages_row_avx512_from_masked(
    pv: &[f64],
    vv: &[f64],
    fast_period: usize,
    slow_period: usize,
    first: usize,
    fast_out: &mut [f64],
    slow_out: &mut [f64],
) {
    let len = pv.len();
    if len == 0 {
        return;
    }

    let warm = first + slow_period - 1;
    if warm >= len {
        return;
    }

    let end = warm + 1;

    let mut i = end - slow_period;
    let mut s_num_v = _mm512_setzero_pd();
    let mut s_den_v = _mm512_setzero_pd();
    while i + 8 <= end {
        let pv8 = _mm512_loadu_pd(pv.as_ptr().add(i));
        let v8 = _mm512_loadu_pd(vv.as_ptr().add(i));
        s_num_v = _mm512_add_pd(s_num_v, pv8);
        s_den_v = _mm512_add_pd(s_den_v, v8);
        i += 8;
    }
    let mut slow_numerator = _mm512_reduce_add_pd(s_num_v);
    let mut slow_denominator = _mm512_reduce_add_pd(s_den_v);
    while i < end {
        slow_numerator += *pv.get_unchecked(i);
        slow_denominator += *vv.get_unchecked(i);
        i += 1;
    }

    let mut j = end - fast_period;
    let mut f_num_v = _mm512_setzero_pd();
    let mut f_den_v = _mm512_setzero_pd();
    while j + 8 <= end {
        let pv8 = _mm512_loadu_pd(pv.as_ptr().add(j));
        let v8 = _mm512_loadu_pd(vv.as_ptr().add(j));
        f_num_v = _mm512_add_pd(f_num_v, pv8);
        f_den_v = _mm512_add_pd(f_den_v, v8);
        j += 8;
    }
    let mut fast_numerator = _mm512_reduce_add_pd(f_num_v);
    let mut fast_denominator = _mm512_reduce_add_pd(f_den_v);
    while j < end {
        fast_numerator += *pv.get_unchecked(j);
        fast_denominator += *vv.get_unchecked(j);
        j += 1;
    }

    slow_out[warm] = if slow_denominator != 0.0 {
        slow_numerator / slow_denominator
    } else {
        0.0
    };
    fast_out[warm] = if fast_denominator != 0.0 {
        fast_numerator / fast_denominator
    } else {
        0.0
    };

    for k in (warm + 1)..len {
        let old_s = k - slow_period;
        let new_pv = *pv.get_unchecked(k);
        let new_vv = *vv.get_unchecked(k);
        let old_pv = *pv.get_unchecked(old_s);
        let old_vv = *vv.get_unchecked(old_s);
        slow_numerator -= old_pv;
        slow_denominator -= old_vv;
        slow_numerator += new_pv;
        slow_denominator += new_vv;
        slow_out[k] = if slow_denominator != 0.0 {
            slow_numerator / slow_denominator
        } else {
            0.0
        };

        let old_f = k - fast_period;
        let old_fp = *pv.get_unchecked(old_f);
        let old_fv = *vv.get_unchecked(old_f);
        fast_numerator -= old_fp;
        fast_denominator -= old_fv;
        fast_numerator += new_pv;
        fast_denominator += new_vv;
        fast_out[k] = if fast_denominator != 0.0 {
            fast_numerator / fast_denominator
        } else {
            0.0
        };
    }
}

#[derive(Debug, Clone)]
pub struct BuffAveragesStream {
    ring_pv: Vec<f64>,
    ring_vv: Vec<f64>,

    cap: usize,

    fast_period: usize,
    slow_period: usize,

    fast_num: f64,
    fast_den: f64,
    slow_num: f64,
    slow_den: f64,

    index: usize,

    warm_target_count: Option<usize>,
}

impl BuffAveragesStream {
    #[inline]
    pub fn try_new(params: BuffAveragesParams) -> Result<Self, BuffAveragesError> {
        let fast_period = params.fast_period.unwrap_or(5);
        let slow_period = params.slow_period.unwrap_or(20);

        if fast_period == 0 {
            return Err(BuffAveragesError::InvalidPeriod {
                period: fast_period,
                data_len: 0,
            });
        }
        if slow_period == 0 {
            return Err(BuffAveragesError::InvalidPeriod {
                period: slow_period,
                data_len: 0,
            });
        }

        let cap = core::cmp::max(fast_period, slow_period);

        Ok(Self {
            ring_pv: vec![0.0; cap],
            ring_vv: vec![0.0; cap],
            cap,
            fast_period,
            slow_period,
            fast_num: 0.0,
            fast_den: 0.0,
            slow_num: 0.0,
            slow_den: 0.0,
            index: 0,
            warm_target_count: None,
        })
    }

    #[inline]
    pub fn update(&mut self, price: f64, volume: f64) -> Option<(f64, f64)> {
        let n = self.index;
        let write_idx = n % self.cap;

        if self.warm_target_count.is_none() && !price.is_nan() {
            self.warm_target_count = Some(n + self.slow_period);
        }

        let valid = !price.is_nan() && !volume.is_nan();
        let pv_new = if valid {
            price.mul_add(volume, 0.0)
        } else {
            0.0
        };
        let vv_new = if valid { volume } else { 0.0 };

        if n >= self.slow_period {
            let idx_out_slow = (n + self.cap - self.slow_period) % self.cap;
            let old_pv = unsafe { *self.ring_pv.get_unchecked(idx_out_slow) };
            let old_vv = unsafe { *self.ring_vv.get_unchecked(idx_out_slow) };
            self.slow_num -= old_pv;
            self.slow_den -= old_vv;
        }
        if n >= self.fast_period {
            let idx_out_fast = (n + self.cap - self.fast_period) % self.cap;
            let old_pv = unsafe { *self.ring_pv.get_unchecked(idx_out_fast) };
            let old_vv = unsafe { *self.ring_vv.get_unchecked(idx_out_fast) };
            self.fast_num -= old_pv;
            self.fast_den -= old_vv;
        }

        unsafe {
            *self.ring_pv.get_unchecked_mut(write_idx) = pv_new;
            *self.ring_vv.get_unchecked_mut(write_idx) = vv_new;
        }

        self.slow_num += pv_new;
        self.slow_den += vv_new;
        self.fast_num += pv_new;
        self.fast_den += vv_new;

        self.index = n + 1;

        if let Some(warm) = self.warm_target_count {
            if self.index >= warm {
                let slow = if self.slow_den != 0.0 {
                    self.slow_num / self.slow_den
                } else {
                    0.0
                };
                let fast = if self.fast_den != 0.0 {
                    self.fast_num / self.fast_den
                } else {
                    0.0
                };
                return Some((fast, slow));
            }
        }
        None
    }
}

#[derive(Clone, Debug)]
pub struct BuffAveragesBatchRange {
    pub fast_period: (usize, usize, usize),
    pub slow_period: (usize, usize, usize),
}

impl Default for BuffAveragesBatchRange {
    fn default() -> Self {
        Self {
            fast_period: (5, 5, 0),
            slow_period: (20, 269, 1),
        }
    }
}

#[derive(Clone, Debug)]
pub struct BuffAveragesBatchOutput {
    pub fast: Vec<f64>,
    pub slow: Vec<f64>,
    pub combos: Vec<(usize, usize)>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone, Debug, Default)]
pub struct BuffAveragesBatchBuilder {
    range: BuffAveragesBatchRange,
    kernel: Kernel,
}

impl BuffAveragesBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline]
    pub fn fast_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.fast_period = (start, end, step);
        self
    }

    #[inline]
    pub fn fast_period_static(mut self, val: usize) -> Self {
        self.range.fast_period = (val, val, 0);
        self
    }

    #[inline]
    pub fn slow_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.slow_period = (start, end, step);
        self
    }

    #[inline]
    pub fn slow_period_static(mut self, val: usize) -> Self {
        self.range.slow_period = (val, val, 0);
        self
    }

    pub fn apply_candles(
        self,
        candles: &Candles,
    ) -> Result<BuffAveragesBatchOutput, BuffAveragesError> {
        let price = source_type(candles, "close");
        let volume = &candles.volume;
        buff_averages_batch_with_kernel(price, volume, &self.range, self.kernel)
    }

    pub fn apply_slices(
        self,
        price: &[f64],
        volume: &[f64],
    ) -> Result<BuffAveragesBatchOutput, BuffAveragesError> {
        buff_averages_batch_with_kernel(price, volume, &self.range, self.kernel)
    }
}

fn expand_grid_ba(r: &BuffAveragesBatchRange) -> Vec<(usize, usize)> {
    fn axis((a, b, s): (usize, usize, usize)) -> Vec<usize> {
        if s == 0 || a == b {
            return vec![a];
        }
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        (lo..=hi).step_by(s).collect()
    }

    let fasts = axis(r.fast_period);
    let slows = axis(r.slow_period);
    let mut v = Vec::with_capacity(fasts.len() * slows.len());

    for &f in &fasts {
        for &s in &slows {
            v.push((f, s));
        }
    }
    v
}

#[inline]
pub fn buff_averages_batch_inner_into(
    price: &[f64],
    volume: &[f64],
    sweep: &BuffAveragesBatchRange,
    kern: Kernel,
    fast_out: &mut [f64],
    slow_out: &mut [f64],
) -> Result<Vec<(usize, usize)>, BuffAveragesError> {
    buff_averages_batch_inner_into_parallel(price, volume, sweep, kern, fast_out, slow_out, false)
}

#[inline]
fn buff_averages_batch_inner_into_parallel(
    price: &[f64],
    volume: &[f64],
    sweep: &BuffAveragesBatchRange,
    kern: Kernel,
    fast_out: &mut [f64],
    slow_out: &mut [f64],
    parallel: bool,
) -> Result<Vec<(usize, usize)>, BuffAveragesError> {
    let combos = expand_grid_ba(sweep);
    if combos.is_empty() {
        let (fs, fe, fp) = sweep.fast_period;
        return Err(BuffAveragesError::InvalidRange {
            start: fs.min(fe),
            end: fs.max(fe),
            step: fp,
        });
    }

    if price.len() != volume.len() || price.is_empty() {
        return Err(BuffAveragesError::MismatchedDataLength {
            price_len: price.len(),
            volume_len: volume.len(),
        });
    }

    let first = price
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(BuffAveragesError::AllValuesNaN)?;

    let max_slow = combos.iter().map(|&(_, s)| s).max().unwrap();
    if price.len() - first < max_slow {
        return Err(BuffAveragesError::NotEnoughValidData {
            needed: max_slow,
            valid: price.len() - first,
        });
    }

    let rows = combos.len();
    let cols = price.len();
    if rows.checked_mul(cols).is_none() {
        return Err(BuffAveragesError::SizeOverflow { rows, cols });
    }
    let expected = rows * cols;
    if fast_out.len() != expected || slow_out.len() != expected {
        return Err(BuffAveragesError::OutputLengthMismatch {
            expected,
            got: core::cmp::min(fast_out.len(), slow_out.len()),
        });
    }

    let fast_mu = unsafe {
        core::slice::from_raw_parts_mut(
            fast_out.as_mut_ptr() as *mut core::mem::MaybeUninit<f64>,
            fast_out.len(),
        )
    };
    let slow_mu = unsafe {
        core::slice::from_raw_parts_mut(
            slow_out.as_mut_ptr() as *mut core::mem::MaybeUninit<f64>,
            slow_out.len(),
        )
    };

    let warms: Vec<usize> = combos.iter().map(|&(_, slow)| first + slow - 1).collect();
    init_matrix_prefixes(fast_mu, cols, &warms);
    init_matrix_prefixes(slow_mu, cols, &warms);

    match kern {
        Kernel::Auto | Kernel::ScalarBatch | Kernel::Avx2Batch | Kernel::Avx512Batch => {}
        other => return Err(BuffAveragesError::InvalidKernelForBatch(other)),
    }

    let simd = match match kern {
        Kernel::Auto => Kernel::ScalarBatch,
        k => k,
    } {
        Kernel::ScalarBatch => Kernel::Scalar,
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2Batch => Kernel::Avx2,
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512Batch => Kernel::Avx512,
        _ => Kernel::Scalar,
    };

    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
    let masked_buffers: Option<(Vec<f64>, Vec<f64>)> =
        if rows > 1 && matches!(simd, Kernel::Avx2 | Kernel::Avx512) {
            let mut pv = vec![0.0; price.len()];
            let mut vv = vec![0.0; price.len()];
            unsafe {
                match simd {
                    Kernel::Avx2 => build_masked_pv_v_avx2(price, volume, &mut pv, &mut vv),
                    Kernel::Avx512 => build_masked_pv_v_avx512(price, volume, &mut pv, &mut vv),
                    _ => {}
                }
            }
            Some((pv, vv))
        } else {
            None
        };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;

            fast_out
                .par_chunks_mut(cols)
                .zip(slow_out.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(row, (fr, sr))| {
                    let (fp, sp) = combos[row];
                    let handled = {
                        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                        {
                            if let Some((pv, vv)) = masked_buffers.as_ref() {
                                unsafe {
                                    match simd {
                                        Kernel::Avx2 => {
                                            buff_averages_row_avx2_from_masked(
                                                pv, vv, fp, sp, first, fr, sr,
                                            );
                                            true
                                        }
                                        Kernel::Avx512 => {
                                            buff_averages_row_avx512_from_masked(
                                                pv, vv, fp, sp, first, fr, sr,
                                            );
                                            true
                                        }
                                        _ => false,
                                    }
                                }
                            } else {
                                false
                            }
                        }
                        #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
                        {
                            false
                        }
                    };

                    if !handled {
                        buff_averages_compute_into(price, volume, fp, sp, first, simd, fr, sr);
                    }
                });
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, &(fp, sp)) in combos.iter().enumerate() {
                let fr = &mut fast_out[row * cols..(row + 1) * cols];
                let sr = &mut slow_out[row * cols..(row + 1) * cols];
                let handled = {
                    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                    {
                        if let Some((pv, vv)) = masked_buffers.as_ref() {
                            unsafe {
                                match simd {
                                    Kernel::Avx2 => {
                                        buff_averages_row_avx2_from_masked(
                                            pv, vv, fp, sp, first, fr, sr,
                                        );
                                        true
                                    }
                                    Kernel::Avx512 => {
                                        buff_averages_row_avx512_from_masked(
                                            pv, vv, fp, sp, first, fr, sr,
                                        );
                                        true
                                    }
                                    _ => false,
                                }
                            }
                        } else {
                            false
                        }
                    }
                    #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
                    {
                        false
                    }
                };

                if !handled {
                    buff_averages_compute_into(price, volume, fp, sp, first, simd, fr, sr);
                }
            }
        }
    } else {
        for (row, &(fp, sp)) in combos.iter().enumerate() {
            let fr = &mut fast_out[row * cols..(row + 1) * cols];
            let sr = &mut slow_out[row * cols..(row + 1) * cols];
            let handled = {
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                {
                    if let Some((pv, vv)) = masked_buffers.as_ref() {
                        unsafe {
                            match simd {
                                Kernel::Avx2 => {
                                    buff_averages_row_avx2_from_masked(
                                        pv, vv, fp, sp, first, fr, sr,
                                    );
                                    true
                                }
                                Kernel::Avx512 => {
                                    buff_averages_row_avx512_from_masked(
                                        pv, vv, fp, sp, first, fr, sr,
                                    );
                                    true
                                }
                                _ => false,
                            }
                        }
                    } else {
                        false
                    }
                }
                #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
                {
                    false
                }
            };

            if !handled {
                buff_averages_compute_into(price, volume, fp, sp, first, simd, fr, sr);
            }
        }
    }

    Ok(combos)
}

pub fn buff_averages_batch_with_kernel(
    price: &[f64],
    volume: &[f64],
    sweep: &BuffAveragesBatchRange,
    k: Kernel,
) -> Result<BuffAveragesBatchOutput, BuffAveragesError> {
    buff_averages_batch_inner(price, volume, sweep, k, false)
}

#[inline(always)]
pub fn buff_averages_batch_par_slice(
    price: &[f64],
    volume: &[f64],
    sweep: &BuffAveragesBatchRange,
    k: Kernel,
) -> Result<BuffAveragesBatchOutput, BuffAveragesError> {
    buff_averages_batch_inner(price, volume, sweep, k, true)
}

#[inline(always)]
fn buff_averages_batch_inner(
    price: &[f64],
    volume: &[f64],
    sweep: &BuffAveragesBatchRange,
    k: Kernel,
    parallel: bool,
) -> Result<BuffAveragesBatchOutput, BuffAveragesError> {
    if price.is_empty() {
        return Err(BuffAveragesError::EmptyInputData);
    }
    if price.len() != volume.len() {
        return Err(BuffAveragesError::MismatchedDataLength {
            price_len: price.len(),
            volume_len: volume.len(),
        });
    }
    let first = price
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(BuffAveragesError::AllValuesNaN)?;
    let combos = expand_grid_ba(sweep);
    if combos.is_empty() {
        let (fs, fe, fp) = sweep.fast_period;
        return Err(BuffAveragesError::InvalidRange {
            start: fs.min(fe),
            end: fs.max(fe),
            step: fp,
        });
    }
    let max_slow = combos.iter().map(|&(_, s)| s).max().unwrap();
    if price.len() - first < max_slow {
        return Err(BuffAveragesError::NotEnoughValidData {
            needed: max_slow,
            valid: price.len() - first,
        });
    }

    let rows = combos.len();
    let cols = price.len();
    if rows.checked_mul(cols).is_none() {
        return Err(BuffAveragesError::SizeOverflow { rows, cols });
    }

    let mut fast_mu = make_uninit_matrix(rows, cols);
    let mut slow_mu = make_uninit_matrix(rows, cols);

    let fast_slice =
        unsafe { core::slice::from_raw_parts_mut(fast_mu.as_mut_ptr() as *mut f64, fast_mu.len()) };
    let slow_slice =
        unsafe { core::slice::from_raw_parts_mut(slow_mu.as_mut_ptr() as *mut f64, slow_mu.len()) };

    buff_averages_batch_inner_into_parallel(
        price, volume, sweep, k, fast_slice, slow_slice, parallel,
    )?;

    let fast = unsafe {
        let ptr = fast_mu.as_mut_ptr() as *mut f64;
        let len = fast_mu.len();
        let cap = fast_mu.capacity();
        core::mem::forget(fast_mu);
        Vec::from_raw_parts(ptr, len, cap)
    };
    let slow = unsafe {
        let ptr = slow_mu.as_mut_ptr() as *mut f64;
        let len = slow_mu.len();
        let cap = slow_mu.capacity();
        core::mem::forget(slow_mu);
        Vec::from_raw_parts(ptr, len, cap)
    };

    Ok(BuffAveragesBatchOutput {
        fast,
        slow,
        combos,
        rows,
        cols,
    })
}

#[cfg(feature = "python")]
#[pyfunction(name = "buff_averages")]
#[pyo3(signature = (price, volume, fast_period=5, slow_period=20, kernel=None))]
pub fn buff_averages_py<'py>(
    py: Python<'py>,
    price: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    fast_period: usize,
    slow_period: usize,
    kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let price_slice = price.as_slice()?;
    let volume_slice = volume.as_slice()?;
    let kern = validate_kernel(kernel, false)?;
    let params = BuffAveragesParams {
        fast_period: Some(fast_period),
        slow_period: Some(slow_period),
    };
    let input = BuffAveragesInput::from_slices(price_slice, volume_slice, params);

    let result = py
        .allow_threads(|| buff_averages_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok((
        result.fast_buff.into_pyarray(py),
        result.slow_buff.into_pyarray(py),
    ))
}

#[cfg(feature = "python")]
#[pyfunction(name = "buff_averages_batch")]
#[pyo3(signature = (price, volume, fast_range, slow_range, kernel=None))]
pub fn buff_averages_batch_py<'py>(
    py: Python<'py>,
    price: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    fast_range: (usize, usize, usize),
    slow_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::IntoPyArray;
    let p = price.as_slice()?;
    let v = volume.as_slice()?;
    let sweep = BuffAveragesBatchRange {
        fast_period: fast_range,
        slow_period: slow_range,
    };
    let kern = validate_kernel(kernel, true)?;

    let combos = expand_grid_ba(&sweep);
    let rows = combos.len();
    let cols = p.len();
    let fast_arr = unsafe { numpy::PyArray1::<f64>::new(py, [rows * cols], false) };
    let slow_arr = unsafe { numpy::PyArray1::<f64>::new(py, [rows * cols], false) };

    let fast_slice = unsafe { fast_arr.as_slice_mut()? };
    let slow_slice = unsafe { slow_arr.as_slice_mut()? };

    let combos = py
        .allow_threads(|| {
            buff_averages_batch_inner_into(p, v, &sweep, kern, fast_slice, slow_slice)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let d = PyDict::new(py);
    d.set_item("fast", fast_arr.reshape((rows, cols))?)?;
    d.set_item("slow", slow_arr.reshape((rows, cols))?)?;
    d.set_item(
        "fast_periods",
        combos
            .iter()
            .map(|c| c.0 as i64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    d.set_item(
        "slow_periods",
        combos
            .iter()
            .map(|c| c.1 as i64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    Ok(d)
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "buff_averages_cuda_batch_dev")]
#[pyo3(signature = (price_f32, volume_f32, fast_range, slow_range, device_id=0))]
pub fn buff_averages_cuda_batch_dev_py(
    py: Python<'_>,
    price_f32: PyReadonlyArray1<'_, f32>,
    volume_f32: PyReadonlyArray1<'_, f32>,
    fast_range: (usize, usize, usize),
    slow_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<(BuffAveragesDeviceArrayF32Py, BuffAveragesDeviceArrayF32Py)> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let price = price_f32.as_slice()?;
    let volume = volume_f32.as_slice()?;
    let sweep = BuffAveragesBatchRange {
        fast_period: fast_range,
        slow_period: slow_range,
    };

    let (fast, slow, rows, cols, ctx, dev) = py.allow_threads(|| {
        let cuda =
            CudaBuffAverages::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let (f, s) = cuda
            .buff_averages_batch_dev(price, volume, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let rows = f.rows;
        let cols = f.cols;
        let ctx = cuda.context_arc();
        let dev = cuda.device_id();
        Ok::<_, pyo3::PyErr>((f.buf, s.buf, rows, cols, ctx, dev))
    })?;

    Ok((
        BuffAveragesDeviceArrayF32Py {
            buf: Some(fast),
            rows,
            cols,
            _ctx: ctx.clone(),
            device_id: dev,
        },
        BuffAveragesDeviceArrayF32Py {
            buf: Some(slow),
            rows,
            cols,
            _ctx: ctx,
            device_id: dev,
        },
    ))
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "buff_averages_cuda_many_series_one_param_dev")]
#[pyo3(signature = (prices_tm_f32, volumes_tm_f32, cols, rows, fast_period, slow_period, device_id=0))]
pub fn buff_averages_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    prices_tm_f32: PyReadonlyArray1<'_, f32>,
    volumes_tm_f32: PyReadonlyArray1<'_, f32>,
    cols: usize,
    rows: usize,
    fast_period: usize,
    slow_period: usize,
    device_id: usize,
) -> PyResult<(BuffAveragesDeviceArrayF32Py, BuffAveragesDeviceArrayF32Py)> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let prices = prices_tm_f32.as_slice()?;
    let volumes = volumes_tm_f32.as_slice()?;

    let (fast, slow, rows_o, cols_o, ctx, dev) = py.allow_threads(|| {
        let cuda =
            CudaBuffAverages::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let (f, s) = cuda
            .buff_averages_many_series_one_param_time_major_dev(
                prices,
                volumes,
                cols,
                rows,
                fast_period,
                slow_period,
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx = cuda.context_arc();
        let dev = cuda.device_id();
        Ok::<_, pyo3::PyErr>((f.buf, s.buf, f.rows, f.cols, ctx, dev))
    })?;

    Ok((
        BuffAveragesDeviceArrayF32Py {
            buf: Some(fast),
            rows: rows_o,
            cols: cols_o,
            _ctx: ctx.clone(),
            device_id: dev,
        },
        BuffAveragesDeviceArrayF32Py {
            buf: Some(slow),
            rows: rows_o,
            cols: cols_o,
            _ctx: ctx,
            device_id: dev,
        },
    ))
}

#[cfg(feature = "python")]
#[pyclass(name = "BuffAveragesStream")]
pub struct BuffAveragesStreamPy {
    stream: BuffAveragesStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl BuffAveragesStreamPy {
    #[new]
    fn new(fast_period: usize, slow_period: usize) -> PyResult<Self> {
        let params = BuffAveragesParams {
            fast_period: Some(fast_period),
            slow_period: Some(slow_period),
        };
        let stream = BuffAveragesStream::try_new(params)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(BuffAveragesStreamPy { stream })
    }

    fn update(&mut self, price: f64, volume: f64) -> Option<(f64, f64)> {
        self.stream.update(price, volume)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct BuffAveragesJsResult {
    pub values: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = buff_averages)]
pub fn buff_averages_unified_js(
    price: &[f64],
    volume: &[f64],
    fast_period: usize,
    slow_period: usize,
) -> Result<JsValue, JsValue> {
    let len = price.len();
    let params = BuffAveragesParams {
        fast_period: Some(fast_period),
        slow_period: Some(slow_period),
    };
    let input = BuffAveragesInput::from_slices(price, volume, params);

    let mut mat = make_uninit_matrix(2, len);
    {
        let warms = {
            let (_, _, _, sp, first, _) = buff_averages_prepare(&input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            vec![first + sp - 1, first + sp - 1]
        };
        init_matrix_prefixes(&mut mat, len, &warms);
    }

    let values = unsafe {
        let flat = core::slice::from_raw_parts_mut(mat.as_mut_ptr() as *mut f64, mat.len());
        let (fast_out, slow_out) = flat.split_at_mut(len);
        buff_averages_into_slices(fast_out, slow_out, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let ptr = mat.as_mut_ptr() as *mut f64;
        let len = mat.len();
        let cap = mat.capacity();
        core::mem::forget(mat);
        Vec::from_raw_parts(ptr, len, cap)
    };

    let js = BuffAveragesJsResult {
        values,
        rows: 2,
        cols: len,
    };
    serde_wasm_bindgen::to_value(&js)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn buff_averages_js(
    price: &[f64],
    volume: &[f64],
    fast_period: usize,
    slow_period: usize,
) -> Result<Vec<f64>, JsValue> {
    let len = price.len();
    let params = BuffAveragesParams {
        fast_period: Some(fast_period),
        slow_period: Some(slow_period),
    };
    let input = BuffAveragesInput::from_slices(price, volume, params);

    let mut mat = make_uninit_matrix(2, len);
    {
        let (_, _, _, sp, first, _) = buff_averages_prepare(&input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let warm = first + sp - 1;
        init_matrix_prefixes(&mut mat, len, &[warm, warm]);
    }

    let values = unsafe {
        let flat = core::slice::from_raw_parts_mut(mat.as_mut_ptr() as *mut f64, mat.len());
        let (fast_out, slow_out) = flat.split_at_mut(len);
        buff_averages_into_slices(fast_out, slow_out, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let ptr = mat.as_mut_ptr() as *mut f64;
        let len = mat.len();
        let cap = mat.capacity();
        core::mem::forget(mat);
        Vec::from_raw_parts(ptr, len, cap)
    };

    Ok(values)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn buff_averages_into(
    price_ptr: *const f64,
    volume_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    fast_period: usize,
    slow_period: usize,
) -> Result<(), JsValue> {
    if price_ptr.is_null() || volume_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to buff_averages_into",
        ));
    }

    unsafe {
        let price = core::slice::from_raw_parts(price_ptr, len);
        let volume = core::slice::from_raw_parts(volume_ptr, len);
        let (fast_out, slow_out) =
            core::slice::from_raw_parts_mut(out_ptr, 2 * len).split_at_mut(len);

        let params = BuffAveragesParams {
            fast_period: Some(fast_period),
            slow_period: Some(slow_period),
        };
        let input = BuffAveragesInput::from_slices(price, volume, params);

        buff_averages_into_slices(fast_out, slow_out, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn buff_averages_alloc(len: usize) -> *mut f64 {
    let mut v = Vec::<f64>::with_capacity(2 * len);
    let ptr = v.as_mut_ptr();
    core::mem::forget(v);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn buff_averages_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, 2 * len, 2 * len);
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct BuffAveragesBatchJsOutput {
    pub values: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
    pub fast_periods: Vec<usize>,
    pub slow_periods: Vec<usize>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = buff_averages_batch)]
pub fn buff_averages_batch_unified_js(
    price: &[f64],
    volume: &[f64],
    fast_range: Vec<usize>,
    slow_range: Vec<usize>,
) -> Result<JsValue, JsValue> {
    if fast_range.len() != 3 || slow_range.len() != 3 {
        return Err(JsValue::from_str(
            "fast_range and slow_range must each have 3 elements [start, end, step]",
        ));
    }

    let sweep = BuffAveragesBatchRange {
        fast_period: (fast_range[0], fast_range[1], fast_range[2]),
        slow_period: (slow_range[0], slow_range[1], slow_range[2]),
    };

    let out = buff_averages_batch_with_kernel(price, volume, &sweep, detect_best_batch_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let mut values = Vec::with_capacity(out.fast.len() + out.slow.len());
    values.extend_from_slice(&out.fast);
    values.extend_from_slice(&out.slow);

    let js = BuffAveragesBatchJsOutput {
        values,
        rows: out.rows * 2,
        cols: out.cols,
        fast_periods: out.combos.iter().map(|c| c.0).collect(),
        slow_periods: out.combos.iter().map(|c| c.1).collect(),
    };

    serde_wasm_bindgen::to_value(&js)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn buff_averages_batch_into(
    price_ptr: *const f64,
    volume_ptr: *const f64,
    out_fast_ptr: *mut f64,
    out_slow_ptr: *mut f64,
    len: usize,
    fast_start: usize,
    fast_end: usize,
    fast_step: usize,
    slow_start: usize,
    slow_end: usize,
    slow_step: usize,
) -> Result<usize, JsValue> {
    if price_ptr.is_null()
        || volume_ptr.is_null()
        || out_fast_ptr.is_null()
        || out_slow_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to buff_averages_batch_into",
        ));
    }
    unsafe {
        let price = core::slice::from_raw_parts(price_ptr, len);
        let volume = core::slice::from_raw_parts(volume_ptr, len);
        let sweep = BuffAveragesBatchRange {
            fast_period: (fast_start, fast_end, fast_step),
            slow_period: (slow_start, slow_end, slow_step),
        };

        let combos = {
            let rows = expand_grid_ba(&sweep).len();
            let fast_out = core::slice::from_raw_parts_mut(out_fast_ptr, rows * len);
            let slow_out = core::slice::from_raw_parts_mut(out_slow_ptr, rows * len);
            buff_averages_batch_inner_into(
                price,
                volume,
                &sweep,
                detect_best_batch_kernel(),
                fast_out,
                slow_out,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?
        };
        Ok(combos.len())
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
#[deprecated(
    since = "1.0.0",
    note = "For weight reuse patterns, use the fast/unsafe API with persistent buffers"
)]
pub struct BuffAveragesContext {
    fast_period: usize,
    slow_period: usize,
    kernel: Kernel,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
#[allow(deprecated)]
impl BuffAveragesContext {
    #[wasm_bindgen(constructor)]
    #[deprecated(
        since = "1.0.0",
        note = "For performance patterns, use the fast/unsafe API with persistent buffers"
    )]
    pub fn new(fast_period: usize, slow_period: usize) -> Result<BuffAveragesContext, JsValue> {
        if fast_period == 0 {
            return Err(JsValue::from_str(&format!(
                "Invalid fast period: {}",
                fast_period
            )));
        }
        if slow_period == 0 {
            return Err(JsValue::from_str(&format!(
                "Invalid slow period: {}",
                slow_period
            )));
        }

        Ok(BuffAveragesContext {
            fast_period,
            slow_period,
            kernel: Kernel::Auto,
        })
    }

    pub fn update_into(
        &self,
        price_ptr: *const f64,
        volume_ptr: *const f64,
        fast_out_ptr: *mut f64,
        slow_out_ptr: *mut f64,
        len: usize,
    ) -> Result<(), JsValue> {
        if len < self.slow_period {
            return Err(JsValue::from_str("Data length less than slow period"));
        }

        if price_ptr.is_null()
            || volume_ptr.is_null()
            || fast_out_ptr.is_null()
            || slow_out_ptr.is_null()
        {
            return Err(JsValue::from_str("null pointer passed to update_into"));
        }

        unsafe {
            let price = std::slice::from_raw_parts(price_ptr, len);
            let volume = std::slice::from_raw_parts(volume_ptr, len);
            let fast_out = std::slice::from_raw_parts_mut(fast_out_ptr, len);
            let slow_out = std::slice::from_raw_parts_mut(slow_out_ptr, len);

            let params = BuffAveragesParams {
                fast_period: Some(self.fast_period),
                slow_period: Some(self.slow_period),
            };
            let input = BuffAveragesInput::from_slices(price, volume, params);

            let needs_temp = price_ptr == fast_out_ptr
                || price_ptr == slow_out_ptr
                || volume_ptr == fast_out_ptr
                || volume_ptr == slow_out_ptr;

            if needs_temp {
                let mut temp_fast = vec![0.0; len];
                let mut temp_slow = vec![0.0; len];

                buff_averages_into_slices(&mut temp_fast, &mut temp_slow, &input, self.kernel)
                    .map_err(|e| JsValue::from_str(&e.to_string()))?;

                fast_out.copy_from_slice(&temp_fast);
                slow_out.copy_from_slice(&temp_slow);
            } else {
                buff_averages_into_slices(fast_out, slow_out, &input, self.kernel)
                    .map_err(|e| JsValue::from_str(&e.to_string()))?;
            }
        }

        Ok(())
    }

    pub fn get_warmup_period(&self) -> usize {
        self.slow_period - 1
    }

    #[wasm_bindgen]
    pub fn compute(&self, price: &[f64], volume: &[f64]) -> Result<Vec<f64>, JsValue> {
        let params = BuffAveragesParams {
            fast_period: Some(self.fast_period),
            slow_period: Some(self.slow_period),
        };
        let input = BuffAveragesInput::from_slices(price, volume, params);
        let result = buff_averages_with_kernel(&input, self.kernel)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let mut output = Vec::with_capacity(price.len() * 2);
        output.extend_from_slice(&result.fast_buff);
        output.extend_from_slice(&result.slow_buff);
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    #[cfg(feature = "proptest")]
    use proptest::prelude::*;
    use std::error::Error;

    fn check_buff_averages_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input =
            BuffAveragesInput::from_candles(&candles, "close", BuffAveragesParams::default());
        let result = buff_averages_with_kernel(&input, kernel)?;

        let expected_fast = [
            58740.30855637,
            59132.28418702,
            59309.76658172,
            59266.10492431,
            59194.11908892,
        ];

        let expected_slow = [
            59209.26229392,
            59201.87047432,
            59217.15739355,
            59195.74527194,
            59196.26139533,
        ];

        let start = result.fast_buff.len().saturating_sub(6);

        for (i, (&fast_val, &slow_val)) in result.fast_buff[start..]
            .iter()
            .take(5)
            .zip(result.slow_buff[start..].iter())
            .enumerate()
        {
            let fast_diff = (fast_val - expected_fast[i]).abs();
            let slow_diff = (slow_val - expected_slow[i]).abs();
            assert!(
                fast_diff < 1e-3,
                "[{}] Buff Averages {:?} fast mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                fast_val,
                expected_fast[i]
            );
            assert!(
                slow_diff < 1e-3,
                "[{}] Buff Averages {:?} slow mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                slow_val,
                expected_slow[i]
            );
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_buff_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let out = buff_averages_with_kernel(&BuffAveragesInput::with_default_candles(&c), kernel)?;

        for (i, &v) in out.fast_buff.iter().enumerate() {
            if v.is_nan() {
                continue;
            }
            let b = v.to_bits();
            assert!(
                b != 0x11111111_11111111 && b != 0x22222222_22222222 && b != 0x33333333_33333333,
                "[{}] poison in fast at {}: {:#x}",
                test_name,
                i,
                b
            );
        }

        for (i, &v) in out.slow_buff.iter().enumerate() {
            if v.is_nan() {
                continue;
            }
            let b = v.to_bits();
            assert!(
                b != 0x11111111_11111111 && b != 0x22222222_22222222 && b != 0x33333333_33333333,
                "[{}] poison in slow at {}: {:#x}",
                test_name,
                i,
                b
            );
        }
        Ok(())
    }

    fn check_buff_nan_prefix(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let input = BuffAveragesInput::with_default_candles(&c);

        let (price, _, _, slow_p, first, _) = buff_averages_prepare(&input, kernel)?;
        let warm = first + slow_p - 1;

        let out = buff_averages_with_kernel(&input, kernel)?;

        assert!(
            out.fast_buff[..warm].iter().all(|x| x.is_nan()),
            "[{}] fast warmup not NaN",
            test_name
        );
        assert!(
            out.slow_buff[..warm].iter().all(|x| x.is_nan()),
            "[{}] slow warmup not NaN",
            test_name
        );
        assert!(
            out.fast_buff[warm..].iter().all(|x| x.is_finite()),
            "[{}] fast post-warm has NaN",
            test_name
        );
        assert!(
            out.slow_buff[warm..].iter().all(|x| x.is_finite()),
            "[{}] slow post-warm has NaN",
            test_name
        );
        Ok(())
    }

    fn check_buff_averages_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = BuffAveragesParams {
            fast_period: None,
            slow_period: None,
        };
        let input = BuffAveragesInput::from_candles(&candles, "close", default_params);
        let output = buff_averages_with_kernel(&input, kernel)?;
        assert_eq!(output.fast_buff.len(), candles.close.len());
        assert_eq!(output.slow_buff.len(), candles.close.len());

        Ok(())
    }

    fn check_buff_averages_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = BuffAveragesInput::with_default_candles(&candles);
        match input.data {
            BuffAveragesData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected BuffAveragesData::Candles"),
        }
        let output = buff_averages_with_kernel(&input, kernel)?;
        assert_eq!(output.fast_buff.len(), candles.close.len());
        assert_eq!(output.slow_buff.len(), candles.close.len());

        Ok(())
    }

    fn check_buff_averages_zero_period(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let volume_data = [100.0, 200.0, 300.0];
        let params = BuffAveragesParams {
            fast_period: Some(0),
            slow_period: Some(10),
        };
        let input = BuffAveragesInput::from_slices(&input_data, &volume_data, params);
        let res = buff_averages_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Buff Averages should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_buff_averages_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let volume_small = [100.0, 200.0, 300.0];
        let params = BuffAveragesParams {
            fast_period: Some(5),
            slow_period: Some(10),
        };
        let input = BuffAveragesInput::from_slices(&data_small, &volume_small, params);
        let res = buff_averages_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Buff Averages should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_buff_averages_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let single_volume = [100.0];
        let params = BuffAveragesParams::default();
        let input = BuffAveragesInput::from_slices(&single_point, &single_volume, params);
        let res = buff_averages_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Buff Averages should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_buff_averages_empty_input(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let empty: [f64; 0] = [];
        let params = BuffAveragesParams::default();
        let input = BuffAveragesInput::from_slices(&empty, &empty, params);
        let res = buff_averages_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Buff Averages should fail with empty input",
            test_name
        );
        Ok(())
    }

    fn check_buff_averages_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let nan_data = [f64::NAN, f64::NAN, f64::NAN];
        let nan_volume = [f64::NAN, f64::NAN, f64::NAN];
        let params = BuffAveragesParams::default();
        let input = BuffAveragesInput::from_slices(&nan_data, &nan_volume, params);
        let res = buff_averages_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Buff Averages should fail with all NaN values",
            test_name
        );
        Ok(())
    }

    fn check_buff_averages_mismatched_lengths(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let price_data = [10.0, 20.0, 30.0];
        let volume_data = [100.0, 200.0];
        let params = BuffAveragesParams::default();
        let input = BuffAveragesInput::from_slices(&price_data, &volume_data, params);
        let res = buff_averages_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Buff Averages should fail with mismatched data lengths",
            test_name
        );
        Ok(())
    }

    fn check_buff_averages_missing_volume(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let price_data = [
            10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0,
            140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0,
        ];
        let params = BuffAveragesParams::default();

        let input = BuffAveragesInput {
            data: BuffAveragesData::Slice(&price_data),
            params,
            volume: None,
        };

        let res = buff_averages_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Buff Averages should fail with missing volume data",
            test_name
        );
        Ok(())
    }

    fn check_buff_averages_batch_single(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let range = BuffAveragesBatchRange {
            fast_period: (5, 5, 0),
            slow_period: (20, 20, 0),
        };

        let price = source_type(&candles, "close");
        let volume = &candles.volume;

        let batch_result = buff_averages_batch_with_kernel(price, volume, &range, kernel)?;

        let single_input =
            BuffAveragesInput::from_slices(price, volume, BuffAveragesParams::default());
        let single_result = buff_averages_with_kernel(&single_input, kernel)?;

        assert_eq!(batch_result.rows, 1, "[{}] Expected 1 row", test_name);
        assert_eq!(
            batch_result.combos.len(),
            1,
            "[{}] Expected 1 combination",
            test_name
        );

        for i in 0..price.len() {
            let batch_fast = batch_result.fast[i];
            let single_fast = single_result.fast_buff[i];
            if batch_fast.is_finite() && single_fast.is_finite() {
                assert!(
                    (batch_fast - single_fast).abs() < 1e-10,
                    "[{}] Fast mismatch at {}: batch={}, single={}",
                    test_name,
                    i,
                    batch_fast,
                    single_fast
                );
            }

            let batch_slow = batch_result.slow[i];
            let single_slow = single_result.slow_buff[i];
            if batch_slow.is_finite() && single_slow.is_finite() {
                assert!(
                    (batch_slow - single_slow).abs() < 1e-10,
                    "[{}] Slow mismatch at {}: batch={}, single={}",
                    test_name,
                    i,
                    batch_slow,
                    single_slow
                );
            }
        }
        Ok(())
    }

    fn check_buff_averages_batch_grid(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let range = BuffAveragesBatchRange {
            fast_period: (3, 7, 2),
            slow_period: (18, 22, 2),
        };

        let price = source_type(&candles, "close");
        let volume = &candles.volume;

        let result = buff_averages_batch_with_kernel(price, volume, &range, kernel)?;

        assert_eq!(result.rows, 9, "[{}] Expected 9 rows", test_name);
        assert_eq!(
            result.cols,
            candles.close.len(),
            "[{}] Cols mismatch",
            test_name
        );
        assert_eq!(
            result.combos.len(),
            9,
            "[{}] Expected 9 combinations",
            test_name
        );
        assert_eq!(
            result.fast.len(),
            9 * candles.close.len(),
            "[{}] Fast size mismatch",
            test_name
        );
        assert_eq!(
            result.slow.len(),
            9 * candles.close.len(),
            "[{}] Slow size mismatch",
            test_name
        );

        let expected_combos = vec![
            (3, 18),
            (3, 20),
            (3, 22),
            (5, 18),
            (5, 20),
            (5, 22),
            (7, 18),
            (7, 20),
            (7, 22),
        ];
        assert_eq!(
            result.combos, expected_combos,
            "[{}] Combinations mismatch",
            test_name
        );

        Ok(())
    }

    fn check_buff_averages_batch_empty(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let price = [];
        let volume = [];

        let range = BuffAveragesBatchRange {
            fast_period: (5, 10, 1),
            slow_period: (15, 20, 1),
        };

        let res = buff_averages_batch_with_kernel(&price, &volume, &range, kernel);
        assert!(
            res.is_err(),
            "[{}] Batch should fail with empty input",
            test_name
        );
        Ok(())
    }

    fn check_buff_averages_batch_parallel(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let range = BuffAveragesBatchRange {
            fast_period: (3, 7, 2),
            slow_period: (18, 22, 2),
        };

        let price = source_type(&candles, "close");
        let volume = &candles.volume;

        let seq_result = buff_averages_batch_with_kernel(price, volume, &range, kernel)?;

        let par_result = buff_averages_batch_par_slice(price, volume, &range, kernel)?;

        assert_eq!(
            seq_result.rows, par_result.rows,
            "[{}] Row count mismatch",
            test_name
        );
        assert_eq!(
            seq_result.cols, par_result.cols,
            "[{}] Col count mismatch",
            test_name
        );
        assert_eq!(
            seq_result.combos, par_result.combos,
            "[{}] Combos mismatch",
            test_name
        );

        for i in 0..seq_result.fast.len() {
            let seq_fast = seq_result.fast[i];
            let par_fast = par_result.fast[i];
            if seq_fast.is_finite() && par_fast.is_finite() {
                assert!(
                    (seq_fast - par_fast).abs() < 1e-10,
                    "[{}] Fast parallel mismatch at {}: seq={}, par={}",
                    test_name,
                    i,
                    seq_fast,
                    par_fast
                );
            } else {
                assert_eq!(
                    seq_fast.is_nan(),
                    par_fast.is_nan(),
                    "[{}] Fast NaN mismatch at {}",
                    test_name,
                    i
                );
            }
        }

        for i in 0..seq_result.slow.len() {
            let seq_slow = seq_result.slow[i];
            let par_slow = par_result.slow[i];
            if seq_slow.is_finite() && par_slow.is_finite() {
                assert!(
                    (seq_slow - par_slow).abs() < 1e-10,
                    "[{}] Slow parallel mismatch at {}: seq={}, par={}",
                    test_name,
                    i,
                    seq_slow,
                    par_slow
                );
            } else {
                assert_eq!(
                    seq_slow.is_nan(),
                    par_slow.is_nan(),
                    "[{}] Slow NaN mismatch at {}",
                    test_name,
                    i
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_buff_averages_stream() -> Result<(), Box<dyn Error>> {
        let params = BuffAveragesParams::default();
        let mut stream = BuffAveragesStream::try_new(params)?;

        let test_data = vec![
            (100.0, 1000.0),
            (110.0, 1100.0),
            (120.0, 1200.0),
            (130.0, 1300.0),
            (140.0, 1400.0),
            (150.0, 1500.0),
            (160.0, 1600.0),
            (170.0, 1700.0),
            (180.0, 1800.0),
            (190.0, 1900.0),
            (200.0, 2000.0),
            (210.0, 2100.0),
            (220.0, 2200.0),
            (230.0, 2300.0),
            (240.0, 2400.0),
            (250.0, 2500.0),
            (260.0, 2600.0),
            (270.0, 2700.0),
            (280.0, 2800.0),
            (290.0, 2900.0),
            (300.0, 3000.0),
        ];

        let mut results = Vec::new();
        for (price, volume) in test_data {
            if let Some(result) = stream.update(price, volume) {
                results.push(result);
            }
        }

        assert!(!results.is_empty(), "Stream should produce results");

        Ok(())
    }

    macro_rules! generate_buff_averages_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $(
                    #[test]
                    fn [<$test_fn _scalar>]() {
                        let _ = $test_fn(stringify!([<$test_fn _scalar>]), Kernel::Scalar);
                    }
                )*

                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $(
                    #[test]
                    fn [<$test_fn _avx2>]() {
                        let _ = $test_fn(stringify!([<$test_fn _avx2>]), Kernel::Avx2);
                    }

                    #[test]
                    fn [<$test_fn _avx512>]() {
                        let _ = $test_fn(stringify!([<$test_fn _avx512>]), Kernel::Avx512);
                    }
                )*
            }
        };
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

                #[test]
                fn [<$fn_name _auto>]() {
                    let _ = $fn_name(stringify!([<$fn_name _auto>]), Kernel::Auto);
                }
            }
        };
    }

    generate_buff_averages_tests!(
        check_buff_averages_accuracy,
        check_buff_averages_partial_params,
        check_buff_averages_default_candles,
        check_buff_averages_zero_period,
        check_buff_averages_period_exceeds_length,
        check_buff_averages_very_small_dataset,
        check_buff_averages_empty_input,
        check_buff_averages_all_nan,
        check_buff_averages_mismatched_lengths,
        check_buff_averages_missing_volume,
        check_buff_nan_prefix
    );

    #[cfg(debug_assertions)]
    generate_buff_averages_tests!(check_buff_no_poison);

    gen_batch_tests!(check_buff_averages_batch_single);
    gen_batch_tests!(check_buff_averages_batch_grid);
    gen_batch_tests!(check_buff_averages_batch_empty);
    gen_batch_tests!(check_buff_averages_batch_parallel);

    #[cfg(feature = "proptest")]
    proptest! {
        #[test]
        fn prop_buff_averages_length_preserved(
            len in 50usize..100,
            fast_period in 2usize..10,
            slow_period in 11usize..30
        ) {

            let data: Vec<f64> = (0..len).map(|i| (i as f64 + 1.0) * 10.0).collect();
            let volume: Vec<f64> = (0..len).map(|i| (i as f64 + 1.0) * 100.0).collect();

            prop_assume!(data.len() > slow_period);

            let params = BuffAveragesParams {
                fast_period: Some(fast_period),
                slow_period: Some(slow_period),
            };
            let input = BuffAveragesInput::from_slices(&data, &volume, params);

            if let Ok(output) = buff_averages(&input) {
                prop_assert_eq!(output.fast_buff.len(), data.len());
                prop_assert_eq!(output.slow_buff.len(), data.len());
            }
        }

        #[test]
        fn prop_buff_averages_nan_handling(
            len in 50usize..100
        ) {

            let mut data: Vec<f64> = (0..len).map(|i| (i as f64 + 1.0) * 10.0).collect();
            let mut volume: Vec<f64> = (0..len).map(|i| (i as f64 + 1.0) * 100.0).collect();


            for i in (0..5).map(|x| x * 10) {
                if i < data.len() {
                    data[i] = f64::NAN;
                    volume[i] = f64::NAN;
                }
            }

            let params = BuffAveragesParams::default();
            let input = BuffAveragesInput::from_slices(&data, &volume, params);


            let _ = buff_averages(&input);
        }
    }

    #[test]
    fn test_buff_averages_into_matches_api() -> Result<(), Box<dyn Error>> {
        let len = 256usize;
        let price: Vec<f64> = (0..len).map(|i| (i as f64) * 1.5 + 10.0).collect();
        let volume: Vec<f64> = (0..len).map(|i| (i as f64) * 2.0 + 100.0).collect();

        let params = BuffAveragesParams::default();
        let input = BuffAveragesInput::from_slices(&price, &volume, params);

        let base = buff_averages(&input)?;

        let mut out_fast = vec![0.0; len];
        let mut out_slow = vec![0.0; len];
        super::buff_averages_into(&input, &mut out_fast, &mut out_slow)?;

        fn eq_nan_or_close(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b) || ((a - b).abs() <= 1e-12)
        }

        assert_eq!(base.fast_buff.len(), out_fast.len());
        assert_eq!(base.slow_buff.len(), out_slow.len());
        for i in 0..len {
            assert!(
                eq_nan_or_close(base.fast_buff[i], out_fast[i]),
                "fast mismatch at {}: {} vs {}",
                i,
                base.fast_buff[i],
                out_fast[i]
            );
            assert!(
                eq_nan_or_close(base.slow_buff[i], out_slow[i]),
                "slow mismatch at {}: {} vs {}",
                i,
                base.slow_buff[i],
                out_slow[i]
            );
        }
        Ok(())
    }
}
