#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::{
    cuda_available,
    moving_averages::{alma_wrapper::DeviceArrayF32, CudaHwma},
};
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
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::memory::DeviceBuffer;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use serde::{Deserialize, Serialize};
use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen::prelude::*;

#[cfg(all(feature = "python", feature = "cuda"))]
use cust::context::Context;
#[cfg(all(feature = "python", feature = "cuda"))]
use std::sync::Arc;

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyo3::pyclass(module = "ta_indicators.cuda", name = "HwmaDeviceArrayF32", unsendable)]
pub struct HwmaDeviceArrayF32Py {
    pub(crate) inner: DeviceArrayF32,
    pub(crate) _ctx: Arc<Context>,
    pub(crate) device_id: u32,
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyo3::pymethods]
impl HwmaDeviceArrayF32Py {
    #[getter]
    fn __cuda_array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        let itemsize = std::mem::size_of::<f32>();
        d.set_item("shape", (self.inner.rows, self.inner.cols))?;
        d.set_item("typestr", "<f4")?;
        d.set_item("strides", (self.inner.cols * itemsize, itemsize))?;
        d.set_item("data", (self.inner.device_ptr() as usize, false))?;

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
        stream: Option<PyObject>,
        max_version: Option<PyObject>,
        dl_device: Option<PyObject>,
        copy: Option<PyObject>,
    ) -> PyResult<pyo3::PyObject> {
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

impl<'a> AsRef<[f64]> for HwmaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            HwmaData::Slice(slice) => slice,
            HwmaData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum HwmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct HwmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct HwmaParams {
    pub na: Option<f64>,
    pub nb: Option<f64>,
    pub nc: Option<f64>,
}

impl Default for HwmaParams {
    fn default() -> Self {
        Self {
            na: Some(0.2),
            nb: Some(0.1),
            nc: Some(0.1),
        }
    }
}

#[derive(Debug, Clone)]
pub struct HwmaInput<'a> {
    pub data: HwmaData<'a>,
    pub params: HwmaParams,
}

impl<'a> HwmaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: HwmaParams) -> Self {
        Self {
            data: HwmaData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: HwmaParams) -> Self {
        Self {
            data: HwmaData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", HwmaParams::default())
    }
    #[inline]
    pub fn get_na(&self) -> f64 {
        self.params.na.unwrap_or(0.2)
    }
    #[inline]
    pub fn get_nb(&self) -> f64 {
        self.params.nb.unwrap_or(0.1)
    }
    #[inline]
    pub fn get_nc(&self) -> f64 {
        self.params.nc.unwrap_or(0.1)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct HwmaBuilder {
    na: Option<f64>,
    nb: Option<f64>,
    nc: Option<f64>,
    kernel: Kernel,
}

impl Default for HwmaBuilder {
    fn default() -> Self {
        Self {
            na: None,
            nb: None,
            nc: None,
            kernel: Kernel::Auto,
        }
    }
}

impl HwmaBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn na(mut self, x: f64) -> Self {
        self.na = Some(x);
        self
    }
    #[inline(always)]
    pub fn nb(mut self, x: f64) -> Self {
        self.nb = Some(x);
        self
    }
    #[inline(always)]
    pub fn nc(mut self, x: f64) -> Self {
        self.nc = Some(x);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<HwmaOutput, HwmaError> {
        let p = HwmaParams {
            na: self.na,
            nb: self.nb,
            nc: self.nc,
        };
        let i = HwmaInput::from_candles(c, "close", p);
        hwma_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<HwmaOutput, HwmaError> {
        let p = HwmaParams {
            na: self.na,
            nb: self.nb,
            nc: self.nc,
        };
        let i = HwmaInput::from_slice(d, p);
        hwma_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<HwmaStream, HwmaError> {
        let p = HwmaParams {
            na: self.na,
            nb: self.nb,
            nc: self.nc,
        };
        HwmaStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum HwmaError {
    #[error("hwma: calculation received empty data array.")]
    EmptyData,
    #[error("hwma: All values in input data are NaN.")]
    AllValuesNaN,
    #[error("hwma: Parameters (na, nb, nc) must be in (0,1). Received: na={na}, nb={nb}, nc={nc}")]
    InvalidParams { na: f64, nb: f64, nc: f64 },
    #[error("hwma: Invalid output buffer size: expected = {expected}, actual = {actual}")]
    InvalidOutputBuffer { expected: usize, actual: usize },
    #[error("hwma: invalid output length, expected {expected}, got {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("hwma: invalid batch range: start={start}, end={end}, step={step}")]
    InvalidRange { start: f64, end: f64, step: f64 },
    #[error("hwma: invalid kernel for batch API: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("hwma: integer overflow during size computation: {0}")]
    IntegerOverflow(&'static str),
    #[error("hwma: invalid period {period} for data_len {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("hwma: not enough valid data: needed {needed}, valid {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("hwma: calculation received empty input data.")]
    EmptyInputData,
}

#[inline]
pub fn hwma(input: &HwmaInput) -> Result<HwmaOutput, HwmaError> {
    hwma_with_kernel(input, Kernel::Auto)
}

pub fn hwma_with_kernel(input: &HwmaInput, kernel: Kernel) -> Result<HwmaOutput, HwmaError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(HwmaError::EmptyData);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(HwmaError::AllValuesNaN)?;
    let na = input.get_na();
    let nb = input.get_nb();
    let nc = input.get_nc();

    if !na.is_finite() || !nb.is_finite() || !nc.is_finite() {
        return Err(HwmaError::InvalidParams { na, nb, nc });
    }
    if !(na > 0.0 && na < 1.0 && nb > 0.0 && nb < 1.0 && nc > 0.0 && nc < 1.0) {
        return Err(HwmaError::InvalidParams { na, nb, nc });
    }

    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    let mut out = alloc_with_nan_prefix(len, first);
    unsafe {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            if matches!(chosen, Kernel::Scalar | Kernel::ScalarBatch) {
                hwma_simd128(data, na, nb, nc, first, &mut out);
                return Ok(HwmaOutput { values: out });
            }
        }

        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => hwma_scalar(data, na, nb, nc, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => hwma_avx2(data, na, nb, nc, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => hwma_avx512(data, na, nb, nc, first, &mut out),
            _ => hwma_scalar(data, na, nb, nc, first, &mut out),
        }
    }

    Ok(HwmaOutput { values: out })
}

#[inline]
pub fn hwma_into(input: &HwmaInput, out: &mut [f64]) -> Result<(), HwmaError> {
    hwma_with_kernel_into(input, Kernel::Auto, out)
}

pub fn hwma_with_kernel_into(
    input: &HwmaInput,
    kernel: Kernel,
    out: &mut [f64],
) -> Result<(), HwmaError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(HwmaError::EmptyData);
    }

    if out.len() != len {
        return Err(HwmaError::OutputLengthMismatch {
            expected: len,
            got: out.len(),
        });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(HwmaError::AllValuesNaN)?;
    let na = input.get_na();
    let nb = input.get_nb();
    let nc = input.get_nc();

    if !na.is_finite() || !nb.is_finite() || !nc.is_finite() {
        return Err(HwmaError::InvalidParams { na, nb, nc });
    }
    if !(na > 0.0 && na < 1.0 && nb > 0.0 && nb < 1.0 && nc > 0.0 && nc < 1.0) {
        return Err(HwmaError::InvalidParams { na, nb, nc });
    }

    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                {
                    hwma_simd128(data, na, nb, nc, first, out);
                }
                #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
                {
                    hwma_scalar(data, na, nb, nc, first, out);
                }
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => hwma_avx2(data, na, nb, nc, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => hwma_avx512(data, na, nb, nc, first, out),
            _ => {
                #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                {
                    hwma_simd128(data, na, nb, nc, first, out);
                }
                #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
                {
                    hwma_scalar(data, na, nb, nc, first, out);
                }
            }
        }
    }

    let qnan = f64::from_bits(0x7ff8_0000_0000_0000);
    for v in &mut out[..first] {
        *v = qnan;
    }

    Ok(())
}

#[inline(always)]
pub fn hwma_scalar(data: &[f64], na: f64, nb: f64, nc: f64, first_valid: usize, out: &mut [f64]) {
    debug_assert_eq!(data.len(), out.len());
    let len = data.len();
    if first_valid >= len {
        return;
    }

    const HALF: f64 = 0.5;
    let one_m_na = 1.0 - na;
    let one_m_nb = 1.0 - nb;
    let one_m_nc = 1.0 - nc;

    let mut f = unsafe { *data.get_unchecked(first_valid) };
    let mut v = 0.0;
    let mut a = 0.0;

    unsafe {
        let mut dp = data.as_ptr().add(first_valid);
        let mut op = out.as_mut_ptr().add(first_valid);
        let end = out.as_mut_ptr().add(len);

        while op.add(1) < end {
            let x0 = *dp;

            let s_prev = HALF.mul_add(a, f + v);

            let f_new = one_m_na.mul_add(s_prev, na * x0);

            let sum_va = v + a;
            let v_new = nb.mul_add(f_new - f, one_m_nb * sum_va);

            let a_new = nc.mul_add(v_new - v, one_m_nc * a);

            let s_new = HALF.mul_add(a_new, f_new + v_new);
            *op = s_new;

            let x1 = *dp.add(1);

            let f2 = one_m_na.mul_add(s_new, na * x1);
            let v2 = nb.mul_add(f2 - f_new, one_m_nb * (v_new + a_new));
            let a2 = nc.mul_add(v2 - v_new, one_m_nc * a_new);
            let s2 = HALF.mul_add(a2, f2 + v2);
            *op.add(1) = s2;

            f = f2;
            v = v2;
            a = a2;
            dp = dp.add(2);
            op = op.add(2);
        }

        if op < end {
            let x = *dp;
            let s_prev = HALF.mul_add(a, f + v);
            let f_new = one_m_na.mul_add(s_prev, na * x);
            let v_new = nb.mul_add(f_new - f, one_m_nb * (v + a));
            let a_new = nc.mul_add(v_new - v, one_m_nc * a);
            *op = HALF.mul_add(a_new, f_new + v_new);
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
unsafe fn hwma_simd128(data: &[f64], na: f64, nb: f64, nc: f64, first: usize, out: &mut [f64]) {
    hwma_scalar(data, na, nb, nc, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn hwma_avx2(data: &[f64], na: f64, nb: f64, nc: f64, first: usize, out: &mut [f64]) {
    debug_assert_eq!(data.len(), out.len());
    let len = data.len();
    if first >= len {
        return;
    }

    const HALF: f64 = 0.5;
    let one_m_na = 1.0 - na;
    let one_m_nb = 1.0 - nb;
    let one_m_nc = 1.0 - nc;

    let mut f = *data.get_unchecked(first);
    let mut v = 0.0;
    let mut a = 0.0;

    let mut dp = data.as_ptr().add(first);
    let mut op = out.as_mut_ptr().add(first);
    let end = out.as_mut_ptr().add(len);

    while op.add(1) < end {
        let x0 = *dp;
        let s_prev = HALF.mul_add(a, f + v);
        let f_new = one_m_na.mul_add(s_prev, na * x0);
        let v_new = nb.mul_add(f_new - f, one_m_nb * (v + a));
        let a_new = nc.mul_add(v_new - v, one_m_nc * a);
        let s_new = HALF.mul_add(a_new, f_new + v_new);
        *op = s_new;

        let x1 = *dp.add(1);
        let f2 = one_m_na.mul_add(s_new, na * x1);
        let v2 = nb.mul_add(f2 - f_new, one_m_nb * (v_new + a_new));
        let a2 = nc.mul_add(v2 - v_new, one_m_nc * a_new);
        *op.add(1) = HALF.mul_add(a2, f2 + v2);

        f = f2;
        v = v2;
        a = a2;
        dp = dp.add(2);
        op = op.add(2);
    }

    if op < end {
        let x = *dp;
        let s_prev = HALF.mul_add(a, f + v);
        let f_new = one_m_na.mul_add(s_prev, na * x);
        let v_new = nb.mul_add(f_new - f, one_m_nb * (v + a));
        let a_new = nc.mul_add(v_new - v, one_m_nc * a);
        *op = HALF.mul_add(a_new, f_new + v_new);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
#[inline]
pub unsafe fn hwma_avx512(data: &[f64], na: f64, nb: f64, nc: f64, first: usize, out: &mut [f64]) {
    hwma_avx2(data, na, nb, nc, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn hwma_avx512_short(
    data: &[f64],
    na: f64,
    nb: f64,
    nc: f64,
    first_valid: usize,
    out: &mut [f64],
) {
    hwma_scalar(data, na, nb, nc, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn hwma_avx512_long(
    data: &[f64],
    na: f64,
    nb: f64,
    nc: f64,
    first_valid: usize,
    out: &mut [f64],
) {
    hwma_scalar(data, na, nb, nc, first_valid, out)
}

#[derive(Debug, Clone)]
pub struct HwmaStream {
    na: f64,
    nb: f64,
    nc: f64,

    one_m_na: f64,
    one_m_nb: f64,
    one_m_nc: f64,

    last_f: f64,
    last_v: f64,
    last_a: f64,
    last_s: f64,

    filled: bool,
}

impl HwmaStream {
    pub fn try_new(params: HwmaParams) -> Result<Self, HwmaError> {
        let na = params.na.unwrap_or(0.2);
        let nb = params.nb.unwrap_or(0.1);
        let nc = params.nc.unwrap_or(0.1);
        if !na.is_finite() || !nb.is_finite() || !nc.is_finite() {
            return Err(HwmaError::InvalidParams { na, nb, nc });
        }
        if !(na > 0.0 && na < 1.0 && nb > 0.0 && nb < 1.0 && nc > 0.0 && nc < 1.0) {
            return Err(HwmaError::InvalidParams { na, nb, nc });
        }
        Ok(Self {
            na,
            nb,
            nc,
            one_m_na: 1.0 - na,
            one_m_nb: 1.0 - nb,
            one_m_nc: 1.0 - nc,
            last_f: f64::NAN,
            last_v: 0.0,
            last_a: 0.0,
            last_s: f64::NAN,
            filled: false,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        if !self.filled {
            self.last_f = value;
            self.last_v = 0.0;
            self.last_a = 0.0;
            self.last_s = value;
            self.filled = true;
            return Some(value);
        }

        let s_prev = self.last_s;

        let f_new = self.one_m_na.mul_add(s_prev, self.na * value);

        let sum_va = self.last_v + self.last_a;
        let v_new = self.nb.mul_add(f_new - self.last_f, self.one_m_nb * sum_va);

        let a_new = self
            .nc
            .mul_add(v_new - self.last_v, self.one_m_nc * self.last_a);

        let s_new = 0.5f64.mul_add(a_new, f_new + v_new);

        self.last_f = f_new;
        self.last_v = v_new;
        self.last_a = a_new;
        self.last_s = s_new;

        Some(s_new)
    }

    #[inline(always)]
    pub fn init_once(&mut self, first_value: f64) -> f64 {
        self.last_f = first_value;
        self.last_v = 0.0;
        self.last_a = 0.0;
        self.last_s = first_value;
        self.filled = true;
        first_value
    }

    #[inline(always)]
    pub fn update_unchecked(&mut self, value: f64) -> f64 {
        debug_assert!(self.filled);
        let f_new = self.one_m_na.mul_add(self.last_s, self.na * value);
        let sum_va = self.last_v + self.last_a;
        let v_new = self.nb.mul_add(f_new - self.last_f, self.one_m_nb * sum_va);
        let a_new = self
            .nc
            .mul_add(v_new - self.last_v, self.one_m_nc * self.last_a);
        let s_new = 0.5f64.mul_add(a_new, f_new + v_new);
        self.last_f = f_new;
        self.last_v = v_new;
        self.last_a = a_new;
        self.last_s = s_new;
        s_new
    }

    #[inline(always)]
    pub fn predict_next(&self, x: f64) -> f64 {
        if !self.filled {
            return x;
        }
        let f_new = self.one_m_na.mul_add(self.last_s, self.na * x);
        let v_new = self.nb.mul_add(
            f_new - self.last_f,
            self.one_m_nb * (self.last_v + self.last_a),
        );
        let a_new = self
            .nc
            .mul_add(v_new - self.last_v, self.one_m_nc * self.last_a);
        0.5f64.mul_add(a_new, f_new + v_new)
    }
}

#[derive(Clone, Debug)]
pub struct HwmaBatchRange {
    pub na: (f64, f64, f64),
    pub nb: (f64, f64, f64),
    pub nc: (f64, f64, f64),
}

impl Default for HwmaBatchRange {
    fn default() -> Self {
        Self {
            na: (0.2, 0.449, 0.001),
            nb: (0.1, 0.1, 0.0),
            nc: (0.1, 0.1, 0.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct HwmaBatchBuilder {
    range: HwmaBatchRange,
    kernel: Kernel,
}

impl HwmaBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline]
    pub fn na_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.na = (start, end, step);
        self
    }
    #[inline]
    pub fn nb_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.nb = (start, end, step);
        self
    }
    #[inline]
    pub fn nc_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.nc = (start, end, step);
        self
    }
    #[inline]
    pub fn na_static(mut self, v: f64) -> Self {
        self.range.na = (v, v, 0.0);
        self
    }
    #[inline]
    pub fn nb_static(mut self, v: f64) -> Self {
        self.range.nb = (v, v, 0.0);
        self
    }
    #[inline]
    pub fn nc_static(mut self, v: f64) -> Self {
        self.range.nc = (v, v, 0.0);
        self
    }

    pub fn apply_slice(self, data: &[f64]) -> Result<HwmaBatchOutput, HwmaError> {
        hwma_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<HwmaBatchOutput, HwmaError> {
        HwmaBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<HwmaBatchOutput, HwmaError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<HwmaBatchOutput, HwmaError> {
        HwmaBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn hwma_batch_with_kernel(
    data: &[f64],
    sweep: &HwmaBatchRange,
    k: Kernel,
) -> Result<HwmaBatchOutput, HwmaError> {
    match k {
        Kernel::Scalar | Kernel::Avx2 | Kernel::Avx512 => {
            return Err(HwmaError::InvalidKernelForBatch(k));
        }
        _ => {}
    }

    let simd = match k {
        Kernel::Auto => Kernel::Scalar,
        Kernel::Avx512Batch | Kernel::Avx512 => Kernel::Avx512,
        Kernel::Avx2Batch | Kernel::Avx2 => Kernel::Avx2,
        Kernel::ScalarBatch | Kernel::Scalar => Kernel::Scalar,
        _ => Kernel::Scalar,
    };
    hwma_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct HwmaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<HwmaParams>,
    pub rows: usize,
    pub cols: usize,
}
impl HwmaBatchOutput {
    pub fn row_for_params(&self, p: &HwmaParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            (c.na.unwrap_or(0.2) - p.na.unwrap_or(0.2)).abs() < 1e-12
                && (c.nb.unwrap_or(0.1) - p.nb.unwrap_or(0.1)).abs() < 1e-12
                && (c.nc.unwrap_or(0.1) - p.nc.unwrap_or(0.1)).abs() < 1e-12
        })
    }
    pub fn values_for(&self, p: &HwmaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
pub fn expand_grid(r: &HwmaBatchRange) -> Vec<HwmaParams> {
    match expand_grid_checked(r) {
        Ok(v) => v,
        Err(_) => vec![HwmaParams {
            na: Some(r.na.0),
            nb: Some(r.nb.0),
            nc: Some(r.nc.0),
        }],
    }
}

#[inline(always)]
fn axis_f64_checked(t: (f64, f64, f64)) -> Result<Vec<f64>, HwmaError> {
    let (start, end, step) = t;
    let eps = 1e-12;
    if step.abs() < eps || (start - end).abs() < eps {
        return Ok(vec![start]);
    }
    let mut v = Vec::new();
    if step > 0.0 {
        if start > end + eps {
            return Err(HwmaError::InvalidRange { start, end, step });
        }
        let mut x = start;
        while x <= end + eps {
            v.push(x);
            x += step;
        }
    } else {
        if start < end - eps {
            return Err(HwmaError::InvalidRange { start, end, step });
        }
        let mut x = start;
        while x >= end - eps {
            v.push(x);
            x += step;
        }
    }
    Ok(v)
}

#[inline(always)]
fn expand_grid_checked(r: &HwmaBatchRange) -> Result<Vec<HwmaParams>, HwmaError> {
    let nas = axis_f64_checked(r.na)?;
    let nbs = axis_f64_checked(r.nb)?;
    let ncs = axis_f64_checked(r.nc)?;
    let cap = nas
        .len()
        .checked_mul(nbs.len())
        .and_then(|x| x.checked_mul(ncs.len()))
        .ok_or(HwmaError::IntegerOverflow("expand_grid capacity"))?;
    let mut out = Vec::with_capacity(cap);
    for &a in &nas {
        for &b in &nbs {
            for &c in &ncs {
                out.push(HwmaParams {
                    na: Some(a),
                    nb: Some(b),
                    nc: Some(c),
                });
            }
        }
    }
    Ok(out)
}

#[inline(always)]
pub fn hwma_batch_slice(
    data: &[f64],
    sweep: &HwmaBatchRange,
    kern: Kernel,
) -> Result<HwmaBatchOutput, HwmaError> {
    hwma_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn hwma_batch_par_slice(
    data: &[f64],
    sweep: &HwmaBatchRange,
    kern: Kernel,
) -> Result<HwmaBatchOutput, HwmaError> {
    hwma_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn hwma_batch_inner(
    data: &[f64],
    sweep: &HwmaBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<HwmaBatchOutput, HwmaError> {
    let combos = expand_grid_checked(sweep)?;
    if combos.is_empty() {
        return Err(HwmaError::EmptyData);
    }
    let len = data.len();
    if len == 0 {
        return Err(HwmaError::EmptyData);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(HwmaError::AllValuesNaN)?;
    for prm in &combos {
        let na = prm.na.unwrap();
        let nb = prm.nb.unwrap();
        let nc = prm.nc.unwrap();
        if !na.is_finite() || !nb.is_finite() || !nc.is_finite() {
            return Err(HwmaError::InvalidParams { na, nb, nc });
        }
        if !(na > 0.0 && na < 1.0 && nb > 0.0 && nb < 1.0 && nc > 0.0 && nc < 1.0) {
            return Err(HwmaError::InvalidParams { na, nb, nc });
        }
    }
    let rows = combos.len();
    let cols = len;
    let total = rows
        .checked_mul(cols)
        .ok_or(HwmaError::IntegerOverflow("rows*cols"))?;
    let warm: Vec<usize> = std::iter::repeat(first).take(rows).collect();

    let mut buf_mu = make_uninit_matrix(rows, cols);
    unsafe { init_matrix_prefixes(&mut buf_mu, cols, &warm) };

    let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, total) };

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let prm = &combos[row];
        let na = prm.na.unwrap();
        let nb = prm.nb.unwrap();
        let nc = prm.nc.unwrap();

        match kern {
            Kernel::Scalar | Kernel::ScalarBatch => {
                hwma_row_scalar(data, first, na, nb, nc, out_row)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => hwma_row_avx2(data, first, na, nb, nc, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                hwma_row_avx512(data, first, na, nb, nc, out_row)
            }
            _ => hwma_row_scalar(data, first, na, nb, nc, out_row),
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out.par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, slice) in out.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
    } else {
        for (row, slice) in out.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    let values = unsafe {
        Vec::from_raw_parts(
            buf_guard.as_mut_ptr() as *mut f64,
            total,
            buf_guard.capacity(),
        )
    };

    Ok(HwmaBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn hwma_batch_inner_into(
    data: &[f64],
    sweep: &HwmaBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<(Vec<HwmaParams>, usize, usize), HwmaError> {
    let combos = expand_grid_checked(sweep)?;
    if combos.is_empty() {
        return Err(HwmaError::EmptyData);
    }
    let len = data.len();
    if len == 0 {
        return Err(HwmaError::EmptyData);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(HwmaError::AllValuesNaN)?;
    for prm in &combos {
        let na = prm.na.unwrap();
        let nb = prm.nb.unwrap();
        let nc = prm.nc.unwrap();
        if !na.is_finite() || !nb.is_finite() || !nc.is_finite() {
            return Err(HwmaError::InvalidParams { na, nb, nc });
        }
        if !(na > 0.0 && na < 1.0 && nb > 0.0 && nb < 1.0 && nc > 0.0 && nc < 1.0) {
            return Err(HwmaError::InvalidParams { na, nb, nc });
        }
    }
    let rows = combos.len();
    let cols = len;
    let expected = rows
        .checked_mul(cols)
        .ok_or(HwmaError::IntegerOverflow("rows*cols"))?;

    if out.len() != expected {
        return Err(HwmaError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }

    let warm: Vec<usize> = std::iter::repeat(first).take(rows).collect();

    let out_uninit = unsafe {
        std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };
    unsafe { init_matrix_prefixes(out_uninit, cols, &warm) };

    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
    {
        const USE_ROW_SIMD: bool = false;
        if USE_ROW_SIMD {
            match kern {
                Kernel::Avx512 | Kernel::Avx512Batch => unsafe {
                    hwma_batch_rows_avx512(data, first, &combos, cols, out);
                    return Ok((combos, rows, cols));
                },
                Kernel::Avx2 | Kernel::Avx2Batch => unsafe {
                    hwma_batch_rows_avx2(data, first, &combos, cols, out);
                    return Ok((combos, rows, cols));
                },
                _ => {}
            }
        }
    }

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let prm = &combos[row];
        let na = prm.na.unwrap();
        let nb = prm.nb.unwrap();
        let nc = prm.nc.unwrap();

        match kern {
            Kernel::Scalar | Kernel::ScalarBatch => {
                hwma_row_scalar(data, first, na, nb, nc, out_row)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => hwma_row_avx2(data, first, na, nb, nc, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                hwma_row_avx512(data, first, na, nb, nc, out_row)
            }
            _ => hwma_row_scalar(data, first, na, nb, nc, out_row),
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out.par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, slice) in out.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
    } else {
        for (row, slice) in out.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    Ok((combos, rows, cols))
}

#[inline(always)]
unsafe fn hwma_row_scalar(data: &[f64], first: usize, na: f64, nb: f64, nc: f64, out: &mut [f64]) {
    hwma_scalar(data, na, nb, nc, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn hwma_row_avx2(data: &[f64], first: usize, na: f64, nb: f64, nc: f64, out: &mut [f64]) {
    hwma_avx2(data, na, nb, nc, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn hwma_row_avx512(
    data: &[f64],
    first: usize,
    na: f64,
    nb: f64,
    nc: f64,
    out: &mut [f64],
) {
    hwma_avx2(data, na, nb, nc, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn hwma_row_avx512_short(
    data: &[f64],
    first: usize,
    na: f64,
    nb: f64,
    nc: f64,
    out: &mut [f64],
) {
    hwma_scalar(data, na, nb, nc, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn hwma_row_avx512_long(
    data: &[f64],
    first: usize,
    na: f64,
    nb: f64,
    nc: f64,
    out: &mut [f64],
) {
    hwma_scalar(data, na, nb, nc, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn hwma_batch_rows_avx2(
    data: &[f64],
    first: usize,
    combos: &[HwmaParams],
    cols: usize,
    out: &mut [f64],
) {
    use core::arch::x86_64::*;
    const LANES: usize = 4;
    let rows = combos.len();
    if rows == 0 || cols == 0 || first >= cols {
        return;
    }

    let mut r = 0;
    while r + LANES <= rows {
        let na_vec = _mm256_set_pd(
            combos[r + 3].na.unwrap(),
            combos[r + 2].na.unwrap(),
            combos[r + 1].na.unwrap(),
            combos[r + 0].na.unwrap(),
        );
        let nb_vec = _mm256_set_pd(
            combos[r + 3].nb.unwrap(),
            combos[r + 2].nb.unwrap(),
            combos[r + 1].nb.unwrap(),
            combos[r + 0].nb.unwrap(),
        );
        let nc_vec = _mm256_set_pd(
            combos[r + 3].nc.unwrap(),
            combos[r + 2].nc.unwrap(),
            combos[r + 1].nc.unwrap(),
            combos[r + 0].nc.unwrap(),
        );

        let one = _mm256_set1_pd(1.0);
        let half = _mm256_set1_pd(0.5);
        let one_m_na = _mm256_sub_pd(one, na_vec);
        let one_m_nb = _mm256_sub_pd(one, nb_vec);
        let one_m_nc = _mm256_sub_pd(one, nc_vec);

        let init = _mm256_set1_pd(*data.get_unchecked(first));
        let mut f = init;
        let mut v = _mm256_set1_pd(0.0);
        let mut a = _mm256_set1_pd(0.0);

        let mut t = first;
        while t + 1 < cols {
            let x0 = _mm256_set1_pd(*data.get_unchecked(t));

            let s_prev0 = {
                let hv = _mm256_mul_pd(half, a);
                _mm256_add_pd(_mm256_add_pd(f, v), hv)
            };
            let f_new0 = _mm256_fmadd_pd(na_vec, x0, _mm256_mul_pd(one_m_na, s_prev0));
            let diff_f0 = _mm256_sub_pd(f_new0, f);
            let sum_va0 = _mm256_add_pd(v, a);
            let v_new0 = _mm256_fmadd_pd(nb_vec, diff_f0, _mm256_mul_pd(one_m_nb, sum_va0));
            let diff_v0 = _mm256_sub_pd(v_new0, v);
            let a_new0 = _mm256_fmadd_pd(nc_vec, diff_v0, _mm256_mul_pd(one_m_nc, a));
            let s_new0 = {
                let ha = _mm256_mul_pd(half, a_new0);
                _mm256_add_pd(_mm256_add_pd(f_new0, v_new0), ha)
            };
            {
                let mut tmp: [f64; LANES] = core::mem::zeroed();
                _mm256_storeu_pd(tmp.as_mut_ptr(), s_new0);
                for j in 0..LANES {
                    let row = r + j;
                    *out.get_unchecked_mut(row * cols + t) = tmp[j];
                }
            }

            let x1 = _mm256_set1_pd(*data.get_unchecked(t + 1));
            let s_prev1 = {
                let hv = _mm256_mul_pd(half, a_new0);
                _mm256_add_pd(_mm256_add_pd(f_new0, v_new0), hv)
            };
            let f_new1 = _mm256_fmadd_pd(na_vec, x1, _mm256_mul_pd(one_m_na, s_prev1));
            let diff_f1 = _mm256_sub_pd(f_new1, f_new0);
            let sum_va1 = _mm256_add_pd(v_new0, a_new0);
            let v_new1 = _mm256_fmadd_pd(nb_vec, diff_f1, _mm256_mul_pd(one_m_nb, sum_va1));
            let diff_v1 = _mm256_sub_pd(v_new1, v_new0);
            let a_new1 = _mm256_fmadd_pd(nc_vec, diff_v1, _mm256_mul_pd(one_m_nc, a_new0));
            let s_new1 = {
                let ha = _mm256_mul_pd(half, a_new1);
                _mm256_add_pd(_mm256_add_pd(f_new1, v_new1), ha)
            };
            {
                let mut tmp: [f64; LANES] = core::mem::zeroed();
                _mm256_storeu_pd(tmp.as_mut_ptr(), s_new1);
                for j in 0..LANES {
                    let row = r + j;
                    *out.get_unchecked_mut(row * cols + (t + 1)) = tmp[j];
                }
            }

            f = f_new1;
            v = v_new1;
            a = a_new1;
            t += 2;
        }

        if t < cols {
            let x = _mm256_set1_pd(*data.get_unchecked(t));
            let s_prev = {
                let hv = _mm256_mul_pd(half, a);
                _mm256_add_pd(_mm256_add_pd(f, v), hv)
            };
            let f_new = _mm256_fmadd_pd(na_vec, x, _mm256_mul_pd(one_m_na, s_prev));
            let diff_f = _mm256_sub_pd(f_new, f);
            let sum_va = _mm256_add_pd(v, a);
            let v_new = _mm256_fmadd_pd(nb_vec, diff_f, _mm256_mul_pd(one_m_nb, sum_va));
            let diff_v = _mm256_sub_pd(v_new, v);
            let a_new = _mm256_fmadd_pd(nc_vec, diff_v, _mm256_mul_pd(one_m_nc, a));
            let s_new = {
                let ha = _mm256_mul_pd(half, a_new);
                _mm256_add_pd(_mm256_add_pd(f_new, v_new), ha)
            };
            let mut tmp: [f64; LANES] = core::mem::zeroed();
            _mm256_storeu_pd(tmp.as_mut_ptr(), s_new);
            for j in 0..LANES {
                let row = r + j;
                *out.get_unchecked_mut(row * cols + t) = tmp[j];
            }
        }

        r += LANES;
    }

    while r < rows {
        let prm = &combos[r];
        let na = prm.na.unwrap();
        let nb = prm.nb.unwrap();
        let nc = prm.nc.unwrap();
        let row_slice = core::slice::from_raw_parts_mut(out.as_mut_ptr().add(r * cols), cols);
        hwma_scalar(data, na, nb, nc, first, row_slice);
        r += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
unsafe fn hwma_batch_rows_avx512(
    data: &[f64],
    first: usize,
    combos: &[HwmaParams],
    cols: usize,
    out: &mut [f64],
) {
    use core::arch::x86_64::*;
    const LANES: usize = 8;
    let rows = combos.len();
    if rows == 0 || cols == 0 || first >= cols {
        return;
    }

    let mut r = 0;
    while r + LANES <= rows {
        let na_vec = _mm512_set_pd(
            combos[r + 7].na.unwrap(),
            combos[r + 6].na.unwrap(),
            combos[r + 5].na.unwrap(),
            combos[r + 4].na.unwrap(),
            combos[r + 3].na.unwrap(),
            combos[r + 2].na.unwrap(),
            combos[r + 1].na.unwrap(),
            combos[r + 0].na.unwrap(),
        );
        let nb_vec = _mm512_set_pd(
            combos[r + 7].nb.unwrap(),
            combos[r + 6].nb.unwrap(),
            combos[r + 5].nb.unwrap(),
            combos[r + 4].nb.unwrap(),
            combos[r + 3].nb.unwrap(),
            combos[r + 2].nb.unwrap(),
            combos[r + 1].nb.unwrap(),
            combos[r + 0].nb.unwrap(),
        );
        let nc_vec = _mm512_set_pd(
            combos[r + 7].nc.unwrap(),
            combos[r + 6].nc.unwrap(),
            combos[r + 5].nc.unwrap(),
            combos[r + 4].nc.unwrap(),
            combos[r + 3].nc.unwrap(),
            combos[r + 2].nc.unwrap(),
            combos[r + 1].nc.unwrap(),
            combos[r + 0].nc.unwrap(),
        );

        let one = _mm512_set1_pd(1.0);
        let half = _mm512_set1_pd(0.5);
        let one_m_na = _mm512_sub_pd(one, na_vec);
        let one_m_nb = _mm512_sub_pd(one, nb_vec);
        let one_m_nc = _mm512_sub_pd(one, nc_vec);

        let init = _mm512_set1_pd(*data.get_unchecked(first));
        let mut f = init;
        let mut v = _mm512_set1_pd(0.0);
        let mut a = _mm512_set1_pd(0.0);

        let mut t = first;
        while t + 1 < cols {
            let x0 = _mm512_set1_pd(*data.get_unchecked(t));
            let s_prev0 = {
                let hv = _mm512_mul_pd(half, a);
                _mm512_add_pd(_mm512_add_pd(f, v), hv)
            };
            let f_new0 = _mm512_fmadd_pd(na_vec, x0, _mm512_mul_pd(one_m_na, s_prev0));
            let diff_f0 = _mm512_sub_pd(f_new0, f);
            let sum_va0 = _mm512_add_pd(v, a);
            let v_new0 = _mm512_fmadd_pd(nb_vec, diff_f0, _mm512_mul_pd(one_m_nb, sum_va0));
            let diff_v0 = _mm512_sub_pd(v_new0, v);
            let a_new0 = _mm512_fmadd_pd(nc_vec, diff_v0, _mm512_mul_pd(one_m_nc, a));
            let s_new0 = {
                let ha = _mm512_mul_pd(half, a_new0);
                _mm512_add_pd(_mm512_add_pd(f_new0, v_new0), ha)
            };
            {
                let mut tmp: [f64; LANES] = core::mem::zeroed();
                _mm512_storeu_pd(tmp.as_mut_ptr(), s_new0);
                for j in 0..LANES {
                    let row = r + j;
                    *out.get_unchecked_mut(row * cols + t) = tmp[j];
                }
            }

            let x1 = _mm512_set1_pd(*data.get_unchecked(t + 1));
            let s_prev1 = {
                let hv = _mm512_mul_pd(half, a_new0);
                _mm512_add_pd(_mm512_add_pd(f_new0, v_new0), hv)
            };
            let f_new1 = _mm512_fmadd_pd(na_vec, x1, _mm512_mul_pd(one_m_na, s_prev1));
            let diff_f1 = _mm512_sub_pd(f_new1, f_new0);
            let sum_va1 = _mm512_add_pd(v_new0, a_new0);
            let v_new1 = _mm512_fmadd_pd(nb_vec, diff_f1, _mm512_mul_pd(one_m_nb, sum_va1));
            let diff_v1 = _mm512_sub_pd(v_new1, v_new0);
            let a_new1 = _mm512_fmadd_pd(nc_vec, diff_v1, _mm512_mul_pd(one_m_nc, a_new0));
            let s_new1 = {
                let ha = _mm512_mul_pd(half, a_new1);
                _mm512_add_pd(_mm512_add_pd(f_new1, v_new1), ha)
            };
            {
                let mut tmp: [f64; LANES] = core::mem::zeroed();
                _mm512_storeu_pd(tmp.as_mut_ptr(), s_new1);
                for j in 0..LANES {
                    let row = r + j;
                    *out.get_unchecked_mut(row * cols + (t + 1)) = tmp[j];
                }
            }

            f = f_new1;
            v = v_new1;
            a = a_new1;
            t += 2;
        }

        if t < cols {
            let x = _mm512_set1_pd(*data.get_unchecked(t));
            let s_prev = {
                let hv = _mm512_mul_pd(half, a);
                _mm512_add_pd(_mm512_add_pd(f, v), hv)
            };
            let f_new = _mm512_fmadd_pd(na_vec, x, _mm512_mul_pd(one_m_na, s_prev));
            let diff_f = _mm512_sub_pd(f_new, f);
            let sum_va = _mm512_add_pd(v, a);
            let v_new = _mm512_fmadd_pd(nb_vec, diff_f, _mm512_mul_pd(one_m_nb, sum_va));
            let diff_v = _mm512_sub_pd(v_new, v);
            let a_new = _mm512_fmadd_pd(nc_vec, diff_v, _mm512_mul_pd(one_m_nc, a));
            let s_new = {
                let ha = _mm512_mul_pd(half, a_new);
                _mm512_add_pd(_mm512_add_pd(f_new, v_new), ha)
            };
            let mut tmp: [f64; LANES] = core::mem::zeroed();
            _mm512_storeu_pd(tmp.as_mut_ptr(), s_new);
            for j in 0..LANES {
                let row = r + j;
                *out.get_unchecked_mut(row * cols + t) = tmp[j];
            }
        }

        r += LANES;
    }

    while r < rows {
        let prm = &combos[r];
        let row_slice = core::slice::from_raw_parts_mut(out.as_mut_ptr().add(r * cols), cols);
        hwma_scalar(
            data,
            prm.na.unwrap(),
            prm.nb.unwrap(),
            prm.nc.unwrap(),
            first,
            row_slice,
        );
        r += 1;
    }
}

#[inline(always)]
pub fn expand_grid_hwma(r: &HwmaBatchRange) -> Vec<HwmaParams> {
    expand_grid(r)
}

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;

#[cfg(feature = "python")]
#[pyfunction(name = "hwma")]
#[pyo3(signature = (data, na, nb, nc, kernel=None))]
pub fn hwma_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    na: f64,
    nb: f64,
    nc: f64,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let params = HwmaParams {
        na: Some(na),
        nb: Some(nb),
        nc: Some(nc),
    };
    let input = HwmaInput::from_slice(slice_in, params);

    let result_vec: Vec<f64> = py
        .allow_threads(|| hwma_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyfunction(name = "hwma_batch")]
#[pyo3(signature = (data, na_range, nb_range, nc_range, kernel=None))]
pub fn hwma_batch_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    na_range: (f64, f64, f64),
    nb_range: (f64, f64, f64),
    nc_range: (f64, f64, f64),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;

    let sweep = HwmaBatchRange {
        na: na_range,
        nb: nb_range,
        nc: nc_range,
    };

    let rows = expand_grid_checked(&sweep)
        .map(|v| v.len())
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let cols = slice_in.len();

    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    let (combos_result, _, _) = py
        .allow_threads(|| {
            let simd = match kern {
                Kernel::Auto => Kernel::Scalar,
                kernel => match kernel {
                    Kernel::Avx512Batch => Kernel::Avx512,
                    Kernel::Avx2Batch => Kernel::Avx2,
                    Kernel::ScalarBatch => Kernel::Scalar,
                    _ => kernel,
                },
            };

            hwma_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;

    dict.set_item(
        "na",
        combos_result
            .iter()
            .map(|p| p.na.unwrap())
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "nb",
        combos_result
            .iter()
            .map(|p| p.nb.unwrap())
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "nc",
        combos_result
            .iter()
            .map(|p| p.nc.unwrap())
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;

    Ok(dict)
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "hwma_cuda_batch_dev")]
#[pyo3(signature = (data, na_range, nb_range, nc_range, device_id=0))]
pub fn hwma_cuda_batch_dev_py(
    py: Python<'_>,
    data: PyReadonlyArray1<'_, f64>,
    na_range: (f64, f64, f64),
    nb_range: (f64, f64, f64),
    nc_range: (f64, f64, f64),
    device_id: usize,
) -> PyResult<HwmaDeviceArrayF32Py> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let slice_in = data.as_slice()?;
    let sweep = HwmaBatchRange {
        na: na_range,
        nb: nb_range,
        nc: nc_range,
    };
    let data_f32: Vec<f32> = slice_in.iter().map(|&v| v as f32).collect();

    let (inner, ctx_arc) = py.allow_threads(|| {
        let cuda = CudaHwma::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx = cuda.context_arc();
        let out = cuda
            .hwma_batch_dev(&data_f32, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok::<_, pyo3::PyErr>((out, ctx))
    })?;

    Ok(HwmaDeviceArrayF32Py {
        inner,
        _ctx: ctx_arc,
        device_id: device_id as u32,
    })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "hwma_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm_f32, na, nb, nc, device_id=0))]
pub fn hwma_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    data_tm_f32: PyReadonlyArray2<'_, f32>,
    na: f64,
    nb: f64,
    nc: f64,
    device_id: usize,
) -> PyResult<HwmaDeviceArrayF32Py> {
    use numpy::PyUntypedArrayMethods;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let flat_in = data_tm_f32.as_slice()?;
    let rows = data_tm_f32.shape()[0];
    let cols = data_tm_f32.shape()[1];
    let params = HwmaParams {
        na: Some(na),
        nb: Some(nb),
        nc: Some(nc),
    };

    let (inner, ctx_arc) = py.allow_threads(|| {
        let cuda = CudaHwma::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx = cuda.context_arc();
        let out = cuda
            .hwma_multi_series_one_param_time_major_dev(flat_in, cols, rows, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok::<_, pyo3::PyErr>((out, ctx))
    })?;

    Ok(HwmaDeviceArrayF32Py {
        inner,
        _ctx: ctx_arc,
        device_id: device_id as u32,
    })
}

#[cfg(feature = "python")]
#[pyclass(name = "HwmaStream")]
pub struct HwmaStreamPy {
    inner: HwmaStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl HwmaStreamPy {
    #[new]
    pub fn new(na: f64, nb: f64, nc: f64) -> PyResult<Self> {
        let params = HwmaParams {
            na: Some(na),
            nb: Some(nb),
            nc: Some(nc),
        };
        match HwmaStream::try_new(params) {
            Ok(stream) => Ok(Self { inner: stream }),
            Err(e) => Err(PyValueError::new_err(format!("HwmaStream error: {}", e))),
        }
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.inner.update(value)
    }
}

#[inline(always)]
pub fn hwma_into_slice(dst: &mut [f64], input: &HwmaInput, kern: Kernel) -> Result<(), HwmaError> {
    hwma_with_kernel_into(input, kern, dst)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn hwma_js(data: &[f64], na: f64, nb: f64, nc: f64) -> Result<Vec<f64>, JsValue> {
    let params = HwmaParams {
        na: Some(na),
        nb: Some(nb),
        nc: Some(nc),
    };
    let input = HwmaInput::from_slice(data, params);

    let mut output = vec![0.0; data.len()];

    hwma_into_slice(&mut output, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct HwmaBatchConfig {
    pub na_range: (f64, f64, f64),
    pub nb_range: (f64, f64, f64),
    pub nc_range: (f64, f64, f64),
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct HwmaBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<HwmaParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = hwma_batch)]
pub fn hwma_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: HwmaBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = HwmaBatchRange {
        na: config.na_range,
        nb: config.nb_range,
        nc: config.nc_range,
    };

    let simd = match detect_best_batch_kernel() {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        _ => Kernel::Scalar,
    };

    let output = hwma_batch_inner(data, &sweep, simd, false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = HwmaBatchJsOutput {
        values: output.values,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    };

    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn hwma_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn hwma_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = hwma_into)]
pub fn hwma_ptr_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    na: f64,
    nb: f64,
    nc: f64,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to hwma_into"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);

        if len == 0 {
            return Err(JsValue::from_str("Empty data"));
        }

        let params = HwmaParams {
            na: Some(na),
            nb: Some(nb),
            nc: Some(nc),
        };
        let input = HwmaInput::from_slice(data, params);

        if in_ptr == out_ptr {
            let mut temp = vec![0.0; len];
            hwma_into_slice(&mut temp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            hwma_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn hwma_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    na_start: f64,
    na_end: f64,
    na_step: f64,
    nb_start: f64,
    nb_end: f64,
    nb_step: f64,
    nc_start: f64,
    nc_end: f64,
    nc_step: f64,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to hwma_batch_into"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);

        let sweep = HwmaBatchRange {
            na: (na_start, na_end, na_step),
            nb: (nb_start, nb_end, nb_step),
            nc: (nc_start, nc_end, nc_step),
        };

        let combos = expand_grid(&sweep);
        let rows = combos.len();
        let cols = len;

        let out = std::slice::from_raw_parts_mut(out_ptr, rows * cols);

        let simd = match detect_best_batch_kernel() {
            Kernel::Avx512Batch => Kernel::Avx512,
            Kernel::Avx2Batch => Kernel::Avx2,
            _ => Kernel::Scalar,
        };

        hwma_batch_inner_into(data, &sweep, simd, false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(rows)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn hwma_batch_js(
    data: &[f64],
    na_start: f64,
    na_end: f64,
    na_step: f64,
    nb_start: f64,
    nb_end: f64,
    nb_step: f64,
    nc_start: f64,
    nc_end: f64,
    nc_step: f64,
) -> Result<Vec<f64>, JsValue> {
    let sweep = HwmaBatchRange {
        na: (na_start, na_end, na_step),
        nb: (nb_start, nb_end, nb_step),
        nc: (nc_start, nc_end, nc_step),
    };

    hwma_batch_inner(data, &sweep, Kernel::Auto, false)
        .map(|output| output.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn hwma_batch_metadata_js(
    na_start: f64,
    na_end: f64,
    na_step: f64,
    nb_start: f64,
    nb_end: f64,
    nb_step: f64,
    nc_start: f64,
    nc_end: f64,
    nc_step: f64,
) -> Vec<f64> {
    let sweep = HwmaBatchRange {
        na: (na_start, na_end, na_step),
        nb: (nb_start, nb_end, nb_step),
        nc: (nc_start, nc_end, nc_step),
    };

    let combos = expand_grid(&sweep);
    let mut result = Vec::new();

    for combo in combos {
        result.push(combo.na.unwrap());
        result.push(combo.nb.unwrap());
        result.push(combo.nc.unwrap());
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    #[cfg(feature = "proptest")]
    use proptest::prelude::*;

    #[test]
    fn test_hwma_into_matches_api() -> Result<(), Box<dyn std::error::Error>> {
        let mut data = vec![f64::NAN; 7];
        data.extend((0..256).map(|i| (i as f64).sin() * 10.0 + 100.0));

        let params = HwmaParams::default();
        let input = HwmaInput::from_slice(&data, params);

        let baseline = hwma(&input)?.values;

        let mut out = vec![0.0; data.len()];
        hwma_into(&input, &mut out)?;

        assert_eq!(baseline.len(), out.len());

        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b)
        }
        for (i, (&a, &b)) in baseline.iter().zip(out.iter()).enumerate() {
            assert!(
                eq_or_both_nan(a, b),
                "mismatch at idx {}: alloc={} into={}",
                i,
                a,
                b
            );
        }
        Ok(())
    }

    fn check_hwma_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = HwmaParams {
            na: None,
            nb: None,
            nc: None,
        };
        let input = HwmaInput::from_candles(&candles, "close", default_params);
        let output = hwma_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_hwma_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = HwmaInput::from_candles(&candles, "close", HwmaParams::default());
        let result = hwma_with_kernel(&input, kernel)?;
        let expected_last_five = [
            57941.04005793378,
            58106.90324194954,
            58250.474156632234,
            58428.90005831887,
            58499.37021151028,
        ];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-3,
                "[{}] HWMA {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_hwma_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = HwmaInput::with_default_candles(&candles);
        match input.data {
            HwmaData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected HwmaData::Candles"),
        }
        let output = hwma_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_hwma_invalid_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [10.0, 20.0, 30.0];
        let params = HwmaParams {
            na: Some(-0.2),
            nb: Some(1.1),
            nc: Some(0.1),
        };
        let input = HwmaInput::from_slice(&data, params);
        let result = hwma_with_kernel(&input, kernel);
        assert!(matches!(result, Err(HwmaError::InvalidParams { .. })));
        Ok(())
    }

    fn check_hwma_invalid_nan_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [1.0, 2.0];
        let params = HwmaParams {
            na: Some(f64::NAN),
            nb: Some(0.5),
            nc: Some(0.5),
        };
        let input = HwmaInput::from_slice(&data, params);
        let res = hwma_with_kernel(&input, kernel);
        assert!(matches!(res, Err(HwmaError::InvalidParams { .. })));
        Ok(())
    }

    fn check_hwma_empty_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data: [f64; 0] = [];
        let params = HwmaParams {
            na: Some(0.2),
            nb: Some(0.1),
            nc: Some(0.1),
        };
        let input = HwmaInput::from_slice(&data, params);
        let result = hwma_with_kernel(&input, kernel);
        assert!(matches!(result, Err(HwmaError::EmptyData)));
        Ok(())
    }

    fn check_hwma_small_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [42.0];
        let params = HwmaParams {
            na: Some(0.2),
            nb: Some(0.1),
            nc: Some(0.1),
        };
        let input = HwmaInput::from_slice(&data, params);
        let result = hwma_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), data.len());
        assert!((result.values[0] - data[0]).abs() < 1e-12);
        Ok(())
    }

    fn check_hwma_slice_data_reinput(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params_1 = HwmaParams::default();
        let input_1 = HwmaInput::from_candles(&candles, "close", params_1);
        let result_1 = hwma_with_kernel(&input_1, kernel)?;
        assert_eq!(result_1.values.len(), candles.close.len());
        let params_2 = HwmaParams {
            na: Some(0.3),
            nb: Some(0.15),
            nc: Some(0.05),
        };
        let input_2 = HwmaInput::from_slice(&result_1.values, params_2);
        let result_2 = hwma_with_kernel(&input_2, kernel)?;
        assert_eq!(result_2.values.len(), result_1.values.len());
        for i in 240..result_2.values.len() {
            assert!(!result_2.values[i].is_nan());
        }
        Ok(())
    }

    fn check_hwma_nan_check(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = HwmaParams::default();
        let input = HwmaInput::from_candles(&candles, "close", params);
        let result = hwma_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), candles.close.len());
        if result.values.len() > 240 {
            for i in 240..result.values.len() {
                assert!(!result.values[i].is_nan());
            }
        }
        Ok(())
    }

    fn check_hwma_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let params = HwmaParams::default();
        let input = HwmaInput::from_candles(&candles, "close", params.clone());
        let batch = hwma_with_kernel(&input, kernel)?.values;

        let mut stream = HwmaStream::try_new(params)?;
        let mut streaming = Vec::with_capacity(candles.close.len());
        for &v in &candles.close {
            match stream.update(v) {
                Some(x) => streaming.push(x),
                None => streaming.push(f64::NAN),
            }
        }

        assert_eq!(batch.len(), streaming.len());
        for (i, (&b, &s)) in batch.iter().zip(streaming.iter()).enumerate() {
            if b.is_nan() && s.is_nan() {
                continue;
            }
            assert!((b - s).abs() < 1e-9, "[{test_name}] mismatch at {i}");
        }
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_hwma_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let main_strat = (
            proptest::collection::vec(
                (-1e6f64..1e6).prop_filter("finite", |x| x.is_finite()),
                20..200,
            ),
            0.01f64..0.99,
            0.01f64..0.99,
            0.01f64..0.99,
        );

        proptest::test_runner::TestRunner::default().run(&main_strat, |(data, na, nb, nc)| {
            let params = HwmaParams {
                na: Some(na),
                nb: Some(nb),
                nc: Some(nc),
            };
            let input = HwmaInput::from_slice(&data, params.clone());
            let HwmaOutput { values: out } = hwma_with_kernel(&input, kernel).unwrap();

            let HwmaOutput { values: ref_out } = hwma_with_kernel(&input, Kernel::Scalar).unwrap();

            prop_assert_eq!(out.len(), data.len(), "Output length should match input");

            if !data.is_empty() && data[0].is_finite() {
                let first_diff = (out[0] - data[0]).abs();
                prop_assert!(
                    first_diff < 1e-12,
                    "First output should match first input: got {}, expected {}, diff {}",
                    out[0],
                    data[0],
                    first_diff
                );
            }

            for i in 0..out.len() {
                let y = out[i];
                let r = ref_out[i];

                if !y.is_finite() || !r.is_finite() {
                    prop_assert_eq!(y.to_bits(), r.to_bits(), "NaN/Inf mismatch at idx {}", i);
                    continue;
                }

                let ulp_diff = y.to_bits().abs_diff(r.to_bits());
                prop_assert!(
                    (y - r).abs() <= 1e-9 || ulp_diff <= 5,
                    "[{}] SIMD mismatch at idx {}: kernel={:.15}, scalar={:.15}, ulp_diff={}",
                    test_name,
                    i,
                    y,
                    r,
                    ulp_diff
                );
            }

            let (data_min, data_max) = data
                .iter()
                .filter(|&&x| x.is_finite())
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &x| {
                    (min.min(x), max.max(x))
                });

            if data_min.is_finite() && data_max.is_finite() {
                let range = (data_max - data_min).abs();

                let max_param = na.max(nb).max(nc);
                let extrapolation_factor = 0.1 + 0.2 * max_param;
                let tolerance = range * extrapolation_factor + 1e-6;

                for (idx, &y) in out.iter().enumerate() {
                    if y.is_finite() {
                        prop_assert!(
                            y >= data_min - tolerance && y <= data_max + tolerance,
                            "idx {}: {} outside bounds [{}, {}] with tolerance {}",
                            idx,
                            y,
                            data_min - tolerance,
                            data_max + tolerance,
                            tolerance
                        );
                    }
                }
            }

            if data.len() > 20 && data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-12) {
                let constant_val = data[0];

                let check_start = out.len() * 9 / 10;
                for (idx, &val) in out[check_start..].iter().enumerate() {
                    if val.is_finite() {
                        let diff = (val - constant_val).abs();
                        prop_assert!(
							diff < 1e-6,
							"Constant data convergence failed at idx {}: expected {}, got {}, diff {}",
							check_start + idx, constant_val, val, diff
						);
                    }
                }
            }

            if (na - 1.0).abs() < 0.01 && nb < 0.01 && nc < 0.01 {
                for (idx, (&y, &x)) in out.iter().zip(data.iter()).enumerate() {
                    if y.is_finite() && x.is_finite() {
                        let diff = (y - x).abs();
                        prop_assert!(
							diff < 0.1,
							"With na≈1, nb≈0, nc≈0, output should follow input at idx {}: x={}, y={}, diff={}",
							idx, x, y, diff
						);
                    }
                }
            }

            Ok(())
        })?;

        let param_strat = (
            proptest::collection::vec(
                (-100f64..100f64).prop_filter("finite", |x| x.is_finite()),
                50..100,
            ),
            prop::strategy::Union::new(vec![
                (0.01f64..0.05).boxed(),
                (0.95f64..0.99).boxed(),
                (0.45f64..0.55).boxed(),
            ]),
            prop::strategy::Union::new(vec![
                (0.01f64..0.05).boxed(),
                (0.95f64..0.99).boxed(),
                (0.45f64..0.55).boxed(),
            ]),
            prop::strategy::Union::new(vec![
                (0.01f64..0.05).boxed(),
                (0.95f64..0.99).boxed(),
                (0.45f64..0.55).boxed(),
            ]),
        );

        proptest::test_runner::TestRunner::default().run(&param_strat, |(data, na, nb, nc)| {
            let params = HwmaParams {
                na: Some(na),
                nb: Some(nb),
                nc: Some(nc),
            };
            let input = HwmaInput::from_slice(&data, params);
            let result = hwma_with_kernel(&input, kernel);

            prop_assert!(
                result.is_ok(),
                "HWMA should handle all parameter combinations"
            );
            let HwmaOutput { values } = result.unwrap();

            for (idx, &val) in values.iter().enumerate() {
                if !val.is_nan() {
                    prop_assert!(
                        val.is_finite(),
                        "Output should be finite at idx {}: got {}",
                        idx,
                        val
                    );
                }
            }

            let avg_param = (na + nb + nc) / 3.0;

            let diffs: Vec<f64> = values
                .windows(2)
                .filter(|w| w[0].is_finite() && w[1].is_finite())
                .map(|w| (w[1] - w[0]).abs())
                .collect();

            if diffs.len() > 10 {
                let avg_diff = diffs.iter().sum::<f64>() / diffs.len() as f64;
                let data_diffs: Vec<f64> = data
                    .windows(2)
                    .filter(|w| w[0].is_finite() && w[1].is_finite())
                    .map(|w| (w[1] - w[0]).abs())
                    .collect();

                if !data_diffs.is_empty() {
                    let data_avg_diff = data_diffs.iter().sum::<f64>() / data_diffs.len() as f64;

                    if avg_param < 0.1 {
                        prop_assert!(
							avg_diff < data_avg_diff * 0.5,
							"Small parameters should produce smooth output: output_diff={}, data_diff={}",
							avg_diff, data_avg_diff
						);
                    } else if avg_param > 0.9 {
                        prop_assert!(
							avg_diff < data_avg_diff * 1.5,
							"Large parameters should produce responsive output: output_diff={}, data_diff={}",
							avg_diff, data_avg_diff
						);
                    }
                }
            }

            Ok(())
        })?;

        let step_strat = (
            10usize..50,
            -100f64..100f64,
            -100f64..100f64,
            0.1f64..0.9,
            0.1f64..0.9,
            0.1f64..0.9,
        );

        proptest::test_runner::TestRunner::default().run(
            &step_strat,
            |(size, level1, level2, na, nb, nc)| {
                let mut data = vec![level1; size];
                data.extend(vec![level2; size]);

                let params = HwmaParams {
                    na: Some(na),
                    nb: Some(nb),
                    nc: Some(nc),
                };
                let input = HwmaInput::from_slice(&data, params);
                let HwmaOutput { values } = hwma_with_kernel(&input, kernel).unwrap();

                let last_quarter = values.len() * 3 / 4;
                let final_values = &values[last_quarter..];
                let avg_final = final_values.iter().filter(|&&v| v.is_finite()).sum::<f64>()
                    / final_values.len() as f64;

                let convergence_error = (avg_final - level2).abs();
                let step_size = (level2 - level1).abs();

                prop_assert!(
                    convergence_error < step_size * 0.1 + 1e-3,
                    "HWMA should converge to new level: expected {}, got {}, error {}",
                    level2,
                    avg_final,
                    convergence_error
                );

                Ok(())
            },
        )?;

        let small_data_strat = (1usize..=5, 0.1f64..0.9, 0.1f64..0.9, 0.1f64..0.9);

        proptest::test_runner::TestRunner::default().run(
            &small_data_strat,
            |(size, na, nb, nc)| {
                let data: Vec<f64> = (1..=size).map(|i| i as f64 * 10.0).collect();

                let params = HwmaParams {
                    na: Some(na),
                    nb: Some(nb),
                    nc: Some(nc),
                };
                let input = HwmaInput::from_slice(&data, params);

                let result = hwma_with_kernel(&input, kernel);
                prop_assert!(result.is_ok(), "HWMA should handle small data sizes");

                let HwmaOutput { values } = result.unwrap();

                prop_assert_eq!(values.len(), data.len(), "Output length should match input");

                if !data.is_empty() {
                    let first_diff = (values[0] - data[0]).abs();
                    prop_assert!(
                        first_diff < 1e-9,
                        "First value should match for small data: got {}, expected {}",
                        values[0],
                        data[0]
                    );
                }

                for (idx, &val) in values.iter().enumerate() {
                    prop_assert!(
                        val.is_finite(),
                        "All values should be finite for small data at idx {}: got {}",
                        idx,
                        val
                    );
                }

                Ok(())
            },
        )?;

        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_hwma_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let test_params = vec![
            HwmaParams::default(),
            HwmaParams {
                na: Some(0.05),
                nb: Some(0.05),
                nc: Some(0.05),
            },
            HwmaParams {
                na: Some(0.1),
                nb: Some(0.1),
                nc: Some(0.1),
            },
            HwmaParams {
                na: Some(0.15),
                nb: Some(0.1),
                nc: Some(0.05),
            },
            HwmaParams {
                na: Some(0.2),
                nb: Some(0.2),
                nc: Some(0.2),
            },
            HwmaParams {
                na: Some(0.3),
                nb: Some(0.3),
                nc: Some(0.3),
            },
            HwmaParams {
                na: Some(0.4),
                nb: Some(0.4),
                nc: Some(0.4),
            },
            HwmaParams {
                na: Some(0.5),
                nb: Some(0.5),
                nc: Some(0.5),
            },
            HwmaParams {
                na: Some(0.6),
                nb: Some(0.6),
                nc: Some(0.6),
            },
            HwmaParams {
                na: Some(0.7),
                nb: Some(0.7),
                nc: Some(0.7),
            },
            HwmaParams {
                na: Some(0.8),
                nb: Some(0.8),
                nc: Some(0.8),
            },
            HwmaParams {
                na: Some(0.9),
                nb: Some(0.9),
                nc: Some(0.9),
            },
            HwmaParams {
                na: Some(0.1),
                nb: Some(0.5),
                nc: Some(0.9),
            },
            HwmaParams {
                na: Some(0.9),
                nb: Some(0.5),
                nc: Some(0.1),
            },
            HwmaParams {
                na: Some(0.5),
                nb: Some(0.3),
                nc: Some(0.2),
            },
            HwmaParams {
                na: Some(0.01),
                nb: Some(0.01),
                nc: Some(0.01),
            },
            HwmaParams {
                na: Some(0.99),
                nb: Some(0.99),
                nc: Some(0.99),
            },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = HwmaInput::from_candles(&candles, "close", params.clone());
            let output = hwma_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
                        with params: na={}, nb={}, nc={}",
                        test_name,
                        val,
                        bits,
                        i,
                        params.na.unwrap_or(0.2),
                        params.nb.unwrap_or(0.1),
                        params.nc.unwrap_or(0.1)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
                        with params: na={}, nb={}, nc={}",
                        test_name,
                        val,
                        bits,
                        i,
                        params.na.unwrap_or(0.2),
                        params.nb.unwrap_or(0.1),
                        params.nc.unwrap_or(0.1)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
                        with params: na={}, nb={}, nc={}",
                        test_name,
                        val,
                        bits,
                        i,
                        params.na.unwrap_or(0.2),
                        params.nb.unwrap_or(0.1),
                        params.nc.unwrap_or(0.1)
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_hwma_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    #[cfg(not(feature = "proptest"))]
    fn check_hwma_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        Ok(())
    }

    macro_rules! generate_all_hwma_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $( #[test] fn [<$test_fn _scalar_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar);
                })*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $( #[test] fn [<$test_fn _avx2_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2);
                })*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $( #[test] fn [<$test_fn _avx512_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512);
                })*

                #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                $( #[test] fn [<$test_fn _simd128_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _simd128_f64>]), Kernel::Scalar);
                })*
            }
        }
    }
    generate_all_hwma_tests!(
        check_hwma_partial_params,
        check_hwma_accuracy,
        check_hwma_default_candles,
        check_hwma_invalid_params,
        check_hwma_invalid_nan_params,
        check_hwma_empty_data,
        check_hwma_small_data,
        check_hwma_slice_data_reinput,
        check_hwma_nan_check,
        check_hwma_streaming,
        check_hwma_property,
        check_hwma_no_poison
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = HwmaBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = HwmaParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        let expected = [
            57941.04005793378,
            58106.90324194954,
            58250.474156632234,
            58428.90005831887,
            58499.37021151028,
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

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let test_configs = vec![
            (0.05, 0.15, 0.05, 0.05, 0.15, 0.05, 0.05, 0.15, 0.05),
            (0.1, 0.5, 0.1, 0.1, 0.5, 0.1, 0.1, 0.5, 0.1),
            (0.5, 0.9, 0.1, 0.5, 0.9, 0.1, 0.5, 0.9, 0.1),
            (0.1, 0.9, 0.4, 0.1, 0.5, 0.2, 0.1, 0.3, 0.1),
            (0.01, 0.05, 0.01, 0.01, 0.05, 0.01, 0.01, 0.05, 0.01),
            (0.8, 0.99, 0.05, 0.8, 0.99, 0.05, 0.8, 0.99, 0.05),
            (0.1, 0.9, 0.2, 0.2, 0.8, 0.3, 0.3, 0.7, 0.2),
            (0.1, 0.9, 0.4, 0.1, 0.9, 0.4, 0.1, 0.9, 0.4),
            (0.1, 0.3, 0.05, 0.1, 0.3, 0.05, 0.1, 0.3, 0.05),
        ];

        for (
            cfg_idx,
            &(na_start, na_end, na_step, nb_start, nb_end, nb_step, nc_start, nc_end, nc_step),
        ) in test_configs.iter().enumerate()
        {
            let output = HwmaBatchBuilder::new()
                .kernel(kernel)
                .na_range(na_start, na_end, na_step)
                .nb_range(nb_start, nb_end, nb_step)
                .nc_range(nc_start, nc_end, nc_step)
                .apply_candles(&c, "close")?;

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
                        at row {} col {} (flat index {}) with params: na={}, nb={}, nc={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.na.unwrap_or(0.2),
                        combo.nb.unwrap_or(0.1),
                        combo.nc.unwrap_or(0.1)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
                        at row {} col {} (flat index {}) with params: na={}, nb={}, nc={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.na.unwrap_or(0.2),
                        combo.nb.unwrap_or(0.1),
                        combo.nc.unwrap_or(0.1)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
                        at row {} col {} (flat index {}) with params: na={}, nb={}, nc={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.na.unwrap_or(0.2),
                        combo.nb.unwrap_or(0.1),
                        combo.nc.unwrap_or(0.1)
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
