#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;

#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::cuda_available;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::moving_averages::CudaEhlersEcema;
#[cfg(all(feature = "python", feature = "cuda"))]
use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};

#[cfg(all(feature = "python", feature = "cuda"))]
mod ecema_python_cuda_handle {
    use super::*;
    use cust::context::Context;
    use cust::memory::DeviceBuffer;
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::PyDict;
    use std::ffi::c_void;
    use std::sync::Arc;

    #[pyclass(module = "ta_indicators.cuda", unsendable, name = "DeviceArrayF32Py")]
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
                        value.as_mut_ptr() as *mut std::ffi::c_void,
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
            use crate::utilities::dlpack_cuda::export_f32_cuda_dlpack_2d;

            let (kdl, alloc_dev) = self.__dlpack_device__();
            if let Some(d) = dl_device.as_ref() {
                if let Ok((dev_type, dev_id)) = d.extract::<(i32, i32)>(py) {
                    if dev_type != kdl || dev_id != alloc_dev {
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
                                "__dlpack__: requested device does not match producer buffer",
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

    pub use DeviceArrayF32Py as EcemaDeviceArrayF32Py;
}

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

use crate::indicators::moving_averages::ema::{ema, ema_into_slice, EmaInput, EmaParams};

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

impl<'a> AsRef<[f64]> for EhlersEcemaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            EhlersEcemaData::Slice(slice) => slice,
            EhlersEcemaData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum EhlersEcemaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct EhlersEcemaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct EhlersEcemaParams {
    pub length: Option<usize>,
    pub gain_limit: Option<usize>,
    pub pine_compatible: Option<bool>,
    pub confirmed_only: Option<bool>,
}

impl Default for EhlersEcemaParams {
    fn default() -> Self {
        Self {
            length: Some(20),
            gain_limit: Some(50),
            pine_compatible: Some(false),
            confirmed_only: Some(false),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EhlersEcemaInput<'a> {
    pub data: EhlersEcemaData<'a>,
    pub params: EhlersEcemaParams,
}

impl<'a> EhlersEcemaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: EhlersEcemaParams) -> Self {
        Self {
            data: EhlersEcemaData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }

    #[inline]
    pub fn from_slice(sl: &'a [f64], p: EhlersEcemaParams) -> Self {
        Self {
            data: EhlersEcemaData::Slice(sl),
            params: p,
        }
    }

    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", EhlersEcemaParams::default())
    }

    #[inline]
    pub fn get_length(&self) -> usize {
        self.params.length.unwrap_or(20)
    }

    #[inline]
    pub fn get_gain_limit(&self) -> usize {
        self.params.gain_limit.unwrap_or(50)
    }

    #[inline]
    pub fn get_pine_compatible(&self) -> bool {
        self.params.pine_compatible.unwrap_or(false)
    }

    #[inline]
    pub fn get_confirmed_only(&self) -> bool {
        self.params.confirmed_only.unwrap_or(false)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct EhlersEcemaBuilder {
    length: Option<usize>,
    gain_limit: Option<usize>,
    kernel: Kernel,
}

impl Default for EhlersEcemaBuilder {
    fn default() -> Self {
        Self {
            length: None,
            gain_limit: None,
            kernel: Kernel::Auto,
        }
    }
}

impl EhlersEcemaBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn length(mut self, n: usize) -> Self {
        self.length = Some(n);
        self
    }

    #[inline(always)]
    pub fn gain_limit(mut self, g: usize) -> Self {
        self.gain_limit = Some(g);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<EhlersEcemaOutput, EhlersEcemaError> {
        let p = EhlersEcemaParams {
            length: self.length,
            gain_limit: self.gain_limit,
            pine_compatible: None,
            confirmed_only: None,
        };
        let i = EhlersEcemaInput::from_candles(c, "close", p);
        ehlers_ecema_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<EhlersEcemaOutput, EhlersEcemaError> {
        let p = EhlersEcemaParams {
            length: self.length,
            gain_limit: self.gain_limit,
            pine_compatible: None,
            confirmed_only: None,
        };
        let i = EhlersEcemaInput::from_slice(d, p);
        ehlers_ecema_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<EhlersEcemaStream, EhlersEcemaError> {
        let p = EhlersEcemaParams {
            length: self.length,
            gain_limit: self.gain_limit,
            pine_compatible: None,
            confirmed_only: None,
        };
        EhlersEcemaStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum EhlersEcemaError {
    #[error("ehlers_ecema: Input data slice is empty.")]
    EmptyInputData,

    #[error("ehlers_ecema: All values are NaN.")]
    AllValuesNaN,

    #[error("ehlers_ecema: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },

    #[error("ehlers_ecema: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },

    #[error("ehlers_ecema: Invalid gain limit: {gain_limit}")]
    InvalidGainLimit { gain_limit: usize },

    #[error("ehlers_ecema: EMA calculation failed: {0}")]
    EmaError(String),

    #[error("ehlers_ecema: Output slice length mismatch: expected {expected}, got {got}")]
    OutputLengthMismatch { expected: usize, got: usize },

    #[error("ehlers_ecema: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },

    #[error("ehlers_ecema: non-batch kernel passed to batch API: {0:?}")]
    InvalidKernelForBatch(Kernel),

    #[error("ehlers_ecema: size computation overflow during allocation ({what})")]
    SizeOverflow { what: &'static str },
}

#[inline]
fn calculate_pine_ema_into(
    dst: &mut [f64],
    data: &[f64],
    _length: usize,
    alpha: f64,
    beta: f64,
    first: usize,
) {
    let len = data.len();
    for v in &mut dst[..first.min(len)] {
        *v = f64::NAN;
    }
    if first >= len {
        return;
    }
    let mut ema = 0.0;
    for i in first..len {
        let src = data[i];
        if src.is_finite() {
            ema = alpha * src + beta * ema;
            dst[i] = ema;
        } else {
            dst[i] = ema;
        }
    }
}

#[inline]
pub fn ehlers_ecema(input: &EhlersEcemaInput) -> Result<EhlersEcemaOutput, EhlersEcemaError> {
    ehlers_ecema_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn ehlers_ecema_prepare<'a>(
    input: &'a EhlersEcemaInput,
    kernel: Kernel,
) -> Result<(&'a [f64], usize, usize, usize, f64, f64, Kernel), EhlersEcemaError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();

    if len == 0 {
        return Err(EhlersEcemaError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(EhlersEcemaError::AllValuesNaN)?;
    let length = input.get_length();
    let gain_limit = input.get_gain_limit();

    if length == 0 || length > len {
        return Err(EhlersEcemaError::InvalidPeriod {
            period: length,
            data_len: len,
        });
    }

    if len - first < length {
        return Err(EhlersEcemaError::NotEnoughValidData {
            needed: length,
            valid: len - first,
        });
    }

    if gain_limit == 0 {
        return Err(EhlersEcemaError::InvalidGainLimit { gain_limit });
    }

    let alpha = 2.0 / (length as f64 + 1.0);
    let beta = 1.0 - alpha;
    let chosen = if matches!(kernel, Kernel::Auto) {
        Kernel::Scalar
    } else {
        kernel
    };

    Ok((data, length, gain_limit, first, alpha, beta, chosen))
}

#[inline(always)]
fn ehlers_ecema_compute_into_with_mode(
    data: &[f64],
    ema_values: &[f64],
    length: usize,
    gain_limit: usize,
    first: usize,
    alpha: f64,
    beta: f64,
    kernel: Kernel,
    pine_compatible: bool,
    confirmed_only: bool,
    out: &mut [f64],
) {
    unsafe {
        match kernel {
            Kernel::Scalar | Kernel::ScalarBatch => ehlers_ecema_scalar_into_with_mode(
                data,
                ema_values,
                length,
                gain_limit,
                first,
                alpha,
                beta,
                pine_compatible,
                confirmed_only,
                out,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => ehlers_ecema_avx2_into_with_mode(
                data,
                ema_values,
                length,
                gain_limit,
                first,
                alpha,
                beta,
                pine_compatible,
                confirmed_only,
                out,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => ehlers_ecema_avx512_into_with_mode(
                data,
                ema_values,
                length,
                gain_limit,
                first,
                alpha,
                beta,
                pine_compatible,
                confirmed_only,
                out,
            ),
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                ehlers_ecema_scalar_into_with_mode(
                    data,
                    ema_values,
                    length,
                    gain_limit,
                    first,
                    alpha,
                    beta,
                    pine_compatible,
                    confirmed_only,
                    out,
                )
            }
            _ => unreachable!(),
        }
    }
}

pub fn ehlers_ecema_with_kernel(
    input: &EhlersEcemaInput,
    kernel: Kernel,
) -> Result<EhlersEcemaOutput, EhlersEcemaError> {
    let (data, length, gain_limit, first, alpha, beta, chosen) =
        ehlers_ecema_prepare(input, kernel)?;
    let pine_compatible = input.get_pine_compatible();
    let confirmed_only = input.get_confirmed_only();

    let mut ema_buf = alloc_with_nan_prefix(
        data.len(),
        if pine_compatible {
            first
        } else {
            first + length - 1
        },
    );
    if pine_compatible {
        calculate_pine_ema_into(&mut ema_buf, data, length, alpha, beta, first);
    } else {
        let ema_input = EmaInput::from_slice(
            data,
            EmaParams {
                period: Some(length),
            },
        );
        ema_into_slice(&mut ema_buf, &ema_input, chosen)
            .map_err(|e| EhlersEcemaError::EmaError(e.to_string()))?;
    }

    let mut out = alloc_with_nan_prefix(
        data.len(),
        if pine_compatible {
            first
        } else {
            first + length - 1
        },
    );

    ehlers_ecema_compute_into_with_mode(
        data,
        &ema_buf,
        length,
        gain_limit,
        first,
        alpha,
        beta,
        chosen,
        pine_compatible,
        confirmed_only,
        &mut out,
    );

    Ok(EhlersEcemaOutput { values: out })
}

#[inline]
pub fn ehlers_ecema_into_slice(
    dst: &mut [f64],
    input: &EhlersEcemaInput,
    kern: Kernel,
) -> Result<(), EhlersEcemaError> {
    let (data, length, gain_limit, first, alpha, beta, chosen) = ehlers_ecema_prepare(input, kern)?;
    if dst.len() != data.len() {
        return Err(EhlersEcemaError::OutputLengthMismatch {
            expected: data.len(),
            got: dst.len(),
        });
    }
    let pine_compatible = input.get_pine_compatible();
    let confirmed_only = input.get_confirmed_only();

    let mut ema_buf = alloc_with_nan_prefix(
        data.len(),
        if pine_compatible {
            first
        } else {
            first + length - 1
        },
    );
    if pine_compatible {
        calculate_pine_ema_into(&mut ema_buf, data, length, alpha, beta, first);
    } else {
        let ema_input = EmaInput::from_slice(
            data,
            EmaParams {
                period: Some(length),
            },
        );
        ema_into_slice(&mut ema_buf, &ema_input, chosen)
            .map_err(|e| EhlersEcemaError::EmaError(e.to_string()))?;
    }

    let warmup_end = if pine_compatible {
        first
    } else {
        first + length - 1
    };
    let dst_len = dst.len();
    for v in &mut dst[..warmup_end.min(dst_len)] {
        *v = f64::NAN;
    }

    ehlers_ecema_compute_into_with_mode(
        data,
        &ema_buf,
        length,
        gain_limit,
        first,
        alpha,
        beta,
        chosen,
        pine_compatible,
        confirmed_only,
        dst,
    );
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn ehlers_ecema_into(
    input: &EhlersEcemaInput,
    out: &mut [f64],
) -> Result<(), EhlersEcemaError> {
    ehlers_ecema_into_slice(out, input, Kernel::Auto)
}

#[inline(always)]
unsafe fn ehlers_ecema_scalar_into_with_mode(
    data: &[f64],
    ema_values: &[f64],
    length: usize,
    gain_limit: usize,
    first: usize,
    alpha: f64,
    beta: f64,
    pine_compatible: bool,
    confirmed_only: bool,
    out: &mut [f64],
) {
    let len = data.len();
    debug_assert_eq!(out.len(), len);
    debug_assert_eq!(ema_values.len(), len);

    let start_idx = if pine_compatible {
        first
    } else {
        first + length - 1
    };
    if start_idx >= len {
        return;
    }

    let gL: i32 = gain_limit as i32;
    let step_s: f64 = 0.1;
    let data_ptr = data.as_ptr();
    let ema_ptr = ema_values.as_ptr();
    let out_ptr = out.as_mut_ptr();

    for i in start_idx..len {
        let src = if confirmed_only && i > 0 {
            *data_ptr.add(i - 1)
        } else {
            *data_ptr.add(i)
        };

        let ema_i = *ema_ptr.add(i);

        let prev_ec = if i == start_idx {
            if pine_compatible {
                0.0
            } else {
                ema_i
            }
        } else {
            *out_ptr.add(i - 1)
        };

        let delta = src - prev_ec;
        let c = alpha * delta;
        let base = alpha.mul_add(ema_i, beta * prev_ec);
        let d = src - base;
        let s = c * step_s;

        let k_best: i32 = if !s.is_finite() || !d.is_finite() || s.abs() <= f64::MIN_POSITIVE {
            -gL
        } else {
            let k_cont = d / s;

            if k_cont <= (-(gL as f64) - 1.0) {
                -gL
            } else if k_cont >= (gL as f64 + 1.0) {
                gL
            } else {
                let mut k0 = k_cont.floor() as i32;
                let mut k1 = k0 + 1;

                if k0 < -gL {
                    k0 = -gL;
                } else if k0 > gL {
                    k0 = gL;
                }
                if k1 < -gL {
                    k1 = -gL;
                } else if k1 > gL {
                    k1 = gL;
                }

                let e0 = (d - s * (k0 as f64)).abs();
                let e1 = (d - s * (k1 as f64)).abs();

                if e0 <= e1 {
                    k0
                } else {
                    k1
                }
            }
        };

        *out_ptr.add(i) = (k_best as f64).mul_add(s, base);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn ehlers_ecema_avx2_into_with_mode(
    data: &[f64],
    ema_values: &[f64],
    length: usize,
    gain_limit: usize,
    first: usize,
    alpha: f64,
    beta: f64,
    pine_compatible: bool,
    confirmed_only: bool,
    out: &mut [f64],
) {
    ehlers_ecema_scalar_into_with_mode(
        data,
        ema_values,
        length,
        gain_limit,
        first,
        alpha,
        beta,
        pine_compatible,
        confirmed_only,
        out,
    )
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn ehlers_ecema_avx512_into_with_mode(
    data: &[f64],
    ema_values: &[f64],
    length: usize,
    gain_limit: usize,
    first: usize,
    alpha: f64,
    beta: f64,
    pine_compatible: bool,
    confirmed_only: bool,
    out: &mut [f64],
) {
    ehlers_ecema_scalar_into_with_mode(
        data,
        ema_values,
        length,
        gain_limit,
        first,
        alpha,
        beta,
        pine_compatible,
        confirmed_only,
        out,
    )
}

#[derive(Clone, Debug, Default)]
pub struct EhlersEcemaBatchRange {
    pub length: (usize, usize, usize),
    pub gain_limit: (usize, usize, usize),
}

#[derive(Clone, Debug)]
pub struct EhlersEcemaBatchBuilder {
    range: EhlersEcemaBatchRange,
    kernel: Kernel,
}

impl Default for EhlersEcemaBatchBuilder {
    fn default() -> Self {
        Self {
            range: EhlersEcemaBatchRange {
                length: (20, 269, 1),
                gain_limit: (50, 50, 0),
            },
            kernel: Kernel::Auto,
        }
    }
}

impl EhlersEcemaBatchBuilder {
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
    pub fn length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.length = (start, end, step);
        self
    }

    #[inline]
    pub fn length_static(mut self, n: usize) -> Self {
        self.range.length = (n, n, 0);
        self
    }

    #[inline]
    pub fn gain_limit_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.gain_limit = (start, end, step);
        self
    }

    #[inline]
    pub fn gain_limit_static(mut self, g: usize) -> Self {
        self.range.gain_limit = (g, g, 0);
        self
    }

    pub fn apply_slice(self, data: &[f64]) -> Result<EhlersEcemaBatchOutput, EhlersEcemaError> {
        ehlers_ecema_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn apply_candles(
        self,
        c: &Candles,
        src: &str,
    ) -> Result<EhlersEcemaBatchOutput, EhlersEcemaError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }

    pub fn with_default_slice(
        data: &[f64],
        k: Kernel,
    ) -> Result<EhlersEcemaBatchOutput, EhlersEcemaError> {
        EhlersEcemaBatchBuilder::new().kernel(k).apply_slice(data)
    }

    pub fn with_default_candles(c: &Candles) -> Result<EhlersEcemaBatchOutput, EhlersEcemaError> {
        EhlersEcemaBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct EhlersEcemaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<EhlersEcemaParams>,
    pub rows: usize,
    pub cols: usize,
}

impl EhlersEcemaBatchOutput {
    pub fn row_for_params(&self, p: &EhlersEcemaParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.length.unwrap_or(20) == p.length.unwrap_or(20)
                && c.gain_limit.unwrap_or(50) == p.gain_limit.unwrap_or(50)
        })
    }

    pub fn values_for(&self, p: &EhlersEcemaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

pub fn ehlers_ecema_batch_with_kernel(
    data: &[f64],
    sweep: &EhlersEcemaBatchRange,
    k: Kernel,
) -> Result<EhlersEcemaBatchOutput, EhlersEcemaError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(EhlersEcemaError::InvalidKernelForBatch(other)),
    };

    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };

    ehlers_ecema_batch_inner(data, sweep, simd, true)
}

#[inline(always)]
fn expand_grid(r: &EhlersEcemaBatchRange) -> Vec<EhlersEcemaParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        let (lo, hi) = if start <= end {
            (start, end)
        } else {
            (end, start)
        };
        (lo..=hi).step_by(step).collect()
    }

    let lengths = axis_usize(r.length);
    let gain_limits = axis_usize(r.gain_limit);

    let cap = lengths.len().checked_mul(gain_limits.len()).unwrap_or(0);
    let mut out = Vec::with_capacity(cap);
    for &l in &lengths {
        for &g in &gain_limits {
            out.push(EhlersEcemaParams {
                length: Some(l),
                gain_limit: Some(g),
                pine_compatible: Some(false),
                confirmed_only: Some(false),
            });
        }
    }
    out
}

#[inline(always)]
fn ehlers_ecema_batch_inner(
    data: &[f64],
    sweep: &EhlersEcemaBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<EhlersEcemaBatchOutput, EhlersEcemaError> {
    let combos = expand_grid(sweep);
    let cols = data.len();
    let rows = combos.len();

    if cols == 0 {
        return Err(EhlersEcemaError::AllValuesNaN);
    }

    if rows.checked_mul(cols).is_none() {
        return Err(EhlersEcemaError::SizeOverflow { what: "rows*cols" });
    }
    if combos.is_empty() {
        return Err(EhlersEcemaError::InvalidRange {
            start: sweep.length.0,
            end: sweep.length.1,
            step: sweep.length.2,
        });
    }

    let mut buf_mu = make_uninit_matrix(rows, cols);

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(EhlersEcemaError::AllValuesNaN)?;
    let max_len = combos
        .iter()
        .map(|p| p.length.unwrap_or(20))
        .max()
        .unwrap_or(1);
    if cols - first < max_len {
        return Err(EhlersEcemaError::NotEnoughValidData {
            needed: max_len,
            valid: cols - first,
        });
    }

    let warm: Vec<usize> = combos
        .iter()
        .map(|p| first + p.length.unwrap_or(20) - 1)
        .collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    ehlers_ecema_batch_inner_into(data, sweep, kern, parallel, out)?;

    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(EhlersEcemaBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn ehlers_ecema_batch_inner_into(
    data: &[f64],
    sweep: &EhlersEcemaBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<EhlersEcemaParams>, EhlersEcemaError> {
    use std::collections::HashMap;

    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(EhlersEcemaError::InvalidRange {
            start: sweep.length.0,
            end: sweep.length.1,
            step: sweep.length.2,
        });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(EhlersEcemaError::AllValuesNaN)?;
    let cols = data.len();

    let mut ema_cache: HashMap<usize, Vec<f64>> = HashMap::new();

    let unique_lengths: std::collections::HashSet<usize> =
        combos.iter().map(|p| p.length.unwrap_or(20)).collect();

    for &length in &unique_lengths {
        let mut buf = alloc_with_nan_prefix(cols, first + length - 1);
        let ema_in = EmaInput::from_slice(
            data,
            EmaParams {
                period: Some(length),
            },
        );
        ema_into_slice(&mut buf, &ema_in, kern)
            .map_err(|e| EhlersEcemaError::EmaError(e.to_string()))?;
        ema_cache.insert(length, buf);
    }

    let do_row = |row: usize, row_out: &mut [f64]| -> Result<(), EhlersEcemaError> {
        let p = &combos[row];
        let length = p.length.unwrap_or(20);
        let gain_limit = p.gain_limit.unwrap_or(50);
        let alpha = 2.0 / (length as f64 + 1.0);
        let beta = 1.0 - alpha;
        let ema_vals = ema_cache.get(&length).unwrap();
        unsafe {
            ehlers_ecema_scalar_into_with_mode(
                data, ema_vals, length, gain_limit, first, alpha, beta, false, false, row_out,
            );
        }
        Ok(())
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        out.par_chunks_mut(cols)
            .enumerate()
            .try_for_each(|(r, s)| do_row(r, s))?;
        #[cfg(target_arch = "wasm32")]
        for (r, s) in out.chunks_mut(cols).enumerate() {
            do_row(r, s)?;
        }
    } else {
        for (r, s) in out.chunks_mut(cols).enumerate() {
            do_row(r, s)?;
        }
    }

    Ok(combos)
}

#[inline(always)]
pub fn ehlers_ecema_batch_slice(
    data: &[f64],
    sweep: &EhlersEcemaBatchRange,
    kern: Kernel,
) -> Result<EhlersEcemaBatchOutput, EhlersEcemaError> {
    ehlers_ecema_batch_inner(
        data,
        sweep,
        match kern {
            Kernel::Auto => detect_best_kernel(),
            k => k,
        },
        /*parallel=*/ false,
    )
}

#[inline(always)]
pub fn ehlers_ecema_batch_par_slice(
    data: &[f64],
    sweep: &EhlersEcemaBatchRange,
    kern: Kernel,
) -> Result<EhlersEcemaBatchOutput, EhlersEcemaError> {
    ehlers_ecema_batch_inner(
        data,
        sweep,
        match kern {
            Kernel::Auto => detect_best_kernel(),
            k => k,
        },
        /*parallel=*/ true,
    )
}

#[derive(Debug, Clone)]
pub struct EhlersEcemaStream {
    length: usize,
    gain_limit: usize,

    gain_limit_i32: i32,
    alpha: f64,
    beta: f64,
    slope_scale: f64,

    count: usize,
    ema_mean: f64,
    ema_filled: bool,
    warm_sum: f64,
    prev_ecema: f64,

    pine_compatible: bool,
    confirmed_only: bool,
    prev_value: Option<f64>,
}

impl EhlersEcemaStream {
    #[inline]
    pub fn try_new(params: EhlersEcemaParams) -> Result<Self, EhlersEcemaError> {
        let length = params.length.unwrap_or(20);
        let gain_limit = params.gain_limit.unwrap_or(50);
        let pine_compatible = params.pine_compatible.unwrap_or(false);
        let confirmed_only = params.confirmed_only.unwrap_or(false);

        if length == 0 {
            return Err(EhlersEcemaError::InvalidPeriod {
                period: length,
                data_len: 0,
            });
        }
        if gain_limit == 0 {
            return Err(EhlersEcemaError::InvalidGainLimit { gain_limit });
        }

        let alpha = 2.0 / (length as f64 + 1.0);
        let beta = 1.0 - alpha;

        Ok(Self {
            length,
            gain_limit,
            gain_limit_i32: gain_limit as i32,
            alpha,
            beta,
            slope_scale: alpha * 0.1,
            count: 0,
            ema_mean: 0.0,
            ema_filled: false,
            warm_sum: 0.0,
            prev_ecema: 0.0,
            pine_compatible,
            confirmed_only,
            prev_value: None,
        })
    }

    #[inline(always)]
    fn round_nearest_tie_down(x: f64) -> i32 {
        let f = x.floor();
        let r = x - f;
        if r > 0.5 {
            (f + 1.0) as i32
        } else {
            f as i32
        }
    }

    #[inline(always)]
    fn step(&self, prev_ec: f64, src: f64, ema_i: f64) -> f64 {
        let base = self.alpha.mul_add(ema_i, self.beta * prev_ec);

        let delta = src - prev_ec;
        let s = self.slope_scale * delta;

        let d = src - base;

        let k_best: i32 = if !s.is_finite() || !d.is_finite() || s.abs() <= f64::MIN_POSITIVE {
            -self.gain_limit_i32
        } else {
            let k_cont = d / s;
            let k = Self::round_nearest_tie_down(k_cont);
            k.clamp(-self.gain_limit_i32, self.gain_limit_i32)
        };

        (k_best as f64).mul_add(s, base)
    }

    #[inline]
    pub fn next(&mut self, value: f64) -> f64 {
        if !value.is_finite() {
            return f64::NAN;
        }

        let src = if self.confirmed_only {
            match self.prev_value {
                Some(prev) => {
                    self.prev_value = Some(value);
                    prev
                }
                None => {
                    self.prev_value = Some(value);
                    value
                }
            }
        } else {
            value
        };

        self.count += 1;

        if self.pine_compatible {
            self.ema_mean = self.alpha.mul_add(src, self.beta * self.ema_mean);

            let prev_ec = if self.count == 1 {
                0.0
            } else {
                self.prev_ecema
            };
            let ec = self.step(prev_ec, src, self.ema_mean);
            self.prev_ecema = ec;
            return ec;
        }

        if !self.ema_filled {
            self.warm_sum += src;
            if self.count < self.length {
                return f64::NAN;
            }

            self.ema_mean = self.warm_sum / self.length as f64;
            self.ema_filled = true;

            let prev_ec = self.ema_mean;
            let ec = self.step(prev_ec, src, self.ema_mean);
            self.prev_ecema = ec;
            return ec;
        }

        self.ema_mean = self.beta * self.ema_mean + self.alpha * src;
        let ec = self.step(self.prev_ecema, src, self.ema_mean);
        self.prev_ecema = ec;
        ec
    }

    #[inline]
    pub fn reset(&mut self) {
        self.count = 0;
        self.ema_mean = 0.0;
        self.ema_filled = false;
        self.warm_sum = 0.0;
        self.prev_ecema = 0.0;
        self.prev_value = None;
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "ehlers_ecema")]
#[pyo3(signature = (data, length=20, gain_limit=50, pine_compatible=false, confirmed_only=false, kernel=None))]
pub fn ehlers_ecema_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    length: usize,
    gain_limit: usize,
    pine_compatible: bool,
    confirmed_only: bool,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let owned_if_needed: Option<Vec<f64>> = match data.as_slice() {
        Ok(_) => None,
        Err(_) => Some(data.as_array().to_owned().into_raw_vec()),
    };
    let slice_in: &[f64] = match &owned_if_needed {
        Some(v) => v.as_slice(),
        None => data.as_slice()?,
    };
    let kern = validate_kernel(kernel, false)?;
    let params = EhlersEcemaParams {
        length: Some(length),
        gain_limit: Some(gain_limit),
        pine_compatible: Some(pine_compatible),
        confirmed_only: Some(confirmed_only),
    };
    let input = EhlersEcemaInput::from_slice(slice_in, params);

    let result_vec: Vec<f64> = py
        .allow_threads(|| ehlers_ecema_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyfunction(name = "ehlers_ecema_batch")]
#[pyo3(signature = (data, length_range, gain_limit_range, kernel=None))]
pub fn ehlers_ecema_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    gain_limit_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    let slice_in = data.as_slice()?;
    let sweep = EhlersEcemaBatchRange {
        length: length_range,
        gain_limit: gain_limit_range,
    };
    let kern = validate_kernel(kernel, true)?;

    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = slice_in.len();

    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let out_slice = unsafe { out_arr.as_slice_mut()? };

    py.allow_threads(|| {
        let simd = match kern {
            Kernel::Auto => detect_best_batch_kernel(),
            k => k,
        };
        ehlers_ecema_batch_inner_into(slice_in, &sweep, simd, true, out_slice)
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "lengths",
        combos
            .iter()
            .map(|p| p.length.unwrap_or(20) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "gain_limits",
        combos
            .iter()
            .map(|p| p.gain_limit.unwrap_or(50) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "ehlers_ecema_cuda_batch_dev")]
#[pyo3(signature = (data_f32, length_range, gain_limit_range, pine_compatible=false, confirmed_only=false, device_id=0))]
pub fn ehlers_ecema_cuda_batch_dev_py(
    py: Python<'_>,
    data_f32: PyReadonlyArray1<'_, f32>,
    length_range: (usize, usize, usize),
    gain_limit_range: (usize, usize, usize),
    pine_compatible: bool,
    confirmed_only: bool,
    device_id: usize,
) -> PyResult<ecema_python_cuda_handle::EcemaDeviceArrayF32Py> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let slice_in = data_f32.as_slice()?;
    let sweep = EhlersEcemaBatchRange {
        length: length_range,
        gain_limit: gain_limit_range,
    };
    let params = EhlersEcemaParams {
        length: None,
        gain_limit: None,
        pine_compatible: Some(pine_compatible),
        confirmed_only: Some(confirmed_only),
    };

    let (arr, ctx, dev_id) = py.allow_threads(|| {
        let cuda =
            CudaEhlersEcema::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx = cuda.context_arc();
        let dev_id = cuda.device_id();
        let arr = cuda
            .ehlers_ecema_batch_dev(slice_in, &sweep, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok::<_, PyErr>((arr, ctx, dev_id))
    })?;

    Ok(ecema_python_cuda_handle::EcemaDeviceArrayF32Py {
        buf: Some(arr.buf),
        rows: arr.rows,
        cols: arr.cols,
        _ctx: ctx,
        device_id: dev_id,
    })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "ehlers_ecema_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm_f32, length=20, gain_limit=50, pine_compatible=false, confirmed_only=false, device_id=0))]
pub fn ehlers_ecema_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    data_tm_f32: PyReadonlyArray2<'_, f32>,
    length: usize,
    gain_limit: usize,
    pine_compatible: bool,
    confirmed_only: bool,
    device_id: usize,
) -> PyResult<ecema_python_cuda_handle::EcemaDeviceArrayF32Py> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let flat_in = data_tm_f32.as_slice()?;
    let shape = data_tm_f32.shape();
    let rows = shape[0];
    let cols = shape[1];
    let params = EhlersEcemaParams {
        length: Some(length),
        gain_limit: Some(gain_limit),
        pine_compatible: Some(pine_compatible),
        confirmed_only: Some(confirmed_only),
    };

    let (arr, ctx, dev_id) = py.allow_threads(|| {
        let cuda =
            CudaEhlersEcema::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx = cuda.context_arc();
        let dev_id = cuda.device_id();
        let arr = cuda
            .ehlers_ecema_many_series_one_param_time_major_dev(flat_in, cols, rows, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok::<_, PyErr>((arr, ctx, dev_id))
    })?;

    Ok(ecema_python_cuda_handle::EcemaDeviceArrayF32Py {
        buf: Some(arr.buf),
        rows: arr.rows,
        cols: arr.cols,
        _ctx: ctx,
        device_id: dev_id,
    })
}

#[cfg(feature = "python")]
#[pyclass(name = "EhlersEcemaStream")]
pub struct EhlersEcemaStreamPy {
    inner: EhlersEcemaStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl EhlersEcemaStreamPy {
    #[new]
    #[pyo3(signature = (length=20, gain_limit=50, pine_compatible=false, confirmed_only=false))]
    pub fn new(
        length: usize,
        gain_limit: usize,
        pine_compatible: bool,
        confirmed_only: bool,
    ) -> PyResult<Self> {
        let params = EhlersEcemaParams {
            length: Some(length),
            gain_limit: Some(gain_limit),
            pine_compatible: Some(pine_compatible),
            confirmed_only: Some(confirmed_only),
        };
        let stream =
            EhlersEcemaStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner: stream })
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        let y = self.inner.next(value);
        if y.is_nan() {
            None
        } else {
            Some(y)
        }
    }

    pub fn next(&mut self, value: f64) -> Option<f64> {
        self.update(value)
    }

    pub fn reset(&mut self) {
        self.inner.reset()
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct EhlersEcemaBatchConfig {
    pub length_range: (usize, usize, usize),
    pub gain_limit_range: (usize, usize, usize),
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct EhlersEcemaBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<EhlersEcemaParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ehlers_ecema_js(
    data: &[f64],
    length: usize,
    gain_limit: usize,
) -> Result<Vec<f64>, JsValue> {
    let params = EhlersEcemaParams {
        length: Some(length),
        gain_limit: Some(gain_limit),
        pine_compatible: Some(false),
        confirmed_only: Some(false),
    };
    let input = EhlersEcemaInput::from_slice(data, params);

    let mut output = vec![0.0; data.len()];

    ehlers_ecema_into_slice(&mut output, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ehlers_ecema_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ehlers_ecema_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = ehlers_ecema_batch)]
pub fn ehlers_ecema_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let cfg: EhlersEcemaBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = EhlersEcemaBatchRange {
        length: cfg.length_range,
        gain_limit: cfg.gain_limit_range,
    };

    let out = ehlers_ecema_batch_inner(data, &sweep, detect_best_kernel(), false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js = EhlersEcemaBatchJsOutput {
        values: out.values,
        combos: out.combos,
        rows: out.rows,
        cols: out.cols,
    };
    serde_wasm_bindgen::to_value(&js)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ehlers_ecema_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length_start: usize,
    length_end: usize,
    length_step: usize,
    gain_start: usize,
    gain_end: usize,
    gain_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer"));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let sweep = EhlersEcemaBatchRange {
            length: (length_start, length_end, length_step),
            gain_limit: (gain_start, gain_end, gain_step),
        };
        let combos = expand_grid(&sweep);
        let rows = combos.len();
        let cols = len;
        let total = rows
            .checked_mul(cols)
            .ok_or_else(|| JsValue::from_str("size overflow for rows*cols"))?;
        let out = std::slice::from_raw_parts_mut(out_ptr, total);

        ehlers_ecema_batch_inner_into(data, &sweep, detect_best_kernel(), false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ehlers_ecema_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length: usize,
    gain_limit: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to ehlers_ecema_into",
        ));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);

        if length == 0 || length > len {
            return Err(JsValue::from_str("Invalid length"));
        }

        if gain_limit == 0 {
            return Err(JsValue::from_str("Invalid gain limit"));
        }

        let params = EhlersEcemaParams {
            length: Some(length),
            gain_limit: Some(gain_limit),
            pine_compatible: Some(false),
            confirmed_only: Some(false),
        };
        let input = EhlersEcemaInput::from_slice(data, params);

        if in_ptr == out_ptr {
            let mut temp = vec![0.0; len];
            ehlers_ecema_into_slice(&mut temp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            ehlers_ecema_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = ehlers_ecema_into_ex)]
pub fn ehlers_ecema_into_ex(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length: usize,
    gain_limit: usize,
    pine_compatible: bool,
    confirmed_only: bool,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to ehlers_ecema_into_ex",
        ));
    }
    if length == 0 || length > len {
        return Err(JsValue::from_str("Invalid length"));
    }
    if gain_limit == 0 {
        return Err(JsValue::from_str("Invalid gain limit"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let params = EhlersEcemaParams {
            length: Some(length),
            gain_limit: Some(gain_limit),
            pine_compatible: Some(pine_compatible),
            confirmed_only: Some(confirmed_only),
        };
        let input = EhlersEcemaInput::from_slice(data, params);

        if in_ptr == out_ptr {
            let mut temp = vec![0.0; len];
            ehlers_ecema_into_slice(&mut temp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            ehlers_ecema_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    #[cfg(feature = "proptest")]
    use proptest::prelude::*;
    use std::error::Error;

    fn check_ehlers_ecema_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = EhlersEcemaParams {
            length: None,
            gain_limit: None,
            pine_compatible: None,
            confirmed_only: None,
        };
        let input = EhlersEcemaInput::from_candles(&candles, "close", default_params);
        let output = ehlers_ecema_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_ehlers_ecema_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let params = EhlersEcemaParams {
            length: Some(20),
            gain_limit: Some(50),
            pine_compatible: Some(false),
            confirmed_only: Some(false),
        };
        let input = EhlersEcemaInput::from_candles(&candles, "close", params);
        let result = ehlers_ecema_with_kernel(&input, kernel)?;

        assert_eq!(result.values.len(), candles.close.len());

        let first_valid = result.values.iter().position(|x| !x.is_nan());
        assert!(
            first_valid.is_some(),
            "[{}] No valid values found",
            test_name
        );

        let expected_last_five = [
            59368.42792078,
            59311.07435861,
            59212.84931613,
            59221.59111692,
            58978.72640292,
        ];

        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-6,
                "[{}] Regular mode mismatch at idx {}: got {}, expected {}",
                test_name,
                i,
                val,
                expected_last_five[i]
            );
        }

        Ok(())
    }

    fn check_ehlers_ecema_pine_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let params = EhlersEcemaParams {
            length: Some(20),
            gain_limit: Some(50),
            pine_compatible: Some(true),
            confirmed_only: Some(false),
        };
        let input = EhlersEcemaInput::from_candles(&candles, "close", params);
        let result = ehlers_ecema_with_kernel(&input, kernel)?;

        assert_eq!(result.values.len(), candles.close.len());

        let expected_last_five = [
            59368.42792078,
            59311.07435861,
            59212.84931613,
            59221.59111692,
            58978.72640292,
        ];

        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-6,
                "[{}] Pine mode mismatch at idx {}: got {}, expected {}",
                test_name,
                i,
                val,
                expected_last_five[i]
            );
        }

        assert!(
            result.values[0].is_finite(),
            "[{}] Pine mode should have valid value at index 0",
            test_name
        );

        Ok(())
    }

    fn check_ehlers_ecema_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = EhlersEcemaInput::with_default_candles(&candles);
        match input.data {
            EhlersEcemaData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected EhlersEcemaData::Candles"),
        }
        let output = ehlers_ecema_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_ehlers_ecema_zero_period(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = EhlersEcemaParams {
            length: Some(0),
            gain_limit: Some(50),
            pine_compatible: Some(false),
            confirmed_only: Some(false),
        };
        let input = EhlersEcemaInput::from_slice(&input_data, params);
        let res = ehlers_ecema_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] Should fail with zero period", test_name);
        Ok(())
    }

    fn check_ehlers_ecema_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = EhlersEcemaParams {
            length: Some(10),
            gain_limit: Some(50),
            pine_compatible: Some(false),
            confirmed_only: Some(false),
        };
        let input = EhlersEcemaInput::from_slice(&data_small, params);
        let res = ehlers_ecema_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_ehlers_ecema_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = EhlersEcemaParams {
            length: Some(20),
            gain_limit: Some(50),
            pine_compatible: Some(false),
            confirmed_only: Some(false),
        };
        let input = EhlersEcemaInput::from_slice(&single_point, params);
        let res = ehlers_ecema_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_ehlers_ecema_empty_input(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let empty: [f64; 0] = [];
        let input = EhlersEcemaInput::from_slice(&empty, EhlersEcemaParams::default());
        let res = ehlers_ecema_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(EhlersEcemaError::EmptyInputData)),
            "[{}] Should fail with empty input",
            test_name
        );
        Ok(())
    }

    fn check_ehlers_ecema_invalid_gain_limit(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let params = EhlersEcemaParams {
            length: Some(2),
            gain_limit: Some(0),
            pine_compatible: Some(false),
            confirmed_only: Some(false),
        };
        let input = EhlersEcemaInput::from_slice(&data, params);
        let res = ehlers_ecema_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(EhlersEcemaError::InvalidGainLimit { .. })),
            "[{}] Should fail with invalid gain limit",
            test_name
        );
        Ok(())
    }

    fn check_ehlers_ecema_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let first_params = EhlersEcemaParams {
            length: Some(10),
            gain_limit: Some(30),
            pine_compatible: Some(false),
            confirmed_only: Some(false),
        };
        let first_input = EhlersEcemaInput::from_candles(&candles, "close", first_params);
        let first_result = ehlers_ecema_with_kernel(&first_input, kernel)?;

        let second_params = EhlersEcemaParams {
            length: Some(10),
            gain_limit: Some(30),
            pine_compatible: Some(false),
            confirmed_only: Some(false),
        };
        let second_input = EhlersEcemaInput::from_slice(&first_result.values, second_params);
        let second_result = ehlers_ecema_with_kernel(&second_input, kernel)?;

        assert_eq!(second_result.values.len(), first_result.values.len());

        let expected_last_five = [
            59324.20351585,
            59282.79818999,
            59207.38519971,
            59194.22630265,
            59025.67038012,
        ];

        let start = second_result.values.len().saturating_sub(5);
        for (i, &val) in second_result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-6,
                "[{}] Reinput mismatch at idx {}: got {}, expected {}",
                test_name,
                i,
                val,
                expected_last_five[i]
            );
        }

        let valid_count = second_result
            .values
            .iter()
            .skip(20)
            .filter(|x| x.is_finite())
            .count();
        assert!(
            valid_count > 0,
            "[{}] No valid values after reinput",
            test_name
        );

        Ok(())
    }

    fn check_ehlers_ecema_nan_handling(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = EhlersEcemaInput::from_candles(
            &candles,
            "close",
            EhlersEcemaParams {
                length: Some(20),
                gain_limit: Some(50),
                pine_compatible: Some(false),
                confirmed_only: Some(false),
            },
        );
        let res = ehlers_ecema_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), candles.close.len());

        if res.values.len() > 40 {
            for (i, &val) in res.values[40..].iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "[{}] Found unexpected NaN at index {}",
                    test_name,
                    40 + i
                );
            }
        }
        Ok(())
    }

    fn check_ehlers_ecema_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let length = 20;
        let gain_limit = 50;

        let input = EhlersEcemaInput::from_candles(
            &candles,
            "close",
            EhlersEcemaParams {
                length: Some(length),
                gain_limit: Some(gain_limit),
                pine_compatible: Some(false),
                confirmed_only: Some(false),
            },
        );
        let batch_output = ehlers_ecema_with_kernel(&input, kernel)?.values;

        let mut stream = EhlersEcemaStream::try_new(EhlersEcemaParams {
            length: Some(length),
            gain_limit: Some(gain_limit),
            pine_compatible: Some(false),
            confirmed_only: Some(false),
        })?;

        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            stream_values.push(stream.next(price));
        }

        assert_eq!(batch_output.len(), stream_values.len());

        for (i, (&b, &s)) in batch_output.iter().zip(stream_values.iter()).enumerate() {
            if i < length - 1 {
                assert!(
                    b.is_nan() || s.is_nan(),
                    "[{}] Expected NaN during warmup at {}: batch={}, stream={}",
                    test_name,
                    i,
                    b,
                    s
                );
            } else if i >= length && b.is_finite() && s.is_finite() {
                let diff = (b - s).abs();
                let relative_diff = diff / b.abs().max(1.0);
                assert!(
                    relative_diff < 0.001,
                    "[{}] Streaming mismatch at idx {}: batch={}, stream={}, diff={}, rel_diff={}",
                    test_name,
                    i,
                    b,
                    s,
                    diff,
                    relative_diff
                );
            }
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_ehlers_ecema_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let test_params = vec![
            EhlersEcemaParams::default(),
            EhlersEcemaParams {
                length: Some(10),
                gain_limit: Some(30),
                pine_compatible: Some(false),
                confirmed_only: Some(false),
            },
            EhlersEcemaParams {
                length: Some(20),
                gain_limit: Some(50),
                pine_compatible: Some(false),
                confirmed_only: Some(false),
            },
            EhlersEcemaParams {
                length: Some(30),
                gain_limit: Some(100),
                pine_compatible: Some(false),
                confirmed_only: Some(false),
            },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = EhlersEcemaInput::from_candles(&candles, "close", params.clone());
            let output = ehlers_ecema_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
                        with params: length={}, gain_limit={}",
                        test_name,
                        val,
                        bits,
                        i,
                        params.length.unwrap_or(20),
                        params.gain_limit.unwrap_or(50)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
                        with params: length={}, gain_limit={}",
                        test_name,
                        val,
                        bits,
                        i,
                        params.length.unwrap_or(20),
                        params.gain_limit.unwrap_or(50)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
                        with params: length={}, gain_limit={}",
                        test_name,
                        val,
                        bits,
                        i,
                        params.length.unwrap_or(20),
                        params.gain_limit.unwrap_or(50)
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_ehlers_ecema_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_ehlers_ecema_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let strat = (5usize..=100).prop_flat_map(|length| {
            (
                prop::collection::vec(
                    (-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                    length..400,
                ),
                Just(length),
                10usize..=100,
            )
        });

        proptest::test_runner::TestRunner::default().run(
            &strat,
            |(data, length, gain_limit)| {
                let params = EhlersEcemaParams {
                    length: Some(length),
                    gain_limit: Some(gain_limit),
                    pine_compatible: Some(false),
                    confirmed_only: Some(false),
                };
                let input = EhlersEcemaInput::from_slice(&data, params);

                let EhlersEcemaOutput { values: out } =
                    ehlers_ecema_with_kernel(&input, kernel).unwrap();
                let EhlersEcemaOutput { values: ref_out } =
                    ehlers_ecema_with_kernel(&input, Kernel::Scalar).unwrap();

                assert_eq!(
                    out.len(),
                    data.len(),
                    "[{}] Output length mismatch",
                    test_name
                );
                assert_eq!(
                    ref_out.len(),
                    data.len(),
                    "[{}] Reference output length mismatch",
                    test_name
                );

                for i in (length - 1)..data.len() {
                    if out[i].is_finite() && ref_out[i].is_finite() {
                        let diff = (out[i] - ref_out[i]).abs();
                        let relative_diff = diff / ref_out[i].abs().max(1.0);
                        assert!(
                            relative_diff < 1e-10,
                            "[{}] Kernel mismatch at idx {}: {} vs {} (diff={})",
                            test_name,
                            i,
                            out[i],
                            ref_out[i],
                            diff
                        );
                    }
                }

                for i in 0..(length - 1) {
                    assert!(
                        out[i].is_nan(),
                        "[{}] Expected NaN during warmup at idx {}",
                        test_name,
                        i
                    );
                }

                Ok(())
            },
        )?;

        Ok(())
    }

    macro_rules! generate_all_ehlers_ecema_tests {
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
                #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                $(
                    #[test]
                    fn [<$test_fn _simd128_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _simd128_f64>]), Kernel::Scalar);
                    }
                )*
            }
        }
    }

    generate_all_ehlers_ecema_tests!(
        check_ehlers_ecema_partial_params,
        check_ehlers_ecema_accuracy,
        check_ehlers_ecema_pine_accuracy,
        check_ehlers_ecema_default_candles,
        check_ehlers_ecema_zero_period,
        check_ehlers_ecema_period_exceeds_length,
        check_ehlers_ecema_very_small_dataset,
        check_ehlers_ecema_empty_input,
        check_ehlers_ecema_invalid_gain_limit,
        check_ehlers_ecema_reinput,
        check_ehlers_ecema_nan_handling,
        check_ehlers_ecema_streaming,
        check_ehlers_ecema_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_ehlers_ecema_tests!(check_ehlers_ecema_property);

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = EhlersEcemaBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = EhlersEcemaParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        let valid_count = row.iter().skip(40).filter(|x| x.is_finite()).count();
        assert!(valid_count > 0, "[{}] No valid values in default row", test);

        Ok(())
    }

    fn check_batch_sweep(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = EhlersEcemaBatchBuilder::new()
            .kernel(kernel)
            .length_range(10, 30, 5)
            .gain_limit_range(30, 70, 10)
            .apply_candles(&c, "close")?;

        let expected_combos = 5 * 5;
        assert_eq!(output.combos.len(), expected_combos);
        assert_eq!(output.rows, expected_combos);
        assert_eq!(output.cols, c.close.len());

        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let test_configs = vec![
            (10, 20, 5, 30, 50, 10),
            (20, 20, 0, 50, 50, 0),
            (15, 25, 2, 40, 60, 5),
        ];

        for (cfg_idx, &(l_start, l_end, l_step, g_start, g_end, g_step)) in
            test_configs.iter().enumerate()
        {
            let output = EhlersEcemaBatchBuilder::new()
                .kernel(kernel)
                .length_range(l_start, l_end, l_step)
                .gain_limit_range(g_start, g_end, g_step)
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
                        at row {} col {} (flat index {}) with params: length={}, gain_limit={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.length.unwrap_or(20),
                        combo.gain_limit.unwrap_or(50)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
                        at row {} col {} (flat index {}) with params: length={}, gain_limit={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.length.unwrap_or(20),
                        combo.gain_limit.unwrap_or(50)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
                        at row {} col {} (flat index {}) with params: length={}, gain_limit={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.length.unwrap_or(20),
                        combo.gain_limit.unwrap_or(50)
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
    gen_batch_tests!(check_batch_sweep);
    gen_batch_tests!(check_batch_no_poison);

    #[test]
    fn test_ehlers_ecema_into_matches_api() -> Result<(), Box<dyn Error>> {
        let len = 256usize;
        let mut data = vec![0.0f64; len];

        for i in 0..3 {
            data[i] = f64::NAN;
        }
        for i in 3..len {
            let x = i as f64;
            data[i] = 1000.0 + (x * 0.1).sin() * 5.0 + x * 0.05;
        }

        let params = EhlersEcemaParams::default();
        let input = EhlersEcemaInput::from_slice(&data, params);

        let baseline = ehlers_ecema(&input)?.values;

        let mut out = vec![0.0f64; len];

        #[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
        {
            ehlers_ecema_into(&input, &mut out)?;
        }
        #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
        {
            ehlers_ecema_into_slice(&mut out, &input, Kernel::Auto)?;
        }

        assert_eq!(baseline.len(), out.len());

        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b) || ((a - b).abs() <= 1e-12)
        }

        for i in 0..len {
            assert!(
                eq_or_both_nan(baseline[i], out[i]),
                "mismatch at idx {}: baseline={} out={}",
                i,
                baseline[i],
                out[i]
            );
        }

        Ok(())
    }
}
