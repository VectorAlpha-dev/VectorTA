#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::moving_averages::alma_wrapper::DeviceArrayF32;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::moving_averages::CudaTrendflex;
use crate::utilities::data_loader::{source_type, Candles};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::utilities::dlpack_cuda::export_f32_cuda_dlpack_2d;
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
use aligned_vec::{AVec, ConstAlign, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::context::Context;
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::memory::DeviceBuffer;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
#[cfg(all(feature = "python", feature = "cuda"))]
use std::sync::Arc;
use thiserror::Error;

impl<'a> AsRef<[f64]> for TrendFlexInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            TrendFlexData::Slice(slice) => slice,
            TrendFlexData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum TrendFlexData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct TrendFlexOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct TrendFlexParams {
    pub period: Option<usize>,
}

impl Default for TrendFlexParams {
    fn default() -> Self {
        Self { period: Some(20) }
    }
}

#[derive(Debug, Clone)]
pub struct TrendFlexInput<'a> {
    pub data: TrendFlexData<'a>,
    pub params: TrendFlexParams,
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyo3::prelude::pyclass(
    module = "ta_indicators.cuda",
    name = "TrendFlexDeviceArrayF32",
    unsendable
)]
pub struct TrendFlexDeviceArrayF32Py {
    pub(crate) inner: DeviceArrayF32,
    pub(crate) _ctx: Arc<Context>,
    pub(crate) device_id: u32,
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyo3::prelude::pymethods]
impl TrendFlexDeviceArrayF32Py {
    #[getter]
    fn __cuda_array_interface__<'py>(
        &self,
        py: pyo3::prelude::Python<'py>,
    ) -> pyo3::PyResult<pyo3::prelude::Bound<'py, pyo3::types::PyDict>> {
        let d = pyo3::types::PyDict::new(py);
        d.set_item("shape", (self.inner.rows, self.inner.cols))?;
        d.set_item("typestr", "<f4")?;
        d.set_item(
            "strides",
            (
                self.inner.cols * std::mem::size_of::<f32>(),
                std::mem::size_of::<f32>(),
            ),
        )?;
        d.set_item("data", (self.inner.device_ptr() as usize, false))?;

        d.set_item("version", 3)?;
        Ok(d)
    }

    fn __dlpack_device__(&self) -> (i32, i32) {
        (2, self.device_id as i32)
    }

    #[pyo3(signature=(stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__<'py>(
        &mut self,
        py: pyo3::prelude::Python<'py>,
        stream: Option<pyo3::PyObject>,
        max_version: Option<pyo3::PyObject>,
        dl_device: Option<pyo3::PyObject>,
        copy: Option<pyo3::PyObject>,
    ) -> pyo3::PyResult<pyo3::prelude::PyObject> {
        let (kdl, alloc_dev) = self.__dlpack_device__();
        if let Some(dev_obj) = dl_device.as_ref() {
            if let Ok((dev_ty, dev_id)) = dev_obj.extract::<(i32, i32)>(py) {
                if dev_ty != kdl || dev_id != alloc_dev {
                    let wants_copy = copy
                        .as_ref()
                        .and_then(|c| c.extract::<bool>(py).ok())
                        .unwrap_or(false);
                    if wants_copy {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "device copy not implemented for __dlpack__",
                        ));
                    } else {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "dl_device mismatch for __dlpack__",
                        ));
                    }
                }
            }
        }
        let _ = stream;

        let dummy = DeviceBuffer::from_slice(&[])
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
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

impl<'a> TrendFlexInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: TrendFlexParams) -> Self {
        Self {
            data: TrendFlexData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: TrendFlexParams) -> Self {
        Self {
            data: TrendFlexData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", TrendFlexParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(20)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct TrendFlexBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for TrendFlexBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl TrendFlexBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<TrendFlexOutput, TrendFlexError> {
        let p = TrendFlexParams {
            period: self.period,
        };
        let i = TrendFlexInput::from_candles(c, "close", p);
        trendflex_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<TrendFlexOutput, TrendFlexError> {
        let p = TrendFlexParams {
            period: self.period,
        };
        let i = TrendFlexInput::from_slice(d, p);
        trendflex_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<TrendFlexStream, TrendFlexError> {
        let p = TrendFlexParams {
            period: self.period,
        };
        TrendFlexStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum TrendFlexError {
    #[error("trendflex: No data provided.")]
    NoDataProvided,
    #[error("trendflex: All values are NaN.")]
    AllValuesNaN,
    #[error("trendflex: period = 0")]
    ZeroTrendFlexPeriod { period: usize },
    #[error("trendflex: period > data len: period = {period}, data_len = {data_len}")]
    TrendFlexPeriodExceedsData { period: usize, data_len: usize },
    #[error(
        "trendflex: smoother period > data len: ss_period = {ss_period}, data_len = {data_len}"
    )]
    SmootherPeriodExceedsData { ss_period: usize, data_len: usize },
    #[error("trendflex: output length mismatch: expected {expected}, got {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("trendflex: not enough valid data: needed {needed}, valid {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("trendflex: invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("trendflex: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("trendflex: dimensions overflow: rows={rows}, cols={cols}")]
    DimensionsOverflow { rows: usize, cols: usize },
}

#[inline]
pub fn trendflex(input: &TrendFlexInput) -> Result<TrendFlexOutput, TrendFlexError> {
    trendflex_with_kernel(input, Kernel::Auto)
}

pub fn trendflex_with_kernel(
    input: &TrendFlexInput,
    kernel: Kernel,
) -> Result<TrendFlexOutput, TrendFlexError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(TrendFlexError::NoDataProvided);
    }

    let period = input.get_period();
    if period == 0 {
        return Err(TrendFlexError::ZeroTrendFlexPeriod { period });
    }
    if period >= len {
        return Err(TrendFlexError::TrendFlexPeriodExceedsData {
            period,
            data_len: len,
        });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(TrendFlexError::AllValuesNaN)?;
    let ss_period = ((period as f64) / 2.0).round() as usize;

    let valid = len - first;
    if valid < period {
        return Err(TrendFlexError::NotEnoughValidData {
            needed: period,
            valid,
        });
    }
    if ss_period > len {
        return Err(TrendFlexError::SmootherPeriodExceedsData {
            ss_period,
            data_len: len,
        });
    }

    let warm = first + period;
    let mut out = alloc_with_nan_prefix(len, warm);

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                trendflex_scalar_into(data, period, ss_period, first, &mut out)?
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                trendflex_avx2_into(data, period, ss_period, first, &mut out)?
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                trendflex_avx512_into(data, period, ss_period, first, &mut out)?
            }
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                trendflex_scalar_into(data, period, ss_period, first, &mut out)?
            }
            Kernel::Auto => unreachable!(),
        }
    }

    Ok(TrendFlexOutput { values: out })
}

pub fn trendflex_into_slice(
    dst: &mut [f64],
    input: &TrendFlexInput,
    kernel: Kernel,
) -> Result<(), TrendFlexError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if dst.len() != len {
        return Err(TrendFlexError::OutputLengthMismatch {
            expected: len,
            got: dst.len(),
        });
    }
    if len == 0 {
        return Err(TrendFlexError::NoDataProvided);
    }
    let period = input.get_period();
    if period == 0 {
        return Err(TrendFlexError::ZeroTrendFlexPeriod { period });
    }
    if period >= len {
        return Err(TrendFlexError::TrendFlexPeriodExceedsData {
            period,
            data_len: len,
        });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(TrendFlexError::AllValuesNaN)?;
    let ss_period = ((period as f64) / 2.0).round() as usize;
    let valid = len - first;
    if valid < period {
        return Err(TrendFlexError::NotEnoughValidData {
            needed: period,
            valid,
        });
    }
    if ss_period > data.len() {
        return Err(TrendFlexError::SmootherPeriodExceedsData {
            ss_period,
            data_len: data.len(),
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                trendflex_scalar_into(data, period, ss_period, first, dst)?
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                trendflex_avx2_into(data, period, ss_period, first, dst)?
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                trendflex_avx512_into(data, period, ss_period, first, dst)?
            }
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                trendflex_scalar_into(data, period, ss_period, first, dst)?
            }
            Kernel::Auto => unreachable!(),
        }
    }

    let warmup_end = first + period;
    for v in &mut dst[..warmup_end] {
        *v = f64::NAN;
    }
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn trendflex_into(input: &TrendFlexInput, out: &mut [f64]) -> Result<(), TrendFlexError> {
    trendflex_into_slice(out, input, Kernel::Auto)
}

#[inline]
unsafe fn trendflex_scalar_into(
    data: &[f64],
    period: usize,
    ss_period: usize,
    first_valid: usize,
    out: &mut [f64],
) -> Result<(), TrendFlexError> {
    use std::f64::consts::PI;

    let len = data.len();
    let warm = first_valid + period;

    for i in 0..warm.min(out.len()) {
        out[i] = f64::NAN;
    }

    if first_valid >= len {
        return Ok(());
    }

    let a = (-1.414_f64 * PI / ss_period as f64).exp();
    let a_sq = a * a;
    let b = 2.0 * a * (1.414_f64 * PI / ss_period as f64).cos();

    let c = (1.0 + a_sq - b) * 0.5;

    let m = len - first_valid;
    if m < period {
        return Ok(());
    }
    if m < ss_period {
        return Err(TrendFlexError::SmootherPeriodExceedsData {
            ss_period,
            data_len: m,
        });
    }

    let x = &data[first_valid..];

    let mut prev2 = x[0];
    let mut prev1 = if m > 1 { x[1] } else { x[0] };

    let mut ring = vec![0.0f64; period];
    let mut head = 0usize;
    let mut sum = 0.0f64;

    ring[head] = prev2;
    sum += prev2;
    head = (head + 1) % period;
    if m > 1 {
        ring[head] = prev1;
        sum += prev1;
        head = (head + 1) % period;
    }

    let tp_f = period as f64;
    let inv_tp = 1.0 / tp_f;
    let mut ms_prev = 0.0f64;

    let mut i = 2usize;
    while i < m && i < period {
        let cur = (-a_sq).mul_add(prev2, b.mul_add(prev1, c * (x[i] + x[i - 1])));
        prev2 = prev1;
        prev1 = cur;

        sum += cur;
        ring[head] = cur;
        head = (head + 1) % period;
        i += 1;
    }

    while i < m {
        let cur = (-a_sq).mul_add(prev2, b.mul_add(prev1, c * (x[i] + x[i - 1])));
        prev2 = prev1;
        prev1 = cur;

        let my_sum = (tp_f * cur - sum) * inv_tp;

        let ms_current = 0.04f64.mul_add(my_sum * my_sum, 0.96f64 * ms_prev);
        ms_prev = ms_current;

        let out_val = if ms_current != 0.0 {
            my_sum / ms_current.sqrt()
        } else {
            0.0
        };
        out[first_valid + i] = out_val;

        let old = ring[head];
        sum += cur - old;
        ring[head] = cur;
        head = (head + 1) % period;

        i += 1;
    }

    Ok(())
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn trendflex_avx2_into(
    data: &[f64],
    period: usize,
    ss_period: usize,
    first_valid: usize,
    out: &mut [f64],
) -> Result<(), TrendFlexError> {
    use std::f64::consts::PI;

    let len = data.len();
    let warm = first_valid + period;
    for i in 0..warm.min(out.len()) {
        *out.get_unchecked_mut(i) = f64::NAN;
    }

    if first_valid >= len {
        return Ok(());
    }

    let a = (-1.414_f64 * PI / ss_period as f64).exp();
    let a_sq = a * a;
    let b = 2.0 * a * (1.414_f64 * PI / ss_period as f64).cos();
    let c = (1.0 + a_sq - b) * 0.5;

    #[inline(always)]
    unsafe fn run_series_avx2(
        x: &[f64],
        period: usize,
        a_sq: f64,
        b: f64,
        c: f64,
        out: &mut [f64],
        out_off: usize,
    ) {
        let n = x.len();
        if n == 0 {
            return;
        }
        let mut prev2 = x[0];
        let mut prev1 = if n > 1 { x[1] } else { x[0] };

        let mut ring = vec![0.0f64; period];
        let mut sum = 0.0f64;
        let mut head = 0usize;

        ring[head] = prev2;
        sum += prev2;
        head = (head + 1) % period;
        if n > 1 {
            ring[head] = prev1;
            sum += prev1;
            head = (head + 1) % period;
        }

        let tp_f = period as f64;
        let inv_tp = 1.0 / tp_f;
        let mut ms_prev = 0.0f64;

        let mut i = 2usize;
        while i < n && i < period {
            let cur = c * (x[i] + x[i - 1]) + b * prev1 - a_sq * prev2;
            prev2 = prev1;
            prev1 = cur;
            sum += cur;
            ring[head] = cur;
            head = (head + 1) % period;
            i += 1;
        }

        while i < n {
            _mm_prefetch(x.as_ptr().add(i + 16).cast(), _MM_HINT_T0);
            let cur = c * (x[i] + x[i - 1]) + b * prev1 - a_sq * prev2;
            prev2 = prev1;
            prev1 = cur;

            let my_sum = (tp_f * cur - sum) * inv_tp;

            let v = _mm_set_sd(my_sum);
            let sq = _mm_mul_sd(v, v);
            let s04 = _mm_mul_sd(_mm_set_sd(0.04), sq);
            let s96 = _mm_mul_sd(_mm_set_sd(0.96), _mm_set_sd(ms_prev));
            let ms_cur = _mm_add_sd(s04, s96);
            let ms_current = _mm_cvtsd_f64(ms_cur);
            ms_prev = ms_current;

            let out_val = if ms_current != 0.0 {
                let denom = _mm_sqrt_sd(_mm_setzero_pd(), _mm_set_sd(ms_current));
                let denom_s = _mm_cvtsd_f64(denom);
                my_sum / denom_s
            } else {
                0.0
            };

            _mm_stream_sd(
                out.get_unchecked_mut(out_off + i) as *mut f64,
                _mm_set_sd(out_val),
            );

            let old = ring[head];
            sum += cur - old;
            ring[head] = cur;
            head = (head + 1) % period;

            i += 1;
        }
    }

    if first_valid == 0 {
        run_series_avx2(data, period, a_sq, b, c, out, 0);
        return Ok(());
    }

    let m = len - first_valid;
    if m < period {
        return Ok(());
    }
    if m < ss_period {
        return Err(TrendFlexError::SmootherPeriodExceedsData {
            ss_period,
            data_len: m,
        });
    }
    let tail = &data[first_valid..];
    run_series_avx2(tail, period, a_sq, b, c, out, first_valid);
    Ok(())
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f,avx512dq,fma")]
unsafe fn trendflex_avx512_into(
    data: &[f64],
    period: usize,
    ss_period: usize,
    first_valid: usize,
    out: &mut [f64],
) -> Result<(), TrendFlexError> {
    use std::f64::consts::PI;

    let len = data.len();
    let warm = first_valid + period;
    for i in 0..warm.min(out.len()) {
        *out.get_unchecked_mut(i) = f64::NAN;
    }

    if first_valid >= len {
        return Ok(());
    }

    let a = (-1.414_f64 * PI / ss_period as f64).exp();
    let a_sq = a * a;
    let b = 2.0 * a * (1.414_f64 * PI / ss_period as f64).cos();
    let c = (1.0 + a_sq - b) * 0.5;

    #[inline(always)]
    unsafe fn run_series_avx512(
        x: &[f64],
        period: usize,
        a_sq: f64,
        b: f64,
        c: f64,
        out: &mut [f64],
        out_off: usize,
    ) {
        let n = x.len();
        if n == 0 {
            return;
        }
        let mut prev2 = *x.get_unchecked(0);
        let mut prev1 = if n > 1 {
            *x.get_unchecked(1)
        } else {
            *x.get_unchecked(0)
        };
        let mut ring = vec![0.0f64; period];
        let mut sum = 0.0f64;
        let mut head = 0usize;

        *ring.get_unchecked_mut(head) = prev2;
        sum += prev2;
        head += 1;
        if head == period {
            head = 0;
        }
        if n > 1 {
            *ring.get_unchecked_mut(head) = prev1;
            sum += prev1;
            head += 1;
            if head == period {
                head = 0;
            }
        }

        let tp_f = period as f64;
        let inv_tp = 1.0 / tp_f;
        let mut ms_prev = 0.0f64;

        let mut i = 2usize;
        while i < n && i < period {
            let cur =
                c * (*x.get_unchecked(i) + *x.get_unchecked(i - 1)) + b * prev1 - a_sq * prev2;
            prev2 = prev1;
            prev1 = cur;
            sum += cur;
            *ring.get_unchecked_mut(head) = cur;
            head += 1;
            if head == period {
                head = 0;
            }
            i += 1;
        }

        let use_stream = n >= 131072;
        let use_unroll = n >= 262144;

        if use_unroll {
            while i + 1 < n {
                _mm_prefetch(x.as_ptr().add(i + 32).cast(), _MM_HINT_T0);

                let cur0 =
                    c * (*x.get_unchecked(i) + *x.get_unchecked(i - 1)) + b * prev1 - a_sq * prev2;
                prev2 = prev1;
                prev1 = cur0;

                let my_sum0 = (tp_f * cur0 - sum) * inv_tp;

                let v0 = _mm_set_sd(my_sum0);
                let sq0 = _mm_mul_sd(v0, v0);
                let ms0 = _mm_fmadd_sd(
                    _mm_set_sd(0.04),
                    sq0,
                    _mm_mul_sd(_mm_set_sd(0.96), _mm_set_sd(ms_prev)),
                );
                let ms0_s = _mm_cvtsd_f64(ms0);
                ms_prev = ms0_s;
                let out0 = if ms0_s != 0.0 {
                    let den0 = _mm_sqrt_sd(_mm_setzero_pd(), _mm_set_sd(ms0_s));
                    my_sum0 / _mm_cvtsd_f64(den0)
                } else {
                    0.0
                };
                if use_stream {
                    _mm_stream_sd(
                        out.get_unchecked_mut(out_off + i) as *mut f64,
                        _mm_set_sd(out0),
                    );
                } else {
                    *out.get_unchecked_mut(out_off + i) = out0;
                }

                let old0 = *ring.get_unchecked(head);
                sum += cur0 - old0;
                *ring.get_unchecked_mut(head) = cur0;
                head += 1;
                if head == period {
                    head = 0;
                }

                let cur1 =
                    c * (*x.get_unchecked(i + 1) + *x.get_unchecked(i)) + b * prev1 - a_sq * prev2;
                prev2 = prev1;
                prev1 = cur1;

                let my_sum1 = (tp_f * cur1 - sum) * inv_tp;
                let v1 = _mm_set_sd(my_sum1);
                let sq1 = _mm_mul_sd(v1, v1);
                let ms1 = _mm_fmadd_sd(
                    _mm_set_sd(0.04),
                    sq1,
                    _mm_mul_sd(_mm_set_sd(0.96), _mm_set_sd(ms_prev)),
                );
                let ms1_s = _mm_cvtsd_f64(ms1);
                ms_prev = ms1_s;
                let out1 = if ms1_s != 0.0 {
                    let den1 = _mm_sqrt_sd(_mm_setzero_pd(), _mm_set_sd(ms1_s));
                    my_sum1 / _mm_cvtsd_f64(den1)
                } else {
                    0.0
                };
                if use_stream {
                    _mm_stream_sd(
                        out.get_unchecked_mut(out_off + i + 1) as *mut f64,
                        _mm_set_sd(out1),
                    );
                } else {
                    *out.get_unchecked_mut(out_off + i + 1) = out1;
                }

                let old1 = *ring.get_unchecked(head);
                sum += cur1 - old1;
                *ring.get_unchecked_mut(head) = cur1;
                head += 1;
                if head == period {
                    head = 0;
                }

                i += 2;
            }
        }

        while i < n {
            _mm_prefetch(x.as_ptr().add(i + 32).cast(), _MM_HINT_T0);
            let cur =
                c * (*x.get_unchecked(i) + *x.get_unchecked(i - 1)) + b * prev1 - a_sq * prev2;
            prev2 = prev1;
            prev1 = cur;

            let my_sum = (tp_f * cur - sum) * inv_tp;
            let v = _mm_set_sd(my_sum);
            let sq = _mm_mul_sd(v, v);
            let ms = _mm_fmadd_sd(
                _mm_set_sd(0.04),
                sq,
                _mm_mul_sd(_mm_set_sd(0.96), _mm_set_sd(ms_prev)),
            );
            let ms_s = _mm_cvtsd_f64(ms);
            ms_prev = ms_s;
            let out_val = if ms_s != 0.0 {
                let den = _mm_sqrt_sd(_mm_setzero_pd(), _mm_set_sd(ms_s));
                my_sum / _mm_cvtsd_f64(den)
            } else {
                0.0
            };
            if use_stream {
                _mm_stream_sd(
                    out.get_unchecked_mut(out_off + i) as *mut f64,
                    _mm_set_sd(out_val),
                );
            } else {
                *out.get_unchecked_mut(out_off + i) = out_val;
            }

            let old = *ring.get_unchecked(head);
            sum += cur - old;
            *ring.get_unchecked_mut(head) = cur;
            head += 1;
            if head == period {
                head = 0;
            }

            i += 1;
        }
    }

    if first_valid == 0 {
        run_series_avx512(data, period, a_sq, b, c, out, 0);
        return Ok(());
    }

    let m = len - first_valid;
    if m < period {
        return Ok(());
    }
    if m < ss_period {
        return Err(TrendFlexError::SmootherPeriodExceedsData {
            ss_period,
            data_len: m,
        });
    }
    let tail = &data[first_valid..];
    run_series_avx512(tail, period, a_sq, b, c, out, first_valid);
    Ok(())
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
unsafe fn trendflex_avx512_short_into(
    data: &[f64],
    period: usize,
    ss_period: usize,
    first_valid: usize,
    out: &mut [f64],
) -> Result<(), TrendFlexError> {
    trendflex_scalar_into(data, period, ss_period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
unsafe fn trendflex_avx512_long_into(
    data: &[f64],
    period: usize,
    ss_period: usize,
    first_valid: usize,
    out: &mut [f64],
) -> Result<(), TrendFlexError> {
    trendflex_scalar_into(data, period, ss_period, first_valid, out)
}

#[derive(Debug, Clone)]
pub struct TrendFlexStream {
    period: usize,
    ss_period: usize,

    a: f64,
    a_sq: f64,
    b: f64,
    c: f64,

    buf: Vec<f64>,
    sum: f64,
    head: usize,

    prev1_ssf: f64,
    prev2_ssf: f64,
    last_raw: f64,

    n_ssf: usize,

    ms_prev: f64,

    inv_p: f64,
}

impl TrendFlexStream {
    pub fn try_new(params: TrendFlexParams) -> Result<Self, TrendFlexError> {
        let period = params.period.unwrap_or(20);
        if period == 0 {
            return Err(TrendFlexError::ZeroTrendFlexPeriod { period });
        }

        let ss_period = ((period as f64) / 2.0).round() as usize;
        if ss_period == 0 {
            return Err(TrendFlexError::SmootherPeriodExceedsData {
                ss_period,
                data_len: 0,
            });
        }

        use std::f64::consts::PI;
        let a = (-1.414_f64 * PI / (ss_period as f64)).exp();
        let a_sq = a * a;
        let b = 2.0 * a * (1.414_f64 * PI / (ss_period as f64)).cos();
        let c = (1.0 + a_sq - b) * 0.5;

        Ok(Self {
            period,
            ss_period,
            a,
            a_sq,
            b,
            c,
            buf: vec![0.0; period],
            sum: 0.0,
            head: 0,
            prev1_ssf: 0.0,
            prev2_ssf: 0.0,
            last_raw: 0.0,
            n_ssf: 0,
            ms_prev: 0.0,
            inv_p: 1.0 / (period as f64),
        })
    }

    #[inline(always)]
    pub fn update(&mut self, x: f64) -> Option<f64> {
        if self.n_ssf == 0 {
            self.prev2_ssf = x;
            self.last_raw = x;

            self.buf[self.head] = x;
            self.sum += x;
            self.head = if self.period > 1 { 1 } else { 0 };
            self.n_ssf = 1;
            return None;
        }

        if self.n_ssf == 1 {
            self.prev1_ssf = x;
            self.last_raw = x;

            if self.period > 1 {
                self.buf[self.head] = x;
                self.sum += x;
                self.head = (self.head + 1) % self.period;
            } else {
                self.buf[0] = x;
                self.sum = x;
            }
            self.n_ssf = 2;
            return None;
        }

        let cur = (-self.a_sq).mul_add(
            self.prev2_ssf,
            self.b.mul_add(self.prev1_ssf, self.c * (x + self.last_raw)),
        );

        let tp_cur_minus_sum = (self.period as f64).mul_add(cur, -self.sum);
        let my_sum = self.inv_p * tp_cur_minus_sum;

        let will_emit = self.n_ssf + 1 > self.period;

        let out_val = if will_emit {
            let sq = my_sum * my_sum;
            let ms_current = 0.04f64.mul_add(sq, 0.96f64 * self.ms_prev);
            self.ms_prev = ms_current;
            if ms_current > 0.0 {
                my_sum / ms_current.sqrt()
            } else {
                0.0
            }
        } else {
            0.0
        };

        let old = self.buf[self.head];
        self.sum += cur - old;
        self.buf[self.head] = cur;
        self.head = (self.head + 1) % self.period;

        self.prev2_ssf = self.prev1_ssf;
        self.prev1_ssf = cur;
        self.last_raw = x;
        self.n_ssf += 1;

        if will_emit {
            Some(out_val)
        } else {
            None
        }
    }
}

#[inline(always)]
pub fn trendflex_batch_inner_into(
    data: &[f64],
    sweep: &TrendFlexBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<TrendFlexParams>, TrendFlexError> {
    let combos = expand_grid(sweep)?;
    if combos.is_empty() {
        return Err(TrendFlexError::InvalidRange {
            start: sweep.period.0,
            end: sweep.period.1,
            step: sweep.period.2,
        });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(TrendFlexError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(TrendFlexError::TrendFlexPeriodExceedsData {
            period: max_p,
            data_len: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();
    let expected = rows
        .checked_mul(cols)
        .ok_or(TrendFlexError::DimensionsOverflow { rows, cols })?;
    if out.len() != expected {
        return Err(TrendFlexError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }

    let warm: Vec<usize> = combos.iter().map(|c| first + c.period.unwrap()).collect();

    for (row, &warmup) in warm.iter().enumerate() {
        let start = row * cols;
        let end = start + warmup;
        out[start..end].fill(f64::NAN);
    }

    let actual_kern = match kern {
        Kernel::Auto => detect_best_batch_kernel(),
        k => k,
    };

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();

        match actual_kern {
            Kernel::Scalar | Kernel::ScalarBatch => {
                trendflex_row_scalar(data, first, period, out_row)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => trendflex_row_avx2(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                trendflex_row_avx512(data, first, period, out_row)
            }
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                trendflex_row_scalar(data, first, period, out_row)
            }
            Kernel::Auto => unreachable!("Auto kernel should have been resolved"),
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;
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

    Ok(combos)
}

#[derive(Clone, Debug)]
pub struct TrendFlexBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for TrendFlexBatchRange {
    fn default() -> Self {
        Self {
            period: (20, 269, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct TrendFlexBatchBuilder {
    range: TrendFlexBatchRange,
    kernel: Kernel,
}

impl TrendFlexBatchBuilder {
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

    pub fn apply_slice(self, data: &[f64]) -> Result<TrendFlexBatchOutput, TrendFlexError> {
        trendflex_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(
        data: &[f64],
        k: Kernel,
    ) -> Result<TrendFlexBatchOutput, TrendFlexError> {
        TrendFlexBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(
        self,
        c: &Candles,
        src: &str,
    ) -> Result<TrendFlexBatchOutput, TrendFlexError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<TrendFlexBatchOutput, TrendFlexError> {
        TrendFlexBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn trendflex_batch_with_kernel(
    data: &[f64],
    sweep: &TrendFlexBatchRange,
    k: Kernel,
) -> Result<TrendFlexBatchOutput, TrendFlexError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(TrendFlexError::InvalidKernelForBatch(k)),
    };

    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    trendflex_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct TrendFlexBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<TrendFlexParams>,
    pub rows: usize,
    pub cols: usize,
}

impl TrendFlexBatchOutput {
    pub fn row_for_params(&self, p: &TrendFlexParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(20) == p.period.unwrap_or(20))
    }
    pub fn values_for(&self, p: &TrendFlexParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &TrendFlexBatchRange) -> Result<Vec<TrendFlexParams>, TrendFlexError> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, TrendFlexError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        if start < end {
            let v: Vec<usize> = (start..=end).step_by(step).collect();
            if v.is_empty() {
                return Err(TrendFlexError::InvalidRange { start, end, step });
            }
            return Ok(v);
        }

        let mut v = Vec::new();
        let mut cur = start;
        while cur >= end {
            v.push(cur);
            if let Some(next) = cur.checked_sub(step) {
                cur = next;
            } else {
                break;
            }
            if cur == usize::MAX {
                break;
            }
        }
        if v.is_empty() {
            return Err(TrendFlexError::InvalidRange { start, end, step });
        }
        Ok(v)
    }

    let periods = axis_usize(r.period)?;
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(TrendFlexParams { period: Some(p) });
    }
    Ok(out)
}

#[inline(always)]
pub fn expand_grid_trendflex(r: &TrendFlexBatchRange) -> Vec<TrendFlexParams> {
    expand_grid(r).unwrap_or_default()
}

#[inline(always)]
pub fn expand_grid_trendflex_checked(
    r: &TrendFlexBatchRange,
) -> Result<Vec<TrendFlexParams>, TrendFlexError> {
    expand_grid(r)
}

#[inline(always)]
pub fn trendflex_batch_slice(
    data: &[f64],
    sweep: &TrendFlexBatchRange,
    kern: Kernel,
) -> Result<TrendFlexBatchOutput, TrendFlexError> {
    trendflex_batch_inner(data, sweep, kern, false)
}
#[inline(always)]
pub fn trendflex_batch_par_slice(
    data: &[f64],
    sweep: &TrendFlexBatchRange,
    kern: Kernel,
) -> Result<TrendFlexBatchOutput, TrendFlexError> {
    trendflex_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn trendflex_batch_inner(
    data: &[f64],
    sweep: &TrendFlexBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<TrendFlexBatchOutput, TrendFlexError> {
    let combos = expand_grid(sweep)?;
    if combos.is_empty() {
        return Err(TrendFlexError::InvalidRange {
            start: sweep.period.0,
            end: sweep.period.1,
            step: sweep.period.2,
        });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(TrendFlexError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(TrendFlexError::TrendFlexPeriodExceedsData {
            period: max_p,
            data_len: data.len() - first,
        });
    }
    let rows = combos.len();
    let cols = data.len();

    rows.checked_mul(cols)
        .ok_or(TrendFlexError::DimensionsOverflow { rows, cols })?;

    let warm: Vec<usize> = combos.iter().map(|c| first + c.period.unwrap()).collect();
    let mut raw = make_uninit_matrix(rows, cols);

    unsafe {
        init_matrix_prefixes(&mut raw, cols, &warm);
    }

    let actual_kern = match kern {
        Kernel::Auto => detect_best_batch_kernel(),
        k => k,
    };

    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();

        let out_row =
            core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

        match actual_kern {
            Kernel::Scalar | Kernel::ScalarBatch => {
                trendflex_row_scalar(data, first, period, out_row)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => trendflex_row_avx2(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                trendflex_row_avx512(data, first, period, out_row)
            }
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                trendflex_row_scalar(data, first, period, out_row)
            }
            Kernel::Auto => unreachable!("Auto kernel should have been resolved"),
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

    use core::mem::ManuallyDrop;
    let mut guard = ManuallyDrop::new(raw);
    let values: Vec<f64> = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(TrendFlexBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
unsafe fn trendflex_row_scalar(data: &[f64], first: usize, period: usize, out_row: &mut [f64]) {
    let ss_period = ((period as f64) / 2.0).round() as usize;
    let _ = trendflex_scalar_into(data, period, ss_period, first, out_row);
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn trendflex_row_avx2(data: &[f64], first: usize, period: usize, out_row: &mut [f64]) {
    let ss_period = ((period as f64) / 2.0).round() as usize;
    let _ = trendflex_avx2_into(data, period, ss_period, first, out_row);
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn trendflex_row_avx512(data: &[f64], first: usize, period: usize, out_row: &mut [f64]) {
    let ss_period = ((period as f64) / 2.0).round() as usize;
    let _ = trendflex_avx512_into(data, period, ss_period, first, out_row);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn check_trendflex_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = TrendFlexParams { period: None };
        let input = TrendFlexInput::from_candles(&candles, "close", default_params);
        let output = trendflex_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_trendflex_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let params = TrendFlexParams { period: Some(20) };
        let input = TrendFlexInput::from_candles(&candles, "close", params);
        let result = trendflex_with_kernel(&input, kernel)?;
        let expected_last_five = [
            -0.19724678008015128,
            -0.1238001236481444,
            -0.10515389737087717,
            -0.1149541079904878,
            -0.16006869484450567,
        ];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-8,
                "[{}] TrendFlex {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_trendflex_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = TrendFlexInput::with_default_candles(&candles);
        match input.data {
            TrendFlexData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected TrendFlexData::Candles"),
        }
        let output = trendflex_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_trendflex_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = TrendFlexParams { period: Some(0) };
        let input = TrendFlexInput::from_slice(&input_data, params);
        let res = trendflex_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] TrendFlex should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_trendflex_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = TrendFlexParams { period: Some(10) };
        let input = TrendFlexInput::from_slice(&data_small, params);
        let res = trendflex_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] TrendFlex should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_trendflex_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = TrendFlexParams { period: Some(9) };
        let input = TrendFlexInput::from_slice(&single_point, params);
        let res = trendflex_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] TrendFlex should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_trendflex_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let first_params = TrendFlexParams { period: Some(20) };
        let first_input = TrendFlexInput::from_candles(&candles, "close", first_params);
        let first_result = trendflex_with_kernel(&first_input, kernel)?;

        let second_params = TrendFlexParams { period: Some(10) };
        let second_input = TrendFlexInput::from_slice(&first_result.values, second_params);
        let second_result = trendflex_with_kernel(&second_input, kernel)?;

        assert_eq!(second_result.values.len(), first_result.values.len());
        if second_result.values.len() > 240 {
            for (i, &val) in second_result.values[240..].iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "[{}] Found unexpected NaN at out-index {}",
                    test_name,
                    240 + i
                );
            }
        }
        Ok(())
    }

    fn check_trendflex_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input =
            TrendFlexInput::from_candles(&candles, "close", TrendFlexParams { period: Some(20) });
        let res = trendflex_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), candles.close.len());
        if res.values.len() > 240 {
            for (i, &val) in res.values[240..].iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "[{}] Found unexpected NaN at out-index {}",
                    test_name,
                    240 + i
                );
            }
        }
        Ok(())
    }

    fn check_trendflex_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let period = 20;

        let input = TrendFlexInput::from_candles(
            &candles,
            "close",
            TrendFlexParams {
                period: Some(period),
            },
        );
        let batch_output = trendflex_with_kernel(&input, kernel)?.values;

        let mut stream = TrendFlexStream::try_new(TrendFlexParams {
            period: Some(period),
        })?;

        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(tf_val) => stream_values.push(tf_val),
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
                "[{}] TrendFlex streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    macro_rules! generate_all_trendflex_tests {
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
    fn check_trendflex_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let test_periods = vec![5, 10, 20, 30, 50, 80, 100, 150];

        for &period in &test_periods {
            let params = TrendFlexParams {
                period: Some(period),
            };
            let input = TrendFlexInput::from_candles(&candles, "close", params);

            if candles.close.len() < period {
                continue;
            }

            let output = match trendflex_with_kernel(&input, kernel) {
                Ok(o) => o,
                Err(_) => continue,
            };

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
    fn check_trendflex_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_trendflex_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let strat = (1usize..=64).prop_flat_map(|period| {
            (
                prop::collection::vec(
                    (-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                    period..400,
                ),
                Just(period),
            )
        });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(data, period)| {
                let input = TrendFlexInput::from_slice(
                    &data,
                    TrendFlexParams {
                        period: Some(period),
                    },
                );
                let output = trendflex_with_kernel(&input, kernel)?;

                prop_assert_eq!(output.values.len(), data.len(), "Output length mismatch");

                let first = data.iter().position(|x| !x.is_nan()).unwrap_or(0);
                let warmup = first + period;

                for i in 0..warmup.min(data.len()) {
                    prop_assert!(
                        output.values[i].is_nan(),
                        "Expected NaN in warmup period at index {}, got {}",
                        i,
                        output.values[i]
                    );
                }

                for i in warmup..output.values.len() {
                    prop_assert!(
                        output.values[i].is_finite(),
                        "Output at index {} is not finite: {}",
                        i,
                        output.values[i]
                    );
                }

                if data.len() > warmup + 10 {
                    let scale_factor = 10.0;
                    let scaled_data: Vec<f64> = data.iter().map(|&x| x * scale_factor).collect();
                    let scaled_input = TrendFlexInput::from_slice(
                        &scaled_data,
                        TrendFlexParams {
                            period: Some(period),
                        },
                    );
                    let scaled_output = trendflex_with_kernel(&scaled_input, kernel)?;

                    let mut similarity_count = 0;
                    let mut total_compared = 0;
                    for i in warmup..output.values.len() {
                        if output.values[i].is_finite() && scaled_output.values[i].is_finite() {
                            let diff = (output.values[i] - scaled_output.values[i]).abs();

                            if diff < 0.5 {
                                similarity_count += 1;
                            }
                            total_compared += 1;
                        }
                    }

                    if total_compared > 0 {
                        let similarity_ratio = similarity_count as f64 / total_compared as f64;
                        prop_assert!(
							similarity_ratio > 0.9,
							"Scale invariance failed: only {:.1}% of values are similar after scaling",
							similarity_ratio * 100.0
						);
                    }
                }

                if data.len() > warmup + 20 {
                    let mut is_increasing = true;
                    let mut is_decreasing = true;
                    for i in (warmup + 1)..data.len().min(warmup + 50) {
                        if data[i] <= data[i - 1] {
                            is_increasing = false;
                        }
                        if data[i] >= data[i - 1] {
                            is_decreasing = false;
                        }
                    }

                    if is_increasing {
                        let positive_count =
                            output.values[warmup..].iter().filter(|&&v| v > 0.0).count();
                        let total = output.values.len() - warmup;
                        let positive_ratio = positive_count as f64 / total as f64;
                        prop_assert!(
							positive_ratio > 0.7,
							"Increasing trend should produce mostly positive values, got {:.1}% positive",
							positive_ratio * 100.0
						);
                    } else if is_decreasing {
                        let negative_count =
                            output.values[warmup..].iter().filter(|&&v| v < 0.0).count();
                        let total = output.values.len() - warmup;
                        let negative_ratio = negative_count as f64 / total as f64;
                        prop_assert!(
							negative_ratio > 0.7,
							"Decreasing trend should produce mostly negative values, got {:.1}% negative",
							negative_ratio * 100.0
						);
                    }
                }

                let all_same = data[first..]
                    .windows(2)
                    .all(|w| (w[0] - w[1]).abs() < 1e-10);
                if all_same && data.len() > warmup + 10 {
                    let last_values = &output.values[(data.len() - 5)..];
                    for val in last_values {
                        prop_assert!(
                            val.abs() < 0.1,
                            "Constant input should produce values near 0, got {}",
                            val
                        );
                    }
                }

                if period == 1 {
                    for i in (first + 1)..output.values.len() {
                        prop_assert!(
                            output.values[i].is_finite(),
                            "Period=1 should still produce finite values at index {}",
                            i
                        );
                    }
                }

                if data.len() > 5 && period >= data.len().saturating_sub(5) && data.len() > period {
                    let last_idx = data.len() - 1;
                    if last_idx >= warmup {
                        prop_assert!(
                            output.values[last_idx].is_finite(),
                            "Large period should still produce finite values at the end"
                        );
                    }
                }

                if cfg!(all(feature = "nightly-avx", target_arch = "x86_64")) {
                    let scalar_output = trendflex_with_kernel(&input, Kernel::Scalar)?;

                    for i in 0..output.values.len() {
                        if output.values[i].is_finite() && scalar_output.values[i].is_finite() {
                            prop_assert!(
                                (output.values[i] - scalar_output.values[i]).abs() < 1e-9,
                                "Kernel consistency failed at index {}: {} vs {}",
                                i,
                                output.values[i],
                                scalar_output.values[i]
                            );
                        } else {
                            prop_assert_eq!(
                                output.values[i].is_nan(),
                                scalar_output.values[i].is_nan(),
                                "NaN mismatch between kernels at index {}",
                                i
                            );
                        }
                    }
                }

                Ok(())
            })
            .map_err(|e| e.into())
    }

    #[cfg(feature = "proptest")]
    generate_all_trendflex_tests!(check_trendflex_property);

    #[test]
    fn test_trendflex_into_slice_validation() {
        let data = vec![1.0, 2.0, 3.0];
        let params = TrendFlexParams { period: Some(10) };
        let input = TrendFlexInput::from_slice(&data, params);
        let mut out = vec![0.0; data.len()];

        let result = trendflex_into_slice(&mut out, &input, Kernel::Scalar);
        assert!(result.is_err());
        match result {
            Err(TrendFlexError::TrendFlexPeriodExceedsData { period, data_len }) => {
                assert_eq!(period, 10);
                assert_eq!(data_len, 3);
            }
            _ => panic!("Expected TrendFlexPeriodExceedsData error"),
        }

        let empty_data: Vec<f64> = vec![];
        let params = TrendFlexParams { period: Some(5) };
        let input = TrendFlexInput::from_slice(&empty_data, params);
        let mut out = vec![];

        let result = trendflex_into_slice(&mut out, &input, Kernel::Scalar);
        assert!(result.is_err());
        match result {
            Err(TrendFlexError::NoDataProvided) => {}
            _ => panic!("Expected NoDataProvided error"),
        }

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let params = TrendFlexParams { period: Some(0) };
        let input = TrendFlexInput::from_slice(&data, params);
        let mut out = vec![0.0; data.len()];

        let result = trendflex_into_slice(&mut out, &input, Kernel::Scalar);
        assert!(result.is_err());
        match result {
            Err(TrendFlexError::ZeroTrendFlexPeriod { period }) => {
                assert_eq!(period, 0);
            }
            _ => panic!("Expected ZeroTrendFlexPeriod error"),
        }

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let params = TrendFlexParams { period: Some(3) };
        let input = TrendFlexInput::from_slice(&data, params);
        let mut out = vec![0.0; data.len()];

        let result = trendflex_into_slice(&mut out, &input, Kernel::Scalar);
        assert!(result.is_ok());
    }

    #[test]
    #[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
    fn test_trendflex_into_matches_api() -> Result<(), Box<dyn Error>> {
        let n = 512usize;
        let mut data = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f64;
            data.push(0.01 * t + (t * 0.05).sin());
        }

        let input = TrendFlexInput::from_slice(&data, TrendFlexParams::default());
        let baseline = trendflex(&input)?.values;

        let mut out = vec![0.0f64; n];
        trendflex_into(&input, &mut out)?;

        assert_eq!(baseline.len(), out.len());
        for i in 0..n {
            let a = baseline[i];
            let b = out[i];
            let equal = if a.is_nan() && b.is_nan() {
                true
            } else {
                (a - b).abs() <= 1e-12
            };
            assert!(equal, "divergence at {}: {} vs {}", i, a, b);
        }
        Ok(())
    }

    #[test]
    fn test_trendflex_batch_kernel_policy() {
        let data = vec![1.0; 50];
        let sweep = TrendFlexBatchRange { period: (5, 10, 1) };

        let result_scalar = trendflex_batch_with_kernel(&data, &sweep, Kernel::Scalar);
        assert!(matches!(
            result_scalar,
            Err(TrendFlexError::InvalidKernelForBatch(Kernel::Scalar))
        ));

        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        {
            let result_avx2 = trendflex_batch_with_kernel(&data, &sweep, Kernel::Avx2);
            assert!(matches!(
                result_avx2,
                Err(TrendFlexError::InvalidKernelForBatch(Kernel::Avx2))
            ));

            let result_avx512 = trendflex_batch_with_kernel(&data, &sweep, Kernel::Avx512);
            assert!(matches!(
                result_avx512,
                Err(TrendFlexError::InvalidKernelForBatch(Kernel::Avx512))
            ));
        }

        let result_scalar_batch = trendflex_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch);
        assert!(result_scalar_batch.is_ok());
    }

    generate_all_trendflex_tests!(
        check_trendflex_partial_params,
        check_trendflex_accuracy,
        check_trendflex_default_candles,
        check_trendflex_zero_period,
        check_trendflex_period_exceeds_length,
        check_trendflex_very_small_dataset,
        check_trendflex_reinput,
        check_trendflex_nan_handling,
        check_trendflex_streaming,
        check_trendflex_no_poison
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = TrendFlexBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = TrendFlexParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        let expected = [
            -0.19724678008015128,
            -0.1238001236481444,
            -0.10515389737087717,
            -0.1149541079904878,
            -0.16006869484450567,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-8,
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

        let test_configs = vec![
            (5, 20, 3),
            (10, 50, 5),
            (20, 100, 10),
            (30, 120, 15),
            (7, 7, 1),
            (80, 80, 1),
            (15, 45, 5),
        ];

        for (start, end, step) in test_configs {
            let output = TrendFlexBatchBuilder::new()
                .kernel(kernel)
                .period_range(start, end, step)
                .apply_candles(&c, "close")?;

            for (idx, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                let row = idx / output.cols;
                let col = idx % output.cols;
                let period = output
                    .combos
                    .get(row)
                    .map(|p| p.period.unwrap_or(0))
                    .unwrap_or(0);

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (period {}, flat index {})",
                        test, val, bits, row, col, period, idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (period {}, flat index {})",
                        test, val, bits, row, col, period, idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (period {}, flat index {})",
                        test, val, bits, row, col, period, idx
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

#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pyfunction(name = "trendflex")]
#[pyo3(signature = (data, period=None, kernel=None))]

pub fn trendflex_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period: Option<usize>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let params = TrendFlexParams { period };
    let trendflex_in = TrendFlexInput::from_slice(slice_in, params);

    let result_vec: Vec<f64> = py
        .allow_threads(|| trendflex_with_kernel(&trendflex_in, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "TrendFlexStream")]
pub struct TrendFlexStreamPy {
    stream: TrendFlexStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl TrendFlexStreamPy {
    #[new]
    fn new(period: Option<usize>) -> PyResult<Self> {
        let params = TrendFlexParams { period };
        let stream =
            TrendFlexStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(TrendFlexStreamPy { stream })
    }

    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "trendflex_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]

pub fn trendflex_batch_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;

    let sweep = TrendFlexBatchRange {
        period: period_range,
    };

    let combos = expand_grid(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = slice_in.len();
    rows.checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("dimensions overflow"))?;

    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    let combos = py
        .allow_threads(|| -> Result<Vec<TrendFlexParams>, TrendFlexError> {
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

            trendflex_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;

    dict.set_item(
        "periods",
        combos
            .iter()
            .map(|p| p.period.unwrap_or(20) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;

    Ok(dict)
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "trendflex_cuda_batch_dev")]
#[pyo3(signature = (data_f32, period_range, device_id=0))]
pub fn trendflex_cuda_batch_dev_py<'py>(
    py: Python<'py>,
    data_f32: numpy::PyReadonlyArray1<'py, f32>,
    period_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<(TrendFlexDeviceArrayF32Py, Bound<'py, pyo3::types::PyDict>)> {
    use crate::cuda::cuda_available;
    use numpy::IntoPyArray;
    use pyo3::types::PyDict;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let slice_in = data_f32.as_slice()?;
    let sweep = TrendFlexBatchRange {
        period: period_range,
    };

    let (inner, combos, ctx_arc, dev_id) = py.allow_threads(|| {
        let cuda =
            CudaTrendflex::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let (dev, combos) = cuda
            .trendflex_batch_dev(slice_in, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        cuda.synchronize()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok::<_, PyErr>((dev, combos, cuda.context_arc_clone(), cuda.device_id()))
    })?;

    let dict = PyDict::new(py);
    let periods: Vec<u64> = combos.iter().map(|c| c.period.unwrap() as u64).collect();
    dict.set_item("periods", periods.into_pyarray(py))?;

    Ok((
        TrendFlexDeviceArrayF32Py {
            inner,
            _ctx: ctx_arc,
            device_id: dev_id,
        },
        dict,
    ))
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "trendflex_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm_f32, period, device_id=0))]
pub fn trendflex_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    data_tm_f32: numpy::PyReadonlyArray2<'_, f32>,
    period: usize,
    device_id: usize,
) -> PyResult<TrendFlexDeviceArrayF32Py> {
    use crate::cuda::cuda_available;
    use numpy::PyUntypedArrayMethods;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let flat_in = data_tm_f32.as_slice()?;
    let rows = data_tm_f32.shape()[0];
    let cols = data_tm_f32.shape()[1];
    let params = TrendFlexParams {
        period: Some(period),
    };

    let (inner, ctx_arc, dev_id) = py.allow_threads(|| {
        let cuda =
            CudaTrendflex::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let dev = cuda
            .trendflex_multi_series_one_param_time_major_dev(flat_in, cols, rows, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.synchronize()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok::<_, PyErr>((dev, cuda.context_arc_clone(), cuda.device_id()))
    })?;

    Ok(TrendFlexDeviceArrayF32Py {
        inner,
        _ctx: ctx_arc,
        device_id: dev_id,
    })
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use serde::{Deserialize, Serialize};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen::prelude::*;

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct TrendFlexBatchConfig {
    pub period_range: (usize, usize, usize),
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct TrendFlexBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<TrendFlexParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]

pub fn trendflex_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
    let params = TrendFlexParams {
        period: Some(period),
    };
    let input = TrendFlexInput::from_slice(data, params);

    trendflex_with_kernel(&input, Kernel::Auto)
        .map(|o| o.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]

pub fn trendflex_batch_js(
    data: &[f64],
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = TrendFlexBatchRange {
        period: (period_start, period_end, period_step),
    };

    trendflex_batch_inner(data, &sweep, Kernel::Auto, false)
        .map(|output| output.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]

pub fn trendflex_batch_metadata_js(
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = TrendFlexBatchRange {
        period: (period_start, period_end, period_step),
    };

    let combos = expand_grid(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let metadata: Vec<f64> = combos
        .iter()
        .map(|combo| combo.period.unwrap_or(20) as f64)
        .collect();

    Ok(metadata)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = trendflex_batch)]
pub fn trendflex_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: TrendFlexBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = TrendFlexBatchRange {
        period: config.period_range,
    };

    let output = trendflex_batch_inner(data, &sweep, Kernel::Auto, false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = TrendFlexBatchJsOutput {
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
pub fn trendflex_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn trendflex_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn trendflex_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to trendflex_into"));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        if period == 0 || period >= len {
            return Err(JsValue::from_str("Invalid period"));
        }
        let input = TrendFlexInput::from_slice(
            data,
            TrendFlexParams {
                period: Some(period),
            },
        );
        if in_ptr == out_ptr {
            let mut tmp = vec![0.0; len];
            trendflex_into_slice(&mut tmp, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            std::slice::from_raw_parts_mut(out_ptr, len).copy_from_slice(&tmp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            trendflex_into_slice(out, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        Ok(())
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn trendflex_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to trendflex_batch_into",
        ));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);

        let sweep = TrendFlexBatchRange {
            period: (period_start, period_end, period_step),
        };

        let combos = expand_grid(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let n_combos = combos.len();
        let total_size = n_combos
            .checked_mul(len)
            .ok_or_else(|| JsValue::from_str("dimensions overflow"))?;

        let out_slice = std::slice::from_raw_parts_mut(out_ptr, total_size);

        trendflex_batch_inner_into(data, &sweep, Kernel::Auto, false, out_slice)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(n_combos)
    }
}
