#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::CudaAd;
use crate::utilities::data_loader::Candles;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::utilities::dlpack_cuda::export_f32_cuda_dlpack_2d;
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, make_uninit_matrix,
};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(all(feature = "python", feature = "cuda"))]
use numpy::PyReadonlyArray1;
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
    #[error("ad: candle field error: {0}")]
    CandleFieldError(String),
    #[error(
        "ad: Data length mismatch: high={high_len}, low={low_len}, close={close_len}, volume={volume_len}"
    )]
    DataLengthMismatch {
        high_len: usize,
        low_len: usize,
        close_len: usize,
        volume_len: usize,
    },
    #[error("ad: invalid period: period={period}, data_len={data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("ad: output length mismatch: expected={expected}, got={got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("ad: not enough valid data: needed={needed}, valid={valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("ad: empty input data")]
    EmptyInputData,
    #[error("ad: all values are NaN")]
    AllValuesNaN,
    #[error("ad: invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: isize,
        end: isize,
        step: isize,
    },
    #[error("ad: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("ad: invalid input: {0}")]
    InvalidInput(String),
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
    if size == 0 {
        return Err(AdError::EmptyInputData);
    }

    let mut chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };

    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
    if matches!(kernel, Kernel::Auto) && matches!(chosen, Kernel::Avx512 | Kernel::Avx512Batch) {
        chosen = Kernel::Avx2;
    }

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

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn ad_into(input: &AdInput, out: &mut [f64]) -> Result<(), AdError> {
    ad_into_slice(out, input, Kernel::Auto)
}

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

    if high.is_empty() {
        return Err(AdError::EmptyInputData);
    }

    if high.len() != low.len() || high.len() != close.len() || high.len() != volume.len() {
        return Err(AdError::DataLengthMismatch {
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
            volume_len: volume.len(),
        });
    }

    if dst.len() != high.len() {
        return Err(AdError::OutputLengthMismatch {
            expected: high.len(),
            got: dst.len(),
        });
    }

    match kern {
        Kernel::Auto => {
            let mut k = detect_best_kernel();
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            if matches!(k, Kernel::Avx512) {
                k = Kernel::Avx2;
            }
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

    Ok(())
}

#[inline]
pub fn ad_scalar(high: &[f64], low: &[f64], close: &[f64], volume: &[f64], out: &mut [f64]) {
    debug_assert_eq!(high.len(), low.len());
    debug_assert_eq!(high.len(), close.len());
    debug_assert_eq!(high.len(), volume.len());
    debug_assert_eq!(high.len(), out.len());

    let mut sum = 0.0f64;
    for ((((&h, &l), &c), &v), o) in high
        .iter()
        .zip(low)
        .zip(close)
        .zip(volume)
        .zip(out.iter_mut())
    {
        let hl = h - l;
        if hl != 0.0 {
            let num = (c - l) - (h - c);
            sum += (num / hl) * v;
        }
        *o = sum;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn ad_avx2(high: &[f64], low: &[f64], close: &[f64], volume: &[f64], out: &mut [f64]) {
    unsafe { ad_avx2_inner(high, low, close, volume, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn ad_avx2_inner(high: &[f64], low: &[f64], close: &[f64], volume: &[f64], out: &mut [f64]) {
    use core::arch::x86_64::*;

    let n = high.len();
    let h = high.as_ptr();
    let l = low.as_ptr();
    let c = close.as_ptr();
    let v = volume.as_ptr();
    let o = out.as_mut_ptr();

    let mut base = 0.0f64;
    let mut i = 0usize;

    while i + 4 <= n {
        let hv = _mm256_loadu_pd(h.add(i));
        let lv = _mm256_loadu_pd(l.add(i));
        let cv = _mm256_loadu_pd(c.add(i));
        let vv = _mm256_loadu_pd(v.add(i));

        let hl = _mm256_sub_pd(hv, lv);
        let num = _mm256_sub_pd(_mm256_sub_pd(cv, lv), _mm256_sub_pd(hv, cv));
        let mfm = _mm256_div_pd(num, hl);
        let mfv_unmasked = _mm256_mul_pd(mfm, vv);

        let z = _mm256_set1_pd(0.0);
        let mask = _mm256_cmp_pd(hl, z, _CMP_NEQ_OQ);
        let mfv = _mm256_and_pd(mfv_unmasked, mask);

        let mut tmp: [f64; 4] = core::mem::zeroed();
        _mm256_storeu_pd(tmp.as_mut_ptr(), mfv);
        *o.add(i + 0) = {
            base += tmp[0];
            base
        };
        *o.add(i + 1) = {
            base += tmp[1];
            base
        };
        *o.add(i + 2) = {
            base += tmp[2];
            base
        };
        *o.add(i + 3) = {
            base += tmp[3];
            base
        };

        i += 4;
    }

    while i < n {
        let hi = *h.add(i);
        let lo = *l.add(i);
        let cl = *c.add(i);
        let vo = *v.add(i);
        let hl = hi - lo;
        if hl != 0.0 {
            let num = (cl - lo) - (hi - cl);
            base += (num / hl) * vo;
        }
        *o.add(i) = base;
        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn ad_avx512(high: &[f64], low: &[f64], close: &[f64], volume: &[f64], out: &mut [f64]) {
    unsafe { ad_avx512_inner(high, low, close, volume, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn ad_avx512_inner(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    out: &mut [f64],
) {
    use core::arch::x86_64::*;

    let n = high.len();
    let h = high.as_ptr();
    let l = low.as_ptr();
    let c = close.as_ptr();
    let v = volume.as_ptr();
    let o = out.as_mut_ptr();

    let mut base = 0.0f64;
    let mut i = 0usize;

    while i + 8 <= n {
        let hv = _mm512_loadu_pd(h.add(i));
        let lv = _mm512_loadu_pd(l.add(i));
        let cv = _mm512_loadu_pd(c.add(i));
        let vv = _mm512_loadu_pd(v.add(i));

        let hl = _mm512_sub_pd(hv, lv);
        let num = _mm512_sub_pd(_mm512_sub_pd(cv, lv), _mm512_sub_pd(hv, cv));
        let mfm = _mm512_div_pd(num, hl);
        let mfv_unmasked = _mm512_mul_pd(mfm, vv);

        let mask = _mm512_cmpneq_pd_mask(hl, _mm512_set1_pd(0.0));
        let mfv = _mm512_maskz_mov_pd(mask, mfv_unmasked);

        let mut tmp = core::mem::MaybeUninit::<[f64; 8]>::uninit();
        _mm512_storeu_pd(tmp.as_mut_ptr() as *mut f64, mfv);
        let vals = tmp.assume_init();

        *o.add(i + 0) = {
            base += vals[0];
            base
        };
        *o.add(i + 1) = {
            base += vals[1];
            base
        };
        *o.add(i + 2) = {
            base += vals[2];
            base
        };
        *o.add(i + 3) = {
            base += vals[3];
            base
        };
        *o.add(i + 4) = {
            base += vals[4];
            base
        };
        *o.add(i + 5) = {
            base += vals[5];
            base
        };
        *o.add(i + 6) = {
            base += vals[6];
            base
        };
        *o.add(i + 7) = {
            base += vals[7];
            base
        };

        i += 8;
    }

    while i < n {
        let hi = *h.add(i);
        let lo = *l.add(i);
        let cl = *c.add(i);
        let vo = *v.add(i);
        let hl = hi - lo;
        if hl != 0.0 {
            let num = (cl - lo) - (hi - cl);
            base += (num / hl) * vo;
        }
        *o.add(i) = base;
        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn ad_avx512_short(high: &[f64], low: &[f64], close: &[f64], volume: &[f64], out: &mut [f64]) {
    ad_avx512(high, low, close, volume, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn ad_avx512_long(high: &[f64], low: &[f64], close: &[f64], volume: &[f64], out: &mut [f64]) {
    ad_avx512(high, low, close, volume, out)
}

#[inline]
pub fn ad_batch_with_kernel(data: &AdBatchInput, k: Kernel) -> Result<AdBatchOutput, AdError> {
    let mut kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(AdError::InvalidKernelForBatch(other)),
    };
    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
    if matches!(k, Kernel::Auto) && matches!(kernel, Kernel::Avx512Batch) {
        kernel = Kernel::Avx2Batch;
    }

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
    let len = rows
        .checked_mul(cols)
        .ok_or_else(|| AdError::InvalidInput("rows*cols overflow".into()))?;

    let mut buf_mu = make_uninit_matrix(rows, cols);
    let values = unsafe {
        let ptr = buf_mu.as_mut_ptr() as *mut f64;
        let slice = std::slice::from_raw_parts_mut(ptr, len);

        ad_batch_inner_into(data, kern, parallel, slice)?;

        Vec::from_raw_parts(ptr, len, len)
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

    if data.lows.len() != rows || data.closes.len() != rows || data.volumes.len() != rows {
        return Err(AdError::DataLengthMismatch {
            high_len: data.highs.len(),
            low_len: data.lows.len(),
            close_len: data.closes.len(),
            volume_len: data.volumes.len(),
        });
    }

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

    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| AdError::InvalidInput("rows*cols overflow".into()))?;
    if out.len() != expected {
        return Err(AdError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }

    let mut actual = match kern {
        Kernel::Auto => detect_best_batch_kernel(),
        k => k,
    };
    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
    if matches!(kern, Kernel::Auto) && matches!(actual, Kernel::Avx512Batch) {
        actual = Kernel::Avx2Batch;
    }

    let do_row = |row: usize, dst: &mut [f64]| unsafe {
        match actual {
            Kernel::Scalar | Kernel::ScalarBatch => ad_row_scalar(
                data.highs[row],
                data.lows[row],
                data.closes[row],
                data.volumes[row],
                dst,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => ad_row_avx2(
                data.highs[row],
                data.lows[row],
                data.closes[row],
                data.volumes[row],
                dst,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => ad_row_avx512(
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
    ad_avx2(high, low, close, volume, out)
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
    ad_avx512(high, low, close, volume, out)
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
    ad_avx512(high, low, close, volume, out)
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
    ad_avx512(high, low, close, volume, out)
}

#[derive(Debug, Clone)]
pub struct AdStream {
    sum: f64,
}

impl AdStream {
    #[inline(always)]
    pub fn try_new() -> Result<Self, AdError> {
        Ok(Self { sum: 0.0 })
    }

    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64, close: f64, volume: f64) -> f64 {
        if volume == 0.0 {
            return self.sum;
        }

        let hl = high - low;
        if hl != 0.0 {
            let num = (close - low) - (high - close);

            self.sum += (num / hl) * volume;
        }
        self.sum
    }
}

#[cfg(all(feature = "python", feature = "cuda"))]
use cust::context::Context;
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::memory::DeviceBuffer;
#[cfg(all(feature = "python", feature = "cuda"))]
use std::sync::Arc;
#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(module = "ta_indicators.cuda", unsendable)]
pub struct AdDeviceArrayF32Py {
    pub(crate) buf: Option<DeviceBuffer<f32>>,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) _ctx: Arc<Context>,
    pub(crate) device_id: u32,
}
#[cfg(all(feature = "python", feature = "cuda"))]
#[pymethods]
impl AdDeviceArrayF32Py {
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
#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "ad_cuda_dev")]
#[pyo3(signature = (high_f32, low_f32, close_f32, volume_f32, device_id=0))]
pub fn ad_cuda_dev_py(
    py: Python<'_>,
    high_f32: PyReadonlyArray1<'_, f32>,
    low_f32: PyReadonlyArray1<'_, f32>,
    close_f32: PyReadonlyArray1<'_, f32>,
    volume_f32: PyReadonlyArray1<'_, f32>,
    device_id: usize,
) -> PyResult<AdDeviceArrayF32Py> {
    use crate::cuda::cuda_available;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let high = high_f32.as_slice()?;
    let low = low_f32.as_slice()?;
    let close = close_f32.as_slice()?;
    let volume = volume_f32.as_slice()?;

    let (buf, rows, cols, ctx, dev_id) = py.allow_threads(|| {
        let cuda = CudaAd::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let out = cuda
            .ad_series_dev(high, low, close, volume)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx = cuda.context_arc();
        Ok::<_, pyo3::PyErr>((out.buf, out.rows, out.cols, ctx, cuda.device_id()))
    })?;

    Ok(AdDeviceArrayF32Py {
        buf: Some(buf),
        rows,
        cols,
        _ctx: ctx,
        device_id: dev_id,
    })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "ad_cuda_many_series_one_param_dev")]
#[pyo3(signature = (high_tm_f32, low_tm_f32, close_tm_f32, volume_tm_f32, cols, rows, device_id=0))]
pub fn ad_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    high_tm_f32: PyReadonlyArray1<'_, f32>,
    low_tm_f32: PyReadonlyArray1<'_, f32>,
    close_tm_f32: PyReadonlyArray1<'_, f32>,
    volume_tm_f32: PyReadonlyArray1<'_, f32>,
    cols: usize,
    rows: usize,
    device_id: usize,
) -> PyResult<AdDeviceArrayF32Py> {
    use crate::cuda::cuda_available;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let high_tm = high_tm_f32.as_slice()?;
    let low_tm = low_tm_f32.as_slice()?;
    let close_tm = close_tm_f32.as_slice()?;
    let volume_tm = volume_tm_f32.as_slice()?;

    let (buf, r_out, c_out, ctx, dev_id) = py.allow_threads(|| {
        let cuda = CudaAd::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let out = cuda
            .ad_many_series_one_param_time_major_dev(
                high_tm, low_tm, close_tm, volume_tm, cols, rows,
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx = cuda.context_arc();
        Ok::<_, pyo3::PyErr>((out.buf, out.rows, out.cols, ctx, cuda.device_id()))
    })?;

    Ok(AdDeviceArrayF32Py {
        buf: Some(buf),
        rows: r_out,
        cols: c_out,
        _ctx: ctx,
        device_id: dev_id,
    })
}

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
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen::prelude::*;

#[cfg(feature = "python")]
#[pyfunction(name = "ad")]
#[pyo3(signature = (high, low, close, volume, kernel=None))]

pub fn ad_py<'py>(
    py: Python<'py>,
    high: numpy::PyReadonlyArray1<'py, f64>,
    low: numpy::PyReadonlyArray1<'py, f64>,
    close: numpy::PyReadonlyArray1<'py, f64>,
    volume: numpy::PyReadonlyArray1<'py, f64>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;
    let volume_slice = volume.as_slice()?;

    if high_slice.is_empty()
        || low_slice.is_empty()
        || close_slice.is_empty()
        || volume_slice.is_empty()
    {
        return Err(PyValueError::new_err("Not enough data"));
    }

    let kern = crate::utilities::kernel_validation::validate_kernel(kernel, false)?;

    let input = AdInput::from_slices(
        high_slice,
        low_slice,
        close_slice,
        volume_slice,
        AdParams::default(),
    );

    let result_vec: Vec<f64> = py
        .allow_threads(|| ad_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

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

    fn update(&mut self, high: f64, low: f64, close: f64, volume: f64) -> f64 {
        self.stream.update(high, low, close, volume)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "ad_batch")]
#[pyo3(signature = (highs, lows, closes, volumes, kernel=None))]

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

    let mut high_arrays: Vec<PyReadonlyArray1<f64>> = Vec::with_capacity(rows);
    let mut low_arrays: Vec<PyReadonlyArray1<f64>> = Vec::with_capacity(rows);
    let mut close_arrays: Vec<PyReadonlyArray1<f64>> = Vec::with_capacity(rows);
    let mut volume_arrays: Vec<PyReadonlyArray1<f64>> = Vec::with_capacity(rows);

    for i in 0..rows {
        let h = highs.get_item(i)?.extract::<PyReadonlyArray1<f64>>()?;
        let l = lows.get_item(i)?.extract::<PyReadonlyArray1<f64>>()?;
        let c = closes.get_item(i)?.extract::<PyReadonlyArray1<f64>>()?;
        let v = volumes.get_item(i)?.extract::<PyReadonlyArray1<f64>>()?;

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

    let high_slices: Vec<&[f64]> = high_arrays.iter().map(|a| a.as_slice().unwrap()).collect();
    let low_slices: Vec<&[f64]> = low_arrays.iter().map(|a| a.as_slice().unwrap()).collect();
    let close_slices: Vec<&[f64]> = close_arrays.iter().map(|a| a.as_slice().unwrap()).collect();
    let volume_slices: Vec<&[f64]> = volume_arrays
        .iter()
        .map(|a| a.as_slice().unwrap())
        .collect();

    let cols = if rows > 0 { high_slices[0].len() } else { 0 };
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow in ad_batch"))?;
    let out_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
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
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ad_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
) -> Result<Vec<f64>, JsValue> {
    if high.is_empty() || low.is_empty() || close.is_empty() || volume.is_empty() {
        return Err(JsValue::from_str("Not enough data"));
    }

    let input = AdInput::from_slices(high, low, close, volume, AdParams::default());

    let mut output = vec![0.0; high.len()];
    ad_into_slice(&mut output, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
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
    let check = rows
        .checked_mul(cols)
        .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;
    if highs_flat.len() != check
        || lows_flat.len() != check
        || closes_flat.len() != check
        || volumes_flat.len() != check
    {
        return Err(JsValue::from_str(
            "Input arrays must have rows*cols elements",
        ));
    }

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

    ad_batch_with_kernel(&batch_input, Kernel::ScalarBatch)
        .map(|o| o.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ad_batch_metadata_js(rows: usize, cols: usize) -> Vec<f64> {
    vec![rows as f64, cols as f64]
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ad_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ad_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
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

        if high_ptr as *const f64 == out_ptr
            || low_ptr as *const f64 == out_ptr
            || close_ptr as *const f64 == out_ptr
            || volume_ptr as *const f64 == out_ptr
        {
            let mut temp = vec![0.0; len];
            ad_into_slice(&mut temp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            ad_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use serde::{Deserialize, Serialize};

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct AdBatchJsOutput {
    pub values: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
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
    let check = rows
        .checked_mul(cols)
        .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;
    if lows_flat.len() != check || closes_flat.len() != check || volumes_flat.len() != check {
        return Err(JsValue::from_str(
            "Input arrays must have rows*cols elements",
        ));
    }

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

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
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

        ad_batch_inner_into(&batch, detect_best_batch_kernel(), false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::{read_candles_from_csv, Candles};
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

    #[cfg(debug_assertions)]
    fn check_ad_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = AdInput::with_default_candles(&candles);
        let output = ad_with_kernel(&input, kernel)?;

        for (i, &val) in output.values.iter().enumerate() {
            if val.is_nan() {
                continue;
            }

            let bits = val.to_bits();

            if bits == 0x11111111_11111111 {
                panic!(
                    "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {}",
                    test_name, val, bits, i
                );
            }

            if bits == 0x22222222_22222222 {
                panic!(
                    "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {}",
                    test_name, val, bits, i
                );
            }

            if bits == 0x33333333_33333333 {
                panic!(
                    "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {}",
                    test_name, val, bits, i
                );
            }
        }

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

        let strat = (10usize..400).prop_flat_map(|len| {
            prop::collection::vec(
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

                let AdOutput { values: out } = ad_with_kernel(&input, kernel).unwrap();

                let AdOutput { values: ref_out } = ad_with_kernel(&input, Kernel::Scalar).unwrap();

                prop_assert_eq!(out.len(), highs.len(), "Output length mismatch");

                for (i, &val) in out.iter().enumerate() {
                    prop_assert!(
                        !val.is_nan(),
                        "Unexpected NaN at index {}: AD should not have NaN values",
                        i
                    );
                }

                for i in 0..out.len() {
                    let y = out[i];
                    let r = ref_out[i];

                    let y_bits = y.to_bits();
                    let r_bits = r.to_bits();

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

                for i in 1..volumes.len() {
                    if volumes[i] == 0.0 {
                        prop_assert!(
                            (out[i] - out[i - 1]).abs() < 1e-10,
                            "AD should not change when volume is 0 at index {}",
                            i
                        );
                    }
                }

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

        let highs: Vec<&[f64]> = vec![&candles.high];
        let lows: Vec<&[f64]> = vec![&candles.low];
        let closes: Vec<&[f64]> = vec![&candles.close];
        let volumes: Vec<&[f64]> = vec![&candles.volume];

        let single = ad_with_kernel(
            &AdInput::from_candles(&candles, AdParams::default()),
            kernel,
        )?
        .values;

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

        let highs: Vec<&[f64]> = vec![&candles.high, &candles.high, &candles.high];
        let lows: Vec<&[f64]> = vec![&candles.low, &candles.low, &candles.low];
        let closes: Vec<&[f64]> = vec![&candles.close, &candles.close, &candles.close];
        let volumes: Vec<&[f64]> = vec![&candles.volume, &candles.volume, &candles.volume];

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

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let mut highs: Vec<&[f64]> = vec![];
        let mut lows: Vec<&[f64]> = vec![];
        let mut closes: Vec<&[f64]> = vec![];
        let mut volumes: Vec<&[f64]> = vec![];

        highs.push(&c.high);
        lows.push(&c.low);
        closes.push(&c.close);
        volumes.push(&c.volume);

        let high_rev: Vec<f64> = c.high.iter().rev().copied().collect();
        let low_rev: Vec<f64> = c.low.iter().rev().copied().collect();
        let close_rev: Vec<f64> = c.close.iter().rev().copied().collect();
        let volume_rev: Vec<f64> = c.volume.iter().rev().copied().collect();

        highs.push(&high_rev);
        lows.push(&low_rev);
        closes.push(&close_rev);
        volumes.push(&volume_rev);

        if c.high.len() > 100 {
            highs.push(&c.high[50..]);
            lows.push(&c.low[50..]);
            closes.push(&c.close[50..]);
            volumes.push(&c.volume[50..]);
        }

        let batch = AdBatchBuilder::new()
            .kernel(kernel)
            .apply_slices(&highs, &lows, &closes, &volumes)?;

        for (idx, &val) in batch.values.iter().enumerate() {
            if val.is_nan() {
                continue;
            }

            let bits = val.to_bits();
            let row = idx / batch.cols;
            let col = idx % batch.cols;

            if bits == 0x11111111_11111111 {
                panic!(
					"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (flat index {})",
					test, val, bits, row, col, idx
				);
            }

            if bits == 0x22222222_22222222 {
                panic!(
					"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (flat index {})",
					test, val, bits, row, col, idx
				);
            }

            if bits == 0x33333333_33333333 {
                panic!(
					"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (flat index {})",
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

    gen_batch_tests!(check_batch_single_row);
    gen_batch_tests!(check_batch_multi_row);
    gen_batch_tests!(check_batch_no_poison);

    #[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
    #[test]
    fn test_ad_into_matches_api() {
        let n = 256usize;
        let mut ts = Vec::with_capacity(n);
        let mut open = Vec::with_capacity(n);
        let mut high = Vec::with_capacity(n);
        let mut low = Vec::with_capacity(n);
        let mut close = Vec::with_capacity(n);
        let mut volume = Vec::with_capacity(n);

        for i in 0..n {
            let i_f = i as f64;
            ts.push(i as i64);
            let o = 100.0 + (i % 13) as f64 * 0.75;
            let l = o - 2.0;
            let h = o + 2.0 + ((i % 3) as f64) * 0.1;
            let c = l + ((i % 5) as f64) * 0.5;
            let v = 1000.0 + 10.0 * i_f;
            open.push(o);
            low.push(l);
            high.push(h);
            close.push(c);
            volume.push(v);
        }

        let candles = Candles::new(
            ts,
            open,
            high.clone(),
            low.clone(),
            close.clone(),
            volume.clone(),
        );
        let input = AdInput::with_default_candles(&candles);

        let baseline = ad(&input).expect("ad() should succeed").values;

        let mut out = vec![0.0; baseline.len()];
        ad_into(&input, &mut out).expect("ad_into() should succeed");

        assert_eq!(out.len(), baseline.len());

        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b) || ((a - b).abs() <= 1e-12)
        }

        for (i, (a, b)) in out
            .iter()
            .copied()
            .zip(baseline.iter().copied())
            .enumerate()
        {
            assert!(
                eq_or_both_nan(a, b),
                "ad_into parity failed at index {}: {} vs {}",
                i,
                a,
                b
            );
        }
    }
}
