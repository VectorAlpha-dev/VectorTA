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
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use serde::{Deserialize, Serialize};
use std::mem::{ManuallyDrop, MaybeUninit};
use thiserror::Error;
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen::prelude::*;

use crate::indicators::sma::{sma, SmaData, SmaError, SmaInput, SmaParams};

#[derive(Debug, Clone)]
pub enum VpciData<'a> {
    Candles {
        candles: &'a Candles,
        close_source: &'a str,
        volume_source: &'a str,
    },
    Slices {
        close: &'a [f64],
        volume: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct VpciOutput {
    pub vpci: Vec<f64>,
    pub vpcis: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct VpciParams {
    pub short_range: Option<usize>,
    pub long_range: Option<usize>,
}

impl Default for VpciParams {
    fn default() -> Self {
        Self {
            short_range: Some(5),
            long_range: Some(25),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VpciInput<'a> {
    pub data: VpciData<'a>,
    pub params: VpciParams,
}

impl<'a> VpciInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        close_source: &'a str,
        volume_source: &'a str,
        params: VpciParams,
    ) -> Self {
        Self {
            data: VpciData::Candles {
                candles,
                close_source,
                volume_source,
            },
            params,
        }
    }

    #[inline]
    pub fn from_slices(close: &'a [f64], volume: &'a [f64], params: VpciParams) -> Self {
        Self {
            data: VpciData::Slices { close, volume },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: VpciData::Candles {
                candles,
                close_source: "close",
                volume_source: "volume",
            },
            params: VpciParams::default(),
        }
    }

    #[inline]
    pub fn get_short_range(&self) -> usize {
        self.params.short_range.unwrap_or(5)
    }
    #[inline]
    pub fn get_long_range(&self) -> usize {
        self.params.long_range.unwrap_or(25)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct VpciBuilder {
    short_range: Option<usize>,
    long_range: Option<usize>,
    kernel: Kernel,
}

impl Default for VpciBuilder {
    fn default() -> Self {
        Self {
            short_range: None,
            long_range: None,
            kernel: Kernel::Auto,
        }
    }
}

impl VpciBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn short_range(mut self, n: usize) -> Self {
        self.short_range = Some(n);
        self
    }
    #[inline(always)]
    pub fn long_range(mut self, n: usize) -> Self {
        self.long_range = Some(n);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<VpciOutput, VpciError> {
        let p = VpciParams {
            short_range: self.short_range,
            long_range: self.long_range,
        };
        let i = VpciInput::from_candles(c, "close", "volume", p);
        vpci_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slices(self, close: &[f64], volume: &[f64]) -> Result<VpciOutput, VpciError> {
        let p = VpciParams {
            short_range: self.short_range,
            long_range: self.long_range,
        };
        let i = VpciInput::from_slices(close, volume, p);
        vpci_with_kernel(&i, self.kernel)
    }
}

#[derive(Clone, Debug)]
pub struct VpciStream {
    short_range: usize,
    long_range: usize,

    close_buf: Vec<f64>,
    volume_buf: Vec<f64>,
    head: usize,
    count: usize,

    sum_c_long: f64,
    sum_v_long: f64,
    sum_cv_long: f64,

    sum_c_short: f64,
    sum_v_short: f64,
    sum_cv_short: f64,

    vpci_vol_buf: Vec<f64>,
    vpci_vol_head: usize,
    sum_vpci_vol_short: f64,

    inv_long: f64,
    inv_short: f64,
}

impl VpciStream {
    pub fn try_new(params: VpciParams) -> Result<Self, VpciError> {
        let short_range = params.short_range.unwrap_or(5);
        let long_range = params.long_range.unwrap_or(25);

        if short_range == 0 || long_range == 0 {
            return Err(VpciError::InvalidPeriod {
                period: 0,
                data_len: 0,
            });
        }
        if short_range > long_range {
            return Err(VpciError::InvalidPeriod {
                period: short_range,
                data_len: long_range,
            });
        }

        Ok(Self {
            short_range,
            long_range,
            close_buf: vec![0.0; long_range],
            volume_buf: vec![0.0; long_range],
            head: 0,
            count: 0,

            sum_c_long: 0.0,
            sum_v_long: 0.0,
            sum_cv_long: 0.0,

            sum_c_short: 0.0,
            sum_v_short: 0.0,
            sum_cv_short: 0.0,

            vpci_vol_buf: vec![0.0; short_range],
            vpci_vol_head: 0,
            sum_vpci_vol_short: 0.0,

            inv_long: 1.0 / (long_range as f64),
            inv_short: 1.0 / (short_range as f64),
        })
    }

    #[inline(always)]
    fn zf(x: f64) -> f64 {
        if x.is_finite() {
            x
        } else {
            0.0
        }
    }

    #[inline(always)]
    pub fn update(&mut self, close: f64, volume: f64) -> Option<(f64, f64)> {
        let c_new = Self::zf(close);
        let v_new = Self::zf(volume);
        let cv_new = c_new * v_new;

        let i = self.head;
        let j = (self.head + self.long_range - self.short_range) % self.long_range;

        let c_old_L = Self::zf(self.close_buf[i]);
        let v_old_L = Self::zf(self.volume_buf[i]);
        let cv_old_L = c_old_L * v_old_L;

        let c_old_S = Self::zf(self.close_buf[j]);
        let v_old_S = Self::zf(self.volume_buf[j]);
        let cv_old_S = c_old_S * v_old_S;

        self.close_buf[i] = close;
        self.volume_buf[i] = volume;

        self.head = (self.head + 1) % self.long_range;
        self.count = self.count.saturating_add(1);

        self.sum_c_long += c_new - c_old_L;
        self.sum_v_long += v_new - v_old_L;
        self.sum_cv_long += cv_new - cv_old_L;

        self.sum_c_short += c_new - c_old_S;
        self.sum_v_short += v_new - v_old_S;
        self.sum_cv_short += cv_new - cv_old_S;

        if self.count < self.long_range {
            return None;
        }

        let sv_l = self.sum_v_long;
        let sc_l = self.sum_c_long;
        let scv_l = self.sum_cv_long;
        let sma_l = sc_l * self.inv_long;
        let vwma_l = if sv_l != 0.0 { scv_l / sv_l } else { f64::NAN };
        let vpc = vwma_l - sma_l;

        let sv_s = self.sum_v_short;
        let sc_s = self.sum_c_short;
        let scv_s = self.sum_cv_short;

        let vpr = if sv_s != 0.0 && sc_s != 0.0 {
            (scv_s * (self.short_range as f64)) / (sv_s * sc_s)
        } else {
            f64::NAN
        };

        let vm = if sv_l != 0.0 {
            (sv_s * (self.long_range as f64)) / (sv_l * (self.short_range as f64))
        } else {
            f64::NAN
        };

        let vpci = vpc * vpr * vm;

        let vpci_vol_new = if vpci.is_finite() { vpci * v_new } else { 0.0 };
        let vpci_vol_old = self.vpci_vol_buf[self.vpci_vol_head];
        self.sum_vpci_vol_short += vpci_vol_new - vpci_vol_old;
        self.vpci_vol_buf[self.vpci_vol_head] = vpci_vol_new;
        self.vpci_vol_head = (self.vpci_vol_head + 1) % self.short_range;

        let denom = sv_s * self.inv_short;
        let vpcis = if denom != 0.0 && denom.is_finite() {
            (self.sum_vpci_vol_short * self.inv_short) / denom
        } else {
            f64::NAN
        };

        Some((vpci, vpcis))
    }
}

#[derive(Debug, Error)]
pub enum VpciError {
    #[error("vpci: Empty input data (All close or volume values are NaN).")]
    EmptyInputData,

    #[error("vpci: All close or volume values are NaN.")]
    AllValuesNaN,

    #[error("vpci: Invalid range (Invalid period): period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },

    #[error("vpci: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },

    #[error("vpci: output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },

    #[error("vpci: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),

    #[error("vpci: invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },

    #[error("vpci: invalid input: {0}")]
    InvalidInput(String),

    #[error("vpci: SMA error: {0}")]
    SmaError(#[from] SmaError),

    #[error("vpci: mismatched input lengths: close = {close_len}, volume = {volume_len}")]
    MismatchedInputLengths { close_len: usize, volume_len: usize },

    #[error("vpci: Mismatched output lengths: vpci_len = {vpci_len}, vpcis_len = {vpcis_len}, expected = {data_len}")]
    MismatchedOutputLengths {
        vpci_len: usize,
        vpcis_len: usize,
        data_len: usize,
    },

    #[error("vpci: Kernel not available")]
    KernelNotAvailable,
}

#[inline(always)]
fn first_valid_both(close: &[f64], volume: &[f64]) -> Option<usize> {
    close
        .iter()
        .zip(volume)
        .position(|(c, v)| !c.is_nan() && !v.is_nan())
}

#[inline(always)]
fn ensure_same_len(close: &[f64], volume: &[f64]) -> Result<(), VpciError> {
    if close.len() != volume.len() {
        return Err(VpciError::MismatchedInputLengths {
            close_len: close.len(),
            volume_len: volume.len(),
        });
    }
    Ok(())
}

#[inline(always)]
fn build_prefix_sums(close: &[f64], volume: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = close.len();
    let mut ps_close = vec![0.0; n + 1];
    let mut ps_vol = vec![0.0; n + 1];
    let mut ps_cv = vec![0.0; n + 1];

    for i in 0..n {
        let c = close[i];
        let v = volume[i];

        let c_val = if c.is_finite() { c } else { 0.0 };
        let v_val = if v.is_finite() { v } else { 0.0 };
        ps_close[i + 1] = ps_close[i] + c_val;
        ps_vol[i + 1] = ps_vol[i] + v_val;
        ps_cv[i + 1] = ps_cv[i] + c_val * v_val;
    }
    (ps_close, ps_vol, ps_cv)
}

#[inline(always)]
fn window_sum(ps: &[f64], start: usize, end_inclusive: usize) -> f64 {
    let a = start;
    let b = end_inclusive + 1;
    ps[b] - ps[a]
}

#[inline(always)]
fn vpci_prepare<'a>(
    input: &'a VpciInput,
    kernel: Kernel,
) -> Result<(&'a [f64], &'a [f64], usize, usize, usize, Kernel), VpciError> {
    let (close, volume) = match &input.data {
        VpciData::Candles {
            candles,
            close_source,
            volume_source,
        } => (
            source_type(candles, close_source),
            source_type(candles, volume_source),
        ),
        VpciData::Slices { close, volume } => (*close, *volume),
    };

    ensure_same_len(close, volume)?;

    let len = close.len();
    if len == 0 {
        return Err(VpciError::EmptyInputData);
    }
    let first = first_valid_both(close, volume).ok_or(VpciError::AllValuesNaN)?;

    let short = input.get_short_range();
    let long = input.get_long_range();
    if short == 0 || long == 0 || short > len || long > len {
        return Err(VpciError::InvalidPeriod {
            period: short.max(long),
            data_len: len,
        });
    }
    if short > long {
        return Err(VpciError::InvalidPeriod {
            period: short,
            data_len: long,
        });
    }
    if (len - first) < long {
        return Err(VpciError::NotEnoughValidData {
            needed: long,
            valid: len - first,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        k => k,
    };

    Ok((close, volume, first, short, long, chosen))
}

#[inline(always)]
fn vpci_scalar_into_from_psums(
    close: &[f64],
    volume: &[f64],
    first: usize,
    short: usize,
    long: usize,
    ps_close: &[f64],
    ps_vol: &[f64],
    ps_cv: &[f64],
    vpci_out: &mut [f64],
    vpcis_out: &mut [f64],
) {
    debug_assert_eq!(close.len(), volume.len());
    let n = close.len();
    let warmup = first + long - 1;
    if warmup >= n {
        return;
    }

    #[inline(always)]
    fn zf(x: f64) -> f64 {
        if x.is_finite() {
            x
        } else {
            0.0
        }
    }

    let inv_long = 1.0 / (long as f64);
    let inv_short = 1.0 / (short as f64);

    let mut sum_vpci_vol_short = 0.0;

    unsafe {
        let pc = ps_close.as_ptr();
        let pv = ps_vol.as_ptr();
        let pcv = ps_cv.as_ptr();
        let vptr = volume.as_ptr();
        let vpci_ptr = vpci_out.as_mut_ptr();
        let vpcis_ptr = vpcis_out.as_mut_ptr();

        let mut i = warmup;
        while i < n {
            let end = i + 1;
            let long_start = end.saturating_sub(long);
            let short_start = end.saturating_sub(short);

            let sc_l = *pc.add(end) - *pc.add(long_start);
            let sv_l = *pv.add(end) - *pv.add(long_start);
            let scv_l = *pcv.add(end) - *pcv.add(long_start);

            let sc_s = *pc.add(end) - *pc.add(short_start);
            let sv_s = *pv.add(end) - *pv.add(short_start);
            let scv_s = *pcv.add(end) - *pcv.add(short_start);

            let sma_l = sc_l * inv_long;
            let sma_s = sc_s * inv_short;
            let sma_v_l = sv_l * inv_long;
            let sma_v_s = sv_s * inv_short;

            let vwma_l = if sv_l != 0.0 { scv_l / sv_l } else { f64::NAN };
            let vwma_s = if sv_s != 0.0 { scv_s / sv_s } else { f64::NAN };

            let vpc = vwma_l - sma_l;
            let vpr = if sma_s != 0.0 {
                vwma_s / sma_s
            } else {
                f64::NAN
            };
            let vm = if sma_v_l != 0.0 {
                sma_v_s / sma_v_l
            } else {
                f64::NAN
            };

            let vpci = vpc * vpr * vm;
            *vpci_ptr.add(i) = vpci;

            let v_i = *vptr.add(i);
            sum_vpci_vol_short += zf(vpci) * zf(v_i);
            if i >= warmup + short {
                let rm_idx = i - short;
                let vpci_rm = *vpci_ptr.add(rm_idx);
                let v_rm = *vptr.add(rm_idx);
                sum_vpci_vol_short -= zf(vpci_rm) * zf(v_rm);
            }

            let denom = sma_v_s;
            *vpcis_ptr.add(i) = if denom != 0.0 && denom.is_finite() {
                (sum_vpci_vol_short * inv_short) / denom
            } else {
                f64::NAN
            };

            i += 1;
        }
    }
}

#[inline(always)]
fn vpci_compute_into(
    close: &[f64],
    volume: &[f64],
    first: usize,
    short: usize,
    long: usize,
    kernel: Kernel,
    vpci_out: &mut [f64],
    vpcis_out: &mut [f64],
) {
    let (ps_c, ps_v, ps_cv) = build_prefix_sums(close, volume);
    match kernel {
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512 => unsafe {
            vpci_avx512_into_from_psums(
                close, volume, first, short, long, &ps_c, &ps_v, &ps_cv, vpci_out, vpcis_out,
            );
        },
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2 => unsafe {
            vpci_avx2_into_from_psums(
                close, volume, first, short, long, &ps_c, &ps_v, &ps_cv, vpci_out, vpcis_out,
            );
        },
        _ => {
            vpci_scalar_into_from_psums(
                close, volume, first, short, long, &ps_c, &ps_v, &ps_cv, vpci_out, vpcis_out,
            );
        }
    }
}

#[inline]
pub fn vpci(input: &VpciInput) -> Result<VpciOutput, VpciError> {
    vpci_with_kernel(input, Kernel::Auto)
}

pub fn vpci_with_kernel(input: &VpciInput, kernel: Kernel) -> Result<VpciOutput, VpciError> {
    let (close, volume, first, short, long, chosen) = vpci_prepare(input, kernel)?;

    let len = close.len();
    let warmup = first + long - 1;
    let mut vpci = alloc_with_nan_prefix(len, warmup);
    let mut vpcis = alloc_with_nan_prefix(len, warmup);

    vpci_compute_into(
        close, volume, first, short, long, chosen, &mut vpci, &mut vpcis,
    );

    Ok(VpciOutput { vpci, vpcis })
}

#[inline]
pub fn vpci_into_slice(
    vpci_dst: &mut [f64],
    vpcis_dst: &mut [f64],
    input: &VpciInput,
    kernel: Kernel,
) -> Result<(), VpciError> {
    let (close, volume, first, short, long, chosen) = vpci_prepare(input, kernel)?;
    if vpci_dst.len() != close.len() || vpcis_dst.len() != close.len() {
        return Err(VpciError::OutputLengthMismatch {
            expected: close.len(),
            got: vpci_dst.len().min(vpcis_dst.len()),
        });
    }
    let warmup = first + long - 1;
    for i in 0..warmup.min(vpci_dst.len()) {
        vpci_dst[i] = f64::NAN;
        vpcis_dst[i] = f64::NAN;
    }
    vpci_compute_into(
        close, volume, first, short, long, chosen, vpci_dst, vpcis_dst,
    );
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn vpci_into(
    input: &VpciInput,
    out_vpci: &mut [f64],
    out_vpcis: &mut [f64],
) -> Result<(), VpciError> {
    vpci_into_slice(out_vpci, out_vpcis, input, Kernel::Auto)
}

#[inline]
pub unsafe fn vpci_scalar(
    close: &[f64],
    volume: &[f64],
    short: usize,
    long: usize,
) -> Result<VpciOutput, VpciError> {
    ensure_same_len(close, volume)?;
    let len = close.len();
    let first = first_valid_both(close, volume).ok_or(VpciError::AllValuesNaN)?;
    let warmup = first + long - 1;

    let mut vpci = alloc_with_nan_prefix(len, warmup);
    let mut vpcis = alloc_with_nan_prefix(len, warmup);

    vpci_compute_into(
        close,
        volume,
        first,
        short,
        long,
        Kernel::Scalar,
        &mut vpci,
        &mut vpcis,
    );

    Ok(VpciOutput { vpci, vpcis })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vpci_avx2(
    close: &[f64],
    volume: &[f64],
    short: usize,
    long: usize,
) -> Result<VpciOutput, VpciError> {
    ensure_same_len(close, volume)?;
    let len = close.len();
    let first = first_valid_both(close, volume).ok_or(VpciError::AllValuesNaN)?;
    let warmup = first + long - 1;

    let mut vpci = alloc_with_nan_prefix(len, warmup);
    let mut vpcis = alloc_with_nan_prefix(len, warmup);

    vpci_compute_into(
        close,
        volume,
        first,
        short,
        long,
        Kernel::Avx2,
        &mut vpci,
        &mut vpcis,
    );

    Ok(VpciOutput { vpci, vpcis })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vpci_avx512(
    close: &[f64],
    volume: &[f64],
    short: usize,
    long: usize,
) -> Result<VpciOutput, VpciError> {
    ensure_same_len(close, volume)?;
    let len = close.len();
    let first = first_valid_both(close, volume).ok_or(VpciError::AllValuesNaN)?;
    let warmup = first + long - 1;

    let mut vpci = alloc_with_nan_prefix(len, warmup);
    let mut vpcis = alloc_with_nan_prefix(len, warmup);

    vpci_compute_into(
        close,
        volume,
        first,
        short,
        long,
        Kernel::Avx512,
        &mut vpci,
        &mut vpcis,
    );

    Ok(VpciOutput { vpci, vpcis })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn vpci_avx2_into_from_psums(
    close: &[f64],
    volume: &[f64],
    first: usize,
    short: usize,
    long: usize,
    ps_close: &[f64],
    ps_vol: &[f64],
    ps_cv: &[f64],
    vpci_out: &mut [f64],
    vpcis_out: &mut [f64],
) {
    use core::arch::x86_64::*;

    let n = close.len();
    let warmup = first + long - 1;
    if warmup >= n {
        return;
    }

    let inv_long = _mm256_set1_pd(1.0 / (long as f64));
    let inv_short = _mm256_set1_pd(1.0 / (short as f64));
    let zero = _mm256_set1_pd(0.0);
    let nan = _mm256_set1_pd(f64::NAN);

    let pc = ps_close.as_ptr();
    let pv = ps_vol.as_ptr();
    let pcv = ps_cv.as_ptr();
    let yptr = vpci_out.as_mut_ptr();

    let mut i = warmup;
    let step = 4usize;
    let vec_end = n.saturating_sub(step) + 1;

    while i < vec_end {
        let end = i + 1;

        let c_end = _mm256_loadu_pd(pc.add(end));
        let c_l = _mm256_loadu_pd(pc.add(end - long));
        let v_end = _mm256_loadu_pd(pv.add(end));
        let v_l = _mm256_loadu_pd(pv.add(end - long));
        let cv_end = _mm256_loadu_pd(pcv.add(end));
        let cv_l = _mm256_loadu_pd(pcv.add(end - long));

        let c_s = _mm256_loadu_pd(pc.add(end - short));
        let v_s = _mm256_loadu_pd(pv.add(end - short));
        let cv_s = _mm256_loadu_pd(pcv.add(end - short));

        let sc_l = _mm256_sub_pd(c_end, c_l);
        let sv_l = _mm256_sub_pd(v_end, v_l);
        let scv_l = _mm256_sub_pd(cv_end, cv_l);

        let sc_s = _mm256_sub_pd(c_end, c_s);
        let sv_s = _mm256_sub_pd(v_end, v_s);
        let scv_s = _mm256_sub_pd(cv_end, cv_s);

        let sma_l = _mm256_mul_pd(sc_l, inv_long);
        let sma_s = _mm256_mul_pd(sc_s, inv_short);
        let sma_v_l = _mm256_mul_pd(sv_l, inv_long);
        let sma_v_s = _mm256_mul_pd(sv_s, inv_short);

        let mask_l = _mm256_cmp_pd(sv_l, zero, _CMP_NEQ_OQ);
        let vwma_l = _mm256_blendv_pd(nan, _mm256_div_pd(scv_l, sv_l), mask_l);

        let mask_s = _mm256_cmp_pd(sv_s, zero, _CMP_NEQ_OQ);
        let vwma_s = _mm256_blendv_pd(nan, _mm256_div_pd(scv_s, sv_s), mask_s);

        let vpc = _mm256_sub_pd(vwma_l, sma_l);
        let mask_vpr = _mm256_cmp_pd(sma_s, zero, _CMP_NEQ_OQ);
        let vpr = _mm256_blendv_pd(nan, _mm256_div_pd(vwma_s, sma_s), mask_vpr);
        let mask_vm = _mm256_cmp_pd(sma_v_l, zero, _CMP_NEQ_OQ);
        let vm = _mm256_blendv_pd(nan, _mm256_div_pd(sma_v_s, sma_v_l), mask_vm);

        let vpci = _mm256_mul_pd(_mm256_mul_pd(vpc, vpr), vm);
        _mm256_storeu_pd(yptr.add(i), vpci);
        i += step;
    }

    while i < n {
        let end = i + 1;
        let long_start = end - long;
        let short_start = end - short;

        let sc_l = *pc.add(end) - *pc.add(long_start);
        let sv_l = *pv.add(end) - *pv.add(long_start);
        let scv_l = *pcv.add(end) - *pcv.add(long_start);
        let sc_s = *pc.add(end) - *pc.add(short_start);
        let sv_s = *pv.add(end) - *pv.add(short_start);
        let scv_s = *pcv.add(end) - *pcv.add(short_start);

        let sma_l = sc_l * (1.0 / long as f64);
        let sma_s = sc_s * (1.0 / short as f64);
        let sma_v_l = sv_l * (1.0 / long as f64);
        let sma_v_s = sv_s * (1.0 / short as f64);

        let vwma_l = if sv_l != 0.0 { scv_l / sv_l } else { f64::NAN };
        let vwma_s = if sv_s != 0.0 { scv_s / sv_s } else { f64::NAN };

        let vpc = vwma_l - sma_l;
        let vpr = if sma_s != 0.0 {
            vwma_s / sma_s
        } else {
            f64::NAN
        };
        let vm = if sma_v_l != 0.0 {
            sma_v_s / sma_v_l
        } else {
            f64::NAN
        };
        *yptr.add(i) = vpc * vpr * vm;
        i += 1;
    }

    #[inline(always)]
    fn zf(x: f64) -> f64 {
        if x.is_finite() {
            x
        } else {
            0.0
        }
    }

    let inv_short_s = 1.0 / (short as f64);
    let vptr = volume.as_ptr();
    let ysp = vpcis_out.as_mut_ptr();

    let mut sum_vpci_vol_short = 0.0;
    let mut t = warmup;
    while t < n {
        let vpci = *yptr.add(t);
        let vi = *vptr.add(t);
        sum_vpci_vol_short += zf(vpci) * zf(vi);
        if t >= warmup + short {
            let rm = t - short;
            sum_vpci_vol_short -= zf(*yptr.add(rm)) * zf(*vptr.add(rm));
        }

        let end = t + 1;
        let sv_s = *pv.add(end) - *pv.add(end - short);
        let denom = sv_s * inv_short_s;
        *ysp.add(t) = if denom != 0.0 && denom.is_finite() {
            (sum_vpci_vol_short * inv_short_s) / denom
        } else {
            f64::NAN
        };
        t += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn vpci_avx512_into_from_psums(
    close: &[f64],
    volume: &[f64],
    first: usize,
    short: usize,
    long: usize,
    ps_close: &[f64],
    ps_vol: &[f64],
    ps_cv: &[f64],
    vpci_out: &mut [f64],
    vpcis_out: &mut [f64],
) {
    use core::arch::x86_64::*;

    let n = close.len();
    let warmup = first + long - 1;
    if warmup >= n {
        return;
    }

    let inv_long = _mm512_set1_pd(1.0 / (long as f64));
    let inv_short = _mm512_set1_pd(1.0 / (short as f64));
    let zero = _mm512_set1_pd(0.0);
    let nan = _mm512_set1_pd(f64::NAN);

    let pc = ps_close.as_ptr();
    let pv = ps_vol.as_ptr();
    let pcv = ps_cv.as_ptr();
    let yptr = vpci_out.as_mut_ptr();

    let mut i = warmup;
    let step = 8usize;
    let vec_end = n.saturating_sub(step) + 1;

    while i < vec_end {
        let end = i + 1;

        let c_end = _mm512_loadu_pd(pc.add(end));
        let c_l = _mm512_loadu_pd(pc.add(end - long));
        let v_end = _mm512_loadu_pd(pv.add(end));
        let v_l = _mm512_loadu_pd(pv.add(end - long));
        let cv_end = _mm512_loadu_pd(pcv.add(end));
        let cv_l = _mm512_loadu_pd(pcv.add(end - long));

        let c_s = _mm512_loadu_pd(pc.add(end - short));
        let v_s = _mm512_loadu_pd(pv.add(end - short));
        let cv_s = _mm512_loadu_pd(pcv.add(end - short));

        let sc_l = _mm512_sub_pd(c_end, c_l);
        let sv_l = _mm512_sub_pd(v_end, v_l);
        let scv_l = _mm512_sub_pd(cv_end, cv_l);

        let sc_s = _mm512_sub_pd(c_end, c_s);
        let sv_s = _mm512_sub_pd(v_end, v_s);
        let scv_s = _mm512_sub_pd(cv_end, cv_s);

        let sma_l = _mm512_mul_pd(sc_l, inv_long);
        let sma_s = _mm512_mul_pd(sc_s, inv_short);
        let sma_v_l = _mm512_mul_pd(sv_l, inv_long);
        let sma_v_s = _mm512_mul_pd(sv_s, inv_short);

        let mk_l = _mm512_cmp_pd_mask(sv_l, zero, _CMP_NEQ_OQ);
        let mk_s = _mm512_cmp_pd_mask(sv_s, zero, _CMP_NEQ_OQ);
        let mk_vr = _mm512_cmp_pd_mask(sma_s, zero, _CMP_NEQ_OQ);
        let mk_vm = _mm512_cmp_pd_mask(sma_v_l, zero, _CMP_NEQ_OQ);

        let vwma_l = _mm512_mask_div_pd(nan, mk_l, scv_l, sv_l);
        let vwma_s = _mm512_mask_div_pd(nan, mk_s, scv_s, sv_s);

        let vpc = _mm512_sub_pd(vwma_l, sma_l);
        let vpr = _mm512_mask_div_pd(nan, mk_vr, vwma_s, sma_s);
        let vm = _mm512_mask_div_pd(nan, mk_vm, sma_v_s, sma_v_l);

        let vpci = _mm512_mul_pd(_mm512_mul_pd(vpc, vpr), vm);
        _mm512_storeu_pd(yptr.add(i), vpci);
        i += step;
    }

    while i < n {
        let end = i + 1;
        let long_start = end - long;
        let short_start = end - short;

        let sc_l = *pc.add(end) - *pc.add(long_start);
        let sv_l = *pv.add(end) - *pv.add(long_start);
        let scv_l = *pcv.add(end) - *pcv.add(long_start);
        let sc_s = *pc.add(end) - *pc.add(short_start);
        let sv_s = *pv.add(end) - *pv.add(short_start);
        let scv_s = *pcv.add(end) - *pcv.add(short_start);

        let sma_l = sc_l * (1.0 / long as f64);
        let sma_s = sc_s * (1.0 / short as f64);
        let sma_v_l = sv_l * (1.0 / long as f64);
        let sma_v_s = sv_s * (1.0 / short as f64);

        let vwma_l = if sv_l != 0.0 { scv_l / sv_l } else { f64::NAN };
        let vwma_s = if sv_s != 0.0 { scv_s / sv_s } else { f64::NAN };

        let vpc = vwma_l - sma_l;
        let vpr = if sma_s != 0.0 {
            vwma_s / sma_s
        } else {
            f64::NAN
        };
        let vm = if sma_v_l != 0.0 {
            sma_v_s / sma_v_l
        } else {
            f64::NAN
        };
        *yptr.add(i) = vpc * vpr * vm;
        i += 1;
    }

    #[inline(always)]
    fn zf(x: f64) -> f64 {
        if x.is_finite() {
            x
        } else {
            0.0
        }
    }

    let inv_short_s = 1.0 / (short as f64);
    let vptr = volume.as_ptr();
    let ysp = vpcis_out.as_mut_ptr();

    let mut sum_vpci_vol_short = 0.0;
    let mut t = warmup;
    while t < n {
        let vpci = *yptr.add(t);
        let vi = *vptr.add(t);
        sum_vpci_vol_short += zf(vpci) * zf(vi);
        if t >= warmup + short {
            let rm = t - short;
            sum_vpci_vol_short -= zf(*yptr.add(rm)) * zf(*vptr.add(rm));
        }

        let end = t + 1;
        let sv_s = *pv.add(end) - *pv.add(end - short);
        let denom = sv_s * inv_short_s;
        *ysp.add(t) = if denom != 0.0 && denom.is_finite() {
            (sum_vpci_vol_short * inv_short_s) / denom
        } else {
            f64::NAN
        };
        t += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vpci_avx512_short(
    close: &[f64],
    volume: &[f64],
    short: usize,
    long: usize,
) -> Result<VpciOutput, VpciError> {
    vpci_avx512(close, volume, short, long)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vpci_avx512_long(
    close: &[f64],
    volume: &[f64],
    short: usize,
    long: usize,
) -> Result<VpciOutput, VpciError> {
    vpci_avx512(close, volume, short, long)
}

#[inline]
pub fn vpci_batch_with_kernel(
    close: &[f64],
    volume: &[f64],
    sweep: &VpciBatchRange,
    kernel: Kernel,
) -> Result<VpciBatchOutput, VpciError> {
    let k = match kernel {
        Kernel::Auto => match detect_best_batch_kernel() {
            Kernel::Avx512Batch => Kernel::Avx2Batch,
            other => other,
        },
        other if other.is_batch() => other,
        other => {
            return Err(VpciError::InvalidKernelForBatch(other));
        }
    };
    let simd = match k {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    vpci_batch_par_slice(close, volume, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct VpciBatchRange {
    pub short_range: (usize, usize, usize),
    pub long_range: (usize, usize, usize),
}

impl Default for VpciBatchRange {
    fn default() -> Self {
        Self {
            short_range: (5, 5, 0),
            long_range: (25, 274, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct VpciBatchBuilder {
    range: VpciBatchRange,
    kernel: Kernel,
}

impl VpciBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    pub fn short_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.short_range = (start, end, step);
        self
    }
    pub fn long_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.long_range = (start, end, step);
        self
    }
    pub fn apply_slices(self, close: &[f64], volume: &[f64]) -> Result<VpciBatchOutput, VpciError> {
        vpci_batch_with_kernel(close, volume, &self.range, self.kernel)
    }
}

#[derive(Clone, Debug)]
pub struct VpciBatchOutput {
    pub vpci: Vec<f64>,
    pub vpcis: Vec<f64>,
    pub combos: Vec<VpciParams>,
    pub rows: usize,
    pub cols: usize,
}
impl VpciBatchOutput {
    pub fn row_for_params(&self, p: &VpciParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.short_range.unwrap_or(5) == p.short_range.unwrap_or(5)
                && c.long_range.unwrap_or(25) == p.long_range.unwrap_or(25)
        })
    }
    pub fn vpci_for(&self, p: &VpciParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.vpci[start..start + self.cols]
        })
    }
    pub fn vpcis_for(&self, p: &VpciParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.vpcis[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &VpciBatchRange) -> Vec<VpciParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        let mut out = Vec::new();
        if start < end {
            let mut v = start;
            loop {
                out.push(v);
                match v.checked_add(step) {
                    Some(next) if next <= end => v = next,
                    _ => break,
                }
            }
        } else {
            let mut v = start;
            loop {
                out.push(v);
                if v == end {
                    break;
                }
                match v.checked_sub(step) {
                    Some(next) if next >= end => v = next,
                    _ => break,
                }
            }
        }
        out
    }

    let shorts = axis_usize(r.short_range);
    let longs = axis_usize(r.long_range);

    let mut out = Vec::with_capacity(shorts.len().saturating_mul(longs.len()));
    for &s in &shorts {
        for &l in &longs {
            out.push(VpciParams {
                short_range: Some(s),
                long_range: Some(l),
            });
        }
    }
    out
}

#[inline(always)]
pub fn vpci_batch_slice(
    close: &[f64],
    volume: &[f64],
    sweep: &VpciBatchRange,
    kernel: Kernel,
) -> Result<VpciBatchOutput, VpciError> {
    vpci_batch_inner(close, volume, sweep, kernel, false)
}

#[inline(always)]
pub fn vpci_batch_par_slice(
    close: &[f64],
    volume: &[f64],
    sweep: &VpciBatchRange,
    kernel: Kernel,
) -> Result<VpciBatchOutput, VpciError> {
    vpci_batch_inner(close, volume, sweep, kernel, true)
}

#[inline(always)]
fn vpci_batch_inner(
    close: &[f64],
    volume: &[f64],
    sweep: &VpciBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<VpciBatchOutput, VpciError> {
    ensure_same_len(close, volume)?;
    let combos = expand_grid(sweep);
    let cols = close.len();
    let rows = combos.len();
    if cols == 0 {
        return Err(VpciError::EmptyInputData);
    }
    if rows == 0 {
        let (start, end, step) = sweep.short_range;
        return Err(VpciError::InvalidRange { start, end, step });
    }

    let first = first_valid_both(close, volume).ok_or(VpciError::AllValuesNaN)?;
    let warmups: Vec<usize> = combos
        .iter()
        .map(|p| first + p.long_range.unwrap() - 1)
        .collect();

    let mut vpci_mu = make_uninit_matrix(rows, cols);
    let mut vpcis_mu = make_uninit_matrix(rows, cols);

    init_matrix_prefixes(&mut vpci_mu, cols, &warmups);
    init_matrix_prefixes(&mut vpcis_mu, cols, &warmups);

    let ptr_v = vpci_mu.as_ptr() as *mut f64;
    let ptr_s = vpcis_mu.as_ptr() as *mut f64;
    let cap_v = vpci_mu.capacity();
    let cap_s = vpcis_mu.capacity();

    let total_len = rows
        .checked_mul(cols)
        .ok_or_else(|| VpciError::InvalidInput("rows*cols overflow in vpci_batch_inner".into()))?;
    let vpci_slice = unsafe { core::slice::from_raw_parts_mut(ptr_v, total_len) };
    let vpcis_slice = unsafe { core::slice::from_raw_parts_mut(ptr_s, total_len) };

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

    let combos = vpci_batch_inner_into(
        close,
        volume,
        sweep,
        simd,
        parallel,
        vpci_slice,
        vpcis_slice,
    )?;

    core::mem::forget(vpci_mu);
    core::mem::forget(vpcis_mu);
    let vpci_vec = unsafe { Vec::from_raw_parts(ptr_v, total_len, cap_v) };
    let vpcis_vec = unsafe { Vec::from_raw_parts(ptr_s, total_len, cap_s) };

    Ok(VpciBatchOutput {
        vpci: vpci_vec,
        vpcis: vpcis_vec,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn vpci_batch_inner_into(
    close: &[f64],
    volume: &[f64],
    sweep: &VpciBatchRange,
    kernel: Kernel,
    parallel: bool,
    vpci_out: &mut [f64],
    vpcis_out: &mut [f64],
) -> Result<Vec<VpciParams>, VpciError> {
    ensure_same_len(close, volume)?;
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        let (start, end, step) = sweep.short_range;
        return Err(VpciError::InvalidRange { start, end, step });
    }
    let len = close.len();
    let first = first_valid_both(close, volume).ok_or(VpciError::AllValuesNaN)?;
    let max_long = combos.iter().map(|c| c.long_range.unwrap()).max().unwrap();
    if len - first < max_long {
        return Err(VpciError::NotEnoughValidData {
            needed: max_long,
            valid: len - first,
        });
    }
    let rows = combos.len();
    let cols = len;

    let (ps_c, ps_v, ps_cv) = build_prefix_sums(close, volume);

    for (row, prm) in combos.iter().enumerate() {
        let warmup = first + prm.long_range.unwrap() - 1;
        let s = row * cols;
        for i in 0..warmup.min(cols) {
            vpci_out[s + i] = f64::NAN;
            vpcis_out[s + i] = f64::NAN;
        }
    }

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;
            vpci_out
                .par_chunks_mut(cols)
                .zip(vpcis_out.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(row, (dst_vpci, dst_vpcis))| {
                    let prm = &combos[row];
                    let short = prm.short_range.unwrap();
                    let long = prm.long_range.unwrap();

                    let use_simd = short <= long;
                    match (use_simd, kernel) {
                        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                        (true, Kernel::Avx512) => unsafe {
                            vpci_avx512_into_from_psums(
                                close, volume, first, short, long, &ps_c, &ps_v, &ps_cv, dst_vpci,
                                dst_vpcis,
                            );
                        },
                        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                        (true, Kernel::Avx2) => unsafe {
                            vpci_avx2_into_from_psums(
                                close, volume, first, short, long, &ps_c, &ps_v, &ps_cv, dst_vpci,
                                dst_vpcis,
                            );
                        },
                        _ => {
                            vpci_scalar_into_from_psums(
                                close, volume, first, short, long, &ps_c, &ps_v, &ps_cv, dst_vpci,
                                dst_vpcis,
                            );
                        }
                    }
                });
        }
        #[cfg(target_arch = "wasm32")]
        {
            for row in 0..rows {
                let prm = &combos[row];
                let short = prm.short_range.unwrap();
                let long = prm.long_range.unwrap();

                let row_off = row * cols;
                let dst_vpci = &mut vpci_out[row_off..row_off + cols];
                let dst_vpcis = &mut vpcis_out[row_off..row_off + cols];

                vpci_scalar_into_from_psums(
                    close, volume, first, short, long, &ps_c, &ps_v, &ps_cv, dst_vpci, dst_vpcis,
                );
            }
        }
    } else {
        for row in 0..rows {
            let prm = &combos[row];
            let short = prm.short_range.unwrap();
            let long = prm.long_range.unwrap();

            let row_off = row * cols;
            let dst_vpci = &mut vpci_out[row_off..row_off + cols];
            let dst_vpcis = &mut vpcis_out[row_off..row_off + cols];

            let use_simd = short <= long;
            match (use_simd, kernel) {
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                (true, Kernel::Avx512) => unsafe {
                    vpci_avx512_into_from_psums(
                        close, volume, first, short, long, &ps_c, &ps_v, &ps_cv, dst_vpci,
                        dst_vpcis,
                    );
                },
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                (true, Kernel::Avx2) => unsafe {
                    vpci_avx2_into_from_psums(
                        close, volume, first, short, long, &ps_c, &ps_v, &ps_cv, dst_vpci,
                        dst_vpcis,
                    );
                },
                _ => {
                    vpci_scalar_into_from_psums(
                        close, volume, first, short, long, &ps_c, &ps_v, &ps_cv, dst_vpci,
                        dst_vpcis,
                    );
                }
            }
        }
    }

    Ok(combos)
}

#[inline(always)]
pub unsafe fn vpci_row_scalar(
    close: &[f64],
    volume: &[f64],
    short: usize,
    long: usize,
) -> Result<VpciOutput, VpciError> {
    vpci_scalar(close, volume, short, long)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vpci_row_avx2(
    close: &[f64],
    volume: &[f64],
    short: usize,
    long: usize,
) -> Result<VpciOutput, VpciError> {
    vpci_avx2(close, volume, short, long)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vpci_row_avx512(
    close: &[f64],
    volume: &[f64],
    short: usize,
    long: usize,
) -> Result<VpciOutput, VpciError> {
    if long <= 32 {
        vpci_row_avx512_short(close, volume, short, long)
    } else {
        vpci_row_avx512_long(close, volume, short, long)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vpci_row_avx512_short(
    close: &[f64],
    volume: &[f64],
    short: usize,
    long: usize,
) -> Result<VpciOutput, VpciError> {
    vpci_avx512(close, volume, short, long)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vpci_row_avx512_long(
    close: &[f64],
    volume: &[f64],
    short: usize,
    long: usize,
) -> Result<VpciOutput, VpciError> {
    vpci_avx512(close, volume, short, long)
}

#[inline(always)]
pub fn expand_grid_vpci(r: &VpciBatchRange) -> Vec<VpciParams> {
    expand_grid(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn check_vpci_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = VpciParams {
            short_range: Some(3),
            long_range: None,
        };
        let input = VpciInput::from_candles(&candles, "close", "volume", params);
        let output = vpci_with_kernel(&input, kernel)?;
        assert_eq!(output.vpci.len(), candles.close.len());
        assert_eq!(output.vpcis.len(), candles.close.len());
        Ok(())
    }

    fn check_vpci_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = VpciParams {
            short_range: Some(5),
            long_range: Some(25),
        };
        let input = VpciInput::from_candles(&candles, "close", "volume", params);
        let output = vpci_with_kernel(&input, kernel)?;

        let vpci_len = output.vpci.len();
        let vpcis_len = output.vpcis.len();
        assert_eq!(vpci_len, candles.close.len());
        assert_eq!(vpcis_len, candles.close.len());

        let vpci_last_five = &output.vpci[vpci_len.saturating_sub(5)..];
        let vpcis_last_five = &output.vpcis[vpcis_len.saturating_sub(5)..];
        let expected_vpci = [
            -319.65148214323426,
            -133.61700649928346,
            -144.76194155503174,
            -83.55576212490328,
            -169.53504207700533,
        ];
        let expected_vpcis = [
            -1049.2826640115732,
            -694.1067814399748,
            -519.6960416662324,
            -330.9401404636258,
            -173.004986803695,
        ];
        for (i, &val) in vpci_last_five.iter().enumerate() {
            let diff = (val - expected_vpci[i]).abs();
            assert!(
                diff < 5e-2,
                "[{}] VPCI mismatch at idx {}: got {}, expected {}",
                test_name,
                i,
                val,
                expected_vpci[i]
            );
        }
        for (i, &val) in vpcis_last_five.iter().enumerate() {
            let diff = (val - expected_vpcis[i]).abs();
            assert!(
                diff < 5e-2,
                "[{}] VPCIS mismatch at idx {}: got {}, expected {}",
                test_name,
                i,
                val,
                expected_vpcis[i]
            );
        }
        Ok(())
    }

    fn check_vpci_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = VpciInput::with_default_candles(&candles);
        let output = vpci_with_kernel(&input, kernel)?;
        assert_eq!(output.vpci.len(), candles.close.len());
        assert_eq!(output.vpcis.len(), candles.close.len());
        Ok(())
    }

    fn check_vpci_slice_input(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let close_data = [10.0, 12.0, 14.0, 13.0, 15.0];
        let volume_data = [100.0, 200.0, 300.0, 250.0, 400.0];
        let params = VpciParams {
            short_range: Some(2),
            long_range: Some(3),
        };
        let input = VpciInput::from_slices(&close_data, &volume_data, params);
        let output = vpci_with_kernel(&input, kernel)?;
        assert_eq!(output.vpci.len(), close_data.len());
        assert_eq!(output.vpcis.len(), close_data.len());
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_vpci_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let test_params = vec![
            VpciParams::default(),
            VpciParams {
                short_range: Some(2),
                long_range: Some(3),
            },
            VpciParams {
                short_range: Some(2),
                long_range: Some(10),
            },
            VpciParams {
                short_range: Some(5),
                long_range: Some(20),
            },
            VpciParams {
                short_range: Some(10),
                long_range: Some(30),
            },
            VpciParams {
                short_range: Some(20),
                long_range: Some(50),
            },
            VpciParams {
                short_range: Some(3),
                long_range: Some(100),
            },
            VpciParams {
                short_range: Some(50),
                long_range: Some(100),
            },
            VpciParams {
                short_range: Some(7),
                long_range: Some(21),
            },
            VpciParams {
                short_range: Some(14),
                long_range: Some(28),
            },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = VpciInput::from_candles(&candles, "close", "volume", params.clone());
            let output = vpci_with_kernel(&input, kernel)?;

            for (i, &val) in output.vpci.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 in VPCI with params: short_range={}, long_range={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.short_range.unwrap_or(5),
                        params.long_range.unwrap_or(25),
                        param_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 in VPCI with params: short_range={}, long_range={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.short_range.unwrap_or(5),
                        params.long_range.unwrap_or(25),
                        param_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 in VPCI with params: short_range={}, long_range={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.short_range.unwrap_or(5),
                        params.long_range.unwrap_or(25),
                        param_idx
                    );
                }
            }

            for (i, &val) in output.vpcis.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 in VPCIS with params: short_range={}, long_range={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.short_range.unwrap_or(5),
                        params.long_range.unwrap_or(25),
                        param_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 in VPCIS with params: short_range={}, long_range={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.short_range.unwrap_or(5),
                        params.long_range.unwrap_or(25),
                        param_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 in VPCIS with params: short_range={}, long_range={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.short_range.unwrap_or(5),
                        params.long_range.unwrap_or(25),
                        param_idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_vpci_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    fn calculate_variance(data: &[f64]) -> f64 {
        let finite_values: Vec<f64> = data.iter().filter(|v| v.is_finite()).copied().collect();

        if finite_values.len() < 2 {
            return 0.0;
        }

        let mean = finite_values.iter().sum::<f64>() / finite_values.len() as f64;
        let variance = finite_values
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / finite_values.len() as f64;

        variance
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_vpci_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let strat = (2usize..=20).prop_flat_map(|short_range| {
            ((short_range + 1)..=50).prop_flat_map(move |long_range| {
                let min_len = long_range + 10;
                (min_len..400).prop_flat_map(move |data_len| {
                    (
                        prop::collection::vec(
                            (100f64..10000f64).prop_filter("finite", |x| x.is_finite()),
                            data_len,
                        ),
                        prop::collection::vec(
                            (1000f64..1000000f64).prop_filter("finite", |x| x.is_finite()),
                            data_len,
                        ),
                        Just(short_range),
                        Just(long_range),
                    )
                })
            })
        });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(close, volume, short_range, long_range)| {
                let params = VpciParams {
                    short_range: Some(short_range),
                    long_range: Some(long_range),
                };
                let input = VpciInput::from_slices(&close, &volume, params);

                let VpciOutput {
                    vpci: out,
                    vpcis: out_smooth,
                } = vpci_with_kernel(&input, kernel).unwrap();
                let VpciOutput {
                    vpci: ref_out,
                    vpcis: ref_out_smooth,
                } = vpci_with_kernel(&input, Kernel::Scalar).unwrap();

                let first_valid = close
                    .iter()
                    .zip(volume.iter())
                    .position(|(c, v)| !c.is_nan() && !v.is_nan())
                    .unwrap_or(0);

                let expected_warmup = first_valid + long_range - 1;

                for i in 0..expected_warmup.min(out.len()) {
                    prop_assert!(
                        out[i].is_nan(),
                        "Expected NaN during warmup at index {}, got {}",
                        i,
                        out[i]
                    );
                    prop_assert!(
                        out_smooth[i].is_nan(),
                        "Expected NaN in VPCIS during warmup at index {}, got {}",
                        i,
                        out_smooth[i]
                    );
                }

                for i in expected_warmup..close.len() {
                    let y = out[i];
                    let ys = out_smooth[i];
                    let r = ref_out[i];
                    let rs = ref_out_smooth[i];

                    if !close[i].is_nan() && !volume[i].is_nan() {
                        prop_assert!(
                            y.is_finite() || r.is_nan(),
                            "VPCI should be finite at idx {} after warmup, got {}",
                            i,
                            y
                        );
                    }

                    if !y.is_finite() || !r.is_finite() {
                        prop_assert!(
                            y.to_bits() == r.to_bits(),
                            "finite/NaN mismatch in VPCI at idx {}: {} vs {}",
                            i,
                            y,
                            r
                        );
                    } else {
                        let y_bits = y.to_bits();
                        let r_bits = r.to_bits();
                        let ulp_diff: u64 = y_bits.abs_diff(r_bits);

                        prop_assert!(
                            (y - r).abs() <= 1e-9 || ulp_diff <= 4,
                            "VPCI mismatch at idx {}: {} vs {} (ULP={})",
                            i,
                            y,
                            r,
                            ulp_diff
                        );
                    }

                    if !ys.is_finite() || !rs.is_finite() {
                        prop_assert!(
                            ys.to_bits() == rs.to_bits(),
                            "finite/NaN mismatch in VPCIS at idx {}: {} vs {}",
                            i,
                            ys,
                            rs
                        );
                    } else {
                        let ys_bits = ys.to_bits();
                        let rs_bits = rs.to_bits();
                        let ulp_diff: u64 = ys_bits.abs_diff(rs_bits);

                        prop_assert!(
                            (ys - rs).abs() <= 1e-9 || ulp_diff <= 4,
                            "VPCIS mismatch at idx {}: {} vs {} (ULP={})",
                            i,
                            ys,
                            rs,
                            ulp_diff
                        );
                    }
                }

                let prices_constant = close.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-9);

                if prices_constant && expected_warmup < close.len() {
                    for i in expected_warmup..close.len() {
                        if out[i].is_finite() {
                            prop_assert!(
                                out[i].abs() <= 1e-6,
                                "VPCI should be ~0 when prices are constant, got {} at index {}",
                                out[i],
                                i
                            );
                        }
                    }
                }

                let volumes_constant = volume.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-9);

                if volumes_constant && expected_warmup < close.len() {
                    for i in expected_warmup..close.len() {
                        if out[i].is_finite() && ref_out[i].is_finite() {
                            prop_assert!(
                                (out[i] - ref_out[i]).abs() <= 1e-9,
                                "VPCI kernels should match exactly with constant volume"
                            );
                        }
                    }
                }

                if expected_warmup + short_range < close.len() {
                    for i in (expected_warmup + short_range)..close.len() {
                        if out[i].is_finite() && volume[i].is_finite() && volume[i] > 0.0 {
                            if !out_smooth[i].is_finite() {
                                let vol_window = &volume[i.saturating_sub(short_range - 1)..=i];
                                let vol_sum: f64 = vol_window.iter().sum();
                                prop_assert!(
									vol_sum.abs() < 1e-10,
									"VPCIS should be finite when VPCI is finite and volume > 0 at index {}",
									i
								);
                            }
                        }
                    }
                }

                if short_range == long_range && expected_warmup < close.len() {
                    for i in expected_warmup..close.len().min(expected_warmup + 10) {
                        if out[i].is_finite() {
                            prop_assert!(
                                !out[i].is_nan(),
                                "VPCI should be valid even when short_range == long_range"
                            );
                        }
                    }
                }

                let extreme_ratio = long_range as f64 / short_range as f64 > 10.0;
                if extreme_ratio && expected_warmup < close.len() {
                    for i in expected_warmup..close.len().min(expected_warmup + 5) {
                        prop_assert!(
                            out[i].is_nan() || out[i].is_finite(),
                            "VPCI should handle extreme parameter ratios gracefully at index {}",
                            i
                        );
                    }
                }

                let valid_count = out
                    .iter()
                    .skip(expected_warmup)
                    .filter(|v| v.is_finite())
                    .count();

                let ref_valid_count = ref_out
                    .iter()
                    .skip(expected_warmup)
                    .filter(|v| v.is_finite())
                    .count();

                prop_assert_eq!(
                    valid_count,
                    ref_valid_count,
                    "Valid value count mismatch between kernels"
                );

                Ok(())
            })
            .unwrap();

        Ok(())
    }

    macro_rules! generate_all_vpci_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $( #[test] fn [<$test_fn _scalar_f64>]() { let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar); } )*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $(
                    #[test] fn [<$test_fn _avx2_f64>]() { let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2); }
                    #[test] fn [<$test_fn _avx512_f64>]() { let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512); }
                )*
            }
        }
    }

    generate_all_vpci_tests!(
        check_vpci_partial_params,
        check_vpci_accuracy,
        check_vpci_default_candles,
        check_vpci_slice_input,
        check_vpci_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_vpci_tests!(check_vpci_property);

    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let close = &c.close;
        let volume = &c.volume;

        let output = VpciBatchBuilder::new()
            .kernel(kernel)
            .apply_slices(close, volume)?;

        let def = VpciParams::default();
        let row = output.vpci_for(&def).expect("default row missing");

        assert_eq!(row.len(), close.len());

        let expected = [
            -319.65148214323426,
            -133.61700649928346,
            -144.76194155503174,
            -83.55576212490328,
            -169.53504207700533,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 5e-2,
                "[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
            );
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let close = &c.close;
        let volume = &c.volume;

        let test_configs = vec![
            (2, 10, 2, 5, 25, 5),
            (5, 15, 5, 20, 40, 10),
            (10, 20, 5, 30, 60, 15),
            (2, 5, 1, 10, 15, 1),
            (20, 30, 2, 40, 60, 5),
            (3, 7, 2, 21, 35, 7),
            (8, 12, 1, 25, 30, 1),
            (2, 50, 10, 10, 100, 20),
        ];

        for (cfg_idx, &(short_start, short_end, short_step, long_start, long_end, long_step)) in
            test_configs.iter().enumerate()
        {
            let output = VpciBatchBuilder::new()
                .kernel(kernel)
                .short_range(short_start, short_end, short_step)
                .long_range(long_start, long_end, long_step)
                .apply_slices(close, volume)?;

            for (idx, &val) in output.vpci.iter().enumerate() {
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
						 in VPCI at row {} col {} (flat index {}) with params: short_range={}, long_range={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.short_range.unwrap_or(5),
                        combo.long_range.unwrap_or(25)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 in VPCI at row {} col {} (flat index {}) with params: short_range={}, long_range={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.short_range.unwrap_or(5),
                        combo.long_range.unwrap_or(25)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 in VPCI at row {} col {} (flat index {}) with params: short_range={}, long_range={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.short_range.unwrap_or(5),
                        combo.long_range.unwrap_or(25)
                    );
                }
            }

            for (idx, &val) in output.vpcis.iter().enumerate() {
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
						 in VPCIS at row {} col {} (flat index {}) with params: short_range={}, long_range={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.short_range.unwrap_or(5),
                        combo.long_range.unwrap_or(25)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 in VPCIS at row {} col {} (flat index {}) with params: short_range={}, long_range={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.short_range.unwrap_or(5),
                        combo.long_range.unwrap_or(25)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 in VPCIS at row {} col {} (flat index {}) with params: short_range={}, long_range={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.short_range.unwrap_or(5),
                        combo.long_range.unwrap_or(25)
                    );
                }
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

#[cfg(test)]
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
mod tests_into {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_vpci_into_matches_api() -> Result<(), Box<dyn std::error::Error>> {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let params = VpciParams::default();
        let input = VpciInput::from_candles(&candles, "close", "volume", params);

        let base = vpci(&input)?;

        let n = candles.close.len();
        let mut y = vec![0.0f64; n];
        let mut ys = vec![0.0f64; n];
        vpci_into(&input, &mut y, &mut ys)?;

        assert_eq!(base.vpci.len(), y.len());
        assert_eq!(base.vpcis.len(), ys.len());

        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-12 || a.to_bits() == b.to_bits()
        }

        for i in 0..n {
            assert!(
                eq_or_both_nan(base.vpci[i], y[i]),
                "VPCI mismatch at {}: base={}, into={}",
                i,
                base.vpci[i],
                y[i]
            );
            assert!(
                eq_or_both_nan(base.vpcis[i], ys[i]),
                "VPCIS mismatch at {}: base={}, into={}",
                i,
                base.vpcis[i],
                ys[i]
            );
        }

        Ok(())
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "vpci")]
#[pyo3(signature = (close, volume, short_range, long_range, kernel=None))]
pub fn vpci_py<'py>(
    py: Python<'py>,
    close: numpy::PyReadonlyArray1<'py, f64>,
    volume: numpy::PyReadonlyArray1<'py, f64>,
    short_range: usize,
    long_range: usize,
    kernel: Option<&str>,
) -> PyResult<(
    Bound<'py, numpy::PyArray1<f64>>,
    Bound<'py, numpy::PyArray1<f64>>,
)> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let close_slice = close.as_slice()?;
    let volume_slice = volume.as_slice()?;

    if close_slice.len() != volume_slice.len() {
        return Err(PyValueError::new_err(
            "Close and volume arrays must have the same length",
        ));
    }

    let kern = validate_kernel(kernel, false)?;
    let params = VpciParams {
        short_range: Some(short_range),
        long_range: Some(long_range),
    };
    let input = VpciInput::from_slices(close_slice, volume_slice, params);

    let (vpci_vec, vpcis_vec) = py
        .allow_threads(|| vpci_with_kernel(&input, kern).map(|o| (o.vpci, o.vpcis)))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok((vpci_vec.into_pyarray(py), vpcis_vec.into_pyarray(py)))
}

#[cfg(feature = "python")]
#[pyclass(name = "VpciStream")]
pub struct VpciStreamPy {
    stream: VpciStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl VpciStreamPy {
    #[new]
    fn new(short_range: usize, long_range: usize) -> PyResult<Self> {
        let params = VpciParams {
            short_range: Some(short_range),
            long_range: Some(long_range),
        };
        let stream =
            VpciStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(VpciStreamPy { stream })
    }

    fn update(&mut self, close: f64, volume: f64) -> Option<(f64, f64)> {
        self.stream.update(close, volume)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "vpci_batch")]
#[pyo3(signature = (close, volume, short_range_tuple, long_range_tuple, kernel=None))]
pub fn vpci_batch_py<'py>(
    py: Python<'py>,
    close: numpy::PyReadonlyArray1<'py, f64>,
    volume: numpy::PyReadonlyArray1<'py, f64>,
    short_range_tuple: (usize, usize, usize),
    long_range_tuple: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let close_slice = close.as_slice()?;
    let volume_slice = volume.as_slice()?;

    if close_slice.len() != volume_slice.len() {
        return Err(PyValueError::new_err(
            "Close and volume arrays must have the same length",
        ));
    }

    let sweep = VpciBatchRange {
        short_range: short_range_tuple,
        long_range: long_range_tuple,
    };

    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = close_slice.len();
    if rows == 0 || cols == 0 {
        return Err(PyValueError::new_err(
            "no parameter combinations or empty input",
        ));
    }
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow in vpci_batch_py"))?;

    let vpci_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let vpcis_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let vpci_slice = unsafe { vpci_arr.as_slice_mut()? };
    let vpcis_slice = unsafe { vpcis_arr.as_slice_mut()? };

    let kern = validate_kernel(kernel, true)?;

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

            vpci_batch_inner_into(
                close_slice,
                volume_slice,
                &sweep,
                simd,
                true,
                vpci_slice,
                vpcis_slice,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("vpci", vpci_arr.reshape((rows, cols))?)?;
    dict.set_item("vpcis", vpcis_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "short_ranges",
        combos
            .iter()
            .map(|p| p.short_range.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "long_ranges",
        combos
            .iter()
            .map(|p| p.long_range.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;

    Ok(dict)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn vpci_js(
    close: &[f64],
    volume: &[f64],
    short_range: usize,
    long_range: usize,
) -> Result<JsValue, JsValue> {
    let params = VpciParams {
        short_range: Some(short_range),
        long_range: Some(long_range),
    };
    let input = VpciInput::from_slices(close, volume, params);

    let out = vpci(&input).map_err(|e| JsValue::from_str(&e.to_string()))?;
    #[derive(Serialize)]
    struct Out {
        vpci: Vec<f64>,
        vpcis: Vec<f64>,
    }
    serde_wasm_bindgen::to_value(&Out {
        vpci: out.vpci,
        vpcis: out.vpcis,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn vpci_into(
    close_ptr: *const f64,
    volume_ptr: *const f64,
    vpci_ptr: *mut f64,
    vpcis_ptr: *mut f64,
    len: usize,
    short_range: usize,
    long_range: usize,
) -> Result<(), JsValue> {
    if close_ptr.is_null() || volume_ptr.is_null() || vpci_ptr.is_null() || vpcis_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to vpci_into"));
    }

    unsafe {
        let close = core::slice::from_raw_parts(close_ptr, len);
        let volume = core::slice::from_raw_parts(volume_ptr, len);
        let vpci = core::slice::from_raw_parts_mut(vpci_ptr, len);
        let vpcis = core::slice::from_raw_parts_mut(vpcis_ptr, len);

        let params = VpciParams {
            short_range: Some(short_range),
            long_range: Some(long_range),
        };
        let input = VpciInput::from_slices(close, volume, params);

        vpci_into_slice(vpci, vpcis, &input, detect_best_kernel())
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn vpci_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn vpci_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct VpciBatchConfig {
    pub short_range: (usize, usize, usize),
    pub long_range: (usize, usize, usize),
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct VpciBatchJsOutput {
    pub vpci: Vec<f64>,
    pub vpcis: Vec<f64>,
    pub combos: Vec<VpciParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "vpci_batch")]
pub fn vpci_batch_unified_js(
    close: &[f64],
    volume: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let cfg: VpciBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
    let sweep = VpciBatchRange {
        short_range: cfg.short_range,
        long_range: cfg.long_range,
    };
    let output = vpci_batch_inner(close, volume, &sweep, detect_best_kernel(), false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let js_out = VpciBatchJsOutput {
        vpci: output.vpci,
        vpcis: output.vpcis,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    };
    serde_wasm_bindgen::to_value(&js_out)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::cuda_available;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::vpci_wrapper::CudaVpci;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::alma::DeviceArrayF32Py;

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "vpci_cuda_batch_dev")]
#[pyo3(signature = (close_f32, volume_f32, short_range_tuple, long_range_tuple, device_id=0))]
pub fn vpci_cuda_batch_dev_py<'py>(
    py: Python<'py>,
    close_f32: numpy::PyReadonlyArray1<'py, f32>,
    volume_f32: numpy::PyReadonlyArray1<'py, f32>,
    short_range_tuple: (usize, usize, usize),
    long_range_tuple: (usize, usize, usize),
    device_id: usize,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::IntoPyArray;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let c = close_f32.as_slice()?;
    let v = volume_f32.as_slice()?;
    if c.len() != v.len() {
        return Err(PyValueError::new_err("length mismatch"));
    }
    let sweep = VpciBatchRange {
        short_range: short_range_tuple,
        long_range: long_range_tuple,
    };
    let (pair, combos, ctx, dev_id_u32) = py.allow_threads(|| {
        let cuda = CudaVpci::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx = cuda.context_arc();
        let dev_id_u32 = cuda.device_id();
        cuda.vpci_batch_dev(c, v, &sweep)
            .map(|(pair, combos)| (pair, combos, ctx, dev_id_u32))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    let dict = PyDict::new(py);
    dict.set_item(
        "vpci",
        Py::new(
            py,
            DeviceArrayF32Py {
                inner: pair.a,
                _ctx: Some(ctx.clone()),
                device_id: Some(dev_id_u32),
            },
        )?,
    )?;
    dict.set_item(
        "vpcis",
        Py::new(
            py,
            DeviceArrayF32Py {
                inner: pair.b,
                _ctx: Some(ctx),
                device_id: Some(dev_id_u32),
            },
        )?,
    )?;
    dict.set_item("rows", combos.len())?;
    dict.set_item("cols", c.len())?;
    dict.set_item(
        "short_ranges",
        combos
            .iter()
            .map(|p| p.short_range.unwrap_or(5) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "long_ranges",
        combos
            .iter()
            .map(|p| p.long_range.unwrap_or(25) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    Ok(dict)
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "vpci_cuda_many_series_one_param_dev")]
#[pyo3(signature = (close_tm_f32, volume_tm_f32, short_range, long_range, device_id=0))]
pub fn vpci_cuda_many_series_one_param_dev_py<'py>(
    py: Python<'py>,
    close_tm_f32: numpy::PyReadonlyArray2<'py, f32>,
    volume_tm_f32: numpy::PyReadonlyArray2<'py, f32>,
    short_range: usize,
    long_range: usize,
    device_id: usize,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::PyUntypedArrayMethods;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let shape = close_tm_f32.shape();
    if shape.len() != 2 {
        return Err(PyValueError::new_err("expected 2D array for close"));
    }
    if volume_tm_f32.shape() != shape {
        return Err(PyValueError::new_err(
            "input arrays must share the same shape",
        ));
    }
    let rows = shape[0];
    let cols = shape[1];
    let c = close_tm_f32.as_slice()?;
    let v = volume_tm_f32.as_slice()?;
    let params = VpciParams {
        short_range: Some(short_range),
        long_range: Some(long_range),
    };
    let (pair, ctx, dev_id_u32) = py.allow_threads(|| {
        let cuda = CudaVpci::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx = cuda.context_arc();
        let dev_id_u32 = cuda.device_id();
        cuda.vpci_many_series_one_param_time_major_dev(c, v, cols, rows, &params)
            .map(|pair| (pair, ctx, dev_id_u32))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    let dict = PyDict::new(py);
    dict.set_item(
        "vpci",
        Py::new(
            py,
            DeviceArrayF32Py {
                inner: pair.a,
                _ctx: Some(ctx.clone()),
                device_id: Some(dev_id_u32),
            },
        )?,
    )?;
    dict.set_item(
        "vpcis",
        Py::new(
            py,
            DeviceArrayF32Py {
                inner: pair.b,
                _ctx: Some(ctx),
                device_id: Some(dev_id_u32),
            },
        )?,
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    dict.set_item("short_range", short_range)?;
    dict.set_item("long_range", long_range)?;
    Ok(dict)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn vpci_batch_into(
    close_ptr: *const f64,
    volume_ptr: *const f64,
    vpci_ptr: *mut f64,
    vpcis_ptr: *mut f64,
    len: usize,
    short_start: usize,
    short_end: usize,
    short_step: usize,
    long_start: usize,
    long_end: usize,
    long_step: usize,
) -> Result<usize, JsValue> {
    if close_ptr.is_null() || volume_ptr.is_null() || vpci_ptr.is_null() || vpcis_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to vpci_batch_into"));
    }

    unsafe {
        let close = std::slice::from_raw_parts(close_ptr, len);
        let volume = std::slice::from_raw_parts(volume_ptr, len);

        let sweep = VpciBatchRange {
            short_range: (short_start, short_end, short_step),
            long_range: (long_start, long_end, long_step),
        };

        let combos = expand_grid_vpci(&sweep);
        let rows = combos.len();
        if rows == 0 {
            return Err(JsValue::from_str(
                "no parameter combinations for vpci_batch_into",
            ));
        }
        let total_len = rows
            .checked_mul(len)
            .ok_or_else(|| JsValue::from_str("rows*len overflow in vpci_batch_into"))?;

        let need_temp = close_ptr == vpci_ptr as *const f64
            || close_ptr == vpcis_ptr as *const f64
            || volume_ptr == vpci_ptr as *const f64
            || volume_ptr == vpcis_ptr as *const f64;

        if need_temp {
            let output = vpci_batch_inner(close, volume, &sweep, detect_best_kernel(), false)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            let vpci_out = std::slice::from_raw_parts_mut(vpci_ptr, total_len);
            let vpcis_out = std::slice::from_raw_parts_mut(vpcis_ptr, total_len);
            vpci_out.copy_from_slice(&output.vpci);
            vpcis_out.copy_from_slice(&output.vpcis);
        } else {
            let vpci_out = std::slice::from_raw_parts_mut(vpci_ptr, total_len);
            let vpcis_out = std::slice::from_raw_parts_mut(vpcis_ptr, total_len);

            vpci_batch_inner_into(
                close,
                volume,
                &sweep,
                detect_best_kernel(),
                false,
                vpci_out,
                vpcis_out,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(rows)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
#[deprecated(
    since = "1.0.0",
    note = "For weight reuse patterns, use the fast/unsafe API with persistent buffers or VpciStream"
)]
pub struct VpciContext {
    short_range: usize,
    long_range: usize,
    kernel: Kernel,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
impl VpciContext {
    #[wasm_bindgen(constructor)]
    pub fn new(short_range: usize, long_range: usize) -> Result<VpciContext, JsValue> {
        if short_range == 0 || long_range == 0 || short_range > long_range {
            return Err(JsValue::from_str("Invalid range parameters"));
        }

        Ok(VpciContext {
            short_range,
            long_range,
            kernel: detect_best_kernel(),
        })
    }

    pub fn update_into(
        &self,
        close_ptr: *const f64,
        volume_ptr: *const f64,
        vpci_ptr: *mut f64,
        vpcis_ptr: *mut f64,
        len: usize,
    ) -> Result<(), JsValue> {
        if close_ptr.is_null() || volume_ptr.is_null() || vpci_ptr.is_null() || vpcis_ptr.is_null()
        {
            return Err(JsValue::from_str("null pointer passed to update_into"));
        }

        if len < self.long_range {
            return Err(JsValue::from_str("Data length less than long range"));
        }

        unsafe {
            let close = std::slice::from_raw_parts(close_ptr, len);
            let volume = std::slice::from_raw_parts(volume_ptr, len);

            let params = VpciParams {
                short_range: Some(self.short_range),
                long_range: Some(self.long_range),
            };
            let input = VpciInput::from_slices(close, volume, params);

            let need_temp = close_ptr == vpci_ptr as *const f64
                || close_ptr == vpcis_ptr as *const f64
                || volume_ptr == vpci_ptr as *const f64
                || volume_ptr == vpcis_ptr as *const f64;

            if need_temp {
                let mut temp_vpci = vec![0.0; len];
                let mut temp_vpcis = vec![0.0; len];
                vpci_into_slice(&mut temp_vpci, &mut temp_vpcis, &input, self.kernel)
                    .map_err(|e| JsValue::from_str(&e.to_string()))?;

                let vpci_out = std::slice::from_raw_parts_mut(vpci_ptr, len);
                let vpcis_out = std::slice::from_raw_parts_mut(vpcis_ptr, len);
                vpci_out.copy_from_slice(&temp_vpci);
                vpcis_out.copy_from_slice(&temp_vpcis);
            } else {
                let vpci_out = std::slice::from_raw_parts_mut(vpci_ptr, len);
                let vpcis_out = std::slice::from_raw_parts_mut(vpcis_ptr, len);
                vpci_into_slice(vpci_out, vpcis_out, &input, self.kernel)
                    .map_err(|e| JsValue::from_str(&e.to_string()))?;
            }
        }

        Ok(())
    }

    pub fn get_warmup_period(&self) -> usize {
        self.long_range - 1
    }
}
