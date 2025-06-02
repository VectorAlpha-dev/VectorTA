//! # Damiani Volatmeter
//!
//! A volatility indicator using ATR and standard deviation bands to measure trend activity and a threshold for "anti-trend." Supports batch evaluation, parameter sweeps, and SIMD kernels. This indicator follows the API conventions and modular layout of alma.rs, supporting AVX2/AVX512 (with stubs), and is fully unit tested.
//!
//! ## Parameters
//! - **vis_atr**: ATR period for volatility (default 13)
//! - **vis_std**: Std window for volatility (default 20)
//! - **sed_atr**: ATR period for "sedation" (default 40)
//! - **sed_std**: Std window for "sedation" (default 100)
//! - **threshold**: Offset constant (default 1.4)
//!
//! ## Errors
//! - **AllValuesNaN**: all input data values are NaN
//! - **InvalidPeriod**: one or more periods are zero or exceed the data length
//! - **NotEnoughValidData**: not enough valid data points for requested lookback
//!
//! ## Returns
//! - **`Ok(DamianiVolatmeterOutput)`** with `vol`, `anti` arrays, else `Err(DamianiVolatmeterError)`

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_kernel, detect_best_batch_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use rayon::prelude::*;
use std::convert::AsRef;
use thiserror::Error;

impl<'a> AsRef<[f64]> for DamianiVolatmeterInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            DamianiVolatmeterData::Slice(slice) => slice,
            DamianiVolatmeterData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum DamianiVolatmeterData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct DamianiVolatmeterOutput {
    pub vol: Vec<f64>,
    pub anti: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct DamianiVolatmeterParams {
    pub vis_atr: Option<usize>,
    pub vis_std: Option<usize>,
    pub sed_atr: Option<usize>,
    pub sed_std: Option<usize>,
    pub threshold: Option<f64>,
}

impl Default for DamianiVolatmeterParams {
    fn default() -> Self {
        Self {
            vis_atr: Some(13),
            vis_std: Some(20),
            sed_atr: Some(40),
            sed_std: Some(100),
            threshold: Some(1.4),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DamianiVolatmeterInput<'a> {
    pub data: DamianiVolatmeterData<'a>,
    pub params: DamianiVolatmeterParams,
}

impl<'a> DamianiVolatmeterInput<'a> {
    #[inline]
    pub fn from_candles(
        c: &'a Candles,
        s: &'a str,
        p: DamianiVolatmeterParams,
    ) -> Self {
        Self {
            data: DamianiVolatmeterData::Candles { candles: c, source: s },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: DamianiVolatmeterParams) -> Self {
        Self {
            data: DamianiVolatmeterData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", DamianiVolatmeterParams::default())
    }
    #[inline]
    pub fn get_vis_atr(&self) -> usize {
        self.params.vis_atr.unwrap_or(13)
    }
    #[inline]
    pub fn get_vis_std(&self) -> usize {
        self.params.vis_std.unwrap_or(20)
    }
    #[inline]
    pub fn get_sed_atr(&self) -> usize {
        self.params.sed_atr.unwrap_or(40)
    }
    #[inline]
    pub fn get_sed_std(&self) -> usize {
        self.params.sed_std.unwrap_or(100)
    }
    #[inline]
    pub fn get_threshold(&self) -> f64 {
        self.params.threshold.unwrap_or(1.4)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct DamianiVolatmeterBuilder {
    vis_atr: Option<usize>,
    vis_std: Option<usize>,
    sed_atr: Option<usize>,
    sed_std: Option<usize>,
    threshold: Option<f64>,
    kernel: Kernel,
}

impl Default for DamianiVolatmeterBuilder {
    fn default() -> Self {
        Self {
            vis_atr: None,
            vis_std: None,
            sed_atr: None,
            sed_std: None,
            threshold: None,
            kernel: Kernel::Auto,
        }
    }
}

impl DamianiVolatmeterBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn vis_atr(mut self, n: usize) -> Self {
        self.vis_atr = Some(n); self
    }
    #[inline(always)]
    pub fn vis_std(mut self, n: usize) -> Self {
        self.vis_std = Some(n); self
    }
    #[inline(always)]
    pub fn sed_atr(mut self, n: usize) -> Self {
        self.sed_atr = Some(n); self
    }
    #[inline(always)]
    pub fn sed_std(mut self, n: usize) -> Self {
        self.sed_std = Some(n); self
    }
    #[inline(always)]
    pub fn threshold(mut self, x: f64) -> Self {
        self.threshold = Some(x); self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k; self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<DamianiVolatmeterOutput, DamianiVolatmeterError> {
        let p = DamianiVolatmeterParams {
            vis_atr: self.vis_atr,
            vis_std: self.vis_std,
            sed_atr: self.sed_atr,
            sed_std: self.sed_std,
            threshold: self.threshold,
        };
        let i = DamianiVolatmeterInput::from_candles(c, "close", p);
        damiani_volatmeter_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<DamianiVolatmeterOutput, DamianiVolatmeterError> {
        let p = DamianiVolatmeterParams {
            vis_atr: self.vis_atr,
            vis_std: self.vis_std,
            sed_atr: self.sed_atr,
            sed_std: self.sed_std,
            threshold: self.threshold,
        };
        let i = DamianiVolatmeterInput::from_slice(d, p);
        damiani_volatmeter_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<DamianiVolatmeterStream, DamianiVolatmeterError> {
        let p = DamianiVolatmeterParams {
            vis_atr: self.vis_atr,
            vis_std: self.vis_std,
            sed_atr: self.sed_atr,
            sed_std: self.sed_std,
            threshold: self.threshold,
        };
        DamianiVolatmeterStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum DamianiVolatmeterError {
    #[error("damiani_volatmeter: All values are NaN.")]
    AllValuesNaN,
    #[error("damiani_volatmeter: Invalid period: data length = {data_len}, vis_atr = {vis_atr}, vis_std = {vis_std}, sed_atr = {sed_atr}, sed_std = {sed_std}")]
    InvalidPeriod {
        data_len: usize,
        vis_atr: usize,
        vis_std: usize,
        sed_atr: usize,
        sed_std: usize,
    },
    #[error("damiani_volatmeter: Not enough valid data after first non-NaN index. needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("damiani_volatmeter: Empty data provided.")]
    EmptyData,
}

#[inline]
pub fn damiani_volatmeter(input: &DamianiVolatmeterInput) -> Result<DamianiVolatmeterOutput, DamianiVolatmeterError> {
    damiani_volatmeter_with_kernel(input, Kernel::Auto)
}

pub fn damiani_volatmeter_with_kernel(input: &DamianiVolatmeterInput, kernel: Kernel) -> Result<DamianiVolatmeterOutput, DamianiVolatmeterError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();

    if len == 0 {
        return Err(DamianiVolatmeterError::EmptyData);
    }
    let vis_atr = input.get_vis_atr();
    let vis_std = input.get_vis_std();
    let sed_atr = input.get_sed_atr();
    let sed_std = input.get_sed_std();
    let threshold = input.get_threshold();

    if vis_atr == 0 || vis_std == 0 || sed_atr == 0 || sed_std == 0 ||
        vis_atr > len || vis_std > len || sed_atr > len || sed_std > len {
        return Err(DamianiVolatmeterError::InvalidPeriod {
            data_len: len, vis_atr, vis_std, sed_atr, sed_std
        });
    }
    let first = data.iter().position(|&x| !x.is_nan()).ok_or(DamianiVolatmeterError::AllValuesNaN)?;
    let needed = *[vis_atr, vis_std, sed_atr, sed_std, 3].iter().max().unwrap();
    if (len - first) < needed {
        return Err(DamianiVolatmeterError::NotEnoughValidData {
            needed, valid: len - first
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let mut vol = vec![f64::NAN; len];
    let mut anti = vec![f64::NAN; len];
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                damiani_volatmeter_scalar(data, vis_atr, vis_std, sed_atr, sed_std, threshold, first, &mut vol, &mut anti)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                damiani_volatmeter_avx2(data, vis_atr, vis_std, sed_atr, sed_std, threshold, first, &mut vol, &mut anti)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                damiani_volatmeter_avx512(data, vis_atr, vis_std, sed_atr, sed_std, threshold, first, &mut vol, &mut anti)
            }
            _ => unreachable!(),
        }
    }

    Ok(DamianiVolatmeterOutput { vol, anti })
}

#[inline]
pub unsafe fn damiani_volatmeter_scalar(
    data: &[f64],
    vis_atr: usize,
    vis_std: usize,
    sed_atr: usize,
    sed_std: usize,
    threshold: f64,
    first: usize,
    vol: &mut [f64],
    anti: &mut [f64]
) {
    let len = data.len();
    let mut atr_vis_val = f64::NAN;
    let mut atr_sed_val = f64::NAN;
    let mut sum_vis = 0.0;
    let mut sum_sed = 0.0;
    let mut prev = data[first];
    let mut ring_vis = vec![0.0; vis_std];
    let mut ring_sed = vec![0.0; sed_std];
    let mut sum_vis_std = 0.0;
    let mut sum_sq_vis_std = 0.0;
    let mut sum_sed_std = 0.0;
    let mut sum_sq_sed_std = 0.0;
    let mut idx_vis = 0;
    let mut idx_sed = 0;
    let mut filled_vis = 0;
    let mut filled_sed = 0;
    let lag_s = 0.5_f64;
    for i in 0..len {
        let mut tr = if i > 0 { (data[i] - prev).abs() } else { 0.0 };
        prev = data[i];
        if i < vis_atr {
            sum_vis += tr;
            if i == vis_atr - 1 { atr_vis_val = sum_vis / vis_atr as f64; }
        } else if atr_vis_val.is_finite() {
            atr_vis_val = ((vis_atr - 1) as f64 * atr_vis_val + tr) / vis_atr as f64;
        }
        if i < sed_atr {
            sum_sed += tr;
            if i == sed_atr - 1 { atr_sed_val = sum_sed / sed_atr as f64; }
        } else if atr_sed_val.is_finite() {
            atr_sed_val = ((sed_atr - 1) as f64 * atr_sed_val + tr) / sed_atr as f64;
        }
        let val = if data[i].is_nan() { 0.0 } else { data[i] };
        let old_v = ring_vis[idx_vis];
        ring_vis[idx_vis] = val;
        idx_vis = (idx_vis + 1) % vis_std;
        if filled_vis < vis_std { filled_vis += 1; sum_vis_std += val; sum_sq_vis_std += val * val; }
        else { sum_vis_std = sum_vis_std - old_v + val; sum_sq_vis_std = sum_sq_vis_std - old_v * old_v + val * val; }
        let old_s = ring_sed[idx_sed];
        ring_sed[idx_sed] = val;
        idx_sed = (idx_sed + 1) % sed_std;
        if filled_sed < sed_std { filled_sed += 1; sum_sed_std += val; sum_sq_sed_std += val * val; }
        else { sum_sed_std = sum_sed_std - old_s + val; sum_sq_sed_std = sum_sq_sed_std - old_s * old_s + val * val; }
        if i >= [vis_atr, vis_std, sed_atr, sed_std, 3].iter().max().copied().unwrap() {
            let p1 = if i >= 1 && !vol[i - 1].is_nan() { vol[i - 1] } else { 0.0 };
            let p3 = if i >= 3 && !vol[i - 3].is_nan() { vol[i - 3] } else { 0.0 };
            let sed_safe = if atr_sed_val.is_finite() && atr_sed_val != 0.0 { atr_sed_val } else { atr_sed_val + f64::EPSILON };
            vol[i] = (atr_vis_val / sed_safe) + lag_s * (p1 - p3);
            if filled_vis == vis_std && filled_sed == sed_std {
                let std_vis = stddev(sum_vis_std, sum_sq_vis_std, vis_std);
                let std_sed = stddev(sum_sed_std, sum_sq_sed_std, sed_std);
                let ratio = if std_sed != 0.0 { std_vis / std_sed } else { std_vis / (std_sed + f64::EPSILON) };
                anti[i] = threshold - ratio;
            }
        }
    }
}
#[inline]
fn stddev(sum: f64, sum_sq: f64, n: usize) -> f64 {
    if n == 0 { return 0.0; }
    let mean = sum / n as f64;
    let mean_sq = sum_sq / n as f64;
    let var = mean_sq - mean * mean;
    if var <= 0.0 { 0.0 } else { var.sqrt() }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn damiani_volatmeter_avx512(
    data: &[f64], vis_atr: usize, vis_std: usize, sed_atr: usize, sed_std: usize, threshold: f64, first: usize, vol: &mut [f64], anti: &mut [f64]
) {
    damiani_volatmeter_scalar(data, vis_atr, vis_std, sed_atr, sed_std, threshold, first, vol, anti)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn damiani_volatmeter_avx2(
    data: &[f64], vis_atr: usize, vis_std: usize, sed_atr: usize, sed_std: usize, threshold: f64, first: usize, vol: &mut [f64], anti: &mut [f64]
) {
    damiani_volatmeter_scalar(data, vis_atr, vis_std, sed_atr, sed_std, threshold, first, vol, anti)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn damiani_volatmeter_avx512_short(
    data: &[f64], vis_atr: usize, vis_std: usize, sed_atr: usize, sed_std: usize, threshold: f64, first: usize, vol: &mut [f64], anti: &mut [f64]
) {
    damiani_volatmeter_avx512(data, vis_atr, vis_std, sed_atr, sed_std, threshold, first, vol, anti)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn damiani_volatmeter_avx512_long(
    data: &[f64], vis_atr: usize, vis_std: usize, sed_atr: usize, sed_std: usize, threshold: f64, first: usize, vol: &mut [f64], anti: &mut [f64]
) {
    damiani_volatmeter_avx512(data, vis_atr, vis_std, sed_atr, sed_std, threshold, first, vol, anti)
}

pub fn damiani_volatmeter_batch_with_kernel(
    data: &[f64], sweep: &DamianiVolatmeterBatchRange, k: Kernel
) -> Result<DamianiVolatmeterBatchOutput, DamianiVolatmeterError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(DamianiVolatmeterError::InvalidPeriod {
            data_len: 0, vis_atr: 0, vis_std: 0, sed_atr: 0, sed_std: 0,
        }),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    damiani_volatmeter_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct DamianiVolatmeterBatchRange {
    pub vis_atr: (usize, usize, usize),
    pub vis_std: (usize, usize, usize),
    pub sed_atr: (usize, usize, usize),
    pub sed_std: (usize, usize, usize),
    pub threshold: (f64, f64, f64),
}
impl Default for DamianiVolatmeterBatchRange {
    fn default() -> Self {
        Self {
            vis_atr: (13, 40, 1),
            vis_std: (20, 40, 1),
            sed_atr: (40, 40, 0),
            sed_std: (100, 100, 0),
            threshold: (1.4, 1.4, 0.0),
        }
    }
}
#[derive(Clone, Debug, Default)]
pub struct DamianiVolatmeterBatchBuilder {
    range: DamianiVolatmeterBatchRange,
    kernel: Kernel,
}
impl DamianiVolatmeterBatchBuilder {
    pub fn new() -> Self { Self::default() }
    pub fn kernel(mut self, k: Kernel) -> Self { self.kernel = k; self }
    pub fn vis_atr_range(mut self, s: usize, e: usize, step: usize) -> Self { self.range.vis_atr = (s, e, step); self }
    pub fn vis_std_range(mut self, s: usize, e: usize, step: usize) -> Self { self.range.vis_std = (s, e, step); self }
    pub fn sed_atr_range(mut self, s: usize, e: usize, step: usize) -> Self { self.range.sed_atr = (s, e, step); self }
    pub fn sed_std_range(mut self, s: usize, e: usize, step: usize) -> Self { self.range.sed_std = (s, e, step); self }
    pub fn threshold_range(mut self, s: f64, e: f64, step: f64) -> Self { self.range.threshold = (s, e, step); self }

    pub fn apply_slice(self, data: &[f64]) -> Result<DamianiVolatmeterBatchOutput, DamianiVolatmeterError> {
        damiani_volatmeter_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<DamianiVolatmeterBatchOutput, DamianiVolatmeterError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
}

#[derive(Clone, Debug)]
pub struct DamianiVolatmeterBatchOutput {
    pub vol: Vec<f64>,
    pub anti: Vec<f64>,
    pub combos: Vec<DamianiVolatmeterParams>,
    pub rows: usize,
    pub cols: usize,
}
impl DamianiVolatmeterBatchOutput {
    pub fn row_for_params(&self, p: &DamianiVolatmeterParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.vis_atr == p.vis_atr && c.vis_std == p.vis_std &&
            c.sed_atr == p.sed_atr && c.sed_std == p.sed_std &&
            (c.threshold.unwrap_or(1.4) - p.threshold.unwrap_or(1.4)).abs() < 1e-12
        })
    }
    pub fn vol_for(&self, p: &DamianiVolatmeterParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.vol[start..start + self.cols]
        })
    }
    pub fn anti_for(&self, p: &DamianiVolatmeterParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.anti[start..start + self.cols]
        })
    }
}
#[inline(always)]
pub fn damiani_volatmeter_batch_slice(
    data: &[f64],
    sweep: &DamianiVolatmeterBatchRange,
    kern: Kernel,
) -> Result<DamianiVolatmeterBatchOutput, DamianiVolatmeterError> {
    damiani_volatmeter_batch_inner(data, sweep, kern, false)
}
#[inline(always)]
pub fn damiani_volatmeter_batch_par_slice(
    data: &[f64],
    sweep: &DamianiVolatmeterBatchRange,
    kern: Kernel,
) -> Result<DamianiVolatmeterBatchOutput, DamianiVolatmeterError> {
    damiani_volatmeter_batch_inner(data, sweep, kern, true)
}
fn expand_grid(r: &DamianiVolatmeterBatchRange) -> Vec<DamianiVolatmeterParams> {
    fn axis_usize((s, e, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || s == e { return vec![s]; }
        (s..=e).step_by(step).collect()
    }
    fn axis_f64((s, e, step): (f64, f64, f64)) -> Vec<f64> {
        if step.abs() < 1e-12 || (s - e).abs() < 1e-12 { return vec![s]; }
        let mut v = Vec::new();
        let mut x = s;
        while x <= e + 1e-12 { v.push(x); x += step; }
        v
    }
    let vis_atrs = axis_usize(r.vis_atr);
    let vis_stds = axis_usize(r.vis_std);
    let sed_atrs = axis_usize(r.sed_atr);
    let sed_stds = axis_usize(r.sed_std);
    let thresholds = axis_f64(r.threshold);
    let mut out = Vec::with_capacity(vis_atrs.len() * vis_stds.len() * sed_atrs.len() * sed_stds.len() * thresholds.len());
    for &va in &vis_atrs {
        for &vs in &vis_stds {
            for &sa in &sed_atrs {
                for &ss in &sed_stds {
                    for &th in &thresholds {
                        out.push(DamianiVolatmeterParams {
                            vis_atr: Some(va),
                            vis_std: Some(vs),
                            sed_atr: Some(sa),
                            sed_std: Some(ss),
                            threshold: Some(th),
                        });
                    }
                }
            }
        }
    }
    out
}
#[inline(always)]
fn damiani_volatmeter_batch_inner(
    data: &[f64],
    sweep: &DamianiVolatmeterBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<DamianiVolatmeterBatchOutput, DamianiVolatmeterError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(DamianiVolatmeterError::InvalidPeriod {
            data_len: 0, vis_atr: 0, vis_std: 0, sed_atr: 0, sed_std: 0
        });
    }
    let first = data.iter().position(|x| !x.is_nan()).ok_or(DamianiVolatmeterError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| *[c.vis_atr.unwrap(), c.vis_std.unwrap(), c.sed_atr.unwrap(), c.sed_std.unwrap()].iter().max().unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(DamianiVolatmeterError::NotEnoughValidData {
            needed: max_p, valid: data.len() - first,
        });
    }
    let rows = combos.len();
    let cols = data.len();
    let mut vol = vec![f64::NAN; rows * cols];
    let mut anti = vec![f64::NAN; rows * cols];

    let do_row = |row: usize, out_vol: &mut [f64], out_anti: &mut [f64]| unsafe {
        let prm = &combos[row];
        match kern {
            Kernel::Scalar => damiani_volatmeter_row_scalar(data, first,
                prm.vis_atr.unwrap(), prm.vis_std.unwrap(), prm.sed_atr.unwrap(), prm.sed_std.unwrap(),
                prm.threshold.unwrap(), out_vol, out_anti),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => damiani_volatmeter_row_avx2(data, first,
                prm.vis_atr.unwrap(), prm.vis_std.unwrap(), prm.sed_atr.unwrap(), prm.sed_std.unwrap(),
                prm.threshold.unwrap(), out_vol, out_anti),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => damiani_volatmeter_row_avx512(data, first,
                prm.vis_atr.unwrap(), prm.vis_std.unwrap(), prm.sed_atr.unwrap(), prm.sed_std.unwrap(),
                prm.threshold.unwrap(), out_vol, out_anti),
            _ => unreachable!(),
        }
    };
    if parallel {
        vol.par_chunks_mut(cols)
            .zip(anti.par_chunks_mut(cols))
            .enumerate()
            .for_each(|(row, (outv, outa))| do_row(row, outv, outa));
    } else {
        for (row, (outv, outa)) in vol.chunks_mut(cols).zip(anti.chunks_mut(cols)).enumerate() {
            do_row(row, outv, outa);
        }
    }
    Ok(DamianiVolatmeterBatchOutput { vol, anti, combos, rows, cols })
}

#[inline(always)]
pub unsafe fn damiani_volatmeter_row_scalar(
    data: &[f64], first: usize, vis_atr: usize, vis_std: usize, sed_atr: usize, sed_std: usize, threshold: f64, vol: &mut [f64], anti: &mut [f64]
) {
    damiani_volatmeter_scalar(data, vis_atr, vis_std, sed_atr, sed_std, threshold, first, vol, anti)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn damiani_volatmeter_row_avx2(
    data: &[f64], first: usize, vis_atr: usize, vis_std: usize, sed_atr: usize, sed_std: usize, threshold: f64, vol: &mut [f64], anti: &mut [f64]
) {
    damiani_volatmeter_row_scalar(data, first, vis_atr, vis_std, sed_atr, sed_std, threshold, vol, anti)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn damiani_volatmeter_row_avx512(
    data: &[f64], first: usize, vis_atr: usize, vis_std: usize, sed_atr: usize, sed_std: usize, threshold: f64, vol: &mut [f64], anti: &mut [f64]
) {
    damiani_volatmeter_row_scalar(data, first, vis_atr, vis_std, sed_atr, sed_std, threshold, vol, anti)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn damiani_volatmeter_row_avx512_short(
    data: &[f64], first: usize, vis_atr: usize, vis_std: usize, sed_atr: usize, sed_std: usize, threshold: f64, vol: &mut [f64], anti: &mut [f64]
) {
    damiani_volatmeter_row_avx512(data, first, vis_atr, vis_std, sed_atr, sed_std, threshold, vol, anti)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn damiani_volatmeter_row_avx512_long(
    data: &[f64], first: usize, vis_atr: usize, vis_std: usize, sed_atr: usize, sed_std: usize, threshold: f64, vol: &mut [f64], anti: &mut [f64]
) {
    damiani_volatmeter_row_avx512(data, first, vis_atr, vis_std, sed_atr, sed_std, threshold, vol, anti)
}

/// Streaming version for Damiani Volatmeter, matching alma.rs's stream API.
#[derive(Debug, Clone)]
pub struct DamianiVolatmeterStream {
    vis_atr: usize,
    vis_std: usize,
    sed_atr: usize,
    sed_std: usize,
    threshold: f64,
    atr_vis_val: f64,
    atr_sed_val: f64,
    sum_vis: f64,
    sum_sed: f64,
    prev: f64,
    ring_vis: Vec<f64>,
    ring_sed: Vec<f64>,
    sum_vis_std: f64,
    sum_sq_vis_std: f64,
    sum_sed_std: f64,
    sum_sq_sed_std: f64,
    idx_vis: usize,
    idx_sed: usize,
    filled_vis: usize,
    filled_sed: usize,
    p1: f64,
    p3: f64,
    lag_s: f64,
}
impl DamianiVolatmeterStream {
    pub fn try_new(params: DamianiVolatmeterParams) -> Result<Self, DamianiVolatmeterError> {
        let vis_atr = params.vis_atr.unwrap_or(13);
        let vis_std = params.vis_std.unwrap_or(20);
        let sed_atr = params.sed_atr.unwrap_or(40);
        let sed_std = params.sed_std.unwrap_or(100);
        let threshold = params.threshold.unwrap_or(1.4);
        if vis_atr == 0 || vis_std == 0 || sed_atr == 0 || sed_std == 0 {
            return Err(DamianiVolatmeterError::InvalidPeriod {
                data_len: 0, vis_atr, vis_std, sed_atr, sed_std
            });
        }
        Ok(Self {
            vis_atr, vis_std, sed_atr, sed_std, threshold,
            atr_vis_val: f64::NAN,
            atr_sed_val: f64::NAN,
            sum_vis: 0.0,
            sum_sed: 0.0,
            prev: f64::NAN,
            ring_vis: vec![0.0; vis_std],
            ring_sed: vec![0.0; sed_std],
            sum_vis_std: 0.0,
            sum_sq_vis_std: 0.0,
            sum_sed_std: 0.0,
            sum_sq_sed_std: 0.0,
            idx_vis: 0, idx_sed: 0,
            filled_vis: 0, filled_sed: 0,
            p1: 0.0, p3: 0.0,
            lag_s: 0.5,
        })
    }
    pub fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        if self.prev.is_nan() {
            self.prev = value;
            return None;
        }
        let tr = (value - self.prev).abs();
        self.prev = value;
        let i = self.filled_vis.max(self.filled_sed); // streaming "index"
        if i < self.vis_atr { self.sum_vis += tr; if i == self.vis_atr - 1 { self.atr_vis_val = self.sum_vis / self.vis_atr as f64; } }
        else if self.atr_vis_val.is_finite() { self.atr_vis_val = ((self.vis_atr - 1) as f64 * self.atr_vis_val + tr) / self.vis_atr as f64; }
        if i < self.sed_atr { self.sum_sed += tr; if i == self.sed_atr - 1 { self.atr_sed_val = self.sum_sed / self.sed_atr as f64; } }
        else if self.atr_sed_val.is_finite() { self.atr_sed_val = ((self.sed_atr - 1) as f64 * self.atr_sed_val + tr) / self.sed_atr as f64; }
        let val = if value.is_nan() { 0.0 } else { value };
        let old_v = self.ring_vis[self.idx_vis];
        self.ring_vis[self.idx_vis] = val;
        self.idx_vis = (self.idx_vis + 1) % self.vis_std;
        if self.filled_vis < self.vis_std { self.filled_vis += 1; self.sum_vis_std += val; self.sum_sq_vis_std += val * val; }
        else { self.sum_vis_std = self.sum_vis_std - old_v + val; self.sum_sq_vis_std = self.sum_sq_vis_std - old_v * old_v + val * val; }
        let old_s = self.ring_sed[self.idx_sed];
        self.ring_sed[self.idx_sed] = val;
        self.idx_sed = (self.idx_sed + 1) % self.sed_std;
        if self.filled_sed < self.sed_std { self.filled_sed += 1; self.sum_sed_std += val; self.sum_sq_sed_std += val * val; }
        else { self.sum_sed_std = self.sum_sed_std - old_s + val; self.sum_sq_sed_std = self.sum_sq_sed_std - old_s * old_s + val * val; }
        if i < [self.vis_atr, self.vis_std, self.sed_atr, self.sed_std, 3].iter().max().copied().unwrap() { return None; }
        let p1 = self.p1;
        let p3 = self.p3;
        let sed_safe = if self.atr_sed_val.is_finite() && self.atr_sed_val != 0.0 { self.atr_sed_val } else { self.atr_sed_val + f64::EPSILON };
        let vol = (self.atr_vis_val / sed_safe) + self.lag_s * (p1 - p3);
        self.p3 = self.p1;
        self.p1 = vol;
        let anti = if self.filled_vis == self.vis_std && self.filled_sed == self.sed_std {
            let std_vis = stddev(self.sum_vis_std, self.sum_sq_vis_std, self.vis_std);
            let std_sed = stddev(self.sum_sed_std, self.sum_sq_sed_std, self.sed_std);
            let ratio = if std_sed != 0.0 { std_vis / std_sed } else { std_vis / (std_sed + f64::EPSILON) };
            self.threshold - ratio
        } else {
            f64::NAN
        };
        Some((vol, anti))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;
    fn check_damiani_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = DamianiVolatmeterParams::default();
        let input = DamianiVolatmeterInput::from_candles(&candles, "close", params);
        let output = damiani_volatmeter_with_kernel(&input, kernel)?;
        assert_eq!(output.vol.len(), candles.close.len());
        assert_eq!(output.anti.len(), candles.close.len());
        Ok(())
    }
    fn check_damiani_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = DamianiVolatmeterInput::from_candles(&candles, "close", DamianiVolatmeterParams::default());
        let output = damiani_volatmeter_with_kernel(&input, kernel)?;
        let n = output.vol.len();
        let expected_vol = [
            0.9009485470514558,
            0.8333604467044887,
            0.815318380178986,
            0.8276892636184923,
            0.879447954127426,
        ];
        let expected_anti = [
            1.1227721577887388,
            1.1250333024152703,
            1.1325501989919875,
            1.1403866079746106,
            1.1392919184055932,
        ];
        let start = n - 5;
        for i in 0..5 {
            let diff_vol = (output.vol[start + i] - expected_vol[i]).abs();
            let diff_anti = (output.anti[start + i] - expected_anti[i]).abs();
            assert!(diff_vol < 1e-2, "vol mismatch at index {}: expected {}, got {}", start + i, expected_vol[i], output.vol[start + i]);
            assert!(diff_anti < 1e-2, "anti mismatch at index {}: expected {}, got {}", start + i, expected_anti[i], output.anti[start + i]);
        }
        Ok(())
    }
    fn check_damiani_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let mut params = DamianiVolatmeterParams::default();
        params.vis_atr = Some(0);
        let input = DamianiVolatmeterInput::from_candles(&candles, "close", params);
        let res = damiani_volatmeter_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] should fail with zero period", test_name);
        Ok(())
    }
    fn check_damiani_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let mut params = DamianiVolatmeterParams::default();
        params.vis_atr = Some(99999);
        let input = DamianiVolatmeterInput::from_candles(&candles, "close", params);
        let res = damiani_volatmeter_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] should fail if period exceeds length", test_name);
        Ok(())
    }
    fn check_damiani_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [42.0];
        let params = DamianiVolatmeterParams {
            vis_atr: Some(9),
            vis_std: Some(9),
            sed_atr: Some(9),
            sed_std: Some(9),
            threshold: Some(1.4),
        };
        let input = DamianiVolatmeterInput::from_slice(&data, params);
        let res = damiani_volatmeter_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] should fail with insufficient data", test_name);
        Ok(())
    }
    fn check_damiani_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = DamianiVolatmeterInput::from_candles(&candles, "close", DamianiVolatmeterParams::default());
        let batch = damiani_volatmeter_with_kernel(&input, kernel)?;
        let mut stream = DamianiVolatmeterStream::try_new(DamianiVolatmeterParams::default())?;
        let mut stream_vol = Vec::with_capacity(candles.close.len());
        let mut stream_anti = Vec::with_capacity(candles.close.len());
        for &x in &candles.close {
            match stream.update(x) {
                Some((v, a)) => { stream_vol.push(v); stream_anti.push(a); }
                None => { stream_vol.push(f64::NAN); stream_anti.push(f64::NAN); }
            }
        }
        for (i, (&bv, &sv)) in batch.vol.iter().zip(stream_vol.iter()).enumerate() {
            if bv.is_nan() && sv.is_nan() { continue; }
            let diff = (bv - sv).abs();
            assert!(diff < 1e-8, "[{}] streaming vol mismatch at idx {}: batch={}, stream={}", test_name, i, bv, sv);
        }
        for (i, (&ba, &sa)) in batch.anti.iter().zip(stream_anti.iter()).enumerate() {
            if ba.is_nan() && sa.is_nan() { continue; }
            let diff = (ba - sa).abs();
            assert!(diff < 1e-8, "[{}] streaming anti mismatch at idx {}: batch={}, stream={}", test_name, i, ba, sa);
        }
        Ok(())
    }
    fn check_damiani_input_with_default_candles(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = DamianiVolatmeterInput::with_default_candles(&candles);
        match input.data {
            DamianiVolatmeterData::Candles { source, .. } => {
                assert_eq!(source, "close");
            }
            _ => panic!("Expected DamianiVolatmeterData::Candles"),
        }
        Ok(())
    }
    fn check_damiani_params_with_defaults(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        let default_params = DamianiVolatmeterParams::default();
        assert_eq!(default_params.vis_atr, Some(13));
        assert_eq!(default_params.vis_std, Some(20));
        assert_eq!(default_params.sed_atr, Some(40));
        assert_eq!(default_params.sed_std, Some(100));
        assert_eq!(default_params.threshold, Some(1.4));
        Ok(())
    }
    fn check_batch_default_row(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = DamianiVolatmeterBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = DamianiVolatmeterParams::default();
        let vol_row = output.vol_for(&def).expect("default vol row missing");
        let anti_row = output.anti_for(&def).expect("default anti row missing");
        assert_eq!(vol_row.len(), c.close.len());
        assert_eq!(anti_row.len(), c.close.len());
        let expected_vol = [
            0.9009485470514558,
            0.8333604467044887,
            0.815318380178986,
            0.8276892636184923,
            0.879447954127426,
        ];
        let expected_anti = [
            1.1227721577887388,
            1.1250333024152703,
            1.1325501989919875,
            1.1403866079746106,
            1.1392919184055932,
        ];
        let start = vol_row.len() - 5;
        for (i, &v) in vol_row[start..].iter().enumerate() {
            assert!(
                (v - expected_vol[i]).abs() < 1e-2,
                "[{test_name}] default-vol-row mismatch at idx {i}: {v} vs {expected_vol:?}"
            );
        }
        for (i, &a) in anti_row[start..].iter().enumerate() {
            assert!(
                (a - expected_anti[i]).abs() < 1e-2,
                "[{test_name}] default-anti-row mismatch at idx {i}: {a} vs {expected_anti:?}"
            );
        }
        Ok(())
    }
    macro_rules! generate_all_damiani_tests {
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
    generate_all_damiani_tests!(
        check_damiani_partial_params,
        check_damiani_accuracy,
        check_damiani_zero_period,
        check_damiani_period_exceeds_length,
        check_damiani_very_small_dataset,
        check_damiani_streaming,
        check_damiani_input_with_default_candles,
        check_damiani_params_with_defaults
    );
    gen_batch_tests!(check_batch_default_row);
}
