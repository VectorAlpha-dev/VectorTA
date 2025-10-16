//! # Squeeze Momentum Indicator (SMI)
//!
//! Detects market volatility "squeeze" conditions by comparing Bollinger Bands to Keltner Channels,
//! while providing momentum oscillator values with directional signal indicators.
//!
//! ## Parameters
//! - **length_bb**: Bollinger Bands lookback period. Defaults to 20.
//! - **mult_bb**: BB standard deviation multiplier. Defaults to 2.0.
//! - **length_kc**: Keltner Channels lookback period. Defaults to 20.
//! - **mult_kc**: KC ATR multiplier. Defaults to 1.5.
//!
//! ## Returns
//! - **`Ok(SqueezeMomentumOutput)`** containing three `Vec<f64>`:
//!   - `squeeze`: Squeeze state (-1=on, 0=neutral, 1=off)
//!   - `momentum`: Linear regression of price momentum
//!   - `momentum_signal`: Directional acceleration signal (±1=accelerating, ±2=decelerating)
//! - **`Err(SqueezeMomentumError)`** on invalid parameters or insufficient data.
//!
//! ## Developer Notes
//! - Decision: SIMD disabled by default. Hot paths rely on monotonic deques (rolling highs/lows) and O(1) sliding OLS, which are branchy/memory‑bound and did not show clear SIMD wins. Runtime selection short‑circuits to scalar.
//! - Scalar: Optimized classic scalar kernel with O(n) rolling updates (no O(N·W) temporaries). Bench (100k): ~2.31 ms vs ~10.6 ms baseline on native CPU.
//! - Streaming Performance: O(1) per update via rings + accumulators; BB vs KC squeeze classification avoids sqrt (exact). Matches scalar outputs.
//! - Memory: Uses `alloc_with_nan_prefix`/`make_uninit_matrix` and zero‑copy batching.
//! - Batch Optimization (ScalarBatch): Shared precompute across rows (TR, SMA(close), TR SMA, rolling highs/lows, RAW → momentum/signal) keyed by unique `length_kc`/`length_bb`, then per‑row band classification; removes redundant work when sweeping params.

#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;

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
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::mem::MaybeUninit;
use thiserror::Error;

// --- Core Data Types ---

#[derive(Debug, Clone)]
pub enum SqueezeMomentumData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
}

#[derive(Debug, Clone, Copy)]
pub struct SqueezeMomentumParams {
    pub length_bb: Option<usize>,
    pub mult_bb: Option<f64>,
    pub length_kc: Option<usize>,
    pub mult_kc: Option<f64>,
}

impl Default for SqueezeMomentumParams {
    fn default() -> Self {
        Self {
            length_bb: Some(20),
            mult_bb: Some(2.0),
            length_kc: Some(20),
            mult_kc: Some(1.5),
        }
    }
}

impl SqueezeMomentumParams {
    /// Resolve optional parameters to their actual values
    pub fn resolve(&self) -> ResolvedParams {
        ResolvedParams {
            length_bb: self.length_bb.unwrap_or(20),
            mult_bb: self.mult_bb.unwrap_or(2.0),
            length_kc: self.length_kc.unwrap_or(20),
            mult_kc: self.mult_kc.unwrap_or(1.5),
        }
    }
}

/// Resolved parameters with no Option types
#[derive(Debug, Clone, Copy)]
struct ResolvedParams {
    length_bb: usize,
    mult_bb: f64,
    length_kc: usize,
    mult_kc: f64,
}

#[derive(Debug, Clone)]
pub struct SqueezeMomentumInput<'a> {
    pub data: SqueezeMomentumData<'a>,
    pub params: SqueezeMomentumParams,
}

impl<'a> SqueezeMomentumInput<'a> {
    #[inline(always)]
    pub fn from_candles(c: &'a Candles, params: SqueezeMomentumParams) -> Self {
        Self {
            data: SqueezeMomentumData::Candles { candles: c },
            params,
        }
    }
    #[inline(always)]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: SqueezeMomentumParams,
    ) -> Self {
        Self {
            data: SqueezeMomentumData::Slices { high, low, close },
            params,
        }
    }
    #[inline(always)]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, SqueezeMomentumParams::default())
    }
}

#[derive(Debug, Clone)]
pub struct SqueezeMomentumOutput {
    pub squeeze: Vec<f64>,
    pub momentum: Vec<f64>,
    pub momentum_signal: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct SqueezeMomentumBuilder {
    length_bb: Option<usize>,
    mult_bb: Option<f64>,
    length_kc: Option<usize>,
    mult_kc: Option<f64>,
    kernel: Kernel,
}

impl Default for SqueezeMomentumBuilder {
    fn default() -> Self {
        Self {
            length_bb: None,
            mult_bb: None,
            length_kc: None,
            mult_kc: None,
            kernel: Kernel::Auto,
        }
    }
}

impl SqueezeMomentumBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn length_bb(mut self, n: usize) -> Self {
        self.length_bb = Some(n);
        self
    }
    #[inline(always)]
    pub fn mult_bb(mut self, x: f64) -> Self {
        self.mult_bb = Some(x);
        self
    }
    #[inline(always)]
    pub fn length_kc(mut self, n: usize) -> Self {
        self.length_kc = Some(n);
        self
    }
    #[inline(always)]
    pub fn mult_kc(mut self, x: f64) -> Self {
        self.mult_kc = Some(x);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<SqueezeMomentumOutput, SqueezeMomentumError> {
        let p = SqueezeMomentumParams {
            length_bb: self.length_bb,
            mult_bb: self.mult_bb,
            length_kc: self.length_kc,
            mult_kc: self.mult_kc,
        };
        let i = SqueezeMomentumInput::from_candles(c, p);
        squeeze_momentum_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<SqueezeMomentumOutput, SqueezeMomentumError> {
        let p = SqueezeMomentumParams {
            length_bb: self.length_bb,
            mult_bb: self.mult_bb,
            length_kc: self.length_kc,
            mult_kc: self.mult_kc,
        };
        let i = SqueezeMomentumInput::from_slices(high, low, close, p);
        squeeze_momentum_with_kernel(&i, self.kernel)
    }
}

#[derive(Debug, Error)]
pub enum SqueezeMomentumError {
    #[error("smi: Empty data provided for Squeeze Momentum.")]
    EmptyData,
    #[error("smi: Invalid length parameter: length = {length}, data length = {data_len}")]
    InvalidLength { length: usize, data_len: usize },
    #[error("smi: High/low/close arrays have inconsistent lengths.")]
    InconsistentDataLength,
    #[error("smi: All values are NaN.")]
    AllValuesNaN,
    #[error("smi: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

// --- Main Kernel Selection ---

#[inline]
pub fn squeeze_momentum(
    input: &SqueezeMomentumInput,
) -> Result<SqueezeMomentumOutput, SqueezeMomentumError> {
    squeeze_momentum_with_kernel(input, Kernel::Auto)
}

/// Write squeeze momentum results directly into pre-allocated slices.
/// This is more efficient than allocating new vectors when the caller already has buffers.
/// All slices must have the same length as the input data.
#[inline]
pub fn squeeze_momentum_into_slices(
    squeeze_dst: &mut [f64],
    momentum_dst: &mut [f64],
    signal_dst: &mut [f64],
    input: &SqueezeMomentumInput,
    kern: Kernel,
) -> Result<(), SqueezeMomentumError> {
    // --- borrow inputs and validate exactly like alma_with_kernel/alma_into_slice ---
    let (high, low, close): (&[f64], &[f64], &[f64]) = match &input.data {
        SqueezeMomentumData::Candles { candles } => (
            source_type(candles, "high"),
            source_type(candles, "low"),
            source_type(candles, "close"),
        ),
        SqueezeMomentumData::Slices { high, low, close } => (*high, *low, *close),
    };
    let n = close.len();
    if n == 0 || high.is_empty() || low.is_empty() {
        return Err(SqueezeMomentumError::EmptyData);
    }
    if high.len() != low.len() || low.len() != close.len() {
        return Err(SqueezeMomentumError::InconsistentDataLength);
    }
    if squeeze_dst.len() != n || momentum_dst.len() != n || signal_dst.len() != n {
        return Err(SqueezeMomentumError::InconsistentDataLength);
    }

    let lbb = input.params.length_bb.unwrap_or(20);
    let lkc = input.params.length_kc.unwrap_or(20);
    let mbb = input.params.mult_bb.unwrap_or(2.0);
    let mkc = input.params.mult_kc.unwrap_or(1.5);
    if lbb == 0 || lbb > n {
        return Err(SqueezeMomentumError::InvalidLength {
            length: lbb,
            data_len: n,
        });
    }
    if lkc == 0 || lkc > n {
        return Err(SqueezeMomentumError::InvalidLength {
            length: lkc,
            data_len: n,
        });
    }

    let first_valid = (0..n)
        .find(|&i| !(high[i].is_nan() || low[i].is_nan() || close[i].is_nan()))
        .ok_or(SqueezeMomentumError::AllValuesNaN)?;
    let need = lbb.max(lkc);
    if n - first_valid < need {
        return Err(SqueezeMomentumError::NotEnoughValidData {
            needed: need,
            valid: n - first_valid,
        });
    }

    // Always use the optimized classic scalar kernel (SMA-based BB/KC)
    // This path is fully loop-jammed and allocation-light, matching alma.rs patterns
    let _ = kern; // runtime selection currently short-circuits to scalar
    unsafe {
        return squeeze_momentum_scalar_classic(
            high,
            low,
            close,
            lbb,
            mbb,
            lkc,
            mkc,
            first_valid,
            squeeze_dst,
            momentum_dst,
            signal_dst,
        );
    }
}

pub fn squeeze_momentum_with_kernel(
    input: &SqueezeMomentumInput,
    kernel: Kernel,
) -> Result<SqueezeMomentumOutput, SqueezeMomentumError> {
    let len = match &input.data {
        SqueezeMomentumData::Candles { candles } => candles.close.len(),
        SqueezeMomentumData::Slices { close, .. } => close.len(),
    };
    let mut squeeze = alloc_with_nan_prefix(len, 0);
    let mut momentum = alloc_with_nan_prefix(len, 0);
    let mut signal = alloc_with_nan_prefix(len, 0);

    squeeze_momentum_into_slices(&mut squeeze, &mut momentum, &mut signal, input, kernel)?;

    Ok(SqueezeMomentumOutput {
        squeeze,
        momentum,
        momentum_signal: signal,
    })
}

// --- Batch Parameter Sweep Support ---

#[derive(Clone, Debug)]
pub struct SqueezeMomentumBatchRange {
    pub length_bb: (usize, usize, usize),
    pub mult_bb: (f64, f64, f64),
    pub length_kc: (usize, usize, usize),
    pub mult_kc: (f64, f64, f64),
}

impl Default for SqueezeMomentumBatchRange {
    fn default() -> Self {
        Self {
            length_bb: (20, 20, 0),
            mult_bb: (2.0, 2.0, 0.0),
            length_kc: (20, 20, 0),
            mult_kc: (1.5, 1.5, 0.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SqueezeMomentumBatchBuilder {
    range: SqueezeMomentumBatchRange,
    kernel: Kernel,
}

impl SqueezeMomentumBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    pub fn length_bb_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.length_bb = (start, end, step);
        self
    }
    pub fn length_bb_static(mut self, p: usize) -> Self {
        self.range.length_bb = (p, p, 0);
        self
    }
    pub fn mult_bb_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.mult_bb = (start, end, step);
        self
    }
    pub fn mult_bb_static(mut self, x: f64) -> Self {
        self.range.mult_bb = (x, x, 0.0);
        self
    }
    pub fn length_kc_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.length_kc = (start, end, step);
        self
    }
    pub fn length_kc_static(mut self, p: usize) -> Self {
        self.range.length_kc = (p, p, 0);
        self
    }
    pub fn mult_kc_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.mult_kc = (start, end, step);
        self
    }
    pub fn mult_kc_static(mut self, x: f64) -> Self {
        self.range.mult_kc = (x, x, 0.0);
        self
    }

    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<SqueezeMomentumBatchOutput, SqueezeMomentumError> {
        squeeze_momentum_batch_with_kernel(high, low, close, &self.range, self.kernel)
    }

    pub fn apply_candles(
        self,
        c: &Candles,
    ) -> Result<SqueezeMomentumBatchOutput, SqueezeMomentumError> {
        let high = source_type(c, "high");
        let low = source_type(c, "low");
        let close = source_type(c, "close");
        self.apply_slices(high, low, close)
    }
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct SqueezeMomentumBatchParams {
    pub length_bb: usize,
    pub mult_bb: f64,
    pub length_kc: usize,
    pub mult_kc: f64,
}

#[derive(Clone, Debug)]
pub struct SqueezeMomentumBatchOutput {
    pub squeeze: Vec<f64>,
    pub momentum: Vec<f64>,
    pub signal: Vec<f64>,
    pub combos: Vec<SqueezeMomentumBatchParams>,
    pub rows: usize,
    pub cols: usize,
}

impl SqueezeMomentumBatchOutput {
    pub fn row_for_params(&self, p: &SqueezeMomentumBatchParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.length_bb == p.length_bb
                && (c.mult_bb - p.mult_bb).abs() < 1e-12
                && c.length_kc == p.length_kc
                && (c.mult_kc - p.mult_kc).abs() < 1e-12
        })
    }
    pub fn values_for(&self, p: &SqueezeMomentumBatchParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.momentum[start..start + self.cols]
        })
    }
}

#[inline]
fn warmups_for(p: &SqueezeMomentumBatchParams) -> (usize, usize, usize) {
    let sq = p.length_bb.max(p.length_kc).saturating_sub(1);
    let mo = p.length_kc.saturating_sub(1);
    let si = mo + 1;
    (sq, mo, si)
}

fn expand_grid_sm(range: &SqueezeMomentumBatchRange) -> Vec<SqueezeMomentumBatchParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    fn axis_f64((start, end, step): (f64, f64, f64)) -> Vec<f64> {
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return vec![start];
        }
        let mut v = Vec::new();
        let mut x = start;
        while x <= end + 1e-12 {
            v.push(x);
            x += step;
        }
        v
    }
    let length_bbs = axis_usize(range.length_bb);
    let mult_bbs = axis_f64(range.mult_bb);
    let length_kcs = axis_usize(range.length_kc);
    let mult_kcs = axis_f64(range.mult_kc);
    let mut out =
        Vec::with_capacity(length_bbs.len() * mult_bbs.len() * length_kcs.len() * mult_kcs.len());
    for &lbb in &length_bbs {
        for &mbb in &mult_bbs {
            for &lkc in &length_kcs {
                for &mkc in &mult_kcs {
                    out.push(SqueezeMomentumBatchParams {
                        length_bb: lbb,
                        mult_bb: mbb,
                        length_kc: lkc,
                        mult_kc: mkc,
                    });
                }
            }
        }
    }
    out
}

pub fn squeeze_momentum_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &SqueezeMomentumBatchRange,
    kernel: Kernel,
) -> Result<SqueezeMomentumBatchOutput, SqueezeMomentumError> {
    let combos = expand_grid_sm(sweep);
    if combos.is_empty() {
        return Err(SqueezeMomentumError::InvalidLength {
            length: 0,
            data_len: 0,
        });
    }
    let n = close.len();
    let first_valid = (0..n)
        .find(|&i| !(high[i].is_nan() || low[i].is_nan() || close[i].is_nan()))
        .ok_or(SqueezeMomentumError::AllValuesNaN)?;
    let need = combos
        .iter()
        .map(|c| c.length_bb.max(c.length_kc))
        .max()
        .unwrap();
    if n - first_valid < need {
        return Err(SqueezeMomentumError::NotEnoughValidData {
            needed: need,
            valid: n - first_valid,
        });
    }

    // Map Kernel::Auto to best batch kernel
    let chosen_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other => other,
    };

    let rows = combos.len();
    let cols = n;

    // allocate three uninit matrices and mark NaN warmups with init_matrix_prefixes
    let mut buf_sq = make_uninit_matrix(rows, cols);
    let mut buf_mo = make_uninit_matrix(rows, cols);
    let mut buf_si = make_uninit_matrix(rows, cols);

    let warm_sq: Vec<usize> = combos.iter().map(|p| warmups_for(p).0).collect();
    let warm_mo: Vec<usize> = combos.iter().map(|p| warmups_for(p).1).collect();
    let warm_si: Vec<usize> = combos.iter().map(|p| warmups_for(p).2).collect();

    init_matrix_prefixes(&mut buf_sq, cols, &warm_sq);
    init_matrix_prefixes(&mut buf_mo, cols, &warm_mo);
    init_matrix_prefixes(&mut buf_si, cols, &warm_si);

    // reborrow as f64 slices
    let mut sq =
        unsafe { core::slice::from_raw_parts_mut(buf_sq.as_mut_ptr() as *mut f64, rows * cols) };
    let mut mo =
        unsafe { core::slice::from_raw_parts_mut(buf_mo.as_mut_ptr() as *mut f64, rows * cols) };
    let mut si =
        unsafe { core::slice::from_raw_parts_mut(buf_si.as_mut_ptr() as *mut f64, rows * cols) };

    if matches!(chosen_kernel, Kernel::ScalarBatch) {
        // Optimized scalar batch: shared precompute across unique window sizes
        squeeze_momentum_batch_fill_scalar_shared(high, low, close, &combos, sq, mo, si);
    } else {
        let do_row = |row: usize, sq_row: &mut [f64], mo_row: &mut [f64], si_row: &mut [f64]| {
            let p = &combos[row];
            // zero-copy per row: write directly into row slices
            let params = SqueezeMomentumParams {
                length_bb: Some(p.length_bb),
                mult_bb: Some(p.mult_bb),
                length_kc: Some(p.length_kc),
                mult_kc: Some(p.mult_kc),
            };
            let input = SqueezeMomentumInput::from_slices(high, low, close, params);
            let _ = squeeze_momentum_into_slices(sq_row, mo_row, si_row, &input, chosen_kernel);
        };

        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;
            sq.par_chunks_mut(cols)
                .zip(mo.par_chunks_mut(cols))
                .zip(si.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(row, ((sq_row, mo_row), si_row))| do_row(row, sq_row, mo_row, si_row));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for row in 0..rows {
                let (sq_row, mo_row, si_row) = (
                    &mut sq[row * cols..(row + 1) * cols],
                    &mut mo[row * cols..(row + 1) * cols],
                    &mut si[row * cols..(row + 1) * cols],
                );
                do_row(row, sq_row, mo_row, si_row);
            }
        }
    }

    // move out Vecs with zero copies
    let squeeze = unsafe {
        Vec::from_raw_parts(
            buf_sq.as_mut_ptr() as *mut f64,
            buf_sq.len(),
            buf_sq.capacity(),
        )
    };
    let momentum = unsafe {
        Vec::from_raw_parts(
            buf_mo.as_mut_ptr() as *mut f64,
            buf_mo.len(),
            buf_mo.capacity(),
        )
    };
    let signal = unsafe {
        Vec::from_raw_parts(
            buf_si.as_mut_ptr() as *mut f64,
            buf_si.len(),
            buf_si.capacity(),
        )
    };
    core::mem::forget(buf_sq);
    core::mem::forget(buf_mo);
    core::mem::forget(buf_si);

    Ok(SqueezeMomentumBatchOutput {
        squeeze,
        momentum,
        signal,
        combos,
        rows,
        cols,
    })
}

pub fn squeeze_momentum_batch_inner_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &SqueezeMomentumBatchRange,
    kernel: Kernel,
    parallel: bool,
    out_squeeze: &mut [f64],
    out_momentum: &mut [f64],
    out_signal: &mut [f64],
) -> Result<Vec<SqueezeMomentumBatchParams>, SqueezeMomentumError> {
    let combos = expand_grid_sm(sweep);
    if combos.is_empty() {
        return Err(SqueezeMomentumError::InvalidLength {
            length: 0,
            data_len: 0,
        });
    }

    // Map Kernel::Auto to best batch kernel
    let chosen_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other => other,
    };

    let rows = combos.len();
    let cols = close.len();
    if out_squeeze.len() != rows * cols
        || out_momentum.len() != rows * cols
        || out_signal.len() != rows * cols
    {
        return Err(SqueezeMomentumError::InconsistentDataLength);
    }

    // mark warmups in-place via MaybeUninit cast + init_matrix_prefixes
    unsafe {
        let sq_mu = core::slice::from_raw_parts_mut(
            out_squeeze.as_mut_ptr() as *mut MaybeUninit<f64>,
            out_squeeze.len(),
        );
        let mo_mu = core::slice::from_raw_parts_mut(
            out_momentum.as_mut_ptr() as *mut MaybeUninit<f64>,
            out_momentum.len(),
        );
        let si_mu = core::slice::from_raw_parts_mut(
            out_signal.as_mut_ptr() as *mut MaybeUninit<f64>,
            out_signal.len(),
        );
        let warm_sq: Vec<usize> = combos.iter().map(|p| warmups_for(p).0).collect();
        let warm_mo: Vec<usize> = combos.iter().map(|p| warmups_for(p).1).collect();
        let warm_si: Vec<usize> = combos.iter().map(|p| warmups_for(p).2).collect();
        init_matrix_prefixes(sq_mu, cols, &warm_sq);
        init_matrix_prefixes(mo_mu, cols, &warm_mo);
        init_matrix_prefixes(si_mu, cols, &warm_si);
    }

    if matches!(chosen_kernel, Kernel::ScalarBatch) {
        // Shared-precompute scalar batch fill
        let combos_ref: &[SqueezeMomentumBatchParams] = &combos;
        squeeze_momentum_batch_fill_scalar_shared(
            high,
            low,
            close,
            combos_ref,
            out_squeeze,
            out_momentum,
            out_signal,
        );
    } else {
        let do_row = |row: usize, sq_row: &mut [f64], mo_row: &mut [f64], si_row: &mut [f64]| {
            let p = &combos[row];
            let params = SqueezeMomentumParams {
                length_bb: Some(p.length_bb),
                mult_bb: Some(p.mult_bb),
                length_kc: Some(p.length_kc),
                mult_kc: Some(p.mult_kc),
            };
            let input = SqueezeMomentumInput::from_slices(high, low, close, params);
            let _ = squeeze_momentum_into_slices(sq_row, mo_row, si_row, &input, kernel);
        };

        if parallel {
            #[cfg(not(target_arch = "wasm32"))]
            {
                use rayon::prelude::*;
                out_squeeze
                    .par_chunks_mut(cols)
                    .zip(out_momentum.par_chunks_mut(cols))
                    .zip(out_signal.par_chunks_mut(cols))
                    .enumerate()
                    .for_each(|(row, ((sq_row, mo_row), si_row))| {
                        do_row(row, sq_row, mo_row, si_row)
                    });
            }
            #[cfg(target_arch = "wasm32")]
            for row in 0..rows {
                let (sq_row, mo_row, si_row) = (
                    &mut out_squeeze[row * cols..(row + 1) * cols],
                    &mut out_momentum[row * cols..(row + 1) * cols],
                    &mut out_signal[row * cols..(row + 1) * cols],
                );
                do_row(row, sq_row, mo_row, si_row);
            }
        } else {
            for row in 0..rows {
                let (sq_row, mo_row, si_row) = (
                    &mut out_squeeze[row * cols..(row + 1) * cols],
                    &mut out_momentum[row * cols..(row + 1) * cols],
                    &mut out_signal[row * cols..(row + 1) * cols],
                );
                do_row(row, sq_row, mo_row, si_row);
            }
        }
    }

    Ok(combos)
}

// --- Batch shared-precompute scalar fill ---

fn sma_slice(data: &[f64], period: usize) -> Vec<f64> {
    let warm = period.saturating_sub(1);
    let n = data.len();
    let mut out = alloc_with_nan_prefix(n, warm);
    if period == 0 || period > n {
        return out;
    }
    let mut sum = 0.0;
    for i in 0..period {
        sum += data[i];
    }
    out[period - 1] = sum / period as f64;
    for i in period..n {
        sum += data[i] - data[i - period];
        out[i] = sum / period as f64;
    }
    out
}

fn squeeze_momentum_batch_fill_scalar_shared(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    combos: &[SqueezeMomentumBatchParams],
    out_sq: &mut [f64],
    out_mo: &mut [f64],
    out_si: &mut [f64],
) {
    use std::collections::{HashMap, HashSet};

    let n = close.len();
    let rows = combos.len();
    let cols = n;

    // First valid index
    let first_valid = (0..n)
        .find(|&i| !(high[i].is_nan() || low[i].is_nan() || close[i].is_nan()))
        .unwrap_or(n);

    // Unique window sizes
    let mut uniq_lkc: HashSet<usize> = HashSet::new();
    let mut uniq_lbb: HashSet<usize> = HashSet::new();
    for p in combos {
        uniq_lkc.insert(p.length_kc);
        uniq_lbb.insert(p.length_bb);
    }

    // Precompute TR once
    let tr = true_range_slice(high, low, close);

    // Precompute per unique lkc: kc_sma, tr_ma, highest, lowest, raw, momentum, signal
    struct KcPrecomp {
        kc_sma: Vec<f64>,
        tr_ma: Vec<f64>,
        highest: Vec<f64>,
        lowest: Vec<f64>,
        raw: Vec<f64>,
        momentum: Vec<f64>,
        signal: Vec<f64>,
    }
    let mut kc_map: HashMap<usize, KcPrecomp> = HashMap::with_capacity(uniq_lkc.len());
    for &lkc in &uniq_lkc {
        // kc_sma and tr_ma using same SMA semantics used elsewhere
        let kc_sma = sma_slice(close, lkc);
        let tr_ma = sma_slice(&tr, lkc);
        let highest = rolling_high_slice(high, lkc);
        let lowest = rolling_low_slice(low, lkc);

        // raw = close - 0.25*(highest+lowest) - 0.5*kc_sma
        let mut raw = alloc_with_nan_prefix(n, lkc.saturating_sub(1));
        for i in first_valid..n {
            if i + 1 >= lkc
                && close[i].is_finite()
                && highest[i].is_finite()
                && lowest[i].is_finite()
                && kc_sma[i].is_finite()
            {
                let mid = 0.5 * (highest[i] + lowest[i]);
                raw[i] = close[i] - 0.5 * (mid + kc_sma[i]);
            }
        }
        let momentum = linearreg_slice(&raw, lkc);
        let mut signal = alloc_with_nan_prefix(n, lkc.saturating_sub(1) + 1);
        let warm_sig = lkc.saturating_sub(1) + 1;
        for i in first_valid..n.saturating_sub(1) {
            let curr = momentum[i];
            let next = momentum[i + 1];
            if curr.is_finite() && next.is_finite() {
                signal[i + 1] = if next > 0.0 {
                    if next > curr {
                        1.0
                    } else {
                        2.0
                    }
                } else {
                    if next < curr {
                        -1.0
                    } else {
                        -2.0
                    }
                };
            } else if i + 1 >= warm_sig {
                signal[i + 1] = f64::NAN;
            }
        }

        kc_map.insert(
            lkc,
            KcPrecomp {
                kc_sma,
                tr_ma,
                highest,
                lowest,
                raw,
                momentum,
                signal,
            },
        );
    }

    // Precompute per unique lbb: bb_sma, dev
    struct BbPrecomp {
        bb_sma: Vec<f64>,
        dev: Vec<f64>,
    }
    let mut bb_map: HashMap<usize, BbPrecomp> = HashMap::with_capacity(uniq_lbb.len());
    for &lbb in &uniq_lbb {
        let bb_sma = sma_slice(close, lbb);
        let dev = stddev_slice(close, lbb);
        bb_map.insert(lbb, BbPrecomp { bb_sma, dev });
    }

    // Fill rows using shared precomputes
    let fill_row = |row: usize, sq_row: &mut [f64], mo_row: &mut [f64], si_row: &mut [f64]| {
        let p = &combos[row];
        let warm_sq = p.length_bb.max(p.length_kc).saturating_sub(1);

        let kc = kc_map.get(&p.length_kc).unwrap();
        let bb = bb_map.get(&p.length_bb).unwrap();

        // Squeeze bands and classification
        for i in first_valid..n {
            if i >= warm_sq
                && kc.kc_sma[i].is_finite()
                && kc.tr_ma[i].is_finite()
                && bb.bb_sma[i].is_finite()
                && bb.dev[i].is_finite()
            {
                let upper_kc = kc.kc_sma[i] + p.mult_kc * kc.tr_ma[i];
                let lower_kc = kc.kc_sma[i] - p.mult_kc * kc.tr_ma[i];
                let d = p.mult_bb * bb.dev[i];
                let upper_bb = bb.bb_sma[i] + d;
                let lower_bb = bb.bb_sma[i] - d;
                let on = lower_bb > lower_kc && upper_bb < upper_kc;
                let off = lower_bb < lower_kc && upper_bb > upper_kc;
                sq_row[i] = if on {
                    -1.0
                } else if off {
                    1.0
                } else {
                    0.0
                };
            } else if i >= warm_sq {
                // Ensure no poison remains beyond warmup
                sq_row[i] = f64::NAN;
            }
        }

        // Momentum and signal identical for all rows with same lkc
        mo_row.copy_from_slice(&kc.momentum);
        si_row.copy_from_slice(&kc.signal);
    };

    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        out_sq
            .par_chunks_mut(cols)
            .zip(out_mo.par_chunks_mut(cols))
            .zip(out_si.par_chunks_mut(cols))
            .enumerate()
            .for_each(|(row, ((sq_row, mo_row), si_row))| fill_row(row, sq_row, mo_row, si_row));
    }
    #[cfg(target_arch = "wasm32")]
    {
        for row in 0..rows {
            let (sq_row, mo_row, si_row) = (
                &mut out_sq[row * cols..(row + 1) * cols],
                &mut out_mo[row * cols..(row + 1) * cols],
                &mut out_si[row * cols..(row + 1) * cols],
            );
            fill_row(row, sq_row, mo_row, si_row);
        }
    }
}

// --- Classic Kernel Optimization ---

/// Classic scalar kernel with O(n) rolling updates and minimal temporaries.
/// - Rolling BB mean/std via sliding sums and FMA
/// - Rolling KC mid and TR via sliding sums
/// - Highest/lowest over KC window via monotonic deques (O(n))
/// - Linear regression on RAW via O(1) sliding OLS accumulators
#[inline(always)]
pub unsafe fn squeeze_momentum_scalar_classic(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    lbb: usize,
    mbb: f64,
    lkc: usize,
    mkc: f64,
    first_valid: usize,
    squeeze_dst: &mut [f64],
    momentum_dst: &mut [f64],
    signal_dst: &mut [f64],
) -> Result<(), SqueezeMomentumError> {
    let n = close.len();
    if n == 0 {
        return Ok(());
    }

    // Warmups (match framework expectations)
    let warm_sq = lbb.max(lkc).saturating_sub(1);
    let warm_m = lkc.saturating_sub(1);
    let warm_sig = warm_m + 1;

    // Initialize only warmup prefixes (match alma pattern)
    squeeze_dst[..warm_sq.min(n)].fill(f64::NAN);
    momentum_dst[..warm_m.min(n)].fill(f64::NAN);
    signal_dst[..warm_sig.min(n)].fill(f64::NAN);

    // Early exit if nothing to compute after warmups
    if first_valid >= n {
        return Ok(());
    }

    // Raw pointers for fast unchecked loads
    let hp = high.as_ptr();
    let lp = low.as_ptr();
    let cp = close.as_ptr();

    // Precompute constants
    let inv_lbb = 1.0 / (lbb as f64);
    let inv_lkc = 1.0 / (lkc as f64);

    // Linear regression constants for window size = lkc over RAW
    let p = lkc as f64;
    let sum_x = 0.5 * p * (p + 1.0);
    let sum_x2 = p * (p + 1.0) * (2.0 * p + 1.0) / 6.0;
    let denom = p * sum_x2 - sum_x * sum_x;
    let inv_denom = 1.0 / denom;
    let x_last_minus_xbar = p - sum_x * inv_lkc; // (p - (p+1)/2)

    // Rolling windows start indices
    let start_bb = first_valid + lbb.saturating_sub(1);
    let start_kc = first_valid + lkc.saturating_sub(1);

    // Initialize rolling sums for BB (mean/std)
    let mut sum_bb = 0.0_f64;
    let mut sumsq_bb = 0.0_f64;
    if start_bb < n {
        let s = start_bb + 1 - lbb;
        for j in s..=start_bb {
            let v = *cp.add(j);
            sum_bb += v;
            sumsq_bb = f64::mul_add(v, v, sumsq_bb);
        }
    }

    // Initialize rolling sums for KC mid (SMA) + TR average
    let mut sum_kc = 0.0_f64;
    let mut sum_tr = 0.0_f64;
    if start_kc < n {
        let s = start_kc + 1 - lkc;
        for j in s..=start_kc {
            sum_kc += *cp.add(j);

            // TR[j]
            let h = *hp.add(j);
            let l = *lp.add(j);
            let tr = if j == 0 {
                (h - l).abs()
            } else {
                let pc = *cp.add(j - 1);
                let tr1 = (h - l).abs();
                let tr2 = (h - pc).abs();
                let tr3 = (l - pc).abs();
                tr1.max(tr2).max(tr3)
            };
            sum_tr += tr;
        }
    }

    // Rolling Highest/Lowest over lkc via manual monotonic deques (ring buffers)
    // Avoid VecDeque overhead in hot loop.
    let mut dq_max_idx = vec![0usize; lkc];
    let mut dq_min_idx = vec![0usize; lkc];
    let mut max_head: usize = 0; // index of front element
    let mut max_len: usize = 0; // current length
    let mut min_head: usize = 0;
    let mut min_len: usize = 0;

    // RAW ring buffer for rolling linear regression (size = lkc)
    let mut raw_buf: Vec<f64> = vec![f64::NAN; lkc];
    let mut rb_pos: usize = 0;
    let mut raw_count: usize = 0;

    // Accumulators for sliding OLS over RAW:
    // S0 = sum y_j, S1 = sum j*y_j for j=1..p
    let mut S0 = 0.0_f64;
    let mut S1 = 0.0_f64;

    // Helpers for ring buffer operations
    #[inline(always)]
    fn rb_back(head: usize, len: usize, cap: usize) -> usize {
        (head + len - 1) % cap
    }
    #[inline(always)]
    fn rb_write_pos(head: usize, len: usize, cap: usize) -> usize {
        (head + len) % cap
    }

    for i in first_valid..n {
        // Update deques for rolling max/min
        let hi = *hp.add(i);
        let lo = *lp.add(i);

        // Evict stale indices for max deque
        while max_len > 0 {
            let idx = dq_max_idx[max_head];
            if idx + lkc <= i {
                max_head = (max_head + 1) % lkc;
                max_len -= 1;
            } else {
                break;
            }
        }
        // Evict stale indices for min deque
        while min_len > 0 {
            let idx = dq_min_idx[min_head];
            if idx + lkc <= i {
                min_head = (min_head + 1) % lkc;
                min_len -= 1;
            } else {
                break;
            }
        }

        // Maintain decreasing deque for highs
        while max_len > 0 {
            let back = rb_back(max_head, max_len, lkc);
            let idx = dq_max_idx[back];
            if *hp.add(idx) <= hi {
                max_len -= 1; // pop back
            } else {
                break;
            }
        }
        let pos = rb_write_pos(max_head, max_len, lkc);
        dq_max_idx[pos] = i;
        max_len += 1;

        // Maintain increasing deque for lows
        while min_len > 0 {
            let back = rb_back(min_head, min_len, lkc);
            let idx = dq_min_idx[back];
            if *lp.add(idx) >= lo {
                min_len -= 1; // pop back
            } else {
                break;
            }
        }
        let pos = rb_write_pos(min_head, min_len, lkc);
        dq_min_idx[pos] = i;
        min_len += 1;

        // Advance BB sums once past first full window
        if i > start_bb {
            let c_new = *cp.add(i);
            let c_old = *cp.add(i - lbb);
            sum_bb += c_new - c_old;
            // sumsq update with FMA
            sumsq_bb = f64::mul_add(c_new, c_new, sumsq_bb - c_old * c_old);
        }

        // Advance KC/TR sums once past their first full window
        if i > start_kc {
            let c_new = *cp.add(i);
            let c_old = *cp.add(i - lkc);
            sum_kc += c_new - c_old;

            // TR[i]
            let tr_new = {
                let h = *hp.add(i);
                let l = *lp.add(i);
                if i == 0 {
                    (h - l).abs()
                } else {
                    let pc = *cp.add(i - 1);
                    let tr1 = (h - l).abs();
                    let tr2 = (h - pc).abs();
                    let tr3 = (l - pc).abs();
                    tr1.max(tr2).max(tr3)
                }
            };
            // TR[i - lkc]
            let old_idx = i - lkc;
            let tr_old = {
                let h = *hp.add(old_idx);
                let l = *lp.add(old_idx);
                if old_idx == 0 {
                    (h - l).abs()
                } else {
                    let pc = *cp.add(old_idx - 1);
                    let tr1 = (h - l).abs();
                    let tr2 = (h - pc).abs();
                    let tr3 = (l - pc).abs();
                    tr1.max(tr2).max(tr3)
                }
            };
            sum_tr += tr_new - tr_old;
        }

        // Compute SQUEEZE state when both windows are ready
        if i >= start_bb && i >= start_kc && i >= warm_sq {
            let mean_bb = sum_bb * inv_lbb;
            let var_bb = f64::mul_add(-mean_bb, mean_bb, sumsq_bb * inv_lbb);
            let dev_bb = var_bb.max(0.0).sqrt();
            let upper_bb = f64::mul_add(mbb, dev_bb, mean_bb);
            let lower_bb = mean_bb - mbb * dev_bb;

            let kc_mid = sum_kc * inv_lkc;
            let tr_avg = sum_tr * inv_lkc;
            let upper_kc = kc_mid + mkc * tr_avg;
            let lower_kc = kc_mid - mkc * tr_avg;

            let on = lower_bb > lower_kc && upper_bb < upper_kc;
            let off = lower_bb < lower_kc && upper_bb > upper_kc;
            *squeeze_dst.get_unchecked_mut(i) = if on {
                -1.0
            } else if off {
                1.0
            } else {
                0.0
            };
        }

        // RAW & Momentum (rolling linear regression over RAW)
        if i >= start_kc {
            // Highest/Lowest in current KC window via deques
            let hi_idx = dq_max_idx[max_head];
            let lo_idx = dq_min_idx[min_head];
            let highest = *hp.add(hi_idx);
            let lowest = *lp.add(lo_idx);

            // KC mid at i
            let kc_mid = sum_kc * inv_lkc;

            // RAW = close - 0.25*(highest+lowest) - 0.5*kc_mid
            let c_i = *cp.add(i);
            let raw_i = c_i - 0.25 * (highest + lowest) - 0.5 * kc_mid;

            // Ring buffer insert
            let y_old = raw_buf[rb_pos];
            raw_buf[rb_pos] = raw_i;
            rb_pos += 1;
            if rb_pos == lkc {
                rb_pos = 0;
            }

            // Grow RAW count up to lkc
            if raw_count < lkc {
                raw_count += 1;
            }

            // Initialize S0/S1 once when RAW window fills
            if raw_count == lkc && i == start_kc + lkc - 1 {
                let mut s0 = 0.0_f64;
                let mut s1 = 0.0_f64;
                // Oldest is at rb_pos, walk j=1..p
                let mut idx = rb_pos;
                let mut j = 1.0_f64;
                for _ in 0..lkc {
                    let y = raw_buf[idx];
                    s0 += y;
                    s1 = f64::mul_add(j, y, s1);
                    j += 1.0;
                    idx += 1;
                    if idx == lkc {
                        idx = 0;
                    }
                }
                S0 = s0;
                S1 = s1;

                // First momentum point
                let b = f64::mul_add(-sum_x, S0, p * S1) * inv_denom;
                let ybar = S0 * inv_lkc;
                let yhat_last = f64::mul_add(b, x_last_minus_xbar, ybar);
                *momentum_dst.get_unchecked_mut(i) = yhat_last;

                // First signal (lagged)
                if i >= 1 {
                    let prev = *momentum_dst.get_unchecked(i - 1);
                    if prev.is_finite() && yhat_last.is_finite() {
                        *signal_dst.get_unchecked_mut(i) = if yhat_last > 0.0 {
                            if yhat_last > prev {
                                1.0
                            } else {
                                2.0
                            }
                        } else {
                            if yhat_last < prev {
                                -1.0
                            } else {
                                -2.0
                            }
                        };
                    } else if i >= warm_sig {
                        *signal_dst.get_unchecked_mut(i) = f64::NAN;
                    }
                }
            } else if raw_count == lkc {
                // O(1) rolling update:
                // S1' = S1 - S0 + p*y_new
                // S0' = S0 - y_old + y_new
                let y_new = raw_i;
                let new_S1 = (S1 - S0) + p * y_new;
                let new_S0 = (S0 - y_old) + y_new;
                S1 = new_S1;
                S0 = new_S0;

                // Momentum at i
                let b = f64::mul_add(-sum_x, S0, p * S1) * inv_denom;
                let ybar = S0 * inv_lkc;
                let yhat_last = f64::mul_add(b, x_last_minus_xbar, ybar);
                *momentum_dst.get_unchecked_mut(i) = yhat_last;

                // Signal (lagged 1)
                if i >= 1 {
                    let prev = *momentum_dst.get_unchecked(i - 1);
                    if prev.is_finite() && yhat_last.is_finite() {
                        *signal_dst.get_unchecked_mut(i) = if yhat_last > 0.0 {
                            if yhat_last > prev {
                                1.0
                            } else {
                                2.0
                            }
                        } else {
                            if yhat_last < prev {
                                -1.0
                            } else {
                                -2.0
                            }
                        };
                    } else if i >= warm_sig {
                        *signal_dst.get_unchecked_mut(i) = f64::NAN;
                    }
                }
            } else {
                // Not enough RAW samples yet for regression window -> explicit NaN
                *momentum_dst.get_unchecked_mut(i) = f64::NAN;
                if i >= warm_sig {
                    *signal_dst.get_unchecked_mut(i) = f64::NAN;
                }
            }
        }
    }

    Ok(())
}

// --- Utilities (Unchanged from scalar, as in original) ---

fn stddev_slice(data: &[f64], period: usize) -> Vec<f64> {
    let warmup = period.saturating_sub(1);
    let mut output = alloc_with_nan_prefix(data.len(), warmup);
    if period == 0 || period > data.len() {
        return output;
    }
    let mut window_sum = 0.0;
    let mut window_sumsq = 0.0;
    for i in 0..period {
        let v = data[i];
        if v.is_finite() {
            window_sum += v;
            window_sumsq += v * v;
        }
    }
    let mut count = period;
    if count > 0 {
        output[period - 1] = variance_to_stddev(window_sum, window_sumsq, count);
    }
    for i in period..data.len() {
        let old_v = data[i - period];
        let new_v = data[i];
        if old_v.is_finite() {
            window_sum -= old_v;
            window_sumsq -= old_v * old_v;
        }
        if new_v.is_finite() {
            window_sum += new_v;
            window_sumsq += new_v * new_v;
        }
        output[i] = variance_to_stddev(window_sum, window_sumsq, count);
    }
    output
}
fn variance_to_stddev(sum: f64, sumsq: f64, count: usize) -> f64 {
    if count < 2 {
        return f64::NAN;
    }
    let mean = sum / (count as f64);
    let var = (sumsq / (count as f64)) - (mean * mean);
    if var.is_sign_negative() {
        f64::NAN
    } else {
        var.sqrt()
    }
}
fn true_range_slice(high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    if high.len() != low.len() || low.len() != close.len() {
        return vec![];
    }
    let mut output = alloc_with_nan_prefix(high.len(), 0); // TR has no warmup
    let mut prev_close = close[0];
    output[0] = high[0].max(low[0]) - low[0].min(high[0]);
    for i in 1..high.len() {
        if !high[i].is_nan() && !low[i].is_nan() && !prev_close.is_nan() {
            let tr1 = high[i] - low[i];
            let tr2 = (high[i] - prev_close).abs();
            let tr3 = (low[i] - prev_close).abs();
            output[i] = tr1.max(tr2).max(tr3);
        }
        prev_close = close[i];
    }
    output
}
fn rolling_high_slice(data: &[f64], period: usize) -> Vec<f64> {
    let warmup = period.saturating_sub(1);
    let n = data.len();
    let mut output = alloc_with_nan_prefix(n, warmup);
    if period == 0 || period > n {
        return output;
    }

    // Monotonic deque of indices for finite values (decreasing by value)
    use std::collections::VecDeque;
    let mut dq: VecDeque<usize> = VecDeque::with_capacity(period);
    for i in 0..n {
        let v = data[i];
        // Evict out-of-window indices from front
        while let Some(&front) = dq.front() {
            if front + period <= i {
                dq.pop_front();
            } else {
                break;
            }
        }
        // Maintain decreasing order; ignore NaNs
        if v.is_finite() {
            while let Some(&idx) = dq.back() {
                if data[idx] <= v {
                    dq.pop_back();
                } else {
                    break;
                }
            }
            dq.push_back(i);
        }
        if i + 1 >= period {
            output[i] = if let Some(&idx) = dq.front() {
                data[idx]
            } else {
                f64::NAN
            };
        }
    }
    output
}
fn rolling_low_slice(data: &[f64], period: usize) -> Vec<f64> {
    let warmup = period.saturating_sub(1);
    let n = data.len();
    let mut output = alloc_with_nan_prefix(n, warmup);
    if period == 0 || period > n {
        return output;
    }

    // Monotonic deque of indices for finite values (increasing by value)
    use std::collections::VecDeque;
    let mut dq: VecDeque<usize> = VecDeque::with_capacity(period);
    for i in 0..n {
        let v = data[i];
        // Evict out-of-window indices from front
        while let Some(&front) = dq.front() {
            if front + period <= i {
                dq.pop_front();
            } else {
                break;
            }
        }
        // Maintain increasing order; ignore NaNs
        if v.is_finite() {
            while let Some(&idx) = dq.back() {
                if data[idx] >= v {
                    dq.pop_back();
                } else {
                    break;
                }
            }
            dq.push_back(i);
        }
        if i + 1 >= period {
            output[i] = if let Some(&idx) = dq.front() {
                data[idx]
            } else {
                f64::NAN
            };
        }
    }
    output
}
fn linearreg_slice(data: &[f64], period: usize) -> Vec<f64> {
    let warmup = period.saturating_sub(1);
    let mut output = alloc_with_nan_prefix(data.len(), warmup);
    if period == 0 || period > data.len() {
        return output;
    }
    for i in (period - 1)..data.len() {
        let subset = &data[i + 1 - period..=i];
        if subset.iter().all(|x| x.is_finite()) {
            output[i] = linear_regression_last_point(subset);
        } else {
            // Explicitly set NaN when input data contains NaN
            output[i] = f64::NAN;
        }
    }
    output
}
fn linear_regression_last_point(window: &[f64]) -> f64 {
    let n = window.len();
    if n < 2 {
        return f64::NAN;
    }
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xy = 0.0;
    let mut sum_x2 = 0.0;
    for (i, &val) in window.iter().enumerate() {
        let x = (i + 1) as f64;
        sum_x += x;
        sum_y += val;
        sum_xy += x * val;
        sum_x2 += x * x;
    }
    let n_f = n as f64;
    let denom = (n_f * sum_x2) - (sum_x * sum_x);
    if denom.abs() < f64::EPSILON {
        return f64::NAN;
    }
    let slope = (n_f * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / n_f;
    let x_last = n_f;
    intercept + slope * x_last
}

// --- Streaming Support ---
//
// O(1) streaming kernel:
// - Rolling BB (mean/variance) via sliding sums and sums-of-squares
// - Rolling KC mid and TR (ATR) via sliding sums and rings
// - Highest/Lowest over KC window via monotonic deques (ring-backed, no alloc)
// - Momentum = OLS endpoint of RAW over lkc via two accumulators (S0, S1)
// - Signal = compare current momentum to previous momentum (1/-1 accelerating, 2/-2 decelerating)
// - Squeeze classification avoids sqrt by comparing BB variance to KC distances (exact, no approx)

/// Streaming implementation for Squeeze Momentum indicator
pub struct SqueezeMomentumStream {
    // original config
    params: SqueezeMomentumParams,
    rp: ResolvedParams,

    // counters
    n: usize,        // number of ticks processed so far
    ready_bb: bool,  // BB window filled once
    ready_kc: bool,  // KC/TR window filled once
    ready_raw: bool, // RAW regression window filled once

    // --- Bollinger Bands state (window = lbb) ---
    lbb: usize,
    inv_lbb: f64,
    sum_bb: f64,
    sumsq_bb: f64,
    rb_close_bb: Vec<f64>, // ring for close (length lbb)
    bb_pos: usize,         // write position in rb_close_bb

    // --- Keltner Channels state (window = lkc) ---
    lkc: usize,
    inv_lkc: f64,
    mbb: f64,
    mkc: f64,

    // SMA(close, lkc)
    sum_kc: f64,
    rb_close_kc: Vec<f64>, // ring for close (length lkc)
    kc_pos: usize,

    // ATR = SMA(TR, lkc)
    sum_tr: f64,
    rb_tr: Vec<f64>, // ring for TR (length lkc)

    prev_close: f64, // previous close (for TR)

    // --- rolling highest/lowest over lkc using monotone deques ---
    ring_high: Vec<f64>, // last lkc highs at indices i % lkc
    ring_low: Vec<f64>,  // last lkc lows  at indices i % lkc

    dq_hi_idx: Vec<usize>, // stores absolute indices (capacity lkc)
    dq_hi_head: usize,
    dq_hi_len: usize,

    dq_lo_idx: Vec<usize>, // stores absolute indices (capacity lkc)
    dq_lo_head: usize,
    dq_lo_len: usize,

    // --- O(1) sliding OLS on RAW over lkc ---
    raw_ring: Vec<f64>, // ring buffer for RAW (length lkc)
    raw_pos: usize,
    raw_count: usize,
    S0: f64, // sum y
    S1: f64, // sum j*y, j=1..p

    // precomputed constants for fixed x = 1..p
    p_f64: f64,
    sum_x: f64,
    sum_x2: f64,
    inv_denom: f64,
    x_last_minus_xbar: f64,

    // last momentum (for signal)
    last_momentum: f64,
}

impl SqueezeMomentumStream {
    /// Create a new SqueezeMomentumStream with given parameters (validated & resolved).
    pub fn try_new(params: SqueezeMomentumParams) -> Result<Self, SqueezeMomentumError> {
        let rp = params.resolve();
        if rp.length_bb == 0 || rp.length_kc == 0 {
            return Err(SqueezeMomentumError::InvalidLength {
                length: rp.length_bb.max(rp.length_kc),
                data_len: 0,
            });
        }

        let lbb = rp.length_bb;
        let lkc = rp.length_kc;

        let inv_lbb = 1.0 / (lbb as f64);
        let inv_lkc = 1.0 / (lkc as f64);

        // OLS precompute for x = 1..p
        let p = lkc as f64;
        let sum_x = 0.5 * p * (p + 1.0);
        let sum_x2 = p * (p + 1.0) * (2.0 * p + 1.0) / 6.0;
        let denom = p * sum_x2 - sum_x * sum_x;
        let inv_denom = if denom.abs() < f64::EPSILON {
            0.0
        } else {
            1.0 / denom
        };
        let x_last_minus_xbar = p - sum_x * inv_lkc;

        Ok(Self {
            params,
            rp,

            n: 0,
            ready_bb: false,
            ready_kc: false,
            ready_raw: false,

            lbb,
            inv_lbb,
            sum_bb: 0.0,
            sumsq_bb: 0.0,
            rb_close_bb: vec![0.0; lbb],
            bb_pos: 0,

            lkc,
            inv_lkc,
            mbb: rp.mult_bb,
            mkc: rp.mult_kc,

            sum_kc: 0.0,
            rb_close_kc: vec![0.0; lkc],
            kc_pos: 0,

            sum_tr: 0.0,
            rb_tr: vec![0.0; lkc],

            prev_close: f64::NAN,

            ring_high: vec![0.0; lkc],
            ring_low: vec![0.0; lkc],

            dq_hi_idx: vec![0; lkc],
            dq_hi_head: 0,
            dq_hi_len: 0,

            dq_lo_idx: vec![0; lkc],
            dq_lo_head: 0,
            dq_lo_len: 0,

            raw_ring: vec![0.0; lkc],
            raw_pos: 0,
            raw_count: 0,
            S0: 0.0,
            S1: 0.0,

            p_f64: p,
            sum_x,
            sum_x2,
            inv_denom,
            x_last_minus_xbar,

            last_momentum: f64::NAN,
        })
    }

    /// Create with default parameters
    pub fn new() -> Self {
        Self::try_new(SqueezeMomentumParams::default()).unwrap()
    }

    #[inline(always)]
    fn dq_back(head: usize, len: usize, cap: usize) -> usize {
        (head + len - 1) % cap
    }
    #[inline(always)]
    fn dq_push_back(dq: &mut [usize], head: &mut usize, len: &mut usize, cap: usize, idx: usize) {
        let pos = (*head + *len) % cap;
        dq[pos] = idx;
        *len += 1;
    }
    #[inline(always)]
    fn dq_pop_front(len: &mut usize, head: &mut usize, cap: usize) {
        *head = (*head + 1) % cap;
        *len -= 1;
    }
    #[inline(always)]
    fn dq_pop_back(len: &mut usize) {
        *len -= 1;
    }

    /// Classification without sqrt:
    /// Returns -1.0 (on), 0.0 (neutral), 1.0 (off).
    #[inline(always)]
    fn classify_squeeze_no_sqrt(
        mean_bb: f64,
        var_bb: f64,
        kc_mid: f64,
        tr_avg: f64,
        mbb: f64,
        mkc: f64,
    ) -> f64 {
        let upper_kc = kc_mid + mkc * tr_avg;
        let lower_kc = kc_mid - mkc * tr_avg;

        // distances from BB mean to KC rails
        let d1 = mean_bb - lower_kc; // must be > m*std for ON
        let d2 = upper_kc - mean_bb; // must be > m*std for ON

        let m = mbb.abs(); // safety

        // ON: require both distances positive and variance below both thresholds
        if d1 > 0.0 && d2 > 0.0 {
            let thr = (d1.min(d2) / m);
            if var_bb < thr * thr {
                return -1.0;
            }
        }

        // OFF: require m*std > max(d1,d2) (clip negatives to zero)
        let t1 = d1.max(0.0);
        let t2 = d2.max(0.0);
        let thr_off = (t1.max(t2) / m);
        if var_bb > thr_off * thr_off {
            return 1.0;
        }
        0.0
    }

    /// Update the stream with a new (high, low, close).
    /// Returns None until both windows are warm and the 1‑bar lagged signal is available.
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<(f64, f64, f64)> {
        // reject non-finite inputs: match batch/scalar expectations of NaN propagation
        if !(high.is_finite() && low.is_finite() && close.is_finite()) {
            self.n = self.n.saturating_add(1);
            return None;
        }

        let i = self.n;
        let lbb = self.lbb;
        let lkc = self.lkc;

        // -----------------------------
        // 1) Maintain BB rolling sums
        // -----------------------------
        if self.ready_bb {
            // evict oldest
            let old = self.rb_close_bb[self.bb_pos];
            self.sum_bb += close - old;
            self.sumsq_bb += close * close - old * old;
            self.rb_close_bb[self.bb_pos] = close;
            self.bb_pos += 1;
            if self.bb_pos == lbb {
                self.bb_pos = 0;
            }
        } else {
            self.sum_bb += close;
            self.sumsq_bb += close * close;
            self.rb_close_bb[self.bb_pos] = close;
            self.bb_pos += 1;
            if self.bb_pos == lbb {
                self.bb_pos = 0;
                self.ready_bb = true;
            }
        }

        // -----------------------------
        // 2) Maintain KC SMA(close) and ATR = SMA(TR)
        // -----------------------------
        // TR for this bar
        let tr = if i == 0 || !self.prev_close.is_finite() {
            (high - low).abs()
        } else {
            let pc = self.prev_close;
            let tr1 = (high - low).abs();
            let tr2 = (high - pc).abs();
            let tr3 = (low - pc).abs();
            tr1.max(tr2).max(tr3)
        };
        self.prev_close = close;

        if self.ready_kc {
            let old_c = self.rb_close_kc[self.kc_pos];
            let old_tr = self.rb_tr[self.kc_pos];
            self.sum_kc += close - old_c;
            self.sum_tr += tr - old_tr;
            self.rb_close_kc[self.kc_pos] = close;
            self.rb_tr[self.kc_pos] = tr;
            // advance
            self.kc_pos += 1;
            if self.kc_pos == lkc {
                self.kc_pos = 0;
            }
        } else {
            self.sum_kc += close;
            self.sum_tr += tr;
            self.rb_close_kc[self.kc_pos] = close;
            self.rb_tr[self.kc_pos] = tr;
            self.kc_pos += 1;
            if self.kc_pos == lkc {
                self.kc_pos = 0;
                self.ready_kc = true;
            }
        }

        // ------------------------------------
        // 3) Maintain deque highs/lows (lkc)
        // ------------------------------------
        let slot = i % lkc;
        self.ring_high[slot] = high;
        self.ring_low[slot] = low;

        // evict stale from fronts: while idx + lkc <= i
        while self.dq_hi_len > 0 {
            let idx = self.dq_hi_idx[self.dq_hi_head];
            if idx + lkc <= i {
                Self::dq_pop_front(&mut self.dq_hi_len, &mut self.dq_hi_head, lkc);
            } else {
                break;
            }
        }
        while self.dq_lo_len > 0 {
            let idx = self.dq_lo_idx[self.dq_lo_head];
            if idx + lkc <= i {
                Self::dq_pop_front(&mut self.dq_lo_len, &mut self.dq_lo_head, lkc);
            } else {
                break;
            }
        }

        // maintain decreasing deque for highs
        while self.dq_hi_len > 0 {
            let back = Self::dq_back(self.dq_hi_head, self.dq_hi_len, lkc);
            let idx = self.dq_hi_idx[back];
            let v_back = self.ring_high[idx % lkc];
            if v_back <= high {
                Self::dq_pop_back(&mut self.dq_hi_len);
            } else {
                break;
            }
        }
        Self::dq_push_back(
            &mut self.dq_hi_idx,
            &mut self.dq_hi_head,
            &mut self.dq_hi_len,
            lkc,
            i,
        );

        // maintain increasing deque for lows
        while self.dq_lo_len > 0 {
            let back = Self::dq_back(self.dq_lo_head, self.dq_lo_len, lkc);
            let idx = self.dq_lo_idx[back];
            let v_back = self.ring_low[idx % lkc];
            if v_back >= low {
                Self::dq_pop_back(&mut self.dq_lo_len);
            } else {
                break;
            }
        }
        Self::dq_push_back(
            &mut self.dq_lo_idx,
            &mut self.dq_lo_head,
            &mut self.dq_lo_len,
            lkc,
            i,
        );

        // -----------------------------
        // 4) If KC window ready, build RAW and update OLS
        // -----------------------------
        let mut momentum = f64::NAN;
        if self.ready_kc {
            let hi_idx = self.dq_hi_idx[self.dq_hi_head];
            let lo_idx = self.dq_lo_idx[self.dq_lo_head];
            let highest = self.ring_high[hi_idx % lkc];
            let lowest = self.ring_low[lo_idx % lkc];
            let kc_mid = self.sum_kc * self.inv_lkc;

            // RAW = close - 0.25*(highest+lowest) - 0.5*kc_mid
            let raw_i = close - 0.25 * (highest + lowest) - 0.5 * kc_mid;

            if self.raw_count < lkc {
                // growing stage: append at raw_pos (== slot)
                self.raw_ring[slot] = raw_i;
                self.raw_count += 1;
                // update S0, S1 with j = raw_count
                let j = self.raw_count as f64;
                self.S0 += raw_i;
                self.S1 = f64::mul_add(j, raw_i, self.S1);
                if self.raw_count == lkc {
                    self.ready_raw = true;
                }
            } else {
                // steady stage: O(1) update
                let y_old = self.raw_ring[slot];
                self.raw_ring[slot] = raw_i;

                // S1' = S1 - S0 + p*y_new
                // S0' = S0 - y_old + y_new
                self.S1 = (self.S1 - self.S0) + self.p_f64 * raw_i;
                self.S0 = (self.S0 - y_old) + raw_i;
            }

            // compute momentum when regression window is ready
            if self.ready_raw {
                let ybar = self.S0 * self.inv_lkc;
                // b = (-sum_x*S0 + p*S1) / denom; handle denom == 0 -> b = 0
                let b_num = -self.sum_x * self.S0 + self.p_f64 * self.S1;
                let b = if self.inv_denom == 0.0 {
                    0.0
                } else {
                    b_num * self.inv_denom
                };
                momentum = f64::mul_add(b, self.x_last_minus_xbar, ybar);
            }
        }

        // -----------------------------
        // 5) Compute squeeze classification (no sqrt)
        // -----------------------------
        let mut squeeze = f64::NAN;
        if self.ready_bb && self.ready_kc {
            let mean_bb = self.sum_bb * self.inv_lbb;
            let var_bb = f64::mul_add(self.sumsq_bb * self.inv_lbb, 1.0, -mean_bb * mean_bb); // E[x^2] - (E[x])^2
            let kc_mid = self.sum_kc * self.inv_lkc;
            let tr_avg = self.sum_tr * self.inv_lkc;
            squeeze = Self::classify_squeeze_no_sqrt(
                mean_bb,
                var_bb.max(0.0),
                kc_mid,
                tr_avg,
                self.mbb,
                self.mkc,
            );
        }

        // -----------------------------
        // 6) Signal (needs previous momentum)
        // -----------------------------
        let mut signal = f64::NAN;
        if self.ready_raw && self.last_momentum.is_finite() && momentum.is_finite() {
            signal = if momentum > 0.0 {
                if momentum > self.last_momentum {
                    1.0
                } else {
                    2.0
                }
            } else {
                if momentum < self.last_momentum {
                    -1.0
                } else {
                    -2.0
                }
            };
        }

        // update prev momentum after signal decision
        if momentum.is_finite() {
            self.last_momentum = momentum;
        }

        // tick++
        self.n = i + 1;

        // warmup: return only when we can output all three
        if squeeze.is_finite() && momentum.is_finite() && signal.is_finite() {
            Some((squeeze, momentum, signal))
        } else {
            None
        }
    }
}

impl Default for SqueezeMomentumStream {
    fn default() -> Self {
        Self::new()
    }
}

// --- Python Bindings ---

#[cfg(feature = "python")]
#[pyclass(name = "SqueezeMomentumStream")]
pub struct SqueezeMomentumStreamPy {
    stream: SqueezeMomentumStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl SqueezeMomentumStreamPy {
    #[new]
    #[pyo3(signature = (length_bb=20, mult_bb=2.0, length_kc=20, mult_kc=1.5))]
    fn new(length_bb: usize, mult_bb: f64, length_kc: usize, mult_kc: f64) -> PyResult<Self> {
        let params = SqueezeMomentumParams {
            length_bb: Some(length_bb),
            mult_bb: Some(mult_bb),
            length_kc: Some(length_kc),
            mult_kc: Some(mult_kc),
        };
        let stream = SqueezeMomentumStream::try_new(params)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(SqueezeMomentumStreamPy { stream })
    }

    fn update(
        &mut self,
        high: f64,
        low: f64,
        close: f64,
    ) -> (Option<f64>, Option<f64>, Option<f64>) {
        match self.stream.update(high, low, close) {
            Some((squeeze, momentum, signal)) => (Some(squeeze), Some(momentum), Some(signal)),
            None => (None, None, None),
        }
    }
}

// --- Python Bindings ---

#[cfg(feature = "python")]
#[pyfunction(name = "squeeze_momentum")]
#[pyo3(signature = (high, low, close, length_bb=20, mult_bb=2.0, length_kc=20, mult_kc=1.5, kernel=None))]
pub fn squeeze_momentum_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    length_bb: usize,
    mult_bb: f64,
    length_kc: usize,
    mult_kc: f64,
    kernel: Option<&str>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let h = high.as_slice()?;
    let l = low.as_slice()?;
    let c = close.as_slice()?;

    let n = c.len();
    let sq = unsafe { PyArray1::<f64>::new(py, [n], false) };
    let mo = unsafe { PyArray1::<f64>::new(py, [n], false) };
    let si = unsafe { PyArray1::<f64>::new(py, [n], false) };

    let mut sq_slice = unsafe { sq.as_slice_mut()? };
    let mut mo_slice = unsafe { mo.as_slice_mut()? };
    let mut si_slice = unsafe { si.as_slice_mut()? };

    let kern = validate_kernel(kernel, false)?;
    let params = SqueezeMomentumParams {
        length_bb: Some(length_bb),
        mult_bb: Some(mult_bb),
        length_kc: Some(length_kc),
        mult_kc: Some(mult_kc),
    };
    let input = SqueezeMomentumInput::from_slices(h, l, c, params);

    py.allow_threads(|| {
        squeeze_momentum_into_slices(&mut sq_slice, &mut mo_slice, &mut si_slice, &input, kern)
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok((sq, mo, si))
}

// ==================== PYTHON CUDA BINDINGS ====================
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::{cuda_available, CudaSqueezeMomentum};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::alma::DeviceArrayF32Py;
#[cfg(all(feature = "python", feature = "cuda"))]
use numpy::PyReadonlyArray1;
#[cfg(all(feature = "python", feature = "cuda"))]
use pyo3::exceptions::PyValueError;

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "squeeze_momentum_cuda_batch_dev")]
#[pyo3(signature = (high_f32, low_f32, close_f32, length_bb_range, mult_bb_range, length_kc_range, mult_kc_range, device_id=0))]
pub fn squeeze_momentum_cuda_batch_dev_py(
    py: Python<'_>,
    high_f32: PyReadonlyArray1<'_, f32>,
    low_f32: PyReadonlyArray1<'_, f32>,
    close_f32: PyReadonlyArray1<'_, f32>,
    length_bb_range: (usize, usize, usize),
    mult_bb_range: (f64, f64, f64),
    length_kc_range: (usize, usize, usize),
    mult_kc_range: (f64, f64, f64),
    device_id: usize,
) -> PyResult<(DeviceArrayF32Py, DeviceArrayF32Py, DeviceArrayF32Py)> {
    if !cuda_available() { return Err(PyValueError::new_err("CUDA not available")); }
    let h = high_f32.as_slice()?;
    let l = low_f32.as_slice()?;
    let c = close_f32.as_slice()?;
    let sweep = SqueezeMomentumBatchRange { length_bb: length_bb_range, mult_bb: mult_bb_range, length_kc: length_kc_range, mult_kc: mult_kc_range };
    let (sq, mo, si) = py.allow_threads(|| {
        let cuda = CudaSqueezeMomentum::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.squeeze_momentum_batch_dev(h, l, c, &sweep).map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    Ok((DeviceArrayF32Py { inner: sq }, DeviceArrayF32Py { inner: mo }, DeviceArrayF32Py { inner: si }))
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "squeeze_momentum_cuda_many_series_one_param_dev")]
#[pyo3(signature = (high_tm_f32, low_tm_f32, close_tm_f32, cols, rows, length_bb, mult_bb, length_kc, mult_kc, device_id=0))]
pub fn squeeze_momentum_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    high_tm_f32: PyReadonlyArray1<'_, f32>,
    low_tm_f32: PyReadonlyArray1<'_, f32>,
    close_tm_f32: PyReadonlyArray1<'_, f32>,
    cols: usize,
    rows: usize,
    length_bb: usize,
    mult_bb: f32,
    length_kc: usize,
    mult_kc: f32,
    device_id: usize,
) -> PyResult<(DeviceArrayF32Py, DeviceArrayF32Py, DeviceArrayF32Py)> {
    if !cuda_available() { return Err(PyValueError::new_err("CUDA not available")); }
    let h = high_tm_f32.as_slice()?; let l = low_tm_f32.as_slice()?; let c = close_tm_f32.as_slice()?;
    let (sq, mo, si) = py.allow_threads(|| {
        let cuda = CudaSqueezeMomentum::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.squeeze_momentum_many_series_one_param_time_major_dev(h, l, c, cols, rows, length_bb, mult_bb, length_kc, mult_kc)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    Ok((DeviceArrayF32Py { inner: sq }, DeviceArrayF32Py { inner: mo }, DeviceArrayF32Py { inner: si }))
}

#[cfg(feature = "python")]
#[pyfunction(name = "squeeze_momentum_batch")]
#[pyo3(signature = (high, low, close, length_bb_range, mult_bb_range, length_kc_range, mult_kc_range, kernel=None))]
pub fn squeeze_momentum_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    length_bb_range: (usize, usize, usize),
    mult_bb_range: (f64, f64, f64),
    length_kc_range: (usize, usize, usize),
    mult_kc_range: (f64, f64, f64),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let h = high.as_slice()?;
    let l = low.as_slice()?;
    let c = close.as_slice()?;

    let sweep = SqueezeMomentumBatchRange {
        length_bb: length_bb_range,
        mult_bb: mult_bb_range,
        length_kc: length_kc_range,
        mult_kc: mult_kc_range,
    };

    let out = py.allow_threads(|| {
        let k = validate_kernel(kernel, true)?;
        let simd = match k {
            Kernel::Auto => detect_best_batch_kernel(),
            other => other,
        };
        squeeze_momentum_batch_with_kernel(h, l, c, &sweep, simd)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    let dict = PyDict::new(py);
    // Return only momentum values as 'values' to match test expectations
    dict.set_item(
        "values",
        PyArray1::from_vec(py, out.momentum).reshape((out.rows, out.cols))?,
    )?;

    // Also include squeeze and signal for completeness
    dict.set_item(
        "squeeze",
        PyArray1::from_vec(py, out.squeeze).reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "signal",
        PyArray1::from_vec(py, out.signal).reshape((out.rows, out.cols))?,
    )?;

    // metadata arrays
    dict.set_item(
        "length_bb",
        PyArray1::from_vec(
            py,
            out.combos
                .iter()
                .map(|p| p.length_bb as i64)
                .collect::<Vec<_>>(),
        ),
    )?;
    dict.set_item(
        "mult_bb",
        PyArray1::from_vec(py, out.combos.iter().map(|p| p.mult_bb).collect::<Vec<_>>()),
    )?;
    dict.set_item(
        "length_kc",
        PyArray1::from_vec(
            py,
            out.combos
                .iter()
                .map(|p| p.length_kc as i64)
                .collect::<Vec<_>>(),
        ),
    )?;
    dict.set_item(
        "mult_kc",
        PyArray1::from_vec(py, out.combos.iter().map(|p| p.mult_kc).collect::<Vec<_>>()),
    )?;
    Ok(dict)
}

// --- WASM Bindings ---

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct SmiResult {
    pub values: Vec<f64>, // [squeeze..., momentum..., signal...]
    pub rows: usize,      // 3
    pub cols: usize,      // len
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn squeeze_momentum_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    length_bb: usize,
    mult_bb: f64,
    length_kc: usize,
    mult_kc: f64,
) -> Result<Vec<f64>, JsValue> {
    let n = close.len();
    let mut sq = vec![f64::NAN; n];
    let mut mo = vec![f64::NAN; n];
    let mut si = vec![f64::NAN; n];
    let params = SqueezeMomentumParams {
        length_bb: Some(length_bb),
        mult_bb: Some(mult_bb),
        length_kc: Some(length_kc),
        mult_kc: Some(mult_kc),
    };
    let input = SqueezeMomentumInput::from_slices(high, low, close, params);
    squeeze_momentum_into_slices(&mut sq, &mut mo, &mut si, &input, detect_best_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let mut values = Vec::with_capacity(3 * n);
    values.extend_from_slice(&sq);
    values.extend_from_slice(&mo);
    values.extend_from_slice(&si);
    Ok(values)
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct SmiBatchConfig {
    pub length_bb_range: (usize, usize, usize),
    pub mult_bb_range: (f64, f64, f64),
    pub length_kc_range: (usize, usize, usize),
    pub mult_kc_range: (f64, f64, f64),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct SmiBatchJsOutput {
    pub values: Vec<f64>, // flattened momentum values only (rows x cols)
    pub rows: usize,      // number of parameter combinations
    pub cols: usize,      // data length
    pub length_bb: Vec<usize>,
    pub mult_bb: Vec<f64>,
    pub length_kc: Vec<usize>,
    pub mult_kc: Vec<f64>,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "squeeze_momentum_batch")]
pub fn squeeze_momentum_batch(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let cfg: SmiBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
    let sweep = SqueezeMomentumBatchRange {
        length_bb: cfg.length_bb_range,
        mult_bb: cfg.mult_bb_range,
        length_kc: cfg.length_kc_range,
        mult_kc: cfg.mult_kc_range,
    };
    let out =
        squeeze_momentum_batch_with_kernel(high, low, close, &sweep, detect_best_batch_kernel())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Extract parameter arrays from combos
    let mut length_bb = Vec::with_capacity(out.combos.len());
    let mut mult_bb = Vec::with_capacity(out.combos.len());
    let mut length_kc = Vec::with_capacity(out.combos.len());
    let mut mult_kc = Vec::with_capacity(out.combos.len());

    for combo in &out.combos {
        length_bb.push(combo.length_bb);
        mult_bb.push(combo.mult_bb);
        length_kc.push(combo.length_kc);
        mult_kc.push(combo.mult_kc);
    }

    let js = SmiBatchJsOutput {
        values: out.momentum, // Only return momentum values
        rows: out.rows,
        cols: out.cols,
        length_bb,
        mult_bb,
        length_kc,
        mult_kc,
    };
    serde_wasm_bindgen::to_value(&js)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn squeeze_momentum_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    vec.resize(len, f64::NAN);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn squeeze_momentum_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn squeeze_momentum_into(
    input_ptr: *const f64,
    sq_ptr: *mut f64,
    mo_ptr: *mut f64,
    si_ptr: *mut f64,
    len: usize,
    length_bb: usize,
    mult_bb: f64,
    length_kc: usize,
    mult_kc: f64,
) -> Result<(), JsValue> {
    if [
        input_ptr as usize,
        sq_ptr as usize,
        mo_ptr as usize,
        si_ptr as usize,
    ]
    .iter()
    .any(|&p| p == 0)
    {
        return Err(JsValue::from_str("null pointer"));
    }
    unsafe {
        let input = core::slice::from_raw_parts(input_ptr, len * 3);
        let h = &input[0..len];
        let l = &input[len..len * 2];
        let c = &input[len * 2..len * 3];
        let sq = core::slice::from_raw_parts_mut(sq_ptr, len);
        let mo = core::slice::from_raw_parts_mut(mo_ptr, len);
        let si = core::slice::from_raw_parts_mut(si_ptr, len);
        let params = SqueezeMomentumParams {
            length_bb: Some(length_bb),
            mult_bb: Some(mult_bb),
            length_kc: Some(length_kc),
            mult_kc: Some(mult_kc),
        };
        let input = SqueezeMomentumInput::from_slices(h, l, c, params);
        squeeze_momentum_into_slices(sq, mo, si, &input, detect_best_kernel())
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

// --- Tests: Parity with alma.rs, for all kernels, errors, accuracy, etc. ---

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn check_smi_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = SqueezeMomentumParams {
            length_bb: None,
            mult_bb: None,
            length_kc: None,
            mult_kc: None,
        };
        let input = SqueezeMomentumInput::from_candles(&candles, params);
        let output = squeeze_momentum_with_kernel(&input, kernel)?;
        assert_eq!(output.squeeze.len(), candles.close.len());
        Ok(())
    }

    fn check_smi_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = SqueezeMomentumInput::with_default_candles(&candles);
        let output = squeeze_momentum_with_kernel(&input, kernel)?;
        let expected_last_five = [-170.9, -155.4, -65.3, -61.1, -178.1];
        let n = output.momentum.len();
        let start = n.saturating_sub(5);
        for (i, &val) in output.momentum[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-1,
                "[{}] SMI {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_smi_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = SqueezeMomentumInput::with_default_candles(&candles);
        let output = squeeze_momentum_with_kernel(&input, kernel)?;
        assert_eq!(output.squeeze.len(), candles.close.len());
        Ok(())
    }

    fn check_smi_zero_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let h = [10.0, 20.0, 30.0];
        let l = [10.0, 20.0, 30.0];
        let c = [10.0, 20.0, 30.0];
        let params = SqueezeMomentumParams {
            length_bb: Some(0),
            mult_bb: None,
            length_kc: Some(0),
            mult_kc: None,
        };
        let input = SqueezeMomentumInput::from_slices(&h, &l, &c, params);
        let res = squeeze_momentum_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] SMI should fail with zero length",
            test_name
        );
        Ok(())
    }

    fn check_smi_length_exceeds(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let h = [10.0, 20.0, 30.0];
        let l = [10.0, 20.0, 30.0];
        let c = [10.0, 20.0, 30.0];
        let params = SqueezeMomentumParams {
            length_bb: Some(10),
            mult_bb: None,
            length_kc: Some(10),
            mult_kc: None,
        };
        let input = SqueezeMomentumInput::from_slices(&h, &l, &c, params);
        let res = squeeze_momentum_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] SMI should fail with length exceeding",
            test_name
        );
        Ok(())
    }

    fn check_smi_all_nan(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let h = [f64::NAN, f64::NAN, f64::NAN];
        let l = [f64::NAN, f64::NAN, f64::NAN];
        let c = [f64::NAN, f64::NAN, f64::NAN];
        let params = SqueezeMomentumParams::default();
        let input = SqueezeMomentumInput::from_slices(&h, &l, &c, params);
        let res = squeeze_momentum_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] SMI should fail with all NaN", test_name);
        Ok(())
    }

    fn check_smi_inconsistent_lengths(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let h = [1.0, 2.0, 3.0];
        let l = [1.0, 2.0];
        let c = [1.0, 2.0, 3.0];
        let params = SqueezeMomentumParams::default();
        let input = SqueezeMomentumInput::from_slices(&h, &l, &c, params);
        let res = squeeze_momentum_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] SMI should fail with inconsistent data lengths",
            test_name
        );
        Ok(())
    }

    fn check_smi_minimum_data(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let h = [10.0, 12.0, 14.0];
        let l = [5.0, 6.0, 7.0];
        let c = [7.0, 11.0, 10.0];
        let params = SqueezeMomentumParams {
            length_bb: Some(5),
            mult_bb: Some(2.0),
            length_kc: Some(5),
            mult_kc: Some(1.5),
        };
        let input = SqueezeMomentumInput::from_slices(&h, &l, &c, params);
        let result = squeeze_momentum_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_smi_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Define comprehensive parameter combinations
        let test_params = vec![
            // Default parameters
            SqueezeMomentumParams::default(),
            // Minimum periods
            SqueezeMomentumParams {
                length_bb: Some(2),
                mult_bb: Some(1.0),
                length_kc: Some(2),
                mult_kc: Some(1.0),
            },
            // Small periods with default multipliers
            SqueezeMomentumParams {
                length_bb: Some(5),
                mult_bb: Some(2.0),
                length_kc: Some(5),
                mult_kc: Some(1.5),
            },
            // Medium periods with various multipliers
            SqueezeMomentumParams {
                length_bb: Some(10),
                mult_bb: Some(1.5),
                length_kc: Some(10),
                mult_kc: Some(2.0),
            },
            SqueezeMomentumParams {
                length_bb: Some(14),
                mult_bb: Some(2.5),
                length_kc: Some(14),
                mult_kc: Some(1.0),
            },
            // Standard periods with edge multipliers
            SqueezeMomentumParams {
                length_bb: Some(20),
                mult_bb: Some(0.5),
                length_kc: Some(20),
                mult_kc: Some(0.5),
            },
            SqueezeMomentumParams {
                length_bb: Some(20),
                mult_bb: Some(3.0),
                length_kc: Some(20),
                mult_kc: Some(3.0),
            },
            // Large periods
            SqueezeMomentumParams {
                length_bb: Some(50),
                mult_bb: Some(2.0),
                length_kc: Some(50),
                mult_kc: Some(1.5),
            },
            // Very large periods
            SqueezeMomentumParams {
                length_bb: Some(100),
                mult_bb: Some(1.5),
                length_kc: Some(100),
                mult_kc: Some(2.0),
            },
            // Asymmetric periods
            SqueezeMomentumParams {
                length_bb: Some(10),
                mult_bb: Some(2.0),
                length_kc: Some(20),
                mult_kc: Some(1.5),
            },
            SqueezeMomentumParams {
                length_bb: Some(30),
                mult_bb: Some(1.5),
                length_kc: Some(15),
                mult_kc: Some(2.5),
            },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = SqueezeMomentumInput::from_candles(&candles, params.clone());
            let output = squeeze_momentum_with_kernel(&input, kernel)?;

            // Check squeeze values
            for (i, &val) in output.squeeze.iter().enumerate() {
                if val.is_nan() {
                    continue; // NaN values are expected during warmup
                }

                let bits = val.to_bits();

                // Check all three poison patterns
                if bits == 0x11111111_11111111 {
                    panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} in squeeze \
						 with params: length_bb={}, mult_bb={}, length_kc={}, mult_kc={} (param set {})",
						test_name, val, bits, i,
						params.length_bb.unwrap_or(20),
						params.mult_bb.unwrap_or(2.0),
						params.length_kc.unwrap_or(20),
						params.mult_kc.unwrap_or(1.5),
						param_idx
					);
                }

                if bits == 0x22222222_22222222 {
                    panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} in squeeze \
						 with params: length_bb={}, mult_bb={}, length_kc={}, mult_kc={} (param set {})",
						test_name, val, bits, i,
						params.length_bb.unwrap_or(20),
						params.mult_bb.unwrap_or(2.0),
						params.length_kc.unwrap_or(20),
						params.mult_kc.unwrap_or(1.5),
						param_idx
					);
                }

                if bits == 0x33333333_33333333 {
                    panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} in squeeze \
						 with params: length_bb={}, mult_bb={}, length_kc={}, mult_kc={} (param set {})",
						test_name, val, bits, i,
						params.length_bb.unwrap_or(20),
						params.mult_bb.unwrap_or(2.0),
						params.length_kc.unwrap_or(20),
						params.mult_kc.unwrap_or(1.5),
						param_idx
					);
                }
            }

            // Check momentum values
            for (i, &val) in output.momentum.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} in momentum \
						 with params: length_bb={}, mult_bb={}, length_kc={}, mult_kc={} (param set {})",
						test_name, val, bits, i,
						params.length_bb.unwrap_or(20),
						params.mult_bb.unwrap_or(2.0),
						params.length_kc.unwrap_or(20),
						params.mult_kc.unwrap_or(1.5),
						param_idx
					);
                }

                if bits == 0x22222222_22222222 {
                    panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} in momentum \
						 with params: length_bb={}, mult_bb={}, length_kc={}, mult_kc={} (param set {})",
						test_name, val, bits, i,
						params.length_bb.unwrap_or(20),
						params.mult_bb.unwrap_or(2.0),
						params.length_kc.unwrap_or(20),
						params.mult_kc.unwrap_or(1.5),
						param_idx
					);
                }

                if bits == 0x33333333_33333333 {
                    panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} in momentum \
						 with params: length_bb={}, mult_bb={}, length_kc={}, mult_kc={} (param set {})",
						test_name, val, bits, i,
						params.length_bb.unwrap_or(20),
						params.mult_bb.unwrap_or(2.0),
						params.length_kc.unwrap_or(20),
						params.mult_kc.unwrap_or(1.5),
						param_idx
					);
                }
            }

            // Check momentum_signal values
            for (i, &val) in output.momentum_signal.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} in momentum_signal \
						 with params: length_bb={}, mult_bb={}, length_kc={}, mult_kc={} (param set {})",
						test_name, val, bits, i,
						params.length_bb.unwrap_or(20),
						params.mult_bb.unwrap_or(2.0),
						params.length_kc.unwrap_or(20),
						params.mult_kc.unwrap_or(1.5),
						param_idx
					);
                }

                if bits == 0x22222222_22222222 {
                    panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} in momentum_signal \
						 with params: length_bb={}, mult_bb={}, length_kc={}, mult_kc={} (param set {})",
						test_name, val, bits, i,
						params.length_bb.unwrap_or(20),
						params.mult_bb.unwrap_or(2.0),
						params.length_kc.unwrap_or(20),
						params.mult_kc.unwrap_or(1.5),
						param_idx
					);
                }

                if bits == 0x33333333_33333333 {
                    panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} in momentum_signal \
						 with params: length_bb={}, mult_bb={}, length_kc={}, mult_kc={} (param set {})",
						test_name, val, bits, i,
						params.length_bb.unwrap_or(20),
						params.mult_bb.unwrap_or(2.0),
						params.length_kc.unwrap_or(20),
						params.mult_kc.unwrap_or(1.5),
						param_idx
					);
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_smi_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(()) // No-op in release builds
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_squeeze_momentum_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Strategy for generating realistic OHLC data
        let strat = (2usize..=50).prop_flat_map(|max_period| {
            let data_len = max_period * 2 + 50; // Ensure sufficient data length
            (
                // Generate base price and variations for OHLC
                prop::collection::vec(
                    (100f64..10000f64).prop_filter("finite", |x| x.is_finite()),
                    data_len,
                ),
                // length_bb: 2 to max_period
                2usize..=max_period.min(30),
                // mult_bb: 0.5 to 3.0
                0.5f64..3.0f64,
                // length_kc: 2 to max_period
                2usize..=max_period.min(30),
                // mult_kc: 0.5 to 3.0
                0.5f64..3.0f64,
            )
        });

        proptest::test_runner::TestRunner::default()
            .run(
                &strat,
                |(base_prices, length_bb, mult_bb, length_kc, mult_kc)| {
                    // Generate realistic OHLC data from base prices
                    let n = base_prices.len();
                    let mut high = Vec::with_capacity(n);
                    let mut low = Vec::with_capacity(n);
                    let mut close = Vec::with_capacity(n);

                    // Check if all base prices are the same (flat market)
                    let is_flat = base_prices.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10);

                    for &base in &base_prices {
                        if is_flat {
                            // For flat markets, keep prices identical
                            high.push(base);
                            low.push(base);
                            close.push(base);
                        } else {
                            // Create realistic OHLC relationships
                            // Simple and realistic: high > close > low with consistent variation
                            let variation = base * 0.01; // 1% variation for more realistic daily moves
                            let h = base + variation;
                            let l = base - variation;
                            let c = base + variation * 0.2; // Close slightly above midpoint (bullish bias)

                            high.push(h);
                            low.push(l);
                            close.push(c);
                        }
                    }

                    let params = SqueezeMomentumParams {
                        length_bb: Some(length_bb),
                        mult_bb: Some(mult_bb),
                        length_kc: Some(length_kc),
                        mult_kc: Some(mult_kc),
                    };
                    let input =
                        SqueezeMomentumInput::from_slices(&high, &low, &close, params.clone());

                    // Get output from tested kernel
                    let output = squeeze_momentum_with_kernel(&input, kernel)?;

                    // Get reference output from scalar kernel
                    let ref_output = squeeze_momentum_with_kernel(&input, Kernel::Scalar)?;

                    // Property 1: Output length matches input length
                    prop_assert_eq!(output.squeeze.len(), n, "Squeeze length mismatch");
                    prop_assert_eq!(output.momentum.len(), n, "Momentum length mismatch");
                    prop_assert_eq!(
                        output.momentum_signal.len(),
                        n,
                        "Momentum signal length mismatch"
                    );

                    // Property 2: Warmup period validation
                    // Different outputs have different warmup periods
                    let squeeze_warmup = length_bb.max(length_kc).saturating_sub(1);
                    let momentum_warmup = length_kc.saturating_sub(1);
                    let signal_warmup = length_kc.saturating_sub(1) + 1; // +1 for the lag

                    // Check squeeze warmup
                    for i in 0..squeeze_warmup.min(n) {
                        prop_assert!(
                            output.squeeze[i].is_nan(),
                            "Expected NaN in squeeze warmup at index {}",
                            i
                        );
                    }

                    // Check momentum warmup
                    for i in 0..momentum_warmup.min(n) {
                        prop_assert!(
                            output.momentum[i].is_nan(),
                            "Expected NaN in momentum warmup at index {}",
                            i
                        );
                    }

                    // Check momentum_signal warmup
                    for i in 0..signal_warmup.min(n) {
                        prop_assert!(
                            output.momentum_signal[i].is_nan(),
                            "Expected NaN in momentum_signal warmup at index {}",
                            i
                        );
                    }

                    // Property 3: Squeeze values are only -1.0, 0.0, or 1.0
                    for (i, &val) in output.squeeze.iter().enumerate() {
                        if !val.is_nan() {
                            prop_assert!(
                                val == -1.0 || val == 0.0 || val == 1.0,
                                "Invalid squeeze value {} at index {}",
                                val,
                                i
                            );
                        }
                    }

                    // Property 4: Momentum signal values are only -2.0, -1.0, 1.0, or 2.0 (or NaN)
                    for (i, &val) in output.momentum_signal.iter().enumerate() {
                        if !val.is_nan() {
                            prop_assert!(
                                val == -2.0 || val == -1.0 || val == 1.0 || val == 2.0,
                                "Invalid momentum_signal value {} at index {}",
                                val,
                                i
                            );
                        }
                    }

                    // Property 5: Kernel consistency - compare with scalar implementation
                    for i in 0..n {
                        // Check squeeze consistency
                        let sq = output.squeeze[i];
                        let ref_sq = ref_output.squeeze[i];
                        if sq.is_finite() && ref_sq.is_finite() {
                            prop_assert!(
                                (sq - ref_sq).abs() < 1e-9,
                                "Squeeze mismatch at index {}: {} vs {}",
                                i,
                                sq,
                                ref_sq
                            );
                        } else {
                            prop_assert_eq!(
                                sq.is_nan(),
                                ref_sq.is_nan(),
                                "NaN mismatch in squeeze at index {}",
                                i
                            );
                        }

                        // Check momentum consistency with ULP tolerance
                        let mom = output.momentum[i];
                        let ref_mom = ref_output.momentum[i];
                        if mom.is_finite() && ref_mom.is_finite() {
                            let mom_bits = mom.to_bits();
                            let ref_bits = ref_mom.to_bits();
                            let ulp_diff = mom_bits.abs_diff(ref_bits);

                            prop_assert!(
                                (mom - ref_mom).abs() <= 1e-9 || ulp_diff <= 5,
                                "Momentum mismatch at index {}: {} vs {} (ULP={})",
                                i,
                                mom,
                                ref_mom,
                                ulp_diff
                            );
                        } else {
                            prop_assert_eq!(
                                mom.is_nan(),
                                ref_mom.is_nan(),
                                "NaN mismatch in momentum at index {}",
                                i
                            );
                        }

                        // Check momentum_signal consistency
                        let sig = output.momentum_signal[i];
                        let ref_sig = ref_output.momentum_signal[i];
                        if sig.is_finite() && ref_sig.is_finite() {
                            prop_assert!(
                                (sig - ref_sig).abs() < 1e-9,
                                "Momentum signal mismatch at index {}: {} vs {}",
                                i,
                                sig,
                                ref_sig
                            );
                        } else {
                            prop_assert_eq!(
                                sig.is_nan(),
                                ref_sig.is_nan(),
                                "NaN mismatch in momentum_signal at index {}",
                                i
                            );
                        }
                    }

                    // Property 6: Momentum reasonableness check
                    // Since momentum uses linear regression, it can extrapolate beyond the immediate price range
                    // We'll use a check based on the overall data range
                    let overall_max = high.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let overall_min = low.iter().cloned().fold(f64::INFINITY, f64::min);
                    let overall_range = overall_max - overall_min;

                    for i in momentum_warmup..n {
                        if output.momentum[i].is_finite() {
                            if overall_range > 1e-10 {
                                // Momentum should be within a reasonable multiple of the overall price range
                                // Linear regression can extrapolate but should remain reasonable
                                prop_assert!(
								output.momentum[i].abs() <= overall_range * 5.0,
								"Momentum {} exceeds reasonable bounds at index {} (overall range: {})",
								output.momentum[i], i, overall_range
							);
                            } else {
                                // For flat markets (range near zero), momentum should be very small
                                prop_assert!(
                                    output.momentum[i].abs() < 1e-4,
                                    "Momentum {} should be near zero for flat market at index {}",
                                    output.momentum[i],
                                    i
                                );
                            }
                        }
                    }

                    Ok(())
                },
            )
            .unwrap();

        Ok(())
    }

    macro_rules! generate_all_smi_tests {
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

    generate_all_smi_tests!(
        check_smi_partial_params,
        check_smi_accuracy,
        check_smi_default_candles,
        check_smi_zero_length,
        check_smi_length_exceeds,
        check_smi_all_nan,
        check_smi_inconsistent_lengths,
        check_smi_minimum_data,
        check_smi_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_smi_tests!(check_squeeze_momentum_property);

    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = SqueezeMomentumBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c)?;
        let def = SqueezeMomentumBatchParams {
            length_bb: 20,
            mult_bb: 2.0,
            length_kc: 20,
            mult_kc: 1.5,
        };
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test various parameter sweep configurations
        let test_configs = vec![
            // (length_bb_start, length_bb_end, length_bb_step, mult_bb_start, mult_bb_end, mult_bb_step,
            //  length_kc_start, length_kc_end, length_kc_step, mult_kc_start, mult_kc_end, mult_kc_step)
            (2, 10, 2, 1.0, 2.0, 0.5, 2, 10, 2, 1.0, 2.0, 0.5), // Small periods
            (5, 25, 5, 2.0, 2.0, 0.0, 5, 25, 5, 1.5, 1.5, 0.0), // Medium periods, static multipliers
            (10, 10, 0, 1.0, 3.0, 0.5, 10, 10, 0, 1.0, 3.0, 0.5), // Static periods, varying multipliers
            (2, 5, 1, 1.5, 1.5, 0.0, 2, 5, 1, 2.0, 2.0, 0.0),     // Dense small range
            (30, 60, 15, 2.0, 2.0, 0.0, 30, 60, 15, 1.5, 1.5, 0.0), // Large periods
            (20, 30, 5, 1.0, 2.5, 0.5, 15, 25, 5, 1.0, 2.0, 0.5), // Mixed ranges
            (8, 12, 1, 0.5, 3.0, 0.5, 8, 12, 1, 0.5, 2.5, 0.5), // Dense medium range with wide multipliers
        ];

        for (
            cfg_idx,
            &(
                lbb_start,
                lbb_end,
                lbb_step,
                mbb_start,
                mbb_end,
                mbb_step,
                lkc_start,
                lkc_end,
                lkc_step,
                mkc_start,
                mkc_end,
                mkc_step,
            ),
        ) in test_configs.iter().enumerate()
        {
            let output = SqueezeMomentumBatchBuilder::new()
                .kernel(kernel)
                .length_bb_range(lbb_start, lbb_end, lbb_step)
                .mult_bb_range(mbb_start, mbb_end, mbb_step)
                .length_kc_range(lkc_start, lkc_end, lkc_step)
                .mult_kc_range(mkc_start, mkc_end, mkc_step)
                .apply_candles(&c)?;

            // Check all three output arrays
            for (name, values) in [
                ("squeeze", &output.squeeze),
                ("momentum", &output.momentum),
                ("signal", &output.signal),
            ] {
                for (idx, &val) in values.iter().enumerate() {
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
							in {} at row {} col {} (flat index {}) with params: length_bb={}, mult_bb={}, length_kc={}, mult_kc={}",
							test,
							cfg_idx,
							val,
							bits,
							name,
							row,
							col,
							idx,
							combo.length_bb,
							combo.mult_bb,
							combo.length_kc,
							combo.mult_kc
						);
                    }

                    if bits == 0x22222222_22222222 {
                        panic!(
							"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
							in {} at row {} col {} (flat index {}) with params: length_bb={}, mult_bb={}, length_kc={}, mult_kc={}",
							test,
							cfg_idx,
							val,
							bits,
							name,
							row,
							col,
							idx,
							combo.length_bb,
							combo.mult_bb,
							combo.length_kc,
							combo.mult_kc
						);
                    }

                    if bits == 0x33333333_33333333 {
                        panic!(
							"[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
							in {} at row {} col {} (flat index {}) with params: length_bb={}, mult_bb={}, length_kc={}, mult_kc={}",
							test,
							cfg_idx,
							val,
							bits,
							name,
							row,
							col,
							idx,
							combo.length_bb,
							combo.mult_bb,
							combo.length_kc,
							combo.mult_kc
						);
                    }
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
