//! # Volatility Adjusted Moving Average (VAMA)
//!
//! VAMA is a moving average that adjusts based on market volatility. It uses an EMA as the base
//! and adds an adjustment based on the average of highest and lowest deviations from the EMA.
//!
//! ## Parameters
//! - **base_period**: Period for the base EMA (default: 113)
//! - **vol_period**: Period for volatility calculation (default: 51)
//! - **smoothing**: Whether to smooth the VAMA (default: true)
//! - **smooth_type**: Type of smoothing - 1=SMA, 2=EMA, 3=WMA (default: 3)
//! - **smooth_period**: Smoothing period (default: 5)
//!
//! ## Returns
//! - **`Ok(VamaOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(VamaError)`** otherwise.
//!
//! ## Developer Notes
//! - Status: scalar core optimized to O(n) via monotonic deques (rolling max/min of deviations).
//! - SIMD: no separate AVX2/AVX512 kernels for the core — sequential window dependency limits gains.
//!          EMA/WMA/SMA reuse existing kernels under `Kernel` selection.
//! - Batch: row-specific EMA precompute was evaluated; kept disabled by default — per-row parallelism
//!          with per-row EMA was faster/more stable in benches. Revisit only with stronger evidence.
//! - Memory: uses zero-copy helpers (alloc_with_nan_prefix); no O(N) temporaries beyond outputs.

// ==================== IMPORTS SECTION ====================
// Feature-gated imports for Python bindings
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::alma::DeviceArrayF32Py;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;

// Feature-gated imports for WASM bindings
#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

// Core imports
use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;

// Import other indicators
use crate::indicators::moving_averages::ema::{ema, ema_into_slice, EmaInput, EmaParams};
use crate::indicators::moving_averages::sma::{sma, sma_into_slice, SmaInput, SmaParams};
use crate::indicators::moving_averages::wma::{wma, wma_into_slice, WmaInput, WmaParams};

// Parallel processing support
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

// Standard library imports
use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

// ==================== TRAIT IMPLEMENTATIONS ====================
impl<'a> AsRef<[f64]> for VamaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            VamaData::Slice(slice) => slice,
            VamaData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

// ==================== DATA STRUCTURES ====================
/// Input data enum supporting both raw slices and candle data
#[derive(Debug, Clone)]
pub enum VamaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

/// Output structure containing calculated values
#[derive(Debug, Clone)]
pub struct VamaOutput {
    pub values: Vec<f64>,
}

/// Parameters structure with optional fields for defaults
#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct VamaParams {
    pub base_period: Option<usize>,
    pub vol_period: Option<usize>,
    pub smoothing: Option<bool>,
    pub smooth_type: Option<usize>,
    pub smooth_period: Option<usize>,
}

impl Default for VamaParams {
    fn default() -> Self {
        Self {
            base_period: Some(113),
            vol_period: Some(51),
            smoothing: Some(true),
            smooth_type: Some(3), // WMA
            smooth_period: Some(5),
        }
    }
}

/// Main input structure combining data and parameters
#[derive(Debug, Clone)]
pub struct VamaInput<'a> {
    pub data: VamaData<'a>,
    pub params: VamaParams,
}

impl<'a> VamaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: VamaParams) -> Self {
        Self {
            data: VamaData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }

    #[inline]
    pub fn from_slice(sl: &'a [f64], p: VamaParams) -> Self {
        Self {
            data: VamaData::Slice(sl),
            params: p,
        }
    }

    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", VamaParams::default())
    }

    #[inline]
    pub fn get_base_period(&self) -> usize {
        self.params.base_period.unwrap_or(113)
    }

    #[inline]
    pub fn get_vol_period(&self) -> usize {
        self.params.vol_period.unwrap_or(51)
    }

    #[inline]
    pub fn get_smoothing(&self) -> bool {
        self.params.smoothing.unwrap_or(true)
    }

    #[inline]
    pub fn get_smooth_type(&self) -> usize {
        self.params.smooth_type.unwrap_or(3)
    }

    #[inline]
    pub fn get_smooth_period(&self) -> usize {
        self.params.smooth_period.unwrap_or(5)
    }
}

// ==================== BUILDER PATTERN ====================
/// Builder for ergonomic API usage
#[derive(Copy, Clone, Debug)]
pub struct VamaBuilder {
    base_period: Option<usize>,
    vol_period: Option<usize>,
    smoothing: Option<bool>,
    smooth_type: Option<usize>,
    smooth_period: Option<usize>,
    kernel: Kernel,
}

impl Default for VamaBuilder {
    fn default() -> Self {
        Self {
            base_period: None,
            vol_period: None,
            smoothing: None,
            smooth_type: None,
            smooth_period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl VamaBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn base_period(mut self, val: usize) -> Self {
        self.base_period = Some(val);
        self
    }

    #[inline(always)]
    pub fn vol_period(mut self, val: usize) -> Self {
        self.vol_period = Some(val);
        self
    }

    #[inline(always)]
    pub fn smoothing(mut self, val: bool) -> Self {
        self.smoothing = Some(val);
        self
    }

    #[inline(always)]
    pub fn smooth_type(mut self, val: usize) -> Self {
        self.smooth_type = Some(val);
        self
    }

    #[inline(always)]
    pub fn smooth_period(mut self, val: usize) -> Self {
        self.smooth_period = Some(val);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<VamaOutput, VamaError> {
        let p = VamaParams {
            base_period: self.base_period,
            vol_period: self.vol_period,
            smoothing: self.smoothing,
            smooth_type: self.smooth_type,
            smooth_period: self.smooth_period,
        };
        let i = VamaInput::from_candles(c, "close", p);
        vama_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<VamaOutput, VamaError> {
        let p = VamaParams {
            base_period: self.base_period,
            vol_period: self.vol_period,
            smoothing: self.smoothing,
            smooth_type: self.smooth_type,
            smooth_period: self.smooth_period,
        };
        let i = VamaInput::from_slice(d, p);
        vama_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<VamaStream, VamaError> {
        let p = VamaParams {
            base_period: self.base_period,
            vol_period: self.vol_period,
            smoothing: self.smoothing,
            smooth_type: self.smooth_type,
            smooth_period: self.smooth_period,
        };
        VamaStream::try_new(p)
    }
}

// ==================== ERROR HANDLING ====================
#[derive(Debug, Error)]
pub enum VamaError {
    #[error("vama: Input data slice is empty.")]
    EmptyInputData,

    #[error("vama: All values are NaN.")]
    AllValuesNaN,

    #[error("vama: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },

    #[error("vama: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },

    #[error("vama: Invalid smooth type: {smooth_type}. Must be 1 (SMA), 2 (EMA), or 3 (WMA)")]
    InvalidSmoothType { smooth_type: usize },

    #[error("vama: EMA calculation failed: {0}")]
    EmaError(String),

    #[error("vama: SMA calculation failed: {0}")]
    SmaError(String),

    #[error("vama: WMA calculation failed: {0}")]
    WmaError(String),
}

// ==================== CORE COMPUTATION FUNCTIONS ====================
/// Core VAMA pass writing unsmoothed values directly into `out`.
///
/// Scalar O(n) implementation using monotonic deques to track rolling
/// max/min of deviation = data[i] - ema[i]. This preserves the exact
/// warmup and NaN semantics of the previous implementation while
/// reducing the inner complexity from O(n · vol_period) to O(n).
#[inline(always)]
fn vama_core_into(
    data: &[f64],
    base_period: usize,
    vol_period: usize,
    first: usize,
    kernel: Kernel,
    out: &mut [f64],
) -> Result<(), VamaError> {
    let len = data.len();

    // 1) Base EMA into temp buffer with NaN prefix
    let mut ema_values = alloc_with_nan_prefix(len, first + base_period - 1);
    let ema_input = EmaInput::from_slice(
        data,
        EmaParams {
            period: Some(base_period),
        },
    );
    ema_into_slice(&mut ema_values, &ema_input, kernel)
        .map_err(|e| VamaError::EmaError(e.to_string()))?;

    // 2) Sliding-extrema using ring-buffer deques (amortized O(1) per step)
    let warmup = first + base_period.max(vol_period) - 1;
    if len <= warmup {
        // out[..warmup] is already prefilled with NaNs by the caller
        return Ok(());
    }

    // Deques for rolling max/min of deviation over `vol_period` window
    let cap = vol_period; // capacity equals window length
    let mut idx_max = vec![0usize; cap];
    let mut val_max = vec![0.0f64; cap];
    let mut head_max = 0usize;
    let mut tail_max = 0usize;

    let mut idx_min = vec![0usize; cap];
    let mut val_min = vec![0.0f64; cap];
    let mut head_min = 0usize;
    let mut tail_min = 0usize;

    // Iterate across the series, updating deques; write only after warmup
    for i in first..len {
        let e = ema_values[i];
        let x = data[i];

        // Respect the initial `first` offset: early windows are shorter
        let window_len = vol_period.min(i + 1 - first);
        let window_start = i + 1 - window_len;

        // Expire out-of-window entries for MAX
        while head_max != tail_max && idx_max[head_max] < window_start {
            head_max += 1;
            if head_max == cap {
                head_max = 0;
            }
        }
        // Expire out-of-window entries for MIN
        while head_min != tail_min && idx_min[head_min] < window_start {
            head_min += 1;
            if head_min == cap {
                head_min = 0;
            }
        }

        // Push current deviation if valid (skip NaNs to mirror original semantics)
        if !(e.is_nan() || x.is_nan()) {
            let d = x - e;

            // Maintain decreasing deque for MAX
            while head_max != tail_max {
                let last = if tail_max == 0 { cap - 1 } else { tail_max - 1 };
                if val_max[last] <= d {
                    tail_max = last; // pop_back
                } else {
                    break;
                }
            }
            idx_max[tail_max] = i;
            val_max[tail_max] = d;
            tail_max += 1;
            if tail_max == cap {
                tail_max = 0;
            }

            // Maintain increasing deque for MIN
            while head_min != tail_min {
                let last = if tail_min == 0 { cap - 1 } else { tail_min - 1 };
                if val_min[last] >= d {
                    tail_min = last; // pop_back
                } else {
                    break;
                }
            }
            idx_min[tail_min] = i;
            val_min[tail_min] = d;
            tail_min += 1;
            if tail_min == cap {
                tail_min = 0;
            }
        }

        // After warmup, emit value. Otherwise, caller keeps NaNs prefix.
        if i >= warmup {
            if e.is_nan() {
                out[i] = f64::NAN;
            } else if head_max != tail_max && head_min != tail_min {
                let up = val_max[head_max];
                let dn = val_min[head_min];
                // mid + 0.5*(up + dn)
                out[i] = (0.5f64).mul_add(up + dn, e);
            } else {
                // No valid deviations in the window → fallback to EMA
                out[i] = e;
            }
        }
    }
    Ok(())
}

/// Main entry point with automatic kernel detection
#[inline]
pub fn vama(input: &VamaInput) -> Result<VamaOutput, VamaError> {
    vama_with_kernel(input, Kernel::Auto)
}

/// Entry point with explicit kernel selection - zero-copy orchestration
pub fn vama_with_kernel(input: &VamaInput, kernel: Kernel) -> Result<VamaOutput, VamaError> {
    let (data, base_p, vol_p, smoothing, smooth_ty, smooth_p, first, chosen) =
        vama_prepare(input, kernel)?;
    let warmup = first + base_p.max(vol_p) - 1;

    if !smoothing {
        // Single allocation with NaN prefix pre-filled
        let mut out = alloc_with_nan_prefix(data.len(), warmup);
        vama_core_into(data, base_p, vol_p, first, chosen, &mut out)?;
        return Ok(VamaOutput { values: out });
    }

    // Two allocations for smoothing, no copies:
    // 1) Work buffer for core pass
    let mut work = alloc_with_nan_prefix(data.len(), warmup);
    vama_core_into(data, base_p, vol_p, first, chosen, &mut work)?;

    // 2) Final output buffer for smoothing result
    let mut out = alloc_with_nan_prefix(data.len(), warmup);
    match smooth_ty {
        1 => {
            let si = SmaInput::from_slice(
                &work,
                SmaParams {
                    period: Some(smooth_p),
                },
            );
            sma_into_slice(&mut out, &si, chosen)
                .map_err(|e| VamaError::SmaError(e.to_string()))?;
        }
        2 => {
            let ei = EmaInput::from_slice(
                &work,
                EmaParams {
                    period: Some(smooth_p),
                },
            );
            ema_into_slice(&mut out, &ei, chosen)
                .map_err(|e| VamaError::EmaError(e.to_string()))?;
        }
        3 => {
            let wi = WmaInput::from_slice(
                &work,
                WmaParams {
                    period: Some(smooth_p),
                },
            );
            wma_into_slice(&mut out, &wi, chosen)
                .map_err(|e| VamaError::WmaError(e.to_string()))?;
        }
        _ => unreachable!(),
    }
    Ok(VamaOutput { values: out })
}

/// Zero-copy into-slice with warmup NaNs written, no final copying
#[inline]
pub fn vama_into_slice(dst: &mut [f64], input: &VamaInput, kern: Kernel) -> Result<(), VamaError> {
    let (data, base_p, vol_p, smoothing, smooth_ty, smooth_p, first, chosen) =
        vama_prepare(input, kern)?;
    if dst.len() != data.len() {
        return Err(VamaError::InvalidPeriod {
            period: dst.len(),
            data_len: data.len(),
        });
    }

    let warmup = first + base_p.max(vol_p) - 1;

    if !smoothing {
        // Write warmup NaNs and compute directly into dst
        for v in &mut dst[..warmup] {
            *v = f64::NAN;
        }
        vama_core_into(data, base_p, vol_p, first, chosen, dst)?;
        return Ok(());
    }

    // Smoothing: one transient work buffer, final in dst
    let mut work = alloc_with_nan_prefix(data.len(), warmup);
    vama_core_into(data, base_p, vol_p, first, chosen, &mut work)?;

    for v in &mut dst[..warmup] {
        *v = f64::NAN;
    }
    match smooth_ty {
        1 => sma_into_slice(
            dst,
            &SmaInput::from_slice(
                &work,
                SmaParams {
                    period: Some(smooth_p),
                },
            ),
            chosen,
        )
        .map_err(|e| VamaError::SmaError(e.to_string()))?,
        2 => ema_into_slice(
            dst,
            &EmaInput::from_slice(
                &work,
                EmaParams {
                    period: Some(smooth_p),
                },
            ),
            chosen,
        )
        .map_err(|e| VamaError::EmaError(e.to_string()))?,
        3 => wma_into_slice(
            dst,
            &WmaInput::from_slice(
                &work,
                WmaParams {
                    period: Some(smooth_p),
                },
            ),
            chosen,
        )
        .map_err(|e| VamaError::WmaError(e.to_string()))?,
        _ => unreachable!(),
    }
    Ok(())
}

/// Prepare and validate input data
#[inline(always)]
fn vama_prepare<'a>(
    input: &'a VamaInput,
    kernel: Kernel,
) -> Result<(&'a [f64], usize, usize, bool, usize, usize, usize, Kernel), VamaError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();

    if len == 0 {
        return Err(VamaError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(VamaError::AllValuesNaN)?;

    let base_period = input.get_base_period();
    let vol_period = input.get_vol_period();
    let smoothing = input.get_smoothing();
    let smooth_type = input.get_smooth_type();
    let smooth_period = input.get_smooth_period();

    // Validation
    if base_period == 0 || base_period > len {
        return Err(VamaError::InvalidPeriod {
            period: base_period,
            data_len: len,
        });
    }
    if vol_period == 0 || vol_period > len {
        return Err(VamaError::InvalidPeriod {
            period: vol_period,
            data_len: len,
        });
    }
    if smoothing {
        if smooth_type < 1 || smooth_type > 3 {
            return Err(VamaError::InvalidSmoothType { smooth_type });
        }
    }

    let needed = base_period.max(vol_period);
    if len - first < needed {
        return Err(VamaError::NotEnoughValidData {
            needed,
            valid: len - first,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };

    Ok((
        data,
        base_period,
        vol_period,
        smoothing,
        smooth_type,
        smooth_period,
        first,
        chosen,
    ))
}

// ==================== STREAMING SUPPORT ====================
// Decision: Streaming upgraded to amortized O(1) via monotonic deques over deviations; smoothing (SMA/EMA/WMA) in O(1).
// Matches batch core and preserves warmup semantics; see alma.rs for style/perf reference.

/// Streaming calculator for real-time updates (amortized O(1) per tick).
#[derive(Debug, Clone)]
pub struct VamaStream {
    // --- config ---
    base_period: usize,
    vol_period: usize,
    smoothing: bool,
    smooth_type: usize, // 1=SMA, 2=EMA, 3=WMA
    smooth_period: usize,

    // --- base EMA state ---
    alpha: f64,     // 2/(base_period+1)
    ema: f64,       // last EMA
    have_ema: bool, // first-tick bootstrap

    // --- monotonic deques over deviations d_t = x_t - ema_t ---
    // We use fixed-size ring buffers with capacity = vol_period+1 to avoid
    // ambiguity between empty/full states when head==tail.
    dq_cap: usize,
    max_idx: Vec<usize>,
    max_val: Vec<f64>,
    max_head: usize,
    max_tail: usize,

    min_idx: Vec<usize>,
    min_val: Vec<f64>,
    min_head: usize,
    min_tail: usize,

    // --- smoothing state (core values are 'c_t') ---
    // SMA/WMA need a ring of the last p core values; EMA needs only one acc.
    sm_ring: Vec<f64>, // length = smooth_period (allocated only if smoothing)
    sm_ptr: usize,
    sm_count: usize,

    // SMA: running sum
    sm_sum: f64,

    // EMA: accumulator + alpha
    sm_alpha: f64,
    sm_ema: f64,
    have_sm_ema: bool,

    // WMA: rolling numerator and simple sum
    wma_num: f64, // N_t
    wma_den: f64, // p*(p+1)/2 precomputed

    // --- indexing / readiness ---
    index: u64,         // number of updates observed so far
    ready_after: usize, // when final (smoothed or core) output becomes available
    pub ready: bool,    // externally visible readiness (kept for API parity)
}

impl VamaStream {
    pub fn try_new(params: VamaParams) -> Result<Self, VamaError> {
        let base_period = params.base_period.unwrap_or(113);
        let vol_period = params.vol_period.unwrap_or(51);
        let smoothing = params.smoothing.unwrap_or(true);
        let smooth_type = params.smooth_type.unwrap_or(3);
        let smooth_period = params.smooth_period.unwrap_or(5);

        if base_period == 0 {
            return Err(VamaError::InvalidPeriod {
                period: base_period,
                data_len: 0,
            });
        }
        if vol_period == 0 {
            return Err(VamaError::InvalidPeriod {
                period: vol_period,
                data_len: 0,
            });
        }
        if smoothing && !(1..=3).contains(&smooth_type) {
            return Err(VamaError::InvalidSmoothType { smooth_type });
        }

        // Precompute EMA alphas
        let alpha = 2.0 / (base_period as f64 + 1.0);
        let sm_alpha = if smoothing && smooth_type == 2 {
            2.0 / (smooth_period as f64 + 1.0)
        } else {
            0.0
        };

        // Monotonic deques capacity: window length + 1 (classic trick so head==tail => empty)
        let dq_cap = vol_period + 1;
        let (max_idx, max_val) = (vec![0usize; dq_cap], vec![0.0f64; dq_cap]);
        let (min_idx, min_val) = (vec![0usize; dq_cap], vec![0.0f64; dq_cap]);

        // Smoothing buffers
        let sm_ring = if smoothing && (smooth_type == 1 || smooth_type == 3) {
            vec![0.0f64; smooth_period]
        } else {
            Vec::new()
        };

        // WMA denominator
        let wma_den = if smoothing && smooth_type == 3 {
            let p = smooth_period as f64;
            p.mul_add(p + 1.0, 0.0) * 0.5 // p*(p+1)/2 with FMA
        } else {
            0.0
        };

        // Readiness:
        // - core VAMA needs max(base_period, vol_period) samples
        // - SMA/WMA add (smooth_period - 1) more before a full window is available
        // - EMA smoothing is defined immediately once core is ready
        let core_ready = base_period.max(vol_period);
        let smooth_lag = if smoothing {
            match smooth_type {
                1 | 3 => smooth_period.saturating_sub(1),
                2 => 0,
                _ => 0,
            }
        } else {
            0
        };
        let ready_after = core_ready + smooth_lag;

        Ok(Self {
            base_period,
            vol_period,
            smoothing,
            smooth_type,
            smooth_period,

            alpha,
            ema: 0.0,
            have_ema: false,

            dq_cap,
            max_idx,
            max_val,
            max_head: 0,
            max_tail: 0,
            min_idx,
            min_val,
            min_head: 0,
            min_tail: 0,

            sm_ring,
            sm_ptr: 0,
            sm_count: 0,
            sm_sum: 0.0,

            sm_alpha,
            sm_ema: 0.0,
            have_sm_ema: false,

            wma_num: 0.0,
            wma_den,

            index: 0,
            ready_after,
            ready: false,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, x: f64) -> Option<f64> {
        let t = self.index as usize;

        // --- 1) base EMA update (fma for precision and speed)
        if !self.have_ema {
            self.ema = x;
            self.have_ema = true;
        } else {
            // ema = ema + alpha * (x - ema)
            self.ema = self.alpha.mul_add(x - self.ema, self.ema);
        }

        // --- 2) maintain monotonic deques on deviation d_t = x - ema
        // Expire indices falling out of the last vol_period samples
        let cutoff = if t + 1 > self.vol_period {
            t + 1 - self.vol_period
        } else {
            0
        };
        while self.max_head != self.max_tail && self.max_idx[self.max_head] < cutoff {
            self.max_head = (self.max_head + 1) % self.dq_cap;
        }
        while self.min_head != self.min_tail && self.min_idx[self.min_head] < cutoff {
            self.min_head = (self.min_head + 1) % self.dq_cap;
        }

        // Push current deviation if finite
        let d = x - self.ema;
        if d.is_finite() {
            // MAX deque: keep decreasing values
            while self.max_head != self.max_tail {
                let last = if self.max_tail == 0 {
                    self.dq_cap - 1
                } else {
                    self.max_tail - 1
                };
                if self.max_val[last] <= d {
                    self.max_tail = last; // pop_back
                } else {
                    break;
                }
            }
            self.max_idx[self.max_tail] = t;
            self.max_val[self.max_tail] = d;
            self.max_tail = (self.max_tail + 1) % self.dq_cap;

            // MIN deque: keep increasing values
            while self.min_head != self.min_tail {
                let last = if self.min_tail == 0 {
                    self.dq_cap - 1
                } else {
                    self.min_tail - 1
                };
                if self.min_val[last] >= d {
                    self.min_tail = last; // pop_back
                } else {
                    break;
                }
            }
            self.min_idx[self.min_tail] = t;
            self.min_val[self.min_tail] = d;
            self.min_tail = (self.min_tail + 1) % self.dq_cap;
        }

        // --- 3) form the unsmoothed core once the EMA/vol windows are ready
        let core_ready = t + 1 >= self.base_period.max(self.vol_period);
        let core = if core_ready {
            if self.max_head != self.max_tail && self.min_head != self.min_tail {
                // e + 0.5*(up + dn) using fma
                (0.5f64).mul_add(
                    self.max_val[self.max_head] + self.min_val[self.min_head],
                    self.ema,
                )
            } else {
                // Fall back to EMA if we have no valid deviations
                self.ema
            }
        } else {
            self.index = self.index.wrapping_add(1);
            return None;
        };

        // --- 4) optional smoothing in O(1)
        let out = if !self.smoothing {
            core
        } else {
            match self.smooth_type {
                // SMA(p): keep running sum over last p core values
                1 => {
                    if self.sm_ring.is_empty() {
                        core
                    } else {
                        if self.sm_count < self.smooth_period {
                            self.sm_ring[self.sm_ptr] = core;
                            self.sm_sum += core;
                            self.sm_ptr = (self.sm_ptr + 1) % self.smooth_period;
                            self.sm_count += 1;
                        } else {
                            let old = self.sm_ring[self.sm_ptr];
                            self.sm_ring[self.sm_ptr] = core;
                            self.sm_ptr = (self.sm_ptr + 1) % self.smooth_period;
                            self.sm_sum = self.sm_sum + core - old;
                        }
                        if self.sm_count < self.smooth_period {
                            // not yet full window -> mirror batch semantics: delay output
                            core
                        } else {
                            self.sm_sum / (self.smooth_period as f64)
                        }
                    }
                }

                // EMA(p): one accumulator
                2 => {
                    if !self.have_sm_ema {
                        self.sm_ema = core;
                        self.have_sm_ema = true;
                    } else {
                        self.sm_ema = self.sm_alpha.mul_add(core - self.sm_ema, self.sm_ema);
                    }
                    self.sm_ema
                }

                // WMA(p): keep simple sum S_t and weighted sum numerator N_t
                3 => {
                    if self.sm_ring.is_empty() {
                        core
                    } else {
                        if self.sm_count < self.smooth_period {
                            // growing phase: N += (k)*core, where k = sm_count+1
                            self.sm_ring[self.sm_ptr] = core;
                            let k = (self.sm_count + 1) as f64;
                            self.wma_num = k.mul_add(core, self.wma_num);
                            self.sm_sum += core;
                            self.sm_ptr = (self.sm_ptr + 1) % self.smooth_period;
                            self.sm_count += 1;
                        } else {
                            // rolling phase:
                            // N_{t+1} = N_t + p*core - S_t
                            // S_{t+1} = S_t + core - old
                            let old = self.sm_ring[self.sm_ptr];
                            let s_prev = self.sm_sum;
                            self.wma_num =
                                (self.smooth_period as f64).mul_add(core, self.wma_num) - s_prev;
                            self.sm_ring[self.sm_ptr] = core;
                            self.sm_ptr = (self.sm_ptr + 1) % self.smooth_period;
                            self.sm_sum = s_prev + core - old;
                        }
                        if self.sm_count < self.smooth_period {
                            core // delay until full window
                        } else {
                            self.wma_num / self.wma_den
                        }
                    }
                }

                _ => core,
            }
        };

        // --- 5) readiness & return
        self.index = self.index.wrapping_add(1);
        if !self.ready && (t + 1) >= self.ready_after {
            self.ready = true;
        }
        Some(out)
    }
}

// ==================== BATCH PROCESSING ====================
/// Batch processing range configuration
#[derive(Clone, Debug)]
pub struct VamaBatchRange {
    pub base_period: (usize, usize, usize),
    pub vol_period: (usize, usize, usize),
}

impl Default for VamaBatchRange {
    fn default() -> Self {
        Self {
            base_period: (100, 130, 10),
            vol_period: (40, 60, 10),
        }
    }
}

/// Expand parameter ranges to all combinations
#[inline(always)]
fn expand_grid_vama(r: &VamaBatchRange) -> Vec<VamaParams> {
    fn axis((s, e, t): (usize, usize, usize)) -> Vec<usize> {
        if t == 0 || s == e {
            return vec![s];
        }
        (s..=e).step_by(t).collect()
    }
    let bs = axis(r.base_period);
    let vs = axis(r.vol_period);
    let mut out = Vec::with_capacity(bs.len() * vs.len());
    for &b in &bs {
        for &v in &vs {
            out.push(VamaParams {
                base_period: Some(b),
                vol_period: Some(v),
                smoothing: Some(false), // batch path unsmoothed, mirrors alma.rs approach
                smooth_type: Some(3),
                smooth_period: Some(5),
            });
        }
    }
    out
}

/// Batch processing output with flat matrix
#[derive(Clone, Debug)]
pub struct VamaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<VamaParams>,
    pub rows: usize,
    pub cols: usize,
}

impl VamaBatchOutput {
    pub fn row_for_params(&self, p: &VamaParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.base_period.unwrap() == p.base_period.unwrap()
                && c.vol_period.unwrap() == p.vol_period.unwrap()
        })
    }
    pub fn values_for(&self, p: &VamaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

/// VamaBatchBuilder parity with AlmaBatchBuilder
#[derive(Clone, Debug, Default)]
pub struct VamaBatchBuilder {
    range: VamaBatchRange,
    kernel: Kernel,
}

impl VamaBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline]
    pub fn base_period_range(mut self, s: usize, e: usize, t: usize) -> Self {
        self.range.base_period = (s, e, t);
        self
    }
    #[inline]
    pub fn base_period_static(mut self, p: usize) -> Self {
        self.range.base_period = (p, p, 0);
        self
    }
    #[inline]
    pub fn vol_period_range(mut self, s: usize, e: usize, t: usize) -> Self {
        self.range.vol_period = (s, e, t);
        self
    }
    #[inline]
    pub fn vol_period_static(mut self, p: usize) -> Self {
        self.range.vol_period = (p, p, 0);
        self
    }
    pub fn apply_slice(self, data: &[f64]) -> Result<VamaBatchOutput, VamaError> {
        vama_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<VamaBatchOutput, VamaError> {
        self.apply_slice(source_type(c, src))
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<VamaBatchOutput, VamaError> {
        VamaBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn with_default_candles(c: &Candles) -> Result<VamaBatchOutput, VamaError> {
        VamaBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

/// Zero-copy batch inner allocation with NaN prefixes
#[inline(always)]
fn vama_batch_inner(
    data: &[f64],
    ranges: &VamaBatchRange,
    simd: Kernel,
    parallel: bool,
) -> Result<VamaBatchOutput, VamaError> {
    let combos = expand_grid_vama(ranges);
    let cols = data.len();
    let rows = combos.len();
    if cols == 0 {
        return Err(VamaError::EmptyInputData);
    }
    if rows == 0 {
        return Err(VamaError::InvalidPeriod {
            period: 0,
            data_len: cols,
        });
    }

    // Validate that all period combinations fit within data length
    for combo in &combos {
        let base_p = combo.base_period.unwrap_or(0);
        let vol_p = combo.vol_period.unwrap_or(0);
        if base_p > cols {
            return Err(VamaError::InvalidPeriod {
                period: base_p,
                data_len: cols,
            });
        }
        if vol_p > cols {
            return Err(VamaError::InvalidPeriod {
                period: vol_p,
                data_len: cols,
            });
        }
    }

    // allocate without zeroing; write NaN prefixes only
    let mut buf_mu = make_uninit_matrix(rows, cols);

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(VamaError::AllValuesNaN)?;
    let warm: Vec<usize> = combos
        .iter()
        .map(|p| first + p.base_period.unwrap().max(p.vol_period.unwrap()) - 1)
        .collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    // fill rows in place
    vama_batch_inner_into_with_simd(data, &combos, first, simd, out, cols, parallel)?;

    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };
    Ok(VamaBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

/// Into-existing buffer (NumPy/WASM) with warmup NaNs and SIMD mapping
#[inline(always)]
fn vama_batch_inner_into_with_simd(
    data: &[f64],
    combos: &[VamaParams],
    first: usize,
    simd: Kernel,
    out: &mut [f64],
    cols: usize,
    parallel: bool,
) -> Result<(), VamaError> {
    // warmup NaNs per row
    for (row, prm) in combos.iter().enumerate() {
        let warm = first + prm.base_period.unwrap().max(prm.vol_period.unwrap()) - 1;
        let rs = row * cols;
        for i in 0..warm.min(cols) {
            out[rs + i] = f64::NAN;
        }
    }

    let do_row = |row: usize, dst: &mut [f64]| -> Result<(), VamaError> {
        let p = &combos[row];
        vama_core_into(
            data,
            p.base_period.unwrap(),
            p.vol_period.unwrap(),
            first,
            simd,
            dst,
        )
    };

    #[cfg(not(target_arch = "wasm32"))]
    {
        if parallel {
            use rayon::prelude::*;
            out.chunks_mut(cols)
                .enumerate()
                .collect::<Vec<_>>()
                .into_par_iter()
                .try_for_each(|(r, dst)| do_row(r, dst))?;
        } else {
            for (r, dst) in out.chunks_mut(cols).enumerate() {
                do_row(r, dst)?;
            }
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        for (r, dst) in out.chunks_mut(cols).enumerate() {
            do_row(r, dst)?;
        }
    }
    Ok(())
}

/// Core VAMA step given a precomputed EMA slice. Used by batch path to share EMA across rows.
#[inline(always)]
fn vama_core_from_ema_into(
    data: &[f64],
    ema_values: &[f64],
    base_period: usize,
    vol_period: usize,
    first: usize,
    out: &mut [f64],
) -> Result<(), VamaError> {
    let len = data.len();
    debug_assert_eq!(ema_values.len(), len);

    let warmup = first + base_period.max(vol_period) - 1;
    if len <= warmup {
        return Ok(());
    }

    // Monotonic deques identical to the single-series core
    let cap = vol_period;
    let mut idx_max = vec![0usize; cap];
    let mut val_max = vec![0.0f64; cap];
    let mut head_max = 0usize;
    let mut tail_max = 0usize;

    let mut idx_min = vec![0usize; cap];
    let mut val_min = vec![0.0f64; cap];
    let mut head_min = 0usize;
    let mut tail_min = 0usize;

    for i in first..len {
        let e = ema_values[i];
        let x = data[i];

        let window_len = vol_period.min(i + 1 - first);
        let window_start = i + 1 - window_len;

        while head_max != tail_max && idx_max[head_max] < window_start {
            head_max += 1;
            if head_max == cap {
                head_max = 0;
            }
        }
        while head_min != tail_min && idx_min[head_min] < window_start {
            head_min += 1;
            if head_min == cap {
                head_min = 0;
            }
        }

        if !(e.is_nan() || x.is_nan()) {
            let d = x - e;
            while head_max != tail_max {
                let last = if tail_max == 0 { cap - 1 } else { tail_max - 1 };
                if val_max[last] <= d {
                    tail_max = last;
                } else {
                    break;
                }
            }
            idx_max[tail_max] = i;
            val_max[tail_max] = d;
            tail_max += 1;
            if tail_max == cap {
                tail_max = 0;
            }

            while head_min != tail_min {
                let last = if tail_min == 0 { cap - 1 } else { tail_min - 1 };
                if val_min[last] >= d {
                    tail_min = last;
                } else {
                    break;
                }
            }
            idx_min[tail_min] = i;
            val_min[tail_min] = d;
            tail_min += 1;
            if tail_min == cap {
                tail_min = 0;
            }
        }

        if i >= warmup {
            if e.is_nan() {
                out[i] = f64::NAN;
            } else if head_max != tail_max && head_min != tail_min {
                let up = val_max[head_max];
                let dn = val_min[head_min];
                out[i] = (0.5f64).mul_add(up + dn, e);
            } else {
                out[i] = e;
            }
        }
    }
    Ok(())
}

/// Keep the old vama_batch_inner_into for backward compatibility
#[inline(always)]
fn vama_batch_inner_into(
    data: &[f64],
    ranges: &VamaBatchRange,
    k: Kernel,
    out: &mut [f64],
    parallel: bool,
) -> Result<(), VamaError> {
    let combos = expand_grid_vama(ranges);
    if combos.is_empty() {
        return Err(VamaError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }

    let cols = data.len();
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(VamaError::AllValuesNaN)?;

    let simd = match k {
        Kernel::Auto => match detect_best_batch_kernel() {
            Kernel::Avx512Batch => Kernel::Avx512,
            Kernel::Avx2Batch => Kernel::Avx2,
            _ => Kernel::Scalar,
        },
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => k.to_non_batch(),
    };

    vama_batch_inner_into_with_simd(data, &combos, first, simd, out, cols, parallel)
}

/// Convenience batch wrapper - single-threaded
#[inline(always)]
pub fn vama_batch_slice(
    data: &[f64],
    r: &VamaBatchRange,
    k: Kernel,
) -> Result<VamaBatchOutput, VamaError> {
    vama_batch_inner(
        data,
        r,
        match k {
            Kernel::Auto => detect_best_kernel(),
            x => x,
        },
        false,
    )
}

/// Convenience batch wrapper - parallel
#[inline(always)]
pub fn vama_batch_par_slice(
    data: &[f64],
    r: &VamaBatchRange,
    k: Kernel,
) -> Result<VamaBatchOutput, VamaError> {
    vama_batch_inner(
        data,
        r,
        match k {
            Kernel::Auto => detect_best_kernel(),
            x => x,
        },
        true,
    )
}

/// Batch processing with parameter sweep using flat matrix
pub fn vama_batch_with_kernel(
    data: &[f64],
    ranges: &VamaBatchRange,
    k: Kernel,
) -> Result<VamaBatchOutput, VamaError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        Kernel::ScalarBatch => Kernel::ScalarBatch, // Explicitly handle ScalarBatch
        other if other.is_batch() => other,
        _ => {
            return Err(VamaError::InvalidPeriod {
                period: 0,
                data_len: 0,
            })
        }
    };
    // Map batch → non-batch SIMD for the inner
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        // Fallback for any other batch kernels (shouldn't happen, but safer than unreachable!)
        _ => Kernel::Scalar,
    };
    // In WASM, don't use parallel processing
    #[cfg(target_arch = "wasm32")]
    let parallel = false;
    #[cfg(not(target_arch = "wasm32"))]
    let parallel = true;
    vama_batch_inner(data, ranges, simd, parallel)
}

// ==================== PYTHON BINDINGS ====================
#[cfg(feature = "python")]
#[pyfunction(name = "vama")]
#[pyo3(signature = (data, base_period=113, vol_period=51, smoothing=true, smooth_type=3, smooth_period=5, kernel=None))]
pub fn vama_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    base_period: usize,
    vol_period: usize,
    smoothing: bool,
    smooth_type: usize,
    smooth_period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let k = validate_kernel(kernel, false)?;
    let data_slice = data.as_slice()?;

    let params = VamaParams {
        base_period: Some(base_period),
        vol_period: Some(vol_period),
        smoothing: Some(smoothing),
        smooth_type: Some(smooth_type),
        smooth_period: Some(smooth_period),
    };
    let input = VamaInput::from_slice(data_slice, params);

    let result_vec: Vec<f64> = py
        .allow_threads(|| vama_with_kernel(&input, k).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyfunction(name = "vama_batch")]
#[pyo3(signature = (data, base_period_range=(100,130,10), vol_period_range=(40,60,10), kernel=None))]
pub fn vama_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    base_period_range: (usize, usize, usize),
    vol_period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    let slice_in = data.as_slice()?;
    let ranges = VamaBatchRange {
        base_period: base_period_range,
        vol_period: vol_period_range,
    };
    let kern = validate_kernel(kernel, true)?;

    // Pre-calculate combos and dimensions
    let combos = expand_grid_vama(&ranges);
    let rows = combos.len();
    let cols = slice_in.len();

    // Pre-allocate NumPy array and write directly into it
    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let out_slice = unsafe { out_arr.as_slice_mut()? };

    let simd = match kern {
        Kernel::Auto => match detect_best_batch_kernel() {
            Kernel::Avx512Batch => Kernel::Avx512,
            Kernel::Avx2Batch => Kernel::Avx2,
            _ => Kernel::Scalar,
        },
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => return Err(PyValueError::new_err("invalid batch kernel")),
    };

    let first = slice_in
        .iter()
        .position(|x| !x.is_nan())
        .ok_or_else(|| PyValueError::new_err("vama: All values are NaN"))?;
    py.allow_threads(|| {
        vama_batch_inner_into_with_simd(slice_in, &combos, first, simd, out_slice, cols, true)
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // dict keys mirror Alma style per-param
    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "base_periods",
        combos
            .iter()
            .map(|p| p.base_period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "vol_periods",
        combos
            .iter()
            .map(|p| p.vol_period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    Ok(dict.into())
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "vama_cuda_batch_dev")]
#[pyo3(signature = (data_f32, base_period_range=(100,130,10), vol_period_range=(40,60,10), device_id=0))]
pub fn vama_cuda_batch_dev_py(
    py: Python<'_>,
    data_f32: PyReadonlyArray1<'_, f32>,
    base_period_range: (usize, usize, usize),
    vol_period_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    use crate::cuda::cuda_available;
    use crate::cuda::moving_averages::CudaVama;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let slice_in = data_f32.as_slice()?;
    let ranges = VamaBatchRange {
        base_period: base_period_range,
        vol_period: vol_period_range,
    };

    let inner = py.allow_threads(|| {
        let cuda = CudaVama::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.vama_batch_dev(slice_in, &ranges)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    Ok(DeviceArrayF32Py { inner })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "vama_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm_f32, base_period, vol_period, device_id=0))]
pub fn vama_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    data_tm_f32: PyReadonlyArray2<'_, f32>,
    base_period: usize,
    vol_period: usize,
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    use crate::cuda::cuda_available;
    use crate::cuda::moving_averages::CudaVama;
    use numpy::PyUntypedArrayMethods;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    if base_period == 0 || vol_period == 0 {
        return Err(PyValueError::new_err(
            "base_period and vol_period must be positive",
        ));
    }

    let flat: &[f32] = data_tm_f32.as_slice()?;
    let shape = data_tm_f32.shape();
    let rows = shape[0];
    let cols = shape[1];
    let params = VamaParams {
        base_period: Some(base_period),
        vol_period: Some(vol_period),
        smoothing: Some(false),
        smooth_type: Some(3),
        smooth_period: Some(5),
    };

    let inner = py.allow_threads(|| {
        let cuda = CudaVama::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.vama_many_series_one_param_time_major_dev(flat, cols, rows, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    Ok(DeviceArrayF32Py { inner })
}

#[cfg(feature = "python")]
#[pyclass(name = "VamaStream")]
pub struct VamaStreamPy {
    stream: VamaStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl VamaStreamPy {
    #[new]
    #[pyo3(signature = (base_period=113, vol_period=51, smoothing=true, smooth_type=3, smooth_period=5))]
    pub fn new(
        base_period: usize,
        vol_period: usize,
        smoothing: bool,
        smooth_type: usize,
        smooth_period: usize,
    ) -> PyResult<Self> {
        let params = VamaParams {
            base_period: Some(base_period),
            vol_period: Some(vol_period),
            smoothing: Some(smoothing),
            smooth_type: Some(smooth_type),
            smooth_period: Some(smooth_period),
        };

        let stream = VamaStream::try_new(params)
            .map_err(|e| PyErr::new::<PyValueError, _>(e.to_string()))?;

        Ok(Self { stream })
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

// ==================== WASM BINDINGS ====================
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vama_js(
    data: &[f64],
    base_period: usize,
    vol_period: usize,
    smoothing: bool,
    smooth_type: usize,
    smooth_period: usize,
) -> Result<Vec<f64>, JsValue> {
    let params = VamaParams {
        base_period: Some(base_period),
        vol_period: Some(vol_period),
        smoothing: Some(smoothing),
        smooth_type: Some(smooth_type),
        smooth_period: Some(smooth_period),
    };
    let input = VamaInput::from_slice(data, params);
    let mut out = vec![0.0; data.len()];
    vama_into_slice(&mut out, &input, detect_best_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(out)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vama_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vama_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vama_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    base_period: usize,
    vol_period: usize,
    smoothing: bool,
    smooth_type: usize,
    smooth_period: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to vama_into"));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let params = VamaParams {
            base_period: Some(base_period),
            vol_period: Some(vol_period),
            smoothing: Some(smoothing),
            smooth_type: Some(smooth_type),
            smooth_period: Some(smooth_period),
        };
        let input = VamaInput::from_slice(data, params);

        if core::ptr::eq(in_ptr, out_ptr) {
            // In-place: use temp buffer
            let mut tmp = vec![0.0; len];
            vama_into_slice(&mut tmp, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&tmp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            vama_into_slice(out, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }
    Ok(())
}

// Batch unified: flattened values + combos + dims, like ALMA
#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct VamaBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<VamaParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct VamaBatchConfig {
    pub base_period_range: (usize, usize, usize),
    pub vol_period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = vama_batch)]
pub fn vama_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let cfg: VamaBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
    let ranges = VamaBatchRange {
        base_period: cfg.base_period_range,
        vol_period: cfg.vol_period_range,
    };

    // Follow ALMA's pattern: use detect_best_kernel() and pass false for parallel
    let out = vama_batch_inner(data, &ranges, detect_best_kernel(), false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let js = VamaBatchJsOutput {
        values: out.values,
        combos: out.combos,
        rows: out.rows,
        cols: out.cols,
    };
    serde_wasm_bindgen::to_value(&js)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

// ==================== UNIT TESTS ====================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    use std::error::Error;

    // ==================== TEST MACROS ====================
    /// Macro to generate tests for all supported kernels
    macro_rules! test_with_kernels {
        ($test_fn:ident) => {
            paste::paste! {
                #[test]
                fn [<$test_fn _scalar>]() {
                    let _ = $test_fn(stringify!([<$test_fn _scalar>]), Kernel::Scalar);
                }

                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test]
                fn [<$test_fn _avx2>]() {
                    let _ = $test_fn(stringify!([<$test_fn _avx2>]), Kernel::Avx2);
                }

                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test]
                fn [<$test_fn _avx512>]() {
                    let _ = $test_fn(stringify!([<$test_fn _avx512>]), Kernel::Avx512);
                }
            }
        };
    }

    /// Macro to generate batch kernel tests
    macro_rules! test_batch_kernels {
        ($test_fn:ident) => {
            paste::paste! {
                #[test]
                fn [<$test_fn _scalar_batch>]() {
                    let _ = $test_fn(stringify!([<$test_fn _scalar_batch>]), Kernel::ScalarBatch);
                }

                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test]
                fn [<$test_fn _avx2_batch>]() {
                    let _ = $test_fn(stringify!([<$test_fn _avx2_batch>]), Kernel::Avx2Batch);
                }

                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test]
                fn [<$test_fn _avx512_batch>]() {
                    let _ = $test_fn(stringify!([<$test_fn _avx512_batch>]), Kernel::Avx512Batch);
                }
            }
        };
    }

    // Helper to check warmup NaNs are correctly set
    fn check_vama_warmup_nan(test_name: &str, k: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(k, test_name);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file)?;
        let input = VamaInput::with_default_candles(&candles);
        let out = vama_with_kernel(&input, k)?;
        let first = candles.close.iter().position(|x| !x.is_nan()).unwrap();
        let warm = first + input.get_base_period().max(input.get_vol_period()) - 1;
        assert!(out.values[..warm].iter().all(|v| v.is_nan()));
        Ok(())
    }

    // Helper to check no poison values in output
    #[cfg(debug_assertions)]
    fn check_vama_no_poison(test_name: &str, k: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(k, test_name);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file)?;
        let input = VamaInput::with_default_candles(&candles);
        let out = vama_with_kernel(&input, k)?;
        for &v in &out.values {
            if v.is_nan() {
                continue;
            }
            let b = v.to_bits();
            assert_ne!(b, 0x11111111_11111111);
            assert_ne!(b, 0x22222222_22222222);
            assert_ne!(b, 0x33333333_33333333);
        }
        Ok(())
    }

    // Test all kernels produce identical results
    fn check_kernel_consistency(test_name: &str) -> Result<(), Box<dyn Error>> {
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file)?;
        let input = VamaInput::with_default_candles(&candles);

        let scalar = vama_with_kernel(&input, Kernel::Scalar)?;

        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") {
                let avx2 = vama_with_kernel(&input, Kernel::Avx2)?;
                for (i, (s, a)) in scalar.values.iter().zip(avx2.values.iter()).enumerate() {
                    if !s.is_nan() && !a.is_nan() {
                        let diff = (s - a).abs();
                        if diff > 1e-10 {
                            panic!("Scalar vs AVX2 mismatch at {}: {} vs {}", i, s, a);
                        }
                    }
                }
            }

            if is_x86_feature_detected!("avx512f") {
                let avx512 = vama_with_kernel(&input, Kernel::Avx512)?;
                for (i, (s, a)) in scalar.values.iter().zip(avx512.values.iter()).enumerate() {
                    if !s.is_nan() && !a.is_nan() {
                        let diff = (s - a).abs();
                        if diff > 1e-10 {
                            panic!("Scalar vs AVX512 mismatch at {}: {} vs {}", i, s, a);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    // Generate tests for all kernels
    test_with_kernels!(check_vama_warmup_nan);

    #[cfg(debug_assertions)]
    test_with_kernels!(check_vama_no_poison);

    test_with_kernels!(check_vama_edge_cases);
    test_with_kernels!(check_vama_smoothing);
    test_batch_kernels!(check_vama_batch_consistency);

    // New batch tests matching alma.rs style
    macro_rules! gen_batch_tests {
        ($fn_name:ident) => {
            paste::paste! {
                #[test] fn [<$fn_name _scalar>]()      { let _ = $fn_name(stringify!([<$fn_name _scalar>]), Kernel::ScalarBatch); }
                #[cfg(all(feature="nightly-avx", target_arch="x86_64"))]
                #[test] fn [<$fn_name _avx2>]()        { let _ = $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch); }
                #[cfg(all(feature="nightly-avx", target_arch="x86_64"))]
                #[test] fn [<$fn_name _avx512>]()      { let _ = $fn_name(stringify!([<$fn_name _avx512>]), Kernel::Avx512Batch); }
                #[test] fn [<$fn_name _auto>]()        { let _ = $fn_name(stringify!([<$fn_name _auto>]), Kernel::Auto); }
            }
        };
    }

    fn check_batch_default_row(test: &str, k: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(k, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let out = VamaBatchBuilder::new()
            .kernel(k)
            .base_period_static(113)
            .vol_period_static(51)
            .apply_candles(&c, "close")?;
        let def = VamaParams {
            base_period: Some(113),
            vol_period: Some(51),
            smoothing: Some(false),
            smooth_type: Some(3),
            smooth_period: Some(5),
        };
        let row = out.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        Ok(())
    }

    fn check_batch_sweep(test: &str, k: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(k, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let out = VamaBatchBuilder::new()
            .kernel(k)
            .base_period_range(100, 104, 2)
            .vol_period_range(40, 44, 2)
            .apply_candles(&c, "close")?;
        assert_eq!(out.rows, 3 * 3); // 100,102,104 x 40,42,44
        assert_eq!(out.cols, c.close.len());
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_vama_batch_no_poison(test: &str, k: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(k, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let out = VamaBatchBuilder::new()
            .kernel(k)
            .base_period_range(2, 6, 2)
            .vol_period_range(2, 6, 2)
            .apply_candles(&c, "close")?;
        for &v in &out.values {
            if v.is_nan() {
                continue;
            }
            let b = v.to_bits();
            assert_ne!(b, 0x11111111_11111111);
            assert_ne!(b, 0x22222222_22222222);
            assert_ne!(b, 0x33333333_33333333);
        }
        Ok(())
    }

    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_sweep);
    #[cfg(debug_assertions)]
    gen_batch_tests!(check_vama_batch_no_poison);

    // Test kernel consistency
    #[test]
    fn vama_kernel_consistency() {
        let _ = check_kernel_consistency("vama_kernel_consistency");
    }

    // ==================== KERNEL-SPECIFIC TEST FUNCTIONS ====================

    // Test edge cases for different kernels
    fn check_vama_edge_cases(test_name: &str, k: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(k, test_name);

        // Test with minimal data
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let base_p = 2;
        let vol_p = 2;
        let params = VamaParams {
            base_period: Some(base_p),
            vol_period: Some(vol_p),
            smoothing: Some(false),
            smooth_type: None,
            smooth_period: None,
        };
        let input = VamaInput::from_slice(&data, params);
        let result = vama_with_kernel(&input, k)?;
        assert_eq!(
            result.values.len(),
            data.len(),
            "[{}] Output length should match input",
            test_name
        );

        // Check warmup period - VAMA calculates warmup based on max(base, vol)
        let warmup = base_p.max(vol_p);

        // Count actual NaN values
        let nan_count = result.values.iter().take_while(|v| v.is_nan()).count();

        // Verify warmup period
        assert!(
            nan_count >= warmup - 1,
            "[{}] Should have at least {} warmup NaN values, got {}",
            test_name,
            warmup - 1,
            nan_count
        );

        // Check that we eventually get non-NaN values
        let non_nan_count = result.values.iter().filter(|v| !v.is_nan()).count();
        assert!(
            non_nan_count > 0,
            "[{}] Should have some non-NaN values",
            test_name
        );

        // Test with single value
        let single = vec![42.0];
        let params_single = VamaParams {
            base_period: Some(1),
            vol_period: Some(1),
            smoothing: Some(false),
            smooth_type: None,
            smooth_period: None,
        };
        let input_single = VamaInput::from_slice(&single, params_single);
        let result_single = vama_with_kernel(&input_single, k)?;
        assert_eq!(
            result_single.values.len(),
            1,
            "[{}] Single value output should have length 1",
            test_name
        );
        assert!(
            !result_single.values[0].is_nan(),
            "[{}] Single value should produce a result",
            test_name
        );

        Ok(())
    }

    // Test smoothing variations
    fn check_vama_smoothing(test_name: &str, k: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(k, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let data: Vec<f64> = candles.close[..100].to_vec();

        // Test all smoothing types
        for smooth_type in 1..=3 {
            let params = VamaParams {
                base_period: Some(10),
                vol_period: Some(5),
                smoothing: Some(true),
                smooth_type: Some(smooth_type),
                smooth_period: Some(3),
            };
            let input = VamaInput::from_slice(&data, params);
            let result = vama_with_kernel(&input, k)?;
            assert_eq!(result.values.len(), data.len());

            // Verify some values are not NaN after warmup
            let non_nan = result.values.iter().filter(|v| !v.is_nan()).count();
            assert!(
                non_nan > 0,
                "Smoothing type {} produced all NaNs",
                smooth_type
            );
        }

        Ok(())
    }

    // Test batch consistency
    fn check_vama_batch_consistency(test_name: &str, k: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(k, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let data: Vec<f64> = candles.close[..50].to_vec();
        let ranges = VamaBatchRange {
            base_period: (10, 12, 2),
            vol_period: (5, 7, 2),
        };

        let batch_result = vama_batch_with_kernel(&data, &ranges, k)?;

        // Verify batch dimensions
        assert_eq!(batch_result.combos.len(), 4); // 2x2 grid
        assert_eq!(
            batch_result.values.len(),
            batch_result.rows * batch_result.cols
        );
        assert_eq!(batch_result.cols, data.len());

        // Compare with individual calculations
        for (idx, combo) in batch_result.combos.iter().enumerate() {
            let bp = combo.base_period.unwrap();
            let vp = combo.vol_period.unwrap();
            let params = VamaParams {
                base_period: Some(bp),
                vol_period: Some(vp),
                smoothing: Some(false),
                smooth_type: None,
                smooth_period: None,
            };
            let input = VamaInput::from_slice(&data, params);
            let single_result = vama_with_kernel(
                &input,
                match k {
                    Kernel::ScalarBatch => Kernel::Scalar,
                    Kernel::Avx2Batch => Kernel::Avx2,
                    Kernel::Avx512Batch => Kernel::Avx512,
                    _ => k,
                },
            )?;

            let row_start = idx * data.len();
            let row_end = row_start + data.len();
            let batch_row = &batch_result.values[row_start..row_end];

            for (i, (&batch_val, &single_val)) in batch_row
                .iter()
                .zip(single_result.values.iter())
                .enumerate()
            {
                if !batch_val.is_nan() && !single_val.is_nan() {
                    let diff = (batch_val - single_val).abs();
                    assert!(
                        diff < 1e-10,
                        "Batch vs single mismatch at row {} col {}: {} vs {}",
                        idx,
                        i,
                        batch_val,
                        single_val
                    );
                }
            }
        }

        Ok(())
    }

    // Basic functionality tests
    #[test]
    fn test_vama_accuracy() -> Result<(), Box<dyn Error>> {
        use crate::utilities::data_loader::read_candles_from_csv;

        // Reference values from PineScript with default settings
        let expected = vec![
            61437.31013970,
            61409.77885185,
            61381.24752811,
            61352.71733871,
            61321.57890702,
        ];

        // Load actual market data from CSV
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let params = VamaParams {
            base_period: Some(113),
            vol_period: Some(51),
            smoothing: Some(true),
            smooth_type: Some(3), // WMA
            smooth_period: Some(5),
        };

        let input = VamaInput::from_candles(&candles, "close", params);
        let output = vama(&input)?;

        // Check last 5 values match expected
        let start = output.values.len() - 5;
        for i in 0..5 {
            let actual = output.values[start + i];
            if !actual.is_nan() {
                let expected_val = expected[i];
                let diff = (actual - expected_val).abs();
                let tolerance = 1e-6; // Very tight tolerance (matches to 6 decimal places)
                assert!(
                    diff < tolerance,
                    "Value mismatch at index {}: expected {:.8}, got {:.8}, diff {:.10}",
                    i,
                    expected_val,
                    actual,
                    diff
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_vama_empty_input() {
        let data: Vec<f64> = vec![];
        let params = VamaParams::default();
        let input = VamaInput::from_slice(&data, params);
        let result = vama(&input);
        assert!(matches!(result, Err(VamaError::EmptyInputData)));
    }

    #[test]
    fn test_vama_all_nan() {
        let data = vec![f64::NAN; 100];
        let params = VamaParams::default();
        let input = VamaInput::from_slice(&data, params);
        let result = vama(&input);
        assert!(matches!(result, Err(VamaError::AllValuesNaN)));
    }

    #[test]
    fn test_vama_invalid_period() {
        let data = vec![1.0; 50];
        let params = VamaParams {
            base_period: Some(100), // Larger than data
            vol_period: Some(51),
            smoothing: Some(false),
            smooth_type: None,
            smooth_period: None,
        };
        let input = VamaInput::from_slice(&data, params);
        let result = vama(&input);
        assert!(matches!(result, Err(VamaError::InvalidPeriod { .. })));
    }

    #[test]
    fn test_vama_batch() -> Result<(), Box<dyn Error>> {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let ranges = VamaBatchRange {
            base_period: (2, 3, 1),
            vol_period: (2, 3, 1),
        };

        // Use ScalarBatch instead of Scalar for batch processing
        let result = vama_batch_with_kernel(&data, &ranges, Kernel::ScalarBatch)?;

        assert_eq!(result.combos.len(), 4); // 2x2 parameter grid
        assert_eq!(result.values.len(), result.rows * result.cols);

        Ok(())
    }

    #[test]
    fn test_vama_builder() -> Result<(), Box<dyn Error>> {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let data: Vec<f64> = candles.close.clone();

        // Test builder with all parameters
        let result = VamaBuilder::new()
            .base_period(20)
            .vol_period(10)
            .smoothing(true)
            .smooth_type(2)
            .smooth_period(5)
            .kernel(Kernel::Auto)
            .apply_slice(&data)?;

        assert_eq!(result.values.len(), data.len());

        // Test builder with defaults (need enough data for default period 113)
        let result_default = VamaBuilder::default().apply_slice(&data)?;

        assert_eq!(result_default.values.len(), data.len());

        Ok(())
    }

    #[test]
    fn test_vama_stream() -> Result<(), Box<dyn Error>> {
        let params = VamaParams::default();
        let mut stream = VamaStream::try_new(params)?;

        // Feed some values
        for i in 0..200 {
            let val = 100.0 + (i as f64) * 0.1;
            let _ = stream.update(val);
        }

        assert!(stream.ready);
        Ok(())
    }

    #[test]
    fn test_vama_input_variants() -> Result<(), Box<dyn Error>> {
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file)?;

        // Test with candles
        let result_candles = VamaInput::with_default_candles(&candles);
        let out_candles = vama(&result_candles)?;
        assert_eq!(out_candles.values.len(), candles.close.len());

        // Test with custom source
        let result_high = VamaInput::from_candles(&candles, "high", VamaParams::default());
        let out_high = vama(&result_high)?;
        assert_eq!(out_high.values.len(), candles.high.len());

        // Test with slice
        let slice_data: Vec<f64> = candles.close.clone();
        let result_slice = VamaInput::from_slice(&slice_data, VamaParams::default());
        let out_slice = vama(&result_slice)?;
        assert_eq!(out_slice.values.len(), slice_data.len());

        Ok(())
    }

    #[test]
    fn test_vama_into_slice() -> Result<(), Box<dyn Error>> {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let data: Vec<f64> = candles.close.clone();
        let mut output = vec![0.0; data.len()];

        let params = VamaParams::default();
        let input = VamaInput::from_slice(&data, params);

        vama_into_slice(&mut output, &input, Kernel::Auto)?;

        // Verify warmup NaNs were written
        let first_valid = output.iter().position(|v| !v.is_nan());
        assert!(first_valid.is_some());

        // Compare with regular vama
        let regular_output = vama(&input)?;
        for (i, (&into_val, &regular_val)) in
            output.iter().zip(regular_output.values.iter()).enumerate()
        {
            if !into_val.is_nan() && !regular_val.is_nan() {
                let diff = (into_val - regular_val).abs();
                assert!(
                    diff < 1e-10,
                    "into_slice mismatch at {}: {} vs {}",
                    i,
                    into_val,
                    regular_val
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_vama_param_validation() {
        let data = vec![1.0; 10];

        // Test zero period
        let params = VamaParams {
            base_period: Some(0),
            vol_period: Some(5),
            smoothing: Some(false),
            smooth_type: None,
            smooth_period: None,
        };
        let input = VamaInput::from_slice(&data, params);
        let result = vama(&input);
        assert!(matches!(result, Err(VamaError::InvalidPeriod { .. })));

        // Test invalid smooth type
        let params = VamaParams {
            base_period: Some(5),
            vol_period: Some(5),
            smoothing: Some(true),
            smooth_type: Some(4), // Invalid
            smooth_period: Some(3),
        };
        let input = VamaInput::from_slice(&data, params);
        let result = vama(&input);
        assert!(matches!(result, Err(VamaError::InvalidSmoothType { .. })));
    }

    // New tests to match Alma parity
    fn check_vama_partial_params(test: &str, k: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(k, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let input = VamaInput::from_candles(
            &c,
            "close",
            VamaParams {
                base_period: None,
                vol_period: None,
                smoothing: None,
                smooth_type: None,
                smooth_period: None,
            },
        );
        let out = vama_with_kernel(&input, k)?;
        assert_eq!(out.values.len(), c.close.len());
        Ok(())
    }

    fn check_vama_default_candles(test: &str, k: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(k, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let input = VamaInput::with_default_candles(&c);
        match input.data {
            VamaData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!(),
        }
        let out = vama_with_kernel(&input, k)?;
        assert_eq!(out.values.len(), c.close.len());
        Ok(())
    }

    fn check_vama_period_errors(test: &str, k: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(k, test);
        let data = vec![1.0; 10];

        // Test zero base period
        let params = VamaParams {
            base_period: Some(0),
            vol_period: Some(5),
            smoothing: Some(false),
            smooth_type: None,
            smooth_period: None,
        };
        let input = VamaInput::from_slice(&data, params);
        let result = vama_with_kernel(&input, k);
        assert!(matches!(result, Err(VamaError::InvalidPeriod { .. })));

        // Test too long period
        let params = VamaParams {
            base_period: Some(100),
            vol_period: Some(5),
            smoothing: Some(false),
            smooth_type: None,
            smooth_period: None,
        };
        let input = VamaInput::from_slice(&data, params);
        let result = vama_with_kernel(&input, k);
        assert!(matches!(result, Err(VamaError::InvalidPeriod { .. })));

        Ok(())
    }

    fn check_vama_small_dataset(test: &str, k: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(k, test);
        let data = vec![1.0, 2.0, 3.0];
        let params = VamaParams {
            base_period: Some(5),
            vol_period: Some(5),
            smoothing: Some(false),
            smooth_type: None,
            smooth_period: None,
        };
        let input = VamaInput::from_slice(&data, params);
        let result = vama_with_kernel(&input, k);
        // Since data length (3) < period (5), we get InvalidPeriod error
        assert!(matches!(result, Err(VamaError::InvalidPeriod { .. })));
        Ok(())
    }

    fn check_vama_reinput(test: &str, k: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(k, test);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let data: Vec<f64> = candles.close[..200].to_vec();
        let params = VamaParams {
            base_period: Some(20),
            vol_period: Some(10),
            smoothing: Some(false),
            smooth_type: None,
            smooth_period: None,
        };

        // First pass
        let input1 = VamaInput::from_slice(&data, params.clone());
        let output1 = vama_with_kernel(&input1, k)?;

        // Second pass - use output as input
        let input2 = VamaInput::from_slice(&output1.values, params);
        let output2 = vama_with_kernel(&input2, k)?;

        // Should have same length
        assert_eq!(
            output2.values.len(),
            output1.values.len(),
            "[{}] Output lengths should match",
            test
        );

        // Values should be different but reasonable after warmup
        let warmup = 30;
        let mut found_diff = false;
        let mut max_diff = 0.0;

        for i in warmup..output1.values.len() {
            if !output1.values[i].is_nan() && !output2.values[i].is_nan() {
                let diff = (output1.values[i] - output2.values[i]).abs();
                max_diff = if diff > max_diff { diff } else { max_diff };
                if diff > 1e-10 {
                    found_diff = true;
                }
                // Values should be in reasonable range (use larger tolerance for real data)
                assert!(
                    diff < 100000.0,
                    "[{}] Reinput difference too large at {}: {} vs {}, diff: {}",
                    test,
                    i,
                    output1.values[i],
                    output2.values[i],
                    diff
                );
            }
        }

        assert!(
            found_diff,
            "[{}] Reinput should produce different values",
            test
        );

        Ok(())
    }

    // Wire up the new tests
    test_with_kernels!(check_vama_partial_params);
    test_with_kernels!(check_vama_default_candles);
    test_with_kernels!(check_vama_period_errors);
    test_with_kernels!(check_vama_small_dataset);
    test_with_kernels!(check_vama_reinput);

    // Property-based testing
    #[cfg(feature = "proptest")]
    mod proptest_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn test_vama_proptest_length_preservation(
                data in prop::collection::vec(0.0f64..1000.0, 10..500),
                base_period in 2usize..50,
                vol_period in 2usize..50,
            ) {
                let params = VamaParams {
                    base_period: Some(base_period.min(data.len())),
                    vol_period: Some(vol_period.min(data.len())),
                    smoothing: Some(false),
                    smooth_type: None,
                    smooth_period: None,
                };
                let input = VamaInput::from_slice(&data, params);

                if let Ok(output) = vama(&input) {
                    prop_assert_eq!(output.values.len(), data.len());
                }
            }

            #[test]
            fn test_vama_proptest_nan_handling(
                mut data in prop::collection::vec(0.0f64..1000.0, 50..200),
                nan_positions in prop::collection::vec(0usize..50, 0..10)
            ) {
                // Insert NaNs at specified positions
                for &pos in &nan_positions {
                    if pos < data.len() {
                        data[pos] = f64::NAN;
                    }
                }

                // Use smaller periods that will work with generated data sizes
                let params = VamaParams {
                    base_period: Some(20.min(data.len())),
                    vol_period: Some(10.min(data.len())),
                    smoothing: Some(false),
                    smooth_type: None,
                    smooth_period: None,
                };
                let input = VamaInput::from_slice(&data, params);

                // Should handle NaNs gracefully - either succeed or fail with AllValuesNaN
                let result = vama(&input);
                prop_assert!(
                    result.is_ok() || matches!(result, Err(VamaError::AllValuesNaN)),
                    "Expected Ok or AllValuesNaN, got {:?}", result
                );
            }

            #[test]
            fn test_vama_proptest_batch_consistency(
                data in prop::collection::vec(0.0f64..1000.0, 100..200),
                base_start in 10usize..30,
                base_end in 31usize..50,
                vol_start in 10usize..30,
                vol_end in 31usize..50,
            ) {
                let ranges = VamaBatchRange {
                    base_period: (base_start, base_end.max(base_start), 5),
                    vol_period: (vol_start, vol_end.max(vol_start), 5),
                };

                if let Ok(batch_result) = vama_batch_with_kernel(&data, &ranges, Kernel::Auto) {
                    prop_assert_eq!(batch_result.values.len(), batch_result.rows * batch_result.cols);
                    prop_assert_eq!(batch_result.cols, data.len());

                    // Each row should correspond to a parameter combination
                    for (idx, params) in batch_result.combos.iter().enumerate() {
                        let single_input = VamaInput::from_slice(&data, params.clone());
                        if let Ok(single_result) = vama(&single_input) {
                            let row_start = idx * data.len();
                            let row_end = row_start + data.len();
                            let batch_row = &batch_result.values[row_start..row_end];

                            // Compare after warmup
                            let warmup = params.base_period.unwrap().max(params.vol_period.unwrap());
                            for i in warmup..data.len() {
                                if !batch_row[i].is_nan() && !single_result.values[i].is_nan() {
                                    let diff = (batch_row[i] - single_result.values[i]).abs();
                                    prop_assert!(diff < 1e-10,
                                        "Batch vs single mismatch at row {} col {}: {} vs {}",
                                        idx, i, batch_row[i], single_result.values[i]);
                                }
                            }
                        }
                    }
                }
            }

            #[test]
            fn test_vama_proptest_determinism(
                data in prop::collection::vec(0.0f64..1000.0, 50..200),
                base_period in 5usize..50,
                vol_period in 5usize..50,
            ) {
                let params = VamaParams {
                    base_period: Some(base_period.min(data.len())),
                    vol_period: Some(vol_period.min(data.len())),
                    smoothing: Some(true),
                    smooth_type: Some(2), // EMA
                    smooth_period: Some(3),
                };
                let input = VamaInput::from_slice(&data, params);

                // Same input should produce same output
                if let (Ok(result1), Ok(result2)) = (vama(&input), vama(&input)) {
                    for (i, (&v1, &v2)) in result1.values.iter().zip(result2.values.iter()).enumerate() {
                        if !v1.is_nan() && !v2.is_nan() {
                            prop_assert!((v1 - v2).abs() < 1e-15,
                                "Non-deterministic result at {}: {} vs {}", i, v1, v2);
                        }
                    }
                }
            }
        }
    }
}
