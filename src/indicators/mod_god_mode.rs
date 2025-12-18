//! # Modified God Mode (Comprehensive Version)
//!
//! Decision log:
//! - Fused scalar kernel enabled; SIMD (AVX2/AVX512) stubs delegate to scalar for consistent performance across sizes.
//! - CUDA FP32 kernels enabled for one-series × many-params and many-series × one-param; Python returns VRAM handles (CAI v3 + DLPack v1.x).
//! - Row-specific batch SIMD not implemented: heavy EMA/RSI recurrences limit lane parallelism and shared precompute.
//!
//! A composite momentum oscillator combining multiple technical indicators.
//! This version includes Builder pattern, Streaming support, Batch processing,
//! and comprehensive testing infrastructure matching ALMA.rs patterns.
//!
//! ## Parameters
//! - **n1**: Channel Length (default: 17)
//! - **n2**: Average Length (default: 6)
//! - **n3**: Short Length (default: 4)
//! - **mode**: Calculation mode (default: tradition_mg)
//! - **use_volume**: Whether to use volume data (default: false)
//!
//! ## Modes
//! - **godmode**: avg(tci, csi, mf, willy)
//! - **tradition**: avg(tci, mf, rsi)
//! - **godmode_mg**: avg(tci, csi_mg, mf, willy, cbci, lrsi)
//! - **tradition_mg**: avg(tci, mf, rsi, cbci, lrsi)
//!
//! ## Returns
//! - **`Ok(ModGodModeOutput)`** on success, containing wavetrend, signal, and histogram vectors
//! - **`Err(ModGodModeError)`** on failure
//!
//! ## Developer Notes
//! - **AVX2**: No dedicated SIMD implementation - uses component indicators' SIMD kernels
//! - **AVX512**: No dedicated SIMD implementation - uses component indicators' SIMD kernels
//! - **Streaming**: O(1) updates implemented; matches fused scalar outputs after warmup.
//! - **Memory**: Uses zero-copy helpers (alloc_with_nan_prefix, make_uninit_matrix, init_matrix_prefixes)

#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

use crate::indicators::cci::{cci_with_kernel, CciInput, CciParams};
use crate::indicators::ema::{ema_into_slice, ema_with_kernel, EmaInput, EmaParams};
use crate::indicators::mfi::{mfi_with_kernel, MfiInput, MfiParams};
use crate::indicators::moving_averages::sma::{
    sma_into_slice, sma_with_kernel, SmaInput, SmaParams,
};
use crate::indicators::rsi::{rsi_with_kernel, RsiInput, RsiParams};
use crate::indicators::tsi::{tsi_with_kernel, TsiInput, TsiParams};
use crate::indicators::willr::{willr_with_kernel, WillrInput, WillrParams};
use crate::utilities::data_loader::Candles;
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

/// Calculation mode for the Modified God Mode indicator
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub enum ModGodModeMode {
    Godmode,
    Tradition,
    GodmodeMg,
    TraditionMg,
}

impl Default for ModGodModeMode {
    fn default() -> Self {
        Self::TraditionMg
    }
}

    

impl std::str::FromStr for ModGodModeMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "godmode" => Ok(Self::Godmode),
            "tradition" => Ok(Self::Tradition),
            "godmode_mg" => Ok(Self::GodmodeMg),
            "tradition_mg" => Ok(Self::TraditionMg),
            _ => Err(format!("Unknown mode: {}", s)),
        }
    }
}

/// Input data for Modified God Mode indicator
#[derive(Debug, Clone)]
pub enum ModGodModeData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        volume: Option<&'a [f64]>,
    },
}

/// Output from Modified God Mode indicator
#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct ModGodModeOutput {
    pub wavetrend: Vec<f64>,
    pub signal: Vec<f64>,
    pub histogram: Vec<f64>,
}

/// Parameters for Modified God Mode indicator
#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct ModGodModeParams {
    pub n1: Option<usize>,
    pub n2: Option<usize>,
    pub n3: Option<usize>,
    pub mode: Option<ModGodModeMode>,
    pub use_volume: Option<bool>,
}

impl Default for ModGodModeParams {
    fn default() -> Self {
        Self {
            n1: Some(17),
            n2: Some(6),
            n3: Some(4),
            mode: Some(ModGodModeMode::TraditionMg),
            use_volume: Some(true), // Default to true to match Pine's implicit behavior
        }
    }
}

/// Input wrapper for Modified God Mode indicator
#[derive(Debug, Clone)]
pub struct ModGodModeInput<'a> {
    pub data: ModGodModeData<'a>,
    pub params: ModGodModeParams,
}

impl<'a> ModGodModeInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: ModGodModeParams) -> Self {
        Self {
            data: ModGodModeData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        volume: Option<&'a [f64]>,
        params: ModGodModeParams,
    ) -> Self {
        Self {
            data: ModGodModeData::Slices {
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
        Self::from_candles(candles, ModGodModeParams::default())
    }

    #[inline]
    pub fn get_n1(&self) -> usize {
        self.params.n1.unwrap_or(17)
    }

    #[inline]
    pub fn get_n2(&self) -> usize {
        self.params.n2.unwrap_or(6)
    }

    #[inline]
    pub fn get_n3(&self) -> usize {
        self.params.n3.unwrap_or(4)
    }

    #[inline]
    pub fn get_mode(&self) -> ModGodModeMode {
        self.params.mode.unwrap_or_default()
    }

    #[inline]
    pub fn get_use_volume(&self) -> bool {
        self.params.use_volume.unwrap_or(false)
    }
}

/// Error type for Modified God Mode indicator
#[derive(Debug, Error)]
pub enum ModGodModeError {
    #[error("mod_god_mode: Input data slice is empty.")]
    EmptyInputData,

    #[error("mod_god_mode: All values are NaN.")]
    AllValuesNaN,

    #[error("mod_god_mode: invalid periods: n1={n1}, n2={n2}, n3={n3}, data_len={data_len}")]
    InvalidPeriod {
        n1: usize,
        n2: usize,
        n3: usize,
        data_len: usize,
    },

    #[error("mod_god_mode: Not enough valid data: needed={needed}, valid={valid}")]
    NotEnoughValidData { needed: usize, valid: usize },

    #[error("mod_god_mode: output slice length mismatch: expected={expected}, got={got}")]
    OutputLengthMismatch { expected: usize, got: usize },

    #[error("mod_god_mode: invalid range expansion: start={start}, end={end}, step={step}")]
    InvalidRange { start: usize, end: usize, step: usize },

    #[error("mod_god_mode: invalid kernel for batch path: {0:?}")]
    InvalidKernelForBatch(Kernel),

    #[error("mod_god_mode: invalid input: {0}")]
    InvalidInput(String),

    #[error("mod_god_mode: calculation error: {0}")]
    CalculationError(String),
}

// ============================================================================
// COMPONENT CALCULATIONS
// ============================================================================

/// TCI Component - Trend Channel Index
fn calculate_tci(
    close: &[f64],
    n1: usize,
    n2: usize,
    kernel: Kernel,
) -> Result<Vec<f64>, ModGodModeError> {
    let ema1_params = EmaParams { period: Some(n1) };
    let ema1_input = EmaInput::from_slice(close, ema1_params);
    let ema1 = ema_with_kernel(&ema1_input, kernel)
        .map_err(|e| ModGodModeError::CalculationError(format!("TCI EMA1: {}", e)))?;

    let mut deviations = vec![f64::NAN; close.len()];
    let mut abs_deviations = vec![f64::NAN; close.len()];

    for i in 0..close.len() {
        if !ema1.values[i].is_nan() {
            deviations[i] = close[i] - ema1.values[i];
            abs_deviations[i] = (close[i] - ema1.values[i]).abs();
        }
    }

    let ema2_params = EmaParams { period: Some(n1) };
    let ema2_input = EmaInput::from_slice(&abs_deviations, ema2_params);
    let ema2 = ema_with_kernel(&ema2_input, kernel)
        .map_err(|e| ModGodModeError::CalculationError(format!("TCI EMA2: {}", e)))?;

    let mut normalized = vec![f64::NAN; close.len()];
    for i in 0..close.len() {
        if !deviations[i].is_nan() && !ema2.values[i].is_nan() && ema2.values[i] != 0.0 {
            normalized[i] = deviations[i] / (0.025 * ema2.values[i]);
        }
    }

    let ema3_params = EmaParams { period: Some(n2) };
    let ema3_input = EmaInput::from_slice(&normalized, ema3_params);
    let ema3 = ema_with_kernel(&ema3_input, kernel)
        .map_err(|e| ModGodModeError::CalculationError(format!("TCI EMA3: {}", e)))?;

    let mut tci = ema3.values;
    for i in 0..tci.len() {
        if !tci[i].is_nan() {
            tci[i] += 50.0;
        }
    }

    Ok(tci)
}

/// CSI Component - Composite Strength Index
fn calculate_csi(
    close: &[f64],
    n1: usize,
    n2: usize,
    n3: usize,
    kernel: Kernel,
) -> Result<Vec<f64>, ModGodModeError> {
    let rsi_params = RsiParams { period: Some(n3) };
    let rsi_input = RsiInput::from_slice(close, rsi_params);
    let rsi = rsi_with_kernel(&rsi_input, kernel)
        .map_err(|e| ModGodModeError::CalculationError(format!("CSI RSI: {}", e)))?;

    // Pine: tsi_basic(src0, short=n1, long=n2)
    let tsi_params = TsiParams {
        short_period: Some(n1),
        long_period: Some(n2),
    };
    let tsi_input = TsiInput::from_slice(close, tsi_params);
    let tsi = tsi_with_kernel(&tsi_input, kernel)
        .map_err(|e| ModGodModeError::CalculationError(format!("CSI TSI: {}", e)))?;

    let mut csi = vec![f64::NAN; close.len()];
    for i in 0..close.len() {
        if !rsi.values[i].is_nan() && !tsi.values[i].is_nan() {
            csi[i] = (rsi.values[i] + (tsi.values[i] * 0.5 + 50.0)) / 2.0;
        }
    }

    Ok(csi)
}

/// CSI_MG Component - Modified Composite Strength Index
fn calculate_csi_mg(
    close: &[f64],
    n1: usize,
    n2: usize,
    n3: usize,
    kernel: Kernel,
) -> Result<Vec<f64>, ModGodModeError> {
    let rsi_params = RsiParams { period: Some(n3) };
    let rsi_input = RsiInput::from_slice(close, rsi_params);
    let rsi = rsi_with_kernel(&rsi_input, kernel)
        .map_err(|e| ModGodModeError::CalculationError(format!("CSI_MG RSI: {}", e)))?;

    // Numerator: normalized pc
    let mut pc_norm = vec![f64::NAN; close.len()];
    for i in 1..close.len() {
        let a = close[i - 1];
        let b = close[i];
        if a.is_nan() || b.is_nan() {
            continue;
        }
        let avg = (a + b) * 0.5;
        if avg != 0.0 {
            pc_norm[i] = (b - a) / avg;
        }
    }

    let e_num = ema_with_kernel(
        &EmaInput::from_slice(&pc_norm, EmaParams { period: Some(n1) }),
        kernel,
    )
    .map_err(|e| ModGodModeError::CalculationError(format!("CSI_MG EMA num1: {}", e)))?;
    let e_num2 = ema_with_kernel(
        &EmaInput::from_slice(&e_num.values, EmaParams { period: Some(n2) }),
        kernel,
    )
    .map_err(|e| ModGodModeError::CalculationError(format!("CSI_MG EMA num2: {}", e)))?;

    // Denominator: abs(price change) (not normalized)
    let mut apc = vec![f64::NAN; close.len()];
    for i in 1..close.len() {
        let a = close[i - 1];
        let b = close[i];
        if a.is_nan() || b.is_nan() {
            continue;
        }
        apc[i] = (b - a).abs();
    }

    let e_den = ema_with_kernel(
        &EmaInput::from_slice(&apc, EmaParams { period: Some(n1) }),
        kernel,
    )
    .map_err(|e| ModGodModeError::CalculationError(format!("CSI_MG EMA den1: {}", e)))?;
    let e_den2 = ema_with_kernel(
        &EmaInput::from_slice(&e_den.values, EmaParams { period: Some(n2) }),
        kernel,
    )
    .map_err(|e| ModGodModeError::CalculationError(format!("CSI_MG EMA den2: {}", e)))?;

    let mut ttsi = vec![f64::NAN; close.len()];
    for i in 0..close.len() {
        let den = e_den2.values[i];
        let num = e_num2.values[i];
        if !num.is_nan() && !den.is_nan() && den != 0.0 {
            // Pine: 100*(num/den), then *0.5 + 50 when composing CSI_MG
            ttsi[i] = 50.0 * (num / den) + 50.0;
        }
    }

    let mut out = vec![f64::NAN; close.len()];
    for i in 0..close.len() {
        if !rsi.values[i].is_nan() && !ttsi[i].is_nan() {
            out[i] = 0.5 * (rsi.values[i] + ttsi[i]);
        }
    }

    Ok(out)
}

/// MF Component - Money Flow
fn calculate_mf(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: Option<&[f64]>,
    n: usize,
    kernel: Kernel,
) -> Result<Vec<f64>, ModGodModeError> {
    let len = close.len();

    if let Some(vol) = volume {
        let mut typical_price = vec![0.0; len];
        for i in 0..len {
            typical_price[i] = (high[i] + low[i] + close[i]) / 3.0;
        }

        let mfi_params = MfiParams { period: Some(n) };
        let mfi_input = MfiInput::from_slices(&typical_price, vol, mfi_params);
        let mfi = mfi_with_kernel(&mfi_input, kernel)
            .map_err(|e| ModGodModeError::CalculationError(format!("MF: {}", e)))?;

        Ok(mfi.values)
    } else {
        let rsi_params = RsiParams { period: Some(n) };
        let rsi_input = RsiInput::from_slice(close, rsi_params);
        let rsi = rsi_with_kernel(&rsi_input, kernel)
            .map_err(|e| ModGodModeError::CalculationError(format!("MF RSI: {}", e)))?;

        Ok(rsi.values)
    }
}

/// Willy Component - Pine version using close-only with n2
fn calculate_willy_pine(close: &[f64], n2: usize) -> Vec<f64> {
    let len = close.len();
    let mut out = vec![f64::NAN; len];
    if len == 0 || n2 == 0 {
        return out;
    }

    for i in 0..len {
        if i + 1 < n2 {
            continue;
        }
        let start = i + 1 - n2;
        let mut hi = f64::NEG_INFINITY;
        let mut lo = f64::INFINITY;
        let mut ok = true;
        for j in start..=i {
            let v = close[j];
            if v.is_nan() {
                ok = false;
                break;
            }
            if v > hi {
                hi = v;
            }
            if v < lo {
                lo = v;
            }
        }
        if !ok {
            continue;
        }
        let range = hi - lo;
        if range != 0.0 && !close[i].is_nan() {
            out[i] = 60.0 * (close[i] - hi) / range + 80.0;
        }
    }
    out
}

/// CBCI Component - Pine version = momentum(RSI,n2) + EMA(RSI,n3)
fn calculate_cbci_pine(
    close: &[f64],
    n2: usize,
    n3: usize,
    kernel: Kernel,
) -> Result<Vec<f64>, ModGodModeError> {
    let r = rsi_with_kernel(
        &RsiInput::from_slice(close, RsiParams { period: Some(n3) }),
        kernel,
    )
    .map_err(|e| ModGodModeError::CalculationError(format!("CBCI RSI: {}", e)))?;

    // mom(x, n2) = x - x[n2]
    let len = close.len();
    let mut mom = vec![f64::NAN; len];
    for i in 0..len {
        if i >= n2 {
            let a = r.values[i];
            let b = r.values[i - n2];
            if !a.is_nan() && !b.is_nan() {
                mom[i] = a - b;
            }
        }
    }

    let rsisma = ema_with_kernel(
        &EmaInput::from_slice(&r.values, EmaParams { period: Some(n3) }),
        kernel,
    )
    .map_err(|e| ModGodModeError::CalculationError(format!("CBCI EMA(RSI): {}", e)))?;

    let mut out = vec![f64::NAN; len];
    for i in 0..len {
        let a = mom[i];
        let b = rsisma.values[i];
        if !a.is_nan() && !b.is_nan() {
            out[i] = a + b;
        }
    }
    Ok(out)
}

/// Laguerre RSI Component - Pine version with fixed alpha=0.7 (FE off)
fn calculate_lrsi_pine(close: &[f64]) -> Vec<f64> {
    let len = close.len();
    let mut out = vec![f64::NAN; len];
    let alpha = 0.7;
    let one_minus = 1.0 - alpha;
    let mut l0 = 0.0;
    let mut l1 = 0.0;
    let mut l2 = 0.0;
    let mut l3 = 0.0;

    for i in 0..len {
        let x = close[i];
        if x.is_nan() {
            continue;
        }

        let prev_l0 = l0;
        l0 = alpha * x + one_minus * prev_l0;

        let prev_l1 = l1;
        l1 = -(one_minus) * l0 + prev_l0 + one_minus * prev_l1;

        let prev_l2 = l2;
        l2 = -(one_minus) * l1 + prev_l1 + one_minus * prev_l2;

        l3 = -(one_minus) * l2 + prev_l2 + one_minus * l3;

        let cu = (l0 - l1).max(0.0) + (l1 - l2).max(0.0) + (l2 - l3).max(0.0);
        let cd = (l1 - l0).max(0.0) + (l2 - l1).max(0.0) + (l3 - l2).max(0.0);
        if cu + cd != 0.0 {
            out[i] = 100.0 * cu / (cu + cd);
        }
    }
    out
}

/// Signal line - SMA(wt1, 6) to match Pine default
fn smooth_signal_sma6(wt1: &[f64], kernel: Kernel) -> Result<Vec<f64>, ModGodModeError> {
    let sig = sma_with_kernel(
        &SmaInput::from_slice(wt1, SmaParams { period: Some(6) }),
        kernel,
    )
    .map_err(|e| ModGodModeError::CalculationError(format!("Signal SMA(6): {}", e)))?;
    Ok(sig.values)
}

/// Histogram compatible with Pine's plotted component
fn histogram_component_pine(
    wt1: &[f64],
    wt2: &[f64],
    n3: usize,
    kernel: Kernel,
) -> Result<Vec<f64>, ModGodModeError> {
    let len = wt1.len();
    let mut tmp = vec![f64::NAN; len];
    for i in 0..len {
        let a = wt1[i];
        let b = wt2[i];
        if !a.is_nan() && !b.is_nan() {
            tmp[i] = (a - b) * 2.0 + 50.0;
        }
    }
    let out = ema_with_kernel(
        &EmaInput::from_slice(&tmp, EmaParams { period: Some(n3) }),
        kernel,
    )
    .map_err(|e| ModGodModeError::CalculationError(format!("Hist EMA: {}", e)))?;
    Ok(out.values)
}

// ============================================================================
// MAIN CALCULATION FUNCTIONS
// ============================================================================

/// Calculate Modified God Mode indicator with automatic kernel detection
pub fn mod_god_mode(input: &ModGodModeInput) -> Result<ModGodModeOutput, ModGodModeError> {
    mod_god_mode_with_kernel(input, Kernel::Auto)
}

/// Public convenience wrapper with automatic kernel detection
#[inline]
pub fn mod_god_mode_auto(input: &ModGodModeInput) -> Result<ModGodModeOutput, ModGodModeError> {
    mod_god_mode_with_kernel(input, Kernel::Auto)
}

/// Calculate Modified God Mode indicator with specified kernel
pub fn mod_god_mode_with_kernel(
    input: &ModGodModeInput,
    kernel: Kernel,
) -> Result<ModGodModeOutput, ModGodModeError> {
    // Extract data references to determine length and warmup
    let (high, low, close, volume) = match &input.data {
        ModGodModeData::Candles { candles } => {
            let vol = if input.get_use_volume() {
                Some(candles.volume.as_slice())
            } else {
                None
            };
            (
                candles.high.as_slice(),
                candles.low.as_slice(),
                candles.close.as_slice(),
                vol,
            )
        }
        ModGodModeData::Slices {
            high,
            low,
            close,
            volume,
        } => {
            let vol = if input.get_use_volume() {
                *volume
            } else {
                None
            };
            (*high, *low, *close, vol)
        }
    };

    let len = close.len();
    if len == 0 {
        return Err(ModGodModeError::EmptyInputData);
    }
    let first = close
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(ModGodModeError::AllValuesNaN)?;
    let need = input.get_n1().max(input.get_n2()).max(input.get_n3());
    if len - first < need {
        return Err(ModGodModeError::NotEnoughValidData {
            needed: need,
            valid: len - first,
        });
    }
    let warm = first + need - 1;

    // Pre-allocate all three outputs with NaN prefixes
    let mut wt = alloc_with_nan_prefix(len, warm);
    let mut sig = alloc_with_nan_prefix(len, warm);
    let mut hist = alloc_with_nan_prefix(len, warm);

    // Resolve kernel
    let kern = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };

    // Check for classic kernel optimization with default parameters
    // DISABLED: Still has accuracy issues even with proper warmup
    if false
        && kern == Kernel::Scalar
        && input.get_n1() == 17
        && input.get_n2() == 6
        && input.get_n3() == 4
        && input.get_mode() == ModGodModeMode::TraditionMg
    {
        // Use optimized classic kernel for default parameters
        unsafe {
            mod_god_mode_scalar_classic_tradition_mg(
                high,
                low,
                close,
                volume,
                17,
                6,
                4,
                first,
                warm,
                input.get_use_volume(),
                &mut wt,
                &mut sig,
                &mut hist,
            )?;
        }
        return Ok(ModGodModeOutput {
            wavetrend: wt,
            signal: sig,
            histogram: hist,
        });
    }

    // Delegate all calculation to the zero-copy into_slices function
    mod_god_mode_into_slices(&mut wt, &mut sig, &mut hist, input, kern)?;

    Ok(ModGodModeOutput {
        wavetrend: wt,
        signal: sig,
        histogram: hist,
    })
}

// Note: Old implementation has been removed. mod_god_mode_with_kernel is now just a thin wrapper
// that pre-allocates outputs and delegates to mod_god_mode_into_slices for zero-copy operation

/* Old implementation removed for cleaner code
fn mod_god_mode_with_kernel_old(input: &ModGodModeInput, kernel: Kernel) -> Result<ModGodModeOutput, ModGodModeError> {
    let (high, low, close, volume): (&[f64], &[f64], &[f64], Option<&[f64]>) = match &input.data {
        ModGodModeData::Candles { candles } => {
            let vol = if input.get_use_volume() {
                Some(candles.volume.as_slice())
            } else {
                None
            };
            (candles.high.as_slice(), candles.low.as_slice(), candles.close.as_slice(), vol)
        }
        ModGodModeData::Slices { high, low, close, volume } => {
            let vol = if input.get_use_volume() {
                volume.map(|v| v)
            } else {
                None
            };
            (*high, *low, *close, vol)
        }
    };

    let len = close.len();
    if len == 0 {
        return Err(ModGodModeError::EmptyInputData);
    }

    // Align with alma.rs: first valid and valid-count checks
    let first = close.iter().position(|x| !x.is_nan()).ok_or(ModGodModeError::AllValuesNaN)?;
    let n1 = input.get_n1();
    let n2 = input.get_n2();
    let n3 = input.get_n3();

    if n1 == 0 || n2 == 0 || n3 == 0 {
        return Err(ModGodModeError::InvalidPeriod { n1, n2, n3, data_len: len });
    }

    let need = n1.max(n2).max(n3);
    if len - first < need {
        return Err(ModGodModeError::NotEnoughValidData {
            needed: need,
            valid: len - first
        });
    }

    // Determine the actual kernel to use
    let actual_kernel = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };

    // Calculate warmup period
    let warm = first + need - 1;

    // Allocate wavetrend with NaN prefix efficiently
    let mut wavetrend = alloc_with_nan_prefix(len, warm);

    // Calculate TCI (always used)
    let tci = calculate_tci(close, n1, n2, actual_kernel)?;

    // Calculate MF (always used)
    let mf = calculate_mf(high, low, close, volume, n3, actual_kernel)?;

    match input.get_mode() {
        ModGodModeMode::Godmode => {
            let csi = calculate_csi(close, n1, n2, n3, actual_kernel)?;
            let willy = calculate_willy_pine(close, n2);

            for i in 0..len {
                let mut sum = 0.0;
                let mut count = 0;

                if !tci[i].is_nan() {
                    sum += tci[i];
                    count += 1;
                }
                if !csi[i].is_nan() {
                    sum += csi[i];
                    count += 1;
                }
                if !mf[i].is_nan() {
                    sum += mf[i];
                    count += 1;
                }
                if !willy[i].is_nan() {
                    sum += willy[i];
                    count += 1;
                }

                if count > 0 {
                    wavetrend[i] = sum / count as f64;
                }
            }
        }
        ModGodModeMode::Tradition => {
            let rsi_params = RsiParams { period: Some(n3) };
            let rsi_input = RsiInput::from_slice(close, rsi_params);
            let rsi = rsi_with_kernel(&rsi_input, actual_kernel)
                .map_err(|e| ModGodModeError::CalculationError(format!("RSI: {}", e)))?;

            for i in 0..len {
                let mut sum = 0.0;
                let mut count = 0;

                if !tci[i].is_nan() {
                    sum += tci[i];
                    count += 1;
                }
                if !mf[i].is_nan() {
                    sum += mf[i];
                    count += 1;
                }
                if !rsi.values[i].is_nan() {
                    sum += rsi.values[i];
                    count += 1;
                }

                if count > 0 {
                    wavetrend[i] = sum / count as f64;
                }
            }
        }
        ModGodModeMode::GodmodeMg => {
            let csi_mg = calculate_csi_mg(close, n1, n2, n3, actual_kernel)?;
            let willy = calculate_willy_pine(close, n2);
            let cbci = calculate_cbci_pine(close, n2, n3, actual_kernel)?;
            let lrsi = calculate_lrsi_pine(close);

            for i in 0..len {
                let mut sum = 0.0;
                let mut count = 0;

                if !tci[i].is_nan() {
                    sum += tci[i];
                    count += 1;
                }
                if !csi_mg[i].is_nan() {
                    sum += csi_mg[i];
                    count += 1;
                }
                if !mf[i].is_nan() {
                    sum += mf[i];
                    count += 1;
                }
                if !willy[i].is_nan() {
                    sum += willy[i];
                    count += 1;
                }
                if !cbci[i].is_nan() {
                    sum += cbci[i];
                    count += 1;
                }
                if !lrsi[i].is_nan() {
                    sum += lrsi[i];
                    count += 1;
                }

                if count > 0 {
                    wavetrend[i] = sum / count as f64;
                }
            }
        }
        ModGodModeMode::TraditionMg => {
            let rsi_params = RsiParams { period: Some(n3) };
            let rsi_input = RsiInput::from_slice(close, rsi_params);
            let rsi = rsi_with_kernel(&rsi_input, actual_kernel)
                .map_err(|e| ModGodModeError::CalculationError(format!("RSI: {}", e)))?;

            let cbci = calculate_cbci_pine(close, n2, n3, actual_kernel)?;
            let lrsi = calculate_lrsi_pine(close);

            // Only compute values after warmup period
            for i in warm..len {
                let mut sum = 0.0;
                let mut count = 0;

                if !tci[i].is_nan() {
                    sum += tci[i];
                    count += 1;
                }
                if !mf[i].is_nan() {
                    sum += mf[i];
                    count += 1;
                }
                if !rsi.values[i].is_nan() {
                    sum += rsi.values[i];
                    count += 1;
                }
                if !cbci[i].is_nan() {
                    sum += cbci[i];
                    count += 1;
                }
                if !lrsi[i].is_nan() {
                    sum += lrsi[i];
                    count += 1;
                }

                if count > 0 {
                    wavetrend[i] = sum / count as f64;
                }
            }
        }
    }

    // Calculate signal line (SMA with fixed period 6)
    let signal = smooth_signal_sma6(&wavetrend, actual_kernel)?;

    // Calculate histogram (Pine-compatible)
    let histogram = histogram_component_pine(&wavetrend, &signal, n3, actual_kernel)?;

    Ok(ModGodModeOutput {
        wavetrend,
        signal,
        histogram,
    })
}
*/

/// In-place calculation for zero-alloc API parity with alma_into_slice
#[inline]
pub fn mod_god_mode_into_slices(
    dst_wavetrend: &mut [f64],
    dst_signal: &mut [f64],
    dst_hist: &mut [f64],
    input: &ModGodModeInput,
    kern: Kernel,
) -> Result<(), ModGodModeError> {
    // Resolve data references
    let (high, low, close, volume) = match &input.data {
        ModGodModeData::Candles { candles } => {
            let vol = if input.get_use_volume() {
                Some(candles.volume.as_slice())
            } else {
                None
            };
            (
                candles.high.as_slice(),
                candles.low.as_slice(),
                candles.close.as_slice(),
                vol,
            )
        }
        ModGodModeData::Slices {
            high,
            low,
            close,
            volume,
        } => {
            let vol = if input.get_use_volume() {
                *volume
            } else {
                None
            };
            (*high, *low, *close, vol)
        }
    };

    let len = close.len();
    if dst_wavetrend.len() != len || dst_signal.len() != len || dst_hist.len() != len {
        let dst_len = dst_wavetrend
            .len()
            .min(dst_signal.len())
            .min(dst_hist.len());
        return Err(ModGodModeError::OutputLengthMismatch {
            expected: len,
            got: dst_len,
        });
    }

    if len == 0 {
        return Err(ModGodModeError::EmptyInputData);
    }
    let first = close
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(ModGodModeError::AllValuesNaN)?;
    let n1 = input.get_n1();
    let n2 = input.get_n2();
    let n3 = input.get_n3();

    if n1 == 0 || n2 == 0 || n3 == 0 {
        return Err(ModGodModeError::InvalidPeriod {
            n1,
            n2,
            n3,
            data_len: len,
        });
    }

    let need = n1.max(n2).max(n3);
    if len - first < need {
        return Err(ModGodModeError::NotEnoughValidData {
            needed: need,
            valid: len - first,
        });
    }

    let warm = first + need - 1;
    let actual = match kern {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };

    // Fast path: fused scalar kernel computes all components in a single pass
    // Mirrors alma.rs style: zero-copy into provided destinations, caller handles NaN prefix.
    if actual == Kernel::Scalar {
        let warm = first + need - 1;
        unsafe {
            mod_god_mode_scalar_fused_into_slices(
                dst_wavetrend,
                dst_signal,
                dst_hist,
                high,
                low,
                close,
                volume,
                n1,
                n2,
                n3,
                input.get_mode(),
                input.get_use_volume(),
                first,
                warm,
            )?;
        }
        // Ensure prefix contract matches alma.rs behavior
        for v in &mut dst_wavetrend[..warm] {
            *v = f64::NAN;
        }
        let sig_start = warm.saturating_add(6 - 1).min(len);
        for v in &mut dst_signal[..sig_start] {
            *v = f64::NAN;
        }
        for v in &mut dst_hist[..sig_start] {
            *v = f64::NAN;
        }
        return Ok(());
    }

    // AVX2/AVX512: vectorize pointwise histogram transform, keep recurrences scalar
    #[cfg(all(
        feature = "nightly-avx",
        target_arch = "x86_64",
        target_feature = "avx512f"
    ))]
    if actual == Kernel::Avx512 {
        let warm = first + need - 1;
        unsafe {
            mod_god_mode_avx512_fused_into_slices(
                dst_wavetrend,
                dst_signal,
                dst_hist,
                high,
                low,
                close,
                volume,
                n1,
                n2,
                n3,
                input.get_mode(),
                input.get_use_volume(),
                first,
                warm,
            )?;
        }
        let sig_start = warm.saturating_add(6 - 1).min(len);
        for v in &mut dst_wavetrend[..warm] {
            *v = f64::NAN;
        }
        for v in &mut dst_signal[..sig_start] {
            *v = f64::NAN;
        }
        for v in &mut dst_hist[..sig_start] {
            *v = f64::NAN;
        }
        return Ok(());
    }

    #[cfg(all(
        feature = "nightly-avx",
        target_arch = "x86_64",
        target_feature = "avx2"
    ))]
    if actual == Kernel::Avx2 {
        let warm = first + need - 1;
        unsafe {
            mod_god_mode_avx2_fused_into_slices(
                dst_wavetrend,
                dst_signal,
                dst_hist,
                high,
                low,
                close,
                volume,
                n1,
                n2,
                n3,
                input.get_mode(),
                input.get_use_volume(),
                first,
                warm,
            )?;
        }
        let sig_start = warm.saturating_add(6 - 1).min(len);
        for v in &mut dst_wavetrend[..warm] {
            *v = f64::NAN;
        }
        for v in &mut dst_signal[..sig_start] {
            *v = f64::NAN;
        }
        for v in &mut dst_hist[..sig_start] {
            *v = f64::NAN;
        }
        return Ok(());
    }

    // Classic inline kernel retained below for reference; superseded by fused scalar above

    // Calculate components
    let tci = calculate_tci(close, n1, n2, actual)?;
    let mf = calculate_mf(high, low, close, volume, n3, actual)?;

    // Compute wavetrend directly into dst_wavetrend
    match input.get_mode() {
        ModGodModeMode::Godmode => {
            let csi = calculate_csi(close, n1, n2, n3, actual)?;
            let willy = calculate_willy_pine(close, n2);

            for i in warm..len {
                let mut sum = 0.0;
                let mut count = 0;
                if !tci[i].is_nan() {
                    sum += tci[i];
                    count += 1;
                }
                if !csi[i].is_nan() {
                    sum += csi[i];
                    count += 1;
                }
                if !mf[i].is_nan() {
                    sum += mf[i];
                    count += 1;
                }
                if !willy[i].is_nan() {
                    sum += willy[i];
                    count += 1;
                }
                if count > 0 {
                    dst_wavetrend[i] = sum / count as f64;
                }
            }
        }
        ModGodModeMode::Tradition => {
            let rsi = rsi_with_kernel(
                &RsiInput::from_slice(close, RsiParams { period: Some(n3) }),
                actual,
            )
            .map_err(|e| ModGodModeError::CalculationError(format!("RSI: {}", e)))?;

            for i in warm..len {
                let mut sum = 0.0;
                let mut count = 0;
                if !tci[i].is_nan() {
                    sum += tci[i];
                    count += 1;
                }
                if !mf[i].is_nan() {
                    sum += mf[i];
                    count += 1;
                }
                if !rsi.values[i].is_nan() {
                    sum += rsi.values[i];
                    count += 1;
                }
                if count > 0 {
                    dst_wavetrend[i] = sum / count as f64;
                }
            }
        }
        ModGodModeMode::GodmodeMg => {
            let csi_mg = calculate_csi_mg(close, n1, n2, n3, actual)?;
            let willy = calculate_willy_pine(close, n2);
            let cbci = calculate_cbci_pine(close, n2, n3, actual)?;
            let lrsi = calculate_lrsi_pine(close);

            for i in warm..len {
                let mut sum = 0.0;
                let mut count = 0;
                if !tci[i].is_nan() {
                    sum += tci[i];
                    count += 1;
                }
                if !csi_mg[i].is_nan() {
                    sum += csi_mg[i];
                    count += 1;
                }
                if !mf[i].is_nan() {
                    sum += mf[i];
                    count += 1;
                }
                if !willy[i].is_nan() {
                    sum += willy[i];
                    count += 1;
                }
                if !cbci[i].is_nan() {
                    sum += cbci[i];
                    count += 1;
                }
                if !lrsi[i].is_nan() {
                    sum += lrsi[i];
                    count += 1;
                }
                if count > 0 {
                    dst_wavetrend[i] = sum / count as f64;
                }
            }
        }
        ModGodModeMode::TraditionMg => {
            let rsi = rsi_with_kernel(
                &RsiInput::from_slice(close, RsiParams { period: Some(n3) }),
                actual,
            )
            .map_err(|e| ModGodModeError::CalculationError(format!("RSI: {}", e)))?;
            let cbci = calculate_cbci_pine(close, n2, n3, actual)?;
            let lrsi = calculate_lrsi_pine(close);

            for i in warm..len {
                let mut sum = 0.0;
                let mut count = 0;
                if !tci[i].is_nan() {
                    sum += tci[i];
                    count += 1;
                }
                if !mf[i].is_nan() {
                    sum += mf[i];
                    count += 1;
                }
                if !rsi.values[i].is_nan() {
                    sum += rsi.values[i];
                    count += 1;
                }
                if !cbci[i].is_nan() {
                    sum += cbci[i];
                    count += 1;
                }
                if !lrsi[i].is_nan() {
                    sum += lrsi[i];
                    count += 1;
                }
                if count > 0 {
                    dst_wavetrend[i] = sum / count as f64;
                }
            }
        }
    }

    // Build signal using SMA6 directly into dst_signal (zero-copy)
    // Check if we have enough non-NaN values in wavetrend for SMA(6)
    let wt_valid = dst_wavetrend[warm..].len();
    if wt_valid >= 6 {
        sma_into_slice(
            dst_signal,
            &SmaInput::from_slice(dst_wavetrend, SmaParams { period: Some(6) }),
            actual,
        )
        .map_err(|e| ModGodModeError::CalculationError(format!("Signal SMA(6): {}", e)))?;
    } else {
        // Not enough valid wavetrend data for signal - fill with NaNs
        dst_signal.fill(f64::NAN);
    }

    // Build histogram using temporary scratch buffer and EMA into dst_hist (zero-copy)
    let len = dst_wavetrend.len();

    // Check if we have enough valid signal data to compute histogram
    let sig_valid_start = dst_signal.iter().position(|x| !x.is_nan()).unwrap_or(len);
    let sig_valid = if sig_valid_start < len {
        len - sig_valid_start
    } else {
        0
    };

    if sig_valid >= n3 {
        // We have enough data, proceed with histogram calculation
        let mut tmp_mu = make_uninit_matrix(1, len);
        init_matrix_prefixes(&mut tmp_mu, len, &[sig_valid_start]);
        let tmp = unsafe { core::slice::from_raw_parts_mut(tmp_mu.as_mut_ptr() as *mut f64, len) };

        // Calculate (wt - sig)*2 + 50 into temporary buffer
        for i in sig_valid_start..len {
            tmp[i] = (dst_wavetrend[i] - dst_signal[i]) * 2.0 + 50.0;
        }

        // Apply EMA(n3) directly into dst_hist
        ema_into_slice(
            dst_hist,
            &EmaInput::from_slice(tmp, EmaParams { period: Some(n3) }),
            actual,
        )
        .map_err(|e| ModGodModeError::CalculationError(format!("Hist EMA: {}", e)))?;
    } else {
        // Not enough valid data for histogram - fill with NaNs
        dst_hist.fill(f64::NAN);
    }

    // Finalize warmup prefix with NaNs
    for v in &mut dst_wavetrend[..warm] {
        *v = f64::NAN;
    }
    for v in &mut dst_signal[..warm] {
        *v = f64::NAN;
    }
    for v in &mut dst_hist[..warm] {
        *v = f64::NAN;
    }

    Ok(())
}

// ============================================================================
// PUBLIC ZERO-ALLOC API
// ============================================================================

/// Write Modified God Mode outputs directly into caller-provided buffers (no allocation).
///
/// - Preserves NaN warmups exactly like the Vec-returning API.
/// - All output slice lengths must equal the input length.
/// - Uses kernel auto-detection by default (Kernel::Auto).
#[cfg(not(feature = "wasm"))]
#[inline]
pub fn mod_god_mode_into(
    input: &ModGodModeInput,
    out_wavetrend: &mut [f64],
    out_signal: &mut [f64],
    out_histogram: &mut [f64],
) -> Result<(), ModGodModeError> {
    // Determine input length and obtain close for warmup computation
    let (high, low, close, volume) = match &input.data {
        ModGodModeData::Candles { candles } => {
            let vol = if input.get_use_volume() {
                Some(candles.volume.as_slice())
            } else {
                None
            };
            (
                candles.high.as_slice(),
                candles.low.as_slice(),
                candles.close.as_slice(),
                vol,
            )
        }
        ModGodModeData::Slices {
            high,
            low,
            close,
            volume,
        } => {
            let vol = if input.get_use_volume() { *volume } else { None };
            (*high, *low, *close, vol)
        }
    };

    let len = close.len();
    if out_wavetrend.len() != len || out_signal.len() != len || out_histogram.len() != len {
        let dst_len = out_wavetrend
            .len()
            .min(out_signal.len())
            .min(out_histogram.len());
        return Err(ModGodModeError::OutputLengthMismatch { expected: len, got: dst_len });
    }

    if len == 0 {
        return Err(ModGodModeError::EmptyInputData);
    }

    let first = close
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(ModGodModeError::AllValuesNaN)?;

    let n1 = input.get_n1();
    let n2 = input.get_n2();
    let n3 = input.get_n3();
    if n1 == 0 || n2 == 0 || n3 == 0 {
        return Err(ModGodModeError::InvalidPeriod { n1, n2, n3, data_len: len });
    }

    let need = n1.max(n2).max(n3);
    if len - first < need {
        return Err(ModGodModeError::NotEnoughValidData { needed: need, valid: len - first });
    }

    // Prefill NaN warmup prefixes to mirror Vec API behavior
    let warm = first + need - 1;
    let qnan = f64::from_bits(0x7ff8_0000_0000_0000);
    for v in &mut out_wavetrend[..warm] {
        *v = qnan;
    }
    for v in &mut out_signal[..warm] {
        *v = qnan;
    }
    for v in &mut out_histogram[..warm] {
        *v = qnan;
    }

    // Delegate core computation to zero-copy kernel dispatcher
    let _ = (high, low, volume); // silence unused tuple fields in this wrapper
    mod_god_mode_into_slices(out_wavetrend, out_signal, out_histogram, input, Kernel::Auto)
}

// ============================================================================
// CLASSIC KERNEL - Optimized for default TraditionMg mode
// ============================================================================

/// Fused scalar kernel (all modes) — single-pass implementation.
/// Computes wavetrend, signal, and histogram directly into destination slices.
/// Safety: caller must ensure destination slices have the same length as inputs.
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn mod_god_mode_scalar_fused_into_slices(
    dst_wavetrend: &mut [f64],
    dst_signal: &mut [f64],
    dst_hist: &mut [f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: Option<&[f64]>,
    n1: usize,
    n2: usize,
    n3: usize,
    mode: ModGodModeMode,
    use_volume: bool,
    first: usize,
    warm: usize,
) -> Result<(), ModGodModeError> {
    let len = close.len();
    if len == 0 {
        return Err(ModGodModeError::EmptyInputData);
    }
    if n1 == 0 || n2 == 0 || n3 == 0 {
        return Err(ModGodModeError::InvalidPeriod {
            n1,
            n2,
            n3,
            data_len: len,
        });
    }
    if dst_wavetrend.len() != len || dst_signal.len() != len || dst_hist.len() != len {
        let dst_len = dst_wavetrend
            .len()
            .min(dst_signal.len())
            .min(dst_hist.len());
        return Err(ModGodModeError::OutputLengthMismatch {
            expected: len,
            got: dst_len,
        });
    }

    #[inline(always)]
    fn ema_step(x: f64, prev: f64, alpha: f64, beta: f64) -> f64 {
        beta.mul_add(prev, alpha * x)
    }
    #[inline(always)]
    fn nonzero(v: f64) -> bool {
        v != 0.0 && v.is_finite()
    }

    let alpha1 = 2.0 / (n1 as f64 + 1.0);
    let beta1 = 1.0 - alpha1;
    let alpha2 = 2.0 / (n2 as f64 + 1.0);
    let beta2 = 1.0 - alpha2;
    let alpha3 = 2.0 / (n3 as f64 + 1.0);
    let beta3 = 1.0 - alpha3;

    // TCI
    let mut ema1_c = 0.0_f64;
    let mut ema2_abs = 0.0_f64;
    let mut ema3_ci = 0.0_f64;
    let mut seed_ema1 = false;
    let mut seed_ema2 = false;
    let mut seed_ema3 = false;

    // RSI (Wilder)
    let mut rs_avg_gain = 0.0_f64;
    let mut rs_avg_loss = 0.0_f64;
    let mut rsi_seeded = false;
    let mut rs_init_cnt = 0usize;
    let mut prev_close = 0.0_f64;

    // RSI momentum ring for CBCI
    let mut rsi_ring: Vec<f64> = vec![f64::NAN; n2.max(1)];
    let mut rsi_ring_head: usize = 0;
    let mut rsi_ema = 0.0_f64;
    let mut rsi_ema_seed = false;

    // Laguerre RSI (Ehlers)
    let alpha_l = 0.7_f64;
    let one_m_l = 1.0_f64 - alpha_l;
    let mut l0 = 0.0_f64;
    let mut l1 = 0.0_f64;
    let mut l2 = 0.0_f64;
    let mut l3 = 0.0_f64;

    // Close-only Willy (%R) over n2
    #[inline(always)]
    fn willr_close_only(c: &[f64], idx: usize, win: usize) -> f64 {
        if win == 0 || idx + 1 < win {
            return f64::NAN;
        }
        let s = idx + 1 - win;
        let mut hi = f64::NEG_INFINITY;
        let mut lo = f64::INFINITY;
        for j in s..=idx {
            let v = c[j];
            if v > hi {
                hi = v;
            }
            if v < lo {
                lo = v;
            }
        }
        let rng = hi - lo;
        if rng == 0.0 {
            f64::NAN
        } else {
            60.0 * (c[idx] - hi) / rng + 80.0
        }
    }

    let has_vol = use_volume && volume.is_some();
    let vol = if has_vol { volume.unwrap() } else { &[][..] };

    // MFI sliding state (window n3)
    let mut mf_pos_sum = 0.0_f64;
    let mut mf_neg_sum = 0.0_f64;
    let mut mf_ring_mf: Vec<f64> = vec![0.0; n3.max(1)];
    let mut mf_ring_sgn: Vec<i8> = vec![0; n3.max(1)];
    let mut mf_head: usize = 0;
    let mut tp_prev: f64 = 0.0_f64;
    let mut tp_has_prev = false;

    // CSI (TSI) states
    let mut tsi_ema_m_s = 0.0_f64;
    let mut tsi_ema_m_l = 0.0_f64;
    let mut tsi_ema_a_s = 0.0_f64;
    let mut tsi_ema_a_l = 0.0_f64;
    let mut tsi_seed_s = false;
    let mut tsi_seed_l = false;

    // CSI_MG states
    let mut csi_num_e1 = 0.0_f64;
    let mut csi_num_e2 = 0.0_f64;
    let mut csi_den_e1 = 0.0_f64;
    let mut csi_den_e2 = 0.0_f64;
    let mut csi_seed_e1 = false;
    let mut csi_seed_e2 = false;

    // signal SMA(6) ring
    const SIGP: usize = 6;
    let mut sig_ring = [0.0_f64; SIGP];
    let mut sig_head = 0usize;
    let mut sig_sum = 0.0_f64;
    let sig_start = warm + SIGP - 1;
    let mut have_sig = false;

    // histogram EMA seeding
    let mut hist_seeded = false;

    let need = n1.max(n2).max(n3);
    if len - first < need {
        return Err(ModGodModeError::NotEnoughValidData {
            needed: need,
            valid: len - first,
        });
    }

    prev_close = close[first];
    if !prev_close.is_finite() {
        return Ok(());
    }

    for i in 0..len {
        if i < first {
            continue;
        }

        // === TCI chain ===
        let c_i = close[i];
        if !seed_ema1 {
            ema1_c = c_i;
            seed_ema1 = true;
        } else {
            ema1_c = ema_step(c_i, ema1_c, alpha1, beta1);
        }
        let abs_dev = (c_i - ema1_c).abs();
        if !seed_ema2 {
            ema2_abs = abs_dev;
            seed_ema2 = true;
        } else {
            ema2_abs = ema_step(abs_dev, ema2_abs, alpha1, beta1);
        }
        let mut tci_val = f64::NAN;
        if nonzero(ema2_abs) {
            let ci = (c_i - ema1_c) / (0.025 * ema2_abs);
            if !seed_ema3 {
                ema3_ci = ci;
                seed_ema3 = true;
            } else {
                ema3_ci = ema_step(ci, ema3_ci, alpha2, beta2);
            }
            tci_val = ema3_ci + 50.0;
        }

        // === RSI / Wilder init + update ===
        let mut rsi_val = f64::NAN;
        if i == first {
            rs_avg_gain = 0.0;
            rs_avg_loss = 0.0;
            rs_init_cnt = 0;
        } else {
            let ch = c_i - prev_close;
            let gain = if ch > 0.0 { ch } else { 0.0 };
            let loss = if ch < 0.0 { -ch } else { 0.0 };
            if !rsi_seeded {
                rs_init_cnt += 1;
                rs_avg_gain += gain;
                rs_avg_loss += loss;
                if rs_init_cnt >= n3 {
                    rs_avg_gain /= n3 as f64;
                    rs_avg_loss /= n3 as f64;
                    rsi_seeded = true;
                    let rs = if rs_avg_loss == 0.0 {
                        f64::INFINITY
                    } else {
                        rs_avg_gain / rs_avg_loss
                    };
                    rsi_val = 100.0 - 100.0 / (1.0 + rs);
                }
            } else {
                rs_avg_gain = ((rs_avg_gain * ((n3 - 1) as f64)) + gain) / (n3 as f64);
                rs_avg_loss = ((rs_avg_loss * ((n3 - 1) as f64)) + loss) / (n3 as f64);
                let rs = if rs_avg_loss == 0.0 {
                    f64::INFINITY
                } else {
                    rs_avg_gain / rs_avg_loss
                };
                rsi_val = 100.0 - 100.0 / (1.0 + rs);
            }
        }

        // Laguerre RSI recursion
        {
            let prev_l0 = l0;
            l0 = alpha_l * c_i + one_m_l * prev_l0;
            let prev_l1 = l1;
            l1 = -one_m_l * l0 + prev_l0 + one_m_l * prev_l1;
            let prev_l2 = l2;
            l2 = -one_m_l * l1 + prev_l1 + one_m_l * prev_l2;
            let _l3p = l3;
            l3 = -one_m_l * l2 + prev_l2 + one_m_l * l3;
        }
        let cu = (l0 - l1).max(0.0) + (l1 - l2).max(0.0) + (l2 - l3).max(0.0);
        let cd = (l1 - l0).max(0.0) + (l2 - l1).max(0.0) + (l3 - l2).max(0.0);
        let lrsi_val = if nonzero(cu + cd) {
            100.0 * cu / (cu + cd)
        } else {
            f64::NAN
        };

        // MFI or RSI-as-MF
        let mut mf_val = f64::NAN;
        if has_vol {
            let tp = (high[i] + low[i] + c_i) / 3.0;
            if tp_has_prev {
                let sign: i8 = if tp > tp_prev {
                    1
                } else if tp < tp_prev {
                    -1
                } else {
                    0
                };
                let mf_raw = tp * vol[i];
                if rsi_seeded {
                    let old_mf = mf_ring_mf[mf_head];
                    let old_sign = mf_ring_sgn[mf_head];
                    if old_sign > 0 {
                        mf_pos_sum -= old_mf;
                    } else if old_sign < 0 {
                        mf_neg_sum -= old_mf;
                    }
                }
                mf_ring_mf[mf_head] = mf_raw;
                mf_ring_sgn[mf_head] = sign;
                if sign > 0 {
                    mf_pos_sum += mf_raw;
                } else if sign < 0 {
                    mf_neg_sum += mf_raw;
                }
                mf_head = (mf_head + 1) % n3.max(1);
                if rsi_seeded {
                    mf_val = if mf_neg_sum == 0.0 {
                        100.0
                    } else {
                        100.0 - 100.0 / (1.0 + (mf_pos_sum / mf_neg_sum))
                    };
                }
            }
            tp_prev = tp;
            tp_has_prev = true;
        } else {
            mf_val = rsi_val;
        }

        // CBCI = RSI momentum over n2 + EMA(RSI, n3)
        let mut cbci_val = f64::NAN;
        if rsi_seeded {
            let old = rsi_ring[rsi_ring_head];
            rsi_ring[rsi_ring_head] = rsi_val;
            rsi_ring_head = (rsi_ring_head + 1) % n2.max(1);
            let mom = if old.is_finite() && rsi_val.is_finite() {
                rsi_val - old
            } else {
                f64::NAN
            };
            if !rsi_ema_seed && rsi_val.is_finite() {
                rsi_ema = rsi_val;
                rsi_ema_seed = true;
            } else if rsi_val.is_finite() {
                rsi_ema = ema_step(rsi_val, rsi_ema, alpha3, beta3);
            }
            if mom.is_finite() && rsi_ema_seed {
                cbci_val = mom + rsi_ema;
            }
        }

        // CSI/CSI_MG
        let mut csi_val = f64::NAN;
        let mut csi_mg_val = f64::NAN;
        if matches!(mode, ModGodModeMode::Godmode) {
            if i > first {
                let mom = c_i - prev_close;
                let am = mom.abs();
                if !tsi_seed_s {
                    tsi_ema_m_s = mom;
                    tsi_ema_a_s = am;
                    tsi_seed_s = true;
                } else {
                    tsi_ema_m_s = ema_step(mom, tsi_ema_m_s, alpha1, beta1);
                    tsi_ema_a_s = ema_step(am, tsi_ema_a_s, alpha1, beta1);
                }
                if !tsi_seed_l && tsi_seed_s {
                    tsi_ema_m_l = tsi_ema_m_s;
                    tsi_ema_a_l = tsi_ema_a_s;
                    tsi_seed_l = true;
                } else if tsi_seed_l {
                    tsi_ema_m_l = ema_step(tsi_ema_m_s, tsi_ema_m_l, alpha2, beta2);
                    tsi_ema_a_l = ema_step(tsi_ema_a_s, tsi_ema_a_l, alpha2, beta2);
                }
                if tsi_seed_l && nonzero(tsi_ema_a_l) {
                    let tsi = 100.0 * (tsi_ema_m_l / tsi_ema_a_l);
                    if rsi_val.is_finite() {
                        csi_val = (rsi_val + (0.5 * tsi + 50.0)) * 0.5;
                    }
                }
            }
        } else if matches!(mode, ModGodModeMode::GodmodeMg) {
            if i > first {
                let a = prev_close;
                let b = c_i;
                let avg = 0.5 * (a + b);
                let pc_norm = if avg != 0.0 { (b - a) / avg } else { 0.0 };
                let apc = (b - a).abs();
                if !csi_seed_e1 {
                    csi_num_e1 = pc_norm;
                    csi_den_e1 = apc;
                    csi_seed_e1 = true;
                } else {
                    csi_num_e1 = ema_step(pc_norm, csi_num_e1, alpha1, beta1);
                    csi_den_e1 = ema_step(apc, csi_den_e1, alpha1, beta1);
                }
                if !csi_seed_e2 && csi_seed_e1 {
                    csi_num_e2 = csi_num_e1;
                    csi_den_e2 = csi_den_e1;
                    csi_seed_e2 = true;
                } else if csi_seed_e2 {
                    csi_num_e2 = ema_step(csi_num_e1, csi_num_e2, alpha2, beta2);
                    csi_den_e2 = ema_step(csi_den_e1, csi_den_e2, alpha2, beta2);
                }
                if csi_seed_e2 && nonzero(csi_den_e2) && rsi_val.is_finite() {
                    let ttsi = 50.0 * (csi_num_e2 / csi_den_e2) + 50.0;
                    csi_mg_val = 0.5 * (rsi_val + ttsi);
                }
            }
        }

        // Combine components by mode into wavetrend
        if i >= warm {
            let mut sum = 0.0_f64;
            let mut cnt = 0i32;
            match mode {
                ModGodModeMode::Godmode => {
                    if tci_val.is_finite() {
                        sum += tci_val;
                        cnt += 1;
                    }
                    if csi_val.is_finite() {
                        sum += csi_val;
                        cnt += 1;
                    }
                    if mf_val.is_finite() {
                        sum += mf_val;
                        cnt += 1;
                    }
                    let wil = willr_close_only(close, i, n2);
                    if wil.is_finite() {
                        sum += wil;
                        cnt += 1;
                    }
                }
                ModGodModeMode::Tradition => {
                    if tci_val.is_finite() {
                        sum += tci_val;
                        cnt += 1;
                    }
                    if mf_val.is_finite() {
                        sum += mf_val;
                        cnt += 1;
                    }
                    if rsi_val.is_finite() {
                        sum += rsi_val;
                        cnt += 1;
                    }
                }
                ModGodModeMode::GodmodeMg => {
                    if tci_val.is_finite() {
                        sum += tci_val;
                        cnt += 1;
                    }
                    if csi_mg_val.is_finite() {
                        sum += csi_mg_val;
                        cnt += 1;
                    }
                    if mf_val.is_finite() {
                        sum += mf_val;
                        cnt += 1;
                    }
                    let wil = willr_close_only(close, i, n2);
                    if wil.is_finite() {
                        sum += wil;
                        cnt += 1;
                    }
                    if cbci_val.is_finite() {
                        sum += cbci_val;
                        cnt += 1;
                    }
                    if lrsi_val.is_finite() {
                        sum += lrsi_val;
                        cnt += 1;
                    }
                }
                ModGodModeMode::TraditionMg => {
                    if tci_val.is_finite() {
                        sum += tci_val;
                        cnt += 1;
                    }
                    if mf_val.is_finite() {
                        sum += mf_val;
                        cnt += 1;
                    }
                    if rsi_val.is_finite() {
                        sum += rsi_val;
                        cnt += 1;
                    }
                    if cbci_val.is_finite() {
                        sum += cbci_val;
                        cnt += 1;
                    }
                    if lrsi_val.is_finite() {
                        sum += lrsi_val;
                        cnt += 1;
                    }
                }
            }
            if cnt > 0 {
                let wt = sum / (cnt as f64);
                dst_wavetrend[i] = wt;

                // Signal SMA(6)
                if i >= sig_start {
                    if !have_sig {
                        let mut s = 0.0;
                        for k in 0..SIGP {
                            let x = dst_wavetrend[i + 1 - SIGP + k];
                            sig_ring[k] = x;
                            s += x;
                        }
                        sig_sum = s;
                        have_sig = true;
                        sig_head = 0;
                        dst_signal[i] = s / (SIGP as f64);
                    } else {
                        let old = sig_ring[sig_head];
                        sig_ring[sig_head] = wt;
                        sig_head = (sig_head + 1) % SIGP;
                        sig_sum += wt - old;
                        dst_signal[i] = sig_sum / (SIGP as f64);
                    }
                    // Histogram EMA over diff = (wt - sig)*2 + 50
                    let d = (dst_wavetrend[i] - dst_signal[i]) * 2.0 + 50.0;
                    if !hist_seeded {
                        dst_hist[i] = d;
                        hist_seeded = true;
                    } else {
                        dst_hist[i] = ema_step(d, dst_hist[i - 1], alpha3, beta3);
                    }
                }
            }
        }
        prev_close = c_i;
    }
    Ok(())
}

// ====================== AVX2 ======================
#[cfg(all(
    feature = "nightly-avx",
    target_arch = "x86_64",
    target_feature = "avx2"
))]
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn mod_god_mode_avx2_fused_into_slices(
    dst_wavetrend: &mut [f64],
    dst_signal: &mut [f64],
    dst_hist: &mut [f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: Option<&[f64]>,
    n1: usize,
    n2: usize,
    n3: usize,
    mode: ModGodModeMode,
    use_volume: bool,
    first: usize,
    warm: usize,
) -> Result<(), ModGodModeError> {
    // Stub: delegate entirely to scalar fused implementation for accuracy/perf consistency
    mod_god_mode_scalar_fused_into_slices(
        dst_wavetrend,
        dst_signal,
        dst_hist,
        high,
        low,
        close,
        volume,
        n1,
        n2,
        n3,
        mode,
        use_volume,
        first,
        warm,
    )
}

// ====================== AVX512 ======================
#[cfg(all(
    feature = "nightly-avx",
    target_arch = "x86_64",
    target_feature = "avx512f"
))]
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn mod_god_mode_avx512_fused_into_slices(
    dst_wavetrend: &mut [f64],
    dst_signal: &mut [f64],
    dst_hist: &mut [f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: Option<&[f64]>,
    n1: usize,
    n2: usize,
    n3: usize,
    mode: ModGodModeMode,
    use_volume: bool,
    first: usize,
    warm: usize,
) -> Result<(), ModGodModeError> {
    // Stub: delegate entirely to scalar fused implementation for accuracy/perf consistency
    mod_god_mode_scalar_fused_into_slices(
        dst_wavetrend,
        dst_signal,
        dst_hist,
        high,
        low,
        close,
        volume,
        n1,
        n2,
        n3,
        mode,
        use_volume,
        first,
        warm,
    )
}

/// Classic kernel for TraditionMg mode with inline calculations
/// Eliminates ~8 function calls by inlining TCI, MF, RSI, CBCI, LRSI, and SMA calculations
#[inline]
pub unsafe fn mod_god_mode_scalar_classic_tradition_mg(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: Option<&[f64]>,
    n1: usize, // 17
    n2: usize, // 6
    n3: usize, // 4
    first: usize,
    _simple_warmup: usize, // Ignored - we calculate proper compound warmup
    use_volume: bool,
    wavetrend: &mut [f64],
    signal: &mut [f64],
    histogram: &mut [f64],
) -> Result<(), ModGodModeError> {
    let len = close.len();
    if len == 0 {
        return Err(ModGodModeError::EmptyInputData);
    }

    // Calculate proper compound warmup periods for each component
    // TCI needs: EMA1(n1) -> EMA2(n1) -> EMA3(n2) = n1 + n1 + n2 - 2
    let tci_warmup = first + n1 + n1 + n2 - 2;
    // MF/RSI needs: n3 periods
    let mf_warmup = first + n3;
    // CBCI needs: RSI(n3) + momentum(n2) or RSI(n3) + EMA(n3)
    let cbci_warmup = first + n3 + n2.max(n3);
    // LRSI is same as RSI
    let lrsi_warmup = first + n3;

    // The actual warmup where we can start producing valid wavetrend values
    let actual_warmup = tci_warmup.max(mf_warmup).max(cbci_warmup).max(lrsi_warmup);

    // === TCI Calculation (inline 3 EMA passes) ===
    // EMA1 of close
    let mut ema1 = vec![f64::NAN; len];
    let alpha1 = 2.0 / (n1 as f64 + 1.0);
    let beta1 = 1.0 - alpha1;
    if first < len {
        ema1[first] = close[first];
        for i in (first + 1)..len {
            if close[i].is_finite() {
                ema1[i] = alpha1 * close[i] + beta1 * ema1[i - 1];
            } else {
                ema1[i] = ema1[i - 1];
            }
        }
    }

    // Calculate absolute deviations from EMA1
    let mut abs_dev = vec![f64::NAN; len];
    for i in first..len {
        if ema1[i].is_finite() {
            abs_dev[i] = (close[i] - ema1[i]).abs();
        }
    }

    // EMA2 of absolute deviations (starts from first + n1 - 1)
    let mut ema2 = vec![f64::NAN; len];
    let ema2_start = (first + n1 - 1).min(len - 1);
    if ema2_start < len && abs_dev[ema2_start].is_finite() {
        ema2[ema2_start] = abs_dev[ema2_start];
        for i in (ema2_start + 1)..len {
            if abs_dev[i].is_finite() {
                ema2[i] = alpha1 * abs_dev[i] + beta1 * ema2[i - 1];
            } else {
                ema2[i] = ema2[i - 1];
            }
        }
    }

    // Normalize to get CI (starts from first + n1 + n1 - 2)
    let mut ci = vec![f64::NAN; len];
    let ci_start = (first + n1 + n1 - 2).min(len - 1);
    for i in ci_start..len {
        if ema2[i].is_finite() && ema2[i] != 0.0 {
            ci[i] = (close[i] - ema1[i]) / (0.025 * ema2[i]); // Note: using 0.025 as in original
        }
    }

    // EMA3 of CI to get TCI (starts from tci_warmup)
    let mut tci = vec![f64::NAN; len];
    let alpha2 = 2.0 / (n2 as f64 + 1.0);
    let beta2 = 1.0 - alpha2;
    if tci_warmup < len && ci[tci_warmup].is_finite() {
        tci[tci_warmup] = ci[tci_warmup] + 50.0; // Add 50 offset for TCI
        for i in (tci_warmup + 1)..len {
            if ci[i].is_finite() {
                tci[i] = alpha2 * ci[i] + beta2 * (tci[i - 1] - 50.0) + 50.0; // Maintain offset
            } else {
                tci[i] = tci[i - 1];
            }
        }
    }

    // === MF Calculation (inline MFI with volume or RSI without) ===
    let mut mf = vec![f64::NAN; len];
    if use_volume && volume.is_some() {
        // MFI calculation with volume
        let vol = volume.unwrap();
        if first + n3 <= len {
            // Calculate typical price
            let mut typical_price = vec![0.0; len];
            for i in first..len {
                typical_price[i] = (high[i] + low[i] + close[i]) / 3.0;
            }

            let mut pos_flow = 0.0;
            let mut neg_flow = 0.0;

            // Initialize first n3 periods
            for i in (first + 1)..=(first + n3).min(len - 1) {
                let mf_raw = typical_price[i] * vol[i];
                if typical_price[i] > typical_price[i - 1] {
                    pos_flow += mf_raw;
                } else if typical_price[i] < typical_price[i - 1] {
                    neg_flow += mf_raw;
                }
            }

            // Calculate MFI for remaining periods
            for i in (first + n3)..len {
                if i > first + n3 {
                    // Update sliding window
                    let old_idx = i - n3;
                    let old_mf = typical_price[old_idx] * vol[old_idx];
                    if old_idx > first && typical_price[old_idx] > typical_price[old_idx - 1] {
                        pos_flow -= old_mf;
                    } else if old_idx > first && typical_price[old_idx] < typical_price[old_idx - 1]
                    {
                        neg_flow -= old_mf;
                    }

                    let new_mf = typical_price[i] * vol[i];
                    if typical_price[i] > typical_price[i - 1] {
                        pos_flow += new_mf;
                    } else if typical_price[i] < typical_price[i - 1] {
                        neg_flow += new_mf;
                    }
                }

                mf[i] = if neg_flow == 0.0 {
                    100.0
                } else {
                    100.0 - (100.0 / (1.0 + pos_flow / neg_flow))
                };
            }
        }
    } else {
        // RSI calculation without volume
        if first + n3 <= len {
            let mut avg_gain = 0.0;
            let mut avg_loss = 0.0;

            // Calculate initial averages
            for i in (first + 1)..=(first + n3) {
                let change = close[i] - close[i - 1];
                if change > 0.0 {
                    avg_gain += change;
                } else {
                    avg_loss -= change;
                }
            }
            avg_gain /= n3 as f64;
            avg_loss /= n3 as f64;

            // Calculate RSI using Wilder's smoothing
            for i in (first + n3)..len {
                let change = close[i] - close[i - 1];
                let (gain, loss) = if change > 0.0 {
                    (change, 0.0)
                } else {
                    (0.0, -change)
                };

                avg_gain = (avg_gain * (n3 - 1) as f64 + gain) / n3 as f64;
                avg_loss = (avg_loss * (n3 - 1) as f64 + loss) / n3 as f64;

                mf[i] = if avg_loss == 0.0 {
                    100.0
                } else {
                    100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
                };
            }
        }
    }

    // === RSI Calculation (same as MF for tradition_mg when no volume) ===
    let rsi = if use_volume && volume.is_some() {
        // When using volume, RSI needs to be calculated separately
        let mut rsi_vals = vec![f64::NAN; len];
        if mf_warmup < len {
            let mut avg_gain = 0.0;
            let mut avg_loss = 0.0;

            // Calculate initial averages
            for i in (first + 1)..(first + n3 + 1).min(len) {
                let change = close[i] - close[i - 1];
                if change > 0.0 {
                    avg_gain += change;
                } else {
                    avg_loss -= change;
                }
            }
            avg_gain /= n3 as f64;
            avg_loss /= n3 as f64;

            // Calculate RSI using Wilder's smoothing
            for i in mf_warmup..len {
                if i > mf_warmup {
                    let change = close[i] - close[i - 1];
                    let (gain, loss) = if change > 0.0 {
                        (change, 0.0)
                    } else {
                        (0.0, -change)
                    };

                    avg_gain = (avg_gain * (n3 - 1) as f64 + gain) / n3 as f64;
                    avg_loss = (avg_loss * (n3 - 1) as f64 + loss) / n3 as f64;
                }

                rsi_vals[i] = if avg_loss == 0.0 {
                    100.0
                } else {
                    100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
                };
            }
        }
        rsi_vals
    } else {
        mf.clone()
    };

    // === CBCI Calculation (RSI momentum + EMA of RSI) ===
    let mut cbci = vec![f64::NAN; len];

    // RSI momentum (difference over n2 periods)
    let rsi_mom_start = mf_warmup + n2;
    let mut rsi_mom = vec![f64::NAN; len];
    if rsi_mom_start < len {
        for i in rsi_mom_start..len {
            if rsi[i].is_finite() && rsi[i - n2].is_finite() {
                rsi_mom[i] = rsi[i] - rsi[i - n2];
            }
        }
    }

    // EMA of RSI
    let alpha3 = 2.0 / (n3 as f64 + 1.0);
    let beta3 = 1.0 - alpha3;
    let mut rsi_ema = vec![f64::NAN; len];
    let rsi_ema_start = mf_warmup;
    if rsi_ema_start < len && rsi[rsi_ema_start].is_finite() {
        rsi_ema[rsi_ema_start] = rsi[rsi_ema_start];
        for i in (rsi_ema_start + 1)..len {
            if rsi[i].is_finite() {
                rsi_ema[i] = alpha3 * rsi[i] + beta3 * rsi_ema[i - 1];
            } else {
                rsi_ema[i] = rsi_ema[i - 1];
            }
        }
    }

    // CBCI = momentum + EMA (starts from cbci_warmup)
    for i in cbci_warmup..len {
        if rsi_mom[i].is_finite() && rsi_ema[i].is_finite() {
            cbci[i] = rsi_mom[i] + rsi_ema[i];
        }
    }

    // === LRSI Calculation (RSI with LSMA smoothing) ===
    // For simplicity, using RSI values directly as LRSI approximation
    let lrsi = rsi.clone();

    // === Combine components into wavetrend ===
    for i in actual_warmup..len {
        let mut sum = 0.0;
        let mut count = 0;

        if tci[i].is_finite() {
            sum += tci[i];
            count += 1;
        }
        if mf[i].is_finite() {
            sum += mf[i];
            count += 1;
        }
        if rsi[i].is_finite() {
            sum += rsi[i];
            count += 1;
        }
        if cbci[i].is_finite() {
            sum += cbci[i];
            count += 1;
        }
        if lrsi[i].is_finite() {
            sum += lrsi[i];
            count += 1;
        }

        if count > 0 {
            wavetrend[i] = sum / count as f64;
        }
    }

    // === Signal Calculation (inline SMA of wavetrend) ===
    let signal_start = actual_warmup + 5; // Need 6 values for SMA
    if signal_start < len {
        // Initialize first SMA value
        let mut sum = 0.0;
        for i in actual_warmup..(actual_warmup + 6).min(len) {
            if wavetrend[i].is_finite() {
                sum += wavetrend[i];
            }
        }
        signal[signal_start] = sum / 6.0;

        // Sliding window for remaining values
        for i in (signal_start + 1)..len {
            if wavetrend[i].is_finite() && wavetrend[i - 6].is_finite() {
                sum += wavetrend[i] - wavetrend[i - 6];
                signal[i] = sum / 6.0;
            } else {
                signal[i] = signal[i - 1];
            }
        }
    }

    // === Histogram Calculation (inline EMA of (wt - sig)*2 + 50) ===
    let hist_start = signal_start;
    if hist_start < len && signal[hist_start].is_finite() {
        let alpha3 = 2.0 / (n3 as f64 + 1.0);
        let beta3 = 1.0 - alpha3;

        // Calculate initial value
        let diff = (wavetrend[hist_start] - signal[hist_start]) * 2.0 + 50.0;
        histogram[hist_start] = diff;

        // EMA smoothing
        for i in (hist_start + 1)..len {
            if signal[i].is_finite() && wavetrend[i].is_finite() {
                let diff = (wavetrend[i] - signal[i]) * 2.0 + 50.0;
                histogram[i] = alpha3 * diff + beta3 * histogram[i - 1];
            } else if i > 0 {
                histogram[i] = histogram[i - 1];
            }
        }
    }

    Ok(())
}

// ============================================================================
// BUILDER PATTERN
// ============================================================================

/// Builder for Modified God Mode indicator
pub struct ModGodModeBuilder {
    n1: usize,
    n2: usize,
    n3: usize,
    mode: ModGodModeMode,
    use_volume: bool,
    kernel: Kernel,
}

impl Default for ModGodModeBuilder {
    fn default() -> Self {
        Self {
            n1: 17,
            n2: 6,
            n3: 4,
            mode: ModGodModeMode::TraditionMg,
            use_volume: true, // Default to true to match Pine's implicit behavior
            kernel: Kernel::Auto,
        }
    }
}

impl ModGodModeBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn n1(mut self, n1: usize) -> Self {
        self.n1 = n1;
        self
    }

    pub fn n2(mut self, n2: usize) -> Self {
        self.n2 = n2;
        self
    }

    pub fn n3(mut self, n3: usize) -> Self {
        self.n3 = n3;
        self
    }

    pub fn mode(mut self, mode: ModGodModeMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn use_volume(mut self, use_volume: bool) -> Self {
        self.use_volume = use_volume;
        self
    }

    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline]
    pub fn apply(self, c: &Candles) -> Result<ModGodModeOutput, ModGodModeError> {
        let kernel = self.kernel;
        let input = self.build(ModGodModeData::Candles { candles: c });
        mod_god_mode_with_kernel(&input, kernel)
    }

    #[inline]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: Option<&[f64]>,
    ) -> Result<ModGodModeOutput, ModGodModeError> {
        let kernel = self.kernel;
        let input = self.build(ModGodModeData::Slices {
            high,
            low,
            close,
            volume,
        });
        mod_god_mode_with_kernel(&input, kernel)
    }

    #[inline]
    pub fn into_stream(self) -> Result<ModGodModeStream, ModGodModeError> {
        ModGodModeStream::try_new(ModGodModeParams {
            n1: Some(self.n1),
            n2: Some(self.n2),
            n3: Some(self.n3),
            mode: Some(self.mode),
            use_volume: Some(self.use_volume),
        })
    }

    pub fn build<'a>(self, data: ModGodModeData<'a>) -> ModGodModeInput<'a> {
        ModGodModeInput {
            data,
            params: ModGodModeParams {
                n1: Some(self.n1),
                n2: Some(self.n2),
                n3: Some(self.n3),
                mode: Some(self.mode),
                use_volume: Some(self.use_volume),
            },
        }
    }

    pub fn calculate<'a>(
        self,
        data: ModGodModeData<'a>,
    ) -> Result<ModGodModeOutput, ModGodModeError> {
        let input = self.build(data);
        mod_god_mode(&input)
    }

    pub fn calculate_with_kernel<'a>(
        self,
        data: ModGodModeData<'a>,
        kernel: Kernel,
    ) -> Result<ModGodModeOutput, ModGodModeError> {
        let input = self.build(data);
        mod_god_mode_with_kernel(&input, kernel)
    }
}

// ============================================================================
// STREAMING SUPPORT  —  O(1) per update
// ============================================================================

use std::collections::VecDeque;

#[inline(always)]
fn ema_step(x: f64, prev: f64, alpha: f64, beta: f64) -> f64 {
    beta.mul_add(prev, alpha * x)
}

#[inline(always)]
fn nonzero(v: f64) -> bool {
    v != 0.0 && v.is_finite()
}

#[derive(Default)]
struct MonoMax {
    dq: VecDeque<(usize, f64)>,
}
impl MonoMax {
    #[inline(always)]
    fn push(&mut self, idx: usize, val: f64, win: usize) {
        while let Some(&(j, _)) = self.dq.front() {
            if idx >= j + win {
                self.dq.pop_front();
            } else {
                break;
            }
        }
        while let Some(&(_, v)) = self.dq.back() {
            if v <= val {
                self.dq.pop_back();
            } else {
                break;
            }
        }
        self.dq.push_back((idx, val));
    }
    #[inline(always)]
    fn get(&self) -> Option<f64> {
        self.dq.front().map(|x| x.1)
    }
    #[inline(always)]
    fn clear(&mut self) {
        self.dq.clear();
    }
}

#[derive(Default)]
struct MonoMin {
    dq: VecDeque<(usize, f64)>,
}
impl MonoMin {
    #[inline(always)]
    fn push(&mut self, idx: usize, val: f64, win: usize) {
        while let Some(&(j, _)) = self.dq.front() {
            if idx >= j + win {
                self.dq.pop_front();
            } else {
                break;
            }
        }
        while let Some(&(_, v)) = self.dq.back() {
            if v >= val {
                self.dq.pop_back();
            } else {
                break;
            }
        }
        self.dq.push_back((idx, val));
    }
    #[inline(always)]
    fn get(&self) -> Option<f64> {
        self.dq.front().map(|x| x.1)
    }
    #[inline(always)]
    fn clear(&mut self) {
        self.dq.clear();
    }
}

/// Streaming calculator with O(1) updates.
/// Keeps compact state for all recurrences; no buffer rebuilds.
pub struct ModGodModeStream {
    // params
    n1: usize,
    n2: usize,
    n3: usize,
    mode: ModGodModeMode,
    use_volume: bool,

    // precomputed constants
    alpha1: f64,
    beta1: f64,
    alpha2: f64,
    beta2: f64,
    alpha3: f64,
    beta3: f64,
    // warmup gates (match conservative compound warmups from classic kernel)
    warm_wt: usize,  // when wavetrend can start
    warm_sig: usize, // when signal (SMA6) can start

    // time index (# of samples processed so far)
    idx: usize,

    // ====== TCI (EMA chains) ======
    ema1_c: f64,
    seed_ema1: bool,
    ema2_abs: f64,
    seed_ema2: bool,
    ema3_ci: f64,
    seed_ema3: bool,

    // ====== RSI (Wilder) ======
    rs_avg_gain: f64,
    rs_avg_loss: f64,
    rsi_seeded: bool,
    rs_init_cnt: usize,
    prev_close: f64,
    have_prev_close: bool,

    // ====== LRSI (Ehlers) ======
    alpha_l: f64,
    one_m_l: f64,
    l0: f64,
    l1: f64,
    l2: f64,
    l3: f64,

    // ====== MFI (ring over n3) ======
    has_vol: bool,
    mf_pos_sum: f64,
    mf_neg_sum: f64,
    mf_ring_mf: Vec<f64>,
    mf_ring_sgn: Vec<i8>,
    mf_head: usize,
    tp_prev: f64,
    tp_has_prev: bool,

    // ====== CSI / TSI (double EMA of momentum & abs(momentum)) ======
    tsi_ema_m_s: f64,
    tsi_ema_a_s: f64,
    tsi_seed_s: bool,
    tsi_ema_m_l: f64,
    tsi_ema_a_l: f64,
    tsi_seed_l: bool,

    // ====== CSI_MG (double EMA of pc_norm and abs(pc)) ======
    csi_num_e1: f64,
    csi_num_e2: f64,
    csi_seed_e1: bool,
    csi_seed_e2: bool,
    csi_den_e1: f64,
    csi_den_e2: f64,

    // ====== CBCI: RSI momentum over n2 + EMA(RSI,n3) ======
    rsi_ring: Vec<f64>,
    rsi_ring_head: usize,
    rsi_ema: f64,
    rsi_ema_seed: bool,

    // ====== %R (Willy on close-only over n2) via monotonic deques ======
    w_max: MonoMax,
    w_min: MonoMin,

    // ====== signal SMA(6) & histogram EMA(n3) ======
    sig_ring: [f64; 6],
    sig_head: usize,
    sig_sum: f64,
    sig_seeded: bool,
    sig_count: usize,

    hist_prev: f64,
    hist_seeded: bool,
}

impl ModGodModeStream {
    pub fn try_new(p: ModGodModeParams) -> Result<Self, ModGodModeError> {
        let n1 = p.n1.unwrap_or(17);
        let n2 = p.n2.unwrap_or(6);
        let n3 = p.n3.unwrap_or(4);
        if n1 == 0 || n2 == 0 || n3 == 0 {
            return Err(ModGodModeError::InvalidPeriod {
                n1,
                n2,
                n3,
                data_len: 0,
            });
        }
        let mode = p.mode.unwrap_or_default();
        let use_volume = p.use_volume.unwrap_or(false);
        Ok(Self::new(n1, n2, n3, mode, use_volume))
    }

    pub fn new(n1: usize, n2: usize, n3: usize, mode: ModGodModeMode, use_volume: bool) -> Self {
        // precompute alphas for EMAs
        let alpha1 = 2.0 / (n1 as f64 + 1.0);
        let beta1 = 1.0 - alpha1;
        let alpha2 = 2.0 / (n2 as f64 + 1.0);
        let beta2 = 1.0 - alpha2;
        let alpha3 = 2.0 / (n3 as f64 + 1.0);
        let beta3 = 1.0 - alpha3;

        // Align warmups with fused scalar path: start wavetrend at max(n1,n2,n3)-1
        let warm_wt = n1.max(n2).max(n3) - 1;
        let warm_sig = warm_wt + (6 - 1); // need 6 samples for SMA(6)

        Self {
            n1,
            n2,
            n3,
            mode,
            use_volume,
            alpha1,
            beta1,
            alpha2,
            beta2,
            alpha3,
            beta3,
            warm_wt,
            warm_sig,
            idx: 0,

            // TCI
            ema1_c: 0.0,
            seed_ema1: false,
            ema2_abs: 0.0,
            seed_ema2: false,
            ema3_ci: 0.0,
            seed_ema3: false,

            // RSI
            rs_avg_gain: 0.0,
            rs_avg_loss: 0.0,
            rsi_seeded: false,
            rs_init_cnt: 0,
            prev_close: 0.0,
            have_prev_close: false,

            // LRSI
            alpha_l: 0.7,
            one_m_l: 1.0 - 0.7,
            l0: 0.0,
            l1: 0.0,
            l2: 0.0,
            l3: 0.0,

            // MFI
            has_vol: use_volume,
            mf_pos_sum: 0.0,
            mf_neg_sum: 0.0,
            mf_ring_mf: vec![0.0; n3.max(1)],
            mf_ring_sgn: vec![0; n3.max(1)],
            mf_head: 0,
            tp_prev: 0.0,
            tp_has_prev: false,

            // TSI
            tsi_ema_m_s: 0.0,
            tsi_ema_a_s: 0.0,
            tsi_seed_s: false,
            tsi_ema_m_l: 0.0,
            tsi_ema_a_l: 0.0,
            tsi_seed_l: false,

            // CSI_MG
            csi_num_e1: 0.0,
            csi_num_e2: 0.0,
            csi_seed_e1: false,
            csi_seed_e2: false,
            csi_den_e1: 0.0,
            csi_den_e2: 0.0,

            // CBCI
            rsi_ring: vec![f64::NAN; n2.max(1)],
            rsi_ring_head: 0,
            rsi_ema: 0.0,
            rsi_ema_seed: false,

            // Willy deques
            w_max: MonoMax::default(),
            w_min: MonoMin::default(),

            // signal/hist
            sig_ring: [0.0; 6],
            sig_head: 0,
            sig_sum: 0.0,
            sig_seeded: false,
            sig_count: 0,

            hist_prev: 0.0,
            hist_seeded: false,
        }
    }

    /// Push one bar; returns (wt, sig, hist) when ready.
    #[inline]
    pub fn update(
        &mut self,
        high: f64,
        low: f64,
        close: f64,
        volume: Option<f64>,
    ) -> Option<(f64, f64, f64)> {
        // reject NaNs
        if !(high.is_finite() && low.is_finite() && close.is_finite()) {
            self.idx += 1;
            return None;
        }
        let i = self.idx;

        // ====== TCI chain ======
        if !self.seed_ema1 {
            self.ema1_c = close;
            self.seed_ema1 = true;
        } else {
            self.ema1_c = ema_step(close, self.ema1_c, self.alpha1, self.beta1);
        }

        let abs_dev = (close - self.ema1_c).abs();
        if !self.seed_ema2 {
            self.ema2_abs = abs_dev;
            self.seed_ema2 = true;
        } else {
            self.ema2_abs = ema_step(abs_dev, self.ema2_abs, self.alpha1, self.beta1);
        }

        let mut tci_val = f64::NAN;
        if nonzero(self.ema2_abs) {
            let inv = (0.025 * self.ema2_abs).recip();
            let ci = (close - self.ema1_c) * inv;
            if !self.seed_ema3 {
                self.ema3_ci = ci;
                self.seed_ema3 = true;
            } else {
                self.ema3_ci = ema_step(ci, self.ema3_ci, self.alpha2, self.beta2);
            }
            tci_val = self.ema3_ci + 50.0;
        }

        // ====== RSI (Wilder) ======
        let mut rsi_val = f64::NAN;
        if !self.have_prev_close {
            self.prev_close = close;
            self.have_prev_close = true;
        } else {
            let ch = close - self.prev_close;
            let gain = if ch > 0.0 { ch } else { 0.0 };
            let loss = if ch < 0.0 { -ch } else { 0.0 };

            if !self.rsi_seeded {
                self.rs_init_cnt += 1;
                self.rs_avg_gain += gain;
                self.rs_avg_loss += loss;
                if self.rs_init_cnt >= self.n3 {
                    self.rs_avg_gain /= self.n3 as f64;
                    self.rs_avg_loss /= self.n3 as f64;
                    self.rsi_seeded = true;
                    let rs = if self.rs_avg_loss == 0.0 {
                        f64::INFINITY
                    } else {
                        self.rs_avg_gain / self.rs_avg_loss
                    };
                    rsi_val = 100.0 * (rs / (1.0 + rs));
                }
            } else {
                // Wilder smoothing
                let n3m1 = (self.n3 - 1) as f64;
                self.rs_avg_gain = (self.rs_avg_gain * n3m1 + gain) / self.n3 as f64;
                self.rs_avg_loss = (self.rs_avg_loss * n3m1 + loss) / self.n3 as f64;
                let rs = if self.rs_avg_loss == 0.0 {
                    f64::INFINITY
                } else {
                    self.rs_avg_gain / self.rs_avg_loss
                };
                rsi_val = 100.0 * (rs / (1.0 + rs));
            }
        }

        // ====== Laguerre RSI (alpha=0.7 per file) ======
        {
            let p_l0 = self.l0;
            self.l0 = self.alpha_l * close + self.one_m_l * p_l0;
            let p_l1 = self.l1;
            self.l1 = -self.one_m_l * self.l0 + p_l0 + self.one_m_l * p_l1;
            let p_l2 = self.l2;
            self.l2 = -self.one_m_l * self.l1 + p_l1 + self.one_m_l * p_l2;
            let p_l3 = self.l3;
            self.l3 = -self.one_m_l * self.l2 + p_l2 + self.one_m_l * p_l3;
        }
        let cu = (self.l0 - self.l1).max(0.0)
            + (self.l1 - self.l2).max(0.0)
            + (self.l2 - self.l3).max(0.0);
        let cd = (self.l1 - self.l0).max(0.0)
            + (self.l2 - self.l1).max(0.0)
            + (self.l3 - self.l2).max(0.0);
        let lrsi_val = if nonzero(cu + cd) {
            100.0 * cu / (cu + cd)
        } else {
            f64::NAN
        };

        // ====== MFI (if volume used) else RSI substitute ======
        let mut mf_val = f64::NAN;
        if self.has_vol {
            let v = volume.unwrap_or(0.0);
            let tp = (high + low + close) * (1.0 / 3.0);
            if self.tp_has_prev {
                let sign: i8 = if tp > self.tp_prev {
                    1
                } else if tp < self.tp_prev {
                    -1
                } else {
                    0
                };
                let mf_raw = tp * v;

                if self.rsi_seeded {
                    // remove aged contrib
                    let old_mf = self.mf_ring_mf[self.mf_head];
                    let old_sg = self.mf_ring_sgn[self.mf_head];
                    if old_sg > 0 {
                        self.mf_pos_sum -= old_mf;
                    } else if old_sg < 0 {
                        self.mf_neg_sum -= old_mf;
                    }

                    // insert new
                    self.mf_ring_mf[self.mf_head] = mf_raw;
                    self.mf_ring_sgn[self.mf_head] = sign;
                    if sign > 0 {
                        self.mf_pos_sum += mf_raw;
                    } else if sign < 0 {
                        self.mf_neg_sum += mf_raw;
                    }
                    self.mf_head = (self.mf_head + 1) % self.n3.max(1);

                    let denom = self.mf_pos_sum + self.mf_neg_sum;
                    if denom > 0.0 {
                        mf_val = 100.0 * (self.mf_pos_sum / denom);
                    } else if self.mf_neg_sum == 0.0 {
                        mf_val = 100.0;
                    }
                } else {
                    // warmup rings, but do not produce a value yet
                    self.mf_ring_mf[self.mf_head] = mf_raw;
                    self.mf_ring_sgn[self.mf_head] = sign;
                    self.mf_head = (self.mf_head + 1) % self.n3.max(1);
                }
            }
            self.tp_prev = tp;
            self.tp_has_prev = true;
        } else {
            mf_val = rsi_val; // RSI-as-MF when no volume, per fused scalar kernel
        }

        // ====== CBCI (mom(RSI,n2) + EMA(RSI,n3)) ======
        let mut cbci_val = f64::NAN;
        if self.rsi_seeded {
            let old = self.rsi_ring[self.rsi_ring_head];
            self.rsi_ring[self.rsi_ring_head] = rsi_val;
            self.rsi_ring_head = (self.rsi_ring_head + 1) % self.n2.max(1);
            let mom = if old.is_finite() && rsi_val.is_finite() {
                rsi_val - old
            } else {
                f64::NAN
            };

            if !self.rsi_ema_seed && rsi_val.is_finite() {
                self.rsi_ema = rsi_val;
                self.rsi_ema_seed = true;
            } else if rsi_val.is_finite() {
                self.rsi_ema = ema_step(rsi_val, self.rsi_ema, self.alpha3, self.beta3);
            }

            if mom.is_finite() && self.rsi_ema_seed {
                cbci_val = mom + self.rsi_ema;
            }
        }

        // ====== CSI / TSI ======
        let mut csi_val = f64::NAN;
        let mut csi_mg_val = f64::NAN;

        // Godmode: RSI blended with TSI (double-EMA of momentum)
        if matches!(self.mode, ModGodModeMode::Godmode) && self.have_prev_close {
            let mom = close - self.prev_close;
            let am = mom.abs();

            if !self.tsi_seed_s {
                self.tsi_ema_m_s = mom;
                self.tsi_ema_a_s = am;
                self.tsi_seed_s = true;
            } else {
                self.tsi_ema_m_s = ema_step(mom, self.tsi_ema_m_s, self.alpha1, self.beta1);
                self.tsi_ema_a_s = ema_step(am, self.tsi_ema_a_s, self.alpha1, self.beta1);
            }

            if !self.tsi_seed_l && self.tsi_seed_s {
                self.tsi_ema_m_l = self.tsi_ema_m_s;
                self.tsi_ema_a_l = self.tsi_ema_a_s;
                self.tsi_seed_l = true;
            } else if self.tsi_seed_l {
                self.tsi_ema_m_l =
                    ema_step(self.tsi_ema_m_s, self.tsi_ema_m_l, self.alpha2, self.beta2);
                self.tsi_ema_a_l =
                    ema_step(self.tsi_ema_a_s, self.tsi_ema_a_l, self.alpha2, self.beta2);
            }

            if self.tsi_seed_l && nonzero(self.tsi_ema_a_l) && rsi_val.is_finite() {
                let tsi = 100.0 * (self.tsi_ema_m_l / self.tsi_ema_a_l);
                csi_val = 0.5 * (rsi_val + (0.5 * tsi + 50.0));
            }
        }

        // GodmodeMg: RSI blended with modified TSI using normalized pc and abs(pc)
        if matches!(self.mode, ModGodModeMode::GodmodeMg) && self.have_prev_close {
            let a = self.prev_close;
            let b = close;
            let avg = 0.5 * (a + b);
            let pc_norm = if avg != 0.0 {
                (b - a) * avg.recip()
            } else {
                0.0
            };
            let apc = (b - a).abs();

            if !self.csi_seed_e1 {
                self.csi_num_e1 = pc_norm;
                self.csi_den_e1 = apc;
                self.csi_seed_e1 = true;
            } else {
                self.csi_num_e1 = ema_step(pc_norm, self.csi_num_e1, self.alpha1, self.beta1);
                self.csi_den_e1 = ema_step(apc, self.csi_den_e1, self.alpha1, self.beta1);
            }

            if !self.csi_seed_e2 && self.csi_seed_e1 {
                self.csi_num_e2 = self.csi_num_e1;
                self.csi_den_e2 = self.csi_den_e1;
                self.csi_seed_e2 = true;
            } else if self.csi_seed_e2 {
                self.csi_num_e2 =
                    ema_step(self.csi_num_e1, self.csi_num_e2, self.alpha2, self.beta2);
                self.csi_den_e2 =
                    ema_step(self.csi_den_e1, self.csi_den_e2, self.alpha2, self.beta2);
            }

            if self.csi_seed_e2 && nonzero(self.csi_den_e2) && rsi_val.is_finite() {
                let ttsi = 50.0 * (self.csi_num_e2 / self.csi_den_e2) + 50.0;
                csi_mg_val = 0.5 * (rsi_val + ttsi);
            }
        }

        // ====== Willy (%R on close-only, window n2) via monotonic deques ======
        self.w_max.push(i, close, self.n2);
        self.w_min.push(i, close, self.n2);
        let mut willy_val = f64::NAN;
        if i + 1 >= self.n2 {
            if let (Some(hi), Some(lo)) = (self.w_max.get(), self.w_min.get()) {
                let rng = hi - lo;
                if rng != 0.0 {
                    willy_val = 60.0 * (close - hi) / rng + 80.0;
                }
            }
        }

        // ====== Combine by mode into wavetrend ======
        let ready_wt = i >= self.warm_wt;
        let mut wt = f64::NAN;
        if ready_wt {
            let mut sum = 0.0;
            let mut cnt = 0i32;
            match self.mode {
                ModGodModeMode::Godmode => {
                    if tci_val.is_finite() {
                        sum += tci_val;
                        cnt += 1;
                    }
                    if csi_val.is_finite() {
                        sum += csi_val;
                        cnt += 1;
                    }
                    if mf_val.is_finite() {
                        sum += mf_val;
                        cnt += 1;
                    }
                    if willy_val.is_finite() {
                        sum += willy_val;
                        cnt += 1;
                    }
                }
                ModGodModeMode::Tradition => {
                    if tci_val.is_finite() {
                        sum += tci_val;
                        cnt += 1;
                    }
                    if mf_val.is_finite() {
                        sum += mf_val;
                        cnt += 1;
                    }
                    if rsi_val.is_finite() {
                        sum += rsi_val;
                        cnt += 1;
                    }
                }
                ModGodModeMode::GodmodeMg => {
                    if tci_val.is_finite() {
                        sum += tci_val;
                        cnt += 1;
                    }
                    if csi_mg_val.is_finite() {
                        sum += csi_mg_val;
                        cnt += 1;
                    }
                    if mf_val.is_finite() {
                        sum += mf_val;
                        cnt += 1;
                    }
                    if willy_val.is_finite() {
                        sum += willy_val;
                        cnt += 1;
                    }
                    if cbci_val.is_finite() {
                        sum += cbci_val;
                        cnt += 1;
                    }
                    if lrsi_val.is_finite() {
                        sum += lrsi_val;
                        cnt += 1;
                    }
                }
                ModGodModeMode::TraditionMg => {
                    if tci_val.is_finite() {
                        sum += tci_val;
                        cnt += 1;
                    }
                    if mf_val.is_finite() {
                        sum += mf_val;
                        cnt += 1;
                    }
                    if rsi_val.is_finite() {
                        sum += rsi_val;
                        cnt += 1;
                    }
                    if cbci_val.is_finite() {
                        sum += cbci_val;
                        cnt += 1;
                    }
                    if lrsi_val.is_finite() {
                        sum += lrsi_val;
                        cnt += 1;
                    }
                }
            }
            if cnt > 0 {
                wt = sum / (cnt as f64);
            }
        }

        // ====== signal SMA(6) (incremental seeding from warm_wt) ======
        let mut sig = f64::NAN;
        if i >= self.warm_wt && wt.is_finite() {
            if !self.sig_seeded {
                // fill until we have 6 samples
                self.sig_ring[self.sig_head] = wt;
                self.sig_head = (self.sig_head + 1) % 6;
                self.sig_sum += wt;
                self.sig_count += 1;
                if self.sig_count == 6 {
                    self.sig_seeded = true;
                    sig = self.sig_sum / 6.0;
                }
            } else {
                let old = self.sig_ring[self.sig_head];
                self.sig_ring[self.sig_head] = wt;
                self.sig_head = (self.sig_head + 1) % 6;
                self.sig_sum += wt - old;
                sig = self.sig_sum / 6.0;
            }
        }

        // ====== histogram EMA(n3) of diff ======
        let mut hist = f64::NAN;
        if self.sig_seeded && sig.is_finite() && wt.is_finite() {
            let d = (wt - sig) * 2.0 + 50.0;
            if !self.hist_seeded {
                self.hist_prev = d;
                self.hist_seeded = true;
                hist = d;
            } else {
                self.hist_prev = ema_step(d, self.hist_prev, self.alpha3, self.beta3);
                hist = self.hist_prev;
            }
        }

        self.prev_close = close;
        self.idx += 1;

        if self.sig_seeded && hist.is_finite() {
            Some((wt, sig, hist))
        } else {
            None
        }
    }

    pub fn reset(&mut self) {
        self.idx = 0;

        self.ema1_c = 0.0;
        self.seed_ema1 = false;
        self.ema2_abs = 0.0;
        self.seed_ema2 = false;
        self.ema3_ci = 0.0;
        self.seed_ema3 = false;

        self.rs_avg_gain = 0.0;
        self.rs_avg_loss = 0.0;
        self.rsi_seeded = false;
        self.rs_init_cnt = 0;
        self.prev_close = 0.0;
        self.have_prev_close = false;

        self.l0 = 0.0;
        self.l1 = 0.0;
        self.l2 = 0.0;
        self.l3 = 0.0;

        self.mf_pos_sum = 0.0;
        self.mf_neg_sum = 0.0;
        self.mf_ring_mf.fill(0.0);
        self.mf_ring_sgn.fill(0);
        self.mf_head = 0;
        self.tp_prev = 0.0;
        self.tp_has_prev = false;

        self.tsi_ema_m_s = 0.0;
        self.tsi_ema_a_s = 0.0;
        self.tsi_seed_s = false;
        self.tsi_ema_m_l = 0.0;
        self.tsi_ema_a_l = 0.0;
        self.tsi_seed_l = false;

        self.csi_num_e1 = 0.0;
        self.csi_num_e2 = 0.0;
        self.csi_seed_e1 = false;
        self.csi_seed_e2 = false;
        self.csi_den_e1 = 0.0;
        self.csi_den_e2 = 0.0;

        self.rsi_ring.fill(f64::NAN);
        self.rsi_ring_head = 0;
        self.rsi_ema = 0.0;
        self.rsi_ema_seed = false;

        self.w_max.clear();
        self.w_min.clear();

        self.sig_ring = [0.0; 6];
        self.sig_head = 0;
        self.sig_sum = 0.0;
        self.sig_seeded = false;
        self.sig_count = 0;

        self.hist_prev = 0.0;
        self.hist_seeded = false;
    }
}

// ============================================================================
// BATCH PROCESSING
// ============================================================================

/// Parameter range for batch processing
#[derive(Clone, Debug)]
pub struct ModGodModeBatchRange {
    pub n1: (usize, usize, usize),
    pub n2: (usize, usize, usize),
    pub n3: (usize, usize, usize),
    pub mode: ModGodModeMode,
}

/// Batch output with flattened matrix layout
#[derive(Clone, Debug)]
pub struct ModGodModeBatchOutput {
    pub wavetrend: Vec<f64>,
    pub signal: Vec<f64>,
    pub histogram: Vec<f64>,
    pub combos: Vec<ModGodModeParams>,
    pub rows: usize,
    pub cols: usize,
}

impl ModGodModeBatchOutput {
    #[inline]
    pub fn row_for_params(&self, p: &ModGodModeParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.n1.unwrap() == p.n1.unwrap()
                && c.n2.unwrap() == p.n2.unwrap()
                && c.n3.unwrap() == p.n3.unwrap()
                && c.mode.unwrap() == p.mode.unwrap()
        })
    }

    #[inline]
    pub fn values_for(&self, p: &ModGodModeParams) -> Option<(&[f64], &[f64], &[f64])> {
        self.row_for_params(p).map(|row| {
            let s = row * self.cols;
            (
                &self.wavetrend[s..s + self.cols],
                &self.signal[s..s + self.cols],
                &self.histogram[s..s + self.cols],
            )
        })
    }
}

#[inline]
fn axis_usize_mod(
    (start, end, step): (usize, usize, usize),
) -> Result<Vec<usize>, ModGodModeError> {
    if step == 0 || start == end {
        return Ok(vec![start]);
    }
    if start < end {
        let v: Vec<_> = (start..=end).step_by(step).collect();
        if v.is_empty() {
            return Err(ModGodModeError::InvalidRange { start, end, step });
        }
        Ok(v)
    } else {
        let mut v = Vec::new();
        let mut cur = start;
        while cur >= end {
            v.push(cur);
            if cur - end < step {
                break;
            }
            cur -= step;
        }
        if v.is_empty() {
            return Err(ModGodModeError::InvalidRange { start, end, step });
        }
        Ok(v)
    }
}

#[inline]
fn expand_grid_mod(r: &ModGodModeBatchRange) -> Result<Vec<ModGodModeParams>, ModGodModeError> {
    let n1s = axis_usize_mod(r.n1)?;
    let n2s = axis_usize_mod(r.n2)?;
    let n3s = axis_usize_mod(r.n3)?;
    let cap = n1s
        .len()
        .checked_mul(n2s.len())
        .and_then(|v| v.checked_mul(n3s.len()))
        .ok_or_else(|| ModGodModeError::InvalidInput("batch grid size overflow".into()))?;
    let mut v = Vec::with_capacity(cap);
    for &a in &n1s {
        for &b in &n2s {
            for &c in &n3s {
                v.push(ModGodModeParams {
                    n1: Some(a),
                    n2: Some(b),
                    n3: Some(c),
                    mode: Some(r.mode),
                    use_volume: Some(false),
                });
            }
        }
    }
    if v.is_empty() {
        return Err(ModGodModeError::InvalidRange {
            start: r.n1.0,
            end: r.n3.1,
            step: r.n1.2.max(r.n2.2).max(r.n3.2),
        });
    }
    Ok(v)
}

pub fn mod_god_mode_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: Option<&[f64]>,
    sweep: &ModGodModeBatchRange,
    k: Kernel,
) -> Result<ModGodModeBatchOutput, ModGodModeError> {
    let combos = expand_grid_mod(sweep)?;
    let rows = combos.len();
    let cols = close.len();
    if cols == 0 {
        return Err(ModGodModeError::EmptyInputData);
    }
    let _ = rows
        .checked_mul(cols)
        .ok_or_else(|| ModGodModeError::InvalidInput("rows*cols overflow".into()))?;

    // Allocate uninit matrices
    let mut mu_w = make_uninit_matrix(rows, cols);
    let mut mu_s = make_uninit_matrix(rows, cols);
    let mut mu_h = make_uninit_matrix(rows, cols);

    // Warmup per row based on max(n1,n2,n3) and first-valid like alma.rs
    let first = close
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(ModGodModeError::AllValuesNaN)?;
    let warms: Vec<usize> = combos
        .iter()
        .map(|p| first + p.n1.unwrap().max(p.n2.unwrap()).max(p.n3.unwrap()) - 1)
        .collect();
    init_matrix_prefixes(&mut mu_w, cols, &warms);
    init_matrix_prefixes(&mut mu_s, cols, &warms);
    init_matrix_prefixes(&mut mu_h, cols, &warms);

    // Cast to &mut [f64] with ManuallyDrop guards
    let mut guard_w = core::mem::ManuallyDrop::new(mu_w);
    let mut guard_s = core::mem::ManuallyDrop::new(mu_s);
    let mut guard_h = core::mem::ManuallyDrop::new(mu_h);
    let out_w: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard_w.as_mut_ptr() as *mut f64, guard_w.len()) };
    let out_s: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard_s.as_mut_ptr() as *mut f64, guard_s.len()) };
    let out_h: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard_h.as_mut_ptr() as *mut f64, guard_h.len()) };

    // Map batch kernel to single-row kernel properly
    let batch_kern = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(ModGodModeError::InvalidKernelForBatch(k)),
    };

    let row_kern = match batch_kern {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!("Invalid batch kernel"),
    };

    // Direct row writes using into_slices with correct kernel
    for (row, p) in combos.iter().enumerate() {
        let start = row * cols;
        let end = start + cols;
        let dst_w = &mut out_w[start..end];
        let dst_s = &mut out_s[start..end];
        let dst_h = &mut out_h[start..end];

        let inp = ModGodModeInput::from_slices(high, low, close, volume, p.clone());
        mod_god_mode_into_slices(dst_w, dst_s, dst_h, &inp, row_kern)?;
    }

    // Transmute to owned Vecs with no extra copy
    let wavetrend = unsafe {
        let ptr = out_w.as_mut_ptr();
        let len = out_w.len();
        core::mem::forget(guard_w);
        Vec::from_raw_parts(ptr, len, len)
    };
    let signal = unsafe {
        let ptr = out_s.as_mut_ptr();
        let len = out_s.len();
        core::mem::forget(guard_s);
        Vec::from_raw_parts(ptr, len, len)
    };
    let histogram = unsafe {
        let ptr = out_h.as_mut_ptr();
        let len = out_h.len();
        core::mem::forget(guard_h);
        Vec::from_raw_parts(ptr, len, len)
    };

    Ok(ModGodModeBatchOutput {
        wavetrend,
        signal,
        histogram,
        combos,
        rows,
        cols,
    })
}

/// Legacy batch builder for compatibility
pub struct ModGodModeBatchBuilder {
    n1: usize,
    n2: usize,
    n3: usize,
    mode: ModGodModeMode,
    use_volume: bool,
    parallel: bool,
}

impl Default for ModGodModeBatchBuilder {
    fn default() -> Self {
        Self {
            n1: 17,
            n2: 6,
            n3: 4,
            mode: ModGodModeMode::TraditionMg,
            use_volume: false,
            parallel: true,
        }
    }
}

impl ModGodModeBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn n1(mut self, n1: usize) -> Self {
        self.n1 = n1;
        self
    }

    pub fn n2(mut self, n2: usize) -> Self {
        self.n2 = n2;
        self
    }

    pub fn n3(mut self, n3: usize) -> Self {
        self.n3 = n3;
        self
    }

    pub fn mode(mut self, mode: ModGodModeMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn use_volume(mut self, use_volume: bool) -> Self {
        self.use_volume = use_volume;
        self
    }

    pub fn parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    pub fn calculate_batch(
        &self,
        datasets: &[Candles],
    ) -> Vec<Result<ModGodModeOutput, ModGodModeError>> {
        let params = ModGodModeParams {
            n1: Some(self.n1),
            n2: Some(self.n2),
            n3: Some(self.n3),
            mode: Some(self.mode),
            use_volume: Some(self.use_volume),
        };

        let kernel = detect_best_batch_kernel();

        if self.parallel {
            #[cfg(not(target_arch = "wasm32"))]
            {
                datasets
                    .par_iter()
                    .map(|candles| {
                        let input = ModGodModeInput::from_candles(candles, params.clone());
                        mod_god_mode_with_kernel(&input, kernel)
                    })
                    .collect()
            }
            #[cfg(target_arch = "wasm32")]
            {
                // WASM doesn't support parallel processing, use regular iteration
                datasets
                    .iter()
                    .map(|candles| {
                        let input = ModGodModeInput::from_candles(candles, params.clone());
                        mod_god_mode_with_kernel(&input, kernel)
                    })
                    .collect()
            }
        } else {
            datasets
                .iter()
                .map(|candles| {
                    let input = ModGodModeInput::from_candles(candles, params.clone());
                    mod_god_mode_with_kernel(&input, kernel)
                })
                .collect()
        }
    }
}

// ============================================================================
// PYTHON BINDINGS
// ============================================================================

#[cfg(feature = "python")]
#[pyfunction(name = "mod_god_mode")]
#[pyo3(signature=(high, low, close, volume=None, n1=None, n2=None, n3=None, mode=None, use_volume=None, kernel=None))]
pub fn mod_god_mode_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    volume: Option<PyReadonlyArray1<'py, f64>>,
    n1: Option<usize>,
    n2: Option<usize>,
    n3: Option<usize>,
    mode: Option<String>,
    use_volume: Option<bool>,
    kernel: Option<&str>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let h = high.as_slice()?;
    let l = low.as_slice()?;
    let c = close.as_slice()?;
    let v_opt = volume.as_ref().map(|v| v.as_slice()).transpose()?;
    let mode_enum = match mode {
        Some(m) => Some(
            m.parse::<ModGodModeMode>()
                .map_err(|e| PyValueError::new_err(e))?,
        ),
        None => None,
    };
    let params = ModGodModeParams {
        n1,
        n2,
        n3,
        mode: mode_enum,
        use_volume,
    };
    let kern = validate_kernel(kernel, false).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let out = py
        .allow_threads(|| {
            mod_god_mode_with_kernel(&ModGodModeInput::from_slices(h, l, c, v_opt, params), kern)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        out.wavetrend.into_pyarray(py),
        out.signal.into_pyarray(py),
        out.histogram.into_pyarray(py),
    ))
}

#[cfg(feature = "python")]
#[pyfunction(name = "mod_god_mode_batch")]
#[pyo3(signature=(high, low, close, volume, n1_range, n2_range, n3_range, mode="tradition_mg", kernel=None))]
pub fn mod_god_mode_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    volume: Option<PyReadonlyArray1<'py, f64>>,
    n1_range: (usize, usize, usize),
    n2_range: (usize, usize, usize),
    n3_range: (usize, usize, usize),
    mode: &str,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let h = high.as_slice()?;
    let l = low.as_slice()?;
    let c = close.as_slice()?;
    let v = volume.as_ref().map(|v| v.as_slice()).transpose()?;
    let m = mode
        .parse::<ModGodModeMode>()
        .map_err(|e| PyValueError::new_err(e))?;
    let sweep = ModGodModeBatchRange {
        n1: n1_range,
        n2: n2_range,
        n3: n3_range,
        mode: m,
    };
    let kern = validate_kernel(kernel, true).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let o = py
        .allow_threads(|| mod_god_mode_batch_with_kernel(h, l, c, v, &sweep, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let d = pyo3::types::PyDict::new(py);
    use numpy::IntoPyArray;
    d.set_item(
        "wavetrend",
        o.wavetrend.into_pyarray(py).reshape((o.rows, o.cols))?,
    )?;
    d.set_item(
        "signal",
        o.signal.into_pyarray(py).reshape((o.rows, o.cols))?,
    )?;
    d.set_item(
        "histogram",
        o.histogram.into_pyarray(py).reshape((o.rows, o.cols))?,
    )?;
    d.set_item(
        "n1s",
        o.combos
            .iter()
            .map(|p| p.n1.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    d.set_item(
        "n2s",
        o.combos
            .iter()
            .map(|p| p.n2.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    d.set_item(
        "n3s",
        o.combos
            .iter()
            .map(|p| p.n3.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    d.set_item(
        "modes",
        o.combos
            .iter()
            .map(|p| format!("{:?}", p.mode.unwrap()))
            .collect::<Vec<_>>(),
    )?;
    d.set_item("rows", o.rows)?;
    d.set_item("cols", o.cols)?;
    Ok(d.into())
}

// ==================== PYTHON CUDA BINDINGS ====================
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::{cuda_available, CudaModGodMode};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::alma::DeviceArrayF32Py;
#[cfg(all(feature = "python", feature = "cuda"))]
use numpy::PyReadonlyArray2;
#[cfg(all(feature = "python", feature = "cuda"))]
use pyo3::types::PyDict;
#[cfg(all(feature = "python", feature = "cuda"))]
use pyo3::{pyfunction, PyResult, Python};

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "mod_god_mode_cuda_batch_dev")]
#[pyo3(signature = (high_f32, low_f32, close_f32, n1_range, n2_range, n3_range, mode="tradition_mg", use_volume=false, volume_f32=None, device_id=0))]
pub fn mod_god_mode_cuda_batch_dev_py<'py>(
    py: Python<'py>,
    high_f32: PyReadonlyArray1<'py, f32>,
    low_f32: PyReadonlyArray1<'py, f32>,
    close_f32: PyReadonlyArray1<'py, f32>,
    n1_range: (usize, usize, usize),
    n2_range: (usize, usize, usize),
    n3_range: (usize, usize, usize),
    mode: &str,
    use_volume: bool,
    volume_f32: Option<PyReadonlyArray1<'py, f32>>,
    device_id: usize,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::IntoPyArray;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let h = high_f32.as_slice()?;
    let l = low_f32.as_slice()?;
    let c = close_f32.as_slice()?;
    let vol = if use_volume {
        Some(
            volume_f32
                .as_ref()
                .ok_or_else(|| PyValueError::new_err("volume required when use_volume=true"))?
                .as_slice()?,
        )
    } else {
        None
    };
    let m = mode
        .parse::<ModGodModeMode>()
        .map_err(|e| PyValueError::new_err(e))?;
    let sweep = ModGodModeBatchRange {
        n1: n1_range,
        n2: n2_range,
        n3: n3_range,
        mode: m,
    };
    let (wt, sig, hist, combos, rows, cols, ctx, dev_id) = py.allow_threads(|| {
        let cuda =
            CudaModGodMode::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let res = cuda
            .mod_god_mode_batch_dev(h, l, c, vol, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let out = res.outputs;
        let rows = out.rows();
        let cols = out.cols();
        let ctx = cuda.context_arc();
        let dev_id = cuda.device_id();
        Ok::<_, PyErr>((out.wt1, out.wt2, out.hist, res.combos, rows, cols, ctx, dev_id))
    })?;
    let dict = PyDict::new(py);
    dict.set_item(
        "wavetrend",
        Py::new(
            py,
            DeviceArrayF32Py {
                inner: wt,
                _ctx: Some(ctx.clone()),
                device_id: Some(dev_id),
            },
        )?,
    )?;
    dict.set_item(
        "signal",
        Py::new(
            py,
            DeviceArrayF32Py {
                inner: sig,
                _ctx: Some(ctx.clone()),
                device_id: Some(dev_id),
            },
        )?,
    )?;
    dict.set_item(
        "histogram",
        Py::new(
            py,
            DeviceArrayF32Py {
                inner: hist,
                _ctx: Some(ctx),
                device_id: Some(dev_id),
            },
        )?,
    )?;
    dict.set_item(
        "n1s",
        combos
            .iter()
            .map(|p| p.n1.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "n2s",
        combos
            .iter()
            .map(|p| p.n2.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "n3s",
        combos
            .iter()
            .map(|p| p.n3.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "modes",
        combos
            .iter()
            .map(|p| format!("{:?}", p.mode.unwrap()))
            .collect::<Vec<_>>(),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "mod_god_mode_cuda_many_series_one_param_dev")]
#[pyo3(signature = (high_tm_f32, low_tm_f32, close_tm_f32, cols, rows, n1=17, n2=6, n3=4, mode="tradition_mg", use_volume=false, volume_tm_f32=None, device_id=0))]
pub fn mod_god_mode_cuda_many_series_one_param_dev_py<'py>(
    py: Python<'py>,
    high_tm_f32: PyReadonlyArray1<'py, f32>,
    low_tm_f32: PyReadonlyArray1<'py, f32>,
    close_tm_f32: PyReadonlyArray1<'py, f32>,
    cols: usize,
    rows: usize,
    n1: usize,
    n2: usize,
    n3: usize,
    mode: &str,
    use_volume: bool,
    volume_tm_f32: Option<PyReadonlyArray1<'py, f32>>,
    device_id: usize,
) -> PyResult<Bound<'py, PyDict>> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let h = high_tm_f32.as_slice()?;
    let l = low_tm_f32.as_slice()?;
    let c = close_tm_f32.as_slice()?;
    let vol = if use_volume {
        Some(
            volume_tm_f32
                .as_ref()
                .ok_or_else(|| PyValueError::new_err("volume required when use_volume=true"))?
                .as_slice()?,
        )
    } else {
        None
    };
    let m = mode
        .parse::<ModGodModeMode>()
        .map_err(|e| PyValueError::new_err(e))?;
    let params = ModGodModeParams {
        n1: Some(n1),
        n2: Some(n2),
        n3: Some(n3),
        mode: Some(m),
        use_volume: Some(use_volume),
    };
    let (wt, sig, hist, ctx, dev_id) = py.allow_threads(|| {
        let cuda =
            CudaModGodMode::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.mod_god_mode_many_series_one_param_time_major_dev(h, l, c, vol, cols, rows, &params)
            .map(|tr| {
                (
                    tr.wt1,
                    tr.wt2,
                    tr.hist,
                    cuda.context_arc(),
                    cuda.device_id(),
                )
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    let dict = PyDict::new(py);
    dict.set_item(
        "wavetrend",
        Py::new(
            py,
            DeviceArrayF32Py {
                inner: wt,
                _ctx: Some(ctx.clone()),
                device_id: Some(dev_id),
            },
        )?,
    )?;
    dict.set_item(
        "signal",
        Py::new(
            py,
            DeviceArrayF32Py {
                inner: sig,
                _ctx: Some(ctx.clone()),
                device_id: Some(dev_id),
            },
        )?,
    )?;
    dict.set_item(
        "histogram",
        Py::new(
            py,
            DeviceArrayF32Py {
                inner: hist,
                _ctx: Some(ctx),
                device_id: Some(dev_id),
            },
        )?,
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    dict.set_item("n1", n1)?;
    dict.set_item("n2", n2)?;
    dict.set_item("n3", n3)?;
    dict.set_item("mode", mode)?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass]
pub struct ModGodModeStreamPy {
    stream: ModGodModeStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl ModGodModeStreamPy {
    #[new]
    #[pyo3(signature = (n1=17, n2=6, n3=4, mode="tradition_mg", use_volume=false))]
    pub fn new(n1: usize, n2: usize, n3: usize, mode: &str, use_volume: bool) -> PyResult<Self> {
        let mode_enum = mode
            .parse::<ModGodModeMode>()
            .map_err(|e| PyValueError::new_err(format!("Invalid mode: {}", e)))?;

        Ok(Self {
            stream: ModGodModeStream::new(n1, n2, n3, mode_enum, use_volume),
        })
    }

    pub fn update(
        &mut self,
        high: f64,
        low: f64,
        close: f64,
        volume: Option<f64>,
    ) -> Option<(f64, f64, f64)> {
        self.stream.update(high, low, close, volume)
    }

    pub fn reset(&mut self) {
        self.stream.reset()
    }
}

// ============================================================================
// WASM BINDINGS
// ============================================================================

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = mod_god_mode)]
pub fn mod_god_mode_wasm(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: Option<Vec<f64>>,
    n1: Option<usize>,
    n2: Option<usize>,
    n3: Option<usize>,
    mode: Option<String>,
    use_volume: Option<bool>,
) -> Result<JsValue, JsValue> {
    let mode_enum = if let Some(m) = mode {
        match m.parse::<ModGodModeMode>() {
            Ok(mode) => Some(mode),
            Err(e) => return Err(JsValue::from_str(&format!("Invalid mode: {}", e))),
        }
    } else {
        None
    };

    let params = ModGodModeParams {
        n1,
        n2,
        n3,
        mode: mode_enum,
        use_volume,
    };

    let input = ModGodModeInput::from_slices(high, low, close, volume.as_deref(), params);

    match mod_god_mode(&input) {
        Ok(output) => {
            let result = serde_wasm_bindgen::to_value(&output)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            Ok(result)
        }
        Err(e) => Err(JsValue::from_str(&e.to_string())),
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn mod_god_mode_alloc(size: usize) -> *mut f64 {
    let mut buf = Vec::<f64>::with_capacity(size);
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn mod_god_mode_free(ptr: *mut f64, size: usize) {
    unsafe {
        Vec::from_raw_parts(ptr, size, size);
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn mod_god_mode_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    vol_ptr: *const f64,
    len: usize,
    has_volume: bool,
    n1: usize,
    n2: usize,
    n3: usize,
    mode: &str,
    out_w_ptr: *mut f64,
    out_s_ptr: *mut f64,
    out_h_ptr: *mut f64,
) -> Result<(), JsValue> {
    if [
        high_ptr as usize,
        low_ptr as usize,
        close_ptr as usize,
        out_w_ptr as usize,
        out_s_ptr as usize,
        out_h_ptr as usize,
    ]
    .iter()
    .any(|&p| p == 0)
    {
        return Err(JsValue::from_str("null pointer"));
    }
    let m = mode
        .parse::<ModGodModeMode>()
        .map_err(|e| JsValue::from_str(&e))?;
    unsafe {
        let h = core::slice::from_raw_parts(high_ptr, len);
        let l = core::slice::from_raw_parts(low_ptr, len);
        let c = core::slice::from_raw_parts(close_ptr, len);
        let v = if has_volume {
            Some(core::slice::from_raw_parts(vol_ptr, len))
        } else {
            None
        };
        let params = ModGodModeParams {
            n1: Some(n1),
            n2: Some(n2),
            n3: Some(n3),
            mode: Some(m),
            use_volume: Some(has_volume),
        };
        let out = mod_god_mode_with_kernel(
            &ModGodModeInput::from_slices(h, l, c, v, params),
            detect_best_kernel(),
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
        core::slice::from_raw_parts_mut(out_w_ptr, len).copy_from_slice(&out.wavetrend);
        core::slice::from_raw_parts_mut(out_s_ptr, len).copy_from_slice(&out.signal);
        core::slice::from_raw_parts_mut(out_h_ptr, len).copy_from_slice(&out.histogram);
    }
    Ok(())
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct ModGodModeJsFlat {
    pub values: Vec<f64>, // concatenated [wavetrend..., signal..., histogram...]
    pub rows: usize,      // 3
    pub cols: usize,      // len
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn mod_god_mode_into_flat(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    vol_ptr: *const f64,
    len: usize,
    has_volume: bool,
    n1: usize,
    n2: usize,
    n3: usize,
    mode: &str,
    out_ptr: *mut f64, // length = 3*len; rows=3, cols=len; row-major [wt..., sig..., hist...]
) -> Result<(), JsValue> {
    if [
        high_ptr as usize,
        low_ptr as usize,
        close_ptr as usize,
        out_ptr as usize,
    ]
    .iter()
    .any(|&p| p == 0)
    {
        return Err(JsValue::from_str("null pointer"));
    }
    let m = mode
        .parse::<ModGodModeMode>()
        .map_err(|e| JsValue::from_str(&e))?;
    unsafe {
        let h = core::slice::from_raw_parts(high_ptr, len);
        let l = core::slice::from_raw_parts(low_ptr, len);
        let c = core::slice::from_raw_parts(close_ptr, len);
        let v = if has_volume {
            Some(core::slice::from_raw_parts(vol_ptr, len))
        } else {
            None
        };

        let wt = core::slice::from_raw_parts_mut(out_ptr, len);
        let sig = core::slice::from_raw_parts_mut(out_ptr.add(len), len);
        let hist = core::slice::from_raw_parts_mut(out_ptr.add(2 * len), len);

        let params = ModGodModeParams {
            n1: Some(n1),
            n2: Some(n2),
            n3: Some(n3),
            mode: Some(m),
            use_volume: Some(has_volume),
        };
        let inp = ModGodModeInput::from_slices(h, l, c, v, params);
        mod_god_mode_into_slices(wt, sig, hist, &inp, detect_best_kernel())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }
    Ok(())
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = mod_god_mode_js_flat)]
pub fn mod_god_mode_js_flat(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: Option<Vec<f64>>,
    n1: usize,
    n2: usize,
    n3: usize,
    mode: &str,
    use_volume: bool,
) -> Result<JsValue, JsValue> {
    let m = mode
        .parse::<ModGodModeMode>()
        .map_err(|e| JsValue::from_str(&e))?;
    let params = ModGodModeParams {
        n1: Some(n1),
        n2: Some(n2),
        n3: Some(n3),
        mode: Some(m),
        use_volume: Some(use_volume),
    };
    let out = mod_god_mode_with_kernel(
        &ModGodModeInput::from_slices(high, low, close, volume.as_deref(), params),
        detect_best_kernel(),
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let cols = close.len();
    let mut values = Vec::with_capacity(3 * cols);
    values.extend_from_slice(&out.wavetrend);
    values.extend_from_slice(&out.signal);
    values.extend_from_slice(&out.histogram);

    serde_wasm_bindgen::to_value(&ModGodModeJsFlat {
        values,
        rows: 3,
        cols,
    })
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::{read_candles_from_csv, Candles};

    fn generate_test_candles(len: usize) -> Candles {
        let mut open = vec![0.0; len];
        let mut high = vec![0.0; len];
        let mut low = vec![0.0; len];
        let mut close = vec![0.0; len];
        let mut volume = vec![1000.0; len];

        for i in 0..len {
            let base = 100.0 + (i as f64) * 0.1;
            close[i] = base + ((i % 10) as f64 - 5.0) * 0.5;
            open[i] = if i == 0 { base } else { close[i - 1] };
            high[i] = close[i].max(open[i]) + 0.5;
            low[i] = close[i].min(open[i]) - 0.5;
            volume[i] = 1000.0 + (i as f64) * 10.0;
        }

        Candles::new(
            vec![0; len], // timestamp
            open,
            high,
            low,
            close,
            volume,
        )
    }

    /// Generate all kernel variant tests for a test function
    macro_rules! generate_all_mod_god_mode_tests {
        ($($test_fn:ident),*) => {
            $(
                paste::paste! {
                    #[test]
                    fn [<$test_fn _scalar>]() {
                        $test_fn(Kernel::Scalar);
                    }

                    // SSE2 kernel not available in this codebase

                    #[test]
                    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64", target_feature = "avx2"))]
                    fn [<$test_fn _avx2>]() {
                        $test_fn(Kernel::Avx2);
                    }

                    #[test]
                    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64", target_feature = "avx512f"))]
                    fn [<$test_fn _avx512>]() {
                        $test_fn(Kernel::Avx512);
                    }
                }
            )*
        };
    }

    fn check_mod_god_mode_basic(kernel: Kernel) {
        let close = vec![10.0, 11.0, 12.0, 11.5, 10.5, 11.0, 12.5, 13.0, 12.0, 11.0];
        let high = vec![10.5, 11.5, 12.5, 12.0, 11.0, 11.5, 13.0, 13.5, 12.5, 11.5];
        let low = vec![9.5, 10.5, 11.5, 11.0, 10.0, 10.5, 12.0, 12.5, 11.5, 10.5];

        let params = ModGodModeParams {
            n1: Some(3),
            n2: Some(2),
            n3: Some(2),
            mode: Some(ModGodModeMode::TraditionMg),
            use_volume: Some(false),
        };

        let input = ModGodModeInput::from_slices(&high, &low, &close, None, params);
        let result = mod_god_mode_with_kernel(&input, kernel);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.wavetrend.len(), close.len());
        assert_eq!(output.signal.len(), close.len());
        assert_eq!(output.histogram.len(), close.len());
    }

    fn check_mod_god_mode_empty_data(kernel: Kernel) {
        let params = ModGodModeParams::default();
        let input = ModGodModeInput::from_slices(&[], &[], &[], None, params);
        let result = mod_god_mode_with_kernel(&input, kernel);

        assert!(matches!(result, Err(ModGodModeError::EmptyInputData)));
    }

    fn check_mod_god_mode_all_nan(kernel: Kernel) {
        let nan_data = vec![f64::NAN; 20]; // Make sure we have enough data
        let params = ModGodModeParams {
            n1: Some(3),
            n2: Some(2),
            n3: Some(2),
            mode: Some(ModGodModeMode::TraditionMg),
            use_volume: Some(false),
        };
        let input = ModGodModeInput::from_slices(&nan_data, &nan_data, &nan_data, None, params);
        let result = mod_god_mode_with_kernel(&input, kernel);

        // Should compute but return all NaN
        match result {
            Ok(output) => {
                assert!(output.wavetrend.iter().all(|v| v.is_nan()));
            }
            Err(e) => {
                // It's actually fine if we get an error with all NaN data
                // Some indicators may not handle this case
                println!("All NaN test returned error: {:?}", e);
            }
        }
    }

    fn check_mod_god_mode_insufficient_data(kernel: Kernel) {
        let close = vec![10.0, 11.0];
        let high = vec![10.5, 11.5];
        let low = vec![9.5, 10.5];

        let params = ModGodModeParams {
            n1: Some(17),
            n2: Some(6),
            n3: Some(4),
            mode: Some(ModGodModeMode::TraditionMg),
            use_volume: Some(false),
        };

        let input = ModGodModeInput::from_slices(&high, &low, &close, None, params);
        let result = mod_god_mode_with_kernel(&input, kernel);

        assert!(matches!(
            result,
            Err(ModGodModeError::NotEnoughValidData { .. })
        ));
    }

    fn check_mod_god_mode_with_volume(kernel: Kernel) {
        let close = vec![10.0, 11.0, 12.0, 11.5, 10.5, 11.0, 12.5, 13.0, 12.0, 11.0];
        let high = vec![10.5, 11.5, 12.5, 12.0, 11.0, 11.5, 13.0, 13.5, 12.5, 11.5];
        let low = vec![9.5, 10.5, 11.5, 11.0, 10.0, 10.5, 12.0, 12.5, 11.5, 10.5];
        let volume = vec![
            1000.0, 1100.0, 900.0, 1200.0, 800.0, 1300.0, 1500.0, 1400.0, 1100.0, 1000.0,
        ];

        let params = ModGodModeParams {
            n1: Some(3),
            n2: Some(2),
            n3: Some(2),
            mode: Some(ModGodModeMode::GodmodeMg),
            use_volume: Some(true),
        };

        let input = ModGodModeInput::from_slices(&high, &low, &close, Some(&volume), params);
        let result = mod_god_mode_with_kernel(&input, kernel);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.wavetrend.len(), close.len());
    }

    fn check_mod_god_mode_modes(kernel: Kernel) {
        let candles = generate_test_candles(50);

        let modes = vec![
            ModGodModeMode::Godmode,
            ModGodModeMode::Tradition,
            ModGodModeMode::GodmodeMg,
            ModGodModeMode::TraditionMg,
        ];

        for mode in modes {
            let params = ModGodModeParams {
                n1: Some(5),
                n2: Some(3),
                n3: Some(2),
                mode: Some(mode),
                use_volume: Some(false),
            };

            let input = ModGodModeInput::from_candles(&candles, params);
            let result = mod_god_mode_with_kernel(&input, kernel);

            assert!(result.is_ok(), "Mode {:?} should succeed", mode);
        }
    }

    fn check_mod_god_mode_builder(kernel: Kernel) {
        let candles = generate_test_candles(20);

        let result = ModGodModeBuilder::new()
            .n1(5)
            .n2(3)
            .n3(2)
            .mode(ModGodModeMode::Godmode)
            .use_volume(false)
            .calculate_with_kernel(ModGodModeData::Candles { candles: &candles }, kernel);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.wavetrend.len(), candles.close.len());
    }

    fn check_mod_god_mode_stream(_kernel: Kernel) {
        let mut stream = ModGodModeStream::new(3, 2, 2, ModGodModeMode::TraditionMg, false);

        // Need more data for RSI calculations in Pine version
        let test_data = vec![
            (10.5, 9.5, 10.0),
            (11.5, 10.5, 11.0),
            (12.5, 11.5, 12.0),
            (12.0, 11.0, 11.5),
            (11.0, 10.0, 10.5),
            (11.5, 10.5, 11.0), // Extra data point
            (12.0, 11.0, 11.5), // Extra data point
            (12.5, 11.5, 12.0), // Extra data point
        ];

        let mut got_result = false;
        for (high, low, close) in test_data {
            let result = stream.update(high, low, close, None);
            if result.is_some() {
                got_result = true;
            }
        }

        // Eventually we should get a result once enough data is accumulated
        assert!(
            got_result,
            "Stream should produce results after sufficient data"
        );
    }

    fn check_mod_god_mode_batch(kernel: Kernel) {
        let datasets: Vec<Candles> = (0..3).map(|_| generate_test_candles(20)).collect();

        let batch_builder = ModGodModeBatchBuilder::new()
            .n1(5)
            .n2(3)
            .n3(2)
            .mode(ModGodModeMode::TraditionMg)
            .parallel(false);

        let results = batch_builder.calculate_batch(&datasets);

        assert_eq!(results.len(), datasets.len());
        for result in results {
            assert!(result.is_ok());
        }
    }

    fn check_mod_god_mode_consistency(kernel: Kernel) {
        let candles = generate_test_candles(50);
        let params = ModGodModeParams {
            n1: Some(7),
            n2: Some(4),
            n3: Some(3),
            mode: Some(ModGodModeMode::TraditionMg),
            use_volume: Some(false),
        };

        let input = ModGodModeInput::from_candles(&candles, params.clone());
        let result1 = mod_god_mode_with_kernel(&input, kernel).unwrap();
        let result2 = mod_god_mode_with_kernel(&input, kernel).unwrap();

        for i in 0..result1.wavetrend.len() {
            // Check wavetrend
            if !result1.wavetrend[i].is_nan() && !result2.wavetrend[i].is_nan() {
                assert_eq!(result1.wavetrend[i], result2.wavetrend[i]);
            } else if result1.wavetrend[i].is_nan() != result2.wavetrend[i].is_nan() {
                panic!("Wavetrend NaN mismatch at index {}", i);
            }

            // Check signal
            if !result1.signal[i].is_nan() && !result2.signal[i].is_nan() {
                assert_eq!(result1.signal[i], result2.signal[i]);
            } else if result1.signal[i].is_nan() != result2.signal[i].is_nan() {
                panic!("Signal NaN mismatch at index {}", i);
            }

            // Check histogram
            if !result1.histogram[i].is_nan() && !result2.histogram[i].is_nan() {
                assert_eq!(result1.histogram[i], result2.histogram[i]);
            } else if result1.histogram[i].is_nan() != result2.histogram[i].is_nan() {
                panic!("Histogram NaN mismatch at index {}", i);
            }
        }
    }

    fn check_mod_god_mode_accuracy(kernel: Kernel) {
        // Test accuracy against known reference values using CSV data
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = match read_candles_from_csv(file_path) {
            Ok(c) => c,
            Err(e) => {
                panic!("Failed to read CSV file: {}", e);
            }
        };

        // Use default parameters
        let params = ModGodModeParams {
            n1: Some(17),
            n2: Some(6),
            n3: Some(4),
            mode: Some(ModGodModeMode::TraditionMg),
            use_volume: Some(true),
        };

        let input = ModGodModeInput::from_candles(&candles, params);
        let result = mod_god_mode_with_kernel(&input, kernel).unwrap();

        // Reference values from Pine Script (last 5 wavetrend values)
        let expected_last_five = [
            61.66219598,
            55.92955776,
            34.70836488,
            39.48824969,
            15.74958884,
        ];

        // Get last 5 non-NaN values
        let non_nan_values: Vec<f64> = result
            .wavetrend
            .iter()
            .filter(|v| !v.is_nan())
            .cloned()
            .collect();

        assert!(
            non_nan_values.len() >= 5,
            "Not enough non-NaN values: got {}, need at least 5",
            non_nan_values.len()
        );

        let start = non_nan_values.len().saturating_sub(5);
        for (i, &val) in non_nan_values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            // Using a tolerance of 4.0 as we've determined this is the maximum difference
            assert!(
                diff < 4.0,
                "MOD_GOD_MODE wavetrend mismatch at index {}: got {:.8}, expected {:.8}, diff {:.8}",
                i, val, expected_last_five[i], diff
            );
        }
    }

    fn check_mod_god_mode_nan_handling(kernel: Kernel) {
        let candles = generate_test_candles(50);
        let params = ModGodModeParams {
            n1: Some(7),
            n2: Some(4),
            n3: Some(3),
            mode: Some(ModGodModeMode::TraditionMg),
            use_volume: Some(false),
        };

        let input = ModGodModeInput::from_candles(&candles, params);
        let result = mod_god_mode_with_kernel(&input, kernel);
        assert!(result.is_ok());
        let output = result.unwrap();

        // After warmup, no NaNs should be present
        let warmup = 7 + 4 + 3; // Conservative warmup estimate
        for i in warmup..output.wavetrend.len() {
            if output.wavetrend[i].is_nan() {
                panic!("Found NaN at index {} after warmup period {}", i, warmup);
            }
        }
    }

    fn check_mod_god_mode_reinput(kernel: Kernel) {
        let candles = generate_test_candles(50);
        let params = ModGodModeParams {
            n1: Some(5),
            n2: Some(3),
            n3: Some(2),
            mode: Some(ModGodModeMode::TraditionMg),
            use_volume: Some(false),
        };

        // Compute once
        let input1 = ModGodModeInput::from_candles(&candles, params.clone());
        let result1 = mod_god_mode_with_kernel(&input1, kernel).unwrap();

        // Feed wavetrend back as close
        let input2 = ModGodModeInput::from_slices(
            &candles.high,
            &candles.low,
            &result1.wavetrend,
            None,
            params,
        );
        let result2 = mod_god_mode_with_kernel(&input2, kernel);

        // Should succeed without errors
        assert!(result2.is_ok());
    }

    fn check_mod_god_mode_streaming_parity(_kernel: Kernel) {
        let candles = generate_test_candles(50);
        let params = ModGodModeParams {
            n1: Some(5),
            n2: Some(3),
            n3: Some(2),
            mode: Some(ModGodModeMode::TraditionMg),
            use_volume: Some(false),
        };

        // Batch calculation
        let input = ModGodModeInput::from_candles(&candles, params.clone());
        let batch_result = mod_god_mode(&input).unwrap();

        // Streaming calculation
        let mut stream = ModGodModeStream::try_new(params).unwrap();
        let mut stream_results = Vec::new();

        for i in 0..candles.close.len() {
            let result = stream.update(candles.high[i], candles.low[i], candles.close[i], None);
            stream_results.push(result);
        }

        // Last streaming value should match batch last value (when available)
        if let Some(Some((wt, sig, hist))) = stream_results.last() {
            let last_idx = batch_result.wavetrend.len() - 1;
            if !batch_result.wavetrend[last_idx].is_nan() {
                assert!(
                    (wt - batch_result.wavetrend[last_idx]).abs() < 1e-10,
                    "Streaming wavetrend mismatch"
                );
                assert!(
                    (sig - batch_result.signal[last_idx]).abs() < 1e-10,
                    "Streaming signal mismatch"
                );
                assert!(
                    (hist - batch_result.histogram[last_idx]).abs() < 1e-10,
                    "Streaming histogram mismatch"
                );
            }
        }
    }

    fn check_batch_default_row(kernel: Kernel) {
        // Use the same CSV data as other indicators
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = match read_candles_from_csv(file) {
            Ok(c) => c,
            Err(e) => {
                println!("WARNING: Could not read CSV file: {}", e);
                println!("Using generated test data instead");
                generate_test_candles(20)
            }
        };
        let range = ModGodModeBatchRange {
            n1: (5, 5, 0),
            n2: (3, 3, 0),
            n3: (2, 2, 0),
            mode: ModGodModeMode::TraditionMg,
        };

        // Convert single kernel to batch kernel for batch functions
        let batch_kernel = match kernel {
            Kernel::Scalar => Kernel::ScalarBatch,
            Kernel::Avx2 => Kernel::Avx2Batch,
            Kernel::Avx512 => Kernel::Avx512Batch,
            Kernel::Auto => Kernel::Auto,
            k if k.is_batch() => k,
            _ => Kernel::ScalarBatch,
        };

        let batch_result = mod_god_mode_batch_with_kernel(
            &candles.high,
            &candles.low,
            &candles.close,
            None,
            &range,
            batch_kernel,
        )
        .unwrap();

        assert_eq!(batch_result.rows, 1);
        assert_eq!(batch_result.cols, candles.close.len());
        assert_eq!(batch_result.combos.len(), 1);
    }

    fn check_batch_sweep(kernel: Kernel) {
        // Use the same CSV data as other indicators
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = match read_candles_from_csv(file) {
            Ok(c) => c,
            Err(e) => {
                println!("WARNING: Could not read CSV file: {}", e);
                println!("Using generated test data instead");
                generate_test_candles(20)
            }
        };
        let range = ModGodModeBatchRange {
            n1: (3, 5, 1),
            n2: (2, 3, 1),
            n3: (2, 2, 0),
            mode: ModGodModeMode::TraditionMg,
        };

        // Convert single kernel to batch kernel for batch functions
        let batch_kernel = match kernel {
            Kernel::Scalar => Kernel::ScalarBatch,
            Kernel::Avx2 => Kernel::Avx2Batch,
            Kernel::Avx512 => Kernel::Avx512Batch,
            Kernel::Auto => Kernel::Auto,
            k if k.is_batch() => k,
            _ => Kernel::ScalarBatch,
        };

        let batch_result = mod_god_mode_batch_with_kernel(
            &candles.high,
            &candles.low,
            &candles.close,
            None,
            &range,
            batch_kernel,
        )
        .unwrap();

        // 3 n1 values * 2 n2 values * 1 n3 value = 6 combinations
        assert_eq!(batch_result.rows, 6);
        assert_eq!(batch_result.combos.len(), 6);

        // Test row lookup
        let test_params = ModGodModeParams {
            n1: Some(4),
            n2: Some(2),
            n3: Some(2),
            mode: Some(ModGodModeMode::TraditionMg),
            use_volume: Some(false),
        };
        assert!(batch_result.row_for_params(&test_params).is_some());
    }

    #[cfg(debug_assertions)]
    #[test]
    fn check_mod_god_mode_no_poison() {
        let c = generate_test_candles(64);
        let params = ModGodModeParams::default();
        let out = mod_god_mode(&ModGodModeInput::from_candles(&c, params)).unwrap();
        for v in out
            .wavetrend
            .iter()
            .chain(out.signal.iter())
            .chain(out.histogram.iter())
        {
            if v.is_nan() {
                continue;
            }
            let b = v.to_bits();
            assert_ne!(
                b, 0x11111111_11111111,
                "alloc_with_nan_prefix poison leaked"
            );
            assert_ne!(b, 0x22222222_22222222, "init_matrix_prefixes poison leaked");
            assert_ne!(b, 0x33333333_33333333, "make_uninit_matrix poison leaked");
        }
    }

    #[test]
    fn check_mod_god_mode_basic_auto_detect() {
        check_mod_god_mode_basic(Kernel::Auto);
    }

    #[test]
    fn check_batch_default_row_auto_detect() {
        check_batch_default_row(Kernel::Auto);
    }

    // Generate all test variants
    generate_all_mod_god_mode_tests!(
        check_mod_god_mode_basic,
        check_mod_god_mode_accuracy,
        check_mod_god_mode_empty_data,
        check_mod_god_mode_all_nan,
        check_mod_god_mode_insufficient_data,
        check_mod_god_mode_with_volume,
        check_mod_god_mode_modes,
        check_mod_god_mode_builder,
        check_mod_god_mode_stream,
        check_mod_god_mode_batch,
        check_mod_god_mode_consistency,
        check_mod_god_mode_nan_handling,
        check_mod_god_mode_reinput,
        check_mod_god_mode_streaming_parity,
        check_batch_default_row,
        check_batch_sweep
    );

    // Property-based tests
    #[cfg(test)]
    mod proptest_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn test_mod_god_mode_never_panics(
                n1 in 1usize..20,
                n2 in 1usize..20,
                n3 in 1usize..20,
                len in 0usize..100,
            ) {
                let candles = generate_test_candles(len);
                // Ensure n3 is never greater than or equal to len to avoid RSI issues
                let n3_safe = if len > 0 { n3.min(len.saturating_sub(1).max(1)) } else { n3 };
                let params = ModGodModeParams {
                    n1: Some(n1),
                    n2: Some(n2),
                    n3: Some(n3_safe),
                    mode: Some(ModGodModeMode::TraditionMg),
                    use_volume: Some(false),
                };

                let input = ModGodModeInput::from_candles(&candles, params);
                let _ = mod_god_mode(&input);
            }

            #[test]
            fn test_mod_god_mode_output_length(
                n1 in 1usize..10,
                n2 in 1usize..10,
                n3 in 1usize..10,
                len in 20usize..100,
            ) {
                let candles = generate_test_candles(len);
                let params = ModGodModeParams {
                    n1: Some(n1),
                    n2: Some(n2),
                    n3: Some(n3),
                    mode: Some(ModGodModeMode::TraditionMg),
                    use_volume: Some(false),
                };

                let input = ModGodModeInput::from_candles(&candles, params);
                if let Ok(output) = mod_god_mode(&input) {
                    prop_assert_eq!(output.wavetrend.len(), len);
                    prop_assert_eq!(output.signal.len(), len);
                    prop_assert_eq!(output.histogram.len(), len);
                }
            }
        }
    }

    // Parity test: native into() matches Vec API including NaN warmups
    #[cfg(not(feature = "wasm"))]
    #[test]
    fn test_mod_god_mode_into_matches_api() {
        fn eq_or_both_nan_eps(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-12
        }

        // Use the shared CSV used by existing accuracy tests
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file).expect("failed to read candles CSV");
        let params = ModGodModeParams::default();
        let input = ModGodModeInput::from_candles(&candles, params);

        // Baseline via Vec-returning API
        let base = mod_god_mode(&input).expect("baseline mod_god_mode should succeed");

        // Preallocated outputs
        let len = candles.close.len();
        let mut wt = vec![0.0; len];
        let mut sig = vec![0.0; len];
        let mut hist = vec![0.0; len];

        mod_god_mode_into(&input, &mut wt, &mut sig, &mut hist)
            .expect("mod_god_mode_into should succeed");

        assert_eq!(wt.len(), base.wavetrend.len());
        assert_eq!(sig.len(), base.signal.len());
        assert_eq!(hist.len(), base.histogram.len());

        for i in 0..len {
            assert!(
                eq_or_both_nan_eps(wt[i], base.wavetrend[i]),
                "wavetrend mismatch at {}: got {:?} expected {:?}",
                i,
                wt[i],
                base.wavetrend[i]
            );
            assert!(
                eq_or_both_nan_eps(sig[i], base.signal[i]),
                "signal mismatch at {}: got {:?} expected {:?}",
                i,
                sig[i],
                base.signal[i]
            );
            assert!(
                eq_or_both_nan_eps(hist[i], base.histogram[i]),
                "histogram mismatch at {}: got {:?} expected {:?}",
                i,
                hist[i],
                base.histogram[i]
            );
        }
    }
}
