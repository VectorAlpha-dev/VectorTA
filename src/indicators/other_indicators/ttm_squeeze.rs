//! # Beardy Squeeze Pro
//!
//! TTM Squeeze momentum oscillator with multi-level squeeze detection.
//! Combines Bollinger Bands and Keltner Channels to identify squeeze conditions.
//!
//! ## Parameters
//! - **length**: Period for calculations (default: 20)
//! - **bb_mult**: Bollinger Band standard deviation multiplier (default: 2.0)
//! - **kc_mult_high**: Keltner Channel multiplier #1 (default: 1.0)
//! - **kc_mult_mid**: Keltner Channel multiplier #2 (default: 1.5)
//! - **kc_mult_low**: Keltner Channel multiplier #3 (default: 2.0)
//!
//! ## Errors
//! - **EmptyData**: Input data is empty
//! - **InvalidLength**: Period is zero or exceeds data length
//! - **NotEnoughData**: Insufficient data for calculation
//! - **AllValuesNaN**: All input values are NaN
//!
//! ## Returns
//! - **momentum**: Momentum oscillator values
//! - **squeeze**: Squeeze state (0=NoSqz, 1=LowSqz, 2=MidSqz, 3=HighSqz)

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyUntypedArrayMethods};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

use crate::indicators::moving_averages::sma::{sma_with_kernel, SmaInput, SmaParams};
use crate::utilities::data_loader::Candles;
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{alloc_with_nan_prefix, detect_best_kernel};
use std::error::Error;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum BeardySqueezeProData<'a> {
    Candles { candles: &'a Candles },
    Slices { high: &'a [f64], low: &'a [f64], close: &'a [f64] },
}

#[derive(Debug, Clone)]
pub struct BeardySqueezeProOutput {
    pub momentum: Vec<f64>,
    pub squeeze: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct BeardySqueezeProParams {
    pub length: Option<usize>,
    pub bb_mult: Option<f64>,
    pub kc_mult_high: Option<f64>,
    pub kc_mult_mid: Option<f64>,
    pub kc_mult_low: Option<f64>,
}

impl Default for BeardySqueezeProParams {
    fn default() -> Self {
        Self {
            length: Some(20),
            bb_mult: Some(2.0),
            kc_mult_high: Some(1.0),
            kc_mult_mid: Some(1.5),
            kc_mult_low: Some(2.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BeardySqueezeProInput<'a> {
    pub data: BeardySqueezeProData<'a>,
    pub params: BeardySqueezeProParams,
}

impl<'a> BeardySqueezeProInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: BeardySqueezeProParams) -> Self {
        Self {
            data: BeardySqueezeProData::Candles { candles },
            params,
        }
    }
    
    #[inline]
    pub fn from_slices(high: &'a [f64], low: &'a [f64], close: &'a [f64], params: BeardySqueezeProParams) -> Self {
        Self {
            data: BeardySqueezeProData::Slices { high, low, close },
            params,
        }
    }
    
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, BeardySqueezeProParams::default())
    }
    
    #[inline]
    pub fn get_length(&self) -> usize {
        self.params.length.unwrap_or(20)
    }
}

// Builder pattern for ergonomic API
#[derive(Debug, Clone)]
pub struct BeardySqueezeProBuilder {
    length: Option<usize>,
    bb_mult: Option<f64>,
    kc_mult_high: Option<f64>,
    kc_mult_mid: Option<f64>,
    kc_mult_low: Option<f64>,
    kernel: Kernel,
}

impl Default for BeardySqueezeProBuilder {
    fn default() -> Self {
        Self {
            length: None,
            bb_mult: None,
            kc_mult_high: None,
            kc_mult_mid: None,
            kc_mult_low: None,
            kernel: Kernel::Auto,
        }
    }
}

impl BeardySqueezeProBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }
    
    #[inline]
    pub fn length(mut self, length: usize) -> Self {
        self.length = Some(length);
        self
    }
    
    #[inline]
    pub fn bb_mult(mut self, mult: f64) -> Self {
        self.bb_mult = Some(mult);
        self
    }
    
    #[inline]
    pub fn kc_mult_high(mut self, mult: f64) -> Self {
        self.kc_mult_high = Some(mult);
        self
    }
    
    #[inline]
    pub fn kc_mult_mid(mut self, mult: f64) -> Self {
        self.kc_mult_mid = Some(mult);
        self
    }
    
    #[inline]
    pub fn kc_mult_low(mut self, mult: f64) -> Self {
        self.kc_mult_low = Some(mult);
        self
    }
    
    #[inline]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }
    
    #[inline]
    pub fn build_params(self) -> BeardySqueezeProParams {
        BeardySqueezeProParams {
            length: self.length,
            bb_mult: self.bb_mult,
            kc_mult_high: self.kc_mult_high,
            kc_mult_mid: self.kc_mult_mid,
            kc_mult_low: self.kc_mult_low,
        }
    }
    
    #[inline]
    pub fn build<'a>(self, candles: &'a Candles) -> Result<BeardySqueezeProOutput, BeardySqueezeProError> {
        let kernel = self.kernel;
        let params = self.build_params();
        let input = BeardySqueezeProInput::from_candles(candles, params);
        beardy_squeeze_pro_with_kernel(&input, kernel)
    }
    
    #[inline]
    pub fn build_slices<'a>(
        self,
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    ) -> Result<BeardySqueezeProOutput, BeardySqueezeProError> {
        let kernel = self.kernel;
        let params = self.build_params();
        let input = BeardySqueezeProInput::from_slices(high, low, close, params);
        beardy_squeeze_pro_with_kernel(&input, kernel)
    }
}

#[derive(Debug, Error)]
pub enum BeardySqueezeProError {
    #[error("BeardySqueezePro: Empty data")]
    EmptyData,
    #[error("BeardySqueezePro: Invalid length {length}")]
    InvalidLength { length: usize },
    #[error("BeardySqueezePro: Not enough data - need {needed}, have {have}")]
    NotEnoughData { needed: usize, have: usize },
    #[error("BeardySqueezePro: All values are NaN")]
    AllValuesNaN,
    #[error("BeardySqueezePro: Inconsistent slice lengths - high={high}, low={low}, close={close}")]
    InconsistentSliceLengths { high: usize, low: usize, close: usize },
    #[error("BeardySqueezePro: SMA error: {0}")]
    SmaError(String),
    #[error("BeardySqueezePro: LinReg error: {0}")]
    LinRegError(String),
}

/// Calculate standard deviation
#[inline]
fn std_dev(data: &[f64], mean: f64, start: usize, end: usize) -> f64 {
    let mut sum_sq = 0.0;
    let mut count = 0;
    
    for i in start..=end {
        if !data[i].is_nan() {
            let diff = data[i] - mean;
            sum_sq += diff * diff;
            count += 1;
        }
    }
    
    if count > 1 {
        (sum_sq / count as f64).sqrt()
    } else {
        f64::NAN
    }
}

/// Calculate true range
#[inline]
fn true_range(high: f64, low: f64, prev_close: Option<f64>) -> f64 {
    match prev_close {
        Some(pc) => {
            let hl = high - low;
            let hc = (high - pc).abs();
            let lc = (low - pc).abs();
            hl.max(hc).max(lc)
        }
        None => high - low,
    }
}

#[inline]
pub fn beardy_squeeze_pro(input: &BeardySqueezeProInput) -> Result<BeardySqueezeProOutput, BeardySqueezeProError> {
    beardy_squeeze_pro_with_kernel(input, Kernel::Auto)
}

pub fn beardy_squeeze_pro_with_kernel(
    input: &BeardySqueezeProInput,
    kernel: Kernel,
) -> Result<BeardySqueezeProOutput, BeardySqueezeProError> {
    // Extract data
    let (high, low, close) = match &input.data {
        BeardySqueezeProData::Candles { candles } => {
            if candles.close.is_empty() {
                return Err(BeardySqueezeProError::EmptyData);
            }
            (&candles.high[..], &candles.low[..], &candles.close[..])
        }
        BeardySqueezeProData::Slices { high, low, close } => {
            if high.len() != low.len() || low.len() != close.len() {
                return Err(BeardySqueezeProError::InconsistentSliceLengths {
                    high: high.len(),
                    low: low.len(),
                    close: close.len(),
                });
            }
            if close.is_empty() {
                return Err(BeardySqueezeProError::EmptyData);
            }
            (*high, *low, *close)
        }
    };
    
    let len = close.len();
    let length = input.get_length();
    let bb_mult = input.params.bb_mult.unwrap_or(2.0);
    let kc_mult_high = input.params.kc_mult_high.unwrap_or(1.0);
    let kc_mult_mid = input.params.kc_mult_mid.unwrap_or(1.5);
    let kc_mult_low = input.params.kc_mult_low.unwrap_or(2.0);
    
    if length == 0 || length > len {
        return Err(BeardySqueezeProError::InvalidLength { length });
    }
    
    // Find first valid index
    let first = close.iter().position(|&x| !x.is_nan()).ok_or(BeardySqueezeProError::AllValuesNaN)?;
    if len - first < length {
        return Err(BeardySqueezeProError::NotEnoughData {
            needed: length,
            have: len - first,
        });
    }
    
    let warmup = first + length - 1;
    
    // Calculate SMA for BB and KC basis (same value)
    let sma_params = SmaParams { period: Some(length) };
    let sma_input = SmaInput::from_slice(close, sma_params);
    let sma_result = sma_with_kernel(&sma_input, kernel)
        .map_err(|e| BeardySqueezeProError::SmaError(e.to_string()))?;
    let sma_values = sma_result.values;
    
    // Calculate True Range SMA for Keltner Channels
    let mut tr_values = vec![f64::NAN; len];
    for i in first..len {
        let tr = if i == first {
            high[i] - low[i]
        } else {
            true_range(high[i], low[i], Some(close[i - 1]))
        };
        tr_values[i] = tr;
    }
    
    let tr_sma_params = SmaParams { period: Some(length) };
    let tr_sma_input = SmaInput::from_slice(&tr_values, tr_sma_params);
    let tr_sma_result = sma_with_kernel(&tr_sma_input, kernel)
        .map_err(|e| BeardySqueezeProError::SmaError(e.to_string()))?;
    let dev_kc = tr_sma_result.values;
    
    // Calculate standard deviation for Bollinger Bands
    let mut bb_upper = vec![f64::NAN; len];
    let mut bb_lower = vec![f64::NAN; len];
    
    for i in warmup..len {
        if !sma_values[i].is_nan() {
            let std = std_dev(close, sma_values[i], i.saturating_sub(length - 1), i);
            let dev = bb_mult * std;
            bb_upper[i] = sma_values[i] + dev;
            bb_lower[i] = sma_values[i] - dev;
        }
    }
    
    // Calculate Keltner Channels
    let mut kc_upper_high = vec![f64::NAN; len];
    let mut kc_lower_high = vec![f64::NAN; len];
    let mut kc_upper_mid = vec![f64::NAN; len];
    let mut kc_lower_mid = vec![f64::NAN; len];
    let mut kc_upper_low = vec![f64::NAN; len];
    let mut kc_lower_low = vec![f64::NAN; len];
    
    for i in warmup..len {
        if !sma_values[i].is_nan() && !dev_kc[i].is_nan() {
            kc_upper_high[i] = sma_values[i] + dev_kc[i] * kc_mult_high;
            kc_lower_high[i] = sma_values[i] - dev_kc[i] * kc_mult_high;
            kc_upper_mid[i] = sma_values[i] + dev_kc[i] * kc_mult_mid;
            kc_lower_mid[i] = sma_values[i] - dev_kc[i] * kc_mult_mid;
            kc_upper_low[i] = sma_values[i] + dev_kc[i] * kc_mult_low;
            kc_lower_low[i] = sma_values[i] - dev_kc[i] * kc_mult_low;
        }
    }
    
    // Calculate squeeze states
    let mut squeeze = alloc_with_nan_prefix(len, warmup);
    
    for i in warmup..len {
        if !bb_upper[i].is_nan() && !bb_lower[i].is_nan() &&
           !kc_upper_low[i].is_nan() && !kc_lower_low[i].is_nan() {
            // Check squeeze conditions
            let no_sqz = bb_lower[i] < kc_lower_low[i] || bb_upper[i] > kc_upper_low[i];
            
            if no_sqz {
                squeeze[i] = 0.0; // NoSqz (Green)
            } else {
                // Check for tighter squeeze levels
                let high_sqz = bb_lower[i] >= kc_lower_high[i] || bb_upper[i] <= kc_upper_high[i];
                let mid_sqz = bb_lower[i] >= kc_lower_mid[i] || bb_upper[i] <= kc_upper_mid[i];
                
                if high_sqz {
                    squeeze[i] = 3.0; // HighSqz (Orange)
                } else if mid_sqz {
                    squeeze[i] = 2.0; // MidSqz (Red)
                } else {
                    squeeze[i] = 1.0; // LowSqz (Black)
                }
            }
        }
    }
    
    // Calculate momentum oscillator
    // mom = linreg(close - avg(avg(highest(high), lowest(low)), sma), length, 0)
    
    // First, calculate highest high and lowest low over period
    let mut mom_input = vec![f64::NAN; len];
    
    for i in warmup..len {
        // Find highest high and lowest low in window
        let start = i.saturating_sub(length - 1);
        let mut highest = f64::NEG_INFINITY;
        let mut lowest = f64::INFINITY;
        
        for j in start..=i {
            if !high[j].is_nan() && high[j] > highest {
                highest = high[j];
            }
            if !low[j].is_nan() && low[j] < lowest {
                lowest = low[j];
            }
        }
        
        if highest != f64::NEG_INFINITY && lowest != f64::INFINITY && !sma_values[i].is_nan() {
            let midpoint = (highest + lowest) / 2.0;
            let avg = (midpoint + sma_values[i]) / 2.0;
            mom_input[i] = close[i] - avg;
        }
    }
    
    // Apply linear regression to momentum input
    // Note: The linreg implementation returns the forecast (extrapolated value)
    // But we need the fitted value at the current position for momentum
    // So we'll calculate linear regression manually to get the correct value
    
    let mut momentum = alloc_with_nan_prefix(len, warmup);
    
    for end_idx in warmup..len {
        let start_idx = end_idx.saturating_sub(length - 1);
        
        // Skip if we don't have valid data
        if mom_input[end_idx].is_nan() {
            continue;
        }
        
        // Calculate linear regression for the window
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;
        let mut count = 0;
        
        for i in start_idx..=end_idx {
            if !mom_input[i].is_nan() {
                let x = (i - start_idx) as f64;
                let y = mom_input[i];
                sum_x += x;
                sum_y += y;
                sum_xy += x * y;
                sum_x2 += x * x;
                count += 1;
            }
        }
        
        if count >= 2 {
            let n = count as f64;
            let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
            let intercept = (sum_y - slope * sum_x) / n;
            
            // Get the fitted value at the current position (x = length - 1)
            momentum[end_idx] = intercept + slope * (length - 1) as f64;
        }
    }
    
    Ok(BeardySqueezeProOutput {
        momentum,
        squeeze,
    })
}

// Streaming support for real-time updates
#[derive(Debug, Clone)]
pub struct BeardySqueezeProStream {
    params: BeardySqueezeProParams,
    high_buffer: Vec<f64>,
    low_buffer: Vec<f64>,
    close_buffer: Vec<f64>,
    initialized: bool,
}

impl BeardySqueezeProStream {
    pub fn try_new(params: BeardySqueezeProParams) -> Result<Self, BeardySqueezeProError> {
        let length = params.length.unwrap_or(20);
        if length == 0 {
            return Err(BeardySqueezeProError::InvalidLength { length });
        }
        
        Ok(Self {
            params,
            high_buffer: Vec::with_capacity(length * 2),
            low_buffer: Vec::with_capacity(length * 2),
            close_buffer: Vec::with_capacity(length * 2),
            initialized: false,
        })
    }
    
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<(f64, f64)> {
        let length = self.params.length.unwrap_or(20);
        
        // Add new values to buffers
        self.high_buffer.push(high);
        self.low_buffer.push(low);
        self.close_buffer.push(close);
        
        // Keep only what we need (2x length for linear regression)
        if self.high_buffer.len() > length * 2 {
            self.high_buffer.remove(0);
            self.low_buffer.remove(0);
            self.close_buffer.remove(0);
        }
        
        // Need at least length bars to calculate
        if self.close_buffer.len() < length {
            return None;
        }
        
        // Calculate using buffered data
        let input = BeardySqueezeProInput::from_slices(
            &self.high_buffer,
            &self.low_buffer,
            &self.close_buffer,
            self.params.clone(),
        );
        
        match beardy_squeeze_pro(&input) {
            Ok(output) => {
                let last_idx = output.momentum.len() - 1;
                Some((output.momentum[last_idx], output.squeeze[last_idx]))
            }
            Err(_) => None,
        }
    }
    
    pub fn reset(&mut self) {
        self.high_buffer.clear();
        self.low_buffer.clear();
        self.close_buffer.clear();
        self.initialized = false;
    }
}

// ==================== PYTHON BINDINGS ====================
#[cfg(feature = "python")]
#[pyfunction(name = "beardy_squeeze_pro")]
#[pyo3(signature = (high, low, close, length=20, bb_mult=2.0, kc_mult_high=1.0, kc_mult_mid=1.5, kc_mult_low=2.0))]
pub fn beardy_squeeze_pro_py<'py>(
    py: Python<'py>,
    high: numpy::PyReadonlyArray1<'py, f64>,
    low: numpy::PyReadonlyArray1<'py, f64>,
    close: numpy::PyReadonlyArray1<'py, f64>,
    length: usize,
    bb_mult: f64,
    kc_mult_high: f64,
    kc_mult_mid: f64,
    kc_mult_low: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;
    
    let params = BeardySqueezeProParams {
        length: Some(length),
        bb_mult: Some(bb_mult),
        kc_mult_high: Some(kc_mult_high),
        kc_mult_mid: Some(kc_mult_mid),
        kc_mult_low: Some(kc_mult_low),
    };
    
    let input = BeardySqueezeProInput::from_slices(high_slice, low_slice, close_slice, params);
    
    let result = py
        .allow_threads(|| beardy_squeeze_pro(&input))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    
    Ok((
        result.momentum.into_pyarray(py),
        result.squeeze.into_pyarray(py),
    ))
}

// ==================== WASM BINDINGS ====================
#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct BeardySqueezeProJsResult {
    pub momentum: Vec<f64>,
    pub squeeze: Vec<f64>,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = beardy_squeeze_pro)]
pub fn beardy_squeeze_pro_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    length: usize,
    bb_mult: f64,
    kc_mult_high: f64,
    kc_mult_mid: f64,
    kc_mult_low: f64,
) -> Result<JsValue, JsValue> {
    let params = BeardySqueezeProParams {
        length: Some(length),
        bb_mult: Some(bb_mult),
        kc_mult_high: Some(kc_mult_high),
        kc_mult_mid: Some(kc_mult_mid),
        kc_mult_low: Some(kc_mult_low),
    };
    
    let input = BeardySqueezeProInput::from_slices(high, low, close, params);
    
    let result = beardy_squeeze_pro(&input)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    
    let js_result = BeardySqueezeProJsResult {
        momentum: result.momentum,
        squeeze: result.squeeze,
    };
    
    serde_wasm_bindgen::to_value(&js_result)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

// ==================== UNIT TESTS ====================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use std::error::Error;

    // Helper macro to skip unsupported kernel tests
    macro_rules! skip_if_unsupported {
        ($kernel:expr, $test_name:expr) => {
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            {
                if matches!($kernel, Kernel::Avx2 | Kernel::Avx512 | Kernel::Avx2Batch | Kernel::Avx512Batch) {
                    eprintln!("Skipping {} - AVX not supported", $test_name);
                    return Ok(());
                }
            }
        };
    }

    fn check_beardy_squeeze_pro_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        
        let input = BeardySqueezeProInput::with_default_candles(&candles);
        let result = beardy_squeeze_pro_with_kernel(&input, kernel)?;
        
        // Check that we have valid output
        assert_eq!(result.momentum.len(), candles.close.len());
        assert_eq!(result.squeeze.len(), candles.close.len());
        
        // Verify warmup period has NaN values  
        let warmup = 19; // length - 1
        for i in 0..warmup {
            assert!(result.momentum[i].is_nan(), "[{}] Expected NaN at index {}", test_name, i);
            assert!(result.squeeze[i].is_nan(), "[{}] Expected NaN at index {}", test_name, i);
        }
        
        // Verify we have valid values after warmup
        assert!(!result.momentum[warmup].is_nan(), "[{}] Expected valid value after warmup", test_name);
        assert!(!result.squeeze[warmup].is_nan(), "[{}] Expected valid value after warmup", test_name);
        
        Ok(())
    }
    
    fn check_beardy_squeeze_pro_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        
        let params = BeardySqueezeProParams {
            length: None,
            bb_mult: None,
            kc_mult_high: None,
            kc_mult_mid: None,
            kc_mult_low: None,
        };
        
        let input = BeardySqueezeProInput::from_candles(&candles, params);
        let result = beardy_squeeze_pro_with_kernel(&input, kernel)?;
        
        assert_eq!(result.momentum.len(), candles.close.len());
        assert_eq!(result.squeeze.len(), candles.close.len());
        
        Ok(())
    }
    
    fn check_beardy_squeeze_pro_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        
        let input = BeardySqueezeProInput::with_default_candles(&candles);
        let result = beardy_squeeze_pro_with_kernel(&input, kernel)?;
        
        assert_eq!(result.momentum.len(), candles.close.len());
        assert_eq!(result.squeeze.len(), candles.close.len());
        
        Ok(())
    }
    
    fn check_beardy_squeeze_pro_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let params = BeardySqueezeProParams {
            length: Some(0),
            bb_mult: None,
            kc_mult_high: None,
            kc_mult_mid: None,
            kc_mult_low: None,
        };
        
        let input = BeardySqueezeProInput::from_slices(&data, &data, &data, params);
        let result = beardy_squeeze_pro_with_kernel(&input, kernel);
        
        assert!(result.is_err(), "[{}] Should fail with zero period", test_name);
        Ok(())
    }
    
    fn check_beardy_squeeze_pro_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = vec![1.0, 2.0, 3.0];
        let params = BeardySqueezeProParams {
            length: Some(10),
            bb_mult: None,
            kc_mult_high: None,
            kc_mult_mid: None,
            kc_mult_low: None,
        };
        
        let input = BeardySqueezeProInput::from_slices(&data, &data, &data, params);
        let result = beardy_squeeze_pro_with_kernel(&input, kernel);
        
        assert!(result.is_err(), "[{}] Should fail when period exceeds length", test_name);
        Ok(())
    }
    
    fn check_beardy_squeeze_pro_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = vec![42.0];
        let params = BeardySqueezeProParams::default();
        
        let input = BeardySqueezeProInput::from_slices(&data, &data, &data, params);
        let result = beardy_squeeze_pro_with_kernel(&input, kernel);
        
        assert!(result.is_err(), "[{}] Should fail with very small dataset", test_name);
        Ok(())
    }
    
    fn check_beardy_squeeze_pro_empty_input(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let empty_data: Vec<f64> = vec![];
        let params = BeardySqueezeProParams::default();
        
        let input = BeardySqueezeProInput::from_slices(&empty_data, &empty_data, &empty_data, params);
        let result = beardy_squeeze_pro_with_kernel(&input, kernel);
        
        assert!(result.is_err(), "[{}] Should fail with empty input", test_name);
        Ok(())
    }
    
    fn check_beardy_squeeze_pro_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let nan_data = vec![f64::NAN; 50];
        let params = BeardySqueezeProParams::default();
        
        let input = BeardySqueezeProInput::from_slices(&nan_data, &nan_data, &nan_data, params);
        let result = beardy_squeeze_pro_with_kernel(&input, kernel);
        
        assert!(result.is_err(), "[{}] Should fail with all NaN values", test_name);
        Ok(())
    }
    
    fn check_beardy_squeeze_pro_inconsistent_slices(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = vec![1.0; 10];
        let low = vec![0.9; 10];
        let close = vec![0.95; 5]; // Different length
        let params = BeardySqueezeProParams::default();
        
        let input = BeardySqueezeProInput::from_slices(&high, &low, &close, params);
        let result = beardy_squeeze_pro_with_kernel(&input, kernel);
        
        assert!(result.is_err(), "[{}] Should fail with inconsistent slice lengths", test_name);
        Ok(())
    }
    
    fn check_beardy_squeeze_pro_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        
        let input = BeardySqueezeProInput::with_default_candles(&candles);
        let result = beardy_squeeze_pro_with_kernel(&input, kernel)?;
        
        // Check that NaN handling is consistent
        assert_eq!(result.momentum.len(), candles.close.len());
        assert_eq!(result.squeeze.len(), candles.close.len());
        
        // After warmup, should have no NaN values
        if result.momentum.len() > 40 {
            for i in 40..result.momentum.len() {
                assert!(!result.momentum[i].is_nan(), "[{}] Unexpected NaN in momentum at {}", test_name, i);
                assert!(!result.squeeze[i].is_nan(), "[{}] Unexpected NaN in squeeze at {}", test_name, i);
            }
        }
        
        Ok(())
    }
    
    fn check_beardy_squeeze_pro_builder(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        
        let result = BeardySqueezeProBuilder::new()
            .length(30)
            .bb_mult(2.5)
            .kc_mult_high(1.2)
            .kc_mult_mid(1.8)
            .kc_mult_low(2.5)
            .kernel(kernel)
            .build(&candles)?;
        
        assert_eq!(result.momentum.len(), candles.close.len());
        assert_eq!(result.squeeze.len(), candles.close.len());
        
        Ok(())
    }
    
    fn check_beardy_squeeze_pro_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        
        let params = BeardySqueezeProParams::default();
        let mut stream = BeardySqueezeProStream::try_new(params.clone())?;
        
        let input = BeardySqueezeProInput::from_candles(&candles, params);
        let batch_result = beardy_squeeze_pro_with_kernel(&input, kernel)?;
        
        let mut stream_momentum = Vec::new();
        let mut stream_squeeze = Vec::new();
        
        for i in 0..candles.close.len().min(100) {
            if let Some((mom, sqz)) = stream.update(candles.high[i], candles.low[i], candles.close[i]) {
                stream_momentum.push(mom);
                stream_squeeze.push(sqz);
            }
        }
        
        // Stream should produce values after warmup
        assert!(!stream_momentum.is_empty(), "[{}] Stream should produce values", test_name);
        
        Ok(())
    }
    
    #[cfg(debug_assertions)]
    fn check_beardy_squeeze_pro_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        
        let test_params = vec![
            BeardySqueezeProParams::default(),
            BeardySqueezeProParams {
                length: Some(10),
                bb_mult: Some(1.5),
                kc_mult_high: Some(0.8),
                kc_mult_mid: Some(1.2),
                kc_mult_low: Some(1.8),
            },
            BeardySqueezeProParams {
                length: Some(30),
                bb_mult: Some(3.0),
                kc_mult_high: Some(1.5),
                kc_mult_mid: Some(2.0),
                kc_mult_low: Some(2.5),
            },
        ];
        
        for params in test_params {
            let input = BeardySqueezeProInput::from_candles(&candles, params);
            let output = beardy_squeeze_pro_with_kernel(&input, kernel)?;
            
            for (i, &val) in output.momentum.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }
                
                let bits = val.to_bits();
                assert!(
                    bits != 0x11111111_11111111 && bits != 0x22222222_22222222 && bits != 0x33333333_33333333,
                    "[{}] Found poison value in momentum at {}: 0x{:016X}", test_name, i, bits
                );
            }
            
            for (i, &val) in output.squeeze.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }
                
                let bits = val.to_bits();
                assert!(
                    bits != 0x11111111_11111111 && bits != 0x22222222_22222222 && bits != 0x33333333_33333333,
                    "[{}] Found poison value in squeeze at {}: 0x{:016X}", test_name, i, bits
                );
            }
        }
        
        Ok(())
    }
    
    #[cfg(not(debug_assertions))]
    fn check_beardy_squeeze_pro_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }
    
    // Test generation macro
    macro_rules! generate_beardy_squeeze_pro_tests {
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
    
    // Generate all tests
    generate_beardy_squeeze_pro_tests!(
        check_beardy_squeeze_pro_accuracy,
        check_beardy_squeeze_pro_partial_params,
        check_beardy_squeeze_pro_default_candles,
        check_beardy_squeeze_pro_zero_period,
        check_beardy_squeeze_pro_period_exceeds_length,
        check_beardy_squeeze_pro_very_small_dataset,
        check_beardy_squeeze_pro_empty_input,
        check_beardy_squeeze_pro_all_nan,
        check_beardy_squeeze_pro_inconsistent_slices,
        check_beardy_squeeze_pro_nan_handling,
        check_beardy_squeeze_pro_builder,
        check_beardy_squeeze_pro_streaming,
        check_beardy_squeeze_pro_no_poison
    );
}