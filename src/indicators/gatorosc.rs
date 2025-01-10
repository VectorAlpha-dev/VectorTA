/// # Gator Oscillator (GATOR)
///
/// The Gator Oscillator is based on Bill Williams' Alligator indicator. It calculates
/// three exponentially smoothed averages (often referred to as the Jaws, Teeth, and Lips)
/// of a given `source`. Then it derives two lines:
/// - **upper** = `abs(Jaws - Teeth)`
/// - **lower** = `-abs(Teeth - Lips)`
///
/// and calculates their 1-period momentum changes:
/// - **upper_change** = `upper[i] - upper[i-1]`
/// - **lower_change** = `-(lower[i] - lower[i-1])`
///
/// A typical configuration uses lengths and shift values of:
/// - Jaws: length=13, shift=8
/// - Teeth: length=8, shift=5
/// - Lips: length=5, shift=3
///
/// ## Parameters
/// - **jaws_length**: The EMA length for Jaws. Defaults to 13.
/// - **jaws_shift**: How many bars to shift Jaws forward. Defaults to 8.
/// - **teeth_length**: The EMA length for Teeth. Defaults to 8.
/// - **teeth_shift**: How many bars to shift Teeth forward. Defaults to 5.
/// - **lips_length**: The EMA length for Lips. Defaults to 5.
/// - **lips_shift**: How many bars to shift Lips forward. Defaults to 3.
///
/// ## Errors
/// - **EmptyData**: gator: Input data slice is empty.
/// - **AllValuesNaN**: gator: All input data values are `NaN`.
/// - **InvalidSettings**: gator: One of the lengths or shifts is zero.
/// - **NotEnoughValidData**: gator: Not enough valid (non-`NaN`) data points to compute the indicator.
///
/// ## Returns
/// - **`Ok(GatorOscOutput)`** on success, containing four `Vec<f64>` matching the input length:
///   - **upper**
///   - **lower**
///   - **upper_change**
///   - **lower_change**
///   Each vector will contain `NaN` for the bars before the indicator can be calculated,
///   but does not propagate extra `NaN`s once enough valid data is available.
/// - **`Err(GatorOscError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum GatorOscData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct GatorOscOutput {
    pub upper: Vec<f64>,
    pub lower: Vec<f64>,
    pub upper_change: Vec<f64>,
    pub lower_change: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct GatorOscParams {
    pub jaws_length: Option<usize>,
    pub jaws_shift: Option<usize>,
    pub teeth_length: Option<usize>,
    pub teeth_shift: Option<usize>,
    pub lips_length: Option<usize>,
    pub lips_shift: Option<usize>,
}

impl Default for GatorOscParams {
    fn default() -> Self {
        Self {
            jaws_length: Some(13),
            jaws_shift: Some(8),
            teeth_length: Some(8),
            teeth_shift: Some(5),
            lips_length: Some(5),
            lips_shift: Some(3),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GatorOscInput<'a> {
    pub data: GatorOscData<'a>,
    pub params: GatorOscParams,
}

impl<'a> GatorOscInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: GatorOscParams) -> Self {
        Self {
            data: GatorOscData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: GatorOscParams) -> Self {
        Self {
            data: GatorOscData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: GatorOscData::Candles {
                candles,
                source: "close",
            },
            params: GatorOscParams::default(),
        }
    }

    pub fn get_jaws_length(&self) -> usize {
        self.params
            .jaws_length
            .unwrap_or_else(|| GatorOscParams::default().jaws_length.unwrap())
    }

    pub fn get_jaws_shift(&self) -> usize {
        self.params
            .jaws_shift
            .unwrap_or_else(|| GatorOscParams::default().jaws_shift.unwrap())
    }

    pub fn get_teeth_length(&self) -> usize {
        self.params
            .teeth_length
            .unwrap_or_else(|| GatorOscParams::default().teeth_length.unwrap())
    }

    pub fn get_teeth_shift(&self) -> usize {
        self.params
            .teeth_shift
            .unwrap_or_else(|| GatorOscParams::default().teeth_shift.unwrap())
    }

    pub fn get_lips_length(&self) -> usize {
        self.params
            .lips_length
            .unwrap_or_else(|| GatorOscParams::default().lips_length.unwrap())
    }

    pub fn get_lips_shift(&self) -> usize {
        self.params
            .lips_shift
            .unwrap_or_else(|| GatorOscParams::default().lips_shift.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum GatorOscError {
    #[error("gator: Empty data provided.")]
    EmptyData,
    #[error("gator: All values are NaN.")]
    AllValuesNaN,
    #[error("gator: Invalid settings (length or shift is zero).")]
    InvalidSettings,
    #[error("gator: Not enough valid data.")]
    NotEnoughValidData,
}

#[inline]
pub fn gatorosc(input: &GatorOscInput) -> Result<GatorOscOutput, GatorOscError> {
    let data: &[f64] = match &input.data {
        GatorOscData::Candles { candles, source } => source_type(candles, source),
        GatorOscData::Slice(slice) => slice,
    };

    if data.is_empty() {
        return Err(GatorOscError::EmptyData);
    }

    let jaws_length = input.get_jaws_length();
    let jaws_shift = input.get_jaws_shift();
    let teeth_length = input.get_teeth_length();
    let teeth_shift = input.get_teeth_shift();
    let lips_length = input.get_lips_length();
    let lips_shift = input.get_lips_shift();

    if jaws_length == 0
        || jaws_shift == 0
        || teeth_length == 0
        || teeth_shift == 0
        || lips_length == 0
        || lips_shift == 0
    {
        return Err(GatorOscError::InvalidSettings);
    }

    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(GatorOscError::AllValuesNaN),
    };

    fn ema(data: &[f64], length: usize, start_idx: usize) -> Vec<f64> {
        let alpha = 1.0 / length as f64;
        let mut output = vec![f64::NAN; data.len()];
        let mut prev = if data[start_idx].is_nan() {
            0.0
        } else {
            data[start_idx]
        };
        output[start_idx] = prev;
        for i in (start_idx + 1)..data.len() {
            let val = if data[i].is_nan() { prev } else { data[i] };
            let next_ema = alpha * val + (1.0 - alpha) * prev;
            output[i] = next_ema;
            prev = next_ema;
        }
        output
    }

    fn shift_series(data: &[f64], shift: usize) -> Vec<f64> {
        let mut shifted = vec![f64::NAN; data.len()];
        for (i, &val) in data.iter().enumerate() {
            let j = i + shift;
            if j < data.len() {
                shifted[j] = val;
            }
        }
        shifted
    }

    let jaws_ema = ema(data, jaws_length, first_valid_idx);
    let jaws = shift_series(&jaws_ema, jaws_shift);

    let teeth_ema = ema(data, teeth_length, first_valid_idx);
    let teeth = shift_series(&teeth_ema, teeth_shift);

    let lips_ema = ema(data, lips_length, first_valid_idx);
    let lips = shift_series(&lips_ema, lips_shift);

    let mut upper = vec![f64::NAN; data.len()];
    let mut lower = vec![f64::NAN; data.len()];
    for i in 0..data.len() {
        if !jaws[i].is_nan() && !teeth[i].is_nan() {
            upper[i] = (jaws[i] - teeth[i]).abs();
        }
        if !teeth[i].is_nan() && !lips[i].is_nan() {
            lower[i] = -(teeth[i] - lips[i]).abs();
        }
    }

    fn one_period_mom(data: &[f64]) -> Vec<f64> {
        let mut output = vec![f64::NAN; data.len()];
        for i in 1..data.len() {
            if data[i].is_nan() || data[i - 1].is_nan() {
                continue;
            }
            output[i] = data[i] - data[i - 1];
        }
        output
    }

    let upper_change_raw = one_period_mom(&upper);
    let lower_change_raw = one_period_mom(&lower);
    let mut upper_change = vec![f64::NAN; data.len()];
    let mut lower_change = vec![f64::NAN; data.len()];
    for i in 0..data.len() {
        if !upper_change_raw[i].is_nan() {
            upper_change[i] = upper_change_raw[i];
        }
        if !lower_change_raw[i].is_nan() {
            lower_change[i] = -lower_change_raw[i];
        }
    }

    let needed = jaws_length
        .max(teeth_length)
        .max(lips_length)
        .saturating_add(jaws_shift.max(teeth_shift).max(lips_shift));
    let valid_data_count = data
        .iter()
        .skip(first_valid_idx)
        .filter(|v| !v.is_nan())
        .count();
    if valid_data_count < needed {
        return Err(GatorOscError::NotEnoughValidData);
    }

    Ok(GatorOscOutput {
        upper,
        lower,
        upper_change,
        lower_change,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_gatorosc_no_nans_after_valid() {
        let data = vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
        let params = GatorOscParams {
            jaws_length: Some(2),
            jaws_shift: Some(1),
            teeth_length: Some(2),
            teeth_shift: Some(1),
            lips_length: Some(2),
            lips_shift: Some(1),
        };
        let input = GatorOscInput::from_slice(&data, params);
        let result = gatorosc(&input).expect("Failed to compute gator osc");
        for (i, &val) in result.upper.iter().enumerate() {
            if i > 1 {
                assert!(
                    !val.is_nan(),
                    "Found unexpected NaN in upper at index {}",
                    i
                );
            }
        }
        for (i, &val) in result.lower.iter().enumerate() {
            if i > 1 {
                assert!(
                    !val.is_nan(),
                    "Found unexpected NaN in lower at index {}",
                    i
                );
            }
        }
        for (i, &val) in result.upper_change.iter().enumerate() {
            if i > 2 {
                assert!(
                    !val.is_nan(),
                    "Found unexpected NaN in upper_change at index {}",
                    i
                );
            }
        }
        for (i, &val) in result.lower_change.iter().enumerate() {
            if i > 2 {
                assert!(
                    !val.is_nan(),
                    "Found unexpected NaN in lower_change at index {}",
                    i
                );
            }
        }
    }

    #[test]
    fn test_gatorosc_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = GatorOscInput::with_default_candles(&candles);
        let output = gatorosc(&input).expect("Failed to calculate gator osc with defaults");
        assert_eq!(output.upper.len(), candles.close.len());
        assert_eq!(output.lower.len(), candles.close.len());
        assert_eq!(output.upper_change.len(), candles.close.len());
        assert_eq!(output.lower_change.len(), candles.close.len());
    }

    #[test]
    fn test_gatorosc_empty_data() {
        let data: Vec<f64> = vec![];
        let params = GatorOscParams::default();
        let input = GatorOscInput::from_slice(&data, params);
        let result = gatorosc(&input);
        assert!(result.is_err());
        match result {
            Err(GatorOscError::EmptyData) => {}
            _ => panic!("Expected EmptyData error"),
        }
    }

    #[test]
    fn test_gatorosc_all_nan() {
        let data = vec![f64::NAN, f64::NAN, f64::NAN];
        let params = GatorOscParams::default();
        let input = GatorOscInput::from_slice(&data, params);
        let result = gatorosc(&input);
        assert!(result.is_err());
        match result {
            Err(GatorOscError::AllValuesNaN) => {}
            _ => panic!("Expected AllValuesNaN error"),
        }
    }

    #[test]
    fn test_gatorosc_invalid_settings() {
        let data = vec![10.0, 20.0, 30.0];
        let params = GatorOscParams {
            jaws_length: Some(0),
            jaws_shift: Some(8),
            teeth_length: Some(8),
            teeth_shift: Some(5),
            lips_length: Some(5),
            lips_shift: Some(3),
        };
        let input = GatorOscInput::from_slice(&data, params);
        let result = gatorosc(&input);
        assert!(result.is_err());
        match result {
            Err(GatorOscError::InvalidSettings) => {}
            _ => panic!("Expected InvalidSettings error"),
        }
    }

    #[test]
    fn test_gatorosc_not_enough_data() {
        let data = vec![10.0];
        let params = GatorOscParams::default();
        let input = GatorOscInput::from_slice(&data, params);
        let result = gatorosc(&input);
        assert!(result.is_err());
        match result {
            Err(GatorOscError::NotEnoughValidData) => {}
            _ => panic!("Expected NotEnoughValidData error"),
        }
    }

    #[test]
    fn test_gatorosc_basic_computation_check() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let params = GatorOscParams {
            jaws_length: Some(2),
            jaws_shift: Some(1),
            teeth_length: Some(2),
            teeth_shift: Some(1),
            lips_length: Some(2),
            lips_shift: Some(1),
        };
        let input = GatorOscInput::from_slice(&data, params);
        let output = gatorosc(&input).expect("Failed to compute gator with small lengths");
        assert_eq!(output.upper.len(), data.len());
        assert_eq!(output.lower.len(), data.len());
        assert_eq!(output.upper_change.len(), data.len());
        assert_eq!(output.lower_change.len(), data.len());
    }
}
