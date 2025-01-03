/// # Triple Exponential Moving Average (TEMA)
///
/// A variant of Exponential Moving Average (EMA) computed three times to further
/// reduce lag and smooth out noise. TEMA is calculated using three consecutive EMAs:  
/// `TEMA = 3*EMA1 - 3*EMA2 + EMA3`,  
/// where each `EMA` is computed over the specified `period`.
///
/// ## Parameters
/// - **period**: Window size (number of data points). Must be ≥ 1.
///
/// ## Errors
/// - **AllValuesNaN**: tema: All input data values are `NaN`.
/// - **InvalidPeriod**: tema: `period` < 1.
/// - **NotEnoughDataPoints**: tema: The data length is insufficient for the requested `period`.
///
/// ## Returns
/// - **`Ok(TemaOutput)`** on success, containing a `Vec<f64>` of length matching the input.
/// - **`Err(TemaError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};

#[derive(Debug, Clone)]
pub enum TemaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct TemaParams {
    pub period: Option<usize>,
}

impl Default for TemaParams {
    fn default() -> Self {
        Self { period: Some(9) }
    }
}

#[derive(Debug, Clone)]
pub struct TemaInput<'a> {
    pub data: TemaData<'a>,
    pub params: TemaParams,
}

impl<'a> TemaInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: TemaParams) -> Self {
        Self {
            data: TemaData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: TemaParams) -> Self {
        Self {
            data: TemaData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: TemaData::Candles {
                candles,
                source: "close",
            },
            params: TemaParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| TemaParams::default().period.unwrap())
    }
}

#[derive(Debug, Clone)]
pub struct TemaOutput {
    pub values: Vec<f64>,
}

use thiserror::Error;

#[derive(Debug, Error)]
pub enum TemaError {
    #[error("Tema: All values in input data are NaN for TEMA calculation.")]
    AllValuesNaN,
    #[error("Tema: Period cannot be zero or negative for TEMA. period = {period}")]
    InvalidPeriod { period: usize },
    #[error(
        "Tema: Not enough data points to calculate TEMA. period = {period}, data length = {data_len}"
    )]
    NotEnoughDataPoints { period: usize, data_len: usize },
}

#[inline]
pub fn tema(input: &TemaInput) -> Result<TemaOutput, TemaError> {
    let data: &[f64] = match &input.data {
        TemaData::Candles { candles, source } => source_type(candles, source),
        TemaData::Slice(slice) => slice,
    };

    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(TemaError::AllValuesNaN),
    };

    let n = data.len();
    let period = input.get_period();

    if period < 1 {
        return Err(TemaError::InvalidPeriod { period });
    }

    if period > n {
        return Err(TemaError::NotEnoughDataPoints {
            period,
            data_len: n,
        });
    }

    let lookback = (period - 1) * 3;
    if n == 0 || n <= lookback {
        return Ok(TemaOutput {
            values: vec![f64::NAN; n],
        });
    }

    let per = 2.0 / (period as f64 + 1.0);
    let per1 = 1.0 - per;

    let mut ema1 = data[first_valid_idx];
    let mut ema2 = 0.0;
    let mut ema3 = 0.0;

    let mut tema_values = vec![f64::NAN; n];

    for i in first_valid_idx..n {
        let price = data[i];

        ema1 = ema1 * per1 + price * per;

        if i == (period - 1) {
            ema2 = ema1;
        }
        if i >= (period - 1) {
            ema2 = ema2 * per1 + ema1 * per;
        }

        if i == 2 * (period - 1) {
            ema3 = ema2;
        }
        if i >= 2 * (period - 1) {
            ema3 = ema3 * per1 + ema2 * per;
        }

        if i >= lookback {
            tema_values[i] = 3.0 * ema1 - 3.0 * ema2 + ema3;
        }
    }

    Ok(TemaOutput {
        values: tema_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_tema_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");
        let params = TemaParams { period: Some(9) };
        let input = TemaInput::from_candles(&candles, "close", params);
        let tema_result = tema(&input).expect("Failed to calculate TEMA");
        let expected_last_five = [
            59281.895570662884,
            59257.25021607971,
            59172.23342859784,
            59175.218345941066,
            58934.24395798363,
        ];
        assert!(tema_result.values.len() >= 5);
        assert_eq!(tema_result.values.len(), close_prices.len());
        let start_index = tema_result.values.len() - 5;
        let result_last_five = &tema_result.values[start_index..];
        for (i, &value) in result_last_five.iter().enumerate() {
            let expected_value = expected_last_five[i];
            assert!(
                (value - expected_value).abs() < 1e-8,
                "TEMA mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }
    }

    #[test]
    fn test_tema_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let default_params = TemaParams { period: None };
        let input = TemaInput::from_candles(&candles, "close", default_params);
        let output = tema(&input).expect("Failed TEMA with default params");
        assert_eq!(output.values.len(), candles.close.len());
        let params_period_14 = TemaParams { period: Some(14) };
        let input2 = TemaInput::from_candles(&candles, "hl2", params_period_14);
        let output2 = tema(&input2).expect("Failed TEMA with period=14, source=hl2");
        assert_eq!(output2.values.len(), candles.close.len());
        let params_custom = TemaParams { period: Some(10) };
        let input3 = TemaInput::from_candles(&candles, "hlc3", params_custom);
        let output3 = tema(&input3).expect("Failed TEMA fully custom");
        assert_eq!(output3.values.len(), candles.close.len());
    }
    #[test]
    fn test_tema_params_with_default() {
        let default_params = TemaParams::default();
        assert_eq!(default_params.period, Some(9));
    }

    #[test]
    fn test_tema_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = TemaInput::with_default_candles(&candles);
        match input.data {
            TemaData::Candles { source, .. } => {
                assert_eq!(source, "close");
            }
            _ => panic!("Expected TemaData::Candles variant"),
        }
        assert_eq!(input.params.period, Some(9));
    }

    #[test]
    fn test_tema_zero_period() {
        let data = [10.0, 20.0, 30.0];
        let params = TemaParams { period: Some(0) };
        let input = TemaInput::from_slice(&data, params);
        let result = tema(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                e.to_string().contains("Period cannot be zero"),
                "Unexpected error: {}",
                e
            );
        }
    }

    #[test]
    fn test_tema_period_exceeding_data_length() {
        let data = [10.0, 20.0, 30.0];
        let params = TemaParams { period: Some(5) };
        let input = TemaInput::from_slice(&data, params);
        let result = tema(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_tema_very_small_data_set() {
        let data = [42.0; 10];
        let params = TemaParams { period: Some(9) };
        let input = TemaInput::from_slice(&data, params);
        let result = tema(&input).expect("Should handle near-minimal data");
        assert_eq!(result.values.len(), data.len());
    }

    #[test]
    fn test_tema_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let first_params = TemaParams { period: Some(9) };
        let first_input = TemaInput::from_candles(&candles, "close", first_params);
        let first_result = tema(&first_input).expect("Failed to calculate first TEMA");
        assert_eq!(first_result.values.len(), candles.close.len());

        let second_params = TemaParams { period: Some(5) };
        let second_input = TemaInput::from_slice(&first_result.values, second_params);
        let second_result = tema(&second_input).expect("Failed to calculate second TEMA");
        assert_eq!(second_result.values.len(), first_result.values.len());
        for i in 240..second_result.values.len() {
            assert!(!second_result.values[i].is_nan());
        }
    }

    #[test]
    fn test_tema_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = TemaParams { period: Some(9) };
        let input = TemaInput::from_candles(&candles, "close", params);
        let result = tema(&input).expect("Failed to calculate TEMA");
        assert_eq!(result.values.len(), candles.close.len());
        if result.values.len() > 50 {
            for i in 50..result.values.len() {
                assert!(!result.values[i].is_nan());
            }
        }
    }
}
