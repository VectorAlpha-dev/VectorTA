use crate::utilities::data_loader::{source_type, Candles};
use std::error::Error;

#[derive(Debug, Clone)]
pub enum EmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct EmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct EmaParams {
    pub period: Option<usize>,
}

impl Default for EmaParams {
    fn default() -> Self {
        EmaParams { period: Some(9) }
    }
}

#[derive(Debug, Clone)]
pub struct EmaInput<'a> {
    pub data: EmaData<'a>,
    pub params: EmaParams,
}

impl<'a> EmaInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: EmaParams) -> Self {
        Self {
            data: EmaData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: EmaParams) -> Self {
        Self {
            data: EmaData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: EmaData::Candles {
                candles,
                source: "close",
            },
            params: EmaParams::default(),
        }
    }

    fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| EmaParams::default().period.unwrap())
    }
}

#[inline]
pub fn ema(input: &EmaInput) -> Result<EmaOutput, Box<dyn Error>> {
    let data: &[f64] = match &input.data {
        EmaData::Candles { candles, source } => source_type(candles, source),
        EmaData::Slice(slice) => slice,
    };
    let len: usize = data.len();
    let period: usize = input.get_period();

    if period == 0 || period > data.len() {
        return Err("Invalid period specified for EMA calculation.".into());
    }

    let len = data.len();
    let alpha = 2.0 / (period as f64 + 1.0);
    let mut ema_values = Vec::with_capacity(len);

    let mut last_ema = data[0];
    ema_values.push(last_ema);

    for &value in &data[1..] {
        last_ema = alpha * value + (1.0 - alpha) * last_ema;
        ema_values.push(last_ema);
    }

    Ok(EmaOutput { values: ema_values })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_ema_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let close_prices = &candles.close;
        let params = EmaParams { period: Some(9) };

        let old_style_input = EmaInput::from_candles(&candles, "close", params);
        let ema_result = ema(&old_style_input).expect("Failed to calculate EMA");

        let expected_last_five_ema = [59302.2, 59277.9, 59230.2, 59215.1, 59103.1];

        assert!(
            ema_result.values.len() >= 5,
            "Not enough EMA values for the test"
        );
        assert_eq!(
            ema_result.values.len(),
            close_prices.len(),
            "EMA output length does not match input length"
        );

        let start_index = ema_result.values.len().saturating_sub(5);
        let result_last_five_ema = &ema_result.values[start_index..];

        for (i, &value) in result_last_five_ema.iter().enumerate() {
            assert!(
                (value - expected_last_five_ema[i]).abs() < 1e-1,
                "EMA value mismatch at index {}: expected {}, got {}",
                i,
                expected_last_five_ema[i],
                value
            );
        }

        let default_input = EmaInput::with_default_candles(&candles);
        let default_ema_result = ema(&default_input).expect("Failed to calculate default EMA");
        assert!(
            !default_ema_result.values.is_empty(),
            "Should produce EMA values with default params"
        );
    }
    #[test]
    fn test_ema_params_with_default_period() {
        let params = EmaParams::default();
        assert_eq!(params.period, Some(9));
    }

    #[test]
    fn test_ema_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = EmaInput::with_default_candles(&candles);
        match input.data {
            EmaData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected EmaData::Candles variant"),
        }
        assert_eq!(input.params.period, Some(9));
    }

    #[test]
    fn test_ema_with_zero_period() {
        let data = [10.0, 20.0, 30.0];
        let params = EmaParams { period: Some(0) };
        let input = EmaInput::from_slice(&data, params);
        let result = ema(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_ema_period_exceeding_data_length() {
        let data = [10.0, 20.0, 30.0];
        let params = EmaParams { period: Some(10) };
        let input = EmaInput::from_slice(&data, params);
        let result = ema(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_ema_very_small_data_set() {
        let data = [42.0];
        let params = EmaParams { period: Some(9) };
        let input = EmaInput::from_slice(&data, params);
        let result = ema(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_ema_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params_first = EmaParams { period: Some(9) };
        let input_first = EmaInput::from_candles(&candles, "close", params_first);
        let result_first = ema(&input_first).expect("Failed to calculate first EMA");
        assert_eq!(result_first.values.len(), candles.close.len());
        let params_second = EmaParams { period: Some(5) };
        let input_second = EmaInput::from_slice(&result_first.values, params_second);
        let result_second = ema(&input_second).expect("Failed to calculate second EMA");
        assert_eq!(result_second.values.len(), result_first.values.len());
    }

    #[test]
    fn test_ema_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = EmaParams { period: Some(9) };
        let input = EmaInput::from_candles(&candles, "close", params);
        let result = ema(&input).expect("Failed to calculate EMA");
        assert_eq!(result.values.len(), candles.close.len());
        if result.values.len() > 240 {
            for i in 240..result.values.len() {
                assert!(!result.values[i].is_nan());
            }
        }
    }
}
