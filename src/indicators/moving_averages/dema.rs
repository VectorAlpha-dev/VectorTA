use crate::utilities::data_loader::{source_type, Candles};
use std::error::Error;

#[derive(Debug, Clone)]
pub enum DemaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct DemaParams {
    pub period: Option<usize>,
}

impl Default for DemaParams {
    fn default() -> Self {
        Self { period: Some(30) }
    }
}

#[derive(Debug, Clone)]
pub struct DemaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct DemaInput<'a> {
    pub data: DemaData<'a>,
    pub params: DemaParams,
}

impl<'a> DemaInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: DemaParams) -> Self {
        Self {
            data: DemaData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: DemaParams) -> Self {
        Self {
            data: DemaData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: DemaData::Candles {
                candles,
                source: "close",
            },
            params: DemaParams::default(),
        }
    }

    #[inline]
    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| DemaParams::default().period.unwrap())
    }
}

#[inline]
pub fn dema(input: &DemaInput) -> Result<DemaOutput, Box<dyn Error>> {
    let data: &[f64] = match &input.data {
        DemaData::Candles { candles, source } => source_type(candles, source),
        DemaData::Slice(slice) => slice,
    };

    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => {
            return Err("All values in input data are NaN.".into());
        }
    };

    let size: usize = data.len();
    let period: usize = input.get_period();

    if period < 1 {
        return Err("Invalid DEMA period (must be >= 1).".into());
    }
    if size < 2 * (period - 1) {
        return Err("Not enough data to calculate DEMA for the specified period.".into());
    }
    let alpha = 2.0 / (period as f64 + 1.0);
    let alpha_1 = 1.0 - alpha;

    let mut output = vec![f64::NAN; size];

    let mut ema = data[first_valid_idx];
    let mut ema2 = ema;

    for i in first_valid_idx..size {
        ema = ema * alpha_1 + data[i] * alpha;

        if i == (period - 1) {
            ema2 = ema;
        }
        if i >= (period - 1) {
            ema2 = ema2 * alpha_1 + ema * alpha;
        }

        if i >= 2 * (period - 1) {
            output[i] = (2.0 * ema) - ema2;
        }
    }

    Ok(DemaOutput { values: output })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_dema_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = DemaParams { period: None };
        let input_default = DemaInput::from_candles(&candles, "close", default_params);
        let output_default = dema(&input_default).expect("Failed DEMA with default params");
        assert_eq!(
            output_default.values.len(),
            candles.close.len(),
            "Output length must match candle data length"
        );

        let params_period_14 = DemaParams { period: Some(14) };
        let input_period_14 = DemaInput::from_candles(&candles, "hl2", params_period_14);
        let output_period_14 =
            dema(&input_period_14).expect("Failed DEMA with period=14, source=hl2");
        assert_eq!(
            output_period_14.values.len(),
            candles.close.len(),
            "Output length must match candle data length"
        );

        let params_custom = DemaParams { period: Some(20) };
        let input_custom = DemaInput::from_candles(&candles, "hlc3", params_custom);
        let output_custom = dema(&input_custom).expect("Failed DEMA fully custom");
        assert_eq!(
            output_custom.values.len(),
            candles.close.len(),
            "Output length must match candle data length"
        );
    }

    #[test]
    fn test_dema_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = DemaInput::with_default_candles(&candles);
        let result = dema(&input).expect("Failed to calculate DEMA");

        let expected_last_five = [
            59189.73193987478,
            59129.24920772847,
            59058.80282420511,
            59011.5555611042,
            58908.370159946775,
        ];

        assert_eq!(
            result.values.len(),
            candles.close.len(),
            "DEMA output length does not match input length"
        );

        let start_index = result.values.len().saturating_sub(5);
        let last_five = &result.values[start_index..];
        for (i, &val) in last_five.iter().enumerate() {
            let exp = expected_last_five[i];
            assert!(
                (val - exp).abs() < 1e-6,
                "DEMA mismatch at index {}: expected {}, got {}",
                start_index + i,
                exp,
                val
            );
        }
    }

    #[test]
    fn test_dema_params_with_default_params() {
        let default_params = DemaParams::default();
        assert_eq!(
            default_params.period,
            Some(30),
            "Expected default period to be Some(30)"
        );
    }

    #[test]
    fn test_dema_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = DemaInput::with_default_candles(&candles);
        match input.data {
            DemaData::Candles { source, .. } => {
                assert_eq!(source, "close", "Default source should be 'close'");
            }
            _ => panic!("Expected DemaData::Candles variant"),
        }
        assert_eq!(
            input.params.period,
            Some(30),
            "Expected default period to be Some(30)"
        );
    }

    #[test]
    fn test_dema_with_zero_period() {
        let input_data = [10.0, 20.0, 30.0];
        let params = DemaParams { period: Some(0) };
        let input = DemaInput::from_slice(&input_data, params);
        let result = dema(&input);
        assert!(result.is_err(), "Expected an error for zero period");
        if let Err(e) = result {
            assert!(
                e.to_string().contains("Invalid DEMA period"),
                "Expected 'Invalid DEMA period' error message, got: {}",
                e
            );
        }
    }

    #[test]
    fn test_dema_with_period_exceeding_data_length() {
        let input_data = [10.0, 20.0, 30.0];
        let params = DemaParams { period: Some(10) };
        let input = DemaInput::from_slice(&input_data, params);
        let result = dema(&input);
        assert!(result.is_err(), "Expected an error for period > data.len()");
    }

    #[test]
    fn test_dema_very_small_data_set() {
        let input_data = [42.0];
        let params = DemaParams { period: Some(9) };
        let input = DemaInput::from_slice(&input_data, params);
        let result = dema(&input);
        assert!(
            result.is_err(),
            "Expected error for data smaller than required length"
        );
    }

    #[test]
    fn test_dema_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let first_params = DemaParams { period: Some(80) };
        let first_input = DemaInput::from_candles(&candles, "close", first_params);
        let first_result = dema(&first_input).expect("Failed to calculate first DEMA");
        assert_eq!(
            first_result.values.len(),
            candles.close.len(),
            "First DEMA output length mismatch"
        );

        let second_params = DemaParams { period: Some(60) };
        let second_input = DemaInput::from_slice(&first_result.values, second_params);
        let second_result = dema(&second_input).expect("Failed to calculate second DEMA");
        assert_eq!(
            second_result.values.len(),
            first_result.values.len(),
            "Second DEMA output length mismatch"
        );

        if second_result.values.len() > 240 {
            for i in 240..second_result.values.len() {
                assert!(
                    !second_result.values[i].is_nan(),
                    "Unexpected NaN at index {} in second DEMA",
                    i
                );
            }
        }
    }

    #[test]
    fn test_dema_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = DemaParams { period: Some(30) };
        let input = DemaInput::from_candles(&candles, "close", params);
        let result = dema(&input).expect("Failed to calculate DEMA");
        assert_eq!(
            result.values.len(),
            candles.close.len(),
            "DEMA output length mismatch"
        );

        if result.values.len() > 240 {
            for i in 240..result.values.len() {
                assert!(
                    !result.values[i].is_nan(),
                    "Unexpected NaN at index {} in final DEMA",
                    i
                );
            }
        }
    }
}
