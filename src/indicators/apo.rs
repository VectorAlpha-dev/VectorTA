use crate::utilities::data_loader::{source_type, Candles};
use std::error::Error;

#[derive(Debug, Clone)]
pub enum ApoData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct ApoParams {
    pub short_period: Option<usize>,
    pub long_period: Option<usize>,
}

impl Default for ApoParams {
    fn default() -> Self {
        Self {
            short_period: Some(10),
            long_period: Some(20),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ApoInput<'a> {
    pub data: ApoData<'a>,
    pub params: ApoParams,
}

impl<'a> ApoInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: ApoParams) -> Self {
        Self {
            data: ApoData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: ApoParams) -> Self {
        Self {
            data: ApoData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: ApoData::Candles {
                candles,
                source: "close",
            },
            params: ApoParams::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ApoOutput {
    pub values: Vec<f64>,
}

#[inline]
pub fn apo(input: &ApoInput) -> Result<ApoOutput, Box<dyn Error>> {
    let short: usize = input.params.short_period.unwrap_or(10);
    let long: usize = input.params.long_period.unwrap_or(20);

    if short == 0 || long == 0 {
        return Err("Invalid period specified for APO calculation.".into());
    }
    if short >= long {
        return Err("Short period must be less than the long period for APO.".into());
    }

    let data: &[f64] = match &input.data {
        ApoData::Candles { candles, source } => source_type(candles, source),
        ApoData::Slice(slice) => slice,
    };

    let len = data.len();
    if len == 0 {
        return Err("No candles available.".into());
    }

    if data.len() < long {
        return Err("Not enough data points to calculate APO".into());
    }

    let mut apo_values = Vec::with_capacity(len);
    apo_values.resize(len, f64::NAN);

    let alpha_short = 2.0 / (short as f64 + 1.0);
    let alpha_long = 2.0 / (long as f64 + 1.0);

    let mut short_ema = data[0];
    let mut long_ema = data[0];

    apo_values[0] = short_ema - long_ema;

    for i in 1..len {
        let price = data[i];
        short_ema = alpha_short * price + (1.0 - alpha_short) * short_ema;
        long_ema = alpha_long * price + (1.0 - alpha_long) * long_ema;

        let apo_val = short_ema - long_ema;
        apo_values[i] = apo_val;
    }

    Ok(ApoOutput { values: apo_values })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_apo_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = ApoInput::with_default_candles(&candles);
        let result = apo(&input).expect("Failed to calculate APO");

        let expected_last_five = [-429.8, -401.6, -386.1, -357.9, -374.1];

        assert!(
            result.values.len() >= 5,
            "Not enough APO values for the test"
        );

        assert_eq!(
            result.values.len(),
            candles.close.len(),
            "APO output length does not match input length!"
        );

        let start_index = result.values.len().saturating_sub(5);
        let result_last_five = &result.values[start_index..];

        for (i, &value) in result_last_five.iter().enumerate() {
            assert!(
                (value - expected_last_five[i]).abs() < 1e-1,
                "APO value mismatch at index {}: expected {}, got {}",
                i,
                expected_last_five[i],
                value
            );
        }

        for val in result.values.iter().skip(20 - 1) {
            assert!(
                val.is_finite(),
                "APO output should be finite after EMAs are established"
            );
        }
    }

    #[test]
    fn test_apo_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let default_params = ApoParams {
            short_period: None,
            long_period: None,
        };
        let input_default = ApoInput::from_candles(&candles, "close", default_params);
        let output_default = apo(&input_default).expect("Failed APO with default params");
        assert_eq!(output_default.values.len(), candles.close.len());
        let params_5_15 = ApoParams {
            short_period: Some(5),
            long_period: Some(15),
        };
        let input_5_15 = ApoInput::from_candles(&candles, "hl2", params_5_15);
        let output_5_15 = apo(&input_5_15).expect("Failed APO with short=5, long=15");
        assert_eq!(output_5_15.values.len(), candles.close.len());
        let params_12_26 = ApoParams {
            short_period: Some(12),
            long_period: Some(26),
        };
        let input_12_26 = ApoInput::from_candles(&candles, "hlc3", params_12_26);
        let output_12_26 = apo(&input_12_26).expect("Failed APO with short=12, long=26");
        assert_eq!(output_12_26.values.len(), candles.close.len());
    }

    #[test]
    fn test_apo_params_with_default_params() {
        let default_params = ApoParams::default();
        assert_eq!(default_params.short_period, Some(10));
        assert_eq!(default_params.long_period, Some(20));
    }

    #[test]
    fn test_apo_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = ApoInput::with_default_candles(&candles);
        match input.data {
            ApoData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected ApoData::Candles variant"),
        }
        assert_eq!(input.params.short_period, Some(10));
        assert_eq!(input.params.long_period, Some(20));
    }

    #[test]
    fn test_apo_with_zero_period() {
        let data = [10.0, 11.0, 12.0, 13.0];
        let params = ApoParams {
            short_period: Some(0),
            long_period: Some(20),
        };
        let input = ApoInput::from_slice(&data, params);
        let result = apo(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Invalid period"));
        }
    }

    #[test]
    fn test_apo_short_period_not_less_than_long_period() {
        let data = [10.0, 20.0, 30.0, 40.0, 50.0];
        let params = ApoParams {
            short_period: Some(20),
            long_period: Some(10),
        };
        let input = ApoInput::from_slice(&data, params);
        let result = apo(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Short period must be less"));
        }
    }

    #[test]
    fn test_apo_with_data_len_less_than_long_period() {
        let data = [10.0, 11.0, 12.0];
        let params = ApoParams {
            short_period: Some(1),
            long_period: Some(5),
        };
        let input = ApoInput::from_slice(&data, params);
        let result = apo(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Not enough data points"));
        }
    }

    #[test]
    fn test_apo_with_empty_data() {
        let data: [f64; 0] = [];
        let params = ApoParams {
            short_period: Some(5),
            long_period: Some(10),
        };
        let input = ApoInput::from_slice(&data, params);
        let result = apo(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("No candles available"));
        }
    }

    #[test]
    fn test_apo_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let first_params = ApoParams {
            short_period: Some(10),
            long_period: Some(20),
        };
        let first_input = ApoInput::from_candles(&candles, "close", first_params);
        let first_result = apo(&first_input).expect("Failed to calculate first APO");
        assert_eq!(first_result.values.len(), candles.close.len());
        let second_params = ApoParams {
            short_period: Some(5),
            long_period: Some(15),
        };
        let second_input = ApoInput::from_slice(&first_result.values, second_params);
        let second_result = apo(&second_input).expect("Failed to calculate second APO");
        assert_eq!(second_result.values.len(), first_result.values.len());
    }

    #[test]
    fn test_apo_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = ApoParams {
            short_period: Some(10),
            long_period: Some(20),
        };
        let input = ApoInput::from_candles(&candles, "close", params);
        let result = apo(&input).expect("Failed to calculate APO");
        if result.values.len() > 30 {
            for i in 30..result.values.len() {
                assert!(!result.values[i].is_nan());
            }
        }
    }
}
