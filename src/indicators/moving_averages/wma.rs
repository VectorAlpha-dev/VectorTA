use crate::utilities::data_loader::{source_type, Candles};
use std::error::Error;

#[derive(Debug, Clone)]
pub enum WmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct WmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct WmaParams {
    pub period: Option<usize>,
}

impl WmaParams {
    pub fn with_default_params() -> Self {
        Self { period: None }
    }
}

#[derive(Debug, Clone)]
pub struct WmaInput<'a> {
    pub data: WmaData<'a>,
    pub params: WmaParams,
}

impl<'a> WmaInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: WmaParams) -> Self {
        Self {
            data: WmaData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: WmaParams) -> Self {
        Self {
            data: WmaData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: WmaData::Candles {
                candles,
                source: "close",
            },
            params: WmaParams::with_default_params(),
        }
    }
}

pub fn wma(input: &WmaInput) -> Result<WmaOutput, Box<dyn Error>> {
    let data: &[f64] = match &input.data {
        WmaData::Candles { candles, source } => source_type(candles, source),
        WmaData::Slice(slice) => slice,
    };
    let len: usize = data.len();
    let period: usize = input.params.period.unwrap_or(30);
    let mut values = vec![f64::NAN; len];
    if period > len {
        return Err("period is greater than data length".into());
    }
    if period <= 1 {
        return Err("Invalid period for WMA calculation".into());
    }

    let lookback = period - 1;
    let sum_of_weights = (period * (period + 1)) >> 1;
    let divider = sum_of_weights as f64;

    let mut weighted_sum = 0.0;
    let mut plain_sum = 0.0;

    for (i, &val) in data.iter().take(lookback).enumerate() {
        weighted_sum += (i as f64 + 1.0) * val;
        plain_sum += val;
    }

    for i in lookback..len {
        let val = data[i];
        weighted_sum += (period as f64) * val;
        plain_sum += val;
        values[i] = weighted_sum / divider;
        weighted_sum -= plain_sum;
        let old_val = data[i - lookback];
        plain_sum -= old_val;
    }
    Ok(WmaOutput { values })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_wma_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = WmaParams { period: None };
        let input = WmaInput::from_candles(&candles, "close", default_params);
        let output = wma(&input).expect("Failed WMA with default params");
        assert_eq!(output.values.len(), candles.close.len());

        let params_period_14 = WmaParams { period: Some(14) };
        let input2 = WmaInput::from_candles(&candles, "hl2", params_period_14);
        let output2 = wma(&input2).expect("Failed WMA with period=14, source=hl2");
        assert_eq!(output2.values.len(), candles.close.len());

        let params_custom = WmaParams { period: Some(20) };
        let input3 = WmaInput::from_candles(&candles, "hlc3", params_custom);
        let output3 = wma(&input3).expect("Failed WMA fully custom");
        assert_eq!(output3.values.len(), candles.close.len());
    }

    #[test]
    fn test_wma_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let data = &candles.close;
        let default_params = WmaParams::with_default_params();
        let input = WmaInput::from_candles(&candles, "close", default_params);
        let result = wma(&input).expect("Failed to calculate WMA");

        let expected_last_five = [
            59638.52903225806,
            59563.7376344086,
            59489.4064516129,
            59432.02580645162,
            59350.58279569892,
        ];

        assert!(result.values.len() >= 5, "Not enough WMA values");
        assert_eq!(
            result.values.len(),
            data.len(),
            "WMA output length should match input length"
        );

        let start_index = result.values.len().saturating_sub(5);
        let last_five = &result.values[start_index..];

        for (i, &value) in last_five.iter().enumerate() {
            assert!(
                (value - expected_last_five[i]).abs() < 1e-6,
                "WMA value mismatch at index {}: expected {}, got {}",
                i,
                expected_last_five[i],
                value
            );
        }

        let period = input.params.period.unwrap_or(30);
        for val in result.values.iter().skip(period - 1) {
            if !val.is_nan() {
                assert!(val.is_finite(), "WMA output should be finite");
            }
        }
    }
}
