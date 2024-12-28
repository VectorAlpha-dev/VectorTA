use crate::utilities::data_loader::{source_type, Candles};
use std::error::Error;

#[derive(Debug, Clone)]
pub enum TrimaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct TrimaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct TrimaParams {
    pub period: Option<usize>,
}

impl TrimaParams {
    pub fn with_default_params() -> Self {
        Self { period: None }
    }
}

#[derive(Debug, Clone)]
pub struct TrimaInput<'a> {
    pub data: TrimaData<'a>,
    pub params: TrimaParams,
}

impl<'a> TrimaInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: TrimaParams) -> Self {
        Self {
            data: TrimaData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: TrimaParams) -> Self {
        Self {
            data: TrimaData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: TrimaData::Candles {
                candles,
                source: "close",
            },
            params: TrimaParams::with_default_params(),
        }
    }
}

pub fn trima(input: &TrimaInput) -> Result<TrimaOutput, Box<dyn Error>> {
    let data: &[f64] = match &input.data {
        TrimaData::Candles { candles, source } => source_type(candles, source),
        TrimaData::Slice(slice) => slice,
    };
    let n: usize = data.len();
    let period: usize = input.params.period.unwrap_or(14);

    if period > n {
        return Err("Not enough data points to calculate TRIMA.".into());
    }
    if period <= 3 {
        return Err("TRIMA period must be greater than 3.".into());
    }

    let mut out = vec![f64::NAN; n];

    let sum_of_weights = if period % 2 == 1 {
        let half = period / 2 + 1;
        (half * half) as f64
    } else {
        let half_up = period / 2 + 1;
        let half_down = period / 2;
        (half_up * half_down) as f64
    };
    let inv_weights = 1.0 / sum_of_weights;

    let lead_period = if period % 2 == 1 {
        period / 2
    } else {
        (period / 2) - 1
    };
    let trail_period = lead_period + 1;

    let mut weight_sum = 0.0;
    let mut lead_sum = 0.0;
    let mut trail_sum = 0.0;
    let mut w = 1;

    for i in 0..(period - 1) {
        let val = data[i];
        weight_sum += val * (w as f64);

        if i + 1 > period - lead_period {
            lead_sum += val;
        }
        if i < trail_period {
            trail_sum += val;
        }

        if i + 1 < trail_period {
            w += 1;
        }
        if i + 1 >= (period - lead_period) {
            w -= 1;
        }
    }

    let mut lsi = (period - 1) as isize - lead_period as isize + 1;
    let mut tsi1 = (period - 1) as isize - period as isize + 1 + trail_period as isize;
    let mut tsi2 = (period - 1) as isize - period as isize + 1;

    for i in (period - 1)..n {
        let val = data[i];

        weight_sum += val;

        out[i] = weight_sum * inv_weights;

        lead_sum += val;
        weight_sum += lead_sum;
        weight_sum -= trail_sum;

        let lsi_idx = lsi as usize;
        let tsi1_idx = tsi1 as usize;
        let tsi2_idx = tsi2 as usize;

        lead_sum -= data[lsi_idx];
        trail_sum += data[tsi1_idx];
        trail_sum -= data[tsi2_idx];

        lsi += 1;
        tsi1 += 1;
        tsi2 += 1;
    }

    Ok(TrimaOutput { values: out })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_trima_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let default_params = TrimaParams { period: None };
        let input = TrimaInput::from_candles(&candles, "close", default_params);
        let output = trima(&input).expect("Failed TRIMA with default params");
        assert_eq!(output.values.len(), candles.close.len());
        let params_period_10 = TrimaParams { period: Some(10) };
        let input2 = TrimaInput::from_candles(&candles, "hl2", params_period_10);
        let output2 = trima(&input2).expect("Failed TRIMA with period=10, source=hl2");
        assert_eq!(output2.values.len(), candles.close.len());
        let params_custom = TrimaParams { period: Some(30) };
        let input3 = TrimaInput::from_candles(&candles, "hlc3", params_custom);
        let output3 = trima(&input3).expect("Failed TRIMA fully custom");
        assert_eq!(output3.values.len(), candles.close.len());
    }

    #[test]
    fn test_trima_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");
        let params = TrimaParams { period: Some(30) };
        let input = TrimaInput::from_candles(&candles, "close", params);
        let trima_result = trima(&input).expect("Failed to calculate TRIMA");
        assert_eq!(
            trima_result.values.len(),
            close_prices.len(),
            "TRIMA output length should match input data length"
        );
        let expected_last_five_trima = [
            59957.916666666664,
            59846.770833333336,
            59750.620833333334,
            59665.2125,
            59581.612499999996,
        ];
        assert!(
            trima_result.values.len() >= 5,
            "Not enough TRIMA values for the test"
        );
        let start_index = trima_result.values.len() - 5;
        let result_last_five_trima = &trima_result.values[start_index..];
        for (i, &value) in result_last_five_trima.iter().enumerate() {
            let expected_value = expected_last_five_trima[i];
            assert!(
                (value - expected_value).abs() < 1e-6,
                "TRIMA value mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }
        let period = input.params.period.unwrap_or(14);
        for i in 0..(period - 1) {
            assert!(
                trima_result.values[i].is_nan(),
                "Expected NaN at early index {} for TRIMA, got {}",
                i,
                trima_result.values[i]
            );
        }
        let default_input = TrimaInput::with_default_candles(&candles);
        let default_trima_result =
            trima(&default_input).expect("Failed to calculate TRIMA with defaults");
        assert!(
            !default_trima_result.values.is_empty(),
            "Should produce some TRIMA values with default params"
        );
    }
}
