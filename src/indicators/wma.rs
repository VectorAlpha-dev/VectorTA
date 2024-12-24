use std::error::Error;

#[derive(Debug, Clone)]
pub struct WmaParams {
    pub period: Option<usize>,
}

impl Default for WmaParams {
    fn default() -> Self {
        WmaParams { period: Some(30) }
    }
}

#[derive(Debug, Clone)]
pub struct WmaInput<'a> {
    pub data: &'a [f64],
    pub params: WmaParams,
}

impl<'a> WmaInput<'a> {
    pub fn new(data: &'a [f64], params: WmaParams) -> Self {
        WmaInput { data, params }
    }

    pub fn with_default_params(data: &'a [f64]) -> Self {
        WmaInput {
            data,
            params: WmaParams::default(),
        }
    }

    fn get_period(&self) -> usize {
        self.params.period.unwrap_or(30)
    }
}

#[derive(Debug, Clone)]
pub struct WmaOutput {
    pub values: Vec<f64>,
}

pub fn calculate_wma(input: &WmaInput) -> Result<WmaOutput, Box<dyn Error>> {
    let data = input.data;
    let len = data.len();
    let period = input.get_period();
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

    for i in 0..lookback {
        let val = data[i];
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
    use crate::indicators::data_loader::read_candles_from_csv;

    #[test]
    fn test_wma_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let data = &candles.close;

        let input = WmaInput::with_default_params(data);
        let result = calculate_wma(&input).expect("Failed to calculate WMA");

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

        let period = input.get_period();
        for val in result.values.iter().skip(period - 1) {
            if !val.is_nan() {
                assert!(val.is_finite(), "WMA output should be finite");
            }
        }
    }
}
