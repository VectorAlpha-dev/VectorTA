use std::error::Error;

#[derive(Debug, Clone)]
pub struct RsiParams {
    pub period: Option<usize>,
}

impl Default for RsiParams {
    fn default() -> Self {
        RsiParams { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct RsiInput<'a> {
    pub data: &'a [f64],
    pub params: RsiParams,
}

impl<'a> RsiInput<'a> {
    pub fn new(data: &'a [f64], params: RsiParams) -> Self {
        RsiInput { data, params }
    }

    pub fn with_default_params(data: &'a [f64]) -> Self {
        RsiInput {
            data,
            params: RsiParams::default(),
        }
    }

    fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| RsiParams::default().period.unwrap())
    }
}

#[derive(Debug, Clone)]
pub struct RsiOutput {
    pub values: Vec<f64>,
}

#[inline]
pub fn rsi(input: &RsiInput) -> Result<RsiOutput, Box<dyn Error>> {
    let data = input.data;
    let period = input.get_period();

    if period == 0 || period > data.len() {
        return Err("Invalid period specified for RSI calculation.".into());
    }

    let len = data.len();
    let mut rsi = Vec::with_capacity(len);

    rsi.extend(std::iter::repeat(f64::NAN).take(period));

    let inv_period = 1.0 / period as f64;
    let beta = 1.0 - inv_period;

    let mut avg_gain = 0.0;
    let mut avg_loss = 0.0;

    for i in 1..=period {
        let delta = data[i] - data[i - 1];
        if delta > 0.0 {
            avg_gain += delta;
        } else {
            avg_loss += -delta;
        }
    }

    avg_gain *= inv_period;
    avg_loss *= inv_period;

    let initial_rsi = if avg_gain + avg_loss == 0.0 {
        50.0
    } else {
        100.0 * avg_gain / (avg_gain + avg_loss)
    };
    rsi.push(initial_rsi);

    for i in (period + 1)..len {
        let delta = data[i] - data[i - 1];
        let gain = if delta > 0.0 { delta } else { 0.0 };
        let loss = if delta < 0.0 { -delta } else { 0.0 };

        avg_gain = inv_period * gain + beta * avg_gain;
        avg_loss = inv_period * loss + beta * avg_loss;

        let current_rsi = if avg_gain + avg_loss == 0.0 {
            50.0
        } else {
            100.0 * avg_gain / (avg_gain + avg_loss)
        };

        rsi.push(current_rsi);
    }

    Ok(RsiOutput { values: rsi })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_rsi_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let params = RsiParams { period: Some(14) };
        let input = RsiInput::new(close_prices, params);
        let rsi_result = rsi(&input).expect("Failed to calculate RSI");

        let expected_last_five_rsi = [43.42, 42.68, 41.62, 42.86, 39.01];

        assert!(
            rsi_result.values.len() >= 5,
            "Not enough RSI values for the test"
        );

        assert_eq!(
            rsi_result.values.len(),
            close_prices.len(),
            "RSI values count should match input data count"
        );

        let start_index = rsi_result.values.len().saturating_sub(5);
        let result_last_five_rsi = &rsi_result.values[start_index..];

        for (i, &value) in result_last_five_rsi.iter().enumerate() {
            assert!(
                (value - expected_last_five_rsi[i]).abs() < 1e-2,
                "RSI value mismatch at index {}: expected {}, got {}",
                i,
                expected_last_five_rsi[i],
                value
            );
        }

        let default_input = RsiInput::with_default_params(close_prices);
        let default_rsi_result =
            rsi(&default_input).expect("Failed to calculate RSI with defaults");
        assert!(
            !default_rsi_result.values.is_empty(),
            "Should produce RSI values with default params"
        );
    }
}
