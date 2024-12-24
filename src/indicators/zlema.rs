#[derive(Debug, Clone)]
pub struct ZlemaParams {
    pub period: Option<usize>,
}

impl Default for ZlemaParams {
    fn default() -> Self {
        ZlemaParams { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct ZlemaInput<'a> {
    pub data: &'a [f64],
    pub params: ZlemaParams,
}

impl<'a> ZlemaInput<'a> {
    pub fn new(data: &'a [f64], params: ZlemaParams) -> Self {
        ZlemaInput { data, params }
    }

    pub fn with_default_params(data: &'a [f64]) -> Self {
        ZlemaInput {
            data,
            params: ZlemaParams::default(),
        }
    }

    fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| ZlemaParams::default().period.unwrap())
    }
}

#[derive(Debug, Clone)]
pub struct ZlemaOutput {
    pub values: Vec<f64>,
}
pub fn calculate_zlema(input: &ZlemaInput) -> Result<ZlemaOutput, Box<dyn std::error::Error>> {
    let data = input.data;
    let period = input.get_period();

    if period == 0 || period > data.len() {
        return Err("Invalid period specified for ZLEMA calculation.".into());
    }

    let len = data.len();
    let lag = (period - 1) / 2;
    let alpha = 2.0 / (period as f64 + 1.0);

    let mut zlema_values = Vec::with_capacity(len);

    let mut last_ema = data[0];
    zlema_values.push(last_ema);

    for i in 1..len {
        let val = if i < lag {
            data[i]
        } else {
            2.0 * data[i] - data[i - lag]
        };

        last_ema = alpha * val + (1.0 - alpha) * last_ema;
        zlema_values.push(last_ema);
    }

    Ok(ZlemaOutput {
        values: zlema_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::data_loader::read_candles_from_csv;

    #[test]
    fn test_zlema_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");
        let params = ZlemaParams { period: Some(14) };
        let input = ZlemaInput::new(close_prices, params);

        let result = calculate_zlema(&input).expect("Failed to calculate ZLEMA");
        let expected_last_five = [59015.1, 59165.2, 59168.1, 59147.0, 58978.9];

        assert!(
            result.values.len() >= 5,
            "Not enough EMA values for the test"
        );
        assert_eq!(
            result.values.len(),
            close_prices.len(),
            "ZLEMA values count should match the input data length"
        );
        let start_index = result.values.len().saturating_sub(5);
        let result_last_five = &result.values[start_index..];

        for (i, &value) in result_last_five.iter().enumerate() {
            assert!(
                (value - expected_last_five[i]).abs() < 1e-1,
                "EMA value mismatch at index {}: expected {}, got {}",
                i,
                expected_last_five[i],
                value
            );
        }

        for val in result.values.iter() {
            assert!(val.is_finite(), "ZLEMA output should be finite");
        }
    }
}
