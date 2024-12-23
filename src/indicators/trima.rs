use std::error::Error;

#[derive(Debug, Clone)]
pub struct TrimaParams {
    pub period: Option<usize>,
}

impl Default for TrimaParams {
    fn default() -> Self {
        TrimaParams { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct TrimaInput<'a> {
    pub data: &'a [f64],
    pub params: TrimaParams,
}

impl<'a> TrimaInput<'a> {
    pub fn new(data: &'a [f64], params: TrimaParams) -> Self {
        TrimaInput { data, params }
    }

    pub fn with_default_params(data: &'a [f64]) -> Self {
        TrimaInput {
            data,
            params: TrimaParams::default(),
        }
    }

    fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| TrimaParams::default().period.unwrap())
    }
}

#[derive(Debug, Clone)]
pub struct TrimaOutput {
    pub values: Vec<f64>,
}

#[inline]
pub fn calculate_trima(input: &TrimaInput) -> Result<TrimaOutput, Box<dyn Error>> {
    let data = input.data;
    let period = input.get_period();
    let n = data.len();

    if period > n {
        return Err("Not enough data points to calculate TRIMA.".into());
    }

    if period <= 3 {
        return Err("TRIMA period must be greater than 3.".into());
    }

    let mut out = Vec::with_capacity(n);

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

        out.push(weight_sum * inv_weights);

        lead_sum += val;

        weight_sum += lead_sum;
        weight_sum -= trail_sum;

        lead_sum -= data[lsi as usize];
        trail_sum += data[tsi1 as usize];
        trail_sum -= data[tsi2 as usize];

        lsi += 1;
        tsi1 += 1;
        tsi2 += 1;
    }

    Ok(TrimaOutput { values: out })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::data_loader::read_candles_from_csv;

    #[test]
    fn test_trima_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let params = TrimaParams { period: Some(30) };
        let input = TrimaInput::new(close_prices, params);
        let trima_result = calculate_trima(&input).expect("Failed to calculate TRIMA");

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

        let default_input = TrimaInput::with_default_params(close_prices);
        let default_trima_result =
            calculate_trima(&default_input).expect("Failed to calculate TRIMA with defaults");
        assert!(
            !default_trima_result.values.is_empty(),
            "Should produce some TRIMA values with default params"
        );
    }
}
