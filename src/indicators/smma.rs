use std::error::Error;

#[derive(Debug, Clone)]
pub struct SmmaParams {
    pub period: Option<usize>,
}

impl Default for SmmaParams {
    fn default() -> Self {
        SmmaParams { period: Some(7) }
    }
}

#[derive(Debug, Clone)]
pub struct SmmaInput<'a> {
    pub data: &'a [f64],
    pub params: SmmaParams,
}

impl<'a> SmmaInput<'a> {
    pub fn new(data: &'a [f64], params: SmmaParams) -> Self {
        SmmaInput { data, params }
    }

    pub fn with_default_params(data: &'a [f64]) -> Self {
        SmmaInput {
            data,
            params: SmmaParams::default(),
        }
    }

    fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| SmmaParams::default().period.unwrap())
    }
}

#[derive(Debug, Clone)]
pub struct SmmaOutput {
    pub values: Vec<f64>,
}

#[inline]
pub fn calculate_smma(input: &SmmaInput) -> Result<SmmaOutput, Box<dyn Error>> {
    let data = input.data;
    let len = data.len();
    let period = input.get_period();

    if period == 0 || period > len {
        return Err("Invalid period specified for SMMA calculation.".into());
    }

    let mut smma_values = Vec::with_capacity(len);

    for _ in 0..(period - 1) {
        smma_values.push(f64::NAN);
    }

    let sum_first_period: f64 = data[..period].iter().sum();
    let first_smma = sum_first_period / (period as f64);
    smma_values.push(first_smma);

    let mut prev_smma = first_smma;
    for i in period..len {
        let new_smma = (prev_smma * (period as f64 - 1.0) + data[i]) / (period as f64);
        smma_values.push(new_smma);
        prev_smma = new_smma;
    }

    Ok(SmmaOutput {
        values: smma_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::data_loader::read_candles_from_csv;

    #[test]
    fn test_smma_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let params = SmmaParams { period: Some(7) };
        let input = SmmaInput::new(close_prices, params);

        let result = calculate_smma(&input).expect("Failed to calculate SMMA");

        assert_eq!(
            result.values.len(),
            close_prices.len(),
            "SMMA output length does not match input length!"
        );

        let expected_last_five = [59434.4, 59398.2, 59346.9, 59319.4, 59224.5];

        assert!(
            result.values.len() >= 5,
            "Not enough SMMA values for the test"
        );

        assert_eq!(
            result.values.len(),
            close_prices.len(),
            "SMMA values count should match input data count"
        );
        let start_index = result.values.len() - 5;
        let actual_last_five = &result.values[start_index..];

        for (i, &actual) in actual_last_five.iter().enumerate() {
            let expected = expected_last_five[i];
            let diff = (actual - expected).abs();
            assert!(
                diff < 1e-1,
                "SMMA mismatch at index {}: expected {}, got {}, diff={}",
                i,
                expected,
                actual,
                diff
            );
        }
    }
}
