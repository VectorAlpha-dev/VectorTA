use std::error::Error;

#[derive(Debug, Clone)]
pub struct SrwmaParams {
    pub period: Option<usize>,
}

impl Default for SrwmaParams {
    fn default() -> Self {
        SrwmaParams { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct SrwmaInput<'a> {
    pub data: &'a [f64],
    pub params: SrwmaParams,
}

impl<'a> SrwmaInput<'a> {
    pub fn new(data: &'a [f64], params: SrwmaParams) -> Self {
        SrwmaInput { data, params }
    }

    pub fn with_default_params(data: &'a [f64]) -> Self {
        SrwmaInput {
            data,
            params: SrwmaParams::default(),
        }
    }

    #[inline]
    fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| SrwmaParams::default().period.unwrap())
    }
}

#[derive(Debug, Clone)]
pub struct SrwmaOutput {
    pub values: Vec<f64>,
}

#[inline]
pub fn calculate_srwma(input: &SrwmaInput) -> Result<SrwmaOutput, Box<dyn Error>> {
    let data = input.data;
    let period = input.get_period();

    if data.is_empty() {
        return Ok(SrwmaOutput { values: vec![] });
    }
    if period == 0 {
        return Err("SRWMA period must be >= 1.".into());
    }

    let len = data.len();
    if period + 1 > len {
        return Ok(SrwmaOutput {
            values: data.to_vec(),
        });
    }

    let mut weights = Vec::with_capacity(period - 1);
    for i in 0..(period - 1) {
        weights.push((period as f64 - i as f64).sqrt());
    }
    let sum_of_weights: f64 = weights.iter().sum();

    let mut srwma_values = data.to_vec();

    for j in (period + 1)..len {
        let mut my_sum = 0.0;
        for (i, &w) in weights.iter().enumerate() {
            my_sum += data[j - i] * w;
        }
        srwma_values[j] = my_sum / sum_of_weights;
    }

    Ok(SrwmaOutput {
        values: srwma_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_srwma_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let params = SrwmaParams { period: Some(14) };
        let input = SrwmaInput::new(close_prices, params);

        let srwma_result = calculate_srwma(&input).expect("SRWMA calculation failed");
        let srwma_values = &srwma_result.values;

        assert_eq!(srwma_values.len(), close_prices.len(), "Length mismatch");

        let expected_last_five = [
            59344.28384704595,
            59282.09151629659,
            59192.76580529367,
            59178.04767548977,
            59110.03801260874,
        ];

        assert!(
            srwma_values.len() >= expected_last_five.len(),
            "Not enough SRWMA values for the test"
        );

        let start_index = srwma_values.len() - expected_last_five.len();
        let actual_last_five = &srwma_values[start_index..];

        for (i, (&actual, &expected)) in actual_last_five
            .iter()
            .zip(expected_last_five.iter())
            .enumerate()
        {
            let diff = (actual - expected).abs();
            assert!(
                diff < 1e-8,
                "SRWMA mismatch at index {}: expected {:.14}, got {:.14}",
                i,
                expected,
                actual
            );
        }
    }
}
