use crate::utilities::data_loader::{source_type, Candles};
use std::error::Error;

#[derive(Debug, Clone)]
pub struct CwmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct CwmaParams {
    pub period: Option<usize>,
}

impl Default for CwmaParams {
    fn default() -> Self {
        CwmaParams { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct CwmaInput<'a> {
    pub candles: &'a Candles,
    pub source: &'a str,
    pub params: CwmaParams,
}

impl<'a> CwmaInput<'a> {
    pub fn new(candles: &'a Candles, source: &'a str, params: CwmaParams) -> Self {
        CwmaInput {
            candles,
            source,
            params,
        }
    }

    pub fn with_default_params(candles: &'a Candles) -> Self {
        CwmaInput {
            candles,
            source: "close",
            params: CwmaParams::default(),
        }
    }

    #[inline]
    fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| CwmaParams::default().period.unwrap())
    }
}

#[inline]
pub fn cwma(input: &CwmaInput) -> Result<CwmaOutput, Box<dyn Error>> {
    let data: &[f64] = source_type(input.candles, input.source);
    let len: usize = data.len();
    let period: usize = input.get_period();

    if data.is_empty() {
        return Ok(CwmaOutput { values: vec![] });
    }
    if period == 0 {
        return Err("CWMA period must be >= 1.".into());
    }

    let len = data.len();
    if period + 1 > len {
        return Ok(CwmaOutput {
            values: data.to_vec(),
        });
    }

    let p_minus_1 = period - 1;
    let mut weights = Vec::with_capacity(p_minus_1);
    for i in 0..p_minus_1 {
        let w = ((period - i) as f64).powi(3);
        weights.push(w);
    }
    let sum_of_weights: f64 = weights.iter().sum();

    let mut cwma_values = data.to_vec();

    for j in (period + 1)..len {
        let mut my_sum = 0.0;
        for (i, &w) in weights.iter().enumerate() {
            my_sum += data[j - i] * w;
        }
        cwma_values[j] = my_sum / sum_of_weights;
    }

    Ok(CwmaOutput {
        values: cwma_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_cwma_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = CwmaInput::with_default_params(&candles);

        let cwma_result = cwma(&input).expect("CWMA calculation failed");
        let cwma_values = &cwma_result.values;

        assert_eq!(cwma_values.len(), candles.close.len(), "Length mismatch");

        let expected_last_five = [
            59224.641237300435,
            59213.64831277214,
            59171.21190130624,
            59167.01279027576,
            59039.413552249636,
        ];

        assert!(
            cwma_values.len() >= expected_last_five.len(),
            "Not enough CWMA values for the test"
        );

        let start_index = cwma_values.len() - expected_last_five.len();
        let actual_last_five = &cwma_values[start_index..];

        for (i, (&actual, &expected)) in actual_last_five
            .iter()
            .zip(expected_last_five.iter())
            .enumerate()
        {
            let diff = (actual - expected).abs();
            assert!(
                diff < 1e-8,
                "CWMA mismatch at index {}: expected {:.14}, got {:.14}",
                start_index + i,
                expected,
                actual
            );
        }

        let period = input.get_period();
        for i in 0..=period {
            let orig_val = candles.close[i];
            let cwma_val = cwma_values[i];
            assert!(
                (orig_val - cwma_val).abs() < f64::EPSILON,
                "Expected CWMA to remain the same as original for index {}",
                i
            );
        }
    }
}