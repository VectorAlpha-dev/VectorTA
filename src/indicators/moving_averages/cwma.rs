use crate::utilities::data_loader::{source_type, Candles};
use std::error::Error;

#[derive(Debug, Clone)]
pub enum CwmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

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
    pub data: CwmaData<'a>,
    pub params: CwmaParams,
}

impl<'a> CwmaInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: CwmaParams) -> Self {
        Self {
            data: CwmaData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: CwmaParams) -> Self {
        Self {
            data: CwmaData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: CwmaData::Candles {
                candles,
                source: "close",
            },
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
    let data: &[f64] = match &input.data {
        CwmaData::Candles { candles, source } => source_type(candles, source),
        CwmaData::Slice(slice) => slice,
    };
    let len: usize = data.len();
    let period: usize = input.get_period();

    if data.is_empty() {
        return Ok(CwmaOutput { values: vec![] });
    }
    if period == 0 {
        return Err("CWMA period must be >= 1.".into());
    }

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
    fn test_cwma_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).unwrap();
        let default_params = CwmaParams { period: None };
        let input_default = CwmaInput::from_candles(&candles, "close", default_params);
        let output_default = cwma(&input_default).unwrap();
        assert_eq!(output_default.values.len(), candles.close.len());

        let params_period_14 = CwmaParams { period: Some(14) };
        let input_period_14 = CwmaInput::from_candles(&candles, "hl2", params_period_14);
        let output_period_14 = cwma(&input_period_14).unwrap();
        assert_eq!(output_period_14.values.len(), candles.close.len());

        let params_custom = CwmaParams { period: Some(20) };
        let input_custom = CwmaInput::from_candles(&candles, "hlc3", params_custom);
        let output_custom = cwma(&input_custom).unwrap();
        assert_eq!(output_custom.values.len(), candles.close.len());
    }

    #[test]
    fn test_cwma_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = CwmaInput::with_default_candles(&candles);

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
    #[test]
    fn test_cwma_params_with_default_params() {
        let default_params = CwmaParams::default();
        assert_eq!(default_params.period, Some(14));
    }

    #[test]
    fn test_cwma_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).unwrap();
        let input = CwmaInput::with_default_candles(&candles);
        match input.data {
            CwmaData::Candles { source, .. } => {
                assert_eq!(source, "close");
            }
            _ => panic!("Unexpected data variant"),
        }
        assert_eq!(input.params.period, Some(14));
    }

    #[test]
    fn test_cwma_with_zero_period() {
        let input_data = [10.0, 20.0, 30.0];
        let params = CwmaParams { period: Some(0) };
        let input = CwmaInput::from_slice(&input_data, params);
        let result = cwma(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("CWMA period must be >= 1."));
        }
    }

    #[test]
    fn test_cwma_with_period_exceeding_data_length() {
        let input_data = [10.0, 20.0, 30.0];
        let params = CwmaParams { period: Some(10) };
        let input = CwmaInput::from_slice(&input_data, params);
        let result = cwma(&input).unwrap();
        assert_eq!(result.values, input_data);
    }

    #[test]
    fn test_cwma_very_small_data_set() {
        let input_data = [42.0];
        let params = CwmaParams { period: Some(9) };
        let input = CwmaInput::from_slice(&input_data, params);
        let result = cwma(&input).unwrap();
        assert_eq!(result.values, input_data);
    }

    #[test]
    fn test_cwma_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).unwrap();
        let first_params = CwmaParams { period: Some(9) };
        let first_input = CwmaInput::from_candles(&candles, "close", first_params);
        let first_result = cwma(&first_input).unwrap();
        assert_eq!(first_result.values.len(), candles.close.len());
        let second_params = CwmaParams { period: Some(5) };
        let second_input = CwmaInput::from_slice(&first_result.values, second_params);
        let second_result = cwma(&second_input).unwrap();
        assert_eq!(second_result.values.len(), first_result.values.len());
    }

    #[test]
    fn test_cwma_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).unwrap();
        let close_prices = &candles.close;
        let params = CwmaParams { period: Some(9) };
        let input = CwmaInput::from_candles(&candles, "close", params);
        let cwma_result = cwma(&input).unwrap();
        assert_eq!(cwma_result.values.len(), close_prices.len());
        if cwma_result.values.len() > 240 {
            for i in 240..cwma_result.values.len() {
                assert!(!cwma_result.values[i].is_nan());
            }
        }
    }
}
