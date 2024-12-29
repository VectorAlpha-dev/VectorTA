use crate::utilities::data_loader::{source_type, Candles};
use std::error::Error;

#[derive(Debug, Clone)]
pub enum EdcfData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct EdcfParams {
    pub period: Option<usize>,
}

impl Default for EdcfParams {
    fn default() -> Self {
        EdcfParams { period: Some(15) }
    }
}

#[derive(Debug, Clone)]
pub struct EdcfOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct EdcfInput<'a> {
    pub data: EdcfData<'a>,
    pub params: EdcfParams,
}

impl<'a> EdcfInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: EdcfParams) -> Self {
        Self {
            data: EdcfData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: EdcfParams) -> Self {
        Self {
            data: EdcfData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: EdcfData::Candles {
                candles,
                source: "close",
            },
            params: EdcfParams::default(),
        }
    }

    #[inline]
    fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| EdcfParams::default().period.unwrap())
    }
}

#[inline]
pub fn edcf(input: &EdcfInput) -> Result<EdcfOutput, Box<dyn Error>> {
    let data: &[f64] = match &input.data {
        EdcfData::Candles { candles, source } => source_type(candles, source),
        EdcfData::Slice(slice) => slice,
    };
    let period: usize = input.get_period();
    let len: usize = data.len();

    if data.is_empty() {
        return Err("No data provided to EDCF filter.".into());
    }
    if period == 0 {
        return Err("EDCF period must be >= 1.".into());
    }

    let mut newseries = vec![f64::NAN; len];

    let mut dist = vec![0.0; len];

    for k in period..len {
        let xk = data[k];
        let mut sum_sq = 0.0;
        for lb in 1..period {
            let diff = xk - data[k - lb];
            sum_sq += diff * diff;
        }
        dist[k] = sum_sq;
    }

    let start_j = 2 * period;
    for j in start_j..len {
        let mut num = 0.0;
        let mut coef_sum = 0.0;

        for i in 0..period {
            let k = j - i;
            let distance = dist[k];
            let base_val = data[k];
            num += distance * base_val;
            coef_sum += distance;
        }

        if coef_sum != 0.0 {
            newseries[j] = num / coef_sum;
        } else {
            newseries[j] = 0.0;
        }
    }

    Ok(EdcfOutput { values: newseries })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_edcf_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = EdcfInput::from_candles(&candles, "hl2", EdcfParams { period: Some(15) });

        let edcf_result = edcf(&input).expect("EDCF calculation failed");
        let edcf_values = &edcf_result.values;

        assert_eq!(
            edcf_values.len(),
            candles.close.len(),
            "EDCF output length does not match input length!"
        );

        let expected_last_five = [
            59593.332275678375,
            59731.70263288801,
            59766.41512339413,
            59655.66162110993,
            59332.492883847,
        ];

        assert!(
            edcf_values.len() >= expected_last_five.len(),
            "Not enough EDCF values for the test"
        );

        let start_index = edcf_values.len() - expected_last_five.len();
        let actual_last_five = &edcf_values[start_index..];

        for (i, (&actual, &expected)) in actual_last_five
            .iter()
            .zip(expected_last_five.iter())
            .enumerate()
        {
            let diff = (actual - expected).abs();
            assert!(
                diff < 1e-8,
                "EDCF mismatch at index {}: expected {:.14}, got {:.14}",
                start_index + i,
                expected,
                actual
            );
        }
    }
    #[test]
    fn test_edcf_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = EdcfInput::with_default_candles(&candles);
        match input.data {
            EdcfData::Candles { source, .. } => {
                assert_eq!(source, "close");
            }
            _ => panic!("Expected EdcfData::Candles"),
        }
        let period = input.get_period();
        assert_eq!(period, 15);
    }

    #[test]
    fn test_edcf_with_default_params() {
        let default_params = EdcfParams::default();
        assert_eq!(default_params.period, Some(15));
    }

    #[test]
    fn test_edcf_with_zero_period() {
        let data = [10.0, 20.0, 30.0];
        let input = EdcfInput::from_slice(&data, EdcfParams { period: Some(0) });
        let result = edcf(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("period must be >= 1"));
        }
    }

    #[test]
    fn test_edcf_with_no_data() {
        let data: [f64; 0] = [];
        let input = EdcfInput::from_slice(&data, EdcfParams { period: Some(15) });
        let result = edcf(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("No data provided"));
        }
    }

    #[test]
    fn test_edcf_with_period_exceeding_data_length() {
        let data = [10.0, 20.0, 30.0];
        let input = EdcfInput::from_slice(&data, EdcfParams { period: Some(10) });
        let result = edcf(&input).unwrap();
        assert_eq!(result.values.len(), data.len());
        for i in 0..result.values.len() {
            if i < 2 * 10 {
                assert!(result.values[i].is_nan());
            }
        }
    }

    #[test]
    fn test_edcf_very_small_data_set() {
        let data = [42.0];
        let input = EdcfInput::from_slice(&data, EdcfParams { period: Some(15) });
        let result = edcf(&input).unwrap();
        assert_eq!(result.values.len(), data.len());
        assert!(result.values[0].is_nan());
    }

    #[test]
    fn test_edcf_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let first_input =
            EdcfInput::from_candles(&candles, "close", EdcfParams { period: Some(15) });
        let first_result = edcf(&first_input).expect("First EDCF failed");
        let second_input =
            EdcfInput::from_slice(&first_result.values, EdcfParams { period: Some(5) });
        let second_result = edcf(&second_input).expect("Second EDCF failed");
        assert_eq!(second_result.values.len(), first_result.values.len());
    }

    #[test]
    fn test_edcf_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = EdcfInput::from_candles(&candles, "close", EdcfParams { period: None });
        let result = edcf(&input).expect("EDCF calculation failed");
        assert_eq!(result.values.len(), candles.close.len());
    }

    #[test]
    fn test_edcf_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = EdcfInput::from_candles(&candles, "close", EdcfParams { period: Some(15) });
        let result = edcf(&input).expect("EDCF calculation failed");
        assert_eq!(result.values.len(), candles.close.len());
        let start_index = 2 * 15;
        if result.values.len() > start_index {
            for i in start_index..result.values.len() {
                assert!(!result.values[i].is_nan());
            }
        }
    }
}
