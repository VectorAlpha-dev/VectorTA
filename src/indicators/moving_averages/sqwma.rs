use crate::utilities::data_loader::{source_type, Candles};
use std::error::Error;

#[derive(Debug, Clone)]
pub enum SqwmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct SqwmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct SqwmaParams {
    pub period: Option<usize>,
}

impl SqwmaParams {
    pub fn with_default_params() -> Self {
        Self { period: None }
    }
}

#[derive(Debug, Clone)]
pub struct SqwmaInput<'a> {
    pub data: SqwmaData<'a>,
    pub params: SqwmaParams,
}

impl<'a> SqwmaInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: SqwmaParams) -> Self {
        Self {
            data: SqwmaData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: SqwmaParams) -> Self {
        Self {
            data: SqwmaData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: SqwmaData::Candles {
                candles,
                source: "close",
            },
            params: SqwmaParams::with_default_params(),
        }
    }
}

#[inline]
pub fn sqwma(input: &SqwmaInput) -> Result<SqwmaOutput, Box<dyn Error>> {
    let data: &[f64] = match &input.data {
        SqwmaData::Candles { candles, source } => source_type(candles, source),
        SqwmaData::Slice(slice) => slice,
    };
    let n: usize = data.len();
    let period: usize = input.params.period.unwrap_or(14);
    if n == 0 {
        return Err("Empty data for SQWMA calculation.".into());
    }

    if period < 2 {
        return Err("SQWMA period must be >= 2.".into());
    }

    if period + 1 > n {
        return Ok(SqwmaOutput {
            values: data.to_vec(),
        });
    }

    let mut weights = Vec::with_capacity(period - 1);
    for i in 0..(period - 1) {
        let w = (period as f64 - i as f64).powi(2);
        weights.push(w);
    }

    let weight_sum: f64 = weights.iter().sum();

    let mut output = data.to_vec();

    #[inline(always)]
    fn sqwma_sum(data: &[f64], j: usize, weights: &[f64]) -> f64 {
        let mut sum_ = 0.0;
        let p_minus_1 = weights.len();

        let mut i = 0;
        while i < p_minus_1.saturating_sub(3) {
            sum_ += data[j - i] * weights[i];
            sum_ += data[j - (i + 1)] * weights[i + 1];
            sum_ += data[j - (i + 2)] * weights[i + 2];
            sum_ += data[j - (i + 3)] * weights[i + 3];
            i += 4;
        }
        while i < p_minus_1 {
            sum_ += data[j - i] * weights[i];
            i += 1;
        }
        sum_
    }

    for j in (period + 1)..n {
        let my_sum = sqwma_sum(data, j, &weights);
        output[j] = my_sum / weight_sum;
    }

    Ok(SqwmaOutput { values: output })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_sqwma_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let default_params = SqwmaParams { period: None };
        let input = SqwmaInput::from_candles(&candles, "close", default_params);
        let output = sqwma(&input).expect("Failed SQWMA with default params");
        assert_eq!(output.values.len(), candles.close.len());
        let params_period_10 = SqwmaParams { period: Some(10) };
        let input2 = SqwmaInput::from_candles(&candles, "hl2", params_period_10);
        let output2 = sqwma(&input2).expect("Failed SQWMA with period=10, source=hl2");
        assert_eq!(output2.values.len(), candles.close.len());
        let params_custom = SqwmaParams { period: Some(20) };
        let input3 = SqwmaInput::from_candles(&candles, "hlc3", params_custom);
        let output3 = sqwma(&input3).expect("Failed SQWMA fully custom");
        assert_eq!(output3.values.len(), candles.close.len());
    }

    #[test]
    fn test_sqwma_accuracy() {
        let expected_last_five = [
            59229.72287968442,
            59211.30867850099,
            59172.516765286,
            59167.73471400394,
            59067.97928994083,
        ];
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let source = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");
        let default_params = SqwmaParams::with_default_params();
        let input = SqwmaInput::from_candles(&candles, "close", default_params);
        let result = sqwma(&input).expect("Failed to calculate SQWMA");
        assert_eq!(result.values.len(), source.len());
        assert!(result.values.len() >= 5);
        let start_idx = result.values.len() - 5;
        let actual_last_five = &result.values[start_idx..];
        for (i, &val) in actual_last_five.iter().enumerate() {
            let exp_val = expected_last_five[i];
            assert!(
                (val - exp_val).abs() < 1e-5,
                "SQWMA mismatch at index {}, expected {}, got {}",
                i,
                exp_val,
                val
            );
        }
        let default_input = SqwmaInput::with_default_candles(&candles);
        let default_result = sqwma(&default_input).expect("Failed default SQWMA");
        assert_eq!(default_result.values.len(), source.len());
    }
    #[test]
    fn test_sqwma_params_with_default_params() {
        let default_params = SqwmaParams::with_default_params();
        assert_eq!(default_params.period, None);
    }

    #[test]
    fn test_sqwma_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).unwrap();
        let input = SqwmaInput::with_default_candles(&candles);
        match input.data {
            SqwmaData::Candles { source, .. } => {
                assert_eq!(source, "close");
            }
            _ => panic!("Unexpected data variant"),
        }
        assert_eq!(input.params.period, None);
    }

    #[test]
    fn test_sqwma_with_empty_data() {
        let input_data: [f64; 0] = [];
        let params = SqwmaParams { period: Some(14) };
        let input = SqwmaInput::from_slice(&input_data, params);
        let result = sqwma(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Empty data for SQWMA calculation"));
        }
    }

    #[test]
    fn test_sqwma_with_zero_period() {
        let input_data = [10.0, 20.0, 30.0];
        let params = SqwmaParams { period: Some(0) };
        let input = SqwmaInput::from_slice(&input_data, params);
        let result = sqwma(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("SQWMA period must be >= 2"));
        }
    }

    #[test]
    fn test_sqwma_with_period_exceeding_data_length() {
        let input_data = [10.0, 20.0, 30.0];
        let params = SqwmaParams { period: Some(14) };
        let input = SqwmaInput::from_slice(&input_data, params);
        let result = sqwma(&input).unwrap();
        assert_eq!(result.values, input_data);
    }

    #[test]
    fn test_sqwma_very_small_data_set() {
        let input_data = [42.0, 43.0, 44.0];
        let params = SqwmaParams { period: Some(3) };
        let input = SqwmaInput::from_slice(&input_data, params);
        let result = sqwma(&input).unwrap();
        assert_eq!(result.values, input_data);
    }

    #[test]
    fn test_sqwma_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).unwrap();
        let first_params = SqwmaParams { period: Some(14) };
        let first_input = SqwmaInput::from_candles(&candles, "close", first_params);
        let first_result = sqwma(&first_input).unwrap();
        assert_eq!(first_result.values.len(), candles.close.len());

        let second_params = SqwmaParams { period: Some(7) };
        let second_input = SqwmaInput::from_slice(&first_result.values, second_params);
        let second_result = sqwma(&second_input).unwrap();
        assert_eq!(second_result.values.len(), first_result.values.len());
    }

    #[test]
    fn test_sqwma_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).unwrap();
        let params = SqwmaParams { period: Some(14) };
        let input = SqwmaInput::from_candles(&candles, "close", params);
        let sqwma_result = sqwma(&input).unwrap();
        for &val in &sqwma_result.values {
            assert!(!val.is_nan());
        }
    }
}
