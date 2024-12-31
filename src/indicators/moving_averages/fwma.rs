use crate::utilities::data_loader::{source_type, Candles};
use std::error::Error;

#[derive(Debug, Clone)]
pub enum FwmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct FwmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct FwmaParams {
    pub period: Option<usize>,
}

impl Default for FwmaParams {
    fn default() -> Self {
        Self { period: Some(5) }
    }
}

#[derive(Debug, Clone)]
pub struct FwmaInput<'a> {
    pub data: FwmaData<'a>,
    pub params: FwmaParams,
}

impl<'a> FwmaInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: FwmaParams) -> Self {
        Self {
            data: FwmaData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: FwmaParams) -> Self {
        Self {
            data: FwmaData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: FwmaData::Candles {
                candles,
                source: "close",
            },
            params: FwmaParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| FwmaParams::default().period.unwrap())
    }
}
#[inline]
pub fn fwma(input: &FwmaInput) -> Result<FwmaOutput, Box<dyn Error>> {
    let data: &[f64] = match &input.data {
        FwmaData::Candles { candles, source } => source_type(candles, source),
        FwmaData::Slice(slice) => slice,
    };
    let len = data.len();
    if len == 0 {
        return Err("No data provided.".into());
    }
    let period = input.get_period();
    if period == 0 || period > len {
        return Err("Invalid period.".into());
    }
    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err("All values are NaN.".into()),
    };
    if (len - first_valid_idx) < period {
        return Err("Not enough valid data.".into());
    }
    if data[first_valid_idx..].iter().any(|&v| v.is_nan()) {
        return Err("NaN found after first valid index.".into());
    }
    let mut fib = Vec::with_capacity(period);
    fib.push(1.0);
    if period > 1 {
        fib.push(1.0);
    }
    for i in 2..period {
        let next = fib[i - 1] + fib[i - 2];
        fib.push(next);
    }
    let fib_sum: f64 = fib.iter().sum();
    fib.iter_mut().for_each(|w| *w /= fib_sum);
    let mut values = vec![f64::NAN; len];
    let end_offset = first_valid_idx + period - 1;
    for i in end_offset..len {
        let start = i + 1 - period;
        let mut sum = 0.0;
        for j in 0..period {
            sum += data[start + j] * fib[j];
        }
        values[i] = sum;
    }
    Ok(FwmaOutput { values })
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_fwma_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input_default = FwmaInput::with_default_candles(&candles);
        let output_default = fwma(&input_default).expect("Failed FWMA with default params");
        assert_eq!(
            output_default.values.len(),
            candles.close.len(),
            "FWMA output length mismatch"
        );

        let params_period_only = FwmaParams { period: Some(10) };
        let input_period_only = FwmaInput::from_candles(&candles, "hl2", params_period_only);
        let output_period_only =
            fwma(&input_period_only).expect("Failed FWMA with period=10, source=hl2");
        assert_eq!(
            output_period_only.values.len(),
            candles.close.len(),
            "FWMA output length mismatch"
        );
    }

    #[test]
    fn test_fwma_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = FwmaInput::with_default_candles(&candles);
        let result = fwma(&input).expect("Failed to calculate FWMA");

        let expected_last_five = [
            59273.583333333336,
            59252.5,
            59167.083333333336,
            59151.0,
            58940.333333333336,
        ];

        assert!(result.values.len() >= 5);
        assert_eq!(
            result.values.len(),
            candles.close.len(),
            "FWMA values count should match input data count"
        );

        let start_index = result.values.len() - 5;
        let last_five = &result.values[start_index..];
        for (i, &val) in last_five.iter().enumerate() {
            let exp = expected_last_five[i];
            assert!(
                (val - exp).abs() < 1e-8,
                "FWMA mismatch at {}: expected {}, got {}",
                i,
                exp,
                val
            );
        }
    }
    #[test]
    fn test_fwma_params_with_default_params() {
        let default_params = FwmaParams::default();
        assert_eq!(default_params.period, Some(5), "Default period should be 5");
    }

    #[test]
    fn test_fwma_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = FwmaInput::with_default_candles(&candles);
        match input.data {
            FwmaData::Candles { source, .. } => {
                assert_eq!(source, "close");
            }
            _ => panic!("Expected FwmaData::Candles variant"),
        }
    }

    #[test]
    fn test_fwma_with_zero_period() {
        let input_data = [10.0, 20.0, 30.0];
        let params = FwmaParams { period: Some(0) };
        let input = FwmaInput::from_slice(&input_data, params);
        let result = fwma(&input);
        assert!(result.is_err(), "Expected an error for zero period");
        if let Err(e) = result {
            assert!(e.to_string().contains("Invalid period"));
        }
    }

    #[test]
    fn test_fwma_with_period_exceeding_data_length() {
        let input_data = [10.0, 20.0, 30.0];
        let params = FwmaParams { period: Some(10) };
        let input = FwmaInput::from_slice(&input_data, params);
        let result = fwma(&input);
        assert!(result.is_err(), "Expected an error for insufficient data");
    }

    #[test]
    fn test_fwma_very_small_data_set() {
        let input_data = [42.0];
        let params = FwmaParams { period: Some(5) };
        let input = FwmaInput::from_slice(&input_data, params);
        let result = fwma(&input);
        assert!(result.is_err(), "Expected an error for insufficient data");
    }

    #[test]
    fn test_fwma_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let first_params = FwmaParams { period: Some(5) };
        let first_input = FwmaInput::from_candles(&candles, "close", first_params);
        let first_result = fwma(&first_input).expect("Failed to calculate first FWMA");
        assert_eq!(
            first_result.values.len(),
            candles.close.len(),
            "FWMA output length mismatch"
        );

        let second_params = FwmaParams { period: Some(3) };
        let second_input = FwmaInput::from_slice(&first_result.values, second_params);
        let second_result = fwma(&second_input).expect("Failed to calculate second FWMA");
        assert_eq!(
            second_result.values.len(),
            first_result.values.len(),
            "FWMA output length mismatch"
        );
        for i in 240..second_result.values.len() {
            assert!(
                !second_result.values[i].is_nan(),
                "NaN found at index {}",
                i
            );
        }
    }

    #[test]
    fn test_fwma_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let period = 5;
        let params = FwmaParams {
            period: Some(period),
        };
        let input = FwmaInput::from_candles(&candles, "close", params);
        let result = fwma(&input).expect("Failed to calculate FWMA");
        assert_eq!(
            result.values.len(),
            candles.close.len(),
            "FWMA output length mismatch"
        );
        if result.values.len() > 50 {
            for i in 50..result.values.len() {
                assert!(!result.values[i].is_nan(), "NaN found at index {}", i);
            }
        }
    }
}
