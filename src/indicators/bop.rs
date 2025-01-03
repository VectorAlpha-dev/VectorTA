/// # Balance of Power (BOP)
///
/// **Formula**: \[ (Close - Open) / (High - Low) \]
///
/// If `(High - Low)` is zero or negative, the output is set to `0.0`.
///
/// ## Parameters
/// Currently, there are no external parameters needed for BOP. A stub `BopParams` is provided for
/// future extensibility.
///
/// ## Errors
/// - **EmptyData**: bop: No data was provided.
/// - **InconsistentLengths**: bop: Input data slices (open, high, low, close) have different lengths.
///
/// ## Returns
/// - **`Ok(BopOutput)`** on success, containing a `Vec<f64>` of BOP values matching the length of the inputs.
/// - **`Err(BopError)`** otherwise.
use crate::utilities::data_loader::Candles;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum BopData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
}

#[derive(Debug, Clone, Default)]
pub struct BopParams {}

#[derive(Debug, Clone)]
pub struct BopInput<'a> {
    pub data: BopData<'a>,
    pub params: BopParams,
}

impl<'a> BopInput<'a> {
    pub fn from_candles(candles: &'a Candles, params: BopParams) -> Self {
        Self {
            data: BopData::Candles { candles },
            params,
        }
    }

    pub fn from_slices(
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: BopParams,
    ) -> Self {
        Self {
            data: BopData::Slices {
                open,
                high,
                low,
                close,
            },
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: BopData::Candles { candles },
            params: BopParams::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BopOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Error)]
pub enum BopError {
    #[error("bop: Data is empty.")]
    EmptyData,
    #[error("bop: Inconsistent lengths.")]
    InconsistentLengths,
    #[error("bop: Candle field error: {0}")]
    CandleFieldError(String),
}

#[inline]
pub fn bop(input: &BopInput) -> Result<BopOutput, BopError> {
    match &input.data {
        BopData::Candles { candles } => {
            let open = candles
                .select_candle_field("open")
                .map_err(|e| BopError::CandleFieldError(e.to_string()))?;
            let high = candles
                .select_candle_field("high")
                .map_err(|e| BopError::CandleFieldError(e.to_string()))?;
            let low = candles
                .select_candle_field("low")
                .map_err(|e| BopError::CandleFieldError(e.to_string()))?;
            let close = candles
                .select_candle_field("close")
                .map_err(|e| BopError::CandleFieldError(e.to_string()))?;
            let len = open.len();
            if len == 0 {
                return Err(BopError::EmptyData);
            }
            if len != high.len() || len != low.len() || len != close.len() {
                return Err(BopError::InconsistentLengths);
            }
            let values: Vec<f64> = open
                .iter()
                .zip(high.iter())
                .zip(low.iter())
                .zip(close.iter())
                .map(|(((o, h), l), c)| {
                    let denom = h - l;
                    if denom <= 0.0 {
                        0.0
                    } else {
                        (c - o) / denom
                    }
                })
                .collect();
            Ok(BopOutput { values })
        }
        BopData::Slices {
            open,
            high,
            low,
            close,
        } => {
            let len = open.len();
            if len == 0 {
                return Err(BopError::EmptyData);
            }
            if len != high.len() || len != low.len() || len != close.len() {
                return Err(BopError::InconsistentLengths);
            }
            let values: Vec<f64> = open
                .iter()
                .zip(high.iter())
                .zip(low.iter())
                .zip(close.iter())
                .map(|(((o, h), l), c)| {
                    let denom = h - l;
                    if denom <= 0.0 {
                        0.0
                    } else {
                        (c - o) / denom
                    }
                })
                .collect();
            Ok(BopOutput { values })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_bop_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = BopParams::default();
        let input_default = BopInput::with_default_candles(&candles);
        let bop_output_default = bop(&input_default).expect("Failed BOP with default params");
        assert_eq!(
            bop_output_default.values.len(),
            candles.close.len(),
            "Default BOP output length mismatch"
        );

        let custom_input = BopInput::from_candles(&candles, default_params);
        let custom_output = bop(&custom_input).expect("Failed BOP custom params");
        assert_eq!(
            custom_output.values.len(),
            candles.close.len(),
            "Custom BOP output length mismatch"
        );
    }

    #[test]
    fn test_bop_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = BopInput::with_default_candles(&candles);
        let bop_result = bop(&input).expect("Failed to calculate BOP");

        assert_eq!(
            bop_result.values.len(),
            candles.close.len(),
            "BOP length mismatch"
        );

        let expected_last_five_bop = [
            0.045454545454545456,
            -0.32398753894080995,
            -0.3844086021505376,
            0.3547400611620795,
            -0.5336179295624333,
        ];
        assert!(
            bop_result.values.len() >= 5,
            "BOP length too short for comparison"
        );

        let start_index = bop_result.values.len() - 5;
        let result_last_five_bop = &bop_result.values[start_index..];
        for (i, &value) in result_last_five_bop.iter().enumerate() {
            let expected_value = expected_last_five_bop[i];
            assert!(
                (value - expected_value).abs() < 1e-10,
                "BOP mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }
    }

    #[test]
    fn test_bop_params_with_default_params() {
        let default_params = BopParams::default();
        assert_eq!(
            std::mem::size_of_val(&default_params),
            0,
            "BopParams is not empty as expected."
        );
    }

    #[test]
    fn test_bop_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = BopInput::with_default_candles(&candles);
        match input.data {
            BopData::Candles { candles: _ } => {}
            _ => panic!("Expected BopData::Candles variant"),
        }
    }

    #[test]
    fn test_bop_with_empty_data() {
        let empty: [f64; 0] = [];
        let params = BopParams::default();
        let input = BopInput::from_slices(&empty, &empty, &empty, &empty, params);

        let result = bop(&input);
        assert!(result.is_err(), "Expected an error for empty data");
    }

    #[test]
    fn test_bop_with_inconsistent_lengths() {
        let open = [1.0, 2.0, 3.0];
        let high = [1.5, 2.5];
        let low = [0.8, 1.8, 2.8];
        let close = [1.2, 2.2, 3.2];

        let params = BopParams::default();
        let input = BopInput::from_slices(&open, &high, &low, &close, params);

        let result = bop(&input);
        assert!(
            result.is_err(),
            "Expected an error for inconsistent input lengths"
        );
    }

    #[test]
    fn test_bop_very_small_data_set() {
        let open = [10.0];
        let high = [12.0];
        let low = [9.5];
        let close = [11.0];

        let params = BopParams::default();
        let input = BopInput::from_slices(&open, &high, &low, &close, params);

        let result = bop(&input).expect("Failed BOP with single data point");
        assert_eq!(
            result.values.len(),
            1,
            "Expected exactly one BOP value for single data point."
        );

        assert!((result.values[0] - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_bop_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let first_input = BopInput::with_default_candles(&candles);
        let first_result = bop(&first_input).expect("Failed to calculate first BOP");

        let second_params = BopParams::default();
        let dummy = vec![0.0; first_result.values.len()];
        let second_input =
            BopInput::from_slices(&dummy, &dummy, &dummy, &first_result.values, second_params);
        let second_result = bop(&second_input).expect("Failed second BOP re-input");

        assert_eq!(
            second_result.values.len(),
            first_result.values.len(),
            "Second BOP output length mismatch"
        );

        for (i, &val) in second_result.values.iter().enumerate() {
            assert!(
                (val - 0.0).abs() < f64::EPSILON,
                "Expected BOP=0.0 for dummy data at index {}, got {}",
                i,
                val
            );
        }
    }

    #[test]
    fn test_bop_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = BopInput::with_default_candles(&candles);
        let bop_result = bop(&input).expect("Failed to calculate BOP");

        assert_eq!(
            bop_result.values.len(),
            candles.close.len(),
            "BOP length mismatch"
        );

        if bop_result.values.len() > 240 {
            for i in 240..bop_result.values.len() {
                assert!(
                    !bop_result.values[i].is_nan(),
                    "Expected no NaN after index 240, but found NaN at index {}",
                    i
                );
            }
        }
    }
}
