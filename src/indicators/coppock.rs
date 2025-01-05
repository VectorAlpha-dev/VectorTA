use crate::indicators::moving_averages::ma::{ma, MaData}; // <--- Adjust path as needed
/// # Coppock Curve (CC)
///
/// The Coppock Curve is a momentum indicator that sums two different ROC values
/// (long and short), and then smooths the sum with a chosen MA (e.g. WMA, SMA, etc.).
///
/// Classic defaults:
/// - Short ROC = 11
/// - Long ROC = 14
/// - MA period = 10
/// - MA type = "wma"
///
/// Formula (classic):
/// ```text
/// Coppock = MA( ROC(price, longPeriod) + ROC(price, shortPeriod), maPeriod )
/// ```
///
/// ## Parameters
/// - **short_roc_period**: Period for short ROC (defaults to 11).
/// - **long_roc_period**: Period for long ROC (defaults to 14).
/// - **ma_period**: Period for smoothing (defaults to 10).
/// - **ma_type**: Type of MA (e.g., `"wma"`, `"ema"`, `"sma"`). Defaults to `"wma"`.
/// - **source**: Candle field (e.g. `"close"`, `"hlc3"`). Defaults to `"close"`.
///
/// ## Errors
/// - **EmptyData**: Input data slice is empty.
/// - **AllValuesNaN**: All data values are `NaN`.
/// - **NotEnoughValidData**: Not enough valid data to compute at least one output.
/// - **InvalidPeriod**: Zero or out-of-bounds short/long/MA periods.
/// - **MaError**: Underlying error from the `ma(...)` function.
///
/// ## Returns
/// - `Ok(CoppockOutput)` on success, containing a vector matching the input length,
///   with leading `NaN`s until the earliest valid index.
/// - `Err(CoppockError)` otherwise.
use crate::utilities::data_loader::{source_type, Candles};
use std::error::Error;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum CoppockData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct CoppockOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct CoppockParams {
    pub short_roc_period: Option<usize>,
    pub long_roc_period: Option<usize>,
    pub ma_period: Option<usize>,
    pub ma_type: Option<String>,
}

impl Default for CoppockParams {
    fn default() -> Self {
        Self {
            short_roc_period: Some(11),
            long_roc_period: Some(14),
            ma_period: Some(10),
            ma_type: Some("wma".to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CoppockInput<'a> {
    pub data: CoppockData<'a>,
    pub params: CoppockParams,
}

impl<'a> CoppockInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: CoppockParams) -> Self {
        Self {
            data: CoppockData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: CoppockParams) -> Self {
        Self {
            data: CoppockData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: CoppockData::Candles {
                candles,
                source: "close",
            },
            params: CoppockParams::default(),
        }
    }

    pub fn get_short_roc_period(&self) -> usize {
        self.params
            .short_roc_period
            .unwrap_or_else(|| CoppockParams::default().short_roc_period.unwrap())
    }

    pub fn get_long_roc_period(&self) -> usize {
        self.params
            .long_roc_period
            .unwrap_or_else(|| CoppockParams::default().long_roc_period.unwrap())
    }

    pub fn get_ma_period(&self) -> usize {
        self.params
            .ma_period
            .unwrap_or_else(|| CoppockParams::default().ma_period.unwrap())
    }

    pub fn get_ma_type(&self) -> String {
        self.params
            .ma_type
            .clone()
            .unwrap_or_else(|| CoppockParams::default().ma_type.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum CoppockError {
    #[error("coppock: Empty data provided.")]
    EmptyData,
    #[error("coppock: All values are NaN.")]
    AllValuesNaN,
    #[error("coppock: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "coppock: Invalid period usage => short={short}, long={long}, ma={ma}, data_len={data_len}"
    )]
    InvalidPeriod {
        short: usize,
        long: usize,
        ma: usize,
        data_len: usize,
    },
    #[error("coppock: Underlying MA error: {0}")]
    MaError(#[from] Box<dyn Error>),
}

#[inline]
pub fn coppock(input: &CoppockInput) -> Result<CoppockOutput, CoppockError> {
    let data: &[f64] = match &input.data {
        CoppockData::Candles { candles, source } => source_type(candles, source),
        CoppockData::Slice(slice) => slice,
    };

    if data.is_empty() {
        return Err(CoppockError::EmptyData);
    }

    let short = input.get_short_roc_period();
    let long = input.get_long_roc_period();
    let ma_p = input.get_ma_period();
    let data_len = data.len();

    if short == 0
        || long == 0
        || ma_p == 0
        || short > data_len
        || long > data_len
        || ma_p > data_len
    {
        return Err(CoppockError::InvalidPeriod {
            short,
            long,
            ma: ma_p,
            data_len,
        });
    }

    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(CoppockError::AllValuesNaN),
    };

    let largest_roc = short.max(long);
    if (data_len - first_valid_idx) < largest_roc {
        return Err(CoppockError::NotEnoughValidData {
            needed: largest_roc,
            valid: data_len - first_valid_idx,
        });
    }

    let mut sum_roc = vec![f64::NAN; data_len];

    let start_idx = first_valid_idx + largest_roc;
    for i in start_idx..data_len {
        let current = data[i];
        let prev_short = data[i - short];
        let short_val = ((current / prev_short) - 1.0) * 100.0;
        let prev_long = data[i - long];
        let long_val = ((current / prev_long) - 1.0) * 100.0;
        sum_roc[i] = short_val + long_val;
    }

    let ma_type = input.get_ma_type();
    let smoothed = ma(&ma_type, MaData::Slice(&sum_roc), ma_p)?;

    Ok(CoppockOutput { values: smoothed })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_coppock_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = CoppockParams::default();
        let input_default = CoppockInput::from_candles(&candles, "close", default_params);
        let output_default = coppock(&input_default).expect("Failed Coppock with defaults");
        assert_eq!(output_default.values.len(), candles.close.len());

        let custom_params = CoppockParams {
            short_roc_period: Some(9),
            long_roc_period: Some(13),
            ma_period: Some(8),
            ma_type: Some("sma".to_string()),
        };
        let input_custom = CoppockInput::from_candles(&candles, "hlc3", custom_params);
        let output_custom = coppock(&input_custom).expect("Failed Coppock with custom");
        assert_eq!(output_custom.values.len(), candles.close.len());
    }

    #[test]
    fn test_coppock_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = CoppockInput::with_default_candles(&candles);
        let coppock_result = coppock(&input).expect("Failed to calculate Coppock");

        assert_eq!(
            coppock_result.values.len(),
            candles.close.len(),
            "Coppock length mismatch"
        );

        let expected_last_five = [
            -1.4542764618985533,
            -1.3795224034983653,
            -1.614331648987457,
            -1.9179048338714915,
            -2.1096548435774625,
        ];

        assert!(
            coppock_result.values.len() >= 5,
            "Not enough data to check the last 5 values"
        );

        let start_idx = coppock_result.values.len() - 5;
        let last_five_values = &coppock_result.values[start_idx..];

        for (i, &actual) in last_five_values.iter().enumerate() {
            let expected = expected_last_five[i];
            assert!(
                (actual - expected).abs() < 1e-7,
                "Coppock mismatch at final 5 index {}: expected {}, got {}",
                i,
                expected,
                actual
            );
        }
    }

    #[test]
    fn test_coppock_params_with_default_params() {
        let defaults = CoppockParams::default();
        assert_eq!(defaults.short_roc_period, Some(11));
        assert_eq!(defaults.long_roc_period, Some(14));
        assert_eq!(defaults.ma_period, Some(10));
        assert_eq!(defaults.ma_type.as_deref(), Some("wma"));
    }

    #[test]
    fn test_coppock_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = CoppockInput::with_default_candles(&candles);

        match input.data {
            CoppockData::Candles { source, .. } => {
                assert_eq!(source, "close", "Expected default source='close'");
            }
            _ => panic!("Expected CoppockData::Candles"),
        }
    }

    #[test]
    fn test_coppock_with_invalid_periods() {
        let data = [10.0, 11.0, 12.0];

        let zero_params = CoppockParams {
            short_roc_period: Some(0),
            long_roc_period: Some(14),
            ma_period: Some(10),
            ma_type: Some("wma".to_string()),
        };
        let zero_input = CoppockInput::from_slice(&data, zero_params);
        let result = coppock(&zero_input);
        assert!(result.is_err(), "Expected error with zero short period");

        let big_params = CoppockParams {
            short_roc_period: Some(14),
            long_roc_period: Some(20),
            ma_period: Some(10),
            ma_type: Some("wma".to_string()),
        };
        let big_input = CoppockInput::from_slice(&data, big_params);
        let result2 = coppock(&big_input);
        assert!(result2.is_err(), "Expected error for short/long>data.len()");
    }

    #[test]
    fn test_coppock_all_nan() {
        let data = [f64::NAN, f64::NAN, f64::NAN];
        let input = CoppockInput::from_slice(&data, CoppockParams::default());
        let result = coppock(&input);
        assert!(result.is_err(), "Expected AllValuesNaN error");
    }

    #[test]
    fn test_coppock_not_enough_valid_data() {
        let data = [f64::NAN, f64::NAN, 10.0, 11.0, 12.0, 13.0, 14.0];
        let input = CoppockInput::from_slice(&data, CoppockParams::default());
        let result = coppock(&input);
        assert!(result.is_err(), "Expected NotEnoughValidData error");
    }

    #[test]
    fn test_coppock_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let first_input = CoppockInput::with_default_candles(&candles);
        let first_result = coppock(&first_input).expect("Failed first Coppock");
        assert_eq!(
            first_result.values.len(),
            candles.close.len(),
            "First length mismatch"
        );

        let second_params = CoppockParams {
            short_roc_period: Some(5),
            long_roc_period: Some(8),
            ma_period: Some(3),
            ma_type: Some("sma".to_string()),
        };
        let second_input = CoppockInput::from_slice(&first_result.values, second_params);
        let second_result = coppock(&second_input).expect("Failed second Coppock");
        assert_eq!(
            second_result.values.len(),
            first_result.values.len(),
            "Second length mismatch"
        );

        for i in 240..second_result.values.len() {
            assert!(
                !second_result.values[i].is_nan(),
                "Expected no NaN after index 30, found NaN at {}",
                i
            );
        }
    }
}
