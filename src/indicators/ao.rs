/// # Awesome Oscillator (AO)
///
/// A technical analysis indicator developed by Bill Williams, designed to measure
/// market momentum by comparing recent price changes against a longer baseline.
/// It uses the difference between two Simple Moving Averages (SMAs) of the
/// median price (`hl2`)—one short and one long—to highlight shifts in market force.
///
/// ## Parameters
/// - **short_period**: The number of bars used for the short SMA (defaults to 5).
/// - **long_period**: The number of bars used for the long SMA (defaults to 34).
///
/// ## Errors
/// - **InvalidPeriods**: ao: One or both periods are zero (`short=0` or `long=0`).
/// - **ShortPeriodNotLess**: ao: `short_period` ≥ `long_period`.
/// - **NoData**: ao: The input slice is empty.
/// - **NotEnoughData**: ao: The input slice is smaller than the `long_period`.
/// - **AllValuesNaN**: ao: All values in the data are `NaN`.
///
/// ## Returns
/// - **`Ok(AoOutput)`** on success, containing a `Vec<f64>` whose length matches
///   the input data. Leading values (before the `long_period` is reached) remain
///   `NaN`, while subsequent bars hold the AO values.
/// - **`Err(AoError)`** otherwise.
///
/// # Example
/// ```
/// // Suppose `candles` is a Candles structure with high, low, and close data
/// let params = AoParams { short_period: Some(5), long_period: Some(34) };
/// let input = AoInput::from_candles(&candles, "hl2", params);
/// let output = ao(&input).unwrap();
/// println!("Awesome Oscillator values: {:?}", output.values);
/// ```
use crate::utilities::data_loader::{source_type, Candles};

#[derive(Debug, Clone)]
pub enum AoData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct AoParams {
    pub short_period: Option<usize>,
    pub long_period: Option<usize>,
}

impl Default for AoParams {
    fn default() -> Self {
        Self {
            short_period: Some(5),
            long_period: Some(34),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AoInput<'a> {
    pub data: AoData<'a>,
    pub params: AoParams,
}

impl<'a> AoInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: AoParams) -> Self {
        Self {
            data: AoData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(data: &'a [f64], params: AoParams) -> Self {
        Self {
            data: AoData::Slice(data),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: AoData::Candles {
                candles,
                source: "hl2",
            },
            params: AoParams::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AoOutput {
    pub values: Vec<f64>,
}

use thiserror::Error;

#[derive(Debug, Error)]
pub enum AoError {
    #[error("AO: Invalid periods for AO calculation: short={short}, long={long}. Both must be greater than 0.")]
    InvalidPeriods { short: usize, long: usize },
    #[error("AO: Short period must be strictly less than long period: short={short}, long={long}")]
    ShortPeriodNotLess { short: usize, long: usize },
    #[error("AO: No data provided (HL2 slice is empty).")]
    NoData,
    #[error(
        "AO: Not enough data to compute AO: requested long period = {long}, data length = {data_len}"
    )]
    NotEnoughData { long: usize, data_len: usize },
    #[error("All values in the data are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn ao(input: &AoInput) -> Result<AoOutput, AoError> {
    let short = input.params.short_period.unwrap_or(5);
    let long = input.params.long_period.unwrap_or(34);

    if short == 0 || long == 0 {
        return Err(AoError::InvalidPeriods { short, long });
    }
    if short >= long {
        return Err(AoError::ShortPeriodNotLess { short, long });
    }

    let data: &[f64] = match &input.data {
        AoData::Candles { candles, source } => source_type(candles, source),
        AoData::Slice(slice) => slice,
    };

    if data.is_empty() {
        return Err(AoError::NoData);
    }

    let len = data.len();
    if long > len {
        return Err(AoError::NotEnoughData {
            long,
            data_len: len,
        });
    }

    let mut ao_values = vec![f64::NAN; len];

    let mut short_sum = 0.0;
    let mut long_sum = 0.0;

    for i in 0..len {
        let val = data[i];
        short_sum += val;
        long_sum += val;

        if i >= short {
            short_sum -= data[i - short];
        }

        if i >= long {
            long_sum -= data[i - long];
        }

        if i >= (long - 1) {
            let short_sma = short_sum / (short as f64);
            let long_sma = long_sum / (long as f64);
            ao_values[i] = short_sma - long_sma;
        }
    }

    Ok(AoOutput { values: ao_values })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_ao_params_with_default_params() {
        let default_params = AoParams::default();
        assert_eq!(default_params.short_period, Some(5));
        assert_eq!(default_params.long_period, Some(34));
    }

    #[test]
    fn test_ao_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = AoInput::with_default_candles(&candles);
        match input.data {
            AoData::Candles { source, .. } => {
                assert_eq!(source, "hl2");
            }
            _ => panic!("Expected AoData::Candles variant"),
        }
        let default_params = AoParams::default();
        assert_eq!(input.params.short_period, default_params.short_period);
        assert_eq!(input.params.long_period, default_params.long_period);
    }

    #[test]
    fn test_ao_with_zero_period() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let zero_short_params = AoParams {
            short_period: Some(0),
            long_period: Some(34),
        };
        let input_zero_short = AoInput::from_candles(&candles, "hl2", zero_short_params);
        let result_zero_short = ao(&input_zero_short);
        assert!(result_zero_short.is_err());

        let zero_long_params = AoParams {
            short_period: Some(5),
            long_period: Some(0),
        };
        let input_zero_long = AoInput::from_candles(&candles, "hl2", zero_long_params);
        let result_zero_long = ao(&input_zero_long);
        assert!(result_zero_long.is_err());
    }

    #[test]
    fn test_ao_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let partial_params = AoParams {
            short_period: Some(3),
            long_period: None,
        };
        let input_partial = AoInput::from_candles(&candles, "hl2", partial_params);
        let result_partial = ao(&input_partial).expect("Failed AO with partial params");
        assert_eq!(result_partial.values.len(), candles.close.len());

        let missing_short_params = AoParams {
            short_period: None,
            long_period: Some(40),
        };
        let input_missing_short = AoInput::from_candles(&candles, "hl2", missing_short_params);
        let result_missing_short = ao(&input_missing_short).expect("Failed AO with missing short");
        assert_eq!(result_missing_short.values.len(), candles.close.len());
    }

    #[test]
    fn test_ao_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let hl2_values: Vec<f64> = candles
            .high
            .iter()
            .zip(&candles.low)
            .map(|(&h, &l)| 0.5 * (h + l))
            .collect();

        let input = AoInput::with_default_candles(&candles);
        let result = ao(&input).expect("Failed to calculate AO");
        let expected_last_five = [-1671.3, -1401.6706, -1262.3559, -1178.4941, -1157.4118];

        assert!(
            result.values.len() >= 5,
            "Not enough AO values for the test"
        );

        assert_eq!(
            result.values.len(),
            hl2_values.len(),
            "AO output length does not match input length"
        );

        let start_index = result.values.len().saturating_sub(5);
        let result_last_five = &result.values[start_index..];

        for (i, &value) in result_last_five.iter().enumerate() {
            assert!(
                (value - expected_last_five[i]).abs() < 1e-1,
                "AO value mismatch at index {}: expected {}, got {}",
                i,
                expected_last_five[i],
                value
            );
        }

        for val in result.values.iter().skip(34 - 1) {
            assert!(
                val.is_finite(),
                "AO output should be finite at valid indices"
            );
        }
    }
    #[test]
    fn test_ao_invalid_period_relationship() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let invalid_params = AoParams {
            short_period: Some(40),
            long_period: Some(10),
        };
        let input = AoInput::from_candles(&candles, "hl2", invalid_params);
        let result = ao(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_ao_period_exceeding_data_length() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let too_long_params = AoParams {
            short_period: Some(5),
            long_period: Some(candles.close.len() + 10),
        };
        let input_too_long = AoInput::from_candles(&candles, "hl2", too_long_params);
        let result_too_long = ao(&input_too_long);
        assert!(result_too_long.is_err());
    }

    #[test]
    fn test_ao_very_small_data_set() {
        let data = [42.0];
        let params = AoParams {
            short_period: Some(5),
            long_period: Some(34),
        };
        let input = AoInput::from_slice(&data, params);
        let result = ao(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_ao_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let first_params = AoParams {
            short_period: Some(5),
            long_period: Some(34),
        };
        let first_input = AoInput::from_candles(&candles, "hl2", first_params);
        let first_result = ao(&first_input).expect("Failed AO (first run)");
        assert_eq!(first_result.values.len(), candles.close.len());

        let second_params = AoParams {
            short_period: Some(3),
            long_period: Some(10),
        };
        let second_input = AoInput::from_slice(&first_result.values, second_params);
        let second_result = ao(&second_input).expect("Failed AO (second run)");
        assert_eq!(second_result.values.len(), first_result.values.len());
    }

    #[test]
    fn test_ao_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let params = AoParams {
            short_period: Some(5),
            long_period: Some(34),
        };
        let input = AoInput::from_candles(&candles, "hl2", params);
        let result = ao(&input).expect("Failed to calculate AO");
        assert_eq!(result.values.len(), candles.close.len());

        if result.values.len() > 240 {
            for i in 240..result.values.len() {
                assert!(!result.values[i].is_nan());
            }
        }
    }
}
