/// # Chaikin Accumulation/Distribution Oscillator (ADOSC)
///
/// A momentum indicator developed by Marc Chaikin, based on the Accumulation/Distribution
/// Line (ADL). It subtracts a longer-term EMA of the ADL from a shorter-term EMA to
/// identify subtle shifts in market buying/selling pressure.
///
/// ## Parameters
/// - **short_period**: The shorter EMA period (defaults to 3).
/// - **long_period**: The longer EMA period (defaults to 10).
///
/// ## Errors
/// - **InvalidPeriod**: adosc: At least one of the periods (`short` or `long`) is zero.
/// - **ShortPeriodGreaterThanLong**: adosc: `short_period` ≥ `long_period`.
/// - **NoCandlesAvailable**: adosc: No candle data available.
/// - **NotEnoughData**: adosc: Not enough data points to compute the longer EMA.
/// - **EmptySlices**: adosc: At least one of the slices (high, low, close, volume) is empty.
///
/// ## Returns
/// - **`Ok(AdoscOutput)`** on success, with a `Vec<f64>` matching the length of the input data,
///   containing the ADOSC values.
/// - **`Err(AdoscError)`** otherwise.
///
/// # Example
/// ```
/// // Suppose we have a `candles` struct with high, low, close, volume data
/// let params = AdoscParams { short_period: Some(3), long_period: Some(10) };
/// let input = AdoscInput::from_candles(&candles, params);
/// let output = adosc(&input).expect("ADOSC calculation failed");
/// println!("ADOSC values: {:?}", output.values);
/// ```
use crate::utilities::data_loader::Candles;

#[derive(Debug, Clone)]
pub enum AdoscData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        volume: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct AdoscParams {
    pub short_period: Option<usize>,
    pub long_period: Option<usize>,
}

impl Default for AdoscParams {
    fn default() -> Self {
        Self {
            short_period: Some(3),
            long_period: Some(10),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AdoscInput<'a> {
    pub data: AdoscData<'a>,
    pub params: AdoscParams,
}

impl<'a> AdoscInput<'a> {
    pub fn from_candles(candles: &'a Candles, params: AdoscParams) -> Self {
        Self {
            data: AdoscData::Candles { candles },
            params,
        }
    }

    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        volume: &'a [f64],
        params: AdoscParams,
    ) -> Self {
        Self {
            data: AdoscData::Slices {
                high,
                low,
                close,
                volume,
            },
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: AdoscData::Candles { candles },
            params: AdoscParams::default(),
        }
    }

    fn get_short_period(&self) -> usize {
        self.params
            .short_period
            .unwrap_or_else(|| AdoscParams::default().short_period.unwrap())
    }

    fn get_long_period(&self) -> usize {
        self.params
            .long_period
            .unwrap_or_else(|| AdoscParams::default().long_period.unwrap())
    }
}

#[derive(Debug, Clone)]
pub struct AdoscOutput {
    pub values: Vec<f64>,
}

use thiserror::Error;

#[derive(Debug, Error)]
pub enum AdoscError {
    #[error(transparent)]
    CandleFieldError(#[from] Box<dyn std::error::Error>),

    #[error("Invalid period for ADOSC calculation: short={short}, long={long}")]
    InvalidPeriod { short: usize, long: usize },

    #[error(
        "Short period must be less than the long period for ADOSC: short={short}, long={long}"
    )]
    ShortPeriodGreaterThanLong { short: usize, long: usize },

    #[error("No candles available for ADOSC.")]
    NoCandlesAvailable,

    #[error("Not enough data points to calculate ADOSC: required={required}, have={have}")]
    NotEnoughData { required: usize, have: usize },

    #[error("One of the slices provided to ADOSC is empty: high={high}, low={low}, close={close}, volume={volume}")]
    EmptySlices {
        high: usize,
        low: usize,
        close: usize,
        volume: usize,
    },
}

#[inline]
pub fn adosc(input: &AdoscInput) -> Result<AdoscOutput, AdoscError> {
    let short = input.get_short_period();
    let long = input.get_long_period();

    if short == 0 || long == 0 {
        return Err(AdoscError::InvalidPeriod { short, long });
    }

    if short >= long {
        return Err(AdoscError::ShortPeriodGreaterThanLong { short, long });
    }

    let (high, low, close, volume) = match &input.data {
        AdoscData::Candles { candles } => {
            if candles.close.is_empty() {
                return Err(AdoscError::NoCandlesAvailable);
            }
            if long > candles.close.len() {
                return Err(AdoscError::NotEnoughData {
                    required: long,
                    have: candles.close.len(),
                });
            }
            (
                candles.select_candle_field("high")?,
                candles.select_candle_field("low")?,
                candles.select_candle_field("close")?,
                candles.select_candle_field("volume")?,
            )
        }
        AdoscData::Slices {
            high,
            low,
            close,
            volume,
        } => {
            if high.is_empty() || low.is_empty() || close.is_empty() || volume.is_empty() {
                return Err(AdoscError::EmptySlices {
                    high: high.len(),
                    low: low.len(),
                    close: close.len(),
                    volume: volume.len(),
                });
            }
            let len = close.len();
            if long > len {
                return Err(AdoscError::NotEnoughData {
                    required: long,
                    have: len,
                });
            }
            (*high, *low, *close, *volume)
        }
    };
    let len = close.len();

    let alpha_short = 2.0 / (short as f64 + 1.0);
    let alpha_long = 2.0 / (long as f64 + 1.0);

    let mut adosc_values = vec![0.0; len];

    let mut sum_ad = 0.0;

    {
        let h = high[0];
        let l = low[0];
        let c = close[0];
        let v = volume[0];

        let hl = h - l;
        let mfm = if hl != 0.0 {
            ((c - l) - (h - c)) / hl
        } else {
            0.0
        };
        let mfv = mfm * v;
        sum_ad += mfv;

        let mut short_ema = sum_ad;
        let mut long_ema = sum_ad;
        adosc_values[0] = short_ema - long_ema;

        for i in 1..len {
            let h = high[i];
            let l = low[i];
            let c = close[i];
            let v = volume[i];

            let hl = h - l;
            let mfm = if hl != 0.0 {
                ((c - l) - (h - c)) / hl
            } else {
                0.0
            };
            let mfv = mfm * v;
            sum_ad += mfv;

            short_ema = alpha_short * sum_ad + (1.0 - alpha_short) * short_ema;
            long_ema = alpha_long * sum_ad + (1.0 - alpha_long) * long_ema;

            adosc_values[i] = short_ema - long_ema;
        }
    }

    Ok(AdoscOutput {
        values: adosc_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_adosc_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = AdoscInput::with_default_candles(&candles);
        let result = adosc(&input).expect("Failed to calculate ADOSC");

        assert_eq!(
            result.values.len(),
            candles.close.len(),
            "ADOSC output length does not match input length"
        );

        let expected_last_five = [-166.2175, -148.9983, -144.9052, -128.5921, -142.0772];
        assert!(
            result.values.len() >= 5,
            "Not enough ADOSC values for the test"
        );

        let start_index = result.values.len().saturating_sub(5);
        let result_last_five = &result.values[start_index..];

        for (i, &actual) in result_last_five.iter().enumerate() {
            let expected = expected_last_five[i];
            assert!(
                (actual - expected).abs() < 1e-1,
                "ADOSC value mismatch at index {}: expected {}, got {}",
                i,
                expected,
                actual
            );
        }

        for (i, &val) in result.values.iter().enumerate() {
            assert!(
                val.is_finite(),
                "ADOSC output at index {} should be finite, got {}",
                i,
                val
            );
        }
    }
    #[test]
    fn test_adosc_params_with_default_params() {
        let default_params = AdoscParams::default();
        assert_eq!(default_params.short_period, Some(3));
        assert_eq!(default_params.long_period, Some(10));
    }

    #[test]
    fn test_adosc_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = AdoscInput::with_default_candles(&candles);
        match input.data {
            AdoscData::Candles { .. } => {}
            _ => panic!("Expected AdoscData::Candles variant"),
        }
        let default_params = AdoscParams::default();
        assert_eq!(input.params.short_period, default_params.short_period);
        assert_eq!(input.params.long_period, default_params.long_period);
    }

    #[test]
    fn test_adosc_with_zero_period() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let zero_short = AdoscParams {
            short_period: Some(0),
            long_period: Some(10),
        };
        let input_zero_short = AdoscInput::from_candles(&candles, zero_short);
        let result_zero_short = adosc(&input_zero_short);
        assert!(result_zero_short.is_err());

        let zero_long = AdoscParams {
            short_period: Some(3),
            long_period: Some(0),
        };
        let input_zero_long = AdoscInput::from_candles(&candles, zero_long);
        let result_zero_long = adosc(&input_zero_long);
        assert!(result_zero_long.is_err());
    }

    #[test]
    fn test_adosc_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let partial_params = AdoscParams {
            short_period: Some(2),
            long_period: None,
        };
        let input_partial = AdoscInput::from_candles(&candles, partial_params);
        let result_partial = adosc(&input_partial).expect("Failed ADOSC with partial params");
        assert_eq!(result_partial.values.len(), candles.close.len());

        let one_missing_short = AdoscParams {
            short_period: None,
            long_period: Some(12),
        };
        let input_missing_short = AdoscInput::from_candles(&candles, one_missing_short);
        let result_missing_short = adosc(&input_missing_short).expect("Failed ADOSC missing short");
        assert_eq!(result_missing_short.values.len(), candles.close.len());
    }

    #[test]
    fn test_adosc_invalid_period_relationship() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let params = AdoscParams {
            short_period: Some(10),
            long_period: Some(5),
        };
        let input = AdoscInput::from_candles(&candles, params);
        let result = adosc(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_adosc_period_exceeding_data_length() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let too_long = AdoscParams {
            short_period: Some(3),
            long_period: Some(candles.close.len() + 1),
        };
        let input_too_long = AdoscInput::from_candles(&candles, too_long);
        let result_too_long = adosc(&input_too_long);
        assert!(result_too_long.is_err());
    }

    #[test]
    fn test_adosc_very_small_data_set() {
        let short = 3;
        let long = 10;
        let high = [10.0];
        let low = [5.0];
        let close = [7.0];
        let volume = [1000.0];

        let params = AdoscParams {
            short_period: Some(short),
            long_period: Some(long),
        };
        let input = AdoscInput::from_slices(&high, &low, &close, &volume, params);
        let result = adosc(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_adosc_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let first_params = AdoscParams {
            short_period: Some(3),
            long_period: Some(10),
        };
        let first_input = AdoscInput::from_candles(&candles, first_params);
        let first_result = adosc(&first_input).expect("Failed to calculate ADOSC (first)");

        assert_eq!(first_result.values.len(), candles.close.len());

        let second_params = AdoscParams {
            short_period: Some(2),
            long_period: Some(6),
        };
        let second_input = AdoscInput::from_slices(
            &first_result.values,
            &first_result.values,
            &first_result.values,
            &first_result.values,
            second_params,
        );
        let second_result = adosc(&second_input).expect("Failed to calculate ADOSC (second)");
        assert_eq!(second_result.values.len(), first_result.values.len());
        if second_result.values.len() > 240 {
            for i in 240..second_result.values.len() {
                assert!(second_result.values[i].is_finite());
            }
        }
    }

    #[test]
    fn test_adosc_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let params = AdoscParams {
            short_period: Some(3),
            long_period: Some(10),
        };
        let input = AdoscInput::from_candles(&candles, params);
        let result = adosc(&input).expect("Failed to calculate ADOSC");
        assert_eq!(result.values.len(), candles.close.len());
        if result.values.len() > 240 {
            for i in 240..result.values.len() {
                assert!(!result.values[i].is_nan());
            }
        }
    }
}
