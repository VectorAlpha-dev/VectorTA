/// # Mass Index (MASS)
///
/// The Mass Index is an indicator that uses the ratio of two exponential moving averages
/// (both using period=9) of the range (high - low) and sums these ratios over `period` bars.
/// This implementation follows the Tulip Indicators reference for MASS, with a default period of 5.
///
/// ## Parameters
/// - **period**: The summation window size. Defaults to 5.
///
/// ## Errors
/// - **EmptyData**: mass: Input data slices are empty.
/// - **DifferentLengthHL**: mass: `high` and `low` slices have different lengths.
/// - **InvalidPeriod**: mass: `period` is zero or exceeds the data length.
/// - **NotEnoughValidData**: mass: Fewer than `16 + period - 1` valid (non-`NaN`) data points remain
///   after the first valid index.
/// - **AllValuesNaN**: mass: All input data values are `NaN`.
///
/// ## Returns
/// - **`Ok(MassOutput)`** on success, containing a `Vec<f64>` matching the input length,
///   with leading `NaN`s until enough data is accumulated for the Mass Index calculation.
/// - **`Err(MassError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};

#[derive(Debug, Clone)]
pub enum MassData<'a> {
    Candles {
        candles: &'a Candles,
        high_source: &'a str,
        low_source: &'a str,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct MassOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct MassParams {
    pub period: Option<usize>,
}

impl Default for MassParams {
    fn default() -> Self {
        Self { period: Some(5) }
    }
}

#[derive(Debug, Clone)]
pub struct MassInput<'a> {
    pub data: MassData<'a>,
    pub params: MassParams,
}

impl<'a> MassInput<'a> {
    pub fn from_candles(
        candles: &'a Candles,
        high_source: &'a str,
        low_source: &'a str,
        params: MassParams,
    ) -> Self {
        Self {
            data: MassData::Candles {
                candles,
                high_source,
                low_source,
            },
            params,
        }
    }

    pub fn from_slices(high: &'a [f64], low: &'a [f64], params: MassParams) -> Self {
        Self {
            data: MassData::Slices { high, low },
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: MassData::Candles {
                candles,
                high_source: "high",
                low_source: "low",
            },
            params: MassParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| MassParams::default().period.unwrap())
    }
}

use thiserror::Error;

#[derive(Debug, Error)]
pub enum MassError {
    #[error("mass: Empty data provided.")]
    EmptyData,
    #[error("mass: High and low slices must have the same length.")]
    DifferentLengthHL,
    #[error("mass: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("mass: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("mass: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn mass(input: &MassInput) -> Result<MassOutput, MassError> {
    let (high, low) = match &input.data {
        MassData::Candles {
            candles,
            high_source,
            low_source,
        } => (
            source_type(candles, high_source),
            source_type(candles, low_source),
        ),
        MassData::Slices { high, low } => (*high, *low),
    };

    if high.is_empty() || low.is_empty() {
        return Err(MassError::EmptyData);
    }
    if high.len() != low.len() {
        return Err(MassError::DifferentLengthHL);
    }

    let period = input.get_period();
    if period == 0 || period > high.len() {
        return Err(MassError::InvalidPeriod {
            period,
            data_len: high.len(),
        });
    }

    let first_valid_idx = match (0..high.len()).find(|&i| !high[i].is_nan() && !low[i].is_nan()) {
        Some(idx) => idx,
        None => return Err(MassError::AllValuesNaN),
    };

    let needed_bars = 16 + period - 1;
    if (high.len() - first_valid_idx) < needed_bars {
        return Err(MassError::NotEnoughValidData {
            needed: needed_bars,
            valid: high.len() - first_valid_idx,
        });
    }

    let mut mass_values = vec![f64::NAN; high.len()];

    let alpha = 2.0 / 10.0;
    let inv_alpha = 1.0 - alpha;
    let mut ema1 = high[first_valid_idx] - low[first_valid_idx];
    let mut ema2 = ema1;

    let mut ring = vec![0.0; period];
    let mut ring_index = 0;
    let mut sum_ratio = 0.0;

    for i in first_valid_idx..high.len() {
        let hl = high[i] - low[i];
        ema1 = ema1.mul_add(inv_alpha, hl * alpha);

        if i == first_valid_idx + 8 {
            ema2 = ema1;
        }

        if i >= first_valid_idx + 8 {
            ema2 = ema2.mul_add(inv_alpha, ema1 * alpha);
        }

        if i >= first_valid_idx + 16 {
            let ratio = ema1 / ema2;
            sum_ratio -= ring[ring_index];
            ring[ring_index] = ratio;
            sum_ratio += ratio;
            ring_index = (ring_index + 1) % period;

            if i >= first_valid_idx + 16 + (period - 1) {
                mass_values[i] = sum_ratio;
            }
        }
    }

    Ok(MassOutput {
        values: mass_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_mass_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = MassParams { period: None };
        let input_default = MassInput::from_candles(&candles, "high", "low", default_params);
        let output_default = mass(&input_default).expect("Failed MASS with default params");
        assert_eq!(output_default.values.len(), candles.high.len());

        let params_period_14 = MassParams { period: Some(14) };
        let input_period_14 = MassInput::from_candles(&candles, "high", "low", params_period_14);
        let output_period_14 =
            mass(&input_period_14).expect("Failed MASS with period=14, high=high, low=low");
        assert_eq!(output_period_14.values.len(), candles.high.len());

        let params_custom = MassParams { period: Some(20) };
        let input_custom = MassInput::from_candles(&candles, "high", "low", params_custom);
        let output_custom = mass(&input_custom).expect("Failed MASS fully custom");
        assert_eq!(output_custom.values.len(), candles.high.len());
    }

    #[test]
    fn test_mass_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let params = MassParams { period: Some(5) };
        let input = MassInput::from_candles(&candles, "high", "low", params);
        let mass_result = mass(&input).expect("Failed to calculate MASS");

        assert_eq!(
            mass_result.values.len(),
            candles.high.len(),
            "MASS length mismatch"
        );

        // Compare last 5 MASS values with the known test values
        let expected_last_five = [
            4.512263952194651,
            4.126178935431121,
            3.838738456245828,
            3.6450956734739375,
            3.6748009093527125,
        ];
        let result_len = mass_result.values.len();
        assert!(
            result_len >= 5,
            "MASS output length is too short for comparison"
        );
        let start_idx = result_len - 5;
        let result_slice = &mass_result.values[start_idx..];
        for (i, &value) in result_slice.iter().enumerate() {
            let expected = expected_last_five[i];
            assert!(
                (value - expected).abs() < 1e-7,
                "MASS mismatch at index {}: expected {}, got {}",
                start_idx + i,
                expected,
                value
            );
        }
    }

    #[test]
    fn test_mass_params_with_default_params() {
        let default_params = MassParams::default();
        assert_eq!(
            default_params.period,
            Some(5),
            "Expected period=5 in default parameters"
        );
    }

    #[test]
    fn test_mass_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = MassInput::with_default_candles(&candles);
        match input.data {
            MassData::Candles {
                high_source,
                low_source,
                ..
            } => {
                assert_eq!(high_source, "high", "Expected default source to be 'high'");
                assert_eq!(low_source, "low", "Expected default source to be 'low'");
            }
            _ => panic!("Expected MassData::Candles variant"),
        }
    }

    #[test]
    fn test_mass_with_zero_period() {
        let high_data = [10.0, 15.0, 20.0];
        let low_data = [5.0, 10.0, 12.0];
        let params = MassParams { period: Some(0) };
        let input = MassInput::from_slices(&high_data, &low_data, params);

        let result = mass(&input);
        assert!(result.is_err(), "Expected an error for zero period");
        if let Err(e) = result {
            assert!(
                e.to_string().contains("Invalid period"),
                "Expected 'Invalid period' error message, got: {}",
                e
            );
        }
    }

    #[test]
    fn test_mass_with_period_exceeding_data_length() {
        let high_data = [10.0, 15.0, 20.0];
        let low_data = [5.0, 10.0, 12.0];
        let params = MassParams { period: Some(10) };
        let input = MassInput::from_slices(&high_data, &low_data, params);

        let result = mass(&input);
        assert!(result.is_err(), "Expected an error for period > data.len()");
    }

    #[test]
    fn test_mass_very_small_data_set() {
        let high_data = [10.0];
        let low_data = [5.0];
        let params = MassParams { period: Some(5) };
        let input = MassInput::from_slices(&high_data, &low_data, params);

        let result = mass(&input);
        assert!(
            result.is_err(),
            "Expected error for data smaller than needed bars"
        );
    }

    #[test]
    fn test_mass_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let first_params = MassParams { period: Some(5) };
        let first_input = MassInput::from_candles(&candles, "high", "low", first_params);
        let first_result = mass(&first_input).expect("Failed to calculate first MASS");

        let second_params = MassParams { period: Some(5) };
        let second_input =
            MassInput::from_slices(&first_result.values, &first_result.values, second_params);
        let second_result = mass(&second_input).expect("Failed to calculate second MASS");

        assert_eq!(
            second_result.values.len(),
            first_result.values.len(),
            "Second MASS output length mismatch"
        );
    }

    #[test]
    fn test_mass_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let high = &candles.high;
        let low = &candles.low;

        let period = 5;
        let params = MassParams {
            period: Some(period),
        };
        let input = MassInput::from_candles(&candles, "high", "low", params);
        let mass_result = mass(&input).expect("Failed to calculate MASS");

        assert_eq!(mass_result.values.len(), high.len(), "MASS length mismatch");

        if mass_result.values.len() > 240 {
            for i in 240..mass_result.values.len() {
                assert!(
                    !mass_result.values[i].is_nan(),
                    "Expected no NaN after index 240, but found NaN at index {}",
                    i
                );
            }
        }
    }
}
