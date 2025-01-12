use crate::indicators::moving_averages::ma::{ma, MaData};
use crate::utilities::data_loader::{source_type, Candles};
/// # Moving Average Bands (MAB)
///
/// Calculates three bands (upper, middle, lower) based on a fast and a slow moving average,
/// plus the standard deviation of their difference over the fast period.
///
/// This replicates the logic of the Python reference, which computes:
/// - `fast_ma = ma(source, fast_period)`
/// - `slow_ma = ma(source, slow_period)`
/// - `sqAvg = rolling_sum((fast_ma - slow_ma)^2, fast_period) / fast_period`
/// - `dev = sqrt(sqAvg)`
/// - `upper_band = slow_ma + devup * dev`
/// - `middle_band = fast_ma`
/// - `lower_band = slow_ma - devdn * dev`
///
/// Like the SMA indicator, MAB will begin producing valid output only after the first valid
/// non-`NaN` data point, and once enough data points have accumulated to fill the required
/// `fast_period` and `slow_period` windows.
///
/// ## Parameters
/// - **fast_period**: Window size for the fast moving average and for the standard deviation
///   computation. Defaults to 10.
/// - **slow_period**: Window size for the slow moving average. Defaults to 50.
/// - **devup**: Multiplier for the deviation added to the slow moving average for the upper band.
///   Defaults to 1.0.
/// - **devdn**: Multiplier for the deviation subtracted from the slow moving average for the lower
///   band. Defaults to 1.0.
/// - **fast_ma_type**: Type of fast moving average (e.g., `"ema"`, `"sma"`). Defaults to `"ema"`.
/// - **slow_ma_type**: Type of slow moving average (e.g., `"ema"`, `"sma"`). Defaults to `"ema"`.
///
/// ## Errors
/// - **EmptyData**: mab: Input data slice is empty.
/// - **InvalidPeriod**: mab: A period is zero or exceeds the data length.
/// - **NotEnoughValidData**: mab: Fewer than `max(fast_period, slow_period)` valid data points remain
///   after the first valid index.
/// - **AllValuesNaN**: mab: All input data values are `NaN`.
///
/// ## Returns
/// - **`Ok(MabOutput)`** on success, containing three `Vec<f64>` (`upperband`, `middleband`,
///   `lowerband`) each matching the input length, with leading `NaN`s until the fast and slow
///   windows are filled.
/// - **`Err(MabError)`** otherwise.
use std::error::Error;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum MabData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct MabOutput {
    pub upperband: Vec<f64>,
    pub middleband: Vec<f64>,
    pub lowerband: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct MabParams {
    pub fast_period: Option<usize>,
    pub slow_period: Option<usize>,
    pub devup: Option<f64>,
    pub devdn: Option<f64>,
    pub fast_ma_type: Option<String>,
    pub slow_ma_type: Option<String>,
}

impl Default for MabParams {
    fn default() -> Self {
        Self {
            fast_period: Some(10),
            slow_period: Some(50),
            devup: Some(1.0),
            devdn: Some(1.0),
            fast_ma_type: Some("sma".to_string()),
            slow_ma_type: Some("sma".to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MabInput<'a> {
    pub data: MabData<'a>,
    pub params: MabParams,
}

impl<'a> MabInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: MabParams) -> Self {
        Self {
            data: MabData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: MabParams) -> Self {
        Self {
            data: MabData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: MabData::Candles {
                candles,
                source: "close",
            },
            params: MabParams::default(),
        }
    }

    pub fn get_fast_period(&self) -> usize {
        self.params
            .fast_period
            .unwrap_or_else(|| MabParams::default().fast_period.unwrap())
    }

    pub fn get_slow_period(&self) -> usize {
        self.params
            .slow_period
            .unwrap_or_else(|| MabParams::default().slow_period.unwrap())
    }

    pub fn get_devup(&self) -> f64 {
        self.params
            .devup
            .unwrap_or_else(|| MabParams::default().devup.unwrap())
    }

    pub fn get_devdn(&self) -> f64 {
        self.params
            .devdn
            .unwrap_or_else(|| MabParams::default().devdn.unwrap())
    }

    pub fn get_fast_ma_type(&self) -> String {
        self.params
            .fast_ma_type
            .as_ref()
            .unwrap_or(&MabParams::default().fast_ma_type.as_ref().unwrap())
            .clone()
    }

    pub fn get_slow_ma_type(&self) -> String {
        self.params
            .slow_ma_type
            .as_ref()
            .unwrap_or(&MabParams::default().slow_ma_type.as_ref().unwrap())
            .clone()
    }
}

#[derive(Debug, Error)]
pub enum MabError {
    #[error("mab: Empty data provided.")]
    EmptyData,
    #[error(
        "mab: Invalid period: fast = {fast_period}, slow = {slow_period}, data length = {data_len}"
    )]
    InvalidPeriod {
        fast_period: usize,
        slow_period: usize,
        data_len: usize,
    },
    #[error(
        "mab: Not enough valid data after first valid index: needed = {needed}, valid = {valid}"
    )]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("mab: All values are NaN.")]
    AllValuesNaN,
    #[error("mab: Underlying MA calculation failed: {0}")]
    MaCalculationError(String),
}

#[inline]
pub fn mab(input: &MabInput) -> Result<MabOutput, MabError> {
    let data: &[f64] = match &input.data {
        MabData::Candles { candles, source } => source_type(candles, source),
        MabData::Slice(slice) => slice,
    };

    if data.is_empty() {
        return Err(MabError::EmptyData);
    }

    let fast_period = input.get_fast_period();
    let slow_period = input.get_slow_period();
    let devup = input.get_devup();
    let devdn = input.get_devdn();
    let fast_ma_type = input.get_fast_ma_type();
    let slow_ma_type = input.get_slow_ma_type();

    if fast_period == 0 || slow_period == 0 || fast_period > data.len() || slow_period > data.len()
    {
        return Err(MabError::InvalidPeriod {
            fast_period,
            slow_period,
            data_len: data.len(),
        });
    }

    let any_non_nan = data.iter().any(|&x| !x.is_nan());
    if !any_non_nan {
        return Err(MabError::AllValuesNaN);
    }

    let fast_ma = ma(&fast_ma_type, MaData::Slice(data), fast_period)
        .map_err(|e| MabError::MaCalculationError(format!("{:?}", e)))?;
    let slow_ma = ma(&slow_ma_type, MaData::Slice(data), slow_period)
        .map_err(|e| MabError::MaCalculationError(format!("{:?}", e)))?;

    let fv_fast = match fast_ma.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(MabError::AllValuesNaN),
    };
    let fv_slow = match slow_ma.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(MabError::AllValuesNaN),
    };
    let second_valid_idx = fv_fast.max(fv_slow);
    let needed = usize::max(fast_period, slow_period);
    if (data.len() - second_valid_idx) < needed {
        return Err(MabError::NotEnoughValidData {
            needed,
            valid: data.len() - second_valid_idx,
        });
    }

    let mut upperband = vec![f64::NAN; data.len()];
    let mut middleband = vec![f64::NAN; data.len()];
    let mut lowerband = vec![f64::NAN; data.len()];

    let mut sum_sq_diff = 0.0;
    for i in second_valid_idx..second_valid_idx + fast_period {
        let diff = fast_ma[i] - slow_ma[i];
        sum_sq_diff += diff * diff;
    }

    let mut dev = (sum_sq_diff / fast_period as f64).sqrt();
    let index_for_first_dev = second_valid_idx + fast_period - 1;
    middleband[index_for_first_dev] = fast_ma[index_for_first_dev];
    upperband[index_for_first_dev] = slow_ma[index_for_first_dev] + devup * dev;
    lowerband[index_for_first_dev] = slow_ma[index_for_first_dev] - devdn * dev;

    for i in (second_valid_idx + fast_period)..data.len() {
        let old_diff = fast_ma[i - fast_period] - slow_ma[i - fast_period];
        let new_diff = fast_ma[i] - slow_ma[i];
        sum_sq_diff += new_diff * new_diff - old_diff * old_diff;
        dev = (sum_sq_diff / fast_period as f64).sqrt();

        middleband[i] = fast_ma[i];
        upperband[i] = slow_ma[i] + devup * dev;
        lowerband[i] = slow_ma[i] - devdn * dev;
    }

    Ok(MabOutput {
        upperband,
        middleband,
        lowerband,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_mab_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let default_params = MabParams {
            fast_period: None,
            ..MabParams::default()
        };
        let input_default = MabInput::from_candles(&candles, "close", default_params);
        let output_default = mab(&input_default).expect("Failed MAB with partial params");
        assert_eq!(output_default.upperband.len(), candles.close.len());
        assert_eq!(output_default.middleband.len(), candles.close.len());
        assert_eq!(output_default.lowerband.len(), candles.close.len());
    }

    #[test]
    fn test_mab_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = MabParams {
            fast_period: Some(10),
            slow_period: Some(50),
            devup: Some(1.0),
            devdn: Some(1.0),
            fast_ma_type: Some("sma".to_string()),
            slow_ma_type: Some("sma".to_string()),
        };
        let input = MabInput::from_candles(&candles, "close", params);
        let mab_result = mab(&input).expect("Failed to calculate MAB");

        assert_eq!(mab_result.upperband.len(), candles.close.len());
        assert_eq!(mab_result.middleband.len(), candles.close.len());
        assert_eq!(mab_result.lowerband.len(), candles.close.len());

        let expected_upper_last_five = [
            64002.843463352016,
            63976.62699738246,
            63949.00496307154,
            63912.13708526151,
            63828.40371728143,
        ];
        let expected_middle_last_five = [
            59213.90000000002,
            59180.800000000025,
            59161.40000000002,
            59132.00000000002,
            59042.40000000002,
        ];
        let expected_lower_last_five = [
            59350.676536647945,
            59296.93300261751,
            59252.75503692843,
            59190.30291473845,
            59070.11628271853,
        ];

        assert!(mab_result.upperband.len() >= 5);
        let start_index = mab_result.upperband.len() - 5;
        for i in 0..5 {
            let got_upper = mab_result.upperband[start_index + i];
            let got_middle = mab_result.middleband[start_index + i];
            let got_lower = mab_result.lowerband[start_index + i];

            let exp_upper = expected_upper_last_five[i];
            let exp_middle = expected_middle_last_five[i];
            let exp_lower = expected_lower_last_five[i];

            assert!(
                (got_upper - exp_upper).abs() < 1e-4,
                "MAB upper mismatch at index {}: expected {}, got {}",
                i,
                exp_upper,
                got_upper
            );
            assert!(
                (got_middle - exp_middle).abs() < 1e-4,
                "MAB middle mismatch at index {}: expected {}, got {}",
                i,
                exp_middle,
                got_middle
            );
            assert!(
                (got_lower - exp_lower).abs() < 1e-4,
                "MAB lower mismatch at index {}: expected {}, got {}",
                i,
                exp_lower,
                got_lower
            );
        }
    }

    #[test]
    fn test_mab_with_zero_period() {
        let input_data = [10.0, 20.0, 30.0];
        let params = MabParams {
            fast_period: Some(0),
            slow_period: Some(5),
            ..MabParams::default()
        };
        let input = MabInput::from_slice(&input_data, params);

        let result = mab(&input);
        assert!(result.is_err(), "Expected an error for zero fast period");
        if let Err(e) = result {
            assert!(
                e.to_string().contains("Invalid period"),
                "Expected 'Invalid period' error message, got: {}",
                e
            );
        }
    }

    #[test]
    fn test_mab_with_period_exceeding_data_length() {
        let input_data = [10.0, 20.0, 30.0];
        let params = MabParams {
            fast_period: Some(2),
            slow_period: Some(10),
            ..MabParams::default()
        };
        let input = MabInput::from_slice(&input_data, params);

        let result = mab(&input);
        assert!(
            result.is_err(),
            "Expected an error for slow period > data.len()"
        );
    }

    #[test]
    fn test_mab_very_small_data_set() {
        let input_data = [42.0];
        let params = MabParams {
            fast_period: Some(10),
            slow_period: Some(20),
            ..MabParams::default()
        };
        let input = MabInput::from_slice(&input_data, params);

        let result = mab(&input);
        assert!(
            result.is_err(),
            "Expected error for data smaller than required period"
        );
    }

    #[test]
    fn test_mab_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let first_params = MabParams {
            fast_period: Some(10),
            slow_period: Some(50),
            ..MabParams::default()
        };
        let first_input = MabInput::from_candles(&candles, "close", first_params);
        let first_result = mab(&first_input).expect("Failed to calculate first MAB");

        assert_eq!(
            first_result.upperband.len(),
            candles.close.len(),
            "First MAB output length mismatch"
        );

        let second_params = MabParams {
            fast_period: Some(10),
            slow_period: Some(50),
            ..MabParams::default()
        };
        let second_input = MabInput::from_slice(&first_result.upperband, second_params);
        let second_result = mab(&second_input).expect("Failed to calculate second MAB");

        assert_eq!(
            second_result.upperband.len(),
            first_result.upperband.len(),
            "Second MAB output length mismatch"
        );

        for (i, &val) in second_result.upperband.iter().enumerate() {
            if i > 240 {
                assert!(
                    !val.is_nan(),
                    "Expected no NaN after index 100, but found NaN at index {}",
                    i
                );
            }
        }
    }

    #[test]
    fn test_mab_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = &candles.close;

        let params = MabParams {
            fast_period: Some(10),
            slow_period: Some(50),
            ..MabParams::default()
        };
        let input = MabInput::from_candles(&candles, "close", params);
        let mab_result = mab(&input).expect("Failed to calculate MAB");

        assert_eq!(mab_result.upperband.len(), close_prices.len());
        assert_eq!(mab_result.middleband.len(), close_prices.len());
        assert_eq!(mab_result.lowerband.len(), close_prices.len());

        if mab_result.upperband.len() > 300 {
            for i in 300..mab_result.upperband.len() {
                assert!(
                    !mab_result.upperband[i].is_nan(),
                    "Expected no NaN in upperband after index 300, found at {}",
                    i
                );
                assert!(
                    !mab_result.middleband[i].is_nan(),
                    "Expected no NaN in middleband after index 300, found at {}",
                    i
                );
                assert!(
                    !mab_result.lowerband[i].is_nan(),
                    "Expected no NaN in lowerband after index 300, found at {}",
                    i
                );
            }
        }
    }

    #[test]
    fn test_mab_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = MabInput::with_default_candles(&candles);
        match input.data {
            MabData::Candles { source, .. } => {
                assert_eq!(source, "close", "Expected default source to be 'close'");
            }
            _ => panic!("Expected MabData::Candles variant"),
        }
    }
}
