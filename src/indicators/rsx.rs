/// # Relative Strength Xtra (RSX)
///
/// A smoothed oscillator similar to RSI that attempts to reduce lag and noise.
/// The calculation uses an IIR filter approach for smoothing while retaining
/// responsiveness.
///
/// ## Parameters
/// - **period**: The lookback window for RSX calculations. Defaults to 14.
///
/// ## Errors
/// - **EmptyData**: rsx: Input data slice is empty.
/// - **InvalidPeriod**: rsx: `period` is zero or exceeds the data length.
/// - **NotEnoughValidData**: rsx: Fewer than `period` valid (non-`NaN`) data points remain
///   after the first valid index.
/// - **AllValuesNaN**: rsx: All input data values are `NaN`.
///
/// ## Returns
/// - **`Ok(RsxOutput)`** on success, containing a `Vec<f64>` matching the input length,
///   with leading `NaN`s until the RSX calculation window is filled.
/// - **`Err(RsxError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum RsxData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct RsxOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct RsxParams {
    pub period: Option<usize>,
}

impl Default for RsxParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct RsxInput<'a> {
    pub data: RsxData<'a>,
    pub params: RsxParams,
}

impl<'a> RsxInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: RsxParams) -> Self {
        Self {
            data: RsxData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: RsxParams) -> Self {
        Self {
            data: RsxData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: RsxData::Candles {
                candles,
                source: "close",
            },
            params: RsxParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| RsxParams::default().period.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum RsxError {
    #[error("rsx: Empty data provided for RSX.")]
    EmptyData,
    #[error("rsx: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("rsx: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("rsx: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn rsx(input: &RsxInput) -> Result<RsxOutput, RsxError> {
    let data: &[f64] = match &input.data {
        RsxData::Candles { candles, source } => source_type(candles, source),
        RsxData::Slice(slice) => slice,
    };

    if data.is_empty() {
        return Err(RsxError::EmptyData);
    }

    let period = input.get_period();
    if period == 0 || period > data.len() {
        return Err(RsxError::InvalidPeriod {
            period,
            data_len: data.len(),
        });
    }

    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(RsxError::AllValuesNaN),
    };

    if (data.len() - first_valid_idx) < period {
        return Err(RsxError::NotEnoughValidData {
            needed: period,
            valid: data.len() - first_valid_idx,
        });
    }

    let mut rsx_values = vec![f64::NAN; data.len()];
    let start_calc_idx = first_valid_idx + period - 1;

    let mut f0 = 0.0;
    let mut f8 = 0.0;
    let mut f18 = 0.0;
    let mut f20 = 0.0;
    let mut f28 = 0.0;
    let mut f30 = 0.0;
    let mut f38 = 0.0;
    let mut f40 = 0.0;
    let mut f48 = 0.0;
    let mut f50 = 0.0;
    let mut f58 = 0.0;
    let mut f60 = 0.0;
    let mut f68 = 0.0;
    let mut f70 = 0.0;
    let mut f78 = 0.0;
    let mut f80 = 0.0;
    let mut f88 = 0.0;
    let mut f90 = 0.0;
    let mut is_initialized = false;

    for i in start_calc_idx..data.len() {
        let val = data[i];
        if !is_initialized {
            f90 = 1.0;
            f0 = 0.0;
            f88 = if period >= 6 {
                (period - 1) as f64
            } else {
                5.0
            };
            f8 = 100.0 * val;
            f18 = 3.0 / (period as f64 + 2.0);
            f20 = 1.0 - f18;
            rsx_values[i] = f64::NAN;
            is_initialized = true;
        } else {
            f90 = if f88 <= f90 { f88 + 1.0 } else { f90 + 1.0 };
            let f10 = f8;
            f8 = 100.0 * val;
            let v8 = f8 - f10;
            f28 = f20 * f28 + f18 * v8;
            f30 = f18 * f28 + f20 * f30;
            let v_c = f28 * 1.5 - f30 * 0.5;
            f38 = f20 * f38 + f18 * v_c;
            f40 = f18 * f38 + f20 * f40;
            let v10 = f38 * 1.5 - f40 * 0.5;
            f48 = f20 * f48 + f18 * v10;
            f50 = f18 * f48 + f20 * f50;
            let v14 = f48 * 1.5 - f50 * 0.5;
            f58 = f20 * f58 + f18 * v8.abs();
            f60 = f18 * f58 + f20 * f60;
            let v18 = f58 * 1.5 - f60 * 0.5;
            f68 = f20 * f68 + f18 * v18;
            f70 = f18 * f68 + f20 * f70;
            let v1c = f68 * 1.5 - f70 * 0.5;
            f78 = f20 * f78 + f18 * v1c;
            f80 = f18 * f78 + f20 * f80;
            let v20_ = f78 * 1.5 - f80 * 0.5;

            if f88 >= f90 && f8 != f10 {
                f0 = 1.0;
            }
            if (f88 - f90).abs() < f64::EPSILON && f0 == 0.0 {
                f90 = 0.0;
            }

            if f88 < f90 && v20_ > 1e-10 {
                let mut v4 = (v14 / v20_ + 1.0) * 50.0;
                if v4 > 100.0 {
                    v4 = 100.0;
                }
                if v4 < 0.0 {
                    v4 = 0.0;
                }
                rsx_values[i] = v4;
            } else {
                rsx_values[i] = 50.0;
            }
        }
    }

    Ok(RsxOutput { values: rsx_values })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_rsx_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = RsxParams { period: None };
        let input_default = RsxInput::from_candles(&candles, "close", default_params);
        let output_default = rsx(&input_default).expect("Failed RSX with default params");
        assert_eq!(output_default.values.len(), candles.close.len());

        let params_period_10 = RsxParams { period: Some(10) };
        let input_period_10 = RsxInput::from_candles(&candles, "hl2", params_period_10);
        let output_period_10 =
            rsx(&input_period_10).expect("Failed RSX with period=10, source=hl2");
        assert_eq!(output_period_10.values.len(), candles.close.len());

        let params_custom = RsxParams { period: Some(20) };
        let input_custom = RsxInput::from_candles(&candles, "hlc3", params_custom);
        let output_custom = rsx(&input_custom).expect("Failed RSX fully custom");
        assert_eq!(output_custom.values.len(), candles.close.len());
    }

    #[test]
    fn test_rsx_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let params = RsxParams { period: Some(14) };
        let input = RsxInput::from_candles(&candles, "close", params);
        let rsx_result = rsx(&input).expect("Failed to calculate RSX");

        assert_eq!(
            rsx_result.values.len(),
            close_prices.len(),
            "RSX length mismatch"
        );

        let expected_last_five_rsx = [
            46.11486311289701,
            46.88048640321688,
            47.174443049619995,
            47.48751360654475,
            46.552886446171684,
        ];
        assert!(rsx_result.values.len() >= 5, "RSX length too short");
        let start_index = rsx_result.values.len() - 5;
        let result_last_five_rsx = &rsx_result.values[start_index..];
        for (i, &value) in result_last_five_rsx.iter().enumerate() {
            let expected_value = expected_last_five_rsx[i];
            assert!(
                (value - expected_value).abs() < 1e-1,
                "RSX mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }
    }

    #[test]
    fn test_rsx_params_with_default_params() {
        let default_params = RsxParams::default();
        assert_eq!(
            default_params.period,
            Some(14),
            "Expected period to be Some(14) in default parameters"
        );
    }

    #[test]
    fn test_rsx_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = RsxInput::with_default_candles(&candles);
        match input.data {
            RsxData::Candles { source, .. } => {
                assert_eq!(source, "close", "Expected default source to be 'close'");
            }
            _ => panic!("Expected RsxData::Candles variant"),
        }
    }

    #[test]
    fn test_rsx_with_zero_period() {
        let input_data = [10.0, 20.0, 30.0];
        let params = RsxParams { period: Some(0) };
        let input = RsxInput::from_slice(&input_data, params);

        let result = rsx(&input);
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
    fn test_rsx_with_period_exceeding_data_length() {
        let input_data = [10.0, 20.0, 30.0];
        let params = RsxParams { period: Some(10) };
        let input = RsxInput::from_slice(&input_data, params);

        let result = rsx(&input);
        assert!(result.is_err(), "Expected an error for period > data.len()");
    }

    #[test]
    fn test_rsx_very_small_data_set() {
        let input_data = [42.0];
        let params = RsxParams { period: Some(14) };
        let input = RsxInput::from_slice(&input_data, params);

        let result = rsx(&input);
        assert!(
            result.is_err(),
            "Expected error for data smaller than period"
        );
    }

    #[test]
    fn test_rsx_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let first_params = RsxParams { period: Some(14) };
        let first_input = RsxInput::from_candles(&candles, "close", first_params);
        let first_result = rsx(&first_input).expect("Failed to calculate first RSX");

        assert_eq!(
            first_result.values.len(),
            candles.close.len(),
            "First RSX output length mismatch"
        );

        let second_params = RsxParams { period: Some(14) };
        let second_input = RsxInput::from_slice(&first_result.values, second_params);
        let second_result = rsx(&second_input).expect("Failed to calculate second RSX");

        assert_eq!(
            second_result.values.len(),
            first_result.values.len(),
            "Second RSX output length mismatch"
        );
    }

    #[test]
    fn test_rsx_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = &candles.close;

        let period = 14;
        let params = RsxParams {
            period: Some(period),
        };
        let input = RsxInput::from_candles(&candles, "close", params);
        let rsx_result = rsx(&input).expect("Failed to calculate RSX");

        assert_eq!(
            rsx_result.values.len(),
            close_prices.len(),
            "RSX length mismatch"
        );

        if rsx_result.values.len() > 50 {
            for i in 50..rsx_result.values.len() {
                assert!(
                    !rsx_result.values[i].is_nan(),
                    "Expected no NaN after index 50, but found NaN at index {}",
                    i
                );
            }
        }
    }
}
