/// # Inverse Fisher Transform RSI (IFT RSI)
///
/// This indicator applies the Inverse Fisher Transform to an RSI sequence,
/// optionally smoothed by a WMA. It calculates the RSI of the chosen `source`
/// with a given `rsi_period`, transforms it by `v1 = 0.1 * (RSI - 50)`, then
/// applies a WMA (period = `wma_period`) to `v1`, and finally performs:
/// `IFT = ( (2*v2)^2 - 1 ) / ( (2*v2)^2 + 1 )` where `v2` is the WMA output.
///
/// ## Parameters
/// - **rsi_period**: Defaults to 5.
/// - **wma_period**: Defaults to 9.
///
/// ## Errors
/// - **EmptyData**: if no data is provided.
/// - **AllValuesNaN**: if all input data is `NaN`.
/// - **InvalidPeriod**: if `rsi_period` or `wma_period` is 0 or exceeds data length.
/// - **NotEnoughValidData**: if fewer valid points remain (after the first non-`NaN`) than
///   the required maximum of `rsi_period` or `wma_period`.
/// - **RsiCalculationError**: if the RSI calculation fails internally.
/// - **WmaCalculationError**: if the WMA calculation fails internally.
///
/// ## Returns
/// - **`Ok(IftRsiOutput)`** on success, containing a `Vec<f64>` with the same length as the input,
///   filled with `NaN` values until the calculation becomes valid.
/// - **`Err(IftRsiError)`** otherwise.
use crate::indicators::rsi::{rsi, RsiError, RsiInput, RsiOutput, RsiParams};
use crate::indicators::wma::{wma, WmaError, WmaInput, WmaOutput, WmaParams};
use crate::utilities::data_loader::{source_type, Candles};

#[derive(Debug, Clone)]
pub enum IftRsiData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct IftRsiParams {
    pub rsi_period: Option<usize>,
    pub wma_period: Option<usize>,
}

impl Default for IftRsiParams {
    fn default() -> Self {
        Self {
            rsi_period: Some(5),
            wma_period: Some(9),
        }
    }
}

#[derive(Debug, Clone)]
pub struct IftRsiInput<'a> {
    pub data: IftRsiData<'a>,
    pub params: IftRsiParams,
}

impl<'a> IftRsiInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: IftRsiParams) -> Self {
        Self {
            data: IftRsiData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: IftRsiParams) -> Self {
        Self {
            data: IftRsiData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: IftRsiData::Candles {
                candles,
                source: "close",
            },
            params: IftRsiParams::default(),
        }
    }

    pub fn get_rsi_period(&self) -> usize {
        self.params
            .rsi_period
            .unwrap_or_else(|| IftRsiParams::default().rsi_period.unwrap())
    }

    pub fn get_wma_period(&self) -> usize {
        self.params
            .wma_period
            .unwrap_or_else(|| IftRsiParams::default().wma_period.unwrap())
    }
}

#[derive(Debug, Clone)]
pub struct IftRsiOutput {
    pub values: Vec<f64>,
}

use thiserror::Error;

#[derive(Debug, Error)]
pub enum IftRsiError {
    #[error("ift_rsi: No data provided.")]
    EmptyData,
    #[error("ift_rsi: All values are NaN.")]
    AllValuesNaN,
    #[error("ift_rsi: Invalid RSI period {rsi_period} or WMA period {wma_period}, data length = {data_len}.")]
    InvalidPeriod {
        rsi_period: usize,
        wma_period: usize,
        data_len: usize,
    },
    #[error("ift_rsi: Not enough valid data. Needed = {needed}, valid = {valid}.")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("ift_rsi: RSI calculation error: {0}")]
    RsiCalculationError(String),
    #[error("ift_rsi: WMA calculation error: {0}")]
    WmaCalculationError(String),
}

#[inline]
pub fn ift_rsi(input: &IftRsiInput) -> Result<IftRsiOutput, IftRsiError> {
    let data: &[f64] = match &input.data {
        IftRsiData::Candles { candles, source } => source_type(candles, source),
        IftRsiData::Slice(slice) => slice,
    };

    if data.is_empty() {
        return Err(IftRsiError::EmptyData);
    }

    let rsi_period = input.get_rsi_period();
    let wma_period = input.get_wma_period();

    if rsi_period == 0 || wma_period == 0 || rsi_period > data.len() || wma_period > data.len() {
        return Err(IftRsiError::InvalidPeriod {
            rsi_period,
            wma_period,
            data_len: data.len(),
        });
    }

    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(IftRsiError::AllValuesNaN),
    };

    let needed = rsi_period.max(wma_period);
    let valid_points = data.len() - first_valid_idx;
    if valid_points < needed {
        return Err(IftRsiError::NotEnoughValidData {
            needed,
            valid: valid_points,
        });
    }

    let sliced_data = &data[first_valid_idx..];

    let mut rsi_values = rsi(&RsiInput::from_slice(
        sliced_data,
        RsiParams {
            period: Some(rsi_period),
        },
    ))
    .map_err(|e| IftRsiError::RsiCalculationError(e.to_string()))?
    .values;

    for val in rsi_values.iter_mut() {
        if !val.is_nan() {
            *val = 0.1 * (*val - 50.0);
        }
    }

    let wma_values = wma(&WmaInput::from_slice(
        &rsi_values,
        WmaParams {
            period: Some(wma_period),
        },
    ))
    .map_err(|e| IftRsiError::WmaCalculationError(e.to_string()))?
    .values;

    let mut ift_values = vec![f64::NAN; data.len()];

    for (i, &w) in wma_values.iter().enumerate() {
        if !w.is_nan() {
            let two_w = 2.0 * w;
            let numerator = two_w * two_w - 1.0;
            let denominator = two_w * two_w + 1.0;
            ift_values[first_valid_idx + i] = numerator / denominator;
        }
    }

    Ok(IftRsiOutput { values: ift_values })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_ift_rsi_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let params = IftRsiParams {
            rsi_period: Some(5),
            wma_period: Some(9),
        };
        let input = IftRsiInput::from_candles(&candles, "close", params);
        let result = ift_rsi(&input).expect("Failed to calculate IFT RSI");

        assert_eq!(
            result.values.len(),
            close_prices.len(),
            "IFT RSI length mismatch"
        );

        let expected_last_five = [
            -0.27763026899967286,
            -0.367418234207824,
            -0.1650156844504996,
            -0.26631220621545837,
            0.28324385010826775,
        ];
        assert!(
            result.values.len() >= 5,
            "Expected at least 5 output values for IFT RSI"
        );
        let start_index = result.values.len() - 5;
        let last_five = &result.values[start_index..];
        for (i, &value) in last_five.iter().enumerate() {
            let diff = (value - expected_last_five[i]).abs();
            assert!(
                diff < 1e-8,
                "IFT RSI mismatch at index {}: expected {}, got {}",
                i,
                expected_last_five[i],
                value
            );
        }
    }

    #[test]
    fn test_ift_rsi_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = IftRsiInput::with_default_candles(&candles);
        let output = ift_rsi(&input).expect("IFT RSI calculation failed with default params");
        assert_eq!(output.values.len(), candles.close.len());
    }

    #[test]
    fn test_ift_rsi_with_zero_period() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let params = IftRsiParams {
            rsi_period: Some(0),
            wma_period: Some(9),
        };
        let input = IftRsiInput::from_candles(&candles, "close", params);
        let result = ift_rsi(&input);
        assert!(result.is_err(), "Expected error for zero RSI period");
        if let Err(e) = result {
            assert!(
                e.to_string().contains("Invalid RSI period"),
                "Expected 'Invalid period' error, got: {}",
                e
            );
        }
    }

    #[test]
    fn test_ift_rsi_with_period_exceeding_data_length() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let params = IftRsiParams {
            rsi_period: Some(99999),
            wma_period: Some(9),
        };
        let input = IftRsiInput::from_candles(&candles, "close", params);

        let result = ift_rsi(&input);
        assert!(
            result.is_err(),
            "Expected error for rsi_period exceeding data length"
        );
    }

    #[test]
    fn test_ift_rsi_very_small_data_set() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let mut candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        candles.close = vec![42.0];
        let params = IftRsiParams {
            rsi_period: Some(5),
            wma_period: Some(9),
        };
        let input = IftRsiInput::from_candles(&candles, "close", params);
        let result = ift_rsi(&input);
        assert!(
            result.is_err(),
            "Expected error for data smaller than period"
        );
    }

    #[test]
    fn test_ift_rsi_nan_only() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let mut candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        candles.close = vec![f64::NAN, f64::NAN];
        let params = IftRsiParams::default();
        let input = IftRsiInput::from_candles(&candles, "close", params);
        let result = ift_rsi(&input);
        assert!(result.is_err(), "Expected error for all-NaN input data");
    }

    #[test]
    fn test_ift_rsi_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let first_params = IftRsiParams {
            rsi_period: Some(5),
            wma_period: Some(9),
        };
        let first_input = IftRsiInput::from_candles(&candles, "close", first_params);
        let first_result = ift_rsi(&first_input).expect("Failed to calculate first IFT RSI");
        assert_eq!(first_result.values.len(), candles.close.len());

        let second_params = IftRsiParams {
            rsi_period: Some(5),
            wma_period: Some(9),
        };
        let second_input = IftRsiInput::from_slice(&first_result.values, second_params);
        let second_result = ift_rsi(&second_input).expect("Failed to calculate second IFT RSI");
        assert_eq!(second_result.values.len(), first_result.values.len());
    }
}
