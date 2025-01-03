/// # Commodity Channel Index (CCI)
///
/// Commodity Channel Index is typically calculated as:
///
/// ```text
/// CCI_t = (price_t - SMA(price, period)) / (0.015 * MeanAbsoluteDeviation(price, period))
/// ```
///
/// where `price_t` is often the "typical price" ((H+L+C)/3), but it can be any data source.
///
/// ## Parameters
/// - **period**: Window size (number of data points). Defaults to 14.
///  
/// ## Errors
/// - **EmptyData**: cci: Input data slice is empty.
/// - **InvalidPeriod**: cci: `period` is zero or exceeds the data length.
/// - **NotEnoughValidData**: cci: Fewer than `period` valid (non-`NaN`) data points remain
///   after the first valid index.
/// - **AllValuesNaN**: cci: All input data values are `NaN`.
///
/// ## Returns
/// - **`Ok(CciOutput)`** on success, containing a `Vec<f64>` matching the input length,
///   with leading `NaN`s until the moving average window is filled.
/// - **`Err(CciError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum CciData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct CciOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct CciParams {
    pub period: Option<usize>,
}

impl Default for CciParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct CciInput<'a> {
    pub data: CciData<'a>,
    pub params: CciParams,
}

impl<'a> CciInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: CciParams) -> Self {
        Self {
            data: CciData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: CciParams) -> Self {
        Self {
            data: CciData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: CciData::Candles {
                candles,
                source: "hlc3",
            },
            params: CciParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| CciParams::default().period.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum CciError {
    #[error("cci: Empty data provided.")]
    EmptyData,
    #[error("cci: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("cci: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("cci: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn cci(input: &CciInput) -> Result<CciOutput, CciError> {
    let data: &[f64] = match &input.data {
        CciData::Candles { candles, source } => source_type(candles, source),
        CciData::Slice(slice) => slice,
    };

    if data.is_empty() {
        return Err(CciError::EmptyData);
    }

    let period = input.get_period();
    if period == 0 || period > data.len() {
        return Err(CciError::InvalidPeriod {
            period,
            data_len: data.len(),
        });
    }

    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(CciError::AllValuesNaN),
    };

    if (data.len() - first_valid_idx) < period {
        return Err(CciError::NotEnoughValidData {
            needed: period,
            valid: data.len() - first_valid_idx,
        });
    }

    let mut cci_values = vec![f64::NAN; data.len()];
    let inv_period = 1.0 / (period as f64);

    let mut sum = 0.0;
    for &val in &data[first_valid_idx..(first_valid_idx + period)] {
        sum += val;
    }
    let mut current_sma = sum * inv_period;

    let mut sum_abs_dev = 0.0;
    for &val in &data[first_valid_idx..(first_valid_idx + period)] {
        sum_abs_dev += (val - current_sma).abs();
    }

    let first_output_idx = first_valid_idx + period - 1;
    let first_price = data[first_output_idx];
    cci_values[first_output_idx] = if sum_abs_dev == 0.0 {
        0.0
    } else {
        (first_price - current_sma) / (0.015 * (sum_abs_dev * inv_period))
    };

    for i in (first_output_idx + 1)..data.len() {
        let exiting_val = data[i - period];
        let entering_val = data[i];

        sum = sum - exiting_val + entering_val;
        current_sma = sum * inv_period;

        sum_abs_dev = 0.0;
        for &val in &data[(i - period + 1)..=i] {
            sum_abs_dev += (val - current_sma).abs();
        }

        cci_values[i] = if sum_abs_dev == 0.0 {
            0.0
        } else {
            (entering_val - current_sma) / (0.015 * (sum_abs_dev * inv_period))
        };
    }

    Ok(CciOutput { values: cci_values })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_cci_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = CciParams { period: None };
        let input_default = CciInput::from_candles(&candles, "close", default_params);
        let output_default = cci(&input_default).expect("Failed CCI with default params");
        assert_eq!(output_default.values.len(), candles.close.len());

        let params_20 = CciParams { period: Some(20) };
        let input_20 = CciInput::from_candles(&candles, "hl2", params_20);
        let output_20 = cci(&input_20).expect("Failed CCI with period=20, source=hl2");
        assert_eq!(output_20.values.len(), candles.close.len());

        let params_custom = CciParams { period: Some(9) };
        let input_custom = CciInput::from_candles(&candles, "hlc3", params_custom);
        let output_custom = cci(&input_custom).expect("Failed CCI fully custom");
        assert_eq!(output_custom.values.len(), candles.close.len());
    }

    #[test]
    fn test_cci_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = CciInput::with_default_candles(&candles);
        let cci_result = cci(&input).expect("Failed to calculate CCI");

        assert_eq!(
            cci_result.values.len(),
            candles.close.len(),
            "CCI length mismatch"
        );

        let expected_last_five_cci = [
            -51.55252564125841,
            -43.50326506381541,
            -64.05117302269149,
            -39.05150631680948,
            -152.50523930896998,
        ];

        assert!(
            cci_result.values.len() >= 5,
            "CCI length is too short to test last 5 values"
        );

        let start_idx = cci_result.values.len() - 5;
        let last_five_cci = &cci_result.values[start_idx..];
        for (i, &value) in last_five_cci.iter().enumerate() {
            let expected = expected_last_five_cci[i];
            assert!(
                (value - expected).abs() < 1e-6,
                "CCI mismatch at last five index {}: expected {}, got {}",
                i,
                expected,
                value
            );
        }

        let period: usize = input.get_period();
        for i in 0..(period - 1) {
            assert!(
                cci_result.values[i].is_nan(),
                "Expected NaN at index {} for initial period warm-up",
                i
            );
        }
    }

    #[test]
    fn test_cci_params_with_default_params() {
        let default_params = CciParams::default();
        assert_eq!(
            default_params.period,
            Some(14),
            "Expected default period to be 14"
        );
    }

    #[test]
    fn test_cci_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = CciInput::with_default_candles(&candles);

        match input.data {
            CciData::Candles { source, .. } => {
                assert_eq!(source, "hlc3", "Expected default source to be 'hlc3'");
            }
            _ => panic!("Expected CciData::Candles variant"),
        }
    }

    #[test]
    fn test_cci_with_zero_period() {
        let input_data = [10.0, 20.0, 30.0];
        let params = CciParams { period: Some(0) };
        let input = CciInput::from_slice(&input_data, params);
        let result = cci(&input);

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
    fn test_cci_with_period_exceeding_data_length() {
        let input_data = [10.0, 20.0, 30.0];
        let params = CciParams { period: Some(10) };
        let input = CciInput::from_slice(&input_data, params);

        let result = cci(&input);
        assert!(result.is_err(), "Expected an error for period > data.len()");
    }

    #[test]
    fn test_cci_very_small_data_set() {
        let input_data = [42.0];
        let params = CciParams { period: Some(9) };
        let input = CciInput::from_slice(&input_data, params);

        let result = cci(&input);
        assert!(
            result.is_err(),
            "Expected error for data smaller than period"
        );
    }

    #[test]
    fn test_cci_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let first_params = CciParams { period: Some(14) };
        let first_input = CciInput::from_candles(&candles, "close", first_params);
        let first_result = cci(&first_input).expect("Failed to calculate first CCI");
        assert_eq!(
            first_result.values.len(),
            candles.close.len(),
            "First CCI output length mismatch"
        );

        let second_params = CciParams { period: Some(14) };
        let second_input = CciInput::from_slice(&first_result.values, second_params);
        let second_result = cci(&second_input).expect("Failed to calculate second CCI");
        assert_eq!(
            second_result.values.len(),
            first_result.values.len(),
            "Second CCI output length mismatch"
        );

        if second_result.values.len() > 28 {
            for i in 28..second_result.values.len() {
                assert!(
                    !second_result.values[i].is_nan(),
                    "Expected no NaN after index 28, found NaN at index {}",
                    i
                );
            }
        }
    }

    #[test]
    fn test_cci_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = &candles.close;

        let period = 14;
        let params = CciParams {
            period: Some(period),
        };
        let input = CciInput::from_candles(&candles, "close", params);
        let cci_result = cci(&input).expect("Failed to calculate CCI");
        assert_eq!(
            cci_result.values.len(),
            close_prices.len(),
            "CCI length mismatch"
        );

        if cci_result.values.len() > 240 {
            for i in 240..cci_result.values.len() {
                assert!(
                    !cci_result.values[i].is_nan(),
                    "Expected no NaN after index 240, found NaN at index {}",
                    i
                );
            }
        }
    }
}
