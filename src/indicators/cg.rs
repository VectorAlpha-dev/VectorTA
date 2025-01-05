/// # Center of Gravity (CG)
///
/// The Center of Gravity (CG) indicator attempts to measure the "center" of prices
/// over a given window, sometimes used for smoothing or cycle analysis.  
///
/// Formula (based on your Python snippet):
///
/// ```ignore
/// for i in range(len(source)):
///     if i > period:
///         num = 0
///         denom = 0
///         for count in range(period):
///             price = source[i - count]
///             num += (1 + count) * price
///             denom += price
///         result = -num / denom
/// ```
///
/// *Leading values* will be `NaN` until at least `period` bars of valid (non-NaN) data are available.
///
/// ## Parameters
/// - **period**: The window size. Defaults to 10.
///
/// ## Errors
/// - **EmptyData**: cg: Input data slice is empty.
/// - **InvalidPeriod**: cg: `period` is zero or exceeds the data length.
/// - **AllValuesNaN**: cg: All input data values are `NaN`.
/// - **NotEnoughValidData**: cg: Fewer than `period` valid (non-`NaN`) data points remain
///   after the first valid index.
///
/// ## Returns
/// - **`Ok(CgOutput)`** on success, containing a `Vec<f64>` matching input length,
///   with leading `NaN` until the warm-up period is reached.
/// - **`Err(CgError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum CgData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct CgOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct CgParams {
    pub period: Option<usize>,
}

impl Default for CgParams {
    fn default() -> Self {
        Self { period: Some(10) }
    }
}

#[derive(Debug, Clone)]
pub struct CgInput<'a> {
    pub data: CgData<'a>,
    pub params: CgParams,
}

impl<'a> CgInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: CgParams) -> Self {
        Self {
            data: CgData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: CgParams) -> Self {
        Self {
            data: CgData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: CgData::Candles {
                candles,
                source: "close",
            },
            params: CgParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| CgParams::default().period.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum CgError {
    #[error("cg: Empty data provided for CG.")]
    EmptyData,
    #[error("cg: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("cg: All values are NaN.")]
    AllValuesNaN,
    #[error("cg: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

#[inline]
pub fn cg(input: &CgInput) -> Result<CgOutput, CgError> {
    let data: &[f64] = match &input.data {
        CgData::Candles { candles, source } => source_type(candles, source),
        CgData::Slice(slice) => slice,
    };

    if data.is_empty() {
        return Err(CgError::EmptyData);
    }
    let period = input.get_period();
    if period == 0 || period > data.len() {
        return Err(CgError::InvalidPeriod {
            period,
            data_len: data.len(),
        });
    }

    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(CgError::AllValuesNaN),
    };
    if (data.len() - first_valid_idx) < (period + 1) {
        return Err(CgError::NotEnoughValidData {
            needed: period + 1,
            valid: data.len() - first_valid_idx,
        });
    }

    let mut cg_values = vec![f64::NAN; data.len()];

    let offset = first_valid_idx + period;

    for i in offset..data.len() {
        let mut num = 0.0;
        let mut denom = 0.0;
        for count in 0..(period - 1) {
            let close = data[i - count];
            num += (1.0 + count as f64) * close;
            denom += close;
        }

        cg_values[i] = if denom.abs() > f64::EPSILON {
            -num / denom
        } else {
            0.0
        };
    }

    Ok(CgOutput { values: cg_values })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_cg_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let partial_params = CgParams { period: Some(12) };
        let input_partial = CgInput::from_candles(&candles, "close", partial_params);
        let output_partial = cg(&input_partial).expect("Failed CG with partial params (period=12)");
        assert_eq!(output_partial.values.len(), candles.close.len());
    }

    #[test]
    fn test_cg_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let params = CgParams { period: Some(10) };
        let input = CgInput::from_candles(&candles, "close", params);
        let result = cg(&input).expect("Failed CG default test");

        let expected_last_five = [
            -4.99905186931943,
            -4.998559827254377,
            -4.9970065675119555,
            -4.9928483984587295,
            -5.004210799262688,
        ];
        assert!(
            result.values.len() >= 5,
            "Not enough data for final 5-values check"
        );
        let start_idx = result.values.len() - 5;
        for (i, &exp) in expected_last_five.iter().enumerate() {
            let idx = start_idx + i;
            let got = result.values[idx];
            assert!(
                (got - exp).abs() < 1e-4,
                "Mismatch in CG at idx {}: expected={}, got={}",
                idx,
                exp,
                got
            );
        }
    }

    #[test]
    fn test_cg_params_default() {
        let default_params = CgParams::default();
        assert_eq!(
            default_params.period,
            Some(10),
            "Expected default period=10 for CG"
        );
    }

    #[test]
    fn test_cg_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = CgInput::with_default_candles(&candles);
        match input.data {
            CgData::Candles { source, .. } => {
                assert_eq!(
                    source, "close",
                    "Expected default source='close' for CGInput::with_default_candles()"
                );
            }
            _ => panic!("Expected CgData::Candles variant"),
        }
    }

    #[test]
    fn test_cg_zero_period() {
        let data = [1.0, 2.0, 3.0];
        let params = CgParams { period: Some(0) };
        let input = CgInput::from_slice(&data, params);

        let result = cg(&input);
        assert!(result.is_err(), "Expected error for zero period");
        if let Err(e) = result {
            assert!(
                e.to_string().contains("Invalid period"),
                "Expected 'Invalid period' error, got: {}",
                e
            );
        }
    }

    #[test]
    fn test_cg_period_exceeding_data() {
        let data = [10.0, 20.0, 30.0];
        let params = CgParams { period: Some(10) };
        let input = CgInput::from_slice(&data, params);

        let result = cg(&input);
        assert!(result.is_err(), "Expected error for period > data.len()");
    }

    #[test]
    fn test_cg_very_small_data() {
        let data = [42.0];
        let input = CgInput::from_slice(&data, CgParams::default());

        let result = cg(&input);
        assert!(
            result.is_err(),
            "Expected error for data smaller than period=10"
        );
    }

    #[test]
    fn test_cg_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let params = CgParams { period: Some(10) };
        let input = CgInput::from_candles(&candles, "close", params);
        let result = cg(&input).expect("Failed CG with real data");

        let check_idx = 240;
        if result.values.len() > check_idx {
            for i in check_idx..result.values.len() {
                if !result.values[i].is_nan() {
                    break;
                }
                if i == result.values.len() - 1 {
                    panic!("All CG values from index {} onward are NaN.", check_idx);
                }
            }
        }
    }
}
