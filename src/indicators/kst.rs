use crate::indicators::moving_averages::sma::{
    sma, SmaData, SmaError, SmaInput, SmaOutput, SmaParams,
};
/// # Know Sure Thing (KST)
///
/// KST is a momentum oscillator based on the smoothed rate-of-change (ROC) values of four different time frames.
/// It can help identify potential turning points in price movement by combining multiple ROC calculations,
/// smoothing them, and summing the results in a weighted manner.
///
/// ## Parameters
/// - **sma_period1**: Smoothing period for the first ROC. Defaults to 10.
/// - **sma_period2**: Smoothing period for the second ROC. Defaults to 10.
/// - **sma_period3**: Smoothing period for the third ROC. Defaults to 10.
/// - **sma_period4**: Smoothing period for the fourth ROC. Defaults to 15.
/// - **roc_period1**: Period for the first ROC calculation. Defaults to 10.
/// - **roc_period2**: Period for the second ROC calculation. Defaults to 15.
/// - **roc_period3**: Period for the third ROC calculation. Defaults to 20.
/// - **roc_period4**: Period for the fourth ROC calculation. Defaults to 30.
/// - **signal_period**: Smoothing period for the signal line. Defaults to 9.
///
/// ## Errors
/// - **EmptyData**: kst: Input data slice is empty.
/// - **AllValuesNaN**: kst: All input data values are `NaN`.
/// - **NotEnoughValidData**: kst: Fewer than the necessary valid (non-`NaN`) data points remain
///   after the first valid index.
/// - **InvalidPeriod**: kst: A requested period is zero or exceeds the data length.
/// - **Roc(...)**: Propagated error from the underlying ROC calculation.
/// - **Sma(...)**: Propagated error from the underlying SMA calculation.
///
/// ## Returns
/// - **`Ok(KstOutput)`** on success, containing two `Vec<f64>`:
///   - `line`: The KST line
///   - `signal`: The KST signal line
///   Both match the input length, with leading `NaN`s until each required window is filled.
/// - **`Err(KstError)`** otherwise.
use crate::indicators::roc::{roc, RocData, RocError, RocInput, RocOutput, RocParams};
use crate::utilities::data_loader::{source_type, Candles};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum KstData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct KstOutput {
    pub line: Vec<f64>,
    pub signal: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct KstParams {
    pub sma_period1: Option<usize>,
    pub sma_period2: Option<usize>,
    pub sma_period3: Option<usize>,
    pub sma_period4: Option<usize>,
    pub roc_period1: Option<usize>,
    pub roc_period2: Option<usize>,
    pub roc_period3: Option<usize>,
    pub roc_period4: Option<usize>,
    pub signal_period: Option<usize>,
}

impl Default for KstParams {
    fn default() -> Self {
        Self {
            sma_period1: Some(10),
            sma_period2: Some(10),
            sma_period3: Some(10),
            sma_period4: Some(15),
            roc_period1: Some(10),
            roc_period2: Some(15),
            roc_period3: Some(20),
            roc_period4: Some(30),
            signal_period: Some(9),
        }
    }
}

#[derive(Debug, Clone)]
pub struct KstInput<'a> {
    pub data: KstData<'a>,
    pub params: KstParams,
}

impl<'a> KstInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: KstParams) -> Self {
        Self {
            data: KstData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: KstParams) -> Self {
        Self {
            data: KstData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: KstData::Candles {
                candles,
                source: "close",
            },
            params: KstParams::default(),
        }
    }

    fn get_or_default(value: Option<usize>, default: usize) -> usize {
        value.unwrap_or(default)
    }
}

#[derive(Debug, Error)]
pub enum KstError {
    #[error("kst: {0}")]
    Roc(#[from] RocError),
    #[error("kst: {0}")]
    Sma(#[from] SmaError),
    #[error("kst: Empty data provided")]
    EmptyData,
    #[error("kst: All values are NaN")]
    AllValuesNaN,
    #[error("kst: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("kst: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
}

#[inline]
pub fn kst(input: &KstInput) -> Result<KstOutput, KstError> {
    let data: &[f64] = match &input.data {
        KstData::Candles { candles, source } => source_type(candles, source),
        KstData::Slice(slice) => slice,
    };

    if data.is_empty() {
        return Err(KstError::EmptyData);
    }

    let p = &input.params;
    let sma_period1 = KstInput::get_or_default(p.sma_period1, 10);
    let sma_period2 = KstInput::get_or_default(p.sma_period2, 10);
    let sma_period3 = KstInput::get_or_default(p.sma_period3, 10);
    let sma_period4 = KstInput::get_or_default(p.sma_period4, 15);
    let roc_period1 = KstInput::get_or_default(p.roc_period1, 10);
    let roc_period2 = KstInput::get_or_default(p.roc_period2, 15);
    let roc_period3 = KstInput::get_or_default(p.roc_period3, 20);
    let roc_period4 = KstInput::get_or_default(p.roc_period4, 30);
    let signal_period = KstInput::get_or_default(p.signal_period, 9);

    let roc1_input = RocInput::from_slice(
        data,
        RocParams {
            period: Some(roc_period1),
        },
    );
    let roc2_input = RocInput::from_slice(
        data,
        RocParams {
            period: Some(roc_period2),
        },
    );
    let roc3_input = RocInput::from_slice(
        data,
        RocParams {
            period: Some(roc_period3),
        },
    );
    let roc4_input = RocInput::from_slice(
        data,
        RocParams {
            period: Some(roc_period4),
        },
    );

    let roc1 = roc(&roc1_input)?;
    let roc2 = roc(&roc2_input)?;
    let roc3 = roc(&roc3_input)?;
    let roc4 = roc(&roc4_input)?;

    let aroc1_input = SmaInput::from_slice(
        &roc1.values,
        SmaParams {
            period: Some(sma_period1),
        },
    );
    let aroc2_input = SmaInput::from_slice(
        &roc2.values,
        SmaParams {
            period: Some(sma_period2),
        },
    );
    let aroc3_input = SmaInput::from_slice(
        &roc3.values,
        SmaParams {
            period: Some(sma_period3),
        },
    );
    let aroc4_input = SmaInput::from_slice(
        &roc4.values,
        SmaParams {
            period: Some(sma_period4),
        },
    );

    let aroc1 = sma(&aroc1_input)?;
    let aroc2 = sma(&aroc2_input)?;
    let aroc3 = sma(&aroc3_input)?;
    let aroc4 = sma(&aroc4_input)?;

    if aroc1.values.is_empty()
        || aroc2.values.is_empty()
        || aroc3.values.is_empty()
        || aroc4.values.is_empty()
    {
        return Err(KstError::EmptyData);
    }

    let mut line = vec![f64::NAN; data.len()];
    for i in 0..data.len() {
        let v1 = aroc1.values[i];
        let v2 = aroc2.values[i];
        let v3 = aroc3.values[i];
        let v4 = aroc4.values[i];
        if v1.is_nan() || v2.is_nan() || v3.is_nan() || v4.is_nan() {
            line[i] = f64::NAN;
        } else {
            line[i] = v1 + 2.0 * v2 + 3.0 * v3 + 4.0 * v4;
        }
    }

    let line_sma_input = SmaInput::from_slice(
        &line,
        SmaParams {
            period: Some(signal_period),
        },
    );
    let line_sma = sma(&line_sma_input)?;
    Ok(KstOutput {
        line,
        signal: line_sma.values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_kst_default_params_length() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = KstInput::with_default_candles(&candles);
        let result = kst(&input).expect("Failed to calculate KST with default params");
        assert_eq!(result.line.len(), candles.close.len());
        assert_eq!(result.signal.len(), candles.close.len());
    }

    #[test]
    fn test_kst_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = KstInput::with_default_candles(&candles);
        let kst_result = kst(&input).expect("Failed to calculate KST");

        assert_eq!(kst_result.line.len(), candles.close.len());
        assert_eq!(kst_result.signal.len(), candles.close.len());
        assert!(kst_result.line.len() >= 5, "KST length too short");
        assert!(kst_result.signal.len() >= 5, "KST signal length too short");

        let expected_last_five_line = [
            -47.38570195278667,
            -44.42926180347176,
            -42.185693049429034,
            -40.10697793942024,
            -40.17466795905724,
        ];
        let expected_last_five_signal = [
            -52.66743277411538,
            -51.559775662725556,
            -50.113844191238954,
            -48.58923772989874,
            -47.01112630514571,
        ];

        let start_idx_line = kst_result.line.len() - 5;
        let start_idx_signal = kst_result.signal.len() - 5;

        for (i, &value) in kst_result.line[start_idx_line..].iter().enumerate() {
            let expected_value = expected_last_five_line[i];
            assert!(
                (value - expected_value).abs() < 1e-1,
                "KST line mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }

        for (i, &value) in kst_result.signal[start_idx_signal..].iter().enumerate() {
            let expected_value = expected_last_five_signal[i];
            assert!(
                (value - expected_value).abs() < 1e-1,
                "KST signal mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }
    }

    #[test]
    fn test_kst_with_custom_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = KstParams {
            sma_period1: Some(5),
            sma_period2: Some(5),
            sma_period3: Some(7),
            sma_period4: Some(9),
            roc_period1: Some(4),
            roc_period2: Some(6),
            roc_period3: Some(10),
            roc_period4: Some(12),
            signal_period: Some(3),
        };
        let input = KstInput::from_candles(&candles, "close", params);
        let result = kst(&input).expect("Failed to calculate KST with custom params");
        assert_eq!(result.line.len(), candles.close.len());
        assert_eq!(result.signal.len(), candles.close.len());
    }

    #[test]
    fn test_kst_with_zero_roc_period() {
        let data = [10.0, 20.0, 30.0, 40.0];
        let params = KstParams {
            roc_period1: Some(0),
            ..KstParams::default()
        };
        let input = KstInput::from_slice(&data, params);
        let result = kst(&input);
        assert!(result.is_err(), "Expected an error for zero ROC period");
        if let Err(e) = result {
            let msg = e.to_string();
            assert!(
                msg.contains("Invalid period") || msg.contains("period = 0"),
                "Expected 'Invalid period' error message, got: {}",
                msg
            );
        }
    }

    #[test]
    fn test_kst_with_zero_sma_period() {
        let data = [10.0, 20.0, 30.0, 40.0];
        let params = KstParams {
            sma_period1: Some(0),
            ..KstParams::default()
        };
        let input = KstInput::from_slice(&data, params);
        let result = kst(&input);
        assert!(result.is_err(), "Expected an error for zero SMA period");
        if let Err(e) = result {
            let msg = e.to_string();
            assert!(
                msg.contains("Invalid period") || msg.contains("period = 0"),
                "Expected 'Invalid period' error message, got: {}",
                msg
            );
        }
    }

    #[test]
    fn test_kst_not_enough_data() {
        let data = [10.0, 20.0];
        let params = KstParams {
            roc_period1: Some(5),
            ..KstParams::default()
        };
        let input = KstInput::from_slice(&data, params);
        let result = kst(&input);
        assert!(result.is_err(), "Expected an error for not enough data");
    }

    #[test]
    fn test_kst_all_nan() {
        let data = [f64::NAN, f64::NAN, f64::NAN];
        let input = KstInput::from_slice(&data, KstParams::default());
        let result = kst(&input);
        assert!(result.is_err(), "Expected an error for all NaN data");
    }
}
