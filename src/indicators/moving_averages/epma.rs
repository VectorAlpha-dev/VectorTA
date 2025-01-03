/// # End Point Moving Average (EPMA)
///
/// A moving average technique that projects a polynomial-based weighting scheme
/// from the data’s end point. It utilizes both a `period` to define how many
/// data points are considered, and an `offset` to shift the weights along the
/// data series. This approach can help emphasize more recent data values or
/// redistribute focus based on the chosen parameters.
///
/// ## Parameters
/// - **period**: Number of data points to include in the calculation (defaults to 11).
///   Must be ≥ 2.
/// - **offset**: Shift applied to the weighting (defaults to 4). Increasing the offset
///   moves the weighting window further back in the series, potentially reducing
///   immediate responsiveness.
///
/// ## Errors
/// - **EmptyDataSlice**: epma: No input data was provided.
/// - **InvalidPeriod**: epma: `period` < 2, making the calculation invalid.
/// - **StartIndexOutOfRange**: epma: The combined `period + offset + 1` exceeds the data length.
///
/// ## Returns
/// - **`Ok(EpmaOutput)`** on success, containing a `Vec<f64>` of length matching the input.
/// - **`Err(EpmaError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};

#[derive(Debug, Clone)]
pub enum EpmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct EpmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct EpmaParams {
    pub period: Option<usize>,
    pub offset: Option<usize>,
}

impl Default for EpmaParams {
    fn default() -> Self {
        Self {
            period: Some(11),
            offset: Some(4),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EpmaInput<'a> {
    pub data: EpmaData<'a>,
    pub params: EpmaParams,
}

impl<'a> EpmaInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: EpmaParams) -> Self {
        Self {
            data: EpmaData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: EpmaParams) -> Self {
        Self {
            data: EpmaData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: EpmaData::Candles {
                candles,
                source: "close",
            },
            params: EpmaParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| EpmaParams::default().period.unwrap())
    }

    pub fn get_offset(&self) -> usize {
        self.params
            .offset
            .unwrap_or_else(|| EpmaParams::default().offset.unwrap())
    }
}
use thiserror::Error;

#[derive(Debug, Error)]
pub enum EpmaError {
    #[error("Empty data slice for EPMA calculation.")]
    EmptyDataSlice,

    #[error("EPMA period must be >= 2: period = {period}.")]
    InvalidPeriod { period: usize },

    #[error("Start index for EPMA is out of range: start_index = {start_index}, data length = {data_len}.")]
    StartIndexOutOfRange { start_index: usize, data_len: usize },
}

#[inline]
pub fn epma(input: &EpmaInput) -> Result<EpmaOutput, EpmaError> {
    let data: &[f64] = match &input.data {
        EpmaData::Candles { candles, source } => source_type(candles, source),
        EpmaData::Slice(slice) => slice,
    };

    let n: usize = data.len();
    if n == 0 {
        return Err(EpmaError::EmptyDataSlice);
    }

    let period = input.get_period();
    let offset = input.get_offset();

    if period < 2 {
        return Err(EpmaError::InvalidPeriod { period });
    }

    let start_index = period + offset + 1;
    if start_index >= n {
        return Err(EpmaError::StartIndexOutOfRange {
            start_index,
            data_len: n,
        });
    }
    let mut output = data.to_vec();

    let p_minus_1 = period - 1;
    let mut weights = Vec::with_capacity(p_minus_1);

    for i in 0..p_minus_1 {
        let w_i32 = (period as i32) - (i as i32) - (offset as i32);
        let w = w_i32 as f64;
        weights.push(w);
    }

    let weight_sum: f64 = weights.iter().sum();

    for j in start_index..n {
        let mut my_sum = 0.0;
        let mut i = 0_usize;

        while i + 3 < p_minus_1 {
            my_sum += data[j - i] * weights[i];
            my_sum += data[j - (i + 1)] * weights[i + 1];
            my_sum += data[j - (i + 2)] * weights[i + 2];
            my_sum += data[j - (i + 3)] * weights[i + 3];
            i += 4;
        }
        while i < p_minus_1 {
            my_sum += data[j - i] * weights[i];
            i += 1;
        }

        output[j] = my_sum / weight_sum;
    }

    Ok(EpmaOutput { values: output })
}

#[cfg(test)]
mod tests {
    use std::result;

    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_epma_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = EpmaParams {
            period: None,
            offset: None,
        };
        let input = EpmaInput::from_candles(&candles, "close", default_params);
        let output = epma(&input).expect("Failed EPMA with default params");
        assert_eq!(output.values.len(), candles.close.len(), "Length mismatch");

        let params_period_14 = EpmaParams {
            period: Some(14),
            offset: None,
        };
        let input2 = EpmaInput::from_candles(&candles, "hl2", params_period_14);
        let output2 = epma(&input2).expect("Failed EPMA with period=14, source=hl2");
        assert_eq!(output2.values.len(), candles.close.len(), "Length mismatch");

        let params_custom = EpmaParams {
            period: Some(10),
            offset: Some(5),
        };
        let input3 = EpmaInput::from_candles(&candles, "hlc3", params_custom);
        let output3 = epma(&input3).expect("Failed EPMA fully custom");
        assert_eq!(output3.values.len(), candles.close.len(), "Length mismatch");
    }

    #[test]
    fn test_epma_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = EpmaParams::default();
        let input = EpmaInput::from_candles(&candles, "close", default_params);
        let result = epma(&input).expect("Failed to calculate EPMA");

        let expected_last_five = [59174.48, 59201.04, 59167.60, 59200.32, 59117.04];

        assert_eq!(
            result.values.len(),
            candles.close.len(),
            "EPMA output length does not match input length!"
        );

        let start_index = result.values.len().saturating_sub(5);
        let result_last_five = &result.values[start_index..];

        for (i, &value) in result_last_five.iter().enumerate() {
            assert!(
                (value - expected_last_five[i]).abs() < 1e-1,
                "EPMA value mismatch at index {}: expected {}, got {}",
                i,
                expected_last_five[i],
                value
            );
        }

        for val in &result.values {
            if !val.is_nan() {
                assert!(
                    val.is_finite(),
                    "EPMA output contains non-finite values (e.g. Inf)."
                );
            }
        }
    }
    #[test]
    fn test_epma_params_with_default_params() {
        let default_params = EpmaParams::default();
        assert_eq!(default_params.period, Some(11), "Period should be 11");
        assert_eq!(default_params.offset, Some(4), "Offset should be 4");
    }

    #[test]
    fn test_epma_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).unwrap();
        let input = EpmaInput::with_default_candles(&candles);
        match input.data {
            EpmaData::Candles { source, .. } => {
                assert_eq!(source, "close");
            }
            _ => panic!("Unexpected EpmaData variant"),
        }
        assert_eq!(input.params.period, Some(11), "Period should be 11");
        assert_eq!(input.params.offset, Some(4), "Offset should be 4");
    }

    #[test]
    fn test_epma_with_zero_period() {
        let input_data = [10.0, 20.0, 30.0];
        let params = EpmaParams {
            period: Some(0),
            offset: Some(2),
        };
        let input = EpmaInput::from_slice(&input_data, params);
        let result = epma(&input);
        assert!(result.is_err(), "Expected an error for zero period");
    }

    #[test]
    fn test_epma_with_period_offset_exceeding_data_length() {
        let input_data = [10.0, 20.0, 30.0];
        let params = EpmaParams {
            period: Some(11),
            offset: Some(4),
        };
        let input = EpmaInput::from_slice(&input_data, params);
        assert!(
            result::Result::is_err(&epma(&input)),
            "Expected an error for period > data.len()"
        );
    }

    #[test]
    fn test_epma_very_small_data_set() {
        let input_data = [42.0, 43.0];
        let params = EpmaParams {
            period: Some(2),
            offset: Some(1),
        };
        let input = EpmaInput::from_slice(&input_data, params);
        assert!(
            result::Result::is_err(&epma(&input)),
            "Expected an error for period > data.len()"
        );
    }

    #[test]
    fn test_epma_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).unwrap();
        let first_params = EpmaParams {
            period: Some(5),
            offset: Some(2),
        };
        let first_input = EpmaInput::from_candles(&candles, "close", first_params);
        let first_result = epma(&first_input).unwrap();
        assert_eq!(
            first_result.values.len(),
            candles.close.len(),
            "Length mismatch"
        );
        let second_params = EpmaParams {
            period: Some(3),
            offset: Some(1),
        };
        let second_input = EpmaInput::from_slice(&first_result.values, second_params);
        let second_result = epma(&second_input).unwrap();
        assert_eq!(
            second_result.values.len(),
            first_result.values.len(),
            "Length mismatch"
        );
        if second_result.values.len() > 240 {
            for i in 240..second_result.values.len() {
                assert!(
                    !second_result.values[i].is_nan(),
                    "Found NaN at index {}.",
                    i
                );
            }
        }
    }

    #[test]
    fn test_epma_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).unwrap();
        let params = EpmaParams {
            period: Some(11),
            offset: Some(4),
        };
        let input = EpmaInput::from_candles(&candles, "close", params.clone());
        let epma_result = epma(&input).unwrap();
        for i in 0..epma_result.values.len() {
            let val = epma_result.values[i];
            if i < (params.period.unwrap() + params.offset.unwrap() + 1) {
                assert_eq!(val, candles.close[i], "Mismatch at index {}", i);
            } else {
                assert!(!val.is_nan(), "Found NaN at index {}", i);
            }
        }
    }
}
