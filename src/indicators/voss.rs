/// # Voss Filter (VOSS)
///
/// The Voss indicator by John F. Ehlers. It applies an IIR filter approach with cyclical analysis,
/// using cosine-based calculations and a bandwidth factor. This filter predicts future values based
/// on past data, with the `predict` parameter controlling the predictive lookahead.
///
/// ## Parameters
/// - **period**: The primary cycle length. Defaults to 20.
/// - **predict**: The predictive lookahead factor (multiplied by 3 internally). Defaults to 3.
/// - **bandwidth**: The bandwidth factor for the filter. Defaults to 0.25.
///
/// ## Errors
/// - **EmptyData**: voss: Input data slice is empty.
/// - **InvalidPeriod**: voss: `period` is zero or exceeds the data length.
/// - **NotEnoughValidData**: voss: Fewer than `max(period, 5, 3*predict)` valid (non-`NaN`) data points remain
///   after the first valid index.
/// - **AllValuesNaN**: voss: All input data values are `NaN`.
///
/// ## Returns
/// - **`Ok(VossOutput)`** on success, containing two `Vec<f64>` (`voss` and `filt`) matching the input length,
///   with leading `NaN`s until the filter can be calculated.
/// - **`Err(VossError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};
use std::f64::consts::PI;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum VossData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct VossOutput {
    pub voss: Vec<f64>,
    pub filt: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct VossParams {
    pub period: Option<usize>,
    pub predict: Option<usize>,
    pub bandwidth: Option<f64>,
}

impl Default for VossParams {
    fn default() -> Self {
        Self {
            period: Some(20),
            predict: Some(3),
            bandwidth: Some(0.25),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VossInput<'a> {
    pub data: VossData<'a>,
    pub params: VossParams,
}

impl<'a> VossInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: VossParams) -> Self {
        Self {
            data: VossData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: VossParams) -> Self {
        Self {
            data: VossData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: VossData::Candles {
                candles,
                source: "close",
            },
            params: VossParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| VossParams::default().period.unwrap())
    }

    pub fn get_predict(&self) -> usize {
        self.params
            .predict
            .unwrap_or_else(|| VossParams::default().predict.unwrap())
    }

    pub fn get_bandwidth(&self) -> f64 {
        self.params
            .bandwidth
            .unwrap_or_else(|| VossParams::default().bandwidth.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum VossError {
    #[error("voss: Empty data provided.")]
    EmptyData,
    #[error("voss: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("voss: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("voss: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn voss(input: &VossInput) -> Result<VossOutput, VossError> {
    let data: &[f64] = match &input.data {
        VossData::Candles { candles, source } => source_type(candles, source),
        VossData::Slice(slice) => slice,
    };

    if data.is_empty() {
        return Err(VossError::EmptyData);
    }

    let period = input.get_period();
    if period == 0 || period > data.len() {
        return Err(VossError::InvalidPeriod {
            period,
            data_len: data.len(),
        });
    }

    let predict = input.get_predict();
    let bandwidth = input.get_bandwidth();

    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(VossError::AllValuesNaN),
    };

    let order = 3 * predict;
    let min_index = period.max(5).max(order);

    if (data.len() - first_valid_idx) < min_index {
        return Err(VossError::NotEnoughValidData {
            needed: min_index,
            valid: data.len() - first_valid_idx,
        });
    }

    let mut voss_values = vec![f64::NAN; data.len()];
    let mut filt_values = vec![f64::NAN; data.len()];

    let f1 = (2.0 * PI / period as f64).cos();
    let g1 = (bandwidth * 2.0 * PI / period as f64).cos();
    let s1 = 1.0 / g1 - (1.0 / (g1 * g1) - 1.0).sqrt();

    for i in first_valid_idx..(first_valid_idx + min_index) {
        filt_values[i] = 0.0;
    }

    for i in (first_valid_idx + min_index)..data.len() {
        let current = data[i];
        let prev_2 = data[i - 2];
        let prev_filt_1 = filt_values[i - 1];
        let prev_filt_2 = filt_values[i - 2];
        filt_values[i] = 0.5 * (1.0 - s1) * (current - prev_2) + f1 * (1.0 + s1) * prev_filt_1
            - s1 * prev_filt_2;
    }

    for i in first_valid_idx..(first_valid_idx + min_index) {
        voss_values[i] = 0.0;
    }

    for i in (first_valid_idx + min_index)..data.len() {
        let mut sumc = 0.0;
        for count in 0..order {
            let idx = i - (order - count);
            sumc += ((count + 1) as f64 / order as f64) * voss_values[idx];
        }
        voss_values[i] = ((3 + order) as f64 / 2.0) * filt_values[i] - sumc;
    }

    Ok(VossOutput {
        voss: voss_values,
        filt: filt_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_voss_default_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = VossInput::with_default_candles(&candles);
        let output = voss(&input).expect("Failed VOSS calculation with default params");
        assert_eq!(output.voss.len(), candles.close.len());
        assert_eq!(output.filt.len(), candles.close.len());
    }

    #[test]
    fn test_voss_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let params = VossParams {
            period: None,
            predict: Some(2),
            bandwidth: None,
        };
        let input = VossInput::from_candles(&candles, "close", params);
        let output = voss(&input).expect("Failed VOSS calculation with partial params");
        assert_eq!(output.voss.len(), candles.close.len());
        assert_eq!(output.filt.len(), candles.close.len());
    }

    #[test]
    fn test_voss_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let params = VossParams {
            period: Some(20),
            predict: Some(3),
            bandwidth: Some(0.25),
        };
        let input = VossInput::from_candles(&candles, "close", params);
        let output = voss(&input).expect("Failed VOSS calculation");

        assert_eq!(output.voss.len(), candles.close.len());
        assert_eq!(output.filt.len(), candles.close.len());

        let expected_voss_last_five = [
            -290.430249544605,
            -269.74949153549596,
            -241.08179139844515,
            -149.2113276943419,
            -138.60361772412466,
        ];
        let expected_filt_last_five = [
            -228.0283989610523,
            -257.79056527053103,
            -270.3220395771822,
            -257.4282859799144,
            -235.78021136041997,
        ];

        assert!(output.voss.len() >= 5);
        assert!(output.filt.len() >= 5);

        let start_index = output.voss.len() - 5;
        for (i, &val) in output.voss[start_index..].iter().enumerate() {
            let expected_val = expected_voss_last_five[i];
            assert!(
                (val - expected_val).abs() < 1e-1,
                "VOSS mismatch at index {}: expected {}, got {}",
                i,
                expected_val,
                val
            );
        }

        for (i, &val) in output.filt[start_index..].iter().enumerate() {
            let expected_val = expected_filt_last_five[i];
            assert!(
                (val - expected_val).abs() < 1e-1,
                "Filt mismatch at index {}: expected {}, got {}",
                i,
                expected_val,
                val
            );
        }
    }

    #[test]
    fn test_voss_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = VossInput::with_default_candles(&candles);
        match input.data {
            VossData::Candles { source, .. } => {
                assert_eq!(source, "close", "Expected default source to be 'close'");
            }
            _ => panic!("Expected VossData::Candles variant"),
        }
    }

    #[test]
    fn test_voss_with_zero_period() {
        let input_data = [10.0, 20.0, 30.0];
        let params = VossParams {
            period: Some(0),
            predict: Some(3),
            bandwidth: Some(0.25),
        };
        let input = VossInput::from_slice(&input_data, params);
        let result = voss(&input);
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
    fn test_voss_with_period_exceeding_data_length() {
        let input_data = [10.0, 20.0, 30.0];
        let params = VossParams {
            period: Some(10),
            predict: Some(3),
            bandwidth: Some(0.25),
        };
        let input = VossInput::from_slice(&input_data, params);
        let result = voss(&input);
        assert!(result.is_err(), "Expected an error for period > data.len()");
    }

    #[test]
    fn test_voss_not_enough_data_after_first_valid() {
        let input_data = [f64::NAN, f64::NAN, 10.0, 20.0];
        let params = VossParams {
            period: Some(20),
            predict: Some(3),
            bandwidth: Some(0.25),
        };
        let input = VossInput::from_slice(&input_data, params);
        let result = voss(&input);
        assert!(
            result.is_err(),
            "Expected error for not enough data after first valid"
        );
    }

    #[test]
    fn test_voss_all_values_nan() {
        let input_data = [f64::NAN, f64::NAN, f64::NAN];
        let params = VossParams::default();
        let input = VossInput::from_slice(&input_data, params);
        let result = voss(&input);
        assert!(result.is_err(), "Expected error for all NaN values");
    }

    #[test]
    fn test_voss_reinput_slice() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let first_params = VossParams {
            period: Some(10),
            predict: Some(2),
            bandwidth: Some(0.2),
        };
        let first_input = VossInput::from_candles(&candles, "close", first_params);
        let first_result = voss(&first_input).expect("Failed VOSS calculation (first pass)");

        let second_params = VossParams {
            period: Some(10),
            predict: Some(2),
            bandwidth: Some(0.2),
        };
        let second_input = VossInput::from_slice(&first_result.voss, second_params);
        let second_result = voss(&second_input).expect("Failed VOSS calculation (second pass)");

        assert_eq!(second_result.voss.len(), first_result.voss.len());
        assert_eq!(second_result.filt.len(), first_result.voss.len());
    }
}
