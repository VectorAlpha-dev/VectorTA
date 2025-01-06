/// # Correlation Cycle (John Ehlers)
///
/// This indicator calculates four outputs:
/// - **Real part (real)**
/// - **Imag part (imag)**
/// - **Angle (angle)**
/// - **Market State (state)**
///
/// The underlying logic uses correlation-based phasor components over a window of size `period`,
/// then computes an angle and applies a threshold to determine the current market state.  
///
/// ## Parameters
/// - **period**: Window size for the correlation, defaults to 20.
/// - **threshold**: Threshold to determine whether the market is stable or rapidly changing, defaults to 9.
///
/// ## Errors
/// - **EmptyData**: correlation_cycle: Input data slice is empty.
/// - **InvalidPeriod**: correlation_cycle: `period` is zero or exceeds the data length.
/// - **NotEnoughValidData**: correlation_cycle: Fewer than `period` total *non-`NaN`* data points are available.
/// - **AllValuesNaN**: correlation_cycle: All input data values are `NaN`.
///
/// ## Returns
/// - **`Ok(CorrelationCycleOutput)`** on success, containing four `Vec<f64>` matching the input length,
///   with leading `NaN` (for `real`, `imag`, `angle`) and `0.0` (for `state`) values until the correlation window is filled.
/// - **`Err(CorrelationCycleError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum CorrelationCycleData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct CorrelationCycleOutput {
    pub real: Vec<f64>,
    pub imag: Vec<f64>,
    pub angle: Vec<f64>,
    pub state: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct CorrelationCycleParams {
    pub period: Option<usize>,
    pub threshold: Option<f64>,
}

impl Default for CorrelationCycleParams {
    fn default() -> Self {
        Self {
            period: Some(20),
            threshold: Some(9.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CorrelationCycleInput<'a> {
    pub data: CorrelationCycleData<'a>,
    pub params: CorrelationCycleParams,
}

impl<'a> CorrelationCycleInput<'a> {
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: CorrelationCycleParams,
    ) -> Self {
        Self {
            data: CorrelationCycleData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: CorrelationCycleParams) -> Self {
        Self {
            data: CorrelationCycleData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: CorrelationCycleData::Candles {
                candles,
                source: "close",
            },
            params: CorrelationCycleParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| CorrelationCycleParams::default().period.unwrap())
    }

    pub fn get_threshold(&self) -> f64 {
        self.params
            .threshold
            .unwrap_or_else(|| CorrelationCycleParams::default().threshold.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum CorrelationCycleError {
    #[error("correlation_cycle: Empty data provided.")]
    EmptyData,
    #[error("correlation_cycle: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("correlation_cycle: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("correlation_cycle: All values are NaN.")]
    AllValuesNaN,
}

use crate::utilities::math_functions::atan64;
#[inline]
pub fn correlation_cycle(
    input: &CorrelationCycleInput,
) -> Result<CorrelationCycleOutput, CorrelationCycleError> {
    let data = match &input.data {
        CorrelationCycleData::Candles { candles, source } => source_type(candles, source),
        CorrelationCycleData::Slice(slice) => slice,
    };
    if data.is_empty() {
        return Err(CorrelationCycleError::EmptyData);
    }
    if data.iter().all(|&x| x.is_nan()) {
        return Err(CorrelationCycleError::AllValuesNaN);
    }
    let period = input.get_period();
    if period == 0 || period > data.len() {
        return Err(CorrelationCycleError::InvalidPeriod {
            period,
            data_len: data.len(),
        });
    }
    let valid_count = data.iter().filter(|&&x| !x.is_nan()).count();
    if valid_count < period {
        return Err(CorrelationCycleError::NotEnoughValidData {
            needed: period,
            valid: valid_count,
        });
    }
    let threshold = input.get_threshold();
    let mut real = vec![f64::NAN; data.len()];
    let mut imag = vec![f64::NAN; data.len()];
    let mut angle = vec![f64::NAN; data.len()];
    let mut state = vec![0.0; data.len()];
    let two_pi = 4.0 * f64::asin(1.0);
    let half_pi = f64::asin(1.0);
    let mut cos_table = vec![0.0; period];
    let mut sin_table = vec![0.0; period];
    for j in 0..period {
        let a = two_pi * (j as f64 + 1.0) / period as f64;
        cos_table[j] = a.cos();
        sin_table[j] = -a.sin();
    }
    for i in period..data.len() {
        let mut rx = 0.0;
        let mut rxx = 0.0;
        let mut rxy = 0.0;
        let mut ryy = 0.0;
        let mut ry = 0.0;
        let mut ix = 0.0;
        let mut ixx = 0.0;
        let mut ixy = 0.0;
        let mut iyy = 0.0;
        let mut iy = 0.0;
        for j in 0..period {
            let idx = i - (j + 1);
            let x = if data[idx].is_nan() { 0.0 } else { data[idx] };
            let yc = cos_table[j];
            let ys = sin_table[j];
            rx += x;
            rxx += x * x;
            rxy += x * yc;
            ryy += yc * yc;
            ry += yc;
            ix += x;
            ixx += x * x;
            ixy += x * ys;
            iyy += ys * ys;
            iy += ys;
        }
        let t1 = (period as f64) * rxx - rx * rx;
        let t2 = (period as f64) * ryy - ry * ry;
        if t1 > 0.0 && t2 > 0.0 {
            real[i] = ((period as f64) * rxy - rx * ry) / (t1 * t2).sqrt();
        }
        let t3 = (period as f64) * ixx - ix * ix;
        let t4 = (period as f64) * iyy - iy * iy;
        if t3 > 0.0 && t4 > 0.0 {
            imag[i] = ((period as f64) * ixy - ix * iy) / (t3 * t4).sqrt();
        }
    }
    for i in period..data.len() {
        let im = imag[i];
        if im == 0.0 {
            angle[i] = 0.0;
        } else {
            let mut a = atan64(real[i] / im) + half_pi;
            a = a.to_degrees();
            if im > 0.0 {
                a -= 180.0;
            }
            angle[i] = a;
        }
    }
    for i in (period + 1)..data.len() {
        let pa = angle[i - 1];
        let ca = angle[i];
        if !pa.is_nan() && !ca.is_nan() && pa > ca && (pa - ca) < 270.0 {
            angle[i] = pa;
        }
    }
    for i in (period + 1)..data.len() {
        let pa = angle[i - 1];
        let ca = angle[i];
        if !pa.is_nan() && !ca.is_nan() && (ca - pa).abs() < threshold {
            state[i] = if ca >= 0.0 { 1.0 } else { -1.0 };
        } else {
            state[i] = 0.0;
        }
    }
    Ok(CorrelationCycleOutput {
        real,
        imag,
        angle,
        state,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_cc_default_params() {
        let default_params = CorrelationCycleParams::default();
        assert_eq!(default_params.period, Some(20));
        assert_eq!(default_params.threshold, Some(9.0));
    }

    #[test]
    fn test_cc_with_slice_basic() {
        let data = [
            10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
        ];
        let params = CorrelationCycleParams {
            period: Some(5),
            threshold: Some(4.0),
        };
        let input = CorrelationCycleInput::from_slice(&data, params);
        let output = correlation_cycle(&input).expect("CC should succeed");
        assert_eq!(output.real.len(), data.len());
        assert_eq!(output.imag.len(), data.len());
        assert_eq!(output.angle.len(), data.len());
        assert_eq!(output.state.len(), data.len());
    }

    #[test]
    fn test_cc_with_empty_data() {
        let data: [f64; 0] = [];
        let params = CorrelationCycleParams::default();
        let input = CorrelationCycleInput::from_slice(&data, params);
        let result = correlation_cycle(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                e.to_string().contains("Empty data"),
                "Wrong error for empty data: {}",
                e
            );
        }
    }

    #[test]
    fn test_cc_with_zero_period() {
        let data = [10.0, 20.0, 30.0];
        let params = CorrelationCycleParams {
            period: Some(0),
            threshold: Some(9.0),
        };
        let input = CorrelationCycleInput::from_slice(&data, params);
        let result = correlation_cycle(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                e.to_string().contains("Invalid period"),
                "Expected 'Invalid period' error, got: {}",
                e
            );
        }
    }

    #[test]
    fn test_cc_period_exceeds_length() {
        let data = [1.0, 2.0, 3.0];
        let params = CorrelationCycleParams {
            period: Some(10),
            threshold: Some(9.0),
        };
        let input = CorrelationCycleInput::from_slice(&data, params);
        let result = correlation_cycle(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                e.to_string().contains("Invalid period"),
                "Expected 'Invalid period' error, got: {}",
                e
            );
        }
    }

    #[test]
    fn test_cc_all_values_nan() {
        let data = [f64::NAN, f64::NAN, f64::NAN];
        let input = CorrelationCycleInput::from_slice(&data, CorrelationCycleParams::default());
        let result = correlation_cycle(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                e.to_string().contains("All values are NaN"),
                "Expected 'All values are NaN' error, got: {}",
                e
            );
        }
    }

    #[test]
    fn test_cc_with_not_enough_valid_data() {
        let data = [f64::NAN, 10.0, f64::NAN, f64::NAN];
        let params = CorrelationCycleParams {
            period: Some(3),
            threshold: Some(9.0),
        };
        let input = CorrelationCycleInput::from_slice(&data, params);
        let result = correlation_cycle(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                e.to_string().contains("Not enough valid data"),
                "Expected 'Not enough valid data' error, got: {}",
                e
            );
        }
    }

    #[test]
    fn test_cc_from_candles_with_defaults() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = CorrelationCycleInput::with_default_candles(&candles);
        let output = correlation_cycle(&input).expect("CC with default candles should succeed");
        assert_eq!(output.real.len(), candles.close.len());
        assert_eq!(output.imag.len(), candles.close.len());
        assert_eq!(output.angle.len(), candles.close.len());
        assert_eq!(output.state.len(), candles.close.len());
    }

    #[test]
    fn test_cc_slice_reinput() {
        let data = [10.0, 10.5, 11.0, 11.5, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0];
        let params = CorrelationCycleParams {
            period: Some(4),
            threshold: Some(2.0),
        };
        let input = CorrelationCycleInput::from_slice(&data, params.clone());
        let first_result = correlation_cycle(&input).expect("First CC failed");
        let second_input = CorrelationCycleInput::from_slice(&first_result.real, params);
        let second_result = correlation_cycle(&second_input).expect("Second CC failed");
        assert_eq!(first_result.real.len(), data.len());
        assert_eq!(second_result.real.len(), data.len());
    }

    #[test]
    fn test_cc_final_five_values() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = CorrelationCycleParams {
            period: Some(20),
            threshold: Some(9.0),
        };
        let input = CorrelationCycleInput::from_candles(&candles, "close", params);
        let output = correlation_cycle(&input).expect("CC calculation failed");

        let expected_last_five_real = [
            -0.3348928030992766,
            -0.2908979303392832,
            -0.10648582811938148,
            -0.09118320471750277,
            0.0826798259258665,
        ];
        let expected_last_five_imag = [
            0.2902308064575494,
            0.4025192756952553,
            0.4704322460080054,
            0.5404405595224989,
            0.5418162415918566,
        ];
        let expected_last_five_angle = [
            -139.0865569687123,
            -125.8553823569915,
            -102.75438860700636,
            -99.576759208278,
            -81.32373697835556,
        ];
        let expected_last_five_state = [0.0, 0.0, 0.0, -1.0, 0.0];

        let start_index = output.real.len() - 5;
        let result_last_five_real = &output.real[start_index..];
        let result_last_five_imag = &output.imag[start_index..];
        let result_last_five_angle = &output.angle[start_index..];
        let result_last_five_state = &output.state[start_index..];

        for i in 0..5 {
            let diff_real = (result_last_five_real[i] - expected_last_five_real[i]).abs();
            let diff_imag = (result_last_five_imag[i] - expected_last_five_imag[i]).abs();
            let diff_angle = (result_last_five_angle[i] - expected_last_five_angle[i]).abs();

            assert!(
                diff_real < 1e-8,
                "Mismatch in real component at index {}: expected {}, got {}",
                i,
                expected_last_five_real[i],
                result_last_five_real[i]
            );
            assert!(
                diff_imag < 1e-8,
                "Mismatch in imag component at index {}: expected {}, got {}",
                i,
                expected_last_five_imag[i],
                result_last_five_imag[i]
            );
        }
    }
}
