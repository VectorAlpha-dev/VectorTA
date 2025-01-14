/// # Mesa Sine Wave (MSW)
///
/// The Mesa Sine Wave indicator attempts to detect turning points in price data
/// by fitting a sine wave function. It outputs two series: the `sine` wave
/// and a leading version of the wave (`lead`).
///
/// ## Parameters
/// - **period**: The window size (number of data points). Defaults to 5.
///
/// ## Errors
/// - **EmptyData**: msw: Input data slice is empty.
/// - **InvalidPeriod**: msw: `period` is zero or exceeds the data length.
/// - **NotEnoughValidData**: msw: Fewer than `period` valid (non-`NaN`) data points remain
///   after the first valid index.
/// - **AllValuesNaN**: msw: All input data values are `NaN`.
///
/// ## Returns
/// - **`Ok(MswOutput)`** on success, containing two `Vec<f64>` of equal length:
///   `sine` and `lead`, both matching the input length, with leading `NaN`s until
///   the Mesa Sine Wave window is filled.
/// - **`Err(MswError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::math_functions::{atan64, fast_cos_f64, fast_sin_f64};
use std::f64::consts::PI;
use thiserror::Error;
#[allow(clippy::approx_constant)]
const TULIP_PI: f64 = 3.1415926;
const TULIP_TPI: f64 = 2.0 * TULIP_PI;

#[derive(Debug, Clone)]
pub enum MswData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct MswOutput {
    pub sine: Vec<f64>,
    pub lead: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct MswParams {
    pub period: Option<usize>,
}

impl Default for MswParams {
    fn default() -> Self {
        Self { period: Some(5) }
    }
}

#[derive(Debug, Clone)]
pub struct MswInput<'a> {
    pub data: MswData<'a>,
    pub params: MswParams,
}

impl<'a> MswInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: MswParams) -> Self {
        Self {
            data: MswData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: MswParams) -> Self {
        Self {
            data: MswData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: MswData::Candles {
                candles,
                source: "close",
            },
            params: MswParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| MswParams::default().period.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum MswError {
    #[error("msw: Empty data provided for MSW.")]
    EmptyData,
    #[error("msw: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("msw: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("msw: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn msw(input: &MswInput) -> Result<MswOutput, MswError> {
    let data: &[f64] = match &input.data {
        MswData::Candles { candles, source } => source_type(candles, source),
        MswData::Slice(slice) => slice,
    };

    if data.is_empty() {
        return Err(MswError::EmptyData);
    }

    let period = input.get_period();
    if period == 0 || period > data.len() {
        return Err(MswError::InvalidPeriod {
            period,
            data_len: data.len(),
        });
    }

    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(MswError::AllValuesNaN),
    };

    if (data.len() - first_valid_idx) < period {
        return Err(MswError::NotEnoughValidData {
            needed: period,
            valid: data.len() - first_valid_idx,
        });
    }

    let mut sine_vals = vec![f64::NAN; data.len()];
    let mut lead_vals = vec![f64::NAN; data.len()];

    let mut cos_table = vec![0.0; period];
    let mut sin_table = vec![0.0; period];
    for j in 0..period {
        let angle = TULIP_TPI * j as f64 / period as f64;
        cos_table[j] = angle.cos();
        sin_table[j] = fast_sin_f64(angle);
    }

    for i in (first_valid_idx + period - 1)..data.len() {
        let mut rp = 0.0;
        let mut ip = 0.0;
        for j in 0..period {
            let weight = data[i - j];
            rp += cos_table[j] * weight;
            ip += sin_table[j] * weight;
        }

        let mut phase = if rp.abs() > 0.001 {
            atan64(ip / rp)
        } else {
            TULIP_PI * if ip < 0.0 { -1.0 } else { 1.0 }
        };

        if rp < 0.0 {
            phase += TULIP_PI;
        }
        phase += TULIP_PI / 2.0;
        if phase < 0.0 {
            phase += TULIP_TPI;
        }
        if phase > TULIP_TPI {
            phase -= TULIP_TPI;
        }

        sine_vals[i] = fast_sin_f64(phase);
        lead_vals[i] = fast_sin_f64(phase + TULIP_PI / 4.0);
    }

    Ok(MswOutput {
        sine: sine_vals,
        lead: lead_vals,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_msw_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = MswParams { period: None };
        let input_default = MswInput::from_candles(&candles, "close", default_params);
        let output_default = msw(&input_default).expect("Failed MSW with default params");
        assert_eq!(output_default.sine.len(), candles.close.len());
        assert_eq!(output_default.lead.len(), candles.close.len());

        let params_period_10 = MswParams { period: Some(10) };
        let input_period_10 = MswInput::from_candles(&candles, "hl2", params_period_10);
        let output_period_10 =
            msw(&input_period_10).expect("Failed MSW with period=10, source=hl2");
        assert_eq!(output_period_10.sine.len(), candles.close.len());
        assert_eq!(output_period_10.lead.len(), candles.close.len());
    }

    #[test]
    fn test_msw_with_nan_data() {
        let input_data = [f64::NAN, f64::NAN, 10.0, 11.0, 12.0, f64::NAN];
        let params = MswParams { period: Some(3) };
        let input = MswInput::from_slice(&input_data, params);
        let result = msw(&input).expect("Failed to calculate MSW with NaN data");
        assert_eq!(result.sine.len(), input_data.len());
        assert_eq!(result.lead.len(), input_data.len());
    }

    #[test]
    fn test_msw_error_conditions() {
        let empty_data: [f64; 0] = [];
        let params = MswParams { period: Some(5) };
        let input_empty = MswInput::from_slice(&empty_data, params.clone());
        assert!(msw(&input_empty).is_err(), "Expected error on empty data");

        let input_small = [10.0, 20.0, 30.0, 40.0];
        let params_invalid_period = MswParams { period: Some(0) };
        let input_zero_period = MswInput::from_slice(&input_small, params_invalid_period);
        assert!(
            msw(&input_zero_period).is_err(),
            "Expected error on zero period"
        );

        let params_large_period = MswParams { period: Some(10) };
        let input_large_period = MswInput::from_slice(&input_small, params_large_period);
        assert!(
            msw(&input_large_period).is_err(),
            "Expected error on period exceeding data length"
        );

        let all_nan_data = [f64::NAN, f64::NAN, f64::NAN];
        let input_all_nan = MswInput::from_slice(&all_nan_data, MswParams { period: Some(2) });
        assert!(
            msw(&input_all_nan).is_err(),
            "Expected error when all values are NaN"
        );
    }

    #[test]
    fn test_msw_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let params = MswParams { period: Some(5) };
        let input = MswInput::from_candles(&candles, "close", params);
        let msw_result = msw(&input).expect("Failed to calculate MSW");

        assert_eq!(
            msw_result.sine.len(),
            close_prices.len(),
            "MSW sine length mismatch"
        );
        assert_eq!(
            msw_result.lead.len(),
            close_prices.len(),
            "MSW lead length mismatch"
        );

        let expected_last_five_sine = [
            -0.49733966449848194,
            -0.8909425976991894,
            -0.709353328514554,
            -0.40483478076837887,
            -0.8817006719953886,
        ];
        let expected_last_five_lead = [
            -0.9651269132969991,
            -0.30888310410390457,
            -0.003182174183612666,
            0.36030983330963545,
            -0.28983704937461496,
        ];

        assert!(
            msw_result.sine.len() >= 5,
            "MSW length too short for last 5 sine values"
        );
        assert!(
            msw_result.lead.len() >= 5,
            "MSW length too short for last 5 lead values"
        );

        let start_index = msw_result.sine.len() - 5;
        for (i, &value) in msw_result.sine[start_index..].iter().enumerate() {
            let expected_value = expected_last_five_sine[i];
            assert!(
                (value - expected_value).abs() < 1e-1,
                "Sine mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }

        for (i, &value) in msw_result.lead[start_index..].iter().enumerate() {
            let expected_value = expected_last_five_lead[i];
            assert!(
                (value - expected_value).abs() < 1e-1,
                "Lead mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }

        let period: usize = 5;
        for i in 0..(period - 1) {
            assert!(msw_result.sine[i].is_nan());
            assert!(msw_result.lead[i].is_nan());
        }

        let default_input = MswInput::with_default_candles(&candles);
        let default_msw_result = msw(&default_input).expect("Failed to calculate MSW defaults");
        assert_eq!(
            default_msw_result.sine.len(),
            close_prices.len(),
            "MSW default sine length mismatch"
        );
        assert_eq!(
            default_msw_result.lead.len(),
            close_prices.len(),
            "MSW default lead length mismatch"
        );
    }
}
