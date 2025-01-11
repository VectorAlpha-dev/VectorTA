/// # HT_TRENDLINE (Hilbert Transform - Instantaneous Trendline)
///
/// Implements the Hilbert Transform - Instantaneous Trendline based on the original algorithm.
/// This indicator attempts to determine the instantaneous trendline of a series, often used
/// to identify cycles and smooth out short-term volatility. The output is aligned with the input length,
/// with `NaN` values before the first valid data point, similar to the SMA approach.
///
/// ## Parameters
/// - No user-adjustable parameters. This indicator is self-tuning.
///
/// ## Errors
/// - **EmptyData**: ht_trendline: Input data slice is empty.
/// - **AllValuesNaN**: ht_trendline: All input data values are `NaN`.
/// - **NotEnoughData**: ht_trendline: At least 64 data points (after the first valid index) are required
///   to compute the Hilbert Transform Trendline.
///
/// ## Returns
/// - **`Ok(HtTrendlineOutput)`** on success, containing a `Vec<f64>` matching the input length,
///   with leading `NaN`s until the HT_TRENDLINE can be computed.
/// - **`Err(HtTrendlineError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};
use std::f64::consts::PI;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum HtTrendlineData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct HtTrendlineOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct HtTrendlineParams;

impl Default for HtTrendlineParams {
    fn default() -> Self {
        Self
    }
}

#[derive(Debug, Clone)]
pub struct HtTrendlineInput<'a> {
    pub data: HtTrendlineData<'a>,
    pub params: HtTrendlineParams,
}

impl<'a> HtTrendlineInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str) -> Self {
        Self {
            data: HtTrendlineData::Candles { candles, source },
            params: HtTrendlineParams::default(),
        }
    }

    pub fn from_slice(slice: &'a [f64]) -> Self {
        Self {
            data: HtTrendlineData::Slice(slice),
            params: HtTrendlineParams::default(),
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: HtTrendlineData::Candles {
                candles,
                source: "close",
            },
            params: HtTrendlineParams::default(),
        }
    }
}

#[derive(Debug, Error)]
pub enum HtTrendlineError {
    #[error("ht_trendline: Empty data provided.")]
    EmptyData,
    #[error("ht_trendline: All values are NaN.")]
    AllValuesNaN,
    #[error("ht_trendline: At least 64 data points (after the first valid index) are required.")]
    NotEnoughData,
}

#[inline]
pub fn ht_trendline(input: &HtTrendlineInput) -> Result<HtTrendlineOutput, HtTrendlineError> {
    let data: &[f64] = match &input.data {
        HtTrendlineData::Candles { candles, source } => source_type(candles, source),
        HtTrendlineData::Slice(slice) => slice,
    };

    if data.is_empty() {
        return Err(HtTrendlineError::EmptyData);
    }

    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(HtTrendlineError::AllValuesNaN),
    };

    if (data.len() - first_valid_idx) < 64 {
        return Err(HtTrendlineError::NotEnoughData);
    }

    let mut out_real = vec![f64::NAN; data.len()];
    let end_idx = data.len() - 1;
    let lookback_total = 63;
    let start_idx = first_valid_idx + lookback_total;

    let mut period_wma_sub = 0.0;
    let mut period_wma_sum = 0.0;
    let mut trailing_wma_value = 0.0;
    let mut smoothed_value;
    let mut trailing_wma_idx = first_valid_idx;
    let mut today = trailing_wma_idx;

    {
        let mut temp = data[today];
        period_wma_sub = temp;
        period_wma_sum = temp;
        today += 1;
        temp = data[today];
        period_wma_sub += temp;
        period_wma_sum += temp * 2.0;
        today += 1;
        temp = data[today];
        period_wma_sub += temp;
        period_wma_sum += temp * 3.0;
        trailing_wma_value = 0.0;
    }

    macro_rules! do_price_wma {
        ($new_price:expr, $store:ident) => {
            period_wma_sub += $new_price;
            period_wma_sub -= trailing_wma_value;
            period_wma_sum += $new_price * 4.0;
            trailing_wma_value = data[trailing_wma_idx];
            trailing_wma_idx += 1;
            $store = period_wma_sum * 0.1;
            period_wma_sum -= period_wma_sub;
        };
    }

    for _ in 0..34 {
        let temp = data[today];
        do_price_wma!(temp, smoothed_value);
        today += 1;
    }

    let mut detrender_odd = [0.0; 3];
    let mut detrender_even = [0.0; 3];
    let mut q1_odd = [0.0; 3];
    let mut q1_even = [0.0; 3];
    let mut j_i_odd = [0.0; 3];
    let mut j_i_even = [0.0; 3];
    let mut j_q_odd = [0.0; 3];
    let mut j_q_even = [0.0; 3];

    let mut hilbert_idx = 0;
    let mut i1_for_odd_prev3 = 0.0;
    let mut i1_for_even_prev3 = 0.0;
    let mut i1_for_odd_prev2 = 0.0;
    let mut i1_for_even_prev2 = 0.0;
    let mut prev_i2 = 0.0;
    let mut prev_q2 = 0.0;
    let mut re = 0.0;
    let mut im = 0.0;
    let mut period = 0.0;
    let mut smooth_period = 0.0;

    let mut i_trend1 = 0.0;
    let mut i_trend2 = 0.0;
    let mut i_trend3 = 0.0;

    macro_rules! do_hilbert_even {
        ($array:ident, $input:expr) => {
            let temp_real = 0.0962 * $input + 0.5769 * $array[(hilbert_idx + 2) % 3];
            $array[hilbert_idx] = temp_real;
        };
    }

    macro_rules! do_hilbert_odd {
        ($array:ident, $input:expr) => {
            let temp_real = 0.0962 * $input + 0.5769 * $array[(hilbert_idx + 1) % 3];
            $array[hilbert_idx] = temp_real;
        };
    }

    let rad2deg = 180.0 / PI;

    while today <= end_idx {
        do_price_wma!(data[today], smoothed_value);

        if (today % 2) == 0 {
            do_hilbert_even!(detrender_even, smoothed_value);
            let detrender = detrender_even[hilbert_idx];
            do_hilbert_even!(q1_even, detrender);
            let q1 = q1_even[hilbert_idx];
            do_hilbert_even!(j_i_even, i1_for_even_prev3);
            let j_i = j_i_even[hilbert_idx];
            do_hilbert_even!(j_q_even, q1);
            let j_q = j_q_even[hilbert_idx];
            hilbert_idx = (hilbert_idx + 1) % 3;

            let q2 = 0.2 * (q1 + j_i) + 0.8 * prev_q2;
            let i2 = 0.2 * (i1_for_even_prev3 - j_q) + 0.8 * prev_i2;
            prev_q2 = q2;
            prev_i2 = i2;
            i1_for_odd_prev3 = i1_for_odd_prev2;
            i1_for_odd_prev2 = detrender;

            re = 0.2 * (i2 * prev_i2 + q2 * prev_q2) + 0.8 * re;
            im = 0.2 * (i2 * prev_q2 - q2 * prev_i2) + 0.8 * im;
        } else {
            do_hilbert_odd!(detrender_odd, smoothed_value);
            let detrender = detrender_odd[hilbert_idx];
            do_hilbert_odd!(q1_odd, detrender);
            let q1 = q1_odd[hilbert_idx];
            do_hilbert_odd!(j_i_odd, i1_for_odd_prev3);
            let j_i = j_i_odd[hilbert_idx];
            do_hilbert_odd!(j_q_odd, q1);
            let j_q = j_q_odd[hilbert_idx];
            hilbert_idx = (hilbert_idx + 1) % 3;

            let q2 = 0.2 * (q1 + j_i) + 0.8 * prev_q2;
            let i2 = 0.2 * (i1_for_odd_prev3 - j_q) + 0.8 * prev_i2;
            prev_q2 = q2;
            prev_i2 = i2;
            i1_for_even_prev3 = i1_for_even_prev2;
            i1_for_even_prev2 = detrender;

            re = 0.2 * (i2 * prev_i2 + q2 * prev_q2) + 0.8 * re;
            im = 0.2 * (i2 * prev_q2 - q2 * prev_i2) + 0.8 * im;
        }

        if im != 0.0 && re != 0.0 {
            period = 360.0 / (im.atan2(re) * rad2deg);
        }

        let temp_real2 = 1.5 * period;
        if period > temp_real2 {
            period = temp_real2;
        }
        let temp_real2 = 0.67 * period;
        if period < temp_real2 {
            period = temp_real2;
        }
        if period < 6.0 {
            period = 6.0;
        } else if period > 50.0 {
            period = 50.0;
        }
        period = 0.2 * period + 0.8 * period;

        smooth_period = 0.33 * period + 0.67 * smooth_period;

        let dc_period = smooth_period + 0.5;
        let dc_period_int = dc_period.floor() as usize;
        let mut sum_val = 0.0;
        let mut idx = today;
        for _ in 0..dc_period_int {
            sum_val += data[idx];
            if idx > 0 {
                idx -= 1;
            }
        }
        if dc_period_int > 0 {
            sum_val /= dc_period_int as f64;
        }

        let out_val = (4.0 * sum_val + 3.0 * i_trend1 + 2.0 * i_trend2 + i_trend3) / 10.0;
        i_trend3 = i_trend2;
        i_trend2 = i_trend1;
        i_trend1 = sum_val;

        if today >= start_idx {
            out_real[today] = out_val;
        }
        today += 1;
    }

    Ok(HtTrendlineOutput { values: out_real })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_ht_trendline_empty_data() {
        let data: [f64; 0] = [];
        let input = HtTrendlineInput::from_slice(&data);
        let result = ht_trendline(&input);
        assert!(result.is_err(), "Expected an error for empty data");
        if let Err(e) = result {
            assert!(
                e.to_string().contains("Empty data"),
                "Expected 'Empty data' error, got: {}",
                e
            );
        }
    }

    #[test]
    fn test_ht_trendline_all_nan() {
        let data = [f64::NAN, f64::NAN, f64::NAN, f64::NAN];
        let input = HtTrendlineInput::from_slice(&data);
        let result = ht_trendline(&input);
        assert!(result.is_err(), "Expected an error for all NaN data");
        if let Err(e) = result {
            assert!(
                e.to_string().contains("All values are NaN"),
                "Expected 'All values are NaN' error, got: {}",
                e
            );
        }
    }

    #[test]
    fn test_ht_trendline_not_enough_data() {
        let data = [60000.0; 63];
        let input = HtTrendlineInput::from_slice(&data);
        let result = ht_trendline(&input);
        assert!(result.is_err(), "Expected an error for not enough data");
        if let Err(e) = result {
            assert!(
                e.to_string().contains("At least 64 data points"),
                "Expected 'NotEnoughData' error, got: {}",
                e
            );
        }
    }

    #[test]
    fn test_ht_trendline_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = HtTrendlineInput::with_default_candles(&candles);
        let output = ht_trendline(&input).expect("Failed HT_TRENDLINE with default candles");
        assert_eq!(
            output.values.len(),
            candles.close.len(),
            "HT length mismatch with default candles"
        );
    }

    #[test]
    fn test_ht_trendline_output_length() {
        let data: Vec<f64> = (0..80).map(|i| i as f64 + 50000.0).collect();
        let input = HtTrendlineInput::from_slice(&data);
        let output = ht_trendline(&input).expect("Failed HT_TRENDLINE on synthetic data");
        assert_eq!(output.values.len(), 80, "HT output length mismatch");
    }

    #[test]
    fn test_ht_trendline_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = HtTrendlineInput::with_default_candles(&candles);
        match input.data {
            HtTrendlineData::Candles { source, .. } => {
                assert_eq!(source, "close", "Expected default source to be 'close'");
            }
            _ => panic!("Expected HtTrendlineData::Candles variant"),
        }
    }

    #[test]
    #[ignore]
    fn test_ht_trendline_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = HtTrendlineInput::from_candles(&candles, "close");
        let ht_result = ht_trendline(&input).expect("Failed to calculate HT_TRENDLINE");
        assert_eq!(
            ht_result.values.len(),
            candles.close.len(),
            "HT_TRENDLINE length mismatch"
        );

        let expected_last_five = [
            59638.11903820817,
            59497.255919442876,
            59431.08089591567,
            59391.23316017316,
            59372.19238095238,
        ];
        assert!(
            ht_result.values.len() >= 5,
            "HT_TRENDLINE length is too short to validate last five values"
        );
        let start_index = ht_result.values.len() - 5;
        let result_last_five = &ht_result.values[start_index..];
        for (i, &value) in result_last_five.iter().enumerate() {
            let expected_value = expected_last_five[i];
            if !value.is_nan() {
                assert!(
                    (value - expected_value).abs() < 1e-1,
                    "HT_TRENDLINE mismatch at index {}: expected {}, got {}",
                    i,
                    expected_value,
                    value
                );
            }
        }

        for i in 0..start_index {
            if i < (63 + 1) {
                assert!(
                    ht_result.values[i].is_nan(),
                    "Expected leading NaNs near index {}, got {}",
                    i,
                    ht_result.values[i]
                );
            }
        }
    }

    #[test]
    #[ignore]
    fn test_ht_trendline_slice_data_reinput() {
        let data: Vec<f64> = (50000..50100).map(|i| i as f64).collect();
        let first_input = HtTrendlineInput::from_slice(&data);
        let first_result =
            ht_trendline(&first_input).expect("Failed to calculate first HT_TRENDLINE");

        assert_eq!(
            first_result.values.len(),
            data.len(),
            "First HT_TRENDLINE output length mismatch"
        );

        let second_input = HtTrendlineInput::from_slice(&first_result.values);
        let second_result =
            ht_trendline(&second_input).expect("Failed to calculate second HT_TRENDLINE");

        assert_eq!(
            second_result.values.len(),
            first_result.values.len(),
            "Second HT_TRENDLINE output length mismatch"
        );

        for i in 64..second_result.values.len() {
            assert!(
                !second_result.values[i].is_nan(),
                "Expected no NaN after index 64, but found NaN at index {}",
                i
            );
        }
    }
}
