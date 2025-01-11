/// # Hilbert Transform - Trend vs Cycle Mode (HT_TRENDMODE)
///
/// This indicator determines whether the market is in a trend or in a cycle phase
/// by applying the Hilbert Transform. It outputs `0.0` for cycle mode and `1.0` for
/// trend mode, matching the input length, with leading `NaN` until enough data is
/// available (requires at least 63 valid data points).
///
/// ## Parameters
/// - **source**: The candle source (e.g., `"close"`) when using `Candles`. Defaults to `"close"`.
///
/// ## Errors
/// - **EmptyData**: ht_trendmode: Input data slice is empty.
/// - **AllValuesNaN**: ht_trendmode: All input data values are `NaN`.
/// - **NotEnoughData**: ht_trendmode: Not enough data (fewer than 63 valid points remain
///   after the first valid index).
///
/// ## Returns
/// - **`Ok(HtTrendModeOutput)`** on success, containing a `Vec<f64>` matching the input length,
///   with leading `NaN`s until the calculation window is filled.
/// - **`Err(HtTrendModeError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum HtTrendModeData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct HtTrendModeParams;

impl Default for HtTrendModeParams {
    fn default() -> Self {
        Self
    }
}

#[derive(Debug, Clone)]
pub struct HtTrendModeInput<'a> {
    pub data: HtTrendModeData<'a>,
    pub params: HtTrendModeParams,
}

impl<'a> HtTrendModeInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: HtTrendModeParams) -> Self {
        Self {
            data: HtTrendModeData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: HtTrendModeParams) -> Self {
        Self {
            data: HtTrendModeData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: HtTrendModeData::Candles {
                candles,
                source: "close",
            },
            params: HtTrendModeParams::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct HtTrendModeOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Error)]
pub enum HtTrendModeError {
    #[error("ht_trendmode: Empty data provided.")]
    EmptyData,
    #[error("ht_trendmode: All values are NaN.")]
    AllValuesNaN,
    #[error("ht_trendmode: Not enough data (need at least 63 valid points).")]
    NotEnoughData,
}

#[inline]
pub fn ht_trendmode(input: &HtTrendModeInput) -> Result<HtTrendModeOutput, HtTrendModeError> {
    let data: &[f64] = match &input.data {
        HtTrendModeData::Candles { candles, source } => source_type(candles, source),
        HtTrendModeData::Slice(slice) => slice,
    };

    if data.is_empty() {
        return Err(HtTrendModeError::EmptyData);
    }

    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(HtTrendModeError::AllValuesNaN),
    };

    let valid_count = data.len() - first_valid_idx;
    if valid_count < 63 {
        return Err(HtTrendModeError::NotEnoughData);
    }

    let mut out_values = vec![f64::NAN; data.len()];
    let lookback = 63;
    let start_idx = first_valid_idx + lookback;
    let mut smooth_price = vec![0.0; 50];
    let mut hilbert_idx = 0usize;
    let mut detrender_odd = [0.0; 3];
    let mut detrender_even = [0.0; 3];
    let mut q1_odd = [0.0; 3];
    let mut q1_even = [0.0; 3];
    let mut ji_odd = [0.0; 3];
    let mut ji_even = [0.0; 3];
    let mut jq_odd = [0.0; 3];
    let mut jq_even = [0.0; 3];
    let rad2deg = 45.0_f64.atan();
    let deg2rad = 1.0 / rad2deg;
    let const_deg2rad_360 = 8.0 * 45.0_f64.atan();
    let mut i_trend1 = 0.0;
    let mut i_trend2 = 0.0;
    let mut i_trend3 = 0.0;
    let mut days_in_trend = 0;
    let mut prev_dc_phase = 0.0;
    let mut dc_phase = 0.0;
    let mut prev_sine = 0.0;
    let mut sine = 0.0;
    let mut prev_lead_sine = 0.0;
    let mut lead_sine = 0.0;
    let mut period = 0.0;
    let mut smooth_period = 0.0;
    let mut prev_q2 = 0.0;
    let mut prev_i2 = 0.0;
    let mut re = 0.0;
    let mut im = 0.0;
    let mut i1_for_even_prev2 = 0.0;
    let mut i1_for_even_prev3 = 0.0;
    let mut i1_for_odd_prev2 = 0.0;
    let mut i1_for_odd_prev3 = 0.0;
    let mut trailing_wma_value = 0.0;
    let mut period_wma_sub = 0.0;
    let mut period_wma_sum = 0.0;
    let mut smoothed_value = 0.0;
    let mut trailing_wma_idx = first_valid_idx;
    let mut today = trailing_wma_idx;

    {
        let mut tmp = data[today];
        period_wma_sub = tmp;
        period_wma_sum = tmp;
        today += 1;
        tmp = data[today];
        period_wma_sub += tmp;
        period_wma_sum += tmp * 2.0;
        today += 1;
        tmp = data[today];
        period_wma_sub += tmp;
        period_wma_sum += tmp * 3.0;
        today += 1;
    }

    macro_rules! do_price_wma {
        ($new_price:expr, $store:expr) => {{
            period_wma_sub += $new_price;
            period_wma_sub -= trailing_wma_value;
            period_wma_sum += $new_price * 4.0;
            trailing_wma_value = data[trailing_wma_idx];
            trailing_wma_idx += 1;
            $store = period_wma_sum * 0.1;
            period_wma_sum -= period_wma_sub;
        }};
    }

    for _ in 0..34 {
        let x = data[today];
        do_price_wma!(x, smoothed_value);
        today += 1;
    }

    fn do_hilbert_even(vars: &mut [f64; 3], input: f64, idx: &mut usize) -> f64 {
        let temp = 0.0962 * input + 0.5769 * vars[(*idx + 2) % 3]
            - 0.5769 * vars[(*idx + 1) % 3]
            - 0.0962 * vars[*idx];
        vars[*idx] = temp;
        *idx = (*idx + 2) % 3;
        temp
    }

    fn do_hilbert_odd(vars: &mut [f64; 3], input: f64, idx: &mut usize) -> f64 {
        let temp = 0.0962 * input + 0.5769 * vars[(*idx + 2) % 3]
            - 0.5769 * vars[(*idx + 1) % 3]
            - 0.0962 * vars[*idx];
        vars[*idx] = temp;
        *idx = (*idx + 2) % 3;
        temp
    }

    let mut smooth_price_idx = 0usize;
    while today < data.len() {
        let adjusted_prev_period = 0.075 * period + 0.54;
        let x = data[today];
        do_price_wma!(x, smoothed_value);
        smooth_price[smooth_price_idx] = smoothed_value;

        let mut detrender_val = 0.0;
        let mut q1_val = 0.0;
        let mut ji_val = 0.0;
        let mut jq_val = 0.0;

        if (today % 2) == 0 {
            detrender_val = do_hilbert_even(&mut detrender_even, smoothed_value, &mut hilbert_idx);
            q1_val = do_hilbert_even(&mut q1_even, detrender_val, &mut hilbert_idx);
            ji_val = do_hilbert_even(&mut ji_even, i1_for_even_prev3, &mut hilbert_idx);
            jq_val = do_hilbert_even(&mut jq_even, q1_val, &mut hilbert_idx);
            i1_for_odd_prev3 = i1_for_odd_prev2;
            i1_for_odd_prev2 = detrender_val;
        } else {
            detrender_val = do_hilbert_odd(&mut detrender_odd, smoothed_value, &mut hilbert_idx);
            q1_val = do_hilbert_odd(&mut q1_odd, detrender_val, &mut hilbert_idx);
            ji_val = do_hilbert_odd(&mut ji_odd, i1_for_odd_prev3, &mut hilbert_idx);
            jq_val = do_hilbert_odd(&mut jq_odd, q1_val, &mut hilbert_idx);
            i1_for_even_prev3 = i1_for_even_prev2;
            i1_for_even_prev2 = detrender_val;
        }

        let q2 = 0.2 * (q1_val + ji_val) + 0.8 * prev_q2;
        let i2 =
            0.2 * (if (today % 2) == 0 {
                i1_for_even_prev3 - jq_val
            } else {
                i1_for_odd_prev3 - jq_val
            }) + 0.8 * prev_i2;

        re = 0.2 * ((i2 * prev_i2) + (q2 * prev_q2)) + 0.8 * re;
        im = 0.2 * ((i2 * prev_q2) - (q2 * prev_i2)) + 0.8 * im;
        prev_q2 = q2;
        prev_i2 = i2;
        let temp_period = period;
        if im != 0.0 && re != 0.0 {
            period = 360.0 / (im.atan2(re) * rad2deg);
        }
        let temp2 = 1.5 * temp_period;
        if period > temp2 {
            period = temp2;
        }
        let temp2 = 0.67 * temp_period;
        if period < temp2 {
            period = temp2;
        }
        if period < 6.0 {
            period = 6.0;
        } else if period > 50.0 {
            period = 50.0;
        }
        period = 0.2 * period + 0.8 * temp_period;
        smooth_period = 0.33 * period + 0.67 * smooth_period;
        prev_dc_phase = dc_phase;
        let dc_period = smooth_period + 0.5;
        let dc_period_int = dc_period as i32;
        let mut real_part = 0.0;
        let mut imag_part = 0.0;
        let mut idx = smooth_price_idx;
        for i in 0..dc_period_int {
            let angle = (i as f64 * const_deg2rad_360) / (dc_period_int as f64);
            let val = smooth_price[idx];
            real_part += angle.sin() * val;
            imag_part += angle.cos() * val;
            if idx == 0 {
                idx = 49;
            } else {
                idx -= 1;
            }
        }
        if imag_part.abs() > 0.0 {
            dc_phase = (real_part.atan2(imag_part)) * rad2deg;
        } else if imag_part.abs() <= 0.01 {
            if real_part < 0.0 {
                dc_phase -= 90.0;
            } else if real_part > 0.0 {
                dc_phase += 90.0;
            }
        }
        dc_phase += 90.0;
        dc_phase += 360.0 / smooth_period;
        if imag_part < 0.0 {
            dc_phase += 180.0;
        }
        if dc_phase > 315.0 {
            dc_phase -= 360.0;
        }
        prev_sine = sine;
        prev_lead_sine = lead_sine;
        sine = (dc_phase * deg2rad).sin();
        lead_sine = ((dc_phase + 45.0) * deg2rad).sin();
        let dc_period_int = dc_period as i32;
        let mut sump = 0.0;
        let mut idxp = today;
        for _ in 0..dc_period_int {
            sump += data[idxp];
            if idxp == 0 {
                break;
            }
            idxp = idxp.saturating_sub(1);
        }
        let avg_dc = if dc_period_int > 0 {
            sump / (dc_period_int as f64)
        } else {
            0.0
        };
        let trendline = (4.0 * avg_dc + 3.0 * i_trend1 + 2.0 * i_trend2 + i_trend3) / 10.0;
        i_trend3 = i_trend2;
        i_trend2 = i_trend1;
        i_trend1 = avg_dc;
        let mut trend = 1.0;
        if ((sine > lead_sine) && (prev_sine <= prev_lead_sine))
            || ((sine < lead_sine) && (prev_sine >= prev_lead_sine))
        {
            days_in_trend = 0;
            trend = 0.0;
        }
        days_in_trend += 1;
        if days_in_trend < (0.5 * smooth_period) as i32 {
            trend = 0.0;
        }
        let diff_phase = dc_phase - prev_dc_phase;
        if smooth_period != 0.0
            && (diff_phase > (0.67 * 360.0 / smooth_period))
            && (diff_phase < (1.5 * 360.0 / smooth_period))
        {
            trend = 0.0;
        }
        let current_price = smooth_price[smooth_price_idx];
        if trendline != 0.0 && ((current_price - trendline).abs() / trendline).abs() >= 0.015 {
            trend = 1.0;
        }
        if today >= start_idx {
            out_values[today] = trend;
        }
        smooth_price_idx = (smooth_price_idx + 1) % 50;
        today += 1;
    }

    Ok(HtTrendModeOutput { values: out_values })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_ht_trendmode_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = HtTrendModeParams::default();
        let input_default = HtTrendModeInput::from_candles(&candles, "close", default_params);
        let output_default =
            ht_trendmode(&input_default).expect("Failed HT_TRENDMODE with defaults");
        assert_eq!(output_default.values.len(), candles.close.len());

        let input_custom = HtTrendModeInput::from_candles(&candles, "hl2", HtTrendModeParams {});
        let output_custom = ht_trendmode(&input_custom).expect("Failed HT_TRENDMODE custom source");
        assert_eq!(output_custom.values.len(), candles.close.len());
    }

    #[test]
    #[ignore]
    fn test_ht_trendmode_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let input = HtTrendModeInput::with_default_candles(&candles);
        let result = ht_trendmode(&input).expect("Failed to calculate HT_TRENDMODE");

        assert_eq!(
            result.values.len(),
            close_prices.len(),
            "HT_TRENDMODE length mismatch"
        );

        let expected_last_five = [1.0, 1.0, 0.0, 0.0, 0.0];
        assert!(
            result.values.len() >= 5,
            "HT_TRENDMODE length too short for accuracy check"
        );
        let start_index = result.values.len() - 5;
        let actual_last_five = &result.values[start_index..];
        for (i, &value) in actual_last_five.iter().enumerate() {
            let expected_value = expected_last_five[i];
            assert!(
                (value - expected_value).abs() < 1e-9,
                "HT_TRENDMODE mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }

        for i in 0..63 {
            assert!(
                result.values[i].is_nan(),
                "Expected leading NaN for warmup period at index {}",
                i
            );
        }
    }

    #[test]
    fn test_ht_trendmode_params_with_default_params() {
        let default_params = HtTrendModeParams::default();
        let _ = default_params;
    }

    #[test]
    fn test_ht_trendmode_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = HtTrendModeInput::with_default_candles(&candles);
        match input.data {
            HtTrendModeData::Candles { source, .. } => {
                assert_eq!(source, "close", "Expected default source to be 'close'");
            }
            _ => panic!("Expected HtTrendModeData::Candles variant"),
        }
    }

    #[test]
    fn test_ht_trendmode_small_data() {
        let input_data = [10.0, 20.0, 30.0, 40.0, 50.0];
        let input = HtTrendModeInput::from_slice(&input_data, HtTrendModeParams {});
        let result = ht_trendmode(&input);
        assert!(result.is_err(), "Expected error for not enough data");
        if let Err(e) = result {
            assert!(
                e.to_string().contains("Not enough data"),
                "Expected 'Not enough data' error message, got: {}",
                e
            );
        }
    }

    #[test]
    fn test_ht_trendmode_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let first_input = HtTrendModeInput::from_candles(&candles, "close", HtTrendModeParams {});
        let first_result = ht_trendmode(&first_input).expect("Failed to calculate HT_TRENDMODE");

        assert_eq!(
            first_result.values.len(),
            candles.close.len(),
            "First HT_TRENDMODE output length mismatch"
        );

        let second_input =
            HtTrendModeInput::from_slice(&first_result.values, HtTrendModeParams::default());
        let second_result = ht_trendmode(&second_input).expect("Failed second HT_TRENDMODE");
        assert_eq!(
            second_result.values.len(),
            first_result.values.len(),
            "Second HT_TRENDMODE output length mismatch"
        );
    }

    #[test]
    fn test_ht_trendmode_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = HtTrendModeInput::from_candles(&candles, "close", HtTrendModeParams {});
        let result = ht_trendmode(&input).expect("Failed to calculate HT_TRENDMODE");

        if result.values.len() > 100 {
            for i in 100..result.values.len() {
                assert!(
                    !result.values[i].is_nan(),
                    "Expected no NaN after index 100, but found NaN at index {}",
                    i
                );
            }
        }
    }

    #[test]
    fn test_ht_trendmode_nan_all() {
        let data = [f64::NAN, f64::NAN, f64::NAN];
        let input = HtTrendModeInput::from_slice(&data, HtTrendModeParams::default());
        let result = ht_trendmode(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("All values are NaN"));
        }
    }
}
