/// # Hilbert Transform - Dominant Cycle Phase (HT_DCPHASE)
///
/// Computes the Dominant Cycle Phase using a Hilbert Transform, as described in
/// the TA-Lib implementation. This indicator returns a phase value (in degrees)
/// that can help analyze cycle behavior in the input data.
///
/// The output length matches the input length, with `NaN` values until
/// the required initial lookback (63 bars) plus the first valid non-`NaN` input.
///
/// ## Parameters
/// - *(none)*
///
/// ## Errors
/// - **EmptyData**: ht_dcphase: Input data slice is empty.
/// - **AllValuesNaN**: ht_dcphase: All input data values are `NaN`.
/// - **NotEnoughValidData**: ht_dcphase: Not enough valid (non-`NaN`) data points
///   for the required lookback of 63 bars.
///
/// ## Returns
/// - **`Ok(HtDcPhaseOutput)`** on success, containing a `Vec<f64>` of phase values.
/// - **`Err(HtDcPhaseError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum HtDcPhaseData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct HtDcPhaseOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct HtDcPhaseParams;

impl Default for HtDcPhaseParams {
    fn default() -> Self {
        Self
    }
}

#[derive(Debug, Clone)]
pub struct HtDcPhaseInput<'a> {
    pub data: HtDcPhaseData<'a>,
    pub params: HtDcPhaseParams,
}

impl<'a> HtDcPhaseInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: HtDcPhaseParams) -> Self {
        Self {
            data: HtDcPhaseData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: HtDcPhaseParams) -> Self {
        Self {
            data: HtDcPhaseData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: HtDcPhaseData::Candles {
                candles,
                source: "close",
            },
            params: HtDcPhaseParams::default(),
        }
    }
}

#[derive(Debug, Error)]
pub enum HtDcPhaseError {
    #[error("ht_dcphase: Empty data provided.")]
    EmptyData,
    #[error("ht_dcphase: All values are NaN.")]
    AllValuesNaN,
    #[error("ht_dcphase: Not enough valid data for lookback of 63 bars.")]
    NotEnoughValidData,
}

#[inline]
pub fn ht_dcphase(input: &HtDcPhaseInput) -> Result<HtDcPhaseOutput, HtDcPhaseError> {
    let data = match &input.data {
        HtDcPhaseData::Candles { candles, source } => source_type(candles, source),
        HtDcPhaseData::Slice(slice) => slice,
    };

    if data.is_empty() {
        return Err(HtDcPhaseError::EmptyData);
    }

    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(HtDcPhaseError::AllValuesNaN),
    };

    let lookback = 63;
    if (data.len() - first_valid_idx) < lookback {
        return Err(HtDcPhaseError::NotEnoughValidData);
    }

    let mut out = vec![f64::NAN; data.len()];

    let start = first_valid_idx + lookback;

    let mut period_wma_sub = 0.0;
    let mut period_wma_sum = 0.0;
    let mut trailing_wma_value = 0.0;
    let rad2deg_const = 45.0_f64 / (1.0_f64).atan();
    let deg2rad360_const = (1.0_f64).atan() * 8.0;
    let mut smooth_price = [0.0; 50];
    let mut smooth_price_idx = 0;
    let mut hilbert_idx = 0;
    let mut detrender_odd = [0.0; 3];
    let mut detrender_even = [0.0; 3];
    let mut q1_odd = [0.0; 3];
    let mut q1_even = [0.0; 3];
    let mut j_i_odd = [0.0; 3];
    let mut j_i_even = [0.0; 3];
    let mut j_q_odd = [0.0; 3];
    let mut j_q_even = [0.0; 3];
    let mut prev_q2 = 0.0;
    let mut prev_i2 = 0.0;
    let mut re = 0.0;
    let mut im = 0.0;
    let mut period = 0.0;
    let mut i1_for_odd_prev3 = 0.0;
    let mut i1_for_odd_prev2 = 0.0;
    let mut i1_for_even_prev3 = 0.0;
    let mut i1_for_even_prev2 = 0.0;
    let mut smooth_period = 0.0;

    let mut trailing_idx = first_valid_idx;
    let mut today = first_valid_idx;

    if trailing_idx + 2 >= data.len() {
        return Err(HtDcPhaseError::NotEnoughValidData);
    }

    let mut temp_real = data[trailing_idx];
    trailing_idx += 1;
    period_wma_sub = temp_real;
    period_wma_sum = temp_real;

    temp_real = data[trailing_idx];
    trailing_idx += 1;
    period_wma_sub += temp_real;
    period_wma_sum += temp_real * 2.0;

    temp_real = data[trailing_idx];
    trailing_idx += 1;
    period_wma_sub += temp_real;
    period_wma_sum += temp_real * 3.0;

    macro_rules! do_price_wma {
        ($new_price:expr, $smoothed_val:ident) => {
            period_wma_sub += $new_price;
            period_wma_sub -= trailing_wma_value;
            period_wma_sum += $new_price * 4.0;
            trailing_wma_value = data[trailing_idx];
            trailing_idx += 1;
            $smoothed_val = period_wma_sum * 0.1;
            period_wma_sum -= period_wma_sub;
        };
    }

    let mut smoothed_value = 0.0;
    let mut warmup = 34;
    while warmup > 0 && today < data.len() {
        temp_real = data[today];
        do_price_wma!(temp_real, smoothed_value);
        today += 1;
        warmup -= 1;
    }

    macro_rules! do_hilbert_even {
        ($arr:ident, $inp:expr) => {
            $arr[hilbert_idx] = 0.0962 * $inp + 0.5769 * $arr[hilbert_idx];
        };
    }
    macro_rules! do_hilbert_odd {
        ($arr:ident, $inp:expr) => {
            $arr[hilbert_idx] = 0.0962 * $inp + 0.5769 * $arr[hilbert_idx];
        };
    }

    while today < data.len() {
        let adjusted_prev_period = 0.075 * period + 0.54;
        let today_val = data[today];
        do_price_wma!(today_val, smoothed_value);
        smooth_price[smooth_price_idx] = smoothed_value;

        if today % 2 == 0 {
            do_hilbert_even!(detrender_even, smoothed_value);
            let detrender = detrender_even[hilbert_idx];
            do_hilbert_even!(q1_even, detrender);
            let q1 = q1_even[hilbert_idx];
            do_hilbert_even!(j_i_even, i1_for_even_prev3);
            let j_i = j_i_even[hilbert_idx];
            do_hilbert_even!(j_q_even, q1);
            let j_q = j_q_even[hilbert_idx];
            let q2 = 0.2 * (q1 + j_i) + 0.8 * prev_q2;
            let i2 = 0.2 * (i1_for_even_prev3 - j_q) + 0.8 * prev_i2;
            i1_for_odd_prev3 = i1_for_odd_prev2;
            i1_for_odd_prev2 = detrender;
            re = 0.2 * ((i2 * prev_i2) + (q2 * prev_q2)) + 0.8 * re;
            im = 0.2 * ((i2 * prev_q2) - (q2 * prev_i2)) + 0.8 * im;
            prev_q2 = q2;
            prev_i2 = i2;
        } else {
            do_hilbert_odd!(detrender_odd, smoothed_value);
            let detrender = detrender_odd[hilbert_idx];
            do_hilbert_odd!(q1_odd, detrender);
            let q1 = q1_odd[hilbert_idx];
            do_hilbert_odd!(j_i_odd, i1_for_odd_prev3);
            let j_i = j_i_odd[hilbert_idx];
            do_hilbert_odd!(j_q_odd, q1);
            let j_q = j_q_odd[hilbert_idx];
            let q2 = 0.2 * (q1 + j_i) + 0.8 * prev_q2;
            let i2 = 0.2 * (i1_for_odd_prev3 - j_q) + 0.8 * prev_i2;
            i1_for_even_prev3 = i1_for_even_prev2;
            i1_for_even_prev2 = detrender;
            re = 0.2 * ((i2 * prev_i2) + (q2 * prev_q2)) + 0.8 * re;
            im = 0.2 * ((i2 * prev_q2) - (q2 * prev_i2)) + 0.8 * im;
            prev_q2 = q2;
            prev_i2 = i2;
        }

        if im != 0.0 && re != 0.0 {
            period = 360.0 / (im.atan2(re) * rad2deg_const);
        }
        let temp_p1 = 1.5 * period;
        if period > temp_p1 {
            period = temp_p1;
        }
        let temp_p2 = 0.67 * period;
        if period < temp_p2 {
            period = temp_p2;
        }
        if period < 6.0 {
            period = 6.0;
        } else if period > 50.0 {
            period = 50.0;
        }
        period = 0.2 * period + 0.8 * adjusted_prev_period;
        smooth_period = 0.33 * period + 0.67 * smooth_period;

        let dc_period = smooth_period + 0.5;
        let dc_period_int = dc_period as i32;
        let mut real_part = 0.0;
        let mut imag_part = 0.0;
        let mut idx = smooth_price_idx;
        for i_cnt in 0..dc_period_int {
            let angle = (i_cnt as f64 * deg2rad360_const) / dc_period_int as f64;
            let val = smooth_price[idx];
            real_part += angle.sin() * val;
            imag_part += angle.cos() * val;
            idx = if idx == 0 { 49 } else { idx - 1 };
        }

        let mut dc_phase = if imag_part.abs() > 0.0 {
            (real_part / imag_part).atan() * rad2deg_const
        } else {
            0.0
        };
        if imag_part.abs() <= 0.01 {
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

        if today >= start {
            out[today] = dc_phase;
        }

        smooth_price_idx = if smooth_price_idx == 49 {
            0
        } else {
            smooth_price_idx + 1
        };
        hilbert_idx = if hilbert_idx == 2 { 0 } else { hilbert_idx + 1 };
        today += 1;
    }

    Ok(HtDcPhaseOutput { values: out })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    #[ignore]
    fn test_ht_dcphase_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input_default = HtDcPhaseInput::with_default_candles(&candles);
        let output_default = ht_dcphase(&input_default).expect("Failed ht_dcphase with default");
        assert_eq!(output_default.values.len(), candles.close.len());

        let input_custom =
            HtDcPhaseInput::from_candles(&candles, "hlc3", HtDcPhaseParams::default());
        let output_custom =
            ht_dcphase(&input_custom).expect("Failed ht_dcphase custom source=hlc3");
        assert_eq!(output_custom.values.len(), candles.close.len());
    }

    #[test]
    #[ignore]
    fn test_ht_dcphase_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let input = HtDcPhaseInput::from_candles(&candles, "close", HtDcPhaseParams::default());
        let result = ht_dcphase(&input).expect("Failed to calculate ht_dcphase");

        assert_eq!(
            result.values.len(),
            close_prices.len(),
            "ht_dcphase length mismatch"
        );

        let expected_last_five_ht_dcphase = [
            0.9170704770290286,
            -37.2309054587854,
            -25.095495030063375,
            -5.465895518256389,
            8.063832945820081,
        ];
        assert!(result.values.len() >= 5, "HT_DCPHASE length too short");
        let start_index = result.values.len() - 5;
        let result_last_five = &result.values[start_index..];
        for (i, &value) in result_last_five.iter().enumerate() {
            let expected_value = expected_last_five_ht_dcphase[i];
            assert!(
                (value - expected_value).abs() < 1e-6,
                "ht_dcphase mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }

        for i in 0..63 {
            assert!(
                result.values[i].is_nan(),
                "Expected NaN at index {} during lookback, got {}",
                i,
                result.values[i]
            );
        }

        let default_input = HtDcPhaseInput::with_default_candles(&candles);
        let default_result = ht_dcphase(&default_input).expect("Failed to calculate defaults");
        assert_eq!(default_result.values.len(), close_prices.len());
    }

    #[test]
    #[ignore]
    fn test_ht_dcphase_empty_data() {
        let input_data: [f64; 0] = [];
        let input = HtDcPhaseInput::from_slice(&input_data, HtDcPhaseParams::default());
        let result = ht_dcphase(&input);
        assert!(result.is_err());
        match result {
            Err(HtDcPhaseError::EmptyData) => {}
            _ => panic!("Expected EmptyData error."),
        }
    }

    #[test]
    #[ignore]
    fn test_ht_dcphase_all_nan() {
        let input_data = [f64::NAN; 100];
        let input = HtDcPhaseInput::from_slice(&input_data, HtDcPhaseParams::default());
        let result = ht_dcphase(&input);
        assert!(result.is_err());
        match result {
            Err(HtDcPhaseError::AllValuesNaN) => {}
            _ => panic!("Expected AllValuesNaN error."),
        }
    }

    #[test]
    #[ignore]
    fn test_ht_dcphase_not_enough_data() {
        let input_data = [f64::NAN, 1.0, 2.0, 3.0];
        let input = HtDcPhaseInput::from_slice(&input_data, HtDcPhaseParams::default());
        let result = ht_dcphase(&input);
        assert!(result.is_err());
        match result {
            Err(HtDcPhaseError::NotEnoughValidData) => {}
            _ => panic!("Expected NotEnoughValidData error."),
        }
    }

    #[test]
    #[ignore]
    fn test_ht_dcphase_minimum_data() {
        let mut input_data = vec![f64::NAN; 63];
        input_data.push(1.0);
        input_data.push(2.0);
        input_data.push(3.0);
        let input = HtDcPhaseInput::from_slice(&input_data, HtDcPhaseParams::default());
        let result = ht_dcphase(&input).unwrap();
        assert_eq!(result.values.len(), input_data.len());
        let last_vals = &result.values[(input_data.len() - 3)..];
        for val in last_vals {
            assert!(!val.is_nan());
        }
    }

    #[test]
    #[ignore]
    fn test_ht_dcphase_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let first_input =
            HtDcPhaseInput::from_candles(&candles, "close", HtDcPhaseParams::default());
        let first_result = ht_dcphase(&first_input).expect("Failed to calculate ht_dcphase first");

        let second_input =
            HtDcPhaseInput::from_slice(&first_result.values, HtDcPhaseParams::default());
        let second_result =
            ht_dcphase(&second_input).expect("Failed to calculate ht_dcphase second");

        assert_eq!(second_result.values.len(), first_result.values.len());
        for i in 63..second_result.values.len() {
            assert!(
                !second_result.values[i].is_nan(),
                "Expected no NaN after index 63 in second result, found NaN at index {}",
                i
            );
        }
    }
}
