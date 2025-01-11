/// # Hilbert Transform Phasor (HT_PHASOR)
///
/// The Hilbert Transform Phasor, as implemented by TA-Lib's HT_PHASOR, returns
/// two components: inphase and quadrature. These can be used to analyze the
/// phase relationship of a price series, potentially giving insight into
/// cyclic or oscillatory behavior.
///
/// ## Parameters
/// This indicator does not have user-defined adjustable parameters beyond the input source.
///
/// ## Errors
/// - **EmptyData**: ht_phasor: Input data slice is empty.
/// - **AllValuesNaN**: ht_phasor: All input data values are `NaN`.
/// - **NotEnoughValidData**: ht_phasor: Fewer than 32 valid (non-`NaN`) data points remain
///   after the first valid index.
///
/// ## Returns
/// - **`Ok(HtPhasorOutput)`** on success, containing two `Vec<f64>` (inphase, quadrature)
///   matching the input length, with leading `NaN`s until the lookback window is filled.
/// - **`Err(HtPhasorError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};

#[derive(Debug, Clone)]
pub enum HtPhasorData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct HtPhasorOutput {
    pub inphase: Vec<f64>,
    pub quadrature: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct HtPhasorParams;

impl Default for HtPhasorParams {
    fn default() -> Self {
        Self
    }
}

#[derive(Debug, Clone)]
pub struct HtPhasorInput<'a> {
    pub data: HtPhasorData<'a>,
    pub params: HtPhasorParams,
}

impl<'a> HtPhasorInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str) -> Self {
        Self {
            data: HtPhasorData::Candles { candles, source },
            params: HtPhasorParams,
        }
    }

    pub fn from_slice(slice: &'a [f64]) -> Self {
        Self {
            data: HtPhasorData::Slice(slice),
            params: HtPhasorParams,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: HtPhasorData::Candles {
                candles,
                source: "close",
            },
            params: HtPhasorParams::default(),
        }
    }
}

use thiserror::Error;

#[derive(Debug, Error)]
pub enum HtPhasorError {
    #[error("ht_phasor: Empty data provided.")]
    EmptyData,
    #[error("ht_phasor: All values are NaN.")]
    AllValuesNaN,
    #[error("ht_phasor: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

#[inline]
pub fn ht_phasor(input: &HtPhasorInput) -> Result<HtPhasorOutput, HtPhasorError> {
    let data: &[f64] = match &input.data {
        HtPhasorData::Candles { candles, source } => source_type(candles, source),
        HtPhasorData::Slice(slice) => slice,
    };

    if data.is_empty() {
        return Err(HtPhasorError::EmptyData);
    }

    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(HtPhasorError::AllValuesNaN),
    };

    let lookback = 32;
    if (data.len() - first_valid_idx) < lookback {
        return Err(HtPhasorError::NotEnoughValidData {
            needed: lookback,
            valid: data.len() - first_valid_idx,
        });
    }

    let mut inphase = vec![f64::NAN; data.len()];
    let mut quadrature = vec![f64::NAN; data.len()];

    let rad2deg = 180.0 / (4.0 * (1.0f64).atan());
    let mut period = 0.0;
    let mut trailing_wma_idx = first_valid_idx;
    let mut today = trailing_wma_idx;

    let mut period_wma_sub = 0.0;
    let mut period_wma_sum = 0.0;
    let mut trailing_wma_value = 0.0;
    let mut smoothed_value = 0.0;

    let mut hilbert_idx = 0;
    let mut detrender_odd = [0.0; 3];
    let mut detrender_even = [0.0; 3];
    let mut q1_odd = [0.0; 3];
    let mut q1_even = [0.0; 3];
    let mut ji_odd = [0.0; 3];
    let mut ji_even = [0.0; 3];
    let mut jq_odd = [0.0; 3];
    let mut jq_even = [0.0; 3];

    let mut prev_q2 = 0.0;
    let mut prev_i2 = 0.0;
    let mut re = 0.0;
    let mut im = 0.0;

    let mut i1_for_odd_prev3 = 0.0;
    let mut i1_for_odd_prev2 = 0.0;
    let mut i1_for_even_prev3 = 0.0;
    let mut i1_for_even_prev2 = 0.0;

    if today + 2 >= data.len() {
        return Err(HtPhasorError::NotEnoughValidData {
            needed: lookback,
            valid: data.len() - first_valid_idx,
        });
    }
    period_wma_sub = data[today];
    period_wma_sum = data[today];
    today += 1;

    period_wma_sub += data[today];
    period_wma_sum += data[today] * 2.0;
    today += 1;

    period_wma_sub += data[today];
    period_wma_sum += data[today] * 3.0;
    today += 1;

    trailing_wma_value = 0.0;

    macro_rules! do_price_wma {
        ($new_price:expr, $smoothed_val:ident) => {
            period_wma_sub += $new_price;
            period_wma_sub -= trailing_wma_value;
            period_wma_sum += $new_price * 4.0;
            trailing_wma_value = data[trailing_wma_idx];
            trailing_wma_idx += 1;
            $smoothed_val = period_wma_sum * 0.1;
            period_wma_sum -= period_wma_sub;
        };
    }

    let mut i = 9;
    while i > 0 && today < data.len() {
        let cur_price = data[today];
        do_price_wma!(cur_price, smoothed_value);
        today += 1;
        i -= 1;
    }

    hilbert_idx = 0;
    detrender_odd.fill(0.0);
    detrender_even.fill(0.0);
    q1_odd.fill(0.0);
    q1_even.fill(0.0);
    ji_odd.fill(0.0);
    ji_even.fill(0.0);
    jq_odd.fill(0.0);
    jq_even.fill(0.0);

    macro_rules! do_hilbert_odd {
        ($buff:ident, $input:expr) => {
            $buff[hilbert_idx] = 0.0962 * $input + 0.5769 * $buff[(hilbert_idx + 2) % 3]
                - 0.5769 * $buff[(hilbert_idx + 1) % 3]
                - 0.0962 * $buff[hilbert_idx];
        };
    }

    macro_rules! do_hilbert_even {
        ($buff:ident, $input:expr) => {
            $buff[hilbert_idx] = 0.0962 * $input + 0.5769 * $buff[(hilbert_idx + 2) % 3]
                - 0.5769 * $buff[(hilbert_idx + 1) % 3]
                - 0.0962 * $buff[hilbert_idx];
        };
    }

    while today < data.len() {
        let cur_price = data[today];
        do_price_wma!(cur_price, smoothed_value);

        let old_period = period;

        if (today % 2) == 0 {
            do_hilbert_even!(detrender_even, smoothed_value);
            let detrender = detrender_even[hilbert_idx];
            do_hilbert_even!(q1_even, detrender);
            let q1 = q1_even[hilbert_idx];

            if today >= (first_valid_idx + lookback) {
                inphase[today] = i1_for_even_prev3;
                quadrature[today] = q1;
            }

            do_hilbert_even!(ji_even, i1_for_even_prev3);
            let j_i = ji_even[hilbert_idx];
            do_hilbert_even!(jq_even, q1);
            let j_q = jq_even[hilbert_idx];

            let q2 = 0.2 * (q1 + j_i) + 0.8 * prev_q2;
            let i2 = 0.2 * (i1_for_even_prev3 - j_q) + 0.8 * prev_i2;

            i1_for_odd_prev3 = i1_for_odd_prev2;
            i1_for_odd_prev2 = detrender;

            prev_q2 = q2;
            prev_i2 = i2;

            re = 0.2 * ((i2 * prev_i2) + (q2 * prev_q2)) + 0.8 * re;
            im = 0.2 * ((i2 * prev_q2) - (q2 * prev_i2)) + 0.8 * im;

            if re.abs() > 1e-14 && im.abs() > 1e-14 {
                period = 360.0 / (im.atan2(re) * rad2deg);
            }

            let mut temp_real2 = 1.5 * old_period;
            if period > temp_real2 {
                period = temp_real2;
            }
            temp_real2 = 0.67 * old_period;
            if period < temp_real2 {
                period = temp_real2;
            }
            if period < 6.0 {
                period = 6.0;
            } else if period > 50.0 {
                period = 50.0;
            }

            period = 0.2 * period + 0.8 * old_period;
        } else {
            do_hilbert_odd!(detrender_odd, smoothed_value);
            let detrender = detrender_odd[hilbert_idx];
            do_hilbert_odd!(q1_odd, detrender);
            let q1 = q1_odd[hilbert_idx];

            if today >= (first_valid_idx + lookback) {
                inphase[today] = i1_for_odd_prev3;
                quadrature[today] = q1;
            }

            do_hilbert_odd!(ji_odd, i1_for_odd_prev3);
            let j_i = ji_odd[hilbert_idx];
            do_hilbert_odd!(jq_odd, q1);
            let j_q = jq_odd[hilbert_idx];

            let q2 = 0.2 * (q1 + j_i) + 0.8 * prev_q2;
            let i2 = 0.2 * (i1_for_odd_prev3 - j_q) + 0.8 * prev_i2;

            i1_for_even_prev3 = i1_for_even_prev2;
            i1_for_even_prev2 = detrender;

            prev_q2 = q2;
            prev_i2 = i2;

            re = 0.2 * ((i2 * prev_i2) + (q2 * prev_q2)) + 0.8 * re;
            im = 0.2 * ((i2 * prev_q2) - (q2 * prev_i2)) + 0.8 * im;

            let old_period = period;
            if re.abs() > 1e-14 && im.abs() > 1e-14 {
                period = 360.0 / (im.atan2(re) * rad2deg);
            }

            let mut temp_real2 = 1.5 * old_period;
            if period > temp_real2 {
                period = temp_real2;
            }
            temp_real2 = 0.67 * old_period;
            if period < temp_real2 {
                period = temp_real2;
            }
            if period < 6.0 {
                period = 6.0;
            } else if period > 50.0 {
                period = 50.0;
            }

            period = 0.2 * period + 0.8 * old_period;
        }

        hilbert_idx = if hilbert_idx == 2 { 0 } else { hilbert_idx + 1 };
        today += 1;
    }

    Ok(HtPhasorOutput {
        inphase,
        quadrature,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_ht_phasor_empty_data() {
        let input = HtPhasorInput::from_slice(&[]);
        let result = ht_phasor(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Empty data"));
        }
    }

    #[test]
    fn test_ht_phasor_all_nan() {
        let input_data = [f64::NAN, f64::NAN, f64::NAN];
        let input = HtPhasorInput::from_slice(&input_data);
        let result = ht_phasor(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("All values are NaN"));
        }
    }

    #[test]
    fn test_ht_phasor_not_enough_valid_data() {
        let input_data = [1.0, 2.0, f64::NAN, 3.0, 4.0];
        let input = HtPhasorInput::from_slice(&input_data);
        let result = ht_phasor(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Not enough valid data"));
        }
    }

    #[test]
    fn test_ht_phasor_small_data_set() {
        let input_data = [42.0, 43.0, 44.0, 45.0];
        let input = HtPhasorInput::from_slice(&input_data);
        let result = ht_phasor(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Not enough valid data"));
        }
    }

    #[test]
    fn test_ht_phasor_basic_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = HtPhasorInput::from_candles(&candles, "close");
        let output = ht_phasor(&input).expect("Failed to calculate HT_PHASOR");
        assert_eq!(output.inphase.len(), candles.close.len());
        assert_eq!(output.quadrature.len(), candles.close.len());
    }

    #[test]
    fn test_ht_phasor_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = HtPhasorInput::with_default_candles(&candles);
        let output = ht_phasor(&input).expect("Failed to calculate HT_PHASOR with defaults");
        assert_eq!(output.inphase.len(), candles.close.len());
        assert_eq!(output.quadrature.len(), candles.close.len());
    }

    #[test]
    fn test_ht_phasor_slice_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input_first = HtPhasorInput::from_candles(&candles, "close");
        let result_first = ht_phasor(&input_first).expect("Failed first HT_PHASOR pass");
        assert_eq!(result_first.inphase.len(), candles.close.len());

        let input_second = HtPhasorInput::from_slice(&result_first.inphase);
        let result_second = ht_phasor(&input_second).expect("Failed second HT_PHASOR pass");
        assert_eq!(result_second.inphase.len(), result_first.inphase.len());
    }

    #[test]
    #[ignore]
    fn test_ht_phasor_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = HtPhasorInput::from_candles(&candles, "close");
        let output = ht_phasor(&input).expect("Failed to compute HT_PHASOR");
        assert_eq!(output.inphase.len(), candles.close.len());
        assert_eq!(output.quadrature.len(), candles.close.len());

        let expected_last_five_inphase = [
            -752.0760064607879,
            -822.2145746615109,
            2.2092217896941237,
            321.7074648360188,
            421.9545997363643,
        ];
        let expected_last_five_quadrature = [
            -333.4172586263912,
            1187.9186549942815,
            1474.2346895155745,
            630.4138451909436,
            13.555637312672577,
        ];
        let count = output.inphase.len();
        assert!(count >= 5, "Not enough output data for comparison");
        let inphase_tail = &output.inphase[count - 5..];
        let quadrature_tail = &output.quadrature[count - 5..];
        for (i, &val) in inphase_tail.iter().enumerate() {
            let diff = (val - expected_last_five_inphase[i]).abs();
            assert!(
                diff < 1e-1,
                "InPhase mismatch at index {}: got={}, expected={}",
                i,
                val,
                expected_last_five_inphase[i]
            );
        }
        for (i, &val) in quadrature_tail.iter().enumerate() {
            let diff = (val - expected_last_five_quadrature[i]).abs();
            assert!(
                diff < 1e-1,
                "Quadrature mismatch at index {}: got={}, expected={}",
                i,
                val,
                expected_last_five_quadrature[i]
            );
        }
    }

    #[test]
    fn test_ht_phasor_params_with_default_params() {
        let default_params = HtPhasorParams::default();
        let input_data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        let input = HtPhasorInput {
            data: HtPhasorData::Slice(&input_data),
            params: default_params,
        };
        let result = ht_phasor(&input);
        assert!(result.is_err());
    }
}
