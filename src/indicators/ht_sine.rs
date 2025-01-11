/// # Hilbert Transform - SineWave (HT_SINE)
///
/// Transforms a price series using the Hilbert Transform to produce the `sine` and
/// `leadsine` components of a sine wave. Commonly used to identify cycle turning
/// points in market data.
///
/// ## Parameters
/// - **source**: Field to read from `candles`, defaults to `"close"` if unspecified.
///   Ignored if using a direct slice.
/// - **None**: No adjustable numerical parameter is typically provided for HT_SINE.
///
/// ## Errors
/// - **EmptyData**: ht_sine: No data provided.
/// - **AllValuesNaN**: ht_sine: All input data values are `NaN`.
/// - **NotEnoughValidData**: ht_sine: Not enough valid (non-`NaN`) data to compute at least
///   one output value. HT_SINE requires at least 63 data points after the first valid index.
///
/// ## Returns
/// - **`Ok(HtSineOutput)`** on success, containing `Vec<f64>` for both `sine` and `leadsine`,
///   each matching the input length, with leading `NaN`s until the algorithm stabilizes
///   (63 bars beyond the first valid data point).
/// - **`Err(HtSineError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};

#[derive(Debug, Clone)]
pub enum HtSineData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct HtSineOutput {
    pub sine: Vec<f64>,
    pub leadsine: Vec<f64>,
}

#[derive(Debug, Clone, Default)]
pub struct HtSineParams;

#[derive(Debug, Clone)]
pub struct HtSineInput<'a> {
    pub data: HtSineData<'a>,
    pub params: HtSineParams,
}

impl<'a> HtSineInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: HtSineParams) -> Self {
        Self {
            data: HtSineData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: HtSineParams) -> Self {
        Self {
            data: HtSineData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: HtSineData::Candles {
                candles,
                source: "close",
            },
            params: HtSineParams::default(),
        }
    }
}

use thiserror::Error;

#[derive(Debug, Error)]
pub enum HtSineError {
    #[error("ht_sine: Empty data provided.")]
    EmptyData,
    #[error("ht_sine: All values are NaN.")]
    AllValuesNaN,
    #[error(
        "ht_sine: Not enough valid data. Need at least 63 valid data points after the first valid index."
    )]
    NotEnoughValidData,
}

#[inline]
pub fn ht_sine(input: &HtSineInput) -> Result<HtSineOutput, HtSineError> {
    let data: &[f64] = match &input.data {
        HtSineData::Candles { candles, source } => source_type(candles, source),
        HtSineData::Slice(slice) => slice,
    };

    if data.is_empty() {
        return Err(HtSineError::EmptyData);
    }

    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(HtSineError::AllValuesNaN),
    };

    let lookback = 63;
    if (data.len() - first_valid_idx) < lookback {
        return Err(HtSineError::NotEnoughValidData);
    }

    let mut sine = vec![f64::NAN; data.len()];
    let mut leadsine = vec![f64::NAN; data.len()];

    let mut period_wma_sub;
    let mut period_wma_sum;
    let mut trailing_wma_value;
    let mut smoothed_value;

    let mut detrender_odd = [0.0; 3];
    let mut detrender_even = [0.0; 3];
    let mut q1_odd = [0.0; 3];
    let mut q1_even = [0.0; 3];
    let mut j_i_odd = [0.0; 3];
    let mut j_i_even = [0.0; 3];
    let mut j_q_odd = [0.0; 3];
    let mut j_q_even = [0.0; 3];

    let mut smooth_price = [0.0; 50];

    let mut period = 0.0;
    let mut smooth_period = 0.0;
    let mut re = 0.0;
    let mut im = 0.0;
    let mut prev_i2 = 0.0;
    let mut prev_q2 = 0.0;
    let mut i1_for_even_prev2 = 0.0;
    let mut i1_for_even_prev3 = 0.0;
    let mut i1_for_odd_prev2 = 0.0;
    let mut i1_for_odd_prev3 = 0.0;

    let mut hilbert_idx = 0;

    let rad2deg = 45.0_f64.to_radians().recip();
    let deg2rad = 1.0 / rad2deg;
    let const_deg2rad_by360 = (std::f64::consts::FRAC_PI_2) * 8.0;

    let mut today = first_valid_idx;
    let mut trailing_wma_idx = first_valid_idx;
    let mut out_idx = first_valid_idx + lookback - 1;

    period_wma_sub = data[trailing_wma_idx];
    period_wma_sum = data[trailing_wma_idx];
    trailing_wma_idx += 1;

    period_wma_sub += data[trailing_wma_idx];
    period_wma_sum += data[trailing_wma_idx] * 2.0;
    trailing_wma_idx += 1;

    period_wma_sub += data[trailing_wma_idx];
    period_wma_sum += data[trailing_wma_idx] * 3.0;
    trailing_wma_idx += 1;

    trailing_wma_value = 0.0;

    macro_rules! do_price_wma {
        ($price:expr, $storage:expr) => {
            period_wma_sub += $price;
            period_wma_sub -= trailing_wma_value;
            period_wma_sum += $price * 4.0;
            if trailing_wma_idx < data.len() {
                trailing_wma_value = data[trailing_wma_idx];
                trailing_wma_idx += 1;
            }
            $storage = period_wma_sum * 0.1;
            period_wma_sum -= period_wma_sub;
        };
    }

    while trailing_wma_idx < (first_valid_idx + 34).min(data.len()) {
        let tmp_val = data[trailing_wma_idx];
        do_price_wma!(tmp_val, smoothed_value);
    }

    fn do_hilbert_even(hilbert: &mut [f64; 3], val: f64, idx: usize) -> f64 {
        let a = -0.0962;
        let b = 0.5769;
        hilbert[idx] = val * a + hilbert[idx] * b;
        hilbert[idx]
    }

    fn do_hilbert_odd(hilbert: &mut [f64; 3], val: f64, idx: usize) -> f64 {
        let a = -0.0962;
        let b = 0.5769;
        hilbert[idx] = val * a + hilbert[idx] * b;
        hilbert[idx]
    }

    while today < data.len() && out_idx < data.len() {
        let val = data[today];
        let prev_period = period;
        let adjusted_prev_period = 0.075 * prev_period + 0.54;

        do_price_wma!(val, smoothed_value);
        smooth_price[out_idx % 50] = smoothed_value;

        if (today % 2) == 0 {
            let d = do_hilbert_even(&mut detrender_even, smoothed_value, hilbert_idx);
            let q = do_hilbert_even(&mut q1_even, d, hilbert_idx);
            let ji = do_hilbert_even(&mut j_i_even, i1_for_even_prev3, hilbert_idx);
            let jq = do_hilbert_even(&mut j_q_even, q, hilbert_idx);
            hilbert_idx = (hilbert_idx + 1) % 3;

            let q2 = 0.2 * (q + ji) + 0.8 * prev_q2;
            let i2 = 0.2 * (i1_for_even_prev3 - jq) + 0.8 * prev_i2;

            prev_q2 = q2;
            prev_i2 = i2;
            i1_for_odd_prev3 = i1_for_odd_prev2;
            i1_for_odd_prev2 = d;

            re = 0.2 * (i2 * prev_i2 + q2 * prev_q2) + 0.8 * re;
            im = 0.2 * (i2 * prev_q2 - q2 * prev_i2) + 0.8 * im;
        } else {
            let d = do_hilbert_odd(&mut detrender_odd, smoothed_value, hilbert_idx);
            let q = do_hilbert_odd(&mut q1_odd, d, hilbert_idx);
            let ji = do_hilbert_odd(&mut j_i_odd, i1_for_odd_prev3, hilbert_idx);
            let jq = do_hilbert_odd(&mut j_q_odd, q, hilbert_idx);
            hilbert_idx = (hilbert_idx + 1) % 3;

            let q2 = 0.2 * (q + ji) + 0.8 * prev_q2;
            let i2 = 0.2 * (i1_for_odd_prev3 - jq) + 0.8 * prev_i2;

            prev_q2 = q2;
            prev_i2 = i2;
            i1_for_even_prev3 = i1_for_even_prev2;
            i1_for_even_prev2 = d;

            re = 0.2 * (i2 * prev_i2 + q2 * prev_q2) + 0.8 * re;
            im = 0.2 * (i2 * prev_q2 - q2 * prev_i2) + 0.8 * im;
        }

        if im != 0.0 && re != 0.0 {
            period = 360.0 / (im.atan2(re) * rad2deg);
        }
        let limit_high = 1.5 * prev_period;
        if period > limit_high {
            period = limit_high;
        }
        let limit_low = 0.67 * prev_period;
        if period < limit_low {
            period = limit_low;
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
        let mut idx_sp = out_idx % 50;

        for i in 0..dc_period_int {
            if dc_period_int <= 0 {
                break;
            }
            let angle = (i as f64 * const_deg2rad_by360) / dc_period_int as f64;
            let sp = smooth_price[idx_sp];
            real_part += angle.sin() * sp;
            imag_part += angle.cos() * sp;
            idx_sp = if idx_sp == 0 { 49 } else { idx_sp - 1 };
        }

        let mut dc_phase = if imag_part.abs() > 1e-14 {
            real_part.atan2(imag_part) * rad2deg
        } else {
            0.0
        };
        dc_phase += 90.0;
        dc_phase += 360.0 / smooth_period;
        if imag_part < 0.0 {
            dc_phase += 180.0;
        }
        while dc_phase > 315.0 {
            dc_phase -= 360.0;
        }

        if out_idx < sine.len() {
            sine[out_idx] = (dc_phase * deg2rad).sin();
            leadsine[out_idx] = ((dc_phase + 45.0) * deg2rad).sin();
        }

        out_idx += 1;
        today += 1;
    }

    Ok(HtSineOutput { sine, leadsine })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_ht_sine_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = HtSineInput::with_default_candles(&candles);
        let output = ht_sine(&input).expect("HT_SINE failed");
        assert_eq!(output.sine.len(), candles.close.len());
        assert_eq!(output.leadsine.len(), candles.close.len());
    }

    #[test]
    fn test_ht_sine_empty_data() {
        let data: [f64; 0] = [];
        let input = HtSineInput::from_slice(&data, HtSineParams::default());
        let result = ht_sine(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                e.to_string().contains("Empty data"),
                "Expected 'Empty data' error, got: {}",
                e
            );
        }
    }

    #[test]
    fn test_ht_sine_all_nan_data() {
        let data = [f64::NAN, f64::NAN, f64::NAN];
        let input = HtSineInput::from_slice(&data, HtSineParams::default());
        let result = ht_sine(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                e.to_string().contains("All values are NaN"),
                "Expected 'AllValuesNaN' error, got: {}",
                e
            );
        }
    }

    #[test]
    fn test_ht_sine_not_enough_data_after_first_valid() {
        let data = [f64::NAN, f64::NAN, 10.0, 11.0, 12.0];
        let input = HtSineInput::from_slice(&data, HtSineParams::default());
        let result = ht_sine(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                e.to_string().contains("Not enough valid data"),
                "Expected 'NotEnoughValidData' error, got: {}",
                e
            );
        }
    }

    #[test]
    fn test_ht_sine_sufficient_data() {
        let mut data = vec![f64::NAN; 10];
        for i in 10..80 {
            data.push(i as f64);
        }
        let input = HtSineInput::from_slice(&data, HtSineParams::default());
        let result = ht_sine(&input);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.sine.len(), data.len());
        assert_eq!(output.leadsine.len(), data.len());
    }

    #[test]
    #[ignore]
    fn test_ht_sine_known_values() {
        let data = (0..70).map(|x| x as f64).collect::<Vec<f64>>();
        let input = HtSineInput::from_slice(&data, HtSineParams::default());
        let output = ht_sine(&input).expect("HT_SINE failed");
        let last_five_sine = &output.sine[output.sine.len() - 5..];
        let last_five_leadsine = &output.leadsine[output.leadsine.len() - 5..];
        let expected_sine = [
            0.016005215883690645,
            -0.6050286767270302,
            -0.42412821967644765,
            -0.09525324042591961,
            0.1402762671395893,
        ];
        let expected_lead = [
            0.7183336033827429,
            0.13518114133751574,
            0.3404534807798844,
            0.6365374059094095,
            0.7993054934261568,
        ];
        for (i, &val) in last_five_sine.iter().enumerate() {
            assert!(
                (val - expected_sine[i]).abs() < 1e-7,
                "Sine mismatch at {}: expected {}, got {}",
                i,
                expected_sine[i],
                val
            );
        }
        for (i, &val) in last_five_leadsine.iter().enumerate() {
            assert!(
                (val - expected_lead[i]).abs() < 1e-7,
                "LeadSine mismatch at {}: expected {}, got {}",
                i,
                expected_lead[i],
                val
            );
        }
    }
}
