/// # Hilbert Transform - Dominant Cycle Period (HT_DCPERIOD)
///
/// This indicator attempts to determine the dominant cycle period present in a time series
/// using a Hilbert Transform approach. This can be used to gauge the
/// periodicity in the data and adapt strategy parameters based on the dominant cycle length.
///
/// ## Parameters
/// - *(none)*: HT_DCPERIOD does not require external parameters.
///
/// ## Errors
/// - **EmptyData**: ht_dcperiod: Input data slice is empty.
/// - **AllValuesNaN**: ht_dcperiod: All input data values are `NaN`.
/// - **NotEnoughValidData**: ht_dcperiod: Fewer than 32 valid (non-`NaN`) data points remain
///   after the first valid index (minimum needed to produce at least one HT_DCPERIOD value).
///
/// ## Returns
/// - **`Ok(HtDcPeriodOutput)`** on success, containing a `Vec<f64>` matching the input length,
///   with leading `NaN`s until HT_DCPERIOD values can be calculated.
/// - **`Err(HtDcPeriodError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum HtDcPeriodData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct HtDcPeriodOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, Default)]
pub struct HtDcPeriodParams;

#[derive(Debug, Clone)]
pub struct HtDcPeriodInput<'a> {
    pub data: HtDcPeriodData<'a>,
    pub params: HtDcPeriodParams,
}

impl<'a> HtDcPeriodInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: HtDcPeriodParams) -> Self {
        Self {
            data: HtDcPeriodData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: HtDcPeriodParams) -> Self {
        Self {
            data: HtDcPeriodData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: HtDcPeriodData::Candles {
                candles,
                source: "close",
            },
            params: HtDcPeriodParams::default(),
        }
    }
}

#[derive(Debug, Error)]
pub enum HtDcPeriodError {
    #[error("ht_dcperiod: Empty data provided.")]
    EmptyData,
    #[error("ht_dcperiod: All values are NaN.")]
    AllValuesNaN,
    #[error("ht_dcperiod: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

#[inline]
pub fn ht_dcperiod(input: &HtDcPeriodInput) -> Result<HtDcPeriodOutput, HtDcPeriodError> {
    let data: Vec<f64> = match &input.data {
        HtDcPeriodData::Candles { candles, source } => source_type(candles, source).to_vec(),
        HtDcPeriodData::Slice(slice) => slice.to_vec(),
    };

    if data.is_empty() {
        return Err(HtDcPeriodError::EmptyData);
    }

    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(HtDcPeriodError::AllValuesNaN),
    };

    let lookback = 32;
    if (data.len() - first_valid_idx) < lookback {
        return Err(HtDcPeriodError::NotEnoughValidData {
            needed: lookback,
            valid: data.len() - first_valid_idx,
        });
    }

    let mut out = vec![f64::NAN; data.len()];
    let rad2_deg = 180.0 / (4.0 * f64::atan(1.0));
    let mut period_wma_sub = 0.0;
    let mut period_wma_sum = 0.0;
    let mut trailing_wma_value = 0.0;
    let mut hilbert_idx = 0;
    let mut detrender_odd = [0.0; 3];
    let mut detrender_even = [0.0; 3];
    let mut q1_odd = [0.0; 3];
    let mut q1_even = [0.0; 3];
    let mut j_i_odd = [0.0; 3];
    let mut j_i_even = [0.0; 3];
    let mut j_q_odd = [0.0; 3];
    let mut j_q_even = [0.0; 3];
    let mut i1_for_odd_prev2 = 0.0;
    let mut i1_for_odd_prev3 = 0.0;
    let mut i1_for_even_prev2 = 0.0;
    let mut i1_for_even_prev3 = 0.0;
    let mut prev_q2 = 0.0;
    let mut prev_i2 = 0.0;
    let mut re = 0.0;
    let mut im = 0.0;
    let mut period = 0.0;
    let mut smooth_period = 0.0;
    let mut trailing_wma_idx = first_valid_idx as isize - lookback as isize;
    if trailing_wma_idx < 0 {
        trailing_wma_idx = 0;
    }
    let mut today = trailing_wma_idx;

    let mut temp_real = data[today as usize];
    period_wma_sub = temp_real;
    period_wma_sum = temp_real;
    today += 1;
    temp_real = data[today as usize];
    period_wma_sub += temp_real;
    period_wma_sum += temp_real * 2.0;
    today += 1;
    temp_real = data[today as usize];
    period_wma_sub += temp_real;
    period_wma_sum += temp_real * 3.0;
    today += 1;

    macro_rules! do_price_wma {
        ($new_price:expr, $smoothed_val:ident) => {
            period_wma_sub += $new_price;
            period_wma_sub -= trailing_wma_value;
            period_wma_sum += $new_price * 4.0;
            trailing_wma_value = data[trailing_wma_idx as usize];
            trailing_wma_idx += 1;
            $smoothed_val = period_wma_sum * 0.1;
            period_wma_sum -= period_wma_sub;
        };
    }

    let mut smoothed_value = 0.0;
    for _ in 0..9 {
        temp_real = data[today as usize];
        do_price_wma!(temp_real, smoothed_value);
        today += 1;
    }

    fn hilbert_transform_even(v: f64, buf: &mut [f64], idx: &mut usize) -> f64 {
        *idx = (*idx + 1) % 3;
        buf[*idx] = 0.0962 * v + 0.5769 * buf[*idx];
        buf[*idx]
    }

    fn hilbert_transform_odd(v: f64, buf: &mut [f64], idx: &mut usize) -> f64 {
        *idx = (*idx + 1) % 3;
        buf[*idx] = 0.0962 * v + 0.5769 * buf[*idx];
        buf[*idx]
    }

    let mut out_idx = 0;
    while (today as usize) < data.len() {
        let adjusted_prev_period = (0.075 * period) + 0.54;
        let today_val = data[today as usize];
        do_price_wma!(today_val, smoothed_value);

        let mut i2;
        let mut q2;
        if (today % 2) == 0 {
            let d_val =
                hilbert_transform_even(smoothed_value, &mut detrender_even, &mut hilbert_idx);
            let q1_val = hilbert_transform_even(d_val, &mut q1_even, &mut hilbert_idx);
            let ji_val = hilbert_transform_even(i1_for_even_prev3, &mut j_i_even, &mut hilbert_idx);
            let jq_val = hilbert_transform_even(q1_val, &mut j_q_even, &mut hilbert_idx);
            q2 = (0.2 * (q1_val + ji_val)) + (0.8 * prev_q2);
            i2 = (0.2 * (i1_for_even_prev3 - jq_val)) + (0.8 * prev_i2);
            i1_for_odd_prev3 = i1_for_odd_prev2;
            i1_for_odd_prev2 = d_val;
        } else {
            let d_val = hilbert_transform_odd(smoothed_value, &mut detrender_odd, &mut hilbert_idx);
            let q1_val = hilbert_transform_odd(d_val, &mut q1_odd, &mut hilbert_idx);
            let ji_val = hilbert_transform_odd(i1_for_odd_prev3, &mut j_i_odd, &mut hilbert_idx);
            let jq_val = hilbert_transform_odd(q1_val, &mut j_q_odd, &mut hilbert_idx);
            q2 = (0.2 * (q1_val + ji_val)) + (0.8 * prev_q2);
            i2 = (0.2 * (i1_for_odd_prev3 - jq_val)) + (0.8 * prev_i2);
            i1_for_even_prev3 = i1_for_even_prev2;
            i1_for_even_prev2 = d_val;
        }

        re = (0.2 * ((i2 * prev_i2) + (q2 * prev_q2))) + (0.8 * re);
        im = (0.2 * ((i2 * prev_q2) - (q2 * prev_i2))) + (0.8 * im);
        prev_i2 = i2;
        prev_q2 = q2;
        let temp_period = period;
        if im != 0.0 && re != 0.0 {
            period = 360.0 / (im.atan2(re) * rad2_deg);
        }
        let temp_real2 = 1.5 * temp_period;
        if period > temp_real2 {
            period = temp_real2;
        }
        let temp_real3 = 0.67 * temp_period;
        if period < temp_real3 {
            period = temp_real3;
        }
        if period < 6.0 {
            period = 6.0;
        } else if period > 50.0 {
            period = 50.0;
        }
        period = (0.2 * period) + (0.8 * temp_period);
        smooth_period = (0.33 * period) + (0.67 * smooth_period);

        if (today as usize) >= first_valid_idx + lookback {
            out[today as usize] = smooth_period;
            out_idx += 1;
        }
        today += 1;
    }

    Ok(HtDcPeriodOutput { values: out })
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_ht_dcperiod_basic() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = HtDcPeriodInput::from_candles(&candles, "close", HtDcPeriodParams);
        let output = ht_dcperiod(&input).expect("Failed to calculate HT_DCPERIOD");
        assert_eq!(output.values.len(), candles.close.len());
    }

    #[test]
    fn test_ht_dcperiod_from_slice() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = HtDcPeriodInput::from_candles(&candles, "close", HtDcPeriodParams);
        let result = ht_dcperiod(&input).expect("Failed to calculate HT_DCPERIOD from slice");
        assert_eq!(result.values.len(), candles.close.len());
    }

    #[test]
    fn test_ht_dcperiod_errors() {
        let empty_data: [f64; 0] = [];
        let input_empty = HtDcPeriodInput::from_slice(&empty_data, HtDcPeriodParams);
        let res_empty = ht_dcperiod(&input_empty);
        assert!(res_empty.is_err());

        let all_nan = [f64::NAN, f64::NAN, f64::NAN];
        let input_nan = HtDcPeriodInput::from_slice(&all_nan, HtDcPeriodParams);
        let res_nan = ht_dcperiod(&input_nan);
        assert!(res_nan.is_err());
    }

    #[test]
    fn test_ht_dcperiod_partial_data() {
        let data = [60000.0, f64::NAN, 59950.0, 59900.0];
        let input = HtDcPeriodInput::from_slice(&data, HtDcPeriodParams);
        let res = ht_dcperiod(&input);
        assert!(res.is_err());
    }

    #[test]
    #[ignore]
    fn test_ht_dcperiod_compare_python_talib() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = HtDcPeriodInput::from_candles(&candles, "close", HtDcPeriodParams);
        let output = ht_dcperiod(&input).expect("Failed to calculate HT_DCPERIOD");
        let len = output.values.len();
        let last_five = &output.values[len - 5..];
        let expected = [
            22.053215131184984,
            21.443032406123155,
            20.827144455403214,
            20.29555647396436,
            19.900406223996097,
        ];
        for (i, &val) in last_five.iter().enumerate() {
            let exp = expected[i];
            if !val.is_nan() {
                assert!(
                    (val - exp).abs() < 1e-1,
                    "HT_DCPERIOD mismatch at index {}: expected {}, got {}",
                    i,
                    exp,
                    val
                );
            }
        }
    }
}
