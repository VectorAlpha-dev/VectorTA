/// # Ultimate Oscillator (ULTOSC)
///
/// The Ultimate Oscillator (ULTOSC) uses three different time periods (short, medium, long) to
/// capture short-term, intermediate-term, and long-term market volatility. It combines these
/// periods into one oscillator value ranging from 0 to 100. The formula calculates a running
/// total of `closeMinusTrueLow` divided by `trueRange` for each of the specified periods,
/// then combines these ratios with weights of 4 (short), 2 (medium), and 1 (long).
///
/// ## Parameters
/// - **timeperiod1**: The shortest window size (default = 7).
/// - **timeperiod2**: The medium window size (default = 14).
/// - **timeperiod3**: The longest window size (default = 28).
///
/// ## Errors
/// - **EmptyData**: ultosc: Input data slice is empty.
/// - **InvalidPeriods**: ultosc: One or more periods is zero or exceeds the data length.
/// - **NotEnoughValidData**: ultosc: Fewer than the largest period valid data points remain
///   after the first valid index (which requires i-1 to be valid).
/// - **AllValuesNaN**: ultosc: All input data values (or their necessary preceding values) are `NaN`.
///
/// ## Returns
/// - **`Ok(UltOscOutput)`** on success, containing a `Vec<f64>` matching the input length,
///   with leading `NaN`s until enough bars have passed to fill the largest window.
/// - **`Err(UltOscError)`** otherwise.
use crate::utilities::data_loader::Candles;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum UltOscData<'a> {
    Candles {
        candles: &'a Candles,
        high_src: &'a str,
        low_src: &'a str,
        close_src: &'a str,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct UltOscOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct UltOscParams {
    pub timeperiod1: Option<usize>,
    pub timeperiod2: Option<usize>,
    pub timeperiod3: Option<usize>,
}

impl Default for UltOscParams {
    fn default() -> Self {
        Self {
            timeperiod1: Some(7),
            timeperiod2: Some(14),
            timeperiod3: Some(28),
        }
    }
}

#[derive(Debug, Clone)]
pub struct UltOscInput<'a> {
    pub data: UltOscData<'a>,
    pub params: UltOscParams,
}

impl<'a> UltOscInput<'a> {
    pub fn from_candles(
        candles: &'a Candles,
        high_src: &'a str,
        low_src: &'a str,
        close_src: &'a str,
        params: UltOscParams,
    ) -> Self {
        Self {
            data: UltOscData::Candles {
                candles,
                high_src,
                low_src,
                close_src,
            },
            params,
        }
    }

    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: UltOscParams,
    ) -> Self {
        Self {
            data: UltOscData::Slices { high, low, close },
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: UltOscData::Candles {
                candles,
                high_src: "high",
                low_src: "low",
                close_src: "close",
            },
            params: UltOscParams::default(),
        }
    }

    pub fn get_timeperiod1(&self) -> usize {
        self.params
            .timeperiod1
            .unwrap_or_else(|| UltOscParams::default().timeperiod1.unwrap())
    }

    pub fn get_timeperiod2(&self) -> usize {
        self.params
            .timeperiod2
            .unwrap_or_else(|| UltOscParams::default().timeperiod2.unwrap())
    }

    pub fn get_timeperiod3(&self) -> usize {
        self.params
            .timeperiod3
            .unwrap_or_else(|| UltOscParams::default().timeperiod3.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum UltOscError {
    #[error("ultosc: Empty data provided.")]
    EmptyData,
    #[error("ultosc: Invalid periods: p1 = {p1}, p2 = {p2}, p3 = {p3}, data length = {data_len}")]
    InvalidPeriods {
        p1: usize,
        p2: usize,
        p3: usize,
        data_len: usize,
    },
    #[error("ultosc: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("ultosc: All values are NaN (or their preceding data is NaN).")]
    AllValuesNaN,
}

#[inline]
pub fn ultosc(input: &UltOscInput) -> Result<UltOscOutput, UltOscError> {
    let (high, low, close) = match &input.data {
        UltOscData::Candles {
            candles,
            high_src,
            low_src,
            close_src,
        } => {
            let high = candles.select_candle_field(high_src).unwrap();
            let low = candles.select_candle_field(low_src).unwrap();
            let close = candles.select_candle_field(close_src).unwrap();
            (high, low, close)
        }
        UltOscData::Slices { high, low, close } => (*high, *low, *close),
    };

    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(UltOscError::EmptyData);
    }

    let p1 = input.get_timeperiod1();
    let p2 = input.get_timeperiod2();
    let p3 = input.get_timeperiod3();
    let length = high.len();
    if p1 == 0 || p2 == 0 || p3 == 0 || p1 > length || p2 > length || p3 > length {
        return Err(UltOscError::InvalidPeriods {
            p1,
            p2,
            p3,
            data_len: length,
        });
    }

    let largest_period = p1.max(p2.max(p3));

    let first_possible = match (1..length).find(|&i| {
        !high[i - 1].is_nan()
            && !low[i - 1].is_nan()
            && !close[i - 1].is_nan()
            && !high[i].is_nan()
            && !low[i].is_nan()
            && !close[i].is_nan()
    }) {
        Some(idx) => idx,
        None => return Err(UltOscError::AllValuesNaN),
    };

    let start_idx = first_possible + (largest_period - 1);
    if start_idx >= length {
        return Err(UltOscError::NotEnoughValidData {
            needed: largest_period,
            valid: length.saturating_sub(first_possible),
        });
    }

    let mut cmtl = vec![f64::NAN; length];
    let mut tr = vec![f64::NAN; length];
    for i in 1..length {
        if high[i].is_nan() || low[i].is_nan() || close[i].is_nan() || close[i - 1].is_nan() {
            continue;
        }
        let true_low = low[i].min(close[i - 1]);
        let mut true_range = high[i] - low[i];
        let diff1 = (high[i] - close[i - 1]).abs();
        if diff1 > true_range {
            true_range = diff1;
        }
        let diff2 = (low[i] - close[i - 1]).abs();
        if diff2 > true_range {
            true_range = diff2;
        }
        cmtl[i] = close[i] - true_low;
        tr[i] = true_range;
    }

    let mut out_values = vec![f64::NAN; length];

    let mut sum1_a = 0.0;
    let mut sum1_b = 0.0;
    let mut sum2_a = 0.0;
    let mut sum2_b = 0.0;
    let mut sum3_a = 0.0;
    let mut sum3_b = 0.0;

    let prime_range_1 = start_idx.saturating_sub(p1 - 1)..start_idx;
    for i in prime_range_1 {
        if i < length && !cmtl[i].is_nan() && !tr[i].is_nan() {
            sum1_a += cmtl[i];
            sum1_b += tr[i];
        }
    }

    let prime_range_2 = start_idx.saturating_sub(p2 - 1)..start_idx;
    for i in prime_range_2 {
        if i < length && !cmtl[i].is_nan() && !tr[i].is_nan() {
            sum2_a += cmtl[i];
            sum2_b += tr[i];
        }
    }

    let prime_range_3 = start_idx.saturating_sub(p3 - 1)..start_idx;
    for i in prime_range_3 {
        if i < length && !cmtl[i].is_nan() && !tr[i].is_nan() {
            sum3_a += cmtl[i];
            sum3_b += tr[i];
        }
    }

    let end_idx = length - 1;
    let mut today = start_idx;
    while today <= end_idx {
        if !cmtl[today].is_nan() && !tr[today].is_nan() {
            sum1_a += cmtl[today];
            sum1_b += tr[today];
            sum2_a += cmtl[today];
            sum2_b += tr[today];
            sum3_a += cmtl[today];
            sum3_b += tr[today];
        }

        let v1 = if sum1_b != 0.0 {
            4.0 * (sum1_a / sum1_b)
        } else {
            0.0
        };
        let v2 = if sum2_b != 0.0 {
            2.0 * (sum2_a / sum2_b)
        } else {
            0.0
        };
        let v3 = if sum3_b != 0.0 { sum3_a / sum3_b } else { 0.0 };
        out_values[today] = 100.0 * (v1 + v2 + v3) / 7.0;

        let trailing_1 = today as isize - (p1 as isize) + 1;
        if trailing_1 >= 0 && (trailing_1 as usize) < length {
            let idx = trailing_1 as usize;
            if !cmtl[idx].is_nan() {
                sum1_a -= cmtl[idx];
            }
            if !tr[idx].is_nan() {
                sum1_b -= tr[idx];
            }
        }

        let trailing_2 = today as isize - (p2 as isize) + 1;
        if trailing_2 >= 0 && (trailing_2 as usize) < length {
            let idx = trailing_2 as usize;
            if !cmtl[idx].is_nan() {
                sum2_a -= cmtl[idx];
            }
            if !tr[idx].is_nan() {
                sum2_b -= tr[idx];
            }
        }

        let trailing_3 = today as isize - (p3 as isize) + 1;
        if trailing_3 >= 0 && (trailing_3 as usize) < length {
            let idx = trailing_3 as usize;
            if !cmtl[idx].is_nan() {
                sum3_a -= cmtl[idx];
            }
            if !tr[idx].is_nan() {
                sum3_b -= tr[idx];
            }
        }

        today += 1;
    }

    Ok(UltOscOutput { values: out_values })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_ultosc_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = UltOscParams {
            timeperiod1: None,
            timeperiod2: None,
            timeperiod3: None,
        };
        let input_default =
            UltOscInput::from_candles(&candles, "high", "low", "close", default_params);
        let output_default = ultosc(&input_default).expect("Failed ULTOSC with default params");
        assert_eq!(output_default.values.len(), candles.close.len());

        let custom_params = UltOscParams {
            timeperiod1: Some(5),
            timeperiod2: Some(10),
            timeperiod3: Some(20),
        };
        let input_custom =
            UltOscInput::from_candles(&candles, "high", "low", "close", custom_params);
        let output_custom = ultosc(&input_custom).expect("Failed ULTOSC with custom params");
        assert_eq!(output_custom.values.len(), candles.close.len());
    }

    #[test]
    fn test_ultosc_accuracy_check_on_sample() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let params = UltOscParams {
            timeperiod1: Some(7),
            timeperiod2: Some(14),
            timeperiod3: Some(28),
        };
        let input = UltOscInput::from_candles(&candles, "high", "low", "close", params);
        let result = ultosc(&input).expect("Failed to calculate ULTOSC");

        assert_eq!(result.values.len(), candles.close.len());

        let expected_last_five = [
            41.25546890298435,
            40.83865967175865,
            48.910324164909625,
            45.43113094857947,
            42.163165136766295,
        ];
        assert!(result.values.len() >= 5, "ULTOSC result length too short");
        let start_idx = result.values.len() - 5;
        let last_five = &result.values[start_idx..];
        for (i, &val) in last_five.iter().enumerate() {
            let exp = expected_last_five[i];
            assert!(
                (val - exp).abs() < 1e-8,
                "ULTOSC mismatch at last five index {}: expected {}, got {}",
                i,
                exp,
                val
            );
        }
    }

    #[test]
    fn test_ultosc_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = UltOscInput::with_default_candles(&candles);
        let result = ultosc(&input).expect("Failed to calculate ULTOSC with defaults");
        assert_eq!(result.values.len(), candles.close.len());
    }

    #[test]
    fn test_ultosc_zero_periods() {
        let input_high = [1.0, 2.0, 3.0];
        let input_low = [0.5, 1.5, 2.5];
        let input_close = [0.8, 1.8, 2.8];
        let params = UltOscParams {
            timeperiod1: Some(0),
            timeperiod2: Some(14),
            timeperiod3: Some(28),
        };
        let input = UltOscInput::from_slices(&input_high, &input_low, &input_close, params);
        let result = ultosc(&input);
        assert!(result.is_err(), "Expected an error for zero period");
    }

    #[test]
    fn test_ultosc_period_exceeds_data_length() {
        let input_high = [1.0, 2.0, 3.0];
        let input_low = [0.5, 1.5, 2.5];
        let input_close = [0.8, 1.8, 2.8];
        let params = UltOscParams {
            timeperiod1: Some(7),
            timeperiod2: Some(14),
            timeperiod3: Some(28),
        };
        let input = UltOscInput::from_slices(&input_high, &input_low, &input_close, params);
        let result = ultosc(&input);
        assert!(
            result.is_err(),
            "Expected an error for period exceeding data length"
        );
    }
}
