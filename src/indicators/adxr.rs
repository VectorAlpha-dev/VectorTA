use crate::utilities::data_loader::Candles;
use std::error::Error;

#[derive(Debug, Clone)]
pub enum AdxrData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct AdxrParams {
    pub period: Option<usize>,
}

impl Default for AdxrParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct AdxrInput<'a> {
    pub data: AdxrData<'a>,
    pub params: AdxrParams,
}

impl<'a> AdxrInput<'a> {
    pub fn from_candles(candles: &'a Candles, params: AdxrParams) -> Self {
        Self {
            data: AdxrData::Candles { candles },
            params,
        }
    }

    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: AdxrParams,
    ) -> Self {
        Self {
            data: AdxrData::Slices { high, low, close },
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: AdxrData::Candles { candles },
            params: AdxrParams::default(),
        }
    }

    fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| AdxrParams::default().period.unwrap())
    }
}

#[derive(Debug, Clone)]
pub struct AdxrOutput {
    pub values: Vec<f64>,
}

#[inline]
pub fn adxr(input: &AdxrInput) -> Result<AdxrOutput, Box<dyn Error>> {
    let period: usize = input.get_period();

    let (high, low, close) = match &input.data {
        AdxrData::Candles { candles } => {
            let high: &[f64] = candles.select_candle_field("high")?;
            let low: &[f64] = candles.select_candle_field("low")?;
            let close: &[f64] = candles.select_candle_field("close")?;
            (high, low, close)
        }
        AdxrData::Slices { high, low, close } => (*high, *low, *close),
    };

    let len: usize = close.len();
    if period == 0 || period > len {
        return Err("Invalid period specified for ADXR calculation.".into());
    }
    if len < period + 1 {
        return Err("Not enough data points to calculate ADXR.".into());
    }

    let mut adx_vals = vec![f64::NAN; len];

    let mut tr_sum = 0.0;
    let mut plus_dm_sum = 0.0;
    let mut minus_dm_sum = 0.0;

    let period_f64 = period as f64;
    let reciprocal_period = 1.0 / period_f64;
    let one_minus_rp = 1.0 - reciprocal_period;

    for i in 1..=period {
        let prev_close = close[i - 1];
        let curr_high = high[i];
        let curr_low = low[i];
        let prev_high = high[i - 1];
        let prev_low = low[i - 1];

        let tr = (curr_high - curr_low)
            .max((curr_high - prev_close).abs())
            .max((curr_low - prev_close).abs());
        tr_sum += tr;

        let up_move = curr_high - prev_high;
        let down_move = prev_low - curr_low;

        if up_move > down_move && up_move > 0.0 {
            plus_dm_sum += up_move;
        }
        if down_move > up_move && down_move > 0.0 {
            minus_dm_sum += down_move;
        }
    }

    let mut atr = tr_sum;
    let mut plus_dm_smooth = plus_dm_sum;
    let mut minus_dm_smooth = minus_dm_sum;

    let plus_di_initial = if atr != 0.0 {
        (plus_dm_smooth / atr) * 100.0
    } else {
        0.0
    };
    let minus_di_initial = if atr != 0.0 {
        (minus_dm_smooth / atr) * 100.0
    } else {
        0.0
    };
    let di_sum = plus_di_initial + minus_di_initial;
    let initial_dx = if di_sum != 0.0 {
        ((plus_di_initial - minus_di_initial).abs() / di_sum) * 100.0
    } else {
        0.0
    };

    let mut dx_sum = initial_dx;
    let mut dx_count = 1;
    let mut last_adx = f64::NAN;
    let mut have_adx = false;

    for i in (period + 1)..len {
        let prev_close = close[i - 1];
        let curr_high = high[i];
        let curr_low = low[i];
        let prev_high = high[i - 1];
        let prev_low = low[i - 1];

        let tr = (curr_high - curr_low)
            .max((curr_high - prev_close).abs())
            .max((curr_low - prev_close).abs());

        let up_move = curr_high - prev_high;
        let down_move = prev_low - curr_low;

        let plus_dm = if up_move > down_move && up_move > 0.0 {
            up_move
        } else {
            0.0
        };
        let minus_dm = if down_move > up_move && down_move > 0.0 {
            down_move
        } else {
            0.0
        };

        atr = atr * one_minus_rp + tr;
        plus_dm_smooth = plus_dm_smooth * one_minus_rp + plus_dm;
        minus_dm_smooth = minus_dm_smooth * one_minus_rp + minus_dm;

        let plus_di_current = if atr != 0.0 {
            (plus_dm_smooth / atr) * 100.0
        } else {
            0.0
        };
        let minus_di_current = if atr != 0.0 {
            (minus_dm_smooth / atr) * 100.0
        } else {
            0.0
        };

        let sum_di_current = plus_di_current + minus_di_current;
        let dx = if sum_di_current != 0.0 {
            ((plus_di_current - minus_di_current).abs() / sum_di_current) * 100.0
        } else {
            0.0
        };

        if dx_count < period {
            dx_sum += dx;
            dx_count += 1;

            if dx_count == period {
                last_adx = dx_sum * reciprocal_period;
                adx_vals[i] = last_adx;
                have_adx = true;
            }
        } else if have_adx {
            let adx_current = ((last_adx * (period_f64 - 1.0)) + dx) * reciprocal_period;
            adx_vals[i] = adx_current;
            last_adx = adx_current;
        }
    }

    let mut adxr_vals = vec![f64::NAN; len];

    for i in (2 * period)..len {
        let adx_i = adx_vals[i];
        let adx_im_p = adx_vals[i - period];
        if adx_i.is_finite() && adx_im_p.is_finite() {
            adxr_vals[i] = (adx_i + adx_im_p) / 2.0;
        }
    }

    Ok(AdxrOutput { values: adxr_vals })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_adxr_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let partial_params = AdxrParams { period: None };
        let input = AdxrInput::from_candles(&candles, partial_params);
        let result = adxr(&input).expect("Failed ADXR with partial params");
        assert_eq!(result.values.len(), candles.close.len());
    }

    #[test]
    fn test_adxr_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let params = AdxrParams { period: Some(14) };
        let input = AdxrInput::from_candles(&candles, params);
        let adxr_result = adxr(&input).expect("Failed to calculate ADXR");

        assert_eq!(
            adxr_result.values.len(),
            candles.close.len(),
            "ADXR output length does not match input length"
        );

        let expected_last_five_adxr = [37.10, 37.3, 37.0, 36.2, 36.3];
        assert!(
            adxr_result.values.len() >= 5,
            "Not enough ADXR values for test"
        );

        let start_index = adxr_result.values.len().saturating_sub(5);
        let result_last_five = &adxr_result.values[start_index..];

        for (i, &actual) in result_last_five.iter().enumerate() {
            let expected = expected_last_five_adxr[i];
            assert!(
                (actual - expected).abs() < 1e-1,
                "ADXR mismatch at final[{}]: expected {}, got {}",
                i,
                expected,
                actual
            );
        }

        let default_input = AdxrInput::with_default_candles(&candles);
        let default_adxr_result =
            adxr(&default_input).expect("Failed to calculate ADXR with defaults");
        assert!(
            !default_adxr_result.values.is_empty(),
            "Should produce ADXR values with default params"
        );
    }
    #[test]
    fn test_adxr_params_with_default_params() {
        let default_params = AdxrParams::default();
        assert_eq!(default_params.period, Some(14));
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = AdxrInput::from_candles(&candles, default_params);
        let result = adxr(&input).expect("Failed ADXR with default params");
        assert_eq!(result.values.len(), candles.close.len());
    }

    #[test]
    fn test_adxr_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = AdxrInput::with_default_candles(&candles);
        match input.data {
            AdxrData::Candles { .. } => {}
            _ => panic!("Expected AdxrData::Candles variant"),
        }
        let result = adxr(&input).expect("Failed ADXR with default_candles");
        assert_eq!(result.values.len(), candles.close.len());
    }

    #[test]
    fn test_adxr_with_zero_period() {
        let high = [10.0, 20.0, 30.0];
        let low = [9.0, 19.0, 29.0];
        let close = [9.5, 19.5, 29.5];
        let params = AdxrParams { period: Some(0) };
        let input = AdxrInput::from_slices(&high, &low, &close, params);
        let result = adxr(&input);
        assert!(result.is_err(), "Expected an error for zero period");
        if let Err(e) = result {
            assert!(e.to_string().contains("Invalid period"));
        }
    }

    #[test]
    fn test_adxr_with_period_exceeding_data_length() {
        let high = [10.0, 20.0];
        let low = [9.0, 19.0];
        let close = [9.5, 19.5];
        let params = AdxrParams { period: Some(10) };
        let input = AdxrInput::from_slices(&high, &low, &close, params);
        let result = adxr(&input);
        assert!(result.is_err(), "Expected an error for period > data.len()");
    }

    #[test]
    fn test_adxr_very_small_data_set() {
        let high = [100.0];
        let low = [99.0];
        let close = [99.5];
        let params = AdxrParams { period: Some(14) };
        let input = AdxrInput::from_slices(&high, &low, &close, params);
        let result = adxr(&input);
        assert!(
            result.is_err(),
            "Expected error for data smaller than period"
        );
    }

    #[test]
    fn test_adxr_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = AdxrParams { period: Some(14) };
        let first_input = AdxrInput::from_candles(&candles, params);
        let first_result = adxr(&first_input).expect("Failed to calculate first ADXR");
        assert_eq!(first_result.values.len(), candles.close.len());
        let high_reinput = &candles.high;
        let low_reinput = &candles.low;
        let close_reinput = &candles.close;
        let second_params = AdxrParams { period: Some(5) };
        let second_input =
            AdxrInput::from_slices(high_reinput, low_reinput, close_reinput, second_params);
        let second_result = adxr(&second_input).expect("Failed second ADXR");
        assert_eq!(second_result.values.len(), candles.close.len());
        if second_result.values.len() > 240 {
            for i in 240..second_result.values.len() {
                assert!(
                    !second_result.values[i].is_nan(),
                    "Found NaN in ADXR at {}",
                    i
                );
            }
        }
    }

    #[test]
    fn test_adxr_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = AdxrParams { period: Some(14) };
        let input = AdxrInput::from_candles(&candles, params);
        let adxr_result = adxr(&input).expect("Failed to calculate ADXR");
        assert_eq!(adxr_result.values.len(), candles.close.len());
        if adxr_result.values.len() > 240 {
            for i in 240..adxr_result.values.len() {
                assert!(
                    !adxr_result.values[i].is_nan(),
                    "Found NaN in ADXR at {}",
                    i
                );
            }
        }
    }
}
