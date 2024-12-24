use crate::utilities::data_loader::Candles;
use std::error::Error;

#[derive(Debug, Clone)]
pub struct AdxrParams {
    pub period: Option<usize>,
}

impl Default for AdxrParams {
    fn default() -> Self {
        AdxrParams { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct AdxrInput<'a> {
    pub candles: &'a Candles,
    pub params: AdxrParams,
}

impl<'a> AdxrInput<'a> {
    pub fn new(candles: &'a Candles, params: AdxrParams) -> Self {
        AdxrInput { candles, params }
    }

    pub fn with_default_params(candles: &'a Candles) -> Self {
        AdxrInput {
            candles,
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

#[inline(always)]
pub fn calculate_adxr(input: &AdxrInput) -> Result<AdxrOutput, Box<dyn Error>> {
    let candles = input.candles;
    let period = input.get_period();

    let high = candles.select_candle_field("high")?;
    let low = candles.select_candle_field("low")?;
    let close = candles.select_candle_field("close")?;

    let len = close.len();
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
    fn test_adxr_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let params = AdxrParams { period: Some(14) };
        let input = AdxrInput::new(&candles, params);
        let adxr_result = calculate_adxr(&input).expect("Failed to calculate ADXR");

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

        let default_input = AdxrInput::with_default_params(&candles);
        let default_adxr_result =
            calculate_adxr(&default_input).expect("Failed to calculate ADXR with defaults");
        assert!(
            !default_adxr_result.values.is_empty(),
            "Should produce ADXR values with default params"
        );
    }
}
