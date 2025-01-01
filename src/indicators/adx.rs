use crate::utilities::data_loader::Candles;
use std::error::Error;

#[derive(Debug, Clone)]
pub enum AdxData<'a> {
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
pub struct AdxParams {
    pub period: Option<usize>,
}

impl Default for AdxParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct AdxInput<'a> {
    pub data: AdxData<'a>,
    pub params: AdxParams,
}

impl<'a> AdxInput<'a> {
    pub fn from_candles(candles: &'a Candles, params: AdxParams) -> Self {
        Self {
            data: AdxData::Candles { candles },
            params,
        }
    }

    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: AdxParams,
    ) -> Self {
        Self {
            data: AdxData::Slices { high, low, close },
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: AdxData::Candles { candles },
            params: AdxParams::default(),
        }
    }

    fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| AdxParams::default().period.unwrap())
    }
}

#[derive(Debug, Clone)]
pub struct AdxOutput {
    pub values: Vec<f64>,
}
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AdxError {
    #[error(transparent)]
    CandleFieldError(#[from] Box<dyn std::error::Error>),

    #[error("Invalid period specified for ADX calculation. period={period}, data_len={data_len}")]
    InvalidPeriod { period: usize, data_len: usize },

    #[error("Not enough data points to calculate ADX. Needed at least {needed}, found {found}")]
    NotEnoughData { needed: usize, found: usize },
}

#[inline]
pub fn adx(input: &AdxInput) -> Result<AdxOutput, AdxError> {
    let period = input.get_period();

    let (high, low, close) = match &input.data {
        AdxData::Candles { candles } => {
            let high = candles.select_candle_field("high")?;
            let low = candles.select_candle_field("low")?;
            let close = candles.select_candle_field("close")?;
            (high, low, close)
        }
        AdxData::Slices { high, low, close } => (*high, *low, *close),
    };

    let len = close.len();
    if period == 0 || period > len {
        return Err(AdxError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if len < period + 1 {
        return Err(AdxError::NotEnoughData {
            needed: period + 1,
            found: len,
        });
    }
    let mut adx_vals = vec![f64::NAN; len];

    let mut tr_sum = 0.0;
    let mut plus_dm_sum = 0.0;
    let mut minus_dm_sum = 0.0;

    let period_f64 = period as f64;
    let reciprocal_period = 1.0 / period_f64;
    let one_minus_rp = 1.0 - reciprocal_period;
    let period_minus_one = period_f64 - 1.0;

    for i in 1..=period {
        let curr_high = high[i];
        let curr_low = low[i];
        let prev_close = close[i - 1];
        let prev_high = high[i - 1];
        let prev_low = low[i - 1];

        let tr = (curr_high - curr_low)
            .max((curr_high - prev_close).abs())
            .max((curr_low - prev_close).abs());

        let up_move = curr_high - prev_high;
        let down_move = prev_low - curr_low;

        if up_move > down_move && up_move > 0.0 {
            plus_dm_sum += up_move;
        }
        if down_move > up_move && down_move > 0.0 {
            minus_dm_sum += down_move;
        }

        tr_sum += tr;
    }

    let mut atr = tr_sum;
    let mut plus_dm_smooth = plus_dm_sum;
    let mut minus_dm_smooth = minus_dm_sum;

    let plus_di_prev = (plus_dm_smooth / atr) * 100.0;
    let minus_di_prev = (minus_dm_smooth / atr) * 100.0;

    let sum_di = plus_di_prev + minus_di_prev;
    let initial_dx = if sum_di != 0.0 {
        ((plus_di_prev - minus_di_prev).abs() / sum_di) * 100.0
    } else {
        0.0
    };

    let mut dx_sum = initial_dx;
    let mut dx_count = 1;
    let mut last_adx = 0.0;
    let mut have_adx = false;

    for i in (period + 1)..len {
        let curr_high = high[i];
        let curr_low = low[i];
        let prev_close = close[i - 1];
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

        let plus_di_current = (plus_dm_smooth / atr) * 100.0;
        let minus_di_current = (minus_dm_smooth / atr) * 100.0;

        let sum_di_current = plus_di_current + minus_di_current;
        let dx = if sum_di_current != 0.0 {
            let diff = (plus_di_current - minus_di_current).abs();
            (diff / sum_di_current) * 100.0
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
            let adx_current = ((last_adx * period_minus_one) + dx) * reciprocal_period;
            adx_vals[i] = adx_current;
            last_adx = adx_current;
        }
    }

    Ok(AdxOutput { values: adx_vals })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_adx_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let params = AdxParams { period: Some(14) };
        let input = AdxInput::from_candles(&candles, params);
        let adx_result = adx(&input).expect("Failed to calculate ADX");

        assert_eq!(
            adx_result.values.len(),
            candles.close.len(),
            "ADX output length does not match input length"
        );

        let expected_last_five_adx = [36.14, 36.52, 37.01, 37.46, 38.47];

        assert!(
            adx_result.values.len() >= 5,
            "Not enough ADX values for the test"
        );

        let start_index = adx_result.values.len().saturating_sub(5);
        let result_last_five_ad = &adx_result.values[start_index..];

        for (i, &value) in result_last_five_ad.iter().enumerate() {
            let expected_value = expected_last_five_adx[i];
            assert!(
                (value - expected_value).abs() < 1e-1,
                "ADX value mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }

        let default_input = AdxInput::with_default_candles(&candles);
        let default_adx_result =
            adx(&default_input).expect("Failed to calculate ADX with defaults");
        assert!(
            !default_adx_result.values.is_empty(),
            "Should produce ADX values with default params"
        );
    }

    #[test]
    fn test_adx_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let default_params = AdxParams { period: None };
        let input_default = AdxInput::from_candles(&candles, default_params);
        let output_default = adx(&input_default).expect("Failed ADX with default params");
        assert_eq!(output_default.values.len(), candles.close.len());
        let params_period_14 = AdxParams { period: Some(14) };
        let input_period_14 = AdxInput::from_candles(&candles, params_period_14);
        let output_period_14 = adx(&input_period_14).expect("Failed ADX with period=14");
        assert_eq!(output_period_14.values.len(), candles.close.len());
        let params_custom = AdxParams { period: Some(20) };
        let input_custom = AdxInput::from_candles(&candles, params_custom);
        let output_custom = adx(&input_custom).expect("Failed ADX with custom period");
        assert_eq!(output_custom.values.len(), candles.close.len());
    }

    #[test]
    fn test_adx_params_with_default_params() {
        let default_params = AdxParams::default();
        assert_eq!(default_params.period, Some(14));
    }

    #[test]
    fn test_adx_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = AdxInput::with_default_candles(&candles);
        match input.data {
            AdxData::Candles { .. } => {}
            _ => panic!("Expected AdxData::Candles variant"),
        }
        assert_eq!(input.params.period, Some(14));
    }

    #[test]
    fn test_adx_with_zero_period() {
        let high = [10.0, 20.0, 30.0];
        let low = [5.0, 15.0, 25.0];
        let close = [9.0, 19.0, 29.0];
        let params = AdxParams { period: Some(0) };
        let input = AdxInput::from_slices(&high, &low, &close, params);
        let result = adx(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_adx_with_period_exceeding_data_length() {
        let high = [10.0, 20.0, 30.0];
        let low = [5.0, 15.0, 25.0];
        let close = [9.0, 19.0, 29.0];
        let params = AdxParams { period: Some(10) };
        let input = AdxInput::from_slices(&high, &low, &close, params);
        let result = adx(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_adx_very_small_data_set() {
        let high = [42.0];
        let low = [41.0];
        let close = [40.5];
        let params = AdxParams { period: Some(14) };
        let input = AdxInput::from_slices(&high, &low, &close, params);
        let result = adx(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_adx_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let first_params = AdxParams { period: Some(14) };
        let first_input = AdxInput::from_candles(&candles, first_params);
        let first_result = adx(&first_input).expect("Failed to calculate first ADX");
        assert_eq!(first_result.values.len(), candles.close.len());
        let second_params = AdxParams { period: Some(5) };
        let second_input = AdxInput::from_slices(
            &candles.high,
            &candles.low,
            &first_result.values,
            second_params,
        );
        let second_result = adx(&second_input).expect("Failed to calculate second ADX");
        assert_eq!(second_result.values.len(), candles.close.len());
        for i in 240..second_result.values.len() {
            assert!(!second_result.values[i].is_nan());
        }
    }

    #[test]
    fn test_adx_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = AdxParams { period: Some(14) };
        let input = AdxInput::from_candles(&candles, params);
        let adx_result = adx(&input).expect("Failed to calculate ADX");
        assert_eq!(adx_result.values.len(), candles.close.len());
        if adx_result.values.len() > 100 {
            for i in 100..adx_result.values.len() {
                assert!(!adx_result.values[i].is_nan());
            }
        }
    }
}
