use crate::utilities::data_loader::{source_type, Candles};
use std::error::Error;

#[derive(Debug, Clone)]
pub enum MaaqData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct MaaqOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct MaaqParams {
    pub period: Option<usize>,
    pub fast_period: Option<usize>,
    pub slow_period: Option<usize>,
}

impl Default for MaaqParams {
    fn default() -> Self {
        Self {
            period: Some(11),
            fast_period: Some(2),
            slow_period: Some(30),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MaaqInput<'a> {
    pub data: MaaqData<'a>,
    pub params: MaaqParams,
}

impl<'a> MaaqInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: MaaqParams) -> Self {
        Self {
            data: MaaqData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: MaaqParams) -> Self {
        Self {
            data: MaaqData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: MaaqData::Candles {
                candles,
                source: "close",
            },
            params: MaaqParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| MaaqParams::default().period.unwrap())
    }

    pub fn get_fast_period(&self) -> usize {
        self.params
            .fast_period
            .unwrap_or_else(|| MaaqParams::default().fast_period.unwrap())
    }

    pub fn get_slow_period(&self) -> usize {
        self.params
            .slow_period
            .unwrap_or_else(|| MaaqParams::default().slow_period.unwrap())
    }
}

use thiserror::Error;

#[derive(Debug, Error)]
pub enum MaaqError {
    #[error("All values are NaN.")]
    AllValuesNaN,
    #[error("Not enough data: needed = {needed}, got = {got}")]
    NotEnoughData { needed: usize, got: usize },
    #[error("MAAQ periods cannot be zero: period = {period}, fast = {fast_p}, slow = {slow_p}")]
    ZeroPeriods {
        period: usize,
        fast_p: usize,
        slow_p: usize,
    },
}

#[inline]
pub fn maaq(input: &MaaqInput) -> Result<MaaqOutput, MaaqError> {
    let data: &[f64] = match &input.data {
        MaaqData::Candles { candles, source } => source_type(candles, source),
        MaaqData::Slice(slice) => slice,
    };

    if data.iter().all(|&x| x.is_nan()) {
        return Err(MaaqError::AllValuesNaN);
    }

    let period = input.get_period();
    let fast_p = input.get_fast_period();
    let slow_p = input.get_slow_period();
    let len = data.len();

    if len < period {
        return Err(MaaqError::NotEnoughData {
            needed: period,
            got: len,
        });
    }

    if period == 0 || fast_p == 0 || slow_p == 0 {
        return Err(MaaqError::ZeroPeriods {
            period,
            fast_p,
            slow_p,
        });
    }

    let fast_sc = 2.0 / (fast_p as f64 + 1.0);
    let slow_sc = 2.0 / (slow_p as f64 + 1.0);

    let mut diff = vec![0.0; len];
    for i in 1..len {
        diff[i] = (data[i] - data[i - 1]).abs();
    }

    let mut maaq_values = vec![f64::NAN; len];
    maaq_values[..period].copy_from_slice(&data[..period]);

    let mut rolling_sum = 0.0;
    for &value in &diff[..period] {
        rolling_sum += value;
    }

    for i in period..len {
        if i >= period {
            rolling_sum += diff[i];
            rolling_sum -= diff[i - period];
        }

        let noise = rolling_sum;
        let signal = (data[i] - data[i - period]).abs();
        let ratio = if noise.abs() < f64::EPSILON {
            0.0
        } else {
            signal / noise
        };

        let sc = ratio.mul_add(fast_sc, slow_sc);
        let temp = sc * sc;

        let prev_val = maaq_values[i - 1];
        maaq_values[i] = prev_val + temp * (data[i] - prev_val);
    }

    Ok(MaaqOutput {
        values: maaq_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_maaq_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = MaaqParams {
            period: None,
            fast_period: None,
            slow_period: None,
        };
        let input_default = MaaqInput::from_candles(&candles, "close", default_params);
        let output_default = maaq(&input_default).expect("Failed MAAQ with default params");
        assert_eq!(output_default.values.len(), candles.close.len());

        let params_custom = MaaqParams {
            period: Some(12),
            fast_period: Some(3),
            slow_period: Some(25),
        };
        let input_custom = MaaqInput::from_candles(&candles, "hl2", params_custom);
        let output_custom = maaq(&input_custom).expect("Failed MAAQ with custom params");
        assert_eq!(output_custom.values.len(), candles.close.len());
    }

    #[test]
    fn test_maaq_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let params = MaaqParams {
            period: Some(11),
            fast_period: Some(2),
            slow_period: Some(30),
        };
        let input = MaaqInput::from_candles(&candles, "close", params);
        let maaq_result = maaq(&input).expect("Failed to calculate MAAQ");
        assert_eq!(maaq_result.values.len(), close_prices.len());

        let expected_last_five = [
            59747.657115949725,
            59740.803138018055,
            59724.24153333905,
            59720.60576365108,
            59673.9954445178,
        ];
        assert!(maaq_result.values.len() >= 5);
        let start_index = maaq_result.values.len() - 5;
        let actual_last_five = &maaq_result.values[start_index..];
        for (i, &val) in actual_last_five.iter().enumerate() {
            let exp = expected_last_five[i];
            assert!(
                (val - exp).abs() < 1e-2,
                "MAAQ mismatch at index {}: expected {}, got {}",
                i,
                exp,
                val
            );
        }
    }
    #[test]
    fn test_maaq_params_with_default_params() {
        let default_params = MaaqParams::default();
        assert_eq!(default_params.period, Some(11));
        assert_eq!(default_params.fast_period, Some(2));
        assert_eq!(default_params.slow_period, Some(30));
    }

    #[test]
    fn test_maaq_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).unwrap();
        let input = MaaqInput::with_default_candles(&candles);
        match input.data {
            MaaqData::Candles { source, .. } => {
                assert_eq!(source, "close");
            }
            _ => panic!("Unexpected data variant"),
        }
    }

    #[test]
    fn test_maaq_with_zero_periods() {
        let input_data = [10.0, 20.0, 30.0, 40.0];
        let params = MaaqParams {
            period: Some(0),
            fast_period: Some(0),
            slow_period: Some(0),
        };
        let input = MaaqInput::from_slice(&input_data, params);
        let result = maaq(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("MAAQ periods cannot be zero"));
        }
    }

    #[test]
    fn test_maaq_insufficient_data() {
        let input_data = [42.0, 43.0, 44.0];
        let params = MaaqParams {
            period: Some(5),
            fast_period: Some(2),
            slow_period: Some(3),
        };
        let input = MaaqInput::from_slice(&input_data, params);
        let result = maaq(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Not enough data"));
        }
    }

    #[test]
    fn test_maaq_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).unwrap();
        let first_params = MaaqParams {
            period: Some(11),
            fast_period: Some(2),
            slow_period: Some(30),
        };
        let first_input = MaaqInput::from_candles(&candles, "close", first_params);
        let first_result = maaq(&first_input).unwrap();
        assert_eq!(first_result.values.len(), candles.close.len());
        let second_params = MaaqParams {
            period: Some(5),
            fast_period: Some(2),
            slow_period: Some(10),
        };
        let second_input = MaaqInput::from_slice(&first_result.values, second_params);
        let second_result = maaq(&second_input).unwrap();
        assert_eq!(second_result.values.len(), first_result.values.len());
        for val in &second_result.values[240..] {
            assert!(!val.is_nan());
        }
    }

    #[test]
    fn test_maaq_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).unwrap();
        let params = MaaqParams {
            period: Some(11),
            fast_period: Some(2),
            slow_period: Some(30),
        };
        let input = MaaqInput::from_candles(&candles, "close", params);
        let maaq_result = maaq(&input).unwrap();
        for &val in &maaq_result.values {
            if !val.is_nan() {
                assert!(val.is_finite());
            }
        }
    }
}
