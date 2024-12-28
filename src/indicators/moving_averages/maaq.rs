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

impl MaaqParams {
    pub fn with_default_params() -> Self {
        Self {
            period: None,
            fast_period: None,
            slow_period: None,
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
            params: MaaqParams::with_default_params(),
        }
    }
}

#[inline]
pub fn maaq(input: &MaaqInput) -> Result<MaaqOutput, Box<dyn Error>> {
    let data: &[f64] = match &input.data {
        MaaqData::Candles { candles, source } => source_type(candles, source),
        MaaqData::Slice(slice) => slice,
    };
    let period: usize = input.params.period.unwrap_or(11);
    let fast_p: usize = input.params.fast_period.unwrap_or(2);
    let slow_p: usize = input.params.slow_period.unwrap_or(30);
    let len: usize = data.len();
    if len < period {
        return Err(format!("Not enough data: length={} < period={}", len, period).into());
    }
    if period == 0 || fast_p == 0 || slow_p == 0 {
        return Err("MAAQ periods cannot be zero.".into());
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
}
