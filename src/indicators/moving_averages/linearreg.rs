use crate::utilities::data_loader::{source_type, Candles};
use std::error::Error;

#[derive(Debug, Clone)]
pub enum LinRegData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct LinRegOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct LinRegParams {
    pub period: Option<usize>,
}

impl LinRegParams {
    pub fn with_default_params() -> Self {
        Self { period: None }
    }
}

#[derive(Debug, Clone)]
pub struct LinRegInput<'a> {
    pub data: LinRegData<'a>,
    pub params: LinRegParams,
}

impl<'a> LinRegInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: LinRegParams) -> Self {
        Self {
            data: LinRegData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: LinRegParams) -> Self {
        Self {
            data: LinRegData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: LinRegData::Candles {
                candles,
                source: "close",
            },
            params: LinRegParams::with_default_params(),
        }
    }
}

#[inline]
pub fn linreg(input: &LinRegInput) -> Result<LinRegOutput, Box<dyn Error>> {
    let data: &[f64] = match &input.data {
        LinRegData::Candles { candles, source } => source_type(candles, source),
        LinRegData::Slice(slice) => slice,
    };
    let size: usize = data.len();
    let period: usize = input.params.period.unwrap_or(14);
    if period < 1 {
        return Err("Invalid period (<1) for linear regression.".into());
    }
    if size == 0 {
        return Err("No data available for linear regression.".into());
    }
    if size < period {
        return Ok(LinRegOutput {
            values: vec![f64::NAN; size],
        });
    }

    let mut values = vec![f64::NAN; size];

    let x = (period * (period + 1)) / 2;
    let x2 = (period * (period + 1) * (2 * period + 1)) / 6;
    let x_f = x as f64;
    let x2_f = x2 as f64;
    let period_f = period as f64;

    let bd = 1.0 / (period_f * x2_f - x_f * x_f);

    let mut y = 0.0;
    let mut xy = 0.0;

    for i in 0..(period - 1) {
        let x_i = (i + 1) as f64;
        let val = data[i];
        y += val;
        xy += val * x_i;
    }

    for i in (period - 1)..size {
        let val = data[i];
        xy += val * (period as f64);
        y += val;

        let b = (period_f * xy - x_f * y) * bd;

        let a = (y - b * x_f) / period_f;
        let forecast = a + b * period_f;

        values[i] = forecast;
        xy -= y;
        let oldest_idx = i - (period - 1);
        let oldest_val = data[oldest_idx];
        y -= oldest_val;
    }

    Ok(LinRegOutput { values })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_linreg_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");
        let params = LinRegParams { period: Some(14) };
        let input = LinRegInput::from_candles(&candles, "close", params);
        let linreg_result = linreg(&input).expect("Failed to calculate Linear Regression");
        let expected_last_five = [
            58929.37142857143,
            58899.42857142857,
            58918.857142857145,
            59100.6,
            58987.94285714286,
        ];
        assert!(linreg_result.values.len() >= 5);
        assert_eq!(linreg_result.values.len(), close_prices.len());
        let start_index = linreg_result.values.len() - 5;
        let result_last_five = &linreg_result.values[start_index..];
        for (i, &value) in result_last_five.iter().enumerate() {
            let expected_value = expected_last_five[i];
            assert!(
                (value - expected_value).abs() < 1e-1,
                "Mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }
    }
    #[test]
    fn test_linreg_params_with_default_params() {
        let params = LinRegParams::with_default_params();
        assert_eq!(params.period, None);
    }

    #[test]
    fn test_linreg_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = LinRegInput::with_default_candles(&candles);
        match input.data {
            LinRegData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected LinRegData::Candles variant"),
        }
        assert_eq!(input.params.period, None);
    }

    #[test]
    fn test_linreg_invalid_period() {
        let data = [10.0, 20.0, 30.0];
        let params = LinRegParams { period: Some(0) };
        let input = LinRegInput::from_slice(&data, params);
        let result = linreg(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_linreg_no_data() {
        let data: [f64; 0] = [];
        let params = LinRegParams { period: Some(14) };
        let input = LinRegInput::from_slice(&data, params);
        let result = linreg(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_linreg_small_data() {
        let data = [10.0, 20.0, 30.0];
        let params = LinRegParams { period: Some(14) };
        let input = LinRegInput::from_slice(&data, params);
        let result = linreg(&input).expect("Should handle data smaller than period");
        assert_eq!(result.values.len(), data.len());
        for &val in &result.values {
            assert!(val.is_nan());
        }
    }

    #[test]
    fn test_linreg_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params_first = LinRegParams { period: Some(14) };
        let input_first = LinRegInput::from_candles(&candles, "close", params_first);
        let result_first = linreg(&input_first).expect("Failed first LinReg");
        assert_eq!(result_first.values.len(), candles.close.len());
        let params_second = LinRegParams { period: Some(10) };
        let input_second = LinRegInput::from_slice(&result_first.values, params_second);
        let result_second = linreg(&input_second).expect("Failed second LinReg");
        assert_eq!(result_second.values.len(), result_first.values.len());
    }

    #[test]
    fn test_linreg_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = LinRegParams { period: Some(14) };
        let input = LinRegInput::from_candles(&candles, "close", params);
        let result = linreg(&input).expect("Failed LinReg");
        assert_eq!(result.values.len(), candles.close.len());
        if result.values.len() > 240 {
            for i in 240..result.values.len() {
                assert!(!result.values[i].is_nan());
            }
        }
    }
}
