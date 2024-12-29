use crate::utilities::data_loader::{source_type, Candles};
use std::error::Error;
use std::f64::consts::PI;

#[derive(Debug, Clone)]
pub enum TrendFlexData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct TrendFlexOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct TrendFlexParams {
    pub period: Option<usize>,
}

impl TrendFlexParams {
    pub fn with_default_params() -> Self {
        Self { period: None }
    }
}

#[derive(Debug, Clone)]
pub struct TrendFlexInput<'a> {
    pub data: TrendFlexData<'a>,
    pub params: TrendFlexParams,
}

impl<'a> TrendFlexInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: TrendFlexParams) -> Self {
        Self {
            data: TrendFlexData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: TrendFlexParams) -> Self {
        Self {
            data: TrendFlexData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: TrendFlexData::Candles {
                candles,
                source: "close",
            },
            params: TrendFlexParams::with_default_params(),
        }
    }
}

#[inline]
pub fn trendflex(input: &TrendFlexInput) -> Result<TrendFlexOutput, Box<dyn Error>> {
    let data: &[f64] = match &input.data {
        TrendFlexData::Candles { candles, source } => source_type(candles, source),
        TrendFlexData::Slice(slice) => slice,
    };
    let len: usize = data.len();
    let trendflex_period: usize = input.params.period.unwrap_or(20);

    if data.is_empty() {
        return Err("No data provided to TrendFlex filter.".into());
    }
    if trendflex_period == 0 {
        return Err("TrendFlex period must be >= 1.".into());
    }
    if trendflex_period > len {
        return Err("TrendFlex period cannot exceed data length.".into());
    }

    let ss_period = ((trendflex_period as f64) / 2.0).round() as usize;
    if ss_period > len {
        return Err("Supersmoother period cannot exceed data length.".into());
    }

    let mut ssf = vec![0.0; len];
    ssf[0] = data[0];
    if len > 1 {
        ssf[1] = data[1];
    }

    let a = (-1.414_f64 * PI / (ss_period as f64)).exp();
    let a_sq = a * a;
    let b = 2.0 * a * (1.414_f64 * PI / (ss_period as f64)).cos();
    let c = (1.0 + a_sq - b) * 0.5;

    for i in 2..len {
        let prev_1 = ssf[i - 1];
        let prev_2 = ssf[i - 2];
        let d_i = data[i];
        let d_im1 = data[i - 1];
        ssf[i] = c * (d_i + d_im1) + b * prev_1 - a_sq * prev_2;
    }

    let mut tf_values: Vec<f64> = vec![f64::NAN; len];
    let mut ms_prev = 0.0;

    let tp_f = trendflex_period as f64;
    let inv_tp = 1.0 / tp_f;

    let mut rolling_sum = 0.0;
    for &value in &ssf[..trendflex_period] {
        rolling_sum += value;
    }

    for i in trendflex_period..len {
        let my_sum = (tp_f * ssf[i] - rolling_sum) * inv_tp;

        let ms_current = 0.04 * my_sum * my_sum + 0.96 * ms_prev;
        ms_prev = ms_current;

        tf_values[i] = if ms_current != 0.0 {
            my_sum / ms_current.sqrt()
        } else {
            0.0
        };

        rolling_sum += ssf[i] - ssf[i - trendflex_period];
    }

    Ok(TrendFlexOutput { values: tf_values })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_trendflex_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = TrendFlexParams { period: None };
        let input_default = TrendFlexInput::from_candles(&candles, "close", default_params);
        let output_default =
            trendflex(&input_default).expect("Failed TrendFlex with default params");
        assert_eq!(output_default.values.len(), candles.close.len());

        let custom_params = TrendFlexParams { period: Some(25) };
        let input_custom = TrendFlexInput::from_candles(&candles, "hlc3", custom_params);
        let output_custom =
            trendflex(&input_custom).expect("Failed TrendFlex with period=25, source=hlc3");
        assert_eq!(output_custom.values.len(), candles.close.len());
    }

    #[test]
    fn test_trendflex_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let params = TrendFlexParams { period: Some(20) };
        let input = TrendFlexInput::from_candles(&candles, "close", params);
        let result = trendflex(&input).expect("TrendFlex calculation failed");
        assert_eq!(result.values.len(), close_prices.len());

        let expected_last_five = [
            -0.19724678008015128,
            -0.1238001236481444,
            -0.10515389737087717,
            -0.1149541079904878,
            -0.16006869484450567,
        ];

        assert!(result.values.len() >= 5);
        let start_index = result.values.len() - 5;
        let actual_last_five = &result.values[start_index..];
        for (i, (&actual, &expected)) in
            actual_last_five.iter().zip(&expected_last_five).enumerate()
        {
            let diff = (actual - expected).abs();
            assert!(
                diff < 1e-12,
                "TrendFlex mismatch at index {}: expected {:.14}, got {:.14}",
                i,
                expected,
                actual
            );
        }
    }
    #[test]
    fn test_trendflex_params_with_default_params() {
        let params = TrendFlexParams::with_default_params();
        assert_eq!(params.period, None);
    }

    #[test]
    fn test_trendflex_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = TrendFlexInput::with_default_candles(&candles);
        match input.data {
            TrendFlexData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected TrendFlexData::Candles"),
        }
        assert_eq!(input.params.period, None);
    }

    #[test]
    fn test_trendflex_no_data() {
        let data: [f64; 0] = [];
        let params = TrendFlexParams { period: Some(20) };
        let input = TrendFlexInput::from_slice(&data, params);
        let result = trendflex(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_trendflex_zero_period() {
        let data = [10.0, 20.0, 30.0];
        let params = TrendFlexParams { period: Some(0) };
        let input = TrendFlexInput::from_slice(&data, params);
        let result = trendflex(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_trendflex_period_exceeding_length() {
        let data = [10.0, 20.0, 30.0];
        let params = TrendFlexParams { period: Some(10) };
        let input = TrendFlexInput::from_slice(&data, params);
        let result = trendflex(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_trendflex_small_data() {
        let data = [42.0];
        let params = TrendFlexParams { period: Some(1) };
        let input = TrendFlexInput::from_slice(&data, params);
        let result = trendflex(&input).expect("Should handle single data point with period=1");
        assert_eq!(result.values.len(), data.len());
        assert!(result.values[0].is_nan());
    }

    #[test]
    fn test_trendflex_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params_1 = TrendFlexParams { period: Some(20) };
        let input_1 = TrendFlexInput::from_candles(&candles, "close", params_1);
        let result_1 = trendflex(&input_1).expect("TrendFlex pass 1 failed");
        assert_eq!(result_1.values.len(), candles.close.len());
        let params_2 = TrendFlexParams { period: Some(10) };
        let input_2 = TrendFlexInput::from_slice(&result_1.values, params_2);
        let result_2 = trendflex(&input_2).expect("TrendFlex pass 2 failed");
        assert_eq!(result_2.values.len(), result_1.values.len());
    }

    #[test]
    fn test_trendflex_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = TrendFlexParams { period: Some(20) };
        let input = TrendFlexInput::from_candles(&candles, "close", params);
        let result = trendflex(&input).expect("TrendFlex failed");
        assert_eq!(result.values.len(), candles.close.len());
        if result.values.len() > 240 {
            for i in 240..result.values.len() {
                assert!(!result.values[i].is_nan());
            }
        }
    }
}
