use std::error::Error;
use std::f64::consts::PI;

#[derive(Debug, Clone)]
pub struct TrendFlexParams {
    pub period: Option<usize>,
}

impl Default for TrendFlexParams {
    fn default() -> Self {
        TrendFlexParams { period: Some(20) }
    }
}

#[derive(Debug, Clone)]
pub struct TrendFlexInput<'a> {
    pub data: &'a [f64],
    pub params: TrendFlexParams,
}

impl<'a> TrendFlexInput<'a> {
    pub fn new(data: &'a [f64], params: TrendFlexParams) -> Self {
        TrendFlexInput { data, params }
    }

    pub fn with_default_params(data: &'a [f64]) -> Self {
        TrendFlexInput {
            data,
            params: TrendFlexParams::default(),
        }
    }

    #[inline]
    fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| TrendFlexParams::default().period.unwrap())
    }
}

#[derive(Debug, Clone)]
pub struct TrendFlexOutput {
    pub values: Vec<f64>,
}

#[inline]
pub fn calculate_trendflex(input: &TrendFlexInput) -> Result<Vec<f64>, Box<dyn Error>> {
    let data = input.data;
    let trendflex_period = input.get_period();
    let len = data.len();

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

    let mut tf_values = vec![f64::NAN; len];
    let mut ms_prev = 0.0;

    let tp_f = trendflex_period as f64;
    let inv_tp = 1.0 / tp_f;

    let mut rolling_sum = 0.0;
    for i in 0..trendflex_period {
        rolling_sum += ssf[i];
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

    Ok(tf_values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_trendflex_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let params = TrendFlexParams { period: Some(20) };
        let input = TrendFlexInput::new(close_prices, params);

        let tf_values = calculate_trendflex(&input).expect("TrendFlex calculation failed");
        assert_eq!(tf_values.len(), close_prices.len(), "Length mismatch");

        let expected_last_five = [
            -0.19724678008015128,
            -0.1238001236481444,
            -0.10515389737087717,
            -0.1149541079904878,
            -0.16006869484450567,
        ];

        assert!(
            tf_values.len() >= expected_last_five.len(),
            "Not enough TrendFlex values for the test"
        );

        let start_index = tf_values.len() - expected_last_five.len();
        let actual_last_five = &tf_values[start_index..];

        for (i, (&actual, &expected)) in actual_last_five
            .iter()
            .zip(expected_last_five.iter())
            .enumerate()
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
}
