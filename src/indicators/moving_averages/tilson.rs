use std::error::Error;

#[derive(Debug, Clone)]
pub struct T3Params {
    pub period: Option<usize>,
    pub volume_factor: Option<f64>,
}

impl Default for T3Params {
    fn default() -> Self {
        T3Params {
            period: Some(5),
            volume_factor: Some(0.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct T3Input<'a> {
    pub data: &'a [f64],
    pub params: T3Params,
}

impl<'a> T3Input<'a> {
    pub fn new(data: &'a [f64], params: T3Params) -> Self {
        T3Input { data, params }
    }

    pub fn with_default_params(data: &'a [f64]) -> Self {
        T3Input {
            data,
            params: T3Params::default(),
        }
    }

    fn get_period(&self) -> usize {
        self.params.period.unwrap_or(14)
    }

    fn get_volume_factor(&self) -> f64 {
        self.params.volume_factor.unwrap_or(0.7)
    }
}

#[derive(Debug, Clone)]
pub struct T3Output {
    pub values: Vec<f64>,
}

#[inline]
pub fn calculate_t3(input: &T3Input) -> Result<T3Output, Box<dyn Error>> {
    let data = input.data;
    let opt_in_time_period = input.get_period();
    let opt_in_v_factor = input.get_volume_factor();
    let length = data.len();
    if opt_in_time_period == 0 || opt_in_time_period > length {
        return Err("Invalid period specified.".into());
    }
    let lookback_total = 6 * (opt_in_time_period - 1);
    let mut out_values = vec![std::f64::NAN; length];
    if lookback_total >= length {
        return Ok(T3Output { values: out_values });
    }
    let start_idx = lookback_total;
    let end_idx = length - 1;
    let k = 2.0 / (opt_in_time_period as f64 + 1.0);
    let one_minus_k = 1.0 - k;
    let mut today = 0;

    let mut temp_real;
    let mut e1;
    let mut e2;
    let mut e3;
    let mut e4;
    let mut e5;
    let mut e6;

    temp_real = 0.0;
    for i in 0..opt_in_time_period {
        temp_real += data[today + i];
    }
    e1 = temp_real / opt_in_time_period as f64;
    today += opt_in_time_period;

    temp_real = e1;
    for _ in 1..opt_in_time_period {
        e1 = (k * data[today]) + (one_minus_k * e1);
        temp_real += e1;
        today += 1;
    }
    e2 = temp_real / opt_in_time_period as f64;

    temp_real = e2;
    for _ in 1..opt_in_time_period {
        e1 = (k * data[today]) + (one_minus_k * e1);
        e2 = (k * e1) + (one_minus_k * e2);
        temp_real += e2;
        today += 1;
    }
    e3 = temp_real / opt_in_time_period as f64;

    temp_real = e3;
    for _ in 1..opt_in_time_period {
        e1 = (k * data[today]) + (one_minus_k * e1);
        e2 = (k * e1) + (one_minus_k * e2);
        e3 = (k * e2) + (one_minus_k * e3);
        temp_real += e3;
        today += 1;
    }
    e4 = temp_real / opt_in_time_period as f64;

    temp_real = e4;
    for _ in 1..opt_in_time_period {
        e1 = (k * data[today]) + (one_minus_k * e1);
        e2 = (k * e1) + (one_minus_k * e2);
        e3 = (k * e2) + (one_minus_k * e3);
        e4 = (k * e3) + (one_minus_k * e4);
        temp_real += e4;
        today += 1;
    }
    e5 = temp_real / opt_in_time_period as f64;

    temp_real = e5;
    for _ in 1..opt_in_time_period {
        e1 = (k * data[today]) + (one_minus_k * e1);
        e2 = (k * e1) + (one_minus_k * e2);
        e3 = (k * e2) + (one_minus_k * e3);
        e4 = (k * e3) + (one_minus_k * e4);
        e5 = (k * e4) + (one_minus_k * e5);
        temp_real += e5;
        today += 1;
    }
    e6 = temp_real / opt_in_time_period as f64;

    while today <= start_idx {
        e1 = (k * data[today]) + (one_minus_k * e1);
        e2 = (k * e1) + (one_minus_k * e2);
        e3 = (k * e2) + (one_minus_k * e3);
        e4 = (k * e3) + (one_minus_k * e4);
        e5 = (k * e4) + (one_minus_k * e5);
        e6 = (k * e5) + (one_minus_k * e6);
        today += 1;
    }

    let temp = opt_in_v_factor * opt_in_v_factor;
    let c1 = -(temp * opt_in_v_factor);
    let c2 = 3.0 * (temp - c1);
    let c3 = -6.0 * temp - 3.0 * (opt_in_v_factor - c1);
    let c4 = 1.0 + 3.0 * opt_in_v_factor - c1 + 3.0 * temp;

    if start_idx < length {
        out_values[start_idx] = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3;
    }

    let mut out_idx = start_idx + 1;
    while today <= end_idx {
        e1 = (k * data[today]) + (one_minus_k * e1);
        e2 = (k * e1) + (one_minus_k * e2);
        e3 = (k * e2) + (one_minus_k * e3);
        e4 = (k * e3) + (one_minus_k * e4);
        e5 = (k * e4) + (one_minus_k * e5);
        e6 = (k * e5) + (one_minus_k * e6);
        out_values[out_idx] = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3;
        out_idx += 1;
        today += 1;
    }

    Ok(T3Output { values: out_values })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_t3_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let params = T3Params {
            period: Some(5),
            volume_factor: Some(0.0),
        };

        let input = T3Input::new(close_prices, params);
        let t3_result = calculate_t3(&input).expect("Failed to calculate T3");

        let expected_last_five_t3 = [
            59304.716332473254,
            59283.56868015526,
            59261.16173577631,
            59240.25895948583,
            59203.544843167765,
        ];

        assert!(t3_result.values.len() >= 5);
        assert_eq!(
            t3_result.values.len(),
            close_prices.len(),
            "T3 output length does not match input length"
        );
        let start_index = t3_result.values.len() - 5;
        let result_last_five_t3 = &t3_result.values[start_index..];

        for (i, &value) in result_last_five_t3.iter().enumerate() {
            let expected_value = expected_last_five_t3[i];
            assert!(
                (value - expected_value).abs() < 1e-10,
                "T3 value mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }

        let default_input = T3Input::with_default_params(close_prices);
        let default_t3_result =
            calculate_t3(&default_input).expect("Failed to calculate T3 with defaults");
        assert!(!default_t3_result.values.is_empty());
    }
}
