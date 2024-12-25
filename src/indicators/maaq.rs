use std::error::Error;
#[derive(Debug, Clone)]
pub struct MaaqParams {
    pub period: Option<usize>,
    pub fast_period: Option<usize>,
    pub slow_period: Option<usize>,
}

impl Default for MaaqParams {
    fn default() -> Self {
        MaaqParams {
            period: Some(11),
            fast_period: Some(2),
            slow_period: Some(30),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MaaqInput<'a> {
    pub data: &'a [f64],
    pub params: MaaqParams,
}

impl<'a> MaaqInput<'a> {
    pub fn new(data: &'a [f64], params: MaaqParams) -> Self {
        MaaqInput { data, params }
    }

    pub fn with_default_params(data: &'a [f64]) -> Self {
        MaaqInput {
            data,
            params: MaaqParams::default(),
        }
    }

    #[inline]
    fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| MaaqParams::default().period.unwrap())
    }

    #[inline]
    fn get_fast_period(&self) -> usize {
        self.params
            .fast_period
            .unwrap_or_else(|| MaaqParams::default().fast_period.unwrap())
    }

    #[inline]
    fn get_slow_period(&self) -> usize {
        self.params
            .slow_period
            .unwrap_or_else(|| MaaqParams::default().slow_period.unwrap())
    }
}

#[derive(Debug, Clone)]
pub struct MaaqOutput {
    pub values: Vec<f64>,
}

#[inline]
pub fn calculate_maaq(input: &MaaqInput) -> Result<MaaqOutput, Box<dyn Error>> {
    let data = input.data;
    let period = input.get_period();
    let fast_p = input.get_fast_period();
    let slow_p = input.get_slow_period();

    let len = data.len();
    if len < period {
        return Err(format!(
            "Not enough data: length={} < period={}",
            len, period
        )
        .into());
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
    for i in 0..period {
        maaq_values[i] = data[i];
    }

    let mut rolling_sum = 0.0;
    for i in 0..period {
        rolling_sum += diff[i];
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
        let input = MaaqInput::new(close_prices, params);
        let maaq_result = calculate_maaq(&input).expect("Failed to calculate MAAQ");
        assert_eq!(
            maaq_result.values.len(),
            close_prices.len(),
            "MAAQ length should match input length"
        );

        let expected_last_five = [
            59747.657115949725,
            59740.803138018055,
            59724.24153333905,
            59720.60576365108,
            59673.9954445178,
        ];
        let len = maaq_result.values.len();
        assert!(
            len >= 5,
            "Need at least 5 data points to compare last 5 values"
        );
        let actual_last_five = &maaq_result.values[len - 5..];

        for (i, &val) in actual_last_five.iter().enumerate() {
            let exp = expected_last_five[i];
            let diff = (val - exp).abs();
            assert!(
                diff < 1e-2,
                "MAAQ mismatch at last-5 index {}: expected {}, got {}",
                i,
                exp,
                val
            );
        }
    }
}
