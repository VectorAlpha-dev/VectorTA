use std::error::Error;

#[derive(Debug, Clone)]
pub struct KamaParams {
    pub period: Option<usize>,
}

impl Default for KamaParams {
    fn default() -> Self {
        KamaParams { period: Some(30) }
    }
}

#[derive(Debug, Clone)]
pub struct KamaInput<'a> {
    pub data: &'a [f64],
    pub params: KamaParams,
}

impl<'a> KamaInput<'a> {
    pub fn new(data: &'a [f64], params: KamaParams) -> Self {
        KamaInput { data, params }
    }

    pub fn with_default_params(data: &'a [f64]) -> Self {
        KamaInput {
            data,
            params: KamaParams::default(),
        }
    }

    fn get_period(&self) -> usize {
        self.params.period.unwrap_or(30)
    }
}

#[derive(Debug, Clone)]
pub struct KamaOutput {
    pub values: Vec<f64>,
}

#[inline]
pub fn calculate_kama(input: &KamaInput) -> Result<KamaOutput, Box<dyn Error>> {
    let data = input.data;
    let len = data.len();
    let period = input.get_period();
    let mut values = vec![f64::NAN; len];
    if period > len {
        return Ok(KamaOutput { values });
    }
    let lookback = period - 1;
    if lookback >= len {
        return Ok(KamaOutput { values });
    }
    let const_max = 2.0 / (30.0 + 1.0);
    let const_diff = (2.0 / (2.0 + 1.0)) - const_max;
    let start_idx = lookback;
    let mut sum_roc1 = 0.0;
    let mut today = start_idx - lookback;
    let mut i = period;
    while i > 0 {
        i -= 1;
        let temp = data[today + 1] - data[today];
        sum_roc1 += temp.abs();
        today += 1;
    }
    let mut prev_kama = data[today];
    values[today] = prev_kama;
    let mut out_idx = 1;
    let mut trailing_idx = start_idx - lookback;
    let mut trailing_value = data[trailing_idx];
    today += 1;
    while today <= start_idx {
        let price = data[today];
        let temp_real = (price - data[trailing_idx]).abs();
        sum_roc1 -= (data[trailing_idx + 1] - trailing_value).abs();
        sum_roc1 += (price - data[today - 1]).abs();
        trailing_value = data[trailing_idx + 1];
        trailing_idx += 1;
        let direction = temp_real;
        let er = if sum_roc1 == 0.0 {
            0.0
        } else {
            direction / sum_roc1
        };
        let sc = (er * const_diff + const_max) * (er * const_diff + const_max);
        prev_kama += (price - prev_kama) * sc;
        today += 1;
    }
    if today <= len {
        values[0] = f64::NAN;
        for i in 1..out_idx {
            values[i] = f64::NAN;
        }
    }
    values[0] = f64::NAN;
    let output_beg = today - 1;
    values[output_beg] = prev_kama;
    out_idx = 1;
    while today < len {
        let price = data[today];
        sum_roc1 -= (data[trailing_idx + 1] - trailing_value).abs();
        sum_roc1 += (price - data[today - 1]).abs();
        trailing_value = data[trailing_idx + 1];
        trailing_idx += 1;
        let direction = (price - data[trailing_idx]).abs();
        let er = if sum_roc1 == 0.0 {
            0.0
        } else {
            direction / sum_roc1
        };
        let sc = (er * const_diff + const_max) * (er * const_diff + const_max);
        prev_kama += (price - prev_kama) * sc;
        values[output_beg + out_idx] = prev_kama;
        out_idx += 1;
        today += 1;
    }
    Ok(KamaOutput { values })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::data_loader::read_candles_from_csv;

    #[test]
    fn test_kama_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let data = &candles.close;
        let input = KamaInput::with_default_params(data);
        let result = calculate_kama(&input).expect("Failed to calculate KAMA");
        let expected_last_five = [
            60234.925553804125,
            60176.838757545665,
            60115.177367962766,
            60071.37070833558,
            59992.79386218023,
        ];
        assert!(result.values.len() >= 5);
        assert_eq!(
            result.values.len(),
            candles.close.len(),
            "KAMA output length does not match input length"
        );
        let start_index = result.values.len().saturating_sub(5);
        let last_five = &result.values[start_index..];
        for (i, &val) in last_five.iter().enumerate() {
            let exp = expected_last_five[i];
            assert!(
                (val - exp).abs() < 1e-6,
                "KAMA mismatch at {}: expected {}, got {}",
                i,
                exp,
                val
            );
        }
    }
}
