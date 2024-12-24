use crate::indicators::data_loader::Candles;
use std::error::Error;

#[derive(Debug, Clone)]
pub struct ApoParams {
    pub short_period: Option<usize>,
    pub long_period: Option<usize>,
}

impl Default for ApoParams {
    fn default() -> Self {
        ApoParams {
            short_period: Some(10),
            long_period: Some(20),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ApoInput<'a> {
    pub candles: &'a Candles,
    pub params: ApoParams,
}

impl<'a> ApoInput<'a> {
    pub fn new(candles: &'a Candles, params: ApoParams) -> Self {
        ApoInput { candles, params }
    }

    pub fn with_default_params(candles: &'a Candles) -> Self {
        ApoInput {
            candles,
            params: ApoParams::default(),
        }
    }

    fn get_short_period(&self) -> usize {
        self.params.short_period.unwrap_or(10)
    }

    fn get_long_period(&self) -> usize {
        self.params.long_period.unwrap_or(20)
    }
}

#[derive(Debug, Clone)]
pub struct ApoOutput {
    pub values: Vec<f64>,
}

#[inline]
pub fn calculate_apo(input: &ApoInput) -> Result<ApoOutput, Box<dyn Error>> {
    let candles = input.candles;
    let short = input.get_short_period();
    let long = input.get_long_period();

    if short == 0 || long == 0 {
        return Err("Invalid period specified for APO calculation.".into());
    }
    if short >= long {
        return Err("Short period must be less than the long period for APO.".into());
    }

    let close = candles.select_candle_field("close")?;
    let len = close.len();
    if len == 0 {
        return Err("No candles available.".into());
    }

    let mut apo_values = Vec::with_capacity(len);
    apo_values.resize(len, f64::NAN);

    let alpha_short = 2.0 / (short as f64 + 1.0);
    let alpha_long = 2.0 / (long as f64 + 1.0);

    let mut short_ema = close[0];
    let mut long_ema = close[0];

    apo_values[0] = short_ema - long_ema;

    for i in 1..len {
        let price = close[i];
        short_ema = alpha_short * price + (1.0 - alpha_short) * short_ema;
        long_ema = alpha_long * price + (1.0 - alpha_long) * long_ema;

        let apo_val = short_ema - long_ema;
        apo_values[i] = apo_val;
    }

    Ok(ApoOutput { values: apo_values })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::data_loader::read_candles_from_csv;

    #[test]
    fn test_apo_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = ApoInput::with_default_params(&candles);
        let result = calculate_apo(&input).expect("Failed to calculate APO");

        let expected_last_five = [-429.8, -401.6, -386.1, -357.9, -374.1];

        assert!(
            result.values.len() >= 5,
            "Not enough APO values for the test"
        );

        assert_eq!(
            result.values.len(),
            candles.close.len(),
            "APO output length does not match input length!"
        );

        let start_index = result.values.len().saturating_sub(5);
        let result_last_five = &result.values[start_index..];

        for (i, &value) in result_last_five.iter().enumerate() {
            assert!(
                (value - expected_last_five[i]).abs() < 1e-1,
                "APO value mismatch at index {}: expected {}, got {}",
                i,
                expected_last_five[i],
                value
            );
        }

        for val in result.values.iter().skip(20 - 1) {
            assert!(
                val.is_finite(),
                "APO output should be finite after EMAs are established"
            );
        }
    }
}
