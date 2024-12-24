use std::error::Error;

#[derive(Debug, Clone)]
pub struct AoParams {
    pub short_period: Option<usize>,
    pub long_period: Option<usize>,
}

impl Default for AoParams {
    fn default() -> Self {
        Self {
            short_period: Some(5),
            long_period: Some(34),
        }
    }
}

#[derive(Debug, Clone)]
// Should use HL2 price input.
pub struct AoInput<'a> {
    pub data: &'a [f64],
    pub params: AoParams,
}

impl<'a> AoInput<'a> {
    pub fn new(data: &'a [f64], params: AoParams) -> Self {
        AoInput { data, params }
    }

    pub fn with_default_params(data: &'a [f64]) -> Self {
        AoInput {
            data,
            params: AoParams::default(),
        }
    }

    #[inline]
    fn get_short_period(&self) -> usize {
        self.params.short_period.unwrap_or(5)
    }

    #[inline]
    fn get_long_period(&self) -> usize {
        self.params.long_period.unwrap_or(34)
    }
}

#[derive(Debug, Clone)]
pub struct AoOutput {
    pub values: Vec<f64>,
}

pub fn calculate_ao(input: &AoInput) -> Result<AoOutput, Box<dyn Error>> {
    let data = input.data;
    let short = input.get_short_period();
    let long = input.get_long_period();

    if short == 0 || long == 0 || short >= long {
        return Err("Invalid periods for AO: short=0 or long=0 or short>=long".into());
    }

    let len = data.len();
    if len == 0 {
        return Err("No HL2 data provided.".into());
    }

    if long > len {
        return Ok(AoOutput {
            values: vec![f64::NAN; len],
        });
    }

    let mut ao_values = vec![f64::NAN; len];

    let mut short_sum = 0.0;
    let mut long_sum = 0.0;

    for i in 0..len {
        let val = data[i];
        short_sum += val;
        long_sum += val;

        if i >= short {
            short_sum -= data[i - short];
        }

        if i >= long {
            long_sum -= data[i - long];
        }

        if i >= (long - 1) {
            let short_sma = short_sum / (short as f64);
            let long_sma = long_sum / (long as f64);
            ao_values[i] = short_sma - long_sma;
        }
    }

    Ok(AoOutput { values: ao_values })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_ao_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let hl2_values: Vec<f64> = candles
            .high
            .iter()
            .zip(&candles.low)
            .map(|(&h, &l)| 0.5 * (h + l))
            .collect();

        let input = AoInput::with_default_params(&hl2_values);
        let result = calculate_ao(&input).expect("Failed to calculate AO");
        let expected_last_five = [-1671.3, -1401.6706, -1262.3559, -1178.4941, -1157.4118];

        assert!(
            result.values.len() >= 5,
            "Not enough AO values for the test"
        );

        assert_eq!(
            result.values.len(),
            hl2_values.len(),
            "AO output length does not match input length"
        );

        let start_index = result.values.len().saturating_sub(5);
        let result_last_five = &result.values[start_index..];

        for (i, &value) in result_last_five.iter().enumerate() {
            assert!(
                (value - expected_last_five[i]).abs() < 1e-1,
                "AO value mismatch at index {}: expected {}, got {}",
                i,
                expected_last_five[i],
                value
            );
        }

        for val in result.values.iter().skip(input.get_long_period() - 1) {
            assert!(
                val.is_finite(),
                "AO output should be finite at valid indices"
            );
        }
    }
}
