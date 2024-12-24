use std::error::Error;

#[derive(Debug, Clone)]
pub struct AlmaParams {
    pub windowsize: Option<usize>,
    pub offset: Option<f64>,
    pub sigma: Option<f64>,
}

impl Default for AlmaParams {
    fn default() -> Self {
        AlmaParams {
            windowsize: Some(9),
            offset: Some(0.85),
            sigma: Some(6.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AlmaInput<'a> {
    pub data: &'a [f64],
    pub params: AlmaParams,
}

impl<'a> AlmaInput<'a> {
    pub fn new(data: &'a [f64], params: AlmaParams) -> Self {
        AlmaInput { data, params }
    }

    pub fn with_default_params(data: &'a [f64]) -> Self {
        AlmaInput {
            data,
            params: AlmaParams::default(),
        }
    }

    fn get_windowsize(&self) -> usize {
        self.params
            .windowsize
            .unwrap_or_else(|| AlmaParams::default().windowsize.unwrap())
    }

    fn get_offset(&self) -> f64 {
        self.params
            .offset
            .unwrap_or_else(|| AlmaParams::default().offset.unwrap())
    }

    fn get_sigma(&self) -> f64 {
        self.params
            .sigma
            .unwrap_or_else(|| AlmaParams::default().sigma.unwrap())
    }
}

#[derive(Debug, Clone)]
pub struct AlmaOutput {
    pub values: Vec<f64>,
}

pub fn calculate_alma(input: &AlmaInput) -> Result<AlmaOutput, Box<dyn Error>> {
    let data = input.data;
    let len = data.len();
    let windowsize = input.get_windowsize();
    let offset = input.get_offset();
    let sigma = input.get_sigma();

    if windowsize == 0 || windowsize > len {
        return Err("Invalid windowsize specified for ALMA calculation.".into());
    }

    let m = offset * (windowsize - 1) as f64;
    let s = windowsize as f64 / sigma;
    let s_sq = s * s;
    let den = 2.0 * s_sq;
    let mut weights = Vec::with_capacity(windowsize);
    let mut norm = 0.0;

    for i in 0..windowsize {
        let dif = i as f64 - m;
        let num = dif * dif;
        let weight = (-num / den).exp();
        weights.push(weight);
        norm += weight;
    }
    let inv_norm = 1.0 / norm;

    let mut alma_values = vec![f64::NAN; len];

    for i in (windowsize - 1)..len {
        let start = i + 1 - windowsize;
        let mut sum = 0.0;
        for (w_i, &w) in weights.iter().enumerate() {
            sum += data[start + w_i] * w;
        }
        alma_values[i] = sum * inv_norm;
    }

    Ok(AlmaOutput {
        values: alma_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::data_loader::read_candles_from_csv;

    #[test]
    fn test_alma_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let input = AlmaInput::with_default_params(close_prices);
        let result = calculate_alma(&input).expect("Failed to calculate ALMA");

        let expected_last_five = [59286.7222, 59273.5343, 59204.3729, 59155.9338, 59026.9253];

        assert_eq!(
            result.values.len(),
            close_prices.len(),
            "ALMA output length does not match input length!"
        );

        assert!(
            result.values.len() >= 5,
            "Not enough ALMA values for the test"
        );

        let start_index = result.values.len().saturating_sub(5);
        let result_last_five = &result.values[start_index..];

        for (i, &value) in result_last_five.iter().enumerate() {
            assert!(
                (value - expected_last_five[i]).abs() < 1e-1,
                "ALMA value mismatch at index {}: expected {}, got {}",
                i,
                expected_last_five[i],
                value
            );
        }

        for val in result.values.iter() {
            if !val.is_nan() {
                assert!(val.is_finite(), "ALMA output should be finite");
            }
        }
    }
}
