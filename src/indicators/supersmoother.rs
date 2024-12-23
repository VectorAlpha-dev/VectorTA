use std::error::Error;
use std::f64::consts::PI;

#[derive(Debug, Clone)]
pub struct SuperSmootherParams {
    pub period: Option<usize>,
}

impl Default for SuperSmootherParams {
    fn default() -> Self {
        SuperSmootherParams { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct SuperSmootherInput<'a> {
    pub data: &'a [f64],
    pub params: SuperSmootherParams,
}

impl<'a> SuperSmootherInput<'a> {
    pub fn new(data: &'a [f64], params: SuperSmootherParams) -> Self {
        SuperSmootherInput { data, params }
    }

    pub fn with_default_params(data: &'a [f64]) -> Self {
        SuperSmootherInput {
            data,
            params: SuperSmootherParams::default(),
        }
    }

    fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| SuperSmootherParams::default().period.unwrap())
    }
}

#[derive(Debug, Clone)]
pub struct SuperSmootherOutput {
    pub values: Vec<f64>,
}

#[inline]
pub fn calculate_supersmoother(
    input: &SuperSmootherInput,
) -> Result<SuperSmootherOutput, Box<dyn Error>> {
    let data = input.data;
    let period = input.get_period();

    if data.is_empty() {
        return Ok(SuperSmootherOutput { values: vec![] });
    }
    if period < 1 {
        return Err("Period must be >= 1 for Super Smoother filter.".into());
    }

    let len = data.len();
    let mut output_values = vec![0.0; len];

    let a = (-1.414_f64 * PI / (period as f64)).exp();
    let a_sq = a * a;
    let b = 2.0 * a * (1.414_f64 * PI / (period as f64)).cos();
    let c = (1.0 + a_sq - b) * 0.5;

    output_values[0] = data[0];
    if len > 1 {
        output_values[1] = data[1];
    }

    for i in 2..len {
        let prev_1 = output_values[i - 1];
        let prev_2 = output_values[i - 2];
        let d_i = data[i];
        let d_im1 = data[i - 1];
        output_values[i] = c * (d_i + d_im1) + b * prev_1 - a_sq * prev_2;
    }

    Ok(SuperSmootherOutput {
        values: output_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::data_loader::read_candles_from_csv;

    #[test]
    fn test_supersmoother_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let params = SuperSmootherParams { period: Some(14) };
        let input = SuperSmootherInput::new(close_prices, params);

        let result = calculate_supersmoother(&input).expect("Failed to calculate SuperSmoother");
        let out_vals = &result.values;

        let expected_last_five = [
            59140.98229179739,
            59172.03593376982,
            59179.40342783722,
            59171.22758152845,
            59127.859841077094,
        ];

        assert!(out_vals.len() >= 5, "Not enough data for test");
        let start_idx = out_vals.len() - 5;
        let actual_last_five = &out_vals[start_idx..];

        for (i, (&actual, &expected)) in
            actual_last_five.iter().zip(&expected_last_five).enumerate()
        {
            let diff = (actual - expected).abs();
            assert!(
                diff < 1e-8,
                "Mismatch at index {}: expected {}, got {}, diff={}",
                i,
                expected,
                actual,
                diff
            );
        }
    }
}
