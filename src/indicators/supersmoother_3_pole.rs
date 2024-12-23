use std::error::Error;
use std::f64::consts::PI;

#[derive(Debug, Clone)]
pub struct SuperSmoother3PoleParams {
    pub period: Option<usize>,
}

impl Default for SuperSmoother3PoleParams {
    fn default() -> Self {
        SuperSmoother3PoleParams { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct SuperSmoother3PoleInput<'a> {
    pub data: &'a [f64],
    pub params: SuperSmoother3PoleParams,
}

impl<'a> SuperSmoother3PoleInput<'a> {
    pub fn new(data: &'a [f64], params: SuperSmoother3PoleParams) -> Self {
        SuperSmoother3PoleInput { data, params }
    }

    pub fn with_default_params(data: &'a [f64]) -> Self {
        SuperSmoother3PoleInput {
            data,
            params: SuperSmoother3PoleParams::default(),
        }
    }

    fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| SuperSmoother3PoleParams::default().period.unwrap())
    }
}

#[derive(Debug, Clone)]
pub struct SuperSmoother3PoleOutput {
    pub values: Vec<f64>,
}

#[inline]
pub fn calculate_supersmoother_3_pole(
    input: &SuperSmoother3PoleInput,
) -> Result<SuperSmoother3PoleOutput, Box<dyn Error>> {
    let data = input.data;
    let period = input.get_period();

    if data.is_empty() {
        return Ok(SuperSmoother3PoleOutput { values: vec![] });
    }
    if period < 1 {
        return Err("Period must be >= 1 for 3-pole SuperSmoother.".into());
    }

    let n = data.len();
    let mut output = vec![0.0; n];

    let a = (-PI / period as f64).exp();
    let b = 2.0 * a * (1.738_f64 * PI / period as f64).cos();
    let c = a * a;

    if n > 0 {
        output[0] = data[0];
    }
    if n > 1 {
        output[1] = data[1];
    }
    if n > 2 {
        output[2] = data[2];
    }

    let coef_source = 1.0 - c * c - b + b * c;
    let coef_prev1 = b + c;
    let coef_prev2 = -c - b * c;
    let coef_prev3 = c * c;

    for i in 3..n {
        let d_i = data[i];
        let o_im1 = output[i - 1];
        let o_im2 = output[i - 2];
        let o_im3 = output[i - 3];

        output[i] =
            coef_source * d_i + coef_prev1 * o_im1 + coef_prev2 * o_im2 + coef_prev3 * o_im3;
    }

    Ok(SuperSmoother3PoleOutput { values: output })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::data_loader::read_candles_from_csv;

    #[test]
    fn test_supersmoother_3pole_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let params = SuperSmoother3PoleParams { period: Some(14) };
        let input = SuperSmoother3PoleInput::new(&close_prices, params);

        let result = calculate_supersmoother_3_pole(&input)
            .expect("Failed to calculate 3-pole SuperSmoother");
        let values = &result.values;

        let expected_last_five = [
            59072.13481064446,
            59089.08032603,
            59111.35711851466,
            59133.14402399381,
            59121.91820047289,
        ];

        assert!(
            values.len() >= 5,
            "Not enough 3-pole SS output to compare final 5 values"
        );

        let start_idx = values.len() - 5;
        let last_five = &values[start_idx..];

        for (i, (&actual, &expected)) in last_five.iter().zip(&expected_last_five).enumerate() {
            let diff = (actual - expected).abs();
            assert!(
                diff < 1e-8,
                "3-pole SuperSmoother mismatch at index {}: expected {}, got {}, diff {}",
                i,
                expected,
                actual,
                diff
            );
        }
    }
}
