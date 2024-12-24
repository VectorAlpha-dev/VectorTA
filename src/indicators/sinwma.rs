use std::error::Error;
use std::f64::consts::PI;

#[derive(Debug, Clone)]
pub struct SinWmaParams {
    pub period: Option<usize>,
}

impl Default for SinWmaParams {
    fn default() -> Self {
        SinWmaParams { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct SinWmaInput<'a> {
    pub data: &'a [f64],
    pub params: SinWmaParams,
}

impl<'a> SinWmaInput<'a> {
    pub fn new(data: &'a [f64], params: SinWmaParams) -> Self {
        SinWmaInput { data, params }
    }

    pub fn with_default_params(data: &'a [f64]) -> Self {
        SinWmaInput {
            data,
            params: SinWmaParams::default(),
        }
    }

    fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| SinWmaParams::default().period.unwrap())
    }
}

#[derive(Debug, Clone)]
pub struct SinWmaOutput {
    pub values: Vec<f64>,
}

#[inline(always)]
fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum = 0.0;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

#[inline]
pub fn calculate_sinwma(input: &SinWmaInput) -> Result<SinWmaOutput, Box<dyn Error>> {
    let data = input.data;
    let period = input.get_period();

    if period == 0 || period > data.len() {
        return Err("Invalid period for SINWMA calculation.".into());
    }

    let mut sines = Vec::with_capacity(period);
    let mut sum_sines = 0.0;
    for k in 0..period {
        let angle = (k as f64 + 1.0) * PI / (period as f64 + 1.0);
        let val = angle.sin();
        sum_sines += val;
        sines.push(val);
    }

    let inv_sum = 1.0 / sum_sines;
    for w in &mut sines {
        *w *= inv_sum;
    }

    let len = data.len();
    let mut sinwma_values = vec![f64::NAN; len];

    for i in (period - 1)..len {
        let start_idx = i + 1 - period;
        let data_window = &data[start_idx..(start_idx + period)];
        let value = dot_product(data_window, &sines);
        sinwma_values[i] = value;
    }

    Ok(SinWmaOutput {
        values: sinwma_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_sinwma_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let params = SinWmaParams { period: Some(14) };
        let input = SinWmaInput::new(close_prices, params);
        let sinwma_result = calculate_sinwma(&input).expect("Failed to calculate SINWMA");

        assert_eq!(
            sinwma_result.values.len(),
            close_prices.len(),
            "SINWMA output length should match input data length"
        );

        let expected_last_five = [
            59376.72903536103,
            59300.76862770367,
            59229.27622157621,
            59178.48781774477,
            59154.66580703081,
        ];
        assert!(
            sinwma_result.values.len() >= 5,
            "Not enough SINWMA values for the test"
        );

        let start_index = sinwma_result.values.len() - 5;
        let result_last_five = &sinwma_result.values[start_index..];

        for (i, &value) in result_last_five.iter().enumerate() {
            let expected_value = expected_last_five[i];
            assert!(
                (value - expected_value).abs() < 1e-6,
                "SINWMA mismatch at last 5 index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }
    }
}
