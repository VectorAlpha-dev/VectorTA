use crate::utilities::data_loader::{source_type, Candles};
use std::error::Error;
use std::f64::consts::PI;

#[derive(Debug, Clone)]
pub struct SinWmaParams {
    pub period: Option<usize>,
}

impl Default for SinWmaParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct SinWmaInput<'a> {
    pub candles: &'a Candles,
    pub source: &'a str,
    pub params: SinWmaParams,
}

impl<'a> SinWmaInput<'a> {
    #[inline]
    pub fn new(candles: &'a Candles, source: &'a str, params: SinWmaParams) -> Self {
        Self { candles, source, params }
    }

    #[inline]
    pub fn with_default_params(candles: &'a Candles) -> Self {
        Self {
            candles,
            source: "close",
            params: SinWmaParams::default(),
        }
    }

    #[inline]
    fn get_period(&self) -> usize {
        self.params.period.unwrap_or_else(|| SinWmaParams::default().period.unwrap())
    }
}

#[derive(Debug, Clone)]
pub struct SinWmaOutput {
    pub values: Vec<f64>,
}

#[inline(always)]
fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

#[inline]
pub fn sinwma(input: &SinWmaInput) -> Result<SinWmaOutput, Box<dyn Error>> {
    let data: &[f64] = source_type(input.candles, input.source);
    let period: usize = input.get_period();
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
        let input = SinWmaInput::new(&candles, "close", params);
        let result = sinwma(&input).expect("Failed to calculate SINWMA");
        assert_eq!(result.values.len(), close_prices.len());
        let expected_last_five = [
            59376.72903536103,
            59300.76862770367,
            59229.27622157621,
            59178.48781774477,
            59154.66580703081,
        ];
        assert!(result.values.len() >= 5);
        let start_index = result.values.len() - 5;
        let last_five = &result.values[start_index..];
        for (i, &value) in last_five.iter().enumerate() {
            let expected_value = expected_last_five[i];
            assert!(
                (value - expected_value).abs() < 1e-6,
                "SINWMA mismatch at {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }
    }

    #[test]
    fn test_sinwma_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let default_params = SinWmaParams { period: None };
        let input = SinWmaInput::new(&candles, "close", default_params);
        let output = sinwma(&input).expect("Failed SINWMA with default params");
        assert_eq!(output.values.len(), candles.close.len());
        let params_period_10 = SinWmaParams { period: Some(10) };
        let input2 = SinWmaInput::new(&candles, "hl2", params_period_10);
        let output2 = sinwma(&input2).expect("Failed SINWMA with period=10, source=hl2");
        assert_eq!(output2.values.len(), candles.close.len());
        let params_custom = SinWmaParams { period: Some(20) };
        let input3 = SinWmaInput::new(&candles, "hlc3", params_custom);
        let output3 = sinwma(&input3).expect("Failed SINWMA fully custom");
        assert_eq!(output3.values.len(), candles.close.len());
    }
}