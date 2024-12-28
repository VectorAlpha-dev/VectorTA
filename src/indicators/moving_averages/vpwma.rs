use crate::utilities::data_loader::{source_type, Candles};
use std::error::Error;

#[derive(Debug, Clone)]
pub struct VwmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct VwmaParams {
    pub period: Option<usize>,
}

impl VwmaParams {
    pub fn with_default_params() -> Self {
        VwmaParams { period: None }
    }
}

#[derive(Debug, Clone)]
pub struct VwmaInput<'a> {
    pub candles: &'a Candles,
    pub source: &'a str,
    pub params: VwmaParams,
}

impl<'a> VwmaInput<'a> {
    pub fn new(candles: &'a Candles, source: &'a str, params: VwmaParams) -> Self {
        VwmaInput { candles, source, params }
    }

    pub fn with_default_params(candles: &'a Candles) -> Self {
        VwmaInput {
            candles,
            source: "close",
            params: VwmaParams::with_default_params(),
        }
    }
}
#[inline]
pub fn vwma(input: &VwmaInput) -> Result<VwmaOutput, Box<dyn Error>> {
    let price: &[f64] = source_type(input.candles, input.source);
    let volume: &Vec<f64> = &input.candles.volume;
    let len: usize = price.len();
    let period: usize = input.params.period.unwrap_or(20);

    if data.len() < (period + 1) {
        return Err(format!(
            "Not enough data: length {} < period+1={}",
            data.len(),
            period + 1
        )
        .into());
    }
    if period < 2 {
        return Err("VPWMA period must be >= 2.".into());
    }
    if power.is_nan() {
        return Err("VPWMA power cannot be NaN.".into());
    }

    let len = data.len();
    let mut vpwma_values = data.to_vec();

    let mut weights = Vec::with_capacity(period - 1);
    for i in 0..(period - 1) {
        let w = (period as f64 - i as f64).powf(power);
        weights.push(w);
    }
    let weight_sum: f64 = weights.iter().sum();

    for j in (period + 1)..len {
        let mut my_sum = 0.0;
        for (i, &w) in weights.iter().enumerate() {
            my_sum = data[j - i].mul_add(w, my_sum);
        }
        vpwma_values[j] = my_sum / weight_sum;
    }

    Ok(VpwmaOutput {
        values: vpwma_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_vwma_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = VwmaParams { period: None };
        let input_default = VwmaInput::new(&candles, "close", default_params);
        let output_default = vwma(&input_default).expect("Failed VWMA with default params");
        assert_eq!(output_default.values.len(), candles.close.len());

        let params_period_14 = VwmaParams { period: Some(14) };
        let input_period_14 = VwmaInput::new(&candles, "hl2", params_period_14);
        let output_period_14 = vwma(&input_period_14).expect("Failed VWMA with period=14, source=hl2");
        assert_eq!(output_period_14.values.len(), candles.close.len());

        let params_custom = VwmaParams { period: Some(30) };
        let input_custom = VwmaInput::new(&candles, "hlc3", params_custom);
        let output_custom = vwma(&input_custom).expect("Failed VWMA fully custom");
        assert_eq!(output_custom.values.len(), candles.close.len());
    }

    #[test]
    fn test_vwma_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let params = VwmaParams { period: Some(20) };
        let input = VwmaInput::new(&candles, "close", params);
        let vwma_result = vwma(&input).expect("Failed to calculate VWMA");

        assert_eq!(vwma_result.values.len(), close_prices.len());

        let expected_last_five_vwma = [
            59201.87047121331,
            59217.157390630266,
            59195.74526905522,
            59196.261392450084,
            59151.22059588594,
        ];
        assert!(vwma_result.values.len() >= 5);
        let start_index = vwma_result.values.len() - 5;
        let result_last_five_vwma = &vwma_result.values[start_index..];

        for (i, &value) in result_last_five_vwma.iter().enumerate() {
            let expected_value = expected_last_five_vwma[i];
            assert!(
                (value - expected_value).abs() < 1e-3,
                "VWMA mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }
    }
}