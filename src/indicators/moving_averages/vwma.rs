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
        VwmaInput {
            candles,
            source,
            params,
        }
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
    let volume = candles.select_candle_field("volume").expect("Failed to extract volume");
    let len: usize = price.len();
    let period: usize = input.params.period.unwrap_or(20);

    if period == 0 || period > price.len() {
        return Err("Invalid period for VWMA calculation.".into());
    }
    let len = price.len();
    if len != volume.len() {
        return Err("Price and volume mismatch.".into());
    }

    let mut vwma_values = vec![f64::NAN; len];

    let mut sum = 0.0;
    let mut vsum = 0.0;

    for i in 0..period {
        sum += price[i] * volume[i];
        vsum += volume[i];
    }
    vwma_values[period - 1] = sum / vsum;

    for i in period..len {
        sum += price[i] * volume[i];
        sum -= price[i - period] * volume[i - period];

        vsum += volume[i];
        vsum -= volume[i - period];

        vwma_values[i] = sum / vsum;
    }

    Ok(VwmaOutput {
        values: vwma_values,
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

        let custom_params = VwmaParams { period: Some(10) };
        let input_custom = VwmaInput::new(&candles, "hlc3", custom_params);
        let output_custom = vwma(&input_custom).expect("Failed VWMA with period=10, source=hlc3");
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
        for (i, &val) in result_last_five_vwma.iter().enumerate() {
            let exp = expected_last_five_vwma[i];
            assert!(
                (val - exp).abs() < 1e-3,
                "VWMA mismatch at index {}: expected {}, got {}",
                i,
                exp,
                val
            );
        }
    }
}