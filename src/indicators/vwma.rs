use crate::utilities::data_loader::Candles;
use std::error::Error;

#[derive(Debug, Clone)]
pub struct VwmaParams {
    pub period: Option<usize>,
}

impl Default for VwmaParams {
    fn default() -> Self {
        VwmaParams { period: Some(20) }
    }
}

#[derive(Debug, Clone)]
pub struct VwmaInput<'a> {
    pub candles: &'a Candles,
    pub params: VwmaParams,
}

impl<'a> VwmaInput<'a> {
    pub fn new(candles: &'a Candles, params: VwmaParams) -> Self {
        VwmaInput { candles, params }
    }

    pub fn with_default_params(candles: &'a Candles) -> Self {
        VwmaInput {
            candles,
            params: VwmaParams::default(),
        }
    }

    fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| VwmaParams::default().period.unwrap())
    }
}

#[derive(Debug, Clone)]
pub struct VwmaOutput {
    pub values: Vec<f64>,
}

#[inline]
pub fn calculate_vwma(input: &VwmaInput) -> Result<VwmaOutput, Box<dyn Error>> {
    let period = input.get_period();
    let candles = input.candles;
    let price = candles.select_candle_field("close")?;
    let volume = candles.select_candle_field("volume")?;

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
    fn test_vwma_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");
        let params = VwmaParams { period: Some(20) };
        let input = VwmaInput::new(&candles, params);
        let vwma_result = calculate_vwma(&input).expect("Failed to calculate VWMA");

        assert_eq!(
            vwma_result.values.len(),
            close_prices.len(),
            "VWMA values count should match the input data length"
        );

        let expected_last_five_vwma = [
            59201.87047121331,
            59217.157390630266,
            59195.74526905522,
            59196.261392450084,
            59151.22059588594,
        ];
        assert!(
            vwma_result.values.len() >= 5,
            "Not enough VWMA values for the test"
        );

        let start_index = vwma_result.values.len() - 5;
        let result_last_five_vwma = &vwma_result.values[start_index..];

        for (i, &value) in result_last_five_vwma.iter().enumerate() {
            let expected_value = expected_last_five_vwma[i];
            assert!(
                (value - expected_value).abs() < 1e-3,
                "VWMA value mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }
    }
}
