use std::error::Error;

#[derive(Debug, Clone)]
pub struct NmaParams {
    pub period: Option<usize>,
}

impl Default for NmaParams {
    fn default() -> Self {
        NmaParams { period: Some(40) }
    }
}

#[derive(Debug, Clone)]
pub struct NmaInput<'a> {
    pub data: &'a [f64],
    pub params: NmaParams,
}

impl<'a> NmaInput<'a> {
    pub fn new(data: &'a [f64], params: NmaParams) -> Self {
        NmaInput { data, params }
    }

    pub fn with_default_params(data: &'a [f64]) -> Self {
        NmaInput {
            data,
            params: NmaParams::default(),
        }
    }

    fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| NmaParams::default().period.unwrap())
    }
}

#[derive(Debug, Clone)]
pub struct NmaOutput {
    pub values: Vec<f64>,
}

#[inline]
pub fn calculate_nma(input: &NmaInput) -> Result<NmaOutput, Box<dyn Error>> {
    let data = input.data;
    let period = input.get_period();

    if period == 0 {
        return Err("NMA period cannot be zero.".into());
    }
    if data.len() < (period + 1) {
        return Err(format!(
            "Not enough data ({}) for NMA with period {} (need at least period+1).",
            data.len(),
            period
        )
        .into());
    }

    let len = data.len();

    let mut ln_values = Vec::with_capacity(len);
    ln_values.extend(data.iter().map(|&val| {
        let clamped = val.max(1e-10);
        clamped.ln() * 1000.0
    }));

    let mut sqrt_diffs = Vec::with_capacity(period);
    for i in 0..period {
        let s0 = (i as f64).sqrt();
        let s1 = ((i + 1) as f64).sqrt();
        sqrt_diffs.push(s1 - s0);
    }

    let mut nma_values = vec![f64::NAN; len];

    for j in (period + 1)..len {
        let mut num = 0.0;
        let mut denom = 0.0;

        for i in 0..period {
            let oi = (ln_values[j - i] - ln_values[j - i - 1]).abs();
            num += oi * sqrt_diffs[i];
            denom += oi;
        }

        let ratio = if denom == 0.0 { 0.0 } else { num / denom };

        let i = period - 1;
        nma_values[j] = data[j - i] * ratio + data[j - i - 1] * (1.0 - ratio);
    }

    Ok(NmaOutput {
        values: nma_values,
    })
}

#[cfg(test)]
 mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_nma_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let params = NmaParams { period: Some(40) };
        let input = NmaInput::new(close_prices, params);
        let nma_result = calculate_nma(&input).expect("Failed to calculate NMA");

        assert_eq!(
            nma_result.values.len(),
            close_prices.len(),
            "NMA values count should match the input data length"
        );

        let period = input.get_period();
        for i in 0..=(period) {
            assert!(
                nma_result.values[i].is_nan(),
                "Expected NaN at index {}, got {}",
                i,
                nma_result.values[i]
            );
        }

        let expected_last_five_nma = [
            64320.486018271724,
            64227.95719984426,
            64180.9249333126,
            63966.35530620797,
            64039.04719192334,
        ];
        assert!(
            nma_result.values.len() >= 5,
            "Not enough NMA values for the test"
        );

        let start_index = nma_result.values.len() - 5;
        let result_last_five_nma = &nma_result.values[start_index..];

        for (i, &value) in result_last_five_nma.iter().enumerate() {
            let expected_value = expected_last_five_nma[i];
            assert!(
                (value - expected_value).abs() < 1e-3,
                "NMA value mismatch at last-5 index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }

        let default_input = NmaInput::with_default_params(close_prices);
        let default_nma_result =
            calculate_nma(&default_input).expect("Failed to calculate NMA with defaults");
        assert_eq!(
            default_nma_result.values.len(),
            close_prices.len(),
            "Should produce full-length NMA values with default params"
        );
    }
}
