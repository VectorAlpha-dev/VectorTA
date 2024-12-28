use crate::utilities::data_loader::{source_type, Candles};
use std::error::Error;

#[derive(Debug, Clone)]
pub struct NmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct NmaParams {
    pub period: Option<usize>,
}

impl NmaParams {
    pub fn with_default_params() -> Self {
        NmaParams { period: None }
    }
}

#[derive(Debug, Clone)]
pub struct NmaInput<'a> {
    pub candles: &'a Candles,
    pub source: &'a str,
    pub params: NmaParams,
}

impl<'a> NmaInput<'a> {
    pub fn new(candles: &'a Candles, source: &'a str, params: NmaParams) -> Self {
        NmaInput {
            candles,
            source,
            params,
        }
    }

    pub fn with_default_params(candles: &'a Candles) -> Self {
        NmaInput {
            candles,
            source: "close",
            params: NmaParams::with_default_params(),
        }
    }
}
#[inline]
pub fn nma(input: &NmaInput) -> Result<NmaOutput, Box<dyn Error>> {
    let data: &[f64] = source_type(input.candles, input.source);
    let len: usize = data.len();
    let period: usize = input.params.period.unwrap_or(40);

    if period == 0 {
        return Err("NMA period cannot be zero.".into());
    }
    if len < (period + 1) {
        return Err(format!(
            "Not enough data ({}) for NMA with period {} (need at least period+1).",
            len, period
        )
        .into());
    }

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

    Ok(NmaOutput { values: nma_values })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_nma_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = NmaParams { period: None };
        let input_default = NmaInput::new(&candles, "close", default_params);
        let output_default = nma(&input_default).expect("Failed NMA with default params");
        assert_eq!(output_default.values.len(), candles.close.len());

        let params_14 = NmaParams { period: Some(14) };
        let input_period_14 = NmaInput::new(&candles, "hl2", params_14);
        let output_period_14 =
            nma(&input_period_14).expect("Failed NMA with period=14, source=hl2");
        assert_eq!(output_period_14.values.len(), candles.close.len());

        let params_custom = NmaParams { period: Some(20) };
        let input_custom = NmaInput::new(&candles, "hlc3", params_custom);
        let output_custom = nma(&input_custom).expect("Failed NMA fully custom");
        assert_eq!(output_custom.values.len(), candles.close.len());
    }

    #[test]
    fn test_nma_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let params = NmaParams { period: Some(40) };
        let input = NmaInput::new(&candles, "close", params);
        let nma_result = nma(&input).expect("Failed to calculate NMA");

        assert_eq!(
            nma_result.values.len(),
            close_prices.len(),
            "NMA values count should match the input data length"
        );

        let period = 40;
        for i in 0..=period {
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
        assert!(nma_result.values.len() >= 5);

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

        let default_input = NmaInput::with_default_params(&candles);
        let default_nma_result =
            nma(&default_input).expect("Failed to calculate NMA with defaults");
        assert_eq!(
            default_nma_result.values.len(),
            close_prices.len(),
            "Should produce full-length NMA values with default params"
        );
    }
}
