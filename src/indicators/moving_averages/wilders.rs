use crate::utilities::data_loader::{source_type, Candles};
use std::error::Error;

#[derive(Debug, Clone)]
pub enum WildersData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct WildersOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct WildersParams {
    pub period: Option<usize>,
}

impl WildersParams {
    pub fn with_default_params() -> Self {
        Self { period: None }
    }
}

#[derive(Debug, Clone)]
pub struct WildersInput<'a> {
    pub data: WildersData<'a>,
    pub params: WildersParams,
}

impl<'a> WildersInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: WildersParams) -> Self {
        Self {
            data: WildersData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: WildersParams) -> Self {
        Self {
            data: WildersData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: WildersData::Candles {
                candles,
                source: "close",
            },
            params: WildersParams::with_default_params(),
        }
    }
}

#[inline]
pub fn wilders(input: &WildersInput) -> Result<WildersOutput, Box<dyn Error>> {
    let data: &[f64] = match &input.data {
        WildersData::Candles { candles, source } => source_type(candles, source),
        WildersData::Slice(slice) => slice,
    };
    let n: usize = data.len();
    let period: usize = input.params.period.unwrap_or(5);

    if period == 0 || period > n {
        return Err("Invalid period specified for Wilder's Moving Average.".into());
    }

    let mut out_values = vec![f64::NAN; n];

    let mut sum = 0.0;
    for i in 0..period {
        sum += data[i];
    }

    let mut val = sum / period as f64;
    out_values[period - 1] = val;

    let alpha = 1.0 / period as f64;

    for i in period..n {
        val = (data[i] - val) * alpha + val;
        out_values[i] = val;
    }

    Ok(WildersOutput { values: out_values })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_wilders_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = WildersParams { period: None };
        let input_default = WildersInput::from_candles(&candles, "close", default_params);
        let output_default = wilders(&input_default).expect("Failed Wilders with default params");
        assert_eq!(output_default.values.len(), candles.close.len());

        let custom_params = WildersParams { period: Some(10) };
        let input_custom = WildersInput::from_candles(&candles, "hl2", custom_params);
        let output_custom =
            wilders(&input_custom).expect("Failed Wilders with period=10, source=hl2");
        assert_eq!(output_custom.values.len(), candles.close.len());
    }

    #[test]
    fn test_wilders_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let params = WildersParams { period: Some(5) };
        let input = WildersInput::from_candles(&candles, "close", params);
        let output = wilders(&input).expect("Failed Wilders calculation");
        assert_eq!(output.values.len(), close_prices.len());

        let expected_last_five = [
            59302.18156619092,
            59277.94525295273,
            59230.15620236219,
            59215.12496188975,
            59103.0999695118,
        ];

        assert!(output.values.len() >= 5);
        let start_idx = output.values.len() - 5;
        let actual_last_five = &output.values[start_idx..];
        for (i, (&actual, &expected)) in
            actual_last_five.iter().zip(&expected_last_five).enumerate()
        {
            let diff = (actual - expected).abs();
            assert!(
                diff < 1e-10,
                "Mismatch at index {}: expected {}, got {}, diff {}",
                i,
                expected,
                actual,
                diff
            );
        }

        let input_default = WildersInput::with_default_candles(&candles);
        let default_output = wilders(&input_default).expect("Wilder's default calculation failed");
        assert!(!default_output.values.is_empty());
    }
}
