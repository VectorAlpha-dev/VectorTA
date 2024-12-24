use crate::indicators::data_loader::Candles;
use std::error::Error;

#[derive(Debug, Clone, Default)]
pub struct AdParams {}

#[derive(Debug, Clone)]
pub struct AdInput<'a> {
    pub candles: &'a Candles,
    pub params: AdParams,
}

impl<'a> AdInput<'a> {
    pub fn new(candles: &'a Candles, params: AdParams) -> Self {
        AdInput { candles, params }
    }

    pub fn with_default_params(candles: &'a Candles) -> Self {
        AdInput {
            candles,
            params: AdParams::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AdOutput {
    pub values: Vec<f64>,
}

#[inline]
pub fn calculate_ad(input: &AdInput) -> Result<AdOutput, Box<dyn Error>> {
    let candles = input.candles;
    let high: &[f64] = candles.select_candle_field("high")?;
    let low: &[f64] = candles.select_candle_field("low")?;
    let close: &[f64] = candles.select_candle_field("close")?;
    let volume: &[f64] = candles.select_candle_field("volume")?;

    let size: usize = high.len();
    if size < 1 {
        return Err("Not enough data points to calculate AD.".into());
    }
    let mut output: Vec<f64> = Vec::with_capacity(size);
    let mut sum: f64 = 0.0;

    for ((&h, &l), (&c, &v)) in high
        .iter()
        .zip(low.iter())
        .zip(close.iter().zip(volume.iter()))
    {
        let hl = h - l;

        if hl != 0.0 {
            let mfm: f64 = ((c - l) - (h - c)) / hl;
            let mfv: f64 = mfm * v;
            sum += mfv;
        }
        output.push(sum);
    }

    Ok(AdOutput { values: output })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::data_loader::read_candles_from_csv;

    #[test]
    fn test_ad_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = AdInput::with_default_params(&candles);
        let ad_result = calculate_ad(&input).expect("Failed to calculate AD");

        assert_eq!(
            ad_result.values.len(),
            candles.close.len(),
            "AD output length does not match input length"
        );

        let expected_last_five_ad = [1645918.16, 1645876.11, 1645824.27, 1645828.87, 1645728.78];

        assert!(
            ad_result.values.len() >= 5,
            "Not enough AD values for the test"
        );
        let start_index = ad_result.values.len() - 5;
        let result_last_five_ad = &ad_result.values[start_index..];

        for (i, &value) in result_last_five_ad.iter().enumerate() {
            let expected_value = expected_last_five_ad[i];
            assert!(
                (value - expected_value).abs() < 1e-1,
                "AD value mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }
    }
}