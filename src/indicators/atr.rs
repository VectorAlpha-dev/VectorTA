use crate::utilities::data_loader::Candles;
use std::error::Error;

#[derive(Debug, Clone)]
pub struct AtrParams {
    pub length: Option<usize>,
}

impl Default for AtrParams {
    fn default() -> Self {
        AtrParams { length: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct AtrInput<'a> {
    pub candles: &'a Candles,
    pub params: AtrParams,
}

impl<'a> AtrInput<'a> {
    pub fn new(candles: &'a Candles, params: AtrParams) -> Self {
        AtrInput { candles, params }
    }

    pub fn with_default_params(candles: &'a Candles) -> Self {
        AtrInput {
            candles,
            params: AtrParams::default(),
        }
    }

    fn get_length(&self) -> usize {
        self.params.length.unwrap_or(14)
    }
}

#[derive(Debug, Clone)]
pub struct AtrOutput {
    pub values: Vec<f64>,
}

#[inline]
pub fn atr(input: &AtrInput) -> Result<AtrOutput, Box<dyn Error>> {
    let candles = input.candles;
    let length = input.get_length();

    if length == 0 {
        return Err("Invalid length for ATR calculation.".into());
    }

    let high = candles.select_candle_field("high")?;
    let low = candles.select_candle_field("low")?;
    let close = candles.select_candle_field("close")?;

    let len = close.len();
    if len == 0 {
        return Err("No candles available.".into());
    }
    if length > len {
        return Err("Not enough data to calculate ATR.".into());
    }

    let mut atr_values = vec![f64::NAN; len];

    let alpha = 1.0 / length as f64;

    let mut sum_tr = 0.0;
    let mut rma = f64::NAN;

    for i in 0..len {
        let tr = if i == 0 {
            high[0] - low[0]
        } else {
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            hl.max(hc).max(lc)
        };

        if i < length {
            sum_tr += tr;
            if i == length - 1 {
                rma = sum_tr / length as f64;
                atr_values[i] = rma;
            }
        } else {
            rma += alpha * (tr - rma);
            atr_values[i] = rma;
        }
    }

    Ok(AtrOutput { values: atr_values })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_atr_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = AtrInput::with_default_params(&candles);
        let result = atr(&input).expect("Failed to calculate ATR");

        let expected_last_five = [916.89, 874.33, 838.45, 801.92, 811.57];

        assert!(result.values.len() >= 5, "Not enough ATR values");
        assert_eq!(
            result.values.len(),
            candles.close.len(),
            "ATR output length does not match input length!"
        );
        let start_index = result.values.len().saturating_sub(5);
        let last_five = &result.values[start_index..];

        for (i, &value) in last_five.iter().enumerate() {
            assert!(
                (value - expected_last_five[i]).abs() < 1e-2,
                "ATR value mismatch at index {}: expected {}, got {}",
                i,
                expected_last_five[i],
                value
            );
        }

        let length = input.get_length();
        for val in result.values.iter().skip(length - 1) {
            if !val.is_nan() {
                assert!(
                    val.is_finite(),
                    "ATR output should be finite after RMA stabilizes"
                );
            }
        }
    }
}
