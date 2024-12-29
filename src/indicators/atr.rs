use crate::utilities::data_loader::Candles;
use std::error::Error;

#[derive(Debug, Clone)]
pub enum AtrData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct AtrParams {
    pub length: Option<usize>,
}

impl Default for AtrParams {
    fn default() -> Self {
        Self { length: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct AtrInput<'a> {
    pub data: AtrData<'a>,
    pub params: AtrParams,
}

impl<'a> AtrInput<'a> {
    pub fn from_candles(candles: &'a Candles, params: AtrParams) -> Self {
        Self {
            data: AtrData::Candles { candles },
            params,
        }
    }

    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: AtrParams,
    ) -> Self {
        Self {
            data: AtrData::Slices { high, low, close },
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: AtrData::Candles { candles },
            params: AtrParams::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AtrOutput {
    pub values: Vec<f64>,
}

#[inline]
pub fn atr(input: &AtrInput) -> Result<AtrOutput, Box<dyn Error>> {
    let length: usize = input.params.length.unwrap_or(14);
    if length == 0 {
        return Err("Invalid length for ATR calculation.".into());
    }

    let (high, low, close) = match &input.data {
        AtrData::Candles { candles } => {
            let high: &[f64] = candles.select_candle_field("high")?;
            let low: &[f64] = candles.select_candle_field("low")?;
            let close: &[f64] = candles.select_candle_field("close")?;
            (high, low, close)
        }
        AtrData::Slices { high, low, close } => {
            if high.len() != low.len() || low.len() != close.len() {
                return Err("Inconsistent slice lengths for ATR calculation.".into());
            }
            (*high, *low, *close)
        }
    };

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
    fn test_atr_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let partial_params = AtrParams { length: None };
        let input_partial = AtrInput::from_candles(&candles, partial_params);
        let result_partial = atr(&input_partial).expect("Failed ATR with partial params");
        assert_eq!(result_partial.values.len(), candles.close.len());

        let zero_and_none_params = AtrParams { length: Some(14) };
        let input_zero_and_none = AtrInput::from_candles(&candles, zero_and_none_params);
        let result_zero_and_none =
            atr(&input_zero_and_none).expect("Failed ATR with zero/none combo");
        assert_eq!(result_zero_and_none.values.len(), candles.close.len());
    }

    #[test]
    fn test_atr_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = AtrInput::with_default_candles(&candles);
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

        let length = 14;
        for val in result.values.iter().skip(length - 1) {
            if !val.is_nan() {
                assert!(
                    val.is_finite(),
                    "ATR output should be finite after RMA stabilizes"
                );
            }
        }
    }
    #[test]
    fn test_atr_params_with_default_params() {
        let default_params = AtrParams::default();
        assert_eq!(default_params.length, Some(14));
    }

    #[test]
    fn test_atr_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = AtrInput::with_default_candles(&candles);
        match input.data {
            AtrData::Candles { .. } => {}
            _ => panic!("Expected AtrData::Candles variant"),
        }
        let default_params = AtrParams::default();
        assert_eq!(input.params.length, default_params.length);
    }

    #[test]
    fn test_atr_with_zero_length() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let zero_length_params = AtrParams { length: Some(0) };
        let input_zero_length = AtrInput::from_candles(&candles, zero_length_params);
        let result_zero_length = atr(&input_zero_length);
        assert!(result_zero_length.is_err());
    }

    #[test]
    fn test_atr_length_exceeding_data_length() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let too_long_params = AtrParams {
            length: Some(candles.close.len() + 10),
        };
        let input_too_long = AtrInput::from_candles(&candles, too_long_params);
        let result_too_long = atr(&input_too_long);
        assert!(result_too_long.is_err());
    }

    #[test]
    fn test_atr_very_small_data_set() {
        let high = [10.0];
        let low = [5.0];
        let close = [7.0];
        let params = AtrParams { length: Some(14) };
        let input = AtrInput::from_slices(&high, &low, &close, params);
        let result = atr(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_atr_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let first_params = AtrParams { length: Some(14) };
        let first_input = AtrInput::from_candles(&candles, first_params);
        let first_result = atr(&first_input).expect("Failed ATR (first run)");
        assert_eq!(first_result.values.len(), candles.close.len());
        let second_params = AtrParams { length: Some(5) };
        let second_input = AtrInput::from_slices(
            &first_result.values,
            &first_result.values,
            &first_result.values,
            second_params,
        );
        let second_result = atr(&second_input).expect("Failed ATR (second run)");
        assert_eq!(second_result.values.len(), first_result.values.len());
    }

    #[test]
    fn test_atr_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = AtrParams { length: Some(14) };
        let input = AtrInput::from_candles(&candles, params);
        let result = atr(&input).expect("Failed to calculate ATR");
        assert_eq!(result.values.len(), candles.close.len());
        if result.values.len() > 240 {
            for i in 240..result.values.len() {
                assert!(!result.values[i].is_nan());
            }
        }
    }
}
