/// # Normalized Moving Average (NMA)
///
/// A technique that computes an adaptive moving average by transforming the input
/// values into log space and weighting differences between consecutive values.
/// The weighting ratio depends on a series of square-root increments. This design
/// aims to normalize large price changes without oversmoothing small fluctuations.
///
/// ## Parameters
/// - **period**: The look-back window size (in bars). Defaults to 40.
///
/// ## Errors
/// - **AllValuesNaN**: nma: All input data values are `NaN`.
/// - **PeriodCannotBeZero**: nma: `period` cannot be zero.
/// - **NotEnoughData**: nma: Data length is less than `period + 1`, making the NMA calculation impossible.
///
/// ## Returns
/// - **`Ok(NmaOutput)`** on success, containing a `Vec<f64>` with the same length as the input.
/// - **`Err(NmaError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};

#[derive(Debug, Clone)]
pub enum NmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct NmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct NmaParams {
    pub period: Option<usize>,
}

impl Default for NmaParams {
    fn default() -> Self {
        Self { period: Some(40) }
    }
}

#[derive(Debug, Clone)]
pub struct NmaInput<'a> {
    pub data: NmaData<'a>,
    pub params: NmaParams,
}

impl<'a> NmaInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: NmaParams) -> Self {
        Self {
            data: NmaData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: NmaParams) -> Self {
        Self {
            data: NmaData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: NmaData::Candles {
                candles,
                source: "close",
            },
            params: NmaParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| NmaParams::default().period.unwrap())
    }
}

use thiserror::Error;

#[derive(Debug, Error)]
pub enum NmaError {
    #[error("nma: All values are NaN.")]
    AllValuesNaN,
    #[error("nma: period cannot be zero.")]
    PeriodCannotBeZero,
    #[error("nma: Not enough data: len = {len}, period = {period} (need at least period + 1).")]
    NotEnoughData { len: usize, period: usize },
}

#[inline]
pub fn nma(input: &NmaInput) -> Result<NmaOutput, NmaError> {
    let data: &[f64] = match &input.data {
        NmaData::Candles { candles, source } => source_type(candles, source),
        NmaData::Slice(slice) => slice,
    };

    let len = data.len();
    let period = input.get_period();

    if period == 0 {
        return Err(NmaError::PeriodCannotBeZero);
    }

    if len < period + 1 {
        return Err(NmaError::NotEnoughData { len, period });
    }

    let first_valid_idx = data.iter().position(|&x| !x.is_nan());
    if first_valid_idx.is_none() {
        return Err(NmaError::AllValuesNaN);
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
        let input_default = NmaInput::from_candles(&candles, "close", default_params);
        let output_default = nma(&input_default).expect("Failed NMA with default params");
        assert_eq!(output_default.values.len(), candles.close.len());

        let params_14 = NmaParams { period: Some(14) };
        let input_period_14 = NmaInput::from_candles(&candles, "hl2", params_14);
        let output_period_14 =
            nma(&input_period_14).expect("Failed NMA with period=14, source=hl2");
        assert_eq!(output_period_14.values.len(), candles.close.len());

        let params_custom = NmaParams { period: Some(20) };
        let input_custom = NmaInput::from_candles(&candles, "hlc3", params_custom);
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
        let input = NmaInput::from_candles(&candles, "close", params);
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

        let default_input = NmaInput::with_default_candles(&candles);
        let default_nma_result =
            nma(&default_input).expect("Failed to calculate NMA with defaults");
        assert_eq!(
            default_nma_result.values.len(),
            close_prices.len(),
            "Should produce full-length NMA values with default params"
        );
    }
    #[test]
    fn test_nma_params_with_default_params() {
        let params = NmaParams::default();
        assert_eq!(params.period, Some(40));
    }

    #[test]
    fn test_nma_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = NmaInput::with_default_candles(&candles);
        match input.data {
            NmaData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected NmaData::Candles variant"),
        }
    }

    #[test]
    fn test_nma_period_zero() {
        let data = [10.0, 20.0, 30.0];
        let params = NmaParams { period: Some(0) };
        let input = NmaInput::from_slice(&data, params);
        let result = nma(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_nma_not_enough_data() {
        let data = [10.0, 20.0, 30.0];
        let params = NmaParams { period: Some(40) };
        let input = NmaInput::from_slice(&data, params);
        let result = nma(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_nma_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params_1 = NmaParams { period: Some(40) };
        let input_1 = NmaInput::from_candles(&candles, "close", params_1);
        let result_1 = nma(&input_1).expect("Failed first NMA");
        assert_eq!(result_1.values.len(), candles.close.len());
        let params_2 = NmaParams { period: Some(20) };
        let input_2 = NmaInput::from_slice(&result_1.values, params_2);
        let result_2 = nma(&input_2).expect("Failed second NMA");
        assert_eq!(result_2.values.len(), result_1.values.len());
        if result_2.values.len() > 240 {
            for i in 240..result_2.values.len() {
                assert!(result_2.values[i].is_finite());
            }
        }
    }

    #[test]
    fn test_nma_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = NmaParams { period: Some(40) };
        let input = NmaInput::from_candles(&candles, "close", params);
        let nma_result = nma(&input).expect("Failed to calculate NMA");
        assert_eq!(nma_result.values.len(), candles.close.len());
        if nma_result.values.len() > 240 {
            for i in 240..nma_result.values.len() {
                assert!(!nma_result.values[i].is_nan());
            }
        }
    }
}
