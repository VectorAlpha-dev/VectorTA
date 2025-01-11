/// # Laguerre RSI (LRSI)
///
/// A momentum oscillator using a Laguerre filter. Calculates an RSI-like measure
/// based on recursive Laguerre transformations of the price data. The price is
/// derived by averaging the high and low values of each candle (or by using the
/// provided slice data).
///
/// ## Parameters
/// - **alpha**: The smoothing factor (0 < alpha < 1). Defaults to 0.2.
///
/// ## Errors
/// - **EmptyData**: lrsi: Input data slice is empty.
/// - **InvalidAlpha**: lrsi: `alpha` must be between 0 and 1 (exclusive).
/// - **AllValuesNaN**: lrsi: All input data values are `NaN`.
///
/// ## Returns
/// - **`Ok(LrsiOutput)`** on success, containing a `Vec<f64>` matching the input length,
///   with leading `NaN`s until the first valid index.
/// - **`Err(LrsiError)`** otherwise.
use crate::utilities::data_loader::Candles;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum LrsiData<'a> {
    Candles { candles: &'a Candles },
    Slices { high: &'a [f64], low: &'a [f64] },
}

#[derive(Debug, Clone)]
pub struct LrsiOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct LrsiParams {
    pub alpha: Option<f64>,
}

impl Default for LrsiParams {
    fn default() -> Self {
        Self { alpha: Some(0.2) }
    }
}

#[derive(Debug, Clone)]
pub struct LrsiInput<'a> {
    pub data: LrsiData<'a>,
    pub params: LrsiParams,
}

impl<'a> LrsiInput<'a> {
    pub fn from_candles(candles: &'a Candles, params: LrsiParams) -> Self {
        Self {
            data: LrsiData::Candles { candles },
            params,
        }
    }

    pub fn from_slices(high: &'a [f64], low: &'a [f64], params: LrsiParams) -> Self {
        Self {
            data: LrsiData::Slices { high, low },
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: LrsiData::Candles { candles },
            params: LrsiParams::default(),
        }
    }

    pub fn get_alpha(&self) -> f64 {
        self.params
            .alpha
            .unwrap_or_else(|| LrsiParams::default().alpha.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum LrsiError {
    #[error("lrsi: Empty data provided.")]
    EmptyData,
    #[error("lrsi: Invalid alpha: alpha = {alpha}. Must be between 0 and 1.")]
    InvalidAlpha { alpha: f64 },
    #[error("lrsi: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn lrsi(input: &LrsiInput) -> Result<LrsiOutput, LrsiError> {
    let (high, low) = match &input.data {
        LrsiData::Candles { candles } => {
            let high = candles.select_candle_field("high").unwrap();
            let low = candles.select_candle_field("low").unwrap();
            (high, low)
        }
        LrsiData::Slices { high, low } => (*high, *low),
    };

    if high.is_empty() || low.is_empty() {
        return Err(LrsiError::EmptyData);
    }

    let alpha = input.get_alpha();
    if !(0.0 < alpha && alpha < 1.0) {
        return Err(LrsiError::InvalidAlpha { alpha });
    }

    let mut price = Vec::with_capacity(high.len());
    for i in 0..high.len() {
        price.push((high[i] + low[i]) / 2.0);
    }

    let first_valid_idx = match price.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(LrsiError::AllValuesNaN),
    };
    let n = high.len();

    let mut l0 = vec![f64::NAN; n];
    let mut l1 = vec![f64::NAN; n];
    let mut l2 = vec![f64::NAN; n];
    let mut l3 = vec![f64::NAN; n];
    let mut rsi_values = vec![f64::NAN; n];

    l0[first_valid_idx] = price[first_valid_idx];
    l1[first_valid_idx] = price[first_valid_idx];
    l2[first_valid_idx] = price[first_valid_idx];
    l3[first_valid_idx] = price[first_valid_idx];
    rsi_values[first_valid_idx] = 0.0;

    let gamma = 1.0 - alpha;
    for i in (first_valid_idx + 1)..n {
        let p = price[i];
        if p.is_nan() {
            continue;
        }

        let l0_prev = l0[i - 1];
        let l1_prev = l1[i - 1];
        let l2_prev = l2[i - 1];
        let l3_prev = l3[i - 1];

        l0[i] = alpha * p + gamma * l0_prev;
        l1[i] = -gamma * l0[i] + l0_prev + gamma * l1_prev;
        l2[i] = -gamma * l1[i] + l1_prev + gamma * l2_prev;
        l3[i] = -gamma * l2[i] + l2_prev + gamma * l3_prev;

        let mut cu = 0.0;
        let mut cd = 0.0;
        if l0[i] >= l1[i] {
            cu += l0[i] - l1[i];
        } else {
            cd += l1[i] - l0[i];
        }
        if l1[i] >= l2[i] {
            cu += l1[i] - l2[i];
        } else {
            cd += l2[i] - l1[i];
        }
        if l2[i] >= l3[i] {
            cu += l2[i] - l3[i];
        } else {
            cd += l3[i] - l2[i];
        }

        rsi_values[i] = if (cu + cd).abs() < f64::EPSILON {
            0.0
        } else {
            cu / (cu + cd)
        };
    }

    Ok(LrsiOutput { values: rsi_values })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_lrsi_default_params() {
        let params = LrsiParams::default();
        assert_eq!(params.alpha, Some(0.2));
    }

    #[test]
    fn test_lrsi_from_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = LrsiInput::from_candles(&candles, LrsiParams::default());
        let output = lrsi(&input);
        assert!(output.is_ok());
        let result = output.unwrap();
        assert_eq!(result.values.len(), candles.close.len());
    }

    #[test]
    fn test_lrsi_from_slices() {
        let high = [1.0, 1.0, 1.0, 1.0, 1.0];
        let low = [1.0, 1.0, 1.0, 1.0, 1.0];
        let params = LrsiParams::default();
        let input = LrsiInput::from_slices(&high, &low, params);
        let output = lrsi(&input).expect("Failed LRSI from slices");
        assert_eq!(output.values.len(), high.len());
    }

    #[test]
    fn test_lrsi_empty_data() {
        let high: [f64; 0] = [];
        let low: [f64; 0] = [];
        let params = LrsiParams::default();
        let input = LrsiInput::from_slices(&high, &low, params);
        let result = lrsi(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Empty data"));
        }
    }

    #[test]
    fn test_lrsi_invalid_alpha() {
        let high = [1.0, 2.0];
        let low = [1.0, 2.0];
        let params = LrsiParams { alpha: Some(1.2) };
        let input = LrsiInput::from_slices(&high, &low, params);
        let result = lrsi(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Invalid alpha"));
        }
    }

    #[test]
    fn test_lrsi_all_nan() {
        let high = [f64::NAN, f64::NAN, f64::NAN];
        let low = [f64::NAN, f64::NAN, f64::NAN];
        let params = LrsiParams::default();
        let input = LrsiInput::from_slices(&high, &low, params);
        let result = lrsi(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("All values are NaN"));
        }
    }

    #[test]
    fn test_lrsi_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = LrsiInput::from_candles(&candles, LrsiParams::default());
        let lrsi_result = lrsi(&input).expect("Failed to calculate LRSI");
        assert_eq!(
            lrsi_result.values.len(),
            candles.close.len(),
            "LRSI length mismatch"
        );

        let expected_last_five_lrsi = [0.0, 0.0, 0.0, 0.0, 0.0];
        assert!(
            lrsi_result.values.len() >= 5,
            "LRSI length is too short for comparison"
        );
        let start_index = lrsi_result.values.len() - 5;
        let result_last_five_lrsi = &lrsi_result.values[start_index..];
        for (i, &value) in result_last_five_lrsi.iter().enumerate() {
            let expected_value = expected_last_five_lrsi[i];
            assert!(
                (value - expected_value).abs() < 1e-9,
                "LRSI mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }

        let default_input = LrsiInput::with_default_candles(&candles);
        let default_lrsi_result = lrsi(&default_input).expect("Failed to calculate LRSI defaults");
        assert_eq!(default_lrsi_result.values.len(), candles.close.len());
    }
}
