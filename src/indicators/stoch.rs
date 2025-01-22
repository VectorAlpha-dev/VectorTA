use crate::indicators::moving_averages::ma::{ma, MaData};
use crate::indicators::utility_functions::{max_rolling, min_rolling};
/// # Stochastic Oscillator (Stoch)
///
/// A momentum indicator comparing a particular closing price to a range of prices over a certain period.
/// The fast %K value is calculated by:  
/// \[ 100 * (CurrentClose - LowestLow) / (HighestHigh - LowestLow) \]  
/// Then two moving averages are applied to derive the slow %K (`k`) and slow %D (`d`).
///
/// ## Parameters
/// - **fastk_period**: The period for the highest high and lowest low. Defaults to 14.
/// - **slowk_period**: The period for the moving average of fast %K. Defaults to 3.
/// - **slowk_ma_type**: The MA type for slow %K, e.g. `"sma"`, `"ema"`, etc. Defaults to `"sma"`.
/// - **slowd_period**: The period for the moving average of slow %K. Defaults to 3.
/// - **slowd_ma_type**: The MA type for slow %D, e.g. `"sma"`, `"ema"`, etc. Defaults to `"sma"`.
///
/// ## Errors
/// - **EmptyData**: stoch: Input data slices (high, low, close) are empty.
/// - **MismatchedLength**: stoch: Input slices (high, low, close) have different lengths.
/// - **InvalidPeriod**: stoch: Period is zero or exceeds the data length.
/// - **NotEnoughValidData**: stoch: Fewer valid (non-`NaN`) data points remain after the first valid index.
/// - **AllValuesNaN**: stoch: All input data values are `NaN`.
///
/// ## Returns
/// - **`Ok(StochOutput)`** on success, containing vectors `k` and `d` (both matching the input length),
///   with leading `NaN`s until each component can be calculated.
/// - **`Err(StochError)`** otherwise.
use crate::utilities::data_loader::Candles;
use std::error::Error;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum StochData<'a> {
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
pub struct StochOutput {
    pub k: Vec<f64>,
    pub d: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct StochParams {
    pub fastk_period: Option<usize>,
    pub slowk_period: Option<usize>,
    pub slowk_ma_type: Option<String>,
    pub slowd_period: Option<usize>,
    pub slowd_ma_type: Option<String>,
}

impl Default for StochParams {
    fn default() -> Self {
        Self {
            fastk_period: Some(14),
            slowk_period: Some(3),
            slowk_ma_type: Some("sma".to_string()),
            slowd_period: Some(3),
            slowd_ma_type: Some("sma".to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct StochInput<'a> {
    pub data: StochData<'a>,
    pub params: StochParams,
}

impl<'a> StochInput<'a> {
    pub fn from_candles(candles: &'a Candles, params: StochParams) -> Self {
        Self {
            data: StochData::Candles { candles },
            params,
        }
    }

    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: StochParams,
    ) -> Self {
        Self {
            data: StochData::Slices { high, low, close },
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: StochData::Candles { candles },
            params: StochParams::default(),
        }
    }

    pub fn get_fastk_period(&self) -> usize {
        self.params
            .fastk_period
            .unwrap_or_else(|| StochParams::default().fastk_period.unwrap())
    }

    pub fn get_slowk_period(&self) -> usize {
        self.params
            .slowk_period
            .unwrap_or_else(|| StochParams::default().slowk_period.unwrap())
    }

    pub fn get_slowk_ma_type(&self) -> String {
        self.params
            .slowk_ma_type
            .clone()
            .unwrap_or_else(|| StochParams::default().slowk_ma_type.unwrap())
    }

    pub fn get_slowd_period(&self) -> usize {
        self.params
            .slowd_period
            .unwrap_or_else(|| StochParams::default().slowd_period.unwrap())
    }

    pub fn get_slowd_ma_type(&self) -> String {
        self.params
            .slowd_ma_type
            .clone()
            .unwrap_or_else(|| StochParams::default().slowd_ma_type.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum StochError {
    #[error("stoch: Empty data provided.")]
    EmptyData,
    #[error("stoch: Mismatched length of input data (high, low, close).")]
    MismatchedLength,
    #[error("stoch: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("stoch: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("stoch: All values are NaN.")]
    AllValuesNaN,
    #[error("stoch: {0}")]
    Other(String),
}

#[inline]
pub fn stoch(input: &StochInput) -> Result<StochOutput, StochError> {
    let (high, low, close) = match &input.data {
        StochData::Candles { candles } => {
            let high = candles
                .select_candle_field("high")
                .map_err(|e| StochError::Other(e.to_string()))?;
            let low = candles
                .select_candle_field("low")
                .map_err(|e| StochError::Other(e.to_string()))?;
            let close = candles
                .select_candle_field("close")
                .map_err(|e| StochError::Other(e.to_string()))?;
            (high, low, close)
        }
        StochData::Slices { high, low, close } => (*high, *low, *close),
    };

    let data_len = high.len();
    if data_len == 0 || low.is_empty() || close.is_empty() {
        return Err(StochError::EmptyData);
    }
    if data_len != low.len() || data_len != close.len() {
        return Err(StochError::MismatchedLength);
    }

    let fastk_period = input.get_fastk_period();
    if fastk_period == 0 || fastk_period > data_len {
        return Err(StochError::InvalidPeriod {
            period: fastk_period,
            data_len,
        });
    }
    let slowk_period = input.get_slowk_period();
    if slowk_period == 0 || slowk_period > data_len {
        return Err(StochError::InvalidPeriod {
            period: slowk_period,
            data_len,
        });
    }
    let slowd_period = input.get_slowd_period();
    if slowd_period == 0 || slowd_period > data_len {
        return Err(StochError::InvalidPeriod {
            period: slowd_period,
            data_len,
        });
    }

    let first_valid_idx = {
        let mut idx = None;
        for i in 0..data_len {
            if !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan() {
                idx = Some(i);
                break;
            }
        }
        match idx {
            Some(i) => i,
            None => return Err(StochError::AllValuesNaN),
        }
    };

    if (data_len - first_valid_idx) < fastk_period {
        return Err(StochError::NotEnoughValidData {
            needed: fastk_period,
            valid: data_len - first_valid_idx,
        });
    }

    let mut hh = vec![f64::NAN; data_len];
    let mut ll = vec![f64::NAN; data_len];
    let max_vals = max_rolling(&high[first_valid_idx..], fastk_period)
        .map_err(|e| StochError::Other(e.to_string()))?;
    let min_vals = min_rolling(&low[first_valid_idx..], fastk_period)
        .map_err(|e| StochError::Other(e.to_string()))?;

    for (i, &val) in max_vals.iter().enumerate() {
        hh[i + first_valid_idx] = val;
    }
    for (i, &val) in min_vals.iter().enumerate() {
        ll[i + first_valid_idx] = val;
    }

    let mut stoch_vals = vec![f64::NAN; data_len];
    for i in (first_valid_idx + fastk_period - 1)..data_len {
        let denom = hh[i] - ll[i];
        if denom.abs() < f64::EPSILON {
            stoch_vals[i] = 50.0;
        } else {
            stoch_vals[i] = 100.0 * (close[i] - ll[i]) / denom;
        }
    }

    let slowk_ma_type = input.get_slowk_ma_type();
    let slowd_ma_type = input.get_slowd_ma_type();

    let k_result = ma(&slowk_ma_type, MaData::Slice(&stoch_vals), slowk_period)
        .map_err(|e| StochError::Other(e.to_string()))?;

    let d_result = ma(&slowd_ma_type, MaData::Slice(&k_result), slowd_period)
        .map_err(|e| StochError::Other(e.to_string()))?;

    Ok(StochOutput {
        k: k_result,
        d: d_result,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_stoch_default_params_on_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = StochInput::with_default_candles(&candles);
        let output = stoch(&input).expect("Failed Stoch with default params");
        assert_eq!(output.k.len(), candles.close.len());
        assert_eq!(output.d.len(), candles.close.len());
    }

    #[test]
    fn test_stoch_custom_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = StochParams {
            fastk_period: Some(10),
            slowk_period: Some(4),
            slowk_ma_type: Some("ema".to_string()),
            slowd_period: Some(4),
            slowd_ma_type: Some("sma".to_string()),
        };
        let input = StochInput::from_candles(&candles, params);
        let output = stoch(&input).expect("Failed Stoch with custom params");
        assert_eq!(output.k.len(), candles.close.len());
        assert_eq!(output.d.len(), candles.close.len());
    }

    #[test]
    fn test_stoch_values_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = StochParams::default();
        let input = StochInput::from_candles(&candles, params);
        let result = stoch(&input).expect("Failed to calculate Stoch");

        assert_eq!(result.k.len(), candles.close.len());
        assert_eq!(result.d.len(), candles.close.len());

        let last_five_k = [
            42.51122827572717,
            40.13864479593807,
            37.853934778363374,
            37.337021714266086,
            36.26053890551548,
        ];
        let last_five_d = [
            41.36561869426493,
            41.7691857059163,
            40.16793595000925,
            38.44320042952222,
            37.15049846604803,
        ];
        assert!(result.k.len() >= 5 && result.d.len() >= 5);
        let k_slice = &result.k[result.k.len() - 5..];
        let d_slice = &result.d[result.d.len() - 5..];
        for i in 0..5 {
            let diff_k = (k_slice[i] - last_five_k[i]).abs();
            let diff_d = (d_slice[i] - last_five_d[i]).abs();
            assert!(
                diff_k < 1e-6,
                "Mismatch in K at {}: got {}, expected {}",
                i,
                k_slice[i],
                last_five_k[i]
            );
            assert!(
                diff_d < 1e-6,
                "Mismatch in D at {}: got {}, expected {}",
                i,
                d_slice[i],
                last_five_d[i]
            );
        }
    }

    #[test]
    fn test_stoch_zero_period() {
        let high = [10.0, 11.0, 12.0];
        let low = [9.0, 9.5, 10.5];
        let close = [9.5, 10.6, 11.5];
        let params = StochParams {
            fastk_period: Some(0),
            ..Default::default()
        };
        let input = StochInput::from_slices(&high, &low, &close, params);
        let result = stoch(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_stoch_period_exceeding_data_length() {
        let high = [10.0, 11.0, 12.0];
        let low = [9.0, 9.5, 10.5];
        let close = [9.5, 10.6, 11.5];
        let params = StochParams {
            fastk_period: Some(10),
            ..Default::default()
        };
        let input = StochInput::from_slices(&high, &low, &close, params);
        let result = stoch(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_stoch_all_nan() {
        let nan_data = [f64::NAN, f64::NAN, f64::NAN];
        let params = StochParams::default();
        let input = StochInput::from_slices(&nan_data, &nan_data, &nan_data, params);
        let result = stoch(&input);
        assert!(result.is_err());
    }
}
