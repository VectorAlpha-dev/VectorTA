use crate::indicators::ema::{ema, EmaError, EmaInput, EmaParams};
use crate::indicators::moving_averages::ma::{ma, MaData};
use crate::indicators::utility_functions::{max_rolling, min_rolling};
/// # Schaff Trend Cycle (STC)
///
/// The STC is an oscillator indicator that uses a MACD calculation (two moving averages),
/// then applies Stochastics computations twice, and finally smoothing with EMA.
///
/// ## Parameters
/// - **fast_period**: The period for the fast MA. Defaults to 23.
/// - **slow_period**: The period for the slow MA. Defaults to 50.
/// - **k_period**: The period used in rolling min/max calculations for %K steps. Defaults to 10.
/// - **d_period**: The period used for EMA smoothing of %K (and again for %D steps). Defaults to 3.
/// - **fast_ma_type**: The moving average type for the fast MA. Defaults to "ema".
/// - **slow_ma_type**: The moving average type for the slow MA. Defaults to "ema".
///
/// ## Errors
/// - **EmptyData**: stc: Input data slice is empty.
/// - **AllValuesNaN**: stc: All input data values are `NaN`.
/// - **NotEnoughValidData**: stc: Fewer than needed valid data points remain after the first valid index.
/// - **MinRollingError** / **MaxRollingError**: Errors returned by rolling min/max calculations.
/// - **MaError**: Errors returned by the generic `ma` function.
///
/// ## Returns
/// - **`Ok(StcOutput)`** on success, containing a `Vec<f64>` matching the input length,
///   with leading `NaN`s until calculations can begin.
/// - **`Err(StcError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};
use std::error::Error;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum StcData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct StcOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct StcParams {
    pub fast_period: Option<usize>,
    pub slow_period: Option<usize>,
    pub k_period: Option<usize>,
    pub d_period: Option<usize>,
    pub fast_ma_type: Option<String>,
    pub slow_ma_type: Option<String>,
}

impl Default for StcParams {
    fn default() -> Self {
        Self {
            fast_period: Some(23),
            slow_period: Some(50),
            k_period: Some(10),
            d_period: Some(3),
            fast_ma_type: Some("ema".to_string()),
            slow_ma_type: Some("ema".to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct StcInput<'a> {
    pub data: StcData<'a>,
    pub params: StcParams,
}

impl<'a> StcInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: StcParams) -> Self {
        Self {
            data: StcData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: StcParams) -> Self {
        Self {
            data: StcData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: StcData::Candles {
                candles,
                source: "close",
            },
            params: StcParams::default(),
        }
    }

    fn get_fast_period(&self) -> usize {
        self.params
            .fast_period
            .unwrap_or_else(|| StcParams::default().fast_period.unwrap())
    }

    fn get_slow_period(&self) -> usize {
        self.params
            .slow_period
            .unwrap_or_else(|| StcParams::default().slow_period.unwrap())
    }

    fn get_k_period(&self) -> usize {
        self.params
            .k_period
            .unwrap_or_else(|| StcParams::default().k_period.unwrap())
    }

    fn get_d_period(&self) -> usize {
        self.params
            .d_period
            .unwrap_or_else(|| StcParams::default().d_period.unwrap())
    }

    fn get_fast_ma_type(&self) -> &str {
        match &self.params.fast_ma_type {
            Some(s) => s,
            None => "ema",
        }
    }

    fn get_slow_ma_type(&self) -> &str {
        match &self.params.slow_ma_type {
            Some(s) => s,
            None => "ema",
        }
    }
}

#[derive(Debug, Error)]
pub enum StcError {
    #[error("stc: Empty data provided.")]
    EmptyData,
    #[error("stc: All values are NaN.")]
    AllValuesNaN,
    #[error("stc: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("stc: rolling error: {0}")]
    RollingError(#[from] crate::indicators::utility_functions::RollingError),
    #[error("stc: MA error: {0}")]
    MaError(#[from] Box<dyn Error>),
    #[error("stc: EMA error: {0}")]
    EmaError(#[from] EmaError),
}

#[inline]
pub fn stc(input: &StcInput) -> Result<StcOutput, StcError> {
    let data: &[f64] = match &input.data {
        StcData::Candles { candles, source } => source_type(candles, source),
        StcData::Slice(slice) => slice,
    };

    if data.is_empty() {
        return Err(StcError::EmptyData);
    }

    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(StcError::AllValuesNaN),
    };

    let fast_period = input.get_fast_period();
    let slow_period = input.get_slow_period();
    let k_period = input.get_k_period();
    let d_period = input.get_d_period();
    let needed = fast_period.max(slow_period).max(k_period).max(d_period);
    let valid_len = data.len() - first_valid_idx;

    if valid_len < needed {
        return Err(StcError::NotEnoughValidData {
            needed,
            valid: valid_len,
        });
    }

    let fast_ma_type = input.get_fast_ma_type();
    let slow_ma_type = input.get_slow_ma_type();
    let fast_ma = ma(
        fast_ma_type,
        MaData::Slice(&data[first_valid_idx..]),
        fast_period,
    )?;
    let slow_ma = ma(
        slow_ma_type,
        MaData::Slice(&data[first_valid_idx..]),
        slow_period,
    )?;

    let macd: Vec<f64> = fast_ma
        .iter()
        .zip(slow_ma.iter())
        .map(|(f, s)| f - s)
        .collect();

    let macd_min = min_rolling(&macd, k_period)?;
    let macd_max = max_rolling(&macd, k_period)?;
    let mut stok = vec![f64::NAN; macd.len()];
    for i in 0..macd.len() {
        let range = macd_max[i] - macd_min[i];
        if range.abs() > f64::EPSILON && !range.is_nan() {
            stok[i] = (macd[i] - macd_min[i]) / range * 100.0;
        }
    }

    let d_ema_input = EmaInput::from_slice(
        &stok,
        EmaParams {
            period: Some(d_period),
        },
    );
    let d_ema_output = ema(&d_ema_input)?;
    let d_vals = d_ema_output.values;

    let d_min = min_rolling(&d_vals, k_period)?;
    let d_max = max_rolling(&d_vals, k_period)?;
    let mut kd = vec![f64::NAN; d_vals.len()];
    for i in 0..d_vals.len() {
        let range = d_max[i] - d_min[i];
        if range.abs() > f64::EPSILON && !range.is_nan() {
            kd[i] = (d_vals[i] - d_min[i]) / range * 100.0;
        }
    }

    let kd_ema_input = EmaInput::from_slice(
        &kd,
        EmaParams {
            period: Some(d_period),
        },
    );
    let kd_ema_output = ema(&kd_ema_input)?;
    let final_stc = kd_ema_output.values;

    let mut stc_values = vec![f64::NAN; data.len()];
    for (i, &val) in final_stc.iter().enumerate() {
        stc_values[first_valid_idx + i] = val;
    }

    Ok(StcOutput { values: stc_values })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_stc_default_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = StcInput::with_default_candles(&candles);
        let output = stc(&input).expect("STC calculation failed with default params");
        assert_eq!(output.values.len(), candles.close.len());
    }

    #[test]
    fn test_stc_last_five_values() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let params = StcParams::default();
        let input = StcInput::from_candles(&candles, "close", params);
        let result = stc(&input).expect("Failed to calculate STC");

        let expected = [
            0.21394384188858884,
            0.10697192094429442,
            0.05348596047214721,
            50.02674298023607,
            49.98686202668157,
        ];

        let n = result.values.len();
        assert!(n >= 5);
        for (i, &exp) in expected.iter().enumerate() {
            let val = result.values[n - 5 + i];
            assert!(
                (val - exp).abs() < 1e-5,
                "Expected {}, got {} at index {}",
                exp,
                val,
                n - 5 + i
            );
        }
    }

    #[test]
    fn test_stc_with_slice_data() {
        let slice_data = [10.0, 11.0, 12.0, 13.0, 14.0];
        let params = StcParams {
            fast_period: Some(2),
            slow_period: Some(3),
            k_period: Some(2),
            d_period: Some(1),
            fast_ma_type: Some("ema".to_string()),
            slow_ma_type: Some("ema".to_string()),
        };

        let input = StcInput::from_slice(&slice_data, params);
        let result = stc(&input).unwrap_or_else(|err| {
            panic!("Failed to compute STC for small slice: {}", err);
        });
        assert_eq!(result.values.len(), slice_data.len());
    }

    #[test]
    fn test_stc_empty_data() {
        let data: [f64; 0] = [];
        let input = StcInput::from_slice(&data, StcParams::default());
        let result = stc(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_stc_all_nan_data() {
        let data = [f64::NAN, f64::NAN, f64::NAN];
        let input = StcInput::from_slice(&data, StcParams::default());
        let result = stc(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_stc_not_enough_valid_data() {
        let data = [f64::NAN, 2.0, 3.0];
        let params = StcParams {
            fast_period: Some(5),
            ..Default::default()
        };
        let input = StcInput::from_slice(&data, params);
        let result = stc(&input);
        assert!(result.is_err());
    }
}
