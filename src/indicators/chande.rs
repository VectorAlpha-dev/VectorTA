/// # Chande Exits (Chandelier Exits)
///
/// Chande Exits (also called Chandelier Exits) is a volatility-based indicator that calculates
/// an exit point using a rolling maximum or minimum, adjusted by an ATR multiplier.
///
/// For a `long` direction:
/// \[ ChandeLong[i] = \max_{(i - period + 1 .. i)}(\text{high}[j]) - ATR[i] * mult \]
///
/// For a `short` direction:
/// \[ ChandeShort[i] = \min_{(i - period + 1 .. i)}(\text{low}[j]) + ATR[i] * mult \]
///
/// Typically, `period` defaults to 22 and `mult` defaults to 3.0, following
/// common usage. Leading values where `i < period - 1` cannot be computed and are set to `NaN`.
///
/// ## Parameters
/// - **period**: Window size for both ATR and rolling max/min. Defaults to 22.
/// - **mult**: ATR multiplier, typically 3.0. Defaults to 3.0.
/// - **direction**: `"long"` or `"short"`. Defaults to `"long"`.
///
/// ## Errors
/// - **EmptyData**: chande: No candle data provided.
/// - **InvalidPeriod**: chande: `period` is 0 or exceeds the available data length.
/// - **NotEnoughData**: chande: Fewer than `period` data points are available.
/// - **InvalidDirection**: chande: `direction` must be `"long"` or `"short"`.
///
/// ## Returns
/// - **`Ok(ChandeOutput)`** on success, containing a `Vec<f64>` of length matching the input,
///   with `NaN` values for indices where insufficient data is available.
/// - **`Err(ChandeError)`** otherwise.
use crate::utilities::data_loader::Candles;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum ChandeData<'a> {
    Candles(&'a Candles),
}

#[derive(Debug, Clone)]
pub struct ChandeOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct ChandeParams {
    pub period: Option<usize>,
    pub mult: Option<f64>,
    pub direction: Option<String>,
}

impl Default for ChandeParams {
    fn default() -> Self {
        Self {
            period: Some(22),
            mult: Some(3.0),
            direction: Some("long".into()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ChandeInput<'a> {
    pub data: ChandeData<'a>,
    pub params: ChandeParams,
}

impl<'a> ChandeInput<'a> {
    pub fn from_candles(candles: &'a Candles, params: ChandeParams) -> Self {
        Self {
            data: ChandeData::Candles(candles),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: ChandeData::Candles(candles),
            params: ChandeParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| ChandeParams::default().period.unwrap())
    }

    pub fn get_mult(&self) -> f64 {
        self.params
            .mult
            .unwrap_or_else(|| ChandeParams::default().mult.unwrap())
    }

    pub fn get_direction(&self) -> &str {
        match &self.params.direction {
            Some(dir) => dir,
            None => "long",
        }
    }
}

#[derive(Debug, Error)]
pub enum ChandeError {
    #[error("chande: No candle data provided.")]
    EmptyData,
    #[error("chande: Invalid period: period={period}, data_len={data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("chande: Not enough data: needed={needed}, available={available}")]
    NotEnoughData { needed: usize, available: usize },
    #[error("chande: Invalid direction: must be 'long' or 'short'.")]
    InvalidDirection,
    #[error("chande: ATR error: {0}")]
    AtrError(#[from] AtrError),
    #[error("chande: Empty data.")]
    AllValuesNaN,
}

#[derive(Debug, Error)]
pub enum AtrError {
    #[error("atr: Invalid length: {length}")]
    InvalidLength { length: usize },
    #[error("atr: Not enough data: length={length}, data_len={data_len}")]
    NotEnoughData { length: usize, data_len: usize },
    #[error("atr: No candles available.")]
    NoCandlesAvailable,
}

#[derive(Debug, Clone)]
pub struct AtrOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub enum AtrData<'a> {
    Candles { candles: &'a Candles },
}

#[derive(Debug, Clone)]
pub struct AtrInput<'a> {
    pub data: AtrData<'a>,
    pub length: usize,
}

impl<'a> AtrInput<'a> {
    pub fn new(candles: &'a Candles, length: usize) -> Self {
        Self {
            data: AtrData::Candles { candles },
            length,
        }
    }

    pub fn get_length(&self) -> usize {
        self.length
    }
}

#[inline]
pub fn chande(input: &ChandeInput) -> Result<ChandeOutput, ChandeError> {
    let ChandeData::Candles(candles) = &input.data;
    let high = candles
        .select_candle_field("high")
        .map_err(|_| ChandeError::EmptyData)?;
    let low = candles
        .select_candle_field("low")
        .map_err(|_| ChandeError::EmptyData)?;
    let close = candles
        .select_candle_field("close")
        .map_err(|_| ChandeError::EmptyData)?;
    let len = high.len();
    if len == 0 {
        return Err(ChandeError::EmptyData);
    }
    let period = input.get_period();
    let mult = input.get_mult();
    let dir = input.get_direction().to_lowercase();
    if dir != "long" && dir != "short" {
        return Err(ChandeError::InvalidDirection);
    }
    if period == 0 || period > len {
        return Err(ChandeError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if period > len {
        return Err(ChandeError::NotEnoughData {
            needed: period,
            available: len,
        });
    }
    let first_valid = match close.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(ChandeError::AllValuesNaN),
    };
    let mut vals = vec![f64::NAN; len];
    let mut atr = vec![f64::NAN; len];
    let alpha = 1.0 / period as f64;
    let mut sum_tr = 0.0;
    let mut rma = f64::NAN;
    for i in first_valid..len {
        let tr = if i == first_valid {
            high[i] - low[i]
        } else {
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            hl.max(hc).max(lc)
        };
        if i < first_valid + period {
            sum_tr += tr;
            if i == first_valid + period - 1 {
                rma = sum_tr / period as f64;
                atr[i] = rma;
            }
        } else {
            rma += alpha * (tr - rma);
            atr[i] = rma;
        }
        if i >= first_valid + period - 1 {
            let a = atr[i];
            if !a.is_nan() {
                let start = i + 1 - period;
                if dir == "long" {
                    let mut m = f64::MIN;
                    for j in start..=i {
                        if high[j] > m {
                            m = high[j];
                        }
                    }
                    vals[i] = m - a * mult;
                } else {
                    let mut m = f64::MAX;
                    for j in start..=i {
                        if low[j] < m {
                            m = low[j];
                        }
                    }
                    vals[i] = m + a * mult;
                }
            }
        }
    }
    Ok(ChandeOutput { values: vals })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_chande_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = ChandeParams {
            period: None,
            mult: None,
            direction: None,
        };
        let input_default = ChandeInput::from_candles(&candles, default_params);
        let output_default =
            chande(&input_default).expect("Failed Chande Exits with default params");
        assert_eq!(output_default.values.len(), candles.close.len());

        let custom_params_long = ChandeParams {
            period: Some(14),
            mult: Some(2.0),
            direction: Some("long".to_string()),
        };
        let input_custom_long = ChandeInput::from_candles(&candles, custom_params_long);
        let output_custom_long =
            chande(&input_custom_long).expect("Failed Chande Exits (long) with custom params");
        assert_eq!(output_custom_long.values.len(), candles.close.len());

        let custom_params_short = ChandeParams {
            period: Some(10),
            mult: Some(3.0),
            direction: Some("short".to_string()),
        };
        let input_custom_short = ChandeInput::from_candles(&candles, custom_params_short);
        let output_custom_short =
            chande(&input_custom_short).expect("Failed Chande Exits (short) with custom params");
        assert_eq!(output_custom_short.values.len(), candles.close.len());
    }

    #[test]
    fn test_chande_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = &candles.close;

        let params = ChandeParams {
            period: Some(22),
            mult: Some(3.0),
            direction: Some("long".into()),
        };
        let input = ChandeInput::from_candles(&candles, params);
        let chande_result = chande(&input).expect("Failed to calculate Chande Exits");

        assert_eq!(
            chande_result.values.len(),
            close_prices.len(),
            "Chande Exits length mismatch"
        );

        let expected_last_five = [
            59444.14115983658,
            58576.49837984401,
            58649.1120898511,
            58724.56154031242,
            58713.39965211639,
        ];

        assert!(
            chande_result.values.len() >= 5,
            "Not enough data to check final five values"
        );
        let start_idx = chande_result.values.len() - 5;
        let actual_last_five = &chande_result.values[start_idx..];
        for (i, &val) in actual_last_five.iter().enumerate() {
            let exp = expected_last_five[i];
            assert!(
                (val - exp).abs() < 1e-4,
                "Chande Exits mismatch at index {}: expected {}, got {}",
                i,
                exp,
                val
            );
        }

        let period = 22;
        for i in 0..(period - 1) {
            assert!(
                chande_result.values[i].is_nan(),
                "Expected leading NaN at index {}",
                i
            );
        }

        let default_input = ChandeInput::with_default_candles(&candles);
        let default_output = chande(&default_input).expect("Failed Chande Exits default");
        assert_eq!(
            default_output.values.len(),
            close_prices.len(),
            "Default input mismatch"
        );
    }

    #[test]
    fn test_chande_params_with_default_params() {
        let default_params = ChandeParams::default();
        assert_eq!(
            default_params.period,
            Some(22),
            "Expected default period=22"
        );
        assert_eq!(default_params.mult, Some(3.0), "Expected default mult=3.0");
        assert_eq!(
            default_params.direction,
            Some("long".into()),
            "Expected default direction=long"
        );
    }

    #[test]
    fn test_chande_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = ChandeInput::with_default_candles(&candles);
        match input.data {
            ChandeData::Candles(_) => {
                assert_eq!(
                    input.get_direction(),
                    "long",
                    "Expected default direction to be 'long'"
                );
            }
        }
    }

    #[test]
    fn test_chande_with_zero_period() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let params = ChandeParams {
            period: Some(0),
            mult: Some(3.0),
            direction: Some("long".into()),
        };
        let input = ChandeInput::from_candles(&candles, params);

        let result = chande(&input);
        assert!(result.is_err(), "Expected an error for zero period");
        if let Err(e) = result {
            assert!(
                e.to_string().contains("Invalid period"),
                "Expected 'Invalid period' error, got: {}",
                e
            );
        }
    }

    #[test]
    fn test_chande_with_period_exceeding_data_length() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let params = ChandeParams {
            period: Some(99999),
            mult: Some(3.0),
            direction: Some("long".into()),
        };
        let input = ChandeInput::from_candles(&candles, params);

        let result = chande(&input);
        assert!(result.is_err(), "Expected error for period>data.len()");
    }

    #[test]
    fn test_chande_with_bad_direction() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let params = ChandeParams {
            period: Some(22),
            mult: Some(3.0),
            direction: Some("neither".into()),
        };
        let input = ChandeInput::from_candles(&candles, params);

        let result = chande(&input);
        assert!(result.is_err(), "Expected InvalidDirection error");
        if let Err(e) = result {
            assert!(
                e.to_string().contains("Invalid direction"),
                "Expected 'Invalid direction' error, got: {}",
                e
            );
        }
    }

    #[test]
    fn test_chande_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let params = ChandeParams {
            period: Some(22),
            mult: Some(3.0),
            direction: Some("long".into()),
        };
        let input = ChandeInput::from_candles(&candles, params);
        let chande_result = chande(&input).expect("Failed to calculate Chande Exits for NaN check");

        assert_eq!(chande_result.values.len(), candles.close.len());

        if chande_result.values.len() > 240 {
            for i in 240..chande_result.values.len() {
                assert!(
                    !chande_result.values[i].is_nan(),
                    "Expected no NaN after index 240, found NaN at index {}",
                    i
                );
            }
        }
    }
}
