/// # Dynamic Trend Index (DTI) by William Blau
///
/// A momentum-based indicator that computes the difference between upward and downward
/// price movements, then applies a triple EMA smoothing to that difference (and its
/// absolute value), producing a value typically scaled between `-100` and `100`.
///
/// ## Parameters
/// - **r**: The period of the first EMA smoothing. Defaults to 14.
/// - **s**: The period of the second EMA smoothing. Defaults to 10.
/// - **u**: The period of the third EMA smoothing. Defaults to 5.
///
/// ## Errors
/// - **EmptyData**: dti: Input data slice is empty.
/// - **CandleFieldError**: dti: Error reading candle data fields.
/// - **InvalidPeriod**: dti: One or more of the EMA periods is zero or exceeds the data length.
/// - **NotEnoughValidData**: dti: Fewer valid (non-`NaN`) data points remain after the first
///   valid index than are needed to compute at least one of the EMAs.
/// - **AllValuesNaN**: dti: All input high/low values are `NaN`.
///
/// ## Returns
/// - **`Ok(DtiOutput)`** on success, containing a `Vec<f64>` matching the input length,
///   with leading `NaN`s until the indicator can be fully calculated.
/// - **`Err(DtiError)`** otherwise.
use crate::utilities::data_loader::Candles;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum DtiData<'a> {
    Candles { candles: &'a Candles },
    Slices { high: &'a [f64], low: &'a [f64] },
}

#[derive(Debug, Clone)]
pub struct DtiOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct DtiParams {
    pub r: Option<usize>,
    pub s: Option<usize>,
    pub u: Option<usize>,
}

impl Default for DtiParams {
    fn default() -> Self {
        Self {
            r: Some(14),
            s: Some(10),
            u: Some(5),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DtiInput<'a> {
    pub data: DtiData<'a>,
    pub params: DtiParams,
}

impl<'a> DtiInput<'a> {
    pub fn from_candles(candles: &'a Candles, params: DtiParams) -> Self {
        Self {
            data: DtiData::Candles { candles },
            params,
        }
    }

    pub fn from_slices(high: &'a [f64], low: &'a [f64], params: DtiParams) -> Self {
        Self {
            data: DtiData::Slices { high, low },
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: DtiData::Candles { candles },
            params: DtiParams::default(),
        }
    }

    pub fn get_r(&self) -> usize {
        self.params
            .r
            .unwrap_or_else(|| DtiParams::default().r.unwrap())
    }

    pub fn get_s(&self) -> usize {
        self.params
            .s
            .unwrap_or_else(|| DtiParams::default().s.unwrap())
    }

    pub fn get_u(&self) -> usize {
        self.params
            .u
            .unwrap_or_else(|| DtiParams::default().u.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum DtiError {
    #[error("dti: Empty data provided.")]
    EmptyData,
    #[error("dti: Candle field error: {0}")]
    CandleFieldError(String),
    #[error("dti: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("dti: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("dti: All high/low values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn dti(input: &DtiInput) -> Result<DtiOutput, DtiError> {
    let (high, low) = match &input.data {
        DtiData::Candles { candles } => {
            let high = candles
                .select_candle_field("high")
                .map_err(|e| DtiError::CandleFieldError(e.to_string()))?;
            let low = candles
                .select_candle_field("low")
                .map_err(|e| DtiError::CandleFieldError(e.to_string()))?;
            (high, low)
        }
        DtiData::Slices { high, low } => (*high, *low),
    };

    if high.is_empty() || low.is_empty() {
        return Err(DtiError::EmptyData);
    }
    let len = high.len();
    if low.len() != len {
        return Err(DtiError::EmptyData);
    }

    let first_valid_idx = match (0..len).find(|&i| !high[i].is_nan() && !low[i].is_nan()) {
        Some(idx) => idx,
        None => return Err(DtiError::AllValuesNaN),
    };

    let r = input.get_r();
    let s = input.get_s();
    let u = input.get_u();

    for &period in &[r, s, u] {
        if period == 0 || period > len {
            return Err(DtiError::InvalidPeriod {
                period,
                data_len: len,
            });
        }
        if (len - first_valid_idx) < period {
            return Err(DtiError::NotEnoughValidData {
                needed: period,
                valid: len - first_valid_idx,
            });
        }
    }

    let alpha_r = 2.0 / (r as f64 + 1.0);
    let alpha_s = 2.0 / (s as f64 + 1.0);
    let alpha_u = 2.0 / (u as f64 + 1.0);

    let alpha_r_1 = 1.0 - alpha_r;
    let alpha_s_1 = 1.0 - alpha_s;
    let alpha_u_1 = 1.0 - alpha_u;

    let mut dti_values = vec![f64::NAN; len];

    let mut e0_r = 0.0;
    let mut e0_s = 0.0;
    let mut e0_u = 0.0;
    let mut e1_r = 0.0;
    let mut e1_s = 0.0;
    let mut e1_u = 0.0;

    dti_values[first_valid_idx] = f64::NAN;

    for i in (first_valid_idx + 1)..len {
        let dh = high[i] - high[i - 1];
        let dl = low[i] - low[i - 1];
        let x_hmu = if dh > 0.0 { dh } else { 0.0 };
        let x_lmd = if dl < 0.0 { -dl } else { 0.0 };

        let x_price = x_hmu - x_lmd;
        let x_price_abs = x_price.abs();

        e0_r = alpha_r * x_price + alpha_r_1 * e0_r;
        e0_s = alpha_s * e0_r + alpha_s_1 * e0_s;
        e0_u = alpha_u * e0_s + alpha_u_1 * e0_u;

        e1_r = alpha_r * x_price_abs + alpha_r_1 * e1_r;
        e1_s = alpha_s * e1_r + alpha_s_1 * e1_s;
        e1_u = alpha_u * e1_s + alpha_u_1 * e1_u;

        if !e1_u.is_nan() && e1_u != 0.0 {
            dti_values[i] = 100.0 * e0_u / e1_u;
        } else {
            dti_values[i] = 0.0;
        }
    }

    Ok(DtiOutput { values: dti_values })
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_dti_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = DtiInput::with_default_candles(&candles);
        let result = dti(&input).expect("DTI calculation failed with default params");
        assert_eq!(result.values.len(), candles.close.len());
    }

    #[test]
    fn test_dti_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let default_params = DtiParams {
            r: None,
            s: None,
            u: None,
        };
        let input_default = DtiInput::from_candles(&candles, default_params);
        let output_default = dti(&input_default).expect("Failed DTI with partial params");
        assert_eq!(output_default.values.len(), candles.close.len());
    }

    #[test]
    fn test_dti_custom_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params_custom = DtiParams {
            r: Some(10),
            s: Some(6),
            u: Some(3),
        };
        let input_custom = DtiInput::from_candles(&candles, params_custom);
        let output_custom = dti(&input_custom).expect("Failed DTI with custom params");
        assert_eq!(output_custom.values.len(), candles.close.len());
    }

    #[test]
    fn test_dti_zero_period() {
        let high = [10.0, 11.0, 12.0];
        let low = [9.0, 10.0, 11.0];
        let params = DtiParams {
            r: Some(0),
            s: Some(10),
            u: Some(5),
        };
        let input = DtiInput::from_slices(&high, &low, params);
        let result = dti(&input);
        assert!(result.is_err(), "Expected error for zero period");
    }

    #[test]
    fn test_dti_period_exceeds_length() {
        let high = [10.0, 11.0];
        let low = [9.0, 10.0];
        let params = DtiParams {
            r: Some(14),
            s: Some(10),
            u: Some(5),
        };
        let input = DtiInput::from_slices(&high, &low, params);
        let result = dti(&input);
        assert!(
            result.is_err(),
            "Expected error for period that exceeds data length"
        );
    }

    #[test]
    fn test_dti_all_nan() {
        let high = [f64::NAN, f64::NAN, f64::NAN];
        let low = [f64::NAN, f64::NAN, f64::NAN];
        let params = DtiParams::default();
        let input = DtiInput::from_slices(&high, &low, params);
        let result = dti(&input);
        assert!(result.is_err(), "Expected AllValuesNaN error");
    }

    #[test]
    fn test_dti_empty_data() {
        let high: [f64; 0] = [];
        let low: [f64; 0] = [];
        let params = DtiParams::default();
        let input = DtiInput::from_slices(&high, &low, params);
        let result = dti(&input);
        assert!(result.is_err(), "Expected EmptyData error");
    }

    #[test]
    fn test_dti_value_check_mocked() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = DtiParams {
            r: Some(14),
            s: Some(10),
            u: Some(5),
        };
        let input = DtiInput::from_candles(&candles, params);
        let result = dti(&input).expect("Failed to calculate DTI");
        assert_eq!(result.values.len(), candles.high.len());

        let expected_last_five = [
            -39.0091620347991,
            -39.75219264093014,
            -40.53941417932286,
            -41.2787749205189,
            -42.93758699380749,
        ];
        let start_idx = result.values.len().saturating_sub(5);
        let got_last_five = &result.values[start_idx..];
        for (i, &val) in got_last_five.iter().enumerate() {
            let exp = expected_last_five[i];
            assert!(
                (val - exp).abs() < 1e-6,
                "Mismatch at index {}: got {}, expected {}",
                i,
                val,
                exp
            );
        }
    }
}
