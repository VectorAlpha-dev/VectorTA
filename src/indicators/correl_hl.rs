/// # Pearson's Correlation Coefficient of High vs. Low (CORREL_HL)
///
/// Measures the strength and direction of the linear relationship between
/// the `high` and `low` fields of candle data over a rolling window of length `period`.
///
/// ## Parameters
/// - **period**: The window size (number of data points). Defaults to 9.
///
/// ## Errors
/// - **EmptyData**: correl_hl: The `high` or `low` arrays are empty.
/// - **InvalidPeriod**: correl_hl: `period` is zero or exceeds the data length.
/// - **DataLengthMismatch**: correl_hl: `high` and `low` arrays must have the same length.
/// - **NotEnoughValidData**: correl_hl: Fewer than `period` valid (non-`NaN`) data points remain
///   after the first valid index.
/// - **AllValuesNaN**: correl_hl: All `high` or `low` values are `NaN`.
///
/// ## Returns
/// - **`Ok(CorrelHlOutput)`** on success, containing a `Vec<f64>` matching the input length,
///   with leading `NaN`s until the rolling window is filled.
/// - **`Err(CorrelHlError)`** otherwise.
use crate::utilities::data_loader::Candles;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum CorrelHlData<'a> {
    Candles { candles: &'a Candles },
    Slices { high: &'a [f64], low: &'a [f64] },
}

#[derive(Debug, Clone)]
pub struct CorrelHlParams {
    pub period: Option<usize>,
}

impl Default for CorrelHlParams {
    fn default() -> Self {
        Self { period: Some(9) }
    }
}

#[derive(Debug, Clone)]
pub struct CorrelHlInput<'a> {
    pub data: CorrelHlData<'a>,
    pub params: CorrelHlParams,
}

impl<'a> CorrelHlInput<'a> {
    pub fn from_candles(candles: &'a Candles, params: CorrelHlParams) -> Self {
        Self {
            data: CorrelHlData::Candles { candles },
            params,
        }
    }

    pub fn from_slices(high: &'a [f64], low: &'a [f64], params: CorrelHlParams) -> Self {
        Self {
            data: CorrelHlData::Slices { high, low },
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: CorrelHlData::Candles { candles },
            params: CorrelHlParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| CorrelHlParams::default().period.unwrap())
    }
}

#[derive(Debug, Clone)]
pub struct CorrelHlOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Error)]
pub enum CorrelHlError {
    #[error("correl_hl: Empty data provided (high or low).")]
    EmptyData,
    #[error("correl_hl: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("correl_hl: Data length mismatch between high and low.")]
    DataLengthMismatch,
    #[error("correl_hl: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("correl_hl: All values are NaN in high or low.")]
    AllValuesNaN,
}

#[inline]
pub fn correl_hl(input: &CorrelHlInput) -> Result<CorrelHlOutput, CorrelHlError> {
    let (high, low) = match &input.data {
        CorrelHlData::Candles { candles } => {
            let high = candles
                .select_candle_field("high")
                .map_err(|_e| CorrelHlError::EmptyData)?;
            let low = candles
                .select_candle_field("low")
                .map_err(|_e| CorrelHlError::EmptyData)?;
            (high, low)
        }
        CorrelHlData::Slices { high, low } => (*high, *low),
    };

    if high.is_empty() || low.is_empty() {
        return Err(CorrelHlError::EmptyData);
    }

    if high.len() != low.len() {
        return Err(CorrelHlError::DataLengthMismatch);
    }

    let period = input.get_period();
    if period == 0 || period > high.len() {
        return Err(CorrelHlError::InvalidPeriod {
            period,
            data_len: high.len(),
        });
    }

    let first_valid_idx = match high
        .iter()
        .zip(low.iter())
        .position(|(&h, &l)| !h.is_nan() && !l.is_nan())
    {
        Some(idx) => idx,
        None => return Err(CorrelHlError::AllValuesNaN),
    };

    if (high.len() - first_valid_idx) < period {
        return Err(CorrelHlError::NotEnoughValidData {
            needed: period,
            valid: high.len() - first_valid_idx,
        });
    }

    #[inline]
    fn corr_from_sums(
        sum_h: f64,
        sum_h2: f64,
        sum_l: f64,
        sum_l2: f64,
        sum_hl: f64,
        period: f64,
    ) -> f64 {
        let cov = sum_hl - (sum_h * sum_l / period);
        let var_h = sum_h2 - (sum_h * sum_h / period);
        let var_l = sum_l2 - (sum_l * sum_l / period);

        if var_h <= 0.0 || var_l <= 0.0 {
            0.0
        } else {
            cov / (var_h.sqrt() * var_l.sqrt())
        }
    }

    let inv_period = 1.0 / (period as f64);
    let mut values = vec![f64::NAN; high.len()];

    let mut sum_h = 0.0;
    let mut sum_h2 = 0.0;
    let mut sum_l = 0.0;
    let mut sum_l2 = 0.0;
    let mut sum_hl = 0.0;

    for i in first_valid_idx..(first_valid_idx + period) {
        let h = high[i];
        let l = low[i];
        sum_h += h;
        sum_h2 += h * h;
        sum_l += l;
        sum_l2 += l * l;
        sum_hl += h * l;
    }

    values[first_valid_idx + period - 1] =
        corr_from_sums(sum_h, sum_h2, sum_l, sum_l2, sum_hl, period as f64);

    for i in (first_valid_idx + period)..high.len() {
        let old_idx = i - period;
        let new_idx = i;

        let old_h = high[old_idx];
        let old_l = low[old_idx];
        let new_h = high[new_idx];
        let new_l = low[new_idx];

        if old_h.is_nan() || old_l.is_nan() || new_h.is_nan() || new_l.is_nan() {
            sum_h = 0.0;
            sum_h2 = 0.0;
            sum_l = 0.0;
            sum_l2 = 0.0;
            sum_hl = 0.0;
            for j in (i - period + 1)..=i {
                let hh = high[j];
                let ll = low[j];
                sum_h += hh;
                sum_h2 += hh * hh;
                sum_l += ll;
                sum_l2 += ll * ll;
                sum_hl += hh * ll;
            }
        } else {
            sum_h += new_h - old_h;
            sum_h2 += new_h * new_h - old_h * old_h;
            sum_l += new_l - old_l;
            sum_l2 += new_l * new_l - old_l * old_l;
            sum_hl += new_h * new_l - old_h * old_l;
        }

        values[i] = corr_from_sums(sum_h, sum_h2, sum_l, sum_l2, sum_hl, period as f64);
    }

    Ok(CorrelHlOutput { values })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_correl_hl_basic_slices() {
        let high = [2.0, 3.0, 5.0, 7.0, 8.0];
        let low = [1.0, 2.0, 2.5, 6.0, 7.0];
        let params = CorrelHlParams { period: Some(3) };
        let input = CorrelHlInput::from_slices(&high, &low, params);
        let output = correl_hl(&input).expect("Failed correl_hl on basic slices");
        assert_eq!(output.values.len(), high.len());
        for i in 0..2 {
            assert!(output.values[i].is_nan());
        }
        for &val in &output.values[2..] {
            assert!(!val.is_nan());
        }
    }

    #[test]
    fn test_correl_hl_with_zero_period() {
        let high = [5.0, 6.0, 7.0];
        let low = [1.0, 2.0, 3.0];
        let params = CorrelHlParams { period: Some(0) };
        let input = CorrelHlInput::from_slices(&high, &low, params);
        let result = correl_hl(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_correl_hl_with_period_exceeding_data_length() {
        let high = [5.0, 6.0, 7.0];
        let low = [1.0, 2.0, 3.0];
        let params = CorrelHlParams { period: Some(10) };
        let input = CorrelHlInput::from_slices(&high, &low, params);
        let result = correl_hl(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_correl_hl_data_length_mismatch() {
        let high = [1.0, 2.0, 3.0];
        let low = [1.0, 2.0];
        let params = CorrelHlParams { period: Some(2) };
        let input = CorrelHlInput::from_slices(&high, &low, params);
        let result = correl_hl(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_correl_hl_with_all_nan() {
        let high = [f64::NAN, f64::NAN, f64::NAN];
        let low = [f64::NAN, f64::NAN, f64::NAN];
        let params = CorrelHlParams { period: Some(2) };
        let input = CorrelHlInput::from_slices(&high, &low, params);
        let result = correl_hl(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_correl_hl_from_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = CorrelHlParams { period: Some(9) };
        let input = CorrelHlInput::from_candles(&candles, params);
        let output = correl_hl(&input).expect("Failed correl_hl from candles");
        assert_eq!(output.values.len(), candles.close.len());
    }

    #[test]
    fn test_correl_hl_reinput() {
        let high = [1.0, 2.0, 3.0, 4.0, 5.0];
        let low = [0.5, 1.0, 1.5, 2.0, 2.5];
        let first_params = CorrelHlParams { period: Some(2) };
        let first_input = CorrelHlInput::from_slices(&high, &low, first_params);
        let first_result = correl_hl(&first_input).unwrap();
        let second_params = CorrelHlParams { period: Some(2) };
        let second_input = CorrelHlInput::from_slices(&first_result.values, &low, second_params);
        let second_result = correl_hl(&second_input).unwrap();
        assert_eq!(second_result.values.len(), low.len());
    }

    #[test]
    fn test_correl_hl_expected_values() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = CorrelHlParams { period: Some(5) };
        let input = CorrelHlInput::from_candles(&candles, params);
        let result = correl_hl(&input).expect("Failed correl_hl calculation");
        let expected = [
            0.04589155420456278,
            0.6491664099299647,
            0.9691259236943873,
            0.9915438003818791,
            0.8460608423095615,
        ];
        assert!(result.values.len() >= 5);
        let start_index = result.values.len() - 5;
        for (i, &val) in result.values[start_index..].iter().enumerate() {
            let exp = expected[i];
            let diff = (val - exp).abs();
            assert!(
                diff < 1e-7,
                "Value mismatch at index {}: expected {}, got {}",
                i,
                exp,
                val
            );
        }
    }
}
