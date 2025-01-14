/// # Money Flow Index (MFI)
///
/// MFI is a momentum indicator that measures the inflow and outflow of money into an asset
/// over a specified period. It uses price and volume to identify overbought or oversold
/// conditions by comparing the "typical price" movement and volume flow.
///
/// ## Parameters
/// - **period**: The window size. Defaults to 14.
///
/// ## Errors
/// - **EmptyData**: mfi: Input data slices or candle fields are empty.
/// - **InvalidPeriod**: mfi: `period` is zero, or exceeds the data length.
/// - **NotEnoughValidData**: mfi: Fewer than `period` valid (non-`NaN`) data points remain
///   after the first valid index.
/// - **AllValuesNaN**: mfi: All computed typical prices or volumes are `NaN`.
///
/// ## Returns
/// - **`Ok(MfiOutput)`** on success, containing a `Vec<f64>` matching the input length,
///   with leading `NaN`s until the MFI window is filled.
/// - **`Err(MfiError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum MfiData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        volume: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct MfiOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct MfiParams {
    pub period: Option<usize>,
}

impl Default for MfiParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct MfiInput<'a> {
    pub data: MfiData<'a>,
    pub params: MfiParams,
}

impl<'a> MfiInput<'a> {
    pub fn from_candles(candles: &'a Candles, params: MfiParams) -> Self {
        Self {
            data: MfiData::Candles { candles },
            params,
        }
    }

    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        volume: &'a [f64],
        params: MfiParams,
    ) -> Self {
        Self {
            data: MfiData::Slices {
                high,
                low,
                close,
                volume,
            },
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: MfiData::Candles { candles },
            params: MfiParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| MfiParams::default().period.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum MfiError {
    #[error("mfi: Empty data provided.")]
    EmptyData,
    #[error("mfi: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("mfi: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("mfi: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn mfi(input: &MfiInput) -> Result<MfiOutput, MfiError> {
    let (high, low, close, volume) = match &input.data {
        MfiData::Candles { candles } => {
            let h = &candles.high;
            let l = &candles.low;
            let c = &candles.close;
            let v = &candles.volume;
            if h.is_empty() || l.is_empty() || c.is_empty() || v.is_empty() {
                return Err(MfiError::EmptyData);
            }
            (h.as_slice(), l.as_slice(), c.as_slice(), v.as_slice())
        }
        MfiData::Slices {
            high,
            low,
            close,
            volume,
        } => {
            if high.is_empty() || low.is_empty() || close.is_empty() || volume.is_empty() {
                return Err(MfiError::EmptyData);
            }
            (*high, *low, *close, *volume)
        }
    };

    let length = high.len();
    if length != low.len() || length != close.len() || length != volume.len() {
        return Err(MfiError::EmptyData);
    }
    if length == 0 {
        return Err(MfiError::EmptyData);
    }

    let period = input.get_period();
    if period == 0 || period > length {
        return Err(MfiError::InvalidPeriod {
            period,
            data_len: length,
        });
    }

    let first_valid_idx = (0..length).find(|&i| {
        !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan() && !volume[i].is_nan()
    });
    let first_valid_idx = match first_valid_idx {
        Some(idx) => idx,
        None => return Err(MfiError::AllValuesNaN),
    };
    if (length - first_valid_idx) < period {
        return Err(MfiError::NotEnoughValidData {
            needed: period,
            valid: length - first_valid_idx,
        });
    }

    let mut mfi_values = vec![f64::NAN; length];
    let mut typical = vec![f64::NAN; length];

    for i in first_valid_idx..length {
        if !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan() {
            typical[i] = (high[i] + low[i] + close[i]) / 3.0;
        }
    }

    let mut pos_buf = vec![0.0; period];
    let mut neg_buf = vec![0.0; period];
    let mut pos_sum = 0.0;
    let mut neg_sum = 0.0;

    let mut prev_typical = typical[first_valid_idx];
    let mut ring_idx = 0;

    for i in (first_valid_idx + 1)..(first_valid_idx + period) {
        let diff = typical[i] - prev_typical;
        prev_typical = typical[i];
        let flow = typical[i] * volume[i];
        if diff > 0.0 {
            pos_buf[ring_idx] = flow;
            neg_buf[ring_idx] = 0.0;
            pos_sum += flow;
        } else if diff < 0.0 {
            neg_buf[ring_idx] = flow;
            pos_buf[ring_idx] = 0.0;
            neg_sum += flow;
        } else {
            pos_buf[ring_idx] = 0.0;
            neg_buf[ring_idx] = 0.0;
        }
        ring_idx = (ring_idx + 1) % period;
    }

    let idx_mfi_start = first_valid_idx + period - 1;
    if idx_mfi_start < length {
        let total = pos_sum + neg_sum;
        mfi_values[idx_mfi_start] = if total < 1e-14 {
            0.0
        } else {
            100.0 * (pos_sum / total)
        };
    }

    for i in (first_valid_idx + period)..length {
        let old_pos = pos_buf[ring_idx];
        let old_neg = neg_buf[ring_idx];
        pos_sum -= old_pos;
        neg_sum -= old_neg;

        let diff = typical[i] - prev_typical;
        prev_typical = typical[i];
        let flow = typical[i] * volume[i];

        if diff > 0.0 {
            pos_buf[ring_idx] = flow;
            neg_buf[ring_idx] = 0.0;
            pos_sum += flow;
        } else if diff < 0.0 {
            neg_buf[ring_idx] = flow;
            pos_buf[ring_idx] = 0.0;
            neg_sum += flow;
        } else {
            pos_buf[ring_idx] = 0.0;
            neg_buf[ring_idx] = 0.0;
        }

        ring_idx = (ring_idx + 1) % period;

        let total = pos_sum + neg_sum;
        mfi_values[i] = if total < 1e-14 {
            0.0
        } else {
            100.0 * (pos_sum / total)
        };
    }

    if mfi_values.iter().skip(first_valid_idx).all(|&x| x.is_nan()) {
        return Err(MfiError::AllValuesNaN);
    }

    Ok(MfiOutput { values: mfi_values })
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_mfi_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = MfiInput::with_default_candles(&candles);
        let output = mfi(&input).expect("MFI calculation failed");
        assert_eq!(output.values.len(), candles.close.len());
    }

    #[test]
    fn test_mfi_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let default_params = MfiParams { period: None };
        let input_default = MfiInput::from_candles(&candles, default_params);
        let output_default = mfi(&input_default).expect("Failed MFI with default params");
        assert_eq!(output_default.values.len(), candles.close.len());

        let params_period_10 = MfiParams { period: Some(10) };
        let input_period_10 = MfiInput::from_candles(&candles, params_period_10);
        let output_period_10 = mfi(&input_period_10).expect("Failed MFI with period=10");
        assert_eq!(output_period_10.values.len(), candles.close.len());
    }

    #[test]
    fn test_mfi_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = MfiParams { period: Some(14) };
        let input = MfiInput::from_candles(&candles, params);
        let mfi_result = mfi(&input).expect("Failed to calculate MFI");
        let expected_last_five_mfi = [
            38.13874339324763,
            37.44139770113819,
            31.02039511395131,
            28.092605898618896,
            25.905204729397813,
        ];
        assert!(mfi_result.values.len() >= 5);
        let start_index = mfi_result.values.len() - 5;
        let result_last_five_mfi = &mfi_result.values[start_index..];
        for (i, &value) in result_last_five_mfi.iter().enumerate() {
            let expected_value = expected_last_five_mfi[i];
            let diff = (value - expected_value).abs();
            assert!(
                diff < 1e-1,
                "MFI mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }
    }

    #[test]
    fn test_mfi_params_with_default_params() {
        let default_params = MfiParams::default();
        assert_eq!(default_params.period, Some(14));
    }

    #[test]
    fn test_mfi_with_invalid_period() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let zero_period_params = MfiParams { period: Some(0) };
        let input_zero = MfiInput::from_candles(&candles, zero_period_params);
        let result_zero = mfi(&input_zero);
        assert!(result_zero.is_err());
        if let Err(e) = result_zero {
            assert!(
                e.to_string().contains("Invalid period"),
                "Expected 'Invalid period' error, got: {}",
                e
            );
        }

        let large_period_params = MfiParams {
            period: Some(9999999),
        };
        let input_large = MfiInput::from_candles(&candles, large_period_params);
        let result_large = mfi(&input_large);
        assert!(result_large.is_err());
    }

    #[test]
    fn test_mfi_with_too_small_data() {
        let input_high = [1.0, 2.0, 3.0];
        let input_low = [0.5, 1.5, 2.5];
        let input_close = [0.8, 1.8, 2.8];
        let input_volume = [100.0, 200.0, 300.0];
        let params = MfiParams { period: Some(5) };
        let input =
            MfiInput::from_slices(&input_high, &input_low, &input_close, &input_volume, params);
        let result = mfi(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_mfi_with_all_nan() {
        let input_high = [f64::NAN, f64::NAN];
        let input_low = [f64::NAN, f64::NAN];
        let input_close = [f64::NAN, f64::NAN];
        let input_volume = [f64::NAN, f64::NAN];
        let params = MfiParams { period: Some(2) };
        let input =
            MfiInput::from_slices(&input_high, &input_low, &input_close, &input_volume, params);
        let result = mfi(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_mfi_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let first_params = MfiParams { period: Some(7) };
        let first_input = MfiInput::from_candles(&candles, first_params);
        let first_result = mfi(&first_input).expect("Failed to calculate first MFI");
        let second_params = MfiParams { period: Some(7) };
        let high_values: Vec<f64> = first_result.values.clone();
        let low_values: Vec<f64> = first_result.values.clone();
        let close_values: Vec<f64> = first_result.values.clone();
        let volume_values: Vec<f64> = vec![10_000.0; first_result.values.len()];
        let second_input = MfiInput::from_slices(
            &high_values,
            &low_values,
            &close_values,
            &volume_values,
            second_params,
        );
        let second_result = mfi(&second_input).expect("Failed to calculate second MFI");
        assert_eq!(second_result.values.len(), first_result.values.len());
    }
}
