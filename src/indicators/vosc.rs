/// # Volume Oscillator (VOSC)
///
/// The Volume Oscillator (VOSC) is calculated using two moving averages (short period and long period)
/// of volume. It is typically used to measure changes in volume trends.
///
/// ## Formula
/// ```ignore
/// vosc = 100 * ((short_avg - long_avg) / long_avg)
/// ```
/// where:
/// - **short_avg** = average volume over `short_period`
/// - **long_avg** = average volume over `long_period`
///
/// ## Parameters
/// - **short_period**: The short window size. Defaults to 2.
/// - **long_period**: The long window size. Defaults to 5.
///
/// ## Errors
/// - **EmptyData**: vosc: Input data slice is empty.
/// - **InvalidShortPeriod**: vosc: `short_period` is zero or exceeds the data length.
/// - **InvalidLongPeriod**: vosc: `long_period` is zero or exceeds the data length.
/// - **ShortPeriodGreaterThanLongPeriod**: vosc: `short_period` is greater than `long_period`.
/// - **NotEnoughValidData**: vosc: Fewer than `long_period` valid (non-`NaN`) data points remain
///   after the first valid index.
/// - **AllValuesNaN**: vosc: All input data values are `NaN`.
///
/// ## Returns
/// - **`Ok(VoscOutput)`** on success, containing a `Vec<f64>` matching the input length,
///   with leading `NaN`s until the first valid index + `long_period` - 1.
/// - **`Err(VoscError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum VoscData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct VoscOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct VoscParams {
    pub short_period: Option<usize>,
    pub long_period: Option<usize>,
}

impl Default for VoscParams {
    fn default() -> Self {
        Self {
            short_period: Some(2),
            long_period: Some(5),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VoscInput<'a> {
    pub data: VoscData<'a>,
    pub params: VoscParams,
}

impl<'a> VoscInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: VoscParams) -> Self {
        Self {
            data: VoscData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: VoscParams) -> Self {
        Self {
            data: VoscData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: VoscData::Candles {
                candles,
                source: "volume",
            },
            params: VoscParams::default(),
        }
    }

    pub fn get_short_period(&self) -> usize {
        self.params
            .short_period
            .unwrap_or_else(|| VoscParams::default().short_period.unwrap())
    }

    pub fn get_long_period(&self) -> usize {
        self.params
            .long_period
            .unwrap_or_else(|| VoscParams::default().long_period.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum VoscError {
    #[error("vosc: Empty data provided for VOSC.")]
    EmptyData,
    #[error("vosc: Invalid short period: short_period = {period}, data length = {data_len}")]
    InvalidShortPeriod { period: usize, data_len: usize },
    #[error("vosc: Invalid long period: long_period = {period}, data length = {data_len}")]
    InvalidLongPeriod { period: usize, data_len: usize },
    #[error("vosc: short_period > long_period")]
    ShortPeriodGreaterThanLongPeriod,
    #[error("vosc: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("vosc: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn vosc(input: &VoscInput) -> Result<VoscOutput, VoscError> {
    let data: &[f64] = match &input.data {
        VoscData::Candles { candles, source } => source_type(candles, source),
        VoscData::Slice(slice) => slice,
    };

    if data.is_empty() {
        return Err(VoscError::EmptyData);
    }

    let short_period = input.get_short_period();
    let long_period = input.get_long_period();

    if short_period == 0 || short_period > data.len() {
        return Err(VoscError::InvalidShortPeriod {
            period: short_period,
            data_len: data.len(),
        });
    }

    if long_period == 0 || long_period > data.len() {
        return Err(VoscError::InvalidLongPeriod {
            period: long_period,
            data_len: data.len(),
        });
    }

    if short_period > long_period {
        return Err(VoscError::ShortPeriodGreaterThanLongPeriod);
    }

    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(VoscError::AllValuesNaN),
    };

    if (data.len() - first_valid_idx) < long_period {
        return Err(VoscError::NotEnoughValidData {
            needed: long_period,
            valid: data.len() - first_valid_idx,
        });
    }

    let mut output_values = vec![f64::NAN; data.len()];
    let mut short_sum = 0.0;
    let mut long_sum = 0.0;
    for i in first_valid_idx..(first_valid_idx + long_period) {
        let v = data[i];
        if i >= (first_valid_idx + long_period - short_period) {
            short_sum += v;
        }
        long_sum += v;
    }

    let short_div = 1.0 / (short_period as f64);
    let long_div = 1.0 / (long_period as f64);
    let init_idx = first_valid_idx + long_period - 1;
    let mut savg = short_sum * short_div;
    let mut lavg = long_sum * long_div;
    output_values[init_idx] = 100.0 * (savg - lavg) / lavg;

    for i in (first_valid_idx + long_period)..data.len() {
        short_sum += data[i];
        short_sum -= data[i - short_period];
        long_sum += data[i];
        long_sum -= data[i - long_period];

        savg = short_sum * short_div;
        lavg = long_sum * long_div;
        output_values[i] = 100.0 * (savg - lavg) / lavg;
    }

    Ok(VoscOutput {
        values: output_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_vosc_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let volume = candles
            .select_candle_field("volume")
            .expect("Failed to extract volume data");

        let params = VoscParams {
            short_period: Some(2),
            long_period: Some(5),
        };
        let input = VoscInput::from_candles(&candles, "volume", params);
        let vosc_result = vosc(&input).expect("Failed to calculate VOSC");

        assert_eq!(
            vosc_result.values.len(),
            volume.len(),
            "VOSC length mismatch"
        );

        let expected_last_five_vosc = [
            -39.478510754298895,
            -25.886077312645188,
            -21.155087549723756,
            -12.36093768813373,
            48.70809369473075,
        ];
        assert!(vosc_result.values.len() >= 5, "VOSC length too short");
        let start_index = vosc_result.values.len() - 5;
        let result_last_five_vosc = &vosc_result.values[start_index..];
        for (i, &value) in result_last_five_vosc.iter().enumerate() {
            let expected_value = expected_last_five_vosc[i];
            assert!(
                (value - expected_value).abs() < 1e-1,
                "VOSC mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }

        for i in 0..(5 - 1) {
            assert!(vosc_result.values[i].is_nan());
        }

        let default_input = VoscInput::with_default_candles(&candles);
        let default_vosc_result = vosc(&default_input).expect("Failed to calculate default VOSC");
        assert_eq!(default_vosc_result.values.len(), volume.len());
    }
}
