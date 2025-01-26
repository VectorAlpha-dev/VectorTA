use crate::indicators::moving_averages::ema::{ema, EmaError, EmaInput, EmaParams};
/// # WaveTrend Indicator
///
/// A technical indicator that calculates an oscillator (WT1, WT2, and WT_DIFF = WT2 - WT1)
/// using a combination of EMA, absolute deviations, and SMA. This version includes:
/// - **WT1**: An EMA of a transformed price series.
/// - **WT2**: An SMA of WT1.
/// - **WT_DIFF**: The difference (WT2 - WT1).
///
/// ## Parameters
/// - **channel_length**: Period for the initial EMA computations. Defaults to 9.
/// - **average_length**: Period for the secondary EMA of the transformed data. Defaults to 12.
/// - **ma_length**: Period for the final SMA on WT1. Defaults to 3.
/// - **factor**: Constant multiplier for the transformation. Defaults to 0.015.
///
/// ## Errors
/// - **EmptyData**: No input data provided.
/// - **AllValuesNaN**: All input data values are NaN.
/// - **InvalidChannelLen**: channel_length is zero or exceeds the data length.
/// - **InvalidAverageLen**: average_length is zero or exceeds the data length.
/// - **InvalidMaLen**: ma_length is zero or exceeds the data length.
/// - **NotEnoughValidData**: Fewer than the required valid (non-NaN) data points remain
///   after the first valid index.
///
/// ## Returns
/// - **`Ok(WavetrendOutput)`** on success, containing three `Vec<f64>` matching the input length,
///   each with leading `NaN`s until the calculations become valid:
///   - **`wt1`**: WaveTrend 1
///   - **`wt2`**: WaveTrend 2
///   - **`wt_diff`**: WT2 - WT1
/// - **`Err(WavetrendError)`** otherwise.
use crate::indicators::moving_averages::sma::{sma, SmaError, SmaInput, SmaParams};
use crate::utilities::data_loader::{source_type, Candles};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum WavetrendData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct WavetrendOutput {
    pub wt1: Vec<f64>,
    pub wt2: Vec<f64>,
    pub wt_diff: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct WavetrendParams {
    pub channel_length: Option<usize>,
    pub average_length: Option<usize>,
    pub ma_length: Option<usize>,
    pub factor: f64,
}

impl Default for WavetrendParams {
    fn default() -> Self {
        Self {
            channel_length: Some(9),
            average_length: Some(12),
            ma_length: Some(3),
            factor: 0.015,
        }
    }
}

#[derive(Debug, Clone)]
pub struct WavetrendInput<'a> {
    pub data: WavetrendData<'a>,
    pub params: WavetrendParams,
}

impl<'a> WavetrendInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: WavetrendParams) -> Self {
        Self {
            data: WavetrendData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: WavetrendParams) -> Self {
        Self {
            data: WavetrendData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: WavetrendData::Candles {
                candles,
                source: "hlc3",
            },
            params: WavetrendParams::default(),
        }
    }

    fn get_channel_length(&self) -> usize {
        self.params
            .channel_length
            .unwrap_or_else(|| WavetrendParams::default().channel_length.unwrap())
    }

    fn get_average_length(&self) -> usize {
        self.params
            .average_length
            .unwrap_or_else(|| WavetrendParams::default().average_length.unwrap())
    }

    fn get_ma_length(&self) -> usize {
        self.params
            .ma_length
            .unwrap_or_else(|| WavetrendParams::default().ma_length.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum WavetrendError {
    #[error("wavetrend: Empty data provided.")]
    EmptyData,
    #[error("wavetrend: All values are NaN.")]
    AllValuesNaN,
    #[error("wavetrend: Invalid channel_length = {channel_length}, data length = {data_len}")]
    InvalidChannelLen {
        channel_length: usize,
        data_len: usize,
    },
    #[error("wavetrend: Invalid average_length = {average_length}, data length = {data_len}")]
    InvalidAverageLen {
        average_length: usize,
        data_len: usize,
    },
    #[error("wavetrend: Invalid ma_length = {ma_length}, data length = {data_len}")]
    InvalidMaLen { ma_length: usize, data_len: usize },
    #[error("wavetrend: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("wavetrend: EMA error {0}")]
    EmaError(#[from] EmaError),
    #[error("wavetrend: SMA error {0}")]
    SmaError(#[from] SmaError),
}

#[inline]
pub fn wavetrend(input: &WavetrendInput) -> Result<WavetrendOutput, WavetrendError> {
    let data: &[f64] = match &input.data {
        WavetrendData::Candles { candles, source } => source_type(candles, source),
        WavetrendData::Slice(slice) => slice,
    };

    if data.is_empty() {
        return Err(WavetrendError::EmptyData);
    }

    let channel_len = input.get_channel_length();
    let average_len = input.get_average_length();
    let ma_len = input.get_ma_length();
    let factor = input.params.factor;

    if channel_len == 0 || channel_len > data.len() {
        return Err(WavetrendError::InvalidChannelLen {
            channel_length: channel_len,
            data_len: data.len(),
        });
    }
    if average_len == 0 || average_len > data.len() {
        return Err(WavetrendError::InvalidAverageLen {
            average_length: average_len,
            data_len: data.len(),
        });
    }
    if ma_len == 0 || ma_len > data.len() {
        return Err(WavetrendError::InvalidMaLen {
            ma_length: ma_len,
            data_len: data.len(),
        });
    }

    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(WavetrendError::AllValuesNaN),
    };

    let needed = *[channel_len, average_len, ma_len]
        .iter()
        .max()
        .unwrap_or(&channel_len);

    let valid = data.len() - first_valid_idx;
    if valid < needed {
        return Err(WavetrendError::NotEnoughValidData { needed, valid });
    }

    let data_valid = &data[first_valid_idx..];

    let esa_input = EmaInput::from_slice(
        data_valid,
        EmaParams {
            period: Some(channel_len),
        },
    );
    let esa_output = ema(&esa_input)?;
    let esa_values = &esa_output.values;

    let mut diff_esa = vec![f64::NAN; data_valid.len()];
    for i in 0..data_valid.len() {
        if !data_valid[i].is_nan() && !esa_values[i].is_nan() {
            diff_esa[i] = (data_valid[i] - esa_values[i]).abs();
        }
    }

    let de_input = EmaInput::from_slice(
        &diff_esa,
        EmaParams {
            period: Some(channel_len),
        },
    );
    let de_output = ema(&de_input)?;
    let de_values = &de_output.values;

    let mut ci = vec![f64::NAN; data_valid.len()];
    for i in 0..data_valid.len() {
        if !data_valid[i].is_nan() && !esa_values[i].is_nan() && !de_values[i].is_nan() {
            let den = factor * de_values[i];
            if den != 0.0 {
                ci[i] = (data_valid[i] - esa_values[i]) / den;
            }
        }
    }

    let wt1_input = EmaInput::from_slice(
        &ci,
        EmaParams {
            period: Some(average_len),
        },
    );
    let wt1_output = ema(&wt1_input)?;
    let wt1_values = &wt1_output.values;

    let wt2_input = SmaInput::from_slice(
        wt1_values,
        SmaParams {
            period: Some(ma_len),
        },
    );
    let wt2_output = sma(&wt2_input)?;
    let wt2_values = &wt2_output.values;

    let mut wt1_final = vec![f64::NAN; data.len()];
    let mut wt2_final = vec![f64::NAN; data.len()];
    let mut diff_final = vec![f64::NAN; data.len()];

    for i in 0..data_valid.len() {
        wt1_final[i + first_valid_idx] = wt1_values[i];
        wt2_final[i + first_valid_idx] = wt2_values[i];
        if !wt1_values[i].is_nan() && !wt2_values[i].is_nan() {
            diff_final[i + first_valid_idx] = wt2_values[i] - wt1_values[i];
        }
    }

    Ok(WavetrendOutput {
        wt1: wt1_final,
        wt2: wt2_final,
        wt_diff: diff_final,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_wavetrend_empty_data() {
        let input_data: [f64; 0] = [];
        let params = WavetrendParams::default();
        let input = WavetrendInput::from_slice(&input_data, params);
        let result = wavetrend(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                e.to_string().contains("Empty data provided"),
                "Expected empty data error, got: {}",
                e
            );
        }
    }

    #[test]
    fn test_wavetrend_all_nan() {
        let input_data = [f64::NAN, f64::NAN, f64::NAN];
        let params = WavetrendParams::default();
        let input = WavetrendInput::from_slice(&input_data, params);
        let result = wavetrend(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_wavetrend_invalid_channel_len() {
        let input_data = [10.0, 20.0, 30.0];
        let params = WavetrendParams {
            channel_length: Some(0),
            average_length: Some(12),
            ma_length: Some(3),
            factor: 0.015,
        };
        let input = WavetrendInput::from_slice(&input_data, params);
        let result = wavetrend(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                e.to_string().contains("Invalid channel_length"),
                "Expected invalid channel_length error, got: {}",
                e
            );
        }
    }

    #[test]
    fn test_wavetrend_invalid_average_len() {
        let input_data = [10.0, 20.0, 30.0];
        let params = WavetrendParams {
            channel_length: Some(9),
            average_length: Some(999),
            ma_length: Some(3),
            factor: 0.015,
        };
        let input = WavetrendInput::from_slice(&input_data, params);
        let result = wavetrend(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_wavetrend_invalid_ma_len() {
        let input_data = [10.0, 20.0, 30.0];
        let params = WavetrendParams {
            channel_length: Some(9),
            average_length: Some(12),
            ma_length: Some(999),
            factor: 0.015,
        };
        let input = WavetrendInput::from_slice(&input_data, params);
        let result = wavetrend(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_wavetrend_not_enough_valid_data() {
        let input_data = [10.0];
        let params = WavetrendParams::default();
        let input = WavetrendInput::from_slice(&input_data, params);
        let result = wavetrend(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_wavetrend_basic_calculation() {
        let input_data = [
            100.0, 105.0, 102.0, 108.0, 109.0, 110.0, 115.0, 112.0, 111.0, 120.0, 119.0, 121.0,
        ];
        let params = WavetrendParams {
            channel_length: Some(3),
            average_length: Some(3),
            ma_length: Some(2),
            factor: 0.015,
        };
        let input = WavetrendInput::from_slice(&input_data, params);
        let result = wavetrend(&input).expect("Failed to calculate Wavetrend");
        assert_eq!(result.wt1.len(), input_data.len());
        assert_eq!(result.wt2.len(), input_data.len());
        assert_eq!(result.wt_diff.len(), input_data.len());
    }

    #[test]
    fn test_wavetrend_with_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = WavetrendInput::with_default_candles(&candles);
        let output = wavetrend(&input).expect("Failed to calculate Wavetrend with default params");
        assert_eq!(output.wt1.len(), candles.close.len());
        assert_eq!(output.wt2.len(), candles.close.len());
        assert_eq!(output.wt_diff.len(), candles.close.len());
    }

    #[test]
    fn test_wavetrend_accuracy_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = WavetrendParams::default();
        let input = WavetrendInput::from_candles(&candles, "hlc3", params);
        let wt_output = wavetrend(&input).expect("Failed to calculate Wavetrend");

        let len = wt_output.wt1.len();
        assert!(len > 5, "Not enough data for final check");

        let last_five_wt1 = &wt_output.wt1[len - 5..];
        let last_five_wt2 = &wt_output.wt2[len - 5..];

        let expected_wt1 = [
            -29.02058232514538,
            -28.207769813591664,
            -31.991808642927193,
            -31.9218051759519,
            -44.956245952893866,
        ];
        let expected_wt2 = [
            -30.651043230696555,
            -28.686329669808583,
            -29.740053593887932,
            -30.707127877490105,
            -36.2899532572575,
        ];

        for (i, &val) in last_five_wt1.iter().enumerate() {
            assert!(
                val.is_nan() == false,
                "WT1 value should not be NaN near the end of the data."
            );
            let diff = (val - expected_wt1[i]).abs();
            assert!(
                diff < 1e-6,
                "WT1 mismatch at last five #{}: expected {}, got {}",
                i,
                expected_wt1[i],
                val
            );
        }

        for (i, &val) in last_five_wt2.iter().enumerate() {
            assert!(
                val.is_nan() == false,
                "WT2 value should not be NaN near the end of the data."
            );
            let diff = (val - expected_wt2[i]).abs();
            assert!(
                diff < 1e-6,
                "WT2 mismatch at last five #{}: expected {}, got {}",
                i,
                expected_wt2[i],
                val
            );
        }

        let last_five_diff = &wt_output.wt_diff[len - 5..];
        for i in 0..5 {
            let computed = last_five_diff[i];
            let expected = expected_wt2[i] - expected_wt1[i];
            let diff = (computed - expected).abs();
            assert!(
                diff < 1e-6,
                "WT_DIFF mismatch at last five #{}: expected {}, got {}",
                i,
                expected,
                computed
            );
        }
    }
}
