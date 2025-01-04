use crate::indicators::deviation::{deviation, DevInput, DevParams};
use crate::indicators::moving_averages::ma::{ma, MaData};
/// # Bollinger Bands Width (BBW)
///
/// Bollinger Bands Width (sometimes called Bandwidth) shows the relative distance between
/// the upper and lower Bollinger Bands compared to the middle band.  
/// It is typically calculated as:
///
/// \[ (upper_band - lower_band) / middle_band \]
///
/// ## Parameters
/// - **period**: The underlying Bollinger Bands MA window. Defaults to 20.
/// - **devup**: Upward multiplier for the deviation. Defaults to 2.0.
/// - **devdn**: Downward multiplier for the deviation. Defaults to 2.0.
/// - **matype**: String specifying the MA type (e.g., "sma", "ema"). Defaults to "sma".
/// - **devtype**: Deviation type (0 => std dev, 1 => mean_ad, 2 => median_ad). Defaults to 0.
///
/// ## Errors
/// - **EmptyData**: Input data slice is empty.
/// - **InvalidPeriod**: `period` is zero or exceeds the data length.
/// - **AllValuesNaN**: All input data values are `NaN`.
/// - **NotEnoughValidData**: Fewer than `period` valid data points after the first valid index.
/// - **UnderlyingFunctionFailed**: If the underlying Bollinger Bands computation fails.
///
/// ## Returns
/// - **`Ok(BollingerBandsWidthOutput)`** on success, containing a single `Vec<f64>` matching the input length:
///   - `values[i] = (upper_band[i] - lower_band[i]) / middle_band[i]`
/// - **`Err(BollingerBandsWidthError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum BollingerBandsWidthData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct BollingerBandsWidthParams {
    pub period: Option<usize>,
    pub devup: Option<f64>,
    pub devdn: Option<f64>,
    pub matype: Option<String>,
    pub devtype: Option<usize>,
}

impl Default for BollingerBandsWidthParams {
    fn default() -> Self {
        Self {
            period: Some(20),
            devup: Some(2.0),
            devdn: Some(2.0),
            matype: Some("sma".to_string()),
            devtype: Some(0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BollingerBandsWidthInput<'a> {
    pub data: BollingerBandsWidthData<'a>,
    pub params: BollingerBandsWidthParams,
}

impl<'a> BollingerBandsWidthInput<'a> {
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: BollingerBandsWidthParams,
    ) -> Self {
        Self {
            data: BollingerBandsWidthData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: BollingerBandsWidthParams) -> Self {
        Self {
            data: BollingerBandsWidthData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: BollingerBandsWidthData::Candles {
                candles,
                source: "close",
            },
            params: BollingerBandsWidthParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| BollingerBandsWidthParams::default().period.unwrap())
    }

    pub fn get_devup(&self) -> f64 {
        self.params
            .devup
            .unwrap_or_else(|| BollingerBandsWidthParams::default().devup.unwrap())
    }

    pub fn get_devdn(&self) -> f64 {
        self.params
            .devdn
            .unwrap_or_else(|| BollingerBandsWidthParams::default().devdn.unwrap())
    }

    pub fn get_matype(&self) -> String {
        self.params
            .matype
            .clone()
            .unwrap_or_else(|| BollingerBandsWidthParams::default().matype.unwrap())
    }

    pub fn get_devtype(&self) -> usize {
        self.params
            .devtype
            .unwrap_or_else(|| BollingerBandsWidthParams::default().devtype.unwrap())
    }
}

#[derive(Debug, Clone)]
pub struct BollingerBandsWidthOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Error)]
pub enum BollingerBandsWidthError {
    #[error("bollinger_bands_width: Empty data provided.")]
    EmptyData,
    #[error("bollinger_bands_width: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("bollinger_bands_width: All values are NaN.")]
    AllValuesNaN,
    #[error("bollinger_bands_width: Underlying MA or Deviation function failed: {0}")]
    UnderlyingFunctionFailed(String),
    #[error(
        "bollinger_bands_width: Not enough valid data for period: needed={needed}, valid={valid}"
    )]
    NotEnoughValidData { needed: usize, valid: usize },
}

#[inline]
pub fn bollinger_bands_width(
    input: &BollingerBandsWidthInput,
) -> Result<BollingerBandsWidthOutput, BollingerBandsWidthError> {
    let data: &[f64] = match &input.data {
        BollingerBandsWidthData::Candles { candles, source } => source_type(candles, source),
        BollingerBandsWidthData::Slice(slice) => slice,
    };

    if data.is_empty() {
        return Err(BollingerBandsWidthError::EmptyData);
    }
    let period = input.get_period();
    if period == 0 || period > data.len() {
        return Err(BollingerBandsWidthError::InvalidPeriod {
            period,
            data_len: data.len(),
        });
    }
    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(BollingerBandsWidthError::AllValuesNaN),
    };
    if (data.len() - first_valid_idx) < period {
        return Err(BollingerBandsWidthError::NotEnoughValidData {
            needed: period,
            valid: data.len() - first_valid_idx,
        });
    }

    let devup = input.get_devup();
    let devdn = input.get_devdn();
    let matype = input.get_matype();
    let devtype = input.get_devtype();

    let ma_data = match &input.data {
        BollingerBandsWidthData::Candles { candles, source } => MaData::Candles { candles, source },
        BollingerBandsWidthData::Slice(slice) => MaData::Slice(slice),
    };
    let middle = ma(&matype, ma_data, period)
        .map_err(|e| BollingerBandsWidthError::UnderlyingFunctionFailed(e.to_string()))?;
    let dev_input = crate::indicators::deviation::DevInput::from_slice(
        data,
        crate::indicators::deviation::DevParams {
            period: Some(period),
            devtype: Some(devtype),
        },
    );
    let dev_values = crate::indicators::deviation::deviation(&dev_input)
        .map_err(|e| BollingerBandsWidthError::UnderlyingFunctionFailed(e.to_string()))?;

    let mut upper_band = 0.0;
    let mut middle_band = 0.0;
    let mut lower_band = 0.0;

    let mut output = vec![f64::NAN; data.len()];

    for i in (first_valid_idx + period - 1)..data.len() {
        middle_band = middle[i];
        upper_band = middle[i] + devup * dev_values[i];
        lower_band = middle[i] - devdn * dev_values[i];
        output[i] = (upper_band - lower_band) / middle_band;
    }

    Ok(BollingerBandsWidthOutput { values: output })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_bbw_default() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = BollingerBandsWidthInput::with_default_candles(&candles);
        let result = bollinger_bands_width(&input).expect("Failed to calculate BBWidth");

        assert_eq!(result.values.len(), candles.close.len());
    }

    #[test]
    fn test_bbw_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let partial_params = BollingerBandsWidthParams {
            period: Some(22),
            devup: Some(2.2),
            devdn: None,
            matype: Some("ema".to_string()),
            devtype: None,
        };
        let input = BollingerBandsWidthInput::from_candles(&candles, "hl2", partial_params);
        let result = bollinger_bands_width(&input).expect("Failed to calculate BBWidth partial");
        assert_eq!(result.values.len(), candles.close.len());
    }

    #[test]
    fn test_bbw_with_zero_period() {
        let data = [10.0, 20.0, 30.0];
        let params = BollingerBandsWidthParams {
            period: Some(0),
            ..BollingerBandsWidthParams::default()
        };
        let input = BollingerBandsWidthInput::from_slice(&data, params);

        let result = bollinger_bands_width(&input);
        assert!(result.is_err(), "Expected an error for zero period");
    }

    #[test]
    fn test_bbw_with_period_exceeding_data_length() {
        let data = [10.0, 20.0, 30.0];
        let params = BollingerBandsWidthParams {
            period: Some(10),
            ..BollingerBandsWidthParams::default()
        };
        let input = BollingerBandsWidthInput::from_slice(&data, params);

        let result = bollinger_bands_width(&input);
        assert!(result.is_err(), "Expected an error for period > data.len()");
    }

    #[test]
    fn test_bbw_very_small_data_set() {
        let data = [42.0];
        let input =
            BollingerBandsWidthInput::from_slice(&data, BollingerBandsWidthParams::default());

        let result = bollinger_bands_width(&input);
        assert!(
            result.is_err(),
            "Expected error for data smaller than default period=20"
        );
    }

    #[test]
    fn test_bbw_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = BollingerBandsWidthInput::with_default_candles(&candles);
        let result = bollinger_bands_width(&input).expect("Failed to calculate BBWidth");

        let check_index = 240;
        if result.values.len() > check_index {
            for i in check_index..result.values.len() {
                if !result.values[i].is_nan() {
                    break;
                }
                if i == (result.values.len() - 1) {
                    panic!(
                        "All BBWidth values from index {} onward are NaN.",
                        check_index
                    );
                }
            }
        }
    }
}
