use crate::indicators::deviation::{deviation, DevInput, DevParams};
use crate::indicators::moving_averages::ma::{ma, MaData};
/// # Bollinger Bands (BB)
///
/// Bollinger Bands are volatility bands typically drawn around a moving average.
/// The middle band is usually an MA of the chosen type, and the upper/lower bands
/// are offset by a certain number of standard deviations (or alternate deviations)
/// above/below the middle band.
///
/// ## Parameters
/// - **period**: The MA window size. Defaults to 20.
/// - **devup**: The upward multiplier for the deviation. Defaults to 2.0.
/// - **devdn**: The downward multiplier for the deviation. Defaults to 2.0.
/// - **matype**: The name of the moving average type (e.g., "sma", "ema"). Defaults to "sma".
/// - **devtype**: The deviation calculation type (e.g., 0 => std dev, 1 => mean_ad, 2 => median_ad).
///   Defaults to 0 (standard deviation).
///
/// ## Errors
/// - **EmptyData**: bollinger_bands: Input data slice is empty.
/// - **InvalidPeriod**: bollinger_bands: `period` is zero or exceeds the data length.
/// - **AllValuesNaN**: bollinger_bands: All input data values are `NaN`.
///
/// ## Returns
/// - **`Ok(BollingerBandsOutput)`** containing three `Vec<f64>` matching input length:
///   - `middle_band`: The MA values.
///   - `upper_band`: `middle_band + devup * deviation`.
///   - `lower_band`: `middle_band - devdn * deviation`.
/// - **`Err(BollingerBandsError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum BollingerBandsData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct BollingerBandsOutput {
    pub upper_band: Vec<f64>,
    pub middle_band: Vec<f64>,
    pub lower_band: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct BollingerBandsParams {
    pub period: Option<usize>,
    pub devup: Option<f64>,
    pub devdn: Option<f64>,
    pub matype: Option<String>,
    pub devtype: Option<usize>,
}

impl Default for BollingerBandsParams {
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
pub struct BollingerBandsInput<'a> {
    pub data: BollingerBandsData<'a>,
    pub params: BollingerBandsParams,
}

impl<'a> BollingerBandsInput<'a> {
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: BollingerBandsParams,
    ) -> Self {
        Self {
            data: BollingerBandsData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: BollingerBandsParams) -> Self {
        Self {
            data: BollingerBandsData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: BollingerBandsData::Candles {
                candles,
                source: "close",
            },
            params: BollingerBandsParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| BollingerBandsParams::default().period.unwrap())
    }

    pub fn get_devup(&self) -> f64 {
        self.params
            .devup
            .unwrap_or_else(|| BollingerBandsParams::default().devup.unwrap())
    }

    pub fn get_devdn(&self) -> f64 {
        self.params
            .devdn
            .unwrap_or_else(|| BollingerBandsParams::default().devdn.unwrap())
    }

    pub fn get_matype(&self) -> String {
        self.params
            .matype
            .clone()
            .unwrap_or_else(|| BollingerBandsParams::default().matype.unwrap())
    }

    pub fn get_devtype(&self) -> usize {
        self.params
            .devtype
            .unwrap_or_else(|| BollingerBandsParams::default().devtype.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum BollingerBandsError {
    #[error("bollinger_bands: Empty data provided.")]
    EmptyData,
    #[error("bollinger_bands: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("bollinger_bands: All values are NaN.")]
    AllValuesNaN,
    #[error("bollinger_bands: Underlying MA or Deviation function failed: {0}")]
    UnderlyingFunctionFailed(String),
    #[error("bollinger_bands: Not enough valid data for period: needed={needed}, valid={valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

#[inline]
pub fn bollinger_bands(
    input: &BollingerBandsInput,
) -> Result<BollingerBandsOutput, BollingerBandsError> {
    let data: &[f64] = match &input.data {
        BollingerBandsData::Candles { candles, source } => source_type(candles, source),
        BollingerBandsData::Slice(slice) => slice,
    };

    if data.is_empty() {
        return Err(BollingerBandsError::EmptyData);
    }
    let period = input.get_period();
    if period == 0 || period > data.len() {
        return Err(BollingerBandsError::InvalidPeriod {
            period,
            data_len: data.len(),
        });
    }
    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(BollingerBandsError::AllValuesNaN),
    };
    if (data.len() - first_valid_idx) < period {
        return Err(BollingerBandsError::NotEnoughValidData {
            needed: period,
            valid: data.len() - first_valid_idx,
        });
    }

    let devup = input.get_devup();
    let devdn = input.get_devdn();
    let matype = input.get_matype();
    let devtype = input.get_devtype();

    let ma_data = match &input.data {
        BollingerBandsData::Candles { candles, source } => MaData::Candles { candles, source },
        BollingerBandsData::Slice(slice) => MaData::Slice(slice),
    };
    let middle = ma(&matype, ma_data, period)
        .map_err(|e| BollingerBandsError::UnderlyingFunctionFailed(e.to_string()))?;
    let dev_input = crate::indicators::deviation::DevInput::from_slice(
        data,
        crate::indicators::deviation::DevParams {
            period: Some(period),
            devtype: Some(devtype),
        },
    );
    let dev_values = crate::indicators::deviation::deviation(&dev_input)
        .map_err(|e| BollingerBandsError::UnderlyingFunctionFailed(e.to_string()))?;

    let mut upper_band = vec![f64::NAN; data.len()];
    let mut middle_band = vec![f64::NAN; data.len()];
    let mut lower_band = vec![f64::NAN; data.len()];

    for i in (first_valid_idx + period - 1)..data.len() {
        if !middle[i].is_nan() && !dev_values[i].is_nan() {
            middle_band[i] = middle[i];
            upper_band[i] = middle[i] + devup * dev_values[i];
            lower_band[i] = middle[i] - devdn * dev_values[i];
        }
    }

    Ok(BollingerBandsOutput {
        upper_band,
        middle_band,
        lower_band,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_bollinger_bands_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let partial_params = BollingerBandsParams {
            period: Some(22),
            devup: None,
            devdn: None,
            matype: Some("sma".to_string()),
            devtype: None,
        };
        let input_partial =
            BollingerBandsInput::from_candles(&candles, "close", partial_params.clone());
        let output_partial = bollinger_bands(&input_partial)
            .expect("Failed Bollinger Bands with partial params (period=22, default devup/devdn)");

        assert_eq!(output_partial.upper_band.len(), candles.close.len());
        assert_eq!(output_partial.middle_band.len(), candles.close.len());
        assert_eq!(output_partial.lower_band.len(), candles.close.len());

        let partial_params2 = BollingerBandsParams {
            period: Some(10),
            matype: Some("ema".to_string()),
            ..BollingerBandsParams::default()
        };
        let input_partial2 = BollingerBandsInput::from_candles(&candles, "hl2", partial_params2);
        let output_partial2 = bollinger_bands(&input_partial2)
            .expect("Failed BB with partial params (EMA, period=10)");
        assert_eq!(output_partial2.middle_band.len(), candles.close.len());
    }

    #[test]
    fn test_bollinger_bands_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = BollingerBandsInput::with_default_candles(&candles);
        let result = bollinger_bands(&input).expect("Failed to calculate Bollinger Bands");

        let expected_middle = [
            59403.199999999975,
            59423.24999999998,
            59370.49999999998,
            59371.39999999998,
            59351.299999999974,
        ];
        let expected_lower = [
            58299.51497247008,
            58351.47038179873,
            58332.65135978715,
            58334.33194052157,
            58275.767369163135,
        ];
        let expected_upper = [
            60506.88502752987,
            60495.029618201224,
            60408.348640212804,
            60408.468059478386,
            60426.83263083681,
        ];

        assert!(
            result.middle_band.len() >= 5,
            "Not enough data for final 5-values check."
        );
        let start_idx = result.middle_band.len() - 5;

        for i in 0..5 {
            let actual_mid = result.middle_band[start_idx + i];
            let actual_low = result.lower_band[start_idx + i];
            let actual_up = result.upper_band[start_idx + i];

            assert!(
                (actual_mid - expected_middle[i]).abs() < 1e-4,
                "Mismatch in middle band at i={}: expected={}, got={}",
                i,
                expected_middle[i],
                actual_mid
            );
            assert!(
                (actual_low - expected_lower[i]).abs() < 1e-4,
                "Mismatch in lower band at i={}: expected={}, got={}",
                i,
                expected_lower[i],
                actual_low
            );
            assert!(
                (actual_up - expected_upper[i]).abs() < 1e-4,
                "Mismatch in upper band at i={}: expected={}, got={}",
                i,
                expected_upper[i],
                actual_up
            );
        }
    }

    #[test]
    fn test_bollinger_bands_params_with_default_params() {
        let default_params = BollingerBandsParams::default();
        assert_eq!(
            default_params.period,
            Some(20),
            "Expected default period=20"
        );
        assert_eq!(
            default_params.devup,
            Some(2.0),
            "Expected default devup=2.0"
        );
        assert_eq!(
            default_params.devdn,
            Some(2.0),
            "Expected default devdn=2.0"
        );
        assert_eq!(
            default_params.matype,
            Some("sma".to_string()),
            "Expected default matype='sma'"
        );
        assert_eq!(
            default_params.devtype,
            Some(0),
            "Expected default devtype=0 (std dev)"
        );
    }

    #[test]
    fn test_bollinger_bands_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = BollingerBandsInput::with_default_candles(&candles);
        match input.data {
            BollingerBandsData::Candles { source, .. } => {
                assert_eq!(
                    source, "close",
                    "Expected default source to be 'close' for BollingerBandsInput"
                );
            }
            _ => panic!("Expected BollingerBandsData::Candles variant"),
        }
    }

    #[test]
    fn test_bollinger_bands_with_zero_period() {
        let data = [10.0, 20.0, 30.0];
        let params = BollingerBandsParams {
            period: Some(0),
            ..BollingerBandsParams::default()
        };
        let input = BollingerBandsInput::from_slice(&data, params);

        let result = bollinger_bands(&input);
        assert!(result.is_err(), "Expected an error for zero period");
        if let Err(e) = result {
            assert!(
                e.to_string().contains("Invalid period"),
                "Expected 'Invalid period' error message, got: {}",
                e
            );
        }
    }

    #[test]
    fn test_bollinger_bands_with_period_exceeding_data_length() {
        let data = [10.0, 20.0, 30.0];
        let params = BollingerBandsParams {
            period: Some(10),
            ..BollingerBandsParams::default()
        };
        let input = BollingerBandsInput::from_slice(&data, params);

        let result = bollinger_bands(&input);
        assert!(result.is_err(), "Expected error for period > data.len()");
    }

    #[test]
    fn test_bollinger_bands_very_small_data_set() {
        let data = [42.0];
        let input = BollingerBandsInput::from_slice(&data, BollingerBandsParams::default());

        let result = bollinger_bands(&input);
        assert!(
            result.is_err(),
            "Expected error for data smaller than default period=20"
        );
    }

    #[test]
    fn test_bollinger_bands_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let first_params = BollingerBandsParams {
            period: Some(20),
            ..BollingerBandsParams::default()
        };
        let first_input =
            BollingerBandsInput::from_candles(&candles, "close", first_params.clone());
        let first_result =
            bollinger_bands(&first_input).expect("Failed to calculate Bollinger Bands (first)");

        assert_eq!(
            first_result.middle_band.len(),
            candles.close.len(),
            "First BB output length mismatch"
        );

        let second_params = BollingerBandsParams {
            period: Some(10),
            ..BollingerBandsParams::default()
        };
        let second_input =
            BollingerBandsInput::from_slice(&first_result.middle_band, second_params.clone());
        let second_result =
            bollinger_bands(&second_input).expect("Failed to calculate Bollinger Bands (second)");

        assert_eq!(
            second_result.middle_band.len(),
            first_result.middle_band.len(),
            "Second BB output length mismatch"
        );
    }

    #[test]
    fn test_bollinger_bands_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let period = 20;
        let params = BollingerBandsParams {
            period: Some(period),
            ..BollingerBandsParams::default()
        };
        let input = BollingerBandsInput::from_candles(&candles, "close", params);
        let result = bollinger_bands(&input).expect("Failed to calculate Bollinger Bands");

        let check_index = 240;
        if result.middle_band.len() > check_index {
            for i in check_index..result.middle_band.len() {
                assert!(
                    !result.middle_band[i].is_nan(),
                    "Expected no NaN in middle band at i={}",
                    i
                );
                assert!(
                    !result.upper_band[i].is_nan(),
                    "Expected no NaN in upper band at i={}",
                    i
                );
                assert!(
                    !result.lower_band[i].is_nan(),
                    "Expected no NaN in lower band at i={}",
                    i
                );
            }
        }
    }
}
