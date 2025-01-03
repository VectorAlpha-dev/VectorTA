/// # Variable Power Weighted Moving Average (VPWMA)
///
/// The Variable Power Weighted Moving Average (VPWMA) adjusts the weights of each
/// price data point in its calculation based on their respective volumes. This
/// means that periods with higher trading volumes have a greater influence on
/// the moving average. By raising the weight to a specified power (`power`),
/// one can control how aggressively recent, high-volume data points dominate
/// the resulting average.
///
/// ## Parameters
/// - **period**: Number of data points in each calculation window (defaults to 14).
/// - **power**: Exponent applied to the volume-based weight function. Higher
///   values give more impact to recent, higher-volume data (defaults to 0.382).
///
/// ## Errors
/// - If `period < 2`, an error is returned (`"VPWMA period must be >= 2."`).
/// - If `power` is `NaN`, an error is returned (`"VPWMA power cannot be NaN."`).
/// - If the data length is less than `period + 1`, an error is returned indicating
///   insufficient data.
///
/// ## Returns
/// - A `Vec<f64>` matching the input length, with leading elements unchanged
///   from the original data slice (or candles), and subsequent elements replaced
///   by the VPWMA.
///
/// # Example
/// ```
/// // Assuming `candles` is a valid Candles structure with volume data
/// let params = VpwmaParams { period: Some(14), power: Some(0.382) };
/// let input = VpwmaInput::from_candles(&candles, "close", params);
/// let result = vpwma(&input).unwrap();
/// println!("VPWMA output: {:?}", result.values);
/// ```
use crate::utilities::data_loader::{source_type, Candles};

#[derive(Debug, Clone)]
pub enum VpwmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct VpwmaParams {
    pub period: Option<usize>,
    pub power: Option<f64>,
}

impl Default for VpwmaParams {
    fn default() -> Self {
        Self {
            period: Some(14),
            power: Some(0.382),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VpwmaInput<'a> {
    pub data: VpwmaData<'a>,
    pub params: VpwmaParams,
}

impl<'a> VpwmaInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: VpwmaParams) -> Self {
        Self {
            data: VpwmaData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: VpwmaParams) -> Self {
        Self {
            data: VpwmaData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: VpwmaData::Candles {
                candles,
                source: "close",
            },
            params: VpwmaParams::default(),
        }
    }

    fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| VpwmaParams::default().period.unwrap())
    }

    fn get_power(&self) -> f64 {
        self.params
            .power
            .unwrap_or_else(|| VpwmaParams::default().power.unwrap())
    }
}

#[derive(Debug, Clone)]
pub struct VpwmaOutput {
    pub values: Vec<f64>,
}

use thiserror::Error;

#[derive(Debug, Error)]
pub enum VpwmaError {
    #[error("Input data is empty, cannot compute VPWMA.")]
    EmptyData,
    #[error("vpwma: Not enough data: length {data_len} < period+1={period_plus_1}")]
    NotEnoughData {
        data_len: usize,
        period_plus_1: usize,
    },
    #[error("VPWMA period must be >= 2. period = {period}")]
    InvalidPeriod { period: usize },
    #[error("VPWMA power cannot be NaN or infinite. power = {power}")]
    InvalidPower { power: f64 },
}

#[inline]
pub fn vpwma(input: &VpwmaInput) -> Result<VpwmaOutput, VpwmaError> {
    let data: &[f64] = match &input.data {
        VpwmaData::Candles { candles, source } => source_type(candles, source),
        VpwmaData::Slice(slice) => slice,
    };

    if data.is_empty() {
        return Err(VpwmaError::EmptyData);
    }

    let period = input.get_period();
    let power = input.get_power();
    let len = data.len();

    if len < period + 1 {
        return Err(VpwmaError::NotEnoughData {
            data_len: len,
            period_plus_1: period + 1,
        });
    }

    if period < 2 {
        return Err(VpwmaError::InvalidPeriod { period });
    }

    if power.is_nan() || power.is_infinite() {
        return Err(VpwmaError::InvalidPower { power });
    }

    let mut vpwma_values = data.to_vec();

    let mut weights = Vec::with_capacity(period - 1);
    for i in 0..(period - 1) {
        let w = (period as f64 - i as f64).powf(power);
        weights.push(w);
    }
    let weight_sum: f64 = weights.iter().sum();

    for j in (period + 1)..len {
        let mut my_sum = 0.0;
        for (i, &w) in weights.iter().enumerate() {
            my_sum = data[j - i].mul_add(w, my_sum);
        }
        vpwma_values[j] = my_sum / weight_sum;
    }

    Ok(VpwmaOutput {
        values: vpwma_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_vpwma_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close");
        let params = VpwmaParams {
            period: Some(14),
            power: Some(0.382),
        };
        let input = VpwmaInput::from_candles(&candles, "close", params);
        let result = vpwma(&input).expect("Failed to calculate VPWMA");
        assert_eq!(result.values.len(), close_prices.len());
        let expected_last_five = [
            59363.927599446455,
            59296.83894519251,
            59196.82476139941,
            59180.8040249446,
            59113.84473799056,
        ];
        let start_index = result.values.len().saturating_sub(5);
        let last_five = &result.values[start_index..];
        for (i, &val) in last_five.iter().enumerate() {
            let exp = expected_last_five[i];
            let diff = (val - exp).abs();
            assert!(
                diff < 1e-2,
                "VPWMA mismatch at index {}: expected {}, got {}",
                i,
                exp,
                val
            );
        }
    }

    #[test]
    fn test_vpwma_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let default_params = VpwmaParams {
            period: None,
            power: None,
        };
        let input = VpwmaInput::from_candles(&candles, "close", default_params);
        let output = vpwma(&input).expect("Failed VPWMA with default params");
        assert_eq!(output.values.len(), candles.close.len());
        let params_period_14 = VpwmaParams {
            period: Some(14),
            power: None,
        };
        let input2 = VpwmaInput::from_candles(&candles, "hl2", params_period_14);
        let output2 = vpwma(&input2).expect("Failed VPWMA with period=14, source=hl2");
        assert_eq!(output2.values.len(), candles.close.len());
        let params_custom = VpwmaParams {
            period: Some(10),
            power: Some(0.5),
        };
        let input3 = VpwmaInput::from_candles(&candles, "hlc3", params_custom);
        let output3 = vpwma(&input3).expect("Failed VPWMA fully custom");
        assert_eq!(output3.values.len(), candles.close.len());
    }
    #[test]
    fn test_vpwma_params_with_default() {
        let default_params = VpwmaParams::default();
        assert_eq!(default_params.period, Some(14));
        assert_eq!(default_params.power, Some(0.382));
    }

    #[test]
    fn test_vpwma_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = VpwmaInput::with_default_candles(&candles);
        match input.data {
            VpwmaData::Candles { source, .. } => {
                assert_eq!(source, "close");
            }
            _ => panic!("Expected VpwmaData::Candles variant"),
        }
        assert_eq!(input.params.period, Some(14));
        assert_eq!(input.params.power, Some(0.382));
    }

    #[test]
    fn test_vpwma_insufficient_data() {
        let data = [42.0, 43.0, 44.0];
        let params = VpwmaParams {
            period: Some(5),
            power: Some(0.382),
        };
        let input = VpwmaInput::from_slice(&data, params);
        let result = vpwma(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_vpwma_with_invalid_period() {
        let data = [10.0, 20.0, 30.0, 40.0, 50.0];
        let params = VpwmaParams {
            period: Some(1),
            power: Some(0.382),
        };
        let input = VpwmaInput::from_slice(&data, params);
        let result = vpwma(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("VPWMA period must be >= 2"));
        }
    }

    #[test]
    fn test_vpwma_nan_power() {
        let data = [10.0, 20.0, 30.0, 40.0, 50.0];
        let params = VpwmaParams {
            period: Some(2),
            power: Some(f64::NAN),
        };
        let input = VpwmaInput::from_slice(&data, params);
        let result = vpwma(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("power cannot be NaN"));
        }
    }

    #[test]
    fn test_vpwma_very_small_data_set() {
        let data = [100.0; 16];
        let params = VpwmaParams {
            period: Some(14),
            power: Some(0.382),
        };
        let input = VpwmaInput::from_slice(&data, params);
        let result = vpwma(&input).expect("Should handle minimal data length");
        assert_eq!(result.values.len(), data.len());
    }

    #[test]
    fn test_vpwma_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let first_params = VpwmaParams {
            period: Some(14),
            power: Some(0.382),
        };
        let first_input = VpwmaInput::from_candles(&candles, "close", first_params);
        let first_result = vpwma(&first_input).expect("Failed to calculate first VPWMA");
        assert_eq!(first_result.values.len(), candles.close.len());

        let second_params = VpwmaParams {
            period: Some(5),
            power: Some(0.5),
        };
        let second_input = VpwmaInput::from_slice(&first_result.values, second_params);
        let second_result = vpwma(&second_input).expect("Failed to calculate second VPWMA");
        assert_eq!(second_result.values.len(), first_result.values.len());
        if second_result.values.len() > 240 {
            for i in 240..second_result.values.len() {
                assert!(!second_result.values[i].is_nan());
            }
        }
    }

    #[test]
    fn test_vpwma_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = VpwmaParams {
            period: Some(14),
            power: Some(0.382),
        };
        let input = VpwmaInput::from_candles(&candles, "close", params);
        let result = vpwma(&input).expect("Failed to calculate VPWMA");
        assert_eq!(result.values.len(), candles.close.len());
        if result.values.len() > 50 {
            for i in 50..result.values.len() {
                assert!(!result.values[i].is_nan());
            }
        }
    }
}
