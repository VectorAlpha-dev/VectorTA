use crate::utilities::data_loader::{source_type, Candles};
use std::error::Error;

#[derive(Debug, Clone)]
pub enum HwmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct HwmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct HwmaParams {
    pub na: Option<f64>,
    pub nb: Option<f64>,
    pub nc: Option<f64>,
}

impl HwmaParams {
    pub fn with_default_params() -> Self {
        HwmaParams {
            na: None,
            nb: None,
            nc: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HwmaInput<'a> {
    pub data: HwmaData<'a>,
    pub params: HwmaParams,
}

impl<'a> HwmaInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: HwmaParams) -> Self {
        Self {
            data: HwmaData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: HwmaParams) -> Self {
        Self {
            data: HwmaData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: HwmaData::Candles {
                candles,
                source: "close",
            },
            params: HwmaParams::with_default_params(),
        }
    }
}

#[inline]
pub fn hwma(input: &HwmaInput) -> Result<HwmaOutput, Box<dyn Error>> {
    let data: &[f64] = match &input.data {
        HwmaData::Candles { candles, source } => source_type(candles, source),
        HwmaData::Slice(slice) => slice,
    };
    let len: usize = data.len();
    let na = input.params.na.unwrap_or(0.2);
    let nb = input.params.nb.unwrap_or(0.1);
    let nc = input.params.nc.unwrap_or(0.1);
    if !(na > 0.0 && na < 1.0 && nb > 0.0 && nb < 1.0 && nc > 0.0 && nc < 1.0) {
        return Err("Parameters (na, nb, nc) must be in (0,1).".into());
    }

    if len == 0 {
        return Err("HWMA calculation received empty data array.".into());
    }

    let mut hwma_values = Vec::with_capacity(len);

    let mut last_f = data[0];
    let mut last_v = 0.0;
    let mut last_a = 0.0;

    for &current_price in data.iter() {
        let f = (1.0 - na) * (last_f + last_v + 0.5 * last_a) + na * current_price;
        let v = (1.0 - nb) * (last_v + last_a) + nb * (f - last_f);
        let a = (1.0 - nc) * last_a + nc * (v - last_v);

        let hwma_val = f + v + 0.5 * a;
        hwma_values.push(hwma_val);

        last_f = f;
        last_v = v;
        last_a = a;
    }

    Ok(HwmaOutput {
        values: hwma_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_hwma_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params_default = HwmaParams {
            na: None,
            nb: None,
            nc: None,
        };
        let input_default = HwmaInput::from_candles(&candles, "close", params_default);
        let result_default = hwma(&input_default).expect("Failed HWMA default");
        assert_eq!(result_default.values.len(), candles.close.len());
        let params_partial = HwmaParams {
            na: Some(0.3),
            nb: None,
            nc: None,
        };
        let input_partial = HwmaInput::from_candles(&candles, "hl2", params_partial);
        let result_partial = hwma(&input_partial).expect("Failed HWMA partial");
        assert_eq!(result_partial.values.len(), candles.close.len());
        let params_custom = HwmaParams {
            na: Some(0.25),
            nb: Some(0.15),
            nc: Some(0.05),
        };
        let input_custom = HwmaInput::from_candles(&candles, "hlc3", params_custom);
        let result_custom = hwma(&input_custom).expect("Failed HWMA custom");
        assert_eq!(result_custom.values.len(), candles.close.len());
    }

    #[test]
    fn test_hwma_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");
        let params = HwmaParams {
            na: Some(0.2),
            nb: Some(0.1),
            nc: Some(0.1),
        };
        let input = HwmaInput::from_candles(&candles, "close", params);
        let result = hwma(&input).expect("Failed to calculate HWMA");
        assert!(result.values.len() > 5);
        let expected_last_five = [
            57941.04005793378,
            58106.90324194954,
            58250.474156632234,
            58428.90005831887,
            58499.37021151028,
        ];
        let start_index = result.values.len() - 5;
        let actual_last_five = &result.values[start_index..];
        assert_eq!(
            result.values.len(),
            close_prices.len(),
            "HWMA output length mismatch"
        );
        for (i, &actual) in actual_last_five.iter().enumerate() {
            let expected = expected_last_five[i];
            assert!(
                (actual - expected).abs() < 1e-3,
                "HWMA mismatch at index {}: expected {}, got {}",
                i,
                expected,
                actual
            );
        }
    }

    #[test]
    fn test_hwma_params_with_default_params() {
        let params = HwmaParams::with_default_params();
        assert_eq!(params.na, None);
        assert_eq!(params.nb, None);
        assert_eq!(params.nc, None);
    }

    #[test]
    fn test_hwma_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = HwmaInput::with_default_candles(&candles);
        match input.data {
            HwmaData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected HwmaData::Candles variant"),
        }
        assert_eq!(input.params.na, None);
        assert_eq!(input.params.nb, None);
        assert_eq!(input.params.nc, None);
    }

    #[test]
    fn test_hwma_invalid_params() {
        let data = [10.0, 20.0, 30.0];
        let params = HwmaParams {
            na: Some(-0.2),
            nb: Some(1.1),
            nc: Some(0.1),
        };
        let input = HwmaInput::from_slice(&data, params);
        let result = hwma(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_hwma_empty_data() {
        let data: [f64; 0] = [];
        let params = HwmaParams {
            na: Some(0.2),
            nb: Some(0.1),
            nc: Some(0.1),
        };
        let input = HwmaInput::from_slice(&data, params);
        let result = hwma(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_hwma_small_data() {
        let data = [42.0];
        let params = HwmaParams {
            na: Some(0.2),
            nb: Some(0.1),
            nc: Some(0.1),
        };
        let input = HwmaInput::from_slice(&data, params);
        let result = hwma(&input).expect("Should handle single data point");
        assert_eq!(result.values.len(), data.len());
        assert_eq!(result.values[0], data[0] + 0.0 + 0.5 * 0.0);
    }

    #[test]
    fn test_hwma_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params_1 = HwmaParams {
            na: Some(0.2),
            nb: Some(0.1),
            nc: Some(0.1),
        };
        let input_1 = HwmaInput::from_candles(&candles, "close", params_1);
        let result_1 = hwma(&input_1).expect("Failed first HWMA");
        assert_eq!(result_1.values.len(), candles.close.len());
        let params_2 = HwmaParams {
            na: Some(0.3),
            nb: Some(0.15),
            nc: Some(0.05),
        };
        let input_2 = HwmaInput::from_slice(&result_1.values, params_2);
        let result_2 = hwma(&input_2).expect("Failed second HWMA");
        assert_eq!(result_2.values.len(), result_1.values.len());
    }

    #[test]
    fn test_hwma_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = HwmaParams {
            na: Some(0.2),
            nb: Some(0.1),
            nc: Some(0.1),
        };
        let input = HwmaInput::from_candles(&candles, "close", params);
        let result = hwma(&input).expect("Failed to calculate HWMA");
        assert_eq!(result.values.len(), candles.close.len());
        if result.values.len() > 240 {
            for i in 240..result.values.len() {
                assert!(!result.values[i].is_nan());
            }
        }
    }
}
