use crate::utilities::data_loader::{source_type, Candles};
use std::error::Error;

#[derive(Debug, Clone)]
pub enum AlligatorData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct AlligatorParams {
    pub jaw_period: Option<usize>,
    pub jaw_offset: Option<usize>,
    pub teeth_period: Option<usize>,
    pub teeth_offset: Option<usize>,
    pub lips_period: Option<usize>,
    pub lips_offset: Option<usize>,
}

impl Default for AlligatorParams {
    fn default() -> Self {
        Self {
            jaw_period: Some(13),
            jaw_offset: Some(8),
            teeth_period: Some(8),
            teeth_offset: Some(5),
            lips_period: Some(5),
            lips_offset: Some(3),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AlligatorInput<'a> {
    pub data: AlligatorData<'a>,
    pub params: AlligatorParams,
}

impl<'a> AlligatorInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: AlligatorParams) -> Self {
        Self {
            data: AlligatorData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: AlligatorParams) -> Self {
        Self {
            data: AlligatorData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: AlligatorData::Candles {
                candles,
                source: "hl2",
            },
            params: AlligatorParams::default(),
        }
    }

    fn get_jaw_period(&self) -> usize {
        self.params.jaw_period.unwrap_or(13)
    }

    fn get_jaw_offset(&self) -> usize {
        self.params.jaw_offset.unwrap_or(8)
    }

    fn get_teeth_period(&self) -> usize {
        self.params.teeth_period.unwrap_or(8)
    }

    fn get_teeth_offset(&self) -> usize {
        self.params.teeth_offset.unwrap_or(5)
    }

    fn get_lips_period(&self) -> usize {
        self.params.lips_period.unwrap_or(5)
    }

    fn get_lips_offset(&self) -> usize {
        self.params.lips_offset.unwrap_or(3)
    }
}

#[derive(Debug, Clone)]
pub struct AlligatorOutput {
    pub jaw: Vec<f64>,
    pub teeth: Vec<f64>,
    pub lips: Vec<f64>,
}

#[inline]
pub fn alligator(input: &AlligatorInput) -> Result<AlligatorOutput, Box<dyn Error>> {
    let data: &[f64] = match &input.data {
        AlligatorData::Candles { candles, source } => source_type(candles, source),
        AlligatorData::Slice(slice) => slice,
    };
    let len: usize = data.len();

    let jaw_period = input.get_jaw_period();
    let jaw_offset = input.get_jaw_offset();
    let teeth_period = input.get_teeth_period();
    let teeth_offset = input.get_teeth_offset();
    let lips_period = input.get_lips_period();
    let lips_offset = input.get_lips_offset();

    let mut jaw = vec![f64::NAN; len];
    let mut teeth = vec![f64::NAN; len];
    let mut lips = vec![f64::NAN; len];

    let mut jaw_sum = 0.0;
    let mut teeth_sum = 0.0;
    let mut lips_sum = 0.0;

    let mut jaw_smma_val = 0.0;
    let mut teeth_smma_val = 0.0;
    let mut lips_smma_val = 0.0;

    let mut jaw_ready = false;
    let mut teeth_ready = false;
    let mut lips_ready = false;

    let jaw_scale = (jaw_period - 1) as f64;
    let jaw_inv_period = 1.0 / jaw_period as f64;

    let teeth_scale = (teeth_period - 1) as f64;
    let teeth_inv_period = 1.0 / teeth_period as f64;

    let lips_scale = (lips_period - 1) as f64;
    let lips_inv_period = 1.0 / lips_period as f64;

    for (i, &data_point) in data.iter().enumerate() {
        if !jaw_ready {
            if i < jaw_period {
                jaw_sum += data_point;
                if i == jaw_period - 1 {
                    jaw_smma_val = jaw_sum / (jaw_period as f64);
                    jaw_ready = true;
                    let shifted_index = i + jaw_offset;
                    if shifted_index < len {
                        jaw[shifted_index] = jaw_smma_val;
                    }
                }
            }
        } else {
            jaw_smma_val = (jaw_smma_val * jaw_scale + data_point) * jaw_inv_period;
            let shifted_index = i + jaw_offset;
            if shifted_index < len {
                jaw[shifted_index] = jaw_smma_val;
            }
        }

        if !teeth_ready {
            if i < teeth_period {
                teeth_sum += data_point;
                if i == teeth_period - 1 {
                    teeth_smma_val = teeth_sum / (teeth_period as f64);
                    teeth_ready = true;
                    let shifted_index = i + teeth_offset;
                    if shifted_index < len {
                        teeth[shifted_index] = teeth_smma_val;
                    }
                }
            }
        } else {
            teeth_smma_val = (teeth_smma_val * teeth_scale + data_point) * teeth_inv_period;
            let shifted_index = i + teeth_offset;
            if shifted_index < len {
                teeth[shifted_index] = teeth_smma_val;
            }
        }

        if !lips_ready {
            if i < lips_period {
                lips_sum += data_point;
                if i == lips_period - 1 {
                    lips_smma_val = lips_sum / (lips_period as f64);
                    lips_ready = true;
                    let shifted_index = i + lips_offset;
                    if shifted_index < len {
                        lips[shifted_index] = lips_smma_val;
                    }
                }
            }
        } else {
            lips_smma_val = (lips_smma_val * lips_scale + data_point) * lips_inv_period;
            let shifted_index = i + lips_offset;
            if shifted_index < len {
                lips[shifted_index] = lips_smma_val;
            }
        }
    }

    Ok(AlligatorOutput { jaw, teeth, lips })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_alligator_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let partial_params = AlligatorParams {
            jaw_period: Some(14),
            jaw_offset: None,
            teeth_period: None,
            teeth_offset: None,
            lips_period: None,
            lips_offset: Some(2),
        };
        let input = AlligatorInput::from_candles(&candles, "hl2", partial_params);
        let result = alligator(&input).expect("Failed to calculate alligator with partial params");
        assert_eq!(result.jaw.len(), candles.close.len());
        assert_eq!(result.teeth.len(), candles.close.len());
        assert_eq!(result.lips.len(), candles.close.len());
    }

    #[test]
    fn test_alligator_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let hl2_prices = candles
            .get_calculated_field("hl2")
            .expect("Failed to extract hl2 prices");

        let input = AlligatorInput::with_default_candles(&candles);
        let result = alligator(&input).expect("Failed to calculate alligator");

        let expected_last_five_jaw_result = [60742.4, 60632.6, 60555.1, 60442.7, 60308.7];
        let expected_last_five_teeth_result = [59908.0, 59757.2, 59684.3, 59653.5, 59621.1];
        let expected_last_five_lips_result = [59355.2, 59371.7, 59376.2, 59334.1, 59316.2];

        let start_index: usize = result.jaw.len() - 5;
        let result_last_five_jaws = &result.jaw[start_index..];
        let result_last_five_teeth = &result.teeth[start_index..];
        let result_last_five_lips = &result.lips[start_index..];

        assert_eq!(
            result.jaw.len(),
            hl2_prices.len(),
            "Alligator jaw output length does not match input length"
        );

        assert_eq!(
            result.teeth.len(),
            hl2_prices.len(),
            "Alligator teeth output length does not match input length"
        );

        assert_eq!(
            result.lips.len(),
            hl2_prices.len(),
            "Alligator lips output length does not match input length"
        );

        for (i, &value) in result_last_five_jaws.iter().enumerate() {
            let expected_value = expected_last_five_jaw_result[i];
            assert!(
                (value - expected_value).abs() < 1e-1,
                "alligator jaw value mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }

        for (i, &value) in result_last_five_teeth.iter().enumerate() {
            let expected_value = expected_last_five_teeth_result[i];
            assert!(
                (value - expected_value).abs() < 1e-1,
                "alligator teeth value mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }

        for (i, &value) in result_last_five_lips.iter().enumerate() {
            let expected_value = expected_last_five_lips_result[i];
            assert!(
                (value - expected_value).abs() < 1e-1,
                "alligator lips value mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }

        let custom_params = AlligatorParams {
            jaw_period: Some(14),
            ..AlligatorParams::default()
        };
        let custom_input = AlligatorInput::from_candles(&candles, "hl2", custom_params);
        let _ = alligator(&custom_input).expect("Alligator calculation with custom params failed");
    }
    #[test]
    fn test_alligator_params_with_default_params() {
        let default_params = AlligatorParams::default();
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = AlligatorInput::from_candles(&candles, "hl2", default_params);
        let result = alligator(&input).expect("Failed to calculate alligator");
        assert_eq!(result.jaw.len(), candles.close.len());
        assert_eq!(result.teeth.len(), candles.close.len());
        assert_eq!(result.lips.len(), candles.close.len());
    }

    #[test]
    fn test_alligator_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = AlligatorInput::with_default_candles(&candles);
        match input.data {
            AlligatorData::Candles { source, .. } => {
                assert_eq!(source, "hl2");
            }
            _ => panic!("Expected AlligatorData::Candles variant"),
        }
        assert!(input.params.jaw_period.is_some());
        assert!(input.params.jaw_offset.is_some());
        assert!(input.params.teeth_period.is_some());
        assert!(input.params.teeth_offset.is_some());
        assert!(input.params.lips_period.is_some());
        assert!(input.params.lips_offset.is_some());
    }

    #[test]
    fn test_alligator_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let first_input = AlligatorInput::with_default_candles(&candles);
        let first_result = alligator(&first_input).expect("Failed to calculate first alligator");
        let second_input =
            AlligatorInput::from_slice(&first_result.jaw, AlligatorParams::default());
        let second_result = alligator(&second_input).expect("Failed to calculate second alligator");
        assert_eq!(second_result.jaw.len(), first_result.jaw.len());
        assert_eq!(second_result.teeth.len(), first_result.teeth.len());
        assert_eq!(second_result.lips.len(), first_result.lips.len());
        for i in 240..second_result.jaw.len() {
            assert!(!second_result.jaw[i].is_nan());
            assert!(!second_result.teeth[i].is_nan());
            assert!(!second_result.lips[i].is_nan());
        }
    }

    #[test]
    fn test_alligator_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = AlligatorInput::with_default_candles(&candles);
        let result = alligator(&input).expect("Failed to calculate alligator");
        if result.jaw.len() > 50 {
            for i in 50..result.jaw.len() {
                assert!(
                    !result.jaw[i].is_nan(),
                    "Expected no NaN in jaw after index {}, found NaN",
                    i
                );
                assert!(
                    !result.teeth[i].is_nan(),
                    "Expected no NaN in teeth after index {}, found NaN",
                    i
                );
                assert!(
                    !result.lips[i].is_nan(),
                    "Expected no NaN in lips after index {}, found NaN",
                    i
                );
            }
        }
    }
}
