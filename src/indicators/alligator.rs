use std::error::Error;

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
        AlligatorParams {
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
    pub data: &'a [f64],
    pub params: AlligatorParams,
}

impl<'a> AlligatorInput<'a> {
    pub fn new(data: &'a [f64], params: AlligatorParams) -> Self {
        AlligatorInput { data, params }
    }

    pub fn with_default_params(data: &'a [f64]) -> Self {
        AlligatorInput {
            data,
            params: AlligatorParams::default(),
        }
    }

    fn get_jaw_period(&self) -> usize {
        self.params
            .jaw_period
            .unwrap_or_else(|| AlligatorParams::default().jaw_period.unwrap())
    }

    fn get_jaw_offset(&self) -> usize {
        self.params
            .jaw_offset
            .unwrap_or_else(|| AlligatorParams::default().jaw_offset.unwrap())
    }

    fn get_teeth_period(&self) -> usize {
        self.params
            .teeth_period
            .unwrap_or_else(|| AlligatorParams::default().teeth_period.unwrap())
    }

    fn get_teeth_offset(&self) -> usize {
        self.params
            .teeth_offset
            .unwrap_or_else(|| AlligatorParams::default().teeth_offset.unwrap())
    }

    fn get_lips_period(&self) -> usize {
        self.params
            .lips_period
            .unwrap_or_else(|| AlligatorParams::default().lips_period.unwrap())
    }

    fn get_lips_offset(&self) -> usize {
        self.params
            .lips_offset
            .unwrap_or_else(|| AlligatorParams::default().lips_offset.unwrap())
    }
}

#[derive(Debug, Clone)]
pub struct AlligatorOutput {
    pub jaw: Vec<f64>,
    pub teeth: Vec<f64>,
    pub lips: Vec<f64>,
}

#[inline]
pub fn calculate_alligator(input: &AlligatorInput) -> Result<AlligatorOutput, Box<dyn Error>> {
    let data = input.data;
    let len = data.len();

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

    for i in 0..len {
        let data_point = data[i];

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
    use crate::indicators::data_loader::read_candles_from_csv;

    #[test]
    fn test_alligator_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let hl2_prices: Vec<f64> = candles
            .get_calculated_field("hl2")
            .expect("Failed to extract hl2 prices");

        let input = AlligatorInput::with_default_params(&hl2_prices);
        let result = calculate_alligator(&input).expect("Failed to calculate alligator");

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
        let custom_input = AlligatorInput::new(&hl2_prices, custom_params);
        let _ = calculate_alligator(&custom_input)
            .expect("Alligator calculation with custom params failed");
    }
}
