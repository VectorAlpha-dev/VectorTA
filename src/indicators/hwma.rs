use std::error::Error;

#[derive(Debug, Clone)]
pub struct HwmaParams {
    pub na: f64,
    pub nb: f64,
    pub nc: f64,
}

impl Default for HwmaParams {
    fn default() -> Self {
        HwmaParams {
            na: 0.2,
            nb: 0.1,
            nc: 0.1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HwmaInput<'a> {
    pub data: &'a [f64],
    pub params: HwmaParams,
}

impl<'a> HwmaInput<'a> {
    pub fn new(data: &'a [f64], params: HwmaParams) -> Self {
        HwmaInput { data, params }
    }

    pub fn with_default_params(data: &'a [f64]) -> Self {
        HwmaInput {
            data,
            params: HwmaParams::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct HwmaOutput {
    pub values: Vec<f64>,
}

#[inline]
pub fn calculate_hwma(input: &HwmaInput) -> Result<HwmaOutput, Box<dyn Error>> {
    let HwmaParams { na, nb, nc } = input.params;

    if !(na > 0.0 && na < 1.0 && nb > 0.0 && nb < 1.0 && nc > 0.0 && nc < 1.0) {
        return Err("Parameters (na, nb, nc) must be in (0,1).".into());
    }
    let data = input.data;
    let len = data.len();
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
    fn test_hwma_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");
        let params = HwmaParams {
            na: 0.2,
            nb: 0.1,
            nc: 0.1,
        };
        let input = HwmaInput::new(close_prices, params);
        let result = calculate_hwma(&input).expect("Failed to calculate HWMA");

        let len = result.values.len();
        assert!(len > 5, "Not enough data to compare last 5 values.");

        let expected_last_five = [
            57941.04005793378,
            58106.90324194954,
            58250.474156632234,
            58428.90005831887,
            58499.37021151028,
        ];

        let start_index = len - 5;
        let actual_last_five = &result.values[start_index..];

        for (i, &actual) in actual_last_five.iter().enumerate() {
            let expected = expected_last_five[i];
            assert!(
                (actual - expected).abs() < 1e-3,
                "HWMA value mismatch at index {}: expected {}, got {}",
                i,
                expected,
                actual
            );
        }
    }
}
