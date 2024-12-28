use crate::utilities::data_loader::{source_type, Candles};
use std::error::Error;

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
    pub candles: &'a Candles,
    pub source: &'a str,
    pub params: VpwmaParams,
}

impl<'a> VpwmaInput<'a> {
    pub fn new(candles: &'a Candles, source: &'a str, params: VpwmaParams) -> Self {
        Self {
            candles,
            source,
            params,
        }
    }

    pub fn with_default_params(candles: &'a Candles) -> Self {
        Self {
            candles,
            source: "close",
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
#[inline]
pub fn vpwma(input: &VpwmaInput) -> Result<VpwmaOutput, Box<dyn Error>> {
    let data: &[f64] = source_type(input.candles, input.source);
    let period: usize = input.get_period();
    let power: f64 = input.get_power();
    let len: usize = data.len();
    if len < period + 1 {
        return Err(format!("Not enough data: length {} < period+1={}", len, period + 1).into());
    }
    if period < 2 {
        return Err("VPWMA period must be >= 2.".into());
    }
    if power.is_nan() {
        return Err("VPWMA power cannot be NaN.".into());
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
        let input = VpwmaInput::new(&candles, "close", params);
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
    fn test_vpwma_with_defaults() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = VpwmaInput::with_default_params(&candles);
        let result = vpwma(&input);
        assert!(result.is_err(), "Should fail due to insufficient data");
    }

    #[test]
    fn test_vpwma_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let default_params = VpwmaParams {
            period: None,
            power: None,
        };
        let input = VpwmaInput::new(&candles, "close", default_params);
        let output = vpwma(&input).expect("Failed VPWMA with default params");
        assert_eq!(output.values.len(), candles.close.len());
        let params_period_14 = VpwmaParams {
            period: Some(14),
            power: None,
        };
        let input2 = VpwmaInput::new(&candles, "hl2", params_period_14);
        let output2 = vpwma(&input2).expect("Failed VPWMA with period=14, source=hl2");
        assert_eq!(output2.values.len(), candles.close.len());
        let params_custom = VpwmaParams {
            period: Some(10),
            power: Some(0.5),
        };
        let input3 = VpwmaInput::new(&candles, "hlc3", params_custom);
        let output3 = vpwma(&input3).expect("Failed VPWMA fully custom");
        assert_eq!(output3.values.len(), candles.close.len());
    }
}
