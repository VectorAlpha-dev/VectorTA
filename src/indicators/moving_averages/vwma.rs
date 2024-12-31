use crate::utilities::data_loader::{source_type, Candles};
use std::error::Error;

#[derive(Debug, Clone)]
pub enum VwmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    CandlesPlusPrices {
        candles: &'a Candles,
        prices: &'a [f64],
    },
}

trait CandlesRef<'a> {
    fn match_candles(&'a self) -> &'a Candles;
}

impl<'a> CandlesRef<'a> for VwmaInput<'a> {
    fn match_candles(&'a self) -> &'a Candles {
        match &self.data {
            VwmaData::Candles { candles, .. } => candles,
            VwmaData::CandlesPlusPrices { candles, .. } => candles,
        }
    }
}

#[derive(Debug, Clone)]
pub struct VwmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct VwmaParams {
    pub period: Option<usize>,
}

impl Default for VwmaParams {
    fn default() -> Self {
        Self { period: Some(20) }
    }
}

#[derive(Debug, Clone)]
pub struct VwmaInput<'a> {
    pub data: VwmaData<'a>,
    pub params: VwmaParams,
}

impl<'a> VwmaInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: VwmaParams) -> Self {
        Self {
            data: VwmaData::Candles { candles, source },
            params,
        }
    }

    pub fn from_candles_plus_prices(
        candles: &'a Candles,
        prices: &'a [f64],
        params: VwmaParams,
    ) -> Self {
        Self {
            data: VwmaData::CandlesPlusPrices { candles, prices },
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: VwmaData::Candles {
                candles,
                source: "close",
            },
            params: VwmaParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| VwmaParams::default().period.unwrap())
    }
}

#[inline]
pub fn vwma(input: &VwmaInput) -> Result<VwmaOutput, Box<dyn Error>> {
    let volume: &[f64] = input.match_candles().select_candle_field("volume")?;

    let price: &[f64] = match &input.data {
        VwmaData::Candles { candles, source } => source_type(candles, source),
        VwmaData::CandlesPlusPrices { prices, .. } => prices,
    };
    let len: usize = price.len();
    let period: usize = input.get_period();

    if period == 0 || period > len {
        return Err("Invalid period for VWMA calculation.".into());
    }
    if len != volume.len() {
        return Err("Price and volume mismatch.".into());
    }

    let mut vwma_values = vec![f64::NAN; len];

    let mut sum = 0.0;
    let mut vsum = 0.0;

    for i in 0..period {
        sum += price[i] * volume[i];
        vsum += volume[i];
    }
    vwma_values[period - 1] = sum / vsum;

    for i in period..len {
        sum += price[i] * volume[i];
        sum -= price[i - period] * volume[i - period];

        vsum += volume[i];
        vsum -= volume[i - period];

        vwma_values[i] = sum / vsum;
    }

    Ok(VwmaOutput {
        values: vwma_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_vwma_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = VwmaParams { period: None };
        let input_default = VwmaInput::from_candles(&candles, "close", default_params);
        let output_default = vwma(&input_default).expect("Failed VWMA with default params");
        assert_eq!(output_default.values.len(), candles.close.len());

        let custom_params = VwmaParams { period: Some(10) };
        let input_custom = VwmaInput::from_candles(&candles, "hlc3", custom_params);
        let output_custom = vwma(&input_custom).expect("Failed VWMA with period=10, source=hlc3");
        assert_eq!(output_custom.values.len(), candles.close.len());
    }

    #[test]
    fn test_vwma_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let params = VwmaParams { period: Some(20) };
        let input = VwmaInput::from_candles(&candles, "close", params);
        let vwma_result = vwma(&input).expect("Failed to calculate VWMA");
        assert_eq!(vwma_result.values.len(), close_prices.len());

        let expected_last_five_vwma = [
            59201.87047121331,
            59217.157390630266,
            59195.74526905522,
            59196.261392450084,
            59151.22059588594,
        ];
        assert!(vwma_result.values.len() >= 5);
        let start_index = vwma_result.values.len() - 5;
        let result_last_five_vwma = &vwma_result.values[start_index..];
        for (i, &val) in result_last_five_vwma.iter().enumerate() {
            let exp = expected_last_five_vwma[i];
            assert!(
                (val - exp).abs() < 1e-3,
                "VWMA mismatch at index {}: expected {}, got {}",
                i,
                exp,
                val
            );
        }
    }
    #[test]
    fn test_vwma_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = VwmaInput::with_default_candles(&candles);
        match input.data {
            VwmaData::Candles { source, .. } => assert_eq!(source, "close"),
            VwmaData::CandlesPlusPrices { .. } => panic!("Expected VwmaData::Candles"),
        }
    }

    #[test]
    fn test_vwma_candles_plus_prices() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let custom_prices = candles
            .close
            .iter()
            .map(|v| v * 1.001)
            .collect::<Vec<f64>>();
        let params = VwmaParams { period: Some(20) };
        let input = VwmaInput::from_candles_plus_prices(&candles, &custom_prices, params);
        let result = vwma(&input).expect("VWMA on custom prices");
        assert_eq!(result.values.len(), custom_prices.len());
    }

    #[test]
    fn test_vwma_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let params_first = VwmaParams { period: Some(20) };
        let input_first = VwmaInput::from_candles(&candles, "close", params_first);
        let result_first = vwma(&input_first).expect("First pass VWMA failed");
        assert_eq!(result_first.values.len(), candles.close.len());

        let params_second = VwmaParams { period: Some(10) };
        let input_second =
            VwmaInput::from_candles_plus_prices(&candles, &result_first.values, params_second);
        let result_second = vwma(&input_second).expect("Second pass VWMA failed");
        assert_eq!(result_second.values.len(), result_first.values.len());
        for i in 240..result_second.values.len() {
            assert!(!result_second.values[i].is_nan());
        }
    }
}
