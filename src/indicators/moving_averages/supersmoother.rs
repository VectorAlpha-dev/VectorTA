use crate::utilities::data_loader::{source_type, Candles};
use std::error::Error;
use std::f64::consts::PI;

#[derive(Debug, Clone)]
pub enum SuperSmootherData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct SuperSmootherOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct SuperSmootherParams {
    pub period: Option<usize>,
}

impl SuperSmootherParams {
    pub fn with_default_params() -> Self {
        Self { period: None }
    }
}

#[derive(Debug, Clone)]
pub struct SuperSmootherInput<'a> {
    pub data: SuperSmootherData<'a>,
    pub params: SuperSmootherParams,
}

impl<'a> SuperSmootherInput<'a> {
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: SuperSmootherParams,
    ) -> Self {
        Self {
            data: SuperSmootherData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: SuperSmootherParams) -> Self {
        Self {
            data: SuperSmootherData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: SuperSmootherData::Candles {
                candles,
                source: "close",
            },
            params: SuperSmootherParams::with_default_params(),
        }
    }
}

#[inline]
pub fn supersmoother(input: &SuperSmootherInput) -> Result<SuperSmootherOutput, Box<dyn Error>> {
    let data: &[f64] = match &input.data {
        SuperSmootherData::Candles { candles, source } => source_type(candles, source),
        SuperSmootherData::Slice(slice) => slice,
    };
    let period: usize = input.params.period.unwrap_or(14);

    if data.is_empty() {
        return Ok(SuperSmootherOutput { values: vec![] });
    }
    if period < 1 {
        return Err("Period must be >= 1 for Super Smoother filter.".into());
    }

    let len: usize = data.len();
    let mut output_values = vec![0.0; len];

    let a = (-1.414_f64 * PI / (period as f64)).exp();
    let a_sq = a * a;
    let b = 2.0 * a * (1.414_f64 * PI / (period as f64)).cos();
    let c = (1.0 + a_sq - b) * 0.5;

    output_values[0] = data[0];
    if len > 1 {
        output_values[1] = data[1];
    }

    for i in 2..len {
        let prev_1 = output_values[i - 1];
        let prev_2 = output_values[i - 2];
        let d_i = data[i];
        let d_im1 = data[i - 1];
        output_values[i] = c * (d_i + d_im1) + b * prev_1 - a_sq * prev_2;
    }

    Ok(SuperSmootherOutput {
        values: output_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_supersmoother_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = SuperSmootherParams { period: None };
        let input_default = SuperSmootherInput::from_candles(&candles, "close", default_params);
        let output_default =
            supersmoother(&input_default).expect("Failed supersmoother with default params");
        assert_eq!(output_default.values.len(), candles.close.len());

        let custom_params = SuperSmootherParams { period: Some(20) };
        let input_custom = SuperSmootherInput::from_candles(&candles, "hlc3", custom_params);
        let output_custom =
            supersmoother(&input_custom).expect("Failed supersmoother with period=20, source=hlc3");
        assert_eq!(output_custom.values.len(), candles.close.len());
    }

    #[test]
    fn test_supersmoother_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let params = SuperSmootherParams { period: Some(14) };
        let input = SuperSmootherInput::from_candles(&candles, "close", params);
        let result = supersmoother(&input).expect("Failed to calculate SuperSmoother");

        let out_vals = &result.values;
        let expected_last_five = [
            59140.98229179739,
            59172.03593376982,
            59179.40342783722,
            59171.22758152845,
            59127.859841077094,
        ];

        assert!(out_vals.len() >= 5);
        assert_eq!(out_vals.len(), close_prices.len());

        let start_idx = out_vals.len() - 5;
        let actual_last_five = &out_vals[start_idx..];
        for (i, (&actual, &expected)) in
            actual_last_five.iter().zip(&expected_last_five).enumerate()
        {
            let diff = (actual - expected).abs();
            assert!(
                diff < 1e-8,
                "Mismatch at index {}: expected {}, got {}, diff={}",
                i,
                expected,
                actual,
                diff
            );
        }
    }
}
