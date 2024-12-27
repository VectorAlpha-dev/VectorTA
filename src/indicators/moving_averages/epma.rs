use crate::utilities::data_loader::{source_type, Candles};
use std::error::Error;

#[derive(Debug, Clone)]
pub struct EpmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct EpmaParams {
    pub period: Option<usize>,
    pub offset: Option<usize>,
}

impl EpmaParams {
    pub fn with_default_params() -> Self {
        EpmaParams {
            period: None,
            offset: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EpmaInput<'a> {
    pub candles: &'a Candles,
    pub source: &'a str,
    pub params: EpmaParams,
}

impl<'a> EpmaInput<'a> {
    pub fn new(candles: &'a Candles, source: &'a str, params: EpmaParams) -> Self {
        EpmaInput {
            candles,
            source,
            params,
        }
    }

    pub fn with_default_params(candles: &'a Candles) -> Self {
        EpmaInput {
            candles,
            source: "close",
            params: EpmaParams::with_default_params(),
        }
    }
}

#[inline]
pub fn epma(input: &EpmaInput) -> Result<EpmaOutput, Box<dyn Error>> {
    let data: &[f64] = source_type(input.candles, input.source);
    let n: usize = data.len();
    if n == 0 {
        return Err("Empty data slice for EPMA calculation.".into());
    }

    let period = input.params.period.unwrap_or(11);
    let offset = input.params.offset.unwrap_or(4);
    if period < 2 {
        return Err("EPMA period must be >= 2.".into());
    }

    let start_index = period + offset + 1;
    if start_index >= n {
        return Ok(EpmaOutput {
            values: data.to_vec(),
        });
    }

    let mut output = data.to_vec();

    let p_minus_1 = period - 1;
    let mut weights = Vec::with_capacity(p_minus_1);

    for i in 0..p_minus_1 {
        let w_i32 = (period as i32) - (i as i32) - (offset as i32);
        let w = w_i32 as f64;
        weights.push(w);
    }

    let weight_sum: f64 = weights.iter().sum();

    for j in start_index..n {
        let mut my_sum = 0.0;
        let mut i = 0_usize;

        while i + 3 < p_minus_1 {
            my_sum += data[j - i] * weights[i];
            my_sum += data[j - (i + 1)] * weights[i + 1];
            my_sum += data[j - (i + 2)] * weights[i + 2];
            my_sum += data[j - (i + 3)] * weights[i + 3];
            i += 4;
        }
        while i < p_minus_1 {
            my_sum += data[j - i] * weights[i];
            i += 1;
        }

        output[j] = my_sum / weight_sum;
    }

    Ok(EpmaOutput { values: output })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_epma_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = EpmaParams {
            period: None,
            offset: None,
        };
        let input = EpmaInput::new(&candles, "close", default_params);
        let output = epma(&input).expect("Failed EPMA with default params");
        assert_eq!(output.values.len(), candles.close.len());

        let params_period_14 = EpmaParams {
            period: Some(14),
            offset: None,
        };
        let input2 = EpmaInput::new(&candles, "hl2", params_period_14);
        let output2 = epma(&input2).expect("Failed EPMA with period=14, source=hl2");
        assert_eq!(output2.values.len(), candles.close.len());

        let params_custom = EpmaParams {
            period: Some(10),
            offset: Some(5),
        };
        let input3 = EpmaInput::new(&candles, "hlc3", params_custom);
        let output3 = epma(&input3).expect("Failed EPMA fully custom");
        assert_eq!(output3.values.len(), candles.close.len());
    }

    #[test]
    fn test_epma_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = EpmaParams::with_default_params();
        let input = EpmaInput::new(&candles, "close", default_params);
        let result = epma(&input).expect("Failed to calculate EPMA");

        let expected_last_five = [59174.48, 59201.04, 59167.60, 59200.32, 59117.04];

        assert_eq!(
            result.values.len(),
            candles.close.len(),
            "EPMA output length does not match input length!"
        );

        let start_index = result.values.len().saturating_sub(5);
        let result_last_five = &result.values[start_index..];

        for (i, &value) in result_last_five.iter().enumerate() {
            assert!(
                (value - expected_last_five[i]).abs() < 1e-1,
                "EPMA value mismatch at index {}: expected {}, got {}",
                i,
                expected_last_five[i],
                value
            );
        }

        for val in &result.values {
            if !val.is_nan() {
                assert!(
                    val.is_finite(),
                    "EPMA output contains non-finite values (e.g. Inf)."
                );
            }
        }
    }
}
