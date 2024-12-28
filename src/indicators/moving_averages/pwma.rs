use crate::utilities::data_loader::{source_type, Candles};
use std::error::Error;

#[derive(Debug, Clone)]
pub enum PwmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct PwmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct PwmaParams {
    pub period: Option<usize>,
}

impl PwmaParams {
    pub fn with_default_params() -> Self {
        Self { period: None }
    }
}

#[derive(Debug, Clone)]
pub struct PwmaInput<'a> {
    pub data: PwmaData<'a>,
    pub params: PwmaParams,
}

impl<'a> PwmaInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: PwmaParams) -> Self {
        Self {
            data: PwmaData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: PwmaParams) -> Self {
        Self {
            data: PwmaData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: PwmaData::Candles {
                candles,
                source: "close",
            },
            params: PwmaParams::with_default_params(),
        }
    }
}

#[inline]
pub fn pwma(input: &PwmaInput) -> Result<PwmaOutput, Box<dyn Error>> {
    let data: &[f64] = match &input.data {
        PwmaData::Candles { candles, source } => source_type(candles, source),
        PwmaData::Slice(slice) => slice,
    };
    let period: usize = input.params.period.unwrap_or(5);
    let len: usize = data.len();
    if period == 0 || period > len {
        return Err("Invalid period specified for PWMA calculation.".into());
    }

    let weights = pascal_weights(period)?;

    let mut output = vec![f64::NAN; len];

    for i in (period - 1)..len {
        let mut weighted_sum = 0.0;
        for k in 0..period {
            let idx = i - k;
            weighted_sum += data[idx] * weights[k];
        }
        output[i] = weighted_sum;
    }

    Ok(PwmaOutput { values: output })
}

#[inline]
fn pascal_weights(period: usize) -> Result<Vec<f64>, Box<dyn Error>> {
    let n = period - 1;
    let mut row = Vec::with_capacity(period);

    for r in 0..=n {
        let c = combination(n, r) as f64;
        row.push(c);
    }

    let sum: f64 = row.iter().sum();
    if sum == 0.0 {
        return Err("Pascal weights sum to zero, invalid period?".into());
    }
    for val in row.iter_mut() {
        *val /= sum;
    }

    Ok(row)
}

#[inline]
fn combination(n: usize, r: usize) -> u64 {
    let r = r.min(n - r);
    if r == 0 {
        return 1;
    }

    let mut numerator: u64 = 1;
    let mut denominator: u64 = 1;

    for i in 0..r {
        numerator *= (n - i) as u64;
        denominator *= (i + 1) as u64;
    }
    numerator / denominator
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_pwma_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = PwmaParams { period: None };
        let input_default = PwmaInput::from_candles(&candles, "close", default_params);
        let output_default = pwma(&input_default).expect("Failed PWMA with default params");
        assert_eq!(output_default.values.len(), candles.close.len());

        let params_custom = PwmaParams { period: Some(8) };
        let input_custom = PwmaInput::from_candles(&candles, "hlc3", params_custom);
        let output_custom = pwma(&input_custom).expect("Failed PWMA with custom params");
        assert_eq!(output_custom.values.len(), candles.close.len());
    }

    #[test]
    fn test_pwma_accuracy() {
        let expected_last_five_pwma = [59313.25, 59309.6875, 59249.3125, 59175.625, 59094.875];
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");
        let params = PwmaParams { period: Some(5) };
        let input = PwmaInput::from_candles(&candles, "close", params);
        let result = pwma(&input).expect("Failed to calculate PWMA");
        assert_eq!(result.values.len(), close_prices.len());
        assert!(result.values.len() >= 5);
        let start_index = result.values.len() - 5;
        let result_last_five = &result.values[start_index..];
        for (i, &val) in result_last_five.iter().enumerate() {
            let expected_val = expected_last_five_pwma[i];
            assert!(
                (val - expected_val).abs() < 1e-3,
                "PWMA mismatch at index {}, expected {}, got {}",
                i,
                expected_val,
                val
            );
        }
    }
}
