use crate::utilities::data_loader::{source_type, Candles};
use std::error::Error;

#[derive(Debug, Clone)]
pub enum FwmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct FwmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct FwmaParams {
    pub period: Option<usize>,
}

impl FwmaParams {
    pub fn with_default_params() -> Self {
        FwmaParams { period: None }
    }
}

#[derive(Debug, Clone)]
pub struct FwmaInput<'a> {
    pub data: FwmaData<'a>,
    pub params: FwmaParams,
}

impl<'a> FwmaInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: FwmaParams) -> Self {
        Self {
            data: FwmaData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: FwmaParams) -> Self {
        Self {
            data: FwmaData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: FwmaData::Candles {
                candles,
                source: "close",
            },
            params: FwmaParams::with_default_params(),
        }
    }
}

#[inline]
pub fn fwma(input: &FwmaInput) -> Result<FwmaOutput, Box<dyn Error>> {
    let data: &[f64] = match &input.data {
        FwmaData::Candles { candles, source } => source_type(candles, source),
        FwmaData::Slice(slice) => slice,
    };
    let len: usize = data.len();
    let period: usize = input.params.period.unwrap_or(5);
    let mut values = vec![f64::NAN; len];
    if period == 0 || period > len {
        return Err("Invalid period specified for FWMA calculation.".into());
    }
    let mut fib = Vec::with_capacity(period);
    {
        let mut a = 1;
        let mut b = 1;
        fib.push(a as f64);
        for _ in 1..period {
            let c = a + b;
            a = b;
            b = c;
            fib.push(a as f64);
        }
    }
    let fib_sum: f64 = fib.iter().sum();
    for w in fib.iter_mut() {
        *w /= fib_sum;
    }
    let end_offset = period - 1;
    for i in end_offset..len {
        let start = i + 1 - period;
        let mut sum = 0.0;
        let fib_slice = &fib[..];
        let data_slice = &data[start..start + period];
        for j in 0..period {
            sum += data_slice[j] * fib_slice[j];
        }
        values[i] = sum;
    }
    Ok(FwmaOutput { values })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_fwma_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input_default = FwmaInput::with_default_candles(&candles);
        let output_default = fwma(&input_default).expect("Failed FWMA with default params");
        assert_eq!(output_default.values.len(), candles.close.len());

        let params_period_only = FwmaParams { period: Some(10) };
        let input_period_only = FwmaInput::from_candles(&candles, "hl2", params_period_only);
        let output_period_only =
            fwma(&input_period_only).expect("Failed FWMA with period=10, source=hl2");
        assert_eq!(output_period_only.values.len(), candles.close.len());
    }

    #[test]
    fn test_fwma_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = FwmaInput::with_default_candles(&candles);
        let result = fwma(&input).expect("Failed to calculate FWMA");

        let expected_last_five = [
            59273.583333333336,
            59252.5,
            59167.083333333336,
            59151.0,
            58940.333333333336,
        ];

        assert!(result.values.len() >= 5);
        assert_eq!(
            result.values.len(),
            candles.close.len(),
            "FWMA values count should match input data count"
        );

        let start_index = result.values.len() - 5;
        let last_five = &result.values[start_index..];
        for (i, &val) in last_five.iter().enumerate() {
            let exp = expected_last_five[i];
            assert!(
                (val - exp).abs() < 1e-8,
                "FWMA mismatch at {}: expected {}, got {}",
                i,
                exp,
                val
            );
        }
    }
}
