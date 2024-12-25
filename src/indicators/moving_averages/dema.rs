use std::error::Error;

#[derive(Debug, Clone)]
pub struct DemaParams {
    pub period: Option<usize>,
}

impl Default for DemaParams {
    fn default() -> Self {
        Self { period: Some(30) }
    }
}

#[derive(Debug, Clone)]
pub struct DemaInput<'a> {
    pub data: &'a [f64],
    pub params: DemaParams,
}

impl<'a> DemaInput<'a> {
    pub fn new(data: &'a [f64], params: DemaParams) -> Self {
        DemaInput { data, params }
    }

    pub fn with_default_params(data: &'a [f64]) -> Self {
        DemaInput {
            data,
            params: DemaParams::default(),
        }
    }

    fn get_period(&self) -> usize {
        self.params.period.unwrap_or(30)
    }
}

#[derive(Debug, Clone)]
pub struct DemaOutput {
    pub values: Vec<f64>,
}

#[inline]
pub fn calculate_dema(input: &DemaInput) -> Result<DemaOutput, Box<dyn Error>> {
    let data = input.data;
    let size = data.len();
    let period = input.get_period();

    if period < 1 {
        return Err("Invalid DEMA period (must be >= 1).".into());
    }
    if size < (2 * (period - 1)) {
        return Err("Invalid data length for DEMA calculation.".into());
    }

    let alpha = 2.0 / (period as f64 + 1.0);
    let alpha_1 = 1.0 - alpha;

    let mut output = vec![f64::NAN; size];

    let mut ema = data[0];
    let mut ema2 = ema;

    for i in 0..size {
        ema = ema * alpha_1 + data[i] * alpha;

        if i == (period - 1) {
            ema2 = ema;
        }
        if i >= (period - 1) {
            ema2 = ema2 * alpha_1 + ema * alpha;
        }

        if i >= 2 * (period - 1) {
            output[i] = (2.0 * ema) - ema2;
        }
    }

    Ok(DemaOutput { values: output })
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_dema_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let data = &candles.close;

        let input = DemaInput::with_default_params(data);
        let result = calculate_dema(&input).expect("Failed to calculate DEMA");

        let expected_last_five = [
            59189.73193987478,
            59129.24920772847,
            59058.80282420511,
            59011.5555611042,
            58908.370159946775,
        ];

        assert!(result.values.len() >= 5);
        assert_eq!(
            result.values.len(),
            candles.close.len(),
            "DEMA output length does not match input length"
        );
        let start_index = result.values.len().saturating_sub(5);
        let last_five = &result.values[start_index..];
        for (i, &val) in last_five.iter().enumerate() {
            let exp = expected_last_five[i];
            assert!(
                (val - exp).abs() < 1e-6,
                "DEMA mismatch at {}: expected {}, got {}",
                i,
                exp,
                val
            );
        }
    }
}
