use std::error::Error;

#[derive(Debug, Clone)]
pub struct FwmaParams {
    pub period: Option<usize>,
}

impl Default for FwmaParams {
    fn default() -> Self {
        FwmaParams { period: Some(5) }
    }
}

#[derive(Debug, Clone)]
pub struct FwmaInput<'a> {
    pub data: &'a [f64],
    pub params: FwmaParams,
}

impl<'a> FwmaInput<'a> {
    pub fn new(data: &'a [f64], params: FwmaParams) -> Self {
        FwmaInput { data, params }
    }

    pub fn with_default_params(data: &'a [f64]) -> Self {
        FwmaInput {
            data,
            params: FwmaParams::default(),
        }
    }

    fn get_period(&self) -> usize {
        self.params.period.unwrap_or(5)
    }
}

#[derive(Debug, Clone)]
pub struct FwmaOutput {
    pub values: Vec<f64>,
}

#[inline]
pub fn calculate_fwma(input: &FwmaInput) -> Result<FwmaOutput, Box<dyn Error>> {
    let data = input.data;
    let len = data.len();
    let period = input.get_period();
    let mut values = vec![f64::NAN; len];
    if period > len {
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
    fn test_fwma_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let data = candles
            .select_candle_field("close")
            .expect("Failed to get close");
        let input = FwmaInput::with_default_params(data);
        let result = calculate_fwma(&input).expect("Failed to calculate FWMA");
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
            data.len(),
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
