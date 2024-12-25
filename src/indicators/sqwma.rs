use std::error::Error;

#[derive(Debug, Clone)]
pub struct SqwmaParams {
    pub period: Option<usize>,
}

impl Default for SqwmaParams {
    fn default() -> Self {
        SqwmaParams { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct SqwmaInput<'a> {
    pub data: &'a [f64],
    pub params: SqwmaParams,
}

impl<'a> SqwmaInput<'a> {
    #[inline]
    pub fn new(data: &'a [f64], params: SqwmaParams) -> Self {
        Self { data, params }
    }

    #[inline]
    pub fn with_default_params(data: &'a [f64]) -> Self {
        Self {
            data,
            params: SqwmaParams::default(),
        }
    }

    #[inline]
    fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| SqwmaParams::default().period.unwrap())
    }
}

#[derive(Debug, Clone)]
pub struct SqwmaOutput {
    pub values: Vec<f64>,
}

#[inline]
pub fn calculate_sqwma(input: &SqwmaInput) -> Result<SqwmaOutput, Box<dyn Error>> {
    let data = input.data;
    let n = data.len();
    if n == 0 {
        return Err("Empty data for SQWMA calculation.".into());
    }

    let period = input.get_period();
    if period < 2 {
        return Err("SQWMA period must be >= 2.".into());
    }

    if period + 1 > n {
        return Ok(SqwmaOutput {
            values: data.to_vec(),
        });
    }

    let mut weights = Vec::with_capacity(period - 1);
    for i in 0..(period - 1) {

        let w = (period as f64 - i as f64).powi(2);
        weights.push(w);
    }

    let weight_sum: f64 = weights.iter().sum();

    let mut output = data.to_vec();

    #[inline(always)]
    fn sqwma_sum(
        data: &[f64],
        j: usize,
        weights: &[f64],
    ) -> f64 {
        let mut sum_ = 0.0;
        let p_minus_1 = weights.len();

        let mut i = 0;
        while i < p_minus_1.saturating_sub(3) {
            sum_ += data[j - i]         * weights[i];
            sum_ += data[j - (i + 1)]   * weights[i + 1];
            sum_ += data[j - (i + 2)]   * weights[i + 2];
            sum_ += data[j - (i + 3)]   * weights[i + 3];
            i += 4;
        }
        while i < p_minus_1 {
            sum_ += data[j - i] * weights[i];
            i += 1;
        }
        sum_
    }

    for j in (period + 1)..n {
        let my_sum = sqwma_sum(&data, j, &weights);
        output[j] = my_sum / weight_sum;
    }

    Ok(SqwmaOutput { values: output })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_sqwma_accuracy() {
        let expected_last_five = [
            59229.72287968442,
            59211.30867850099,
            59172.516765286,
            59167.73471400394,
            59067.97928994083,
        ];

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let source = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let params = SqwmaParams { period: Some(14) };
        let input = SqwmaInput::new(source, params);

        let result = calculate_sqwma(&input).expect("Failed to calculate SQWMA");
        assert_eq!(result.values.len(), source.len());

        assert!(
            result.values.len() >= 5,
            "Not enough data for last-5 check"
        );
        let start_idx = result.values.len() - 5;
        let actual_last_five = &result.values[start_idx..];

        for (i, &val) in actual_last_five.iter().enumerate() {
            let exp_val = expected_last_five[i];
            assert!(
                (val - exp_val).abs() < 1e-5,
                "SQWMA mismatch at index {}, expected {}, got {}",
                i,
                exp_val,
                val
            );
        }

        let default_input = SqwmaInput::with_default_params(source);
        let default_result = calculate_sqwma(&default_input).expect("Failed default SQWMA");
        assert!(!default_result.values.is_empty());
    }
}
