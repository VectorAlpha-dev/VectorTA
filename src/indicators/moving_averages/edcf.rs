use crate::utilities::data_loader::{source_type, Candles};
use std::error::Error;

#[derive(Debug, Clone)]
pub struct EdcfOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct EdcfParams {
    pub period: Option<usize>,
}

impl Default for EdcfParams {
    fn default() -> Self {
        EdcfParams { period: Some(15) }
    }
}

#[derive(Debug, Clone)]
pub struct EdcfInput<'a> {
    pub candles: &'a Candles,
    pub source: &'a str,
    pub params: EdcfParams,
}

impl<'a> EdcfInput<'a> {
    pub fn new(candles: &'a Candles, source: &'a str, params: EdcfParams) -> Self {
        EdcfInput {
            candles,
            source,
            params,
        }
    }

    pub fn with_default_params(candles: &'a Candles) -> Self {
        EdcfInput {
            candles,
            source: "close",
            params: EdcfParams::default(),
        }
    }

    #[inline]
    fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| EdcfParams::default().period.unwrap())
    }
}

#[inline]
pub fn edcf(input: &EdcfInput) -> Result<EdcfOutput, Box<dyn Error>> {
    let data: &[f64] = source_type(input.candles, input.source);
    let period: usize = input.get_period();
    let len: usize = data.len();

    if data.is_empty() {
        return Err("No data provided to EDCF filter.".into());
    }
    if period == 0 {
        return Err("EDCF period must be >= 1.".into());
    }

    let mut newseries = vec![f64::NAN; len];

    let mut dist = vec![0.0; len];

    for k in period..len {
        let xk = data[k];
        let mut sum_sq = 0.0;
        for lb in 1..period {
            let diff = xk - data[k - lb];
            sum_sq += diff * diff;
        }
        dist[k] = sum_sq;
    }

    let start_j = 2 * period;
    for j in start_j..len {
        let mut num = 0.0;
        let mut coef_sum = 0.0;

        for i in 0..period {
            let k = j - i;
            let distance = dist[k];
            let base_val = data[k];
            num += distance * base_val;
            coef_sum += distance;
        }

        if coef_sum != 0.0 {
            newseries[j] = num / coef_sum;
        } else {
            newseries[j] = 0.0;
        }
    }

    Ok(EdcfOutput { values: newseries })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_edcf_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = EdcfInput::new(&candles, "hl2", EdcfParams { period: Some(15) });

        let edcf_result = edcf(&input).expect("EDCF calculation failed");
        let edcf_values = &edcf_result.values;

        assert_eq!(
            edcf_values.len(),
            candles.close.len(),
            "EDCF output length does not match input length!"
        );

        let expected_last_five = [
            59593.332275678375,
            59731.70263288801,
            59766.41512339413,
            59655.66162110993,
            59332.492883847,
        ];

        assert!(
            edcf_values.len() >= expected_last_five.len(),
            "Not enough EDCF values for the test"
        );

        let start_index = edcf_values.len() - expected_last_five.len();
        let actual_last_five = &edcf_values[start_index..];

        for (i, (&actual, &expected)) in actual_last_five
            .iter()
            .zip(expected_last_five.iter())
            .enumerate()
        {
            let diff = (actual - expected).abs();
            assert!(
                diff < 1e-8,
                "EDCF mismatch at index {}: expected {:.14}, got {:.14}",
                start_index + i,
                expected,
                actual
            );
        }
    }
}
