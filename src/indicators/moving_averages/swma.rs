use crate::utilities::data_loader::{source_type, Candles};
use std::error::Error;

#[derive(Debug, Clone)]
pub struct SwmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct SwmaParams {
    pub period: Option<usize>,
}

impl SwmaParams {
    pub fn with_default_params() -> Self {
        SwmaParams { period: None }
    }
}

#[derive(Debug, Clone)]
pub struct SwmaInput<'a> {
    pub candles: &'a Candles,
    pub source: &'a str,
    pub params: SwmaParams,
}

impl<'a> SwmaInput<'a> {
    pub fn new(candles: &'a Candles, source: &'a str, params: SwmaParams) -> Self {
        SwmaInput {
            candles,
            source,
            params,
        }
    }

    pub fn with_default_params(candles: &'a Candles) -> Self {
        SwmaInput {
            candles,
            source: "close",
            params: SwmaParams::with_default_params(),
        }
    }
}

#[inline]
pub fn swma(input: &SwmaInput) -> Result<SwmaOutput, Box<dyn Error>> {
    let data: &[f64] = source_type(input.candles, input.source);
    let len: usize = data.len();

    if data.is_empty() {
        return Ok(SwmaOutput { values: vec![] });
    }
    if period == 0 {
        return Err("SWMA period must be >= 1.".into());
    }
    if period > data.len() {
        return Err("SWMA period cannot exceed data length.".into());
    }

    let len: usize = data.len();

    let weights = build_symmetric_triangle(period);

    let mut swma_values = vec![f64::NAN; len];

    for i in (period - 1)..len {
        let window_start = i + 1 - period;
        let window = &data[window_start..=i];
        let mut sum = 0.0;
        for (w_idx, &val) in window.iter().enumerate() {
            sum += val * weights[w_idx];
        }
        swma_values[i] = sum;
    }

    Ok(SwmaOutput {
        values: swma_values,
    })
}

#[inline]
fn build_symmetric_triangle(n: usize) -> Vec<f64> {
    let n = n.max(2);

    let triangle: Vec<f64> = if n == 2 {
        vec![1.0, 1.0]
    } else if n % 2 == 0 {
        let half = n / 2;
        let mut front: Vec<f64> = (1..=half).map(|x| x as f64).collect();
        let mut back = front.clone();
        back.reverse();
        front.extend(back);
        front
    } else {
        let half_plus = ((n + 1) as f64 / 2.0).floor() as usize;
        let mut front: Vec<f64> = (1..=half_plus).map(|x| x as f64).collect();
        let mut tri = front.clone();
        front.pop();
        front.reverse();
        tri.extend(front);
        tri
    };

    let sum: f64 = triangle.iter().sum();
    triangle.into_iter().map(|x| x / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_swma_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = SwmaParams { period: None };
        let input = SwmaInput::new(&candles, "close", default_params);
        let output = swma(&input).expect("Failed SWMA with default params");
        assert_eq!(output.values.len(), candles.close.len());

        let params_period_10 = SwmaParams { period: Some(10) };
        let input2 = SwmaInput::new(&candles, "hl2", params_period_10);
        let output2 = swma(&input2).expect("Failed SWMA with period=10, source=hl2");
        assert_eq!(output2.values.len(), candles.close.len());

        let params_custom = SwmaParams { period: Some(20) };
        let input3 = SwmaInput::new(&candles, "hlc3", params_custom);
        let output3 = swma(&input3).expect("Failed SWMA fully custom");
        assert_eq!(output3.values.len(), candles.close.len());
    }

    #[test]
    fn test_swma_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let source = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let default_params = SwmaParams::with_default_params();
        let input = SwmaInput::new(&candles, "close", default_params);
        let result = swma(&input).expect("SWMA calculation failed");
        let len = result.values.len();
        assert_eq!(len, candles.close.len(), "Length mismatch");

        let expected_last_five = [
            59288.22222222222,
            59301.99999999999,
            59247.33333333333,
            59179.88888888889,
            59080.99999999999,
        ];
        assert!(
            len >= expected_last_five.len(),
            "Not enough SWMA values for the test"
        );
        let start_index = len - expected_last_five.len();
        let actual_last_five = &result.values[start_index..];
        for (i, (&actual, &expected)) in actual_last_five.iter().zip(expected_last_five.iter()).enumerate() {
            let diff = (actual - expected).abs();
            assert!(
                diff < 1e-8,
                "SWMA mismatch at index {}: expected {:.14}, got {:.14}",
                i,
                expected,
                actual
            );
        }
    }
}