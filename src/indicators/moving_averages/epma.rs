use std::error::Error;

#[derive(Debug, Clone)]
pub struct EpmaParams {
    pub period: Option<usize>,
    pub offset: Option<usize>,
}

impl Default for EpmaParams {
    fn default() -> Self {
        EpmaParams {
            period: Some(11),
            offset: Some(4),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EpmaInput<'a> {
    pub data: &'a [f64],
    pub params: EpmaParams,
}

impl<'a> EpmaInput<'a> {
    #[inline]
    pub fn new(data: &'a [f64], params: EpmaParams) -> Self {
        Self { data, params }
    }

    #[inline]
    pub fn with_default_params(data: &'a [f64]) -> Self {
        Self {
            data,
            params: EpmaParams::default(),
        }
    }

    #[inline]
    fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| EpmaParams::default().period.unwrap())
    }

    #[inline]
    fn get_offset(&self) -> usize {
        self.params
            .offset
            .unwrap_or_else(|| EpmaParams::default().offset.unwrap())
    }
}

#[derive(Debug, Clone)]
pub struct EpmaOutput {
    pub values: Vec<f64>,
}

#[inline]
pub fn calculate_epma(input: &EpmaInput) -> Result<EpmaOutput, Box<dyn Error>> {
    let data = input.data;
    let n = data.len();
    if n == 0 {
        return Err("Empty data slice for EPMA calculation.".into());
    }

    let period = input.get_period();
    let offset = input.get_offset();
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
    fn test_epma_accuracy() {
        let expected_last_five = [59174.48, 59201.04, 59167.6, 59200.32, 59117.04];

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let source = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let params = EpmaParams {
            period: Some(11),
            offset: Some(4),
        };
        let input = EpmaInput::new(source, params);

        let result = calculate_epma(&input).expect("EPMA failed");
        assert_eq!(result.values.len(), source.len());

        assert!(result.values.len() >= 5, "Not enough data for last-5 check");
        let start_idx = result.values.len() - 5;
        let actual_last_five = &result.values[start_idx..];

        for (i, &val) in actual_last_five.iter().enumerate() {
            let expected_val = expected_last_five[i];
            let diff = (val - expected_val).abs();
            assert!(
                diff < 1e-2,
                "EPMA mismatch at index {}, expected {}, got {}, diff={}",
                i,
                expected_val,
                val,
                diff
            );
        }

        let default_input = EpmaInput::with_default_params(source);
        let default_result = calculate_epma(&default_input).expect("default EPMA failed");
        assert_eq!(default_result.values.len(), source.len());
    }
}
