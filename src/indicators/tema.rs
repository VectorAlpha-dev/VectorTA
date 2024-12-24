use std::error::Error;

#[derive(Debug, Clone)]
pub struct TemaParams {
    pub period: Option<usize>,
}

impl Default for TemaParams {
    fn default() -> Self {
        Self { period: Some(9) }
    }
}

#[derive(Debug, Clone)]
pub struct TemaInput<'a> {
    pub data: &'a [f64],
    pub params: TemaParams,
}

impl<'a> TemaInput<'a> {
    pub fn new(data: &'a [f64], params: TemaParams) -> Self {
        TemaInput { data, params }
    }

    pub fn with_default_params(data: &'a [f64]) -> Self {
        TemaInput {
            data,
            params: TemaParams::default(),
        }
    }

    fn get_period(&self) -> usize {
        self.params.period.unwrap_or(9)
    }
}

#[derive(Debug, Clone)]
pub struct TemaOutput {
    pub values: Vec<f64>,
}

#[inline]
pub fn calculate_tema(input: &TemaInput) -> Result<TemaOutput, Box<dyn Error>> {
    let data = input.data;
    let n = data.len();
    let period = input.get_period();

    if period < 1 {
        return Err("Period cannot be zero or negative for TEMA.".into());
    }

    if period > n {
        return Err("Not enough data points to calculate TEMA.".into());
    }

    let lookback = (period - 1) * 3;
    if n == 0 || n <= lookback {
        return Ok(TemaOutput {
            values: vec![f64::NAN; n],
        });
    }

    let per = 2.0 / (period as f64 + 1.0);
    let per1 = 1.0 - per;

    let mut ema1 = data[0];
    let mut ema2 = 0.0;
    let mut ema3 = 0.0;

    let mut tema_values = vec![f64::NAN; n];

    for i in 0..n {
        let price = data[i];

        ema1 = ema1 * per1 + price * per;

        if i == (period - 1) {
            ema2 = ema1;
        }
        if i >= (period - 1) {
            ema2 = ema2 * per1 + ema1 * per;
        }

        if i == 2 * (period - 1) {
            ema3 = ema2;
        }
        if i >= 2 * (period - 1) {
            ema3 = ema3 * per1 + ema2 * per;
        }

        if i >= lookback {
            tema_values[i] = 3.0 * ema1 - 3.0 * ema2 + ema3;
        }
    }

    Ok(TemaOutput {
        values: tema_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::data_loader::read_candles_from_csv;

    #[test]
    fn test_tema_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let params = TemaParams { period: Some(9) };
        let input = TemaInput::new(close_prices, params);
        let tema_result = calculate_tema(&input).expect("Failed to calculate TEMA");

        let expected_last_five_tema = [
            59281.895570662884,
            59257.25021607971,
            59172.23342859784,
            59175.218345941066,
            58934.24395798363,
        ];

        assert!(tema_result.values.len() >= 5);
        assert_eq!(
            tema_result.values.len(),
            close_prices.len(),
            "Output count should match input data count"
        );
        let start_index = tema_result.values.len() - 5;
        let result_last_five_tema = &tema_result.values[start_index..];

        for (i, &value) in result_last_five_tema.iter().enumerate() {
            let expected_value = expected_last_five_tema[i];
            assert!(
                (value - expected_value).abs() < 1e-8,
                "TEMA value mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }
    }
}
