use std::error::Error;

#[derive(Debug, Clone)]
pub struct HmaParams {
    pub period: Option<usize>,
}

impl Default for HmaParams {
    fn default() -> Self {
        HmaParams { period: Some(5) }
    }
}

#[derive(Debug, Clone)]
pub struct HmaInput<'a> {
    pub data: &'a [f64],
    pub params: HmaParams,
}

impl<'a> HmaInput<'a> {
    pub fn new(data: &'a [f64], params: HmaParams) -> Self {
        HmaInput { data, params }
    }

    pub fn with_default_params(data: &'a [f64]) -> Self {
        HmaInput {
            data,
            params: HmaParams::default(),
        }
    }

    fn get_period(&self) -> usize {
        self.params.period.unwrap_or(5)
    }
}

#[derive(Debug, Clone)]
pub struct HmaOutput {
    pub values: Vec<f64>,
}

#[inline]
pub fn calculate_hma(input: &HmaInput) -> Result<HmaOutput, Box<dyn Error>> {
    let data = input.data;
    let len = data.len();
    let period = input.get_period();
    let mut values = vec![f64::NAN; len];

    if period == 0 || period > len {
        return Ok(HmaOutput { values });
    }

    let half = period / 2;
    if half == 0 {
        return Ok(HmaOutput { values });
    }

    let sqrtp = (period as f64).sqrt().floor() as usize;
    if sqrtp == 0 {
        return Ok(HmaOutput { values });
    }

    let sum_w_half = (half * (half + 1)) >> 1;
    let denom_half = sum_w_half as f64;

    let sum_w_full = (period * (period + 1)) >> 1;
    let denom_full = sum_w_full as f64;

    let sum_w_sqrt = (sqrtp * (sqrtp + 1)) >> 1;
    let denom_sqrt = sum_w_sqrt as f64;

    let lookback_half = half - 1;
    let lookback_full = period - 1;

    let half_f = half as f64;
    let period_f = period as f64;
    let sqrtp_f = sqrtp as f64;

    let mut wma_half = vec![f64::NAN; len];
    let mut wma_full = vec![f64::NAN; len];

    let mut period_sub_half = 0.0;
    let mut period_sum_half = 0.0;
    let mut in_idx = 0;
    let mut i_half = 1;
    while in_idx < lookback_half {
        let val = data[in_idx];
        period_sub_half += val;
        period_sum_half += val * (i_half as f64);
        in_idx += 1;
        i_half += 1;
    }

    let mut period_sub_full = 0.0;
    let mut period_sum_full = 0.0;
    let mut in_idx_full = 0;
    let mut i_full = 1;
    while in_idx_full < lookback_full {
        let val = data[in_idx_full];
        period_sub_full += val;
        period_sum_full += val * (i_full as f64);
        in_idx_full += 1;
        i_full += 1;
    }

    if in_idx < len {
        let val = data[in_idx];
        in_idx += 1;
        period_sub_half += val;
        period_sum_half += val * half_f;

        wma_half[lookback_half] = period_sum_half / denom_half;
        period_sum_half -= period_sub_half;

        let mut trailing_idx_half = 1;
        let mut trailing_value_half = data[0];

        if in_idx_full < len {
            let valf = data[in_idx_full];
            in_idx_full += 1;
            period_sub_full += valf;
            period_sum_full += valf * period_f;

            wma_full[lookback_full] = period_sum_full / denom_full;
            period_sum_full -= period_sub_full;

            let mut trailing_idx_full = 1;
            let mut trailing_value_full = data[0];

            while in_idx < len || in_idx_full < len {
                if in_idx < len {
                    let new_val = data[in_idx];
                    in_idx += 1;

                    period_sub_half += new_val;
                    period_sub_half -= trailing_value_half;
                    period_sum_half += new_val * half_f;

                    trailing_value_half = data[trailing_idx_half];
                    trailing_idx_half += 1;

                    wma_half[in_idx - 1] = period_sum_half / denom_half;
                    period_sum_half -= period_sub_half;
                }

                if in_idx_full < len {
                    let new_valf = data[in_idx_full];
                    in_idx_full += 1;

                    period_sub_full += new_valf;
                    period_sub_full -= trailing_value_full;
                    period_sum_full += new_valf * period_f;

                    trailing_value_full = data[trailing_idx_full];
                    trailing_idx_full += 1;

                    wma_full[in_idx_full - 1] = period_sum_full / denom_full;
                    period_sum_full -= period_sub_full;
                }
            }
        }
    }

    let mut diff = vec![f64::NAN; len];
    for i in 0..len {
        let a = wma_half[i];
        let b = wma_full[i];
        if a.is_finite() && b.is_finite() {
            diff[i] = 2.0 * a - b;
        }
    }

    let mut wma_sqrt = vec![f64::NAN; len];
    {
        let lookback_sqrt = sqrtp - 1;
        let mut period_sub_sqrt = 0.0;
        let mut period_sum_sqrt = 0.0;
        let mut in_idx_sqrt = 0;
        let mut i_s = 1;

        while in_idx_sqrt < lookback_sqrt {
            let val = diff[in_idx_sqrt];
            if val.is_finite() {
                period_sub_sqrt += val;
                period_sum_sqrt += val * (i_s as f64);
            }
            in_idx_sqrt += 1;
            i_s += 1;
        }

        if in_idx_sqrt < len {
            let val = diff[in_idx_sqrt];
            in_idx_sqrt += 1;
            if val.is_finite() {
                period_sub_sqrt += val;
                period_sum_sqrt += val * sqrtp_f;
            }
            let mut trailing_idx_sqrt = 1;
            let mut trailing_value_sqrt = diff[0];

            wma_sqrt[lookback_sqrt] = if trailing_value_sqrt.is_finite() {
                period_sum_sqrt / denom_sqrt
            } else {
                f64::NAN
            };
            period_sum_sqrt -= period_sub_sqrt;

            while in_idx_sqrt < len {
                let new_val = diff[in_idx_sqrt];
                in_idx_sqrt += 1;

                if new_val.is_finite() {
                    period_sub_sqrt += new_val;
                }
                if trailing_value_sqrt.is_finite() {
                    period_sub_sqrt -= trailing_value_sqrt;
                }
                if new_val.is_finite() {
                    period_sum_sqrt += new_val * sqrtp_f;
                }

                trailing_value_sqrt = diff[trailing_idx_sqrt];
                trailing_idx_sqrt += 1;

                wma_sqrt[in_idx_sqrt - 1] = if period_sub_sqrt != 0.0 {
                    period_sum_sqrt / denom_sqrt
                } else {
                    f64::NAN
                };

                period_sum_sqrt -= period_sub_sqrt;
            }
        }
    }

    for i in 0..len {
        values[i] = wma_sqrt[i];
    }

    Ok(HmaOutput { values })
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_hma_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let data = candles.select_candle_field("close").expect("Failed");
        let input = HmaInput::with_default_params(data);
        let result = calculate_hma(&input).expect("Failed hma");
        let expected_last_five = [
            59334.13333336847,
            59201.4666667018,
            59047.77777781293,
            59048.71111114628,
            58803.44444447962,
        ];
        assert!(result.values.len() >= 5);
        assert_eq!(
            result.values.len(),
            data.len(),
            "HMA values count should match input data count"
        );
        let start = result.values.len() - 5;
        let last_five = &result.values[start..];
        for (i, &val) in last_five.iter().enumerate() {
            let exp = expected_last_five[i];
            assert!(
                (val - exp).abs() < 1e-3,
                "HMA mismatch at {}: expected {}, got {}",
                i,
                exp,
                val
            );
        }
    }
}
