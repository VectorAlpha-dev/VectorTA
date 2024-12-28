use crate::utilities::data_loader::Candles;
use std::error::Error;

#[derive(Debug, Clone)]
pub enum AroonData<'a> {
    Candles { candles: &'a Candles },
    SlicesHL { high: &'a [f64], low: &'a [f64] },
}

#[derive(Debug, Clone)]
pub struct AroonParams {
    pub length: Option<usize>,
}

impl Default for AroonParams {
    fn default() -> Self {
        Self { length: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct AroonInput<'a> {
    pub data: AroonData<'a>,
    pub params: AroonParams,
}

impl<'a> AroonInput<'a> {
    pub fn from_candles(candles: &'a Candles, params: AroonParams) -> Self {
        Self {
            data: AroonData::Candles { candles },
            params,
        }
    }

    pub fn from_slices_hl(high: &'a [f64], low: &'a [f64], params: AroonParams) -> Self {
        Self {
            data: AroonData::SlicesHL { high, low },
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: AroonData::Candles { candles },
            params: AroonParams::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AroonOutput {
    pub aroon_up: Vec<f64>,
    pub aroon_down: Vec<f64>,
}

#[inline]
pub fn aroon(input: &AroonInput) -> Result<AroonOutput, Box<dyn Error>> {
    let (high, low) = match &input.data {
        AroonData::Candles { candles } => {
            if candles.close.is_empty() {
                return Err("No candles available.".into());
            }
            (
                candles.select_candle_field("high")?,
                candles.select_candle_field("low")?,
            )
        }
        AroonData::SlicesHL { high, low } => {
            if high.is_empty() || low.is_empty() {
                return Err("One or both of the slices for Aroon are empty.".into());
            }
            if high.len() != low.len() {
                return Err("Mismatch in high/low slice length.".into());
            }
            (*high, *low)
        }
    };
    let length = input.params.length.unwrap_or(14);
    if high.len() < length {
        return Err(format!(
            "Not enough data points ({} total) for Aroon length {}.",
            high.len(),
            length
        )
        .into());
    }
    let len = low.len();
    if length == 0 {
        return Err("Invalid length specified for Aroon calculation.".into());
    }
    let mut aroon_up = vec![f64::NAN; len];
    let mut aroon_down = vec![f64::NAN; len];

    let window = length + 1;
    let inv_length = 1.0 / length as f64;

    for i in (window - 1)..len {
        let start = i + 1 - window;
        let mut highest_val = high[start];
        let mut lowest_val = low[start];
        let mut highest_idx = start;
        let mut lowest_idx = start;

        for j in (start + 1)..=i {
            let h_val = high[j];
            if h_val > highest_val {
                highest_val = h_val;
                highest_idx = j;
            }
            let l_val = low[j];
            if l_val < lowest_val {
                lowest_val = l_val;
                lowest_idx = j;
            }
        }

        let offset_highest = i - highest_idx;
        let offset_lowest = i - lowest_idx;

        aroon_up[i] = (length as f64 - offset_highest as f64) * inv_length * 100.0;
        aroon_down[i] = (length as f64 - offset_lowest as f64) * inv_length * 100.0;
    }

    Ok(AroonOutput {
        aroon_up,
        aroon_down,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_aroon_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = AroonInput::with_default_candles(&candles);
        let result = aroon(&input).expect("Failed to calculate Aroon");

        let expected_up_last_five = [21.43, 14.29, 7.14, 0.0, 0.0];
        let expected_down_last_five = [71.43, 64.29, 57.14, 50.0, 42.86];

        assert!(
            result.aroon_up.len() >= 5 && result.aroon_down.len() >= 5,
            "Not enough Aroon values"
        );

        assert_eq!(
            result.aroon_up.len(),
            result.aroon_down.len(),
            "Aroon Up and Down lengths mismatch"
        );

        assert_eq!(
            result.aroon_up.len(),
            candles.close.len(),
            "Aroon output length does not match input length"
        );

        let start_index = result.aroon_up.len().saturating_sub(5);

        let up_last_five = &result.aroon_up[start_index..];
        let down_last_five = &result.aroon_down[start_index..];

        for (i, &value) in up_last_five.iter().enumerate() {
            assert!(
                (value - expected_up_last_five[i]).abs() < 1e-2,
                "Aroon Up mismatch at index {}: expected {}, got {}",
                i,
                expected_up_last_five[i],
                value
            );
        }

        for (i, &value) in down_last_five.iter().enumerate() {
            assert!(
                (value - expected_down_last_five[i]).abs() < 1e-2,
                "Aroon Down mismatch at index {}: expected {}, got {}",
                i,
                expected_down_last_five[i],
                value
            );
        }

        let length = 14;
        for val in result.aroon_up.iter().skip(length) {
            if !val.is_nan() {
                assert!(
                    val.is_finite(),
                    "Aroon Up should be finite after enough data"
                );
            }
        }
        for val in result.aroon_down.iter().skip(length) {
            if !val.is_nan() {
                assert!(
                    val.is_finite(),
                    "Aroon Down should be finite after enough data"
                );
            }
        }
    }
}
