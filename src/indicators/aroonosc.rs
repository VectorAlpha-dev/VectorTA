use crate::utilities::data_loader::Candles;
use std::error::Error;

#[derive(Debug, Clone)]
pub enum AroonOscData<'a> {
    Candles { candles: &'a Candles },
    SlicesHL { high: &'a [f64], low: &'a [f64] },
}

#[derive(Debug, Clone)]
pub struct AroonOscParams {
    pub length: Option<usize>,
}

impl Default for AroonOscParams {
    fn default() -> Self {
        Self { length: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct AroonOscInput<'a> {
    pub data: AroonOscData<'a>,
    pub params: AroonOscParams,
}

impl<'a> AroonOscInput<'a> {
    pub fn from_candles(candles: &'a Candles, params: AroonOscParams) -> Self {
        Self {
            data: AroonOscData::Candles { candles },
            params,
        }
    }

    pub fn from_slices_hl(high: &'a [f64], low: &'a [f64], params: AroonOscParams) -> Self {
        Self {
            data: AroonOscData::SlicesHL { high, low },
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: AroonOscData::Candles { candles },
            params: AroonOscParams::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AroonOscOutput {
    pub values: Vec<f64>,
}

#[inline]
pub fn aroon_osc(input: &AroonOscInput) -> Result<AroonOscOutput, Box<dyn Error>> {
    let length = input.params.length.unwrap_or(14);
    if length == 0 {
        return Err("Invalid length specified for Aroon Osc calculation.".into());
    }

    let (high, low) = match &input.data {
        AroonOscData::Candles { candles } => {
            if candles.close.is_empty() {
                return Err("No candles available.".into());
            }
            (
                candles.select_candle_field("high")?,
                candles.select_candle_field("low")?,
            )
        }
        AroonOscData::SlicesHL { high, low } => {
            if high.is_empty() || low.is_empty() {
                return Err("One or both of the slices for AroonOsc are empty.".into());
            }
            if high.len() != low.len() {
                return Err("Mismatch in high/low slice length.".into());
            }
            (*high, *low)
        }
    };
    let len = low.len();
    if high.len() < length {
        return Err(format!(
            "Not enough data points ({} total) for Aroon Osc length {}.",
            high.len(),
            length
        )
        .into());
    }

    let mut values = vec![f64::NAN; len];
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

        let up = (length as f64 - offset_highest as f64) * inv_length * 100.0;
        let down = (length as f64 - offset_lowest as f64) * inv_length * 100.0;

        values[i] = up - down;
    }

    Ok(AroonOscOutput { values })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_aroon_osc_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = AroonOscInput::with_default_candles(&candles);
        let result = aroon_osc(&input).expect("Failed to calculate Aroon Osc");

        let expected_last_five = [-50.0, -50.0, -50.0, -50.0, -42.8571];

        assert!(result.values.len() >= 5, "Not enough Aroon Osc values");
        assert_eq!(
            result.values.len(),
            candles.close.len(),
            "Aroon Osc output length does not match input length!"
        );
        let start_index = result.values.len().saturating_sub(5);
        let last_five = &result.values[start_index..];

        for (i, &value) in last_five.iter().enumerate() {
            assert!(
                (value - expected_last_five[i]).abs() < 1e-2,
                "Aroon Osc mismatch at index {}: expected {}, got {}",
                i,
                expected_last_five[i],
                value
            );
        }

        let length = 14;
        for val in result.values.iter().skip(length) {
            if !val.is_nan() {
                assert!(
                    val.is_finite(),
                    "Aroon Osc should be finite after enough data"
                );
            }
        }
    }
}
