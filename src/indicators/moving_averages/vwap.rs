use crate::utilities::data_loader::{source_type, Candles};
use chrono::{Datelike, NaiveDateTime, Utc};
use std::error::Error;

#[derive(Debug, Clone)]
pub struct VwapOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct VwapParams {
    pub anchor: Option<String>,
}

impl VwapParams {
    pub fn with_default_params() -> Self {
        VwapParams {
            anchor: Some("1d".to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VwapInput<'a> {
    pub candles: &'a Candles,
    pub source: &'a str,
    pub params: VwapParams,
}

impl<'a> VwapInput<'a> {
    pub fn new(candles: &'a Candles, source: &'a str, params: VwapParams) -> Self {
        VwapInput {
            candles,
            source,
            params,
        }
    }

    pub fn with_default_params(candles: &'a Candles) -> Self {
        VwapInput {
            candles,
            source: "hlc3",
            params: VwapParams::with_default_params(),
        }
    }

    fn get_anchor(&self) -> &str {
        self.params.anchor.as_deref().unwrap_or("1d")
    }
}

#[inline]
pub fn vwap(input: &VwapInput) -> Result<VwapOutput, Box<dyn Error>> {
    let timestamps: &[i64] = input.candles.get_timestamp()?;
    let prices: &[f64] = source_type(input.candles, input.source);
    let volumes: &[f64] = input.candles.select_candle_field("volume")?;

    let n = prices.len();
    if timestamps.len() != n || volumes.len() != n {
        return Err("Mismatch in length of timestamps, prices, or volumes".into());
    }
    if n == 0 {
        return Err("No data for VWAP calculation".into());
    }

    let (count, unit_char) = parse_anchor(input.get_anchor())?;

    let mut vwap_values = vec![f64::NAN; n];
    let mut current_group_id = -1_i64;
    let mut volume_sum = 0.0;
    let mut vol_price_sum = 0.0;

    for i in 0..n {
        let ts_ms = timestamps[i];
        let price = prices[i];
        let volume = volumes[i];

        let group_id = match unit_char {
            'm' => {
                let bucket_ms = (count as i64) * 60_000;
                ts_ms / bucket_ms
            }
            'h' => {
                let bucket_ms = (count as i64) * 3_600_000;
                ts_ms / bucket_ms
            }
            'd' => {
                let bucket_ms = (count as i64) * 86_400_000;
                ts_ms / bucket_ms
            }
            'M' => floor_to_month(ts_ms, count)?,
            _ => return Err(format!("Unsupported anchor unit '{}'", unit_char).into()),
        };

        if group_id != current_group_id {
            current_group_id = group_id;
            volume_sum = 0.0;
            vol_price_sum = 0.0;
        }

        volume_sum += volume;
        vol_price_sum += volume * price;

        vwap_values[i] = if volume_sum > 0.0 {
            vol_price_sum / volume_sum
        } else {
            f64::NAN
        };
    }

    Ok(VwapOutput {
        values: vwap_values,
    })
}

#[inline]
fn parse_anchor(anchor: &str) -> Result<(u32, char), Box<dyn std::error::Error>> {
    let mut idx = 0;
    for (pos, ch) in anchor.char_indices() {
        if !ch.is_ascii_digit() {
            idx = pos;
            break;
        }
    }
    if idx == 0 {
        return Err(format!("No numeric portion found in anchor '{}'", anchor).into());
    }

    let num_part = &anchor[..idx];
    let unit_part = &anchor[idx..];
    let count = num_part
        .parse::<u32>()
        .map_err(|_| format!("Failed parsing numeric portion '{}'", num_part))?;

    if unit_part.len() != 1 {
        return Err(format!("Anchor unit must be 1 char (found '{}')", unit_part).into());
    }

    let mut unit_char = unit_part.chars().next().unwrap();
    unit_char = match unit_char {
        'H' => 'h',
        'D' => 'd',
        c => c,
    };

    match unit_char {
        'm' | 'h' | 'd' | 'M' => Ok((count, unit_char)),
        _ => Err(format!("Unsupported unit '{}'", unit_char).into()),
    }
}

#[inline]
fn floor_to_month(ts_ms: i64, count: u32) -> Result<i64, Box<dyn Error>> {
    let naive = NaiveDateTime::from_timestamp(ts_ms / 1000, ((ts_ms % 1000) * 1_000_000) as u32);
    let dt_utc = chrono::DateTime::<Utc>::from_utc(naive, Utc);

    let year = dt_utc.year();
    let month = dt_utc.month();

    if count == 1 {
        let group_id = (year as i64) * 12 + (month as i64 - 1);
        Ok(group_id)
    } else {
        let quarter_group = (month as i64 - 1) / (count as i64);
        let group_id = (year as i64) * 100 + quarter_group;
        Ok(group_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_vwap_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let params_default = VwapParams { anchor: None };
        let input_default = VwapInput::new(&candles, "hlc3", params_default);
        let output_default = vwap(&input_default).expect("Failed VWAP default anchor");
        assert_eq!(output_default.values.len(), candles.close.len());

        let params_1h = VwapParams {
            anchor: Some("1h".to_string()),
        };
        let input_1h = VwapInput::new(&candles, "close", params_1h);
        let output_1h = vwap(&input_1h).expect("Failed VWAP with anchor=1h");
        assert_eq!(output_1h.values.len(), candles.close.len());

        let params_2M = VwapParams {
            anchor: Some("2M".to_string()),
        };
        let input_2M = VwapInput::new(&candles, "hl2", params_2M);
        let output_2M = vwap(&input_2M).expect("Failed VWAP with anchor=2M");
        assert_eq!(output_2M.values.len(), candles.close.len());
    }

    #[test]
    fn test_vwap_accuracy() {
        let expected_last_five_vwap = [
            59353.05963230107,
            59330.15815713043,
            59289.94649532547,
            59274.6155462414,
            58730.0,
        ];
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let timestamps = candles.get_timestamp().unwrap();
        let volumes = candles.select_candle_field("volume").unwrap();
        let prices = candles.get_calculated_field("hlc3").unwrap();

        let params = VwapParams {
            anchor: Some("1D".to_string()),
        };
        let input = VwapInput {
            candles: &candles,
            source: "hlc3",
            params,
        };
        let result = calculate_vwap(&input).expect("Failed to calculate VWAP");
        assert_eq!(
            result.values.len(),
            prices.len(),
            "Mismatch in output length"
        );
        assert!(result.values.len() >= 5, "Not enough data points for test");

        let start_idx = result.values.len() - 5;
        let actual_last_five = &result.values[start_idx..];

        for (i, &vwap_val) in actual_last_five.iter().enumerate() {
            let exp_val = expected_last_five_vwap[i];
            assert!(
                (vwap_val - exp_val).abs() < 1e-5,
                "VWAP mismatch at index {} => expected {}, got {}",
                i,
                exp_val,
                vwap_val
            );
        }
    }
}
