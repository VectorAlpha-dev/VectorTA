use chrono::{NaiveDateTime, Datelike, Utc};
use std::error::Error;

#[derive(Debug, Clone)]
pub struct VwapParams {
    pub anchor: Option<String>,
}

impl Default for VwapParams {
    fn default() -> Self {
        VwapParams {
            anchor: Some("1d".to_string()),
        }
    }
}

impl VwapParams {
    pub fn new(anchor: Option<String>) -> Self {
        VwapParams {
            anchor: anchor.map(|a| a.to_lowercase()),
        }
    }

    pub fn set_anchor(&mut self, anchor: Option<String>) {
        self.anchor = anchor.map(|a| a.to_lowercase());
    }
}

#[derive(Debug, Clone)]
pub struct VwapInput<'a> {
    pub timestamps: &'a [i64],
    pub prices: &'a [f64],
    pub volumes: &'a [f64],
    pub params: VwapParams,
}

impl<'a> VwapInput<'a> {
    #[inline]
    pub fn new(
        timestamps: &'a [i64],
        prices: &'a [f64],
        volumes: &'a [f64],
        params: VwapParams,
    ) -> Self {
        Self {
            timestamps,
            prices,
            volumes,
            params,
        }
    }

    #[inline]
    pub fn with_default_params(
        timestamps: &'a [i64],
        prices: &'a [f64],
        volumes: &'a [f64],
    ) -> Self {
        Self {
            timestamps,
            prices,
            volumes,
            params: VwapParams::default(),
        }
    }

    #[inline]
    fn get_anchor(&self) -> &str {
        self.params.anchor.as_deref().unwrap_or("1d")
    }
}

#[derive(Debug, Clone)]
pub struct VwapOutput {
    pub values: Vec<f64>,
}

#[inline]
pub fn calculate_vwap(input: &VwapInput) -> Result<VwapOutput, Box<dyn Error>> {
    let n = input.prices.len();
    if input.timestamps.len() != n || input.volumes.len() != n {
        return Err("Mismatch in length of timestamps / prices / volumes".into());
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
        let ts_ms = input.timestamps[i];
        let price = input.prices[i];
        let volume = input.volumes[i];

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
            'M' => {
                floor_to_month(ts_ms, count)?
            }
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

    Ok(VwapOutput { values: vwap_values })
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

        let timestamps: &[i64] = candles.get_timestamp().unwrap();
        let volumes = candles.select_candle_field("volume").unwrap();
        let prices = candles.get_calculated_field("hlc3").unwrap();

        let params = VwapParams {
            anchor: Some("1D".to_string()),
        };
        let input = VwapInput::new(timestamps, prices, volumes, params);

        let result = calculate_vwap(&input).expect("Failed to calculate VWAP");
        assert_eq!(
            result.values.len(),
            prices.len(),
            "VWAP result length mismatch"
        );

        assert!(
            result.values.len() >= 5,
            "Not enough data to compare last 5 VWAP values"
        );
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
