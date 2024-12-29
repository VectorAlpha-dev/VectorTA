use crate::utilities::data_loader::{source_type, Candles};
use std::error::Error;

#[derive(Debug, Clone)]
pub enum TilsonData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct TilsonOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct TilsonParams {
    pub period: Option<usize>,
    pub volume_factor: Option<f64>,
}

impl TilsonParams {
    pub fn with_default_params() -> Self {
        Self {
            period: None,
            volume_factor: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TilsonInput<'a> {
    pub data: TilsonData<'a>,
    pub params: TilsonParams,
}

impl<'a> TilsonInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: TilsonParams) -> Self {
        Self {
            data: TilsonData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: TilsonParams) -> Self {
        Self {
            data: TilsonData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: TilsonData::Candles {
                candles,
                source: "close",
            },
            params: TilsonParams::with_default_params(),
        }
    }
}

#[inline]
pub fn tilson(input: &TilsonInput) -> Result<TilsonOutput, Box<dyn Error>> {
    let data: &[f64] = match &input.data {
        TilsonData::Candles { candles, source } => source_type(candles, source),
        TilsonData::Slice(slice) => slice,
    };
    let length: usize = data.len();
    let opt_in_time_period = input.params.period.unwrap_or(5);
    let opt_in_v_factor = input.params.volume_factor.unwrap_or(0.0);
    let length = data.len();
    if opt_in_time_period == 0 || opt_in_time_period > length {
        return Err("Invalid period specified.".into());
    }
    let lookback_total = 6 * (opt_in_time_period - 1);
    let mut out_values = vec![std::f64::NAN; length];
    if lookback_total >= length {
        return Ok(TilsonOutput { values: out_values });
    }
    let start_idx = lookback_total;
    let end_idx = length - 1;
    let k = 2.0 / (opt_in_time_period as f64 + 1.0);
    let one_minus_k = 1.0 - k;
    let mut today = 0;

    let mut temp_real;
    let mut e1;
    let mut e2;
    let mut e3;
    let mut e4;
    let mut e5;
    let mut e6;

    temp_real = 0.0;
    for i in 0..opt_in_time_period {
        temp_real += data[today + i];
    }
    e1 = temp_real / opt_in_time_period as f64;
    today += opt_in_time_period;

    temp_real = e1;
    for _ in 1..opt_in_time_period {
        e1 = (k * data[today]) + (one_minus_k * e1);
        temp_real += e1;
        today += 1;
    }
    e2 = temp_real / opt_in_time_period as f64;

    temp_real = e2;
    for _ in 1..opt_in_time_period {
        e1 = (k * data[today]) + (one_minus_k * e1);
        e2 = (k * e1) + (one_minus_k * e2);
        temp_real += e2;
        today += 1;
    }
    e3 = temp_real / opt_in_time_period as f64;

    temp_real = e3;
    for _ in 1..opt_in_time_period {
        e1 = (k * data[today]) + (one_minus_k * e1);
        e2 = (k * e1) + (one_minus_k * e2);
        e3 = (k * e2) + (one_minus_k * e3);
        temp_real += e3;
        today += 1;
    }
    e4 = temp_real / opt_in_time_period as f64;

    temp_real = e4;
    for _ in 1..opt_in_time_period {
        e1 = (k * data[today]) + (one_minus_k * e1);
        e2 = (k * e1) + (one_minus_k * e2);
        e3 = (k * e2) + (one_minus_k * e3);
        e4 = (k * e3) + (one_minus_k * e4);
        temp_real += e4;
        today += 1;
    }
    e5 = temp_real / opt_in_time_period as f64;

    temp_real = e5;
    for _ in 1..opt_in_time_period {
        e1 = (k * data[today]) + (one_minus_k * e1);
        e2 = (k * e1) + (one_minus_k * e2);
        e3 = (k * e2) + (one_minus_k * e3);
        e4 = (k * e3) + (one_minus_k * e4);
        e5 = (k * e4) + (one_minus_k * e5);
        temp_real += e5;
        today += 1;
    }
    e6 = temp_real / opt_in_time_period as f64;

    while today <= start_idx {
        e1 = (k * data[today]) + (one_minus_k * e1);
        e2 = (k * e1) + (one_minus_k * e2);
        e3 = (k * e2) + (one_minus_k * e3);
        e4 = (k * e3) + (one_minus_k * e4);
        e5 = (k * e4) + (one_minus_k * e5);
        e6 = (k * e5) + (one_minus_k * e6);
        today += 1;
    }

    let temp = opt_in_v_factor * opt_in_v_factor;
    let c1 = -(temp * opt_in_v_factor);
    let c2 = 3.0 * (temp - c1);
    let c3 = -6.0 * temp - 3.0 * (opt_in_v_factor - c1);
    let c4 = 1.0 + 3.0 * opt_in_v_factor - c1 + 3.0 * temp;

    if start_idx < length {
        out_values[start_idx] = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3;
    }

    let mut out_idx = start_idx + 1;
    while today <= end_idx {
        e1 = (k * data[today]) + (one_minus_k * e1);
        e2 = (k * e1) + (one_minus_k * e2);
        e3 = (k * e2) + (one_minus_k * e3);
        e4 = (k * e3) + (one_minus_k * e4);
        e5 = (k * e4) + (one_minus_k * e5);
        e6 = (k * e5) + (one_minus_k * e6);
        out_values[out_idx] = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3;
        out_idx += 1;
        today += 1;
    }

    Ok(TilsonOutput { values: out_values })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_tilson_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = TilsonParams {
            period: None,
            volume_factor: None,
        };
        let input_default = TilsonInput::from_candles(&candles, "close", default_params);
        let output_default = tilson(&input_default).expect("Failed T3/Tilson with default params");
        assert_eq!(output_default.values.len(), candles.close.len());

        let params_custom_period = TilsonParams {
            period: Some(10),
            volume_factor: None,
        };
        let input_custom_period = TilsonInput::from_candles(&candles, "hl2", params_custom_period);
        let output_custom_period =
            tilson(&input_custom_period).expect("Failed T3/Tilson with period=10, source=hl2");
        assert_eq!(output_custom_period.values.len(), candles.close.len());

        let params_fully_custom = TilsonParams {
            period: Some(7),
            volume_factor: Some(0.9),
        };
        let input_fully_custom = TilsonInput::from_candles(&candles, "hlc3", params_fully_custom);
        let output_fully_custom =
            tilson(&input_fully_custom).expect("Failed T3/Tilson fully custom");
        assert_eq!(output_fully_custom.values.len(), candles.close.len());
    }

    #[test]
    fn test_tilson_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let params = TilsonParams {
            period: Some(5),
            volume_factor: Some(0.0),
        };
        let input = TilsonInput::from_candles(&candles, "close", params);
        let t3_result = tilson(&input).expect("Failed to calculate T3/Tilson");

        let expected_last_five_t3 = [
            59304.716332473254,
            59283.56868015526,
            59261.16173577631,
            59240.25895948583,
            59203.544843167765,
        ];
        assert!(t3_result.values.len() >= 5);
        assert_eq!(t3_result.values.len(), close_prices.len());

        let start_index = t3_result.values.len() - 5;
        let result_last_five_t3 = &t3_result.values[start_index..];
        for (i, &value) in result_last_five_t3.iter().enumerate() {
            let expected_value = expected_last_five_t3[i];
            assert!(
                (value - expected_value).abs() < 1e-10,
                "T3 mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }

        let default_input = TilsonInput::with_default_candles(&candles);
        let default_t3_result =
            tilson(&default_input).expect("Failed to calculate T3 with defaults");
        assert_eq!(default_t3_result.values.len(), close_prices.len());
    }

    #[test]
    fn test_tilson_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = TilsonInput::with_default_candles(&candles);
        match input.data {
            TilsonData::Candles { source, .. } => {
                assert_eq!(source, "close");
            }
            _ => panic!("Expected TilsonData::Candles"),
        }
        let period = input.params.period.unwrap_or(5);
        let v_factor = input.params.volume_factor.unwrap_or(0.0);
        assert_eq!(period, 5);
        assert!(v_factor.abs() < f64::EPSILON);
    }

    #[test]
    fn test_tilson_with_default_params() {
        let default_params = TilsonParams::with_default_params();
        assert_eq!(default_params.period, None);
        assert_eq!(default_params.volume_factor, None);
    }

    #[test]
    fn test_tilson_with_no_data() {
        let data: [f64; 0] = [];
        let params = TilsonParams {
            period: Some(5),
            volume_factor: Some(0.0),
        };
        let input = TilsonInput::from_slice(&data, params);
        let result = tilson(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Invalid period specified."));
        }
    }

    #[test]
    fn test_tilson_very_small_data_set() {
        let data = [42.0];
        let params = TilsonParams {
            period: Some(5),
            volume_factor: Some(0.0),
        };
        let input = TilsonInput::from_slice(&data, params);
        let result = tilson(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Invalid period specified."));
        }
    }

    #[test]
    fn test_tilson_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let first_input = TilsonInput::from_candles(
            &candles,
            "close",
            TilsonParams {
                period: Some(5),
                volume_factor: Some(0.0),
            },
        );
        let first_result = tilson(&first_input).expect("First T3/Tilson failed");
        let second_input = TilsonInput::from_slice(
            &first_result.values,
            TilsonParams {
                period: Some(3),
                volume_factor: Some(0.7),
            },
        );
        let second_result = tilson(&second_input).expect("Second T3/Tilson failed");
        assert_eq!(second_result.values.len(), first_result.values.len());
    }

    #[test]
    fn test_tilson_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = TilsonInput::from_candles(
            &candles,
            "close",
            TilsonParams {
                period: Some(5),
                volume_factor: Some(0.0),
            },
        );
        let result = tilson(&input).expect("T3/Tilson calculation failed");
        assert_eq!(result.values.len(), candles.close.len());
        for (idx, &val) in result.values.iter().enumerate().skip(50) {
            assert!(val.is_finite(), "NaN found at index {}", idx);
        }
    }
}
