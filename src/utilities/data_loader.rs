extern crate csv;
extern crate lazy_static;
extern crate serde;

use csv::ReaderBuilder;
use std::error::Error;
use std::fs::File;

#[derive(Debug, Clone)]
pub struct Candles {
    pub timestamp: Vec<i64>,
    pub open: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub close: Vec<f64>,
    pub volume: Vec<f64>,
}

impl Candles {
    pub fn new(
        timestamp: Vec<i64>,
        open: Vec<f64>,
        high: Vec<f64>,
        low: Vec<f64>,
        close: Vec<f64>,
        volume: Vec<f64>,
    ) -> Self {
        Candles {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
        }
    }

    pub fn get_timestamp(&self) -> Result<&[i64], Box<dyn Error>> {
        Ok(&self.timestamp)
    }

    pub fn select_candle_field(&self, field: &str) -> Result<&[f64], Box<dyn Error>> {
        match field.to_lowercase().as_str() {
            "open" => Ok(&self.open),
            "high" => Ok(&self.high),
            "low" => Ok(&self.low),
            "close" => Ok(&self.close),
            "volume" => Ok(&self.volume),
            _ => Err(format!("Invalid field: {}", field).into()),
        }
    }

    pub fn get_calculated_field(&self, field: &str) -> Result<Vec<f64>, Box<dyn Error>> {
        match field.to_lowercase().as_str() {
            "hl2" => Ok(self.hl2()),
            "hlc3" => Ok(self.hlc3()),
            "ohlc4" => Ok(self.ohlc4()),
            "hlcc4" => Ok(self.hlcc4()),
            _ => Err(format!("Invalid calculated field: {}", field).into()),
        }
    }

    pub fn hl2(&self) -> Vec<f64> {
        self.high
            .iter()
            .zip(self.low.iter())
            .map(|(&high, &low)| (high + low) / 2.0)
            .collect()
    }

    pub fn hlc3(&self) -> Vec<f64> {
        self.high
            .iter()
            .zip(self.low.iter())
            .zip(self.close.iter())
            .map(|((&high, &low), &close)| (high + low + close) / 3.0)
            .collect()
    }

    pub fn ohlc4(&self) -> Vec<f64> {
        self.open
            .iter()
            .zip(self.high.iter())
            .zip(self.low.iter())
            .zip(self.close.iter())
            .map(|(((&open, &high), &low), &close)| (open + high + low + close) / 4.0)
            .collect()
    }

    pub fn hlcc4(&self) -> Vec<f64> {
        self.high
            .iter()
            .zip(self.low.iter())
            .zip(self.close.iter())
            .map(|((&high, &low), &close)| (high + low + 2.0 * close) / 4.0)
            .collect()
    }
}

pub fn read_candles_from_csv(file_path: &str) -> Result<Candles, Box<dyn Error>> {
    let file = File::open(file_path)?;
    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);

    let mut timestamp = Vec::new();
    let mut open = Vec::new();
    let mut high = Vec::new();
    let mut low = Vec::new();
    let mut close = Vec::new();
    let mut volume = Vec::new();

    for result in rdr.records() {
        let record = result?;
        timestamp.push(record[0].parse::<i64>()?);
        open.push(record[1].parse::<f64>()?);
        high.push(record[3].parse::<f64>()?);
        low.push(record[4].parse::<f64>()?);
        close.push(record[2].parse::<f64>()?);
        volume.push(record[5].parse::<f64>()?);
    }

    Ok(Candles::new(
        timestamp,
        open,
        high,
        low,
        close,
        volume,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_congruency() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load CSV for testing");

        let len = candles.timestamp.len();
        assert_eq!(candles.open.len(), len, "Open length mismatch");
        assert_eq!(candles.high.len(), len, "High length mismatch");
        assert_eq!(candles.low.len(), len, "Low length mismatch");
        assert_eq!(candles.close.len(), len, "Close length mismatch");
        assert_eq!(candles.volume.len(), len, "Volume length mismatch");
    }

    #[test]
    fn test_calculated_fields_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load CSV for testing");

        let hl2 = candles.get_calculated_field("hl2").expect("Failed to get HL2");
        let hlc3 = candles.get_calculated_field("hlc3").expect("Failed to get HLC3");
        let ohlc4 = candles.get_calculated_field("ohlc4").expect("Failed to get OHLC4");
        let hlcc4 = candles.get_calculated_field("hlcc4").expect("Failed to get HLCC4");

        let len = candles.timestamp.len();
        assert_eq!(hl2.len(), len, "HL2 length mismatch");
        assert_eq!(hlc3.len(), len, "HLC3 length mismatch");
        assert_eq!(ohlc4.len(), len, "OHLC4 length mismatch");
        assert_eq!(hlcc4.len(), len, "HLCC4 length mismatch");

        let expected_last_5_hl2 = [59166.0, 59244.5, 59118.0, 59146.5, 58767.5];
        let expected_last_5_hlc3 = [59205.7, 59223.3, 59091.7, 59149.3, 58730.0];
        let expected_last_5_ohlc4 = [59221.8, 59238.8, 59114.3, 59121.8, 58836.3];
        let expected_last_5_hlcc4 = [59225.5, 59212.8, 59078.5, 59150.8, 58711.3];

        fn compare_last_five(actual: &[f64], expected: &[f64], field_name: &str) {
            let start = actual.len().saturating_sub(5);
            let actual_slice = &actual[start..];
            for (i, (&a, &e)) in actual_slice.iter().zip(expected.iter()).enumerate() {
                let diff = (a - e).abs();
                assert!(
                    diff < 1e-1,
                    "Mismatch in {} at last-5 index {}: expected {}, got {}",
                    field_name,
                    i,
                    e,
                    a
                );
            }
        }
        compare_last_five(&hl2, &expected_last_5_hl2, "HL2");
        compare_last_five(&hlc3, &expected_last_5_hlc3, "HLC3");
        compare_last_five(&ohlc4, &expected_last_5_ohlc4, "OHLC4");
        compare_last_five(&hlcc4, &expected_last_5_hlcc4, "HLCC4");
    }
}