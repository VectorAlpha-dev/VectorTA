extern crate csv;
extern crate serde;

use csv::ReaderBuilder;
use std::error::Error;
use std::fs::File;

#[derive(Debug, Clone, Copy)]
pub struct CandleFieldFlags {
    pub open: bool,
    pub high: bool,
    pub low: bool,
    pub close: bool,
    pub volume: bool,
}

#[derive(Debug, Clone)]
pub struct Candles {
    pub timestamp: Vec<i64>,
    pub open: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub close: Vec<f64>,
    pub volume: Vec<f64>,
    pub fields: CandleFieldFlags,
    pub hl2: Vec<f64>,
    pub hlc3: Vec<f64>,
    pub ohlc4: Vec<f64>,
    pub hlcc4: Vec<f64>,
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
        let mut candles = Candles {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
            fields: CandleFieldFlags {
                open: true,
                high: true,
                low: true,
                close: true,
                volume: true,
            },
            hl2: Vec::new(),
            hlc3: Vec::new(),
            ohlc4: Vec::new(),
            hlcc4: Vec::new(),
        };

        candles.precompute_fields();

        candles
    }

    pub fn new_with_fields(
        timestamp: Vec<i64>,
        open: Vec<f64>,
        high: Vec<f64>,
        low: Vec<f64>,
        close: Vec<f64>,
        volume: Vec<f64>,
        fields: CandleFieldFlags,
    ) -> Self {
        let mut candles = Candles {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
            fields,
            hl2: Vec::new(),
            hlc3: Vec::new(),
            ohlc4: Vec::new(),
            hlcc4: Vec::new(),
        };

        candles.precompute_fields();

        candles
    }

    pub fn get_timestamp(&self) -> Result<&[i64], Box<dyn Error>> {
        Ok(&self.timestamp)
    }

    fn compute_hl2(&self) -> Vec<f64> {
        self.high
            .iter()
            .zip(self.low.iter())
            .map(|(h, l)| (h + l) / 2.0)
            .collect()
    }

    fn compute_hlc3(&self) -> Vec<f64> {
        self.high
            .iter()
            .zip(self.low.iter())
            .zip(self.close.iter())
            .map(|((&h, &l), &c)| (h + l + c) / 3.0)
            .collect()
    }

    fn compute_ohlc4(&self) -> Vec<f64> {
        self.open
            .iter()
            .zip(self.high.iter())
            .zip(self.low.iter())
            .zip(self.close.iter())
            .map(|(((&o, &h), &l), &c)| (o + h + l + c) / 4.0)
            .collect()
    }

    fn compute_hlcc4(&self) -> Vec<f64> {
        self.high
            .iter()
            .zip(self.low.iter())
            .zip(self.close.iter())
            .map(|((&h, &l), &c)| (h + l + 2.0 * c) / 4.0)
            .collect()
    }

    pub fn get_calculated_field(&self, field: &str) -> Result<&[f64], Box<dyn std::error::Error>> {
        match field.to_lowercase().as_str() {
            "hl2" => Ok(&self.hl2),
            "hlc3" => Ok(&self.hlc3),
            "ohlc4" => Ok(&self.ohlc4),
            "hlcc4" => Ok(&self.hlcc4),
            _ => Err(format!("Invalid calculated field: {}", field).into()),
        }
    }

    pub fn select_candle_field(&self, field: &str) -> Result<&[f64], Box<dyn std::error::Error>> {
        match field.to_lowercase().as_str() {
            "open" => Ok(&self.open),
            "high" => Ok(&self.high),
            "low" => Ok(&self.low),
            "close" => Ok(&self.close),
            "volume" => Ok(&self.volume),
            _ => Err(format!("Invalid field: {}", field).into()),
        }
    }

    fn precompute_fields(&mut self) {
        let len = self.high.len();
        let mut hl2 = Vec::with_capacity(len);
        let mut hlc3 = Vec::with_capacity(len);
        let mut ohlc4 = Vec::with_capacity(len);
        let mut hlcc4 = Vec::with_capacity(len);

        for i in 0..len {
            let o = self.open[i];
            let h = self.high[i];
            let l = self.low[i];
            let c = self.close[i];

            hl2.push((h + l) / 2.0);
            hlc3.push((h + l + c) / 3.0);
            ohlc4.push((o + h + l + c) / 4.0);
            hlcc4.push((h + l + 2.0 * c) / 4.0);
        }

        self.hl2 = hl2;
        self.hlc3 = hlc3;
        self.ohlc4 = ohlc4;
        self.hlcc4 = hlcc4;
    }
}

pub fn read_candles_from_csv(file_path: &str) -> Result<Candles, Box<dyn Error>> {
    use std::io;

    let file = File::open(file_path)?;
    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);

    let header_len = rdr.headers().map(|h| h.len()).unwrap_or(0);
    if header_len < 2 {
        return Err("CSV must have at least 2 columns: timestamp, close".into());
    }

    let (fields, idx_open, idx_close, idx_high, idx_low, idx_volume) = if header_len >= 3 {
        (
            CandleFieldFlags {
                open: true,
                close: true,
                high: header_len > 3,
                low: header_len > 4,
                volume: header_len > 5,
            },
            Some(1usize),
            2usize,
            if header_len > 3 { Some(3usize) } else { None },
            if header_len > 4 { Some(4usize) } else { None },
            if header_len > 5 { Some(5usize) } else { None },
        )
    } else {
        (
            CandleFieldFlags {
                open: false,
                close: true,
                high: false,
                low: false,
                volume: false,
            },
            None,
            1usize,
            None,
            None,
            None,
        )
    };

    let mut timestamp = Vec::new();
    let mut open = Vec::new();
    let mut high = Vec::new();
    let mut low = Vec::new();
    let mut close = Vec::new();
    let mut volume = Vec::new();

    for result in rdr.records() {
        let record = result?;

        let ts: i64 = record
            .get(0)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing timestamp column"))?
            .parse()?;
        let c: f64 = record
            .get(idx_close)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing close column"))?
            .parse()?;
        timestamp.push(ts);
        close.push(c);

        let o: f64 = match idx_open {
            Some(i) => record
                .get(i)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing open column"))?
                .parse()?,
            None => f64::NAN,
        };
        open.push(o);

        let h: f64 = match idx_high {
            Some(i) => record
                .get(i)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing high column"))?
                .parse()?,
            None => f64::NAN,
        };
        high.push(h);

        let l: f64 = match idx_low {
            Some(i) => record
                .get(i)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing low column"))?
                .parse()?,
            None => f64::NAN,
        };
        low.push(l);

        let v: f64 = match idx_volume {
            Some(i) => record
                .get(i)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing volume column"))?
                .parse()?,
            None => f64::NAN,
        };
        volume.push(v);
    }

    Ok(Candles::new_with_fields(
        timestamp,
        open,
        high,
        low,
        close,
        volume,
        fields,
    ))
}

pub fn source_type<'a>(candles: &'a Candles, source: &str) -> &'a [f64] {
    if source.eq_ignore_ascii_case("open") {
        &candles.open
    } else if source.eq_ignore_ascii_case("high") {
        &candles.high
    } else if source.eq_ignore_ascii_case("low") {
        &candles.low
    } else if source.eq_ignore_ascii_case("close") {
        &candles.close
    } else if source.eq_ignore_ascii_case("volume") {
        &candles.volume
    } else if source.eq_ignore_ascii_case("hl2") {
        &candles.hl2
    } else if source.eq_ignore_ascii_case("hlc3") {
        &candles.hlc3
    } else if source.eq_ignore_ascii_case("ohlc4") {
        &candles.ohlc4
    } else if source.eq_ignore_ascii_case("hlcc4") || source.eq_ignore_ascii_case("hlcc") {
        &candles.hlcc4
    } else {
        eprintln!("Warning: Invalid price source '{source}'. Defaulting to 'close'.");
        &candles.close
    }
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

        let hl2 = candles
            .get_calculated_field("hl2")
            .expect("Failed to get HL2");
        let hlc3 = candles
            .get_calculated_field("hlc3")
            .expect("Failed to get HLC3");
        let ohlc4 = candles
            .get_calculated_field("ohlc4")
            .expect("Failed to get OHLC4");
        let hlcc4 = candles
            .get_calculated_field("hlcc4")
            .expect("Failed to get HLCC4");

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
        compare_last_five(hl2, &expected_last_5_hl2, "HL2");
        compare_last_five(hlc3, &expected_last_5_hlc3, "HLC3");
        compare_last_five(ohlc4, &expected_last_5_ohlc4, "OHLC4");
        compare_last_five(hlcc4, &expected_last_5_hlcc4, "HLCC4");
    }

    #[test]
    fn test_precompute_fields_direct() {
        let timestamp = vec![1, 2, 3];
        let open = vec![100.0, 200.0, 300.0];
        let high = vec![110.0, 220.0, 330.0];
        let low = vec![90.0, 180.0, 270.0];
        let close = vec![105.0, 190.0, 310.0];
        let volume = vec![1000.0, 2000.0, 3000.0];

        let candles = Candles::new(timestamp, open, high, low, close, volume);

        let hl2 = candles.get_calculated_field("hl2").unwrap();
        assert_eq!(hl2, &[100.0, 200.0, 300.0]);

        let hlc3 = candles.get_calculated_field("hlc3").unwrap();
        let expected_hlc3 = &[101.6667, 196.6667, 303.3333];
        for (actual, expected) in hlc3.iter().zip(expected_hlc3.iter()) {
            assert!((actual - expected).abs() < 1e-4);
        }

        let ohlc4 = candles.get_calculated_field("ohlc4").unwrap();
        let expected_ohlc4 = &[101.25, 197.5, 302.5];
        for (actual, expected) in ohlc4.iter().zip(expected_ohlc4.iter()) {
            assert!((actual - expected).abs() < 1e-4);
        }

        let hlcc4 = candles.get_calculated_field("hlcc4").unwrap();
        let expected_hlcc4 = &[102.5, 195.0, 305.0];
        for (actual, expected) in hlcc4.iter().zip(expected_hlcc4.iter()) {
            assert!((actual - expected).abs() < 1e-4);
        }
    }
}
