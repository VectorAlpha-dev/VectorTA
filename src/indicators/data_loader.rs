extern crate csv;
extern crate lazy_static;
extern crate serde;

use csv::ReaderBuilder;
use std::error::Error;
use std::fs::File;
use std::sync::atomic::{AtomicUsize, Ordering};

static LOAD_COUNTER: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Clone)]
pub struct Candles {
    pub timestamp: Vec<f64>,
    pub open: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub close: Vec<f64>,
    pub volume: Vec<f64>,
}

impl Candles {
    pub fn new(
        timestamp: Vec<f64>,
        open: Vec<f64>,
        close: Vec<f64>,
        high: Vec<f64>,
        low: Vec<f64>,
        volume: Vec<f64>,
    ) -> Self {
        Candles {
            timestamp,
            open,
            close,
            high,
            low,
            volume,
        }
    }

    pub fn select_candle_field(&self, field: &str) -> Result<&[f64], Box<dyn Error>> {
        match field.to_lowercase().as_str() {
            "timestamp" => Ok(&self.timestamp),
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
        timestamp.push(record[0].parse()?);
        open.push(record[1].parse()?);
        high.push(record[2].parse()?);
        low.push(record[3].parse()?);
        close.push(record[4].parse()?);
        volume.push(record[5].parse()?);
    }

    LOAD_COUNTER.fetch_add(1, Ordering::SeqCst);

    Ok(Candles::new(timestamp, open, high, low, close, volume))
}
