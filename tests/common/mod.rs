/// Common test utilities for comparing binding outputs with Rust outputs
use vector_ta::utilities::data_loader::{read_candles_from_csv, Candles};
use std::error::Error;

/// Test data holder that matches the structure used in Python/WASM tests
pub struct TestData {
    pub candles: Candles,
}

impl TestData {
    /// Load test data from the standard CSV file
    pub fn load() -> Result<Self, Box<dyn Error>> {
        let candles = read_candles_from_csv("src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv")?;
        Ok(TestData { candles })
    }

    /// Get close prices as slice
    pub fn close_prices(&self) -> &[f64] {
        &self.candles.close
    }

    /// Get high prices as slice
    pub fn high_prices(&self) -> &[f64] {
        &self.candles.high
    }

    /// Get low prices as slice
    pub fn low_prices(&self) -> &[f64] {
        &self.candles.low
    }

    /// Get open prices as slice
    pub fn open_prices(&self) -> &[f64] {
        &self.candles.open
    }

    /// Get volume as slice
    pub fn volume(&self) -> &[f64] {
        &self.candles.volume
    }
}

/// Compare two arrays with a tolerance
pub fn assert_array_close(actual: &[f64], expected: &[f64], rtol: f64, atol: f64, name: &str) {
    assert_eq!(actual.len(), expected.len(),
        "{}: Length mismatch: actual {} vs expected {}", name, actual.len(), expected.len());

    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        if a.is_nan() && e.is_nan() {
            continue; 
        }

        let diff = (a - e).abs();
        let tol = atol + rtol * e.abs();

        assert!(diff <= tol,
            "{}: Value mismatch at index {}: actual {} vs expected {} (diff: {}, tol: {})",
            name, i, a, e, diff, tol);
    }
}