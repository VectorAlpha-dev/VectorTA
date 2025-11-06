use std::sync::Mutex;
use std::time::Duration;
use std::path::Path;
use crate::unified_benchmark::{UnifiedMeasurement, LibraryType};
use crate::json_export::BenchmarkJsonExport;

/// Global collector for benchmark measurements
/// This collects all measurements during the benchmark run
/// and exports them to JSON at the end
pub struct BenchmarkCollector {
    measurements: Mutex<Vec<UnifiedMeasurement>>,
}

impl BenchmarkCollector {
    /// Create a new collector
    pub fn new() -> Self {
        Self {
            measurements: Mutex::new(Vec::new()),
        }
    }

    /// Add a measurement
    pub fn add_measurement(
        &self,
        indicator: &str,
        library: LibraryType,
        duration_total: Duration,
        data_size: usize,
        iterations: usize,
    ) {
        let measurement = UnifiedMeasurement::new(
            indicator.to_string(),
            library,
            duration_total,
            data_size,
            iterations.max(1),
        );

        if let Ok(mut measurements) = self.measurements.lock() {
            measurements.push(measurement);
        }
    }

    /// Export all collected measurements to JSON
    pub fn export_to_json(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let measurements = self.measurements.lock().unwrap();

        if measurements.is_empty() {
            println!("⚠️ No measurements collected, skipping JSON export");
            return Ok(());
        }

        let export = BenchmarkJsonExport::from_measurements(&measurements, None);
        export.save_to_file(path)?;

        println!("\n✅ Benchmark results exported to: {}", path.display());
        println!("   Total measurements: {}", measurements.len());

        // Count unique indicators and libraries
        let mut indicators = std::collections::HashSet::new();
        let mut libraries = std::collections::HashSet::new();
        for m in measurements.iter() {
            indicators.insert(&m.indicator);
            libraries.insert(&m.library);
        }

        println!("   Indicators tested: {}", indicators.len());
        println!("   Libraries compared: {}", libraries.len());

        Ok(())
    }
}

// Global instance
lazy_static::lazy_static! {
    pub static ref COLLECTOR: BenchmarkCollector = BenchmarkCollector::new();
}
