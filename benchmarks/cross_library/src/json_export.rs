use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use crate::unified_benchmark::{UnifiedMeasurement, LibraryType};
use crate::benchmark_methodology::ComparisonMode;
use crate::ffi_overhead::FfiOverheadProfile;

/// Main structure for exporting benchmark results to JSON
#[derive(Debug, Serialize, Deserialize)]
pub struct BenchmarkJsonExport {
    /// Metadata about the benchmark run
    pub metadata: BenchmarkMetadata,

    /// FFI overhead measurements
    pub ffi_profile: Option<FfiProfileData>,

    /// Results for each indicator
    pub indicators: Vec<IndicatorResults>,

    /// Summary statistics
    pub summary: BenchmarkSummary,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BenchmarkMetadata {
    pub timestamp: String,
    pub total_measurements: usize,
    pub data_sizes_tested: Vec<String>,
    pub libraries_tested: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FfiProfileData {
    pub call_overhead_ns: f64,
    pub marshalling_ns_per_kb: f64,
    pub validation_overhead_ns: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct IndicatorResults {
    pub name: String,
    pub data_size: usize,
    pub measurements: Vec<LibraryMeasurement>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LibraryMeasurement {
    pub library: String,
    pub raw_time_us: f64,
    pub throughput_mops: f64,
    pub ffi_compensated_time_us: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BenchmarkSummary {
    pub fastest_by_indicator: HashMap<String, String>,
    pub average_throughput_by_library: HashMap<String, f64>,
}

impl BenchmarkJsonExport {
    /// Create a new JSON export from measurements
    pub fn from_measurements(
        measurements: &[UnifiedMeasurement],
        ffi_profile: Option<&FfiOverheadProfile>,
    ) -> Self {
        let timestamp = chrono::Utc::now().to_rfc3339();

        // Extract unique data sizes and libraries
        let mut data_sizes = std::collections::HashSet::new();
        let mut libraries = std::collections::HashSet::new();

        for measurement in measurements {
            data_sizes.insert(measurement.data_size);
            libraries.insert(measurement.library.display_name());
        }

        let mut data_sizes_vec: Vec<String> = data_sizes
            .iter()
            .map(|&size| format_data_size(size))
            .collect();
        data_sizes_vec.sort();

        let mut libraries_vec: Vec<String> = libraries
            .iter()
            .map(|&lib| lib.to_string())
            .collect();
        libraries_vec.sort();

        // Convert FFI profile if available
        let ffi_profile_data = ffi_profile.map(|profile| FfiProfileData {
            call_overhead_ns: profile.call_overhead_ns,
            marshalling_ns_per_kb: profile.marshalling_overhead_ns_per_kb,
            validation_overhead_ns: profile.validation_overhead_ns,
        });

        // Group measurements by indicator and data size
        let mut indicator_map: HashMap<(String, usize), Vec<&UnifiedMeasurement>> = HashMap::new();
        for measurement in measurements {
            let key = (measurement.indicator.clone(), measurement.data_size);
            indicator_map.entry(key).or_insert_with(Vec::new).push(measurement);
        }

        // Convert to indicator results
        let mut indicators = Vec::new();
        for ((indicator_name, data_size), measurements) in indicator_map {
            let mut library_measurements = Vec::new();

            for measurement in measurements {
                let avg_duration = measurement.average_duration();
                let raw_time_us = avg_duration.as_secs_f64() * 1_000_000.0;

                // Calculate FFI compensated time if applicable
                let ffi_compensated_time_us = if measurement.library.uses_ffi() {
                    ffi_profile.map(|profile| {
                        let overhead = profile.estimate_overhead(measurement.data_size * 8);
                        let compensated = avg_duration.saturating_sub(overhead);
                        compensated.as_secs_f64() * 1_000_000.0
                    })
                } else {
                    None
                };

                library_measurements.push(LibraryMeasurement {
                    library: measurement.library.display_name().to_string(),
                    raw_time_us,
                    throughput_mops: measurement.throughput_mops(),
                    ffi_compensated_time_us,
                });
            }

            indicators.push(IndicatorResults {
                name: indicator_name,
                data_size,
                measurements: library_measurements,
            });
        }

        // Sort indicators by name and data size
        indicators.sort_by(|a, b| {
            a.name.cmp(&b.name).then(a.data_size.cmp(&b.data_size))
        });

        // Calculate summary statistics
        let summary = calculate_summary(&indicators);

        BenchmarkJsonExport {
            metadata: BenchmarkMetadata {
                timestamp,
                total_measurements: measurements.len(),
                data_sizes_tested: data_sizes_vec,
                libraries_tested: libraries_vec,
            },
            ffi_profile: ffi_profile_data,
            indicators,
            summary,
        }
    }

    /// Save the export to a JSON file
    pub fn save_to_file(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        let mut file = File::create(path)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    /// Convert to JSON string
    pub fn to_json_string(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

fn format_data_size(size: usize) -> String {
    if size >= 1_000_000 {
        format!("{}M", size / 1_000_000)
    } else if size >= 1_000 {
        format!("{}k", size / 1_000)
    } else {
        format!("{}", size)
    }
}

fn calculate_summary(indicators: &[IndicatorResults]) -> BenchmarkSummary {
    let mut fastest_by_indicator = HashMap::new();
    let mut throughput_by_library: HashMap<String, Vec<f64>> = HashMap::new();

    for indicator in indicators {
        // Find fastest library for this indicator
        if let Some(fastest) = indicator.measurements
            .iter()
            .min_by(|a, b| a.raw_time_us.partial_cmp(&b.raw_time_us).unwrap())
        {
            fastest_by_indicator.insert(
                format!("{} ({})", indicator.name, format_data_size(indicator.data_size)),
                fastest.library.clone(),
            );
        }

        // Collect throughput data
        for measurement in &indicator.measurements {
            throughput_by_library
                .entry(measurement.library.clone())
                .or_insert_with(Vec::new)
                .push(measurement.throughput_mops);
        }
    }

    // Calculate average throughput
    let average_throughput_by_library = throughput_by_library
        .into_iter()
        .map(|(library, throughputs)| {
            let avg = throughputs.iter().sum::<f64>() / throughputs.len() as f64;
            (library, avg)
        })
        .collect();

    BenchmarkSummary {
        fastest_by_indicator,
        average_throughput_by_library,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unified_benchmark::UnifiedMeasurement;
    use std::time::Duration;

    #[test]
    fn test_json_export() {
        let measurements = vec![
            UnifiedMeasurement::new(
                "SMA".to_string(),
                LibraryType::RustNative,
                Duration::from_micros(100),
                10000,
                100,
            ),
            UnifiedMeasurement::new(
                "SMA".to_string(),
                LibraryType::TulipFFI,
                Duration::from_micros(150),
                10000,
                100,
            ),
        ];

        let export = BenchmarkJsonExport::from_measurements(&measurements, None);

        assert_eq!(export.metadata.total_measurements, 2);
        assert_eq!(export.indicators.len(), 1);
        assert_eq!(export.indicators[0].measurements.len(), 2);

        // Test JSON serialization
        let json = export.to_json_string().unwrap();
        assert!(json.contains("\"SMA\""));
        assert!(json.contains("\"Rust Native\""));
        assert!(json.contains("\"Tulip\""));
    }
}