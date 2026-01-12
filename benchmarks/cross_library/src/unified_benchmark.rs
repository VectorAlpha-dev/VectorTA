use std::collections::HashMap;
use std::time::{Duration, Instant};
use std::path::Path;
use crate::ffi_overhead::FfiOverheadProfile;
use crate::benchmark_methodology::{ComparisonMode, BenchmarkResult};
use crate::json_export::BenchmarkJsonExport;


#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LibraryType {
    RustNative,
    RustFFI,
    TulipFFI,
    TalibFFI,
}

impl LibraryType {

    pub fn uses_ffi(&self) -> bool {
        matches!(self, LibraryType::RustFFI | LibraryType::TulipFFI | LibraryType::TalibFFI)
    }


    pub fn display_name(&self) -> &str {
        match self {
            LibraryType::RustNative => "Rust Native",
            LibraryType::RustFFI => "Rust FFI",
            LibraryType::TulipFFI => "Tulip",
            LibraryType::TalibFFI => "TA-LIB",
        }
    }
}


#[derive(Debug, Clone)]
pub struct UnifiedMeasurement {
    pub indicator: String,
    pub library: LibraryType,
    pub raw_duration: Duration,
    pub data_size: usize,
    pub iterations: usize,
}

impl UnifiedMeasurement {

    pub fn new(
        indicator: String,
        library: LibraryType,
        raw_duration: Duration,
        data_size: usize,
        iterations: usize,
    ) -> Self {
        Self {
            indicator,
            library,
            raw_duration,
            data_size,
            iterations,
        }
    }


    pub fn average_duration(&self) -> Duration {
        self.raw_duration / self.iterations as u32
    }


    pub fn throughput_mops(&self) -> f64 {
        let ops = self.data_size as f64;
        let seconds = self.average_duration().as_secs_f64();
        (ops / seconds) / 1_000_000.0
    }
}


pub struct UnifiedBenchmarkRunner {
    measurements: Vec<UnifiedMeasurement>,
    ffi_profile: Option<FfiOverheadProfile>,
    cached_results: HashMap<(String, LibraryType, ComparisonMode), BenchmarkResult>,
}

impl UnifiedBenchmarkRunner {

    pub fn new() -> Self {
        Self {
            measurements: Vec::new(),
            ffi_profile: None,
            cached_results: HashMap::new(),
        }
    }


    pub fn profile_ffi_overhead(&mut self, data_size: usize, iterations: usize) {
        println!("📊 Profiling FFI overhead...");
        let profile = FfiOverheadProfile::profile(data_size, iterations);
        println!("  ✓ Call overhead: {:.2} ns", profile.call_overhead_ns);
        println!("  ✓ Marshalling: {:.2} ns/KB", profile.marshalling_overhead_ns_per_kb);
        println!("  ✓ Validation: {:.2} ns", profile.validation_overhead_ns);
        self.ffi_profile = Some(profile);
    }


    pub fn add_measurement(&mut self, measurement: UnifiedMeasurement) {
        self.measurements.push(measurement);

        self.cached_results.clear();
    }


    pub fn benchmark<F>(
        &mut self,
        indicator: &str,
        library: LibraryType,
        data_size: usize,
        iterations: usize,
        mut bench_fn: F,
    ) where
        F: FnMut(),
    {

        for _ in 0..10 {
            bench_fn();
        }


        let start = Instant::now();
        for _ in 0..iterations {
            bench_fn();
        }
        let duration = start.elapsed();

        self.add_measurement(UnifiedMeasurement::new(
            indicator.to_string(),
            library,
            duration,
            data_size,
            iterations,
        ));
    }


    pub fn analyze(
        &mut self,
        measurement: &UnifiedMeasurement,
        mode: ComparisonMode,
    ) -> BenchmarkResult {

        let cache_key = (
            measurement.indicator.clone(),
            measurement.library.clone(),
            mode.clone(),
        );

        if let Some(cached) = self.cached_results.get(&cache_key) {
            return cached.clone();
        }


        let avg_duration = measurement.average_duration();

        let (final_duration, ffi_compensated) = match mode {
            ComparisonMode::RawPerformance => {

                (avg_duration, None)
            }
            ComparisonMode::AlgorithmEfficiency => {

                if measurement.library.uses_ffi() {
                    if let Some(ref profile) = self.ffi_profile {
                        let overhead = profile.estimate_overhead(measurement.data_size * 8);
                        let compensated = avg_duration.saturating_sub(overhead);
                        (compensated, Some(compensated))
                    } else {
                        (avg_duration, None)
                    }
                } else {
                    (avg_duration, None)
                }
            }
            ComparisonMode::EqualFooting => {

                if measurement.library.uses_ffi() || measurement.library == LibraryType::RustFFI {
                    (avg_duration, None)
                } else {

                    if let Some(ffi_measurement) = self.find_ffi_equivalent(measurement) {
                        (ffi_measurement.average_duration(), None)
                    } else {
                        (avg_duration, None)
                    }
                }
            }
        };

        let result = BenchmarkResult {
            library: measurement.library.display_name().to_string(),
            indicator: measurement.indicator.clone(),
            data_size: measurement.data_size,
            raw_duration: final_duration,
            ffi_compensated_duration: ffi_compensated,
            throughput_mops: (measurement.data_size as f64 / final_duration.as_secs_f64()) / 1_000_000.0,
        };


        self.cached_results.insert(cache_key, result.clone());

        result
    }


    fn find_ffi_equivalent(&self, native: &UnifiedMeasurement) -> Option<&UnifiedMeasurement> {
        self.measurements.iter().find(|m| {
            m.indicator == native.indicator
                && m.library == LibraryType::RustFFI
                && m.data_size == native.data_size
        })
    }


    pub fn generate_comparison(&mut self, mode: ComparisonMode) -> Vec<BenchmarkResult> {
        let filtered: Vec<_> = self.measurements
            .iter()
            .filter(|m| {

                if matches!(mode, ComparisonMode::EqualFooting) {
                    m.library.uses_ffi()
                } else {
                    true
                }
            })
            .cloned()
            .collect();

        filtered
            .into_iter()
            .map(|m| self.analyze(&m, mode.clone()))
            .collect()
    }


    pub fn generate_full_report(&mut self) -> String {
        let mut report = String::new();

        report.push_str("\n📊 THREE-TIER BENCHMARK REPORT\n");
        report.push_str("=" .repeat(80).as_str());
        report.push_str("\n\n");


        if let Some(ref profile) = self.ffi_profile {
            report.push_str("📈 FFI Overhead Profile:\n");
            report.push_str(&format!("  • Call overhead: {:.2} ns\n", profile.call_overhead_ns));
            report.push_str(&format!("  • Marshalling: {:.2} ns/KB\n", profile.marshalling_overhead_ns_per_kb));
            report.push_str(&format!("  • Validation: {:.2} ns\n\n", profile.validation_overhead_ns));
        }


        for mode in [
            ComparisonMode::RawPerformance,
            ComparisonMode::AlgorithmEfficiency,
            ComparisonMode::EqualFooting,
        ] {
            report.push_str(&self.format_comparison_table(mode));
            report.push_str("\n");
        }

        report
    }


    fn format_comparison_table(&mut self, mode: ComparisonMode) -> String {
        let mut output = String::new();

        let mode_name = match mode {
            ComparisonMode::RawPerformance => "Raw Performance (User Experience)",
            ComparisonMode::AlgorithmEfficiency => "Algorithm Efficiency (FFI-Compensated)",
            ComparisonMode::EqualFooting => "Equal Footing (All FFI)",
        };

        output.push_str(&format!("\n{}\n", mode_name));
        output.push_str(&"=".repeat(mode_name.len()));
        output.push_str("\n\n");


        let mut indicators: HashMap<String, Vec<BenchmarkResult>> = HashMap::new();
        for result in self.generate_comparison(mode.clone()) {
            indicators
                .entry(result.indicator.clone())
                .or_insert_with(Vec::new)
                .push(result);
        }

        for (indicator, mut results) in indicators {
            output.push_str(&format!("📌 {}\n", indicator));
            output.push_str(&format!("{:<15} {:>12} {:>12} {:>12}\n",
                "Library", "Time (µs)", "Throughput", "vs Best"));
            output.push_str(&"-".repeat(55));
            output.push_str("\n");


            results.sort_by_key(|r| r.raw_duration);
            let best_time = results[0].raw_duration;

            for result in results {
                let time_us = result.raw_duration.as_secs_f64() * 1_000_000.0;
                let vs_best = if result.raw_duration == best_time {
                    "baseline".to_string()
                } else {
                    format!("+{:.1}%",
                        (result.raw_duration.as_secs_f64() / best_time.as_secs_f64() - 1.0) * 100.0)
                };

                output.push_str(&format!("{:<15} {:>12.2} {:>12.2} {:>12}\n",
                    result.library,
                    time_us,
                    result.throughput_mops,
                    vs_best
                ));
            }
            output.push_str("\n");
        }

        output
    }


    pub fn get_statistics(&self) -> String {
        let mut stats = String::new();

        stats.push_str("\n📊 Benchmark Statistics:\n");
        stats.push_str(&format!("  • Total measurements: {}\n", self.measurements.len()));

        let indicators: std::collections::HashSet<_> =
            self.measurements.iter().map(|m| &m.indicator).collect();
        stats.push_str(&format!("  • Indicators tested: {}\n", indicators.len()));

        let libraries: std::collections::HashSet<_> =
            self.measurements.iter().map(|m| &m.library).collect();
        stats.push_str(&format!("  • Libraries compared: {}\n", libraries.len()));

        if let Some(ref profile) = self.ffi_profile {
            let avg_data_size = self.measurements.iter()
                .map(|m| m.data_size)
                .sum::<usize>() / self.measurements.len().max(1);
            let est_overhead = profile.estimate_overhead(avg_data_size * 8);
            stats.push_str(&format!("  • Avg FFI overhead: {:.2} µs\n",
                est_overhead.as_secs_f64() * 1_000_000.0));
        }

        stats
    }


    pub fn export_to_json(&self) -> BenchmarkJsonExport {
        BenchmarkJsonExport::from_measurements(
            &self.measurements,
            self.ffi_profile.as_ref(),
        )
    }


    pub fn save_json(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let export = self.export_to_json();
        export.save_to_file(path)?;
        println!("✅ Benchmark results saved to: {}", path.display());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unified_runner() {
        let mut runner = UnifiedBenchmarkRunner::new();


        runner.profile_ffi_overhead(1000, 100);


        runner.add_measurement(UnifiedMeasurement::new(
            "SMA".to_string(),
            LibraryType::RustNative,
            Duration::from_micros(100),
            10000,
            100,
        ));

        runner.add_measurement(UnifiedMeasurement::new(
            "SMA".to_string(),
            LibraryType::RustFFI,
            Duration::from_micros(125),
            10000,
            100,
        ));

        runner.add_measurement(UnifiedMeasurement::new(
            "SMA".to_string(),
            LibraryType::TulipFFI,
            Duration::from_micros(150),
            10000,
            100,
        ));


        let raw_results = runner.generate_comparison(ComparisonMode::RawPerformance);
        assert_eq!(raw_results.len(), 3);

        let equal_results = runner.generate_comparison(ComparisonMode::EqualFooting);
        assert_eq!(equal_results.len(), 2);


        let report = runner.generate_full_report();
        assert!(report.contains("Raw Performance"));
        assert!(report.contains("Algorithm Efficiency"));
        assert!(report.contains("Equal Footing"));
    }
}