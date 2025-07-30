use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use plotters::prelude::*;
use tabled::{Table, Tabled};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub indicator: String,
    pub data_size: String,
    pub library: String,
    pub mean_time_ms: f64,
    pub median_time_ms: f64,
    pub throughput_mb_per_sec: f64,
}

#[derive(Debug, Clone, Tabled)]
pub struct ComparisonRow {
    #[tabled(rename = "Indicator")]
    pub indicator: String,
    #[tabled(rename = "Data Size")]
    pub data_size: String,
    #[tabled(rename = "Rust (ms)")]
    pub rust_time: f64,
    #[tabled(rename = "Rust FFI (ms)")]
    pub rust_ffi_time: f64,
    #[tabled(rename = "Tulip (ms)")]
    pub tulip_time: f64,
    #[tabled(rename = "TA-Lib (ms)")]
    pub talib_time: Option<f64>,
    #[tabled(rename = "Rust/Tulip")]
    pub rust_vs_tulip: f64,
    #[tabled(rename = "FFI Overhead")]
    pub ffi_overhead_pct: f64,
}

pub struct ReportGenerator {
    results: Vec<BenchmarkResult>,
}

impl ReportGenerator {
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
        }
    }

    pub fn add_result(&mut self, result: BenchmarkResult) {
        self.results.push(result);
    }

    pub fn generate_html_report(&self, output_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let mut html = String::from(r#"
<!DOCTYPE html>
<html>
<head>
    <title>Cross-Library TA Indicator Benchmark Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1, h2 { color: #333; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        .fast { color: green; font-weight: bold; }
        .moderate { color: orange; }
        .slow { color: red; }
        img { max-width: 100%; height: auto; margin: 20px 0; }
    </style>
</head>
<body>
    <h1>Cross-Library Technical Analysis Indicator Benchmark</h1>
    <p>Comparing Rust-TA indicators against Tulip C and TA-Lib</p>
"#);

        // Group results by indicator and data size
        let mut grouped: HashMap<(String, String), HashMap<String, BenchmarkResult>> = HashMap::new();
        
        for result in &self.results {
            let key = (result.indicator.clone(), result.data_size.clone());
            grouped.entry(key)
                .or_insert_with(HashMap::new)
                .insert(result.library.clone(), result.clone());
        }

        // Create comparison table
        html.push_str("<h2>Performance Comparison</h2>\n<table>\n");
        html.push_str("<tr><th>Indicator</th><th>Data Size</th><th>Rust (ms)</th><th>Rust FFI (ms)</th>");
        html.push_str("<th>Tulip (ms)</th><th>TA-Lib (ms)</th><th>Rust/Tulip</th><th>FFI Overhead %</th></tr>\n");

        for ((indicator, data_size), libraries) in grouped.iter() {
            let rust_result = libraries.get("rust");
            let rust_ffi_result = libraries.get("rust_ffi");
            let tulip_result = libraries.get("tulip");
            let talib_result = libraries.get("talib");

            if let (Some(rust), Some(tulip)) = (rust_result, tulip_result) {
                let rust_time = rust.median_time_ms;
                let rust_ffi_time = rust_ffi_result.map(|r| r.median_time_ms).unwrap_or(rust_time);
                let tulip_time = tulip.median_time_ms;
                let talib_time = talib_result.map(|r| r.median_time_ms);
                
                let ratio = rust_time / tulip_time;
                let ffi_overhead = ((rust_ffi_time / rust_time) - 1.0) * 100.0;
                
                let ratio_class = if ratio < 0.8 { "fast" } 
                                 else if ratio < 1.2 { "moderate" } 
                                 else { "slow" };

                html.push_str(&format!(
                    "<tr><td>{}</td><td>{}</td><td>{:.3}</td><td>{:.3}</td><td>{:.3}</td><td>{}</td><td class='{}'>{:.2}x</td><td>{:.1}%</td></tr>\n",
                    indicator, data_size, rust_time, rust_ffi_time, tulip_time,
                    talib_time.map(|t| format!("{:.3}", t)).unwrap_or_else(|| "N/A".to_string()),
                    ratio_class, ratio, ffi_overhead
                ));
            }
        }
        
        html.push_str("</table>\n");

        // Add charts
        html.push_str("<h2>Performance Charts</h2>\n");
        
        // Generate and embed charts
        self.generate_comparison_chart("comparison_chart.png")?;
        html.push_str("<img src='comparison_chart.png' alt='Performance Comparison Chart'>\n");

        // Add summary statistics
        html.push_str("<h2>Summary</h2>\n<ul>\n");
        
        let rust_wins = grouped.iter()
            .filter(|(_, libs)| {
                if let (Some(rust), Some(tulip)) = (libs.get("rust"), libs.get("tulip")) {
                    rust.median_time_ms < tulip.median_time_ms
                } else {
                    false
                }
            })
            .count();
        
        let total_comparisons = grouped.len();
        let win_percentage = (rust_wins as f64 / total_comparisons as f64) * 100.0;
        
        html.push_str(&format!("<li>Rust outperforms Tulip in {}/{} cases ({:.1}%)</li>\n", 
                               rust_wins, total_comparisons, win_percentage));
        
        // Calculate average speedup
        let speedups: Vec<f64> = grouped.iter()
            .filter_map(|(_, libs)| {
                if let (Some(rust), Some(tulip)) = (libs.get("rust"), libs.get("tulip")) {
                    Some(tulip.median_time_ms / rust.median_time_ms)
                } else {
                    None
                }
            })
            .collect();
        
        if !speedups.is_empty() {
            let avg_speedup = speedups.iter().sum::<f64>() / speedups.len() as f64;
            html.push_str(&format!("<li>Average speedup vs Tulip: {:.2}x</li>\n", avg_speedup));
        }

        html.push_str("</ul>\n</body>\n</html>");

        let mut file = File::create(output_path)?;
        file.write_all(html.as_bytes())?;
        
        Ok(())
    }

    fn generate_comparison_chart(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let root = BitMapBackend::new(filename, (1024, 768)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption("Technical Indicator Performance Comparison", ("Arial", 30))
            .margin(20)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(
                0f64..10f64,
                0f64..100f64,
            )?;

        chart.configure_mesh()
            .x_desc("Indicator")
            .y_desc("Time (ms)")
            .draw()?;

        // TODO: Add actual data plotting

        root.present()?;
        Ok(())
    }

    pub fn generate_csv_report(&self, output_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let mut wtr = csv::Writer::from_path(output_path)?;
        
        wtr.write_record(&[
            "indicator", "data_size", "library", "mean_ms", "median_ms", "throughput_mb_s"
        ])?;

        for result in &self.results {
            wtr.write_record(&[
                &result.indicator,
                &result.data_size,
                &result.library,
                &result.mean_time_ms.to_string(),
                &result.median_time_ms.to_string(),
                &result.throughput_mb_per_sec.to_string(),
            ])?;
        }

        wtr.flush()?;
        Ok(())
    }
}