use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use cross_library_benchmark::{tulip, rust_ffi};
use cross_library_benchmark::utils::CandleData;
use my_project::indicators;
use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

// Data sizes to benchmark
const DATA_SIZES: &[(&str, &str)] = &[
    ("4k", "../../src/data/4kCandles.csv"),
    ("10k", "../../src/data/10kCandles.csv"),
    ("100k", "../../src/data/bitfinex btc-usd 100,000 candles ends 09-01-24.csv"),
    ("1M", "../../src/data/1MillionCandles.csv"),
];

// Indicator mappings between libraries
struct IndicatorMapping {
    rust_name: &'static str,
    tulip_name: &'static str,
    talib_name: Option<&'static str>,
    inputs: Vec<&'static str>,
    options: Vec<f64>,
}

fn get_indicator_mappings() -> Vec<IndicatorMapping> {
    vec![
        // Simple Moving Average
        IndicatorMapping {
            rust_name: "sma",
            tulip_name: "sma",
            talib_name: Some("SMA"),
            inputs: vec!["close"],
            options: vec![14.0],
        },
        // Exponential Moving Average
        IndicatorMapping {
            rust_name: "ema",
            tulip_name: "ema",
            talib_name: Some("EMA"),
            inputs: vec!["close"],
            options: vec![14.0],
        },
        // Relative Strength Index
        IndicatorMapping {
            rust_name: "rsi",
            tulip_name: "rsi",
            talib_name: Some("RSI"),
            inputs: vec!["close"],
            options: vec![14.0],
        },
        // Bollinger Bands
        IndicatorMapping {
            rust_name: "bollinger_bands",
            tulip_name: "bbands",
            talib_name: Some("BBANDS"),
            inputs: vec!["close"],
            options: vec![20.0, 2.0],
        },
        // MACD
        IndicatorMapping {
            rust_name: "macd",
            tulip_name: "macd",
            talib_name: Some("MACD"),
            inputs: vec!["close"],
            options: vec![12.0, 26.0, 9.0],
        },
        // Average True Range
        IndicatorMapping {
            rust_name: "atr",
            tulip_name: "atr",
            talib_name: Some("ATR"),
            inputs: vec!["high", "low", "close"],
            options: vec![14.0],
        },
        // Stochastic
        IndicatorMapping {
            rust_name: "stoch",
            tulip_name: "stoch",
            talib_name: Some("STOCH"),
            inputs: vec!["high", "low", "close"],
            options: vec![14.0, 3.0, 3.0],
        },
        // Aroon
        IndicatorMapping {
            rust_name: "aroon",
            tulip_name: "aroon",
            talib_name: Some("AROON"),
            inputs: vec!["high", "low"],
            options: vec![14.0],
        },
        // ADX
        IndicatorMapping {
            rust_name: "adx",
            tulip_name: "adx",
            talib_name: Some("ADX"),
            inputs: vec!["high", "low"],
            options: vec![14.0],
        },
        // CCI
        IndicatorMapping {
            rust_name: "cci",
            tulip_name: "cci",
            talib_name: Some("CCI"),
            inputs: vec!["high", "low", "close"],
            options: vec![14.0],
        },
    ]
}

fn benchmark_rust_indicator(
    c: &mut Criterion,
    indicator: &IndicatorMapping,
    data: &CandleData,
    size_name: &str,
) {
    let group_name = format!("{}/{}", indicator.rust_name, size_name);
    let mut group = c.benchmark_group(&group_name);
    
    // Set throughput based on data size
    group.throughput(Throughput::Elements(data.len() as u64));
    group.measurement_time(Duration::from_millis(900));
    group.warm_up_time(Duration::from_millis(150));
    group.sample_size(10);

    // Benchmark Rust implementation (direct call)
    match indicator.rust_name {
        "sma" => {
            let input = indicators::sma::SmaInput::from_slice(&data.close, indicators::sma::SmaParams { period: Some(14) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(indicators::sma::sma(input));
                });
            });
        }
        "ema" => {
            let input = indicators::ema::EmaInput::from_slice(&data.close, indicators::ema::EmaParams { period: Some(14) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(indicators::ema::ema(input));
                });
            });
        }
        "rsi" => {
            let input = indicators::rsi::RsiInput::from_slice(&data.close, indicators::rsi::RsiParams { period: Some(14) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(indicators::rsi::rsi(input));
                });
            });
        }
        "bollinger_bands" => {
            let input = indicators::bollinger_bands::BollingerBandsInput::from_slice(
                &data.close, 
                indicators::bollinger_bands::BollingerBandsParams { 
                    period: Some(20),
                    multiplier_upper: Some(2.0),
                    multiplier_lower: Some(2.0),
                    ma_type: Some("sma".to_string()),
                    ddof: Some(0),
                }
            );
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(indicators::bollinger_bands::bollinger_bands(input));
                });
            });
        }
        "atr" => {
            let input = indicators::atr::AtrInput::from_slices(
                &data.high,
                &data.low, 
                &data.close,
                indicators::atr::AtrParams { period: Some(14) }
            );
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(indicators::atr::atr(input));
                });
            });
        }
        _ => {}
    }

    // Benchmark Rust implementation (through FFI)
    let mut rust_output = vec![0.0; data.len()];
    match indicator.rust_name {
        "sma" => {
            group.bench_function("rust_ffi", |b| {
                b.iter(|| {
                    unsafe {
                        rust_ffi::rust_sma(
                            data.len() as i32,
                            data.close.as_ptr(),
                            14,
                            rust_output.as_mut_ptr(),
                        );
                    }
                });
            });
        }
        "ema" => {
            group.bench_function("rust_ffi", |b| {
                b.iter(|| {
                    unsafe {
                        rust_ffi::rust_ema(
                            data.len() as i32,
                            data.close.as_ptr(),
                            14,
                            rust_output.as_mut_ptr(),
                        );
                    }
                });
            });
        }
        "rsi" => {
            group.bench_function("rust_ffi", |b| {
                b.iter(|| {
                    unsafe {
                        rust_ffi::rust_rsi(
                            data.len() as i32,
                            data.close.as_ptr(),
                            14,
                            rust_output.as_mut_ptr(),
                        );
                    }
                });
            });
        }
        "atr" => {
            group.bench_function("rust_ffi", |b| {
                b.iter(|| {
                    unsafe {
                        rust_ffi::rust_atr(
                            data.len() as i32,
                            data.high.as_ptr(),
                            data.low.as_ptr(),
                            data.close.as_ptr(),
                            14,
                            rust_output.as_mut_ptr(),
                        );
                    }
                });
            });
        }
        _ => {}
    }

    // Benchmark Tulip implementation
    let mut output_buffers: Vec<Vec<f64>> = match indicator.tulip_name {
        "bbands" => vec![vec![0.0; data.len()]; 3], // 3 outputs
        "macd" => vec![vec![0.0; data.len()]; 3],   // 3 outputs  
        "stoch" => vec![vec![0.0; data.len()]; 2],  // 2 outputs
        "aroon" => vec![vec![0.0; data.len()]; 2],  // 2 outputs
        _ => vec![vec![0.0; data.len()]; 1],        // 1 output
    };

    let inputs: Vec<&[f64]> = indicator.inputs.iter().map(|&input_name| {
        match input_name {
            "open" => &data.open[..],
            "high" => &data.high[..],
            "low" => &data.low[..],
            "close" => &data.close[..],
            "volume" => &data.volume[..],
            _ => panic!("Unknown input: {}", input_name),
        }
    }).collect();

    group.bench_function("tulip", |b| {
        b.iter(|| {
            let mut output_refs: Vec<&mut [f64]> = output_buffers
                .iter_mut()
                .map(|v| &mut v[..])
                .collect();
            
            unsafe {
                let _ = black_box(tulip::call_indicator(
                    indicator.tulip_name,
                    data.len(),
                    &inputs,
                    &indicator.options,
                    &mut output_refs,
                ));
            }
        });
    });

    group.finish();
}

fn setup_benchmarks(c: &mut Criterion) {
    let mappings = get_indicator_mappings();

    for (size_name, csv_path) in DATA_SIZES {
        let path = Path::new(csv_path);
        if !path.exists() {
            eprintln!("Warning: Dataset {} not found at {}", size_name, csv_path);
            continue;
        }

        let data = CandleData::from_csv(path)
            .expect(&format!("Failed to load {}", csv_path));

        // Only benchmark first few indicators for now
        for mapping in mappings.iter().take(5) {
            benchmark_rust_indicator(c, mapping, &data, size_name);
        }
    }
}

criterion_group!(benches, setup_benchmarks);
criterion_main!(benches);