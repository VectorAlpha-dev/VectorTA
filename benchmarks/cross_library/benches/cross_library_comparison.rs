use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use cross_library_benchmark::{tulip, rust_ffi};
use cross_library_benchmark::utils::CandleData;
use my_project::indicators::moving_averages::{sma, ema};
use my_project::indicators::{rsi, atr, bollinger_bands, macd, adx, cci, stoch, aroon};
use my_project::utilities::data_loader::Candles;
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
        // === Core Indicators (10) - Already implemented ===
        IndicatorMapping {
            rust_name: "sma",
            tulip_name: "sma",
            talib_name: Some("SMA"),
            inputs: vec!["close"],
            options: vec![14.0],
        },
        IndicatorMapping {
            rust_name: "ema",
            tulip_name: "ema",
            talib_name: Some("EMA"),
            inputs: vec!["close"],
            options: vec![14.0],
        },
        IndicatorMapping {
            rust_name: "rsi",
            tulip_name: "rsi",
            talib_name: Some("RSI"),
            inputs: vec!["close"],
            options: vec![14.0],
        },
        IndicatorMapping {
            rust_name: "bollinger_bands",
            tulip_name: "bbands",
            talib_name: Some("BBANDS"),
            inputs: vec!["close"],
            options: vec![20.0, 2.0],
        },
        IndicatorMapping {
            rust_name: "macd",
            tulip_name: "macd",
            talib_name: Some("MACD"),
            inputs: vec!["close"],
            options: vec![12.0, 26.0, 9.0],
        },
        IndicatorMapping {
            rust_name: "atr",
            tulip_name: "atr",
            talib_name: Some("ATR"),
            inputs: vec!["high", "low", "close"],
            options: vec![14.0],
        },
        IndicatorMapping {
            rust_name: "stoch",
            tulip_name: "stoch",
            talib_name: Some("STOCH"),
            inputs: vec!["high", "low", "close"],
            options: vec![14.0, 3.0, 3.0],
        },
        IndicatorMapping {
            rust_name: "aroon",
            tulip_name: "aroon",
            talib_name: Some("AROON"),
            inputs: vec!["high", "low"],
            options: vec![14.0],
        },
        IndicatorMapping {
            rust_name: "adx",
            tulip_name: "adx",
            talib_name: Some("ADX"),
            inputs: vec!["high", "low"],
            options: vec![14.0],
        },
        IndicatorMapping {
            rust_name: "cci",
            tulip_name: "cci",
            talib_name: Some("CCI"),
            inputs: vec!["high", "low", "close"],
            options: vec![14.0],
        },
        
        // === Additional Moving Averages ===
        IndicatorMapping {
            rust_name: "dema",
            tulip_name: "dema",
            talib_name: Some("DEMA"),
            inputs: vec!["close"],
            options: vec![14.0],
        },
        IndicatorMapping {
            rust_name: "tema",
            tulip_name: "tema",
            talib_name: Some("TEMA"),
            inputs: vec!["close"],
            options: vec![14.0],
        },
        IndicatorMapping {
            rust_name: "wma",
            tulip_name: "wma",
            talib_name: Some("WMA"),
            inputs: vec!["close"],
            options: vec![14.0],
        },
        IndicatorMapping {
            rust_name: "kama",
            tulip_name: "kama",
            talib_name: Some("KAMA"),
            inputs: vec!["close"],
            options: vec![14.0],
        },
        IndicatorMapping {
            rust_name: "trima",
            tulip_name: "trima",
            talib_name: Some("TRIMA"),
            inputs: vec!["close"],
            options: vec![14.0],
        },
        IndicatorMapping {
            rust_name: "hma",
            tulip_name: "hma",
            talib_name: Some("HMA"),
            inputs: vec!["close"],
            options: vec![14.0],
        },
        
        // === Momentum Indicators ===
        IndicatorMapping {
            rust_name: "apo",
            tulip_name: "apo",
            talib_name: Some("APO"),
            inputs: vec!["close"],
            options: vec![12.0, 26.0],
        },
        IndicatorMapping {
            rust_name: "cmo",
            tulip_name: "cmo",
            talib_name: Some("CMO"),
            inputs: vec!["close"],
            options: vec![14.0],
        },
        IndicatorMapping {
            rust_name: "dpo",
            tulip_name: "dpo",
            talib_name: Some("DPO"),
            inputs: vec!["close"],
            options: vec![14.0],
        },
        IndicatorMapping {
            rust_name: "mom",
            tulip_name: "mom",
            talib_name: Some("MOM"),
            inputs: vec!["close"],
            options: vec![14.0],
        },
        IndicatorMapping {
            rust_name: "ppo",
            tulip_name: "ppo",
            talib_name: Some("PPO"),
            inputs: vec!["close"],
            options: vec![12.0, 26.0, 9.0],
        },
        IndicatorMapping {
            rust_name: "roc",
            tulip_name: "roc",
            talib_name: Some("ROC"),
            inputs: vec!["close"],
            options: vec![14.0],
        },
        IndicatorMapping {
            rust_name: "rocr",
            tulip_name: "rocr",
            talib_name: Some("ROCR"),
            inputs: vec!["close"],
            options: vec![14.0],
        },
        IndicatorMapping {
            rust_name: "willr",
            tulip_name: "willr",
            talib_name: Some("WILLR"),
            inputs: vec!["high", "low", "close"],
            options: vec![14.0],
        },
        
        // === Volume Indicators ===
        IndicatorMapping {
            rust_name: "ad",
            tulip_name: "ad",
            talib_name: Some("AD"),
            inputs: vec!["high", "low", "close", "volume"],
            options: vec![],
        },
        IndicatorMapping {
            rust_name: "adosc",
            tulip_name: "adosc",
            talib_name: Some("ADOSC"),
            inputs: vec!["high", "low", "close", "volume"],
            options: vec![3.0, 10.0],
        },
        IndicatorMapping {
            rust_name: "obv",
            tulip_name: "obv",
            talib_name: Some("OBV"),
            inputs: vec!["close", "volume"],
            options: vec![],
        },
        IndicatorMapping {
            rust_name: "mfi",
            tulip_name: "mfi",
            talib_name: Some("MFI"),
            inputs: vec!["high", "low", "close", "volume"],
            options: vec![14.0],
        },
        
        // === Other Common Indicators ===
        IndicatorMapping {
            rust_name: "ao",
            tulip_name: "ao",
            talib_name: Some("AO"),
            inputs: vec!["high", "low"],
            options: vec![],
        },
        IndicatorMapping {
            rust_name: "bop",
            tulip_name: "bop",
            talib_name: Some("BOP"),
            inputs: vec!["open", "high", "low", "close"],
            options: vec![],
        },
        IndicatorMapping {
            rust_name: "natr",
            tulip_name: "natr",
            talib_name: Some("NATR"),
            inputs: vec!["high", "low", "close"],
            options: vec![14.0],
        },
        IndicatorMapping {
            rust_name: "stddev",
            tulip_name: "stddev",
            talib_name: Some("STDDEV"),
            inputs: vec!["close"],
            options: vec![14.0],
        },
        IndicatorMapping {
            rust_name: "var",
            tulip_name: "var",
            talib_name: Some("VAR"),
            inputs: vec!["close"],
            options: vec![14.0],
        },
        IndicatorMapping {
            rust_name: "ultosc",
            tulip_name: "ultosc",
            talib_name: Some("ULTOSC"),
            inputs: vec!["high", "low", "close"],
            options: vec![7.0, 14.0, 28.0],
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
            let input = sma::SmaInput::from_slice(&data.close, sma::SmaParams { period: Some(14) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(sma::sma(input));
                });
            });
        }
        "ema" => {
            let input = ema::EmaInput::from_slice(&data.close, ema::EmaParams { period: Some(14) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(ema::ema(input));
                });
            });
        }
        "rsi" => {
            let input = rsi::RsiInput::from_slice(&data.close, rsi::RsiParams { period: Some(14) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(rsi::rsi(input));
                });
            });
        }
        "bollinger_bands" => {
            let input = bollinger_bands::BollingerBandsInput::from_slice(
                &data.close, 
                bollinger_bands::BollingerBandsParams { 
                    period: Some(20),
                    devup: Some(2.0),
                    devdn: Some(2.0),
                    matype: Some("sma".to_string()),
                    devtype: Some(0),
                }
            );
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(bollinger_bands::bollinger_bands(input));
                });
            });
        }
        "atr" => {
            // Convert CandleData to Candles
            let mut hl2 = vec![0.0; data.len()];
            let mut hlc3 = vec![0.0; data.len()];
            let mut ohlc4 = vec![0.0; data.len()];
            let mut hlcc4 = vec![0.0; data.len()];
            
            for i in 0..data.len() {
                hl2[i] = (data.high[i] + data.low[i]) / 2.0;
                hlc3[i] = (data.high[i] + data.low[i] + data.close[i]) / 3.0;
                ohlc4[i] = (data.open[i] + data.high[i] + data.low[i] + data.close[i]) / 4.0;
                hlcc4[i] = (data.high[i] + data.low[i] + data.close[i] + data.close[i]) / 4.0;
            }
            
            let candles = Candles {
                high: data.high.clone(),
                low: data.low.clone(),
                close: data.close.clone(),
                open: data.open.clone(),
                volume: data.volume.clone(),
                timestamp: data.timestamps.clone(),
                hl2,
                hlc3,
                ohlc4,
                hlcc4,
            };
            let input = atr::AtrInput::from_candles(&candles, atr::AtrParams { length: Some(14) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(atr::atr(input));
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
    #[cfg(not(feature = "talib"))]
    let has_talib = false;
    #[cfg(feature = "talib")]
    let has_talib = std::env::var("TALIB_PATH").is_ok();
    
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

    // Benchmark TA-LIB implementation if available
    #[cfg(feature = "talib")]
    {
        if has_talib && indicator.talib_name.is_some() {
            group.bench_function("talib", |b| {
                b.iter(|| {
                    // TA-LIB benchmark would go here when bindings are available
                    // For now, this is a placeholder
                    // Example:
                    // unsafe {
                    //     let result = TA_SMA(...);
                    // }
                    
                    // Placeholder computation to prevent optimization
                    let _ = black_box(&data.close[0]);
                });
            });
        }
    }
    
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