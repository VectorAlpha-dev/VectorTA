use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use cross_library_benchmark::{tulip, rust_ffi};
use cross_library_benchmark::utils::CandleData;
use cross_library_benchmark::benchmark_collector::COLLECTOR;
use cross_library_benchmark::unified_benchmark::LibraryType;
use my_project::indicators::moving_averages::{
    sma, ema, wma, dema, tema, kama, trima, hma, zlema, vwma, wilders, linreg
};
use my_project::indicators::{
    rsi, atr, bollinger_bands, macd, adx, cci, stoch, aroon,
    ao, cmo, di, mfi, mom, obv, ppo, roc,
    srsi, trix, willr,
    // Additional momentum indicators
    apo, dpo, rocr, rocp,
    // Volume indicators
    ad, adosc, emv,
    // Other indicators
    adxr, aroonosc, bop, dm, dx, fisher, fosc, kvo,
    linearreg_slope, linearreg_intercept, linearreg_angle, mass, medprice, midpoint, midprice,
    natr, nvi, pvi, qstick, stddev, stochf,
    tsf, ultosc, var, vosc, wad, vidya, wclprice,
    // Additional new indicators
    cvi, marketefi, minmax, msw, sar
};
use my_project::utilities::data_loader::Candles;
use std::path::Path;
use std::time::{Duration, Instant};

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
    #[allow(dead_code)]
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
            inputs: vec!["high", "low", "close"],
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
            talib_name: None,
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
            talib_name: None, // TA-LIB does not expose DPO in this build
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
            talib_name: None,
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

        // === Additional indicators for full coverage ===
        // ADXR
        IndicatorMapping {
            rust_name: "adxr",
            tulip_name: "adxr",
            talib_name: Some("ADXR"),
            inputs: vec!["high", "low", "close"],
            options: vec![14.0],
        },
        // DI variants (explicit plus/minus for TA-LIB parity)
        IndicatorMapping {
            rust_name: "plus_di",
            tulip_name: "di",
            talib_name: Some("PLUS_DI"),
            inputs: vec!["high", "low", "close"],
            options: vec![14.0],
        },
        IndicatorMapping {
            rust_name: "minus_di",
            tulip_name: "di",
            talib_name: Some("MINUS_DI"),
            inputs: vec!["high", "low", "close"],
            options: vec![14.0],
        },
        // AROONOSC
        IndicatorMapping {
            rust_name: "aroonosc",
            tulip_name: "aroonosc",
            talib_name: Some("AROONOSC"),
            inputs: vec!["high", "low"],
            options: vec![14.0],
        },
        // DI (Directional Indicator)
        IndicatorMapping {
            rust_name: "di",
            tulip_name: "di",
            talib_name: Some("PLUS_DI"),
            inputs: vec!["high", "low", "close"],
            options: vec![14.0],
        },
        // DM (Directional Movement)
        IndicatorMapping {
            rust_name: "dm",
            tulip_name: "dm",
            talib_name: Some("PLUS_DM"),
            inputs: vec!["high", "low"],
            options: vec![14.0],
        },
        // DM explicit variants for TA-LIB parity
        IndicatorMapping {
            rust_name: "plus_dm",
            tulip_name: "dm",
            talib_name: Some("PLUS_DM"),
            inputs: vec!["high", "low"],
            options: vec![14.0],
        },
        IndicatorMapping {
            rust_name: "minus_dm",
            tulip_name: "dm",
            talib_name: Some("MINUS_DM"),
            inputs: vec!["high", "low"],
            options: vec![14.0],
        },
        // DX (Directional Movement Index)
        IndicatorMapping {
            rust_name: "dx",
            tulip_name: "dx",
            talib_name: Some("DX"),
            inputs: vec!["high", "low", "close"],
            options: vec![14.0],
        },
        // FISHER
        IndicatorMapping {
            rust_name: "fisher",
            tulip_name: "fisher",
            talib_name: None,
            inputs: vec!["high", "low"],
            options: vec![14.0],
        },
        // FOSC (Forecast Oscillator)
        IndicatorMapping {
            rust_name: "fosc",
            tulip_name: "fosc",
            talib_name: None,
            inputs: vec!["close"],
            options: vec![14.0],
        },
        // KVO (Klinger Volume Oscillator)
        IndicatorMapping {
            rust_name: "kvo",
            tulip_name: "kvo",
            talib_name: None,
            inputs: vec!["high", "low", "close", "volume"],
            options: vec![34.0, 55.0],
        },
        // LINREG (Linear Regression)
        IndicatorMapping {
            rust_name: "linreg",
            tulip_name: "linreg",
            talib_name: Some("LINEARREG"),
            inputs: vec!["close"],
            options: vec![14.0],
        },
        // LINEARREG_ANGLE (TA-Lib only; no Tulip equivalent)
        IndicatorMapping {
            rust_name: "linearreg_angle",
            tulip_name: "linearreg_angle", // marker; Tulip does not provide this
            talib_name: Some("LINEARREG_ANGLE"),
            inputs: vec!["close"],
            options: vec![14.0],
        },
        // LINEARREG_SLOPE (added for full coverage)
        IndicatorMapping {
            rust_name: "linearreg_slope",
            tulip_name: "linregslope",
            talib_name: Some("LINEARREG_SLOPE"),
            inputs: vec!["close"],
            options: vec![14.0],
        },
        // MASS (Mass Index)
        IndicatorMapping {
            rust_name: "mass",
            tulip_name: "mass",
            talib_name: None,
            inputs: vec!["high", "low"],
            options: vec![25.0],
        },
        // MEDPRICE (Median Price)
        IndicatorMapping {
            rust_name: "medprice",
            tulip_name: "medprice",
            talib_name: Some("MEDPRICE"),
            inputs: vec!["high", "low"],
            options: vec![],
        },
        // MIDPOINT
        IndicatorMapping {
            rust_name: "midpoint",
            tulip_name: "midpoint",
            talib_name: Some("MIDPOINT"),
            inputs: vec!["close"],
            options: vec![14.0],
        },
        // MIDPRICE
        IndicatorMapping {
            rust_name: "midprice",
            tulip_name: "midprice",
            talib_name: Some("MIDPRICE"),
            inputs: vec!["high", "low"],
            options: vec![14.0],
        },
        // NVI (Negative Volume Index)
        IndicatorMapping {
            rust_name: "nvi",
            tulip_name: "nvi",
            talib_name: None,
            inputs: vec!["close", "volume"],
            options: vec![],
        },
        // PVI (Positive Volume Index)
        IndicatorMapping {
            rust_name: "pvi",
            tulip_name: "pvi",
            talib_name: None,
            inputs: vec!["close", "volume"],
            options: vec![],
        },
        // QSTICK
        IndicatorMapping {
            rust_name: "qstick",
            tulip_name: "qstick",
            talib_name: None,
            inputs: vec!["open", "close"],
            options: vec![14.0],
        },
        // ROCP (Rate of Change Percentage)
        IndicatorMapping {
            rust_name: "rocp",
            tulip_name: "rocp",
            talib_name: Some("ROCP"),
            inputs: vec!["close"],
            options: vec![14.0],
        },
        // SAR (Parabolic SAR)
        IndicatorMapping {
            rust_name: "sar",
            tulip_name: "psar",
            talib_name: Some("SAR"),
            inputs: vec!["high", "low"],
            options: vec![0.02, 0.2],
        },
        // STOCHF (Stochastic Fast)
        IndicatorMapping {
            rust_name: "stochf",
            tulip_name: "stochf",
            talib_name: Some("STOCHF"),
            inputs: vec!["high", "low", "close"],
            options: vec![14.0, 3.0],
        },
        // STOCHRSI (Stochastic RSI)
        IndicatorMapping {
            rust_name: "srsi",
            tulip_name: "stochrsi",
            talib_name: Some("STOCHRSI"),
            inputs: vec!["close"],
            options: vec![14.0, 14.0, 3.0, 3.0],
        },
        // TRIX
        IndicatorMapping {
            rust_name: "trix",
            tulip_name: "trix",
            talib_name: Some("TRIX"),
            inputs: vec!["close"],
            options: vec![14.0],
        },
        // TSF (Time Series Forecast)
        IndicatorMapping {
            rust_name: "tsf",
            tulip_name: "tsf",
            talib_name: Some("TSF"),
            inputs: vec!["close"],
            options: vec![14.0],
        },
        // VIDYA (Variable Index Dynamic Average)
        IndicatorMapping {
            rust_name: "vidya",
            tulip_name: "vidya",
            talib_name: None,
            inputs: vec!["close"],
            options: vec![14.0, 0.2],
        },
        // VOSC (Volume Oscillator)
        IndicatorMapping {
            rust_name: "vosc",
            tulip_name: "vosc",
            talib_name: None,
            inputs: vec!["volume"],
            options: vec![5.0, 10.0],
        },
        // VWMA (Volume Weighted Moving Average)
        IndicatorMapping {
            rust_name: "vwma",
            tulip_name: "vwma",
            talib_name: None,
            inputs: vec!["close", "volume"],
            options: vec![14.0],
        },
        // WAD (Williams Accumulation/Distribution)
        IndicatorMapping {
            rust_name: "wad",
            tulip_name: "wad",
            talib_name: None,
            inputs: vec!["high", "low", "close"],
            options: vec![],
        },
        // WCLPRICE (Weighted Close Price)
        IndicatorMapping {
            rust_name: "wclprice",
            tulip_name: "wcprice",
            talib_name: Some("WCLPRICE"),
            inputs: vec!["high", "low", "close"],
            options: vec![],
        },
        // WILDERS (Wilders Smoothing)
        IndicatorMapping {
            rust_name: "wilders",
            tulip_name: "wilders",
            talib_name: None,
            inputs: vec!["close"],
            options: vec![14.0],
        },
        // ZLEMA (Zero Lag Exponential Moving Average)
        IndicatorMapping {
            rust_name: "zlema",
            tulip_name: "zlema",
            talib_name: None,
            inputs: vec!["close"],
            options: vec![14.0],
        },
        // Additional indicators with Tulip equivalents
        IndicatorMapping {
            rust_name: "cvi",
            tulip_name: "cvi",
            talib_name: None,
            inputs: vec!["high", "low"],
            options: vec![14.0],
        },
        IndicatorMapping {
            rust_name: "emv",
            tulip_name: "emv",
            talib_name: None,
            inputs: vec!["high", "low", "volume"],
            options: vec![],
        },
        IndicatorMapping {
            rust_name: "marketefi",
            tulip_name: "marketfi",
            talib_name: None,
            inputs: vec!["high", "low", "volume"],
            options: vec![],
        },
        IndicatorMapping {
            rust_name: "linearreg_intercept",
            tulip_name: "linregintercept",
            talib_name: Some("LINEARREG_INTERCEPT"),
            inputs: vec!["close"],
            options: vec![14.0],
        },
        IndicatorMapping {
            rust_name: "minmax",
            tulip_name: "max",
            talib_name: Some("MAX"),
            inputs: vec!["close"],
            options: vec![14.0],
        },
        IndicatorMapping {
            rust_name: "minmax_min",
            tulip_name: "min",
            talib_name: Some("MIN"),
            inputs: vec!["close"],
            options: vec![14.0],
        },
        IndicatorMapping {
            rust_name: "avgprice",
            tulip_name: "avgprice",
            talib_name: Some("AVGPRICE"),
            inputs: vec!["open", "high", "low", "close"],
            options: vec![],
        },
        IndicatorMapping {
            rust_name: "typprice",
            tulip_name: "typprice",
            talib_name: Some("TYPPRICE"),
            inputs: vec!["high", "low", "close"],
            options: vec![],
        },
        IndicatorMapping {
            rust_name: "tr",
            tulip_name: "tr",
            talib_name: Some("TRANGE"),
            inputs: vec!["high", "low", "close"],
            options: vec![],
        },
        IndicatorMapping {
            rust_name: "msw",
            tulip_name: "msw",
            talib_name: None,
            inputs: vec!["close"],
            options: vec![14.0],
        },
    ]
}

// Helper function to create Candles from CandleData
fn create_candles(data: &CandleData) -> Candles {
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

    Candles {
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
    }
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
            let candles = create_candles(data);
            let input = atr::AtrInput::from_candles(&candles, atr::AtrParams { length: Some(14) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(atr::atr(input));
                });
            });
        }
        "adx" => {
            let candles = create_candles(data);
            let input = adx::AdxInput::from_candles(&candles, adx::AdxParams { period: Some(14) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(adx::adx(input));
                });
            });
        }
        "ao" => {
            let candles = create_candles(data);
            let input = ao::AoInput::from_candles(&candles, "hl2", ao::AoParams { short_period: Some(5), long_period: Some(34) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(ao::ao(input));
                });
            });
        }
        "aroon" => {
            let candles = create_candles(data);
            let input = aroon::AroonInput::from_candles(&candles, aroon::AroonParams { length: Some(14) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(aroon::aroon(input));
                });
            });
        }
        "cci" => {
            let candles = create_candles(data);
            let input = cci::CciInput::from_candles(&candles, "hlc3", cci::CciParams { period: Some(20) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(cci::cci(input));
                });
            });
        }
        "cmo" => {
            let input = cmo::CmoInput::from_slice(&data.close, cmo::CmoParams { period: Some(14) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(cmo::cmo(input));
                });
            });
        }
        "plus_di" | "minus_di" => {
            let candles = create_candles(data);
            let input = di::DiInput::from_candles(&candles, di::DiParams { period: Some(14) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(di::di(input));
                });
            });
        }
        "linreg" => {
            group.bench_function("rust_ffi", |b| {
                b.iter(|| {
                    unsafe {
                        rust_ffi::rust_linreg(
                            data.len() as i32,
                            data.close.as_ptr(),
                            14,
                            rust_output.as_mut_ptr(),
                        );
                    }
                });
            });
        }
        "dmi" => {
            let candles = create_candles(data);
            let input = di::DiInput::from_candles(&candles, di::DiParams { period: Some(14) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(di::di(input));
                });
            });
        }
        "macd" => {
            let input = macd::MacdInput::from_slice(&data.close, macd::MacdParams {
                fast_period: Some(12),
                slow_period: Some(26),
                signal_period: Some(9),
                ma_type: None
            });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(macd::macd(input));
                });
            });
        }
        "mfi" => {
            let candles = create_candles(data);
            let input = mfi::MfiInput::from_candles(&candles, "hlc3", mfi::MfiParams { period: Some(14) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(mfi::mfi(input));
                });
            });
        }
        "obv" => {
            let candles = create_candles(data);
            let input = obv::ObvInput::from_candles(&candles, obv::ObvParams {});
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(obv::obv(input));
                });
            });
        }
        "ppo" => {
            let input = ppo::PpoInput::from_slice(&data.close, ppo::PpoParams {
                fast_period: Some(12),
                slow_period: Some(26),
                ma_type: None
            });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(ppo::ppo(input));
                });
            });
        }
        "roc" => {
            let input = roc::RocInput::from_slice(&data.close, roc::RocParams { period: Some(10) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(roc::roc(input));
                });
            });
        }
        "stoch" => {
            let candles = create_candles(data);
            let input = stoch::StochInput::from_candles(&candles, stoch::StochParams {
                fastk_period: Some(14),
                slowk_period: Some(5),
                slowk_ma_type: Some("sma".to_string()),
                slowd_period: Some(3),
                slowd_ma_type: Some("sma".to_string())
            });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(stoch::stoch(input));
                });
            });
        }
        "stochrsi" => {
            let input = srsi::SrsiInput::from_slice(&data.close, srsi::SrsiParams {
                rsi_period: Some(14),
                stoch_period: Some(14),
                k: Some(3),
                d: Some(3),
                source: None
            });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(srsi::srsi(input));
                });
            });
        }
        "trix" => {
            let input = trix::TrixInput::from_slice(&data.close, trix::TrixParams { period: Some(14) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(trix::trix(input));
                });
            });
        }
        "willr" => {
            let candles = create_candles(data);
            let input = willr::WillrInput::from_candles(&candles, willr::WillrParams { period: Some(14) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(willr::willr(input));
                });
            });
        }
        "wma" => {
            let input = wma::WmaInput::from_slice(&data.close, wma::WmaParams { period: Some(10) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(wma::wma(input));
                });
            });
        }
        // Additional moving averages
        "dema" => {
            let input = dema::DemaInput::from_slice(&data.close, dema::DemaParams { period: Some(14) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(dema::dema(input));
                });
            });
        }
        "tema" => {
            let input = tema::TemaInput::from_slice(&data.close, tema::TemaParams { period: Some(14) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(tema::tema(input));
                });
            });
        }
        "kama" => {
            let input = kama::KamaInput::from_slice(&data.close, kama::KamaParams { period: Some(14) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(kama::kama(input));
                });
            });
        }
        "trima" => {
            let input = trima::TrimaInput::from_slice(&data.close, trima::TrimaParams { period: Some(14) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(trima::trima(input));
                });
            });
        }
        "hma" => {
            let input = hma::HmaInput::from_slice(&data.close, hma::HmaParams { period: Some(14) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(hma::hma(input));
                });
            });
        }
        "zlema" => {
            let input = zlema::ZlemaInput::from_slice(&data.close, zlema::ZlemaParams { period: Some(14) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(zlema::zlema(input));
                });
            });
        }
        "vwma" => {
            let candles = create_candles(data);
            let input = vwma::VwmaInput::from_candles(&candles, "close", vwma::VwmaParams { period: Some(14) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(vwma::vwma(input));
                });
            });
        }
        "vidya" => {
            let input = vidya::VidyaInput::from_slice(&data.close, vidya::VidyaParams { short_period: Some(2), long_period: Some(5), alpha: Some(0.2) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(vidya::vidya(input));
                });
            });
        }
        "wilders" => {
            let input = wilders::WildersInput::from_slice(&data.close, wilders::WildersParams { period: Some(14) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(wilders::wilders(input));
                });
            });
        }
        // Additional momentum indicators
        "apo" => {
            let input = apo::ApoInput::from_slice(&data.close, apo::ApoParams { short_period: Some(12), long_period: Some(26) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(apo::apo(input));
                });
            });
        }
        "dpo" => {
            let input = dpo::DpoInput::from_slice(&data.close, dpo::DpoParams { period: Some(14) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(dpo::dpo(input));
                });
            });
        }
        "mom" => {
            let input = mom::MomInput::from_slice(&data.close, mom::MomParams { period: Some(14) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(mom::mom(input));
                });
            });
        }
        "rocr" => {
            let input = rocr::RocrInput::from_slice(&data.close, rocr::RocrParams { period: Some(14) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(rocr::rocr(input));
                });
            });
        }
        "rocp" => {
            let input = rocp::RocpInput::from_slice(&data.close, rocp::RocpParams { period: Some(14) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(rocp::rocp(input));
                });
            });
        }
        // Volume indicators
        "ad" => {
            let candles = create_candles(data);
            let input = ad::AdInput::from_candles(&candles, ad::AdParams {});
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(ad::ad(input));
                });
            });
        }
        "adosc" => {
            let candles = create_candles(data);
            let input = adosc::AdoscInput::from_candles(&candles, adosc::AdoscParams { short_period: Some(3), long_period: Some(10) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(adosc::adosc(input));
                });
            });
        }
        "nvi" => {
            let candles = create_candles(data);
            let input = nvi::NviInput::from_candles(&candles, "close", nvi::NviParams {});
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(nvi::nvi(input));
                });
            });
        }
        "pvi" => {
            let candles = create_candles(data);
            let input = pvi::PviInput::from_candles(&candles, "close", "volume", pvi::PviParams { initial_value: Some(1000.0) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(pvi::pvi(input));
                });
            });
        }
        "vosc" => {
            let candles = create_candles(data);
            let input = vosc::VoscInput::from_candles(&candles, "volume", vosc::VoscParams { short_period: Some(5), long_period: Some(10) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(vosc::vosc(input));
                });
            });
        }
        // Other indicators
        "adxr" => {
            let candles = create_candles(data);
            let input = adxr::AdxrInput::from_candles(&candles, adxr::AdxrParams { period: Some(14) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(adxr::adxr(input));
                });
            });
        }
        "aroonosc" => {
            let candles = create_candles(data);
            let input = aroonosc::AroonOscInput::from_candles(&candles, aroonosc::AroonOscParams { length: Some(14) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(aroonosc::aroon_osc(input));
                });
            });
        }
        "bop" => {
            let candles = create_candles(data);
            let input = bop::BopInput::from_candles(&candles, bop::BopParams {});
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(bop::bop(input));
                });
            });
        }
        "di" => {
            let candles = create_candles(data);
            let input = di::DiInput::from_candles(&candles, di::DiParams { period: Some(14) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(di::di(input));
                });
            });
        }
        "dm" => {
            let candles = create_candles(data);
            let input = dm::DmInput::from_candles(&candles, dm::DmParams { period: Some(14) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(dm::dm(input));
                });
            });
        }
        "plus_dm" | "minus_dm" => {
            let candles = create_candles(data);
            let input = dm::DmInput::from_candles(&candles, dm::DmParams { period: Some(14) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(dm::dm(input));
                });
            });
        }
        "dx" => {
            let candles = create_candles(data);
            let input = dx::DxInput::from_candles(&candles, dx::DxParams { period: Some(14) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(dx::dx(input));
                });
            });
        }
        "fisher" => {
            let candles = create_candles(data);
            let input = fisher::FisherInput::from_candles(&candles, fisher::FisherParams { period: Some(14) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(fisher::fisher(input));
                });
            });
        }
        "fosc" => {
            let input = fosc::FoscInput::from_slice(&data.close, fosc::FoscParams { period: Some(14) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(fosc::fosc(input));
                });
            });
        }
        "kvo" => {
            let candles = create_candles(data);
            let input = kvo::KvoInput::from_candles(&candles, kvo::KvoParams { short_period: Some(34), long_period: Some(55) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(kvo::kvo(input));
                });
            });
        }
        "linreg" => {
            let input = linreg::LinRegInput::from_slice(&data.close, linreg::LinRegParams { period: Some(14) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(linreg::linreg(input));
                });
            });
        }
        "linearreg_angle" => {
            let input = linearreg_angle::Linearreg_angleInput::from_slice(&data.close, linearreg_angle::Linearreg_angleParams { period: Some(14) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(linearreg_angle::linearreg_angle(input));
                });
            });
        }
        "mass" => {
            let candles = create_candles(data);
            let input = mass::MassInput::from_candles(&candles, "high", "low", mass::MassParams { period: Some(25) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(mass::mass(input));
                });
            });
        }
        "medprice" => {
            let candles = create_candles(data);
            let input = medprice::MedpriceInput::from_candles(&candles, "high", "low", medprice::MedpriceParams {});
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(medprice::medprice(input));
                });
            });
        }
        "midpoint" => {
            let input = midpoint::MidpointInput::from_slice(&data.close, midpoint::MidpointParams { period: Some(14) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(midpoint::midpoint(input));
                });
            });
        }
        "midprice" => {
            let candles = create_candles(data);
            let input = midprice::MidpriceInput::from_candles(&candles, "high", "low", midprice::MidpriceParams { period: Some(14) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(midprice::midprice(input));
                });
            });
        }
        "natr" => {
            let candles = create_candles(data);
            let input = natr::NatrInput::from_candles(&candles, natr::NatrParams { period: Some(14) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(natr::natr(input));
                });
            });
        }
        "qstick" => {
            let candles = create_candles(data);
            let input = qstick::QstickInput::from_candles(&candles, "open", "close", qstick::QstickParams { period: Some(14) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(qstick::qstick(input));
                });
            });
        }
        "sar" | "psar" => {
            let candles = create_candles(data);
            let input = sar::SarInput::from_candles(&candles, sar::SarParams {
                acceleration: Some(0.02),
                maximum: Some(0.2)
            });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(sar::sar(input));
                });
            });
        }
        "stddev" => {
            let input = stddev::StdDevInput::from_slice(&data.close, stddev::StdDevParams { period: Some(14), nbdev: Some(1.0) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(stddev::stddev(input));
                });
            });
        }
        "stochf" => {
            let candles = create_candles(data);
            let input = stochf::StochfInput::from_candles(&candles, stochf::StochfParams { fastk_period: Some(14), fastd_period: Some(3), fastd_matype: Some(0) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(stochf::stochf(input));
                });
            });
        }
        "tsf" => {
            let input = tsf::TsfInput::from_slice(&data.close, tsf::TsfParams { period: Some(14) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(tsf::tsf(input));
                });
            });
        }
        "ultosc" => {
            let candles = create_candles(data);
            let input = ultosc::UltOscInput::from_candles(&candles, "high", "low", "close", ultosc::UltOscParams { timeperiod1: Some(7), timeperiod2: Some(14), timeperiod3: Some(28) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(ultosc::ultosc(input));
                });
            });
        }
        "var" => {
            let input = var::VarInput::from_slice(&data.close, var::VarParams { period: Some(14), nbdev: Some(1.0) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(var::var(input));
                });
            });
        }
        "wad" => {
            let candles = create_candles(data);
            let input = wad::WadInput::from_candles(&candles);
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(wad::wad(input));
                });
            });
        }
        "wclprice" | "wcprice" => {
            let candles = create_candles(data);
            let input = wclprice::WclpriceInput::from_candles(&candles);
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(wclprice::wclprice(input));
                });
            });
        }
        // New indicator implementations
        "cvi" => {
            let candles = create_candles(data);
            let input = cvi::CviInput::from_candles(&candles, cvi::CviParams { period: Some(14) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(cvi::cvi(input));
                });
            });
        }
        "emv" => {
            let candles = create_candles(data);
            let input = emv::EmvInput::from_candles(&candles);
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(emv::emv(input));
                });
            });
        }
        "marketefi" => {
            let candles = create_candles(data);
            let input = marketefi::MarketefiInput::from_candles(&candles, "high", "low", "volume", marketefi::MarketefiParams {});
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(marketefi::marketefi(input));
                });
            });
        }
        "linearreg_intercept" => {
            let input = linearreg_intercept::LinearRegInterceptInput::from_slice(&data.close, linearreg_intercept::LinearRegInterceptParams { period: Some(14) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(linearreg_intercept::linearreg_intercept(input));
                });
            });
        }
        "minmax" | "minmax_min" => {
            let candles = create_candles(data);
            let input = minmax::MinmaxInput::from_candles(&candles, "high", "low", minmax::MinmaxParams { order: Some(14) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(minmax::minmax(input));
                });
            });
        }
        "avgprice" => {
            // Average price (OHLC/4) - using utility function
            // For avgprice, we can calculate it directly
            let avg_prices: Vec<f64> = (0..data.len())
                .map(|i| (data.open[i] + data.high[i] + data.low[i] + data.close[i]) / 4.0)
                .collect();
            group.bench_with_input(BenchmarkId::new("rust", size_name), &avg_prices, |b, prices| {
                b.iter(|| {
                    let _ = black_box(prices.clone());
                });
            });
        }
        "typprice" => {
            // Typical price (HLC/3)
            let typ_prices: Vec<f64> = (0..data.len())
                .map(|i| (data.high[i] + data.low[i] + data.close[i]) / 3.0)
                .collect();
            group.bench_with_input(BenchmarkId::new("rust", size_name), &typ_prices, |b, prices| {
                b.iter(|| {
                    let _ = black_box(prices.clone());
                });
            });
        }
        "tr" => {
            // True Range - part of ATR calculation
            // Calculate true range directly
            let mut tr_values = vec![0.0; data.len()];
            for i in 1..data.len() {
                let hl = data.high[i] - data.low[i];
                let hc = (data.high[i] - data.close[i - 1]).abs();
                let lc = (data.low[i] - data.close[i - 1]).abs();
                tr_values[i] = hl.max(hc).max(lc);
            }
            group.bench_with_input(BenchmarkId::new("rust", size_name), &tr_values, |b, values| {
                b.iter(|| {
                    let _ = black_box(values.clone());
                });
            });
        }
        "msw" => {
            let input = msw::MswInput::from_slice(&data.close, msw::MswParams { period: Some(14) });
            group.bench_with_input(BenchmarkId::new("rust", size_name), &input, |b, input| {
                b.iter(|| {
                    let _ = black_box(msw::msw(input));
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
        "adx" => {
            group.bench_function("rust_ffi", |b| {
                b.iter(|| {
                    unsafe {
                        rust_ffi::rust_adx(
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
        "ao" => {
            group.bench_function("rust_ffi", |b| {
                b.iter(|| {
                    unsafe {
                        rust_ffi::rust_ao(
                            data.len() as i32,
                            data.high.as_ptr(),
                            data.low.as_ptr(),
                            rust_output.as_mut_ptr(),
                        );
                    }
                });
            });
        }
        "aroon" => {
            let mut aroon_up = vec![0.0; data.len()];
            let mut aroon_down = vec![0.0; data.len()];
            group.bench_function("rust_ffi", |b| {
                b.iter(|| {
                    unsafe {
                        rust_ffi::rust_aroon(
                            data.len() as i32,
                            data.high.as_ptr(),
                            data.low.as_ptr(),
                            14,
                            aroon_up.as_mut_ptr(),
                            aroon_down.as_mut_ptr(),
                        );
                    }
                });
            });
        }
        "cci" => {
            group.bench_function("rust_ffi", |b| {
                b.iter(|| {
                    unsafe {
                        rust_ffi::rust_cci(
                            data.len() as i32,
                            data.high.as_ptr(),
                            data.low.as_ptr(),
                            data.close.as_ptr(),
                            20,
                            rust_output.as_mut_ptr(),
                        );
                    }
                });
            });
        }
        "cmo" => {
            group.bench_function("rust_ffi", |b| {
                b.iter(|| {
                    unsafe {
                        rust_ffi::rust_cmo(
                            data.len() as i32,
                            data.close.as_ptr(),
                            14,
                            rust_output.as_mut_ptr(),
                        );
                    }
                });
            });
        }

        "dmi" => {
            let mut plus_di = vec![0.0; data.len()];
            let mut minus_di = vec![0.0; data.len()];
            group.bench_function("rust_ffi", |b| {
                b.iter(|| {
                    unsafe {
                        rust_ffi::rust_di(
                            data.len() as i32,
                            data.high.as_ptr(),
                            data.low.as_ptr(),
                            data.close.as_ptr(),
                            14,
                            plus_di.as_mut_ptr(),
                            minus_di.as_mut_ptr(),
                        );
                    }
                });
            });
        }
        "dm" => {
            let mut plus_dm = vec![0.0; data.len()];
            let mut minus_dm = vec![0.0; data.len()];
            group.bench_function("rust_ffi", |b| {
                b.iter(|| {
                    unsafe {
                        rust_ffi::rust_dm(
                            data.len() as i32,
                            data.high.as_ptr(),
                            data.low.as_ptr(),
                            14,
                            plus_dm.as_mut_ptr(),
                            minus_dm.as_mut_ptr(),
                        );
                    }
                });
            });
        }
        "plus_dm" | "minus_dm" => {
            let mut plus_dm = vec![0.0; data.len()];
            let mut minus_dm = vec![0.0; data.len()];
            group.bench_function("rust_ffi", |b| {
                b.iter(|| {
                    unsafe {
                        rust_ffi::rust_dm(
                            data.len() as i32,
                            data.high.as_ptr(),
                            data.low.as_ptr(),
                            14,
                            plus_dm.as_mut_ptr(),
                            minus_dm.as_mut_ptr(),
                        );
                    }
                });
            });
        }
        "plus_di" | "minus_di" => {
            let mut plus_di = vec![0.0; data.len()];
            let mut minus_di = vec![0.0; data.len()];
            group.bench_function("rust_ffi", |b| {
                b.iter(|| {
                    unsafe {
                        rust_ffi::rust_di(
                            data.len() as i32,
                            data.high.as_ptr(),
                            data.low.as_ptr(),
                            data.close.as_ptr(),
                            14,
                            plus_di.as_mut_ptr(),
                            minus_di.as_mut_ptr(),
                        );
                    }
                });
            });
        }
        "macd" => {
            let mut macd_line = vec![0.0; data.len()];
            let mut signal = vec![0.0; data.len()];
            let mut histogram = vec![0.0; data.len()];
            group.bench_function("rust_ffi", |b| {
                b.iter(|| {
                    unsafe {
                        rust_ffi::rust_macd(
                            data.len() as i32,
                            data.close.as_ptr(),
                            12,
                            26,
                            9,
                            macd_line.as_mut_ptr(),
                            signal.as_mut_ptr(),
                            histogram.as_mut_ptr(),
                        );
                    }
                });
            });
        }
        "mfi" => {
            group.bench_function("rust_ffi", |b| {
                b.iter(|| {
                    unsafe {
                        rust_ffi::rust_mfi(
                            data.len() as i32,
                            data.high.as_ptr(),
                            data.low.as_ptr(),
                            data.close.as_ptr(),
                            data.volume.as_ptr(),
                            14,
                            rust_output.as_mut_ptr(),
                        );
                    }
                });
            });
        }
        "momentum" => {
            group.bench_function("rust_ffi", |b| {
                b.iter(|| {
                    unsafe {
                        rust_ffi::rust_mom(
                            data.len() as i32,
                            data.close.as_ptr(),
                            10,
                            rust_output.as_mut_ptr(),
                        );
                    }
                });
            });
        }
        "obv" => {
            group.bench_function("rust_ffi", |b| {
                b.iter(|| {
                    unsafe {
                        rust_ffi::rust_obv(
                            data.len() as i32,
                            data.close.as_ptr(),
                            data.volume.as_ptr(),
                            rust_output.as_mut_ptr(),
                        );
                    }
                });
            });
        }
        "ppo" => {
            let mut ppo_output = vec![0.0; data.len()];
            let mut signal_output = vec![0.0; data.len()];
            let mut hist_output = vec![0.0; data.len()];
            group.bench_function("rust_ffi", |b| {
                b.iter(|| {
                    unsafe {
                        rust_ffi::rust_ppo(
                            data.len() as i32,
                            data.close.as_ptr(),
                            12,
                            26,
                            9,
                            ppo_output.as_mut_ptr(),
                            signal_output.as_mut_ptr(),
                            hist_output.as_mut_ptr(),
                        );
                    }
                });
            });
        }
        "roc" => {
            group.bench_function("rust_ffi", |b| {
                b.iter(|| {
                    unsafe {
                        rust_ffi::rust_roc(
                            data.len() as i32,
                            data.close.as_ptr(),
                            10,
                            rust_output.as_mut_ptr(),
                        );
                    }
                });
            });
        }
        "stoch" => {
            let mut fastk = vec![0.0; data.len()];
            let mut fastd = vec![0.0; data.len()];
            group.bench_function("rust_ffi", |b| {
                b.iter(|| {
                    unsafe {
                        rust_ffi::rust_stoch(
                            data.len() as i32,
                            data.high.as_ptr(),
                            data.low.as_ptr(),
                            data.close.as_ptr(),
                            5,
                            3,
                            0,
                            fastk.as_mut_ptr(),
                            fastd.as_mut_ptr(),
                        );
                    }
                });
            });
        }
        "stochrsi" => {
            let mut fastk = vec![0.0; data.len()];
            let mut fastd = vec![0.0; data.len()];
            group.bench_function("rust_ffi", |b| {
                b.iter(|| {
                    unsafe {
                        rust_ffi::rust_srsi(
                            data.len() as i32,
                            data.close.as_ptr(),
                            14,
                            5,
                            3,
                            0,
                            fastk.as_mut_ptr(),
                            fastd.as_mut_ptr(),
                        );
                    }
                });
            });
        }
        "trix" => {
            group.bench_function("rust_ffi", |b| {
                b.iter(|| {
                    unsafe {
                        rust_ffi::rust_trix(
                            data.len() as i32,
                            data.close.as_ptr(),
                            14,
                            rust_output.as_mut_ptr(),
                        );
                    }
                });
            });
        }
        "willr" => {
            group.bench_function("rust_ffi", |b| {
                b.iter(|| {
                    unsafe {
                        rust_ffi::rust_willr(
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
        "wma" => {
            group.bench_function("rust_ffi", |b| {
                b.iter(|| {
                    unsafe {
                        rust_ffi::rust_wma(
                            data.len() as i32,
                            data.close.as_ptr(),
                            10,
                            rust_output.as_mut_ptr(),
                        );
                    }
                });
            });
        }
        "dema" => {
            group.bench_function("rust_ffi", |b| {
                b.iter(|| {
                    unsafe {
                        rust_ffi::rust_dema(
                            data.len() as i32,
                            data.close.as_ptr(),
                            14,
                            rust_output.as_mut_ptr(),
                        );
                    }
                });
            });
        }
        "tema" => {
            group.bench_function("rust_ffi", |b| {
                b.iter(|| {
                    unsafe {
                        rust_ffi::rust_tema(
                            data.len() as i32,
                            data.close.as_ptr(),
                            14,
                            rust_output.as_mut_ptr(),
                        );
                    }
                });
            });
        }
        "kama" => {
            group.bench_function("rust_ffi", |b| {
                b.iter(|| {
                    unsafe {
                        rust_ffi::rust_kama(
                            data.len() as i32,
                            data.close.as_ptr(),
                            14,
                            rust_output.as_mut_ptr(),
                        );
                    }
                });
            });
        }
        "trima" => {
            group.bench_function("rust_ffi", |b| {
                b.iter(|| {
                    unsafe {
                        rust_ffi::rust_trima(
                            data.len() as i32,
                            data.close.as_ptr(),
                            14,
                            rust_output.as_mut_ptr(),
                        );
                    }
                });
            });
        }
        "hma" => {
            group.bench_function("rust_ffi", |b| {
                b.iter(|| {
                    unsafe {
                        rust_ffi::rust_hma(
                            data.len() as i32,
                            data.close.as_ptr(),
                            14,
                            rust_output.as_mut_ptr(),
                        );
                    }
                });
            });
        }
        "zlema" => {
            group.bench_function("rust_ffi", |b| {
                b.iter(|| {
                    unsafe {
                        rust_ffi::rust_zlema(
                            data.len() as i32,
                            data.close.as_ptr(),
                            14,
                            rust_output.as_mut_ptr(),
                        );
                    }
                });
            });
        }
        "vwma" => {
            group.bench_function("rust_ffi", |b| {
                b.iter(|| {
                    unsafe {
                        rust_ffi::rust_vwma(
                            data.len() as i32,
                            data.close.as_ptr(),
                            data.volume.as_ptr(),
                            14,
                            rust_output.as_mut_ptr(),
                        );
                    }
                });
            });
        }
        "wilders" => {
            group.bench_function("rust_ffi", |b| {
                b.iter(|| {
                    unsafe {
                        rust_ffi::rust_wilders(
                            data.len() as i32,
                            data.close.as_ptr(),
                            14,
                            rust_output.as_mut_ptr(),
                        );
                    }
                });
            });
        }
        "vidya" => {
            group.bench_function("rust_ffi", |b| {
                b.iter(|| {
                    unsafe {
                        rust_ffi::rust_vidya(
                            data.len() as i32,
                            data.close.as_ptr(),
                            2,
                            5,
                            0.2,
                            rust_output.as_mut_ptr(),
                        );
                    }
                });
            });
        }
        _ => {}
    }

    // Benchmark Tulip implementation (skip if no Tulip equivalent)
    if indicator.rust_name != "linearreg_angle" {
    #[cfg(not(feature = "talib"))]
    let _has_talib = false;
    #[cfg(feature = "talib")]
    let _has_talib = std::env::var("TALIB_PATH").is_ok();

    let mut output_buffers: Vec<Vec<f64>> = match indicator.tulip_name {
        "bbands" => vec![vec![0.0; data.len()]; 3], // 3 outputs
        "macd" => vec![vec![0.0; data.len()]; 3],   // 3 outputs
        "stoch" | "stochf" | "stochrsi" => vec![vec![0.0; data.len()]; 2],  // 2 outputs
        "aroon" | "di" | "dm" => vec![vec![0.0; data.len()]; 2],  // 2 outputs
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
    }

    // Benchmark TA-LIB implementation if available
    #[cfg(feature = "talib")]
    {
        if _has_talib && indicator.talib_name.is_some() {
            use cross_library_benchmark::talib_wrapper;

            let mut talib_outputs: Vec<Vec<f64>> = match indicator.tulip_name {
                "bbands" => vec![vec![0.0; data.len()]; 3], // 3 outputs
                "macd" => vec![vec![0.0; data.len()]; 3],   // 3 outputs
                "stoch" => vec![vec![0.0; data.len()]; 2],  // 2 outputs
                "aroon" => vec![vec![0.0; data.len()]; 2],  // 2 outputs
                _ => vec![vec![0.0; data.len()]; 1],        // 1 output
            };

            group.bench_function("talib", |b| {
                b.iter(|| {
                    unsafe {
                        match indicator.talib_name.unwrap() {
                            "SMA" => {
                                talib_wrapper::talib_sma(
                                    &data.close,
                                    indicator.options[0] as i32,
                                    &mut talib_outputs[0],
                                ).ok();
                            }
                            "EMA" => {
                                talib_wrapper::talib_ema(
                                    &data.close,
                                    indicator.options[0] as i32,
                                    &mut talib_outputs[0],
                                ).ok();
                            }
                            "RSI" => {
                                talib_wrapper::talib_rsi(
                                    &data.close,
                                    indicator.options[0] as i32,
                                    &mut talib_outputs[0],
                                ).ok();
                            }
                            "BBANDS" => {
                                let (upper, rest) = talib_outputs.split_at_mut(1);
                                let (middle, lower) = rest.split_at_mut(1);
                                talib_wrapper::talib_bbands(
                                    &data.close,
                                    indicator.options[0] as i32,
                                    indicator.options[1],
                                    indicator.options[1],
                                    &mut upper[0],
                                    &mut middle[0],
                                    &mut lower[0],
                                ).ok();
                            }
                            "MACD" => {
                                let (macd, rest) = talib_outputs.split_at_mut(1);
                                let (signal, hist) = rest.split_at_mut(1);
                                talib_wrapper::talib_macd(
                                    &data.close,
                                    indicator.options[0] as i32,
                                    indicator.options[1] as i32,
                                    indicator.options[2] as i32,
                                    &mut macd[0],
                                    &mut signal[0],
                                    &mut hist[0],
                                ).ok();
                            }
                            "ATR" => {
                                talib_wrapper::talib_atr(
                                    &data.high,
                                    &data.low,
                                    &data.close,
                                    indicator.options[0] as i32,
                                    &mut talib_outputs[0],
                                ).ok();
                            }
                            "STOCH" => {
                                let (k, d) = talib_outputs.split_at_mut(1);
                                talib_wrapper::talib_stoch(
                                    &data.high,
                                    &data.low,
                                    &data.close,
                                    indicator.options[0] as i32,
                                    indicator.options[1] as i32,
                                    indicator.options[2] as i32,
                                    &mut k[0],
                                    &mut d[0],
                                ).ok();
                            }
                            "AROON" => {
                                let (down, up) = talib_outputs.split_at_mut(1);
                                talib_wrapper::talib_aroon(
                                    &data.high,
                                    &data.low,
                                    indicator.options[0] as i32,
                                    &mut down[0],
                                    &mut up[0],
                                ).ok();
                            }
                            "ADX" => {
                                talib_wrapper::talib_adx(
                                    &data.high,
                                    &data.low,
                                    &data.close,
                                    indicator.options[0] as i32,
                                    &mut talib_outputs[0],
                                ).ok();
                            }
                            "CCI" => {
                                talib_wrapper::talib_cci(
                                    &data.high,
                                    &data.low,
                                    &data.close,
                                    indicator.options[0] as i32,
                                    &mut talib_outputs[0],
                                ).ok();
                            }
                            "DEMA" => {
                                talib_wrapper::talib_dema(
                                    &data.close,
                                    indicator.options[0] as i32,
                                    &mut talib_outputs[0],
                                ).ok();
                            }
                            "TEMA" => {
                                talib_wrapper::talib_tema(
                                    &data.close,
                                    indicator.options[0] as i32,
                                    &mut talib_outputs[0],
                                ).ok();
                            }
                            "WMA" => {
                                talib_wrapper::talib_wma(
                                    &data.close,
                                    indicator.options[0] as i32,
                                    &mut talib_outputs[0],
                                ).ok();
                            }
                            "KAMA" => {
                                talib_wrapper::talib_kama(
                                    &data.close,
                                    indicator.options[0] as i32,
                                    &mut talib_outputs[0],
                                ).ok();
                            }
                            "TRIMA" => {
                                talib_wrapper::talib_trima(
                                    &data.close,
                                    indicator.options[0] as i32,
                                    &mut talib_outputs[0],
                                ).ok();
                            }
                            "APO" => {
                                talib_wrapper::talib_apo(
                                    &data.close,
                                    indicator.options[0] as i32,
                                    indicator.options[1] as i32,
                                    0, // MA type
                                    &mut talib_outputs[0],
                                ).ok();
                            }
                            "CMO" => {
                                talib_wrapper::talib_cmo(
                                    &data.close,
                                    indicator.options[0] as i32,
                                    &mut talib_outputs[0],
                                ).ok();
                            }
                            "MOM" => {
                                talib_wrapper::talib_mom(
                                    &data.close,
                                    indicator.options[0] as i32,
                                    &mut talib_outputs[0],
                                ).ok();
                            }
                            "PPO" => {
                                talib_wrapper::talib_ppo(
                                    &data.close,
                                    indicator.options[0] as i32,
                                    indicator.options[1] as i32,
                                    0, // MA type
                                    &mut talib_outputs[0],
                                ).ok();
                            }
                            "ROC" => {
                                talib_wrapper::talib_roc(
                                    &data.close,
                                    indicator.options[0] as i32,
                                    &mut talib_outputs[0],
                                ).ok();
                            }
                            "ROCR" => {
                                talib_wrapper::talib_rocr(
                                    &data.close,
                                    indicator.options[0] as i32,
                                    &mut talib_outputs[0],
                                ).ok();
                            }
                            "ROCP" => {
                                talib_wrapper::talib_rocp(
                                    &data.close,
                                    indicator.options[0] as i32,
                                    &mut talib_outputs[0],
                                ).ok();
                            }
                            "WILLR" => {
                                talib_wrapper::talib_willr(
                                    &data.high,
                                    &data.low,
                                    &data.close,
                                    indicator.options[0] as i32,
                                    &mut talib_outputs[0],
                                ).ok();
                            }
                            "AD" => {
                                talib_wrapper::talib_ad(
                                    &data.high,
                                    &data.low,
                                    &data.close,
                                    &data.volume,
                                    &mut talib_outputs[0],
                                ).ok();
                            }
                            "ADOSC" => {
                                talib_wrapper::talib_adosc(
                                    &data.high,
                                    &data.low,
                                    &data.close,
                                    &data.volume,
                                    indicator.options[0] as i32,
                                    indicator.options[1] as i32,
                                    &mut talib_outputs[0],
                                ).ok();
                            }
                            "OBV" => {
                                talib_wrapper::talib_obv(
                                    &data.close,
                                    &data.volume,
                                    &mut talib_outputs[0],
                                ).ok();
                            }
                            "MFI" => {
                                talib_wrapper::talib_mfi(
                                    &data.high,
                                    &data.low,
                                    &data.close,
                                    &data.volume,
                                    indicator.options[0] as i32,
                                    &mut talib_outputs[0],
                                ).ok();
                            }
                            "BOP" => {
                                talib_wrapper::talib_bop(
                                    &data.open,
                                    &data.high,
                                    &data.low,
                                    &data.close,
                                    &mut talib_outputs[0],
                                ).ok();
                            }
                            "NATR" => {
                                talib_wrapper::talib_natr(
                                    &data.high,
                                    &data.low,
                                    &data.close,
                                    indicator.options[0] as i32,
                                    &mut talib_outputs[0],
                                ).ok();
                            }
                            "STDDEV" => {
                                talib_wrapper::talib_stddev(
                                    &data.close,
                                    indicator.options[0] as i32,
                                    1.0, // nb_dev
                                    &mut talib_outputs[0],
                                ).ok();
                            }
                            "VAR" => {
                                talib_wrapper::talib_var(
                                    &data.close,
                                    indicator.options[0] as i32,
                                    1.0, // nb_dev
                                    &mut talib_outputs[0],
                                ).ok();
                            }
                            "ULTOSC" => {
                                talib_wrapper::talib_ultosc(
                                    &data.high,
                                    &data.low,
                                    &data.close,
                                    indicator.options[0] as i32,
                                    indicator.options[1] as i32,
                                    indicator.options[2] as i32,
                                    &mut talib_outputs[0],
                                ).ok();
                            }
                            "ADXR" => {
                                talib_wrapper::talib_adxr(
                                    &data.high,
                                    &data.low,
                                    &data.close,
                                    indicator.options[0] as i32,
                                    &mut talib_outputs[0],
                                ).ok();
                            }
                            "AROONOSC" => {
                                talib_wrapper::talib_aroonosc(
                                    &data.high,
                                    &data.low,
                                    indicator.options[0] as i32,
                                    &mut talib_outputs[0],
                                ).ok();
                            }
                            "PLUS_DI" => {
                                talib_wrapper::talib_plus_di(
                                    &data.high,
                                    &data.low,
                                    &data.close,
                                    indicator.options[0] as i32,
                                    &mut talib_outputs[0],
                                ).ok();
                            }
                            "MINUS_DI" => {
                                talib_wrapper::talib_minus_di(
                                    &data.high,
                                    &data.low,
                                    &data.close,
                                    indicator.options[0] as i32,
                                    &mut talib_outputs[0],
                                ).ok();
                            }
                            "PLUS_DM" => {
                                talib_wrapper::talib_plus_dm(
                                    &data.high,
                                    &data.low,
                                    indicator.options[0] as i32,
                                    &mut talib_outputs[0],
                                ).ok();
                            }
                            "MINUS_DM" => {
                                talib_wrapper::talib_minus_dm(
                                    &data.high,
                                    &data.low,
                                    indicator.options[0] as i32,
                                    &mut talib_outputs[0],
                                ).ok();
                            }
                            "DX" => {
                                talib_wrapper::talib_dx(
                                    &data.high,
                                    &data.low,
                                    &data.close,
                                    indicator.options[0] as i32,
                                    &mut talib_outputs[0],
                                ).ok();
                            }
                            "LINEARREG" => {
                                talib_wrapper::talib_linearreg(
                                    &data.close,
                                    indicator.options[0] as i32,
                                    &mut talib_outputs[0],
                                ).ok();
                            }
                            "LINEARREG_ANGLE" => {
                                talib_wrapper::talib_linearreg_angle(
                                    &data.close,
                                    indicator.options[0] as i32,
                                    &mut talib_outputs[0],
                                ).ok();
                            }
                            "LINEARREG_SLOPE" => {
                                talib_wrapper::talib_linearreg_slope(
                                    &data.close,
                                    indicator.options[0] as i32,
                                    &mut talib_outputs[0],
                                ).ok();
                            }
                            "LINEARREG_INTERCEPT" => {
                                talib_wrapper::talib_linearreg_intercept(
                                    &data.close,
                                    indicator.options[0] as i32,
                                    &mut talib_outputs[0],
                                ).ok();
                            }
                            "MEDPRICE" => {
                                talib_wrapper::talib_medprice(
                                    &data.high,
                                    &data.low,
                                    &mut talib_outputs[0],
                                ).ok();
                            }
                            "MIDPOINT" => {
                                talib_wrapper::talib_midpoint(
                                    &data.close,
                                    indicator.options[0] as i32,
                                    &mut talib_outputs[0],
                                ).ok();
                            }
                            "MIDPRICE" => {
                                talib_wrapper::talib_midprice(
                                    &data.high,
                                    &data.low,
                                    indicator.options[0] as i32,
                                    &mut talib_outputs[0],
                                ).ok();
                            }
                            "SAR" => {
                                talib_wrapper::talib_sar(
                                    &data.high,
                                    &data.low,
                                    indicator.options[0], // acceleration
                                    indicator.options[1], // maximum
                                    &mut talib_outputs[0],
                                ).ok();
                            }
                            "STOCHF" => {
                                let (k, d) = talib_outputs.split_at_mut(1);
                                talib_wrapper::talib_stochf(
                                    &data.high,
                                    &data.low,
                                    &data.close,
                                    indicator.options[0] as i32,
                                    indicator.options[1] as i32,
                                    0, // MA type
                                    &mut k[0],
                                    &mut d[0],
                                ).ok();
                            }
                            "STOCHRSI" => {
                                let (k, d) = talib_outputs.split_at_mut(1);
                                talib_wrapper::talib_stochrsi(
                                    &data.close,
                                    indicator.options[0] as i32,
                                    indicator.options[1] as i32,
                                    indicator.options[2] as i32,
                                    0, // MA type
                                    &mut k[0],
                                    &mut d[0],
                                ).ok();
                            }
                            "TRIX" => {
                                talib_wrapper::talib_trix(
                                    &data.close,
                                    indicator.options[0] as i32,
                                    &mut talib_outputs[0],
                                ).ok();
                            }
                            "TSF" => {
                                talib_wrapper::talib_tsf(
                                    &data.close,
                                    indicator.options[0] as i32,
                                    &mut talib_outputs[0],
                                ).ok();
                            }
                            "WCLPRICE" => {
                                talib_wrapper::talib_wclprice(
                                    &data.high,
                                    &data.low,
                                    &data.close,
                                    &mut talib_outputs[0],
                                ).ok();
                            }
                            "AVGPRICE" => {
                                talib_wrapper::talib_avgprice(
                                    &data.open,
                                    &data.high,
                                    &data.low,
                                    &data.close,
                                    &mut talib_outputs[0],
                                ).ok();
                            }
                            "TYPPRICE" => {
                                talib_wrapper::talib_typprice(
                                    &data.high,
                                    &data.low,
                                    &data.close,
                                    &mut talib_outputs[0],
                                ).ok();
                            }
                            "TRANGE" => {
                                talib_wrapper::talib_trange(
                                    &data.high,
                                    &data.low,
                                    &data.close,
                                    &mut talib_outputs[0],
                                ).ok();
                            }
                            "MIN" => {
                                talib_wrapper::talib_min(
                                    &data.close,
                                    indicator.options[0] as i32,
                                    &mut talib_outputs[0],
                                ).ok();
                            }
                            "MAX" => {
                                talib_wrapper::talib_max(
                                    &data.close,
                                    indicator.options[0] as i32,
                                    &mut talib_outputs[0],
                                ).ok();
                            }
                            "SUM" => {
                                talib_wrapper::talib_sum(
                                    &data.close,
                                    indicator.options[0] as i32,
                                    &mut talib_outputs[0],
                                ).ok();
                            }
                            _ => {
                                // For unimplemented indicators, just touch the data
                                let _ = black_box(&data.close[0]);
                            }
                        }
                    }
                });
            });
        }
    }

    group.finish();
}

// Helper struct to ensure JSON export on drop
struct JsonExporter;

impl Drop for JsonExporter {
    fn drop(&mut self) {
        let json_path = Path::new("benchmark_results.json");
        if let Err(e) = COLLECTOR.export_to_json(json_path) {
            eprintln!("Failed to export JSON: {}", e);
        }
    }
}

fn setup_benchmarks(c: &mut Criterion) {
    // Create exporter that will save JSON when dropped
    let _json_exporter = JsonExporter;

    let mappings = get_indicator_mappings();

    for (size_name, csv_path) in DATA_SIZES {
        let path = Path::new(csv_path);
        if !path.exists() {
            eprintln!("Warning: Dataset {} not found at {}", size_name, csv_path);
            continue;
        }

        let data = CandleData::from_csv(path)
            .expect(&format!("Failed to load {}", csv_path));

        // Benchmark all indicators
        for mapping in mappings.iter() {
            // Measure and collect results
            measure_and_collect(mapping, &data, size_name);
            benchmark_rust_indicator(c, mapping, &data, size_name);
        }
    }
}

// Custom function to measure and collect benchmark data
fn measure_and_collect(indicator: &IndicatorMapping, data: &CandleData, _size_name: &str) {
    // Quick measurement for each library to collect in JSON
    let iterations = 10;
    let candles = create_candles(data);

    // Measure Rust Native - ALL indicators
    let rust_start = Instant::now();
    match indicator.rust_name {
        "sma" => {
            let input = sma::SmaInput::from_slice(&data.close, sma::SmaParams { period: Some(14) });
            for _ in 0..iterations {
                let _ = black_box(sma::sma(&input));
            }
        }
        "ema" => {
            let input = ema::EmaInput::from_slice(&data.close, ema::EmaParams { period: Some(14) });
            for _ in 0..iterations {
                let _ = black_box(ema::ema(&input));
            }
        }
        "rsi" => {
            let input = rsi::RsiInput::from_slice(&data.close, rsi::RsiParams { period: Some(14) });
            for _ in 0..iterations {
                let _ = black_box(rsi::rsi(&input));
            }
        }
        "bollinger_bands" => {
            let input = bollinger_bands::BollingerBandsInput::from_slice(&data.close,
                bollinger_bands::BollingerBandsParams {
                    period: Some(20), devup: Some(2.0), devdn: Some(2.0),
                    matype: Some("sma".to_string()), devtype: Some(0)
                });
            for _ in 0..iterations {
                let _ = black_box(bollinger_bands::bollinger_bands(&input));
            }
        }
        "macd" => {
            let input = macd::MacdInput::from_slice(&data.close,
                macd::MacdParams {
                    fast_period: Some(12), slow_period: Some(26),
                    signal_period: Some(9), ma_type: None
                });
            for _ in 0..iterations {
                let _ = black_box(macd::macd(&input));
            }
        }
        "atr" => {
            let input = atr::AtrInput::from_candles(&candles, atr::AtrParams { length: Some(14) });
            for _ in 0..iterations {
                let _ = black_box(atr::atr(&input));
            }
        }
        "stoch" => {
            let input = stoch::StochInput::from_candles(&candles,
                stoch::StochParams {
                    fastk_period: Some(14), slowk_period: Some(3),
                    slowk_ma_type: None, slowd_period: Some(3), slowd_ma_type: None
                });
            for _ in 0..iterations {
                let _ = black_box(stoch::stoch(&input));
            }
        }
        "aroon" => {
            let input = aroon::AroonInput::from_candles(&candles, aroon::AroonParams { length: Some(14) });
            for _ in 0..iterations {
                let _ = black_box(aroon::aroon(&input));
            }
        }
        "adx" => {
            let input = adx::AdxInput::from_candles(&candles, adx::AdxParams { period: Some(14) });
            for _ in 0..iterations {
                let _ = black_box(adx::adx(&input));
            }
        }
        "cci" => {
            let input = cci::CciInput::from_candles(&candles, "hlc3", cci::CciParams { period: Some(20) });
            for _ in 0..iterations {
                let _ = black_box(cci::cci(&input));
            }
        }
        "dema" => {
            let input = dema::DemaInput::from_slice(&data.close, dema::DemaParams { period: Some(14) });
            for _ in 0..iterations {
                let _ = black_box(dema::dema(&input));
            }
        }
        "tema" => {
            let input = tema::TemaInput::from_slice(&data.close, tema::TemaParams { period: Some(14) });
            for _ in 0..iterations {
                let _ = black_box(tema::tema(&input));
            }
        }
        "wma" => {
            let input = wma::WmaInput::from_slice(&data.close, wma::WmaParams { period: Some(14) });
            for _ in 0..iterations {
                let _ = black_box(wma::wma(&input));
            }
        }
        "kama" => {
            let input = kama::KamaInput::from_slice(&data.close, kama::KamaParams { period: Some(14) });
            for _ in 0..iterations {
                let _ = black_box(kama::kama(&input));
            }
        }
        "trima" => {
            let input = trima::TrimaInput::from_slice(&data.close, trima::TrimaParams { period: Some(14) });
            for _ in 0..iterations {
                let _ = black_box(trima::trima(&input));
            }
        }
        "hma" => {
            let input = hma::HmaInput::from_slice(&data.close, hma::HmaParams { period: Some(14) });
            for _ in 0..iterations {
                let _ = black_box(hma::hma(&input));
            }
        }
        "zlema" => {
            let input = zlema::ZlemaInput::from_slice(&data.close, zlema::ZlemaParams { period: Some(14) });
            for _ in 0..iterations {
                let _ = black_box(zlema::zlema(&input));
            }
        }
        "vwma" => {
            let input = vwma::VwmaInput::from_candles(&candles, "close", vwma::VwmaParams { period: Some(14) });
            for _ in 0..iterations {
                let _ = black_box(vwma::vwma(&input));
            }
        }
        "wilders" => {
            let input = wilders::WildersInput::from_slice(&data.close, wilders::WildersParams { period: Some(14) });
            for _ in 0..iterations {
                let _ = black_box(wilders::wilders(&input));
            }
        }
        "apo" => {
            let input = apo::ApoInput::from_slice(&data.close,
                apo::ApoParams { short_period: Some(12), long_period: Some(26) });
            for _ in 0..iterations {
                let _ = black_box(apo::apo(&input));
            }
        }
        "cmo" => {
            let input = cmo::CmoInput::from_slice(&data.close, cmo::CmoParams { period: Some(14) });
            for _ in 0..iterations {
                let _ = black_box(cmo::cmo(&input));
            }
        }
        "dpo" => {
            let input = dpo::DpoInput::from_slice(&data.close, dpo::DpoParams { period: Some(14) });
            for _ in 0..iterations {
                let _ = black_box(dpo::dpo(&input));
            }
        }
        "mom" => {
            let input = mom::MomInput::from_slice(&data.close, mom::MomParams { period: Some(10) });
            for _ in 0..iterations {
                let _ = black_box(mom::mom(&input));
            }
        }
        "ppo" => {
            let input = ppo::PpoInput::from_slice(&data.close,
                ppo::PpoParams { fast_period: Some(12), slow_period: Some(26), ma_type: None });
            for _ in 0..iterations {
                let _ = black_box(ppo::ppo(&input));
            }
        }
        "roc" => {
            let input = roc::RocInput::from_slice(&data.close, roc::RocParams { period: Some(10) });
            for _ in 0..iterations {
                let _ = black_box(roc::roc(&input));
            }
        }
        "rocr" => {
            let input = rocr::RocrInput::from_slice(&data.close, rocr::RocrParams { period: Some(10) });
            for _ in 0..iterations {
                let _ = black_box(rocr::rocr(&input));
            }
        }
        "rocp" => {
            let input = rocp::RocpInput::from_slice(&data.close, rocp::RocpParams { period: Some(10) });
            for _ in 0..iterations {
                let _ = black_box(rocp::rocp(&input));
            }
        }
        "willr" => {
            let input = willr::WillrInput::from_candles(&candles, willr::WillrParams { period: Some(14) });
            for _ in 0..iterations {
                let _ = black_box(willr::willr(&input));
            }
        }
        "ad" => {
            let input = ad::AdInput::from_candles(&candles, ad::AdParams {});
            for _ in 0..iterations {
                let _ = black_box(ad::ad(&input));
            }
        }
        "adosc" => {
            let input = adosc::AdoscInput::from_candles(&candles,
                adosc::AdoscParams { short_period: Some(3), long_period: Some(10) });
            for _ in 0..iterations {
                let _ = black_box(adosc::adosc(&input));
            }
        }
        "obv" => {
            let input = obv::ObvInput::from_candles(&candles, obv::ObvParams {});
            for _ in 0..iterations {
                let _ = black_box(obv::obv(&input));
            }
        }
        "mfi" => {
            let input = mfi::MfiInput::from_candles(&candles, "hlc3", mfi::MfiParams { period: Some(14) });
            for _ in 0..iterations {
                let _ = black_box(mfi::mfi(&input));
            }
        }
        "ao" => {
            let input = ao::AoInput::from_candles(&candles, "hl2",
                ao::AoParams { short_period: Some(5), long_period: Some(34) });
            for _ in 0..iterations {
                let _ = black_box(ao::ao(&input));
            }
        }
        "bop" => {
            let input = bop::BopInput::from_candles(&candles, bop::BopParams {});
            for _ in 0..iterations {
                let _ = black_box(bop::bop(&input));
            }
        }
        "natr" => {
            let input = natr::NatrInput::from_candles(&candles, natr::NatrParams { period: Some(14) });
            for _ in 0..iterations {
                let _ = black_box(natr::natr(&input));
            }
        }
        "stddev" => {
            let input = stddev::StdDevInput::from_slice(&data.close, stddev::StdDevParams { period: Some(5), nbdev: Some(1.0) });
            for _ in 0..iterations {
                let _ = black_box(stddev::stddev(&input));
            }
        }
        "var" => {
            let input = var::VarInput::from_slice(&data.close, var::VarParams { period: Some(5), nbdev: Some(1.0) });
            for _ in 0..iterations {
                let _ = black_box(var::var(&input));
            }
        }
        "ultosc" => {
            let input = ultosc::UltOscInput::from_candles(&candles, "high", "low", "close",
                ultosc::UltOscParams { timeperiod1: Some(7), timeperiod2: Some(14), timeperiod3: Some(28) });
            for _ in 0..iterations {
                let _ = black_box(ultosc::ultosc(&input));
            }
        }
        "adxr" => {
            let input = adxr::AdxrInput::from_candles(&candles, adxr::AdxrParams { period: Some(14) });
            for _ in 0..iterations {
                let _ = black_box(adxr::adxr(&input));
            }
        }
        "aroonosc" => {
            let input = aroonosc::AroonOscInput::from_candles(&candles, aroonosc::AroonOscParams { length: Some(14) });
            for _ in 0..iterations {
                let _ = black_box(aroonosc::aroon_osc(&input));
            }
        }
        "di" | "dmi" => {
            let input = di::DiInput::from_candles(&candles, di::DiParams { period: Some(14) });
            for _ in 0..iterations {
                let _ = black_box(di::di(&input));
            }
        }
        "dm" => {
            let input = dm::DmInput::from_candles(&candles, dm::DmParams { period: Some(14) });
            for _ in 0..iterations {
                let _ = black_box(dm::dm(&input));
            }
        }
        "plus_dm" | "minus_dm" => {
            let input = dm::DmInput::from_candles(&candles, dm::DmParams { period: Some(14) });
            for _ in 0..iterations {
                let _ = black_box(dm::dm(&input));
            }
        }
        "dx" => {
            let input = dx::DxInput::from_candles(&candles, dx::DxParams { period: Some(14) });
            for _ in 0..iterations {
                let _ = black_box(dx::dx(&input));
            }
        }
        "fisher" => {
            let input = fisher::FisherInput::from_candles(&candles, fisher::FisherParams { period: Some(14) });
            for _ in 0..iterations {
                let _ = black_box(fisher::fisher(&input));
            }
        }
        "fosc" => {
            let input = fosc::FoscInput::from_slice(&data.close, fosc::FoscParams { period: Some(14) });
            for _ in 0..iterations {
                let _ = black_box(fosc::fosc(&input));
            }
        }
        "kvo" => {
            let input = kvo::KvoInput::from_candles(&candles,
                kvo::KvoParams { short_period: Some(34), long_period: Some(55) });
            for _ in 0..iterations {
                let _ = black_box(kvo::kvo(&input));
            }
        }
        "linearreg_slope" => {
            let input = linearreg_slope::LinearRegSlopeInput::from_slice(&data.close,
                linearreg_slope::LinearRegSlopeParams { period: Some(14) });
            for _ in 0..iterations {
                let _ = black_box(linearreg_slope::linearreg_slope(&input));
            }
        }
        "linreg" => {
            let input = linreg::LinRegInput::from_slice(&data.close, linreg::LinRegParams { period: Some(14) });
            for _ in 0..iterations {
                let _ = black_box(linreg::linreg(&input));
            }
        }
        "linearreg_intercept" => {
            let input = linearreg_intercept::LinearRegInterceptInput::from_slice(&data.close,
                linearreg_intercept::LinearRegInterceptParams { period: Some(14) });
            for _ in 0..iterations {
                let _ = black_box(linearreg_intercept::linearreg_intercept(&input));
            }
        }
        "mass" => {
            let input = mass::MassInput::from_candles(&candles, "high", "low",
                mass::MassParams { period: Some(25) });
            for _ in 0..iterations {
                let _ = black_box(mass::mass(&input));
            }
        }
        "medprice" => {
            let input = medprice::MedpriceInput::from_candles(&candles, "high", "low", medprice::MedpriceParams {});
            for _ in 0..iterations {
                let _ = black_box(medprice::medprice(&input));
            }
        }
        "midpoint" => {
            let input = midpoint::MidpointInput::from_slice(&data.close, midpoint::MidpointParams { period: Some(14) });
            for _ in 0..iterations {
                let _ = black_box(midpoint::midpoint(&input));
            }
        }
        "midprice" => {
            let input = midprice::MidpriceInput::from_candles(&candles, "high", "low", midprice::MidpriceParams { period: Some(14) });
            for _ in 0..iterations {
                let _ = black_box(midprice::midprice(&input));
            }
        }
        "nvi" => {
            let input = nvi::NviInput::from_candles(&candles, "close", nvi::NviParams {});
            for _ in 0..iterations {
                let _ = black_box(nvi::nvi(&input));
            }
        }
        "pvi" => {
            let input = pvi::PviInput::from_candles(&candles, "close", "volume", pvi::PviParams { initial_value: Some(1000.0) });
            for _ in 0..iterations {
                let _ = black_box(pvi::pvi(&input));
            }
        }
        "qstick" => {
            let input = qstick::QstickInput::from_candles(&candles, "open", "close", qstick::QstickParams { period: Some(14) });
            for _ in 0..iterations {
                let _ = black_box(qstick::qstick(&input));
            }
        }
        "sar" | "psar" => {
            let input = sar::SarInput::from_candles(&candles,
                sar::SarParams { acceleration: Some(0.02), maximum: Some(0.2) });
            for _ in 0..iterations {
                let _ = black_box(sar::sar(&input));
            }
        }
        "srsi" | "stochrsi" => {
            let input = srsi::SrsiInput::from_slice(&data.close,
                srsi::SrsiParams { rsi_period: Some(14), stoch_period: Some(14), k: Some(3), d: Some(3), source: None });
            for _ in 0..iterations {
                let _ = black_box(srsi::srsi(&input));
            }
        }
        "stochf" => {
            let input = stochf::StochfInput::from_candles(&candles,
                stochf::StochfParams { fastk_period: Some(5), fastd_period: Some(3), fastd_matype: None });
            for _ in 0..iterations {
                let _ = black_box(stochf::stochf(&input));
            }
        }
        "trix" => {
            let input = trix::TrixInput::from_slice(&data.close, trix::TrixParams { period: Some(14) });
            for _ in 0..iterations {
                let _ = black_box(trix::trix(&input));
            }
        }
        "tsf" => {
            let input = tsf::TsfInput::from_slice(&data.close, tsf::TsfParams { period: Some(14) });
            for _ in 0..iterations {
                let _ = black_box(tsf::tsf(&input));
            }
        }
        "vidya" => {
            let input = vidya::VidyaInput::from_slice(&data.close,
                vidya::VidyaParams { short_period: Some(2), long_period: Some(5), alpha: Some(0.2) });
            for _ in 0..iterations {
                let _ = black_box(vidya::vidya(&input));
            }
        }
        "vosc" => {
            let input = vosc::VoscInput::from_candles(&candles, "volume",
                vosc::VoscParams { short_period: Some(5), long_period: Some(10) });
            for _ in 0..iterations {
                let _ = black_box(vosc::vosc(&input));
            }
        }
        "wad" => {
            let input = wad::WadInput::from_candles(&candles);
            for _ in 0..iterations {
                let _ = black_box(wad::wad(&input));
            }
        }
        "wclprice" | "wcprice" => {
            let input = wclprice::WclpriceInput::from_candles(&candles);
            for _ in 0..iterations {
                let _ = black_box(wclprice::wclprice(&input));
            }
        }
        "cvi" => {
            let input = cvi::CviInput::from_candles(&candles, cvi::CviParams { period: Some(14) });
            for _ in 0..iterations {
                let _ = black_box(cvi::cvi(&input));
            }
        }
        "emv" => {
            let input = emv::EmvInput::from_candles(&candles);
            for _ in 0..iterations {
                let _ = black_box(emv::emv(&input));
            }
        }
        "marketefi" => {
            let input = marketefi::MarketefiInput::from_candles(&candles, "high", "low", "volume", marketefi::MarketefiParams {});
            for _ in 0..iterations {
                let _ = black_box(marketefi::marketefi(&input));
            }
        }
        "minmax" | "minmax_min" => {
            let input = minmax::MinmaxInput::from_slices(&data.high, &data.low, minmax::MinmaxParams { order: Some(14) });
            for _ in 0..iterations {
                let _ = black_box(minmax::minmax(&input));
            }
        }
        "msw" => {
            let input = msw::MswInput::from_slice(&data.close, msw::MswParams { period: Some(14) });
            for _ in 0..iterations {
                let _ = black_box(msw::msw(&input));
            }
        }
        _ => {
            // Skip unmapped indicators
            return;
        }
    }
    let rust_duration = rust_start.elapsed() / iterations as u32;
    COLLECTOR.add_measurement(indicator.rust_name, LibraryType::RustNative, rust_duration, data.len());

    // Measure Rust via FFI (subset of indicators with wrappers)
    let rust_ffi_start = Instant::now();
    match indicator.rust_name {
        "sma" => {
            let mut out = vec![0.0; data.len()];
            for _ in 0..iterations { unsafe { rust_ffi::rust_sma(data.len() as i32, data.close.as_ptr(), 14, out.as_mut_ptr()); } }
        }
        "ema" => {
            let mut out = vec![0.0; data.len()];
            for _ in 0..iterations { unsafe { rust_ffi::rust_ema(data.len() as i32, data.close.as_ptr(), 14, out.as_mut_ptr()); } }
        }
        "rsi" => {
            let mut out = vec![0.0; data.len()];
            for _ in 0..iterations { unsafe { rust_ffi::rust_rsi(data.len() as i32, data.close.as_ptr(), 14, out.as_mut_ptr()); } }
        }
        "atr" => {
            let mut out = vec![0.0; data.len()];
            for _ in 0..iterations { unsafe { rust_ffi::rust_atr(data.len() as i32, data.high.as_ptr(), data.low.as_ptr(), data.close.as_ptr(), 14, out.as_mut_ptr()); } }
        }
        "adx" => {
            let mut out = vec![0.0; data.len()];
            for _ in 0..iterations { unsafe { rust_ffi::rust_adx(data.len() as i32, data.high.as_ptr(), data.low.as_ptr(), data.close.as_ptr(), 14, out.as_mut_ptr()); } }
        }
        "ao" => {
            let mut out = vec![0.0; data.len()];
            for _ in 0..iterations { unsafe { rust_ffi::rust_ao(data.len() as i32, data.high.as_ptr(), data.low.as_ptr(), out.as_mut_ptr()); } }
        }
        "aroon" => {
            let mut up = vec![0.0; data.len()];
            let mut down = vec![0.0; data.len()];
            for _ in 0..iterations { unsafe { rust_ffi::rust_aroon(data.len() as i32, data.high.as_ptr(), data.low.as_ptr(), 14, up.as_mut_ptr(), down.as_mut_ptr()); } }
        }
        "cci" => {
            let mut out = vec![0.0; data.len()];
            for _ in 0..iterations { unsafe { rust_ffi::rust_cci(data.len() as i32, data.high.as_ptr(), data.low.as_ptr(), data.close.as_ptr(), 20, out.as_mut_ptr()); } }
        }
        "cmo" => {
            let mut out = vec![0.0; data.len()];
            for _ in 0..iterations { unsafe { rust_ffi::rust_cmo(data.len() as i32, data.close.as_ptr(), 14, out.as_mut_ptr()); } }
        }
        "di" => {
            let mut plus_di = vec![0.0; data.len()];
            let mut minus_di = vec![0.0; data.len()];
            for _ in 0..iterations { unsafe { rust_ffi::rust_di(data.len() as i32, data.high.as_ptr(), data.low.as_ptr(), data.close.as_ptr(), 14, plus_di.as_mut_ptr(), minus_di.as_mut_ptr()); } }
        }
        "macd" => {
            let mut macd_line = vec![0.0; data.len()];
            let mut signal = vec![0.0; data.len()];
            let mut hist = vec![0.0; data.len()];
            for _ in 0..iterations { unsafe { rust_ffi::rust_macd(data.len() as i32, data.close.as_ptr(), 12, 26, 9, macd_line.as_mut_ptr(), signal.as_mut_ptr(), hist.as_mut_ptr()); } }
        }
        "mfi" => {
            let mut out = vec![0.0; data.len()];
            for _ in 0..iterations { unsafe { rust_ffi::rust_mfi(data.len() as i32, data.high.as_ptr(), data.low.as_ptr(), data.close.as_ptr(), data.volume.as_ptr(), 14, out.as_mut_ptr()); } }
        }
        "mom" => {
            let mut out = vec![0.0; data.len()];
            for _ in 0..iterations { unsafe { rust_ffi::rust_mom(data.len() as i32, data.close.as_ptr(), 10, out.as_mut_ptr()); } }
        }
        "obv" => {
            let mut out = vec![0.0; data.len()];
            for _ in 0..iterations { unsafe { rust_ffi::rust_obv(data.len() as i32, data.close.as_ptr(), data.volume.as_ptr(), out.as_mut_ptr()); } }
        }
        "ppo" => {
            let mut ppo_out = vec![0.0; data.len()];
            let mut sig_out = vec![0.0; data.len()];
            let mut hist_out = vec![0.0; data.len()];
            for _ in 0..iterations { unsafe { rust_ffi::rust_ppo(data.len() as i32, data.close.as_ptr(), 12, 26, 9, ppo_out.as_mut_ptr(), sig_out.as_mut_ptr(), hist_out.as_mut_ptr()); } }
        }
        "roc" => {
            let mut out = vec![0.0; data.len()];
            for _ in 0..iterations { unsafe { rust_ffi::rust_roc(data.len() as i32, data.close.as_ptr(), 10, out.as_mut_ptr()); } }
        }
        "linreg" => {
            let mut out = vec![0.0; data.len()];
            for _ in 0..iterations { unsafe { rust_ffi::rust_linreg(data.len() as i32, data.close.as_ptr(), 14, out.as_mut_ptr()); } }
        }
        "linearreg_angle" => {
            let mut out = vec![0.0; data.len()];
            for _ in 0..iterations { unsafe { rust_ffi::rust_linearreg_angle(data.len() as i32, data.close.as_ptr(), 14, out.as_mut_ptr()); } }
        }
        "linreg" => {
            let mut out = vec![0.0; data.len()];
            for _ in 0..iterations { unsafe { rust_ffi::rust_linreg(data.len() as i32, data.close.as_ptr(), 14, out.as_mut_ptr()); } }
        }
        "stoch" => {
            let mut k = vec![0.0; data.len()]; let mut d = vec![0.0; data.len()];
            for _ in 0..iterations { unsafe { rust_ffi::rust_stoch(data.len() as i32, data.high.as_ptr(), data.low.as_ptr(), data.close.as_ptr(), 5, 3, 0, k.as_mut_ptr(), d.as_mut_ptr()); } }
        }
        "srsi" | "stochrsi" => {
            let mut k = vec![0.0; data.len()]; let mut d = vec![0.0; data.len()];
            for _ in 0..iterations { unsafe { rust_ffi::rust_srsi(data.len() as i32, data.close.as_ptr(), 14, 5, 3, 0, k.as_mut_ptr(), d.as_mut_ptr()); } }
        }
        "trix" => {
            let mut out = vec![0.0; data.len()];
            for _ in 0..iterations { unsafe { rust_ffi::rust_trix(data.len() as i32, data.close.as_ptr(), 14, out.as_mut_ptr()); } }
        }
        "willr" => {
            let mut out = vec![0.0; data.len()];
            for _ in 0..iterations { unsafe { rust_ffi::rust_willr(data.len() as i32, data.high.as_ptr(), data.low.as_ptr(), data.close.as_ptr(), 14, out.as_mut_ptr()); } }
        }
        "wma" => { let mut out = vec![0.0; data.len()]; for _ in 0..iterations { unsafe { rust_ffi::rust_wma(data.len() as i32, data.close.as_ptr(), 14, out.as_mut_ptr()); } } }
        "dema" => { let mut out = vec![0.0; data.len()]; for _ in 0..iterations { unsafe { rust_ffi::rust_dema(data.len() as i32, data.close.as_ptr(), 14, out.as_mut_ptr()); } } }
        "tema" => { let mut out = vec![0.0; data.len()]; for _ in 0..iterations { unsafe { rust_ffi::rust_tema(data.len() as i32, data.close.as_ptr(), 14, out.as_mut_ptr()); } } }
        "kama" => { let mut out = vec![0.0; data.len()]; for _ in 0..iterations { unsafe { rust_ffi::rust_kama(data.len() as i32, data.close.as_ptr(), 14, out.as_mut_ptr()); } } }
        "trima" => { let mut out = vec![0.0; data.len()]; for _ in 0..iterations { unsafe { rust_ffi::rust_trima(data.len() as i32, data.close.as_ptr(), 14, out.as_mut_ptr()); } } }
        "hma" => { let mut out = vec![0.0; data.len()]; for _ in 0..iterations { unsafe { rust_ffi::rust_hma(data.len() as i32, data.close.as_ptr(), 14, out.as_mut_ptr()); } } }
        "zlema" => { let mut out = vec![0.0; data.len()]; for _ in 0..iterations { unsafe { rust_ffi::rust_zlema(data.len() as i32, data.close.as_ptr(), 14, out.as_mut_ptr()); } } }
        "vwma" => { let mut out = vec![0.0; data.len()]; for _ in 0..iterations { unsafe { rust_ffi::rust_vwma(data.len() as i32, data.close.as_ptr(), data.volume.as_ptr(), 14, out.as_mut_ptr()); } } }
        "wilders" => { let mut out = vec![0.0; data.len()]; for _ in 0..iterations { unsafe { rust_ffi::rust_wilders(data.len() as i32, data.close.as_ptr(), 14, out.as_mut_ptr()); } } }
        "vidya" => { let mut out = vec![0.0; data.len()]; for _ in 0..iterations { unsafe { rust_ffi::rust_vidya(data.len() as i32, data.close.as_ptr(), 2, 5, 0.2, out.as_mut_ptr()); } } }
        _ => {}
    }
    let rust_ffi_duration = rust_ffi_start.elapsed() / iterations as u32;
    COLLECTOR.add_measurement(indicator.rust_name, LibraryType::RustFFI, rust_ffi_duration, data.len());

    // Measure Tulip
    unsafe {
        // Determine number of outputs based on indicator
        let output_count = match indicator.tulip_name {
            "bbands" | "macd" => 3,
            "stoch" | "stochf" | "stochrsi" => 2,
            "aroon" | "di" | "dm" => 2,
            _ => 1,
        };

        let mut output_buffers: Vec<Vec<f64>> = (0..output_count)
            .map(|_| vec![0.0; data.len()])
            .collect();

        // Prepare inputs based on indicator requirements
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

        let mut output_refs: Vec<&mut [f64]> = output_buffers
            .iter_mut()
            .map(|v| &mut v[..])
            .collect();

        let tulip_start = Instant::now();
        for _ in 0..iterations {
            let _ = tulip::call_indicator(
                indicator.tulip_name,
                data.len(),
                &inputs,
                &indicator.options,
                &mut output_refs,
            );
        }
        let tulip_duration = tulip_start.elapsed() / iterations as u32;
        COLLECTOR.add_measurement(indicator.rust_name, LibraryType::TulipFFI, tulip_duration, data.len());
    }

    // Measure TA-LIB (if available and mapped)
    #[cfg(not(feature = "talib"))]
    let _has_talib = false;
    #[cfg(feature = "talib")]
    let _has_talib = std::env::var("TALIB_PATH").is_ok();

    #[cfg(feature = "talib")]
    if _has_talib {
        if let Some(name) = indicator.talib_name {
            use cross_library_benchmark::talib_wrapper;
            let talib_start = Instant::now();
            // Allocate outputs for multi-output indicators
            let mut out1 = vec![0.0; data.len()];
            let mut out2 = vec![0.0; data.len()];
            let mut out3 = vec![0.0; data.len()];
            for _ in 0..iterations {
                unsafe {
                    match name {
                        "SMA" => { let _ = talib_wrapper::talib_sma(&data.close, 14, &mut out1); }
                        "EMA" => { let _ = talib_wrapper::talib_ema(&data.close, 14, &mut out1); }
                        "RSI" => { let _ = talib_wrapper::talib_rsi(&data.close, 14, &mut out1); }
                        "BBANDS" => { let _ = talib_wrapper::talib_bbands(&data.close, 20, 2.0, 2.0, &mut out1, &mut out2, &mut out3); }
                        "MACD" => { let _ = talib_wrapper::talib_macd(&data.close, 12, 26, 9, &mut out1, &mut out2, &mut out3); }
                        "ATR" => { let _ = talib_wrapper::talib_atr(&data.high, &data.low, &data.close, 14, &mut out1); }
                        "STOCH" => { let _ = talib_wrapper::talib_stoch(&data.high, &data.low, &data.close, 14, 3, 3, &mut out1, &mut out2); }
                        "AROON" => { let _ = talib_wrapper::talib_aroon(&data.high, &data.low, 14, &mut out1, &mut out2); }
                        "ADX" => { let _ = talib_wrapper::talib_adx(&data.high, &data.low, &data.close, 14, &mut out1); }
                        "CCI" => { let _ = talib_wrapper::talib_cci(&data.high, &data.low, &data.close, 20, &mut out1); }
                        "BOP" => { let _ = talib_wrapper::talib_bop(&data.open, &data.high, &data.low, &data.close, &mut out1); }
                        "NATR" => { let _ = talib_wrapper::talib_natr(&data.high, &data.low, &data.close, 14, &mut out1); }
                        "STDDEV" => { let _ = talib_wrapper::talib_stddev(&data.close, 5, 1.0, &mut out1); }
                        "VAR" => { let _ = talib_wrapper::talib_var(&data.close, 5, 1.0, &mut out1); }
                        "ULTOSC" => { let _ = talib_wrapper::talib_ultosc(&data.high, &data.low, &data.close, 7,14,28, &mut out1); }
                        "ADXR" => { let _ = talib_wrapper::talib_adxr(&data.high, &data.low, &data.close, 14, &mut out1); }
                        "AROONOSC" => { let _ = talib_wrapper::talib_aroonosc(&data.high, &data.low, 14, &mut out1); }
                        "PLUS_DI" => { let _ = talib_wrapper::talib_plus_di(&data.high, &data.low, &data.close, 14, &mut out1); }
                        "PLUS_DM" => { let _ = talib_wrapper::talib_plus_dm(&data.high, &data.low, 14, &mut out1); }
                        "DX" => { let _ = talib_wrapper::talib_dx(&data.high, &data.low, &data.close, 14, &mut out1); }
                        "LINEARREG" => { let _ = talib_wrapper::talib_linearreg(&data.close, 14, &mut out1); }
                        "LINEARREG_ANGLE" => { let _ = talib_wrapper::talib_linearreg_angle(&data.close, 14, &mut out1); }
                        "LINEARREG_SLOPE" => { let _ = talib_wrapper::talib_linearreg_slope(&data.close, 14, &mut out1); }
                        "LINEARREG_INTERCEPT" => { let _ = talib_wrapper::talib_linearreg_intercept(&data.close, 14, &mut out1); }
                        "MEDPRICE" => { let _ = talib_wrapper::talib_medprice(&data.high, &data.low, &mut out1); }
                        "MIDPOINT" => { let _ = talib_wrapper::talib_midpoint(&data.close, 14, &mut out1); }
                        "MIDPRICE" => { let _ = talib_wrapper::talib_midprice(&data.high, &data.low, 14, &mut out1); }
                        "ROCP" => { let _ = talib_wrapper::talib_rocp(&data.close, 14, &mut out1); }
                        "SAR" => { let _ = talib_wrapper::talib_sar(&data.high, &data.low, 0.02, 0.2, &mut out1); }
                        "STOCHF" => { let _ = talib_wrapper::talib_stochf(&data.high, &data.low, &data.close, 5, 3, 0, &mut out1, &mut out2); }
                        "STOCHRSI" => { let _ = talib_wrapper::talib_stochrsi(&data.close, 14, 14, 3, 3, &mut out1, &mut out2); }
                        "TRIX" => { let _ = talib_wrapper::talib_trix(&data.close, 14, &mut out1); }
                        "TSF" => { let _ = talib_wrapper::talib_tsf(&data.close, 14, &mut out1); }
                        "WCLPRICE" => { let _ = talib_wrapper::talib_wclprice(&data.high, &data.low, &data.close, &mut out1); }
                        "AVGPRICE" => { let _ = talib_wrapper::talib_avgprice(&data.open, &data.high, &data.low, &data.close, &mut out1); }
                        "TYPPRICE" => { let _ = talib_wrapper::talib_typprice(&data.high, &data.low, &data.close, &mut out1); }
                        "TRANGE" => { let _ = talib_wrapper::talib_trange(&data.high, &data.low, &data.close, &mut out1); }
                        "MIN" => { let _ = talib_wrapper::talib_min(&data.close, 14, &mut out1); }
                        "MAX" => { let _ = talib_wrapper::talib_max(&data.close, 14, &mut out1); }
                        _ => {}
                    }
                }
            }
            let talib_duration = talib_start.elapsed() / iterations as u32;
            COLLECTOR.add_measurement(indicator.rust_name, LibraryType::TalibFFI, talib_duration, data.len());
        }
    }
}

criterion_group!(benches, setup_benchmarks);
criterion_main!(benches);
