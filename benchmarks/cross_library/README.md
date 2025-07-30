# Cross-Library TA Indicator Benchmarks

This benchmark suite compares the performance of Rust-TA indicators against Tulip Indicators (C) and TA-Lib (C).

## Architecture

The benchmark framework provides:

1. **Fair FFI Comparison**: Rust indicators are called through FFI (like C libraries) to ensure apples-to-apples comparison
2. **Multiple Data Sizes**: Tests with 4k, 10k, 100k, and 1M candle datasets
3. **Comprehensive Metrics**: Measures throughput, latency, and FFI overhead
4. **Report Generation**: HTML and CSV reports with performance comparisons

## Structure

```
benchmarks/cross_library/
├── src/
│   ├── lib.rs           # Main library with Tulip/TA-Lib bindings
│   ├── rust_ffi.rs      # FFI exports for Rust indicators
│   └── report.rs        # Report generation utilities
├── benches/
│   └── cross_library_comparison.rs  # Criterion benchmark suite
├── build.rs             # Build script for C libraries
└── Cargo.toml
```

## Building

### Prerequisites

On Windows:
- Visual Studio 2019+ or Build Tools for Visual Studio
- Rust toolchain (stable)

On Linux/macOS:
- GCC or Clang
- Make

### Build Steps

1. Initialize submodules (if not done):
```bash
git submodule update --init --recursive
```

2. Build the benchmark:
```bash
cd benchmarks/cross_library
cargo build --release
```

Note: TA-Lib requires manual setup on Windows. Download pre-built binaries from ta-lib.org.

## Running Benchmarks

Run all benchmarks:
```bash
cargo bench
```

Run specific indicator:
```bash
cargo bench -- sma
```

Generate HTML report:
```bash
cargo bench -- --save-baseline comparison
```

## Indicators Benchmarked

Currently comparing these indicators:
- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)
- Relative Strength Index (RSI)
- Bollinger Bands
- MACD
- Average True Range (ATR)
- Stochastic
- Aroon
- ADX
- CCI

## Performance Considerations

The benchmarks measure:
1. **Direct Rust Performance**: Native Rust function calls
2. **Rust FFI Performance**: Rust called through C FFI
3. **Tulip Performance**: C library performance
4. **TA-Lib Performance**: C library performance (when available)

Key metrics:
- Median execution time (ms)
- Throughput (MB/s)
- FFI overhead percentage
- Performance ratios (Rust/Tulip, Rust/TA-Lib)

## Results

Results are saved in:
- `target/criterion/` - Raw Criterion data
- `benchmarks/cross_library/results/` - Generated reports

## Adding New Indicators

1. Add indicator mapping in `benches/cross_library_comparison.rs`:
```rust
IndicatorMapping {
    rust_name: "indicator_name",
    tulip_name: "ti_indicator",
    talib_name: Some("TA_INDICATOR"),
    inputs: vec!["close"],
    options: vec![14.0],
}
```

2. Add FFI wrapper in `src/rust_ffi.rs`:
```rust
#[no_mangle]
pub unsafe extern "C" fn rust_indicator(...) -> c_int {
    // Implementation
}
```

3. Update benchmark cases in the benchmark runner.

## Troubleshooting

### Windows Build Issues
- Ensure Visual Studio or Build Tools are installed
- For TA-Lib, download pre-built binaries and set LIB/INCLUDE paths

### Missing Indicators
- Check that the indicator exists in both libraries
- Verify parameter compatibility between implementations

### Performance Anomalies
- Ensure release builds with optimizations
- Check for debug assertions in release mode
- Verify data alignment for SIMD operations