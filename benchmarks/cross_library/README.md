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
- (Optional) TA-LIB - See [TALIB_SETUP.md](TALIB_SETUP.md) for installation guide

On Linux/macOS:
- GCC or Clang
- Make
- (Optional) TA-LIB development libraries

### Build Steps

1. Initialize submodules (if not done):
```bash
git submodule update --init --recursive
```

2. Build the benchmark (without TA-LIB):
```bash
cd benchmarks/cross_library
cargo build --release
```

3. Build with TA-LIB support (optional):
```bash
# First, install TA-LIB and set TALIB_PATH environment variable
# See TALIB_SETUP.md for detailed instructions
cargo build --release --features talib
```

## Running Benchmarks

Run benchmarks without TA-LIB:
```bash
cargo bench
```

Run benchmarks with TA-LIB (if installed):
```bash
cargo bench --features talib
```

Run specific indicator:
```bash
cargo bench -- sma
```

Generate HTML report:
```bash
cargo bench -- --save-baseline comparison
```

Test TA-LIB installation:
```bash
cargo run --example test_talib --features talib
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
4. **TA-Lib Performance**: C library performance (when available with `--features talib`)

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
- For TA-Lib, see [TALIB_SETUP.md](TALIB_SETUP.md) for detailed Windows installation

### TA-LIB Integration Issues
- Run `cargo run --example test_talib --features talib` to verify installation
- Check that `TALIB_PATH` environment variable is set correctly
- Ensure you have the 64-bit version of TA-LIB for 64-bit Rust

### Missing Indicators
- Check that the indicator exists in both libraries
- Verify parameter compatibility between implementations

### Performance Anomalies
- Ensure release builds with optimizations
- Check for debug assertions in release mode
- Verify data alignment for SIMD operations

## TA-LIB Support

TA-LIB integration is optional. The benchmarks work with just Tulip Indicators, but adding TA-LIB provides:
- Industry-standard comparison baseline
- Additional indicator implementations
- Cross-validation of results

See [TALIB_SETUP.md](TALIB_SETUP.md) for installation instructions.