# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Rust-Backtester is a high-performance technical analysis library implementing 178+ indicators (targeting 300 total). The project emphasizes performance optimization with SIMD instructions, batch processing, and WebAssembly support.

## Common Development Commands

### Building
```bash
cargo build                 # Debug build
cargo build --release       # Release build with optimizations
cargo build --features nightly-avx # Build with AVX2/AVX512 SIMD support (requires nightly Rust)
cargo build --features wasm # Build with WebAssembly support
cargo build --features python # Build with Python bindings
cargo build --release --features nightly-avx # Optimized build with SIMD
```

### Testing
```bash
cargo test --lib           # Run library tests
cargo test --verbose --lib # Run tests with verbose output

# Test specific indicator (with nightly-avx for SIMD optimizations)
cargo test --features nightly-avx --lib indicators::moving_averages::maaq -- --nocapture
cargo test --features nightly-avx --lib indicators::rsi -- --nocapture
```

### Benchmarking
```bash
cargo bench                # Run all benchmarks

# Benchmark specific indicator (with nightly-avx for SIMD optimizations)
cargo bench --features nightly-avx --bench indicator_benchmark -- maaq
cargo bench --features nightly-avx --bench indicator_benchmark -- rsi
```

### Linting & Formatting
```bash
cargo clippy              # Run Rust linter
cargo fmt                 # Format code
cargo audit               # Security audit (from GitHub Actions)
```

## Architecture

### Module Structure
- `src/indicators/` - Main indicator implementations (178 files)
  - `moving_averages/` - Specialized moving average indicators (30+ types)
  - Each indicator is self-contained with error handling, documentation, and SIMD optimization
- `src/utilities/` - Shared utilities
  - `data_loader.rs` - CSV data loading and source type handling
  - `math_functions.rs` - Mathematical operations
  - `enums.rs` - Kernel selection for SIMD optimization
  - `helpers.rs` - SIMD kernel detection and batch processing
  - `aligned_vector.rs` - Cache-aligned memory for SIMD

### Key Design Patterns

1. **SIMD Optimization**: Indicators use runtime kernel detection to select optimal SIMD instructions (AVX512, AVX2, SSE2, or scalar)
2. **Batch Processing**: Many indicators support batch operations for processing multiple securities
3. **Error Handling**: Each indicator has its own error type with specific error cases
4. **Input Flexibility**: Indicators accept either raw slices or Candles with source selection (open/high/low/close)

### Performance Features
- Cache-aligned vectors for SIMD operations
- Rayon parallel processing support
- Feature flags for nightly AVX optimizations
- Proptest for property-based testing
- Criterion benchmarking suite

### Third-Party Dependencies
- SLEEF (SIMD Library for Evaluating Elementary Functions) in `third_party/sleef/`
- Used for optimized mathematical operations

## Website Documentation

A separate Astro-based website exists in `website/` for showcasing indicators with interactive charts. It has its own build system and CLAUDE.md.

## Testing Individual Indicators

### Rust Tests
To test a specific indicator with nightly-avx features:
```bash
cargo test --features nightly-avx --lib indicators::<indicator_name> -- --nocapture
```

Examples:
```bash
cargo test --features nightly-avx --lib indicators::rsi -- --nocapture
cargo test --features nightly-avx --lib indicators::moving_averages::ema -- --nocapture
cargo test --features nightly-avx --lib indicators::bollinger_bands -- --nocapture
```

### Python and WASM Binding Tests
Test bindings for all indicators:
```bash
./test_bindings.sh              # Run all Python and WASM tests
./test_bindings.sh alma         # Test only ALMA indicator
./test_bindings.sh --python     # Run only Python tests
./test_bindings.sh --wasm       # Run only WASM tests
```

Generate test files for new indicators:
```bash
python scripts/generate_binding_tests.py <indicator_name>
```

## Adding New Indicators

1. Create new file in `src/indicators/` (or `src/indicators/moving_averages/` for MAs)
2. Follow existing patterns for error types, input handling, and SIMD optimization
3. Add module export in `src/indicators/mod.rs`
4. Include comprehensive documentation with parameters and error cases
5. Implement tests and benchmarks
6. Update indicator count in README.md

## SIMD Kernel Selection

The codebase automatically selects the best SIMD instruction set at runtime:
- `Kernel::Avx512` - Latest Intel/AMD processors (requires `--features nightly-avx`)
- `Kernel::Avx2` - Modern processors (requires `--features nightly-avx`)
- `Kernel::Sse2` - Older x86 processors  
- `Kernel::Scalar` - Fallback for compatibility

Use `detect_best_kernel()` or `detect_best_batch_kernel()` helpers.

**Note**: AVX2 and AVX512 kernels are only available when building with nightly Rust and the `nightly-avx` feature flag.