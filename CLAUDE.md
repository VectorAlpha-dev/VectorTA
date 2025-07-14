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

### Quick Testing Commands
For any indicator, use these two commands:
```bash
# Test Rust implementation (replace 'indicator_name' with actual indicator)
cargo test --features nightly-avx --lib indicators::moving_averages::indicator_name -- --nocapture

# Test Python and WASM bindings (Linux/Mac/WSL)
./test_bindings.sh indicator_name

# Test Python and WASM bindings (Windows native - run separately):
# For Python (first time setup):
# If you get virtualenv errors, delete and recreate it:
rmdir /s /q .venv
python -m venv .venv
.venv\Scripts\activate
pip install maturin pytest pytest-xdist numpy
maturin develop --features python --release
python tests/python/run_all_tests.py indicator_name

# For WASM (first time setup):
cargo install wasm-pack
wasm-pack build --target nodejs --features wasm
cd tests/wasm && npm test -- indicator_name
```

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

Note: For moving averages, use the full path: `indicators::moving_averages::indicator_name`

### Python and WASM Binding Tests
Test bindings for all indicators:
```bash
./test_bindings.sh              # Run all Python and WASM tests
./test_bindings.sh alma         # Test only ALMA indicator
./test_bindings.sh --python     # Run only Python tests
./test_bindings.sh --wasm       # Run only WASM tests

# Windows Native:
test_bindings.bat indicator_name  # Test both Python and WASM bindings for specific indicator
```

Generate test files for new indicators:
```bash
python scripts/generate_binding_tests.py <indicator_name>
```

### Testing Commands Reference
```bash
# Test Rust unit tests for specific indicator
cargo test --features nightly-avx --lib indicators::indicator_name -- --nocapture
# For moving averages:
cargo test --features nightly-avx --lib indicators::moving_averages::indicator_name -- --nocapture

# Run benchmarks for specific indicator  
cargo bench --features nightly-avx --bench indicator_benchmark -- indicator_name

# Test Python and WASM bindings (Windows)
test_bindings.bat indicator_name
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

## CRITICAL: Mass Editing Rust Files

**NEVER use mass editing scripts on Rust files**. Previous attempts to fix compilation errors using automated scripts across 160+ files introduced significant syntax errors, particularly with:
- Incorrect bracket placement in if-else blocks
- Malformed `#[cfg()]` conditional compilation blocks
- Context-unaware pattern matching that broke valid code

**Always make manual, targeted fixes** to Rust files:
- Fix compilation errors one file at a time
- Review each change in context before applying
- Preserve the existing logic of scalar, AVX2, and AVX512 kernels
- Use the Read tool to understand the full context before making changes

This is especially important when dealing with WASM conditional compilation or any cross-cutting concerns that affect multiple files.