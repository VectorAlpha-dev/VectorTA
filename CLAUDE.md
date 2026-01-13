# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Environment Note: You are running in a Windows PowerShell environment via Git Bash. The Bash tool executes Windows commands directly. Use Windows batch files without ./ prefix (e.g., `test_bindings.bat alma` not `./test_bindings.bat alma`).**

**Note: All commands in this file are for Windows environments. Use Command Prompt or PowerShell.**

**Formatting Policy: Do NOT run `rustfmt` or `cargo fmt` automatically. Avoid any auto-formatting that could reorder imports or rewrite code, as it tends to cause merge conflicts. Only format if explicitly requested for a specific file.**

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

# Test Python and WASM bindings (Windows):
test_bindings.bat indicator_name

# Or test Python and WASM bindings separately:
# For Python (first time setup):
# If you get virtualenv errors, delete and recreate it:
rmdir /s /q .venv
python -m venv .venv
.venv\Scripts\activate
pip install maturin pytest pytest-xdist numpy
maturin develop --features python --release
python tests\python\run_all_tests.py indicator_name

# For WASM (first time setup):
cargo install wasm-pack
wasm-pack build --target nodejs --features wasm
cd tests\wasm && npm test -- indicator_name
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
# Windows:
test_bindings.bat               # Run all Python and WASM tests
test_bindings.bat alma          # Test only ALMA indicator
test_bindings.bat --python      # Run only Python tests
test_bindings.bat --wasm        # Run only WASM tests
```

Generate test files for new indicators:
```bash
python scripts\generate_binding_tests.py <indicator_name>
```

### Testing Commands Reference (Windows)

**IMPORTANT**: Use these exact commands on Windows to avoid cross-platform issues.

#### Rust Unit Tests (Always Works)
```bash
# Test specific indicator
cargo test --features nightly-avx --lib indicators::indicator_name -- --nocapture

# For moving averages:
cargo test --features nightly-avx --lib indicators::moving_averages::indicator_name -- --nocapture

# Examples that work:
cargo test --features nightly-avx --lib indicators::moving_averages::reflex -- --nocapture
cargo test --features nightly-avx --lib indicators::rsi -- --nocapture
```

#### Python Binding Tests
```bash
# First, ensure Python bindings are built:
python -m pip install maturin pytest numpy --user --quiet
python -m maturin develop --features python --release

# Then run tests (use forward slashes):
python tests/python/test_indicator_name.py

# Note: If module not found error occurs, the bindings are likely installed in venv
# but you're using system Python. Use test_bindings.bat instead.
```

#### WASM Binding Tests (Recommended Method)
```bash
# Navigate to WASM test directory first
cd tests\wasm

# Run specific indicator test
npm test -- test_indicator_name.js

# Or run directly with node
node --test test_indicator_name.js

# Examples that work:
cd tests\wasm && npm test -- test_reflex.js
cd tests\wasm && node --test test_alma.js
```

#### Using test_bindings.bat (Currently Unreliable)
```bash
# This should work but has environment issues:
test_bindings.bat indicator_name

# If it fails, use the individual methods above
```

### Common Issues and Solutions

1. **Path Issues**: Use forward slashes in paths when possible
2. **Python Environment**: If `ModuleNotFoundError`, ensure you're using the same Python that maturin used
3. **Command Not Found**: Don't use Unix commands (like `call`) in Git Bash on Windows
4. **Backslash Issues**: Avoid `.venv\Scripts\python.exe` - the backslashes cause problems

## Adding New Indicators

1. Create new file in `src/indicators/` (or `src/indicators/moving_averages/` for MAs)
2. **MANDATORY**: Follow the "Indicator Development Standards" section below exactly
3. Add module export in `src/indicators/mod.rs`:
   ```rust
   pub mod indicator_name;
   pub use indicator_name::{indicator_name, IndicatorNameInput, IndicatorNameOutput, IndicatorNameParams};
   ```
4. Register bindings:
   - Python: Add to `src/python.rs`
   - WASM: Add to `src/wasm.rs`
   - Benchmarks: Add to `benches/indicator_benchmark.rs` and `benchmarks/criterion_comparable_benchmark.py`
5. Generate binding tests: `python scripts\generate_binding_tests.py indicator_name`
6. Update indicator count in README.md

## SIMD Kernel Selection

The codebase automatically selects the best SIMD instruction set at runtime:
- `Kernel::Avx512` - Latest Intel/AMD processors (requires `--features nightly-avx`)
- `Kernel::Avx2` - Modern processors (requires `--features nightly-avx`)
- `Kernel::Sse2` - Older x86 processors
- `Kernel::Scalar` - Fallback for compatibility

Use `detect_best_kernel()` or `detect_best_batch_kernel()` helpers.

**Note**: AVX2 and AVX512 kernels are only available when building with nightly Rust and the `nightly-avx` feature flag.

## CUDA Development

The build script compiles every `kernels/cuda/**/*.cu` to PTX and embeds it into the crate (when `--features cuda` is enabled).

### Key Facts
- PTX target: uses `-arch=compute_XX` for `-ptx`. Default is `compute_89` (RTX 4090). If unsupported by your nvcc, it auto-falls back to `compute_80`.
- No runtime linkage to `cudart` — only PTX is generated and loaded via `cust::Module::from_ptx`.
- Windows: easiest is the "x64 Native Tools Command Prompt for VS", but the build also attempts to auto-detect MSVC and pass `-ccbin`.

### Environment Variables
- `CUDA_ARCH`: preferred arch (examples: `89`, `8.9`, `sm_89`, `compute_89`). Defaults to `compute_89`.
- `CUDA_ARCHS`: space/comma separated; first non-empty token is used.
- `NVCC`: path to `nvcc` (overrides autodiscovery).
- `NVCC_ARGS`: extra flags passed to `nvcc` (e.g., `"--keep --verbose"`).
- `CUDA_DEBUG=1`: add `-lineinfo` to PTX for easier debugging.
- `CUDA_FAST_MATH=0|1`: globally disable/enable fast-math (some kernels opt out by default when numerically sensitive).
- `CUDA_FILTER`: only compile kernels whose path contains any of these substrings (comma/space separated). Example: `CUDA_FILTER=alma,ema`.
- `CUDA_KERNEL_DIR`: override kernels root (default `kernels/cuda`).
- `CUDA_PLACEHOLDER_ON_FAIL=1`: if `nvcc` fails for a kernel, emit a minimal placeholder PTX so the crate still builds. Only use for local iteration — wrappers using placeholders will fail at runtime.

### Common Workflows
```bash
# Build all CUDA PTX + crate
cargo build --features cuda

# Old nvcc toolkits (<= 11.x) with modern GPUs
CUDA_ARCH=80 cargo build --features cuda

# Build a subset of kernels
CUDA_FILTER=alma,ema cargo build --features cuda

# Verbose debug PTX
CUDA_DEBUG=1 NVCC_ARGS="--keep --verbose" cargo build --features cuda
```

### Running CUDA Tests
```bash
# All CUDA tests
cargo test --features cuda -- --nocapture

# Single test file
cargo test --features cuda --test sma_cuda -- --nocapture

# Single test by name
cargo test --features cuda sma_cuda_one_series_many_params_matches_cpu -- --nocapture
```
Tests skip gracefully if no CUDA device is present (tests call `cuda_available()`).

## CRITICAL: Debugging and Code Modification Guidelines

### Never Use Mass Editing Scripts

**NEVER use mass editing scripts on Rust files**. Previous attempts to fix compilation errors using automated scripts across 160+ files introduced significant syntax errors, particularly with:
- Incorrect bracket placement in if-else blocks
- Malformed `#[cfg()]` conditional compilation blocks
- Context-unaware pattern matching that broke valid code

### Debugging Best Practices

**"Think harder" during debugging - work step by step:**

1. **Understand Before Fixing**
   - Read the FULL error message carefully
   - Use the Read tool to examine the complete context
   - Trace through the code logic before making changes
   - Consider why the original code was written that way

2. **Step-by-Step Approach**
   - Fix ONE issue at a time
   - Test after EACH change
   - Don't assume similar-looking code has the same problem
   - Verify your fix doesn't break other functionality

3. **Common Debugging Patterns**
   ```bash
   # Test a specific indicator after changes
   cargo test --features nightly-avx --lib indicators::indicator_name -- --nocapture

   # Check if your changes compile
   cargo check --features nightly-avx

   # Run clippy to catch potential issues
   cargo clippy --features nightly-avx
   ```

4. **When Dealing with SIMD Code**
   - Test scalar version first
   - Then test AVX2/AVX512 versions separately
   - Ensure all kernels produce identical results
   - Remember: AVX code requires `--features nightly-avx`

5. **Avoid Shortcuts**
   - Don't copy-paste fixes between files without understanding context
   - Don't use regex/find-replace across multiple files
   - Don't assume one fix applies everywhere
   - Each indicator may have unique requirements

**Remember**: Taking time to understand the problem thoroughly saves time in the long run. A well-thought-out fix is better than a quick fix that introduces new bugs.

This is especially important when dealing with:
- WASM conditional compilation
- Cross-cutting concerns affecting multiple files
- SIMD kernel implementations
- Platform-specific code

## Indicator Development Quality Standards

**IMPORTANT**: While indicators vary widely in implementation, all must meet these minimum quality standards. ALMA.rs (`src/indicators/moving_averages/alma.rs`) serves as a reference implementation demonstrating best practices.

### MANDATORY Requirements

These are non-negotiable requirements for ALL indicators:

#### 1. Zero Memory Copy Operations

**CRITICAL: For OUTPUT vectors matching input data length, ALWAYS use helper functions:**

```rust
// ✅ CORRECT - For output vectors:
let mut out = alloc_with_nan_prefix(data.len(), warmup_period);

// ❌ WRONG - NEVER use these patterns for output:
let mut out = vec![f64::NAN; data.len()];  // NEVER for output!
let mut out = Vec::with_capacity(data.len());
out.resize(data.len(), f64::NAN);
```

**For batch/matrix operations:**
```rust
// Use these three functions together:
let mut buf_mu = make_uninit_matrix(rows, cols);
init_matrix_prefixes(&mut buf_mu, cols, &warmup_periods);
// Then convert to mutable slice for computation
```

**NUANCE: Small intermediate vectors are acceptable:**
```rust
// ✅ OK - Small weight/coefficient vectors (size << data.len()):
let mut weights = AVec::with_capacity(CACHELINE_ALIGN, period);  // OK if period << data.len()
let mut sqrt_diffs = vec![0.0; period];  // OK for small period
let coefficients = vec![0.1, 0.2, 0.3, 0.4];  // OK for constants

// ❌ WRONG - Output or data-sized vectors:
let mut temp_data = vec![0.0; data.len()];  // NEVER!
let mut ln_values = Vec::with_capacity(data.len());  // NEVER!
```

**Rule of thumb**: If the vector size is proportional to the input data length (O(n)), use the uninitialized memory helpers. If it's a small fixed size or proportional to a parameter like period (typically O(1) or O(period) where period << n), regular Vec or AVec is acceptable.

#### 2. Binding Requirements

**Python Bindings:**
- Add function to `src/python.rs`
- Add to benchmark file: `benchmarks/criterion_comparable_benchmark.py`
- Ensure test exists: `tests/python/test_indicator_name.py`

**WASM Bindings:**
- Add function to `src/wasm.rs` (if applicable)
- Ensure test exists: `tests/wasm/test_indicator_name.js`

**Benchmark Integration:**
- Add to `benches/indicator_benchmark.rs`
- Include both single and batch versions if applicable

#### 3. Error Handling

All indicators must handle these error cases appropriately:
- Empty input data
- All NaN values
- Invalid parameters (e.g., period > data length)
- Insufficient data for calculation

#### 4. Testing Requirements

Minimum test coverage:
- Basic functionality test
- Edge cases (empty data, all NaN)
- Parameter validation
- Accuracy verification against known values

### RECOMMENDED Guidelines

These patterns from ALMA.rs are recommended but can be adapted based on indicator requirements:

#### API Design Suggestions

Consider using these patterns where appropriate:

```rust
// Input structure with flexibility for candles or raw slices
pub struct IndicatorInput<'a> {
    pub data: IndicatorData<'a>,
    pub params: IndicatorParams,
}

// Builder pattern for ergonomic API
pub struct IndicatorBuilder {
    // parameters
}

// Streaming support for real-time updates
pub struct IndicatorStream {
    // state management
}
```

#### SIMD Optimization

**Goal**: Implement AVX2/AVX512 kernels that are functionally identical to scalar and measurably faster. Success means all unit tests pass and AVX2/AVX512 benchmarks beat scalar.

**Scalar-First Optimization (Required Before SIMD)**:
- Optimize the scalar implementation first; SIMD comes only after scalar is as fast and clean as possible
- If the scalar indicator calls helpers in hot loops, prefer loop-jamming/inlining to eliminate call overhead — except when helpers involve dynamic dispatch (e.g., "MA" selector)
- Every scalar optimization must be validated with unit tests and benchmark runs
- Unsafe in scalar is allowed if SIMD kernels are stubs/disabled and the change provides a clear, measured benefit

**Guarding and Targets**:
- Gate all SIMD with `#[cfg(feature = "nightly-avx")]` and `#[cfg(target_arch = "x86_64")]`
- Keep scalar codepath as the reference implementation; SIMD must match outputs bit-for-bit or within existing tolerances

**Kernel Selection**:
- Use `detect_best_kernel()` / `detect_best_batch_kernel()` to pick AVX512 → AVX2 → SSE2 → Scalar at runtime
- Use `skip_if_unsupported!(kernel, fn_name)` in tests/benches to conditionally skip where HW support is missing

**Benchmark Registration (Required Before SIMD)**:
- Ensure the indicator is registered in `benches/indicator_benchmark.rs` so scalar vs AVX2/AVX512 comparisons can run
- Discover available IDs: `cargo bench --bench indicator_benchmark -- --list`

**SIMD Tests (Must Pass)**:
```bash
# Scalar
cargo test --lib indicators::<module>::<indicator> -- --nocapture

# Nightly + SIMD
cargo +nightly test --features nightly-avx --lib indicators::<module>::<indicator> -- --nocapture
```

**SIMD Benchmarks**:
```bash
# Generic form for any indicator at 100k candles
cargo bench --features nightly-avx --bench indicator_benchmark -- <ind>/<ind>_scalar/100k
cargo bench --features nightly-avx --bench indicator_benchmark -- <ind>/<ind>_avx2/100k
cargo bench --features nightly-avx --bench indicator_benchmark -- <ind>/<ind>_avx512/100k
```

**Acceptance Criteria for SIMD PRs**:
- Stable build passes (`cargo build`)
- Nightly + `nightly-avx` tests pass for the indicator(s) touched
- Benchmarks show >5% improvement of AVX2/AVX512 vs scalar at realistic sizes (e.g., 100k)

**When SIMD Underperforms or Is Unstable**:
- If AVX2/AVX512 is slower than scalar after repeated attempts, keep kernels as stubs delegating to scalar
- Short-circuit runtime selection to scalar while leaving SIMD code present for future work
- Add a brief module-level comment explaining why SIMD is disabled (e.g., memory-bound, branch-heavy, short windows)
- Do NOT change unit test reference values — fix the implementation instead

#### Performance Tips

- Use `AVec<f64>` for cache-aligned SIMD operations
- Calculate warmup period once and reuse
- Consider batch operations for parameter sweeps

#### Row-Specific Batch Kernels (SIMD)

Row-specific kernels are specialized SIMD implementations for batch functions (many parameter combinations across rows) where each row corresponds to a parameter set. Only attempt when there is a clear opportunity to share precomputed data.

**When to Attempt**:
- There is shared precomputation reusable across parameter rows (e.g., Gaussian weights and normalization for ALMA)
- Inner loops can avoid per-row recomputation of identical terms, or enable better cache locality
- Memory access can be kept contiguous and aligned (SoA layout, transposed buffers, or tiling)

**Requirements**:
- Accuracy: outputs must match scalar/baseline (tests unchanged)
- Performance: must be >5% faster than non row-specific batch kernel at realistic sizes
- Feature-gating: keep under `nightly-avx`; use runtime batch-kernel selector to opt into AVX2/AVX512 only when beneficial

**Implementation Guidance**:
- Use `alma.rs` as the successful reference for structure, docs, and optimization patterns
- Precompute once per tile or per period: weights, norms, constants, prefix sums, etc., and share across rows
- Hoist invariants outside inner loops; prefer FMA; reduce branches
- Keep outputs and working sets contiguous and aligned
- Isolate intrinsics in small functions; annotate safety and alignment assumptions

**AVX512 Short/Long Period Variants**:
- Only introduce separate "short" and "long" period AVX512 kernels when there is demonstrated, clear benefit (as in ALMA)
- AVX2 typically uses a single kernel unless profiling shows a similar split helps

### Quality Checklist

Before completing an indicator:
- [ ] **MANDATORY**: Uses zero-copy memory operations
- [ ] **MANDATORY**: Python bindings added and tested
- [ ] **MANDATORY**: WASM bindings added (if applicable)
- [ ] **MANDATORY**: Added to benchmark files
- [ ] **MANDATORY**: Handles all error cases
- [ ] **MANDATORY**: Has minimum test coverage
- [ ] Implements SIMD optimizations where beneficial
- [ ] Documentation explains all parameters
- [ ] Follows consistent naming conventions

**Note**: While implementation details may vary, maintaining consistent quality standards and optimization practices ensures the library remains performant and user-friendly.

## Decision Logging

Add a brief note at the top of each indicator module and/or SIMD submodule summarizing the status and rationale. Examples:
- "SIMD enabled because …; fastest on AVX2/AVX512 by >5% at 100k."
- "SIMD implemented but disabled by default: memory-bound/branch-heavy; slower than scalar. Revisit after layout change X."
- "Row-specific batch kernels not attempted; no shared precompute to exploit."

Keep notes concise (1–3 lines). This helps future contributors avoid churn when SIMD or row-specific kernels are not promising.

## CI & PRs

### CI Gate
- `cargo build` (stable) must pass
- Include nightly AVX tests when touching SIMD code
- Do not rely on `CUDA_PLACEHOLDER_ON_FAIL` in CI — placeholder PTX is for local iteration only

### Commit & PR Guidelines
- Use Conventional Commits (e.g., `feat(indicators): add alma warmup`)
- PRs should include:
  - Description of changes
  - Linked issues (if applicable)
  - Affected indicators/kernels
  - Test notes
  - Benchmark deltas with commands and environment

### Constraints
- Do NOT change unit test reference values — fix the implementation instead
- Keep changes minimal and focused; match surrounding style
- Document public items with `///` (parameters and error cases in natural language)
- Prefer `once_cell` over `lazy_static`
- Feature-gate appropriately: `wasm`, `python`, `nightly-avx`, `cuda`

## Troubleshooting

### CUDA Issues
- **`nvcc` not found**: Set `NVCC` or `CUDA_PATH`/`CUDA_HOME`, or ensure it's on PATH
- **Arch unsupported by nvcc**: Set `CUDA_ARCH=80` or upgrade CUDA Toolkit; build.rs will also retry with `compute_80` automatically
- **Only some kernels fail to compile**: Use `CUDA_FILTER` to iterate faster; optionally set `CUDA_PLACEHOLDER_ON_FAIL=1` locally to unblock building other tests
- **Cross-crate PTX includes**: Remember `OUT_DIR` is per-crate; compile PTX in that crate's build.rs if needed

### Windows-Specific
- Recommended: open the "x64 Native Tools Command Prompt for VS" to ensure `cl.exe` and Windows SDK are found by `nvcc`
- Alternative: rely on build.rs auto-detection of MSVC and automatic `-ccbin` setting. Ensure VS C++ Build Tools and SDK are installed