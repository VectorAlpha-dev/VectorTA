# Agents Guide (Authoritative)

This document is the authoritative reference for working in this repository. It consolidates build/run instructions, development conventions, and gotchas for CUDA, SIMD, WASM, and Python bindings. Do not change unit test reference values — they are the source of truth. Fix root causes instead.

## Structure

- `src/`: Core library. Indicators under `src/indicators/`. Use `src/indicators/moving_averages/alma.rs` as the reference for API, docs, style, and performance patterns.
- `src/cuda/`: CUDA wrappers and helpers (feature-gated by `cuda`). PTX is embedded at compile time via `include_str!(concat!(env!("OUT_DIR"), "/<name>.ptx"))`.
- `kernels/`: Architecture-specific code (currently CUDA `.cu` sources under `kernels/cuda/**`).
- `tests/`: Integration tests per indicator (keep scope tight to module under test; many files named `*_cuda.rs`).
- `benches/`: Criterion benchmarks (`indicator_benchmark`, `cuda_bench`).
- Demos (`gpu_backtester_demo`, `optimizer_demo`): not part of the main build. Safe to leave; not maintained actively.

## Toolchains

- Rust stable required to pass: `cargo build`.
- Nightly for SIMD: `cargo +nightly build --features nightly-avx` (AVX2/AVX512).
- Toolchain file: `rust-toolchain.toml` (pinned nightly for formatting; stable fallback OK).

## Build & Check

- Quick checks: `cargo check` (add `--features wasm|python|nightly-avx|cuda` as needed).
- Format: `cargo +nightly fmt`.
- Lint: `cargo clippy --all-targets --all-features -D warnings`.

## CUDA Development

The build script compiles every `kernels/cuda/**/*.cu` to PTX and embeds it into the crate (when `--features cuda` is enabled).

Key facts:
- PTX target: uses `-arch=compute_XX` for `-ptx`. Default is `compute_89` (RTX 4090). If unsupported by your nvcc, it auto-falls back to `compute_80`.
- No runtime linkage to `cudart` — only PTX is generated and loaded via `cust::Module::from_ptx`.
- Windows: easiest is the “x64 Native Tools Command Prompt for VS”, but the build also attempts to auto-detect MSVC and pass `-ccbin`.

Environment toggles (supported by build.rs):
- `CUDA_ARCH`: preferred arch (examples: `89`, `8.9`, `sm_89`, `compute_89`). Defaults to `compute_89`.
- `CUDA_ARCHS`: space/comma separated; first non-empty token is used.
- `NVCC`: path to `nvcc` (overrides autodiscovery).
- `NVCC_ARGS`: extra flags passed to `nvcc` (e.g., `"--keep --verbose"`).
- `CUDA_DEBUG=1`: add `-lineinfo` to PTX for easier debugging.
- `CUDA_FAST_MATH=0|1`: globally disable/enable fast-math (some kernels opt out by default when numerically sensitive).
- `CUDA_FILTER`: only compile kernels whose path contains any of these substrings (comma/space separated). Example: `CUDA_FILTER=alma,ema`.
- `CUDA_KERNEL_DIR`: override kernels root (default `kernels/cuda`).
- `CUDA_PLACEHOLDER_ON_FAIL=1`: if `nvcc` fails for a kernel, emit a minimal placeholder PTX so the crate still builds. Only use for local iteration — wrappers using placeholders will fail at runtime when functions are missing.

Common workflows:
- Build all CUDA PTX + crate: `cargo build --features cuda`
- Old nvcc toolkits (<= 11.x) with modern GPUs: `CUDA_ARCH=80 cargo build --features cuda`
- Build a subset of kernels: `CUDA_FILTER=alma,ema cargo build --features cuda`
- Verbose debug PTX: `CUDA_DEBUG=1 NVCC_ARGS="--keep --verbose" cargo build --features cuda`

Running CUDA tests:
- All CUDA tests: `cargo test --features cuda -- --nocapture`
- Single test file: `cargo test --features cuda --test sma_cuda -- --nocapture`
- Single test by name: `cargo test --features cuda sma_cuda_one_series_many_params_matches_cpu -- --nocapture`
- Skips gracefully if no CUDA device is present (tests call `cuda_available()`).

Cross-crate demos:
- `gpu_backtester_demo` embeds `double_crossover.ptx` from its own `OUT_DIR`. It is not part of the root build; ignore unless developing that demo. If you need it, add a local `build.rs` in that crate to compile its PTX.

## SIMD Development

Goal: implement AVX2/AVX512 kernels that are functionally identical to scalar and measurably faster. Success means all unit tests pass and AVX2/AVX512 benchmarks beat scalar. Do not benchmark batch mode unless you implemented per-row optimized kernels for the batch function.

- Guarding and targets
  - Gate all SIMD with `#[cfg(feature = "nightly-avx")]` and `#[cfg(target_arch = "x86_64")]`.
  - Keep scalar codepath as the reference implementation; SIMD must match outputs bit-for-bit or within existing tolerances.

- Kernel selection
  - Use existing helpers `detect_best_kernel()` / `detect_best_batch_kernel()` to pick AVX512 → AVX2 → SSE2 → Scalar at runtime.
  - Use `skip_if_unsupported!(kernel, fn_name)` in tests/benches to conditionally skip where HW support is missing.

- API and performance patterns
  - Treat `src/indicators/moving_averages/alma.rs` as the gold standard for API shape, docs, warmup handling, allocation, and optimization.
  - Output buffers: use uninitialized or helper-based allocation with warmup prefixes; never allocate O(n) temporaries for outputs.
  - Use cache-aligned vectors where beneficial (e.g., `aligned-vec`, AVec patterns).
  - Minimize `unsafe`; isolate intrinsics inside small, well-named functions.

### Scalar-first optimization (required before SIMD)
- Optimize the scalar implementation of the indicator first; SIMD comes only after the scalar path is as fast and clean as possible.
- If the scalar indicator calls other helpers in the hot loop, prefer loop-jamming/inlining those helpers into the indicator loop to eliminate call overhead and enable hoisting invariants — except when the helper involves dynamic dispatch (e.g., the "MA" selector), in which case keep the selector at a higher level and avoid inlining dynamic branches into hot loops.
- Every scalar optimization/change must be validated with unit tests and a benchmark run for that indicator. Use the existing commands in this guide (scalar tests and Criterion benches) and record the deltas in your PR description.
- Keep the scalar path safe: do not introduce `unsafe` into the scalar implementation if it is currently safe. Only use `unsafe` inside tightly scoped SIMD/intrinsics blocks later.

### Benchmark registration (required before SIMD)
- Ensure the Rust indicator is registered in `benches/indicator_benchmark.rs` so its scalar path has a baseline benchmark before adding SIMD.
- Verify that both the single-series and (if applicable) batch variants are wired into the Criterion groups with stable IDs. Discover IDs with: `cargo bench --bench indicator_benchmark -- --list`.
- If an indicator isn’t registered, add its entries following the existing style and naming so scalar vs AVX2/AVX512 comparisons are apples-to-apples.

- Tests (must pass)
  - Scalar: `cargo test --lib indicators::<module>::<indicator> -- --nocapture`
  - Nightly + SIMD: `cargo +nightly test --features nightly-avx --lib indicators::<module>::<indicator> -- --nocapture`
  - Optional AVX512-only scope: `cargo +nightly test --features nightly-avx --lib indicators::_avx512 -- --nocapture`

- Benchmarks (single-series only unless batch is per-row optimized)
  - Use the Criterion harness `indicator_benchmark` and compare scalar vs AVX2 vs AVX512.
  - Generic form for any indicator `<ind>` at 100k candles:
    - `cargo bench --features nightly-avx --bench indicator_benchmark -- <ind>/<ind>_scalar/100k`
    - `cargo bench --features nightly-avx --bench indicator_benchmark -- <ind>/<ind>_avx2/100k`
    - `cargo bench --features nightly-avx --bench indicator_benchmark -- <ind>/<ind>_avx512/100k`
  - Discover available IDs: `cargo bench --bench indicator_benchmark -- --list`.
  - Batch benches are auto-registered via macros for all indicators. Treat batch results as advisory only until per-row optimized batch kernels exist; do not use batch benches as acceptance criteria.

- Acceptance criteria for SIMD PRs
  - Stable build passes (`cargo build`).
  - Nightly + `nightly-avx` tests pass for the indicator(s) touched.
  - Benchmarks show >5% improvement of AVX2/AVX512 vs scalar at realistic sizes (e.g., 100k). Include commands used; e.g., `RUSTFLAGS="-C target-cpu=native"`.

- When SIMD underperforms or is unstable
  - Definition of failure: after repeated attempts you still have unresolved SIMD accuracy issues and/or consistent slower performance than scalar (beyond run-to-run noise).
  - If you found the indicator with AVX2/AVX512 kernels as stubs delegating to the scalar path, keep them as stubs (do not force-enable a slower/incorrect SIMD path).
  - Prefer disabling runtime selection for the SIMD path for that indicator (short-circuit to scalar) while leaving the SIMD code present for future work.
  - Add a brief module-level comment explaining why SIMD is disabled (e.g., memory-bound, branch-heavy, short windows, architecture downclock). Do not change unit test reference values.

## Row-Specific Batch Kernels (SIMD)

Row-specific kernels are specialized SIMD implementations for batch functions (many parameter combinations across rows) where each row corresponds to a parameter set. They should only be attempted when there is a clear opportunity to share precomputed data and avoid redundant work between rows.

- When to attempt
  - There is shared precomputation reusable across parameter rows (e.g., Gaussian weights and normalization for ALMA).
  - The inner loops can be structured to avoid per-row recomputation of identical terms, or enable better cache locality and fewer branches.
  - You can keep memory access contiguous and aligned (SoA layout, transposed buffers, or tiling), enabling efficient vector loads/stores.

- Requirements (identical to single-series SIMD)
  - Accuracy: outputs must match scalar/baseline (tests unchanged).
  - Performance: must be >5% faster than the non row-specific batch kernel at realistic sizes.
  - Feature-gating and selection: keep under `nightly-avx`; use the runtime batch-kernel selector to opt into AVX2/AVX512 only when beneficial.
  - Do not benchmark batch variants unless per-row kernels are implemented and wired in.

- Implementation guidance
  - Use `alma.rs` as the successful reference for structure, docs, and optimization patterns.
  - Precompute once per tile or per period: weights, norms, constants, prefix sums, etc., and share them across rows.
  - Hoist invariants outside inner loops; prefer FMA; reduce branches; consider short/long loop specializations to keep hot loops tight.
  - Keep outputs and working sets contiguous and aligned; prefer time-major or parameter-major layouts that minimize striding.
  - Isolate intrinsics in small functions; annotate safety and alignment assumptions.

- AVX512 short/long period variants
  - Only introduce separate “short” and “long” period AVX512 kernels when there is a demonstrated, clear benefit (as in ALMA). Avoid complexity without measurable wins.
  - AVX2 typically uses a single kernel unless profiling shows a similar split helps.

- Benchmarks for batch row-specific kernels
  - Batch variants are registered for all indicators by macros. Only treat these results as acceptance once row-specific kernels are implemented and wired in; otherwise, consider them informational/advisory.
  - Discover exact IDs with: `cargo bench --bench indicator_benchmark -- --list`.
  - As with single-series, run with `RUSTFLAGS="-C target-cpu=native"` and test multiple sizes (10k, 100k).

- When batch SIMD underperforms or is unstable
  - Same policy as single-series: keep AVX2/AVX512 batch kernels as stubs delegating to scalar batch if that’s how you found them; short-circuit selection to scalar batch; add a brief comment explaining why SIMD is disabled for now. Do not change reference outputs.

## Decision Logging

- Add a brief note at the top of each indicator module and/or SIMD submodule summarizing the status and rationale, for example:
  - “SIMD enabled because …; fastest on AVX2/AVX512 by >5% at 100k.”
  - “SIMD implemented but disabled by default: memory-bound/branch-heavy; slower than scalar. Revisit after layout change X.”
  - “Row-specific batch kernels not attempted; no shared precompute to exploit.”
- Keep the note concise (1–3 lines). This helps future contributors avoid churn when SIMD or row-specific kernels are not promising.

## Testing & Benchmarks

- Scalar-only tests: `cargo test --lib indicators:: -- --nocapture`.
- SIMD tests (nightly): `cargo +nightly test --features nightly-avx`.
- CUDA tests: see CUDA section above.
- Benches: `RUSTFLAGS="-C target-cpu=native" cargo bench --bench indicator_benchmark --`
- GPU benches: `cargo bench --bench cuda_bench --features cuda --` (ensure a device is available).

## Conventions & Constraints

- Do not change unit test reference values. Fix the implementation.
- Keep changes minimal and focused; match surrounding style.
- Document public items with `///` (parameters and error cases in natural language).
- Prefer `once_cell` over `lazy_static`.
- Use `assert_same_len!(a, b)` and similar helpers where applicable.
- For outputs of length N, use zero-copy/uninitialized allocation helpers (see `alma.rs` comments in CLAUDE.md). Avoid allocating O(N) temporaries.
- Feature-gate appropriately: `wasm`, `python`, `nightly-avx`, `cuda`.

## Windows-specific Notes

- Recommended: open the “x64 Native Tools Command Prompt for VS” to ensure `cl.exe` and Windows SDK are found by `nvcc`.
- Alternative: rely on build.rs auto-detection of MSVC and automatic `-ccbin` setting. Ensure VS C++ Build Tools and SDK are installed.

## CI & PRs

- CI gate: `cargo build` (stable) must pass.
- Include nightly AVX tests when touching SIMD code.
- Use Conventional Commits (e.g., `feat(indicators): add alma warmup`).
- PRs should include description, linked issues, affected indicators/kernels, test notes, and benchmark deltas with commands and environment.

### CI Guardrails

- Do not rely on `CUDA_PLACEHOLDER_ON_FAIL` in CI. Placeholder PTX is for local iteration only; CI should fail if any kernel PTX fails to compile.
- For indicators where SIMD/row-specific kernels are disabled due to underperformance, ensure runtime selection short-circuits to scalar in release builds, and document the rationale in the module.

## Troubleshooting

- `nvcc` not found: set `NVCC` or `CUDA_PATH`/`CUDA_HOME`, or ensure it’s on PATH.
- Arch unsupported by your nvcc: set `CUDA_ARCH=80` or upgrade CUDA Toolkit; the build will also retry with `compute_80` automatically.
- Only some kernels fail to compile: use `CUDA_FILTER` to iterate faster; optionally set `CUDA_PLACEHOLDER_ON_FAIL=1` locally to unblock building and running other tests.
- Cross-crate PTX includes: remember `OUT_DIR` is per-crate; compile PTX in that crate’s build.rs if needed.
