# SIMD / Scalar Optimization Tracker

Started: 2025-12-18  
Branch: `simd`  
Current HEAD: `0f10d459`

This document is the long-running execution log + checklist for iteratively optimizing every indicator’s Rust kernels (scalar and SIMD where applicable) without sacrificing accuracy in regular unit tests.

## Hard Constraints (Do Not Violate)

- Do **not** change regular unit test reference values or expected characteristics.
- Proptests may be made slightly less strict only to eliminate float-associativity flake; keep changes minimal.
- Do **not** remove features (`python`, `wasm`, `nightly-avx`, `cuda`) or “fix” things by feature-gating work away.
- Do **not** add memory copy operations (no new O(N) “copy to temp then compute” patterns).
- Do **not** reduce kernel optimization/speed as a side-effect; always establish a baseline and confirm improvement.
- Do **not** run `cargo fmt` / rustfmt automatically.

## Standard Workflow (Per Indicator)

For each indicator:

1) **Locate**: module/file, public entrypoints, batch APIs, kernel selection, warmup rules.
2) **Baseline tests** (targeted):
   - `cargo test --features nightly-avx --lib indicators::<module_path> -- --nocapture`
3) **Baseline benches** (targeted; same filters before/after):
   - `RUSTFLAGS="-C target-cpu=native" cargo bench --features nightly-avx --bench indicator_benchmark -- <bench_id_filters>`
   - Prefer `100k` as the primary size; use `1M` when you need lower noise.
4) **Single change**: make one optimization change (hoist invariants, reduce bounds checks, reduce branching, better SIMD utilization, eliminate redundant work, etc.).
5) **Validate**: re-run the same tests and benches.
6) **Keep/revert**:
   - Keep only if improvement is consistent and correctness is preserved.
7) **Record**: update this file with baseline and “after” results + a short rationale.

## Benchmark Notes

- The canonical bench harness is `benches/indicator_benchmark.rs`.
- Bench IDs are discoverable via:
  - `cargo bench --features nightly-avx --bench indicator_benchmark -- --list`
  - Or search the existing `bench_list.txt`.

## Status Legend

- `[ ]` Not started
- `[~]` Baseline recorded (tests + benches)
- `[x] Optimized / no further wins found (for now)`
- `[-]` Skip as trivial (e.g., simple price averages with no meaningful headroom)

## Work Log

#### sma
- Module: `src/indicators/moving_averages/sma.rs`
- Tests: `cargo test --features nightly-avx --lib indicators::moving_averages::sma -- --nocapture`
- Bench filters:
  - Single: `sma/sma_.*100k`
  - Batch: `sma_batch/sma_batch_.*100k`
- Baseline (100k, point estimate):
  - Single: scalar ≈ 36.10 µs, avx2 ≈ 35.92 µs (note: Avx2 is short-circuited to scalar), avx512 ≈ 28.57 µs
  - Batch: scalarbatch ≈ 11.27 ms, avx2batch ≈ 11.28 ms, avx512batch ≈ 11.34 ms (note: kernel selection was effectively scalar)
- Change:
  - Batch prefix-sum row loop now uses kernel-specific SIMD (AVX2/AVX512) for the `ps[i] - ps[i-period]` pass.
  - Removed redundant warmup-prefix initialization in the allocating batch wrapper.
- After (100k, point estimate):
  - Single: scalar ≈ 35.78 µs, avx2 ≈ 35.88 µs (still scalar), avx512 ≈ 28.49 µs
  - Batch: scalarbatch ≈ 11.39 ms (within noise), avx2batch ≈ 11.02 ms, avx512batch ≈ 11.06 ms
- Result: kept (batch AVX2/AVX512 improved; no API/accuracy changes; unit tests unchanged)

#### ema
- Module: `src/indicators/moving_averages/ema.rs`
- Tests: `cargo test --features nightly-avx --lib indicators::moving_averages::ema -- --nocapture`
- Bench filters:
  - Single: `ema/ema_.*100k`
  - Batch: `ema_batch/ema_batch_.*100k`
- Baseline (100k, point estimate):
  - Single: scalar ≈ 116.39 µs, avx2 ≈ 113.76 µs, avx512 ≈ 114.38 µs (SIMD kernels are stubs; small diffs likely noise)
  - Batch: scalarbatch ≈ 10.99 ms, avx2batch ≈ 10.76 ms, avx512batch ≈ 10.68 ms (row kernels are stubs; diffs likely noise)
- Change:
  - Removed duplicate stores in the scalar EMA loops (store `mean`/`prev` once per iteration after the update/skip decision).
  - Applied the same pattern to the batch row kernel (`ema_row_scalar`) to keep per-row behavior identical.
- After (100k, point estimate):
  - Single: scalar ≈ 113.26 µs, avx2 ≈ 113.40 µs, avx512 ≈ 113.25 µs
  - Batch: scalarbatch ≈ 10.97 ms, avx2batch ≈ 10.87 ms, avx512batch ≈ 10.85 ms (no significant change detected)
- Result: kept (single-series scalar improved ~3%; regular unit tests unchanged, including strict batch-vs-stream consistency)

#### wma
- Module: `src/indicators/moving_averages/wma.rs`
- Tests: `cargo test --features nightly-avx --lib indicators::moving_averages::wma -- --nocapture`
- Bench filters:
  - Single: `wma/wma_.*100k`
  - Batch: `wma_batch/wma_batch_.*100k`
- Baseline (100k, point estimate):
  - Single: scalar ≈ 70.12 µs, avx2 ≈ 70.20 µs, avx512 ≈ 72.13 µs (AVX* are stubs; deltas mostly noise)
  - Batch: scalarbatch ≈ 11.53 ms, avx2batch ≈ 11.59 ms, avx512batch ≈ 11.57 ms (AVX* are stubs; deltas mostly noise)
- Change: none kept (attempted div→mul-by-recip; no reliable improvement and reverted to preserve accuracy)
- Result: baseline recorded; revisit later if adding a real SIMD kernel

#### alma
- Module: `src/indicators/moving_averages/alma.rs`
- Tests: `cargo test --features nightly-avx --lib indicators::moving_averages::alma -- --nocapture`
- Bench filters:
  - Single: `alma/alma_.*100k`
  - Batch: `alma_batch/alma_batch_.*100k`
- Baseline (100k, point estimate):
  - Single: scalar ≈ 169.91 µs, avx2 ≈ 86.57 µs, avx512 ≈ 62.93 µs
  - Batch: scalarbatch ≈ 35.88 ms, avx2batch ≈ 22.04 ms, avx512batch ≈ 17.69 ms
- Change: not attempted yet (baseline only; already the reference implementation)
- Result: baseline recorded; revisit after less-mature kernels are improved

#### buff_averages
- Module: `src/indicators/moving_averages/buff_averages.rs`
- Tests: `cargo test --features nightly-avx --lib indicators::moving_averages::buff_averages -- --nocapture`
- Bench filters:
  - Single: `buff_averages/buff_averages_.*100k`
  - Batch: `buff_averages_batch/buff_averages_batch_.*100k`
- Baseline (100k, point estimate):
  - Single: scalar ≈ 235.50 µs, avx2 ≈ 234.04 µs, avx512 ≈ 223.15 µs
  - Batch: scalarbatch ≈ 223.74 µs, avx2batch ≈ 485.02 µs, avx512batch ≈ 484.96 µs (regression for 1-row default sweep)
- Change:
  - Skip the masked pv/vv precompute path when `rows == 1` (default builder sweep), since the extra O(n) pass dominates for a single row.
- After (100k, point estimate):
  - Batch: scalarbatch ≈ 223.52 µs, avx2batch ≈ 224.18 µs, avx512batch ≈ 223.62 µs
- Result: kept (removes large batch regression for forced AVX kernels with single-row sweeps; multi-row path unchanged)

#### cwma
- Module: `src/indicators/moving_averages/cwma.rs`
- Tests: `cargo test --features nightly-avx --lib indicators::moving_averages::cwma -- --nocapture`
- Bench filters:
  - Single: `cwma/cwma_.*100k`
  - Batch: `cwma_batch/cwma_batch_.*100k`
- Baseline (100k, point estimate):
  - Single: scalar ≈ 198.79 µs, avx2 ≈ 107.03 µs, avx512 ≈ 118.71 µs
  - Batch: scalarbatch ≈ 41.13 ms, avx2batch ≈ 19.00 ms, avx512batch ≈ 15.10 ms
- Change:
  - `cwma_avx512` now short-circuits to the AVX2 kernel when `weights.len() < 24` (mirrors existing long-kernel fallback; avoids AVX-512 downclock penalty on small windows).
- After (100k, point estimate):
  - Single: avx512 ≈ 107.42 µs
- Result: kept (improves forced AVX512 single-series performance for small periods; strict unit tests unchanged)

#### dema
- Module: `src/indicators/moving_averages/dema.rs`
- Tests: `cargo test --features nightly-avx --lib indicators::moving_averages::dema -- --nocapture`
- Bench filters:
  - Single: `dema/dema_.*100k`
  - Batch: `dema_batch/dema_batch_.*100k`
- Baseline (100k, point estimate):
  - Single: scalar ≈ 79.11 µs, avx2 ≈ 146.16 µs, avx512 ≈ 35.47 µs
  - Batch: scalarbatch ≈ 9.97 ms, avx2batch ≈ 10.14 ms, avx512batch ≈ 10.10 ms
- Change: not attempted yet (baseline only; AVX2 is known-underperforming and Auto already avoids it)
- Result: baseline recorded; revisit if batch path becomes a hotspot

#### dma
- Module: `src/indicators/moving_averages/dma.rs`
- Tests: `cargo test --features nightly-avx --lib indicators::moving_averages::dma -- --nocapture`
- Bench filters:
  - Single: `dma/dma_.*100k`
  - Batch: `dma_batch/dma_batch_.*100k`
- Baseline (100k, point estimate):
  - Single: scalar ≈ 1.578 ms, avx2 ≈ 1.554 ms, avx512 ≈ 1.548 ms
  - Batch: scalarbatch ≈ 1.552 ms, avx2batch ≈ 1.553 ms, avx512batch ≈ 1.552 ms
- Change: not attempted yet (baseline only)
- Result: baseline recorded

#### edcf
- Module: `src/indicators/moving_averages/edcf.rs`
- Tests: `cargo test --features nightly-avx --lib indicators::moving_averages::edcf -- --nocapture`
- Bench filters:
  - Single: `edcf/edcf_.*100k`
  - Batch: `edcf_batch/edcf_batch_.*100k`
- Baseline (100k, point estimate):
  - Single: scalar ≈ 528.36 µs, avx2 ≈ 331.16 µs, avx512 ≈ 254.00 µs
  - Batch: scalarbatch ≈ 5.233 ms, avx2batch ≈ 2.508 ms, avx512batch ≈ 2.407 ms
- Change:
  - Replaced the two-pass dist-buffer kernel (O(n·period) + full-length scratch) with an O(n) rolling-sums kernel using an O(period) ring buffer and rolling aggregates.
  - Fixed `EdcfStream` sum maintenance to match batch semantics and preserve long-run accuracy; `Kernel::Auto` now selects `Scalar` / `ScalarBatch` for this recurrence-bound kernel.
- After (100k, point estimate):
  - Single: scalar ≈ 173.66 µs, avx2 ≈ 176.08 µs, avx512 ≈ 175.43 µs
  - Batch: scalarbatch ≈ 1.812 ms, avx2batch ≈ 1.821 ms, avx512batch ≈ 1.880 ms
- Result: kept (≈3× faster single-series; ≈2.9× faster batch; regular unit test references unchanged)

#### ehlers_ecema
- Module: `src/indicators/moving_averages/ehlers_ecema.rs`
- Tests: `cargo test --features nightly-avx --lib indicators::moving_averages::ehlers_ecema -- --nocapture`
- Bench filters:
  - Single: `ehlers_ecema/ehlers_ecema_.*100k`
  - Batch: `ehlers_ecema_batch/ehlers_ecema_batch_.*100k`
- Baseline (100k, point estimate):
  - Single: scalar ≈ 585.16 µs, avx2 ≈ 574.45 µs, avx512 ≈ 588.72 µs (SIMD kernels are stubs)
  - Batch: scalarbatch ≈ 544.66 µs, avx2batch ≈ 544.72 µs, avx512batch ≈ 546.47 µs (batch kernels are stubs)
- Change: attempted small scalar loop refactor (split first iteration + confirmed_only branch); regressed ~3% in batch benches → reverted.
- Result: baseline recorded; no wins kept (already close to local optimum / branch-heavy closed-form selection)

#### ehlers_itrend
- Module: `src/indicators/moving_averages/ehlers_itrend.rs`
- Tests: `cargo test --features nightly-avx --lib indicators::moving_averages::ehlers_itrend -- --nocapture`
- Bench filters:
  - Single: `ehlers_itrend/ehlers_itrend_.*100k`
  - Batch: `ehlers_itrend_batch/ehlers_itrend_batch_.*100k`
- Baseline (100k, point estimate):
  - Single: scalar ≈ 4.265 ms, avx2 ≈ 4.238 ms, avx512 ≈ 4.319 ms (AVX kernels delegate to scalar)
  - Batch: scalarbatch ≈ 3.940 ms, avx2batch ≈ 3.958 ms, avx512batch ≈ 3.944 ms
- Change: not attempted yet (baseline only; DSP-heavy + recurrence-bound)
- Result: baseline recorded

#### ehlers_kama
- Module: `src/indicators/moving_averages/ehlers_kama.rs`
- Tests: `cargo test --features nightly-avx --lib indicators::moving_averages::ehlers_kama -- --nocapture`
- Bench filters:
  - Single: `ehlers_kama/ehlers_kama_.*100k`
  - Batch: `ehlers_kama_batch/ehlers_kama_batch_.*100k`
- Baseline (100k, point estimate):
  - Single: scalar ≈ 128.21 µs, avx2 ≈ 128.24 µs, avx512 ≈ 130.12 µs
  - Batch: scalarbatch ≈ 2.140 ms, avx2batch ≈ 2.152 ms, avx512batch ≈ 2.183 ms
- Change:
  - `Kernel::Auto` now resolves to `Scalar` / `ScalarBatch` (recurrence-bound; avoids AVX512 downclock).
  - Added `period == 1` identity fast-path (prevents `start-1` underflow; preserves warmups).
  - Fixed scalar seeding to ignore NaN-prefix diff at `first_valid` (matches batch semantics).
- After (100k): forced-kernel benches unchanged within noise; `Kernel::Auto` now matches scalar.
- Result: kept (correctness + better default behavior on AVX512-capable CPUs)

#### ehlers_pma
- Module: `src/indicators/moving_averages/ehlers_pma.rs`
- Tests: `cargo test --features nightly-avx --lib indicators::moving_averages::ehlers_pma -- --nocapture`
- Bench filters:
  - Single: `ehlers_pma/ehlers_pma_.*100k`
  - Batch: `ehlers_pma_batch/ehlers_pma_batch_.*100k`
- Baseline (100k, point estimate):
  - Single: scalar ≈ 368.95 µs, avx2 ≈ 373.07 µs, avx512 ≈ 372.68 µs (AVX kernels are stubs)
  - Batch: scalarbatch ≈ 377.73 µs, avx2batch ≈ 380.79 µs, avx512batch ≈ 381.31 µs
- Change:
  - Removed O(n) temporary buffers (`wma1`, `wma2`) from main + into APIs by computing WMA2 from a 7-value ring buffer.
  - Kept exact arithmetic order for WMA7/WMA7/WMA4 (tests remain strict to 1e-8 vs PineScript reference).
- After (100k, point estimate):
  - Single: scalar ≈ 348.47 µs, avx2 ≈ 349.47 µs, avx512 ≈ 349.63 µs
  - Batch: scalarbatch ≈ 353.81 µs, avx2batch ≈ 349.22 µs, avx512batch ≈ 352.39 µs
- Result: kept (≈5–6% faster single; ≈6–9% faster batch; outputs unchanged)

#### ehma
- Module: `src/indicators/moving_averages/ehma.rs`
- Tests: `cargo test --features nightly-avx --lib indicators::moving_averages::ehma -- --nocapture`
- Bench filters:
  - Single: `ehma/ehma_.*100k`
  - Batch: `ehma_batch/ehma_batch_.*100k`
- Baseline (100k, point estimate):
  - Single: scalar ≈ 288.51 µs, avx2 ≈ 176.32 µs, avx512 ≈ 221.53 µs
  - Batch: scalarbatch ≈ 291.02 µs, avx2batch ≈ 172.41 µs, avx512batch ≈ 225.79 µs
- Change:
  - `Kernel::Auto` now prefers AVX2 over AVX512 (single + batch + Python batch), since AVX512 downclock makes it slower for the default period.
- After (100k): forced-kernel benches unchanged; Auto now routes to AVX2 on AVX512-capable CPUs.
- Result: kept (default Auto performance improved on AVX512-capable CPUs)

#### epma
- Module: `src/indicators/moving_averages/epma.rs`
- Tests: `cargo test --features nightly-avx --lib indicators::moving_averages::epma -- --nocapture`
- Bench filters:
  - Single: `epma/epma_.*100k`
  - Batch: `epma_batch/epma_batch_.*100k`
- Baseline (100k, point estimate):
  - Single: scalar ≈ 216.95 µs, avx2 ≈ 118.89 µs, avx512 ≈ 61.59 µs
  - Batch: scalarbatch ≈ 1.250 ms, avx2batch ≈ 1.211 ms (noisy), avx512batch ≈ 1.193 ms
- Change: not attempted yet (already strongly SIMD-optimized; no safe low-risk wins identified from quick scan)
- Result: baseline recorded; leave as-is for now

#### frama
- Module: `src/indicators/moving_averages/frama.rs`
- Tests: `cargo test --features nightly-avx --lib indicators::moving_averages::frama -- --nocapture`
- Bench filters:
  - Single: `frama/frama_.*100k`
  - Batch: `frama_batch/frama_batch_.*100k`
- Baseline (100k, point estimate):
  - Single: scalar ≈ 2.215 ms, avx2 ≈ 1.838 ms, avx512 ≈ 2.628 ms
  - Batch: scalarbatch ≈ 2.184 ms, avx2batch ≈ 1.796 ms, avx512batch ≈ 2.587 ms
- Change:
  - `Kernel::Auto` now prefers AVX2 over AVX512 (single + batch + Python batch), since AVX512 is consistently slower here due to downclock.
- After (100k): forced-kernel benches unchanged; Auto now routes to AVX2 on AVX512-capable CPUs.
- Result: kept (significant default Auto speedup on AVX512-capable CPUs)

#### fwma
- Module: `src/indicators/moving_averages/fwma.rs`
- Tests: `cargo test --features nightly-avx --lib indicators::moving_averages::fwma -- --nocapture`
- Bench filters:
  - Single: `fwma/fwma_.*100k`
  - Batch: `fwma_batch/fwma_batch_.*100k`
- Baseline (100k, point estimate):
  - Single: scalar ≈ 106.96 µs, avx2 ≈ 88.99 µs, avx512 ≈ 47.27 µs
  - Batch: scalarbatch ≈ 9.80 ms, avx2batch ≈ 7.36 ms, avx512batch ≈ 6.33 ms
- Change: not attempted yet (already strongly SIMD-optimized; no safe low-risk wins identified from quick scan)
- Result: baseline recorded; leave as-is for now

#### gaussian
- Module: `src/indicators/moving_averages/gaussian.rs`
- Tests: `cargo test --features nightly-avx --lib indicators::moving_averages::gaussian -- --nocapture`
- Bench filters:
  - Single: `gaussian/gaussian_.*100k`
  - Batch: `gaussian_batch/gaussian_batch_.*100k`
- Baseline (100k, point estimate):
  - Single: scalar ≈ 283.94 µs, avx2 ≈ 277.80 µs, avx512 ≈ 277.60 µs (single-series SIMD delegates to scalar/FMA path; deltas mostly noise)
  - Batch: scalarbatch ≈ 20.27 ms, avx2batch ≈ 20.27 ms, avx512batch ≈ 21.04 ms
- Change:
  - Batch `Kernel::Auto` now maps AVX512Batch → AVX2Batch to avoid AVX512 downclock when row-tiling provides no win (also applied to Python batch).
- After (100k): forced-kernel benches unchanged; Auto now avoids AVX512Batch on AVX512-capable CPUs.
- Result: kept (improves default Auto behavior for batch)

#### highpass_2_pole
- Module: `src/indicators/moving_averages/highpass_2_pole.rs`
- Tests: `cargo test --features nightly-avx --lib indicators::moving_averages::highpass_2_pole -- --nocapture`
- Bench filters:
  - Single: `highpass_2_pole/highpass_2_pole_.*100k`
  - Batch: `highpass_2_pole_batch/highpass_2_pole_batch_.*100k`
- Baseline (100k, point estimate):
  - Single: scalar ≈ 76.23 µs, avx2 ≈ 75.70 µs, avx512 ≈ 76.53 µs
  - Batch: scalarbatch ≈ 76.34 µs, avx2batch ≈ 76.27 µs, avx512batch ≈ 75.98 µs
- Change: not attempted yet (already near ceiling; SIMD deltas are within noise at 100k)
- Result: baseline recorded

#### highpass
- Module: `src/indicators/moving_averages/highpass.rs`
- Tests: `cargo test --features nightly-avx --lib indicators::moving_averages::highpass -- --nocapture`
- Bench filters:
  - Single: `highpass/highpass_.*100k`
  - Batch: `highpass_batch/highpass_batch_.*100k`
- Baseline (100k, point estimate):
  - Single: scalar ≈ 69.91 µs, avx2 ≈ 78.80 µs, avx512 ≈ 80.04 µs
  - Batch: scalarbatch ≈ 119.11 µs, avx2batch ≈ 120.75 µs, avx512batch ≈ 119.43 µs
- Change:
  - Python batch `Kernel::Auto` now matches the Rust batch API (`Auto` → `ScalarBatch`), since SIMD underperforms for this IIR.
- After (100k): forced-kernel benches unchanged; Auto behavior is consistent across Rust + Python.
- Result: kept (fixes perf/semantics mismatch for Python batch Auto)

#### hma
- Module: `src/indicators/moving_averages/hma.rs`
- Tests: `cargo test --features nightly-avx --lib indicators::moving_averages::hma -- --nocapture`
- Bench filters:
  - Single: `hma/hma_.*100k`
  - Batch: `hma_batch/hma_batch_.*100k`
- Baseline (100k, point estimate):
  - Single: scalar ≈ 208.01 µs, avx2 ≈ 208.61 µs, avx512 ≈ 141.73 µs
  - Batch: scalarbatch ≈ 6.2188 ms, avx2batch ≈ 6.0639 ms, avx512batch ≈ 5.9038 ms
- Change: not attempted yet (already heavily optimized; no safe incremental wins identified without risking cross-kernel parity)
- Result: baseline recorded

#### hwma
- Module: `src/indicators/moving_averages/hwma.rs`
- Tests: `cargo test --features nightly-avx --lib indicators::moving_averages::hwma -- --nocapture`
- Bench filters:
  - Single: `hwma/hwma_.*100k`
  - Batch: `hwma_batch/hwma_batch_.*100k`
- Baseline (100k, point estimate):
  - Single: scalar ≈ 385.34 µs, avx2 ≈ 385.88 µs, avx512 ≈ 386.50 µs
  - Batch: scalarbatch ≈ 385.67 µs, avx2batch ≈ 390.04 µs, avx512batch ≈ 390.07 µs
- Change:
  - `Kernel::Auto` now routes to `Scalar` / `ScalarBatch` (Rust + Python batch) since HWMA is loop-carried and AVX2/AVX512 target_feature variants can underperform due to downclock.
- After (100k): forced-kernel benches unchanged; default Auto now uses the fastest kernel on this CPU (scalar).
- Result: kept (improves default `Auto` performance; SIMD kernels remain available explicitly)

#### jma
- Module: `src/indicators/moving_averages/jma.rs`
- Tests: `cargo test --features nightly-avx --lib indicators::moving_averages::jma -- --nocapture`
- Bench filters:
  - Single: `jma/jma_.*100k`
  - Batch: `jma_batch/jma_batch_.*100k`
- Baseline (100k, point estimate):
  - Single: scalar ≈ 175.87 µs, avx2 ≈ 156.91 µs, avx512 ≈ 156.32 µs
  - Batch: scalarbatch ≈ 10.876 ms, avx2batch ≈ 11.065 ms, avx512batch ≈ 10.743 ms
- Change:
  - Scalar kernel now uses `mul_add` (FMA) in the hot loop to match the AVX2/AVX512 algebra and remove extra mul+add instructions.
- After (100k, point estimate):
  - Single: scalar ≈ 157.30 µs, avx2 ≈ 156.67 µs, avx512 ≈ 156.14 µs
  - Batch: forced-kernel benches fluctuate; no stable conclusion beyond “no regressions” from limited reruns.
- Result: kept (single-series scalar improved by ~9–10% while preserving unit test expectations)

#### jsa
- Module: `src/indicators/moving_averages/jsa.rs`
- Tests: `cargo test --features nightly-avx --lib indicators::moving_averages::jsa -- --nocapture`
- Bench filters:
  - Single: `jsa/jsa_.*100k`
  - Batch: `jsa_batch/jsa_batch_.*100k`
- Baseline (100k, point estimate):
  - Single: scalar ≈ 9.533 µs, avx2 ≈ 10.295 µs, avx512 ≈ 9.880 µs
  - Batch: scalarbatch ≈ 4.762 ms, avx2batch ≈ 4.683 ms, avx512batch ≈ 4.683 ms
- Change:
  - `Kernel::Auto` now routes to `Scalar` for the single-series API to avoid AVX downclock (scalar autovec is fastest here).
- After (100k): forced-kernel benches unchanged; default single-series Auto now uses scalar.
- Result: kept (improves default `Auto` performance; batch Auto behavior unchanged)

### Template (Copy Per Indicator)

```
#### <indicator>
- Module: `src/indicators/...`
- Tests: <command(s)>
- Bench filters: <ids used>
- Baseline (100k): scalar=..., avx2=..., avx512=... (batch: ...)
- Change: <1–2 lines>
- After (100k): scalar=..., avx2=..., avx512=... (batch: ...)
- Result: <kept/reverted> (<reason>)
```

## Checklist

### Moving Averages (`src/indicators/moving_averages/`)
- [~] alma (`src/indicators/moving_averages/alma.rs`)
- [~] buff_averages (`src/indicators/moving_averages/buff_averages.rs`)
- [~] cwma (`src/indicators/moving_averages/cwma.rs`)
- [~] dema (`src/indicators/moving_averages/dema.rs`)
- [~] dma (`src/indicators/moving_averages/dma.rs`)
- [x] edcf (`src/indicators/moving_averages/edcf.rs`)
- [~] ehlers_ecema (`src/indicators/moving_averages/ehlers_ecema.rs`)
- [~] ehlers_itrend (`src/indicators/moving_averages/ehlers_itrend.rs`)
- [x] ehlers_kama (`src/indicators/moving_averages/ehlers_kama.rs`)
- [x] ehlers_pma (`src/indicators/moving_averages/ehlers_pma.rs`)
- [x] ehma (`src/indicators/moving_averages/ehma.rs`)
- [~] ema (`src/indicators/moving_averages/ema.rs`)
- [~] epma (`src/indicators/moving_averages/epma.rs`)
- [x] frama (`src/indicators/moving_averages/frama.rs`)
- [~] fwma (`src/indicators/moving_averages/fwma.rs`)
- [x] gaussian (`src/indicators/moving_averages/gaussian.rs`)
- [~] highpass_2_pole (`src/indicators/moving_averages/highpass_2_pole.rs`)
- [x] highpass (`src/indicators/moving_averages/highpass.rs`)
- [~] hma (`src/indicators/moving_averages/hma.rs`)
- [x] hwma (`src/indicators/moving_averages/hwma.rs`)
- [x] jma (`src/indicators/moving_averages/jma.rs`)
- [x] jsa (`src/indicators/moving_averages/jsa.rs`)
- [ ] kama (`src/indicators/moving_averages/kama.rs`)
- [ ] linreg (`src/indicators/moving_averages/linreg.rs`)
- [ ] maaq (`src/indicators/moving_averages/maaq.rs`)
- [ ] mama (`src/indicators/moving_averages/mama.rs`)
- [ ] mwdx (`src/indicators/moving_averages/mwdx.rs`)
- [ ] nama (`src/indicators/moving_averages/nama.rs`)
- [ ] nma (`src/indicators/moving_averages/nma.rs`)
- [ ] pwma (`src/indicators/moving_averages/pwma.rs`)
- [ ] reflex (`src/indicators/moving_averages/reflex.rs`)
- [ ] sama (`src/indicators/moving_averages/sama.rs`)
- [ ] sinwma (`src/indicators/moving_averages/sinwma.rs`)
- [~] sma (`src/indicators/moving_averages/sma.rs`)
- [ ] smma (`src/indicators/moving_averages/smma.rs`)
- [ ] sqwma (`src/indicators/moving_averages/sqwma.rs`)
- [ ] srwma (`src/indicators/moving_averages/srwma.rs`)
- [ ] supersmoother_3_pole (`src/indicators/moving_averages/supersmoother_3_pole.rs`)
- [ ] supersmoother (`src/indicators/moving_averages/supersmoother.rs`)
- [ ] swma (`src/indicators/moving_averages/swma.rs`)
- [ ] tema (`src/indicators/moving_averages/tema.rs`)
- [ ] tilson (`src/indicators/moving_averages/tilson.rs`)
- [ ] tradjema (`src/indicators/moving_averages/tradjema.rs`)
- [ ] trendflex (`src/indicators/moving_averages/trendflex.rs`)
- [ ] trima (`src/indicators/moving_averages/trima.rs`)
- [ ] uma (`src/indicators/moving_averages/uma.rs`)
- [ ] vama (`src/indicators/moving_averages/volatility_adjusted_ma.rs`)
- [ ] volume_adjusted_ma (`src/indicators/moving_averages/volume_adjusted_ma.rs`)
- [ ] vpwma (`src/indicators/moving_averages/vpwma.rs`)
- [ ] vwap (`src/indicators/moving_averages/vwap.rs`)
- [ ] vwma (`src/indicators/moving_averages/vwma.rs`)
- [ ] wilders (`src/indicators/moving_averages/wilders.rs`)
- [~] wma (`src/indicators/moving_averages/wma.rs`)
- [ ] zlema (`src/indicators/moving_averages/zlema.rs`)

### Other Indicators (`src/indicators/`)
- [ ] acosc (`src/indicators/acosc.rs`)
- [ ] ad (`src/indicators/ad.rs`)
- [ ] adosc (`src/indicators/adosc.rs`)
- [ ] adx (`src/indicators/adx.rs`)
- [ ] adxr (`src/indicators/adxr.rs`)
- [ ] alligator (`src/indicators/alligator.rs`)
- [ ] alphatrend (`src/indicators/alphatrend.rs`)
- [ ] ao (`src/indicators/ao.rs`)
- [ ] apo (`src/indicators/apo.rs`)
- [ ] aroon (`src/indicators/aroon.rs`)
- [ ] aroonosc (`src/indicators/aroonosc.rs`)
- [ ] aso (`src/indicators/aso.rs`)
- [ ] atr (`src/indicators/atr.rs`)
- [ ] avsl (`src/indicators/avsl.rs`)
- [ ] bandpass (`src/indicators/bandpass.rs`)
- [ ] bollinger_bands (`src/indicators/bollinger_bands.rs`)
- [ ] bollinger_bands_width (`src/indicators/bollinger_bands_width.rs`)
- [ ] bop (`src/indicators/bop.rs`)
- [ ] cci (`src/indicators/cci.rs`)
- [ ] cci_cycle (`src/indicators/cci_cycle.rs`)
- [ ] ce (`src/indicators/ce.rs`)
- [ ] cg (`src/indicators/cg.rs`)
- [ ] chande (`src/indicators/chande.rs`)
- [ ] chandelier_exit (`src/indicators/chandelier_exit.rs`)
- [ ] chop (`src/indicators/chop.rs`)
- [ ] cksp (`src/indicators/cksp.rs`)
- [ ] cmo (`src/indicators/cmo.rs`)
- [ ] coda (`src/indicators/coda.rs`)
- [ ] coppock (`src/indicators/coppock.rs`)
- [ ] cora_wave (`src/indicators/cora_wave.rs`)
- [ ] correl_hl (`src/indicators/correl_hl.rs`)
- [ ] correlation_cycle (`src/indicators/correlation_cycle.rs`)
- [ ] cvi (`src/indicators/cvi.rs`)
- [ ] damiani_volatmeter (`src/indicators/damiani_volatmeter.rs`)
- [ ] dec_osc (`src/indicators/dec_osc.rs`)
- [ ] decycler (`src/indicators/decycler.rs`)
- [ ] deviation (`src/indicators/deviation.rs`)
- [ ] devstop (`src/indicators/devstop.rs`)
- [ ] di (`src/indicators/di.rs`)
- [ ] dm (`src/indicators/dm.rs`)
- [ ] donchian (`src/indicators/donchian.rs`)
- [ ] dpo (`src/indicators/dpo.rs`)
- [ ] dti (`src/indicators/dti.rs`)
- [ ] dx (`src/indicators/dx.rs`)
- [ ] efi (`src/indicators/efi.rs`)
- [ ] ehlers_fisher_transform (`src/indicators/ehlers_fisher_transform.rs`)
- [ ] emd (`src/indicators/emd.rs`)
- [ ] emv (`src/indicators/emv.rs`)
- [ ] er (`src/indicators/er.rs`)
- [ ] eri (`src/indicators/eri.rs`)
- [ ] fisher (`src/indicators/fisher.rs`)
- [ ] fosc (`src/indicators/fosc.rs`)
- [ ] fvg_trailing_stop (`src/indicators/fvg_trailing_stop.rs`)
- [ ] gatorosc (`src/indicators/gatorosc.rs`)
- [ ] halftrend (`src/indicators/halftrend.rs`)
- [ ] hti (`src/indicators/hti.rs`)
- [ ] hurst (`src/indicators/hurst.rs`)
- [ ] ift_rsi (`src/indicators/ift_rsi.rs`)
- [ ] inertia (`src/indicators/inertia.rs`)
- [ ] kdj (`src/indicators/kdj.rs`)
- [ ] kelch (`src/indicators/kelch.rs`)
- [ ] keltner (`src/indicators/keltner.rs`)
- [ ] kst (`src/indicators/kst.rs`)
- [ ] kurtosis (`src/indicators/kurtosis.rs`)
- [ ] kvo (`src/indicators/kvo.rs`)
- [ ] lag (`src/indicators/lag.rs`)
- [ ] linearreg_angle (`src/indicators/linearreg_angle.rs`)
- [ ] linearreg_intercept (`src/indicators/linearreg_intercept.rs`)
- [ ] linearreg_slope (`src/indicators/linearreg_slope.rs`)
- [ ] lrsi (`src/indicators/lrsi.rs`)
- [ ] lpc (`src/indicators/lpc.rs`)
- [ ] mab (`src/indicators/mab.rs`)
- [ ] macd (`src/indicators/macd.rs`)
- [ ] macz (`src/indicators/macz.rs`)
- [ ] marketefi (`src/indicators/marketefi.rs`)
- [ ] mass (`src/indicators/mass.rs`)
- [ ] mean_ad (`src/indicators/mean_ad.rs`)
- [ ] medium_ad (`src/indicators/medium_ad.rs`)
- [ ] medprice (`src/indicators/medprice.rs`)
- [ ] mfi (`src/indicators/mfi.rs`)
- [ ] midpoint (`src/indicators/midpoint.rs`)
- [ ] midprice (`src/indicators/midprice.rs`)
- [ ] minmax (`src/indicators/minmax.rs`)
- [ ] mod_god_mode (`src/indicators/mod_god_mode.rs`)
- [ ] mom (`src/indicators/mom.rs`)
- [ ] msw (`src/indicators/msw.rs`)
- [ ] nadaraya_watson_envelope (`src/indicators/nadaraya_watson_envelope.rs`)
- [ ] natr (`src/indicators/natr.rs`)
- [ ] net_myrsi (`src/indicators/net_myrsi.rs`)
- [ ] nvi (`src/indicators/nvi.rs`)
- [ ] obv (`src/indicators/obv.rs`)
- [ ] ott (`src/indicators/ott.rs`)
- [ ] pfe (`src/indicators/pfe.rs`)
- [ ] pivot (`src/indicators/pivot.rs`)
- [ ] pma (`src/indicators/pma.rs`)
- [ ] ppo (`src/indicators/ppo.rs`)
- [ ] psl (`src/indicators/psl.rs`)
- [ ] pvi (`src/indicators/pvi.rs`)
- [ ] qqe (`src/indicators/qqe.rs`)
- [ ] roc (`src/indicators/roc.rs`)
- [ ] rocr (`src/indicators/rocr.rs`)
- [ ] rsi (`src/indicators/rsi.rs`)
- [ ] rsmk (`src/indicators/rsmk.rs`)
- [ ] rsx (`src/indicators/rsx.rs`)
- [ ] rvi (`src/indicators/rvi.rs`)
- [ ] safezonestop (`src/indicators/safezonestop.rs`)
- [ ] sar (`src/indicators/sar.rs`)
- [ ] si (`src/indicators/si.rs`)
- [ ] skew (`src/indicators/skew.rs`)
- [ ] sma_cross (`src/indicators/sma_cross.rs`)
- [ ] smi (`src/indicators/smi.rs`)
- [ ] smi_ergodic (`src/indicators/smi_ergodic.rs`)
- [ ] smi_ergodic_bands (`src/indicators/smi_ergodic_bands.rs`)
- [ ] smi_ergodic_trigger (`src/indicators/smi_ergodic_trigger.rs`)
- [ ] sortino_ratio (`src/indicators/sortino_ratio.rs`)
- [ ] squeeze_momentum (`src/indicators/squeeze_momentum.rs`)
- [ ] srsi (`src/indicators/srsi.rs`)
- [ ] stc (`src/indicators/stc.rs`)
- [ ] stddev (`src/indicators/stddev.rs`)
- [ ] stoch (`src/indicators/stoch.rs`)
- [ ] stochrsi (`src/indicators/stochrsi.rs`)
- [ ] supertrend (`src/indicators/supertrend.rs`)
- [ ] t3 (`src/indicators/t3.rs`)
- [ ] tema_full (`src/indicators/tema_full.rs`)
- [ ] tmo (`src/indicators/tmo.rs`)
- [ ] tr (`src/indicators/tr.rs`)
- [ ] trix (`src/indicators/trix.rs`)
- [ ] tsi (`src/indicators/tsi.rs`)
- [ ] ttf (`src/indicators/ttf.rs`)
- [ ] typprice (`src/indicators/typprice.rs`)
- [ ] ultimate_oscillator (`src/indicators/ultimate_oscillator.rs`)
- [ ] var (`src/indicators/var.rs`)
- [ ] vhf (`src/indicators/vhf.rs`)
- [ ] vidya (`src/indicators/vidya.rs`)
- [ ] volatility_stop (`src/indicators/volatility_stop.rs`)
- [ ] vortex (`src/indicators/vortex.rs`)
- [ ] vosc (`src/indicators/vosc.rs`)
- [ ] voss (`src/indicators/voss.rs`)
- [ ] vpt (`src/indicators/vpt.rs`)
- [ ] vwmacd (`src/indicators/vwmacd.rs`)
- [ ] vwap_bands (`src/indicators/vwap_bands.rs`)
- [ ] wclprice (`src/indicators/wclprice.rs`)
- [ ] willr (`src/indicators/willr.rs`)
- [ ] wpr (`src/indicators/wpr.rs`)
- [ ] zscore (`src/indicators/zscore.rs`)
