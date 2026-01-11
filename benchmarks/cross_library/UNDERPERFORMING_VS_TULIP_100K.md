# Underperforming vs Tulip C (100k)

This report lists indicators where **Rust Native** is **slower than Tulip C by more than 5%** for the **100k** dataset (100,000 candles).

Source: `benchmarks/cross_library/benchmark_results.full.nightly-avx25.json` (`metadata.timestamp = 2026-01-05T04:36:25.306468300+00:00`).

## Status updates (after `full.nightly-avx25`)

- 2026-01-04: `ultosc` now uses Tulip-style true range (`true_high - true_low`) which is equivalent for well-formed candles (`high >= low`), and is faster than Tulip at 100k in a filtered run: Rust Native `258.17us`, Tulip `491.59us` (ratio `0.525`). Source: `benchmarks/cross_library/benchmark_results.ultosc.nightly-avx23.json`.
- 2026-01-04: `aroon` scalar hot loop (all-finite path) optimized; now faster than Tulip at 100k in a filtered run: Rust Native `589.07us`, Tulip `637.90us` (ratio `0.923`). Source: `benchmarks/cross_library/benchmark_results.aroon.nightly-avx24.json`.
- 2026-01-05: `bollinger_bands` optimized; now faster than Tulip at 100k in the full run: Rust Native `120.49us`, Tulip `164.24us` (ratio `0.734`). Source: `benchmarks/cross_library/benchmark_results.full.nightly-avx25.json`.
- 2026-01-05: `tsf` and `fosc` optimized; both now faster than Tulip at 100k in the full run: `tsf` Rust Native `92.87us` vs Tulip `94.20us` (ratio `0.986`), `fosc` Rust Native `96.24us` vs Tulip `108.30us` (ratio `0.889`). Source: `benchmarks/cross_library/benchmark_results.full.nightly-avx25.json`.

Definition:
- Include an indicator if `RustNative.raw_time_us / Tulip.raw_time_us > 1.05` at `data_size = 100000`.

Notes:
- The `benchmark_results*.json` timings come from the internal `measure_and_collect` sweep (currently `iterations = 10`), not Criterion’s statistical estimates; very fast indicators (tens of microseconds) can bounce above/below the 5% threshold between runs. Confirm close calls with `cargo bench --features nightly-avx --bench cross_library_comparison -- "<indicator>/100k"`.
- Tulip timings are missing for some indicators (e.g., `dx`, `midpoint`, `midprice`, `rocp`, `stochf`, `wclprice`) and are not included here.
- `minmax*` and `minmax_min*` are mapped to Tulip `max` and `min` in `benchmarks/cross_library/benches/cross_library_comparison.rs`, so those rows are not strictly apples-to-apples.
- `trix+` is not apples-to-apples: Rust TRIX applies triple EMA to `ln(price)` and outputs a scaled delta (`(ema3 - prev) * 10000`), while Tulip TRIX runs on raw price and outputs a percent change (`(ema3 - last) / ema3 * 100`).
- `marketefi!` is not strictly apples-to-apples: Rust emits `NaN` when `volume == 0`, while Tulip `marketfi` computes `(high - low) / volume` without guarding. The 100k dataset contains **~8.94%** zero-volume rows.
- `emv!` is not strictly apples-to-apples on this dataset: Rust emits `NaN` when `high == low` (range == 0), while Tulip `emv` does not guard and typically yields ~0 for that step (`br = volume/10000/0 -> inf`, `(hl - last)/inf -> 0`). The 100k dataset contains **~50%** `high == low` rows.
- This benchmark run was executed with cross-library feature `nightly-avx` enabled (Rust AVX2/AVX-512 kernels where available) and `RUSTFLAGS="-C target-cpu=native"`.

Summary:
- Total indicators with Tulip data at 100k: **67**
- Slower than Tulip by >5% at 100k: **13**

| Indicator | Rust Native (us) | Tulip (us) | Rust/Tulip | Rust slower by |
|-----------|------------------|------------|------------|----------------|
| minmax_min* | 1233.73 | 313.82 | 3.931 | 293.1% |
| minmax* | 1092.95 | 333.13 | 3.281 | 228.1% |
| trix+ | 344.80 | 161.91 | 2.130 | 113.0% |
| bop | 45.19 | 29.55 | 1.529 | 52.9% |
| linearreg_slope | 119.11 | 80.08 | 1.487 | 48.7% |
| emv! | 319.04 | 215.63 | 1.480 | 48.0% |
| marketefi! | 41.75 | 31.08 | 1.343 | 34.3% |
| medprice | 29.29 | 23.27 | 1.259 | 25.9% |
| vosc | 90.01 | 78.07 | 1.153 | 15.3% |
| qstick | 81.82 | 71.26 | 1.148 | 14.8% |
| var | 87.10 | 77.71 | 1.121 | 12.1% |
| wma | 78.53 | 70.53 | 1.113 | 11.3% |
| srsi | 1443.52 | 1369.56 | 1.054 | 5.4% |
