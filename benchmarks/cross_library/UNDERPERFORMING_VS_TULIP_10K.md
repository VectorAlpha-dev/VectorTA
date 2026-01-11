# Underperforming vs Tulip C (10k)

This report lists indicators where **Rust Native** is **slower than Tulip C by more than 5%** for the **10k** dataset (10,000 candles).

Source: `benchmarks/cross_library/benchmark_results.json` (`metadata.timestamp = 2026-01-03T17:26:25.213077700+00:00`).

Definition:
- Include an indicator if `RustNative.raw_time_us / Tulip.raw_time_us > 1.05` at `data_size = 10000`.

Notes:
- Tulip timings are missing for some indicators (e.g., `dx`, `midpoint`, `midprice`, `rocp`, `stochf`, `wclprice`) and are not included here.
- `minmax` and `minmax_min` are mapped to Tulip `max` and `min` in `benchmarks/cross_library/benches/cross_library_comparison.rs`, so those rows are not strictly apples-to-apples.

Summary:
- Total indicators with Tulip data at 10k: **67**
- Slower than Tulip by >5% at 10k: **32**

| Indicator | Rust Native (µs) | Tulip (µs) | Rust/Tulip | Rust slower by |
|-----------|------------------|------------|------------|----------------|
| minmax | 61.66 | 11.63 | 5.302 | 430.2% |
| minmax_min | 66.9 | 13.66 | 4.898 | 389.8% |
| marketefi | 10.45 | 2.47 | 4.231 | 323.1% |
| aroon | 163.48 | 44.77 | 3.652 | 265.2% |
| aroonosc | 79.11 | 26.12 | 3.029 | 202.9% |
| trix | 49.16 | 16.45 | 2.988 | 198.8% |
| linearreg_slope | 23.65 | 8.1 | 2.92 | 192% |
| stoch | 152.16 | 66.1 | 2.302 | 130.2% |
| willr | 68.41 | 34.94 | 1.958 | 95.8% |
| pvi | 17.25 | 9.2 | 1.875 | 87.5% |
| nvi | 16 | 8.81 | 1.816 | 81.6% |
| tsf | 16.72 | 9.32 | 1.794 | 79.4% |
| fisher | 188.85 | 118.12 | 1.599 | 59.9% |
| adx | 37.23 | 25.13 | 1.481 | 48.1% |
| sar | 34.35 | 23.82 | 1.442 | 44.2% |
| natr | 20.62 | 14.69 | 1.404 | 40.4% |
| bollinger_bands | 21.89 | 15.73 | 1.392 | 39.2% |
| stddev | 20.43 | 15.81 | 1.292 | 29.2% |
| wma | 8.74 | 7.03 | 1.243 | 24.3% |
| vwma | 8.69 | 7.15 | 1.215 | 21.5% |
| mom | 2.41 | 1.99 | 1.211 | 21.1% |
| medprice | 1.86 | 1.54 | 1.208 | 20.8% |
| dpo | 9.17 | 7.76 | 1.182 | 18.2% |
| wad | 24.21 | 20.58 | 1.176 | 17.6% |
| macd | 23.57 | 20.51 | 1.149 | 14.9% |
| fosc | 12.89 | 11.31 | 1.14 | 14% |
| linearreg_intercept | 10.39 | 9.28 | 1.12 | 12% |
| var | 8.8 | 8.01 | 1.099 | 9.9% |
| ad | 11.29 | 10.3 | 1.096 | 9.6% |
| cci | 49.22 | 44.92 | 1.096 | 9.6% |
| qstick | 7.85 | 7.44 | 1.055 | 5.5% |
| srsi | 147.95 | 140.39 | 1.054 | 5.4% |

