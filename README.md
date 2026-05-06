# VectorTA

VectorTA is a Rust crate of 340 implemented technical analysis indicators focused on speed, predictable allocations, and practical execution flexibility, with optional SIMD/CUDA acceleration and optional Python/WASM bindings. In addition to standard single call APIs, much of the library also supports streaming/stateful updates, batch parameter sweeps, and registry-driven dispatch across multiple input and output shapes.

It is intended for workloads where throughput and execution behavior actually matter, including research pipelines, backtesting systems, and high throughput production use cases. The library is not limited to close only, single output indicators either. It spans a wide range of input structures, multi-output studies, and execution paths across Rust, Python, WASM, and CUDA.

The CUDA bindings are predominantly only worth using if used in a VRAM-resident workflow. For example, it is possible to achieve a benchmark timing of 3.129 ms for 250 million calculated ALMA indicator data points on an RTX 4090, whereas the CPU (AMD 9950X) AVX-512, AVX2, and scalar timings are approximately 140.61 ms, 188.64 ms, and 386.20 ms, respectively. The Python bindings also expose GPU-oriented workflows for a subset of indicators, including device-resident outputs intended for high-throughput research pipelines.

The Tauri backtest optimization demo application using this library can achieve 58,300 backtests for a double ALMA crossover strategy over 200k data points in only 85.863 milliseconds on the same hardware (RTX 4090 + AMD 9950X).

For the full indicator list, API reference, and usage guides, see: https://vectoralpha.dev/projects/ta

## Benchmarks

The table below compares indicators that have a Tulip Indicators C counterpart. The CPU workload is 100,000 candles with one parameter set per indicator measured as average per call time after warmup. The CUDA columns use the corresponding GPU benchmark scenario for the same indicator, as most are 1M x 250 one-series parameter runs, while rows labeled otherwise use that indicator's available single CUDA profile. CUDA timings are reported as the measured scenario time divided by 250, rounded to microseconds, and the CUDA speedup is relative to Tulip C CPU. Lower time is better within the same workload.  

Environment: AMD Ryzen 9 9950X, NVIDIA GeForce RTX 4090, Windows 11 Pro, Rust `1.95.0-nightly`.

| Indicator | Tulip C CPU 100k | VectorTA CPU 100k | VectorTA vs Tulip | VectorTA CUDA workload | VectorTA CUDA / 250 | CUDA / 250 vs Tulip |
|---|---:|---:|---:|---:|---:|---:|
| mfi | 0.652 ms | 0.137 ms | 4.76x | 1M x 250 | 5.19 us | 125.58x |
| cmo | 0.827 ms | 0.189 ms | 4.37x | 1M x 250 | 67.54 us | 12.24x |
| msw | 13.394 ms | 3.144 ms | 4.26x | 1M x 250 | 653.62 us | 20.49x |
| kama | 1.365 ms | 0.330 ms | 4.14x | 1M x 250 | 1,565.78 us | 0.87x |
| obv | 0.336 ms | 0.098 ms | 3.44x | 1M one series | 0.17 us | 1,953.49x |
| adxr | 0.635 ms | 0.265 ms | 2.40x | 1M x 250 | 780.37 us | 0.81x |
| bop | 0.067 ms | 0.030 ms | 2.20x | 1M x 250 | 12.83 us | 5.22x |
| dema | 0.151 ms | 0.079 ms | 1.91x | 1M x 250 | 134.61 us | 1.12x |
| sma | 0.069 ms | 0.037 ms | 1.90x | 1M x 250 | 11.48 us | 6.01x |
| rsi | 0.389 ms | 0.208 ms | 1.87x | 1M x 250 | 51.62 us | 7.54x |
| trima | 0.106 ms | 0.060 ms | 1.75x | 1M x 250 | 30.59 us | 3.47x |
| wad | 0.372 ms | 0.218 ms | 1.71x | 1M one series | 0.22 us | 1,660.71x |
| di | 0.556 ms | 0.352 ms | 1.58x | 1M x 250 | 259.94 us | 2.14x |
| adx | 0.528 ms | 0.345 ms | 1.53x | 1M x 250 | 511.60 us | 1.03x |
| mom | 0.020 ms | 0.013 ms | 1.50x | 1M x 250 | 4.38 us | 4.57x |
| srsi | 1.466 ms | 1.012 ms | 1.45x | 1M x 250 | 45.32 us | 32.34x |
| apo | 0.147 ms | 0.108 ms | 1.36x | 1M x 250 | 36.52 us | 4.03x |
| mass | 0.454 ms | 0.335 ms | 1.36x | 1M x 250 | 4.42 us | 102.81x |
| ema | 0.139 ms | 0.103 ms | 1.35x | 1M x 250 | 22.98 us | 6.05x |
| zlema | 0.140 ms | 0.106 ms | 1.32x | 1M x 250 | 40.77 us | 3.43x |
| nvi | 0.224 ms | 0.180 ms | 1.24x | 1M one series | 1.48 us | 150.94x |
| ultosc | 0.512 ms | 0.420 ms | 1.22x | 1M x 250 | 13.28 us | 38.57x |

More benchmarks can be found at https://vectoralpha.dev/projects/ta/benchmarks/

## Install

Add the crate as `vector-ta` and import it as `vector_ta`:

```toml
[dependencies]
vector-ta = "0.2.9"
```

For full SIMD functionality on `x86_64`, use a nightly Rust toolchain and enable the `nightly-avx` feature. Stable Rust still works for the scalar implementation.

## Rust usage

Example: computing ADX from HLC slices

```rust
use vector_ta::indicators::adx::{adx, AdxInput, AdxParams};

fn compute_adx(
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let input = AdxInput::from_slices(high, low, close, AdxParams {period: Some(14)});
    Ok(adx(&input)?.values)
}
```

## Features

- `cuda`: GPU acceleration using prebuilt PTX for `compute_89` shipped in the crate.
- `cuda-build-ptx`: compile PTX from `kernels/cuda/**` using `nvcc`.
- `nightly-avx`: runtime selected AVX2 and AVX512 kernels on `x86_64`.
- `python`: PyO3 bindings built from source with `maturin`.
- `wasm`: wasm-bindgen bindings built from source with `wasm-pack`.

## Python (optional)

Build and install into a virtualenv:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip maturin numpy
maturin develop --release --features python
```

## WASM (optional)

Build the Node-targeted package with `wasm-pack`:

```bash
rustup target add wasm32-unknown-unknown
wasm-pack build --target nodejs --release --features wasm
```

## CUDA (optional)

Enable the CUDA feature:

```toml
[dependencies]
vector-ta = {version = "0.2.9", features = ["cuda"]}
```

Notes:
- To force-disable CUDA probing/usage (tests/CI): set `CUDA_FORCE_SKIP=1`.
- To override where prebuilt PTX is sourced from, set `VECTOR_TA_PREBUILT_PTX_DIR` (see docs link above).

## License

Apache-2.0 (see `LICENSE`).
