# VectorTA

VectorTA is a Rust crate of 340 implemented technical analysis indicators focused on speed, predictable allocations, and practical execution flexibility, with optional SIMD/CUDA acceleration and optional Python/WASM bindings. In addition to standard single call APIs, much of the library also supports streaming/stateful updates, batch parameter sweeps, and registry-driven dispatch across multiple input and output shapes.

It is intended for workloads where throughput and execution behavior actually matter, including research pipelines, backtesting systems, and high throughput production use cases. The library is not limited to close only, single output indicators either. It spans a wide range of input structures, multi-output studies, and execution paths across Rust, Python, WASM, and CUDA.

The CUDA bindings are predominantly only worth using if used in a VRAM-resident workflow. For example, it is possible to achieve a benchmark timing of 3.129 ms for 250 million calculated ALMA indicator data points on an RTX 4090, whereas the CPU (AMD 9950X) AVX-512, AVX2, and scalar timings are approximately 140.61 ms, 188.64 ms, and 386.20 ms, respectively. The Python bindings also expose GPU-oriented workflows for a subset of indicators, including device-resident outputs intended for high-throughput research pipelines.

The Tauri backtest optimization demo application using this library can achieve 58,300 backtests for a double ALMA crossover strategy over 200k data points in only 85.863 milliseconds on the same hardware (RTX 4090 + AMD 9950X).

For the full indicator list, API reference, and usage guides, see: https://vectoralpha.dev/projects/ta



## Rust usage

Example: computing ADX from HLC slices

```rust
use vector_ta::indicators::adx::{adx, AdxInput, AdxParams};

fn compute_adx(
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let input = AdxInput::from_slices(high, low, close, AdxParams { period: Some(14) });
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
vector-ta = { version = "0.2.2", features = ["cuda"] }
```

Notes:
- To force-disable CUDA probing/usage (tests/CI): set `CUDA_FORCE_SKIP=1`.
- To override where prebuilt PTX is sourced from, set `VECTOR_TA_PREBUILT_PTX_DIR` (see docs link above).

## License

Apache-2.0 (see `LICENSE`).
