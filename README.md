# vector-ta

VectorTA is a Rust crate of over 350 implemented technical analysis indicators focused on speed, predictable allocations, and practical execution flexibility, with optional SIMD/CUDA acceleration and optional Python/WASM bindings. In addition to standard single-call APIs, much of the library also supports streaming/stateful updates, batch parameter sweeps, and registry-driven dispatch across multiple input and output shapes.

Full documentation (indicator list, API reference, and guides): https://vectoralpha.dev/projects/ta

The CUDA bindings are predominantly only worth using if used in a VRAM-resident workflow. For example, it is possible to achieve a benchmark timing of 3.69 ms for 250 million calculated ALMA indicator data points on an RTX 4090, whereas the CPU (AMD 9950X) AVX-512, AVX2, and scalar timings are approximately 140.61 ms, 188.64 ms, and 386.20 ms, respectively. The Python bindings also expose GPU-oriented workflows for a subset of indicators, including device-resident outputs intended for high-throughput research pipelines.

The Tauri backtest optimization demo application using this library can achieve 58,300 backtests for a double ALMA crossover strategy over 200k data points in only 85.863 milliseconds on the same hardware (RTX 4090 + AMD 9950X). 



## Rust usage

Example: ADX over HLC slices

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

- `cuda`: GPU acceleration using prebuilt PTX (compute_89) shipped in the crate.
- `cuda-build-ptx`: You can compile PTX from `kernels/cuda/**` using `nvcc`.
- `nightly-avx`: Runtime-selected AVX2/AVX512 kernels on `x86_64` (nightly required).
- `python`: PyO3 bindings (build from source via `maturin`).
- `wasm`: wasm-bindgen bindings (build from source via `wasm-pack`).

## Python (optional)

Build + install into a virtualenv:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip maturin numpy
maturin develop --release --features python
```

## WASM (optional)

Build with `wasm-pack`:

```bash
rustup target add wasm32-unknown-unknown
wasm-pack build --target nodejs --release --features wasm
```

## CUDA (optional)

Enable:

```toml
[dependencies]
vector-ta = { version = "0.1.8", features = ["cuda"] }
```

Notes:
- To force-disable CUDA probing/usage (tests/CI): set `CUDA_FORCE_SKIP=1`.
- To override where prebuilt PTX is sourced from, set `VECTOR_TA_PREBUILT_PTX_DIR` (see docs link above).

## License

Apache-2.0 (see `LICENSE`).
