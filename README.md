# vector-ta

VectorTA is a Rust crate of technical analysis indicators focused on speed and predictable allocations, with optional SIMD/CUDA acceleration and optional Python/WASM bindings.

Full documentation (indicator list, API reference, and guides): https://vectoralpha.dev/projects/ta

The CUDA bindings are predominantly only worth using if used in a VRAM-resident workflow. For example, I can achieve a benchmark timing of 6.08 ms for 250 million calculated ALMA indicator data points on an RTX 4090, whereas the CPU (AMD 9950X) AVX-512, AVX2, and scalar timings are approximately 140.61 ms, 188.64 ms, and 386.20 ms, respectively.

The Tauri backtest optimization demo application using this library can achieve 58300 backtests for a double ALMA crossover strategy over 200k data points in only 158.71 milliseconds on the same hardware (RTX 4090 + AMD 9950X). 



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
vector-ta = { version = "0.1.2", features = ["cuda"] }
```

Notes:
- To force-disable CUDA probing/usage (tests/CI): set `CUDA_FORCE_SKIP=1`.
- To override where prebuilt PTX is sourced from, set `VECTOR_TA_PREBUILT_PTX_DIR` (see docs link above).

## License

Apache-2.0 (see `LICENSE`).
