# vector-ta

VectorTA is a Rust crate of technical analysis indicators focused on speed and predictable allocations, with optional SIMD/CUDA acceleration and optional Python/WASM bindings.

Full documentation (indicator list, API reference, and guides): https://vectoralpha.dev/projects/ta

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

- `cuda`: GPU acceleration using prebuilt PTX (compute_89) shipped in the crate. Consumers do not need `nvcc`.
- `cuda-build-ptx`: Maintainer-only: compile PTX from `kernels/cuda/**` using `nvcc`.
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
vector-ta = { version = "0.1.1", features = ["cuda"] }
```

Notes:
- To force-disable CUDA probing/usage (tests/CI): set `CUDA_FORCE_SKIP=1`.
- To override where prebuilt PTX is sourced from, set `VECTOR_TA_PREBUILT_PTX_DIR` (see docs link above).

## License

Apache-2.0 (see `LICENSE`).
