# vector-ta

High-performance technical analysis indicators, implemented in Rust, with optional SIMD/CUDA acceleration and optional Python/WASM bindings.

## Install

```toml
[dependencies]
vector-ta = "0.1"
```

## Usage

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

## Demos

- `ta_desktop_demo` (primary demo): Tauri desktop app for the double-MA optimizer.
  - Run (CPU): `cd ta_desktop_demo; cargo run -p ta_desktop_demo_app --release`
  - Run (GPU MA sweep): `cd ta_desktop_demo; cargo run -p ta_desktop_demo_app --release --features cuda`
  - Run (GPU kernel backend): `cd ta_desktop_demo; cargo run -p ta_desktop_demo_app --release --features cuda,cuda-backtest-kernel`
- `gpu_backtester_demo` (legacy CLI): kept mostly for reference.

## CUDA

CUDA is optional and feature-gated. This crate ships prebuilt PTX for `compute_89` (RTX 4000 / Ada) so consumers do not need the CUDA Toolkit or `nvcc` installed.

Enable:

```toml
[dependencies]
vector-ta = { version = "0.1", features = ["cuda"] }
```

Example: ADX batch on GPU (outputs a `rows x cols` time-major matrix as a flat slice)

```rust
use vector_ta::cuda::{cuda_available, CudaAdx};
use vector_ta::indicators::adx::AdxBatchRange;

fn adx_batch_cuda(
    high: &[f32],
    low: &[f32],
    close: &[f32],
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    if !cuda_available() {
        return Err("CUDA not available on this system".into());
    }

    let cuda = CudaAdx::new(0)?;
    let sweep = AdxBatchRange { period: (14, 263, 1) };

    // rows = number of parameter combinations, cols = series length
    // Here: (263 - 14 + 1) = 250 rows
    let mut out = vec![0.0f32; 250 * high.len()];
    let (_rows, _cols, _params) = cuda.adx_batch_into_host_f32(high, low, close, &sweep, &mut out)?;
    Ok(out)
}
```

Notes:
- The first call may include driver JIT work (loading PTX to SASS); warm up before benchmarking.
- To force-disable CUDA probing/usage (tests/CI): set `CUDA_FORCE_SKIP=1`.
- To override where prebuilt PTX is sourced from, set `VECTOR_TA_PREBUILT_PTX_DIR` to a directory containing `*.ptx` files named like `adx_kernel.ptx`.

## Features

- `nightly-avx`: Enables AVX2/AVX512 kernels on `x86_64` (requires nightly; runtime-selected).
- `cuda`: Uses prebuilt PTX (compute_89) staged into the build output; requires an NVIDIA driver and a CUDA-capable GPU, but does not require `nvcc` for consumers.
- `cuda-build-ptx`: Maintainer-only: compile PTX from `kernels/cuda/**` using `nvcc` and stage it for `cuda` builds.
- `python`: Builds Python bindings via PyO3 (`extension-module`).
- `wasm`: Exposes WASM bindings via `wasm-bindgen`.

## Python bindings (PyO3)

Python bindings are optional and feature-gated. They are built from source (this repo) via `maturin`.

Build + install into a virtualenv:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip maturin numpy
maturin develop --release --features python
```

Usage:

```python
import numpy as np
import vector_ta

high = np.asarray([10.0, 11.0, 12.0], dtype=np.float64)
low = np.asarray([9.0, 9.5, 10.0], dtype=np.float64)
close = np.asarray([9.5, 10.5, 11.0], dtype=np.float64)

adx = vector_ta.adx(high, low, close, period=14, kernel="auto")
```

Notes:
- `kernel` can be `"auto"`, `"scalar"`, `"avx2"`, or `"avx512"` (AVX options require `--features python,nightly-avx` and a supported CPU).
- CUDA-backed Python APIs require building with `--features python,cuda` and a working NVIDIA driver/GPU.

## WASM bindings (wasm-pack)

WASM bindings are optional and feature-gated. Build them with `wasm-pack`:

```bash
rustup target add wasm32-unknown-unknown
wasm-pack build --target nodejs --release --features wasm
```

Then import the generated module from `pkg/` (the filename is based on the crate name, with `-` typically becoming `_`):

```js
const wasm = await import("./pkg/vector_ta.js");
const out = wasm.adx_js(new Float64Array(high), new Float64Array(low), new Float64Array(close), 14);
```

## License

MIT (see `LICENSE`).
