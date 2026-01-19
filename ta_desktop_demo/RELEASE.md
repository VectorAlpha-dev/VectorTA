# VectorTA Optimizer — Windows Release Notes

This repo currently contains both the app and the `vector-ta` library for convenience. Long-term, the desktop app will likely move to its own repo and depend on `vector-ta` as an external crate.

## Build Variants

There are three practical Windows variants:

1. **CPU-only** (default)
2. **GPU (MA sweep)** (`--features cuda`)
3. **GPU (kernel)** (`--features cuda-backtest-kernel`) — fastest path

## Prereqs (Windows)

- Rust stable
- For installers/bundles: `tauri-cli`
  - `cargo install tauri-cli --locked`

> GPU builds require an NVIDIA driver at runtime. The **CUDA Toolkit is not required** for the GPU paths when using prebuilt PTX.

## Local Dev Run (no bundler)

Runs directly via Cargo using the static frontend in `ta_desktop_demo/dist`:

```powershell
cd ta_desktop_demo
cargo run -p ta_desktop_demo_app --release
```

GPU variants:

```powershell
cd ta_desktop_demo
cargo run -p ta_desktop_demo_app --release --features cuda
cargo run -p ta_desktop_demo_app --release --features cuda-backtest-kernel
```

## Installer / Bundle Build

From `ta_desktop_demo/`:

```powershell
cd ta_desktop_demo
cargo tauri build
```

GPU variants:

```powershell
cd ta_desktop_demo
cargo tauri build --features cuda
```

```powershell
cd ta_desktop_demo
$env:VECTORBT_USE_PREBUILT_PTX="1"
cargo tauri build --features cuda-backtest-kernel
```

## Backtest Kernel PTX

The GPU backtest kernel (`double_crossover`) can be built either from:

- **Prebuilt PTX** (default fallback): `ta_desktop_demo/crates/app_core/src/kernels/double_crossover.prebuilt.ptx`
- **Local nvcc build** (developer mode)

Environment toggles:

- `VECTORBT_USE_PREBUILT_PTX=1` forces using the prebuilt PTX (recommended for CI/packaging).
- `VECTORBT_REQUIRE_NVCC=1` fails the build if `nvcc` is missing or compilation fails.
