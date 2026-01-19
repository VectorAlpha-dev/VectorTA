# VectorTA Optimizer

Tauri desktop app for brute-force double moving-average optimization (and future strategy expansion).

## Run

From the repo root:

```powershell
cd ta_desktop_demo
cargo run -p ta_desktop_demo_app --release
```

### GPU backends

GPU MA sweep (does not require `nvcc` - uses `vector-ta` prebuilt PTX):

```powershell
cargo run -p ta_desktop_demo_app --release --features cuda
```

GPU kernel backend (VRAM-resident, fastest). Does **not** require `nvcc` when using the prebuilt PTX:

```powershell
cargo run -p ta_desktop_demo_app --release --features cuda,cuda-backtest-kernel
```

Notes:
- GPU kernel backend currently supports `sma`, `ema`, `wma`, and `alma` for the MA computation.
- Prebuilt PTX control (backtest kernel):
  - Force prebuilt PTX (recommended for CI/packaging): `$env:VECTORBT_USE_PREBUILT_PTX="1"`
  - Require `nvcc` (developer mode): `$env:VECTORBT_REQUIRE_NVCC="1"`
