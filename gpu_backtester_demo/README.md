# GPU Backtester Demo (ALMA only)

This is a standalone Rust binary that demonstrates a GPU-only, tiled, brute-force optimizer for a double-crossover MA strategy using ALMA kernels.

> Note: This crate is kept mostly for reference. The maintained demo is the Tauri GUI app in `ta_desktop_demo`.

To run the GUI demo:

```
cd ta_desktop_demo
cargo run -p ta_desktop_demo_app --release --features cuda,cuda-backtest-kernel
```

- Loads a single price series to GPU once.
- Computes ALMA tiles for fast and slow parameter grids on GPU.
- Runs a double-crossover backtest kernel entirely on GPU, writing per-pair metrics.
- Tiles across the fast×slow grid to stay under VRAM; optional global device result.

Build and run (requires CUDA toolkit `nvcc` and a CUDA-capable GPU):

```
cd gpu_backtester_demo
cargo run --release -- --synth-len 200000 --fast-period 5:50:1 --slow-period 20:200:5 --metrics 5
```

CSV input example:

```
cargo run --release -- \
  --csv ../data/eurusd.csv --column close \
  --fast-period 5:50:1 --slow-period 20:200:5 \
  --offset 0.85 --sigma 6.0 --fee 0.0005
```

Options:
- `--fast-period start:end:step` and `--slow-period start:end:step`
- `--offset`, `--sigma` shared for both ALMAs
- `--commission` fraction per side (applied on entry/exit; alias `--fee` kept)
- `--fast-tile`, `--slow-tile` to force tile sizes (auto if 0)
- `--metrics` number of metrics per pair (fixed layout: total_return, trades, max_dd, mean_ret, std_ret)

Notes:
- Uses ALMA CUDA kernels from the parent crate (`my_project`), and a demo backtest CUDA kernel compiled here.
- Keeps device buffers in a single CUDA context established by `CudaAlma`.
- If the global result buffer does not fit in VRAM, results are copied back per tile and scattered on host.

Env:
- `CUDA_ARCH` overrides the PTX arch (default compute_89).
- You can disable VRAM guardrails by omitting, though tiling mitigates memory pressure.

Windows note:
- `nvcc` also needs MSVC (`cl.exe`). Easiest is running from a “Developer PowerShell for VS” / “x64 Native Tools Command Prompt”.

Future work:
- Multi-stream overlap between ALMA tile compute and backtest evaluates.
- Support other MA types (EMA/SMA) and fused compute+backtest paths for light MAs.
- Python binding that accepts device arrays (CuPy/PyTorch) via DLPack.
