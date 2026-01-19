# VectorTA Optimizer Progress Tracker

_Last updated: 2026-01-17_

This document tracks what is **done**, what is **in progress**, and what is **missing** for the "VectorBT-style" **double moving-average (double-MA) brute-force optimizer** desktop demo.

For the longer-form plan and design notes, see `../VECTORBT_DESKTOP_ROADMAP.md`.

---

## Product Goal (North Star)

- A high-end, professional desktop app that brute-forces a fixed **double-MA crossover** strategy over large parameter grids.
- First-class **VRAM-resident GPU** execution: copy market data to GPU once, keep intermediate indicator/backtest data on GPU, and only copy **reduced results** back (best + top-K + optional heatmap).
- CPU path remains competitive and is used for compatibility, correctness comparisons, and when GPU is unavailable.
- The UI exposes **all moving averages** available in the `vector-ta` crate, with capability-aware fallbacks (CPU-only / CUDA MA sweep / CUDA kernel).

---

## Current State (What Works Today)

### End-to-end GUI

- [x] Tauri app launches and runs locally on Windows.
- [x] Load CSV candles into app state (`load_price_data`) and get a `data_id`.
- [x] Run optimization (`run_double_ma_optimization`) with:
  - backend selection: Auto / CPU / GPU (MA sweep) / GPU (kernel)
  - fast/slow period ranges (enforces `fast < slow`)
  - MA selection dropdowns (from a registry)
  - strategy knobs: commission, epsilon band, long-only, allow flip, trade on next bar
  - objective selection: PnL / Sharpe / MaxDrawdown
  - `top_k`, `include_all` (guarded), and binned heatmap (`heatmap_bins`)
  - per-pair metrics CSV export (`export_all_csv_path`, GPU-kernel only; requires `include_all=false`)
- [x] Progress events in UI via `double_ma_progress` + cancel button.
- [x] Results UI:
  - best summary
  - sortable top-K table
  - drilldown: equity + drawdown curves + trade stats/trades (best + selected top-K row)
  - optional binned heatmap canvas (objective score)
  - export JSON + export CSV
  - local run history (localStorage)

### Core logic and testability

- [x] Optimization logic extracted to `ta_desktop_demo/crates/app_core` (no WebView dependency), enabling unit tests + Criterion benchmarks.
- [x] CLI runner exists for headless runs: `ta_desktop_demo/crates/app_core/src/bin/ta_cli.rs`.

### Backends

- [x] **Auto backend**: picks GPU (kernel) → GPU (MA sweep) → CPU based on build flags, CUDA availability, and MA capabilities.
- [x] **CPU backend** (baseline): tiled execution with top-K reduction and optional heatmap.
- [x] **GPU "MA sweep" backend**: moving averages generated via CUDA selector, results evaluated and reduced on CPU.
- [x] **GPU "kernel" backend** (**VRAM-resident per tile**): moving averages + backtest computed on GPU for each tile, then reduced (top-K / heatmap) and copied back.
  - Current GPU kernel MA support (VRAM-resident): `alma`, `sma`, `ema`, `wma`, `dema`, `tema`, `jsa`, `smma`, `sqwma`, `highpass`, `swma`, `trima`, `sinwma`, `epma`, `wilders`, `maaq`, `mwdx`, `cwma`, `cora_wave`, `fwma`, `pwma`, `srwma`, `supersmoother`, `supersmoother_3_pole`, `zlema`, `nma`, `hma`, `jma`, `edcf`, `ehlers_itrend`, `vwma`, `vpwma`, `frama`.
  - VRAM-budget override for forcing batching: `VECTORBT_KERNEL_VRAM_BUDGET_MB`.

---

## Test Status (Known-Good Commands)

- [x] Root CUDA test suite: `cargo test --features cuda`
- [x] App workspace tests: `cargo test --manifest-path ta_desktop_demo/Cargo.toml`
- [x] App core + GPU kernel tests: `cargo test --manifest-path ta_desktop_demo/Cargo.toml -p ta_app_core --features cuda-backtest-kernel`

Notes:
- CUDA test builds include PTX compilation via the repo build scripts.
- GPU-kernel batching correctness has parity tests (batched vs unbatched best+heatmap) under `ta_app_core` (SMA + ALMA + JSA + CWMA + CoRa Wave + EPMA + PWMA + SRWMA + SuperSmoother + ZLEMA + NMA + HMA + JMA + EDCF + Ehlers ITrend + VWMA + VPWMA + FRAMA).

---

## Benchmarks Snapshot (Baseline)

**Machine**
- GPU: NVIDIA GeForce RTX 4090 (24GB) | Driver `591.74` | CUDA `13.1`
- CPU: AMD Ryzen 9 9950X (16c/32t)
- RAM: ~96GB
- Git: `c498fd638` (short hash)

**Command**
```powershell
cargo bench --manifest-path ta_desktop_demo/Cargo.toml -p ta_app_core --bench double_ma_opt --features cuda-backtest-kernel -- --noplot
```

**Results (200k candles, 58,300 fast<slow pairs in "large")**

- CPU (SMA/SMA): ~`2.60s`
- CPU (ALMA/ALMA): ~`2.71s`
- GPU MA sweep (SMA/SMA): ~`2.84s`
- GPU MA sweep (ALMA/ALMA): ~`2.98s`
- GPU kernel (SMA/SMA): ~`0.173s`
- GPU kernel (ALMA/ALMA): ~`0.223s`
- GPU kernel forced batched @ ~1GB budget (2026-01-16 run):
  - `large_batched_1gb` SMA/SMA: ~`0.721s`, ALMA/ALMA: ~`0.900s`
  - `xlarge_batched_1gb` SMA/SMA: ~`2.464s`, ALMA/ALMA: ~`2.858s`

**GPU-kernel (2026-01-16 local run, Sharpe objective, 200k/large/58,300 pairs)**
- SMA/SMA: ~`0.186s`, ALMA/ALMA: ~`0.189s`
- JSA/JSA: ~`0.178s`, CWMA/CWMA: ~`0.182s`, EPMA/EPMA: ~`0.198s`
- CoRa Wave/CoRa Wave: ~`0.452s` (`large_batched_1gb`: ~`3.12s`)
- NMA/NMA: ~`0.185s` (`large_batched_1gb`: ~`0.601s`)
- SuperSmoother/SuperSmoother: ~`0.233s`, ZLEMA/ZLEMA: ~`0.231s`
- `large_batched_1gb` (58,300 pairs): JSA/JSA ~`0.571s`, CWMA/CWMA ~`0.580s`, EPMA/EPMA ~`0.618s`, SuperSmoother/SuperSmoother ~`0.822s`, ZLEMA/ZLEMA ~`0.985s`

**Extra GPU-kernel scale checks (200k candles)**
- `xlarge` (370,750 pairs): SMA/SMA ~`2.39s`, ALMA/ALMA ~`2.56s`
- `xxlarge` (1,491,500 pairs): SMA/SMA ~`8.77s`, ALMA/ALMA ~`9.07s`

Interpretation:
- The **GPU kernel** path is already the performance "showcase" backend.
- ALMA is a good headline demo MA because it's materially more complex than SMA but performs similarly in the GPU-kernel path.

---

## Known Limitations / Gaps

### MA coverage and MA parameters

- [x] Dynamic per-MA parameter forms in the UI (schema-driven; parametric MAs include **ALMA**, Gaussian, JMA, etc).
- [x] Full parameter plumbing for parametric MAs through CPU + CUDA MA sweep.

### GPU kernel backend generalization

- [x] Expand GPU-kernel MA support beyond `sma/ema/wma/alma` (or add capability-aware fallback to GPU MA sweep / CPU).
- [x] Make the GPU-kernel backend consume the same "MA param schema" used by the UI (not special-cased fields).
- [x] Prebuilt PTX distribution for the backtest kernel (no `nvcc` required if using the prebuilt PTX):
  - `VECTORBT_USE_PREBUILT_PTX=1` forces using `src/kernels/double_crossover.prebuilt.ptx`.
  - `VECTORBT_REQUIRE_NVCC=1` fails the build if `nvcc` is missing/fails (developer mode).

### Post-run analysis depth ("VectorBT rival" features)

Current UX is "best + top-K + heatmap + curves". Missing:
- [x] Trade list and trade statistics (win rate, avg win/loss, profit factor, expectancy).
- [x] Slice-and-dice views: filter/sort by multiple metrics, compare runs, show Pareto front.

### Optimization modes / search

- [x] "Auto" mode now resolves to Grid or Coarse→Fine (adaptive) depending on grid size and output flags.
- [x] Additional optimization modes (optional, later): coarse→fine (implemented); random/Bayesian/halving are future work.

---

## Next Work (Priority Order)

### P0 (immediate, aligns with "VRAM-resident is the ideal path")

- [x] Make MA selection + params fully registry-driven end-to-end (UI → request → CPU/GPU backends).
- [x] Add ALMA batched-vs-unbatched parity test (match current SMA parity coverage).
- [x] Add "selected row drilldown" minimal analysis: equity curve + drawdown for best and for a clicked top-K row.
- [x] Make the GPU-kernel backend optionally output "per-pair metrics" in a controlled way (chunked export / streaming) without blowing RAM.

### P1 (performance + scalability)

- [x] VRAM-aware tile planner improvements (better model, better defaults).
- [x] Reduce overhead in the GPU-kernel path (less host synchronization; fewer per-tile allocations/copies).
- [x] Add more benchmark scenarios (larger grids, forced VRAM batching).
- [x] Further GPU-kernel perf work: persistent device buffers across tiles and reduced cudaMalloc churn.

### P2 (product polish)

- [x] Saved presets and reproducibility: export/import run configs; stable run IDs; "rerun this run".
- [x] Packaging: Windows installer story, GPU feature gating, and clear "works without CUDA toolkit" default build.
- [x] UI design polish (VectorBT-like usability): better layout, validation messaging, progress details, settings persistence.

---

## Useful Dev Commands (Windows PowerShell)

Run Tauri app:
```powershell
cd ta_desktop_demo
cargo run -p ta_desktop_demo_app --release
```

Run Tauri app with GPU backends:
```powershell
cd ta_desktop_demo
cargo run -p ta_desktop_demo_app --release --features cuda
cargo run -p ta_desktop_demo_app --release --features cuda,cuda-backtest-kernel
```

Run CLI (headless):
```powershell
cd ta_desktop_demo
cargo run -p ta_app_core --bin ta_cli --release --features cuda-backtest-kernel,cli -- --backend auto --fast-ma alma --slow-ma alma --fast-period 5:200:1 --slow-period 10:400:1 --synth-len 200000
```

Run CLI with per-pair export (GPU kernel only):
```powershell
cd ta_desktop_demo
cargo run -p ta_app_core --bin ta_cli --release --features cuda-backtest-kernel,cli -- --backend gpu-kernel --fast-ma alma --slow-ma alma --fast-period 5:200:1 --slow-period 10:400:1 --synth-len 200000 --export-all-csv-path "C:\\tmp\\all_pairs.csv"
```

Force GPU-kernel batching via VRAM budget:
```powershell
$env:VECTORBT_KERNEL_VRAM_BUDGET_MB="1024"
```

Disable persistent GPU-kernel scratch buffers (frees VRAM after each run):
```powershell
$env:VECTORBT_KERNEL_PERSIST_BUFFERS="0"
```
