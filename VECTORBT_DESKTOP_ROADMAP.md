# VectorBT Desktop Demo (ta_desktop_demo) — Feature-Complete Roadmap

This document outlines the work needed to make `ta_desktop_demo` a “VectorBT”-style, **fully feature complete** desktop backtest/optimization demo that:

- Uses a **fixed double moving-average (double‑MA) strategy**.
- Lets the user pick **any moving average available in the VectorTA (`vector-ta`) crate**, plus MA‑specific parameters.
- Runs a **brute-force grid search** efficiently by evaluating many parameter combinations in a **vectorized** way.
- Supports both CPU and GPU execution, with the GPU path keeping data **VRAM resident** and only copying **final results** back to host.
- Automatically **tiles / chunks** the search space to fit within RAM/VRAM budgets.

> Repo note: today the app and the library are in one repo for convenience. The long‑term intent is to move the desktop app into its own repo and depend on `vector-ta` as an external crate. This plan assumes we keep the boundary clean (the app uses public APIs).

---

## 0) Current State (What’s Already Working)

### App (`ta_desktop_demo`)
- Tauri GUI with commands:
  - `load_price_data` (CSV -> candles stored in state)
  - `run_double_ma_optimization` (tiled grid search; returns best + top-K; emits `double_ma_progress` events)
  - `cancel_double_ma_optimization` (cooperative cancel; stops between tiles)
- CPU backend:
  - Tiled MA generation via `vector-ta` CPU batch dispatch (`ma_batch`) with per‑period fallback
  - Streams results into a top‑K reducer (no full-grid matrices required)
- GPU "MA sweep" backend:
  - Uses `vector-ta` CUDA MA selector sweep (`CudaMaSelector::ma_sweep_to_host_f32`) to generate MA tiles on GPU, then evaluates on CPU
  - Streams results into a top‑K reducer (no full-grid matrices required)
- GPU kernel backend:
  - Optional `cuda-backtest-kernel` feature builds a demo backtest kernel PTX and runs a tiled VRAM‑resident path
  - Streams results directly into a top‑K reducer (supports `sma`, `ema`, `wma`, `alma` for MA computation)

- UI improvements:
  - MA dropdowns are populated from the library MA registry (with capability flags)
  - Frontend validates backend/MA compatibility and shows progress from `double_ma_progress`
  - Strategy controls (commission, epsilon band, flip/long-only/trade-on-next flags)
  - Post-run analysis: best summary, sortable top-K table, binned objective heatmap, CSV/JSON export, local run history + settings persistence

### Library (`vector-ta`)
- CPU MA single dispatch: `ma()` (`src/indicators/moving_averages/ma.rs`)
- CPU MA period-sweep batch dispatch: `ma_batch()` (`src/indicators/moving_averages/ma_batch.rs`)
- MA registry: `list_moving_averages()` (`src/indicators/moving_averages/registry.rs`)
- CUDA MA selector:
  - Single: `CudaMaSelector::ma_to_host_f32`
  - Period sweep: `CudaMaSelector::ma_sweep_to_host_f32`
  - Device outputs: `ma_to_device`, `ma_sweep_to_device` (building blocks for VRAM-resident flows)

---

## 1) Define “Feature Complete” (Acceptance Criteria)

### Strategy & Optimization
- Fixed strategy: double‑MA crossover (fast vs slow) with a clear ruleset:
  - signal definition (fast > slow, optional epsilon band)
  - trade timing (same bar vs next bar)
  - long-only / allow short / allow flip
  - fees/commission/slippage
- Brute-force optimizer:
  - grid search across parameter ranges
  - supports **huge search spaces** via tiling
  - supports objectives (PnL/Sharpe/Drawdown/etc.)
  - returns best + top‑K and/or summary statistics

### MA Coverage
- UI exposes **all MAs supported by `vector-ta`** (with truthful “capabilities” flags).
- For each MA type, user can configure **MA-specific parameters** (at least as fixed values; ranges optional).
- Clear behavior for “not supported” cases:
  - requires candles (high/low/volume) vs close-only
  - dual outputs (e.g., MAMA-like) and which series is used
  - not period-based (no period sweep) -> disable or provide specialized UI

### CPU Backend
- Vectorized / tiled evaluation:
  - does not allocate “full grid” matrices when the grid is larger than memory budget
  - avoids O(grid_size) temporary allocations beyond the tile

### GPU Backend
- VRAM-resident end-to-end path:
  - price + required candle fields copied once to device
  - per-tile MA computation stays on device
  - per-tile backtest runs on device
  - only reduced/top‑K (or per‑combo metrics, if requested) copied back
- Tile planner driven by VRAM budget + headroom
- Correctness parity vs CPU (within defined tolerances)

---

## 2) Unify the MA Abstraction (CPU + CUDA)

Today we have:
- CPU: `ma()` and `ma_batch()` dispatch functions
- CUDA: `CudaMaSelector` (single + period sweep)

To be “VectorBT feature complete”, the app should not have custom per‑MA branching. Instead, it should query a registry and call a unified interface.

### 2.1 Add a “MA Registry” in `vector-ta`
Create a single authoritative place that describes each MA:
- stable id / name (e.g., `"sma"`, `"alma"`, …)
- display name
- required inputs:
  - `prices-only`, `candles(OHLC)`, `candles(volume)`, etc.
- parameter schema:
  - `period` (range-able)
  - extra params (offset, sigma, gain, power, poles, etc.)
  - defaults and valid ranges
- capability flags:
  - CPU single supported
  - CPU period-sweep supported
  - CUDA single supported
  - CUDA period-sweep supported
  - CUDA **device-sweep** supported (required for VRAM-resident)
  - outputs: single series vs dual series (and how to select)

Expose this to the app as a JSON-serializable structure via a Tauri command like `list_moving_averages()`.

### 2.2 Add a “MA Params” representation (typed)
Define a stable Rust representation that can cross the app/library boundary:
- `MaKind` enum (or string)
- `MaParams` enum keyed by kind, containing typed fields
- `MaSweep` representation for ranges:
  - minimum required: allow period range
  - optional: allow extra param ranges (future)

The app should store and pass MA selections as:
```
FastMa = { kind, params, period_range }
SlowMa = { kind, params, period_range }
```

### 2.3 Extend CPU batch dispatch to accept MA-specific params
`ma_batch()` currently uses defaults for non-period params for many MAs.
Add a parallel API that takes typed params, e.g.:
- `ma_batch_params(kind, data, period_range, params, kernel)`

Internally, it should call the indicator’s batch function with the user’s params (not defaults).

### 2.4 Extend CUDA selector to accept MA-specific params
Similarly, extend `CudaMaSelector` to accept parameters:
- `ma_to_device(kind, data, period, params)`
- `ma_sweep_to_device(kind, data, start, end, step, params)`

This prevents the app from having to “know” which MA needs which extra knobs.

> Practical compromise: start with “extra params are fixed values” (not swept). Sweeping extra params multiplies grid size quickly and should be added later with explicit guardrails.

---

## 3) Strategy Engine: Make CPU and GPU Evaluate the *Same* Rules

Right now, different paths compute different metrics and trade rules.
To be feature complete:

### 3.1 Define a canonical strategy spec
Create a shared strategy spec used by all backends:
- flags: long-only, allow flip, trade-on-next, enforce fast < slow
- fees/commission/slippage model
- epsilon band (`eps_rel`, `eps_abs`)
- output metrics definition

### 3.2 Define a canonical metrics struct
Pick the metrics set (and keep consistent across backends), e.g.:
- PnL / total return
- max drawdown
- Sharpe (define exact formula)
- trade count
- optionally exposure stats

### 3.3 Make CPU compute the same metrics
Implement a CPU evaluation kernel that matches GPU behavior:
- precompute returns/log-returns once
- evaluate many parameter pairs in a tile-friendly loop
- produce identical metrics (within floating-point expectations)

---

## 4) CPU Backend: Vectorized + Tiled Brute Force

### 4.1 RAM-aware tiling
Add a planner that:
- estimates bytes needed per MA matrix + working buffers + output
- chooses tile sizes (fast_tile × slow_tile) to fit within a RAM budget
- supports `top_k` reduction to avoid storing full-grid outputs

### 4.2 Compute MA matrices per tile, not per full grid
Instead of building MA matrices for the whole range and holding them:
- Build fast MA tile matrix (periods subset)
- Build slow MA tile matrix (periods subset)
- Evaluate all pairs in that tile
- Reduce / persist only results, then reuse buffers for the next tile

### 4.3 CPU vectorization
Once tiling is correct:
- Use time-major layout for cache locality (t-major MA matrices)
- Consider AVX2/AVX512 kernels for signal comparisons & return accumulation
- Keep per-combo state in SoA where possible

---

## 5) GPU Backend: VRAM-Resident End-to-End Pipeline

### 5.1 Device data model
Create a `DeviceMarketData` that can live for the whole optimization run:
- device buffers for close + other required fields (open/high/low/volume)
- computed returns/log-returns buffer
- optional “first_valid” indices per source

### 5.2 Device-resident MA sweeps
Use `CudaMaSelector::ma_sweep_to_device(...)` (once parameterized) to generate:
- MA matrix on device for a subset of periods (tile)
- repeat for fast and slow MA choices

### 5.3 Transpose and memory layout
Backtest kernels should read MA values in time-major form:
- maintain a device transpose kernel
- keep transposed buffers for the current tile only

### 5.4 Backtest kernel: generic for any MA type
The strategy kernel should only care about two MA time-major matrices + returns:
- inputs: `fast_ma_T`, `slow_ma_T`, `returns`, period arrays, first_valid, flags, etc.
- outputs: per-pair metrics buffer

### 5.5 VRAM-aware tiling planner
Implement a VRAM budget planner based on `mem_get_info()` + headroom:
- estimate bytes for:
  - input series
  - fast MA tile (rows×cols)
  - slow MA tile (rows×cols)
  - transpose outputs
  - output metrics tile
  - scratch (weights, etc.)
- choose tile sizes dynamically
- if the full results matrix is too large, copy results per tile and aggregate on host

### 5.6 Overlap compute + transfer (performance)
Once correctness is stable:
- use multiple CUDA streams
- pipeline:
  - compute MA tile N+1 while backtesting tile N
  - copy results for tile N while computing tile N+1

### 5.7 GPU correctness testing
Add tests that:
- run a small grid on CPU and GPU
- assert metrics are equal (or within tolerance), without changing existing reference vectors

---

## 6) UI/UX: Make MA Selection and Parameters “Real”

### 6.1 Dynamic form driven by MA registry
The UI should:
- render MA list from `list_moving_averages()`
- render MA-specific parameter inputs dynamically
- validate ranges (period and extra params) before running

### 6.2 Long-running job controls
Add:
- progress reporting (tiles completed / total)
- cancel button (cooperative cancellation)
- logging panel (backend used, tile sizes chosen, memory usage)

### 6.3 Results UX
Provide:
- sortable top‑K table
- export CSV/JSON
- optionally a heatmap (fast vs slow) for a chosen metric (computed per tile)

---

## 7) “All Moving Averages” Support Plan

Not all MAs are equally easy to support end-to-end on GPU. Plan for tiers:

### Tier A: prices-only, period-based, single output
Ideal for first “complete” pass across CPU+GPU.

### Tier B: candles-required (high/low/volume)
Requires device-resident candle fields and selector support for those MAs.

### Tier C: dual-output or special-output MAs
Needs an explicit “which output series?” choice in UI and backends.

### Tier D: non period-based or multi-parameter MAs
Require specialized sweep logic (or restrict optimization dimensions).

For each MA, the registry should truthfully state which tier it belongs to and which backends support it today.

---

## 8) Repo Split (Future)

When moving the desktop app into its own repo:
- `vector-ta` must expose only stable, public APIs used by the app:
  - MA registry + param schema
  - CPU MA selector/sweep API
  - CUDA MA selector/sweep-to-device API
  - helper utilities for candles parsing (or the app owns CSV parsing)
- decide how the demo backtest kernel PTX is shipped:
  - compile PTX in the app repo (requires nvcc for that feature), or
  - ship prebuilt PTX for the kernel like `vector-ta` does (preferred for end users)
- CI:
  - app builds without CUDA toolkit by default
  - optional CUDA jobs validate `--features cuda` and `--features cuda-backtest-kernel`

---

## 9) Suggested Milestones

### Milestone 1 (Feature-complete CPU + GPU MA sweep)
- MA registry + dynamic UI parameter forms
- CPU: tiled brute-force grid search (RAM budget)
- GPU: MA sweep acceleration + CPU backtest (current approach) but with the same strategy rules & metrics as CPU

### Milestone 2 (GPU VRAM-resident “VectorBT”)
- CUDA: ma_sweep_to_device + transpose + backtest kernel
- VRAM tiling + top‑K reduction
- correctness tests CPU vs GPU

### Milestone 3 (All MAs exposed + capability-aware)
- Tier A coverage end-to-end (CPU+GPU)
- Tier B/C with clear UI choices and fallbacks
- Tier D with explicit constraints / specialized handling

---

## 10) Open Questions (Decisions to Lock In)

Before implementing everything, confirm:
- Which strategy rule set is canonical (trade timing, fees, long/short)?
- Which metrics are required for “feature complete”?
- Do we optimize only **periods** for most MAs, with extra params fixed? (recommended initially)
- Do we require GPU results to match CPU bit-for-bit, or within a tolerance?
- Should the default distribution require `nvcc`? (ideally “no”, unless user opts in)
