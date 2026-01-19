# VRAM-Resident GPU Kernel Refactor (MA Generalization)

_Last updated: 2026-01-17_

This document tracks the work to make the **GPU-kernel (VRAM-resident)** backend support **many/all CUDA-fast moving averages** without hardcoding each MA in `ta_app_core`.

It complements `ta_desktop_demo/PROGRESS.md` (which tracks the initial "VectorBT-style demo" buildout).

---

## Goal

- Keep the "showcase" backend: **copy candles once to VRAM**, compute indicators + backtest **entirely on GPU**, and only copy **reduced results** back (best / top-K / heatmap / optional export).
- Expand MA coverage so that most MAs that are faster on CUDA than CPU can run **fully VRAM-resident**.
- Avoid per-MA bespoke code in `ta_app_core` (minimize match-hardcoding and one-off buffers).

---

## Current State

- `ta_app_core` GPU-kernel backend computes:
  - MA matrices on GPU (selector-driven; see `ta_desktop_demo/crates/app_core/src/vram_ma.rs`)
  - transpose to time-major on GPU
  - backtest + metrics + reduction on GPU
  - host copies only for reduced outputs (or export)
- `ta_app_core` GPU "MA sweep" backend supports broad MA coverage but **copies MA matrices back to host** for CPU evaluation (not end-to-end VRAM-resident).

---

## Refactor Strategy (High-Level)

### 1) Add a "device-input MA selector" API in VectorTA

Expose a single entry point that can compute **MA sweeps directly from `DeviceBuffer<f32>` input** (already-resident prices), with the same param-map schema used by the UI.

Status: **done** (`VramMaComputer` lives in `vector-ta` under `src/cuda/moving_averages/vram_ma.rs`, and `ta_app_core` re-exports it).

### 2) Use it in `ta_app_core` GPU-kernel backend

Status: **done** (the GPU-kernel backend uses `VramMaComputer` instead of hardcoded MA branches).

### 3) Capability-aware behavior

Status: **done** for the current price-only VRAM-kernel MA set (unsupported MAs fall back to GPU MA sweep or CPU, per backend selection rules).

---

## Work Items / Checklist

### P0 - Make GPU-kernel MA selection modular (price-only first)

- [x] Define "VRAM-resident MA capability" rules (price-only vs needs candles).
- [x] Add device-input MA selector API in the `vector-ta` crate (so the standalone app repo can call it directly).
- [x] Implement device-input paths for a first wave of CUDA-fast, price-only MAs (VRAM-resident in GPU-kernel path):
  - Implemented: `sma`, `ema`, `wma`, `alma`, `dema`, `tema`, `jsa`, `smma`, `sqwma`, `highpass`, `swma`, `trima`, `sinwma`, `epma`, `wilders`, `maaq`, `mwdx`, `cwma`, `fwma`, `pwma`, `srwma`, `supersmoother`, `supersmoother_3_pole`, `zlema`, `nma`, `hma`, `jma`, `edcf`, `ehlers_itrend`.
  - Implemented (speedup-table set): `cora_wave`.
  - Still missing from the speedup table set: `hwma`.
- [x] Wire `ta_app_core` GPU-kernel backend to use selector-driven MA compute (no per-MA branches in the backend itself).
- [x] Ensure no extra large host<->device copies beyond initial upload + reduced result readback/export (small param arrays still upload per run/tile).

### P1 - Candle/volume/OHLC-required MAs (still VRAM-resident)

- [x] Decide and document which fields are uploaded to VRAM (as needed): `close` (always), `ma_source` prices (only if a selected MA uses source), plus `volume` (VWMA), and `high/low` (FRAMA).
- [x] Extend device-input selector to accept candles on device (`VramMaInputs`) and implement VRAM-kernel support for: `vwma`, `vpwma`, `frama`.
- [x] Add UI validation for MAs requiring volume/OHLC when CSV lacks those fields.

### P2 - Performance work (after correctness)

- [x] Reuse/persist device buffers across tiles (avoid cudaMalloc churn for MA matrices / transpose buffers).
  - Implemented via `KernelScratch` stored in `CudaKernelRuntime` (`ta_app_core`).
  - Opt-out: set `VECTORBT_KERNEL_PERSIST_BUFFERS=0` to release buffers after each run.
- [x] Cache per-period constants on device where beneficial (e.g., ALMA weights; weight windows for weighted MAs).
- [ ] Keep a small "fast path" (ALMA/SMA only) only if it's measurably faster than the generic selector path.

---

## Correctness + Benchmarking

### Unit tests (required)

- [x] Add parity tests for at least 3 new MAs (GPU-kernel vs CPU best/topK/heatmap match):
  - `jsa`, `cwma`, `cora_wave`, `epma`, `pwma`, `srwma`, `supersmoother`, `zlema`, `nma`, `hma`, `jma`, `edcf`, `ehlers_itrend` parity tests are in `ta_desktop_demo/crates/app_core/src/double_ma.rs`.
- [x] Keep existing "batched vs unbatched VRAM budget" parity coverage and extend it to at least one new MA.

### Benchmarks (required)

- [x] Extend `ta_app_core` benches to include GPU-kernel for CUDA-fast MAs beyond ALMA/SMA:
  - `double_ma/gpu_kernel_extra/*` in `ta_desktop_demo/crates/app_core/benches/double_ma_opt.rs`.
- [ ] Compare old hardcoded vs new selector-driven path (stretch goal; only possible if we keep both paths around).
- [x] Keep forced VRAM budget batched cases (via `VECTORBT_KERNEL_VRAM_BUDGET_MB`).

---

## Notes / Non-goals (for this refactor)

- This refactor does **not** add new strategies beyond double-MA crossover.
- We prioritize **VRAM-resident performance** over adding "many strategies" right now.
- We do not run `cargo fmt` automatically in this repo.
