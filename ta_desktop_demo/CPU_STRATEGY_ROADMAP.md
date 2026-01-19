# CPU Composite Strategy Roadmap (HalfTrend + MACZ + OTT + MA Filters)

_Last updated: 2026-01-19_

This roadmap adds a **second strategy** to the Tauri desktop optimizer:

1. **Strategy A (existing):** Double Moving Average crossover (CPU / GPU sweep / GPU kernel).
2. **Strategy B (new):** A **CPU-only composite strategy** that combines:
   - **HalfTrend** regime filter (trend state)
   - **MACZ** (momentum / normalized MACD-style signal)
   - **OTT** (trend-following stop/trigger line)
   - **One or more MA crossovers** (e.g., JMA/ALMA/HMA pairs), optionally plus light filters (ROC/RSI).

The key idea is to keep Strategy B CPU-only, but **use a preliminary GPU double-MA optimization pass** to
reduce the MA parameter search space before running the more complex CPU composite optimization.

> Note on “web search”: this environment does not provide an external web-browsing tool.
> The plan below is based on the existing codebase + common evolutionary optimization practice.
> If we want citations/links, we can add them later from a human-curated source list.

---

## Goals

- Keep the app’s “speed showcase” intact (GPU-kernel double MA crossover; SMA/ALMA fast path supported).
- Add a second “realistic” strategy type (HalfTrend/MACZ/OTT + MA filters) that is **CPU-only**.
- Replace brute-force grid search for high-dimensional strategies with a **budgeted, efficient optimizer**
  (genetic / evolutionary), with deterministic seeds and reproducible runs.
- Use GPU only where it’s already strongest:
  - **Stage 0:** fast MA-pair pre-scan with Strategy A (GPU kernel) to narrow MA periods/types used in Strategy B.
  - **Stage 1:** CPU composite optimization within the narrowed space.

Non-goals (for this roadmap):
- Making Strategy B VRAM-resident end-to-end on GPU (that is a separate project).
- Adding many unrelated strategies; we focus on the composite one first.

---

## Current State (relevant capabilities)

- Indicators implemented in Rust (core crate `vector-ta`):
  - `halftrend` (`src/indicators/halftrend.rs`)
  - `macz` (`src/indicators/macz.rs`)
  - `ott` (`src/indicators/ott.rs`)
  - many MAs, including `jma`, `alma`, `hma`, etc.
- Desktop app currently supports only the **double MA crossover** strategy (`ta_desktop_demo/crates/app_core/src/double_ma.rs`).
- Optimization modes:
  - `Grid` and `CoarseToFine` exist for the double-MA strategy.
  - No general “budgeted” optimizer for mixed parameter types exists yet.

---

## Strategy B: Proposed Spec (CPU-only)

### Inputs

- Candles (CSV): at minimum `close`; for most variants we’ll want `high/low` (HalfTrend, HLCC4 source), and
  optionally `volume` (MACZ volume variant if desired).
- Price source selection: likely start with `hlcc4` (to match the example strategy) but allow `close`.

### Core signal structure (starting point)

We will implement a Strategy B baseline that matches the “shape” of the provided Jesse strategy:

- **Regime filter:** HalfTrend trend state gates long vs short bias.
- **Momentum filter:** MACZ histogram (or signal) gate; optional.
- **Trend trigger:** OTT “signal” (price vs ott line / sign); optional.
- **MA filter block:** one or more MA comparisons/crossovers:
  - JMA(fast) vs JMA(slow)
  - ALMA(fast) vs ALMA(slow)
  - HMA(fast) vs HMA(slow)
- Optional light filters: ROC slope, RSI threshold bands.

We should keep the first implementation minimal:
- Start with **HalfTrend + one MA pair** (e.g., JMA or ALMA).
- Add MACZ + OTT as optional toggles once the core architecture is stable.

### Parameter constraints (important)

To avoid nonsense combinations and speed up search, enforce constraints at the optimizer level:
- For each MA pair: `fast_period < slow_period` (or reject/repair).
- RSI band: `rsi_short_threshold < rsi_long_threshold` (creates a neutral zone).
- Period ranges clamped to valid indicator requirements (warmup).

---

## Two-Stage Optimization Pipeline

### Stage 0 (GPU): “MA Pre-Scan” using Strategy A

Run one or more GPU-kernel double-MA optimizations to identify promising MA periods for Strategy B’s MA block.

**Example flow**
- Choose a dataset (same CSV).
- For each MA family we plan to use in Strategy B (e.g., `jma`, `alma`, `hma`):
  - Run double-MA optimization with `fast_ma = slow_ma = <ma>`
  - Output top-K `(fast_period, slow_period)` pairs (and score metrics).
- Convert the top-K set into a narrower search space for Strategy B:
  - Option A: “Candidate set” (discrete list of allowed periods for fast/slow)
  - Option B: “Narrowed ranges” (min/max around top results + margin)

**Why this helps**
- It converts a huge MA-period grid into a small discrete set or narrow bands, freeing CPU budget for the
  high-dimensional indicator mix.

### Stage 1 (CPU): Composite Strategy optimization (Genetic / Evolutionary)

Optimize the full composite parameter vector, but with MA periods/types constrained by Stage 0 output.

Key requirement: an **anytime** optimizer:
- Runs for a fixed time or eval budget.
- Streams best-so-far and top-K to the UI.
- Deterministic with a known seed (for reproducibility).

---

## Optimizer Choice: “Best Practical Genetic Optimizer” for This App

We need something that handles **mixed discrete + continuous** parameters efficiently, without bringing in
heavy dependencies.

Recommended approach (single-objective first):

- **Population-based evolutionary optimizer** with:
  - elitism (keep best N),
  - tournament selection,
  - crossover:
    - int/discrete: uniform or 1-point crossover
    - float: blend crossover (or SBX-style “real-coded” crossover)
  - mutation:
    - int: +/- k steps with clamp
    - float: Gaussian-like perturbation or step-based perturbation with clamp
    - categorical: random switch among allowed values
  - constraint repair (fast<slow, RSI band, etc.)
- Add an optional **local refinement** step (“memetic”):
  - every M generations, take top few and do small stepwise neighborhood search.

Multi-objective (later / optional):
- If we want to optimize for Sharpe + drawdown simultaneously, add a Pareto mode
  (NSGA-II-style ranking/crowding). This is a second phase; start with a single objective to ship sooner.

---

## Implementation Plan (Milestones)

### P0 — Design + Types (plumbing)

- [ ] Add `StrategyKind` to the app request model (e.g., `DoubleMa`, `CompositeCpu`).
- [ ] Define `CompositeCpuRequest` and `CompositeCpuResult` in `ta_desktop_demo/crates/app_core`.
- [ ] Decide the minimal Strategy B parameter schema (v1):
  - required: HalfTrend amplitude + one MA pair
  - optional: RSI/ROC filters, MACZ, OTT toggles
- [ ] Extend run history serialization to store strategy kind + params + result.

### P1 — CPU Backtester for Strategy B (correctness first)

- [ ] Implement a CPU backtest loop for Strategy B (position state machine + metrics).
- [ ] Implement indicator evaluation for Strategy B with minimal allocations:
  - precompute source series once (e.g., `hlcc4`)
  - reuse buffers where possible
- [ ] Add unit tests:
  - deterministic candles fixture
  - strategy invariants (no NaN explosions, warmup correctness)
  - sanity checks on trade counts and metric sign

### P2 — Budgeted Optimizer (Genetic) in `ta_desktop_demo/crates/optimizer`

- [ ] Add `OptimizationMode::Genetic` (and plumb it through UI + app_core).
- [ ] Implement a generic “param schema + genome” layer that supports:
  - ints, floats, bools, enums/discrete sets
  - constraints (repair + rejection)
- [ ] Implement evolutionary loop with:
  - deterministic RNG seeded from request
  - rayon-parallel evaluation
  - `top_k` aggregation + best tracking (reuse the `StreamAggregator` pattern)
  - cancellation support + progress events
- [ ] Add “evaluation budget” and “time budget” controls.

### P3 — GPU Pre-Scan integration (Stage 0)

- [ ] Add a backend helper: `run_ma_prescan_gpu()` that runs Strategy A GPU-kernel multiple times (per MA family).
- [ ] Add a strategy-specific reducer that converts Stage 0 outputs into constrained MA domains:
  - candidate sets or narrowed min/max ranges
- [ ] Wire to Strategy B optimizer:
  - if enabled, run Stage 0 first, then run Strategy B with constraints
- [ ] Persist Stage 0 artifacts in the result (for transparency/repro).

### P4 — UI/UX integration in the Tauri app

- [ ] Add a “Strategy” selector in the UI: `Double MA (GPU/CPU)` vs `Composite (CPU)`.
- [ ] For composite strategy:
  - show parameter schema editor (ranges + toggles)
  - show optimization budget controls (time/evals/population)
  - show Stage 0 “GPU pre-scan” toggle and summary of the narrowed MA space
- [ ] Stream progress:
  - Stage 0 progress (per MA family)
  - Stage 1 progress (generation + eval count + best score)
- [ ] Add exports:
  - best params JSON
  - top-K CSV
  - optional full run archive (settings + results)

### P5 — Validation + Benchmarking

- [ ] Compare Strategy B outputs against a small python reference implementation (optional, off-repo) or sanity tests.
- [ ] Add CPU performance benchmarks for Strategy B evaluation throughput (evals/sec) at 10k/100k/1M candles.
- [ ] Add profiling notes + hotspots:
  - indicator compute vs backtest loop
  - allocations to eliminate

---

## Data Requirements / Validation Rules (UI + backend)

- HalfTrend requires `high`, `low`, `close`.
- Many MA sources (HLCC4/HL2/OHLC4) require `high/low/open/close`.
- MACZ may optionally use `volume` (decide whether Strategy B requires it; if optional, degrade gracefully).
- OTT requires at least the selected MA source series.

The UI should prevent “Run” when required fields are missing, with clear messaging.

---

## Open Questions (decisions needed early)

1. **Exact Strategy B logic**
   - Do we replicate the Jesse boolean logic exactly, or define a cleaner “signal graph” with toggles?
2. **Objective**
   - Sharpe vs PnL vs max drawdown; do we add composite objectives (risk-adjusted)?
3. **Out-of-sample evaluation**
   - Train/test split vs walk-forward; when do we require it?
4. **MA families for Stage 0**
   - Which MAs do we pre-scan? (JMA/ALMA/HMA only, or allow user selection?)
5. **Budget defaults**
   - Typical “interactive” run: e.g., 30–120s, population 128–512, eval budget 10k–100k (depends on candles).

---

## Stretch Goals

- [ ] Add multi-objective Pareto optimization (e.g., Sharpe + drawdown) and a Pareto front visualization.
- [ ] Add resume/restart: persist population state and continue a run later.
- [ ] Add “portfolio” optimization across multiple CSVs (multi-market robustness).
- [ ] Consider a GPU-assisted partial evaluation for Strategy B (e.g., MA block on GPU) while keeping backtest CPU.

