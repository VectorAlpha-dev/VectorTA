# Repository Guidelines

## Context & References

- Read `CLAUDE.md` first for background and decisions to date.
- WASM and Python bindings are mostly working; SIMD exists for some moving averages. CUDA/PTX may follow after SIMD is complete.

## Project Structure

- `src/`: Core library. Indicators under `src/indicators/`; use `moving_averages/alma.rs` as the reference for API, docs, and style.
- `tests/`: Integration tests per indicator (e.g., `alma_*`). Keep scope tight to the module under test.
- `benches/`: Bench harnesses (e.g., `indicator_benchmark`).
- `kernels/`: Architecture‑specific work (SIMD/CUDA exploration). Wrap SIMD behind feature gates.

## Build & Check

- Stable: `cargo build` (must pass without `nightly-avx`).
- Nightly (AVX2/AVX512): `cargo +nightly build --all-features`.
- Quick checks: `cargo check`, `cargo check --features python`, `cargo check --features wasm`, `cargo check --features nightly-avx`.
- Toolchain: see `rust-toolchain.toml` (pinned `nightly-2025-05-28`; stable fallback).

## Test & Bench

- Scalar: `cargo test --lib indicators:: -- --nocapture`.
- Nightly AVX2: `cargo +nightly test --features nightly-avx --lib indicators:: -- --nocapture`.
- AVX512: `cargo +nightly test --features nightly-avx --lib indicators::_avx512 -- --nocapture`.
- Example (moving averages): `cargo test --features nightly-avx --lib indicators::moving_averages::tsi -- --nocapture`.
- All unit tests with SIMD: `cargo test --features nightly-avx`.
- Bindings: `test_bindings.bat` (all) or `test_binding.bat indicator_name` (one binding).
- Benches: `cargo bench --bench indicator_benchmark --` (set `RUSTFLAGS="-C target-cpu=native"`).

## Style & Conventions

- Format/lint: `cargo +nightly fmt`; `cargo clippy --all-targets --all-features -D warnings`.
- Prefer `once_cell` over `lazy_static`; document public items with `///` (params and errors in natural language).
- Gate SIMD with `#[cfg(feature = "nightly-avx")]`; use `skip_if_unsupported!(kernel, fn_name)` and `assert_same_len!(a, b)`.
- Naming: files/modules `snake_case`; types/traits `UpperCamelCase`; constants `UPPER_SNAKE_CASE`.

## Benchmarks & External Libs

- Install TA‑Lib and Tulip (TULIPC) to benchmark overlapping indicators; compare our SIMD vs their scalar. Platform‑specific install steps may vary; keep scripts/config minimal and reproducible.

## PRs

- Use Conventional Commits (e.g., `feat(indicators): add alma warmup`).
- PRs include description, linked issues, affected indicators/kernels, test notes, and benchmark deltas (commands + env).
- CI gate: `cargo build` (stable) must pass; include nightly AVX tests when touching SIMD.

