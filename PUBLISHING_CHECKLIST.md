# Publishing Checklist (crates.io)

Goal: publish this repository as a Rust library crate on crates.io (package: `vector-ta`, crate: `vector_ta`), usable as a dependency by other Rust apps, without forcing optional toolchains (CUDA/Python/WASM) on default consumers.

Notes / constraints:
- Do not run `cargo fmt`/`rustfmt` automatically in this repo unless explicitly requested.
- Do not change unit test reference values; fix root causes instead.
- Network operations (`cargo publish`, etc.) will require network access.

---

## 0) Decisions (blockers)

- [ ] Choose crates.io **package name** (planned: `vector-ta` so `use vector_ta::...` works).
- [ ] Confirm desired crate identity: library focus vs “backtester framework” (README + description currently say “Rust-Backtester”).
- [ ] Decide which features are "supported for crates.io consumers":
  - [ ] `cuda` (uses prebuilt PTX for compute_89; consumers do not need `nvcc`)
  - [ ] `cuda-build-ptx` (maintainer-only: compile PTX from `kernels/cuda/**` using `nvcc`)
  - [ ] `python` (PyO3 extension-module)
  - [ ] `wasm` (wasm-bindgen)
  - [ ] `nightly-avx` (nightly-only SIMD)

---

## 1) Cargo.toml: crates.io metadata (required / strongly recommended)

- [ ] Add `description = "..."` (short, crates.io-friendly).
- [ ] Add `license = "..."` (SPDX) **or** `license-file = "LICENSE"` (LICENSE exists at repo root).
- [ ] Add `readme = "README.md"`.
- [ ] Add `repository = "..."` (git URL).
- [ ] Optional but recommended:
  - [ ] `homepage = "..."` (if different from repository)
  - [ ] `documentation = "..."` (docs.rs URL or custom)
  - [ ] `keywords = [...]`
  - [ ] `categories = [...]`
  - [ ] `rust-version = "..."` (minimum supported Rust)

---

## 2) Packaging hygiene (avoid shipping huge/irrelevant files)

Current repo contains large directories (e.g. `target*`, `node_modules/`, demos, generated artifacts). crates.io packages should be small and reproducible.

- [ ] Prefer a strict whitelist:
  - [ ] Add `[package] include = [...]` that covers only what consumers need:
    - `Cargo.toml`, `README.md`, `LICENSE`
    - `src/**`
    - `build.rs` (if needed)
    - `kernels/ptx/**` (required for `cuda` prebuilt PTX)
    - `kernels/cuda/**` (optional; useful if you want to ship kernel sources)
    - `tests/**` (optional; but useful)
    - `benches/**` (optional; usually omitted from published crate)
- [ ] Ensure `exclude = [...]` also blocks common large dirs:
  - [ ] `target/**`, `target*/**`
  - [ ] `node_modules/**`
  - [ ] `.venv/**`, `.pytest_cache/**`, `pkg/**`, `pkg_bak/**`
  - [ ] demo crates (if not intended as published workspace members)

---

## 3) Default features & dependency surface (make “normal use” easy)

Publishing goal: `cargo add <crate>` should build without requiring CUDA/Python/WASM toolchains.

- [ ] Revisit `[features] default = [...]`
  - [ ] Remove `proptest` from default if it’s only for tests/property tests.
  - [ ] Keep `cuda`, `python`, `wasm`, `nightly-avx` opt-in only.
- [ ] Ensure optional dependencies are truly optional and gated by their features.

---

## 4) Build script + build-dependencies (avoid unnecessary requirements)

`build.rs` stages prebuilt PTX into `OUT_DIR` for `feature = "cuda"` builds, and only runs `nvcc` when `feature = "cuda-build-ptx"` is enabled.

- [ ] Audit `Cargo.toml [build-dependencies]` for unused items:
  - [ ] Remove `cmake`, `bindgen`, `which`, `cc` if not used by `build.rs` (or only needed in other non-published subprojects).
- [ ] Confirm `build.rs` is safe for crates.io consumers:
  - [ ] No `nvcc` invocation unless `CARGO_FEATURE_CUDA_BUILD_PTX` is set.
  - [ ] No network calls.
  - [ ] Clear error messages when `cuda` is enabled but prebuilt PTX is missing.

---

## 5) Library crate output types (rlib/cdylib policy)

This crate currently builds as `crate-type = ["cdylib", "rlib"]`.

- [ ] Decide whether to keep `cdylib` in the published crate:
  - [ ] If Python extension is part of the published story, keep it and document how to build/use it.
  - [ ] If not, consider moving Python bindings to a separate crate (workspace member) so the core library stays “pure Rust”.

---

## 6) README & docs.rs readiness

- [ ] Update `README.md` to match the crate being published:
  - [ ] Minimal usage example (`Cargo.toml` + a short code snippet).
  - [ ] Feature matrix (`cuda`/`python`/`wasm`/`nightly-avx`) and toolchain requirements.
  - [ ] API entry points / module overview.
- [ ] Optional: add docs.rs metadata:
  - [ ] `[package.metadata.docs.rs]` with sensible `features` and/or `rustdoc-args`.

---

## 7) Pre-publish validation (local)

Run these before any publish attempt:

- [ ] WSL note: if your repo lives on `/mnt/c/**` (Windows drive), `cargo package`/`cargo publish` can fail with a bogus “could not learn metadata for ... .crate (os error 2)”. Workaround: run with a Linux-filesystem target dir, e.g. `CARGO_TARGET_DIR=/tmp/vector_ta_target cargo package` (or clone the repo under your WSL home dir).
- [ ] `cargo check` (default features)
- [ ] `cargo test` (default features)
- [ ] `cargo clippy --all-targets -D warnings` (as applicable)
- [ ] `cargo package` (ensures the packaged tarball builds)
- [ ] `cargo publish --dry-run` (requires crates.io auth but no publish)

---

## 8) Publish steps (crates.io)

- [ ] Ensure crates.io account and token are set up.
- [ ] `cargo login` (token)
- [ ] `cargo publish`
- [ ] Verify installability from a clean project:
  - [ ] New temp project: add dependency, `cargo build`

---

## 9) Downstream adoption

Out of scope for this checklist (downstream apps live in separate repos), but recommended after publish:

- [ ] Add as a dependency by version, enable only needed features.
- [ ] Validate builds on target platforms (Windows first, then Linux/macOS as applicable).

---

## 10) Versioning & release process

- [ ] Decide versioning policy (SemVer; pre-1.0 rules if staying `0.x`).
- [ ] Tag releases in git (`vX.Y.Z`) and keep a simple changelog (optional).
