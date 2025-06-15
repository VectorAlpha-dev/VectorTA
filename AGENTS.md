This is a large TA indicator library that will eventually have avx2, avx512, scalar, and cuda kernels for most indicators. It's expected to have over 300 indicators when complete. 

nightly  : cargo +nightly build --all-features   (unlocks AVX2 / AVX512 via feature nightly-avx)

stable   : cargo build                           (must succeed without nightly-avx)

Tool‑chain pinning (root‑level rust‑toolchain.toml):
[toolchain]
channel    = "nightly-2025-05-28"
components = ["rustfmt", "clippy"]
[overrides]
default    = "stable"    # fallback if nightly fails

Compiler flags used in CI and benches:
export RUSTFLAGS="-C target-cpu=native"

Test recipes

Single indicator (scalar)     : cargo test --lib indicators:: -- --nocapture

AVX2 kernels (nightly)        : cargo +nightly test --features nightly-avx --lib indicators:: -- --nocapture

AVX512 kernels (nightly)      : cargo +nightly test --features nightly-avx --lib indicators::_avx512 -- --nocapture

bench recipes : cargo bench --bench indicator_benchmark -- 

Default behaviour: Start with the “Single indicator (scalar)” recipe for speed. Widen scope only when asked.

Scope & context rules
• When asked to “fix the tests for indicators::”, load only:
– src/indicators/.rs (implementation)
– Any tests for 
– Reference file: src/indicators/moving_averages/alma.rs (gold‑standard API & style)
• Do not modify other modules unless explicitly instructed.

Coding style checklist
• Prefer once_cell over lazy_static
• Public items must have /// documentation for params and errors. Use natural human language. 
• Wrap SIMD code with #[cfg(feature = "nightly-avx")]

skip_if_unsupported!(kernel, fn_name)   // early exit on unsupported SIMD
assert_same_len!(vec1, vec2)            // guard against length mismatch