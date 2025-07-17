@echo off
REM Verify Rust performance with native CPU optimizations

echo Setting Rust optimization flags...
set RUSTFLAGS=-C target-cpu=native -C opt-level=3

echo.
echo Building optimized Rust benchmarks...
cargo bench --features nightly-avx --bench indicator_benchmark -- alma --exact

echo.
echo Running specific ALMA benchmark...
cargo bench --features nightly-avx --bench indicator_benchmark -- "alma/1M/scalar" --exact
cargo bench --features nightly-avx --bench indicator_benchmark -- "alma/1M/AVX2" --exact
cargo bench --features nightly-avx --bench indicator_benchmark -- "alma/1M/AVX-512" --exact

pause