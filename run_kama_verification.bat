@echo off
echo Running KAMA Performance Verification...
echo.

REM First ensure Rust benchmarks are fresh
echo Step 1: Running Rust benchmarks...
cargo bench --features nightly-avx --bench indicator_benchmark -- kama/kama_scalar/1M kama/kama_avx2/1M kama/kama_avx512/1M

echo.
echo Step 2: Running Python verification...
call .venv\Scripts\activate
python verify_kama_performance.py

pause