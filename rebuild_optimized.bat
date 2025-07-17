@echo off
REM Rebuild with maximum optimizations

echo Cleaning previous builds...
cargo clean

echo.
echo Setting optimization flags...
set RUSTFLAGS=-C target-cpu=native -C opt-level=3 -C lto=fat -C embed-bitcode=yes

echo.
echo Building optimized Python bindings...
maturin develop --features python,nightly-avx --release

echo.
echo Build complete. Run benchmarks with:
echo   python -OO benchmarks\direct_comparison.py
echo   run_benchmark.bat alma

pause