@echo off

set SCRIPT_DIR=%~dp0

cd /d "%SCRIPT_DIR:~0,-1%"

call .venv\Scripts\activate

echo Setting optimization flags...
echo =======================================
set RUSTFLAGS=-C target-cpu=native -C opt-level=3

set PYTHONOPTIMIZE=2

set NPY_RELAXED_STRIDES_CHECKING=1

set OPENBLAS_NUM_THREADS=1
set MKL_NUM_THREADS=1

echo.
echo Building Python bindings with native CPU optimizations...
echo =======================================
maturin develop --features python,nightly-avx --release
if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to build Python bindings
    pause
    exit /b 1
)
echo.

echo Python Benchmark (Accurate Methodology with Optimizations)
echo =======================================
echo Running with:
echo   - PYTHONOPTIMIZE=2 (assertions disabled)
echo   - RUSTFLAGS=-C target-cpu=native
echo   - NPY_RELAXED_STRIDES_CHECKING=1
echo   - Single-threaded BLAS
echo.

python -OO benchmarks\criterion_comparable_benchmark.py %*

pause