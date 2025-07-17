@echo off
REM Run Python benchmarks with accurate methodology and optimizations

REM Get the directory where this batch file is located
set SCRIPT_DIR=%~dp0

REM Change to the script directory (removes trailing backslash)
cd /d "%SCRIPT_DIR:~0,-1%"

REM Activate virtual environment
call .venv\Scripts\activate

echo Setting optimization flags...
echo =======================================
REM CPU-specific optimizations for maximum SIMD performance
set RUSTFLAGS=-C target-cpu=native -C opt-level=3

REM Python optimizations - disable assertions and debug checks
set PYTHONOPTIMIZE=2

REM NumPy optimizations - relax stride checking
set NPY_RELAXED_STRIDES_CHECKING=1

REM OpenBLAS thread settings (if NumPy uses OpenBLAS)
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

REM Run Python with -OO flag for maximum optimization
python -OO benchmarks\criterion_comparable_benchmark.py %*

pause