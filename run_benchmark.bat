@echo off
REM Run Python benchmarks with accurate methodology
cd /d "C:\Rust Projects\my_project"
call .venv\Scripts\activate

echo Building Python bindings...
echo =======================================
maturin develop --features python --release
if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to build Python bindings
    pause
    exit /b 1
)
echo.

echo Python Benchmark (Accurate Methodology)
echo =======================================
echo.

python benchmarks\criterion_comparable_benchmark.py %*

pause