@echo off
echo Running All Benchmarks (Rust + Python)
echo ======================================
echo.

set SCRIPT_DIR=%~dp0

cd /d "%SCRIPT_DIR:~0,-1%"

call .venv\Scripts\activate

echo Do you want to run Rust benchmarks first? (y/n)
set /p runrust=
if /i "%runrust%"=="y" (
    echo Running Rust benchmarks...
    cargo bench --features nightly-avx --bench indicator_benchmark
    echo.
)

echo Running Python benchmarks with accurate methodology...
python benchmarks\criterion_comparable_benchmark.py %*

echo.
echo Done! Results saved to benchmarks\criterion_comparable_results.json
pause