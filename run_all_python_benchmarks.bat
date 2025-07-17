@echo off
echo Running All Benchmarks (Rust + Python)
echo ======================================
echo.

REM Get the directory where this batch file is located
set SCRIPT_DIR=%~dp0

REM Change to the script directory (removes trailing backslash)
cd /d "%SCRIPT_DIR:~0,-1%"

REM Activate virtual environment
call .venv\Scripts\activate

REM Run Rust benchmarks first if not already done
echo Do you want to run Rust benchmarks first? (y/n)
set /p runrust=
if /i "%runrust%"=="y" (
    echo Running Rust benchmarks...
    cargo bench --features nightly-avx --bench indicator_benchmark
    echo.
)

REM Run the improved Python benchmark
echo Running Python benchmarks with accurate methodology...
python benchmarks\criterion_comparable_benchmark.py %*

echo.
echo Done! Results saved to benchmarks\criterion_comparable_results.json
pause