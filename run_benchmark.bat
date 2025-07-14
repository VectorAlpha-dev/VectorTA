@echo off
REM Run Python benchmarks with accurate methodology
cd /d "C:\Rust Projects\my_project"
call .venv\Scripts\activate

echo Python Benchmark (Accurate Methodology)
echo =======================================
echo.

python benchmarks\criterion_comparable_benchmark.py %*

pause