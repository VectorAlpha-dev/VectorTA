@echo off
REM Quick test to verify ALMA performance

cd /d "C:\Rust Projects\my_project-bindings-4"
call .venv\Scripts\activate

echo Testing ALMA optimization...
echo =======================================

REM Set optimization flags
set PYTHONOPTIMIZE=2
set NPY_RELAXED_STRIDES_CHECKING=1

echo.
echo Running minimal benchmark...
python -OO benchmarks\minimal_alma_bench.py

echo.
echo You can now run the full benchmark with:
echo   run_benchmark.bat alma

pause