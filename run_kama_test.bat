@echo off
echo Activating virtual environment...
call .venv\Scripts\activate

echo Running KAMA optimization test...
python test_kama_optimization.py

echo.
echo Running benchmark...
python -OO benchmarks\criterion_comparable_benchmark.py kama

pause