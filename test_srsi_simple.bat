@echo off
echo Testing SRSI Python bindings...
echo ================================

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Build Python bindings
echo Building Python bindings...
python -m maturin develop --features python,nightly-avx --release

REM Run SRSI tests
echo.
echo Running SRSI tests...
python -m pytest tests/python/test_srsi.py -v

echo.
echo Test complete!