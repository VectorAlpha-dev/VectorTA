@echo off
echo ========================================
echo SRSI Indicator Test Suite
echo ========================================
echo.

REM Check if virtual environment exists
if not exist ".venv" (
    echo Creating Python virtual environment...
    python -m venv .venv
)

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
.venv\Scripts\python.exe -m pip install --upgrade pip --quiet
.venv\Scripts\python.exe -m pip install maturin pytest pytest-xdist numpy --quiet

REM Build Python bindings with optimizations
echo.
echo Building Python bindings with optimizations...
set "RUSTFLAGS=-Zdylib-lto"
.venv\Scripts\python.exe -m maturin develop --features python,nightly-avx --release

if %errorlevel% neq 0 (
    echo ERROR: Failed to build Python bindings
    exit /b 1
)

REM Run SRSI Python tests
echo.
echo ========================================
echo Running SRSI Python tests...
echo ========================================
.venv\Scripts\python.exe -m pytest tests/python/test_srsi.py -v

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Some tests failed!
    exit /b 1
) else (
    echo.
    echo SUCCESS: All SRSI tests passed!
)

REM Quick functionality check
echo.
echo ========================================
echo Running quick functionality check...
echo ========================================
.venv\Scripts\python.exe -c "import my_project as ta; import numpy as np; data = np.random.rand(100)*100+50; k,d = ta.srsi(data); print('SRSI single: OK'); r = ta.srsi_batch(data, (14,14,0), (14,14,0), (3,3,0), (3,3,0)); print('SRSI batch: OK'); s = ta.SrsiStream(14,14,3,3); [s.update(v) for v in data[:30]]; print('SRSI stream: OK')"

echo.
echo ========================================
echo All tests completed successfully!
echo ========================================