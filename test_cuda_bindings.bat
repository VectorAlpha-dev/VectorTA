@echo off
REM Test CUDA bindings (Rust + Python) for one indicator (e.g., alma) or all CUDA tests.
REM Usage:
REM   test_cuda_bindings.bat           # build + run all CUDA tests (Rust + Python)
REM   test_cuda_bindings.bat alma      # build + run only ALMA CUDA tests

setlocal enabledelayedexpansion

REM Change to this script directory
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR:~0,-1%"

set INDICATOR=%~1
if "%INDICATOR%"=="" (
  set INDICATOR=all
)

echo ==============================================
echo [CUDA] Testing CUDA bindings - %INDICATOR%
echo ==============================================

REM Ensure nightly toolchain available for cargo +nightly (Rust CUDA tests)
where cargo >nul 2>nul
if errorlevel 1 (
  echo Error: cargo not found in PATH
  exit /b 1
)

REM Setup Python venv
if not exist .venv (
  echo Creating Python venv...
  python -m venv .venv || goto :py_fail
)
call .venv\Scripts\activate.bat || goto :py_fail
.venv\Scripts\python.exe -m pip install --upgrade pip --quiet
.venv\Scripts\python.exe -m pip install maturin pytest numpy --quiet

REM Clean any existing installed module to avoid stale symbols
.venv\Scripts\python.exe -c "import sys,shutil,pathlib; site=[p for p in sys.path if p.endswith('site-packages')][0]; [shutil.rmtree(str(p), ignore_errors=True) for pat in ('my_project*','ta_indicators*') for p in pathlib.Path(site).glob(pat)]"

REM Build Python module with CUDA
echo Building Python module (python+cuda)...
set RUSTFLAGS=
maturin develop --features "python,cuda" --release || goto :py_fail

REM Run Python CUDA tests
echo Running Python CUDA tests...
if /I "%INDICATOR%"=="alma" (
  .venv\Scripts\python.exe -m pytest -q tests\python\test_alma_cuda.py || goto :py_tests_fail
) else (
  REM Run all *_cuda python tests
  .venv\Scripts\python.exe -m pytest -q tests\python -k cuda || goto :py_tests_fail
)
echo Python CUDA tests passed

REM Run Rust CUDA tests (feature-gated)
echo Running Rust CUDA tests...
if /I "%INDICATOR%"=="alma" (
  cargo +nightly test --features cuda alma_cuda -- --nocapture || goto :rs_tests_fail
) else (
  cargo +nightly test --features cuda cuda -- --nocapture || goto :rs_tests_fail
)
echo Rust CUDA tests passed

echo.
echo All CUDA binding tests passed.
exit /b 0

:py_fail
echo Error: Python/CUDA build setup failed.
exit /b 1

:py_tests_fail
echo Error: Python CUDA tests failed.
exit /b 1

:rs_tests_fail
echo Error: Rust CUDA tests failed.
exit /b 1
