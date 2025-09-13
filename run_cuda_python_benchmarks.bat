@echo off
REM Build and run Python CUDA benchmarks
REM Usage:
REM   run_cuda_python_benchmarks.bat           -> run all CUDA python benches (current: ALMA)
REM   run_cuda_python_benchmarks.bat alma      -> run only ALMA CUDA python benches

setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR:~0,-1%"

set IND=%~1
if "%IND%"=="" set IND=all

if not exist .venv (
  echo Creating Python venv...
  python -m venv .venv || goto :fail
)
call .venv\Scripts\activate.bat || goto :fail
.venv\Scripts\python.exe -m pip install --upgrade pip --quiet
.venv\Scripts\python.exe -m pip install maturin numpy --quiet

REM Clean any existing installed module to avoid stale symbols
.venv\Scripts\python.exe -c "import sys,shutil,pathlib; site=[p for p in sys.path if p.endswith('site-packages')][0]; [shutil.rmtree(str(p), ignore_errors=True) for pat in ('my_project*','ta_indicators*') for p in pathlib.Path(site).glob(pat)]"

echo Building Python module (python+cuda)...
set RUSTFLAGS=
maturin develop --features "python,cuda" --release || goto :fail

echo Running CUDA Python benchmarks for: %IND%
if /I "%IND%"=="alma" (
  .venv\Scripts\python.exe -OO benchmarks\bench_alma_cuda.py || goto :fail
) else if /I "%IND%"=="all" (
  REM Currently only ALMA benchmarks exist; extend here as new kernels are added
  .venv\Scripts\python.exe -OO benchmarks\bench_alma_cuda.py || goto :fail
) else (
  echo Unknown indicator: %IND%
  echo Supported: alma, all
  goto :fail
)

echo.
echo CUDA Python benchmarks completed.
exit /b 0

:fail
echo Error running CUDA Python benchmarks.
exit /b 1
