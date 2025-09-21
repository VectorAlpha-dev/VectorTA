@echo off
REM Build and run Python CUDA benchmarks
REM Usage:
REM   run_cuda_python_benchmarks.bat           -> run all CUDA python benches (ALMA, WMA, SuperSmoother3Pole, TrAdjEMA)
REM   run_cuda_python_benchmarks.bat alma      -> run only ALMA CUDA python benches
REM   run_cuda_python_benchmarks.bat wma       -> run only WMA CUDA python benches
REM   run_cuda_python_benchmarks.bat epma      -> run only EPMA CUDA python benches
REM   run_cuda_python_benchmarks.bat highpass  -> run only HighPass CUDA python benches
REM   run_cuda_python_benchmarks.bat sinwma    -> run only SINWMA CUDA python benches
REM   run_cuda_python_benchmarks.bat kama      -> run only KAMA CUDA python benches
REM   run_cuda_python_benchmarks.bat nama      -> run only NAMA CUDA python benches
REM   run_cuda_python_benchmarks.bat ss3p      -> run only SuperSmoother 3-Pole CUDA python benches
REM   run_cuda_python_benchmarks.bat tradjema  -> run only TrAdjEMA CUDA python benches

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
) else if /I "%IND%"=="wma" (
  .venv\Scripts\python.exe -OO benchmarks\bench_wma_cuda.py || goto :fail
) else if /I "%IND%"=="epma" (
  .venv\Scripts\python.exe -OO benchmarks\bench_epma_cuda.py || goto :fail
) else if /I "%IND%"=="sinwma" (
  .venv\Scripts\python.exe -OO benchmarks\bench_sinwma_cuda.py || goto :fail
) else if /I "%IND%"=="highpass" (
  .venv\Scripts\python.exe -OO benchmarks\bench_highpass_cuda.py || goto :fail
) else if /I "%IND%"=="kama" (
  .venv\Scripts\python.exe -OO benchmarks\bench_kama_cuda.py || goto :fail
) else if /I "%IND%"=="nama" (
  .venv\Scripts\python.exe -OO benchmarks\bench_nama_cuda.py || goto :fail
) else if /I "%IND%"=="ecema" (
  .venv\Scripts\python.exe -OO benchmarks\bench_ehlers_ecema_cuda.py || goto :fail
) else if /I "%IND%"=="ss3p" (
  .venv\Scripts\python.exe -OO benchmarks\bench_supersmoother_3_pole_cuda.py || goto :fail
) else if /I "%IND%"=="tradjema" (
  .venv\Scripts\python.exe -OO benchmarks\bench_tradjema_cuda.py || goto :fail
) else if /I "%IND%"=="all" (
  .venv\Scripts\python.exe -OO benchmarks\bench_alma_cuda.py || goto :fail
  .venv\Scripts\python.exe -OO benchmarks\bench_wma_cuda.py || goto :fail
  .venv\Scripts\python.exe -OO benchmarks\bench_epma_cuda.py || goto :fail
  .venv\Scripts\python.exe -OO benchmarks\bench_highpass_cuda.py || goto :fail
  .venv\Scripts\python.exe -OO benchmarks\bench_sinwma_cuda.py || goto :fail
  .venv\Scripts\python.exe -OO benchmarks\bench_kama_cuda.py || goto :fail
  .venv\Scripts\python.exe -OO benchmarks\bench_nama_cuda.py || goto :fail
  .venv\Scripts\python.exe -OO benchmarks\bench_ehlers_ecema_cuda.py || goto :fail
  .venv\Scripts\python.exe -OO benchmarks\bench_supersmoother_3_pole_cuda.py || goto :fail
  .venv\Scripts\python.exe -OO benchmarks\bench_tradjema_cuda.py || goto :fail
) else (
  echo Unknown indicator: %IND%
  echo Supported: alma, wma, epma, highpass, sinwma, kama, nama, ecema, ss3p, tradjema, all
  goto :fail
)

echo.
echo CUDA Python benchmarks completed.
exit /b 0

:fail
echo Error running CUDA Python benchmarks.
exit /b 1
