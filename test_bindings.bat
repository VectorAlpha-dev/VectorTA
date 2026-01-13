@echo off

setlocal EnableExtensions EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"

cd /d "%SCRIPT_DIR:~0,-1%"

for %%I in ("%CD%") do set PROJECT_NAME=%%~nxI

echo Working directory: %PROJECT_NAME% (%CD%)

set "RUN_PYTHON=true"
set "RUN_WASM=true"
set "TEST_PATTERN="
set "USE_NIGHTLY_AVX=true"
set "FAILED=false"

:parse_args
if "%~1"=="" goto start_tests
if /I "%~1"=="--python" set "RUN_WASM=false" & shift & goto parse_args
if /I "%~1"=="--pytohn" set "RUN_WASM=false" & shift & goto parse_args
if /I "%~1"=="--wasm"   set "RUN_PYTHON=false" & shift & goto parse_args
if /I "%~1"=="--avx"    set "USE_NIGHTLY_AVX=true" & shift & goto parse_args
if /I "%~1"=="--help" goto show_help
set "TEST_PATTERN=%~1"
shift
goto parse_args

:show_help
echo Usage: %~nx0 [options] [test_pattern]
echo   --python     Run only Python tests
echo   --wasm       Run only WASM tests
echo   --avx        Build Python bindings with nightly AVX (requires rustup nightly)
echo   --help       Show this help message
echo.
echo Examples:
echo   %~nx0
echo   %~nx0 alma
echo   %~nx0 --python alma
echo   %~nx0 --python --avx alma
exit /b 0

:start_tests
echo Running binding tests...
echo ==================================

if /I "%RUN_PYTHON%"=="true" goto py_tests
goto after_python

:py_tests
echo.
echo Setting up Python environment...

set "PYO3_PYTHON="
set "MATURIN_PYTHON="
set "PYTHONHOME="
set "PYTHONPATH="

if not exist ".venv" goto make_venv
goto venv_ready

:make_venv
echo Creating Python virtual environment...
python -m venv .venv
if errorlevel 1 goto try_py_launcher
goto venv_ready

:try_py_launcher
where py >nul 2>&1
if errorlevel 1 goto venv_create_fail
py -3 -m venv .venv
if errorlevel 1 goto venv_create_fail

:venv_ready
set "VENV_PY=.venv\Scripts\python.exe"

call :ensure_venv_layout
if errorlevel 1 goto venv_create_fail

if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat" >nul 2>&1

set "PYO3_PYTHON=%VENV_PY%"
set "MATURIN_PYTHON=%VENV_PY%"
set "PYTHON=%VENV_PY%"
set "VIRTUAL_ENV=%CD%\.venv"
set "PATH=%CD%\.venv\Scripts;%PATH%"

"%VENV_PY%" -m pip install --upgrade pip --quiet
"%VENV_PY%" -m pip install maturin pytest pytest-xdist numpy pandas --quiet

echo Building Python bindings...
set "RUSTFLAGS="
if /I "%USE_NIGHTLY_AVX%"=="true" goto build_py_avx
"%VENV_PY%" -m maturin develop --features python --release --pip-path "%CD%\.venv\Scripts\pip.exe"
if errorlevel 1 goto py_build_fail
goto py_built

:build_py_avx
set "RUSTUP_TOOLCHAIN=nightly"
"%VENV_PY%" -m maturin develop --features python,nightly-avx --release --pip-path "%CD%\.venv\Scripts\pip.exe"
if errorlevel 1 goto py_build_fail

:py_built
echo Python build successful
echo Pre-generating reference outputs to avoid test-time Cargo locks...
if /I "%USE_NIGHTLY_AVX%"=="true" (
  set "RUSTUP_TOOLCHAIN=nightly"
  cargo build --quiet --release --features nightly-avx --bin generate_references >nul 2>&1
) else (
  cargo build --quiet --release --bin generate_references >nul 2>&1
)
set "RUST_REF_BIN=%CD%\target\release\generate_references.exe"
if not exist "%RUST_REF_BIN%" set "RUST_REF_BIN=%CD%\target-py\release\generate_references.exe"
echo Using reference binary: %RUST_REF_BIN%
echo.
echo Running Python tests...
if "%TEST_PATTERN%"=="" goto run_all_py
"%VENV_PY%" tests\python\run_all_tests.py %TEST_PATTERN%
if errorlevel 1 goto py_tests_fail
echo Python tests passed
goto after_python

:run_all_py
"%VENV_PY%" tests\python\run_all_tests.py
if errorlevel 1 goto py_tests_fail
echo Python tests passed
goto after_python

:venv_create_fail
echo Failed to create virtual environment. Ensure Python is installed and on PATH.
set "FAILED=true"
goto after_python


:ensure_venv_layout
if exist ".venv\Scripts\python.exe" (
  for /f "usebackq tokens=*" %%P in (`".venv\Scripts\python.exe" -c "import sys; print(sys.executable)" 2^>nul`) do set "_VENV_EXE=%%P"
  if not defined _VENV_EXE goto recreate_venv
  echo %_VENV_EXE% | findstr /I "/usr/bin\\python.exe" >nul 2>&1
  if %errorlevel%==0 goto recreate_venv
  exit /b 0
)
if exist ".venv\bin\activate" (
  echo Detected POSIX virtual environment at .venv\bin; recreating for Windows...
  rmdir /s /q .venv
)

:recreate_venv
echo Recreating Python virtual environment with Windows launcher...
where py >nul 2>&1
if errorlevel 1 (
  python -m venv .venv
  if errorlevel 1 exit /b 1
  exit /b 0
)
py -3 -m venv .venv
if errorlevel 1 exit /b 1
exit /b 0

:py_build_fail
echo Python build failed
set "FAILED=true"
goto after_python

:py_tests_fail
echo Python tests failed
set "FAILED=true"
goto after_python

:after_python

if /I "%RUN_WASM%"=="true" goto wasm_tests
goto epilogue

:wasm_tests
echo.
echo Cleaning WASM build artifacts...
if exist "pkg" rmdir /s /q pkg 2>nul
if exist "target\wasm32-unknown-unknown" rmdir /s /q "target\wasm32-unknown-unknown" 2>nul

echo Building WASM bindings...
set "CARGO_PROFILE_RELEASE_LTO=off"
wasm-pack build --target nodejs --features wasm
if errorlevel 1 goto wasm_build_fail

echo WASM build successful
echo.
echo Running WASM tests...
pushd tests\wasm >nul
call npm install
if "%TEST_PATTERN%"=="" goto run_all_wasm
call npm test -- %TEST_PATTERN%
if errorlevel 1 goto wasm_tests_fail
echo WASM tests passed
popd >nul
goto epilogue

:run_all_wasm
call npm test
if errorlevel 1 goto wasm_tests_fail
echo WASM tests passed
popd >nul
goto epilogue

:wasm_build_fail
echo WASM build failed
set "FAILED=true"
goto epilogue

:wasm_tests_fail
echo WASM tests failed
set "FAILED=true"
popd >nul
goto epilogue

:epilogue
echo.
echo ==================================
if /I "%FAILED%"=="true" goto fail_summary
echo All tests passed!
exit /b 0

:fail_summary
echo Tests completed with failures
exit /b 1
