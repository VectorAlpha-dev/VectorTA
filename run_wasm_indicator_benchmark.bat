@echo off
setlocal

echo WASM Indicator Performance Benchmark
echo =====================================
echo.

REM Check if Node.js is available
where node >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: Node.js not found. Please install Node.js.
    exit /b 1
)

REM Ensure WASM package is built
if not exist pkg\my_project.js (
    echo Building WASM package...
    call wasm-pack build --target nodejs --features wasm
    if %errorlevel% neq 0 (
        echo Error: Failed to build WASM package
        exit /b 1
    )
)

REM Navigate to benchmarks directory
cd benchmarks

REM Check if help is requested
if "%1"=="-h" goto :help
if "%1"=="--help" goto :help
if "%1"=="/?" goto :help

REM Run benchmark with GC exposed for better measurements
echo Running benchmark with GC control...
echo.
node --expose-gc wasm_indicator_benchmark.js %*
goto :end

:help
echo Usage: run_wasm_indicator_benchmark.bat [indicators...]
echo.
echo Examples:
echo   run_wasm_indicator_benchmark.bat          - Run all indicators
echo   run_wasm_indicator_benchmark.bat alma     - Run only ALMA
echo   run_wasm_indicator_benchmark.bat alma sma - Run ALMA and SMA
echo.
echo Available indicators are listed in benchmarks/wasm_indicator_benchmark.js

:end
REM Return to original directory
cd ..

echo.
echo Benchmark complete!