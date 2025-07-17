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

REM Check for rebuild flag
set FORCE_REBUILD=0
if "%1"=="--rebuild" (
    set FORCE_REBUILD=1
    shift
)

REM Build WASM package with optimizations
if %FORCE_REBUILD%==1 (
    echo Force rebuilding WASM package with optimizations...
    goto :build
)
if not exist pkg\my_project.js (
    echo Building WASM package with optimizations...
    goto :build
)
goto :skipbuild

:build
call wasm-pack build --target nodejs --features wasm --release -- --features nightly-avx
if %errorlevel% neq 0 (
    echo Error: Failed to build WASM package
    exit /b 1
)
echo WASM package built successfully!
echo.

:skipbuild

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
echo Usage: run_wasm_indicator_benchmark.bat [--rebuild] [indicators...]
echo.
echo Options:
echo   --rebuild    Force rebuild WASM package with optimizations before running
echo.
echo Examples:
echo   run_wasm_indicator_benchmark.bat                - Run all indicators
echo   run_wasm_indicator_benchmark.bat alma           - Run only ALMA
echo   run_wasm_indicator_benchmark.bat alma sma       - Run ALMA and SMA
echo   run_wasm_indicator_benchmark.bat --rebuild      - Rebuild and run all
echo   run_wasm_indicator_benchmark.bat --rebuild alma - Rebuild and run ALMA
echo.
echo Available indicators are listed in benchmarks/wasm_indicator_benchmark.js

:end
REM Return to original directory
cd ..

echo.
echo Benchmark complete!