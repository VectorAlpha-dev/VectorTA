@echo off
REM Run WASM benchmarks with optimizations
REM This script works from any directory by changing to its own location first

REM Get the directory where this batch file is located
set SCRIPT_DIR=%~dp0

REM Change to the script directory (removes trailing backslash)
cd /d "%SCRIPT_DIR:~0,-1%"

REM Store the project root directory name for display
for %%I in ("%CD%") do set PROJECT_NAME=%%~nxI

echo Working directory: %PROJECT_NAME% (%CD%)
echo Building WASM with optimizations...
echo =======================================

REM Set build flags for optimization (no LTO for WASM)
REM Don't use target-cpu=native for WASM as it's not applicable
set RUSTFLAGS=-C opt-level=3 -C embed-bitcode=yes
set CARGO_PROFILE_RELEASE_LTO=off

REM Build WASM module
call wasm-pack build --target nodejs --features wasm

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: WASM build failed!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Running WASM benchmarks...
echo =======================================

REM Run benchmarks with GC exposed for better control
node --expose-gc benchmarks\wasm_benchmark.js %*

pause