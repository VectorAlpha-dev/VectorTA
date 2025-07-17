@echo off
REM Build Python bindings with optimizations
REM This script works from any directory by changing to its own location first

REM Get the directory where this batch file is located
set SCRIPT_DIR=%~dp0

REM Change to the script directory (removes trailing backslash)
cd /d "%SCRIPT_DIR:~0,-1%"

REM Store the project root directory name for display
for %%I in ("%CD%") do set PROJECT_NAME=%%~nxI

echo Working directory: %PROJECT_NAME% (%CD%)
echo Building Python bindings...
echo ==========================

REM Enable LTO for Python builds
set CARGO_PROFILE_RELEASE_LTO=thin

REM Build with maturin
maturin develop --features python,nightly-avx --release

if %errorlevel% equ 0 (
    echo.
    echo Python bindings built successfully!
) else (
    echo.
    echo Python build failed!
    exit /b 1
)

pause