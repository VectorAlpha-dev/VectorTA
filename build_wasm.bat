@echo off
REM Build WASM bindings without LTO
REM This script works from any directory by changing to its own location first

REM Get the directory where this batch file is located
set SCRIPT_DIR=%~dp0

REM Change to the script directory (removes trailing backslash)
cd /d "%SCRIPT_DIR:~0,-1%"

REM Store the project root directory name for display
for %%I in ("%CD%") do set PROJECT_NAME=%%~nxI

echo Working directory: %PROJECT_NAME% (%CD%)
echo Building WASM bindings...
echo ========================

REM Disable LTO for WASM builds to avoid cdylib issues
set CARGO_PROFILE_RELEASE_LTO=off

REM Build with wasm-pack
wasm-pack build --target nodejs --out-name vector_ta --features wasm

if %errorlevel% equ 0 (
    echo.
    echo WASM bindings built successfully!
) else (
    echo.
    echo WASM build failed!
    exit /b 1
)

pause
