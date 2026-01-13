@echo off

set SCRIPT_DIR=%~dp0

cd /d "%SCRIPT_DIR:~0,-1%"

for %%I in ("%CD%") do set PROJECT_NAME=%%~nxI

echo Working directory: %PROJECT_NAME% (%CD%)
echo Building WASM bindings...
echo ========================

set CARGO_PROFILE_RELEASE_LTO=off

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
