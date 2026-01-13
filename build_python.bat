@echo off

set SCRIPT_DIR=%~dp0

cd /d "%SCRIPT_DIR:~0,-1%"

for %%I in ("%CD%") do set PROJECT_NAME=%%~nxI

echo Working directory: %PROJECT_NAME% (%CD%)
echo Building Python bindings...
echo ==========================

set CARGO_PROFILE_RELEASE_LTO=thin

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