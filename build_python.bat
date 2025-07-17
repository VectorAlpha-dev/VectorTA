@echo off
REM Build Python bindings with optimizations

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