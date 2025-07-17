@echo off
REM Build WASM bindings without LTO

echo Building WASM bindings...
echo ========================

REM Disable LTO for WASM builds to avoid cdylib issues
set CARGO_PROFILE_RELEASE_LTO=off

REM Build with wasm-pack
wasm-pack build --target nodejs --features wasm

if %errorlevel% equ 0 (
    echo.
    echo WASM bindings built successfully!
) else (
    echo.
    echo WASM build failed!
    exit /b 1
)

pause