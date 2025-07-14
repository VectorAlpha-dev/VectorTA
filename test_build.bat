@echo off
echo Testing Python build...
call .venv\Scripts\activate
maturin develop --features python --release
if %errorlevel% equ 0 (
    echo Python build successful!
) else (
    echo Python build failed!
    exit /b 1
)

echo.
echo Testing WASM build...
wasm-pack build --target nodejs --features wasm
if %errorlevel% equ 0 (
    echo WASM build successful!
) else (
    echo WASM build failed!
    exit /b 1
)

echo.
echo All builds completed successfully!