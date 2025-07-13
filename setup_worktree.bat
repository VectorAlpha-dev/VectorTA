@echo off
REM Setup script for binding worktrees

echo Setting up Python environment...
if not exist ".venv" (
    python -m venv .venv
)
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install maturin pytest pytest-xdist numpy

echo.
echo Building WASM package...
call wasm-pack build --target nodejs --features wasm

echo.
echo Installing Node modules...
cd tests\wasm
call npm install

REM Fix Windows import paths if the script exists
if exist "fix_windows_imports.js" (
    echo.
    echo Fixing Windows import paths...
    node fix_windows_imports.js
)

cd ..\..

echo.
echo Setup complete! You can now run: test_bindings.bat