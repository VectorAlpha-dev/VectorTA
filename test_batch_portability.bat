@echo off
REM Test script to verify batch file portability

REM Get the directory where this batch file is located
set SCRIPT_DIR=%~dp0

REM Change to the script directory (removes trailing backslash)
cd /d "%SCRIPT_DIR:~0,-1%"

REM Store the project root directory name for display
for %%I in ("%CD%") do set PROJECT_NAME=%%~nxI

echo ========================================
echo Testing batch file portability
echo ========================================
echo Script location: %SCRIPT_DIR%
echo Project name: %PROJECT_NAME%
echo Current directory: %CD%
echo ========================================

REM Test that we can access project files
if exist "Cargo.toml" (
    echo [PASS] Found Cargo.toml in project root
) else (
    echo [FAIL] Could not find Cargo.toml
)

if exist "tests\wasm\test_alma.js" (
    echo [PASS] Found test_alma.js in expected location
) else (
    echo [FAIL] Could not find test_alma.js
)

echo ========================================
echo Test complete!
pause