@echo off
REM Template for batch files that work from any directory
REM This script automatically changes to its own location

REM Get the directory where this batch file is located
set SCRIPT_DIR=%~dp0

REM Change to the script directory (removes trailing backslash)
cd /d "%SCRIPT_DIR:~0,-1%"

REM Store the project root directory name for display
for %%I in ("%CD%") do set PROJECT_NAME=%%~nxI

echo ========================================
echo Project: %PROJECT_NAME%
echo Path: %CD%
echo ========================================

REM Your commands go here
REM All paths are now relative to the script location
REM Examples:

REM Run cargo commands
REM cargo build --release

REM Access subdirectories
REM cd tests\wasm
REM or better:
REM pushd tests\wasm
REM ... do work ...
REM popd

REM Run scripts in subdirectories
REM call tests\some_script.bat

REM Access files relative to project root
REM type README.md

pause