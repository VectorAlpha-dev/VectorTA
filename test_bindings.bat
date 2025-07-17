@echo off
REM Windows batch file for testing bindings
REM Usage:
REM   test_bindings.bat              - Run all tests
REM   test_bindings.bat alma         - Run only alma tests
REM   test_bindings.bat --python     - Run only Python tests
REM   test_bindings.bat --wasm       - Run only WASM tests

setlocal enabledelayedexpansion

REM Colors don't work the same in Windows, so using simple text
set "RUN_PYTHON=true"
set "RUN_WASM=true"
set "TEST_PATTERN="

REM Parse arguments
:parse_args
if "%~1"=="" goto :start_tests
if "%~1"=="--python" (
    set "RUN_WASM=false"
    shift
    goto :parse_args
)
if "%~1"=="--wasm" (
    set "RUN_PYTHON=false"
    shift
    goto :parse_args
)
if "%~1"=="--help" (
    echo Usage: %0 [options] [test_pattern]
    echo Options:
    echo   --python     Run only Python tests
    echo   --wasm       Run only WASM tests
    echo   --help       Show this help message
    echo.
    echo Examples:
    echo   %0                    # Run all tests
    echo   %0 alma               # Run only alma tests
    echo   %0 --python alma      # Run only Python alma tests
    exit /b 0
)
set "TEST_PATTERN=%~1"
shift
goto :parse_args

:start_tests
echo Running binding tests...
echo ==================================

REM Python tests
if "%RUN_PYTHON%"=="true" (
    echo.
    echo Setting up Python environment...
    
    REM Check if venv exists
    if not exist ".venv" (
        echo Creating Python virtual environment...
        python -m venv .venv
    )
    
    REM Activate virtual environment
    call .venv\Scripts\activate.bat
    
    REM Install dependencies using venv Python explicitly
    .venv\Scripts\python.exe -m pip install --upgrade pip --quiet
    .venv\Scripts\python.exe -m pip install maturin pytest pytest-xdist numpy --quiet
    
    REM Ensure maturin is in PATH for this session
    for /f %%i in ('.venv\Scripts\python.exe -m site --user-base') do set PATH=%%i\Scripts;!PATH!
    
    echo.
    echo Building Python bindings...
    .venv\Scripts\python.exe -m maturin develop --features python,nightly-avx --release
    if !errorlevel! equ 0 (
        echo Python build successful
        
        echo.
        echo Running Python tests...
        if "%TEST_PATTERN%"=="" (
            .venv\Scripts\python.exe tests/python/run_all_tests.py
        ) else (
            .venv\Scripts\python.exe tests/python/run_all_tests.py %TEST_PATTERN%
        )
        if !errorlevel! neq 0 (
            echo Python tests failed
            set "FAILED=true"
        ) else (
            echo Python tests passed
        )
    ) else (
        echo Python build failed
        set "FAILED=true"
    )
)

REM WASM tests
if "%RUN_WASM%"=="true" (
    echo.
    echo Building WASM bindings...
    wasm-pack build --target nodejs --features wasm
    if !errorlevel! equ 0 (
        echo WASM build successful
        
        echo.
        echo Running WASM tests...
        cd tests\wasm
        call npm install
        if "%TEST_PATTERN%"=="" (
            call npm test
        ) else (
            call npm test -- %TEST_PATTERN%
        )
        if !errorlevel! neq 0 (
            echo WASM tests failed
            set "FAILED=true"
        ) else (
            echo WASM tests passed
        )
        cd ..\..
    ) else (
        echo WASM build failed
        set "FAILED=true"
    )
)

echo.
echo ==================================
if "%FAILED%"=="true" (
    echo Tests completed with failures
    exit /b 1
) else (
    echo All tests passed!
    exit /b 0
)