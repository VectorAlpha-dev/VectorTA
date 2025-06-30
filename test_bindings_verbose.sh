#!/bin/bash
# Verbose version with debugging and full paths

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Checking environment...${NC}"
echo "Current user: $(whoami)"
echo "Current directory: $(pwd)"
echo "PATH: $PATH"
echo ""

# Check for required tools
echo "Checking for required tools..."
if command -v cargo &> /dev/null; then
    echo "✓ cargo found: $(which cargo)"
    echo "  version: $(cargo --version)"
else
    echo "✗ cargo not found"
    echo "  Please install Rust: https://rustup.rs/"
    exit 1
fi

if command -v python &> /dev/null; then
    echo "✓ python found: $(which python)"
    echo "  version: $(python --version)"
else
    echo "✗ python not found"
    exit 1
fi

if command -v maturin &> /dev/null; then
    echo "✓ maturin found: $(which maturin)"
else
    echo "✗ maturin not found"
    echo "  Installing maturin..."
    pip install maturin
fi

if command -v wasm-pack &> /dev/null; then
    echo "✓ wasm-pack found: $(which wasm-pack)"
else
    echo "✗ wasm-pack not found"
    echo "  Please install: curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh"
    echo "  Or: cargo install wasm-pack"
fi

echo ""
echo -e "${YELLOW}Running binding tests...${NC}"
echo "=================================="

# Parse arguments
RUN_PYTHON=true
RUN_WASM=true
TEST_PATTERN=""

for arg in "$@"; do
    case $arg in
        --python)
            RUN_WASM=false
            ;;
        --wasm)
            RUN_PYTHON=false
            ;;
        *)
            TEST_PATTERN="$arg"
            ;;
    esac
done

TOTAL_START=$(date +%s)
FAILED=false

# Python tests
if [ "$RUN_PYTHON" = true ]; then
    echo -e "\n${YELLOW}Building Python bindings...${NC}"
    
    # Try different maturin commands
    if command -v maturin &> /dev/null; then
        echo "Running: maturin develop --features python"
        if maturin develop --features python; then
            echo -e "${GREEN}✓ Python build successful${NC}"
            
            echo -e "\n${YELLOW}Running Python tests...${NC}"
            START=$(date +%s)
            
            if python tests/python/run_all_tests.py $TEST_PATTERN; then
                END=$(date +%s)
                DURATION=$((END - START))
                echo -e "${GREEN}✓ Python tests passed (${DURATION}s)${NC}"
            else
                END=$(date +%s)
                DURATION=$((END - START))
                echo -e "${RED}✗ Python tests failed (${DURATION}s)${NC}"
                FAILED=true
            fi
        else
            echo -e "${RED}✗ Python build failed${NC}"
            echo "Try running manually:"
            echo "  maturin develop --features python"
            echo "  or"
            echo "  python -m pip install maturin"
            echo "  maturin develop --features python"
            FAILED=true
        fi
    else
        echo -e "${RED}✗ maturin not available${NC}"
        FAILED=true
    fi
fi

# WASM tests
if [ "$RUN_WASM" = true ]; then
    if command -v wasm-pack &> /dev/null; then
        echo -e "\n${YELLOW}Building WASM bindings...${NC}"
        echo "Running: wasm-pack build --features wasm --target nodejs"
        if wasm-pack build --features wasm --target nodejs; then
            echo -e "${GREEN}✓ WASM build successful${NC}"
            
            echo -e "\n${YELLOW}Running WASM tests...${NC}"
            START=$(date +%s)
            
            cd tests/wasm
            if node run_all_tests.js $TEST_PATTERN; then
                END=$(date +%s)
                DURATION=$((END - START))
                echo -e "${GREEN}✓ WASM tests passed (${DURATION}s)${NC}"
            else
                END=$(date +%s)
                DURATION=$((END - START))
                echo -e "${RED}✗ WASM tests failed (${DURATION}s)${NC}"
                FAILED=true
            fi
            cd ../..
        else
            echo -e "${RED}✗ WASM build failed${NC}"
            FAILED=true
        fi
    else
        echo -e "${RED}✗ wasm-pack not available${NC}"
        echo "Skipping WASM tests"
    fi
fi

# Summary
TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - TOTAL_START))

echo -e "\n=================================="
if [ "$FAILED" = true ]; then
    echo -e "${RED}✗ Some tests failed (total time: ${TOTAL_DURATION}s)${NC}"
    exit 1
else
    echo -e "${GREEN}✓ All binding tests passed (total time: ${TOTAL_DURATION}s)${NC}"
    exit 0
fi