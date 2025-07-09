#!/bin/bash
# Simple version without virtual environment
# Requires: pip install maturin pytest pytest-xdist numpy

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Running binding tests (no venv)...${NC}"
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
    
    # Check if maturin is installed
    if ! command -v maturin &> /dev/null; then
        echo "Installing maturin..."
        pip install --user maturin
    fi
    
    # Build and install the module
    if maturin build --features python --release; then
        echo "Installing wheel..."
        pip install --user --force-reinstall target/wheels/*.whl
        echo -e "${GREEN}✓ Python build successful${NC}"
        
        echo -e "\n${YELLOW}Running Python tests...${NC}"
        START=$(date +%s)
        
        # Run tests
        if python -m pytest tests/python/test_alma.py -v; then
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
        FAILED=true
    fi
fi

# WASM tests
if [ "$RUN_WASM" = true ]; then
    echo -e "\n${YELLOW}Building WASM bindings...${NC}"
    if wasm-pack build -- --features wasm --no-default-features; then
        echo -e "${GREEN}✓ WASM build successful${NC}"
        
        echo -e "\n${YELLOW}Running WASM tests...${NC}"
        START=$(date +%s)
        
        cd tests/wasm
        npm install
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