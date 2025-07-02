#!/bin/bash
# Run all binding tests - equivalent to 'cargo test --features nightly-avx'
# Usage:
#   ./test_bindings.sh              # Run all tests
#   ./test_bindings.sh alma         # Run only alma tests
#   ./test_bindings.sh --python     # Run only Python tests
#   ./test_bindings.sh --wasm       # Run only WASM tests

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

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
        --help|-h)
            echo "Usage: $0 [options] [test_pattern]"
            echo "Options:"
            echo "  --python     Run only Python tests"
            echo "  --wasm       Run only WASM tests"
            echo "  --help       Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                    # Run all tests"
            echo "  $0 alma               # Run only alma tests"
            echo "  $0 --python alma      # Run only Python alma tests"
            exit 0
            ;;
        -*)
            # Skip other flags
            ;;
        *)
            TEST_PATTERN="$arg"
            ;;
    esac
done

echo -e "${YELLOW}Running binding tests...${NC}"
echo "=================================="

TOTAL_START=$(date +%s)
FAILED=false

# Python tests
if [ "$RUN_PYTHON" = true ]; then
    echo -e "\n${YELLOW}Setting up Python environment...${NC}"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d ".venv" ]; then
        echo "Creating Python virtual environment..."
        # Try python3 first, then python
        if command -v python3 &> /dev/null; then
            python3 -m venv .venv --without-pip
            source .venv/bin/activate
            curl https://bootstrap.pypa.io/get-pip.py | python
        else
            python -m venv .venv
            source .venv/bin/activate
        fi
    else
        # Activate existing virtual environment
        source .venv/bin/activate
    fi
    
    # Install dependencies
    python -m pip install --upgrade pip
    python -m pip install maturin pytest pytest-xdist numpy
    
    echo -e "\n${YELLOW}Building Python bindings...${NC}"
    if maturin develop --features python --release; then
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