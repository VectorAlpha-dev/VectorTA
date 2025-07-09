#!/bin/bash
# Setup environment and run tests

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Setting up environment...${NC}"

# Source Rust environment
if [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
    echo "✓ Rust environment loaded"
else
    echo -e "${RED}✗ Rust not found. Please install from https://rustup.rs/${NC}"
    exit 1
fi

# Check cargo
if ! command -v cargo &> /dev/null; then
    echo -e "${RED}✗ cargo not found in PATH${NC}"
    exit 1
fi
echo "✓ cargo found: $(cargo --version)"

# Install wasm-pack if not present
if ! command -v wasm-pack &> /dev/null; then
    echo "Installing wasm-pack..."
    cargo install wasm-pack
fi

# Install Python packages
echo -e "\n${YELLOW}Installing Python dependencies...${NC}"
pip install --user maturin pytest pytest-xdist numpy

echo -e "\n${YELLOW}Running tests...${NC}"
echo "=================================="

FAILED=false

# Python tests
echo -e "\n${YELLOW}Building Python bindings...${NC}"
if python -m maturin build --features python --release; then
    echo "Installing wheel..."
    pip install --user --force-reinstall target/wheels/*.whl
    echo -e "${GREEN}✓ Python build successful${NC}"
    
    echo -e "\n${YELLOW}Running Python tests...${NC}"
    if python -m pytest tests/python/test_alma.py -v; then
        echo -e "${GREEN}✓ Python tests passed${NC}"
    else
        echo -e "${RED}✗ Python tests failed${NC}"
        FAILED=true
    fi
else
    echo -e "${RED}✗ Python build failed${NC}"
    FAILED=true
fi

# WASM tests
echo -e "\n${YELLOW}Building WASM bindings...${NC}"
if wasm-pack build -- --features wasm --no-default-features; then
    echo -e "${GREEN}✓ WASM build successful${NC}"
    
    echo -e "\n${YELLOW}Running WASM tests...${NC}"
    cd tests/wasm
    npm install
    if node run_all_tests.js; then
        echo -e "${GREEN}✓ WASM tests passed${NC}"
    else
        echo -e "${RED}✗ WASM tests failed${NC}"
        FAILED=true
    fi
    cd ../..
else
    echo -e "${RED}✗ WASM build failed${NC}"
    echo "Make sure you have wasm target: rustup target add wasm32-unknown-unknown"
    FAILED=true
fi

echo -e "\n=================================="
if [ "$FAILED" = true ]; then
    echo -e "${RED}✗ Some tests failed${NC}"
    exit 1
else
    echo -e "${GREEN}✓ All tests passed!${NC}"
    exit 0
fi