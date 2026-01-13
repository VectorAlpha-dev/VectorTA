#!/bin/bash


set -e

YELLOW='\033[1;33m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}🧪 Running Binding Tests (Updated for PyO3 0.20+)${NC}"
echo "======================================================="


run_test() {
    local test_name=$1
    local test_cmd=$2

    echo -e "\n${YELLOW}Running $test_name...${NC}"
    if eval "$test_cmd"; then
        echo -e "${GREEN}✓ $test_name passed${NC}"
        return 0
    else
        echo -e "${RED}✗ $test_name failed${NC}"
        return 1
    fi
}


failed_tests=()


if ! command -v cargo &> /dev/null; then

    if [ -f "$HOME/.cargo/bin/cargo" ]; then
        export PATH="$HOME/.cargo/bin:$PATH"
    elif [ -f "/usr/local/cargo/bin/cargo" ]; then
        export PATH="/usr/local/cargo/bin:$PATH"
    else
        echo -e "${RED}✗ Cargo not found. Please install Rust.${NC}"
        exit 1
    fi
fi


if ! run_test "Rust native tests (no features)" "cargo test --lib --no-default-features 2>&1 | tail -20"; then
    failed_tests+=("Rust native")
fi


echo -e "\n${YELLOW}Testing Python binding compilation...${NC}"
if cargo check --features python 2>&1 | tail -20; then
    echo -e "${GREEN}✓ Python bindings compile${NC}"
else
    echo -e "${RED}✗ Python bindings have compilation errors${NC}"
    failed_tests+=("Python compilation")
fi


echo -e "\n${YELLOW}Testing WASM binding compilation...${NC}"
if cargo check --features wasm --target wasm32-unknown-unknown 2>&1 | tail -20; then
    echo -e "${GREEN}✓ WASM bindings compile${NC}"
else
    echo -e "${YELLOW}⚠️  WASM target not installed or compilation errors${NC}"
    echo "   Install with: rustup target add wasm32-unknown-unknown"
fi


if command -v python3 &> /dev/null; then
    echo -e "\n${YELLOW}Python environment:${NC}"
    python3 --version


    if [ -n "$VIRTUAL_ENV" ]; then
        echo -e "${GREEN}✓ Virtual environment active: $VIRTUAL_ENV${NC}"
    else
        echo -e "${YELLOW}⚠️  No virtual environment active${NC}"
        echo "   Consider creating one with: python3 -m venv .venv && source .venv/bin/activate"
    fi


    if python3 -m pip show maturin &> /dev/null; then
        echo -e "${GREEN}✓ maturin is installed${NC}"


        echo -e "\n${YELLOW}Building Python module with maturin...${NC}"
        if [ -n "$VIRTUAL_ENV" ] || [ -d ".venv" ]; then
            if maturin develop --features python 2>&1 | tail -10; then
                echo -e "${GREEN}✓ Python module built successfully${NC}"


                if [ -f "test_python_simple.py" ]; then
                    if ! run_test "Simple Python integration test" "python3 test_python_simple.py"; then
                        failed_tests+=("Python integration")
                    fi
                fi
            else
                echo -e "${RED}✗ Failed to build Python module${NC}"
                failed_tests+=("Python module build")
            fi
        else
            echo -e "${YELLOW}Skipping maturin develop (no virtual environment)${NC}"
            echo "Building wheel instead..."
            if maturin build --features python --release 2>&1 | tail -10; then
                echo -e "${GREEN}✓ Python wheel built successfully${NC}"
                echo "   Install with: pip install target/wheels/*.whl"
            else
                echo -e "${RED}✗ Failed to build Python wheel${NC}"
                failed_tests+=("Python wheel build")
            fi
        fi
    else
        echo -e "${YELLOW}⚠️  maturin not installed${NC}"
        echo "   Install with: pip install maturin"
    fi
else
    echo -e "${YELLOW}⚠️  Python 3 not found${NC}"
fi


echo -e "\n${YELLOW}=======================================================${NC}"
echo -e "${YELLOW}Test Summary:${NC}"
echo -e "${YELLOW}=======================================================${NC}"

if [ ${
    echo -e "${GREEN}✓ All available tests passed!${NC}"
    echo -e "\n${GREEN}Next steps:${NC}"
    echo "1. If you haven't already, create a virtual environment:"
    echo "   python3 -m venv .venv && source .venv/bin/activate"
    echo "2. Install development dependencies:"
    echo "   pip install maturin numpy pytest"
    echo "3. Build and test the Python module:"
    echo "   maturin develop --features python"
    echo "   python test_python_simple.py"
    exit 0
else
    echo -e "${RED}✗ Failed tests:${NC}"
    for test in "${failed_tests[@]}"; do
        echo -e "  ${RED}- $test${NC}"
    done
    echo -e "\n${YELLOW}Troubleshooting tips:${NC}"
    echo "1. Check that you have the correct Rust toolchain:"
    echo "   rustup default stable"
    echo "2. Update your dependencies:"
    echo "   cargo update"
    echo "3. Check the compilation errors above for specific issues"
    exit 1
fi