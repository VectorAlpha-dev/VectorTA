#!/bin/bash
# Final test runner with proper error detection and setup instructions

set -e

YELLOW='\033[1;33m'
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🧪 Testing Python and WASM Bindings for Technical Indicators${NC}"
echo "============================================================"

# Function to run tests and report results
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

# Check for cargo
if ! command -v cargo &> /dev/null; then
    if [ -f "$HOME/.cargo/bin/cargo" ]; then
        export PATH="$HOME/.cargo/bin:$PATH"
    else
        echo -e "${RED}✗ Cargo not found. Please install Rust from https://rustup.rs${NC}"
        exit 1
    fi
fi

echo -e "${BLUE}Step 1: Testing Basic Rust Compilation${NC}"
echo "---------------------------------------"

# Test basic compilation
if cargo check --quiet 2>/dev/null; then
    echo -e "${GREEN}✓ Basic Rust compilation successful${NC}"
else
    echo -e "${RED}✗ Basic Rust compilation failed${NC}"
    echo "Please fix Rust compilation errors first"
    exit 1
fi

echo -e "\n${BLUE}Step 2: Testing Python Bindings${NC}"
echo "--------------------------------"

# Check Python bindings compilation
echo -n "Checking Python binding compilation... "
if cargo check --features python --quiet 2>/dev/null; then
    echo -e "${GREEN}✓ Success${NC}"
    PYTHON_BINDINGS_OK=true
else
    echo -e "${RED}✗ Failed${NC}"
    echo "Python bindings have compilation errors. Running detailed check:"
    cargo check --features python 2>&1 | tail -30
    PYTHON_BINDINGS_OK=false
fi

# Python environment check
if command -v python3 &> /dev/null; then
    echo -e "\n${BLUE}Python Environment:${NC}"
    echo "  Python version: $(python3 --version)"
    
    # Check for virtual environment
    if [ -n "$VIRTUAL_ENV" ]; then
        echo -e "  ${GREEN}✓ Virtual environment active: $(basename $VIRTUAL_ENV)${NC}"
        VENV_ACTIVE=true
    elif [ -d ".venv" ]; then
        echo -e "  ${YELLOW}⚠ Virtual environment found but not activated${NC}"
        echo -e "  ${YELLOW}  Run: source .venv/bin/activate${NC}"
        VENV_ACTIVE=false
    else
        echo -e "  ${YELLOW}⚠ No virtual environment found${NC}"
        echo -e "  ${YELLOW}  Create one with: python3 -m venv .venv${NC}"
        VENV_ACTIVE=false
    fi
    
    # Check for maturin
    if python3 -c "import maturin" 2>/dev/null || command -v maturin &> /dev/null; then
        echo -e "  ${GREEN}✓ maturin is installed${NC}"
        MATURIN_OK=true
    else
        echo -e "  ${YELLOW}⚠ maturin not installed${NC}"
        echo -e "  ${YELLOW}  Install with: pip install maturin${NC}"
        MATURIN_OK=false
    fi
    
    # Try to build if everything is OK
    if [ "$PYTHON_BINDINGS_OK" = true ] && [ "$MATURIN_OK" = true ]; then
        echo -e "\n${BLUE}Building Python module:${NC}"
        
        if [ "$VENV_ACTIVE" = true ]; then
            echo "Using maturin develop (virtual environment active)..."
            if maturin develop --features python --quiet 2>/dev/null; then
                echo -e "${GREEN}✓ Python module built and installed${NC}"
                
                # Run simple test
                echo -e "\n${BLUE}Testing Python module:${NC}"
                if python3 -c "import ta_indicators; print('✓ Module imports successfully')" 2>/dev/null; then
                    echo -e "${GREEN}✓ Python module works${NC}"
                    
                    if [ -f "test_python_simple.py" ]; then
                        echo "Running integration tests..."
                        python3 test_python_simple.py || true
                    fi
                else
                    echo -e "${RED}✗ Python module import failed${NC}"
                fi
            else
                echo -e "${RED}✗ Failed to build Python module${NC}"
                echo "Trying detailed build..."
                maturin develop --features python 2>&1 | tail -20
            fi
        else
            echo "Building wheel (no virtual environment)..."
            if maturin build --features python --release --quiet 2>/dev/null; then
                echo -e "${GREEN}✓ Python wheel built${NC}"
                echo -e "  Install with: pip install target/wheels/*.whl"
            else
                echo -e "${RED}✗ Failed to build Python wheel${NC}"
            fi
        fi
    fi
else
    echo -e "${YELLOW}⚠ Python 3 not found${NC}"
fi

echo -e "\n${BLUE}Step 3: Testing WASM Bindings${NC}"
echo "------------------------------"

# Check if wasm target is installed
if rustup target list --installed | grep -q wasm32-unknown-unknown; then
    echo -e "${GREEN}✓ WASM target installed${NC}"
    
    # Test WASM compilation
    echo -n "Checking WASM binding compilation... "
    if cargo check --features wasm --target wasm32-unknown-unknown --quiet 2>/dev/null; then
        echo -e "${GREEN}✓ Success${NC}"
        
        # Check for wasm-pack
        if command -v wasm-pack &> /dev/null; then
            echo -e "${GREEN}✓ wasm-pack installed${NC}"
            echo "You can run WASM tests with:"
            echo "  wasm-pack test --node --features wasm"
            echo "  wasm-pack test --chrome --headless --features wasm"
        else
            echo -e "${YELLOW}⚠ wasm-pack not installed${NC}"
            echo "  Install with: curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh"
        fi
    else
        echo -e "${RED}✗ Failed${NC}"
        echo "WASM bindings have compilation errors"
    fi
else
    echo -e "${YELLOW}⚠ WASM target not installed${NC}"
    echo "  Install with: rustup target add wasm32-unknown-unknown"
fi

echo -e "\n${BLUE}Step 4: Running Rust Tests${NC}"
echo "--------------------------"

# Run basic tests
echo "Running unit tests..."
if cargo test --lib --quiet -- --test-threads=1 2>/dev/null; then
    echo -e "${GREEN}✓ All Rust tests passed${NC}"
else
    echo -e "${YELLOW}⚠ Some tests failed${NC}"
    echo "Run 'cargo test --lib' for details"
fi

# Summary and next steps
echo -e "\n${BLUE}============================================================${NC}"
echo -e "${BLUE}Summary and Next Steps:${NC}"
echo -e "${BLUE}============================================================${NC}"

echo -e "\n${GREEN}Setup Instructions:${NC}"
echo "1. Create and activate a Python virtual environment:"
echo "   python3 -m venv .venv"
echo "   source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate"
echo ""
echo "2. Install Python dependencies:"
echo "   pip install maturin numpy pytest"
echo ""
echo "3. Build and test Python bindings:"
echo "   maturin develop --features python"
echo "   python test_python_simple.py"
echo ""
echo "4. Install WASM target and tools:"
echo "   rustup target add wasm32-unknown-unknown"
echo "   curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh"
echo ""
echo "5. Test WASM bindings:"
echo "   wasm-pack test --node --features wasm"

if [ "$PYTHON_BINDINGS_OK" = true ] && [ "$VENV_ACTIVE" = true ] && [ "$MATURIN_OK" = true ]; then
    echo -e "\n${GREEN}✓ Your environment is properly set up for Python development!${NC}"
else
    echo -e "\n${YELLOW}⚠ Some setup steps are still needed (see above)${NC}"
fi

echo -e "\n${BLUE}For detailed documentation, see:${NC}"
echo "  - tests/python_bindings_test_guide.md"
echo "  - tests/bindings_test_setup.md"