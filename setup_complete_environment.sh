#!/bin/bash


set -e

YELLOW='\033[1;33m'
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🔧 Complete Environment Setup for Technical Indicators Library${NC}"
echo "=============================================================="


if ! command -v cargo &> /dev/null; then
    echo -e "${RED}✗ Cargo not found. Please install Rust first:${NC}"
    echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi


echo -e "\n${BLUE}Step 1: Installing WASM target${NC}"
if rustup target list --installed | grep -q wasm32-unknown-unknown; then
    echo -e "${GREEN}✓ WASM target already installed${NC}"
else
    echo "Installing wasm32-unknown-unknown target..."
    rustup target add wasm32-unknown-unknown
    echo -e "${GREEN}✓ WASM target installed${NC}"
fi


echo -e "\n${BLUE}Step 2: Python Virtual Environment${NC}"
if [ -d ".venv" ]; then
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
else
    echo "Creating Python virtual environment..."
    python3 -m venv .venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

echo -e "${YELLOW}To activate the virtual environment, run:${NC}"
echo "  source .venv/bin/activate  # On Linux/Mac"
echo "  .venv\\Scripts\\activate     # On Windows"


if [ -n "$VIRTUAL_ENV" ]; then
    echo -e "${GREEN}✓ Virtual environment is active${NC}"


    echo -e "\n${BLUE}Step 3: Installing Python dependencies${NC}"
    pip install --upgrade pip
    pip install maturin numpy pytest
    echo -e "${GREEN}✓ Python dependencies installed${NC}"


    echo -e "\n${BLUE}Step 4: Building Python module${NC}"
    maturin develop --features python
    echo -e "${GREEN}✓ Python module built and installed${NC}"
else
    echo -e "\n${YELLOW}⚠ Virtual environment not active${NC}"
    echo "After activating the virtual environment, run:"
    echo "  pip install --upgrade pip"
    echo "  pip install maturin numpy pytest"
    echo "  maturin develop --features python"
fi


echo -e "\n${BLUE}Step 5: Installing wasm-pack${NC}"
if command -v wasm-pack &> /dev/null; then
    echo -e "${GREEN}✓ wasm-pack already installed${NC}"
else
    echo "Installing wasm-pack..."
    curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
    echo -e "${GREEN}✓ wasm-pack installed${NC}"
fi


echo -e "\n${BLUE}Step 6: Creating test files${NC}"

if [ ! -f "tests/test_zscore_integration.py" ]; then
    mkdir -p tests
    cat > tests/test_zscore_integration.py << 'EOF'
import pytest
import numpy as np
import ta_indicators

def test_zscore_values():
    """Test that zscore produces expected values"""

    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2, dtype=np.float64)

    result = ta_indicators.zscore(
        data=data,
        period=10,
        ma_type="sma",
        nbdev=1.0,
        devtype=0
    )

    assert len(result) == len(data)

    assert all(np.isnan(result[i]) for i in range(9))

    assert not np.isnan(result[9])

def test_zscore_stream():
    """Test streaming zscore calculation"""
    stream = ta_indicators.ZscoreStream(
        period=5,
        ma_type="sma",
        nbdev=2.0,
        devtype=0
    )


    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    results = []

    for val in values:
        result = stream.update(val)
        results.append(result)


    assert all(r is None for r in results[:4])

    assert all(r is not None for r in results[4:])

def test_alma_values():
    """Test ALMA indicator"""
    data = np.random.randn(100) * 100 + 1000

    result = ta_indicators.alma(
        data=data,
        period=9,
        offset=0.85,
        sigma=6.0
    )

    assert len(result) == len(data)

    non_nan_count = np.sum(~np.isnan(result))
    assert non_nan_count > len(data) - 9

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
EOF
    echo -e "${GREEN}✓ Created tests/test_zscore_integration.py${NC}"
fi


echo -e "\n${BLUE}=============================================================="
echo -e "Setup Summary:${NC}"
echo "=============================================================="

echo -e "\n${GREEN}Installed/Configured:${NC}"
echo "✓ WASM target (wasm32-unknown-unknown)"
echo "✓ Python virtual environment (.venv)"
if [ -n "$VIRTUAL_ENV" ]; then
    echo "✓ Python dependencies (maturin, numpy, pytest)"
    echo "✓ Built Python module"
fi
echo "✓ Test files"

echo -e "\n${GREEN}Quick Test Commands:${NC}"
echo "# Test Python bindings:"
if [ -z "$VIRTUAL_ENV" ]; then
    echo "source .venv/bin/activate"
fi
echo "python test_python_simple.py"
echo "pytest tests/test_zscore_integration.py -v"
echo ""
echo "# Test WASM bindings:"
echo "wasm-pack test --node --features wasm"
echo ""
echo "# Run all tests:"
echo "./test_bindings_final.sh"

echo -e "\n${GREEN}✓ Setup complete!${NC}"