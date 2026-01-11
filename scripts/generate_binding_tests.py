#!/usr/bin/env python3
"""
Generate Python and WASM test files for a new indicator.
Usage: python generate_binding_tests.py <indicator_name>
"""

import sys
import os
from pathlib import Path

PYTHON_TEST_TEMPLATE = '''"""
Python binding tests for {INDICATOR_UPPER} indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path to import the built module
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'target/wheels'))

try:
    import my_project as ta_indicators
except ImportError:
    pytest.skip("Python module not built. Run 'maturin develop --features python' first", allow_module_level=True)

from test_utils import load_test_data, assert_close, EXPECTED_OUTPUTS


class Test{IndicatorTitle}:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_{indicator}_accuracy(self, test_data):
        """Test {INDICATOR_UPPER} matches expected values from Rust tests"""
        # TODO: Update expected values in test_utils.py EXPECTED_OUTPUTS
        # TODO: Implement test based on Rust check_{indicator}_accuracy
        pass
    
    def test_{indicator}_errors(self):
        """Test error handling"""
        # TODO: Implement error tests based on Rust tests
        pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
'''

WASM_TEST_TEMPLATE = '''/**
 * WASM binding tests for {INDICATOR_UPPER} indicator.
 * These tests mirror the Rust unit tests to ensure WASM bindings work correctly.
 */
const test = require('node:test');
const assert = require('node:assert');
const path = require('path');
const {{ 
    loadTestData, 
    assertArrayClose, 
    assertClose,
    isNaN,
    assertAllNaN,
    assertNoNaN,
    EXPECTED_OUTPUTS 
}} = require('./test_utils');

let wasm;
let testData;

test.before(async () => {{
    // Load WASM module
    try {{
        const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
        wasm = await import(wasmPath);
        await wasm.default();
    }} catch (error) {{
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }}
    
    testData = loadTestData();
}});

test('{INDICATOR_UPPER} accuracy', () => {{
    // TODO: Update expected values in test_utils.js EXPECTED_OUTPUTS
    // TODO: Implement test based on Rust check_{indicator}_accuracy
}});

test('{INDICATOR_UPPER} error handling', () => {{
    // TODO: Implement error tests based on Rust tests
}});

test.after(() => {{
    console.log('{INDICATOR_UPPER} WASM tests completed');
}});
'''

def generate_tests(indicator_name):
    """Generate Python and WASM test files for an indicator"""
    indicator_lower = indicator_name.lower()
    indicator_upper = indicator_name.upper()
    indicator_title = indicator_name.title()
    
    
    python_test_path = Path(f'tests/python/test_{indicator_lower}.py')
    if python_test_path.exists():
        print(f"Python test already exists: {python_test_path}")
    else:
        python_content = PYTHON_TEST_TEMPLATE.format(
            indicator=indicator_lower,
            INDICATOR_UPPER=indicator_upper,
            IndicatorTitle=indicator_title
        )
        python_test_path.write_text(python_content)
        print(f"Created: {python_test_path}")
    
    
    wasm_test_path = Path(f'tests/wasm/test_{indicator_lower}.js')
    if wasm_test_path.exists():
        print(f"WASM test already exists: {wasm_test_path}")
    else:
        wasm_content = WASM_TEST_TEMPLATE.format(
            indicator=indicator_lower,
            INDICATOR_UPPER=indicator_upper,
            IndicatorTitle=indicator_title
        )
        wasm_test_path.write_text(wasm_content)
        print(f"Created: {wasm_test_path}")
    
    print(f"\nNext steps for {indicator_name}:")
    print("1. Update EXPECTED_OUTPUTS in test_utils.py/js with values from Rust tests")
    print("2. Implement test methods based on the Rust unit tests")
    print("3. Ensure the indicator has Python/WASM bindings in the .rs file")
    print(f"4. Run tests: ./test_bindings.sh {indicator_lower}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python generate_binding_tests.py <indicator_name>")
        print("Example: python generate_binding_tests.py ema")
        sys.exit(1)
    
    indicator_name = sys.argv[1]
    generate_tests(indicator_name)

if __name__ == '__main__':
    main()