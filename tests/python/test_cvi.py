"""
Python binding tests for CVI indicator.
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


class TestCvi:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_cvi_accuracy(self, test_data):
        """Test CVI matches expected values from Rust tests"""
        # TODO: Update expected values in test_utils.py EXPECTED_OUTPUTS
        # TODO: Implement test based on Rust check_cvi_accuracy
        pass
    
    def test_cvi_errors(self):
        """Test error handling"""
        # TODO: Implement error tests based on Rust tests
        pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
