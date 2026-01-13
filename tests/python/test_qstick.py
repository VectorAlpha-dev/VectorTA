"""
Python binding tests for QSTICK indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import pytest
import numpy as np
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'target/wheels'))

try:
    import my_project as ta_indicators
except ImportError:
    pytest.skip("Python module not built. Run 'maturin develop --features python' first", allow_module_level=True)

from test_utils import load_test_data, assert_close, EXPECTED_OUTPUTS


class TestQstick:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_qstick_accuracy(self, test_data):
        """Test QSTICK matches expected values from Rust tests"""
        open_data = test_data['open']
        close_data = test_data['close']
        period = 5


        result = ta_indicators.qstick(open_data, close_data, period)


        expected_last_five = [219.4, 61.6, -51.8, -53.4, -123.2]


        for i, expected in enumerate(expected_last_five):
            actual = result[-(5-i)]

            assert_close(actual, expected, rtol=0.0, atol=1e-1,
                         msg=f"QSTICK mismatch at index {i}: expected {expected}, got {actual}")

    def test_qstick_errors(self):
        """Test error handling"""

        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.qstick(np.array([10.0, 20.0]), np.array([15.0, 25.0]), 0)


        with pytest.raises(ValueError, match="Invalid period|Not enough"):
            ta_indicators.qstick(np.array([10.0, 20.0]), np.array([15.0, 25.0]), 10)


        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.qstick(np.array([10.0, 20.0, 30.0]), np.array([15.0, 25.0]), 5)


        with pytest.raises(ValueError):
            ta_indicators.qstick(np.array([]), np.array([]), 5)

    def test_qstick_batch(self, test_data):
        """Test batch processing"""
        open_data = test_data['open']
        close_data = test_data['close']


        result = ta_indicators.qstick_batch(open_data, close_data, (5, 20, 5))

        assert 'values' in result
        assert 'periods' in result
        assert result['values'].shape[0] == 4
        assert result['values'].shape[1] == len(open_data)
        assert len(result['periods']) == 4


        single_result = ta_indicators.qstick(open_data, close_data, 5)
        np.testing.assert_array_almost_equal(result['values'][0], single_result, decimal=10)

    def test_qstick_stream(self):
        """Test streaming functionality"""
        stream = ta_indicators.QstickStream(5)


        test_data = [
            (10.0, 15.0),
            (15.0, 20.0),
            (20.0, 18.0),
            (18.0, 22.0),
            (22.0, 25.0),
            (25.0, 23.0)
        ]

        results = []
        for open_val, close_val in test_data:
            result = stream.update(open_val, close_val)
            results.append(result)


        assert all(r is None for r in results[:4])


        assert results[4] is not None
        assert results[5] is not None



        diffs = [close - open for open, close in test_data[:5]]
        expected = sum(diffs) / 5
        assert_close(results[4], expected, 1e-10)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
