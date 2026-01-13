"""
Python binding tests for ULTOSC indicator.
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

from test_utils import load_test_data, assert_close


class TestUltOsc:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_ultosc_accuracy(self, test_data):
        """Test ULTOSC matches expected values from Rust tests"""
        high = np.array(test_data['high'], dtype=np.float64)
        low = np.array(test_data['low'], dtype=np.float64)
        close = np.array(test_data['close'], dtype=np.float64)


        result = ta_indicators.ultosc(high, low, close)

        assert len(result) == len(close)


        expected_last_five = [
            41.25546890298435,
            40.83865967175865,
            48.910324164909625,
            45.43113094857947,
            42.163165136766295,
        ]


        for i in range(5):
            assert_close(result[-(5-i)], expected_last_five[i], rtol=0, atol=1e-8)

    def test_ultosc_custom_params(self, test_data):
        """Test ULTOSC with custom parameters"""
        high = np.array(test_data['high'], dtype=np.float64)
        low = np.array(test_data['low'], dtype=np.float64)
        close = np.array(test_data['close'], dtype=np.float64)


        result = ta_indicators.ultosc(high, low, close, timeperiod1=5, timeperiod2=10, timeperiod3=20)

        assert len(result) == len(close)
        assert not np.isnan(result[-1])

    def test_ultosc_batch(self, test_data):
        """Test batch calculation"""
        high = np.array(test_data['high'][:100], dtype=np.float64)
        low = np.array(test_data['low'][:100], dtype=np.float64)
        close = np.array(test_data['close'][:100], dtype=np.float64)


        result = ta_indicators.ultosc_batch(
            high, low, close,
            timeperiod1_range=(5, 9, 2),
            timeperiod2_range=(12, 16, 2),
            timeperiod3_range=(26, 30, 2)
        )

        assert 'values' in result
        assert 'timeperiod1' in result
        assert 'timeperiod2' in result
        assert 'timeperiod3' in result

        values = np.array(result['values'])
        assert values.shape == (27, len(close))


        timeperiod1s = np.array(result['timeperiod1'])
        timeperiod2s = np.array(result['timeperiod2'])
        timeperiod3s = np.array(result['timeperiod3'])

        assert len(timeperiod1s) == 27
        assert len(timeperiod2s) == 27
        assert len(timeperiod3s) == 27


        target_idx = None
        for i in range(27):
            if timeperiod1s[i] == 7 and timeperiod2s[i] == 14 and timeperiod3s[i] == 28:
                target_idx = i
                break

        assert target_idx is not None, "Could not find (7, 14, 28) combination"


        single_result = ta_indicators.ultosc(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
        np.testing.assert_array_almost_equal(values[target_idx], single_result, decimal=10)

    def test_ultosc_stream(self):
        """Test streaming functionality"""
        stream = ta_indicators.UltOscStream()


        test_prices = [
            (10.5, 9.5, 10.0),
            (11.0, 9.8, 10.5),
            (11.5, 10.0, 11.0),
            (12.0, 10.5, 11.5),
            (11.8, 10.8, 11.2),
        ]

        results = []
        for high, low, close in test_prices:
            result = stream.update(high, low, close)
            results.append(result)


        assert results[0] is None
        assert results[1] is None


        stream2 = ta_indicators.UltOscStream(timeperiod1=3, timeperiod2=5, timeperiod3=7)


        for high, low, close in test_prices:
            result = stream2.update(high, low, close)

    def test_ultosc_errors(self):
        """Test error handling"""

        with pytest.raises(Exception):
            ta_indicators.ultosc(np.array([]), np.array([]), np.array([]))


        with pytest.raises(Exception):
            ta_indicators.ultosc(
                np.array([1.0, 2.0]),
                np.array([0.5, 1.5, 2.5]),
                np.array([0.8, 1.8])
            )


        high = np.array([10.0, 11.0, 12.0], dtype=np.float64)
        low = np.array([9.0, 10.0, 11.0], dtype=np.float64)
        close = np.array([9.5, 10.5, 11.5], dtype=np.float64)

        with pytest.raises(Exception):
            ta_indicators.ultosc(high, low, close, timeperiod1=0, timeperiod2=14, timeperiod3=28)


        with pytest.raises(Exception):
            ta_indicators.ultosc(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=50)

    def test_ultosc_nan_handling(self, test_data):
        """Test handling of NaN values"""
        high = np.array(test_data['high'], dtype=np.float64)
        low = np.array(test_data['low'], dtype=np.float64)
        close = np.array(test_data['close'], dtype=np.float64)


        high[0:5] = np.nan
        low[0:5] = np.nan
        close[0:5] = np.nan

        result = ta_indicators.ultosc(high, low, close)

        assert len(result) == len(close)

        for i in range(10):
            assert np.isnan(result[i])

    def test_ultosc_kernel_selection(self, test_data):
        """Test different kernel selections"""
        high = np.array(test_data['high'][:100], dtype=np.float64)
        low = np.array(test_data['low'][:100], dtype=np.float64)
        close = np.array(test_data['close'][:100], dtype=np.float64)


        result_auto = ta_indicators.ultosc(high, low, close, kernel=None)
        result_scalar = ta_indicators.ultosc(high, low, close, kernel='scalar')


        np.testing.assert_array_almost_equal(result_auto, result_scalar, decimal=10)


        try:
            result_avx2 = ta_indicators.ultosc(high, low, close, kernel='avx2')
            np.testing.assert_array_almost_equal(result_auto, result_avx2, decimal=10)
        except:
            pass

    def test_ultosc_consistency(self, test_data):
        """Test that repeated calculations give same results"""
        high = np.array(test_data['high'][:50], dtype=np.float64)
        low = np.array(test_data['low'][:50], dtype=np.float64)
        close = np.array(test_data['close'][:50], dtype=np.float64)


        result1 = ta_indicators.ultosc(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
        result2 = ta_indicators.ultosc(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)


        np.testing.assert_array_equal(result1, result2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
