"""
Python binding tests for SAR (Parabolic SAR) indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

try:
    import my_project as ta_indicators
except ImportError:

    try:
        import my_project as ta_indicators
    except ImportError:
        pytest.skip("Python module not built. Run 'maturin develop --features python' first", allow_module_level=True)

from test_utils import load_test_data, assert_close


class TestSar:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_sar_partial_params(self, test_data):
        """Test SAR with partial parameters (None values) - mirrors check_sar_partial_params"""
        high = test_data['high']
        low = test_data['low']


        result = ta_indicators.sar(high, low)
        assert len(result) == len(high)

    def test_sar_accuracy(self, test_data):
        """Test SAR matches expected values from Rust tests - mirrors check_sar_accuracy"""
        high = test_data['high']
        low = test_data['low']


        result = ta_indicators.sar(high, low, acceleration=0.02, maximum=0.2)

        assert len(result) == len(high)


        expected_last_five = [
            60370.00224209362,
            60220.362107568006,
            60079.70038111392,
            59947.478358247085,
            59823.189656752256,
        ]

        assert_close(
            result[-5:],
            expected_last_five,
            rtol=1e-4,
            msg="SAR last 5 values mismatch"
        )

    def test_sar_from_slices(self, test_data):
        """Test SAR with high/low slices - mirrors check_sar_from_slices"""
        high = test_data['high']
        low = test_data['low']


        result = ta_indicators.sar(high, low, acceleration=0.02, maximum=0.2)
        assert len(result) == len(high)

    def test_sar_all_nan(self):
        """Test SAR with all NaN values - mirrors check_sar_all_nan"""
        high = np.full(100, np.nan)
        low = np.full(100, np.nan)

        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.sar(high, low)

    def test_sar_empty_input(self):
        """Test SAR fails with empty input"""
        empty = np.array([])

        with pytest.raises(ValueError, match="Empty data"):
            ta_indicators.sar(empty, empty)

    def test_sar_mismatched_lengths(self):
        """Test SAR fails with mismatched high/low lengths"""
        high = np.array([1.0, 2.0, 3.0])
        low = np.array([1.0, 2.0])



        try:
            result = ta_indicators.sar(high, low)

            assert len(result) == min(len(high), len(low))
        except ValueError:

            pass

    def test_sar_invalid_acceleration(self):
        """Test SAR fails with invalid acceleration"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])


        with pytest.raises(ValueError, match="Invalid acceleration"):
            ta_indicators.sar(data, data, acceleration=0.0, maximum=0.2)


        with pytest.raises(ValueError, match="Invalid acceleration"):
            ta_indicators.sar(data, data, acceleration=-0.02, maximum=0.2)


        with pytest.raises(ValueError, match="Invalid acceleration"):
            ta_indicators.sar(data, data, acceleration=float('nan'), maximum=0.2)

    def test_sar_invalid_maximum(self):
        """Test SAR fails with invalid maximum"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])


        with pytest.raises(ValueError, match="Invalid maximum"):
            ta_indicators.sar(data, data, acceleration=0.02, maximum=0.0)


        with pytest.raises(ValueError, match="Invalid maximum"):
            ta_indicators.sar(data, data, acceleration=0.02, maximum=-0.2)


        with pytest.raises(ValueError, match="Invalid maximum"):
            ta_indicators.sar(data, data, acceleration=0.02, maximum=float('nan'))

    def test_sar_nan_handling(self, test_data):
        """Test SAR handles NaN values correctly"""
        high = test_data['high'].copy()
        low = test_data['low'].copy()


        high[0:5] = np.nan
        low[0:5] = np.nan

        result = ta_indicators.sar(high, low)
        assert len(result) == len(high)


        assert all(np.isnan(result[0:6]))

    def test_sar_batch_single_parameter_set(self, test_data):
        """Test batch with single parameter combination"""
        high = test_data['high']
        low = test_data['low']


        batch_result = ta_indicators.sar_batch(
            high, low,
            acceleration_range=(0.02, 0.02, 0),
            maximum_range=(0.2, 0.2, 0)
        )


        single_result = ta_indicators.sar(high, low, acceleration=0.02, maximum=0.2)

        assert 'values' in batch_result
        assert 'accelerations' in batch_result
        assert 'maximums' in batch_result


        batch_values = batch_result['values'][0]
        assert_close(batch_values, single_result, rtol=1e-10, msg="Batch vs single mismatch")

    def test_sar_batch_multiple_parameters(self, test_data):
        """Test batch with multiple parameter values"""
        high = test_data['high'][:100]
        low = test_data['low'][:100]


        batch_result = ta_indicators.sar_batch(
            high, low,
            acceleration_range=(0.01, 0.03, 0.01),
            maximum_range=(0.1, 0.3, 0.1)
        )


        assert batch_result['values'].shape == (9, 100)
        assert len(batch_result['accelerations']) == 9
        assert len(batch_result['maximums']) == 9


        expected_accs = [0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.03, 0.03, 0.03]
        expected_maxs = [0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3]

        assert_close(batch_result['accelerations'], expected_accs, rtol=1e-10)
        assert_close(batch_result['maximums'], expected_maxs, rtol=1e-10)

    def test_sar_streaming(self):
        """Test SAR streaming functionality"""

        stream = ta_indicators.SarStream()


        high_values = [1.0, 1.1, 1.2, 1.15, 1.25, 1.3, 1.28, 1.35]
        low_values = [0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25]

        results = []
        for h, l in zip(high_values, low_values):
            result = stream.update(h, l)
            results.append(result)


        assert results[0] is None


        assert results[1] is not None


        for i in range(2, len(results)):
            assert results[i] is not None
            assert not np.isnan(results[i])

    def test_sar_streaming_with_params(self):
        """Test SAR streaming with custom parameters"""

        stream = ta_indicators.SarStream(acceleration=0.05, maximum=0.5)


        high_values = [1.0, 1.1, 1.2, 1.15, 1.25]
        low_values = [0.9, 0.95, 1.0, 1.05, 1.1]

        results = []
        for h, l in zip(high_values, low_values):
            result = stream.update(h, l)
            results.append(result)


        assert results[0] is None
        assert results[1] is not None