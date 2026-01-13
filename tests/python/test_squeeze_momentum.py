"""
Python binding tests for Squeeze Momentum indicator.
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
from rust_comparison import compare_with_rust


class TestSqueezeMomentum:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_squeeze_momentum_partial_params(self, test_data):
        """Test Squeeze Momentum with partial parameters (None values) - mirrors check_smi_partial_params"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']


        squeeze, momentum, momentum_signal = ta_indicators.squeeze_momentum(
            high, low, close
        )
        assert len(squeeze) == len(close)
        assert len(momentum) == len(close)
        assert len(momentum_signal) == len(close)

    def test_squeeze_momentum_accuracy(self, test_data):
        """Test Squeeze Momentum matches expected values from Rust tests - mirrors check_smi_accuracy"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        expected_last_five = [-170.9, -155.4, -65.3, -61.1, -178.1]


        squeeze, momentum, momentum_signal = ta_indicators.squeeze_momentum(
            high, low, close,
            length_bb=20,
            mult_bb=2.0,
            length_kc=20,
            mult_kc=1.5
        )

        assert len(squeeze) == len(close)
        assert len(momentum) == len(close)
        assert len(momentum_signal) == len(close)


        last_5_momentum = momentum[-5:]
        for i, (computed, expected) in enumerate(zip(last_5_momentum, expected_last_five)):
            if np.isnan(expected):
                assert np.isnan(computed), f"Expected NaN at index {i}, got {computed}"
            else:
                assert_close(computed, expected, rtol=1e-1,
                           msg=f"SMI momentum mismatch at index {i}")

    def test_squeeze_momentum_default_candles(self, test_data):
        """Test Squeeze Momentum with default parameters - mirrors check_smi_default_candles"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']


        squeeze, momentum, momentum_signal = ta_indicators.squeeze_momentum(
            high, low, close
        )
        assert len(squeeze) == len(close)
        assert len(momentum) == len(close)
        assert len(momentum_signal) == len(close)

    def test_squeeze_momentum_zero_length(self):
        """Test Squeeze Momentum fails with zero length - mirrors check_smi_zero_length"""
        high = np.array([10.0, 20.0, 30.0])
        low = np.array([10.0, 20.0, 30.0])
        close = np.array([10.0, 20.0, 30.0])

        with pytest.raises(ValueError, match="Invalid length"):
            ta_indicators.squeeze_momentum(high, low, close, length_bb=0, length_kc=0)

    def test_squeeze_momentum_length_exceeds(self):
        """Test Squeeze Momentum fails when length exceeds data - mirrors check_smi_length_exceeds"""
        high = np.array([10.0, 20.0, 30.0])
        low = np.array([10.0, 20.0, 30.0])
        close = np.array([10.0, 20.0, 30.0])

        with pytest.raises(ValueError, match="Invalid length"):
            ta_indicators.squeeze_momentum(high, low, close, length_bb=10, length_kc=10)

    def test_squeeze_momentum_all_nan(self):
        """Test Squeeze Momentum fails with all NaN - mirrors check_smi_all_nan"""
        high = np.array([np.nan, np.nan, np.nan])
        low = np.array([np.nan, np.nan, np.nan])
        close = np.array([np.nan, np.nan, np.nan])

        with pytest.raises(ValueError):
            ta_indicators.squeeze_momentum(high, low, close)

    def test_squeeze_momentum_inconsistent_lengths(self):
        """Test Squeeze Momentum fails with inconsistent data lengths - mirrors check_smi_inconsistent_lengths"""
        high = np.array([1.0, 2.0, 3.0])
        low = np.array([1.0, 2.0])
        close = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="inconsistent lengths"):
            ta_indicators.squeeze_momentum(high, low, close)

    def test_squeeze_momentum_minimum_data(self):
        """Test Squeeze Momentum with minimum required data - mirrors check_smi_minimum_data"""
        high = np.array([10.0, 12.0, 14.0])
        low = np.array([5.0, 6.0, 7.0])
        close = np.array([7.0, 11.0, 10.0])


        with pytest.raises(ValueError):
            ta_indicators.squeeze_momentum(
                high, low, close,
                length_bb=5,
                mult_bb=2.0,
                length_kc=5,
                mult_kc=1.5
            )

    def test_squeeze_momentum_stream(self):
        """Test Squeeze Momentum streaming functionality"""

        stream = ta_indicators.SqueezeMomentumStream()


        squeeze, momentum, signal = stream.update(100.0, 95.0, 98.0)
        assert squeeze is None
        assert momentum is None
        assert signal is None


        test_data = [
            (102.0, 96.0, 99.0),
            (105.0, 98.0, 103.0),
            (104.0, 99.0, 102.0),
            (106.0, 100.0, 104.0),
            (108.0, 102.0, 107.0),
            (107.0, 103.0, 105.0),
            (109.0, 104.0, 108.0),
            (110.0, 105.0, 109.0),
            (112.0, 106.0, 110.0),
            (111.0, 107.0, 109.0),
            (113.0, 108.0, 112.0),
            (114.0, 109.0, 113.0),
            (115.0, 110.0, 114.0),
            (116.0, 111.0, 115.0),
            (117.0, 112.0, 116.0),
            (118.0, 113.0, 117.0),
            (119.0, 114.0, 118.0),
            (120.0, 115.0, 119.0),
            (121.0, 116.0, 120.0),
        ]

        for high, low, close in test_data:
            squeeze, momentum, signal = stream.update(high, low, close)


        assert squeeze is not None
        assert momentum is not None
        assert signal is not None

    def test_squeeze_momentum_batch_single_param(self, test_data):
        """Test batch operation with single parameter combination"""
        high = test_data['high'][:100]
        low = test_data['low'][:100]
        close = test_data['close'][:100]


        result = ta_indicators.squeeze_momentum_batch(
            high, low, close,
            length_bb_range=(20, 20, 0),
            mult_bb_range=(2.0, 2.0, 0.0),
            length_kc_range=(20, 20, 0),
            mult_kc_range=(1.5, 1.5, 0.0)
        )


        squeeze, momentum, momentum_signal = ta_indicators.squeeze_momentum(
            high, low, close,
            length_bb=20,
            mult_bb=2.0,
            length_kc=20,
            mult_kc=1.5
        )

        assert result['values'].shape == (1, 100)
        np.testing.assert_allclose(result['values'][0], momentum, equal_nan=True)
        assert len(result['length_bb']) == 1
        assert result['length_bb'][0] == 20
        assert len(result['mult_bb']) == 1
        assert result['mult_bb'][0] == 2.0
        assert len(result['length_kc']) == 1
        assert result['length_kc'][0] == 20
        assert len(result['mult_kc']) == 1
        assert result['mult_kc'][0] == 1.5

    def test_squeeze_momentum_batch_multiple_params(self, test_data):
        """Test batch operation with multiple parameter combinations"""
        high = test_data['high'][:50]
        low = test_data['low'][:50]
        close = test_data['close'][:50]


        result = ta_indicators.squeeze_momentum_batch(
            high, low, close,
            length_bb_range=(15, 25, 5),
            mult_bb_range=(2.0, 2.0, 0.0),
            length_kc_range=(20, 20, 0),
            mult_kc_range=(1.0, 2.0, 0.5)
        )


        assert result['values'].shape == (9, 50)
        assert len(result['length_bb']) == 9
        assert len(result['mult_bb']) == 9
        assert len(result['length_kc']) == 9
        assert len(result['mult_kc']) == 9


        expected_length_bb = [15, 15, 15, 20, 20, 20, 25, 25, 25]
        expected_mult_kc = [1.0, 1.5, 2.0] * 3

        np.testing.assert_array_equal(result['length_bb'], expected_length_bb)
        np.testing.assert_allclose(result['mult_kc'], expected_mult_kc)

    def test_squeeze_momentum_edge_cases(self):
        """Test edge cases for Squeeze Momentum"""

        high = np.ones(20) * 100
        low = np.ones(20) * 90
        close = np.ones(20) * 95

        squeeze, momentum, momentum_signal = ta_indicators.squeeze_momentum(
            high, low, close,
            length_bb=20,
            length_kc=20
        )

        assert len(squeeze) == 20
        assert len(momentum) == 20
        assert len(momentum_signal) == 20


        assert all(np.isnan(squeeze[:19]))
        assert all(np.isnan(momentum[:19]))
        assert all(np.isnan(momentum_signal[:19]))


        with pytest.raises(ValueError):
            ta_indicators.squeeze_momentum(
                np.array([]), np.array([]), np.array([])
            )

    def test_squeeze_momentum_rust_comparison(self, test_data):
        """Compare Python binding output with Rust output"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']


        squeeze, momentum, momentum_signal = ta_indicators.squeeze_momentum(
            high, low, close
        )




        compare_with_rust('squeeze_momentum', momentum, 'hlc', {
            'length_bb': 20,
            'mult_bb': 2.0,
            'length_kc': 20,
            'mult_kc': 1.5
        })