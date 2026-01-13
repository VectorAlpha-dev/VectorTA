"""
Python binding tests for Aroon indicator.
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

from test_utils import load_test_data, assert_close, EXPECTED_OUTPUTS
from rust_comparison import compare_with_rust


class TestAroon:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_aroon_partial_params(self, test_data):
        """Test Aroon with partial parameters (None values) - mirrors check_aroon_partial_params"""
        high = test_data['high']
        low = test_data['low']


        result = ta_indicators.aroon(high, low, 14)
        up, down = result
        assert len(up) == len(high)
        assert len(down) == len(low)

    def test_aroon_accuracy(self, test_data):
        """Test Aroon matches expected values from Rust tests - mirrors check_aroon_accuracy"""
        high = test_data['high']
        low = test_data['low']
        expected = EXPECTED_OUTPUTS['aroon']

        result = ta_indicators.aroon(
            high, low,
            length=expected['default_params']['length']
        )
        up, down = result

        assert len(up) == len(high)
        assert len(down) == len(low)


        assert_close(
            up[-5:],
            expected['last_5_up'],
            rtol=0.0,
            atol=1e-2,
            msg="Aroon up last 5 values mismatch"
        )
        assert_close(
            down[-5:],
            expected['last_5_down'],
            rtol=0.0,
            atol=1e-2,
            msg="Aroon down last 5 values mismatch"
        )





    def test_aroon_default_candles(self, test_data):
        """Test Aroon with default parameters - mirrors check_aroon_default_candles"""
        high = test_data['high']
        low = test_data['low']


        result = ta_indicators.aroon(high, low, 14)
        up, down = result
        assert len(up) == len(high)
        assert len(down) == len(low)

    def test_aroon_zero_length(self):
        """Test Aroon fails with zero length - mirrors check_aroon_zero_length"""
        high = np.array([10.0, 11.0, 12.0])
        low = np.array([9.0, 10.0, 11.0])

        with pytest.raises(ValueError, match="Invalid length"):
            ta_indicators.aroon(high, low, length=0)

    def test_aroon_length_exceeds_data(self):
        """Test Aroon fails when length exceeds data length - mirrors check_aroon_length_exceeds_data"""
        high = np.array([10.0, 11.0, 12.0])
        low = np.array([9.0, 10.0, 11.0])

        with pytest.raises(ValueError, match="Invalid length"):
            ta_indicators.aroon(high, low, length=14)

    def test_aroon_very_small_dataset(self):
        """Test Aroon fails with insufficient data - mirrors check_aroon_very_small_data_set"""
        high = np.array([100.0])
        low = np.array([99.5])

        with pytest.raises(ValueError, match="Invalid length|Not enough valid data"):
            ta_indicators.aroon(high, low, length=14)

    def test_aroon_empty_input(self):
        """Test Aroon fails with empty input"""
        empty = np.array([])

        with pytest.raises(ValueError, match="Input data slice is empty"):
            ta_indicators.aroon(empty, empty, length=14)

    def test_aroon_mismatched_lengths(self):
        """Test Aroon fails with mismatched input lengths"""
        high = np.array([10.0, 11.0, 12.0])
        low = np.array([9.0, 10.0])

        with pytest.raises(ValueError, match="High/low length mismatch"):
            ta_indicators.aroon(high, low, length=2)

    def test_aroon_reinput(self, test_data):
        """Test Aroon applied with different parameters - mirrors check_aroon_reinput"""
        high = test_data['high']
        low = test_data['low']


        first_result = ta_indicators.aroon(high, low, length=14)
        first_up, first_down = first_result
        assert len(first_up) == len(high)
        assert len(first_down) == len(low)


        second_result = ta_indicators.aroon(high, low, length=5)
        second_up, second_down = second_result
        assert len(second_up) == len(high)
        assert len(second_down) == len(low)

    def test_aroon_nan_handling(self, test_data):
        """Test Aroon handles NaN values correctly - mirrors check_aroon_nan_handling"""
        high = test_data['high']
        low = test_data['low']

        result = ta_indicators.aroon(high, low, length=14)
        up, down = result
        assert len(up) == len(high)
        assert len(down) == len(low)


        if len(up) > 240:
            assert not np.any(np.isnan(up[240:])), "Found unexpected NaN in up after warmup period"
            assert not np.any(np.isnan(down[240:])), "Found unexpected NaN in down after warmup period"


        expected_warmup = 14
        assert np.all(np.isnan(up[:expected_warmup])), "Expected NaN in up warmup period"
        assert np.all(np.isnan(down[:expected_warmup])), "Expected NaN in down warmup period"

    def test_aroon_streaming(self, test_data):
        """Test Aroon streaming API - mirrors check_aroon_streaming"""
        high = test_data['high']
        low = test_data['low']
        length = 14


        batch_result = ta_indicators.aroon(high, low, length=length)
        batch_up, batch_down = batch_result


        stream = ta_indicators.AroonStream(length=length)
        stream_up = []
        stream_down = []

        for h, l in zip(high, low):
            result = stream.update(h, l)
            if result is not None:
                stream_up.append(result[0])
                stream_down.append(result[1])
            else:
                stream_up.append(np.nan)
                stream_down.append(np.nan)

        stream_up = np.array(stream_up)
        stream_down = np.array(stream_down)


        assert len(batch_up) == len(stream_up)
        assert len(batch_down) == len(stream_down)


        mask_up = ~(np.isnan(batch_up) | np.isnan(stream_up))
        mask_down = ~(np.isnan(batch_down) | np.isnan(stream_down))

        if np.any(mask_up):
            assert_close(
                batch_up[mask_up],
                stream_up[mask_up],
                rtol=0.0,
                atol=1e-8,
                msg="Aroon up streaming mismatch"
            )

        if np.any(mask_down):
            assert_close(
                batch_down[mask_down],
                stream_down[mask_down],
                rtol=0.0,
                atol=1e-8,
                msg="Aroon down streaming mismatch"
            )

    def test_aroon_batch(self, test_data):
        """Test Aroon batch processing - mirrors check_batch_default_row"""
        high = test_data['high']
        low = test_data['low']

        result = ta_indicators.aroon_batch(
            high, low,
            length_range=(14, 14, 0),
        )

        assert 'up' in result
        assert 'down' in result
        assert 'lengths' in result


        up = result['up']
        down = result['down']
        assert up.shape[0] == 1
        assert up.shape[1] == len(high)
        assert down.shape[0] == 1
        assert down.shape[1] == len(low)


        default_up = up[0]
        default_down = down[0]
        expected = EXPECTED_OUTPUTS['aroon']


        assert_close(
            default_up[-5:],
            expected['last_5_up'],
            rtol=1e-2,
            msg="Aroon batch default up row mismatch"
        )
        assert_close(
            default_down[-5:],
            expected['last_5_down'],
            rtol=1e-2,
            msg="Aroon batch default down row mismatch"
        )

    def test_aroon_all_nan_input(self):
        """Test Aroon with all NaN values"""
        all_nan = np.full(100, np.nan)


        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.aroon(all_nan, all_nan, length=14)

    def test_aroon_batch_multiple_lengths(self, test_data):
        """Test Aroon batch with multiple lengths"""
        high = test_data['high'][:100]
        low = test_data['low'][:100]

        result = ta_indicators.aroon_batch(
            high, low,
            length_range=(10, 20, 5),
        )

        assert 'up' in result
        assert 'down' in result
        assert 'lengths' in result


        up = result['up']
        down = result['down']
        assert up.shape[0] == 3
        assert up.shape[1] == 100
        assert down.shape[0] == 3
        assert down.shape[1] == 100


        assert len(result['lengths']) == 3
        assert list(result['lengths']) == [10, 15, 20]


        for i, length in enumerate([10, 15, 20]):
            row_up = up[i]
            row_down = down[i]
            expected_warmup = length

            assert np.all(np.isnan(row_up[:expected_warmup])), f"Expected NaN in up warmup for length {length}"
            assert np.all(np.isnan(row_down[:expected_warmup])), f"Expected NaN in down warmup for length {length}"

            if expected_warmup < 100:
                assert not np.all(np.isnan(row_up[expected_warmup:])), f"Expected values in up after warmup for length {length}"
                assert not np.all(np.isnan(row_down[expected_warmup:])), f"Expected values in down after warmup for length {length}"

    def test_aroon_kernel_selection(self, test_data):
        """Test Aroon with different kernel selections"""
        high = test_data['high'][:100]
        low = test_data['low'][:100]


        result_scalar = ta_indicators.aroon(high, low, length=14, kernel="scalar")
        scalar_up, scalar_down = result_scalar
        assert len(scalar_up) == 100
        assert len(scalar_down) == 100


        result_auto = ta_indicators.aroon(high, low, length=14)
        auto_up, auto_down = result_auto
        assert len(auto_up) == 100
        assert len(auto_down) == 100



        mask_up = ~(np.isnan(scalar_up) | np.isnan(auto_up))
        mask_down = ~(np.isnan(scalar_down) | np.isnan(auto_down))

        if np.any(mask_up):
            assert_close(
                scalar_up[mask_up],
                auto_up[mask_up],
                rtol=0.0,
                atol=1e-10,
                msg="Kernel up results should match"
            )

        if np.any(mask_down):
            assert_close(
                scalar_down[mask_down],
                auto_down[mask_down],
                rtol=0.0,
                atol=1e-10,
                msg="Kernel down results should match"
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
