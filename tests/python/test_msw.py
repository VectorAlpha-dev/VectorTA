"""
Python binding tests for MSW (Mesa Sine Wave) indicator.
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


class TestMsw:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_msw_partial_params(self, test_data):
        """Test MSW with partial parameters (None values) - mirrors check_msw_partial_params"""
        close = test_data['close']


        sine, lead = ta_indicators.msw(close, 5)
        assert isinstance(sine, np.ndarray)
        assert isinstance(lead, np.ndarray)
        assert len(sine) == len(close)
        assert len(lead) == len(close)

    def test_msw_accuracy(self, test_data):
        """Test MSW matches expected values from Rust tests - mirrors check_msw_accuracy"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['msw']
        period = expected['default_params']['period']

        sine, lead = ta_indicators.msw(close, period=period)

        assert len(sine) == len(close)
        assert len(lead) == len(close)


        assert_close(
            sine[-5:],
            expected['last_5_sine'],
            rtol=1e-1,
            msg="MSW sine last 5 values mismatch"
        )
        assert_close(
            lead[-5:],
            expected['last_5_lead'],
            rtol=1e-1,
            msg="MSW lead last 5 values mismatch"
        )

    def test_msw_default_candles(self, test_data):
        """Test MSW with default parameters - mirrors check_msw_default_candles"""
        close = test_data['close']


        sine, lead = ta_indicators.msw(close, 5)
        assert len(sine) == len(close)
        assert len(lead) == len(close)

    def test_msw_zero_period(self):
        """Test MSW fails with zero period - mirrors check_msw_zero_period"""
        data = np.array([10.0, 20.0, 30.0])

        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.msw(data, period=0)

    def test_msw_period_exceeds_length(self):
        """Test MSW fails when period exceeds data length - mirrors check_msw_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])

        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.msw(data_small, period=10)

    def test_msw_very_small_dataset(self):
        """Test MSW fails with insufficient data - mirrors check_msw_very_small_dataset"""
        single_point = np.array([42.0])

        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.msw(single_point, period=5)

    def test_msw_empty_input(self):
        """Test MSW fails with empty input"""
        empty = np.array([])

        with pytest.raises(ValueError, match="Empty data"):
            ta_indicators.msw(empty, period=5)

    def test_msw_nan_handling(self, test_data):
        """Test MSW handles NaN values correctly - mirrors check_msw_nan_handling"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['msw']
        period = expected['default_params']['period']

        sine, lead = ta_indicators.msw(close, period=period)
        assert len(sine) == len(close)
        assert len(lead) == len(close)


        expected_warmup = expected['warmup_period']
        assert np.all(np.isnan(sine[:expected_warmup])), "Expected NaN in sine warmup period"
        assert np.all(np.isnan(lead[:expected_warmup])), "Expected NaN in lead warmup period"


        if len(sine) > expected_warmup:
            non_nan_start = max(expected_warmup, 240)
            if len(sine) > non_nan_start:
                assert not np.any(np.isnan(sine[non_nan_start:])), "Found unexpected NaN in sine after warmup period"
                assert not np.any(np.isnan(lead[non_nan_start:])), "Found unexpected NaN in lead after warmup period"

    def test_msw_streaming(self, test_data):
        """Test MSW streaming API - mirrors check_msw_streaming"""
        close = test_data['close']
        period = 5


        batch_sine, batch_lead = ta_indicators.msw(close, period=period)


        stream = ta_indicators.MswStream(period=period)
        stream_sine = []
        stream_lead = []

        for price in close:
            result = stream.update(price)
            if result is not None:
                stream_sine.append(result[0])
                stream_lead.append(result[1])
            else:
                stream_sine.append(np.nan)
                stream_lead.append(np.nan)

        stream_sine = np.array(stream_sine)
        stream_lead = np.array(stream_lead)


        assert len(batch_sine) == len(stream_sine)
        assert len(batch_lead) == len(stream_lead)


        mask_sine = ~(np.isnan(batch_sine) | np.isnan(stream_sine))
        mask_lead = ~(np.isnan(batch_lead) | np.isnan(stream_lead))

        if np.any(mask_sine):
            assert_close(
                batch_sine[mask_sine],
                stream_sine[mask_sine],
                rtol=1e-9,
                msg="MSW sine streaming mismatch"
            )

        if np.any(mask_lead):
            assert_close(
                batch_lead[mask_lead],
                stream_lead[mask_lead],
                rtol=1e-9,
                msg="MSW lead streaming mismatch"
            )

    def test_msw_batch(self, test_data):
        """Test MSW batch operation"""
        close = test_data['close']


        result = ta_indicators.msw_batch(
            close,
            period_range=(5, 30, 5)
        )

        assert 'sine' in result
        assert 'lead' in result
        assert 'periods' in result


        expected_rows = 6
        assert result['sine'].shape[0] == expected_rows
        assert result['lead'].shape[0] == expected_rows
        assert result['sine'].shape[1] == len(close)
        assert result['lead'].shape[1] == len(close)
        assert len(result['periods']) == expected_rows


        expected_periods = [5, 10, 15, 20, 25, 30]
        assert list(result['periods']) == expected_periods

    def test_msw_with_kernel(self, test_data):
        """Test MSW with different kernel options"""
        close = test_data['close']
        period = 5


        for kernel in ['scalar', 'avx2', 'avx512', None]:
            try:
                sine, lead = ta_indicators.msw(close, period=period, kernel=kernel)
                assert isinstance(sine, np.ndarray)
                assert isinstance(lead, np.ndarray)
                assert len(sine) == len(close)
                assert len(lead) == len(close)
            except ValueError as e:

                if "Unsupported kernel" not in str(e) and "kernel not compiled" not in str(e):
                    raise

    def test_msw_all_nan_input(self):
        """Test MSW with all NaN values - mirrors ALMA test pattern"""
        all_nan = np.full(100, np.nan)

        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.msw(all_nan, period=5)

    def test_msw_mixed_nan_input(self):
        """Test MSW with mixed NaN values at the beginning"""
        mixed_data = np.array([np.nan, np.nan, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0])
        period = 3

        sine, lead = ta_indicators.msw(mixed_data, period=period)
        assert len(sine) == len(mixed_data)
        assert len(lead) == len(mixed_data)


        assert np.isnan(sine[0])
        assert np.isnan(sine[1])
        assert np.isnan(lead[0])
        assert np.isnan(lead[1])


        for i in range(4, len(sine)):
            assert not np.isnan(sine[i]), f"Unexpected NaN in sine at index {i}"
            assert not np.isnan(lead[i]), f"Unexpected NaN in lead at index {i}"

    def test_msw_simple_pattern(self):
        """Test MSW with a simple predictable pattern"""
        simple_data = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4, 5], dtype=np.float64)
        period = 5

        sine, lead = ta_indicators.msw(simple_data, period=period)
        assert len(sine) == len(simple_data)
        assert len(lead) == len(simple_data)


        for i in range(period - 1):
            assert np.isnan(sine[i]), f"Expected NaN in sine at index {i}"
            assert np.isnan(lead[i]), f"Expected NaN in lead at index {i}"


        for i in range(period - 1, len(sine)):
            assert not np.isnan(sine[i]), f"Unexpected NaN in sine at index {i}"
            assert not np.isnan(lead[i]), f"Unexpected NaN in lead at index {i}"


            assert -1.0 <= sine[i] <= 1.0, f"Sine value {sine[i]} at index {i} is out of range [-1, 1]"
            assert -1.0 <= lead[i] <= 1.0, f"Lead value {lead[i]} at index {i} is out of range [-1, 1]"

    def test_msw_batch_multiple_periods(self, test_data):
        """Test MSW batch with multiple period values"""
        close = test_data['close'][:100]


        result = ta_indicators.msw_batch(
            close,
            period_range=(5, 15, 5)
        )

        assert 'sine' in result
        assert 'lead' in result
        assert 'periods' in result


        expected_rows = 3
        assert result['sine'].shape[0] == expected_rows
        assert result['lead'].shape[0] == expected_rows
        assert result['sine'].shape[1] == len(close)
        assert result['lead'].shape[1] == len(close)
        assert len(result['periods']) == expected_rows


        expected_periods = [5, 10, 15]
        assert list(result['periods']) == expected_periods


        for i, period in enumerate(expected_periods):
            single_sine, single_lead = ta_indicators.msw(close, period=period)
            batch_sine = result['sine'][i]
            batch_lead = result['lead'][i]


            mask = ~(np.isnan(single_sine) | np.isnan(batch_sine))
            if np.any(mask):
                assert_close(
                    single_sine[mask],
                    batch_sine[mask],
                    rtol=1e-9,
                    msg=f"Sine mismatch for period {period}"
                )

            mask = ~(np.isnan(single_lead) | np.isnan(batch_lead))
            if np.any(mask):
                assert_close(
                    single_lead[mask],
                    batch_lead[mask],
                    rtol=1e-9,
                    msg=f"Lead mismatch for period {period}"
                )

    def test_msw_batch_edge_cases(self):
        """Test MSW batch processing edge cases"""
        small_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)


        single_batch = ta_indicators.msw_batch(
            small_data,
            period_range=(5, 5, 0)
        )

        assert single_batch['sine'].shape == (1, 10)
        assert single_batch['lead'].shape == (1, 10)
        assert list(single_batch['periods']) == [5]


        large_step = ta_indicators.msw_batch(
            small_data,
            period_range=(3, 5, 10)
        )


        assert large_step['sine'].shape == (1, 10)
        assert large_step['lead'].shape == (1, 10)
        assert list(large_step['periods']) == [3]


        with pytest.raises(ValueError, match="Empty data|All values are NaN"):
            ta_indicators.msw_batch(np.array([]), period_range=(5, 5, 0))


if __name__ == "__main__":

    pytest.main([__file__, "-v"])