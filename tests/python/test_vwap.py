"""
Python binding tests for VWAP indicator.
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


class TestVwap:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_vwap_partial_params(self, test_data):
        """Test VWAP with default parameters - mirrors check_vwap_partial_params"""
        timestamps = test_data['timestamp']
        volumes = test_data['volume']
        prices = (test_data['high'] + test_data['low'] + test_data['close']) / 3.0


        result = ta_indicators.vwap(timestamps, volumes, prices)
        assert len(result) == len(prices)

    def test_vwap_accuracy(self, test_data):
        """Test VWAP matches expected values from Rust tests - mirrors check_vwap_accuracy"""
        timestamps = test_data['timestamp']
        volumes = test_data['volume']
        prices = (test_data['high'] + test_data['low'] + test_data['close']) / 3.0


        expected_last_five = [
            59353.05963230107,
            59330.15815713043,
            59289.94649532547,
            59274.6155462414,
            58730.0,
        ]

        result = ta_indicators.vwap(
            timestamps,
            volumes,
            prices,
            anchor="1D"
        )

        assert len(result) == len(prices)


        assert_close(
            result[-5:],
            expected_last_five,
            rtol=1e-5,
            msg="VWAP last 5 values mismatch"
        )

    def test_vwap_anchor_parsing_error(self, test_data):
        """Test VWAP fails with invalid anchor - mirrors check_vwap_anchor_parsing_error"""
        timestamps = test_data['timestamp']
        volumes = test_data['volume']
        prices = (test_data['high'] + test_data['low'] + test_data['close']) / 3.0

        with pytest.raises(ValueError, match="Error parsing anchor"):
            ta_indicators.vwap(timestamps, volumes, prices, anchor="xyz")

    def test_vwap_mismatch_lengths(self):
        """Test VWAP fails when array lengths don't match"""
        timestamps = np.array([1000, 2000, 3000], dtype=np.int64)
        volumes = np.array([100.0, 200.0])
        prices = np.array([10.0, 20.0, 30.0])

        with pytest.raises(ValueError, match="Mismatch in length"):
            ta_indicators.vwap(timestamps, volumes, prices)

    def test_vwap_empty_data(self):
        """Test VWAP fails with empty input"""
        empty_ts = np.array([], dtype=np.int64)
        empty_vol = np.array([])
        empty_price = np.array([])

        with pytest.raises(ValueError, match="No data for VWAP calculation"):
            ta_indicators.vwap(empty_ts, empty_vol, empty_price)

    def test_vwap_streaming(self, test_data):
        """Test VWAP streaming functionality"""
        timestamps = test_data['timestamp']
        volumes = test_data['volume']
        prices = (test_data['high'] + test_data['low'] + test_data['close']) / 3.0


        batch_result = ta_indicators.vwap(timestamps, volumes, prices, anchor="1d")


        stream = ta_indicators.VwapStream(anchor="1d")
        stream_values = []

        for i in range(len(timestamps)):
            result = stream.update(timestamps[i], prices[i], volumes[i])
            stream_values.append(result if result is not None else np.nan)

        stream_values = np.array(stream_values)


        assert len(batch_result) == len(stream_values)


        valid_mask = ~np.isnan(batch_result) & ~np.isnan(stream_values)
        assert_close(
            batch_result[valid_mask],
            stream_values[valid_mask],
            rtol=1e-9,
            atol=1e-9,
            msg="VWAP streaming mismatch"
        )

    def test_vwap_batch(self, test_data):
        """Test VWAP batch processing - mirrors check_batch_anchor_grid"""
        timestamps = test_data['timestamp']
        volumes = test_data['volume']
        prices = (test_data['high'] + test_data['low'] + test_data['close']) / 3.0

        result = ta_indicators.vwap_batch(
            timestamps,
            volumes,
            prices,
            anchor_range=("1d", "3d", 1)
        )

        assert 'values' in result
        assert 'anchors' in result


        assert result['values'].shape[0] == 3
        assert result['values'].shape[1] == len(prices)
        assert list(result['anchors']) == ["1d", "2d", "3d"]


        single_vwap = ta_indicators.vwap(timestamps, volumes, prices, anchor="1d")
        assert_close(
            result['values'][0],
            single_vwap,
            rtol=1e-9,
            msg="VWAP batch 1d row mismatch"
        )

    def test_vwap_default_params(self, test_data):
        """Test VWAP with default parameters - mirrors check_vwap_with_default_params"""

        timestamps = test_data['timestamp']
        volumes = test_data['volume']
        prices = test_data['close']


        result = ta_indicators.vwap(timestamps, volumes, prices)
        assert len(result) == len(prices)

    def test_vwap_nan_handling(self, test_data):
        """Test VWAP handles finite values correctly - mirrors check_vwap_nan_handling"""
        timestamps = test_data['timestamp']
        volumes = test_data['volume']
        prices = (test_data['high'] + test_data['low'] + test_data['close']) / 3.0

        result = ta_indicators.vwap(timestamps, volumes, prices)
        assert len(result) == len(prices)


        for val in result:
            if not np.isnan(val):
                assert np.isfinite(val), "Found non-finite value in VWAP output"

    def test_vwap_all_nan_input(self):
        """Test VWAP with all NaN values - should handle gracefully"""

        timestamps = np.arange(100, dtype=np.int64) * 1000
        volumes = np.full(100, 100.0)
        all_nan = np.full(100, np.nan)


        result = ta_indicators.vwap(timestamps, volumes, all_nan)
        assert len(result) == len(all_nan)
        assert np.all(np.isnan(result)), "Expected all NaN output for all NaN prices"

    def test_vwap_zero_volume(self, test_data):
        """Test VWAP behavior with zero volume periods"""
        timestamps = test_data['timestamp'][:100]
        prices = test_data['close'][:100]
        volumes = test_data['volume'][:100].copy()


        volumes[10:20] = 0.0

        result = ta_indicators.vwap(timestamps, volumes, prices)
        assert len(result) == len(prices)



        non_nan_count = np.sum(~np.isnan(result))
        assert non_nan_count > 0, "Expected some valid VWAP values"

    def test_vwap_invalid_timestamps(self):
        """Test VWAP with invalid timestamps"""
        volumes = np.array([100.0, 200.0, 300.0])
        prices = np.array([10.0, 20.0, 30.0])


        negative_ts = np.array([-1000, -500, 0], dtype=np.int64)
        result = ta_indicators.vwap(negative_ts, volumes, prices)
        assert len(result) == len(prices)

    def test_vwap_warmup_period(self, test_data):
        """Test VWAP warmup period behavior for different anchors"""
        timestamps = test_data['timestamp'][:100]
        volumes = test_data['volume'][:100]
        prices = test_data['close'][:100]


        result_1m = ta_indicators.vwap(timestamps, volumes, prices, anchor="1m")
        assert len(result_1m) == len(prices)


        result_1d = ta_indicators.vwap(timestamps, volumes, prices, anchor="1d")
        assert len(result_1d) == len(prices)


        assert np.any(~np.isnan(result_1m)), "Expected some valid values for 1m anchor"
        assert np.any(~np.isnan(result_1d)), "Expected some valid values for 1d anchor"

    def test_vwap_volume_weighting(self):
        """Test VWAP correctly weights by volume"""


        base_ts = 1609459200000
        timestamps = np.array([base_ts, base_ts + 3600000, base_ts + 7200000], dtype=np.int64)
        prices = np.array([100.0, 200.0, 300.0])
        volumes = np.array([1.0, 2.0, 3.0])

        result = ta_indicators.vwap(timestamps, volumes, prices, anchor="1d")





        expected = [100.0, 500.0/3.0, 1400.0/6.0]

        assert_close(result, expected, rtol=1e-9, msg="VWAP volume weighting incorrect")

    def test_vwap_batch_multi_anchor(self, test_data):
        """Test VWAP batch with multiple anchor combinations"""
        timestamps = test_data['timestamp'][:200]
        volumes = test_data['volume'][:200]
        prices = test_data['close'][:200]


        result = ta_indicators.vwap_batch(
            timestamps,
            volumes,
            prices,
            anchor_range=("1h", "4h", 1)
        )

        assert 'values' in result
        assert 'anchors' in result


        assert result['values'].shape[0] == 4
        assert result['values'].shape[1] == len(prices)
        assert list(result['anchors']) == ["1h", "2h", "3h", "4h"]


        single_1h = ta_indicators.vwap(timestamps, volumes, prices, anchor="1h")
        assert_close(
            result['values'][0],
            single_1h,
            rtol=1e-9,
            msg="Batch 1h row doesn't match single calculation"
        )


        single_4h = ta_indicators.vwap(timestamps, volumes, prices, anchor="4h")
        assert_close(
            result['values'][3],
            single_4h,
            rtol=1e-9,
            msg="Batch 4h row doesn't match single calculation"
        )

    def test_vwap_batch_static_anchor(self, test_data):
        """Test VWAP batch with static anchor (single value)"""
        timestamps = test_data['timestamp'][:100]
        volumes = test_data['volume'][:100]
        prices = test_data['close'][:100]


        result = ta_indicators.vwap_batch(
            timestamps,
            volumes,
            prices,
            anchor_range=("1d", "1d", 0)
        )

        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(prices)
        assert result['anchors'] == ["1d"]


        single = ta_indicators.vwap(timestamps, volumes, prices, anchor="1d")
        assert_close(
            result['values'][0],
            single,
            rtol=1e-9,
            msg="Static batch doesn't match single calculation"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
