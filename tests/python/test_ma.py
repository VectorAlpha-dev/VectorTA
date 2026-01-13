"""
Python binding tests for MA dispatcher.
These tests ensure the MA dispatcher correctly routes to various moving average implementations.
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


class TestMaDispatcher:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_ma_sma(self, test_data):
        """Test MA dispatcher with SMA"""
        close = test_data['close']


        result = ta_indicators.ma(close, "sma", 20)
        assert len(result) == len(close)


        direct_result = ta_indicators.sma(close, 20)
        assert_close(result, direct_result, rtol=1e-10, msg="MA dispatcher SMA mismatch")

    def test_ma_ema(self, test_data):
        """Test MA dispatcher with EMA"""
        close = test_data['close']


        result = ta_indicators.ma(close, "ema", 20)
        assert len(result) == len(close)


        direct_result = ta_indicators.ema(close, 20)
        assert_close(result, direct_result, rtol=1e-10, msg="MA dispatcher EMA mismatch")

    def test_ma_wma(self, test_data):
        """Test MA dispatcher with WMA"""
        close = test_data['close']


        result = ta_indicators.ma(close, "wma", 20)
        assert len(result) == len(close)


        direct_result = ta_indicators.wma(close, 20)
        assert_close(result, direct_result, rtol=1e-10, msg="MA dispatcher WMA mismatch")

    def test_ma_alma(self, test_data):
        """Test MA dispatcher with ALMA (uses default offset and sigma)"""
        close = test_data['close']


        result = ta_indicators.ma(close, "alma", 9)
        assert len(result) == len(close)


        direct_result = ta_indicators.alma(close, 9, 0.85, 6.0)
        assert_close(result, direct_result, rtol=1e-10, msg="MA dispatcher ALMA mismatch")

    def test_ma_dema(self, test_data):
        """Test MA dispatcher with DEMA"""
        close = test_data['close']


        result = ta_indicators.ma(close, "dema", 20)
        assert len(result) == len(close)


        direct_result = ta_indicators.dema(close, 20)
        assert_close(result, direct_result, rtol=1e-10, msg="MA dispatcher DEMA mismatch")

    def test_ma_tema(self, test_data):
        """Test MA dispatcher with TEMA"""
        close = test_data['close']


        result = ta_indicators.ma(close, "tema", 20)
        assert len(result) == len(close)


        direct_result = ta_indicators.tema(close, 20)
        assert_close(result, direct_result, rtol=1e-10, msg="MA dispatcher TEMA mismatch")

    def test_ma_hma(self, test_data):
        """Test MA dispatcher with HMA"""
        close = test_data['close']


        result = ta_indicators.ma(close, "hma", 20)
        assert len(result) == len(close)


        direct_result = ta_indicators.hma(close, 20)
        assert_close(result, direct_result, rtol=1e-10, msg="MA dispatcher HMA mismatch")

    def test_ma_kama(self, test_data):
        """Test MA dispatcher with KAMA"""
        close = test_data['close']


        result = ta_indicators.ma(close, "kama", 10)
        assert len(result) == len(close)


        direct_result = ta_indicators.kama(close, 10)
        assert_close(result, direct_result, rtol=1e-10, msg="MA dispatcher KAMA mismatch")

    def test_ma_zlema(self, test_data):
        """Test MA dispatcher with ZLEMA"""
        close = test_data['close']


        result = ta_indicators.ma(close, "zlema", 20)
        assert len(result) == len(close)


        direct_result = ta_indicators.zlema(close, 20)
        assert_close(result, direct_result, rtol=1e-10, msg="MA dispatcher ZLEMA mismatch")

    def test_ma_wilders(self, test_data):
        """Test MA dispatcher with Wilders"""
        close = test_data['close']


        result = ta_indicators.ma(close, "wilders", 14)
        assert len(result) == len(close)


        direct_result = ta_indicators.wilders(close, 14)
        assert_close(result, direct_result, rtol=1e-10, msg="MA dispatcher Wilders mismatch")

    def test_ma_invalid_type(self, test_data):
        """Test MA dispatcher with invalid MA type defaults to SMA"""
        close = test_data['close']


        result = ta_indicators.ma(close, "invalid_ma_type", 20)
        assert len(result) == len(close)


        sma_result = ta_indicators.sma(close, 20)
        assert_close(result, sma_result, rtol=1e-10, msg="MA dispatcher should default to SMA")

    def test_ma_case_insensitive(self, test_data):
        """Test MA dispatcher is case insensitive"""
        close = test_data['close']


        result_upper = ta_indicators.ma(close, "SMA", 20)
        result_lower = ta_indicators.ma(close, "sma", 20)
        assert_close(result_upper, result_lower, rtol=1e-10, msg="MA dispatcher should be case insensitive")


        result_mixed = ta_indicators.ma(close, "EmA", 20)
        ema_result = ta_indicators.ema(close, 20)
        assert_close(result_mixed, ema_result, rtol=1e-10, msg="MA dispatcher should handle mixed case")

    def test_ma_empty_input(self):
        """Test MA dispatcher fails with empty input"""
        empty = np.array([])

        with pytest.raises(ValueError):
            ta_indicators.ma(empty, "sma", 20)

    def test_ma_zero_period(self):
        """Test MA dispatcher fails with zero period"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        with pytest.raises(ValueError):
            ta_indicators.ma(data, "sma", 0)

    def test_ma_period_exceeds_length(self):
        """Test MA dispatcher fails when period exceeds data length"""
        data = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError):
            ta_indicators.ma(data, "sma", 10)

    def test_ma_all_nan_input(self):
        """Test MA dispatcher with all NaN values"""
        all_nan = np.full(100, np.nan)

        with pytest.raises(ValueError):
            ta_indicators.ma(all_nan, "sma", 20)

    def test_ma_all_supported_types(self, test_data):
        """Test MA dispatcher with all supported MA types"""
        close = test_data['close'][:1000]


        ma_types = [
            "sma", "ema", "dema", "tema", "smma", "zlema", "alma", "cwma",
            "edcf", "fwma", "gaussian", "highpass", "highpass2", "hma",
            "jma", "jsa", "kama", "linreg", "nma", "pwma", "reflex",
            "sinwma", "sqwma", "srwma", "supersmoother", "supersmoother_3_pole",
            "swma", "tilson", "trendflex", "trima", "wilders", "wma"
        ]

        for ma_type in ma_types:
            try:
                result = ta_indicators.ma(close, ma_type, 20)
                assert len(result) == len(close), f"MA type {ma_type} returned wrong length"

                assert not np.all(np.isnan(result[100:])), f"MA type {ma_type} returned all NaN"
            except Exception as e:
                pytest.fail(f"MA type {ma_type} failed: {str(e)}")

    def test_ma_special_types_with_defaults(self, test_data):
        """Test MA types that have special default handling"""
        close = test_data['close'][:1000]


        result = ta_indicators.ma(close, "hwma", 20)
        assert len(result) == len(close)


        result = ta_indicators.ma(close, "maaq", 20)
        assert len(result) == len(close)


        result = ta_indicators.ma(close, "mama", 20)
        assert len(result) == len(close)


        result = ta_indicators.ma(close, "mwdx", 20)
        assert len(result) == len(close)


        result = ta_indicators.ma(close, "ehlers_itrend", 50)
        assert len(result) == len(close)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])