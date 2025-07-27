"""
Python binding tests for ERI indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

try:
    import my_project as ta_indicators
except ImportError:
    # If not in virtual environment, try to import from installed location
    try:
        import my_project as ta_indicators
    except ImportError:
        pytest.skip("Python module not built. Run 'maturin develop --features python' first", allow_module_level=True)

from test_utils import load_test_data, assert_close, EXPECTED_OUTPUTS
from rust_comparison import compare_with_rust


class TestEri:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_eri_partial_params(self, test_data):
        """Test ERI with partial parameters - mirrors check_eri_partial_params"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        # Test with all default params
        bull, bear = ta_indicators.eri(high, low, close, 13, "ema")
        assert len(bull) == len(close)
        assert len(bear) == len(close)
        
        # Test with different source
        hl2 = (high + low) / 2
        bull2, bear2 = ta_indicators.eri(high, low, hl2, 14, "ema")
        assert len(bull2) == len(hl2)
        assert len(bear2) == len(hl2)
        
        # Test with SMA
        hlc3 = (high + low + close) / 3
        bull3, bear3 = ta_indicators.eri(high, low, hlc3, 20, "sma")
        assert len(bull3) == len(hlc3)
        assert len(bear3) == len(hlc3)
    
    def test_eri_accuracy(self, test_data):
        """Test ERI matches expected values from Rust tests - mirrors check_eri_accuracy"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        bull, bear = ta_indicators.eri(high, low, close, period=13, ma_type="ema")
        
        assert len(bull) == len(close)
        assert len(bear) == len(close)
        
        # Check last 5 values match expected
        expected_bull_last_five = [
            -103.35343557205488,
            6.839912366813223,
            -42.851503685589705,
            -9.444146016219747,
            11.476446271808527,
        ]
        expected_bear_last_five = [
            -433.3534355720549,
            -314.1600876331868,
            -414.8515036855897,
            -336.44414601621975,
            -925.5235537281915,
        ]
        
        assert_close(
            bull[-5:], 
            expected_bull_last_five,
            rtol=1e-2,
            msg="ERI bull last 5 values mismatch"
        )
        
        assert_close(
            bear[-5:], 
            expected_bear_last_five,
            rtol=1e-2,
            msg="ERI bear last 5 values mismatch"
        )
        
        # Compare full output with Rust
        compare_with_rust('eri', (bull, bear), 'close', {'period': 13, 'ma_type': 'ema'})
    
    def test_eri_default_params(self, test_data):
        """Test ERI with default parameters - mirrors check_eri_default_candles"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        # Default params: period=13, ma_type="ema"
        bull, bear = ta_indicators.eri(high, low, close, 13, "ema")
        assert len(bull) == len(close)
        assert len(bear) == len(close)
    
    def test_eri_zero_period(self):
        """Test ERI fails with zero period - mirrors check_eri_zero_period"""
        high = np.array([10.0, 20.0, 30.0])
        low = np.array([8.0, 18.0, 28.0])
        src = np.array([9.0, 19.0, 29.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.eri(high, low, src, period=0, ma_type="ema")
    
    def test_eri_period_exceeds_length(self):
        """Test ERI fails when period exceeds data length - mirrors check_eri_period_exceeds_length"""
        high = np.array([10.0, 20.0, 30.0])
        low = np.array([8.0, 18.0, 28.0])
        src = np.array([9.0, 19.0, 29.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.eri(high, low, src, period=10, ma_type="ema")
    
    def test_eri_very_small_dataset(self):
        """Test ERI fails with insufficient data - mirrors check_eri_very_small_dataset"""
        high = np.array([42.0])
        low = np.array([40.0])
        src = np.array([41.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.eri(high, low, src, period=9, ma_type="ema")
    
    def test_eri_empty_input(self):
        """Test ERI fails with empty input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Empty data"):
            ta_indicators.eri(empty, empty, empty, period=13, ma_type="ema")
    
    def test_eri_mismatched_lengths(self):
        """Test ERI fails with mismatched array lengths"""
        high = np.array([10.0, 20.0, 30.0])
        low = np.array([8.0, 18.0])
        src = np.array([9.0, 19.0, 29.0])
        
        with pytest.raises(ValueError, match="must have the same length"):
            ta_indicators.eri(high, low, src, period=2, ma_type="ema")
    
    def test_eri_reinput(self, test_data):
        """Test ERI applied twice (re-input) - mirrors check_eri_reinput"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        # First pass
        bull1, bear1 = ta_indicators.eri(high, low, close, period=14, ma_type="ema")
        assert len(bull1) == len(close)
        assert len(bear1) == len(close)
        
        # Second pass - apply ERI to ERI output
        bull2, bear2 = ta_indicators.eri(bull1, bear1, bull1, period=14, ma_type="ema")
        assert len(bull2) == len(bull1)
        assert len(bear2) == len(bear1)
        
        # Check that after index 28, no NaN values exist
        for i in range(28, len(bull2)):
            assert not np.isnan(bull2[i]), f"Expected no NaN in bull at index {i}"
            assert not np.isnan(bear2[i]), f"Expected no NaN in bear at index {i}"
    
    def test_eri_nan_handling(self, test_data):
        """Test ERI handles NaN values correctly - mirrors check_eri_nan_handling"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        bull, bear = ta_indicators.eri(high, low, close, period=13, ma_type="ema")
        assert len(bull) == len(close)
        assert len(bear) == len(close)
        
        # After warmup period (240), no NaN values should exist
        if len(bull) > 240:
            assert not np.any(np.isnan(bull[240:])), "Found unexpected NaN in bull after warmup period"
            assert not np.any(np.isnan(bear[240:])), "Found unexpected NaN in bear after warmup period"
    
    def test_eri_streaming(self, test_data):
        """Test ERI streaming matches batch calculation"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        period = 13
        ma_type = "ema"
        
        # Batch calculation
        batch_bull, batch_bear = ta_indicators.eri(high, low, close, period=period, ma_type=ma_type)
        
        # Streaming calculation
        stream = ta_indicators.EriStream(period=period, ma_type=ma_type)
        stream_bull = []
        stream_bear = []
        
        for h, l, c in zip(high, low, close):
            result = stream.update(h, l, c)
            if result is not None:
                bull_val, bear_val = result
                stream_bull.append(bull_val)
                stream_bear.append(bear_val)
            else:
                stream_bull.append(np.nan)
                stream_bear.append(np.nan)
        
        stream_bull = np.array(stream_bull)
        stream_bear = np.array(stream_bear)
        
        # Compare batch vs streaming
        assert len(batch_bull) == len(stream_bull)
        assert len(batch_bear) == len(stream_bear)
        
        # Compare values where both are not NaN
        for i, (bb, sb, be, se) in enumerate(zip(batch_bull, stream_bull, batch_bear, stream_bear)):
            if np.isnan(bb) and np.isnan(sb) and np.isnan(be) and np.isnan(se):
                continue
            assert_close(bb, sb, rtol=1e-9, atol=1e-9, 
                        msg=f"ERI bull streaming mismatch at index {i}")
            assert_close(be, se, rtol=1e-9, atol=1e-9, 
                        msg=f"ERI bear streaming mismatch at index {i}")
    
    def test_eri_batch(self, test_data):
        """Test ERI batch processing - mirrors check_batch_default_row"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        result = ta_indicators.eri_batch(
            high,
            low,
            close,
            period_range=(13, 13, 0),  # Default period only
            ma_type="ema"
        )
        
        assert 'bull_values' in result
        assert 'bear_values' in result
        assert 'periods' in result
        assert 'ma_types' in result
        
        # Should have 1 combination (default params)
        assert result['bull_values'].shape[0] == 1
        assert result['bull_values'].shape[1] == len(close)
        assert result['bear_values'].shape[0] == 1
        assert result['bear_values'].shape[1] == len(close)
        
        # Extract the single row
        default_bull = result['bull_values'][0]
        default_bear = result['bear_values'][0]
        
        expected_bull = [
            -103.35343557205488,
            6.839912366813223,
            -42.851503685589705,
            -9.444146016219747,
            11.476446271808527,
        ]
        expected_bear = [
            -433.3534355720549,
            -314.1600876331868,
            -414.8515036855897,
            -336.44414601621975,
            -925.5235537281915,
        ]
        
        # Check last 5 values match
        assert_close(
            default_bull[-5:],
            expected_bull,
            rtol=1e-2,
            msg="ERI batch bull default row mismatch"
        )
        
        assert_close(
            default_bear[-5:],
            expected_bear,
            rtol=1e-2,
            msg="ERI batch bear default row mismatch"
        )
    
    def test_eri_batch_multiple_periods(self, test_data):
        """Test ERI batch with multiple periods"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        
        result = ta_indicators.eri_batch(
            high,
            low,
            close,
            period_range=(10, 20, 5),  # 10, 15, 20
            ma_type="ema"
        )
        
        # Should have 3 combinations
        assert result['bull_values'].shape[0] == 3
        assert result['bear_values'].shape[0] == 3
        assert len(result['periods']) == 3
        assert list(result['periods']) == [10, 15, 20]
    
    def test_eri_all_nan_input(self):
        """Test ERI with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.eri(all_nan, all_nan, all_nan, period=13, ma_type="ema")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])