"""
Python binding tests for RSMK indicator.
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


class TestRsmk:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_rsmk_basic_functionality(self, test_data):
        """Test RSMK with basic functionality"""
        close = test_data['close']
        
        # Use default parameters
        indicator, signal = ta_indicators.rsmk(
            close, close,  # main and compare
            lookback=90,
            period=3,
            signal_period=20
        )
        
        assert len(indicator) == len(close)
        assert len(signal) == len(close)
        
        # Check that we have non-NaN values after warmup
        assert not np.all(np.isnan(indicator))
        assert not np.all(np.isnan(signal))
    
    def test_rsmk_accuracy(self, test_data):
        """Test RSMK matches expected values from Rust tests"""
        close = test_data['close']
        
        if 'rsmk' not in EXPECTED_OUTPUTS:
            pytest.skip("RSMK expected outputs not yet defined in test_utils.py")
        
        expected = EXPECTED_OUTPUTS['rsmk']
        
        indicator, signal = ta_indicators.rsmk(
            close, close,
            lookback=expected['default_params']['lookback'],
            period=expected['default_params']['period'],
            signal_period=expected['default_params']['signal_period']
        )
        
        assert len(indicator) == len(close)
        assert len(signal) == len(close)
        
        # Check last 5 values match expected
        assert_close(
            indicator[-5:],
            expected['indicator_last_5'],
            rtol=1e-7,
            msg="RSMK indicator last 5 values mismatch"
        )
        
        assert_close(
            signal[-5:],
            expected['signal_last_5'],
            rtol=1e-7,
            msg="RSMK signal last 5 values mismatch"
        )
    
    def test_rsmk_partial_params(self, test_data):
        """Test RSMK with partial parameters (None values)"""
        close = test_data['close']
        
        # Test with optional MA types as None (should use defaults)
        indicator, signal = ta_indicators.rsmk(
            close, close,
            lookback=90,
            period=3,
            signal_period=20,
            matype=None,
            signal_matype=None
        )
        
        assert len(indicator) == len(close)
        assert len(signal) == len(close)
    
    def test_rsmk_custom_ma_types(self, test_data):
        """Test RSMK with different MA type combinations"""
        close = test_data['close'][:500]  # Use smaller dataset for performance
        
        ma_types = ["ema", "sma", "wma", "dema", "tema"]
        
        for matype in ma_types[:3]:  # Test a few combinations
            for signal_matype in ma_types[:3]:
                indicator, signal = ta_indicators.rsmk(
                    close, close,
                    lookback=20,
                    period=3,
                    signal_period=9,
                    matype=matype,
                    signal_matype=signal_matype
                )
                
                assert len(indicator) == len(close), f"Failed with matype={matype}, signal_matype={signal_matype}"
                assert len(signal) == len(close), f"Failed with matype={matype}, signal_matype={signal_matype}"
    
    def test_rsmk_zero_period(self):
        """Test RSMK fails with zero period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.rsmk(
                input_data, input_data,
                lookback=0,
                period=0,
                signal_period=0
            )
    
    def test_rsmk_period_exceeds_length(self):
        """Test RSMK fails when period exceeds data length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.rsmk(
                data_small, data_small,
                lookback=90,
                period=3,
                signal_period=20
            )
    
    def test_rsmk_insufficient_data(self):
        """Test RSMK fails with insufficient data"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data"):
            ta_indicators.rsmk(
                single_point, single_point,
                lookback=90,
                period=3,
                signal_period=20
            )
    
    def test_rsmk_empty_input(self):
        """Test RSMK fails with empty input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Input data slice is empty|Empty data"):
            ta_indicators.rsmk(
                empty, empty,
                lookback=90,
                period=3,
                signal_period=20
            )
    
    def test_rsmk_all_nan(self):
        """Test RSMK fails with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.rsmk(
                all_nan, all_nan,
                lookback=2,
                period=1,
                signal_period=1
            )
    
    def test_rsmk_mismatched_lengths(self):
        """Test RSMK fails with mismatched main/compare lengths"""
        main = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        compare = np.array([1.0, 2.0, 3.0])
        
        with pytest.raises(ValueError, match="Invalid period|Mismatched lengths"):
            ta_indicators.rsmk(
                main, compare,
                lookback=2,
                period=1,
                signal_period=1
            )
    
    def test_rsmk_invalid_ma_type(self):
        """Test RSMK fails with invalid MA type"""
        input_data = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
        
        with pytest.raises(ValueError, match="Error from MA function"):
            ta_indicators.rsmk(
                input_data, input_data,
                lookback=2,
                period=3,
                signal_period=3,
                matype="nonexistent_ma",
                signal_matype="ema"
            )
    
    def test_rsmk_nan_handling(self, test_data):
        """Test RSMK handles NaN values correctly"""
        close = test_data['close']
        
        indicator, signal = ta_indicators.rsmk(
            close, close,
            lookback=90,
            period=3,
            signal_period=20
        )
        
        assert len(indicator) == len(close)
        assert len(signal) == len(close)
        
        # Check warmup period handling
        # Warmup calculation: indicator_warmup = first_valid + lookback + period - 1
        # signal_warmup = indicator_warmup + signal_period - 1
        # With lookback=90, period=3: indicator_warmup = 0 + 90 + 3 - 1 = 92
        # signal_warmup = 92 + 20 - 1 = 111
        
        # After warmup, no NaN values should exist
        if len(indicator) > 111:
            # Check a reasonable range after warmup
            assert not np.any(np.isnan(signal[150:])), "Found unexpected NaN in signal after warmup period"
    
    def test_rsmk_streaming(self, test_data):
        """Test RSMK streaming matches batch calculation"""
        close = test_data['close'][:500]  # Use smaller dataset for performance
        lookback = 20
        period = 3
        signal_period = 9
        
        # Batch calculation
        batch_indicator, batch_signal = ta_indicators.rsmk(
            close, close,
            lookback=lookback,
            period=period,
            signal_period=signal_period
        )
        
        # Streaming calculation
        stream = ta_indicators.RsmkStream(
            lookback=lookback,
            period=period,
            signal_period=signal_period,
            matype=None,  # Use default
            signal_matype=None  # Use default
        )
        
        stream_indicators = []
        stream_signals = []
        
        for price in close:
            result = stream.update(price, price)  # main and compare
            if result is not None:
                stream_indicators.append(result[0] if result[0] is not None else np.nan)
                stream_signals.append(result[1] if result[1] is not None else np.nan)
            else:
                stream_indicators.append(np.nan)
                stream_signals.append(np.nan)
        
        stream_indicators = np.array(stream_indicators)
        stream_signals = np.array(stream_signals)
        
        # Compare batch vs streaming
        assert len(batch_indicator) == len(stream_indicators)
        assert len(batch_signal) == len(stream_signals)
        
        # Compare values where both are not NaN
        for i, (b_ind, s_ind) in enumerate(zip(batch_indicator, stream_indicators)):
            if not np.isnan(b_ind) and not np.isnan(s_ind):
                assert_close(b_ind, s_ind, rtol=1e-9, atol=1e-9,
                           msg=f"RSMK indicator streaming mismatch at index {i}")
        
        for i, (b_sig, s_sig) in enumerate(zip(batch_signal, stream_signals)):
            if not np.isnan(b_sig) and not np.isnan(s_sig):
                assert_close(b_sig, s_sig, rtol=1e-9, atol=1e-9,
                           msg=f"RSMK signal streaming mismatch at index {i}")
    
    def test_rsmk_batch_single_params(self, test_data):
        """Test RSMK batch processing with single parameter set"""
        close = test_data['close'][:500]  # Use smaller dataset
        
        result = ta_indicators.rsmk_batch(
            close, close,
            lookback_range=(90, 90, 0),  # Single lookback
            period_range=(3, 3, 0),      # Single period
            signal_period_range=(20, 20, 0)  # Single signal period
        )
        
        assert 'indicator' in result  # Note: singular, not plural
        assert 'signal' in result     # Note: singular, not plural
        assert 'lookbacks' in result
        assert 'periods' in result
        assert 'signal_periods' in result
        assert 'matypes' in result
        assert 'signal_matypes' in result
        
        # Should have 1 combination
        assert result['indicator'].shape[0] == 1
        assert result['signal'].shape[0] == 1
        assert result['indicator'].shape[1] == len(close)
        assert result['signal'].shape[1] == len(close)
        
        # Compare with single calculation
        single_indicator, single_signal = ta_indicators.rsmk(
            close, close,
            lookback=90,
            period=3,
            signal_period=20
        )
        
        # Extract the single row from batch
        batch_indicator = result['indicator'][0]
        batch_signal = result['signal'][0]
        
        # Check they match (ignore warmup NaNs by comparing only finite indices)
        ind_mask = ~(np.isnan(batch_indicator) | np.isnan(single_indicator))
        sig_mask = ~(np.isnan(batch_signal) | np.isnan(single_signal))

        assert_close(batch_indicator[ind_mask], single_indicator[ind_mask], rtol=1e-9,
                     msg="RSMK batch indicator mismatch with single calculation")
        assert_close(batch_signal[sig_mask], single_signal[sig_mask], rtol=1e-9,
                     msg="RSMK batch signal mismatch with single calculation")
    
    def test_rsmk_batch_multiple_params(self, test_data):
        """Test RSMK batch processing with parameter sweeps"""
        close = test_data['close'][:200]  # Use smaller dataset for performance
        
        result = ta_indicators.rsmk_batch(
            close, close,
            lookback_range=(10, 20, 10),     # 2 values: 10, 20
            period_range=(2, 4, 2),           # 2 values: 2, 4
            signal_period_range=(5, 10, 5),   # 2 values: 5, 10
            matype="ema",
            signal_matype="sma"
        )
        
        # Should have 2 * 2 * 2 = 8 combinations
        expected_combos = 8
        assert result['indicator'].shape[0] == expected_combos
        assert result['signal'].shape[0] == expected_combos
        assert result['indicator'].shape[1] == len(close)
        assert result['signal'].shape[1] == len(close)
        
        # Check metadata arrays
        assert len(result['lookbacks']) == expected_combos
        assert len(result['periods']) == expected_combos
        assert len(result['signal_periods']) == expected_combos
        assert len(result['matypes']) == expected_combos
        assert len(result['signal_matypes']) == expected_combos
        
        # Check MA types are set (the batch may use per-combo types)
        assert all(isinstance(ma, str) for ma in result['matypes'])
        assert all(isinstance(ma, str) for ma in result['signal_matypes'])
    
    def test_rsmk_compare_zero_handling(self):
        """Test RSMK handles zeros in compare data properly"""
        main = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
        compare = np.array([10.0, 0.0, 12.0, 0.0, 14.0, 15.0])  # Has zeros
        
        # RSMK should handle zeros by treating ln(main/0) as NaN
        # It may not error but will produce NaN values
        indicator, signal = ta_indicators.rsmk(
            main, compare,
            lookback=2,
            period=2,
            signal_period=2
        )
        
        # Should have NaN values where compare is zero
        assert np.isnan(indicator[1]) or np.isnan(indicator[3]), \
            "Expected NaN values where compare is zero"
    
    def test_rsmk_different_data_sources(self, test_data):
        """Test RSMK with different main and compare data"""
        close = test_data['close'][:200]
        high = test_data['high'][:200]
        
        # Use close as main, high as compare
        indicator, signal = ta_indicators.rsmk(
            close, high,
            lookback=20,
            period=3,
            signal_period=9
        )
        
        assert len(indicator) == len(close)
        assert len(signal) == len(close)
        
        # Results should be different from using close for both
        indicator_same, signal_same = ta_indicators.rsmk(
            close, close,
            lookback=20,
            period=3,
            signal_period=9
        )
        
        # After warmup, values should be different
        valid_idx = 50  # Well past warmup
        if not np.isnan(indicator[valid_idx]) and not np.isnan(indicator_same[valid_idx]):
            assert abs(indicator[valid_idx] - indicator_same[valid_idx]) > 1e-10, \
                "Expected different results when using different compare data"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
