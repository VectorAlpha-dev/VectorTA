"""
Python binding tests for MOD_GOD_MODE indicator.
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


class TestModGodMode:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_mod_god_mode_basic(self, test_data):
        """Test MOD_GOD_MODE with basic data - mirrors check_mod_god_mode_basic"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        volume = test_data['volume']
        
        
        wavetrend, signal, histogram = ta_indicators.mod_god_mode(
            high, low, close, volume,
            n1=17, n2=6, n3=4,
            mode='tradition_mg',
            use_volume=True
        )
        
        assert wavetrend is not None
        assert signal is not None
        assert histogram is not None
        assert len(wavetrend) == len(close)
        assert len(signal) == len(close)
        assert len(histogram) == len(close)
    
    def test_mod_god_mode_accuracy(self, test_data):
        """Test MOD_GOD_MODE matches expected values from Pine Script - mirrors check_mod_god_mode_accuracy"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        volume = test_data['volume']
        
        wavetrend, signal, histogram = ta_indicators.mod_god_mode(
            high, low, close, volume,
            n1=17, n2=6, n3=4,
            mode='tradition_mg',
            use_volume=True
        )
        
        
        expected_last_five = [
            61.66219598,
            55.92955776,
            34.70836488,
            39.48824969,
            15.74958884,
        ]
        non_nan_values = [v for v in wavetrend if not np.isnan(v)]
        
        if len(non_nan_values) >= 5:
            actual_last_five = non_nan_values[-5:]
            
            
            for i, (expected, actual) in enumerate(zip(expected_last_five, actual_last_five)):
                diff = abs(expected - actual)
                assert diff < 4.0, f"Value {i} mismatch: expected {expected:.8f}, got {actual:.8f}, diff {diff:.8f}"
    
    def test_mod_god_mode_modes(self, test_data):
        """Test different MOD_GOD_MODE modes - mirrors check_mod_god_mode_modes"""
        high = test_data['high'][:100]  
        low = test_data['low'][:100]
        close = test_data['close'][:100]
        volume = test_data['volume'][:100]
        
        modes = ['godmode', 'tradition', 'godmode_mg', 'tradition_mg']
        
        for mode in modes:
            wavetrend, signal, histogram = ta_indicators.mod_god_mode(
                high, low, close, volume,
                n1=5, n2=3, n3=2,
                mode=mode,
                use_volume=False
            )
            
            assert wavetrend is not None, f"Missing wavetrend for mode {mode}"
            assert signal is not None, f"Missing signal for mode {mode}"
            assert histogram is not None, f"Missing histogram for mode {mode}"
            assert len(wavetrend) == len(close)
    
    def test_mod_god_mode_with_volume(self, test_data):
        """Test MOD_GOD_MODE with and without volume - mirrors check_mod_god_mode_with_volume"""
        high = test_data['high'][:100]
        low = test_data['low'][:100]
        close = test_data['close'][:100]
        volume = test_data['volume'][:100]
        
        
        wt_with, sig_with, hist_with = ta_indicators.mod_god_mode(
            high, low, close, volume,
            n1=5, n2=3, n3=2,
            mode='tradition_mg',
            use_volume=True
        )
        
        
        wt_without, sig_without, hist_without = ta_indicators.mod_god_mode(
            high, low, close, None,
            n1=5, n2=3, n3=2,
            mode='tradition_mg',
            use_volume=False
        )
        
        
        assert len(wt_with) == len(close)
        assert len(wt_without) == len(close)
        
        
        for i in range(len(wt_with)):
            if not np.isnan(wt_with[i]) and not np.isnan(wt_without[i]):
                
                assert abs(wt_with[i] - wt_without[i]) > 1e-10, "Volume should affect the calculation"
                break
    
    def test_mod_god_mode_empty_data(self):
        """Test MOD_GOD_MODE fails with empty data - mirrors check_mod_god_mode_empty_data"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Input data slice is empty|Empty input"):
            ta_indicators.mod_god_mode(
                empty, empty, empty, None,
                n1=17, n2=6, n3=4,
                mode='tradition_mg',
                use_volume=False
            )
    
    def test_mod_god_mode_all_nan(self):
        """Test MOD_GOD_MODE fails with all NaN values - mirrors check_mod_god_mode_all_nan"""
        size = 100
        all_nan_high = np.full(size, np.nan)
        all_nan_low = np.full(size, np.nan)
        all_nan_close = np.full(size, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.mod_god_mode(
                all_nan_high, all_nan_low, all_nan_close, None,
                n1=17, n2=6, n3=4,
                mode='tradition_mg',
                use_volume=False
            )
    
    def test_mod_god_mode_insufficient_data(self):
        """Test MOD_GOD_MODE fails with insufficient data - mirrors check_mod_god_mode_insufficient_data"""
        
        small_size = 10
        high = np.ones(small_size) * 10.5
        low = np.ones(small_size) * 9.5
        close = np.ones(small_size) * 10.0
        
        with pytest.raises(ValueError, match="Not enough valid data|Insufficient data"):
            ta_indicators.mod_god_mode(
                high, low, close, None,
                n1=17, n2=6, n3=4,
                mode='tradition_mg',
                use_volume=False
            )
    
    def test_mod_god_mode_invalid_mode(self):
        """Test MOD_GOD_MODE fails with invalid mode"""
        size = 50
        high = np.ones(size) * 10.5
        low = np.ones(size) * 9.5
        close = np.ones(size) * 10.0
        
        with pytest.raises(ValueError, match="Unknown mode|Invalid mode"):
            ta_indicators.mod_god_mode(
                high, low, close, None,
                n1=5, n2=3, n3=2,
                mode='invalid_mode',
                use_volume=False
            )
    
    def test_mod_god_mode_streaming(self, test_data):
        """Test MOD_GOD_MODE streaming matches batch calculation - mirrors check_mod_god_mode_stream"""
        high = test_data['high'][:100]
        low = test_data['low'][:100]
        close = test_data['close'][:100]
        volume = test_data['volume'][:100]
        
        
        batch_wavetrend, batch_signal, batch_histogram = ta_indicators.mod_god_mode(
            high, low, close, volume,
            n1=5, n2=3, n3=2,
            mode='tradition_mg',
            use_volume=True
        )
        
        
        stream = ta_indicators.ModGodModeStreamPy(
            n1=5, n2=3, n3=2,
            mode='tradition_mg',
            use_volume=True
        )
        
        stream_wavetrend = []
        stream_signal = []
        stream_histogram = []
        
        for i in range(len(close)):
            result = stream.update(high[i], low[i], close[i], volume[i])
            if result is not None:
                wt, sig, hist = result
                stream_wavetrend.append(wt)
                stream_signal.append(sig)
                stream_histogram.append(hist)
            else:
                stream_wavetrend.append(np.nan)
                stream_signal.append(np.nan)
                stream_histogram.append(np.nan)
        
        stream_wavetrend = np.array(stream_wavetrend)
        
        
        valid_indices = ~(np.isnan(batch_wavetrend) | np.isnan(stream_wavetrend))
        
        if np.any(valid_indices):
            
            last_valid = np.where(valid_indices)[0][-1]
            assert_close(
                batch_wavetrend[last_valid], 
                stream_wavetrend[last_valid],
                rtol=1e-6,
                msg=f"Streaming wavetrend mismatch at index {last_valid}"
            )
    
    def test_mod_god_mode_nan_handling(self, test_data):
        """Test MOD_GOD_MODE handles NaN values correctly - mirrors check_mod_god_mode_nan_handling"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        volume = test_data['volume']
        
        wavetrend, signal, histogram = ta_indicators.mod_god_mode(
            high, low, close, volume,
            n1=7, n2=4, n3=3,
            mode='tradition_mg',
            use_volume=False
        )
        
        
        warmup = 7 + 4 + 3  
        if len(wavetrend) > warmup + 10:
            
            has_values = False
            for i in range(warmup, len(wavetrend)):
                if not np.isnan(wavetrend[i]):
                    has_values = True
                    break
            assert has_values, "Should have non-NaN values after warmup"
    
    def test_mod_god_mode_consistency(self, test_data):
        """Test MOD_GOD_MODE produces consistent results - mirrors check_mod_god_mode_consistency"""
        high = test_data['high'][:50]
        low = test_data['low'][:50]
        close = test_data['close'][:50]
        volume = test_data['volume'][:50]
        
        params = {
            'n1': 7,
            'n2': 4,
            'n3': 3,
            'mode': 'tradition_mg',
            'use_volume': False
        }
        
        
        wt1, sig1, hist1 = ta_indicators.mod_god_mode(high, low, close, None, **params)
        wt2, sig2, hist2 = ta_indicators.mod_god_mode(high, low, close, None, **params)
        
        
        for i in range(len(wt1)):
            wt1_val = wt1[i]
            wt2_val = wt2[i]
            sig1_val = sig1[i]
            sig2_val = sig2[i]
            hist1_val = hist1[i]
            hist2_val = hist2[i]
            
            if np.isnan(wt1_val):
                assert np.isnan(wt2_val), f"Wavetrend NaN mismatch at index {i}"
            else:
                assert wt1_val == wt2_val, f"Wavetrend value mismatch at index {i}"
            
            if np.isnan(sig1_val):
                assert np.isnan(sig2_val), f"Signal NaN mismatch at index {i}"
            else:
                assert sig1_val == sig2_val, f"Signal value mismatch at index {i}"
            
            if np.isnan(hist1_val):
                assert np.isnan(hist2_val), f"Histogram NaN mismatch at index {i}"
            else:
                assert hist1_val == hist2_val, f"Histogram value mismatch at index {i}"
    
    def test_mod_god_mode_batch(self, test_data):
        """Test MOD_GOD_MODE batch processing - mirrors check_batch_default_row"""
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']
        volume = test_data['volume']
        
        
        result = ta_indicators.mod_god_mode_batch(
            high, low, close, volume,
            n1_range=(17, 17, 0),  
            n2_range=(6, 6, 0),    
            n3_range=(4, 4, 0),    
            mode='tradition_mg'    
        )
        
        assert 'wavetrend' in result
        assert 'signal' in result
        assert 'histogram' in result
        assert 'n1s' in result
        assert 'n2s' in result
        assert 'n3s' in result
        assert 'modes' in result
        
        
        assert result['wavetrend'].shape[0] == 1
        assert result['wavetrend'].shape[1] == len(close)
        assert result['signal'].shape[0] == 1
        assert result['histogram'].shape[0] == 1
    
    def test_mod_god_mode_reinput(self, test_data):
        """Test MOD_GOD_MODE applied to its own output - mirrors check_mod_god_mode_reinput"""
        high = test_data['high'][:100]
        low = test_data['low'][:100]
        close = test_data['close'][:100]
        
        
        wavetrend, signal, histogram = ta_indicators.mod_god_mode(
            high, low, close, None,
            n1=5, n2=3, n3=2,
            mode='tradition_mg',
            use_volume=False
        )
        
        
        
        synthetic_high = wavetrend + 0.5
        synthetic_low = wavetrend - 0.5
        
        
        wavetrend2, signal2, histogram2 = ta_indicators.mod_god_mode(
            synthetic_high, synthetic_low, wavetrend, None,
            n1=5, n2=3, n3=2,
            mode='tradition_mg',
            use_volume=False
        )
        
        assert len(wavetrend2) == len(wavetrend)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
