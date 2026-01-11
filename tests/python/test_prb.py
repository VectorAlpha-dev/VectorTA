"""
Python binding tests for PRB (Polynomial Regression Bands) indicator.
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


class TestPrb:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_prb_accuracy(self, test_data):
        """Test PRB matches expected values from Rust tests - mirrors check_prb_accuracy"""
        expected = EXPECTED_OUTPUTS['prb']
        
        
        result = ta_indicators.prb(
            test_data['close'],
            smooth_data=expected['default_params']['smooth_data'],
            smooth_period=expected['default_params']['smooth_period'],
            regression_period=expected['default_params']['regression_period'],
            polynomial_order=expected['default_params']['polynomial_order'],
            regression_offset=expected['default_params']['regression_offset'],
            ndev=expected['default_params']['ndev']
        )
        
        
        values, upper_band, lower_band = result
        assert len(values) == len(test_data['close'])
        assert len(upper_band) == len(test_data['close'])
        assert len(lower_band) == len(test_data['close'])
        
        
        non_nan_values = values[~np.isnan(values)]
        assert len(non_nan_values) >= 5, "Should have at least 5 non-NaN values"
        
        assert_close(
            non_nan_values[-5:],
            expected['last_5_main_values'],
            rtol=0.01,  
            msg="PRB last 5 values mismatch"
        )
        
        
        for i in range(len(values)):
            if not np.isnan(values[i]):
                assert not np.isnan(upper_band[i]), f"Upper band should not be NaN when main is not NaN at index {i}"
                assert not np.isnan(lower_band[i]), f"Lower band should not be NaN when main is not NaN at index {i}"
                assert upper_band[i] > values[i], f"Upper band should be > main at index {i}"
                assert values[i] > lower_band[i], f"Main should be > lower band at index {i}"
                
                
                
                
                band_width = upper_band[i] - values[i]
                assert band_width > 0, f"Band width should be positive at index {i}"
    
    def test_prb_partial_params(self, test_data):
        """Test PRB with partial parameters - mirrors check_prb_partial_params"""
        close = test_data['close']
        
        
        result = ta_indicators.prb(
            close,
            smooth_data=True,
            smooth_period=10,
            regression_period=100,
            polynomial_order=2,
            regression_offset=0,
            ndev=2.0
        )
        
        values, upper_band, lower_band = result
        assert len(values) == len(close)
        assert len(upper_band) == len(close)
        assert len(lower_band) == len(close)
    
    def test_prb_zero_period(self):
        """Test PRB fails with zero period - mirrors check_prb_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.prb(
                input_data, 
                smooth_data=False,
                smooth_period=10,
                regression_period=0,
                polynomial_order=2,
                regression_offset=0,
                ndev=2.0
            )
    
    def test_prb_period_exceeds_length(self):
        """Test PRB fails when period exceeds data length - mirrors check_prb_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.prb(
                data_small,
                smooth_data=False,
                smooth_period=10,
                regression_period=10,
                polynomial_order=2,
                regression_offset=0,
                ndev=2.0
            )
    
    def test_prb_very_small_dataset(self):
        """Test PRB fails with insufficient data - mirrors check_prb_very_small_dataset"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.prb(
                single_point,
                smooth_data=False,
                smooth_period=10,
                regression_period=100,
                polynomial_order=2,
                regression_offset=0,
                ndev=2.0
            )
    
    def test_prb_empty_input(self):
        """Test PRB fails with empty input - mirrors check_prb_empty_input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Input data slice is empty"):
            ta_indicators.prb(
                empty,
                smooth_data=True,
                smooth_period=10,
                regression_period=100,
                polynomial_order=2,
                regression_offset=0,
                ndev=2.0
            )
    
    def test_prb_invalid_smooth_period(self):
        """Test PRB fails with invalid smooth period - mirrors check_prb_invalid_smooth_period"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        with pytest.raises(ValueError, match="Invalid smooth period"):
            ta_indicators.prb(
                data,
                smooth_data=True,
                smooth_period=1,  
                regression_period=2,
                polynomial_order=2,
                regression_offset=0,
                ndev=2.0
            )
    
    def test_prb_invalid_order(self):
        """Test PRB fails with invalid polynomial order - mirrors check_prb_invalid_order"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        with pytest.raises(ValueError, match="Invalid polynomial order"):
            ta_indicators.prb(
                data,
                smooth_data=False,
                smooth_period=10,
                regression_period=2,
                polynomial_order=0,  
                regression_offset=0,
                ndev=2.0
            )
    
    def test_prb_reinput(self, test_data):
        """Test PRB applied twice (re-input) - mirrors check_prb_reinput"""
        close = test_data['close']
        
        
        first_result = ta_indicators.prb(
            close,
            smooth_data=False,
            smooth_period=10,
            regression_period=50,
            polynomial_order=2,
            regression_offset=0,
            ndev=2.0
        )
        
        first_values, first_upper, first_lower = first_result
        assert len(first_values) == len(close)
        
        
        second_result = ta_indicators.prb(
            first_values,
            smooth_data=False,
            smooth_period=10,
            regression_period=50,
            polynomial_order=2,
            regression_offset=0,
            ndev=2.0
        )
        
        second_values, second_upper, second_lower = second_result
        assert len(second_values) == len(first_values)
        
        
        non_nan_first = first_values[~np.isnan(first_values)]
        non_nan_second = second_values[~np.isnan(second_values)]
        assert len(non_nan_second) > 0, "Should have non-NaN values after reinput"
    
    def test_prb_nan_handling(self, test_data):
        """Test PRB handles NaN values correctly - mirrors check_prb_nan_handling"""
        close = test_data['close']
        
        result = ta_indicators.prb(
            close,
            smooth_data=False,
            smooth_period=10,
            regression_period=50,
            polynomial_order=2,
            regression_offset=0,
            ndev=2.0
        )
        
        values, upper_band, lower_band = result
        assert len(values) == len(close)
        
        
        if len(values) > 240:
            assert not np.any(np.isnan(values[240:])), "Found unexpected NaN after warmup period"
            assert not np.any(np.isnan(upper_band[240:])), "Found unexpected NaN in upper band after warmup"
            assert not np.any(np.isnan(lower_band[240:])), "Found unexpected NaN in lower band after warmup"
        
        
        warmup = 50 - 1  
        assert np.all(np.isnan(values[:warmup])), "Expected NaN in warmup period for main values"
        assert np.all(np.isnan(upper_band[:warmup])), "Expected NaN in warmup period for upper band"
        assert np.all(np.isnan(lower_band[:warmup])), "Expected NaN in warmup period for lower band"
    
    def test_prb_streaming(self, test_data):
        """Test PRB streaming matches batch calculation - mirrors check_prb_streaming"""
        close = test_data['close'][:100]  
        
        
        batch_result = ta_indicators.prb(
            close,
            smooth_data=False,
            smooth_period=10,
            regression_period=10,
            polynomial_order=2,
            regression_offset=0,
            ndev=2.0
        )
        batch_values, batch_upper, batch_lower = batch_result
        
        
        stream = ta_indicators.PrbStreamPy(
            smooth_data=False,
            smooth_period=10,
            regression_period=10,
            polynomial_order=2,
            regression_offset=0,
            ndev=2.0
        )
        
        stream_values = []
        stream_upper = []
        stream_lower = []
        
        for price in close:
            result = stream.update(price)
            if result is not None:
                val, up, lo = result
                stream_values.append(val)
                stream_upper.append(up)
                stream_lower.append(lo)
            else:
                stream_values.append(np.nan)
                stream_upper.append(np.nan)
                stream_lower.append(np.nan)
        
        stream_values = np.array(stream_values)
        stream_upper = np.array(stream_upper)
        stream_lower = np.array(stream_lower)
        
        
        assert len(batch_values) == len(stream_values)
        
        
        for i, (b_val, s_val, b_up, s_up, b_lo, s_lo) in enumerate(
            zip(batch_values, stream_values, batch_upper, stream_upper, batch_lower, stream_lower)
        ):
            if np.isnan(b_val) and np.isnan(s_val):
                continue
            assert_close(b_val, s_val, rtol=1e-9, atol=1e-9, 
                        msg=f"PRB streaming main value mismatch at index {i}")
            assert_close(b_up, s_up, rtol=1e-9, atol=1e-9,
                        msg=f"PRB streaming upper band mismatch at index {i}")
            assert_close(b_lo, s_lo, rtol=1e-9, atol=1e-9,
                        msg=f"PRB streaming lower band mismatch at index {i}")
    
    def test_prb_batch(self, test_data):
        """Test PRB batch processing - mirrors check_batch_default_row"""
        close = test_data['close'][:200]  
        
        result = ta_indicators.prb_batch(
            close,
            smooth_data=False,
            smooth_period_start=10,
            smooth_period_end=10,
            smooth_period_step=0,
            regression_period_start=100,
            regression_period_end=100,
            regression_period_step=0,
            polynomial_order_start=2,
            polynomial_order_end=2,
            polynomial_order_step=0,
            regression_offset_start=0,
            regression_offset_end=0,
            regression_offset_step=0,
            kernel=None  
        )
        
        assert 'values' in result
        assert 'upper' in result
        assert 'lower' in result
        assert 'smooth_periods' in result
        assert 'regression_periods' in result
        assert 'polynomial_orders' in result
        assert 'regression_offsets' in result
        assert 'rows' in result
        assert 'cols' in result
        
        
        assert result['rows'] == 1
        assert result['cols'] == len(close)
        assert result['values'].shape == (1, len(close))
        assert result['upper'].shape == (1, len(close))
        assert result['lower'].shape == (1, len(close))
        
        
        main_row = result['values'][0]
        upper_row = result['upper'][0]
        lower_row = result['lower'][0]
        
        for i in range(len(main_row)):
            if not np.isnan(main_row[i]):
                assert upper_row[i] > main_row[i], f"Upper should be > main at index {i}"
                assert main_row[i] > lower_row[i], f"Main should be > lower at index {i}"
    
    def test_prb_batch_sweep(self, test_data):
        """Test PRB batch with parameter sweep"""
        close = test_data['close'][:100]  
        
        result = ta_indicators.prb_batch(
            close,
            smooth_data=False,
            smooth_period_start=8,
            smooth_period_end=12,
            smooth_period_step=2,
            regression_period_start=50,
            regression_period_end=60,
            regression_period_step=5,
            polynomial_order_start=1,
            polynomial_order_end=3,
            polynomial_order_step=1,
            regression_offset_start=0,
            regression_offset_end=2,
            regression_offset_step=1,
            kernel=None  
        )
        
        
        expected_combos = 3 * 3 * 3 * 3
        assert result['rows'] == expected_combos
        assert result['cols'] == len(close)
        assert result['values'].shape == (expected_combos, len(close))
        assert result['upper'].shape == (expected_combos, len(close))
        assert result['lower'].shape == (expected_combos, len(close))
        
        
        for row_idx in range(expected_combos):
            main_row = result['values'][row_idx]
            upper_row = result['upper'][row_idx]
            lower_row = result['lower'][row_idx]
            
            
            for i in range(len(main_row)):
                if not np.isnan(main_row[i]):
                    assert upper_row[i] > main_row[i], f"Row {row_idx}: Upper should be > main at index {i}"
                    assert main_row[i] > lower_row[i], f"Row {row_idx}: Main should be > lower at index {i}"
    
    def test_prb_all_nan_input(self):
        """Test PRB with all NaN values"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.prb(
                all_nan,
                smooth_data=True,
                smooth_period=10,
                regression_period=100,
                polynomial_order=2,
                regression_offset=0,
                ndev=2.0
            )
    
    def test_prb_no_smoothing(self, test_data):
        """Test PRB without smoothing"""
        close = test_data['close'][:150]
        
        result = ta_indicators.prb(
            close,
            smooth_data=False,  
            smooth_period=10,   
            regression_period=50,
            polynomial_order=2,
            regression_offset=0,
            ndev=2.0
        )
        
        values, upper_band, lower_band = result
        assert len(values) == len(close)
        assert len(upper_band) == len(close)
        assert len(lower_band) == len(close)
        
        
        non_nan_count = np.sum(~np.isnan(values))
        assert non_nan_count > 0
    
    def test_prb_linear_regression(self):
        """Test PRB with linear regression (order=1)"""
        
        data = np.arange(100, dtype=float) * 10 + 1000
        
        result = ta_indicators.prb(
            data,
            smooth_data=False,
            smooth_period=10,
            regression_period=20,
            polynomial_order=1,  
            regression_offset=0,
            ndev=2.0
        )
        
        values, upper_band, lower_band = result
        assert len(values) == len(data)
        
        
        
        non_nan_values = values[~np.isnan(values)]
        assert len(non_nan_values) > 0
        
        
        for i in range(len(values)):
            if not np.isnan(values[i]):
                assert upper_band[i] > values[i]
                assert values[i] > lower_band[i]
    
    def test_prb_cubic_regression(self, test_data):
        """Test PRB with cubic regression (order=3)"""
        close = test_data['close'][:200]
        
        result = ta_indicators.prb(
            close,
            smooth_data=True,
            smooth_period=10,
            regression_period=50,
            polynomial_order=3,  
            regression_offset=0,
            ndev=2.0
        )
        
        values, upper_band, lower_band = result
        assert len(values) == len(close)
        
        non_nan_count = np.sum(~np.isnan(values))
        assert non_nan_count > 0
        
        
        for i in range(len(values)):
            if not np.isnan(values[i]):
                assert upper_band[i] > values[i]
                assert values[i] > lower_band[i]
    
    def test_prb_with_offset(self):
        """Test PRB with regression offset"""
        data = np.random.randn(150) * 50 + 1000
        
        
        result_pos = ta_indicators.prb(
            data,
            smooth_data=True,
            smooth_period=10,
            regression_period=50,
            polynomial_order=2,
            regression_offset=5,  
            ndev=2.0
        )
        
        
        result_neg = ta_indicators.prb(
            data,
            smooth_data=True,
            smooth_period=10,
            regression_period=50,
            polynomial_order=2,
            regression_offset=-5,  
            ndev=2.0
        )
        
        values_pos, upper_pos, lower_pos = result_pos
        values_neg, upper_neg, lower_neg = result_neg
        
        assert len(values_pos) == len(data)
        assert len(values_neg) == len(data)
        
        
        assert np.sum(~np.isnan(values_pos)) > 0
        assert np.sum(~np.isnan(values_neg)) > 0
        
        
        non_nan_idx = ~(np.isnan(values_pos) | np.isnan(values_neg))
        if np.any(non_nan_idx):
            assert not np.allclose(values_pos[non_nan_idx], values_neg[non_nan_idx])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])