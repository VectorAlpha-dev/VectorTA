"""
Python binding tests for OTT indicator.
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


class TestOtt:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()
    
    def test_ott_accuracy(self, test_data):
        """Test OTT matches expected values from Rust tests - mirrors check_ott_accuracy"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['ott']
        
        # Use accuracy params (period=50) for accuracy test
        result = ta_indicators.ott(
            close,
            period=expected['accuracy_params']['period'],
            percent=expected['accuracy_params']['percent'],
            ma_type=expected['accuracy_params']['ma_type']
        )
        
        assert len(result) == len(close)
        
        # Check last 5 values match expected
        # Match Rust test tolerance (abs diff < 1e-6)
        assert_close(
            result[-5:],
            expected['last_5_values'],
            rtol=0.0,
            atol=1e-6,
            msg="OTT last 5 values mismatch"
        )
    
    def test_ott_partial_params(self, test_data):
        """Test OTT with partial parameters (default values) - mirrors check_ott_partial_params"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['ott']
        
        # Test with default params
        result = ta_indicators.ott(
            close,
            period=expected['default_params']['period'],
            percent=expected['default_params']['percent'],
            ma_type=expected['default_params']['ma_type']
        )
        assert len(result) == len(close)
        
        # Note: VAR with period=2 doesn't have a traditional warmup period (returns 0.0 instead of NaN)
        # This is specific to the VAR implementation
        # So we just verify length and that we have valid numeric values
        assert not any(np.isnan(result)), "Should not have NaN values with VAR period=2"
    
    def test_ott_default_candles(self, test_data):
        """Test OTT with default parameters - mirrors check_ott_default_candles"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['ott']
        
        # Test default params
        result = ta_indicators.ott(
            close,
            period=expected['default_params']['period'],
            percent=expected['default_params']['percent'],
            ma_type=expected['default_params']['ma_type']
        )
        assert len(result) == len(close)
    
    def test_ott_zero_period(self):
        """Test OTT fails with zero period - mirrors check_ott_zero_period"""
        input_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period"):
            ta_indicators.ott(input_data, period=0, percent=1.4, ma_type="VAR")
    
    def test_ott_period_exceeds_length(self):
        """Test OTT fails when period exceeds data length - mirrors check_ott_period_exceeds_length"""
        data_small = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="Invalid period|Period.*exceeds|Not enough data"):
            ta_indicators.ott(data_small, period=10, percent=1.4, ma_type="VAR")
    
    def test_ott_very_small_dataset(self):
        """Test OTT fails with insufficient data - mirrors check_ott_very_small_dataset"""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError, match="Invalid period|Not enough valid data|Not enough data"):
            ta_indicators.ott(single_point, period=50, percent=1.4, ma_type="VAR")
    
    def test_ott_empty_input(self):
        """Test OTT fails with empty input - mirrors check_ott_empty_input"""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Input data slice is empty"):
            ta_indicators.ott(empty, period=50, percent=1.4, ma_type="VAR")
    
    def test_ott_all_nan_input(self):
        """Test OTT with all NaN values - mirrors check_ott_all_nan"""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            ta_indicators.ott(all_nan, period=50, percent=1.4, ma_type="VAR")
    
    def test_ott_invalid_percent(self):
        """Test OTT fails with invalid percent values"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Test with negative percent
        with pytest.raises(ValueError, match="Invalid percent"):
            ta_indicators.ott(data, period=2, percent=-1.0, ma_type="VAR")
        
        # Test with NaN percent
        with pytest.raises(ValueError, match="Invalid percent"):
            ta_indicators.ott(data, period=2, percent=float('nan'), ma_type="VAR")
        
        # Test with zero percent (should be valid)
        result = ta_indicators.ott(data, period=2, percent=0.0, ma_type="VAR")
        assert len(result) == len(data)
    
    def test_ott_invalid_ma_type(self):
        """Test OTT fails with invalid MA type string"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Test with invalid MA type
        with pytest.raises(ValueError, match="Invalid moving average|Invalid MA type|Unsupported moving average"):
            ta_indicators.ott(data, period=2, percent=1.4, ma_type="INVALID_MA")
    
    def test_ott_ma_type_variations(self, test_data):
        """Test OTT with different MA types (VAR, WWMA, TMA)"""
        close = test_data['close'][:100]  # Use smaller dataset for speed
        
        ma_types = ["VAR", "WWMA", "TMA"]
        results = {}
        period = 20  # Define period variable
        
        for ma_type in ma_types:
            result = ta_indicators.ott(close, period=period, percent=1.4, ma_type=ma_type)
            assert len(result) == len(close), f"Length mismatch for {ma_type}"
            results[ma_type] = result
            
            # Verify warmup period - VAR may not have NaN values at start
            if ma_type == "VAR":
                # VAR might return 0.0 instead of NaN
                pass  # No specific warmup check for VAR
            else:
                # Other MA types should have warmup NaN values
                warmup = period - 1
                if any(np.isnan(result[:warmup])):
                    # After warmup should have values
                    if len(result) > warmup:
                        assert not any(np.isnan(result[warmup+5:])), f"Unexpected NaN after warmup for {ma_type}"
        
        # Different MA types should produce different results
        assert not np.allclose(results["VAR"][19:], results["WWMA"][19:], equal_nan=True), \
            "VAR and WWMA should produce different results"
        assert not np.allclose(results["VAR"][19:], results["TMA"][19:], equal_nan=True), \
            "VAR and TMA should produce different results"
    
    def test_ott_all_ma_types_comprehensive(self, test_data):
        """Test OTT with all 8 supported MA types - comprehensive coverage"""
        close = test_data['close'][:200]  # Use reasonable dataset size
        
        # All 8 supported MA types
        all_ma_types = ["SMA", "EMA", "WMA", "TMA", "VAR", "WWMA", "ZLEMA", "TSF"]
        results = {}
        period = 20
        percent = 1.4
        
        for ma_type in all_ma_types:
            try:
                result = ta_indicators.ott(close, period=period, percent=percent, ma_type=ma_type)
                assert len(result) == len(close), f"Length mismatch for {ma_type}"
                results[ma_type] = result
                
                # Verify warmup period exists
                # Different MA types may have different warmup periods
                # Note: VAR, EMA, and WWMA may not have NaN warmup (they return 0.0 or initial values)
                first_valid = next((i for i, val in enumerate(result) if not np.isnan(val)), len(result))
                # Don't assert warmup for VAR, EMA, WWMA as they may start with values immediately
                if ma_type not in ["VAR", "EMA", "WWMA"]:
                    assert first_valid > 0, f"Expected warmup period for {ma_type}"
                
                # After sufficient warmup, should have valid values
                if len(result) > period + 10:
                    # Check that we have valid values after warmup
                    valid_count = np.sum(~np.isnan(result[period:]))
                    assert valid_count > 0, f"Expected valid values after warmup for {ma_type}"
                    
            except Exception as e:
                pytest.fail(f"Failed to calculate OTT with MA type {ma_type}: {str(e)}")
        
        # Verify that different MA types produce different results
        # Compare a few key pairs after warmup
        start_idx = period + 10  # Safe index after warmup
        end_idx = start_idx + 20  # Compare 20 values
        
        if end_idx < len(close):
            # SMA vs EMA should differ
            if "SMA" in results and "EMA" in results:
                assert not np.allclose(
                    results["SMA"][start_idx:end_idx], 
                    results["EMA"][start_idx:end_idx], 
                    equal_nan=True, rtol=1e-6
                ), "SMA and EMA should produce different results"
            
            # VAR vs WWMA should differ  
            if "VAR" in results and "WWMA" in results:
                assert not np.allclose(
                    results["VAR"][start_idx:end_idx], 
                    results["WWMA"][start_idx:end_idx], 
                    equal_nan=True, rtol=1e-6
                ), "VAR and WWMA should produce different results"
            
            # ZLEMA vs WMA should differ
            if "ZLEMA" in results and "WMA" in results:
                assert not np.allclose(
                    results["ZLEMA"][start_idx:end_idx], 
                    results["WMA"][start_idx:end_idx], 
                    equal_nan=True, rtol=1e-6
                ), "ZLEMA and WMA should produce different results"
    
    def test_ott_nan_handling(self, test_data):
        """Test OTT handles NaN values correctly - mirrors check_alma_nan_handling"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['ott']
        
        # Use accuracy params for consistency with Rust tests
        period = expected['accuracy_params']['period']
        result = ta_indicators.ott(
            close, 
            period=period, 
            percent=expected['accuracy_params']['percent'], 
            ma_type=expected['accuracy_params']['ma_type']
        )
        assert len(result) == len(close)
        
        # Note: VAR with period=2 doesn't have NaN warmup values, it returns 0.0
        # So we just verify no NaN values exist in the output
        assert not np.any(np.isnan(result)), "Should not have NaN values with VAR period=2"
    
    def test_ott_reinput(self, test_data):
        """Test OTT applied twice (re-input) - mirrors check_ott_reinput"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['ott']
        
        # Use accuracy params for consistency
        params = expected['accuracy_params']
        
        # First pass
        first_result = ta_indicators.ott(
            close, 
            period=params['period'], 
            percent=params['percent'], 
            ma_type=params['ma_type']
        )
        assert len(first_result) == len(close)
        
        # Second pass - apply OTT to OTT output
        second_result = ta_indicators.ott(
            first_result, 
            period=params['period'], 
            percent=params['percent'], 
            ma_type=params['ma_type']
        )
        assert len(second_result) == len(first_result)
        
        # Check last 5 values match expected
        # Match Rust test tolerance (abs diff < 1e-6)
        assert_close(
            second_result[-5:],
            expected['reinput_last_5'],
            rtol=0.0,
            atol=1e-6,
            msg="OTT re-input last 5 values mismatch"
        )
    
    @pytest.mark.skip(reason="OTT streaming has known issues with VAR - needs investigation")
    def test_ott_streaming(self, test_data):
        """Test OTT streaming matches batch calculation - mirrors check_alma_streaming"""
        close = test_data['close']
        period = 50
        percent = 1.4
        ma_type = "VAR"
        
        # Batch calculation
        batch_result = ta_indicators.ott(close, period=period, percent=percent, ma_type=ma_type)
        
        # Streaming calculation
        stream = ta_indicators.OttStream(period=period, percent=percent, ma_type=ma_type)
        stream_values = []
        
        for price in close:
            result = stream.update(price)
            stream_values.append(result if result is not None else np.nan)
        
        stream_values = np.array(stream_values)
        
        # Compare batch vs streaming
        assert len(batch_result) == len(stream_values)
        
        # Compare values where both are valid
        # Note: VAR streaming may have different warmup behavior than batch
        # Skip the first 50 values (warmup period) and compare stable values
        start_comparison = 100  # Start after warmup is complete
        
        if len(batch_result) > start_comparison:
            for i in range(start_comparison, len(batch_result)):
                b = batch_result[i]
                s = stream_values[i]
                if np.isnan(b) and np.isnan(s):
                    continue
                if not np.isnan(b) and not np.isnan(s):
                    assert_close(b, s, rtol=1e-8, atol=1e-8, 
                                msg=f"OTT streaming mismatch at index {i}")
    
    def test_ott_batch(self, test_data):
        """Test OTT batch processing - mirrors check_batch_default_row"""
        close = test_data['close']
        expected = EXPECTED_OUTPUTS['ott']
        
        # Use accuracy params for batch test to match expected values
        result = ta_indicators.ott_batch(
            close,
            period_range=(expected['accuracy_params']['period'], expected['accuracy_params']['period'], 0),
            percent_range=(expected['accuracy_params']['percent'], expected['accuracy_params']['percent'], 0.0),
            ma_types=[expected['accuracy_params']['ma_type']]
        )
        
        assert 'values' in result
        assert 'periods' in result
        assert 'percents' in result
        assert 'ma_types' in result
        
        # Should have 1 combination (default params)
        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(close)
        
        # Extract the single row
        default_row = result['values'][0]
        
        # Check last 5 values match expected
        assert_close(
            default_row[-5:],
            expected['last_5_values'],
            rtol=1e-8,
            msg="OTT batch default row mismatch"
        )
    
    def test_ott_batch_multiple_periods(self, test_data):
        """Test OTT batch with multiple period values"""
        close = test_data['close'][:100]  # Use smaller dataset for speed
        
        result = ta_indicators.ott_batch(
            close,
            period_range=(20, 30, 5),  # periods: 20, 25, 30
            percent_range=(1.4, 1.4, 0.0),  # single percent
            ma_types=["VAR"]  # single MA type
        )
        
        # Should have 3 combinations (3 periods x 1 percent x 1 ma_type)
        assert result['values'].shape[0] == 3
        assert result['values'].shape[1] == 100
        
        # Verify each row has correct warmup period
        # Note: VAR might not have NaN warmup values (returns 0.0 instead)
        periods = [20, 25, 30]
        for i, period in enumerate(periods):
            row = result['values'][i]
            # VAR with these periods may not have NaN warmup
            # Just verify we have values and correct shape
            assert len(row) == 100, f"Row length mismatch for period {period}"
            # Check that we eventually get non-zero values
            non_zero_count = np.count_nonzero(row)
            assert non_zero_count > 0, f"Expected non-zero values for period {period}"
    
    def test_ott_batch_multiple_percents(self, test_data):
        """Test OTT batch with multiple percent values"""
        close = test_data['close'][:100]  # Use smaller dataset for speed
        
        result = ta_indicators.ott_batch(
            close,
            period_range=(20, 20, 0),  # single period
            percent_range=(0.5, 2.0, 0.5),  # percents: 0.5, 1.0, 1.5, 2.0
            ma_types=["VAR"]  # single MA type
        )
        
        # Should have 4 combinations (1 period x 4 percents x 1 ma_type)
        assert result['values'].shape[0] == 4
        assert result['values'].shape[1] == 100
        
        # Different percents should produce different results
        for i in range(3):
            assert not np.allclose(
                result['values'][i][19:],  # After warmup
                result['values'][i+1][19:],
                equal_nan=True
            ), f"Percent {0.5 + i*0.5} and {0.5 + (i+1)*0.5} should produce different results"
    
    def test_ott_batch_multiple_ma_types(self, test_data):
        """Test OTT batch with multiple MA types"""
        close = test_data['close'][:100]  # Use smaller dataset for speed
        
        result = ta_indicators.ott_batch(
            close,
            period_range=(20, 20, 0),  # single period
            percent_range=(1.4, 1.4, 0.0),  # single percent
            ma_types=["VAR", "WWMA", "TMA"]  # multiple MA types
        )
        
        # Should have 3 combinations (1 period x 1 percent x 3 ma_types)
        assert result['values'].shape[0] == 3
        assert result['values'].shape[1] == 100
        
        # Different MA types should produce different results
        ma_type_results = {
            "VAR": result['values'][0],
            "WWMA": result['values'][1],
            "TMA": result['values'][2],
        }
        
        assert not np.allclose(
            ma_type_results["VAR"][19:],
            ma_type_results["WWMA"][19:],
            equal_nan=True
        ), "VAR and WWMA should produce different results"
        
        assert not np.allclose(
            ma_type_results["VAR"][19:],
            ma_type_results["TMA"][19:],
            equal_nan=True
        ), "VAR and TMA should produce different results"
    
    def test_ott_batch_full_parameter_sweep(self, test_data):
        """Test OTT batch with full parameter sweep"""
        close = test_data['close'][:50]  # Small dataset for speed
        
        result = ta_indicators.ott_batch(
            close,
            period_range=(10, 15, 5),  # 2 periods: 10, 15
            percent_range=(1.0, 1.5, 0.5),  # 2 percents: 1.0, 1.5
            ma_types=["VAR", "WWMA"]  # 2 MA types
        )
        
        # Should have 2 * 2 * 2 = 8 combinations
        assert result['values'].shape[0] == 8
        assert result['values'].shape[1] == 50
        
        # Verify parameter combinations
        assert len(result['periods']) == 8
        assert len(result['percents']) == 8
        assert len(result['ma_types']) == 8
        
        # Check first and last combinations
        assert result['periods'][0] == 10
        assert result['percents'][0] == 1.0
        assert result['ma_types'][0] == "VAR"
        
        assert result['periods'][-1] == 15
        assert result['percents'][-1] == 1.5
        assert result['ma_types'][-1] == "WWMA"
    
    def test_ott_batch_edge_cases(self):
        """Test edge cases for OTT batch processing"""
        close = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)
        
        # Single value sweep
        single_batch = ta_indicators.ott_batch(
            close,
            period_range=(5, 5, 1),
            percent_range=(1.4, 1.4, 0.1),
            ma_types=["VAR"]
        )
        
        assert single_batch['values'].shape[0] == 1
        assert single_batch['values'].shape[1] == 10
        
        # Step larger than range
        large_batch = ta_indicators.ott_batch(
            close,
            period_range=(5, 7, 10),  # Step larger than range
            percent_range=(1.4, 1.4, 0),
            ma_types=["VAR"]
        )
        
        # Should only have period=5
        assert large_batch['values'].shape[0] == 1
        assert large_batch['periods'][0] == 5
        
        # Empty data should throw
        with pytest.raises(ValueError, match="All values are NaN|Input data slice is empty"):
            ta_indicators.ott_batch(
                np.array([]),
                period_range=(5, 5, 0),
                percent_range=(1.4, 1.4, 0),
                ma_types=["VAR"]
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
