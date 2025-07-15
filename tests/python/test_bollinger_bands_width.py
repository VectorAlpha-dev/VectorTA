"""
Python binding tests for Bollinger Bands Width indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import pytest
import numpy as np
import os
import sys

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import my_project

from test_utils import (
    load_test_data,
    assert_close,
    assert_array_close,
    assert_all_nan,
    assert_no_nan,
    EXPECTED_OUTPUTS
)


class TestBollingerBandsWidth:
    """Test suite for Bollinger Bands Width Python bindings."""
    
    @classmethod
    def setup_class(cls):
        """Load test data once for all tests."""
        cls.data = load_test_data()
        cls.close = np.array(cls.data['close'], dtype=np.float64)
        cls.high = np.array(cls.data['high'], dtype=np.float64)
        cls.low = np.array(cls.data['low'], dtype=np.float64)
        cls.expected = EXPECTED_OUTPUTS.get('bollinger_bands_width', {})
    
    def test_partial_params(self):
        """Test BBW with partial parameters - mirrors check_bbw_partial_params."""
        # Custom period and devup, default others
        result = my_project.bollinger_bands_width(
            self.close,
            period=22,
            devup=2.2,
            devdn=2.0,  # default
            matype="ema",
            devtype=None  # use default
        )
        
        assert len(result) == len(self.close)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
    
    def test_default_params(self):
        """Test BBW with default parameters - mirrors check_bbw_default."""
        # Use default params
        result = my_project.bollinger_bands_width(
            self.close,
            period=20,
            devup=2.0,
            devdn=2.0
        )
        
        assert len(result) == len(self.close)
        
        # Check warmup period
        assert_all_nan(result[:19], "Expected NaN in warmup period")
        
        # After warmup, should have values
        assert_no_nan(result[240:], "Expected no NaN after sufficient warmup")
    
    def test_accuracy(self):
        """Test BBW accuracy with expected values."""
        result = my_project.bollinger_bands_width(
            self.close,
            period=self.expected.get('default_params', {}).get('period', 20),
            devup=self.expected.get('default_params', {}).get('devup', 2.0),
            devdn=self.expected.get('default_params', {}).get('devdn', 2.0)
        )
        
        assert len(result) == len(self.close)
        
        # If we have expected values, check them
        if 'last_5_values' in self.expected:
            last_5 = result[-5:]
            assert_array_close(
                last_5,
                self.expected['last_5_values'],
                rtol=1e-8,
                msg="BBW last 5 values mismatch"
            )
    
    def test_zero_period(self):
        """Test BBW fails with zero period - mirrors check_bbw_zero_period."""
        with pytest.raises(ValueError, match="period"):
            my_project.bollinger_bands_width(
                np.array([10.0, 20.0, 30.0]),
                period=0,
                devup=2.0,
                devdn=2.0
            )
    
    def test_period_exceeds_length(self):
        """Test BBW fails when period exceeds data length - mirrors check_bbw_period_exceeds_length."""
        small_data = np.array([10.0, 20.0, 30.0])
        
        with pytest.raises(ValueError, match="period"):
            my_project.bollinger_bands_width(
                small_data,
                period=10,
                devup=2.0,
                devdn=2.0
            )
    
    def test_very_small_dataset(self):
        """Test BBW fails with insufficient data - mirrors check_bbw_very_small_dataset."""
        single_point = np.array([42.0])
        
        with pytest.raises(ValueError):
            my_project.bollinger_bands_width(
                single_point,
                period=20,
                devup=2.0,
                devdn=2.0
            )
    
    def test_empty_input(self):
        """Test BBW fails with empty input."""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="Empty"):
            my_project.bollinger_bands_width(
                empty,
                period=20,
                devup=2.0,
                devdn=2.0
            )
    
    def test_all_nan_input(self):
        """Test BBW with all NaN values."""
        all_nan = np.full(100, np.nan)
        
        with pytest.raises(ValueError, match="All values are NaN"):
            my_project.bollinger_bands_width(
                all_nan,
                period=20,
                devup=2.0,
                devdn=2.0
            )
    
    def test_different_matypes(self):
        """Test BBW with different moving average types."""
        matypes = ["sma", "ema", "wma", "dema", "tema"]
        
        for matype in matypes:
            result = my_project.bollinger_bands_width(
                self.close[:100],  # Use smaller dataset
                period=14,
                devup=2.0,
                devdn=2.0,
                matype=matype
            )
            
            assert len(result) == 100
            assert isinstance(result, np.ndarray)
    
    def test_different_devtypes(self):
        """Test BBW with different deviation types."""
        devtypes = [0, 1, 2]  # stddev, mean_ad, median_ad
        
        for devtype in devtypes:
            result = my_project.bollinger_bands_width(
                self.close[:100],  # Use smaller dataset
                period=14,
                devup=2.0,
                devdn=2.0,
                devtype=devtype
            )
            
            assert len(result) == 100
            assert isinstance(result, np.ndarray)
    
    def test_nan_handling(self):
        """Test BBW handles NaN values correctly."""
        result = my_project.bollinger_bands_width(
            self.close,
            period=20,
            devup=2.0,
            devdn=2.0
        )
        
        assert len(result) == len(self.close)
        
        # First period-1 values should be NaN
        assert_all_nan(result[:19], "Expected NaN in warmup period")
        
        # After a reasonable warmup, no NaN values should exist
        if len(result) > 240:
            assert_no_nan(result[240:], "Found unexpected NaN after warmup")
    
    def test_streaming_api(self):
        """Test streaming BBW calculation."""
        period = 20
        stream = my_project.BollingerBandsWidthStream(
            period=period,
            devup=2.0,
            devdn=2.0,
            matype="sma",  # Default moving average type
            devtype=0      # Default deviation type (stddev)
        )
        
        # Feed data one by one
        results = []
        for i, val in enumerate(self.close[:50]):
            result = stream.update(val)
            results.append(result)
        
        # First period-1 updates should return None
        for i in range(period - 1):
            assert results[i] is None, f"Expected None at index {i}"
        
        # After that, should get values
        for i in range(period - 1, 50):
            assert results[i] is not None, f"Expected value at index {i}"
            assert not np.isnan(results[i]), f"Got NaN at index {i}"
    
    def test_batch_single_params(self):
        """Test batch processing with single parameter set."""
        # Single parameter combination
        result = my_project.bollinger_bands_width_batch(
            self.close,
            period_range=(20, 20, 0),
            devup_range=(2.0, 2.0, 0),
            devdn_range=(2.0, 2.0, 0)
        )
        
        assert 'values' in result
        assert 'periods' in result
        assert 'devups' in result
        assert 'devdns' in result
        
        # Should have 1 row
        assert result['values'].shape[0] == 1
        assert result['values'].shape[1] == len(self.close)
        
        # Compare with single calculation
        single_result = my_project.bollinger_bands_width(
            self.close,
            period=20,
            devup=2.0,
            devdn=2.0
        )
        
        assert_array_close(
            result['values'][0],
            single_result,
            rtol=1e-10,
            msg="Batch vs single mismatch"
        )
    
    def test_batch_multiple_params(self):
        """Test batch processing with multiple parameter combinations."""
        # Multiple periods and deviations
        result = my_project.bollinger_bands_width_batch(
            self.close[:100],  # Use smaller dataset
            period_range=(10, 30, 10),  # 10, 20, 30
            devup_range=(1.5, 2.5, 0.5),  # 1.5, 2.0, 2.5
            devdn_range=(2.0, 2.0, 0)  # 2.0
        )
        
        # Should have 3 * 3 * 1 = 9 combinations
        assert result['values'].shape[0] == 9
        assert result['values'].shape[1] == 100
        
        # Check metadata arrays
        assert len(result['periods']) == 9
        assert len(result['devups']) == 9
        assert len(result['devdns']) == 9
        
        # Verify first combination
        assert result['periods'][0] == 10
        assert result['devups'][0] == 1.5
        assert result['devdns'][0] == 2.0
    
    def test_kernel_selection(self):
        """Test different kernel selections."""
        kernels = [None, "scalar", "avx2", "avx512"]
        
        for kernel in kernels:
            try:
                result = my_project.bollinger_bands_width(
                    self.close[:100],
                    period=20,
                    devup=2.0,
                    devdn=2.0,
                    kernel=kernel
                )
                assert len(result) == 100
            except ValueError as e:
                # Some kernels might not be supported on this platform
                assert "not supported" in str(e) or "Unsupported" in str(e)
    
    def test_edge_cases(self):
        """Test edge cases for BBW."""
        # Very small period
        result = my_project.bollinger_bands_width(
            self.close[:10],
            period=2,
            devup=2.0,
            devdn=2.0
        )
        assert len(result) == 10
        assert not np.isnan(result[1])  # Should have value after period-1
        
        # Large deviations
        result = my_project.bollinger_bands_width(
            self.close[:50],
            period=10,
            devup=5.0,
            devdn=5.0
        )
        assert len(result) == 50
        
        # Asymmetric deviations
        result = my_project.bollinger_bands_width(
            self.close[:50],
            period=10,
            devup=3.0,
            devdn=1.0
        )
        assert len(result) == 50


if __name__ == "__main__":
    pytest.main([__file__])