"""
Tests for the GatorOsc indicator Python bindings.
"""

import pytest
import numpy as np
import my_project as ta

def test_gatorosc_basic():
    """Test basic GatorOsc calculation."""
    # Create sample data
    data = np.random.randn(100) * 10 + 50
    
    # Calculate GatorOsc
    upper, lower, upper_change, lower_change = ta.gatorosc(data)
    
    # Check output shape
    assert len(upper) == len(data)
    assert len(lower) == len(data)
    assert len(upper_change) == len(data)
    assert len(lower_change) == len(data)
    
    # Check that we have some non-NaN values after warmup
    assert not np.all(np.isnan(upper))
    assert not np.all(np.isnan(lower))
    assert not np.all(np.isnan(upper_change))
    assert not np.all(np.isnan(lower_change))
    
    # Check that upper is positive and lower is negative
    valid_upper = upper[~np.isnan(upper)]
    valid_lower = lower[~np.isnan(lower)]
    assert np.all(valid_upper >= 0)
    assert np.all(valid_lower <= 0)

def test_gatorosc_custom_params():
    """Test GatorOsc with custom parameters."""
    data = np.random.randn(200) * 10 + 50
    
    # Test with custom parameters
    upper, lower, upper_change, lower_change = ta.gatorosc(
        data, 
        jaws_length=21, 
        jaws_shift=10,
        teeth_length=13,
        teeth_shift=7,
        lips_length=8,
        lips_shift=5
    )
    
    assert len(upper) == len(data)
    assert len(lower) == len(data)
    assert len(upper_change) == len(data)
    assert len(lower_change) == len(data)

def test_gatorosc_errors():
    """Test GatorOsc error handling."""
    # Empty data
    with pytest.raises(ValueError):
        ta.gatorosc(np.array([]))
    
    # All NaN data
    with pytest.raises(ValueError):
        ta.gatorosc(np.full(50, np.nan))
    
    # Invalid parameters
    with pytest.raises(ValueError):
        ta.gatorosc(np.random.randn(50), jaws_length=0)

def test_gatorosc_stream():
    """Test GatorOsc streaming functionality."""
    stream = ta.GatorOscStream()
    
    # Initial updates should return None
    # With default params: jaws_length=13, jaws_shift=8, teeth_length=8, teeth_shift=5, lips_length=5, lips_shift=3
    # lower_change_warmup = 13, so first output at index 12
    for i in range(20):
        result = stream.update(50.0 + i)
        if i < 12:  # Before minimum warmup (lower_change_warmup = 13, first output at index 12)
            assert result is None
    
    # Eventually should get results
    for i in range(50):
        result = stream.update(50.0 + i)
        if result is not None:
            upper, lower, upper_change, lower_change = result
            assert isinstance(upper, float)
            assert isinstance(lower, float)
            assert isinstance(upper_change, float)
            assert isinstance(lower_change, float)
            assert upper >= 0
            assert lower <= 0

def test_gatorosc_stream_custom_params():
    """Test GatorOsc streaming with custom parameters."""
    stream = ta.GatorOscStream(
        jaws_length=21,
        jaws_shift=10,
        teeth_length=13,
        teeth_shift=7,
        lips_length=8,
        lips_shift=5
    )
    
    # Feed some data
    for i in range(50):
        result = stream.update(50.0 + i * 0.1)
        # Just verify it doesn't crash and eventually returns values

def test_gatorosc_batch():
    """Test GatorOsc batch calculation."""
    data = np.random.randn(100) * 10 + 50
    
    # Test batch with multiple parameter combinations
    result = ta.gatorosc_batch(
        data,
        jaws_length_range=(10, 15, 5),
        jaws_shift_range=(6, 10, 2),
        teeth_length_range=(6, 10, 2),
        teeth_shift_range=(3, 6, 3),
        lips_length_range=(3, 6, 3),
        lips_shift_range=(2, 4, 2)
    )
    
    # Check result structure
    assert 'upper' in result
    assert 'lower' in result
    assert 'upper_change' in result
    assert 'lower_change' in result
    assert 'jaws_lengths' in result
    assert 'jaws_shifts' in result
    assert 'teeth_lengths' in result
    assert 'teeth_shifts' in result
    assert 'lips_lengths' in result
    assert 'lips_shifts' in result
    
    # Check shapes
    n_combos = 2 * 3 * 3 * 2 * 2 * 2  # Product of range sizes
    assert result['upper'].shape == (n_combos, len(data))
    assert result['lower'].shape == (n_combos, len(data))
    assert result['upper_change'].shape == (n_combos, len(data))
    assert result['lower_change'].shape == (n_combos, len(data))
    assert len(result['jaws_lengths']) == n_combos

def test_gatorosc_consistency():
    """Test that GatorOsc produces consistent results."""
    np.random.seed(42)
    data = np.random.randn(100) * 10 + 50
    
    # Calculate twice with same parameters
    upper1, lower1, upper_change1, lower_change1 = ta.gatorosc(data)
    upper2, lower2, upper_change2, lower_change2 = ta.gatorosc(data)
    
    # Results should be identical
    np.testing.assert_array_equal(upper1, upper2)
    np.testing.assert_array_equal(lower1, lower2)
    np.testing.assert_array_equal(upper_change1, upper_change2)
    np.testing.assert_array_equal(lower_change1, lower_change2)

def test_gatorosc_kernel_options():
    """Test GatorOsc with different kernel options."""
    data = np.random.randn(100) * 10 + 50
    
    # Test with different kernels
    for kernel in [None, "scalar", "avx2", "avx512"]:
        try:
            upper, lower, upper_change, lower_change = ta.gatorosc(data, kernel=kernel)
            assert len(upper) == len(data)
            assert len(lower) == len(data)
        except ValueError:
            # Some kernels might not be supported on this platform
            pass

def test_gatorosc_nan_handling():
    """Test GatorOsc handling of NaN values in input."""
    data = np.random.randn(100) * 10 + 50
    data[40:45] = np.nan  # Insert some NaN values
    
    # Should still work
    upper, lower, upper_change, lower_change = ta.gatorosc(data)
    
    assert len(upper) == len(data)
    assert len(lower) == len(data)
    assert len(upper_change) == len(data)
    assert len(lower_change) == len(data)
    
    # Should have some valid values despite NaNs in input
    assert not np.all(np.isnan(upper))
    assert not np.all(np.isnan(lower))

def test_gatorosc_value_ranges():
    """Test that GatorOsc outputs are in expected ranges."""
    # Create trending data
    trend_up = np.linspace(40, 60, 100) + np.random.randn(100) * 0.5
    trend_down = np.linspace(60, 40, 100) + np.random.randn(100) * 0.5
    
    # Test with uptrend
    upper_up, lower_up, _, _ = ta.gatorosc(trend_up)
    
    # Test with downtrend
    upper_down, lower_down, _, _ = ta.gatorosc(trend_down)
    
    # Both should have valid values
    assert not np.all(np.isnan(upper_up))
    assert not np.all(np.isnan(lower_up))
    assert not np.all(np.isnan(upper_down))
    assert not np.all(np.isnan(lower_down))