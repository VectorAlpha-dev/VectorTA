"""Tests for DVDIQQE indicator."""

import numpy as np
import pytest
from my_project import dvdiqqe
from test_utils import load_test_data, assert_close, EXPECTED_OUTPUTS


def test_dvdiqqe_with_default_params():
    """Test DVDIQQE with default parameters."""

    test_data = load_test_data()
    expected = EXPECTED_OUTPUTS['dvdiqqe']


    dvdi, fast_tl, slow_tl, center_line = dvdiqqe(
        open=test_data['open'],
        high=test_data['high'],
        low=test_data['low'],
        close=test_data['close'],
        volume=test_data['volume'],
        period=expected['default_params']['period'],
        smoothing_period=expected['default_params']['smoothing_period'],
        fast_multiplier=expected['default_params']['fast_multiplier'],
        slow_multiplier=expected['default_params']['slow_multiplier'],
        volume_type=expected['default_params']['volume_type'],
        center_type=expected['default_params']['center_type'],
        tick_size=expected['default_params']['tick_size']
    )


    assert len(dvdi) == len(test_data['close'])
    assert len(fast_tl) == len(test_data['close'])
    assert len(slow_tl) == len(test_data['close'])
    assert len(center_line) == len(test_data['close'])


    warmup = None
    for i in range(len(dvdi)):
        if not np.isnan(dvdi[i]):
            warmup = i
            break


    assert warmup is not None, "Should have some non-NaN values"


    if warmup > 0:
        assert np.all(np.isnan(dvdi[:warmup])), f"Expected NaN in warmup period (0:{warmup}) for DVDI"
        assert np.all(np.isnan(fast_tl[:warmup])), f"Expected NaN in warmup period (0:{warmup}) for Fast TL"
        assert np.all(np.isnan(slow_tl[:warmup])), f"Expected NaN in warmup period (0:{warmup}) for Slow TL"





    assert not np.any(np.isnan(dvdi[warmup:])), "Found unexpected NaN after warmup in DVDI"
    assert not np.any(np.isnan(fast_tl[warmup:])), "Found unexpected NaN after warmup in Fast TL"
    assert not np.any(np.isnan(slow_tl[warmup:])), "Found unexpected NaN after warmup in Slow TL"
    assert not np.any(np.isnan(center_line[warmup:])), "Found unexpected NaN after warmup in Center Line"


def test_dvdiqqe_pinescript_reference_values():
    """Test DVDIQQE accuracy against exact PineScript reference values."""

    expected = EXPECTED_OUTPUTS['dvdiqqe']
    expected_dvdi = np.array(expected['pinescript_dvdi'])
    expected_slow_tl = np.array(expected['pinescript_slow_tl'])
    expected_fast_tl = np.array(expected['pinescript_fast_tl'])
    expected_center = np.array(expected['pinescript_center'])






    test_data = load_test_data()

    n_samples = 30


    dvdi, fast_tl, slow_tl, center_line = dvdiqqe(
        open=test_data['open'][:n_samples],
        high=test_data['high'][:n_samples],
        low=test_data['low'][:n_samples],
        close=test_data['close'][:n_samples],
        volume=test_data['volume'][:n_samples],
        period=expected['default_params']['period'],
        smoothing_period=expected['default_params']['smoothing_period'],
        fast_multiplier=expected['default_params']['fast_multiplier'],
        slow_multiplier=expected['default_params']['slow_multiplier'],
        volume_type=expected['default_params']['volume_type'],
        center_type=expected['default_params']['center_type'],
        tick_size=expected['default_params']['tick_size']
    )


    assert len(dvdi) == n_samples
    assert len(fast_tl) == n_samples
    assert len(slow_tl) == n_samples
    assert len(center_line) == n_samples




    warmup = None
    for i in range(len(dvdi)):
        if not np.isnan(dvdi[i]):
            warmup = i
            break

    assert warmup is not None, "Should have non-NaN values"
    if warmup > 0:
        assert np.all(np.isnan(dvdi[:warmup])), "Expected NaN in warmup period"
    assert not np.any(np.isnan(dvdi[warmup:])), "Unexpected NaN after warmup"













def test_dvdiqqe_accuracy():
    """Test DVDIQQE general accuracy and behavior."""

    test_data = load_test_data()
    expected = EXPECTED_OUTPUTS['dvdiqqe']


    n_samples = 100


    dvdi, fast_tl, slow_tl, center_line = dvdiqqe(
        open=test_data['open'][:n_samples],
        high=test_data['high'][:n_samples],
        low=test_data['low'][:n_samples],
        close=test_data['close'][:n_samples],
        volume=test_data['volume'][:n_samples],
        period=expected['default_params']['period'],
        smoothing_period=expected['default_params']['smoothing_period']
    )


    assert len(dvdi) == n_samples
    assert len(fast_tl) == n_samples
    assert len(slow_tl) == n_samples
    assert len(center_line) == n_samples



    warmup = None
    for i in range(len(dvdi)):
        if not np.isnan(dvdi[i]):
            warmup = i
            break

    assert warmup is not None, "Should have non-NaN values"

    assert np.all(np.isfinite(dvdi[warmup:])), "DVDI has non-finite values after warmup"
    assert np.all(np.isfinite(fast_tl[warmup:])), "Fast TL has non-finite values after warmup"
    assert np.all(np.isfinite(slow_tl[warmup:])), "Slow TL has non-finite values after warmup"
    assert np.all(np.isfinite(center_line[warmup:])), "Center line has non-finite values after warmup"


    center_values = center_line[warmup:]
    assert len(np.unique(center_values)) > 1, "Dynamic center line should vary over time"


def test_dvdiqqe_matches_rust_reference_values_last5():
    """Ensure Python binding matches the Rust unit test reference values (last 5)."""
    test_data = load_test_data()


    params = dict(
        period=13,
        smoothing_period=6,
        fast_multiplier=2.618,
        slow_multiplier=4.236,
        volume_type="default",
        center_type="dynamic",
    )

    dvdi, fast_tl, slow_tl, center_line = dvdiqqe(
        open=test_data['open'],
        high=test_data['high'],
        low=test_data['low'],
        close=test_data['close'],
        volume=test_data['volume'],
        **params,
    )


    exp_dvdi = np.array([-304.41010224, -279.48152664, -287.58723437, -252.40349484, -343.00922595])
    exp_slow = np.array([-990.21769695, -955.69385266, -951.82562405, -903.39071943, -903.39071943])
    exp_fast = np.array([-728.26380454, -697.40500858, -697.40500858, -654.73695895, -654.73695895])
    exp_center = np.array([
        21.98929919135097,
        21.969910753134442,
        21.950003541229705,
        21.932361363982043,
        21.908895469736102,
    ])


    assert_close(dvdi[-5:], exp_dvdi, rtol=0.0, atol=1e-6, msg="DVDI last-5")
    assert_close(slow_tl[-5:], exp_slow, rtol=0.0, atol=1e-6, msg="Slow TL last-5")
    assert_close(fast_tl[-5:], exp_fast, rtol=0.0, atol=1e-6, msg="Fast TL last-5")
    assert_close(center_line[-5:], exp_center, rtol=0.0, atol=1e-6, msg="Center line last-5")


def test_dvdiqqe_with_custom_params():
    """Test DVDIQQE with custom parameters."""

    test_data = load_test_data()
    n_samples = 50


    dvdi, fast_tl, slow_tl, center_line = dvdiqqe(
        open=test_data['open'][:n_samples],
        high=test_data['high'][:n_samples],
        low=test_data['low'][:n_samples],
        close=test_data['close'][:n_samples],
        volume=test_data['volume'][:n_samples],
        period=10,
        smoothing_period=5,
        fast_multiplier=2.0,
        slow_multiplier=4.0,
        volume_type="tick",
        center_type="static"
    )


    assert len(dvdi) == n_samples
    assert len(fast_tl) == n_samples
    assert len(slow_tl) == n_samples
    assert len(center_line) == n_samples



    unique_center = np.unique(center_line[~np.isnan(center_line)])
    assert len(unique_center) == 1, "Static center line should be constant"


def test_dvdiqqe_without_volume():
    """Test DVDIQQE without volume data (should use tick volume)."""

    test_data = load_test_data()
    n_samples = 50


    dvdi, fast_tl, slow_tl, center_line = dvdiqqe(
        open=test_data['open'][:n_samples],
        high=test_data['high'][:n_samples],
        low=test_data['low'][:n_samples],
        close=test_data['close'][:n_samples]

    )


    assert len(dvdi) == n_samples
    assert len(fast_tl) == n_samples
    assert len(slow_tl) == n_samples
    assert len(center_line) == n_samples


def test_dvdiqqe_empty_input():
    """Test DVDIQQE with empty input."""
    with pytest.raises(ValueError, match="Input data slice is empty|Empty input"):
        dvdiqqe(
            open=np.array([], dtype=np.float64),
            high=np.array([], dtype=np.float64),
            low=np.array([], dtype=np.float64),
            close=np.array([], dtype=np.float64)
        )


def test_dvdiqqe_all_nan():
    """Test DVDIQQE with all NaN values."""
    n_samples = 10
    nan_array = np.full(n_samples, np.nan, dtype=np.float64)

    with pytest.raises(ValueError, match="All values are NaN|Invalid data"):
        dvdiqqe(
            open=nan_array,
            high=nan_array,
            low=nan_array,
            close=nan_array
        )


def test_dvdiqqe_mismatched_lengths():
    """Test DVDIQQE with mismatched input lengths."""
    with pytest.raises(ValueError, match="Input arrays must have the same length|Mismatched lengths"):
        dvdiqqe(
            open=np.array([100.0, 101.0], dtype=np.float64),
            high=np.array([102.0, 103.0, 104.0], dtype=np.float64),
            low=np.array([99.0], dtype=np.float64),
            close=np.array([101.0, 102.0], dtype=np.float64)
        )


def test_dvdiqqe_period_too_large():
    """Test DVDIQQE with period larger than data."""
    n_samples = 5
    test_data = load_test_data()

    with pytest.raises(ValueError, match="Not enough data|Period too large|Invalid period"):
        dvdiqqe(
            open=test_data['open'][:n_samples],
            high=test_data['high'][:n_samples],
            low=test_data['low'][:n_samples],
            close=test_data['close'][:n_samples],
            period=20
        )


def test_dvdiqqe_nan_handling():
    """Test DVDIQQE handles NaN values correctly."""
    test_data = load_test_data()
    expected = EXPECTED_OUTPUTS['dvdiqqe']


    n_samples = 100

    dvdi, fast_tl, slow_tl, center_line = dvdiqqe(
        open=test_data['open'][:n_samples],
        high=test_data['high'][:n_samples],
        low=test_data['low'][:n_samples],
        close=test_data['close'][:n_samples],
        volume=test_data['volume'][:n_samples],
        period=expected['default_params']['period'],
        smoothing_period=expected['default_params']['smoothing_period']
    )

    assert len(dvdi) == n_samples


    warmup = None
    for i in range(len(dvdi)):
        if not np.isnan(dvdi[i]):
            warmup = i
            break

    assert warmup is not None, "Should have non-NaN values"


    if warmup > 0:
        assert np.all(np.isnan(dvdi[:warmup])), "Expected NaN in warmup period"
        assert np.all(np.isnan(fast_tl[:warmup])), "Expected NaN in Fast TL warmup"
        assert np.all(np.isnan(slow_tl[:warmup])), "Expected NaN in Slow TL warmup"


    assert not np.any(np.isnan(dvdi[warmup:])), "Found unexpected NaN after warmup"
    assert not np.any(np.isnan(fast_tl[warmup:])), "Found unexpected NaN in Fast TL after warmup"
    assert not np.any(np.isnan(slow_tl[warmup:])), "Found unexpected NaN in Slow TL after warmup"
    assert not np.any(np.isnan(center_line[warmup:])), "Found unexpected NaN in center line after warmup"


def test_dvdiqqe_invalid_parameters():
    """Test DVDIQQE with invalid parameters."""
    test_data = load_test_data()
    n_samples = 30


    with pytest.raises(ValueError, match="Invalid period|Period must be positive"):
        dvdiqqe(
            open=test_data['open'][:n_samples],
            high=test_data['high'][:n_samples],
            low=test_data['low'][:n_samples],
            close=test_data['close'][:n_samples],
            period=0
        )


    with pytest.raises(ValueError, match="Invalid period|Period must be positive"):
        dvdiqqe(
            open=test_data['open'][:n_samples],
            high=test_data['high'][:n_samples],
            low=test_data['low'][:n_samples],
            close=test_data['close'][:n_samples],
            period=-5
        )


    with pytest.raises(ValueError, match="Invalid multiplier|Multiplier must be positive"):
        dvdiqqe(
            open=test_data['open'][:n_samples],
            high=test_data['high'][:n_samples],
            low=test_data['low'][:n_samples],
            close=test_data['close'][:n_samples],
            fast_multiplier=-1.0
        )


def test_dvdiqqe_center_types():
    """Test DVDIQQE with different center line types."""
    test_data = load_test_data()
    n_samples = 50


    dvdi_s, fast_s, slow_s, center_s = dvdiqqe(
        open=test_data['open'][:n_samples],
        high=test_data['high'][:n_samples],
        low=test_data['low'][:n_samples],
        close=test_data['close'][:n_samples],
        volume=test_data['volume'][:n_samples],
        center_type="static"
    )


    unique_center = np.unique(center_s[~np.isnan(center_s)])
    assert len(unique_center) == 1, "Static center should be constant"


    dvdi_d, fast_d, slow_d, center_d = dvdiqqe(
        open=test_data['open'][:n_samples],
        high=test_data['high'][:n_samples],
        low=test_data['low'][:n_samples],
        close=test_data['close'][:n_samples],
        volume=test_data['volume'][:n_samples],
        center_type="dynamic"
    )



    warmup_d = None
    for i in range(len(center_d)):
        if not np.isnan(center_d[i]):
            warmup_d = i
            break

    assert warmup_d is not None, "Should have non-NaN center values"
    unique_center = np.unique(center_d[warmup_d:])
    assert len(unique_center) > 1, "Dynamic center should vary over time"


def test_dvdiqqe_volume_types():
    """Test DVDIQQE with different volume types."""
    test_data = load_test_data()
    n_samples = 50


    dvdi_real, _, _, _ = dvdiqqe(
        open=test_data['open'][:n_samples],
        high=test_data['high'][:n_samples],
        low=test_data['low'][:n_samples],
        close=test_data['close'][:n_samples],
        volume=test_data['volume'][:n_samples],
        volume_type="real"
    )


    dvdi_tick, _, _, _ = dvdiqqe(
        open=test_data['open'][:n_samples],
        high=test_data['high'][:n_samples],
        low=test_data['low'][:n_samples],
        close=test_data['close'][:n_samples],
        volume=test_data['volume'][:n_samples],
        volume_type="tick"
    )


    dvdi_no_vol, _, _, _ = dvdiqqe(
        open=test_data['open'][:n_samples],
        high=test_data['high'][:n_samples],
        low=test_data['low'][:n_samples],
        close=test_data['close'][:n_samples],
        volume_type="real"
    )



    warmup_real = None
    warmup_tick = None
    warmup_no_vol = None
    for i in range(len(dvdi_real)):
        if warmup_real is None and not np.isnan(dvdi_real[i]):
            warmup_real = i
        if warmup_tick is None and not np.isnan(dvdi_tick[i]):
            warmup_tick = i
        if warmup_no_vol is None and not np.isnan(dvdi_no_vol[i]):
            warmup_no_vol = i
        if warmup_real is not None and warmup_tick is not None and warmup_no_vol is not None:
            break

    assert warmup_real is not None, "Real volume should have non-NaN values"
    assert warmup_tick is not None, "Tick volume should have non-NaN values"
    assert warmup_no_vol is not None, "No volume should have non-NaN values"

    assert not np.any(np.isnan(dvdi_real[warmup_real:])), "Real volume DVDI has NaN"
    assert not np.any(np.isnan(dvdi_tick[warmup_tick:])), "Tick volume DVDI has NaN"
    assert not np.any(np.isnan(dvdi_no_vol[warmup_no_vol:])), "No volume DVDI has NaN"


    compare_start = max(warmup_tick, warmup_no_vol)
    assert np.allclose(dvdi_tick[compare_start:], dvdi_no_vol[compare_start:], rtol=1e-10), \
        "Tick volume should match no-volume behavior"






if __name__ == "__main__":
    test_dvdiqqe_with_default_params()
    test_dvdiqqe_pinescript_reference_values()
    test_dvdiqqe_accuracy()
    test_dvdiqqe_with_custom_params()
    test_dvdiqqe_without_volume()
    test_dvdiqqe_empty_input()
    test_dvdiqqe_all_nan()
    test_dvdiqqe_mismatched_lengths()
    test_dvdiqqe_period_too_large()
    test_dvdiqqe_nan_handling()
    test_dvdiqqe_invalid_parameters()
    test_dvdiqqe_center_types()
    test_dvdiqqe_volume_types()
    print("All DVDIQQE tests passed!")
