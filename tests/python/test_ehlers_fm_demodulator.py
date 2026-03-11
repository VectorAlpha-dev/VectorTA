import math

import numpy as np
import pytest

try:
    import my_project as ta_indicators
except ImportError:
    pytest.skip(
        "Python module not built. Run 'maturin develop --features python' first",
        allow_module_level=True,
    )


def reference_fm_demodulator(open_, close, period):
    open_ = np.asarray(open_, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    if open_.size == 0 or close.size == 0:
        raise ValueError("empty")
    if open_.shape != close.shape:
        raise ValueError("length mismatch")
    if period == 0 or period > open_.size:
        raise ValueError("Invalid period")

    valid_idx = np.flatnonzero(~np.isnan(open_) & ~np.isnan(close))
    if len(valid_idx) == 0:
        raise ValueError("All values are NaN")
    first = int(valid_idx[0])
    needed = max(1, period - 2)
    if open_.size - first < needed:
        raise ValueError("Not enough valid data")

    out = np.full(open_.shape, np.nan, dtype=np.float64)
    a1 = math.exp(-1.414 * math.pi / period)
    b1 = 2.0 * a1 * math.cos(1.414 * math.pi / period)
    c2 = b1
    c3 = -(a1 * a1)
    c1 = 1.0 - c2 - c3
    prev_hl = 0.0
    ss1 = 0.0
    ss2 = 0.0
    valid_count = 0
    warmup_bars = max(0, period - 3)

    for i in range(first, open_.size):
        if np.isnan(open_[i]) or np.isnan(close[i]):
            prev_hl = 0.0
            ss1 = 0.0
            ss2 = 0.0
            valid_count = 0
            continue
        derivative = close[i] - open_[i]
        hl = min(1.0, max(-1.0, 10.0 * derivative))
        if valid_count < 3:
            value = derivative
        else:
            value = c1 * (hl + prev_hl) * 0.5 + c2 * ss1 + c3 * ss2
        prev_hl = hl
        ss2 = ss1
        ss1 = value
        valid_count += 1
        if valid_count > warmup_bars:
            out[i] = value

    return out


class TestEhlersFmDemodulator:
    @pytest.fixture(scope="class")
    def sample_ohlc(self):
        open_ = np.ones(8, dtype=np.float64)
        close = np.array([1.1, 0.9, 1.2, 0.8, 1.05, 0.95, 1.15, 0.85], dtype=np.float64)
        return open_, close

    def test_matches_local_reference(self, sample_ohlc):
        open_, close = sample_ohlc
        result = ta_indicators.ehlers_fm_demodulator(open_, close, period=5)
        expected = reference_fm_demodulator(open_, close, 5)
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12, equal_nan=True)

    def test_small_period_reference_case(self):
        open_ = np.ones(6, dtype=np.float64)
        close = np.array([1.1, 0.9, 1.2, 0.8, 1.05, 0.95], dtype=np.float64)
        result = ta_indicators.ehlers_fm_demodulator(open_, close, period=3)
        expected = np.array(
            [
                0.10000000000000009,
                -0.09999999999999998,
                0.19999999999999996,
                0.013357467332142794,
                -0.26250860145491506,
                -0.011431966667873657,
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12, equal_nan=True)

    def test_stream_matches_batch(self, sample_ohlc):
        open_, close = sample_ohlc
        batch = ta_indicators.ehlers_fm_demodulator(open_, close, period=5)
        stream = ta_indicators.EhlersFmDemodulatorStream(period=5)
        streamed = np.array(
            [np.nan if (v := stream.update(float(o), float(c))) is None else v for o, c in zip(open_, close)],
            dtype=np.float64,
        )
        np.testing.assert_allclose(streamed, batch, rtol=1e-12, atol=1e-12, equal_nan=True)

    def test_batch_matches_single(self, sample_ohlc):
        open_, close = sample_ohlc
        batch = ta_indicators.ehlers_fm_demodulator_batch(open_, close, period_range=(5, 7, 1))
        assert batch["values"].shape == (3, len(open_))
        assert list(batch["periods"]) == [5, 6, 7]
        for row, period in enumerate(batch["periods"]):
            single = ta_indicators.ehlers_fm_demodulator(open_, close, period=int(period))
            np.testing.assert_allclose(
                batch["values"][row], single, rtol=1e-12, atol=1e-12, equal_nan=True
            )

    def test_period_minus_three_warmup(self, sample_ohlc):
        open_, close = sample_ohlc
        result = ta_indicators.ehlers_fm_demodulator(open_, close, period=5)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert not np.isnan(result[2])

    def test_kernel_parameter_accepts_repo_values(self, sample_ohlc):
        open_, close = sample_ohlc
        baseline = ta_indicators.ehlers_fm_demodulator(open_, close, period=5, kernel="scalar")
        for kernel in [None, "auto", "scalar"]:
            result = ta_indicators.ehlers_fm_demodulator(open_, close, period=5, kernel=kernel)
            np.testing.assert_allclose(result, baseline, rtol=1e-12, atol=1e-12, equal_nan=True)

    def test_invalid_inputs(self, sample_ohlc):
        open_, close = sample_ohlc
        with pytest.raises(ValueError, match="invalid period"):
            ta_indicators.ehlers_fm_demodulator(open_, close, period=0)
        with pytest.raises(ValueError, match="length mismatch"):
            ta_indicators.ehlers_fm_demodulator(open_[:-1], close, period=5)
        with pytest.raises(ValueError, match="all values are NaN"):
            ta_indicators.ehlers_fm_demodulator(np.full(8, np.nan), np.full(8, np.nan), period=5)
