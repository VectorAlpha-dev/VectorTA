import numpy as np
import pytest

from test_utils import load_test_data

try:
    import vector_ta as mp
except ImportError:
    try:
        import my_project as mp
    except ImportError:
        pytest.skip(
            "Python module not built. Run 'maturin develop --features python' first",
            allow_module_level=True,
        )


class TestEvasiveSuperTrend:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        open_ = test_data["open"][:256]
        high = test_data["high"][:256]
        low = test_data["low"][:256]
        close = test_data["close"][:256]

        band, state, noisy, changed = mp.evasive_supertrend(
            open_, high, low, close, 10, 3.0, 1.0, 0.5
        )

        assert len(band) == len(close)
        assert len(state) == len(close)
        assert len(noisy) == len(close)
        assert len(changed) == len(close)
        assert np.isfinite(band).any()

    def test_invalid_params(self, test_data):
        open_ = test_data["open"][:64]
        high = test_data["high"][:64]
        low = test_data["low"][:64]
        close = test_data["close"][:64]

        with pytest.raises(ValueError, match="Invalid atr_length"):
            mp.evasive_supertrend(open_, high, low, close, 0, 3.0, 1.0, 0.5)

        with pytest.raises(ValueError, match="Invalid base_multiplier"):
            mp.evasive_supertrend(open_, high, low, close, 10, 0.0, 1.0, 0.5)

        with pytest.raises(ValueError, match="Invalid noise_threshold"):
            mp.evasive_supertrend(open_, high, low, close, 10, 3.0, 0.0, 0.5)

        with pytest.raises(ValueError, match="Invalid expansion_alpha"):
            mp.evasive_supertrend(open_, high, low, close, 10, 3.0, 1.0, -0.1)

    def test_batch_single_param_matches_single(self, test_data):
        open_ = test_data["open"][:220]
        high = test_data["high"][:220]
        low = test_data["low"][:220]
        close = test_data["close"][:220]

        batch = mp.evasive_supertrend_batch(
            open_,
            high,
            low,
            close,
            atr_length_range=(10, 10, 0),
            base_multiplier_range=(3.0, 3.0, 0.0),
            noise_threshold_range=(1.0, 1.0, 0.0),
            expansion_alpha_range=(0.5, 0.5, 0.0),
        )
        single = mp.evasive_supertrend(open_, high, low, close, 10, 3.0, 1.0, 0.5)

        assert batch["rows"] == 1
        assert batch["cols"] == len(close)
        np.testing.assert_array_equal(batch["atr_lengths"], np.array([10], dtype=np.uint64))
        np.testing.assert_allclose(batch["base_multipliers"], np.array([3.0]))
        np.testing.assert_allclose(batch["noise_thresholds"], np.array([1.0]))
        np.testing.assert_allclose(batch["expansion_alphas"], np.array([0.5]))

        for key, expected in zip(("band", "state", "noisy", "changed"), single):
            np.testing.assert_allclose(
                batch[key][0], expected, rtol=1e-12, atol=1e-12, equal_nan=True
            )

    def test_stream_matches_batch(self, test_data):
        open_ = test_data["open"][:220]
        high = test_data["high"][:220]
        low = test_data["low"][:220]
        close = test_data["close"][:220]

        batch = mp.evasive_supertrend(open_, high, low, close, 10, 3.0, 1.0, 0.5)
        stream = mp.EvasiveSuperTrendStream(10, 3.0, 1.0, 0.5)

        streamed = [[], [], [], []]
        for o, h, l, c in zip(open_, high, low, close):
            point = stream.update(float(o), float(h), float(l), float(c))
            if point is None:
                for arr in streamed:
                    arr.append(np.nan)
            else:
                for arr, value in zip(streamed, point):
                    arr.append(value)

        for expected, got in zip(batch, streamed):
            np.testing.assert_allclose(expected, got, rtol=1e-12, atol=1e-12, equal_nan=True)

    def test_batch_multi_param_contract(self, test_data):
        open_ = test_data["open"][:180]
        high = test_data["high"][:180]
        low = test_data["low"][:180]
        close = test_data["close"][:180]

        batch = mp.evasive_supertrend_batch(
            open_,
            high,
            low,
            close,
            atr_length_range=(10, 12, 2),
            base_multiplier_range=(3.0, 3.5, 0.5),
            noise_threshold_range=(1.0, 1.0, 0.0),
            expansion_alpha_range=(0.5, 0.5, 0.0),
        )

        assert batch["rows"] == 4
        assert batch["cols"] == len(close)
        assert batch["band"].shape == (4, len(close))
