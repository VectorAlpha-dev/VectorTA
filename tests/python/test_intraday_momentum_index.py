import numpy as np
import pytest

from test_utils import load_test_data

try:
    import vector_ta as mp
except ImportError:
    try:
        import vector_ta as mp
    except ImportError:
        pytest.skip(
            "Python module not built. Run 'maturin develop --features python' first",
            allow_module_level=True,
        )


class TestIntradayMomentumIndex:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        open_ = test_data["open"]
        close = test_data["close"]
        imi, upper_hit, lower_hit, signal = mp.intraday_momentum_index(
            open_,
            close,
            length=14,
            length_ma=6,
            mult=2.0,
            length_bb=20,
            apply_smoothing=True,
            low_band=10,
        )

        assert len(imi) == len(close)
        assert len(upper_hit) == len(close)
        assert len(lower_hit) == len(close)
        assert len(signal) == len(close)
        assert np.isfinite(imi).any()
        assert np.isfinite(signal).any()
        valid = imi[np.isfinite(imi)]
        assert np.all((valid >= 0.0) & (valid <= 100.0))

    def test_kernel_parity(self, test_data):
        open_ = test_data["open"]
        close = test_data["close"]
        auto = mp.intraday_momentum_index(
            open_,
            close,
            length=11,
            length_ma=5,
            mult=1.75,
            length_bb=16,
            apply_smoothing=True,
            low_band=9,
        )
        scalar = mp.intraday_momentum_index(
            open_,
            close,
            length=11,
            length_ma=5,
            mult=1.75,
            length_bb=16,
            apply_smoothing=True,
            low_band=9,
            kernel="scalar",
        )
        for a, b in zip(auto, scalar):
            np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10, equal_nan=True)

    def test_invalid_params(self, test_data):
        open_ = test_data["open"]
        close = test_data["close"]

        with pytest.raises(ValueError, match="Invalid length"):
            mp.intraday_momentum_index(open_, close, length=0)

        with pytest.raises(ValueError, match="Inconsistent slice lengths"):
            mp.intraday_momentum_index(open_[:-1], close)

    def test_stream_matches_batch(self, test_data):
        open_ = test_data["open"]
        close = test_data["close"]
        batch = mp.intraday_momentum_index(
            open_,
            close,
            length=14,
            length_ma=6,
            mult=2.0,
            length_bb=20,
            apply_smoothing=True,
            low_band=10,
        )

        stream = mp.IntradayMomentumIndexStream(14, 6, 2.0, 20, True, 10)
        out_imi = []
        out_upper = []
        out_lower = []
        out_signal = []
        for o, c in zip(open_, close):
            out = stream.update(o, c)
            if out is None:
                out_imi.append(np.nan)
                out_upper.append(np.nan)
                out_lower.append(np.nan)
                out_signal.append(np.nan)
            else:
                out_imi.append(out[0])
                out_upper.append(out[1])
                out_lower.append(out[2])
                out_signal.append(out[3])

        np.testing.assert_allclose(out_imi, batch[0], rtol=1e-10, atol=1e-10, equal_nan=True)
        np.testing.assert_allclose(out_upper, batch[1], rtol=1e-10, atol=1e-10, equal_nan=True)
        np.testing.assert_allclose(out_lower, batch[2], rtol=1e-10, atol=1e-10, equal_nan=True)
        np.testing.assert_allclose(out_signal, batch[3], rtol=1e-10, atol=1e-10, equal_nan=True)

    def test_batch_single_param_matches_single(self, test_data):
        open_ = test_data["open"][:256]
        close = test_data["close"][:256]
        batch = mp.intraday_momentum_index_batch(
            open_,
            close,
            length_range=(14, 14, 0),
            length_ma_range=(6, 6, 0),
            mult_range=(2.0, 2.0, 0.0),
            length_bb_range=(20, 20, 0),
            apply_smoothing=False,
            low_band_range=(10, 10, 0),
        )
        single = mp.intraday_momentum_index(
            open_,
            close,
            length=14,
            length_ma=6,
            mult=2.0,
            length_bb=20,
            apply_smoothing=False,
            low_band=10,
        )

        assert batch["rows"] == 1
        assert batch["cols"] == len(close)
        np.testing.assert_allclose(batch["imi"][0], single[0], rtol=1e-10, atol=1e-10, equal_nan=True)
        np.testing.assert_allclose(
            batch["upper_hit"][0], single[1], rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            batch["lower_hit"][0], single[2], rtol=1e-10, atol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            batch["signal"][0], single[3], rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_batch_metadata(self, test_data):
        open_ = test_data["open"][:192]
        close = test_data["close"][:192]
        batch = mp.intraday_momentum_index_batch(
            open_,
            close,
            length_range=(12, 14, 2),
            length_ma_range=(5, 6, 1),
            mult_range=(1.5, 1.5, 0.0),
            length_bb_range=(18, 18, 0),
            apply_smoothing=True,
            low_band_range=(9, 9, 0),
        )

        assert batch["rows"] == 4
        assert batch["cols"] == len(close)
        assert batch["imi"].shape == (4, len(close))
        assert batch["upper_hit"].shape == (4, len(close))
        assert batch["lower_hit"].shape == (4, len(close))
        assert batch["signal"].shape == (4, len(close))
        np.testing.assert_array_equal(batch["lengths"], np.array([12, 12, 14, 14], dtype=np.uint64))
        np.testing.assert_array_equal(batch["length_mas"], np.array([5, 6, 5, 6], dtype=np.uint64))
        np.testing.assert_allclose(batch["mults"], np.array([1.5, 1.5, 1.5, 1.5]))
        np.testing.assert_array_equal(batch["length_bbs"], np.array([18, 18, 18, 18], dtype=np.uint64))
        assert batch["apply_smoothing"] == [True, True, True, True]
        np.testing.assert_array_equal(batch["low_bands"], np.array([9, 9, 9, 9], dtype=np.uint64))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
