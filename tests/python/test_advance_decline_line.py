import numpy as np
import pytest

from test_utils import load_test_data

try:
    import my_project as mp
except ImportError:
    try:
        import vector_ta as mp
    except ImportError:
        pytest.skip(
            "Python module not built. Run 'maturin develop --features python' first",
            allow_module_level=True,
        )


class TestAdvanceDeclineLine:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_exact_cumulative_behavior(self):
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        values = mp.advance_decline_line(data)
        np.testing.assert_allclose(values, np.array([1.0, 3.0, 6.0, 10.0]))

    def test_output_contract(self, test_data):
        close = test_data["close"][:256]
        values = mp.advance_decline_line(close)
        assert len(values) == len(close)
        assert np.isfinite(values).all()

    def test_kernel_parity(self, test_data):
        close = test_data["close"][:256]
        auto = mp.advance_decline_line(close)
        scalar = mp.advance_decline_line(close, kernel="scalar")
        np.testing.assert_allclose(auto, scalar, rtol=1e-10, atol=1e-10, equal_nan=True)

    def test_all_nan_rejected(self):
        data = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
        with pytest.raises(ValueError, match="All values are NaN"):
            mp.advance_decline_line(data)

    def test_streaming_matches_batch_with_reset(self):
        data = np.array([1.0, 2.0, np.nan, 3.0, 4.0, 5.0], dtype=np.float64)
        batch = mp.advance_decline_line(data)
        stream = mp.AdvanceDeclineLineStream()
        streamed = []
        for value in data:
            out = stream.update(value)
            streamed.append(np.nan if out is None else out)
        np.testing.assert_allclose(
            streamed, batch, rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_batch_single_row_matches_single(self, test_data):
        close = test_data["close"][:128]
        batch = mp.advance_decline_line_batch(close)
        single = mp.advance_decline_line(close)

        assert batch["rows"] == 1
        assert batch["cols"] == len(close)
        assert batch["values"].shape == (1, len(close))
        np.testing.assert_allclose(
            batch["values"][0], single, rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_invalid_window_recovers(self, test_data):
        close = test_data["close"][:96].copy()
        close[30] = np.nan
        values = mp.advance_decline_line(close)
        assert np.isnan(values[30])
        assert values[31] == close[31]
        assert values[32] == close[31] + close[32]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
