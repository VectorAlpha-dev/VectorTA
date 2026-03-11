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


class TestGoertzelCycleCompositeWave:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_output_contract(self, test_data):
        close = test_data["close"][:320]
        values = mp.goertzel_cycle_composite_wave(close, 32, 1, 2)

        assert len(values) == len(close)
        assert int(np.flatnonzero(~np.isnan(values))[0]) == 160

    def test_invalid_params(self, test_data):
        close = test_data["close"][:128]
        with pytest.raises(ValueError, match="Invalid parameter max_period"):
            mp.goertzel_cycle_composite_wave(close, 1, 1, 2)

    def test_stream_matches_batch(self, test_data):
        close = test_data["close"][:420]
        batch = mp.goertzel_cycle_composite_wave(close, 32, 1, 2)

        stream = mp.GoertzelCycleCompositeWaveStream(32, 1, 2)
        stream_values = []
        for value in close:
            out = stream.update(float(value))
            stream_values.append(np.nan if out is None else out)

        np.testing.assert_allclose(
            stream_values, batch, rtol=1e-12, atol=1e-12, equal_nan=True
        )

    def test_batch_single_param_matches_single(self, test_data):
        close = test_data["close"][:360]
        batch = mp.goertzel_cycle_composite_wave_batch(
            close,
            max_period_range=(32, 32, 0),
            start_at_cycle_range=(1, 1, 0),
            use_top_cycles_range=(2, 2, 0),
        )
        single = mp.goertzel_cycle_composite_wave(close, 32, 1, 2)

        assert batch["rows"] == 1
        assert batch["cols"] == len(close)
        np.testing.assert_allclose(
            batch["values"][0], single, rtol=1e-12, atol=1e-12, equal_nan=True
        )

    def test_batch_metadata_multiple_ranges(self, test_data):
        close = test_data["close"][:360]
        batch = mp.goertzel_cycle_composite_wave_batch(
            close,
            max_period_range=(24, 28, 2),
            start_at_cycle_range=(1, 1, 0),
            use_top_cycles_range=(2, 2, 0),
        )

        assert batch["rows"] == 3
        assert batch["cols"] == len(close)
        np.testing.assert_array_equal(
            batch["max_periods"], np.array([24, 26, 28], dtype=np.uint64)
        )
