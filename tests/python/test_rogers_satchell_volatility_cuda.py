"""
Python binding tests for Rogers-Satchell CUDA paths.
"""
import pytest
import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    import vector_ta as mp
except ImportError:
    try:
        import my_project as mp
    except ImportError:
        pytest.skip(
            "Python module not built. Run 'maturin develop --features python,cuda' first",
            allow_module_level=True,
        )

from test_utils import load_test_data


def _cuda_available() -> bool:
    if cp is None:
        return False
    if not hasattr(mp, "rogers_satchell_volatility_cuda_batch_dev"):
        return False
    try:
        x = np.array([2.0, 2.1, 2.2, 2.3], dtype=np.float32)
        handle = mp.rogers_satchell_volatility_cuda_batch_dev(
            x,
            x + 0.1,
            x - 0.1,
            x + 0.05,
            lookback_range=(2, 2, 0),
            signal_length_range=(2, 2, 0),
        )
        _ = cp.asarray(handle["rs"])
        _ = cp.asarray(handle["signal"])
        return True
    except Exception as exc:
        msg = str(exc).lower()
        if "cuda not available" in msg or "nvcc" in msg or "ptx" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestRogersSatchellVolatilityCuda:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_cuda_batch_matches_cpu(self, test_data):
        open_ = test_data["open"].astype(np.float64)
        high = test_data["high"].astype(np.float64)
        low = test_data["low"].astype(np.float64)
        close = test_data["close"].astype(np.float64)
        lookback = 20
        signal_length = 6

        cpu_rs, cpu_signal = mp.rogers_satchell_volatility(
            open_, high, low, close, lookback=lookback, signal_length=signal_length
        )

        handle = mp.rogers_satchell_volatility_cuda_batch_dev(
            open_.astype(np.float32),
            high.astype(np.float32),
            low.astype(np.float32),
            close.astype(np.float32),
            lookback_range=(lookback, lookback, 0),
            signal_length_range=(signal_length, signal_length, 0),
        )

        gpu_rs = cp.asnumpy(cp.asarray(handle["rs"]))[0]
        gpu_signal = cp.asnumpy(cp.asarray(handle["signal"]))[0]

        np.testing.assert_allclose(
            gpu_rs, cpu_rs, rtol=1e-5, atol=1e-6, equal_nan=True
        )
        np.testing.assert_allclose(
            gpu_signal, cpu_signal, rtol=1e-5, atol=1e-6, equal_nan=True
        )

    def test_cuda_many_series_one_param_matches_cpu(self, test_data):
        rows = 1024
        cols = 4
        lookback = 14
        signal_length = 5

        open_base = test_data["open"][:rows].astype(np.float64)
        high_base = test_data["high"][:rows].astype(np.float64)
        low_base = test_data["low"][:rows].astype(np.float64)
        close_base = test_data["close"][:rows].astype(np.float64)

        open_tm = np.zeros((rows, cols), dtype=np.float64)
        high_tm = np.zeros((rows, cols), dtype=np.float64)
        low_tm = np.zeros((rows, cols), dtype=np.float64)
        close_tm = np.zeros((rows, cols), dtype=np.float64)
        for j in range(cols):
            factor = 1.0 + 0.01 * j
            open_tm[:, j] = open_base * factor
            high_tm[:, j] = high_base * factor
            low_tm[:, j] = low_base * factor
            close_tm[:, j] = close_base * factor

        cpu_rs = np.zeros_like(close_tm)
        cpu_signal = np.zeros_like(close_tm)
        for j in range(cols):
            open_col = np.ascontiguousarray(open_tm[:, j])
            high_col = np.ascontiguousarray(high_tm[:, j])
            low_col = np.ascontiguousarray(low_tm[:, j])
            close_col = np.ascontiguousarray(close_tm[:, j])
            rs, signal = mp.rogers_satchell_volatility(
                open_col,
                high_col,
                low_col,
                close_col,
                lookback=lookback,
                signal_length=signal_length,
            )
            cpu_rs[:, j] = rs
            cpu_signal[:, j] = signal

        result = mp.rogers_satchell_volatility_cuda_many_series_one_param_dev(
            open_tm.astype(np.float32),
            high_tm.astype(np.float32),
            low_tm.astype(np.float32),
            close_tm.astype(np.float32),
            lookback,
            signal_length,
        )
        gpu_rs = cp.asnumpy(cp.asarray(result["rs"]))
        gpu_signal = cp.asnumpy(cp.asarray(result["signal"]))

        assert gpu_rs.shape == close_tm.shape
        assert gpu_signal.shape == close_tm.shape
        np.testing.assert_allclose(
            gpu_rs, cpu_rs, rtol=1e-5, atol=1e-6, equal_nan=True
        )
        np.testing.assert_allclose(
            gpu_signal, cpu_signal, rtol=1e-5, atol=1e-6, equal_nan=True
        )
