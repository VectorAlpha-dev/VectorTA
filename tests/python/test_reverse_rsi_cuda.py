"""CUDA bindings tests for Reverse RSI indicator."""
import numpy as np
import pytest

try:
    import cupy as cp
except ImportError:  # pragma: no cover - optional dependency
    cp = None

try:
    import my_project as ti
except ImportError:
    pytest.skip(
        "Python module not built. Run 'maturin develop --features python,cuda' first",
        allow_module_level=True,
    )

from test_utils import load_test_data, assert_close


def _cuda_available() -> bool:
    if cp is None:
        return False
    if not hasattr(ti, "reverse_rsi_cuda_batch_dev"):
        return False
    try:
        arr = np.array([np.nan, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        handle, meta = ti.reverse_rsi_cuda_batch_dev(
            arr, rsi_length_range=(3, 3, 0), rsi_level_range=(50.0, 50.0, 0.0)
        )
        _ = cp.asarray(handle)
        assert meta["rsi_lengths"][0] == 3
        return True
    except Exception as exc:  # pragma: no cover - probing path
        msg = str(exc).lower()
        if "cuda not available" in msg or "ptx" in msg or "driver" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings missing")
class TestReverseRsiCuda:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_reverse_rsi_cuda_batch_matches_cpu(self, test_data):
        close = test_data["close"][:4096].astype(np.float64)
        rlen_range = (7, 21, 7)
        lvl_range = (30.0, 70.0, 20.0)

        # CPU baseline aligned to FP32 input (to match GPU input semantics)
        cpu = ti.reverse_rsi_batch(
            close.astype(np.float32).astype(np.float64), rlen_range, lvl_range
        )
        cpu_vals = cpu["values"].astype(np.float32)

        handle, meta = ti.reverse_rsi_cuda_batch_dev(
            close.astype(np.float32),
            rsi_length_range=rlen_range,
            rsi_level_range=lvl_range,
        )
        assert list(meta["rsi_lengths"]) == [7, 14, 21]
        assert list(meta["rsi_levels"]) == [30.0, 50.0, 70.0]

        gpu_vals = cp.asnumpy(cp.asarray(handle)).reshape(cpu_vals.shape)
        assert_close(gpu_vals, cpu_vals, rtol=8e-4, atol=8e-4, msg="CUDA ReverseRSI batch mismatch")

    def test_reverse_rsi_cuda_many_series_one_param_matches_cpu(self, test_data):
        T = 2048
        N = 4
        base = test_data["close"][:T].astype(np.float64)
        data_tm = np.zeros((T, N), dtype=np.float64)
        for j in range(N):
            data_tm[:, j] = base * (1.0 + 0.02 * j)

        rsi_length = 14
        rsi_level = 55.0

        # CPU baseline aligned to FP32 input
        cpu_tm = np.zeros_like(data_tm)
        for j in range(N):
            cpu_tm[:, j] = ti.reverse_rsi(
                data_tm[:, j].astype(np.float32).astype(np.float64), rsi_length, rsi_level
            )

        handle = ti.reverse_rsi_cuda_many_series_one_param_dev(
            data_tm.astype(np.float32), N, T, rsi_length, rsi_level
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle))

        assert gpu_tm.shape == data_tm.shape
        assert_close(gpu_tm, cpu_tm, rtol=8e-4, atol=8e-4, msg="CUDA ReverseRSI many-series mismatch")

