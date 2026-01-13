"""CUDA bindings tests for the EMV indicator."""
import pytest
import numpy as np

try:
    import cupy as cp
except ImportError:
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
    if not hasattr(ti, "emv_cuda_batch_dev"):
        return False

    try:
        h = np.array([np.nan, 2.0, 3.0], dtype=np.float32)
        l = np.array([np.nan, 1.0, 1.5], dtype=np.float32)
        v = np.array([np.nan, 1000.0, 800.0], dtype=np.float32)
        handle = ti.emv_cuda_batch_dev(h, l, v)
        _ = cp.asarray(handle)
        return True
    except Exception as exc:
        msg = str(exc).lower()
        if "cuda not available" in msg or "ptx" in msg or "driver" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings missing")
class TestEmvCuda:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_emv_cuda_batch_matches_cpu(self, test_data):
        high = test_data["high"].astype(np.float64)
        low = test_data["low"].astype(np.float64)
        close = test_data["close"].astype(np.float64)
        volume = test_data["volume"].astype(np.float64)


        cpu = ti.emv(
            high.astype(np.float32).astype(np.float64),
            low.astype(np.float32).astype(np.float64),
            close.astype(np.float32).astype(np.float64),
            volume.astype(np.float32).astype(np.float64),
        )

        handle = ti.emv_cuda_batch_dev(
            high.astype(np.float32),
            low.astype(np.float32),
            volume.astype(np.float32),
        )
        gpu = cp.asnumpy(cp.asarray(handle))[0]


        assert_close(gpu, cpu, rtol=1e-5, atol=1e-5, msg="CUDA EMV batch mismatch")

    def test_emv_cuda_many_series_one_param_matches_cpu(self, test_data):
        T = 2048
        N = 4
        base = test_data["close"][:T].astype(np.float64)
        data_tm = np.zeros((T, N), dtype=np.float64)
        vol_tm = np.zeros((T, N), dtype=np.float64)
        for j in range(N):
            data_tm[:, j] = base * (1.0 + 0.01 * j)
            vol_tm[:, j] = test_data["volume"][:T] * (1.0 + 0.05 * j)


        high_tm = data_tm + 0.15
        low_tm = data_tm - 0.15


        cpu_tm = np.zeros_like(data_tm)
        for j in range(N):
            cpu_tm[:, j] = ti.emv(
                high_tm[:, j].astype(np.float32).astype(np.float64),
                low_tm[:, j].astype(np.float32).astype(np.float64),
                data_tm[:, j].astype(np.float32).astype(np.float64),
                vol_tm[:, j].astype(np.float32).astype(np.float64),
            )

        handle = ti.emv_cuda_many_series_one_param_dev(
            high_tm.astype(np.float32),
            low_tm.astype(np.float32),
            vol_tm.astype(np.float32),
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle))

        assert gpu_tm.shape == data_tm.shape
        assert_close(gpu_tm, cpu_tm, rtol=1e-5, atol=1e-5, msg="CUDA EMV many-series mismatch")

