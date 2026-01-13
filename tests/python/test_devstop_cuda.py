"""
CUDA bindings tests for the DevStop indicator.
Follows ALMA-style API: batch_dev returns (DeviceArrayF32Py, meta dict),
many_series_one_param_dev returns DeviceArrayF32Py.
"""
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
    if not hasattr(ti, "devstop_cuda_batch_dev"):
        return False

    try:
        h = np.array([np.nan, 2.0, 3.0], dtype=np.float32)
        l = np.array([np.nan, 1.0, 2.1], dtype=np.float32)
        handle, meta = ti.devstop_cuda_batch_dev(
            h, l,
            period_range=(2, 2, 0),
            mult_range=(1.0, 1.0, 0.0),
            devtype_range=(0, 0, 0),
            direction="long",
        )
        _ = cp.asarray(handle)
        return True
    except Exception as exc:
        msg = str(exc).lower()
        if "cuda not available" in msg or "ptx" in msg or "driver" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or devstop CUDA bindings missing")
class TestDevStopCuda:
    @pytest.fixture(scope="class")
    def data(self):
        return load_test_data()

    def test_devstop_cuda_batch_matches_cpu(self, data):
        high = data["high"].astype(np.float64)
        low = data["low"].astype(np.float64)
        period = 20
        mult = 1.5


        cpu = ti.devstop(high.astype(np.float32).astype(np.float64),
                         low.astype(np.float32).astype(np.float64),
                         period, mult, 0, "long", "sma")

        handle, meta = ti.devstop_cuda_batch_dev(
            high.astype(np.float32),
            low.astype(np.float32),
            period_range=(period, period, 0),
            mult_range=(mult, mult, 0.0),
            devtype_range=(0, 0, 0),
            direction="long",
        )
        assert meta["periods"].tolist() == [period]
        assert abs(meta["mults"].tolist()[0] - mult) < 1e-6

        gpu = cp.asnumpy(cp.asarray(handle))[0]

        assert_close(gpu, cpu, rtol=1e-4, atol=2e-3, msg="DevStop CUDA batch vs CPU mismatch")

    def test_devstop_cuda_many_series_one_param_matches_cpu(self, data):
        T = 2048
        N = 4
        h = data["high"][:T].astype(np.float64)
        l = data["low"][:T].astype(np.float64)
        high_tm = np.zeros((T, N), dtype=np.float64)
        low_tm = np.zeros((T, N), dtype=np.float64)
        for j in range(N):

            high_tm[:, j] = h * (1.0 + 0.005 * j)
            low_tm[:, j] = l * (1.0 + 0.005 * j)

        period = 20
        mult = 1.0

        cpu_tm = np.zeros_like(high_tm)
        for j in range(N):
            cpu_tm[:, j] = ti.devstop(
                high_tm[:, j].astype(np.float32).astype(np.float64),
                low_tm[:, j].astype(np.float32).astype(np.float64),
                period, mult, 0, "long", "sma",
            )

        handle = ti.devstop_cuda_many_series_one_param_dev(
            high_tm.astype(np.float32),
            low_tm.astype(np.float32),
            period,
            float(mult),
            "long",
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle))

        assert gpu_tm.shape == cpu_tm.shape
        assert_close(gpu_tm, cpu_tm, rtol=1e-4, atol=2e-3, msg="DevStop CUDA many-series vs CPU mismatch")

