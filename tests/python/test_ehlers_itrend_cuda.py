"""
Python binding tests for Ehlers ITrend CUDA kernels.
Mirrors CPU coverage to ensure the FP32 device implementation matches the
reference scalar output within a small tolerance.
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
    if not hasattr(ti, "ehlers_itrend_cuda_batch_dev"):
        return False
    try:
        data = np.array([np.nan, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        handle = ti.ehlers_itrend_cuda_batch_dev(
            data.astype(np.float32),
            warmup_range=(4, 4, 0),
            max_dc_range=(20, 20, 0),
        )
        _ = cp.asarray(handle)
        return True
    except Exception as exc:
        msg = str(exc).lower()
        if "cuda not available" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestEhlersItrendCuda:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_ehlers_itrend_cuda_batch_matches_cpu(self, test_data):
        close = test_data["close"].astype(np.float64)
        warmup = 12
        max_dc = 50

        cpu = ti.ehlers_itrend(close, warmup, max_dc)

        handle = ti.ehlers_itrend_cuda_batch_dev(
            close.astype(np.float32),
            warmup_range=(warmup, warmup, 0),
            max_dc_range=(max_dc, max_dc, 0),
        )
        gpu = cp.asnumpy(cp.asarray(handle))[0]

        assert_close(gpu, cpu, rtol=1e-4, atol=1e-4, msg="CUDA batch vs CPU mismatch")

    def test_ehlers_itrend_cuda_many_series_one_param_matches_cpu(self, test_data):
        T = 1024
        N = 3
        base_series = test_data["close"][:T].astype(np.float64)
        data_tm = np.zeros((T, N), dtype=np.float64)
        for j in range(N):
            data_tm[:, j] = base_series * (1.0 + 0.02 * j) + j * 0.1

        warmup = 10
        max_dc = 45

        cpu_tm = np.zeros_like(data_tm)
        for j in range(N):
            cpu_tm[:, j] = ti.ehlers_itrend(data_tm[:, j], warmup, max_dc)

        handle = ti.ehlers_itrend_cuda_many_series_one_param_dev(
            data_tm.astype(np.float32), warmup, max_dc
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle))

        assert gpu_tm.shape == data_tm.shape
        assert_close(gpu_tm, cpu_tm, rtol=1e-4, atol=1e-4, msg="CUDA many-series vs CPU mismatch")
