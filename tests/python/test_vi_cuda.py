"""CUDA bindings tests for the Vortex Indicator (VI)."""
import pytest
import numpy as np

try:
    import cupy as cp
except ImportError:  # pragma: no cover - optional dependency
    cp = None

try:
    import my_project as ti
except ImportError:  # pragma: no cover - module missing when Python build skipped
    pytest.skip(
        "Python module not built. Run 'maturin develop --features python,cuda' first",
        allow_module_level=True,
    )

from test_utils import load_test_data, assert_close


def _cuda_available() -> bool:
    if cp is None:
        return False
    if not hasattr(ti, "vi_cuda_batch_dev"):
        return False
    try:
        # Simple probe: tiny arrays with a valid prefix
        h = np.array([np.nan, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
        l = np.array([np.nan, 1.0, 2.0, 2.5, 3.0], dtype=np.float32)
        c = np.array([np.nan, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        dev = ti.vi_cuda_batch_dev(h, l, c, (3, 3, 0))
        _ = cp.asarray(dev["plus"])  # ensure CuPy can wrap the handle
        return True
    except Exception as exc:  # pragma: no cover - probing path
        msg = str(exc).lower()
        if "cuda not available" in msg or "ptx" in msg or "driver" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings missing")
class TestViCuda:
    @pytest.fixture(scope="class")
    def triplet(self):
        data = load_test_data()
        return (
            data["high"].astype(np.float64),
            data["low"].astype(np.float64),
            data["close"].astype(np.float64),
        )

    def test_vi_cuda_batch_matches_cpu(self, triplet):
        high, low, close = triplet
        period = 14

        cpu = ti.vi(high, low, close, period)

        dev = ti.vi_cuda_batch_dev(
            high.astype(np.float32),
            low.astype(np.float32),
            close.astype(np.float32),
            (period, period, 0),
        )
        plus_gpu = cp.asnumpy(cp.asarray(dev["plus"]))[0]
        minus_gpu = cp.asnumpy(cp.asarray(dev["minus"]))[0]

        assert_close(plus_gpu, cpu["plus"], rtol=5e-4, atol=5e-5, msg="VI+ batch mismatch")
        assert_close(minus_gpu, cpu["minus"], rtol=5e-4, atol=5e-5, msg="VI- batch mismatch")

    def test_vi_cuda_many_series_one_param_matches_cpu(self, triplet):
        high, low, close = triplet
        T = 2048
        N = 4
        h_tm = np.zeros((T, N), dtype=np.float64)
        l_tm = np.zeros((T, N), dtype=np.float64)
        c_tm = np.zeros((T, N), dtype=np.float64)
        for j in range(N):
            h_tm[:, j] = high[:T] * (1.0 + 0.01 * j)
            l_tm[:, j] = low[:T] * (1.0 + 0.01 * j)
            c_tm[:, j] = close[:T] * (1.0 + 0.01 * j)

        period = 10
        plus_cpu = np.zeros_like(h_tm)
        minus_cpu = np.zeros_like(h_tm)
        for j in range(N):
            out = ti.vi(h_tm[:, j], l_tm[:, j], c_tm[:, j], period)
            plus_cpu[:, j] = out["plus"]
            minus_cpu[:, j] = out["minus"]

        dev = ti.vi_cuda_many_series_one_param_dev(
            h_tm.astype(np.float32),
            l_tm.astype(np.float32),
            c_tm.astype(np.float32),
            period,
        )
        plus_gpu = cp.asnumpy(cp.asarray(dev["plus"]))
        minus_gpu = cp.asnumpy(cp.asarray(dev["minus"]))

        assert_close(plus_gpu, plus_cpu, rtol=5e-4, atol=5e-5, msg="VI+ many-series mismatch")
        assert_close(minus_gpu, minus_cpu, rtol=5e-4, atol=5e-5, msg="VI- many-series mismatch")

