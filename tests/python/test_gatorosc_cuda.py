"""
Python CUDA binding tests for Gator Oscillator (GATOR).
Skips gracefully when CUDA or bindings are unavailable.
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
    if not hasattr(ti, "gatorosc_cuda_batch_dev"):
        return False
    try:
        x = np.arange(0, 128, dtype=np.float32)
        x[:5] = np.nan
        out = ti.gatorosc_cuda_batch_dev(
            x,
            (13, 13, 0), (8, 8, 0),
            (8, 8, 0), (5, 5, 0),
            (5, 5, 0), (3, 3, 0),
        )
        _ = cp.asarray(out[0])
        return True
    except Exception as e:
        msg = str(e).lower()
        if "cuda not available" in msg or "nvcc" in msg or "ptx" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or bindings not built")
class TestGatorOscCuda:
    @pytest.fixture(scope="class")
    def close(self):
        return load_test_data()["close"].astype(np.float64)

    def test_batch_matches_cpu(self, close):
        p = dict(jl=13, js=8, tl=8, ts=5, ll=5, ls=3)
        u_cpu, l_cpu, uc_cpu, lc_cpu = ti.gatorosc(
            close, p["jl"], p["js"], p["tl"], p["ts"], p["ll"], p["ls"]
        )

        (u_dev, l_dev, uc_dev, lc_dev) = ti.gatorosc_cuda_batch_dev(
            close.astype(np.float32),
            (p["jl"], p["jl"], 0), (p["js"], p["js"], 0),
            (p["tl"], p["tl"], 0), (p["ts"], p["ts"], 0),
            (p["ll"], p["ll"], 0), (p["ls"], p["ls"], 0),
        )
        u_gpu = cp.asnumpy(cp.asarray(u_dev))[0]
        l_gpu = cp.asnumpy(cp.asarray(l_dev))[0]
        uc_gpu = cp.asnumpy(cp.asarray(uc_dev))[0]
        lc_gpu = cp.asnumpy(cp.asarray(lc_dev))[0]

        assert_close(u_gpu, u_cpu, rtol=1e-3, atol=2e-3, msg="upper mismatch")
        assert_close(l_gpu, l_cpu, rtol=1e-3, atol=2e-3, msg="lower mismatch")
        assert_close(uc_gpu, uc_cpu, rtol=2e-3, atol=3e-3, msg="upper_change mismatch")
        assert_close(lc_gpu, lc_cpu, rtol=2e-3, atol=3e-3, msg="lower_change mismatch")

    def test_many_series_one_param_matches_cpu(self, close):
        T = 1024
        N = 4
        base = close[:T]
        series = np.vstack([base * (1.0 + 0.01 * j) for j in range(N)]).T

        p = dict(jl=13, js=8, tl=8, ts=5, ll=5, ls=3)

        u_cpu = np.zeros_like(series)
        l_cpu = np.zeros_like(series)
        uc_cpu = np.zeros_like(series)
        lc_cpu = np.zeros_like(series)
        for j in range(N):
            u, l, uc, lc = ti.gatorosc(
                series[:, j].astype(np.float64),
                p["jl"], p["js"], p["tl"], p["ts"], p["ll"], p["ls"]
            )
            u_cpu[:, j] = u
            l_cpu[:, j] = l
            uc_cpu[:, j] = uc
            lc_cpu[:, j] = lc

        (u_dev, l_dev, uc_dev, lc_dev) = ti.gatorosc_cuda_many_series_one_param_dev(
            series.astype(np.float32).ravel(),
            N,
            T,
            p["jl"], p["js"], p["tl"], p["ts"], p["ll"], p["ls"],
        )
        u_gpu = cp.asnumpy(cp.asarray(u_dev)).reshape(T, N)
        l_gpu = cp.asnumpy(cp.asarray(l_dev)).reshape(T, N)
        uc_gpu = cp.asnumpy(cp.asarray(uc_dev)).reshape(T, N)
        lc_gpu = cp.asnumpy(cp.asarray(lc_dev)).reshape(T, N)

        assert_close(u_gpu, u_cpu, rtol=1e-3, atol=2e-3, msg="upper TM mismatch")
        assert_close(l_gpu, l_cpu, rtol=1e-3, atol=2e-3, msg="lower TM mismatch")
        assert_close(uc_gpu, uc_cpu, rtol=2e-3, atol=3e-3, msg="upper_change TM mismatch")
        assert_close(lc_gpu, lc_cpu, rtol=2e-3, atol=3e-3, msg="lower_change TM mismatch")

