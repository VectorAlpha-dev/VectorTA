"""Python binding tests for Fisher CUDA kernels."""
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

from test_utils import assert_close


def _cuda_available() -> bool:
    if cp is None:
        return False
    if not hasattr(ti, "fisher_cuda_batch_dev"):
        return False
    try:

        high = np.array([np.nan, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        low = high - 1.0
        h = ti.fisher_cuda_batch_dev(high, low, (3, 3, 0))
        _ = cp.asarray(h["fisher"])
        _ = cp.asarray(h["signal"])
        return True
    except Exception as exc:
        msg = str(exc).lower()
        if "cuda not available" in msg or "nvcc" in msg or "ptx" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or bindings not built")
class TestFisherCuda:
    def test_fisher_cuda_batch_matches_cpu(self):
        n = 4096
        i = np.arange(n, dtype=np.float64)
        high = np.sin(i * 0.002) + 0.001 * i
        low = high - 0.3 - 0.05 * np.cos(i * 0.1)
        high[:10] = np.nan
        low[:10] = np.nan

        sweep = (9, 45, 6)
        cpu = ti.fisher_batch(high, low, sweep)
        out = ti.fisher_cuda_batch_dev(high.astype(np.float32), low.astype(np.float32), sweep)
        g_fish = cp.asnumpy(cp.asarray(out["fisher"]))
        g_sig = cp.asnumpy(cp.asarray(out["signal"]))
        g_fish = g_fish.reshape(cpu["fisher"].shape)
        g_sig = g_sig.reshape(cpu["signal"].shape)
        assert_close(g_fish, cpu["fisher"], rtol=1e-4, atol=1e-5, msg="Fisher batch mismatch")
        assert_close(g_sig, cpu["signal"], rtol=1e-4, atol=1e-5, msg="Signal batch mismatch")

    def test_fisher_cuda_many_series_one_param_matches_cpu(self):
        rows = 2048
        cols = 4
        i = np.arange(rows, dtype=np.float64)
        base = np.sin(i * 0.0023) + 0.0002 * i
        high_tm = np.zeros((rows, cols), dtype=np.float64)
        low_tm = np.zeros((rows, cols), dtype=np.float64)
        for s in range(cols):
            high_tm[:, s] = base * (1.0 + 0.02 * s)
            low_tm[:, s] = high_tm[:, s] - 0.25 - 0.07 * np.cos(i * (0.11 + 0.01 * s))
        high_tm[:3, :] = np.nan
        low_tm[:3, :] = np.nan
        period = 13

        cpu_fish = np.zeros_like(high_tm)
        cpu_sig = np.zeros_like(low_tm)
        for s in range(cols):
            fish, sig = ti.fisher(high_tm[:, s], low_tm[:, s], period)
            cpu_fish[:, s] = fish
            cpu_sig[:, s] = sig

        out = ti.fisher_cuda_many_series_one_param_dev(
            high_tm.astype(np.float32).ravel(),
            low_tm.astype(np.float32).ravel(),
            cols,
            rows,
            period,
        )
        g_fish = cp.asnumpy(cp.asarray(out["fisher"]))
        g_sig = cp.asnumpy(cp.asarray(out["signal"]))
        g_fish = g_fish.reshape(rows, cols)
        g_sig = g_sig.reshape(rows, cols)
        assert_close(g_fish, cpu_fish, rtol=1e-4, atol=1e-5, msg="Fisher many-series mismatch")
        assert_close(g_sig, cpu_sig, rtol=1e-4, atol=1e-5, msg="Signal many-series mismatch")

