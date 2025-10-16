"""CUDA bindings tests for the OTTO indicator (HOTT/LOTT)."""
import pytest
import numpy as np

try:
    import cupy as cp
except ImportError:  # pragma: no cover
    cp = None

try:
    import my_project as ti
except ImportError:
    pytest.skip(
        "Python module not built. Run 'maturin develop --features python,cuda' first",
        allow_module_level=True,
    )

from test_utils import assert_close


def _has(name: str) -> bool:
    return hasattr(ti, name)


def _cuda_available() -> bool:
    if cp is None:
        return False
    if not (_has("otto_cuda_batch_dev") and _has("otto_cuda_many_series_one_param_dev")):
        return False
    # Probe
    try:
        data = np.linspace(1.0, 2.0, 260, dtype=np.float32)
        hott, lott = ti.otto_cuda_batch_dev(
            data, (2, 2, 0), (0.6, 0.6, 0.0), (10, 10, 0), (25, 25, 0)
        )
        _ = cp.asarray(hott)
        _ = cp.asarray(lott)
        return True
    except Exception as exc:  # pragma: no cover
        msg = str(exc).lower()
        if "cuda not available" in msg or "ptx" in msg or "driver" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or otto cuda bindings missing")
class TestOttoCuda:
    def test_otto_cuda_batch_matches_cpu_var(self):
        n = 2048
        data = (np.sin(np.linspace(0, 6.28, n)) * 0.01 + np.linspace(0, 0.2, n)).astype(
            np.float64
        )
        params = dict(
            ott_period=2,
            ott_percent=0.6,
            fast_vidya_length=10,
            slow_vidya_length=25,
            correcting_constant=100000.0,
            ma_type="VAR",
        )
        cpu_hott, cpu_lott = ti.otto(data, **params)

        hott_h, lott_h = ti.otto_cuda_batch_dev(
            data.astype(np.float32),
            (params["ott_period"], params["ott_period"], 0),
            (params["ott_percent"], params["ott_percent"], 0.0),
            (params["fast_vidya_length"], params["fast_vidya_length"], 0),
            (params["slow_vidya_length"], params["slow_vidya_length"], 0),
            (params["correcting_constant"], params["correcting_constant"], 0.0),
            [params["ma_type"]],
        )
        gpu_hott = cp.asnumpy(cp.asarray(hott_h))[0]
        gpu_lott = cp.asnumpy(cp.asarray(lott_h))[0]

        assert_close(gpu_hott, cpu_hott, rtol=3e-3, atol=3e-4, msg="OTTO HOTT mismatch")
        assert_close(gpu_lott, cpu_lott, rtol=3e-3, atol=3e-4, msg="OTTO LOTT mismatch")

    def test_otto_cuda_many_series_one_param_matches_cpu_var(self):
        cols = 300
        rows = 64
        base = np.linspace(0, 1, cols, dtype=np.float64)
        tm = np.zeros((cols, rows), dtype=np.float64)
        for s in range(rows):
            tm[:, s] = np.sin(base * (1.0 + 0.05 * s)) * 0.01 + base * 0.1

        params = dict(
            ott_period=2,
            ott_percent=0.6,
            fast_vidya_length=10,
            slow_vidya_length=25,
            correcting_constant=100000.0,
            ma_type="VAR",
        )
        cpu_hott = np.zeros_like(tm)
        cpu_lott = np.zeros_like(tm)
        for s in range(rows):
            h, l = ti.otto(tm[:, s], **params)
            cpu_hott[:, s] = h
            cpu_lott[:, s] = l

        hott_h, lott_h = ti.otto_cuda_many_series_one_param_dev(
            tm.astype(np.float32),
            cols,
            rows,
            params["ott_period"],
            params["ott_percent"],
            params["fast_vidya_length"],
            params["slow_vidya_length"],
            params["correcting_constant"],
            params["ma_type"],
        )
        gpu_hott = cp.asnumpy(cp.asarray(hott_h))
        gpu_lott = cp.asnumpy(cp.asarray(lott_h))

        assert_close(gpu_hott, cpu_hott, rtol=3e-3, atol=3e-4, msg="OTTO HOTT TM mismatch")
        assert_close(gpu_lott, cpu_lott, rtol=3e-3, atol=3e-4, msg="OTTO LOTT TM mismatch")

