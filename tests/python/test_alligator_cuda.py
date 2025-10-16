"""
Python binding tests for Alligator CUDA kernels.
Skips gracefully when CUDA/CuPy or CUDA bindings are unavailable.
"""
import numpy as np
import pytest

try:
    import cupy as cp
except ImportError:  # pragma: no cover
    cp = None

try:
    import my_project as ti
except ImportError:  # pragma: no cover
    pytest.skip(
        "Python module not built. Run 'maturin develop --features python,cuda' first",
        allow_module_level=True,
    )

from test_utils import assert_close, load_test_data


def _cuda_available() -> bool:
    if cp is None:
        return False
    if not hasattr(ti, "alligator_cuda_batch_dev"):
        return False
    try:
        probe = np.array([np.nan, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        out = ti.alligator_cuda_batch_dev(
            probe,
            (5, 5, 0),
            (3, 3, 0),
            (8, 8, 0),
            (5, 5, 0),
            (13, 13, 0),
            (8, 8, 0),
        )
        _ = cp.asarray(out["jaw"])  # ensure device handle is usable
        _ = cp.asarray(out["teeth"])
        _ = cp.asarray(out["lips"])
        return True
    except Exception as exc:  # pragma: no cover
        msg = str(exc).lower()
        if "cuda not available" in msg or "ptx" in msg or "nvcc" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestAlligatorCuda:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_alligator_cuda_batch_matches_cpu(self, test_data):
        close = test_data["hl2"][:4096].astype(np.float64)

        jp = (10, 18, 4)
        jo = (3, 6, 1)
        tp = (6, 14, 4)
        to = (2, 5, 1)
        lp = (3, 9, 3)
        lo = (1, 3, 1)

        cpu = ti.alligator_batch(close, jp, jo, tp, to, lp, lo)
        jaw_cpu = cpu["jaw"]
        teeth_cpu = cpu["teeth"]
        lips_cpu = cpu["lips"]

        out = ti.alligator_cuda_batch_dev(
            close.astype(np.float32), jp, jo, tp, to, lp, lo
        )
        jaw_gpu = cp.asnumpy(cp.asarray(out["jaw"]))
        teeth_gpu = cp.asnumpy(cp.asarray(out["teeth"]))
        lips_gpu = cp.asnumpy(cp.asarray(out["lips"]))

        assert jaw_gpu.shape == jaw_cpu.shape
        assert teeth_gpu.shape == teeth_cpu.shape
        assert lips_gpu.shape == lips_cpu.shape

        assert_close(jaw_gpu, jaw_cpu, rtol=1e-5, atol=1e-6, msg="jaw mismatch")
        assert_close(teeth_gpu, teeth_cpu, rtol=1e-5, atol=1e-6, msg="teeth mismatch")
        assert_close(lips_gpu, lips_cpu, rtol=1e-5, atol=1e-6, msg="lips mismatch")

    def test_alligator_cuda_many_series_one_param_matches_cpu(self, test_data):
        # time-major (rows=t, cols=series)
        cols = 3
        rows = 2048
        data_tm = np.full((rows, cols), np.nan, dtype=np.float64)
        for j in range(cols):
            for t in range(16, rows):
                x = float(t) + 0.07 * float(j)
                data_tm[t, j] = np.cos(0.0021 * x) + 0.0006 * x

        params = dict(jaw_period=13, jaw_offset=8, teeth_period=8, teeth_offset=5, lips_period=5, lips_offset=3)

        # CPU reference (convert to row-major per-series)
        jaw_cpu = np.full_like(data_tm, np.nan)
        teeth_cpu = np.full_like(data_tm, np.nan)
        lips_cpu = np.full_like(data_tm, np.nan)
        for j in range(cols):
            out = ti.alligator(data_tm[:, j], **params)
            jaw_cpu[:, j] = out["jaw"]
            teeth_cpu[:, j] = out["teeth"]
            lips_cpu[:, j] = out["lips"]

        out = ti.alligator_cuda_many_series_one_param_dev(
            data_tm.astype(np.float32),
            params["jaw_period"], params["jaw_offset"],
            params["teeth_period"], params["teeth_offset"],
            params["lips_period"], params["lips_offset"],
        )
        jaw_gpu = cp.asnumpy(cp.asarray(out["jaw"]))
        teeth_gpu = cp.asnumpy(cp.asarray(out["teeth"]))
        lips_gpu = cp.asnumpy(cp.asarray(out["lips"]))

        assert jaw_gpu.shape == jaw_cpu.shape == (rows, cols)
        assert_close(jaw_gpu, jaw_cpu, rtol=1e-5, atol=1e-6, msg="jaw mismatch")
        assert_close(teeth_gpu, teeth_cpu, rtol=1e-5, atol=1e-6, msg="teeth mismatch")
        assert_close(lips_gpu, lips_cpu, rtol=1e-5, atol=1e-6, msg="lips mismatch")

