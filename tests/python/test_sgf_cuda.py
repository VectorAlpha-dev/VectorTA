import numpy as np
import pytest

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    import vector_ta as ti
except ImportError:
    pytest.skip("Python module not built. Run 'maturin develop --features python,cuda' first", allow_module_level=True)


def _cuda_available() -> bool:
    if cp is None:
        return False
    if not hasattr(ti, "sgf_cuda_batch_dev"):
        return False
    try:
        data = np.array([np.nan, np.nan, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        handle = ti.sgf_cuda_batch_dev(data.astype(np.float64), (5, 5, 0), (2, 2, 0))
        _ = cp.asarray(handle)
        return True
    except Exception as e:
        msg = str(e).lower()
        return not any(tag in msg for tag in ("cuda not available", "ptx", "nvcc"))


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestSgfCuda:
    def test_sgf_cuda_batch_matches_cpu(self):
        x = np.arange(256, dtype=np.float64)
        data = 1.0 + 0.5 * x - 0.01 * x * x
        cpu = ti.sgf(data, 9, 2)
        handle = ti.sgf_cuda_batch_dev(data.astype(np.float64), (9, 9, 0), (2, 2, 0))
        gpu = cp.asnumpy(cp.asarray(handle))[0]
        np.testing.assert_allclose(gpu, cpu, rtol=1e-5, atol=1e-5, equal_nan=True)

    def test_sgf_cuda_many_series_one_param_matches_cpu(self):
        rows = 256
        cols = 4
        data_tm = np.zeros((rows, cols), dtype=np.float32)
        for j in range(cols):
            x = np.arange(rows, dtype=np.float64) + j * 0.25
            data_tm[:, j] = (1.0 + 0.4 * x - 0.01 * x * x).astype(np.float32)

        cpu = np.column_stack([ti.sgf(np.ascontiguousarray(data_tm[:, j]).astype(np.float64), 9, 2) for j in range(cols)])
        handle = ti.sgf_cuda_many_series_one_param_dev(data_tm, 9, 2)
        gpu = cp.asnumpy(cp.asarray(handle))
        np.testing.assert_allclose(gpu, cpu, rtol=1e-5, atol=1e-5, equal_nan=True)
