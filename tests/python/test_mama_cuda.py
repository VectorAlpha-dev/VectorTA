"""Python binding tests for MAMA CUDA kernels."""
import pytest
import numpy as np

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

from test_utils import assert_close, load_test_data


def _cuda_available() -> bool:
    if cp is None:
        return False
    if not hasattr(ti, 'mama_cuda_batch_dev'):
        return False
    try:
        sample = np.linspace(1.0, 3.0, 12, dtype=np.float32)
        handles = ti.mama_cuda_batch_dev(sample, (0.4, 0.4, 0.0), (0.05, 0.05, 0.0))
        mama_handle, fama_handle = handles
        _ = cp.asarray(mama_handle)
        _ = cp.asarray(fama_handle)
        return True
    except Exception as exc:  # pragma: no cover - defensive skip path
        msg = str(exc).lower()
        if 'cuda not available' in msg or 'nvcc' in msg or 'ptx' in msg:
            return False
        return True


@pytest.mark.skipif(
    not _cuda_available(), reason="CUDA not available or cuda bindings not built"
)
class TestMamaCuda:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_mama_cuda_batch_matches_cpu(self, test_data):
        close = test_data['close'].astype(np.float64)
        fast_range = (0.35, 0.55, 0.1)
        slow_range = (0.03, 0.09, 0.03)

        cpu = ti.mama_batch(close, fast_range, slow_range)
        cpu_m = cpu['mama']
        cpu_f = cpu['fama']

        mama_handle, fama_handle = ti.mama_cuda_batch_dev(
            close.astype(np.float32), fast_range, slow_range
        )
        gpu_m = cp.asnumpy(cp.asarray(mama_handle))
        gpu_f = cp.asnumpy(cp.asarray(fama_handle))

        assert gpu_m.shape == cpu_m.shape
        assert gpu_f.shape == cpu_f.shape
        assert_close(gpu_m, cpu_m, rtol=1e-4, atol=1e-5, msg="CUDA MAMA batch mismatch")
        assert_close(gpu_f, cpu_f, rtol=1e-4, atol=1e-5, msg="CUDA FAMA batch mismatch")

    def test_mama_cuda_many_series_one_param_matches_cpu(self, test_data):
        T = 480
        N = 3
        base = test_data['close'][:T].astype(np.float64)
        data_tm = np.zeros((T, N), dtype=np.float64)
        for j in range(N):
            data_tm[:, j] = base * (1.0 + 0.015 * j) + 0.05 * j

        fast_limit = 0.5
        slow_limit = 0.06

        cpu_m = np.zeros_like(data_tm)
        cpu_f = np.zeros_like(data_tm)
        for j in range(N):
            mama_vals, fama_vals = ti.mama(data_tm[:, j], fast_limit, slow_limit)
            cpu_m[:, j] = mama_vals
            cpu_f[:, j] = fama_vals

        mama_handle, fama_handle = ti.mama_cuda_many_series_one_param_dev(
            data_tm.astype(np.float32), fast_limit, slow_limit
        )
        gpu_m = cp.asnumpy(cp.asarray(mama_handle))
        gpu_f = cp.asnumpy(cp.asarray(fama_handle))

        assert gpu_m.shape == cpu_m.shape
        assert gpu_f.shape == cpu_f.shape
        assert_close(gpu_m, cpu_m, rtol=1e-4, atol=1e-5, msg="CUDA MAMA many-series mismatch")
        assert_close(gpu_f, cpu_f, rtol=1e-4, atol=1e-5, msg="CUDA FAMA many-series mismatch")
