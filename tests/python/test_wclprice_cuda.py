"""Python CUDA binding tests for WCLPRICE."""
import numpy as np
import pytest

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    import my_project as ti
except ImportError:
    pytest.skip("Python module not built. Run 'maturin develop --features python,cuda' first", allow_module_level=True)

from test_utils import load_test_data, assert_close


def _cuda_available() -> bool:
    if cp is None:
        return False
    if not hasattr(ti, "wclprice_cuda_dev"):
        return False
    try:
        high = np.array([np.nan, 2.0, 2.5, 2.75], dtype=np.float32)
        low = np.array([np.nan, 1.0, 1.5, 1.75], dtype=np.float32)
        close = np.array([np.nan, 1.5, 1.75, 2.0], dtype=np.float32)
        handle = ti.wclprice_cuda_dev(high, low, close)
        _ = cp.asarray(handle)
        return True
    except Exception as exc:
        msg = str(exc).lower()
        if "cuda not available" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestWclpriceCuda:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_wclprice_cuda_matches_cpu(self, test_data):
        high = test_data["high"][:2048].astype(np.float64)
        low = test_data["low"][:2048].astype(np.float64)
        close = test_data["close"][:2048].astype(np.float64)

        cpu = ti.wclprice(high, low, close)

        handle = ti.wclprice_cuda_dev(
            high.astype(np.float32),
            low.astype(np.float32),
            close.astype(np.float32),
        )
        gpu = cp.asnumpy(cp.asarray(handle))[0]

        assert_close(gpu, cpu, rtol=1e-5, atol=1e-6, msg="CUDA vs CPU mismatch")

    def test_wclprice_cuda_nan_propagation(self):
        high = np.array([np.nan, np.nan, 5.0, 5.5, 6.0], dtype=np.float64)
        low = np.array([np.nan, np.nan, 4.5, 5.0, 5.5], dtype=np.float64)
        close = np.array([np.nan, 4.8, 5.0, 5.2, 5.4], dtype=np.float64)

        cpu = ti.wclprice(high, low, close)
        handle = ti.wclprice_cuda_dev(
            high.astype(np.float32),
            low.astype(np.float32),
            close.astype(np.float32),
        )
        gpu = cp.asnumpy(cp.asarray(handle))[0]

        assert np.isnan(gpu[0]) and np.isnan(gpu[1])
        assert_close(gpu, cpu, rtol=1e-5, atol=1e-6, msg="NaN propagation mismatch")
