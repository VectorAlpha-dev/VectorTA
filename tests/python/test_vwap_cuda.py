"""Python binding tests for VWAP CUDA kernels."""
import pytest
import numpy as np

try:
    import cupy as cp
except ImportError:  # pragma: no cover - optional dependency for CUDA tests
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
    if not hasattr(ti, "vwap_cuda_batch_dev"):
        return False
    try:
        ts = np.array([1_600_000_000_000, 1_600_000_060_000, 1_600_000_120_000], dtype=np.int64)
        vols = np.array([0.0, 1.0, 1.5], dtype=np.float64)
        prices = np.array([100.0, 101.0, 102.5], dtype=np.float64)
        handle = ti.vwap_cuda_batch_dev(ts, vols, prices, ("1m", "1m", 0))
        _ = cp.asarray(handle)
        return True
    except Exception as exc:  # pragma: no cover - availability probe
        msg = str(exc).lower()
        if "cuda not available" in msg or "ptx" in msg or "nvcc" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestVwapCuda:
    @pytest.fixture(scope="class")
    def test_data(self):
        data = load_test_data()
        limit = 2048  # keep runtime manageable for CI
        return {
            "timestamp": data["timestamp"][:limit].astype(np.int64),
            "volume": data["volume"][:limit].astype(np.float64),
            "price": data["close"][:limit].astype(np.float64),
        }

    def test_vwap_cuda_batch_matches_cpu(self, test_data):
        ts = test_data["timestamp"]
        volumes = test_data["volume"]
        prices = test_data["price"]

        anchor_range = ("1m", "3m", 1)

        cpu = ti.vwap_batch(ts, volumes, prices, anchor_range)
        cpu_values = np.asarray(cpu["values"], dtype=np.float64)

        handle = ti.vwap_cuda_batch_dev(ts, volumes, prices, anchor_range)
        gpu = cp.asnumpy(cp.asarray(handle)).astype(np.float64)

        assert gpu.shape == cpu_values.shape
        assert_close(gpu, cpu_values, rtol=1e-4, atol=5e-4, msg="CUDA batch vs CPU mismatch")
