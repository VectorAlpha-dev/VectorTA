"""Python binding tests for WAD CUDA kernels."""
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
    if not hasattr(ti, "wad_cuda_dev"):
        return False
    try:
        base = np.array([100.0, 100.5, 101.0, 100.75, 100.9], dtype=np.float32)
        high = base + 0.6
        low = base - 0.6
        _ = ti.wad_cuda_dev(high, low, base)
        return True
    except Exception as exc:  # pragma: no cover - defensive guard
        message = str(exc).lower()
        if "cuda not available" in message or "ptx" in message:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestWadCuda:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_wad_cuda_matches_cpu(self, test_data):
        high = test_data["high"][:2048].astype(np.float64)
        low = test_data["low"][:2048].astype(np.float64)
        close = test_data["close"][:2048].astype(np.float64)

        cpu = ti.wad(high, low, close)

        handle = ti.wad_cuda_dev(
            high.astype(np.float32),
            low.astype(np.float32),
            close.astype(np.float32),
        )

        gpu = cp.asnumpy(cp.asarray(handle))
        assert gpu.shape == (1, cpu.shape[0])

        assert_close(
            gpu[0],
            cpu.astype(np.float32),
            rtol=1e-3,
            atol=1e-3,
            msg="CUDA WAD mismatch",
        )
