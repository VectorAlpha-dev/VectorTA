"""Python binding tests for AD CUDA kernels."""
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
    if not hasattr(ti, "ad_cuda_dev"):
        return False
    try:
        # Tiny smoke test
        n = 64
        base = np.linspace(100.0, 101.0, n, dtype=np.float32)
        high = base + 0.5
        low = base - 0.5
        vol = np.linspace(800.0, 1200.0, n, dtype=np.float32)
        _ = ti.ad_cuda_dev(high, low, base, vol)
        return True
    except Exception as exc:  # pragma: no cover - defensive guard
        message = str(exc).lower()
        if "cuda not available" in message or "ptx" in message:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestAdCuda:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_ad_cuda_series_matches_cpu(self, test_data):
        high = test_data["high"][:2048].astype(np.float64)
        low = test_data["low"][:2048].astype(np.float64)
        close = test_data["close"][:2048].astype(np.float64)
        volume = test_data["volume"][:2048].astype(np.float64)

        cpu = ti.ad(high, low, close, volume)

        handle = ti.ad_cuda_dev(
            high.astype(np.float32),
            low.astype(np.float32),
            close.astype(np.float32),
            volume.astype(np.float32),
        )

        gpu = cp.asnumpy(cp.asarray(handle))
        assert gpu.shape == (1, cpu.shape[0])

        # Allow modest tolerance for FP32 accumulation
        assert_close(
            gpu[0],
            cpu.astype(np.float32),
            rtol=2e-3,
            atol=2e-2,
            msg="CUDA AD series mismatch",
        )

    def test_ad_cuda_dlpack_matches_cuda_array_interface(self, test_data):
        """Shared DLPack helper: AD handle round-trips via DLPack."""
        high = test_data["high"][:512].astype(np.float64)
        low = test_data["low"][:512].astype(np.float64)
        close = test_data["close"][:512].astype(np.float64)
        volume = test_data["volume"][:512].astype(np.float64)

        handle = ti.ad_cuda_dev(
            high.astype(np.float32),
            low.astype(np.float32),
            close.astype(np.float32),
            volume.astype(np.float32),
        )

        gpu_cai = cp.asarray(handle)
        gpu_dlpack = cp.fromDlpack(handle)

        assert gpu_cai.shape == gpu_dlpack.shape
        assert_close(
            cp.asnumpy(gpu_dlpack),
            cp.asnumpy(gpu_cai),
            rtol=1e-6,
            atol=1e-6,
            msg="AD DLPack vs __cuda_array_interface__ mismatch",
        )
