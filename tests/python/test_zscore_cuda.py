"""Python binding tests for Zscore CUDA kernels."""
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
    if not hasattr(ti, "zscore_cuda_batch_dev"):
        return False
    try:
        data = np.array([np.nan, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        handle, meta = ti.zscore_cuda_batch_dev(
            data,
            period_range=(3, 3, 0),
            nbdev_range=(1.0, 1.0, 0.0),
        )
        _ = cp.asarray(handle)
        assert meta["periods"][0] == 3
        return True
    except Exception as exc:  # pragma: no cover - defensive guard
        message = str(exc).lower()
        if "cuda not available" in message or "ptx" in message:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestZscoreCuda:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_zscore_cuda_batch_matches_cpu(self, test_data):
        close = test_data["close"][:2048].astype(np.float64)

        period_range = (10, 30, 10)
        nbdev_range = (0.5, 1.5, 0.5)

        cpu = ti.zscore_batch(
            close,
            period_range=period_range,
            ma_type="sma",
            nbdev_range=nbdev_range,
            devtype_range=(0, 0, 0),
        )
        cpu_vals = cpu["values"].astype(np.float32)

        handle, meta = ti.zscore_cuda_batch_dev(
            close.astype(np.float32),
            period_range,
            nbdev_range,
        )

        gpu_vals = cp.asnumpy(cp.asarray(handle)).reshape(cpu_vals.shape)

        assert_close(gpu_vals, cpu_vals, rtol=3e-4, atol=3e-4, msg="CUDA zscore mismatch")
        assert np.array_equal(meta["periods"], cpu["periods"])  # type: ignore[index]
        assert np.allclose(meta["nbdevs"], cpu["nbdevs"], atol=1e-6)  # type: ignore[index]
        assert [*meta["ma_types"]] == ["sma"] * cpu_vals.shape[0]
        assert np.array_equal(meta["devtypes"], cpu["devtypes"])  # type: ignore[index]

