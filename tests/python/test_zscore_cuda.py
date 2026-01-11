"""Python binding tests for Zscore CUDA kernels."""
import numpy as np
import pytest

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
    except Exception as exc:  
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
        assert np.array_equal(meta["periods"], cpu["periods"])  
        assert np.allclose(meta["nbdevs"], cpu["nbdevs"], atol=1e-6)  
        assert [*meta["ma_types"]] == ["sma"] * cpu_vals.shape[0]
        assert np.array_equal(meta["devtypes"], cpu["devtypes"])  

    def test_zscore_cuda_many_series_one_param_matches_cpu(self, test_data):
        if not hasattr(ti, "zscore_cuda_many_series_one_param_dev"):
            pytest.skip("many-series zscore CUDA binding not present")

        
        cols = 4
        rows = 512
        price = test_data["close"][: rows * cols].astype(np.float64)
        price = price.reshape(rows, cols).copy(order="C")  
        for s in range(cols):
            for t in range(0, rows, 127):
                price[t, s] = np.nan

        period = 14
        nbdev = 2.0

        
        cpu_tm = np.full((rows, cols), np.nan, dtype=np.float64)
        for s in range(cols):
            col = price[:, s].copy()
            out = ti.zscore_batch(
                col,
                period_range=(period, period, 0),
                ma_type="sma",
                nbdev_range=(nbdev, nbdev, 0.0),
                devtype_range=(0, 0, 0),
            )
            cpu_tm[:, s] = out["values"].ravel()

        handle = ti.zscore_cuda_many_series_one_param_dev(
            price.astype(np.float32).ravel(),
            cols,
            rows,
            period,
            nbdev,
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle)).reshape(rows, cols)

        assert_close(gpu_tm, cpu_tm.astype(np.float32), rtol=4e-4, atol=4e-4, msg="CUDA zscore many-series mismatch")
