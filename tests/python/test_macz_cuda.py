"""Python binding tests for MAC-Z CUDA kernels."""
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
    if not hasattr(ti, "macz_cuda_batch_dev"):
        return False
    try:
        data = np.array([np.nan, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        vol = np.array([np.nan, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        handle, meta = ti.macz_cuda_batch_dev(
            data,
            vol,
            fast_length_range=(3, 3, 0),
            slow_length_range=(5, 5, 0),
            signal_length_range=(2, 2, 0),
            lengthz_range=(3, 3, 0),
            length_stdev_range=(5, 5, 0),
            a_range=(1.0, 1.0, 0.0),
            b_range=(1.0, 1.0, 0.0),
        )
        _ = cp.asarray(handle)
        assert int(meta["fast_lengths"][0]) == 3
        return True
    except Exception as exc:
        msg = str(exc).lower()
        if "cuda not available" in msg or "ptx" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestMaczCuda:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_macz_cuda_batch_matches_cpu(self, test_data):
        close = test_data["close"][:4096].astype(np.float64)
        volume = test_data["volume"][:4096].astype(np.float64)

        sweep = dict(
            fast_length_range=(10, 14, 2),
            slow_length_range=(20, 30, 5),
            signal_length_range=(7, 11, 2),
            lengthz_range=(18, 22, 2),
            length_stdev_range=(20, 30, 5),
            a_range=(0.8, 1.2, 0.2),
            b_range=(0.8, 1.2, 0.2),
        )

        cpu = ti.macz_batch(close, volume, **sweep)
        cpu_vals = cpu["values"].astype(np.float32)

        handle, meta = ti.macz_cuda_batch_dev(
            close.astype(np.float32),
            volume.astype(np.float32),
            **sweep,
        )
        gpu_vals = cp.asnumpy(cp.asarray(handle)).reshape(cpu_vals.shape)

        assert_close(gpu_vals, cpu_vals, rtol=5e-4, atol=7e-4, msg="CUDA MAC-Z histogram mismatch")

        assert np.array_equal(meta["fast_lengths"], cpu["fast_lengths"])
        assert np.array_equal(meta["slow_lengths"], cpu["slow_lengths"])

    def test_macz_cuda_many_series_one_param_matches_cpu(self, test_data):
        cols, rows = 5, 1024
        tm = np.full((rows, cols), np.nan, dtype=np.float64)
        vol = np.full((rows, cols), np.nan, dtype=np.float64)
        for s in range(cols):
            for t in range(s, rows):
                x = t + s * 0.2
                tm[t, s] = np.sin(x * 0.002) + 0.0003 * x
                vol[t, s] = np.abs(np.cos(x * 0.001)) + 0.4

        params = dict(
            fast_length=12,
            slow_length=25,
            signal_length=9,
            lengthz=20,
            length_stdev=25,
            a=1.0,
            b=1.0,
            use_lag=False,
            gamma=0.02,
        )


        cpu_tm = np.full_like(tm, np.nan)
        for s in range(cols):
            out = ti.macz(tm[:, s], vol[:, s], **params)
            cpu_tm[:, s] = out
        cpu_f32 = cpu_tm.astype(np.float32)

        handle = ti.macz_cuda_many_series_one_param_dev(
            tm.astype(np.float32).ravel(),
            vol.astype(np.float32).ravel(),
            cols,
            rows,
            **params,
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle)).reshape(rows, cols)
        assert_close(gpu_tm, cpu_f32, rtol=5e-4, atol=7e-4, msg="CUDA MAC-Z TM mismatch")

