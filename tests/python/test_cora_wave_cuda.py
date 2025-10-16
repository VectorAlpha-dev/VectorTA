"""Python binding tests for CoRa Wave CUDA kernels."""
import pytest
import numpy as np

try:
    import cupy as cp
except ImportError:  # optional dependency for CUDA path
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
    if not hasattr(ti, 'cora_wave_cuda_batch_dev'):
        return False
    try:
        x = np.array([np.nan, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        handle, _meta = ti.cora_wave_cuda_batch_dev(
            x.astype(np.float32),
            period_range=(5, 5, 0),
            r_multi_range=(2.0, 2.0, 0.0),
            smooth=True,
        )
        _ = cp.asarray(handle)
        return True
    except Exception as e:
        msg = str(e).lower()
        if 'cuda not available' in msg or 'nvcc' in msg or 'ptx' in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestCoraWaveCuda:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_cora_wave_cuda_batch_matches_cpu(self, test_data):
        close = test_data['close']
        period = 20
        r_multi = 2.0
        smooth = True

        cpu = ti.cora_wave(close, period=period, r_multi=r_multi, smooth=smooth)

        handle, meta = ti.cora_wave_cuda_batch_dev(
            close.astype(np.float32),
            period_range=(period, period, 0),
            r_multi_range=(r_multi, r_multi, 0.0),
            smooth=smooth,
        )
        gpu = cp.asarray(handle)
        gpu_first = cp.asnumpy(gpu)[0]

        assert_close(
            gpu_first,
            cpu,
            rtol=5e-3,
            atol=1e-4,
            msg="CoRa Wave CUDA batch vs CPU mismatch",
        )

    def test_cora_wave_cuda_many_series_one_param_matches_cpu(self, test_data):
        T = 1024
        N = 4
        period = 24
        r_multi = 2.0
        smooth = True
        series = test_data['close'][:T].astype(np.float64)
        data_tm = np.zeros((T, N), dtype=np.float64)
        for j in range(N):
            data_tm[:, j] = series * (1.0 + 0.03 * j)

        cpu_tm = np.zeros_like(data_tm)
        for j in range(N):
            cpu_tm[:, j] = ti.cora_wave(data_tm[:, j], period=period, r_multi=r_multi, smooth=smooth)

        handle = ti.cora_wave_cuda_many_series_one_param_dev(
            data_tm.astype(np.float32), cols=N, rows=T, period=period, r_multi=r_multi, smooth=smooth
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle))

        assert gpu_tm.shape == data_tm.shape
        assert_close(
            gpu_tm,
            cpu_tm,
            rtol=6e-3,
            atol=1e-4,
            msg="CoRa Wave CUDA many-series vs CPU mismatch",
        )

