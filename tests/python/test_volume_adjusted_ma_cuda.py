"""Python binding tests for Volume Adjusted MA CUDA kernels."""
import pytest
import numpy as np

try:
    import cupy as cp
except ImportError:  # pragma: no cover - optional dependency for CUDA path
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
    if not hasattr(ti, 'volume_adjusted_ma_cuda_batch_dev'):
        return False
    try:
        price = np.array([np.nan, 1.0, 1.5, 2.0, 2.5], dtype=np.float32)
        volume = np.array([np.nan, 200.0, 210.0, 220.0, 230.0], dtype=np.float32)
        handle = ti.volume_adjusted_ma_cuda_batch_dev(
            price,
            volume,
            length_range=(3, 3, 0),
            vi_factor_range=(0.6, 0.6, 0.0),
            sample_period_range=(0, 0, 0),
            strict=True,
        )
        _ = cp.asarray(handle)
        return True
    except Exception as exc:  # pragma: no cover - defensive skip
        msg = str(exc).lower()
        if 'cuda not available' in msg or 'nvcc' in msg or 'ptx' in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestVolumeAdjustedMaCuda:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_volume_adjusted_ma_cuda_batch_matches_cpu(self, test_data):
        close = test_data['close']
        volume = test_data['volume']

        length_range = (5, 17, 4)
        vi_range = (0.5, 0.9, 0.2)
        sample_period_range = (0, 12, 4)

        cpu = ti.VolumeAdjustedMa_batch(
            close,
            volume,
            length_range,
            vi_range,
            sample_period_range,
            strict=None,
        )
        cpu_values = np.asarray(cpu['values'])

        handle = ti.volume_adjusted_ma_cuda_batch_dev(
            close.astype(np.float32),
            volume.astype(np.float32),
            length_range,
            vi_range,
            sample_period_range,
            strict=None,
        )
        gpu = cp.asnumpy(cp.asarray(handle))

        assert gpu.shape == cpu_values.shape
        assert_close(
            gpu,
            cpu_values,
            rtol=1e-5,
            atol=1e-6,
            msg="VAMA CUDA batch vs CPU mismatch",
        )

    def test_volume_adjusted_ma_cuda_batch_strict_only_matches_cpu(self, test_data):
        close = test_data['close']
        volume = test_data['volume']

        length_range = (7, 25, 6)
        vi_range = (0.55, 0.95, 0.2)
        sample_period_range = (0, 0, 0)

        cpu = ti.VolumeAdjustedMa_batch(
            close,
            volume,
            length_range,
            vi_range,
            sample_period_range,
            strict=True,
        )
        cpu_values = np.asarray(cpu['values'])

        handle = ti.volume_adjusted_ma_cuda_batch_dev(
            close.astype(np.float32),
            volume.astype(np.float32),
            length_range,
            vi_range,
            sample_period_range,
            strict=True,
        )
        gpu = cp.asnumpy(cp.asarray(handle))

        assert gpu.shape == cpu_values.shape
        assert_close(
            gpu,
            cpu_values,
            rtol=1e-5,
            atol=1e-6,
            msg="VAMA CUDA strict batch vs CPU mismatch",
        )

    def test_volume_adjusted_ma_cuda_many_series_one_param_matches_cpu(self, test_data):
        T = 1024
        N = 4
        length = 20
        vi_factor = 0.68
        strict = False
        sample_period = 10

        price_tm = np.zeros((T, N), dtype=np.float64)
        volume_tm = np.zeros((T, N), dtype=np.float64)
        base_price = test_data['close'][:T]
        base_volume = test_data['volume'][:T]
        for j in range(N):
            scale = 1.0 + 0.05 * j
            price_tm[:, j] = base_price * scale
            volume_tm[:, j] = (base_volume * (0.7 + 0.1 * j)) + (25.0 * j)

        cpu_tm = np.zeros_like(price_tm)
        for j in range(N):
            cpu_tm[:, j] = ti.VolumeAdjustedMa(
                price_tm[:, j],
                volume_tm[:, j],
                length=length,
                vi_factor=vi_factor,
                strict=strict,
                sample_period=sample_period,
            )

        handle = ti.volume_adjusted_ma_cuda_many_series_one_param_dev(
            price_tm.astype(np.float32),
            volume_tm.astype(np.float32),
            length,
            vi_factor,
            strict=strict,
            sample_period=sample_period,
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle))

        assert gpu_tm.shape == cpu_tm.shape
        assert_close(
            gpu_tm,
            cpu_tm,
            rtol=1e-5,
            atol=1e-6,
            msg="VAMA CUDA many-series vs CPU mismatch",
        )
