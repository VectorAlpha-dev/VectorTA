import pytest
import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    import my_project as ti
except ImportError:
    pytest.skip("Python module not built. Run 'maturin develop --features python,cuda' first", allow_module_level=True)

from test_utils import load_test_data


def _cuda_available() -> bool:
    if not hasattr(ti, 'pattern_recognition_cuda_host_f32'):
        return False
    try:
        n = 128
        base = np.linspace(100.0, 110.0, n, dtype=np.float32)
        open_ = base
        high = base + 1.0
        low = base - 1.0
        close = base + 0.2
        _ = ti.pattern_recognition_cuda_host_f32(open_, high, low, close)
        return True
    except Exception as exc:
        msg = str(exc).lower()
        if 'cuda not available' in msg or 'ptx' in msg or 'nvcc' in msg:
            return False
        return True


def _cuda_device_available() -> bool:
    if cp is None:
        return False
    if not hasattr(ti, 'pattern_recognition_cuda_batch_dev'):
        return False
    try:
        n = 128
        base = np.linspace(100.0, 110.0, n, dtype=np.float32)
        open_ = base
        high = base + 1.0
        low = base - 1.0
        close = base + 0.2
        handle = ti.pattern_recognition_cuda_batch_dev(open_, high, low, close)
        _ = cp.asarray(handle)
        return True
    except Exception as exc:
        msg = str(exc).lower()
        if 'cuda not available' in msg or 'ptx' in msg or 'nvcc' in msg:
            return False
        return True


class TestPatternRecognitionCuda:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    @pytest.mark.skipif(not _cuda_available(), reason='CUDA host path unavailable')
    def test_pattern_recognition_cuda_host_contract(self, test_data):
        n = 512
        open_ = test_data['open'][:n].astype(np.float32)
        high = test_data['high'][:n].astype(np.float32)
        low = test_data['low'][:n].astype(np.float32)
        close = test_data['close'][:n].astype(np.float32)

        out = ti.pattern_recognition_cuda_host_f32(open_, high, low, close)

        assert 'values' in out
        assert 'pattern_ids' in out
        assert 'rows' in out
        assert 'cols' in out
        assert 'warmup' in out

        values = out['values']
        pattern_ids = out['pattern_ids']
        assert values.shape == (out['rows'], out['cols'])
        assert values.dtype == np.float32
        assert out['cols'] == n
        assert len(pattern_ids) == out['rows']
        assert 'cdldoji' in pattern_ids

        uniques = np.unique(values)
        assert np.all(np.isin(uniques, [0.0, 1.0]))

    @pytest.mark.skipif(not _cuda_device_available(), reason='CUDA device path unavailable')
    def test_pattern_recognition_cuda_device_matches_host(self, test_data):
        n = 512
        open_ = test_data['open'][:n].astype(np.float32)
        high = test_data['high'][:n].astype(np.float32)
        low = test_data['low'][:n].astype(np.float32)
        close = test_data['close'][:n].astype(np.float32)

        host = ti.pattern_recognition_cuda_host_f32(open_, high, low, close)
        handle = ti.pattern_recognition_cuda_batch_dev(open_, high, low, close)
        gpu = cp.asnumpy(cp.asarray(handle))

        assert gpu.shape == host['values'].shape
        np.testing.assert_array_equal(gpu, host['values'])

    @pytest.mark.skipif(not _cuda_available(), reason='CUDA host path unavailable')
    def test_pattern_recognition_cuda_host_matches_cpu(self, test_data):
        n = 512
        open_f32 = test_data['open'][:n].astype(np.float32)
        high_f32 = test_data['high'][:n].astype(np.float32)
        low_f32 = test_data['low'][:n].astype(np.float32)
        close_f32 = test_data['close'][:n].astype(np.float32)

        cpu = ti.pattern_recognition(
            open_f32.astype(np.float64),
            high_f32.astype(np.float64),
            low_f32.astype(np.float64),
            close_f32.astype(np.float64),
        )
        host = ti.pattern_recognition_cuda_host_f32(open_f32, high_f32, low_f32, close_f32)

        assert host['values'].shape == cpu['values'].shape
        cpu_f32 = cpu['values'].astype(np.float32)
        mismatches = np.count_nonzero(host['values'] != cpu_f32)
        ratio = mismatches / cpu_f32.size
        assert ratio <= 0.01, f"mismatch ratio too high: {mismatches}/{cpu_f32.size} ({ratio:.6f})"
        assert host['pattern_ids'] == cpu['pattern_ids']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
