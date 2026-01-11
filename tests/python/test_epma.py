"""
Python binding tests for EPMA indicator.
These tests mirror the Rust unit tests and use identical reference values
and tolerance (absolute 1e-1 on last-5 checks).
"""
import pytest
import numpy as np

try:
    import my_project as ti
except ImportError:
    pytest.skip("Python module not built. Run 'maturin develop --features python' first", allow_module_level=True)

from test_utils import load_test_data, assert_close, EXPECTED_OUTPUTS
from rust_comparison import compare_with_rust


class TestEpma:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_epma_accuracy(self, test_data):
        close = test_data['close']
        exp = EXPECTED_OUTPUTS['epma']

        out = ti.epma(close, exp['default_params']['period'], exp['default_params']['offset'])
        assert len(out) == len(close)

        
        assert_close(
            out[-5:],
            exp['last_5_values'],
            rtol=0.0,
            atol=1e-1,
            msg="EPMA last 5 values mismatch",
        )

        
        compare_with_rust('epma', out, 'close', exp['default_params'])

    def test_epma_invalid_params(self):
        data = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        with pytest.raises(ValueError):
            ti.epma(data, 0, 4)
        with pytest.raises(ValueError):
            ti.epma(data, 10, 0)

    def test_epma_empty_and_nan_input(self):
        empty = np.array([], dtype=np.float64)
        with pytest.raises(ValueError):
            ti.epma(empty, 11, 4)

        all_nan = np.full(64, np.nan, dtype=np.float64)
        with pytest.raises(ValueError):
            ti.epma(all_nan, 11, 4)

    def test_epma_streaming(self, test_data):
        close = test_data['close']
        exp = EXPECTED_OUTPUTS['epma']
        p = exp['default_params']['period']
        o = exp['default_params']['offset']

        batch = ti.epma(close, p, o)
        stream = ti.EpmaStream(p, o)
        stream_vals = []
        for v in close:
            x = stream.update(float(v))
            stream_vals.append(np.nan if x is None else x)

        assert len(batch) == len(stream_vals)
        
        for i in range(len(batch)):
            if not (np.isnan(batch[i]) or np.isnan(stream_vals[i])):
                assert abs(batch[i] - stream_vals[i]) < 1e-9, \
                    f"Streaming mismatch at {i}: batch={batch[i]}, stream={stream_vals[i]}"

    def test_epma_batch(self, test_data):
        close = test_data['close']
        exp = EXPECTED_OUTPUTS['epma']
        p, o = exp['default_params']['period'], exp['default_params']['offset']

        
        batch = ti.epma_batch(close, (p, p, 1), (o, o, 1))
        assert 'values' in batch and 'periods' in batch and 'offsets' in batch
        assert batch['values'].shape == (1, len(close))
        assert list(batch['periods']) == [p]
        assert list(batch['offsets']) == [o]

        
        assert_close(
            batch['values'][0, -5:],
            exp['last_5_values'],
            rtol=0.0,
            atol=1e-1,
            msg='EPMA batch default row mismatch',
        )

        
        
        batch2 = ti.epma_batch(close[:200], (5, 11, 3), (1, 3, 2))  
        rows = len(batch2['periods'])
        assert batch2['values'].shape[0] == rows
        assert rows == len(batch2['offsets'])
        
        for i in range(rows):
            pi = int(batch2['periods'][i])
            oi = int(batch2['offsets'][i])
            single = ti.epma(close[:200], pi, oi)
            assert_close(batch2['values'][i], single, rtol=1e-10, atol=1e-12,
                         msg=f'EPMA batch row mismatch for period={pi}, offset={oi}')
