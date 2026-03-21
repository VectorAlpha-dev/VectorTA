import pytest
import numpy as np

try:
    import vector_ta as ti
except ImportError:
    pytest.skip("Python module not built. Run 'maturin develop --features python' first", allow_module_level=True)

from test_utils import load_test_data


class TestPatternRecognition:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_pattern_recognition_output_contract(self, test_data):
        open_ = test_data['open']
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']

        out = ti.pattern_recognition(open_, high, low, close)

        assert 'values' in out
        assert 'pattern_ids' in out
        assert 'rows' in out
        assert 'cols' in out
        assert 'warmup' in out

        values = out['values']
        pattern_ids = out['pattern_ids']

        assert values.shape == (out['rows'], out['cols'])
        assert values.dtype == np.uint8
        assert out['cols'] == len(close)
        assert len(pattern_ids) == out['rows']
        assert out['rows'] > 0
        assert 'cdldoji' in pattern_ids

        uniques = np.unique(values)
        assert np.all(np.isin(uniques, [0, 1]))

    def test_pattern_recognition_kernel_parity(self, test_data):
        open_ = test_data['open']
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']

        auto_out = ti.pattern_recognition(open_, high, low, close)
        scalar_out = ti.pattern_recognition(open_, high, low, close, kernel='scalar')

        np.testing.assert_array_equal(auto_out['values'], scalar_out['values'])
        assert auto_out['pattern_ids'] == scalar_out['pattern_ids']
        assert auto_out['rows'] == scalar_out['rows']
        assert auto_out['cols'] == scalar_out['cols']

    def test_pattern_recognition_deterministic(self, test_data):
        open_ = test_data['open']
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']

        out1 = ti.pattern_recognition(open_, high, low, close)
        out2 = ti.pattern_recognition(open_, high, low, close)

        np.testing.assert_array_equal(out1['values'], out2['values'])
        assert out1['pattern_ids'] == out2['pattern_ids']

    def test_pattern_recognition_invalid_kernel(self, test_data):
        open_ = test_data['open']
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']

        with pytest.raises(ValueError):
            ti.pattern_recognition(open_, high, low, close, kernel='invalid_kernel')

    def test_pattern_recognition_length_mismatch(self, test_data):
        open_ = test_data['open']
        high = test_data['high']
        low = test_data['low']
        close = test_data['close']

        with pytest.raises(ValueError, match='length mismatch'):
            ti.pattern_recognition(open_[:-1], high, low, close)

    def test_pattern_recognition_empty_input(self):
        empty = np.array([], dtype=np.float64)

        with pytest.raises(ValueError, match='Length=0|Not enough data points'):
            ti.pattern_recognition(empty, empty, empty, empty)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
