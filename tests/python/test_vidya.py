"""
Python binding tests for VIDYA indicator.
Checks reference outputs and error behavior against Rust tests.
"""
import pytest
import numpy as np

try:
    import my_project as ta
except Exception as e:
    pytest.skip("Python module not built. Run 'maturin develop --features python' first", allow_module_level=True)

from test_utils import load_test_data, assert_close


class TestVidya:
    @pytest.fixture(scope="class")
    def close(self):
        candles = load_test_data()
        return candles["close"]

    def test_accuracy_last5_matches_rust(self, close):
        # Rust reference values (last 5 from check_vidya_accuracy)
        expected_last5 = np.array([
            59553.42785306692,
            59503.60445032524,
            59451.72283651444,
            59413.222561244685,
            59239.716526894175,
        ])

        out = ta.vidya(close.astype(np.float64), 2, 5, 0.2)
        assert len(out) == len(close)

        # Compare last 5 with <= Rust tolerance (abs tol 1e-1)
        last5 = np.asarray(out)[-5:]
        assert_close(last5, expected_last5, rtol=0.0, atol=1e-1, msg="VIDYA last5")

    def test_warmup_and_nans(self, close):
        out = ta.vidya(close.astype(np.float64), 2, 5, 0.2)
        # Warmup = first_valid(0) + long - 2 = 3
        assert np.isnan(out[0]) and np.isnan(out[1]) and np.isnan(out[2])
        assert not np.any(np.isnan(out[3:])), "No NaN after warmup"

    def test_error_handling(self):
        data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        with pytest.raises(Exception):
            ta.vidya(np.array([], dtype=np.float64), 2, 5, 0.2)
        with pytest.raises(Exception):
            ta.vidya(data, 0, 5, 0.2)          # invalid short
        with pytest.raises(Exception):
            ta.vidya(data, 3, 2, 0.2)          # short > long
        with pytest.raises(Exception):
            ta.vidya(data, 2, 5, -0.1)         # alpha < 0
        with pytest.raises(Exception):
            ta.vidya(data, 2, 5, 1.1)          # alpha > 1
        with pytest.raises(Exception):
            ta.vidya(np.array([1.0, 2.0], dtype=np.float64), 2, 5, 0.2)  # data too short

    def test_reinput_len_matches(self, close):
        first = ta.vidya(close.astype(np.float64), 2, 5, 0.2)
        second = ta.vidya(np.asarray(first, dtype=np.float64), 2, 5, 0.2)
        assert len(first) == len(second) == len(close)


if __name__ == "__main__":
    pytest.main([__file__, "-q"]) 
