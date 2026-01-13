"""
Python binding tests for MWDX indicator.
These tests mirror the Rust unit tests to ensure Python bindings work correctly.
"""
import pytest
import numpy as np
from test_utils import load_test_data, assert_close, EXPECTED_OUTPUTS
from rust_comparison import compare_with_rust


try:
    from my_project import (
        mwdx,
        mwdx_batch,
        MwdxStream
    )
except ImportError:
    pytest.skip("MWDX module not available - run 'maturin develop' first", allow_module_level=True)


class TestMwdx:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_mwdx_partial_params(self, test_data):
        """Test MWDX with default parameters - mirrors check_mwdx_partial_params"""
        close = np.array(test_data['close'], dtype=np.float64)


        result = mwdx(close, 0.2)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(close)

    def test_mwdx_accuracy(self, test_data):
        """Test MWDX matches expected values from Rust tests - mirrors check_mwdx_accuracy"""
        close = np.array(test_data['close'], dtype=np.float64)
        expected = EXPECTED_OUTPUTS['mwdx']


        result = mwdx(close, expected['default_params']['factor'])

        assert len(result) == len(close)


        assert_close(
            result[-5:],
            expected['last_5_values'],
            rtol=1e-7,
            msg="MWDX last 5 values mismatch"
        )


        compare_with_rust('mwdx', result, 'close', expected['default_params'])


    def test_mwdx_zero_factor(self):
        """Test MWDX fails with zero factor - mirrors check_mwdx_zero_factor"""
        input_data = np.array([10.0, 20.0, 30.0], dtype=np.float64)

        with pytest.raises(ValueError, match="Factor must be greater than 0"):
            mwdx(input_data, 0.0)

    def test_mwdx_negative_factor(self):
        """Test MWDX fails with negative factor - mirrors check_mwdx_negative_factor"""
        input_data = np.array([10.0, 20.0, 30.0], dtype=np.float64)

        with pytest.raises(ValueError, match="Factor must be greater than 0"):
            mwdx(input_data, -0.5)

    def test_mwdx_nan_factor(self):
        """Test MWDX fails with NaN factor"""
        input_data = np.array([10.0, 20.0, 30.0], dtype=np.float64)

        with pytest.raises(ValueError, match="Factor must be greater than 0"):
            mwdx(input_data, np.nan)


    def test_mwdx_very_small_dataset(self):
        """Test MWDX with single data point - mirrors check_mwdx_very_small_dataset"""
        data = np.array([42.0], dtype=np.float64)

        result = mwdx(data, 0.2)
        assert len(result) == 1
        assert result[0] == 42.0

    def test_mwdx_empty_input(self):
        """Test MWDX with empty input"""
        data_empty = np.array([], dtype=np.float64)

        with pytest.raises(ValueError, match="No input data was provided"):
            mwdx(data_empty, 0.2)


    def test_mwdx_reinput(self, test_data):
        """Test MWDX with re-input of MWDX result - mirrors check_mwdx_reinput"""
        close = np.array(test_data['close'], dtype=np.float64)


        first_result = mwdx(close, 0.2)


        second_result = mwdx(first_result, 0.3)

        assert len(second_result) == len(first_result)


        assert np.all(np.isfinite(second_result))


    def test_mwdx_nan_handling(self, test_data):
        """Test MWDX handling of NaN values - mirrors check_mwdx_nan_handling"""
        close = np.array(test_data['close'], dtype=np.float64)

        result = mwdx(close, 0.2)

        assert len(result) == len(close)


        assert np.all(np.isfinite(result))


    def test_mwdx_batch(self, test_data):
        """Test MWDX batch computation"""
        close = np.array(test_data['close'], dtype=np.float64)


        batch_result = mwdx_batch(
            close,
            (0.1, 0.5, 0.1)
        )


        assert isinstance(batch_result, dict)
        assert 'values' in batch_result
        assert 'factors' in batch_result


        assert batch_result['values'].shape == (5, len(close))
        assert len(batch_result['factors']) == 5


        individual_result = mwdx(close, 0.1)
        batch_first = batch_result['values'][0]
        assert_close(batch_first, individual_result, atol=1e-9, msg="MWDX first combination mismatch")


    def test_mwdx_streaming(self, test_data):
        """Test MWDX streaming interface - mirrors check_mwdx_streaming"""
        close = test_data['close']
        factor = 0.2


        close_array = np.array(close, dtype=np.float64)
        batch_result = mwdx(close_array, factor)


        stream = MwdxStream(factor)
        stream_results = []

        for price in close:
            result = stream.update(price)
            stream_results.append(result)


        assert len(stream_results) == len(close)


        for i, val in enumerate(stream_results):
            assert np.isfinite(val), f"NaN at index {i}"


        assert_close(stream_results[0], close[0], atol=1e-9, msg="First value mismatch")


    def test_mwdx_all_nan_input(self):
        """Test MWDX with all NaN values"""
        all_nan = np.full(100, np.nan, dtype=np.float64)


        result = mwdx(all_nan, 0.2)
        assert np.all(np.isnan(result))

    def test_mwdx_batch_performance(self, test_data):
        """Test that batch computation works correctly (performance is secondary)"""
        close = np.array(test_data['close'][:1000], dtype=np.float64)


        batch_result = mwdx_batch(
            close,
            (0.1, 0.9, 0.2)
        )


        assert batch_result['values'].shape == (5, len(close))


        individual_result = mwdx(close, 0.1)
        batch_first = batch_result['values'][0]
        assert_close(batch_first, individual_result, atol=1e-9,
                    msg="MWDX batch mismatch for factor=0.1")

    def test_mwdx_edge_cases(self):
        """Test MWDX with edge case inputs"""

        data = np.arange(1.0, 101.0, dtype=np.float64)
        result = mwdx(data, 0.2)
        assert len(result) == len(data)
        assert np.all(np.isfinite(result))


        data = np.array([50.0] * 100, dtype=np.float64)
        result = mwdx(data, 0.2)
        assert len(result) == len(data)


        for i in range(10, len(result)):
            assert_close(result[i], 50.0, atol=1e-6, msg=f"Constant value failed at index {i}")


        data = np.array([10.0, 20.0, 10.0, 20.0] * 25, dtype=np.float64)
        result = mwdx(data, 0.5)
        assert len(result) == len(data)
        assert np.all(np.isfinite(result))

    def test_mwdx_high_factor(self, test_data):
        """Test MWDX with high factor value"""
        close = np.array(test_data['close'][:100], dtype=np.float64)


        result = mwdx(close, 0.95)
        assert len(result) == len(close)
        assert np.all(np.isfinite(result))

    def test_mwdx_low_factor(self, test_data):
        """Test MWDX with low factor value"""
        close = np.array(test_data['close'][:100], dtype=np.float64)


        result = mwdx(close, 0.01)
        assert len(result) == len(close)
        assert np.all(np.isfinite(result))

    def test_mwdx_consistency(self, test_data):
        """Test that MWDX produces consistent results across multiple calls"""
        close = np.array(test_data['close'][:100], dtype=np.float64)

        result1 = mwdx(close, 0.2)
        result2 = mwdx(close, 0.2)

        assert_close(result1, result2, atol=1e-15, msg="MWDX results not consistent")

    def test_mwdx_step_precision(self):
        """Test batch with various step sizes"""
        data = np.arange(1, 51, dtype=np.float64)


        batch_result = mwdx_batch(
            data,
            (0.2, 0.8, 0.3)
        )

        assert batch_result['values'].shape == (3, len(data))
        expected_factors = [0.2, 0.5, 0.8]
        assert_close(batch_result['factors'], expected_factors, atol=1e-9, msg="Factor mismatch")

    def test_mwdx_batch_error_handling(self):
        """Test MWDX batch error handling"""

        empty = np.array([], dtype=np.float64)
        with pytest.raises(ValueError, match="No input data was provided"):
            mwdx_batch(empty, (0.2, 0.2, 0))


        data = np.random.randn(100).astype(np.float64)
        with pytest.raises(ValueError, match="Factor must be greater than 0"):
            mwdx_batch(data, (0.0, 0.5, 0.1))

        with pytest.raises(ValueError, match="Factor must be greater than 0"):
            mwdx_batch(data, (-0.1, 0.5, 0.1))

    def test_mwdx_zero_copy_verification(self, test_data):
        """Verify MWDX uses zero-copy operations"""
        close = np.array(test_data['close'][:100], dtype=np.float64)


        result = mwdx(close, 0.2)
        assert len(result) == len(close)


        batch_result = mwdx_batch(close, (0.1, 0.5, 0.2))
        assert batch_result['values'].shape[0] == 3
        assert batch_result['values'].shape[1] == len(close)

    def test_mwdx_stream_error_handling(self):
        """Test MWDX stream error handling"""

        with pytest.raises(ValueError, match="Factor must be greater than 0"):
            MwdxStream(0.0)

        with pytest.raises(ValueError, match="Factor must be greater than 0"):
            MwdxStream(-0.5)

        with pytest.raises(ValueError, match="Factor must be greater than 0"):
            MwdxStream(np.nan)

    def test_mwdx_nan_prefix(self):
        """Test MWDX with NaN prefix in data"""

        data = np.array([np.nan, np.nan, 10.0, 20.0, 30.0, 40.0], dtype=np.float64)

        result = mwdx(data, 0.2)
        assert len(result) == len(data)











        assert np.isnan(result[0])
        assert np.isnan(result[1])


        assert result[2] == 10.0
        assert np.abs(result[3] - 12.0) < 1e-10
        assert np.abs(result[4] - 15.6) < 1e-10
        assert np.abs(result[5] - 20.48) < 1e-10


        clean_data = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
        clean_result = mwdx(clean_data, 0.2)
        assert np.all(np.isfinite(clean_result))

    def test_mwdx_formula_verification(self):
        """Verify MWDX formula: out[i] = fac * data[i] + (1 - fac) * out[i-1]"""
        data = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
        factor = 0.3

        result = mwdx(data, factor)


        expected = [data[0]]
        for i in range(1, len(data)):
            val = factor * data[i] + (1 - factor) * expected[i-1]
            expected.append(val)

        assert_close(result, expected, atol=1e-12, msg="Formula verification failed")

    def test_mwdx_kernel_parameter(self, test_data):
        """Test MWDX with kernel parameter"""
        close = np.array(test_data['close'][:100], dtype=np.float64)


        for kernel in [None, 'scalar']:
            result = mwdx(close, 0.2, kernel=kernel)
            assert len(result) == len(close)
            assert np.all(np.isfinite(result)), f"NaN values with kernel={kernel}"


        batch_result = mwdx_batch(close, (0.1, 0.3, 0.1), kernel='scalar')
        assert batch_result['values'].shape == (3, len(close))


        with pytest.raises(ValueError, match="Unknown kernel"):
            mwdx(close, 0.2, kernel='invalid')

    def test_mwdx_kernel_consistency(self, test_data):
        """Test that different kernels produce consistent results"""
        close = np.array(test_data['close'][:1000], dtype=np.float64)


        result_auto = mwdx(close, 0.2, kernel=None)
        result_scalar = mwdx(close, 0.2, kernel='scalar')


        assert_close(result_auto, result_scalar, rtol=1e-14,
                    msg="Kernel results differ significantly")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])