import numpy as np
import pytest

from test_utils import load_test_data

try:
    import vector_ta as mp
except ImportError:
    try:
        import my_project as mp
    except ImportError:
        pytest.skip(
            "Python module not built. Run 'maturin develop --features python' first",
            allow_module_level=True,
        )


def manual_sma(values: np.ndarray, length: int) -> np.ndarray:
    out = np.full(values.shape[0], np.nan, dtype=np.float64)
    for i in range(length - 1, values.shape[0]):
        window = values[i + 1 - length : i + 1]
        if np.all(np.isfinite(window)):
            out[i] = np.mean(window)
    return out


def manual_mesa(source: np.ndarray, length: int) -> np.ndarray:
    n = source.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    hp = np.full(n, np.nan, dtype=np.float64)
    filt = np.full(n, np.nan, dtype=np.float64)

    pi = 3.14
    alpha1 = (np.cos(0.707 * 2.0 * pi / 48.0) + np.sin(0.707 * 2.0 * pi / 48.0) - 1.0) / np.cos(
        0.707 * 2.0 * pi / 48.0
    )
    hp_coef = (1.0 - alpha1 * 0.5) * (1.0 - alpha1 * 0.5)
    hp_feedback_1 = 2.0 * (1.0 - alpha1)
    hp_feedback_2 = -((1.0 - alpha1) * (1.0 - alpha1))
    a1 = np.exp(-1.414 * pi / 10.0)
    b1 = 2.0 * a1 * np.cos(1.414 * pi / 10.0)
    c2 = b1
    c3 = -(a1 * a1)
    c1 = 1.0 - c2 - c3

    def nz(values: np.ndarray, idx: int) -> float:
        if idx < 0:
            return 0.0
        value = values[idx]
        return value if np.isfinite(value) else 0.0

    for i in range(n):
        if not np.isfinite(source[i]):
            continue
        hp[i] = hp_coef * (source[i] - 2.0 * nz(source, i - 1) + nz(source, i - 2))
        hp[i] += hp_feedback_1 * nz(hp, i - 1) + hp_feedback_2 * nz(hp, i - 2)
        filt[i] = c1 * hp[i] + c2 * nz(filt, i - 1) + c3 * nz(filt, i - 2)

    for i in range(n):
        if not np.isfinite(filt[i]):
            continue
        highest = filt[i]
        lowest = filt[i]
        for count in range(length):
            value = nz(filt, i - count)
            highest = max(highest, value)
            lowest = min(lowest, value)
        denom = highest - lowest
        if denom == 0.0 or not np.isfinite(denom):
            continue
        stoc = (filt[i] - lowest) / denom
        if np.isfinite(stoc):
            out[i] = c1 * stoc + c2 * nz(out, i - 1) + c3 * nz(out, i - 2)
    return out


def manual_reference(
    source: np.ndarray,
    length_1: int = 48,
    length_2: int = 21,
    length_3: int = 9,
    length_4: int = 6,
    trigger_length: int = 2,
):
    mesa_1 = manual_mesa(source, length_1)
    mesa_2 = manual_mesa(source, length_2)
    mesa_3 = manual_mesa(source, length_3)
    mesa_4 = manual_mesa(source, length_4)
    return {
        "mesa_1": mesa_1,
        "mesa_2": mesa_2,
        "mesa_3": mesa_3,
        "mesa_4": mesa_4,
        "trigger_1": manual_sma(mesa_1, trigger_length),
        "trigger_2": manual_sma(mesa_2, trigger_length),
        "trigger_3": manual_sma(mesa_3, trigger_length),
        "trigger_4": manual_sma(mesa_4, trigger_length),
    }


class TestMesaStochasticMultiLength:
    def test_manual_reference_matches_binding(self):
        data = load_test_data()
        source = np.asarray(data["close"][:180], dtype=np.float64)
        expected = manual_reference(source, 48, 21, 9, 6, 2)
        result = mp.mesa_stochastic_multi_length(
            source,
            length_1=48,
            length_2=21,
            length_3=9,
            length_4=6,
            trigger_length=2,
        )
        for key, expected_values in expected.items():
            np.testing.assert_allclose(
                np.asarray(result[key], dtype=np.float64),
                expected_values,
                rtol=0.0,
                atol=1e-12,
                equal_nan=True,
            )

    def test_stream_matches_batch(self):
        data = load_test_data()
        source = np.asarray(data["close"][:160], dtype=np.float64)
        batch = mp.mesa_stochastic_multi_length(source)
        stream = mp.MesaStochasticMultiLengthStream(48, 21, 9, 6, 2)
        last = None
        for value in source:
            last = stream.update(float(value))
        assert last is not None
        for key in batch:
            assert np.isclose(last[key], batch[key][-1], equal_nan=True)

    def test_batch_metadata_and_first_row_match_single(self):
        data = load_test_data()
        source = np.asarray(data["close"][:160], dtype=np.float64)
        result = mp.mesa_stochastic_multi_length_batch(
            source,
            length_1_range=(48, 50, 2),
            length_2_range=(21, 21, 0),
            length_3_range=(9, 9, 0),
            length_4_range=(6, 6, 0),
            trigger_length_range=(2, 2, 0),
        )
        assert result["rows"] == 2
        assert result["cols"] == source.shape[0]
        np.testing.assert_array_equal(result["length_1"], np.array([48, 50], dtype=np.int64))
        single = mp.mesa_stochastic_multi_length(source)
        np.testing.assert_allclose(result["mesa_1"][0], single["mesa_1"], rtol=0.0, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(
            result["trigger_1"][0], single["trigger_1"], rtol=0.0, atol=1e-12, equal_nan=True
        )

    def test_kernel_parity(self):
        data = load_test_data()
        source = np.asarray(data["close"][:160], dtype=np.float64)
        auto = mp.mesa_stochastic_multi_length(source)
        scalar = mp.mesa_stochastic_multi_length(source, kernel="scalar")
        for key in auto:
            np.testing.assert_allclose(auto[key], scalar[key], rtol=0.0, atol=1e-12, equal_nan=True)

    def test_invalid_period(self):
        data = load_test_data()
        source = np.asarray(data["close"][:64], dtype=np.float64)
        with pytest.raises(ValueError, match="Invalid period"):
            mp.mesa_stochastic_multi_length(source, length_1=0)
