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


def manual_reference(data: np.ndarray, domestic_cycle_length: int = 15):
    out = np.full(data.shape[0], np.nan, dtype=np.float64)
    if domestic_cycle_length <= 0:
        return out
    angle = 2.0 * np.pi / float(domestic_cycle_length)
    for i in range(data.shape[0]):
        if i + 1 < domestic_cycle_length:
            continue
        real = 0.0
        imag = 0.0
        valid = True
        for j in range(domestic_cycle_length):
            value = float(data[i - j])
            if not np.isfinite(value):
                valid = False
                break
            theta = angle * float(j)
            real += np.cos(theta) * value
            imag += np.sin(theta) * value
        if not valid:
            continue
        phase = np.degrees(np.arctan(imag / real)) if abs(real) > 0.001 else (90.0 if imag > 0.0 else -90.0 if imag < 0.0 else 0.0)
        if real < 0.0:
            phase += 180.0
        phase += 90.0
        if phase < 0.0:
            phase += 360.0
        if phase > 360.0:
            phase -= 360.0
        out[i] = phase
    return out


class TestL1EhlersPhasor:
    def test_manual_reference_matches_binding(self):
        data = load_test_data()
        source = np.asarray(data["close"][:160], dtype=np.float64)
        expected = manual_reference(source, 15)
        result = mp.l1_ehlers_phasor(source, domestic_cycle_length=15)
        np.testing.assert_allclose(result, expected, rtol=0.0, atol=5e-11, equal_nan=True)

    def test_stream_matches_batch(self):
        data = load_test_data()
        source = np.asarray(data["close"][:144], dtype=np.float64)
        batch = mp.l1_ehlers_phasor(source, domestic_cycle_length=15)
        stream = mp.L1EhlersPhasorStream(15)
        streamed = np.array([stream.update(float(x)) for x in source], dtype=np.float64)
        np.testing.assert_allclose(streamed, batch, rtol=0.0, atol=5e-11, equal_nan=True)

    def test_batch_metadata_and_first_row_match_single(self):
        data = load_test_data()
        source = np.asarray(data["close"][:128], dtype=np.float64)
        result = mp.l1_ehlers_phasor_batch(
            source,
            domestic_cycle_length_range=(15, 17, 2),
        )
        assert result["rows"] == 2
        assert result["cols"] == source.shape[0]
        np.testing.assert_allclose(
            result["domestic_cycle_lengths"],
            np.array([15, 17], dtype=np.uint64),
            rtol=0.0,
            atol=0.0,
        )
        single = mp.l1_ehlers_phasor(source, domestic_cycle_length=15)
        np.testing.assert_allclose(
            result["values"][0], single, rtol=0.0, atol=5e-11, equal_nan=True
        )

    def test_invalid_length(self):
        data = load_test_data()
        source = np.asarray(data["close"][:64], dtype=np.float64)
        with pytest.raises(ValueError, match="Invalid domestic_cycle_length"):
            mp.l1_ehlers_phasor(source, domestic_cycle_length=0)
