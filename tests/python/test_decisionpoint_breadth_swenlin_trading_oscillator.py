import numpy as np
import pytest

try:
    import vector_ta as mp
except ImportError:
    try:
        import vector_ta as mp
    except ImportError:
        pytest.skip(
            "Python module not built. Run 'maturin develop --features python' first",
            allow_module_level=True,
        )


def sample_breadth(length: int):
    advancing = np.empty(length, dtype=np.float64)
    declining = np.empty(length, dtype=np.float64)
    for i in range(length):
        x = float(i)
        advancing[i] = 1500.0 + x * 0.8 + np.sin(x * 0.07) * 120.0 + 40.0
        declining[i] = 1300.0 + x * 0.5 + np.cos(x * 0.05) * 95.0 + 30.0
    return advancing, declining


def test_decisionpoint_breadth_swenlin_trading_oscillator_output_contract():
    advancing, declining = sample_breadth(256)
    values = mp.decisionpoint_breadth_swenlin_trading_oscillator(advancing, declining)

    assert len(values) == len(advancing)
    assert int(np.flatnonzero(~np.isnan(values))[0]) == 4
    assert np.isfinite(values[-1])


def test_decisionpoint_breadth_swenlin_trading_oscillator_rejects_invalid_input():
    advancing, declining = sample_breadth(16)

    with pytest.raises(ValueError, match="Inconsistent slice lengths"):
        mp.decisionpoint_breadth_swenlin_trading_oscillator(advancing[:-1], declining)


def test_decisionpoint_breadth_swenlin_trading_oscillator_stream_matches_batch_with_reset():
    advancing, declining = sample_breadth(180)
    advancing[90] = 0.0
    declining[90] = 0.0

    batch = mp.decisionpoint_breadth_swenlin_trading_oscillator(advancing, declining)
    stream = mp.DecisionPointBreadthSwenlinTradingOscillatorStream()
    streamed = []
    for adv, dec in zip(advancing, declining):
        out = stream.update(float(adv), float(dec))
        streamed.append(np.nan if out is None else out)

    np.testing.assert_allclose(
        np.asarray(streamed), batch, rtol=1e-12, atol=1e-12, equal_nan=True
    )


def test_decisionpoint_breadth_swenlin_trading_oscillator_batch_matches_single():
    advancing, declining = sample_breadth(192)
    batch = mp.decisionpoint_breadth_swenlin_trading_oscillator_batch(advancing, declining)
    values = mp.decisionpoint_breadth_swenlin_trading_oscillator(advancing, declining)

    assert batch["rows"] == 1
    assert batch["cols"] == len(advancing)
    np.testing.assert_allclose(batch["values"][0], values, rtol=1e-12, atol=1e-12, equal_nan=True)
