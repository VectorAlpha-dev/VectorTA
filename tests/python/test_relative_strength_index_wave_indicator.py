import numpy as np
import pytest

from test_utils import load_test_data

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


def manual_reference(source, high, low, rsi_length=14, length1=2, length2=5, length3=9, length4=13):
    def make_rsi_state(period):
        return {
            "period": period,
            "inv_p": 1.0 / period,
            "beta": 1.0 - 1.0 / period,
            "initialized": False,
            "prev": np.nan,
            "deltas_seen": 0,
            "sum_gain": 0.0,
            "sum_loss": 0.0,
            "avg_gain": 0.0,
            "avg_loss": 0.0,
            "ready": False,
        }

    def update_rsi(state, value):
        if not np.isfinite(value):
            state.update(
                initialized=False,
                prev=np.nan,
                deltas_seen=0,
                sum_gain=0.0,
                sum_loss=0.0,
                avg_gain=0.0,
                avg_loss=0.0,
                ready=False,
            )
            return np.nan
        if not state["initialized"]:
            state["initialized"] = True
            state["prev"] = value
            return np.nan
        delta = value - state["prev"]
        state["prev"] = value
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)
        if not state["ready"]:
            state["sum_gain"] += gain
            state["sum_loss"] += loss
            state["deltas_seen"] += 1
            if state["deltas_seen"] < state["period"]:
                return np.nan
            state["avg_gain"] = state["sum_gain"] * state["inv_p"]
            state["avg_loss"] = state["sum_loss"] * state["inv_p"]
            state["ready"] = True
        else:
            state["avg_gain"] = state["avg_gain"] * state["beta"] + state["inv_p"] * gain
            state["avg_loss"] = state["avg_loss"] * state["beta"] + state["inv_p"] * loss
        denom = state["avg_gain"] + state["avg_loss"]
        return 50.0 if denom == 0.0 else 100.0 * state["avg_gain"] / denom

    def make_wma_state(length):
        return {
            "len": length,
            "denom": float(length * (length + 1) // 2),
            "buf": np.zeros(length, dtype=np.float64),
            "pos": 0,
            "count": 0,
            "sum": 0.0,
            "weighted_sum": 0.0,
        }

    def update_wma(state, value):
        if not np.isfinite(value):
            state["pos"] = 0
            state["count"] = 0
            state["sum"] = 0.0
            state["weighted_sum"] = 0.0
            return np.nan
        if state["count"] < state["len"]:
            state["buf"][state["count"]] = value
            state["count"] += 1
            state["sum"] += value
            state["weighted_sum"] += state["count"] * value
            return state["weighted_sum"] / state["denom"] if state["count"] == state["len"] else np.nan
        old_sum = state["sum"]
        old = state["buf"][state["pos"]]
        state["buf"][state["pos"]] = value
        state["pos"] = (state["pos"] + 1) % state["len"]
        state["weighted_sum"] = state["weighted_sum"] + state["len"] * value - old_sum
        state["sum"] = old_sum + value - old
        return state["weighted_sum"] / state["denom"]

    rsi_source = make_rsi_state(rsi_length)
    rsi_high = make_rsi_state(rsi_length)
    rsi_low = make_rsi_state(rsi_length)
    wma1 = make_wma_state(length1)
    wma2 = make_wma_state(length2)
    wma3 = make_wma_state(length3)
    wma4 = make_wma_state(length4)
    prev_slo = None

    out1 = np.full(source.shape[0], np.nan, dtype=np.float64)
    out2 = np.full(source.shape[0], np.nan, dtype=np.float64)
    out3 = np.full(source.shape[0], np.nan, dtype=np.float64)
    out4 = np.full(source.shape[0], np.nan, dtype=np.float64)
    state = np.full(source.shape[0], np.nan, dtype=np.float64)

    for i in range(source.shape[0]):
        s = float(source[i])
        h = float(high[i])
        l = float(low[i])
        if not np.isfinite(s) or not np.isfinite(h) or not np.isfinite(l):
            r = [rsi_source, rsi_high, rsi_low]
            for item in r:
                item.update(
                    initialized=False,
                    prev=np.nan,
                    deltas_seen=0,
                    sum_gain=0.0,
                    sum_loss=0.0,
                    avg_gain=0.0,
                    avg_loss=0.0,
                    ready=False,
                )
            for item in [wma1, wma2, wma3, wma4]:
                item["pos"] = 0
                item["count"] = 0
                item["sum"] = 0.0
                item["weighted_sum"] = 0.0
            prev_slo = None
            continue

        src_rsi = update_rsi(rsi_source, s)
        high_rsi = update_rsi(rsi_high, h)
        low_rsi = update_rsi(rsi_low, l)
        if not np.isfinite(src_rsi) or not np.isfinite(high_rsi) or not np.isfinite(low_rsi):
            continue

        hlc_rsi = (high_rsi + low_rsi + 2.0 * src_rsi) * 0.25
        out1[i] = update_wma(wma1, hlc_rsi)
        out2[i] = update_wma(wma2, hlc_rsi)
        out3[i] = update_wma(wma3, hlc_rsi)
        out4[i] = update_wma(wma4, hlc_rsi)
        if np.isfinite(out1[i]) and np.isfinite(out2[i]):
            slo = out1[i] - out2[i]
            prev = 0.0 if prev_slo is None else prev_slo
            if slo > 0.0:
                state[i] = 2.0 if slo > prev else 1.0
            elif slo < 0.0:
                state[i] = -2.0 if slo < prev else -1.0
            else:
                state[i] = 0.0
            prev_slo = slo

    return out1, out2, out3, out4, state


class TestRelativeStrengthIndexWaveIndicator:
    def test_manual_reference_matches_binding(self):
        data = load_test_data()
        high = np.asarray(data["high"][:180], dtype=np.float64)
        low = np.asarray(data["low"][:180], dtype=np.float64)
        close = np.asarray(data["close"][:180], dtype=np.float64)
        source = (high + low + 2.0 * close) * 0.25
        exp1, exp2, exp3, exp4, exp_state = manual_reference(source, high, low)
        result = mp.relative_strength_index_wave_indicator(
            source,
            high,
            low,
            rsi_length=14,
            length1=2,
            length2=5,
            length3=9,
            length4=13,
        )
        np.testing.assert_allclose(result["rsi_ma1"], exp1, rtol=0.0, atol=5e-12, equal_nan=True)
        np.testing.assert_allclose(result["rsi_ma2"], exp2, rtol=0.0, atol=5e-12, equal_nan=True)
        np.testing.assert_allclose(result["rsi_ma3"], exp3, rtol=0.0, atol=5e-12, equal_nan=True)
        np.testing.assert_allclose(result["rsi_ma4"], exp4, rtol=0.0, atol=5e-12, equal_nan=True)
        np.testing.assert_allclose(result["state"], exp_state, rtol=0.0, atol=5e-12, equal_nan=True)

    def test_stream_matches_batch(self):
        data = load_test_data()
        high = np.asarray(data["high"][:144], dtype=np.float64)
        low = np.asarray(data["low"][:144], dtype=np.float64)
        close = np.asarray(data["close"][:144], dtype=np.float64)
        source = (high + low + 2.0 * close) * 0.25
        batch = mp.relative_strength_index_wave_indicator(source, high, low)
        stream = mp.RelativeStrengthIndexWaveIndicatorStream(14, 2, 5, 9, 13)
        streamed = np.array(
            [stream.update(float(s), float(h), float(l)) for s, h, l in zip(source, high, low)],
            dtype=np.float64,
        )
        np.testing.assert_allclose(streamed[:, 0], batch["rsi_ma1"], rtol=0.0, atol=5e-12, equal_nan=True)
        np.testing.assert_allclose(streamed[:, 1], batch["rsi_ma2"], rtol=0.0, atol=5e-12, equal_nan=True)
        np.testing.assert_allclose(streamed[:, 2], batch["rsi_ma3"], rtol=0.0, atol=5e-12, equal_nan=True)
        np.testing.assert_allclose(streamed[:, 3], batch["rsi_ma4"], rtol=0.0, atol=5e-12, equal_nan=True)
        np.testing.assert_allclose(streamed[:, 4], batch["state"], rtol=0.0, atol=5e-12, equal_nan=True)

    def test_batch_metadata_and_first_row_match_single(self):
        data = load_test_data()
        high = np.asarray(data["high"][:128], dtype=np.float64)
        low = np.asarray(data["low"][:128], dtype=np.float64)
        close = np.asarray(data["close"][:128], dtype=np.float64)
        source = (high + low + 2.0 * close) * 0.25
        result = mp.relative_strength_index_wave_indicator_batch(
            source,
            high,
            low,
            rsi_length_range=(14, 15, 1),
        )
        assert result["rows"] == 2
        assert result["cols"] == source.shape[0]
        np.testing.assert_allclose(result["rsi_lengths"], np.array([14, 15]), rtol=0.0, atol=0.0)
        single = mp.relative_strength_index_wave_indicator(source, high, low)
        np.testing.assert_allclose(result["rsi_ma1"][0], single["rsi_ma1"], rtol=0.0, atol=5e-12, equal_nan=True)
        np.testing.assert_allclose(result["rsi_ma2"][0], single["rsi_ma2"], rtol=0.0, atol=5e-12, equal_nan=True)
        np.testing.assert_allclose(result["rsi_ma3"][0], single["rsi_ma3"], rtol=0.0, atol=5e-12, equal_nan=True)
        np.testing.assert_allclose(result["rsi_ma4"][0], single["rsi_ma4"], rtol=0.0, atol=5e-12, equal_nan=True)
        np.testing.assert_allclose(result["state"][0], single["state"], rtol=0.0, atol=5e-12, equal_nan=True)

    def test_invalid_parameters(self):
        data = load_test_data()
        high = np.asarray(data["high"][:64], dtype=np.float64)
        low = np.asarray(data["low"][:64], dtype=np.float64)
        close = np.asarray(data["close"][:64], dtype=np.float64)
        source = (high + low + 2.0 * close) * 0.25
        with pytest.raises(ValueError, match="Invalid rsi_length"):
            mp.relative_strength_index_wave_indicator(source, high, low, rsi_length=0)
