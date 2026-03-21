import numpy as np
import pytest

try:
    import vector_ta as ta_indicators
except ImportError:
    try:
        import vector_ta as ta_indicators
    except ImportError:
        pytest.skip("Python module not built.", allow_module_level=True)


def naive_evwma(prices, volumes, length=30, absolute_volume_millions=134.0, use_volume_sum=False):
    prices = np.asarray(prices, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    out = np.full(prices.shape, np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(prices) & np.isfinite(volumes))
    if valid.size == 0:
        return out

    first = int(valid[0])
    prev = np.nan
    absolute_volume = absolute_volume_millions * 1_000_000.0
    ring = np.zeros(length, dtype=float)
    rolling_sum = 0.0
    count = 0
    head = 0

    for idx in range(first, len(prices)):
        price = prices[idx]
        volume = volumes[idx]
        if use_volume_sum:
            volume_value = volume if np.isfinite(volume) else 0.0
            if count < length:
                ring[count] = volume_value
                rolling_sum += volume_value
                count += 1
            else:
                rolling_sum += volume_value - ring[head]
                ring[head] = volume_value
                head = (head + 1) % length
        volume_period = rolling_sum if use_volume_sum else absolute_volume
        if (
            not np.isfinite(price)
            or not np.isfinite(volume)
            or not np.isfinite(volume_period)
            or volume_period == 0.0
        ):
            prev = np.nan
            continue
        base = prev if np.isfinite(prev) else price
        value = ((volume_period - volume) * base + volume * price) / volume_period
        out[idx] = value
        prev = value

    return out


def sample_series(length=96):
    prices = np.array(
        [100.0 + i * 0.23 + np.sin(i * 0.17) for i in range(length)],
        dtype=float,
    )
    volumes = np.array(
        [1000.0 + (i % 9) * 57.0 + i * 2.0 for i in range(length)],
        dtype=float,
    )
    return prices, volumes


def assert_series_close(actual, expected):
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    assert actual.shape == expected.shape
    mask = ~(np.isnan(actual) & np.isnan(expected))
    assert np.allclose(actual[mask], expected[mask], atol=1e-10, rtol=0.0)


def test_evwma_matches_naive_absolute_mode():
    prices, volumes = sample_series()
    actual = ta_indicators.elastic_volume_weighted_moving_average(prices, volumes)
    expected = naive_evwma(prices, volumes)
    assert_series_close(actual, expected)


def test_evwma_matches_naive_volume_sum_mode():
    prices, volumes = sample_series()
    actual = ta_indicators.elastic_volume_weighted_moving_average(
        prices,
        volumes,
        length=7,
        absolute_volume_millions=200.0,
        use_volume_sum=True,
    )
    expected = naive_evwma(
        prices,
        volumes,
        length=7,
        absolute_volume_millions=200.0,
        use_volume_sum=True,
    )
    assert_series_close(actual, expected)


def test_evwma_stream_matches_batch():
    prices, volumes = sample_series(80)
    batch = ta_indicators.elastic_volume_weighted_moving_average(
        prices,
        volumes,
        length=9,
        absolute_volume_millions=220.0,
        use_volume_sum=True,
    )
    stream = ta_indicators.ElasticVolumeWeightedMovingAverageStream(
        length=9,
        absolute_volume_millions=220.0,
        use_volume_sum=True,
    )
    streamed = []
    for price, volume in zip(prices, volumes):
        value = stream.update(price, volume)
        streamed.append(value if value is not None else np.nan)
    streamed = np.asarray(streamed, dtype=float)
    stream.reset()
    streamed_after_reset = []
    for price, volume in zip(prices, volumes):
        value = stream.update(price, volume)
        streamed_after_reset.append(value if value is not None else np.nan)
    streamed_after_reset = np.asarray(streamed_after_reset, dtype=float)
    assert_series_close(streamed, batch)
    assert_series_close(streamed_after_reset, batch)


def test_evwma_batch_matches_single():
    prices, volumes = sample_series(84)
    batch = ta_indicators.elastic_volume_weighted_moving_average_batch(
        prices,
        volumes,
        length_range=(5, 9, 2),
        absolute_volume_millions=180.0,
        use_volume_sum=True,
    )
    assert batch["values"].shape == (3, len(prices))
    for row, length in enumerate(batch["lengths"]):
        single = ta_indicators.elastic_volume_weighted_moving_average(
            prices,
            volumes,
            length=int(length),
            absolute_volume_millions=180.0,
            use_volume_sum=True,
        )
        assert_series_close(batch["values"][row], single)


def test_evwma_errors():
    prices, volumes = sample_series(16)
    with pytest.raises(ValueError, match="input data slice is empty|Input data slice is empty"):
        ta_indicators.elastic_volume_weighted_moving_average(
            np.array([], dtype=float), np.array([], dtype=float)
        )
    with pytest.raises(ValueError, match="length mismatch"):
        ta_indicators.elastic_volume_weighted_moving_average(prices, volumes[:-1])
    with pytest.raises(ValueError, match="invalid length|Invalid length"):
        ta_indicators.elastic_volume_weighted_moving_average(prices, volumes, length=0)
    with pytest.raises(ValueError, match="invalid absolute_volume_millions|Invalid absolute_volume_millions"):
        ta_indicators.elastic_volume_weighted_moving_average(
            prices,
            volumes,
            absolute_volume_millions=0.0,
            use_volume_sum=False,
        )
