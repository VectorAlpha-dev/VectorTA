import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;

function sampleSeries(length = 96) {
    const prices = new Float64Array(length);
    const volumes = new Float64Array(length);
    for (let i = 0; i < length; i++) {
        prices[i] = 100.0 + i * 0.23 + Math.sin(i * 0.17);
        volumes[i] = 1000.0 + (i % 9) * 57.0 + i * 2.0;
    }
    return { prices, volumes };
}

function naiveEvwma(prices, volumes, length = 30, absoluteVolumeMillions = 134.0, useVolumeSum = false) {
    const out = new Float64Array(prices.length);
    out.fill(NaN);
    let first = -1;
    for (let i = 0; i < prices.length; i++) {
        if (Number.isFinite(prices[i]) && Number.isFinite(volumes[i])) {
            first = i;
            break;
        }
    }
    if (first < 0) {
        return out;
    }

    const ring = new Float64Array(length);
    const absoluteVolume = absoluteVolumeMillions * 1_000_000.0;
    let rollingSum = 0.0;
    let count = 0;
    let head = 0;
    let prev = NaN;

    for (let i = first; i < prices.length; i++) {
        const price = prices[i];
        const volume = volumes[i];
        if (useVolumeSum) {
            const volumeValue = Number.isFinite(volume) ? volume : 0.0;
            if (count < length) {
                ring[count] = volumeValue;
                rollingSum += volumeValue;
                count += 1;
            } else {
                rollingSum += volumeValue - ring[head];
                ring[head] = volumeValue;
                head = (head + 1) % length;
            }
        }
        const volumePeriod = useVolumeSum ? rollingSum : absoluteVolume;
        if (!Number.isFinite(price) || !Number.isFinite(volume) || !Number.isFinite(volumePeriod) || volumePeriod === 0.0) {
            prev = NaN;
            continue;
        }
        const base = Number.isFinite(prev) ? prev : price;
        const value = ((volumePeriod - volume) * base + volume * price) / volumePeriod;
        out[i] = value;
        prev = value;
    }
    return out;
}

function assertSeriesClose(actual, expected, message) {
    assert.strictEqual(actual.length, expected.length, `${message}: length mismatch`);
    for (let i = 0; i < actual.length; i++) {
        const a = actual[i];
        const e = expected[i];
        if (Number.isNaN(a) && Number.isNaN(e)) {
            continue;
        }
        assert.ok(Math.abs(a - e) <= 1e-10, `${message}: mismatch at ${i}: ${a} vs ${e}`);
    }
}

function getMemory(mod) {
    if (mod.memory && mod.memory.buffer) {
        return mod.memory;
    }
    if (typeof mod.__wbindgen_memory === 'function') {
        const memory = mod.__wbindgen_memory();
        if (memory && memory.buffer) {
            return memory;
        }
    }
    if (mod.__wasm && mod.__wasm.memory && mod.__wasm.memory.buffer) {
        return mod.__wasm.memory;
    }
    return null;
}

function getFloat64View(ptr, len) {
    const memory = getMemory(wasm);
    assert(memory && memory.buffer, 'WASM memory export unavailable');
    return new Float64Array(memory.buffer, ptr, len);
}

test.before(async () => {
    const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const importPath = process.platform === 'win32'
        ? 'file:///' + wasmPath.replace(/\\/g, '/')
        : wasmPath;
    wasm = await import(importPath);
});

test('EVWMA matches naive absolute mode', () => {
    const { prices, volumes } = sampleSeries();
    const actual = wasm.elastic_volume_weighted_moving_average(prices, volumes, undefined, undefined, undefined);
    const expected = naiveEvwma(prices, volumes);
    assertSeriesClose(actual, expected, 'absolute mode');
});

test('EVWMA matches naive volume-sum mode', () => {
    const { prices, volumes } = sampleSeries();
    const actual = wasm.elastic_volume_weighted_moving_average(prices, volumes, 7, 200.0, true);
    const expected = naiveEvwma(prices, volumes, 7, 200.0, true);
    assertSeriesClose(actual, expected, 'volume-sum mode');
});

test('EVWMA batch matches single', () => {
    const { prices, volumes } = sampleSeries(84);
    const batch = wasm.elastic_volume_weighted_moving_average_batch(prices, volumes, {
        length_range: [5, 9, 2],
        absolute_volume_millions: 180.0,
        use_volume_sum: true,
    });
    assert.strictEqual(batch.rows, 3);
    assert.strictEqual(batch.cols, prices.length);
    for (let row = 0; row < batch.rows; row++) {
        const start = row * batch.cols;
        const end = start + batch.cols;
        const single = wasm.elastic_volume_weighted_moving_average(
            prices,
            volumes,
            batch.combos[row].length,
            180.0,
            true,
        );
        assertSeriesClose(batch.values.slice(start, end), single, `batch row ${row}`);
    }
});

test('EVWMA zero-copy paths work', () => {
    const { prices, volumes } = sampleSeries(48);
    const len = prices.length;
    const batchRows = 3;
    const inPricePtr = wasm.elastic_volume_weighted_moving_average_alloc(len);
    const inVolumePtr = wasm.elastic_volume_weighted_moving_average_alloc(len);
    const outPtr = wasm.elastic_volume_weighted_moving_average_alloc(batchRows * len);
    try {
        getFloat64View(inPricePtr, len).set(prices);
        getFloat64View(inVolumePtr, len).set(volumes);
        wasm.elastic_volume_weighted_moving_average_into(
            inPricePtr,
            inVolumePtr,
            outPtr,
            len,
            7,
            200.0,
            true,
        );
        const actual = Array.from(getFloat64View(outPtr, len));
        const expected = wasm.elastic_volume_weighted_moving_average(prices, volumes, 7, 200.0, true);
        assertSeriesClose(actual, expected, 'zero-copy single');

        const config = {
            length_range: [5, 9, 2],
            absolute_volume_millions: 180.0,
            use_volume_sum: true,
        };
        const rows = wasm.elastic_volume_weighted_moving_average_batch_into(
            inPricePtr,
            inVolumePtr,
            outPtr,
            len,
            config,
        );
        assert.strictEqual(rows, batchRows);
        const batch = wasm.elastic_volume_weighted_moving_average_batch(prices, volumes, config);
        assertSeriesClose(Array.from(getFloat64View(outPtr, rows * len)), batch.values, 'zero-copy batch');
    } finally {
        wasm.elastic_volume_weighted_moving_average_free(inPricePtr, len);
        wasm.elastic_volume_weighted_moving_average_free(inVolumePtr, len);
        wasm.elastic_volume_weighted_moving_average_free(outPtr, batchRows * len);
    }
});

test('EVWMA stream and reset work', () => {
    const { prices, volumes } = sampleSeries(80);
    const stream = new wasm.ElasticVolumeWeightedMovingAverageStreamWasm(9, 220.0, true);
    const batch = wasm.elastic_volume_weighted_moving_average(prices, volumes, 9, 220.0, true);
    const streamed = [];
    for (let i = 0; i < prices.length; i++) {
        const value = stream.update(prices[i], volumes[i]);
        streamed.push(value ?? NaN);
    }
    assertSeriesClose(streamed, batch, 'stream');
    stream.reset();
    const streamedAgain = [];
    for (let i = 0; i < prices.length; i++) {
        const value = stream.update(prices[i], volumes[i]);
        streamedAgain.push(value ?? NaN);
    }
    assertSeriesClose(streamedAgain, batch, 'stream reset');
});

test('EVWMA errors', () => {
    const { prices, volumes } = sampleSeries(16);
    assert.throws(
        () => wasm.elastic_volume_weighted_moving_average([], [], undefined, undefined, undefined),
        /input data slice is empty|Input data slice is empty/
    );
    assert.throws(
        () => wasm.elastic_volume_weighted_moving_average(prices, volumes.subarray(0, 8), undefined, undefined, undefined),
        /length mismatch/
    );
    assert.throws(
        () => wasm.elastic_volume_weighted_moving_average(prices, volumes, 0, undefined, true),
        /invalid length|Invalid length/
    );
    assert.throws(
        () => wasm.elastic_volume_weighted_moving_average(prices, volumes, undefined, 0.0, false),
        /invalid absolute_volume_millions|Invalid absolute_volume_millions/
    );
});
