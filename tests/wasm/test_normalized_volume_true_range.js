import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { loadTestData, assertArrayClose, isNaN } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;

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

function naiveNormalizedVolumeTrueRange(open, high, low, close, volume, style = 'body', outlierRange = 5.0, atrLength = 14, volumeLength = 14) {
    const length = close.length;
    const normalizedVolume = new Array(length).fill(NaN);
    const normalizedTrueRange = new Array(length).fill(NaN);
    const baseline = new Array(length).fill(NaN);
    const atr = new Array(length).fill(NaN);
    const averageVolume = new Array(length).fill(NaN);

    let absSum = 0.0;
    let volumeSum = 0.0;
    let count = 0;
    let absVarianceSum = 0.0;
    let absQualifying = 0;
    let absPStdev = NaN;
    let volumeVarianceSum = 0.0;
    let volumeQualifying = 0;
    let volumePStdev = NaN;
    let prevClose = NaN;
    let havePrevClose = false;

    let atrFirst = NaN;
    let atrReady = false;
    const atrRing = new Array(atrLength).fill(0.0);
    let atrSum = 0.0;
    let atrHead = 0;

    let volumeFirst = NaN;
    let volumeReady = false;
    const volumeRing = new Array(volumeLength).fill(0.0);
    let volumeSumMa = 0.0;
    let volumeHead = 0;

    for (let idx = 0; idx < length; idx++) {
        const valid = style === 'body'
            ? Number.isFinite(open[idx]) && Number.isFinite(close[idx]) && Number.isFinite(volume[idx])
            : style === 'hl'
                ? Number.isFinite(high[idx]) && Number.isFinite(low[idx]) && Number.isFinite(volume[idx])
                : Number.isFinite(close[idx]) && Number.isFinite(volume[idx]);

        if (!valid) {
            if (Number.isFinite(close[idx])) {
                prevClose = close[idx];
                havePrevClose = true;
            }
            continue;
        }

        const prev = havePrevClose ? prevClose : close[idx];
        let start;
        let finish;
        if (style === 'body') {
            start = open[idx];
            finish = close[idx];
        } else if (style === 'hl') {
            start = low[idx];
            finish = high[idx];
        } else {
            start = prev;
            finish = close[idx];
        }

        prevClose = close[idx];
        havePrevClose = true;

        const denom = Math.min(start, finish);
        if (!Number.isFinite(denom) || denom <= 0.0) {
            continue;
        }

        const absPercent = Math.abs(finish - start) / denom;
        if (!Number.isFinite(absPercent)) {
            continue;
        }

        count += 1;
        absSum += absPercent;
        volumeSum += volume[idx];
        const avgAbsPercent = absSum / count;
        const avgVolume = volumeSum / count;

        if (absPercent > avgAbsPercent) {
            const delta = absPercent - avgAbsPercent;
            absVarianceSum += delta * delta;
            absQualifying += 1;
        }
        if (absQualifying >= 2) {
            absPStdev = Math.sqrt(absVarianceSum / (absQualifying - 1));
        }

        if (volume[idx] > avgVolume) {
            const delta = volume[idx] - avgVolume;
            volumeVarianceSum += delta * delta;
            volumeQualifying += 1;
        }
        if (volumeQualifying >= 2) {
            volumePStdev = Math.sqrt(volumeVarianceSum / (volumeQualifying - 1));
        }

        const absPercentMax = Number.isFinite(absPStdev) ? avgAbsPercent + absPStdev * outlierRange : NaN;
        const normalizedAvgPercent = Number.isFinite(absPercentMax) && absPercentMax > 0.0 ? avgAbsPercent / absPercentMax : NaN;
        const scaleFactor = Number.isFinite(normalizedAvgPercent) && normalizedAvgPercent > 0.0 && normalizedAvgPercent < 1.0 && Number.isFinite(volumePStdev) && volumePStdev > 0.0
            ? avgVolume * (1.0 - normalizedAvgPercent) / (normalizedAvgPercent * volumePStdev)
            : NaN;
        const maxVolume = Number.isFinite(scaleFactor) && Number.isFinite(volumePStdev)
            ? avgVolume + volumePStdev * scaleFactor
            : NaN;

        const normTr = Number.isFinite(absPercentMax) && absPercentMax > 0.0
            ? Math.min(absPercent, absPercentMax) / absPercentMax
            : NaN;
        const normVol = Number.isFinite(maxVolume) && maxVolume > 0.0
            ? Math.min(volume[idx], maxVolume) / maxVolume
            : NaN;
        const normAvgVol = Number.isFinite(maxVolume) && maxVolume > 0.0
            ? avgVolume / maxVolume
            : NaN;

        const normVolPct = normVol * 100.0;
        const normTrPct = normTr * 100.0;
        const baselinePct = normAvgVol * 100.0;

        let atrValue = NaN;
        if (!atrReady) {
            if (Number.isFinite(normTrPct)) {
                atrFirst = normTrPct;
                atrRing.fill(atrFirst);
                atrSum = atrFirst * atrLength;
                atrHead = 1 % atrLength;
                atrReady = true;
                atrValue = atrFirst;
            }
        } else {
            const source = Number.isFinite(normTrPct) ? normTrPct : atrFirst;
            atrSum += source - atrRing[atrHead];
            atrRing[atrHead] = source;
            atrHead = (atrHead + 1) % atrLength;
            atrValue = atrSum / atrLength;
        }

        let averageVolumeValue = NaN;
        if (!volumeReady) {
            if (Number.isFinite(normVolPct)) {
                volumeFirst = normVolPct;
                volumeRing.fill(volumeFirst);
                volumeSumMa = volumeFirst * volumeLength;
                volumeHead = 1 % volumeLength;
                volumeReady = true;
                averageVolumeValue = volumeFirst;
            }
        } else {
            const source = Number.isFinite(normVolPct) ? normVolPct : volumeFirst;
            volumeSumMa += source - volumeRing[volumeHead];
            volumeRing[volumeHead] = source;
            volumeHead = (volumeHead + 1) % volumeLength;
            averageVolumeValue = volumeSumMa / volumeLength;
        }

        if ([normVolPct, normTrPct, baselinePct, atrValue, averageVolumeValue].every(Number.isFinite)) {
            normalizedVolume[idx] = normVolPct;
            normalizedTrueRange[idx] = normTrPct;
            baseline[idx] = baselinePct;
            atr[idx] = atrValue;
            averageVolume[idx] = averageVolumeValue;
        }
    }

    return {
        normalizedVolume,
        normalizedTrueRange,
        baseline,
        atr,
        averageVolume,
    };
}

test.before(async () => {
    const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const importPath = process.platform === 'win32'
        ? 'file:///' + wasmPath.replace(/\\/g, '/')
        : wasmPath;
    wasm = await import(importPath);
    testData = loadTestData();
});

test('Normalized Volume True Range basic output shape', () => {
    const result = wasm.normalized_volume_true_range_js(
        testData.open,
        testData.high,
        testData.low,
        testData.close,
        testData.volume,
        'body',
        5.0,
        14,
        14
    );
    for (const key of ['normalized_volume', 'normalized_true_range', 'baseline', 'atr', 'average_volume']) {
        assert.strictEqual(result[key].length, testData.close.length);
        let hasFinite = false;
        for (let i = 32; i < result[key].length; i++) {
            hasFinite ||= !isNaN(result[key][i]);
        }
        assert(hasFinite);
    }
});

test('Normalized Volume True Range matches naive reference', () => {
    const open = [10.0, 10.5, 11.0, 11.2, 11.4, 11.1, 11.6, 11.9];
    const high = [10.8, 11.1, 11.5, 11.6, 11.8, 11.7, 12.2, 12.3];
    const low = [9.9, 10.2, 10.8, 11.0, 11.1, 10.9, 11.2, 11.6];
    const close = [10.6, 10.9, 11.2, 11.3, 11.2, 11.5, 12.0, 12.1];
    const volume = [1000.0, 1200.0, 1800.0, 1600.0, 2100.0, 1700.0, 2400.0, 2200.0];
    const actual = wasm.normalized_volume_true_range_js(open, high, low, close, volume, 'delta', 4.0, 3, 2);
    const expected = naiveNormalizedVolumeTrueRange(open, high, low, close, volume, 'delta', 4.0, 3, 2);
    assertArrayClose(actual.normalized_volume, expected.normalizedVolume, 1e-12, 1e-12, 'normalized_volume mismatch');
    assertArrayClose(actual.normalized_true_range, expected.normalizedTrueRange, 1e-12, 1e-12, 'normalized_true_range mismatch');
    assertArrayClose(actual.baseline, expected.baseline, 1e-12, 1e-12, 'baseline mismatch');
    assertArrayClose(actual.atr, expected.atr, 1e-12, 1e-12, 'atr mismatch');
    assertArrayClose(actual.average_volume, expected.averageVolume, 1e-12, 1e-12, 'average_volume mismatch');
});

test('Normalized Volume True Range batch matches single', () => {
    const single = wasm.normalized_volume_true_range_js(
        testData.open,
        testData.high,
        testData.low,
        testData.close,
        testData.volume,
        'body',
        5.0,
        14,
        14
    );
    const batch = wasm.normalized_volume_true_range_batch(
        testData.open,
        testData.high,
        testData.low,
        testData.close,
        testData.volume,
        {
            true_range_style: 'body',
            outlier_range_range: [5.0, 5.0, 0.0],
            atr_length_range: [14, 14, 0],
            volume_length_range: [14, 14, 0],
        }
    );

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, testData.close.length);
    assertArrayClose(batch.normalized_volume, single.normalized_volume, 1e-12, 1e-12, 'batch normalized_volume mismatch');
    assertArrayClose(batch.normalized_true_range, single.normalized_true_range, 1e-12, 1e-12, 'batch normalized_true_range mismatch');
    assertArrayClose(batch.baseline, single.baseline, 1e-12, 1e-12, 'batch baseline mismatch');
    assertArrayClose(batch.atr, single.atr, 1e-12, 1e-12, 'batch atr mismatch');
    assertArrayClose(batch.average_volume, single.average_volume, 1e-12, 1e-12, 'batch average_volume mismatch');
});

test('Normalized Volume True Range zero-copy and stream paths work', async () => {
    const len = testData.close.length;
    const openPtr = wasm.normalized_volume_true_range_alloc(len);
    const highPtr = wasm.normalized_volume_true_range_alloc(len);
    const lowPtr = wasm.normalized_volume_true_range_alloc(len);
    const closePtr = wasm.normalized_volume_true_range_alloc(len);
    const volumePtr = wasm.normalized_volume_true_range_alloc(len);
    const normalizedVolumePtr = wasm.normalized_volume_true_range_alloc(len);
    const normalizedTrueRangePtr = wasm.normalized_volume_true_range_alloc(len);
    const baselinePtr = wasm.normalized_volume_true_range_alloc(len);
    const atrPtr = wasm.normalized_volume_true_range_alloc(len);
    const averageVolumePtr = wasm.normalized_volume_true_range_alloc(len);

    try {
        const memory = getMemory(wasm);
        assert(memory, 'WASM memory accessor unavailable');
        new Float64Array(memory.buffer, openPtr, len).set(testData.open);
        new Float64Array(memory.buffer, highPtr, len).set(testData.high);
        new Float64Array(memory.buffer, lowPtr, len).set(testData.low);
        new Float64Array(memory.buffer, closePtr, len).set(testData.close);
        new Float64Array(memory.buffer, volumePtr, len).set(testData.volume);

        wasm.normalized_volume_true_range_into(
            openPtr,
            highPtr,
            lowPtr,
            closePtr,
            volumePtr,
            normalizedVolumePtr,
            normalizedTrueRangePtr,
            baselinePtr,
            atrPtr,
            averageVolumePtr,
            len,
            'hl',
            6.0,
            9,
            6
        );

        const expected = wasm.normalized_volume_true_range_js(
            testData.open,
            testData.high,
            testData.low,
            testData.close,
            testData.volume,
            'hl',
            6.0,
            9,
            6
        );

        assertArrayClose(Array.from(new Float64Array(memory.buffer, normalizedVolumePtr, len)), expected.normalized_volume, 1e-12, 1e-12, 'zero-copy normalized_volume mismatch');
        assertArrayClose(Array.from(new Float64Array(memory.buffer, normalizedTrueRangePtr, len)), expected.normalized_true_range, 1e-12, 1e-12, 'zero-copy normalized_true_range mismatch');
        assertArrayClose(Array.from(new Float64Array(memory.buffer, baselinePtr, len)), expected.baseline, 1e-12, 1e-12, 'zero-copy baseline mismatch');
        assertArrayClose(Array.from(new Float64Array(memory.buffer, atrPtr, len)), expected.atr, 1e-12, 1e-12, 'zero-copy atr mismatch');
        assertArrayClose(Array.from(new Float64Array(memory.buffer, averageVolumePtr, len)), expected.average_volume, 1e-12, 1e-12, 'zero-copy average_volume mismatch');

        const rows = wasm.normalized_volume_true_range_batch_into(
            openPtr,
            highPtr,
            lowPtr,
            closePtr,
            volumePtr,
            normalizedVolumePtr,
            normalizedTrueRangePtr,
            baselinePtr,
            atrPtr,
            averageVolumePtr,
            len,
            6.0,
            6.0,
            0.0,
            9,
            9,
            0,
            6,
            6,
            0,
            'hl'
        );
        assert.strictEqual(rows, 1);
        assertArrayClose(Array.from(new Float64Array(memory.buffer, normalizedVolumePtr, len)), expected.normalized_volume, 1e-12, 1e-12, 'batch-into normalized_volume mismatch');
        assertArrayClose(Array.from(new Float64Array(memory.buffer, normalizedTrueRangePtr, len)), expected.normalized_true_range, 1e-12, 1e-12, 'batch-into normalized_true_range mismatch');
        assertArrayClose(Array.from(new Float64Array(memory.buffer, baselinePtr, len)), expected.baseline, 1e-12, 1e-12, 'batch-into baseline mismatch');
        assertArrayClose(Array.from(new Float64Array(memory.buffer, atrPtr, len)), expected.atr, 1e-12, 1e-12, 'batch-into atr mismatch');
        assertArrayClose(Array.from(new Float64Array(memory.buffer, averageVolumePtr, len)), expected.average_volume, 1e-12, 1e-12, 'batch-into average_volume mismatch');

        const stream = new wasm.NormalizedVolumeTrueRangeStreamWasm('hl', 6.0, 9, 6);
        const actual = {
            normalized_volume: new Array(len).fill(NaN),
            normalized_true_range: new Array(len).fill(NaN),
            baseline: new Array(len).fill(NaN),
            atr: new Array(len).fill(NaN),
            average_volume: new Array(len).fill(NaN),
        };
        for (let i = 0; i < len; i++) {
            const result = await stream.update(
                testData.open[i],
                testData.high[i],
                testData.low[i],
                testData.close[i],
                testData.volume[i]
            );
            if (result !== null) {
                actual.normalized_volume[i] = result.normalized_volume;
                actual.normalized_true_range[i] = result.normalized_true_range;
                actual.baseline[i] = result.baseline;
                actual.atr[i] = result.atr;
                actual.average_volume[i] = result.average_volume;
            }
        }
        assertArrayClose(actual.normalized_volume, expected.normalized_volume, 1e-12, 1e-12, 'stream normalized_volume mismatch');
        assertArrayClose(actual.normalized_true_range, expected.normalized_true_range, 1e-12, 1e-12, 'stream normalized_true_range mismatch');
        assertArrayClose(actual.baseline, expected.baseline, 1e-12, 1e-12, 'stream baseline mismatch');
        assertArrayClose(actual.atr, expected.atr, 1e-12, 1e-12, 'stream atr mismatch');
        assertArrayClose(actual.average_volume, expected.average_volume, 1e-12, 1e-12, 'stream average_volume mismatch');

        stream.reset();
        assert.strictEqual(await stream.update(NaN, NaN, NaN, NaN, NaN), null);
    } finally {
        for (const ptr of [openPtr, highPtr, lowPtr, closePtr, volumePtr, normalizedVolumePtr, normalizedTrueRangePtr, baselinePtr, atrPtr, averageVolumePtr]) {
            wasm.normalized_volume_true_range_free(ptr, len);
        }
    }
});

test('Normalized Volume True Range errors', () => {
    const sample = new Float64Array([10.0, 11.0, 12.0, 13.0]);
    assert.throws(
        () => wasm.normalized_volume_true_range_js(sample, sample, sample, sample, sample, 'body', 0.25, 14, 14),
        /invalid outlier_range|Invalid outlier_range/i
    );
    assert.throws(
        () => wasm.normalized_volume_true_range_js(sample, sample, sample, sample, sample, 'body', 5.0, 1, 14),
        /invalid atr_length|Invalid atr_length/i
    );
    assert.throws(
        () => wasm.normalized_volume_true_range_js(sample, sample, sample, sample, sample, 'body', 5.0, 14, 1),
        /invalid volume_length|Invalid volume_length/i
    );
    assert.throws(
        () => wasm.normalized_volume_true_range_js([], [], [], [], [], 'body', 5.0, 14, 14),
        /empty|Empty/i
    );
    const allNaN = new Float64Array(32).fill(NaN);
    assert.throws(
        () => wasm.normalized_volume_true_range_js(allNaN, allNaN, allNaN, allNaN, allNaN, 'body', 5.0, 14, 14),
        /all values are NaN/i
    );
});
