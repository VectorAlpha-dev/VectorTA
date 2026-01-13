
import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';
import { parse } from 'csv-parse/sync';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;

test.before(async () => {

    try {
        const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
        const importPath = process.platform === 'win32'
            ? 'file:///' + wasmPath.replace(/\\/g, '/')
            : wasmPath;
        wasm = await import(importPath);
    } catch (error) {
        console.error('Failed to load WASM module:', error);
        throw error;
    }


    const csvPath = path.join(__dirname, '../../src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv');
    try {
        const csvContent = fs.readFileSync(csvPath, 'utf-8');
        const records = parse(csvContent, {
            columns: false,
            skip_empty_lines: true
        });


        testData = {
            open: new Float64Array(records.map(r => parseFloat(r[1]))),
            high: new Float64Array(records.map(r => parseFloat(r[3]))),
            low: new Float64Array(records.map(r => parseFloat(r[4]))),
            close: new Float64Array(records.map(r => parseFloat(r[2]))),
            volume: new Float64Array(records.map(r => parseFloat(r[5])))
        };
    } catch (error) {
        console.error('Failed to load test data:', error);
        throw error;
    }
});

test('AlphaTrend accuracy', async () => {
    const expected = wasm.EXPECTED_OUTPUTS?.alphatrend || {
        defaultParams: { coeff: 1.0, period: 14, noVolume: false },
        k1Last5Values: [
            60243.00,
            60243.00,
            60138.92857143,
            60088.42857143,
            59937.21428571
        ],
        k2Last5Values: [
            60542.42857143,
            60454.14285714,
            60243.00,
            60243.00,
            60138.92857143
        ]
    };

    const result = wasm.alphatrend_js(
        testData.open,
        testData.high,
        testData.low,
        testData.close,
        testData.volume,
        expected.defaultParams.coeff,
        expected.defaultParams.period,
        expected.defaultParams.noVolume
    );


    const len = testData.close.length;
    assert.strictEqual(result.cols, len);
    assert.strictEqual(result.rows, 2);
    assert.strictEqual(result.values.length, len * 2);

    const k1 = result.values.slice(0, len);
    const k2 = result.values.slice(len);


    const k1_last5 = k1.slice(-5);
    for (let i = 0; i < 5; i++) {
        if (!isNaN(k1_last5[i])) {
            const diff = Math.abs(k1_last5[i] - expected.k1Last5Values[i]);
            assert.ok(
                diff < 1e-6,
                `K1 mismatch at index ${i}: got ${k1_last5[i]}, expected ${expected.k1Last5Values[i]}`
            );
        }
    }


    const k2_last5 = k2.slice(-5);
    for (let i = 0; i < 5; i++) {
        if (!isNaN(k2_last5[i])) {
            const diff = Math.abs(k2_last5[i] - expected.k2Last5Values[i]);
            assert.ok(
                diff < 1e-6,
                `K2 mismatch at index ${i}: got ${k2_last5[i]}, expected ${expected.k2Last5Values[i]}`
            );
        }
    }
});

test('AlphaTrend with RSI (no_volume=true)', () => {
    const result = wasm.alphatrend_js(
        testData.open,
        testData.high,
        testData.low,
        testData.close,
        testData.volume,
        1.0,
        14,
        true
    );

    const len = testData.close.length;
    assert.strictEqual(result.cols, len);
    assert.strictEqual(result.rows, 2);
    assert.strictEqual(result.values.length, len * 2);

    const k1 = result.values.slice(0, len);
    const k2 = result.values.slice(len);


    const nonNanK1 = k1.filter(v => !isNaN(v));
    const nonNanK2 = k2.filter(v => !isNaN(v));
    assert.ok(nonNanK1.length > 0, 'K1 should have non-NaN values');
    assert.ok(nonNanK2.length > 0, 'K2 should have non-NaN values');
});

test('AlphaTrend zero period', () => {
    const open = new Float64Array([10.0, 20.0, 30.0]);
    const high = new Float64Array([12.0, 22.0, 32.0]);
    const low = new Float64Array([8.0, 18.0, 28.0]);
    const close = new Float64Array([11.0, 21.0, 31.0]);
    const volume = new Float64Array([100.0, 200.0, 300.0]);

    assert.throws(() => {
        wasm.alphatrend_js(open, high, low, close, volume, 1.0, 0, false);
    }, /Invalid period/);
});

test('AlphaTrend empty input', () => {
    const empty = new Float64Array([]);

    assert.throws(() => {
        wasm.alphatrend_js(empty, empty, empty, empty, empty, 1.0, 14, false);
    }, /empty/i);
});

test('AlphaTrend inconsistent lengths', () => {
    const open = new Float64Array([10.0, 20.0, 30.0]);
    const high = new Float64Array([12.0, 22.0]);
    const low = new Float64Array([8.0, 18.0, 28.0]);
    const close = new Float64Array([11.0, 21.0, 31.0]);
    const volume = new Float64Array([100.0, 200.0, 300.0]);

    assert.throws(() => {
        wasm.alphatrend_js(open, high, low, close, volume, 1.0, 14, false);
    }, /Inconsistent data lengths/);
});

test('AlphaTrend invalid coefficient', () => {

    const data = new Float64Array(Array(20).fill(0).map((_, i) => 10.0 + i * 5));

    assert.throws(() => {
        wasm.alphatrend_js(data, data, data, data, data, -1.0, 14, false);
    }, /Invalid coefficient/);
});

test('AlphaTrend period exceeds length', () => {
    const smallData = new Float64Array([10.0, 20.0, 30.0]);

    assert.throws(() => {
        wasm.alphatrend_js(
            smallData, smallData, smallData, smallData, smallData,
            1.0, 10, false
        );
    }, /Invalid period/);
});

test('AlphaTrend all NaN', () => {
    const nanData = new Float64Array([NaN, NaN, NaN]);

    assert.throws(() => {
        wasm.alphatrend_js(nanData, nanData, nanData, nanData, nanData, 1.0, 14, false);
    }, /All values are NaN/);
});

test('AlphaTrend different parameters', () => {

    const size = 100;
    const open = new Float64Array(size);
    const high = new Float64Array(size);
    const low = new Float64Array(size);
    const close = new Float64Array(size);
    const volume = new Float64Array(size);


    for (let i = 0; i < size; i++) {
        const base = 100 + i * 0.5;
        open[i] = base;
        high[i] = base + 2;
        low[i] = base - 2;
        close[i] = base + Math.sin(i * 0.1);
        volume[i] = 1000 + i * 10;
    }


    const coeffs = [0.5, 1.0, 2.0];
    for (const coeff of coeffs) {
        const result = wasm.alphatrend_js(
            open, high, low, close, volume,
            coeff, 14, false
        );
        assert.strictEqual(result.values.length, size * 2, `Length check for coeff=${coeff}`);
    }


    const periods = [7, 14, 21];
    for (const period of periods) {
        const result = wasm.alphatrend_js(
            open, high, low, close, volume,
            1.0, period, false
        );
        assert.strictEqual(result.values.length, size * 2, `Length check for period=${period}`);
    }


    const resultMFI = wasm.alphatrend_js(
        open, high, low, close, volume,
        1.0, 14, false
    );
    const resultRSI = wasm.alphatrend_js(
        open, high, low, close, volume,
        1.0, 14, true
    );


    const k1_mfi = resultMFI.values.slice(0, size);
    const k1_rsi = resultRSI.values.slice(0, size);

    let foundDifference = false;
    for (let i = 20; i < size; i++) {
        if (!isNaN(k1_mfi[i]) && !isNaN(k1_rsi[i])) {
            if (Math.abs(k1_mfi[i] - k1_rsi[i]) > 1e-10) {
                foundDifference = true;
                break;
            }
        }
    }
    assert.ok(foundDifference, 'MFI and RSI modes should produce different results');
});

test('AlphaTrend memory allocation', () => {
    const size = 1000;
    const ptr = wasm.alphatrend_alloc_flat(size);

    assert.ok(ptr !== 0, 'Should allocate memory successfully');


    wasm.alphatrend_free_flat(ptr, size);
});

test('AlphaTrend warmup period', () => {
    const expected = wasm.EXPECTED_OUTPUTS?.alphatrend || {
        defaultParams: { coeff: 1.0, period: 14, noVolume: false },
        warmupPeriod: 13
    };

    const result = wasm.alphatrend_js(
        testData.open,
        testData.high,
        testData.low,
        testData.close,
        testData.volume,
        expected.defaultParams.coeff,
        expected.defaultParams.period,
        expected.defaultParams.noVolume
    );

    const len = testData.close.length;
    const k1 = result.values.slice(0, len);
    const k2 = result.values.slice(len);
    const warmup = expected.warmupPeriod;


    for (let i = 0; i < warmup && i < k1.length; i++) {
        assert.ok(isNaN(k1[i]), `Expected NaN in K1 warmup at index ${i}`);
        assert.ok(isNaN(k2[i]), `Expected NaN in K2 warmup at index ${i}`);
    }


    if (k1.length > warmup + 10) {
        let foundNonNaN = false;
        for (let i = warmup; i < warmup + 10; i++) {
            if (!isNaN(k1[i])) {
                foundNonNaN = true;
                break;
            }
        }
        assert.ok(foundNonNaN, 'K1 should have real values after warmup');

        foundNonNaN = false;
        for (let i = warmup; i < warmup + 10; i++) {
            if (!isNaN(k2[i])) {
                foundNonNaN = true;
                break;
            }
        }
        assert.ok(foundNonNaN, 'K2 should have real values after warmup');
    }
});

test('AlphaTrend batch processing', () => {

    if (typeof wasm.alphatrend_batch !== 'function') {
        console.log('AlphaTrend batch function not yet exposed in WASM bindings');

        return;
    }


    const size = Math.min(200, testData.close.length);
    const open = new Float64Array(testData.open.slice(0, size));
    const high = new Float64Array(testData.high.slice(0, size));
    const low = new Float64Array(testData.low.slice(0, size));
    const close = new Float64Array(testData.close.slice(0, size));
    const volume = new Float64Array(testData.volume.slice(0, size));




    const singleResult = wasm.alphatrend_batch(
        open, high, low, close, volume,
        1.0, 1.0, 0,
        14, 14, 0,
        false
    );

    assert.ok(singleResult.values, 'Batch result should have values');



    assert.strictEqual(singleResult.rows, 2, 'Should have 2 rows for single combo (K1 and K2)');
    assert.strictEqual(singleResult.cols, size, 'Should have correct column count');


    assert.strictEqual(singleResult.values.length, 2 * size, 'Single combo should have 2 * size values');


    const multiResult = wasm.alphatrend_batch(
        open, high, low, close, volume,
        0.5, 2.0, 0.5,
        7, 21, 7,
        false
    );



    const expectedRows = 24;
    assert.strictEqual(
        multiResult.rows,
        expectedRows,
        `Should have ${expectedRows} rows (12 combos * 2 for K1 and K2)`
    );


    assert.strictEqual(
        multiResult.values.length,
        expectedRows * size,
        'Batch values should have correct total length'
    );
});

test('AlphaTrend NaN distribution', () => {

    const size = Math.min(300, testData.close.length);
    const open = new Float64Array(testData.open.slice(0, size));
    const high = new Float64Array(testData.high.slice(0, size));
    const low = new Float64Array(testData.low.slice(0, size));
    const close = new Float64Array(testData.close.slice(0, size));
    const volume = new Float64Array(testData.volume.slice(0, size));


    const nanIndices = [50, 100, 150, 200, 250].filter(i => i < size);
    for (const idx of nanIndices) {
        close[idx] = NaN;
        high[idx] = NaN;
        low[idx] = NaN;
    }


    const result = wasm.alphatrend_js(
        open, high, low, close, volume,
        1.0, 14, false
    );

    const k1 = result.values.slice(0, size);
    const k2 = result.values.slice(size);

    assert.strictEqual(k1.length, size, 'K1 should have correct length');
    assert.strictEqual(k2.length, size, 'K2 should have correct length');


    for (const idx of nanIndices) {

        assert.ok(
            isNaN(k1[idx]) || isNaN(k2[idx]),
            `Expected NaN propagation at index ${idx}`
        );
    }
});

test('AlphaTrend with real market conditions', () => {

    const size = 50;
    const open = new Float64Array(size);
    const high = new Float64Array(size);
    const low = new Float64Array(size);
    const close = new Float64Array(size);
    const volume = new Float64Array(size);


    for (let i = 0; i < size; i++) {
        const base = 100 + i * 2;
        open[i] = base;
        high[i] = base + 3;
        low[i] = base - 1;
        close[i] = base + 2;
        volume[i] = 1000 + Math.random() * 500;
    }

    const uptrend = wasm.alphatrend_js(
        open, high, low, close, volume,
        1.0, 14, false
    );


    for (let i = 0; i < size; i++) {
        const base = 200 - i * 2;
        open[i] = base;
        high[i] = base + 1;
        low[i] = base - 3;
        close[i] = base - 2;
        volume[i] = 1000 + Math.random() * 500;
    }

    const downtrend = wasm.alphatrend_js(
        open, high, low, close, volume,
        1.0, 14, false
    );


    for (let i = 0; i < size; i++) {
        const base = 150 + Math.sin(i * 0.5) * 5;
        open[i] = base;
        high[i] = base + 2;
        low[i] = base - 2;
        close[i] = base + Math.sin(i * 0.5 + 1) * 1;
        volume[i] = 1000 + Math.random() * 500;
    }

    const sideways = wasm.alphatrend_js(
        open, high, low, close, volume,
        1.0, 14, false
    );


    assert.strictEqual(uptrend.values.length, size * 2);
    assert.strictEqual(downtrend.values.length, size * 2);
    assert.strictEqual(sideways.values.length, size * 2);
});