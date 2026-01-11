/**
 * WASM binding tests for Reflex indicator.
 * These tests mirror the Rust unit tests to ensure WASM bindings work correctly.
 */
import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { 
    loadTestData, 
    assertArrayClose, 
    assertClose,
    isNaN,
    assertAllNaN,
    assertNoNaN,
    EXPECTED_OUTPUTS 
} from './test_utils.js';
import { compareWithRust } from './rust-comparison.js';

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
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }
    
    testData = loadTestData();
});

test('Reflex partial params', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.reflex_js(close, 20);
    assert(result instanceof Float64Array);
    assert.strictEqual(result.length, close.length);
});

test('Reflex accuracy', async () => {
    
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.reflex || {};
    
    const result = wasm.reflex_js(close, 20);
    
    assert.strictEqual(result.length, close.length);
    
    
    const expectedLastFive = [
        0.8085220962465361,
        0.445264715886137,
        0.13861699036615063,
        -0.03598639652007061,
        -0.224906760543743
    ];
    
    const actualLastFive = result.slice(-5);
    
    assertArrayClose(
        actualLastFive,
        expectedLastFive,
        1e-7,
        "Reflex last 5 values mismatch"
    );
    
    
    await compareWithRust('reflex', result, 'close', { period: 20 });
});

test('Reflex default candles', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.reflex_js(close, 20);
    assert.strictEqual(result.length, close.length);
});

test('Reflex zero period', () => {
    
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.reflex_js(inputData, 0);
    }, /period must be >=2|invalid period/i);
});

test('Reflex period less than two', () => {
    
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.reflex_js(inputData, 1);
    }, /period must be >=2|invalid period/i);
});

test('Reflex period exceeds length', () => {
    
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.reflex_js(dataSmall, 10);
    }, /invalid period|not enough valid data|not enough data/i);
});

test('Reflex very small dataset', () => {
    
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.reflex_js(singlePoint, 5);
    }, /invalid period|not enough valid data|not enough data/i);
});

test('Reflex empty input', () => {
    
    const dataEmpty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.reflex_js(dataEmpty, 20);
    }, /empty/i);
});

test('Reflex NaN handling', () => {
    
    const close = new Float64Array(testData.close);
    const period = 14;
    
    const result = wasm.reflex_js(close, period);
    
    assert.strictEqual(result.length, close.length);
    
    
    for (let i = 0; i < period; i++) {
        assert.strictEqual(result[i], 0.0, `Expected zero at index ${i} during warmup`);
    }
    
    
    if (result.length > period) {
        for (let i = period; i < result.length; i++) {
            if (!isNaN(close[i])) {
                assert(isFinite(result[i]), `Found unexpected non-finite value at index ${i}`);
            }
        }
    }
});

test('Reflex batch', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const batch_result = wasm.reflex_batch_js(
        close, 
        20, 20, 0    
    );
    
    
    const rows_cols = wasm.reflex_batch_rows_cols_js(20, 20, 0, close.length);
    const rows = rows_cols[0];
    const cols = rows_cols[1];
    
    assert(batch_result instanceof Float64Array);
    assert.strictEqual(rows, 1); 
    assert.strictEqual(cols, close.length);
    assert.strictEqual(batch_result.length, rows * cols);
    
    
    const individual_result = wasm.reflex_js(close, 20);
    assertArrayClose(batch_result, individual_result, 1e-9, "Batch vs single results");
    
    
    const expectedLastFive = [
        0.8085220962465361,
        0.445264715886137,
        0.13861699036615063,
        -0.03598639652007061,
        -0.224906760543743
    ];
    
    const actualLastFive = batch_result.slice(-5);
    assertArrayClose(
        actualLastFive,
        expectedLastFive,
        1e-7,
        "Reflex batch last 5 values mismatch"
    );
});

test('Reflex batch multiple periods', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const batch_result = wasm.reflex_batch_js(
        close, 
        10, 30, 5    
    );
    
    
    const rows_cols = wasm.reflex_batch_rows_cols_js(10, 30, 5, close.length);
    const rows = rows_cols[0];
    const cols = rows_cols[1];
    
    assert(batch_result instanceof Float64Array);
    assert.strictEqual(rows, 5); 
    assert.strictEqual(cols, close.length);
    assert.strictEqual(batch_result.length, rows * cols);
    
    
    const metadata = wasm.reflex_batch_metadata_js(10, 30, 5);
    assert.strictEqual(metadata.length, 5);
    assert.deepStrictEqual(Array.from(metadata), [10, 15, 20, 25, 30]);
    
    
    const individual_result = wasm.reflex_js(close, 10);
    const batch_first = batch_result.slice(0, close.length);
    
    
    const warmup = 10;
    for (let i = warmup; i < close.length; i++) {
        assertClose(batch_first[i], individual_result[i], 1e-9, `Batch mismatch at ${i}`);
    }
});

test('Reflex all NaN input', () => {
    
    const allNaN = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        allNaN[i] = NaN;
    }
    
    assert.throws(() => {
        wasm.reflex_js(allNaN, 20);
    }, /All values.*NaN/);
});

test('Reflex batch error conditions', () => {
    
    const allNaN = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        allNaN[i] = NaN;
    }
    
    assert.throws(() => {
        wasm.reflex_batch_js(allNaN, 10, 20, 5);
    }, /All values.*NaN/);
    
    
    const smallData = new Float64Array([1.0, 2.0, 3.0]);
    assert.throws(() => {
        wasm.reflex_batch_js(smallData, 10, 20, 5);
    }, /not enough valid data|invalid period|not enough/i);
});

test('Reflex edge cases', () => {
    
    const data = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        data[i] = i + 1;
    }
    
    const result = wasm.reflex_js(data, 20);
    assert.strictEqual(result.length, data.length);
    
    
    for (let i = 0; i < 20; i++) {
        assert.strictEqual(result[i], 0.0, `Expected zero at index ${i} during warmup`);
    }
    
    
    for (let i = 20; i < result.length; i++) {
        assert(isFinite(result[i]), `NaN at index ${i}`);
    }
    
    
    const constantData = new Float64Array(100).fill(50.0);
    const constantResult = wasm.reflex_js(constantData, 20);
    
    assert.strictEqual(constantResult.length, constantData.length);
    
    
    for (let i = 0; i < 20; i++) {
        assert.strictEqual(constantResult[i], 0.0, `Expected zero at index ${i} during warmup`);
    }
    
    
    
    
    const oscillatingData = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        oscillatingData[i] = (i % 2 === 0) ? 10.0 : 20.0;
    }
    
    const oscillatingResult = wasm.reflex_js(oscillatingData, 20);
    assert.strictEqual(oscillatingResult.length, oscillatingData.length);
    
    
    for (let i = 0; i < 20; i++) {
        assert.strictEqual(oscillatingResult[i], 0.0, `Expected zero at index ${i} during warmup`);
    }
    
    
    for (let i = 20; i < oscillatingResult.length; i++) {
        assert(isFinite(oscillatingResult[i]), `Expected finite value at index ${i}`);
    }
});

test('Reflex warmup behavior', () => {
    
    const close = new Float64Array(testData.close);
    const period = 20;
    
    const result = wasm.reflex_js(close, period);
    
    
    for (let i = 0; i < period; i++) {
        assert.strictEqual(result[i], 0.0, `Expected zero at index ${i} during warmup`);
    }
    
    
    let firstValid = 0;
    for (let i = 0; i < close.length; i++) {
        if (!isNaN(close[i])) {
            firstValid = i;
            break;
        }
    }
    
    const warmup = firstValid + period;
    
    
    if (warmup < result.length) {
        for (let i = warmup; i < result.length; i++) {
            assert(isFinite(result[i]), `Expected finite value at index ${i} after warmup`);
        }
    }
});

test('Reflex consistency', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const result1 = wasm.reflex_js(close, 20);
    const result2 = wasm.reflex_js(close, 20);
    
    assertArrayClose(result1, result2, 1e-15, "Reflex results not consistent");
});

test('Reflex batch metadata', () => {
    
    const metadata = wasm.reflex_batch_metadata_js(
        10, 30, 5    
    );
    
    
    assert.strictEqual(metadata.length, 5);
    assert.strictEqual(metadata[0], 10);
    assert.strictEqual(metadata[1], 15);
    assert.strictEqual(metadata[2], 20);
    assert.strictEqual(metadata[3], 25);
    assert.strictEqual(metadata[4], 30);
});

test('Reflex step precision', () => {
    
    const data = new Float64Array(50);
    for (let i = 0; i < 50; i++) {
        data[i] = i + 1;
    }
    
    const batch_result = wasm.reflex_batch_js(
        data,
        10, 20, 2     
    );
    
    
    const rows_cols = wasm.reflex_batch_rows_cols_js(10, 20, 2, data.length);
    const rows = rows_cols[0];
    
    
    assert.strictEqual(rows, 6);
    assert.strictEqual(batch_result.length, 6 * data.length);
    
    
    const metadata = wasm.reflex_batch_metadata_js(10, 20, 2);
    assert.strictEqual(metadata.length, 6);
    assert.deepStrictEqual(Array.from(metadata), [10, 12, 14, 16, 18, 20]);
});

test('Reflex formula verification', () => {
    
    const data = new Float64Array([10.0, 11.0, 12.0, 11.0, 10.0, 9.0, 10.0, 11.0, 12.0, 13.0]);
    const period = 5;
    
    const result = wasm.reflex_js(data, period);
    
    
    assert.strictEqual(result.length, data.length);
    
    
    for (let i = 0; i < period; i++) {
        assert.strictEqual(result[i], 0.0, `Expected zero during warmup at index ${i}`);
    }
    
    
    for (let i = period; i < result.length; i++) {
        assert(isFinite(result[i]), `Expected finite value at index ${i}`);
    }
    
    
    const uniqueValues = new Set();
    for (let i = period; i < result.length; i++) {
        uniqueValues.add(result[i]);
    }
    assert(uniqueValues.size > 1, "Expected varying values for oscillating input");
});
