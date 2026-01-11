/**
 * WASM binding tests for CHOP indicator.
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

test('CHOP partial params', () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    
    const result = wasm.chop_js(high, low, close, 30, 100.0, 1);
    assert.strictEqual(result.length, close.length);
    
    
    for (let i = 0; i < 29; i++) {
        assert.ok(isNaN(result[i]), `Expected NaN at index ${i}`);
    }
    
    
    const validValues = result.slice(29).filter(v => !isNaN(v));
    assert.ok(validValues.length > 0, "Should have valid values after warmup");
});

test('CHOP accuracy', async () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    
    const expectedLast5 = [
        49.98214330294626,
        48.90450693742312,
        46.63648608318844,
        46.19823574588033,
        56.22876423352909,
    ];
    
    const result = wasm.chop_js(high, low, close, 14, 100.0, 1);
    
    assert.strictEqual(result.length, close.length);
    
    
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expectedLast5,
        1e-8,
        "CHOP last 5 values mismatch"
    );
    
    
    await compareWithRust('chop', result, 'hlc', {period: 14, scalar: 100.0, drift: 1});
});

test('CHOP default candles', () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const result = wasm.chop_js(high, low, close, 14, 100.0, 1);
    assert.strictEqual(result.length, close.length);
    
    
    assertAllNaN(result.slice(0, 13), "Expected NaN in warmup period");
});

test('CHOP zero period', () => {
    
    const data = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.chop_js(data, data, data, 0, 100.0, 1);
    }, /Invalid period/);
});

test('CHOP period exceeds length', () => {
    
    const smallData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.chop_js(smallData, smallData, smallData, 10, 100.0, 1);
    }, /Invalid period/);
});

test('CHOP very small dataset', () => {
    
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.chop_js(singlePoint, singlePoint, singlePoint, 14, 100.0, 1);
    }, /Invalid period|Not enough valid data/);
});

test('CHOP empty input', () => {
    
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.chop_js(empty, empty, empty, 14, 100.0, 1);
    }, /Empty|empty/);
});

test('CHOP invalid drift', () => {
    
    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
    
    assert.throws(() => {
        wasm.chop_js(data, data, data, 3, 100.0, 0);
    }, /Invalid drift|drift/);
});

test('CHOP all NaN input', () => {
    
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.chop_js(allNaN, allNaN, allNaN, 14, 100.0, 1);
    }, /All.*NaN/);
});

test('CHOP basic functionality', () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    
    const result = wasm.chop_js(high, low, close, 14, 100.0, 1);
    
    assert.strictEqual(result.length, close.length);
    assert.ok(result instanceof Float64Array);
    
    
    for (let i = 0; i < 13; i++) {
        assert.ok(isNaN(result[i]), `Expected NaN at index ${i}`);
    }
    
    
    const validValues = result.slice(13).filter(v => !isNaN(v));
    assert.ok(validValues.length > 0, "Should have valid values after warmup");
    
    
    for (const val of validValues) {
        assert.ok(val >= 0, `CHOP value ${val} should be non-negative`);
        assert.ok(val <= 200, `CHOP value ${val} seems too large`);
    }
});

test('CHOP nan handling', () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const result = wasm.chop_js(high, low, close, 14, 100.0, 1);
    assert.strictEqual(result.length, close.length);
    
    
    if (result.length > 240) {
        let hasValidValues = false;
        for (let i = 240; i < result.length; i++) {
            if (!isNaN(result[i])) {
                hasValidValues = true;
                break;
            }
        }
        assert.ok(hasValidValues, "Should have some valid values after index 240");
    }
    
    
    assertAllNaN(result.slice(0, 13), "Expected NaN in warmup period");
});

test('CHOP with custom parameters', () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    
    const period = 20;
    const scalar = 50.0;
    const drift = 2;
    
    const result = wasm.chop_js(high, low, close, period, scalar, drift);
    
    assert.strictEqual(result.length, close.length);
    
    
    for (let i = 0; i < period - 1; i++) {
        assert.ok(isNaN(result[i]), `Expected NaN at index ${i}`);
    }
    
    
    const resultDefault = wasm.chop_js(high, low, close, 14, 100.0, 1);
    let differentCount = 0;
    for (let i = period; i < result.length; i++) {
        if (!isNaN(result[i]) && !isNaN(resultDefault[i]) && 
            Math.abs(result[i] - resultDefault[i]) > 1e-10) {
            differentCount++;
        }
    }
    assert.ok(differentCount > 0, "Custom parameters should produce different results");
});



test('CHOP fast API (in-place)', () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    
    const highPtr = wasm.chop_alloc(len);
    const lowPtr = wasm.chop_alloc(len);
    const closePtr = wasm.chop_alloc(len);
    const outPtr = wasm.chop_alloc(len);
    
    try {
        
        const highView = new Float64Array(wasm.__wasm.memory.buffer, highPtr, len);
        const lowView = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, len);
        const closeView = new Float64Array(wasm.__wasm.memory.buffer, closePtr, len);
        
        highView.set(high);
        lowView.set(low);
        closeView.set(close);
        
        
        wasm.chop_into(
            highPtr,
            lowPtr,
            closePtr,
            outPtr,
            len,
            14, 100.0, 1
        );
        
        
        const result = Array.from(new Float64Array(wasm.__wasm.memory.buffer, outPtr, len));
        
        
        const expected = wasm.chop_js(high, low, close, 14, 100.0, 1);
        assertArrayClose(result, expected, 1e-10, "Fast API should match safe API");
        
        
        wasm.chop_into(
            highPtr,
            lowPtr,
            closePtr,
            closePtr,  
            len,
            14, 100.0, 1
        );
        
        
        const closeResult = Array.from(new Float64Array(wasm.__wasm.memory.buffer, closePtr, len));
        assertArrayClose(closeResult, expected, 1e-10, "In-place operation should match safe API");
        
    } finally {
        wasm.chop_free(highPtr, len);
        wasm.chop_free(lowPtr, len);
        wasm.chop_free(closePtr, len);
        wasm.chop_free(outPtr, len);
    }
});

test('CHOP batch processing', () => {
    
    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const config = {
        period_range: [10, 20, 5],    
        scalar_range: [50.0, 100.0, 50.0], 
        drift_range: [1, 2, 1]        
    };
    
    const result = wasm.chop_batch(high, low, close, config);
    
    assert.ok(result.values);
    assert.ok(result.combos);
    assert.ok(result.rows);
    assert.ok(result.cols);
    
    
    assert.strictEqual(result.rows, 12);
    assert.strictEqual(result.cols, close.length);
    assert.strictEqual(result.combos.length, 12);
    assert.strictEqual(result.values.length, 12 * close.length);
    
    
    const combos = result.combos;
    const periods = combos.map(c => c.period).filter((v, i, a) => a.indexOf(v) === i);
    const scalars = combos.map(c => c.scalar).filter((v, i, a) => a.indexOf(v) === i);
    const drifts = combos.map(c => c.drift).filter((v, i, a) => a.indexOf(v) === i);
    
    assert.deepStrictEqual(periods.sort((a,b) => a-b), [10, 15, 20]);
    assert.deepStrictEqual(scalars.sort((a,b) => a-b), [50.0, 100.0]);
    assert.deepStrictEqual(drifts.sort((a,b) => a-b), [1, 2]);
});

test('CHOP batch fast API', () => {
    
    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    const close = new Float64Array(testData.close.slice(0, 100));
    const len = close.length;
    
    
    const expectedRows = 3 * 2 * 2; 
    const totalSize = expectedRows * len;
    
    
    const highPtr = wasm.chop_alloc(len);
    const lowPtr = wasm.chop_alloc(len);
    const closePtr = wasm.chop_alloc(len);
    
    
    const outPtr = wasm.chop_alloc(totalSize);
    
    try {
        
        const highView = new Float64Array(wasm.__wasm.memory.buffer, highPtr, len);
        const lowView = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, len);
        const closeView = new Float64Array(wasm.__wasm.memory.buffer, closePtr, len);
        
        highView.set(high);
        lowView.set(low);
        closeView.set(close);
        
        const rows = wasm.chop_batch_into(
            highPtr,
            lowPtr,
            closePtr,
            outPtr,
            len,
            10, 20, 5,      
            50.0, 100.0, 50.0, 
            1, 2, 1         
        );
        
        assert.strictEqual(rows, expectedRows);
        
        
        const result = new Float64Array(wasm.__wasm.memory.buffer, outPtr, totalSize);
        
        
        const firstRow = result.slice(0, len);
        const expected = wasm.chop_js(high, low, close, 10, 50.0, 1);
        assertArrayClose(firstRow, expected, 1e-10, "First batch row should match single calculation");
        
    } finally {
        wasm.chop_free(highPtr, len);
        wasm.chop_free(lowPtr, len);
        wasm.chop_free(closePtr, len);
        wasm.chop_free(outPtr, totalSize);
    }
});

test('CHOP batch metadata verification', () => {
    
    const high = new Float64Array(testData.high.slice(0, 50));
    const low = new Float64Array(testData.low.slice(0, 50));
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const config = {
        period_range: [10, 15, 5],    
        scalar_range: [50.0, 100.0, 50.0], 
        drift_range: [1, 2, 1]        
    };
    
    const result = wasm.chop_batch(high, low, close, config);
    
    
    assert.strictEqual(result.rows, 8);
    assert.strictEqual(result.cols, close.length);
    assert.strictEqual(result.combos.length, 8);
    
    
    assert.strictEqual(result.combos[0].period, 10);
    assert.strictEqual(result.combos[0].scalar, 50.0);
    assert.strictEqual(result.combos[0].drift, 1);
    
    
    assert.strictEqual(result.combos[7].period, 15);
    assert.strictEqual(result.combos[7].scalar, 100.0);
    assert.strictEqual(result.combos[7].drift, 2);
    
    
    const firstRow = result.values.slice(0, close.length);
    const expected = wasm.chop_js(high, low, close, 10, 50.0, 1);
    assertArrayClose(firstRow, expected, 1e-10, "First batch row should match single calculation");
});

test('CHOP batch single parameter set', () => {
    
    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const config = {
        period_range: [14, 14, 0],
        scalar_range: [100.0, 100.0, 0],
        drift_range: [1, 1, 0]
    };
    
    const batchResult = wasm.chop_batch(high, low, close, config);
    
    
    const singleResult = wasm.chop_js(high, low, close, 14, 100.0, 1);
    
    assert.strictEqual(batchResult.values.length, singleResult.length);
    assertArrayClose(batchResult.values, singleResult, 1e-10, "Batch vs single mismatch");
});

test('CHOP batch parameter sweep', () => {
    
    const high = new Float64Array(testData.high.slice(0, 50));
    const low = new Float64Array(testData.low.slice(0, 50));
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const config = {
        period_range: [10, 20, 5],      
        scalar_range: [50.0, 100.0, 50.0], 
        drift_range: [1, 2, 1]           
    };
    
    const result = wasm.chop_batch(high, low, close, config);
    
    
    assert.strictEqual(result.rows, 12);
    assert.strictEqual(result.cols, close.length);
    
    
    const periods = new Set(result.combos.map(c => c.period));
    const scalars = new Set(result.combos.map(c => c.scalar));
    const drifts = new Set(result.combos.map(c => c.drift));
    
    assert.deepStrictEqual([...periods].sort(), [10, 15, 20]);
    assert.deepStrictEqual([...scalars].sort((a,b) => a-b), [50.0, 100.0]);
    assert.deepStrictEqual([...drifts].sort(), [1, 2]);
    
    
    for (let i = 0; i < result.combos.length; i++) {
        if (result.combos[i].period === 10 &&
            result.combos[i].scalar === 50.0 &&
            result.combos[i].drift === 1) {
            const rowStart = i * close.length;
            const rowEnd = rowStart + close.length;
            const rowData = result.values.slice(rowStart, rowEnd);
            const expected = wasm.chop_js(high, low, close, 10, 50.0, 1);
            assertArrayClose(rowData, expected, 1e-10, "Batch row should match single");
            break;
        }
    }
});

test('CHOP consistency check', () => {
    
    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    const close = new Float64Array(testData.close.slice(0, 100));
    
    
    const result1 = wasm.chop_js(high, low, close, 14, 100.0, 1);
    const result2 = wasm.chop_js(high, low, close, 20, 50.0, 2);
    
    let differentCount = 0;
    for (let i = 20; i < result1.length; i++) {
        if (!isNaN(result1[i]) && !isNaN(result2[i]) &&
            Math.abs(result1[i] - result2[i]) > 1e-10) {
            differentCount++;
        }
    }
    assert.ok(differentCount > 0, "Different parameters should produce different results");
    
    
    const periods = [10, 14, 20, 30];
    for (const period of periods) {
        const result = wasm.chop_js(high, low, close, period, 100.0, 1);
        
        
        for (let i = 0; i < period - 1; i++) {
            assert.ok(isNaN(result[i]), `Expected NaN at index ${i} for period ${period}`);
        }
        
        
        let hasValid = false;
        for (let i = period; i < Math.min(period + 10, result.length); i++) {
            if (!isNaN(result[i])) {
                hasValid = true;
                break;
            }
        }
        assert.ok(hasValid, `Should have valid values after warmup for period ${period}`);
    }
});

test('CHOP memory management', () => {
    
    const sizes = [10, 100, 1000, 10000];
    
    for (const size of sizes) {
        const ptr = wasm.chop_alloc(size);
        assert.ok(ptr !== 0, `Should allocate memory for size ${size}`);
        
        
        const arr = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        for (let i = 0; i < size; i++) {
            arr[i] = i * 1.5;
        }
        
        
        assert.doesNotThrow(() => {
            wasm.chop_free(ptr, size);
        }, `Should free memory for size ${size}`);
    }
    
    
    assert.doesNotThrow(() => {
        wasm.chop_free(0, 100);
    }, "Should handle null pointer in free");
});

test('CHOP null pointer handling', () => {
    
    const len = 100;
    
    assert.throws(() => {
        wasm.chop_into(0, 0, 0, 0, len, 14, 100.0, 1);
    }, /Null pointer/);
    
    const validPtr = wasm.chop_alloc(len);
    try {
        assert.throws(() => {
            wasm.chop_into(validPtr, 0, validPtr, validPtr, len, 14, 100.0, 1);
        }, /Null pointer/);
    } finally {
        wasm.chop_free(validPtr, len);
    }
});

test('CHOP NaN input handling', () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    
    high[10] = NaN;
    high[11] = NaN;
    low[15] = NaN;
    close[5] = NaN;
    
    const result = wasm.chop_js(high, low, close, 14, 100.0, 1);
    
    assert.strictEqual(result.length, close.length);
    
    
    
    const validFromIndex = 30; 
    const validValues = result.slice(validFromIndex).filter(v => !isNaN(v));
    assert.ok(validValues.length > 0, "Should have valid values after NaN inputs");
});