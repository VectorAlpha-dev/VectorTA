/**
 * WASM binding tests for SINWMA indicator.
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

test('SINWMA partial params', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.sinwma_js(close, 14);
    assert(result instanceof Float64Array);
    assert.strictEqual(result.length, close.length);
});

test('SINWMA default candles', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.sinwma_js(close, 14);
    assert.strictEqual(result.length, close.length);
});

test('SINWMA accuracy', async () => {
    
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.sinwma;
    
    const result = wasm.sinwma_js(close, expected.defaultParams.period);
    
    assert.strictEqual(result.length, close.length);
    
    
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-6,
        "SINWMA last 5 values mismatch"
    );
    
    
    await compareWithRust('sinwma', result, 'close', expected.defaultParams);
});

test('SINWMA invalid period', () => {
    
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    
    assert.throws(() => {
        wasm.sinwma_js(inputData, 0);
    });
});

test('SINWMA period exceeds length', () => {
    
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.sinwma_js(dataSmall, 10);
    });
});

test('SINWMA very small dataset', () => {
    
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.sinwma_js(singlePoint, 14);
    });
});

test('SINWMA empty input', () => {
    
    const dataEmpty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.sinwma_js(dataEmpty, 14);
    });
});

test('SINWMA period one', () => {
    
    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
    
    const result = wasm.sinwma_js(data, 1);
    
    
    assertArrayClose(result, data, 1e-10, "Period=1 should act as passthrough");
});

test('SINWMA reinput', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const firstResult = wasm.sinwma_js(close, 14);
    
    
    const secondResult = wasm.sinwma_js(firstResult, 5);
    
    assert.strictEqual(secondResult.length, firstResult.length);
    
    
    
    let firstValid = 0;
    for (let i = 0; i < close.length; i++) {
        if (!isNaN(close[i])) {
            firstValid = i;
            break;
        }
    }
    
    let firstValidInFirst = 0;
    for (let i = 0; i < firstResult.length; i++) {
        if (!isNaN(firstResult[i])) {
            firstValidInFirst = i;
            break;
        }
    }
    const warmup = firstValidInFirst + 5 - 1;  
    for (let i = warmup; i < secondResult.length; i++) {
        assert(isFinite(secondResult[i]), `NaN found at index ${i}`);
    }
});

test('SINWMA NaN handling', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.sinwma_js(close, 14);
    
    assert.strictEqual(result.length, close.length);
    
    
    assertAllNaN(result.slice(0, 13), "Expected NaN in warmup period (first 13 values)");
    
    
    
    let firstValid = 0;
    for (let i = 0; i < close.length; i++) {
        if (!isNaN(close[i])) {
            firstValid = i;
            break;
        }
    }
    const warmup = firstValid + 14 - 1;  
    if (result.length > warmup) {
        for (let i = warmup; i < result.length; i++) {
            assert(isFinite(result[i]), `Found unexpected NaN at index ${i} after warmup`);
        }
    }
});

test('SINWMA batch', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const batch_result = wasm.sinwma_batch_js(
        close, 
        10, 30, 5    
    );
    
    
    const rows_cols = wasm.sinwma_batch_rows_cols_js(10, 30, 5, close.length);
    const rows = rows_cols[0];
    const cols = rows_cols[1];
    
    assert(batch_result instanceof Float64Array);
    assert.strictEqual(rows, 5); 
    assert.strictEqual(cols, close.length);
    assert.strictEqual(batch_result.length, rows * cols);
    
    
    const individual_result = wasm.sinwma_js(close, 10);
    const batch_first = batch_result.slice(0, close.length);
    
    
    
    let firstValid = 0;
    for (let i = 0; i < close.length; i++) {
        if (!isNaN(close[i])) {
            firstValid = i;
            break;
        }
    }
    const warmup = firstValid + 10 - 1;  
    for (let i = warmup; i < close.length; i++) {
        assertClose(batch_first[i], individual_result[i], 1e-9, `Batch mismatch at ${i}`);
    }
});

test('SINWMA different periods', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const testPeriods = [5, 10, 14, 20, 50];
    
    for (const period of testPeriods) {
        const result = wasm.sinwma_js(close, period);
        assert.strictEqual(result.length, close.length);
        
        
        
        let firstValid = 0;
        for (let i = 0; i < close.length; i++) {
            if (!isNaN(close[i])) {
                firstValid = i;
                break;
            }
        }
        const warmup = firstValid + period - 1;
        if (warmup < result.length) {
            for (let i = warmup; i < result.length; i++) {
                assert(isFinite(result[i]), `NaN at index ${i} for period=${period}`);
            }
        }
    }
});

test('SINWMA batch performance', () => {
    
    const close = new Float64Array(testData.close.slice(0, 1000)); 
    
    
    const startBatch = performance.now();
    const batchResult = wasm.sinwma_batch_js(
        close,
        10, 50, 10    
    );
    const batchTime = performance.now() - startBatch;
    
    
    const metadata = wasm.sinwma_batch_metadata_js(10, 50, 10);
    
    const startSingle = performance.now();
    const singleResults = [];
    
    for (const period of metadata) {
        const result = wasm.sinwma_js(close, period);
        singleResults.push(...result);
    }
    const singleTime = performance.now() - startSingle;
    
    
    console.log(`Batch time: ${batchTime.toFixed(2)}ms, Single time: ${singleTime.toFixed(2)}ms`);
    
    
    assertArrayClose(batchResult, singleResults, 1e-9, 'Batch vs single results');
});

test('SINWMA edge cases', () => {
    
    
    
    const data = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        data[i] = i + 1;
    }
    const result = wasm.sinwma_js(data, 14);
    assert.strictEqual(result.length, data.length);
    
    
    for (let i = 14; i < result.length; i++) {
        assert(isFinite(result[i]), `NaN at index ${i}`);
    }
    
    
    const constantData = new Float64Array(100).fill(50.0);
    const constantResult = wasm.sinwma_js(constantData, 14);
    
    assert.strictEqual(constantResult.length, constantData.length);
    
    
    for (let i = 14; i < constantResult.length; i++) {
        assert(isFinite(constantResult[i]), `NaN at index ${i} for constant input`);
    }
});

test('SINWMA batch metadata', () => {
    
    const metadata = wasm.sinwma_batch_metadata_js(
        10, 30, 5    
    );
    
    
    assert.strictEqual(metadata.length, 5);
    assert.strictEqual(metadata[0], 10);
    assert.strictEqual(metadata[1], 15);
    assert.strictEqual(metadata[2], 20);
    assert.strictEqual(metadata[3], 25);
    assert.strictEqual(metadata[4], 30);
});

test('SINWMA consistency across calls', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const result1 = wasm.sinwma_js(close, 14);
    const result2 = wasm.sinwma_js(close, 14);
    
    assertArrayClose(result1, result2, 1e-15, "SINWMA results not consistent");
});

test('SINWMA step precision', () => {
    
    const data = new Float64Array(50);
    for (let i = 0; i < 50; i++) {
        data[i] = i + 1;
    }
    
    const batch_result = wasm.sinwma_batch_js(
        data,
        10, 20, 2     
    );
    
    
    const rows_cols = wasm.sinwma_batch_rows_cols_js(10, 20, 2, data.length);
    const rows = rows_cols[0];
    
    
    assert.strictEqual(rows, 6);
    assert.strictEqual(batch_result.length, 6 * data.length);
    
    
    const metadata = wasm.sinwma_batch_metadata_js(10, 20, 2);
    assert.strictEqual(metadata.length, 6);
    assert.strictEqual(metadata[0], 10);
    assert.strictEqual(metadata[1], 12);
    assert.strictEqual(metadata[2], 14);
    assert.strictEqual(metadata[3], 16);
    assert.strictEqual(metadata[4], 18);
    assert.strictEqual(metadata[5], 20);
});

test('SINWMA warmup behavior', () => {
    
    const close = new Float64Array(testData.close);
    const period = 14;
    
    const result = wasm.sinwma_js(close, period);
    
    
    let firstValid = 0;
    for (let i = 0; i < close.length; i++) {
        if (!isNaN(close[i])) {
            firstValid = i;
            break;
        }
    }
    
    const warmup = firstValid + period - 1;
    
    
    for (let i = firstValid; i < warmup && i < result.length; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i} during warmup`);
    }
    
    
    for (let i = warmup; i < result.length; i++) {
        assert(isFinite(result[i]), `Expected finite value at index ${i} after warmup`);
    }
});

test('SINWMA oscillating data', () => {
    
    const data = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        data[i] = (i % 2 === 0) ? 10.0 : 20.0;
    }
    
    const result = wasm.sinwma_js(data, 14);
    assert.strictEqual(result.length, data.length);
    
    
    for (let i = 14; i < result.length; i++) {
        assert(isFinite(result[i]), `Expected finite value at index ${i}`);
    }
});

test('SINWMA small step size', () => {
    
    const data = new Float64Array(50);
    for (let i = 0; i < 50; i++) {
        data[i] = i + 1;
    }
    
    const batch_result = wasm.sinwma_batch_js(
        data,
        10, 14, 1     
    );
    
    const rows_cols = wasm.sinwma_batch_rows_cols_js(10, 14, 1, data.length);
    const rows = rows_cols[0];
    
    assert.strictEqual(rows, 5);
    assert.strictEqual(batch_result.length, 5 * data.length);
});

test('SINWMA formula verification', () => {
    
    const data = new Float64Array([10.0, 11.0, 12.0, 11.0, 10.0, 9.0, 10.0, 11.0, 12.0, 13.0]);
    const period = 5;
    
    const result = wasm.sinwma_js(data, period);
    
    
    assert.strictEqual(result.length, data.length);
    
    
    for (let i = period; i < result.length; i++) {
        assert(isFinite(result[i]), `Expected finite value at index ${i}`);
    }
});

test('SINWMA all NaN input', () => {
    
    const allNaN = new Float64Array(100).fill(NaN);
    
    assert.throws(() => {
        wasm.sinwma_js(allNaN, 14);
    }, /All values are NaN/);
});

test('SINWMA batch error conditions', () => {
    
    const data = new Float64Array([1, 2, 3, 4, 5]);
    
    
    assert.throws(() => {
        wasm.sinwma_batch_js(data, 10, 20, 5);
    });
    
    
    const empty = new Float64Array([]);
    assert.throws(() => {
        wasm.sinwma_batch_js(empty, 10, 20, 5);
    });
});


test('SINWMA zero-copy in-place operation', () => {
    
    const data = new Float64Array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]);
    const period = 5;
    
    
    const ptr = wasm.sinwma_alloc(data.length);
    assert(ptr !== 0, 'Failed to allocate memory');
    
    
    const memory = wasm.__wasm.memory;
    const memView = new Float64Array(
        memory.buffer,
        ptr,
        data.length
    );
    
    
    memView.set(data);
    
    
    try {
        wasm.sinwma_into(ptr, ptr, data.length, period);
        
        
        const regularResult = wasm.sinwma_js(data, period);
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                continue; 
            }
            assert(Math.abs(regularResult[i] - memView[i]) < 1e-10,
                   `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${memView[i]}`);
        }
    } finally {
        
        wasm.sinwma_free(ptr, data.length);
    }
});

test('SINWMA zero-copy with large dataset', () => {
    const size = 100000;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.sin(i * 0.01) + Math.random() * 0.1;
    }
    
    const ptr = wasm.sinwma_alloc(size);
    assert(ptr !== 0, 'Failed to allocate large buffer');
    
    try {
        const memory = wasm.__wasm.memory;
        const memView = new Float64Array(memory.buffer, ptr, size);
        memView.set(data);
        
        wasm.sinwma_into(ptr, ptr, size, 14);
        
        
        const memory2 = wasm.__wasm.memory;
        const memView2 = new Float64Array(memory2.buffer, ptr, size);
        
        
        for (let i = 0; i < 13; i++) {
            assert(isNaN(memView2[i]), `Expected NaN at warmup index ${i}`);
        }
        
        
        for (let i = 13; i < Math.min(100, size); i++) {
            assert(!isNaN(memView2[i]), `Unexpected NaN at index ${i}`);
        }
    } finally {
        wasm.sinwma_free(ptr, size);
    }
});

test('SINWMA zero-copy error handling', () => {
    
    assert.throws(() => {
        wasm.sinwma_into(0, 0, 10, 14);
    }, /null pointer|invalid memory/i);
    
    
    const ptr = wasm.sinwma_alloc(10);
    try {
        
        assert.throws(() => {
            wasm.sinwma_into(ptr, ptr, 10, 0);
        }, /Invalid period/);
        
        
        assert.throws(() => {
            wasm.sinwma_into(ptr, ptr, 10, 20);
        }, /Invalid period/);
    } finally {
        wasm.sinwma_free(ptr, 10);
    }
});

test('SINWMA zero-copy memory management', () => {
    
    const sizes = [100, 1000, 10000, 100000];
    
    for (const size of sizes) {
        const ptr = wasm.sinwma_alloc(size);
        assert(ptr !== 0, `Failed to allocate ${size} elements`);
        
        
        const memory = wasm.__wasm.memory;
        const memView = new Float64Array(memory.buffer, ptr, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            memView[i] = i * 1.5;
        }
        
        
        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(memView[i], i * 1.5, `Memory corruption at index ${i}`);
        }
        
        
        wasm.sinwma_free(ptr, size);
    }
});

test('SINWMA batch new API', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const config = {
        period_range: [10, 20, 5]  
    };
    
    const result = wasm.sinwma_batch(close, config);
    
    
    assert(result.values, 'Result should have values array');
    assert(result.periods, 'Result should have periods array');
    assert(result.rows, 'Result should have rows count');
    assert(result.cols, 'Result should have cols count');
    
    
    assert.strictEqual(result.rows, 3, 'Should have 3 parameter combinations');
    assert.strictEqual(result.cols, close.length, 'Cols should match input length');
    assert.strictEqual(result.values.length, result.rows * result.cols, 'Values array size mismatch');
    
    
    assert.deepStrictEqual(result.periods, [10, 15, 20], 'Periods mismatch');
    
    
    const individual = wasm.sinwma_js(close, 10);
    const firstRow = result.values.slice(0, close.length);
    
    for (let i = 9; i < close.length; i++) {  
        assertClose(firstRow[i], individual[i], 1e-9, 
            `Batch result mismatch at index ${i}`);
    }
});

test('SINWMA leading NaN', () => {
    
    const data = new Float64Array([NaN, NaN, NaN, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    
    const result = wasm.sinwma_js(data, 5);
    
    assert.strictEqual(result.length, data.length);
    
    
    assertAllNaN(result.slice(0, 7), "Expected NaN during warmup with leading NaN");
    for (let i = 7; i < result.length; i++) {
        assert(isFinite(result[i]), `Expected finite value at index ${i} after warmup`);
    }
});

test('SINWMA batch single parameter', () => {
    
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.sinwma;
    
    const config = {
        period_range: [14, 14, 0]  
    };
    
    const result = wasm.sinwma_batch(close, config);
    
    
    assert(result.values, 'Should have values array');
    assert(result.periods, 'Should have periods array');
    assert(typeof result.rows === 'number', 'Should have rows count');
    assert(typeof result.cols === 'number', 'Should have cols count');
    
    
    assert.strictEqual(result.rows, 1);
    assert.strictEqual(result.cols, close.length);
    assert.strictEqual(result.periods[0], 14);
    
    
    const defaultRow = result.values.slice(0, close.length);
    
    
    const last5 = defaultRow.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-6,
        "SINWMA batch default row mismatch"
    );
});

test('SINWMA constant input normalization', () => {
    
    const data = new Float64Array(20).fill(1.0);
    
    
    for (const period of [3, 5, 7]) {
        const result = wasm.sinwma_js(data, period);
        
        
        let firstValidOutput = null;
        for (let i = period; i < result.length; i++) {
            if (isFinite(result[i])) {
                if (firstValidOutput === null) {
                    firstValidOutput = result[i];
                } else {
                    assertClose(result[i], firstValidOutput, 1e-9, 
                        `Expected constant output for period=${period} at index ${i}`);
                }
            }
        }
    }
});
