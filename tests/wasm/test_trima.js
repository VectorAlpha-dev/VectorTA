/**
 * WASM binding tests for TRIMA indicator.
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
        const wasmPath = path.join(__dirname, '../../pkg/my_project.js');
        const importPath = process.platform === 'win32' 
            ? 'file:///' + wasmPath.replace(/\\/g, '/')
            : wasmPath;
        wasm = await import(importPath);
    } catch (error) {
        console.error('Failed to load WASM module:', error);
        throw error;
    }
    
    
    testData = loadTestData();
});

test('trima_partial_params', () => {
    const close = testData.close;
    
    
    const result = wasm.trima_js(close, 30);
    assert.equal(result.length, close.length);
});

test('trima_accuracy', () => {
    const close = testData.close;
    const expected = EXPECTED_OUTPUTS.trima;
    
    const result = wasm.trima_js(close, expected.defaultParams.period);
    
    assert.equal(result.length, close.length);
    
    
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-6,
        "TRIMA last 5 values mismatch"
    );
    
    
    compareWithRust('trima', result, 'close', expected.defaultParams);
});

test('trima_default_candles', () => {
    const close = testData.close;
    
    
    const result = wasm.trima_js(close, 30);
    assert.equal(result.length, close.length);
});

test('trima_zero_period', () => {
    const inputData = [10.0, 20.0, 30.0];
    
    assert.throws(() => {
        wasm.trima_js(inputData, 0);
    }, /Invalid period/);
});

test('trima_period_exceeds_length', () => {
    const dataSmall = [10.0, 20.0, 30.0];
    
    assert.throws(() => {
        wasm.trima_js(dataSmall, 10);
    }, /Invalid period/);
});

test('trima_very_small_dataset', () => {
    const singlePoint = [42.0];
    
    assert.throws(() => {
        wasm.trima_js(singlePoint, 9);
    }, /Invalid period/);
});

test('trima_empty_input', () => {
    const empty = [];
    
    assert.throws(() => {
        wasm.trima_js(empty, 9);
    }, /empty/i);
});

test('trima_period_too_small', () => {
    const data = [1.0, 2.0, 3.0, 4.0, 5.0];
    
    
    assert.throws(() => {
        wasm.trima_js(data, 3);
    }, /Period too small/);
    
    
    assert.throws(() => {
        wasm.trima_js(data, 2);
    }, /Period too small/);
    
    
    assert.throws(() => {
        wasm.trima_js(data, 1);
    }, /Period too small/);
});

test('trima_all_nan_input', () => {
    const allNan = new Array(100).fill(NaN);
    
    assert.throws(() => {
        wasm.trima_js(allNan, 20);
    }, /All values are NaN/);
});

test('trima_reinput', () => {
    const close = testData.close;
    const expected = EXPECTED_OUTPUTS.trima;
    
    
    const firstResult = wasm.trima_js(close, 30);
    assert.equal(firstResult.length, close.length);
    
    
    const secondResult = wasm.trima_js(firstResult, 10);
    assert.equal(secondResult.length, firstResult.length);
    
    
    const last5 = secondResult.slice(-5);
    assertArrayClose(
        last5,
        expected.reinputLast5,
        1e-6,
        "TRIMA re-input last 5 values mismatch"
    );
});

test('trima_nan_handling', () => {
    const close = testData.close;
    
    const result = wasm.trima_js(close, 30);
    assert.equal(result.length, close.length);
    
    
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert.ok(!isNaN(result[i]), `Unexpected NaN at index ${i}`);
        }
    }
    
    
    for (let i = 0; i < 29; i++) {
        assert.ok(isNaN(result[i]), `Expected NaN at index ${i}`);
    }
});

test('trima_batch_old_api', () => {
    const close = testData.close;
    const expected = EXPECTED_OUTPUTS.trima;
    
    const result = wasm.trima_batch_js(close, expected.defaultParams.period, expected.defaultParams.period, 0);
    const metadata = wasm.trima_batch_metadata_js(expected.defaultParams.period, expected.defaultParams.period, 0);
    
    
    assert.equal(metadata.length, 1);
    assert.equal(metadata[0], expected.defaultParams.period);
    
    
    assert.equal(result.length, close.length);
    
    
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-6,
        "TRIMA batch mismatch"
    );
});

test('trima_batch_multiple_periods', () => {
    const close = testData.close;
    
    const result = wasm.trima_batch_js(close, 10, 30, 10);
    const metadata = wasm.trima_batch_metadata_js(10, 30, 10);
    
    
    assert.equal(metadata.length, 3);
    assert.deepEqual(Array.from(metadata), [10, 20, 30]);
    
    
    assert.equal(result.length, 3 * close.length);
});

test('trima_candles_with_nan', () => {
    
    const dataWithNaN = testData.close.slice();
    dataWithNaN[0] = NaN;
    dataWithNaN[1] = NaN;
    dataWithNaN[2] = NaN;
    
    const result = wasm.trima_js(dataWithNaN, 30);
    assert.equal(result.length, dataWithNaN.length);
    
    
    
    let firstValidIdx = 3; 
    let warmup = firstValidIdx + 29; 
    
    for (let i = 0; i < warmup; i++) {
        assert.ok(isNaN(result[i]), `Expected NaN at index ${i}`);
    }
});

test('trima_consistency_check', () => {
    const close = testData.close;
    
    
    const result1 = wasm.trima_js(close, 20);
    const result2 = wasm.trima_js(close, 20);
    
    assert.equal(result1.length, result2.length);
    
    for (let i = 0; i < result1.length; i++) {
        if (isNaN(result1[i]) && isNaN(result2[i])) {
            continue;
        }
        assert.equal(result1[i], result2[i], `Inconsistent results at index ${i}`);
    }
});

test('trima_edge_cases', () => {
    
    const data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    const result = wasm.trima_js(data, 4);
    assert.equal(result.length, data.length);
    
    
    for (let i = 0; i < 3; i++) {
        assert.ok(isNaN(result[i]), `Expected NaN at index ${i}`);
    }
    
    
    for (let i = 4; i < result.length; i++) {
        assert.ok(!isNaN(result[i]), `Unexpected NaN at index ${i}`);
    }
});


test('TRIMA batch - new ergonomic API with single parameter', () => {
    const close = new Float64Array(testData.close);
    
    const result = wasm.trima_batch(close, {
        period_range: [30, 30, 0]
    });
    
    
    assert(result.values, 'Should have values array');
    assert(result.combos, 'Should have combos array');
    assert(typeof result.rows === 'number', 'Should have rows count');
    assert(typeof result.cols === 'number', 'Should have cols count');
    
    
    assert.strictEqual(result.rows, 1);
    assert.strictEqual(result.cols, close.length);
    assert.strictEqual(result.combos.length, 1);
    assert.strictEqual(result.values.length, close.length);
    
    
    const combo = result.combos[0];
    assert.strictEqual(combo.period, 30);
    
    
    const oldResult = wasm.trima_js(close, 30);
    for (let i = 0; i < oldResult.length; i++) {
        if (isNaN(oldResult[i]) && isNaN(result.values[i])) {
            continue; 
        }
        assert(Math.abs(oldResult[i] - result.values[i]) < 1e-9,
               `Value mismatch at index ${i}`);
    }
});

test('TRIMA batch - new API with multiple parameters', () => {
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const result = wasm.trima_batch(close, {
        period_range: [10, 20, 10]  
    });
    
    
    assert.strictEqual(result.rows, 2);
    assert.strictEqual(result.cols, 50);
    assert.strictEqual(result.combos.length, 2);
    assert.strictEqual(result.values.length, 100);
    
    
    const expectedCombos = [
        { period: 10 },
        { period: 20 }
    ];
    
    for (let i = 0; i < expectedCombos.length; i++) {
        assert.strictEqual(result.combos[i].period, expectedCombos[i].period);
    }
    
    
    const oldResult = wasm.trima_js(close, 10);
    const firstRow = result.values.slice(0, result.cols);
    for (let i = 0; i < oldResult.length; i++) {
        if (isNaN(oldResult[i]) && isNaN(firstRow[i])) {
            continue; 
        }
        assert(Math.abs(oldResult[i] - firstRow[i]) < 1e-9,
               `Value mismatch at index ${i}`);
    }
});

test('TRIMA batch - new API error handling', () => {
    const close = new Float64Array(testData.close.slice(0, 10));
    
    
    assert.throws(() => {
        wasm.trima_batch(close, {
            period_range: [9, 9] 
        });
    }, /Invalid config/);
    
    
    assert.throws(() => {
        wasm.trima_batch(close, {
            
        });
    }, /Invalid config/);
    
    
    assert.throws(() => {
        wasm.trima_batch(close, {
            period_range: "invalid"
        });
    }, /Invalid config/);
});


test('TRIMA zero-copy API', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const period = 5;
    
    
    const ptr = wasm.trima_alloc(data.length);
    assert(ptr !== 0, 'Failed to allocate memory');
    
    
    const memView = new Float64Array(
        wasm.__wasm.memory.buffer,
        ptr,
        data.length
    );
    
    
    memView.set(data);
    
    
    try {
        wasm.trima_into(ptr, ptr, data.length, period);
        
        
        const regularResult = wasm.trima_js(data, period);
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                continue; 
            }
            assert(Math.abs(regularResult[i] - memView[i]) < 1e-10,
                   `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${memView[i]}`);
        }
    } finally {
        
        wasm.trima_free(ptr, data.length);
    }
});

test('TRIMA zero-copy with large dataset', () => {
    const size = 100000;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.sin(i * 0.01) + Math.random() * 0.1;
    }
    
    const ptr = wasm.trima_alloc(size);
    assert(ptr !== 0, 'Failed to allocate large buffer');
    
    try {
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        memView.set(data);
        
        wasm.trima_into(ptr, ptr, size, 30);
        
        
        const memView2 = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        
        
        for (let i = 0; i < 29; i++) {
            assert(isNaN(memView2[i]), `Expected NaN at warmup index ${i}`);
        }
        
        
        for (let i = 29; i < Math.min(100, size); i++) {
            assert(!isNaN(memView2[i]), `Unexpected NaN at index ${i}`);
        }
    } finally {
        wasm.trima_free(ptr, size);
    }
});


test('TRIMA zero-copy error handling', () => {
    
    assert.throws(() => {
        wasm.trima_into(0, 0, 10, 5);
    }, /null pointer|invalid memory/i);
    
    
    const ptr = wasm.trima_alloc(10);
    try {
        
        assert.throws(() => {
            wasm.trima_into(ptr, ptr, 10, 0);
        }, /Invalid period/);
        
        
        assert.throws(() => {
            wasm.trima_into(ptr, ptr, 10, 3);
        }, /Invalid period|Period too small/);
    } finally {
        wasm.trima_free(ptr, 10);
    }
});


test('TRIMA zero-copy memory management', () => {
    
    const sizes = [100, 1000, 10000, 100000];
    
    for (const size of sizes) {
        const ptr = wasm.trima_alloc(size);
        assert(ptr !== 0, `Failed to allocate ${size} elements`);
        
        
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            memView[i] = i * 1.5;
        }
        
        
        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(memView[i], i * 1.5, `Memory corruption at index ${i}`);
        }
        
        
        wasm.trima_free(ptr, size);
    }
});


test('TRIMA SIMD128 consistency', () => {
    
    
    const testCases = [
        { size: 10, period: 5 },
        { size: 100, period: 20 },
        { size: 1000, period: 50 },
        { size: 10000, period: 100 }
    ];
    
    for (const testCase of testCases) {
        const data = new Float64Array(testCase.size);
        for (let i = 0; i < testCase.size; i++) {
            data[i] = Math.sin(i * 0.1) + Math.cos(i * 0.05);
        }
        
        const result = wasm.trima_js(data, testCase.period);
        
        
        assert.strictEqual(result.length, data.length);
        
        
        for (let i = 0; i < testCase.period - 1; i++) {
            assert(isNaN(result[i]), `Expected NaN at warmup index ${i} for size=${testCase.size}`);
        }
        
        
        let sumAfterWarmup = 0;
        let countAfterWarmup = 0;
        for (let i = testCase.period - 1; i < result.length; i++) {
            assert(!isNaN(result[i]), `Unexpected NaN at index ${i} for size=${testCase.size}`);
            sumAfterWarmup += result[i];
            countAfterWarmup++;
        }
        
        
        const avgAfterWarmup = sumAfterWarmup / countAfterWarmup;
        assert(Math.abs(avgAfterWarmup) < 10, `Average value ${avgAfterWarmup} seems unreasonable`);
    }
});

test('TRIMA batch edge cases', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100));
    
    
    const singleBatch = wasm.trima_batch(close, {
        period_range: [20, 20, 1]
    });
    
    assert.strictEqual(singleBatch.values.length, 100);
    assert.strictEqual(singleBatch.combos.length, 1);
    
    
    const largeBatch = wasm.trima_batch(close, {
        period_range: [10, 15, 20]  
    });
    
    
    assert.strictEqual(largeBatch.values.length, 100);
    assert.strictEqual(largeBatch.combos.length, 1);
    assert.strictEqual(largeBatch.combos[0].period, 10);
    
    
    assert.throws(() => {
        wasm.trima_batch(new Float64Array([]), {
            period_range: [10, 10, 0]
        });
    }, /All values are NaN|No data/);
});

test('TRIMA batch full parameter sweep', () => {
    
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const batchResult = wasm.trima_batch(close, {
        period_range: [10, 20, 5]  
    });
    
    
    assert.strictEqual(batchResult.combos.length, 3);
    assert.strictEqual(batchResult.rows, 3);
    assert.strictEqual(batchResult.cols, 50);
    assert.strictEqual(batchResult.values.length, 150);
    
    
    const expectedPeriods = [10, 15, 20];
    for (let combo = 0; combo < batchResult.combos.length; combo++) {
        const period = batchResult.combos[combo].period;
        assert.strictEqual(period, expectedPeriods[combo]);
        
        const rowStart = combo * 50;
        const rowData = batchResult.values.slice(rowStart, rowStart + 50);
        
        
        for (let i = 0; i < period - 1; i++) {
            assert(isNaN(rowData[i]), `Expected NaN at warmup index ${i} for period ${period}`);
        }
        
        
        for (let i = period - 1; i < 50; i++) {
            assert(!isNaN(rowData[i]), `Unexpected NaN at index ${i} for period ${period}`);
        }
    }
});

test('TRIMA batch metadata verification', () => {
    
    const close = new Float64Array(30); 
    close.fill(100);
    
    const result = wasm.trima_batch(close, {
        period_range: [10, 20, 10]  
    });
    
    
    assert.strictEqual(result.combos.length, 2);
    
    
    assert.strictEqual(result.combos[0].period, 10);
    assert.strictEqual(result.combos[1].period, 20);
});

test('TRIMA batch vs single consistency', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const periods = [10, 15, 20, 25, 30];
    
    
    const batch = wasm.trima_batch(close, {
        period_range: [10, 30, 5]
    });
    
    
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batch.values.slice(rowStart, rowEnd);
        
        const singleResult = wasm.trima_js(close, periods[i]);
        assertArrayClose(
            Array.from(rowData),
            singleResult,
            1e-10,
            `Period ${periods[i]} mismatch`
        );
    }
});

test('TRIMA batch_into low-level API', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const periods = [5, 7]; 
    const rows = 2;
    const cols = data.length;
    
    
    const inPtr = wasm.trima_alloc(data.length);
    const outPtr = wasm.trima_alloc(rows * cols);
    
    try {
        
        const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, data.length);
        inView.set(data);
        
        
        const resultRows = wasm.trima_batch_into(
            inPtr, outPtr, data.length,
            5, 7, 2  
        );
        
        assert.strictEqual(resultRows, 2, 'Should return 2 rows');
        
        
        const outView = new Float64Array(wasm.__wasm.memory.buffer, outPtr, rows * cols);
        
        
        const firstRow = Array.from(outView.slice(0, cols));
        const expected1 = wasm.trima_js(data, 5);
        assertArrayClose(firstRow, expected1, 1e-10, 'First row mismatch');
        
        
        const secondRow = Array.from(outView.slice(cols, 2 * cols));
        const expected2 = wasm.trima_js(data, 7);
        assertArrayClose(secondRow, expected2, 1e-10, 'Second row mismatch');
    } finally {
        wasm.trima_free(inPtr, data.length);
        wasm.trima_free(outPtr, rows * cols);
    }
});
