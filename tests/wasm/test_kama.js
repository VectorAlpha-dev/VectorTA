/**
 * WASM binding tests for KAMA indicator.
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

test('KAMA partial params', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.kama_js(close, 30);
    assert.strictEqual(result.length, close.length);
});

test('KAMA accuracy', async () => {
    
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.kama;
    
    const result = wasm.kama_js(
        close,
        expected.defaultParams.period
    );
    
    assert.strictEqual(result.length, close.length);
    
    
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-6,
        "KAMA last 5 values mismatch"
    );
    
    
    await compareWithRust('kama', result, 'close', expected.defaultParams);
});

test('KAMA default candles', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.kama_js(close, 30);
    assert.strictEqual(result.length, close.length);
});

test('KAMA zero period', () => {
    
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.kama_js(inputData, 0);
    }, /Invalid period/);
});

test('KAMA period exceeds length', () => {
    
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.kama_js(dataSmall, 10);
    }, /Invalid period/);
});

test('KAMA very small dataset', () => {
    
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.kama_js(singlePoint, 30);
    }, /Invalid period|Not enough valid data/);
});

test('KAMA empty input', () => {
    
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.kama_js(empty, 30);
    }, /Input data slice is empty/);
});

test('KAMA all NaN input', () => {
    
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.kama_js(allNaN, 30);
    }, /All values are NaN/);
});

test('KAMA NaN handling', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.kama_js(close, 30);
    assert.strictEqual(result.length, close.length);
    
    
    if (result.length > 30) {
        for (let i = 30; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
    
    
    assertAllNaN(result.slice(0, 30), "Expected NaN in warmup period");
});

test('KAMA batch single parameter set', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const batchResult = wasm.kama_batch(close, {
        period_range: [30, 30, 0]
    });
    
    
    const singleResult = wasm.kama_js(close, 30);
    
    assert.strictEqual(batchResult.values.length, singleResult.length);
    assertArrayClose(batchResult.values, singleResult, 1e-10, "Batch vs single mismatch");
});

test('KAMA batch multiple periods', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100)); 
    
    
    const batchResult = wasm.kama_batch(close, {
        period_range: [10, 40, 10]
    });
    
    
    assert.strictEqual(batchResult.values.length, 4 * 100);
    assert.strictEqual(batchResult.rows, 4);
    assert.strictEqual(batchResult.cols, 100);
    
    
    const periods = [10, 20, 30, 40];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.values.slice(rowStart, rowEnd);
        
        const singleResult = wasm.kama_js(close, periods[i]);
        assertArrayClose(
            rowData, 
            singleResult, 
            1e-10, 
            `Period ${periods[i]} mismatch`
        );
    }
});

test('KAMA batch metadata from result', () => {
    
    const close = new Float64Array(50); 
    close.fill(100);
    
    const result = wasm.kama_batch(close, {
        period_range: [10, 30, 10]  
    });
    
    
    assert.strictEqual(result.combos.length, 3);
    
    
    assert.strictEqual(result.combos[0].period, 10);
    assert.strictEqual(result.combos[1].period, 20);
    assert.strictEqual(result.combos[2].period, 30);
});

test('KAMA batch full parameter sweep', () => {
    
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const batchResult = wasm.kama_batch(close, {
        period_range: [5, 15, 5]  
    });
    
    
    assert.strictEqual(batchResult.combos.length, 3);
    assert.strictEqual(batchResult.rows, 3);
    assert.strictEqual(batchResult.cols, 50);
    assert.strictEqual(batchResult.values.length, 3 * 50);
    
    
    for (let combo = 0; combo < batchResult.combos.length; combo++) {
        const period = batchResult.combos[combo].period;
        
        const rowStart = combo * 50;
        const rowData = batchResult.values.slice(rowStart, rowStart + 50);
        
        
        for (let i = 0; i < period; i++) {
            assert(isNaN(rowData[i]), `Expected NaN at warmup index ${i} for period ${period}`);
        }
        
        
        for (let i = period; i < 50; i++) {
            assert(!isNaN(rowData[i]), `Unexpected NaN at index ${i} for period ${period}`);
        }
    }
});

test('KAMA batch edge cases', () => {
    
    const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    
    
    const singleBatch = wasm.kama_batch(close, {
        period_range: [5, 5, 1]
    });
    
    assert.strictEqual(singleBatch.values.length, 10);
    assert.strictEqual(singleBatch.combos.length, 1);
    
    
    const largeBatch = wasm.kama_batch(close, {
        period_range: [5, 7, 10] 
    });
    
    
    assert.strictEqual(largeBatch.values.length, 10);
    assert.strictEqual(largeBatch.combos.length, 1);
    
    
    assert.throws(() => {
        wasm.kama_batch(new Float64Array([]), {
            period_range: [30, 30, 0]
        });
    }, /Input data slice is empty/);
});

test('KAMA batch - new ergonomic API with single parameter', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.kama_batch(close, {
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
    
    
    const oldResult = wasm.kama_js(close, 30);
    for (let i = 0; i < oldResult.length; i++) {
        if (isNaN(oldResult[i]) && isNaN(result.values[i])) {
            continue; 
        }
        assert(Math.abs(oldResult[i] - result.values[i]) < 1e-10,
               `Value mismatch at index ${i}`);
    }
});

test('KAMA batch - new API with multiple parameters', () => {
    
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const result = wasm.kama_batch(close, {
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
    
    
    const secondRow = result.values.slice(result.cols, 2 * result.cols);
    assert.strictEqual(secondRow.length, 50);
    
    
    const oldResult = wasm.kama_js(close, 10);
    const firstRow = result.values.slice(0, result.cols);
    for (let i = 0; i < oldResult.length; i++) {
        if (isNaN(oldResult[i]) && isNaN(firstRow[i])) {
            continue; 
        }
        assert(Math.abs(oldResult[i] - firstRow[i]) < 1e-10,
               `Value mismatch at index ${i}`);
    }
});

test('KAMA batch - new API error handling', () => {
    const close = new Float64Array(testData.close.slice(0, 10));
    
    
    assert.throws(() => {
        wasm.kama_batch(close, {
            period_range: [30, 30] 
        });
    }, /Invalid config/);
    
    
    assert.throws(() => {
        wasm.kama_batch(close, {
            
        });
    }, /Invalid config/);
    
    
    assert.throws(() => {
        wasm.kama_batch(close, {
            period_range: "invalid"
        });
    }, /Invalid config/);
});


test('KAMA zero-copy API', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const period = 5;
    
    
    const ptr = wasm.kama_alloc(data.length);
    assert(ptr !== 0, 'Failed to allocate memory');
    
    
    const memView = new Float64Array(
        wasm.__wasm.memory.buffer,
        ptr,
        data.length
    );
    
    
    memView.set(data);
    
    
    try {
        wasm.kama_into(ptr, ptr, data.length, period);
        
        
        const regularResult = wasm.kama_js(data, period);
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                continue; 
            }
            assert(Math.abs(regularResult[i] - memView[i]) < 1e-10,
                   `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${memView[i]}`);
        }
    } finally {
        
        wasm.kama_free(ptr, data.length);
    }
});

test('KAMA zero-copy with large dataset', () => {
    const size = 100000;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.sin(i * 0.01) + Math.random() * 0.1;
    }
    
    const ptr = wasm.kama_alloc(size);
    assert(ptr !== 0, 'Failed to allocate large buffer');
    
    try {
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        memView.set(data);
        
        wasm.kama_into(ptr, ptr, size, 30);
        
        
        const memView2 = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        
        
        for (let i = 0; i < 30; i++) {
            assert(isNaN(memView2[i]), `Expected NaN at warmup index ${i}`);
        }
        
        
        for (let i = 30; i < Math.min(100, size); i++) {
            assert(!isNaN(memView2[i]), `Unexpected NaN at index ${i}`);
        }
    } finally {
        wasm.kama_free(ptr, size);
    }
});


test('KAMA SIMD128 consistency', () => {
    
    
    const testCases = [
        { size: 10, period: 5 },
        { size: 100, period: 30 },
        { size: 1000, period: 50 },
        { size: 10000, period: 100 }
    ];
    
    for (const testCase of testCases) {
        const data = new Float64Array(testCase.size);
        for (let i = 0; i < testCase.size; i++) {
            data[i] = Math.sin(i * 0.1) + Math.cos(i * 0.05);
        }
        
        const result = wasm.kama_js(data, testCase.period);
        
        
        assert.strictEqual(result.length, data.length);
        
        
        for (let i = 0; i < testCase.period; i++) {
            assert(isNaN(result[i]), `Expected NaN at warmup index ${i} for size=${testCase.size}`);
        }
        
        
        let sumAfterWarmup = 0;
        let countAfterWarmup = 0;
        for (let i = testCase.period; i < result.length; i++) {
            assert(!isNaN(result[i]), `Unexpected NaN at index ${i} for size=${testCase.size}`);
            sumAfterWarmup += result[i];
            countAfterWarmup++;
        }
        
        
        const avgAfterWarmup = sumAfterWarmup / countAfterWarmup;
        assert(Math.abs(avgAfterWarmup) < 10, `Average value ${avgAfterWarmup} seems unreasonable`);
    }
});


test('KAMA zero-copy error handling', () => {
    
    assert.throws(() => {
        wasm.kama_into(0, 0, 10, 30);
    }, /null pointer|invalid memory/i);
    
    
    const ptr = wasm.kama_alloc(10);
    try {
        
        assert.throws(() => {
            wasm.kama_into(ptr, ptr, 10, 0);
        }, /Invalid period/);
        
        
        assert.throws(() => {
            wasm.kama_into(ptr, ptr, 10, 20);
        }, /Invalid period/);
    } finally {
        wasm.kama_free(ptr, 10);
    }
});


test('KAMA zero-copy memory management', () => {
    
    const sizes = [100, 1000, 10000, 100000];
    
    for (const size of sizes) {
        const ptr = wasm.kama_alloc(size);
        assert(ptr !== 0, `Failed to allocate ${size} elements`);
        
        
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            memView[i] = i * 1.5;
        }
        
        
        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(memView[i], i * 1.5, `Memory corruption at index ${i}`);
        }
        
        
        wasm.kama_free(ptr, size);
    }
});

test('KAMA two values', () => {
    
    const data = new Float64Array([1.0, 2.0]);
    
    
    const result = wasm.kama_js(data, 1);
    assert.strictEqual(result.length, 2);
    assert(isNaN(result[0])); 
    assert(isFinite(result[1])); 
});

test('KAMA warmup period calculation', () => {
    
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const testCases = [
        { period: 5, expectedWarmup: 5 },
        { period: 10, expectedWarmup: 10 },
        { period: 20, expectedWarmup: 20 },
        { period: 30, expectedWarmup: 30 },
    ];
    
    for (const { period, expectedWarmup } of testCases) {
        const result = wasm.kama_js(close, period);
        
        
        for (let i = 0; i < expectedWarmup && i < result.length; i++) {
            assert(isNaN(result[i]), `Expected NaN at index ${i} for period=${period}`);
        }
        
        
        if (expectedWarmup < result.length) {
            assert(!isNaN(result[expectedWarmup]), 
                `Expected valid value at index ${expectedWarmup} for period=${period}`);
        }
    }
});

test('KAMA consistency across calls', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const result1 = wasm.kama_js(close, 30);
    const result2 = wasm.kama_js(close, 30);
    
    assertArrayClose(result1, result2, 1e-15, "KAMA results not consistent");
});

test('KAMA batch performance comparison', () => {
    
    const close = new Float64Array(testData.close.slice(0, 1000)); 
    
    
    const startBatch = performance.now();
    const batchResult = wasm.kama_batch(close, {
        period_range: [10, 50, 10]  
    });
    const batchTime = performance.now() - startBatch;
    
    const startSingle = performance.now();
    const singleResults = [];
    for (let period = 10; period <= 50; period += 10) {
        singleResults.push(...wasm.kama_js(close, period));
    }
    const singleTime = performance.now() - startSingle;
    
    
    console.log(`Batch time: ${batchTime.toFixed(2)}ms, Single time: ${singleTime.toFixed(2)}ms`);
    
    
    assertArrayClose(batchResult.values, singleResults, 1e-9, 'Batch vs single results');
});

test('KAMA batch - legacy API compatibility', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const batch_result = wasm.kama_batch_js(close, 10, 30, 10);  
    const metadata = wasm.kama_batch_metadata_js(10, 30, 10);
    
    
    assert.strictEqual(metadata.length, 3);
    assert.deepStrictEqual(Array.from(metadata), [10, 20, 30]);
    
    
    assert.strictEqual(batch_result.length, 3 * close.length);
    
    
    const individual_result = wasm.kama_js(close, 10);
    const first_row = batch_result.slice(0, close.length);
    assertArrayClose(first_row, individual_result, 1e-9, 'Legacy API first row');
});

test.after(() => {
    console.log('KAMA WASM tests completed');
});