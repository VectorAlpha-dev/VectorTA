/**
 * WASM binding tests for Linear Regression Intercept indicator.
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
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }
    
    testData = loadTestData();
});

test('Linear Regression Intercept partial params', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.linearreg_intercept_js(close, 14);
    assert.strictEqual(result.length, close.length);
});

test('Linear Regression Intercept accuracy', async () => {
    
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.linearreg_intercept;
    
    const result = wasm.linearreg_intercept_js(
        close,
        expected.defaultParams.period
    );
    
    assert.strictEqual(result.length, close.length);
    
    
    const last5 = result.slice(-5);
    
    
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-6,
        "Linear Regression Intercept last 5 values mismatch"
    );
    
    
    await compareWithRust('linearreg_intercept', result, 'close', expected.defaultParams);
});

test('Linear Regression Intercept default candles', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.linearreg_intercept_js(close, 14);
    assert.strictEqual(result.length, close.length);
});

test('Linear Regression Intercept zero period', () => {
    
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.linearreg_intercept_js(inputData, 0);
    }, /Invalid period/);
});

test('Linear Regression Intercept period exceeds length', () => {
    
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.linearreg_intercept_js(dataSmall, 10);
    }, /Invalid period/);
});

test('Linear Regression Intercept very small dataset', () => {
    
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.linearreg_intercept_js(singlePoint, 14);
    }, /Invalid period|Not enough valid data/);
});

test('Linear Regression Intercept empty input', () => {
    
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.linearreg_intercept_js(empty, 14);
    }, /Input data slice is empty/);
});

test('Linear Regression Intercept reinput', () => {
    
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.linearreg_intercept;
    
    
    const firstResult = wasm.linearreg_intercept_js(close, 14);
    assert.strictEqual(firstResult.length, close.length);
    
    
    const secondResult = wasm.linearreg_intercept_js(firstResult, 14);
    assert.strictEqual(secondResult.length, firstResult.length);
    
    
    if (expected.reinputLast5) {
        const last5 = secondResult.slice(-5);
        assertArrayClose(
            last5,
            expected.reinputLast5,
            1e-8,
            "Linear Regression Intercept re-input last 5 values mismatch"
        );
    }
});

test('Linear Regression Intercept NaN handling', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.linearreg_intercept_js(close, 14);
    assert.strictEqual(result.length, close.length);
    
    
    assertAllNaN(result.slice(0, 13), "Expected NaN in warmup period");
    
    
    for (let i = 13; i < Math.min(100, result.length); i++) {
        assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
    }
});

test('Linear Regression Intercept all NaN input', () => {
    
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.linearreg_intercept_js(allNaN, 14);
    }, /All values are NaN/);
});

test('Linear Regression Intercept period=1 edge case', () => {
    
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    
    const result = wasm.linearreg_intercept_js(data, 1);
    
    
    assertArrayClose(result, data, 1e-10, "Period=1 should return input values");
});

test('Linear Regression Intercept linear trend property', () => {
    
    const data = new Float64Array(50);
    for (let i = 0; i < 50; i++) {
        data[i] = 2.0 * i + 10.0;
    }
    const period = 10;
    
    const result = wasm.linearreg_intercept_js(data, period);
    
    
    const warmup = period - 1;
    for (let i = warmup + 5; i < warmup + 10; i++) {
        const windowStart = i - period + 1;
        const expected = data[windowStart];
        assertClose(result[i], expected, 1e-9, 
                   `Linear trend mismatch at index ${i}`);
    }
});

test('Linear Regression Intercept batch single parameter set', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const batchResult = wasm.linearreg_intercept_batch(close, {
        period_range: [14, 14, 0]
    });
    
    
    const singleResult = wasm.linearreg_intercept_js(close, 14);
    
    assert.strictEqual(batchResult.values.length, singleResult.length);
    assertArrayClose(batchResult.values, singleResult, 1e-10, "Batch vs single mismatch");
});

test('Linear Regression Intercept batch multiple periods', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100)); 
    
    
    const batchResult = wasm.linearreg_intercept_batch(close, {
        period_range: [10, 14, 2]      
    });
    
    
    assert.strictEqual(batchResult.values.length, 3 * 100);
    assert.strictEqual(batchResult.rows, 3);
    assert.strictEqual(batchResult.cols, 100);
    
    
    const periods = [10, 12, 14];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.values.slice(rowStart, rowEnd);
        
        const singleResult = wasm.linearreg_intercept_js(close, periods[i]);
        assertArrayClose(
            rowData, 
            singleResult, 
            1e-10, 
            `Period ${periods[i]} mismatch`
        );
    }
});

test('Linear Regression Intercept batch metadata from result', () => {
    
    const close = new Float64Array(20); 
    close.fill(100);
    
    const result = wasm.linearreg_intercept_batch(close, {
        period_range: [10, 14, 2]      
    });
    
    
    assert.strictEqual(result.combos.length, 3);
    
    
    assert.strictEqual(result.combos[0].period, 10);
    
    
    assert.strictEqual(result.combos[2].period, 14);
});

test('Linear Regression Intercept batch full parameter sweep', () => {
    
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const batchResult = wasm.linearreg_intercept_batch(close, {
        period_range: [10, 15, 5]      
    });
    
    
    assert.strictEqual(batchResult.combos.length, 2);
    assert.strictEqual(batchResult.rows, 2);
    assert.strictEqual(batchResult.cols, 50);
    assert.strictEqual(batchResult.values.length, 2 * 50);
    
    
    for (let combo = 0; combo < batchResult.combos.length; combo++) {
        const period = batchResult.combos[combo].period;
        
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

test('Linear Regression Intercept batch edge cases', () => {
    
    const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    
    
    const singleBatch = wasm.linearreg_intercept_batch(close, {
        period_range: [5, 5, 1]
    });
    
    assert.strictEqual(singleBatch.values.length, 10);
    assert.strictEqual(singleBatch.combos.length, 1);
    
    
    const largeBatch = wasm.linearreg_intercept_batch(close, {
        period_range: [5, 7, 10] 
    });
    
    
    assert.strictEqual(largeBatch.values.length, 10);
    assert.strictEqual(largeBatch.combos.length, 1);
    
    
    assert.throws(() => {
        wasm.linearreg_intercept_batch(new Float64Array([]), {
            period_range: [10, 10, 0]
        });
    }, /Input data slice is empty/);
});


test('Linear Regression Intercept zero-copy API', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const period = 5;
    
    
    const ptr = wasm.linearreg_intercept_alloc(data.length);
    assert(ptr !== 0, 'Failed to allocate memory');
    
    
    const memView = new Float64Array(
        wasm.__wasm.memory.buffer,
        ptr,
        data.length
    );
    
    
    memView.set(data);
    
    
    try {
        wasm.linearreg_intercept_into(ptr, ptr, data.length, period);
        
        
        const regularResult = wasm.linearreg_intercept_js(data, period);
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                continue; 
            }
            assert(Math.abs(regularResult[i] - memView[i]) < 1e-10,
                   `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${memView[i]}`);
        }
    } finally {
        
        wasm.linearreg_intercept_free(ptr, data.length);
    }
});

test('Linear Regression Intercept zero-copy with large dataset', () => {
    const size = 100000;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.sin(i * 0.01) + Math.random() * 0.1;
    }
    
    const ptr = wasm.linearreg_intercept_alloc(size);
    assert(ptr !== 0, 'Failed to allocate large buffer');
    
    try {
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        memView.set(data);
        
        wasm.linearreg_intercept_into(ptr, ptr, size, 12);
        
        
        const memView2 = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        
        
        for (let i = 0; i < 11; i++) {
            assert(isNaN(memView2[i]), `Expected NaN at warmup index ${i}`);
        }
        
        
        for (let i = 11; i < Math.min(100, size); i++) {
            assert(!isNaN(memView2[i]), `Unexpected NaN at index ${i}`);
        }
    } finally {
        wasm.linearreg_intercept_free(ptr, size);
    }
});


test('Linear Regression Intercept zero-copy error handling', () => {
    
    assert.throws(() => {
        wasm.linearreg_intercept_into(0, 0, 10, 10);
    }, /null pointer|invalid memory/i);
    
    
    const ptr = wasm.linearreg_intercept_alloc(10);
    try {
        
        assert.throws(() => {
            wasm.linearreg_intercept_into(ptr, ptr, 10, 0);
        }, /Invalid period/);
        
        
        assert.throws(() => {
            wasm.linearreg_intercept_into(ptr, ptr, 10, 20);
        }, /Invalid period/);
    } finally {
        wasm.linearreg_intercept_free(ptr, 10);
    }
});


test('Linear Regression Intercept zero-copy memory management', () => {
    
    const sizes = [100, 1000, 10000, 100000];
    
    for (const size of sizes) {
        const ptr = wasm.linearreg_intercept_alloc(size);
        assert(ptr !== 0, `Failed to allocate ${size} elements`);
        
        
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            memView[i] = i * 1.5;
        }
        
        
        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(memView[i], i * 1.5, `Memory corruption at index ${i}`);
        }
        
        
        wasm.linearreg_intercept_free(ptr, size);
    }
});

test('Linear Regression Intercept batch into API', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const periods = 3; 
    const totalSize = periods * data.length;
    
    
    const inPtr = wasm.linearreg_intercept_alloc(data.length);
    const outPtr = wasm.linearreg_intercept_alloc(totalSize);
    
    try {
        
        const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, data.length);
        inView.set(data);
        
        
        const rows = wasm.linearreg_intercept_batch_into(
            inPtr, outPtr, data.length,
            2, 6, 2  
        );
        
        
        assert.strictEqual(rows, 3, "batch_into should return number of rows");
        
        
        const outView = new Float64Array(wasm.__wasm.memory.buffer, outPtr, totalSize);
        
        
        const expectedPeriods = [2, 4, 6];
        for (let i = 0; i < expectedPeriods.length; i++) {
            const period = expectedPeriods[i];
            const rowStart = i * data.length;
            const rowData = Array.from(outView.slice(rowStart, rowStart + data.length));
            
            const singleResult = wasm.linearreg_intercept_js(data, period);
            
            for (let j = 0; j < data.length; j++) {
                if (isNaN(singleResult[j]) && isNaN(rowData[j])) {
                    continue;
                }
                assert(Math.abs(singleResult[j] - rowData[j]) < 1e-10,
                       `Batch mismatch for period ${period} at index ${j}`);
            }
        }
    } finally {
        wasm.linearreg_intercept_free(inPtr, data.length);
        wasm.linearreg_intercept_free(outPtr, totalSize);
    }
});

test.after(() => {
    console.log('Linear Regression Intercept WASM tests completed');
});
