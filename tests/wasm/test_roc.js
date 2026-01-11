/**
 * WASM binding tests for ROC indicator.
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

test('ROC partial params', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.roc_js(close, 9);
    assert.strictEqual(result.length, close.length);
});

test('ROC accuracy', async () => {
    
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.roc;
    
    const result = wasm.roc_js(
        close,
        expected.defaultParams.period
    );
    
    assert.strictEqual(result.length, close.length);
    
    
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-7,
        "ROC last 5 values mismatch"
    );
    
    
    await compareWithRust('roc', result, 'close', expected.defaultParams);
});

test('ROC default candles', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.roc_js(close, 9);
    assert.strictEqual(result.length, close.length);
});

test('ROC zero period', () => {
    
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.roc_js(inputData, 0);
    }, /Invalid period/);
});

test('ROC period exceeds length', () => {
    
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.roc_js(dataSmall, 10);
    }, /Invalid period/);
});

test('ROC very small dataset', () => {
    
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.roc_js(singlePoint, 9);
    }, /Invalid period|Not enough valid data/);
});

test('ROC empty input', () => {
    
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.roc_js(empty, 9);
    }, /Empty data provided|Input data slice is empty/);
});

test('ROC reinput', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const firstResult = wasm.roc_js(close, 14);
    assert.strictEqual(firstResult.length, close.length);
    
    
    const secondResult = wasm.roc_js(firstResult, 14);
    assert.strictEqual(secondResult.length, firstResult.length);
    
    
    for (let i = 28; i < secondResult.length; i++) {
        assert(!isNaN(secondResult[i]), 
               `Expected no NaN after index 28, found NaN at ${i}`);
    }
});

test('ROC NaN handling', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.roc_js(close, 9);
    assert.strictEqual(result.length, close.length);
    
    
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
    
    
    assertAllNaN(result.slice(0, 9), "Expected NaN in warmup period");
});

test('ROC all NaN input', () => {
    
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.roc_js(allNaN, 10);
    }, /All values are NaN/);
});

test('ROC batch single parameter set', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const batchResult = wasm.roc_batch(close, {
        period_range: [10, 10, 0]
    });
    
    
    const singleResult = wasm.roc_js(close, 10);
    
    assert.strictEqual(batchResult.values.length, singleResult.length);
    assertArrayClose(batchResult.values, singleResult, 1e-10, "Batch vs single mismatch");
});

test('ROC batch multiple periods', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100)); 
    
    
    const batchResult = wasm.roc_batch(close, {
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
        
        const singleResult = wasm.roc_js(close, periods[i]);
        assertArrayClose(
            rowData, 
            singleResult, 
            1e-10, 
            `Period ${periods[i]} mismatch`
        );
    }
});

test('ROC batch metadata from result', () => {
    
    const close = new Float64Array(20); 
    close.fill(100);
    
    const result = wasm.roc_batch(close, {
        period_range: [10, 14, 2]  
    });
    
    
    assert.strictEqual(result.combos.length, 3);
    
    
    assert.strictEqual(result.combos[0].period, 10);
    
    
    assert.strictEqual(result.combos[2].period, 14);
});

test('ROC batch full parameter sweep', () => {
    
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const batchResult = wasm.roc_batch(close, {
        period_range: [10, 14, 2]  
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

test('ROC batch edge cases', () => {
    
    const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    
    
    const singleBatch = wasm.roc_batch(close, {
        period_range: [5, 5, 1]
    });
    
    assert.strictEqual(singleBatch.values.length, 15);
    assert.strictEqual(singleBatch.combos.length, 1);
    
    
    const largeBatch = wasm.roc_batch(close, {
        period_range: [5, 7, 10] 
    });
    
    
    assert.strictEqual(largeBatch.values.length, 15);
    assert.strictEqual(largeBatch.combos.length, 1);
    
    
    assert.throws(() => {
        wasm.roc_batch(new Float64Array([]), {
            period_range: [10, 10, 0]
        });
    }, /All values are NaN/);
});


test('ROC zero-copy API', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const period = 5;
    
    
    const ptr = wasm.roc_alloc(data.length);
    assert(ptr !== 0, 'Failed to allocate memory');
    
    
    const memView = new Float64Array(
        wasm.__wasm.memory.buffer,
        ptr,
        data.length
    );
    
    
    memView.set(data);
    
    
    try {
        wasm.roc_into(ptr, ptr, data.length, period);
        
        
        const regularResult = wasm.roc_js(data, period);
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                continue; 
            }
            assert(Math.abs(regularResult[i] - memView[i]) < 1e-10,
                   `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${memView[i]}`);
        }
    } finally {
        
        wasm.roc_free(ptr, data.length);
    }
});

test('ROC zero-copy with large dataset', () => {
    const size = 100000;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.sin(i * 0.01) + Math.random() * 0.1 + 100; 
    }
    
    const ptr = wasm.roc_alloc(size);
    assert(ptr !== 0, 'Failed to allocate large buffer');
    
    try {
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        memView.set(data);
        
        wasm.roc_into(ptr, ptr, size, 10);
        
        
        const memView2 = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        
        
        for (let i = 0; i < 10; i++) {
            assert(isNaN(memView2[i]), `Expected NaN at warmup index ${i}`);
        }
        
        
        for (let i = 10; i < Math.min(100, size); i++) {
            assert(!isNaN(memView2[i]), `Unexpected NaN at index ${i}`);
        }
    } finally {
        wasm.roc_free(ptr, size);
    }
});


test('ROC zero-copy error handling', () => {
    
    assert.throws(() => {
        wasm.roc_into(0, 0, 10, 10);
    }, /null pointer|Null pointer/i);
    
    
    const ptr = wasm.roc_alloc(10);
    try {
        
        assert.throws(() => {
            wasm.roc_into(ptr, ptr, 10, 0);
        }, /Invalid period/);
        
        
        assert.throws(() => {
            wasm.roc_into(ptr, ptr, 10, 20);
        }, /Invalid period/);
    } finally {
        wasm.roc_free(ptr, 10);
    }
});


test('ROC zero-copy memory management', () => {
    
    const sizes = [100, 1000, 10000, 100000];
    
    for (const size of sizes) {
        const ptr = wasm.roc_alloc(size);
        assert(ptr !== 0, `Failed to allocate ${size} elements`);
        
        
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            memView[i] = i * 1.5;
        }
        
        
        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(memView[i], i * 1.5, `Memory corruption at index ${i}`);
        }
        
        
        wasm.roc_free(ptr, size);
    }
});

test.after(() => {
    console.log('ROC WASM tests completed');
});