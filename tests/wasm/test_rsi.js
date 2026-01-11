/**
 * WASM binding tests for RSI indicator.
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
        
        if (wasm && wasm.default) {
            wasm = wasm.default;
        }
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }
    
    testData = loadTestData();
});

test('RSI partial params', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.rsi_js(close, 14);
    assert.strictEqual(result.length, close.length);
});

test('RSI accuracy', async () => {
    
    const close = new Float64Array(testData.close);
    const expectedLastFive = [43.42, 42.68, 41.62, 42.86, 39.01];
    
    const result = wasm.rsi_js(close, 14);
    
    assert.strictEqual(result.length, close.length);
    
    
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expectedLastFive,
        1e-2,  
        "RSI last 5 values mismatch"
    );
    
    
    await compareWithRust('rsi', result, 'close', { period: 14 });
});

test('RSI default candles', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.rsi_js(close, 14);
    assert.strictEqual(result.length, close.length);
});

test('RSI zero period', () => {
    
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.rsi_js(inputData, 0);
    }, /Invalid period/);
});

test('RSI period exceeds length', () => {
    
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.rsi_js(dataSmall, 10);
    }, /Invalid period/);
});

test('RSI very small dataset', () => {
    
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.rsi_js(singlePoint, 14);
    }, /Invalid period|Not enough valid data/);
});

test('RSI empty input', () => {
    
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.rsi_js(empty, 14);
    }, /Input data slice is empty|Invalid period/);
});

test('RSI reinput', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const firstResult = wasm.rsi_js(close, 14);
    assert.strictEqual(firstResult.length, close.length);
    
    
    const secondResult = wasm.rsi_js(firstResult, 5);
    assert.strictEqual(secondResult.length, firstResult.length);
    
    
    if (secondResult.length > 240) {
        for (let i = 240; i < secondResult.length; i++) {
            assert(!isNaN(secondResult[i]), `Found unexpected NaN at index ${i} after warmup period`);
        }
    }
});

test('RSI nan handling', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.rsi_js(close, 14);
    assert.strictEqual(result.length, close.length);
    
    
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i} after warmup period`);
        }
    }
});

test('RSI all nan input', () => {
    
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.rsi_js(allNaN, 14);
    }, /All values are NaN/);
});

test('RSI batch single parameter set', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const batchResult = wasm.rsi_batch(close, {
        period_range: [14, 14, 0]
    });
    
    
    const singleResult = wasm.rsi_js(close, 14);
    
    assert.strictEqual(batchResult.values.length, singleResult.length);
    assertArrayClose(batchResult.values, singleResult, 1e-10, "Batch vs single mismatch");
});

test('RSI batch multiple periods', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100)); 
    
    
    const batchResult = wasm.rsi_batch(close, {
        period_range: [10, 18, 4]  
    });
    
    
    assert.strictEqual(batchResult.values.length, 3 * 100);
    assert.strictEqual(batchResult.rows, 3);
    assert.strictEqual(batchResult.cols, 100);
    
    
    const periods = [10, 14, 18];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.values.slice(rowStart, rowEnd);
        
        const singleResult = wasm.rsi_js(close, periods[i]);
        assertArrayClose(
            rowData, 
            singleResult, 
            1e-10, 
            `Period ${periods[i]} mismatch`
        );
    }
});

test('RSI batch metadata from result', () => {
    
    const close = new Float64Array(20); 
    close.fill(100);
    
    const result = wasm.rsi_batch(close, {
        period_range: [5, 15, 5]  
    });
    
    
    assert.strictEqual(result.combos.length, 3);
    
    
    assert.strictEqual(result.combos[0].period, 5);
    assert.strictEqual(result.combos[1].period, 10);
    assert.strictEqual(result.combos[2].period, 15);
});

test('RSI batch full parameter sweep', () => {
    
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const batchResult = wasm.rsi_batch(close, {
        period_range: [10, 20, 10]  
    });
    
    
    assert.strictEqual(batchResult.combos.length, 2);
    assert.strictEqual(batchResult.rows, 2);
    assert.strictEqual(batchResult.cols, 50);
    assert.strictEqual(batchResult.values.length, 2 * 50);
    
    
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

test('RSI batch edge cases', () => {
    
    const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    
    
    const singleBatch = wasm.rsi_batch(close, {
        period_range: [5, 5, 1]
    });
    
    assert.strictEqual(singleBatch.values.length, 10);
    assert.strictEqual(singleBatch.combos.length, 1);
    
    
    const largeBatch = wasm.rsi_batch(close, {
        period_range: [5, 7, 10]  
    });
    
    
    assert.strictEqual(largeBatch.values.length, 10);
    assert.strictEqual(largeBatch.combos.length, 1);
    
    
    assert.throws(() => {
        wasm.rsi_batch(new Float64Array([]), {
            period_range: [14, 14, 0]
        });
    }, /Invalid period|empty/i);
});


test('RSI zero-copy API', () => {
    const data = new Float64Array([
        45.15, 46.26, 46.50, 46.23, 46.08, 46.03, 46.83, 47.69,
        47.54, 49.25, 49.23, 48.20, 47.57, 47.61, 48.08, 47.21,
        46.76, 46.68, 46.21, 47.47, 47.98, 47.13, 46.58, 46.03,
        46.54, 46.79, 47.05, 47.49, 47.27, 47.96, 47.24
    ]);
    const period = 14;
    
    
    const ptr = wasm.rsi_alloc(data.length);
    assert(ptr !== 0, 'Failed to allocate memory');
    
    
    const memView = new Float64Array(
        wasm.__wasm.memory.buffer,
        ptr,
        data.length
    );
    
    
    memView.set(data);
    
    
    try {
        wasm.rsi_into(ptr, ptr, data.length, period);
        
        
        const regularResult = wasm.rsi_js(data, period);
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                continue; 
            }
            assert(Math.abs(regularResult[i] - memView[i]) < 1e-10,
                   `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${memView[i]}`);
        }
    } finally {
        
        wasm.rsi_free(ptr, data.length);
    }
});

test('RSI zero-copy with large dataset', () => {
    const size = 100000;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = 50 + Math.sin(i * 0.01) * 20 + Math.random() * 5;
    }
    
    const ptr = wasm.rsi_alloc(size);
    assert(ptr !== 0, 'Failed to allocate large buffer');
    
    try {
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        memView.set(data);
        
        wasm.rsi_into(ptr, ptr, size, 14);
        
        
        const memView2 = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        
        
        for (let i = 0; i < 14; i++) {
            assert(isNaN(memView2[i]), `Expected NaN at warmup index ${i}`);
        }
        
        
        for (let i = 14; i < Math.min(100, size); i++) {
            assert(!isNaN(memView2[i]), `Unexpected NaN at index ${i}`);
            
            assert(memView2[i] >= 0 && memView2[i] <= 100, 
                   `RSI value out of range at ${i}: ${memView2[i]}`);
        }
    } finally {
        wasm.rsi_free(ptr, size);
    }
});

test('RSI zero-copy error handling', () => {
    
    assert.throws(() => {
        wasm.rsi_into(0, 0, 10, 14);
    }, /null pointer|invalid memory/i);
    
    
    const ptr = wasm.rsi_alloc(10);
    try {
        
        assert.throws(() => {
            wasm.rsi_into(ptr, ptr, 10, 0);
        }, /Invalid period/);
        
        
        assert.throws(() => {
            wasm.rsi_into(ptr, ptr, 10, 20);
        }, /Invalid period/);
    } finally {
        wasm.rsi_free(ptr, 10);
    }
});


test('RSI zero-copy memory management', () => {
    
    const sizes = [100, 1000, 10000, 100000];
    
    for (const size of sizes) {
        const ptr = wasm.rsi_alloc(size);
        assert(ptr !== 0, `Failed to allocate ${size} elements`);
        
        
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            memView[i] = i * 1.5;
        }
        
        
        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(memView[i], i * 1.5, `Memory corruption at index ${i}`);
        }
        
        
        wasm.rsi_free(ptr, size);
    }
});

test.after(() => {
    console.log('RSI WASM tests completed');
});
