/**
 * WASM binding tests for WILDERS indicator.
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

test('Wilders partial params', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.wilders_js(close, 5);
    assert.strictEqual(result.length, close.length);
});

test('Wilders accuracy', async () => {
    
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.wilders;
    
    const result = wasm.wilders_js(
        close,
        expected.defaultParams.period
    );
    
    assert.strictEqual(result.length, close.length);
    
    
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-8,
        "Wilders last 5 values mismatch"
    );
    
    
    await compareWithRust('wilders', result, 'close', expected.defaultParams);
});

test('Wilders default candles', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.wilders_js(close, 5);
    assert.strictEqual(result.length, close.length);
});

test('Wilders zero period', () => {
    
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.wilders_js(inputData, 0);
    }, /Invalid period/);
});

test('Wilders period exceeds length', () => {
    
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.wilders_js(dataSmall, 10);
    }, /Invalid period/);
});

test('Wilders very small dataset', () => {
    
    const singlePoint = new Float64Array([42.0]);
    
    
    const result = wasm.wilders_js(singlePoint, 1);
    assert.strictEqual(result.length, 1);
});

test('Wilders empty input', () => {
    
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.wilders_js(empty, 5);
    }, /Input data slice is empty/);
});

test('Wilders NaN handling', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.wilders_js(close, 5);
    assert.strictEqual(result.length, close.length);
    
    
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
    
    
    const warmupEnd = 4;
    assertAllNaN(result.slice(0, warmupEnd), "Expected NaN in warmup period");
    
    
    assert(!isNaN(result[warmupEnd]), `Expected valid value at index ${warmupEnd} (first output)`);
});

test('Wilders all NaN input', () => {
    
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.wilders_js(allNaN, 5);
    }, /All values are NaN/);
});

test('Wilders batch single parameter set', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const batchResult = wasm.wilders_batch_js(
        close,
        5, 5, 0      
    );
    
    
    const singleResult = wasm.wilders_js(close, 5);
    
    assert.strictEqual(batchResult.length, singleResult.length);
    assertArrayClose(batchResult, singleResult, 1e-10, "Batch vs single mismatch");
});

test('Wilders batch multiple periods', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100)); 
    
    
    const batchResult = wasm.wilders_batch_js(
        close,
        5, 10, 1      
    );
    
    
    assert.strictEqual(batchResult.length, 6 * 100);
    
    
    const periods = [5, 6, 7, 8, 9, 10];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.slice(rowStart, rowEnd);
        
        const singleResult = wasm.wilders_js(close, periods[i]);
        assertArrayClose(
            rowData, 
            singleResult, 
            1e-10, 
            `Period ${periods[i]} mismatch`
        );
    }
});

test('Wilders batch metadata', () => {
    
    const metadata = wasm.wilders_batch_metadata_js(
        5, 10, 1      
    );
    
    
    assert.strictEqual(metadata.length, 6);
    
    
    for (let i = 0; i < 6; i++) {
        assert.strictEqual(metadata[i], 5 + i);
    }
});

test('Wilders batch full parameter sweep', () => {
    
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const batchResult = wasm.wilders_batch_js(
        close,
        5, 7, 2      
    );
    
    const metadata = wasm.wilders_batch_metadata_js(
        5, 7, 2
    );
    
    
    assert.strictEqual(metadata.length, 2);
    assert.strictEqual(batchResult.length, 2 * 50);
    
    
    for (let combo = 0; combo < 2; combo++) {
        const period = metadata[combo];
        const rowStart = combo * 50;
        const rowData = batchResult.slice(rowStart, rowStart + 50);
        const warmupEnd = period - 1;
        
        
        for (let i = 0; i < warmupEnd; i++) {
            assert(isNaN(rowData[i]), `Expected NaN at warmup index ${i} for period ${period}`);
        }
        
        
        assert(!isNaN(rowData[warmupEnd]), `Expected valid value at index ${warmupEnd} for period ${period}`);
        
        
        for (let i = warmupEnd; i < 50; i++) {
            assert(!isNaN(rowData[i]), `Unexpected NaN at index ${i} for period ${period}`);
        }
    }
});

test('Wilders batch unified API', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100));
    
    
    if (typeof wasm.wilders_batch === 'function') {
        const result = wasm.wilders_batch(close, {
            period_range: [5, 7, 1]  
        });
        
        
        assert(result.values, 'Should have values array');
        assert(result.combos, 'Should have combos array');
        assert(typeof result.rows === 'number', 'Should have rows count');
        assert(typeof result.cols === 'number', 'Should have cols count');
        
        
        assert.strictEqual(result.rows, 3);
        assert.strictEqual(result.cols, 100);
        assert.strictEqual(result.combos.length, 3);
        assert.strictEqual(result.values.length, 300);
        
        
        for (let i = 0; i < 3; i++) {
            assert.strictEqual(result.combos[i].period, 5 + i);
        }
        
        
        const oldResult = wasm.wilders_js(close, 5);
        const firstRow = result.values.slice(0, 100);
        for (let i = 0; i < oldResult.length; i++) {
            if (isNaN(oldResult[i]) && isNaN(firstRow[i])) {
                continue; 
            }
            assert(Math.abs(oldResult[i] - firstRow[i]) < 1e-10,
                   `Value mismatch at index ${i}`);
        }
    } else {
        
        const batchResult = wasm.wilders_batch_js(close, 5, 7, 1);
        assert.strictEqual(batchResult.length, 3 * 100);
    }
});

test('Wilders batch edge cases', () => {
    
    const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    
    
    const singleBatch = wasm.wilders_batch_js(
        close,
        5, 5, 1
    );
    
    assert.strictEqual(singleBatch.length, 10);
    
    
    const largeBatch = wasm.wilders_batch_js(
        close,
        5, 7, 10 
    );
    
    
    assert.strictEqual(largeBatch.length, 10);
    
    
    assert.throws(() => {
        wasm.wilders_batch_js(
            new Float64Array([]),
            5, 5, 0
        );
    }, /All values are NaN/);
});


test('Wilders zero-copy API', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const period = 5;
    
    
    const ptr = wasm.wilders_alloc(data.length);
    assert(ptr !== 0, 'Failed to allocate memory');
    
    
    const memory = wasm.__wasm.memory.buffer;
    const memView = new Float64Array(
        memory,
        ptr,
        data.length
    );
    
    
    memView.set(data);
    
    
    try {
        wasm.wilders_into(ptr, ptr, data.length, period);
        
        
        const regularResult = wasm.wilders_js(data, period);
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                continue; 
            }
            assert(Math.abs(regularResult[i] - memView[i]) < 1e-10,
                   `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${memView[i]}`);
        }
    } finally {
        
        wasm.wilders_free(ptr, data.length);
    }
});

test('Wilders zero-copy with large dataset', () => {
    const size = 10000;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.sin(i * 0.01) + Math.random() * 0.1;
    }
    
    const ptr = wasm.wilders_alloc(size);
    assert(ptr !== 0, 'Failed to allocate large buffer');
    
    try {
        const memory = wasm.__wasm.memory.buffer;
        const memView = new Float64Array(memory, ptr, size);
        memView.set(data);
        
        wasm.wilders_into(ptr, ptr, size, 5);
        
        
        const memory2 = wasm.__wasm.memory.buffer;
        const memView2 = new Float64Array(memory2, ptr, size);
        
        
        for (let i = 0; i < 4; i++) {
            assert(isNaN(memView2[i]), `Expected NaN at warmup index ${i}`);
        }
        
        
        for (let i = 4; i < Math.min(100, size); i++) {
            assert(!isNaN(memView2[i]), `Unexpected NaN at index ${i}`);
        }
    } finally {
        wasm.wilders_free(ptr, size);
    }
});

test('Wilders zero-copy error handling', () => {
    
    assert.throws(() => {
        wasm.wilders_into(0, 0, 10, 5);
    }, /null pointer|invalid memory/i);
    
    
    const ptr = wasm.wilders_alloc(10);
    try {
        
        assert.throws(() => {
            wasm.wilders_into(ptr, ptr, 10, 0);
        }, /Invalid period/);
        
        
        assert.throws(() => {
            wasm.wilders_into(ptr, ptr, 10, 20);
        }, /Invalid period/);
    } finally {
        wasm.wilders_free(ptr, 10);
    }
});

test('Wilders memory management', () => {
    
    const sizes = [100, 1000, 10000];
    
    for (const size of sizes) {
        const ptr = wasm.wilders_alloc(size);
        assert(ptr !== 0, `Failed to allocate ${size} elements`);
        
        
        const memory = wasm.__wasm.memory.buffer;
        const memView = new Float64Array(memory, ptr, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            memView[i] = i * 1.5;
        }
        
        
        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(memView[i], i * 1.5, `Memory corruption at index ${i}`);
        }
        
        
        wasm.wilders_free(ptr, size);
    }
});

test('Wilders NaN in initial window', () => {
    
    const data = new Float64Array([1.0, 2.0, NaN, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    
    assert.throws(() => {
        wasm.wilders_js(data, 5);
    }, /Not enough valid data/);
});

test.after(() => {
    console.log('Wilders WASM tests completed');
});
