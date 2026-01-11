/**
 * WASM binding tests for UI (Ulcer Index) indicator.
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
    assertNoNaN
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

test('UI partial params', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.ui_js(close, 14, 100.0);
    assert.strictEqual(result.length, close.length);
});

test('UI accuracy', async () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.ui_js(close, 14, 100.0);
    
    assert.strictEqual(result.length, close.length);
    
    
    const expectedLastFive = [
        3.514342861283708,
        3.304986039846459,
        3.2011859814326304,
        3.1308860017483373,
        2.909612553474519,
    ];
    
    
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expectedLastFive,
        1e-6,
        "UI last 5 values mismatch"
    );
    
    
    const warmupPeriod = 14 * 2 - 2; 
    for (let i = 0; i < warmupPeriod; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i} during warmup period`);
    }
    
    
    await compareWithRust('ui', result, 'close', { period: 14, scalar: 100.0 });
});

test('UI default candles', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.ui_js(close, 14, 100.0);
    assert.strictEqual(result.length, close.length);
});

test('UI zero period', () => {
    
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.ui_js(inputData, 0, 100.0);
    }, /Invalid period/);
});

test('UI period exceeds length', () => {
    
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.ui_js(dataSmall, 10, 100.0);
    }, /Invalid period/);
});

test('UI very small dataset', () => {
    
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.ui_js(singlePoint, 14, 100.0);
    }, /Invalid period/);
});

test('UI empty input', () => {
    
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.ui_js(empty, 14, 100.0);
    }, /Empty|empty/);
});

test('UI all NaN input', () => {
    
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.ui_js(allNaN, 14, 100.0);
    }, /All values are NaN/);
});

test('UI different scalars', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const result1 = wasm.ui_js(close, 14, 50.0);
    assert.strictEqual(result1.length, close.length);
    
    
    const result2 = wasm.ui_js(close, 14, 200.0);
    assert.strictEqual(result2.length, close.length);
    
    
    
    const resultDefault = wasm.ui_js(close, 14, 100.0);
    for (let i = 14 * 2 - 2; i < close.length; i++) { 
        if (!isNaN(resultDefault[i]) && !isNaN(result2[i])) {
            const ratio = result2[i] / resultDefault[i];
            assert(Math.abs(ratio - 2.0) < 0.01, `Scalar scaling incorrect at index ${i}`);
        }
    }
});

test('UI batch single parameter set', () => {
    
    const close = new Float64Array(testData.close);
    
    const batchResult = wasm.ui_batch(close, {
        period_range: [14, 14, 0],
        scalar_range: [100.0, 100.0, 0]
    });
    
    
    const singleResult = wasm.ui_js(close, 14, 100.0);
    
    assert.strictEqual(batchResult.values.length, singleResult.length);
    assertArrayClose(batchResult.values, singleResult, 1e-10, "Batch vs single mismatch");
});

test('UI batch multiple periods', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100)); 
    
    
    const batchResult = wasm.ui_batch(close, {
        period_range: [10, 14, 2],
        scalar_range: [100.0, 100.0, 0]
    });
    
    
    assert.strictEqual(batchResult.values.length, 3 * 100);
    assert.strictEqual(batchResult.rows, 3);
    assert.strictEqual(batchResult.cols, 100);
    
    
    const periods = [10, 12, 14];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.values.slice(rowStart, rowEnd);
        
        const singleResult = wasm.ui_js(close, periods[i], 100.0);
        assertArrayClose(
            rowData, 
            singleResult, 
            1e-10, 
            `Period ${periods[i]} mismatch`
        );
    }
});

test('UI batch multiple scalars', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100)); 
    
    
    const batchResult = wasm.ui_batch(close, {
        period_range: [14, 14, 0],
        scalar_range: [50.0, 150.0, 50.0]
    });
    
    
    assert.strictEqual(batchResult.values.length, 3 * 100);
    assert.strictEqual(batchResult.rows, 3);
    assert.strictEqual(batchResult.cols, 100);
    assert.strictEqual(batchResult.combos.length, 3);
    
    
    assert.strictEqual(batchResult.combos[0].scalar, 50.0);
    assert.strictEqual(batchResult.combos[1].scalar, 100.0);
    assert.strictEqual(batchResult.combos[2].scalar, 150.0);
});

test('UI batch metadata from result', () => {
    
    
    const close = new Float64Array(30);
    close.fill(100);
    
    const result = wasm.ui_batch(close, {
        period_range: [10, 14, 2],     
        scalar_range: [50.0, 100.0, 50.0] 
    });
    
    
    assert.strictEqual(result.combos.length, 6);
    
    
    assert.strictEqual(result.combos[0].period, 10);
    assert.strictEqual(result.combos[0].scalar, 50.0);
    
    
    assert.strictEqual(result.combos[5].period, 14);
    assert.strictEqual(result.combos[5].scalar, 100.0);
});


test('UI zero-copy API (in-place)', () => {
    
    const data = new Float64Array([100, 105, 95, 98, 102, 99, 101, 104, 103, 107]);
    const period = 5;
    const scalar = 100.0;
    
    
    const ptr = wasm.ui_alloc(data.length);
    assert(ptr !== 0, 'Failed to allocate memory');
    
    
    const memView = new Float64Array(
        wasm.__wasm.memory.buffer,
        ptr,
        data.length
    );
    
    
    memView.set(data);
    
    
    try {
        wasm.ui_into(ptr, ptr, data.length, period, scalar);
        
        
        const regularResult = wasm.ui_js(data, period, scalar);
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                continue; 
            }
            assert(Math.abs(regularResult[i] - memView[i]) < 1e-10,
                   `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${memView[i]}`);
        }
    } finally {
        
        wasm.ui_free(ptr, data.length);
    }
});

test('UI zero-copy with large dataset', () => {
    const size = 100000;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = 100 + Math.sin(i * 0.01) * 10 + Math.random() * 2;
    }
    
    const ptr = wasm.ui_alloc(size);
    assert(ptr !== 0, 'Failed to allocate large buffer');
    
    try {
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        memView.set(data);
        
        wasm.ui_into(ptr, ptr, size, 14, 100.0);
        
        
        const memView2 = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        
        
        const warmup = 14 * 2 - 2;
        for (let i = 0; i < warmup; i++) {
            assert(isNaN(memView2[i]), `Expected NaN at warmup index ${i}`);
        }
        
        
        for (let i = warmup; i < Math.min(warmup + 10, size); i++) {
            assert(!isNaN(memView2[i]), `Unexpected NaN at index ${i}`);
        }
    } finally {
        wasm.ui_free(ptr, size);
    }
});


test('UI zero-copy error handling', () => {
    
    assert.throws(() => {
        wasm.ui_into(0, 0, 10, 14, 100.0);
    }, /null pointer|Null pointer/i);
    
    
    const ptr = wasm.ui_alloc(10);
    try {
        
        assert.throws(() => {
            wasm.ui_into(ptr, ptr, 10, 0, 100.0);
        }, /Invalid period/);
        
        
        assert.throws(() => {
            wasm.ui_into(ptr, ptr, 10, 15, 100.0);
        }, /Invalid period/);
    } finally {
        wasm.ui_free(ptr, 10);
    }
});


test('UI zero-copy memory management', () => {
    
    const sizes = [100, 1000, 10000, 100000];
    
    for (const size of sizes) {
        const ptr = wasm.ui_alloc(size);
        assert(ptr !== 0, `Failed to allocate ${size} elements`);
        
        
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            memView[i] = i * 1.5;
        }
        
        
        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(memView[i], i * 1.5, `Memory corruption at index ${i}`);
        }
        
        
        wasm.ui_free(ptr, size);
    }
});


test('UI batch fast API', () => {
    const size = 100;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = 100 + Math.sin(i * 0.1) * 5;
    }
    
    
    const inPtr = wasm.ui_alloc(size);
    assert(inPtr !== 0, 'Failed to allocate input buffer');
    
    
    const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, size);
    inView.set(data);
    
    
    
    const outSize = 6 * size;
    const outPtr = wasm.ui_alloc(outSize);
    assert(outPtr !== 0, 'Failed to allocate output buffer');
    
    try {
        const numCombos = wasm.ui_batch_into(
            inPtr, outPtr, size,
            10, 14, 2,      
            50.0, 100.0, 50.0  
        );
        
        assert.strictEqual(numCombos, 6, 'Expected 6 combinations');
        
        
        const outView = new Float64Array(wasm.__wasm.memory.buffer, outPtr, outSize);
        
        
        let hasNonNaN = false;
        for (let i = 0; i < outSize; i++) {
            if (!isNaN(outView[i])) {
                hasNonNaN = true;
                break;
            }
        }
        assert(hasNonNaN, 'Output should contain some non-NaN values');
    } finally {
        wasm.ui_free(inPtr, size);
        wasm.ui_free(outPtr, outSize);
    }
});

test.after(() => {
    console.log('UI WASM tests completed');
});