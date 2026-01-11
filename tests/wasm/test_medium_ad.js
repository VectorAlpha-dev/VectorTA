/**
 * WASM binding tests for Medium AD indicator.
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

test('Medium AD partial params', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.medium_ad_js(close, 5);
    assert.strictEqual(result.length, close.length);
});

test('Medium AD accuracy', async () => {
    
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const hl2 = new Float64Array(high.length);
    
    for (let i = 0; i < high.length; i++) {
        hl2[i] = (high[i] + low[i]) / 2;
    }
    
    const result = wasm.medium_ad_js(hl2, 5);
    
    assert.strictEqual(result.length, hl2.length);
    
    
    const expected_last_five = [220.0, 78.5, 126.5, 48.0, 28.5];
    const last5 = result.slice(-5);
    
    assertArrayClose(
        last5,
        expected_last_five,
        0.1, 
        "Medium AD last 5 values mismatch"
    );
    
});

test('Medium AD default candles', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.medium_ad_js(close, 5);
    assert.strictEqual(result.length, close.length);
});

test('Medium AD zero period', () => {
    
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.medium_ad_js(inputData, 0);
    }, /Invalid period/);
});

test('Medium AD period exceeds length', () => {
    
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.medium_ad_js(dataSmall, 10);
    }, /Invalid period/);
});

test('Medium AD very small dataset', () => {
    
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.medium_ad_js(singlePoint, 5);
    }, /Invalid period|Not enough valid data/);
});

test('Medium AD empty input', () => {
    
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.medium_ad_js(empty, 5);
    }, /Invalid period/);
});

test('Medium AD NaN handling', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.medium_ad_js(close, 5);
    assert.strictEqual(result.length, close.length);
    
    
    for (let i = 0; i < 4; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i} during warmup, got ${result[i]}`);
    }
    
    
    if (result.length > 60) {
        for (let i = 60; i < result.length; i++) {
            if (isNaN(result[i])) {
                
                let hasNaNInWindow = false;
                for (let j = Math.max(0, i - 4); j <= i; j++) {
                    if (isNaN(close[j])) {
                        hasNaNInWindow = true;
                        break;
                    }
                }
                assert(hasNaNInWindow, `Found unexpected NaN at index ${i}`);
            }
        }
    }
});

test('Medium AD all NaN input', () => {
    
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    
    const result = wasm.medium_ad_js(allNaN, 5);
    assert.strictEqual(result.length, 100, 'Result length should match input');
    
    
    for (let i = 0; i < result.length; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i}, got ${result[i]}`);
    }
});

test('Medium AD fast/unsafe API', () => {
    
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    
    const outPtr = wasm.medium_ad_alloc(len);
    const inPtr = wasm.medium_ad_alloc(len);
    
    try {
        
        const wasmMemory = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);
        wasmMemory.set(close);
        
        
        wasm.medium_ad_into(inPtr, outPtr, len, 5);
        
        
        const result = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);
        
        
        const resultArray = [];
        for (let i = 0; i < len; i++) {
            resultArray.push(result[i]);
        }
        
        
        const safeResult = wasm.medium_ad_js(close, 5);
        assertArrayClose(resultArray, safeResult, 1e-10, "Fast API result mismatch");
        
    } finally {
        
        wasm.medium_ad_free(inPtr, len);
        wasm.medium_ad_free(outPtr, len);
    }
});

test('Medium AD fast API with aliasing', () => {
    
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    
    const ptr = wasm.medium_ad_alloc(len);
    
    try {
        
        const wasmMemory = new Float64Array(wasm.__wasm.memory.buffer, ptr, len);
        wasmMemory.set(close);
        
        
        wasm.medium_ad_into(ptr, ptr, len, 5); 
        
        
        const result = new Float64Array(wasm.__wasm.memory.buffer, ptr, len);
        
        
        const safeResult = wasm.medium_ad_js(close, 5);
        assertArrayClose(Array.from(result), safeResult, 1e-10, "In-place operation result mismatch");
        
    } finally {
        
        wasm.medium_ad_free(ptr, len);
    }
});

test('Medium AD batch single parameter set', () => {
    
    const close = new Float64Array(testData.close);
    
    const config = {
        period_range: [5, 5, 0] 
    };
    
    const batchResult = wasm.medium_ad_batch(close, config);
    
    
    assert(batchResult.values, "Batch result should have values");
    assert(batchResult.combos, "Batch result should have combos");
    assert(batchResult.rows, "Batch result should have rows");
    assert(batchResult.cols, "Batch result should have cols");
    
    assert.strictEqual(batchResult.rows, 1, "Should have 1 parameter combination");
    assert.strictEqual(batchResult.cols, close.length, "Columns should match data length");
    assert.strictEqual(batchResult.combos.length, 1, "Should have 1 combo");
    assert.strictEqual(batchResult.combos[0].period, 5, "Period should be 5");
    
    
    const singleResult = wasm.medium_ad_js(close, 5);
    const firstRow = batchResult.values.slice(0, batchResult.cols);
    
    assertArrayClose(firstRow, singleResult, 1e-10, "Batch vs single mismatch");
});

test('Medium AD batch multiple periods', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100)); 
    
    const config = {
        period_range: [5, 15, 5] 
    };
    
    const batchResult = wasm.medium_ad_batch(close, config);
    
    
    assert.strictEqual(batchResult.rows, 3, "Should have 3 parameter combinations");
    assert.strictEqual(batchResult.cols, 100, "Columns should be 100");
    assert.strictEqual(batchResult.values.length, 3 * 100, "Values array size mismatch");
    assert.strictEqual(batchResult.combos.length, 3, "Should have 3 combos");
    
    
    const expectedPeriods = [5, 10, 15];
    for (let i = 0; i < batchResult.combos.length; i++) {
        assert.strictEqual(batchResult.combos[i].period, expectedPeriods[i], 
            `Combo ${i} has wrong period`);
    }
    
    
    for (let i = 0; i < expectedPeriods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.values.slice(rowStart, rowEnd);
        
        const singleResult = wasm.medium_ad_js(close, expectedPeriods[i]);
        assertArrayClose(
            rowData,
            singleResult,
            1e-10,
            `Period ${expectedPeriods[i]} batch mismatch`
        );
    }
});

test('Medium AD batch edge cases', () => {
    
    const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    
    
    const singleConfig = { period_range: [3, 3, 1] };
    const singleBatch = wasm.medium_ad_batch(close, singleConfig);
    assert.strictEqual(singleBatch.rows, 1, "Single sweep should have 1 row");
    assert.strictEqual(singleBatch.combos[0].period, 3, "Single sweep period should be 3");
    
    
    const largeConfig = { period_range: [3, 5, 10] };
    const largeBatch = wasm.medium_ad_batch(close, largeConfig);
    assert.strictEqual(largeBatch.rows, 1, "Large step should only have period=3");
    assert.strictEqual(largeBatch.combos[0].period, 3, "Large step period should be 3");
    
    
    assert.throws(() => {
        wasm.medium_ad_batch(new Float64Array([]), { period_range: [5, 5, 0] });
    }, /Invalid period|All values are NaN/);
});

test('Medium AD memory management', () => {
    
    const len = 1000;
    
    
    const ptr = wasm.medium_ad_alloc(len);
    assert(ptr !== 0, "Should allocate non-null pointer");
    
    
    const wasmMemory = new Float64Array(wasm.__wasm.memory.buffer, ptr, len);
    for (let i = 0; i < len; i++) {
        wasmMemory[i] = i * 1.5;
    }
    
    
    for (let i = 0; i < len; i++) {
        assert.strictEqual(wasmMemory[i], i * 1.5, `Memory corruption at index ${i}`);
    }
    
    
    wasm.medium_ad_free(ptr, len);
    
    
    wasm.medium_ad_free(0, len); 
});

test('Medium AD error handling', () => {
    
    
    
    assert.throws(() => {
        wasm.medium_ad_into(0, 100, 10, 5);
    }, /Null pointer/);
    
    assert.throws(() => {
        wasm.medium_ad_into(100, 0, 10, 5);
    }, /Null pointer/);
    
    
    const ptr = wasm.medium_ad_alloc(10);
    try {
        assert.throws(() => {
            wasm.medium_ad_into(ptr, ptr, 10, 0);
        }, /Invalid period/);
        
        assert.throws(() => {
            wasm.medium_ad_into(ptr, ptr, 10, 20);
        }, /Invalid period/);
    } finally {
        wasm.medium_ad_free(ptr, 10);
    }
});

test('Medium AD warmup period', () => {
    
    const close = new Float64Array(testData.close.slice(0, 20)); 
    const period = 5;
    
    const result = wasm.medium_ad_js(close, period);
    
    
    for (let i = 0; i < period - 1; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i} during warmup`);
    }
    
    
    assert(!isNaN(result[period - 1]), `Expected valid value at index ${period - 1}`);
});