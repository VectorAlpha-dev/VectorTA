/**
 * WASM binding tests for DX indicator.
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

test('DX accuracy', () => {
    
    const high = testData.high;
    const low = testData.low;
    const close = testData.close;
    const expected = EXPECTED_OUTPUTS.dx;
    
    const result = wasm.dx_js(
        high, low, close,
        expected.defaultParams.period
    );
    
    assert.strictEqual(result.length, high.length, 'Output length should match input length');
    
    
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-4,  
        "DX last 5 values mismatch"
    );
    
    
    for (let i = 0; i < result.length; i++) {
        if (!isNaN(result[i])) {
            assert(result[i] >= 0 && result[i] <= 100, 
                `DX value at index ${i} should be between 0 and 100, got ${result[i]}`);
        }
    }
});

test('DX basic functionality', () => {
    const high = testData.high;
    const low = testData.low;
    const close = testData.close;
    
    
    const result = wasm.dx_js(high, low, close, 14);
    assert.strictEqual(result.length, high.length, 'Output length should match input length');
    
    
    const warmup = 14 - 1;
    for (let i = 0; i < warmup; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i} during warmup period`);
    }
    
    
    let hasValidValues = false;
    for (let i = warmup + 10; i < result.length; i++) {
        if (!isNaN(result[i])) {
            hasValidValues = true;
            break;
        }
    }
    assert(hasValidValues, 'Should have valid values after warmup period');
});

test('DX warmup period validation', () => {
    const high = testData.high.slice(0, 100);
    const low = testData.low.slice(0, 100);
    const close = testData.close.slice(0, 100);
    
    const period = 14;
    const result = wasm.dx_js(high, low, close, period);
    
    
    
    
    const expectedWarmup = period - 1;
    
    
    let lastNaNIndex = -1;
    for (let i = 0; i < result.length; i++) {
        if (isNaN(result[i])) {
            lastNaNIndex = i;
        } else {
            break;
        }
    }
    
    assert(lastNaNIndex >= expectedWarmup - 1, 
        `Expected at least ${expectedWarmup} NaN values, but last NaN was at index ${lastNaNIndex}`);
    
    
    if (result.length > expectedWarmup + 10) {
        for (let i = expectedWarmup + 10; i < Math.min(expectedWarmup + 20, result.length); i++) {
            assert(!isNaN(result[i]), `Unexpected NaN at index ${i} after warmup period`);
        }
    }
});

test('DX value range validation', () => {
    const high = testData.high;
    const low = testData.low;
    const close = testData.close;
    
    
    const periods = [5, 10, 14, 20, 50];
    
    for (const period of periods) {
        if (period >= high.length) continue;
        
        const result = wasm.dx_js(high, low, close, period);
        
        
        let minVal = Infinity;
        let maxVal = -Infinity;
        let validCount = 0;
        
        for (let i = 0; i < result.length; i++) {
            if (!isNaN(result[i])) {
                validCount++;
                minVal = Math.min(minVal, result[i]);
                maxVal = Math.max(maxVal, result[i]);
                
                assert(result[i] >= -1e-9, 
                    `DX value at index ${i} should be >= 0, got ${result[i]} for period ${period}`);
                assert(result[i] <= 100.0 + 1e-9, 
                    `DX value at index ${i} should be <= 100, got ${result[i]} for period ${period}`);
            }
        }
        
        
        assert(validCount > 0, `No valid values found for period ${period}`);
        
        console.log(`Period ${period}: min=${minVal.toFixed(2)}, max=${maxVal.toFixed(2)}, valid=${validCount}`);
    }
});

test('DX fast API', () => {
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low); 
    const close = new Float64Array(testData.close);
    const len = high.length;
    
    
    const highPtr = wasm.dx_alloc(len);
    const lowPtr = wasm.dx_alloc(len);
    const closePtr = wasm.dx_alloc(len);
    const outPtr = wasm.dx_alloc(len);
    
    try {
        
        const highMem = new Float64Array(wasm.__wasm.memory.buffer, highPtr, len);
        const lowMem = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, len);
        const closeMem = new Float64Array(wasm.__wasm.memory.buffer, closePtr, len);
        highMem.set(high);
        lowMem.set(low);
        closeMem.set(close);
        
        
        wasm.dx_into(
            highPtr,
            lowPtr,
            closePtr,
            outPtr,
            len,
            14
        );
        
        
        const result = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);
        const resultCopy = Float64Array.from(result);
        
        
        const safeResult = wasm.dx_js(high, low, close, 14);
        assertArrayClose(resultCopy, safeResult, 1e-10, 'Fast API should match safe API');
        
    } finally {
        wasm.dx_free(highPtr, len);
        wasm.dx_free(lowPtr, len);
        wasm.dx_free(closePtr, len);
        wasm.dx_free(outPtr, len);
    }
});

test('DX fast API with aliasing', () => {
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    const len = high.length;
    
    
    const highPtr = wasm.dx_alloc(len);
    const lowPtr = wasm.dx_alloc(len);
    const closePtr = wasm.dx_alloc(len);
    
    try {
        
        const highMem = new Float64Array(wasm.__wasm.memory.buffer, highPtr, len);
        const lowMem = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, len);
        const closeMem = new Float64Array(wasm.__wasm.memory.buffer, closePtr, len);
        highMem.set(high);
        lowMem.set(low);
        closeMem.set(close);
        
        
        wasm.dx_into(
            highPtr,
            lowPtr,
            closePtr,
            highPtr,  
            len,
            14
        );
        
        
        const result = new Float64Array(wasm.__wasm.memory.buffer, highPtr, len);
        const resultCopy = Float64Array.from(result);
        
        
        const expected = wasm.dx_js(high, low, close, 14);
        assertArrayClose(resultCopy, expected, 1e-10, 'Should handle aliasing correctly');
        
    } finally {
        wasm.dx_free(highPtr, len);
        wasm.dx_free(lowPtr, len);
        wasm.dx_free(closePtr, len);
    }
});

test('DX batch API with unified interface', () => {
    const high = testData.high.slice(0, 100);  
    const low = testData.low.slice(0, 100);
    const close = testData.close.slice(0, 100);
    
    const config = {
        period_range: [10, 20, 5]  
    };
    
    const result = wasm.dx_batch(high, low, close, config);
    
    
    assert(result.values, 'Result should have values array');
    assert(result.combos, 'Result should have combos array');
    assert.strictEqual(result.rows, 3, 'Should have 3 parameter combinations');
    assert.strictEqual(result.cols, high.length, 'Cols should match input length');
    assert.strictEqual(result.values.length, 3 * high.length, 'Values should be flattened matrix');
    
    
    const periods = [10, 15, 20];
    for (let i = 0; i < periods.length; i++) {
        const batchRow = result.values.slice(i * high.length, (i + 1) * high.length);
        const singleResult = wasm.dx_js(high, low, close, periods[i]);
        assertArrayClose(batchRow, singleResult, 1e-10, 
            `Batch result for period ${periods[i]} should match single computation`);
    }
});

test('DX batch with single parameter', () => {
    const high = testData.high;
    const low = testData.low;
    const close = testData.close;
    
    const config = {
        period_range: [14, 14, 0]  
    };
    
    const result = wasm.dx_batch(high, low, close, config);
    
    
    assert.strictEqual(result.combos.length, 1);
    assert.strictEqual(result.rows, 1);
    assert.strictEqual(result.cols, high.length);
    
    
    const batchRow = result.values;
    const expected = EXPECTED_OUTPUTS.dx.last5Values;
    
    
    const last5 = batchRow.slice(-5);
    assertArrayClose(
        last5,
        expected,
        1e-4,
        "DX batch default row mismatch"
    );
});

test('DX batch metadata', () => {
    const high = new Float64Array(50);
    const low = new Float64Array(50);
    const close = new Float64Array(50);
    
    
    for (let i = 0; i < 50; i++) {
        high[i] = 100 + Math.sin(i * 0.1) * 10;
        low[i] = 90 + Math.sin(i * 0.1) * 10;
        close[i] = 95 + Math.sin(i * 0.1) * 10;
    }
    
    const config = {
        period_range: [10, 20, 10]  
    };
    
    const result = wasm.dx_batch(high, low, close, config);
    
    
    assert.strictEqual(result.combos.length, 2);
    
    
    assert.strictEqual(result.combos[0].period, 10);
    assert.strictEqual(result.combos[1].period, 20);
    
    
    const row1 = result.values.slice(0, 50);
    const row2 = result.values.slice(50, 100);
    
    
    let nanCount1 = 0;
    for (let i = 0; i < 15; i++) {
        if (isNaN(row1[i])) nanCount1++;
    }
    
    
    let nanCount2 = 0;
    for (let i = 0; i < 25; i++) {
        if (isNaN(row2[i])) nanCount2++;
    }
    
    assert(nanCount1 >= 9, `Expected at least 9 NaN values for period 10, got ${nanCount1}`);
    assert(nanCount2 >= 19, `Expected at least 19 NaN values for period 20, got ${nanCount2}`);
});

test('DX error handling', () => {
    const high = testData.high;
    const low = testData.low;
    const close = testData.close;
    
    
    assert.throws(() => {
        wasm.dx_js([], [], [], 14);
    }, 'Should throw on empty input');
    
    
    const result = wasm.dx_js(high.slice(0, 10), low.slice(0, 5), close.slice(0, 10), 2);
    assert.strictEqual(result.length, 5, 'Should use minimum input length');
    
    
    assert.throws(() => {
        wasm.dx_js(high.slice(0, 10), low.slice(0, 10), close.slice(0, 10), 20);
    }, 'Should throw when period exceeds data length');
    
    
    assert.throws(() => {
        wasm.dx_js(high, low, close, 0);
    }, 'Should throw on zero period');
});

test('DX all NaN input', () => {
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.dx_js(allNaN, allNaN, allNaN, 14);
    }, /All high, low, and close values are NaN/, 'Should throw when all values are NaN');
});

test('DX batch edge cases', () => {
    const high = new Float64Array([100, 102, 101, 103, 104, 105, 103, 102, 104, 106]);
    const low = new Float64Array([98, 99, 98, 100, 101, 102, 100, 99, 101, 103]);
    const close = new Float64Array([99, 101, 100, 102, 103, 104, 102, 101, 103, 105]);
    
    
    const singleBatch = wasm.dx_batch(high, low, close, {
        period_range: [5, 5, 1]
    });
    
    assert.strictEqual(singleBatch.values.length, 10);
    assert.strictEqual(singleBatch.combos.length, 1);
    
    
    const largeBatch = wasm.dx_batch(high, low, close, {
        period_range: [5, 7, 10]  
    });
    
    assert.strictEqual(largeBatch.values.length, 10);
    assert.strictEqual(largeBatch.combos.length, 1);
    assert.strictEqual(largeBatch.combos[0].period, 5);
    
    
    assert.throws(() => {
        wasm.dx_batch([], [], [], {
            period_range: [14, 14, 0]
        });
    }, 'Should throw on empty data');
});

test('DX memory management', () => {
    
    const sizes = [100, 1000, 10000];
    
    for (const size of sizes) {
        const ptr = wasm.dx_alloc(size);
        assert(ptr !== 0, `Failed to allocate ${size} elements`);
        
        
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            memView[i] = i * 2.5;
        }
        
        
        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(memView[i], i * 2.5, `Memory corruption at index ${i}`);
        }
        
        
        wasm.dx_free(ptr, size);
    }
});

test.after(() => {
    console.log('DX WASM tests completed');
});