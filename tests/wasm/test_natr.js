/**
 * Comprehensive WASM binding tests for NATR indicator.
 * Matches the quality and coverage of ALMA tests.
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

test('NATR partial params', () => {
    
    const result1 = wasm.natr_js(testData.high, testData.low, testData.close, 14);
    assert.strictEqual(result1.length, testData.close.length);
    
    
    const result2 = wasm.natr_js(testData.high, testData.low, testData.close, 7);
    assert.strictEqual(result2.length, testData.close.length);
});

test('NATR accuracy', () => {
    const period = EXPECTED_OUTPUTS['natr']['default_params']['period'];
    const expected = EXPECTED_OUTPUTS['natr']['last_5_values'];
    
    
    const result = wasm.natr_js(testData.high, testData.low, testData.close, period);
    
    assert.strictEqual(result.length, testData.close.length, 'Output length should match input length');
    
    
    const actual = result.slice(-5);
    assertArrayClose(actual, expected, 1e-8, 'NATR last 5 values should match expected');
    
    
    for (let i = 0; i < period - 1; i++) {
        assert(isNaN(result[i]), `Value at index ${i} should be NaN during warmup`);
    }
});

test('NATR zero period', () => {
    assert.throws(
        () => wasm.natr_js([10, 20, 30], [5, 10, 15], [7, 15, 25], 0),
        /Invalid period/,
        'Should throw error for zero period'
    );
});

test('NATR period exceeds length', () => {
    assert.throws(
        () => wasm.natr_js([10, 20, 30], [5, 10, 15], [7, 15, 25], 10),
        /Invalid period/,
        'Should throw error when period exceeds data length'
    );
});

test('NATR very small dataset', () => {
    assert.throws(
        () => wasm.natr_js([42], [40], [41], 14),
        /Invalid period/,
        'Should throw error for dataset smaller than period'
    );
});

test('NATR empty input', () => {
    assert.throws(
        () => wasm.natr_js([], [], [], 14),
        /Empty data/,
        'Should throw error for empty input'
    );
});

test('NATR mismatched input lengths', () => {
    assert.throws(
        () => wasm.natr_js([10, 20, 30], [5, 10], [7, 15, 25], 2),
        /Mismatched input lengths/,
        'Should throw error for mismatched input lengths'
    );
});

test('NATR all NaN input', () => {
    const nanArray = new Array(20).fill(NaN);
    assert.throws(
        () => wasm.natr_js(nanArray, nanArray, nanArray, 14),
        /All values are NaN/,
        'Should throw error when all values are NaN'
    );
});


test('NATR NaN handling', () => {
    
    const high = testData.high.slice(0, 100);
    const low = testData.low.slice(0, 100);
    const close = testData.close.slice(0, 100);
    
    
    for (let i = 10; i < 15; i++) {
        high[i] = NaN;
        low[i] = NaN;
        close[i] = NaN;
    }
    
    const result = wasm.natr_js(high, low, close, 14);
    assert.strictEqual(result.length, 100, 'Output length should match input length');
    
    
    for (let i = 0; i < 13; i++) {
        assert(isNaN(result[i]), `Index ${i} should be NaN during warmup`);
    }
    
    
    assert(isNaN(result[13]), 'Index 13 should be NaN due to NaN values in window');
});

test('NATR reinput', () => {
    
    const result1 = wasm.natr_js(testData.high, testData.low, testData.close, 14);
    assert.strictEqual(result1.length, testData.close.length);
    
    
    const result2 = wasm.natr_js(result1, result1, result1, 14);
    assert.strictEqual(result2.length, result1.length);
    
    
    const validCount = result2.slice(28).filter(v => !isNaN(v)).length;
    assert(validCount > 0, 'Should have valid values after double warmup');
});

test('NATR fast API', () => {
    const period = 14;
    const len = testData.close.length;
    
    
    const highPtr = wasm.natr_alloc(len);
    const lowPtr = wasm.natr_alloc(len);
    const closePtr = wasm.natr_alloc(len);
    const outPtr = wasm.natr_alloc(len);
    
    try {
        
        const highMem = new Float64Array(wasm.__wasm.memory.buffer, highPtr, len);
        const lowMem = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, len);
        const closeMem = new Float64Array(wasm.__wasm.memory.buffer, closePtr, len);
        
        highMem.set(testData.high);
        lowMem.set(testData.low);
        closeMem.set(testData.close);
        
        
        wasm.natr_into(
            highPtr,
            lowPtr,
            closePtr,
            outPtr,
            len,
            period
        );
        
        
        const memory = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);
        const result = Array.from(memory);
        
        
        const expected = wasm.natr_js(testData.high, testData.low, testData.close, period);
        assertArrayClose(result, expected, 1e-10, 'Fast API should match safe API');
        
    } finally {
        
        wasm.natr_free(highPtr, len);
        wasm.natr_free(lowPtr, len);
        wasm.natr_free(closePtr, len);
        wasm.natr_free(outPtr, len);
    }
});

test('NATR batch API', () => {
    const config = {
        period_range: [10, 20, 5]  
    };
    
    const result = wasm.natr_batch(testData.high, testData.low, testData.close, config);
    
    assert(result.values, 'Batch result should have values');
    assert(result.combos, 'Batch result should have combos');
    assert(result.rows, 'Batch result should have rows');
    assert(result.cols, 'Batch result should have cols');
    
    assert.strictEqual(result.rows, 3, 'Should have 3 rows (periods)');
    assert.strictEqual(result.cols, testData.close.length, 'Columns should match input length');
    assert.strictEqual(result.combos.length, 3, 'Should have 3 parameter combinations');
    
    
    for (let i = 0; i < result.rows; i++) {
        const period = result.combos[i].period;
        const rowStart = i * result.cols;
        const rowEnd = (i + 1) * result.cols;
        const batchRow = result.values.slice(rowStart, rowEnd);
        
        const single = wasm.natr_js(testData.high, testData.low, testData.close, period);
        assertArrayClose(batchRow, single, 1e-9, `Batch row ${i} (period ${period}) should match single calculation`);
    }
});

test('NATR batch single parameter set', () => {
    const config = {
        period_range: [14, 14, 0]
    };
    
    const batch_result = wasm.natr_batch(testData.high, testData.low, testData.close, config);
    const single_result = wasm.natr_js(testData.high, testData.low, testData.close, 14);
    
    assert(batch_result.values, 'Batch result should have values');
    assert(batch_result.combos, 'Batch result should have combos');
    assert.strictEqual(batch_result.rows, 1, 'Should have 1 row');
    assert.strictEqual(batch_result.cols, testData.close.length, 'Columns should match input length');
    
    
    const batchRow = batch_result.values.slice(0, batch_result.cols);
    assertArrayClose(batchRow, single_result, 1e-9, 'Batch should match single calculation');
});

test('NATR batch metadata from result', () => {
    const config = {
        period_range: [7, 21, 7]  
    };
    
    const batch_result = wasm.natr_batch(testData.high, testData.low, testData.close, config);
    
    assert.strictEqual(batch_result.combos.length, 3, 'Should have 3 parameter combinations');
    assert.strictEqual(batch_result.combos[0].period, 7);
    assert.strictEqual(batch_result.combos[1].period, 14);
    assert.strictEqual(batch_result.combos[2].period, 21);
});

test('NATR batch edge cases', () => {
    
    const config1 = {
        period_range: [14, 14, 0]
    };
    const result1 = wasm.natr_batch(testData.high, testData.low, testData.close, config1);
    assert.strictEqual(result1.rows, 1, 'Step=0 should produce single row');
    
    
    const config2 = {
        period_range: [5, 50, 5]
    };
    const result2 = wasm.natr_batch(testData.high, testData.low, testData.close, config2);
    assert.strictEqual(result2.rows, 10, 'Should have 10 rows for range 5-50 step 5');
});

test('NATR batch with mismatched lengths', () => {
    const config = {
        period_range: [14, 14, 0]
    };
    
    const shortLow = testData.low.slice(0, -10);
    
    assert.throws(
        () => wasm.natr_batch(testData.high, shortLow, testData.close, config),
        /Mismatched input lengths/,
        'Should throw error for mismatched input lengths in batch'
    );
});

test('NATR fast API in-place (aliasing)', () => {
    const period = 14;
    const len = testData.close.length;
    
    
    const highPtr = wasm.natr_alloc(len);
    const lowPtr = wasm.natr_alloc(len);
    const closePtr = wasm.natr_alloc(len);
    
    try {
        
        const highMem = new Float64Array(wasm.__wasm.memory.buffer, highPtr, len);
        const lowMem = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, len);
        const closeMem = new Float64Array(wasm.__wasm.memory.buffer, closePtr, len);
        
        highMem.set(testData.high);
        lowMem.set(testData.low);
        closeMem.set(testData.close);
        
        
        wasm.natr_into(
            highPtr,
            lowPtr,
            closePtr,
            highPtr,  
            len,
            period
        );
        
        
        const memory = new Float64Array(wasm.__wasm.memory.buffer, highPtr, len);
        const result = Array.from(memory);
        
        
        const expected = wasm.natr_js(testData.high, testData.low, testData.close, period);
        assertArrayClose(result, expected, 1e-10, 'In-place operation should match safe API');
        
    } finally {
        
        wasm.natr_free(highPtr, len);
        wasm.natr_free(lowPtr, len);
        wasm.natr_free(closePtr, len);
    }
});

test('NATR zero-copy API', () => {
    const len = testData.close.length;
    
    
    const highPtr = wasm.natr_alloc(len);
    const lowPtr = wasm.natr_alloc(len);
    const closePtr = wasm.natr_alloc(len);
    const outPtr = wasm.natr_alloc(len);
    
    assert(highPtr !== 0, 'Should allocate high memory');
    assert(lowPtr !== 0, 'Should allocate low memory');
    assert(closePtr !== 0, 'Should allocate close memory');
    assert(outPtr !== 0, 'Should allocate output memory');
    
    try {
        
        const highMem = new Float64Array(wasm.__wasm.memory.buffer, highPtr, len);
        const lowMem = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, len);
        const closeMem = new Float64Array(wasm.__wasm.memory.buffer, closePtr, len);
        
        highMem.set(testData.high);
        lowMem.set(testData.low);
        closeMem.set(testData.close);
        
        
        wasm.natr_into(
            highPtr,
            lowPtr,
            closePtr,
            outPtr,
            len,
            14
        );
        
        
        const result = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);
        
        
        const expected = wasm.natr_js(testData.high, testData.low, testData.close, 14);
        assertArrayClose(Array.from(result), expected, 1e-10, 'Zero-copy should match regular API');
        
    } finally {
        
        wasm.natr_free(highPtr, len);
        wasm.natr_free(lowPtr, len);
        wasm.natr_free(closePtr, len);
        wasm.natr_free(outPtr, len);
    }
});

test('NATR zero-copy with large dataset', () => {
    
    const largeLen = 10000;
    const high = new Float64Array(largeLen);
    const low = new Float64Array(largeLen);
    const close = new Float64Array(largeLen);
    
    
    for (let i = 0; i < largeLen; i++) {
        const base = 100 + Math.sin(i * 0.1) * 10;
        high[i] = base + 2;
        low[i] = base - 2;
        close[i] = base;
    }
    
    
    const highPtr = wasm.natr_alloc(largeLen);
    const lowPtr = wasm.natr_alloc(largeLen);
    const closePtr = wasm.natr_alloc(largeLen);
    const outPtr = wasm.natr_alloc(largeLen);
    
    assert(highPtr !== 0, 'Should allocate high memory');
    assert(lowPtr !== 0, 'Should allocate low memory');
    assert(closePtr !== 0, 'Should allocate close memory');
    assert(outPtr !== 0, 'Should allocate output memory');
    
    try {
        
        const highMem = new Float64Array(wasm.__wasm.memory.buffer, highPtr, largeLen);
        const lowMem = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, largeLen);
        const closeMem = new Float64Array(wasm.__wasm.memory.buffer, closePtr, largeLen);
        
        highMem.set(high);
        lowMem.set(low);
        closeMem.set(close);
        
        wasm.natr_into(highPtr, lowPtr, closePtr, outPtr, largeLen, 14);
        
        const result = new Float64Array(wasm.__wasm.memory.buffer, outPtr, largeLen);
        
        
        for (let i = 0; i < 13; i++) {
            assert(isNaN(result[i]), `Index ${i} should be NaN during warmup`);
        }
        
        
        let validCount = 0;
        for (let i = 13; i < largeLen; i++) {
            if (!isNaN(result[i])) validCount++;
        }
        assert(validCount > 0, 'Should have valid values after warmup');
        
    } finally {
        wasm.natr_free(highPtr, largeLen);
        wasm.natr_free(lowPtr, largeLen);
        wasm.natr_free(closePtr, largeLen);
        wasm.natr_free(outPtr, largeLen);
    }
});

test('NATR zero-copy error handling', () => {
    
    assert.throws(() => {
        wasm.natr_into(0, 0, 0, 0, 10, 14);
    }, /Null pointer|null pointer/i);
    
    
    const ptr = wasm.natr_alloc(10);
    try {
        
        assert.throws(() => {
            wasm.natr_into(ptr, ptr, ptr, ptr, 10, 0);
        }, /Invalid period/);
        
        
        assert.throws(() => {
            wasm.natr_into(ptr, ptr, ptr, ptr, 10, 20);
        }, /Invalid period/);
    } finally {
        wasm.natr_free(ptr, 10);
    }
});

test('NATR zero-copy memory management', () => {
    
    const ptrs = [];
    const sizes = [100, 500, 1000, 5000];
    
    
    for (const size of sizes) {
        const ptr = wasm.natr_alloc(size);
        assert(ptr !== 0, `Should allocate ${size} elements`);
        ptrs.push({ ptr, size });
    }
    
    
    for (const { ptr, size } of ptrs) {
        
        const highPtr = wasm.natr_alloc(size);
        const lowPtr = wasm.natr_alloc(size);
        const closePtr = wasm.natr_alloc(size);
        
        
        const highMem = new Float64Array(wasm.__wasm.memory.buffer, highPtr, size);
        const lowMem = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, size);
        const closeMem = new Float64Array(wasm.__wasm.memory.buffer, closePtr, size);
        
        highMem.fill(100);
        lowMem.fill(90);
        closeMem.fill(95);
        
        
        wasm.natr_into(highPtr, lowPtr, closePtr, ptr, size, 14);
        
        
        const result = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        assert.strictEqual(result.length, size);
        
        
        wasm.natr_free(highPtr, size);
        wasm.natr_free(lowPtr, size);
        wasm.natr_free(closePtr, size);
        wasm.natr_free(ptr, size);
    }
});

test('NATR handles zero close price', () => {
    
    const high = [100, 110, 105, 108];
    const low = [95, 100, 98, 102];
    const close = [98, 105, 0, 106];  
    
    const result = wasm.natr_js(high, low, close, 2);
    
    
    assert(isNaN(result[2]), 'Result should be NaN when close price is zero');
});

test('NATR handles infinite values', () => {
    
    const high = [100, 110, Infinity, 108];
    const low = [95, 100, 98, 102];
    const close = [98, 105, 100, 106];
    
    const result = wasm.natr_js(high, low, close, 2);
    
    
    assert(result.some(v => !isNaN(v) && isFinite(v)), 'Should have some valid finite values');
});

test('NATR batch API with zero-copy', () => {
    const config = {
        period_range: [10, 20, 5]
    };
    
    
    const batch_result = wasm.natr_batch(testData.high, testData.low, testData.close, config);
    
    
    if (wasm.natr_batch_into) {
        const rows = 3; 
        const cols = testData.close.length;
        const totalSize = rows * cols;
        
        
        const highPtr = wasm.natr_alloc(cols);
        const lowPtr = wasm.natr_alloc(cols);
        const closePtr = wasm.natr_alloc(cols);
        const outPtr = wasm.natr_alloc(totalSize);
        
        try {
            
            const highMem = new Float64Array(wasm.__wasm.memory.buffer, highPtr, cols);
            const lowMem = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, cols);
            const closeMem = new Float64Array(wasm.__wasm.memory.buffer, closePtr, cols);
            
            highMem.set(testData.high);
            lowMem.set(testData.low);
            closeMem.set(testData.close);
            
            const resultRows = wasm.natr_batch_into(
                highPtr,
                lowPtr,
                closePtr,
                outPtr,
                cols,
                10, 20, 5
            );
            
            assert.strictEqual(resultRows, rows, 'Should return correct number of rows');
            
            
            const result = new Float64Array(wasm.__wasm.memory.buffer, outPtr, totalSize);
            assertArrayClose(
                Array.from(result),
                batch_result.values,
                1e-10,
                'Batch into should match regular batch'
            );
            
        } finally {
            wasm.natr_free(highPtr, cols);
            wasm.natr_free(lowPtr, cols);
            wasm.natr_free(closePtr, cols);
            wasm.natr_free(outPtr, totalSize);
        }
    }
});

test.after(() => {
    console.log('NATR WASM tests completed');
});
