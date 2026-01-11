/**
 * WASM binding tests for MEAN_AD indicator.
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

test('MEAN_AD accuracy', () => {
    
    const hl2 = testData.high.map((h, i) => (h + testData.low[i]) / 2);
    const result = wasm.mean_ad_js(hl2, 5);
    
    const expected = EXPECTED_OUTPUTS.meanAd.last5Values;
    const last5 = result.slice(-5);
    
    for (let i = 0; i < 5; i++) {
        assertClose(last5[i], expected[i], 1e-1, `mean_ad last value ${i} mismatch`);
    }
});

test('MEAN_AD default params', () => {
    const result = wasm.mean_ad_js(testData.close, 5);
    assert.strictEqual(result.length, testData.close.length, 'Output length mismatch');
    
    
    
    for (let i = 0; i < 8; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i}`);
    }
    
    
    for (let i = 240; i < result.length; i++) {
        assert(!isNaN(result[i]), `Unexpected NaN at index ${i}`);
    }
});

test('MEAN_AD error handling', () => {
    
    assert.throws(() => {
        wasm.mean_ad_js(new Float64Array(0), 5);
    }, 'Empty data should throw error');
    
    
    assert.throws(() => {
        wasm.mean_ad_js(new Float64Array([1, 2, 3]), 0);
    }, 'Zero period should throw error');
    
    
    assert.throws(() => {
        wasm.mean_ad_js(new Float64Array([1, 2, 3]), 10);
    }, 'Period > data length should throw error');
});

test('MEAN_AD fast API', () => {
    const data = new Float64Array(testData.close);
    const period = 5;
    
    
    const inPtr = wasm.mean_ad_alloc(data.length);
    const outPtr = wasm.mean_ad_alloc(data.length);
    
    try {
        
        const memory = new Float64Array(wasm.__wasm.memory.buffer, inPtr, data.length);
        memory.set(data);
        
        
        const expected = wasm.mean_ad_js(data, period);
        
        
        wasm.mean_ad_into(inPtr, outPtr, data.length, period);
        
        
        const result = new Float64Array(wasm.__wasm.memory.buffer, outPtr, data.length);
        
        assertArrayClose(result, expected, 1e-10, 'Fast API mismatch');
        
        
        
        const memoryForReset = new Float64Array(wasm.__wasm.memory.buffer, inPtr, data.length);
        memoryForReset.set(data); 
        wasm.mean_ad_into(inPtr, inPtr, data.length, period);
        
        
        const inPlaceResult = new Float64Array(wasm.__wasm.memory.buffer, inPtr, data.length);
        assertArrayClose(inPlaceResult, expected, 1e-10, 'In-place operation mismatch');
    } finally {
        wasm.mean_ad_free(inPtr, data.length);
        wasm.mean_ad_free(outPtr, data.length);
    }
});

test('MEAN_AD batch API', () => {
    const config = {
        period_range: [5, 50, 5] 
    };
    
    const result = wasm.mean_ad_batch(testData.close.slice(0, 100), config);
    
    assert(result.values, 'Batch result should have values');
    assert(result.periods, 'Batch result should have periods');
    assert.strictEqual(result.rows, 10, 'Should have 10 period combinations');
    assert.strictEqual(result.cols, 100, 'Should have 100 data points');
    assert.strictEqual(result.values.length, 1000, 'Values array size mismatch');
    assert.strictEqual(result.periods.length, 10, 'Periods array size mismatch');
    
    
    const expectedPeriods = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50];
    assert.deepStrictEqual(result.periods, expectedPeriods, 'Period values mismatch');
});

test('MEAN_AD NaN handling', () => {
    const dataWithNaN = new Float64Array([NaN, NaN, 1, 2, 3, 4, 5, NaN, 6, 7]);
    const result = wasm.mean_ad_js(dataWithNaN, 3);
    
    
    for (let i = 0; i < 6; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i}`);
    }
    
    
    assert(!isNaN(result[6]), 'Expected valid value at index 6');
    
});

test('MEAN_AD all NaN input', () => {
    const allNaN = new Float64Array([NaN, NaN, NaN, NaN, NaN]);
    assert.throws(() => {
        wasm.mean_ad_js(allNaN, 3);
    }, 'All NaN values should throw error');
});

test('MEAN_AD partial params', () => {
    
    const result = wasm.mean_ad_js(testData.close, 5);
    assert(result.length === testData.close.length, 'Should handle default parameters');
});

test('MEAN_AD zero-copy with large dataset', () => {
    const size = 100000;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.sin(i / 100) * 100 + 50;
    }
    
    const inPtr = wasm.mean_ad_alloc(size);
    const outPtr = wasm.mean_ad_alloc(size);
    
    try {
        
        const memory = new Float64Array(wasm.__wasm.memory.buffer, inPtr, size);
        memory.set(data);
        
        const start = performance.now();
        wasm.mean_ad_into(inPtr, outPtr, size, 50);
        const fastTime = performance.now() - start;
        
        const start2 = performance.now();
        const safeResult = wasm.mean_ad_js(data, 50);
        const safeTime = performance.now() - start2;
        
        
        console.log(`Fast API: ${fastTime.toFixed(2)}ms, Safe API: ${safeTime.toFixed(2)}ms`);
        
        
        const result = new Float64Array(wasm.__wasm.memory.buffer, outPtr, size);
        assertArrayClose(result, safeResult, 1e-10, 'Large dataset mismatch');
    } finally {
        wasm.mean_ad_free(inPtr, size);
        wasm.mean_ad_free(outPtr, size);
    }
});

test('MEAN_AD zero-copy memory management', () => {
    
    for (let i = 0; i < 10; i++) {
        const ptr = wasm.mean_ad_alloc(1000);
        assert(ptr !== 0, 'Allocation should return non-zero pointer');
        wasm.mean_ad_free(ptr, 1000);
    }
});

test('MEAN_AD batch edge cases', () => {
    
    const config = {
        period_range: [2, 3, 1] 
    };
    
    const minData = new Float64Array([1, 2, 3, 4, 5]);
    const result = wasm.mean_ad_batch(minData, config);
    
    assert.strictEqual(result.rows, 2, 'Should have 2 rows');
    assert.strictEqual(result.cols, 5, 'Should have 5 columns');
    assert.deepStrictEqual(result.periods, [2, 3], 'Should have periods 2 and 3');
});

test('MEAN_AD batch metadata', () => {
    const config = {
        period_range: [10, 50, 10] 
    };
    
    const result = wasm.mean_ad_batch(testData.close.slice(0, 200), config);
    
    
    assert(result.values, 'Should have values array');
    assert(result.periods, 'Should have periods array');
    assert(result.rows === 5, 'Should have 5 rows');
    assert(result.cols === 200, 'Should have 200 columns');
    
    
    assert.deepStrictEqual(result.periods, [10, 20, 30, 40, 50], 'Periods mismatch');
});

test('MEAN_AD batch fast API', () => {
    const data = new Float64Array(testData.close.slice(0, 100));
    const periods = { start: 5, end: 15, step: 5 }; 
    
    const inPtr = wasm.mean_ad_alloc(data.length);
    const outSize = 3 * data.length;
    const outPtr = wasm.mean_ad_alloc(outSize);
    
    try {
        
        const memory = new Float64Array(wasm.__wasm.memory.buffer, inPtr, data.length);
        memory.set(data);
        
        const rows = wasm.mean_ad_batch_into(
            inPtr, outPtr, data.length,
            periods.start, periods.end, periods.step
        );
        
        assert.strictEqual(rows, 3, 'Should return 3 rows');
        
        
        const config = {
            period_range: [periods.start, periods.end, periods.step]
        };
        const safeResult = wasm.mean_ad_batch(data, config);
        
        const fastResult = new Float64Array(wasm.__wasm.memory.buffer, outPtr, outSize);
        assertArrayClose(fastResult, safeResult.values, 1e-10, 'Batch fast API mismatch');
    } finally {
        wasm.mean_ad_free(inPtr, data.length);
        wasm.mean_ad_free(outPtr, outSize);
    }
});

test('MEAN_AD batch single parameter set', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const batchResult = wasm.mean_ad_batch(close, {
        period_range: [5, 5, 0]
    });
    
    
    const singleResult = wasm.mean_ad_js(close, 5);
    
    assert.strictEqual(batchResult.values.length, singleResult.length);
    assertArrayClose(batchResult.values, singleResult, 1e-10, "Batch vs single mismatch");
});

test('MEAN_AD batch multiple periods', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100)); 
    
    
    const batchResult = wasm.mean_ad_batch(close, {
        period_range: [5, 15, 5]
    });
    
    
    assert.strictEqual(batchResult.values.length, 3 * 100);
    assert.strictEqual(batchResult.rows, 3);
    assert.strictEqual(batchResult.cols, 100);
    assert.deepStrictEqual(batchResult.periods, [5, 10, 15]);
    
    
    const periods = [5, 10, 15];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.values.slice(rowStart, rowEnd);
        
        const singleResult = wasm.mean_ad_js(close, periods[i]);
        assertArrayClose(
            rowData, 
            singleResult, 
            1e-10, 
            `Period ${periods[i]} mismatch`
        );
    }
});

test('MEAN_AD batch full parameter sweep', () => {
    
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const batchResult = wasm.mean_ad_batch(close, {
        period_range: [5, 10, 5]  
    });
    
    
    assert.strictEqual(batchResult.periods.length, 2);
    assert.strictEqual(batchResult.rows, 2);
    assert.strictEqual(batchResult.cols, 50);
    assert.strictEqual(batchResult.values.length, 2 * 50);
    
    
    for (let row = 0; row < batchResult.rows; row++) {
        const period = batchResult.periods[row];
        const rowStart = row * 50;
        const rowData = batchResult.values.slice(rowStart, rowStart + 50);
        
        
        const warmup = 2 * period - 2;
        
        
        for (let i = 0; i < warmup; i++) {
            assert(isNaN(rowData[i]), `Expected NaN at warmup index ${i} for period ${period}`);
        }
        
        
        for (let i = warmup; i < Math.min(warmup + 10, 50); i++) {
            assert(!isNaN(rowData[i]), `Unexpected NaN at index ${i} for period ${period}`);
        }
    }
});

test('MEAN_AD SIMD128 consistency', () => {
    
    
    const testCases = [
        { size: 10, period: 3 },
        { size: 100, period: 5 },
        { size: 1000, period: 10 },
        { size: 10000, period: 20 }
    ];
    
    for (const testCase of testCases) {
        const data = new Float64Array(testCase.size);
        for (let i = 0; i < testCase.size; i++) {
            data[i] = Math.sin(i * 0.1) + Math.cos(i * 0.05) + 100;
        }
        
        const result = wasm.mean_ad_js(data, testCase.period);
        
        
        assert.strictEqual(result.length, data.length);
        
        
        const warmup = 2 * testCase.period - 2;
        for (let i = 0; i < warmup; i++) {
            assert(isNaN(result[i]), `Expected NaN at warmup index ${i} for size=${testCase.size}`);
        }
        
        
        let sumAfterWarmup = 0;
        let countAfterWarmup = 0;
        for (let i = warmup; i < result.length; i++) {
            assert(!isNaN(result[i]), `Unexpected NaN at index ${i} for size=${testCase.size}`);
            sumAfterWarmup += result[i];
            countAfterWarmup++;
        }
        
        
        const avgAfterWarmup = sumAfterWarmup / countAfterWarmup;
        assert(avgAfterWarmup >= 0, `Average value ${avgAfterWarmup} should be non-negative`);
        assert(avgAfterWarmup < 1000, `Average value ${avgAfterWarmup} seems unreasonable`);
    }
});

test('MEAN_AD zero-copy error handling', () => {
    
    assert.throws(() => {
        wasm.mean_ad_into(0, 0, 10, 5);
    }, /null pointer|invalid memory/i);
    
    
    const ptr = wasm.mean_ad_alloc(10);
    try {
        
        assert.throws(() => {
            wasm.mean_ad_into(ptr, ptr, 10, 0);
        }, /Invalid period/);
        
        
        assert.throws(() => {
            wasm.mean_ad_into(ptr, ptr, 10, 20);
        }, /Invalid period/);
    } finally {
        wasm.mean_ad_free(ptr, 10);
    }
});

test.after(() => {
    console.log('MEAN_AD WASM tests completed');
});
