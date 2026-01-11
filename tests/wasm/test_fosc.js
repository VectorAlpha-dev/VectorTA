/**
 * WASM binding tests for FOSC indicator.
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
    
    
    
    const pkgPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const esmImportPath = process.platform === 'win32'
        ? 'file:///' + pkgPath.replace(/\\/g, '/')
        : pkgPath;

    try {
        wasm = await import(esmImportPath);
    } catch (error) {
        
        
        
        const msg = String(error && error.message || error);
        const needsFallback = msg.includes("Cannot find package 'env'") || msg.includes('vector_ta_bg.wasm');
        if (!needsFallback) {
            console.error('Failed to load WASM pkg module:', error);
            throw error;
        }
        
        const { createRequire } = await import('node:module');
        const require = createRequire(import.meta.url);
        wasm = require(pkgPath);
    }

    testData = loadTestData();
});

test('FOSC accuracy - mirrors check_fosc_expected_values_reference', () => {
    const period = EXPECTED_OUTPUTS.fosc.defaultParams.period;
    
    
    const result = wasm.fosc_js(testData.close, period);
    assert.equal(result.length, testData.close.length);
    
    
    const lastN = 5;
    const startIndex = result.length - lastN;
    const actualLast5 = result.slice(startIndex);
    assertArrayClose(actualLast5, EXPECTED_OUTPUTS.fosc.last5Values, 1e-7, 'FOSC last 5 values');
    
    
    let firstValid = 0;
    for (let i = 0; i < testData.close.length; i++) {
        if (!isNaN(testData.close[i])) {
            firstValid = i;
            break;
        }
    }
    const expectedWarmup = firstValid + period - 1;
    
    
    for (let i = 0; i < expectedWarmup && i < result.length; i++) {
        assert(isNaN(result[i]), `Expected NaN at warmup index ${i}`);
    }
    
    
    if (expectedWarmup < result.length) {
        assert(!isNaN(result[expectedWarmup]), `Expected first valid value at index ${expectedWarmup}`);
    }
});

test('FOSC fast API', () => {
    const period = EXPECTED_OUTPUTS.fosc.defaultParams.period;
    const len = testData.close.length;
    
    
    const outPtr = wasm.fosc_alloc(len);
    
    try {
        
        const inPtr = wasm.fosc_alloc(len);
        const inArray = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);
        inArray.set(testData.close);
        
        
        wasm.fosc_into(inPtr, outPtr, len, period);
        
        
        const result = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);
        const resultCopy = Array.from(result);
        
        
        const safeResult = wasm.fosc_js(testData.close, period);
        assertArrayClose(resultCopy, safeResult, 1e-10, 'Fast API matches safe API');
        
        
        wasm.fosc_into(inPtr, inPtr, len, period);
        const inPlaceResult = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);
        assertArrayClose(Array.from(inPlaceResult), safeResult, 1e-10, 'In-place operation');
        
        
        wasm.fosc_free(inPtr, len);
    } finally {
        
        wasm.fosc_free(outPtr, len);
    }
});

test('FOSC error handling - mirrors Rust error tests', () => {
    
    assert.throws(() => {
        wasm.fosc_js(testData.close, 0);
    }, /Invalid period/, 'Should fail with zero period');
    
    
    const smallData = [10.0, 20.0, 30.0];
    assert.throws(() => {
        wasm.fosc_js(smallData, 10);
    }, /Invalid period/, 'Should fail when period exceeds length');
    
    
    assert.throws(() => {
        wasm.fosc_js([], 5);
    }, /Empty input data/, 'Should fail with empty data');
    
    
    const nanData = new Array(10).fill(NaN);
    assert.throws(() => {
        wasm.fosc_js(nanData, 5);
    }, /All values are NaN/, 'Should fail with all NaN values');
});

test('FOSC batch API - mirrors check_batch_default_row', () => {
    const config = {
        period_range: [3, 10, 1]  
    };
    
    const result = wasm.fosc_batch(testData.close, config);
    
    
    assert(result.values, 'Should have values array');
    assert(result.combos, 'Should have combos array');
    assert(result.rows, 'Should have rows count');
    assert(result.cols, 'Should have cols count');
    
    
    const expectedRows = 8; 
    assert.equal(result.rows, expectedRows);
    assert.equal(result.cols, testData.close.length);
    assert.equal(result.values.length, result.rows * result.cols);
    assert.equal(result.combos.length, expectedRows);
    
    
    const defaultRow = result.combos.findIndex(c => c.period === 5);
    assert(defaultRow >= 0, 'Should find period=5 row');
    
    const rowStart = defaultRow * result.cols;
    const rowValues = result.values.slice(rowStart, rowStart + result.cols);
    const lastN = 5;
    const actualLast5 = rowValues.slice(-lastN);
    assertArrayClose(actualLast5, EXPECTED_OUTPUTS.fosc.last5Values, 1e-7, 'Batch default row matches');
    
    
    const singleResult = wasm.fosc_js(testData.close, 5);
    assertArrayClose(rowValues, singleResult, 1e-10, 'Batch row should match single calculation');
});

test('FOSC batch fast API', () => {
    const periodStart = 3, periodEnd = 7, periodStep = 2; 
    const expectedRows = 3;
    const len = testData.close.length;
    const totalSize = expectedRows * len;
    
    
    const inPtr = wasm.fosc_alloc(len);
    const outPtr = wasm.fosc_alloc(totalSize);
    
    try {
        
        const inArray = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);
        inArray.set(testData.close);
        
        
        const rows = wasm.fosc_batch_into(
            inPtr, outPtr, len,
            periodStart, periodEnd, periodStep
        );
        
        assert.equal(rows, expectedRows, 'Correct number of rows');
        
        
        const results = new Float64Array(wasm.__wasm.memory.buffer, outPtr, totalSize);
        
        
        if (periodStart <= 5 && periodEnd >= 5 && (5 - periodStart) % periodStep === 0) {
            const rowIndex = Math.floor((5 - periodStart) / periodStep);
            const rowStart = rowIndex * len;
            const rowValues = Array.from(results.slice(rowStart, rowStart + len));
            const actualLast5 = rowValues.slice(-5);
            assertArrayClose(actualLast5, EXPECTED_OUTPUTS.fosc.last5Values, 1e-7, 
                'Batch fast API period=5 matches expected');
        }
    } finally {
        wasm.fosc_free(inPtr, len);
        wasm.fosc_free(outPtr, totalSize);
    }
});

test('FOSC NaN handling - mirrors check_fosc_with_nan_data', () => {
    const period = 5;
    
    
    const dataWithNaN = [NaN, NaN, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    const result = wasm.fosc_js(dataWithNaN, period);
    
    assert.equal(result.length, dataWithNaN.length, 'Result length matches input');
    
    
    const firstValidIdx = 2;
    const minWarmup = firstValidIdx + period - 1; 
    
    
    
    for (let i = 0; i < minWarmup; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i} during minimum warmup`);
    }
    
    
    
    let hasValid = false;
    for (let i = 0; i < result.length; i++) {
        if (!isNaN(result[i])) {
            hasValid = true;
            break;
        }
    }
    assert(hasValid, 'Should have at least some valid values in the output');
});

test('FOSC zero-copy API', () => {
    const period = 5;
    const len = 100;
    const data = new Float64Array(len);
    for (let i = 0; i < len; i++) {
        data[i] = Math.sin(i * 0.1) + Math.random() * 0.1;
    }
    
    
    const inPtr = wasm.fosc_alloc(len);
    const outPtr = wasm.fosc_alloc(len);
    
    try {
        
        const inArray = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);
        inArray.set(data);
        
        
        wasm.fosc_into(inPtr, outPtr, len, period);
        
        
        const outArray = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);
        const result = Array.from(outArray);
        
        
        const safeResult = wasm.fosc_js(data, period);
        assertArrayClose(result, safeResult, 1e-10, 'Zero-copy matches safe API');
        
        
        for (let i = 0; i < period - 1; i++) {
            assert(isNaN(result[i]), `Expected NaN at warmup index ${i}`);
        }
        
        
        for (let i = period - 1; i < Math.min(period + 10, len); i++) {
            assert(!isNaN(result[i]), `Expected valid value at index ${i}`);
        }
    } finally {
        wasm.fosc_free(inPtr, len);
        wasm.fosc_free(outPtr, len);
    }
});

test('FOSC memory management', () => {
    
    const sizes = [10, 100, 1000, 10000];
    
    for (const size of sizes) {
        const ptr = wasm.fosc_alloc(size);
        assert(ptr !== 0, `Failed to allocate ${size} elements`);
        
        
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            memView[i] = i * 2.5;
        }
        
        
        for (let i = 0; i < Math.min(10, size); i++) {
            assert.equal(memView[i], i * 2.5, `Memory corruption at index ${i}`);
        }
        
        
        wasm.fosc_free(ptr, size);
    }
});

test('FOSC edge cases', () => {
    
    const data = [1.0, 2.0, 3.0, 4.0, 5.0];
    const result = wasm.fosc_js(data, 2);
    assert.equal(result.length, data.length, 'Result length matches input');
    assert(isNaN(result[0]), 'First value should be NaN');
    assert(!isNaN(result[1]), 'Second value should be valid for period=2');
    
    
    const constantData = new Array(20).fill(100.0);
    const constantResult = wasm.fosc_js(constantData, 5);
    
    
    for (let i = 6; i < constantResult.length; i++) {
        if (!isNaN(constantResult[i])) {
            assert(Math.abs(constantResult[i]) < 1e-6, 
                `FOSC should be ~0 for constant data at index ${i}, got ${constantResult[i]}`);
        }
    }
    
    
    const linearData = [];
    for (let i = 1; i <= 20; i++) {
        linearData.push(i);
    }
    const linearResult = wasm.fosc_js(linearData, 5);
    
    
    const validValues = linearResult.slice(5).filter(v => !isNaN(v));
    if (validValues.length > 2) {
        const mean = validValues.reduce((a, b) => a + b, 0) / validValues.length;
        const variance = validValues.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / validValues.length;
        const stdDev = Math.sqrt(variance);
        assert(stdDev < 1.0, `FOSC should be consistent for linear data, std=${stdDev}`);
    }
});

test('FOSC batch metadata - mirrors check_batch_sweep', () => {
    const config = {
        period_range: [5, 15, 5]  
    };
    
    const testSlice = testData.close.slice(0, 50);
    const result = wasm.fosc_batch(testSlice, config);
    
    
    assert.equal(result.rows, 3, 'Should have 3 rows');
    assert.equal(result.cols, 50, 'Should have 50 columns');
    assert.equal(result.combos.length, 3, 'Should have 3 combinations');
    
    
    assert.equal(result.combos[0].period, 5, 'First combo period should be 5');
    assert.equal(result.combos[1].period, 10, 'Second combo period should be 10');
    assert.equal(result.combos[2].period, 15, 'Third combo period should be 15');
    
    
    let firstValid = 0;
    for (let i = 0; i < testSlice.length; i++) {
        if (!isNaN(testSlice[i])) {
            firstValid = i;
            break;
        }
    }
    
    
    for (let row = 0; row < 3; row++) {
        const period = result.combos[row].period;
        const rowStart = row * result.cols;
        const expectedWarmup = firstValid + period - 1;
        
        
        for (let i = 0; i < Math.min(expectedWarmup, result.cols); i++) {
            assert(isNaN(result.values[rowStart + i]), 
                `Row ${row} (period=${period}): Expected NaN at warmup index ${i}`);
        }
        
        
        if (expectedWarmup < result.cols) {
            assert(!isNaN(result.values[rowStart + expectedWarmup]), 
                `Row ${row} (period=${period}): Expected first valid value at index ${expectedWarmup}`);
        }
        
        
        const singleResult = wasm.fosc_js(testSlice, period);
        const rowValues = result.values.slice(rowStart, rowStart + result.cols);
        assertArrayClose(rowValues, singleResult, 1e-10, 
            `Batch row ${row} (period=${period}) should match single calculation`);
    }
});





test('FOSC SIMD128 consistency', () => {
    
    
    const testCases = [
        { size: 10, period: 3 },
        { size: 100, period: 5 },
        { size: 1000, period: 10 },
        { size: 5000, period: 20 }
    ];
    
    for (const testCase of testCases) {
        const data = new Float64Array(testCase.size);
        for (let i = 0; i < testCase.size; i++) {
            data[i] = Math.sin(i * 0.1) * 100 + Math.cos(i * 0.05) * 50;
        }
        
        const result = wasm.fosc_js(data, testCase.period);
        
        
        assert.equal(result.length, data.length, 
            `Size ${testCase.size}: Result length mismatch`);
        
        
        for (let i = 0; i < testCase.period - 1; i++) {
            assert(isNaN(result[i]), 
                `Size ${testCase.size}: Expected NaN at warmup index ${i}`);
        }
        
        
        let validCount = 0;
        for (let i = testCase.period - 1; i < result.length; i++) {
            if (!isNaN(result[i])) {
                validCount++;
                
                
            }
        }
        
        assert(validCount > 0, `Size ${testCase.size}: No valid values found`);
    }
});

test.after(() => {
    console.log('FOSC WASM tests completed');
});
