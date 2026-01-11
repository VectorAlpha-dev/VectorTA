/**
 * WASM binding tests for DEVIATION indicator.
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
        
        
        
        try {
            const { createRequire } = await import('node:module');
            const require = createRequire(import.meta.url);
            
            
            wasm = require(path.join(__dirname, 'my_project.cjs'));
        } catch (fallbackErr) {
            console.error('Failed to load WASM module via pkg and local wrapper.');
            console.error('Hint: run `wasm-pack build --features wasm --target nodejs` and ensure Node >=18.');
            throw fallbackErr;
        }
    }

    testData = loadTestData();
});

test('Deviation basic functionality', () => {
    const close = new Float64Array(testData.close);
    
    
    const result = wasm.deviation_js(close, 20, 0);
    assert(result, 'Should have result');
    assert.strictEqual(result.length, close.length, 'Result length should match input');
    
    
    for (let i = 0; i < 19; i++) {
        assert(isNaN(result[i]), `Value at index ${i} should be NaN during warmup`);
    }
    
    
    for (let i = 19; i < result.length; i++) {
        assert(!isNaN(result[i]), `Value at index ${i} should not be NaN after warmup`);
        assert(result[i] >= 0, `Standard deviation at index ${i} should be non-negative`);
    }
});

test('Deviation accuracy (last 5 match references)', () => {
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.deviation;
    const { period, devtype } = expected.defaultParams;
    const result = wasm.deviation_js(close, period, devtype);
    assert.strictEqual(result.length, close.length);
    const last5 = Array.from(result.slice(-5));
    assertArrayClose(last5, expected.last5Values, 2e-8, 'Deviation last 5 values mismatch');
});

test('Deviation different types', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    
    
    const types = [
        { type: 0, name: 'Simple' },
        { type: 1, name: 'MeanAbsolute' },
        { type: 2, name: 'Median' },
        { type: 3, name: 'Mode' }
    ];
    
    for (const { type, name } of types) {
        const result = wasm.deviation_js(data, 5, type);
        assert(result, `Should have result for ${name} deviation`);
        assert.strictEqual(result.length, data.length, `Result length should match input for ${name}`);
        
        
        for (let i = 0; i < 4; i++) {
            assert(isNaN(result[i]), `${name}: Value at index ${i} should be NaN during warmup`);
        }
        
        
        for (let i = 4; i < result.length; i++) {
            assert(!isNaN(result[i]), `${name}: Value at index ${i} should not be NaN after warmup`);
        }
    }
});

test('Deviation error handling', () => {
    const data = new Float64Array([1, 2, 3, 4, 5]);
    
    
    assert.throws(() => {
        wasm.deviation_js(data, 10, 0);
    }, /NotEnoughData|period/);
    
    
    assert.throws(() => {
        wasm.deviation_js(data, 0, 0);
    }, /InvalidPeriod|period/);
    
    
    const emptyData = new Float64Array([]);
    assert.throws(() => {
        wasm.deviation_js(emptyData, 2, 0);
    }, /NotEnoughData|EmptyData|empty input data/i);
});

test('Deviation fast API (in-place)', () => {
    const close = new Float64Array(testData.close.slice(0, 100));
    const len = close.length;
    
    
    const inPtr = wasm.deviation_alloc(len);
    const outPtr = wasm.deviation_alloc(len);
    
    try {
        
        const memory = wasm.__wasm.memory.buffer;
        const inView = new Float64Array(memory, inPtr, len);
        const outView = new Float64Array(memory, outPtr, len);
        
        
        inView.set(close);
        
        
        wasm.deviation_into(inPtr, len, 20, 0, outPtr);
        
        
        const safeResult = wasm.deviation_js(close, 20, 0);
        assertArrayClose(Array.from(outView), safeResult, 1e-10, "Fast API should match safe API");
        
        
        inView.set(close); 
        wasm.deviation_into(inPtr, len, 20, 0, inPtr); 
        assertArrayClose(Array.from(inView), safeResult, 1e-10, "In-place should match safe API");
    } finally {
        
        wasm.deviation_free(inPtr, len);
        wasm.deviation_free(outPtr, len);
    }
});

test('Deviation batch single parameter set', () => {
    const close = new Float64Array(testData.close.slice(0, 100));
    
    
    const batchResult = wasm.deviation_batch(close, {
        period_range: [20, 20, 0],
        devtype_range: [0, 0, 0]
    });
    
    assert(batchResult, 'Should have result');
    assert.strictEqual(batchResult.combos, 1, 'Should have 1 combination');
    assert.strictEqual(batchResult.cols, close.length, 'Cols should match input length');
    assert.strictEqual(batchResult.values.length, close.length, 'Values should match input length');
    
    
    const singleResult = wasm.deviation_js(close, 20, 0);
    
    
    assertArrayClose(batchResult.values, singleResult, 3e-10, "Batch vs single mismatch");
});

test('Deviation batch multiple parameters', () => {
    const close = new Float64Array(testData.close.slice(0, 100));
    
    
    const batchResult = wasm.deviation_batch(close, {
        period_range: [10, 30, 10],   
        devtype_range: [0, 2, 1]       
    });
    
    
    assert.strictEqual(batchResult.combos, 9);
    assert.strictEqual(batchResult.cols, 100);
    assert.strictEqual(batchResult.values.length, 9 * 100);
    
    
    const firstCombo = batchResult.values.slice(0, 100);
    const singleResult = wasm.deviation_js(close, 10, 0);
    assertArrayClose(firstCombo, singleResult, 3e-10, "First combination mismatch");
});

test('Deviation batch metadata', () => {
    
    const metadata = wasm.deviation_batch_metadata(
        10, 30, 10,   
        0, 2, 1       
    );
    
    
    assert.strictEqual(metadata.length, 18);
    
    
    const expected = [
        10, 0,  
        10, 1,  
        10, 2,  
        20, 0,  
        20, 1,  
        20, 2,  
        30, 0,  
        30, 1,  
        30, 2   
    ];
    
    assertArrayClose(metadata, expected, 0, "Metadata mismatch");
});

test('Deviation streaming API', () => {
    const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const period = 5;
    
    
    const stream = new wasm.DeviationStream(period, 0);
    
    
    const results = [];
    for (const value of data) {
        const result = stream.update(value);
        results.push(result === undefined ? NaN : result);
    }
    
    
    const batchResult = wasm.deviation_js(new Float64Array(data), period, 0);
    assertArrayClose(results, batchResult, 1e-10, "Streaming vs batch mismatch");
});

test('Deviation NaN handling', () => {
    const dataWithNaN = new Float64Array([1, 2, NaN, 4, 5, 6, 7, 8, 9, 10]);
    
    const result = wasm.deviation_js(dataWithNaN, 5, 0);
    assert(result, 'Should handle NaN in input');
    assert.strictEqual(result.length, dataWithNaN.length);
    
    
    assert(isNaN(result[2]), 'NaN should propagate');
    assert(isNaN(result[3]), 'NaN should affect window');
    assert(isNaN(result[4]), 'NaN should affect window');
    assert(isNaN(result[5]), 'NaN should affect window');
    assert(isNaN(result[6]), 'NaN should affect window');
    
    
    assert(!isNaN(result[7]), 'Should recover after NaN leaves window');
});

test('Deviation edge cases', () => {
    
    const data = new Float64Array([1, 2, 3, 4, 5]);
    const result = wasm.deviation_js(data, 2, 0);
    assert.strictEqual(result.length, data.length);
    assert(isNaN(result[0]));
    assert(!isNaN(result[1]));
    
    
    const sameData = new Float64Array(10).fill(42.0);
    const sameResult = wasm.deviation_js(sameData, 5, 0);
    for (let i = 4; i < sameResult.length; i++) {
        assertClose(sameResult[i], 0.0, 1e-10, `Deviation should be 0 for identical values at index ${i}`);
    }
});

test('Deviation fast batch API', () => {
    const close = new Float64Array(testData.close.slice(0, 100));
    const len = close.length;
    
    
    const inPtr = wasm.deviation_alloc(len);
    const combos = 3 * 3; 
    const outPtr = wasm.deviation_alloc(len * combos);
    
    try {
        
        const memory = wasm.__wasm.memory.buffer;
        const inView = new Float64Array(memory, inPtr, len);
        
        
        inView.set(close);
        
        
        const rows = wasm.deviation_batch_into(
            inPtr, outPtr, len,
            10, 30, 10,  
            0, 2, 1      
        );
        
        assert.strictEqual(rows, 9, 'Should return 9 combinations');
        
        
        const outView = new Float64Array(memory, outPtr, len * combos);
        
        
        const firstCombo = Array.from(outView.slice(0, len));
        const expected = wasm.deviation_js(close, 10, 0);
        assertArrayClose(firstCombo, expected, 3e-10, "First batch combo should match single calc");
    } finally {
        
        wasm.deviation_free(inPtr, len);
        wasm.deviation_free(outPtr, len * combos);
    }
});

test('Deviation invalid devtype', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    
    
    assert.throws(() => {
        wasm.deviation_js(data, 5, 4);
    }, /Invalid.*devtype|calculation error/i);
});

test('Deviation period edge validation', () => {
    const data = new Float64Array(100).fill(1.0).map((_, i) => i + 1);
    
    
    const result1 = wasm.deviation_js(data, 1, 0);
    assert.strictEqual(result1.length, data.length);
    
    for (let i = 0; i < result1.length; i++) {
        assertClose(result1[i], 0.0, 1e-10, `Period=1 should give 0 deviation at index ${i}`);
    }
    
    
    const resultMax = wasm.deviation_js(data, data.length, 0);
    assert.strictEqual(resultMax.length, data.length);
    
    for (let i = 0; i < data.length - 1; i++) {
        assert(isNaN(resultMax[i]), `Should be NaN at index ${i}`);
    }
    assert(!isNaN(resultMax[data.length - 1]), 'Last value should be valid');
});

test('Deviation streaming vs batch consistency', () => {
    
    const data = [10, 15, 20, 18, 22, 25, 23, 27, 30, 28];
    const period = 4;
    
    for (let devtype = 0; devtype <= 3; devtype++) {
        const stream = new wasm.DeviationStream(period, devtype);
        const streamResults = [];
        
        for (const value of data) {
            const result = stream.update(value);
            streamResults.push(result === undefined ? NaN : result);
        }
        
        const batchResult = wasm.deviation_js(new Float64Array(data), period, devtype);
        assertArrayClose(
            streamResults, 
            batchResult, 
            1e-10, 
            `Streaming vs batch mismatch for devtype ${devtype}`
        );
    }
});

test('Deviation increasing/decreasing trends', () => {
    
    const increasingData = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const decreasingData = new Float64Array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1]);
    const period = 5;
    
    const incResult = wasm.deviation_js(increasingData, period, 0);
    const decResult = wasm.deviation_js(decreasingData, period, 0);
    
    
    for (let i = period - 1; i < increasingData.length; i++) {
        assertClose(
            incResult[i], 
            decResult[i], 
            1e-10, 
            `Deviation should be same for symmetric data at index ${i}`
        );
    }
});

test.after(() => {
    console.log('DEVIATION WASM tests completed');
});
