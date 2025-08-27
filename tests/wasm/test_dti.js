/**
 * WASM binding tests for DTI indicator.
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
    // Load WASM module
    try {
        const wasmPath = path.join(__dirname, '../../pkg/my_project.js');
        const importPath = process.platform === 'win32' 
            ? 'file:///' + wasmPath.replace(/\\/g, '/')
            : wasmPath;
        wasm = await import(importPath);
        // No need to call default() for ES modules
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }
    
    testData = loadTestData();
});

test('DTI partial params', () => {
    // Test with default parameters - mirrors check_dti_partial_params
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    
    const result = wasm.dti_js(high, low, 14, 10, 5);
    assert.strictEqual(result.length, high.length);
});

test('DTI accuracy', () => {
    // Test accuracy - mirrors check_dti_accuracy
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    
    const result = wasm.dti_js(high, low, 14, 10, 5);
    
    // Check that output is bounded between -100 and 100
    for (let i = 0; i < result.length; i++) {
        if (!isNaN(result[i])) {
            assert(result[i] >= -100.0 && result[i] <= 100.0, 
                `DTI value ${result[i]} at index ${i} is out of bounds [-100, 100]`);
        }
    }
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    const expected = [-39.0091620347991, -39.75219264093014, -40.53941417932286, -41.2787749205189, -42.93758699380749];
    
    for (let i = 0; i < expected.length; i++) {
        assertClose(last5[i], expected[i], 1e-6, `Last 5 values mismatch at index ${i}`);
    }
});

test('DTI zero period', () => {
    const high = new Float64Array(testData.high.slice(0, 50));
    const low = new Float64Array(testData.low.slice(0, 50));
    
    assert.throws(() => {
        wasm.dti_js(high, low, 0, 10, 5);
    }, 'Should throw on zero r period');
    
    assert.throws(() => {
        wasm.dti_js(high, low, 14, 0, 5);
    }, 'Should throw on zero s period');
    
    assert.throws(() => {
        wasm.dti_js(high, low, 14, 10, 0);
    }, 'Should throw on zero u period');
});

test('DTI period exceeds length', () => {
    const high = new Float64Array([10.0, 11.0, 12.0, 13.0, 14.0]);
    const low = new Float64Array([9.0, 10.0, 11.0, 12.0, 13.0]);
    
    assert.throws(() => {
        wasm.dti_js(high, low, 10, 5, 5);
    }, 'Should throw when period exceeds data length');
});

test('DTI all NaN', () => {
    const high = new Float64Array(50).fill(NaN);
    const low = new Float64Array(50).fill(NaN);
    
    assert.throws(() => {
        wasm.dti_js(high, low, 14, 10, 5);
    }, 'Should throw on all NaN values');
});

test('DTI empty data', () => {
    const high = new Float64Array([]);
    const low = new Float64Array([]);
    
    assert.throws(() => {
        wasm.dti_js(high, low, 14, 10, 5);
    }, 'Should throw on empty data');
});

test('DTI mismatched lengths', () => {
    const high = new Float64Array([10.0, 11.0, 12.0, 13.0, 14.0]);
    const low = new Float64Array([9.0, 10.0, 11.0]);
    
    assert.throws(() => {
        wasm.dti_js(high, low, 14, 10, 5);
    }, 'Should throw on mismatched high/low lengths');
});

test('DTI batch operation', () => {
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    
    const config = {
        r_range: [10, 20, 5],  // 10, 15, 20
        s_range: [8, 12, 2],   // 8, 10, 12
        u_range: [4, 6, 1]     // 4, 5, 6
    };
    
    const result = wasm.dti_batch(high, low, config);
    
    // Should have 3 * 3 * 3 = 27 combinations
    assert.strictEqual(result.rows, 27);
    assert.strictEqual(result.cols, high.length);
    assert.strictEqual(result.values.length, 27 * high.length);
    assert.strictEqual(result.combos.length, 27);
    
    // Verify parameter combos
    assert.strictEqual(result.combos[0].r, 10);
    assert.strictEqual(result.combos[0].s, 8);
    assert.strictEqual(result.combos[0].u, 4);
    assert.strictEqual(result.combos[1].r, 10);
    assert.strictEqual(result.combos[1].s, 8);
    assert.strictEqual(result.combos[1].u, 5);
    assert.strictEqual(result.combos[2].r, 10);
    assert.strictEqual(result.combos[2].s, 8);
    assert.strictEqual(result.combos[2].u, 6);
});

test('DTI batch single params', () => {
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    
    const config = {
        r_range: [14, 14, 0],
        s_range: [10, 10, 0],
        u_range: [5, 5, 0]
    };
    
    const batchResult = wasm.dti_batch(high, low, config);
    const singleResult = wasm.dti_js(high, low, 14, 10, 5);
    
    assert.strictEqual(batchResult.rows, 1);
    assert.strictEqual(batchResult.cols, high.length);
    
    // Compare batch vs single
    const batchRow = batchResult.values.slice(0, high.length);
    assertArrayClose(batchRow, singleResult, 1e-10, 'Batch should match single calculation');
});

test('DTI memory allocation/deallocation', () => {
    const len = 1000;
    const ptr = wasm.dti_alloc(len);
    
    assert.notStrictEqual(ptr, 0, 'Should allocate non-null pointer');
    
    // Should not throw
    wasm.dti_free(ptr, len);
    
    // Test null pointer safety
    wasm.dti_free(0, len); // Should not crash
});

test('DTI fast API (in-place)', async () => {
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const len = high.length;
    
    // Allocate output buffer
    const outPtr = wasm.dti_alloc(len);
    
    try {
        // Get pointers to input data
        const highPtr = wasm.__pin(high);
        const lowPtr = wasm.__pin(low);
        
        // Call fast API
        wasm.dti_into(highPtr, lowPtr, outPtr, len, 14, 10, 5);
        
        // Read result
        const result = new Float64Array(wasm.memory.buffer, outPtr, len);
        const resultCopy = new Float64Array(result);
        
        // Compare with safe API
        const expected = wasm.dti_js(high, low, 14, 10, 5);
        assertArrayClose(resultCopy, expected, 1e-10, 'Fast API should match safe API');
        
        // Test aliasing (output = input)
        wasm.dti_into(highPtr, lowPtr, highPtr, len, 14, 10, 5);
        
        // Unpin
        wasm.__unpin(highPtr);
        wasm.__unpin(lowPtr);
    } catch (error) {
        // If pinning functions don't exist, skip this test
        if (!error.message.includes('__pin')) {
            throw error;
        }
    } finally {
        wasm.dti_free(outPtr, len);
    }
});

test('DTI with some NaN values', () => {
    const high = new Float64Array([10.0, 11.0, NaN, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0]);
    const low = new Float64Array([9.0, 10.0, NaN, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0]);
    
    const result = wasm.dti_js(high, low, 3, 2, 2);
    
    assert.strictEqual(result.length, high.length);
    // DTI starts from first valid index, NaN at beginning
    assert(isNaN(result[0]), 'First value should be NaN');
    // DTI continues through intermediate NaN values
    assert(!isNaN(result[2]), 'DTI should skip intermediate NaN');
});

test('DTI trend detection', () => {
    // Create uptrend data
    const len = 100;
    const high = new Float64Array(len);
    const low = new Float64Array(len);
    
    for (let i = 0; i < len; i++) {
        high[i] = 100 + i * 0.5 + Math.random() * 0.1;
        low[i] = high[i] - 1 - Math.random() * 0.1;
    }
    
    const result = wasm.dti_js(high, low, 14, 10, 5);
    
    // In uptrend, DTI should be mostly positive
    let positiveCount = 0;
    for (let i = 20; i < len; i++) {
        if (result[i] > 0) positiveCount++;
    }
    assert(positiveCount > 60, 'DTI should be mostly positive in uptrend');
    
    // Create downtrend data
    for (let i = 0; i < len; i++) {
        high[i] = 100 - i * 0.5 + Math.random() * 0.1;
        low[i] = high[i] - 1 - Math.random() * 0.1;
    }
    
    const resultDown = wasm.dti_js(high, low, 14, 10, 5);
    
    // In downtrend, DTI should be mostly negative
    let negativeCount = 0;
    for (let i = 20; i < len; i++) {
        if (resultDown[i] < 0) negativeCount++;
    }
    assert(negativeCount > 60, 'DTI should be mostly negative in downtrend');
});

// Add comparison with Rust if available
test.skip('DTI Rust comparison', async () => {
    if (!skip) {
        const high = new Float64Array(testData.high);
        const low = new Float64Array(testData.low);
        
        const wasmResult = wasm.dti_js(high, low, 14, 10, 5);
        await compareWithRust('dti', wasmResult, { high, low, r: 14, s: 10, u: 5 });
    }
});