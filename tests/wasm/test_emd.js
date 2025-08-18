/**
 * WASM binding tests for EMD indicator.
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

test('EMD accuracy', () => {
    // Test against expected values from Rust unit tests
    const result = wasm.emd_js(
        testData.high,
        testData.low,
        testData.close,
        testData.volume,
        20,      // period
        0.5,     // delta
        0.1      // fraction
    );
    
    assert.strictEqual(result.rows, 3, 'Should have 3 output rows');
    assert.strictEqual(result.cols, testData.close.length, 'Cols should match input length');
    
    const values = result.values;
    const len = testData.close.length;
    
    // Extract the three bands
    const upperband = values.slice(0, len);
    const middleband = values.slice(len, 2 * len);
    const lowerband = values.slice(2 * len, 3 * len);
    
    // Expected values from Rust tests (last 5 values)
    const expectedLastFiveUpper = [
        50.33760237677157,
        50.28850695686447,
        50.23941153695737,
        50.19031611705027,
        48.709744457737344,
    ];
    const expectedLastFiveMiddle = [
        -368.71064280396706,
        -399.11033986231377,
        -421.9368852621732,
        -437.879217150269,
        -447.3257167904511,
    ];
    const expectedLastFiveLower = [
        -60.67834136221248,
        -60.93110347122829,
        -61.68154077026321,
        -62.43197806929814,
        -63.18241536833306,
    ];
    
    // Check last 5 values
    for (let i = 0; i < 5; i++) {
        assertClose(
            upperband[len - 5 + i],
            expectedLastFiveUpper[i],
            1e-6,
            `EMD upperband mismatch at index ${len - 5 + i}`
        );
        assertClose(
            middleband[len - 5 + i],
            expectedLastFiveMiddle[i],
            1e-6,
            `EMD middleband mismatch at index ${len - 5 + i}`
        );
        assertClose(
            lowerband[len - 5 + i],
            expectedLastFiveLower[i],
            1e-6,
            `EMD lowerband mismatch at index ${len - 5 + i}`
        );
    }
});

test('EMD error handling', () => {
    // Test with empty data
    assert.throws(() => {
        wasm.emd_js([], [], [], [], 20, 0.5, 0.1);
    }, /All values are NaN/, 'Should throw on empty data');
    
    // Test with mismatched lengths
    assert.throws(() => {
        wasm.emd_js(
            [1, 2, 3],
            [1, 2],      // Different length
            [1, 2, 3],
            [1, 2, 3],
            20, 0.5, 0.1
        );
    }, /must have the same length/, 'Should throw on mismatched array lengths');
    
    // Test with invalid period
    assert.throws(() => {
        const data = new Float64Array(10).fill(1.0);
        wasm.emd_js(data, data, data, data, 0, 0.5, 0.1);
    }, /Invalid period/, 'Should throw on zero period');
    
    // Test with not enough data
    assert.throws(() => {
        const data = new Float64Array(10).fill(1.0);
        wasm.emd_js(data, data, data, data, 20, 0.5, 0.1);
    }, /Invalid period/, 'Should throw on insufficient data');
});

test('EMD fast/unsafe API', () => {
    const len = testData.close.length;
    
    // Allocate input and output buffers
    const highPtr = wasm.emd_alloc(len);
    const lowPtr = wasm.emd_alloc(len);
    const closePtr = wasm.emd_alloc(len);
    const volumePtr = wasm.emd_alloc(len);
    const upperPtr = wasm.emd_alloc(len);
    const middlePtr = wasm.emd_alloc(len);
    const lowerPtr = wasm.emd_alloc(len);
    
    try {
        // Copy input data to WASM memory
        const highView = new Float64Array(wasm.__wasm.memory.buffer, highPtr, len);
        const lowView = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, len);
        const closeView = new Float64Array(wasm.__wasm.memory.buffer, closePtr, len);
        const volumeView = new Float64Array(wasm.__wasm.memory.buffer, volumePtr, len);
        
        highView.set(testData.high);
        lowView.set(testData.low);
        closeView.set(testData.close);
        volumeView.set(testData.volume);
        
        // Call fast API
        wasm.emd_into(
            highPtr,
            lowPtr,
            closePtr,
            volumePtr,
            upperPtr,
            middlePtr,
            lowerPtr,
            len,
            20,      // period
            0.5,     // delta
            0.1      // fraction
        );
        
        // Create typed arrays from pointers (recreate in case memory grew)
        const upperband = new Float64Array(wasm.__wasm.memory.buffer, upperPtr, len);
        const middleband = new Float64Array(wasm.__wasm.memory.buffer, middlePtr, len);
        const lowerband = new Float64Array(wasm.__wasm.memory.buffer, lowerPtr, len);
        
        // Expected values from Rust tests (last 5 values)
        const expectedLastFiveUpper = [
            50.33760237677157,
            50.28850695686447,
            50.23941153695737,
            50.19031611705027,
            48.709744457737344,
        ];
        
        // Check last 5 values of upperband
        for (let i = 0; i < 5; i++) {
            assertClose(
                upperband[len - 5 + i],
                expectedLastFiveUpper[i],
                1e-6,
                `Fast API upperband mismatch at index ${len - 5 + i}`
            );
        }
    } finally {
        // Clean up allocated memory
        wasm.emd_free(highPtr, len);
        wasm.emd_free(lowPtr, len);
        wasm.emd_free(closePtr, len);
        wasm.emd_free(volumePtr, len);
        wasm.emd_free(upperPtr, len);
        wasm.emd_free(middlePtr, len);
        wasm.emd_free(lowerPtr, len);
    }
});

test('EMD in-place operation (aliasing)', () => {
    const len = testData.close.length;
    
    // Allocate buffers for all inputs and outputs
    const highPtr = wasm.emd_alloc(len);  // Will also be used for upperband output
    const lowPtr = wasm.emd_alloc(len);
    const closePtr = wasm.emd_alloc(len);
    const volumePtr = wasm.emd_alloc(len);
    const middlePtr = wasm.emd_alloc(len);
    const lowerPtr = wasm.emd_alloc(len);
    
    try {
        // Copy input data to WASM memory
        const highView = new Float64Array(wasm.__wasm.memory.buffer, highPtr, len);
        const lowView = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, len);
        const closeView = new Float64Array(wasm.__wasm.memory.buffer, closePtr, len);
        const volumeView = new Float64Array(wasm.__wasm.memory.buffer, volumePtr, len);
        
        highView.set(testData.high);
        lowView.set(testData.low);
        closeView.set(testData.close);
        volumeView.set(testData.volume);
        
        // Call fast API with input as output (aliasing - highPtr used for both input and upperband output)
        wasm.emd_into(
            highPtr,         // high input
            lowPtr,
            closePtr,
            volumePtr,
            highPtr,         // upperband output (same as high input - aliasing!)
            middlePtr,
            lowerPtr,
            len,
            20,      // period
            0.5,     // delta
            0.1      // fraction
        );
        
        // Recreate view after operation
        const upperband = new Float64Array(wasm.__wasm.memory.buffer, highPtr, len);
        
        // The data array should now contain upperband values
        const expectedLastFiveUpper = [
            50.33760237677157,
            50.28850695686447,
            50.23941153695737,
            50.19031611705027,
            48.709744457737344,
        ];
        
        // Check that in-place operation worked correctly
        for (let i = 0; i < 5; i++) {
            assertClose(
                upperband[len - 5 + i],
                expectedLastFiveUpper[i],
                1e-6,
                `In-place upperband mismatch at index ${len - 5 + i}`
            );
        }
    } finally {
        wasm.emd_free(highPtr, len);
        wasm.emd_free(lowPtr, len);
        wasm.emd_free(closePtr, len);
        wasm.emd_free(volumePtr, len);
        wasm.emd_free(middlePtr, len);
        wasm.emd_free(lowerPtr, len);
    }
});

test('EMD batch processing', () => {
    const config = {
        period_range: [20, 22, 2],      // 2 values: 20, 22
        delta_range: [0.5, 0.6, 0.1],   // 2 values: 0.5, 0.6
        fraction_range: [0.1, 0.2, 0.1] // 2 values: 0.1, 0.2
    };
    
    const result = wasm.emd_batch(
        testData.high,
        testData.low,
        testData.close,
        testData.volume,
        config
    );
    
    // Should have 2*2*2 = 8 combinations
    assert.strictEqual(result.rows, 8, 'Should have 8 parameter combinations');
    assert.strictEqual(result.cols, testData.close.length, 'Cols should match input length');
    assert.strictEqual(result.upperband.length, 8 * testData.close.length, 'Upperband should have rows*cols elements');
    assert.strictEqual(result.middleband.length, 8 * testData.close.length, 'Middleband should have rows*cols elements');
    assert.strictEqual(result.lowerband.length, 8 * testData.close.length, 'Lowerband should have rows*cols elements');
    assert.strictEqual(result.combos.length, 8, 'Should have 8 parameter combinations');
    
    // Verify first combination matches single calculation
    const singleResult = wasm.emd_js(
        testData.high,
        testData.low,
        testData.close,
        testData.volume,
        20,      // period
        0.5,     // delta
        0.1      // fraction
    );
    
    const singleValues = singleResult.values;
    const len = testData.close.length;
    const singleUpper = singleValues.slice(0, len);
    
    // Compare first row of batch with single calculation
    for (let i = len - 5; i < len; i++) {
        assertClose(
            result.upperband[i],
            singleUpper[i],
            1e-6,
            `Batch upperband mismatch at index ${i}`
        );
    }
});

test('EMD batch fast API', () => {
    const len = testData.close.length;
    const expectedRows = 2 * 2 * 2; // 8 combinations
    const totalLen = expectedRows * len;
    
    // Allocate input and output buffers
    const highPtr = wasm.emd_alloc(len);
    const lowPtr = wasm.emd_alloc(len);
    const closePtr = wasm.emd_alloc(len);
    const volumePtr = wasm.emd_alloc(len);
    const upperPtr = wasm.emd_alloc(totalLen);
    const middlePtr = wasm.emd_alloc(totalLen);
    const lowerPtr = wasm.emd_alloc(totalLen);
    
    try {
        // Copy input data to WASM memory
        const highView = new Float64Array(wasm.__wasm.memory.buffer, highPtr, len);
        const lowView = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, len);
        const closeView = new Float64Array(wasm.__wasm.memory.buffer, closePtr, len);
        const volumeView = new Float64Array(wasm.__wasm.memory.buffer, volumePtr, len);
        
        highView.set(testData.high);
        lowView.set(testData.low);
        closeView.set(testData.close);
        volumeView.set(testData.volume);
        
        const rows = wasm.emd_batch_into(
            highPtr,
            lowPtr,
            closePtr,
            volumePtr,
            upperPtr,
            middlePtr,
            lowerPtr,
            len,
            20, 22, 2,      // period range
            0.5, 0.6, 0.1,  // delta range
            0.1, 0.2, 0.1   // fraction range
        );
        
        assert.strictEqual(rows, expectedRows, 'Should return correct number of rows');
        
        // Create typed arrays from pointers (recreate in case memory grew)
        const upperband = new Float64Array(wasm.__wasm.memory.buffer, upperPtr, totalLen);
        
        // Verify first row matches single calculation
        const singleResult = wasm.emd_js(
            testData.high,
            testData.low,
            testData.close,
            testData.volume,
            20,      // period
            0.5,     // delta
            0.1      // fraction
        );
        
        const singleValues = singleResult.values;
        const singleUpper = singleValues.slice(0, len);
        
        // Compare first row of batch with single calculation
        for (let i = len - 5; i < len; i++) {
            assertClose(
                upperband[i],
                singleUpper[i],
                1e-6,
                `Batch fast API upperband mismatch at index ${i}`
            );
        }
    } finally {
        wasm.emd_free(highPtr, len);
        wasm.emd_free(lowPtr, len);
        wasm.emd_free(closePtr, len);
        wasm.emd_free(volumePtr, len);
        wasm.emd_free(upperPtr, totalLen);
        wasm.emd_free(middlePtr, totalLen);
        wasm.emd_free(lowerPtr, totalLen);
    }
});

test.after(() => {
    console.log('EMD WASM tests completed');
});
