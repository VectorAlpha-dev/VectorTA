/**
 * WASM binding tests for MFI (Money Flow Index) indicator.
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
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }
    
    testData = loadTestData();
});

test('MFI basic calculation', () => {
    // Test basic MFI calculation with default parameters
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    // Calculate typical price (HLC3)
    const typicalPrice = new Float64Array(high.length);
    for (let i = 0; i < high.length; i++) {
        typicalPrice[i] = (high[i] + low[i] + close[i]) / 3.0;
    }
    
    const result = wasm.mfi_js(typicalPrice, volume, 14);
    assert.strictEqual(result.length, typicalPrice.length);
    
    // Check warmup period
    for (let i = 0; i < 13; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i} during warmup`);
    }
    
    // Check that we have valid values after warmup
    let hasValidValues = false;
    for (let i = 13; i < result.length; i++) {
        if (!isNaN(result[i])) {
            hasValidValues = true;
            break;
        }
    }
    assert(hasValidValues, "Expected valid values after warmup period");
    
    // MFI should be bounded between 0 and 100
    for (let i = 13; i < result.length; i++) {
        if (!isNaN(result[i])) {
            assert(result[i] >= 0.0 && result[i] <= 100.0, 
                `MFI value ${result[i]} at index ${i} out of bounds [0, 100]`);
        }
    }
});

test('MFI accuracy', () => {
    // Test MFI matches expected values from Rust tests
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    // Calculate typical price
    const typicalPrice = new Float64Array(high.length);
    for (let i = 0; i < high.length; i++) {
        typicalPrice[i] = (high[i] + low[i] + close[i]) / 3.0;
    }
    
    const result = wasm.mfi_js(typicalPrice, volume, 14);
    
    // Expected values from Rust tests
    const expectedLast5 = [
        38.13874339324763,
        37.44139770113819,
        31.02039511395131,
        28.092605898618896,
        25.905204729397813
    ];
    
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expectedLast5,
        1e-3,  // Allow for some floating point differences
        "MFI last 5 values mismatch"
    );
});

test('MFI custom period', () => {
    // Test MFI with different period values
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    // Calculate typical price
    const typicalPrice = new Float64Array(high.length);
    for (let i = 0; i < high.length; i++) {
        typicalPrice[i] = (high[i] + low[i] + close[i]) / 3.0;
    }
    
    // Test with period=7
    const result7 = wasm.mfi_js(typicalPrice, volume, 7);
    assert.strictEqual(result7.length, typicalPrice.length);
    for (let i = 0; i < 6; i++) {
        assert(isNaN(result7[i]), `Expected NaN at index ${i} for period=7`);
    }
    
    // Test with period=21
    const result21 = wasm.mfi_js(typicalPrice, volume, 21);
    assert.strictEqual(result21.length, typicalPrice.length);
    for (let i = 0; i < 20; i++) {
        assert(isNaN(result21[i]), `Expected NaN at index ${i} for period=21`);
    }
    
    // Results should be different
    let foundDifference = false;
    for (let i = 20; i < result7.length; i++) {
        if (!isNaN(result7[i]) && !isNaN(result21[i]) && Math.abs(result7[i] - result21[i]) > 1e-10) {
            foundDifference = true;
            break;
        }
    }
    assert(foundDifference, "Expected different results for different periods");
});

test('MFI empty input', () => {
    // Test MFI fails with empty input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.mfi_js(empty, empty, 14);
    }, /Empty data/);
});

test('MFI mismatched array lengths', () => {
    // Test MFI fails with mismatched array lengths
    const typicalPrice = new Float64Array([1, 2, 3]);
    const volume = new Float64Array([1, 2]);
    
    assert.throws(() => {
        wasm.mfi_js(typicalPrice, volume, 14);
    }, /Empty data/);
});

test('MFI zero period', () => {
    // Test MFI fails with zero period
    const data = new Float64Array(20).fill(100);
    const volume = new Float64Array(20).fill(1000);
    
    assert.throws(() => {
        wasm.mfi_js(data, volume, 0);
    }, /Invalid period/);
});

test('MFI period exceeds length', () => {
    // Test MFI fails when period exceeds data length
    const data = new Float64Array([100, 101, 102]);
    const volume = new Float64Array([1000, 1001, 1002]);
    
    assert.throws(() => {
        wasm.mfi_js(data, volume, 10);
    }, /Invalid period/);
});

test('MFI all NaN input', () => {
    // Test MFI fails with all NaN values
    const data = new Float64Array(20).fill(NaN);
    const volume = new Float64Array(20).fill(NaN);
    
    assert.throws(() => {
        wasm.mfi_js(data, volume, 14);
    }, /All values are NaN/);
});

test('MFI zero volume', () => {
    // Test MFI with zero volume
    const n = 30;
    const typicalPrice = new Float64Array(n);
    const volume = new Float64Array(n).fill(0);
    
    // Generate some price data
    for (let i = 0; i < n; i++) {
        typicalPrice[i] = 100 + i;
    }
    
    const result = wasm.mfi_js(typicalPrice, volume, 14);
    
    // When volume is zero, MFI should be 0
    for (let i = 14; i < result.length; i++) {
        assert(Math.abs(result[i] - 0.0) < 1e-10, 
            `Expected MFI to be 0 with zero volume, got ${result[i]} at index ${i}`);
    }
});

test('MFI memory allocation/deallocation', () => {
    // Test memory management functions
    const len = 1000;
    
    // Allocate memory
    const ptr = wasm.mfi_alloc(len);
    assert(ptr !== 0, "Expected non-null pointer from mfi_alloc");
    
    // Free memory
    wasm.mfi_free(ptr, len);
    
    // Test null pointer handling
    wasm.mfi_free(0, len); // Should not crash
});

test('MFI fast API (in-place)', () => {
    // Test the fast API with in-place operation
    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    const close = new Float64Array(testData.close.slice(0, 100));
    const volume = new Float64Array(testData.volume.slice(0, 100));
    
    // Calculate typical price
    const typicalPrice = new Float64Array(high.length);
    for (let i = 0; i < high.length; i++) {
        typicalPrice[i] = (high[i] + low[i] + close[i]) / 3.0;
    }
    
    // First get expected result with safe API
    const expected = wasm.mfi_js(typicalPrice, volume, 14);
    
    // Allocate output buffer
    const len = typicalPrice.length;
    const outPtr = wasm.mfi_alloc(len);
    
    try {
        // Use fast API
        wasm.mfi_into(
            typicalPrice.buffer,
            typicalPrice.byteOffset,
            volume.buffer,
            volume.byteOffset,
            outPtr,
            len,
            14
        );
        
        // Read result
        const result = new Float64Array(wasm.memory.buffer, outPtr, len);
        
        // Compare results
        assertArrayClose(
            Array.from(result),
            Array.from(expected),
            1e-10,
            "Fast API result mismatch"
        );
    } finally {
        wasm.mfi_free(outPtr, len);
    }
});

test('MFI fast API with aliasing', () => {
    // Test fast API with aliasing (output = input)
    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    const close = new Float64Array(testData.close.slice(0, 100));
    const volume = new Float64Array(testData.volume.slice(0, 100));
    
    // Calculate typical price and make a copy
    const typicalPrice = new Float64Array(high.length);
    for (let i = 0; i < high.length; i++) {
        typicalPrice[i] = (high[i] + low[i] + close[i]) / 3.0;
    }
    const typicalPriceCopy = new Float64Array(typicalPrice);
    
    // Get expected result
    const expected = wasm.mfi_js(typicalPrice, volume, 14);
    
    // Use fast API with aliasing (output = typical price input)
    wasm.mfi_into(
        typicalPriceCopy.buffer,
        typicalPriceCopy.byteOffset,
        volume.buffer,
        volume.byteOffset,
        typicalPriceCopy.buffer,
        typicalPriceCopy.byteOffset,
        typicalPriceCopy.length,
        14
    );
    
    // Compare results
    assertArrayClose(
        Array.from(typicalPriceCopy),
        Array.from(expected),
        1e-10,
        "Fast API with aliasing result mismatch"
    );
});

test('MFI batch API', () => {
    // Test batch processing with multiple periods
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    // Calculate typical price
    const typicalPrice = new Float64Array(high.length);
    for (let i = 0; i < high.length; i++) {
        typicalPrice[i] = (high[i] + low[i] + close[i]) / 3.0;
    }
    
    // Configure batch parameters
    const config = {
        period_range: [10, 20, 5]  // periods: 10, 15, 20
    };
    
    const result = wasm.mfi_batch(typicalPrice, volume, config);
    
    // Check structure
    assert(result.values, "Expected values in batch result");
    assert(result.combos, "Expected combos in batch result");
    assert.strictEqual(result.rows, 3, "Expected 3 rows (periods)");
    assert.strictEqual(result.cols, typicalPrice.length, "Expected cols to match data length");
    assert.strictEqual(result.combos.length, 3, "Expected 3 parameter combinations");
    
    // Check that values is flat array of correct size
    assert.strictEqual(result.values.length, result.rows * result.cols, 
        "Expected values array to be rows * cols");
    
    // Verify first row matches single calculation
    const singleResult = wasm.mfi_js(typicalPrice, volume, 10);
    const firstRow = result.values.slice(0, result.cols);
    assertArrayClose(
        firstRow,
        singleResult,
        1e-10,
        "Batch first row should match single calculation"
    );
});