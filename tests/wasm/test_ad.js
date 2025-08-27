/**
 * WASM binding tests for AD indicator.
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

test('AD partial params', () => {
    // Test with default parameters - mirrors check_ad_partial_params
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    // AD has no parameters
    const result = wasm.ad_js(high, low, close, volume);
    assert.strictEqual(result.length, close.length);
});

test('AD accuracy', async () => {
    // Test AD matches expected values from Rust tests - mirrors check_ad_accuracy
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    const expected = EXPECTED_OUTPUTS.ad;
    
    const result = wasm.ad_js(high, low, close, volume);
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-1,  // AD values are large, so use larger tolerance
        "AD last 5 values mismatch"
    );
    
    // Compare full output with Rust
    await compareWithRust('ad', result, 'ohlcv', expected.defaultParams);
});

test('AD reinput', () => {
    // Test AD with reinput data - mirrors check_ad_with_slice_data_reinput
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    // First pass
    const firstResult = wasm.ad_js(high, low, close, volume);
    assert.strictEqual(firstResult.length, close.length);
    
    // Second pass - use AD output as all inputs
    const secondResult = wasm.ad_js(firstResult, firstResult, firstResult, firstResult);
    assert.strictEqual(secondResult.length, firstResult.length);
    
    // Check no NaN after index 50
    if (secondResult.length > 50) {
        for (let i = 50; i < secondResult.length; i++) {
            assert(!isNaN(secondResult[i]), `Found unexpected NaN at index ${i}`);
        }
    }
});

test('AD NaN handling', () => {
    // Test AD handles NaN values correctly - mirrors check_ad_accuracy_nan_check
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    const result = wasm.ad_js(high, low, close, volume);
    assert.strictEqual(result.length, close.length);
    
    // AD has no warmup period, check after index 50
    if (result.length > 50) {
        for (let i = 50; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
});

test('AD data length mismatch', () => {
    // Test AD fails with mismatched input lengths
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low.slice(0, 100)); // Shorter
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    assert.throws(() => {
        wasm.ad_js(high, low, close, volume);
    }, /Data length mismatch/);
});

test('AD empty input', () => {
    // Test AD fails with empty input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.ad_js(empty, empty, empty, empty);
    }, /Not enough data/);
});

test('AD batch single security', () => {
    // Test batch with single security
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    // Flatten arrays (single security)
    const batchResult = wasm.ad_batch_js(
        high,
        low,
        close,
        volume,
        1  // 1 security
    );
    
    // Should match single calculation
    const singleResult = wasm.ad_js(high, low, close, volume);
    
    assert.strictEqual(batchResult.length, singleResult.length);
    assertArrayClose(batchResult, singleResult, 1e-10, "Batch vs single mismatch");
});

test('AD batch multiple securities', () => {
    // Test batch with multiple securities
    const high = new Float64Array(testData.high.slice(0, 100)); // Smaller for speed
    const low = new Float64Array(testData.low.slice(0, 100));
    const close = new Float64Array(testData.close.slice(0, 100));
    const volume = new Float64Array(testData.volume.slice(0, 100));
    
    // Create flattened arrays for 3 securities
    const rows = 3;
    const cols = 100;
    const highsFlat = new Float64Array(rows * cols);
    const lowsFlat = new Float64Array(rows * cols);
    const closesFlat = new Float64Array(rows * cols);
    const volumesFlat = new Float64Array(rows * cols);
    
    // Fill flattened arrays
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            const idx = i * cols + j;
            highsFlat[idx] = high[j];
            lowsFlat[idx] = low[j];
            closesFlat[idx] = close[j];
            volumesFlat[idx] = volume[j];
        }
    }
    
    const batchResult = wasm.ad_batch_js(
        highsFlat,
        lowsFlat,
        closesFlat,
        volumesFlat,
        rows
    );
    
    // Should have rows * cols values
    assert.strictEqual(batchResult.length, rows * cols);
    
    // Each row should match individual calculation
    const singleResult = wasm.ad_js(high, low, close, volume);
    for (let i = 0; i < rows; i++) {
        const rowStart = i * cols;
        const rowEnd = rowStart + cols;
        const rowData = batchResult.slice(rowStart, rowEnd);
        
        assertArrayClose(
            rowData,
            singleResult,
            1e-10,
            `Row ${i} mismatch`
        );
    }
});

test('AD batch metadata', () => {
    // Test metadata function returns correct dimensions
    const metadata = wasm.ad_batch_metadata_js(3, 100);
    
    // Should return [rows, cols]
    assert.strictEqual(metadata.length, 2);
    assert.strictEqual(metadata[0], 3);
    assert.strictEqual(metadata[1], 100);
});

test('AD batch edge cases', () => {
    // Test edge cases for batch processing
    const high = new Float64Array([1, 2, 3, 4, 5]);
    const low = new Float64Array([0.5, 1.5, 2.5, 3.5, 4.5]);
    const close = new Float64Array([0.8, 1.8, 2.8, 3.8, 4.8]);
    const volume = new Float64Array([100, 200, 300, 400, 500]);
    
    // Single security
    const singleBatch = wasm.ad_batch_js(
        high,
        low,
        close,
        volume,
        1
    );
    
    assert.strictEqual(singleBatch.length, 5);
    
    // Empty data should throw
    assert.throws(() => {
        wasm.ad_batch_js(
            new Float64Array([]),
            new Float64Array([]),
            new Float64Array([]),
            new Float64Array([]),
            0
        );
    }, /Empty input data/);
});

test('AD all NaN input', () => {
    // Test AD with all NaN values
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    // AD returns all NaN when input is all NaN (doesn't throw error)
    const result = wasm.ad_js(allNaN, allNaN, allNaN, allNaN);
    assert.strictEqual(result.length, allNaN.length);
    
    // Check all values are NaN
    for (let i = 0; i < result.length; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i}`);
    }
});

test('AD no warmup period', () => {
    // Test AD has no warmup period - starts from 0
    const high = new Float64Array(testData.high.slice(0, 50));
    const low = new Float64Array(testData.low.slice(0, 50));
    const close = new Float64Array(testData.close.slice(0, 50));
    const volume = new Float64Array(testData.volume.slice(0, 50));
    
    const result = wasm.ad_js(high, low, close, volume);
    
    // AD should not have NaN at start
    assert(!isNaN(result[0]), "AD should not have NaN at index 0");
});

test('AD high-low validation', () => {
    // Test AD handles invalid high/low relationships
    const high = new Float64Array([100.0, 90.0, 95.0]);
    const low = new Float64Array([105.0, 95.0, 90.0]); // low > high
    const close = new Float64Array([102.0, 92.0, 93.0]);
    const volume = new Float64Array([1000.0, 1500.0, 1200.0]);
    
    // Should handle gracefully
    const result = wasm.ad_js(high, low, close, volume);
    assert.strictEqual(result.length, close.length);
});

test('AD zero volume', () => {
    // Test AD with zero volume periods
    const high = new Float64Array([100, 101, 102, 103, 104]);
    const low = new Float64Array([99, 100, 101, 102, 103]);
    const close = new Float64Array([99.5, 100.5, 101.5, 102.5, 103.5]);
    const volume = new Float64Array([1000, 0, 2000, 0, 3000]); // Zero volumes
    
    const result = wasm.ad_js(high, low, close, volume);
    
    // When volume is 0, AD should not change
    assertClose(result[1], result[0], 1e-10, "AD should not change when volume is 0");
    assertClose(result[3], result[2], 1e-10, "AD should not change when volume is 0");
});

test('AD batch - new ergonomic API', () => {
    // Test the new ergonomic batch API with structured output
    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    const close = new Float64Array(testData.close.slice(0, 100));
    const volume = new Float64Array(testData.volume.slice(0, 100));
    
    // Create flattened arrays for 2 securities
    const rows = 2;
    const cols = 100;
    const highsFlat = new Float64Array(rows * cols);
    const lowsFlat = new Float64Array(rows * cols);
    const closesFlat = new Float64Array(rows * cols);
    const volumesFlat = new Float64Array(rows * cols);
    
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            const idx = i * cols + j;
            highsFlat[idx] = high[j];
            lowsFlat[idx] = low[j];
            closesFlat[idx] = close[j];
            volumesFlat[idx] = volume[j];
        }
    }
    
    // Use new ergonomic API (ad_batch)
    const result = wasm.ad_batch(highsFlat, lowsFlat, closesFlat, volumesFlat, rows);
    
    // Verify structure
    assert(result.values, 'Should have values array');
    assert(typeof result.rows === 'number', 'Should have rows count');
    assert(typeof result.cols === 'number', 'Should have cols count');
    assert.strictEqual(result.rows, 2);
    assert.strictEqual(result.cols, 100);
    assert.strictEqual(result.values.length, 200);
});

test('AD zero-copy API', () => {
    // Test zero-copy API for AD
    const data_len = 10;
    const high = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const low = new Float64Array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]);
    const close = new Float64Array([0.8, 1.8, 2.8, 3.8, 4.8, 5.8, 6.8, 7.8, 8.8, 9.8]);
    const volume = new Float64Array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]);
    
    // Allocate buffers
    const highPtr = wasm.ad_alloc(data_len);
    const lowPtr = wasm.ad_alloc(data_len);
    const closePtr = wasm.ad_alloc(data_len);
    const volumePtr = wasm.ad_alloc(data_len);
    const outPtr = wasm.ad_alloc(data_len);
    
    assert(highPtr !== 0, 'Failed to allocate high buffer');
    assert(lowPtr !== 0, 'Failed to allocate low buffer');
    assert(closePtr !== 0, 'Failed to allocate close buffer');
    assert(volumePtr !== 0, 'Failed to allocate volume buffer');
    assert(outPtr !== 0, 'Failed to allocate output buffer');
    
    try {
        // Copy data to WASM memory
        const highView = new Float64Array(wasm.__wasm.memory.buffer, highPtr, data_len);
        const lowView = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, data_len);
        const closeView = new Float64Array(wasm.__wasm.memory.buffer, closePtr, data_len);
        const volumeView = new Float64Array(wasm.__wasm.memory.buffer, volumePtr, data_len);
        
        highView.set(high);
        lowView.set(low);
        closeView.set(close);
        volumeView.set(volume);
        
        // Compute AD
        wasm.ad_into(highPtr, lowPtr, closePtr, volumePtr, outPtr, data_len);
        
        // Get results
        const outView = new Float64Array(wasm.__wasm.memory.buffer, outPtr, data_len);
        
        // Compare with regular API
        const regularResult = wasm.ad_js(high, low, close, volume);
        for (let i = 0; i < data_len; i++) {
            assertClose(outView[i], regularResult[i], 1e-10, 
                       `Zero-copy mismatch at index ${i}`);
        }
    } finally {
        // Clean up
        wasm.ad_free(highPtr, data_len);
        wasm.ad_free(lowPtr, data_len);
        wasm.ad_free(closePtr, data_len);
        wasm.ad_free(volumePtr, data_len);
        wasm.ad_free(outPtr, data_len);
    }
});

test('AD memory management', () => {
    // Test multiple allocations and frees
    const sizes = [100, 1000, 10000];
    
    for (const size of sizes) {
        const ptr = wasm.ad_alloc(size);
        assert(ptr !== 0, `Failed to allocate ${size} elements`);
        
        // Write pattern
        const view = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            view[i] = i * 1.5;
        }
        
        // Verify pattern
        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(view[i], i * 1.5, `Memory corruption at index ${i}`);
        }
        
        wasm.ad_free(ptr, size);
    }
});

test('AD SIMD128 consistency', () => {
    // Verify SIMD produces same results
    const testCases = [
        { size: 10 },
        { size: 100 },
        { size: 1000 }
    ];
    
    for (const testCase of testCases) {
        const high = new Float64Array(testCase.size);
        const low = new Float64Array(testCase.size);
        const close = new Float64Array(testCase.size);
        const volume = new Float64Array(testCase.size);
        
        for (let i = 0; i < testCase.size; i++) {
            // Generate more realistic OHLC data where close varies relative to high/low
            const base = 100 + Math.sin(i * 0.1) * 10;
            const range = 5 + Math.abs(Math.sin(i * 0.15)) * 5;
            high[i] = base + range;
            low[i] = base - range;
            // Close varies between high and low, not always in middle
            const closeRatio = 0.5 + 0.4 * Math.sin(i * 0.2);
            close[i] = low[i] + (high[i] - low[i]) * closeRatio;
            volume[i] = 1000 + i * 10;
        }
        
        const result = wasm.ad_js(high, low, close, volume);
        
        // Basic sanity checks
        assert.strictEqual(result.length, testCase.size);
        
        // AD should not have NaN values
        for (let i = 0; i < result.length; i++) {
            assert(!isNaN(result[i]), `Unexpected NaN at index ${i} for size=${testCase.size}`);
        }
        
        // AD should be cumulative
        // Verify values are computed correctly (not all zeros or same value)
        if (testCase.size > 10) {
            let hasVariation = false;
            for (let i = 1; i < result.length; i++) {
                if (Math.abs(result[i] - result[i-1]) > 1e-10) {
                    hasVariation = true;
                    break;
                }
            }
            assert(hasVariation, "AD values should show variation");
        }
    }
});

// Note: Streaming tests would require streaming functions to be exposed in WASM bindings

test.after(() => {
    console.log('AD WASM tests completed');
});