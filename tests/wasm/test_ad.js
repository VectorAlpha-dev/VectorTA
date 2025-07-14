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

// Note: Streaming tests would require streaming functions to be exposed in WASM bindings

test.after(() => {
    console.log('AD WASM tests completed');
});