/**
 * WASM binding tests for EFI indicator.
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

test('EFI partial params', () => {
    // Test with default parameters - mirrors check_efi_partial_params
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    const result = wasm.efi_js(close, volume, 13);
    assert.strictEqual(result.length, close.length);
});

test('EFI accuracy', async () => {
    // Test EFI matches expected values from Rust tests - mirrors check_efi_accuracy
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    const expected = EXPECTED_OUTPUTS.efi;
    
    const result = wasm.efi_js(
        close,
        volume,
        expected.default_params.period
    );
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last_5_values,
        1e-6,
        "EFI last 5 values mismatch"
    );
});

test('EFI zero period', () => {
    // Test EFI fails with zero period - mirrors check_efi_zero_period
    const price = new Float64Array([10.0, 20.0, 30.0]);
    const volume = new Float64Array([100.0, 200.0, 300.0]);
    
    assert.throws(() => {
        wasm.efi_js(price, volume, 0);
    }, /Invalid period/);
});

test('EFI period exceeds length', () => {
    // Test EFI fails when period exceeds data length - mirrors check_efi_period_exceeds_length
    const price = new Float64Array([10.0, 20.0, 30.0]);
    const volume = new Float64Array([100.0, 200.0, 300.0]);
    
    assert.throws(() => {
        wasm.efi_js(price, volume, 10);
    }, /Invalid period/);
});

test('EFI nan handling', () => {
    // Test EFI handles NaN values correctly - mirrors check_efi_nan_handling
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    const result = wasm.efi_js(close, volume, 13);
    assert.strictEqual(result.length, close.length);
    
    // First value should be NaN (need at least 2 values for difference)
    assert(isNaN(result[0]), "First value should be NaN");
    
    // After sufficient data, no NaN values should exist
    // Check that we have non-NaN values after warmup
    const nonNanStart = result.findIndex(v => !isNaN(v));
    assert(nonNanStart >= 0, "All values are NaN");
    
    // After index 50, no NaN values should exist
    if (result.length > 50) {
        const hasNaNAfter50 = result.slice(50).some(v => isNaN(v));
        assert(!hasNaNAfter50, "Found NaN values after warmup period");
    }
});

test('EFI empty data', () => {
    // Test EFI with empty data - mirrors Rust empty data test
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.efi_js(empty, empty, 13);
    }, /Empty data/);
});

test('EFI mismatched lengths', () => {
    // Test EFI with mismatched price and volume lengths
    const price = new Float64Array([1.0, 2.0, 3.0]);
    const volume = new Float64Array([100.0, 200.0]);  // Different length
    
    assert.throws(() => {
        wasm.efi_js(price, volume, 2);
    }, /Empty data/);
});

test('EFI all nan', () => {
    // Test EFI with all NaN values
    const allNan = new Float64Array(100).fill(NaN);
    
    assert.throws(() => {
        wasm.efi_js(allNan, allNan, 13);
    }, /All values are NaN/);
});

test('EFI memory allocation', () => {
    // Test memory allocation and deallocation
    const len = 1000;
    const ptr = wasm.efi_alloc(len);
    
    assert(ptr !== 0, "Failed to allocate memory");
    
    // Cleanup
    wasm.efi_free(ptr, len);
});

test('EFI fast API', () => {
    // Test the fast API (_into function)
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    const len = close.length;
    
    // Allocate output buffer
    const outPtr = wasm.efi_alloc(len);
    
    // Run calculation
    wasm.efi_into(close, volume, outPtr, len, 13);
    
    // Read result
    const memory = new Float64Array(wasm.memory.buffer, outPtr, len);
    const result = Array.from(memory);
    
    // Cleanup
    wasm.efi_free(outPtr, len);
    
    // Verify result matches safe API
    const safeResult = wasm.efi_js(close, volume, 13);
    assertArrayClose(result, safeResult, 1e-10, "Fast API result mismatch");
});

test('EFI batch processing', () => {
    // Test batch processing
    const close = new Float64Array(testData.close.slice(0, 100)); // Use smaller dataset for speed
    const volume = new Float64Array(testData.volume.slice(0, 100));
    
    const config = {
        period_range: [10, 20, 5]  // 10, 15, 20
    };
    
    const result = wasm.efi_batch(close, volume, config);
    
    assert(result.values, "Batch result missing values");
    assert(result.combos, "Batch result missing combos");
    assert.strictEqual(result.rows, 3, "Expected 3 rows");
    assert.strictEqual(result.cols, 100, "Expected 100 columns");
    assert.strictEqual(result.values.length, 300, "Expected 300 values total");
    
    // Verify each row matches individual calculation
    const periods = [10, 15, 20];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = result.values.slice(rowStart, rowEnd);
        
        const singleResult = wasm.efi_js(close, volume, periods[i]);
        assertArrayClose(
            rowData, 
            singleResult, 
            1e-10, 
            `Batch row ${i} (period ${periods[i]}) mismatch`
        );
    }
});

test('EFI aliasing detection', () => {
    // Test that the fast API handles aliasing correctly
    const close = new Float64Array(testData.close.slice(0, 100));
    const volume = new Float64Array(testData.volume.slice(0, 100));
    const len = close.length;
    
    // Use price array as output (aliasing)
    const closePtr = close;
    
    // This should handle aliasing internally
    wasm.efi_into(close, volume, closePtr, len, 13);
    
    // The result should be valid EFI values, not corrupted
    assert(!isNaN(closePtr[10]), "Aliasing produced NaN");
    assert(Math.abs(closePtr[10]) > 0.001, "Aliasing produced zero");
});