/**
 * WASM binding tests for MarketEFI indicator.
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

test('MarketEFI accuracy', async () => {
    // Test MarketEFI matches expected values from Rust tests - mirrors check_marketefi_accuracy
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const volume = new Float64Array(testData.volume);
    
    const result = wasm.marketefi_js(high, low, volume);
    
    assert.strictEqual(result.length, high.length);
    
    // Expected last 5 values from Rust test
    const expectedLastFive = [
        2.8460112192104607,
        3.020938522420525,
        3.0474861329079292,
        3.691017115591989,
        2.247810963176202,
    ];
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expectedLastFive,
        1e-6,
        "MarketEFI last 5 values mismatch"
    );
    
    // Compare full output with Rust
    await compareWithRust('marketefi', result, 'hlv', {});
});

test('MarketEFI NaN handling', () => {
    // Test MarketEFI NaN handling - mirrors check_marketefi_nan_handling
    const high = new Float64Array([NaN, 2.0, 3.0]);
    const low = new Float64Array([NaN, 1.0, 2.0]);
    const volume = new Float64Array([NaN, 1.0, 1.0]);
    
    const result = wasm.marketefi_js(high, low, volume);
    
    assert(isNaN(result[0]), "First value should be NaN");
    assertClose(result[1], 1.0, 1e-8, "Second value mismatch");
    assertClose(result[2], 1.0, 1e-8, "Third value mismatch");
});

test('MarketEFI empty data', () => {
    // Test MarketEFI with empty data - mirrors check_marketefi_empty_data
    const high = new Float64Array([]);
    const low = new Float64Array([]);
    const volume = new Float64Array([]);
    
    assert.throws(() => {
        wasm.marketefi_js(high, low, volume);
    }, /Empty data/);
});

test('MarketEFI mismatched lengths', () => {
    // Test MarketEFI with mismatched input lengths
    const high = new Float64Array([1.0, 2.0, 3.0]);
    const low = new Float64Array([0.5, 1.5]); // Different length
    const volume = new Float64Array([1.0, 1.0, 1.0]);
    
    assert.throws(() => {
        wasm.marketefi_js(high, low, volume);
    }, /Mismatched data length/);
});

test('MarketEFI zero volume', () => {
    // Test MarketEFI with zero volume
    const high = new Float64Array([2.0, 3.0, 4.0]);
    const low = new Float64Array([1.0, 2.0, 3.0]);
    const volume = new Float64Array([1.0, 0.0, 2.0]); // Zero volume in middle
    
    const result = wasm.marketefi_js(high, low, volume);
    
    assertClose(result[0], 1.0, 1e-8);
    assert(isNaN(result[1]), "Zero volume should produce NaN");
    assertClose(result[2], 0.5, 1e-8);
});

test('MarketEFI all NaN input', () => {
    // Test MarketEFI with all NaN values
    const high = new Float64Array(10).fill(NaN);
    const low = new Float64Array(10).fill(NaN);
    const volume = new Float64Array(10).fill(NaN);
    
    assert.throws(() => {
        wasm.marketefi_js(high, low, volume);
    }, /All values are NaN/);
});

test('MarketEFI batch single row', () => {
    // Test MarketEFI batch functionality - since no parameters, returns single row
    const high = testData.high.slice(0, 100); // Use smaller dataset for speed
    const low = testData.low.slice(0, 100);
    const volume = testData.volume.slice(0, 100);
    
    const config = {}; // Empty config since no parameters
    const batchResult = wasm.marketefi_batch(
        new Float64Array(high),
        new Float64Array(low),
        new Float64Array(volume),
        config
    );
    
    assert(batchResult.values);
    assert.strictEqual(batchResult.rows, 1, "Should have 1 row (no parameter sweep)");
    assert.strictEqual(batchResult.cols, 100, "Should have 100 columns");
    
    // Compare with single calculation
    const singleResult = wasm.marketefi_js(
        new Float64Array(high),
        new Float64Array(low),
        new Float64Array(volume)
    );
    
    // Extract first row and compare
    const firstRow = batchResult.values.slice(0, 100);
    assertArrayClose(firstRow, singleResult, 1e-10, "Batch should match single result");
});

test('MarketEFI fast API (in-place)', () => {
    // Test the fast/unsafe API with aliasing
    const len = 3;
    
    // Allocate buffers
    const highPtr = wasm.marketefi_alloc(len);
    const lowPtr = wasm.marketefi_alloc(len);
    const volumePtr = wasm.marketefi_alloc(len);
    
    try {
        // Initialize data
        const highView = new Float64Array(wasm.__wasm.memory.buffer, highPtr, len);
        const lowView = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, len);
        const volumeView = new Float64Array(wasm.__wasm.memory.buffer, volumePtr, len);
        
        highView.set([3.0, 4.0, 5.0]);
        lowView.set([2.0, 3.0, 3.0]);
        volumeView.set([1.0, 2.0, 2.0]);
        
        // Test in-place operation (output aliasing with high)
        wasm.marketefi_into(
            highPtr,
            lowPtr,
            volumePtr,
            highPtr,  // Output to high buffer (aliasing)
            len
        );
        
        // Read result (recreate view in case memory grew)
        const result = new Float64Array(wasm.__wasm.memory.buffer, highPtr, len);
        
        // Expected values: (high - low) / volume
        assertClose(result[0], 1.0, 1e-8);
        assertClose(result[1], 0.5, 1e-8);
        assertClose(result[2], 1.0, 1e-8);
    } finally {
        wasm.marketefi_free(highPtr, len);
        wasm.marketefi_free(lowPtr, len);
        wasm.marketefi_free(volumePtr, len);
    }
});

test('MarketEFI fast API (separate buffers)', () => {
    // Test the fast/unsafe API without aliasing
    const len = 1000;
    
    // Allocate buffers
    const highPtr = wasm.marketefi_alloc(len);
    const lowPtr = wasm.marketefi_alloc(len);
    const volumePtr = wasm.marketefi_alloc(len);
    const outPtr = wasm.marketefi_alloc(len);
    
    try {
        // Initialize data
        const highView = new Float64Array(wasm.__wasm.memory.buffer, highPtr, len);
        const lowView = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, len);
        const volumeView = new Float64Array(wasm.__wasm.memory.buffer, volumePtr, len);
        
        for (let i = 0; i < len; i++) {
            highView[i] = 100 + i * 0.1;
            lowView[i] = 99 + i * 0.1;
            volumeView[i] = 1000 + i;
        }
        
        wasm.marketefi_into(
            highPtr,
            lowPtr,
            volumePtr,
            outPtr,
            len
        );
        
        // Read result (recreate view in case memory grew)
        const result = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);
        
        // Compare with safe API using the same data
        const highArray = new Float64Array(highView);
        const lowArray = new Float64Array(lowView);
        const volumeArray = new Float64Array(volumeView);
        const expected = wasm.marketefi_js(highArray, lowArray, volumeArray);
        assertArrayClose(result, expected, 1e-10, "Fast API should match safe API");
    } finally {
        wasm.marketefi_free(highPtr, len);
        wasm.marketefi_free(lowPtr, len);
        wasm.marketefi_free(volumePtr, len);
        wasm.marketefi_free(outPtr, len);
    }
});