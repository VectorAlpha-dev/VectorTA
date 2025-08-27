/**
 * WASM binding tests for VWMA indicator.
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

test('VWMA partial params', () => {
    // Test with default parameters - mirrors check_vwma_partial_params
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    const result = wasm.vwma_js(close, volume, 20);
    assert.strictEqual(result.length, close.length);
    
    // Test with custom period
    const result_custom = wasm.vwma_js(close, volume, 10);
    assert.strictEqual(result_custom.length, close.length);
});

test('VWMA accuracy', async () => {
    // Test VWMA matches expected values from Rust tests - mirrors check_vwma_accuracy
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    const expected = EXPECTED_OUTPUTS.vwma;
    
    const result = wasm.vwma_js(
        close,
        volume,
        expected.defaultParams.period
    );
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-3,  // VWMA uses 1e-3 tolerance in Rust tests
        "VWMA last 5 values mismatch"
    );
    
    // Compare full output with Rust
    await compareWithRust('vwma', result, 'close', expected.defaultParams);
});

test('VWMA price volume mismatch', () => {
    // Test VWMA fails when price and volume lengths don't match
    const prices = new Float64Array([10.0, 20.0, 30.0, 40.0, 50.0]);
    const volumes = new Float64Array([100.0, 200.0, 300.0]);  // Shorter array
    
    assert.throws(() => {
        wasm.vwma_js(prices, volumes, 3);
    }, /Price and volume mismatch/);
});

test('VWMA invalid period', () => {
    // Test VWMA fails with invalid period
    const prices = new Float64Array([10.0, 20.0, 30.0]);
    const volumes = new Float64Array([100.0, 200.0, 300.0]);
    
    // Period = 0
    assert.throws(() => {
        wasm.vwma_js(prices, volumes, 0);
    }, /Invalid period/);
    
    // Period exceeds length
    assert.throws(() => {
        wasm.vwma_js(prices, volumes, 10);
    }, /Invalid period/);
});

test('VWMA all NaN values', () => {
    // Test VWMA fails when all values are NaN
    const prices = new Float64Array(10);
    const volumes = new Float64Array(10);
    prices.fill(NaN);
    volumes.fill(NaN);
    
    assert.throws(() => {
        wasm.vwma_js(prices, volumes, 5);
    }, /All/);
});

test('VWMA not enough valid data', () => {
    // Test VWMA fails with insufficient valid data
    // First 8 values are NaN, only 2 valid values for period=5
    const prices = new Float64Array([NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 10.0, 20.0]);
    const volumes = new Float64Array([NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 100.0, 200.0]);
    
    assert.throws(() => {
        wasm.vwma_js(prices, volumes, 5);
    }, /Not enough valid/);
});

test('VWMA with default candles', () => {
    // Test VWMA with default parameters - mirrors check_vwma_input_with_default_candles
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    // Default period is 20
    const result = wasm.vwma_js(close, volume, 20);
    assert.strictEqual(result.length, close.length);
});

test('VWMA candles plus prices', () => {
    // Test VWMA with custom prices - mirrors check_vwma_candles_plus_prices
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    // Use slightly modified prices
    const custom_prices = close.map(v => v * 1.001);
    
    const result = wasm.vwma_js(custom_prices, volume, 20);
    assert.strictEqual(result.length, custom_prices.length);
});

test('VWMA slice reinput', () => {
    // Test VWMA applied twice (re-input) - mirrors check_vwma_slice_data_reinput
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    // First pass
    const firstResult = wasm.vwma_js(close, volume, 20);
    assert.strictEqual(firstResult.length, close.length);
    
    // Second pass - use VWMA output as prices, keep same volumes
    const secondResult = wasm.vwma_js(firstResult, volume, 10);
    assert.strictEqual(secondResult.length, firstResult.length);
    
    // After warmup, should have valid values
    // First pass warmup: first + period - 1 = 0 + 20 - 1 = 19
    // Second pass warmup: first_warmup + period2 - 1 = 19 + 10 - 1 = 28
    const expectedWarmup = 28;
    for (let i = expectedWarmup; i < secondResult.length; i++) {
        assert(!isNaN(secondResult[i]), `Unexpected NaN at index ${i}`);
    }
});

test('VWMA NaN handling', () => {
    // Test VWMA handles NaN values correctly
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    const result = wasm.vwma_js(close, volume, 20);
    assert.strictEqual(result.length, close.length);
    
    // First period-1 values should be NaN (warmup)
    assertAllNaN(result.slice(0, 19), "Expected NaN in warmup period");
    
    // After warmup period, no NaN values should exist
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
});

test('VWMA batch single parameter set', () => {
    // Test batch with single parameter combination
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    // Single parameter set: period=20
    const batchResult = wasm.vwma_batch_js(
        close,
        volume,
        20, 20, 0  // period range
    );
    
    // Should match single calculation
    const singleResult = wasm.vwma_js(close, volume, 20);
    
    assert.strictEqual(batchResult.length, singleResult.length);
    assertArrayClose(batchResult, singleResult, 1e-10, "Batch vs single mismatch");
});

test('VWMA batch multiple periods', () => {
    // Test batch with multiple period values
    const close = new Float64Array(testData.close.slice(0, 100)); // Use smaller dataset for speed
    const volume = new Float64Array(testData.volume.slice(0, 100));
    
    // Multiple periods: 10, 15, 20
    const batchResult = wasm.vwma_batch_js(
        close,
        volume,
        10, 20, 5  // period range
    );
    
    // Should have 3 rows * 100 cols = 300 values
    assert.strictEqual(batchResult.length, 3 * 100);
    
    // Verify each row matches individual calculation
    const periods = [10, 15, 20];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.slice(rowStart, rowEnd);
        
        const singleResult = wasm.vwma_js(close, volume, periods[i]);
        assertArrayClose(
            rowData, 
            singleResult, 
            1e-10, 
            `Period ${periods[i]} mismatch`
        );
    }
});

test('VWMA batch metadata', () => {
    // Test metadata function returns correct parameter combinations
    const metadata = wasm.vwma_batch_metadata_js(
        10, 30, 5  // period: 10, 15, 20, 25, 30
    );
    
    // Should have 5 combinations
    assert.strictEqual(metadata.length, 5);
    
    // Check values
    assert.strictEqual(metadata[0], 10);
    assert.strictEqual(metadata[1], 15);
    assert.strictEqual(metadata[2], 20);
    assert.strictEqual(metadata[3], 25);
    assert.strictEqual(metadata[4], 30);
});

test('VWMA batch edge cases', () => {
    // Test edge cases for batch processing
    const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const volume = new Float64Array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]);
    
    // Single value sweep
    const singleBatch = wasm.vwma_batch_js(
        close,
        volume,
        5, 5, 1
    );
    
    assert.strictEqual(singleBatch.length, 10);
    
    // Step larger than range
    const largeBatch = wasm.vwma_batch_js(
        close,
        volume,
        5, 7, 10  // Step larger than range
    );
    
    // Should only have period=5
    assert.strictEqual(largeBatch.length, 10);
    
    // Empty data should throw
    assert.throws(() => {
        wasm.vwma_batch_js(
            new Float64Array([]),
            new Float64Array([]),
            10, 10, 0
        );
    }, /All/);
});

// Note: Streaming tests would require streaming functions to be exposed in WASM bindings

test('VWMA fast/unsafe API basic', () => {
    // Test fast API with separate output buffer
    const close = new Float64Array(testData.close.slice(0, 100));
    const volume = new Float64Array(testData.volume.slice(0, 100));
    
    // Allocate buffers in WASM memory
    const closePtr = wasm.vwma_alloc(100);
    const volumePtr = wasm.vwma_alloc(100);
    const outPtr = wasm.vwma_alloc(100);
    
    try {
        // Get WASM memory reference (check both possible locations)
        const memory = wasm.__wasm?.memory || wasm.memory;
        if (!memory) {
            throw new Error('WASM memory not accessible');
        }
        
        // Copy input data into WASM memory
        const closeView = new Float64Array(memory.buffer, closePtr, 100);
        const volumeView = new Float64Array(memory.buffer, volumePtr, 100);
        closeView.set(close);
        volumeView.set(volume);
        
        // Call fast API with WASM pointers
        wasm.vwma_into(
            closePtr,
            volumePtr,
            outPtr,
            100,
            20  // period
        );
        
        // Read results from output buffer
        const result = new Float64Array(memory.buffer, outPtr, 100);
        const resultCopy = new Float64Array(result); // Copy before freeing
        
        // Compare with safe API
        const safeResult = wasm.vwma_js(close, volume, 20);
        assertArrayClose(resultCopy, safeResult, 1e-10, "Fast vs safe API mismatch");
    } finally {
        // Clean up
        wasm.vwma_free(closePtr, 100);
        wasm.vwma_free(volumePtr, 100);
        wasm.vwma_free(outPtr, 100);
    }
});

test('VWMA fast API in-place operation (aliasing)', () => {
    // Test fast API with output aliasing price input
    const prices = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const volumes = new Float64Array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]);
    
    // Make a copy for comparison
    const pricesCopy = new Float64Array(prices);
    
    // Allocate buffers in WASM memory
    const pricePtr = wasm.vwma_alloc(10);
    const volumePtr = wasm.vwma_alloc(10);
    
    try {
        // Get WASM memory reference
        const memory = wasm.__wasm?.memory || wasm.memory;
        if (!memory) {
            throw new Error('WASM memory not accessible');
        }
        
        // Copy input data into WASM memory
        const priceView = new Float64Array(memory.buffer, pricePtr, 10);
        const volumeView = new Float64Array(memory.buffer, volumePtr, 10);
        priceView.set(prices);
        volumeView.set(volumes);
        
        // Use price buffer as output (in-place)
        wasm.vwma_into(
            pricePtr,
            volumePtr,
            pricePtr,  // Output to same as price input (aliasing)
            10,
            3  // period
        );
        
        // Read results from price buffer (which was modified in-place)
        const result = new Float64Array(memory.buffer, pricePtr, 10);
        const resultCopy = new Float64Array(result); // Copy before comparison
        
        // Verify with safe API
        const expected = wasm.vwma_js(pricesCopy, volumes, 3);
        assertArrayClose(resultCopy, expected, 1e-10, "In-place operation mismatch");
    } finally {
        // Clean up
        wasm.vwma_free(pricePtr, 10);
        wasm.vwma_free(volumePtr, 10);
    }
});

test('VWMA fast API null pointer handling', () => {
    // Test null pointer error handling
    assert.throws(() => {
        wasm.vwma_into(0, 0, 0, 10, 5);
    }, /Null pointer/);
});

test('VWMA memory allocation and deallocation', () => {
    // Test alloc/free functions
    const sizes = [10, 100, 1000];
    
    // Get WASM memory reference
    const memory = wasm.__wasm?.memory || wasm.memory;
    if (!memory) {
        throw new Error('WASM memory not accessible');
    }
    
    for (const size of sizes) {
        const ptr = wasm.vwma_alloc(size);
        assert(ptr !== 0, `Failed to allocate ${size} elements`);
        
        // Write some data
        const arr = new Float64Array(memory.buffer, ptr, size);
        for (let i = 0; i < size; i++) {
            arr[i] = i;
        }
        
        // Verify data
        for (let i = 0; i < size; i++) {
            assert.strictEqual(arr[i], i);
        }
        
        // Free memory
        wasm.vwma_free(ptr, size);
    }
});

test('VWMA batch unified API with serde config', () => {
    // Test new unified batch API with configuration object
    const close = new Float64Array(testData.close.slice(0, 50));
    const volume = new Float64Array(testData.volume.slice(0, 50));
    
    const config = {
        period_range: [10, 20, 5]  // [start, end, step]
    };
    
    const result = wasm.vwma_batch(close, volume, config);
    
    // Verify result structure
    assert(result.values instanceof Array, "Expected values array");
    assert(result.combos instanceof Array, "Expected combos array");
    assert.strictEqual(result.rows, 3, "Expected 3 rows (periods: 10, 15, 20)");
    assert.strictEqual(result.cols, 50, "Expected 50 columns");
    assert.strictEqual(result.values.length, 150, "Expected 150 total values");
    
    // Verify combos
    assert.strictEqual(result.combos.length, 3);
    assert.strictEqual(result.combos[0].period, 10);
    assert.strictEqual(result.combos[1].period, 15);
    assert.strictEqual(result.combos[2].period, 20);
});

test('VWMA batch fast API', () => {
    // Test batch fast API
    const close = new Float64Array(testData.close.slice(0, 30));
    const volume = new Float64Array(testData.volume.slice(0, 30));
    
    // Get WASM memory reference
    const memory = wasm.__wasm?.memory || wasm.memory;
    if (!memory) {
        throw new Error('WASM memory not accessible');
    }
    
    // Allocate input buffers in WASM memory
    const closePtr = wasm.vwma_alloc(30);
    const volumePtr = wasm.vwma_alloc(30);
    
    // Allocate output for 2 periods × 30 values = 60 total
    const outPtr = wasm.vwma_alloc(60);
    
    try {
        // Copy input data into WASM memory
        const closeView = new Float64Array(memory.buffer, closePtr, 30);
        const volumeView = new Float64Array(memory.buffer, volumePtr, 30);
        closeView.set(close);
        volumeView.set(volume);
        
        const rows = wasm.vwma_batch_into(
            closePtr,
            volumePtr,
            outPtr,
            30,
            10, 15, 5  // periods: 10, 15
        );
        
        assert.strictEqual(rows, 2, "Expected 2 rows");
        
        // Read results
        const result = new Float64Array(memory.buffer, outPtr, 60);
        
        // Verify against individual calculations
        const expected1 = wasm.vwma_js(close, volume, 10);
        const expected2 = wasm.vwma_js(close, volume, 15);
        
        assertArrayClose(
            result.slice(0, 30), 
            expected1, 
            1e-10, 
            "Batch row 1 mismatch"
        );
        assertArrayClose(
            result.slice(30, 60), 
            expected2, 
            1e-10, 
            "Batch row 2 mismatch"
        );
    } finally {
        wasm.vwma_free(closePtr, 30);
        wasm.vwma_free(volumePtr, 30);
        wasm.vwma_free(outPtr, 60);
    }
});

test('VWMA batch fast API with aliasing', () => {
    // Test batch fast API with aliasing - single period so output size equals input size
    const prices = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const volumes = new Float64Array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]);
    
    // Get WASM memory reference
    const memory = wasm.__wasm?.memory || wasm.memory;
    if (!memory) {
        throw new Error('WASM memory not accessible');
    }
    
    // Allocate input buffers in WASM memory
    const pricePtr = wasm.vwma_alloc(10);
    const volumePtr = wasm.vwma_alloc(10);
    
    try {
        // Copy input data into WASM memory
        const priceView = new Float64Array(memory.buffer, pricePtr, 10);
        const volumeView = new Float64Array(memory.buffer, volumePtr, 10);
        priceView.set(prices);
        volumeView.set(volumes);
        
        // Make copies for verification
        const pricesCopy = new Float64Array(prices);
        const volumesCopy = new Float64Array(volumes);
        
        // Test with output buffer aliasing price buffer
        // Single period so output size = input size
        const rows = wasm.vwma_batch_into(
            pricePtr,
            volumePtr,
            pricePtr,  // Output aliases with prices
            10,
            3, 3, 1  // single period: 3
        );
        
        assert.strictEqual(rows, 1, "Expected 1 row");
        
        // Read results from price buffer (now contains output)
        const result = new Float64Array(memory.buffer, pricePtr, 10);
        const resultCopy = new Float64Array(result); // Copy for comparison
        
        // Verify against expected
        const expected = wasm.vwma_js(pricesCopy, volumesCopy, 3);
        assertArrayClose(
            resultCopy, 
            expected, 
            1e-10, 
            "Batch aliasing mismatch"
        );
    } finally {
        wasm.vwma_free(pricePtr, 10);
        wasm.vwma_free(volumePtr, 10);
    }
});

test('VWMA zero volume', () => {
    // Test VWMA handles zero volume correctly
    const prices = new Float64Array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]);
    const volumes = new Float64Array([100.0, 0.0, 300.0, 0.0, 500.0, 0.0, 700.0, 0.0, 900.0, 0.0]);
    
    const result = wasm.vwma_js(prices, volumes, 3);
    assert.strictEqual(result.length, prices.length);
    
    // Check that we get NaN where all volumes in window are zero
    // but valid values where at least one volume is non-zero
    assert(!isNaN(result[2]), "Index 2 should have valid value with non-zero volumes in window");
});

test('VWMA partial NaN data', () => {
    // Test VWMA with NaN values in middle of dataset
    const close = new Float64Array(testData.close.slice(0, 200));
    const volume = new Float64Array(testData.volume.slice(0, 200));
    
    // Inject NaN values in middle of data
    for (let i = 100; i < 110; i++) {
        close[i] = NaN;
        volume[i] = NaN;
    }
    
    // VWMA should handle NaN gracefully but might propagate them
    // This test verifies the function doesn't crash
    const result = wasm.vwma_js(close, volume, 20);
    assert.strictEqual(result.length, close.length);
    
    // Should have NaN during warmup
    assertAllNaN(result.slice(0, 19), "Expected NaN in warmup period");
    
    // Note: NaN handling depends on implementation - some may continue
    // producing NaN until all data is valid again
});

test('VWMA warmup period verification', () => {
    // Test VWMA warmup period calculation matches Rust exactly
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    const period = 20;
    
    const result = wasm.vwma_js(close, volume, period);
    
    // Warmup should be first + period - 1 = 0 + 20 - 1 = 19
    // So indices 0-18 should be NaN, index 19 should be first valid
    assertAllNaN(result.slice(0, 19), "First 19 values should be NaN");
    assert(!isNaN(result[19]), "Index 19 should be first valid value");
});

test('VWMA batch multiple parameter sweeps', () => {
    // Test batch with multiple period values
    const close = new Float64Array(testData.close.slice(0, 100));
    const volume = new Float64Array(testData.volume.slice(0, 100));
    
    // Test with periods: 10, 20, 30
    const batchResult = wasm.vwma_batch_js(
        close,
        volume,
        10, 30, 10  // period range
    );
    
    // Should have 3 rows * 100 cols = 300 values
    assert.strictEqual(batchResult.length, 3 * 100);
    
    // Verify each row matches individual calculation
    const periods = [10, 20, 30];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.slice(rowStart, rowEnd);
        
        const singleResult = wasm.vwma_js(close, volume, periods[i]);
        assertArrayClose(
            rowData, 
            singleResult, 
            1e-10, 
            `Period ${periods[i]} mismatch in batch`
        );
    }
});

test.after(() => {
    console.log('VWMA WASM tests completed');
});
