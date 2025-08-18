/**
 * WASM binding tests for MINMAX indicator.
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

test('MINMAX partial params', () => {
    const high = testData.high;
    const low = testData.low;
    
    // Test with default order (3)
    const result = wasm.minmax_js(high, low, 3);
    assert.strictEqual(result.is_min.length, high.length, 'is_min length mismatch');
    assert.strictEqual(result.is_max.length, high.length, 'is_max length mismatch');
    assert.strictEqual(result.last_min.length, high.length, 'last_min length mismatch');
    assert.strictEqual(result.last_max.length, high.length, 'last_max length mismatch');
});

test('MINMAX accuracy', () => {
    const high = testData.high;
    const low = testData.low;
    const order = 3;
    
    const result = wasm.minmax_js(high, low, order);
    assert.strictEqual(result.is_min.length, high.length, 'is_min length mismatch');
    
    // Check last 5 values match expected
    const count = result.last_min.length;
    assert(count >= 5, 'Not enough data to check last 5');
    
    const expected_last_five_min = [57876.0, 57876.0, 57876.0, 57876.0, 57876.0];
    const expected_last_five_max = [60102.0, 60102.0, 60102.0, 60102.0, 60102.0];
    
    const last_min_slice = result.last_min.slice(-5);
    const last_max_slice = result.last_max.slice(-5);
    
    assertArrayClose(last_min_slice, expected_last_five_min, 0.1, 'MINMAX last_min mismatch');
    assertArrayClose(last_max_slice, expected_last_five_max, 0.1, 'MINMAX last_max mismatch');
});

test('MINMAX zero order', () => {
    const high = [10.0, 20.0, 30.0];
    const low = [1.0, 2.0, 3.0];
    
    assert.throws(() => {
        wasm.minmax_js(high, low, 0);
    }, /Invalid order/, 'Should throw error for zero order');
});

test('MINMAX order exceeds length', () => {
    const high = [10.0, 20.0, 30.0];
    const low = [1.0, 2.0, 3.0];
    
    assert.throws(() => {
        wasm.minmax_js(high, low, 10);
    }, /Invalid order/, 'Should throw error when order exceeds length');
});

test('MINMAX all NaN input', () => {
    const high = [NaN, NaN, NaN];
    const low = [NaN, NaN, NaN];
    
    assert.throws(() => {
        wasm.minmax_js(high, low, 1);
    }, /All values are NaN/, 'Should throw error for all NaN data');
});

test('MINMAX basic slices', () => {
    const high = [50.0, 55.0, 60.0, 55.0, 50.0, 45.0, 50.0, 55.0];
    const low = [40.0, 38.0, 35.0, 38.0, 40.0, 42.0, 41.0, 39.0];
    const order = 2;
    
    const result = wasm.minmax_js(high, low, order);
    assert.strictEqual(result.is_min.length, 8);
    assert.strictEqual(result.is_max.length, 8);
    assert.strictEqual(result.last_min.length, 8);
    assert.strictEqual(result.last_max.length, 8);
    
    // Verify some expected local extrema
    // Index 2: low[2]=35.0 should be a local minimum (lowest in neighborhood)
    // Index 2: high[2]=60.0 should be a local maximum (highest in neighborhood)
    assert(!isNaN(result.is_min[2]), 'Should have found a minimum at index 2');
    assert(!isNaN(result.is_max[2]), 'Should have found a maximum at index 2');
});

test('MINMAX batch processing', () => {
    const high = testData.high.slice(0, 100); // Use smaller dataset
    const low = testData.low.slice(0, 100);
    
    const config = {
        order_range: [2, 5, 1] // Orders: 2, 3, 4, 5
    };
    
    const result = wasm.minmax_batch(high, low, config);
    
    // Should have 4 combinations
    assert.strictEqual(result.rows, 4, 'Should have 4 parameter combinations');
    assert.strictEqual(result.is_min.length, 4, 'Should have 4 rows of is_min');
    assert.strictEqual(result.is_max.length, 4, 'Should have 4 rows of is_max');
    
    // Check first row matches single calculation
    const single_result = wasm.minmax_js(high, low, 2);
    assertArrayClose(result.is_min[0], single_result.is_min, 1e-9, 'Batch vs single mismatch for is_min');
    assertArrayClose(result.is_max[0], single_result.is_max, 1e-9, 'Batch vs single mismatch for is_max');
});

test('MINMAX fast API', () => {
    const high = testData.high.slice(0, 100);
    const low = testData.low.slice(0, 100);
    const len = high.length;
    const order = 3;
    
    // Allocate input and output buffers
    const high_ptr = wasm.minmax_alloc(len);
    const low_ptr = wasm.minmax_alloc(len);
    const is_min_ptr = wasm.minmax_alloc(len);
    const is_max_ptr = wasm.minmax_alloc(len);
    const last_min_ptr = wasm.minmax_alloc(len);
    const last_max_ptr = wasm.minmax_alloc(len);
    
    try {
        // Create views into WASM memory for inputs
        const highMemView = new Float64Array(wasm.__wasm.memory.buffer, high_ptr, len);
        const lowMemView = new Float64Array(wasm.__wasm.memory.buffer, low_ptr, len);
        
        // Copy input data into WASM memory
        highMemView.set(high);
        lowMemView.set(low);
        
        // Call fast API with pointers
        wasm.minmax_into(
            high_ptr,
            low_ptr,
            is_min_ptr,
            is_max_ptr,
            last_min_ptr,
            last_max_ptr,
            len,
            order
        );
        
        // Read results back (recreate views in case memory grew)
        const is_min_result = new Float64Array(wasm.__wasm.memory.buffer, is_min_ptr, len);
        const is_max_result = new Float64Array(wasm.__wasm.memory.buffer, is_max_ptr, len);
        const last_min_result = new Float64Array(wasm.__wasm.memory.buffer, last_min_ptr, len);
        const last_max_result = new Float64Array(wasm.__wasm.memory.buffer, last_max_ptr, len);
        
        // Compare with safe API
        const safe_result = wasm.minmax_js(high, low, order);
        assertArrayClose(Array.from(is_min_result), safe_result.is_min, 1e-9, 'Fast vs safe API mismatch for is_min');
        assertArrayClose(Array.from(is_max_result), safe_result.is_max, 1e-9, 'Fast vs safe API mismatch for is_max');
        assertArrayClose(Array.from(last_min_result), safe_result.last_min, 1e-9, 'Fast vs safe API mismatch for last_min');
        assertArrayClose(Array.from(last_max_result), safe_result.last_max, 1e-9, 'Fast vs safe API mismatch for last_max');
        
    } finally {
        // Clean up all allocated memory
        wasm.minmax_free(high_ptr, len);
        wasm.minmax_free(low_ptr, len);
        wasm.minmax_free(is_min_ptr, len);
        wasm.minmax_free(is_max_ptr, len);
        wasm.minmax_free(last_min_ptr, len);
        wasm.minmax_free(last_max_ptr, len);
    }
});

test.after(() => {
    console.log('MINMAX WASM tests completed');
});
