/**
 * WASM binding tests for Aroon Oscillator indicator.
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

test('AroonOsc partial params', () => {
    // Test with partial parameters - mirrors check_aroonosc_partial_params
    const highArray = new Float64Array(testData.high);
    const lowArray = new Float64Array(testData.low);
    
    // Test with length=20 (overriding default of 14)
    const result = wasm.aroonosc_js(highArray, lowArray, 20);
    
    assert.strictEqual(result.length, highArray.length);
});

test('AroonOsc accuracy', async () => {
    // Test Aroon Oscillator matches expected values from Rust tests - mirrors check_aroonosc_accuracy
    const highArray = new Float64Array(testData.high);
    const lowArray = new Float64Array(testData.low);
    
    // Expected values from Rust tests
    const expectedLastFive = [-50.0, -50.0, -50.0, -50.0, -42.8571];
    
    const result = wasm.aroonosc_js(highArray, lowArray, 14); // Default length
    
    assert(result.length >= 5, "Not enough Aroon Osc values");
    assert.strictEqual(result.length, highArray.length);
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    
    assertArrayClose(
        last5,
        expectedLastFive,
        0.01, // Using 1e-2 tolerance as in Rust tests
        "Aroon Osc last 5 values mismatch"
    );
});

test('AroonOsc default params', () => {
    // Test Aroon Oscillator with default parameters
    const highArray = new Float64Array(testData.high);
    const lowArray = new Float64Array(testData.low);
    
    const result = wasm.aroonosc_js(highArray, lowArray, 14);
    assert.strictEqual(result.length, highArray.length);
});

test('AroonOsc zero length', () => {
    // Test Aroon Oscillator fails with zero length
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0, 25.0]);
    
    assert.throws(() => {
        wasm.aroonosc_js(high, low, 0);
    }, /Invalid length/);
});

test('AroonOsc length exceeds data', () => {
    // Test Aroon Oscillator fails when length exceeds data length
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0, 25.0]);
    
    assert.throws(() => {
        wasm.aroonosc_js(high, low, 10);
    }, /Not enough data/);
});

test('AroonOsc mismatched arrays', () => {
    // Test Aroon Oscillator fails with mismatched high/low arrays
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0]); // Different length
    
    assert.throws(() => {
        wasm.aroonosc_js(high, low, 14);
    }, /High and low arrays must have same length/);
});

test('AroonOsc NaN handling', () => {
    // Test Aroon Oscillator handles NaN values correctly - mirrors check_aroonosc_nan_handling
    const highArray = new Float64Array(testData.high);
    const lowArray = new Float64Array(testData.low);
    
    const result = wasm.aroonosc_js(highArray, lowArray, 14);
    
    // After warmup period (50), no NaN values should exist
    if (result.length > 50) {
        for (let i = 50; i < result.length; i++) {
            assert(!isNaN(result[i]), `Expected no NaN after index ${i}, but found NaN`);
        }
    }
});

test('AroonOsc reinput', () => {
    // Test Aroon Oscillator with reinput of its own output - mirrors Rust test
    const highArray = new Float64Array(testData.high);
    const lowArray = new Float64Array(testData.low);
    
    // First pass
    const firstResult = wasm.aroonosc_js(highArray, lowArray, 10);
    const firstResultArray = new Float64Array(firstResult);
    
    // Second pass - use first result as both high and low
    const secondResult = wasm.aroonosc_js(firstResultArray, firstResultArray, 5);
    assert.strictEqual(secondResult.length, firstResult.length);
    
    // Check that we get valid values after warmup
    for (let i = 20; i < secondResult.length; i++) {
        assert(!isNaN(secondResult[i]), `Found unexpected NaN at index ${i}`);
    }
});

test('AroonOsc batch single parameter set', () => {
    // Test batch with single parameter combination
    const highArray = new Float64Array(testData.high);
    const lowArray = new Float64Array(testData.low);
    
    // Single parameter set
    const batchResult = wasm.aroonosc_batch_js(
        highArray,
        lowArray,
        14, 14, 0  // length range (single value)
    );
    
    // Should match single calculation
    const singleResult = wasm.aroonosc_js(highArray, lowArray, 14);
    
    // Batch result should be same as single result
    assert.strictEqual(batchResult.length, singleResult.length);
    
    assertArrayClose(batchResult, singleResult, 1e-10, "Batch vs single mismatch");
});

test('AroonOsc batch multiple lengths', () => {
    // Test batch with multiple length values
    const high = testData.high.slice(0, 100);
    const low = testData.low.slice(0, 100);
    const highArray = new Float64Array(high);
    const lowArray = new Float64Array(low);
    
    // Multiple lengths: 10, 14, 18
    const batchResult = wasm.aroonosc_batch_js(
        highArray,
        lowArray,
        10, 18, 4  // length range
    );
    
    // Should have 3 rows * 100 cols = 300 values
    assert.strictEqual(batchResult.length, 3 * 100);
    
    // Verify each row matches individual calculation
    const lengths = [10, 14, 18];
    
    for (let i = 0; i < lengths.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const row = batchResult.slice(rowStart, rowEnd);
        
        const singleResult = wasm.aroonosc_js(highArray, lowArray, lengths[i]);
        
        assertArrayClose(
            row, 
            singleResult, 
            1e-10, 
            `Length ${lengths[i]} mismatch`
        );
    }
});

test('AroonOsc batch metadata', () => {
    // Test metadata function returns correct parameter combinations
    const metadata = wasm.aroonosc_batch_metadata_js(
        10, 20, 2  // length: 10, 12, 14, 16, 18, 20
    );
    
    // Should have 6 values (one per length)
    assert.strictEqual(metadata.length, 6);
    
    // Check values
    const expected = [10, 12, 14, 16, 18, 20];
    for (let i = 0; i < expected.length; i++) {
        assert.strictEqual(metadata[i], expected[i]);
    }
});

test('AroonOsc batch full parameter sweep', () => {
    // Test full parameter sweep matching expected structure
    const high = testData.high.slice(0, 50);
    const low = testData.low.slice(0, 50);
    const highArray = new Float64Array(high);
    const lowArray = new Float64Array(low);
    
    const batchResult = wasm.aroonosc_batch_js(
        highArray,
        lowArray,
        10, 20, 5  // 3 length values: 10, 15, 20
    );
    
    const metadata = wasm.aroonosc_batch_metadata_js(10, 20, 5);
    
    // Should have 3 combinations
    assert.strictEqual(metadata.length, 3);
    assert.strictEqual(batchResult.length, 3 * 50); // 3 combos * 50 data points
});

test('AroonOsc batch edge cases', () => {
    // Test edge cases for batch processing
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    
    // Single value sweep
    const singleBatch = wasm.aroonosc_batch_js(
        data,
        data,
        5, 5, 1
    );
    
    assert.strictEqual(singleBatch.length, 15); // 1 combo * 15 data points
    
    // Empty data should throw
    assert.throws(() => {
        wasm.aroonosc_batch_js(
            new Float64Array([]),
            new Float64Array([]),
            10, 20, 2
        );
    });
});

// New API tests
test('AroonOsc batch - new ergonomic API with single parameter', () => {
    // Test the new API with a single parameter combination
    const highArray = new Float64Array(testData.high);
    const lowArray = new Float64Array(testData.low);
    
    const result = wasm.aroonosc_batch(highArray, lowArray, {
        length_range: [14, 14, 0]
    });
    
    // Verify structure
    assert(result.values, 'Should have values array');
    assert(result.combos, 'Should have combos array');
    assert(typeof result.rows === 'number', 'Should have rows count');
    assert(typeof result.cols === 'number', 'Should have cols count');
    
    // Verify dimensions
    assert.strictEqual(result.rows, 1);
    assert.strictEqual(result.cols, highArray.length);
    assert.strictEqual(result.combos.length, 1);
    assert.strictEqual(result.values.length, highArray.length);
    
    // Verify parameters
    const combo = result.combos[0];
    assert.strictEqual(combo.length, 14);
    
    // Compare with old API
    const oldResult = wasm.aroonosc_js(highArray, lowArray, 14);
    
    assertArrayClose(result.values, oldResult, 1e-10, "New vs old API mismatch");
});

test('AroonOsc batch - new API with multiple parameters', () => {
    // Test with multiple parameter combinations
    const high = testData.high.slice(0, 50);
    const low = testData.low.slice(0, 50);
    const highArray = new Float64Array(high);
    const lowArray = new Float64Array(low);
    
    const result = wasm.aroonosc_batch(highArray, lowArray, {
        length_range: [10, 20, 5]  // 10, 15, 20
    });
    
    // Should have 3 combinations
    assert.strictEqual(result.rows, 3);
    assert.strictEqual(result.cols, 50);
    assert.strictEqual(result.combos.length, 3);
    assert.strictEqual(result.values.length, 150); // 3 rows * 50 cols
    
    // Verify each combo
    const expectedLengths = [10, 15, 20];
    
    for (let i = 0; i < expectedLengths.length; i++) {
        assert.strictEqual(result.combos[i].length, expectedLengths[i]);
    }
    
    // Verify values match old API
    for (let i = 0; i < result.rows; i++) {
        const rowStart = i * result.cols;
        const rowEnd = rowStart + result.cols;
        const row = result.values.slice(rowStart, rowEnd);
        
        const oldResult = wasm.aroonosc_js(highArray, lowArray, result.combos[i].length);
        
        assertArrayClose(row, oldResult, 1e-10, `Row ${i} mismatch`);
    }
});

test('AroonOsc batch - new API error handling', () => {
    const highArray = new Float64Array(testData.high.slice(0, 10));
    const lowArray = new Float64Array(testData.low.slice(0, 10));
    
    // Invalid config structure
    assert.throws(() => {
        wasm.aroonosc_batch(highArray, lowArray, {
            length_range: [10, 20] // Missing step
        });
    }, /Invalid config/);
    
    // Missing required field
    assert.throws(() => {
        wasm.aroonosc_batch(highArray, lowArray, {
            // Missing length_range
        });
    }, /Invalid config/);
    
    // Invalid data type
    assert.throws(() => {
        wasm.aroonosc_batch(highArray, lowArray, {
            length_range: "invalid"
        });
    }, /Invalid config/);
});

test.after(() => {
    console.log('AroonOsc WASM tests completed');
});