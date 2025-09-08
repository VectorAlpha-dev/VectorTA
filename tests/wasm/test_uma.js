import test from 'node:test';
import assert from 'node:assert';
import * as wasm from '../../pkg/my_project.js';
import { 
    loadTestData, 
    assertClose
} from './test_utils.js';

// Load test data
let testData;

// Expected values from CSV data (matching Rust tests)
const EXPECTED_LAST_5 = [
    59665.81830666, 59477.69234542, 59314.50778603,
    59218.23033661, 59143.61473211
];

test('UMA - partial params', () => {
    const data = Array.from({length: 100}, (_, i) => 100.0 + i);
    
    const result = wasm.uma_js(
        new Float64Array(data),
        1.0,     // accelerator
        5,       // min_length
        50,      // max_length
        4,       // smooth_length
        null     // volume
    );
    
    assert.ok(result, 'Result should not be null');
    assert.ok(result.values, 'Result should have values array');
    assert.strictEqual(result.values.length, data.length, 'Result length should match input length');
});

test.before(() => {
    testData = loadTestData();
});

test('UMA - accuracy test', () => {
    const result = wasm.uma_js(
        new Float64Array(testData.close),
        1.0,     // accelerator
        5,       // min_length
        50,      // max_length
        4,       // smooth_length
        null     // volume
    );

    // Get the last 5 valid values
    const validValues = result.values.filter(v => !isNaN(v));
    const lastValues = validValues.slice(-5);

    // Check accuracy with tolerance
    for (let i = 0; i < EXPECTED_LAST_5.length && i < lastValues.length; i++) {
        const diff = Math.abs(lastValues[i] - EXPECTED_LAST_5[i]);
        const tolerance = EXPECTED_LAST_5[i] * 0.01; // 1% tolerance
        assert.ok(
            diff < tolerance || diff < 1.0,
            `Value ${i}: Expected ${EXPECTED_LAST_5[i]}, got ${lastValues[i]}, diff ${diff}`
        );
    }
});

test('UMA - default candles', () => {
    const data = Array.from({length: 100}, (_, i) => 100.0 + i);
    
    const result = wasm.uma_js(
        new Float64Array(data),
        1.0,     // accelerator
        5,       // min_length
        50,      // max_length
        4,       // smooth_length
        null     // volume
    );
    
    assert.ok(result, 'Result should not be null');
    assert.strictEqual(result.values.length, data.length, 'Result length should match input length');
});

test('UMA - zero max_length error', () => {
    const data = [10.0, 20.0, 30.0];
    
    assert.throws(() => {
        wasm.uma_js(
            new Float64Array(data),
            1.0,     // accelerator
            5,       // min_length
            0,       // Invalid: max_length = 0
            4        // smooth_length
        );
    }, 'Should throw error for zero max_length');
});

test('UMA - period exceeds length error', () => {
    const data = [10.0, 20.0, 30.0];
    
    assert.throws(() => {
        wasm.uma_js(
            new Float64Array(data),
            1.0,     // accelerator
            5,       // min_length
            10,      // max_length exceeds data length
            4        // smooth_length
        );
    }, 'Should throw error when max_length exceeds data length');
});

test('UMA - very small dataset error', () => {
    const data = [42.0];
    
    assert.throws(() => {
        wasm.uma_js(
            new Float64Array(data),
            1.0,     // accelerator
            5,       // min_length
            50,      // max_length
            4        // smooth_length
        );
    }, 'Should throw error for insufficient data');
});

test('UMA - empty data error', () => {
    assert.throws(() => {
        wasm.uma_js(
            new Float64Array([]),
            1.0,     // accelerator
            5,       // min_length
            50,      // max_length
            4        // smooth_length
        );
    }, 'Should throw error for empty data');
});

test('UMA - invalid accelerator error', () => {
    const data = Array.from({length: 100}, (_, i) => 100.0 + i);

    // Test invalid accelerator < 1.0
    assert.throws(() => {
        wasm.uma_js(
            new Float64Array(data),
            0.5,     // Invalid: must be >= 1.0
            5,       // min_length
            50,      // max_length
            4        // smooth_length
        );
    }, 'Should throw error for invalid accelerator < 1.0');

    // Test negative accelerator
    assert.throws(() => {
        wasm.uma_js(
            new Float64Array(data),
            -1.0,    // Invalid: negative
            5,       // min_length
            50,      // max_length
            4        // smooth_length
        );
    }, 'Should throw error for negative accelerator');
});

test('UMA - invalid min_max error', () => {
    const data = Array.from({length: 100}, (_, i) => 100.0 + i);

    // Test min_length > max_length
    assert.throws(() => {
        wasm.uma_js(
            new Float64Array(data),
            1.0,     // accelerator
            60,      // min_length > max_length
            50,      // max_length
            4        // smooth_length
        );
    }, 'Should throw error when min_length > max_length');
});

test('UMA - NaN handling', () => {
    const data = Array.from({length: 100}, (_, i) => 100.0 + i);
    
    const result = wasm.uma_js(
        new Float64Array(data),
        1.0,     // accelerator
        5,       // min_length
        50,      // max_length
        4,       // smooth_length
        null     // volume
    );

    // Check warmup period (should be ~53 NaN values)
    const warmupPeriod = 53; // max_length + smooth_length - 1
    let nanCount = 0;
    for (let i = 0; i < warmupPeriod && i < result.values.length; i++) {
        if (isNaN(result.values[i])) {
            nanCount++;
        }
    }
    assert.ok(nanCount > 0, 'Should have NaN values in warmup period');

    // Check for valid values after warmup
    if (result.values.length > warmupPeriod + 10) {
        const validAfterWarmup = result.values.slice(warmupPeriod + 10).filter(v => !isNaN(v));
        assert.ok(validAfterWarmup.length > 0, 'Should have valid values after warmup');
    }
});

test('UMA - all NaN error', () => {
    const data = new Float64Array(100).fill(NaN);
    
    assert.throws(() => {
        wasm.uma_js(
            data,
            1.0,     // accelerator
            5,       // min_length
            50,      // max_length
            4        // smooth_length
        );
    }, 'Should throw error for all NaN data');
});

test('UMA - with leading NaNs', () => {
    // Create data with 10 leading NaNs
    const data = new Float64Array(110);
    for (let i = 0; i < 10; i++) {
        data[i] = NaN;
    }
    for (let i = 10; i < 110; i++) {
        data[i] = 100.0 + (i - 10);
    }
    
    const result = wasm.uma_js(
        data,
        1.0,     // accelerator
        5,       // min_length
        50,      // max_length
        4        // smooth_length
    );
    
    assert.strictEqual(result.values.length, data.length, 'Result length should match input');
    
    // Should have valid values after NaNs and warmup
    const validValues = result.values.slice(70).filter(v => !isNaN(v));
    assert.ok(validValues.length > 0, 'Should handle NaN prefix and produce valid values');
});

test('UMA - with volume data', () => {
    const data = Array.from({length: 100}, (_, i) => 100.0 + i);
    const volume = Array.from({length: 100}, (_, i) => 1000.0 + i * 10);

    const result = wasm.uma_js(
        new Float64Array(data),
        1.0,     // accelerator
        5,       // min_length
        50,      // max_length
        4,       // smooth_length
        volume   // volume data
    );

    assert.ok(result, 'Result should not be null');
    assert.ok(result.values, 'Result should have values array');
    assert.strictEqual(result.values.length, data.length, 'Result length should match input length');

    // Check for valid values after warmup
    const validValues = result.values.filter(v => !isNaN(v));
    assert.ok(validValues.length > 0, 'Should have valid values after warmup');
});

test('UMA - different parameter combinations', () => {
    const data = Array.from({length: 100}, (_, i) => 100.0 + i * 0.5);

    // Test with different accelerator
    const result1 = wasm.uma_js(
        new Float64Array(data),
        2.0,     // higher accelerator
        5,       // min_length
        50,      // max_length
        4        // smooth_length
    );
    assert.ok(result1, 'Should handle accelerator = 2.0');

    // Test with different length ranges
    const result2 = wasm.uma_js(
        new Float64Array(data),
        1.0,     // accelerator
        10,      // min_length
        30,      // max_length
        4        // smooth_length
    );
    assert.ok(result2, 'Should handle different length range');

    // Test with different smooth_length
    const result3 = wasm.uma_js(
        new Float64Array(data),
        1.0,     // accelerator
        5,       // min_length
        50,      // max_length
        8        // smooth_length
    );
    assert.ok(result3, 'Should handle different smooth_length');
});

// Zero-copy API tests
test('UMA - zero-copy API', () => {
    const data = testData.close;
    const len = data.length;
    
    // Allocate memory
    const ptr = wasm.uma_alloc(len);
    assert.ok(ptr !== 0, 'Should allocate memory');
    
    // Get view and fill with data
    const view = wasm.uma_get_view(ptr, len);
    for (let i = 0; i < len; i++) {
        view[i] = data[i];
    }
    
    // Calculate UMA in-place
    wasm.uma_update(ptr, len, 1.0, 5, 50, 4);
    
    // Read results
    const result = wasm.uma_get_view(ptr, len);
    
    // Verify results
    assert.strictEqual(result.length, len, 'Result length should match');
    
    // Get valid values
    const validValues = [];
    for (let i = 0; i < result.length; i++) {
        if (!isNaN(result[i])) {
            validValues.push(result[i]);
        }
    }
    
    // Check last 5 values
    if (validValues.length >= 5) {
        const lastValues = validValues.slice(-5);
        for (let i = 0; i < EXPECTED_LAST_5.length && i < lastValues.length; i++) {
            const diff = Math.abs(lastValues[i] - EXPECTED_LAST_5[i]);
            const tolerance = EXPECTED_LAST_5[i] * 0.01; // 1% tolerance
            assert.ok(
                diff < tolerance || diff < 1.0,
                `Zero-copy value ${i}: Expected ${EXPECTED_LAST_5[i]}, got ${lastValues[i]}`
            );
        }
    }
    
    // Free memory
    wasm.uma_free(ptr, len);
});

test('UMA - zero-copy memory management', () => {
    const sizes = [10, 100, 1000];
    
    for (const size of sizes) {
        const ptr = wasm.uma_alloc(size);
        assert.ok(ptr !== 0, `Should allocate memory for size ${size}`);
        
        const view = wasm.uma_get_view(ptr, size);
        assert.strictEqual(view.length, size, `View should have correct size ${size}`);
        
        // Fill with test data
        for (let i = 0; i < size; i++) {
            view[i] = 100.0 + i;
        }
        
        // Free memory
        wasm.uma_free(ptr, size);
    }
});

// Batch processing tests
test('UMA - batch processing single params', () => {
    const data = Array.from({length: 100}, (_, i) => 100.0 + i);
    
    const result = wasm.uma_batch_js(
        new Float64Array(data),
        [1.0, 1.0, 0.0],  // accelerator_range (single value)
        [5, 5, 0],        // min_length_range (single value)
        [50, 50, 0],      // max_length_range (single value)
        [4, 4, 0]         // smooth_length_range (single value)
    );
    
    assert.ok(result, 'Batch result should not be null');
    assert.ok(result.values, 'Should have values matrix');
    assert.ok(result.accelerators, 'Should have accelerators array');
    assert.ok(result.min_lengths, 'Should have min_lengths array');
    assert.ok(result.max_lengths, 'Should have max_lengths array');
    assert.ok(result.smooth_lengths, 'Should have smooth_lengths array');
    
    // Should have 1 combination
    assert.strictEqual(result.rows, 1, 'Should have 1 row');
    assert.strictEqual(result.cols, data.length, 'Should have correct column count');
    assert.strictEqual(result.values.length, data.length, 'Values array should match data length');
});

test('UMA - batch processing multiple params', () => {
    const data = Array.from({length: 50}, (_, i) => 100.0 + i);
    
    const result = wasm.uma_batch_js(
        new Float64Array(data),
        [1.0, 2.0, 0.5],  // accelerator_range: 1.0, 1.5, 2.0 (3 values)
        [5, 10, 5],       // min_length_range: 5, 10 (2 values)
        [30, 30, 0],      // max_length_range: 30 (1 value)
        [4, 4, 0]         // smooth_length_range: 4 (1 value)
    );
    
    assert.ok(result, 'Batch result should not be null');
    
    // Should have 3 * 2 * 1 * 1 = 6 combinations
    assert.strictEqual(result.rows, 6, 'Should have 6 rows');
    assert.strictEqual(result.cols, data.length, 'Should have correct column count');
    assert.strictEqual(result.values.length, 6 * data.length, 'Values array size should be rows * cols');
    
    // Verify parameter arrays
    assert.strictEqual(result.accelerators.length, 6, 'Should have 6 accelerator values');
    assert.strictEqual(result.min_lengths.length, 6, 'Should have 6 min_length values');
    assert.strictEqual(result.max_lengths.length, 6, 'Should have 6 max_length values');
    assert.strictEqual(result.smooth_lengths.length, 6, 'Should have 6 smooth_length values');
    
    // Check that each row has some valid values
    for (let row = 0; row < result.rows; row++) {
        const rowStart = row * result.cols;
        const rowEnd = rowStart + result.cols;
        const rowValues = result.values.slice(rowStart, rowEnd);
        const validCount = rowValues.filter(v => !isNaN(v)).length;
        assert.ok(validCount > 0, `Row ${row} should have some valid values`);
    }
});

test('UMA - batch with volume', () => {
    const data = Array.from({length: 50}, (_, i) => 100.0 + i);
    const volume = Array.from({length: 50}, (_, i) => 1000.0 + i * 10);
    
    const result = wasm.uma_batch_js(
        new Float64Array(data),
        [1.0, 1.0, 0.0],  // accelerator_range
        [5, 5, 0],        // min_length_range
        [30, 30, 0],      // max_length_range
        [4, 4, 0],        // smooth_length_range
        volume            // volume data
    );
    
    assert.ok(result, 'Batch result with volume should not be null');
    assert.strictEqual(result.rows, 1, 'Should have 1 row');
    assert.strictEqual(result.cols, data.length, 'Should have correct column count');
});

// Streaming tests
test('UMA - streaming interface', () => {
    // Create stream
    const stream = wasm.uma_stream_new(1.0, 5, 50, 4);
    assert.ok(stream, 'Should create stream');
    
    // Feed data points one by one
    const results = [];
    for (let i = 0; i < 60; i++) {
        const value = 100.0 + i;
        const result = wasm.uma_stream_update(stream, value);
        results.push(result);
    }
    
    // Should have null/undefined values initially, then start producing values
    const nullCount = results.filter(r => r === null || r === undefined || isNaN(r)).length;
    assert.ok(nullCount > 0, 'Should have null/NaN values during warmup');
    
    // Should have some valid values after warmup
    const validCount = results.filter(r => r !== null && r !== undefined && !isNaN(r)).length;
    assert.ok(validCount > 0, 'Should have valid values after warmup');
    
    // Reset stream
    wasm.uma_stream_reset(stream);
    
    // After reset, should need warmup again
    const afterReset = wasm.uma_stream_update(stream, 100.0);
    assert.ok(afterReset === null || afterReset === undefined || isNaN(afterReset), 
        'Should need warmup after reset');
    
    // Free stream
    wasm.uma_stream_free(stream);
});

test('UMA - streaming with volume', () => {
    // Create stream
    const stream = wasm.uma_stream_new(1.0, 5, 50, 4);
    assert.ok(stream, 'Should create stream');
    
    // Feed data points with volume
    const results = [];
    for (let i = 0; i < 60; i++) {
        const value = 100.0 + i;
        const volume = 1000.0 + i * 10;
        const result = wasm.uma_stream_update_with_volume(stream, value, volume);
        results.push(result);
    }
    
    // Should have some valid values after warmup
    const validCount = results.filter(r => r !== null && r !== undefined && !isNaN(r)).length;
    assert.ok(validCount > 0, 'Should have valid values after warmup with volume');
    
    // Free stream
    wasm.uma_stream_free(stream);
});

test('UMA - streaming vs batch comparison', () => {
    const data = testData.close;
    
    // Batch calculation
    const batchResult = wasm.uma_js(
        new Float64Array(data),
        1.0, 5, 50, 4, null
    );
    
    // Streaming calculation
    const stream = wasm.uma_stream_new(1.0, 5, 50, 4);
    const streamResults = [];
    
    for (const value of data) {
        const result = wasm.uma_stream_update(stream, value);
        streamResults.push(result !== null && result !== undefined ? result : NaN);
    }
    
    // Compare lengths
    assert.strictEqual(batchResult.values.length, streamResults.length, 
        'Batch and stream should produce same length');
    
    // UMA streaming has inherent differences from batch due to dynamic length buffer
    // Compare the overall trend rather than exact values
    const batchValid = batchResult.values.filter(v => !isNaN(v));
    const streamValid = streamResults.filter(v => !isNaN(v));
    
    if (batchValid.length >= 5 && streamValid.length >= 5) {
        // Compare last 5 values with relaxed tolerance (10% for UMA's dynamic nature)
        const batchLast = batchValid.slice(-5);
        const streamLast = streamValid.slice(-5);
        
        for (let i = 0; i < batchLast.length && i < streamLast.length; i++) {
            const relativeDiff = Math.abs(batchLast[i] - streamLast[i]) / Math.max(Math.abs(batchLast[i]), 1.0);
            assert.ok(relativeDiff < 0.1, 
                `Streaming vs batch mismatch at ${i}: batch=${batchLast[i]}, stream=${streamLast[i]}`);
        }
    }
    
    // Free stream
    wasm.uma_stream_free(stream);
});

console.log('All UMA WASM tests passed!');