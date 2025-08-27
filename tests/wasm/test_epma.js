/**
 * WASM binding tests for EPMA indicator.
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
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }
    
    testData = loadTestData();
});

test('EPMA partial params', () => {
    // Test with default parameters - mirrors check_epma_partial_params
    const close = new Float64Array(testData.close);
    
    // Default parameters (period=11, offset=4)
    const result = wasm.epma_js(close, 11, 4);
    assert.strictEqual(result.length, close.length);
    
    // Custom parameters
    const resultCustom1 = wasm.epma_js(close, 15, 6);
    assert.strictEqual(resultCustom1.length, close.length);
    
    const resultCustom2 = wasm.epma_js(close, 9, 3);
    assert.strictEqual(resultCustom2.length, close.length);
});

test('EPMA accuracy', async () => {
    // Test accuracy matches expected values - mirrors check_epma_accuracy
    const close = new Float64Array(testData.close);
    const expectedLast5 = [59174.48, 59201.04, 59167.60, 59200.32, 59117.04];
    
    const result = wasm.epma_js(close, 11, 4);
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expectedLast5,
        0.1,  // 1e-1 tolerance as in Rust test
        "EPMA last 5 values mismatch"
    );
    
    // Compare full output with Rust
    await compareWithRust('epma', result, 'close', { period: 11, offset: 4 });
});

test('EPMA empty input', () => {
    // Test error with empty input - mirrors check_epma_empty_input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.epma_js(empty, 11, 4);
    }, /Input data slice is empty/);
});

test('EPMA zero period', () => {
    // Test error with zero period - mirrors check_epma_zero_period
    const data = new Float64Array([10.0, 20.0, 30.0]);
    
    // With period=0 and offset=4, it will fail with "Invalid offset"
    assert.throws(() => {
        wasm.epma_js(data, 0, 4);
    }, /Invalid offset/);
    
    // With period=0 and offset=0, it will fail with "Invalid offset" because 0 >= 0
    assert.throws(() => {
        wasm.epma_js(data, 0, 0);
    }, /Invalid offset/);
});

test('EPMA period exceeds length', () => {
    // Test error when period exceeds data length - mirrors check_epma_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.epma_js(dataSmall, 10, 4);
    }, /Invalid period/);
});

test('EPMA very small dataset', () => {
    // Test error with insufficient data - mirrors check_epma_very_small_dataset
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.epma_js(singlePoint, 9, 4);
    }, /Invalid period|Not enough valid data/);
});

test('EPMA invalid offset', () => {
    // Test error with invalid offset - mirrors check_epma_invalid_offset
    const data = new Float64Array([1.0, 2.0, 3.0, 4.0]);
    
    // Offset >= period
    assert.throws(() => {
        wasm.epma_js(data, 3, 3);
    }, /Invalid offset/);
    
    assert.throws(() => {
        wasm.epma_js(data, 3, 4);
    }, /Invalid offset/);
});

test('EPMA all NaN input', () => {
    // Test error with all NaN values
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.epma_js(allNaN, 11, 4);
    }, /All values are NaN/);
});

test('EPMA period too small', () => {
    // Test error with period < 2
    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
    
    // With period=1 and offset=4, it will fail with "Invalid offset"
    assert.throws(() => {
        wasm.epma_js(data, 1, 4);
    }, /Invalid offset/);
    
    // With period=1 and offset=0, it will fail with "Invalid period"
    assert.throws(() => {
        wasm.epma_js(data, 1, 0);
    }, /Invalid period/);
});

test('EPMA not enough valid data', () => {
    // Test error when there's not enough valid data after NaN prefix
    const data = new Float64Array([NaN, NaN, 1.0, 2.0, 3.0]);
    
    // With period=3, offset=2, needs period+offset+1=6 valid values
    assert.throws(() => {
        wasm.epma_js(data, 3, 2);
    }, /Not enough valid data/);
});

test('EPMA reinput', () => {
    // Test applying indicator twice - mirrors check_epma_reinput
    const close = new Float64Array(testData.close);
    
    // First pass with period=9, offset=4
    const firstResult = wasm.epma_js(close, 9, 4);
    assert.strictEqual(firstResult.length, close.length);
    
    // Check warmup for first pass: 9 + 4 + 1 = 14
    assertAllNaN(firstResult.slice(0, 14), "First pass warmup should be NaN");
    
    // Second pass with period=3, offset=0 (explicitly set to avoid error)
    const secondResult = wasm.epma_js(firstResult, 3, 0);
    assert.strictEqual(secondResult.length, firstResult.length);
    
    // Check warmup for second pass: 3 + 0 + 1 = 4
    assertAllNaN(secondResult.slice(0, 4), "Second pass warmup should be NaN");
});

test('EPMA NaN handling', () => {
    // Test NaN handling - mirrors check_epma_nan_handling
    const close = new Float64Array(testData.close);
    
    const result = wasm.epma_js(close, 11, 4);
    assert.strictEqual(result.length, close.length);
    
    // First period+offset+1 values should be NaN (11+4+1=16)
    assertAllNaN(result.slice(0, 16), "Expected NaN in warmup period");
    
    // After warmup period (240), no NaN values should exist
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
});

test('EPMA batch single parameter set', () => {
    // Test batch with single parameter combination
    const close = new Float64Array(testData.close);
    
    // Single parameter set using unified API
    const config = {
        period_range: [11, 11, 0],  // period range
        offset_range: [4, 4, 0]     // offset range
    };
    
    const batchResult = wasm.epma_batch(close, config);
    
    // Should match single calculation
    const singleResult = wasm.epma_js(close, 11, 4);
    
    assert.strictEqual(batchResult.values.length, singleResult.length);
    assertArrayClose(batchResult.values, singleResult, 1e-10, "Batch vs single mismatch");
});

test('EPMA batch multiple parameters', () => {
    // Test batch with multiple parameter combinations
    const close = new Float64Array(testData.close.slice(0, 100));
    
    // Multiple parameters: period 5,7,9 x offset 1,2,3
    const config = {
        period_range: [5, 9, 2],    // period: 5, 7, 9
        offset_range: [1, 3, 1]     // offset: 1, 2, 3
    };
    
    const batchResult = wasm.epma_batch(close, config);
    
    // Should have 3 x 3 = 9 rows * 100 cols = 900 values
    assert.strictEqual(batchResult.values.length, 9 * 100);
    assert.strictEqual(batchResult.combos.length, 9);
    
    // Verify first combination matches individual calculation
    const firstRow = batchResult.values.slice(0, 100);
    const singleResult = wasm.epma_js(close, 5, 1);
    assertArrayClose(
        firstRow, 
        singleResult, 
        1e-10, 
        "First batch row mismatch"
    );
});

test('EPMA batch metadata', () => {
    // Test metadata through batch function result
    const close = new Float64Array(testData.close.slice(0, 50));
    const config = {
        period_range: [5, 9, 2],    // period: 5, 7, 9
        offset_range: [1, 3, 1]     // offset: 1, 2, 3
    };
    
    const result = wasm.epma_batch(close, config);
    
    // Should have 3 x 3 = 9 combinations
    assert.strictEqual(result.combos.length, 9);
    
    // Check first combination
    assert.strictEqual(result.combos[0].period, 5);   // period
    assert.strictEqual(result.combos[0].offset, 1);   // offset
    
    // Check second combination
    assert.strictEqual(result.combos[1].period, 5);   // period
    assert.strictEqual(result.combos[1].offset, 2);   // offset
    
    // Check fourth combination (second period)
    assert.strictEqual(result.combos[3].period, 7);   // period
    assert.strictEqual(result.combos[3].offset, 1);   // offset
});

test('EPMA batch warmup validation', () => {
    // Test batch warmup period handling
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const config = {
        period_range: [5, 10, 5],   // period: 5, 10
        offset_range: [2, 4, 2]     // offset: 2, 4
    };
    
    const batchResult = wasm.epma_batch(close, config);
    
    assert.strictEqual(batchResult.combos.length, 4);  // 2 periods x 2 offsets
    
    // Check warmup periods for each combination
    for (let combo = 0; combo < batchResult.combos.length; combo++) {
        const period = batchResult.combos[combo].period;
        const offset = batchResult.combos[combo].offset;
        const warmup = period + offset + 1;
        const rowStart = combo * 50;
        const rowData = batchResult.values.slice(rowStart, rowStart + 50);
        
        // First warmup values should be NaN
        for (let i = 0; i < warmup; i++) {
            assert(isNaN(rowData[i]), 
                `Expected NaN at index ${i} for period=${period}, offset=${offset}`);
        }
        
        // After warmup should have values
        for (let i = warmup; i < 50; i++) {
            assert(!isNaN(rowData[i]), 
                `Unexpected NaN at index ${i} for period=${period}, offset=${offset}`);
        }
    }
});

test('EPMA batch edge cases', () => {
    // Test edge cases for batch processing
    const close = new Float64Array(20).fill(0).map((_, i) => i + 1);
    
    // Single value sweep
    const singleConfig = {
        period_range: [5, 5, 1],
        offset_range: [1, 1, 1]
    };
    const singleBatch = wasm.epma_batch(close, singleConfig);
    
    assert.strictEqual(singleBatch.values.length, 20);
    assert.strictEqual(singleBatch.combos.length, 1);
    
    // Step = 0 should return single value when start=end
    const zeroStepConfig = {
        period_range: [7, 7, 0],
        offset_range: [2, 2, 0]
    };
    const zeroStepBatch = wasm.epma_batch(close, zeroStepConfig);
    
    assert.strictEqual(zeroStepBatch.values.length, 20); // Single combination
    assert.strictEqual(zeroStepBatch.combos.length, 1);
    
    // Empty data should throw
    assert.throws(() => {
        const emptyConfig = {
            period_range: [11, 11, 0],
            offset_range: [4, 4, 0]
        };
        wasm.epma_batch(new Float64Array([]), emptyConfig);
    }, /Input data slice is empty|All values are NaN/);
});

test('EPMA batch performance test', () => {
    // Test that batch is more efficient than multiple single calls
    const close = new Float64Array(testData.close.slice(0, 200));
    
    // Batch calculation
    const batchConfig = {
        period_range: [5, 15, 2],   // 6 period values
        offset_range: [1, 4, 1]     // 4 offset values (max 4 to ensure offset < min period of 5)
    };
    
    const startBatch = Date.now();
    const batchResult = wasm.epma_batch(close, batchConfig);
    const batchTime = Date.now() - startBatch;
    
    // Equivalent single calculations
    const startSingle = Date.now();
    const singleResults = [];
    for (let period = 5; period <= 15; period += 2) {
        for (let offset = 1; offset <= 4; offset += 1) {
            // All offsets are valid since max offset=4 < min period=5
            singleResults.push(...wasm.epma_js(close, period, offset));
        }
    }
    const singleTime = Date.now() - startSingle;
    
    // Batch should have same total length
    assert.strictEqual(batchResult.values.length, singleResults.length);
    
    // Log performance (batch should be faster)
    console.log(`  EPMA Batch time: ${batchTime}ms, Single calls time: ${singleTime}ms`);
});

// Zero-copy API tests
test('EPMA zero-copy API', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const period = 5;
    const offset = 2;
    
    // Allocate buffer
    const ptr = wasm.epma_alloc(data.length);
    assert(ptr !== 0, 'Failed to allocate memory');
    
    // Create view into WASM memory
    const memView = new Float64Array(
        wasm.__wasm.memory.buffer,
        ptr,
        data.length
    );
    
    // Copy data into WASM memory
    memView.set(data);
    
    // Compute EPMA in-place
    try {
        wasm.epma_into(ptr, ptr, data.length, period, offset);
        
        // Verify results match regular API
        const regularResult = wasm.epma_js(data, period, offset);
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                continue; // Both NaN is OK
            }
            assert(Math.abs(regularResult[i] - memView[i]) < 1e-10,
                   `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${memView[i]}`);
        }
    } finally {
        // Always free memory
        wasm.epma_free(ptr, data.length);
    }
});

test('EPMA zero-copy with large dataset', () => {
    const size = 10000;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.sin(i * 0.01) + Math.random() * 0.1;
    }
    
    const ptr = wasm.epma_alloc(size);
    assert(ptr !== 0, 'Failed to allocate large buffer');
    
    try {
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        memView.set(data);
        
        wasm.epma_into(ptr, ptr, size, 11, 4);
        
        // Recreate view in case memory grew
        const memView2 = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        
        // Check warmup period has NaN (period + offset + 1 = 11 + 4 + 1 = 16)
        for (let i = 0; i < 16; i++) {
            assert(isNaN(memView2[i]), `Expected NaN at warmup index ${i}`);
        }
        
        // Check after warmup has values
        for (let i = 16; i < Math.min(100, size); i++) {
            assert(!isNaN(memView2[i]), `Unexpected NaN at index ${i}`);
        }
    } finally {
        wasm.epma_free(ptr, size);
    }
});

test('EPMA zero-copy error handling', () => {
    // Test null pointer
    assert.throws(() => {
        wasm.epma_into(0, 0, 10, 9, 3);
    }, /null pointer|Null pointer/i);
    
    // Test invalid parameters with allocated memory
    const ptr = wasm.epma_alloc(10);
    try {
        // Initialize memory with valid data
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, 10);
        for (let i = 0; i < 10; i++) {
            memView[i] = i + 1.0;
        }
        
        // Invalid period (period=0 triggers offset error since 3 >= 0)
        assert.throws(() => {
            wasm.epma_into(ptr, ptr, 10, 0, 3);
        }, /Invalid offset/);
        
        // Invalid offset (offset >= period)
        assert.throws(() => {
            wasm.epma_into(ptr, ptr, 10, 5, 5);
        }, /Invalid offset/);
    } finally {
        wasm.epma_free(ptr, 10);
    }
});

test('EPMA zero-copy memory management', () => {
    // Allocate and free multiple times to ensure no leaks
    const sizes = [100, 1000, 10000];
    
    for (const size of sizes) {
        const ptr = wasm.epma_alloc(size);
        assert(ptr !== 0, `Failed to allocate ${size} elements`);
        
        // Write pattern to verify memory
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            memView[i] = i * 1.5;
        }
        
        // Verify pattern
        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(memView[i], i * 1.5, `Memory corruption at index ${i}`);
        }
        
        // Free memory
        wasm.epma_free(ptr, size);
    }
});

test('EPMA batch unified API', async () => {
    const close = new Float64Array(testData.close.slice(0, 100));
    
    // Test unified batch API with serde
    const config = {
        period_range: [5, 9, 2],  // 5, 7, 9
        offset_range: [1, 3, 1]   // 1, 2, 3
    };
    
    const result = wasm.epma_batch(close, config);
    
    // Verify structure
    assert(result.values, 'Missing values in batch result');
    assert(result.combos, 'Missing combos in batch result');
    assert(result.rows === 9, `Expected 9 rows, got ${result.rows}`);
    assert(result.cols === 100, `Expected 100 cols, got ${result.cols}`);
    assert(result.values.length === 900, `Expected 900 values, got ${result.values.length}`);
    
    // Verify first combination matches individual calculation
    const firstCombo = result.combos[0];
    const firstRow = result.values.slice(0, 100);
    const singleResult = wasm.epma_js(close, firstCombo.period || 11, firstCombo.offset || 4);
    
    assertArrayClose(
        firstRow, 
        singleResult, 
        1e-10, 
        "First batch row mismatch with unified API"
    );
});

test('EPMA batch - new unified API with single parameter', () => {
    const close = new Float64Array(testData.close.slice(0, 100));
    
    // Test with single parameter set using unified API
    const config = {
        period_range: [11, 11, 0],  // period: just 11
        offset_range: [4, 4, 0]     // offset: just 4
    };
    
    const result = wasm.epma_batch(close, config);
    
    assert(result.values, 'Missing values in result');
    assert(result.combos, 'Missing combos in result');
    assert.strictEqual(result.values.length, 100, 'Values length mismatch');
    assert.strictEqual(result.combos.length, 1, 'Should have 1 combination');
    assert.strictEqual(result.combos[0].period, 11, 'Period mismatch');
    assert.strictEqual(result.combos[0].offset, 4, 'Offset mismatch');
    
    // Compare with single calculation
    const singleResult = wasm.epma_js(close, 11, 4);
    assertArrayClose(result.values, singleResult, 1e-10, 
        'Batch single parameter result mismatch');
});

test('EPMA batch - new unified API with multiple parameters', () => {
    const close = new Float64Array(testData.close.slice(0, 50));
    
    // Test with multiple parameter combinations
    const config = {
        period_range: [5, 9, 2],    // periods: 5, 7, 9
        offset_range: [1, 3, 1]     // offsets: 1, 2, 3
    };
    
    const result = wasm.epma_batch(close, config);
    
    assert(result.values, 'Missing values in result');
    assert(result.combos, 'Missing combos in result');
    assert.strictEqual(result.combos.length, 9, 'Should have 9 combinations');
    assert.strictEqual(result.values.length, 450, 'Should have 9*50 values');
    assert.strictEqual(result.rows, 9, 'Should have 9 rows');
    assert.strictEqual(result.cols, 50, 'Should have 50 cols');
    
    // Verify first combination (period=5, offset=1)
    const firstRow = result.values.slice(0, 50);
    const singleResult = wasm.epma_js(close, 5, 1);
    assertArrayClose(firstRow, singleResult, 1e-10,
        'First batch row mismatch');
    
    // Check warmup periods
    for (let i = 0; i < 9; i++) {
        const period = result.combos[i].period;
        const offset = result.combos[i].offset;
        const warmup = period + offset + 1;
        const rowStart = i * 50;
        
        // Check NaN values in warmup
        for (let j = 0; j < warmup && j < 50; j++) {
            assert(isNaN(result.values[rowStart + j]),
                `Expected NaN at row ${i} index ${j}`);
        }
    }
});

test('EPMA zero-copy error handling', () => {
    const data = new Float64Array([1, 2, 3, 4, 5]);
    const ptr = wasm.epma_alloc(data.length);
    
    try {
        // Copy data
        const view = new Float64Array(wasm.__wasm.memory.buffer, ptr, data.length);
        view.set(data);
        
        // Test invalid period
        assert.throws(
            () => wasm.epma_into(ptr, ptr, data.length, 0, 4),
            /Invalid offset/,
            'Should fail with period=0'
        );
        
        // Test invalid offset (>= period)
        assert.throws(
            () => wasm.epma_into(ptr, ptr, data.length, 3, 3),
            /Invalid offset/,
            'Should fail with offset >= period'
        );
        
        // Test period exceeds data length
        assert.throws(
            () => wasm.epma_into(ptr, ptr, data.length, 10, 4),
            /Invalid/,  // Could be 'Invalid period' or 'Invalid offset'
            'Should fail when period > data length'
        );
    } finally {
        wasm.epma_free(ptr, data.length);
    }
});

test('EPMA batch zero-copy API', () => {
    const close = new Float64Array(testData.close.slice(0, 50));
    
    // Allocate memory for input and output
    const inPtr = wasm.epma_alloc(close.length);
    const rows = 4; // 2 periods x 2 offsets
    const outPtr = wasm.epma_alloc(close.length * rows);
    
    try {
        // Copy data to input buffer
        const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, close.length);
        inView.set(close);
        
        // Execute batch computation
        const numRows = wasm.epma_batch_into(
            inPtr, outPtr, close.length,
            5, 10, 5,     // period: 5, 10
            2, 4, 2       // offset: 2, 4
        );
        
        assert.strictEqual(numRows, rows, `Expected ${rows} rows, got ${numRows}`);
        
        // Verify results
        const outView = new Float64Array(wasm.__wasm.memory.buffer, outPtr, close.length * rows);
        
        // Check each row has appropriate NaN warmup
        const combinations = [];
        for (let p = 5; p <= 10; p += 5) {
            for (let o = 2; o <= 4; o += 2) {
                combinations.push({ period: p, offset: o });
            }
        }
        for (let row = 0; row < rows; row++) {
            const period = combinations[row].period;
            const offset = combinations[row].offset;
            const warmup = period + offset + 1;
            const rowStart = row * 50;
            
            // Check warmup NaNs
            for (let i = 0; i < warmup && i < 50; i++) {
                assert(isNaN(outView[rowStart + i]), 
                    `Expected NaN at row ${row} index ${i}`);
            }
        }
    } finally {
        wasm.epma_free(inPtr, close.length);
        wasm.epma_free(outPtr, close.length * rows);
    }
});

// Note: Streaming tests would require streaming functions to be exposed in WASM bindings

test.after(() => {
    console.log('EPMA WASM tests completed');
});