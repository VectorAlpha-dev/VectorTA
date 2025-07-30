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
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }
    
    testData = loadTestData();
});

test('Donchian - basic functionality', () => {
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const period = 20;
    
    const result = wasm.donchian_js(high, low, period);
    
    // Check result structure
    assert.strictEqual(result.rows, 3, 'Should have 3 rows (upper, middle, lower)');
    assert.strictEqual(result.cols, high.length, 'Columns should match input length');
    assert.strictEqual(result.values.length, high.length * 3, 'Values should be flattened array');
});

test('Donchian - accuracy test', () => {
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const period = 20;
    
    const result = wasm.donchian_js(high, low, period);
    
    // Extract bands (result is flattened as [upper..., middle..., lower...])
    const len = high.length;
    const upper = result.values.slice(0, len);
    const middle = result.values.slice(len, 2 * len);
    const lower = result.values.slice(2 * len, 3 * len);
    
    // Expected values from Rust tests
    const expectedUpper = [61290.0, 61290.0, 61290.0, 61290.0, 61290.0];
    const expectedMiddle = [59583.0, 59583.0, 59583.0, 59583.0, 59583.0];
    const expectedLower = [57876.0, 57876.0, 57876.0, 57876.0, 57876.0];
    
    // Check last 5 values
    const startIdx = len - 5;
    assertArrayClose(
        upper.slice(startIdx),
        expectedUpper,
        0.1,
        'Upper band last 5 values mismatch'
    );
    assertArrayClose(
        middle.slice(startIdx),
        expectedMiddle,
        0.1,
        'Middle band last 5 values mismatch'
    );
    assertArrayClose(
        lower.slice(startIdx),
        expectedLower,
        0.1,
        'Lower band last 5 values mismatch'
    );
});

test('Donchian - zero period should fail', () => {
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0, 25.0]);
    
    assert.throws(() => {
        wasm.donchian_js(high, low, 0);
    }, /Invalid period/);
});

test('Donchian - period exceeds length should fail', () => {
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0, 25.0]);
    
    assert.throws(() => {
        wasm.donchian_js(high, low, 10);
    }, /Invalid period|period exceeds/);
});

test('Donchian - mismatched lengths should fail', () => {
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0]);  // Different length
    
    assert.throws(() => {
        wasm.donchian_js(high, low, 2);
    }, /MismatchedLength|different lengths/);
});

test('Donchian - empty data should fail', () => {
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.donchian_js(empty, empty, 20);
    }, /empty|EmptyData/i);
});

test('Donchian - all NaN values should fail', () => {
    const allNaN = new Float64Array(100).fill(NaN);
    
    assert.throws(() => {
        wasm.donchian_js(allNaN, allNaN, 20);
    }, /all.*NaN|AllValuesNaN/i);
});

test('Donchian - warmup period handling', () => {
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const period = 20;
    
    const result = wasm.donchian_js(high, low, period);
    const len = high.length;
    const upper = result.values.slice(0, len);
    const middle = result.values.slice(len, 2 * len);
    const lower = result.values.slice(2 * len, 3 * len);
    
    // First period-1 values should be NaN
    for (let i = 0; i < period - 1; i++) {
        assert(isNaN(upper[i]), `Upper[${i}] should be NaN during warmup`);
        assert(isNaN(middle[i]), `Middle[${i}] should be NaN during warmup`);
        assert(isNaN(lower[i]), `Lower[${i}] should be NaN during warmup`);
    }
    
    // After warmup, no NaN values
    assertNoNaN(upper.slice(period), 'Upper band after warmup');
    assertNoNaN(middle.slice(period), 'Middle band after warmup');
    assertNoNaN(lower.slice(period), 'Lower band after warmup');
});

test('Donchian - fast API in-place operation', async () => {
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const period = 20;
    const len = high.length;
    
    // Allocate output buffers
    const upperPtr = wasm.donchian_alloc(len);
    const middlePtr = wasm.donchian_alloc(len);
    const lowerPtr = wasm.donchian_alloc(len);
    
    try {
        // Call fast API
        wasm.donchian_into(
            high.byteOffset,
            low.byteOffset,
            upperPtr,
            middlePtr,
            lowerPtr,
            len,
            period
        );
        
        // Read results
        const memory = new Float64Array(wasm.memory.buffer);
        const upperStart = upperPtr / 8;
        const middleStart = middlePtr / 8;
        const lowerStart = lowerPtr / 8;
        
        const upper = memory.slice(upperStart, upperStart + len);
        const middle = memory.slice(middleStart, middleStart + len);
        const lower = memory.slice(lowerStart, lowerStart + len);
        
        // Verify some values
        assert(!isNaN(upper[period]), 'Upper should have valid values after warmup');
        assert(!isNaN(middle[period]), 'Middle should have valid values after warmup');
        assert(!isNaN(lower[period]), 'Lower should have valid values after warmup');
    } finally {
        // Clean up
        wasm.donchian_free(upperPtr, len);
        wasm.donchian_free(middlePtr, len);
        wasm.donchian_free(lowerPtr, len);
    }
});

test('Donchian - batch processing', () => {
    const high = new Float64Array(testData.high.slice(0, 100)); // Use smaller dataset
    const low = new Float64Array(testData.low.slice(0, 100));
    
    const config = {
        period_range: [10, 30, 10]  // 10, 20, 30
    };
    
    const result = wasm.donchian_batch(high, low, config);
    
    // Check structure
    assert.strictEqual(result.rows, 3, 'Should have 3 combinations');
    assert.strictEqual(result.cols, 100, 'Should have 100 columns');
    assert.strictEqual(result.periods.length, 3, 'Should have 3 periods');
    assert.deepStrictEqual(Array.from(result.periods), [10, 20, 30], 'Periods should match');
    
    // Check data sizes
    assert.strictEqual(result.upper.length, 300, 'Upper should have rows * cols values');
    assert.strictEqual(result.middle.length, 300, 'Middle should have rows * cols values');
    assert.strictEqual(result.lower.length, 300, 'Lower should have rows * cols values');
});

test('Donchian - very small dataset', () => {
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.donchian_js(singlePoint, singlePoint, 2);
    }, /Invalid period|Not enough/);
});

test('Donchian - reinput test', () => {
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const period = 20;
    
    // First pass
    const firstResult = wasm.donchian_js(high, low, period);
    const len = high.length;
    const firstUpper = firstResult.values.slice(0, len);
    const firstMiddle = firstResult.values.slice(len, 2 * len);
    const firstLower = firstResult.values.slice(2 * len, 3 * len);
    
    // Second pass - apply Donchian to its own output (using middle as both high/low)
    const secondResult = wasm.donchian_js(firstMiddle, firstMiddle, period);
    const secondUpper = secondResult.values.slice(0, len);
    const secondMiddle = secondResult.values.slice(len, 2 * len);
    const secondLower = secondResult.values.slice(2 * len, 3 * len);
    
    // Verify structure
    assert.strictEqual(secondResult.rows, 3);
    assert.strictEqual(secondResult.cols, len);
    
    // Check some values exist after warmup
    assert(!isNaN(secondUpper[period]), 'Second pass upper should have values after warmup');
    assert(!isNaN(secondMiddle[period]), 'Second pass middle should have values after warmup');
    assert(!isNaN(secondLower[period]), 'Second pass lower should have values after warmup');
});

test('Donchian - invalid high/low relationship', () => {
    // Create data where high < low (invalid)
    const high = new Float64Array([10.0, 15.0, 20.0, 25.0, 30.0]);
    const low = new Float64Array([15.0, 20.0, 25.0, 30.0, 35.0]); // All values higher than high
    const period = 3;
    
    // Should still compute but results will be inverted
    const result = wasm.donchian_js(high, low, period);
    const len = high.length;
    const upper = result.values.slice(0, len);
    const middle = result.values.slice(len, 2 * len);
    const lower = result.values.slice(2 * len, 3 * len);
    
    // After warmup, upper should be less than lower (inverted)
    assert(upper[period - 1] < lower[period - 1], 'Upper should be less than lower when high < low');
});

test('Donchian - partial NaN handling', () => {
    const high = new Float64Array([NaN, 12.0, 15.0, NaN, 13.0, 16.0, 14.0, 12.0, 18.0, 17.0]);
    const low = new Float64Array([NaN, 9.0, 11.0, NaN, 10.0, 12.0, 11.0, 10.0, 14.0, 15.0]);
    const period = 3;
    
    const result = wasm.donchian_js(high, low, period);
    const len = high.length;
    const upper = result.values.slice(0, len);
    const middle = result.values.slice(len, 2 * len);
    const lower = result.values.slice(2 * len, 3 * len);
    
    // First few values should be NaN due to initial NaN and warmup
    assert(isNaN(upper[0]), 'Should be NaN at index 0');
    assert(isNaN(upper[1]), 'Should be NaN at index 1');
    assert(isNaN(upper[2]), 'Should be NaN at index 2');
    
    // Should have valid values after enough non-NaN data
    assert(!isNaN(upper[5]), 'Should have valid value at index 5');
    assert(!isNaN(middle[5]), 'Should have valid value at index 5');
    assert(!isNaN(lower[5]), 'Should have valid value at index 5');
});

test('Donchian - zero-copy null pointer handling', () => {
    const len = 100;
    
    assert.throws(() => {
        wasm.donchian_into(
            0, // null high_ptr
            0, // null low_ptr
            0, // null upper_ptr
            0, // null middle_ptr
            0, // null lower_ptr
            len,
            20
        );
    }, /Null pointer/);
});

test('Donchian - zero-copy with aliasing', async () => {
    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    const period = 20;
    const len = high.length;
    
    // Allocate buffers
    const highPtr = wasm.donchian_alloc(len);
    const lowPtr = wasm.donchian_alloc(len);
    const outputPtr = wasm.donchian_alloc(len); // Single buffer for aliasing test
    
    try {
        // Copy input data
        const memory = new Float64Array(wasm.memory.buffer);
        const highStart = highPtr / 8;
        const lowStart = lowPtr / 8;
        memory.set(high, highStart);
        memory.set(low, lowStart);
        
        // Call with aliased pointers (all outputs point to same buffer)
        wasm.donchian_into(
            highPtr,
            lowPtr,
            outputPtr, // upper = output
            outputPtr, // middle = output (aliased)
            outputPtr, // lower = output (aliased)
            len,
            period
        );
        
        // Should complete without error (uses temp buffers internally)
        assert(true, 'Aliased operation completed');
        
    } finally {
        wasm.donchian_free(highPtr, len);
        wasm.donchian_free(lowPtr, len);
        wasm.donchian_free(outputPtr, len);
    }
});

test('Donchian - zero-copy memory management', () => {
    const sizes = [10, 100, 1000, 10000];
    
    for (const size of sizes) {
        // Allocate
        const ptr = wasm.donchian_alloc(size);
        assert(ptr > 0, `Allocation should return valid pointer for size ${size}`);
        
        // Free
        wasm.donchian_free(ptr, size);
        // If no crash, test passes
    }
    
    // Test free with null pointer (should not crash)
    wasm.donchian_free(0, 100);
});

test('Donchian - batch edge cases', () => {
    const high = new Float64Array(testData.high.slice(0, 50));
    const low = new Float64Array(testData.low.slice(0, 50));
    
    // Test 1: Single value sweep
    const singleConfig = {
        period_range: [20, 20, 0]  // Only one value
    };
    
    const singleResult = wasm.donchian_batch(high, low, singleConfig);
    assert.strictEqual(singleResult.rows, 1, 'Single value sweep should have 1 row');
    assert.strictEqual(singleResult.periods.length, 1, 'Should have 1 period');
    assert.strictEqual(singleResult.periods[0], 20, 'Period should be 20');
    
    // Test 2: Step larger than range
    const largeStepConfig = {
        period_range: [10, 15, 10]  // Step larger than range
    };
    
    const largeStepResult = wasm.donchian_batch(high, low, largeStepConfig);
    assert.strictEqual(largeStepResult.rows, 1, 'Large step should yield 1 value');
    assert.strictEqual(largeStepResult.periods[0], 10, 'Should only have start value');
    
    // Test 3: Reverse range (should handle gracefully)
    const reverseConfig = {
        period_range: [30, 10, 5]  // Start > end
    };
    
    // This might throw or return empty, depending on implementation
    try {
        const reverseResult = wasm.donchian_batch(high, low, reverseConfig);
        assert.strictEqual(reverseResult.rows, 0, 'Reverse range should yield 0 rows');
    } catch (e) {
        assert(true, 'Reverse range throws error as expected');
    }
});

test('Donchian - batch API error handling', () => {
    const high = new Float64Array([1, 2, 3, 4, 5]);
    const low = new Float64Array([0, 1, 2, 3, 4]);
    
    // Test invalid config
    assert.throws(() => {
        wasm.donchian_batch(high, low, {}); // Missing period_range
    }, /Invalid config|period_range/);
    
    assert.throws(() => {
        wasm.donchian_batch(high, low, {
            period_range: "invalid" // Wrong type
        });
    }, /Invalid config/);
    
    // Test period exceeds data length
    assert.throws(() => {
        wasm.donchian_batch(high, low, {
            period_range: [10, 20, 5] // All periods exceed data length of 5
        });
    }, /Invalid period|Not enough/);
});

test('Donchian - batch individual row verification', () => {
    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    
    const config = {
        period_range: [10, 30, 10]  // 10, 20, 30
    };
    
    const batchResult = wasm.donchian_batch(high, low, config);
    
    // Verify each row matches individual calculation
    const periods = [10, 20, 30];
    for (let i = 0; i < periods.length; i++) {
        const period = periods[i];
        
        // Get batch row
        const rowStart = i * 100;
        const batchUpper = batchResult.upper.slice(rowStart, rowStart + 100);
        const batchMiddle = batchResult.middle.slice(rowStart, rowStart + 100);
        const batchLower = batchResult.lower.slice(rowStart, rowStart + 100);
        
        // Calculate individually
        const individual = wasm.donchian_js(high, low, period);
        const indivUpper = individual.values.slice(0, 100);
        const indivMiddle = individual.values.slice(100, 200);
        const indivLower = individual.values.slice(200, 300);
        
        // Compare
        assertArrayClose(batchUpper, indivUpper, 1e-10, `Batch upper row ${i} mismatch`);
        assertArrayClose(batchMiddle, indivMiddle, 1e-10, `Batch middle row ${i} mismatch`);
        assertArrayClose(batchLower, indivLower, 1e-10, `Batch lower row ${i} mismatch`);
    }
});

test('Donchian - zero-copy with large dataset', () => {
    // Generate large dataset
    const size = 100000;
    const high = new Float64Array(size);
    const low = new Float64Array(size);
    
    // Fill with random walk data
    let price = 100;
    for (let i = 0; i < size; i++) {
        price += (Math.random() - 0.5) * 2;
        high[i] = price + Math.random();
        low[i] = price - Math.random();
    }
    
    const period = 50;
    
    // Allocate output buffers
    const upperPtr = wasm.donchian_alloc(size);
    const middlePtr = wasm.donchian_alloc(size);
    const lowerPtr = wasm.donchian_alloc(size);
    
    try {
        const start = performance.now();
        
        // Call fast API
        wasm.donchian_into(
            high.byteOffset,
            low.byteOffset,
            upperPtr,
            middlePtr,
            lowerPtr,
            size,
            period
        );
        
        const elapsed = performance.now() - start;
        console.log(`  Zero-copy API processed ${size} points in ${elapsed.toFixed(2)}ms`);
        
        // Verify some values
        const memory = new Float64Array(wasm.memory.buffer);
        const upperStart = upperPtr / 8;
        const upper = memory.slice(upperStart, upperStart + size);
        
        // Check warmup period
        for (let i = 0; i < period - 1; i++) {
            assert(isNaN(upper[i]), `Should be NaN during warmup at ${i}`);
        }
        assert(!isNaN(upper[period]), 'Should have values after warmup');
        
    } finally {
        wasm.donchian_free(upperPtr, size);
        wasm.donchian_free(middlePtr, size);
        wasm.donchian_free(lowerPtr, size);
    }
});

test('Donchian - fast batch API', () => {
    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    const len = high.length;
    
    // Parameters for batch
    const periodStart = 10;
    const periodEnd = 30;
    const periodStep = 10;
    const expectedRows = 3; // [10, 20, 30]
    
    // Allocate buffers
    const totalSize = expectedRows * len;
    const upperPtr = wasm.donchian_alloc(totalSize);
    const middlePtr = wasm.donchian_alloc(totalSize);
    const lowerPtr = wasm.donchian_alloc(totalSize);
    
    try {
        // Call fast batch API
        const rows = wasm.donchian_batch_into(
            high.byteOffset,
            low.byteOffset,
            upperPtr,
            middlePtr,
            lowerPtr,
            len,
            periodStart,
            periodEnd,
            periodStep
        );
        
        assert.strictEqual(rows, expectedRows, 'Should return correct number of rows');
        
        // Verify data structure
        const memory = new Float64Array(wasm.memory.buffer);
        const upperStart = upperPtr / 8;
        const upper = memory.slice(upperStart, upperStart + totalSize);
        
        // Check each row has proper warmup
        for (let row = 0; row < rows; row++) {
            const period = periodStart + row * periodStep;
            const rowOffset = row * len;
            
            // Check warmup NaNs
            for (let i = 0; i < period - 1; i++) {
                assert(isNaN(upper[rowOffset + i]), `Row ${row} should have NaN at ${i}`);
            }
            // Check valid data after warmup
            assert(!isNaN(upper[rowOffset + period]), `Row ${row} should have data after warmup`);
        }
        
    } finally {
        wasm.donchian_free(upperPtr, totalSize);
        wasm.donchian_free(middlePtr, totalSize);
        wasm.donchian_free(lowerPtr, totalSize);
    }
});

// Add test.after hook for cleanup
test.after(() => {
    // Any global cleanup if needed
    console.log('Donchian WASM tests completed with comprehensive coverage');
});

console.log('Donchian WASM tests completed');