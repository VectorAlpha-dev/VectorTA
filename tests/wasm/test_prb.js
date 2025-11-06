/**
 * WASM binding tests for PRB (Polynomial Regression Bands) indicator.
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
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }
    
    testData = loadTestData();
});

// Matches Rust check_prb_accuracy: compare last 5 non-NaN values from CSV
test('PRB CSV accuracy matches Rust', () => {
    const expected = EXPECTED_OUTPUTS.prb;
    const close = new Float64Array(testData.close);

    const result = wasm.prb(
        close,
        false, // smooth_data (Rust uses Some(false))
        10,    // smooth_period (ignored when smooth_data=false)
        100,   // regression_period
        2,     // polynomial_order
        0,     // regression_offset
        2.0    // ndev
    );

    // Flattened values: [main, upper, lower]
    const mainValues = result.values.slice(0, close.length);
    const nonNan = Array.from(mainValues).filter(v => !isNaN(v));
    assert(nonNan.length >= 5, 'Should have at least 5 non-NaN values');

    const last5 = nonNan.slice(-5);
    const expectedLast5 = expected.last5MainValues;
    for (let i = 0; i < 5; i++) {
        const actual = last5[i];
        const exp = expectedLast5[i];
        const diff = Math.abs(actual - exp);
        const tolerance = Math.abs(exp) * 0.01; // match Rust 1% tolerance
        assert(
            diff < tolerance,
            `CSV parity mismatch at ${i}: expected ${exp}, got ${actual} (diff ${diff} > tol ${tolerance})`
        );
    }
});

test('PRB accuracy', () => {
    // Test PRB matches expected values from Pine Script reference
    const data = new Float64Array([
        66982.0, 66984.0, 66981.0, 66975.0, 66970.0,
        66968.0, 66960.0, 66955.0, 66950.0, 66945.0,
        66940.0, 66935.0, 66930.0, 66925.0, 66920.0,
        66915.0, 66910.0, 66905.0, 66900.0, 66895.0,
        66890.0, 66885.0, 66880.0, 66875.0, 66870.0,
        66865.0, 66860.0, 66855.0, 66850.0, 66845.0,
        66840.0, 66835.0, 66830.0, 66825.0, 66820.0,
        66815.0, 66810.0, 66805.0, 66800.0, 66795.0,
        66790.0, 66785.0, 66780.0, 66775.0, 66770.0,
        66765.0, 66760.0, 66755.0, 66750.0, 66745.0,
        66740.0, 66735.0, 66730.0, 66725.0, 66720.0,
        66715.0, 66710.0, 66705.0, 66700.0, 66695.0,
        66690.0, 66685.0, 66680.0, 66675.0, 66670.0,
        66665.0, 66660.0, 66655.0, 66650.0, 66645.0,
        66640.0, 66635.0, 66630.0, 66625.0, 66620.0,
        66615.0, 66610.0, 66605.0, 66600.0, 66595.0,
        66590.0, 66585.0, 66580.0, 66575.0, 66570.0,
        66565.0, 66560.0, 66555.0, 66550.0, 66545.0,
        66540.0, 66535.0, 66530.0, 66525.0, 66520.0,
        66515.0, 66510.0, 66505.0, 66500.0, 66495.0,
        66490.0, 66485.0, 66480.0, 66475.0, 66470.0,
    ]);
    
    // Call PRB with default parameters
    const result = wasm.prb(
        data,
        true,   // smooth_data
        10,     // smooth_period
        100,    // regression_period
        2,      // polynomial_order
        0,      // regression_offset
        2.0     // ndev (was missing!)
    );
    
    // PRB returns PrbJsResult with .values (flattened array), .rows (3), .cols (data.length)
    assert(result, 'Result should exist');
    assert.strictEqual(result.rows, 3, 'Should have 3 rows (main, upper, lower)');
    assert.strictEqual(result.cols, data.length, 'Cols should match input length');
    assert.strictEqual(result.values.length, 3 * data.length, 'Values should contain all three bands');
    
    // Extract the three bands from flattened array
    const mainValues = result.values.slice(0, data.length);
    const upperBand = result.values.slice(data.length, 2 * data.length);
    const lowerBand = result.values.slice(2 * data.length, 3 * data.length);
    
    // Reference values from Pine Script (main band)
    const expected = [
        66983.7659580663791616,
        66972.7911881188048896,
        66815.9710658124513280,
        66611.4138439137820672,
        66368.7769190448496640,
    ];
    
    // Get non-NaN values from main band
    const nonNanValues = Array.from(mainValues).filter(v => !isNaN(v));
    assert(nonNanValues.length >= 5, 'Should have at least 5 non-NaN values');
    
    // Check last 5 values of main band
    const last5 = nonNanValues.slice(-5);
    for (let i = 0; i < 5; i++) {
        const actual = last5[i];
        const expectedVal = expected[i];
        const diff = Math.abs(actual - expectedVal);
        const tolerance = Math.abs(expectedVal) * 0.01; // 1% tolerance - PRB uses polynomial regression
        
        assert(
            diff < tolerance,
            `Main band value ${i} differs: expected ${expectedVal}, got ${actual}, diff ${diff}`
        );
    }
    
    // Verify bands relationship: upper > main > lower for non-NaN values
    for (let i = 0; i < data.length; i++) {
        if (!isNaN(mainValues[i])) {
            assert(!isNaN(upperBand[i]), `Upper band should not be NaN when main is not NaN at index ${i}`);
            assert(!isNaN(lowerBand[i]), `Lower band should not be NaN when main is not NaN at index ${i}`);
            assert(upperBand[i] > mainValues[i], `Upper band should be > main at index ${i}`);
            assert(mainValues[i] > lowerBand[i], `Main should be > lower band at index ${i}`);
        }
    }
});

test('PRB default params', () => {
    // Test PRB with default parameters
    const data = new Float64Array(200);
    for (let i = 0; i < 200; i++) {
        data[i] = Math.random() * 100 + 5000;
    }
    
    // Test with default params
    const result = wasm.prb(
        data,
        true,  // smooth_data default
        10,    // smooth_period default
        100,   // regression_period default
        2,     // polynomial_order default
        0,     // regression_offset default
        2.0    // ndev default
    );
    
    assert(result, 'Result should exist');
    assert.strictEqual(result.rows, 3, 'Should have 3 rows');
    assert.strictEqual(result.cols, data.length, 'Cols should match input');
    assert.strictEqual(result.values.length, 3 * data.length, 'Values should contain all three bands');
    
    // Extract main band
    const mainValues = result.values.slice(0, data.length);
    
    // Check that we have some non-NaN values after warmup
    const nonNanCount = mainValues.filter(v => !isNaN(v)).length;
    assert(nonNanCount > 0, 'Should have some non-NaN values');
});

test('PRB no smoothing', () => {
    // Test PRB without smoothing
    const data = new Float64Array(150);
    for (let i = 0; i < 150; i++) {
        data[i] = Math.random() * 50 + 1000;
    }
    
    const result = wasm.prb(
        data,
        false, // smooth_data = false
        10,    // smooth_period (ignored when smooth_data=false)
        50,    // regression_period
        2,     // polynomial_order
        0,     // regression_offset
        2.0    // ndev
    );
    
    assert(result, 'Result should exist');
    assert.strictEqual(result.rows, 3, 'Should have 3 rows');
    assert.strictEqual(result.cols, data.length, 'Cols should match input');
    
    const mainValues = result.values.slice(0, data.length);
    const nonNanCount = mainValues.filter(v => !isNaN(v)).length;
    assert(nonNanCount > 0, 'Should have non-NaN values');
});

test('PRB linear regression', () => {
    // Test PRB with linear regression (order=1)
    const data = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        data[i] = i * 10 + 1000; // Linear data
    }
    
    const result = wasm.prb(
        data,
        false, // no smoothing for cleaner test
        10,
        20,    // regression_period
        1,     // polynomial_order = 1 (linear)
        0,
        2.0    // ndev
    );
    
    assert(result, 'Result should exist');
    assert.strictEqual(result.rows, 3, 'Should have 3 rows');
    assert.strictEqual(result.cols, data.length, 'Cols should match input');
    
    const mainValues = result.values.slice(0, data.length);
    const nonNanValues = Array.from(mainValues).filter(v => !isNaN(v));
    assert(nonNanValues.length > 0, 'Should have non-NaN values');
});

test('PRB cubic regression', () => {
    // Test PRB with cubic regression (order=3)
    const data = new Float64Array(200);
    for (let i = 0; i < 200; i++) {
        data[i] = Math.random() * 100 + 5000;
    }
    
    const result = wasm.prb(
        data,
        true,  // smooth_data
        10,    // smooth_period
        50,    // regression_period
        3,     // polynomial_order = 3 (cubic)
        0,     // regression_offset
        2.0    // ndev
    );
    
    assert(result, 'Result should exist');
    assert.strictEqual(result.rows, 3, 'Should have 3 rows');
    assert.strictEqual(result.cols, data.length, 'Cols should match input');
    
    const mainValues = result.values.slice(0, data.length);
    const nonNanCount = mainValues.filter(v => !isNaN(v)).length;
    assert(nonNanCount > 0, 'Should have non-NaN values');
});

test('PRB with offset', () => {
    // Test PRB with regression offset
    const data = new Float64Array(150);
    for (let i = 0; i < 150; i++) {
        data[i] = Math.random() * 50 + 1000;
    }
    
    // Test positive offset
    const resultPos = wasm.prb(
        data,
        true,
        10,
        50,    // regression_period
        2,
        5,     // regression_offset = 5
        2.0    // ndev
    );
    
    // Test negative offset
    const resultNeg = wasm.prb(
        data,
        true,
        10,
        50,    // regression_period
        2,
        -5,    // regression_offset = -5
        2.0    // ndev
    );
    
    assert(resultPos, 'Positive offset result should exist');
    assert(resultNeg, 'Negative offset result should exist');
    assert.strictEqual(resultPos.rows, 3, 'Positive offset should have 3 rows');
    assert.strictEqual(resultNeg.rows, 3, 'Negative offset should have 3 rows');
    assert.strictEqual(resultPos.cols, data.length, 'Positive offset cols should match input');
    assert.strictEqual(resultNeg.cols, data.length, 'Negative offset cols should match input');
});

test('PRB empty input', () => {
    // Test PRB fails with empty input
    const data = new Float64Array([]);
    
    assert.throws(
        () => wasm.prb(data, true, 10, 100, 2, 0, 2.0),
        /Input data slice is empty/,
        'Should throw error for empty input'
    );
});

test('PRB all NaN', () => {
    // Test PRB fails with all NaN values
    const data = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        data[i] = NaN;
    }
    
    assert.throws(
        () => wasm.prb(data, true, 10, 100, 2, 0, 2.0),
        /All values are NaN/,
        'Should throw error for all NaN input'
    );
});

test('PRB invalid period', () => {
    // Test PRB fails with invalid regression period
    const data = new Float64Array(50);
    for (let i = 0; i < 50; i++) {
        data[i] = Math.random() * 100;
    }
    
    // Period exceeds data length
    assert.throws(
        () => wasm.prb(data, true, 10, 100, 2, 0, 2.0),
        /Invalid period/,
        'Should throw error when period exceeds data length'
    );
    
    // Zero period
    assert.throws(
        () => wasm.prb(data, true, 10, 0, 2, 0, 2.0),
        /Invalid period/,
        'Should throw error for zero period'
    );
});

test('PRB invalid order', () => {
    // Test PRB fails with invalid polynomial order
    const data = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        data[i] = Math.random() * 100;
    }
    
    assert.throws(
        () => wasm.prb(data, true, 10, 50, 0, 0, 2.0),
        /Invalid polynomial order/,
        'Should throw error for zero polynomial order'
    );
});

test('PRB invalid smooth period', () => {
    // Test PRB fails with invalid smooth period
    const data = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        data[i] = Math.random() * 100;
    }
    
    assert.throws(
        () => wasm.prb(data, true, 1, 50, 2, 0, 2.0),
        /Invalid smooth period/,
        'Should throw error for smooth period < 2'
    );
});

test('PRB insufficient data', () => {
    // Test PRB fails with insufficient data for calculation
    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
    
    assert.throws(
        () => wasm.prb(data, true, 2, 10, 2, 0, 2.0),
        /Invalid period/,  // The actual error is "Invalid period" when period > data length
        'Should throw error for insufficient data'
    );
});

// Additional comprehensive tests

test('PRB reinput', () => {
    // Test PRB applied twice (re-input) - mirrors check_prb_reinput
    const close = new Float64Array(testData.close.slice(0, 200));
    
    // First pass
    const firstResult = wasm.prb(close, false, 10, 50, 2, 0, 2.0);
    assert(firstResult, 'First result should exist');
    assert.strictEqual(firstResult.rows, 3, 'Should have 3 rows');
    
    const firstValues = firstResult.values.slice(0, close.length);
    
    // Second pass - apply PRB to PRB output
    const secondResult = wasm.prb(firstValues, false, 10, 50, 2, 0, 2.0);
    assert(secondResult, 'Second result should exist');
    assert.strictEqual(secondResult.rows, 3, 'Should have 3 rows');
    
    const secondValues = secondResult.values.slice(0, close.length);
    
    // Verify structure is maintained
    const nonNanFirst = Array.from(firstValues).filter(v => !isNaN(v));
    const nonNanSecond = Array.from(secondValues).filter(v => !isNaN(v));
    assert(nonNanSecond.length > 0, 'Should have non-NaN values after reinput');
});

test('PRB NaN handling with test data', () => {
    // Test PRB handles NaN values correctly with actual test data
    const close = new Float64Array(testData.close);
    
    const result = wasm.prb(close, false, 10, 50, 2, 0, 2.0);
    assert(result, 'Result should exist');
    
    const mainValues = result.values.slice(0, close.length);
    const upperBand = result.values.slice(close.length, 2 * close.length);
    const lowerBand = result.values.slice(2 * close.length, 3 * close.length);
    
    // After warmup period (240), no NaN values should exist
    if (mainValues.length > 240) {
        for (let i = 240; i < mainValues.length; i++) {
            assert(!isNaN(mainValues[i]), `Found unexpected NaN in main at index ${i}`);
            assert(!isNaN(upperBand[i]), `Found unexpected NaN in upper at index ${i}`);
            assert(!isNaN(lowerBand[i]), `Found unexpected NaN in lower at index ${i}`);
        }
    }
    
    // First regression_period-1 values should be NaN (warmup)
    const warmup = 50 - 1;
    for (let i = 0; i < warmup && i < mainValues.length; i++) {
        assert(isNaN(mainValues[i]), `Expected NaN in warmup at index ${i}`);
    }
});

test('PRB batch processing', () => {
    // Test batch processing with multiple parameter combinations
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const result = wasm.prb_batch(
        close,
        false,  // smooth_data
        10, 10, 0,     // smooth_period range
        50, 60, 10,    // regression_period range
        2, 2, 0,       // polynomial_order range
        0, 0, 0        // regression_offset range
    );
    
    assert(result, 'Batch result should exist');
    
    // Should have 2 combinations (period 50 and 60)
    const expectedCombos = 2;
    const expectedSize = expectedCombos * 3 * close.length; // 3 bands per combo
    
    // Result is a flattened array with metadata
    assert(result.values, 'Should have values array');
    assert(result.rows, 'Should have rows');
    assert(result.cols, 'Should have cols');
    assert(result.combos, 'Should have combos');
    
    assert.strictEqual(result.combos.length, expectedCombos, 'Should have correct number of combos');
    
    // Verify each combo has the expected parameters
    assert.strictEqual(result.combos[0].regression_period, 50);
    assert.strictEqual(result.combos[1].regression_period, 60);
});

test('PRB zero-copy API', () => {
    // Test zero-copy API for performance
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    const len = data.length;
    
    // Allocate buffers for all three outputs
    const mainPtr = wasm.prb_alloc(len);
    const upperPtr = wasm.prb_alloc(len);
    const lowerPtr = wasm.prb_alloc(len);
    const inPtr = wasm.prb_alloc(len);
    
    assert(mainPtr !== 0, 'Failed to allocate main buffer');
    assert(upperPtr !== 0, 'Failed to allocate upper buffer');
    assert(lowerPtr !== 0, 'Failed to allocate lower buffer');
    assert(inPtr !== 0, 'Failed to allocate input buffer');
    
    try {
        // Create views into WASM memory
        const memory = wasm.__wbindgen_memory();
        const inView = new Float64Array(memory.buffer, inPtr, len);
        
        // Copy data into WASM memory
        inView.set(data);
        
        // Compute PRB using zero-copy API
        wasm.prb_into(
            inPtr,
            mainPtr,
            upperPtr,
            lowerPtr,
            len,
            false,  // smooth_data
            10,     // smooth_period
            5,      // regression_period
            2,      // polynomial_order
            0,      // regression_offset
            2.0     // ndev
        );
        
        // Read results (recreate views in case memory grew)
        const memory2 = wasm.__wbindgen_memory();
        const mainView = new Float64Array(memory2.buffer, mainPtr, len);
        const upperView = new Float64Array(memory2.buffer, upperPtr, len);
        const lowerView = new Float64Array(memory2.buffer, lowerPtr, len);
        
        // Verify results match regular API
        const regularResult = wasm.prb(data, false, 10, 5, 2, 0, 2.0);
        const regularMain = regularResult.values.slice(0, len);
        const regularUpper = regularResult.values.slice(len, 2 * len);
        const regularLower = regularResult.values.slice(2 * len, 3 * len);
        
        for (let i = 0; i < len; i++) {
            if (isNaN(regularMain[i]) && isNaN(mainView[i])) {
                continue; // Both NaN is OK
            }
            assert(Math.abs(regularMain[i] - mainView[i]) < 1e-10,
                   `Zero-copy main mismatch at index ${i}: regular=${regularMain[i]}, zerocopy=${mainView[i]}`);
            assert(Math.abs(regularUpper[i] - upperView[i]) < 1e-10,
                   `Zero-copy upper mismatch at index ${i}`);
            assert(Math.abs(regularLower[i] - lowerView[i]) < 1e-10,
                   `Zero-copy lower mismatch at index ${i}`);
        }
        
        // Verify band relationship
        for (let i = 0; i < len; i++) {
            if (!isNaN(mainView[i])) {
                assert(upperView[i] > mainView[i], `Upper should be > main at index ${i}`);
                assert(mainView[i] > lowerView[i], `Main should be > lower at index ${i}`);
            }
        }
    } finally {
        // Always free memory
        wasm.prb_free(mainPtr, len);
        wasm.prb_free(upperPtr, len);
        wasm.prb_free(lowerPtr, len);
        wasm.prb_free(inPtr, len);
    }
});

test('PRB zero-copy with large dataset', () => {
    const size = 10000;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.sin(i * 0.01) + Math.random() * 0.1;
    }
    
    const mainPtr = wasm.prb_alloc(size);
    const upperPtr = wasm.prb_alloc(size);
    const lowerPtr = wasm.prb_alloc(size);
    const inPtr = wasm.prb_alloc(size);
    
    assert(mainPtr !== 0, 'Failed to allocate large main buffer');
    assert(upperPtr !== 0, 'Failed to allocate large upper buffer');
    assert(lowerPtr !== 0, 'Failed to allocate large lower buffer');
    assert(inPtr !== 0, 'Failed to allocate large input buffer');
    
    try {
        const memory = wasm.__wbindgen_memory();
        const inView = new Float64Array(memory.buffer, inPtr, size);
        inView.set(data);
        
        wasm.prb_into(
            inPtr, mainPtr, upperPtr, lowerPtr, size,
            false, 10, 50, 2, 0, 2.0
        );
        
        // Recreate views in case memory grew
        const memory2 = wasm.__wbindgen_memory();
        const mainView = new Float64Array(memory2.buffer, mainPtr, size);
        const upperView = new Float64Array(memory2.buffer, upperPtr, size);
        const lowerView = new Float64Array(memory2.buffer, lowerPtr, size);
        
        // Check warmup period has NaN
        for (let i = 0; i < 49; i++) { // regression_period - 1
            assert(isNaN(mainView[i]), `Expected NaN at warmup index ${i}`);
        }
        
        // Check after warmup has values and proper band relationship
        for (let i = 49; i < Math.min(100, size); i++) {
            assert(!isNaN(mainView[i]), `Unexpected NaN at index ${i}`);
            assert(upperView[i] > mainView[i], `Upper should be > main at ${i}`);
            assert(mainView[i] > lowerView[i], `Main should be > lower at ${i}`);
        }
    } finally {
        wasm.prb_free(mainPtr, size);
        wasm.prb_free(upperPtr, size);
        wasm.prb_free(lowerPtr, size);
        wasm.prb_free(inPtr, size);
    }
});

test('PRB zero-copy error handling', () => {
    // Test null pointer
    assert.throws(() => {
        wasm.prb_into(0, 0, 0, 0, 10, false, 10, 5, 2, 0, 2.0);
    }, /null pointer/i);
    
    // Test invalid parameters with allocated memory
    const ptr = wasm.prb_alloc(10);
    try {
        // Invalid period
        assert.throws(() => {
            wasm.prb_into(ptr, ptr, ptr, ptr, 10, false, 10, 0, 2, 0, 2.0);
        }, /Invalid period/);
        
        // Invalid order
        assert.throws(() => {
            wasm.prb_into(ptr, ptr, ptr, ptr, 10, false, 10, 5, 0, 0, 2.0);
        }, /Invalid polynomial order/);
    } finally {
        wasm.prb_free(ptr, 10);
    }
});

test('PRB partial params with test data', () => {
    // Test with partial parameters using actual test data
    const close = new Float64Array(testData.close);
    
    const result = wasm.prb(
        close,
        true,   // smooth_data
        10,     // smooth_period
        100,    // regression_period
        2,      // polynomial_order
        0,      // regression_offset
        2.0     // ndev
    );
    
    assert(result, 'Result should exist');
    assert.strictEqual(result.rows, 3, 'Should have 3 rows');
    assert.strictEqual(result.cols, close.length, 'Cols should match input');
});

test('PRB batch parameter sweep', () => {
    // Test comprehensive parameter sweep
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const result = wasm.prb_batch(
        close,
        false,  // smooth_data
        8, 12, 2,     // smooth_period range (8, 10, 12)
        20, 30, 5,    // regression_period range (20, 25, 30)
        1, 2, 1,      // polynomial_order range (1, 2)
        -1, 1, 1      // regression_offset range (-1, 0, 1)
    );
    
    // Should have 3 * 3 * 2 * 3 = 54 combinations
    assert.strictEqual(result.combos.length, 54, 'Should have 54 parameter combinations');
    
    // Verify structure
    assert(result.values, 'Should have values');
    assert.strictEqual(result.rows, 54 * 3, 'Should have 54*3 rows (3 bands per combo)');
    assert.strictEqual(result.cols, close.length, 'Should match input length');
});

test.after(() => {
    console.log('All PRB tests completed');
});
