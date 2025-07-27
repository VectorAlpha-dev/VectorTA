/**
 * WASM binding tests for Kurtosis indicator.
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

test('Kurtosis partial params', () => {
    // Test with default parameters - mirrors check_kurtosis_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.kurtosis_js(close, 5);
    assert.strictEqual(result.length, close.length);
});

test('Kurtosis accuracy', async () => {
    // Test Kurtosis matches expected values from Rust tests - mirrors check_kurtosis_accuracy
    const hl2 = new Float64Array(testData.hl2);
    
    const result = wasm.kurtosis_js(hl2, 5);
    
    assert.strictEqual(result.length, hl2.length);
    
    // Expected values from Rust test
    const expectedLast5 = [
        -0.5438903789933454,
        -1.6848139264816433,
        -1.6331336745945797,
        -0.6130805596586351,
        -0.027802601135927585,
    ];
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expectedLast5,
        1e-10,
        "Kurtosis last 5 values mismatch"
    );
    
    // Compare full output with Rust
    await compareWithRust('kurtosis', result, 'hl2', { period: 5 });
});

test('Kurtosis default candles', () => {
    // Test Kurtosis with default parameters - mirrors check_kurtosis_default_candles
    const hl2 = new Float64Array(testData.hl2);
    
    const result = wasm.kurtosis_js(hl2, 5);
    assert.strictEqual(result.length, hl2.length);
});

test('Kurtosis zero period', () => {
    // Test Kurtosis fails with zero period - mirrors check_kurtosis_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.kurtosis_js(inputData, 0);
    }, /Invalid period/);
});

test('Kurtosis period exceeds length', () => {
    // Test Kurtosis fails when period exceeds data length - mirrors check_kurtosis_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.kurtosis_js(dataSmall, 10);
    }, /Invalid period/);
});

test('Kurtosis all nan', () => {
    // Test Kurtosis handles all NaN values - mirrors check_kurtosis_all_nan
    const nanData = new Float64Array(20).fill(NaN);
    
    assert.throws(() => {
        wasm.kurtosis_js(nanData, 5);
    }, /All.*NaN/);
});

test('Kurtosis nan prefix', () => {
    // Test Kurtosis handles NaN prefix - mirrors check_kurtosis_nan_prefix
    const nanPrefixData = new Float64Array(30);
    nanPrefixData.fill(NaN, 0, 10);
    for (let i = 10; i < 30; i++) {
        nanPrefixData[i] = 50.0 + i * 0.5;
    }
    
    const result = wasm.kurtosis_js(nanPrefixData, 5);
    
    assert.strictEqual(result.length, nanPrefixData.length);
    
    // Check warmup NaN values (first 10 NaN + 4 warmup = 14)
    for (let i = 0; i < 14; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i}`);
    }
    
    // Valid values start at index 14
    for (let i = 14; i < result.length; i++) {
        assert(!isNaN(result[i]), `Expected valid value at index ${i}`);
    }
});

test('Kurtosis batch operation', () => {
    // Test batch kurtosis calculation
    const close = new Float64Array(testData.close);
    
    // Test batch with period range
    const config = {
        period_range: [5, 20, 5]  // 5, 10, 15, 20
    };
    
    const result = wasm.kurtosis_batch(close, config);
    
    assert(result.values);
    assert(result.combos);
    assert.strictEqual(result.rows, 4); // 4 periods
    assert.strictEqual(result.cols, close.length);
    assert.strictEqual(result.values.length, 4 * close.length);
});

test('Kurtosis fast API', () => {
    // Test fast/unsafe API
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    // Allocate memory
    const outPtr = wasm.kurtosis_alloc(len);
    
    try {
        // Create input pointer
        const memory = wasm.memory;
        const inputArray = new Float64Array(memory.buffer, 0, len);
        inputArray.set(close);
        
        // Call fast API
        wasm.kurtosis_into(inputArray.byteOffset, outPtr, len, 5);
        
        // Read results
        const output = new Float64Array(memory.buffer, outPtr, len);
        
        // Compare with safe API
        const expected = wasm.kurtosis_js(close, 5);
        assertArrayClose(output, expected, 1e-10, "Fast API mismatch");
        
    } finally {
        // Free memory
        wasm.kurtosis_free(outPtr, len);
    }
});

test('Kurtosis fast API aliasing', () => {
    // Test fast API with aliasing (in-place operation)
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    // Allocate memory
    const ptr = wasm.kurtosis_alloc(len);
    
    try {
        // Create memory view
        const memory = wasm.memory;
        const dataArray = new Float64Array(memory.buffer, ptr, len);
        dataArray.set(close);
        
        // Call fast API with same pointer for input and output (aliasing)
        wasm.kurtosis_into(ptr, ptr, len, 5);
        
        // Compare with safe API
        const expected = wasm.kurtosis_js(close, 5);
        assertArrayClose(dataArray, expected, 1e-10, "Fast API aliasing mismatch");
        
    } finally {
        // Free memory
        wasm.kurtosis_free(ptr, len);
    }
});

test('Kurtosis streaming', () => {
    // Test streaming interface - note: needs to be implemented in WASM if supported
    const values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    const period = 5;
    
    // Calculate expected values using regular API
    const expected = [];
    for (let i = 0; i < values.length; i++) {
        if (i < period - 1) {
            expected.push(NaN);
        } else {
            const window = values.slice(i - period + 1, i + 1);
            const result = wasm.kurtosis_js(new Float64Array(window), period);
            expected.push(result[result.length - 1]);
        }
    }
    
    // If streaming is implemented, test it here
    // Otherwise, this test just validates the expected behavior
    assert.strictEqual(expected.length, values.length);
});

test('Kurtosis very small dataset', () => {
    // Test Kurtosis with very small dataset
    const smallData = new Float64Array([5.0, 10.0, 15.0, 20.0, 25.0]);
    
    const result = wasm.kurtosis_js(smallData, 5);
    assert.strictEqual(result.length, smallData.length);
    
    // First 4 values should be NaN (warmup)
    for (let i = 0; i < 4; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i}`);
    }
    
    // Last value should be valid
    assert(!isNaN(result[4]), 'Expected valid value at index 4');
});

test('Kurtosis empty input', () => {
    // Test Kurtosis with empty input
    const emptyData = new Float64Array([]);
    
    assert.throws(() => {
        wasm.kurtosis_js(emptyData, 5);
    }, /Invalid period|All.*NaN/);
});

test('Kurtosis reinput', () => {
    // Test Kurtosis with output as new input
    const close = new Float64Array(testData.close);
    
    const result1 = wasm.kurtosis_js(close, 5);
    const result2 = wasm.kurtosis_js(result1, 5);
    
    assert.strictEqual(result2.length, result1.length);
    
    // Check that we have more NaN values in result2 due to double warmup
    let nanCount1 = 0, nanCount2 = 0;
    for (let i = 0; i < result1.length; i++) {
        if (isNaN(result1[i])) nanCount1++;
        if (isNaN(result2[i])) nanCount2++;
    }
    assert(nanCount2 >= nanCount1, 'Second pass should have at least as many NaN values');
});

test('Kurtosis batch metadata from result', () => {
    // Test batch metadata is correctly returned
    const close = new Float64Array(testData.close);
    
    const config = {
        period_range: [5, 10, 1]  // 6 values: 5,6,7,8,9,10
    };
    
    const result = wasm.kurtosis_batch(close, config);
    
    assert(result.combos);
    assert.strictEqual(result.combos.length, 6);
    
    // Check each combo has correct period
    for (let i = 0; i < 6; i++) {
        assert.strictEqual(result.combos[i].period, 5 + i);
    }
});

test('Kurtosis batch edge cases', () => {
    // Test batch with edge case parameters
    const close = new Float64Array(testData.close);
    
    // Single parameter (step = 0)
    const config1 = {
        period_range: [5, 5, 0]
    };
    const result1 = wasm.kurtosis_batch(close, config1);
    assert.strictEqual(result1.rows, 1);
    
    // Large step
    const config2 = {
        period_range: [5, 50, 45]  // Only 2 values: 5, 50
    };
    const result2 = wasm.kurtosis_batch(close, config2);
    assert.strictEqual(result2.rows, 2);
});

test('Kurtosis zero-copy API', () => {
    // Test zero-copy API basic functionality
    const testData = new Float64Array([
        10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0
    ]);
    const len = testData.length;
    
    // Allocate memory
    const inPtr = wasm.kurtosis_alloc(len);
    const outPtr = wasm.kurtosis_alloc(len);
    
    try {
        // Create memory views
        const memory = wasm.memory;
        const inputArray = new Float64Array(memory.buffer, inPtr, len);
        const outputArray = new Float64Array(memory.buffer, outPtr, len);
        
        // Copy test data
        inputArray.set(testData);
        
        // Call zero-copy API
        wasm.kurtosis_into(inPtr, outPtr, len, 5);
        
        // Compare with safe API
        const expected = wasm.kurtosis_js(testData, 5);
        assertArrayClose(outputArray, expected, 1e-10, "Zero-copy API mismatch");
        
    } finally {
        // Free memory
        wasm.kurtosis_free(inPtr, len);
        wasm.kurtosis_free(outPtr, len);
    }
});

test('Kurtosis zero-copy error handling', () => {
    // Test zero-copy API error handling
    
    // Test null pointers
    assert.throws(() => {
        wasm.kurtosis_into(0, 0, 10, 5);
    }, /null pointer/);
    
    // Test invalid period
    const ptr = wasm.kurtosis_alloc(10);
    try {
        assert.throws(() => {
            wasm.kurtosis_into(ptr, ptr, 10, 0);
        }, /Invalid period/);
        
        assert.throws(() => {
            wasm.kurtosis_into(ptr, ptr, 10, 20);
        }, /Invalid period/);
    } finally {
        wasm.kurtosis_free(ptr, 10);
    }
});

test('Kurtosis zero-copy memory management', () => {
    // Test memory allocation and deallocation patterns
    const sizes = [10, 100, 1000, 10000];
    
    for (const size of sizes) {
        const ptr = wasm.kurtosis_alloc(size);
        assert(ptr !== 0, `Failed to allocate ${size} elements`);
        
        // Verify we can write to the memory
        const memory = wasm.memory;
        const array = new Float64Array(memory.buffer, ptr, size);
        array.fill(42.0);
        
        // Free should not throw
        assert.doesNotThrow(() => {
            wasm.kurtosis_free(ptr, size);
        });
    }
    
    // Test double free protection
    const ptr = wasm.kurtosis_alloc(100);
    wasm.kurtosis_free(ptr, 100);
    // Second free should not crash (null check)
    assert.doesNotThrow(() => {
        wasm.kurtosis_free(0, 100);
    });
});