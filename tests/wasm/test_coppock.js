/**
 * WASM binding tests for COPPOCK indicator.
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

test('COPPOCK accuracy', () => {
    // Test COPPOCK matches expected values from Rust tests
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.coppock;
    
    const result = wasm.coppock_js(
        close,
        expected.defaultParams.short,
        expected.defaultParams.long,
        expected.defaultParams.ma,
        expected.defaultParams.ma_type
    );
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-8,
        "COPPOCK last 5 values mismatch"
    );
});

test('COPPOCK default parameters', () => {
    // Test with default parameters
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.coppock;
    
    const result = wasm.coppock_js(
        close, 
        expected.defaultParams.short,
        expected.defaultParams.long,
        expected.defaultParams.ma,
        expected.defaultParams.ma_type
    );
    assert.strictEqual(result.length, close.length);
    
    // Calculate warmup period: max(short, long) + (ma - 1)
    const warmup = Math.max(expected.defaultParams.short, expected.defaultParams.long) + (expected.defaultParams.ma - 1);
    
    // After warmup period, values should be present
    let foundNonNaN = false;
    for (let i = warmup; i < result.length; i++) {
        if (!isNaN(result[i])) {
            foundNonNaN = true;
            break;
        }
    }
    assert(foundNonNaN, "No valid values found after warmup");
});

test('COPPOCK zero period', () => {
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.coppock_js(inputData, 0, 14, 10, "wma");
    }, /Invalid period/);
});

test('COPPOCK period exceeds length', () => {
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.coppock_js(dataSmall, 11, 14, 10, "wma");
    }, /Invalid period|Not enough valid data/);
});

test('COPPOCK very small dataset', () => {
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.coppock_js(singlePoint, 11, 14, 10, "wma");
    }, /Invalid period|Not enough valid data/);
});

test('COPPOCK empty input', () => {
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.coppock_js(empty, 11, 14, 10, "wma");
    }, /Empty data/);
});

test('COPPOCK all NaN input', () => {
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.coppock_js(allNaN, 11, 14, 10, "wma");
    }, /All values are NaN/);
});

test('COPPOCK batch single parameter', () => {
    // Test batch with single parameter combination
    const close = new Float64Array(testData.close);
    
    const batchResult = wasm.coppock_batch(close, {
        short_range: [11, 11, 0],
        long_range: [14, 14, 0],
        ma_range: [10, 10, 0],
        ma_type: "wma"
    });
    
    // Should match single calculation
    const singleResult = wasm.coppock_js(close, 11, 14, 10, "wma");
    
    assert.strictEqual(batchResult.values.length, singleResult.length);
    assertArrayClose(batchResult.values, singleResult, 1e-10, "Batch vs single mismatch");
});

test('COPPOCK batch multiple parameters', () => {
    // Test batch with multiple parameter combinations
    const close = new Float64Array(testData.close.slice(0, 100)); // Use smaller dataset for speed
    
    // Multiple parameters
    const batchResult = wasm.coppock_batch(close, {
        short_range: [10, 12, 2],    // 10, 12
        long_range: [14, 16, 2],     // 14, 16
        ma_range: [8, 10, 2],        // 8, 10
        ma_type: "wma"
    });
    
    // Should have 2 * 2 * 2 = 8 combinations
    assert.strictEqual(batchResult.combos.length, 8);
    assert.strictEqual(batchResult.rows, 8);
    assert.strictEqual(batchResult.cols, 100);
    assert.strictEqual(batchResult.values.length, 8 * 100);
    
    // Verify structure
    for (let combo = 0; combo < batchResult.combos.length; combo++) {
        const short_period = batchResult.combos[combo].short_roc_period;
        const long_period = batchResult.combos[combo].long_roc_period;
        const ma_period = batchResult.combos[combo].ma_period;
        assert(batchResult.combos[combo].ma_type === "wma", "MA type should be set");
        
        const rowStart = combo * 100;
        const rowData = batchResult.values.slice(rowStart, rowStart + 100);
        
        // Warmup should be NaN
        const warmup = Math.max(short_period, long_period) + ma_period - 1;
        for (let i = 0; i < Math.min(warmup, 100); i++) {
            assert(isNaN(rowData[i]), `Expected NaN at warmup index ${i}`);
        }
    }
});

test('COPPOCK zero-copy API', () => {
    // Test the fast zero-copy API
    const data = new Float64Array([
        10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
        110, 120, 130, 140, 150, 160, 170, 180, 190, 200
    ]);
    
    const short_period = 5;
    const long_period = 7;
    const ma_period = 3;
    
    // Allocate buffer
    const ptr = wasm.coppock_alloc(data.length);
    assert(ptr !== 0, 'Failed to allocate memory');
    
    // Create view into WASM memory
    const memView = new Float64Array(
        wasm.__wasm.memory.buffer,
        ptr,
        data.length
    );
    
    // Copy data into WASM memory
    memView.set(data);
    
    // Compute COPPOCK in-place
    try {
        wasm.coppock_into(ptr, ptr, data.length, short_period, long_period, ma_period, "wma");
        
        // Verify results match regular API
        const regularResult = wasm.coppock_js(data, short_period, long_period, ma_period, "wma");
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                continue; // Both NaN is OK
            }
            assert(Math.abs(regularResult[i] - memView[i]) < 1e-10,
                   `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${memView[i]}`);
        }
    } finally {
        // Always free memory
        wasm.coppock_free(ptr, data.length);
    }
});

test('COPPOCK zero-copy error handling', () => {
    // Test null pointer
    assert.throws(() => {
        wasm.coppock_into(0, 0, 10, 11, 14, 10, "wma");
    }, /null pointer|Null pointer/i);
    
    // Test invalid parameters with allocated memory
    const ptr = wasm.coppock_alloc(10);
    try {
        // Invalid period
        assert.throws(() => {
            wasm.coppock_into(ptr, ptr, 10, 0, 14, 10, "wma");
        }, /Invalid period/);
        
        // Period exceeds length
        assert.throws(() => {
            wasm.coppock_into(ptr, ptr, 10, 20, 14, 10, "wma");
        }, /Period exceeds data length/);
    } finally {
        wasm.coppock_free(ptr, 10);
    }
});

test('COPPOCK supported MA types', () => {
    // Test that different MA types work
    const close = new Float64Array(testData.close.slice(0, 100));
    
    // Only test MA types that are actually supported
    const maTypes = ['sma', 'ema', 'wma'];
    const results = {};
    
    for (const maType of maTypes) {
        const result = wasm.coppock_js(close, 11, 14, 10, maType);
        assert.strictEqual(result.length, close.length);
        
        // Calculate warmup period
        const warmup = Math.max(11, 14) + (10 - 1);
        
        // After warmup, should have values
        let foundValid = false;
        for (let i = warmup; i < result.length; i++) {
            if (!isNaN(result[i])) {
                foundValid = true;
                break;
            }
        }
        assert(foundValid, `No valid values for MA type ${maType}`);
        results[maType] = result;
    }
    
    // Verify different MA types produce different results
    let allSame = true;
    for (let i = 30; i < 100; i++) {
        if (!isNaN(results['sma'][i]) && !isNaN(results['ema'][i])) {
            if (Math.abs(results['sma'][i] - results['ema'][i]) > 1e-10) {
                allSame = false;
                break;
            }
        }
    }
    assert(!allSame, "Different MA types should produce different results");
});

test('COPPOCK memory leak prevention', () => {
    // Allocate and free multiple times to ensure no leaks
    const sizes = [100, 1000, 10000];
    
    for (const size of sizes) {
        const ptr = wasm.coppock_alloc(size);
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
        wasm.coppock_free(ptr, size);
    }
});

test.after(() => {
    console.log('COPPOCK WASM tests completed');
});
