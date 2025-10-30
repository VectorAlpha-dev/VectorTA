/**
 * WASM binding tests for HMA indicator.
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
    // Load WASM wrappers and ensure the wasm instance is bound for both ESM and nodejs targets
    try {
        // First try pkg output. Handle both ESM and nodejs (CJS) targets.
        const pkgPath = path.join(__dirname, '../../pkg/my_project.js');
        const pkgImportPath = process.platform === 'win32'
            ? 'file:///' + pkgPath.replace(/\\/g, '/')
            : pkgPath;
        const mod = await import(pkgImportPath);
        const candidate = (mod && mod.default) ? mod.default : mod;
        if (candidate && (typeof candidate.hma_js === 'function' || candidate.__wasm)) {
            if (candidate.__wasm) {
                // Nodejs target: bind wrappers to underlying wasm
                const wrappersPath = path.join(__dirname, '../../pkg/my_project_bg.js');
                const wrappersImportPath = process.platform === 'win32'
                    ? 'file:///' + wrappersPath.replace(/\\/g, '/')
                    : wrappersPath;
                const wrappers = await import(wrappersImportPath);
                if (typeof wrappers.__wbg_set_wasm === 'function') {
                    wrappers.__wbg_set_wasm(candidate.__wasm);
                }
                wasm = wrappers;
            } else {
                wasm = candidate; // ESM target re-exports wrappers
            }
        } else {
            throw new Error('pkg module missing expected exports');
        }
    } catch (e1) {
        // Fallback to local test build (CJS with wrappers baked-in)
        try {
            const { pathToFileURL } = await import('url');
            const localPath = path.join(__dirname, 'my_project.js');
            // Use createRequire in the test runtime to load CJS under ESM
            const { createRequire } = await import('module');
            const require = createRequire(pathToFileURL(import.meta.url));
            wasm = require(localPath);
        } catch (e2) {
            console.error('Failed to load WASM module. pkg error:', e1);
            console.error('Local fallback error:', e2);
            throw e2;
        }
    }

    testData = loadTestData();
});

test('HMA partial params', () => {
    // Test with default parameters - mirrors check_hma_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.hma_js(close, 5);
    assert.strictEqual(result.length, close.length);
});

test('HMA accuracy', async () => {
    // Test HMA matches expected values from Rust tests - mirrors check_hma_accuracy
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.hma;
    
    const result = wasm.hma_js(close, expected.defaultParams.period);
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-3,  // Using 1e-3 as in Rust test
        "HMA last 5 values mismatch"
    );
    
    // Compare full output with Rust
    // await compareWithRust('hma', result, 'close', expected.defaultParams);
});

test('HMA default candles', () => {
    // Test HMA with default parameters - mirrors check_hma_default_candles
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.hma;
    
    const result = wasm.hma_js(close, expected.defaultParams.period);
    assert.strictEqual(result.length, close.length);
});

test('HMA zero period', () => {
    // Test HMA fails with zero period - mirrors check_hma_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.hma_js(inputData, 0);
    }, /Invalid period|Invalid or insufficient data/);
});

test('HMA period exceeds length', () => {
    // Test HMA fails when period exceeds data length - mirrors check_hma_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.hma_js(dataSmall, 10);
    }, /Invalid|insufficient data/);
});

test('HMA very small dataset', () => {
    // Test HMA with very small dataset - mirrors check_hma_very_small_dataset
    const dataSingle = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.hma_js(dataSingle, 5);
    }, /Invalid|insufficient data/);
});

test('HMA empty input', () => {
    // Test HMA with empty input - mirrors check_hma_empty_input
    const dataEmpty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.hma_js(dataEmpty, 5);
    }, /All NaN|Invalid|insufficient data/);
});

test('HMA all NaN', () => {
    // Test HMA with all NaN input
    const data = new Float64Array([NaN, NaN, NaN, NaN, NaN]);
    
    assert.throws(() => {
        wasm.hma_js(data, 3);
    }, /All NaN|Invalid/);
});

test('HMA NaN handling', () => {
    // Test HMA handling of NaN values - mirrors check_hma_nan_handling
    const close = new Float64Array(testData.close);
    const period = 5;
    
    const result = wasm.hma_js(close, period);
    
    assert.strictEqual(result.length, close.length);
    
    // Calculate expected warmup period
    const sqrtPeriod = Math.floor(Math.sqrt(period));
    const warmupPeriod = period + sqrtPeriod - 2;
    
    // After warmup period, no NaN values should exist
    if (result.length > warmupPeriod) {
        for (let i = warmupPeriod; i < result.length; i++) {
            assert(!isNaN(result[i]), `Unexpected NaN at index ${i}`);
        }
    }
});

test('HMA warmup period calculation', () => {
    // Test that warmup period is correctly calculated
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const testCases = [
        { period: 3, expectedWarmup: 3 + 1 - 2 },  // sqrt(3) = 1, warmup = 2
        { period: 5, expectedWarmup: 5 + 2 - 2 },  // sqrt(5) = 2, warmup = 5
        { period: 10, expectedWarmup: 10 + 3 - 2 }, // sqrt(10) = 3, warmup = 11
        { period: 16, expectedWarmup: 16 + 4 - 2 }, // sqrt(16) = 4, warmup = 18
    ];
    
    for (const { period, expectedWarmup } of testCases) {
        const result = wasm.hma_js(close, period);
        
        // Check NaN values up to warmup period
        for (let i = 0; i < expectedWarmup && i < result.length; i++) {
            assert(isNaN(result[i]), `Expected NaN at index ${i} for period=${period}`);
        }
        
        // Check valid values after warmup
        if (expectedWarmup < result.length) {
            assert(!isNaN(result[expectedWarmup]), 
                `Expected valid value at index ${expectedWarmup} for period=${period}`);
        }
    }
});

test('HMA batch', () => {
    // Test HMA batch computation
    const close = new Float64Array(testData.close);
    
    // Test period range 3-9 step 2
    const period_start = 3;
    const period_end = 9;
    const period_step = 2;  // periods: 3, 5, 7, 9
    
    const batch_result = wasm.hma_batch_js(
        close, 
        period_start, period_end, period_step
    );
    const metadata = wasm.hma_batch_metadata_js(
        period_start, period_end, period_step
    );
    
    // Metadata should contain period values
    assert.strictEqual(metadata.length, 4);  // 4 periods
    
    // Batch result should contain all individual results flattened
    assert.strictEqual(batch_result.length, 4 * close.length);  // 4 periods
    
    // Verify each row matches individual calculation
    let row_idx = 0;
    for (const period of [3, 5, 7, 9]) {
        const individual_result = wasm.hma_js(close, period);
        
        // Extract row from batch result
        const row_start = row_idx * close.length;
        const row = batch_result.slice(row_start, row_start + close.length);
        
        assertArrayClose(row, individual_result, 1e-9, `Period ${period}`);
        row_idx++;
    }
});

test('HMA batch single period', () => {
    // Test batch with single period matches single computation
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.hma;
    
    const batch_result = wasm.hma_batch_js(close, 5, 5, 0);
    const single_result = wasm.hma_js(close, 5);
    
    assert.strictEqual(batch_result.length, close.length);
    assertArrayClose(batch_result, single_result, 1e-10, 'Batch vs single');
    
    // Check matches expected values
    const last5 = batch_result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-3,
        "HMA batch last 5 values mismatch"
    );
});

test('HMA different periods', () => {
    // Test HMA with different period values
    const close = new Float64Array(testData.close);
    
    // Test various period values
    for (const period of [3, 5, 10, 20]) {
        const result = wasm.hma_js(close, period);
        assert.strictEqual(result.length, close.length);
        
        // Calculate expected warmup period
        const sqrtPeriod = Math.floor(Math.sqrt(period));
        const warmup = period + sqrtPeriod - 2;
        
        // Verify no NaN after warmup period
        for (let i = warmup; i < result.length; i++) {
            assert(!isNaN(result[i]), `Unexpected NaN at index ${i} for period=${period}`);
        }
    }
});

test('HMA batch metadata', () => {
    // Test metadata function returns correct parameter combinations
    const metadata = wasm.hma_batch_metadata_js(5, 15, 5);
    
    // Should have 3 periods: 5, 10, 15
    assert.strictEqual(metadata.length, 3);
    assert.strictEqual(metadata[0], 5);
    assert.strictEqual(metadata[1], 10);
    assert.strictEqual(metadata[2], 15);
});

test('HMA zero half', () => {
    // Test HMA fails when period/2 is zero
    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
    
    // Period=1 would result in half=0
    assert.throws(() => {
        wasm.hma_js(data, 1);
    }, /Cannot calculate half of period|Invalid|insufficient/);
});

test('HMA small periods', () => {
    // Test HMA with small periods
    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    
    // Period=2 should work (sqrt(2) ≈ 1.4 → 1)
    const result = wasm.hma_js(data, 2);
    assert.strictEqual(result.length, data.length);
    
    // Check warmup period for period=2
    // warmup = period + sqrt(period) - 2 = 2 + 1 - 2 = 1
    assert(isNaN(result[0]));
    assert(!isNaN(result[1])); // Should have valid value from index 1
});

test('HMA not enough valid data', () => {
    // Test HMA with insufficient valid data after NaN prefix
    const data = new Float64Array([NaN, NaN, 1.0, 2.0, 3.0]);
    
    // With period=4, needs at least 4 valid values
    assert.throws(() => {
        wasm.hma_js(data, 4);
    }, /Invalid|insufficient/);
});

test('HMA batch edge cases', () => {
    // Test edge cases for batch processing
    const close = new Float64Array(testData.close.slice(0, 100)); // Use smaller dataset for speed
    
    // Step larger than range (should get single period)
    const batch1 = wasm.hma_batch_js(close, 5, 7, 10);
    const metadata1 = wasm.hma_batch_metadata_js(5, 7, 10);
    assert.strictEqual(metadata1.length, 1); // Only period=5
    assert.strictEqual(batch1.length, close.length);
    
    // Step = 0 (should get single period)
    const batch2 = wasm.hma_batch_js(close, 5, 5, 0);
    const metadata2 = wasm.hma_batch_metadata_js(5, 5, 0);
    assert.strictEqual(metadata2.length, 1);
    assert.strictEqual(batch2.length, close.length);
});

test('HMA consistency across periods', () => {
    // Test that HMA produces consistent results for different periods
    // HMA is designed to reduce lag while maintaining smoothness
    // Due to its unique calculation method, variance relationships may differ from simple moving averages
    const close = new Float64Array(testData.close.slice(0, 200)); // Use more data for better statistics
    
    const hma5 = wasm.hma_js(close, 5);
    const hma10 = wasm.hma_js(close, 10);
    const hma20 = wasm.hma_js(close, 20);
    
    // Calculate mean absolute difference from previous value (measures responsiveness)
    const responsiveness = (arr) => {
        const validValues = [];
        for (let i = 1; i < arr.length; i++) {
            if (!isNaN(arr[i]) && !isNaN(arr[i-1])) {
                validValues.push(Math.abs(arr[i] - arr[i-1]));
            }
        }
        if (validValues.length === 0) return 0;
        return validValues.reduce((a, b) => a + b, 0) / validValues.length;
    };
    
    const resp5 = responsiveness(hma5);
    const resp10 = responsiveness(hma10);
    const resp20 = responsiveness(hma20);
    
    // Smaller period should be more responsive (larger average change)
    assert(resp5 > resp10, `HMA(5) responsiveness ${resp5} should be > HMA(10) responsiveness ${resp10}`);
    assert(resp10 > resp20, `HMA(10) responsiveness ${resp10} should be > HMA(20) responsiveness ${resp20}`);
    
    // Also verify all produce valid results
    assert(hma5.filter(v => !isNaN(v)).length > 0, "HMA(5) should produce valid values");
    assert(hma10.filter(v => !isNaN(v)).length > 0, "HMA(10) should produce valid values");
    assert(hma20.filter(v => !isNaN(v)).length > 0, "HMA(20) should produce valid values");
});

test('HMA with specific data patterns', () => {
    // Test with constant data - HMA should equal the constant
    const constantData = new Float64Array(50);
    constantData.fill(100.0);
    
    const result = wasm.hma_js(constantData, 5);
    const validResult = result.filter(v => !isNaN(v));
    
    // All valid values should be close to 100
    for (const val of validResult) {
        assertClose(val, 100.0, 1e-10, "HMA of constant should equal constant");
    }
    
    // Test with linear trend
    const linearData = new Float64Array(50);
    for (let i = 0; i < 50; i++) {
        linearData[i] = i + 1;
    }
    
    const linearResult = wasm.hma_js(linearData, 5);
    assert.strictEqual(linearResult.length, linearData.length);
});

// ==================== Zero-Copy API Tests ====================

test('HMA zero-copy API basic', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const period = 5;
    
    // Allocate buffer
    const ptr = wasm.hma_alloc(data.length);
    assert(ptr !== 0, 'Failed to allocate memory');
    
    try {
        // Create view into WASM memory
        const memView = new Float64Array(
            wasm.__wasm.memory.buffer,
            ptr,
            data.length
        );
        
        // Copy data into WASM memory
        memView.set(data);
        
        // Compute HMA in-place
        wasm.hma_into(ptr, ptr, data.length, period);
        
        // Verify results match regular API
        const regularResult = wasm.hma_js(data, period);
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                continue; // Both NaN is OK
            }
            assert(Math.abs(regularResult[i] - memView[i]) < 1e-10,
                   `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${memView[i]}`);
        }
    } finally {
        // Always free memory
        wasm.hma_free(ptr, data.length);
    }
});

test('HMA zero-copy with large dataset', () => {
    const size = 10000;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.sin(i * 0.01) + Math.random() * 0.1;
    }
    
    const ptr = wasm.hma_alloc(size);
    assert(ptr !== 0, 'Failed to allocate large buffer');
    
    try {
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        memView.set(data);
        
        wasm.hma_into(ptr, ptr, size, 10);
        
        // Recreate view in case memory grew
        const memView2 = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        
        // Check warmup period (period=10, sqrt=3, warmup = 10+3-2 = 11)
        const warmup = 11;
        for (let i = 0; i < warmup; i++) {
            assert(isNaN(memView2[i]), `Expected NaN at warmup index ${i}`);
        }
        
        // Check after warmup has values
        for (let i = warmup; i < Math.min(warmup + 10, size); i++) {
            assert(!isNaN(memView2[i]), `Unexpected NaN at index ${i}`);
        }
    } finally {
        wasm.hma_free(ptr, size);
    }
});

test('HMA zero-copy error handling', () => {
    // Test null pointer
    assert.throws(() => {
        wasm.hma_into(0, 0, 10, 5);
    }, /null pointer/i);
    
    // Test invalid parameters with allocated memory
    const ptr = wasm.hma_alloc(10);
    try {
        // Invalid period
        assert.throws(() => {
            wasm.hma_into(ptr, ptr, 10, 0);
        }, /Invalid/);
        
        // Period too large
        assert.throws(() => {
            wasm.hma_into(ptr, ptr, 10, 20);
        }, /Invalid/);
    } finally {
        wasm.hma_free(ptr, 10);
    }
});

test('HMA zero-copy memory management', () => {
    // Allocate and free multiple times to ensure no leaks
    const sizes = [100, 1000, 5000];
    
    for (const size of sizes) {
        const ptr = wasm.hma_alloc(size);
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
        wasm.hma_free(ptr, size);
    }
});

test('HMA zero-copy batch processing', () => {
    const size = 100;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.sin(i * 0.1) + 50;
    }
    
    const periods = [3, 5, 7, 9];
    const totalSize = size * periods.length;
    
    const inPtr = wasm.hma_alloc(size);
    const outPtr = wasm.hma_alloc(totalSize);
    
    try {
        // Copy input data
        const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, size);
        inView.set(data);
        
        // Process batch
        const rowsProcessed = wasm.hma_batch_into(
            inPtr, outPtr, size,
            3, 9, 2  // period_start, period_end, period_step
        );
        
        assert.strictEqual(rowsProcessed, periods.length);
        
        // Verify results
        const outView = new Float64Array(wasm.__wasm.memory.buffer, outPtr, totalSize);
        
        for (let i = 0; i < periods.length; i++) {
            const rowStart = i * size;
            const row = Array.from(outView.slice(rowStart, rowStart + size));
            
            // Compare with regular API
            const expected = wasm.hma_js(data, periods[i]);
            for (let j = 0; j < size; j++) {
                if (isNaN(expected[j]) && isNaN(row[j])) continue;
                assertClose(row[j], expected[j], 1e-10, 
                    `Batch row ${i} (period=${periods[i]}) at index ${j}`);
            }
        }
    } finally {
        wasm.hma_free(inPtr, size);
        wasm.hma_free(outPtr, totalSize);
    }
});

test('HMA unified batch API', () => {
    // Test the new unified batch API
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const config = {
        period_range: [5, 15, 5]  // periods: 5, 10, 15
    };
    
    const result = wasm.hma_batch(close, config);
    
    assert(result.values, 'Should have values array');
    assert(result.combos, 'Should have combos array');
    assert(typeof result.rows === 'number', 'Should have rows count');
    assert(typeof result.cols === 'number', 'Should have cols count');
    
    assert.strictEqual(result.rows, 3);
    assert.strictEqual(result.cols, 50);
    assert.strictEqual(result.combos.length, 3);
    assert.strictEqual(result.values.length, 150);
    
    // Check parameter combinations
    assert.strictEqual(result.combos[0].period, 5);
    assert.strictEqual(result.combos[1].period, 10);
    assert.strictEqual(result.combos[2].period, 15);
});

test.after(() => {
    console.log('HMA WASM tests completed');
});
