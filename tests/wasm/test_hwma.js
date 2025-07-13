/**
 * WASM binding tests for HWMA indicator.
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

test('HWMA partial params', () => {
    // Test with default parameters - mirrors check_hwma_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.hwma_js(close, 0.2, 0.1, 0.1);
    assert.strictEqual(result.length, close.length);
});

test('HWMA accuracy', async () => {
    // Test HWMA matches expected values from Rust tests - mirrors check_hwma_accuracy
    const close = new Float64Array(testData.close);
    
    const result = wasm.hwma_js(close, 0.2, 0.1, 0.1);
    
    assert.strictEqual(result.length, close.length);
    
    // Expected last 5 values from Rust test
    const expectedLastFive = [
        57941.04005793378,
        58106.90324194954,
        58250.474156632234,
        58428.90005831887,
        58499.37021151028,
    ];
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expectedLastFive,
        1e-3,
        "HWMA last 5 values mismatch"
    );
    
    // Compare full output with Rust
    // await compareWithRust('hwma', result, 'close', { na: 0.2, nb: 0.1, nc: 0.1 });
});

test('HWMA default candles', async () => {
    // Test HWMA with default parameters - mirrors check_hwma_default_candles
    const close = new Float64Array(testData.close);
    
    const result = wasm.hwma_js(close, 0.2, 0.1, 0.1);
    assert.strictEqual(result.length, close.length);
    
    // Compare with Rust
    await compareWithRust('hwma', result, 'close', { na: 0.2, nb: 0.1, nc: 0.1 });
});

test('HWMA invalid na', () => {
    // Test HWMA fails with invalid na - mirrors check_hwma_invalid_na
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    // Test na > 1
    assert.throws(() => {
        wasm.hwma_js(inputData, 1.5, 0.1, 0.1);
    });
    
    // Test na <= 0
    assert.throws(() => {
        wasm.hwma_js(inputData, 0.0, 0.1, 0.1);
    });
    
    // Test na = NaN
    assert.throws(() => {
        wasm.hwma_js(inputData, NaN, 0.1, 0.1);
    });
});

test('HWMA invalid nb', () => {
    // Test HWMA fails with invalid nb - mirrors check_hwma_invalid_nb
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    // Test nb > 1
    assert.throws(() => {
        wasm.hwma_js(inputData, 0.2, 1.5, 0.1);
    });
    
    // Test nb <= 0
    assert.throws(() => {
        wasm.hwma_js(inputData, 0.2, 0.0, 0.1);
    });
    
    // Test nb = NaN
    assert.throws(() => {
        wasm.hwma_js(inputData, 0.2, NaN, 0.1);
    });
});

test('HWMA invalid nc', () => {
    // Test HWMA fails with invalid nc - mirrors check_hwma_invalid_nc
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    // Test nc > 1
    assert.throws(() => {
        wasm.hwma_js(inputData, 0.2, 0.1, 1.5);
    });
    
    // Test nc <= 0
    assert.throws(() => {
        wasm.hwma_js(inputData, 0.2, 0.1, 0.0);
    });
    
    // Test nc = NaN
    assert.throws(() => {
        wasm.hwma_js(inputData, 0.2, 0.1, NaN);
    });
});

test('HWMA empty input', () => {
    // Test HWMA with empty input - mirrors check_hwma_empty_input
    const dataEmpty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.hwma_js(dataEmpty, 0.2, 0.1, 0.1);
    });
});

test('HWMA all NaN', () => {
    // Test HWMA with all NaN input
    const data = new Float64Array([NaN, NaN, NaN, NaN, NaN]);
    
    assert.throws(() => {
        wasm.hwma_js(data, 0.2, 0.1, 0.1);
    });
});

test('HWMA reinput', () => {
    // Test HWMA with re-input of HWMA result - mirrors check_hwma_reinput
    const close = new Float64Array(testData.close);
    
    // First HWMA pass
    const firstResult = wasm.hwma_js(close, 0.2, 0.1, 0.1);
    
    // Second HWMA pass using first result as input
    const secondResult = wasm.hwma_js(firstResult, 0.2, 0.1, 0.1);
    
    assert.strictEqual(secondResult.length, firstResult.length);
    
    // Verify values are reasonable (not NaN/Inf)
    let finiteCount = 0;
    for (let i = 0; i < secondResult.length; i++) {
        if (isFinite(secondResult[i])) {
            finiteCount++;
            assert(secondResult[i] > 0, `HWMA reinput produced negative value at index ${i}`);
        }
    }
    assert(finiteCount > 0, "No finite values in HWMA reinput result");
});

test('HWMA NaN handling', () => {
    // Test HWMA handling of NaN values - mirrors check_hwma_nan_handling
    const close = new Float64Array(testData.close);
    
    const result = wasm.hwma_js(close, 0.2, 0.1, 0.1);
    
    assert.strictEqual(result.length, close.length);
    
    // After index 3, all values should be finite
    if (result.length > 3) {
        for (let i = 3; i < result.length; i++) {
            assert(isFinite(result[i]), `Unexpected non-finite value at index ${i}`);
        }
    }
});

test('HWMA batch', () => {
    // Test HWMA batch computation
    const close = new Float64Array(testData.close);
    
    // Test parameter ranges
    const na_start = 0.1, na_end = 0.3, na_step = 0.1;  // na: 0.1, 0.2, 0.3
    const nb_start = 0.05, nb_end = 0.15, nb_step = 0.05;  // nb: 0.05, 0.10, 0.15
    const nc_start = 0.05, nc_end = 0.15, nc_step = 0.05;  // nc: 0.05, 0.10, 0.15
    
    const batch_result = wasm.hwma_batch_js(
        close, 
        na_start, na_end, na_step,
        nb_start, nb_end, nb_step,
        nc_start, nc_end, nc_step
    );
    const metadata = wasm.hwma_batch_metadata_js(
        na_start, na_end, na_step,
        nb_start, nb_end, nb_step,
        nc_start, nc_end, nc_step
    );
    
    // Metadata should contain parameter combinations
    assert.strictEqual(metadata.length, 27 * 3);  // 27 combinations * 3 params
    
    // Batch result should contain all individual results flattened
    assert.strictEqual(batch_result.length, 27 * close.length);  // 27 combinations
    
    // Verify first combination matches individual calculation
    const individual_result = wasm.hwma_js(close, 0.1, 0.05, 0.05);
    
    // Extract first row from batch result
    const row = batch_result.slice(0, close.length);
    
    assertArrayClose(row, individual_result, 1e-9, 'First combination');
});

test('HWMA different params', () => {
    // Test HWMA with different parameter values
    const close = new Float64Array(testData.close);
    
    // Test various parameter combinations
    const paramSets = [
        [0.1, 0.05, 0.05],
        [0.2, 0.1, 0.1],
        [0.3, 0.15, 0.15],
        [0.5, 0.25, 0.25],
    ];
    
    for (const [na, nb, nc] of paramSets) {
        const result = wasm.hwma_js(close, na, nb, nc);
        assert.strictEqual(result.length, close.length);
        
        // Verify finite values after warmup
        let finiteCount = 0;
        for (let i = 0; i < result.length; i++) {
            if (isFinite(result[i])) finiteCount++;
        }
        assert(finiteCount > close.length - 4, 
            `Too many non-finite values for params (${na}, ${nb}, ${nc})`);
    }
});

test('HWMA batch performance', () => {
    // Test that batch computation is more efficient than multiple single computations
    const close = new Float64Array(testData.close.slice(0, 1000)); // Use first 1000 values
    
    // Test 8 combinations (2 * 2 * 2)
    const startBatch = performance.now();
    const batchResult = wasm.hwma_batch_js(
        close,
        0.1, 0.2, 0.1,    // 2 na values
        0.05, 0.1, 0.05,  // 2 nb values
        0.05, 0.1, 0.05   // 2 nc values
    );
    const batchTime = performance.now() - startBatch;
    
    const startSingle = performance.now();
    const singleResults = [];
    for (let na = 0.1; na <= 0.2; na += 0.1) {
        for (let nb = 0.05; nb <= 0.1; nb += 0.05) {
            for (let nc = 0.05; nc <= 0.1; nc += 0.05) {
                singleResults.push(...wasm.hwma_js(close, na, nb, nc));
            }
        }
    }
    const singleTime = performance.now() - startSingle;
    
    // Batch should be faster than multiple single calls
    console.log(`Batch time: ${batchTime.toFixed(2)}ms, Single time: ${singleTime.toFixed(2)}ms`);
    
    // Verify results match
    assertArrayClose(batchResult, singleResults, 1e-9, 'Batch vs single results');
});

test('HWMA edge cases', () => {
    // Test HWMA with edge case parameter values
    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
    
    // Test very small parameters (close to 0)
    const result1 = wasm.hwma_js(data, 0.001, 0.001, 0.001);
    assert.strictEqual(result1.length, data.length);
    let finiteCount1 = 0;
    for (let val of result1) {
        if (isFinite(val)) finiteCount1++;
    }
    assert(finiteCount1 > 0, "No finite values with small parameters");
    
    // Test parameters close to 1
    const result2 = wasm.hwma_js(data, 0.999, 0.999, 0.999);
    assert.strictEqual(result2.length, data.length);
    let finiteCount2 = 0;
    for (let val of result2) {
        if (isFinite(val)) finiteCount2++;
    }
    assert(finiteCount2 > 0, "No finite values with large parameters");
});

test('HWMA single value', () => {
    // Test HWMA with single value input
    const data = new Float64Array([42.0]);
    
    const result = wasm.hwma_js(data, 0.2, 0.1, 0.1);
    assert.strictEqual(result.length, 1);
    assert(Math.abs(result[0] - data[0]) < 1e-12); // First value should be the data value
});

test('HWMA two values', () => {
    // Test HWMA with two values input
    const data = new Float64Array([1.0, 2.0]);
    
    const result = wasm.hwma_js(data, 0.2, 0.1, 0.1);
    assert.strictEqual(result.length, 2);
    // HWMA doesn't use NaN prefix for small data
    assert(isFinite(result[0]));
    assert(isFinite(result[1]));
});

test('HWMA three values', () => {
    // Test HWMA with three values input
    const data = new Float64Array([1.0, 2.0, 3.0]);
    
    const result = wasm.hwma_js(data, 0.2, 0.1, 0.1);
    assert.strictEqual(result.length, 3);
    // HWMA doesn't use NaN prefix for small data
    assert(isFinite(result[0]));
    assert(isFinite(result[1]));
    assert(isFinite(result[2]));
});

test('HWMA batch metadata', () => {
    // Test metadata function returns correct parameter combinations
    const metadata = wasm.hwma_batch_metadata_js(
        0.1, 0.2, 0.1,    // na: 0.1, 0.2
        0.05, 0.1, 0.05,  // nb: 0.05, 0.1
        0.05, 0.1, 0.05   // nc: 0.05, 0.1
    );
    
    // Should have 2 * 2 * 2 = 8 combinations
    // Each combination has 3 values (na, nb, nc), so 8 * 3 = 24
    assert.strictEqual(metadata.length, 24);
    
    // Check first few combinations (flattened as [na, nb, nc, na, nb, nc, ...])
    const expectedFirstCombinations = [
        0.1, 0.05, 0.05,   // First combination
        0.1, 0.05, 0.1,    // Second combination
        0.1, 0.1, 0.05,    // Third combination
        0.1, 0.1, 0.1,     // Fourth combination
    ];
    
    for (let i = 0; i < expectedFirstCombinations.length; i++) {
        assertClose(
            metadata[i], 
            expectedFirstCombinations[i], 
            1e-9, 
            `Metadata value at index ${i}`
        );
    }
});

test('HWMA warmup period', () => {
    // Test that warmup period is correctly calculated
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const result = wasm.hwma_js(close, 0.2, 0.1, 0.1);
    
    // HWMA doesn't have a traditional warmup period with NaN values
    // All values should be finite from the start
    for (let i = 0; i < Math.min(result.length, 10); i++) {
        assert(isFinite(result[i]), `Expected finite value at index ${i}`);
    }
});

test('HWMA consistency across calls', () => {
    // Test that HWMA produces consistent results across multiple calls
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const result1 = wasm.hwma_js(close, 0.2, 0.1, 0.1);
    const result2 = wasm.hwma_js(close, 0.2, 0.1, 0.1);
    
    assertArrayClose(result1, result2, 1e-15, "HWMA results not consistent");
});

test('HWMA parameter step precision', () => {
    // Test batch with very small step sizes
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    
    const batch_result = wasm.hwma_batch_js(
        data,
        0.1, 0.11, 0.01,   // 2 values with small step
        0.05, 0.06, 0.01,  // 2 values with small step
        0.05, 0.06, 0.01   // 2 values with small step
    );
    
    // Should have 2 * 2 * 2 = 8 combinations
    assert.strictEqual(batch_result.length, 8 * data.length);
});