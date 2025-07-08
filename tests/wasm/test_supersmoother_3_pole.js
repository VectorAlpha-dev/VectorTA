import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { 
    loadTestData, 
    EXPECTED_SUPERSMOOTHER_3_POLE
} from './test_utils.js';
import { compareWithRust } from './rust-comparison.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;

test.before(async () => {
    // Load WASM module
    try {
        const wasmPath = path.join(__dirname, '../../pkg/my_project.js');
        wasm = await import(wasmPath);
        // No need to call default() for ES modules
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }
});

test('SuperSmoother3Pole accuracy test', (t) => {
    const data = loadTestData();
    const closePrices = data.close;
    const period = 14;
    
    // Calculate SuperSmoother3Pole
    const result = wasm.supersmoother_3_pole_js(closePrices, period);
    
    // Check output length
    assert.strictEqual(result.length, closePrices.length);
    
    // For 3-pole supersmoother, first 3 values are initialized to input values
    // It doesn't have a traditional NaN warmup period
    for (let i = 0; i < Math.min(3, result.length); i++) {
        assert.ok(!isNaN(result[i]), `Value at index ${i} should not be NaN`);
    }
    
    // Test last 5 values against expected
    const last5 = result.slice(-5);
    const expected = EXPECTED_SUPERSMOOTHER_3_POLE;
    
    for (let i = 0; i < 5; i++) {
        const diff = Math.abs(last5[i] - expected[i]);
        assert.ok(
            diff < 1e-6,
            `Value mismatch at position ${i}: expected ${expected[i]}, got ${last5[i]}, diff ${diff}`
        );
    }
});

test('SuperSmoother3Pole batch processing', (t) => {
    const data = loadTestData();
    const closePrices = data.close;
    
    // Test batch with multiple periods
    const batchResult = wasm.supersmoother_3_pole_batch_js(
        closePrices,
        10,  // period_start
        20,  // period_end
        5    // period_step
    );
    
    // Get metadata
    const metadata = wasm.supersmoother_3_pole_batch_metadata_js(10, 20, 5);
    const numPeriods = metadata.length;
    
    // Check dimensions
    assert.strictEqual(batchResult.length, numPeriods * closePrices.length);
    assert.strictEqual(numPeriods, 3); // periods: 10, 15, 20
    
    // Verify each period's results
    for (let p = 0; p < numPeriods; p++) {
        const period = metadata[p];
        const rowStart = p * closePrices.length;
        const row = batchResult.slice(rowStart, rowStart + closePrices.length);
        
        // Check that first 3 values are not NaN (3-pole initialization)
        for (let i = 0; i < Math.min(3, row.length); i++) {
            assert.ok(!isNaN(row[i]), `Value at index ${i} for period ${period} should not be NaN`);
        }
    }
});

test('SuperSmoother3Pole with NaN handling', (t) => {
    const data = loadTestData();
    const dataWithNan = [...data.close];
    
    // Insert NaN values
    for (let i = 10; i < 15; i++) {
        dataWithNan[i] = NaN;
    }
    
    // Should compute without error
    const result = wasm.supersmoother_3_pole_js(dataWithNan, 14);
    
    // Check that NaN propagates appropriately
    for (let i = 10; i < 30; i++) {
        assert.ok(isNaN(result[i]), `Expected NaN at position ${i}`);
    }
});

test('SuperSmoother3Pole with leading NaNs', (t) => {
    // Create data starting with NaNs
    const data = new Float64Array(20);
    for (let i = 0; i < 5; i++) {
        data[i] = NaN;
    }
    for (let i = 5; i < 20; i++) {
        data[i] = i - 4; // 1, 2, 3, ...
    }
    
    const period = 3;
    const result = wasm.supersmoother_3_pole_js(data, period);
    
    // For 3-pole supersmoother with leading NaNs:
    // The warmup period is first_non_nan + period = 5 + 3 = 8
    // So all values up to index 8 will be NaN
    
    // Check that NaN input produces NaN output
    for (let i = 0; i < 5; i++) {
        assert.ok(isNaN(result[i]), `Expected NaN at index ${i} where input is NaN`);
    }
    
    // Due to warmup calculation, values remain NaN through warmup period
    for (let i = 5; i < 8; i++) {
        assert.ok(isNaN(result[i]), `Expected NaN at index ${i} due to warmup`);
    }
});

test('SuperSmoother3Pole error handling', (t) => {
    const data = loadTestData();
    const closePrices = data.close;
    
    // Test with period = 0
    assert.throws(
        () => wasm.supersmoother_3_pole_js(closePrices, 0),
        /Invalid period/,
        'Should throw error for period = 0'
    );
    
    // Test with period > data length
    assert.throws(
        () => wasm.supersmoother_3_pole_js(closePrices.slice(0, 5), 10),
        /Invalid period/,
        'Should throw error when period exceeds data length'
    );
    
    // Test with all NaN data
    const allNan = new Float64Array(10).fill(NaN);
    assert.throws(
        () => wasm.supersmoother_3_pole_js(allNan, 5),
        /All values are NaN/,
        'Should throw error for all NaN input'
    );
});

test('SuperSmoother3Pole edge cases', (t) => {
    const data = loadTestData();
    const closePrices = data.close;
    
    // Test with minimum period (1)
    const result1 = wasm.supersmoother_3_pole_js(closePrices, 1);
    assert.strictEqual(result1.length, closePrices.length);
    
    // Test with very small dataset
    const smallData = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
    const result2 = wasm.supersmoother_3_pole_js(smallData, 2);
    assert.strictEqual(result2.length, smallData.length);
    
    // For 3-pole filter, first 3 values should match input (initial conditions)
    assert.strictEqual(result2[0], 1.0);
    assert.strictEqual(result2[1], 2.0);
    assert.strictEqual(result2[2], 3.0);
});

test('SuperSmoother3Pole consistency check', (t) => {
    const data = loadTestData();
    const closePrices = data.close;
    const period = 14;
    
    // Run multiple times to ensure consistency
    const result1 = wasm.supersmoother_3_pole_js(closePrices, period);
    const result2 = wasm.supersmoother_3_pole_js(closePrices, period);
    
    // Results should be identical
    for (let i = 0; i < result1.length; i++) {
        if (isNaN(result1[i]) && isNaN(result2[i])) continue;
        assert.strictEqual(result1[i], result2[i], `Inconsistent result at index ${i}`);
    }
});

test('Compare SuperSmoother3Pole with Rust implementation', async (t) => {
    const data = loadTestData();
    const closePrices = data.close;
    const period = 14;
    
    // Get WASM result
    const wasmResult = wasm.supersmoother_3_pole_js(closePrices, period);
    
    // Compare with Rust
    const result = await compareWithRust('supersmoother_3_pole', wasmResult, 'close', { period });
    
    // compareWithRust will throw if there's a mismatch, so if we get here, test passed
    assert.ok(result, 'Comparison with Rust succeeded');
});

test('SuperSmoother3Pole batch metadata', (t) => {
    // Test metadata generation
    const metadata = wasm.supersmoother_3_pole_batch_metadata_js(5, 15, 5);
    
    // Should return [5, 10, 15]
    assert.strictEqual(metadata.length, 3);
    assert.strictEqual(metadata[0], 5);
    assert.strictEqual(metadata[1], 10);
    assert.strictEqual(metadata[2], 15);
    
    // Test with step = 0 (single value)
    const singleMeta = wasm.supersmoother_3_pole_batch_metadata_js(7, 7, 0);
    assert.strictEqual(singleMeta.length, 1);
    assert.strictEqual(singleMeta[0], 7);
});

test('SuperSmoother3Pole reinput consistency', (t) => {
    const data = loadTestData();
    const closePrices = data.close;
    
    // First pass
    const result1 = wasm.supersmoother_3_pole_js(closePrices, 14);
    
    // Second pass with different period
    const result2 = wasm.supersmoother_3_pole_js(result1, 7);
    
    // Results should be valid
    assert.strictEqual(result2.length, closePrices.length);
    
    // Check that we have some non-NaN values
    let hasValid = false;
    for (let i = 0; i < result2.length; i++) {
        if (!isNaN(result2[i])) {
            hasValid = true;
            break;
        }
    }
    assert.ok(hasValid, 'Expected some valid values in reinput result');
});