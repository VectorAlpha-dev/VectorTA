import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { 
    loadTestData, 
    EXPECTED_SUPERSMOOTHER
} from './test_utils.js';
import { compareWithRust } from './rust-comparison.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;

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
});

test('SuperSmoother accuracy test', (t) => {
    const data = loadTestData();
    const closePrices = data.close;
    const period = 14;
    
    // Calculate SuperSmoother
    const result = wasm.supersmoother_js(closePrices, period);
    
    // Check output length
    assert.strictEqual(result.length, closePrices.length);
    
    // For 2-pole supersmoother, warmup period = period - 1
    // So first 13 values should be NaN for period=14
    for (let i = 0; i < 13; i++) {
        assert.ok(isNaN(result[i]), `Expected NaN at index ${i} during warmup`);
    }
    
    // Values at indices 13 and 14 should be initialized (not NaN)
    assert.ok(!isNaN(result[13]), 'Value at index 13 should be initialized');
    assert.ok(!isNaN(result[14]), 'Value at index 14 should be initialized');
    
    // Test last 5 values against expected
    const last5 = result.slice(-5);
    const expected = EXPECTED_SUPERSMOOTHER;
    
    for (let i = 0; i < 5; i++) {
        const diff = Math.abs(last5[i] - expected[i]);
        assert.ok(
            diff < 1e-6,
            `Value mismatch at position ${i}: expected ${expected[i]}, got ${last5[i]}, diff ${diff}`
        );
    }
});

test('SuperSmoother batch processing', (t) => {
    const data = loadTestData();
    const closePrices = data.close;
    
    // Test batch with multiple periods
    const batchResult = wasm.supersmoother_batch_js(
        closePrices,
        10,  // period_start
        20,  // period_end
        5    // period_step
    );
    
    // Get metadata
    const metadata = wasm.supersmoother_batch_metadata_js(10, 20, 5);
    const numPeriods = metadata.length;
    
    // Check dimensions
    assert.strictEqual(batchResult.length, numPeriods * closePrices.length);
    assert.strictEqual(numPeriods, 3); // periods: 10, 15, 20
    
    // Verify each period's results
    for (let p = 0; p < numPeriods; p++) {
        const period = metadata[p];
        const rowStart = p * closePrices.length;
        const row = batchResult.slice(rowStart, rowStart + closePrices.length);
        
        // Check warmup period for each row
        const warmupLength = period - 1;
        for (let i = 0; i < warmupLength; i++) {
            assert.ok(isNaN(row[i]), `Expected NaN at index ${i} for period ${period}`);
        }
        
        // Should have valid values after warmup
        assert.ok(!isNaN(row[warmupLength]), `Expected valid value at index ${warmupLength} for period ${period}`);
    }
});

test('SuperSmoother with NaN handling', (t) => {
    const data = loadTestData();
    const dataWithNan = [...data.close];
    
    // Insert NaN values
    for (let i = 10; i < 15; i++) {
        dataWithNan[i] = NaN;
    }
    
    // Should compute without error
    const result = wasm.supersmoother_js(dataWithNan, 14);
    
    // Check that NaN propagates appropriately
    // With period=14, NaN at 10-14 affects output from 10 to at least 23
    for (let i = 10; i < 24; i++) {
        assert.ok(isNaN(result[i]), `Expected NaN at position ${i}`);
    }
});

test('SuperSmoother with leading NaNs', (t) => {
    // Create data starting with NaNs
    const data = new Float64Array(20);
    for (let i = 0; i < 5; i++) {
        data[i] = NaN;
    }
    for (let i = 5; i < 20; i++) {
        data[i] = i - 4; // 1, 2, 3, ...
    }
    
    const period = 3;
    const result = wasm.supersmoother_js(data, period);
    
    // For 2-pole supersmoother with leading NaNs:
    // first_non_nan = 5
    // NaN up to first_non_nan + period - 1 = 5 + 3 - 1 = 7
    // Two initial values at indices 7 and 8
    // Main calculation starts at index 9
    
    // Check that NaN input produces NaN output
    for (let i = 0; i < 5; i++) {
        assert.ok(isNaN(result[i]), `Expected NaN at index ${i} where input is NaN`);
    }
    
    // Due to warmup, values remain NaN
    for (let i = 5; i < 7; i++) {
        assert.ok(isNaN(result[i]), `Expected NaN at index ${i} due to warmup`);
    }
    
    // Initial values should be set from data
    assert.strictEqual(result[7], data[7], 'Expected initial value at index 7');
    assert.strictEqual(result[8], data[8], 'Expected initial value at index 8');
});

test('SuperSmoother error handling', (t) => {
    const data = loadTestData();
    const closePrices = data.close;
    
    // Test with period = 0
    assert.throws(
        () => wasm.supersmoother_js(closePrices, 0),
        /Invalid period/,
        'Should throw error for period = 0'
    );
    
    // Test with period > data length
    assert.throws(
        () => wasm.supersmoother_js(closePrices.slice(0, 5), 10),
        /Invalid period/,
        'Should throw error when period exceeds data length'
    );
    
    // Test with all NaN data
    const allNan = new Float64Array(10).fill(NaN);
    assert.throws(
        () => wasm.supersmoother_js(allNan, 5),
        /All values are NaN/,
        'Should throw error for all NaN input'
    );
    
    // Test with empty data
    const empty = new Float64Array(0);
    assert.throws(
        () => wasm.supersmoother_js(empty, 5),
        /Empty data/,
        'Should throw error for empty input'
    );
});

test('SuperSmoother edge cases', (t) => {
    const data = loadTestData();
    const closePrices = data.close;
    
    // Test with minimum period (1)
    const result1 = wasm.supersmoother_js(closePrices, 1);
    assert.strictEqual(result1.length, closePrices.length);
    
    // Test with very small dataset
    const smallData = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
    const result2 = wasm.supersmoother_js(smallData, 2);
    assert.strictEqual(result2.length, smallData.length);
    
    // For 2-pole filter, warmup is period-1, so first value is NaN
    assert.ok(isNaN(result2[0]));
    assert.ok(!isNaN(result2[1]));
});

test('SuperSmoother consistency check', (t) => {
    const data = loadTestData();
    const closePrices = data.close;
    const period = 14;
    
    // Run multiple times to ensure consistency
    const result1 = wasm.supersmoother_js(closePrices, period);
    const result2 = wasm.supersmoother_js(closePrices, period);
    
    // Results should be identical
    for (let i = 0; i < result1.length; i++) {
        if (isNaN(result1[i]) && isNaN(result2[i])) continue;
        assert.strictEqual(result1[i], result2[i], `Inconsistent result at index ${i}`);
    }
});

test('Compare SuperSmoother with Rust implementation', async (t) => {
    const data = loadTestData();
    const closePrices = data.close;
    const period = 14;
    
    // Get WASM result
    const wasmResult = wasm.supersmoother_js(closePrices, period);
    
    // Compare with Rust
    const result = await compareWithRust('supersmoother', wasmResult, 'close', { period });
    
    // compareWithRust will throw if there's a mismatch, so if we get here, test passed
    assert.ok(result, 'Comparison with Rust succeeded');
});

test('SuperSmoother batch metadata', (t) => {
    // Test metadata generation
    const metadata = wasm.supersmoother_batch_metadata_js(5, 15, 5);
    
    // Should return [5, 10, 15]
    assert.strictEqual(metadata.length, 3);
    assert.strictEqual(metadata[0], 5);
    assert.strictEqual(metadata[1], 10);
    assert.strictEqual(metadata[2], 15);
    
    // Test with step = 0 (single value)
    const singleMeta = wasm.supersmoother_batch_metadata_js(7, 7, 0);
    assert.strictEqual(singleMeta.length, 1);
    assert.strictEqual(singleMeta[0], 7);
});

test('SuperSmoother reinput consistency', (t) => {
    const data = loadTestData();
    const closePrices = data.close;
    
    // First pass
    const result1 = wasm.supersmoother_js(closePrices, 14);
    
    // Second pass with different period
    const result2 = wasm.supersmoother_js(result1, 7);
    
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