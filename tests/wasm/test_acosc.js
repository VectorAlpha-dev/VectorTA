/**
 * WASM binding tests for ACOSC indicator.
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
        // No need to call default() for ES modules
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }
    
    testData = loadTestData();
});

test('ACOSC partial params', () => {
    // Test with default parameters - mirrors check_acosc_partial_params
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    
    const result = wasm.acosc_js(high, low);
    assert.strictEqual(result.length, high.length * 2); // osc + change
});

test('ACOSC accuracy', async () => {
    // Test ACOSC matches expected values from Rust tests - mirrors check_acosc_accuracy
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const expected = EXPECTED_OUTPUTS.acosc;
    
    const result = wasm.acosc_js(high, low);
    
    // Result contains flattened [osc, change] arrays
    const len = high.length;
    const osc = result.slice(0, len);
    const change = result.slice(len, len * 2);
    
    assert.strictEqual(osc.length, len);
    assert.strictEqual(change.length, len);
    
    // Check last 5 osc values match expected
    const last5Osc = osc.slice(-5);
    assertArrayClose(
        last5Osc,
        expected.last5Osc,
        0.1,  // ACOSC uses 1e-1 tolerance in Rust tests
        "ACOSC osc last 5 values mismatch"
    );
    
    // Check last 5 change values match expected
    const last5Change = change.slice(-5);
    assertArrayClose(
        last5Change,
        expected.last5Change,
        0.1,  // ACOSC uses 1e-1 tolerance in Rust tests
        "ACOSC change last 5 values mismatch"
    );
    
    // Skip Rust comparison for now - compareWithRust doesn't handle multi-output indicators yet
    // TODO: Update compareWithRust to handle indicators with multiple outputs
    // await compareWithRust('acosc', {osc, change}, 'high_low', expected.defaultParams);
});

test('ACOSC too short', () => {
    // Test ACOSC fails with insufficient data
    const high = new Float64Array([100.0, 101.0]);
    const low = new Float64Array([99.0, 98.0]);
    
    assert.throws(() => {
        wasm.acosc_js(high, low);
    }, /Not enough data/);
});

test('ACOSC length mismatch', () => {
    // Test ACOSC fails when high and low lengths don't match
    const high = new Float64Array([100.0, 101.0, 102.0]);
    const low = new Float64Array([99.0, 98.0]);  // Shorter array
    
    assert.throws(() => {
        wasm.acosc_js(high, low);
    }, /Mismatch/);
});

test('ACOSC NaN handling', () => {
    // Test ACOSC handles NaN values correctly
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    
    const result = wasm.acosc_js(high, low);
    const len = high.length;
    const osc = result.slice(0, len);
    const change = result.slice(len, len * 2);
    
    // First 38 values should be NaN (warmup period = 34 + 5 - 1)
    assertAllNaN(osc.slice(0, 38), "Expected NaN in warmup period for osc");
    assertAllNaN(change.slice(0, 38), "Expected NaN in warmup period for change");
    
    // After warmup period, no NaN values should exist
    if (osc.length > 240) {
        for (let i = 240; i < osc.length; i++) {
            assert(!isNaN(osc[i]), `Found unexpected NaN in osc at index ${i}`);
            assert(!isNaN(change[i]), `Found unexpected NaN in change at index ${i}`);
        }
    }
});

test('ACOSC batch single result', () => {
    // Test batch with single result (since ACOSC has no parameters)
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    
    // Batch should return same as single
    const batchResult = wasm.acosc_batch_js(high, low);
    const singleResult = wasm.acosc_js(high, low);
    
    assert.strictEqual(batchResult.length, singleResult.length);
    assertArrayClose(batchResult, singleResult, 1e-10, "Batch vs single mismatch");
});

test('ACOSC batch metadata', () => {
    // Test metadata function returns empty array (no parameters)
    const metadata = wasm.acosc_batch_metadata_js();
    
    // Should be empty since ACOSC has no parameters
    assert.strictEqual(metadata.length, 0);
});

test('ACOSC edge cases', () => {
    // Test edge cases
    const smallHigh = new Float64Array(40); // Just enough data
    const smallLow = new Float64Array(40);
    
    // Fill with test data
    for (let i = 0; i < 40; i++) {
        smallHigh[i] = 100 + i;
        smallLow[i] = 95 + i;
    }
    
    const result = wasm.acosc_js(smallHigh, smallLow);
    assert.strictEqual(result.length, 80); // 40 osc + 40 change
    
    // Empty data should throw
    assert.throws(() => {
        wasm.acosc_js(new Float64Array([]), new Float64Array([]));
    }, /Not enough data/);
});

test.after(() => {
    console.log('ACOSC WASM tests completed');
});
