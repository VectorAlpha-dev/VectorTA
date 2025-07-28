/**
 * WASM binding tests for KAUFMANSTOP indicator.
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

test('KAUFMANSTOP with default parameters', () => {
    const { high, low } = testData;
    const result = wasm.kaufmanstop_js(high, low, 22, 2.0, 'long', 'sma');
    
    // WASM returns Array, not Float64Array
    assert.ok(Array.isArray(result));
    assert.strictEqual(result.length, high.length);
    
    // Check warmup period (should have NaN values)
    const warmupPeriod = 22 + 21; // period + (period - 1) based on Rust implementation
    for (let i = 0; i < Math.min(warmupPeriod, result.length); i++) {
        assert.ok(isNaN(result[i]), `Expected NaN at index ${i} during warmup`);
    }
    
    // Values after warmup should not be NaN
    for (let i = warmupPeriod; i < result.length; i++) {
        assert.ok(isFinite(result[i]), `Value at index ${i} should be finite`);
    }
});

test('KAUFMANSTOP accuracy', () => {
    // Test KAUFMANSTOP matches expected values from Rust tests
    const { high, low } = testData;
    
    const result = wasm.kaufmanstop_js(high, low, 22, 2.0, 'long', 'sma');
    
    assert.strictEqual(result.length, high.length);
    
    // Expected last 5 values from Rust test
    const expectedLast5 = [
        56711.545454545456,
        57132.72727272727,
        57015.72727272727,
        57137.18181818182,
        56516.09090909091,
    ];
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expectedLast5,
        1e-1,  // Use same tolerance as Rust test
        "KAUFMANSTOP last 5 values mismatch"
    );
});

test('KAUFMANSTOP with short direction', () => {
    const { high, low } = testData;
    
    const resultLong = wasm.kaufmanstop_js(high.slice(0, 100), low.slice(0, 100), 22, 2.0, 'long', 'sma');
    const resultShort = wasm.kaufmanstop_js(high.slice(0, 100), low.slice(0, 100), 22, 2.0, 'short', 'sma');
    
    // Results should be different
    let foundDifference = false;
    for (let i = 43; i < 100; i++) { // After warmup period
        if (!isNaN(resultLong[i]) && !isNaN(resultShort[i]) && resultLong[i] !== resultShort[i]) {
            foundDifference = true;
            break;
        }
    }
    assert.ok(foundDifference, "Long and short directions should produce different results");
});

test('KAUFMANSTOP zero period', () => {
    // Test KAUFMANSTOP fails with zero period
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0, 25.0]);
    
    assert.throws(() => {
        wasm.kaufmanstop_js(high, low, 0, 2.0, 'long', 'sma');
    }, /Invalid period/);
});

test('KAUFMANSTOP period exceeds length', () => {
    // Test KAUFMANSTOP fails when period exceeds data length
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0, 25.0]);
    
    assert.throws(() => {
        wasm.kaufmanstop_js(high, low, 10, 2.0, 'long', 'sma');
    }, /Invalid period/);
});

test('KAUFMANSTOP mismatched lengths', () => {
    // Test KAUFMANSTOP fails when high and low have different lengths
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0]); // Different length
    
    assert.throws(() => {
        wasm.kaufmanstop_js(high, low, 2, 2.0, 'long', 'sma');
    }, /Invalid period|same length/);
});

test('KAUFMANSTOP empty data', () => {
    // Test KAUFMANSTOP fails with empty arrays
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.kaufmanstop_js(empty, empty, 22, 2.0, 'long', 'sma');
    }, /Empty data/);
});

test('KAUFMANSTOP batch processing', () => {
    const { high, low } = testData;
    
    // Test batch processing with small range
    const result = wasm.kaufmanstop_batch_js(
        high, low,
        20, 24, 2,      // period range: 20, 22, 24
        1.5, 2.5, 0.5,  // mult range: 1.5, 2.0, 2.5
        'long', 'sma'
    );
    
    assert.ok(result);
    assert.ok(result.values);
    assert.ok(result.combos);
    assert.strictEqual(result.rows, 9); // 3 periods × 3 mults
    assert.strictEqual(result.cols, high.length);
    assert.strictEqual(result.values.length, 9 * high.length);
    assert.strictEqual(result.combos.length, 9);
    
    // Verify combo structure
    for (const combo of result.combos) {
        assert.ok('period' in combo);
        assert.ok('mult' in combo);
        assert.ok('direction' in combo);
        assert.ok('ma_type' in combo);
        assert.strictEqual(combo.direction, 'long');
        assert.strictEqual(combo.ma_type, 'sma');
    }
});

test('KAUFMANSTOP fast API', () => {
    const { high, low } = testData;
    const len = high.length;
    
    // Allocate output buffer
    const outPtr = wasm.kaufmanstop_alloc(len);
    
    try {
        // Create typed arrays for input data
        const highArray = new Float64Array(high);
        const lowArray = new Float64Array(low);
        
        // Call fast API
        wasm.kaufmanstop_into(
            highArray.buffer,
            lowArray.buffer,
            outPtr,
            len,
            22, 2.0, 'long', 'sma'
        );
        
        // Read result from memory
        const memory = wasm.memory;
        const result = new Float64Array(memory.buffer, outPtr, len);
        
        // Verify length
        assert.strictEqual(result.length, len);
        
        // Verify some values are computed
        const warmupPeriod = 43;
        let foundFiniteValue = false;
        for (let i = warmupPeriod; i < result.length; i++) {
            if (isFinite(result[i])) {
                foundFiniteValue = true;
                break;
            }
        }
        assert.ok(foundFiniteValue, "Should have computed some finite values");
        
    } finally {
        // Free allocated memory
        wasm.kaufmanstop_free(outPtr, len);
    }
});

test.after(() => {
    console.log('KAUFMANSTOP WASM tests completed');
});
