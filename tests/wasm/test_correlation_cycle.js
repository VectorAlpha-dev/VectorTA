/**
 * WASM binding tests for CORRELATION_CYCLE indicator.
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

test('CORRELATION_CYCLE accuracy - Safe API', () => {
    const data = testData.close;
    const expected = EXPECTED_OUTPUTS.correlation_cycle;
    
    // Test with default parameters
    const result = wasm.correlation_cycle_js(data, expected.default_params.period, expected.default_params.threshold);
    
    // Extract the last 5 values for each output
    const n = result.real.length;
    const lastReal = result.real.slice(n - 5);
    const lastImag = result.imag.slice(n - 5);
    const lastAngle = result.angle.slice(n - 5);
    
    // Check accuracy
    assertArrayClose(lastReal, expected.last_5_values.real, 1e-8, 1e-10, 'Real output mismatch');
    assertArrayClose(lastImag, expected.last_5_values.imag, 1e-8, 1e-10, 'Imag output mismatch');
    assertArrayClose(lastAngle, expected.last_5_values.angle, 1e-8, 1e-10, 'Angle output mismatch');
});

test('CORRELATION_CYCLE accuracy - Fast API', () => {
    const data = testData.close;
    const len = data.length;
    const expected = EXPECTED_OUTPUTS.correlation_cycle;
    
    // Allocate buffers
    const inPtr = wasm.correlation_cycle_alloc(len);
    const realPtr = wasm.correlation_cycle_alloc(len);
    const imagPtr = wasm.correlation_cycle_alloc(len);
    const anglePtr = wasm.correlation_cycle_alloc(len);
    const statePtr = wasm.correlation_cycle_alloc(len);
    
    try {
        // Copy input data
        const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);
        inView.set(data);
        
        // Compute
        wasm.correlation_cycle_into(
            inPtr, realPtr, imagPtr, anglePtr, statePtr, len,
            expected.default_params.period, expected.default_params.threshold
        );
        
        // Read results
        const realView = new Float64Array(wasm.__wasm.memory.buffer, realPtr, len);
        const imagView = new Float64Array(wasm.__wasm.memory.buffer, imagPtr, len);
        const angleView = new Float64Array(wasm.__wasm.memory.buffer, anglePtr, len);
        
        // Extract last 5 values
        const lastReal = Array.from(realView.slice(len - 5));
        const lastImag = Array.from(imagView.slice(len - 5));
        const lastAngle = Array.from(angleView.slice(len - 5));
        
        // Check accuracy
        assertArrayClose(lastReal, expected.last_5_values.real, 1e-8, 1e-10, 'Fast API real output mismatch');
        assertArrayClose(lastImag, expected.last_5_values.imag, 1e-8, 1e-10, 'Fast API imag output mismatch');
        assertArrayClose(lastAngle, expected.last_5_values.angle, 1e-8, 1e-10, 'Fast API angle output mismatch');
    } finally {
        // Clean up
        wasm.correlation_cycle_free(inPtr, len);
        wasm.correlation_cycle_free(realPtr, len);
        wasm.correlation_cycle_free(imagPtr, len);
        wasm.correlation_cycle_free(anglePtr, len);
        wasm.correlation_cycle_free(statePtr, len);
    }
});

test('CORRELATION_CYCLE - Fast API aliasing test', () => {
    const data = testData.close.slice(0, 100); // Use smaller data for test
    const len = data.length;
    const expected = EXPECTED_OUTPUTS.correlation_cycle;
    
    // Allocate single buffer that will be used as both input and output
    const bufferPtr = wasm.correlation_cycle_alloc(len);
    const imagPtr = wasm.correlation_cycle_alloc(len);
    const anglePtr = wasm.correlation_cycle_alloc(len);
    const statePtr = wasm.correlation_cycle_alloc(len);
    
    try {
        // Copy input data to buffer
        const bufferView = new Float64Array(wasm.__wasm.memory.buffer, bufferPtr, len);
        bufferView.set(data);
        
        // Compute with aliasing (input and real output share same pointer)
        wasm.correlation_cycle_into(
            bufferPtr, bufferPtr, imagPtr, anglePtr, statePtr, len,
            expected.default_params.period, expected.default_params.threshold
        );
        
        // The function should handle aliasing correctly
        // Just verify it doesn't crash and produces valid output
        const resultView = new Float64Array(wasm.__wasm.memory.buffer, bufferPtr, len);
        
        // Check that warmup values are NaN
        for (let i = 0; i < expected.default_params.period; i++) {
            assert(isNaN(resultView[i]), `Expected NaN at index ${i} but got ${resultView[i]}`);
        }
        
        // Check that we have some non-NaN values after warmup
        let hasValidValues = false;
        for (let i = expected.default_params.period; i < len; i++) {
            if (!isNaN(resultView[i])) {
                hasValidValues = true;
                break;
            }
        }
        assert(hasValidValues, 'Expected some valid values after warmup period');
    } finally {
        // Clean up
        wasm.correlation_cycle_free(bufferPtr, len);
        wasm.correlation_cycle_free(imagPtr, len);
        wasm.correlation_cycle_free(anglePtr, len);
        wasm.correlation_cycle_free(statePtr, len);
    }
});

test('CORRELATION_CYCLE error handling', () => {
    // Test empty data
    assert.throws(() => {
        wasm.correlation_cycle_js([], 20, 9.0);
    }, /Empty data/, 'Should throw on empty data');
    
    // Test all NaN values
    const nanData = new Array(100).fill(NaN);
    assert.throws(() => {
        wasm.correlation_cycle_js(nanData, 20, 9.0);
    }, /All values are NaN/, 'Should throw on all NaN values');
    
    // Test invalid period
    const data = testData.close.slice(0, 50);
    assert.throws(() => {
        wasm.correlation_cycle_js(data, 100, 9.0); // period > data length
    }, /Invalid period/, 'Should throw on invalid period');
    
    // Test null pointers in fast API
    assert.throws(() => {
        wasm.correlation_cycle_into(0, 0, 0, 0, 0, 100, 20, 9.0);
    }, /Null pointer/, 'Should throw on null pointers');
});

test('CORRELATION_CYCLE batch API', () => {
    const data = testData.close.slice(0, 1000); // Use smaller data for batch test
    
    // Test batch computation
    const result = wasm.correlation_cycle_batch_js(
        data,
        15, 25, 5,    // period range: 15, 20, 25
        8.0, 10.0, 1.0  // threshold range: 8.0, 9.0, 10.0
    );
    
    // Should have 3 * 3 = 9 combinations
    assert.strictEqual(result.rows, 9, 'Expected 9 parameter combinations');
    assert.strictEqual(result.cols, data.length, 'Expected output length to match input');
    
    // Verify we have all 4 outputs
    assert(result.real && result.real.length === 9 * data.length, 'Expected real output');
    assert(result.imag && result.imag.length === 9 * data.length, 'Expected imag output');
    assert(result.angle && result.angle.length === 9 * data.length, 'Expected angle output');
    assert(result.state && result.state.length === 9 * data.length, 'Expected state output');
    
    // Verify parameter combinations
    assert.strictEqual(result.combos.length, 9, 'Expected 9 parameter combinations');
    assert.strictEqual(result.combos[0].period, 15, 'First combo should have period 15');
    assert.strictEqual(result.combos[0].threshold, 8.0, 'First combo should have threshold 8.0');
});

test.after(() => {
    console.log('CORRELATION_CYCLE WASM tests completed');
});
