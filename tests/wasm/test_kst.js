/**
 * WASM binding tests for KST indicator.
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

test('KST default params', () => {
    // Test with default parameters
    const close = new Float64Array(testData.close);
    
    const result = wasm.kst_js(close, 10, 10, 10, 15, 10, 15, 20, 30, 9);
    assert(result.line, 'Result should have line property');
    assert(result.signal, 'Result should have signal property');
    assert.strictEqual(result.line.length, close.length);
    assert.strictEqual(result.signal.length, close.length);
});

test('KST accuracy', async () => {
    // Test KST matches expected values from Rust tests
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.kst;
    
    const result = wasm.kst_js(
        close,
        expected.defaultParams.sma_period1,
        expected.defaultParams.sma_period2,
        expected.defaultParams.sma_period3,
        expected.defaultParams.sma_period4,
        expected.defaultParams.roc_period1,
        expected.defaultParams.roc_period2,
        expected.defaultParams.roc_period3,
        expected.defaultParams.roc_period4,
        expected.defaultParams.signal_period
    );
    
    assert.strictEqual(result.line.length, close.length);
    assert.strictEqual(result.signal.length, close.length);
    
    // Check last 5 values match expected
    const last5Line = result.line.slice(-5);
    const last5Signal = result.signal.slice(-5);
    assertArrayClose(
        last5Line,
        expected.last5Values.line,
        1e-8,
        "KST line last 5 values mismatch"
    );
    assertArrayClose(
        last5Signal,
        expected.last5Values.signal,
        1e-8,
        "KST signal last 5 values mismatch"
    );
    
    // Compare full output with Rust
    await compareWithRust('kst', result, 'close', expected.defaultParams);
});

test('KST all NaN values', () => {
    // Test KST with all NaN values
    const inputData = new Float64Array(10).fill(NaN);
    
    assert.throws(() => {
        wasm.kst_js(inputData, 10, 10, 10, 15, 10, 15, 20, 30, 9);
    }, /All values are NaN/);
});

test('KST zero periods', () => {
    // Test KST fails with zero periods
    const inputData = new Float64Array([10.0, 20.0, 30.0, 40.0, 50.0]);
    
    // Test zero SMA period
    assert.throws(() => {
        wasm.kst_js(inputData, 0, 10, 10, 15, 10, 15, 20, 30, 9);
    }, /Invalid period/);
    
    // Test zero ROC period
    assert.throws(() => {
        wasm.kst_js(inputData, 10, 10, 10, 15, 0, 15, 20, 30, 9);
    }, /Invalid period/);
    
    // Test zero signal period
    assert.throws(() => {
        wasm.kst_js(inputData, 10, 10, 10, 15, 10, 15, 20, 30, 0);
    }, /Invalid period/);
});

test('KST insufficient data', () => {
    // Test KST fails when not enough data for periods
    const dataSmall = new Float64Array([10.0, 20.0, 30.0, 40.0, 50.0]);
    
    assert.throws(() => {
        wasm.kst_js(dataSmall, 10, 10, 10, 15, 10, 15, 20, 30, 9);
    }, /Not enough valid data/);
});

test('KST with NaN prefix', () => {
    // Test KST with NaN values at start
    const inputData = new Float64Array(100);
    inputData.fill(NaN, 0, 20);
    for (let i = 20; i < 100; i++) {
        inputData[i] = 100.0 + Math.sin(i * 0.1) * 10.0;
    }
    
    const result = wasm.kst_js(inputData, 5, 5, 5, 10, 5, 10, 15, 20, 5);
    assert.strictEqual(result.line.length, inputData.length);
    assert.strictEqual(result.signal.length, inputData.length);
    
    // Check warmup period
    const warmup = 20 + 10 - 1; // roc_period4 + sma_period4 - 1
    for (let i = 0; i < warmup; i++) {
        assert(isNaN(result.line[i]), `Line[${i}] should be NaN during warmup`);
        assert(isNaN(result.signal[i]), `Signal[${i}] should be NaN during warmup`);
    }
    
    // After warmup, values should not be NaN
    for (let i = warmup + 10; i < result.line.length; i++) {
        assert(!isNaN(result.line[i]), `Line[${i}] should not be NaN after warmup`);
        assert(!isNaN(result.signal[i]), `Signal[${i}] should not be NaN after warmup`);
    }
});

test('KST fast/unsafe API', () => {
    // Test the fast/unsafe API with pre-allocated memory
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    // Allocate memory for input and outputs
    const inPtr = wasm.kst_alloc(len);
    const lineOutPtr = wasm.kst_alloc(len);
    const signalOutPtr = wasm.kst_alloc(len);
    
    try {
        // Create views on the WASM memory
        const memory = wasm.memory;
        const inArray = new Float64Array(memory.buffer, inPtr, len);
        
        // Copy input data
        inArray.set(close);
        
        // Call the fast API
        wasm.kst_into(
            inPtr, lineOutPtr, signalOutPtr, len,
            10, 10, 10, 15, 10, 15, 20, 30, 9
        );
        
        // Read results
        const lineResult = new Float64Array(memory.buffer, lineOutPtr, len);
        const signalResult = new Float64Array(memory.buffer, signalOutPtr, len);
        
        // Compare with safe API
        const safeResult = wasm.kst_js(close, 10, 10, 10, 15, 10, 15, 20, 30, 9);
        assertArrayClose(
            Array.from(lineResult),
            safeResult.line,
            1e-10,
            "Fast API line should match safe API"
        );
        assertArrayClose(
            Array.from(signalResult),
            safeResult.signal,
            1e-10,
            "Fast API signal should match safe API"
        );
    } finally {
        // Clean up allocated memory
        wasm.kst_free(inPtr, len);
        wasm.kst_free(lineOutPtr, len);
        wasm.kst_free(signalOutPtr, len);
    }
});

test('KST fast API with aliasing', () => {
    // Test fast API when input and output share the same memory
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    // Allocate memory for input (will also be used for line output)
    const ptr = wasm.kst_alloc(len);
    const signalPtr = wasm.kst_alloc(len);
    
    try {
        // Create views on the WASM memory
        const memory = wasm.memory;
        const array = new Float64Array(memory.buffer, ptr, len);
        
        // Copy input data
        array.set(close);
        
        // Call fast API with aliasing (input and line output share memory)
        wasm.kst_into(
            ptr, ptr, signalPtr, len,
            10, 10, 10, 15, 10, 15, 20, 30, 9
        );
        
        // Read results
        const lineResult = new Float64Array(memory.buffer, ptr, len);
        const signalResult = new Float64Array(memory.buffer, signalPtr, len);
        
        // Compare with safe API
        const safeResult = wasm.kst_js(close, 10, 10, 10, 15, 10, 15, 20, 30, 9);
        assertArrayClose(
            Array.from(lineResult),
            safeResult.line,
            1e-10,
            "Aliased fast API line should match safe API"
        );
        assertArrayClose(
            Array.from(signalResult),
            safeResult.signal,
            1e-10,
            "Aliased fast API signal should match safe API"
        );
    } finally {
        // Clean up allocated memory
        wasm.kst_free(ptr, len);
        wasm.kst_free(signalPtr, len);
    }
});

test('KST batch calculation', () => {
    // Test batch KST calculation
    const close = new Float64Array(testData.close.slice(0, 100)); // Use smaller dataset for batch
    
    const config = {
        sma_period1_range: [8, 12, 2],
        sma_period2_range: [8, 12, 2],
        sma_period3_range: [8, 12, 2],
        sma_period4_range: [12, 18, 3],
        roc_period1_range: [8, 12, 2],
        roc_period2_range: [12, 18, 3],
        roc_period3_range: [18, 22, 2],
        roc_period4_range: [25, 35, 5],
        signal_period_range: [7, 11, 2]
    };
    
    const result = wasm.kst_batch(close, config);
    
    assert(result.line, 'Batch result should have line property');
    assert(result.signal, 'Batch result should have signal property');
    assert(result.combos, 'Batch result should have combos property');
    assert(result.rows, 'Batch result should have rows property');
    assert(result.cols, 'Batch result should have cols property');
    
    // Check dimensions
    assert.strictEqual(result.cols, close.length);
    assert(result.rows > 0, 'Should have at least one parameter combination');
    assert.strictEqual(result.line.length, result.rows * result.cols);
    assert.strictEqual(result.signal.length, result.rows * result.cols);
    assert.strictEqual(result.combos.length, result.rows);
    
    // Verify first combination matches single calculation
    const firstCombo = result.combos[0];
    const singleResult = wasm.kst_js(
        close,
        firstCombo.sma_period1,
        firstCombo.sma_period2,
        firstCombo.sma_period3,
        firstCombo.sma_period4,
        firstCombo.roc_period1,
        firstCombo.roc_period2,
        firstCombo.roc_period3,
        firstCombo.roc_period4,
        firstCombo.signal_period
    );
    
    const firstRowLine = result.line.slice(0, result.cols);
    const firstRowSignal = result.signal.slice(0, result.cols);
    assertArrayClose(
        firstRowLine,
        singleResult.line,
        1e-10,
        "Batch first row line should match single calculation"
    );
    assertArrayClose(
        firstRowSignal,
        singleResult.signal,
        1e-10,
        "Batch first row signal should match single calculation"
    );
});

console.log('All KST WASM tests passed! 🎉');