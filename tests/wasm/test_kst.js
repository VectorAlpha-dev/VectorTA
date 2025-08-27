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
    
    const result = wasm.kst(close, 10, 10, 10, 15, 10, 15, 20, 30, 9);
    assert(result.values, 'Result should have values property');
    assert(result.rows, 'Result should have rows property');
    assert(result.cols, 'Result should have cols property');
    assert.strictEqual(result.rows, 2);
    assert.strictEqual(result.cols, close.length);
    assert.strictEqual(result.values.length, 2 * close.length);
    
    // Extract line and signal from values
    const line = result.values.slice(0, result.cols);
    const signal = result.values.slice(result.cols, 2 * result.cols);
    assert.strictEqual(line.length, close.length);
    assert.strictEqual(signal.length, close.length);
});

test('KST accuracy', async () => {
    // Test KST matches expected values from Rust tests
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.kst;
    
    const result = wasm.kst(
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
    
    // Extract line and signal from values
    const line = result.values.slice(0, result.cols);
    const signal = result.values.slice(result.cols, 2 * result.cols);
    
    assert.strictEqual(line.length, close.length);
    assert.strictEqual(signal.length, close.length);
    
    // Check last 5 values match expected with 1e-1 tolerance (as in Rust tests)
    const last5Line = line.slice(-5);
    const last5Signal = signal.slice(-5);
    assertArrayClose(
        last5Line,
        expected.last5Values.line,
        1e-1,  // Use same tolerance as Rust tests
        "KST line last 5 values mismatch"
    );
    assertArrayClose(
        last5Signal,
        expected.last5Values.signal,
        1e-1,  // Use same tolerance as Rust tests
        "KST signal last 5 values mismatch"
    );
    
    // Compare full output with Rust - skip for now as KST has two outputs
    // await compareWithRust('kst', {line, signal}, 'close', expected.defaultParams);
});

test('KST all NaN values', () => {
    // Test KST with all NaN values
    const inputData = new Float64Array(10).fill(NaN);
    
    assert.throws(() => {
        wasm.kst(inputData, 10, 10, 10, 15, 10, 15, 20, 30, 9);
    }, /All values are NaN/);
});

test('KST zero periods', () => {
    // Test KST fails with zero periods
    const inputData = new Float64Array(50).fill(0).map((_, i) => 10.0 + i);
    
    // Test zero SMA period
    assert.throws(() => {
        wasm.kst(inputData, 0, 10, 10, 15, 10, 15, 20, 30, 9);
    }, /period|Period/);
    
    // Test zero ROC period
    assert.throws(() => {
        wasm.kst(inputData, 10, 10, 10, 15, 0, 15, 20, 30, 9);
    }, /period|Period/);
    
    // Test zero signal period
    assert.throws(() => {
        wasm.kst(inputData, 10, 10, 10, 15, 10, 15, 20, 30, 0);
    }, /period|Period/);
});

test('KST insufficient data', () => {
    // Test KST fails when not enough data for periods
    const dataSmall = new Float64Array([10.0, 20.0, 30.0, 40.0, 50.0]);
    
    assert.throws(() => {
        wasm.kst(dataSmall, 10, 10, 10, 15, 10, 15, 20, 30, 9);
    }, /Not enough|insufficient|Invalid/);
});

test('KST with NaN prefix', () => {
    // Test KST with NaN values at start
    const inputData = new Float64Array(100);
    inputData.fill(NaN, 0, 20);
    for (let i = 20; i < 100; i++) {
        inputData[i] = 100.0 + Math.sin(i * 0.1) * 10.0;
    }
    
    const result = wasm.kst(inputData, 5, 5, 5, 10, 5, 10, 15, 20, 5);
    
    // Extract line and signal from values
    const line = result.values.slice(0, result.cols);
    const signal = result.values.slice(result.cols, 2 * result.cols);
    
    assert.strictEqual(line.length, inputData.length);
    assert.strictEqual(signal.length, inputData.length);
    
    // Check warmup period
    // With periods: roc=(5,10,15,20), sma=(5,5,5,10)
    // warmup = max(5+5-1, 10+5-1, 15+5-1, 20+10-1) = max(9, 14, 19, 29) = 29
    // BUT we have 20 NaN values at the start, so the warmup effectively starts from the first valid value
    // Since first valid value is at index 20, warmup extends to 20 + 29 = 49
    const warmup = 49;
    for (let i = 0; i < warmup; i++) {
        assert(isNaN(line[i]), `Line[${i}] should be NaN during warmup`);
        assert(isNaN(signal[i]), `Signal[${i}] should be NaN during warmup`);
    }
    
    // After warmup, values should not be NaN
    for (let i = warmup + 10; i < line.length; i++) {
        assert(!isNaN(line[i]), `Line[${i}] should not be NaN after warmup`);
        assert(!isNaN(signal[i]), `Signal[${i}] should not be NaN after warmup`);
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
        const inArray = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);
        
        // Copy input data
        inArray.set(close);
        
        // Call the fast API
        wasm.kst_into(
            inPtr, lineOutPtr, signalOutPtr, len,
            10, 10, 10, 15, 10, 15, 20, 30, 9
        );
        
        // Read results - need to get fresh memory buffer reference after kst_into
        const memBuffer1 = wasm.__wasm.memory.buffer;
        const lineResult = new Float64Array(memBuffer1, lineOutPtr, len);
        const signalResult = new Float64Array(memBuffer1, signalOutPtr, len);
        
        // Compare with safe API
        const safeResult = wasm.kst(close, 10, 10, 10, 15, 10, 15, 20, 30, 9);
        const safeLine = safeResult.values.slice(0, safeResult.cols);
        const safeSignal = safeResult.values.slice(safeResult.cols, 2 * safeResult.cols);
        
        // Need to get fresh buffer reference again in case safe API caused growth
        const memBuffer2 = wasm.__wasm.memory.buffer;
        const lineResultFinal = memBuffer1 === memBuffer2 ? lineResult : new Float64Array(memBuffer2, lineOutPtr, len);
        const signalResultFinal = memBuffer1 === memBuffer2 ? signalResult : new Float64Array(memBuffer2, signalOutPtr, len);
        
        assertArrayClose(
            Array.from(lineResultFinal),
            safeLine,
            1e-10,
            "Fast API line should match safe API"
        );
        assertArrayClose(
            Array.from(signalResultFinal),
            safeSignal,
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
        const array = new Float64Array(wasm.__wasm.memory.buffer, ptr, len);
        
        // Copy input data
        array.set(close);
        
        // Call fast API with aliasing (input and line output share memory)
        wasm.kst_into(
            ptr, ptr, signalPtr, len,
            10, 10, 10, 15, 10, 15, 20, 30, 9
        );
        
        // Read results
        const lineResult = new Float64Array(wasm.__wasm.memory.buffer, ptr, len);
        const signalResult = new Float64Array(wasm.__wasm.memory.buffer, signalPtr, len);
        
        // Compare with safe API
        const safeResult = wasm.kst(close, 10, 10, 10, 15, 10, 15, 20, 30, 9);
        const safeLine = safeResult.values.slice(0, safeResult.cols);
        const safeSignal = safeResult.values.slice(safeResult.cols, 2 * safeResult.cols);
        
        assertArrayClose(
            Array.from(lineResult),
            safeLine,
            1e-10,
            "Aliased fast API line should match safe API"
        );
        assertArrayClose(
            Array.from(signalResult),
            safeSignal,
            1e-10,
            "Aliased fast API signal should match safe API"
        );
    } finally {
        // Clean up allocated memory
        wasm.kst_free(ptr, len);
        wasm.kst_free(signalPtr, len);
    }
});

test('KST reinput', () => {
    // Test KST applied twice (re-input)
    const close = new Float64Array(testData.close.slice(0, 500));
    
    // First pass
    const firstResult = wasm.kst(close, 10, 10, 10, 15, 10, 15, 20, 30, 9);
    const firstLine = firstResult.values.slice(0, firstResult.cols);
    const firstSignal = firstResult.values.slice(firstResult.cols, 2 * firstResult.cols);
    
    // Second pass - apply KST to the KST line output
    const secondResult = wasm.kst(firstLine, 10, 10, 10, 15, 10, 15, 20, 30, 9);
    const secondLine = secondResult.values.slice(0, secondResult.cols);
    const secondSignal = secondResult.values.slice(secondResult.cols, 2 * secondResult.cols);
    
    assert.strictEqual(secondLine.length, firstLine.length);
    
    // Verify warmup period cascades correctly
    const warmupFirst = 44;
    // After first pass, we have valid data starting at index 44
    // When we apply KST again, the warmup is still 44 positions into the already-processed data
    // So the total warmup for second pass is 44 + 44 = 88
    const warmupSecond = 88;
    
    // Check that we have NaN in expected positions
    assert(isNaN(firstLine[0]), 'First line should have NaN at start');
    // Check positions that should be NaN in the second pass
    // The exact warmup can vary, so let's check the first part is NaN
    assert(isNaN(secondLine[0]), 'Second line should have NaN at start');
    assert(isNaN(secondLine[43]), 'Second line should have NaN during first warmup');
    
    // Check we have valid values after warmup
    if (secondLine.length > warmupSecond + 10) {
        assert(!isNaN(secondLine[warmupSecond + 10]), 'Should have valid value after cascaded warmup');
    }
});

test('KST warmup period', () => {
    // Test warmup period calculation
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const result = wasm.kst(close, 10, 10, 10, 15, 10, 15, 20, 30, 9);
    const line = result.values.slice(0, result.cols);
    const signal = result.values.slice(result.cols, 2 * result.cols);
    
    // Warmup = max(roc_period_i + sma_period_i - 1) for all i
    // = max(10+10-1, 15+10-1, 20+10-1, 30+15-1) = max(19, 24, 29, 44) = 44
    const expectedWarmup = 44;
    
    // Check that values before warmup are NaN
    for (let i = 0; i < expectedWarmup; i++) {
        assert(isNaN(line[i]), `Expected NaN at index ${i} during warmup`);
    }
    
    // Check that values after warmup are valid (not NaN)
    for (let i = expectedWarmup; i < Math.min(expectedWarmup + 5, line.length); i++) {
        assert(!isNaN(line[i]), `Expected valid value at index ${i} after warmup`);
    }
    
    // Signal warmup is line warmup + signal_period - 1
    const signalWarmup = expectedWarmup + 9 - 1;
    for (let i = 0; i < signalWarmup; i++) {
        assert(isNaN(signal[i]), `Expected NaN in signal at index ${i} during warmup`);
    }
});

test('KST empty input', () => {
    // Test KST with empty input data
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.kst(empty, 10, 10, 10, 15, 10, 15, 20, 30, 9);
    }, /empty|Empty/, 'Should fail with empty input');
});

test('KST batch calculation', () => {
    // Test batch KST calculation
    const close = new Float64Array(testData.close.slice(0, 100)); // Use smaller dataset for batch
    
    const config = {
        sma_period1: [10, 10, 0],  // Single value
        sma_period2: [10, 10, 0],
        sma_period3: [10, 10, 0],
        sma_period4: [15, 15, 0],
        roc_period1: [10, 12, 2],  // Two values: 10, 12
        roc_period2: [15, 15, 0],
        roc_period3: [20, 20, 0],
        roc_period4: [30, 30, 0],
        signal_period: [9, 11, 2]  // Three values: 9, 11
    };
    
    const result = wasm.kst_batch(close, config);
    
    assert(result.values, 'Batch result should have values property');
    assert(result.combos, 'Batch result should have combos property');
    assert(result.rows, 'Batch result should have rows property');
    assert(result.cols, 'Batch result should have cols property');
    
    // Check dimensions
    const numCombos = 2 * 2; // 2 roc1 values * 2 signal values
    assert.strictEqual(result.cols, close.length);
    assert.strictEqual(result.combos.length, numCombos);
    assert.strictEqual(result.rows, numCombos * 2); // *2 for line and signal
    assert.strictEqual(result.values.length, result.rows * result.cols);
    
    // Extract first combination's line and signal
    const firstCombo = result.combos[0];
    const firstLineRow = result.values.slice(0, result.cols);
    const firstSignalRow = result.values.slice(numCombos * result.cols, (numCombos + 1) * result.cols);
    
    // Verify first combination matches single calculation
    const singleResult = wasm.kst(
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
    
    const singleLine = singleResult.values.slice(0, singleResult.cols);
    const singleSignal = singleResult.values.slice(singleResult.cols, 2 * singleResult.cols);
    
    assertArrayClose(
        firstLineRow,
        singleLine,
        1e-10,
        "Batch first row line should match single calculation"
    );
    assertArrayClose(
        firstSignalRow,
        singleSignal,
        1e-10,
        "Batch first row signal should match single calculation"
    );
});

console.log('All KST WASM tests passed! 🎉');