/**
 * WASM binding tests for VWMACD indicator.
 * These tests mirror the Rust unit tests to ensure WASM bindings work correctly.
 */
const test = require('node:test');
const assert = require('node:assert');
const path = require('path');
const { 
    loadTestData, 
    assertArrayClose, 
    assertClose,
    isNaN,
    assertAllNaN,
    assertNoNaN,
    EXPECTED_OUTPUTS 
} = require('./test_utils');

let wasm;
let testData;

test.before(async () => {
    // Load WASM module
    try {
        const wasmPath = path.join(__dirname, '../../pkg/my_project.js');
        wasm = await import(wasmPath);
        await wasm.default();
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }
    
    testData = loadTestData();
});

test('VWMACD accuracy', () => {
    // Test VWMACD matches expected values from Rust tests
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    // Test with default parameters
    const result = wasm.vwmacd_js(close, volume, 12, 26, 9, 'sma', 'sma', 'ema');
    
    assert.ok(result.macd, 'Result should have macd array');
    assert.ok(result.signal, 'Result should have signal array');
    assert.ok(result.hist, 'Result should have hist array');
    assert.strictEqual(result.macd.length, close.length);
    assert.strictEqual(result.signal.length, close.length);
    assert.strictEqual(result.hist.length, close.length);
    
    // Expected values from Rust tests
    const expectedMacd = [
        -394.95161155,
        -508.29106210,
        -490.70190723,
        -388.94996199,
        -341.13720646,
    ];
    
    const expectedSignal = [
        -539.48861567,
        -533.24910496,
        -524.73966541,
        -497.58172247,
        -466.29282108,
    ];
    
    const expectedHistogram = [
        144.53700412,
        24.95804286,
        34.03775818,
        108.63176274,
        125.15561462,
    ];
    
    // Check last 5 values
    const macdLast5 = result.macd.slice(-5);
    const signalLast5 = result.signal.slice(-5);
    const histLast5 = result.hist.slice(-5);
    
    assertArrayClose(macdLast5, expectedMacd, 1e-3, "MACD last 5 values mismatch");
    assertArrayClose(signalLast5, expectedSignal, 1e-3, "Signal last 5 values mismatch");
    assertArrayClose(histLast5, expectedHistogram, 1e-3, "Histogram last 5 values mismatch");
});

test('VWMACD custom MA types', () => {
    // Test VWMACD with custom MA types
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    // Test with custom MA types
    const result = wasm.vwmacd_js(
        close, volume, 
        12, 26, 9,
        'ema', 'wma', 'sma'
    );
    
    assert.ok(result.macd);
    assert.ok(result.signal);
    assert.ok(result.hist);
    assert.strictEqual(result.macd.length, close.length);
});

test('VWMACD error handling - NaN data', () => {
    // Test with all NaN data
    const nanData = new Float64Array(10).fill(NaN);
    
    assert.throws(() => {
        wasm.vwmacd_js(nanData, nanData, 12, 26, 9, 'sma', 'sma', 'ema');
    }, /All values are NaN/);
});

test('VWMACD error handling - zero period', () => {
    // Test with zero period
    const data = new Float64Array([1.0, 2.0, 3.0]);
    const volume = new Float64Array([100.0, 200.0, 300.0]);
    
    assert.throws(() => {
        wasm.vwmacd_js(data, volume, 0, 26, 9, 'sma', 'sma', 'ema');
    }, /Invalid period/);
});

test('VWMACD error handling - period exceeds data', () => {
    // Test with period exceeding data length
    const data = new Float64Array([1.0, 2.0, 3.0]);
    const volume = new Float64Array([100.0, 200.0, 300.0]);
    
    assert.throws(() => {
        wasm.vwmacd_js(data, volume, 12, 10, 9, 'sma', 'sma', 'ema');
    }, /Invalid period/);
});

test('VWMACD batch processing', () => {
    // Test batch processing
    const close = new Float64Array(testData.close.slice(0, 100)); // Use smaller dataset
    const volume = new Float64Array(testData.volume.slice(0, 100));
    
    const config = {
        fast_range: [10, 14, 2],
        slow_range: [20, 26, 3],
        signal_range: [5, 9, 2]
    };
    
    const result = wasm.vwmacd_batch(close, volume, config);
    
    assert.ok(result.values, 'Result should have values array');
    assert.ok(result.combos, 'Result should have combos array');
    assert.ok(result.rows, 'Result should have rows');
    assert.ok(result.cols, 'Result should have cols');
    
    // Should have 3 * 3 * 3 = 27 combinations
    const expectedCombos = 3 * 3 * 3;
    assert.strictEqual(result.combos.length, expectedCombos);
    assert.strictEqual(result.rows, expectedCombos);
    assert.strictEqual(result.cols, close.length);
    
    // Values should contain MACD results only (batch output doesn't include signal/hist)
    assert.strictEqual(result.values.length, expectedCombos * close.length);
});

test('VWMACD fast API - basic', () => {
    // Test fast API without aliasing
    const close = new Float64Array(testData.close.slice(0, 100));
    const volume = new Float64Array(testData.volume.slice(0, 100));
    const len = close.length;
    
    // Allocate output buffers
    const macdPtr = wasm.vwmacd_alloc(len);
    const signalPtr = wasm.vwmacd_alloc(len);
    const histPtr = wasm.vwmacd_alloc(len);
    
    try {
        // Compute VWMACD
        wasm.vwmacd_into(
            close, volume,
            macdPtr, signalPtr, histPtr,
            len,
            12, 26, 9,
            'sma', 'sma', 'ema'
        );
        
        // Read results
        const macd = new Float64Array(wasm.memory.buffer, macdPtr, len);
        const signal = new Float64Array(wasm.memory.buffer, signalPtr, len);
        const hist = new Float64Array(wasm.memory.buffer, histPtr, len);
        
        // Verify some values are not NaN after warmup
        assert.ok(macd.slice(30).some(v => !isNaN(v)), 'MACD should have valid values');
        assert.ok(signal.slice(30).some(v => !isNaN(v)), 'Signal should have valid values');
        assert.ok(hist.slice(30).some(v => !isNaN(v)), 'Histogram should have valid values');
        
    } finally {
        // Clean up
        wasm.vwmacd_free(macdPtr, len);
        wasm.vwmacd_free(signalPtr, len);
        wasm.vwmacd_free(histPtr, len);
    }
});

test('VWMACD fast API - aliasing', () => {
    // Test fast API with aliasing (output same as input)
    const data = new Float64Array(testData.close.slice(0, 100));
    const volume = new Float64Array(testData.volume.slice(0, 100));
    const len = data.length;
    
    // Create copies for comparison
    const originalData = new Float64Array(data);
    
    // Use data buffer as output (aliasing)
    const dataPtr = data.byteOffset;
    const volumePtr = volume.byteOffset;
    
    // This should handle aliasing correctly
    wasm.vwmacd_into(
        data, volume,
        dataPtr, volumePtr, dataPtr,  // Reuse input pointers
        len,
        12, 26, 9,
        'sma', 'sma', 'ema'
    );
    
    // Verify data was modified (not equal to original)
    assert.ok(
        !data.every((v, i) => v === originalData[i]),
        'Data should be modified when used as output'
    );
});

test.after(() => {
    console.log('VWMACD WASM tests completed');
});
