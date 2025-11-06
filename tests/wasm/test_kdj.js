/**
 * WASM binding tests for KDJ indicator.
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

// ========== Basic Functionality Tests ==========

test('KDJ accuracy', () => {
    const high = testData.high;
    const low = testData.low;
    const close = testData.close;
    
    // Test with default parameters
    const result = wasm.kdj(high, low, close, 9, 3, "sma", 3, "sma");
    
    assert.ok(result, 'KDJ should return a result');
    assert.equal(result.rows, 3, 'Should have 3 rows (K, D, J)');
    assert.equal(result.cols, close.length, 'Should have same number of columns as input');
    assert.equal(result.values.length, close.length * 3, 'Values array should be flattened K, D, J');
    
    // Extract K, D, J from flattened array
    const k = result.values.slice(0, close.length);
    const d = result.values.slice(close.length, close.length * 2);
    const j = result.values.slice(close.length * 2);
    
    // Expected values from Rust tests (last 5 values)
    const expectedK = [
        58.04341315415984,
        61.56034740940419,
        58.056304282719545,
        56.10961365678364,
        51.43992326447119,
    ];
    const expectedD = [
        49.57659409278555,
        56.81719223571944,
        59.22002161542779,
        58.57542178296905,
        55.20194706799139,
    ];
    const expectedJ = [
        74.97705127690843,
        71.04665775677368,
        55.72886961730306,
        51.17799740441281,
        43.91587565743079,
    ];
    
    // Check last 5 values
    assertArrayClose(k.slice(-5), expectedK, 'KDJ K last 5 values mismatch');
    assertArrayClose(d.slice(-5), expectedD, 'KDJ D last 5 values mismatch');
    assertArrayClose(j.slice(-5), expectedJ, 'KDJ J last 5 values mismatch');
});

test('KDJ warmup period validation', () => {
    const high = testData.high;
    const low = testData.low;
    const close = testData.close;
    
    const fastK = 9;
    const slowK = 3;
    const slowD = 3;
    
    const result = wasm.kdj(high, low, close, fastK, slowK, "sma", slowD, "sma");
    
    // Extract K, D, J from flattened array
    const k = result.values.slice(0, close.length);
    const d = result.values.slice(close.length, close.length * 2);
    const j = result.values.slice(close.length * 2);
    
    // Find first valid index
    let firstValid = 0;
    for (let i = 0; i < high.length; i++) {
        if (!isNaN(high[i]) && !isNaN(low[i]) && !isNaN(close[i])) {
            firstValid = i;
            break;
        }
    }
    
    // Calculate warmup periods
    const kWarmup = firstValid + fastK + slowK - 2;
    const dWarmup = kWarmup + slowD - 1;
    const jWarmup = dWarmup;
    
    // Verify K warmup
    for (let i = 0; i < Math.min(kWarmup, k.length); i++) {
        assert.ok(isNaN(k[i]), `Expected NaN in K warmup at index ${i}`);
    }
    if (kWarmup < k.length) {
        assert.ok(!isNaN(k[kWarmup]), `Expected valid value in K after warmup at index ${kWarmup}`);
    }
    
    // Verify D warmup
    for (let i = 0; i < Math.min(dWarmup, d.length); i++) {
        assert.ok(isNaN(d[i]), `Expected NaN in D warmup at index ${i}`);
    }
    if (dWarmup < d.length) {
        assert.ok(!isNaN(d[dWarmup]), `Expected valid value in D after warmup at index ${dWarmup}`);
    }
    
    // Verify J warmup
    for (let i = 0; i < Math.min(jWarmup, j.length); i++) {
        assert.ok(isNaN(j[i]), `Expected NaN in J warmup at index ${i}`);
    }
    if (jWarmup < j.length) {
        assert.ok(!isNaN(j[jWarmup]), `Expected valid value in J after warmup at index ${jWarmup}`);
    }
});

test('KDJ with different MA types', () => {
    const high = testData.high;
    const low = testData.low;
    const close = testData.close;
    
    // Test various MA type combinations
    const maTypes = ["sma", "ema", "wma"];
    
    for (const slowKMa of maTypes) {
        for (const slowDMa of maTypes) {
            const result = wasm.kdj(high, low, close, 9, 3, slowKMa, 3, slowDMa);
            
            assert.equal(result.rows, 3, `Should have 3 rows with MA types ${slowKMa}/${slowDMa}`);
            assert.equal(result.cols, close.length, `Should have correct cols with MA types ${slowKMa}/${slowDMa}`);
            
            // Extract K values and verify some are valid
            const k = result.values.slice(0, close.length);
            const validK = k.filter(v => !isNaN(v));
            assert.ok(validK.length > 0, `Should have valid K values with MA types ${slowKMa}/${slowDMa}`);
        }
    }
});

// ========== Error Handling Tests ==========

test('KDJ error handling', () => {
    // Test empty input
    assert.throws(() => {
        wasm.kdj([], [], [], 9, 3, "sma", 3, "sma");
    }, 'Should throw on empty input');
    
    // Test period exceeds length
    const shortData = [1, 2, 3];
    assert.throws(() => {
        wasm.kdj(shortData, shortData, shortData, 10, 3, "sma", 3, "sma");
    }, 'Should throw when period exceeds data length');
    
    // Test zero period
    const data = [1, 2, 3, 4, 5];
    assert.throws(() => {
        wasm.kdj(data, data, data, 0, 3, "sma", 3, "sma");
    }, 'Should throw on zero period');
    
    // Test all NaN values
    const nanData = [NaN, NaN, NaN, NaN, NaN];
    assert.throws(() => {
        wasm.kdj(nanData, nanData, nanData, 3, 2, "sma", 2, "sma");
    }, 'Should throw on all NaN values');
});

test('KDJ with all NaN large dataset', () => {
    const allNaN = new Array(100).fill(NaN);
    
    assert.throws(() => {
        wasm.kdj(allNaN, allNaN, allNaN, 9, 3, "sma", 3, "sma");
    }, 'Should throw on large all-NaN dataset');
});

// ========== Advanced Tests ==========

test('KDJ reinput', () => {
    const high = testData.high;
    const low = testData.low;
    const close = testData.close;
    
    // First pass
    const firstResult = wasm.kdj(high, low, close, 9, 3, "sma", 3, "sma");
    const firstK = firstResult.values.slice(0, close.length);
    const firstD = firstResult.values.slice(close.length, close.length * 2);
    const firstJ = firstResult.values.slice(close.length * 2);
    
    // Second pass - apply KDJ to KDJ output (using K as all three inputs)
    const secondResult = wasm.kdj(firstK, firstK, firstK, 9, 3, "sma", 3, "sma");
    
    assert.equal(secondResult.rows, 3, 'Reinput should produce 3 rows');
    assert.equal(secondResult.cols, close.length, 'Reinput should maintain length');
    
    // Extract values and verify some are valid
    const secondK = secondResult.values.slice(0, close.length);
    const validSecondK = secondK.filter(v => !isNaN(v));
    assert.ok(validSecondK.length > 0, 'Reinput should produce some valid values');
});

test('KDJ NaN handling', () => {
    const high = testData.high;
    const low = testData.low;
    const close = testData.close;
    
    const result = wasm.kdj(high, low, close, 9, 3, "sma", 3, "sma");
    
    // Extract K, D, J from flattened array
    const k = result.values.slice(0, close.length);
    const d = result.values.slice(close.length, close.length * 2);
    const j = result.values.slice(close.length * 2);
    
    // After warmup period (50), no NaN values should exist
    if (k.length > 50) {
        for (let i = 50; i < k.length; i++) {
            assert.ok(!isNaN(k[i]), `Found unexpected NaN in K at index ${i}`);
            assert.ok(!isNaN(d[i]), `Found unexpected NaN in D at index ${i}`);
            assert.ok(!isNaN(j[i]), `Found unexpected NaN in J at index ${i}`);
        }
    }
});

test('KDJ with partial NaN data', () => {
    // Create controlled synthetic data
    const size = 100;
    const high = new Array(size);
    const low = new Array(size);
    const close = new Array(size);
    
    // Generate simple trending data
    for (let i = 0; i < size; i++) {
        close[i] = 100 + i * 0.1 + Math.sin(i * 0.1) * 2;
        high[i] = close[i] + Math.abs(Math.sin(i * 0.2));
        low[i] = close[i] - Math.abs(Math.sin(i * 0.2));
    }
    
    // Inject a single NaN early
    high[15] = NaN;
    low[15] = NaN;
    close[15] = NaN;
    
    const result = wasm.kdj(high, low, close, 9, 3, "sma", 3, "sma");
    
    assert.equal(result.rows, 3, 'Should handle partial NaN data');
    assert.equal(result.cols, close.length, 'Should maintain length with partial NaN');
    
    // Extract K and verify we have valid values before NaN
    const k = result.values.slice(0, close.length);
    const validBefore = k.slice(12, 15).filter(v => !isNaN(v));
    assert.ok(validBefore.length > 0, 'Should have valid values before NaN');
    
    // Note: KDJ doesn't recover from NaN gaps - this is expected behavior
});

test('KDJ numerical stability', () => {
    const size = 50;
    
    // Very large values
    const largeHigh = new Array(size).fill(1e10);
    const largeLow = new Array(size).fill(1e10 - 100);
    const largeClose = new Array(size).fill(1e10 - 50);
    
    const largeResult = wasm.kdj(largeHigh, largeLow, largeClose, 9, 3, "sma", 3, "sma");
    const largeK = largeResult.values.slice(0, size);
    const validLargeK = largeK.filter(v => !isNaN(v));
    
    // Should not produce infinity and should be in [0, 100] range
    for (const val of validLargeK) {
        assert.ok(isFinite(val), 'K should be finite for large values');
        assert.ok(val >= 0 && val <= 100, `K should be in [0, 100] range, got ${val}`);
    }
    
    // Very small positive values
    const smallHigh = new Array(size).fill(1e-10);
    const smallLow = new Array(size).fill(1e-10 - 1e-12);
    const smallClose = new Array(size).fill(1e-10 - 5e-13);
    
    const smallResult = wasm.kdj(smallHigh, smallLow, smallClose, 9, 3, "sma", 3, "sma");
    const smallK = smallResult.values.slice(0, size);
    const validSmallK = smallK.filter(v => !isNaN(v));
    
    for (const val of validSmallK) {
        assert.ok(isFinite(val), 'K should be finite for small values');
    }
});

// ========== Fast API Tests ==========

test('KDJ fast API basic (no input aliasing)', () => {
    const high = testData.high;
    const low = testData.low;
    const close = testData.close;
    const len = close.length;
    
    // Allocate output memory
    const kPtr = wasm.kdj_alloc(len);
    const dPtr = wasm.kdj_alloc(len);
    const jPtr = wasm.kdj_alloc(len);
    
    let highPtr, lowPtr, closePtr;
    
    try {
        // Create input memory
        highPtr = wasm.kdj_alloc(len);
        lowPtr = wasm.kdj_alloc(len);
        closePtr = wasm.kdj_alloc(len);
        
        const memory = wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : (wasm.__wasm ? wasm.__wasm.memory : wasm.memory);
        const highMem = new Float64Array(memory.buffer, highPtr, len);
        const lowMem = new Float64Array(memory.buffer, lowPtr, len);
        const closeMem = new Float64Array(memory.buffer, closePtr, len);
        
        highMem.set(high);
        lowMem.set(low);
        closeMem.set(close);
        
        // Test normal operation (no aliasing)
        wasm.kdj_into(
            highPtr, lowPtr, closePtr,
            kPtr, dPtr, jPtr,
            len, 9, 3, "sma", 3, "sma"
        );
        
        const kResult = new Float64Array(memory.buffer, kPtr, len);
        const dResult = new Float64Array(memory.buffer, dPtr, len);
        const jResult = new Float64Array(memory.buffer, jPtr, len);
        
        // Verify some values exist
        assert.ok(!isNaN(kResult[len - 1]), 'K should have valid values');
        assert.ok(!isNaN(dResult[len - 1]), 'D should have valid values');
        assert.ok(!isNaN(jResult[len - 1]), 'J should have valid values');
        
        // Note: For KDJ, aliasing outputs to input buffers is not supported
        // because inputs are read across a rolling window. This test verifies
        // the fast API with distinct input/output buffers.
        
    } finally {
        // Clean up all allocated memory
        if (highPtr) wasm.kdj_free(highPtr, len);
        if (lowPtr) wasm.kdj_free(lowPtr, len);
        if (closePtr) wasm.kdj_free(closePtr, len);
        wasm.kdj_free(kPtr, len);
        wasm.kdj_free(dPtr, len);
        wasm.kdj_free(jPtr, len);
    }
});

// ========== Batch Processing Tests ==========

test('KDJ batch processing', () => {
    const high = testData.high;
    const low = testData.low;
    const close = testData.close;
    
    const config = {
        fast_k_period: [5, 15, 5],      // 5, 10, 15
        slow_k_period: [3, 6, 3],       // 3, 6
        slow_k_ma_type: "sma",
        slow_d_period: [3, 6, 3],       // 3, 6
        slow_d_ma_type: "sma"
    };
    
    const result = wasm.kdj_batch(high, low, close, config);
    
    assert.ok(result, 'Batch should return a result');
    assert.ok(result.combos, 'Should have parameter combinations');
    
    // Should have 3 * 2 * 2 = 12 combinations
    const expectedCombos = 3 * 2 * 2;
    assert.equal(result.combos.length, expectedCombos, `Should have ${expectedCombos} combinations`);
    assert.equal(result.rows, expectedCombos * 3, `Should have ${expectedCombos * 3} rows (K, D, J for each combo)`);
    assert.equal(result.cols, close.length, 'Should have same columns as input');
    assert.equal(result.values.length, expectedCombos * 3 * close.length, 'Values should contain all K, D, J for all combos');
    
    // Verify parameter combinations
    const fastKValues = [5, 10, 15];
    const slowKValues = [3, 6];
    const slowDValues = [3, 6];
    
    let comboIndex = 0;
    for (const fastK of fastKValues) {
        for (const slowK of slowKValues) {
            for (const slowD of slowDValues) {
                const combo = result.combos[comboIndex];
                assert.equal(combo.fast_k_period, fastK, `Combo ${comboIndex} fast_k mismatch`);
                assert.equal(combo.slow_k_period, slowK, `Combo ${comboIndex} slow_k mismatch`);
                assert.equal(combo.slow_d_period, slowD, `Combo ${comboIndex} slow_d mismatch`);
                
                // Verify warmup period for this combo
                const expectedWarmup = fastK + slowK + slowD - 3;
                const baseIndex = comboIndex * 3 * close.length;
                const kValues = result.values.slice(baseIndex, baseIndex + close.length);
                
                if (expectedWarmup > 0 && expectedWarmup < kValues.length) {
                    assert.ok(isNaN(kValues[0]), `Expected NaN in warmup for combo ${comboIndex}`);
                }
                
                comboIndex++;
            }
        }
    }
});

test('KDJ batch with single parameter set', () => {
    const high = testData.high;
    const low = testData.low;
    const close = testData.close;
    
    const config = {
        fast_k_period: [9, 9, 0],  // Single value
        slow_k_period: [3, 3, 0],  // Single value
        slow_k_ma_type: "sma",
        slow_d_period: [3, 3, 0],  // Single value
        slow_d_ma_type: "sma"
    };
    
    const result = wasm.kdj_batch(high, low, close, config);
    
    assert.equal(result.combos.length, 1, 'Should have 1 combination');
    assert.equal(result.rows, 3, 'Should have 3 rows (K, D, J)');
    assert.equal(result.cols, close.length, 'Should have same columns as input');
});

test.after(() => {
    console.log('KDJ WASM tests completed');
});
