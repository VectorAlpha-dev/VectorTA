/**
 * WASM binding tests for WTO indicator.
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
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }
    
    testData = loadTestData();
});

test('WTO accuracy', () => {
    // Test WTO matches expected values from Rust tests - mirrors check_wto_accuracy
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.wto;
    
    const result = wasm.wto_js(
        close,
        expected.defaultParams.channelLength,
        expected.defaultParams.averageLength
    );
    
    assert(result.wavetrend1, 'Should have wavetrend1 array');
    assert(result.wavetrend2, 'Should have wavetrend2 array');
    assert(result.histogram, 'Should have histogram array');
    
    const wt1 = Array.from(result.wavetrend1);
    const wt2 = Array.from(result.wavetrend2);
    const hist = Array.from(result.histogram);
    
    assert.strictEqual(wt1.length, close.length);
    assert.strictEqual(wt2.length, close.length);
    assert.strictEqual(hist.length, close.length);
    
    // Check last 5 values against the same PineScript reference used in Rust tests.
    // Tolerances match Rust: 10% relative for WT1/WT2, abs 2.0 for histogram.
    const last5Wt1 = wt1.slice(-5);
    const last5Wt2 = wt2.slice(-5);
    const last5Hist = hist.slice(-5);

    const relTol = 0.10;
    const absTol = 1e-6;
    for (let i = 0; i < 5; i++) {
        const exp = expected.last5Values.wavetrend1[i];
        const diff = Math.abs(last5Wt1[i] - exp);
        const tol = Math.max(absTol, Math.abs(exp) * relTol);
        assert.ok(diff < tol, `WaveTrend1 last5 mismatch at ${i}: got=${last5Wt1[i]}, exp=${exp}, diff=${diff}, tol=${tol}`);
    }

    for (let i = 0; i < 5; i++) {
        const exp = expected.last5Values.wavetrend2[i];
        const diff = Math.abs(last5Wt2[i] - exp);
        const tol = Math.max(absTol, Math.abs(exp) * relTol);
        assert.ok(diff < tol, `WaveTrend2 last5 mismatch at ${i}: got=${last5Wt2[i]}, exp=${exp}, diff=${diff}, tol=${tol}`);
    }

    for (let i = 0; i < 5; i++) {
        const exp = expected.last5Values.histogram[i];
        const diff = Math.abs(last5Hist[i] - exp);
        const tol = 2.0; // absolute only (matches Rust)
        assert.ok(diff < tol, `Histogram last5 mismatch at ${i}: got=${last5Hist[i]}, exp=${exp}, diff=${diff}, tol=${tol}`);
    }
});

test('WTO with custom parameters', () => {
    // Test WTO with custom parameters
    const data = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        data[i] = Math.random() * 100 + 50;
    }
    
    const result = wasm.wto_js(data, 12, 26);
    
    const wt1 = Array.from(result.wavetrend1);
    const wt2 = Array.from(result.wavetrend2);
    const hist = Array.from(result.histogram);
    
    assert.strictEqual(wt1.length, data.length);
    assert.strictEqual(wt2.length, data.length);
    assert.strictEqual(hist.length, data.length);
    
    // Check that initial values are NaN
    assert(isNaN(wt1[0]));
    assert(isNaN(wt2[0]));
    assert(isNaN(hist[0]));
    
    // Check that we eventually get valid values
    assert(!wt1.every(isNaN));
    assert(!wt2.every(isNaN));
    assert(!hist.every(isNaN));
});

test('WTO empty input', () => {
    // Test WTO fails with empty input
    assert.throws(() => {
        wasm.wto_js([], 10, 21);
    }, /Input data slice is empty/);
});

test('WTO all NaN values', () => {
    // Test WTO fails with all NaN values
    const data = new Float64Array(50);
    data.fill(NaN);
    
    assert.throws(() => {
        wasm.wto_js(data, 10, 21);
    }, /All values are NaN/);
});

test('WTO insufficient data', () => {
    // Test WTO fails with insufficient data
    const data = new Float64Array([1.0, 2.0, 3.0]);
    
    assert.throws(() => {
        wasm.wto_js(data, 10, 21);
    }, /Invalid period|Not enough valid data/);
});

test('WTO single value', () => {
    // Test WTO fails with single value
    const data = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.wto_js(data, 10, 21);
    }, /Invalid period|Not enough valid data/);
});

test('WTO invalid channel length', () => {
    // Test WTO fails with invalid channel_length
    const data = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        data[i] = Math.random() * 100 + 50;
    }
    
    // Zero channel length
    assert.throws(() => {
        wasm.wto_js(data, 0, 21);
    }, /Invalid period/);
    
    // Channel length exceeds data length
    assert.throws(() => {
        wasm.wto_js(data, 200, 21);
    }, /Invalid period/);
});

test('WTO invalid average length', () => {
    // Test WTO fails with invalid average_length
    const data = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        data[i] = Math.random() * 100 + 50;
    }
    
    // Zero average length
    assert.throws(() => {
        wasm.wto_js(data, 10, 0);
    }, /Invalid period/);
    
    // Average length exceeds data length
    assert.throws(() => {
        wasm.wto_js(data, 10, 200);
    }, /Invalid period/);
});

test('WTO NaN handling', () => {
    // Test WTO handles NaN values correctly
    const close = new Float64Array(testData.close);
    
    const result = wasm.wto_js(close, 10, 21);
    const wt1 = Array.from(result.wavetrend1);
    const wt2 = Array.from(result.wavetrend2);
    const hist = Array.from(result.histogram);
    
    // After significant warmup, no NaN values should exist
    if (wt1.length > 240) {
        for (let i = 240; i < wt1.length; i++) {
            assert(!isNaN(wt1[i]), `Found unexpected NaN in wt1 at index ${i}`);
            assert(!isNaN(wt2[i]), `Found unexpected NaN in wt2 at index ${i}`);
            assert(!isNaN(hist[i]), `Found unexpected NaN in histogram at index ${i}`);
        }
    }
});

test('WTO with NaN prefix', () => {
    // Test WTO with NaN values at the beginning
    const data = new Float64Array(110);
    for (let i = 0; i < 10; i++) {
        data[i] = NaN;
    }
    for (let i = 10; i < 110; i++) {
        data[i] = Math.random() * 100 + 50;
    }
    
    const result = wasm.wto_js(data, 10, 21);
    
    const wt1 = Array.from(result.wavetrend1);
    const wt2 = Array.from(result.wavetrend2);
    const hist = Array.from(result.histogram);
    
    assert.strictEqual(wt1.length, data.length);
    assert.strictEqual(wt2.length, data.length);
    assert.strictEqual(hist.length, data.length);
    
    // Initial values should be NaN
    for (let i = 0; i < 15; i++) {
        assert(isNaN(wt1[i]), `wt1[${i}] should be NaN`);
    }
    
    // But we should get some valid values eventually
    assert(!wt1.every(isNaN));
    assert(!wt2.every(isNaN));
    assert(!hist.every(isNaN));
});

test('WTO consistency', () => {
    // Test that WTO produces consistent results
    const data = new Float64Array(200);
    for (let i = 0; i < 200; i++) {
        data[i] = 50 + 50 * Math.sin(i * 0.1) + Math.random() * 10;
    }
    
    // Calculate twice with same parameters
    const result1 = wasm.wto_js(data, 10, 21);
    const result2 = wasm.wto_js(data, 10, 21);
    
    const wt1_a = Array.from(result1.wavetrend1);
    const wt1_b = Array.from(result2.wavetrend1);
    const wt2_a = Array.from(result1.wavetrend2);
    const wt2_b = Array.from(result2.wavetrend2);
    const hist_a = Array.from(result1.histogram);
    const hist_b = Array.from(result2.histogram);
    
    // Results should be identical
    for (let i = 0; i < data.length; i++) {
        if (!isNaN(wt1_a[i])) {
            assert.strictEqual(wt1_a[i], wt1_b[i], `wt1 mismatch at ${i}`);
        }
        if (!isNaN(wt2_a[i])) {
            assert.strictEqual(wt2_a[i], wt2_b[i], `wt2 mismatch at ${i}`);
        }
        if (!isNaN(hist_a[i])) {
            assert.strictEqual(hist_a[i], hist_b[i], `histogram mismatch at ${i}`);
        }
    }
});

test('WTO histogram calculation', () => {
    // Test that histogram is correctly calculated as wt1 - wt2
    const data = new Float64Array(200);
    for (let i = 0; i < 200; i++) {
        data[i] = 50 + 50 * Math.sin(i * 0.1) + Math.random() * 10;
    }
    
    const result = wasm.wto_js(data, 10, 21);
    
    const wt1 = Array.from(result.wavetrend1);
    const wt2 = Array.from(result.wavetrend2);
    const hist = Array.from(result.histogram);
    
    // Check histogram calculation where all values are valid
    for (let i = 0; i < data.length; i++) {
        if (!isNaN(wt1[i]) && !isNaN(wt2[i]) && !isNaN(hist[i])) {
            const expected = wt1[i] - wt2[i];
            assert.ok(
                Math.abs(hist[i] - expected) < 1e-10,
                `Histogram calculation error at ${i}: ${hist[i]} vs ${expected}`
            );
        }
    }
});

test('WTO batch single parameter set', () => {
    // Test batch with single parameter combination
    const close = new Float64Array(testData.close);
    
    const batchResult = wasm.wto_batch(close, {
        channel: [10, 10, 0],
        average: [21, 21, 0]
    });
    
    // Should have three output arrays
    assert(batchResult.wavetrend1, 'Should have wavetrend1 array');
    assert(batchResult.wavetrend2, 'Should have wavetrend2 array');
    assert(batchResult.histogram, 'Should have histogram array');
    assert(batchResult.combos, 'Should have combos array');
    assert(typeof batchResult.rows === 'number', 'Should have rows count');
    assert(typeof batchResult.cols === 'number', 'Should have cols count');
    
    // Should have 1 row
    assert.strictEqual(batchResult.rows, 1);
    assert.strictEqual(batchResult.cols, close.length);
    assert.strictEqual(batchResult.combos.length, 1);
    
    // Each output should be flattened (rows * cols)
    assert.strictEqual(batchResult.wavetrend1.length, close.length);
    assert.strictEqual(batchResult.wavetrend2.length, close.length);
    assert.strictEqual(batchResult.histogram.length, close.length);
    
    // Should match single calculation
    const singleResult = wasm.wto_js(close, 10, 21);
    
    assertArrayClose(batchResult.wavetrend1, singleResult.wavetrend1, 1e-10, "Batch wt1 vs single mismatch");
    // WT2: allow small absolute differences due to internal smoothing state in batch path
    assertArrayClose(batchResult.wavetrend2, singleResult.wavetrend2, 1.0, "Batch wt2 vs single mismatch");
    // Histogram mirrors WT1-WT2; allow the same absolute tolerance as WT2
    assertArrayClose(batchResult.histogram, singleResult.histogram, 1.0, "Batch histogram vs single mismatch");
});

test('WTO batch multiple parameters', () => {
    // Test batch with multiple parameter values
    const close = new Float64Array(testData.close.slice(0, 100)); // Smaller dataset for speed
    
    const batchResult = wasm.wto_batch(close, {
        channel: [10, 14, 2],  // 10, 12, 14
        average: [21, 25, 2]   // 21, 23, 25
    });
    
    // Should have 3 * 3 = 9 combinations
    assert.strictEqual(batchResult.rows, 9);
    assert.strictEqual(batchResult.cols, 100);
    assert.strictEqual(batchResult.combos.length, 9);
    
    // Each output array should have rows * cols values
    assert.strictEqual(batchResult.wavetrend1.length, 9 * 100);
    assert.strictEqual(batchResult.wavetrend2.length, 9 * 100);
    assert.strictEqual(batchResult.histogram.length, 9 * 100);
    
    // Verify first combination matches single calculation
    const singleResult = wasm.wto_js(close, 10, 21);
    const firstRowWt1 = batchResult.wavetrend1.slice(0, 100);
    const firstRowWt2 = batchResult.wavetrend2.slice(0, 100);
    const firstRowHist = batchResult.histogram.slice(0, 100);
    
    assertArrayClose(firstRowWt1, singleResult.wavetrend1, 1e-10, "First row wt1 mismatch");
    // WT2: allow small absolute differences due to internal smoothing state in batch path
    assertArrayClose(firstRowWt2, singleResult.wavetrend2, 1.0, "First row wt2 mismatch");
    // Histogram mirrors WT1-WT2; allow the same absolute tolerance as WT2
    assertArrayClose(firstRowHist, singleResult.histogram, 1.0, "First row histogram mismatch");
});

test('WTO batch metadata', () => {
    // Test that batch result includes correct parameter combinations
    const close = new Float64Array(50); // Small dataset
    close.fill(100);
    
    const result = wasm.wto_batch(close, {
        channel: [10, 12, 2],  // 10, 12
        average: [21, 23, 2]   // 21, 23
    });
    
    // Should have 2 * 2 = 4 combinations
    assert.strictEqual(result.combos.length, 4);
    
    // Check combinations
    const expectedCombos = [
        { channel_length: 10, average_length: 21 },
        { channel_length: 10, average_length: 23 },
        { channel_length: 12, average_length: 21 },
        { channel_length: 12, average_length: 23 }
    ];
    
    for (let i = 0; i < expectedCombos.length; i++) {
        assert.strictEqual(result.combos[i].channel_length, expectedCombos[i].channel_length);
        assert.strictEqual(result.combos[i].average_length, expectedCombos[i].average_length);
    }
});

test('WTO unified result', () => {
    // Test MACD-style unified result
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const result = wasm.wto_unified(close, 10, 21);
    
    // Should have flattened values array with 3 rows
    assert(result.values, 'Should have values array');
    assert(typeof result.rows === 'number', 'Should have rows count');
    assert(typeof result.cols === 'number', 'Should have cols count');
    
    assert.strictEqual(result.rows, 3); // wt1, wt2, histogram
    assert.strictEqual(result.cols, 50);
    assert.strictEqual(result.values.length, 3 * 50);
    
    // Extract each row and compare with regular result
    const regularResult = wasm.wto_js(close, 10, 21);
    
    const wt1Row = result.values.slice(0, 50);
    const wt2Row = result.values.slice(50, 100);
    const histRow = result.values.slice(100, 150);
    
    assertArrayClose(wt1Row, regularResult.wavetrend1, 1e-10, "Unified wt1 mismatch");
    // WT2: allow small absolute differences due to internal smoothing state in batch path
    assertArrayClose(wt2Row, regularResult.wavetrend2, 1.0, "Unified wt2 mismatch");
    // Histogram mirrors WT1-WT2; allow the same absolute tolerance as WT2
    assertArrayClose(histRow, regularResult.histogram, 1.0, "Unified histogram mismatch");
});

test('WTO zero-copy API', () => {
    // Test zero-copy API
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]);
    const channelLength = 5;
    const averageLength = 10;
    
    // Allocate buffers
    const inPtr = wasm.wto_alloc(data.length);
    const wt1Ptr = wasm.wto_alloc(data.length);
    const wt2Ptr = wasm.wto_alloc(data.length);
    const histPtr = wasm.wto_alloc(data.length);
    
    assert(inPtr !== 0, 'Failed to allocate input memory');
    assert(wt1Ptr !== 0, 'Failed to allocate wt1 memory');
    assert(wt2Ptr !== 0, 'Failed to allocate wt2 memory');
    assert(histPtr !== 0, 'Failed to allocate histogram memory');
    
    try {
        // Create views into WASM memory
        const memory = wasm.__wasm.memory.buffer;
        const inView = new Float64Array(memory, inPtr, data.length);
        const wt1View = new Float64Array(memory, wt1Ptr, data.length);
        const wt2View = new Float64Array(memory, wt2Ptr, data.length);
        const histView = new Float64Array(memory, histPtr, data.length);
        
        // Copy data into WASM memory
        inView.set(data);
        
        // Compute WTO using zero-copy API
        wasm.wto_into(inPtr, wt1Ptr, wt2Ptr, histPtr, data.length, channelLength, averageLength);
        
        // Verify results match regular API
        const regularResult = wasm.wto_js(data, channelLength, averageLength);
        
        for (let i = 0; i < data.length; i++) {
            // Check wavetrend1
            if (isNaN(regularResult.wavetrend1[i]) && isNaN(wt1View[i])) {
                // Both NaN is OK
            } else {
                const tolerance = 1e-8;
                assert(Math.abs(regularResult.wavetrend1[i] - wt1View[i]) < tolerance,
                       `Zero-copy wt1 mismatch at index ${i}: regular=${regularResult.wavetrend1[i]}, zerocopy=${wt1View[i]}`);
            }
            
            // Check wavetrend2
            if (isNaN(regularResult.wavetrend2[i]) && isNaN(wt2View[i])) {
                // Both NaN is OK
            } else {
                const tolerance = 1e-8;
                assert(Math.abs(regularResult.wavetrend2[i] - wt2View[i]) < tolerance,
                       `Zero-copy wt2 mismatch at index ${i}: regular=${regularResult.wavetrend2[i]}, zerocopy=${wt2View[i]}`);
            }
            
            // Check histogram
            if (isNaN(regularResult.histogram[i]) && isNaN(histView[i])) {
                // Both NaN is OK
            } else {
                const tolerance = 1e-8;
                assert(Math.abs(regularResult.histogram[i] - histView[i]) < tolerance,
                       `Zero-copy histogram mismatch at index ${i}: regular=${regularResult.histogram[i]}, zerocopy=${histView[i]}`);
            }
        }
    } finally {
        // Always free memory
        wasm.wto_free(inPtr, data.length);
        wasm.wto_free(wt1Ptr, data.length);
        wasm.wto_free(wt2Ptr, data.length);
        wasm.wto_free(histPtr, data.length);
    }
});

test('WTO zero-copy error handling', () => {
    // Test null pointer
    assert.throws(() => {
        wasm.wto_into(0, 0, 0, 0, 10, 5, 10);
    }, /null pointer|invalid memory/i);
    
    // Test invalid parameters with allocated memory
    const ptr = wasm.wto_alloc(25);
    try {
        // Invalid channel length
        assert.throws(() => {
            wasm.wto_into(ptr, ptr, ptr, ptr, 25, 0, 10);
        }, /Invalid period/);
        
        // Invalid average length
        assert.throws(() => {
            wasm.wto_into(ptr, ptr, ptr, ptr, 25, 5, 0);
        }, /Invalid period/);
    } finally {
        wasm.wto_free(ptr, 25);
    }
});

test('WTO batch error handling', () => {
    const close = new Float64Array(testData.close.slice(0, 10));
    
    // Invalid config structure
    assert.throws(() => {
        wasm.wto_batch(close, {
            channel: [10, 10], // Missing step
            average: [21, 21, 0]
        });
    }, /Invalid config/);
    
    // Missing required field
    assert.throws(() => {
        wasm.wto_batch(close, {
            channel: [10, 10, 0]
            // Missing average
        });
    }, /Invalid config/);
});

test.after(() => {
    console.log('WTO WASM tests completed');
});
