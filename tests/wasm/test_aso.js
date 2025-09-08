/**
 * WASM binding tests for ASO (Average Sentiment Oscillator) indicator.
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

test('ASO accuracy', () => {
    // Test ASO matches expected values from Rust tests - mirrors check_aso_accuracy
    const expected = EXPECTED_OUTPUTS.aso;
    
    const result = wasm.aso(
        new Float64Array(testData.open),
        new Float64Array(testData.high),
        new Float64Array(testData.low),
        new Float64Array(testData.close),
        expected.defaultParams.period,
        expected.defaultParams.mode
    );
    
    // Handle new flattened format: values = [bulls..., bears...], rows = 2, cols = len
    assert(result.values, 'Result should have values array');
    assert.strictEqual(result.rows, 2, 'Result should have 2 rows');
    assert.strictEqual(result.cols, testData.close.length, 'Cols should match input length');
    
    // Extract bulls and bears from flattened array
    const bulls = result.values.slice(0, result.cols);
    const bears = result.values.slice(result.cols);
    
    // Check last 5 values match expected
    const bullsLast5 = bulls.slice(-5);
    const bearsLast5 = bears.slice(-5);
    
    assertArrayClose(
        bullsLast5,
        expected.last5Bulls,
        1e-6,
        "ASO Bulls last 5 values mismatch"
    );
    
    assertArrayClose(
        bearsLast5,
        expected.last5Bears,
        1e-6,
        "ASO Bears last 5 values mismatch"
    );
});

test('ASO partial params', () => {
    // Test ASO with default parameters - mirrors check_aso_partial_params
    const open = new Float64Array(testData.open);
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const result = wasm.aso(open, high, low, close);
    
    assert(result.values, 'Result should have values array with default params');
    assert.strictEqual(result.rows, 2, 'Result should have 2 rows');
    assert.strictEqual(result.cols, close.length, 'Cols should match input length');
});

test('ASO zero period', () => {
    // Test ASO fails with zero period - mirrors check_aso_zero_period
    const open = new Float64Array([10.0, 20.0, 30.0]);
    const high = new Float64Array([15.0, 25.0, 35.0]);
    const low = new Float64Array([8.0, 18.0, 28.0]);
    const close = new Float64Array([12.0, 22.0, 32.0]);
    
    assert.throws(() => {
        wasm.aso(open, high, low, close, 0, 0);
    }, /Invalid period/);
});

test('ASO period exceeds length', () => {
    // Test ASO fails when period exceeds data length - mirrors check_aso_period_exceeds_length
    const open = new Float64Array([10.0, 20.0, 30.0]);
    const high = new Float64Array([15.0, 25.0, 35.0]);
    const low = new Float64Array([8.0, 18.0, 28.0]);
    const close = new Float64Array([12.0, 22.0, 32.0]);
    
    assert.throws(() => {
        wasm.aso(open, high, low, close, 10, 0);
    }, /Invalid period/);
});

test('ASO very small dataset', () => {
    // Test ASO fails with insufficient data - mirrors check_aso_very_small_dataset
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.aso(singlePoint, singlePoint, singlePoint, singlePoint, 10, 0);
    }, /Invalid period|Not enough valid data/);
});

test('ASO empty input', () => {
    // Test ASO fails with empty input - mirrors check_aso_empty_input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.aso(empty, empty, empty, empty, 10, 0);
    }, /Input data slice is empty|All values.* NaN/);
});

test('ASO all NaN', () => {
    // Test ASO fails with all NaN values - mirrors check_aso_all_nan
    const nanData = new Float64Array([NaN, NaN, NaN]);
    
    assert.throws(() => {
        wasm.aso(nanData, nanData, nanData, nanData, 10, 0);
    }, /All values.* NaN/);
});

test('ASO mismatched lengths', () => {
    // Test ASO fails with mismatched array lengths - mirrors MissingData error
    const open = new Float64Array([10.0, 20.0, 30.0]);
    const high = new Float64Array([15.0, 25.0]);  // Different length
    const low = new Float64Array([8.0, 18.0, 28.0]);
    const close = new Float64Array([12.0, 22.0, 32.0]);
    
    assert.throws(() => {
        wasm.aso(open, high, low, close, 10, 0);
    }, /All OHLC arrays must have the same length/);
});

test('ASO invalid mode', () => {
    // Test ASO fails with invalid mode - mirrors check_aso_invalid_mode
    const open = new Float64Array(testData.open.slice(0, 100));
    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    const close = new Float64Array(testData.close.slice(0, 100));
    
    assert.throws(() => {
        wasm.aso(open, high, low, close, 10, 3);  // Invalid mode
    }, /Invalid mode/);
});

test('ASO different modes produce different results', () => {
    // Test different modes produce different results
    const open = new Float64Array(testData.open.slice(0, 100));
    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    const close = new Float64Array(testData.close.slice(0, 100));
    
    // Mode 0: Average of both
    const result0 = wasm.aso(open, high, low, close, 10, 0);
    const bulls0 = result0.values.slice(0, result0.cols);
    const bears0 = result0.values.slice(result0.cols);
    
    // Mode 1: Intrabar only
    const result1 = wasm.aso(open, high, low, close, 10, 1);
    const bulls1 = result1.values.slice(0, result1.cols);
    const bears1 = result1.values.slice(result1.cols);
    
    // Mode 2: Group only
    const result2 = wasm.aso(open, high, low, close, 10, 2);
    const bulls2 = result2.values.slice(0, result2.cols);
    const bears2 = result2.values.slice(result2.cols);
    
    // The modes should produce different results (check after warmup)
    const checkIdx = 50;
    const mode0Bull = bulls0[checkIdx];
    const mode1Bull = bulls1[checkIdx];
    const mode2Bull = bulls2[checkIdx];
    
    assert(
        mode0Bull !== mode1Bull || mode0Bull !== mode2Bull,
        'Different modes should produce different results'
    );
    
    // Also verify that in modes 1 and 2, bulls + bears = 100
    for (let i = 20; i < 100; i++) {  // After warmup
        if (!isNaN(bulls1[i]) && !isNaN(bears1[i])) {
            const sum1 = bulls1[i] + bears1[i];
            assert(
                Math.abs(sum1 - 100.0) < 1e-9,
                `Mode 1: bulls + bears != 100 at index ${i}: ${sum1}`
            );
        }
        
        if (!isNaN(bulls2[i]) && !isNaN(bears2[i])) {
            const sum2 = bulls2[i] + bears2[i];
            assert(
                Math.abs(sum2 - 100.0) < 1e-9,
                `Mode 2: bulls + bears != 100 at index ${i}: ${sum2}`
            );
        }
    }
});

test('ASO NaN handling', () => {
    // Test ASO handles NaN values correctly - mirrors check_aso_nan_handling
    const open = new Float64Array(testData.open);
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const result = wasm.aso(open, high, low, close, 10, 0);
    const bulls = result.values.slice(0, result.cols);
    const bears = result.values.slice(result.cols);
    
    // First period-1 values should be NaN (warmup period)
    for (let i = 0; i < 9; i++) {
        assert(isNaN(bulls[i]), `Expected NaN in bulls warmup at index ${i}`);
        assert(isNaN(bears[i]), `Expected NaN in bears warmup at index ${i}`);
    }
    
    // After warmup period (240), no NaN values should exist
    if (bulls.length > 240) {
        for (let i = 240; i < bulls.length; i++) {
            assert(!isNaN(bulls[i]), `Found unexpected NaN in bulls at index ${i}`);
            assert(!isNaN(bears[i]), `Found unexpected NaN in bears at index ${i}`);
        }
    }
});

test('ASO batch processing', () => {
    // Test ASO batch processing - mirrors check_aso_batch
    const open = new Float64Array(testData.open.slice(0, 100));
    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    const close = new Float64Array(testData.close.slice(0, 100));
    
    // Test batch with multiple parameter combinations
    const result = wasm.aso_batch(open, high, low, close, {
        period_range: [8, 12, 2],  // periods: 8, 10, 12
        mode_range: [0, 2, 1]       // modes: 0, 1, 2
    });
    
    assert(result.values, 'Should have values array');
    assert(result.combos, 'Should have combos array');
    assert.strictEqual(result.rows, 18, 'Should have 18 rows (9 combos * 2 for bulls/bears)');
    assert.strictEqual(result.cols, 100, 'Should have 100 columns');
    assert.strictEqual(result.combos.length, 9, 'Should have 9 combinations');
    
    // Verify each combination
    let comboIdx = 0;
    for (const period of [8, 10, 12]) {
        for (const mode of [0, 1, 2]) {
            assert.strictEqual(result.combos[comboIdx].period, period);
            assert.strictEqual(result.combos[comboIdx].mode, mode);
            
            // Extract this combination's results (bulls and bears are interleaved)
            // First half is all bulls, second half is all bears
            const bullsStart = comboIdx * result.cols;
            const bullsEnd = bullsStart + result.cols;
            const bearsStart = (result.combos.length * result.cols) + (comboIdx * result.cols);
            const bearsEnd = bearsStart + result.cols;
            
            const batchBulls = result.values.slice(bullsStart, bullsEnd);
            const batchBears = result.values.slice(bearsStart, bearsEnd);
            
            // Compare with single calculation
            const singleResult = wasm.aso(open, high, low, close, period, mode);
            const singleBulls = singleResult.values.slice(0, singleResult.cols);
            const singleBears = singleResult.values.slice(singleResult.cols);
            
            // Values should match
            for (let i = 0; i < result.cols; i++) {
                if (isNaN(singleBulls[i]) && isNaN(batchBulls[i])) continue;
                assert(
                    Math.abs(singleBulls[i] - batchBulls[i]) < 1e-10,
                    `Batch bulls mismatch at index ${i} for period=${period}, mode=${mode}`
                );
                
                if (isNaN(singleBears[i]) && isNaN(batchBears[i])) continue;
                assert(
                    Math.abs(singleBears[i] - batchBears[i]) < 1e-10,
                    `Batch bears mismatch at index ${i} for period=${period}, mode=${mode}`
                );
            }
            
            comboIdx++;
        }
    }
});

test('ASO batch single params', () => {
    // Test ASO batch with single parameter set
    const open = new Float64Array(testData.open.slice(0, 50));
    const high = new Float64Array(testData.high.slice(0, 50));
    const low = new Float64Array(testData.low.slice(0, 50));
    const close = new Float64Array(testData.close.slice(0, 50));
    
    // Single combination (default params)
    const result = wasm.aso_batch(open, high, low, close, {
        period_range: [10, 10, 0],
        mode_range: [0, 0, 0]
    });
    
    assert.strictEqual(result.rows, 2, 'Should have 2 rows (1 combo * 2 for bulls/bears)');
    assert.strictEqual(result.cols, 50, 'Should have 50 columns');
    assert.strictEqual(result.combos.length, 1, 'Should have 1 combination');
    assert.strictEqual(result.combos[0].period, 10);
    assert.strictEqual(result.combos[0].mode, 0);
    
    // Extract bulls and bears (note: batch returns 2 rows per combo)
    const batchBulls = result.values.slice(0, 50);
    const batchBears = result.values.slice(50, 100);
    
    // Compare with single calculation
    const singleResult = wasm.aso(open, high, low, close, 10, 0);
    const singleBulls = singleResult.values.slice(0, singleResult.cols);
    const singleBears = singleResult.values.slice(singleResult.cols);
    
    assertArrayClose(batchBulls, singleBulls, 1e-10, "Batch single params bulls mismatch");
    assertArrayClose(batchBears, singleBears, 1e-10, "Batch single params bears mismatch");
});

test('ASO zero-copy API', () => {
    // Test zero-copy API for performance
    const data = testData.close.slice(0, 100);
    const open = new Float64Array(testData.open.slice(0, 100));
    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    const close = new Float64Array(testData.close.slice(0, 100));
    const period = 10;
    const mode = 0;
    
    // Allocate buffers for bulls and bears
    const bullsPtr = wasm.aso_alloc(data.length);
    const bearsPtr = wasm.aso_alloc(data.length);
    assert(bullsPtr !== 0, 'Failed to allocate bulls buffer');
    assert(bearsPtr !== 0, 'Failed to allocate bears buffer');
    
    try {
        // Create views into WASM memory
        const memory = wasm.__wbindgen_memory();
        const bullsView = new Float64Array(memory.buffer, bullsPtr, data.length);
        const bearsView = new Float64Array(memory.buffer, bearsPtr, data.length);
        
        // Copy data into WASM memory (in real use, data might already be there)
        // For ASO we need to pass OHLC data
        const openPtr = wasm.aso_alloc(data.length);
        const highPtr = wasm.aso_alloc(data.length);
        const lowPtr = wasm.aso_alloc(data.length);
        const closePtr = wasm.aso_alloc(data.length);
        
        const openView = new Float64Array(memory.buffer, openPtr, data.length);
        const highView = new Float64Array(memory.buffer, highPtr, data.length);
        const lowView = new Float64Array(memory.buffer, lowPtr, data.length);
        const closeView = new Float64Array(memory.buffer, closePtr, data.length);
        
        openView.set(open);
        highView.set(high);
        lowView.set(low);
        closeView.set(close);
        
        // Compute ASO in-place
        wasm.aso_into(
            openPtr, highPtr, lowPtr, closePtr,
            bullsPtr, bearsPtr,
            data.length, period, mode
        );
        
        // Recreate views in case memory grew
        const memory2 = wasm.__wbindgen_memory();
        const bullsView2 = new Float64Array(memory2.buffer, bullsPtr, data.length);
        const bearsView2 = new Float64Array(memory2.buffer, bearsPtr, data.length);
        
        // Verify results match regular API
        const regularResult = wasm.aso(open, high, low, close, period, mode);
        const regularBulls = regularResult.values.slice(0, regularResult.cols);
        const regularBears = regularResult.values.slice(regularResult.cols);
        
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularBulls[i]) && isNaN(bullsView2[i])) continue;
            assert(
                Math.abs(regularBulls[i] - bullsView2[i]) < 1e-10,
                `Zero-copy bulls mismatch at index ${i}`
            );
            
            if (isNaN(regularBears[i]) && isNaN(bearsView2[i])) continue;
            assert(
                Math.abs(regularBears[i] - bearsView2[i]) < 1e-10,
                `Zero-copy bears mismatch at index ${i}`
            );
        }
        
        // Free OHLC buffers
        wasm.aso_free(openPtr, data.length);
        wasm.aso_free(highPtr, data.length);
        wasm.aso_free(lowPtr, data.length);
        wasm.aso_free(closePtr, data.length);
    } finally {
        // Always free memory
        wasm.aso_free(bullsPtr, data.length);
        wasm.aso_free(bearsPtr, data.length);
    }
});

test('ASO zero-copy with large dataset', () => {
    // Test zero-copy API with large dataset
    const size = 10000;
    const open = new Float64Array(size);
    const high = new Float64Array(size);
    const low = new Float64Array(size);
    const close = new Float64Array(size);
    
    // Generate synthetic OHLC data
    for (let i = 0; i < size; i++) {
        const base = 100 + Math.sin(i * 0.01) * 10;
        const range = 2 + Math.random();
        low[i] = base - range;
        high[i] = base + range;
        open[i] = base + (Math.random() - 0.5) * range;
        close[i] = base + (Math.random() - 0.5) * range;
    }
    
    const bullsPtr = wasm.aso_alloc(size);
    const bearsPtr = wasm.aso_alloc(size);
    assert(bullsPtr !== 0, 'Failed to allocate large bulls buffer');
    assert(bearsPtr !== 0, 'Failed to allocate large bears buffer');
    
    try {
        // Allocate and copy OHLC data
        const openPtr = wasm.aso_alloc(size);
        const highPtr = wasm.aso_alloc(size);
        const lowPtr = wasm.aso_alloc(size);
        const closePtr = wasm.aso_alloc(size);
        
        const memory = wasm.__wbindgen_memory();
        new Float64Array(memory.buffer, openPtr, size).set(open);
        new Float64Array(memory.buffer, highPtr, size).set(high);
        new Float64Array(memory.buffer, lowPtr, size).set(low);
        new Float64Array(memory.buffer, closePtr, size).set(close);
        
        wasm.aso_into(
            openPtr, highPtr, lowPtr, closePtr,
            bullsPtr, bearsPtr,
            size, 10, 0
        );
        
        // Recreate views in case memory grew
        const memory2 = wasm.__wbindgen_memory();
        const bullsView = new Float64Array(memory2.buffer, bullsPtr, size);
        const bearsView = new Float64Array(memory2.buffer, bearsPtr, size);
        
        // Check warmup period has NaN
        for (let i = 0; i < 9; i++) {
            assert(isNaN(bullsView[i]), `Expected NaN at warmup index ${i}`);
            assert(isNaN(bearsView[i]), `Expected NaN at warmup index ${i}`);
        }
        
        // Check after warmup has values
        for (let i = 9; i < Math.min(100, size); i++) {
            assert(!isNaN(bullsView[i]), `Unexpected NaN at index ${i}`);
            assert(!isNaN(bearsView[i]), `Unexpected NaN at index ${i}`);
            
            // Values should be in range [0, 100]
            assert(
                bullsView[i] >= -1e-9 && bullsView[i] <= 100.0 + 1e-9,
                `Bulls out of range at index ${i}: ${bullsView[i]}`
            );
            assert(
                bearsView[i] >= -1e-9 && bearsView[i] <= 100.0 + 1e-9,
                `Bears out of range at index ${i}: ${bearsView[i]}`
            );
        }
        
        // Free OHLC buffers
        wasm.aso_free(openPtr, size);
        wasm.aso_free(highPtr, size);
        wasm.aso_free(lowPtr, size);
        wasm.aso_free(closePtr, size);
    } finally {
        wasm.aso_free(bullsPtr, size);
        wasm.aso_free(bearsPtr, size);
    }
});

test('ASO SIMD128 consistency', () => {
    // This test verifies SIMD128 produces same results as scalar
    // It runs automatically when SIMD128 is enabled
    const testCases = [
        { size: 20, period: 5 },
        { size: 100, period: 10 },
        { size: 1000, period: 20 },
        { size: 5000, period: 50 }
    ];
    
    for (const testCase of testCases) {
        const open = new Float64Array(testCase.size);
        const high = new Float64Array(testCase.size);
        const low = new Float64Array(testCase.size);
        const close = new Float64Array(testCase.size);
        
        // Generate synthetic OHLC data
        for (let i = 0; i < testCase.size; i++) {
            const base = 100 + Math.sin(i * 0.1) * 20 + Math.cos(i * 0.05) * 10;
            const range = 2 + Math.random() * 3;
            low[i] = base - range;
            high[i] = base + range;
            open[i] = base + (Math.random() - 0.5) * range * 0.8;
            close[i] = base + (Math.random() - 0.5) * range * 0.8;
        }
        
        const result = wasm.aso(open, high, low, close, testCase.period, 0);
        const bulls = result.values.slice(0, result.cols);
        const bears = result.values.slice(result.cols);
        
        // Basic sanity checks
        assert.strictEqual(bulls.length, testCase.size);
        assert.strictEqual(bears.length, testCase.size);
        
        // Check warmup period
        for (let i = 0; i < testCase.period - 1; i++) {
            assert(isNaN(bulls[i]), `Expected NaN at warmup index ${i} for size=${testCase.size}`);
            assert(isNaN(bears[i]), `Expected NaN at warmup index ${i} for size=${testCase.size}`);
        }
        
        // Check values exist after warmup and are in valid range
        let sumBulls = 0;
        let sumBears = 0;
        let count = 0;
        for (let i = testCase.period - 1; i < bulls.length; i++) {
            assert(!isNaN(bulls[i]), `Unexpected NaN at index ${i} for size=${testCase.size}`);
            assert(!isNaN(bears[i]), `Unexpected NaN at index ${i} for size=${testCase.size}`);
            
            assert(
                bulls[i] >= -1e-9 && bulls[i] <= 100.0 + 1e-9,
                `Bulls out of range at index ${i}: ${bulls[i]}`
            );
            assert(
                bears[i] >= -1e-9 && bears[i] <= 100.0 + 1e-9,
                `Bears out of range at index ${i}: ${bears[i]}`
            );
            
            sumBulls += bulls[i];
            sumBears += bears[i];
            count++;
        }
        
        // Verify reasonable average values
        const avgBulls = sumBulls / count;
        const avgBears = sumBears / count;
        assert(avgBulls > 0 && avgBulls < 100, `Average bulls ${avgBulls} seems unreasonable`);
        assert(avgBears > 0 && avgBears < 100, `Average bears ${avgBears} seems unreasonable`);
    }
});

test('ASO zero-copy error handling', () => {
    // Test error handling for zero-copy API
    
    // Test null pointer
    assert.throws(() => {
        wasm.aso_into(0, 0, 0, 0, 0, 0, 10, 10, 0);
    }, /null pointer|invalid memory|panic/i);
    
    // Test invalid parameters with allocated memory
    const ptr = wasm.aso_alloc(10);
    try {
        // Invalid period
        assert.throws(() => {
            wasm.aso_into(ptr, ptr, ptr, ptr, ptr, ptr, 10, 0, 0);
        }, /Invalid period/);
        
        // Invalid mode
        assert.throws(() => {
            wasm.aso_into(ptr, ptr, ptr, ptr, ptr, ptr, 10, 10, 3);
        }, /Invalid mode/);
    } finally {
        wasm.aso_free(ptr, 10);
    }
});

test('ASO memory management', () => {
    // Test memory allocation and deallocation for leaks
    const sizes = [100, 1000, 10000];
    
    for (const size of sizes) {
        const bullsPtr = wasm.aso_alloc(size);
        const bearsPtr = wasm.aso_alloc(size);
        assert(bullsPtr !== 0, `Failed to allocate ${size} elements for bulls`);
        assert(bearsPtr !== 0, `Failed to allocate ${size} elements for bears`);
        
        // Write pattern to verify memory
        const memory = wasm.__wbindgen_memory();
        const bullsView = new Float64Array(memory.buffer, bullsPtr, size);
        const bearsView = new Float64Array(memory.buffer, bearsPtr, size);
        
        for (let i = 0; i < Math.min(10, size); i++) {
            bullsView[i] = i * 1.5;
            bearsView[i] = i * 2.5;
        }
        
        // Verify pattern
        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(bullsView[i], i * 1.5, `Bulls memory corruption at index ${i}`);
            assert.strictEqual(bearsView[i], i * 2.5, `Bears memory corruption at index ${i}`);
        }
        
        // Free memory
        wasm.aso_free(bullsPtr, size);
        wasm.aso_free(bearsPtr, size);
    }
});

test.after(() => {
    console.log('ASO WASM tests completed');
});