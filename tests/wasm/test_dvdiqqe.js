import test from 'node:test';
import assert from 'node:assert';
import * as wasm from '../../pkg/vector_ta.js';
import { loadTestData, assertArrayClose, assertClose, isNaN, assertAllNaN, assertNoNaN, EXPECTED_OUTPUTS } from './test_utils.js';

test('DVDIQQE with default parameters', () => {
    
    const testData = loadTestData();
    const expected = EXPECTED_OUTPUTS.dvdiqqe;
    
    
    const open = new Float64Array(testData.open);
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    
    const result = wasm.dvdiqqe(
        open,
        high,
        low,
        close,
        Array.from(volume),  
        expected.defaultParams.period,
        expected.defaultParams.smoothingPeriod,
        expected.defaultParams.fastMultiplier,
        expected.defaultParams.slowMultiplier,
        expected.defaultParams.volumeType,
        expected.defaultParams.centerType,
        expected.defaultParams.tickSize
    );
    
    
    assert.ok(result.values, 'Result should have values array');
    assert.strictEqual(result.rows, 4, 'Result should have 4 rows');
    assert.strictEqual(result.cols, close.length, 'Columns should match input length');
    assert.strictEqual(result.values.length, 4 * close.length, 'Values should be 4×input length');
    
    
    const dvdi = result.values.slice(0, result.cols);
    const fast_tl = result.values.slice(result.cols, 2 * result.cols);
    const slow_tl = result.values.slice(2 * result.cols, 3 * result.cols);
    const center_line = result.values.slice(3 * result.cols, 4 * result.cols);
    
    
    const warmup = 25; 
    
    
    for (let i = 0; i < warmup; i++) {
        assert(isNaN(dvdi[i]), `Expected NaN at warmup index ${i} for DVDI`);
        assert(isNaN(fast_tl[i]), `Expected NaN at warmup index ${i} for Fast TL`);
        assert(isNaN(slow_tl[i]), `Expected NaN at warmup index ${i} for Slow TL`);
    }
    
    
    for (let i = warmup; i < Math.min(warmup + 50, result.cols); i++) {
        assert(!isNaN(dvdi[i]), `Unexpected NaN at index ${i} for DVDI`);
        assert(!isNaN(fast_tl[i]), `Unexpected NaN at index ${i} for Fast TL`);
        assert(!isNaN(slow_tl[i]), `Unexpected NaN at index ${i} for Slow TL`);
        assert(!isNaN(center_line[i]), `Unexpected NaN at index ${i} for Center Line`);
    }
});

test('DVDIQQE with custom parameters', () => {
    
    const testData = loadTestData();
    const n_samples = 50;
    
    const open = new Float64Array(testData.open.slice(0, n_samples));
    const high = new Float64Array(testData.high.slice(0, n_samples));
    const low = new Float64Array(testData.low.slice(0, n_samples));
    const close = new Float64Array(testData.close.slice(0, n_samples));
    const volume = new Float64Array(testData.volume.slice(0, n_samples));
    
    
    const result = wasm.dvdiqqe(
        open,
        high,
        low,
        close,
        Array.from(volume),
        10,     
        5,      
        2.0,    
        4.0,    
        "tick", 
        "static" 
    );
    
    
    assert.ok(result.values, 'Result should have values array');
    assert.strictEqual(result.rows, 4, 'Result should have 4 rows');
    assert.strictEqual(result.cols, n_samples, 'Columns should match input length');
    assert.strictEqual(result.values.length, 4 * n_samples, 'Values should be 4×input length');
    
    
    const center_line = result.values.slice(3 * result.cols, 4 * result.cols);
    const validCenterValues = center_line.filter(v => !isNaN(v));
    const uniqueCenter = [...new Set(validCenterValues)];
    assert.strictEqual(uniqueCenter.length, 1, 'Static center line should be constant');
});

test('DVDIQQE without volume', () => {
    
    const testData = loadTestData();
    const n_samples = 50;
    
    const open = new Float64Array(testData.open.slice(0, n_samples));
    const high = new Float64Array(testData.high.slice(0, n_samples));
    const low = new Float64Array(testData.low.slice(0, n_samples));
    const close = new Float64Array(testData.close.slice(0, n_samples));
    
    
    const result = wasm.dvdiqqe(
        open, 
        high, 
        low, 
        close,
        undefined,  
        13,         
        6,          
        3.0,        
        5.0,        
        "tick",     
        "dynamic",  
        0.0001      
    );
    
    
    assert.ok(result.values, 'Result should have values array');
    assert.strictEqual(result.rows, 4, 'Result should have 4 rows');
    assert.strictEqual(result.cols, n_samples, 'Columns should match input length');
    assert.strictEqual(result.values.length, 4 * n_samples, 'Values should be 4×input length');
});

test('DVDIQQE accuracy check', () => {
    
    const expected = EXPECTED_OUTPUTS.dvdiqqe;
    
    
    const testData = loadTestData();
    const n_samples = 100;
    
    const open = new Float64Array(testData.open.slice(0, n_samples));
    const high = new Float64Array(testData.high.slice(0, n_samples));
    const low = new Float64Array(testData.low.slice(0, n_samples));
    const close = new Float64Array(testData.close.slice(0, n_samples));
    const volume = new Float64Array(testData.volume.slice(0, n_samples));
    
    
    const result = wasm.dvdiqqe(
        open,
        high,
        low,
        close,
        Array.from(volume),
        expected.defaultParams.period,
        expected.defaultParams.smoothingPeriod,
        expected.defaultParams.fastMultiplier,
        expected.defaultParams.slowMultiplier,
        expected.defaultParams.volumeType,
        expected.defaultParams.centerType,
        expected.defaultParams.tickSize
    );
    
    
    assert.ok(result.values);
    assert.strictEqual(result.values.length, 4 * n_samples);
    
    
    const dvdi = result.values.slice(0, result.cols);
    const fast_tl = result.values.slice(result.cols, 2 * result.cols);
    const slow_tl = result.values.slice(2 * result.cols, 3 * result.cols);
    const center_line = result.values.slice(3 * result.cols, 4 * result.cols);
    
    
    const warmup = 25; 
    for (let i = warmup; i < n_samples; i++) {
        assert.ok(isFinite(dvdi[i]), `DVDI at ${i} should be finite`);
        assert.ok(isFinite(fast_tl[i]), `Fast TL at ${i} should be finite`);
        assert.ok(isFinite(slow_tl[i]), `Slow TL at ${i} should be finite`);
        assert.ok(isFinite(center_line[i]), `Center line at ${i} should be finite`);
    }
    
    
    const centerValues = center_line.slice(warmup);
    const centerSet = new Set(centerValues);
    assert.ok(centerSet.size > 1, 'Dynamic center line should vary over time');
});

test('DVDIQQE matches Rust reference values (last 5)', () => {
    
    const expectedDvdi = [-304.41010224, -279.48152664, -287.58723437, -252.40349484, -343.00922595];
    const expectedSlow = [-990.21769695, -955.69385266, -951.82562405, -903.39071943, -903.39071943];
    const expectedFast = [-728.26380454, -697.40500858, -697.40500858, -654.73695895, -654.73695895];
    
    const expectedCenter = [
        21.98929919135097,
        21.969910753134442,
        21.950003541229705,
        21.932361363982043,
        21.908895469736102,
    ];

    const testData = loadTestData();
    const open = new Float64Array(testData.open);
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);

    
    const result = wasm.dvdiqqe(
        open,
        high,
        low,
        close,
        Array.from(volume),
        13,     
        6,      
        2.618,  
        4.236,  
        'default',
        'dynamic'
    );

    const cols = result.cols;
    const dvdi = result.values.slice(0, cols);
    const fast_tl = result.values.slice(cols, 2 * cols);
    const slow_tl = result.values.slice(2 * cols, 3 * cols);
    const center_line = result.values.slice(3 * cols, 4 * cols);

    
    const last5 = (arr) => arr.slice(arr.length - 5);
    assertArrayClose(last5(dvdi), expectedDvdi, 1e-6, 'DVDI last-5 mismatch vs Rust');
    assertArrayClose(last5(slow_tl), expectedSlow, 1e-6, 'Slow TL last-5 mismatch vs Rust');
    assertArrayClose(last5(fast_tl), expectedFast, 1e-6, 'Fast TL last-5 mismatch vs Rust');
    assertArrayClose(last5(center_line), expectedCenter, 1e-6, 'Center line last-5 mismatch vs Rust');
});

test('DVDIQQE error handling - empty input', () => {
    const empty = new Float64Array(0);
    
    assert.throws(() => {
        wasm.dvdiqqe(empty, empty, empty, empty);
    }, /Input data slice is empty|Empty input/, 'Should throw on empty input');
});

test('DVDIQQE error handling - all NaN', () => {
    const n_samples = 10;
    const nanArray = new Float64Array(n_samples).fill(NaN);
    
    assert.throws(() => {
        wasm.dvdiqqe(nanArray, nanArray, nanArray, nanArray);
    }, /All values are NaN|Invalid data/, 'Should throw on all NaN values');
});

test('DVDIQQE error handling - mismatched lengths', () => {
    const open = Float64Array.from([100.0, 101.0]);
    const high = Float64Array.from([102.0, 103.0, 104.0]);
    const low = Float64Array.from([99.0]);
    const close = Float64Array.from([101.0, 102.0]);
    
    assert.throws(() => {
        wasm.dvdiqqe(open, high, low, close);
    }, /Input arrays must have the same length|Mismatched lengths/, 'Should throw on mismatched input lengths');
});

test('DVDIQQE error handling - period too large', () => {
    const n_samples = 5;
    const data = new Float64Array(n_samples).fill(100);
    
    assert.throws(() => {
        wasm.dvdiqqe(
            data,
            data,
            data,
            data,
            undefined,
            20  
        );
    }, /Not enough data|Period too large|Invalid period/, 'Should throw when period exceeds data length');
});

test('DVDIQQE NaN handling', () => {
    
    const testData = loadTestData();
    const expected = EXPECTED_OUTPUTS.dvdiqqe;
    const n_samples = 100;
    
    const open = new Float64Array(testData.open.slice(0, n_samples));
    const high = new Float64Array(testData.high.slice(0, n_samples));
    const low = new Float64Array(testData.low.slice(0, n_samples));
    const close = new Float64Array(testData.close.slice(0, n_samples));
    const volume = new Float64Array(testData.volume.slice(0, n_samples));
    
    const result = wasm.dvdiqqe(
        open, high, low, close, Array.from(volume),
        expected.defaultParams.period,
        expected.defaultParams.smoothingPeriod
    );
    
    
    const dvdi = result.values.slice(0, result.cols);
    const fast_tl = result.values.slice(result.cols, 2 * result.cols);
    const slow_tl = result.values.slice(2 * result.cols, 3 * result.cols);
    const center_line = result.values.slice(3 * result.cols, 4 * result.cols);
    
    
    const warmup = 25;
    for (let i = 0; i < warmup; i++) {
        assert(isNaN(dvdi[i]), `Expected NaN at warmup index ${i}`);
    }
    
    
    for (let i = warmup; i < n_samples; i++) {
        assert(!isNaN(dvdi[i]), `Unexpected NaN at index ${i} in DVDI`);
        assert(!isNaN(fast_tl[i]), `Unexpected NaN at index ${i} in Fast TL`);
        assert(!isNaN(slow_tl[i]), `Unexpected NaN at index ${i} in Slow TL`);
        assert(!isNaN(center_line[i]), `Unexpected NaN at index ${i} in Center Line`);
    }
});

test('DVDIQQE invalid parameters', () => {
    const testData = loadTestData();
    const n_samples = 30;
    const open = new Float64Array(testData.open.slice(0, n_samples));
    const high = new Float64Array(testData.high.slice(0, n_samples));
    const low = new Float64Array(testData.low.slice(0, n_samples));
    const close = new Float64Array(testData.close.slice(0, n_samples));
    
    
    assert.throws(() => {
        wasm.dvdiqqe(open, high, low, close, undefined, 0);
    }, /Invalid period|Period must be positive/, 'Should throw on zero period');
    
    
    assert.throws(() => {
        wasm.dvdiqqe(
            open, high, low, close, undefined,
            13, 6, -1.0  
        );
    }, /Invalid multiplier|Multiplier must be positive/, 'Should throw on negative multiplier');
});

test('DVDIQQE center types', () => {
    const testData = loadTestData();
    const n_samples = 50;
    const open = new Float64Array(testData.open.slice(0, n_samples));
    const high = new Float64Array(testData.high.slice(0, n_samples));
    const low = new Float64Array(testData.low.slice(0, n_samples));
    const close = new Float64Array(testData.close.slice(0, n_samples));
    const volume = new Float64Array(testData.volume.slice(0, n_samples));
    
    
    const staticResult = wasm.dvdiqqe(
        open, high, low, close, Array.from(volume),
        13, 6, 3.0, 5.0, "real", "static"
    );
    
    const staticCenter = staticResult.values.slice(3 * staticResult.cols, 4 * staticResult.cols);
    const validStaticCenter = staticCenter.filter(v => !isNaN(v));
    const uniqueStatic = [...new Set(validStaticCenter)];
    assert.strictEqual(uniqueStatic.length, 1, 'Static center should be constant');
    
    
    const dynamicResult = wasm.dvdiqqe(
        open, high, low, close, Array.from(volume),
        13, 6, 3.0, 5.0, "real", "dynamic"
    );
    
    const dynamicCenter = dynamicResult.values.slice(3 * dynamicResult.cols, 4 * dynamicResult.cols);
    const warmup = 25;
    const validDynamicCenter = dynamicCenter.slice(warmup);
    const uniqueDynamic = [...new Set(validDynamicCenter)];
    assert.ok(uniqueDynamic.length > 1, 'Dynamic center should vary over time');
});

test('DVDIQQE volume types', () => {
    const testData = loadTestData();
    const n_samples = 50;
    const open = new Float64Array(testData.open.slice(0, n_samples));
    const high = new Float64Array(testData.high.slice(0, n_samples));
    const low = new Float64Array(testData.low.slice(0, n_samples));
    const close = new Float64Array(testData.close.slice(0, n_samples));
    const volume = new Float64Array(testData.volume.slice(0, n_samples));
    
    
    const realResult = wasm.dvdiqqe(
        open, high, low, close, Array.from(volume),
        13, 6, 3.0, 5.0, "real", "dynamic"
    );
    
    
    const tickResult = wasm.dvdiqqe(
        open, high, low, close, Array.from(volume),
        13, 6, 3.0, 5.0, "tick", "dynamic"
    );
    
    
    const noVolumeResult = wasm.dvdiqqe(
        open, high, low, close, undefined,
        13, 6, 3.0, 5.0, "real", "dynamic"
    );
    
    
    const warmup = 25; 
    const realDvdi = realResult.values.slice(0, realResult.cols);
    const tickDvdi = tickResult.values.slice(0, tickResult.cols);
    const noVolDvdi = noVolumeResult.values.slice(0, noVolumeResult.cols);
    
    
    for (let i = warmup; i < n_samples; i++) {
        assert(!isNaN(realDvdi[i]), `Real volume DVDI has NaN at ${i}`);
        assert(!isNaN(tickDvdi[i]), `Tick volume DVDI has NaN at ${i}`);
        assert(!isNaN(noVolDvdi[i]), `No volume DVDI has NaN at ${i}`);
    }
    
    
    let tickMatchesNoVol = true;
    for (let i = warmup; i < n_samples; i++) {
        if (Math.abs(tickDvdi[i] - noVolDvdi[i]) > 1e-10) {
            tickMatchesNoVol = false;
            break;
        }
    }
    assert.ok(tickMatchesNoVol, 'Tick volume should match no-volume behavior');
    
    
    
    
});

test('DVDIQQE PineScript reference validation', () => {
    
    const expected = EXPECTED_OUTPUTS.dvdiqqe;
    
    
    assert.ok(expected.pinescriptDvdi, 'Should have PineScript DVDI reference values');
    assert.ok(expected.pinescriptSlowTl, 'Should have PineScript Slow TL reference values');
    assert.ok(expected.pinescriptFastTl, 'Should have PineScript Fast TL reference values');
    assert.ok(expected.pinescriptCenter, 'Should have PineScript Center reference values');
    
    
    
});


