const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');
const csv = require('csv-parse/sync');
const wasm = require('../../pkg/rust_backtester');

// Helper function to read CSV data
function readCSV(filePath) {
    const fileContent = fs.readFileSync(filePath, 'utf-8');
    return csv.parse(fileContent, {
        columns: true,
        skip_empty_lines: true,
        cast: true
    });
}

test('beardy_squeeze_pro basic functionality', () => {
    // Load test data
    const dataPath = path.join(__dirname, '..', '..', 'src', 'data', '2018-09-01-2024-Bitfinex_Spot-4h.csv');
    const records = readCSV(dataPath);
    
    const high = records.map(r => parseFloat(r.high));
    const low = records.map(r => parseFloat(r.low));
    const close = records.map(r => parseFloat(r.close));
    
    // Test with default parameters
    const result = wasm.beardy_squeeze_pro(
        new Float64Array(high),
        new Float64Array(low),
        new Float64Array(close),
        20,  // length
        2.0,  // bb_mult
        1.0,  // kc_mult_high
        1.5,  // kc_mult_mid
        2.0   // kc_mult_low
    );
    
    assert.ok(result, 'Should return a result');
    assert.ok(result.momentum, 'Should have momentum array');
    assert.ok(result.squeeze, 'Should have squeeze array');
    assert.equal(result.momentum.length, close.length, 'Momentum length should match input');
    assert.equal(result.squeeze.length, close.length, 'Squeeze length should match input');
    
    // Test accuracy with reference values
    const expectedMom = [
        -170.88428571,
        -155.36642857,
        -65.28107143,
        -61.14321429,
        -178.12464286,
    ];
    
    const expectedSqz = [0.0, 0.0, 0.0, 0.0, 0.0];
    
    // The warmup period for length=20
    const startIdx = 19; // Updated to match Rust implementation (length - 1)
    
    // Check momentum values
    for (let i = 0; i < expectedMom.length; i++) {
        const actual = result.momentum[startIdx + i];
        const expected = expectedMom[i];
        const diff = Math.abs(actual - expected);
        assert.ok(diff < 0.01, `Momentum mismatch at index ${i}: expected ${expected}, got ${actual}`);
    }
    
    // Check squeeze values
    for (let i = 0; i < expectedSqz.length; i++) {
        const actual = result.squeeze[startIdx + i];
        const expected = expectedSqz[i];
        assert.equal(actual, expected, `Squeeze mismatch at index ${i}: expected ${expected}, got ${actual}`);
    }
});

test('beardy_squeeze_pro with custom parameters', () => {
    // Generate test data
    const n = 100;
    const high = new Float64Array(n);
    const low = new Float64Array(n);
    const close = new Float64Array(n);
    
    // Simple synthetic data
    for (let i = 0; i < n; i++) {
        high[i] = 100 + Math.sin(i * 0.1) * 10 + Math.random() * 2;
        low[i] = high[i] - 2 - Math.random();
        close[i] = (high[i] + low[i]) / 2 + (Math.random() - 0.5) * 0.5;
    }
    
    // Test with custom parameters
    const result = wasm.beardy_squeeze_pro(
        high,
        low,
        close,
        30,   // length
        2.5,  // bb_mult
        1.2,  // kc_mult_high
        1.8,  // kc_mult_mid
        2.5   // kc_mult_low
    );
    
    assert.ok(result, 'Should return a result');
    assert.equal(result.momentum.length, n, 'Momentum length should match input');
    assert.equal(result.squeeze.length, n, 'Squeeze length should match input');
    
    // Check that warmup period has NaN values
    assert.ok(isNaN(result.momentum[0]), 'First momentum value should be NaN');
    assert.ok(isNaN(result.squeeze[0]), 'First squeeze value should be NaN');
    
    // Check that we have valid values after warmup
    assert.ok(!isNaN(result.momentum[n - 1]), 'Last momentum value should be valid');
    assert.ok(!isNaN(result.squeeze[n - 1]), 'Last squeeze value should be valid');
    
    // Check squeeze values are in valid range (0-3)
    for (let i = 30; i < n; i++) {
        if (!isNaN(result.squeeze[i])) {
            assert.ok(result.squeeze[i] >= 0 && result.squeeze[i] <= 3,
                `Squeeze value at ${i} should be between 0 and 3, got ${result.squeeze[i]}`);
        }
    }
});

test('beardy_squeeze_pro edge cases', () => {
    // Test with minimal data
    const minHigh = new Float64Array(21).fill(1.0);
    const minLow = new Float64Array(21).fill(0.9);
    const minClose = new Float64Array(21).fill(0.95);
    
    const result1 = wasm.beardy_squeeze_pro(
        minHigh,
        minLow,
        minClose,
        20,
        2.0,
        1.0,
        1.5,
        2.0
    );
    
    assert.ok(result1, 'Should handle minimal data');
    assert.equal(result1.momentum.length, 21, 'Should return correct length');
    
    // Test with NaN values
    const nanHigh = new Float64Array(60);
    const nanLow = new Float64Array(60);
    const nanClose = new Float64Array(60);
    
    // First 10 values are NaN, rest are valid
    for (let i = 0; i < 60; i++) {
        if (i < 10) {
            nanHigh[i] = NaN;
            nanLow[i] = NaN;
            nanClose[i] = NaN;
        } else {
            nanHigh[i] = 1.0;
            nanLow[i] = 0.9;
            nanClose[i] = 0.95;
        }
    }
    
    const result2 = wasm.beardy_squeeze_pro(
        nanHigh,
        nanLow,
        nanClose,
        20,
        2.0,
        1.0,
        1.5,
        2.0
    );
    
    assert.ok(result2, 'Should handle NaN values');
    assert.equal(result2.momentum.length, 60, 'Should return correct length');
    assert.ok(isNaN(result2.momentum[10]), 'Should have NaN in early periods');
    assert.ok(!isNaN(result2.momentum[59]), 'Should have valid values later');
});

test('beardy_squeeze_pro validation', () => {
    // Test error cases
    const empty = new Float64Array(0);
    const valid = new Float64Array([1, 2, 3, 4, 5]);
    
    // Test empty data
    assert.throws(() => {
        wasm.beardy_squeeze_pro(empty, empty, empty, 20, 2.0, 1.0, 1.5, 2.0);
    }, 'Should throw on empty data');
    
    // Test mismatched lengths
    assert.throws(() => {
        wasm.beardy_squeeze_pro(valid, empty, valid, 20, 2.0, 1.0, 1.5, 2.0);
    }, 'Should throw on mismatched lengths');
    
    // Test insufficient data
    assert.throws(() => {
        wasm.beardy_squeeze_pro(valid, valid, valid, 20, 2.0, 1.0, 1.5, 2.0);
    }, 'Should throw on insufficient data');
});