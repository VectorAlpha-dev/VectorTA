const test = require('ava');
const { loadTestData, assertClose } = require('./test_utils');

// Import the WASM module - adjust path as needed  
let wasm;
test.before(async () => {
    // Import using dynamic import for ESM compatibility
    wasm = await import('../../pkg/my_project.js');
});

test('Bollinger Bands - partial params', t => {
    const data = loadTestData();
    const close = data.close;
    
    // Test with period=22 (overriding default of 20)
    const result = wasm.bollinger_bands_js(close, 22, 2.0, 2.0, "sma", 0);
    
    // Result should have 3x the length (upper, middle, lower)
    t.is(result.length, close.length * 3);
});

test('Bollinger Bands - accuracy test', t => {
    const data = loadTestData();
    const close = data.close;
    
    // Use default parameters
    const result = wasm.bollinger_bands_js(close, 20, 2.0, 2.0, "sma", 0);
    
    // Extract bands (result is flattened as [upper..., middle..., lower...])
    const len = close.length;
    const upper = result.slice(0, len);
    const middle = result.slice(len, 2 * len);
    const lower = result.slice(2 * len, 3 * len);
    
    // Expected values from Rust tests
    const expectedMiddle = [
        59403.199999999975,
        59423.24999999998,
        59370.49999999998,
        59371.39999999998,
        59351.299999999974,
    ];
    const expectedLower = [
        58299.51497247008,
        58351.47038179873,
        58332.65135978715,
        58334.33194052157,
        58275.767369163135,
    ];
    const expectedUpper = [
        60506.88502752987,
        60495.029618201224,
        60408.348640212804,
        60408.468059478386,
        60426.83263083681,
    ];
    
    // Check last 5 values
    const startIdx = len - 5;
    for (let i = 0; i < 5; i++) {
        assertClose(t, upper[startIdx + i], expectedUpper[i], 1e-4, `Upper band mismatch at ${i}`);
        assertClose(t, middle[startIdx + i], expectedMiddle[i], 1e-4, `Middle band mismatch at ${i}`);
        assertClose(t, lower[startIdx + i], expectedLower[i], 1e-4, `Lower band mismatch at ${i}`);
    }
});

test('Bollinger Bands - zero period should fail', t => {
    const inputData = [10.0, 20.0, 30.0];
    
    const error = t.throws(() => {
        wasm.bollinger_bands_js(inputData, 0, 2.0, 2.0, "sma", 0);
    });
    
    t.true(error.message.includes('Invalid period'));
});

test('Bollinger Bands - period exceeds length should fail', t => {
    const smallData = [10.0, 20.0, 30.0];
    
    const error = t.throws(() => {
        wasm.bollinger_bands_js(smallData, 10, 2.0, 2.0, "sma", 0);
    });
    
    t.true(error.message.includes('Invalid period'));
});

test('Bollinger Bands - very small dataset should fail', t => {
    const singlePoint = [42.0];
    
    const error = t.throws(() => {
        wasm.bollinger_bands_js(singlePoint, 20, 2.0, 2.0, "sma", 0);
    });
    
    t.true(error.message.includes('Invalid period') || error.message.includes('Not enough valid data'));
});

test('Bollinger Bands - empty input should fail', t => {
    const empty = [];
    
    const error = t.throws(() => {
        wasm.bollinger_bands_js(empty, 20, 2.0, 2.0, "sma", 0);
    });
    
    t.true(error.message.includes('Empty data'));
});

test('Bollinger Bands - reinput test', t => {
    const data = loadTestData();
    const close = data.close;
    
    // First pass
    const result1 = wasm.bollinger_bands_js(close, 20, 2.0, 2.0, "sma", 0);
    const len = close.length;
    const middle1 = result1.slice(len, 2 * len);
    
    // Second pass - apply to middle band
    const result2 = wasm.bollinger_bands_js(middle1, 10, 2.0, 2.0, "sma", 0);
    
    t.is(result2.length, middle1.length * 3);
});

test('Bollinger Bands - NaN handling', t => {
    const data = loadTestData();
    const close = data.close;
    
    const result = wasm.bollinger_bands_js(close, 20, 2.0, 2.0, "sma", 0);
    const len = close.length;
    const upper = result.slice(0, len);
    const middle = result.slice(len, 2 * len);
    const lower = result.slice(2 * len, 3 * len);
    
    // After warmup period (240), no NaN values should exist
    if (len > 240) {
        for (let i = 240; i < len; i++) {
            t.false(isNaN(upper[i]), `NaN found in upper at ${i}`);
            t.false(isNaN(middle[i]), `NaN found in middle at ${i}`);
            t.false(isNaN(lower[i]), `NaN found in lower at ${i}`);
        }
    }
    
    // First period-1 values should be NaN
    for (let i = 0; i < 19; i++) {
        t.true(isNaN(upper[i]), `Expected NaN in upper warmup at ${i}`);
        t.true(isNaN(middle[i]), `Expected NaN in middle warmup at ${i}`);
        t.true(isNaN(lower[i]), `Expected NaN in lower warmup at ${i}`);
    }
});

test('Bollinger Bands - batch single params', t => {
    const data = loadTestData();
    const close = data.close;
    
    const result = wasm.bollinger_bands_batch_js(
        close,
        20, 20, 0,  // period range (single value)
        2.0, 2.0, 0.0,  // devup range (single value)
        2.0, 2.0, 0.0,  // devdn range (single value)
        "sma",
        0
    );
    
    // Result should have 1 row * cols * 3 bands
    const expectedLength = 1 * close.length * 3;
    t.is(result.length, expectedLength);
    
    // Extract middle band from batch result
    const len = close.length;
    const middle = result.slice(len, 2 * len);
    
    // Compare with single calculation
    const singleResult = wasm.bollinger_bands_js(close, 20, 2.0, 2.0, "sma", 0);
    const singleMiddle = singleResult.slice(len, 2 * len);
    
    // Should match
    for (let i = 0; i < len; i++) {
        if (!isNaN(middle[i]) && !isNaN(singleMiddle[i])) {
            assertClose(t, middle[i], singleMiddle[i], 1e-10, `Mismatch at ${i}`);
        }
    }
});

test('Bollinger Bands - batch multiple params', t => {
    const data = loadTestData();
    const close = data.close.slice(0, 200); // Use smaller dataset for speed
    
    const result = wasm.bollinger_bands_batch_js(
        close,
        10, 30, 10,  // periods: 10, 20, 30
        1.0, 3.0, 1.0,  // devups: 1.0, 2.0, 3.0
        2.0, 2.0, 0.0,  // devdns: 2.0
        "sma",
        0
    );
    
    // Should have 3 * 3 * 1 = 9 rows
    const expectedLength = 9 * close.length * 3;
    t.is(result.length, expectedLength);
});

test('Bollinger Bands - batch metadata', t => {
    const metadata = wasm.bollinger_bands_batch_metadata_js(
        10, 30, 10,  // periods: 10, 20, 30
        1.0, 3.0, 1.0,  // devups: 1.0, 2.0, 3.0
        2.0, 2.0, 0.0,  // devdns: 2.0
        "sma",
        0
    );
    
    // Should have 9 combinations * 4 values per combo
    t.is(metadata.length, 9 * 4);
    
    // Check first combination
    t.is(metadata[0], 10);  // period
    t.is(metadata[1], 1.0); // devup
    t.is(metadata[2], 2.0); // devdn
    t.is(metadata[3], 0);   // devtype
});

test('Bollinger Bands - ergonomic API', t => {
    const data = loadTestData();
    const close = data.close.slice(0, 100); // Smaller dataset
    
    const config = {
        period_range: [20, 20, 0],
        devup_range: [2.0, 2.0, 0.0],
        devdn_range: [2.0, 2.0, 0.0],
        matype: "sma",
        devtype: 0
    };
    
    const result = wasm.bollinger_bands_batch(close, config);
    
    t.truthy(result.upper);
    t.truthy(result.middle);
    t.truthy(result.lower);
    t.truthy(result.combos);
    t.is(result.rows, 1);
    t.is(result.cols, close.length);
    
    // Check combos structure
    t.is(result.combos.length, 1);
    t.is(result.combos[0].period, 20);
    t.is(result.combos[0].devup, 2.0);
    t.is(result.combos[0].devdn, 2.0);
    t.is(result.combos[0].matype, "sma");
    t.is(result.combos[0].devtype, 0);
});

test('Bollinger Bands - different MA types', t => {
    const data = loadTestData();
    const close = data.close.slice(0, 100);
    
    // Test with EMA
    const resultEma = wasm.bollinger_bands_js(close, 20, 2.0, 2.0, "ema", 0);
    const len = close.length;
    const middleEma = resultEma.slice(len, 2 * len);
    
    // Test with SMA
    const resultSma = wasm.bollinger_bands_js(close, 20, 2.0, 2.0, "sma", 0);
    const middleSma = resultSma.slice(len, 2 * len);
    
    // Results should be different after warmup
    let foundDifference = false;
    for (let i = 20; i < len; i++) {
        if (!isNaN(middleEma[i]) && !isNaN(middleSma[i])) {
            if (Math.abs(middleEma[i] - middleSma[i]) > 1e-8) {
                foundDifference = true;
                break;
            }
        }
    }
    t.true(foundDifference, "EMA and SMA results should differ");
});

test('Bollinger Bands - different deviation types', t => {
    const data = loadTestData();
    const close = data.close.slice(0, 100);
    
    // Test with standard deviation (0)
    const result0 = wasm.bollinger_bands_js(close, 20, 2.0, 2.0, "sma", 0);
    const len = close.length;
    const upper0 = result0.slice(0, len);
    
    // Test with mean absolute deviation (1)
    const result1 = wasm.bollinger_bands_js(close, 20, 2.0, 2.0, "sma", 1);
    const upper1 = result1.slice(0, len);
    
    // Band widths should be different
    let foundDifference = false;
    for (let i = 20; i < len; i++) {
        if (!isNaN(upper0[i]) && !isNaN(upper1[i])) {
            if (Math.abs(upper0[i] - upper1[i]) > 1e-8) {
                foundDifference = true;
                break;
            }
        }
    }
    t.true(foundDifference, "Different deviation types should produce different results");
});

test('Bollinger Bands - asymmetric bands', t => {
    const data = loadTestData();
    const close = data.close.slice(0, 100);
    
    // Test with different devup and devdn
    const result = wasm.bollinger_bands_js(close, 20, 3.0, 1.0, "sma", 0);
    const len = close.length;
    const upper = result.slice(0, len);
    const middle = result.slice(len, 2 * len);
    const lower = result.slice(2 * len, 3 * len);
    
    // Check that bands are asymmetric after warmup
    for (let i = 20; i < len && i < 30; i++) {
        if (!isNaN(upper[i]) && !isNaN(middle[i]) && !isNaN(lower[i])) {
            const upperDist = upper[i] - middle[i];
            const lowerDist = middle[i] - lower[i];
            // Upper distance should be ~3x lower distance
            const ratio = upperDist / lowerDist;
            t.true(Math.abs(ratio - 3.0) < 0.1, `Asymmetric ratio incorrect at ${i}: ${ratio}`);
        }
    }
});