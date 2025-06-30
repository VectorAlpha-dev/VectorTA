/**
 * Common utilities for WASM binding tests
 */
const fs = require('fs');
const path = require('path');
const { parse } = require('csv-parse/sync');

function loadTestData() {
    const csvPath = path.join(__dirname, '../../src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv');
    const content = fs.readFileSync(csvPath, 'utf8');
    const records = parse(content, { 
        columns: true,
        skip_empty_lines: true,
        cast: true
    });
    
    const candles = {
        open: [],
        high: [],
        low: [],
        close: [],
        volume: []
    };
    
    for (const row of records) {
        candles.open.push(parseFloat(row.open));
        candles.high.push(parseFloat(row.high));
        candles.low.push(parseFloat(row.low));
        candles.close.push(parseFloat(row.close));
        candles.volume.push(parseFloat(row.volume || 0));
    }
    
    return candles;
}

function assertClose(actual, expected, tolerance = 1e-8, msg = "") {
    const diff = Math.abs(actual - expected);
    if (diff > tolerance) {
        const errorMsg = msg ? `${msg}: ` : "";
        throw new Error(`${errorMsg}Expected ${expected}, got ${actual} (diff: ${diff})`);
    }
}

function assertArrayClose(actual, expected, tolerance = 1e-8, msg = "") {
    if (actual.length !== expected.length) {
        throw new Error(`${msg}: Length mismatch: ${actual.length} vs ${expected.length}`);
    }
    for (let i = 0; i < actual.length; i++) {
        const diff = Math.abs(actual[i] - expected[i]);
        if (diff > tolerance) {
            const errorMsg = msg ? `${msg}: ` : "";
            throw new Error(`${errorMsg}Mismatch at index ${i}: expected ${expected[i]}, got ${actual[i]} (diff: ${diff})`);
        }
    }
}

function isNaN(value) {
    return value !== value;
}

function assertAllNaN(array, msg = "") {
    for (let i = 0; i < array.length; i++) {
        if (!isNaN(array[i])) {
            throw new Error(`${msg}: Expected NaN at index ${i}, got ${array[i]}`);
        }
    }
}

function assertNoNaN(array, msg = "") {
    for (let i = 0; i < array.length; i++) {
        if (isNaN(array[i])) {
            throw new Error(`${msg}: Unexpected NaN at index ${i}`);
        }
    }
}

// Expected outputs from Rust tests - these must match EXACTLY
const EXPECTED_OUTPUTS = {
    alma: {
        defaultParams: { period: 9, offset: 0.85, sigma: 6.0 },
        last5Values: [
            59286.72216704,
            59273.53428138,
            59204.37290721,
            59155.93381742,
            59026.92526112
        ],
        // Re-input test expected values
        reinputLast5: [
            59140.73195170,
            59211.58090986,
            59238.16030697,
            59222.63528822,
            59165.14427332
        ]
    }
};

module.exports = { 
    loadTestData, 
    assertClose, 
    assertArrayClose, 
    isNaN,
    assertAllNaN,
    assertNoNaN,
    EXPECTED_OUTPUTS 
};