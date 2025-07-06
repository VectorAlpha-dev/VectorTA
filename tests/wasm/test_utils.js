/**
 * Common utilities for WASM binding tests
 */
import fs from 'fs';
import path from 'path';
import { parse } from 'csv-parse/sync';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

function loadTestData() {
    const csvPath = path.join(__dirname, '../../src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv');
    const content = fs.readFileSync(csvPath, 'utf8');
    const records = parse(content, { 
        columns: false,
        skip_empty_lines: true,
        cast: true,
        from_line: 2  // Skip header row
    });
    
    const candles = {
        open: [],
        high: [],
        low: [],
        close: [],
        volume: []
    };
    
    for (const row of records) {
        if (row.length < 6) continue;
        // CSV format matches Rust: timestamp[0], open[1], close[2], high[3], low[4], volume[5]
        candles.open.push(parseFloat(row[1]));
        candles.close.push(parseFloat(row[2]));
        candles.high.push(parseFloat(row[3]));
        candles.low.push(parseFloat(row[4]));
        candles.volume.push(parseFloat(row[5]));
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
    },
    cwma: {
        defaultParams: { period: 14 },
        last5Values: [
            59224.641237300435,
            59213.64831277214,
            59171.21190130624,
            59167.01279027576,
            59039.413552249636
        ]
    },
    dema: {
        defaultParams: { period: 30 },
        last5Values: [
            59189.73193987478,
            59129.24920772847,
            59058.80282420511,
            59011.5555611042,
            58908.370159946775
        ]
    },
    edcf: {
        defaultParams: { period: 15 },
        last5Values: [
            59593.332275678375,
            59731.70263288801,
            59766.41512339413,
            59655.66162110993,
            59332.492883847
        ]
    },
    ema: {
        defaultParams: { period: 9 },
        lastFive: [
            59302.2,
            59277.9,
            59230.2,
            59215.1,
            59103.1
        ]
    }
};

export { 
    loadTestData, 
    assertClose, 
    assertArrayClose, 
    isNaN,
    assertAllNaN,
    assertNoNaN,
    EXPECTED_OUTPUTS 
};