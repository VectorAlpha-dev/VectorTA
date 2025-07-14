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
        timestamp: [],
        open: [],
        high: [],
        low: [],
        close: [],
        volume: []
    };
    
    for (const row of records) {
        if (row.length < 6) continue;
        // CSV format matches Rust: timestamp[0], open[1], close[2], high[3], low[4], volume[5]
        candles.timestamp.push(parseFloat(row[0]));  // JS numbers are f64
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
    },
    sqwma: {
        defaultParams: { period: 14 },
        last5Values: [
            59229.72287968442,
            59211.30867850099,
            59172.516765286,
            59167.73471400394,
            59067.97928994083
        ]
    },
    srwma: {
        defaultParams: { period: 14 },
        last5Values: [
            59344.28384704595,
            59282.09151629659,
            59192.76580529367,
            59178.04767548977,
            59110.03801260874
        ]
    },
    supersmoother_3_pole: {
        defaultParams: { period: 14 },
        last5Values: [
            59072.13481064446,
            59089.08032603,
            59111.35711851466,
            59133.14402399381,
            59121.91820047289
        ]
    },
    supersmoother: {
        defaultParams: { period: 14 },
        last5Values: [
            59140.98229179739,
            59172.03593376982,
            59179.40342783722,
            59171.22758152845,
            59127.859841077094
        ]
    },
    wilders: {
        defaultParams: { period: 5 },
        last5Values: [
            59302.18156619092,
            59277.94525295273,
            59230.15620236219,
            59215.12496188975,
            59103.0999695118
        ]
    },
    ad: {
        defaultParams: {},
        last5Values: [
            1645918.16,
            1645876.11,
            1645824.27,
            1645828.87,
            1645728.78
        ]
    },
    adx: {
        defaultParams: { period: 14 },
        last5Values: [
            36.14,
            36.52,
            37.01,
            37.46,
            38.47
        ]
    },
    vwma: {
        defaultParams: { period: 20 },
        last5Values: [
            59201.87047121331,
            59217.157390630266,
            59195.74526905522,
            59196.261392450084,
            59151.22059588594
        ]
    },
    acosc: {
        defaultParams: {},  // ACOSC has no parameters
        last5Osc: [
            273.30,
            383.72,
            357.7,
            291.25,
            176.84
        ],
        last5Change: [
            49.6,
            110.4,
            -26.0,
            -66.5,
            -114.4
        ]
    },
    vwap: {
        defaultParams: { anchor: '1d' },
        last5Values: [
            59353.05963230107,
            59330.15815713043,
            59289.94649532547,
            59274.6155462414,
            58730.0
        ],
        anchor1D: [
            59353.05963230107,
            59330.15815713043,
            59289.94649532547,
            59274.6155462414,
            58730.0
        ]
    },
    zlema: {
        defaultParams: { period: 14 },
        last5Values: [
            59015.1,
            59165.2,
            59168.1,
            59147.0,
            58978.9
        ]
    },
    vpwma: {
        defaultParams: { period: 14, power: 0.382 },
        last5Values: [
            59363.927599446455,
            59296.83894519251,
            59196.82476139941,
            59180.8040249446,
            59113.84473799056
        ]
    },
    wma: {
        defaultParams: { period: 30 },
        last5Values: [
            59638.52903225806,
            59563.7376344086,
            59489.4064516129,
            59432.02580645162,
            59350.58279569892
        ]
    },
    adxr: {
        defaultParams: { period: 14 },
        last5Values: [
            37.10,
            37.3,
            37.0,
            36.2,
            36.3
        ]
    },
    adosc: {
        defaultParams: { short_period: 3, long_period: 10 },
        last5Values: [-166.2175, -148.9983, -144.9052, -128.5921, -142.0772]
    },
    ao: {
        defaultParams: { short_period: 5, long_period: 34 },
        last5Values: [-1671.3, -1401.6706, -1262.3559, -1178.4941, -1157.4118]
    },
    atr: {
        defaultParams: { length: 14 },
        last5Values: [916.89, 874.33, 838.45, 801.92, 811.57]
    }
};

// Convenience constants for individual indicators
const EXPECTED_SUPERSMOOTHER_3_POLE = EXPECTED_OUTPUTS.supersmoother_3_pole.last5Values;
const EXPECTED_SUPERSMOOTHER = EXPECTED_OUTPUTS.supersmoother.last5Values;

export { 
    loadTestData, 
    assertClose, 
    assertArrayClose, 
    isNaN,
    assertAllNaN,
    assertNoNaN,
    EXPECTED_OUTPUTS,
    EXPECTED_SUPERSMOOTHER_3_POLE,
    EXPECTED_SUPERSMOOTHER
};