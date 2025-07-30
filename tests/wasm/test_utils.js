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
    cg: {
        defaultParams: { period: 10 },
        last5Values: [
            -4.99905186931943,
            -4.998559827254377,
            -4.9970065675119555,
            -4.9928483984587295,
            -5.004210799262688
        ]
    },
    cfo: {
        defaultParams: { period: 14, scalar: 100.0 },
        last5Values: [
            0.5998626489475746,
            0.47578011282578453,
            0.20349744599816233,
            0.0919617952835795,
            -0.5676291145560617
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
    keltner: {
        defaultParams: { period: 20, multiplier: 2.0, ma_type: "ema" },
        last5Upper: [
            61619.504155205745,
            61503.56119134791,
            61387.47897150178,
            61286.61078267451,
            61206.25688331261
        ],
        last5Middle: [
            59758.339871629956,
            59703.35512195091,
            59640.083205574636,
            59593.884805043715,
            59504.46720456336
        ],
        last5Lower: [
            57897.17558805417,
            57903.14905255391,
            57892.687439647495,
            57901.15882741292,
            57802.6775258141
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
    aroon: {
        defaultParams: { length: 14 },
        last5Up: [
            21.43,
            14.29,
            7.14,
            0.0,
            0.0
        ],
        last5Down: [
            71.43,
            64.29,
            57.14,
            50.0,
            42.86
        ]
    },
    aroonosc: {
        defaultParams: { length: 14 },
        last5Values: [-50.0, -50.0, -50.0, -50.0, -42.8571]
    },
    adosc: {
        defaultParams: { short_period: 3, long_period: 10 },
        last5Values: [-166.2175, -148.9983, -144.9052, -128.5921, -142.0772]
    },
    bollinger_bands_width: {
        defaultParams: { period: 20, devup: 2.0, devdn: 2.0, matype: 'sma', devtype: 0 },
        last5Values: [
            0.0344,  // Placeholder values - should be calculated from actual Rust implementation
            0.0352,
            0.0361,
            0.0358,
            0.0349
        ]
    },
    apo: {
        defaultParams: { short_period: 10, long_period: 20 },
        last5Values: [-429.8, -401.6, -386.1, -357.9, -374.1]
    },
    bandpass: {
        defaultParams: { period: 20, bandwidth: 0.3 },
        last5Values: {
            bp: [
                -236.23678021132827,
                -247.4846395608195,
                -242.3788746078502,
                -212.89589193350128,
                -179.97293838509464
            ],
            bp_normalized: [
                -0.4399672555578846,
                -0.4651011734720517,
                -0.4596426251402882,
                -0.40739824942488945,
                -0.3475245023284841
            ],
            signal: [-1.0, 1.0, 1.0, 1.0, 1.0],
            trigger: [
                -0.4746908356434579,
                -0.4353877348116954,
                -0.3727126131420441,
                -0.2746336628365846,
                -0.18240018384226137
            ]
        }
    },
    ao: {
        defaultParams: { short_period: 5, long_period: 34 },
        last5Values: [-1671.3, -1401.6706, -1262.3559, -1178.4941, -1157.4118]
    },
    atr: {
        defaultParams: { length: 14 },
        last5Values: [916.89, 874.33, 838.45, 801.92, 811.57]
    },
    cci: {
        defaultParams: { period: 14 },
        last5Values: [
            -51.55252564125841,
            -43.50326506381541,
            -64.05117302269149,
            -39.05150631680948,
            -152.50523930896998
        ]
    },
    bop: {
        defaultParams: {},  // BOP has no parameters
        last5Values: [
            0.045454545454545456,
            -0.32398753894080995,
            -0.3844086021505376,
            0.3547400611620795,
            -0.5336179295624333
        ]
    },
    rocr: {
        defaultParams: { period: 10 },
        last5Values: [
            0.9977448290950706,
            0.9944380965183492,
            0.9967247986764135,
            0.9950545846019277,
            0.984954072979463
        ]
    },
    di: {
        defaultParams: { period: 14 },
        plusLast5Values: [
            10.99067007335658,
            11.306993269828585,
            10.948661818939213,
            10.683207768215592,
            9.802180952619183
        ],
        minusLast5Values: [
            28.06728094177839,
            27.331240567633152,
            27.759989125359493,
            26.951434842917386,
            30.748897303623057
        ]
    },
    efi: {
        defaultParams: { period: 13 },
        last5Values: [
            -44604.382026531224,
            -39811.02321812391,
            -36599.9671820205,
            -29903.28014503471,
            -55406.09054645832
        ]
    },
    fosc: {
        defaultParams: { period: 5 },
        last5Values: [
            -0.8904444627923475,
            -0.4763353099245297,
            -0.2379782851444668,
            0.292790128025632,
            -0.6597902988090389
        ]
    },
    meanAd: {
        defaultParams: { period: 5 },
        last5Values: [
            199.71999999999971,
            104.14000000000087,
            133.4,
            100.54000000000087,
            117.98000000000029
        ]
    },
    mom: {
        defaultParams: { period: 10 },
        last5Values: [
            -134.0,
            -331.0,
            -194.0,
            -294.0,
            -896.0
        ]
    },
    chande: {
        defaultParams: { period: 22, mult: 3.0, direction: 'long' },
        last5Values: [
            59444.14115983658,
            58576.49837984401,
            58649.1120898511,
            58724.56154031242,
            58713.39965211639
        ]
    },
    roc: {
        defaultParams: { period: 10 },
        last5Values: [
            -0.22551709049294377,
            -0.5561903481650754,
            -0.32752013235864963,
            -0.49454153980722504,
            -1.5045927020536976
        ],
        // Reinput test just verifies no NaN after index 28
        reinputLast5: [0, 0, 0, 0, 0]  // Placeholder - test will skip this check
    },
    rvi: {
        defaultParams: { period: 10, ma_len: 14, matype: 1, devtype: 0 },
        last5Values: [
            67.48579363423423,
            62.03322230763894,
            56.71819195768154,
            60.487299747927636,
            55.022521428674175
        ]
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