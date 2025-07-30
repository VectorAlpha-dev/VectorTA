#!/usr/bin/env node
/**
 * Generic WASM Indicator Performance Benchmark
 * 
 * Designed to benchmark multiple indicators with consistent methodology.
 * Supports two main API patterns: Safe/Simple and Fast/Unsafe.
 * 
 * To add a new indicator:
 * 1. Add its configuration to the INDICATORS object below
 * 2. Ensure the WASM bindings follow the standard naming patterns:
 *    - Safe API: indicator_js(data, ...params) -> Vec<f64>
 *    - Fast API: indicator_alloc/free/into functions
 *    - Batch API (optional): indicator_batch(data, config) -> BatchResult
 * 3. Run: node --expose-gc wasm_indicator_benchmark.js indicator_name
 */

import { performance } from 'perf_hooks';
import { readFileSync, writeFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Benchmark configuration
const CONFIG = {
    warmupTargetMs: 150,    // 150ms warmup period
    sampleCount: 10,        // Number of samples to collect
    minIterations: 10,      // Minimum iterations per sample
    disableGC: true,        // Disable GC during measurement
};

// Data sizes to benchmark
const DATA_SIZES = {
    '10k': 10_000,
    '100k': 100_000,
    '1M': 1_000_000,
};

/**
 * Indicator configurations
 * Add new indicators here by following the pattern
 */
const INDICATORS = {
    frama: {
        name: 'FRAMA',
        // Safe API
        safe: {
            fn: 'frama_js',
            params: { window: 10, sc: 300, fc: 1 }
        },
        needsMultipleInputs: true,
        // Fast/Unsafe API
        fast: {
            allocFn: 'frama_alloc',
            freeFn: 'frama_free',
            computeFn: 'frama_into',
            params: { window: 10, sc: 300, fc: 1 },
            needsMultipleInputs: true
        },
        // Batch API
        batch: {
            fn: 'frama_batch',
            fastFn: 'frama_batch_into',
            config: {
                small: {
                    window_range: [8, 12, 2],      // 3 values
                    sc_range: [200, 300, 100],     // 2 values
                    fc_range: [1, 2, 1]            // 2 values = 12 combinations
                },
                medium: {
                    window_range: [6, 14, 2],      // 5 values
                    sc_range: [100, 400, 100],    // 4 values
                    fc_range: [1, 3, 1]            // 3 values = 60 combinations
                }
            }
        }
    },
    mom: {
        name: 'MOM',
        // Safe API
        safe: {
            fn: 'mom_js',
            params: { period: 10 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'mom_alloc',
            freeFn: 'mom_free',
            computeFn: 'mom_into',
            params: { period: 10 }
        },
        // Batch API
        batch: {
            fn: 'mom_batch',
            fastFn: 'mom_batch_into',
            config: {
                small: {
                    period_range: [5, 15, 5]       // 3 values
                },
                medium: {
                    period_range: [5, 50, 5]       // 10 values
                }
            }
        }
    },
    pwma: {
        name: 'PWMA',
        // Safe API
        safe: {
            fn: 'pwma_js',
            params: { period: 14 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'pwma_alloc',
            freeFn: 'pwma_free',
            computeFn: 'pwma_into',
            params: { period: 14 }
        },
        // Batch API
        batch: {
            fn: 'pwma_batch_js',
            fastFn: 'pwma_batch_into',
            config: {
                small: {
                    period_range: [5, 15, 2]       // 6 values
                },
                medium: {
                    period_range: [5, 25, 2]       // 11 values
                }
            }
        }
    },
    vosc: {
        name: 'VOSC',
        // Safe API
        safe: {
            fn: 'vosc_js',
            params: { short_period: 2, long_period: 5 },
            inputs: ['volume']  // VOSC uses volume data
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'vosc_alloc',
            freeFn: 'vosc_free',
            computeFn: 'vosc_into',
            params: { short_period: 2, long_period: 5 },
            inputs: ['volume']
        },
        // Batch API
        batch: {
            fn: 'vosc_batch',
            config: {
                small: {
                    short_period_range: [2, 4, 1],   // 3 values
                    long_period_range: [5, 7, 1]     // 3 values = 9 combinations (filtered for valid)
                },
                medium: {
                    short_period_range: [2, 10, 1],  // 9 values
                    long_period_range: [10, 20, 2]   // 6 values = many valid combinations
                }
            },
            inputs: ['volume']  // VOSC uses volume data for batch API too
        }
    },
    adxr: {
        name: 'ADXR',
        // Safe API
        safe: {
            fn: 'adxr_js',
            params: { period: 14 },
            needsMultipleInputs: true
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'adxr_alloc',
            freeFn: 'adxr_free',
            computeFn: 'adxr_into',
            params: { period: 14 },
            needsMultipleInputs: true
        },
        // Batch API
        batch: {
            fn: 'adxr_batch',
            fastFn: 'adxr_batch_into',
            config: {
                small: {
                    period_range: [10, 20, 5]  // 3 values: 10, 15, 20
                },
                medium: {
                    period_range: [10, 30, 5]  // 5 values: 10, 15, 20, 25, 30
                }
            },
            needsMultipleInputs: true
        }
    },
    rocp: {
        name: 'ROCP',
        // Safe API
        safe: {
            fn: 'rocp_js',
            params: { period: 10 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'rocp_alloc',
            freeFn: 'rocp_free',
            computeFn: 'rocp_into',
            params: { period: 10 }
        },
        // Batch API
        batch: {
            fn: 'rocp_batch',
            fastFn: 'rocp_batch_into',
            config: {
                small: {
                    period_range: [5, 15, 5]       // 3 values
                },
                medium: {
                    period_range: [5, 50, 5]       // 10 values
                }
            }
        }
    },
    alma: {
        name: 'ALMA',
        // Safe API
        safe: {
            fn: 'alma_js',
            params: { period: 9, offset: 0.85, sigma: 6.0 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'alma_alloc',
            freeFn: 'alma_free',
            computeFn: 'alma_into',
            params: { period: 9, offset: 0.85, sigma: 6.0 }
        },
        // Batch API (optional)
        batch: {
            fn: 'alma_batch',
            config: {
                // Reduced combinations for faster execution
                small: {
                    period_range: [5, 15, 5],      // 3 values
                    offset_range: [0.7, 0.9, 0.1], // 3 values  
                    sigma_range: [4.0, 8.0, 2.0]   // 3 values
                    // Total: 27 combinations
                },
                medium: {
                    period_range: [5, 25, 4],      // 6 values
                    offset_range: [0.5, 0.9, 0.1], // 5 values
                    sigma_range: [3.0, 9.0, 3.0]   // 3 values
                    // Total: 90 combinations
                }
            },
            // Fast batch API (optional)
            fastFn: 'alma_batch_into'
        }
    },
    damiani_volatmeter: {
        name: 'Damiani Volatmeter',
        // Safe API
        safe: {
            fn: 'damiani_volatmeter_js',
            params: { vis_atr: 13, vis_std: 20, sed_atr: 40, sed_std: 100, threshold: 1.4 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'damiani_volatmeter_alloc',
            freeFn: 'damiani_volatmeter_free',
            computeFn: 'damiani_volatmeter_into',
            params: { vis_atr: 13, vis_std: 20, sed_atr: 40, sed_std: 100, threshold: 1.4 },
            dualOutput: true  // Has two outputs (vol and anti)
        },
        // Batch API
        batch: {
            fn: 'damiani_volatmeter_batch',
            config: {
                small: {
                    vis_atr_range: [10, 20, 5],      // 3 values
                    vis_std_range: [15, 25, 5],      // 3 values
                    sed_atr_range: [40, 40, 0],      // 1 value
                    sed_std_range: [100, 100, 0],    // 1 value
                    threshold_range: [1.4, 1.4, 0.0] // 1 value = 9 combinations
                },
                medium: {
                    vis_atr_range: [10, 30, 5],      // 5 values
                    vis_std_range: [15, 35, 5],      // 5 values
                    sed_atr_range: [30, 50, 10],     // 3 values
                    sed_std_range: [80, 120, 20],    // 3 values
                    threshold_range: [1.0, 2.0, 0.5] // 3 values = 675 combinations
                }
            },
            fastFn: 'damiani_volatmeter_batch_into'
        }
    },
    aroon: {
        name: 'Aroon',
        needsMultipleInputs: true,  // Uses high, low
        // Safe API
        safe: {
            fn: 'aroon_js',
            params: { length: 14 },
            needsMultipleInputs: true
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'aroon_alloc',
            freeFn: 'aroon_free',
            computeFn: 'aroon_into',
            params: { length: 14 },
            needsMultipleInputs: true,
            dualOutput: true  // Has two outputs (up and down)
        },
        // Batch API
        batch: {
            fn: 'aroon_batch',
            config: {
                small: {
                    length_range: [10, 20, 5]       // 3 values: 10, 15, 20
                },
                medium: {
                    length_range: [5, 25, 5]        // 5 values: 5, 10, 15, 20, 25
                }
            },
            // Fast batch API
            fastFn: 'aroon_batch_into',
            dualOutput: true
        }
    },
    mean_ad: {
        name: 'Mean Absolute Deviation',
        // Safe API
        safe: {
            fn: 'mean_ad_js',
            params: { period: 5 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'mean_ad_alloc',
            freeFn: 'mean_ad_free',
            computeFn: 'mean_ad_into',
            params: { period: 5 }
        },
        // Batch API
        batch: {
            fn: 'mean_ad_batch',
            config: {
                small: {
                    period_range: [5, 15, 5]       // 3 values: 5, 10, 15
                },
                medium: {
                    period_range: [5, 25, 5]       // 5 values: 5, 10, 15, 20, 25
                }
            }
        }
    },
    bollinger_bands: {
        name: 'Bollinger Bands',
        // Safe API
        safe: {
            fn: 'bollinger_bands_js',
            params: { period: 20, devup: 2.0, devdn: 2.0, matype: 'sma', devtype: 0 },
            // Bollinger Bands returns 3 outputs flattened as [upper..., middle..., lower...]
            multiOutput: 3
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'bollinger_bands_alloc',
            freeFn: 'bollinger_bands_free',
            computeFn: 'bollinger_bands_into',
            params: { period: 20, devup: 2.0, devdn: 2.0, matype: 'sma', devtype: 0 },
            // Multiple output pointers for upper, middle, lower
            multiOutput: 3
        },
        // Batch API
        batch: {
            fn: 'bollinger_bands_batch',
            config: {
                small: {
                    period_range: [10, 30, 10],     // 3 values
                    devup_range: [1.0, 3.0, 1.0],   // 3 values
                    devdn_range: [2.0, 2.0, 0.0],   // 1 value
                    matype: 'sma',
                    devtype: 0
                    // Total: 9 combinations
                },
                medium: {
                    period_range: [10, 50, 10],     // 5 values
                    devup_range: [1.0, 3.0, 0.5],   // 5 values
                    devdn_range: [1.0, 3.0, 0.5],   // 5 values
                    matype: 'sma',
                    devtype: 0
                    // Total: 125 combinations
                }
            }
        }
    },
    vlma: {
        name: 'VLMA',
        // Safe API
        safe: {
            fn: 'vlma_js',
            params: { min_period: 5, max_period: 50, matype: 'sma', devtype: 0 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'vlma_alloc',
            freeFn: 'vlma_free',
            computeFn: 'vlma_into',
            params: { min_period: 5, max_period: 50, matype: 'sma', devtype: 0 }
        },
        // Batch API
        batch: {
            fn: 'vlma_batch',
            config: {
                small: {
                    min_period_range: [5, 15, 5],    // 3 values: 5, 10, 15
                    max_period_range: [30, 50, 10],  // 3 values: 30, 40, 50
                    devtype_range: [0, 2, 1],        // 3 values: 0, 1, 2
                    matype: 'sma'
                    // Total: 27 combinations
                },
                medium: {
                    min_period_range: [5, 25, 5],    // 5 values: 5, 10, 15, 20, 25
                    max_period_range: [30, 60, 10],  // 4 values: 30, 40, 50, 60
                    devtype_range: [0, 2, 1],        // 3 values: 0, 1, 2
                    matype: 'ema'
                    // Total: 60 combinations
                }
            },
            fastFn: 'vlma_batch_into'
        }
    },
    keltner: {
        name: 'Keltner Channels',
        needsMultipleInputs: true,  // Uses high, low, close, source
        hasMultipleOutputs: 3,      // Returns upper, middle, lower bands
        // Safe API
        safe: {
            fn: 'keltner_js',
            params: { period: 20, multiplier: 2.0, ma_type: 'ema' }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'keltner_alloc',
            freeFn: 'keltner_free',
            computeFn: 'keltner_into',
            params: { period: 20, multiplier: 2.0, ma_type: 'ema' },
            needsMultipleInputs: true,
            hasMultipleOutputs: 3
        },
        // Batch API
        batch: {
            fn: 'keltner_batch',
            config: {
                small: {
                    period_range: [10, 30, 10],        // 3 values: 10, 20, 30
                    multiplier_range: [1.0, 3.0, 1.0], // 3 values: 1.0, 2.0, 3.0
                    ma_type: 'ema'
                    // Total: 9 combinations
                },
                medium: {
                    period_range: [10, 50, 10],        // 5 values
                    multiplier_range: [1.0, 4.0, 0.5], // 7 values  
                    ma_type: 'ema'
                    // Total: 35 combinations
                }
            }
        }
    },
    fisher: {
        name: 'Fisher Transform',
        needsMultipleInputs: true,  // Uses high, low
        hasMultipleOutputs: 2,      // Returns fisher and signal
        // Safe API
        safe: {
            fn: 'fisher_js',
            params: { period: 9 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'fisher_alloc',
            freeFn: 'fisher_free',
            computeFn: 'fisher_into',
            params: { period: 9 },
            needsMultipleInputs: true,
            hasMultipleOutputs: 2
        },
        // Batch API
        batch: {
            fn: 'fisher_batch',
            fastFn: 'fisher_batch_into',
            config: {
                small: {
                    period_range: [5, 15, 5]       // 3 values: 5, 10, 15
                },
                medium: {
                    period_range: [5, 25, 4]       // 6 values: 5, 9, 13, 17, 21, 25
                }
            },
            needsMultipleInputs: true,
            hasMultipleOutputs: 2
        }
    },
    ao: {
        name: 'AO',
        needsMultipleInputs: true,  // Uses high, low
        // Safe API
        safe: {
            fn: 'ao_js',
            params: { short_period: 5, long_period: 34 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'ao_alloc',
            freeFn: 'ao_free',
            computeFn: 'ao_into',
            params: { short_period: 5, long_period: 34 },
            needsMultipleInputs: true
        },
        // Batch API
        batch: {
            fn: 'ao_batch',
            fastFn: 'ao_batch_into',
            config: {
                small: {
                    short_period_range: [3, 7, 2],   // 3 values: 3, 5, 7
                    long_period_range: [20, 40, 10]  // 3 values: 20, 30, 40 = 9 combinations
                },
                medium: {
                    short_period_range: [3, 11, 2],  // 5 values: 3, 5, 7, 9, 11
                    long_period_range: [20, 50, 10]  // 4 values: 20, 30, 40, 50 = 20 combinations
                }
            },
            needsMultipleInputs: true
        }
    },
    adosc: {
        name: 'ADOSC',
        needsMultipleInputs: true,  // Uses high, low, close, volume
        // Safe API
        safe: {
            fn: 'adosc_js',
            params: { short_period: 3, long_period: 10 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'adosc_alloc',
            freeFn: 'adosc_free',
            computeFn: 'adosc_into',
            params: { short_period: 3, long_period: 10 },
            needsMultipleInputs: true
        },
        // Batch API
        batch: {
            fn: 'adosc_batch',
            config: {
                small: {
                    short_period_range: [2, 4, 1],   // 3 values
                    long_period_range: [8, 12, 2]    // 3 values = 9 combinations
                },
                medium: {
                    short_period_range: [2, 6, 1],   // 5 values
                    long_period_range: [8, 16, 2]    // 5 values = 25 combinations
                }
            }
        }
    },
    chande: {
        name: 'Chande',
        needsMultipleInputs: true,  // Uses high, low, close
        // Safe API
        safe: {
            fn: 'chande_js',
            params: { period: 22, mult: 3.0, direction: 'long' }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'chande_alloc',
            freeFn: 'chande_free',
            computeFn: 'chande_into',
            params: { period: 22, mult: 3.0, direction: 'long' },
            needsMultipleInputs: true
        },
        // Batch API
        batch: {
            fn: 'chande_batch_js',
            fastFn: 'chande_batch_into',
            config: {
                small: {
                    period_range: [15, 25, 5],      // 3 values: 15, 20, 25
                    mult_range: [2.0, 4.0, 1.0],    // 3 values: 2.0, 3.0, 4.0
                    direction: 'long'               // 9 combinations total
                },
                medium: {
                    period_range: [10, 30, 5],      // 5 values: 10, 15, 20, 25, 30
                    mult_range: [2.0, 5.0, 0.5],    // 7 values: 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0
                    direction: 'short'              // 35 combinations total
                }
            }
        }
    },
    acosc: {
        name: 'ACOSC',
        needsMultipleInputs: true,  // Uses high, low (not close)
        // Safe API
        safe: {
            fn: 'acosc_js',
            params: {}  // No parameters for ACOSC
        },
        // Fast/Unsafe API  
        fast: {
            allocFn: 'acosc_alloc',
            freeFn: 'acosc_free',
            computeFn: 'acosc_into',
            params: {},
            needsMultipleInputs: true,
            // For ACOSC: high_ptr, low_ptr, osc_ptr, change_ptr, len
            dualOutput: true  // Has two outputs (osc and change)
        },
        // Batch API
        batch: {
            fn: 'acosc_batch',
            config: {
                // ACOSC has no parameters, so batch always returns 1 row
                small: {},
                medium: {}
            }
        }
    },
    medprice: {
        name: 'MEDPRICE',
        needsMultipleInputs: true,  // Uses high, low
        // Safe API
        safe: {
            fn: 'medprice_js',
            params: {}  // No parameters for MEDPRICE
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'medprice_alloc',
            freeFn: 'medprice_free',
            computeFn: 'medprice_into',
            params: {},
            needsMultipleInputs: true
        },
        // Batch API
        batch: {
            fn: 'medprice_batch',
            config: {
                // MEDPRICE has no parameters, so batch always returns 1 row
                small: {},
                medium: {}
            }
        }
    },
    vpwma: {
        name: 'VPWMA',
        // Safe API
        safe: {
            fn: 'vpwma_js',
            params: { period: 14, power: 0.382 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'vpwma_alloc',
            freeFn: 'vpwma_free',
            computeFn: 'vpwma_into',
            params: { period: 14, power: 0.382 }
        },
        // Batch API
        batch: {
            fn: 'vpwma_batch_js',
            config: {
                small: {
                    period_range: [10, 20, 5],      // 3 values
                    power_range: [0.2, 0.6, 0.2]    // 3 values = 9 combinations
                },
                medium: {
                    period_range: [10, 30, 5],      // 5 values
                    power_range: [0.1, 0.9, 0.2]    // 5 values = 25 combinations
                }
            },
            // Fast batch API
            fastFn: 'vpwma_batch_into'
        }
    },
    edcf: {
        name: 'EDCF',
        // Safe API
        safe: {
            fn: 'edcf_js',
            params: { period: 15 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'edcf_alloc',
            freeFn: 'edcf_free',
            computeFn: 'edcf_into',
            params: { period: 15 }
        }
        // No batch API available
    },
    highpass: {
        name: 'HighPass',
        // Safe API
        safe: {
            fn: 'highpass_js',
            params: { period: 48 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'highpass_alloc',
            freeFn: 'highpass_free',
            computeFn: 'highpass_into',
            params: { period: 48 }
        },
        // Batch API
        batch: {
            fn: 'highpass_batch',
            config: {
                small: {
                    period_range: [30, 60, 10]  // 4 values: 30, 40, 50, 60
                },
                medium: {
                    period_range: [20, 80, 10]  // 7 values: 20, 30, 40, ..., 80
                }
            },
            // Fast batch API
            fastFn: 'highpass_batch_into'
        }
    },
    jsa: {
        name: 'JSA',
        // Safe API
        safe: {
            fn: 'jsa_js',
            params: { period: 30 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'jsa_alloc',
            freeFn: 'jsa_free',
            computeFn: 'jsa_fast',
            params: { period: 30 }
        },
        // Batch API
        batch: {
            fn: 'jsa_batch',
            fastFn: 'jsa_batch_into',
            config: {
                small: {
                    period_range: [10, 40, 10]  // 4 values: 10, 20, 30, 40
                },
                medium: {
                    period_range: [10, 50, 5]   // 9 values: 10, 15, 20, ..., 50
                }
            }
        }
    },
    linearreg_slope: {
        name: 'LinearRegSlope',
        // Safe API
        safe: {
            fn: 'linearreg_slope_js',
            params: { period: 14 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'linearreg_slope_alloc',
            freeFn: 'linearreg_slope_free',
            computeFn: 'linearreg_slope_into',
            params: { period: 14 }
        },
        // Batch API
        batch: {
            fn: 'linearreg_slope_batch',
            config: {
                small: {
                    period_range: [10, 20, 5]  // 3 values: 10, 15, 20
                },
                medium: {
                    period_range: [5, 25, 5]   // 5 values: 5, 10, 15, 20, 25
                }
            }
        }
    },
    maaq: {
        name: 'MAAQ',
        // Safe API
        safe: {
            fn: 'maaq_js',
            params: { period: 11, fast_period: 2, slow_period: 30 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'maaq_alloc',
            freeFn: 'maaq_free',
            computeFn: 'maaq_into',
            params: { period: 11, fast_period: 2, slow_period: 30 }
        },
        // Batch API
        batch: {
            fn: 'maaq_batch_js',
            fastFn: 'maaq_batch_into',
            config: {
                small: {
                    period_range: [10, 20, 5],       // 3 values: 10, 15, 20
                    fast_period_range: [2, 4, 1],    // 3 values: 2, 3, 4
                    slow_period_range: [20, 40, 10]  // 3 values: 20, 30, 40
                    // Total: 27 combinations
                },
                medium: {
                    period_range: [10, 30, 5],       // 5 values: 10, 15, 20, 25, 30
                    fast_period_range: [2, 6, 2],    // 3 values: 2, 4, 6
                    slow_period_range: [20, 50, 10]  // 4 values: 20, 30, 40, 50
                    // Total: 60 combinations
                }
            }
        }
    },
    smma: {
        name: 'SMMA',
        // Safe API
        safe: {
            fn: 'smma',
            params: { period: 7 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'smma_alloc',
            freeFn: 'smma_free',
            computeFn: 'smma_into',
            params: { period: 7 }
        },
        // Batch API
        batch: {
            fn: 'smma_batch_new',
            config: {
                small: {
                    period_range: [5, 15, 5]  // 3 values: 5, 10, 15
                },
                medium: {
                    period_range: [5, 25, 5]  // 5 values: 5, 10, 15, 20, 25
                }
            },
            // Fast batch API
            fastFn: 'smma_batch_into'
        }
    },
    supersmoother: {
        name: 'SuperSmoother',
        // Safe API
        safe: {
            fn: 'supersmoother_js',
            params: { period: 14 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'supersmoother_alloc',
            freeFn: 'supersmoother_free',
            computeFn: 'supersmoother_into',
            params: { period: 14 }
        },
        // Batch API
        batch: {
            fn: 'supersmoother_batch_js',  // Use the JS API that takes parameters
            fastFn: 'supersmoother_batch_into',
            config: {
                small: {
                    period_range: [5, 15, 2]       // 6 values
                },
                medium: {
                    period_range: [5, 25, 2]       // 11 values
                }
            }
        }
    },
    ehlers_itrend: {
        name: 'Ehlers Instantaneous Trendline',
        // Safe API
        safe: {
            fn: 'ehlers_itrend_js',
            params: { warmup_bars: 20, max_dc_period: 48 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'ehlers_itrend_alloc',
            freeFn: 'ehlers_itrend_free',
            computeFn: 'ehlers_itrend_into',
            params: { warmup_bars: 20, max_dc_period: 48 }
        },
        // Batch API (optional)
        batch: {
            fn: 'ehlers_itrend_batch',
            config: {
                // Reduced combinations for faster execution
                small: {
                    warmup_bars_range: [10, 20, 5],      // 3 values
                    max_dc_period_range: [40, 50, 5]     // 3 values
                    // Total: 9 combinations
                },
                medium: {
                    warmup_bars_range: [10, 30, 5],      // 5 values
                    max_dc_period_range: [30, 60, 10]    // 4 values
                    // Total: 20 combinations
                }
            }
        }
    },
    fwma: {
        name: 'FWMA',
        // Safe API
        safe: {
            fn: 'fwma_js',
            params: { period: 5 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'fwma_alloc',
            freeFn: 'fwma_free',
            computeFn: 'fwma_into',
            params: { period: 5 }
        },
        // Batch API (optional)
        batch: {
            fn: 'fwma_batch_js',
            config: {
                // Reduced combinations for faster execution
                small: {
                    period_range: [3, 15, 3]       // 5 values
                    // Total: 5 combinations
                },
                medium: {
                    period_range: [3, 30, 3]       // 10 values
                    // Total: 10 combinations
                }
            },
            // Fast batch API (optional)
            fastFn: 'fwma_batch_into'
        }
    },
    hma: {
        name: 'Hull Moving Average (HMA)',
        // Safe API
        safe: {
            fn: 'hma_js',
            params: { period: 5 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'hma_alloc',
            freeFn: 'hma_free',
            computeFn: 'hma_into',
            params: { period: 5 }
        },
        // Batch API
        batch: {
            fn: 'hma_batch_js',  // Use JS API for parameters
            config: {
                small: {
                    period_range: [5, 15, 5]       // 3 values: 5, 10, 15
                },
                medium: {
                    period_range: [5, 25, 5]       // 5 values: 5, 10, 15, 20, 25
                }
            },
            // Fast batch API
            fastFn: 'hma_batch_into'
        }
    },
    kama: {
        name: 'KAMA',
        // Safe API
        safe: {
            fn: 'kama_js',
            params: { period: 30 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'kama_alloc',
            freeFn: 'kama_free',
            computeFn: 'kama_into',
            params: { period: 30 }
        },
        // Batch API
        batch: {
            fn: 'kama_batch_js',
            config: {
                small: {
                    period_range: [10, 30, 10]  // 3 values: 10, 20, 30
                },
                medium: {
                    period_range: [10, 50, 5]   // 9 values: 10, 15, 20, ..., 50
                }
            },
            fastFn: 'kama_batch_into'
        }
    },
    fosc: {
        name: 'FOSC (Forecast Oscillator)',
        // Safe API
        safe: {
            fn: 'fosc_js',
            params: { period: 5 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'fosc_alloc',
            freeFn: 'fosc_free',
            computeFn: 'fosc_into',
            params: { period: 5 }
        },
        // Batch API
        batch: {
            fn: 'fosc_batch',
            config: {
                small: {
                    period_range: [3, 10, 1]  // 8 values: 3, 4, 5, 6, 7, 8, 9, 10
                },
                medium: {
                    period_range: [3, 20, 1]  // 18 values: 3, 4, 5, ..., 20
                }
            },
            fastFn: 'fosc_batch_into'
        }
    },
    sqwma: {
        name: 'SQWMA (Square Weighted Moving Average)',
        // Safe API
        safe: {
            fn: 'sqwma_js',
            params: { period: 14 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'sqwma_alloc',
            freeFn: 'sqwma_free',
            computeFn: 'sqwma_into',
            params: { period: 14 }
        },
        // Batch API
        batch: {
            fn: 'sqwma_batch_js',
            config: {
                small: {
                    period_range: [5, 20, 5]    // 4 values: 5, 10, 15, 20
                },
                medium: {
                    period_range: [5, 30, 5]    // 6 values: 5, 10, 15, 20, 25, 30
                }
            },
            fastFn: 'sqwma_batch_into'
        }
    },
    mama: {
        name: 'MAMA (MESA Adaptive Moving Average)',
        // Safe API
        safe: {
            fn: 'mama_js',
            params: { fast_limit: 0.5, slow_limit: 0.05 }
        },
        // Fast/Unsafe API  
        fast: {
            allocFn: 'mama_alloc',
            freeFn: 'mama_free',
            computeFn: 'mama_into',
            params: { fast_limit: 0.5, slow_limit: 0.05 },
            // MAMA has dual outputs
            dualOutput: true
        },
        // Batch API
        batch: {
            fn: 'mama_batch_js',
            config: {
                small: {
                    fast_limit_range: [0.3, 0.7, 0.2],  // 3 values: 0.3, 0.5, 0.7
                    slow_limit_range: [0.03, 0.07, 0.02] // 3 values: 0.03, 0.05, 0.07
                    // Total: 9 combinations
                },
                medium: {
                    fast_limit_range: [0.2, 0.8, 0.1],  // 7 values
                    slow_limit_range: [0.02, 0.08, 0.01] // 7 values
                    // Total: 49 combinations
                }
            },
            // Batch metadata functions
            metadataFn: 'mama_batch_metadata_js',
            rowsColsFn: 'mama_batch_rows_cols_js',
            // Fast batch API
            fastFn: 'mama_batch_into',
            dualOutput: true
        }
    },
    medium_ad: {
        name: 'Medium AD',
        // Safe API
        safe: {
            fn: 'medium_ad_js',
            params: { period: 5 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'medium_ad_alloc',
            freeFn: 'medium_ad_free',
            computeFn: 'medium_ad_into',
            params: { period: 5 }
        },
        // Batch API
        batch: {
            fn: 'medium_ad_batch',
            config: {
                small: {
                    period_range: [3, 7, 1]       // 5 values: 3, 4, 5, 6, 7
                },
                medium: {
                    period_range: [3, 15, 1]      // 13 values: 3, 4, 5, ..., 15
                }
            },
            // Fast batch API
            fastFn: 'medium_ad_batch_into'
        }
    },
    minmax: {
        name: 'MinMax',
        needsMultipleInputs: true,  // Uses high, low
        hasMultipleOutputs: 4,      // Returns is_min, is_max, last_min, last_max
        // Safe API
        safe: {
            fn: 'minmax_js',
            params: { order: 3 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'minmax_alloc',
            freeFn: 'minmax_free',
            computeFn: 'minmax_into',
            params: { order: 3 },
            needsMultipleInputs: true,
            hasMultipleOutputs: 4
        },
        // Batch API
        batch: {
            fn: 'minmax_batch',
            config: {
                small: {
                    order_range: [2, 5, 1]       // 4 values: 2, 3, 4, 5
                },
                medium: {
                    order_range: [3, 20, 1]      // 18 values: 3, 4, 5, ..., 20
                }
            },
            needsMultipleInputs: true,
            hasMultipleOutputs: 4,
            // Fast batch API
            fastFn: 'minmax_batch_into'
        }
    },
    reflex: {
        name: 'Reflex',
        // Safe API
        safe: {
            fn: 'reflex_js',
            params: { period: 20 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'reflex_alloc',
            freeFn: 'reflex_free',
            computeFn: 'reflex_into',
            params: { period: 20 }
        },
        // Batch API
        batch: {
            fn: 'reflex_batch_js',
            config: {
                small: {
                    period_range: [10, 30, 10]  // 3 values: 10, 20, 30
                },
                medium: {
                    period_range: [10, 50, 5]   // 9 values: 10, 15, 20, ..., 50
                }
            },
            // Batch metadata functions
            metadataFn: 'reflex_batch_metadata_js',
            rowsColsFn: 'reflex_batch_rows_cols_js'
            // Note: No fastFn for reflex batch as it doesn't have batch_into
        }
    },
    rocr: {
        name: 'ROCR',
        // Safe API
        safe: {
            fn: 'rocr_js',
            params: { period: 9 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'rocr_alloc',
            freeFn: 'rocr_free',
            computeFn: 'rocr_into',
            params: { period: 9 }
        },
        // Batch API
        batch: {
            fn: 'rocr_batch',
            config: {
                small: {
                    period_range: [5, 15, 5]     // 3 values: 5, 10, 15
                },
                medium: {
                    period_range: [5, 20, 3]     // 6 values: 5, 8, 11, 14, 17, 20
                }
            },
            // Fast batch API
            fastFn: 'rocr_batch_into'
        }
    },
    swma: {
        name: 'SWMA (Symmetric Weighted Moving Average)',
        // Safe API
        safe: {
            fn: 'swma_js',
            params: { period: 5 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'swma_alloc',
            freeFn: 'swma_free',
            computeFn: 'swma_into',
            params: { period: 5 }
        },
        // Batch API
        batch: {
            fn: 'swma_batch',  // This uses the new unified function name swma_batch_unified_js
            config: {
                small: {
                    period_range: [3, 15, 3]       // 5 values: 3, 6, 9, 12, 15
                    // Total: 5 combinations
                },
                medium: {
                    period_range: [3, 30, 3]       // 10 values: 3, 6, 9, ..., 30
                    // Total: 10 combinations
                }
            },
            // Fast batch API
            fastFn: 'swma_batch_into'
        }
    },
    cwma: {
        name: 'CWMA',
        // Safe API
        safe: {
            fn: 'cwma_js',
            params: { period: 14 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'cwma_alloc',
            freeFn: 'cwma_free',
            computeFn: 'cwma_into',
            params: { period: 14 }
        },
        // Batch API
        batch: {
            fn: 'cwma_batch',
            config: {
                small: {
                    period_range: [10, 20, 5]      // 3 values
                },
                medium: {
                    period_range: [10, 50, 5]      // 9 values
                }
            },
            // Fast batch API
            fastFn: 'cwma_batch_into'
        }
    },
    dema: {
        name: 'DEMA',
        // Safe API
        safe: {
            fn: 'dema_js',
            params: { period: 30 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'dema_alloc',
            freeFn: 'dema_free',
            computeFn: 'dema_into',
            params: { period: 30 }
        },
        // Batch API
        batch: {
            fn: 'dema_batch',
            config: {
                small: {
                    period_range: [10, 30, 10]     // 3 values
                },
                medium: {
                    period_range: [10, 50, 10]     // 5 values
                }
            }
        }
    },
    epma: {
        name: 'EPMA',
        // Safe API
        safe: {
            fn: 'epma_js',
            params: { period: 11, offset: 4 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'epma_alloc',
            freeFn: 'epma_free',
            computeFn: 'epma_into',
            params: { period: 11, offset: 4 }
        },
        // Batch API
        batch: {
            fn: 'epma_batch',
            config: {
                small: {
                    period_range: [5, 15, 5],      // 3 values
                    offset_range: [2, 4, 1]        // 3 values
                    // Total: 9 combinations
                },
                medium: {
                    period_range: [5, 25, 2],      // 11 values
                    offset_range: [1, 4, 1]        // 4 values
                    // Total: 44 combinations
                }
            },
            // Fast batch API
            fastFn: 'epma_batch_into'
        }
    },
    jma: {
        name: 'JMA',
        // Safe API
        safe: {
            fn: 'jma_js',
            params: { period: 7, phase: 50.0, power: 2 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'jma_alloc',
            freeFn: 'jma_free',
            computeFn: 'jma_into',
            params: { period: 7, phase: 50.0, power: 2 }
        },
        // Batch API
        batch: {
            fn: 'jma_batch',
            config: {
                small: {
                    period_range: [5, 15, 5],      // 3 values
                    phase_range: [0.0, 100.0, 50.0], // 3 values
                    power_range: [1, 3, 1]         // 3 values
                    // Total: 27 combinations
                },
                medium: {
                    period_range: [5, 25, 5],      // 5 values
                    phase_range: [0.0, 100.0, 25.0], // 5 values
                    power_range: [1, 3, 1]         // 3 values
                    // Total: 75 combinations
                }
            },
            // Fast batch API
            fastFn: 'jma_batch_into'
        }
    },
    highpass_2_pole: {
        name: 'HighPass 2-Pole',
        // Safe API
        safe: {
            fn: 'highpass_2_pole_js',
            params: { period: 48, k: 0.707 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'highpass_2_pole_alloc',
            freeFn: 'highpass_2_pole_free',
            computeFn: 'highpass_2_pole_into',
            params: { period: 48, k: 0.707 }
        },
        // Batch API
        batch: {
            fn: 'highpass_2_pole_batch',
            config: {
                small: {
                    period_range: [20, 60, 20],    // 3 values
                    k_range: [0.5, 0.9, 0.2]       // 3 values
                    // Total: 9 combinations
                },
                medium: {
                    period_range: [20, 80, 10],    // 7 values
                    k_range: [0.3, 0.9, 0.1]       // 7 values
                    // Total: 49 combinations
                }
            },
            // Fast batch API
            fastFn: 'highpass_2_pole_batch_into'
        }
    },
    nma: {
        name: 'NMA',
        // Safe API
        safe: {
            fn: 'nma_js',
            params: { period: 40 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'nma_alloc',
            freeFn: 'nma_free',
            computeFn: 'nma_into',
            params: { period: 40 }
        },
        // Batch API
        batch: {
            fn: 'nma_batch',
            config: {
                small: {
                    period_range: [20, 60, 20]     // 3 values
                },
                medium: {
                    period_range: [10, 90, 10]     // 9 values
                }
            },
            // Fast batch API
            fastFn: 'nma_batch_into'
        }
    },
    sma: {
        name: 'SMA',
        // Safe API
        safe: {
            fn: 'sma',
            params: { period: 14 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'sma_alloc',
            freeFn: 'sma_free',
            computeFn: 'sma_into',
            params: { period: 14 }
        },
        // Batch API
        batch: {
            fn: 'sma_batch',
            config: {
                small: {
                    period_range: [5, 15, 5]       // 3 values
                },
                medium: {
                    period_range: [5, 25, 2]       // 11 values
                }
            },
            // Fast batch API
            fastFn: 'sma_batch_into'
        }
    },
    supersmoother_3_pole: {
        name: 'SuperSmoother 3-Pole',
        // Safe API
        safe: {
            fn: 'supersmoother_3_pole_js',
            params: { period: 14 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'supersmoother_3_pole_alloc',
            freeFn: 'supersmoother_3_pole_free',
            computeFn: 'supersmoother_3_pole_into',
            params: { period: 14 }
        },
        // Batch API
        batch: {
            fn: 'supersmoother_3_pole_batch',
            config: {
                small: {
                    period_range: [10, 20, 5]      // 3 values
                },
                medium: {
                    period_range: [10, 30, 2]      // 11 values
                }
            },
            // Fast batch API
            fastFn: 'supersmoother_3_pole_batch_into'
        }
    },
    ema: {
        name: 'EMA',
        safe: {
            fn: 'ema_js',
            params: { period: 20 }
        },
        fast: {
            allocFn: 'ema_alloc',
            freeFn: 'ema_free',
            computeFn: 'ema_into',
            params: { period: 20 }
        },
        batch: {
            fn: 'ema_batch',
            config: {
                small: {
                    period_range: [5, 20, 5]      // 4 values
                },
                medium: {
                    period_range: [5, 50, 5]      // 10 values  
                }
            },
            fastFn: 'ema_batch_into'
        }
    },
    trima: {
        name: 'TRIMA',
        // Safe API
        safe: {
            fn: 'trima_js',
            params: { period: 30 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'trima_alloc',
            freeFn: 'trima_free',
            computeFn: 'trima_into',
            params: { period: 30 }
        },
        // Batch API
        batch: {
            fn: 'trima_batch',
            config: {
                small: {
                    period_range: [10, 30, 10]      // 3 values
                },
                medium: {
                    period_range: [10, 50, 5]       // 9 values
                }
            },
            // Fast batch API
            fastFn: 'trima_batch_into'
        }
    },
    tema: {
        name: 'TEMA',
        // Safe API
        safe: {
            fn: 'tema_js',
            params: { period: 14 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'tema_alloc',
            freeFn: 'tema_free',
            computeFn: 'tema_into',
            params: { period: 14 }
        },
        // Batch API
        batch: {
            fn: 'tema_batch',
            config: {
                small: {
                    period_range: [5, 15, 5]       // 3 values
                },
                medium: {
                    period_range: [5, 30, 5]       // 6 values
                }
            },
            fastFn: 'tema_batch_into'
        }
    },
    wilders: {
        name: 'Wilders',
        // Safe API
        safe: {
            fn: 'wilders_js',
            params: { period: 5 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'wilders_alloc',
            freeFn: 'wilders_free',
            computeFn: 'wilders_into',
            params: { period: 5 }
        },
        // Batch API
        batch: {
            fn: 'wilders_batch',
            config: {
                small: {
                    period_range: [5, 15, 5]       // 3 values
                    // Total: 3 combinations
                },
                medium: {
                    period_range: [5, 25, 4]       // 6 values
                    // Total: 6 combinations
                }
            }
        }
    },
    wma: {
        name: 'WMA (Weighted Moving Average)',
        // Safe API
        safe: {
            fn: 'wma_js',
            params: { period: 30 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'wma_alloc',
            freeFn: 'wma_free',
            computeFn: 'wma_into',
            params: { period: 30 }
        },
        // Batch API
        batch: {
            fn: 'wma_batch_js',
            config: {
                small: {
                    period_range: [10, 30, 10]     // 3 values: 10, 20, 30
                },
                medium: {
                    period_range: [10, 50, 5]      // 9 values: 10, 15, 20, ..., 50
                }
            },
            fastFn: 'wma_batch_into'
        }
    },
    gaussian: {
        name: 'Gaussian',
        safe: {
            fn: 'gaussian_js',
            params: { period: 14, poles: 4 }
        },
        fast: {
            allocFn: 'gaussian_alloc',
            freeFn: 'gaussian_free',
            computeFn: 'gaussian_into',
            params: { period: 14, poles: 4 }
        },
        batch: {
            fn: 'gaussian_batch',
            config: {
                small: {
                    period_range: [10, 20, 5],    // 3 values: 10, 15, 20
                    poles_range: [2, 4, 1]        // 3 values: 2, 3, 4
                },
                medium: {
                    period_range: [10, 50, 5],    // 9 values
                    poles_range: [1, 4, 1]        // 4 values  
                }
            },
            fastFn: 'gaussian_batch_into'
        }
    },
    hwma: {
        name: 'HWMA',
        // Safe API
        safe: {
            fn: 'hwma_js',
            params: { na: 0.2, nb: 0.1, nc: 0.1 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'hwma_alloc',
            freeFn: 'hwma_free',
            computeFn: 'hwma_into',
            params: { na: 0.2, nb: 0.1, nc: 0.1 }
        },
        // Batch API
        batch: {
            fn: 'hwma_batch',
            config: {
                small: {
                    na_range: [0.1, 0.3, 0.1],      // 3 values
                    nb_range: [0.1, 0.2, 0.1],      // 2 values
                    nc_range: [0.1, 0.2, 0.1]       // 2 values
                    // Total: 12 combinations
                },
                medium: {
                    na_range: [0.1, 0.5, 0.1],      // 5 values
                    nb_range: [0.1, 0.3, 0.1],      // 3 values
                    nc_range: [0.1, 0.3, 0.1]       // 3 values
                    // Total: 45 combinations
                }
            },
            // Fast batch API
            fastFn: 'hwma_batch_into'
        }
    },
    mwdx: {
        name: 'MWDX',
        // Safe API
        safe: {
            fn: 'mwdx_js',
            params: { factor: 0.2 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'mwdx_alloc',
            freeFn: 'mwdx_free',
            computeFn: 'mwdx_into',
            params: { factor: 0.2 }
        },
        // Batch API
        batch: {
            fn: 'mwdx_batch',  // Updated to use new structured API
            config: {
                small: {
                    factor_range: [0.1, 0.3, 0.1]      // 3 values: 0.1, 0.2, 0.3
                },
                medium: {
                    factor_range: [0.1, 0.5, 0.1]      // 5 values: 0.1, 0.2, 0.3, 0.4, 0.5
                }
            },
            // Fast batch API
            fastFn: 'mwdx_batch_into'
        }
    },
    srwma: {
        name: 'SRWMA',
        // Safe API
        safe: {
            fn: 'srwma_js',
            params: { period: 14 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'srwma_alloc',
            freeFn: 'srwma_free',
            computeFn: 'srwma_into',
            params: { period: 14 }
        },
        // Batch API
        batch: {
            fn: 'srwma_batch',
            config: {
                small: {
                    period_range: [10, 20, 5]      // 3 values
                },
                medium: {
                    period_range: [10, 50, 5]      // 9 values
                }
            },
            // Fast batch API
            fastFn: 'srwma_batch_into'
        }
    },
    linreg: {
        name: 'LinReg',
        // Safe API
        safe: {
            fn: 'linreg_js',
            params: { period: 14 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'linreg_alloc',
            freeFn: 'linreg_free',
            computeFn: 'linreg_into',
            params: { period: 14 }
        },
        // Batch API
        batch: {
            fn: 'linreg_batch',
            config: {
                small: {
                    period_range: [10, 20, 5]      // 3 values
                },
                medium: {
                    period_range: [10, 50, 5]      // 9 values
                }
            },
            // Fast batch API
            fastFn: 'linreg_batch_into'
        }
    },
    sinwma: {
        name: 'SINWMA',
        // Safe API
        safe: {
            fn: 'sinwma_js',
            params: { period: 14 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'sinwma_alloc',
            freeFn: 'sinwma_free',
            computeFn: 'sinwma_into',
            params: { period: 14 }
        },
        // Batch API
        batch: {
            fn: 'sinwma_batch',
            config: {
                small: {
                    period_range: [10, 20, 5]      // 3 values: 10, 15, 20
                },
                medium: {
                    period_range: [10, 50, 5]      // 9 values
                }
            }
            // No fast batch API yet for sinwma
        }
    },
    tilson: {
        name: 'Tilson T3',
        safe: {
            fn: 'tilson_js',
            params: { period: 14, volume_factor: 0.7 }
        },
        fast: {
            allocFn: 'tilson_alloc',
            freeFn: 'tilson_free',
            computeFn: 'tilson_into',
            params: { period: 14, volume_factor: 0.7 }
        },
        batch: {
            fn: 'tilson_batch',
            config: {
                small: {
                    period_range: [5, 15, 5],         // 3 values: 5, 10, 15
                    volume_factor_range: [0.0, 0.7, 0.35]  // 3 values: 0.0, 0.35, 0.7
                },
                medium: {
                    period_range: [5, 25, 5],         // 5 values: 5, 10, 15, 20, 25
                    volume_factor_range: [0.0, 0.8, 0.2]   // 5 values: 0.0, 0.2, 0.4, 0.6, 0.8
                }
            }
        }
    },
    trendflex: {
        name: 'TrendFlex',
        // Safe API
        safe: {
            fn: 'trendflex_js',
            params: { period: 20 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'trendflex_alloc',
            freeFn: 'trendflex_free',
            computeFn: 'trendflex_into',
            params: { period: 20 }
        },
        // Batch API
        batch: {
            fn: 'trendflex_batch',
            config: {
                small: {
                    period_range: [10, 30, 10]      // 3 values
                },
                medium: {
                    period_range: [10, 40, 2]       // 16 values
                }
            },
            // Fast batch API
            fastFn: 'trendflex_batch_into'
        }
    },
    ttm_trend: {
        name: 'TTM Trend',
        // Safe API
        safe: {
            fn: 'ttm_trend_js',
            params: { period: 5 }
        },
        needsMultipleInputs: true,  // Requires source and close arrays
        outputIsU8: true,           // Returns Uint8Array instead of Float64Array
        // Fast/Unsafe API
        fast: {
            allocFn: 'ttm_trend_alloc',
            allocU8Fn: 'ttm_trend_alloc_u8',  // Special allocator for u8 output
            freeFn: 'ttm_trend_free',
            freeU8Fn: 'ttm_trend_free_u8',    // Special free for u8
            computeFn: 'ttm_trend_into',
            params: { period: 5 },
            needsMultipleInputs: true,
            outputIsU8: true
        },
        // Batch API
        batch: {
            fn: 'ttm_trend_batch',
            config: {
                small: {
                    period_range: [5, 15, 5]       // 3 values
                },
                medium: {
                    period_range: [5, 20, 1]       // 16 values
                }
            },
            outputIsU8: true  // Batch also returns u8 values
        }
    },
    alligator: {
        name: 'Alligator',
        // Safe API
        safe: {
            fn: 'alligator_js',
            params: { 
                jaw_period: 13, 
                jaw_offset: 8, 
                teeth_period: 8, 
                teeth_offset: 5, 
                lips_period: 5, 
                lips_offset: 3 
            }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'alligator_alloc',
            freeFn: 'alligator_free',
            computeFn: 'alligator_into',
            params: { 
                jaw_period: 13, 
                jaw_offset: 8, 
                teeth_period: 8, 
                teeth_offset: 5, 
                lips_period: 5, 
                lips_offset: 3 
            },
            // Alligator has multiple outputs
            outputCount: 3
        },
        // Batch API
        batch: {
            fn: 'alligator_batch',
            config: {
                small: {
                    jaw_period_range: [13, 13, 0],
                    jaw_offset_range: [8, 8, 0],
                    teeth_period_range: [8, 8, 0],
                    teeth_offset_range: [5, 5, 0],
                    lips_period_range: [5, 5, 0],
                    lips_offset_range: [3, 3, 0]
                    // Total: 1 combination (default params)
                },
                medium: {
                    jaw_period_range: [10, 20, 5],      // 3 values
                    jaw_offset_range: [6, 10, 2],       // 3 values
                    teeth_period_range: [6, 10, 2],     // 3 values
                    teeth_offset_range: [4, 6, 1],      // 3 values
                    lips_period_range: [4, 6, 1],       // 3 values
                    lips_offset_range: [2, 4, 1]        // 3 values
                    // Total: 729 combinations (too many, but shows capability)
                }
            }
        }
    },
    correlation_cycle: {
        name: 'Correlation Cycle',
        // Safe API
        safe: {
            fn: 'correlation_cycle_js',
            params: { period: 20, threshold: 9.0 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'correlation_cycle_alloc',
            freeFn: 'correlation_cycle_free',
            computeFn: 'correlation_cycle_into',
            params: { period: 20, threshold: 9.0 },
            outputCount: 4  // 4 outputs: real, imag, angle, state
        },
        // Batch API
        batch: {
            fn: 'correlation_cycle_batch_js',
            config: {
                small: {
                    period_range: [15, 25, 5],          // 3 values
                    threshold_range: [8.0, 10.0, 1.0]   // 3 values = 9 combinations
                },
                medium: {
                    period_range: [10, 30, 5],          // 5 values
                    threshold_range: [7.0, 11.0, 1.0]   // 5 values = 25 combinations
                }
            }
        }
    },
    vwma: {
        name: 'VWMA (Volume Weighted Moving Average)',
        // Safe API
        safe: {
            fn: 'vwma_js',
            params: { period: 20 },
            inputs: ['prices', 'volumes']  // Special: requires both price and volume
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'vwma_alloc',
            freeFn: 'vwma_free',
            computeFn: 'vwma_into',
            params: { period: 20 },
            inputs: ['prices', 'volumes']  // Special: requires both price and volume
        },
        // Batch API
        batch: {
            fn: 'vwma_batch',
            config: {
                small: {
                    period_range: [10, 30, 5]      // 5 values: 10, 15, 20, 25, 30
                },
                medium: {
                    period_range: [5, 50, 5]       // 10 values: 5, 10, 15, ..., 50
                }
            },
            // Fast batch API
            fastFn: 'vwma_batch_into',
            inputs: ['prices', 'volumes']  // Special: requires both price and volume
        }
    },
    ad: {
        name: 'AD (Accumulation/Distribution)',
        // Safe API
        safe: {
            fn: 'ad_js',
            params: {},  // AD has no parameters
            inputs: ['high', 'low', 'close', 'volume']  // Requires OHLCV data
        },
        // Fast/Unsafe API (to be implemented)
        fast: {
            allocFn: 'ad_alloc',
            freeFn: 'ad_free',
            computeFn: 'ad_into',
            params: {},  // AD has no parameters
            inputs: ['high', 'low', 'close', 'volume']  // Requires OHLCV data
        },
        // Batch API
        batch: {
            fn: 'ad_batch_js',
            config: {
                small: {
                    // AD has no parameters, so batch is just processing multiple securities
                },
                medium: {
                    // AD has no parameters, so batch is just processing multiple securities
                }
            },
            inputs: ['high', 'low', 'close', 'volume']  // Requires OHLCV data
        }
    },
    vwap: {
        name: 'VWAP',
        // Safe API
        safe: {
            fn: 'vwap_js',
            params: { anchor: '1d', kernel: null },  // anchor and kernel as separate params
            needsVwapInputs: true  // Special flag for VWAP's unique inputs
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'vwap_alloc',
            freeFn: 'vwap_free',
            computeFn: 'vwap_into',
            params: { anchor: '1d' },
            needsVwapInputs: true
        },
        // Batch API
        batch: {
            fn: 'vwap_batch',
            config: {
                small: {
                    anchor_range: ['1m', '15m', 14]    // 1m, 15m (2 values)
                },
                medium: {
                    anchor_range: ['1m', '1h', 59]     // 1m, 1h (2 values)
                }
            },
            needsVwapInputs: true
        }
    },
    zlema: {
        name: 'ZLEMA',
        // Safe API
        safe: {
            fn: 'zlema_js',
            params: { period: 14 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'zlema_alloc',
            freeFn: 'zlema_free',
            computeFn: 'zlema_into',
            params: { period: 14 }
        },
        // Batch API
        batch: {
            fn: 'zlema_batch',
            fastFn: 'zlema_batch_into',
            config: {
                small: {
                    period_range: [10, 20, 2]      // 6 values
                },
                medium: {
                    period_range: [5, 25, 5]       // 5 values
                }
            }
        }
    },
    adx: {
        name: 'ADX',
        // Safe API - requires high, low, close
        safe: {
            fn: 'adx_js',
            params: { period: 14 }
        },
        needsMultipleInputs: true, // ADX needs high, low, close arrays
        // Fast/Unsafe API
        fast: {
            allocFn: 'adx_alloc',
            freeFn: 'adx_free',
            computeFn: 'adx_into',
            params: { period: 14 },
            needsMultipleInputs: true
        },
        // Batch API
        batch: {
            fn: 'adx_batch',
            fastFn: 'adx_batch_into',
            config: {
                small: {
                    period_range: [10, 18, 4]      // 3 values: 10, 14, 18
                },
                medium: {
                    period_range: [10, 30, 5]      // 5 values: 10, 15, 20, 25, 30
                }
            }
        }
    },
    cksp: {
        name: 'CKSP',
        // Safe API
        safe: {
            fn: 'cksp_js',
            params: { p: 10, x: 1.0, q: 9 },
            needsMultipleInputs: true,
            outputSize: 2  // Returns 2x input length (long + short)
        },
        // Fast/Unsafe API  
        fast: {
            allocFn: 'cksp_alloc',
            freeFn: 'cksp_free',
            computeFn: 'cksp_into',
            params: { p: 10, x: 1.0, q: 9 },
            needsMultipleInputs: true,
            outputCount: 2  // Two separate output arrays
        },
        // Batch API
        batch: {
            fn: 'cksp_batch',
            config: {
                small: {
                    p_range: [5, 15, 5],        // 3 values
                    x_range: [0.5, 1.5, 0.5],   // 3 values  
                    q_range: [5, 10, 5]         // 2 values = 18 combinations
                },
                medium: {
                    p_range: [5, 25, 5],        // 5 values
                    x_range: [0.5, 2.0, 0.5],   // 4 values
                    q_range: [5, 15, 5]         // 3 values = 60 combinations
                }
            }
        }
    },
    emd: {
        name: 'EMD',
        needsMultipleInputs: true,  // Uses high, low, close, volume
        // Safe API
        safe: {
            fn: 'emd_js',
            params: { period: 20, delta: 0.5, fraction: 0.1 },
            needsMultipleInputs: true,
            resultType: 'EmdResult'  // Returns object with values, rows, cols
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'emd_alloc',
            freeFn: 'emd_free',
            computeFn: 'emd_into',
            params: { period: 20, delta: 0.5, fraction: 0.1 },
            needsMultipleInputs: true,
            tripleOutput: true  // Has three outputs (upperband, middleband, lowerband)
        },
        // Batch API
        batch: {
            fn: 'emd_batch',
            config: {
                small: {
                    period_range: [20, 22, 2],      // 2 values: 20, 22
                    delta_range: [0.5, 0.6, 0.1],   // 2 values: 0.5, 0.6
                    fraction_range: [0.1, 0.2, 0.1] // 2 values: 0.1, 0.2 = 8 combinations
                },
                medium: {
                    period_range: [15, 25, 5],      // 3 values: 15, 20, 25
                    delta_range: [0.3, 0.7, 0.2],   // 3 values: 0.3, 0.5, 0.7
                    fraction_range: [0.05, 0.2, 0.05] // 4 values: 0.05, 0.1, 0.15, 0.2 = 36 combinations
                }
            },
            // Fast batch API
            fastFn: 'emd_batch_into',
            tripleOutput: true,
            needsMultipleInputs: true
        }
    },
    gatorosc: {
        name: 'GatorOsc',
        // Safe API
        safe: {
            fn: 'gatorosc_js',
            params: { 
                jaws_length: 13,
                jaws_shift: 8,
                teeth_length: 8,
                teeth_shift: 5,
                lips_length: 5,
                lips_shift: 3
            },
            resultType: 'GatorOscJsOutput'  // Returns object with values, rows, cols
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'gatorosc_alloc',
            freeFn: 'gatorosc_free',
            computeFn: 'gatorosc_into',
            params: { 
                jaws_length: 13,
                jaws_shift: 8,
                teeth_length: 8,
                teeth_shift: 5,
                lips_length: 5,
                lips_shift: 3
            },
            quadOutput: true  // Has four outputs (upper, lower, upper_change, lower_change)
        },
        // Batch API
        batch: {
            fn: 'gatorosc_batch',
            config: {
                small: {
                    jaws_length_range: [13, 13, 0],
                    jaws_shift_range: [8, 8, 0],
                    teeth_length_range: [8, 8, 0],
                    teeth_shift_range: [5, 5, 0],
                    lips_length_range: [5, 5, 0],
                    lips_shift_range: [3, 3, 0]  // 1 combination
                },
                medium: {
                    jaws_length_range: [10, 15, 5],   // 2 values
                    jaws_shift_range: [6, 10, 2],     // 3 values
                    teeth_length_range: [6, 10, 2],   // 3 values
                    teeth_shift_range: [3, 6, 3],     // 2 values
                    lips_length_range: [3, 6, 3],     // 2 values
                    lips_shift_range: [2, 4, 2]       // 2 values = 144 combinations
                }
            }
        }
    },
    kurtosis: {
        name: 'Kurtosis',
        // Safe API
        safe: {
            fn: 'kurtosis_js',
            params: { period: 5 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'kurtosis_alloc',
            freeFn: 'kurtosis_free',
            computeFn: 'kurtosis_into',
            params: { period: 5 }
        },
        // Batch API
        batch: {
            fn: 'kurtosis_batch',
            fastFn: 'kurtosis_batch_into',
            config: {
                small: {
                    period_range: [5, 15, 5]       // 3 values: 5, 10, 15
                },
                medium: {
                    period_range: [5, 50, 5]       // 10 values: 5, 10, 15, ..., 50
                }
            }
        }
    },
    mab: {
        name: 'MAB (Moving Average Bands)',
        // Safe API
        safe: {
            fn: 'mab_js',
            params: { fast_period: 10, slow_period: 50, devup: 1.0, devdn: 1.0, fast_ma_type: 'sma', slow_ma_type: 'sma' },
            outputLength: 3  // Returns flattened array with 3 bands
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'mab_alloc',
            freeFn: 'mab_free',
            computeFn: 'mab_into',
            params: { fast_period: 10, slow_period: 50, devup: 1.0, devdn: 1.0, fast_ma_type: 'sma', slow_ma_type: 'sma' },
            tripleOutput: true  // Has three outputs (upper, middle, lower)
        },
        // Batch API
        batch: {
            fn: 'mab_batch',
            fastFn: 'mab_batch_into',
            config: {
                small: {
                    fast_period_range: [10, 15, 5],     // 2 values: 10, 15
                    slow_period_range: [50, 50, 0],     // 1 value: 50
                    devup_range: [1.0, 2.0, 0.5],       // 3 values: 1.0, 1.5, 2.0
                    devdn_range: [1.0, 1.0, 0.0],       // 1 value: 1.0
                    fast_ma_type: 'sma',
                    slow_ma_type: 'sma'
                    // Total: 2 * 1 * 3 * 1 = 6 combinations
                },
                medium: {
                    fast_period_range: [10, 20, 5],     // 3 values: 10, 15, 20
                    slow_period_range: [40, 60, 10],    // 3 values: 40, 50, 60
                    devup_range: [0.5, 2.5, 0.5],       // 5 values: 0.5, 1.0, 1.5, 2.0, 2.5
                    devdn_range: [0.5, 1.5, 0.5],       // 3 values: 0.5, 1.0, 1.5
                    fast_ma_type: 'sma',
                    slow_ma_type: 'sma'
                    // Total: 3 * 3 * 5 * 3 = 135 combinations
                }
            },
            tripleOutput: true
        }
    },
    msw: {
        name: 'MSW',
        // Safe API
        safe: {
            fn: 'msw_js',
            params: { period: 5 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'msw_alloc',
            freeFn: 'msw_free',
            computeFn: 'msw_into',
            params: { period: 5 },
            dualOutput: true  // Has two outputs (sine and lead)
        },
        // Batch API
        batch: {
            fn: 'msw_batch',
            config: {
                small: {
                    period_range: [5, 15, 5]       // 3 values: 5, 10, 15
                },
                medium: {
                    period_range: [5, 30, 5]       // 6 values: 5, 10, 15, 20, 25, 30
                }
            },
            // Fast batch API
            fastFn: 'msw_batch_into',
            dualOutput: true
        }
    },
    pma: {
        name: 'PMA',
        // Safe API
        safe: {
            fn: 'pma_js',
            params: {} // PMA has no parameters
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'pma_alloc',
            freeFn: 'pma_free',
            computeFn: 'pma_into',
            params: {},
            dualOutput: true  // Has two outputs (predict and trigger)
        },
        // Batch API
        batch: {
            fn: 'pma_batch',
            config: {
                small: {
                    // PMA has no parameters, but we need a dummy config
                    dummy: 0
                },
                medium: {
                    // PMA has no parameters, but we need a dummy config
                    dummy: 0
                }
            },
            // Fast batch API
            fastFn: 'pma_batch_into',
            dualOutput: true
        }
    },
    sar: {
        name: 'SAR',
        needsMultipleInputs: true,  // Uses high, low
        // Safe API
        safe: {
            fn: 'sar_js',
            params: { acceleration: 0.02, maximum: 0.2 },
            needsMultipleInputs: true
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'sar_alloc',
            freeFn: 'sar_free',
            computeFn: 'sar_into',
            params: { acceleration: 0.02, maximum: 0.2 },
            needsMultipleInputs: true
        },
        // Batch API
        batch: {
            fn: 'sar_batch',
            config: {
                small: {
                    acceleration_range: [0.01, 0.03, 0.01],  // 3 values
                    maximum_range: [0.1, 0.3, 0.1]           // 3 values = 9 combinations
                },
                medium: {
                    acceleration_range: [0.01, 0.05, 0.01],  // 5 values
                    maximum_range: [0.1, 0.5, 0.1]           // 5 values = 25 combinations
                }
            }
        }
    },
    supertrend: {
        name: 'SuperTrend',
        // Safe API
        safe: {
            fn: 'supertrend_js',
            params: { period: 10, factor: 3.0 },
            needsMultipleInputs: true,
            dualOutput: true  // Has two outputs (trend and changed)
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'supertrend_alloc',
            freeFn: 'supertrend_free',
            computeFn: 'supertrend_into',
            params: { period: 10, factor: 3.0 },
            needsMultipleInputs: true,
            dualOutput: true  // Has two outputs (trend and changed)
        },
        // Batch API
        batch: {
            fn: 'supertrend_batch',
            config: {
                small: {
                    period_range: [8, 12, 2],      // 3 values
                    factor_range: [2.0, 4.0, 1.0]  // 3 values = 9 combinations
                },
                medium: {
                    period_range: [5, 15, 2],      // 6 values
                    factor_range: [1.0, 5.0, 1.0]  // 5 values = 30 combinations
                }
            },
            needsMultipleInputs: true
        }
    },
    ultosc: {
        name: 'ULTOSC',
        // Safe API
        safe: {
            fn: 'ultosc_js',
            params: { timeperiod1: 7, timeperiod2: 14, timeperiod3: 28 },
            needsMultipleInputs: true  // Requires high, low, close
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'ultosc_alloc',
            freeFn: 'ultosc_free',
            computeFn: 'ultosc_into',
            params: { timeperiod1: 7, timeperiod2: 14, timeperiod3: 28 },
            needsMultipleInputs: true
        },
        // Batch API
        batch: {
            fn: 'ultosc_batch',
            config: {
                small: {
                    timeperiod1_range: [5, 9, 2],    // 3 values
                    timeperiod2_range: [12, 16, 2],  // 3 values
                    timeperiod3_range: [26, 30, 2]   // 3 values = 27 combinations
                },
                medium: {
                    timeperiod1_range: [5, 11, 2],   // 4 values
                    timeperiod2_range: [10, 18, 2],  // 5 values
                    timeperiod3_range: [24, 32, 2]   // 5 values = 100 combinations
                }
            },
            needsMultipleInputs: true
        }
    },
    voss: {
        name: 'VOSS',
        // Safe API
        safe: {
            fn: 'voss_js',
            params: { period: 20, predict: 3, bandwidth: 0.25 },
            outputLength: 2  // Returns flattened array with 2 outputs (voss, filt)
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'voss_alloc',
            freeFn: 'voss_free',
            computeFn: 'voss_into',
            params: { period: 20, predict: 3, bandwidth: 0.25 },
            dualOutput: true  // Has two outputs (voss and filt)
        },
        // Batch API
        batch: {
            fn: 'voss_batch',
            config: {
                small: {
                    period_range: [10, 20, 5],      // 3 values
                    predict_range: [2, 4, 1],       // 3 values
                    bandwidth_range: [0.2, 0.3, 0.1] // 2 values = 18 combinations
                },
                medium: {
                    period_range: [10, 30, 5],      // 5 values
                    predict_range: [2, 5, 1],       // 4 values
                    bandwidth_range: [0.1, 0.4, 0.1] // 4 values = 80 combinations
                }
            },
            fastFn: 'voss_batch_into'
        }
    },
    wavetrend: {
        name: 'WaveTrend',
        // Safe API
        safe: {
            fn: 'wavetrend_js',
            params: { channel_length: 9, average_length: 12, ma_length: 3, factor: 0.015 },
            tripleOutput: true  // Has three outputs (wt1, wt2, wt_diff)
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'wavetrend_alloc',
            freeFn: 'wavetrend_free',
            computeFn: 'wavetrend_into',
            params: { channel_length: 9, average_length: 12, ma_length: 3, factor: 0.015 },
            tripleOutput: true  // Has three outputs
        },
        // Batch API
        batch: {
            fn: 'wavetrend_batch',
            config: {
                small: {
                    channel_length_range: [9, 11, 2],      // 2 values
                    average_length_range: [12, 14, 2],     // 2 values
                    ma_length_range: [3, 3, 0],            // 1 value
                    factor_range: [0.015, 0.020, 0.005]    // 2 values = 8 combinations
                },
                medium: {
                    channel_length_range: [7, 13, 2],      // 4 values
                    average_length_range: [10, 16, 2],     // 4 values
                    ma_length_range: [3, 5, 1],            // 3 values
                    factor_range: [0.010, 0.025, 0.005]    // 4 values = 192 combinations
                }
            }
        }
    },
    apo: {
        name: 'APO',
        // Safe API
        safe: {
            fn: 'apo_js',
            params: { short_period: 10, long_period: 20 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'apo_alloc',
            freeFn: 'apo_free',
            computeFn: 'apo_into',
            params: { short_period: 10, long_period: 20 }
        },
        // Batch API
        batch: {
            fn: 'apo_batch',
            config: {
                small: {
                    short_period_range: [5, 15, 5],    // 3 values: 5, 10, 15
                    long_period_range: [20, 30, 10]    // 2 values: 20, 30 = 6 combinations
                },
                medium: {
                    short_period_range: [5, 15, 2],    // 6 values: 5, 7, 9, 11, 13, 15
                    long_period_range: [20, 40, 5]     // 5 values: 20, 25, 30, 35, 40 = 30 combinations
                }
            }
        }
    },
    chop: {
        name: 'CHOP',
        needsMultipleInputs: true,  // Uses high, low, close
        // Safe API
        safe: {
            fn: 'chop_js',
            params: { period: 14, scalar: 100.0, drift: 1 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'chop_alloc',
            freeFn: 'chop_free',
            computeFn: 'chop_into',
            params: { period: 14, scalar: 100.0, drift: 1 },
            needsMultipleInputs: true
        },
        // Batch API
        batch: {
            fn: 'chop_batch',
            fastFn: 'chop_batch_into',
            config: {
                small: {
                    period_range: [10, 20, 5],      // 3 values: 10, 15, 20
                    scalar_range: [50.0, 100.0, 50.0], // 2 values: 50, 100
                    drift_range: [1, 2, 1]          // 2 values: 1, 2
                    // Total: 12 combinations
                },
                medium: {
                    period_range: [10, 30, 5],      // 5 values: 10, 15, 20, 25, 30
                    scalar_range: [50.0, 150.0, 25.0], // 5 values: 50, 75, 100, 125, 150
                    drift_range: [1, 3, 1]          // 3 values: 1, 2, 3
                    // Total: 75 combinations
                }
            },
            needsMultipleInputs: true
        }
    },
    cvi: {
        name: 'CVI',
        needsMultipleInputs: true,  // Uses high, low
        // Safe API
        safe: {
            fn: 'cvi_js',
            params: { period: 10 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'cvi_alloc',
            freeFn: 'cvi_free',
            computeFn: 'cvi_into',
            params: { period: 10 },
            needsMultipleInputs: true
        },
        // Batch API
        batch: {
            fn: 'cvi_batch',
            fastFn: 'cvi_batch_into',
            config: {
                small: {
                    period_range: [5, 15, 5]        // 3 values: 5, 10, 15
                },
                medium: {
                    period_range: [5, 25, 5]        // 5 values: 5, 10, 15, 20, 25
                }
            },
            needsMultipleInputs: true
        }
    },
    di: {
        name: 'DI',
        needsMultipleInputs: true,  // Uses high, low, close
        // Safe API
        safe: {
            fn: 'di_js',
            params: { period: 14 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'di_alloc',
            freeFn: 'di_free',
            computeFn: 'di_into',
            params: { period: 14 },
            needsMultipleInputs: true,
            dualOutput: true  // Has two outputs (plus and minus)
        },
        // Batch API
        batch: {
            fn: 'di_batch',
            fastFn: 'di_batch_into',
            config: {
                small: {
                    period_range: [10, 20, 5]      // 3 values: 10, 15, 20
                },
                medium: {
                    period_range: [10, 30, 5]      // 5 values: 10, 15, 20, 25, 30
                }
            },
            needsMultipleInputs: true,
            dualOutput: true
        }
    },
    efi: {
        name: 'EFI',
        needsMultipleInputs: true,  // Uses close (price) and volume
        // Safe API
        safe: {
            fn: 'efi_js',
            params: { period: 13 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'efi_alloc',
            freeFn: 'efi_free',
            computeFn: 'efi_into',
            params: { period: 13 },
            needsMultipleInputs: true
        },
        // Batch API
        batch: {
            fn: 'efi_batch',
            fastFn: 'efi_batch_into',
            config: {
                small: {
                    period_range: [10, 20, 5]      // 3 values: 10, 15, 20
                },
                medium: {
                    period_range: [10, 30, 5]      // 5 values: 10, 15, 20, 25, 30
                }
            },
            needsMultipleInputs: true
        }
    },
    kst: {
        name: 'KST',
        // Safe API
        safe: {
            fn: 'kst_js',
            params: { 
                sma_period1: 10, 
                sma_period2: 10, 
                sma_period3: 10, 
                sma_period4: 15,
                roc_period1: 10,
                roc_period2: 15,
                roc_period3: 20,
                roc_period4: 30,
                signal_period: 9
            },
            multiOutput: true  // Returns {line, signal}
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'kst_alloc',
            freeFn: 'kst_free',
            computeFn: 'kst_into',
            params: { 
                sma_period1: 10, 
                sma_period2: 10, 
                sma_period3: 10, 
                sma_period4: 15,
                roc_period1: 10,
                roc_period2: 15,
                roc_period3: 20,
                roc_period4: 30,
                signal_period: 9
            },
            multiOutput: true  // Has separate line and signal outputs
        },
        // Batch API
        batch: {
            fn: 'kst_batch',
            fastFn: 'kst_batch_into',
            config: {
                small: {
                    sma_period1_range: [8, 12, 2],      // 3 values
                    sma_period2_range: [8, 12, 2],      // 3 values
                    sma_period3_range: [8, 12, 2],      // 3 values
                    sma_period4_range: [12, 18, 3],     // 3 values
                    roc_period1_range: [8, 12, 2],      // 3 values
                    roc_period2_range: [12, 18, 3],     // 3 values
                    roc_period3_range: [18, 22, 2],     // 3 values
                    roc_period4_range: [25, 35, 5],     // 3 values
                    signal_period_range: [7, 11, 2]     // 3 values = 3^9 = too many!
                },
                medium: {
                    // More reasonable subset for batch
                    sma_period1_range: [10, 10, 1],     // 1 value (fixed)
                    sma_period2_range: [10, 10, 1],     // 1 value (fixed)
                    sma_period3_range: [10, 10, 1],     // 1 value (fixed)
                    sma_period4_range: [10, 20, 5],     // 3 values
                    roc_period1_range: [10, 10, 1],     // 1 value (fixed)
                    roc_period2_range: [10, 20, 5],     // 3 values
                    roc_period3_range: [15, 25, 5],     // 3 values
                    roc_period4_range: [25, 35, 5],     // 3 values
                    signal_period_range: [7, 13, 3]     // 3 values = 3^5 = 243 combinations
                }
            },
            multiOutput: true  // Returns {line, signal, combos, rows, cols}
        }
    },
    lrsi: {
        name: 'LRSI',
        needsMultipleInputs: true,  // Uses high/low
        // Safe API
        safe: {
            fn: 'lrsi_js',
            params: { alpha: 0.2 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'lrsi_alloc',
            freeFn: 'lrsi_free',
            computeFn: 'lrsi_into',
            params: { alpha: 0.2 },
            needsMultipleInputs: true
        },
        // Batch API
        batch: {
            fn: 'lrsi_batch',
            config: {
                small: {
                    alpha_range: [0.1, 0.3, 0.1]     // 3 values
                },
                medium: {
                    alpha_range: [0.1, 0.5, 0.05]    // 9 values
                }
            }
        }
    },
    pivot: {
        name: 'PIVOT',
        needsMultipleInputs: true,  // Uses high, low, close, open
        // Safe API
        safe: {
            fn: 'pivot_js',
            params: { mode: 3 },  // Camarilla mode by default
            // Pivot returns 9 outputs flattened as [r4..., r3..., r2..., r1..., pp..., s1..., s2..., s3..., s4...]
            multiOutput: 9
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'pivot_alloc',
            freeFn: 'pivot_free',
            computeFn: 'pivot_into',
            params: { mode: 3 },
            // Multiple output pointers for all 9 levels
            multiOutput: 9,
            needsMultipleInputs: true
        },
        // Batch API
        batch: {
            fn: 'pivot_batch',
            config: {
                small: {
                    mode_range: [0, 4, 1]  // 5 values: All 5 modes
                },
                medium: {
                    mode_range: [0, 4, 1]  // Same as small for pivot
                }
            },
            needsMultipleInputs: true
        }
    },
    safezonestop: {
        name: 'SafeZoneStop',
        needsMultipleInputs: true,  // Uses high, low
        // Safe API
        safe: {
            fn: 'safezonestop_js',
            params: { period: 22, mult: 2.5, max_lookback: 3, direction: 'long' }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'safezonestop_alloc',
            freeFn: 'safezonestop_free',
            computeFn: 'safezonestop_into',
            params: { period: 22, mult: 2.5, max_lookback: 3, direction: 'long' },
            needsMultipleInputs: true,
            needsSafeZoneStopInputs: true  // Special flag for SafeZoneStop's unique inputs
        },
        // Batch API
        batch: {
            fn: 'safezonestop_batch',
            config: {
                small: {
                    period_range: [14, 30, 8],       // 3 values
                    mult_range: [2.0, 3.0, 0.5],     // 3 values
                    max_lookback_range: [2, 4, 1],   // 3 values = 27 combinations
                    direction: 'long'
                },
                medium: {
                    period_range: [10, 50, 10],      // 5 values
                    mult_range: [2.0, 4.0, 0.5],     // 5 values
                    max_lookback_range: [2, 6, 2],   // 3 values = 75 combinations
                    direction: 'long'
                }
            }
        }
    },
    stochf: {
        name: 'StochF',
        needsMultipleInputs: true,  // Uses high, low, close
        // Safe API
        safe: {
            fn: 'stochf_js',
            params: { fastk_period: 5, fastd_period: 3, fastd_matype: 0 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'stochf_alloc',
            freeFn: 'stochf_free',
            computeFn: 'stochf_into',
            params: { fastk_period: 5, fastd_period: 3, fastd_matype: 0 },
            needsMultipleInputs: true,
            needsStochFInputs: true  // Special flag for StochF's three inputs and two outputs
        },
        // Batch API
        batch: {
            fn: 'stochf_batch',
            config: {
                small: {
                    fastk_range: [5, 14, 1],       // 10 values
                    fastd_range: [3, 5, 1],        // 3 values = 30 combinations
                    fastd_matype: 0
                },
                medium: {
                    fastk_range: [5, 50, 5],       // 10 values
                    fastd_range: [3, 10, 1],       // 8 values = 80 combinations
                    fastd_matype: 0
                }
            },
            // Fast batch API (optional)
            fastFn: 'stochf_batch_into'
        }
    },
    ui: {
        name: 'UI (Ulcer Index)',
        // Safe API
        safe: {
            fn: 'ui_js',
            params: { period: 14, scalar: 100.0 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'ui_alloc',
            freeFn: 'ui_free',
            computeFn: 'ui_into',
            params: { period: 14, scalar: 100.0 }
        },
        // Batch API
        batch: {
            fn: 'ui_batch',
            config: {
                small: {
                    period_range: [10, 20, 5],      // 3 values
                    scalar_range: [50.0, 150.0, 50.0] // 3 values = 9 combinations
                },
                medium: {
                    period_range: [5, 50, 5],       // 10 values
                    scalar_range: [50.0, 200.0, 50.0] // 4 values = 40 combinations
                }
            },
            // Fast batch API
            fastFn: 'ui_batch_into'
        }
    },
    wad: {
        name: 'WAD',
        needsMultipleInputs: true,  // Uses high, low, close
        // Safe API
        safe: {
            fn: 'wad_js',
            params: {}  // No parameters for WAD
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'wad_alloc',
            freeFn: 'wad_free',
            computeFn: 'wad_into',
            params: {},
            needsMultipleInputs: true
        },
        // Batch API
        batch: {
            fn: 'wad_batch',
            config: {
                // WAD has no parameters, so batch always returns 1 row
                small: {},
                medium: {}
            }
        }
    },
    bollinger_bands_width: {
        name: 'Bollinger Bands Width',
        // Safe API
        safe: {
            fn: 'bollinger_bands_width_js',
            params: { period: 20, devup: 2.0, devdn: 2.0, matype: "sma", devtype: 0 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'bollinger_bands_width_alloc',
            freeFn: 'bollinger_bands_width_free',
            computeFn: 'bollinger_bands_width_into',
            params: { period: 20, devup: 2.0, devdn: 2.0, matype: "sma", devtype: 0 }
        },
        // Batch API
        batch: {
            fn: 'bollinger_bands_width_batch',
            config: {
                small: {
                    period_range: [10, 30, 10],     // 3 values: 10, 20, 30
                    devup_range: [1.5, 2.5, 0.5],   // 3 values: 1.5, 2.0, 2.5
                    devdn_range: [2.0, 2.0, 0]      // 1 value: 2.0 = 9 combinations
                },
                medium: {
                    period_range: [10, 40, 10],     // 4 values: 10, 20, 30, 40
                    devup_range: [1.0, 3.0, 0.5],   // 5 values: 1.0, 1.5, 2.0, 2.5, 3.0
                    devdn_range: [1.5, 2.5, 0.5]    // 3 values: 1.5, 2.0, 2.5 = 60 combinations
                }
            }
        }
    },
    dti: {
        name: 'DTI',
        needsMultipleInputs: true,  // Uses high, low
        // Safe API
        safe: {
            fn: 'dti_js',
            params: { r: 14, s: 10, u: 5 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'dti_alloc',
            freeFn: 'dti_free',
            computeFn: 'dti_into',
            params: { r: 14, s: 10, u: 5 },
            needsMultipleInputs: true
        },
        // Batch API
        batch: {
            fn: 'dti_batch',
            fastFn: 'dti_batch_into',
            config: {
                small: {
                    r_range: [10, 20, 5],      // 3 values
                    s_range: [8, 12, 2],       // 3 values
                    u_range: [4, 6, 1]         // 3 values = 27 combinations
                },
                medium: {
                    r_range: [10, 30, 5],      // 5 values
                    s_range: [5, 15, 2],       // 6 values
                    u_range: [3, 7, 1]         // 5 values = 150 combinations
                }
            },
            needsMultipleInputs: true
        }
    },
    dx: {
        name: 'DX',
        needsMultipleInputs: true,  // Uses high, low, close
        // Safe API
        safe: {
            fn: 'dx_js',
            params: { period: 14 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'dx_alloc',
            freeFn: 'dx_free',
            computeFn: 'dx_into',
            params: { period: 14 },
            needsMultipleInputs: true
        },
        // Batch API
        batch: {
            fn: 'dx_batch',
            fastFn: 'dx_batch_into',
            config: {
                small: {
                    period_range: [10, 20, 5]      // 3 values: 10, 15, 20
                },
                medium: {
                    period_range: [10, 30, 5]      // 5 values: 10, 15, 20, 25, 30
                }
            },
            needsMultipleInputs: true
        }
    },
    roc: {
        name: 'ROC',
        // Safe API
        safe: {
            fn: 'roc_js',
            params: { period: 10 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'roc_alloc',
            freeFn: 'roc_free',
            computeFn: 'roc_into',
            params: { period: 10 }
        },
        // Batch API
        batch: {
            fn: 'roc_batch',
            config: {
                small: {
                    period_range: [5, 15, 5]  // 3 values: 5, 10, 15
                },
                medium: {
                    period_range: [5, 25, 5]  // 5 values: 5, 10, 15, 20, 25
                }
            }
        }
    },
    rvi: {
        name: 'RVI (Relative Vigor Index)',
        // Safe API
        safe: {
            fn: 'rvi_js',
            params: { period: 10, ma_len: 14, matype: 1, devtype: 0 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'rvi_alloc',
            freeFn: 'rvi_free',
            computeFn: 'rvi_into',
            params: { period: 10, ma_len: 14, matype: 1, devtype: 0 }
        },
        // Batch API
        batch: {
            fn: 'rvi_batch',
            config: {
                small: {
                    period_range: [10, 20, 5],    // 3 values: 10, 15, 20
                    ma_len_range: [10, 14, 4],    // 2 values: 10, 14
                    matype_range: [0, 1, 1],      // 2 values: 0, 1
                    devtype_range: [0, 0, 0]      // 1 value: 0
                    // Total: 3 * 2 * 2 * 1 = 12 combinations
                },
                medium: {
                    period_range: [10, 30, 5],    // 5 values: 10, 15, 20, 25, 30
                    ma_len_range: [10, 20, 5],    // 3 values: 10, 15, 20
                    matype_range: [0, 1, 1],      // 2 values: 0, 1
                    devtype_range: [0, 2, 1]      // 3 values: 0, 1, 2
                    // Total: 5 * 3 * 2 * 3 = 90 combinations
                }
            }
        }
    },
    stddev: {
        name: 'StdDev',
        // Safe API
        safe: {
            fn: 'stddev_js',
            params: { period: 5, nbdev: 1.0 }
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'stddev_alloc',
            freeFn: 'stddev_free',
            computeFn: 'stddev_into',
            params: { period: 5, nbdev: 1.0 }
        },
        // Batch API
        batch: {
            fn: 'stddev_batch',
            fastFn: 'stddev_batch_into',
            config: {
                small: {
                    period_range: [5, 15, 5],      // 3 values: 5, 10, 15
                    nbdev_range: [1.0, 2.0, 0.5]   // 3 values: 1.0, 1.5, 2.0
                    // Total: 3 * 3 = 9 combinations
                },
                medium: {
                    period_range: [5, 25, 5],      // 5 values: 5, 10, 15, 20, 25
                    nbdev_range: [0.5, 2.5, 0.5]   // 5 values: 0.5, 1.0, 1.5, 2.0, 2.5
                    // Total: 5 * 5 = 25 combinations
                }
            }
        }
    },
    vwmacd: {
        name: 'VWMACD (Volume Weighted MACD)',
        // Safe API
        safe: {
            fn: 'vwmacd_js',
            params: { 
                fast_period: 12, 
                slow_period: 26, 
                signal_period: 9,
                fast_ma_type: "sma",
                slow_ma_type: "sma",
                signal_ma_type: "ema"
            },
            inputs: ['close', 'volume']  // Requires both close and volume
        },
        // Fast/Unsafe API
        fast: {
            allocFn: 'vwmacd_alloc',
            freeFn: 'vwmacd_free',
            computeFn: 'vwmacd_into',
            params: { 
                fast_period: 12, 
                slow_period: 26, 
                signal_period: 9,
                fast_ma_type: "sma",
                slow_ma_type: "sma",
                signal_ma_type: "ema"
            },
            inputs: ['close', 'volume'],  // Requires both close and volume
            outputCount: 3  // 3 outputs: macd, signal, histogram
        },
        // Batch API
        batch: {
            fn: 'vwmacd_batch',
            config: {
                small: {
                    fast_range: [10, 14, 2],    // 3 values: 10, 12, 14
                    slow_range: [24, 28, 2],    // 3 values: 24, 26, 28
                    signal_range: [8, 10, 1],   // 3 values: 8, 9, 10
                    fast_ma_type: "sma",
                    slow_ma_type: "sma",
                    signal_ma_type: "ema"
                    // Total: 3 * 3 * 3 = 27 combinations
                },
                medium: {
                    fast_range: [8, 16, 2],     // 5 values: 8, 10, 12, 14, 16
                    slow_range: [20, 30, 2],    // 6 values: 20, 22, 24, 26, 28, 30
                    signal_range: [6, 12, 2],   // 4 values: 6, 8, 10, 12
                    fast_ma_type: "sma",
                    slow_ma_type: "sma", 
                    signal_ma_type: "ema"
                    // Total: 5 * 6 * 4 = 120 combinations
                }
            },
            inputs: ['close', 'volume']  // Batch also requires both inputs
        }
    }
};

class WasmIndicatorBenchmark {
    constructor() {
        this.wasm = null;
        this.data = {};
        this.results = {};
    }

    async initialize() {
        // Load WASM module
        console.log('Loading WASM module...');
        try {
            const { createRequire } = await import('module');
            const require = createRequire(import.meta.url);
            const wasmPath = join(__dirname, '../pkg/my_project.js');
            this.wasm = require(wasmPath);
            console.log('WASM module loaded successfully');
        } catch (error) {
            console.error('Failed to load WASM module:', error);
            console.error('Run "wasm-pack build --features wasm --target nodejs" first');
            process.exit(1);
        }

        // Load test data
        this.loadData();
    }

    loadData() {
        console.log('Loading test data...');
        
        const csvPath = join(__dirname, '../src/data/1MillionCandles.csv');
        const content = readFileSync(csvPath, 'utf8');
        const lines = content.trim().split('\n');
        
        // Skip header
        lines.shift();
        
        // Parse OHLC data
        const opens = [];
        const highs = [];
        const lows = [];
        // Note: CSV format is timestamp,open,close,high,low,volume
        // So close is at index 2, not 4!
        const closes = [];
        const volumes = [];
        const timestamps = [];
        
        for (const line of lines) {
            const parts = line.split(',');
            if (parts.length >= 6) {
                opens.push(parseFloat(parts[1]));
                closes.push(parseFloat(parts[2]));
                highs.push(parseFloat(parts[3]));
                lows.push(parseFloat(parts[4]));
                volumes.push(parseFloat(parts[5])); // Volume is column 5
            }
        }
        
        // Create different size datasets with single close array and OHLC data
        this.data['10k'] = new Float64Array(closes.slice(0, 10_000));
        this.data['100k'] = new Float64Array(closes.slice(0, 100_000));
        this.data['1M'] = new Float64Array(closes);
        
        // Also store OHLC data for indicators that need it
        this.ohlcData = {
            '10k': {
                open: new Float64Array(opens.slice(0, 10_000)),
                high: new Float64Array(highs.slice(0, 10_000)),
                low: new Float64Array(lows.slice(0, 10_000)),
                close: new Float64Array(closes.slice(0, 10_000)),
                volume: new Float64Array(volumes.slice(0, 10_000))
            },
            '100k': {
                open: new Float64Array(opens.slice(0, 100_000)),
                high: new Float64Array(highs.slice(0, 100_000)),
                low: new Float64Array(lows.slice(0, 100_000)),
                close: new Float64Array(closes.slice(0, 100_000)),
                volume: new Float64Array(volumes.slice(0, 100_000))
            },
            '1M': {
                open: new Float64Array(opens),
                high: new Float64Array(highs),
                low: new Float64Array(lows),
                close: new Float64Array(closes),
                volume: new Float64Array(volumes)
            }
        };
        
        // Store VWAP data (timestamps, volumes, prices)
        this.vwapData = {
            '10k': {
                timestamps: new Float64Array(timestamps.slice(0, 10_000)),
                volumes: new Float64Array(volumes.slice(0, 10_000)),
                prices: new Float64Array(closes.slice(0, 10_000).map((c, i) => 
                    (highs[i] + lows[i] + c) / 3.0  // HLC3 price
                ))
            },
            '100k': {
                timestamps: new Float64Array(timestamps.slice(0, 100_000)),
                volumes: new Float64Array(volumes.slice(0, 100_000)),
                prices: new Float64Array(closes.slice(0, 100_000).map((c, i) => 
                    (highs[i] + lows[i] + c) / 3.0
                ))
            },
            '1M': {
                timestamps: new Float64Array(timestamps),
                volumes: new Float64Array(volumes),
                prices: new Float64Array(closes.map((c, i) => 
                    (highs[i] + lows[i] + c) / 3.0
                ))
            }
        };
        
        console.log(`Loaded data sizes: ${Object.keys(this.data).join(', ')}`);
    }

    /**
     * Generic benchmark function
     */
    benchmarkFunction(fn, name, metadata = {}) {
        const gcWasEnabled = global.gc ? true : false;
        if (CONFIG.disableGC && global.gc) {
            global.gc();
        }

        try {
            // Warmup phase
            let warmupElapsed = 0;
            let warmupIterations = 0;
            const warmupStart = performance.now();
            
            while (warmupElapsed < CONFIG.warmupTargetMs) {
                fn();
                warmupIterations++;
                warmupElapsed = performance.now() - warmupStart;
            }

            // Sampling phase
            const samples = [];
            
            for (let i = 0; i < CONFIG.sampleCount; i++) {
                const iterations = Math.max(CONFIG.minIterations, Math.floor(warmupIterations / 10));
                
                const start = performance.now();
                for (let j = 0; j < iterations; j++) {
                    fn();
                }
                const end = performance.now();
                
                const timePerIteration = (end - start) / iterations;
                samples.push(timePerIteration);
            }

            // Calculate statistics
            samples.sort((a, b) => a - b);
            const median = samples[Math.floor(samples.length / 2)];
            const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
            const min = samples[0];
            const max = samples[samples.length - 1];
            
            // Calculate throughput if data size is known
            const dataSize = metadata.dataSize || 0;
            const throughput = dataSize > 0 ? dataSize / (median * 1000) / 1000 : 0;

            return {
                name,
                median,
                mean,
                min,
                max,
                samples: samples.length,
                warmupIterations,
                throughput,
                ...metadata
            };
        } finally {
            if (CONFIG.disableGC && gcWasEnabled && global.gc) {
                global.gc();
            }
        }
    }

    /**
     * Benchmark safe/simple API for an indicator
     */
    benchmarkSafeAPI(indicatorKey, indicatorConfig) {
        console.log(`\n--- ${indicatorConfig.name} Safe API ---`);
        
        const { fn, params } = indicatorConfig.safe;
        const wasmFn = this.wasm[fn];
        
        if (!wasmFn) {
            console.log(`  Function ${fn} not found, skipping...`);
            return;
        }

        for (const [sizeName, data] of Object.entries(this.data)) {
            const benchName = `${indicatorKey}_safe_${sizeName}`;
            
            const result = this.benchmarkFunction(() => {
                const paramArray = this.prepareParams(params, data, indicatorConfig, sizeName);
                wasmFn.apply(this.wasm, paramArray);
            }, benchName, {
                dataSize: data.length,
                api: 'safe',
                indicator: indicatorKey
            });

            this.results[benchName] = result;
            this.printResult(result);
        }
    }

    /**
     * Benchmark fast/unsafe API for an indicator
     */
    benchmarkFastAPI(indicatorKey, indicatorConfig) {
        console.log(`\n--- ${indicatorConfig.name} Fast/Unsafe API ---`);
        
        const { allocFn, freeFn, computeFn, params, dualOutput, outputCount } = indicatorConfig.fast;
        
        if (!this.wasm[allocFn] || !this.wasm[freeFn] || !this.wasm[computeFn]) {
            console.log(`  Fast API functions not found, skipping...`);
            return;
        }

        for (const [sizeName, data] of Object.entries(this.data)) {
            const benchName = `${indicatorKey}_fast_${sizeName}`;
            const len = data.length;
            
            let inPtr, outPtr, highPtr, lowPtr, closePtr, outPtr2, outPtr3, timestampsPtr, volumesPtr, pricesPtr, volumePtr;
            
            try {
                // Handle custom inputs
                if (indicatorConfig.fast && indicatorConfig.fast.inputs) {
                    const ohlc = this.ohlcData[sizeName];
                    const inputPtrs = {};
                    
                    // Special handling for VWMACD which has multiple inputs AND outputs
                    if (indicatorConfig.name === 'VWMACD (Volume Weighted MACD)' && outputCount === 3) {
                        // Allocate input buffers
                        closePtr = this.wasm[allocFn](len);
                        volumePtr = this.wasm[allocFn](len);
                        
                        // Allocate output buffers
                        outPtr = this.wasm[allocFn](len);    // MACD
                        outPtr2 = this.wasm[allocFn](len);   // Signal
                        outPtr3 = this.wasm[allocFn](len);   // Histogram
                        
                        // Copy input data
                        const closeView = new Float64Array(this.wasm.__wasm.memory.buffer, closePtr, len);
                        const volumeView = new Float64Array(this.wasm.__wasm.memory.buffer, volumePtr, len);
                        closeView.set(ohlc.close);
                        volumeView.set(ohlc.volume);
                        
                        const result = this.benchmarkFunction(() => {
                            const paramArray = [closePtr, volumePtr, outPtr, outPtr2, outPtr3, len,
                                params.fast_period, params.slow_period, params.signal_period,
                                params.fast_ma_type, params.slow_ma_type, params.signal_ma_type];
                            this.wasm[computeFn].apply(this.wasm, paramArray);
                        }, benchName, {
                            dataSize: len,
                            api: 'fast',
                            indicator: indicatorKey
                        });

                        this.results[benchName] = result;
                        this.printResult(result);
                        
                        // Store pointers for cleanup
                        Object.assign(this, { closePtr, volumePtr, outPtr, outPtr2, outPtr3 });
                    } else {
                        // Standard custom inputs handling
                        // Allocate buffers for each input
                        for (const input of indicatorConfig.fast.inputs) {
                            inputPtrs[input] = this.wasm[allocFn](len);
                        }
                        outPtr = this.wasm[allocFn](len);
                        
                        // Copy data for each input
                        for (const input of indicatorConfig.fast.inputs) {
                            const view = new Float64Array(this.wasm.__wasm.memory.buffer, inputPtrs[input], len);
                            if (input === 'prices') {
                                view.set(ohlc.close);
                            } else if (input === 'volumes') {
                                view.set(ohlc.volume);
                            } else if (ohlc[input]) {
                                view.set(ohlc[input]);
                            }
                        }
                        
                        const result = this.benchmarkFunction(() => {
                            // Build parameter array based on inputs
                            const paramArray = [];
                            for (const input of indicatorConfig.fast.inputs) {
                                paramArray.push(inputPtrs[input]);
                            }
                            paramArray.push(outPtr);
                            paramArray.push(len);
                            
                            // Add other parameters
                            for (const value of Object.values(params)) {
                                paramArray.push(value);
                            }
                            
                            this.wasm[computeFn].apply(this.wasm, paramArray);
                        }, benchName, {
                            dataSize: len,
                            api: 'fast',
                            indicator: indicatorKey
                        });

                        this.results[benchName] = result;
                        this.printResult(result);
                        
                        // Store pointers for cleanup in finally block
                        Object.assign(this, { inputPtrs, outPtr });
                    }
                }
                // Handle multiple inputs if needed (legacy)
                else if (indicatorConfig.fast.needsMultipleInputs) {
                    const ohlc = this.ohlcData[sizeName];
                    
                    // Allocate buffers for high, low, close
                    highPtr = this.wasm[allocFn](len);
                    lowPtr = this.wasm[allocFn](len);
                    closePtr = this.wasm[allocFn](len);
                    
                    // Special handling for TTM Trend which outputs u8
                    if (indicatorConfig.fast.outputIsU8 && indicatorConfig.fast.allocU8Fn) {
                        outPtr = this.wasm[indicatorConfig.fast.allocU8Fn](len);
                    } else {
                        outPtr = this.wasm[allocFn](len);
                    }
                    
                    // Allocate volume buffer for ADOSC and EMD
                    if (indicatorConfig.name === 'ADOSC' || indicatorConfig.name === 'EMD') {
                        volumePtr = this.wasm[allocFn](len);
                    }
                    
                    // Allocate second and third output buffers for triple output indicators
                    if (indicatorConfig.fast.tripleOutput) {
                        outPtr2 = this.wasm[allocFn](len);
                        outPtr3 = this.wasm[allocFn](len);
                    } else if (indicatorConfig.fast.dualOutput) {
                        outPtr2 = this.wasm[allocFn](len);
                    }
                    
                    // Copy data
                    // Special handling for TTM Trend which needs (source, close) instead of (high, low, close)
                    if (indicatorConfig.name === 'TTM Trend') {
                        // For TTM Trend, use HL2 as source
                        const sourceView = new Float64Array(this.wasm.__wasm.memory.buffer, highPtr, len);
                        const closeView = new Float64Array(this.wasm.__wasm.memory.buffer, lowPtr, len);
                        
                        // Calculate HL2 = (high + low) / 2
                        const hl2 = new Float64Array(len);
                        for (let i = 0; i < len; i++) {
                            hl2[i] = (ohlc.high[i] + ohlc.low[i]) / 2;
                        }
                        
                        sourceView.set(hl2);
                        closeView.set(ohlc.close.slice(0, len));
                    } else {
                        const highView = new Float64Array(this.wasm.__wasm.memory.buffer, highPtr, len);
                        const lowView = new Float64Array(this.wasm.__wasm.memory.buffer, lowPtr, len);
                        const closeView = new Float64Array(this.wasm.__wasm.memory.buffer, closePtr, len);
                        
                        highView.set(ohlc.high.slice(0, len));
                        lowView.set(ohlc.low.slice(0, len));
                        closeView.set(ohlc.close.slice(0, len));
                    }
                    
                    // Copy volume data for ADOSC and EMD
                    if (indicatorConfig.name === 'ADOSC' || indicatorConfig.name === 'EMD') {
                        const volumeView = new Float64Array(this.wasm.__wasm.memory.buffer, volumePtr, len);
                        // Use slice to ensure we don't exceed the allocated buffer length
                        volumeView.set(ohlc.volume.slice(0, len));
                    }
                    
                    // Allocate second output buffer if indicator has dual outputs
                    outPtr2 = (indicatorConfig.fast.dualOutput || indicatorConfig.fast.needsStochFInputs) ? this.wasm[allocFn](len) : null;
                    
                    // Debug removed for performance
                    
                    const result = this.benchmarkFunction(() => {
                        // Pass the full indicatorConfig so name is available
                        const modifiedConfig = Object.assign({}, indicatorConfig.fast, { name: indicatorConfig.name });
                        const paramArray = this.prepareFastParams(params, null, outPtr, len, modifiedConfig, highPtr, lowPtr, closePtr, indicatorConfig.fast.dualOutput || indicatorConfig.fast.tripleOutput, outPtr2, volumePtr, null, null, null, outPtr3);
                        this.wasm[computeFn].apply(this.wasm, paramArray);
                    }, benchName, {
                        dataSize: len,
                        api: 'fast',
                        indicator: indicatorKey
                    });

                    this.results[benchName] = result;
                    this.printResult(result);
                } else if (indicatorConfig.fast.needsVwapInputs) {
                    // Handle VWAP indicators that need timestamps, volumes, prices
                    const vwap = this.vwapData[sizeName];
                    
                    // Allocate buffers for VWAP inputs
                    timestampsPtr = this.wasm[allocFn](len);
                    volumesPtr = this.wasm[allocFn](len);
                    pricesPtr = this.wasm[allocFn](len);
                    outPtr = this.wasm[allocFn](len);
                    
                    // Copy data
                    const timestampsView = new Float64Array(this.wasm.__wasm.memory.buffer, timestampsPtr, len);
                    const volumesView = new Float64Array(this.wasm.__wasm.memory.buffer, volumesPtr, len);
                    const pricesView = new Float64Array(this.wasm.__wasm.memory.buffer, pricesPtr, len);
                    
                    timestampsView.set(vwap.timestamps);
                    volumesView.set(vwap.volumes);
                    pricesView.set(vwap.prices);
                    
                    const result = this.benchmarkFunction(() => {
                        const paramArray = this.prepareFastParams(params, null, outPtr, len, indicatorConfig.fast, null, null, null, false, null, null, timestampsPtr, volumesPtr, pricesPtr);
                        this.wasm[computeFn].apply(this.wasm, paramArray);
                    }, benchName, {
                        dataSize: len,
                        api: 'fast',
                        indicator: indicatorKey
                    });

                    this.results[benchName] = result;
                    this.printResult(result);
                } else if (outputCount === 3) {
                    // Handle triple output indicators (alligator)
                    inPtr = this.wasm[allocFn](len);
                    outPtr = this.wasm[allocFn](len);   // jaw
                    outPtr2 = this.wasm[allocFn](len);  // teeth
                    outPtr3 = this.wasm[allocFn](len);  // lips
                    
                    // Copy data once
                    const inView = new Float64Array(this.wasm.__wasm.memory.buffer, inPtr, len);
                    inView.set(data);
                    
                    const result = this.benchmarkFunction(() => {
                        const paramArray = [inPtr, outPtr, outPtr2, outPtr3, len];
                        // Add indicator parameters
                        for (const value of Object.values(params)) {
                            paramArray.push(value);
                        }
                        this.wasm[computeFn].apply(this.wasm, paramArray);
                    }, benchName, {
                        dataSize: len,
                        api: 'fast',
                        indicator: indicatorKey
                    });

                    this.results[benchName] = result;
                    this.printResult(result);
                } else if (indicatorConfig.fast.quadOutput) {
                    // Handle quad output indicators (gatorosc)
                    inPtr = this.wasm[allocFn](len);
                    outPtr = this.wasm[allocFn](len);   // upper
                    outPtr2 = this.wasm[allocFn](len);  // lower
                    outPtr3 = this.wasm[allocFn](len);  // upper_change
                    const outPtr4 = this.wasm[allocFn](len);  // lower_change
                    
                    // Copy data once
                    const inView = new Float64Array(this.wasm.__wasm.memory.buffer, inPtr, len);
                    inView.set(data);
                    
                    const result = this.benchmarkFunction(() => {
                        const paramArray = [inPtr, outPtr, outPtr2, outPtr3, outPtr4, len];
                        // Add indicator parameters
                        for (const value of Object.values(params)) {
                            paramArray.push(value);
                        }
                        this.wasm[computeFn].apply(this.wasm, paramArray);
                    }, benchName, {
                        dataSize: len,
                        api: 'fast',
                        indicator: indicatorKey
                    });

                    this.results[benchName] = result;
                    this.printResult(result);
                    
                    // Free the extra pointer
                    if (outPtr4) this.wasm[freeFn](outPtr4, len);
                } else if (outputCount === 4) {
                    // Handle quadruple output indicators (correlation_cycle)
                    inPtr = this.wasm[allocFn](len);
                    outPtr = this.wasm[allocFn](len);   // real
                    outPtr2 = this.wasm[allocFn](len);  // imag
                    outPtr3 = this.wasm[allocFn](len);  // angle
                    const outPtr4 = this.wasm[allocFn](len);  // state
                    
                    // Copy data once
                    const inView = new Float64Array(this.wasm.__wasm.memory.buffer, inPtr, len);
                    inView.set(data);
                    
                    const result = this.benchmarkFunction(() => {
                        const paramArray = [inPtr, outPtr, outPtr2, outPtr3, outPtr4, len];
                        // Add indicator parameters
                        for (const value of Object.values(params)) {
                            paramArray.push(value);
                        }
                        this.wasm[computeFn].apply(this.wasm, paramArray);
                    }, benchName, {
                        dataSize: len,
                        api: 'fast',
                        indicator: indicatorKey
                    });

                    this.results[benchName] = result;
                    this.printResult(result);
                    
                    // Store outPtr4 for cleanup
                    this.outPtr4 = outPtr4;
                } else {
                    // Pre-allocate buffers outside of benchmark
                    inPtr = this.wasm[allocFn](len);
                    outPtr = this.wasm[allocFn](len);
                    outPtr2 = indicatorConfig.fast.dualOutput ? this.wasm[allocFn](len) : null;
                    
                    // Copy data once
                    const inView = new Float64Array(this.wasm.__wasm.memory.buffer, inPtr, len);
                    inView.set(data);
                    
                    const result = this.benchmarkFunction(() => {
                        const paramArray = this.prepareFastParams(params, inPtr, outPtr, len, indicatorConfig.fast, null, null, null, indicatorConfig.fast.dualOutput, outPtr2, null);
                        this.wasm[computeFn].apply(this.wasm, paramArray);
                    }, benchName, {
                        dataSize: len,
                        api: 'fast',
                        indicator: indicatorKey
                    });

                    this.results[benchName] = result;
                    this.printResult(result);
                }
            } finally {
                // Clean up allocated memory
                if (indicatorConfig.fast.needsVwapInputs) {
                    if (timestampsPtr) this.wasm[freeFn](timestampsPtr, len);
                    if (volumesPtr) this.wasm[freeFn](volumesPtr, len);
                    if (pricesPtr) this.wasm[freeFn](pricesPtr, len);
                    if (outPtr) this.wasm[freeFn](outPtr, len);
                } else if (this.inputPtrs) {
                    // Clean up custom input pointers
                    for (const ptr of Object.values(this.inputPtrs)) {
                        if (ptr) this.wasm[freeFn](ptr, len);
                    }
                    delete this.inputPtrs;
                    if (this.outPtr) {
                        this.wasm[freeFn](this.outPtr, len);
                        delete this.outPtr;
                    }
                } else if (this.closePtr && this.volumePtr) {
                    // Clean up VWMACD pointers
                    if (this.closePtr) this.wasm[freeFn](this.closePtr, len);
                    if (this.volumePtr) this.wasm[freeFn](this.volumePtr, len);
                    if (this.outPtr) this.wasm[freeFn](this.outPtr, len);
                    if (this.outPtr2) this.wasm[freeFn](this.outPtr2, len);
                    if (this.outPtr3) this.wasm[freeFn](this.outPtr3, len);
                    delete this.closePtr;
                    delete this.volumePtr;
                    delete this.outPtr;
                    delete this.outPtr2;
                    delete this.outPtr3;
                } else if (indicatorConfig.fast.needsMultipleInputs) {
                    if (highPtr) this.wasm[freeFn](highPtr, len);
                    if (lowPtr) this.wasm[freeFn](lowPtr, len);
                    if (closePtr) this.wasm[freeFn](closePtr, len);
                    if (volumePtr) this.wasm[freeFn](volumePtr, len);
                    
                    // Special handling for TTM Trend which outputs u8
                    if (outPtr) {
                        if (indicatorConfig.fast.outputIsU8 && indicatorConfig.fast.freeU8Fn) {
                            this.wasm[indicatorConfig.fast.freeU8Fn](outPtr, len);
                        } else {
                            this.wasm[freeFn](outPtr, len);
                        }
                    }
                    if (outPtr2) this.wasm[freeFn](outPtr2, len);
                    if (outPtr3) this.wasm[freeFn](outPtr3, len);
                } else if (outputCount === 3) {
                    if (inPtr) this.wasm[freeFn](inPtr, len);
                    if (outPtr) this.wasm[freeFn](outPtr, len);
                    if (outPtr2) this.wasm[freeFn](outPtr2, len);
                    if (outPtr3) this.wasm[freeFn](outPtr3, len);
                } else if (indicatorConfig.fast.quadOutput) {
                    if (inPtr) this.wasm[freeFn](inPtr, len);
                    if (outPtr) this.wasm[freeFn](outPtr, len);
                    if (outPtr2) this.wasm[freeFn](outPtr2, len);
                    if (outPtr3) this.wasm[freeFn](outPtr3, len);
                    // outPtr4 was already freed above right after benchmark
                } else if (outputCount === 4) {
                    if (inPtr) this.wasm[freeFn](inPtr, len);
                    if (outPtr) this.wasm[freeFn](outPtr, len);
                    if (outPtr2) this.wasm[freeFn](outPtr2, len);
                    if (outPtr3) this.wasm[freeFn](outPtr3, len);
                    if (this.outPtr4) {
                        this.wasm[freeFn](this.outPtr4, len);
                        delete this.outPtr4;
                    }
                } else {
                    if (inPtr) this.wasm[freeFn](inPtr, len);
                    if (outPtr) this.wasm[freeFn](outPtr, len);
                    if (outPtr2) this.wasm[freeFn](outPtr2, len);
                }
            }
        }
    }

    /**
     * Benchmark batch API if available
     */
    benchmarkBatchAPI(indicatorKey, indicatorConfig) {
        if (!indicatorConfig.batch) return;
        
        console.log(`\n--- ${indicatorConfig.name} Batch API ---`);
        
        const { fn, config } = indicatorConfig.batch;
        const wasmFn = this.wasm[fn];
        
        if (!wasmFn) {
            console.log(`  Batch function ${fn} not found, skipping...`);
            return;
        }

        // Only test with 10k data for batch operations
        const data = this.data['10k'];
        const sizeName = '10k';
        
        for (const [configName, batchConfig] of Object.entries(config)) {
            const benchName = `${indicatorKey}_batch_${configName}`;
            
            const result = this.benchmarkFunction(() => {
                if (indicatorConfig.needsVwapInputs || indicatorConfig.batch?.needsVwapInputs) {
                    const vwap = this.vwapData[sizeName];
                    wasmFn.call(this.wasm, vwap.timestamps, vwap.volumes, vwap.prices, { anchor_range: batchConfig.anchor_range });
                } else if (indicatorKey === 'ad') {
                    // AD batch requires flattened arrays and rows parameter
                    const ohlc = this.ohlcData[sizeName];
                    // Simulate batch of 10 securities
                    const rows = 10;
                    const cols = Math.floor(ohlc.high.length / rows);
                    const flatSize = rows * cols;
                    
                    // Create flattened arrays (repeat same data for each row)
                    const highs_flat = new Float64Array(flatSize);
                    const lows_flat = new Float64Array(flatSize);
                    const closes_flat = new Float64Array(flatSize);
                    const volumes_flat = new Float64Array(flatSize);
                    
                    for (let i = 0; i < rows; i++) {
                        const offset = i * cols;
                        highs_flat.set(ohlc.high.subarray(0, cols), offset);
                        lows_flat.set(ohlc.low.subarray(0, cols), offset);
                        closes_flat.set(ohlc.close.subarray(0, cols), offset);
                        volumes_flat.set(ohlc.volume.subarray(0, cols), offset);
                    }
                    
                    wasmFn.call(this.wasm, highs_flat, lows_flat, closes_flat, volumes_flat, rows);
                } else if (indicatorConfig.needsMultipleInputs || indicatorConfig.fast?.needsMultipleInputs) {
                    const ohlc = this.ohlcData[sizeName];
                    // ADOSC and EMD need volume in addition to high, low, close
                    if (indicatorConfig.name === 'ADOSC' || indicatorConfig.name === 'EMD') {
                        // ADOSC and EMD use the new ergonomic batch API with config object
                        wasmFn.call(this.wasm, ohlc.high, ohlc.low, ohlc.close, ohlc.volume, batchConfig);
                    } else if (indicatorConfig.name === 'SafeZoneStop') {
                        // SafeZoneStop only needs high/low (no close) and uses config object
                        wasmFn.call(this.wasm, ohlc.high, ohlc.low, batchConfig);
                    } else if (indicatorConfig.name === 'TTM Trend') {
                        // TTM Trend needs source (HL2) and close
                        const hl2 = new Float64Array(ohlc.high.length);
                        for (let i = 0; i < ohlc.high.length; i++) {
                            hl2[i] = (ohlc.high[i] + ohlc.low[i]) / 2;
                        }
                        wasmFn.call(this.wasm, hl2, ohlc.close, batchConfig);
                    } else {
                        wasmFn.call(this.wasm, ohlc.high, ohlc.low, ohlc.close, batchConfig);
                    }
                } else if ((indicatorKey === 'pwma' || indicatorKey === 'supersmoother') && batchConfig.period_range) {
                    // PWMA and SuperSmoother have special batch APIs that take individual parameters
                    const [start, end, step] = batchConfig.period_range;
                    wasmFn.call(this.wasm, data, start, end, step);
                } else if (indicatorKey === 'trendflex' || indicatorKey === 'wilders' || indicatorKey === 'alma' || 
                           indicatorKey === 'highpass' || indicatorKey === 'jsa' || indicatorKey === 'maaq' || 
                           indicatorKey === 'smma' || indicatorKey === 'ehlers_itrend' ||
                           indicatorKey === 'cwma' || indicatorKey === 'dema' || indicatorKey === 'epma' || 
                           indicatorKey === 'jma' || indicatorKey === 'highpass_2_pole' || indicatorKey === 'nma' || 
                           indicatorKey === 'sma' || indicatorKey === 'supersmoother_3_pole' || indicatorKey === 'ema' || 
                           indicatorKey === 'tema' || indicatorKey === 'gaussian' || indicatorKey === 'hwma' || 
                           indicatorKey === 'mwdx' || indicatorKey === 'srwma' || indicatorKey === 'linreg' || 
                           indicatorKey === 'sinwma' || indicatorKey === 'zlema' || indicatorKey === 'adx') {
                    // These indicators use the new ergonomic batch API with config object
                    wasmFn.call(this.wasm, data, batchConfig);
                } else {
                    const params = this.prepareBatchParams(indicatorKey, data, batchConfig, sizeName);
                    wasmFn.apply(this.wasm, params);
                }
            }, benchName, {
                dataSize: data.length,
                api: 'batch',
                indicator: indicatorKey,
                batchSize: configName
            });

            this.results[benchName] = result;
            this.printResult(result);
            
            // Calculate total combinations for batch
            let totalCombinations = 1;
            if (batchConfig.period_range) {
                const periods = Math.floor((batchConfig.period_range[1] - batchConfig.period_range[0]) / batchConfig.period_range[2]) + 1;
                totalCombinations = periods;
                
                // Handle optional parameters
                if (batchConfig.offset_range) {
                    const offsets = Math.floor((batchConfig.offset_range[1] - batchConfig.offset_range[0]) / batchConfig.offset_range[2]) + 1;
                    totalCombinations *= offsets;
                }
                if (batchConfig.sigma_range) {
                    const sigmas = Math.floor((batchConfig.sigma_range[1] - batchConfig.sigma_range[0]) / batchConfig.sigma_range[2]) + 1;
                    totalCombinations *= sigmas;
                }
                
                // Handle volume_factor_range for tilson
                if (batchConfig.volume_factor_range) {
                    const vFactors = Math.floor((batchConfig.volume_factor_range[1] - batchConfig.volume_factor_range[0]) / batchConfig.volume_factor_range[2]) + 1;
                    totalCombinations *= vFactors;
                }
            }
            
            // Handle ADOSC's special ranges
            if (batchConfig.short_period_range && batchConfig.long_period_range) {
                const shortPeriods = Math.floor((batchConfig.short_period_range[1] - batchConfig.short_period_range[0]) / batchConfig.short_period_range[2]) + 1;
                const longPeriods = Math.floor((batchConfig.long_period_range[1] - batchConfig.long_period_range[0]) / batchConfig.long_period_range[2]) + 1;
                totalCombinations = shortPeriods * longPeriods;
            }
            console.log(`  Total combinations: ${totalCombinations}`);
        }
    }

    /**
     * Prepare parameters for safe API call
     */
    prepareParams(params, data, indicatorConfig, sizeName) {
        // Check if this indicator needs VWAP inputs
        if (indicatorConfig.safe?.needsVwapInputs || indicatorConfig.needsVwapInputs) {
            const vwap = this.vwapData[sizeName];
            const result = [vwap.timestamps, vwap.volumes, vwap.prices];
            
            // Add parameters in order
            for (const value of Object.values(params)) {
                result.push(value);
            }
            
            return result;
        }
        
        // Check if indicator specifies custom inputs
        if (indicatorConfig.safe && indicatorConfig.safe.inputs) {
            const result = [];
            const ohlc = this.ohlcData[sizeName];
            
            // Add each specified input in order
            for (const input of indicatorConfig.safe.inputs) {
                if (input === 'prices') {
                    result.push(ohlc.close);
                } else if (input === 'volumes') {
                    result.push(ohlc.volume);
                } else if (ohlc[input]) {
                    result.push(ohlc[input]);
                } else {
                    throw new Error(`Unknown input type: ${input}`);
                }
            }
            
            // Add parameters in order
            for (const value of Object.values(params)) {
                result.push(value);
            }
            
            return result;
        }
        
        // Check if this indicator needs multiple inputs (legacy)
        if (indicatorConfig.needsMultipleInputs || (indicatorConfig.safe && indicatorConfig.safe.needsMultipleInputs)) {
            const ohlc = this.ohlcData[sizeName];
            
            // Special case for ACOSC which only needs high/low
            if (indicatorConfig.name === 'ACOSC') {
                const result = [ohlc.high, ohlc.low];
                
                // Add parameters in order
                for (const value of Object.values(params)) {
                    result.push(value);
                }
                
                return result;
            }
            
            // Special case for SAR which also only needs high/low
            if (indicatorConfig.name === 'SAR') {
                const result = [ohlc.high, ohlc.low];
                
                // Add parameters in order
                for (const value of Object.values(params)) {
                    result.push(value);
                }
                
                return result;
            }
            
            // Special case for SafeZoneStop which needs high/low and direction string
            if (indicatorConfig.name === 'SafeZoneStop') {
                const result = [ohlc.high, ohlc.low];
                
                // Add parameters in order (period, mult, max_lookback, direction)
                result.push(params.period);
                result.push(params.mult);
                result.push(params.max_lookback);
                result.push(params.direction);
                
                return result;
            }
            
            // Special case for ADOSC which needs high, low, close, volume
            if (indicatorConfig.name === 'ADOSC') {
                const result = [ohlc.high, ohlc.low, ohlc.close, ohlc.volume];
                
                // Add parameters in order
                for (const value of Object.values(params)) {
                    result.push(value);
                }
                
                return result;
            }
            
            // Special case for EMD which needs high, low, close, volume
            if (indicatorConfig.name === 'EMD') {
                const result = [ohlc.high, ohlc.low, ohlc.close, ohlc.volume];
                
                // Add parameters in order
                for (const value of Object.values(params)) {
                    result.push(value);
                }
                
                return result;
            }
            
            // Special case for TTM Trend which needs source (HL2) and close
            if (indicatorConfig.name === 'TTM Trend') {
                // Calculate HL2 = (high + low) / 2
                const hl2 = new Float64Array(ohlc.high.length);
                for (let i = 0; i < ohlc.high.length; i++) {
                    hl2[i] = (ohlc.high[i] + ohlc.low[i]) / 2;
                }
                
                const result = [hl2, ohlc.close];
                
                // Add parameters in order
                for (const value of Object.values(params)) {
                    result.push(value);
                }
                
                return result;
            }
            
            // Default case: high, low, close
            const result = [ohlc.high, ohlc.low, ohlc.close];
            
            // Add parameters in order
            for (const value of Object.values(params)) {
                result.push(value);
            }
            
            return result;
        }
        
        // Standard single data array
        const result = [data];
        
        // Add parameters in order (assumes params object maintains order)
        for (const value of Object.values(params)) {
            result.push(value);
        }
        
        return result;
    }

    /**
     * Prepare parameters for batch API call
     */
    prepareBatchParams(indicatorKey, data, batchConfig, sizeName) {
        // Special handling for different indicators
        if (indicatorKey === 'mama') {
            // MAMA expects: data, fast_limit_start, fast_limit_end, fast_limit_step, slow_limit_start, slow_limit_end, slow_limit_step
            const fast = batchConfig.fast_limit_range || [0.5, 0.5, 0];
            const slow = batchConfig.slow_limit_range || [0.05, 0.05, 0];
            return [data, fast[0], fast[1], fast[2], slow[0], slow[1], slow[2]];
        } else if (indicatorKey === 'sqwma' || indicatorKey === 'fwma' || indicatorKey === 'hma' || indicatorKey === 'kama' || indicatorKey === 'wma') {
            // These indicators expect: data, period_start, period_end, period_step
            const period = batchConfig.period_range;
            return [data, period[0], period[1], period[2]];
        } else if (indicatorKey === 'vpwma') {
            // VPWMA expects: data, period_start, period_end, period_step, power_start, power_end, power_step
            const period = batchConfig.period_range || [14, 14, 1];
            const power = batchConfig.power_range || [0.382, 0.382, 0.1];
            return [data, period[0], period[1], period[2], power[0], power[1], power[2]];
        } else if (indicatorKey === 'swma' || indicatorKey === 'trima') {
            // SWMA and TRIMA use the new unified batch API with serde config
            return [data, { period_range: batchConfig.period_range }];
        } else if (indicatorKey === 'vwap') {
            // VWAP uses the new unified batch API with serde config
            const vwap = this.vwapData[Object.keys(this.vwapData)[0]]; // Get appropriate size
            return [vwap.timestamps, vwap.volumes, vwap.prices, { anchor_range: batchConfig.anchor_range }];
        } else if (indicatorKey === 'tilson') {
            // Tilson uses the new unified batch API with serde config
            return [data, { 
                period_range: batchConfig.period_range,
                volume_factor_range: batchConfig.volume_factor_range || [0.0, 0.0, 0.0]
            }];
        } else if (indicatorKey === 'vwma') {
            // VWMA uses the new unified batch API with serde config
            const ohlc = this.ohlcData[sizeName];
            return [ohlc.close, ohlc.volume, { period_range: batchConfig.period_range }];
        } else if (indicatorKey === 'vwmacd') {
            // VWMACD uses the new unified batch API with serde config
            const ohlc = this.ohlcData[sizeName];
            return [ohlc.close, ohlc.volume, batchConfig];
        } else if (batchConfig.period_range) {
            // Most indicators with period ranges
            const period = batchConfig.period_range;
            const result = [data, [period[0], period[1], period[2]]];
            
            // Add additional ranges if present
            if (batchConfig.offset_range) {
                const offset = batchConfig.offset_range;
                result.push([offset[0], offset[1], offset[2]]);
            }
            if (batchConfig.sigma_range) {
                const sigma = batchConfig.sigma_range;
                result.push([sigma[0], sigma[1], sigma[2]]);
            }
            
            return result;
        } else {
            // Default: pass data and config as-is
            return [data, batchConfig];
        }
    }

    /**
     * Prepare parameters for fast API call
     */
    prepareFastParams(params, inPtr, outPtr, len, indicatorConfig, highPtr, lowPtr, closePtr, dualOutput = false, outPtr2 = null, volumePtr = null, timestampsPtr = null, volumesPtr = null, pricesPtr = null, outPtr3 = null) {
        // Check if this indicator needs VWAP inputs
        if (indicatorConfig.needsVwapInputs) {
            // For VWAP: timestamps_ptr, volumes_ptr, prices_ptr, out_ptr, len, ...params
            const result = [timestampsPtr, volumesPtr, pricesPtr, outPtr, len];
            
            // Add indicator parameters
            for (const value of Object.values(params)) {
                result.push(value);
            }
            
            return result;
        }
        
        // Check if this indicator needs multiple inputs
        if (indicatorConfig.needsMultipleInputs || (indicatorConfig.fast && indicatorConfig.fast.needsMultipleInputs)) {
            // Special case for ACOSC: high_ptr, low_ptr, osc_ptr, change_ptr, len
            if (indicatorConfig.name === 'ACOSC') {
                const result = [highPtr, lowPtr, outPtr, outPtr2, len];
                
                // Add indicator parameters
                for (const value of Object.values(params)) {
                    result.push(value);
                }
                
                return result;
            }
            
            // Special case for SAR: high_ptr, low_ptr, out_ptr, len, acceleration, maximum
            if (indicatorConfig.name === 'SAR') {
                const result = [highPtr, lowPtr, outPtr, len];
                
                // Add indicator parameters
                for (const value of Object.values(params)) {
                    result.push(value);
                }
                
                return result;
            }
            
            // Special case for SafeZoneStop: high_ptr, low_ptr, out_ptr, len, period, mult, max_lookback, direction
            if (indicatorConfig.name === 'SafeZoneStop') {
                const result = [highPtr, lowPtr, outPtr, len];
                
                // Add parameters in specific order
                result.push(params.period);
                result.push(params.mult);
                result.push(params.max_lookback);
                result.push(params.direction);
                
                return result;
            }
            
            // Special case for ADOSC: high_ptr, low_ptr, close_ptr, volume_ptr, out_ptr, len, short_period, long_period
            if (indicatorConfig.name === 'ADOSC') {
                const result = [highPtr, lowPtr, closePtr, volumePtr, outPtr, len, params.short_period, params.long_period];
                return result;
            }
            
            // Special case for EMD: high_ptr, low_ptr, close_ptr, volume_ptr, upper_ptr, middle_ptr, lower_ptr, len, period, delta, fraction
            if (indicatorConfig.name === 'EMD') {
                const result = [highPtr, lowPtr, closePtr, volumePtr, outPtr, outPtr2, outPtr3 || outPtr, len, params.period, params.delta, params.fraction];
                return result;
            }
            
            // Special case for StochF: high_ptr, low_ptr, close_ptr, k_ptr, d_ptr, len, fastk_period, fastd_period, fastd_matype
            if (indicatorConfig.needsStochFInputs) {
                const result = [highPtr, lowPtr, closePtr, outPtr, outPtr2, len];
                
                // Add parameters in specific order
                result.push(params.fastk_period);
                result.push(params.fastd_period);
                result.push(params.fastd_matype);
                
                return result;
            }
            
            // Special case for VWMACD: close_ptr, volume_ptr, macd_ptr, signal_ptr, hist_ptr, len, ...params
            if (indicatorConfig.name === 'VWMACD (Volume Weighted MACD)') {
                const result = [closePtr, volumePtr, outPtr, outPtr2, outPtr3, len];
                
                // Add indicator parameters in order
                result.push(params.fast_period);
                result.push(params.slow_period);
                result.push(params.signal_period);
                result.push(params.fast_ma_type);
                result.push(params.slow_ma_type);
                result.push(params.signal_ma_type);
                
                return result;
            }
            
            // Special case for TTM Trend: source_ptr, close_ptr, out_ptr, len, period
            if (indicatorConfig.name === 'TTM Trend') {
                const result = [highPtr, lowPtr, outPtr, len, params.period];  // highPtr is source, lowPtr is close
                return result;
            }
            
            // For FRAMA/ADXR and others: high_ptr, low_ptr, close_ptr, out_ptr, len, ...params
            const result = [highPtr, lowPtr, closePtr, outPtr, len];
            
            // Add indicator parameters
            for (const value of Object.values(params)) {
                result.push(value);
            }
            
            return result;
        }
        
        // Standard fast API: in_ptr, out_ptr, len, ...params
        // For dual output indicators like MAMA: in_ptr, out_mama_ptr, out_fama_ptr, len, ...params
        const result = dualOutput 
            ? [inPtr, outPtr, outPtr2, len]
            : [inPtr, outPtr, len];
        
        // Add indicator parameters
        for (const value of Object.values(params)) {
            result.push(value);
        }
        
        return result;
    }

    printResult(result) {
        console.log(`\n${result.name}:`);
        console.log(`  Median: ${result.median.toFixed(3)} ms`);
        console.log(`  Mean:   ${result.mean.toFixed(3)} ms`);
        console.log(`  Min:    ${result.min.toFixed(3)} ms`);
        console.log(`  Max:    ${result.max.toFixed(3)} ms`);
        if (result.throughput > 0) {
            console.log(`  Throughput: ${result.throughput.toFixed(1)} M elem/s`);
        }
        console.log(`  Samples: ${result.samples}, Warmup iterations: ${result.warmupIterations}`);
    }

    printSummary() {
        console.log('\n' + '='.repeat(80));
        console.log('SUMMARY');
        console.log('='.repeat(80));
        
        // Group results by indicator
        const byIndicator = {};
        for (const [name, result] of Object.entries(this.results)) {
            const indicator = result.indicator;
            if (!byIndicator[indicator]) {
                byIndicator[indicator] = [];
            }
            byIndicator[indicator].push(result);
        }

        // Print comparison table for each indicator
        for (const [indicator, results] of Object.entries(byIndicator)) {
            const config = INDICATORS[indicator];
            console.log(`\n${config.name} Performance Comparison:`);
            console.log(`${'Size'.padEnd(10)} ${'Safe API (ms)'.padStart(15)} ${'Fast API (ms)'.padStart(15)} ${'Speedup'.padStart(10)}`);
            console.log('-'.repeat(52));
            
            for (const size of ['10k', '100k', '1M']) {
                const safe = results.find(r => r.name === `${indicator}_safe_${size}`);
                const fast = results.find(r => r.name === `${indicator}_fast_${size}`);
                
                if (safe && fast) {
                    const speedup = safe.median / fast.median;
                    console.log(
                        `${size.padEnd(10)} ${safe.median.toFixed(3).padStart(15)} ${fast.median.toFixed(3).padStart(15)} ${speedup.toFixed(2).padStart(9)}x`
                    );
                }
            }
        }

        // Save results
        const outputPath = join(__dirname, 'wasm_indicator_benchmark_results.json');
        const jsonResults = {
            timestamp: new Date().toISOString(),
            config: CONFIG,
            results: this.results,
        };
        
        try {
            writeFileSync(outputPath, JSON.stringify(jsonResults, null, 2));
            console.log(`\nResults saved to: ${outputPath}`);
        } catch (error) {
            console.error('Failed to save results:', error);
        }
    }

    async runIndicator(indicatorKey, indicatorConfig) {
        console.log('\n' + '='.repeat(80));
        console.log(`Benchmarking ${indicatorConfig.name}`);
        console.log('='.repeat(80));

        // Benchmark safe API
        if (indicatorConfig.safe) {
            this.benchmarkSafeAPI(indicatorKey, indicatorConfig);
        }

        // Benchmark fast API
        if (indicatorConfig.fast) {
            this.benchmarkFastAPI(indicatorKey, indicatorConfig);
        }

        // Benchmark batch API (if available)
        if (indicatorConfig.batch) {
            this.benchmarkBatchAPI(indicatorKey, indicatorConfig);
        }
    }

    async run(options = {}) {
        await this.initialize();
        
        const { indicators = Object.keys(INDICATORS) } = options;
        
        console.log('\nWASM Indicator Performance Benchmark');
        console.log('='.repeat(80));
        console.log('Configuration:');
        console.log(`  Warmup: ${CONFIG.warmupTargetMs}ms`);
        console.log(`  Samples: ${CONFIG.sampleCount}`);
        console.log(`  Min iterations: ${CONFIG.minIterations}`);
        console.log(`  GC disabled: ${CONFIG.disableGC}`);
        console.log(`  Indicators: ${indicators.join(', ')}`);

        // Run benchmarks for each indicator
        for (const indicatorKey of indicators) {
            const config = INDICATORS[indicatorKey];
            if (config) {
                await this.runIndicator(indicatorKey, config);
            } else {
                console.log(`\nWarning: Unknown indicator '${indicatorKey}'`);
            }
        }

        // Print summary
        this.printSummary();
    }
}

// Command line interface
async function main() {
    const args = process.argv.slice(2);
    
    // Check if running with GC control
    if (!global.gc && CONFIG.disableGC) {
        console.warn('\nWarning: GC control not available. Run with: node --expose-gc wasm_indicator_benchmark.js\n');
    }
    
    // Parse command line arguments
    let indicators = [];
    
    for (let i = 0; i < args.length; i++) {
        if (args[i] === '--help' || args[i] === '-h') {
            console.log('\nUsage: node --expose-gc wasm_indicator_benchmark.js [indicators...]');
            console.log('\nAvailable indicators:');
            for (const [key, config] of Object.entries(INDICATORS)) {
                console.log(`  ${key.padEnd(10)} - ${config.name}`);
            }
            console.log('\nExamples:');
            console.log('  node --expose-gc wasm_indicator_benchmark.js          # Run all indicators');
            console.log('  node --expose-gc wasm_indicator_benchmark.js alma     # Run only ALMA');
            console.log('  node --expose-gc wasm_indicator_benchmark.js alma sma # Run ALMA and SMA');
            return;
        } else {
            indicators.push(args[i]);
        }
    }

    // Default to all indicators if none specified
    if (indicators.length === 0) {
        indicators = Object.keys(INDICATORS);
    }

    const benchmark = new WasmIndicatorBenchmark();
    await benchmark.run({ indicators });
}

// Run if called directly
if (import.meta.url.startsWith('file://')) {
    main().catch(console.error);
}

export { WasmIndicatorBenchmark, INDICATORS };