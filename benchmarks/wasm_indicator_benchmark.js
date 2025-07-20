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
        },
        // Batch API
        batch: {
            fn: 'edcf_batch',
            config: {
                small: {
                    period_range: [10, 20, 5]  // 3 values: 10, 15, 20
                },
                medium: {
                    period_range: [15, 50, 5]  // 8 values: 15, 20, 25, ..., 50
                }
            }
        }
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
            fn: 'supersmoother_batch_js',  // Use the old API for benchmarking
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
            fn: 'fwma_batch',
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
            fn: 'hma_batch',  // This calls hma_batch_unified_js
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
            fn: 'kama_batch',
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
        const volumes = [];
        
        for (const line of lines) {
            const parts = line.split(',');
            if (parts.length >= 6) {
                opens.push(parseFloat(parts[1]));
                closes.push(parseFloat(parts[2]));
                highs.push(parseFloat(parts[3]));
                lows.push(parseFloat(parts[4]));
                volumes.push(parseFloat(parts[5]));
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
                close: new Float64Array(closes)
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
            
            let inPtr, outPtr, highPtr, lowPtr, closePtr, outPtr2, outPtr3, timestampsPtr, volumesPtr, pricesPtr, outPtr2;
            
            try {
                // Handle custom inputs
                if (indicatorConfig.fast && indicatorConfig.fast.inputs) {
                    const ohlc = this.ohlcData[sizeName];
                    const inputPtrs = {};
                    
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
                // Handle multiple inputs if needed (legacy)
                else if (indicatorConfig.fast.needsMultipleInputs) {
                    const ohlc = this.ohlcData[sizeName];
                    
                    // Allocate buffers for high, low, close
                    highPtr = this.wasm[allocFn](len);
                    lowPtr = this.wasm[allocFn](len);
                    closePtr = this.wasm[allocFn](len);
                    outPtr = this.wasm[allocFn](len);
                    
                    // Allocate volume buffer for ADOSC
                    if (indicatorConfig.name === 'ADOSC') {
                        volumePtr = this.wasm[allocFn](len);
                    }
                    
                    // Copy data
                    const highView = new Float64Array(this.wasm.__wasm.memory.buffer, highPtr, len);
                    const lowView = new Float64Array(this.wasm.__wasm.memory.buffer, lowPtr, len);
                    const closeView = new Float64Array(this.wasm.__wasm.memory.buffer, closePtr, len);
                    
                    highView.set(ohlc.high);
                    lowView.set(ohlc.low);
                    closeView.set(ohlc.close);
                    
                    // Copy volume data for ADOSC
                    if (indicatorConfig.name === 'ADOSC') {
                        const volumeView = new Float64Array(this.wasm.__wasm.memory.buffer, volumePtr, len);
                        volumeView.set(ohlc.volume);
                    }
                    
                    // Allocate second output buffer if indicator has dual outputs
                    const outPtr2 = indicatorConfig.fast.dualOutput ? this.wasm[allocFn](len) : null;
                    
                    // Debug removed for performance
                    
                    const result = this.benchmarkFunction(() => {
                        // Pass the full indicatorConfig so name is available
                        const modifiedConfig = Object.assign({}, indicatorConfig.fast, { name: indicatorConfig.name });
                        const paramArray = this.prepareFastParams(params, null, outPtr, len, modifiedConfig, highPtr, lowPtr, closePtr, indicatorConfig.fast.dualOutput, outPtr2, volumePtr);
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
                } else {
                    // Pre-allocate buffers outside of benchmark
                    inPtr = this.wasm[allocFn](len);
                            outPtr2 = dualOutput ? this.wasm[allocFn](len) : null;
                    outPtr = this.wasm[allocFn](len);
                    
                    // Copy data once
                    const inView = new Float64Array(this.wasm.__wasm.memory.buffer, inPtr, len);
                    inView.set(data);
                    
                    const result = this.benchmarkFunction(() => {
                        const paramArray = this.prepareFastParams(params, inPtr, outPtr, len, indicatorConfig.fast, dualOutput, outPtr2);
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
                } else if (indicatorConfig.fast.needsMultipleInputs) {
                    if (highPtr) this.wasm[freeFn](highPtr, len);
                    if (lowPtr) this.wasm[freeFn](lowPtr, len);
                    if (closePtr) this.wasm[freeFn](closePtr, len);
                    if (volumePtr) this.wasm[freeFn](volumePtr, len);
                    if (outPtr) this.wasm[freeFn](outPtr, len);
                } else if (outputCount === 3) {
                    if (inPtr) this.wasm[freeFn](inPtr, len);
                    if (outPtr) this.wasm[freeFn](outPtr, len);
                    if (outPtr2) this.wasm[freeFn](outPtr2, len);
                    if (outPtr3) this.wasm[freeFn](outPtr3, len);
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
                    // ADOSC needs volume in addition to high, low, close
                    if (indicatorConfig.name === 'ADOSC') {
                        // ADOSC uses the new ergonomic batch API with config object
                        wasmFn.call(this.wasm, ohlc.high, ohlc.low, ohlc.close, ohlc.volume, batchConfig);
                    } else {
                        wasmFn.call(this.wasm, ohlc.high, ohlc.low, ohlc.close, batchConfig);
                    }
                } else if ((indicatorKey === 'pwma' || indicatorKey === 'supersmoother') && batchConfig.period_range) {
                    // PWMA and SuperSmoother have special batch APIs that take individual parameters
                    const [start, end, step] = batchConfig.period_range;
                    wasmFn.call(this.wasm, data, start, end, step);
                } else if (indicatorKey === 'trendflex' || indicatorKey === 'wilders') {
                    // TrendFlex and Wilders use the new ergonomic batch API with config object
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
            
            // Special case for ADOSC which needs high, low, close, volume
            if (indicatorConfig.name === 'ADOSC') {
                const result = [ohlc.high, ohlc.low, ohlc.close, ohlc.volume];
                
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
    prepareFastParams(params, inPtr, outPtr, len, indicatorConfig, highPtr, lowPtr, closePtr, dualOutput = false, outPtr2 = null, timestampsPtr = null, volumesPtr = null, pricesPtr = null) {
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
            
            // Special case for ADOSC: high_ptr, low_ptr, close_ptr, volume_ptr, out_ptr, len, short_period, long_period
            if (indicatorConfig.name === 'ADOSC') {
                const result = [highPtr, lowPtr, closePtr, volumePtr, outPtr, len, params.short_period, params.long_period];
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