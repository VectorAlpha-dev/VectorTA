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
    }
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
    // Example: RSI with batch support (uncomment when RSI WASM bindings are added)
    /*
    rsi: {
        name: 'RSI',
        safe: {
            fn: 'rsi_js',
            params: { period: 14 }
        },
        fast: {
            allocFn: 'rsi_alloc',
            freeFn: 'rsi_free',
            computeFn: 'rsi_into',
            params: { period: 14 }
        },
        batch: {
            fn: 'rsi_batch',
            config: {
                small: {
                    period_range: [10, 20, 5]  // 3 values: 10, 15, 20
                },
                medium: {
                    period_range: [10, 30, 2]  // 11 values: 10, 12, 14, ..., 30
                }
            }
        }
    }
    */
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
        
        // Parse close prices
        // Note: CSV format is timestamp,open,close,high,low,volume
        // So close is at index 2, not 4!
        const closes = [];
        for (const line of lines) {
            const parts = line.split(',');
            if (parts.length >= 5) {
                closes.push(parseFloat(parts[2])); // Close is column 2
            }
        }
        
        // Create different size datasets
        this.data['10k'] = new Float64Array(closes.slice(0, 10_000));
        this.data['100k'] = new Float64Array(closes.slice(0, 100_000));
        this.data['1M'] = new Float64Array(closes);
        
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
                const paramArray = this.prepareParams(params, data);
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
        
        const { allocFn, freeFn, computeFn, params, dualOutput } = indicatorConfig.fast;
        
        if (!this.wasm[allocFn] || !this.wasm[freeFn] || !this.wasm[computeFn]) {
            console.log(`  Fast API functions not found, skipping...`);
            return;
        }

        for (const [sizeName, data] of Object.entries(this.data)) {
            const benchName = `${indicatorKey}_fast_${sizeName}`;
            const len = data.length;
            
            // Pre-allocate buffers outside of benchmark
            const inPtr = this.wasm[allocFn](len);
            const outPtr = this.wasm[allocFn](len);
            const outPtr2 = dualOutput ? this.wasm[allocFn](len) : null;
            
            try {
                // Copy data once
                const inView = new Float64Array(this.wasm.__wasm.memory.buffer, inPtr, len);
                inView.set(data);
                
                const result = this.benchmarkFunction(() => {
                    const paramArray = this.prepareFastParams(params, inPtr, outPtr, len, dualOutput, outPtr2);
                    this.wasm[computeFn].apply(this.wasm, paramArray);
                }, benchName, {
                    dataSize: len,
                    api: 'fast',
                    indicator: indicatorKey
                });

                this.results[benchName] = result;
                this.printResult(result);
            } finally {
                this.wasm[freeFn](inPtr, len);
                this.wasm[freeFn](outPtr, len);
                if (dualOutput && outPtr2) {
                    this.wasm[freeFn](outPtr2, len);
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
        
        for (const [configName, batchConfig] of Object.entries(config)) {
            const benchName = `${indicatorKey}_batch_${configName}`;
            
            const result = this.benchmarkFunction(() => {
                const params = this.prepareBatchParams(indicatorKey, data, batchConfig);
                wasmFn.apply(this.wasm, params);
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
                totalCombinations *= periods;
            }
            if (batchConfig.offset_range) {
                const offsets = Math.floor((batchConfig.offset_range[1] - batchConfig.offset_range[0]) / batchConfig.offset_range[2]) + 1;
                totalCombinations *= offsets;
            }
            if (batchConfig.sigma_range) {
                const sigmas = Math.floor((batchConfig.sigma_range[1] - batchConfig.sigma_range[0]) / batchConfig.sigma_range[2]) + 1;
                totalCombinations *= sigmas;
            }
            if (batchConfig.fast_limit_range) {
                const fast_limits = Math.floor((batchConfig.fast_limit_range[1] - batchConfig.fast_limit_range[0]) / batchConfig.fast_limit_range[2]) + 1;
                totalCombinations *= fast_limits;
            }
            if (batchConfig.slow_limit_range) {
                const slow_limits = Math.floor((batchConfig.slow_limit_range[1] - batchConfig.slow_limit_range[0]) / batchConfig.slow_limit_range[2]) + 1;
                totalCombinations *= slow_limits;
            }
            console.log(`  Total combinations: ${totalCombinations}`);
        }
    }

    /**
     * Prepare parameters for safe API call
     */
    prepareParams(params, data) {
        // Start with data array
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
    prepareBatchParams(indicatorKey, data, batchConfig) {
        // Special handling for different indicators
        if (indicatorKey === 'mama') {
            // MAMA expects: data, fast_limit_start, fast_limit_end, fast_limit_step, slow_limit_start, slow_limit_end, slow_limit_step
            const fast = batchConfig.fast_limit_range || [0.5, 0.5, 0];
            const slow = batchConfig.slow_limit_range || [0.05, 0.05, 0];
            return [data, fast[0], fast[1], fast[2], slow[0], slow[1], slow[2]];
        } else if (indicatorKey === 'sqwma' || indicatorKey === 'fwma' || indicatorKey === 'hma' || indicatorKey === 'kama') {
            // These indicators expect: data, period_start, period_end, period_step
            const period = batchConfig.period_range;
            return [data, period[0], period[1], period[2]];
        } else if (indicatorKey === 'swma') {
            // SWMA uses the new unified batch API with serde config
            return [data, { period_range: batchConfig.period_range }];
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
    prepareFastParams(params, inPtr, outPtr, len, dualOutput = false, outPtr2 = null) {
        // Fast API typically takes: in_ptr, out_ptr, len, ...params
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