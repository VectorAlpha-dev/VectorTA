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


const CONFIG = {
    warmupTargetMs: 150,    
    sampleCount: 10,        
    minIterations: 10,      
    disableGC: true,        
};


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
        
        safe: {
            fn: 'frama_js',
            params: { window: 10, sc: 300, fc: 1 }
        },
        needsMultipleInputs: true,
        
        fast: {
            allocFn: 'frama_alloc',
            freeFn: 'frama_free',
            computeFn: 'frama_into',
            params: { window: 10, sc: 300, fc: 1 },
            needsMultipleInputs: true
        },
        
        batch: {
            fn: 'frama_batch',
            fastFn: 'frama_batch_into',
            config: {
                small: {
                    window_range: [8, 12, 2],      
                    sc_range: [200, 300, 100],     
                    fc_range: [1, 2, 1]            
                },
                medium: {
                    window_range: [6, 14, 2],      
                    sc_range: [100, 400, 100],    
                    fc_range: [1, 3, 1]            
                }
            }
        }
    },
    mom: {
        name: 'MOM',
        
        safe: {
            fn: 'mom_js',
            params: { period: 10 }
        },
        
        fast: {
            allocFn: 'mom_alloc',
            freeFn: 'mom_free',
            computeFn: 'mom_into',
            params: { period: 10 }
        },
        
        batch: {
            fn: 'mom_batch',
            fastFn: 'mom_batch_into',
            config: {
                small: {
                    period_range: [5, 15, 5]       
                },
                medium: {
                    period_range: [5, 50, 5]       
                }
            }
        }
    },
    pwma: {
        name: 'PWMA',
        
        safe: {
            fn: 'pwma_js',
            params: { period: 14 }
        },
        
        fast: {
            allocFn: 'pwma_alloc',
            freeFn: 'pwma_free',
            computeFn: 'pwma_into',
            params: { period: 14 }
        },
        
        batch: {
            fn: 'pwma_batch_js',
            fastFn: 'pwma_batch_into',
            config: {
                small: {
                    period_range: [5, 15, 2]       
                },
                medium: {
                    period_range: [5, 25, 2]       
                }
            }
        }
    },
    percentile_nearest_rank: {
        name: 'Percentile Nearest Rank',
        
        safe: {
            fn: 'percentile_nearest_rank_js',
            params: { length: 15, percentage: 50 }
        },
        
        fast: {
            allocFn: 'percentile_nearest_rank_alloc',
            freeFn: 'percentile_nearest_rank_free',
            computeFn: 'percentile_nearest_rank_into',
            params: { length: 15, percentage: 50 }
        },
        
        batch: {
            fn: 'percentile_nearest_rank_batch',
            config: {
                small: {
                    length_range: [10, 20, 5],       
                    percentage_range: [25, 75, 25]   
                },
                medium: {
                    length_range: [10, 30, 5],       
                    percentage_range: [10, 90, 20]   
                }
            }
        }
    },
    cg: {
        name: 'CG',
        
        safe: {
            fn: 'cg_js',
            params: { period: 10 }
        },
        
        fast: {
            allocFn: 'cg_alloc',
            freeFn: 'cg_free',
            computeFn: 'cg_into',
            params: { period: 10 }
        },
        
        batch: {
            fn: 'cg_batch',
            fastFn: 'cg_batch_into',
            config: {
                small: {
                    period_range: [5, 15, 2]       
                },
                medium: {
                    period_range: [5, 25, 2]       
                }
            }
        }
    },
    vidya: {
        name: 'VIDYA',
        
        safe: {
            fn: 'vidya_js',
            params: { short_period: 2, long_period: 5, alpha: 0.2 }
        },
        
        fast: {
            allocFn: 'vidya_alloc',
            freeFn: 'vidya_free',
            computeFn: 'vidya_into',
            params: { short_period: 2, long_period: 5, alpha: 0.2 }
        },
        
        batch: {
            fn: 'vidya_batch',
            fastFn: 'vidya_batch_into',
            config: {
                small: {
                    short_period_range: [2, 4, 1],     
                    long_period_range: [5, 7, 1],      
                    alpha_range: [0.1, 0.3, 0.1]       
                },
                medium: {
                    short_period_range: [2, 5, 1],     
                    long_period_range: [5, 10, 1],     
                    alpha_range: [0.1, 0.4, 0.1]       
                }
            }
        }
    },
    vosc: {
        name: 'VOSC',
        
        safe: {
            fn: 'vosc_js',
            params: { short_period: 2, long_period: 5 },
            inputs: ['volume']  
        },
        
        fast: {
            allocFn: 'vosc_alloc',
            freeFn: 'vosc_free',
            computeFn: 'vosc_into',
            params: { short_period: 2, long_period: 5 },
            inputs: ['volume']
        },
        
        batch: {
            fn: 'vosc_batch',
            config: {
                small: {
                    short_period_range: [2, 4, 1],   
                    long_period_range: [5, 7, 1]     
                },
                medium: {
                    short_period_range: [2, 10, 1],  
                    long_period_range: [10, 20, 2]   
                }
            },
            inputs: ['volume']  
        }
    },
    adxr: {
        name: 'ADXR',
        
        safe: {
            fn: 'adxr_js',
            params: { period: 14 },
            needsMultipleInputs: true
        },
        
        fast: {
            allocFn: 'adxr_alloc',
            freeFn: 'adxr_free',
            computeFn: 'adxr_into',
            params: { period: 14 },
            needsMultipleInputs: true
        },
        
        batch: {
            fn: 'adxr_batch',
            fastFn: 'adxr_batch_into',
            config: {
                small: {
                    period_range: [10, 20, 5]  
                },
                medium: {
                    period_range: [10, 30, 5]  
                }
            },
            needsMultipleInputs: true
        }
    },
    rocp: {
        name: 'ROCP',
        
        safe: {
            fn: 'rocp_js',
            params: { period: 10 }
        },
        
        fast: {
            allocFn: 'rocp_alloc',
            freeFn: 'rocp_free',
            computeFn: 'rocp_into',
            params: { period: 10 }
        },
        
        batch: {
            fn: 'rocp_batch',
            fastFn: 'rocp_batch_into',
            config: {
                small: {
                    period_range: [5, 15, 5]       
                },
                medium: {
                    period_range: [5, 50, 5]       
                }
            }
        }
    },
    alma: {
        name: 'ALMA',
        
        safe: {
            fn: 'alma_js',
            params: { period: 9, offset: 0.85, sigma: 6.0 }
        },
        
        fast: {
            allocFn: 'alma_alloc',
            freeFn: 'alma_free',
            computeFn: 'alma_into',
            params: { period: 9, offset: 0.85, sigma: 6.0 }
        },
        
        batch: {
            fn: 'alma_batch',
            config: {
                
                small: {
                    period_range: [5, 15, 5],      
                    offset_range: [0.7, 0.9, 0.1], 
                    sigma_range: [4.0, 8.0, 2.0]   
                    
                },
                medium: {
                    period_range: [5, 25, 4],      
                    offset_range: [0.5, 0.9, 0.1], 
                    sigma_range: [3.0, 9.0, 3.0]   
                    
                }
            },
            
            fastFn: 'alma_batch_into'
        }
    },
    obv: {
        name: 'OBV',
        
        safe: {
            fn: 'obv_js',
            params: {}, 
            needsMultipleInputs: true 
        },
        
        fast: {
            allocFn: 'obv_alloc',
            freeFn: 'obv_free',
            computeFn: 'obv_into',
            params: {},
            needsMultipleInputs: true
        },
        
        batch: {
            fn: 'obv_batch',
            config: {
                small: {}, 
                medium: {}
            },
            needsMultipleInputs: true
        }
    },
    otto: {
        name: 'OTT',
        
        safe: {
            fn: 'ott_js',
            params: { period: 2, percent: 1.4, ma_type: 'VAR' }
        },
        
        fast: {
            allocFn: 'ott_alloc',
            freeFn: 'ott_free',
            computeFn: 'ott_into',
            params: { period: 2, percent: 1.4, ma_type: 'VAR' }
        },
        
        batch: {
            fn: 'ott_batch',
            config: {
                small: {
                    period_range: [2, 10, 2],        
                    percent_range: [1.0, 2.0, 0.5],  
                    ma_type: 'VAR'                   
                },
                medium: {
                    period_range: [2, 20, 2],        
                    percent_range: [0.5, 3.0, 0.5],  
                    ma_type: 'VAR'                   
                }
            }
        }
    },
    ott: {
        name: 'OTT',
        
        safe: {
            fn: 'ott_js',
            params: { period: 2, percent: 1.4, ma_type: 'VAR' }
        },
        
        fast: {
            allocFn: 'ott_alloc',
            freeFn: 'ott_free',
            computeFn: 'ott_into',
            params: { period: 2, percent: 1.4, ma_type: 'VAR' }
        },
        
        batch: {
            fn: 'ott_batch',
            fastFn: 'ott_batch_into',
            config: {
                small: {
                    period_range: [2, 4, 1],        
                    percent_range: [1.0, 2.0, 0.5], 
                    ma_types: ['VAR', 'SMA', 'EMA'] 
                },
                medium: {
                    period_range: [2, 6, 1],        
                    percent_range: [0.5, 2.5, 0.5], 
                    ma_types: ['VAR', 'SMA', 'EMA', 'WMA', 'ZLEMA'] 
                }
            }
        }
    },
    qstick: {
        name: 'QSTICK',
        
        safe: {
            fn: 'qstick_js',
            params: { period: 5 },
            needsMultipleInputs: true 
        },
        
        fast: {
            allocFn: 'qstick_alloc',
            freeFn: 'qstick_free',
            computeFn: 'qstick_into',
            params: { period: 5 },
            needsMultipleInputs: true
        },
        
        batch: {
            fn: 'qstick_batch',
            config: {
                small: {
                    period_range: [5, 20, 5]  
                },
                medium: {
                    period_range: [5, 25, 5]  
                }
            },
            needsMultipleInputs: true
        }
    },
    damiani_volatmeter: {
        name: 'Damiani Volatmeter',
        
        safe: {
            fn: 'damiani_volatmeter_js',
            params: { vis_atr: 13, vis_std: 20, sed_atr: 40, sed_std: 100, threshold: 1.4 }
        },
        
        fast: {
            allocFn: 'damiani_volatmeter_alloc',
            freeFn: 'damiani_volatmeter_free',
            computeFn: 'damiani_volatmeter_into',
            params: { vis_atr: 13, vis_std: 20, sed_atr: 40, sed_std: 100, threshold: 1.4 },
            dualOutput: true  
        },
        
        batch: {
            fn: 'damiani_volatmeter_batch',
            config: {
                small: {
                    vis_atr_range: [10, 20, 5],      
                    vis_std_range: [15, 25, 5],      
                    sed_atr_range: [40, 40, 0],      
                    sed_std_range: [100, 100, 0],    
                    threshold_range: [1.4, 1.4, 0.0] 
                },
                medium: {
                    vis_atr_range: [10, 30, 5],      
                    vis_std_range: [15, 35, 5],      
                    sed_atr_range: [30, 50, 10],     
                    sed_std_range: [80, 120, 20],    
                    threshold_range: [1.0, 2.0, 0.5] 
                }
            },
            fastFn: 'damiani_volatmeter_batch_into'
        }
    },
    aroon: {
        name: 'Aroon',
        needsMultipleInputs: true,  
        
        safe: {
            fn: 'aroon_js',
            params: { length: 14 },
            needsMultipleInputs: true
        },
        
        fast: {
            allocFn: 'aroon_alloc',
            freeFn: 'aroon_free',
            computeFn: 'aroon_into',
            params: { length: 14 },
            needsMultipleInputs: true,
            dualOutput: true  
        },
        
        batch: {
            fn: 'aroon_batch',
            config: {
                small: {
                    length_range: [10, 20, 5]       
                },
                medium: {
                    length_range: [5, 25, 5]        
                }
            },
            
            fastFn: 'aroon_batch_into',
            dualOutput: true
        }
    },
    mean_ad: {
        name: 'Mean Absolute Deviation',
        
        safe: {
            fn: 'mean_ad_js',
            params: { period: 5 }
        },
        
        fast: {
            allocFn: 'mean_ad_alloc',
            freeFn: 'mean_ad_free',
            computeFn: 'mean_ad_into',
            params: { period: 5 }
        },
        
        batch: {
            fn: 'mean_ad_batch',
            config: {
                small: {
                    period_range: [5, 15, 5]       
                },
                medium: {
                    period_range: [5, 25, 5]       
                }
            }
        }
    },
    macz: {
        name: 'MACZ',
        
        safe: {
            fn: 'macz_js',
            params: { 
                fast_length: 12, 
                slow_length: 25, 
                signal_length: 9,
                lengthz: 20,
                length_stdev: 25,
                a: 1.0,
                b: 1.0,
                use_lag: false,
                gamma: 0.02
            }
        },
        
        fast: {
            allocFn: 'macz_alloc',
            freeFn: 'macz_free',
            computeFn: 'macz_into',
            params: { 
                fast_length: 12, 
                slow_length: 25, 
                signal_length: 9,
                lengthz: 20,
                length_stdev: 25,
                a: 1.0,
                b: 1.0,
                use_lag: false,
                gamma: 0.02
            }
        },
        
        batch: {
            fn: 'macz_batch',
            config: {
                small: {
                    fast_length_range: [10, 14, 2],
                    slow_length_range: [20, 30, 5],
                    signal_length_range: [7, 11, 2],
                    lengthz_range: [18, 22, 2],
                    length_stdev_range: [20, 30, 5],
                    a_range: [0.8, 1.2, 0.2],
                    b_range: [0.8, 1.2, 0.2],
                    use_lag_range: [false, false, false],
                    gamma_range: [0.01, 0.03, 0.01]
                },
                medium: {
                    fast_length_range: [8, 16, 2],
                    slow_length_range: [20, 35, 5],
                    signal_length_range: [5, 13, 2],
                    lengthz_range: [15, 25, 5],
                    length_stdev_range: [20, 35, 5],
                    a_range: [0.5, 1.5, 0.25],
                    b_range: [0.5, 1.5, 0.25],
                    use_lag_range: [false, false, false],
                    gamma_range: [0.01, 0.05, 0.01]
                }
            }
        }
    },
    bollinger_bands: {
        name: 'Bollinger Bands',
        
        safe: {
            fn: 'bollinger_bands_js',
            params: { period: 20, devup: 2.0, devdn: 2.0, matype: 'sma', devtype: 0 },
            
            multiOutput: 3
        },
        
        fast: {
            allocFn: 'bollinger_bands_alloc',
            freeFn: 'bollinger_bands_free',
            computeFn: 'bollinger_bands_into',
            params: { period: 20, devup: 2.0, devdn: 2.0, matype: 'sma', devtype: 0 },
            
            multiOutput: 3
        },
        
        batch: {
            fn: 'bollinger_bands_batch',
            config: {
                small: {
                    period_range: [10, 30, 10],     
                    devup_range: [1.0, 3.0, 1.0],   
                    devdn_range: [2.0, 2.0, 0.0],   
                    matype: 'sma',
                    devtype: 0
                    
                },
                medium: {
                    period_range: [10, 50, 10],     
                    devup_range: [1.0, 3.0, 0.5],   
                    devdn_range: [1.0, 3.0, 0.5],   
                    matype: 'sma',
                    devtype: 0
                    
                }
            }
        }
    },
    bop: {
        name: 'BOP',
        needsMultipleInputs: true,  
        
        safe: {
            fn: 'bop_js',
            needsMultipleInputs: true
        },
        
        fast: {
            allocFn: 'bop_alloc',
            freeFn: 'bop_free',
            computeFn: 'bop_into',
            needsMultipleInputs: true
        },
        
        batch: {
            fn: 'bop_batch_js',
            fastFn: 'bop_batch_into',
            needsMultipleInputs: true
        }
    },
    vlma: {
        name: 'VLMA',
        
        safe: {
            fn: 'vlma_js',
            params: { min_period: 5, max_period: 50, matype: 'sma', devtype: 0 }
        },
        
        fast: {
            allocFn: 'vlma_alloc',
            freeFn: 'vlma_free',
            computeFn: 'vlma_into',
            params: { min_period: 5, max_period: 50, matype: 'sma', devtype: 0 }
        },
        
        batch: {
            fn: 'vlma_batch',
            config: {
                small: {
                    min_period_range: [5, 15, 5],    
                    max_period_range: [30, 50, 10],  
                    devtype_range: [0, 2, 1],        
                    matype: 'sma'
                    
                },
                medium: {
                    min_period_range: [5, 25, 5],    
                    max_period_range: [30, 60, 10],  
                    devtype_range: [0, 2, 1],        
                    matype: 'ema'
                    
                }
            },
            fastFn: 'vlma_batch_into'
        }
    },
    keltner: {
        name: 'Keltner Channels',
        needsMultipleInputs: true,  
        hasMultipleOutputs: 3,      
        
        safe: {
            fn: 'keltner_js',
            params: { period: 20, multiplier: 2.0, ma_type: 'ema' }
        },
        
        fast: {
            allocFn: 'keltner_alloc',
            freeFn: 'keltner_free',
            computeFn: 'keltner_into',
            params: { period: 20, multiplier: 2.0, ma_type: 'ema' },
            needsMultipleInputs: true,
            hasMultipleOutputs: 3
        },
        
        batch: {
            fn: 'keltner_batch',
            config: {
                small: {
                    period_range: [10, 30, 10],        
                    multiplier_range: [1.0, 3.0, 1.0], 
                    ma_type: 'ema'
                    
                },
                medium: {
                    period_range: [10, 50, 10],        
                    multiplier_range: [1.0, 4.0, 0.5], 
                    ma_type: 'ema'
                    
                }
            }
        }
    },
    fisher: {
        name: 'Fisher Transform',
        needsMultipleInputs: true,  
        hasMultipleOutputs: 2,      
        
        safe: {
            fn: 'fisher_js',
            params: { period: 9 }
        },
        
        fast: {
            allocFn: 'fisher_alloc',
            freeFn: 'fisher_free',
            computeFn: 'fisher_into',
            params: { period: 9 },
            needsMultipleInputs: true,
            hasMultipleOutputs: 2
        },
        
        batch: {
            fn: 'fisher_batch',
            fastFn: 'fisher_batch_into',
            config: {
                small: {
                    period_range: [5, 15, 5]       
                },
                medium: {
                    period_range: [5, 25, 4]       
                }
            },
            needsMultipleInputs: true,
            hasMultipleOutputs: 2
        }
    },
    fvg_trailing_stop: {
        name: 'FVG Trailing Stop',
        needsMultipleInputs: true,  
        hasMultipleOutputs: 4,      
        
        safe: {
            fn: 'fvgTrailingStop',
            params: { unmitigated_fvg_lookback: 5, smoothing_length: 9, reset_on_cross: false }
        },
        
        fast: {
            allocFn: 'fvgTrailingStopAlloc',
            freeFn: 'fvgTrailingStopFree',
            computeFn: 'fvgTrailingStopZeroCopy',
            params: { unmitigated_fvg_lookback: 5, smoothing_length: 9, reset_on_cross: false },
            needsMultipleInputs: true,
            hasMultipleOutputs: 4
        },
        
        batch: {
            fn: 'fvgTrailingStopBatch',
            config: {
                small: {
                    lookback_range: [3, 5, 1],       
                    smoothing_range: [1, 3, 1],      
                    reset_include_false: true,
                    reset_include_true: true         
                },
                medium: {
                    lookback_range: [3, 7, 1],       
                    smoothing_range: [1, 5, 1],      
                    reset_include_false: true,
                    reset_include_true: false        
                }
            },
            needsMultipleInputs: true,
            hasMultipleOutputs: 4
        }
    },
    ao: {
        name: 'AO',
        needsMultipleInputs: true,  
        
        safe: {
            fn: 'ao_js',
            params: { short_period: 5, long_period: 34 }
        },
        
        fast: {
            allocFn: 'ao_alloc',
            freeFn: 'ao_free',
            computeFn: 'ao_into',
            params: { short_period: 5, long_period: 34 },
            needsMultipleInputs: true
        },
        
        batch: {
            fn: 'ao_batch',
            fastFn: 'ao_batch_into',
            config: {
                small: {
                    short_period_range: [3, 7, 2],   
                    long_period_range: [20, 40, 10]  
                },
                medium: {
                    short_period_range: [3, 11, 2],  
                    long_period_range: [20, 50, 10]  
                }
            },
            needsMultipleInputs: true
        }
    },
    adosc: {
        name: 'ADOSC',
        needsMultipleInputs: true,  
        
        safe: {
            fn: 'adosc_js',
            params: { short_period: 3, long_period: 10 }
        },
        
        fast: {
            allocFn: 'adosc_alloc',
            freeFn: 'adosc_free',
            computeFn: 'adosc_into',
            params: { short_period: 3, long_period: 10 },
            needsMultipleInputs: true
        },
        
        batch: {
            fn: 'adosc_batch',
            config: {
                small: {
                    short_period_range: [2, 4, 1],   
                    long_period_range: [8, 12, 2]    
                },
                medium: {
                    short_period_range: [2, 6, 1],   
                    long_period_range: [8, 16, 2]    
                }
            }
        }
    },
    kvo: {
        name: 'KVO',
        needsMultipleInputs: true,  
        
        safe: {
            fn: 'kvo_js',
            params: { short_period: 2, long_period: 5 }
        },
        
        fast: {
            allocFn: 'kvo_alloc',
            freeFn: 'kvo_free',
            computeFn: 'kvo_into',
            params: { short_period: 2, long_period: 5 },
            needsMultipleInputs: true
        },
        
        batch: {
            fn: 'kvo_batch',
            fastFn: 'kvo_batch_into',
            config: {
                small: {
                    short_period_range: [2, 4, 1],   
                    long_period_range: [5, 7, 1]     
                },
                medium: {
                    short_period_range: [2, 6, 1],   
                    long_period_range: [5, 15, 2]    
                }
            }
        }
    },
    chande: {
        name: 'Chande',
        needsMultipleInputs: true,  
        
        safe: {
            fn: 'chande_js',
            params: { period: 22, mult: 3.0, direction: 'long' }
        },
        
        fast: {
            allocFn: 'chande_alloc',
            freeFn: 'chande_free',
            computeFn: 'chande_into',
            params: { period: 22, mult: 3.0, direction: 'long' },
            needsMultipleInputs: true
        },
        
        batch: {
            fn: 'chande_batch_js',
            fastFn: 'chande_batch_into',
            config: {
                small: {
                    period_range: [15, 25, 5],      
                    mult_range: [2.0, 4.0, 1.0],    
                    direction: 'long'               
                },
                medium: {
                    period_range: [10, 30, 5],      
                    mult_range: [2.0, 5.0, 0.5],    
                    direction: 'short'              
                }
            }
        }
    },
    devstop: {
        name: 'DevStop',
        needsMultipleInputs: true,  
        
        safe: {
            fn: 'devstop_js',
            params: { period: 20, mult: 2.0, devtype: 0, direction: 'long', ma_type: 'sma' }
        },
        
        fast: {
            allocFn: 'devstop_alloc',
            freeFn: 'devstop_free',
            computeFn: 'devstop_into',
            params: { period: 20, mult: 2.0, devtype: 0, direction: 'long', ma_type: 'sma' },
            needsMultipleInputs: true
        },
        
        batch: {
            fn: 'devstop_batch_unified_js',
            config: {
                small: {
                    period_range: [15, 25, 5],      
                    mult_range: [1.5, 2.5, 0.5],    
                    devtype_range: [0, 2, 1]        
                },
                medium: {
                    period_range: [10, 30, 5],      
                    mult_range: [1.0, 3.0, 0.5],    
                    devtype_range: [0, 2, 1]        
                }
            },
            needsMultipleInputs: true
        }
    },
    chandelier_exit: {
        name: 'Chandelier Exit',
        needsMultipleInputs: true,  
        
        safe: {
            fn: 'ce_js',
            params: { period: 22, mult: 3.0, use_close: true }
        },
        
        fast: {
            allocFn: 'ce_alloc',
            freeFn: 'ce_free',
            computeFn: 'ce_into',
            params: { period: 22, mult: 3.0, use_close: true },
            needsMultipleInputs: true,
            dualOutput: true  
        },
        
        batch: {
            fn: 'ce_batch',
            fastFn: 'ce_batch_into',
            config: {
                small: {
                    period_range: [20, 24, 2],      
                    mult_range: [2.5, 3.5, 0.5]     
                },
                medium: {
                    period_range: [15, 30, 5],      
                    mult_range: [2.0, 4.0, 0.5]     
                }
            },
            needsMultipleInputs: true,
            dualOutput: true
        }
    },
    acosc: {
        name: 'ACOSC',
        needsMultipleInputs: true,  
        
        safe: {
            fn: 'acosc_js',
            params: {}  
        },
        
        fast: {
            allocFn: 'acosc_alloc',
            freeFn: 'acosc_free',
            computeFn: 'acosc_into',
            params: {},
            needsMultipleInputs: true,
            
            dualOutput: true  
        },
        
        batch: {
            fn: 'acosc_batch',
            config: {
                
                small: {},
                medium: {}
            }
        }
    },
    marketefi: {
        name: 'MarketEFI',
        needsMultipleInputs: true,  
        
        safe: {
            fn: 'marketefi_js',
            params: {}  
        },
        
        fast: {
            allocFn: 'marketefi_alloc',
            freeFn: 'marketefi_free',
            computeFn: 'marketefi_into',
            params: {},
            needsMultipleInputs: true
        },
        
        batch: {
            fn: 'marketefi_batch',
            config: {
                
                small: {},
                medium: {}
            }
        }
    },
    cci: {
        name: 'CCI',
        
        safe: {
            fn: 'cci_js',
            params: { period: 14 }
        },
        
        fast: {
            allocFn: 'cci_alloc',
            freeFn: 'cci_free',
            computeFn: 'cci_into',
            params: { period: 14 }
        },
        
        batch: {
            fn: 'cci_batch_js',
            config: {
                small: {
                    period_range: [10, 20, 5]  
                },
                medium: {
                    period_range: [10, 30, 5]  
                }
            }
        }
    },
    medprice: {
        name: 'MEDPRICE',
        needsMultipleInputs: true,  
        
        safe: {
            fn: 'medprice_js',
            params: {}  
        },
        
        fast: {
            allocFn: 'medprice_alloc',
            freeFn: 'medprice_free',
            computeFn: 'medprice_into',
            params: {},
            needsMultipleInputs: true
        },
        
        batch: {
            fn: 'medprice_batch',
            config: {
                
                small: {},
                medium: {}
            }
        }
    },
    vpwma: {
        name: 'VPWMA',
        
        safe: {
            fn: 'vpwma_js',
            params: { period: 14, power: 0.382 }
        },
        
        fast: {
            allocFn: 'vpwma_alloc',
            freeFn: 'vpwma_free',
            computeFn: 'vpwma_into',
            params: { period: 14, power: 0.382 }
        },
        
        batch: {
            fn: 'vpwma_batch_js',
            config: {
                small: {
                    period_range: [10, 20, 5],      
                    power_range: [0.2, 0.6, 0.2]    
                },
                medium: {
                    period_range: [10, 30, 5],      
                    power_range: [0.1, 0.9, 0.2]    
                }
            },
            
            fastFn: 'vpwma_batch_into'
        }
    },
    edcf: {
        name: 'EDCF',
        
        safe: {
            fn: 'edcf_js',
            params: { period: 15 }
        },
        
        fast: {
            allocFn: 'edcf_alloc',
            freeFn: 'edcf_free',
            computeFn: 'edcf_into',
            params: { period: 15 }
        }
        
    },
    ehma: {
        name: 'EHMA',
        
        safe: {
            fn: 'ehma_js',
            params: { period: 14 }
        },
        
        fast: {
            allocFn: 'ehma_alloc',
            freeFn: 'ehma_free',
            computeFn: 'ehma_into',
            params: { period: 14 }
        },
        
        batch: {
            fn: 'ehma_batch_unified_js',
            config: {
                small: {
                    periods: [14, 14, 0]
                },
                medium: {
                    periods: [10, 30, 2]
                },
                large: {
                    periods: [5, 50, 5]
                }
            },
            fastFn: 'ehma_batch_into'
        }
    },
    eri: {
        name: 'ERI',
        needsMultipleInputs: true,  
        
        safe: {
            fn: 'eri_js',
            params: { period: 13, ma_type: 'ema' }
        },
        
        fast: {
            allocFn: 'eri_alloc',
            freeFn: 'eri_free',
            computeFn: 'eri_into',
            params: { period: 13, ma_type: 'ema' },
            numOutputs: 2  
        },
        
        batch: {
            fn: 'eri_batch',
            config: {
                small: {
                    period_range: [10, 20, 5],  
                    ma_type: 'ema'
                },
                medium: {
                    period_range: [10, 30, 5],  
                    ma_type: 'ema'
                }
            }
        }
    },
    highpass: {
        name: 'HighPass',
        
        safe: {
            fn: 'highpass_js',
            params: { period: 48 }
        },
        
        fast: {
            allocFn: 'highpass_alloc',
            freeFn: 'highpass_free',
            computeFn: 'highpass_into',
            params: { period: 48 }
        },
        
        batch: {
            fn: 'highpass_batch',
            config: {
                small: {
                    period_range: [30, 60, 10]  
                },
                medium: {
                    period_range: [20, 80, 10]  
                }
            },
            
            fastFn: 'highpass_batch_into'
        }
    },
    jsa: {
        name: 'JSA',
        
        safe: {
            fn: 'jsa_js',
            params: { period: 30 }
        },
        
        fast: {
            allocFn: 'jsa_alloc',
            freeFn: 'jsa_free',
            computeFn: 'jsa_fast',
            params: { period: 30 }
        },
        
        batch: {
            fn: 'jsa_batch',
            fastFn: 'jsa_batch_into',
            config: {
                small: {
                    period_range: [10, 40, 10]  
                },
                medium: {
                    period_range: [10, 50, 5]   
                }
            }
        }
    },
    linearreg_slope: {
        name: 'LinearRegSlope',
        
        safe: {
            fn: 'linearreg_slope_js',
            params: { period: 14 }
        },
        
        fast: {
            allocFn: 'linearreg_slope_alloc',
            freeFn: 'linearreg_slope_free',
            computeFn: 'linearreg_slope_into',
            params: { period: 14 }
        },
        
        batch: {
            fn: 'linearreg_slope_batch',
            config: {
                small: {
                    period_range: [10, 20, 5]  
                },
                medium: {
                    period_range: [5, 25, 5]   
                }
            }
        }
    },
    maaq: {
        name: 'MAAQ',
        
        safe: {
            fn: 'maaq_js',
            params: { period: 11, fast_period: 2, slow_period: 30 }
        },
        
        fast: {
            allocFn: 'maaq_alloc',
            freeFn: 'maaq_free',
            computeFn: 'maaq_into',
            params: { period: 11, fast_period: 2, slow_period: 30 }
        },
        
        batch: {
            fn: 'maaq_batch_js',
            fastFn: 'maaq_batch_into',
            config: {
                small: {
                    period_range: [10, 20, 5],       
                    fast_period_range: [2, 4, 1],    
                    slow_period_range: [20, 40, 10]  
                    
                },
                medium: {
                    period_range: [10, 30, 5],       
                    fast_period_range: [2, 6, 2],    
                    slow_period_range: [20, 50, 10]  
                    
                }
            }
        }
    },
    smma: {
        name: 'SMMA',
        
        safe: {
            fn: 'smma',
            params: { period: 7 }
        },
        
        fast: {
            allocFn: 'smma_alloc',
            freeFn: 'smma_free',
            computeFn: 'smma_into',
            params: { period: 7 }
        },
        
        batch: {
            fn: 'smma_batch_new',
            config: {
                small: {
                    period_range: [5, 15, 5]  
                },
                medium: {
                    period_range: [5, 25, 5]  
                }
            },
            
            fastFn: 'smma_batch_into'
        }
    },
    supersmoother: {
        name: 'SuperSmoother',
        
        safe: {
            fn: 'supersmoother_js',
            params: { period: 14 }
        },
        
        fast: {
            allocFn: 'supersmoother_alloc',
            freeFn: 'supersmoother_free',
            computeFn: 'supersmoother_into',
            params: { period: 14 }
        },
        
        batch: {
            fn: 'supersmoother_batch_js',  
            fastFn: 'supersmoother_batch_into',
            config: {
                small: {
                    period_range: [5, 15, 2]       
                },
                medium: {
                    period_range: [5, 25, 2]       
                }
            }
        }
    },
    ehlers_ecema: {
        name: 'Ehlers ECEMA',
        
        safe: {
            fn: 'ehlers_ecema_js',
            params: { length: 20, gain_limit: 50 }
        },
        
        fast: {
            allocFn: 'ehlers_ecema_alloc',
            freeFn: 'ehlers_ecema_free',
            computeFn: 'ehlers_ecema_into',
            params: { length: 20, gain_limit: 50 }
        },
        
        batch: {
            fn: 'ehlers_ecema_batch',
            config: {
                small: {
                    length_range: [10, 30, 10],        
                    gain_limit_range: [30, 60, 15]     
                    
                },
                medium: {
                    length_range: [10, 30, 5],         
                    gain_limit_range: [30, 60, 10]     
                    
                }
            },
            
            fastFn: 'ehlers_ecema_batch_into'
        }
    },
    ehlers_itrend: {
        name: 'Ehlers Instantaneous Trendline',
        
        safe: {
            fn: 'ehlers_itrend_js',
            params: { warmup_bars: 20, max_dc_period: 48 }
        },
        
        fast: {
            allocFn: 'ehlers_itrend_alloc',
            freeFn: 'ehlers_itrend_free',
            computeFn: 'ehlers_itrend_into',
            params: { warmup_bars: 20, max_dc_period: 48 }
        },
        
        batch: {
            fn: 'ehlers_itrend_batch',
            config: {
                
                small: {
                    warmup_bars_range: [10, 20, 5],      
                    max_dc_period_range: [40, 50, 5]     
                    
                },
                medium: {
                    warmup_bars_range: [10, 30, 5],      
                    max_dc_period_range: [30, 60, 10]    
                    
                }
            }
        }
    },
    ehlers_kama: {
        name: 'Ehlers KAMA',
        
        safe: {
            fn: 'ehlers_kama_js',
            params: { period: 14 }
        },
        
        fast: {
            allocFn: 'ehlers_kama_alloc',
            freeFn: 'ehlers_kama_free',
            computeFn: 'ehlers_kama_into',
            params: { period: 14 }
        },
        
        batch: {
            fn: 'ehlers_kama_batch',
            fastFn: 'ehlers_kama_batch_into',
            config: {
                small: {
                    period_range: [5, 20, 3]       
                },
                medium: {
                    period_range: [5, 50, 5]       
                }
            }
        }
    },
    ehlers_pma: {
        name: 'Ehlers PMA',
        
        safe: {
            fn: 'ehlers_pma',
            params: {} 
        },
        
        fast: {
            allocFn: 'ehlers_pma_alloc',
            freeFn: 'ehlers_pma_free',
            computeFn: 'ehlers_pma_into',
            params: {}, 
            dualOutput: true  
        }
        
    },
    fwma: {
        name: 'FWMA',
        
        safe: {
            fn: 'fwma_js',
            params: { period: 5 }
        },
        
        fast: {
            allocFn: 'fwma_alloc',
            freeFn: 'fwma_free',
            computeFn: 'fwma_into',
            params: { period: 5 }
        },
        
        batch: {
            fn: 'fwma_batch_js',
            config: {
                
                small: {
                    period_range: [3, 15, 3]       
                    
                },
                medium: {
                    period_range: [3, 30, 3]       
                    
                }
            },
            
            fastFn: 'fwma_batch_into'
        }
    },
    hma: {
        name: 'Hull Moving Average (HMA)',
        
        safe: {
            fn: 'hma_js',
            params: { period: 5 }
        },
        
        fast: {
            allocFn: 'hma_alloc',
            freeFn: 'hma_free',
            computeFn: 'hma_into',
            params: { period: 5 }
        },
        
        batch: {
            fn: 'hma_batch_js',  
            config: {
                small: {
                    period_range: [5, 15, 5]       
                },
                medium: {
                    period_range: [5, 25, 5]       
                }
            },
            
            fastFn: 'hma_batch_into'
        }
    },
    kama: {
        name: 'KAMA',
        
        safe: {
            fn: 'kama_js',
            params: { period: 30 }
        },
        
        fast: {
            allocFn: 'kama_alloc',
            freeFn: 'kama_free',
            computeFn: 'kama_into',
            params: { period: 30 }
        },
        
        batch: {
            fn: 'kama_batch_js',
            config: {
                small: {
                    period_range: [10, 30, 10]  
                },
                medium: {
                    period_range: [10, 50, 5]   
                }
            },
            fastFn: 'kama_batch_into'
        }
    },
    kdj: {
        name: 'KDJ (Stochastic with J line)',
        needsMultipleInputs: true,  
        
        safe: {
            fn: 'kdj_js',
            params: { 
                fast_k_period: 9, 
                slow_k_period: 3, 
                slow_k_ma_type: "sma", 
                slow_d_period: 3, 
                slow_d_ma_type: "sma" 
            },
            multipleOutputs: 3  
        },
        
        fast: {
            allocFn: 'kdj_alloc',
            freeFn: 'kdj_free',
            computeFn: 'kdj_into',
            params: { 
                fast_k_period: 9, 
                slow_k_period: 3, 
                slow_k_ma_type: "sma", 
                slow_d_period: 3, 
                slow_d_ma_type: "sma" 
            },
            needsMultipleInputs: true,
            multipleOutputs: 3  
        },
        
        batch: {
            fn: 'kdj_batch',
            config: {
                small: {
                    fast_k_period_range: [5, 15, 5],      
                    slow_k_period_range: [3, 3, 0],       
                    slow_k_ma_type: "sma",
                    slow_d_period_range: [3, 3, 0],       
                    slow_d_ma_type: "sma"
                    
                },
                medium: {
                    fast_k_period_range: [5, 25, 5],      
                    slow_k_period_range: [2, 4, 1],       
                    slow_k_ma_type: "sma",
                    slow_d_period_range: [2, 4, 1],       
                    slow_d_ma_type: "sma"
                    
                }
            },
            
            fastFn: 'kdj_batch_into',
            needsMultipleInputs: true,
            multipleOutputs: 3
        }
    },
    fosc: {
        name: 'FOSC (Forecast Oscillator)',
        
        safe: {
            fn: 'fosc_js',
            params: { period: 5 }
        },
        
        fast: {
            allocFn: 'fosc_alloc',
            freeFn: 'fosc_free',
            computeFn: 'fosc_into',
            params: { period: 5 }
        },
        
        batch: {
            fn: 'fosc_batch',
            config: {
                small: {
                    period_range: [3, 10, 1]  
                },
                medium: {
                    period_range: [3, 20, 1]  
                }
            },
            fastFn: 'fosc_batch_into'
        }
    },
    sqwma: {
        name: 'SQWMA (Square Weighted Moving Average)',
        
        safe: {
            fn: 'sqwma_js',
            params: { period: 14 }
        },
        
        fast: {
            allocFn: 'sqwma_alloc',
            freeFn: 'sqwma_free',
            computeFn: 'sqwma_into',
            params: { period: 14 }
        },
        
        batch: {
            fn: 'sqwma_batch_js',
            config: {
                small: {
                    period_range: [5, 20, 5]    
                },
                medium: {
                    period_range: [5, 30, 5]    
                }
            },
            fastFn: 'sqwma_batch_into'
        }
    },
    mama: {
        name: 'MAMA (MESA Adaptive Moving Average)',
        
        safe: {
            fn: 'mama_js',
            params: { fast_limit: 0.5, slow_limit: 0.05 }
        },
        
        fast: {
            allocFn: 'mama_alloc',
            freeFn: 'mama_free',
            computeFn: 'mama_into',
            params: { fast_limit: 0.5, slow_limit: 0.05 },
            
            dualOutput: true
        },
        
        batch: {
            fn: 'mama_batch_js',
            config: {
                small: {
                    fast_limit_range: [0.3, 0.7, 0.2],  
                    slow_limit_range: [0.03, 0.07, 0.02] 
                    
                },
                medium: {
                    fast_limit_range: [0.2, 0.8, 0.1],  
                    slow_limit_range: [0.02, 0.08, 0.01] 
                    
                }
            },
            
            metadataFn: 'mama_batch_metadata_js',
            rowsColsFn: 'mama_batch_rows_cols_js',
            
            fastFn: 'mama_batch_into',
            dualOutput: true
        }
    },
    lpc: {
        name: 'Linear Prediction Central',
        needsMultipleInputs: true,  
        
        safe: {
            fn: 'lpc_js',
            params: { 
                cutoff_type: 'adaptive',
                fixed_period: 30,
                cycle_mult: 1.5,
                tr_mult: 1.0,
                max_cycle_limit: 60
            }
        },
        
        fast: {
            allocFn: 'lpc_alloc',
            freeFn: 'lpc_free',
            computeFn: 'lpc_into',
            params: { 
                cutoff_type: 'adaptive',
                fixed_period: 30,
                cycle_mult: 1.5,
                tr_mult: 1.0,
                max_cycle_limit: 60
            },
            needsMultipleInputs: true,
            tripleOutput: true  
        },
        
        batch: {
            fn: 'lpc_batch',
            fastFn: 'lpc_batch_into',
            config: {
                small: {
                    fixed_period_range: [20, 40, 10],    
                    cycle_mult_range: [1.0, 2.0, 0.5],   
                    tr_mult_range: [1.0, 2.0, 0.5]       
                    
                },
                medium: {
                    fixed_period_range: [20, 60, 5],     
                    cycle_mult_range: [1.0, 2.5, 0.25],  
                    tr_mult_range: [0.5, 2.0, 0.25]      
                    
                }
            }
        }
    },
    mass: {
        name: 'Mass Index',
        needsMultipleInputs: true,  
        
        safe: {
            fn: 'mass_js',
            params: { period: 5 }
        },
        
        fast: {
            allocFn: 'mass_alloc',
            freeFn: 'mass_free',
            computeFn: 'mass_into',
            params: { period: 5 },
            needsMultipleInputs: true
        },
        
        batch: {
            fn: 'mass_batch',
            fastFn: 'mass_batch_into',
            config: {
                small: {
                    period_range: [5, 25, 5]       
                },
                medium: {
                    period_range: [5, 30, 2]       
                }
            }
        }
    },
    midprice: {
        name: 'Midprice',
        needsMultipleInputs: true,  
        
        safe: {
            fn: 'midprice_js',
            params: { period: 14 }
        },
        
        fast: {
            allocFn: 'midprice_alloc',
            freeFn: 'midprice_free',
            computeFn: 'midprice_into',
            params: { period: 14 },
            needsMultipleInputs: true
        },
        
        batch: {
            fn: 'midprice_batch',
            fastFn: 'midprice_batch_into',
            config: {
                small: {
                    period_range: [10, 20, 5]      
                },
                medium: {
                    period_range: [5, 25, 5]       
                }
            }
        }
    },
    medium_ad: {
        name: 'Medium AD',
        
        safe: {
            fn: 'medium_ad_js',
            params: { period: 5 }
        },
        
        fast: {
            allocFn: 'medium_ad_alloc',
            freeFn: 'medium_ad_free',
            computeFn: 'medium_ad_into',
            params: { period: 5 }
        },
        
        batch: {
            fn: 'medium_ad_batch',
            config: {
                small: {
                    period_range: [3, 7, 1]       
                },
                medium: {
                    period_range: [3, 15, 1]      
                }
            },
            
            fastFn: 'medium_ad_batch_into'
        }
    },
    minmax: {
        name: 'MinMax',
        needsMultipleInputs: true,  
        hasMultipleOutputs: 4,      
        
        safe: {
            fn: 'minmax_js',
            params: { order: 3 }
        },
        
        fast: {
            allocFn: 'minmax_alloc',
            freeFn: 'minmax_free',
            computeFn: 'minmax_into',
            params: { order: 3 },
            needsMultipleInputs: true,
            hasMultipleOutputs: 4
        },
        
        batch: {
            fn: 'minmax_batch',
            config: {
                small: {
                    order_range: [2, 5, 1]       
                },
                medium: {
                    order_range: [3, 20, 1]      
                }
            },
            needsMultipleInputs: true,
            hasMultipleOutputs: 4,
            
            fastFn: 'minmax_batch_into'
        }
    },
    mod_god_mode: {
        name: 'Modified God Mode',
        
        safe: {
            fn: 'mod_god_mode',
            params: { 
                n1: 17, 
                n2: 6, 
                n3: 4, 
                mode: 'tradition_mg', 
                use_volume: true 
            },
            inputs: ['high', 'low', 'close', 'volume']  
        },
        
        fast: {
            allocFn: 'mod_god_mode_alloc',
            freeFn: 'mod_god_mode_free',
            computeFn: 'mod_god_mode_into',
            params: { 
                n1: 17, 
                n2: 6, 
                n3: 4, 
                mode: 'tradition_mg', 
                has_volume: true 
            },
            inputs: ['high', 'low', 'close', 'volume'],
            tripleOutput: true  
        },
        
        batch: {
            fn: 'mod_god_mode_batch',
            config: {
                small: {
                    n1_range: [10, 20, 5],      
                    n2_range: [4, 8, 2],        
                    n3_range: [2, 6, 2],        
                    mode: 'tradition_mg',
                    use_volume: true
                },
                medium: {
                    n1_range: [10, 25, 3],      
                    n2_range: [3, 9, 2],        
                    n3_range: [2, 8, 2],        
                    mode: 'tradition_mg',
                    use_volume: true
                }
            }
        }
    },
    range_filter: {
        name: 'Range Filter',
        
        safe: {
            fn: 'range_filter_js',
            params: { 
                range_size: 2.618, 
                range_period: 14, 
                smooth: true, 
                filter_period: 27, 
                filter_type: 'close' 
            }
        },
        
        fast: {
            allocFn: 'range_filter_alloc',
            freeFn: 'range_filter_free',
            computeFn: 'range_filter_into',
            params: { 
                range_size: 2.618, 
                range_period: 14, 
                smooth: true, 
                filter_period: 27, 
                filter_type: 'close' 
            }
        },
        
        batch: {
            fn: 'range_filter_batch',
            config: {
                small: {
                    range_size_range: [2.0, 3.0, 0.5],      
                    range_period_range: [10, 20, 5],        
                    smooth: true,
                    filter_period: 27,
                    filter_type: 'close'                    
                },
                medium: {
                    range_size_range: [2.0, 3.5, 0.3],      
                    range_period_range: [10, 30, 5],        
                    smooth: true,
                    filter_period: 27,
                    filter_type: 'close'                    
                }
            }
        }
    },
    reflex: {
        name: 'Reflex',
        
        safe: {
            fn: 'reflex_js',
            params: { period: 20 }
        },
        
        fast: {
            allocFn: 'reflex_alloc',
            freeFn: 'reflex_free',
            computeFn: 'reflex_into',
            params: { period: 20 }
        },
        
        batch: {
            fn: 'reflex_batch_js',
            config: {
                small: {
                    period_range: [10, 30, 10]  
                },
                medium: {
                    period_range: [10, 50, 5]   
                }
            },
            
            metadataFn: 'reflex_batch_metadata_js',
            rowsColsFn: 'reflex_batch_rows_cols_js'
            
        }
    },
    rocr: {
        name: 'ROCR',
        
        safe: {
            fn: 'rocr_js',
            params: { period: 9 }
        },
        
        fast: {
            allocFn: 'rocr_alloc',
            freeFn: 'rocr_free',
            computeFn: 'rocr_into',
            params: { period: 9 }
        },
        
        batch: {
            fn: 'rocr_batch',
            config: {
                small: {
                    period_range: [5, 15, 5]     
                },
                medium: {
                    period_range: [5, 20, 3]     
                }
            },
            
            fastFn: 'rocr_batch_into'
        }
    },
    swma: {
        name: 'SWMA (Symmetric Weighted Moving Average)',
        
        safe: {
            fn: 'swma_js',
            params: { period: 5 }
        },
        
        fast: {
            allocFn: 'swma_alloc',
            freeFn: 'swma_free',
            computeFn: 'swma_into',
            params: { period: 5 }
        },
        
        batch: {
            fn: 'swma_batch',  
            config: {
                small: {
                    period_range: [3, 15, 3]       
                    
                },
                medium: {
                    period_range: [3, 30, 3]       
                    
                }
            },
            
            fastFn: 'swma_batch_into'
        }
    },
    cwma: {
        name: 'CWMA',
        
        safe: {
            fn: 'cwma_js',
            params: { period: 14 }
        },
        
        fast: {
            allocFn: 'cwma_alloc',
            freeFn: 'cwma_free',
            computeFn: 'cwma_into',
            params: { period: 14 }
        },
        
        batch: {
            fn: 'cwma_batch',
            config: {
                small: {
                    period_range: [10, 20, 5]      
                },
                medium: {
                    period_range: [10, 50, 5]      
                }
            },
            
            fastFn: 'cwma_batch_into'
        }
    },
    er: {
        name: 'ER (Kaufman Efficiency Ratio)',
        
        safe: {
            fn: 'er_js',
            params: { period: 5 }
        },
        
        fast: {
            allocFn: 'er_alloc',
            freeFn: 'er_free',
            computeFn: 'er_into',
            params: { period: 5 }
        },
        
        batch: {
            fn: 'er_batch',
            config: {
                small: {
                    period_range: [5, 15, 5]       
                },
                medium: {
                    period_range: [5, 30, 5]       
                }
            },
            
            fastFn: 'er_batch_into'
        }
    },
    decycler: {
        name: 'Decycler',
        
        safe: {
            fn: 'decycler_js',
            params: { hp_period: 125, k: 0.707 }
        },
        
        fast: {
            allocFn: 'decycler_alloc',
            freeFn: 'decycler_free',
            computeFn: 'decycler_into',
            params: { hp_period: 125, k: 0.707 }
        },
        
        batch: {
            fn: 'decycler_batch',
            config: {
                small: {
                    hp_period_range: [100, 150, 25],    
                    k_range: [0.5, 0.9, 0.2]            
                    
                },
                medium: {
                    hp_period_range: [100, 150, 10],    
                    k_range: [0.5, 0.9, 0.1]            
                    
                }
            },
            
            fastFn: 'decycler_batch_into'
        }
    },
    dema: {
        name: 'DEMA',
        
        safe: {
            fn: 'dema_js',
            params: { period: 30 }
        },
        
        fast: {
            allocFn: 'dema_alloc',
            freeFn: 'dema_free',
            computeFn: 'dema_into',
            params: { period: 30 }
        },
        
        batch: {
            fn: 'dema_batch',
            config: {
                small: {
                    period_range: [10, 30, 10]     
                },
                medium: {
                    period_range: [10, 50, 10]     
                }
            }
        }
    },
    epma: {
        name: 'EPMA',
        
        safe: {
            fn: 'epma_js',
            params: { period: 11, offset: 4 }
        },
        
        fast: {
            allocFn: 'epma_alloc',
            freeFn: 'epma_free',
            computeFn: 'epma_into',
            params: { period: 11, offset: 4 }
        },
        
        batch: {
            fn: 'epma_batch',
            config: {
                small: {
                    period_range: [5, 15, 5],      
                    offset_range: [2, 4, 1]        
                    
                },
                medium: {
                    period_range: [5, 25, 2],      
                    offset_range: [1, 4, 1]        
                    
                }
            },
            
            fastFn: 'epma_batch_into'
        }
    },
    jma: {
        name: 'JMA',
        
        safe: {
            fn: 'jma_js',
            params: { period: 7, phase: 50.0, power: 2 }
        },
        
        fast: {
            allocFn: 'jma_alloc',
            freeFn: 'jma_free',
            computeFn: 'jma_into',
            params: { period: 7, phase: 50.0, power: 2 }
        },
        
        batch: {
            fn: 'jma_batch',
            config: {
                small: {
                    period_range: [5, 15, 5],      
                    phase_range: [0.0, 100.0, 50.0], 
                    power_range: [1, 3, 1]         
                    
                },
                medium: {
                    period_range: [5, 25, 5],      
                    phase_range: [0.0, 100.0, 25.0], 
                    power_range: [1, 3, 1]         
                    
                }
            },
            
            fastFn: 'jma_batch_into'
        }
    },
    highpass_2_pole: {
        name: 'HighPass 2-Pole',
        
        safe: {
            fn: 'highpass_2_pole_js',
            params: { period: 48, k: 0.707 }
        },
        
        fast: {
            allocFn: 'highpass_2_pole_alloc',
            freeFn: 'highpass_2_pole_free',
            computeFn: 'highpass_2_pole_into',
            params: { period: 48, k: 0.707 }
        },
        
        batch: {
            fn: 'highpass_2_pole_batch',
            config: {
                small: {
                    period_range: [20, 60, 20],    
                    k_range: [0.5, 0.9, 0.2]       
                    
                },
                medium: {
                    period_range: [20, 80, 10],    
                    k_range: [0.3, 0.9, 0.1]       
                    
                }
            },
            
            fastFn: 'highpass_2_pole_batch_into'
        }
    },
    nama: {
        name: 'NAMA (Nonlinear Adaptive Moving Average)',
        
        safe: {
            fn: 'nama_js',
            params: { period: 30 }
        },
        
        fast: {
            allocFn: 'nama_alloc',
            freeFn: 'nama_free',
            computeFn: 'nama_into',
            params: { period: 30 }
        },
        
        batch: {
            fn: 'nama_batch',
            config: {
                small: {
                    period_range: [20, 40, 10]     
                },
                medium: {
                    period_range: [10, 50, 5]      
                }
            },
            
            fastFn: 'nama_batch_into'
        }
    },
    nma: {
        name: 'NMA',
        
        safe: {
            fn: 'nma_js',
            params: { period: 40 }
        },
        
        fast: {
            allocFn: 'nma_alloc',
            freeFn: 'nma_free',
            computeFn: 'nma_into',
            params: { period: 40 }
        },
        
        batch: {
            fn: 'nma_batch',
            config: {
                small: {
                    period_range: [20, 60, 20]     
                },
                medium: {
                    period_range: [10, 90, 10]     
                }
            },
            
            fastFn: 'nma_batch_into'
        }
    },
    sma: {
        name: 'SMA',
        
        safe: {
            fn: 'sma',
            params: { period: 14 }
        },
        
        fast: {
            allocFn: 'sma_alloc',
            freeFn: 'sma_free',
            computeFn: 'sma_into',
            params: { period: 14 }
        },
        
        batch: {
            fn: 'sma_batch',
            config: {
                small: {
                    period_range: [5, 15, 5]       
                },
                medium: {
                    period_range: [5, 25, 2]       
                }
            },
            
            fastFn: 'sma_batch_into'
        }
    },
    supersmoother_3_pole: {
        name: 'SuperSmoother 3-Pole',
        
        safe: {
            fn: 'supersmoother_3_pole_js',
            params: { period: 14 }
        },
        
        fast: {
            allocFn: 'supersmoother_3_pole_alloc',
            freeFn: 'supersmoother_3_pole_free',
            computeFn: 'supersmoother_3_pole_into',
            params: { period: 14 }
        },
        
        batch: {
            fn: 'supersmoother_3_pole_batch',
            config: {
                small: {
                    period_range: [10, 20, 5]      
                },
                medium: {
                    period_range: [10, 30, 2]      
                }
            },
            
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
                    period_range: [5, 20, 5]      
                },
                medium: {
                    period_range: [5, 50, 5]      
                }
            },
            fastFn: 'ema_batch_into'
        }
    },
    trima: {
        name: 'TRIMA',
        
        safe: {
            fn: 'trima_js',
            params: { period: 30 }
        },
        
        fast: {
            allocFn: 'trima_alloc',
            freeFn: 'trima_free',
            computeFn: 'trima_into',
            params: { period: 30 }
        },
        
        batch: {
            fn: 'trima_batch',
            config: {
                small: {
                    period_range: [10, 30, 10]      
                },
                medium: {
                    period_range: [10, 50, 5]       
                }
            },
            
            fastFn: 'trima_batch_into'
        }
    },
    tema: {
        name: 'TEMA',
        
        safe: {
            fn: 'tema_js',
            params: { period: 14 }
        },
        
        fast: {
            allocFn: 'tema_alloc',
            freeFn: 'tema_free',
            computeFn: 'tema_into',
            params: { period: 14 }
        },
        
        batch: {
            fn: 'tema_batch',
            config: {
                small: {
                    period_range: [5, 15, 5]       
                },
                medium: {
                    period_range: [5, 30, 5]       
                }
            },
            fastFn: 'tema_batch_into'
        }
    },
    wilders: {
        name: 'Wilders',
        
        safe: {
            fn: 'wilders_js',
            params: { period: 5 }
        },
        
        fast: {
            allocFn: 'wilders_alloc',
            freeFn: 'wilders_free',
            computeFn: 'wilders_into',
            params: { period: 5 }
        },
        
        batch: {
            fn: 'wilders_batch',
            config: {
                small: {
                    period_range: [5, 15, 5]       
                    
                },
                medium: {
                    period_range: [5, 25, 4]       
                    
                }
            }
        }
    },
    willr: {
        name: 'Williams %R',
        
        safe: {
            fn: 'willr_js',
            params: { period: 14 },
            needsMultipleInputs: true  
        },
        
        fast: {
            allocFn: 'willr_alloc',
            freeFn: 'willr_free',
            computeFn: 'willr_into',
            params: { period: 14 },
            needsMultipleInputs: true
        },
        
        batch: {
            fn: 'willr_batch',
            fastFn: 'willr_batch_into',
            config: {
                small: {
                    period_range: [10, 20, 5]       
                },
                medium: {
                    period_range: [10, 30, 5]       
                }
            },
            needsMultipleInputs: true
        }
    },
    wma: {
        name: 'WMA (Weighted Moving Average)',
        
        safe: {
            fn: 'wma_js',
            params: { period: 30 }
        },
        
        fast: {
            allocFn: 'wma_alloc',
            freeFn: 'wma_free',
            computeFn: 'wma_into',
            params: { period: 30 }
        },
        
        batch: {
            fn: 'wma_batch_js',
            config: {
                small: {
                    period_range: [10, 30, 10]     
                },
                medium: {
                    period_range: [10, 50, 5]      
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
                    period_range: [10, 20, 5],    
                    poles_range: [2, 4, 1]        
                },
                medium: {
                    period_range: [10, 50, 5],    
                    poles_range: [1, 4, 1]        
                }
            },
            fastFn: 'gaussian_batch_into'
        }
    },
    hwma: {
        name: 'HWMA',
        
        safe: {
            fn: 'hwma_js',
            params: { na: 0.2, nb: 0.1, nc: 0.1 }
        },
        
        fast: {
            allocFn: 'hwma_alloc',
            freeFn: 'hwma_free',
            computeFn: 'hwma_into',
            params: { na: 0.2, nb: 0.1, nc: 0.1 }
        },
        
        batch: {
            fn: 'hwma_batch',
            config: {
                small: {
                    na_range: [0.1, 0.3, 0.1],      
                    nb_range: [0.1, 0.2, 0.1],      
                    nc_range: [0.1, 0.2, 0.1]       
                    
                },
                medium: {
                    na_range: [0.1, 0.5, 0.1],      
                    nb_range: [0.1, 0.3, 0.1],      
                    nc_range: [0.1, 0.3, 0.1]       
                    
                }
            },
            
            fastFn: 'hwma_batch_into'
        }
    },
    mwdx: {
        name: 'MWDX',
        
        safe: {
            fn: 'mwdx_js',
            params: { factor: 0.2 }
        },
        
        fast: {
            allocFn: 'mwdx_alloc',
            freeFn: 'mwdx_free',
            computeFn: 'mwdx_into',
            params: { factor: 0.2 }
        },
        
        batch: {
            fn: 'mwdx_batch',  
            config: {
                small: {
                    factor_range: [0.1, 0.3, 0.1]      
                },
                medium: {
                    factor_range: [0.1, 0.5, 0.1]      
                }
            },
            
            fastFn: 'mwdx_batch_into'
        }
    },
    srwma: {
        name: 'SRWMA',
        
        safe: {
            fn: 'srwma_js',
            params: { period: 14 }
        },
        
        fast: {
            allocFn: 'srwma_alloc',
            freeFn: 'srwma_free',
            computeFn: 'srwma_into',
            params: { period: 14 }
        },
        
        batch: {
            fn: 'srwma_batch',
            config: {
                small: {
                    period_range: [10, 20, 5]      
                },
                medium: {
                    period_range: [10, 50, 5]      
                }
            },
            
            fastFn: 'srwma_batch_into'
        }
    },
    deviation: {
        name: 'Deviation',
        
        safe: {
            fn: 'deviation_js',
            params: { period: 20, devtype: 0 }
        },
        
        fast: {
            allocFn: 'deviation_alloc',
            freeFn: 'deviation_free',
            computeFn: 'deviation_into',
            params: { period: 20, devtype: 0 }
        },
        
        batch: {
            fn: 'deviation_batch',
            config: {
                small: {
                    period_range: [10, 30, 10],      
                    devtype_range: [0, 2, 1]         
                },
                medium: {
                    period_range: [10, 50, 5],       
                    devtype_range: [0, 2, 1]         
                }
            }
        }
    },
    linreg: {
        name: 'LinReg',
        
        safe: {
            fn: 'linreg_js',
            params: { period: 14 }
        },
        
        fast: {
            allocFn: 'linreg_alloc',
            freeFn: 'linreg_free',
            computeFn: 'linreg_into',
            params: { period: 14 }
        },
        
        batch: {
            fn: 'linreg_batch',
            config: {
                small: {
                    period_range: [10, 20, 5]      
                },
                medium: {
                    period_range: [10, 50, 5]      
                }
            },
            
            fastFn: 'linreg_batch_into'
        }
    },
    linearreg_intercept: {
        name: 'Linear Regression Intercept',
        
        safe: {
            fn: 'linearreg_intercept_js',
            params: { period: 12 }
        },
        
        fast: {
            allocFn: 'linearreg_intercept_alloc',
            freeFn: 'linearreg_intercept_free',
            computeFn: 'linearreg_intercept_into',
            params: { period: 12 }
        },
        
        batch: {
            fn: 'linearreg_intercept_batch',
            config: {
                small: {
                    period_range: [10, 20, 5]      
                },
                medium: {
                    period_range: [10, 50, 5]      
                }
            },
            
            fastFn: 'linearreg_intercept_batch_into'
        }
    },
    rsx: {
        name: 'RSX (Relative Strength Xtra)',
        
        safe: {
            fn: 'rsx_js',
            params: { period: 14 }
        },
        
        fast: {
            allocFn: 'rsx_alloc',
            freeFn: 'rsx_free',
            computeFn: 'rsx_into',
            params: { period: 14 }
        },
        
        batch: {
            fn: 'rsx_batch',
            config: {
                small: {
                    period_range: [10, 20, 5]      
                },
                medium: {
                    period_range: [10, 50, 5]      
                }
            },
            
            fastFn: 'rsx_batch_into'
        }
    },
    linearreg_angle: {
        name: 'Linear Regression Angle',
        
        safe: {
            fn: 'linearreg_angle_js',
            params: { period: 14 }
        },
        
        fast: {
            allocFn: 'linearreg_angle_alloc',
            freeFn: 'linearreg_angle_free',
            computeFn: 'linearreg_angle_into',
            params: { period: 14 }
        },
        
        batch: {
            fn: 'linearreg_angle_batch',
            config: {
                small: {
                    period_range: [10, 20, 5]      
                },
                medium: {
                    period_range: [10, 50, 5]      
                }
            }
        }
    },
    sinwma: {
        name: 'SINWMA',
        
        safe: {
            fn: 'sinwma_js',
            params: { period: 14 }
        },
        
        fast: {
            allocFn: 'sinwma_alloc',
            freeFn: 'sinwma_free',
            computeFn: 'sinwma_into',
            params: { period: 14 }
        },
        
        batch: {
            fn: 'sinwma_batch',
            config: {
                small: {
                    period_range: [10, 20, 5]      
                },
                medium: {
                    period_range: [10, 50, 5]      
                }
            }
            
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
                    period_range: [5, 15, 5],         
                    volume_factor_range: [0.0, 0.7, 0.35]  
                },
                medium: {
                    period_range: [5, 25, 5],         
                    volume_factor_range: [0.0, 0.8, 0.2]   
                }
            }
        }
    },
    trendflex: {
        name: 'TrendFlex',
        
        safe: {
            fn: 'trendflex_js',
            params: { period: 20 }
        },
        
        fast: {
            allocFn: 'trendflex_alloc',
            freeFn: 'trendflex_free',
            computeFn: 'trendflex_into',
            params: { period: 20 }
        },
        
        batch: {
            fn: 'trendflex_batch',
            config: {
                small: {
                    period_range: [10, 30, 10]      
                },
                medium: {
                    period_range: [10, 40, 2]       
                }
            },
            
            fastFn: 'trendflex_batch_into'
        }
    },
    trix: {
        name: 'TRIX',
        
        safe: {
            fn: 'trix_js',
            params: { period: 18 }
        },
        
        fast: {
            allocFn: 'trix_alloc',
            freeFn: 'trix_free',
            computeFn: 'trix_into',
            params: { period: 18 }
        },
        
        batch: {
            fn: 'trix_batch',
            fastFn: 'trix_batch_into',
            config: {
                small: {
                    period_range: [14, 22, 4]      
                },
                medium: {
                    period_range: [10, 30, 5]      
                }
            }
        }
    },
    ttm_trend: {
        name: 'TTM Trend',
        
        safe: {
            fn: 'ttm_trend_js',
            params: { period: 5 }
        },
        needsMultipleInputs: true,  
        outputIsU8: true,           
        
        fast: {
            allocFn: 'ttm_trend_alloc',
            allocU8Fn: 'ttm_trend_alloc_u8',  
            freeFn: 'ttm_trend_free',
            freeU8Fn: 'ttm_trend_free_u8',    
            computeFn: 'ttm_trend_into',
            params: { period: 5 },
            needsMultipleInputs: true,
            outputIsU8: true
        },
        
        batch: {
            fn: 'ttm_trend_batch',
            config: {
                small: {
                    period_range: [5, 15, 5]       
                },
                medium: {
                    period_range: [5, 20, 1]       
                }
            },
            outputIsU8: true  
        }
    },
    alligator: {
        name: 'Alligator',
        
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
            
            outputCount: 3
        },
        
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
                    
                },
                medium: {
                    jaw_period_range: [10, 20, 5],      
                    jaw_offset_range: [6, 10, 2],       
                    teeth_period_range: [6, 10, 2],     
                    teeth_offset_range: [4, 6, 1],      
                    lips_period_range: [4, 6, 1],       
                    lips_offset_range: [2, 4, 1]        
                    
                }
            }
        }
    },
    correlation_cycle: {
        name: 'Correlation Cycle',
        
        safe: {
            fn: 'correlation_cycle_js',
            params: { period: 20, threshold: 9.0 }
        },
        
        fast: {
            allocFn: 'correlation_cycle_alloc',
            freeFn: 'correlation_cycle_free',
            computeFn: 'correlation_cycle_into',
            params: { period: 20, threshold: 9.0 },
            outputCount: 4  
        },
        
        batch: {
            fn: 'correlation_cycle_batch_js',
            config: {
                small: {
                    period_range: [15, 25, 5],          
                    threshold_range: [8.0, 10.0, 1.0]   
                },
                medium: {
                    period_range: [10, 30, 5],          
                    threshold_range: [7.0, 11.0, 1.0]   
                }
            }
        }
    },
    volume_adjusted_ma: {
        name: 'Volume Adjusted MA',
        
        safe: {
            fn: 'volume_adjusted_ma_js',
            params: { length: 13, vi_factor: 0.67, strict: true, sample_period: 0 },
            inputs: ['prices', 'volumes']  
        },
        
        fast: {
            allocFn: 'volume_adjusted_ma_alloc',
            freeFn: 'volume_adjusted_ma_free',
            computeFn: 'volume_adjusted_ma_into',
            params: { length: 13, vi_factor: 0.67, strict: true, sample_period: 0 },
            inputs: ['prices', 'volumes']  
        },
        
        batch: {
            fn: 'volume_adjusted_ma_batch',
            fastFn: 'volume_adjusted_ma_batch_into',
            config: {
                small: {
                    length_range: [10, 20, 5],           
                    vi_factor_range: [0.5, 1.0, 0.25],   
                    sample_period_range: [0, 0, 0],      
                    strict: true                         
                    
                },
                medium: {
                    length_range: [5, 25, 5],            
                    vi_factor_range: [0.3, 1.0, 0.1],    
                    sample_period_range: [0, 10, 5],     
                    strict: true                         
                    
                }
            },
            inputs: ['prices', 'volumes']  
        }
    },
    vwma: {
        name: 'VWMA (Volume Weighted Moving Average)',
        
        safe: {
            fn: 'vwma_js',
            params: { period: 20 },
            inputs: ['prices', 'volumes']  
        },
        
        fast: {
            allocFn: 'vwma_alloc',
            freeFn: 'vwma_free',
            computeFn: 'vwma_into',
            params: { period: 20 },
            inputs: ['prices', 'volumes']  
        },
        
        batch: {
            fn: 'vwma_batch',
            config: {
                small: {
                    period_range: [10, 30, 5]      
                },
                medium: {
                    period_range: [5, 50, 5]       
                }
            },
            
            fastFn: 'vwma_batch_into',
            inputs: ['prices', 'volumes']  
        }
    },
    vwmacd: {
        name: 'VWMACD (Volume Weighted MACD)',
        
        safe: {
            fn: 'vwmacd_js',
            params: { 
                fast_period: 12, 
                slow_period: 26, 
                signal_period: 9,
                fast_ma_type: 'sma',
                slow_ma_type: 'sma',
                signal_ma_type: 'ema'
            },
            inputs: ['close', 'volume']  
        },
        
        fast: {
            allocFn: 'vwmacd_alloc',
            freeFn: 'vwmacd_free',
            computeFn: 'vwmacd_into',
            params: { 
                fast_period: 12, 
                slow_period: 26, 
                signal_period: 9,
                fast_ma_type: 'sma',
                slow_ma_type: 'sma',
                signal_ma_type: 'ema'
            },
            inputs: ['close', 'volume'],  
            outputs: ['macd', 'signal', 'hist']  
        },
        
        batch: {
            fn: 'vwmacd_batch',
            config: {
                small: {
                    fast_range: [10, 14, 2],      
                    slow_range: [20, 26, 3],      
                    signal_range: [5, 9, 2]       
                    
                },
                medium: {
                    fast_range: [8, 16, 2],       
                    slow_range: [20, 30, 2],      
                    signal_range: [5, 13, 2]      
                    
                }
            },
            inputs: ['close', 'volume']  
        }
    },
    ad: {
        name: 'AD (Accumulation/Distribution)',
        
        safe: {
            fn: 'ad_js',
            params: {},  
            inputs: ['high', 'low', 'close', 'volume']  
        },
        
        fast: {
            allocFn: 'ad_alloc',
            freeFn: 'ad_free',
            computeFn: 'ad_into',
            params: {},  
            inputs: ['high', 'low', 'close', 'volume']  
        },
        
        batch: {
            fn: 'ad_batch_js',
            config: {
                small: {
                    
                },
                medium: {
                    
                }
            },
            inputs: ['high', 'low', 'close', 'volume']  
        }
    },
    vwap: {
        name: 'VWAP',
        
        safe: {
            fn: 'vwap_js',
            params: { anchor: '1d', kernel: null },  
            needsVwapInputs: true  
        },
        
        fast: {
            allocFn: 'vwap_alloc',
            freeFn: 'vwap_free',
            computeFn: 'vwap_into',
            params: { anchor: '1d' },
            needsVwapInputs: true
        },
        
        batch: {
            fn: 'vwap_batch',
            config: {
                small: {
                    anchor_range: ['1m', '15m', 14]    
                },
                medium: {
                    anchor_range: ['1m', '1h', 59]     
                }
            },
            needsVwapInputs: true
        }
    },
    zlema: {
        name: 'ZLEMA',
        
        safe: {
            fn: 'zlema_js',
            params: { period: 14 }
        },
        
        fast: {
            allocFn: 'zlema_alloc',
            freeFn: 'zlema_free',
            computeFn: 'zlema_into',
            params: { period: 14 }
        },
        
        batch: {
            fn: 'zlema_batch',
            fastFn: 'zlema_batch_into',
            config: {
                small: {
                    period_range: [10, 20, 2]      
                },
                medium: {
                    period_range: [5, 25, 5]       
                }
            }
        }
    },
    adx: {
        name: 'ADX',
        
        safe: {
            fn: 'adx_js',
            params: { period: 14 }
        },
        needsMultipleInputs: true, 
        
        fast: {
            allocFn: 'adx_alloc',
            freeFn: 'adx_free',
            computeFn: 'adx_into',
            params: { period: 14 },
            needsMultipleInputs: true
        },
        
        batch: {
            fn: 'adx_batch',
            fastFn: 'adx_batch_into',
            config: {
                small: {
                    period_range: [10, 18, 4]      
                },
                medium: {
                    period_range: [10, 30, 5]      
                }
            }
        }
    },
    bandpass: {
        name: 'BandPass',
        
        safe: {
            fn: 'bandpass_js',
            params: { period: 20, bandwidth: 0.3 }
        },
        
        fast: {
            allocFn: 'bandpass_alloc',
            freeFn: 'bandpass_free',
            computeFn: 'bandpass_into',
            params: { period: 20, bandwidth: 0.3 },
            outputCount: 4  
        },
        
        batch: {
            fn: 'bandpass_batch',
            config: {
                small: {
                    period_range: [10, 30, 10],       
                    bandwidth_range: [0.2, 0.4, 0.1]  
                },
                medium: {
                    period_range: [10, 30, 5],        
                    bandwidth_range: [0.2, 0.4, 0.05] 
                }
            }
        }
    },
    correl_hl: {
        name: 'CORREL_HL',
        
        safe: {
            fn: 'correl_hl_js',
            params: { period: 9 },
            needsMultipleInputs: true  
        },
        
        fast: {
            allocFn: 'correl_hl_alloc',
            freeFn: 'correl_hl_free',
            computeFn: 'correl_hl_into',
            params: { period: 9 },
            needsMultipleInputs: true  
        },
        
        batch: {
            fn: 'correl_hl_batch',
            fastFn: 'correl_hl_batch_into',
            config: {
                small: {
                    period_range: [5, 20, 5]       
                },
                medium: {
                    period_range: [5, 50, 5]       
                }
            }
        }
    },
    dti: {
        name: 'DTI',
        
        safe: {
            fn: 'dti_js',
            params: { r: 14, s: 10, u: 5 }
        },
        needsMultipleInputs: true,  
        
        fast: {
            allocFn: 'dti_alloc',
            freeFn: 'dti_free',
            computeFn: 'dti_into',
            params: { r: 14, s: 10, u: 5 },
            needsMultipleInputs: true
        },
        
        batch: {
            fn: 'dti_batch',
            config: {
                small: {
                    r_range: [10, 20, 5],    
                    s_range: [8, 12, 2],     
                    u_range: [4, 6, 1]       
                },
                medium: {
                    r_range: [10, 30, 5],    
                    s_range: [5, 15, 2],     
                    u_range: [3, 7, 1]       
                }
            }
        }
    },
    stc: {
        name: 'STC',
        
        safe: {
            fn: 'stc_js',
            params: { fast_period: 23, slow_period: 50, k_period: 10, d_period: 3, fast_ma_type: "ema", slow_ma_type: "ema" }
        },
        
        fast: {
            allocFn: 'stc_alloc',
            freeFn: 'stc_free',
            computeFn: 'stc_into',
            params: { fast_period: 23, slow_period: 50, k_period: 10, d_period: 3, fast_ma_type: "ema", slow_ma_type: "ema" }
        },
        
        batch: {
            fn: 'stc_batch',
            config: {
                small: {
                    fast_period_range: [20, 30, 5],    
                    slow_period_range: [45, 55, 5],    
                    k_period_range: [10, 10, 1],       
                    d_period_range: [3, 3, 1]          
                },
                medium: {
                    fast_period_range: [20, 30, 5],    
                    slow_period_range: [45, 55, 5],    
                    k_period_range: [8, 12, 2],        
                    d_period_range: [3, 3, 1]          
                }
            }
        }
    },
    tsi: {
        name: 'TSI',
        
        safe: {
            fn: 'tsi_js',
            params: { long_period: 25, short_period: 13 }
        },
        
        fast: {
            allocFn: 'tsi_alloc',
            freeFn: 'tsi_free',
            computeFn: 'tsi_into',
            params: { long_period: 25, short_period: 13 }
        },
        
        batch: {
            fn: 'tsi_batch',
            config: {
                small: {
                    long_period_range: [20, 30, 5],    
                    short_period_range: [10, 15, 5]    
                },
                medium: {
                    long_period_range: [20, 35, 5],    
                    short_period_range: [10, 20, 5]    
                }
            }
        }
    },
    aso: {
        name: 'ASO',
        needsMultipleInputs: true,  
        
        safe: {
            fn: 'aso_js',
            params: { period: 10, mode: 0 },
            needsMultipleInputs: true,
            multiOutput: 2  
        },
        
        fast: {
            allocFn: 'aso_alloc',
            freeFn: 'aso_free',
            computeFn: 'aso_into',
            params: { period: 10, mode: 0 },
            needsMultipleInputs: true,
            multiOutput: 2  
        },
        
        batch: {
            fn: 'aso_batch_unified_js',
            fastFn: 'aso_batch_into',
            config: {
                small: {
                    period_range: [8, 12, 2],    
                    mode_range: [0, 2, 1]        
                },
                medium: {
                    period_range: [6, 14, 2],    
                    mode_range: [0, 2, 1]        
                }
            },
            needsMultipleInputs: true,
            multiOutput: 2
        }
    },
    atr: {
        name: 'ATR',
        
        safe: {
            fn: 'atr',
            params: { length: 14 }
        },
        needsMultipleInputs: true, 
        
        fast: {
            allocFn: 'atr_alloc',
            freeFn: 'atr_free',
            computeFn: 'atr_into',
            params: { length: 14 },
            needsMultipleInputs: true
        },
        
        batch: {
            fn: 'atr_batch',
            fastFn: 'atr_batch_into',
            config: {
                small: {
                    length_range: [10, 18, 4]      
                },
                medium: {
                    length_range: [10, 30, 5]      
                }
            }
        }
    },
    cfo: {
        name: 'CFO',
        
        safe: {
            fn: 'cfo_js',
            params: { period: 14, scalar: 100.0 }
        },
        
        fast: {
            allocFn: 'cfo_alloc',
            freeFn: 'cfo_free',
            computeFn: 'cfo_into',
            params: { period: 14, scalar: 100.0 }
        },
        
        batch: {
            fn: 'cfo_batch',
            fastFn: 'cfo_batch_into',
            config: {
                small: {
                    period_range: [10, 20, 5],      
                    scalar_range: [50.0, 150.0, 50.0]  
                    
                },
                medium: {
                    period_range: [5, 25, 5],       
                    scalar_range: [25.0, 175.0, 25.0]  
                    
                }
            }
        }
    },
    coppock: {
        name: 'Coppock',
        
        safe: {
            fn: 'coppock_js',
            params: { short_period: 11, long_period: 14, ma_period: 10, ma_type: 'wma' }
        },
        
        fast: {
            allocFn: 'coppock_alloc',
            freeFn: 'coppock_free',
            computeFn: 'coppock_into',
            params: { short_period: 11, long_period: 14, ma_period: 10, ma_type: 'wma' }
        },
        
        batch: {
            fn: 'coppock_batch',
            config: {
                small: {
                    short_range: [10, 12, 2],      
                    long_range: [14, 16, 2],       
                    ma_range: [8, 10, 2],          
                    ma_type: 'wma'                 
                    
                },
                medium: {
                    short_range: [8, 14, 2],       
                    long_range: [12, 18, 2],       
                    ma_range: [6, 12, 2],          
                    ma_type: 'wma'                 
                    
                }
            }
        }
    },
    cora_wave: {
        name: 'CoRa Wave',
        
        safe: {
            fn: 'cora_wave_js',
            params: { 
                period: 48,
                r_multi: 4,
                v_coef: 0.75,
                v_exp: 0.991,
                v_min: 3.996,
                lma_period: 10,
                std_period: 48,
                std_multi: 0.1,
                max: 4.0
            }
        },
        
        fast: {
            allocFn: 'cora_wave_alloc',
            freeFn: 'cora_wave_free',
            computeFn: 'cora_wave_into',
            params: { 
                period: 48,
                r_multi: 4,
                v_coef: 0.75,
                v_exp: 0.991,
                v_min: 3.996,
                lma_period: 10,
                std_period: 48,
                std_multi: 0.1,
                max: 4.0
            }
        },
        
        batch: {
            fn: 'cora_wave_batch',
            config: {
                small: {
                    period_range: [20, 30, 10],        
                    r_multi_range: [2, 4, 2],          
                    v_coef_range: [0.75, 0.75, 0],     
                    v_exp_range: [0.991, 0.991, 0],    
                    v_min_range: [3.996, 3.996, 0],    
                    lma_period_range: [10, 10, 0],     
                    std_period_range: [48, 48, 0],     
                    std_multi_range: [0.1, 0.1, 0],    
                    max_range: [4.0, 4.0, 0]           
                    
                },
                medium: {
                    period_range: [20, 60, 10],        
                    r_multi_range: [2, 6, 1],          
                    v_coef_range: [0.75, 0.75, 0],     
                    v_exp_range: [0.991, 0.991, 0],    
                    v_min_range: [3.996, 3.996, 0],    
                    lma_period_range: [10, 10, 0],     
                    std_period_range: [48, 48, 0],     
                    std_multi_range: [0.1, 0.1, 0],    
                    max_range: [4.0, 4.0, 0]           
                    
                }
            },
            fastFn: 'cora_wave_batch_into'
        }
    },
    dpo: {
        name: 'DPO',
        
        safe: {
            fn: 'dpo_js',
            params: { period: 5 }
        },
        
        fast: {
            allocFn: 'dpo_alloc',
            freeFn: 'dpo_free',
            computeFn: 'dpo_into',
            params: { period: 5 }
        },
        
        batch: {
            fn: 'dpo_batch',
            fastFn: 'dpo_batch_into',
            config: {
                small: {
                    period_range: [5, 15, 5]       
                },
                medium: {
                    period_range: [5, 25, 5]       
                },
                large: {
                    period_range: [5, 60, 5]       
                }
            }
        }
    },
    kaufmanstop: {
        name: 'Kaufman Stop',
        
        safe: {
            fn: 'kaufmanstop_js',
            params: { period: 22, mult: 2.0, direction: 'long', ma_type: 'sma' }
        },
        needsMultipleInputs: true,  
        
        fast: {
            allocFn: 'kaufmanstop_alloc',
            freeFn: 'kaufmanstop_free',
            computeFn: 'kaufmanstop_into',
            params: { period: 22, mult: 2.0, direction: 'long', ma_type: 'sma' },
            needsMultipleInputs: true
        },
        
        batch: {
            fn: 'kaufmanstop_batch_js',
            fastFn: 'kaufmanstop_batch_into',
            config: {
                small: {
                    period_range: [20, 24, 2],      
                    mult_range: [1.5, 2.5, 0.5],    
                    direction: 'long',               
                    ma_type: 'sma'                  
                    
                },
                medium: {
                    period_range: [18, 26, 2],      
                    mult_range: [1.0, 3.0, 0.5],    
                    direction: 'long',               
                    ma_type: 'sma'                  
                    
                }
            }
        }
    },
    midpoint: {
        name: 'Midpoint',
        
        safe: {
            fn: 'midpoint_js',
            params: { period: 14 }
        },
        
        fast: {
            allocFn: 'midpoint_alloc',
            freeFn: 'midpoint_free',
            computeFn: 'midpoint_into',
            params: { period: 14 }
        },
        
        batch: {
            fn: 'midpoint_batch',
            config: {
                small: {
                    period_range: [10, 20, 2]       
                },
                medium: {
                    period_range: [5, 25, 2]        
                },
                large: {
                    period_range: [5, 50, 5]        
                }
            }
        }
    },
    vi: {
        name: 'VI (Vortex Indicator)',
        needsMultipleInputs: true,  
        
        safe: {
            fn: 'vi_js',
            params: { period: 14 }
        },
        
        fast: {
            allocFn: 'vi_alloc',
            freeFn: 'vi_free',
            computeFn: 'vi_into',
            params: { period: 14 },
            needsMultipleInputs: true,
            dualOutput: true  
        },
        
        batch: {
            fn: 'vi_batch',
            config: {
                small: {
                    period_range: [10, 20, 2]  
                },
                medium: {
                    period_range: [10, 30, 2]  
                }
            },
            fastFn: 'vi_batch_into',
            dualOutput: true
        }
    },
    vpt: {
        name: 'VPT (Volume Price Trend)',
        
        safe: {
            fn: 'vpt_js',
            params: {},  
            inputs: ['close', 'volume']  
        },
        
        fast: {
            allocFn: 'vpt_alloc',
            freeFn: 'vpt_free',
            computeFn: 'vpt_into',
            params: {},  
            inputs: ['close', 'volume']  
        },
        
        batch: {
            fn: 'vpt_batch',
            config: {},  
            fastFn: 'vpt_batch_into'
        }
    },
    nvi: {
        name: 'NVI (Negative Volume Index)',
        
        safe: {
            fn: 'nvi_js',
            params: {},  
            inputs: ['close', 'volume']  
        },
        
        fast: {
            allocFn: 'nvi_alloc',
            freeFn: 'nvi_free',
            computeFn: 'nvi_into',
            params: {},  
            inputs: ['close', 'volume']  
        }
        
    },
    nadaraya_watson_envelope: {
        name: 'Nadaraya-Watson Envelope',
        
        safe: {
            fn: 'nadaraya_watson_envelope_js',
            params: { bandwidth: 8.0, multiplier: 50.0, lookback: 500 }
        },
        
        fast: {
            allocFn: 'nadaraya_watson_envelope_alloc',
            freeFn: 'nadaraya_watson_envelope_free',
            computeFn: 'nadaraya_watson_envelope_into',
            params: { bandwidth: 8.0, multiplier: 50.0, lookback: 500 },
            dualOutput: true  
        },
        
        batch: {
            fn: 'nadaraya_watson_envelope_batch',
            config: {
                small: {
                    bandwidth_range: [5.0, 10.0, 2.5],       
                    multiplier_range: [30.0, 70.0, 20.0],    
                    lookback_range: [300, 700, 200]          
                },
                medium: {
                    bandwidth_range: [4.0, 12.0, 2.0],       
                    multiplier_range: [20.0, 80.0, 15.0],    
                    lookback_range: [200, 800, 150]          
                }
            }
        }
    },
    pvi: {
        name: 'PVI (Positive Volume Index)',
        
        safe: {
            fn: 'pvi_js',
            params: { initial_value: 1000.0 },
            needsMultipleInputs: true  
        },
        
        fast: {
            allocFn: 'pvi_alloc',
            freeFn: 'pvi_free',
            computeFn: 'pvi_into',
            params: { initial_value: 1000.0 },
            needsMultipleInputs: true  
        },
        
        batch: {
            fn: 'pvi_batch',
            fastFn: 'pvi_batch_into',
            config: {
                small: {
                    initial_value_range: [900.0, 1100.0, 100.0]  
                },
                medium: {
                    initial_value_range: [800.0, 1200.0, 50.0]   
                }
            }
        }
    },
    
    rsmk: {
        name: 'RSMK',
        
        safe: {
            fn: 'rsmk_js',
            params: { lookback: 90, period: 3, signal_period: 20 },
            prepareData: (data) => ({
                main: data.close,
                compare: data.close,
                lookback: 90,
                period: 3,
                signal_period: 20,
                matype: null,
                signal_matype: null
            })
        },
        needsMultipleInputs: true,
        
        fast: {
            allocFn: 'rsmk_alloc',
            freeFn: 'rsmk_free',
            computeFn: 'rsmk_into',
            params: { lookback: 90, period: 3, signal_period: 20 },
            needsMultipleInputs: true,
            outputCount: 2  
        },
        
        batch: {
            fn: 'rsmk_batch',
            config: {
                small: {
                    lookback_range: [85, 95, 5],
                    period_range: [2, 4, 1],
                    signal_period_range: [18, 22, 2],
                    matype: "ema",
                    signal_matype: "ema"
                },
                medium: {
                    lookback_range: [80, 100, 5],
                    period_range: [2, 5, 1],
                    signal_period_range: [15, 25, 2],
                    matype: "ema",
                    signal_matype: "ema"
                }
            }
        }
    },
    
    srsi: {
        name: 'SRSI',
        
        safe: {
            fn: 'srsi_js',
            params: { rsi_period: 14, stoch_period: 14, k: 3, d: 3 }
        },
        
        fast: {
            allocFn: 'srsi_alloc',
            freeFn: 'srsi_free',
            computeFn: 'srsi_into',
            params: { rsi_period: 14, stoch_period: 14, k: 3, d: 3 },
            outputCount: 2  
        },
        
        batch: {
            fn: 'srsi_batch',
            config: {
                small: {
                    rsi_period_range: [10, 14, 2],       
                    stoch_period_range: [10, 14, 2],     
                    k_range: [2, 4, 1],                  
                    d_range: [2, 3, 1]                   
                    
                },
                medium: {
                    rsi_period_range: [10, 20, 2],       
                    stoch_period_range: [10, 20, 2],     
                    k_range: [2, 5, 1],                  
                    d_range: [2, 4, 1]                   
                    
                }
            },
            
            fastFn: 'srsi_batch_into'
        }
    },
    tsf: {
        name: 'TSF',
        
        safe: {
            fn: 'tsf_js',
            params: { period: 14 }
        },
        
        fast: {
            allocFn: 'tsf_alloc',
            freeFn: 'tsf_free',
            computeFn: 'tsf_into',
            params: { period: 14 }
        },
        
        batch: {
            fn: 'tsf_batch',
            config: {
                small: {
                    period_range: [10, 20, 5]      
                },
                medium: {
                    period_range: [10, 30, 2]      
                }
            },
            
            fastFn: 'tsf_batch_into'
        }
    },
    zscore: {
        name: 'ZSCORE',
        
        safe: {
            fn: 'zscore_js',
            params: { period: 14, ma_type: "sma", nbdev: 1.0, devtype: 0 }
        },
        
        fast: {
            allocFn: 'zscore_alloc',
            freeFn: 'zscore_free',
            computeFn: 'zscore_into',
            params: { period: 14, ma_type: "sma", nbdev: 1.0, devtype: 0 }
        },
        
        batch: {
            fn: 'zscore_batch',
            fastFn: 'zscore_batch_into',
            config: {
                small: {
                    period_range: [10, 20, 5],     
                    ma_type: "sma",
                    nbdev_range: [1.0, 2.0, 0.5],  
                    devtype_range: [0, 0, 0]       
                },
                medium: {
                    period_range: [10, 30, 5],     
                    ma_type: "sma",
                    nbdev_range: [0.5, 2.5, 0.5],  
                    devtype_range: [0, 1, 1]       
                }
            }
        }
    },
    aroonosc: {
        name: 'AroonOsc',
        
        safe: {
            fn: 'aroonosc_js',
            params: { length: 14 }
        },
        needsMultipleInputs: true, 
        
        fast: {
            allocFn: 'aroonosc_alloc',
            freeFn: 'aroonosc_free',
            computeFn: 'aroonosc_into',
            params: { length: 14 },
            needsMultipleInputs: true
        },
        
        batch: {
            fn: 'aroonosc_batch',
            fastFn: 'aroonosc_batch_into',
            config: {
                small: {
                    length_range: [10, 20, 5]      
                },
                medium: {
                    length_range: [10, 30, 5]      
                }
            }
        }
    },
    avsl: {
        name: 'AVSL',
        
        safe: {
            fn: 'avsl_js',
            params: { fast_period: 12, slow_period: 26, multiplier: 2.0 }
        },
        needsMultipleInputs: true, 
        
        fast: {
            allocFn: 'avsl_alloc',
            freeFn: 'avsl_free',
            computeFn: 'avsl_into',
            params: { fast_period: 12, slow_period: 26, multiplier: 2.0 },
            needsMultipleInputs: true
        },
        
        batch: {
            fn: 'avsl_batch_js',
            fastFn: 'avsl_batch_into',
            config: {
                small: {
                    fast_period_range: [10, 20, 5],     
                    slow_period_range: [20, 30, 5],     
                    multiplier_range: [1.5, 2.5, 0.5]   
                    
                },
                medium: {
                    fast_period_range: [8, 16, 2],      
                    slow_period_range: [20, 40, 5],     
                    multiplier_range: [1.0, 3.0, 0.5]   
                    
                }
            }
        }
    },
    cmo: {
        name: 'CMO',
        
        safe: {
            fn: 'cmo_js',
            params: { period: 14 }
        },
        
        fast: {
            allocFn: 'cmo_alloc',
            freeFn: 'cmo_free',
            computeFn: 'cmo_into',
            params: { period: 14 }
        },
        
        batch: {
            fn: 'cmo_batch',
            config: {
                small: {
                    period_range: [10, 20, 5]      
                },
                medium: {
                    period_range: [5, 30, 5]       
                }
            },
            
            fastFn: 'cmo_batch_into'
        }
    },
    dec_osc: {
        name: 'DEC_OSC',
        
        safe: {
            fn: 'dec_osc_js',
            params: { hp_period: 125, k: 1.0 }
        },
        
        fast: {
            allocFn: 'dec_osc_alloc',
            freeFn: 'dec_osc_free',
            computeFn: 'dec_osc_into',
            params: { hp_period: 125, k: 1.0 }
        },
        
        batch: {
            fn: 'dec_osc_batch',
            config: {
                small: {
                    hp_period_range: [100, 150, 25],    
                    k_range: [0.5, 1.5, 0.5]            
                },
                medium: {
                    hp_period_range: [50, 200, 25],     
                    k_range: [0.5, 2.0, 0.3]            
                }
            }
        }
    },
    donchian: {
        name: 'Donchian',
        needsMultipleInputs: true,  
        outputCount: 3,  
        
        safe: {
            fn: 'donchian_js',
            params: { period: 20 }
        },
        
        fast: {
            allocFn: 'donchian_alloc',
            freeFn: 'donchian_free',
            computeFn: 'donchian_into',
            params: { period: 20 },
            needsMultipleInputs: true,
            tripleOutput: true  
        },
        
        batch: {
            fn: 'donchian_batch',
            fastFn: 'donchian_batch_into',
            config: {
                small: {
                    period_range: [10, 30, 10]  
                },
                medium: {
                    period_range: [10, 50, 10]  
                }
            },
            needsMultipleInputs: true
        }
    },
    emv: {
        name: 'EMV',
        needsMultipleInputs: true,  
        needsVolume: true,  
        
        safe: {
            fn: 'emv_js',
            params: {}  
        },
        
        fast: {
            allocFn: 'emv_alloc',
            freeFn: 'emv_free',
            computeFn: 'emv_into',
            params: {},
            needsMultipleInputs: true,
            needsVolume: true
        },
        
        batch: {
            fn: 'emv_batch',
            fastFn: 'emv_batch_into',
            config: {
                
                small: {},
                medium: {}
            },
            needsMultipleInputs: true,
            needsVolume: true
        }
    },
    ift_rsi: {
        name: 'IFT RSI',
        
        safe: {
            fn: 'ift_rsi_js',
            params: { rsi_period: 5, wma_period: 9 }
        },
        
        fast: {
            allocFn: 'ift_rsi_alloc',
            freeFn: 'ift_rsi_free',
            computeFn: 'ift_rsi_into',
            params: { rsi_period: 5, wma_period: 9 }
        },
        
        batch: {
            fn: 'ift_rsi_batch',
            config: {
                small: {
                    rsi_period_range: [5, 7, 1],     
                    wma_period_range: [9, 11, 1]     
                },
                medium: {
                    rsi_period_range: [5, 14, 3],    
                    wma_period_range: [8, 14, 2]     
                }
            }
        }
    },
    macd: {
        name: 'MACD',
        
        safe: {
            fn: 'macd_js',
            params: { fast_period: 12, slow_period: 26, signal_period: 9, ma_type: 'ema' }
        },
        
        fast: {
            allocFn: 'macd_alloc',
            freeFn: 'macd_free',
            computeFn: 'macd_into',
            params: { fast_period: 12, slow_period: 26, signal_period: 9, ma_type: 'ema' },
            tripleOutput: true 
        },
        
        batch: {
            fn: 'macd_batch',
            config: {
                small: {
                    fast_period_range: [10, 14, 2],   
                    slow_period_range: [24, 28, 2],   
                    signal_period_range: [8, 10, 1],  
                    ma_type: 'ema'                    
                },
                medium: {
                    fast_period_range: [8, 16, 2],    
                    slow_period_range: [20, 32, 3],   
                    signal_period_range: [7, 11, 1],  
                    ma_type: 'ema'                    
                }
            }
        }
    },
    mfi: {
        name: 'MFI',
        
        safe: {
            fn: 'mfi_js',
            params: { period: 14 },
            dataFn: (data) => ({
                typical_price: data.typical_price,
                volume: data.volume
            }),
            prepareData: (candles) => ({
                typical_price: Array.from({ length: candles.close.length }, (_, i) => 
                    (candles.high[i] + candles.low[i] + candles.close[i]) / 3.0
                ),
                volume: candles.volume
            })
        },
        
        fast: {
            allocFn: 'mfi_alloc',
            freeFn: 'mfi_free',
            computeFn: 'mfi_into',
            params: { period: 14 },
            needsMultipleInputs: true,
            dataFn: (data) => ({
                typical_price: data.typical_price,
                volume: data.volume
            }),
            prepareData: (candles) => ({
                typical_price: Array.from({ length: candles.close.length }, (_, i) => 
                    (candles.high[i] + candles.low[i] + candles.close[i]) / 3.0
                ),
                volume: candles.volume
            })
        },
        
        batch: {
            fn: 'mfi_batch',
            needsMultipleInputs: true,
            dataFn: (data) => ({
                typical_price: data.typical_price,
                volume: data.volume
            }),
            prepareData: (candles) => ({
                typical_price: Array.from({ length: candles.close.length }, (_, i) => 
                    (candles.high[i] + candles.low[i] + candles.close[i]) / 3.0
                ),
                volume: candles.volume
            }),
            config: {
                small: {
                    period_range: [10, 20, 5]  
                },
                medium: {
                    period_range: [7, 21, 2]   
                }
            }
        }
    },
    natr: {
        name: 'NATR',
        needsMultipleInputs: true,  
        
        safe: {
            fn: 'natr_js',
            params: { period: 14 }
        },
        
        fast: {
            allocFn: 'natr_alloc',
            freeFn: 'natr_free',
            computeFn: 'natr_into',
            params: { period: 14 },
            needsMultipleInputs: true
        },
        
        batch: {
            fn: 'natr_batch',
            fastFn: 'natr_batch_into',
            config: {
                small: {
                    period_range: [10, 20, 5]      
                },
                medium: {
                    period_range: [10, 30, 5]      
                }
            },
            needsMultipleInputs: true
        }
    },
    net_myrsi: {
        name: 'NET_MYRSI',
        
        safe: {
            fn: 'net_myrsi_js',
            params: { period: 14 }
        },
        
        fast: {
            allocFn: 'net_myrsi_alloc',
            freeFn: 'net_myrsi_free',
            computeFn: 'net_myrsi_into',
            params: { period: 14 }
        },
        
        batch: {
            fn: 'net_myrsi_batch',
            fastFn: 'net_myrsi_batch_into',
            config: {
                small: {
                    period_range: [10, 20, 2]       
                },
                medium: {
                    period_range: [5, 30, 5]        
                }
            }
        }
    },
    ppo: {
        name: 'PPO (Percentage Price Oscillator)',
        
        safe: {
            fn: 'ppo_js',
            params: { fast_period: 12, slow_period: 26, ma_type: 'sma' }
        },
        
        fast: {
            allocFn: 'ppo_alloc',
            freeFn: 'ppo_free',
            computeFn: 'ppo_into',
            params: { fast_period: 12, slow_period: 26, ma_type: 'sma' }
        },
        
        batch: {
            fn: 'ppo_batch',
            fastFn: 'ppo_batch_into',
            config: {
                small: {
                    fast_period_range: [10, 14, 2],  
                    slow_period_range: [24, 28, 2],  
                    ma_type: 'sma'                   
                },
                medium: {
                    fast_period_range: [10, 20, 2],  
                    slow_period_range: [22, 32, 2],  
                    ma_type: 'ema'                   
                }
            }
        }
    },
    pfe: {
        name: 'PFE',
        
        safe: {
            fn: 'pfe_js',
            params: { period: 10, smoothing: 5 }
        },
        
        fast: {
            allocFn: 'pfe_alloc',
            freeFn: 'pfe_free',
            computeFn: 'pfe_into',
            params: { period: 10, smoothing: 5 }
        },
        
        batch: {
            fn: 'pfe_batch',
            fastFn: 'pfe_batch_into',
            config: {
                small: {
                    period_range: [8, 12, 2],       
                    smoothing_range: [3, 7, 2]      
                },
                medium: {
                    period_range: [5, 20, 5],       
                    smoothing_range: [2, 10, 2]     
                }
            }
        }
    },
    prb: {
        name: 'PRB',
        
        safe: {
            fn: 'prb_js',
            params: { 
                use_trend: true,
                poly_count: 10,
                poly_window: 100,
                stdev_count: 2,
                stdev_offset: 0
            }
        },
        
        fast: {
            allocFn: 'prb_alloc',
            freeFn: 'prb_free',
            computeFn: 'prb_into',
            params: { 
                use_trend: true,
                poly_count: 10,
                poly_window: 100,
                stdev_count: 2,
                stdev_offset: 0
            }
        },
        
        batch: {
            fn: 'prb_batch',
            config: {
                small: {
                    use_trend: true,
                    poly_count_range: [5, 15, 5],        
                    poly_window_range: [50, 150, 50],    
                    stdev_count_range: [1, 3, 1],        
                    stdev_offset_range: [0, 2, 1]        
                },
                medium: {
                    use_trend: true,
                    poly_count_range: [5, 20, 3],        
                    poly_window_range: [50, 200, 30],    
                    stdev_count_range: [1, 3, 1],        
                    stdev_offset_range: [0, 2, 1]        
                }
            }
        }
    },
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
                    period_range: [10, 20, 2]  
                },
                medium: {
                    period_range: [5, 30, 5]   
                }
            }
        }
    },
    squeeze_momentum: {
        name: 'Squeeze Momentum',
        needsMultipleInputs: true,  
        multipleOutputs: 3,         
        
        safe: {
            fn: 'squeeze_momentum_js',
            params: { length_bb: 20, mult_bb: 2.0, length_kc: 20, mult_kc: 1.5 }
        },
        
        fast: {
            allocFn: 'squeeze_momentum_alloc',
            freeFn: 'squeeze_momentum_free',
            computeFn: 'squeeze_momentum_into',
            params: { length_bb: 20, mult_bb: 2.0, length_kc: 20, mult_kc: 1.5 },
            needsMultipleInputs: true,
            multipleOutputs: 3
        },
        
        batch: {
            fn: 'squeeze_momentum_batch',
            config: {
                small: {
                    length_bb_range: [15, 25, 5],     
                    mult_bb_range: [2.0, 2.0, 0.0],   
                    length_kc_range: [20, 20, 0],     
                    mult_kc_range: [1.0, 2.0, 0.5]    
                },
                medium: {
                    length_bb_range: [10, 30, 5],     
                    mult_bb_range: [1.5, 2.5, 0.5],   
                    length_kc_range: [15, 25, 5],     
                    mult_kc_range: [1.0, 2.0, 0.5]    
                }
            }
        }
    },
    var: {
        name: 'VAR',
        safe: {
            fn: 'var_js',
            params: { period: 14, nbdev: 1.0 }
        },
        fast: {
            allocFn: 'var_alloc',
            freeFn: 'var_free',
            computeFn: 'var_into',
            params: { period: 14, nbdev: 1.0 }
        },
        batch: {
            fn: 'var_batch',
            fastFn: 'var_batch_into',
            config: {
                small: {
                    period_range: [10, 20, 5],      
                    nbdev_range: [1.0, 1.0, 0.0]    
                },
                medium: {
                    period_range: [10, 30, 5],      
                    nbdev_range: [0.5, 2.0, 0.5]    
                }
            }
        }
    },
    vpci: {
        name: 'VPCI',
        needsMultipleInputs: true,  
        needsVolume: true,
        dualOutput: true,           
        
        safe: {
            fn: 'vpci_js',
            params: { short_range: 5, long_range: 25 }
        },
        
        fast: {
            allocFn: 'vpci_alloc',
            freeFn: 'vpci_free',
            computeFn: 'vpci_into',
            params: { short_range: 5, long_range: 25 },
            needsMultipleInputs: true,
            needsVolume: true,
            dualOutput: true
        },
        
        batch: {
            fn: 'vpci_batch',
            config: {
                small: {
                    short_range: [5, 10, 5],       
                    long_range: [20, 30, 10]       
                },
                medium: {
                    short_range: [5, 15, 5],       
                    long_range: [20, 40, 5]        
                }
            },
            
            fastFn: 'vpci_batch_into',
            needsMultipleInputs: true,
            needsVolume: true
        }
    },
    wclprice: {
        name: 'WCLPRICE',
        
        safe: {
            fn: 'wclprice_js',
            params: {}  
        },
        needsMultipleInputs: true,  
        
        fast: {
            allocFn: 'wclprice_alloc',
            freeFn: 'wclprice_free',
            computeFn: 'wclprice_into',
            params: {},  
            needsMultipleInputs: true
        },
        
        batch: {
            fn: 'wclprice_batch',
            fastFn: 'wclprice_batch_into',
            config: {
                
                small: {},
                medium: {}
            },
            needsMultipleInputs: true
        }
    },
    wto: {
        name: 'WTO',
        
        safe: {
            fn: 'wto_js',
            params: { channel_length: 10, average_length: 21 }
        },
        
        fast: {
            allocFn: 'wto_alloc',
            freeFn: 'wto_free',
            computeFn: 'wto_into',
            params: { channel_length: 10, average_length: 21 },
            outputCount: 3  
        },
        
        batch: {
            fn: 'wto_batch',
            fastFn: 'wto_batch_into',
            config: {
                small: {
                    channel_range: [8, 12, 2],       
                    average_range: [15, 25, 5]       
                },
                medium: {
                    channel_range: [5, 20, 3],       
                    average_range: [10, 30, 5]       
                }
            }
        }
    },
    cksp: {
        name: 'CKSP',
        
        safe: {
            fn: 'cksp_js',
            params: { p: 10, x: 1.0, q: 9 },
            needsMultipleInputs: true,
            outputSize: 2  
        },
        
        fast: {
            allocFn: 'cksp_alloc',
            freeFn: 'cksp_free',
            computeFn: 'cksp_into',
            params: { p: 10, x: 1.0, q: 9 },
            needsMultipleInputs: true,
            outputCount: 2  
        },
        
        batch: {
            fn: 'cksp_batch',
            config: {
                small: {
                    p_range: [5, 15, 5],        
                    x_range: [0.5, 1.5, 0.5],   
                    q_range: [5, 10, 5]         
                },
                medium: {
                    p_range: [5, 25, 5],        
                    x_range: [0.5, 2.0, 0.5],   
                    q_range: [5, 15, 5]         
                }
            }
        }
    },
    emd: {
        name: 'EMD',
        needsMultipleInputs: true,  
        
        safe: {
            fn: 'emd_js',
            params: { period: 20, delta: 0.5, fraction: 0.1 },
            needsMultipleInputs: true,
            resultType: 'EmdResult'  
        },
        
        fast: {
            allocFn: 'emd_alloc',
            freeFn: 'emd_free',
            computeFn: 'emd_into',
            params: { period: 20, delta: 0.5, fraction: 0.1 },
            needsMultipleInputs: true,
            tripleOutput: true  
        },
        
        batch: {
            fn: 'emd_batch',
            config: {
                small: {
                    period_range: [20, 22, 2],      
                    delta_range: [0.5, 0.6, 0.1],   
                    fraction_range: [0.1, 0.2, 0.1] 
                },
                medium: {
                    period_range: [15, 25, 5],      
                    delta_range: [0.3, 0.7, 0.2],   
                    fraction_range: [0.05, 0.2, 0.05] 
                }
            },
            
            fastFn: 'emd_batch_into',
            tripleOutput: true,
            needsMultipleInputs: true
        }
    },
    ttm_squeeze: {
        name: 'TTM Squeeze',
        needsMultipleInputs: true,  
        multipleOutputs: 2,         
        
        safe: {
            fn: 'ttm_squeeze_js',
            params: { 
                bb_period: 20, 
                bb_stddev: 2.0,
                kc_period_ema: 1.0,
                kc_mult_low: 1.5,
                kc_mult_high: 2.0
            },
            needsMultipleInputs: true
        },
        
        fast: {
            allocFn: 'ttm_squeeze_alloc',
            freeFn: 'ttm_squeeze_free',
            computeFn: 'ttm_squeeze_into',
            params: { 
                bb_period: 20, 
                bb_stddev: 2.0,
                kc_period_ema: 1.0,
                kc_mult_low: 1.5,
                kc_mult_high: 2.0
            },
            needsMultipleInputs: true,
            outputCount: 2  
        },
        
    },
    gatorosc: {
        name: 'GatorOsc',
        
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
            resultType: 'GatorOscJsOutput'  
        },
        
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
            quadOutput: true  
        },
        
        batch: {
            fn: 'gatorosc_batch',
            config: {
                small: {
                    jaws_length_range: [13, 13, 0],
                    jaws_shift_range: [8, 8, 0],
                    teeth_length_range: [8, 8, 0],
                    teeth_shift_range: [5, 5, 0],
                    lips_length_range: [5, 5, 0],
                    lips_shift_range: [3, 3, 0]  
                },
                medium: {
                    jaws_length_range: [10, 15, 5],   
                    jaws_shift_range: [6, 10, 2],     
                    teeth_length_range: [6, 10, 2],   
                    teeth_shift_range: [3, 6, 3],     
                    lips_length_range: [3, 6, 3],     
                    lips_shift_range: [2, 4, 2]       
                }
            }
        }
    },
    kurtosis: {
        name: 'Kurtosis',
        
        safe: {
            fn: 'kurtosis_js',
            params: { period: 5 }
        },
        
        fast: {
            allocFn: 'kurtosis_alloc',
            freeFn: 'kurtosis_free',
            computeFn: 'kurtosis_into',
            params: { period: 5 }
        },
        
        batch: {
            fn: 'kurtosis_batch',
            fastFn: 'kurtosis_batch_into',
            config: {
                small: {
                    period_range: [5, 15, 5]       
                },
                medium: {
                    period_range: [5, 50, 5]       
                }
            }
        }
    },
    mab: {
        name: 'MAB (Moving Average Bands)',
        
        safe: {
            fn: 'mab_js',
            params: { fast_period: 10, slow_period: 50, devup: 1.0, devdn: 1.0, fast_ma_type: 'sma', slow_ma_type: 'sma' },
            outputLength: 3  
        },
        
        fast: {
            allocFn: 'mab_alloc',
            freeFn: 'mab_free',
            computeFn: 'mab_into',
            params: { fast_period: 10, slow_period: 50, devup: 1.0, devdn: 1.0, fast_ma_type: 'sma', slow_ma_type: 'sma' },
            tripleOutput: true  
        },
        
        batch: {
            fn: 'mab_batch',
            fastFn: 'mab_batch_into',
            config: {
                small: {
                    fast_period_range: [10, 15, 5],     
                    slow_period_range: [50, 50, 0],     
                    devup_range: [1.0, 2.0, 0.5],       
                    devdn_range: [1.0, 1.0, 0.0],       
                    fast_ma_type: 'sma',
                    slow_ma_type: 'sma'
                    
                },
                medium: {
                    fast_period_range: [10, 20, 5],     
                    slow_period_range: [40, 60, 10],    
                    devup_range: [0.5, 2.5, 0.5],       
                    devdn_range: [0.5, 1.5, 0.5],       
                    fast_ma_type: 'sma',
                    slow_ma_type: 'sma'
                    
                }
            },
            tripleOutput: true
        }
    },
    msw: {
        name: 'MSW',
        
        safe: {
            fn: 'msw_js',
            params: { period: 5 }
        },
        
        fast: {
            allocFn: 'msw_alloc',
            freeFn: 'msw_free',
            computeFn: 'msw_into',
            params: { period: 5 },
            dualOutput: true  
        },
        
        batch: {
            fn: 'msw_batch',
            config: {
                small: {
                    period_range: [5, 15, 5]       
                },
                medium: {
                    period_range: [5, 30, 5]       
                }
            },
            
            fastFn: 'msw_batch_into',
            dualOutput: true
        }
    },
    pma: {
        name: 'PMA',
        
        safe: {
            fn: 'pma_js',
            params: {} 
        },
        
        fast: {
            allocFn: 'pma_alloc',
            freeFn: 'pma_free',
            computeFn: 'pma_into',
            params: {},
            dualOutput: true  
        },
        
        batch: {
            fn: 'pma_batch',
            config: {
                small: {
                    
                    dummy: 0
                },
                medium: {
                    
                    dummy: 0
                }
            },
            
            fastFn: 'pma_batch_into',
            dualOutput: true
        }
    },
    sar: {
        name: 'SAR',
        needsMultipleInputs: true,  
        
        safe: {
            fn: 'sar_js',
            params: { acceleration: 0.02, maximum: 0.2 },
            needsMultipleInputs: true
        },
        
        fast: {
            allocFn: 'sar_alloc',
            freeFn: 'sar_free',
            computeFn: 'sar_into',
            params: { acceleration: 0.02, maximum: 0.2 },
            needsMultipleInputs: true
        },
        
        batch: {
            fn: 'sar_batch',
            config: {
                small: {
                    acceleration_range: [0.01, 0.03, 0.01],  
                    maximum_range: [0.1, 0.3, 0.1]           
                },
                medium: {
                    acceleration_range: [0.01, 0.05, 0.01],  
                    maximum_range: [0.1, 0.5, 0.1]           
                }
            }
        }
    },
    supertrend: {
        name: 'SuperTrend',
        
        safe: {
            fn: 'supertrend_js',
            params: { period: 10, factor: 3.0 },
            needsMultipleInputs: true,
            dualOutput: true  
        },
        
        fast: {
            allocFn: 'supertrend_alloc',
            freeFn: 'supertrend_free',
            computeFn: 'supertrend_into',
            params: { period: 10, factor: 3.0 },
            needsMultipleInputs: true,
            dualOutput: true  
        },
        
        batch: {
            fn: 'supertrend_batch',
            config: {
                small: {
                    period_range: [8, 12, 2],      
                    factor_range: [2.0, 4.0, 1.0]  
                },
                medium: {
                    period_range: [5, 15, 2],      
                    factor_range: [1.0, 5.0, 1.0]  
                }
            },
            needsMultipleInputs: true
        }
    },
    ultosc: {
        name: 'ULTOSC',
        
        safe: {
            fn: 'ultosc_js',
            params: { timeperiod1: 7, timeperiod2: 14, timeperiod3: 28 },
            needsMultipleInputs: true  
        },
        
        fast: {
            allocFn: 'ultosc_alloc',
            freeFn: 'ultosc_free',
            computeFn: 'ultosc_into',
            params: { timeperiod1: 7, timeperiod2: 14, timeperiod3: 28 },
            needsMultipleInputs: true
        },
        
        batch: {
            fn: 'ultosc_batch',
            config: {
                small: {
                    timeperiod1_range: [5, 9, 2],    
                    timeperiod2_range: [12, 16, 2],  
                    timeperiod3_range: [26, 30, 2]   
                },
                medium: {
                    timeperiod1_range: [5, 11, 2],   
                    timeperiod2_range: [10, 18, 2],  
                    timeperiod3_range: [24, 32, 2]   
                }
            },
            needsMultipleInputs: true
        }
    },
    voss: {
        name: 'VOSS',
        
        safe: {
            fn: 'voss_js',
            params: { period: 20, predict: 3, bandwidth: 0.25 },
            outputLength: 2  
        },
        
        fast: {
            allocFn: 'voss_alloc',
            freeFn: 'voss_free',
            computeFn: 'voss_into',
            params: { period: 20, predict: 3, bandwidth: 0.25 },
            dualOutput: true  
        },
        
        batch: {
            fn: 'voss_batch',
            config: {
                small: {
                    period_range: [10, 20, 5],      
                    predict_range: [2, 4, 1],       
                    bandwidth_range: [0.2, 0.3, 0.1] 
                },
                medium: {
                    period_range: [10, 30, 5],      
                    predict_range: [2, 5, 1],       
                    bandwidth_range: [0.1, 0.4, 0.1] 
                }
            },
            fastFn: 'voss_batch_into'
        }
    },
    wavetrend: {
        name: 'WaveTrend',
        
        safe: {
            fn: 'wavetrend_js',
            params: { channel_length: 9, average_length: 12, ma_length: 3, factor: 0.015 },
            tripleOutput: true  
        },
        
        fast: {
            allocFn: 'wavetrend_alloc',
            freeFn: 'wavetrend_free',
            computeFn: 'wavetrend_into',
            params: { channel_length: 9, average_length: 12, ma_length: 3, factor: 0.015 },
            tripleOutput: true  
        },
        
        batch: {
            fn: 'wavetrend_batch',
            config: {
                small: {
                    channel_length_range: [9, 11, 2],      
                    average_length_range: [12, 14, 2],     
                    ma_length_range: [3, 3, 0],            
                    factor_range: [0.015, 0.020, 0.005]    
                },
                medium: {
                    channel_length_range: [7, 13, 2],      
                    average_length_range: [10, 16, 2],     
                    ma_length_range: [3, 5, 1],            
                    factor_range: [0.010, 0.025, 0.005]    
                }
            }
        }
    },
    apo: {
        name: 'APO',
        
        safe: {
            fn: 'apo_js',
            params: { short_period: 10, long_period: 20 }
        },
        
        fast: {
            allocFn: 'apo_alloc',
            freeFn: 'apo_free',
            computeFn: 'apo_into',
            params: { short_period: 10, long_period: 20 }
        },
        
        batch: {
            fn: 'apo_batch',
            config: {
                small: {
                    short_period_range: [5, 15, 5],    
                    long_period_range: [20, 30, 10]    
                },
                medium: {
                    short_period_range: [5, 15, 2],    
                    long_period_range: [20, 40, 5]     
                }
            }
        }
    },
    chop: {
        name: 'CHOP',
        needsMultipleInputs: true,  
        
        safe: {
            fn: 'chop_js',
            params: { period: 14, scalar: 100.0, drift: 1 }
        },
        
        fast: {
            allocFn: 'chop_alloc',
            freeFn: 'chop_free',
            computeFn: 'chop_into',
            params: { period: 14, scalar: 100.0, drift: 1 },
            needsMultipleInputs: true
        },
        
        batch: {
            fn: 'chop_batch',
            fastFn: 'chop_batch_into',
            config: {
                small: {
                    period_range: [10, 20, 5],      
                    scalar_range: [50.0, 100.0, 50.0], 
                    drift_range: [1, 2, 1]          
                    
                },
                medium: {
                    period_range: [10, 30, 5],      
                    scalar_range: [50.0, 150.0, 25.0], 
                    drift_range: [1, 3, 1]          
                    
                }
            },
            needsMultipleInputs: true
        }
    },
    cvi: {
        name: 'CVI',
        needsMultipleInputs: true,  
        
        safe: {
            fn: 'cvi_js',
            params: { period: 10 }
        },
        
        fast: {
            allocFn: 'cvi_alloc',
            freeFn: 'cvi_free',
            computeFn: 'cvi_into',
            params: { period: 10 },
            needsMultipleInputs: true
        },
        
        batch: {
            fn: 'cvi_batch',
            fastFn: 'cvi_batch_into',
            config: {
                small: {
                    period_range: [5, 15, 5]        
                },
                medium: {
                    period_range: [5, 25, 5]        
                }
            },
            needsMultipleInputs: true
        }
    },
    di: {
        name: 'DI',
        needsMultipleInputs: true,  
        
        safe: {
            fn: 'di_js',
            params: { period: 14 }
        },
        
        fast: {
            allocFn: 'di_alloc',
            freeFn: 'di_free',
            computeFn: 'di_into',
            params: { period: 14 },
            needsMultipleInputs: true,
            dualOutput: true  
        },
        
        batch: {
            fn: 'di_batch',
            fastFn: 'di_batch_into',
            config: {
                small: {
                    period_range: [10, 20, 5]      
                },
                medium: {
                    period_range: [10, 30, 5]      
                }
            },
            needsMultipleInputs: true,
            dualOutput: true
        }
    },
    dma: {
        name: 'DMA',
        
        safe: {
            fn: 'dma_js',
            params: { 
                hull_length: 9,
                ema_length: 9,
                ema_gain_limit: 9,
                hull_ma_type: 'sma',
                ema_ma_type: 'ema'
            }
        },
        
        fast: {
            allocFn: 'dma_alloc',
            freeFn: 'dma_free',
            computeFn: 'dma_into',
            params: { 
                hull_length: 9,
                ema_length: 9,
                ema_gain_limit: 9,
                hull_ma_type: 'sma',
                ema_ma_type: 'ema'
            }
        },
        
        batch: {
            fn: 'dma_batch_unified_js',
            config: {
                small: {
                    hull_length_range: [9, 50, 1],
                    ema_length_range: [9, 50, 1]
                },
                medium: {
                    hull_length_range: [5, 30, 5],
                    ema_length_range: [5, 30, 5]
                }
            },
            fastFn: 'dma_batch_into'
        }
    },
    dm: {
        name: 'DM',
        needsMultipleInputs: true,  
        
        safe: {
            fn: 'dm_js',
            params: { period: 14 }
        },
        
        fast: {
            allocFn: 'dm_alloc',
            freeFn: 'dm_free',
            computeFn: 'dm_into',
            params: { period: 14 },
            needsMultipleInputs: true,
            dualOutput: true  
        },
        
        batch: {
            fn: 'dm_batch',
            fastFn: 'dm_batch_into',
            config: {
                small: {
                    period_range: [10, 20, 5]      
                },
                medium: {
                    period_range: [10, 30, 5]      
                }
            },
            needsMultipleInputs: true,
            dualOutput: true
        }
    },
    efi: {
        name: 'EFI',
        needsMultipleInputs: true,  
        
        safe: {
            fn: 'efi_js',
            params: { period: 13 }
        },
        
        fast: {
            allocFn: 'efi_alloc',
            freeFn: 'efi_free',
            computeFn: 'efi_into',
            params: { period: 13 },
            needsMultipleInputs: true
        },
        
        batch: {
            fn: 'efi_batch',
            fastFn: 'efi_batch_into',
            config: {
                small: {
                    period_range: [10, 20, 5]      
                },
                medium: {
                    period_range: [10, 30, 5]      
                }
            },
            needsMultipleInputs: true
        }
    },
    kst: {
        name: 'KST',
        
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
            multiOutput: true  
        },
        
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
            multiOutput: true  
        },
        
        batch: {
            fn: 'kst_batch',
            fastFn: 'kst_batch_into',
            config: {
                small: {
                    sma_period1_range: [8, 12, 2],      
                    sma_period2_range: [8, 12, 2],      
                    sma_period3_range: [8, 12, 2],      
                    sma_period4_range: [12, 18, 3],     
                    roc_period1_range: [8, 12, 2],      
                    roc_period2_range: [12, 18, 3],     
                    roc_period3_range: [18, 22, 2],     
                    roc_period4_range: [25, 35, 5],     
                    signal_period_range: [7, 11, 2]     
                },
                medium: {
                    
                    sma_period1_range: [10, 10, 1],     
                    sma_period2_range: [10, 10, 1],     
                    sma_period3_range: [10, 10, 1],     
                    sma_period4_range: [10, 20, 5],     
                    roc_period1_range: [10, 10, 1],     
                    roc_period2_range: [10, 20, 5],     
                    roc_period3_range: [15, 25, 5],     
                    roc_period4_range: [25, 35, 5],     
                    signal_period_range: [7, 13, 3]     
                }
            },
            multiOutput: true  
        }
    },
    lrsi: {
        name: 'LRSI',
        needsMultipleInputs: true,  
        
        safe: {
            fn: 'lrsi_js',
            params: { alpha: 0.2 }
        },
        
        fast: {
            allocFn: 'lrsi_alloc',
            freeFn: 'lrsi_free',
            computeFn: 'lrsi_into',
            params: { alpha: 0.2 },
            needsMultipleInputs: true
        },
        
        batch: {
            fn: 'lrsi_batch',
            config: {
                small: {
                    alpha_range: [0.1, 0.3, 0.1]     
                },
                medium: {
                    alpha_range: [0.1, 0.5, 0.05]    
                }
            }
        }
    },
    pivot: {
        name: 'PIVOT',
        needsMultipleInputs: true,  
        
        safe: {
            fn: 'pivot_js',
            params: { mode: 3 },  
            
            multiOutput: 9
        },
        
        fast: {
            allocFn: 'pivot_alloc',
            freeFn: 'pivot_free',
            computeFn: 'pivot_into',
            params: { mode: 3 },
            
            multiOutput: 9,
            needsMultipleInputs: true
        },
        
        batch: {
            fn: 'pivot_batch',
            config: {
                small: {
                    mode_range: [0, 4, 1]  
                },
                medium: {
                    mode_range: [0, 4, 1]  
                }
            },
            needsMultipleInputs: true
        }
    },
    qqe: {
        name: 'QQE (Quantitative Qualitative Estimation)',
        
        safe: {
            fn: 'qqe_js',
            params: { rsi_period: 14, smoothing_period: 5, wilders_period: 4.236 }
        },
        
        fast: {
            allocFn: 'qqe_alloc',
            freeFn: 'qqe_free',
            computeFn: 'qqe_into',
            params: { rsi_period: 14, smoothing_period: 5, wilders_period: 4.236 },
            outputCount: 2  
        },
        
        batch: {
            fn: 'qqe_batch',
            config: {
                small: {
                    rsi_period_range: [10, 20, 5],       
                    smoothing_period_range: [3, 7, 2],   
                    wilders_period_range: [3.0, 5.0, 1.0] 
                    
                },
                medium: {
                    rsi_period_range: [10, 30, 5],       
                    smoothing_period_range: [3, 9, 2],   
                    wilders_period_range: [2.0, 5.0, 1.0] 
                    
                }
            },
            
            fastFn: 'qqe_batch_into'
        }
    },
    safezonestop: {
        name: 'SafeZoneStop',
        needsMultipleInputs: true,  
        
        safe: {
            fn: 'safezonestop_js',
            params: { period: 22, mult: 2.5, max_lookback: 3, direction: 'long' }
        },
        
        fast: {
            allocFn: 'safezonestop_alloc',
            freeFn: 'safezonestop_free',
            computeFn: 'safezonestop_into',
            params: { period: 22, mult: 2.5, max_lookback: 3, direction: 'long' },
            needsMultipleInputs: true,
            needsSafeZoneStopInputs: true  
        },
        
        batch: {
            fn: 'safezonestop_batch',
            config: {
                small: {
                    period_range: [14, 30, 8],       
                    mult_range: [2.0, 3.0, 0.5],     
                    max_lookback_range: [2, 4, 1],   
                    direction: 'long'
                },
                medium: {
                    period_range: [10, 50, 10],      
                    mult_range: [2.0, 4.0, 0.5],     
                    max_lookback_range: [2, 6, 2],   
                    direction: 'long'
                }
            }
        }
    },
    stochf: {
        name: 'StochF',
        needsMultipleInputs: true,  
        
        safe: {
            fn: 'stochf_js',
            params: { fastk_period: 5, fastd_period: 3, fastd_matype: 0 }
        },
        
        fast: {
            allocFn: 'stochf_alloc',
            freeFn: 'stochf_free',
            computeFn: 'stochf_into',
            params: { fastk_period: 5, fastd_period: 3, fastd_matype: 0 },
            needsMultipleInputs: true,
            needsStochFInputs: true  
        },
        
        batch: {
            fn: 'stochf_batch',
            config: {
                small: {
                    fastk_range: [5, 14, 1],       
                    fastd_range: [3, 5, 1],        
                    fastd_matype: 0
                },
                medium: {
                    fastk_range: [5, 50, 5],       
                    fastd_range: [3, 10, 1],       
                    fastd_matype: 0
                }
            },
            
            fastFn: 'stochf_batch_into'
        }
    },
    reverse_rsi: {
        name: 'Reverse RSI',
        
        safe: {
            fn: 'reverse_rsi_js',
            params: { period: 14, target_rsi: 50.0 }
        },
        
        fast: {
            allocFn: 'reverse_rsi_alloc',
            freeFn: 'reverse_rsi_free',
            computeFn: 'reverse_rsi_into',
            params: { period: 14, target_rsi: 50.0 }
        }
    },
    ui: {
        name: 'UI (Ulcer Index)',
        
        safe: {
            fn: 'ui_js',
            params: { period: 14, scalar: 100.0 }
        },
        
        fast: {
            allocFn: 'ui_alloc',
            freeFn: 'ui_free',
            computeFn: 'ui_into',
            params: { period: 14, scalar: 100.0 }
        },
        
        batch: {
            fn: 'ui_batch',
            config: {
                small: {
                    period_range: [10, 20, 5],      
                    scalar_range: [50.0, 150.0, 50.0] 
                },
                medium: {
                    period_range: [5, 50, 5],       
                    scalar_range: [50.0, 200.0, 50.0] 
                }
            },
            
            fastFn: 'ui_batch_into'
        }
    },
    wad: {
        name: 'WAD',
        needsMultipleInputs: true,  
        
        safe: {
            fn: 'wad_js',
            params: {}  
        },
        
        fast: {
            allocFn: 'wad_alloc',
            freeFn: 'wad_free',
            computeFn: 'wad_into',
            params: {},
            needsMultipleInputs: true
        },
        
        batch: {
            fn: 'wad_batch',
            config: {
                
                small: {},
                medium: {}
            }
        }
    },
    bollinger_bands_width: {
        name: 'Bollinger Bands Width',
        
        safe: {
            fn: 'bollinger_bands_width_js',
            params: { period: 20, devup: 2.0, devdn: 2.0, matype: "sma", devtype: 0 }
        },
        
        fast: {
            allocFn: 'bollinger_bands_width_alloc',
            freeFn: 'bollinger_bands_width_free',
            computeFn: 'bollinger_bands_width_into',
            params: { period: 20, devup: 2.0, devdn: 2.0, matype: "sma", devtype: 0 }
        },
        
        batch: {
            fn: 'bollinger_bands_width_batch',
            config: {
                small: {
                    period_range: [10, 30, 10],     
                    devup_range: [1.5, 2.5, 0.5],   
                    devdn_range: [2.0, 2.0, 0]      
                },
                medium: {
                    period_range: [10, 40, 10],     
                    devup_range: [1.0, 3.0, 0.5],   
                    devdn_range: [1.5, 2.5, 0.5]    
                }
            }
        }
    },
    buff_averages: {
        name: 'Buff Averages',
        needsMultipleInputs: true,  
        
        safe: {
            fn: 'buff_averages_js',
            params: { 
                fast_period: 10,
                slow_period: 5
            }
        },
        
        fast: {
            allocFn: 'buff_averages_alloc',
            freeFn: 'buff_averages_free',
            computeFn: 'buff_averages_into',
            params: { 
                fast_period: 10,
                slow_period: 5
            },
            needsMultipleInputs: true,
            dualOutput: true  
        },
        
        batch: {
            fn: 'buff_averages_batch_unified_js',
            config: {
                small: {
                    fast_period_range: [10, 20, 1],
                    slow_period_range: [5, 10, 1]
                },
                medium: {
                    fast_period_range: [10, 30, 5],
                    slow_period_range: [5, 15, 5]
                }
            },
            fastFn: 'buff_averages_batch_into',
            needsMultipleInputs: true
        }
    },
    dvdiqqe: {
        name: 'DVDIQQE',
        needsMultipleInputs: true,  
        hasMultipleOutputs: 4,      
        
        safe: {
            fn: 'dvdiqqe',
            params: { 
                period: 13, 
                smoothing_period: 6, 
                fast_multiplier: 2.618, 
                slow_multiplier: 4.236, 
                volume_type: 'default', 
                center_type: 'dynamic' 
            }
        },
        
        fast: {
            allocFn: 'dvdiqqe_alloc',
            freeFn: 'dvdiqqe_free',
            computeFn: 'dvdiqqe_into',
            params: { 
                period: 13, 
                smoothing_period: 6, 
                fast_multiplier: 2.618, 
                slow_multiplier: 4.236, 
                volume_type: 'default', 
                center_type: 'dynamic' 
            },
            needsMultipleInputs: true,
            hasMultipleOutputs: 4
        },
        
        batch: {
            fn: 'dvdiqqe_batch_unified',
            fastFn: 'dvdiqqe_batch_into',
            config: {
                small: {
                    period_range: [10, 20, 5],           
                    smoothing_period_range: [4, 8, 2],   
                    fast_multiplier_range: [2.0, 3.0, 0.5], 
                    slow_multiplier_range: [4.0, 5.0, 0.5], 
                    volume_type: 'default',
                    center_type: 'dynamic'                
                },
                medium: {
                    period_range: [10, 30, 5],           
                    smoothing_period_range: [4, 10, 2],  
                    fast_multiplier_range: [2.0, 3.5, 0.5], 
                    slow_multiplier_range: [3.5, 5.0, 0.5], 
                    volume_type: 'default',
                    center_type: 'dynamic'                
                }
            },
            needsMultipleInputs: true,
            hasMultipleOutputs: 4
        }
    },
    dx: {
        name: 'DX',
        needsMultipleInputs: true,  
        
        safe: {
            fn: 'dx_js',
            params: { period: 14 }
        },
        
        fast: {
            allocFn: 'dx_alloc',
            freeFn: 'dx_free',
            computeFn: 'dx_into',
            params: { period: 14 },
            needsMultipleInputs: true
        },
        
        batch: {
            fn: 'dx_batch',
            fastFn: 'dx_batch_into',
            config: {
                small: {
                    period_range: [10, 20, 5]      
                },
                medium: {
                    period_range: [10, 30, 5]      
                }
            },
            needsMultipleInputs: true
        }
    },
    roc: {
        name: 'ROC',
        
        safe: {
            fn: 'roc_js',
            params: { period: 10 }
        },
        
        fast: {
            allocFn: 'roc_alloc',
            freeFn: 'roc_free',
            computeFn: 'roc_into',
            params: { period: 10 }
        },
        
        batch: {
            fn: 'roc_batch',
            config: {
                small: {
                    period_range: [5, 15, 5]  
                },
                medium: {
                    period_range: [5, 25, 5]  
                }
            }
        }
    },
    rvi: {
        name: 'RVI (Relative Vigor Index)',
        
        safe: {
            fn: 'rvi_js',
            params: { period: 10, ma_len: 14, matype: 1, devtype: 0 }
        },
        
        fast: {
            allocFn: 'rvi_alloc',
            freeFn: 'rvi_free',
            computeFn: 'rvi_into',
            params: { period: 10, ma_len: 14, matype: 1, devtype: 0 }
        },
        
        batch: {
            fn: 'rvi_batch',
            config: {
                small: {
                    period_range: [10, 20, 5],    
                    ma_len_range: [10, 14, 4],    
                    matype_range: [0, 1, 1],      
                    devtype_range: [0, 0, 0]      
                    
                },
                medium: {
                    period_range: [10, 30, 5],    
                    ma_len_range: [10, 20, 5],    
                    matype_range: [0, 1, 1],      
                    devtype_range: [0, 2, 1]      
                    
                }
            }
        }
    },
    stddev: {
        name: 'StdDev',
        
        safe: {
            fn: 'stddev_js',
            params: { period: 5, nbdev: 1.0 }
        },
        
        fast: {
            allocFn: 'stddev_alloc',
            freeFn: 'stddev_free',
            computeFn: 'stddev_into',
            params: { period: 5, nbdev: 1.0 }
        },
        
        batch: {
            fn: 'stddev_batch',
            fastFn: 'stddev_batch_into',
            config: {
                small: {
                    period_range: [5, 15, 5],      
                    nbdev_range: [1.0, 2.0, 0.5]   
                    
                },
                medium: {
                    period_range: [5, 25, 5],      
                    nbdev_range: [0.5, 2.5, 0.5]   
                    
                }
            }
        }
    },
    cci_cycle: {
        name: 'CCI_CYCLE',
        
        safe: {
            fn: 'cci_cycle_js',
            params: { period: 14 }
        },
        
        fast: {
            allocFn: 'cci_cycle_alloc',
            freeFn: 'cci_cycle_free',
            computeFn: 'cci_cycle_into',
            params: { period: 14 }
        },
        
        batch: {
            fn: 'cci_cycle_batch',
            fastFn: 'cci_cycle_batch_into',
            config: {
                small: {
                    period_range: [10, 20, 5]  
                },
                medium: {
                    period_range: [10, 30, 5]  
                }
            }
        }
    },
    uma: {
        name: 'UMA',
        
        safe: {
            fn: 'uma_js',
            params: { 
                accelerator: 1.0, 
                min_length: 5, 
                max_length: 50, 
                smooth_length: 4,
                volume: null 
            }
        },
        
        fast: {
            allocFn: 'uma_alloc',
            freeFn: 'uma_free',
            computeFn: 'uma_into',
            params: { 
                accelerator: 1.0, 
                min_length: 5, 
                max_length: 50, 
                smooth_length: 4 
            }
        },
        
        batch: {
            fn: 'uma_batch',
            config: {
                small: {
                    accelerator_range: [1.0, 1.0, 0.0],     
                    min_length_range: [5, 5, 0],            
                    max_length_range: [50, 50, 0],          
                    smooth_length_range: [4, 4, 0]          
                },
                medium: {
                    accelerator_range: [0.5, 2.0, 0.5],     
                    min_length_range: [5, 15, 5],           
                    max_length_range: [30, 60, 10],         
                    smooth_length_range: [2, 6, 2]          
                }
            }
        }
    },
    vama: {
        name: 'VAMA (Volatility Adjusted MA)',
        
        safe: {
            fn: 'vama_js',
            params: { base_period: 14, vol_period: 14, smoothing: true, smooth_type: 2, smooth_period: 3 },
            run: (wasm, data, params) => wasm.vama_js(
                data.close, 
                params.base_period, 
                params.vol_period,
                params.smoothing,
                params.smooth_type,
                params.smooth_period
            )
        },
        
        fast: {
            alloc: 'vama_alloc',
            free: 'vama_free',
            fn: 'vama_into',
            params: { base_period: 14, vol_period: 14, smoothing: true, smooth_type: 2, smooth_period: 3 },
            run: (wasm, data, inPtr, outPtr, len, params) => wasm.vama_into(
                inPtr, 
                outPtr, 
                len, 
                params.base_period, 
                params.vol_period,
                params.smoothing,
                params.smooth_type,
                params.smooth_period
            )
        },
        
        batch: {
            fn: 'vama_batch_js',
            params: {
                base_period_range: [10, 20, 2],
                vol_period_range: [10, 20, 2],
                smoothing: true,
                smooth_type: 2,
                smooth_period: 3
            },
            run: (wasm, data, params) => wasm.vama_batch_js(
                data.close,
                params.base_period_range,
                params.vol_period_range,
                params.smoothing,
                params.smooth_type,
                params.smooth_period
            )
        }
    },
    halftrend: {
        name: 'HalfTrend',
        needsMultipleInputs: true,  
        
        safe: {
            fn: 'halftrend_js',
            params: { 
                amplitude: 2, 
                channel_deviation: 2, 
                atr_period: 100 
            }
        },
        
        fast: {
            allocFn: 'halftrend_alloc',
            freeFn: 'halftrend_free',
            computeFn: 'halftrend_into',
            params: { 
                amplitude: 2, 
                channel_deviation: 2, 
                atr_period: 100 
            },
            needsMultipleInputs: true,
            multipleOutputs: 6  
        },
        
        batch: {
            fn: 'halftrend_batch',
            fastFn: 'halftrend_batch_into',
            config: {
                small: {
                    amplitude_range: [2, 4, 1],           
                    channel_deviation_range: [1.5, 2.5, 0.5], 
                    atr_period_range: [50, 100, 50]       
                    
                },
                medium: {
                    amplitude_range: [2, 6, 1],           
                    channel_deviation_range: [1.0, 3.0, 0.5], 
                    atr_period_range: [50, 150, 25]       
                    
                }
            },
            needsMultipleInputs: true,
            multipleOutputs: 6
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

        
        this.loadData();
    }

    loadData() {
        console.log('Loading test data...');
        
        const csvPath = join(__dirname, '../src/data/1MillionCandles.csv');
        const content = readFileSync(csvPath, 'utf8');
        const lines = content.trim().split('\n');
        
        
        lines.shift();
        
        
        const opens = [];
        const highs = [];
        const lows = [];
        
        
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
                volumes.push(parseFloat(parts[5])); 
            }
        }
        
        
        this.data['10k'] = new Float64Array(closes.slice(0, 10_000));
        this.data['100k'] = new Float64Array(closes.slice(0, 100_000));
        this.data['1M'] = new Float64Array(closes);
        
        
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
        
        
        this.vwapData = {
            '10k': {
                timestamps: new Float64Array(timestamps.slice(0, 10_000)),
                volumes: new Float64Array(volumes.slice(0, 10_000)),
                prices: new Float64Array(closes.slice(0, 10_000).map((c, i) => 
                    (highs[i] + lows[i] + c) / 3.0  
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
            
            let warmupElapsed = 0;
            let warmupIterations = 0;
            const warmupStart = performance.now();
            
            while (warmupElapsed < CONFIG.warmupTargetMs) {
                fn();
                warmupIterations++;
                warmupElapsed = performance.now() - warmupStart;
            }

            
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

            
            samples.sort((a, b) => a - b);
            const median = samples[Math.floor(samples.length / 2)];
            const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
            const min = samples[0];
            const max = samples[samples.length - 1];
            
            
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
                
                if (indicatorConfig.fast && indicatorConfig.fast.inputs) {
                    const ohlc = this.ohlcData[sizeName];
                    const inputPtrs = {};
                    
                    
                    if (indicatorConfig.name === 'VWMACD (Volume Weighted MACD)' && outputCount === 3) {
                        
                        closePtr = this.wasm[allocFn](len);
                        volumePtr = this.wasm[allocFn](len);
                        
                        
                        outPtr = this.wasm[allocFn](len);    
                        outPtr2 = this.wasm[allocFn](len);   
                        outPtr3 = this.wasm[allocFn](len);   
                        
                        
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
                        
                        
                        Object.assign(this, { closePtr, volumePtr, outPtr, outPtr2, outPtr3 });
                    } else {
                        
                        
                        for (const input of indicatorConfig.fast.inputs) {
                            inputPtrs[input] = this.wasm[allocFn](len);
                        }
                        outPtr = this.wasm[allocFn](len);
                        
                        
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
                            
                            const paramArray = [];
                            for (const input of indicatorConfig.fast.inputs) {
                                paramArray.push(inputPtrs[input]);
                            }
                            paramArray.push(outPtr);
                            paramArray.push(len);
                            
                            
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
                        
                        
                        Object.assign(this, { inputPtrs, outPtr });
                    }
                }
                
                else if (indicatorConfig.fast.needsMultipleInputs) {
                    const ohlc = this.ohlcData[sizeName];
                    
                    
                    highPtr = this.wasm[allocFn](len);
                    lowPtr = this.wasm[allocFn](len);
                    closePtr = this.wasm[allocFn](len);
                    
                    
                    if (indicatorConfig.fast.outputIsU8 && indicatorConfig.fast.allocU8Fn) {
                        outPtr = this.wasm[indicatorConfig.fast.allocU8Fn](len);
                    } else {
                        outPtr = this.wasm[allocFn](len);
                    }
                    
                    
                    if (indicatorConfig.name === 'ADOSC' || indicatorConfig.name === 'EMD') {
                        volumePtr = this.wasm[allocFn](len);
                    }
                    
                    
                    if (indicatorConfig.fast.tripleOutput) {
                        outPtr2 = this.wasm[allocFn](len);
                        outPtr3 = this.wasm[allocFn](len);
                    } else if (indicatorConfig.fast.dualOutput) {
                        outPtr2 = this.wasm[allocFn](len);
                    }
                    
                    
                    
                    if (indicatorConfig.name === 'TTM Trend') {
                        
                        const sourceView = new Float64Array(this.wasm.__wasm.memory.buffer, highPtr, len);
                        const closeView = new Float64Array(this.wasm.__wasm.memory.buffer, lowPtr, len);
                        
                        
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
                    
                    
                    if (indicatorConfig.name === 'ADOSC' || indicatorConfig.name === 'EMD') {
                        const volumeView = new Float64Array(this.wasm.__wasm.memory.buffer, volumePtr, len);
                        
                        volumeView.set(ohlc.volume.slice(0, len));
                    }
                    
                    
                    outPtr2 = (indicatorConfig.fast.dualOutput || indicatorConfig.fast.needsStochFInputs) ? this.wasm[allocFn](len) : null;
                    
                    
                    outPtr3 = indicatorConfig.fast.tripleOutput ? this.wasm[allocFn](len) : null;
                    
                    
                    
                    const result = this.benchmarkFunction(() => {
                        
                        const modifiedConfig = Object.assign({}, indicatorConfig.fast, { name: indicatorConfig.name });
                        const paramArray = this.prepareFastParams(params, null, outPtr, len, modifiedConfig, highPtr, lowPtr, closePtr, indicatorConfig.fast.dualOutput || indicatorConfig.fast.tripleOutput, outPtr2, volumePtr, null, null, null, outPtr3, null, null, null, outPtr3);
                        this.wasm[computeFn].apply(this.wasm, paramArray);
                    }, benchName, {
                        dataSize: len,
                        api: 'fast',
                        indicator: indicatorKey
                    });

                    this.results[benchName] = result;
                    this.printResult(result);
                } else if (indicatorConfig.fast.needsVwapInputs) {
                    
                    const vwap = this.vwapData[sizeName];
                    
                    
                    timestampsPtr = this.wasm[allocFn](len);
                    volumesPtr = this.wasm[allocFn](len);
                    pricesPtr = this.wasm[allocFn](len);
                    outPtr = this.wasm[allocFn](len);
                    
                    
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
                    
                    inPtr = this.wasm[allocFn](len);
                    outPtr = this.wasm[allocFn](len);   
                    outPtr2 = this.wasm[allocFn](len);  
                    outPtr3 = this.wasm[allocFn](len);  
                    
                    
                    const inView = new Float64Array(this.wasm.__wasm.memory.buffer, inPtr, len);
                    inView.set(data);
                    
                    const result = this.benchmarkFunction(() => {
                        const paramArray = [inPtr, outPtr, outPtr2, outPtr3, len];
                        
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
                } else if (outputCount === 4) {
                    
                    inPtr = this.wasm[allocFn](len);
                    const outPtr1 = this.wasm[allocFn](len);   
                    const outPtr2 = this.wasm[allocFn](len);   
                    const outPtr3 = this.wasm[allocFn](len);   
                    const outPtr4 = this.wasm[allocFn](len);   
                    
                    
                    const inView = new Float64Array(this.wasm.__wasm.memory.buffer, inPtr, len);
                    inView.set(data);
                    
                    const result = this.benchmarkFunction(() => {
                        const paramArray = [inPtr, outPtr1, outPtr2, outPtr3, outPtr4, len];
                        
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
                    
                    
                    outPtr = outPtr1;
                    outPtr2 = outPtr2;
                    outPtr3 = outPtr3;
                    this.outPtr4 = outPtr4;  
                } else if (indicatorConfig.fast.quadOutput) {
                    
                    inPtr = this.wasm[allocFn](len);
                    outPtr = this.wasm[allocFn](len);   
                    outPtr2 = this.wasm[allocFn](len);  
                    outPtr3 = this.wasm[allocFn](len);  
                    const outPtr4 = this.wasm[allocFn](len);  
                    
                    
                    const inView = new Float64Array(this.wasm.__wasm.memory.buffer, inPtr, len);
                    inView.set(data);
                    
                    const result = this.benchmarkFunction(() => {
                        const paramArray = [inPtr, outPtr, outPtr2, outPtr3, outPtr4, len];
                        
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
                    
                    
                    if (outPtr4) this.wasm[freeFn](outPtr4, len);
                } else if (outputCount === 4) {
                    
                    inPtr = this.wasm[allocFn](len);
                    outPtr = this.wasm[allocFn](len);   
                    outPtr2 = this.wasm[allocFn](len);  
                    outPtr3 = this.wasm[allocFn](len);  
                    const outPtr4 = this.wasm[allocFn](len);  
                    
                    
                    const inView = new Float64Array(this.wasm.__wasm.memory.buffer, inPtr, len);
                    inView.set(data);
                    
                    const result = this.benchmarkFunction(() => {
                        const paramArray = [inPtr, outPtr, outPtr2, outPtr3, outPtr4, len];
                        
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
                    
                    
                    this.outPtr4 = outPtr4;
                } else {
                    
                    inPtr = this.wasm[allocFn](len);
                    outPtr = this.wasm[allocFn](len);
                    outPtr2 = indicatorConfig.fast.dualOutput ? this.wasm[allocFn](len) : null;
                    outPtr3 = indicatorConfig.fast.tripleOutput ? this.wasm[allocFn](len) : null;
                    
                    
                    const inView = new Float64Array(this.wasm.__wasm.memory.buffer, inPtr, len);
                    inView.set(data);
                    
                    const result = this.benchmarkFunction(() => {
                        const paramArray = this.prepareFastParams(params, inPtr, outPtr, len, indicatorConfig.fast, null, null, null, indicatorConfig.fast.dualOutput, outPtr2, null, null, null, null, outPtr3);
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
                
                if (indicatorConfig.fast.needsVwapInputs) {
                    if (timestampsPtr) this.wasm[freeFn](timestampsPtr, len);
                    if (volumesPtr) this.wasm[freeFn](volumesPtr, len);
                    if (pricesPtr) this.wasm[freeFn](pricesPtr, len);
                    if (outPtr) this.wasm[freeFn](outPtr, len);
                } else if (this.inputPtrs) {
                    
                    for (const ptr of Object.values(this.inputPtrs)) {
                        if (ptr) this.wasm[freeFn](ptr, len);
                    }
                    delete this.inputPtrs;
                    if (this.outPtr) {
                        this.wasm[freeFn](this.outPtr, len);
                        delete this.outPtr;
                    }
                } else if (this.closePtr && this.volumePtr) {
                    
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
                } else if (outputCount === 4) {
                    if (inPtr) this.wasm[freeFn](inPtr, len);
                    if (outPtr) this.wasm[freeFn](outPtr, len);
                    if (outPtr2) this.wasm[freeFn](outPtr2, len);
                    if (outPtr3) this.wasm[freeFn](outPtr3, len);
                    if (this.outPtr4) {
                        this.wasm[freeFn](this.outPtr4, len);
                        delete this.outPtr4;
                    }
                } else if (indicatorConfig.fast.quadOutput) {
                    if (inPtr) this.wasm[freeFn](inPtr, len);
                    if (outPtr) this.wasm[freeFn](outPtr, len);
                    if (outPtr2) this.wasm[freeFn](outPtr2, len);
                    if (outPtr3) this.wasm[freeFn](outPtr3, len);
                    
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

        
        const data = this.data['10k'];
        const sizeName = '10k';
        
        for (const [configName, batchConfig] of Object.entries(config)) {
            const benchName = `${indicatorKey}_batch_${configName}`;
            
            const result = this.benchmarkFunction(() => {
                if (indicatorConfig.needsVwapInputs || indicatorConfig.batch?.needsVwapInputs) {
                    const vwap = this.vwapData[sizeName];
                    wasmFn.call(this.wasm, vwap.timestamps, vwap.volumes, vwap.prices, { anchor_range: batchConfig.anchor_range });
                } else if (indicatorKey === 'ad') {
                    
                    const ohlc = this.ohlcData[sizeName];
                    
                    const rows = 10;
                    const cols = Math.floor(ohlc.high.length / rows);
                    const flatSize = rows * cols;
                    
                    
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
                    
                    if (indicatorConfig.name === 'ADOSC' || indicatorConfig.name === 'EMD') {
                        
                        wasmFn.call(this.wasm, ohlc.high, ohlc.low, ohlc.close, ohlc.volume, batchConfig);
                    } else if (indicatorConfig.name === 'EMV') {
                        
                        wasmFn.call(this.wasm, ohlc.high, ohlc.low, ohlc.close, ohlc.volume, batchConfig);
                    } else if (indicatorConfig.name === 'AroonOsc' || indicatorConfig.name === 'ACOSC') {
                        
                        wasmFn.call(this.wasm, ohlc.high, ohlc.low, batchConfig);
                    } else if (indicatorConfig.name === 'SafeZoneStop') {
                        
                        wasmFn.call(this.wasm, ohlc.high, ohlc.low, batchConfig);
                    } else if (indicatorConfig.name === 'TTM Trend') {
                        
                        const hl2 = new Float64Array(ohlc.high.length);
                        for (let i = 0; i < ohlc.high.length; i++) {
                            hl2[i] = (ohlc.high[i] + ohlc.low[i]) / 2;
                        }
                        wasmFn.call(this.wasm, hl2, ohlc.close, batchConfig);
                    } else {
                        wasmFn.call(this.wasm, ohlc.high, ohlc.low, ohlc.close, batchConfig);
                    }
                } else if ((indicatorKey === 'pwma' || indicatorKey === 'supersmoother') && batchConfig.period_range) {
                    
                    const [start, end, step] = batchConfig.period_range;
                    wasmFn.call(this.wasm, data, start, end, step);
                } else if (indicatorKey === 'trendflex' || indicatorKey === 'wilders' || indicatorKey === 'alma' || 
                           indicatorKey === 'highpass' || indicatorKey === 'jsa' || indicatorKey === 'maaq' || 
                           indicatorKey === 'smma' || indicatorKey === 'ehlers_itrend' ||
                           indicatorKey === 'cwma' || indicatorKey === 'decycler' || indicatorKey === 'dema' || indicatorKey === 'epma' || 
                           indicatorKey === 'jma' || indicatorKey === 'highpass_2_pole' || indicatorKey === 'nma' || 
                           indicatorKey === 'sma' || indicatorKey === 'supersmoother_3_pole' || indicatorKey === 'ema' || 
                           indicatorKey === 'tema' || indicatorKey === 'gaussian' || indicatorKey === 'hwma' || 
                           indicatorKey === 'mwdx' || indicatorKey === 'srwma' || indicatorKey === 'linreg' || 
                           indicatorKey === 'sinwma' || indicatorKey === 'zlema' || indicatorKey === 'adx' || 
                           indicatorKey === 'bandpass' || indicatorKey === 'tsf') {
                    
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
            
            
            let totalCombinations = 1;
            if (batchConfig.period_range) {
                const periods = Math.floor((batchConfig.period_range[1] - batchConfig.period_range[0]) / batchConfig.period_range[2]) + 1;
                totalCombinations = periods;
                
                
                if (batchConfig.offset_range) {
                    const offsets = Math.floor((batchConfig.offset_range[1] - batchConfig.offset_range[0]) / batchConfig.offset_range[2]) + 1;
                    totalCombinations *= offsets;
                }
                if (batchConfig.sigma_range) {
                    const sigmas = Math.floor((batchConfig.sigma_range[1] - batchConfig.sigma_range[0]) / batchConfig.sigma_range[2]) + 1;
                    totalCombinations *= sigmas;
                }
                
                
                if (batchConfig.volume_factor_range) {
                    const vFactors = Math.floor((batchConfig.volume_factor_range[1] - batchConfig.volume_factor_range[0]) / batchConfig.volume_factor_range[2]) + 1;
                    totalCombinations *= vFactors;
                }
                
                
                if (batchConfig.bandwidth_range) {
                    const bandwidths = Math.floor((batchConfig.bandwidth_range[1] - batchConfig.bandwidth_range[0]) / batchConfig.bandwidth_range[2]) + 1;
                    totalCombinations *= bandwidths;
                }
                
                
                if (batchConfig.k_range) {
                    const kValues = Math.floor((batchConfig.k_range[1] - batchConfig.k_range[0]) / batchConfig.k_range[2]) + 1;
                    totalCombinations *= kValues;
                }
            }
            
            
            if (batchConfig.hp_period_range && batchConfig.k_range) {
                const hpPeriods = Math.floor((batchConfig.hp_period_range[1] - batchConfig.hp_period_range[0]) / batchConfig.hp_period_range[2]) + 1;
                const kValues = Math.floor((batchConfig.k_range[1] - batchConfig.k_range[0]) / batchConfig.k_range[2]) + 1;
                totalCombinations = hpPeriods * kValues;
            }
            
            
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
        
        if (indicatorConfig.safe?.needsVwapInputs || indicatorConfig.needsVwapInputs) {
            const vwap = this.vwapData[sizeName];
            const result = [vwap.timestamps, vwap.volumes, vwap.prices];
            
            
            for (const value of Object.values(params)) {
                result.push(value);
            }
            
            return result;
        }
        
        
        if (indicatorConfig.safe && indicatorConfig.safe.inputs) {
            const result = [];
            const ohlc = this.ohlcData[sizeName];
            
            
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
            
            
            for (const value of Object.values(params)) {
                result.push(value);
            }
            
            return result;
        }
        
        
        if (indicatorConfig.needsMultipleInputs || (indicatorConfig.safe && indicatorConfig.safe.needsMultipleInputs)) {
            const ohlc = this.ohlcData[sizeName];
            
            
            if (indicatorConfig.name === 'ACOSC') {
                const result = [ohlc.high, ohlc.low];
                
                
                for (const value of Object.values(params)) {
                    result.push(value);
                }
                
                return result;
            }
            
            
            if (indicatorConfig.name === 'AroonOsc') {
                const result = [ohlc.high, ohlc.low];
                
                
                for (const value of Object.values(params)) {
                    result.push(value);
                }
                
                return result;
            }
            
            
            if (indicatorConfig.name === 'SAR') {
                const result = [ohlc.high, ohlc.low];
                
                
                for (const value of Object.values(params)) {
                    result.push(value);
                }
                
                return result;
            }
            
            
            if (indicatorConfig.name === 'SafeZoneStop') {
                const result = [ohlc.high, ohlc.low];
                
                
                result.push(params.period);
                result.push(params.mult);
                result.push(params.max_lookback);
                result.push(params.direction);
                
                return result;
            }
            
            
            if (indicatorConfig.name === 'ADOSC') {
                const result = [ohlc.high, ohlc.low, ohlc.close, ohlc.volume];
                
                
                for (const value of Object.values(params)) {
                    result.push(value);
                }
                
                return result;
            }
            
            
            if (indicatorConfig.name === 'EMV') {
                const result = [ohlc.high, ohlc.low, ohlc.close, ohlc.volume];
                
                
                
                return result;
            }
            
            
            if (indicatorConfig.name === 'EMD') {
                const result = [ohlc.high, ohlc.low, ohlc.close, ohlc.volume];
                
                
                for (const value of Object.values(params)) {
                    result.push(value);
                }
                
                return result;
            }
            
            
            if (indicatorConfig.name === 'TTM Trend') {
                
                const hl2 = new Float64Array(ohlc.high.length);
                for (let i = 0; i < ohlc.high.length; i++) {
                    hl2[i] = (ohlc.high[i] + ohlc.low[i]) / 2;
                }
                
                const result = [hl2, ohlc.close];
                
                
                for (const value of Object.values(params)) {
                    result.push(value);
                }
                
                return result;
            }
            
            
            const result = [ohlc.high, ohlc.low, ohlc.close];
            
            
            for (const value of Object.values(params)) {
                result.push(value);
            }
            
            return result;
        }
        
        
        const result = [data];
        
        
        for (const value of Object.values(params)) {
            result.push(value);
        }
        
        return result;
    }

    /**
     * Prepare parameters for batch API call
     */
    prepareBatchParams(indicatorKey, data, batchConfig, sizeName) {
        
        if (indicatorKey === 'mama') {
            
            const fast = batchConfig.fast_limit_range || [0.5, 0.5, 0];
            const slow = batchConfig.slow_limit_range || [0.05, 0.05, 0];
            return [data, fast[0], fast[1], fast[2], slow[0], slow[1], slow[2]];
        } else if (indicatorKey === 'sqwma' || indicatorKey === 'fwma' || indicatorKey === 'hma' || indicatorKey === 'kama' || indicatorKey === 'wma') {
            
            const period = batchConfig.period_range;
            return [data, period[0], period[1], period[2]];
        } else if (indicatorKey === 'vpwma') {
            
            const period = batchConfig.period_range || [14, 14, 1];
            const power = batchConfig.power_range || [0.382, 0.382, 0.1];
            return [data, period[0], period[1], period[2], power[0], power[1], power[2]];
        } else if (indicatorKey === 'swma' || indicatorKey === 'trima') {
            
            return [data, { period_range: batchConfig.period_range }];
        } else if (indicatorKey === 'vwap') {
            
            const vwap = this.vwapData[Object.keys(this.vwapData)[0]]; 
            return [vwap.timestamps, vwap.volumes, vwap.prices, { anchor_range: batchConfig.anchor_range }];
        } else if (indicatorKey === 'tilson') {
            
            return [data, { 
                period_range: batchConfig.period_range,
                volume_factor_range: batchConfig.volume_factor_range || [0.0, 0.0, 0.0]
            }];
        } else if (indicatorKey === 'vwma') {
            
            const ohlc = this.ohlcData[sizeName];
            return [ohlc.close, ohlc.volume, { period_range: batchConfig.period_range }];
        } else if (indicatorKey === 'vwmacd') {
            
            const ohlc = this.ohlcData[sizeName];
            return [ohlc.close, ohlc.volume, batchConfig];
        } else if (batchConfig.period_range) {
            
            const period = batchConfig.period_range;
            const result = [data, [period[0], period[1], period[2]]];
            
            
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
            
            return [data, batchConfig];
        }
    }

    /**
     * Prepare parameters for fast API call
     */
    prepareFastParams(params, inPtr, outPtr, len, indicatorConfig, highPtr, lowPtr, closePtr, dualOutput = false, outPtr2 = null, volumePtr = null, timestampsPtr = null, volumesPtr = null, pricesPtr = null, outPtr3 = null) {
        
        if (indicatorConfig.needsVwapInputs) {
            
            const result = [timestampsPtr, volumesPtr, pricesPtr, outPtr, len];
            
            
            for (const value of Object.values(params)) {
                result.push(value);
            }
            
            return result;
        }
        
        
        if (indicatorConfig.needsMultipleInputs || (indicatorConfig.fast && indicatorConfig.fast.needsMultipleInputs)) {
            
            if (indicatorConfig.name === 'ACOSC') {
                const result = [highPtr, lowPtr, outPtr, outPtr2, len];
                
                
                for (const value of Object.values(params)) {
                    result.push(value);
                }
                
                return result;
            }
            
            
            if (indicatorConfig.name === 'AroonOsc') {
                const result = [highPtr, lowPtr, outPtr, len, params.length];
                return result;
            }
            
            
            if (indicatorConfig.name === 'SAR') {
                const result = [highPtr, lowPtr, outPtr, len];
                
                
                for (const value of Object.values(params)) {
                    result.push(value);
                }
                
                return result;
            }
            
            
            if (indicatorConfig.name === 'SafeZoneStop') {
                const result = [highPtr, lowPtr, outPtr, len];
                
                
                result.push(params.period);
                result.push(params.mult);
                result.push(params.max_lookback);
                result.push(params.direction);
                
                return result;
            }
            
            
            if (indicatorConfig.name === 'ADOSC') {
                const result = [highPtr, lowPtr, closePtr, volumePtr, outPtr, len, params.short_period, params.long_period];
                return result;
            }
            
            
            if (indicatorConfig.name === 'EMV') {
                const result = [highPtr, lowPtr, closePtr, volumePtr, outPtr, len];
                return result;
            }
            
            
            if (indicatorConfig.name === 'VI (Vortex Indicator)') {
                const result = [highPtr, lowPtr, closePtr, outPtr, outPtr2, len, params.period];
                return result;
            }
            
            
            if (indicatorConfig.name === 'Donchian') {
                const result = [highPtr, lowPtr, outPtr, outPtr2, outPtr3, len, params.period];
                return result;
            }
            
            
            if (indicatorConfig.name === 'EMD') {
                const result = [highPtr, lowPtr, closePtr, volumePtr, outPtr, outPtr2, outPtr3 || outPtr, len, params.period, params.delta, params.fraction];
                return result;
            }
            
            
            if (indicatorConfig.needsStochFInputs) {
                const result = [highPtr, lowPtr, closePtr, outPtr, outPtr2, len];
                
                
                result.push(params.fastk_period);
                result.push(params.fastd_period);
                result.push(params.fastd_matype);
                
                return result;
            }
            
            
            if (indicatorConfig.name === 'VWMACD (Volume Weighted MACD)') {
                const result = [closePtr, volumePtr, outPtr, outPtr2, outPtr3, len];
                
                
                result.push(params.fast_period);
                result.push(params.slow_period);
                result.push(params.signal_period);
                result.push(params.fast_ma_type);
                result.push(params.slow_ma_type);
                result.push(params.signal_ma_type);
                
                return result;
            }
            
            
            if (indicatorConfig.name === 'TTM Trend') {
                const result = [highPtr, lowPtr, outPtr, len, params.period];  
                return result;
            }
            
            
            const result = [highPtr, lowPtr, closePtr, outPtr, len];
            
            
            for (const value of Object.values(params)) {
                result.push(value);
            }
            
            return result;
        }
        
        
        if (indicatorConfig.name === 'MACD') {
            const result = [inPtr, outPtr, outPtr2, outPtr3, len];
            
            
            for (const value of Object.values(params)) {
                result.push(value);
            }
            
            return result;
        }
        
        
        
        const result = dualOutput 
            ? [inPtr, outPtr, outPtr2, len]
            : [inPtr, outPtr, len];
        
        
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
        
        
        const byIndicator = {};
        for (const [name, result] of Object.entries(this.results)) {
            const indicator = result.indicator;
            if (!byIndicator[indicator]) {
                byIndicator[indicator] = [];
            }
            byIndicator[indicator].push(result);
        }

        
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

        
        if (indicatorConfig.safe) {
            this.benchmarkSafeAPI(indicatorKey, indicatorConfig);
        }

        
        if (indicatorConfig.fast) {
            this.benchmarkFastAPI(indicatorKey, indicatorConfig);
        }

        
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

        
        for (const indicatorKey of indicators) {
            const config = INDICATORS[indicatorKey];
            if (config) {
                await this.runIndicator(indicatorKey, config);
            } else {
                console.log(`\nWarning: Unknown indicator '${indicatorKey}'`);
            }
        }

        
        this.printSummary();
    }
}


async function main() {
    const args = process.argv.slice(2);
    
    
    if (!global.gc && CONFIG.disableGC) {
        console.warn('\nWarning: GC control not available. Run with: node --expose-gc wasm_indicator_benchmark.js\n');
    }
    
    
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

    
    if (indicators.length === 0) {
        indicators = Object.keys(INDICATORS);
    }

    const benchmark = new WasmIndicatorBenchmark();
    await benchmark.run({ indicators });
}


if (import.meta.url.startsWith('file://')) {
    main().catch(console.error);
}

export { WasmIndicatorBenchmark, INDICATORS };