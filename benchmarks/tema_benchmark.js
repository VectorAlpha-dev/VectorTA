// Quick TEMA benchmark test
import { run } from './wasm_indicator_benchmark.js';

// Only benchmark TEMA
process.env.INDICATORS = 'tema';

// Run the benchmark
run();