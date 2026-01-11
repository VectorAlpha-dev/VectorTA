/**
 * Utilities for comparing WASM binding outputs with native Rust outputs
 */
import { exec } from 'child_process';
import { promisify } from 'util';
import path from 'path';
import { fileURLToPath } from 'url';
import { existsSync } from 'fs';

const execAsync = promisify(exec);
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.join(__dirname, '../..');

/**
 * Get Rust output for an indicator
 * @param {string} indicatorName - Name of the indicator
 * @param {string} source - Data source (default: 'close')
 * @returns {Promise<Object>} Rust output data
 */
export async function getRustOutput(indicatorName, source = 'close') {
    
    const binaryPath = path.join(projectRoot, 'target', 'release', 'generate_references');
    const binaryExists = existsSync(binaryPath) || existsSync(binaryPath + '.exe');
    
    if (!binaryExists) {
        
        
        let buildError;
        for (let attempt = 0; attempt < 3; attempt++) {
            try {
                await execAsync('cargo build --release --bin generate_references', {
                    cwd: projectRoot,
                    env: { ...process.env, CARGO_INCREMENTAL: '0' } 
                });
                buildError = null;
                break;
            } catch (error) {
                buildError = error;
                if (attempt < 2) {
                    
                    await new Promise(resolve => setTimeout(resolve, 1000));
                }
            }
        }
        
        if (buildError) {
            throw new Error(`Failed to build reference generator: ${buildError.message}`);
        }
    }
    
    
    try {
        const { stdout, stderr } = await execAsync(
            `cargo run --release --bin generate_references -- ${indicatorName} ${source}`,
            {
                cwd: projectRoot,
                maxBuffer: 10 * 1024 * 1024 
            }
        );
        
        if (stderr) {
            console.warn(`Reference generator stderr: ${stderr}`);
        }
        
        return JSON.parse(stdout);
    } catch (error) {
        throw new Error(`Failed to generate reference for ${indicatorName}: ${error.message}`);
    }
}

/**
 * Compare WASM output with Rust output
 * @param {string} indicatorName - Name of the indicator
 * @param {Float64Array|Array} wasmOutput - Output from WASM binding
 * @param {string} source - Data source (default: 'close')
 * @param {Object} params - Parameters used
 * @param {number} tolerance - Comparison tolerance (default: 1e-8)
 * @returns {Promise<boolean>} True if outputs match
 */
export async function compareWithRust(indicatorName, wasmOutput, source = 'close', params = null, tolerance = 1e-8) {
    const rustData = await getRustOutput(indicatorName, source);
    
    
    if (indicatorName === 'damiani_volatmeter' && typeof wasmOutput === 'object' && 'vol' in wasmOutput && 'anti' in wasmOutput) {
        
        const rustVol = rustData.vol;
        const rustAnti = rustData.anti;
        
        
        if (params) {
            const rustParams = rustData.params;
            for (const [key, value] of Object.entries(params)) {
                if (rustParams[key] !== undefined && rustParams[key] !== value) {
                    throw new Error(`Parameter mismatch for ${key}: Rust=${rustParams[key]}, WASM=${value}`);
                }
            }
        }
        
        
        compareArrays(wasmOutput.vol, rustVol, indicatorName + ' vol', tolerance);
        
        compareArrays(wasmOutput.anti, rustAnti, indicatorName + ' anti', tolerance);
        
        return true;
    }
    
    if (indicatorName === 'di' && typeof wasmOutput === 'object' && 'plus' in wasmOutput && 'minus' in wasmOutput) {
        
        const rustPlus = rustData.plus;
        const rustMinus = rustData.minus;
        
        
        if (params) {
            const rustParams = rustData.params;
            for (const [key, value] of Object.entries(params)) {
                if (rustParams[key] !== undefined && rustParams[key] !== value) {
                    throw new Error(`Parameter mismatch for ${key}: Rust=${rustParams[key]}, WASM=${value}`);
                }
            }
        }
        
        
        compareArrays(wasmOutput.plus, rustPlus, indicatorName + ' plus', tolerance);
        
        compareArrays(wasmOutput.minus, rustMinus, indicatorName + ' minus', tolerance);
        
        return true;
    }
    
    
    const rustOutput = rustData.values;
    
    
    if (params) {
        const rustParams = rustData.params;
        for (const [key, value] of Object.entries(params)) {
            if (rustParams[key] !== undefined && rustParams[key] !== value) {
                throw new Error(`Parameter mismatch for ${key}: Rust=${rustParams[key]}, WASM=${value}`);
            }
        }
    }
    
    
    if (wasmOutput.length !== rustOutput.length) {
        throw new Error(`Length mismatch: WASM=${wasmOutput.length}, Rust=${rustOutput.length}`);
    }
    
    
    for (let i = 0; i < wasmOutput.length; i++) {
        const wasmVal = wasmOutput[i];
        const rustVal = rustOutput[i];
        
        
        if (isNaN(wasmVal) && isNaN(rustVal)) {
            continue;
        }
        
        const diff = Math.abs(wasmVal - rustVal);
        const tol = tolerance * (1 + Math.abs(rustVal));
        
        if (diff > tol) {
            throw new Error(
                `${indicatorName} mismatch at index ${i}: ` +
                `WASM=${wasmVal}, Rust=${rustVal}, ` +
                `diff=${diff}, tol=${tol}`
            );
        }
    }
    
    return true;
}

/**
 * Helper function to compare two arrays
 */
function compareArrays(wasmArray, rustArray, name, tolerance) {
    if (wasmArray.length !== rustArray.length) {
        throw new Error(`Length mismatch for ${name}: WASM=${wasmArray.length}, Rust=${rustArray.length}`);
    }
    
    for (let i = 0; i < wasmArray.length; i++) {
        const wasmVal = wasmArray[i];
        const rustVal = rustArray[i];
        
        
        if (isNaN(wasmVal) && isNaN(rustVal)) {
            continue;
        }
        
        const diff = Math.abs(wasmVal - rustVal);
        const tol = tolerance * (1 + Math.abs(rustVal));
        
        if (diff > tol) {
            throw new Error(
                `${name} mismatch at index ${i}: ` +
                `WASM=${wasmVal}, Rust=${rustVal}, ` +
                `diff=${diff}, tol=${tol}`
            );
        }
    }
}

/**
 * Helper to compare arrays with tolerance
 * @param {Array|Float64Array} actual - Actual values
 * @param {Array} expected - Expected values
 * @param {number} tolerance - Comparison tolerance
 * @param {string} msg - Error message prefix
 */
export function assertArrayClose(actual, expected, tolerance = 1e-8, msg = '') {
    if (actual.length !== expected.length) {
        throw new Error(`${msg}: Length mismatch: ${actual.length} vs ${expected.length}`);
    }
    
    for (let i = 0; i < actual.length; i++) {
        const diff = Math.abs(actual[i] - expected[i]);
        if (diff > tolerance) {
            throw new Error(
                `${msg}: Mismatch at index ${i}: ` +
                `expected ${expected[i]}, got ${actual[i]} (diff: ${diff})`
            );
        }
    }
}