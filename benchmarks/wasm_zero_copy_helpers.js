/**
 * Zero-copy helper functions for WASM ALMA implementation
 * 
 * These helpers provide a convenient JavaScript API for the zero-copy
 * WASM functions, handling memory management and view creation.
 */

export class AlmaZeroCopy {
    constructor(wasm) {
        this.wasm = wasm;
        
        this.memory = wasm.__wasm ? wasm.__wasm.memory : wasm.memory;
    }

    /**
     * Create a Float64Array view into WASM memory
     * @param {number} ptr - Pointer to memory
     * @param {number} len - Length of array
     * @returns {Float64Array} View into WASM memory
     */
    createView(ptr, len) {
        return new Float64Array(this.memory.buffer, ptr, len);
    }

    /**
     * Allocate memory in WASM for input/output buffers
     * @param {number} len - Length of buffer
     * @returns {{ptr: number, view: Float64Array}} Pointer and view
     */
    allocBuffer(len) {
        const ptr = this.wasm.alma_alloc(len);
        const view = this.createView(ptr, len);
        return { ptr, view };
    }

    /**
     * Free allocated memory
     * @param {number} ptr - Pointer to memory
     * @param {number} len - Length of buffer
     */
    freeBuffer(ptr, len) {
        this.wasm.alma_free(ptr, len);
    }

    /**
     * Run ALMA with zero-copy using pre-allocated buffers
     * @param {Float64Array} inputData - Input data
     * @param {Object} params - ALMA parameters
     * @returns {Float64Array} Output array (view into WASM memory)
     */
    runWithPreallocated(inputData, params = {}) {
        const { period = 9, offset = 0.85, sigma = 6.0 } = params;
        const len = inputData.length;

        
        const input = this.allocBuffer(len);
        const output = this.allocBuffer(len);

        try {
            
            input.view.set(inputData);

            
            const result = this.wasm.alma_into(
                input.ptr,
                output.ptr,
                len,
                period,
                offset,
                sigma
            );

            if (result !== undefined) {
                throw new Error('ALMA computation failed');
            }

            
            return output.view;
        } catch (error) {
            
            this.freeBuffer(input.ptr, len);
            this.freeBuffer(output.ptr, len);
            throw error;
        }
    }

    /**
     * Run ALMA with zero-copy, returning a copy of the result
     * @param {Float64Array} inputData - Input data
     * @param {Object} params - ALMA parameters
     * @returns {Float64Array} Copy of output data
     */
    run(inputData, params = {}) {
        const { period = 9, offset = 0.85, sigma = 6.0 } = params;
        const len = inputData.length;

        
        const input = this.allocBuffer(len);
        const output = this.allocBuffer(len);

        try {
            
            input.view.set(inputData);

            
            const result = this.wasm.alma_into(
                input.ptr,
                output.ptr,
                len,
                period,
                offset,
                sigma
            );

            if (result !== undefined) {
                throw new Error('ALMA computation failed');
            }

            
            const currentOutputView = new Float64Array(
                this.wasm.__wasm.memory.buffer,
                output.ptr,
                len
            );
            
            
            const resultCopy = new Float64Array(currentOutputView);
            
            
            this.freeBuffer(input.ptr, len);
            this.freeBuffer(output.ptr, len);
            
            return resultCopy;
        } catch (error) {
            
            this.freeBuffer(input.ptr, len);
            this.freeBuffer(output.ptr, len);
            throw error;
        }
    }
}





/**
 * Benchmark helper that reuses buffers
 */
export class AlmaBenchmarkHelper {
    constructor(wasm, dataSize) {
        this.wasm = wasm;
        
        this.memory = wasm.__wasm ? wasm.__wasm.memory : wasm.memory;
        this.dataSize = dataSize;
        
        
        this.inputPtr = this.wasm.alma_alloc(dataSize);
        this.outputPtr = this.wasm.alma_alloc(dataSize);
        
        
        this.inputView = new Float64Array(this.memory.buffer, this.inputPtr, dataSize);
        this.outputView = new Float64Array(this.memory.buffer, this.outputPtr, dataSize);
    }

    /**
     * Run ALMA with minimal overhead
     * @param {Float64Array} data - Input data (must be same size as constructor)
     * @param {Object} params - ALMA parameters
     */
    run(data, params = {}) {
        const { period = 9, offset = 0.85, sigma = 6.0 } = params;
        
        
        this.inputView.set(data);
        
        
        const result = this.wasm.alma_into(
            this.inputPtr,
            this.outputPtr,
            this.dataSize,
            period,
            offset,
            sigma
        );
        
        if (result !== undefined) {
            throw new Error('ALMA computation failed');
        }
        
        
        return this.outputView;
    }

    /**
     * Clean up allocated memory
     */
    free() {
        this.wasm.alma_free(this.inputPtr, this.dataSize);
        this.wasm.alma_free(this.outputPtr, this.dataSize);
    }
}