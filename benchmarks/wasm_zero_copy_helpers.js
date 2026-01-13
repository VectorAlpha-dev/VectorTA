

export class AlmaZeroCopy {
    constructor(wasm) {
        this.wasm = wasm;

        this.memory = wasm.__wasm ? wasm.__wasm.memory : wasm.memory;
    }

    
    createView(ptr, len) {
        return new Float64Array(this.memory.buffer, ptr, len);
    }

    
    allocBuffer(len) {
        const ptr = this.wasm.alma_alloc(len);
        const view = this.createView(ptr, len);
        return { ptr, view };
    }

    
    freeBuffer(ptr, len) {
        this.wasm.alma_free(ptr, len);
    }

    
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

    
    free() {
        this.wasm.alma_free(this.inputPtr, this.dataSize);
        this.wasm.alma_free(this.outputPtr, this.dataSize);
    }
}