
let imports = {};
imports['__wbindgen_placeholder__'] = module.exports;
let wasm;
const { TextEncoder, TextDecoder } = require(`util`);

let WASM_VECTOR_LEN = 0;

let cachedUint8ArrayMemory0 = null;

function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

let cachedTextEncoder = new TextEncoder('utf-8');

const encodeString = (typeof cachedTextEncoder.encodeInto === 'function'
    ? function (arg, view) {
    return cachedTextEncoder.encodeInto(arg, view);
}
    : function (arg, view) {
    const buf = cachedTextEncoder.encode(arg);
    view.set(buf);
    return {
        read: arg.length,
        written: buf.length
    };
});

function passStringToWasm0(arg, malloc, realloc) {

    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }

    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = encodeString(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

let cachedDataViewMemory0 = null;

function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || (cachedDataViewMemory0.buffer.detached === undefined && cachedDataViewMemory0.buffer !== wasm.memory.buffer)) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

function addToExternrefTable0(obj) {
    const idx = wasm.__externref_table_alloc();
    wasm.__wbindgen_export_5.set(idx, obj);
    return idx;
}

function handleError(f, args) {
    try {
        return f.apply(this, args);
    } catch (e) {
        const idx = addToExternrefTable0(e);
        wasm.__wbindgen_exn_store(idx);
    }
}

function isLikeNone(x) {
    return x === undefined || x === null;
}

function debugString(val) {
    
    const type = typeof val;
    if (type == 'number' || type == 'boolean' || val == null) {
        return  `${val}`;
    }
    if (type == 'string') {
        return `"${val}"`;
    }
    if (type == 'symbol') {
        const description = val.description;
        if (description == null) {
            return 'Symbol';
        } else {
            return `Symbol(${description})`;
        }
    }
    if (type == 'function') {
        const name = val.name;
        if (typeof name == 'string' && name.length > 0) {
            return `Function(${name})`;
        } else {
            return 'Function';
        }
    }
    
    if (Array.isArray(val)) {
        const length = val.length;
        let debug = '[';
        if (length > 0) {
            debug += debugString(val[0]);
        }
        for(let i = 1; i < length; i++) {
            debug += ', ' + debugString(val[i]);
        }
        debug += ']';
        return debug;
    }
    
    const builtInMatches = /\[object ([^\]]+)\]/.exec(toString.call(val));
    let className;
    if (builtInMatches && builtInMatches.length > 1) {
        className = builtInMatches[1];
    } else {
        
        return toString.call(val);
    }
    if (className == 'Object') {
        
        
        
        try {
            return 'Object(' + JSON.stringify(val) + ')';
        } catch (_) {
            return 'Object';
        }
    }
    
    if (val instanceof Error) {
        return `${val.name}: ${val.message}\n${val.stack}`;
    }
    
    return className;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });

cachedTextDecoder.decode();

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

let cachedFloat64ArrayMemory0 = null;

function getFloat64ArrayMemory0() {
    if (cachedFloat64ArrayMemory0 === null || cachedFloat64ArrayMemory0.byteLength === 0) {
        cachedFloat64ArrayMemory0 = new Float64Array(wasm.memory.buffer);
    }
    return cachedFloat64ArrayMemory0;
}

function getArrayF64FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat64ArrayMemory0().subarray(ptr / 8, ptr / 8 + len);
}

function passArrayF64ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 8, 8) >>> 0;
    getFloat64ArrayMemory0().set(arg, ptr / 8);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function takeFromExternrefTable0(idx) {
    const value = wasm.__wbindgen_export_5.get(idx);
    wasm.__externref_table_dealloc(idx);
    return value;
}
/**
 * @param {Float64Array} close
 * @param {Float64Array} volume
 * @param {number} fast_period
 * @param {number} slow_period
 * @param {number} signal_period
 * @param {string} fast_ma_type
 * @param {string} slow_ma_type
 * @param {string} signal_ma_type
 * @returns {Float64Array}
 */
module.exports.vwmacd_js = function(close, volume, fast_period, slow_period, signal_period, fast_ma_type, slow_ma_type, signal_ma_type) {
    const ptr0 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(volume, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passStringToWasm0(fast_ma_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passStringToWasm0(slow_ma_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len3 = WASM_VECTOR_LEN;
    const ptr4 = passStringToWasm0(signal_ma_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len4 = WASM_VECTOR_LEN;
    const ret = wasm.vwmacd_js(ptr0, len0, ptr1, len1, fast_period, slow_period, signal_period, ptr2, len2, ptr3, len3, ptr4, len4);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v6 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v6;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.vwmacd_alloc = function(len) {
    const ret = wasm.vwmacd_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.vwmacd_free = function(ptr, len) {
    wasm.vwmacd_free(ptr, len);
};

/**
 * @param {number} close_ptr
 * @param {number} volume_ptr
 * @param {number} macd_ptr
 * @param {number} signal_ptr
 * @param {number} hist_ptr
 * @param {number} len
 * @param {number} fast_period
 * @param {number} slow_period
 * @param {number} signal_period
 * @param {string} fast_ma_type
 * @param {string} slow_ma_type
 * @param {string} signal_ma_type
 */
module.exports.vwmacd_into = function(close_ptr, volume_ptr, macd_ptr, signal_ptr, hist_ptr, len, fast_period, slow_period, signal_period, fast_ma_type, slow_ma_type, signal_ma_type) {
    const ptr0 = passStringToWasm0(fast_ma_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passStringToWasm0(slow_ma_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passStringToWasm0(signal_ma_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.vwmacd_into(close_ptr, volume_ptr, macd_ptr, signal_ptr, hist_ptr, len, fast_period, slow_period, signal_period, ptr0, len0, ptr1, len1, ptr2, len2);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} close_ptr
 * @param {number} volume_ptr
 * @param {number} macd_out_ptr
 * @param {number} signal_out_ptr
 * @param {number} hist_out_ptr
 * @param {number} len
 * @param {number} fast_start
 * @param {number} fast_end
 * @param {number} fast_step
 * @param {number} slow_start
 * @param {number} slow_end
 * @param {number} slow_step
 * @param {number} signal_start
 * @param {number} signal_end
 * @param {number} signal_step
 * @param {string} fast_ma_type
 * @param {string} slow_ma_type
 * @param {string} signal_ma_type
 * @returns {number}
 */
module.exports.vwmacd_batch_into = function(close_ptr, volume_ptr, macd_out_ptr, signal_out_ptr, hist_out_ptr, len, fast_start, fast_end, fast_step, slow_start, slow_end, slow_step, signal_start, signal_end, signal_step, fast_ma_type, slow_ma_type, signal_ma_type) {
    const ptr0 = passStringToWasm0(fast_ma_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passStringToWasm0(slow_ma_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passStringToWasm0(signal_ma_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.vwmacd_batch_into(close_ptr, volume_ptr, macd_out_ptr, signal_out_ptr, hist_out_ptr, len, fast_start, fast_end, fast_step, slow_start, slow_end, slow_step, signal_start, signal_end, signal_step, ptr0, len0, ptr1, len1, ptr2, len2);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @param {number} devup
 * @param {number} devdn
 * @param {string} matype
 * @param {number} devtype
 * @returns {Float64Array}
 */
module.exports.bollinger_bands_js = function(data, period, devup, devdn, matype, devtype) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passStringToWasm0(matype, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.bollinger_bands_js(ptr0, len0, period, devup, devdn, ptr1, len1, devtype);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
};

/**
 * @param {Float64Array} data
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @param {number} devup_start
 * @param {number} devup_end
 * @param {number} devup_step
 * @param {number} devdn_start
 * @param {number} devdn_end
 * @param {number} devdn_step
 * @param {string} matype
 * @param {number} devtype
 * @returns {Float64Array}
 */
module.exports.bollinger_bands_batch_js = function(data, period_start, period_end, period_step, devup_start, devup_end, devup_step, devdn_start, devdn_end, devdn_step, matype, devtype) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passStringToWasm0(matype, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.bollinger_bands_batch_js(ptr0, len0, period_start, period_end, period_step, devup_start, devup_end, devup_step, devdn_start, devdn_end, devdn_step, ptr1, len1, devtype);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
};

/**
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @param {number} devup_start
 * @param {number} devup_end
 * @param {number} devup_step
 * @param {number} devdn_start
 * @param {number} devdn_end
 * @param {number} devdn_step
 * @param {string} matype
 * @param {number} devtype
 * @returns {Float64Array}
 */
module.exports.bollinger_bands_batch_metadata_js = function(period_start, period_end, period_step, devup_start, devup_end, devup_step, devdn_start, devdn_end, devdn_step, matype, devtype) {
    const ptr0 = passStringToWasm0(matype, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.bollinger_bands_batch_metadata_js(period_start, period_end, period_step, devup_start, devup_end, devup_step, devdn_start, devdn_end, devdn_step, ptr0, len0, devtype);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.bollinger_bands_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.bollinger_bands_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.bollinger_bands_alloc = function(len) {
    const ret = wasm.bollinger_bands_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.bollinger_bands_free = function(ptr, len) {
    wasm.bollinger_bands_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} upper_ptr
 * @param {number} middle_ptr
 * @param {number} lower_ptr
 * @param {number} len
 * @param {number} period
 * @param {number} devup
 * @param {number} devdn
 * @param {string} matype
 * @param {number} devtype
 */
module.exports.bollinger_bands_into = function(in_ptr, upper_ptr, middle_ptr, lower_ptr, len, period, devup, devdn, matype, devtype) {
    const ptr0 = passStringToWasm0(matype, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.bollinger_bands_into(in_ptr, upper_ptr, middle_ptr, lower_ptr, len, period, devup, devdn, ptr0, len0, devtype);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} in_ptr
 * @param {number} upper_ptr
 * @param {number} middle_ptr
 * @param {number} lower_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @param {number} devup_start
 * @param {number} devup_end
 * @param {number} devup_step
 * @param {number} devdn_start
 * @param {number} devdn_end
 * @param {number} devdn_step
 * @param {string} matype
 * @param {number} devtype
 * @returns {number}
 */
module.exports.bollinger_bands_batch_into = function(in_ptr, upper_ptr, middle_ptr, lower_ptr, len, period_start, period_end, period_step, devup_start, devup_end, devup_step, devdn_start, devdn_end, devdn_step, matype, devtype) {
    const ptr0 = passStringToWasm0(matype, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.bollinger_bands_batch_into(in_ptr, upper_ptr, middle_ptr, lower_ptr, len, period_start, period_end, period_step, devup_start, devup_end, devup_step, devdn_start, devdn_end, devdn_step, ptr0, len0, devtype);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} close
 * @param {number} fast_k_period
 * @param {number} slow_k_period
 * @param {string} slow_k_ma_type
 * @param {number} slow_d_period
 * @param {string} slow_d_ma_type
 * @returns {any}
 */
module.exports.kdj_js = function(high, low, close, fast_k_period, slow_k_period, slow_k_ma_type, slow_d_period, slow_d_ma_type) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passStringToWasm0(slow_k_ma_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len3 = WASM_VECTOR_LEN;
    const ptr4 = passStringToWasm0(slow_d_ma_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len4 = WASM_VECTOR_LEN;
    const ret = wasm.kdj_js(ptr0, len0, ptr1, len1, ptr2, len2, fast_k_period, slow_k_period, ptr3, len3, slow_d_period, ptr4, len4);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.kdj_alloc = function(len) {
    const ret = wasm.kdj_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.kdj_free = function(ptr, len) {
    wasm.kdj_free(ptr, len);
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} close_ptr
 * @param {number} k_ptr
 * @param {number} d_ptr
 * @param {number} j_ptr
 * @param {number} len
 * @param {number} fast_k_period
 * @param {number} slow_k_period
 * @param {string} slow_k_ma_type
 * @param {number} slow_d_period
 * @param {string} slow_d_ma_type
 */
module.exports.kdj_into = function(high_ptr, low_ptr, close_ptr, k_ptr, d_ptr, j_ptr, len, fast_k_period, slow_k_period, slow_k_ma_type, slow_d_period, slow_d_ma_type) {
    const ptr0 = passStringToWasm0(slow_k_ma_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passStringToWasm0(slow_d_ma_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.kdj_into(high_ptr, low_ptr, close_ptr, k_ptr, d_ptr, j_ptr, len, fast_k_period, slow_k_period, ptr0, len0, slow_d_period, ptr1, len1);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} close
 * @param {any} config
 * @returns {any}
 */
module.exports.kdj_batch = function(high, low, close, config) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.kdj_batch(ptr0, len0, ptr1, len1, ptr2, len2, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} close_ptr
 * @param {number} k_out_ptr
 * @param {number} d_out_ptr
 * @param {number} j_out_ptr
 * @param {number} len
 * @param {number} fast_k_start
 * @param {number} fast_k_end
 * @param {number} fast_k_step
 * @param {number} slow_k_start
 * @param {number} slow_k_end
 * @param {number} slow_k_step
 * @param {string} slow_k_ma_type
 * @param {number} slow_d_start
 * @param {number} slow_d_end
 * @param {number} slow_d_step
 * @param {string} slow_d_ma_type
 * @returns {number}
 */
module.exports.kdj_batch_into = function(high_ptr, low_ptr, close_ptr, k_out_ptr, d_out_ptr, j_out_ptr, len, fast_k_start, fast_k_end, fast_k_step, slow_k_start, slow_k_end, slow_k_step, slow_k_ma_type, slow_d_start, slow_d_end, slow_d_step, slow_d_ma_type) {
    const ptr0 = passStringToWasm0(slow_k_ma_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passStringToWasm0(slow_d_ma_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.kdj_batch_into(high_ptr, low_ptr, close_ptr, k_out_ptr, d_out_ptr, j_out_ptr, len, fast_k_start, fast_k_end, fast_k_step, slow_k_start, slow_k_end, slow_k_step, ptr0, len0, slow_d_start, slow_d_end, slow_d_step, ptr1, len1);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * Safe API that allocates a new vector and returns it.
 * `high` and `low` are JavaScript Float64Arrays
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {number} period
 * @param {number} mult
 * @param {string} direction
 * @param {string} ma_type
 * @returns {Float64Array}
 */
module.exports.kaufmanstop_js = function(high, low, period, mult, direction, ma_type) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passStringToWasm0(direction, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passStringToWasm0(ma_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len3 = WASM_VECTOR_LEN;
    const ret = wasm.kaufmanstop_js(ptr0, len0, ptr1, len1, period, mult, ptr2, len2, ptr3, len3);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v5 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v5;
};

/**
 * Fast API that writes directly to pre-allocated memory.
 * Performs aliasing checks between input and output pointers.
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 * @param {number} mult
 * @param {string} direction
 * @param {string} ma_type
 */
module.exports.kaufmanstop_into = function(high_ptr, low_ptr, out_ptr, len, period, mult, direction, ma_type) {
    const ptr0 = passStringToWasm0(direction, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passStringToWasm0(ma_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.kaufmanstop_into(high_ptr, low_ptr, out_ptr, len, period, mult, ptr0, len0, ptr1, len1);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * Allocates memory for a f64 array of the given length.
 * Returns a pointer that must be freed with kaufmanstop_free.
 * @param {number} len
 * @returns {number}
 */
module.exports.kaufmanstop_alloc = function(len) {
    const ret = wasm.kaufmanstop_alloc(len);
    return ret >>> 0;
};

/**
 * Frees memory allocated by kaufmanstop_alloc.
 * SAFETY: ptr must have been allocated by kaufmanstop_alloc with the same length.
 * @param {number} ptr
 * @param {number} len
 */
module.exports.kaufmanstop_free = function(ptr, len) {
    wasm.kaufmanstop_free(ptr, len);
};

/**
 * Batch computation with parameter sweep.
 * Returns JavaScript object with { values: Float64Array, combos: Array, rows: number, cols: number }
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @param {number} mult_start
 * @param {number} mult_end
 * @param {number} mult_step
 * @param {string} direction
 * @param {string} ma_type
 * @returns {any}
 */
module.exports.kaufmanstop_batch_js = function(high, low, period_start, period_end, period_step, mult_start, mult_end, mult_step, direction, ma_type) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passStringToWasm0(direction, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passStringToWasm0(ma_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len3 = WASM_VECTOR_LEN;
    const ret = wasm.kaufmanstop_batch_js(ptr0, len0, ptr1, len1, period_start, period_end, period_step, mult_start, mult_end, mult_step, ptr2, len2, ptr3, len3);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * Unified batch API that accepts a config object with ranges.
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {any} config
 * @returns {any}
 */
module.exports.kaufmanstop_batch_unified_js = function(high, low, config) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.kaufmanstop_batch_unified_js(ptr0, len0, ptr1, len1, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * Fast batch API that writes to pre-allocated memory.
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @param {number} mult_start
 * @param {number} mult_end
 * @param {number} mult_step
 * @param {string} direction
 * @param {string} ma_type
 * @returns {any}
 */
module.exports.kaufmanstop_batch_into = function(high_ptr, low_ptr, out_ptr, len, period_start, period_end, period_step, mult_start, mult_end, mult_step, direction, ma_type) {
    const ptr0 = passStringToWasm0(direction, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passStringToWasm0(ma_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.kaufmanstop_batch_into(high_ptr, low_ptr, out_ptr, len, period_start, period_end, period_step, mult_start, mult_end, mult_step, ptr0, len0, ptr1, len1);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {Float64Array} data
 * @param {number} min_period
 * @param {number} max_period
 * @param {string} matype
 * @param {number} devtype
 * @returns {Float64Array}
 */
module.exports.vlma_js = function(data, min_period, max_period, matype, devtype) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passStringToWasm0(matype, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.vlma_js(ptr0, len0, min_period, max_period, ptr1, len1, devtype);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} min_period
 * @param {number} max_period
 * @param {string} matype
 * @param {number} devtype
 */
module.exports.vlma_into = function(in_ptr, out_ptr, len, min_period, max_period, matype, devtype) {
    const ptr0 = passStringToWasm0(matype, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.vlma_into(in_ptr, out_ptr, len, min_period, max_period, ptr0, len0, devtype);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.vlma_alloc = function(len) {
    const ret = wasm.vlma_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.vlma_free = function(ptr, len) {
    wasm.vlma_free(ptr, len);
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.vlma_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.vlma_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} min_period_start
 * @param {number} min_period_end
 * @param {number} min_period_step
 * @param {number} max_period_start
 * @param {number} max_period_end
 * @param {number} max_period_step
 * @param {number} devtype_start
 * @param {number} devtype_end
 * @param {number} devtype_step
 * @param {string} matype
 * @returns {number}
 */
module.exports.vlma_batch_into = function(in_ptr, out_ptr, len, min_period_start, min_period_end, min_period_step, max_period_start, max_period_end, max_period_step, devtype_start, devtype_end, devtype_step, matype) {
    const ptr0 = passStringToWasm0(matype, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.vlma_batch_into(in_ptr, out_ptr, len, min_period_start, min_period_end, min_period_step, max_period_start, max_period_end, max_period_step, devtype_start, devtype_end, devtype_step, ptr0, len0);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @param {string} ma_type
 * @param {number} nbdev
 * @param {number} devtype
 * @returns {Float64Array}
 */
module.exports.zscore_js = function(data, period, ma_type, nbdev, devtype) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passStringToWasm0(ma_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.zscore_js(ptr0, len0, period, ptr1, len1, nbdev, devtype);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.zscore_alloc = function(len) {
    const ret = wasm.zscore_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.zscore_free = function(ptr, len) {
    wasm.zscore_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 * @param {string} ma_type
 * @param {number} nbdev
 * @param {number} devtype
 */
module.exports.zscore_into = function(in_ptr, out_ptr, len, period, ma_type, nbdev, devtype) {
    const ptr0 = passStringToWasm0(ma_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.zscore_into(in_ptr, out_ptr, len, period, ptr0, len0, nbdev, devtype);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.zscore_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.zscore_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @param {string} ma_type
 * @param {number} nbdev_start
 * @param {number} nbdev_end
 * @param {number} nbdev_step
 * @param {number} devtype_start
 * @param {number} devtype_end
 * @param {number} devtype_step
 * @returns {number}
 */
module.exports.zscore_batch_into = function(in_ptr, out_ptr, len, period_start, period_end, period_step, ma_type, nbdev_start, nbdev_end, nbdev_step, devtype_start, devtype_end, devtype_step) {
    const ptr0 = passStringToWasm0(ma_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.zscore_batch_into(in_ptr, out_ptr, len, period_start, period_end, period_step, ptr0, len0, nbdev_start, nbdev_end, nbdev_step, devtype_start, devtype_end, devtype_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @param {number} devup
 * @param {number} devdn
 * @param {string | null} [matype]
 * @param {number | null} [devtype]
 * @returns {Float64Array}
 */
module.exports.bollinger_bands_width_js = function(data, period, devup, devdn, matype, devtype) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    var ptr1 = isLikeNone(matype) ? 0 : passStringToWasm0(matype, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    var len1 = WASM_VECTOR_LEN;
    const ret = wasm.bollinger_bands_width_js(ptr0, len0, period, devup, devdn, ptr1, len1, isLikeNone(devtype) ? 0x100000001 : (devtype) >>> 0);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
};

/**
 * @param {Float64Array} data
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @param {number} devup_start
 * @param {number} devup_end
 * @param {number} devup_step
 * @param {number} devdn_start
 * @param {number} devdn_end
 * @param {number} devdn_step
 * @returns {Float64Array}
 */
module.exports.bollinger_bands_width_batch_js = function(data, period_start, period_end, period_step, devup_start, devup_end, devup_step, devdn_start, devdn_end, devdn_step) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.bollinger_bands_width_batch_js(ptr0, len0, period_start, period_end, period_step, devup_start, devup_end, devup_step, devdn_start, devdn_end, devdn_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @param {number} devup_start
 * @param {number} devup_end
 * @param {number} devup_step
 * @param {number} devdn_start
 * @param {number} devdn_end
 * @param {number} devdn_step
 * @returns {Float64Array}
 */
module.exports.bollinger_bands_width_batch_metadata_js = function(period_start, period_end, period_step, devup_start, devup_end, devup_step, devdn_start, devdn_end, devdn_step) {
    const ret = wasm.bollinger_bands_width_batch_metadata_js(period_start, period_end, period_step, devup_start, devup_end, devup_step, devdn_start, devdn_end, devdn_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.bollinger_bands_width_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.bollinger_bands_width_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 * @param {number} devup
 * @param {number} devdn
 * @param {string | null} [matype]
 * @param {number | null} [devtype]
 */
module.exports.bollinger_bands_width_into = function(in_ptr, out_ptr, len, period, devup, devdn, matype, devtype) {
    var ptr0 = isLikeNone(matype) ? 0 : passStringToWasm0(matype, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    var len0 = WASM_VECTOR_LEN;
    const ret = wasm.bollinger_bands_width_into(in_ptr, out_ptr, len, period, devup, devdn, ptr0, len0, isLikeNone(devtype) ? 0x100000001 : (devtype) >>> 0);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.bollinger_bands_width_alloc = function(len) {
    const ret = wasm.bollinger_bands_width_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.bollinger_bands_width_free = function(ptr, len) {
    wasm.bollinger_bands_width_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @param {number} devup_start
 * @param {number} devup_end
 * @param {number} devup_step
 * @param {number} devdn_start
 * @param {number} devdn_end
 * @param {number} devdn_step
 * @returns {number}
 */
module.exports.bollinger_bands_width_batch_into = function(in_ptr, out_ptr, len, period_start, period_end, period_step, devup_start, devup_end, devup_step, devdn_start, devdn_end, devdn_step) {
    const ret = wasm.bollinger_bands_width_batch_into(in_ptr, out_ptr, len, period_start, period_end, period_step, devup_start, devup_end, devup_step, devdn_start, devdn_end, devdn_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} main
 * @param {Float64Array} compare
 * @param {number} lookback
 * @param {number} period
 * @param {number} signal_period
 * @param {string | null} [matype]
 * @param {string | null} [signal_matype]
 * @returns {Float64Array}
 */
module.exports.rsmk_js = function(main, compare, lookback, period, signal_period, matype, signal_matype) {
    const ptr0 = passArrayF64ToWasm0(main, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(compare, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    var ptr2 = isLikeNone(matype) ? 0 : passStringToWasm0(matype, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    var len2 = WASM_VECTOR_LEN;
    var ptr3 = isLikeNone(signal_matype) ? 0 : passStringToWasm0(signal_matype, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    var len3 = WASM_VECTOR_LEN;
    const ret = wasm.rsmk_js(ptr0, len0, ptr1, len1, lookback, period, signal_period, ptr2, len2, ptr3, len3);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v5 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v5;
};

/**
 * @param {number} in_ptr
 * @param {number} indicator_ptr
 * @param {number} signal_ptr
 * @param {number} len
 * @param {number} compare_ptr
 * @param {number} lookback
 * @param {number} period
 * @param {number} signal_period
 * @param {string | null} [matype]
 * @param {string | null} [signal_matype]
 */
module.exports.rsmk_into = function(in_ptr, indicator_ptr, signal_ptr, len, compare_ptr, lookback, period, signal_period, matype, signal_matype) {
    var ptr0 = isLikeNone(matype) ? 0 : passStringToWasm0(matype, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    var len0 = WASM_VECTOR_LEN;
    var ptr1 = isLikeNone(signal_matype) ? 0 : passStringToWasm0(signal_matype, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    var len1 = WASM_VECTOR_LEN;
    const ret = wasm.rsmk_into(in_ptr, indicator_ptr, signal_ptr, len, compare_ptr, lookback, period, signal_period, ptr0, len0, ptr1, len1);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.rsmk_alloc = function(len) {
    const ret = wasm.rsmk_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.rsmk_free = function(ptr, len) {
    wasm.rsmk_free(ptr, len);
};

/**
 * @param {Float64Array} main
 * @param {Float64Array} compare
 * @param {any} config
 * @returns {any}
 */
module.exports.rsmk_batch = function(main, compare, config) {
    const ptr0 = passArrayF64ToWasm0(main, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(compare, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.rsmk_batch(ptr0, len0, ptr1, len1, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} close
 * @param {Float64Array} source
 * @param {number} period
 * @param {number} multiplier
 * @param {string} ma_type
 * @returns {any}
 */
module.exports.keltner = function(high, low, close, source, period, multiplier, ma_type) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArrayF64ToWasm0(source, wasm.__wbindgen_malloc);
    const len3 = WASM_VECTOR_LEN;
    const ptr4 = passStringToWasm0(ma_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len4 = WASM_VECTOR_LEN;
    const ret = wasm.keltner(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, period, multiplier, ptr4, len4);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * Fast API with aliasing detection - separate pointers for each output
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} close_ptr
 * @param {number} source_ptr
 * @param {number} upper_ptr
 * @param {number} middle_ptr
 * @param {number} lower_ptr
 * @param {number} len
 * @param {number} period
 * @param {number} multiplier
 * @param {string} ma_type
 */
module.exports.keltner_into = function(high_ptr, low_ptr, close_ptr, source_ptr, upper_ptr, middle_ptr, lower_ptr, len, period, multiplier, ma_type) {
    const ptr0 = passStringToWasm0(ma_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.keltner_into(high_ptr, low_ptr, close_ptr, source_ptr, upper_ptr, middle_ptr, lower_ptr, len, period, multiplier, ptr0, len0);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * Memory allocation for WASM
 * @param {number} len
 * @returns {number}
 */
module.exports.keltner_alloc = function(len) {
    const ret = wasm.keltner_alloc(len);
    return ret >>> 0;
};

/**
 * Memory deallocation for WASM
 * @param {number} ptr
 * @param {number} len
 */
module.exports.keltner_free = function(ptr, len) {
    wasm.keltner_free(ptr, len);
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} close
 * @param {Float64Array} source
 * @param {any} config
 * @returns {any}
 */
module.exports.keltner_batch = function(high, low, close, source, config) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArrayF64ToWasm0(source, wasm.__wbindgen_malloc);
    const len3 = WASM_VECTOR_LEN;
    const ret = wasm.keltner_batch(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} h_ptr
 * @param {number} l_ptr
 * @param {number} c_ptr
 * @param {number} s_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 * @param {number} multiplier
 * @param {string} ma_type
 */
module.exports.keltner_into_concat = function(h_ptr, l_ptr, c_ptr, s_ptr, out_ptr, len, period, multiplier, ma_type) {
    const ptr0 = passStringToWasm0(ma_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.keltner_into_concat(h_ptr, l_ptr, c_ptr, s_ptr, out_ptr, len, period, multiplier, ptr0, len0);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} data
 * @param {number} fast_period
 * @param {number} slow_period
 * @param {number} k_period
 * @param {number} d_period
 * @param {string} fast_ma_type
 * @param {string} slow_ma_type
 * @returns {Float64Array}
 */
module.exports.stc_js = function(data, fast_period, slow_period, k_period, d_period, fast_ma_type, slow_ma_type) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passStringToWasm0(fast_ma_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passStringToWasm0(slow_ma_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.stc_js(ptr0, len0, fast_period, slow_period, k_period, d_period, ptr1, len1, ptr2, len2);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v4 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v4;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.stc_alloc = function(len) {
    const ret = wasm.stc_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.stc_free = function(ptr, len) {
    wasm.stc_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} fast_period
 * @param {number} slow_period
 * @param {number} k_period
 * @param {number} d_period
 * @param {string} fast_ma_type
 * @param {string} slow_ma_type
 */
module.exports.stc_into = function(in_ptr, out_ptr, len, fast_period, slow_period, k_period, d_period, fast_ma_type, slow_ma_type) {
    const ptr0 = passStringToWasm0(fast_ma_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passStringToWasm0(slow_ma_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.stc_into(in_ptr, out_ptr, len, fast_period, slow_period, k_period, d_period, ptr0, len0, ptr1, len1);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.stc_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.stc_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {Float64Array} data
 * @param {number} fast_period
 * @param {number} slow_period
 * @param {number} signal_period
 * @param {string} ma_type
 * @returns {MacdResult}
 */
module.exports.macd_js = function(data, fast_period, slow_period, signal_period, ma_type) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passStringToWasm0(ma_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.macd_js(ptr0, len0, fast_period, slow_period, signal_period, ptr1, len1);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return MacdResult.__wrap(ret[0]);
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.macd_alloc = function(len) {
    const ret = wasm.macd_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.macd_free = function(ptr, len) {
    wasm.macd_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} macd_ptr
 * @param {number} signal_ptr
 * @param {number} hist_ptr
 * @param {number} len
 * @param {number} fast_period
 * @param {number} slow_period
 * @param {number} signal_period
 * @param {string} ma_type
 */
module.exports.macd_into = function(in_ptr, macd_ptr, signal_ptr, hist_ptr, len, fast_period, slow_period, signal_period, ma_type) {
    const ptr0 = passStringToWasm0(ma_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.macd_into(in_ptr, macd_ptr, signal_ptr, hist_ptr, len, fast_period, slow_period, signal_period, ptr0, len0);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.macd_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.macd_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {Float64Array} data
 * @param {number} fast_period
 * @param {number} slow_period
 * @param {string} ma_type
 * @returns {Float64Array}
 */
module.exports.ppo_js = function(data, fast_period, slow_period, ma_type) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passStringToWasm0(ma_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.ppo_js(ptr0, len0, fast_period, slow_period, ptr1, len1);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.ppo_alloc = function(len) {
    const ret = wasm.ppo_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.ppo_free = function(ptr, len) {
    wasm.ppo_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} fast_period
 * @param {number} slow_period
 * @param {string} ma_type
 */
module.exports.ppo_into = function(in_ptr, out_ptr, len, fast_period, slow_period, ma_type) {
    const ptr0 = passStringToWasm0(ma_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.ppo_into(in_ptr, out_ptr, len, fast_period, slow_period, ptr0, len0);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.ppo_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.ppo_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} fast_period_start
 * @param {number} fast_period_end
 * @param {number} fast_period_step
 * @param {number} slow_period_start
 * @param {number} slow_period_end
 * @param {number} slow_period_step
 * @param {string} ma_type
 * @returns {number}
 */
module.exports.ppo_batch_into = function(in_ptr, out_ptr, len, fast_period_start, fast_period_end, fast_period_step, slow_period_start, slow_period_end, slow_period_step, ma_type) {
    const ptr0 = passStringToWasm0(ma_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.ppo_batch_into(in_ptr, out_ptr, len, fast_period_start, fast_period_end, fast_period_step, slow_period_start, slow_period_end, slow_period_step, ptr0, len0);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} short_period
 * @param {number} long_period
 * @param {number} ma_period
 * @param {string | null} [ma_type]
 * @returns {Float64Array}
 */
module.exports.coppock_js = function(data, short_period, long_period, ma_period, ma_type) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    var ptr1 = isLikeNone(ma_type) ? 0 : passStringToWasm0(ma_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    var len1 = WASM_VECTOR_LEN;
    const ret = wasm.coppock_js(ptr0, len0, short_period, long_period, ma_period, ptr1, len1);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.coppock_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.coppock_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.coppock_alloc = function(len) {
    const ret = wasm.coppock_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.coppock_free = function(ptr, len) {
    wasm.coppock_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} short_period
 * @param {number} long_period
 * @param {number} ma_period
 * @param {string | null} [ma_type]
 */
module.exports.coppock_into = function(in_ptr, out_ptr, len, short_period, long_period, ma_period, ma_type) {
    var ptr0 = isLikeNone(ma_type) ? 0 : passStringToWasm0(ma_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    var len0 = WASM_VECTOR_LEN;
    const ret = wasm.coppock_into(in_ptr, out_ptr, len, short_period, long_period, ma_period, ptr0, len0);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} source
 * @param {number} period
 * @param {string} ma_type
 * @returns {Float64Array}
 */
module.exports.eri_js = function(high, low, source, period, ma_type) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(source, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passStringToWasm0(ma_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len3 = WASM_VECTOR_LEN;
    const ret = wasm.eri_js(ptr0, len0, ptr1, len1, ptr2, len2, period, ptr3, len3);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v5 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v5;
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} source_ptr
 * @param {number} bull_ptr
 * @param {number} bear_ptr
 * @param {number} len
 * @param {number} period
 * @param {string} ma_type
 */
module.exports.eri_into = function(high_ptr, low_ptr, source_ptr, bull_ptr, bear_ptr, len, period, ma_type) {
    const ptr0 = passStringToWasm0(ma_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.eri_into(high_ptr, low_ptr, source_ptr, bull_ptr, bear_ptr, len, period, ptr0, len0);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.eri_alloc = function(len) {
    const ret = wasm.eri_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.eri_free = function(ptr, len) {
    wasm.eri_free(ptr, len);
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} source
 * @param {any} config
 * @returns {any}
 */
module.exports.eri_batch = function(high, low, source, config) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(source, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.eri_batch(ptr0, len0, ptr1, len1, ptr2, len2, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {Float64Array} close
 * @param {Float64Array} volume
 * @param {number} short_range
 * @param {number} long_range
 * @returns {Float64Array}
 */
module.exports.vpci_js = function(close, volume, short_range, long_range) {
    const ptr0 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(volume, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.vpci_js(ptr0, len0, ptr1, len1, short_range, long_range);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
};

/**
 * @param {number} close_ptr
 * @param {number} volume_ptr
 * @param {number} vpci_ptr
 * @param {number} vpcis_ptr
 * @param {number} len
 * @param {number} short_range
 * @param {number} long_range
 */
module.exports.vpci_into = function(close_ptr, volume_ptr, vpci_ptr, vpcis_ptr, len, short_range, long_range) {
    const ret = wasm.vpci_into(close_ptr, volume_ptr, vpci_ptr, vpcis_ptr, len, short_range, long_range);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.vpci_alloc = function(len) {
    const ret = wasm.vpci_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.vpci_free = function(ptr, len) {
    wasm.vpci_free(ptr, len);
};

/**
 * @param {Float64Array} close
 * @param {Float64Array} volume
 * @param {any} config
 * @returns {any}
 */
module.exports.vpci_batch = function(close, volume, config) {
    const ptr0 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(volume, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.vpci_batch(ptr0, len0, ptr1, len1, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} close_ptr
 * @param {number} volume_ptr
 * @param {number} vpci_ptr
 * @param {number} vpcis_ptr
 * @param {number} len
 * @param {number} short_start
 * @param {number} short_end
 * @param {number} short_step
 * @param {number} long_start
 * @param {number} long_end
 * @param {number} long_step
 * @returns {number}
 */
module.exports.vpci_batch_into = function(close_ptr, volume_ptr, vpci_ptr, vpcis_ptr, len, short_start, short_end, short_step, long_start, long_end, long_step) {
    const ret = wasm.vpci_batch_into(close_ptr, volume_ptr, vpci_ptr, vpcis_ptr, len, short_start, short_end, short_step, long_start, long_end, long_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} jaws_length
 * @param {number} jaws_shift
 * @param {number} teeth_length
 * @param {number} teeth_shift
 * @param {number} lips_length
 * @param {number} lips_shift
 * @returns {any}
 */
module.exports.gatorosc_js = function(data, jaws_length, jaws_shift, teeth_length, teeth_shift, lips_length, lips_shift) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.gatorosc_js(ptr0, len0, jaws_length, jaws_shift, teeth_length, teeth_shift, lips_length, lips_shift);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.gatorosc_alloc = function(len) {
    const ret = wasm.gatorosc_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.gatorosc_free = function(ptr, len) {
    wasm.gatorosc_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} upper_ptr
 * @param {number} lower_ptr
 * @param {number} upper_change_ptr
 * @param {number} lower_change_ptr
 * @param {number} len
 * @param {number} jaws_length
 * @param {number} jaws_shift
 * @param {number} teeth_length
 * @param {number} teeth_shift
 * @param {number} lips_length
 * @param {number} lips_shift
 */
module.exports.gatorosc_into = function(in_ptr, upper_ptr, lower_ptr, upper_change_ptr, lower_change_ptr, len, jaws_length, jaws_shift, teeth_length, teeth_shift, lips_length, lips_shift) {
    const ret = wasm.gatorosc_into(in_ptr, upper_ptr, lower_ptr, upper_change_ptr, lower_change_ptr, len, jaws_length, jaws_shift, teeth_length, teeth_shift, lips_length, lips_shift);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.gatorosc_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.gatorosc_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} in_ptr
 * @param {number} upper_ptr
 * @param {number} lower_ptr
 * @param {number} upper_change_ptr
 * @param {number} lower_change_ptr
 * @param {number} len
 * @param {number} jaws_length_start
 * @param {number} jaws_length_end
 * @param {number} jaws_length_step
 * @param {number} jaws_shift_start
 * @param {number} jaws_shift_end
 * @param {number} jaws_shift_step
 * @param {number} teeth_length_start
 * @param {number} teeth_length_end
 * @param {number} teeth_length_step
 * @param {number} teeth_shift_start
 * @param {number} teeth_shift_end
 * @param {number} teeth_shift_step
 * @param {number} lips_length_start
 * @param {number} lips_length_end
 * @param {number} lips_length_step
 * @param {number} lips_shift_start
 * @param {number} lips_shift_end
 * @param {number} lips_shift_step
 * @returns {number}
 */
module.exports.gatorosc_batch_into = function(in_ptr, upper_ptr, lower_ptr, upper_change_ptr, lower_change_ptr, len, jaws_length_start, jaws_length_end, jaws_length_step, jaws_shift_start, jaws_shift_end, jaws_shift_step, teeth_length_start, teeth_length_end, teeth_length_step, teeth_shift_start, teeth_shift_end, teeth_shift_step, lips_length_start, lips_length_end, lips_length_step, lips_shift_start, lips_shift_end, lips_shift_step) {
    const ret = wasm.gatorosc_batch_into(in_ptr, upper_ptr, lower_ptr, upper_change_ptr, lower_change_ptr, len, jaws_length_start, jaws_length_end, jaws_length_step, jaws_shift_start, jaws_shift_end, jaws_shift_step, teeth_length_start, teeth_length_end, teeth_length_step, teeth_shift_start, teeth_shift_end, teeth_shift_step, lips_length_start, lips_length_end, lips_length_step, lips_shift_start, lips_shift_end, lips_shift_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * WASM interface for KST calculation (safe API)
 * Returns an object with line and signal arrays
 * @param {Float64Array} data
 * @param {number} sma_period1
 * @param {number} sma_period2
 * @param {number} sma_period3
 * @param {number} sma_period4
 * @param {number} roc_period1
 * @param {number} roc_period2
 * @param {number} roc_period3
 * @param {number} roc_period4
 * @param {number} signal_period
 * @returns {any}
 */
module.exports.kst_js = function(data, sma_period1, sma_period2, sma_period3, sma_period4, roc_period1, roc_period2, roc_period3, roc_period4, signal_period) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.kst_js(ptr0, len0, sma_period1, sma_period2, sma_period3, sma_period4, roc_period1, roc_period2, roc_period3, roc_period4, signal_period);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * Fast/unsafe WASM interface with pre-allocated memory
 * @param {number} in_ptr
 * @param {number} line_out_ptr
 * @param {number} signal_out_ptr
 * @param {number} len
 * @param {number} sma_period1
 * @param {number} sma_period2
 * @param {number} sma_period3
 * @param {number} sma_period4
 * @param {number} roc_period1
 * @param {number} roc_period2
 * @param {number} roc_period3
 * @param {number} roc_period4
 * @param {number} signal_period
 */
module.exports.kst_into = function(in_ptr, line_out_ptr, signal_out_ptr, len, sma_period1, sma_period2, sma_period3, sma_period4, roc_period1, roc_period2, roc_period3, roc_period4, signal_period) {
    const ret = wasm.kst_into(in_ptr, line_out_ptr, signal_out_ptr, len, sma_period1, sma_period2, sma_period3, sma_period4, roc_period1, roc_period2, roc_period3, roc_period4, signal_period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * Allocate memory for output arrays
 * @param {number} len
 * @returns {number}
 */
module.exports.kst_alloc = function(len) {
    const ret = wasm.kst_alloc(len);
    return ret >>> 0;
};

/**
 * Free allocated memory
 * @param {number} ptr
 * @param {number} len
 */
module.exports.kst_free = function(ptr, len) {
    wasm.kst_free(ptr, len);
};

/**
 * WASM interface for batch KST calculation
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.kst_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.kst_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * Fast batch calculation into pre-allocated memory
 * @param {number} in_ptr
 * @param {number} line_out_ptr
 * @param {number} signal_out_ptr
 * @param {number} len
 * @param {number} sma_period1_start
 * @param {number} sma_period1_end
 * @param {number} sma_period1_step
 * @param {number} sma_period2_start
 * @param {number} sma_period2_end
 * @param {number} sma_period2_step
 * @param {number} sma_period3_start
 * @param {number} sma_period3_end
 * @param {number} sma_period3_step
 * @param {number} sma_period4_start
 * @param {number} sma_period4_end
 * @param {number} sma_period4_step
 * @param {number} roc_period1_start
 * @param {number} roc_period1_end
 * @param {number} roc_period1_step
 * @param {number} roc_period2_start
 * @param {number} roc_period2_end
 * @param {number} roc_period2_step
 * @param {number} roc_period3_start
 * @param {number} roc_period3_end
 * @param {number} roc_period3_step
 * @param {number} roc_period4_start
 * @param {number} roc_period4_end
 * @param {number} roc_period4_step
 * @param {number} signal_period_start
 * @param {number} signal_period_end
 * @param {number} signal_period_step
 * @returns {number}
 */
module.exports.kst_batch_into = function(in_ptr, line_out_ptr, signal_out_ptr, len, sma_period1_start, sma_period1_end, sma_period1_step, sma_period2_start, sma_period2_end, sma_period2_step, sma_period3_start, sma_period3_end, sma_period3_step, sma_period4_start, sma_period4_end, sma_period4_step, roc_period1_start, roc_period1_end, roc_period1_step, roc_period2_start, roc_period2_end, roc_period2_step, roc_period3_start, roc_period3_end, roc_period3_step, roc_period4_start, roc_period4_end, roc_period4_step, signal_period_start, signal_period_end, signal_period_step) {
    const ret = wasm.kst_batch_into(in_ptr, line_out_ptr, signal_out_ptr, len, sma_period1_start, sma_period1_end, sma_period1_step, sma_period2_start, sma_period2_end, sma_period2_step, sma_period3_start, sma_period3_end, sma_period3_step, sma_period4_start, sma_period4_end, sma_period4_step, roc_period1_start, roc_period1_end, roc_period1_step, roc_period2_start, roc_period2_end, roc_period2_step, roc_period3_start, roc_period3_end, roc_period3_step, roc_period4_start, roc_period4_end, roc_period4_step, signal_period_start, signal_period_end, signal_period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @param {number} power
 * @returns {Float64Array}
 */
module.exports.vpwma_js = function(data, period, power) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.vpwma_js(ptr0, len0, period, power);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {Float64Array} data
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @param {number} power_start
 * @param {number} power_end
 * @param {number} power_step
 * @returns {Float64Array}
 */
module.exports.vpwma_batch_js = function(data, period_start, period_end, period_step, power_start, power_end, power_step) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.vpwma_batch_js(ptr0, len0, period_start, period_end, period_step, power_start, power_end, power_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @param {number} power_start
 * @param {number} power_end
 * @param {number} power_step
 * @returns {Float64Array}
 */
module.exports.vpwma_batch_metadata_js = function(period_start, period_end, period_step, power_start, power_end, power_step) {
    const ret = wasm.vpwma_batch_metadata_js(period_start, period_end, period_step, power_start, power_end, power_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.vpwma_alloc = function(len) {
    const ret = wasm.vpwma_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.vpwma_free = function(ptr, len) {
    wasm.vpwma_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 * @param {number} power
 */
module.exports.vpwma_into = function(in_ptr, out_ptr, len, period, power) {
    const ret = wasm.vpwma_into(in_ptr, out_ptr, len, period, power);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.vpwma_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.vpwma_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} data_len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @param {number} power_start
 * @param {number} power_end
 * @param {number} power_step
 * @returns {number}
 */
module.exports.vpwma_batch_into = function(in_ptr, out_ptr, data_len, period_start, period_end, period_step, power_start, power_end, power_step) {
    const ret = wasm.vpwma_batch_into(in_ptr, out_ptr, data_len, period_start, period_end, period_step, power_start, power_end, power_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} vis_atr
 * @param {number} vis_std
 * @param {number} sed_atr
 * @param {number} sed_std
 * @param {number} threshold
 * @returns {Float64Array}
 */
module.exports.damiani_volatmeter_js = function(data, vis_atr, vis_std, sed_atr, sed_std, threshold) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.damiani_volatmeter_js(ptr0, len0, vis_atr, vis_std, sed_atr, sed_std, threshold);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.damiani_volatmeter_alloc = function(len) {
    const ret = wasm.damiani_volatmeter_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.damiani_volatmeter_free = function(ptr, len) {
    wasm.damiani_volatmeter_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} vol_ptr
 * @param {number} anti_ptr
 * @param {number} len
 * @param {number} vis_atr
 * @param {number} vis_std
 * @param {number} sed_atr
 * @param {number} sed_std
 * @param {number} threshold
 */
module.exports.damiani_volatmeter_into = function(in_ptr, vol_ptr, anti_ptr, len, vis_atr, vis_std, sed_atr, sed_std, threshold) {
    const ret = wasm.damiani_volatmeter_into(in_ptr, vol_ptr, anti_ptr, len, vis_atr, vis_std, sed_atr, sed_std, threshold);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.damiani_volatmeter_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.damiani_volatmeter_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} in_ptr
 * @param {number} vol_ptr
 * @param {number} anti_ptr
 * @param {number} len
 * @param {number} vis_atr_start
 * @param {number} vis_atr_end
 * @param {number} vis_atr_step
 * @param {number} vis_std_start
 * @param {number} vis_std_end
 * @param {number} vis_std_step
 * @param {number} sed_atr_start
 * @param {number} sed_atr_end
 * @param {number} sed_atr_step
 * @param {number} sed_std_start
 * @param {number} sed_std_end
 * @param {number} sed_std_step
 * @param {number} threshold_start
 * @param {number} threshold_end
 * @param {number} threshold_step
 * @returns {number}
 */
module.exports.damiani_volatmeter_batch_into = function(in_ptr, vol_ptr, anti_ptr, len, vis_atr_start, vis_atr_end, vis_atr_step, vis_std_start, vis_std_end, vis_std_step, sed_atr_start, sed_atr_end, sed_atr_step, sed_std_start, sed_std_end, sed_std_step, threshold_start, threshold_end, threshold_step) {
    const ret = wasm.damiani_volatmeter_batch_into(in_ptr, vol_ptr, anti_ptr, len, vis_atr_start, vis_atr_end, vis_atr_step, vis_std_start, vis_std_end, vis_std_step, sed_atr_start, sed_atr_end, sed_atr_step, sed_std_start, sed_std_end, sed_std_step, threshold_start, threshold_end, threshold_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} close
 * @param {number} length_bb
 * @param {number} mult_bb
 * @param {number} length_kc
 * @param {number} mult_kc
 * @returns {Float64Array}
 */
module.exports.squeeze_momentum_js = function(high, low, close, length_bb, mult_bb, length_kc, mult_kc) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.squeeze_momentum_js(ptr0, len0, ptr1, len1, ptr2, len2, length_bb, mult_bb, length_kc, mult_kc);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v4 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v4;
};

/**
 * @param {number} in_ptr
 * @param {number} squeeze_ptr
 * @param {number} momentum_ptr
 * @param {number} momentum_signal_ptr
 * @param {number} len
 * @param {number} length_bb
 * @param {number} mult_bb
 * @param {number} length_kc
 * @param {number} mult_kc
 */
module.exports.squeeze_momentum_into = function(in_ptr, squeeze_ptr, momentum_ptr, momentum_signal_ptr, len, length_bb, mult_bb, length_kc, mult_kc) {
    const ret = wasm.squeeze_momentum_into(in_ptr, squeeze_ptr, momentum_ptr, momentum_signal_ptr, len, length_bb, mult_bb, length_kc, mult_kc);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.squeeze_momentum_alloc = function(len) {
    const ret = wasm.squeeze_momentum_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.squeeze_momentum_free = function(ptr, len) {
    wasm.squeeze_momentum_free(ptr, len);
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} close
 * @param {any} config
 * @returns {any}
 */
module.exports.squeeze_momentum_batch = function(high, low, close, config) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.squeeze_momentum_batch(ptr0, len0, ptr1, len1, ptr2, len2, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @param {number} ma_len
 * @param {number} matype
 * @param {number} devtype
 * @returns {Float64Array}
 */
module.exports.rvi_js = function(data, period, ma_len, matype, devtype) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.rvi_js(ptr0, len0, period, ma_len, matype, devtype);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.rvi_alloc = function(len) {
    const ret = wasm.rvi_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.rvi_free = function(ptr, len) {
    wasm.rvi_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 * @param {number} ma_len
 * @param {number} matype
 * @param {number} devtype
 */
module.exports.rvi_into = function(in_ptr, out_ptr, len, period, ma_len, matype, devtype) {
    const ret = wasm.rvi_into(in_ptr, out_ptr, len, period, ma_len, matype, devtype);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.rvi_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.rvi_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} p_start
 * @param {number} p_end
 * @param {number} p_step
 * @param {number} m_start
 * @param {number} m_end
 * @param {number} m_step
 * @param {number} t_start
 * @param {number} t_end
 * @param {number} t_step
 * @param {number} d_start
 * @param {number} d_end
 * @param {number} d_step
 * @returns {number}
 */
module.exports.rvi_batch_into = function(in_ptr, out_ptr, len, p_start, p_end, p_step, m_start, m_end, m_step, t_start, t_end, t_step, d_start, d_end, d_step) {
    const ret = wasm.rvi_batch_into(in_ptr, out_ptr, len, p_start, p_end, p_step, m_start, m_end, m_step, t_start, t_end, t_step, d_start, d_end, d_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} fast_period
 * @param {number} slow_period
 * @param {number} devup
 * @param {number} devdn
 * @param {string} fast_ma_type
 * @param {string} slow_ma_type
 * @returns {Float64Array}
 */
module.exports.mab_js = function(data, fast_period, slow_period, devup, devdn, fast_ma_type, slow_ma_type) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passStringToWasm0(fast_ma_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passStringToWasm0(slow_ma_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.mab_js(ptr0, len0, fast_period, slow_period, devup, devdn, ptr1, len1, ptr2, len2);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v4 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v4;
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.mab_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.mab_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.mab_alloc = function(len) {
    const ret = wasm.mab_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.mab_free = function(ptr, len) {
    wasm.mab_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} upper_ptr
 * @param {number} middle_ptr
 * @param {number} lower_ptr
 * @param {number} len
 * @param {number} fast_period
 * @param {number} slow_period
 * @param {number} devup
 * @param {number} devdn
 * @param {string} fast_ma_type
 * @param {string} slow_ma_type
 */
module.exports.mab_into = function(in_ptr, upper_ptr, middle_ptr, lower_ptr, len, fast_period, slow_period, devup, devdn, fast_ma_type, slow_ma_type) {
    const ptr0 = passStringToWasm0(fast_ma_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passStringToWasm0(slow_ma_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.mab_into(in_ptr, upper_ptr, middle_ptr, lower_ptr, len, fast_period, slow_period, devup, devdn, ptr0, len0, ptr1, len1);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} in_ptr
 * @param {number} upper_ptr
 * @param {number} middle_ptr
 * @param {number} lower_ptr
 * @param {number} len
 * @param {number} fast_period_start
 * @param {number} fast_period_end
 * @param {number} fast_period_step
 * @param {number} slow_period_start
 * @param {number} slow_period_end
 * @param {number} slow_period_step
 * @param {number} devup_start
 * @param {number} devup_end
 * @param {number} devup_step
 * @param {number} devdn_start
 * @param {number} devdn_end
 * @param {number} devdn_step
 * @param {string} fast_ma_type
 * @param {string} slow_ma_type
 * @returns {number}
 */
module.exports.mab_batch_into = function(in_ptr, upper_ptr, middle_ptr, lower_ptr, len, fast_period_start, fast_period_end, fast_period_step, slow_period_start, slow_period_end, slow_period_step, devup_start, devup_end, devup_step, devdn_start, devdn_end, devdn_step, fast_ma_type, slow_ma_type) {
    const ptr0 = passStringToWasm0(fast_ma_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passStringToWasm0(slow_ma_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.mab_batch_into(in_ptr, upper_ptr, middle_ptr, lower_ptr, len, fast_period_start, fast_period_end, fast_period_step, slow_period_start, slow_period_end, slow_period_step, devup_start, devup_end, devup_step, devdn_start, devdn_end, devdn_step, ptr0, len0, ptr1, len1);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @param {number} offset
 * @param {number} sigma
 * @returns {Float64Array}
 */
module.exports.alma_js = function(data, period, offset, sigma) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.alma_js(ptr0, len0, period, offset, sigma);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.alma_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.alma_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.alma_alloc = function(len) {
    const ret = wasm.alma_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.alma_free = function(ptr, len) {
    wasm.alma_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 * @param {number} offset
 * @param {number} sigma
 */
module.exports.alma_into = function(in_ptr, out_ptr, len, period, offset, sigma) {
    const ret = wasm.alma_into(in_ptr, out_ptr, len, period, offset, sigma);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @param {number} offset_start
 * @param {number} offset_end
 * @param {number} offset_step
 * @param {number} sigma_start
 * @param {number} sigma_end
 * @param {number} sigma_step
 * @returns {number}
 */
module.exports.alma_batch_into = function(in_ptr, out_ptr, len, period_start, period_end, period_step, offset_start, offset_end, offset_step, sigma_start, sigma_end, sigma_step) {
    const ret = wasm.alma_batch_into(in_ptr, out_ptr, len, period_start, period_end, period_step, offset_start, offset_end, offset_step, sigma_start, sigma_end, sigma_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.swma_js = function(data, period) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.swma_js(ptr0, len0, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {Float64Array} data
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {Float64Array}
 */
module.exports.swma_batch_js = function(data, period_start, period_end, period_step) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.swma_batch_js(ptr0, len0, period_start, period_end, period_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {Float64Array}
 */
module.exports.swma_batch_metadata_js = function(period_start, period_end, period_step) {
    const ret = wasm.swma_batch_metadata_js(period_start, period_end, period_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.swma_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.swma_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.swma_alloc = function(len) {
    const ret = wasm.swma_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.swma_free = function(ptr, len) {
    wasm.swma_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.swma_into = function(in_ptr, out_ptr, len, period) {
    const ret = wasm.swma_into(in_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.swma_batch_into = function(in_ptr, out_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.swma_batch_into(in_ptr, out_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} close
 * @param {number} fastk_period
 * @param {number} fastd_period
 * @param {number} fastd_matype
 * @returns {Float64Array}
 */
module.exports.stochf_js = function(high, low, close, fastk_period, fastd_period, fastd_matype) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.stochf_js(ptr0, len0, ptr1, len1, ptr2, len2, fastk_period, fastd_period, fastd_matype);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v4 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v4;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.stochf_alloc = function(len) {
    const ret = wasm.stochf_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.stochf_free = function(ptr, len) {
    wasm.stochf_free(ptr, len);
};

/**
 * @param {number} in_high_ptr
 * @param {number} in_low_ptr
 * @param {number} in_close_ptr
 * @param {number} out_k_ptr
 * @param {number} out_d_ptr
 * @param {number} len
 * @param {number} fastk_period
 * @param {number} fastd_period
 * @param {number} fastd_matype
 */
module.exports.stochf_into = function(in_high_ptr, in_low_ptr, in_close_ptr, out_k_ptr, out_d_ptr, len, fastk_period, fastd_period, fastd_matype) {
    const ret = wasm.stochf_into(in_high_ptr, in_low_ptr, in_close_ptr, out_k_ptr, out_d_ptr, len, fastk_period, fastd_period, fastd_matype);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} close
 * @param {any} config
 * @returns {any}
 */
module.exports.stochf_batch = function(high, low, close, config) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.stochf_batch(ptr0, len0, ptr1, len1, ptr2, len2, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} in_high_ptr
 * @param {number} in_low_ptr
 * @param {number} in_close_ptr
 * @param {number} out_k_ptr
 * @param {number} out_d_ptr
 * @param {number} len
 * @param {number} fastk_start
 * @param {number} fastk_end
 * @param {number} fastk_step
 * @param {number} fastd_start
 * @param {number} fastd_end
 * @param {number} fastd_step
 * @param {number} fastd_matype
 * @returns {number}
 */
module.exports.stochf_batch_into = function(in_high_ptr, in_low_ptr, in_close_ptr, out_k_ptr, out_d_ptr, len, fastk_start, fastk_end, fastk_step, fastd_start, fastd_end, fastd_step, fastd_matype) {
    const ret = wasm.stochf_batch_into(in_high_ptr, in_low_ptr, in_close_ptr, out_k_ptr, out_d_ptr, len, fastk_start, fastk_end, fastk_step, fastd_start, fastd_end, fastd_step, fastd_matype);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} close
 * @param {number | null} [window]
 * @param {number | null} [sc]
 * @param {number | null} [fc]
 * @returns {Float64Array}
 */
module.exports.frama_js = function(high, low, close, window, sc, fc) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.frama_js(ptr0, len0, ptr1, len1, ptr2, len2, isLikeNone(window) ? 0x100000001 : (window) >>> 0, isLikeNone(sc) ? 0x100000001 : (sc) >>> 0, isLikeNone(fc) ? 0x100000001 : (fc) >>> 0);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v4 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v4;
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} close
 * @param {number} window_start
 * @param {number} window_end
 * @param {number} window_step
 * @param {number} sc_start
 * @param {number} sc_end
 * @param {number} sc_step
 * @param {number} fc_start
 * @param {number} fc_end
 * @param {number} fc_step
 * @returns {Float64Array}
 */
module.exports.frama_batch_js = function(high, low, close, window_start, window_end, window_step, sc_start, sc_end, sc_step, fc_start, fc_end, fc_step) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.frama_batch_js(ptr0, len0, ptr1, len1, ptr2, len2, window_start, window_end, window_step, sc_start, sc_end, sc_step, fc_start, fc_end, fc_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v4 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v4;
};

let cachedUint32ArrayMemory0 = null;

function getUint32ArrayMemory0() {
    if (cachedUint32ArrayMemory0 === null || cachedUint32ArrayMemory0.byteLength === 0) {
        cachedUint32ArrayMemory0 = new Uint32Array(wasm.memory.buffer);
    }
    return cachedUint32ArrayMemory0;
}

function getArrayU32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}
/**
 * @param {number} window_start
 * @param {number} window_end
 * @param {number} window_step
 * @param {number} sc_start
 * @param {number} sc_end
 * @param {number} sc_step
 * @param {number} fc_start
 * @param {number} fc_end
 * @param {number} fc_step
 * @returns {Uint32Array}
 */
module.exports.frama_batch_metadata_js = function(window_start, window_end, window_step, sc_start, sc_end, sc_step, fc_start, fc_end, fc_step) {
    const ret = wasm.frama_batch_metadata_js(window_start, window_end, window_step, sc_start, sc_end, sc_step, fc_start, fc_end, fc_step);
    var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v1;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.frama_alloc = function(len) {
    const ret = wasm.frama_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.frama_free = function(ptr, len) {
    wasm.frama_free(ptr, len);
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} close_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number | null} [window]
 * @param {number | null} [sc]
 * @param {number | null} [fc]
 */
module.exports.frama_into = function(high_ptr, low_ptr, close_ptr, out_ptr, len, window, sc, fc) {
    const ret = wasm.frama_into(high_ptr, low_ptr, close_ptr, out_ptr, len, isLikeNone(window) ? 0x100000001 : (window) >>> 0, isLikeNone(sc) ? 0x100000001 : (sc) >>> 0, isLikeNone(fc) ? 0x100000001 : (fc) >>> 0);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} close
 * @param {any} config
 * @returns {any}
 */
module.exports.frama_batch = function(high, low, close, config) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.frama_batch(ptr0, len0, ptr1, len1, ptr2, len2, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} close_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} window_start
 * @param {number} window_end
 * @param {number} window_step
 * @param {number} sc_start
 * @param {number} sc_end
 * @param {number} sc_step
 * @param {number} fc_start
 * @param {number} fc_end
 * @param {number} fc_step
 * @returns {number}
 */
module.exports.frama_batch_into = function(high_ptr, low_ptr, close_ptr, out_ptr, len, window_start, window_end, window_step, sc_start, sc_end, sc_step, fc_start, fc_end, fc_step) {
    const ret = wasm.frama_batch_into(high_ptr, low_ptr, close_ptr, out_ptr, len, window_start, window_end, window_step, sc_start, sc_end, sc_step, fc_start, fc_end, fc_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @returns {any}
 */
module.exports.msw_js = function(data, period) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.msw_js(ptr0, len0, period);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @returns {any}
 */
module.exports.msw_wasm = function(data, period) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.msw_wasm(ptr0, len0, period);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.msw_into_flat = function(in_ptr, out_ptr, len, period) {
    const ret = wasm.msw_into_flat(in_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} in_ptr
 * @param {number} sine_ptr
 * @param {number} lead_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.msw_into = function(in_ptr, sine_ptr, lead_ptr, len, period) {
    const ret = wasm.msw_into(in_ptr, sine_ptr, lead_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.msw_alloc = function(len) {
    const ret = wasm.msw_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.msw_free = function(ptr, len) {
    wasm.msw_free(ptr, len);
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.msw_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.msw_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {Float64Array} data
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {any}
 */
module.exports.msw_batch_js = function(data, period_start, period_end, period_step) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.msw_batch_js(ptr0, len0, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {Float64Array}
 */
module.exports.msw_batch_metadata_js = function(period_start, period_end, period_step) {
    const ret = wasm.msw_batch_metadata_js(period_start, period_end, period_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.msw_batch_into_flat = function(in_ptr, out_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.msw_batch_into_flat(in_ptr, out_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {number} in_ptr
 * @param {number} sine_ptr
 * @param {number} lead_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.msw_batch_into = function(in_ptr, sine_ptr, lead_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.msw_batch_into(in_ptr, sine_ptr, lead_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} channel_length
 * @param {number} average_length
 * @param {number} ma_length
 * @param {number} factor
 * @returns {Float64Array}
 */
module.exports.wavetrend_js = function(data, channel_length, average_length, ma_length, factor) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.wavetrend_js(ptr0, len0, channel_length, average_length, ma_length, factor);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} in_ptr
 * @param {number} wt1_ptr
 * @param {number} wt2_ptr
 * @param {number} wt_diff_ptr
 * @param {number} len
 * @param {number} channel_length
 * @param {number} average_length
 * @param {number} ma_length
 * @param {number} factor
 */
module.exports.wavetrend_into = function(in_ptr, wt1_ptr, wt2_ptr, wt_diff_ptr, len, channel_length, average_length, ma_length, factor) {
    const ret = wasm.wavetrend_into(in_ptr, wt1_ptr, wt2_ptr, wt_diff_ptr, len, channel_length, average_length, ma_length, factor);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.wavetrend_alloc = function(len) {
    const ret = wasm.wavetrend_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.wavetrend_free = function(ptr, len) {
    wasm.wavetrend_free(ptr, len);
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.wavetrend_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.wavetrend_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {Float64Array} data
 * @param {number} jaw_period
 * @param {number} jaw_offset
 * @param {number} teeth_period
 * @param {number} teeth_offset
 * @param {number} lips_period
 * @param {number} lips_offset
 * @returns {Float64Array}
 */
module.exports.alligator_js = function(data, jaw_period, jaw_offset, teeth_period, teeth_offset, lips_period, lips_offset) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.alligator_js(ptr0, len0, jaw_period, jaw_offset, teeth_period, teeth_offset, lips_period, lips_offset);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.alligator_alloc = function(len) {
    const ret = wasm.alligator_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.alligator_free = function(ptr, len) {
    wasm.alligator_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} jaw_ptr
 * @param {number} teeth_ptr
 * @param {number} lips_ptr
 * @param {number} len
 * @param {number} jaw_period
 * @param {number} jaw_offset
 * @param {number} teeth_period
 * @param {number} teeth_offset
 * @param {number} lips_period
 * @param {number} lips_offset
 */
module.exports.alligator_into = function(in_ptr, jaw_ptr, teeth_ptr, lips_ptr, len, jaw_period, jaw_offset, teeth_period, teeth_offset, lips_period, lips_offset) {
    const ret = wasm.alligator_into(in_ptr, jaw_ptr, teeth_ptr, lips_ptr, len, jaw_period, jaw_offset, teeth_period, teeth_offset, lips_period, lips_offset);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} data
 * @param {number} jaw_period_start
 * @param {number} jaw_period_end
 * @param {number} jaw_period_step
 * @param {number} jaw_offset_start
 * @param {number} jaw_offset_end
 * @param {number} jaw_offset_step
 * @param {number} teeth_period_start
 * @param {number} teeth_period_end
 * @param {number} teeth_period_step
 * @param {number} teeth_offset_start
 * @param {number} teeth_offset_end
 * @param {number} teeth_offset_step
 * @param {number} lips_period_start
 * @param {number} lips_period_end
 * @param {number} lips_period_step
 * @param {number} lips_offset_start
 * @param {number} lips_offset_end
 * @param {number} lips_offset_step
 * @returns {Float64Array}
 */
module.exports.alligator_batch_js = function(data, jaw_period_start, jaw_period_end, jaw_period_step, jaw_offset_start, jaw_offset_end, jaw_offset_step, teeth_period_start, teeth_period_end, teeth_period_step, teeth_offset_start, teeth_offset_end, teeth_offset_step, lips_period_start, lips_period_end, lips_period_step, lips_offset_start, lips_offset_end, lips_offset_step) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.alligator_batch_js(ptr0, len0, jaw_period_start, jaw_period_end, jaw_period_step, jaw_offset_start, jaw_offset_end, jaw_offset_step, teeth_period_start, teeth_period_end, teeth_period_step, teeth_offset_start, teeth_offset_end, teeth_offset_step, lips_period_start, lips_period_end, lips_period_step, lips_offset_start, lips_offset_end, lips_offset_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} jaw_period_start
 * @param {number} jaw_period_end
 * @param {number} jaw_period_step
 * @param {number} jaw_offset_start
 * @param {number} jaw_offset_end
 * @param {number} jaw_offset_step
 * @param {number} teeth_period_start
 * @param {number} teeth_period_end
 * @param {number} teeth_period_step
 * @param {number} teeth_offset_start
 * @param {number} teeth_offset_end
 * @param {number} teeth_offset_step
 * @param {number} lips_period_start
 * @param {number} lips_period_end
 * @param {number} lips_period_step
 * @param {number} lips_offset_start
 * @param {number} lips_offset_end
 * @param {number} lips_offset_step
 * @returns {Float64Array}
 */
module.exports.alligator_batch_metadata_js = function(jaw_period_start, jaw_period_end, jaw_period_step, jaw_offset_start, jaw_offset_end, jaw_offset_step, teeth_period_start, teeth_period_end, teeth_period_step, teeth_offset_start, teeth_offset_end, teeth_offset_step, lips_period_start, lips_period_end, lips_period_step, lips_offset_start, lips_offset_end, lips_offset_step) {
    const ret = wasm.alligator_batch_metadata_js(jaw_period_start, jaw_period_end, jaw_period_step, jaw_offset_start, jaw_offset_end, jaw_offset_step, teeth_period_start, teeth_period_end, teeth_period_step, teeth_offset_start, teeth_offset_end, teeth_offset_step, lips_period_start, lips_period_end, lips_period_step, lips_offset_start, lips_offset_end, lips_offset_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.alligator_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.alligator_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} _close
 * @param {Float64Array} _volume
 * @param {number} period
 * @param {number} delta
 * @param {number} fraction
 * @returns {any}
 */
module.exports.emd_js = function(high, low, _close, _volume, period, delta, fraction) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(_close, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArrayF64ToWasm0(_volume, wasm.__wbindgen_malloc);
    const len3 = WASM_VECTOR_LEN;
    const ret = wasm.emd_js(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, period, delta, fraction);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.emd_alloc = function(len) {
    const ret = wasm.emd_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.emd_free = function(ptr, len) {
    wasm.emd_free(ptr, len);
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} _close_ptr
 * @param {number} _volume_ptr
 * @param {number} upper_ptr
 * @param {number} middle_ptr
 * @param {number} lower_ptr
 * @param {number} len
 * @param {number} period
 * @param {number} delta
 * @param {number} fraction
 */
module.exports.emd_into = function(high_ptr, low_ptr, _close_ptr, _volume_ptr, upper_ptr, middle_ptr, lower_ptr, len, period, delta, fraction) {
    const ret = wasm.emd_into(high_ptr, low_ptr, _close_ptr, _volume_ptr, upper_ptr, middle_ptr, lower_ptr, len, period, delta, fraction);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} _close
 * @param {Float64Array} _volume
 * @param {any} config
 * @returns {any}
 */
module.exports.emd_batch = function(high, low, _close, _volume, config) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(_close, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArrayF64ToWasm0(_volume, wasm.__wbindgen_malloc);
    const len3 = WASM_VECTOR_LEN;
    const ret = wasm.emd_batch(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} close_ptr
 * @param {number} volume_ptr
 * @param {number} upper_ptr
 * @param {number} middle_ptr
 * @param {number} lower_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @param {number} delta_start
 * @param {number} delta_end
 * @param {number} delta_step
 * @param {number} fraction_start
 * @param {number} fraction_end
 * @param {number} fraction_step
 * @returns {number}
 */
module.exports.emd_batch_into = function(high_ptr, low_ptr, close_ptr, volume_ptr, upper_ptr, middle_ptr, lower_ptr, len, period_start, period_end, period_step, delta_start, delta_end, delta_step, fraction_start, fraction_end, fraction_step) {
    const ret = wasm.emd_batch_into(high_ptr, low_ptr, close_ptr, volume_ptr, upper_ptr, middle_ptr, lower_ptr, len, period_start, period_end, period_step, delta_start, delta_end, delta_step, fraction_start, fraction_end, fraction_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} timestamps
 * @param {Float64Array} volumes
 * @param {Float64Array} prices
 * @param {string | null} [anchor]
 * @param {string | null} [kernel]
 * @returns {Float64Array}
 */
module.exports.vwap_js = function(timestamps, volumes, prices, anchor, kernel) {
    const ptr0 = passArrayF64ToWasm0(timestamps, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(volumes, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(prices, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    var ptr3 = isLikeNone(anchor) ? 0 : passStringToWasm0(anchor, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    var len3 = WASM_VECTOR_LEN;
    var ptr4 = isLikeNone(kernel) ? 0 : passStringToWasm0(kernel, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    var len4 = WASM_VECTOR_LEN;
    const ret = wasm.vwap_js(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, ptr4, len4);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v6 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v6;
};

/**
 * @param {number} timestamps_ptr
 * @param {number} volumes_ptr
 * @param {number} prices_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {string | null} [anchor]
 */
module.exports.vwap_into = function(timestamps_ptr, volumes_ptr, prices_ptr, out_ptr, len, anchor) {
    var ptr0 = isLikeNone(anchor) ? 0 : passStringToWasm0(anchor, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    var len0 = WASM_VECTOR_LEN;
    const ret = wasm.vwap_into(timestamps_ptr, volumes_ptr, prices_ptr, out_ptr, len, ptr0, len0);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.vwap_alloc = function(len) {
    const ret = wasm.vwap_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.vwap_free = function(ptr, len) {
    wasm.vwap_free(ptr, len);
};

/**
 * @param {Float64Array} timestamps
 * @param {Float64Array} volumes
 * @param {Float64Array} prices
 * @param {any} config
 * @returns {any}
 */
module.exports.vwap_batch = function(timestamps, volumes, prices, config) {
    const ptr0 = passArrayF64ToWasm0(timestamps, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(volumes, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(prices, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.vwap_batch(ptr0, len0, ptr1, len1, ptr2, len2, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} timestamps_ptr
 * @param {number} volumes_ptr
 * @param {number} prices_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {string} anchor_start
 * @param {string} anchor_end
 * @param {number} anchor_step
 * @returns {number}
 */
module.exports.vwap_batch_into = function(timestamps_ptr, volumes_ptr, prices_ptr, out_ptr, len, anchor_start, anchor_end, anchor_step) {
    const ptr0 = passStringToWasm0(anchor_start, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passStringToWasm0(anchor_end, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.vwap_batch_into(timestamps_ptr, volumes_ptr, prices_ptr, out_ptr, len, ptr0, len0, ptr1, len1, anchor_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

function getArrayJsValueFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    const mem = getDataViewMemory0();
    const result = [];
    for (let i = ptr; i < ptr + 4 * len; i += 4) {
        result.push(wasm.__wbindgen_export_5.get(mem.getUint32(i, true)));
    }
    wasm.__externref_drop_slice(ptr, len);
    return result;
}
/**
 * @param {string} anchor_start
 * @param {string} anchor_end
 * @param {number} anchor_step
 * @returns {string[]}
 */
module.exports.vwap_batch_metadata_js = function(anchor_start, anchor_end, anchor_step) {
    const ptr0 = passStringToWasm0(anchor_start, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passStringToWasm0(anchor_end, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.vwap_batch_metadata_js(ptr0, len0, ptr1, len1, anchor_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayJsValueFromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v3;
};

/**
 * @param {Float64Array} data
 * @param {number | null} [period]
 * @param {number | null} [offset]
 * @returns {Float64Array}
 */
module.exports.epma_js = function(data, period, offset) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.epma_js(ptr0, len0, isLikeNone(period) ? 0x100000001 : (period) >>> 0, isLikeNone(offset) ? 0x100000001 : (offset) >>> 0);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.epma_alloc = function(len) {
    const ret = wasm.epma_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.epma_free = function(ptr, len) {
    wasm.epma_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number | null} [period]
 * @param {number | null} [offset]
 */
module.exports.epma_into = function(in_ptr, out_ptr, len, period, offset) {
    const ret = wasm.epma_into(in_ptr, out_ptr, len, isLikeNone(period) ? 0x100000001 : (period) >>> 0, isLikeNone(offset) ? 0x100000001 : (offset) >>> 0);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.epma_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.epma_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {Float64Array} data
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @param {number} offset_start
 * @param {number} offset_end
 * @param {number} offset_step
 * @returns {Float64Array}
 */
module.exports.epma_batch_js = function(data, period_start, period_end, period_step, offset_start, offset_end, offset_step) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.epma_batch_js(ptr0, len0, period_start, period_end, period_step, offset_start, offset_end, offset_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @param {number} offset_start
 * @param {number} offset_end
 * @param {number} offset_step
 * @returns {Uint32Array}
 */
module.exports.epma_batch_metadata_js = function(period_start, period_end, period_step, offset_start, offset_end, offset_step) {
    const ret = wasm.epma_batch_metadata_js(period_start, period_end, period_step, offset_start, offset_end, offset_step);
    var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v1;
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @param {number} offset_start
 * @param {number} offset_end
 * @param {number} offset_step
 * @returns {number}
 */
module.exports.epma_batch_into = function(in_ptr, out_ptr, len, period_start, period_end, period_step, offset_start, offset_end, offset_step) {
    const ret = wasm.epma_batch_into(in_ptr, out_ptr, len, period_start, period_end, period_step, offset_start, offset_end, offset_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} close
 * @param {number} p
 * @param {number} x
 * @param {number} q
 * @returns {Float64Array}
 */
module.exports.cksp_js = function(high, low, close, p, x, q) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.cksp_js(ptr0, len0, ptr1, len1, ptr2, len2, p, x, q);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v4 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v4;
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} close_ptr
 * @param {number} long_ptr
 * @param {number} short_ptr
 * @param {number} len
 * @param {number} p
 * @param {number} x
 * @param {number} q
 */
module.exports.cksp_into = function(high_ptr, low_ptr, close_ptr, long_ptr, short_ptr, len, p, x, q) {
    const ret = wasm.cksp_into(high_ptr, low_ptr, close_ptr, long_ptr, short_ptr, len, p, x, q);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.cksp_alloc = function(len) {
    const ret = wasm.cksp_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.cksp_free = function(ptr, len) {
    wasm.cksp_free(ptr, len);
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} close
 * @param {any} config
 * @returns {any}
 */
module.exports.cksp_batch = function(high, low, close, config) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.cksp_batch(ptr0, len0, ptr1, len1, ptr2, len2, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} close_ptr
 * @param {number} long_ptr
 * @param {number} short_ptr
 * @param {number} len
 * @param {number} p_start
 * @param {number} p_end
 * @param {number} p_step
 * @param {number} x_start
 * @param {number} x_end
 * @param {number} x_step
 * @param {number} q_start
 * @param {number} q_end
 * @param {number} q_step
 * @returns {number}
 */
module.exports.cksp_batch_into = function(high_ptr, low_ptr, close_ptr, long_ptr, short_ptr, len, p_start, p_end, p_step, x_start, x_end, x_step, q_start, q_end, q_step) {
    const ret = wasm.cksp_batch_into(high_ptr, low_ptr, close_ptr, long_ptr, short_ptr, len, p_start, p_end, p_step, x_start, x_end, x_step, q_start, q_end, q_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.srwma_js = function(data, period) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.srwma_js(ptr0, len0, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.srwma_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.srwma_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {Float64Array} data
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {Float64Array}
 */
module.exports.srwma_batch_js = function(data, period_start, period_end, period_step) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.srwma_batch_js(ptr0, len0, period_start, period_end, period_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {Uint32Array}
 */
module.exports.srwma_batch_metadata_js = function(period_start, period_end, period_step) {
    const ret = wasm.srwma_batch_metadata_js(period_start, period_end, period_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v1;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.srwma_alloc = function(len) {
    const ret = wasm.srwma_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.srwma_free = function(ptr, len) {
    wasm.srwma_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.srwma_into = function(in_ptr, out_ptr, len, period) {
    const ret = wasm.srwma_into(in_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.srwma_batch_into = function(in_ptr, out_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.srwma_batch_into(in_ptr, out_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * Safe API: Single calculation with automatic memory management
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} close
 * @param {number} period
 * @param {number} mult
 * @param {string} direction
 * @returns {Float64Array}
 */
module.exports.chande_js = function(high, low, close, period, mult, direction) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passStringToWasm0(direction, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len3 = WASM_VECTOR_LEN;
    const ret = wasm.chande_js(ptr0, len0, ptr1, len1, ptr2, len2, period, mult, ptr3, len3);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v5 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v5;
};

/**
 * Safe API: Batch processing with JavaScript-friendly output
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} close
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @param {number} mult_start
 * @param {number} mult_end
 * @param {number} mult_step
 * @param {string} direction
 * @returns {any}
 */
module.exports.chande_batch_js = function(high, low, close, period_start, period_end, period_step, mult_start, mult_end, mult_step, direction) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passStringToWasm0(direction, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len3 = WASM_VECTOR_LEN;
    const ret = wasm.chande_batch_js(ptr0, len0, ptr1, len1, ptr2, len2, period_start, period_end, period_step, mult_start, mult_end, mult_step, ptr3, len3);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * Memory allocation for WASM
 * @param {number} len
 * @returns {number}
 */
module.exports.chande_alloc = function(len) {
    const ret = wasm.chande_alloc(len);
    return ret >>> 0;
};

/**
 * Memory deallocation for WASM
 * @param {number} ptr
 * @param {number} len
 */
module.exports.chande_free = function(ptr, len) {
    wasm.chande_free(ptr, len);
};

/**
 * Fast/Zero-copy API: Compute directly into pre-allocated buffer
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} close_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 * @param {number} mult
 * @param {string} direction
 */
module.exports.chande_into = function(high_ptr, low_ptr, close_ptr, out_ptr, len, period, mult, direction) {
    const ptr0 = passStringToWasm0(direction, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.chande_into(high_ptr, low_ptr, close_ptr, out_ptr, len, period, mult, ptr0, len0);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * Fast/Zero-copy API: Batch computation into pre-allocated buffer
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} close_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @param {number} mult_start
 * @param {number} mult_end
 * @param {number} mult_step
 * @param {string} direction
 * @returns {number}
 */
module.exports.chande_batch_into = function(high_ptr, low_ptr, close_ptr, out_ptr, len, period_start, period_end, period_step, mult_start, mult_end, mult_step, direction) {
    const ptr0 = passStringToWasm0(direction, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.chande_batch_into(high_ptr, low_ptr, close_ptr, out_ptr, len, period_start, period_end, period_step, mult_start, mult_end, mult_step, ptr0, len0);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * Compute the Tilson T3 Moving Average.
 *
 * # Arguments
 * * `data` - Input data array
 * * `period` - Period (must be >= 1)
 * * `volume_factor` - Volume factor (0.0 to 1.0), defaults to 0.0
 *
 * # Returns
 * Array of Tilson values, same length as input
 * @param {Float64Array} data
 * @param {number} period
 * @param {number | null} [volume_factor]
 * @returns {Float64Array}
 */
module.exports.tilson_js = function(data, period, volume_factor) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.tilson_js(ptr0, len0, period, !isLikeNone(volume_factor), isLikeNone(volume_factor) ? 0 : volume_factor);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * Compute Tilson for multiple parameter combinations in a single pass.
 *
 * # Arguments
 * * `data` - Input data array
 * * `period_start`, `period_end`, `period_step` - Period range parameters
 * * `v_factor_start`, `v_factor_end`, `v_factor_step` - Volume factor range parameters
 *
 * # Returns
 * Flattened array of values (row-major order)
 * @param {Float64Array} data
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @param {number} v_factor_start
 * @param {number} v_factor_end
 * @param {number} v_factor_step
 * @returns {Float64Array}
 */
module.exports.tilson_batch_js = function(data, period_start, period_end, period_step, v_factor_start, v_factor_end, v_factor_step) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.tilson_batch_js(ptr0, len0, period_start, period_end, period_step, v_factor_start, v_factor_end, v_factor_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * Get metadata about batch computation.
 *
 * # Arguments
 * * Period and volume factor range parameters (same as tilson_batch_js)
 *
 * # Returns
 * Array containing [periods array, volume_factors array] flattened
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @param {number} v_factor_start
 * @param {number} v_factor_end
 * @param {number} v_factor_step
 * @returns {Float64Array}
 */
module.exports.tilson_batch_metadata_js = function(period_start, period_end, period_step, v_factor_start, v_factor_end, v_factor_step) {
    const ret = wasm.tilson_batch_metadata_js(period_start, period_end, period_step, v_factor_start, v_factor_end, v_factor_step);
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.tilson_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.tilson_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.tilson_alloc = function(len) {
    const ret = wasm.tilson_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.tilson_free = function(ptr, len) {
    wasm.tilson_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 * @param {number} volume_factor
 */
module.exports.tilson_into = function(in_ptr, out_ptr, len, period, volume_factor) {
    const ret = wasm.tilson_into(in_ptr, out_ptr, len, period, volume_factor);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @param {number} v_factor_start
 * @param {number} v_factor_end
 * @param {number} v_factor_step
 * @returns {number}
 */
module.exports.tilson_batch_into = function(in_ptr, out_ptr, len, period_start, period_end, period_step, v_factor_start, v_factor_end, v_factor_step) {
    const ret = wasm.tilson_batch_into(in_ptr, out_ptr, len, period_start, period_end, period_step, v_factor_start, v_factor_end, v_factor_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {number} period
 * @param {number} mult
 * @param {number} max_lookback
 * @param {string} direction
 * @returns {Float64Array}
 */
module.exports.safezonestop_js = function(high, low, period, mult, max_lookback, direction) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passStringToWasm0(direction, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.safezonestop_js(ptr0, len0, ptr1, len1, period, mult, max_lookback, ptr2, len2);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v4 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v4;
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 * @param {number} mult
 * @param {number} max_lookback
 * @param {string} direction
 */
module.exports.safezonestop_into = function(high_ptr, low_ptr, out_ptr, len, period, mult, max_lookback, direction) {
    const ptr0 = passStringToWasm0(direction, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.safezonestop_into(high_ptr, low_ptr, out_ptr, len, period, mult, max_lookback, ptr0, len0);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.safezonestop_alloc = function(len) {
    const ret = wasm.safezonestop_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.safezonestop_free = function(ptr, len) {
    wasm.safezonestop_free(ptr, len);
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {any} config
 * @returns {any}
 */
module.exports.safezonestop_batch = function(high, low, config) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.safezonestop_batch(ptr0, len0, ptr1, len1, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @param {number} mult_start
 * @param {number} mult_end
 * @param {number} mult_step
 * @param {number} max_lookback_start
 * @param {number} max_lookback_end
 * @param {number} max_lookback_step
 * @param {string} direction
 * @returns {number}
 */
module.exports.safezonestop_batch_into = function(high_ptr, low_ptr, out_ptr, len, period_start, period_end, period_step, mult_start, mult_end, mult_step, max_lookback_start, max_lookback_end, max_lookback_step, direction) {
    const ptr0 = passStringToWasm0(direction, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.safezonestop_batch_into(high_ptr, low_ptr, out_ptr, len, period_start, period_end, period_step, mult_start, mult_end, mult_step, max_lookback_start, max_lookback_end, max_lookback_step, ptr0, len0);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} close
 * @param {number} period
 * @param {number} scalar
 * @param {number} drift
 * @returns {Float64Array}
 */
module.exports.chop_js = function(high, low, close, period, scalar, drift) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.chop_js(ptr0, len0, ptr1, len1, ptr2, len2, period, scalar, drift);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v4 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v4;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.chop_alloc = function(len) {
    const ret = wasm.chop_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.chop_free = function(ptr, len) {
    wasm.chop_free(ptr, len);
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} close_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 * @param {number} scalar
 * @param {number} drift
 */
module.exports.chop_into = function(high_ptr, low_ptr, close_ptr, out_ptr, len, period, scalar, drift) {
    const ret = wasm.chop_into(high_ptr, low_ptr, close_ptr, out_ptr, len, period, scalar, drift);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} close
 * @param {any} config
 * @returns {any}
 */
module.exports.chop_batch = function(high, low, close, config) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.chop_batch(ptr0, len0, ptr1, len1, ptr2, len2, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} close_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @param {number} scalar_start
 * @param {number} scalar_end
 * @param {number} scalar_step
 * @param {number} drift_start
 * @param {number} drift_end
 * @param {number} drift_step
 * @returns {number}
 */
module.exports.chop_batch_into = function(high_ptr, low_ptr, close_ptr, out_ptr, len, period_start, period_end, period_step, scalar_start, scalar_end, scalar_step, drift_start, drift_end, drift_step) {
    const ret = wasm.chop_batch_into(high_ptr, low_ptr, close_ptr, out_ptr, len, period_start, period_end, period_step, scalar_start, scalar_end, scalar_step, drift_start, drift_end, drift_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.sqwma_js = function(data, period) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.sqwma_js(ptr0, len0, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {Float64Array} data
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {Float64Array}
 */
module.exports.sqwma_batch_js = function(data, period_start, period_end, period_step) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.sqwma_batch_js(ptr0, len0, period_start, period_end, period_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {Float64Array}
 */
module.exports.sqwma_batch_metadata_js = function(period_start, period_end, period_step) {
    const ret = wasm.sqwma_batch_metadata_js(period_start, period_end, period_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.sqwma_alloc = function(len) {
    const ret = wasm.sqwma_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.sqwma_free = function(ptr, len) {
    wasm.sqwma_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.sqwma_into = function(in_ptr, out_ptr, len, period) {
    const ret = wasm.sqwma_into(in_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.sqwma_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.sqwma_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.sqwma_batch_into = function(in_ptr, out_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.sqwma_batch_into(in_ptr, out_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {number} period
 * @returns {DonchianResult}
 */
module.exports.donchian_js = function(high, low, period) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.donchian_js(ptr0, len0, ptr1, len1, period);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return DonchianResult.__wrap(ret[0]);
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} upper_ptr
 * @param {number} middle_ptr
 * @param {number} lower_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.donchian_into = function(high_ptr, low_ptr, upper_ptr, middle_ptr, lower_ptr, len, period) {
    const ret = wasm.donchian_into(high_ptr, low_ptr, upper_ptr, middle_ptr, lower_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.donchian_alloc = function(len) {
    const ret = wasm.donchian_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.donchian_free = function(ptr, len) {
    wasm.donchian_free(ptr, len);
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {any} config
 * @returns {any}
 */
module.exports.donchian_batch = function(high, low, config) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.donchian_batch(ptr0, len0, ptr1, len1, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} upper_ptr
 * @param {number} middle_ptr
 * @param {number} lower_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.donchian_batch_into = function(high_ptr, low_ptr, upper_ptr, middle_ptr, lower_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.donchian_batch_into(high_ptr, low_ptr, upper_ptr, middle_ptr, lower_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} close
 * @param {number} timeperiod1
 * @param {number} timeperiod2
 * @param {number} timeperiod3
 * @returns {Float64Array}
 */
module.exports.ultosc_js = function(high, low, close, timeperiod1, timeperiod2, timeperiod3) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.ultosc_js(ptr0, len0, ptr1, len1, ptr2, len2, timeperiod1, timeperiod2, timeperiod3);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v4 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v4;
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} close_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} timeperiod1
 * @param {number} timeperiod2
 * @param {number} timeperiod3
 */
module.exports.ultosc_into = function(high_ptr, low_ptr, close_ptr, out_ptr, len, timeperiod1, timeperiod2, timeperiod3) {
    const ret = wasm.ultosc_into(high_ptr, low_ptr, close_ptr, out_ptr, len, timeperiod1, timeperiod2, timeperiod3);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.ultosc_alloc = function(len) {
    const ret = wasm.ultosc_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.ultosc_free = function(ptr, len) {
    wasm.ultosc_free(ptr, len);
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} close
 * @param {any} config
 * @returns {any}
 */
module.exports.ultosc_batch = function(high, low, close, config) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.ultosc_batch(ptr0, len0, ptr1, len1, ptr2, len2, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {number} order
 * @returns {any}
 */
module.exports.minmax_js = function(high, low, order) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.minmax_js(ptr0, len0, ptr1, len1, order);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.minmax_alloc = function(len) {
    const ret = wasm.minmax_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.minmax_free = function(ptr, len) {
    wasm.minmax_free(ptr, len);
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} is_min_ptr
 * @param {number} is_max_ptr
 * @param {number} last_min_ptr
 * @param {number} last_max_ptr
 * @param {number} len
 * @param {number} order
 */
module.exports.minmax_into = function(high_ptr, low_ptr, is_min_ptr, is_max_ptr, last_min_ptr, last_max_ptr, len, order) {
    const ret = wasm.minmax_into(high_ptr, low_ptr, is_min_ptr, is_max_ptr, last_min_ptr, last_max_ptr, len, order);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {any} config
 * @returns {any}
 */
module.exports.minmax_batch = function(high, low, config) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.minmax_batch(ptr0, len0, ptr1, len1, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} is_min_ptr
 * @param {number} is_max_ptr
 * @param {number} last_min_ptr
 * @param {number} last_max_ptr
 * @param {number} len
 * @param {number} order_start
 * @param {number} order_end
 * @param {number} order_step
 * @returns {number}
 */
module.exports.minmax_batch_into = function(high_ptr, low_ptr, is_min_ptr, is_max_ptr, last_min_ptr, last_max_ptr, len, order_start, order_end, order_step) {
    const ret = wasm.minmax_batch_into(high_ptr, low_ptr, is_min_ptr, is_max_ptr, last_min_ptr, last_max_ptr, len, order_start, order_end, order_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number | null} [period]
 * @param {number | null} [threshold]
 * @returns {any}
 */
module.exports.correlation_cycle_js = function(data, period, threshold) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.correlation_cycle_js(ptr0, len0, isLikeNone(period) ? 0x100000001 : (period) >>> 0, !isLikeNone(threshold), isLikeNone(threshold) ? 0 : threshold);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {Float64Array} data
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @param {number} threshold_start
 * @param {number} threshold_end
 * @param {number} threshold_step
 * @returns {any}
 */
module.exports.correlation_cycle_batch_js = function(data, period_start, period_end, period_step, threshold_start, threshold_end, threshold_step) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.correlation_cycle_batch_js(ptr0, len0, period_start, period_end, period_step, threshold_start, threshold_end, threshold_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} data_len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @param {number} threshold_start
 * @param {number} threshold_end
 * @param {number} threshold_step
 * @returns {any}
 */
module.exports.correlation_cycle_batch_metadata_js = function(data_len, period_start, period_end, period_step, threshold_start, threshold_end, threshold_step) {
    const ret = wasm.correlation_cycle_batch_metadata_js(data_len, period_start, period_end, period_step, threshold_start, threshold_end, threshold_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.correlation_cycle_alloc = function(len) {
    const ret = wasm.correlation_cycle_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.correlation_cycle_free = function(ptr, len) {
    wasm.correlation_cycle_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} real_ptr
 * @param {number} imag_ptr
 * @param {number} angle_ptr
 * @param {number} state_ptr
 * @param {number} len
 * @param {number | null} [period]
 * @param {number | null} [threshold]
 */
module.exports.correlation_cycle_into = function(in_ptr, real_ptr, imag_ptr, angle_ptr, state_ptr, len, period, threshold) {
    const ret = wasm.correlation_cycle_into(in_ptr, real_ptr, imag_ptr, angle_ptr, state_ptr, len, isLikeNone(period) ? 0x100000001 : (period) >>> 0, !isLikeNone(threshold), isLikeNone(threshold) ? 0 : threshold);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @param {number} fast_period
 * @param {number} slow_period
 * @returns {Float64Array}
 */
module.exports.maaq_js = function(data, period, fast_period, slow_period) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.maaq_js(ptr0, len0, period, fast_period, slow_period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {Float64Array}
 */
module.exports.maaq_batch_js = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.maaq_batch_js(ptr0, len0, config);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.maaq_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.maaq_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @param {number} fast_period_start
 * @param {number} fast_period_end
 * @param {number} fast_period_step
 * @param {number} slow_period_start
 * @param {number} slow_period_end
 * @param {number} slow_period_step
 * @returns {Float64Array}
 */
module.exports.maaq_batch_metadata_js = function(period_start, period_end, period_step, fast_period_start, fast_period_end, fast_period_step, slow_period_start, slow_period_end, slow_period_step) {
    const ret = wasm.maaq_batch_metadata_js(period_start, period_end, period_step, fast_period_start, fast_period_end, fast_period_step, slow_period_start, slow_period_end, slow_period_step);
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.maaq_alloc = function(len) {
    const ret = wasm.maaq_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.maaq_free = function(ptr, len) {
    wasm.maaq_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 * @param {number} fast_period
 * @param {number} slow_period
 */
module.exports.maaq_into = function(in_ptr, out_ptr, len, period, fast_period, slow_period) {
    const ret = wasm.maaq_into(in_ptr, out_ptr, len, period, fast_period, slow_period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {any} config
 */
module.exports.maaq_batch_into = function(in_ptr, out_ptr, len, config) {
    const ret = wasm.maaq_batch_into(in_ptr, out_ptr, len, config);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * Compute the Trend Flex Filter (TrendFlex) of the input data.
 *
 * # Arguments
 * * `data` - Input data array
 * * `period` - Primary lookback period
 *
 * # Returns
 * Array of TrendFlex values, same length as input
 * @param {Float64Array} data
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.trendflex_js = function(data, period) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.trendflex_js(ptr0, len0, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * Compute TrendFlex for multiple period values in a single pass.
 *
 * # Arguments
 * * `data` - Input data array
 * * `period_start`, `period_end`, `period_step` - Period range parameters
 *
 * # Returns
 * Flattened array of values (row-major order)
 * @param {Float64Array} data
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {Float64Array}
 */
module.exports.trendflex_batch_js = function(data, period_start, period_end, period_step) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.trendflex_batch_js(ptr0, len0, period_start, period_end, period_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * Get metadata about batch computation.
 *
 * # Arguments
 * * Period range parameters (same as trendflex_batch_js)
 *
 * # Returns
 * Array containing period values
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {Float64Array}
 */
module.exports.trendflex_batch_metadata_js = function(period_start, period_end, period_step) {
    const ret = wasm.trendflex_batch_metadata_js(period_start, period_end, period_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.trendflex_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.trendflex_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.trendflex_alloc = function(len) {
    const ret = wasm.trendflex_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.trendflex_free = function(ptr, len) {
    wasm.trendflex_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.trendflex_into = function(in_ptr, out_ptr, len, period) {
    const ret = wasm.trendflex_into(in_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.trendflex_batch_into = function(in_ptr, out_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.trendflex_batch_into(in_ptr, out_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} close
 * @param {number} period
 * @returns {DiJsOutput}
 */
module.exports.di_js = function(high, low, close, period) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.di_js(ptr0, len0, ptr1, len1, ptr2, len2, period);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return DiJsOutput.__wrap(ret[0]);
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.di_alloc = function(len) {
    const ret = wasm.di_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.di_free = function(ptr, len) {
    wasm.di_free(ptr, len);
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} close_ptr
 * @param {number} plus_ptr
 * @param {number} minus_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.di_into = function(high_ptr, low_ptr, close_ptr, plus_ptr, minus_ptr, len, period) {
    const ret = wasm.di_into(high_ptr, low_ptr, close_ptr, plus_ptr, minus_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} close
 * @param {any} config
 * @returns {any}
 */
module.exports.di_batch = function(high, low, close, config) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.di_batch(ptr0, len0, ptr1, len1, ptr2, len2, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} close_ptr
 * @param {number} plus_ptr
 * @param {number} minus_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.di_batch_into = function(high_ptr, low_ptr, close_ptr, plus_ptr, minus_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.di_batch_into(high_ptr, low_ptr, close_ptr, plus_ptr, minus_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * Safe API - allocates and returns Vec<f64>
 * @param {Float64Array} data
 * @param {number} period
 * @param {number} scalar
 * @returns {Float64Array}
 */
module.exports.ui_js = function(data, period, scalar) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.ui_js(ptr0, len0, period, scalar);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * Fast API with aliasing detection - zero allocations unless aliased
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 * @param {number} scalar
 */
module.exports.ui_into = function(in_ptr, out_ptr, len, period, scalar) {
    const ret = wasm.ui_into(in_ptr, out_ptr, len, period, scalar);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * Memory allocation for WASM
 * @param {number} len
 * @returns {number}
 */
module.exports.ui_alloc = function(len) {
    const ret = wasm.ui_alloc(len);
    return ret >>> 0;
};

/**
 * Memory deallocation for WASM
 * @param {number} ptr
 * @param {number} len
 */
module.exports.ui_free = function(ptr, len) {
    wasm.ui_free(ptr, len);
};

/**
 * Batch processing API
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.ui_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.ui_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * Fast batch API with raw pointers
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @param {number} scalar_start
 * @param {number} scalar_end
 * @param {number} scalar_step
 * @returns {number}
 */
module.exports.ui_batch_into = function(in_ptr, out_ptr, len, period_start, period_end, period_step, scalar_start, scalar_end, scalar_step) {
    const ret = wasm.ui_batch_into(in_ptr, out_ptr, len, period_start, period_end, period_step, scalar_start, scalar_end, scalar_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

function getArrayU8FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint8ArrayMemory0().subarray(ptr / 1, ptr / 1 + len);
}
/**
 * @param {Float64Array} source
 * @param {Float64Array} close
 * @param {number} period
 * @returns {Uint8Array}
 */
module.exports.ttm_trend_js = function(source, close, period) {
    const ptr0 = passArrayF64ToWasm0(source, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.ttm_trend_js(ptr0, len0, ptr1, len1, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v3;
};

/**
 * @param {number} source_ptr
 * @param {number} close_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.ttm_trend_into = function(source_ptr, close_ptr, out_ptr, len, period) {
    const ret = wasm.ttm_trend_into(source_ptr, close_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.ttm_trend_alloc = function(len) {
    const ret = wasm.ttm_trend_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.ttm_trend_alloc_u8 = function(len) {
    const ret = wasm.ttm_trend_alloc_u8(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.ttm_trend_free = function(ptr, len) {
    wasm.ttm_trend_free(ptr, len);
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.ttm_trend_free_u8 = function(ptr, len) {
    wasm.ttm_trend_free_u8(ptr, len);
};

/**
 * @param {Float64Array} source
 * @param {Float64Array} close
 * @param {any} config
 * @returns {any}
 */
module.exports.ttm_trend_batch = function(source, close, config) {
    const ptr0 = passArrayF64ToWasm0(source, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.ttm_trend_batch(ptr0, len0, ptr1, len1, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * Compute the Triangular Moving Average (TRIMA) of the input data.
 *
 * # Arguments
 * * `data` - Input data array
 * * `period` - Window size for the moving average (must be > 3)
 *
 * # Returns
 * Array of TRIMA values, same length as input
 * @param {Float64Array} data
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.trima_js = function(data, period) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.trima_js(ptr0, len0, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.trima_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.trima_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * Compute TRIMA for multiple period values in a single pass.
 *
 * # Arguments
 * * `data` - Input data array
 * * `period_start`, `period_end`, `period_step` - Period range parameters
 *
 * # Returns
 * Flattened array of values (row-major order)
 * @param {Float64Array} data
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {Float64Array}
 */
module.exports.trima_batch_js = function(data, period_start, period_end, period_step) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.trima_batch_js(ptr0, len0, period_start, period_end, period_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * Get metadata about batch computation.
 *
 * # Arguments
 * * Period range parameters (same as trima_batch_js)
 *
 * # Returns
 * Array containing period values
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {Float64Array}
 */
module.exports.trima_batch_metadata_js = function(period_start, period_end, period_step) {
    const ret = wasm.trima_batch_metadata_js(period_start, period_end, period_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.trima_alloc = function(len) {
    const ret = wasm.trima_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.trima_free = function(ptr, len) {
    wasm.trima_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.trima_into = function(in_ptr, out_ptr, len, period) {
    const ret = wasm.trima_into(in_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.trima_batch_into = function(in_ptr, out_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.trima_batch_into(in_ptr, out_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} close
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.natr_js = function(high, low, close, period) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.natr_js(ptr0, len0, ptr1, len1, ptr2, len2, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v4 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v4;
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} close_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.natr_into = function(high_ptr, low_ptr, close_ptr, out_ptr, len, period) {
    const ret = wasm.natr_into(high_ptr, low_ptr, close_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.natr_alloc = function(len) {
    const ret = wasm.natr_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.natr_free = function(ptr, len) {
    wasm.natr_free(ptr, len);
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} close
 * @param {any} config
 * @returns {any}
 */
module.exports.natr_batch = function(high, low, close, config) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.natr_batch(ptr0, len0, ptr1, len1, ptr2, len2, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} close_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.natr_batch_into = function(high_ptr, low_ptr, close_ptr, out_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.natr_batch_into(high_ptr, low_ptr, close_ptr, out_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @param {number} phase
 * @param {number} power
 * @returns {Float64Array}
 */
module.exports.jma_js = function(data, period, phase, power) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.jma_js(ptr0, len0, period, phase, power);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.jma_alloc = function(len) {
    const ret = wasm.jma_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.jma_free = function(ptr, len) {
    wasm.jma_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 * @param {number} phase
 * @param {number} power
 */
module.exports.jma_into = function(in_ptr, out_ptr, len, period, phase, power) {
    const ret = wasm.jma_into(in_ptr, out_ptr, len, period, phase, power);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.jma_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.jma_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {Float64Array} data
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @param {number} phase_start
 * @param {number} phase_end
 * @param {number} phase_step
 * @param {number} power_start
 * @param {number} power_end
 * @param {number} power_step
 * @returns {Float64Array}
 */
module.exports.jma_batch_js = function(data, period_start, period_end, period_step, phase_start, phase_end, phase_step, power_start, power_end, power_step) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.jma_batch_js(ptr0, len0, period_start, period_end, period_step, phase_start, phase_end, phase_step, power_start, power_end, power_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @param {number} phase_start
 * @param {number} phase_end
 * @param {number} phase_step
 * @param {number} power_start
 * @param {number} power_end
 * @param {number} power_step
 * @returns {Float64Array}
 */
module.exports.jma_batch_metadata_js = function(period_start, period_end, period_step, phase_start, phase_end, phase_step, power_start, power_end, power_step) {
    const ret = wasm.jma_batch_metadata_js(period_start, period_end, period_step, phase_start, phase_end, phase_step, power_start, power_end, power_step);
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @param {number} phase_start
 * @param {number} phase_end
 * @param {number} phase_step
 * @param {number} power_start
 * @param {number} power_end
 * @param {number} power_step
 * @returns {number}
 */
module.exports.jma_batch_into = function(in_ptr, out_ptr, len, period_start, period_end, period_step, phase_start, phase_end, phase_step, power_start, power_end, power_step) {
    const ret = wasm.jma_batch_into(in_ptr, out_ptr, len, period_start, period_end, period_step, phase_start, phase_end, phase_step, power_start, power_end, power_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @param {number} poles
 * @returns {Float64Array}
 */
module.exports.gaussian_js = function(data, period, poles) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.gaussian_js(ptr0, len0, period, poles);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 * @param {number} poles
 */
module.exports.gaussian_into = function(in_ptr, out_ptr, len, period, poles) {
    const ret = wasm.gaussian_into(in_ptr, out_ptr, len, period, poles);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.gaussian_alloc = function(len) {
    const ret = wasm.gaussian_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.gaussian_free = function(ptr, len) {
    wasm.gaussian_free(ptr, len);
};

/**
 * @param {Float64Array} data
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @param {number} poles_start
 * @param {number} poles_end
 * @param {number} poles_step
 * @returns {Float64Array}
 */
module.exports.gaussian_batch_js = function(data, period_start, period_end, period_step, poles_start, poles_end, poles_step) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.gaussian_batch_js(ptr0, len0, period_start, period_end, period_step, poles_start, poles_end, poles_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @param {number} poles_start
 * @param {number} poles_end
 * @param {number} poles_step
 * @returns {Float64Array}
 */
module.exports.gaussian_batch_metadata_js = function(period_start, period_end, period_step, poles_start, poles_end, poles_step) {
    const ret = wasm.gaussian_batch_metadata_js(period_start, period_end, period_step, poles_start, poles_end, poles_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.gaussian_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.gaussian_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @param {number} poles_start
 * @param {number} poles_end
 * @param {number} poles_step
 * @returns {number}
 */
module.exports.gaussian_batch_into = function(in_ptr, out_ptr, len, period_start, period_end, period_step, poles_start, poles_end, poles_step) {
    const ret = wasm.gaussian_batch_into(in_ptr, out_ptr, len, period_start, period_end, period_step, poles_start, poles_end, poles_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.correl_hl_js = function(high, low, period) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.correl_hl_js(ptr0, len0, ptr1, len1, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.correl_hl_into = function(high_ptr, low_ptr, out_ptr, len, period) {
    const ret = wasm.correl_hl_into(high_ptr, low_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.correl_hl_alloc = function(len) {
    const ret = wasm.correl_hl_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.correl_hl_free = function(ptr, len) {
    wasm.correl_hl_free(ptr, len);
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {any} config
 * @returns {any}
 */
module.exports.correl_hl_batch = function(high, low, config) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.correl_hl_batch(ptr0, len0, ptr1, len1, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.correl_hl_batch_into = function(high_ptr, low_ptr, out_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.correl_hl_batch_into(high_ptr, low_ptr, out_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} long_period
 * @param {number} short_period
 * @returns {Float64Array}
 */
module.exports.tsi_js = function(data, long_period, short_period) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.tsi_js(ptr0, len0, long_period, short_period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} long_period
 * @param {number} short_period
 */
module.exports.tsi_into = function(in_ptr, out_ptr, len, long_period, short_period) {
    const ret = wasm.tsi_into(in_ptr, out_ptr, len, long_period, short_period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.tsi_alloc = function(len) {
    const ret = wasm.tsi_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.tsi_free = function(ptr, len) {
    wasm.tsi_free(ptr, len);
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.tsi_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.tsi_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} long_period_start
 * @param {number} long_period_end
 * @param {number} long_period_step
 * @param {number} short_period_start
 * @param {number} short_period_end
 * @param {number} short_period_step
 * @returns {number}
 */
module.exports.tsi_batch_into = function(in_ptr, out_ptr, len, long_period_start, long_period_end, long_period_step, short_period_start, short_period_end, short_period_step) {
    const ret = wasm.tsi_batch_into(in_ptr, out_ptr, len, long_period_start, long_period_end, long_period_step, short_period_start, short_period_end, short_period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} short_period
 * @param {number} long_period
 * @param {number} alpha
 * @returns {Float64Array}
 */
module.exports.vidya_js = function(data, short_period, long_period, alpha) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.vidya_js(ptr0, len0, short_period, long_period, alpha);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} short_period
 * @param {number} long_period
 * @param {number} alpha
 */
module.exports.vidya_into = function(in_ptr, out_ptr, len, short_period, long_period, alpha) {
    const ret = wasm.vidya_into(in_ptr, out_ptr, len, short_period, long_period, alpha);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.vidya_alloc = function(len) {
    const ret = wasm.vidya_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.vidya_free = function(ptr, len) {
    wasm.vidya_free(ptr, len);
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.vidya_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.vidya_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} short_period_start
 * @param {number} short_period_end
 * @param {number} short_period_step
 * @param {number} long_period_start
 * @param {number} long_period_end
 * @param {number} long_period_step
 * @param {number} alpha_start
 * @param {number} alpha_end
 * @param {number} alpha_step
 * @returns {number}
 */
module.exports.vidya_batch_into = function(in_ptr, out_ptr, len, short_period_start, short_period_end, short_period_step, long_period_start, long_period_end, long_period_step, alpha_start, alpha_end, alpha_step) {
    const ret = wasm.vidya_batch_into(in_ptr, out_ptr, len, short_period_start, short_period_end, short_period_step, long_period_start, long_period_end, long_period_step, alpha_start, alpha_end, alpha_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} rsi_period
 * @param {number} wma_period
 * @returns {Float64Array}
 */
module.exports.ift_rsi_js = function(data, rsi_period, wma_period) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.ift_rsi_js(ptr0, len0, rsi_period, wma_period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} rsi_period
 * @param {number} wma_period
 */
module.exports.ift_rsi_into = function(in_ptr, out_ptr, len, rsi_period, wma_period) {
    const ret = wasm.ift_rsi_into(in_ptr, out_ptr, len, rsi_period, wma_period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.ift_rsi_alloc = function(len) {
    const ret = wasm.ift_rsi_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.ift_rsi_free = function(ptr, len) {
    wasm.ift_rsi_free(ptr, len);
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.ift_rsi_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.ift_rsi_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @param {number} devtype
 * @returns {Float64Array}
 */
module.exports.deviation_js = function(data, period, devtype) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.deviation_js(ptr0, len0, period, devtype);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {DeviationBatchResult}
 */
module.exports.deviation_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.deviation_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return DeviationBatchResult.__wrap(ret[0]);
};

/**
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @param {number} devtype_start
 * @param {number} devtype_end
 * @param {number} devtype_step
 * @returns {Float64Array}
 */
module.exports.deviation_batch_metadata = function(period_start, period_end, period_step, devtype_start, devtype_end, devtype_step) {
    const ret = wasm.deviation_batch_metadata(period_start, period_end, period_step, devtype_start, devtype_end, devtype_step);
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.deviation_alloc = function(len) {
    const ret = wasm.deviation_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.deviation_free = function(ptr, len) {
    wasm.deviation_free(ptr, len);
};

/**
 * @param {number} data_ptr
 * @param {number} len
 * @param {number} period
 * @param {number} devtype
 * @param {number} out_ptr
 */
module.exports.deviation_into = function(data_ptr, len, period, devtype, out_ptr) {
    const ret = wasm.deviation_into(data_ptr, len, period, devtype, out_ptr);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @param {number} devtype_start
 * @param {number} devtype_end
 * @param {number} devtype_step
 * @returns {number}
 */
module.exports.deviation_batch_into = function(in_ptr, out_ptr, len, period_start, period_end, period_step, devtype_start, devtype_end, devtype_step) {
    const ret = wasm.deviation_batch_into(in_ptr, out_ptr, len, period_start, period_end, period_step, devtype_start, devtype_end, devtype_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} na
 * @param {number} nb
 * @param {number} nc
 * @returns {Float64Array}
 */
module.exports.hwma_js = function(data, na, nb, nc) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.hwma_js(ptr0, len0, na, nb, nc);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.hwma_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.hwma_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.hwma_alloc = function(len) {
    const ret = wasm.hwma_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.hwma_free = function(ptr, len) {
    wasm.hwma_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} na
 * @param {number} nb
 * @param {number} nc
 */
module.exports.hwma_into = function(in_ptr, out_ptr, len, na, nb, nc) {
    const ret = wasm.hwma_into(in_ptr, out_ptr, len, na, nb, nc);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} na_start
 * @param {number} na_end
 * @param {number} na_step
 * @param {number} nb_start
 * @param {number} nb_end
 * @param {number} nb_step
 * @param {number} nc_start
 * @param {number} nc_end
 * @param {number} nc_step
 * @returns {number}
 */
module.exports.hwma_batch_into = function(in_ptr, out_ptr, len, na_start, na_end, na_step, nb_start, nb_end, nb_step, nc_start, nc_end, nc_step) {
    const ret = wasm.hwma_batch_into(in_ptr, out_ptr, len, na_start, na_end, na_step, nb_start, nb_end, nb_step, nc_start, nc_end, nc_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} na_start
 * @param {number} na_end
 * @param {number} na_step
 * @param {number} nb_start
 * @param {number} nb_end
 * @param {number} nb_step
 * @param {number} nc_start
 * @param {number} nc_end
 * @param {number} nc_step
 * @returns {Float64Array}
 */
module.exports.hwma_batch_js = function(data, na_start, na_end, na_step, nb_start, nb_end, nb_step, nc_start, nc_end, nc_step) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.hwma_batch_js(ptr0, len0, na_start, na_end, na_step, nb_start, nb_end, nb_step, nc_start, nc_end, nc_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} na_start
 * @param {number} na_end
 * @param {number} na_step
 * @param {number} nb_start
 * @param {number} nb_end
 * @param {number} nb_step
 * @param {number} nc_start
 * @param {number} nc_end
 * @param {number} nc_step
 * @returns {Float64Array}
 */
module.exports.hwma_batch_metadata_js = function(na_start, na_end, na_step, nb_start, nb_end, nb_step, nc_start, nc_end, nc_step) {
    const ret = wasm.hwma_batch_metadata_js(na_start, na_end, na_step, nb_start, nb_end, nb_step, nc_start, nc_end, nc_step);
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.pwma_js = function(data, period) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.pwma_js(ptr0, len0, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {Float64Array} data
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {Float64Array}
 */
module.exports.pwma_batch_js = function(data, period_start, period_end, period_step) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.pwma_batch_js(ptr0, len0, period_start, period_end, period_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {Float64Array}
 */
module.exports.pwma_batch_metadata_js = function(period_start, period_end, period_step) {
    const ret = wasm.pwma_batch_metadata_js(period_start, period_end, period_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
};

/**
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @param {number} data_len
 * @returns {Uint32Array}
 */
module.exports.pwma_batch_rows_cols_js = function(period_start, period_end, period_step, data_len) {
    const ret = wasm.pwma_batch_rows_cols_js(period_start, period_end, period_step, data_len);
    var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v1;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.pwma_alloc = function(len) {
    const ret = wasm.pwma_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.pwma_free = function(ptr, len) {
    wasm.pwma_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.pwma_into = function(in_ptr, out_ptr, len, period) {
    const ret = wasm.pwma_into(in_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.pwma_batch_into = function(in_ptr, out_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.pwma_batch_into(in_ptr, out_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @param {number} predict
 * @param {number} bandwidth
 * @returns {Float64Array}
 */
module.exports.voss_js = function(data, period, predict, bandwidth) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.voss_js(ptr0, len0, period, predict, bandwidth);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} in_ptr
 * @param {number} voss_ptr
 * @param {number} filt_ptr
 * @param {number} len
 * @param {number} period
 * @param {number} predict
 * @param {number} bandwidth
 */
module.exports.voss_into = function(in_ptr, voss_ptr, filt_ptr, len, period, predict, bandwidth) {
    const ret = wasm.voss_into(in_ptr, voss_ptr, filt_ptr, len, period, predict, bandwidth);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.voss_alloc = function(len) {
    const ret = wasm.voss_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.voss_free = function(ptr, len) {
    wasm.voss_free(ptr, len);
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.voss_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.voss_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @param {number} predict_start
 * @param {number} predict_end
 * @param {number} predict_step
 * @param {number} bandwidth_start
 * @param {number} bandwidth_end
 * @param {number} bandwidth_step
 * @returns {Float64Array}
 */
module.exports.voss_batch_metadata_js = function(period_start, period_end, period_step, predict_start, predict_end, predict_step, bandwidth_start, bandwidth_end, bandwidth_step) {
    const ret = wasm.voss_batch_metadata_js(period_start, period_end, period_step, predict_start, predict_end, predict_step, bandwidth_start, bandwidth_end, bandwidth_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
};

/**
 * @param {number} in_ptr
 * @param {number} voss_ptr
 * @param {number} filt_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @param {number} predict_start
 * @param {number} predict_end
 * @param {number} predict_step
 * @param {number} bandwidth_start
 * @param {number} bandwidth_end
 * @param {number} bandwidth_step
 */
module.exports.voss_batch_into = function(in_ptr, voss_ptr, filt_ptr, len, period_start, period_end, period_step, predict_start, predict_end, predict_step, bandwidth_start, bandwidth_end, bandwidth_step) {
    const ret = wasm.voss_batch_into(in_ptr, voss_ptr, filt_ptr, len, period_start, period_end, period_step, predict_start, predict_end, predict_step, bandwidth_start, bandwidth_end, bandwidth_step);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.fwma_js = function(data, period) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.fwma_js(ptr0, len0, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {Float64Array} data
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {Float64Array}
 */
module.exports.fwma_batch_js = function(data, period_start, period_end, period_step) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.fwma_batch_js(ptr0, len0, period_start, period_end, period_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {Float64Array}
 */
module.exports.fwma_batch_metadata_js = function(period_start, period_end, period_step) {
    const ret = wasm.fwma_batch_metadata_js(period_start, period_end, period_step);
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.fwma_alloc = function(len) {
    const ret = wasm.fwma_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.fwma_free = function(ptr, len) {
    wasm.fwma_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.fwma_into = function(in_ptr, out_ptr, len, period) {
    const ret = wasm.fwma_into(in_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.fwma_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.fwma_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.fwma_batch_into = function(in_ptr, out_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.fwma_batch_into(in_ptr, out_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.nma_js = function(data, period) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.nma_js(ptr0, len0, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.nma_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.nma_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {Float64Array} data
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {Float64Array}
 */
module.exports.nma_batch_js = function(data, period_start, period_end, period_step) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.nma_batch_js(ptr0, len0, period_start, period_end, period_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {Float64Array}
 */
module.exports.nma_batch_metadata_js = function(period_start, period_end, period_step) {
    const ret = wasm.nma_batch_metadata_js(period_start, period_end, period_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
};

/**
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @param {number} data_len
 * @returns {Uint32Array}
 */
module.exports.nma_batch_rows_cols_js = function(period_start, period_end, period_step, data_len) {
    const ret = wasm.nma_batch_rows_cols_js(period_start, period_end, period_step, data_len);
    var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v1;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.nma_alloc = function(len) {
    const ret = wasm.nma_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.nma_free = function(ptr, len) {
    wasm.nma_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.nma_into = function(in_ptr, out_ptr, len, period) {
    const ret = wasm.nma_into(in_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.nma_batch_into = function(in_ptr, out_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.nma_batch_into(in_ptr, out_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * Safe WASM API for KVO calculation
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} close
 * @param {Float64Array} volume
 * @param {number} short_period
 * @param {number} long_period
 * @returns {Float64Array}
 */
module.exports.kvo_js = function(high, low, close, volume, short_period, long_period) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArrayF64ToWasm0(volume, wasm.__wbindgen_malloc);
    const len3 = WASM_VECTOR_LEN;
    const ret = wasm.kvo_js(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, short_period, long_period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v5 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v5;
};

/**
 * Fast WASM API for KVO with aliasing detection
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} close_ptr
 * @param {number} volume_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} short_period
 * @param {number} long_period
 */
module.exports.kvo_into = function(high_ptr, low_ptr, close_ptr, volume_ptr, out_ptr, len, short_period, long_period) {
    const ret = wasm.kvo_into(high_ptr, low_ptr, close_ptr, volume_ptr, out_ptr, len, short_period, long_period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * Allocate memory for WASM operations
 * @param {number} len
 * @returns {number}
 */
module.exports.kvo_alloc = function(len) {
    const ret = wasm.kvo_alloc(len);
    return ret >>> 0;
};

/**
 * Free memory allocated by kvo_alloc
 * @param {number} ptr
 * @param {number} len
 */
module.exports.kvo_free = function(ptr, len) {
    wasm.kvo_free(ptr, len);
};

/**
 * Safe batch API for KVO
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} close
 * @param {Float64Array} volume
 * @param {any} config
 * @returns {any}
 */
module.exports.kvo_batch = function(high, low, close, volume, config) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArrayF64ToWasm0(volume, wasm.__wbindgen_malloc);
    const len3 = WASM_VECTOR_LEN;
    const ret = wasm.kvo_batch(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * Fast batch API for KVO with raw pointers
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} close_ptr
 * @param {number} volume_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} short_period_start
 * @param {number} short_period_end
 * @param {number} short_period_step
 * @param {number} long_period_start
 * @param {number} long_period_end
 * @param {number} long_period_step
 * @returns {number}
 */
module.exports.kvo_batch_into = function(high_ptr, low_ptr, close_ptr, volume_ptr, out_ptr, len, short_period_start, short_period_end, short_period_step, long_period_start, long_period_end, long_period_step) {
    const ret = wasm.kvo_batch_into(high_ptr, low_ptr, close_ptr, volume_ptr, out_ptr, len, short_period_start, short_period_end, short_period_step, long_period_start, long_period_end, long_period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.sinwma_js = function(data, period) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.sinwma_js(ptr0, len0, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.sinwma_alloc = function(len) {
    const ret = wasm.sinwma_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.sinwma_free = function(ptr, len) {
    wasm.sinwma_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.sinwma_into = function(in_ptr, out_ptr, len, period) {
    const ret = wasm.sinwma_into(in_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.sinwma_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.sinwma_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {Float64Array} data
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {Float64Array}
 */
module.exports.sinwma_batch_js = function(data, period_start, period_end, period_step) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.sinwma_batch_js(ptr0, len0, period_start, period_end, period_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {Uint32Array}
 */
module.exports.sinwma_batch_metadata_js = function(period_start, period_end, period_step) {
    const ret = wasm.sinwma_batch_metadata_js(period_start, period_end, period_step);
    var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v1;
};

/**
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @param {number} data_len
 * @returns {Uint32Array}
 */
module.exports.sinwma_batch_rows_cols_js = function(period_start, period_end, period_step, data_len) {
    const ret = wasm.sinwma_batch_rows_cols_js(period_start, period_end, period_step, data_len);
    var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v1;
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @param {number} bandwidth
 * @returns {BandPassResult}
 */
module.exports.bandpass_js = function(data, period, bandwidth) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.bandpass_js(ptr0, len0, period, bandwidth);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return BandPassResult.__wrap(ret[0]);
};

/**
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @param {number} bandwidth_start
 * @param {number} bandwidth_end
 * @param {number} bandwidth_step
 * @returns {Float64Array}
 */
module.exports.bandpass_batch_metadata = function(period_start, period_end, period_step, bandwidth_start, bandwidth_end, bandwidth_step) {
    const ret = wasm.bandpass_batch_metadata(period_start, period_end, period_step, bandwidth_start, bandwidth_end, bandwidth_step);
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {BandPassBatchResult}
 */
module.exports.bandpass_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.bandpass_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return BandPassBatchResult.__wrap(ret[0]);
};

/**
 * @param {number} in_ptr
 * @param {number} bp_ptr
 * @param {number} bp_normalized_ptr
 * @param {number} signal_ptr
 * @param {number} trigger_ptr
 * @param {number} len
 * @param {number} period
 * @param {number} bandwidth
 */
module.exports.bandpass_into = function(in_ptr, bp_ptr, bp_normalized_ptr, signal_ptr, trigger_ptr, len, period, bandwidth) {
    const ret = wasm.bandpass_into(in_ptr, bp_ptr, bp_normalized_ptr, signal_ptr, trigger_ptr, len, period, bandwidth);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.bandpass_alloc = function(len) {
    const ret = wasm.bandpass_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.bandpass_free = function(ptr, len) {
    wasm.bandpass_free(ptr, len);
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {number} r
 * @param {number} s
 * @param {number} u
 * @returns {Float64Array}
 */
module.exports.dti_js = function(high, low, r, s, u) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.dti_js(ptr0, len0, ptr1, len1, r, s, u);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} r
 * @param {number} s
 * @param {number} u
 */
module.exports.dti_into = function(high_ptr, low_ptr, out_ptr, len, r, s, u) {
    const ret = wasm.dti_into(high_ptr, low_ptr, out_ptr, len, r, s, u);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.dti_alloc = function(len) {
    const ret = wasm.dti_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.dti_free = function(ptr, len) {
    wasm.dti_free(ptr, len);
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {any} config
 * @returns {any}
 */
module.exports.dti_batch = function(high, low, config) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.dti_batch(ptr0, len0, ptr1, len1, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} r_start
 * @param {number} r_end
 * @param {number} r_step
 * @param {number} s_start
 * @param {number} s_end
 * @param {number} s_step
 * @param {number} u_start
 * @param {number} u_end
 * @param {number} u_step
 * @returns {number}
 */
module.exports.dti_batch_into = function(high_ptr, low_ptr, out_ptr, len, r_start, r_end, r_step, s_start, s_end, s_step, u_start, u_end, u_step) {
    const ret = wasm.dti_batch_into(high_ptr, low_ptr, out_ptr, len, r_start, r_end, r_step, s_start, s_end, s_step, u_start, u_end, u_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number | null} [warmup_bars]
 * @param {number | null} [max_dc_period]
 * @returns {Float64Array}
 */
module.exports.ehlers_itrend_js = function(data, warmup_bars, max_dc_period) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.ehlers_itrend_js(ptr0, len0, isLikeNone(warmup_bars) ? 0x100000001 : (warmup_bars) >>> 0, isLikeNone(max_dc_period) ? 0x100000001 : (max_dc_period) >>> 0);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.ehlers_itrend_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.ehlers_itrend_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {Float64Array} data
 * @param {number} warmup_bars_start
 * @param {number} warmup_bars_end
 * @param {number} warmup_bars_step
 * @param {number} max_dc_period_start
 * @param {number} max_dc_period_end
 * @param {number} max_dc_period_step
 * @returns {Float64Array}
 */
module.exports.ehlers_itrend_batch_js = function(data, warmup_bars_start, warmup_bars_end, warmup_bars_step, max_dc_period_start, max_dc_period_end, max_dc_period_step) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.ehlers_itrend_batch_js(ptr0, len0, warmup_bars_start, warmup_bars_end, warmup_bars_step, max_dc_period_start, max_dc_period_end, max_dc_period_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} warmup_bars_start
 * @param {number} warmup_bars_end
 * @param {number} warmup_bars_step
 * @param {number} max_dc_period_start
 * @param {number} max_dc_period_end
 * @param {number} max_dc_period_step
 * @returns {Float64Array}
 */
module.exports.ehlers_itrend_batch_metadata_js = function(warmup_bars_start, warmup_bars_end, warmup_bars_step, max_dc_period_start, max_dc_period_end, max_dc_period_step) {
    const ret = wasm.ehlers_itrend_batch_metadata_js(warmup_bars_start, warmup_bars_end, warmup_bars_step, max_dc_period_start, max_dc_period_end, max_dc_period_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.ehlers_itrend_alloc = function(len) {
    const ret = wasm.ehlers_itrend_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.ehlers_itrend_free = function(ptr, len) {
    wasm.ehlers_itrend_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number | null} [warmup_bars]
 * @param {number | null} [max_dc_period]
 */
module.exports.ehlers_itrend_into = function(in_ptr, out_ptr, len, warmup_bars, max_dc_period) {
    const ret = wasm.ehlers_itrend_into(in_ptr, out_ptr, len, isLikeNone(warmup_bars) ? 0x100000001 : (warmup_bars) >>> 0, isLikeNone(max_dc_period) ? 0x100000001 : (max_dc_period) >>> 0);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} warmup_bars_start
 * @param {number} warmup_bars_end
 * @param {number} warmup_bars_step
 * @param {number} max_dc_period_start
 * @param {number} max_dc_period_end
 * @param {number} max_dc_period_step
 * @returns {number}
 */
module.exports.ehlers_itrend_batch_into = function(in_ptr, out_ptr, len, warmup_bars_start, warmup_bars_end, warmup_bars_step, max_dc_period_start, max_dc_period_end, max_dc_period_step) {
    const ret = wasm.ehlers_itrend_batch_into(in_ptr, out_ptr, len, warmup_bars_start, warmup_bars_end, warmup_bars_step, max_dc_period_start, max_dc_period_end, max_dc_period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {string} ma_type
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.ma = function(data, ma_type, period) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passStringToWasm0(ma_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.ma(ptr0, len0, ptr1, len1, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {number} length
 * @returns {Float64Array}
 */
module.exports.aroonosc_js = function(high, low, length) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.aroonosc_js(ptr0, len0, ptr1, len1, length);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {number} length_start
 * @param {number} length_end
 * @param {number} length_step
 * @returns {Float64Array}
 */
module.exports.aroonosc_batch_js = function(high, low, length_start, length_end, length_step) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.aroonosc_batch_js(ptr0, len0, ptr1, len1, length_start, length_end, length_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
};

/**
 * @param {number} length_start
 * @param {number} length_end
 * @param {number} length_step
 * @returns {Float64Array}
 */
module.exports.aroonosc_batch_metadata_js = function(length_start, length_end, length_step) {
    const ret = wasm.aroonosc_batch_metadata_js(length_start, length_end, length_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {any} config
 * @returns {any}
 */
module.exports.aroonosc_batch = function(high, low, config) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.aroonosc_batch(ptr0, len0, ptr1, len1, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.aroonosc_alloc = function(len) {
    const ret = wasm.aroonosc_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.aroonosc_free = function(ptr, len) {
    wasm.aroonosc_free(ptr, len);
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} length
 */
module.exports.aroonosc_into = function(high_ptr, low_ptr, out_ptr, len, length) {
    const ret = wasm.aroonosc_into(high_ptr, low_ptr, out_ptr, len, length);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} length_start
 * @param {number} length_end
 * @param {number} length_step
 * @returns {number}
 */
module.exports.aroonosc_batch_into = function(high_ptr, low_ptr, out_ptr, len, length_start, length_end, length_step) {
    const ret = wasm.aroonosc_batch_into(high_ptr, low_ptr, out_ptr, len, length_start, length_end, length_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} hp_period
 * @param {number} k
 * @returns {Float64Array}
 */
module.exports.decycler_js = function(data, hp_period, k) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.decycler_js(ptr0, len0, hp_period, k);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} hp_period
 * @param {number} k
 */
module.exports.decycler_into = function(in_ptr, out_ptr, len, hp_period, k) {
    const ret = wasm.decycler_into(in_ptr, out_ptr, len, hp_period, k);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.decycler_alloc = function(len) {
    const ret = wasm.decycler_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.decycler_free = function(ptr, len) {
    wasm.decycler_free(ptr, len);
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.decycler_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.decycler_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} hp_period_start
 * @param {number} hp_period_end
 * @param {number} hp_period_step
 * @param {number} k_start
 * @param {number} k_end
 * @param {number} k_step
 * @returns {number}
 */
module.exports.decycler_batch_into = function(in_ptr, out_ptr, len, hp_period_start, hp_period_end, hp_period_step, k_start, k_end, k_step) {
    const ret = wasm.decycler_batch_into(in_ptr, out_ptr, len, hp_period_start, hp_period_end, hp_period_step, k_start, k_end, k_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @param {number} scalar
 * @returns {Float64Array}
 */
module.exports.cfo_js = function(data, period, scalar) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.cfo_js(ptr0, len0, period, scalar);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {Float64Array} data
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @param {number} scalar_start
 * @param {number} scalar_end
 * @param {number} scalar_step
 * @returns {Float64Array}
 */
module.exports.cfo_batch_js = function(data, period_start, period_end, period_step, scalar_start, scalar_end, scalar_step) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.cfo_batch_js(ptr0, len0, period_start, period_end, period_step, scalar_start, scalar_end, scalar_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @param {number} scalar_start
 * @param {number} scalar_end
 * @param {number} scalar_step
 * @returns {Float64Array}
 */
module.exports.cfo_batch_metadata_js = function(period_start, period_end, period_step, scalar_start, scalar_end, scalar_step) {
    const ret = wasm.cfo_batch_metadata_js(period_start, period_end, period_step, scalar_start, scalar_end, scalar_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.cfo_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.cfo_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.cfo_alloc = function(len) {
    const ret = wasm.cfo_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.cfo_free = function(ptr, len) {
    wasm.cfo_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 * @param {number} scalar
 */
module.exports.cfo_into = function(in_ptr, out_ptr, len, period, scalar) {
    const ret = wasm.cfo_into(in_ptr, out_ptr, len, period, scalar);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @param {number} scalar_start
 * @param {number} scalar_end
 * @param {number} scalar_step
 * @returns {number}
 */
module.exports.cfo_batch_into = function(in_ptr, out_ptr, len, period_start, period_end, period_step, scalar_start, scalar_end, scalar_step) {
    const ret = wasm.cfo_batch_into(in_ptr, out_ptr, len, period_start, period_end, period_step, scalar_start, scalar_end, scalar_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.mass_js = function(high, low, period) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.mass_js(ptr0, len0, ptr1, len1, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.mass_into = function(high_ptr, low_ptr, out_ptr, len, period) {
    const ret = wasm.mass_into(high_ptr, low_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.mass_alloc = function(len) {
    const ret = wasm.mass_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.mass_free = function(ptr, len) {
    wasm.mass_free(ptr, len);
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {any} config
 * @returns {any}
 */
module.exports.mass_batch = function(high, low, config) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.mass_batch(ptr0, len0, ptr1, len1, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.mass_batch_into = function(high_ptr, low_ptr, out_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.mass_batch_into(high_ptr, low_ptr, out_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.trix_js = function(data, period) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.trix_js(ptr0, len0, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.trix_alloc = function(len) {
    const ret = wasm.trix_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.trix_free = function(ptr, len) {
    wasm.trix_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.trix_into = function(in_ptr, out_ptr, len, period) {
    const ret = wasm.trix_into(in_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.trix_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.trix_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.trix_batch_into = function(in_ptr, out_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.trix_batch_into(in_ptr, out_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} rsi_period
 * @param {number} stoch_period
 * @param {number} k
 * @param {number} d
 * @returns {Float64Array}
 */
module.exports.srsi_js = function(data, rsi_period, stoch_period, k, d) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.srsi_js(ptr0, len0, rsi_period, stoch_period, k, d);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.srsi_alloc = function(len) {
    const ret = wasm.srsi_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.srsi_free = function(ptr, len) {
    wasm.srsi_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} k_ptr
 * @param {number} d_ptr
 * @param {number} len
 * @param {number} rsi_period
 * @param {number} stoch_period
 * @param {number} k
 * @param {number} d
 */
module.exports.srsi_into = function(in_ptr, k_ptr, d_ptr, len, rsi_period, stoch_period, k, d) {
    const ret = wasm.srsi_into(in_ptr, k_ptr, d_ptr, len, rsi_period, stoch_period, k, d);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.srsi_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.srsi_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} in_ptr
 * @param {number} k_ptr
 * @param {number} d_ptr
 * @param {number} len
 * @param {number} rsi_period_start
 * @param {number} rsi_period_end
 * @param {number} rsi_period_step
 * @param {number} stoch_period_start
 * @param {number} stoch_period_end
 * @param {number} stoch_period_step
 * @param {number} k_start
 * @param {number} k_end
 * @param {number} k_step
 * @param {number} d_start
 * @param {number} d_end
 * @param {number} d_step
 * @returns {number}
 */
module.exports.srsi_batch_into = function(in_ptr, k_ptr, d_ptr, len, rsi_period_start, rsi_period_end, rsi_period_step, stoch_period_start, stoch_period_end, stoch_period_step, k_start, k_end, k_step, d_start, d_end, d_step) {
    const ret = wasm.srsi_batch_into(in_ptr, k_ptr, d_ptr, len, rsi_period_start, rsi_period_end, rsi_period_step, stoch_period_start, stoch_period_end, stoch_period_step, k_start, k_end, k_step, d_start, d_end, d_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.mean_ad_js = function(data, period) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.mean_ad_js(ptr0, len0, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.mean_ad_into = function(in_ptr, out_ptr, len, period) {
    const ret = wasm.mean_ad_into(in_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.mean_ad_alloc = function(len) {
    const ret = wasm.mean_ad_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.mean_ad_free = function(ptr, len) {
    wasm.mean_ad_free(ptr, len);
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.mean_ad_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.mean_ad_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.mean_ad_batch_into = function(in_ptr, out_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.mean_ad_batch_into(in_ptr, out_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.supersmoother_3_pole_js = function(data, period) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.supersmoother_3_pole_js(ptr0, len0, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.supersmoother_3_pole_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.supersmoother_3_pole_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {Float64Array} data
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {Float64Array}
 */
module.exports.supersmoother_3_pole_batch_js = function(data, period_start, period_end, period_step) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.supersmoother_3_pole_batch_js(ptr0, len0, period_start, period_end, period_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {Float64Array}
 */
module.exports.supersmoother_3_pole_batch_metadata_js = function(period_start, period_end, period_step) {
    const ret = wasm.supersmoother_3_pole_batch_metadata_js(period_start, period_end, period_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.supersmoother_3_pole_alloc = function(len) {
    const ret = wasm.supersmoother_3_pole_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.supersmoother_3_pole_free = function(ptr, len) {
    wasm.supersmoother_3_pole_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.supersmoother_3_pole_into = function(in_ptr, out_ptr, len, period) {
    const ret = wasm.supersmoother_3_pole_into(in_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.supersmoother_3_pole_batch_into = function(in_ptr, out_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.supersmoother_3_pole_batch_into(in_ptr, out_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.hma_js = function(data, period) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.hma_js(ptr0, len0, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.hma_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.hma_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {Float64Array} data
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {Float64Array}
 */
module.exports.hma_batch_js = function(data, period_start, period_end, period_step) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.hma_batch_js(ptr0, len0, period_start, period_end, period_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {Float64Array}
 */
module.exports.hma_batch_metadata_js = function(period_start, period_end, period_step) {
    const ret = wasm.hma_batch_metadata_js(period_start, period_end, period_step);
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.hma_alloc = function(len) {
    const ret = wasm.hma_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.hma_free = function(ptr, len) {
    wasm.hma_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.hma_into = function(in_ptr, out_ptr, len, period) {
    const ret = wasm.hma_into(in_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.hma_batch_into = function(in_ptr, out_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.hma_batch_into(in_ptr, out_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} close
 * @param {number} period
 * @param {number} factor
 * @returns {Float64Array}
 */
module.exports.supertrend_js = function(high, low, close, period, factor) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.supertrend_js(ptr0, len0, ptr1, len1, ptr2, len2, period, factor);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v4 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v4;
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} close_ptr
 * @param {number} trend_ptr
 * @param {number} changed_ptr
 * @param {number} len
 * @param {number} period
 * @param {number} factor
 */
module.exports.supertrend_into = function(high_ptr, low_ptr, close_ptr, trend_ptr, changed_ptr, len, period, factor) {
    const ret = wasm.supertrend_into(high_ptr, low_ptr, close_ptr, trend_ptr, changed_ptr, len, period, factor);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.supertrend_alloc = function(len) {
    const ret = wasm.supertrend_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.supertrend_free = function(ptr, len) {
    wasm.supertrend_free(ptr, len);
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} close
 * @param {any} config
 * @returns {any}
 */
module.exports.supertrend_batch = function(high, low, close, config) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.supertrend_batch(ptr0, len0, ptr1, len1, ptr2, len2, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.fisher_js = function(high, low, period) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.fisher_js(ptr0, len0, ptr1, len1, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} fisher_ptr
 * @param {number} signal_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.fisher_into = function(high_ptr, low_ptr, fisher_ptr, signal_ptr, len, period) {
    const ret = wasm.fisher_into(high_ptr, low_ptr, fisher_ptr, signal_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.fisher_alloc = function(len) {
    const ret = wasm.fisher_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.fisher_free = function(ptr, len) {
    wasm.fisher_free(ptr, len);
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {any} config
 * @returns {any}
 */
module.exports.fisher_batch = function(high, low, config) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.fisher_batch(ptr0, len0, ptr1, len1, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} fisher_ptr
 * @param {number} signal_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.fisher_batch_into = function(high_ptr, low_ptr, fisher_ptr, signal_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.fisher_batch_into(high_ptr, low_ptr, fisher_ptr, signal_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} close
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.adxr_js = function(high, low, close, period) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.adxr_js(ptr0, len0, ptr1, len1, ptr2, len2, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v4 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v4;
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} close
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {Float64Array}
 */
module.exports.adxr_batch_js = function(high, low, close, period_start, period_end, period_step) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.adxr_batch_js(ptr0, len0, ptr1, len1, ptr2, len2, period_start, period_end, period_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v4 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v4;
};

/**
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {Float64Array}
 */
module.exports.adxr_batch_metadata_js = function(period_start, period_end, period_step) {
    const ret = wasm.adxr_batch_metadata_js(period_start, period_end, period_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} close
 * @param {any} config
 * @returns {any}
 */
module.exports.adxr_batch = function(high, low, close, config) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.adxr_batch(ptr0, len0, ptr1, len1, ptr2, len2, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.adxr_alloc = function(len) {
    const ret = wasm.adxr_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.adxr_free = function(ptr, len) {
    wasm.adxr_free(ptr, len);
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} close_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.adxr_into = function(high_ptr, low_ptr, close_ptr, out_ptr, len, period) {
    const ret = wasm.adxr_into(high_ptr, low_ptr, close_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} close_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.adxr_batch_into = function(high_ptr, low_ptr, close_ptr, out_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.adxr_batch_into(high_ptr, low_ptr, close_ptr, out_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.tsf_js = function(data, period) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.tsf_js(ptr0, len0, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.tsf_alloc = function(len) {
    const ret = wasm.tsf_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.tsf_free = function(ptr, len) {
    wasm.tsf_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.tsf_into = function(in_ptr, out_ptr, len, period) {
    const ret = wasm.tsf_into(in_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.tsf_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.tsf_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.tsf_batch_into = function(in_ptr, out_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.tsf_batch_into(in_ptr, out_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @param {number} nbdev
 * @returns {Float64Array}
 */
module.exports.var_js = function(data, period, nbdev) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.var_js(ptr0, len0, period, nbdev);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.var_alloc = function(len) {
    const ret = wasm.var_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.var_free = function(ptr, len) {
    wasm.var_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 * @param {number} nbdev
 */
module.exports.var_into = function(in_ptr, out_ptr, len, period, nbdev) {
    const ret = wasm.var_into(in_ptr, out_ptr, len, period, nbdev);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.var_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.var_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @param {number} nbdev_start
 * @param {number} nbdev_end
 * @param {number} nbdev_step
 * @returns {number}
 */
module.exports.var_batch_into = function(in_ptr, out_ptr, len, period_start, period_end, period_step, nbdev_start, nbdev_end, nbdev_step) {
    const ret = wasm.var_batch_into(in_ptr, out_ptr, len, period_start, period_end, period_step, nbdev_start, nbdev_end, nbdev_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} fast_limit
 * @param {number} slow_limit
 * @returns {Float64Array}
 */
module.exports.mama_js = function(data, fast_limit, slow_limit) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.mama_js(ptr0, len0, fast_limit, slow_limit);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {Float64Array} data
 * @param {number} fast_limit_start
 * @param {number} fast_limit_end
 * @param {number} fast_limit_step
 * @param {number} slow_limit_start
 * @param {number} slow_limit_end
 * @param {number} slow_limit_step
 * @returns {Float64Array}
 */
module.exports.mama_batch_js = function(data, fast_limit_start, fast_limit_end, fast_limit_step, slow_limit_start, slow_limit_end, slow_limit_step) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.mama_batch_js(ptr0, len0, fast_limit_start, fast_limit_end, fast_limit_step, slow_limit_start, slow_limit_end, slow_limit_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} fast_limit_start
 * @param {number} fast_limit_end
 * @param {number} fast_limit_step
 * @param {number} slow_limit_start
 * @param {number} slow_limit_end
 * @param {number} slow_limit_step
 * @returns {Float64Array}
 */
module.exports.mama_batch_metadata_js = function(fast_limit_start, fast_limit_end, fast_limit_step, slow_limit_start, slow_limit_end, slow_limit_step) {
    const ret = wasm.mama_batch_metadata_js(fast_limit_start, fast_limit_end, fast_limit_step, slow_limit_start, slow_limit_end, slow_limit_step);
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
};

/**
 * @param {number} fast_limit_start
 * @param {number} fast_limit_end
 * @param {number} fast_limit_step
 * @param {number} slow_limit_start
 * @param {number} slow_limit_end
 * @param {number} slow_limit_step
 * @param {number} data_len
 * @returns {Uint32Array}
 */
module.exports.mama_batch_rows_cols_js = function(fast_limit_start, fast_limit_end, fast_limit_step, slow_limit_start, slow_limit_end, slow_limit_step, data_len) {
    const ret = wasm.mama_batch_rows_cols_js(fast_limit_start, fast_limit_end, fast_limit_step, slow_limit_start, slow_limit_end, slow_limit_step, data_len);
    var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v1;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.mama_alloc = function(len) {
    const ret = wasm.mama_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.mama_free = function(ptr, len) {
    wasm.mama_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_mama_ptr
 * @param {number} out_fama_ptr
 * @param {number} len
 * @param {number} fast_limit
 * @param {number} slow_limit
 */
module.exports.mama_into = function(in_ptr, out_mama_ptr, out_fama_ptr, len, fast_limit, slow_limit) {
    const ret = wasm.mama_into(in_ptr, out_mama_ptr, out_fama_ptr, len, fast_limit, slow_limit);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} in_ptr
 * @param {number} out_mama_ptr
 * @param {number} out_fama_ptr
 * @param {number} len
 * @param {number} fast_limit_start
 * @param {number} fast_limit_end
 * @param {number} fast_limit_step
 * @param {number} slow_limit_start
 * @param {number} slow_limit_end
 * @param {number} slow_limit_step
 * @returns {number}
 */
module.exports.mama_batch_into = function(in_ptr, out_mama_ptr, out_fama_ptr, len, fast_limit_start, fast_limit_end, fast_limit_step, slow_limit_start, slow_limit_end, slow_limit_step) {
    const ret = wasm.mama_batch_into(in_ptr, out_mama_ptr, out_fama_ptr, len, fast_limit_start, fast_limit_end, fast_limit_step, slow_limit_start, slow_limit_end, slow_limit_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.tema_js = function(data, period) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.tema_js(ptr0, len0, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.tema_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.tema_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {Float64Array} data
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {Float64Array}
 */
module.exports.tema_batch_js = function(data, period_start, period_end, period_step) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.tema_batch_js(ptr0, len0, period_start, period_end, period_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.tema_alloc = function(len) {
    const ret = wasm.tema_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.tema_free = function(ptr, len) {
    wasm.tema_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.tema_into = function(in_ptr, out_ptr, len, period) {
    const ret = wasm.tema_into(in_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.tema_batch_into = function(in_ptr, out_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.tema_batch_into(in_ptr, out_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} close
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.dx_js = function(high, low, close, period) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.dx_js(ptr0, len0, ptr1, len1, ptr2, len2, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v4 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v4;
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} close_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.dx_into = function(high_ptr, low_ptr, close_ptr, out_ptr, len, period) {
    const ret = wasm.dx_into(high_ptr, low_ptr, close_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.dx_alloc = function(len) {
    const ret = wasm.dx_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.dx_free = function(ptr, len) {
    wasm.dx_free(ptr, len);
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} close
 * @param {any} config
 * @returns {any}
 */
module.exports.dx_batch = function(high, low, close, config) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.dx_batch(ptr0, len0, ptr1, len1, ptr2, len2, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} close_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 */
module.exports.dx_batch_into = function(high_ptr, low_ptr, close_ptr, out_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.dx_batch_into(high_ptr, low_ptr, close_ptr, out_ptr, len, period_start, period_end, period_step);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} close
 * @param {number} length
 * @returns {Float64Array}
 */
module.exports.atr = function(high, low, close, length) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.atr(ptr0, len0, ptr1, len1, ptr2, len2, length);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v4 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v4;
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} close
 * @param {number} length_start
 * @param {number} length_end
 * @param {number} length_step
 * @returns {Float64Array}
 */
module.exports.atrBatch = function(high, low, close, length_start, length_end, length_step) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.atrBatch(ptr0, len0, ptr1, len1, ptr2, len2, length_start, length_end, length_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v4 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v4;
};

/**
 * @param {number} length_start
 * @param {number} length_end
 * @param {number} length_step
 * @returns {Float64Array}
 */
module.exports.atrBatchMetadata = function(length_start, length_end, length_step) {
    const ret = wasm.atrBatchMetadata(length_start, length_end, length_step);
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
};

module.exports.atr_batch = function(high, low, close, config) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.atr_batch(ptr0, len0, ptr1, len1, ptr2, len2, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.atr_alloc = function(len) {
    const ret = wasm.atr_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.atr_free = function(ptr, len) {
    wasm.atr_free(ptr, len);
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} close_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} length
 */
module.exports.atr_into = function(high_ptr, low_ptr, close_ptr, out_ptr, len, length) {
    const ret = wasm.atr_into(high_ptr, low_ptr, close_ptr, out_ptr, len, length);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} close_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} length_start
 * @param {number} length_end
 * @param {number} length_step
 */
module.exports.atr_batch_into = function(high_ptr, low_ptr, close_ptr, out_ptr, len, length_start, length_end, length_step) {
    const ret = wasm.atr_batch_into(high_ptr, low_ptr, close_ptr, out_ptr, len, length_start, length_end, length_step);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @param {number} nbdev
 * @returns {Float64Array}
 */
module.exports.stddev_js = function(data, period, nbdev) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.stddev_js(ptr0, len0, period, nbdev);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 * @param {number} nbdev
 */
module.exports.stddev_into = function(in_ptr, out_ptr, len, period, nbdev) {
    const ret = wasm.stddev_into(in_ptr, out_ptr, len, period, nbdev);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.stddev_alloc = function(len) {
    const ret = wasm.stddev_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.stddev_free = function(ptr, len) {
    wasm.stddev_free(ptr, len);
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.stddev_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.stddev_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {any} config
 * @returns {any}
 */
module.exports.stddev_batch_into = function(in_ptr, out_ptr, len, config) {
    const ret = wasm.stddev_batch_into(in_ptr, out_ptr, len, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {number} short_period
 * @param {number} long_period
 * @returns {Float64Array}
 */
module.exports.ao_js = function(high, low, short_period, long_period) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.ao_js(ptr0, len0, ptr1, len1, short_period, long_period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {number} short_start
 * @param {number} short_end
 * @param {number} short_step
 * @param {number} long_start
 * @param {number} long_end
 * @param {number} long_step
 * @returns {Float64Array}
 */
module.exports.ao_batch_js = function(high, low, short_start, short_end, short_step, long_start, long_end, long_step) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.ao_batch_js(ptr0, len0, ptr1, len1, short_start, short_end, short_step, long_start, long_end, long_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
};

/**
 * @param {number} short_start
 * @param {number} short_end
 * @param {number} short_step
 * @param {number} long_start
 * @param {number} long_end
 * @param {number} long_step
 * @returns {Float64Array}
 */
module.exports.ao_batch_metadata_js = function(short_start, short_end, short_step, long_start, long_end, long_step) {
    const ret = wasm.ao_batch_metadata_js(short_start, short_end, short_step, long_start, long_end, long_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {any} config
 * @returns {any}
 */
module.exports.ao_batch = function(high, low, config) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.ao_batch(ptr0, len0, ptr1, len1, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.ao_alloc = function(len) {
    const ret = wasm.ao_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.ao_free = function(ptr, len) {
    wasm.ao_free(ptr, len);
};

/**
 * @param {number} in_high_ptr
 * @param {number} in_low_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} short_period
 * @param {number} long_period
 */
module.exports.ao_into = function(in_high_ptr, in_low_ptr, out_ptr, len, short_period, long_period) {
    const ret = wasm.ao_into(in_high_ptr, in_low_ptr, out_ptr, len, short_period, long_period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} in_high_ptr
 * @param {number} in_low_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} short_period_start
 * @param {number} short_period_end
 * @param {number} short_period_step
 * @param {number} long_period_start
 * @param {number} long_period_end
 * @param {number} long_period_step
 * @returns {number}
 */
module.exports.ao_batch_into = function(in_high_ptr, in_low_ptr, out_ptr, len, short_period_start, short_period_end, short_period_step, long_period_start, long_period_end, long_period_step) {
    const ret = wasm.ao_batch_into(in_high_ptr, in_low_ptr, out_ptr, len, short_period_start, short_period_end, short_period_step, long_period_start, long_period_end, long_period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} short_period
 * @param {number} long_period
 * @returns {Float64Array}
 */
module.exports.apo_js = function(data, short_period, long_period) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.apo_js(ptr0, len0, short_period, long_period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} short_period
 * @param {number} long_period
 */
module.exports.apo_into = function(in_ptr, out_ptr, len, short_period, long_period) {
    const ret = wasm.apo_into(in_ptr, out_ptr, len, short_period, long_period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.apo_alloc = function(len) {
    const ret = wasm.apo_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.apo_free = function(ptr, len) {
    wasm.apo_free(ptr, len);
};

/**
 * @param {Float64Array} data
 * @param {number} short_period_start
 * @param {number} short_period_end
 * @param {number} short_period_step
 * @param {number} long_period_start
 * @param {number} long_period_end
 * @param {number} long_period_step
 * @returns {Float64Array}
 */
module.exports.apo_batch_js = function(data, short_period_start, short_period_end, short_period_step, long_period_start, long_period_end, long_period_step) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.apo_batch_js(ptr0, len0, short_period_start, short_period_end, short_period_step, long_period_start, long_period_end, long_period_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} short_period_start
 * @param {number} short_period_end
 * @param {number} short_period_step
 * @param {number} long_period_start
 * @param {number} long_period_end
 * @param {number} long_period_step
 * @returns {Float64Array}
 */
module.exports.apo_batch_metadata_js = function(short_period_start, short_period_end, short_period_step, long_period_start, long_period_end, long_period_step) {
    const ret = wasm.apo_batch_metadata_js(short_period_start, short_period_end, short_period_step, long_period_start, long_period_end, long_period_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} short_period_start
 * @param {number} short_period_end
 * @param {number} short_period_step
 * @param {number} long_period_start
 * @param {number} long_period_end
 * @param {number} long_period_step
 * @returns {number}
 */
module.exports.apo_batch_into = function(in_ptr, out_ptr, len, short_period_start, short_period_end, short_period_step, long_period_start, long_period_end, long_period_step) {
    const ret = wasm.apo_batch_into(in_ptr, out_ptr, len, short_period_start, short_period_end, short_period_step, long_period_start, long_period_end, long_period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.apo_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.apo_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} close
 * @param {number} period
 * @returns {any}
 */
module.exports.vi_js = function(high, low, close, period) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.vi_js(ptr0, len0, ptr1, len1, ptr2, len2, period);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.vi_alloc = function(len) {
    const ret = wasm.vi_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.vi_free = function(ptr, len) {
    wasm.vi_free(ptr, len);
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} close_ptr
 * @param {number} plus_ptr
 * @param {number} minus_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.vi_into = function(high_ptr, low_ptr, close_ptr, plus_ptr, minus_ptr, len, period) {
    const ret = wasm.vi_into(high_ptr, low_ptr, close_ptr, plus_ptr, minus_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} close
 * @param {any} config
 * @returns {any}
 */
module.exports.vi_batch = function(high, low, close, config) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.vi_batch(ptr0, len0, ptr1, len1, ptr2, len2, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} close_ptr
 * @param {number} plus_ptr
 * @param {number} minus_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.vi_batch_into = function(high_ptr, low_ptr, close_ptr, plus_ptr, minus_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.vi_batch_into(high_ptr, low_ptr, close_ptr, plus_ptr, minus_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.cwma_js = function(data, period) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.cwma_js(ptr0, len0, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.cwma_alloc = function(len) {
    const ret = wasm.cwma_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.cwma_free = function(ptr, len) {
    wasm.cwma_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.cwma_into = function(in_ptr, out_ptr, len, period) {
    const ret = wasm.cwma_into(in_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} data
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {Float64Array}
 */
module.exports.cwma_batch_js = function(data, period_start, period_end, period_step) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.cwma_batch_js(ptr0, len0, period_start, period_end, period_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {Float64Array}
 */
module.exports.cwma_batch_metadata_js = function(period_start, period_end, period_step) {
    const ret = wasm.cwma_batch_metadata_js(period_start, period_end, period_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.cwma_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.cwma_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.cwma_batch_into = function(in_ptr, out_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.cwma_batch_into(in_ptr, out_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} close
 * @param {Float64Array} volume
 * @param {number} short_period
 * @param {number} long_period
 * @returns {Float64Array}
 */
module.exports.adosc_js = function(high, low, close, volume, short_period, long_period) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArrayF64ToWasm0(volume, wasm.__wbindgen_malloc);
    const len3 = WASM_VECTOR_LEN;
    const ret = wasm.adosc_js(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, short_period, long_period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v5 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v5;
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} close
 * @param {Float64Array} volume
 * @param {number} short_period_start
 * @param {number} short_period_end
 * @param {number} short_period_step
 * @param {number} long_period_start
 * @param {number} long_period_end
 * @param {number} long_period_step
 * @returns {Float64Array}
 */
module.exports.adosc_batch_js = function(high, low, close, volume, short_period_start, short_period_end, short_period_step, long_period_start, long_period_end, long_period_step) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArrayF64ToWasm0(volume, wasm.__wbindgen_malloc);
    const len3 = WASM_VECTOR_LEN;
    const ret = wasm.adosc_batch_js(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, short_period_start, short_period_end, short_period_step, long_period_start, long_period_end, long_period_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v5 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v5;
};

/**
 * @param {number} short_period_start
 * @param {number} short_period_end
 * @param {number} short_period_step
 * @param {number} long_period_start
 * @param {number} long_period_end
 * @param {number} long_period_step
 * @returns {Float64Array}
 */
module.exports.adosc_batch_metadata_js = function(short_period_start, short_period_end, short_period_step, long_period_start, long_period_end, long_period_step) {
    const ret = wasm.adosc_batch_metadata_js(short_period_start, short_period_end, short_period_step, long_period_start, long_period_end, long_period_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} close
 * @param {Float64Array} volume
 * @param {any} config
 * @returns {any}
 */
module.exports.adosc_batch = function(high, low, close, volume, config) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArrayF64ToWasm0(volume, wasm.__wbindgen_malloc);
    const len3 = WASM_VECTOR_LEN;
    const ret = wasm.adosc_batch(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.adosc_alloc = function(len) {
    const ret = wasm.adosc_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.adosc_free = function(ptr, len) {
    wasm.adosc_free(ptr, len);
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} close_ptr
 * @param {number} volume_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} short_period
 * @param {number} long_period
 */
module.exports.adosc_into = function(high_ptr, low_ptr, close_ptr, volume_ptr, out_ptr, len, short_period, long_period) {
    const ret = wasm.adosc_into(high_ptr, low_ptr, close_ptr, volume_ptr, out_ptr, len, short_period, long_period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} close_ptr
 * @param {number} volume_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} short_period_start
 * @param {number} short_period_end
 * @param {number} short_period_step
 * @param {number} long_period_start
 * @param {number} long_period_end
 * @param {number} long_period_step
 * @returns {number}
 */
module.exports.adosc_batch_into = function(high_ptr, low_ptr, close_ptr, volume_ptr, out_ptr, len, short_period_start, short_period_end, short_period_step, long_period_start, long_period_end, long_period_step) {
    const ret = wasm.adosc_batch_into(high_ptr, low_ptr, close_ptr, volume_ptr, out_ptr, len, short_period_start, short_period_end, short_period_step, long_period_start, long_period_end, long_period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.linearreg_intercept_js = function(data, period) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.linearreg_intercept_js(ptr0, len0, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.linearreg_intercept_alloc = function(len) {
    const ret = wasm.linearreg_intercept_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.linearreg_intercept_free = function(ptr, len) {
    wasm.linearreg_intercept_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.linearreg_intercept_into = function(in_ptr, out_ptr, len, period) {
    const ret = wasm.linearreg_intercept_into(in_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.linearreg_intercept_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.linearreg_intercept_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.linearreg_intercept_batch_into = function(in_ptr, out_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.linearreg_intercept_batch_into(in_ptr, out_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.cvi_js = function(high, low, period) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.cvi_js(ptr0, len0, ptr1, len1, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.cvi_alloc = function(len) {
    const ret = wasm.cvi_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.cvi_free = function(ptr, len) {
    wasm.cvi_free(ptr, len);
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.cvi_into = function(high_ptr, low_ptr, out_ptr, len, period) {
    const ret = wasm.cvi_into(high_ptr, low_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {any} config
 * @returns {any}
 */
module.exports.cvi_batch = function(high, low, config) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.cvi_batch(ptr0, len0, ptr1, len1, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.cvi_batch_into = function(high_ptr, low_ptr, out_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.cvi_batch_into(high_ptr, low_ptr, out_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.zlema_js = function(data, period) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.zlema_js(ptr0, len0, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {Float64Array} data
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {Float64Array}
 */
module.exports.zlema_batch_js = function(data, period_start, period_end, period_step) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.zlema_batch_js(ptr0, len0, period_start, period_end, period_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {Float64Array}
 */
module.exports.zlema_batch_metadata_js = function(period_start, period_end, period_step) {
    const ret = wasm.zlema_batch_metadata_js(period_start, period_end, period_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.zlema_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.zlema_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.zlema_alloc = function(len) {
    const ret = wasm.zlema_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.zlema_free = function(ptr, len) {
    wasm.zlema_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.zlema_into = function(in_ptr, out_ptr, len, period) {
    const ret = wasm.zlema_into(in_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.zlema_batch_into = function(in_ptr, out_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.zlema_batch_into(in_ptr, out_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {number} length
 * @returns {any}
 */
module.exports.aroon_js = function(high, low, length) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.aroon_js(ptr0, len0, ptr1, len1, length);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {number} length_start
 * @param {number} length_end
 * @param {number} length_step
 * @returns {any}
 */
module.exports.aroon_batch_js = function(high, low, length_start, length_end, length_step) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.aroon_batch_js(ptr0, len0, ptr1, len1, length_start, length_end, length_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} length_start
 * @param {number} length_end
 * @param {number} length_step
 * @returns {Float64Array}
 */
module.exports.aroon_batch_metadata_js = function(length_start, length_end, length_step) {
    const ret = wasm.aroon_batch_metadata_js(length_start, length_end, length_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {any} config
 * @returns {any}
 */
module.exports.aroon_batch = function(high, low, config) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.aroon_batch(ptr0, len0, ptr1, len1, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.aroon_alloc = function(len) {
    const ret = wasm.aroon_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.aroon_free = function(ptr, len) {
    wasm.aroon_free(ptr, len);
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} up_ptr
 * @param {number} down_ptr
 * @param {number} len
 * @param {number} length
 */
module.exports.aroon_into = function(high_ptr, low_ptr, up_ptr, down_ptr, len, length) {
    const ret = wasm.aroon_into(high_ptr, low_ptr, up_ptr, down_ptr, len, length);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} up_ptr
 * @param {number} down_ptr
 * @param {number} len
 * @param {number} length_start
 * @param {number} length_end
 * @param {number} length_step
 * @returns {number}
 */
module.exports.aroon_batch_into = function(high_ptr, low_ptr, up_ptr, down_ptr, len, length_start, length_end, length_step) {
    const ret = wasm.aroon_batch_into(high_ptr, low_ptr, up_ptr, down_ptr, len, length_start, length_end, length_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.rsx_js = function(data, period) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.rsx_js(ptr0, len0, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.rsx_into = function(in_ptr, out_ptr, len, period) {
    const ret = wasm.rsx_into(in_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.rsx_alloc = function(len) {
    const ret = wasm.rsx_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.rsx_free = function(ptr, len) {
    wasm.rsx_free(ptr, len);
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.rsx_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.rsx_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.rsx_batch_into = function(in_ptr, out_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.rsx_batch_into(in_ptr, out_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.wilders_js = function(data, period) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.wilders_js(ptr0, len0, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.wilders_alloc = function(len) {
    const ret = wasm.wilders_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.wilders_free = function(ptr, len) {
    wasm.wilders_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.wilders_into = function(in_ptr, out_ptr, len, period) {
    const ret = wasm.wilders_into(in_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.wilders_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.wilders_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {Float64Array} data
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {Float64Array}
 */
module.exports.wilders_batch_js = function(data, period_start, period_end, period_step) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.wilders_batch_js(ptr0, len0, period_start, period_end, period_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.wilders_batch_into = function(in_ptr, out_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.wilders_batch_into(in_ptr, out_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {Float64Array}
 */
module.exports.wilders_batch_metadata_js = function(period_start, period_end, period_step) {
    const ret = wasm.wilders_batch_metadata_js(period_start, period_end, period_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.medium_ad_js = function(data, period) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.medium_ad_js(ptr0, len0, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.medium_ad_alloc = function(len) {
    const ret = wasm.medium_ad_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.medium_ad_free = function(ptr, len) {
    wasm.medium_ad_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.medium_ad_into = function(in_ptr, out_ptr, len, period) {
    const ret = wasm.medium_ad_into(in_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.medium_ad_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.medium_ad_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.medium_ad_batch_into = function(in_ptr, out_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.medium_ad_batch_into(in_ptr, out_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} close
 * @param {Float64Array} open
 * @param {number} mode
 * @returns {Float64Array}
 */
module.exports.pivot_js = function(high, low, close, open, mode) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArrayF64ToWasm0(open, wasm.__wbindgen_malloc);
    const len3 = WASM_VECTOR_LEN;
    const ret = wasm.pivot_js(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, mode);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v5 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v5;
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} close_ptr
 * @param {number} open_ptr
 * @param {number} r4_ptr
 * @param {number} r3_ptr
 * @param {number} r2_ptr
 * @param {number} r1_ptr
 * @param {number} pp_ptr
 * @param {number} s1_ptr
 * @param {number} s2_ptr
 * @param {number} s3_ptr
 * @param {number} s4_ptr
 * @param {number} len
 * @param {number} mode
 */
module.exports.pivot_into = function(high_ptr, low_ptr, close_ptr, open_ptr, r4_ptr, r3_ptr, r2_ptr, r1_ptr, pp_ptr, s1_ptr, s2_ptr, s3_ptr, s4_ptr, len, mode) {
    const ret = wasm.pivot_into(high_ptr, low_ptr, close_ptr, open_ptr, r4_ptr, r3_ptr, r2_ptr, r1_ptr, pp_ptr, s1_ptr, s2_ptr, s3_ptr, s4_ptr, len, mode);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.pivot_alloc = function(len) {
    const ret = wasm.pivot_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.pivot_free = function(ptr, len) {
    wasm.pivot_free(ptr, len);
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} close
 * @param {Float64Array} open
 * @param {any} config
 * @returns {any}
 */
module.exports.pivot_batch = function(high, low, close, open, config) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArrayF64ToWasm0(open, wasm.__wbindgen_malloc);
    const len3 = WASM_VECTOR_LEN;
    const ret = wasm.pivot_batch(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.smma = function(data, period) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.smma(ptr0, len0, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.smma_batch_new = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.smma_batch_new(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {Float64Array} data
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {Float64Array}
 */
module.exports.smma_batch = function(data, period_start, period_end, period_step) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.smma_batch(ptr0, len0, period_start, period_end, period_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * Get metadata about the batch computation (periods used)
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {Uint32Array}
 */
module.exports.smma_batch_metadata = function(period_start, period_end, period_step) {
    const ret = wasm.smma_batch_metadata(period_start, period_end, period_step);
    var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v1;
};

/**
 * Get the dimensions of the batch output
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @param {number} data_len
 * @returns {Uint32Array}
 */
module.exports.smma_batch_rows_cols = function(period_start, period_end, period_step, data_len) {
    const ret = wasm.smma_batch_rows_cols(period_start, period_end, period_step, data_len);
    var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v1;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.smma_alloc = function(len) {
    const ret = wasm.smma_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.smma_free = function(ptr, len) {
    wasm.smma_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.smma_into = function(in_ptr, out_ptr, len, period) {
    const ret = wasm.smma_into(in_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.smma_batch_into = function(in_ptr, out_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.smma_batch_into(in_ptr, out_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} prices
 * @param {Float64Array} volumes
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.vwma_js = function(prices, volumes, period) {
    const ptr0 = passArrayF64ToWasm0(prices, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(volumes, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.vwma_js(ptr0, len0, ptr1, len1, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
};

/**
 * @param {Float64Array} prices
 * @param {Float64Array} volumes
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {Float64Array}
 */
module.exports.vwma_batch_js = function(prices, volumes, period_start, period_end, period_step) {
    const ptr0 = passArrayF64ToWasm0(prices, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(volumes, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.vwma_batch_js(ptr0, len0, ptr1, len1, period_start, period_end, period_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
};

/**
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {Float64Array}
 */
module.exports.vwma_batch_metadata_js = function(period_start, period_end, period_step) {
    const ret = wasm.vwma_batch_metadata_js(period_start, period_end, period_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.vwma_alloc = function(len) {
    const ret = wasm.vwma_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.vwma_free = function(ptr, len) {
    wasm.vwma_free(ptr, len);
};

/**
 * @param {number} price_ptr
 * @param {number} volume_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.vwma_into = function(price_ptr, volume_ptr, out_ptr, len, period) {
    const ret = wasm.vwma_into(price_ptr, volume_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} prices
 * @param {Float64Array} volumes
 * @param {any} config
 * @returns {any}
 */
module.exports.vwma_batch = function(prices, volumes, config) {
    const ptr0 = passArrayF64ToWasm0(prices, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(volumes, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.vwma_batch(ptr0, len0, ptr1, len1, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} price_ptr
 * @param {number} volume_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.vwma_batch_into = function(price_ptr, volume_ptr, out_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.vwma_batch_into(price_ptr, volume_ptr, out_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} close
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.adx_js = function(high, low, close, period) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.adx_js(ptr0, len0, ptr1, len1, ptr2, len2, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v4 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v4;
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} close
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {Float64Array}
 */
module.exports.adx_batch_js = function(high, low, close, period_start, period_end, period_step) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.adx_batch_js(ptr0, len0, ptr1, len1, ptr2, len2, period_start, period_end, period_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v4 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v4;
};

/**
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {Float64Array}
 */
module.exports.adx_batch_metadata_js = function(period_start, period_end, period_step) {
    const ret = wasm.adx_batch_metadata_js(period_start, period_end, period_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} close
 * @param {any} config
 * @returns {any}
 */
module.exports.adx_batch = function(high, low, close, config) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.adx_batch(ptr0, len0, ptr1, len1, ptr2, len2, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.adx_alloc = function(len) {
    const ret = wasm.adx_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.adx_free = function(ptr, len) {
    wasm.adx_free(ptr, len);
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} close_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.adx_into = function(high_ptr, low_ptr, close_ptr, out_ptr, len, period) {
    const ret = wasm.adx_into(high_ptr, low_ptr, close_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} close_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.adx_batch_into = function(high_ptr, low_ptr, close_ptr, out_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.adx_batch_into(high_ptr, low_ptr, close_ptr, out_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.midprice_js = function(high, low, period) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.midprice_js(ptr0, len0, ptr1, len1, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.midprice_alloc = function(len) {
    const ret = wasm.midprice_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.midprice_free = function(ptr, len) {
    wasm.midprice_free(ptr, len);
};

/**
 * @param {number} in_high_ptr
 * @param {number} in_low_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.midprice_into = function(in_high_ptr, in_low_ptr, out_ptr, len, period) {
    const ret = wasm.midprice_into(in_high_ptr, in_low_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {any} config
 * @returns {any}
 */
module.exports.midprice_batch = function(high, low, config) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.midprice_batch(ptr0, len0, ptr1, len1, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} in_high_ptr
 * @param {number} in_low_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.midprice_batch_into = function(in_high_ptr, in_low_ptr, out_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.midprice_batch_into(in_high_ptr, in_low_ptr, out_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.kurtosis_js = function(data, period) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.kurtosis_js(ptr0, len0, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.kurtosis_alloc = function(len) {
    const ret = wasm.kurtosis_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.kurtosis_free = function(ptr, len) {
    wasm.kurtosis_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.kurtosis_into = function(in_ptr, out_ptr, len, period) {
    const ret = wasm.kurtosis_into(in_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.kurtosis_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.kurtosis_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.kurtosis_batch_into = function(in_ptr, out_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.kurtosis_batch_into(in_ptr, out_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @param {number} k
 * @returns {Float64Array}
 */
module.exports.highpass_2_pole_js = function(data, period, k) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.highpass_2_pole_js(ptr0, len0, period, k);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.highpass_2_pole_alloc = function(len) {
    const ret = wasm.highpass_2_pole_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.highpass_2_pole_free = function(ptr, len) {
    wasm.highpass_2_pole_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 * @param {number} k
 */
module.exports.highpass_2_pole_into = function(in_ptr, out_ptr, len, period, k) {
    const ret = wasm.highpass_2_pole_into(in_ptr, out_ptr, len, period, k);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.highpass_2_pole_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.highpass_2_pole_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @param {number} k_start
 * @param {number} k_end
 * @param {number} k_step
 */
module.exports.highpass_2_pole_batch_into = function(in_ptr, out_ptr, len, period_start, period_end, period_step, k_start, k_end, k_step) {
    const ret = wasm.highpass_2_pole_batch_into(in_ptr, out_ptr, len, period_start, period_end, period_step, k_start, k_end, k_step);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} typical_price
 * @param {Float64Array} volume
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.mfi_js = function(typical_price, volume, period) {
    const ptr0 = passArrayF64ToWasm0(typical_price, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(volume, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.mfi_js(ptr0, len0, ptr1, len1, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
};

/**
 * @param {number} typical_price_ptr
 * @param {number} volume_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.mfi_into = function(typical_price_ptr, volume_ptr, out_ptr, len, period) {
    const ret = wasm.mfi_into(typical_price_ptr, volume_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.mfi_alloc = function(len) {
    const ret = wasm.mfi_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.mfi_free = function(ptr, len) {
    wasm.mfi_free(ptr, len);
};

/**
 * @param {Float64Array} typical_price
 * @param {Float64Array} volume
 * @param {any} config
 * @returns {any}
 */
module.exports.mfi_batch = function(typical_price, volume, config) {
    const ptr0 = passArrayF64ToWasm0(typical_price, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(volume, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.mfi_batch(ptr0, len0, ptr1, len1, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} typical_price_ptr
 * @param {number} volume_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.mfi_batch_into = function(typical_price_ptr, volume_ptr, out_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.mfi_batch_into(typical_price_ptr, volume_ptr, out_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.dpo_js = function(data, period) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.dpo_js(ptr0, len0, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.dpo_into = function(in_ptr, out_ptr, len, period) {
    const ret = wasm.dpo_into(in_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.dpo_alloc = function(len) {
    const ret = wasm.dpo_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.dpo_free = function(ptr, len) {
    wasm.dpo_free(ptr, len);
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.dpo_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.dpo_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.dpo_batch_into = function(in_ptr, out_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.dpo_batch_into(in_ptr, out_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.kama_js = function(data, period) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.kama_js(ptr0, len0, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.kama_alloc = function(len) {
    const ret = wasm.kama_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.kama_free = function(ptr, len) {
    wasm.kama_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.kama_into = function(in_ptr, out_ptr, len, period) {
    const ret = wasm.kama_into(in_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.kama_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.kama_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.kama_batch_into = function(in_ptr, out_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.kama_batch_into(in_ptr, out_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {Float64Array}
 */
module.exports.kama_batch_js = function(data, period_start, period_end, period_step) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.kama_batch_js(ptr0, len0, period_start, period_end, period_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {Float64Array}
 */
module.exports.kama_batch_metadata_js = function(period_start, period_end, period_step) {
    const ret = wasm.kama_batch_metadata_js(period_start, period_end, period_step);
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.jsa_js = function(data, period) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.jsa_js(ptr0, len0, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.jsa_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.jsa_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {Float64Array} data
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {Float64Array}
 */
module.exports.jsa_batch_simple = function(data, period_start, period_end, period_step) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.jsa_batch_simple(ptr0, len0, period_start, period_end, period_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.jsa_alloc = function(len) {
    const ret = wasm.jsa_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.jsa_free = function(ptr, len) {
    wasm.jsa_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.jsa_fast = function(in_ptr, out_ptr, len, period) {
    const ret = wasm.jsa_fast(in_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.jsa_batch_into = function(in_ptr, out_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.jsa_batch_into(in_ptr, out_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * Compute Simple Moving Average (SMA) for the given data
 * @param {Float64Array} data
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.sma = function(data, period) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.sma(ptr0, len0, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.sma_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.sma_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {Float64Array} data
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {Float64Array}
 */
module.exports.smaBatch = function(data, period_start, period_end, period_step) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.smaBatch(ptr0, len0, period_start, period_end, period_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {Uint32Array}
 */
module.exports.smaBatchMetadata = function(period_start, period_end, period_step) {
    const ret = wasm.smaBatchMetadata(period_start, period_end, period_step);
    var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v1;
};

/**
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @param {number} data_len
 * @returns {Uint32Array}
 */
module.exports.smaBatchRowsCols = function(period_start, period_end, period_step, data_len) {
    const ret = wasm.smaBatchRowsCols(period_start, period_end, period_step, data_len);
    var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v1;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.sma_alloc = function(len) {
    const ret = wasm.sma_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.sma_free = function(ptr, len) {
    wasm.sma_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.sma_into = function(in_ptr, out_ptr, len, period) {
    const ret = wasm.sma_into(in_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.sma_batch_into = function(in_ptr, out_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.sma_batch_into(in_ptr, out_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} price
 * @param {Float64Array} volume
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.efi_js = function(price, volume, period) {
    const ptr0 = passArrayF64ToWasm0(price, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(volume, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.efi_js(ptr0, len0, ptr1, len1, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.efi_alloc = function(len) {
    const ret = wasm.efi_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.efi_free = function(ptr, len) {
    wasm.efi_free(ptr, len);
};

/**
 * @param {number} in_price_ptr
 * @param {number} in_volume_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.efi_into = function(in_price_ptr, in_volume_ptr, out_ptr, len, period) {
    const ret = wasm.efi_into(in_price_ptr, in_volume_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} price
 * @param {Float64Array} volume
 * @param {any} config
 * @returns {any}
 */
module.exports.efi_batch = function(price, volume, config) {
    const ptr0 = passArrayF64ToWasm0(price, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(volume, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.efi_batch(ptr0, len0, ptr1, len1, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} in_price_ptr
 * @param {number} in_volume_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.efi_batch_into = function(in_price_ptr, in_volume_ptr, out_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.efi_batch_into(in_price_ptr, in_volume_ptr, out_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number | null} [period]
 * @returns {Float64Array}
 */
module.exports.cmo_js = function(data, period) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.cmo_js(ptr0, len0, isLikeNone(period) ? 0x100000001 : (period) >>> 0);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number | null} [period]
 */
module.exports.cmo_into = function(in_ptr, out_ptr, len, period) {
    const ret = wasm.cmo_into(in_ptr, out_ptr, len, isLikeNone(period) ? 0x100000001 : (period) >>> 0);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.cmo_alloc = function(len) {
    const ret = wasm.cmo_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.cmo_free = function(ptr, len) {
    wasm.cmo_free(ptr, len);
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.cmo_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.cmo_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.cmo_batch_into = function(in_ptr, out_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.cmo_batch_into(in_ptr, out_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.linreg_js = function(data, period) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.linreg_js(ptr0, len0, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.linreg_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.linreg_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.linreg_alloc = function(len) {
    const ret = wasm.linreg_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.linreg_free = function(ptr, len) {
    wasm.linreg_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.linreg_into = function(in_ptr, out_ptr, len, period) {
    const ret = wasm.linreg_into(in_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.linreg_batch_into = function(in_ptr, out_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.linreg_batch_into(in_ptr, out_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.cci_js = function(data, period) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.cci_js(ptr0, len0, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.cci_into = function(in_ptr, out_ptr, len, period) {
    const ret = wasm.cci_into(in_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.cci_alloc = function(len) {
    const ret = wasm.cci_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.cci_free = function(ptr, len) {
    wasm.cci_free(ptr, len);
};

/**
 * @param {Float64Array} data
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {Float64Array}
 */
module.exports.cci_batch_js = function(data, period_start, period_end, period_step) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.cci_batch_js(ptr0, len0, period_start, period_end, period_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {Float64Array}
 */
module.exports.cci_batch_metadata_js = function(period_start, period_end, period_step) {
    const ret = wasm.cci_batch_metadata_js(period_start, period_end, period_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.cci_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.cci_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.cci_batch_into = function(in_ptr, out_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.cci_batch_into(in_ptr, out_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.er_js = function(data, period) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.er_js(ptr0, len0, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.er_alloc = function(len) {
    const ret = wasm.er_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.er_free = function(ptr, len) {
    wasm.er_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.er_into = function(in_ptr, out_ptr, len, period) {
    const ret = wasm.er_into(in_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.er_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.er_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.er_batch_into = function(in_ptr, out_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.er_batch_into(in_ptr, out_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.mom_js = function(data, period) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.mom_js(ptr0, len0, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.mom_into = function(in_ptr, out_ptr, len, period) {
    const ret = wasm.mom_into(in_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.mom_alloc = function(len) {
    const ret = wasm.mom_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.mom_free = function(ptr, len) {
    wasm.mom_free(ptr, len);
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.mom_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.mom_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.mom_batch_into = function(in_ptr, out_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.mom_batch_into(in_ptr, out_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.cg_js = function(data, period) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.cg_js(ptr0, len0, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.cg_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.cg_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.cg_alloc = function(len) {
    const ret = wasm.cg_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.cg_free = function(ptr, len) {
    wasm.cg_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.cg_into = function(in_ptr, out_ptr, len, period) {
    const ret = wasm.cg_into(in_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.cg_batch_into = function(in_ptr, out_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.cg_batch_into(in_ptr, out_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.supersmoother_js = function(data, period) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.supersmoother_js(ptr0, len0, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.supersmoother_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.supersmoother_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {Float64Array} data
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {Float64Array}
 */
module.exports.supersmoother_batch_js = function(data, period_start, period_end, period_step) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.supersmoother_batch_js(ptr0, len0, period_start, period_end, period_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {Float64Array}
 */
module.exports.supersmoother_batch_metadata_js = function(period_start, period_end, period_step) {
    const ret = wasm.supersmoother_batch_metadata_js(period_start, period_end, period_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.supersmoother_alloc = function(len) {
    const ret = wasm.supersmoother_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.supersmoother_free = function(ptr, len) {
    wasm.supersmoother_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.supersmoother_into = function(in_ptr, out_ptr, len, period) {
    const ret = wasm.supersmoother_into(in_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.supersmoother_batch_into = function(in_ptr, out_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.supersmoother_batch_into(in_ptr, out_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {number} acceleration
 * @param {number} maximum
 * @returns {Float64Array}
 */
module.exports.sar_js = function(high, low, acceleration, maximum) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.sar_js(ptr0, len0, ptr1, len1, acceleration, maximum);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} acceleration
 * @param {number} maximum
 */
module.exports.sar_into = function(high_ptr, low_ptr, out_ptr, len, acceleration, maximum) {
    const ret = wasm.sar_into(high_ptr, low_ptr, out_ptr, len, acceleration, maximum);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.sar_alloc = function(len) {
    const ret = wasm.sar_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.sar_free = function(ptr, len) {
    wasm.sar_free(ptr, len);
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {any} config
 * @returns {any}
 */
module.exports.sar_batch = function(high, low, config) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.sar_batch(ptr0, len0, ptr1, len1, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.highpass_js = function(data, period) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.highpass_js(ptr0, len0, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.highpass_alloc = function(len) {
    const ret = wasm.highpass_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.highpass_free = function(ptr, len) {
    wasm.highpass_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.highpass_into = function(in_ptr, out_ptr, len, period) {
    const ret = wasm.highpass_into(in_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.highpass_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.highpass_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {Float64Array} data
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {Float64Array}
 */
module.exports.highpass_batch_js = function(data, period_start, period_end, period_step) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.highpass_batch_js(ptr0, len0, period_start, period_end, period_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.highpass_batch_into = function(in_ptr, out_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.highpass_batch_into(in_ptr, out_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {Float64Array}
 */
module.exports.highpass_batch_metadata_js = function(period_start, period_end, period_step) {
    const ret = wasm.highpass_batch_metadata_js(period_start, period_end, period_step);
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.midpoint_js = function(data, period) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.midpoint_js(ptr0, len0, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.midpoint_alloc = function(len) {
    const ret = wasm.midpoint_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.midpoint_free = function(ptr, len) {
    wasm.midpoint_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.midpoint_into = function(in_ptr, out_ptr, len, period) {
    const ret = wasm.midpoint_into(in_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.midpoint_batch_into = function(in_ptr, out_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.midpoint_batch_into(in_ptr, out_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.midpoint_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.midpoint_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.rocp_js = function(data, period) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.rocp_js(ptr0, len0, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.rocp_alloc = function(len) {
    const ret = wasm.rocp_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.rocp_free = function(ptr, len) {
    wasm.rocp_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.rocp_into = function(in_ptr, out_ptr, len, period) {
    const ret = wasm.rocp_into(in_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.rocp_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.rocp_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.rocp_batch_into = function(in_ptr, out_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.rocp_batch_into(in_ptr, out_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.rsi_js = function(data, period) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.rsi_js(ptr0, len0, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.rsi_alloc = function(len) {
    const ret = wasm.rsi_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.rsi_free = function(ptr, len) {
    wasm.rsi_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.rsi_into = function(in_ptr, out_ptr, len, period) {
    const ret = wasm.rsi_into(in_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.rsi_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.rsi_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.rocr_js = function(data, period) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.rocr_js(ptr0, len0, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.rocr_alloc = function(len) {
    const ret = wasm.rocr_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.rocr_free = function(ptr, len) {
    wasm.rocr_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.rocr_into = function(in_ptr, out_ptr, len, period) {
    const ret = wasm.rocr_into(in_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.rocr_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.rocr_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.rocr_batch_into = function(in_ptr, out_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.rocr_batch_into(in_ptr, out_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.reflex_js = function(data, period) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.reflex_js(ptr0, len0, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {Float64Array} data
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {Float64Array}
 */
module.exports.reflex_batch_js = function(data, period_start, period_end, period_step) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.reflex_batch_js(ptr0, len0, period_start, period_end, period_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {Uint32Array}
 */
module.exports.reflex_batch_metadata_js = function(period_start, period_end, period_step) {
    const ret = wasm.reflex_batch_metadata_js(period_start, period_end, period_step);
    var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v1;
};

/**
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @param {number} data_len
 * @returns {Uint32Array}
 */
module.exports.reflex_batch_rows_cols_js = function(period_start, period_end, period_step, data_len) {
    const ret = wasm.reflex_batch_rows_cols_js(period_start, period_end, period_step, data_len);
    var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v1;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.reflex_alloc = function(len) {
    const ret = wasm.reflex_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.reflex_free = function(ptr, len) {
    wasm.reflex_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.reflex_into = function(in_ptr, out_ptr, len, period) {
    const ret = wasm.reflex_into(in_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.edcf_js = function(data, period) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.edcf_js(ptr0, len0, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {Float64Array} data
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {Float64Array}
 */
module.exports.edcf_batch_js = function(data, period_start, period_end, period_step) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.edcf_batch_js(ptr0, len0, period_start, period_end, period_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {Float64Array}
 */
module.exports.edcf_batch_metadata_js = function(period_start, period_end, period_step) {
    const ret = wasm.edcf_batch_metadata_js(period_start, period_end, period_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.edcf_alloc = function(len) {
    const ret = wasm.edcf_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.edcf_free = function(ptr, len) {
    wasm.edcf_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.edcf_into = function(in_ptr, out_ptr, len, period) {
    const ret = wasm.edcf_into(in_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.edcf_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.edcf_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.edcf_batch_into = function(in_ptr, out_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.edcf_batch_into(in_ptr, out_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} open
 * @param {Float64Array} close
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.qstick_js = function(open, close, period) {
    const ptr0 = passArrayF64ToWasm0(open, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.qstick_js(ptr0, len0, ptr1, len1, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
};

/**
 * @param {number} open_ptr
 * @param {number} close_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.qstick_into = function(open_ptr, close_ptr, out_ptr, len, period) {
    const ret = wasm.qstick_into(open_ptr, close_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.qstick_alloc = function(len) {
    const ret = wasm.qstick_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.qstick_free = function(ptr, len) {
    wasm.qstick_free(ptr, len);
};

/**
 * @param {Float64Array} open
 * @param {Float64Array} close
 * @param {any} config
 * @returns {any}
 */
module.exports.qstick_batch = function(open, close, config) {
    const ptr0 = passArrayF64ToWasm0(open, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.qstick_batch(ptr0, len0, ptr1, len1, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} open_ptr
 * @param {number} close_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.qstick_batch_into = function(open_ptr, close_ptr, out_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.qstick_batch_into(open_ptr, close_ptr, out_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.ema_js = function(data, period) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.ema_js(ptr0, len0, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.ema_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.ema_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {Float64Array}
 */
module.exports.ema_batch_metadata_js = function(period_start, period_end, period_step) {
    const ret = wasm.ema_batch_metadata_js(period_start, period_end, period_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.ema_alloc = function(len) {
    const ret = wasm.ema_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.ema_free = function(ptr, len) {
    wasm.ema_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.ema_into = function(in_ptr, out_ptr, len, period) {
    const ret = wasm.ema_into(in_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.ema_batch_into = function(in_ptr, out_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.ema_batch_into(in_ptr, out_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.linearreg_slope_js = function(data, period) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.linearreg_slope_js(ptr0, len0, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.linearreg_slope_into = function(in_ptr, out_ptr, len, period) {
    const ret = wasm.linearreg_slope_into(in_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.linearreg_slope_alloc = function(len) {
    const ret = wasm.linearreg_slope_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.linearreg_slope_free = function(ptr, len) {
    wasm.linearreg_slope_free(ptr, len);
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.linearreg_slope_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.linearreg_slope_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {Float64Array} data
 * @param {number} short_period
 * @param {number} long_period
 * @returns {Float64Array}
 */
module.exports.vosc_js = function(data, short_period, long_period) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.vosc_js(ptr0, len0, short_period, long_period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} short_period
 * @param {number} long_period
 */
module.exports.vosc_into = function(in_ptr, out_ptr, len, short_period, long_period) {
    const ret = wasm.vosc_into(in_ptr, out_ptr, len, short_period, long_period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.vosc_alloc = function(len) {
    const ret = wasm.vosc_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.vosc_free = function(ptr, len) {
    wasm.vosc_free(ptr, len);
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.vosc_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.vosc_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.fosc_js = function(data, period) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.fosc_js(ptr0, len0, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.fosc_alloc = function(len) {
    const ret = wasm.fosc_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.fosc_free = function(ptr, len) {
    wasm.fosc_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.fosc_into = function(in_ptr, out_ptr, len, period) {
    const ret = wasm.fosc_into(in_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.fosc_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.fosc_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.fosc_batch_into = function(in_ptr, out_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.fosc_batch_into(in_ptr, out_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} close
 * @param {Float64Array} volume
 * @param {number} initial_value
 * @returns {Float64Array}
 */
module.exports.pvi_js = function(close, volume, initial_value) {
    const ptr0 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(volume, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.pvi_js(ptr0, len0, ptr1, len1, initial_value);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
};

/**
 * @param {number} close_ptr
 * @param {number} volume_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} initial_value
 */
module.exports.pvi_into = function(close_ptr, volume_ptr, out_ptr, len, initial_value) {
    const ret = wasm.pvi_into(close_ptr, volume_ptr, out_ptr, len, initial_value);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.pvi_alloc = function(len) {
    const ret = wasm.pvi_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.pvi_free = function(ptr, len) {
    wasm.pvi_free(ptr, len);
};

/**
 * @param {Float64Array} close
 * @param {Float64Array} volume
 * @param {any} config
 * @returns {any}
 */
module.exports.pvi_batch = function(close, volume, config) {
    const ptr0 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(volume, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.pvi_batch(ptr0, len0, ptr1, len1, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} close_ptr
 * @param {number} volume_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} initial_value_start
 * @param {number} initial_value_end
 * @param {number} initial_value_step
 * @returns {number}
 */
module.exports.pvi_batch_into = function(close_ptr, volume_ptr, out_ptr, len, initial_value_start, initial_value_end, initial_value_step) {
    const ret = wasm.pvi_batch_into(close_ptr, volume_ptr, out_ptr, len, initial_value_start, initial_value_end, initial_value_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} hp_period
 * @param {number} k
 * @returns {Float64Array}
 */
module.exports.dec_osc_js = function(data, hp_period, k) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.dec_osc_js(ptr0, len0, hp_period, k);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} hp_period
 * @param {number} k
 */
module.exports.dec_osc_into = function(in_ptr, out_ptr, len, hp_period, k) {
    const ret = wasm.dec_osc_into(in_ptr, out_ptr, len, hp_period, k);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.dec_osc_alloc = function(len) {
    const ret = wasm.dec_osc_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.dec_osc_free = function(ptr, len) {
    wasm.dec_osc_free(ptr, len);
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.dec_osc_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.dec_osc_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {Float64Array} data
 * @param {number} factor
 * @returns {Float64Array}
 */
module.exports.mwdx_js = function(data, factor) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.mwdx_js(ptr0, len0, factor);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.mwdx_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.mwdx_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {Float64Array} data
 * @param {number} factor_start
 * @param {number} factor_end
 * @param {number} factor_step
 * @returns {Float64Array}
 */
module.exports.mwdx_batch_js = function(data, factor_start, factor_end, factor_step) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.mwdx_batch_js(ptr0, len0, factor_start, factor_end, factor_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} factor_start
 * @param {number} factor_end
 * @param {number} factor_step
 * @returns {Float64Array}
 */
module.exports.mwdx_batch_metadata_js = function(factor_start, factor_end, factor_step) {
    const ret = wasm.mwdx_batch_metadata_js(factor_start, factor_end, factor_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
};

/**
 * @param {number} factor_start
 * @param {number} factor_end
 * @param {number} factor_step
 * @param {number} data_len
 * @returns {Uint32Array}
 */
module.exports.mwdx_batch_rows_cols_js = function(factor_start, factor_end, factor_step, data_len) {
    const ret = wasm.mwdx_batch_rows_cols_js(factor_start, factor_end, factor_step, data_len);
    var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v1;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.mwdx_alloc = function(len) {
    const ret = wasm.mwdx_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.mwdx_free = function(ptr, len) {
    wasm.mwdx_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} factor
 */
module.exports.mwdx_into = function(in_ptr, out_ptr, len, factor) {
    const ret = wasm.mwdx_into(in_ptr, out_ptr, len, factor);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} factor_start
 * @param {number} factor_end
 * @param {number} factor_step
 * @returns {number}
 */
module.exports.mwdx_batch_into = function(in_ptr, out_ptr, len, factor_start, factor_end, factor_step) {
    const ret = wasm.mwdx_batch_into(in_ptr, out_ptr, len, factor_start, factor_end, factor_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.dema_js = function(data, period) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.dema_js(ptr0, len0, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.dema_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.dema_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {Float64Array}
 */
module.exports.dema_batch_metadata_js = function(period_start, period_end, period_step) {
    const ret = wasm.dema_batch_metadata_js(period_start, period_end, period_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.dema_alloc = function(len) {
    const ret = wasm.dema_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.dema_free = function(ptr, len) {
    wasm.dema_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.dema_into = function(in_ptr, out_ptr, len, period) {
    const ret = wasm.dema_into(in_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.dema_batch_into = function(in_ptr, out_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.dema_batch_into(in_ptr, out_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.linearreg_angle_js = function(data, period) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.linearreg_angle_js(ptr0, len0, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.linearreg_angle_alloc = function(len) {
    const ret = wasm.linearreg_angle_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.linearreg_angle_free = function(ptr, len) {
    wasm.linearreg_angle_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.linearreg_angle_into = function(in_ptr, out_ptr, len, period) {
    const ret = wasm.linearreg_angle_into(in_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.linearreg_angle_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.linearreg_angle_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {number} alpha
 * @returns {Float64Array}
 */
module.exports.lrsi_js = function(high, low, alpha) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.lrsi_js(ptr0, len0, ptr1, len1, alpha);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {any} config
 * @returns {any}
 */
module.exports.lrsi_batch = function(high, low, config) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.lrsi_batch(ptr0, len0, ptr1, len1, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.lrsi_alloc = function(len) {
    const ret = wasm.lrsi_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.lrsi_free = function(ptr, len) {
    wasm.lrsi_free(ptr, len);
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} alpha
 */
module.exports.lrsi_into = function(high_ptr, low_ptr, out_ptr, len, alpha) {
    const ret = wasm.lrsi_into(high_ptr, low_ptr, out_ptr, len, alpha);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.wma_js = function(data, period) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.wma_js(ptr0, len0, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {Float64Array} data
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {Float64Array}
 */
module.exports.wma_batch_js = function(data, period_start, period_end, period_step) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.wma_batch_js(ptr0, len0, period_start, period_end, period_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {Float64Array}
 */
module.exports.wma_batch_metadata_js = function(period_start, period_end, period_step) {
    const ret = wasm.wma_batch_metadata_js(period_start, period_end, period_step);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.wma_alloc = function(len) {
    const ret = wasm.wma_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.wma_free = function(ptr, len) {
    wasm.wma_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.wma_into = function(in_ptr, out_ptr, len, period) {
    const ret = wasm.wma_into(in_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period_start
 * @param {number} period_end
 * @param {number} period_step
 * @returns {number}
 */
module.exports.wma_batch_into = function(in_ptr, out_ptr, len, period_start, period_end, period_step) {
    const ret = wasm.wma_batch_into(in_ptr, out_ptr, len, period_start, period_end, period_step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} data
 * @param {number} period
 * @returns {Float64Array}
 */
module.exports.roc_js = function(data, period) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.roc_js(ptr0, len0, period);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.roc_alloc = function(len) {
    const ret = wasm.roc_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.roc_free = function(ptr, len) {
    wasm.roc_free(ptr, len);
};

/**
 * @param {number} in_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @param {number} period
 */
module.exports.roc_into = function(in_ptr, out_ptr, len, period) {
    const ret = wasm.roc_into(in_ptr, out_ptr, len, period);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} data
 * @param {any} config
 * @returns {any}
 */
module.exports.roc_batch = function(data, config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.roc_batch(ptr0, len0, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {Float64Array} data
 * @returns {Float64Array}
 */
module.exports.pma_js = function(data) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.pma_js(ptr0, len0);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
};

/**
 * @param {number} in_ptr
 * @param {number} predict_ptr
 * @param {number} trigger_ptr
 * @param {number} len
 */
module.exports.pma_into = function(in_ptr, predict_ptr, trigger_ptr, len) {
    const ret = wasm.pma_into(in_ptr, predict_ptr, trigger_ptr, len);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.pma_alloc = function(len) {
    const ret = wasm.pma_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.pma_free = function(ptr, len) {
    wasm.pma_free(ptr, len);
};

/**
 * @param {Float64Array} data
 * @param {any} _config
 * @returns {any}
 */
module.exports.pma_batch = function(data, _config) {
    const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.pma_batch(ptr0, len0, _config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} in_ptr
 * @param {number} predict_ptr
 * @param {number} trigger_ptr
 * @param {number} len
 * @returns {number}
 */
module.exports.pma_batch_into = function(in_ptr, predict_ptr, trigger_ptr, len) {
    const ret = wasm.pma_batch_into(in_ptr, predict_ptr, trigger_ptr, len);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} volume
 * @returns {Float64Array}
 */
module.exports.marketefi_js = function(high, low, volume) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(volume, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.marketefi_js(ptr0, len0, ptr1, len1, ptr2, len2);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v4 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v4;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.marketefi_alloc = function(len) {
    const ret = wasm.marketefi_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.marketefi_free = function(ptr, len) {
    wasm.marketefi_free(ptr, len);
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} volume_ptr
 * @param {number} out_ptr
 * @param {number} len
 */
module.exports.marketefi_into = function(high_ptr, low_ptr, volume_ptr, out_ptr, len) {
    const ret = wasm.marketefi_into(high_ptr, low_ptr, volume_ptr, out_ptr, len);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} volume
 * @param {any} _config
 * @returns {any}
 */
module.exports.marketefi_batch = function(high, low, volume, _config) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(volume, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.marketefi_batch(ptr0, len0, ptr1, len1, ptr2, len2, _config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} volume_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @returns {number}
 */
module.exports.marketefi_batch_into = function(high_ptr, low_ptr, volume_ptr, out_ptr, len) {
    const ret = wasm.marketefi_batch_into(high_ptr, low_ptr, volume_ptr, out_ptr, len);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @returns {Float64Array}
 */
module.exports.acosc_js = function(high, low) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.acosc_js(ptr0, len0, ptr1, len1);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {any} _config
 * @returns {any}
 */
module.exports.acosc_batch = function(high, low, _config) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.acosc_batch(ptr0, len0, ptr1, len1, _config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @returns {Float64Array}
 */
module.exports.acosc_batch_js = function(high, low) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.acosc_batch_js(ptr0, len0, ptr1, len1);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
};

/**
 * @returns {Float64Array}
 */
module.exports.acosc_batch_metadata_js = function() {
    const ret = wasm.acosc_batch_metadata_js();
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} osc_ptr
 * @param {number} change_ptr
 * @param {number} len
 */
module.exports.acosc_into = function(high_ptr, low_ptr, osc_ptr, change_ptr, len) {
    const ret = wasm.acosc_into(high_ptr, low_ptr, osc_ptr, change_ptr, len);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.acosc_alloc = function(len) {
    const ret = wasm.acosc_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.acosc_free = function(ptr, len) {
    wasm.acosc_free(ptr, len);
};

/**
 * @param {Float64Array} open
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} close
 * @returns {Float64Array}
 */
module.exports.bop_js = function(open, high, low, close) {
    const ptr0 = passArrayF64ToWasm0(open, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len3 = WASM_VECTOR_LEN;
    const ret = wasm.bop_js(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v5 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v5;
};

/**
 * @param {number} open_ptr
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} close_ptr
 * @param {number} out_ptr
 * @param {number} len
 */
module.exports.bop_into = function(open_ptr, high_ptr, low_ptr, close_ptr, out_ptr, len) {
    const ret = wasm.bop_into(open_ptr, high_ptr, low_ptr, close_ptr, out_ptr, len);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.bop_alloc = function(len) {
    const ret = wasm.bop_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.bop_free = function(ptr, len) {
    wasm.bop_free(ptr, len);
};

/**
 * @param {Float64Array} open
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} close
 * @returns {Float64Array}
 */
module.exports.bop_batch_js = function(open, high, low, close) {
    const ptr0 = passArrayF64ToWasm0(open, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len3 = WASM_VECTOR_LEN;
    const ret = wasm.bop_batch_js(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v5 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v5;
};

/**
 * @param {number} open_ptr
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} close_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @returns {number}
 */
module.exports.bop_batch_into = function(open_ptr, high_ptr, low_ptr, close_ptr, out_ptr, len) {
    const ret = wasm.bop_batch_into(open_ptr, high_ptr, low_ptr, close_ptr, out_ptr, len);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @returns {Float64Array}
 */
module.exports.bop_batch_metadata_js = function() {
    const ret = wasm.bop_batch_metadata_js();
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
};

/**
 * @param {Float64Array} open
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} close
 * @param {any} _config
 * @returns {any}
 */
module.exports.bop_batch = function(open, high, low, close, _config) {
    const ptr0 = passArrayF64ToWasm0(open, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len3 = WASM_VECTOR_LEN;
    const ret = wasm.bop_batch(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, _config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} close
 * @param {Float64Array} volume
 * @returns {Float64Array}
 */
module.exports.ad_js = function(high, low, close, volume) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArrayF64ToWasm0(volume, wasm.__wbindgen_malloc);
    const len3 = WASM_VECTOR_LEN;
    const ret = wasm.ad_js(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v5 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v5;
};

/**
 * @param {Float64Array} highs_flat
 * @param {Float64Array} lows_flat
 * @param {Float64Array} closes_flat
 * @param {Float64Array} volumes_flat
 * @param {number} rows
 * @returns {Float64Array}
 */
module.exports.ad_batch_js = function(highs_flat, lows_flat, closes_flat, volumes_flat, rows) {
    const ptr0 = passArrayF64ToWasm0(highs_flat, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(lows_flat, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(closes_flat, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArrayF64ToWasm0(volumes_flat, wasm.__wbindgen_malloc);
    const len3 = WASM_VECTOR_LEN;
    const ret = wasm.ad_batch_js(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, rows);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v5 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v5;
};

/**
 * @param {number} rows
 * @param {number} cols
 * @returns {Float64Array}
 */
module.exports.ad_batch_metadata_js = function(rows, cols) {
    const ret = wasm.ad_batch_metadata_js(rows, cols);
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.ad_alloc = function(len) {
    const ret = wasm.ad_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.ad_free = function(ptr, len) {
    wasm.ad_free(ptr, len);
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} close_ptr
 * @param {number} volume_ptr
 * @param {number} out_ptr
 * @param {number} len
 */
module.exports.ad_into = function(high_ptr, low_ptr, close_ptr, volume_ptr, out_ptr, len) {
    const ret = wasm.ad_into(high_ptr, low_ptr, close_ptr, volume_ptr, out_ptr, len);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} close
 * @returns {Float64Array}
 */
module.exports.wclprice_js = function(high, low, close) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.wclprice_js(ptr0, len0, ptr1, len1, ptr2, len2);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v4 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v4;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.wclprice_alloc = function(len) {
    const ret = wasm.wclprice_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.wclprice_free = function(ptr, len) {
    wasm.wclprice_free(ptr, len);
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} close_ptr
 * @param {number} out_ptr
 * @param {number} len
 */
module.exports.wclprice_into = function(high_ptr, low_ptr, close_ptr, out_ptr, len) {
    const ret = wasm.wclprice_into(high_ptr, low_ptr, close_ptr, out_ptr, len);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} close
 * @param {any} _config
 * @returns {any}
 */
module.exports.wclprice_batch = function(high, low, close, _config) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.wclprice_batch(ptr0, len0, ptr1, len1, ptr2, len2, _config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} close_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @returns {number}
 */
module.exports.wclprice_batch_into = function(high_ptr, low_ptr, close_ptr, out_ptr, len) {
    const ret = wasm.wclprice_batch_into(high_ptr, low_ptr, close_ptr, out_ptr, len);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @returns {Float64Array}
 */
module.exports.medprice_js = function(high, low) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.medprice_js(ptr0, len0, ptr1, len1);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.medprice_alloc = function(len) {
    const ret = wasm.medprice_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.medprice_free = function(ptr, len) {
    wasm.medprice_free(ptr, len);
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} out_ptr
 * @param {number} len
 */
module.exports.medprice_into = function(high_ptr, low_ptr, out_ptr, len) {
    const ret = wasm.medprice_into(high_ptr, low_ptr, out_ptr, len);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {any} config
 * @returns {any}
 */
module.exports.medprice_batch = function(high, low, config) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.medprice_batch(ptr0, len0, ptr1, len1, config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} close
 * @param {Float64Array} volume
 * @returns {Float64Array}
 */
module.exports.emv_js = function(high, low, close, volume) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArrayF64ToWasm0(volume, wasm.__wbindgen_malloc);
    const len3 = WASM_VECTOR_LEN;
    const ret = wasm.emv_js(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v5 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v5;
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} close_ptr
 * @param {number} volume_ptr
 * @param {number} out_ptr
 * @param {number} len
 */
module.exports.emv_into = function(high_ptr, low_ptr, close_ptr, volume_ptr, out_ptr, len) {
    const ret = wasm.emv_into(high_ptr, low_ptr, close_ptr, volume_ptr, out_ptr, len);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.emv_alloc = function(len) {
    const ret = wasm.emv_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.emv_free = function(ptr, len) {
    wasm.emv_free(ptr, len);
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} close
 * @param {Float64Array} volume
 * @param {any} _config
 * @returns {any}
 */
module.exports.emv_batch = function(high, low, close, volume, _config) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArrayF64ToWasm0(volume, wasm.__wbindgen_malloc);
    const len3 = WASM_VECTOR_LEN;
    const ret = wasm.emv_batch(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, _config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} close_ptr
 * @param {number} volume_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @returns {number}
 */
module.exports.emv_batch_into = function(high_ptr, low_ptr, close_ptr, volume_ptr, out_ptr, len) {
    const ret = wasm.emv_batch_into(high_ptr, low_ptr, close_ptr, volume_ptr, out_ptr, len);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} price
 * @param {Float64Array} volume
 * @returns {Float64Array}
 */
module.exports.vpt_js = function(price, volume) {
    const ptr0 = passArrayF64ToWasm0(price, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(volume, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.vpt_js(ptr0, len0, ptr1, len1);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.vpt_alloc = function(len) {
    const ret = wasm.vpt_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.vpt_free = function(ptr, len) {
    wasm.vpt_free(ptr, len);
};

/**
 * @param {number} price_ptr
 * @param {number} volume_ptr
 * @param {number} out_ptr
 * @param {number} len
 */
module.exports.vpt_into = function(price_ptr, volume_ptr, out_ptr, len) {
    const ret = wasm.vpt_into(price_ptr, volume_ptr, out_ptr, len);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} price
 * @param {Float64Array} volume
 * @param {any} _config
 * @returns {any}
 */
module.exports.vpt_batch = function(price, volume, _config) {
    const ptr0 = passArrayF64ToWasm0(price, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(volume, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.vpt_batch(ptr0, len0, ptr1, len1, _config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {number} price_ptr
 * @param {number} volume_ptr
 * @param {number} out_ptr
 * @param {number} len
 * @returns {number}
 */
module.exports.vpt_batch_into = function(price_ptr, volume_ptr, out_ptr, len) {
    const ret = wasm.vpt_batch_into(price_ptr, volume_ptr, out_ptr, len);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0] >>> 0;
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} close
 * @returns {Float64Array}
 */
module.exports.wad_js = function(high, low, close) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.wad_js(ptr0, len0, ptr1, len1, ptr2, len2);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v4 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v4;
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.wad_alloc = function(len) {
    const ret = wasm.wad_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.wad_free = function(ptr, len) {
    wasm.wad_free(ptr, len);
};

/**
 * @param {number} high_ptr
 * @param {number} low_ptr
 * @param {number} close_ptr
 * @param {number} out_ptr
 * @param {number} len
 */
module.exports.wad_into = function(high_ptr, low_ptr, close_ptr, out_ptr, len) {
    const ret = wasm.wad_into(high_ptr, low_ptr, close_ptr, out_ptr, len);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {Float64Array} high
 * @param {Float64Array} low
 * @param {Float64Array} close
 * @param {any} _config
 * @returns {any}
 */
module.exports.wad_batch = function(high, low, close, _config) {
    const ptr0 = passArrayF64ToWasm0(high, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(low, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.wad_batch(ptr0, len0, ptr1, len1, ptr2, len2, _config);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {Float64Array} close
 * @param {Float64Array} volume
 * @returns {Float64Array}
 */
module.exports.obv_js = function(close, volume) {
    const ptr0 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(volume, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.obv_js(ptr0, len0, ptr1, len1);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
};

/**
 * @param {number} close_ptr
 * @param {number} volume_ptr
 * @param {number} out_ptr
 * @param {number} len
 */
module.exports.obv_into = function(close_ptr, volume_ptr, out_ptr, len) {
    const ret = wasm.obv_into(close_ptr, volume_ptr, out_ptr, len);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.obv_alloc = function(len) {
    const ret = wasm.obv_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.obv_free = function(ptr, len) {
    wasm.obv_free(ptr, len);
};

/**
 * @param {Float64Array} close
 * @param {Float64Array} volume
 * @returns {any}
 */
module.exports.obv_batch = function(close, volume) {
    const ptr0 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(volume, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.obv_batch(ptr0, len0, ptr1, len1);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
};

/**
 * @param {Float64Array} close
 * @param {Float64Array} volume
 * @returns {Float64Array}
 */
module.exports.nvi_js = function(close, volume) {
    const ptr0 = passArrayF64ToWasm0(close, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(volume, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.nvi_js(ptr0, len0, ptr1, len1);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
};

/**
 * @param {number} close_ptr
 * @param {number} volume_ptr
 * @param {number} out_ptr
 * @param {number} len
 */
module.exports.nvi_into = function(close_ptr, volume_ptr, out_ptr, len) {
    const ret = wasm.nvi_into(close_ptr, volume_ptr, out_ptr, len);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
};

/**
 * @param {number} len
 * @returns {number}
 */
module.exports.nvi_alloc = function(len) {
    const ret = wasm.nvi_alloc(len);
    return ret >>> 0;
};

/**
 * @param {number} ptr
 * @param {number} len
 */
module.exports.nvi_free = function(ptr, len) {
    wasm.nvi_free(ptr, len);
};

const AlmaContextFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_almacontext_free(ptr >>> 0, 1));

class AlmaContext {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        AlmaContextFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_almacontext_free(ptr, 0);
    }
    /**
     * @param {number} period
     * @param {number} offset
     * @param {number} sigma
     */
    constructor(period, offset, sigma) {
        const ret = wasm.almacontext_new(period, offset, sigma);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0] >>> 0;
        AlmaContextFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {number} in_ptr
     * @param {number} out_ptr
     * @param {number} len
     */
    update_into(in_ptr, out_ptr, len) {
        const ret = wasm.almacontext_update_into(this.__wbg_ptr, in_ptr, out_ptr, len);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * @returns {number}
     */
    get_warmup_period() {
        const ret = wasm.almacontext_get_warmup_period(this.__wbg_ptr);
        return ret >>> 0;
    }
}
module.exports.AlmaContext = AlmaContext;

const AtrContextFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_atrcontext_free(ptr >>> 0, 1));

class AtrContext {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        AtrContextFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_atrcontext_free(ptr, 0);
    }
    /**
     * @param {number} length
     */
    constructor(length) {
        const ret = wasm.atrcontext_new(length);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0] >>> 0;
        AtrContextFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {number} high
     * @param {number} low
     * @param {number} close
     * @returns {number | undefined}
     */
    update(high, low, close) {
        const ret = wasm.atrcontext_update(this.__wbg_ptr, high, low, close);
        return ret[0] === 0 ? undefined : ret[1];
    }
    reset() {
        const ret = wasm.atrcontext_reset(this.__wbg_ptr);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
}
module.exports.AtrContext = AtrContext;

const BandPassBatchResultFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_bandpassbatchresult_free(ptr >>> 0, 1));

class BandPassBatchResult {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(BandPassBatchResult.prototype);
        obj.__wbg_ptr = ptr;
        BandPassBatchResultFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        BandPassBatchResultFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_bandpassbatchresult_free(ptr, 0);
    }
    /**
     * @returns {Float64Array}
     */
    get values() {
        const ret = wasm.bandpassbatchresult_values(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @returns {number}
     */
    get combos() {
        const ret = wasm.bandpassbatchresult_combos(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get outputs() {
        const ret = wasm.bandpassbatchresult_outputs(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get cols() {
        const ret = wasm.bandpassbatchresult_cols(this.__wbg_ptr);
        return ret >>> 0;
    }
}
module.exports.BandPassBatchResult = BandPassBatchResult;

const BandPassResultFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_bandpassresult_free(ptr >>> 0, 1));

class BandPassResult {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(BandPassResult.prototype);
        obj.__wbg_ptr = ptr;
        BandPassResultFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        BandPassResultFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_bandpassresult_free(ptr, 0);
    }
    /**
     * @returns {Float64Array}
     */
    get values() {
        const ret = wasm.bandpassresult_values(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @returns {number}
     */
    get rows() {
        const ret = wasm.bandpassbatchresult_combos(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get cols() {
        const ret = wasm.bandpassbatchresult_outputs(this.__wbg_ptr);
        return ret >>> 0;
    }
}
module.exports.BandPassResult = BandPassResult;

const DeviationBatchResultFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_deviationbatchresult_free(ptr >>> 0, 1));

class DeviationBatchResult {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(DeviationBatchResult.prototype);
        obj.__wbg_ptr = ptr;
        DeviationBatchResultFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        DeviationBatchResultFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_deviationbatchresult_free(ptr, 0);
    }
    /**
     * @returns {Float64Array}
     */
    get values() {
        const ret = wasm.deviationbatchresult_values(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @returns {number}
     */
    get combos() {
        const ret = wasm.deviationbatchresult_combos(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get cols() {
        const ret = wasm.deviationbatchresult_cols(this.__wbg_ptr);
        return ret >>> 0;
    }
}
module.exports.DeviationBatchResult = DeviationBatchResult;

const DiJsOutputFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_dijsoutput_free(ptr >>> 0, 1));

class DiJsOutput {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(DiJsOutput.prototype);
        obj.__wbg_ptr = ptr;
        DiJsOutputFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        DiJsOutputFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_dijsoutput_free(ptr, 0);
    }
    /**
     * @returns {Float64Array}
     */
    get plus() {
        const ret = wasm.dijsoutput_plus(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @returns {Float64Array}
     */
    get minus() {
        const ret = wasm.dijsoutput_minus(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
}
module.exports.DiJsOutput = DiJsOutput;

const DonchianResultFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_donchianresult_free(ptr >>> 0, 1));

class DonchianResult {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(DonchianResult.prototype);
        obj.__wbg_ptr = ptr;
        DonchianResultFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        DonchianResultFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_donchianresult_free(ptr, 0);
    }
    /**
     * @returns {Float64Array}
     */
    get values() {
        const ret = wasm.donchianresult_values(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @returns {number}
     */
    get rows() {
        const ret = wasm.donchianresult_rows(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get cols() {
        const ret = wasm.donchianresult_cols(this.__wbg_ptr);
        return ret >>> 0;
    }
}
module.exports.DonchianResult = DonchianResult;

const EpmaContextFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_epmacontext_free(ptr >>> 0, 1));

class EpmaContext {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        EpmaContextFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_epmacontext_free(ptr, 0);
    }
    /**
     * @param {number} period
     * @param {number} offset
     */
    constructor(period, offset) {
        const ret = wasm.epmacontext_new(period, offset);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0] >>> 0;
        EpmaContextFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {Float64Array} data
     * @returns {Float64Array}
     */
    compute(data) {
        const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.epmacontext_compute(this.__wbg_ptr, ptr0, len0);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v2;
    }
}
module.exports.EpmaContext = EpmaContext;

const FisherContextFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_fishercontext_free(ptr >>> 0, 1));

class FisherContext {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        FisherContextFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_fishercontext_free(ptr, 0);
    }
    /**
     * @param {number} period
     */
    constructor(period) {
        const ret = wasm.fishercontext_new(period);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0] >>> 0;
        FisherContextFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {number} high_ptr
     * @param {number} low_ptr
     * @param {number} fisher_ptr
     * @param {number} signal_ptr
     * @param {number} len
     */
    update_into(high_ptr, low_ptr, fisher_ptr, signal_ptr, len) {
        const ret = wasm.fishercontext_update_into(this.__wbg_ptr, high_ptr, low_ptr, fisher_ptr, signal_ptr, len);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * @returns {number}
     */
    get_period() {
        const ret = wasm.fishercontext_get_period(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get_warmup_period() {
        const ret = wasm.fishercontext_get_warmup_period(this.__wbg_ptr);
        return ret >>> 0;
    }
}
module.exports.FisherContext = FisherContext;

const MacdResultFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_macdresult_free(ptr >>> 0, 1));

class MacdResult {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(MacdResult.prototype);
        obj.__wbg_ptr = ptr;
        MacdResultFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        MacdResultFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_macdresult_free(ptr, 0);
    }
    /**
     * @returns {Float64Array}
     */
    get values() {
        const ret = wasm.macdresult_values(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @returns {number}
     */
    get rows() {
        const ret = wasm.macdresult_rows(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get cols() {
        const ret = wasm.macdresult_cols(this.__wbg_ptr);
        return ret >>> 0;
    }
}
module.exports.MacdResult = MacdResult;

const MassStreamWasmFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_massstreamwasm_free(ptr >>> 0, 1));

class MassStreamWasm {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        MassStreamWasmFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_massstreamwasm_free(ptr, 0);
    }
    /**
     * @param {number} period
     */
    constructor(period) {
        const ret = wasm.massstreamwasm_new(period);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0] >>> 0;
        MassStreamWasmFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {number} high
     * @param {number} low
     * @returns {number | undefined}
     */
    update(high, low) {
        const ret = wasm.massstreamwasm_update(this.__wbg_ptr, high, low);
        return ret[0] === 0 ? undefined : ret[1];
    }
}
module.exports.MassStreamWasm = MassStreamWasm;

const PmaStreamWasmFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_pmastreamwasm_free(ptr >>> 0, 1));

class PmaStreamWasm {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        PmaStreamWasmFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_pmastreamwasm_free(ptr, 0);
    }
    constructor() {
        const ret = wasm.pmastreamwasm_new();
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0] >>> 0;
        PmaStreamWasmFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {number} value
     * @returns {Float64Array}
     */
    update(value) {
        const ret = wasm.pmastreamwasm_update(this.__wbg_ptr, value);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
}
module.exports.PmaStreamWasm = PmaStreamWasm;

const SqueezeMomentumResultFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_squeezemomentumresult_free(ptr >>> 0, 1));

class SqueezeMomentumResult {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        SqueezeMomentumResultFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_squeezemomentumresult_free(ptr, 0);
    }
    /**
     * @returns {Float64Array}
     */
    get values() {
        const ret = wasm.squeezemomentumresult_values(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @returns {number}
     */
    get rows() {
        const ret = wasm.squeezemomentumresult_rows(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get cols() {
        const ret = wasm.squeezemomentumresult_cols(this.__wbg_ptr);
        return ret >>> 0;
    }
}
module.exports.SqueezeMomentumResult = SqueezeMomentumResult;

const TilsonContextFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_tilsoncontext_free(ptr >>> 0, 1));

class TilsonContext {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        TilsonContextFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_tilsoncontext_free(ptr, 0);
    }
    /**
     * @param {number} period
     * @param {number} volume_factor
     */
    constructor(period, volume_factor) {
        const ret = wasm.tilsoncontext_new(period, volume_factor);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0] >>> 0;
        TilsonContextFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {number} value
     * @returns {number | undefined}
     */
    update(value) {
        const ret = wasm.tilsoncontext_update(this.__wbg_ptr, value);
        return ret[0] === 0 ? undefined : ret[1];
    }
    reset() {
        wasm.tilsoncontext_reset(this.__wbg_ptr);
    }
    /**
     * @returns {number}
     */
    get_warmup_period() {
        const ret = wasm.tilsoncontext_get_warmup_period(this.__wbg_ptr);
        return ret >>> 0;
    }
}
module.exports.TilsonContext = TilsonContext;

const TrimaContextFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_trimacontext_free(ptr >>> 0, 1));

class TrimaContext {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        TrimaContextFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_trimacontext_free(ptr, 0);
    }
    /**
     * @param {number} period
     */
    constructor(period) {
        const ret = wasm.trimacontext_new(period);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0] >>> 0;
        TrimaContextFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {number} in_ptr
     * @param {number} out_ptr
     * @param {number} len
     */
    update_into(in_ptr, out_ptr, len) {
        const ret = wasm.trimacontext_update_into(this.__wbg_ptr, in_ptr, out_ptr, len);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * @returns {number}
     */
    get_warmup_period() {
        const ret = wasm.trimacontext_get_warmup_period(this.__wbg_ptr);
        return ret >>> 0;
    }
}
module.exports.TrimaContext = TrimaContext;

const VpciContextFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_vpcicontext_free(ptr >>> 0, 1));

class VpciContext {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        VpciContextFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_vpcicontext_free(ptr, 0);
    }
    /**
     * @param {number} short_range
     * @param {number} long_range
     */
    constructor(short_range, long_range) {
        const ret = wasm.vpcicontext_new(short_range, long_range);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0] >>> 0;
        VpciContextFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {number} close_ptr
     * @param {number} volume_ptr
     * @param {number} vpci_ptr
     * @param {number} vpcis_ptr
     * @param {number} len
     */
    update_into(close_ptr, volume_ptr, vpci_ptr, vpcis_ptr, len) {
        const ret = wasm.vpcicontext_update_into(this.__wbg_ptr, close_ptr, volume_ptr, vpci_ptr, vpcis_ptr, len);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * @returns {number}
     */
    get_warmup_period() {
        const ret = wasm.vpcicontext_get_warmup_period(this.__wbg_ptr);
        return ret >>> 0;
    }
}
module.exports.VpciContext = VpciContext;

const VpwmaContextFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_vpwmacontext_free(ptr >>> 0, 1));

class VpwmaContext {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        VpwmaContextFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_vpwmacontext_free(ptr, 0);
    }
    /**
     * @param {number} period
     * @param {number} power
     */
    constructor(period, power) {
        const ret = wasm.vpwmacontext_new(period, power);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0] >>> 0;
        VpwmaContextFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {number} in_ptr
     * @param {number} out_ptr
     * @param {number} len
     */
    update_into(in_ptr, out_ptr, len) {
        const ret = wasm.vpwmacontext_update_into(this.__wbg_ptr, in_ptr, out_ptr, len);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * @returns {number}
     */
    get_warmup_period() {
        const ret = wasm.vpwmacontext_get_warmup_period(this.__wbg_ptr);
        return ret >>> 0;
    }
}
module.exports.VpwmaContext = VpwmaContext;

const VwapContextFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_vwapcontext_free(ptr >>> 0, 1));

class VwapContext {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        VwapContextFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_vwapcontext_free(ptr, 0);
    }
    /**
     * @param {string} anchor
     */
    constructor(anchor) {
        const ptr0 = passStringToWasm0(anchor, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.vwapcontext_new(ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0] >>> 0;
        VwapContextFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {number} timestamps_ptr
     * @param {number} volumes_ptr
     * @param {number} prices_ptr
     * @param {number} out_ptr
     * @param {number} len
     */
    update_into(timestamps_ptr, volumes_ptr, prices_ptr, out_ptr, len) {
        const ret = wasm.vwapcontext_update_into(this.__wbg_ptr, timestamps_ptr, volumes_ptr, prices_ptr, out_ptr, len);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
}
module.exports.VwapContext = VwapContext;

const VwmacdJsOutputFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_vwmacdjsoutput_free(ptr >>> 0, 1));

class VwmacdJsOutput {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        VwmacdJsOutputFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_vwmacdjsoutput_free(ptr, 0);
    }
    /**
     * @returns {Float64Array}
     */
    get macd() {
        const ret = wasm.__wbg_get_vwmacdjsoutput_macd(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @param {Float64Array} arg0
     */
    set macd(arg0) {
        const ptr0 = passArrayF64ToWasm0(arg0, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_vwmacdjsoutput_macd(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @returns {Float64Array}
     */
    get signal() {
        const ret = wasm.__wbg_get_vwmacdjsoutput_signal(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @param {Float64Array} arg0
     */
    set signal(arg0) {
        const ptr0 = passArrayF64ToWasm0(arg0, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_vwmacdjsoutput_signal(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @returns {Float64Array}
     */
    get hist() {
        const ret = wasm.__wbg_get_vwmacdjsoutput_hist(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @param {Float64Array} arg0
     */
    set hist(arg0) {
        const ptr0 = passArrayF64ToWasm0(arg0, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_vwmacdjsoutput_hist(this.__wbg_ptr, ptr0, len0);
    }
}
module.exports.VwmacdJsOutput = VwmacdJsOutput;

module.exports.__wbg_String_8f0eb39a4a4c2f66 = function(arg0, arg1) {
    const ret = String(arg1);
    const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len1 = WASM_VECTOR_LEN;
    getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
    getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
};

module.exports.__wbg_buffer_609cc3eee51ed158 = function(arg0) {
    const ret = arg0.buffer;
    return ret;
};

module.exports.__wbg_call_672a4d21634d4a24 = function() { return handleError(function (arg0, arg1) {
    const ret = arg0.call(arg1);
    return ret;
}, arguments) };

module.exports.__wbg_done_769e5ede4b31c67b = function(arg0) {
    const ret = arg0.done;
    return ret;
};

module.exports.__wbg_get_67b2ba62fc30de12 = function() { return handleError(function (arg0, arg1) {
    const ret = Reflect.get(arg0, arg1);
    return ret;
}, arguments) };

module.exports.__wbg_get_b9b93047fe3cf45b = function(arg0, arg1) {
    const ret = arg0[arg1 >>> 0];
    return ret;
};

module.exports.__wbg_getwithrefkey_1dc361bd10053bfe = function(arg0, arg1) {
    const ret = arg0[arg1];
    return ret;
};

module.exports.__wbg_instanceof_ArrayBuffer_e14585432e3737fc = function(arg0) {
    let result;
    try {
        result = arg0 instanceof ArrayBuffer;
    } catch (_) {
        result = false;
    }
    const ret = result;
    return ret;
};

module.exports.__wbg_instanceof_Uint8Array_17156bcf118086a9 = function(arg0) {
    let result;
    try {
        result = arg0 instanceof Uint8Array;
    } catch (_) {
        result = false;
    }
    const ret = result;
    return ret;
};

module.exports.__wbg_isArray_a1eab7e0d067391b = function(arg0) {
    const ret = Array.isArray(arg0);
    return ret;
};

module.exports.__wbg_isSafeInteger_343e2beeeece1bb0 = function(arg0) {
    const ret = Number.isSafeInteger(arg0);
    return ret;
};

module.exports.__wbg_iterator_9a24c88df860dc65 = function() {
    const ret = Symbol.iterator;
    return ret;
};

module.exports.__wbg_length_a446193dc22c12f8 = function(arg0) {
    const ret = arg0.length;
    return ret;
};

module.exports.__wbg_length_e2d2a49132c1b256 = function(arg0) {
    const ret = arg0.length;
    return ret;
};

module.exports.__wbg_new_405e22f390576ce2 = function() {
    const ret = new Object();
    return ret;
};

module.exports.__wbg_new_5e0be73521bc8c17 = function() {
    const ret = new Map();
    return ret;
};

module.exports.__wbg_new_78c8a92080461d08 = function(arg0) {
    const ret = new Float64Array(arg0);
    return ret;
};

module.exports.__wbg_new_78feb108b6472713 = function() {
    const ret = new Array();
    return ret;
};

module.exports.__wbg_new_a12002a7f91c75be = function(arg0) {
    const ret = new Uint8Array(arg0);
    return ret;
};

module.exports.__wbg_newwithbyteoffsetandlength_93c8e0c1a479fa1a = function(arg0, arg1, arg2) {
    const ret = new Float64Array(arg0, arg1 >>> 0, arg2 >>> 0);
    return ret;
};

module.exports.__wbg_next_25feadfc0913fea9 = function(arg0) {
    const ret = arg0.next;
    return ret;
};

module.exports.__wbg_next_6574e1a8a62d1055 = function() { return handleError(function (arg0) {
    const ret = arg0.next();
    return ret;
}, arguments) };

module.exports.__wbg_push_737cfc8c1432c2c6 = function(arg0, arg1) {
    const ret = arg0.push(arg1);
    return ret;
};

module.exports.__wbg_set_37837023f3d740e8 = function(arg0, arg1, arg2) {
    arg0[arg1 >>> 0] = arg2;
};

module.exports.__wbg_set_3f1d0b984ed272ed = function(arg0, arg1, arg2) {
    arg0[arg1] = arg2;
};

module.exports.__wbg_set_65595bdd868b3009 = function(arg0, arg1, arg2) {
    arg0.set(arg1, arg2 >>> 0);
};

module.exports.__wbg_set_8fc6bf8a5b1071d1 = function(arg0, arg1, arg2) {
    const ret = arg0.set(arg1, arg2);
    return ret;
};

module.exports.__wbg_set_bb8cecf6a62b9f46 = function() { return handleError(function (arg0, arg1, arg2) {
    const ret = Reflect.set(arg0, arg1, arg2);
    return ret;
}, arguments) };

module.exports.__wbg_value_cd1ffa7b1ab794f1 = function(arg0) {
    const ret = arg0.value;
    return ret;
};

module.exports.__wbindgen_as_number = function(arg0) {
    const ret = +arg0;
    return ret;
};

module.exports.__wbindgen_bigint_from_i64 = function(arg0) {
    const ret = arg0;
    return ret;
};

module.exports.__wbindgen_bigint_from_u64 = function(arg0) {
    const ret = BigInt.asUintN(64, arg0);
    return ret;
};

module.exports.__wbindgen_bigint_get_as_i64 = function(arg0, arg1) {
    const v = arg1;
    const ret = typeof(v) === 'bigint' ? v : undefined;
    getDataViewMemory0().setBigInt64(arg0 + 8 * 1, isLikeNone(ret) ? BigInt(0) : ret, true);
    getDataViewMemory0().setInt32(arg0 + 4 * 0, !isLikeNone(ret), true);
};

module.exports.__wbindgen_boolean_get = function(arg0) {
    const v = arg0;
    const ret = typeof(v) === 'boolean' ? (v ? 1 : 0) : 2;
    return ret;
};

module.exports.__wbindgen_debug_string = function(arg0, arg1) {
    const ret = debugString(arg1);
    const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len1 = WASM_VECTOR_LEN;
    getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
    getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
};

module.exports.__wbindgen_error_new = function(arg0, arg1) {
    const ret = new Error(getStringFromWasm0(arg0, arg1));
    return ret;
};

module.exports.__wbindgen_in = function(arg0, arg1) {
    const ret = arg0 in arg1;
    return ret;
};

module.exports.__wbindgen_init_externref_table = function() {
    const table = wasm.__wbindgen_export_5;
    const offset = table.grow(4);
    table.set(0, undefined);
    table.set(offset + 0, undefined);
    table.set(offset + 1, null);
    table.set(offset + 2, true);
    table.set(offset + 3, false);
    ;
};

module.exports.__wbindgen_is_bigint = function(arg0) {
    const ret = typeof(arg0) === 'bigint';
    return ret;
};

module.exports.__wbindgen_is_function = function(arg0) {
    const ret = typeof(arg0) === 'function';
    return ret;
};

module.exports.__wbindgen_is_object = function(arg0) {
    const val = arg0;
    const ret = typeof(val) === 'object' && val !== null;
    return ret;
};

module.exports.__wbindgen_is_string = function(arg0) {
    const ret = typeof(arg0) === 'string';
    return ret;
};

module.exports.__wbindgen_is_undefined = function(arg0) {
    const ret = arg0 === undefined;
    return ret;
};

module.exports.__wbindgen_jsval_eq = function(arg0, arg1) {
    const ret = arg0 === arg1;
    return ret;
};

module.exports.__wbindgen_jsval_loose_eq = function(arg0, arg1) {
    const ret = arg0 == arg1;
    return ret;
};

module.exports.__wbindgen_memory = function() {
    const ret = wasm.memory;
    return ret;
};

module.exports.__wbindgen_number_get = function(arg0, arg1) {
    const obj = arg1;
    const ret = typeof(obj) === 'number' ? obj : undefined;
    getDataViewMemory0().setFloat64(arg0 + 8 * 1, isLikeNone(ret) ? 0 : ret, true);
    getDataViewMemory0().setInt32(arg0 + 4 * 0, !isLikeNone(ret), true);
};

module.exports.__wbindgen_number_new = function(arg0) {
    const ret = arg0;
    return ret;
};

module.exports.__wbindgen_string_get = function(arg0, arg1) {
    const obj = arg1;
    const ret = typeof(obj) === 'string' ? obj : undefined;
    var ptr1 = isLikeNone(ret) ? 0 : passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    var len1 = WASM_VECTOR_LEN;
    getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
    getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
};

module.exports.__wbindgen_string_new = function(arg0, arg1) {
    const ret = getStringFromWasm0(arg0, arg1);
    return ret;
};

module.exports.__wbindgen_throw = function(arg0, arg1) {
    throw new Error(getStringFromWasm0(arg0, arg1));
};
imports.wbg = { memory: new WebAssembly.Memory({initial:7}) };
const path = require('path').join(__dirname, 'my_project_bg.wasm');
const bytes = require('fs').readFileSync(path);

const wasmModule = new WebAssembly.Module(bytes);
const wasmInstance = new WebAssembly.Instance(wasmModule, imports);
wasm = wasmInstance.exports;
module.exports.__wasm = wasm;

wasm.__wbindgen_start();

