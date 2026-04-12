use std::env;

use cust::{
    context::CurrentContext,
    device::DeviceAttribute,
    error::CudaResult,
    module::{Module, ModuleJitOption, OptLevel},
};

const TARGET_CUBIN_MAJOR: i32 = 8;
const TARGET_CUBIN_MINOR: i32 = 9;
const FORCE_PTX_ENV: &str = "VECTOR_TA_CUDA_FORCE_PTX";
const FORCE_CUBIN_ENV: &str = "VECTOR_TA_CUDA_FORCE_CUBIN";
const DEBUG_ENV: &str = "CUDA_MODULE_LOAD_DEBUG";

fn env_flag(name: &str) -> bool {
    matches!(
        env::var(name).ok().as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("True")
    )
}

fn debug_enabled() -> bool {
    env_flag(DEBUG_ENV)
}

fn current_context_is_sm89() -> CudaResult<bool> {
    let device = CurrentContext::get_device()?;
    let major = device.get_attribute(DeviceAttribute::ComputeCapabilityMajor)?;
    let minor = device.get_attribute(DeviceAttribute::ComputeCapabilityMinor)?;
    Ok(major == TARGET_CUBIN_MAJOR && minor == TARGET_CUBIN_MINOR)
}

fn load_ptx_module(ptx: &str) -> CudaResult<Module> {
    let jit_opts = &[
        ModuleJitOption::DetermineTargetFromContext,
        ModuleJitOption::OptLevel(OptLevel::O2),
    ];

    Module::from_ptx(ptx, jit_opts)
        .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
        .or_else(|_| Module::from_ptx(ptx, &[]))
}

pub(crate) fn load_module_for_current_context(
    ptx: &str,
    sm89_cubin: &[u8],
    stem: &str,
) -> CudaResult<Module> {
    let force_ptx = env_flag(FORCE_PTX_ENV);
    let force_cubin = env_flag(FORCE_CUBIN_ENV);

    let prefer_cubin = if force_ptx {
        false
    } else if force_cubin {
        true
    } else {
        current_context_is_sm89()?
    };

    if prefer_cubin {
        match Module::from_cubin(sm89_cubin, &[]) {
            Ok(module) => {
                if debug_enabled() {
                    eprintln!("[cuda-module-loader] loaded sm_89 cubin for {stem}");
                }
                return Ok(module);
            }
            Err(err) => {
                if debug_enabled() {
                    eprintln!(
                        "[cuda-module-loader] cubin load failed for {stem}, falling back to PTX: {err:?}"
                    );
                }
            }
        }
    } else if debug_enabled() {
        eprintln!("[cuda-module-loader] using PTX path for {stem}");
    }

    load_ptx_module(ptx)
}

#[macro_export]
macro_rules! load_cuda_embedded_module {
    ($stem:literal) => {{
        $crate::cuda::module_loader::load_module_for_current_context(
            include_str!(concat!(env!("OUT_DIR"), "/", $stem, ".ptx")),
            include_bytes!(concat!(env!("OUT_DIR"), "/", $stem, "_sm89.cubin")),
            $stem,
        )
    }};
}
