use std::sync::OnceLock;
use crate::utilities::enums::Kernel;

#[inline(always)]
pub fn detect_best_kernel() -> Kernel {
    static CHOICE: OnceLock<Kernel> = OnceLock::new();

    *CHOICE.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::is_x86_feature_detected;
            if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("fma") {
                return Kernel::Avx512;
            }
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return Kernel::Avx2;
            }
        }
        Kernel::Scalar
    })
}