
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Kernel {
    Auto,
    Scalar,
    Avx2,
    Avx512,
}

impl Default for Kernel {
    fn default() -> Self {
        Kernel::Auto
    }
}