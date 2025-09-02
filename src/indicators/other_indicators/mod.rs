// Other indicators module

pub mod alphatrend;
pub use alphatrend::{
    alphatrend, alphatrend_with_kernel, AlphaTrendInput, AlphaTrendOutput, AlphaTrendParams,
    AlphaTrendError, AlphaTrendData, AlphaTrendBuilder, AlphaTrendStream,
};