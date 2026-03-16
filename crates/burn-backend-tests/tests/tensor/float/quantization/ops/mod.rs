pub use super::*;

mod matmul;
mod quantize;

// TODO: re-enable for cubecl backends when inputs are valid for packed U32 storage
#[cfg(feature = "ndarray")]
mod extended;

// Tests for cubecl q_slice/q_expand/q_flip/q_gather/q_select using U32-aligned shapes
#[cfg(all(feature = "quantization", not(feature = "ndarray")))]
mod cubecl_qtensor_ops;
