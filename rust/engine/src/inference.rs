/// Board sizes supported by self-play and inference batching.
pub const SUPPORTED_BOARD_DIMS: [usize; 3] = [9, 13, 19];

pub mod backend;
pub mod inputs;
pub mod onnx;
pub mod outputs;
pub mod policy;
pub mod runtime;
mod symmetry;
