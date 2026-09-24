use crate::inference::{
    inputs::NNInput,
    policy::{BOARD_POLICY_SIZE, POLICY_SIZE},
};

#[derive(Debug, PartialEq)]
pub(super) struct ValueTarget {
    pub(super) win_probability: f32,
    pub(super) score_mean: f32,
    pub(super) score_stdev: f32,
    /// 0 = opponent, 1 = neutral, 2 = current player.
    pub(super) ownership: [u8; BOARD_POLICY_SIZE],
}

// Eventually this will become a packed representation for network transfer.
// Record finalization is already the allocation boundary where packing belongs.
pub(super) struct TrainingSample {
    pub(super) input: NNInput,
    pub(super) policy_target: [f32; POLICY_SIZE],
    pub(super) value_target: ValueTarget,
}
