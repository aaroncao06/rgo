use crate::game::board::Player;
use crate::inference::policy::{BOARD_POLICY_SIZE, POLICY_SIZE};
use crate::inference::symmetry::Symmetry;

const SCORE_MULTIPLIER: f32 = 20.0;

// opponent policy refers to the response
// all nn outputs that we train, some not used during mcts
#[allow(dead_code)]
struct RawNNOutputs {
    policy_logits: [f32; POLICY_SIZE],
    win_logit: f32,
    raw_score_mean: f32,
    raw_score_stdev_logit: f32,
    soft_policy_logits: [f32; POLICY_SIZE],
    opponent_policy_logits: [f32; POLICY_SIZE],
    soft_opponent_policy_logits: [f32; POLICY_SIZE],
    ownership_logits: [f32; BOARD_POLICY_SIZE],
}

#[derive(Clone)]
pub(crate) struct NNOutput {
    policy: [f32; POLICY_SIZE], // logits -> probs
    win: f32,                   // logit -> white prob
    score_mean: f32,            // score mean -> white score mean
    score_aux: f32,             // stdev logit -> white score mean sq
    processed: bool,            //just to be safe now that we are modifying in place
    // Optional logits -> signed white ownership; no inline map on ordinary evaluations.
    ownership: Option<Box<[f32; BOARD_POLICY_SIZE]>>,
}

fn masked_softmax_in_place(policy: &mut [f32; POLICY_SIZE], legal_mask: &[bool; POLICY_SIZE]) {
    let mut max_logit = f32::NEG_INFINITY;
    let mut legal_count = 0_usize;
    for i in 0..POLICY_SIZE {
        if legal_mask[i] {
            debug_assert!(policy[i].is_finite(), "policy logits must be finite");
            legal_count += 1;
            max_logit = max_logit.max(policy[i]);
        }
    }
    debug_assert!(legal_count > 0, "position has no legal moves");
    let mut sum = 0_f32;
    for i in 0..POLICY_SIZE {
        if legal_mask[i] {
            let weight = (policy[i] - max_logit).exp();
            policy[i] = weight;
            sum += weight;
        } else {
            policy[i] = 0_f32;
        }
    }
    debug_assert!(
        sum.is_finite() && sum >= 1_f32,
        "invalid policy normalization"
    );
    let inv_sum = 1_f32 / sum;
    for i in 0..POLICY_SIZE {
        if legal_mask[i] {
            policy[i] *= inv_sum;
        }
    }
}

fn sigmoid(logit: f32) -> f32 {
    if logit >= 0.0 {
        1.0 / (1.0 + (-logit).exp())
    } else {
        let exp_logit = logit.exp();
        exp_logit / (1.0 + exp_logit)
    }
}

fn softplus(value: f32) -> f32 {
    value.max(0.0) + (-value.abs()).exp().ln_1p()
}
fn win_prob_to_white(prob: f32, next_player: Player) -> f32 {
    match next_player {
        Player::White => prob,
        Player::Black => 1.0 - prob,
    }
}
fn signed_value_to_white(value: f32, next_player: Player) -> f32 {
    match next_player {
        Player::White => value,
        Player::Black => -value,
    }
}
fn score_mean_sq(mean: f32, stdev: f32) -> f32 {
    mean.mul_add(mean, stdev * stdev)
}
impl NNOutput {
    pub(crate) fn from_raw(
        policy_logits: [f32; POLICY_SIZE],
        win_logit: f32,
        raw_score_mean: f32,
        raw_score_stdev_logit: f32,
    ) -> Self {
        // Finite raw activations are part of the model/backend contract.
        debug_assert!(
            win_logit.is_finite()
                && raw_score_mean.is_finite()
                && raw_score_stdev_logit.is_finite(),
            "nonfinite value output"
        );
        Self {
            policy: policy_logits,
            win: win_logit,
            score_mean: raw_score_mean,
            score_aux: raw_score_stdev_logit,
            processed: false,
            ownership: None,
        }
    }
    pub(crate) fn with_ownership_logits(mut self, logits: [f32; BOARD_POLICY_SIZE]) -> Self {
        debug_assert!(!self.processed);
        debug_assert!(
            logits.iter().all(|v| v.is_finite()),
            "nonfinite ownership output"
        );
        self.ownership = Some(Box::new(logits));
        self
    }
    pub(super) fn restore_symmetry_in_place(&mut self, symmetry: Symmetry) {
        debug_assert!(!self.processed);
        let spatial_policy = self
            .policy
            .first_chunk_mut::<BOARD_POLICY_SIZE>()
            .expect("policy contains a full board plane");
        symmetry.restore_output(spatial_policy);
        if let Some(ownership) = self.ownership.as_mut() {
            symmetry.restore_output(ownership);
        }
    }
    pub(crate) fn has_ownership(&self) -> bool {
        self.ownership.is_some()
    }
    pub(crate) fn white_ownership(&self) -> Option<&[f32; BOARD_POLICY_SIZE]> {
        debug_assert!(self.processed);
        self.ownership.as_deref()
    }
    pub(crate) fn process_in_place(
        &mut self,
        next_player: Player,
        legal_mask: &[bool; POLICY_SIZE],
    ) {
        debug_assert!(!self.processed, "NN output processed twice");

        masked_softmax_in_place(&mut self.policy, legal_mask);

        let current_win_prob = sigmoid(self.win);
        let score_mean = self.score_mean * SCORE_MULTIPLIER;
        let score_stdev = softplus(self.score_aux) * SCORE_MULTIPLIER;

        self.win = win_prob_to_white(current_win_prob, next_player);
        self.score_mean = signed_value_to_white(score_mean, next_player);
        self.score_aux = score_mean_sq(score_mean, score_stdev);
        self.process_ownership_in_place(next_player);
        self.processed = true;
    }

    /// Upgrade using this fresh allocation and ownership map, without mutating
    /// the shared cached output or postprocessing policy/value a second time.
    pub(crate) fn process_with_cached_values_in_place(
        &mut self,
        next_player: Player,
        cached: &Self,
    ) {
        debug_assert!(!self.processed && cached.processed);
        debug_assert!(self.has_ownership() && !cached.has_ownership());
        self.policy = cached.policy;
        self.win = cached.win;
        self.score_mean = cached.score_mean;
        self.score_aux = cached.score_aux;
        self.process_ownership_in_place(next_player);
        self.processed = true;
    }

    fn process_ownership_in_place(&mut self, next_player: Player) {
        if let Some(ownership) = self.ownership.as_mut() {
            for value in ownership.iter_mut() {
                *value = signed_value_to_white(value.tanh(), next_player);
            }
        }
    }
    pub(crate) fn policy_probs(&self) -> &[f32; POLICY_SIZE] {
        debug_assert!(self.processed);
        &self.policy
    }

    pub(crate) fn policy_probs_mut(&mut self) -> &mut [f32; POLICY_SIZE] {
        debug_assert!(self.processed);
        &mut self.policy
    }

    pub(crate) fn white_win_prob(&self) -> f32 {
        debug_assert!(self.processed);
        self.win
    }

    pub(crate) fn white_score_mean(&self) -> f32 {
        debug_assert!(self.processed);
        self.score_mean
    }

    pub(crate) fn white_score_mean_sq(&self) -> f32 {
        debug_assert!(self.processed);
        self.score_aux
    }

    pub(crate) fn is_processed(&self) -> bool {
        self.processed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ownership_upgrade_reuses_fresh_map_and_preserves_processed_predictions() {
        for player in [Player::Black, Player::White] {
            let mut logits = [0.0; POLICY_SIZE];
            logits[1] = 2.0;
            let mut legal = [true; POLICY_SIZE];
            legal[0] = false;
            let mut cached = NNOutput::from_raw(logits, 1.0, 2.0, -1.0);
            cached.process_in_place(player, &legal);
            let mut fresh = NNOutput::from_raw([5.0; POLICY_SIZE], -2.0, -3.0, 2.0)
                .with_ownership_logits([1.0; BOARD_POLICY_SIZE]);
            let map_address = fresh.ownership.as_ref().unwrap().as_ptr();
            fresh.process_with_cached_values_in_place(player, &cached);
            assert_eq!(fresh.white_ownership().unwrap().as_ptr(), map_address);
            assert_eq!(fresh.policy_probs(), cached.policy_probs());
            assert_eq!(fresh.policy_probs()[0], 0.0);
            assert_eq!(fresh.white_win_prob(), cached.white_win_prob());
            assert_eq!(fresh.white_score_mean(), cached.white_score_mean());
            assert_eq!(fresh.white_score_mean_sq(), cached.white_score_mean_sq());
            let expected = signed_value_to_white(1.0_f32.tanh(), player);
            assert!(
                fresh
                    .white_ownership()
                    .unwrap()
                    .iter()
                    .all(|&value| (value - expected).abs() < 1e-6)
            );
            assert!(!cached.has_ownership());
        }
    }

    #[test]
    fn ownership_is_optional_and_converted_from_logits_to_white() {
        for player in [Player::Black, Player::White] {
            let mut output = NNOutput::from_raw([0.0; POLICY_SIZE], 0.0, 0.0, 0.0)
                .with_ownership_logits([1.0; BOARD_POLICY_SIZE]);
            output.process_in_place(player, &[true; POLICY_SIZE]);
            let expected = if player == Player::White {
                1.0_f32.tanh()
            } else {
                -1.0_f32.tanh()
            };
            assert!(
                output
                    .white_ownership()
                    .unwrap()
                    .iter()
                    .all(|&value| (value - expected).abs() < 1e-6)
            );
            let mut cloned = output.clone();
            cloned.policy_probs_mut()[0] = 0.0;
            assert_eq!(cloned.white_ownership(), output.white_ownership());
            assert_ne!(cloned.policy_probs()[0], output.policy_probs()[0]);
        }
        let mut output = NNOutput::from_raw([0.0; POLICY_SIZE], 0.0, 0.0, 0.0);
        output.process_in_place(Player::Black, &[true; POLICY_SIZE]);
        assert!(output.white_ownership().is_none());
    }

    fn assert_close(actual: f32, expected: f32) {
        let tolerance = 1e-5 * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual}"
        );
    }

    fn process(
        policy_logits: [f32; POLICY_SIZE],
        win_logit: f32,
        raw_score_mean: f32,
        raw_score_stdev_logit: f32,
        next_player: Player,
        legal_mask: &[bool; POLICY_SIZE],
    ) -> NNOutput {
        let mut output = NNOutput::from_raw(
            policy_logits,
            win_logit,
            raw_score_mean,
            raw_score_stdev_logit,
        );
        assert!(!output.is_processed());
        output.process_in_place(next_player, legal_mask);
        assert!(output.is_processed());
        output
    }

    #[test]
    fn masked_softmax_zeros_illegal_moves_and_normalizes_legal_moves() {
        let policy_logits = [0.0; POLICY_SIZE];
        let mut legal_mask = [false; POLICY_SIZE];
        legal_mask[3] = true;
        legal_mask[17] = true;

        let output = process(policy_logits, 0.0, 0.0, 0.0, Player::White, &legal_mask);
        let probabilities = output.policy_probs();

        assert_close(probabilities[3], 0.5);
        assert_close(probabilities[17], 0.5);
        assert_eq!(probabilities[0], 0.0);
        assert_close(probabilities.iter().sum(), 1.0);
    }

    #[test]
    fn masked_softmax_is_stable_for_extreme_logits() {
        let mut policy_logits = [0.0; POLICY_SIZE];
        policy_logits[4] = 1_000.0;
        policy_logits[9] = -1_000.0;
        let mut legal_mask = [false; POLICY_SIZE];
        legal_mask[4] = true;
        legal_mask[9] = true;

        let output = process(policy_logits, 0.0, 0.0, 0.0, Player::White, &legal_mask);
        let probabilities = output.policy_probs();

        assert!(
            probabilities
                .iter()
                .all(|probability| probability.is_finite())
        );
        assert_close(probabilities[4], 1.0);
        assert_eq!(probabilities[9], 0.0);
        assert_close(probabilities.iter().sum(), 1.0);
    }

    #[test]
    fn processing_scales_scores_and_converts_perspective() {
        let policy_logits = [0.0; POLICY_SIZE];
        let mut legal_mask = [false; POLICY_SIZE];
        legal_mask[0] = true;
        let win_logit = 3.0_f32.ln();
        let raw_score_mean = 0.5;
        let raw_score_stdev_logit = 0.0;

        let white = process(
            policy_logits,
            win_logit,
            raw_score_mean,
            raw_score_stdev_logit,
            Player::White,
            &legal_mask,
        );
        let black = process(
            policy_logits,
            win_logit,
            raw_score_mean,
            raw_score_stdev_logit,
            Player::Black,
            &legal_mask,
        );

        let expected_mean = raw_score_mean * SCORE_MULTIPLIER;
        let expected_stdev = std::f32::consts::LN_2 * SCORE_MULTIPLIER;
        let expected_mean_sq =
            expected_mean.mul_add(expected_mean, expected_stdev * expected_stdev);

        assert_close(white.white_win_prob(), 0.75);
        assert_close(black.white_win_prob(), 0.25);
        assert_close(white.white_score_mean(), expected_mean);
        assert_close(black.white_score_mean(), -expected_mean);
        assert_close(white.white_score_mean_sq(), expected_mean_sq);
        assert_close(black.white_score_mean_sq(), expected_mean_sq);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "nonfinite value output")]
    fn raw_outputs_reject_nonfinite_scalar_values() {
        let _ = NNOutput::from_raw([0.0; POLICY_SIZE], f32::NAN, 0.0, 0.0);
    }
}
