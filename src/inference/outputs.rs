use crate::game::board::Player;
use crate::inference::policy::{BOARD_POLICY_SIZE, POLICY_SIZE};

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

pub struct NNOutput {
    policy: [f32; POLICY_SIZE], // logits -> probs
    win: f32,                   // logit -> white prob
    score_mean: f32,            // score mean -> white score mean
    score_aux: f32,             // stdev logit -> white score mean sq
    processed: bool,            //just to be safe now that we are modifying in place
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
    assert!(
        sum.is_finite() && sum >= 1_f32,
        "invalid policy normalization"
    ); //keep for now
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
        assert!(
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
        }
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
        self.processed = true;
    }
    pub fn policy_probs(&self) -> &[f32; POLICY_SIZE] {
        debug_assert!(self.processed);
        &self.policy
    }

    pub fn white_win_prob(&self) -> f32 {
        debug_assert!(self.processed);
        self.win
    }

    pub fn white_score_mean(&self) -> f32 {
        debug_assert!(self.processed);
        self.score_mean
    }

    pub fn white_score_mean_sq(&self) -> f32 {
        debug_assert!(self.processed);
        self.score_aux
    }

    pub fn is_processed(&self) -> bool {
        self.processed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    #[should_panic(expected = "nonfinite value output")]
    fn raw_outputs_reject_nonfinite_scalar_values() {
        let _ = NNOutput::from_raw([0.0; POLICY_SIZE], f32::NAN, 0.0, 0.0);
    }
}
