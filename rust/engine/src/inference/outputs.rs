use crate::game::board::Player;
use crate::inference::symmetry::Symmetry;
use std::sync::Arc;

const SCORE_MULTIPLIER: f32 = 20.0;

// opponent policy refers to the response
// all nn outputs that we train, some not used during mcts
#[allow(dead_code)]
struct RawNNOutputs {
    policy_logits: Box<[f32]>,
    win_logit: f32,
    raw_score_mean: f32,
    raw_score_stdev_logit: f32,
    soft_policy_logits: Box<[f32]>,
    opponent_policy_logits: Box<[f32]>,
    soft_opponent_policy_logits: Box<[f32]>,
    ownership_logits: Box<[f32]>,
}

#[derive(Clone)]
pub struct NNOutput {
    policy: Box<[f32]>, // Active row-major board_area, then pass; logits -> probs.
    win: f32,           // logit -> white prob
    score_mean: f32,    // score mean -> white score mean
    score_aux: f32,     // stdev logit -> white score mean sq
    processed: bool,    //just to be safe now that we are modifying in place
    // Processed exclusively, then shared unchanged when root policy is cloned.
    ownership: Option<Arc<[f32]>>,
}

fn masked_softmax_in_place(policy: &mut [f32], legal_mask: &[bool]) {
    debug_assert_eq!(policy.len(), legal_mask.len());
    let mut max_logit = f32::NEG_INFINITY;
    let mut legal_count = 0_usize;
    for i in 0..policy.len() {
        if legal_mask[i] {
            debug_assert!(policy[i].is_finite(), "policy logits must be finite");
            legal_count += 1;
            max_logit = max_logit.max(policy[i]);
        }
    }
    debug_assert!(legal_count > 0, "position has no legal moves");
    let mut sum = 0_f32;
    for i in 0..policy.len() {
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
    for i in 0..policy.len() {
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
    pub fn from_raw(
        policy_logits: Box<[f32]>,
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
    /// Ownership must remain exclusively owned through symmetry restoration
    /// and processing. After processing, output clones share the immutable map.
    pub fn with_ownership_logits(mut self, logits: Arc<[f32]>) -> Self {
        debug_assert!(!self.processed);
        debug_assert!(
            logits.iter().all(|v| v.is_finite()),
            "nonfinite ownership output"
        );
        self.ownership = Some(logits);
        self
    }
    pub(super) fn restore_symmetry_in_place(&mut self, symmetry: Symmetry, board_dim: usize) {
        debug_assert!(!self.processed);
        debug_assert_eq!(self.policy.len(), board_dim * board_dim + 1);
        let spatial_policy = &mut self.policy[..board_dim * board_dim];
        symmetry.restore_output(spatial_policy, board_dim, board_dim);
        if let Some(ownership) = self.ownership.as_mut() {
            let ownership =
                Arc::get_mut(ownership).expect("raw ownership must be exclusively owned");
            symmetry.restore_output(ownership, board_dim, board_dim);
        }
    }
    pub fn has_ownership(&self) -> bool {
        self.ownership.is_some()
    }
    /// Exact-dim, row-major ownership plane: index with the active board stride.
    pub fn white_ownership(&self) -> Option<&[f32]> {
        debug_assert!(self.processed);
        self.ownership.as_deref()
    }
    pub(crate) fn process_in_place(
        &mut self,
        next_player: Player,
        legal_mask: &[bool],
        board_dim: usize,
    ) {
        debug_assert!(!self.processed, "NN output processed twice");

        debug_assert_eq!(self.policy.len(), board_dim * board_dim + 1);
        let legal_mask = &legal_mask[..self.policy.len()];
        masked_softmax_in_place(&mut self.policy, legal_mask);

        let current_win_prob = sigmoid(self.win);
        let score_mean = self.score_mean * SCORE_MULTIPLIER;
        let score_stdev = softplus(self.score_aux) * SCORE_MULTIPLIER;

        self.win = win_prob_to_white(current_win_prob, next_player);
        self.score_mean = signed_value_to_white(score_mean, next_player);
        self.score_aux = score_mean_sq(score_mean, score_stdev);
        self.process_ownership_in_place(next_player, board_dim);
        self.processed = true;
    }

    /// Upgrade using this fresh allocation and ownership map, without mutating
    /// the shared cached output or postprocessing policy/value a second time.
    pub(crate) fn process_with_cached_values_in_place(
        &mut self,
        next_player: Player,
        cached: &Self,
        board_dim: usize,
    ) {
        debug_assert!(!self.processed && cached.processed);
        debug_assert!(self.has_ownership() && !cached.has_ownership());
        debug_assert_eq!(self.policy.len(), board_dim * board_dim + 1);
        self.policy.copy_from_slice(&cached.policy);
        self.win = cached.win;
        self.score_mean = cached.score_mean;
        self.score_aux = cached.score_aux;
        self.process_ownership_in_place(next_player, board_dim);
        self.processed = true;
    }

    fn process_ownership_in_place(&mut self, next_player: Player, board_dim: usize) {
        if let Some(ownership) = self.ownership.as_mut() {
            let ownership =
                Arc::get_mut(ownership).expect("raw ownership must be exclusively owned");
            debug_assert_eq!(ownership.len(), board_dim * board_dim);
            for value in ownership.iter_mut() {
                *value = signed_value_to_white(value.tanh(), next_player);
            }
        }
    }
    pub fn policy_probs(&self) -> &[f32] {
        debug_assert!(self.processed);
        &self.policy
    }

    pub(crate) fn policy_probs_mut(&mut self) -> &mut [f32] {
        debug_assert!(self.processed);
        &mut self.policy
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
    use crate::{
        game::board::{MAX_BOARD_AREA, MAX_BOARD_DIM},
        inference::policy::MAX_POLICY_SIZE,
    };

    #[test]
    fn compact_policy_restores_every_symmetry_and_preserves_pass_and_allocations() {
        for dim in 1..=MAX_BOARD_DIM {
            let board_area = dim * dim;
            let canonical: Vec<_> = (0..board_area).map(|i| i as f32 + 0.25).collect();
            for symmetry in Symmetry::ALL {
                let mut logits = vec![0.0; board_area + 1].into_boxed_slice();
                for y in 0..dim {
                    for x in 0..dim {
                        let end = dim - 1;
                        let (tx, ty) = match symmetry {
                            Symmetry::Identity => (x, y),
                            Symmetry::FlipY => (x, end - y),
                            Symmetry::FlipX => (end - x, y),
                            Symmetry::FlipXY => (end - x, end - y),
                            Symmetry::Transpose => (y, x),
                            Symmetry::TransposeFlipY => (end - y, x),
                            Symmetry::TransposeFlipX => (y, end - x),
                            Symmetry::TransposeFlipXY => (end - y, end - x),
                        };
                        logits[tx + ty * dim] = canonical[x + y * dim];
                    }
                }
                logits[board_area] = 1234.0;
                let ownership = logits[..board_area].into();
                let mut output =
                    NNOutput::from_raw(logits, 0.0, 0.0, 0.0).with_ownership_logits(ownership);
                let policy_address = output.policy.as_ptr();
                let ownership_address = output.ownership.as_ref().unwrap().as_ptr();
                output.restore_symmetry_in_place(symmetry, dim);
                assert_eq!(
                    &output.policy[..board_area],
                    canonical,
                    "dim={dim}, {symmetry:?}"
                );
                assert_eq!(output.policy[board_area], 1234.0);
                assert_eq!(output.policy.as_ptr(), policy_address);
                assert_eq!(&**output.ownership.as_ref().unwrap(), canonical);
                assert_eq!(
                    output.ownership.as_ref().unwrap().as_ptr(),
                    ownership_address
                );
            }
        }
    }

    #[test]
    fn ownership_upgrade_reuses_fresh_map_and_preserves_processed_predictions() {
        for player in [Player::Black, Player::White] {
            let mut logits = [0.0; MAX_POLICY_SIZE];
            logits[1] = 2.0;
            let mut legal = [true; MAX_POLICY_SIZE];
            legal[0] = false;
            let mut cached = NNOutput::from_raw(logits.into(), 1.0, 2.0, -1.0);
            cached.process_in_place(player, &legal, MAX_BOARD_DIM);
            let mut fresh = NNOutput::from_raw([5.0; MAX_POLICY_SIZE].into(), -2.0, -3.0, 2.0)
                .with_ownership_logits([1.0; MAX_BOARD_AREA].into());
            let map_address = fresh.ownership.as_ref().unwrap().as_ptr();
            fresh.process_with_cached_values_in_place(player, &cached, MAX_BOARD_DIM);
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
    fn ownership_processing_and_cache_upgrades_use_exact_size_maps() {
        use crate::{
            game::{game_state::GameState, rules::Rules},
            inference::policy::legal_mask,
        };
        for board_dim in 1..=MAX_BOARD_DIM {
            let state = GameState::new(Rules {
                board_dim,
                ..Rules::default()
            });
            let mut legal = legal_mask(&state);
            legal[0] = false; // Ownership still applies to occupied/illegal active board_area.
            for player in [Player::Black, Player::White] {
                let mut cached = NNOutput::from_raw(
                    vec![0.0; board_dim * board_dim + 1].into_boxed_slice(),
                    1.0,
                    2.0,
                    -1.0,
                );
                cached.process_in_place(player, &legal, board_dim);
                for upgrade in [false, true] {
                    let mut output = NNOutput::from_raw(
                        vec![0.0; board_dim * board_dim + 1].into_boxed_slice(),
                        1.0,
                        2.0,
                        -1.0,
                    )
                    .with_ownership_logits(vec![2.0; board_dim * board_dim].into());
                    let address = output.ownership.as_ref().unwrap().as_ptr();
                    let policy_address = output.policy.as_ptr();
                    if upgrade {
                        output.process_with_cached_values_in_place(player, &cached, board_dim);
                    } else {
                        output.process_in_place(player, &legal, board_dim);
                    }
                    assert_eq!(output.policy_probs(), cached.policy_probs());
                    assert_eq!(output.policy_probs().as_ptr(), policy_address);
                    assert_eq!(output.policy_probs().len(), board_dim * board_dim + 1);
                    assert_eq!(output.policy_probs()[0], 0.0);
                    assert!(
                        (output
                            .policy_probs()
                            .iter()
                            .copied()
                            .map(f64::from)
                            .sum::<f64>()
                            - 1.0)
                            .abs()
                            < 1e-6
                    );
                    let ownership = output.white_ownership().unwrap();
                    assert_eq!(ownership.as_ptr(), address);
                    assert_eq!(ownership.len(), board_dim * board_dim);
                    let expected = signed_value_to_white(2.0_f32.tanh(), player);
                    assert!(ownership.iter().all(|&value| value == expected));
                }
            }
        }
    }

    #[test]
    fn ownership_is_optional_and_converted_from_logits_to_white() {
        for player in [Player::Black, Player::White] {
            let mut output = NNOutput::from_raw([0.0; MAX_POLICY_SIZE].into(), 0.0, 0.0, 0.0)
                .with_ownership_logits([1.0; MAX_BOARD_AREA].into());
            output.process_in_place(player, &[true; MAX_POLICY_SIZE], MAX_BOARD_DIM);
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
            assert!(Arc::ptr_eq(
                cloned.ownership.as_ref().unwrap(),
                output.ownership.as_ref().unwrap(),
            ));
            assert_ne!(cloned.policy.as_ptr(), output.policy.as_ptr());
            assert_ne!(cloned.policy_probs()[0], output.policy_probs()[0]);
        }
        let mut output = NNOutput::from_raw([0.0; MAX_POLICY_SIZE].into(), 0.0, 0.0, 0.0);
        output.process_in_place(Player::Black, &[true; MAX_POLICY_SIZE], MAX_BOARD_DIM);
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
        policy_logits: [f32; MAX_POLICY_SIZE],
        win_logit: f32,
        raw_score_mean: f32,
        raw_score_stdev_logit: f32,
        next_player: Player,
        legal_mask: &[bool; MAX_POLICY_SIZE],
    ) -> NNOutput {
        let mut output = NNOutput::from_raw(
            policy_logits.into(),
            win_logit,
            raw_score_mean,
            raw_score_stdev_logit,
        );
        assert!(!output.is_processed());
        output.process_in_place(next_player, legal_mask, MAX_BOARD_DIM);
        assert!(output.is_processed());
        output
    }

    #[test]
    fn masked_softmax_zeros_illegal_moves_and_normalizes_legal_moves() {
        let policy_logits = [0.0; MAX_POLICY_SIZE];
        let mut legal_mask = [false; MAX_POLICY_SIZE];
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
        let mut policy_logits = [0.0; MAX_POLICY_SIZE];
        policy_logits[4] = 1_000.0;
        policy_logits[9] = -1_000.0;
        let mut legal_mask = [false; MAX_POLICY_SIZE];
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
        let policy_logits = [0.0; MAX_POLICY_SIZE];
        let mut legal_mask = [false; MAX_POLICY_SIZE];
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
        let _ = NNOutput::from_raw([0.0; MAX_POLICY_SIZE].into(), f32::NAN, 0.0, 0.0);
    }
}
