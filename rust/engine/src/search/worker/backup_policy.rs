//! Complete parent-estimate policy: transposition contributions, value-dependent
//! weighting, and statistics aggregation. The backup path walk stays in the worker.

use super::SearchWorker;
use crate::{
    game::board::Player,
    search::{
        node::{SearchNode, SearchStats},
        node_store::NodeStore,
        utility::white_utility,
    },
};
use std::sync::OnceLock;

pub(super) struct ChildContribution {
    white_win: f64,
    white_score: f64,
    white_score_mean_sq: f64,
    white_utility: f64,
    white_utility_mean_sq: f64,
    raw_weight: f64,
    adjusted_weight: f64,
    weight_sq_sum: f64,
}

impl<N: NodeStore> SearchWorker<N> {
    pub(super) fn recompute_node_stats(&mut self, node: &mut SearchNode, player: Player) {
        let contributions = &mut self.child_contributions;
        contributions.clear();
        let mut total_child_weight = 0.0;
        // Like KataGo, count the completed playout independently of child
        // statistics. Cycles can make edge visits exceed a child's visits.
        let visits = node.visits() + 1;

        for edge in node.edges() {
            if edge.visits() <= 0 {
                continue;
            }
            // A self-loop must read through the current borrow, not create an
            // alias from its raw pointer while `node` is mutably borrowed.
            let child = if std::ptr::eq(edge.child().as_ptr(), node) {
                &*node
            } else {
                // SAFETY: other graph nodes remain live and are not mutably
                // borrowed during this parent's recomputation.
                unsafe { edge.child().as_ref() }
            };
            if child.visits() <= 0 || child.weight_sum() <= 0.0 {
                continue;
            }

            // A transposed child may have visits from other parent edges. Only the
            // fraction attributable to this edge contributes to this parent.
            let edge_fraction = f64::from(edge.visits()) / f64::from(child.visits().max(1));
            let raw_weight = child.weight_sum() * edge_fraction;
            total_child_weight += raw_weight;
            contributions.push(ChildContribution {
                white_win: child.white_win_rate(),
                white_score: child.white_score_mean(),
                white_score_mean_sq: child.white_score_mean_sq(),
                white_utility: child.white_utility(),
                white_utility_mean_sq: child.white_utility_mean_sq(),
                raw_weight,
                adjusted_weight: raw_weight,
                // Scaling every sample's weight scales its square quadratically.
                weight_sq_sum: child.weight_sq_sum() * edge_fraction * edge_fraction,
            });
        }

        adjust_weights(
            contributions,
            total_child_weight,
            player,
            self.params.value_weight_exponent,
        );

        let output = node.nn_output();
        let direct_white_win = f64::from(output.white_win_prob());
        let direct_white_score = f64::from(output.white_score_mean());
        let direct_white_score_mean_sq = f64::from(output.white_score_mean_sq());
        let direct_white_utility = white_utility(
            direct_white_win,
            direct_white_score,
            direct_white_score_mean_sq,
            self.recent_score_center,
            self.params,
        );

        let mut white_win_sum = direct_white_win;
        let mut white_score_sum = direct_white_score;
        let mut white_score_mean_sq_sum = direct_white_score_mean_sq;
        let mut white_utility_sum = direct_white_utility;
        let mut white_utility_sq_sum = direct_white_utility * direct_white_utility;
        let mut weight_sq_sum = 1.0;
        for child in contributions.iter() {
            let weight = child.adjusted_weight;
            let weight_scaling = weight / child.raw_weight;
            white_win_sum += weight * child.white_win;
            white_score_sum += weight * child.white_score;
            white_score_mean_sq_sum += weight * child.white_score_mean_sq;
            white_utility_sum += weight * child.white_utility;
            white_utility_sq_sum += weight * child.white_utility_mean_sq;
            weight_sq_sum += weight_scaling * weight_scaling * child.weight_sq_sum;
        }

        node.replace_stats(SearchStats {
            visits,
            white_win_sum,
            white_score_sum,
            white_score_mean_sq_sum,
            white_utility_sum,
            white_utility_sq_sum,
            weight_sum: 1.0 + total_child_weight,
            weight_sq_sum,
        });
    }
}

fn adjust_weights(
    contributions: &mut [ChildContribution],
    total_child_weight: f64,
    player: Player,
    exponent: f64,
) {
    if total_child_weight > 0.0 && exponent != 0.0 {
        let simple_utility = contributions
            .iter()
            .map(|child| {
                let utility = match player {
                    Player::White => child.white_utility,
                    Player::Black => -child.white_utility,
                };
                utility * child.raw_weight
            })
            .sum::<f64>()
            / total_child_weight;

        let mut adjusted_weight_sum = 0.0;
        for child in contributions.iter_mut() {
            let utility = match player {
                Player::White => child.white_utility,
                Player::Black => -child.white_utility,
            };
            let precision = 1.5 * child.raw_weight.sqrt();
            let stdev = (1e-8 + 1.0 / precision).sqrt();
            let z = (utility - simple_utility) / stdev;
            let value_weight = (student_t_cdf_degrees_3(z) + 0.0001).powf(exponent);
            child.adjusted_weight *= value_weight;
            adjusted_weight_sum += child.adjusted_weight;
        }

        let normalization = total_child_weight / adjusted_weight_sum;
        for child in contributions.iter_mut() {
            child.adjusted_weight *= normalization;
        }
    }
}

// Match KataGo's DistributionTable: 2,000 samples across [-50, 50], forced
// endpoint probabilities of 0 and 1, and linear interpolation between samples.
fn student_t_cdf_degrees_3(value: f64) -> f64 {
    const SIZE: usize = 2000;
    static TABLE: OnceLock<[f64; SIZE]> = OnceLock::new();
    let table = TABLE.get_or_init(|| {
        std::array::from_fn(|i| {
            if i == 0 {
                0.0
            } else if i == SIZE - 1 {
                1.0
            } else {
                let z = -50.0 + i as f64 * 100.0 / (SIZE - 1) as f64;
                // Closed-form Student-t(3) CDF used only to initialize the table.
                let sqrt_three = 3.0_f64.sqrt();
                0.5 + (z.atan2(sqrt_three) + z * sqrt_three / (z * z + 3.0)) / std::f64::consts::PI
            }
        })
    });
    let position = (SIZE - 1) as f64 * (value + 50.0) / 100.0;
    if position <= 0.0 {
        return 0.0;
    }
    let index = position as usize;
    if index >= SIZE - 1 {
        return 1.0;
    }
    let fraction = position - index as f64;
    table[index] + fraction * (table[index + 1] - table[index])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn student_t_cdf_degrees_3_is_centered_and_symmetric() {
        assert!((student_t_cdf_degrees_3(0.0) - 0.5).abs() < 1e-12);
        let positive = student_t_cdf_degrees_3(2.0);
        let negative = student_t_cdf_degrees_3(-2.0);
        assert!((positive + negative - 1.0).abs() < 1e-12);
        assert!(positive > 0.5);
    }

    #[test]
    fn student_t_table_clamps_and_linearly_interpolates() {
        assert_eq!(student_t_cdf_degrees_3(-100.0), 0.0);
        assert_eq!(student_t_cdf_degrees_3(-50.0), 0.0);
        assert_eq!(student_t_cdf_degrees_3(50.0), 1.0);
        assert_eq!(student_t_cdf_degrees_3(100.0), 1.0);
        let left = -50.0 + 1030.0 * 100.0 / 1999.0;
        let right = -50.0 + 1031.0 * 100.0 / 1999.0;
        let midpoint = (left + right) * 0.5;
        let expected = (student_t_cdf_degrees_3(left) + student_t_cdf_degrees_3(right)) * 0.5;
        assert!((student_t_cdf_degrees_3(midpoint) - expected).abs() < 1e-12);
        // Reference Student-t(3) probability at x=2 (allow table interpolation error).
        assert!((student_t_cdf_degrees_3(2.0) - 0.9303370157205785).abs() < 1e-4);
    }
}
