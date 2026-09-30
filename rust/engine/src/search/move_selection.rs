//! Pure move-selection formulas ported from KataGo's search helpers.

use rand::RngExt;

/// Invert the exploration score, cap by the existing weight, and round up.
pub(super) fn reduced_weight(
    weight: f64,
    self_utility: f64,
    policy: f64,
    explore_scaling: f64,
    best_selection_value: f64,
) -> f64 {
    if weight <= 0.0 {
        return 0.0;
    }
    let gap = best_selection_value - self_utility;
    let inverse = if policy < 0.0 {
        0.0
    } else if gap <= 0.0 {
        1e100
    } else {
        (explore_scaling * policy / gap - 1.0).max(0.0)
    };
    weight.min(inverse).ceil()
}

/// Compute the utility LCB with KataGo's variance prior and effective sample size.
///
/// The caller scales BOTH weight sums linearly by edge_visits / child_visits,
/// including `weight_sq_sum` (see KataGo's `NodeStats::childWeightSq`). Utility
/// moments are unscaled; `self_utility` is the mean from the mover's perspective.
/// Any ending-score bonus must be added to the returned LCB separately, so it
/// does not affect the variance calculated from the original utility moments.
pub(super) fn lcb_and_radius(
    self_utility: f64,
    utility_mean_sq: f64,
    weight_sum: f64,
    weight_sq_sum: f64,
    utility_radius: f64,
    lcb_stdevs: f64,
) -> (f64, f64) {
    if weight_sum <= 0.0 || weight_sq_sum <= 0.0 {
        let radius = 2.0 * utility_radius * lcb_stdevs;
        return (-radius, radius);
    }
    let ess = weight_sum * weight_sum / weight_sq_sum;
    let prior_weight = weight_sum / (ess * ess * ess);
    let mean_sq = utility_mean_sq.max(self_utility * self_utility + 1e-8);
    let mean_sq = (mean_sq * weight_sum
        + (mean_sq + utility_radius * utility_radius) * prior_weight)
        / (weight_sum + prior_weight);
    let weight_sum = weight_sum + prior_weight;
    let weight_sq_sum = weight_sq_sum + prior_weight * prior_weight;
    let ess = weight_sum * weight_sum / weight_sq_sum;
    let variance = mean_sq - self_utility * self_utility;
    let radius = (variance / ess).sqrt() * lcb_stdevs;
    (self_utility - radius, radius)
}

/// Boost the best eligible LCB, using KataGo's `useNonBuggyLcb = true` behavior.
/// `reference_weight` is the best child's weight before retrospective reduction.
pub(super) fn adjust_lcb(
    weights: &mut [f64],
    lcbs: &[(f64, f64)],
    reference_weight: f64,
    min_prop: f64,
) {
    debug_assert_eq!(weights.len(), lcbs.len());
    let mut best_lcb = -1e10;
    let mut best_index = None;
    for (i, (&weight, &(lcb, _))) in weights.iter().zip(lcbs).enumerate() {
        if weight > 0.0 && weight >= min_prop * reference_weight && lcb > best_lcb {
            best_lcb = lcb;
            best_index = Some(i);
        }
    }
    if let Some(best_index) = best_index {
        let mut adjusted_weight = weights[best_index];
        for (i, (&weight, &(lcb, radius))) in weights.iter().zip(lcbs).enumerate() {
            if i == best_index {
                continue;
            }
            let excess = best_lcb - lcb;
            if excess < 0.0 {
                continue;
            }
            let radius_factor = (radius + excess) / (radius + 0.20 * excess);
            let lower_bound = radius_factor * radius_factor * weight;
            if lower_bound > adjusted_weight {
                adjusted_weight = lower_bound;
            }
        }
        weights[best_index] = adjusted_weight;
    }
}

/// Prune before subtracting; both amounts are capped at maximum weight / 64.
pub(super) fn prune_weights(weights: &mut [f64], subtract: f64, prune: f64) {
    let max_weight = weights.iter().copied().fold(0.0, f64::max);
    let subtract = subtract.min(max_weight / 64.0);
    let prune = prune.min(max_weight / 64.0);
    for weight in weights {
        *weight = if *weight < prune {
            0.0
        } else {
            (*weight - subtract).max(0.0)
        };
    }
}

/// Interpolate on a square board whose side length is `board_size`.
pub(super) fn temperature(
    turn_number: usize,
    board_size: usize,
    early: f64,
    late: f64,
    halflife: f64,
) -> f64 {
    let raw_halflives = turn_number as f64 / halflife;
    let halflives = raw_halflives * 19.0 / board_size as f64;
    late + (early - late) * 0.5_f64.powf(halflives)
}

/// Sample with `onlyBelowProb = 1`, ignoring nonpositive weights.
/// Requires finite weights, at least one positive weight, and finite temperature.
/// Temperatures at or below 1e-4 select the first maximum without consuming RNG.
pub(super) fn sample_index<R: rand::Rng + ?Sized>(
    weights: &[f64],
    temperature: f64,
    rng: &mut R,
) -> usize {
    debug_assert!(temperature.is_finite());
    let mut max_weight = 0.0;
    let mut best_index = 0;
    for (i, &weight) in weights.iter().enumerate() {
        debug_assert!(weight.is_finite());
        if weight > max_weight {
            max_weight = weight;
            best_index = i;
        }
    }
    debug_assert!(max_weight > 0.0);
    if temperature <= 1e-4 {
        return best_index;
    }
    let log_max = max_weight.ln();
    let scaled: Vec<f64> = weights
        .iter()
        .map(|&weight| {
            if weight <= 0.0 {
                0.0
            } else {
                ((weight.ln() - log_max) / temperature).exp()
            }
        })
        .collect();
    let mut remaining = rng.random::<f64>() * scaled.iter().sum::<f64>();
    let mut last_positive = best_index;
    for (i, &weight) in scaled.iter().enumerate() {
        if weight > 0.0 {
            last_positive = i;
            if remaining < weight {
                return i;
            }
            remaining -= weight;
        }
    }
    // Rounding at the upper endpoint must never select a zero-weight move.
    last_positive
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::SmallRng};

    fn close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1e-12 * expected.abs().max(1.0),
            "{actual} != {expected}"
        );
    }

    #[test]
    fn inverse_reduction_caps_clamps_and_rounds() {
        // 8 * 0.5 / (1.5 - 0.5) - 1 = 3.
        assert_eq!(reduced_weight(20.0, 0.5, 0.5, 8.0, 1.5), 3.0);
        assert_eq!(reduced_weight(2.2, 0.5, 0.5, 8.0, 1.5), 3.0);
        assert_eq!(reduced_weight(20.0, 0.5, 0.55, 8.0, 1.5), 4.0);
        assert_eq!(reduced_weight(20.0, 0.5, 0.0, 8.0, 1.5), 0.0);
        assert_eq!(reduced_weight(20.0, 0.5, -1.0, 8.0, 0.5), 0.0);
        assert_eq!(reduced_weight(20.0, 0.5, 0.5, 8.0, 0.5), 20.0);
        assert_eq!(reduced_weight(1e101, 0.5, 0.5, 8.0, 0.0), 1e100);
        assert_eq!(reduced_weight(0.0, 0.5, 0.5, 8.0, 1.5), 0.0);
    }

    #[test]
    fn lcb_hand_calculated_prior_and_ess() {
        // ESS=2, prior=1/4, adjusted second moment=17/18,
        // adjusted ESS=27/11, variance=25/36; radius^2=275/243.
        let (lcb, radius) = lcb_and_radius(0.5, 0.5, 2.0, 2.0, 2.0, 2.0);
        let expected = (275.0_f64 / 243.0).sqrt();
        close(radius, expected);
        close(lcb, 0.5 - expected);
        let (negative_lcb, negative_radius) = lcb_and_radius(-0.5, 0.5, 2.0, 2.0, 2.0, 2.0);
        close(negative_radius, expected);
        close(negative_lcb, -0.5 - expected);
    }

    #[test]
    fn lcb_matches_native_katago_reference_fixtures() {
        // Produced by the local KataGo getSelfUtilityLCBAndRadius function,
        // with ending bonuses disabled and canonical utility radius / stdevs.
        // A transposed child with raw sums (100,160) through 25/100 visits.
        let (lcb, radius) = lcb_and_radius(0.2, 0.29, 25.0, 40.0, 1.35, 5.0);
        close(lcb, -0.43289383658508801);
        close(radius, 0.63289383658508802);
        let (lcb, radius) = lcb_and_radius(-0.2, 0.29, 25.0, 40.0, 1.35, 5.0);
        close(lcb, -0.83289383658508798);
        close(radius, 0.63289383658508802);
        let (lcb, radius) = lcb_and_radius(0.0, 0.0, 1.0, 1.0, 1.35, 5.0);
        close(lcb, -3.3750000185185187);
        close(radius, 3.3750000185185187);
        assert_eq!(lcb_and_radius(0.0, 0.0, 0.0, 0.0, 1.35, 5.0), (-13.5, 13.5));
    }

    #[test]
    fn lcb_zero_weights_and_variance_floor() {
        for (sum, sq_sum) in [(0.0, 1.0), (1.0, 0.0), (-1.0, 1.0)] {
            assert_eq!(
                lcb_and_radius(0.5, 0.5, sum, sq_sum, 2.0, 3.0),
                (-12.0, 12.0)
            );
        }
        // ESS=1 becomes 2; prior adds 1/2 to the floored variance.
        let (lcb, radius) = lcb_and_radius(0.0, -1.0, 1.0, 1.0, 1.0, 2.0);
        close(radius, (1.0_f64 + 2e-8).sqrt());
        close(lcb, -radius);
    }

    #[test]
    fn caller_scales_squared_weight_linearly_by_edge_fraction() {
        let edge_fraction = 0.25;
        // Raw sums (8,8) become (2,2), giving ESS=2, not ESS=8.
        let actual = lcb_and_radius(0.5, 0.5, 8.0 * edge_fraction, 8.0 * edge_fraction, 2.0, 2.0);
        close(actual.1, (275.0_f64 / 243.0).sqrt());
        let incorrectly_squared = lcb_and_radius(
            0.5,
            0.5,
            8.0 * edge_fraction,
            8.0 * edge_fraction * edge_fraction,
            2.0,
            2.0,
        );
        assert!(actual.1 > incorrectly_squared.1);
    }

    #[test]
    fn lcb_boost_includes_index_zero_and_threshold_equality() {
        let mut weights = [10.0, 100.0, 9.0, 0.0];
        // Index 2 has a better LCB but misses the threshold, and is skipped
        // when computing the bonus. Index 3 cannot be a candidate either.
        adjust_lcb(
            &mut weights,
            &[(2.0, 1.0), (0.0, 2.0), (3.0, 1.0), (4.0, 1.0)],
            100.0,
            0.1,
        );
        // Competitor radius factor = (2+2)/(2+0.2*2) = 5/3.
        close(weights[0], 2500.0 / 9.0);
        assert_eq!(&weights[1..], &[100.0, 9.0, 0.0]);
    }

    #[test]
    fn lcb_ties_no_candidate_and_bonus_cap() {
        let mut weights = [1.0, 10.0];
        adjust_lcb(&mut weights, &[(2.0, 1.0), (2.0, 1.0)], 10.0, 0.1);
        assert_eq!(weights, [10.0, 10.0]);
        let mut weights = [1.0, 10.0];
        adjust_lcb(&mut weights, &[(0.0, 1.0), (1.0, 1.0)], 100.0, 0.2);
        assert_eq!(weights, [1.0, 10.0]);
        adjust_lcb(&mut weights, &[(0.0, 0.0), (1.0, 1.0)], 10.0, 0.1);
        assert_eq!(weights, [1.0, 25.0]);
        adjust_lcb(&mut [], &[], 0.0, 0.0);
    }

    #[test]
    fn pruning_caps_and_tests_threshold_before_subtraction() {
        let mut weights = [128.0, 2.0, 1.999, 0.0];
        prune_weights(&mut weights, 0.5, 50.0);
        assert_eq!(weights, [127.5, 1.5, 0.0, 0.0]);
        let mut weights = [128.0, 2.0, 1.0];
        prune_weights(&mut weights, 50.0, 0.0);
        assert_eq!(weights, [126.0, 0.0, 0.0]);
        prune_weights(&mut [], 1.0, 1.0);
    }

    #[test]
    fn temperature_scales_for_nine_by_nine() {
        close(temperature(0, 9, 1.0, 0.2, 19.0), 1.0);
        close(temperature(9, 9, 1.0, 0.2, 19.0), 0.6);
        close(temperature(18, 9, 1.0, 0.2, 19.0), 0.4);
        close(temperature(19, 19, 1.0, 0.2, 19.0), 0.6);
    }

    #[test]
    fn near_zero_temperature_uses_first_tie_without_randomness() {
        let mut rng = SmallRng::seed_from_u64(42);
        let mut untouched = rng.clone();
        for temp in [0.0, 1e-5, 1e-4] {
            assert_eq!(sample_index(&[0.0, 4.0, 4.0, 1.0], temp, &mut rng), 1);
            assert_eq!(sample_index(&[4.0, 4.0], temp, &mut rng), 0);
        }
        assert_eq!(rng.random::<f64>(), untouched.random::<f64>());
    }

    #[test]
    fn seeded_sampling_matches_hand_calculated_probabilities() {
        let mut rng = SmallRng::seed_from_u64(1234);
        let mut draws = rng.clone();
        let mut counts = [0; 2];
        // Temperature 2 transforms weights 1:9 to 1:3.
        for _ in 0..10_000 {
            let expected = if draws.random::<f64>() < 0.25 { 1 } else { 2 };
            let index = sample_index(&[-1.0, 1.0, 9.0, 0.0], 2.0, &mut rng);
            assert_eq!(index, expected);
            counts[index - 1] += 1;
        }
        assert!((counts[0] as f64 / 10_000.0 - 0.25).abs() < 0.02);
    }

    #[test]
    fn extreme_weights_and_temperatures_remain_stable() {
        let mut rng = SmallRng::seed_from_u64(7);
        for _ in 0..100 {
            assert_eq!(sample_index(&[1e-300, 1e300, 0.0], 0.00010001, &mut rng), 1);
            // The raw sum overflows, but normalization by the maximum does not.
            assert!(sample_index(&[f64::MAX, f64::MAX, 0.0], 1.0, &mut rng) < 2);
        }
        let mut draws = rng.clone();
        for _ in 0..100 {
            let expected = if draws.random::<f64>() < 0.5 { 0 } else { 1 };
            assert_eq!(
                sample_index(&[1e-300, 1e300, 0.0], 1e300, &mut rng),
                expected
            );
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn sampling_requires_a_positive_weight() {
        sample_index(&[0.0, -1.0], 1.0, &mut SmallRng::seed_from_u64(0));
    }
}
