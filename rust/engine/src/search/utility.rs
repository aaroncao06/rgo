use std::{f64::consts::PI, sync::OnceLock};

use crate::{game::board::BOARD_SIZE, search::search_params::SearchParams};

const EXTRA_SCORE_DISTRIBUTION_RADIUS: usize = 60;
// KataGo builds this table at its compile-time maximum board size, currently
// 19, and rescales lookups for the actual board area.
const TABLE_ASSUMED_BOARD_SIZE: usize = 19;
const TABLE_MEAN_RADIUS: usize =
    TABLE_ASSUMED_BOARD_SIZE * TABLE_ASSUMED_BOARD_SIZE + EXTRA_SCORE_DISTRIBUTION_RADIUS;
const TABLE_MEAN_LEN: usize = TABLE_MEAN_RADIUS * 2;
const TABLE_STDEV_LEN: usize =
    TABLE_ASSUMED_BOARD_SIZE * TABLE_ASSUMED_BOARD_SIZE + EXTRA_SCORE_DISTRIBUTION_RADIUS;
const STEPS_PER_UNIT: i32 = 10;
const BOUND_STDEVS: i32 = 5;

static EXPECTED_SCORE_VALUE_TABLE: OnceLock<Box<[f64]>> = OnceLock::new();

pub(crate) fn white_utility(
    white_win_probability: f64,
    white_score_mean: f64,
    white_score_mean_sq: f64,
    recent_score_center: f64,
    params: SearchParams,
) -> f64 {
    debug_assert!((0.0..=1.0).contains(&white_win_probability));

    let win_loss_utility = (2.0 * white_win_probability - 1.0) * params.win_loss_utility_factor;
    let score_stdev = score_stdev(white_score_mean, white_score_mean_sq);
    let sqrt_board_area = BOARD_SIZE as f64;
    let static_score_value =
        expected_white_score_value(white_score_mean, score_stdev, 0.0, 2.0, sqrt_board_area);
    let dynamic_score_value = expected_white_score_value(
        white_score_mean,
        score_stdev,
        recent_score_center,
        params.dynamic_score_center_scale,
        sqrt_board_area,
    );

    win_loss_utility
        + static_score_value * params.static_score_utility_factor
        + dynamic_score_value * params.dynamic_score_utility_factor
}

pub(crate) fn recent_score_center(expected_score: f64, params: SearchParams) -> f64 {
    let mut center = expected_score * (1.0 - params.dynamic_score_center_zero_weight);
    let cap = BOARD_SIZE as f64 * params.dynamic_score_center_scale;
    center = center.clamp(expected_score - cap, expected_score + cap);
    center
}

fn score_stdev(score_mean: f64, score_mean_sq: f64) -> f64 {
    (score_mean_sq - score_mean * score_mean).max(0.0).sqrt()
}

fn score_value(score: f64, center: f64, scale: f64, sqrt_board_area: f64) -> f64 {
    ((score - center) / (scale * sqrt_board_area)).atan() * (2.0 / PI)
}

fn expected_white_score_value(
    white_score_mean: f64,
    white_score_stdev: f64,
    center: f64,
    scale: f64,
    sqrt_board_area: f64,
) -> f64 {
    let table = EXPECTED_SCORE_VALUE_TABLE.get_or_init(build_expected_score_value_table);
    let scale_factor = TABLE_ASSUMED_BOARD_SIZE as f64 / (scale * sqrt_board_area);
    let mean_scaled = (white_score_mean - center) * scale_factor;
    let stdev_scaled = white_score_stdev * scale_factor;

    let mean_rounded = mean_scaled.round();
    let stdev_floored = stdev_scaled.floor();
    let mean_index = mean_rounded as isize + TABLE_MEAN_RADIUS as isize;
    let stdev_index = stdev_floored as isize;

    let mean_index_0 = mean_index.clamp(0, TABLE_MEAN_LEN as isize - 1) as usize;
    let mean_index_1 = (mean_index + 1).clamp(0, TABLE_MEAN_LEN as isize - 1) as usize;
    let stdev_index_0 = stdev_index.clamp(0, TABLE_STDEV_LEN as isize - 1) as usize;
    let stdev_index_1 = (stdev_index + 1).clamp(0, TABLE_STDEV_LEN as isize - 1) as usize;

    let mean_fraction = mean_scaled - mean_rounded + 0.5;
    let stdev_fraction = stdev_scaled - stdev_floored;

    let value_00 = table[mean_index_0 * TABLE_STDEV_LEN + stdev_index_0];
    let value_01 = table[mean_index_0 * TABLE_STDEV_LEN + stdev_index_1];
    let value_10 = table[mean_index_1 * TABLE_STDEV_LEN + stdev_index_0];
    let value_11 = table[mean_index_1 * TABLE_STDEV_LEN + stdev_index_1];

    let mean_0 = value_00 + stdev_fraction * (value_01 - value_00);
    let mean_1 = value_10 + stdev_fraction * (value_11 - value_10);
    mean_0 + mean_fraction * (mean_1 - mean_0)
}

fn build_expected_score_value_table() -> Box<[f64]> {
    let min_stdev_step = -BOUND_STDEVS * STEPS_PER_UNIT;
    let max_stdev_step = BOUND_STDEVS * STEPS_PER_UNIT;
    let normal_weights: Vec<f64> = (min_stdev_step..=max_stdev_step)
        .map(|step| {
            let standard_deviations = f64::from(step) / f64::from(STEPS_PER_UNIT);
            (-0.5 * standard_deviations * standard_deviations).exp()
        })
        .collect();

    let min_score_step = -(TABLE_MEAN_RADIUS as i32 * STEPS_PER_UNIT
        + STEPS_PER_UNIT / 2
        + BOUND_STDEVS * TABLE_STDEV_LEN as i32 * STEPS_PER_UNIT);
    let max_score_step = -min_score_step;
    let score_values: Vec<f64> = (min_score_step..=max_score_step)
        .map(|step| {
            let score = f64::from(step) / f64::from(STEPS_PER_UNIT);
            score_value(score, 0.0, 1.0, TABLE_ASSUMED_BOARD_SIZE as f64)
        })
        .collect();

    let mut table = vec![0.0; TABLE_MEAN_LEN * TABLE_STDEV_LEN];
    for mean_index in 0..TABLE_MEAN_LEN {
        let mean_steps =
            (mean_index as i32 - TABLE_MEAN_RADIUS as i32) * STEPS_PER_UNIT - STEPS_PER_UNIT / 2;
        for stdev_index in 0..TABLE_STDEV_LEN {
            let mut weight_sum = 0.0;
            let mut weighted_value_sum = 0.0;
            for (offset, weight) in (min_stdev_step..=max_stdev_step).zip(&normal_weights) {
                let score_steps = mean_steps + stdev_index as i32 * offset;
                let value = score_values[(score_steps - min_score_step) as usize];
                weight_sum += weight;
                weighted_value_sum += weight * value;
            }
            table[mean_index * TABLE_STDEV_LEN + stdev_index] = weighted_value_sum / weight_sum;
        }
    }
    table.into_boxed_slice()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-9,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn score_value_is_bounded_symmetric_and_centered() {
        assert_close(score_value(0.0, 0.0, 2.0, 9.0), 0.0);
        assert_close(
            score_value(18.0, 0.0, 2.0, 9.0),
            -score_value(-18.0, 0.0, 2.0, 9.0),
        );
        assert!(score_value(1e9, 0.0, 2.0, 9.0) < 1.0);
        assert!(score_value(-1e9, 0.0, 2.0, 9.0) > -1.0);
    }

    #[test]
    fn expected_score_value_is_symmetric() {
        let positive = expected_white_score_value(10.0, 3.0, 0.0, 2.0, 9.0);
        let negative = expected_white_score_value(-10.0, 3.0, 0.0, 2.0, 9.0);
        assert!((positive + negative).abs() < 1e-9);
    }

    #[test]
    fn canonical_utility_combines_win_and_score_values() {
        let params = SearchParams::KATAGO_SELFPLAY8_MAIN_B18;
        assert_close(white_utility(0.5, 0.0, 0.0, 0.0, params), 0.0);
        assert!(white_utility(0.75, 5.0, 25.0, 0.0, params) > 0.5);
        assert!(white_utility(0.25, -5.0, 25.0, 0.0, params) < -0.5);
    }

    #[test]
    fn recent_score_center_is_pulled_toward_zero_and_capped() {
        let params = SearchParams::KATAGO_SELFPLAY8_MAIN_B18;
        assert_close(recent_score_center(8.0, params), 6.0);
        assert_close(recent_score_center(40.0, params), 35.5);
        assert_close(recent_score_center(-40.0, params), -35.5);
    }
}
