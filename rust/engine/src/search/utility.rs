use std::sync::OnceLock;

mod score_value_table;
use score_value_table::build_expected_score_value_table;
#[cfg(test)]
use score_value_table::score_value;
use score_value_table::{
    TABLE_ASSUMED_BOARD_SIZE, TABLE_MEAN_LEN, TABLE_MEAN_RADIUS, TABLE_STDEV_LEN,
};

use crate::{game::board::BOARD_SIZE, search::search_params::SearchParams};

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

/// Shift the score mean without changing its uncertainty. Root ending bonuses
/// are selection-only adjustments, not additional samples or training targets.
pub(crate) fn score_utility_diff(
    mean: f64,
    mean_sq: f64,
    delta: f64,
    center: f64,
    params: SearchParams,
) -> f64 {
    if delta == 0.0 {
        return 0.0;
    }
    let stdev = score_stdev(mean, mean_sq);
    let size = BOARD_SIZE as f64;
    let static_diff = expected_white_score_value(mean + delta, stdev, 0.0, 2.0, size)
        - expected_white_score_value(mean, stdev, 0.0, 2.0, size);
    let dynamic_diff = expected_white_score_value(
        mean + delta,
        stdev,
        center,
        params.dynamic_score_center_scale,
        size,
    ) - expected_white_score_value(
        mean,
        stdev,
        center,
        params.dynamic_score_center_scale,
        size,
    );
    static_diff * params.static_score_utility_factor
        + dynamic_diff * params.dynamic_score_utility_factor
}

fn score_stdev(score_mean: f64, score_mean_sq: f64) -> f64 {
    (score_mean_sq - score_mean * score_mean).max(0.0).sqrt()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ending_score_shift_preserves_predicted_variance() {
        let params = SearchParams::KATAGO_SELFPLAY8_MAIN_B18;
        let mean = 3.0;
        let variance = 16.0;
        let delta = 0.5;
        let center = 2.0;
        let expected = white_utility(
            0.6,
            mean + delta,
            (mean + delta) * (mean + delta) + variance,
            center,
            params,
        ) - white_utility(0.6, mean, mean * mean + variance, center, params);
        assert!(
            (score_utility_diff(mean, mean * mean + variance, delta, center, params) - expected)
                .abs()
                < 1e-14
        );
        assert_eq!(
            score_utility_diff(mean, mean * mean + variance, 0.0, center, params),
            0.0
        );
    }

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
