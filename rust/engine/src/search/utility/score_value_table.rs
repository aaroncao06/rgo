use std::f64::consts::PI;

pub(super) const EXTRA_SCORE_DISTRIBUTION_RADIUS: usize = 60;
// KataGo builds this table at its compile-time maximum board size, currently
// 19, and rescales lookups for the actual board area.
pub(super) const TABLE_ASSUMED_BOARD_DIM: usize = 19;
pub(super) const TABLE_MEAN_RADIUS: usize =
    TABLE_ASSUMED_BOARD_DIM * TABLE_ASSUMED_BOARD_DIM + EXTRA_SCORE_DISTRIBUTION_RADIUS;
pub(super) const TABLE_MEAN_LEN: usize = TABLE_MEAN_RADIUS * 2;
pub(super) const TABLE_STDEV_LEN: usize =
    TABLE_ASSUMED_BOARD_DIM * TABLE_ASSUMED_BOARD_DIM + EXTRA_SCORE_DISTRIBUTION_RADIUS;
pub(super) const STEPS_PER_UNIT: i32 = 10;
pub(super) const BOUND_STDEVS: i32 = 5;

pub(super) fn score_value(score: f64, center: f64, scale: f64, sqrt_board_area: f64) -> f64 {
    ((score - center) / (scale * sqrt_board_area)).atan() * (2.0 / PI)
}

pub(super) fn build_expected_score_value_table() -> Box<[f64]> {
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
            score_value(score, 0.0, 1.0, TABLE_ASSUMED_BOARD_DIM as f64)
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
