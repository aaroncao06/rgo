#[derive(Debug, Clone, Copy)]
pub(crate) struct SearchParams {
    pub(crate) win_loss_utility_factor: f64,
    pub(crate) static_score_utility_factor: f64,
    pub(crate) dynamic_score_utility_factor: f64,
    pub(crate) dynamic_score_center_zero_weight: f64,
    pub(crate) dynamic_score_center_scale: f64,
    pub(crate) cpuct_exploration: f64,
    pub(crate) cpuct_exploration_log: f64,
    pub(crate) cpuct_exploration_base: f64,
    pub(crate) fpu_reduction_max: f64,
    pub(crate) root_fpu_reduction_max: f64,
    pub(crate) fpu_parent_weight_by_visited_policy_pow: f64,
}

impl SearchParams {
    /// Values from KataGo's shipped `selfplay8mainb18.cfg`, including defaults
    /// from `Setup::loadParams` for fields omitted by that file.
    pub(crate) const KATAGO_SELFPLAY8_MAIN_B18: Self = Self {
        win_loss_utility_factor: 1.0,
        static_score_utility_factor: 0.05,
        dynamic_score_utility_factor: 0.30,
        dynamic_score_center_zero_weight: 0.25,
        dynamic_score_center_scale: 0.50,
        cpuct_exploration: 1.05,
        cpuct_exploration_log: 0.28,
        cpuct_exploration_base: 500.0,
        fpu_reduction_max: 0.2,
        root_fpu_reduction_max: 0.0,
        fpu_parent_weight_by_visited_policy_pow: 2.0,
    };
}
