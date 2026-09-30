#[derive(Debug, Clone, Copy, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SearchParams {
    pub win_loss_utility_factor: f64,
    pub static_score_utility_factor: f64,
    pub dynamic_score_utility_factor: f64,
    pub dynamic_score_center_zero_weight: f64,
    pub dynamic_score_center_scale: f64,
    pub cpuct_exploration: f64,
    pub cpuct_exploration_log: f64,
    pub cpuct_exploration_base: f64,
    pub fpu_reduction_max: f64,
    pub root_fpu_reduction_max: f64,
    pub fpu_parent_weight_by_visited_policy_pow: f64,
    pub root_desired_per_child_visits_coeff: f64,
    pub value_weight_exponent: f64,
    pub chosen_move_temperature_early: f64,
    pub chosen_move_temperature: f64,
    pub chosen_move_temperature_halflife: f64,
    pub chosen_move_subtract: f64,
    pub chosen_move_prune: f64,
    pub root_noise_enabled: bool,
    pub root_dirichlet_noise_total_concentration: f64,
    pub root_dirichlet_noise_weight: f64,
    pub root_policy_temperature_early: f64,
    pub root_policy_temperature: f64,
    pub root_ending_bonus_points: f64,
    pub root_prune_useless_moves: bool,
    pub use_lcb_for_selection: bool,
    pub lcb_stdevs: f64,
    pub min_visit_prop_for_lcb: f64,
}

impl Default for SearchParams {
    fn default() -> Self {
        Self::KATAGO_SELFPLAY8_MAIN_B18
    }
}

// KataGo's canonical config also averages four root symmetries. We defer that
// deliberately: randomized inference already samples orientations across
// self-play, while root averaging bypasses the cache and costs three extra root
// evaluations per move. Revisit it after measuring orientation bias and
// end-to-end inference throughput rather than assuming the tradeoff is useful.

// KataGo's canonical config also applies subtree-value bias. We defer it
// deliberately: it transfers search-discovered value errors between nodes
// using a handcrafted, transition-based 5x5 Go pattern. Matching local
// patterns need not have matching errors in different global positions, and
// graph transpositions make the first incoming transition arbitrarily choose
// the shared node's correction context. Revisit it only after measuring an
// isolated benefit that justifies this complexity and path dependence.

impl SearchParams {
    /// Check numeric/distribution requirements, not recommended tuning ranges.
    pub fn validate(&self) -> Result<(), &'static str> {
        let finite = [
            self.win_loss_utility_factor,
            self.static_score_utility_factor,
            self.dynamic_score_utility_factor,
            self.cpuct_exploration,
            self.cpuct_exploration_log,
            self.fpu_reduction_max,
            self.root_fpu_reduction_max,
            self.fpu_parent_weight_by_visited_policy_pow,
            self.root_desired_per_child_visits_coeff,
            self.value_weight_exponent,
            self.chosen_move_temperature_early,
            self.chosen_move_temperature,
            self.chosen_move_subtract,
            self.chosen_move_prune,
            self.root_ending_bonus_points,
            self.lcb_stdevs,
        ];
        if !finite.iter().all(|value| value.is_finite()) {
            return Err("search parameters must be finite");
        }
        // Visited policy mass can be zero; a negative power would produce infinity.
        if self.fpu_parent_weight_by_visited_policy_pow < 0.0 {
            return Err("FPU visited-policy exponent must be nonnegative");
        }
        let positive = [
            self.dynamic_score_center_scale,
            self.cpuct_exploration_base,
            self.chosen_move_temperature_halflife,
            self.root_policy_temperature_early,
            self.root_policy_temperature,
        ];
        if !positive
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
        {
            return Err(
                "search scales, exploration base, temperature halflife, and policy temperatures must be positive and finite",
            );
        }
        // Positivity is a distribution requirement, not a guarantee against
        // floating-point underflow. Extremely small concentrations (e.g. 0.001)
        // can make every gamma draw zero, causing a debug assertion or NaNs when
        // root noise is normalized. We defer a robust/log-space sampler: this
        // is not a practical concern at the default total concentration 10.83.
        if !self.root_dirichlet_noise_total_concentration.is_finite()
            || (self.root_noise_enabled && self.root_dirichlet_noise_total_concentration <= 0.0)
        {
            return Err(
                "noise concentration must be finite and positive when root noise is enabled",
            );
        }
        // LCB adjustment divides by combinations of confidence radii.
        if self.use_lcb_for_selection && self.lcb_stdevs <= 0.0 {
            return Err("LCB standard-deviation multiplier must be positive when LCB is enabled");
        }
        if ![
            self.dynamic_score_center_zero_weight,
            self.root_dirichlet_noise_weight,
            self.min_visit_prop_for_lcb,
        ]
        .iter()
        .all(|value| (0.0..=1.0).contains(value))
        {
            return Err("search mixing weights and LCB visit proportion must be in [0, 1]");
        }
        Ok(())
    }

    /// Values from KataGo's shipped `selfplay8mainb18.cfg`, including defaults
    /// from `Setup::loadParams` for fields omitted by that file.
    pub const KATAGO_SELFPLAY8_MAIN_B18: Self = Self {
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
        root_desired_per_child_visits_coeff: 2.0,
        value_weight_exponent: 0.5,
        chosen_move_temperature_early: 0.75,
        chosen_move_temperature: 0.15,
        chosen_move_temperature_halflife: 19.0,
        chosen_move_subtract: 0.0,
        chosen_move_prune: 1.0,
        root_noise_enabled: true,
        root_dirichlet_noise_total_concentration: 10.83,
        root_dirichlet_noise_weight: 0.25,
        root_policy_temperature_early: 1.5,
        root_policy_temperature: 1.1,
        root_ending_bonus_points: 0.5,
        root_prune_useless_moves: true,
        use_lcb_for_selection: true,
        lcb_stdevs: 5.0,
        min_visit_prop_for_lcb: 0.15,
    };
}
