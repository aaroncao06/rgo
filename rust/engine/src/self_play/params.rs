use rand::RngExt;

use crate::{
    game::rules::Rules,
    search::{node_store::FixedArenaNodeStore, worker::SearchBudget},
};

#[derive(Debug, Clone, Copy, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct SearchBudgetTier {
    probability: f64,
    budget: SearchBudget,
}

impl SearchBudgetTier {
    pub(super) const fn new(probability: f64, budget: SearchBudget) -> Self {
        Self {
            probability,
            budget,
        }
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(transparent)]
pub(super) struct SearchBudgetPolicy {
    tiers: Box<[SearchBudgetTier]>,
}

impl SearchBudgetPolicy {
    pub(super) fn validate(&self) -> Result<(), &'static str> {
        if self.tiers.is_empty() {
            return Err("search budget policy needs a tier");
        }
        if !self
            .tiers
            .iter()
            .all(|tier| tier.probability.is_finite() && tier.probability > 0.0)
        {
            return Err("search budget probabilities must be positive and finite");
        }
        let total: f64 = self.tiers.iter().map(|tier| tier.probability).sum();
        if (total - 1.0).abs() > 1e-12 || !total.is_finite() {
            return Err("search budget probabilities must sum to one");
        }
        for tier in &self.tiers {
            tier.budget.validate()?;
            // Self-play currently uses the fixed arena. Its index limit is not
            // an algorithm-level restriction on other SearchWorker stores.
            FixedArenaNodeStore::validate_capacity(tier.budget.max_nodes())?;
        }
        Ok(())
    }

    pub(super) fn new(tiers: Vec<SearchBudgetTier>) -> Self {
        let policy = Self {
            tiers: tiers.into_boxed_slice(),
        };
        policy.validate().expect("invalid search budget policy");
        policy
    }

    pub(super) fn fixed(budget: SearchBudget) -> Self {
        Self::new(vec![SearchBudgetTier::new(1.0, budget)])
    }

    pub(super) fn sample<R: RngExt + ?Sized>(&self, rng: &mut R) -> SearchBudget {
        if self.tiers.len() == 1 {
            return self.tiers[0].budget;
        }

        self.budget_for_draw(rng.random_range(0.0..1.0))
    }

    fn budget_for_draw(&self, draw: f64) -> SearchBudget {
        debug_assert!((0.0..1.0).contains(&draw));
        let mut cumulative_probability = 0.0;
        for tier in &self.tiers[..self.tiers.len() - 1] {
            cumulative_probability += tier.probability;
            if draw < cumulative_probability {
                return tier.budget;
            }
        }
        self.tiers.last().expect("policy is nonempty").budget
    }
}

#[derive(Clone, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(super) struct SelfPlayParams {
    pub(super) rules: Rules,
    pub(super) search_budget_policy: SearchBudgetPolicy,
    pub(super) randomize_inference_symmetry: bool,
}

impl Default for SelfPlayParams {
    fn default() -> Self {
        Self {
            rules: Rules::default(),
            search_budget_policy: SearchBudgetPolicy::fixed(SearchBudget::new(512, 1024)),
            randomize_inference_symmetry: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::{SeedableRng, rngs::SmallRng};

    use super::*;

    #[test]
    fn fixed_policy_always_returns_its_budget() {
        let budget = SearchBudget::new(512, 2048);
        let policy = SearchBudgetPolicy::fixed(budget);
        let mut rng = SmallRng::seed_from_u64(1);

        assert_eq!(policy.sample(&mut rng), budget);
    }

    #[test]
    fn default_params_use_a_fixed_512_node_budget() {
        let params = SelfPlayParams::default();
        let mut rng = SmallRng::seed_from_u64(1);

        assert_eq!(
            params.search_budget_policy.sample(&mut rng),
            SearchBudget::new(512, 1024)
        );
        assert!(params.randomize_inference_symmetry);
    }

    #[test]
    fn weighted_policy_selects_by_cumulative_probability() {
        let shallow = SearchBudget::new(128, 512);
        let deep = SearchBudget::new(1024, 4096);
        let policy = SearchBudgetPolicy::new(vec![
            SearchBudgetTier::new(0.75, shallow),
            SearchBudgetTier::new(0.25, deep),
        ]);

        assert_eq!(policy.budget_for_draw(0.0), shallow);
        assert_eq!(policy.budget_for_draw(0.749), shallow);
        assert_eq!(policy.budget_for_draw(0.75), deep);
        assert_eq!(policy.budget_for_draw(0.999), deep);
    }
}
