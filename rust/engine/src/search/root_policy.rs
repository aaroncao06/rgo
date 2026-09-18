//! Root policy preprocessing, following KataGo's searchhelpers.cpp.

use rand::RngExt;
use std::f64::consts::TAU;

use crate::inference::policy::POLICY_SIZE;

/// Apply the stable power transform before adding root noise.
pub(super) fn apply_temperature(
    policy: &mut [f32; POLICY_SIZE],
    legal: &[bool; POLICY_SIZE],
    temperature: f64,
) {
    assert!(temperature.is_finite() && temperature > 0.0);

    if (temperature - 1.0).abs() > f64::EPSILON {
        let max_probability = policy
            .iter()
            .zip(legal)
            .filter_map(|(&probability, &is_legal)| is_legal.then_some(f64::from(probability)))
            .fold(0.0, f64::max);
        assert!(max_probability > 0.0);
        let log_max = max_probability.ln();
        let inverse_temperature = 1.0 / temperature;
        let mut sum = 0.0;
        for (probability, &is_legal) in policy.iter_mut().zip(legal) {
            if is_legal && *probability > 0.0 {
                let transformed = (f64::from(*probability).ln() - log_max) * inverse_temperature;
                *probability = transformed.exp() as f32;
                sum += f64::from(*probability);
            }
        }
        assert!(sum > 0.0 && sum.is_finite());
        for (probability, &is_legal) in policy.iter_mut().zip(legal) {
            if is_legal {
                *probability = (f64::from(*probability) / sum) as f32;
            }
        }
    }
}

/// Mix in KataGo's Dirichlet draw, whose alpha distribution combines uniform
/// mass with the clipped log policy after temperature adjustment.
pub(super) fn add_dirichlet_noise<R: RngExt + ?Sized>(
    policy: &mut [f32; POLICY_SIZE],
    legal: &[bool; POLICY_SIZE],
    noise_total_concentration: f64,
    noise_weight: f64,
    rng: &mut R,
) {
    if noise_weight == 0.0 {
        return;
    }
    assert!((0.0..=1.0).contains(&noise_weight));
    assert!(noise_total_concentration > 0.0);

    let legal_count = legal.iter().filter(|&&is_legal| is_legal).count();
    assert!(legal_count > 0);
    let legal_count_f64 = legal_count as f64;
    let mut alpha = [0.0; POLICY_SIZE];
    let mut log_policy_sum = 0.0;
    for i in 0..POLICY_SIZE {
        if legal[i] {
            alpha[i] = (f64::from(policy[i]).min(0.01) + 1e-20).ln();
            log_policy_sum += alpha[i];
        }
    }
    let log_policy_mean = log_policy_sum / legal_count_f64;
    let mut alpha_prop_sum = 0.0;
    for i in 0..POLICY_SIZE {
        if legal[i] {
            alpha[i] = (alpha[i] - log_policy_mean).max(0.0);
            alpha_prop_sum += alpha[i];
        }
    }
    if alpha_prop_sum <= 0.0 {
        for i in 0..POLICY_SIZE {
            if legal[i] {
                alpha[i] = 1.0 / legal_count_f64;
            }
        }
    } else {
        for i in 0..POLICY_SIZE {
            if legal[i] {
                alpha[i] = 0.5 * (alpha[i] / alpha_prop_sum + 1.0 / legal_count_f64);
            }
        }
    }

    let mut noise = [0.0; POLICY_SIZE];
    let mut noise_sum = 0.0;
    // Reuse both Box–Muller samples across gamma draws for this root.
    let mut spare_normal = None;
    for i in 0..POLICY_SIZE {
        if legal[i] {
            noise[i] = gamma_sample(alpha[i] * noise_total_concentration, rng, &mut spare_normal);
            noise_sum += noise[i];
        }
    }
    assert!(noise_sum > 0.0 && noise_sum.is_finite());
    for i in 0..POLICY_SIZE {
        if legal[i] {
            let draw = noise[i] / noise_sum;
            policy[i] = (draw * noise_weight + f64::from(policy[i]) * (1.0 - noise_weight)) as f32;
        }
    }
}

fn gamma_sample<R: RngExt + ?Sized>(
    shape: f64,
    rng: &mut R,
    spare_normal: &mut Option<f64>,
) -> f64 {
    assert!(shape > 0.0);
    if shape <= 1.0 {
        let sample = gamma_sample(shape + 1.0, rng, spare_normal);
        let uniform = rng.random::<f64>();
        return sample * uniform.powf(1.0 / shape);
    }

    // Marsaglia and Tsang's algorithm, matching KataGo's Rand::nextGamma.
    let d = shape - 1.0 / 3.0;
    let c = (1.0 / 3.0) / d.sqrt();
    loop {
        let x = standard_normal(rng, spare_normal);
        let vtmp = 1.0 + c * x;
        if vtmp <= 0.0 {
            continue;
        }
        let v = vtmp * vtmp * vtmp;
        let uniform = rng.random::<f64>();
        let xx = x * x;
        if uniform < 1.0 - 0.0331 * xx * xx
            || uniform == 0.0
            || uniform.ln() < 0.5 * xx + d * (1.0 - v + v.ln())
        {
            return d * v;
        }
    }
}

fn standard_normal<R: RngExt + ?Sized>(rng: &mut R, spare_normal: &mut Option<f64>) -> f64 {
    if let Some(sample) = spare_normal.take() {
        return sample;
    }
    loop {
        let u1 = rng.random::<f64>();
        let u2 = rng.random::<f64>();
        if u1 > 0.0 {
            let radius = (-2.0 * u1.ln()).sqrt();
            let (sin, cos) = (TAU * u2).sin_cos();
            *spare_normal = Some(radius * sin);
            return radius * cos;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::SmallRng};

    #[test]
    fn root_policy_temperature_matches_katago_power_transform() {
        let mut policy = [0.0; POLICY_SIZE];
        let mut legal = [false; POLICY_SIZE];
        for (index, probability) in [0.25_f32, 0.5, 0.25].into_iter().enumerate() {
            policy[index] = probability;
            legal[index] = true;
        }

        apply_temperature(&mut policy, &legal, 2.0);

        assert!((policy[0] - 0.29289322).abs() < 1e-6);
        assert!((policy[1] - 0.41421357).abs() < 1e-6);
        assert!((policy[2] - 0.29289322).abs() < 1e-6);
        assert_eq!(policy[3], 0.0);
    }

    #[test]
    fn noise_reaches_zero_probability_legal_moves_but_not_illegal_moves() {
        let mut policy = [0.0; POLICY_SIZE];
        policy[0] = 1.0;
        let mut legal = [false; POLICY_SIZE];
        legal[0] = true;
        legal[1] = true;
        let mut rng = SmallRng::seed_from_u64(19);

        apply_temperature(&mut policy, &legal, 1.5);
        assert_eq!(policy[1], 0.0);
        add_dirichlet_noise(&mut policy, &legal, 10.83, 0.25, &mut rng);

        assert!(policy[1] > 0.0 && policy[1] <= 0.25);
        assert!(policy[0] >= 0.75);
        assert!((policy[0] + policy[1] - 1.0).abs() < 1e-6);
        assert!(policy[2..].iter().all(|&probability| probability == 0.0));
    }

    #[test]
    fn root_dirichlet_noise_preserves_legal_probability_mass() {
        let mut policy = [0.0; POLICY_SIZE];
        let mut legal = [false; POLICY_SIZE];
        for (index, probability) in [0.6_f32, 0.3, 0.1].into_iter().enumerate() {
            policy[index] = probability;
            legal[index] = true;
        }
        let mut rng = SmallRng::seed_from_u64(2);

        add_dirichlet_noise(&mut policy, &legal, 10.83, 0.25, &mut rng);

        let legal_sum: f32 = policy
            .iter()
            .zip(legal)
            .filter_map(|(&probability, is_legal)| is_legal.then_some(probability))
            .sum();
        assert!((legal_sum - 1.0).abs() < 1e-6);
        assert!(policy[..3].iter().all(|&probability| probability > 0.0));
        assert!(policy[3..].iter().all(|&probability| probability == 0.0));
    }
}
