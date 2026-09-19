//! Dihedral board symmetries at the neural-network boundary.

use crate::game::board::BOARD_SIZE;
use crate::inference::policy::BOARD_POLICY_SIZE;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Symmetry {
    Identity,
    FlipY,
    FlipX,
    FlipXY,
    Transpose,
    TransposeFlipY,
    TransposeFlipX,
    TransposeFlipXY,
}

impl Symmetry {
    pub(super) const ALL: [Self; 8] = [
        Self::Identity,
        Self::FlipY,
        Self::FlipX,
        Self::FlipXY,
        Self::Transpose,
        Self::TransposeFlipY,
        Self::TransposeFlipX,
        Self::TransposeFlipXY,
    ];

    pub(super) fn from_index(index: usize) -> Self {
        Self::ALL[index % Self::ALL.len()]
    }

    /// Map a canonical board index into the model's transformed coordinates.
    fn transform_index(self, index: usize) -> usize {
        debug_assert!(index < BOARD_POLICY_SIZE);
        let mut x = index % BOARD_SIZE;
        let mut y = index / BOARD_SIZE;
        let bits = self as u8;

        if bits & 0b001 != 0 {
            y = BOARD_SIZE - 1 - y;
        }
        if bits & 0b010 != 0 {
            x = BOARD_SIZE - 1 - x;
        }
        if bits & 0b100 != 0 {
            std::mem::swap(&mut x, &mut y);
        }
        x + y * BOARD_SIZE
    }

    /// Transform each canonical spatial plane into model coordinates.
    pub(super) fn transform_planes(self, values: &mut [f32]) {
        let (planes, remainder) = values.as_chunks_mut::<BOARD_POLICY_SIZE>();
        debug_assert!(remainder.is_empty());
        let mut transformed = [0.0; BOARD_POLICY_SIZE];
        for plane in planes {
            for canonical_index in 0..BOARD_POLICY_SIZE {
                transformed[self.transform_index(canonical_index)] = plane[canonical_index];
            }
            plane.copy_from_slice(&transformed);
        }
    }

    /// Convert one model-coordinate spatial output back to canonical coordinates.
    pub(super) fn restore_output(self, values: &mut [f32; BOARD_POLICY_SIZE]) {
        let transformed = *values;
        for canonical_index in 0..BOARD_POLICY_SIZE {
            values[canonical_index] = transformed[self.transform_index(canonical_index)];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_symmetry_round_trips_spatial_values() {
        let original: [f32; BOARD_POLICY_SIZE] = std::array::from_fn(|index| index as f32);
        for symmetry in Symmetry::ALL {
            let mut transformed = original;
            symmetry.transform_planes(&mut transformed);
            symmetry.restore_output(&mut transformed);
            assert_eq!(transformed, original, "{symmetry:?}");
        }
    }

    #[test]
    fn restoring_policy_leaves_pass_output_untouched() {
        let mut policy = [0.0; BOARD_POLICY_SIZE + 1];
        for (index, value) in policy[..BOARD_POLICY_SIZE].iter_mut().enumerate() {
            *value = index as f32;
        }
        policy[BOARD_POLICY_SIZE] = 1234.0;
        let original = policy;

        Symmetry::TransposeFlipX.transform_planes(&mut policy[..BOARD_POLICY_SIZE]);
        Symmetry::TransposeFlipX.restore_output(
            policy
                .first_chunk_mut::<BOARD_POLICY_SIZE>()
                .expect("policy contains a full board plane"),
        );

        assert_eq!(policy, original);
    }

    #[test]
    fn transformation_is_applied_independently_to_each_plane() {
        let mut planes = [0.0; 2 * BOARD_POLICY_SIZE];
        planes[0] = 1.0;
        planes[BOARD_POLICY_SIZE + 1] = 2.0;

        Symmetry::TransposeFlipXY.transform_planes(&mut planes);

        let first_target = Symmetry::TransposeFlipXY.transform_index(0);
        let second_target = Symmetry::TransposeFlipXY.transform_index(1);
        assert_eq!(planes[first_target], 1.0);
        assert_eq!(planes[BOARD_POLICY_SIZE + second_target], 2.0);
        assert_eq!(planes.iter().filter(|&&value| value != 0.0).count(), 2);
    }
}
