//! Dihedral board symmetries at the neural-network boundary.

use crate::{game::board::MAX_BOARD_AREA, game::board::MAX_BOARD_DIM};

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

    /// Transform each active spatial square in place, leaving padding untouched.
    pub(super) fn transform_planes(self, values: &mut [f32], board_dim: usize) {
        let (planes, remainder) = values.as_chunks_mut::<MAX_BOARD_AREA>();
        debug_assert!(remainder.is_empty());
        debug_assert!(board_dim <= MAX_BOARD_DIM);
        if self == Self::Identity {
            return;
        }
        for plane in planes {
            self.flip(plane, board_dim, MAX_BOARD_DIM);
            self.transpose(plane, board_dim, MAX_BOARD_DIM);
        }
    }

    /// Restore an active spatial square in place, using its storage stride.
    pub(super) fn restore_output(self, values: &mut [f32], board_dim: usize, stride: usize) {
        debug_assert_eq!(values.len(), stride * stride);
        debug_assert!(board_dim <= stride);
        if self == Self::Identity {
            return;
        }
        // Forward mapping flips first, then transposes. Undo in reverse order.
        self.transpose(values, board_dim, stride);
        self.flip(values, board_dim, stride);
    }

    fn transpose(self, values: &mut [f32], board_dim: usize, stride: usize) {
        if self as u8 & 0b100 != 0 {
            for y in 0..board_dim {
                for x in 0..y {
                    values.swap(x + y * stride, y + x * stride);
                }
            }
        }
    }

    fn flip(self, values: &mut [f32], board_dim: usize, stride: usize) {
        if self as u8 & 0b010 != 0 {
            for row in values.chunks_exact_mut(stride).take(board_dim) {
                row[..board_dim].reverse();
            }
        }
        if self as u8 & 0b001 != 0 {
            for y in 0..board_dim / 2 {
                let (top, bottom) = values.split_at_mut((board_dim - 1 - y) * stride);
                top[y * stride..y * stride + board_dim].swap_with_slice(&mut bottom[..board_dim]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_transforms_match_coordinates_and_preserve_padding() {
        let original: [f32; 3 * MAX_BOARD_AREA] = std::array::from_fn(|index| (index + 1) as f32);
        for dim in 1..=MAX_BOARD_DIM {
            for symmetry in Symmetry::ALL {
                let mut expected = original;
                for plane in 0..3 {
                    for y in 0..dim {
                        for x in 0..dim {
                            let end = dim - 1;
                            let (tx, ty) = match symmetry {
                                Symmetry::Identity => (x, y),
                                Symmetry::FlipY => (x, end - y),
                                Symmetry::FlipX => (end - x, y),
                                Symmetry::FlipXY => (end - x, end - y),
                                Symmetry::Transpose => (y, x),
                                Symmetry::TransposeFlipY => (end - y, x),
                                Symmetry::TransposeFlipX => (y, end - x),
                                Symmetry::TransposeFlipXY => (end - y, end - x),
                            };
                            let offset = plane * MAX_BOARD_AREA;
                            expected[offset + tx + ty * MAX_BOARD_DIM] =
                                original[offset + x + y * MAX_BOARD_DIM];
                        }
                    }
                }
                let mut transformed = original;
                symmetry.transform_planes(&mut transformed, dim);
                assert_eq!(transformed, expected, "dim={dim}, {symmetry:?}");
            }
        }
    }

    #[test]
    fn every_symmetry_round_trips_spatial_values() {
        let original: [f32; MAX_BOARD_AREA] = std::array::from_fn(|index| index as f32);
        for symmetry in Symmetry::ALL {
            let mut transformed = original;
            symmetry.transform_planes(&mut transformed, MAX_BOARD_DIM);
            symmetry.restore_output(&mut transformed, MAX_BOARD_DIM, MAX_BOARD_DIM);
            assert_eq!(transformed, original, "{symmetry:?}");
        }
    }

    #[test]
    fn every_symmetry_restores_compact_and_padded_planes_on_smaller_boards() {
        use crate::inference::policy::active_rows;
        for dim in 1..=MAX_BOARD_DIM {
            let canonical: [f32; MAX_BOARD_AREA] = std::array::from_fn(|i| {
                if i % MAX_BOARD_DIM < dim && i / MAX_BOARD_DIM < dim {
                    (i + 1) as f32
                } else {
                    0.0
                }
            });
            let expected_compact: Vec<_> =
                active_rows(&canonical, dim).flatten().copied().collect();
            for symmetry in Symmetry::ALL {
                let mut transformed = canonical;
                symmetry.transform_planes(&mut transformed, dim);
                let mut compact: Box<[_]> =
                    active_rows(&transformed, dim).flatten().copied().collect();
                let allocation = compact.as_ptr();
                symmetry.restore_output(&mut compact, dim, dim);
                assert_eq!(&*compact, expected_compact, "dim={dim}, {symmetry:?}");
                assert_eq!(compact.as_ptr(), allocation);

                // Restoring a fixed-capacity input plane leaves padding untouched.
                let mut expected_padded = canonical;
                for i in 0..MAX_BOARD_AREA {
                    if i % MAX_BOARD_DIM >= dim || i / MAX_BOARD_DIM >= dim {
                        transformed[i] = 1000.0 + i as f32;
                        expected_padded[i] = transformed[i];
                    }
                }
                symmetry.restore_output(&mut transformed, dim, MAX_BOARD_DIM);
                assert_eq!(transformed, expected_padded, "dim={dim}, {symmetry:?}");
            }
        }
    }

    #[test]
    fn restoring_policy_leaves_pass_output_untouched() {
        let mut policy = [0.0; MAX_BOARD_AREA + 1];
        for (index, value) in policy[..MAX_BOARD_AREA].iter_mut().enumerate() {
            *value = index as f32;
        }
        policy[MAX_BOARD_AREA] = 1234.0;
        let original = policy;

        Symmetry::TransposeFlipX.transform_planes(&mut policy[..MAX_BOARD_AREA], MAX_BOARD_DIM);
        Symmetry::TransposeFlipX.restore_output(
            policy
                .first_chunk_mut::<MAX_BOARD_AREA>()
                .expect("policy contains a full board plane"),
            MAX_BOARD_DIM,
            MAX_BOARD_DIM,
        );

        assert_eq!(policy, original);
    }

    #[test]
    fn transformation_is_applied_independently_to_each_plane() {
        let mut planes = [0.0; 2 * MAX_BOARD_AREA];
        planes[0] = 1.0;
        planes[MAX_BOARD_AREA + 1] = 2.0;

        Symmetry::TransposeFlipXY.transform_planes(&mut planes, MAX_BOARD_DIM);

        let first_target = MAX_BOARD_AREA - 1;
        let second_target = first_target - MAX_BOARD_DIM;
        assert_eq!(planes[first_target], 1.0);
        assert_eq!(planes[MAX_BOARD_AREA + second_target], 2.0);
        assert_eq!(planes.iter().filter(|&&value| value != 0.0).count(), 2);
    }
}
