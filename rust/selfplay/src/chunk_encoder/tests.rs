use super::test_support::*;
use super::*;
use crate::game::board::MAX_BOARD_DIM;
use crate::game::{game_state::GameState, rules::Rules};
use crate::inference::policy::MAX_POLICY_SIZE;
use rgo_artifacts::chunk::{CHUNK_FORMAT_VERSION, CHUNK_MAGIC, verify_chunk_checksum};

fn sample(board_dim: usize) -> TestSample {
    let mut input = NNInput::encode(&GameState::new(Rules {
        board_dim,
        ..Rules::TROMP_TAYLORISH_9
    }));
    // Vary cells across rows and planes; invalid padding catches accidental
    // packing of storage cells outside the active board.
    input.spatial = std::array::from_fn(|i| {
        let cell = i % MAX_BOARD_AREA;
        if cell / MAX_BOARD_DIM < board_dim && cell % MAX_BOARD_DIM < board_dim {
            ((i * 13 + i / 5 + i / 17) % 2) as u8
        } else {
            u8::MAX
        }
    });
    input.global[1] = -0.0;
    TestSample {
        input,
        policy_target: std::array::from_fn(|i| f16::from_f64(i as f64 / MAX_POLICY_SIZE as f64)),
        value_target: ValueTarget {
            win_target: 1.0,
            final_score: 3.5,
            ownership: std::array::from_fn(|i| {
                if i / MAX_BOARD_DIM < board_dim && i % MAX_BOARD_DIM < board_dim {
                    (i % 3) as u8
                } else {
                    3
                }
            }),
        },
    }
}

#[test]
fn chunk_encoding_stores_only_active_cells_and_pass_for_all_board_dims() {
    for board_dim in 1..=MAX_BOARD_DIM {
        let sample = sample(board_dim);
        let bytes = encode_chunk(std::slice::from_ref(&sample));
        assert_eq!(&bytes[..8], &CHUNK_MAGIC);
        assert_eq!(read_u32(&bytes, 8), CHUNK_FORMAT_VERSION);
        assert_eq!(read_u32(&bytes, 12), NUM_SPATIAL_FEATURES as u32);
        assert_eq!(read_u32(&bytes, 16), 1);
        assert_eq!(read_u32(&bytes, 20), NUM_GLOBAL_FEATURES as u32);
        assert_eq!(
            bytes.len(),
            CHUNK_HEADER_SIZE + training_record_size(board_dim) + CHUNK_CHECKSUM_SIZE
        );
        assert!(verify_chunk_checksum(&bytes));

        let mut offset = CHUNK_HEADER_SIZE;
        assert_eq!(usize::from(bytes[offset]), board_dim);
        offset += 1;
        for plane in 0..NUM_SPATIAL_FEATURES {
            let cells = read_packed(&bytes, &mut offset, board_dim * board_dim, 1);
            for y in 0..board_dim {
                for x in 0..board_dim {
                    assert_eq!(
                        cells[y * board_dim + x],
                        sample.input.spatial[plane * MAX_BOARD_AREA + y * MAX_BOARD_DIM + x]
                    );
                }
            }
        }
        for value in sample.input.global {
            assert_eq!(read_f32(&bytes, offset).to_bits(), value.to_bits());
            offset += 4;
        }
        for value in &sample.policy_target[..board_dim * board_dim + 1] {
            let stored = f16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap());
            assert_eq!(stored.to_bits(), value.to_bits());
            offset += 2;
        }
        for value in [
            sample.value_target.win_target,
            sample.value_target.final_score,
        ] {
            assert_eq!(read_f32(&bytes, offset).to_bits(), value.to_bits());
            offset += 4;
        }
        let ownership = read_packed(&bytes, &mut offset, board_dim * board_dim, 2);
        for y in 0..board_dim {
            for x in 0..board_dim {
                assert_eq!(
                    ownership[y * board_dim + x],
                    sample.value_target.ownership[y * MAX_BOARD_DIM + x]
                );
            }
        }
        assert_eq!(offset + CHUNK_CHECKSUM_SIZE, bytes.len());
    }
}

#[test]
fn policy_storage_uses_little_endian_binary16_and_preserves_subnormals() {
    let mut sample = sample(1);
    sample.policy_target[0] = f16::from_f64(1.0 / 3.0);
    sample.policy_target[1] = f16::from_bits(1);
    let bytes = encode_chunk(&[sample]);
    let policy_offset =
        CHUNK_HEADER_SIZE + 1 + NUM_SPATIAL_FEATURES + NUM_GLOBAL_FEATURES * size_of::<f32>();
    assert_eq!(
        &bytes[policy_offset..policy_offset + 4],
        &[0x55, 0x35, 0x01, 0x00]
    );
    assert!(verify_chunk_checksum(&bytes));
}

#[test]
fn packed_fields_have_lsb_first_order_and_independent_byte_boundaries() {
    let mut sample = sample(3);
    for (plane, values) in [[1, 0, 1, 1, 0, 0, 1, 0, 1], [0; 9], [1; 9]]
        .into_iter()
        .enumerate()
    {
        for (i, value) in values.into_iter().enumerate() {
            sample.input.spatial[plane * MAX_BOARD_AREA + i / 3 * MAX_BOARD_DIM + i % 3] = value;
        }
    }
    for i in 0..9 {
        sample.value_target.ownership[i / 3 * MAX_BOARD_DIM + i % 3] = (i % 3) as u8;
    }
    let bytes = encode_chunk(&[sample]);
    let spatial_offset = CHUNK_HEADER_SIZE + 1;
    assert_eq!(
        &bytes[spatial_offset..spatial_offset + 6],
        &[0x4d, 0x01, 0x00, 0x00, 0xff, 0x01]
    );
    let ownership_end = bytes.len() - CHUNK_CHECKSUM_SIZE;
    assert_eq!(
        &bytes[ownership_end - 3..ownership_end],
        &[0x24, 0x49, 0x02]
    );
}

#[test]
fn packing_supports_all_inference_board_areas() {
    for board_dim in [9, 13, 19] {
        let area = board_dim * board_dim;
        let mut bytes = Vec::new();
        extend_packed::<1>(&mut bytes, (0..area).map(|i| (i % 2) as u8));
        extend_packed::<2>(&mut bytes, (0..area).map(|i| (i % 3) as u8));
        let mut offset = 0;
        assert_eq!(
            read_packed(&bytes, &mut offset, area, 1),
            (0..area).map(|i| (i % 2) as u8).collect::<Vec<_>>()
        );
        assert_eq!(
            read_packed(&bytes, &mut offset, area, 2),
            (0..area).map(|i| (i % 3) as u8).collect::<Vec<_>>()
        );
        assert_eq!(offset, bytes.len());
    }
}

#[test]
fn mixed_dim_records_are_self_describing_in_full_and_partial_chunks() {
    for capacity in [5, 7] {
        let mut encoder = ChunkEncoder::new(capacity);
        for board_dim in [9, 13, 19, 3, 5] {
            sample(board_dim).push_into(&mut encoder);
        }
        assert_eq!(encoder.record_count(), 5);
        let bytes = encoder.finish();
        assert_eq!(read_u32(bytes, 16), 5);
        let mut offset = CHUNK_HEADER_SIZE;
        for board_dim in [9, 13, 19, 3, 5] {
            assert_eq!(usize::from(bytes[offset]), board_dim);
            offset += training_record_size(board_dim);
        }
        assert_eq!(offset + CHUNK_CHECKSUM_SIZE, bytes.len());
        assert!(verify_chunk_checksum(bytes));
    }
}

#[test]
fn encoder_reuses_allocation_across_full_and_partial_chunks() {
    let mut encoder = ChunkEncoder::new(4);
    let mut allocation = None;
    // Start with the largest payload so subsequent resets need no growth.
    // Alternate full and partial chunks, capacities, dimensions, and data.
    for (capacity, board_dim, count) in [(4, 19, 4), (6, 9, 2), (3, 3, 3), (1, 19, 1), (4, 9, 3)] {
        encoder.reset(capacity);
        assert_eq!(encoder.record_count(), 0);
        assert_eq!(encoder.remaining_capacity(), capacity);
        let samples: Vec<_> = (0..count)
            .map(|tag| {
                let mut sample = sample(board_dim);
                sample.value_target.final_score = tag as f32 + capacity as f32;
                sample
            })
            .collect();
        for sample in &samples {
            sample.push_into(&mut encoder);
        }
        assert_eq!(encoder.record_count(), count);
        assert_eq!(encoder.remaining_capacity(), capacity - count);
        let bytes = encoder.finish();
        assert_eq!(read_u32(bytes, 16), count as u32);
        assert_eq!(
            bytes.len(),
            CHUNK_HEADER_SIZE + count * training_record_size(board_dim) + CHUNK_CHECKSUM_SIZE
        );
        assert!(verify_chunk_checksum(bytes));
        assert_eq!(bytes, encode_chunk(&samples));
        assert_eq!(encoder.record_count(), 0);
        let current = (encoder.bytes.as_ptr(), encoder.bytes.capacity());
        if let Some(allocation) = allocation {
            assert_eq!(current, allocation);
        }
        allocation = Some(current);
    }
}

#[test]
fn chunk_checksum_rejects_corrupted_payload() {
    let sample = TestSample {
        input: NNInput::encode(&GameState::new(Rules::TROMP_TAYLORISH_9)),
        policy_target: [f16::from_f32(0.25); MAX_POLICY_SIZE],
        value_target: ValueTarget {
            win_target: 1.0,
            final_score: 3.5,
            ownership: [2; MAX_BOARD_AREA],
        },
    };
    let mut bytes = encode_chunk(&[sample]);

    bytes[CHUNK_HEADER_SIZE] ^= 1;

    assert!(!verify_chunk_checksum(&bytes));
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn read_f32(bytes: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn read_packed(bytes: &[u8], offset: &mut usize, count: usize, bits: usize) -> Vec<u8> {
    let len = (count * bits).div_ceil(8);
    let packed = &bytes[*offset..*offset + len];
    let values = (0..count)
        .map(|i| (packed[i * bits / 8] >> (i * bits % 8)) & ((1 << bits) - 1))
        .collect();
    let used_bits = count * bits % 8;
    if used_bits != 0 {
        assert_eq!(
            packed[len - 1] >> used_bits,
            0,
            "nonzero trailing padding bits"
        );
    }
    *offset += len;
    values
}
