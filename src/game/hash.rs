use super::board::{ARRAY_LEN, Color, Loc};

static ZOBRIST: Zobrist = Zobrist::new();

const ZOBRIST_SEED: u64 = 0x7267_6f5f_7a6f_6272;
struct SplitMix64 {
    state: u64,
}
impl SplitMix64 {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    const fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);

        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    const fn next_hash(&mut self) -> PositionHash {
        let high = self.next_u64();
        let low = self.next_u64();
        ((high as u128) << 64) | (low as u128)
    }
}
pub(crate) type PositionHash = u128;
struct Zobrist {
    stone_hashes: [[PositionHash; 4]; ARRAY_LEN],
}
impl Zobrist {
    const fn new() -> Self {
        let mut rng = SplitMix64::new(ZOBRIST_SEED);
        let mut stone_hashes = [[0; 4]; ARRAY_LEN];
        let mut i = 0;
        while i < ARRAY_LEN {
            stone_hashes[i][Color::Black as usize] = rng.next_hash();
            stone_hashes[i][Color::White as usize] = rng.next_hash();
            i += 1;
        }
        Self { stone_hashes }
    }
}

pub(crate) fn stone_hash(loc: Loc, color: Color) -> PositionHash {
    ZOBRIST.stone_hashes[loc.index()][color as usize]
}
