use super::board::{ARRAY_LEN, Color, Loc, Player};

static ZOBRIST: Zobrist = Zobrist::new();

const PLAYER_HASH_SEED: u64 = u64::from_be_bytes(*b"players!");
const STONE_HASH_SEED: u64 = u64::from_be_bytes(*b"stones!!");
const SUPERKO_HASH_SEED: u64 = u64::from_be_bytes(*b"superko!");
const SUICIDE_HASH_SEED: u64 = u64::from_be_bytes(*b"suicide."); // for the binary rule
const KOMI_HASH_SEED: u64 = u64::from_be_bytes(*b"komi!!!!");
const PASS_HASH_SEED: u64 = u64::from_be_bytes(*b"passes!!");

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
    superko_hashes: [PositionHash; ARRAY_LEN],
    player_hashes: [PositionHash; 4],
    suicide_hash: PositionHash,
}
impl Zobrist {
    const fn new() -> Self {
        Self {
            stone_hashes: Self::build_stone_hashes(),
            superko_hashes: Self::build_superko_hashes(),
            player_hashes: Self::build_player_hashes(),
            suicide_hash: Self::build_suicide_hash(),
        }
    }
    const fn build_stone_hashes() -> [[PositionHash; 4]; ARRAY_LEN] {
        let mut rng = SplitMix64::new(STONE_HASH_SEED);
        let mut stone_hashes = [[0; 4]; ARRAY_LEN];
        let mut i = 0;
        while i < ARRAY_LEN {
            stone_hashes[i][Color::Black as usize] = rng.next_hash();
            stone_hashes[i][Color::White as usize] = rng.next_hash();
            i += 1;
        }
        stone_hashes
    }
    const fn build_superko_hashes() -> [PositionHash; ARRAY_LEN] {
        let mut rng = SplitMix64::new(SUPERKO_HASH_SEED);
        let mut superko_hashes = [0; ARRAY_LEN];
        let mut i = 0;
        while i < ARRAY_LEN {
            superko_hashes[i] = rng.next_hash();
            i += 1;
        }
        superko_hashes
    }
    const fn build_player_hashes() -> [PositionHash; 4] {
        let mut rng = SplitMix64::new(PLAYER_HASH_SEED);
        let black = rng.next_hash();
        let white = rng.next_hash();
        [0, black, white, 0]
    }
    const fn build_suicide_hash() -> PositionHash {
        let mut rng = SplitMix64::new(SUICIDE_HASH_SEED);
        rng.next_hash()
    }
}

pub(crate) fn stone_hash(loc: Loc, color: Color) -> PositionHash {
    ZOBRIST.stone_hashes[loc.index()][color as usize]
}

pub(crate) fn player_hash(player: Player) -> PositionHash {
    ZOBRIST.player_hashes[player as usize]
}

pub(crate) fn superko_hash(loc: Loc) -> PositionHash {
    ZOBRIST.superko_hashes[loc.index()]
}

pub(crate) fn suicide_hash() -> PositionHash {
    ZOBRIST.suicide_hash
}

const fn scalar_hash(seed: u64, value: u64) -> PositionHash {
    let mut rng = SplitMix64::new(seed ^ value);
    rng.next_hash()
}

pub(crate) fn komi_hash(komi: f32) -> PositionHash {
    scalar_hash(KOMI_HASH_SEED, komi.to_bits() as u64)
}

pub(crate) fn pass_hash(count: u8) -> PositionHash {
    scalar_hash(PASS_HASH_SEED, count as u64)
}
