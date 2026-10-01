use super::board::{ARRAY_LEN, Color, Loc, Player};

static ZOBRIST: Zobrist = Zobrist::new();

const PLAYER_HASH_SEED: u64 = u64::from_be_bytes(*b"players!");
const STONE_HASH_SEED: u64 = u64::from_be_bytes(*b"stones!!");
const SUPERKO_HASH_SEED: u64 = u64::from_be_bytes(*b"superko!");
const SUICIDE_HASH_SEED: u64 = u64::from_be_bytes(*b"suicide."); // for the binary rule
const KOMI_HASH_SEED: u64 = u64::from_be_bytes(*b"komi!!!!");
const PASS_HASH_SEED: u64 = u64::from_be_bytes(*b"passes!!");
const BOARD_SIZE_HASH_SEED: u64 = u64::from_be_bytes(*b"boardsz!");
const SPLITMIX64_INCREMENT: u64 = 0x9E37_79B9_7F4A_7C15;

const fn splitmix64_finalize(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

pub(crate) const fn splitmix64(value: u64) -> u64 {
    splitmix64_finalize(value.wrapping_add(SPLITMIX64_INCREMENT))
}

pub(crate) const fn nasam(mut value: u64) -> u64 {
    value ^= value.rotate_right(25) ^ value.rotate_right(47);
    value = value.wrapping_mul(0x9E6C_63D0_676A_9A99);
    value ^= (value >> 23) ^ (value >> 51);
    value = value.wrapping_mul(0x9E6D_62D0_6F6A_9A9B);
    value ^ (value >> 23) ^ (value >> 51)
}

struct SplitMix64 {
    state: u64,
}
impl SplitMix64 {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    const fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(SPLITMIX64_INCREMENT);
        splitmix64_finalize(self.state)
    }
    const fn next_hash(&mut self) -> Hash128 {
        let high = self.next_u64();
        let low = self.next_u64();
        ((high as u128) << 64) | (low as u128)
    }
}
pub(crate) type Hash128 = u128;
struct Zobrist {
    stone_hashes: [[Hash128; 4]; ARRAY_LEN],
    superko_hashes: [Hash128; ARRAY_LEN],
    player_hashes: [Hash128; 4],
    suicide_hash: Hash128,
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
    const fn build_stone_hashes() -> [[Hash128; 4]; ARRAY_LEN] {
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
    const fn build_superko_hashes() -> [Hash128; ARRAY_LEN] {
        let mut rng = SplitMix64::new(SUPERKO_HASH_SEED);
        let mut superko_hashes = [0; ARRAY_LEN];
        let mut i = 0;
        while i < ARRAY_LEN {
            superko_hashes[i] = rng.next_hash();
            i += 1;
        }
        superko_hashes
    }
    const fn build_player_hashes() -> [Hash128; 4] {
        let mut rng = SplitMix64::new(PLAYER_HASH_SEED);
        let black = rng.next_hash();
        let white = rng.next_hash();
        [0, black, white, 0]
    }
    const fn build_suicide_hash() -> Hash128 {
        let mut rng = SplitMix64::new(SUICIDE_HASH_SEED);
        rng.next_hash()
    }
}

pub(super) fn stone_hash(loc: Loc, color: Color) -> Hash128 {
    ZOBRIST.stone_hashes[loc.index()][color as usize]
}

pub(super) fn player_hash(player: Player) -> Hash128 {
    ZOBRIST.player_hashes[player as usize]
}

pub(super) fn superko_hash(loc: Loc) -> Hash128 {
    ZOBRIST.superko_hashes[loc.index()]
}

pub(crate) fn suicide_hash() -> Hash128 {
    ZOBRIST.suicide_hash
}

const fn scalar_hash(seed: u64, value: u64) -> Hash128 {
    let mut rng = SplitMix64::new(seed ^ value);
    rng.next_hash()
}

pub(crate) fn komi_hash(komi: f32) -> Hash128 {
    scalar_hash(KOMI_HASH_SEED, komi.to_bits() as u64)
}

pub(super) fn pass_hash(count: u8) -> Hash128 {
    scalar_hash(PASS_HASH_SEED, count as u64)
}

pub(super) fn board_size_hash(size: usize) -> Hash128 {
    scalar_hash(BOARD_SIZE_HASH_SEED, size as u64)
}
