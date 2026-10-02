#[derive(Clone, Copy, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Rules {
    pub board_dim: usize,
    pub komi: f32,
    pub multi_stone_suicide_legal: bool,
}

impl Default for Rules {
    fn default() -> Self {
        Self::TROMP_TAYLORISH_9
    }
}

impl Rules {
    pub const TROMP_TAYLORISH_9: Self = Self {
        board_dim: 9,
        komi: 7.5,
        multi_stone_suicide_legal: true,
    };
    pub const OGS_CHINESE_9: Self = Self {
        board_dim: 9,
        komi: 7.5,
        multi_stone_suicide_legal: false,
    };

    pub fn validate(&self) -> Result<(), &'static str> {
        if !(1..=super::board::MAX_BOARD_DIM).contains(&self.board_dim) {
            return Err("board_dim must fit the board storage capacity");
        }
        if !self.komi.is_finite() {
            return Err("komi must be finite");
        }
        Ok(())
    }
}
