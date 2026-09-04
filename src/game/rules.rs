pub struct Rules {
    pub komi: f32,
    pub multi_stone_suicide_legal: bool,
}

impl Rules {
    pub const TROMP_TAYLORISH: Self = Self {
        komi: 7.5,
        multi_stone_suicide_legal: true,
    };
    pub const OGS_CHINESE: Self = Self {
        komi: 7.5,
        multi_stone_suicide_legal: false,
    };
}
