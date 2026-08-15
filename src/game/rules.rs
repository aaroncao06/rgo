pub struct Rules {
    pub komi: f32,
    pub multi_stone_suicide_legal: bool,
}

impl Rules {
    const TROMP_TAYLORISH: Self = Self {
        komi: 7.5,
        multi_stone_suicide_legal: true,
    };
}
