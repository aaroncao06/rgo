#[derive(Clone, Copy)]
pub(crate) struct Rules {
    pub(crate) komi: f32,
    pub(crate) multi_stone_suicide_legal: bool,
}

impl Rules {
    pub(crate) const TROMP_TAYLORISH: Self = Self {
        komi: 7.5,
        multi_stone_suicide_legal: true,
    };
    pub(crate) const OGS_CHINESE: Self = Self {
        komi: 7.5,
        multi_stone_suicide_legal: false,
    };
}
