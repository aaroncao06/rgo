use rand::{SeedableRng, rngs::SmallRng};

use super::*;
use crate::{inference::onnx::InferenceDevice, search::worker::SearchBudget};

const EXAMPLE: &str = include_str!("../../../../configs/self_play.toml");

#[test]
fn example_reuses_algorithm_defaults() {
    let config = SelfPlayConfig::from_toml(EXAMPLE).unwrap();
    assert_eq!(config.worker_threads, 2);
    assert_eq!(config.workers_per_thread, 4);
    assert!(matches!(config.chunk, ChunkMode::PerGame));
    assert!(matches!(
        config.inference.executors[0].device,
        InferenceDevice::Cpu { intra_threads: 1 }
    ));
    let mut rng = SmallRng::seed_from_u64(0);
    assert_eq!(
        config.self_play.search_budget_policy.sample(&mut rng),
        SearchBudget::new(512, 1024)
    );
    assert_eq!(
        config.search.cpuct_exploration,
        SearchParams::default().cpuct_exploration
    );
    assert!(config.self_play.randomize_inference_symmetry);
    assert_eq!(config.self_play.rules.board_dim, 9);
}

#[test]
fn partial_algorithm_overrides_preserve_other_defaults() {
    let source = format!(
        "{EXAMPLE}\n[self_play]\nrandomize_inference_symmetry = false\n[self_play.rules]\nkomi = 6.5\n[search]\ncpuct_exploration = 1.2\n"
    );
    let config = SelfPlayConfig::from_toml(&source).unwrap();
    assert!(!config.self_play.randomize_inference_symmetry);
    assert_eq!(config.self_play.rules.komi, 6.5);
    assert!(config.self_play.rules.multi_stone_suicide_legal);
    assert_eq!(config.self_play.rules.board_dim, 9);
    assert_eq!(config.search.cpuct_exploration, 1.2);
    assert_eq!(config.search.lcb_stdevs, SearchParams::default().lcb_stdevs);
}

#[test]
fn board_dim_must_be_supported_and_fit_storage() {
    for dim in [9, 13, 19] {
        let config = SelfPlayConfig::from_toml(&format!(
            "{EXAMPLE}\n[self_play.rules]\nboard_dim = {dim}\n"
        ))
        .unwrap();
        assert_eq!(config.self_play.rules.board_dim, dim);
    }
    for dim in [0, 1, 5, 7, 8, 10, 20, usize::MAX] {
        assert!(matches!(
            SelfPlayConfig::from_toml(&format!(
                "{EXAMPLE}\n[self_play.rules]\nboard_dim = {dim}\n"
            )),
            Err(ConfigError::Invalid(_)) | Err(ConfigError::Parse(_))
        ));
    }
}

#[test]
fn supports_fixed_chunks_and_budget_tiers() {
    let source = format!(
        "{}\n[self_play]\nsearch_budget_policy = [\n{{ probability = 0.75, budget = {{ max_nodes = 128, max_playouts = 256 }} }},\n{{ probability = 0.25, budget = {{ max_nodes = 1024, max_playouts = 2048 }} }}\n]\n",
        EXAMPLE.replace(
            "chunk = { mode = \"per_game\" }",
            "chunk = { mode = \"fixed_records\", records = 25000 }"
        )
    );
    let config = SelfPlayConfig::from_toml(&source).unwrap();
    assert!(matches!(config.chunk, ChunkMode::FixedRecords(25000)));
    config.self_play.search_budget_policy.validate().unwrap();
}

#[test]
fn unknown_fields_are_errors_at_every_level() {
    for source in [
        format!("misspelled = 1\n{EXAMPLE}"),
        EXAMPLE.replace("cache_capacity = 65536", "cache_capcity = 65536"),
        EXAMPLE.replace("intra_threads = 1", "intra_threads = 1, typo = 2"),
        EXAMPLE.replace("mode = \"per_game\"", "mode = \"per_game\", typo = 2"),
        format!("{EXAMPLE}\n[self_play]\ntypo = true\n"),
        format!("{EXAMPLE}\n[self_play.rules]\ntypo = true\n"),
        format!("{EXAMPLE}\n[search]\ntypo = 0.5\n"),
    ] {
        assert!(matches!(
            SelfPlayConfig::from_toml(&source),
            Err(ConfigError::Parse(_))
        ));
    }
}

#[test]
fn invalid_operational_settings_return_errors_without_panicking() {
    for source in [
        EXAMPLE.replace("worker_threads = 2", "worker_threads = 0"),
        EXAMPLE.replace("workers_per_thread = 4", "workers_per_thread = 0"),
        EXAMPLE.replace("cache_capacity = 65536", "cache_capacity = 3"),
        EXAMPLE.replace("num_cache_shards = 16", "num_cache_shards = 131072"),
        EXAMPLE.replace("max_batch_size = 8", "max_batch_size = 0"),
        EXAMPLE.replace("intra_threads = 1", "intra_threads = 0"),
        EXAMPLE.replace(
            "mode = \"per_game\"",
            "mode = \"fixed_records\", records = 0",
        ),
        EXAMPLE.replace("model_dir = \"models\"", "model_dir = \"\""),
    ] {
        assert!(matches!(
            SelfPlayConfig::from_toml(&source),
            Err(ConfigError::Invalid(_))
        ));
    }
    let mut config = SelfPlayConfig::from_toml(EXAMPLE).unwrap();
    config.inference.executors.clear();
    assert!(config.validate().is_err());
    config.worker_threads = usize::MAX;
    assert!(config.validate().is_err());
}

#[test]
fn invalid_algorithm_settings_return_errors_without_panicking() {
    for overrides in [
        "[self_play]\nsearch_budget_policy = []",
        "[self_play]\nsearch_budget_policy = [{ probability = 0.5, budget = { max_nodes = 512, max_playouts = 1024 } }]",
        "[self_play]\nsearch_budget_policy = [{ probability = 1.0, budget = { max_nodes = 512, max_playouts = 1 } }]",
        "[self_play]\nsearch_budget_policy = [{ probability = nan, budget = { max_nodes = 512, max_playouts = 1024 } }]",
        "[self_play.rules]\nkomi = inf",
        "[search]\ncpuct_exploration_base = 0.0",
        "[search]\nchosen_move_temperature = nan",
        "[search]\nroot_policy_temperature = 0.0",
        "[search]\nroot_dirichlet_noise_weight = 1.1",
        "[search]\nfpu_parent_weight_by_visited_policy_pow = -1.0",
        "[search]\nroot_dirichlet_noise_total_concentration = 0.0",
        "[search]\nlcb_stdevs = 0.0",
    ] {
        assert!(matches!(
            SelfPlayConfig::from_toml(&format!("{EXAMPLE}\n{overrides}\n")),
            Err(ConfigError::Invalid(_))
        ));
    }
}

#[test]
fn search_validation_does_not_impose_tuning_ranges() {
    let source = format!(
        "{EXAMPLE}\n[search]\nroot_ending_bonus_points = -0.5\nfpu_reduction_max = -0.2\ncpuct_exploration = -1.0\ncpuct_exploration_log = -0.2\nwin_loss_utility_factor = 2.0\nvalue_weight_exponent = -0.5\nchosen_move_subtract = -1.0\nchosen_move_prune = -1.0\ndynamic_score_center_scale = 0.1\ncpuct_exploration_base = 1.0\nroot_policy_temperature = 200.0\nlcb_stdevs = 20.0\n"
    );
    let config = SelfPlayConfig::from_toml(&source).unwrap();
    assert_eq!(config.search.root_ending_bonus_points, -0.5);
}

#[test]
fn disabled_features_do_not_require_positive_unused_parameters() {
    let source = format!(
        "{EXAMPLE}\n[search]\nroot_noise_enabled = false\nroot_dirichlet_noise_total_concentration = 0.0\nuse_lcb_for_selection = false\nlcb_stdevs = 0.0\n"
    );
    SelfPlayConfig::from_toml(&source).unwrap();
}

#[test]
fn zero_search_budget_is_valid() {
    let source = format!(
        "{EXAMPLE}\n[self_play]\nsearch_budget_policy = [{{ probability = 1.0, budget = {{ max_nodes = 0, max_playouts = 0 }} }}]\n"
    );
    SelfPlayConfig::from_toml(&source).unwrap();
}

#[cfg(target_pointer_width = "64")]
#[test]
fn self_play_rejects_budgets_beyond_the_fixed_arena_index_range() {
    let source = format!(
        "{EXAMPLE}\n[self_play]\nsearch_budget_policy = [{{ probability = 1.0, budget = {{ max_nodes = 4294967296, max_playouts = 4294967296 }} }}]\n"
    );
    assert!(matches!(
        SelfPlayConfig::from_toml(&source),
        Err(ConfigError::Invalid(
            "node capacity exceeds the fixed arena's u32 index range"
        ))
    ));
}

#[test]
fn cuda_device_configuration_is_feature_gated() {
    let source = EXAMPLE.replace(
        "type = \"cpu\", intra_threads = 1",
        "type = \"cuda\", device_id = 0",
    );
    if cfg!(feature = "cuda") {
        assert!(SelfPlayConfig::from_toml(&source).is_ok());
    } else {
        assert!(matches!(
            SelfPlayConfig::from_toml(&source),
            Err(ConfigError::Invalid(_))
        ));
    }
    assert!(matches!(
        SelfPlayConfig::from_toml(&source.replace("device_id = 0", "device_id = -1")),
        Err(ConfigError::Invalid(_))
    ));
}

#[test]
fn load_reads_the_checked_in_example() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/self_play.toml");
    assert!(SelfPlayConfig::load(&path).is_ok());
    assert!(matches!(
        SelfPlayConfig::load(&path.join("missing")),
        Err(ConfigError::Read(_))
    ));
}
