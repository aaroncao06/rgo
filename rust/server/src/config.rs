use crate::{
    Error, checkpoints::CheckpointConfig, model_export::ModelExportConfig, replay::ReplayConfig,
};
use rgo_protocol::client_server::valid_run_name;
use serde::Deserialize;
use std::{
    fs,
    net::SocketAddr,
    path::{Path, PathBuf},
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub listen: SocketAddr,
    pub data_dir: PathBuf,
    pub run_name: String,
    #[serde(default = "default_prune_old_shards")]
    pub prune_old_shards: bool,
    #[serde(default)]
    pub checkpoints: CheckpointConfig,
    pub replay: ReplayConfig,
    pub model_export: ModelExportConfig,
}

fn default_prune_old_shards() -> bool {
    true
}

impl Config {
    pub fn load(path: &Path) -> Result<Self, Error> {
        let config: Self = toml::from_str(
            &fs::read_to_string(path)
                .map_err(|error| format!("could not read server config: {error}"))?,
        )
        .map_err(|error| format!("invalid server config: {error}"))?;
        if !config.listen.ip().is_loopback() || config.data_dir.as_os_str().is_empty() {
            return Err("listen must be loopback and data_dir must not be empty".into());
        }
        if !valid_run_name(&config.run_name) {
            return Err(
                "run_name must contain 1–64 lowercase ASCII letters, digits, '-' or '_'".into(),
            );
        }
        config.replay.validate()?;
        if config.model_export.every_shards == 0 {
            return Err("model_export.every_shards must be positive".into());
        }
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retention_defaults_on_but_can_be_disabled() {
        let base = "listen = '127.0.0.1:0'\ndata_dir = 'data'\nrun_name = 'test-run'\n";
        let replay = "[replay]\ncapacity_records = 10\nwindow_records = 5\nsample_group_records = 5\nsample_groups_per_shard = 1\nrecords_per_shard = 5\n";
        let export = "[model_export]\nevery_shards = 10\n";
        let default: Config = toml::from_str(&format!("{base}{replay}{export}")).unwrap();
        assert!(default.checkpoints.prune_old_checkpoints);
        assert!(default.prune_old_shards);
        assert_eq!(default.model_export.every_shards, 10);

        let empty: Config =
            toml::from_str(&format!("{base}[checkpoints]\n{replay}{export}")).unwrap();
        assert!(empty.checkpoints.prune_old_checkpoints);

        let disabled: Config = toml::from_str(&format!(
            "{base}prune_old_shards = false\n[checkpoints]\nprune_old_checkpoints = false\n{replay}{export}"
        ))
        .unwrap();
        assert!(!disabled.checkpoints.prune_old_checkpoints);
        assert!(!disabled.prune_old_shards);
    }

    #[test]
    fn validates_run_name_and_model_export_cadence() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("server.toml");
        for (run_name, every_shards, expected_error) in [
            ("test-run", 10, None),
            ("Test-Run", 10, Some("run_name must contain")),
            (
                "test-run",
                0,
                Some("model_export.every_shards must be positive"),
            ),
        ] {
            fs::write(
                &path,
                format!(
                    r#"listen = '127.0.0.1:0'
data_dir = 'data'
run_name = '{run_name}'
[model_export]
every_shards = {every_shards}
[replay]
capacity_records = 10
window_records = 5
sample_group_records = 5
sample_groups_per_shard = 1
records_per_shard = 5
"#
                ),
            )
            .unwrap();
            let result = Config::load(&path);
            if let Some(expected_error) = expected_error {
                let error = result.err().expect("invalid config was accepted");
                assert!(error.to_string().contains(expected_error), "{error}");
            } else {
                assert!(result.is_ok());
            }
        }
    }
}
