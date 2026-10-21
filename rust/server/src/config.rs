use crate::{Error, checkpoints::CheckpointConfig, replay::ReplayConfig};
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
    #[serde(default)]
    pub checkpoints: CheckpointConfig,
    pub replay: ReplayConfig,
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
        config.replay.validate()?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_pruning_defaults_on_but_can_be_disabled() {
        let base = "listen = '127.0.0.1:0'\ndata_dir = 'data'\n";
        let replay = "[replay]\ncapacity_records = 10\nwindow_records = 5\nsample_group_records = 5\nsample_groups_per_shard = 1\nshard_records = 5\n";
        let default: Config = toml::from_str(&format!("{base}{replay}")).unwrap();
        assert!(default.checkpoints.prune_old_checkpoints);

        let empty: Config = toml::from_str(&format!("{base}[checkpoints]\n{replay}")).unwrap();
        assert!(empty.checkpoints.prune_old_checkpoints);

        let disabled: Config = toml::from_str(&format!(
            "{base}[checkpoints]\nprune_old_checkpoints = false\n{replay}"
        ))
        .unwrap();
        assert!(!disabled.checkpoints.prune_old_checkpoints);
    }
}
