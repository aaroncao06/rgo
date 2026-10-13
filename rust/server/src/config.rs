use crate::{Error, replay::ReplayConfig};
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
