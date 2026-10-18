use rgo_protocol::client_server::valid_name;
use serde::Deserialize;
use std::{
    fs,
    net::SocketAddr,
    path::{Path, PathBuf},
    time::Duration,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub server: SocketAddr,
    pub client_id: String,
    pub selfplay_config: PathBuf,
    pub selfplay_binary: Option<PathBuf>,
    pub model_poll_interval_ms: u64,
    pub request_timeout_secs: u64,
    pub retry_timeout_secs: u64,
}

impl Config {
    pub fn load(path: &Path) -> Result<Self, String> {
        let config: Self =
            toml::from_str(&fs::read_to_string(path).map_err(|error| error.to_string())?)
                .map_err(|error| format!("invalid client config: {error}"))?;
        if !config.server.ip().is_loopback() || !valid_name(&config.client_id) {
            return Err("server must be loopback and client_id must contain 1–64 ASCII letters, digits, '-' or '_'".into());
        }
        if config.selfplay_config.as_os_str().is_empty()
            || config
                .selfplay_binary
                .as_ref()
                .is_some_and(|path| path.as_os_str().is_empty())
            || config.model_poll_interval_ms == 0
            || config.request_timeout_secs == 0
            || config.retry_timeout_secs == 0
        {
            return Err("paths and timing limits must not be empty or zero".into());
        }
        let now = std::time::Instant::now();
        if [
            config.request_timeout(),
            config.retry_timeout(),
            config.poll_interval(),
        ]
        .into_iter()
        .any(|duration| now.checked_add(duration).is_none())
        {
            return Err("timing limit exceeds the platform clock range".into());
        }
        Ok(config)
    }

    pub fn prepare_output_dir(&self) -> std::io::Result<PathBuf> {
        #[derive(Deserialize)]
        struct WorkerOutput {
            output_dir: PathBuf,
        }
        let worker: WorkerOutput = toml::from_str(&fs::read_to_string(&self.selfplay_config)?)
            .map_err(std::io::Error::other)?;
        if worker.output_dir.as_os_str().is_empty() {
            return Err(std::io::Error::other("worker output_dir must not be empty"));
        }
        rgo_fs::prepare_directory(&worker.output_dir)
    }

    pub fn request_timeout(&self) -> Duration {
        Duration::from_secs(self.request_timeout_secs)
    }
    pub fn retry_timeout(&self) -> Duration {
        Duration::from_secs(self.retry_timeout_secs)
    }
    pub fn poll_interval(&self) -> Duration {
        Duration::from_millis(self.model_poll_interval_ms)
    }
}
