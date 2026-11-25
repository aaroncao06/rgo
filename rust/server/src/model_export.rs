//! Python model initialization and ONNX conversion on the storage worker.

use crate::Error;
use serde::Deserialize;
use std::{
    path::Path,
    process::{Command, Stdio},
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelExportConfig {
    pub every_shards: u64,
}

pub fn initialize_checkpoint(directory: &Path) -> Result<(), Error> {
    run_helper("server.initialize", &[directory])
}

/// The storage worker keeps the input in place until this helper finishes.
pub fn export_checkpoint(checkpoint: &Path, output: &Path) -> Result<(), Error> {
    run_helper("server.export", &[checkpoint, output])
}

fn run_helper(module: &str, paths: &[&Path]) -> Result<(), Error> {
    let python = if cfg!(windows) {
        ".venv/Scripts/python.exe"
    } else {
        ".venv/bin/python"
    };
    let mut command = Command::new(python);
    command
        .args(["-m", module])
        .args(paths)
        .env("PYTHONPATH", "python")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit());
    // Let the server handle terminal shutdown while this operation finishes.
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
        command.creation_flags(CREATE_NEW_PROCESS_GROUP);
    }
    let status = command
        .status()
        .map_err(|error| format!("could not run {module} with {python}: {error}"))?;
    if !status.success() {
        return Err(format!("{module} failed for {paths:?}: {status}").into());
    }
    Ok(())
}
