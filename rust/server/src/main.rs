//! Server process owning replay storage and sampling.

mod config;
mod replay;
mod service;
mod transport;

type Error = Box<dyn std::error::Error + Send + Sync>;

#[tokio::main(flavor = "current_thread")]
async fn main() {
    const USAGE: &str = "Usage: rgo-server <server.toml>";
    let mut args = std::env::args_os().skip(1);
    let Some(path) = args.next() else {
        eprintln!("{USAGE}");
        std::process::exit(2);
    };
    if path == "--help" || path == "-h" {
        println!("{USAGE}");
        return;
    }
    if args.next().is_some() {
        eprintln!("{USAGE}");
        std::process::exit(2);
    }
    let result = match config::Config::load(std::path::Path::new(&path)) {
        Ok(config) => transport::run(config).await,
        Err(error) => Err(error),
    };
    if let Err(error) = result {
        eprintln!("rgo-server: {error}");
        std::process::exit(1);
    }
}
