mod config;
mod connection;
mod supervisor;

use rgo_protocol::server::{Request, Response};
use std::path::Path;

#[tokio::main(flavor = "current_thread")]
async fn main() {
    const USAGE: &str = "Usage: rgo-client <client.toml> [selfplay | announce-model <version> | sample-shard <name.npz> | stats]";
    let mut args = std::env::args_os().skip(1);
    let Some(path) = args.next() else {
        eprintln!("{USAGE}");
        std::process::exit(2);
    };
    if path == "--help" || path == "-h" {
        println!("{USAGE}");
        return;
    }
    let command = args.next().unwrap_or_else(|| "selfplay".into());
    let request = match command.to_str() {
        Some("selfplay") => None,
        Some("stats") => Some(Request::Stats {}),
        Some("announce-model") => args
            .next()
            .and_then(|version| version.to_str()?.parse().ok())
            .map(|version| Request::AnnounceModel { version }),
        Some("sample-shard") => args
            .next()
            .and_then(|name| name.into_string().ok())
            .map(|filename| Request::SampleShard { filename }),
        _ => {
            eprintln!("{USAGE}");
            std::process::exit(2);
        }
    };
    if args.next().is_some() || (command != "selfplay" && request.is_none()) {
        eprintln!("{USAGE}");
        std::process::exit(2);
    }
    let result = async {
        let config = config::Config::load(Path::new(&path)).map_err(std::io::Error::other)?;
        if let Some(request) = request {
            let mut connection = connection::Connection::open(&config).await?;
            let response = connection.request(&request).await?;
            if let Response::Error { message } = response {
                return Err(std::io::Error::other(message));
            }
            println!(
                "{}",
                serde_json::to_string(&response).map_err(std::io::Error::other)?
            );
            Ok(())
        } else {
            supervisor::run(&config).await
        }
    }
    .await;
    if let Err(error) = result {
        eprintln!("rgo-client: {error}");
        std::process::exit(1);
    }
}
