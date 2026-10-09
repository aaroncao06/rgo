//! Server process owning replay storage and sampling.

#[allow(
    dead_code,
    reason = "replay storage awaits server startup and transport"
)]
mod replay;

fn main() {
    eprintln!("rgo-server: server startup and transport are not implemented yet");
    std::process::exit(1);
}
