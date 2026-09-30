//! Standalone self-play application built on the reusable engine.

use rgo_engine::{game, inference, search};

/// Change this one value to reproduce or vary all self-play randomness.
const RNG_SEED: u64 = 0;

mod chunk_assembler;
mod chunk_sink;
mod config;
mod orchestrator;
mod params;
mod training_data;
mod worker;

// Configuration and shutdown wiring are the next implementation step.
fn main() {}
