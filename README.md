# rgo

An experimental Go engine for self-play and training research, with an initial
focus on low-cost 9x9 experiments. Self-play and inference support 9x9, 13x13,
and 19x19 boards.

The Rust engine implements positional superko, area scoring, batched ONNX
inference with a model-scoped cache, and PUCT graph search. The self-play worker
generates games and writes training chunks under the control of a parent
process. The trainer, supervising client/server, and interactive player are
not implemented yet. No trained model is included; the ONNX files under the
engine's test directory are synthetic fixtures.

## Planned training architecture

The long-term goal is low-cost Go training using heterogeneous compute
contributed by friends or volunteers. The proposed coordination layer would
support a single machine, a fixed allocation across multiple GPUs, and remote
clients through the same component interfaces.

The planned responsibilities are:

- **Server:** own the bounded replay store, sample training shards, publish
  models and recovery checkpoints, track training demand, and assign workloads.
  Neural inference and gradient computation run on clients.
- **Client:** manage a compute cluster that may contain several GPUs and CPU
  workers. It handles hardware allocation, process supervision, and artifact
  transfers. The server assigns work to the client; the client manages execution
  on its hardware.
- **Compute workers:** self-play consumes ONNX models and produces training
  chunks; a Python trainer consumes sampled shards and produces resumable
  checkpoints and exported models. Both use local files and lifecycle controls.
  Clients provide shared paths locally or stage downloaded artifacts remotely.

Initially, one client would train a model at a time while other clients generate
self-play data. Manual assignments would support stable hardware allocations;
an automatic scheduler would balance new data against training samples consumed,
using measured throughput and the self-play cost of assigning a client to
training. Training could migrate at recovery checkpoints, carrying optimizer
state with it. Recovery checkpoints and models published to self-play would
have separate cadences.

The research direction is making these component boundaries work efficiently
across local and distributed deployments. Scheduling, transfer costs, and
learning efficiency still need experimental validation.

The current repository implements the engine and self-play worker. Next steps:

1. Add local client/server coordination, replay retention, and shard sampling.
2. Add the trainer and validate a complete 9x9 training loop.
3. Add remote artifact transfers behind the same worker interfaces.
4. Add workload scheduling and checkpoint-based trainer migration.

## Build and test

Use a recent Rust toolchain supporting edition 2024. From the repository root:

```sh
cargo build --release --manifest-path rust/Cargo.toml --workspace
cargo test --manifest-path rust/Cargo.toml --workspace
cargo test --release --manifest-path rust/Cargo.toml --workspace
cargo fmt --manifest-path rust/Cargo.toml --all -- --check
cargo clippy --manifest-path rust/Cargo.toml --workspace --all-targets
```

CPU inference is enabled by default. The `ort` dependency downloads and copies
ONNX Runtime binaries during the build; the first build needs network access
unless the dependencies and runtime are already cached or supplied locally.
CUDA inference requires the `cuda` feature and a compatible CUDA-enabled ONNX
Runtime installation:

```sh
cargo build --release --manifest-path rust/Cargo.toml -p rgo-selfplay --features cuda
```

CoreML (`--features coreml`) accelerates inference on Apple platforms using
MLProgram format, requiring macOS 12+. WebGPU (`--features webgpu`) uses Dawn
to target Metal, Vulkan, or Direct3D 12. To build both on Apple Silicon:

```sh
cargo build --release --manifest-path rust/Cargo.toml -p rgo-selfplay --features coreml,webgpu
```

Choose the provider per executor in the
[worker configuration](configs/README.md). Hardware smoke tests are opt-in:

```sh
cargo test --manifest-path rust/Cargo.toml -p rgo-engine --features coreml,webgpu matches_cpu_across_dynamic_shapes -- --ignored
```

These use a synthetic model to check the tensor boundary; a trained model's
operator coverage and throughput must be measured separately.

The project currently emits warnings for unfinished helpers and helpers used
only in tests.

## Run the self-play worker

The worker takes one TOML configuration path and communicates through
newline-delimited JSON on stdin and stdout. See
[configuration and the process protocol](configs/README.md).

```sh
mkdir -p models
cargo run --release --manifest-path rust/Cargo.toml -p rgo-selfplay -- configs/self_play.toml
```

The example starts eight self-play tasks on two worker threads and uses one
CPU inference executor. It waits for a `model_ready` command naming a completed
ONNX model in `models/`. Model files are announced explicitly; the worker does
not scan the directory. Paths in the configuration are relative to the process
working directory.

Each completed game is sent to a dedicated chunk-writer thread. The writer
replays its moves, generates position features, and encodes policy targets and
actual final-game outcomes. It uses synchronous file I/O, syncs and atomically
publishes each chunk, then emits `chunk_ready`. A `finish` command or stdin EOF
drains active games and pending chunks before exit.

## Model contract

Models must carry the ONNX metadata property `rgo.io_version = "0"`. All model
inputs and outputs are float32 tensors with a dynamic batch dimension. Spatial
dimensions must also be declared dynamic so the same model can serve all three
board sizes. For a batch of `N` positions on an `S` by `S` board:

- `spatial`: `[N, 3, S, S]`, containing player stones, opponent stones, and
  current positional-superko bans as binary planes.
- `global`: `[N, 2]`, containing player-relative komi and consecutive ending
  passes.
- `policy_logits`: `[N, S*S+1]`, with row-major board moves followed by pass.
- `value`: `[N, 3]`, containing a win logit, raw score mean, and score-deviation
  logit, all relative to the player to move. Score mean is scaled by 20;
  deviation is `softplus(logit) * 20`.
- `ownership_logits`: `[N, 1, S, S]`, with positive values favoring the player
  to move. The engine applies `tanh`; ownership is requested when needed for
  root endgame heuristics.

The adapter validates tensor declarations when loading and output shapes when
evaluating. Legal-move masking, symmetry restoration, and output processing
belong to the engine. See the
[ONNX adapter](rust/engine/src/inference/onnx.rs),
[input encoder](rust/engine/src/inference/inputs.rs), and
[output processing](rust/engine/src/inference/outputs.rs).

## Project layout

- [`rust/artifacts`](rust/artifacts/src/lib.rs): shared model/chunk identities
  and filename conventions, plus the binary chunk schema, record sizes, header
  encoding, checksum verification, and validated borrowed record views.
- [`rust/engine`](rust/engine/src/lib.rs): reusable board, inference, and search
  library, independent of self-play.
- [`rust/selfplay`](rust/selfplay/src/main.rs): process configuration, game
  workers, model handoff, pipe protocol, and chunk publication.
- [`configs`](configs/README.md): example self-play configuration and usage.

Search currently resets its graph for every move. The search defaults are
defined in [SearchParams](rust/engine/src/search/params.rs); they follow a
selected KataGo self-play baseline rather than implementing all KataGo
features. Training chunks store compact inputs, FP16 policies, and
player-relative final win/score/ownership labels. The initial chunk layout is
still under development; the shared schema is in
[rgo-artifacts::chunk](rust/artifacts/src/chunk.rs), and its self-play encoder is in
[chunk_encoder.rs](rust/selfplay/src/chunk_encoder.rs).

## License and acknowledgments

rgo is licensed under the [MIT License](LICENSE).

Thanks to David J Wu (lightvector) and the
[KataGo contributors](https://github.com/lightvector/KataGo) for their work on
Go search and self-play training. KataGo informs our search policies, root
heuristics, and regression tests. Its notice for adapted portions is preserved
in [third-party notices](THIRD_PARTY_NOTICES.md).
