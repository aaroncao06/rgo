# rgo

An experimental Go engine for low-cost 9x9 self-play and training research.

The `rgo-engine` Rust library implements board rules, positional superko,
scoring, ONNX inference with a batched runtime/cache, and graph search using a
selected KataGo self-play baseline. The separate `rgo-selfplay` application
owns game generation, checkpoint watching, configuration, and training-record
output. Its executable loads a TOML configuration, watches for local models,
and finishes active games and drains chunks on Ctrl-C. The Python trainer
remains to be implemented.

The engine has no dependency on self-play. A future interactive player can
reuse the same library. Client/server orchestration will wrap standalone
self-play and training components in separate applications.

## Build and test

Run from the repository root with a Rust toolchain supporting edition 2024:

```sh
cd rust
cargo build --workspace
cargo test --workspace
cargo test --workspace --release
cargo fmt --all -- --check
cargo clippy --workspace --all-targets
```

The crates currently produce warnings for unfinished or unused helpers. Miri
tests that initialize the score-utility table are slow because they interpret its
numerical initialization; there is no separate Miri table-generation path.

## Run local self-play

From the repository root, create the model directory and run:

```sh
mkdir -p models
cargo run --release --manifest-path rust/Cargo.toml -p rgo-selfplay -- configs/self_play.toml
```

Publish completed models atomically as `models/<version>.onnx`. Self-play waits
if the directory is empty, creates the configured chunks directory, and loads
newer models at move boundaries. Ctrl-C (or SIGTERM on Unix) requests a graceful
finish; a second shutdown signal forces immediate exit, including during blocked
model startup. Configuration and runtime errors print a diagnostic and exit with
a nonzero status. See
[configuration details](configs/README.md).

## Source map

| Location | Responsibility |
|---|---|
| `rust/engine/src/lib.rs` | Reusable engine library; game, inference, and search modules |
| `rust/selfplay/src/main.rs` | Configuration loading, directory provisioning, runtime startup, and shutdown signals |
| `rust/selfplay/src/` | Self-play workers, orchestration, checkpoint watching, configuration, and training chunks |
| `rust/engine/src/game/board.rs` | Coordinates, colors, chains, local legality, move application, and position hashing |
| `rust/engine/src/game/board/scoring.rs` | Read-only area scoring and pass-alive analysis |
| `rust/engine/src/game/game_state.rs` | Turns, rules, positional-superko history, scratch reset, and shared current-state hashing |
| `rust/engine/src/game/hash.rs` | Deterministic hash primitives and Zobrist keys |
| `rust/engine/src/inference/{inputs,outputs,policy}.rs` | Model encoding, output postprocessing, and policy indexing |
| `rust/engine/src/inference/backend.rs` | Backend contract |
| `rust/engine/src/inference/runtime.rs` | Model lifetime, client evaluation, and executor loop |
| `rust/engine/src/inference/runtime/{cache,queue}.rs` | Model cache and request/batch synchronization |
| `rust/engine/src/search/` | Worker, graph identity/storage, search statistics, utility, and selection formulas |
| `rust/engine/src/search/worker.rs` | Graph lifecycle, playout execution, and reverse backup walk |
| `rust/engine/src/search/worker/selection_policy.rs` | Complete descent policy: child scanning, PUCT, FPU, exploration scaling, and forced visits |
| `rust/engine/src/search/worker/backup_policy.rs` | Complete parent-estimate policy: transposition contributions, value weighting, and moment aggregation |
| `rust/engine/src/search/worker/root_policy.rs` | Root preprocessing and final selection: weights, reduced-weight/LCB adjustments, fallback, and temperature sampling |
| `rust/engine/src/search/worker/root_endgame.rs` | Root ending-score bonus and useless-move pruning |

The worker executes searches; private policy modules decide which child to
explore, how to estimate a parent's value, and which move to play. Each policy
keeps its implementation together, including its loops and bookkeeping. The
reverse path walk stays in the worker and delegates parent recomputation to the
backup policy. Modules use the existing worker and scratch storage, with no new
runtime objects or dynamic dispatch. Shared numerical helpers stay in
`search/{move_selection,root_policy,utility}.rs`. More fundamental algorithm
changes may still require changes to the core mechanics.

Large test suites live in child `tests.rs` modules; shorter suites remain inline.
Python model/trainer directories are placeholders with no implementation yet.
Client/server orchestration will be built on top of standalone self-play and
training components.

## Documentation

- [Architecture and boundaries](docs/01_PROJECT_OBJECTIVE_AND_ARCHITECTURE.md)
- [Board geometry](docs/02_BOARD_GEOMETRY.md), [chains](docs/03_CHAINS.md),
  [local legality](docs/04_KO_AND_LEGALITY.md), [moves and game end](docs/05_MOVE_AND_GAMEEND.md),
  [superko and hashes](docs/06_SUPERKO_AND_HISTORY.md), and [scoring](docs/07_SCORING.md)
- [Model I/O contract](docs/08_MODEL_IO.md)
- [Search ownership and storage](docs/09_SEARCH_SYNCHRONIZATION.md)
- [Runtime design rationale](docs/11_KATAGO_KZERO_RUNTIME_REVIEW.md)
- [Search formulas](docs/13_KATAGO_SEARCH_UTILITY_AND_SELECTION.md) and
  [implemented/deferred search features](docs/search_feature_review.md)
- Future work: [experiments](docs/12_DEFERRED_EXPERIMENTS.md),
  [distributed orchestration](docs/distributed_selfplay_training.md),
  [interactive application](docs/10_INTERACTIVE_APP.md), and
  [bidirectional-search research](docs/99_BIDIRECTIONAL_SEARCH_RESEARCH.md)
- Historical discussion: [original review worksheet](docs/REVIEW_FEEDBACK.md)

Future plans and historical feedback do not imply that those interfaces are
implemented. The numbered specifications describe the current baseline and
identify deferred work; the source defines the actual internal API.
