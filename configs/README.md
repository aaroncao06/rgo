# Self-play worker configuration

[`self_play.toml`](self_play.toml) is a runnable CPU configuration for the
`rgo-selfplay` worker. From the repository root:

```sh
mkdir -p models
cargo run --release --manifest-path rust/Cargo.toml -p rgo-selfplay -- configs/self_play.toml
```

The executable takes exactly one configuration path. Directory paths are
relative to the worker's working directory, not the configuration file. The
model directory must already exist; it can be empty while the worker waits for
a model. The worker creates the output directory. Use one worker process per
output directory because its writer uses a shared temporary filename.

The executable is designed to be launched by a supervising process. It uses
local stdin/stdout pipes; the parent publishes models, consumes chunks, and
handles any remote communication. A production supervisor and trainer are not
included yet.

## Settings and defaults

The top-level operational settings are required:

- `worker_threads`: positive number of self-play worker threads.
- `workers_per_thread`: positive number of game tasks per worker thread. The
  example's `2 * 4` configuration runs eight tasks.
- `output_dir`: directory for published `chunk-<32-hex-digit-id>.rgo` files.
- `chunk`: publication mode, described below.
- `inference`: model directory, executor configurations, and cache settings.

The `[inference]` table contains:

- `model_dir`: directory containing atomically published `<version>.onnx`
  models, where the version is an unsigned 64-bit integer.
- `cache_capacity` and `num_cache_shards`: evaluation-cache dimensions. The
  example uses 65,536 entries and 16 shards; both must be powers of two, with
  the shard count no larger than the capacity.
- `executors`: one or more `[[inference.executors]]` entries. Each executor has
  a `device`, positive `base_batch_size`, and optional `intra_threads`.

`intra_threads` is a session setting on each executor, defaults to 1, and must
be between 1 and 2,147,483,647. It controls ONNX Runtime CPU work, including CPU
fallback operations for GPU providers; it does not control GPU kernel threads.

CPU devices use `device = { type = "cpu" }`. CUDA devices use
`device = { type = "cuda", device_id = 0 }` and require a CUDA-enabled build
and compatible ONNX Runtime. Device IDs must be nonnegative.

`base_batch_size` is the maximum batch size for 19x19 positions. Smaller boards
use `floor(base_batch_size * 361 / board_dim^2)`: a base of 8 permits 35 positions
at 9x9, 17 at 13x13, or 8 at 19x19. Requests are batched by board size. Executors
run with available requests without waiting for a full batch; all executors use
the announced model.

Optional algorithm settings inherit Rust defaults, including when only some
fields are overridden:

- `[self_play.rules]`: `board_dim = 9`, `komi = 7.5`, and
  `multi_stone_suicide_legal = true`. Self-play supports board dimensions 9, 13,
  and 19. Rules use area scoring and positional superko.
- `[self_play]`: `randomize_inference_symmetry = true` and a fixed search budget
  of 512 non-root nodes with a 1,024-playout safety limit per move.
- `[search]`: defaults from
  [`SearchParams::KATAGO_SELFPLAY8_MAIN_B18`](../rust/engine/src/search/params.rs).
  Root symmetry averaging and subtree-value bias are deferred.

Unknown fields and invalid values are rejected before workers start. Search
validation checks numerical/distribution constraints, rather than imposing
recommended tuning ranges. Extremely small Dirichlet concentrations can
underflow during sampling; use the default unless deliberately experimenting
with the sampler. Randomness is controlled by the code-defined `RNG_SEED` in
[`main.rs`](../rust/selfplay/src/main.rs), with separate streams per worker.

To vary the search budget, add probability/budget pairs to `[self_play]`:

```toml
[self_play]
search_budget_policy = [
  { probability = 0.9, budget = { max_nodes = 512, max_playouts = 1024 } },
  { probability = 0.1, budget = { max_nodes = 2048, max_playouts = 4096 } },
]
```

The policy is sampled once per move. Probabilities must be positive, finite,
and sum to one. Each budget's `max_playouts` must be at least its `max_nodes`.

## Chunk publication

The example uses `chunk = { mode = "per_game" }`, publishing one chunk for
each finished game. To combine or split games into fixed-size chunks, replace
that top-level setting with:

```toml
chunk = { mode = "fixed_records", records = 25000 }
```

The record count must fit a positive `u32`; 25,000 is an example, not a tuned
default. Graceful shutdown publishes a final partial chunk if needed.

Workers retain move records until a game finishes. A dedicated OS thread
replays and encodes those records, writes through `std::fs`, syncs the file,
renames it into its final path, and syncs the directory on Unix. Only then does
it report the chunk to the parent. The completed-game queue and event delivery
are bounded, so a slow consumer applies backpressure.

The initial chunk format has a 24-byte header, encoded records, and a 32-byte
SHA-256 checksum over the header and records. The header contains the
`RGOCHNK\0` magic followed by four little-endian `u32` fields: format version
(currently 1), spatial-feature count (3), record count, and global-feature count
(2). Each record contains, in order:

1. Board dimension as `u8`.
2. Three binary spatial planes, packed at one bit per active cell, each starting
   on a byte boundary.
3. Two player-relative global features as little-endian `f32`.
4. Row-major policy probabilities, followed by pass, as little-endian `f16`.
5. Final win target and final score as little-endian `f32`. Win targets are 0
   for loss, 0.5 for draw, and 1 for win; scores are player-relative.
6. Final ownership at two bits per active cell: 0 opponent, 1 neutral, 2 player.

Packed cells occupy the least significant bits first; unused high bits are
zero. Only active board cells are stored. Records occupy 235 bytes at 9x9,
466 at 13x13, or 970 at 19x19. The layout is still under development; refer to
the [encoder](../rust/selfplay/src/training_data.rs) for the current definition.

## Commands and events

Send one JSON command per line on stdin:

```json
{"type":"model_ready","version":42}
{"type":"finish"}
```

Send `model_ready` only after completely publishing `<model_dir>/42.onnx`.
The worker checks that exact file and validates the ONNX contract when loading
it. Duplicate or older version announcements are ignored. Unannounced files
do not trigger model changes; adopted versions are switched at move boundaries
through coordinated model handoff. The model must meet the
[V0 tensor contract](../README.md#model-contract).

`finish` stops new games, drains active games, and flushes pending chunks.
Closing stdin requests the same graceful finish, including when no model has
been announced. Commands are limited to 4,096 bytes including the newline;
malformed JSON, unknown command types, and unknown fields are errors.

Stdout contains newline-delimited JSON events. An illustrative sequence for a
200-record 9x9 chunk is:

```json
{"type":"ready","protocol_version":1}
{"type":"chunk_ready","path":"self_play_chunks/chunk-00000000000000000000000000000001.rgo","bytes":47056,"records":200}
{"type":"stopped"}
```

`ready` means the control interface is ready, not that a model has loaded.
`chunk_ready` reports a fully published chunk, its byte size, and actual record
count. Relative event paths are relative to the worker's working directory.
`stopped` follows all final chunk events after a successful drain.

The parent must continuously drain stdout and stderr and monitor exit status.
Diagnostics go to stderr. Runtime and command failures emit a best-effort
`{"type":"error","message":"..."}` event and exit with status 1; failures
before the event interface starts are reported on stderr. Invalid launch
arguments exit with status 2. `--help` prints usage and exits successfully.

On Unix, SIGINT or SIGTERM requests a graceful finish; a second shutdown signal
forces immediate exit. An in-progress model load can delay graceful shutdown.
There is no force-exit pipe command: a supervisor can send `finish`, then kill
the child if its shutdown deadline expires. Pipe commands and OS signal counts
are independent.
