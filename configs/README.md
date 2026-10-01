# Self-play worker configuration

`self_play.toml` is an example consumed by `SelfPlayConfig::load` in the
`rust/selfplay` worker process. It depends on the reusable `rust/engine` library.
The client launches and supervises this worker; there is no standalone
filesystem-watcher mode. The future client setup interface can construct this
same configuration before launching it.

## Client-to-worker pipe protocol

The internal process launch interface takes one config path:

```sh
rgo-selfplay configs/self_play.toml
```

The worker always uses local stdin/stdout pipes. The client
owns server communication and stages remote artifacts in the configured paths;
shared-storage clients use those paths directly. The model directory must exist,
but self-play does not scan or watch it. The client/server supervisor is not yet
implemented; integration tests currently exercise this boundary as the parent
process. Launch arguments and `--help` are an internal worker interface, rather
than an additional user-facing execution mode.

Send one JSON command per line on stdin:

```json
{"type":"model_ready","version":42}
{"type":"finish"}
```

`model_ready` means the client has completely published `<model_dir>/42.onnx`.
Self-play checks that exact path before forwarding the version to its existing
model-handoff logic. Missing or non-file paths and invalid ONNX models are
errors. Versions must fit an unsigned 64-bit integer; duplicate and older
announcements are ignored. Unannounced files do not trigger model changes.
`finish` stops new games and drains active games and chunks. Closing stdin also
requests a graceful finish, so a disconnected supervisor does not leave an
uncontrolled producer running. Commands are limited to 4096 bytes per line,
including the newline; malformed JSON, unknown commands, and unknown fields
are errors.

Stdout contains only newline-delimited JSON events; diagnostics remain on stderr:

```json
{"type":"ready","protocol_version":1}
{"type":"chunk_ready","path":"self_play_chunks/chunk-<id>.rgo","bytes":2870,"records":2}
{"type":"stopped"}
```

`ready` means the control interface is ready, not that a model has loaded.
`chunk_ready` is emitted only after the complete chunk is atomically published
and synced; its path follows the configured output path and is relative to the
child's working directory when that configuration is relative. `records` is the
actual number of training records, including for a final partial chunk. It comes
directly from the encoder; the client/server can use it for replay inventory and
should validate it against the file header when accepting the chunk. `stopped`
follows all final chunk events after a successful drain. Runtime and command
failures produce a best-effort `{"type":"error","message":"..."}` event and
exit with status 1. Startup failures before the interface starts report on
stderr with a nonzero exit status. The supervisor must monitor exit status and
continuously drain stdout and stderr. Event delivery is bounded and applies
backpressure rather than dropping chunk notifications. SIGINT or SIGTERM on
Unix also requests a graceful finish; a second shutdown signal forces immediate
exit. Signal handling runs independently of model startup: graceful finishing
waits for an in-progress load to return, while forced exit remains available.
The pipe protocol has no force-exit command. The parent requests graceful
shutdown with `finish`, then kills the child directly if a timeout expires or
the user requests forced shutdown. A `finish` command does not count as the
first OS shutdown signal; the direct signal handler counts signals separately.
An empty model directory is valid; the worker waits for its first model-ready
command and can finish while waiting.

## Configuration fields

`SelfPlayConfig` combines operational settings with the existing `SelfPlayParams`,
`SearchParams`, `ModelRuntimeConfig`, and `ChunkMode`; it does not duplicate their
fields. Operational settings are required. Omitted algorithm tables or fields
inherit the defaults defined in Rust (`SelfPlayParams::default()` and
`SearchParams::KATAGO_SELFPLAY8_MAIN_B18`). Unknown fields and invalid values are
rejected before workers are started.

`self_play.rules.board_size` defaults to 9. The engine has runtime square-board
geometry within its storage capacity, but the current model supports only 9x9.
Self-play rejects other board sizes before starting workers; tensor and chunk
dimensions remain unchanged.

Search validation checks numeric and probability/distribution requirements, not
KataGo's recommended tuning bounds. Finite negative bonuses or exploration
coefficients are allowed for experiments. Positive scales/denominators and valid
mixing weights remain required; enabled noise and LCB require positive
concentration and confidence multipliers, respectively.

Known numerical limitation: extremely small root Dirichlet total concentrations
(for example, `0.001`) can underflow every gamma draw to zero. Normalization then
triggers a debug assertion or produces NaN policy entries in release builds.
Validation currently requires a positive concentration but does not rule out
this case. The default `10.83` is not practically affected; robust/log-space
sampling is deferred. Avoid extremely small concentrations in experiments until
the sampler is made robust.

Directory paths are relative to the process working directory, not the config
file. The example assumes the repository root. The model directory must already
exist; the executable creates the output directory before starting the file
sink and syncs newly created directory entries on Unix. Use one sink per output
directory, and publish complete models atomically as `<version>.onnx`.

Each inference executor has its own device and batch limit. CPU thread counts
are explicit. CUDA entries require a CUDA-enabled build and ONNX Runtime.
The operating system/backend still validates whether the requested device and
model can actually be loaded.

`chunk = { mode = "per_game" }` publishes each finished game immediately.
`chunk = { mode = "fixed_records", records = 25000 }` combines/splits games into
fixed-size chunks, with a possible final partial chunk on graceful shutdown.
That number is an example, not a tuned default.

`self_play.search_budget_policy` is a list of probability/budget pairs sampled
per move. Probabilities must be positive and sum to one. If omitted, the policy
is the existing fixed 512-node / 1024-playout budget. Randomness still uses the
existing code-defined seed; there is no second seed setting in the file.
