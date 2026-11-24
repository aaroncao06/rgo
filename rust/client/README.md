# Local client

`rgo-client` supervises one self-play process and talks to `rgo-server` over
loopback TCP. The processes share a filesystem. Remote transfers, scheduling,
and Python trainer supervision are later steps.

Build the workspace so the client, server, and worker executables exist:

```sh
cargo build --release --manifest-path rust/Cargo.toml --workspace
rust/target/release/rgo-server configs/server.toml
```

In another terminal, from the same repository root:

```sh
rust/target/release/rgo-client configs/client.toml
```

The default command is `selfplay`. The server confirms the protocol version
and supplies its active run name in `Welcome`. The client creates
`data_dir/<run_name>/selfplay/`, recovers pending chunks there, and passes that
directory and the worker TOML to `rgo-selfplay`. Other runs' directories are
preserved. The worker directory must share a
filesystem with server replay storage. After the worker's `ready` event, it
immediately polls for the latest published model path and repeats that same request
at the configured interval. Each model notification supplies its complete file
path; the worker derives its version from the canonical `<version>.onnx` basename.
Model polls use `GetLatestModel`. Completed chunks use `SubmitChunk`.
Completed chunk paths are submitted for acceptance one at a time. The server validates new IDs,
durably registers their hard links, and removes the producer filenames. The client
does not delete accepted files.

## Configuration

[client.toml](../../configs/client.toml) requires:

- `server`: loopback socket address matching the server listener.
- `data_dir`: storage root. The example uses `data/`, shared with the server.
- `client_id`: 1–64 ASCII letters, digits, `-`, or `_`, carried in the greeting.
  The server currently only validates it. Concurrent clients need distinct
  work directories. Locks enforce ownership of each run's self-play directory.
- `selfplay_config`: worker TOML path.
- `model_poll_interval_ms`: positive polling interval; the example uses
  1,800,000 (30 minutes). The first poll is immediate after worker readiness.
- `request_timeout_secs`: positive connection/request deadline; the example
  uses 300 seconds (5 minutes). This includes queue wait and server processing.
  Model-version polls bypass the storage queue after the greeting completes.
- `retry_timeout_secs`: positive total budget for connection setup or a retryable
  request; the example uses 600 seconds (10 minutes). It includes attempts and
  backoff waits. Each attempt is capped by the request deadline or the remaining
  budget, whichever is shorter; success starts a fresh budget for the next operation.

Optional `selfplay_binary` selects the worker executable. Otherwise it uses
`rgo-selfplay` beside the client executable, with `.exe` on Windows. Relative
paths refer to the client's working directory; the worker inherits it.
Inference devices and search settings remain in the worker configuration.
Build the worker with the desired provider features independently of the client.

## Replay statistics

This command connects without starting self-play and prints a JSON response:

```sh
rust/target/release/rgo-client configs/client.toml stats
```

It sends `GetReplayStats` to the server.

The server discovers canonical published models on startup and publishes new
exports from accepted checkpoints without restarting. Clients adopt them through
their ordinary model polls.
Manually placed model files still require a server restart for discovery.
The self-play worker validates the
[model contract](../../README.md#model-contract) when loading it.

Indexed `GetTrainingShard` requests are defined by the shared protocol, but trainer
coordination is not wired into this CLI yet. The server returns a stable shard
path for a ready request index and retains it until covered by an accepted
checkpoint. Until enough replay
exists for a new full shard, its response path is `null`. Checkpoint
acceptance and covered-shard cleanup are implemented on the server; client
trainer integration remains deferred.

## Shutdown and recovery

Ctrl-C, or SIGTERM on Unix, sends `finish` to the worker and continues accepting
final chunks until its `stopped` event and successful process exit. A second
signal kills and reaps the worker. Worker failures, explicit
server rejections, invalid protocol messages, and exhausted retry budgets also
stop the child and return a nonzero status. The worker is never automatically
restarted. Stop self-play clients before the server.
Server error responses are received normally and handled by the supervisor or
control command. They are not transport failures and do not themselves discard
the connection or trigger reconnection.

The client holds `.client.lock`; the worker holds `.selfplay.lock` throughout
its drain. A new client refuses recovery while an old worker still owns the output directory,
including after an unexpected parent exit. Restart once that worker exits.
Completed chunks remaining in the output directory are retried before launching a new worker;
durable server receipts make retries safe even after replay eviction.
Connection failures and timeouts discard the socket, wait 1, 2, 4, 8, 16,
then 30 seconds between attempts, and reconnect with a new Hello. If the server
advertises a different run name, the client stops before resending the request.
Restart the client to join that run. Model polls,
chunk acceptance, checkpoint acceptance, shard requests, and stats requests can
be repeated safely. A model poll after reconnection
returns the server's current file path. Healthy workers stay alive during retries, with bounded
queues applying backpressure while acceptance is unavailable. Signals and worker
failure still interrupt supervision. The normal retry budget also applies to
server operations during shutdown; expiry is a fatal error that reaps the worker.
After that, restart the server if needed and run the client again. Active games
can be lost on forced exit; completed,
published chunks remain recoverable.

Shutdown during connection setup or pending-chunk recovery exits without starting
a worker. An in-flight acceptance can still finish on the server and is safe to
retry. On Unix, directory creation syncs newly created parent entries; existing
directories are assumed to be provisioned.

Shared messages and framing live in [rgo-protocol](../protocol/src/lib.rs).
Server requests are newline-delimited JSON with a 64 KiB frame limit. Worker
commands retain their 4,096-byte limit. The loopback interface is intended for
trusted local processes and has no authentication.
