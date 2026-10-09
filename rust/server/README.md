# Server replay core

`rgo-server` is a server binary with replay storage and sampling implemented as
an internal module. Startup, client supervision, transport, model publication,
and workload scheduling are later steps. Running the binary currently reports
that startup and transport are not implemented and exits with status 1. It
depends on the shared artifact schema and performs no neural compute.
Replay accepts the same 9x9, 13x13, and 19x19 boards as self-play and inference.

The replay implementation separates its responsibilities:

- [replay.rs](src/replay.rs) defines configuration, store state, and errors.
- [chunk_reader.rs](src/replay/chunk_reader.rs) validates incoming chunks and
  exposes borrowed record fields using the shared artifact schema.
- [storage.rs](src/replay/storage.rs) handles intake, retention, and recovery.
- [sampling.rs](src/replay/sampling.rs) selects chunk groups and record indices,
  then publishes sampled shards.
- [shard.rs](src/replay/shard.rs) writes the selected fields as NPZ arrays.

Tests are grouped under `src/replay/tests/` by sampling, shard export, storage,
and recovery, with common fixtures in `src/replay/tests.rs`.
Reader tests live separately in `src/replay/chunk_reader/tests.rs`.

## Storage

Open `ReplayStore` with an existing, exclusively owned directory and explicit
`ReplayConfig` limits. The directory contains `replay.sqlite`, `replay.lock`,
and server-owned files under `chunks/`. SQLite is bundled, so building requires
a C compiler but no separately installed SQLite library.

Input files come from trusted producers whose writers bound chunk sizes.
Replay imposes no separate input byte limit. Self-play's per-game chunks are
capped at 1,024 records, fitting under 1 MiB even for all-19x19 chunks;
fixed-record chunks use their configured record count.

The caller must provision the root directory durably before opening the store.
On Unix, persist newly created directory entries by syncing their parents
during provisioning. The store creates its own contents and syncs the root;
it does not create or open ancestor directories.

- `ingest_file(id, path)` takes ownership of a finished, immutable regular file
  by hard-linking it into replay storage without copying its bytes. The intake
  file must be durably published on the same filesystem, outside `chunks/`.
  On Windows, intake files must allow write access for synchronization.
  After durable acceptance, the store removes the intake filename. The local
  client must not delete or modify accepted files.
  New IDs reject symlinks and receive format and checksum validation before
  hard-linking. Accepted immutable files are not revalidated during sampling
  or reopening.
- The first accepted ID wins, including after restart or eviction. A later
  intake file under that ID is discarded without reading it, regardless of its
  contents. If the intake filename is already gone, a retained receipt returns
  `AlreadyAccepted`; an unknown ID with a missing file remains an error.
- `set_window(records)` changes the active window within `capacity_records`.
  The window is a soft record target, rounded up to whole newest chunks. An
  oldest chunk is evicted only when newer files still cover the target. Every
  record in a retained file is eligible for sampling; a chunk larger than the
  target remains whole. `capacity_records` caps the configured target, so the
  actual retained count may exceed it by less than one chunk's record count.
  Increasing the window permits new arrivals to fill it; it cannot restore
  records already evicted.

The file and replay directory are synced before the acceptance/retention
transaction commits. SQLite uses DELETE journaling with `cache_spill = OFF`, so
modified catalog pages remain in memory until commit. The store explicitly syncs
the catalog directory after SQL updates and before commit on Unix, ensuring the
rollback journal's filename is durable before database pages are overwritten.
Schema creation uses the same transaction path. SQLite uses `synchronous = EXTRA`
to sync rollback-journal deletion before commit returns. Catalog changes are
durable before intake or evicted chunk files are removed. The store syncs the
catalog directory after each write commit on Unix because SQLite can silently
skip its directory sync if opening the directory fails. On macOS, SQLite also
uses `fullfsync = ON` for journal/database writes, and the explicit directory
sync flushes journal deletion through the drive's cache before file cleanup.
`fullfsync` is enabled before `synchronous`: preparing the latter can trigger
hot-journal recovery, which must fully sync the restored database on macOS.
The intake filename remains available until registration and eviction cleanup
succeed, then is removed and its directory synced before acknowledgment.
On reopening, the store loads chunk metadata from SQLite and checks file
availability without reading their payloads. It finishes pending deletions and
removes uncommitted replay links while preserving their intake files for retry.
Recovery runs only during open, and any failure returns no usable handle.
Directory synchronization is performed on Unix. The Windows path currently lacks
directory-sync barriers and does not provide the same power-loss durability
guarantee.

After a storage mutation fails, every operation (including sampling and stats)
returns `ReplayError::ReopenRequired`. Drop the handle, reopen the store, and
retry ingestion with the same ID: a failed operation may have committed before
reporting an error. Input validation errors, source-file read errors, and shard
delivery failures leave the handle usable. `stats()` returns a `Result` so it
cannot report stale state from a failed mutation.

The in-memory catalog keeps chunk IDs, byte lengths, and record counts.
Record indices and tensor bytes are loaded only for a sampled group.
Intake stages one additional chunk before
eviction; failed cleanup requires reopening before further intake. Acceptance
receipts remain in SQLite after file eviction so retries never reintroduce old
data; receipt metadata grows with the number of accepted chunks.

## Sampling

`sample_group_records` is the soft input-record target per group.
`sample_groups_per_shard` bounds the number of fresh groups contributing to one
shard; the example configuration uses four. `shard_records` sets the output-row
target for every shard. All three must be positive and are independent of the
replay window target.

`sample_into(rng, output)` writes one uncompressed NPZ. It requires an
empty, seekable output and returns the actual number of selected rows:

1. Shuffle the current window's chunk references once for this request.
2. Take consecutive whole chunks until the per-group input target is reached,
   including the entire final chunk. Repeat for up to the configured number of
   groups, without reusing chunks. There are at most as many groups as output
   rows; a small window can provide fewer groups or a smaller final group.
3. Allocate output rows in proportion to each group's input-record count.
   Recalculate from the remaining output and input counts, with integer rounding;
   the final group receives the remainder. If selected input is insufficient,
   retain it all and return fewer rows.
4. Load one group into a contiguous byte buffer, construct record offsets, and
   uniformly select its quota without replacement. Copy only selected packed
   records into the shard buffer, then free that group's source bytes and indices.
5. Shuffle the accumulated selected-record offsets, partition by board size,
   and write the NPZ arrays from borrowed views of the selected bytes.

Every request starts with the then-current window. There is no pass order
retained between requests. Rows are unique within a shard and can recur in
later shards; replay files remain intact. Selection is uniform within each
group, with approximately equal retention fractions subject to quota rounding.
This approximates mixing across the window rather than independent uniform
window-wide draws. More groups spread a shard's rows across more source chunks
at the cost of more file reads; they do not multiply the output row target.

Sampling holds the exclusive store handle for the entire shard, including all
groups and publication. Ingestion and eviction run between sampling requests,
so selected files cannot disappear between groups. No deferred eviction queue
or per-file pins are needed.

Memory holds one loaded source group, its row indices, the accumulated selected
packed records and their offsets, borrowed views grouped by board size, and a
small write buffer. The soft input target controls memory approximately:
19x19 source records occupy 970 bytes each, and smaller boards use less. On a
64-bit platform, both source-record offsets and selected-record offsets use
8 bytes each. Selected-output capacity is reserved once per group before copying
its records. Source files are read once per group without checksum/header
revalidation. Only selected records are copied; array fields are gathered from
their packed bytes without materializing separate full tensor arrays.

`sample_file(rng, destination)` syncs and atomically publishes the same
NPZ, rejecting an existing destination. The directory must exist. Failed
sampling or delivery leaves replay usable; `sample_into` may leave partial
bytes that the caller must discard. A directory-sync failure can leave a
published file, so callers must handle that destination before retrying.
Delivery files belong to the caller and must be removed after consumption.

## NPZ arrays

`format_version` is a scalar `uint32` with value 1. `board_sizes` is a sorted
`uint8` array listing the board dimensions present. For dimension `S`, arrays
use keys such as `9/spatial`, with `N` selected rows in that group:

- `S/spatial`: `uint8 [N, 3, ceil(S*S/8)]`, three bit-packed planes.
- `S/global`: `float32 [N, 2]`, komi and consecutive ending passes.
- `S/policy`: `float16 [N, S*S+1]`, including pass in the last column.
- `S/value`: `float32 [N, 2]`, player-relative win and final score targets.
- `S/ownership`: `uint8 [N, ceil(2*S*S/8)]`, two-bit ownership labels.

Packing matches the `.rgo` schema: row-major cells, least significant bits
first, independently byte-aligned spatial planes. The exporter preserves field
values and performs no unpacking, normalization, or augmentation. All arrays
in a board group share the same shuffled row order. Absent sizes have no arrays.
There is one NPZ per shard, including when multiple board sizes are present.

Python can inspect the groups with NumPy:

```python
import numpy as np

with np.load("shard.npz", allow_pickle=False) as shard:
    for size in shard["board_sizes"]:
        policy = shard[f"{int(size)}/policy"]
        print(int(size), policy.shape[0])
```

The trainer and its unpacking/minibatch loader are not implemented yet.
Canonical self-play and replay storage continue to use `.rgo` files.

Filesystem and SQLite operations are synchronous. The future async server
should serialize them on a storage worker with bounded requests. The caller
owns RNG initialization and lifetime; a failed delivery may advance the RNG
even though replay contents remain unchanged.

Run the storage tests independently of the engine and ONNX Runtime:

```sh
cargo test --manifest-path rust/Cargo.toml -p rgo-server
cargo clippy --manifest-path rust/Cargo.toml -p rgo-server --all-targets -- -D warnings
```
