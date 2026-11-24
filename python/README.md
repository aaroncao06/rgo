# Python trainer components

Shared model definitions live in `model/`, training code in `trainer/`, and
Python helpers invoked by the Rust server in `server/`.

Use [uv](https://docs.astral.sh/uv/) from the repository root. `pyproject.toml`
declares Python 3.9–3.12 and pins NumPy 1.26.4 with PyTorch 2.2.2; `uv.lock`
records their resolved dependencies. The default local Python is 3.12.
PyTorch 2.2.2 cannot convert arrays from NumPy 2.x.

From the repository root:

```sh
uv sync --locked
PYTHONPATH=python:python/tests uv run --locked python -m unittest discover -s python/tests -p 'test_*.py'
```

`TrainingCoordinator` in `trainer/coordinator.py` connects indexed shard paths,
one-shard training, and local recovery checkpoints. `trainer.runner.run` adapts
it to newline-delimited JSON streams: it reports `ready`, requests the next
shard index, accepts either an absolute local path or
`finish`, and reports checkpoint completion. Requesting the next index implies
that the previous shard finished training. Checkpoints are immutable
`<last-shard-index>.pt` files under `checkpoints/` in the supplied trainer
directory.

The trainer pipe worker takes its TOML and a work directory:

```sh
PYTHONPATH=python uv run --locked python -m trainer.main configs/trainer.toml data/run-a/trainer
```

Its first stdin message is `start` with an absolute `checkpoint_path` or
`null`. A `null` path starts a fresh run. The trainer saves an initial `-1.pt`
and reports `checkpoint_ready` before requesting shard 0. The TOML specifies
the model, device, learning rate, and batch sizes. The current worker builds
the temporary `DummyModel` and Adam optimizer.

Rust client supervision is planned. The client will recover pending
checkpoints, obtain the resume path from the server, and wait for checkpoint
acceptance and any due ONNX export before delivering the next shard.

`server.export` converts a checkpoint using the same temporary `DummyModel`
architecture into a float32 ONNX model with dynamic batch and board dimensions
and the engine's `rgo.io_version = "0"` metadata:

```sh
PYTHONPATH=python uv run --locked python -m server.export data/run-a/server/checkpoints/-1.pt data/run-a/server/models/0.onnx
```

The output directory must exist. Export syncs a temporary file, publishes the
final filename without overwriting an existing model, and syncs its directory
on POSIX. ONNX Runtime is a development dependency used to test numerical
agreement with PyTorch. The Rust server invokes this helper synchronously on
its storage worker after accepting an eligible checkpoint and during recovery.
Checkpoint acceptance replies after any due export finishes. It uses the project's
`.venv` interpreter and `python/` source directory from the repository root;
only publication cadence is configured under `[model_export]`. Old-model pruning
remains deferred.
