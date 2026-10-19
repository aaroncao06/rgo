# Python trainer components

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
`finish`, and reports training/checkpoint completion. Checkpoints are immutable
`<last-shard-index>.pt` files in the configured directory. The client
can retry any completed files left there after interrupted acceptance before
starting a new trainer process.

`PYTHONPATH=python uv run --locked python -m trainer.main configs/trainer.toml` runs the
client-supervised trainer pipe worker. Its first stdin message is `start` with
the absolute path returned by the server's `GetLatestCheckpoint` request. Only
when the server has neither an accepted checkpoint nor a published model may
that path be `null`. The trainer then saves an initial `-1.pt` and reports
`checkpoint_ready`. The client submits
it and waits for server acceptance and initial ONNX publication before handling
the next shard request. Later `checkpoint_ready` events use the same serial
client handling: submit and await acceptance before delivering another shard.
If an accepted `-1.pt` is resumed after an interrupted export, the client still
waits for initial ONNX publication before delivering shard 0. The TOML
specifies a model, device, learning rate, checkpoint directory, and batch sizes.
`build_trainer`
currently constructs the temporary `DummyModel` and Adam optimizer; there is
no importable factory name in the TOML. The real ViT model and Rust client
launcher are still to be implemented.
