# Local self-play configuration

`self_play.toml` is an example consumed by `SelfPlayConfig::load`. Executable
configuration and shutdown wiring in `main` are not implemented yet.

`SelfPlayConfig` combines operational settings with the existing `SelfPlayParams`,
`SearchParams`, `ModelRuntimeConfig`, and `ChunkMode`; it does not duplicate their
fields. Operational settings are required. Omitted algorithm tables or fields
inherit the defaults defined in Rust (`SelfPlayParams::default()` and
`SearchParams::KATAGO_SELFPLAY8_MAIN_B18`). Unknown fields and invalid values are
rejected before workers are started.

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
exist; the file sink creates the output directory. Use one sink per output
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
