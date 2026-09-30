# CelSMP

`.csmp` is a storage format for a trained model. One file holds the weights and
states, in a readable header, what is known about them: training and
validation loss, perplexity, whether the weights went through SFT, the
training stage, step and token counts, and how the tensors are compressed.

A `.pt` checkpoint is a pickle: you must load it, and trust it, to learn
anything about it. A `.csmp` answers those questions from its header, runs no
code when read, and checks every tensor against a sha256.

```text
$ python -m celsmp info checkpoints/cellm-civ-100m-job-12696.csmp
file            checkpoints/cellm-civ-100m-job-12696.csmp  (458.57 MiB, CSMP 1.0, 2026-09-30T14:40:34+00:00)
model           cellm-civ-100m cellm-civ 99,338,501 parameters
SFT             no  (regime: pretrain)
training        stage=society  step=74,294  tokens=1,217,232,896  train_seconds=43273.0  dataset=base-packed-16384
train loss      n/a   perplexity n/a
val loss        4.4289   perplexity 83.84
other metrics   val_loss_backbone_only=6.1386  val_tokens=32,768
compression     raw (lossless), 458.39 MiB -> 458.39 MiB (1.00x)
[weights]       43 tensors, 378.95 MiB
[civilization]  89 tensors, 79.45 MiB
```

## Install

```bash
pip install -e .            # NumPy only: save, load, read_header, describe, verify
pip install -e ".[torch]"   # plus torch tensors, pack and unpack
```

## Library

```python
import celsmp

# write: any nested structure of dicts, lists, tuples, scalars and tensors
celsmp.save("model.csmp",
            {"weights": model.state_dict(), "optimizer": optimizer.state_dict()},
            model={"name": "toy", "architecture": "cellm-civ"},
            training={"sft": False, "stage": "society", "step": 74294},
            metrics={"train_loss": 3.91, "val_loss": 4.43})

# inspect: header only, no tensor is read
header = celsmp.read_header("model.csmp")
header["metrics"]["val_perplexity"], header["training"]["sft"]
print(celsmp.describe("model.csmp", config=True))   # summary plus the model configuration

# read
csmp = celsmp.load("model.csmp", sections=["weights"], as_torch=True)
model.load_state_dict(csmp.weights)
print(csmp.describe(tensors=True))                   # same summary from a loaded file

# convert an existing checkpoint file, and back
celsmp.pack("checkpoint.pt", run_dir="runs/civ-100m", name="cellm-civ-100m")
celsmp.unpack("checkpoint.csmp", "restored.pt")
```

| Function | Purpose |
| --- | --- |
| `save(path, sections, *, model, training, metrics, codec, quality, table, notes)` | Write a file. `sections["weights"]` is mandatory. Returns the header. |
| `load(path, sections=None, *, as_torch=False, mmap=False, check=True)` | Read a file into a `CSMP` (`.header`, `.sections`, `.weights`, `.metrics`, `.training`, `.model`, `.compression`). |
| `read_header(path)` | Metadata only. |
| `describe(path, tensors=False, config=False)` | The summary shown above, as a string; `tensors` lists every tensor, `config` the model configuration. Also a method: `csmp.describe()`, and `print(csmp)`. |
| `verify(path)` | Check the header and every tensor sha256; raises `CSMPError`. |
| `pack(checkpoint, output=None, *, sft, run_dir, metrics, with_optimizer, codec, ...)` | `.pt` file to `.csmp`. Returns the output path. |
| `unpack(path, output=None)` | `.csmp` back to the `.pt` it came from. |
| `from_checkpoint(payload)`, `to_checkpoint(csmp)` | The same conversion on in-memory dicts. |
| `run_metrics(run_dir)` | Last records of a CelLM-Civ run's `metrics.jsonl` and `eval.jsonl`. |

`training["sft"]` must be a bool and defaults to `False`. Perplexity defaults
to `exp(loss)` when only the loss is given. Supported dtypes: `bool`,
`int8`–`int64`, `uint8`–`uint64`, `float16`, `float32`, `float64`.

`pack` understands CelLM-Civ checkpoints (`format: cellm-civ/1`: weights,
civilization, trainer counters, config) and plain state dicts, either bare or
under `model`, `state_dict` or `model_state_dict`. It loads checkpoints with
`weights_only=True`; pass `unsafe_pickle=True` for a trusted file that needs
more. The optimizer is left out unless `with_optimizer=True`.

## Command line

The same operations from a shell:

```bash
python -m celsmp pack checkpoint.pt --run-dir runs/civ-100m --name cellm-civ-100m
python -m celsmp pack checkpoint.pt --train-loss 3.91 --val-loss 4.43 --sft
python -m celsmp pack checkpoint.pt --codec dct --quality 90   # lossy, opt-in
python -m celsmp info model.csmp [--tensors] [--config] [--json]
python -m celsmp verify model.csmp
python -m celsmp unpack model.csmp -o checkpoint.pt
```

## Agent skill

[skills/csmp/](skills/csmp/) is a skill for coding agents: how to inspect,
produce and load `.csmp` files without reading tensors they do not need. For
Claude Code, link or copy the folder into `.claude/skills/`.

## Compression

| Codec | Kind | What it does |
| --- | --- | --- |
| `raw` (default) | lossless | Tensor bytes as they are, 64-byte aligned, memory-mappable. |
| `zlib` | lossless | Deflate on every tensor. |
| `dct` | **lossy** | JPEG pipeline on the weights; everything else stored with `zlib`. |

`dct` views each float tensor of the `weights` section as a plane
(`shape[0]` × the rest), cuts it into 8×8 blocks, applies an orthonormal DCT,
divides by a quantization table scaled by `--quality` (1–100, the IJG rule),
rounds, reorders in zigzag and deflates. Tensors with fewer than 8×8 elements,
non-float tensors and all other sections (civilization, optimizer) stay
lossless. The file records that it is lossy (header and preamble flag), the
table used, and the relative error of every tensor.

Weights are not images. The JPEG luminance table (`--table jpeg`) quantizes
high frequencies coarsely, which the eye forgives and a model does not, so the
default table is `flat`: the same step for every frequency.

Measured on the 99.3M-parameter CelLM-Civ checkpoint of job 12696, validation
loss on 32,768 held-out tokens (original weights: 4.4289):

| Codec | Weights | Whole file | Relative RMSE | Val loss | Δ |
| --- | --- | --- | --- | --- | --- |
| `raw` | 379 MiB | 459 MiB | 0 | 4.4289 | 0 |
| `zlib` | 351 MiB | 417 MiB | 0 | 4.4289 | 0 |
| `dct` q90 flat | 81 MiB | 146 MiB | 0.027 | 4.4463 | +0.017 |
| `dct` q95 jpeg | 75 MiB | 141 MiB | 0.061 | 4.5127 | +0.084 |
| `dct` q90 jpeg | 58 MiB | 124 MiB | 0.122 | 4.7795 | +0.351 |
| `dct` q50 flat | 40 MiB | 106 MiB | 0.145 | 4.9465 | +0.518 |
| `dct` q75 jpeg | 34 MiB | 100 MiB | 0.300 | 7.0984 | +2.669 |

One checkpoint, one validation sample: treat the numbers as a guide and
re-measure the loss of any lossy file you intend to keep. The metrics in the
header always describe the weights *before* lossy compression.

## File layout

All integers are little-endian.

| Offset | Size | Field |
| --- | --- | --- |
| 0 | 8 | Magic `89 43 53 4D 50 0D 0A 1A` (`\x89CSMP\r\n\x1a`) |
| 8 | 2 | Major version (1). A reader rejects a major it does not know. |
| 10 | 2 | Minor version (0) |
| 12 | 4 | Flags. Bit 0: at least one tensor is stored lossily. |
| 16 | 8 | Header length `H` in bytes |
| 24 | 32 | sha256 of the header |
| 56 | `H` | Header: UTF-8 JSON |
| … | | Zero padding to a 64-byte boundary: start of the data region |
| data | | Tensor payloads, each starting on a 64-byte boundary |

Header keys:

| Key | Content |
| --- | --- |
| `format`, `version`, `created` | `"csmp"`, `"1.0"`, UTC timestamp |
| `model` | `name`, `architecture`, `parameters`, `config`, anything else |
| `training` | `sft` (bool, always present), `regime`, plus `stage`, `step`, `tokens`, `train_seconds`, `dataset`, `state` when known |
| `metrics` | `train_loss`, `train_perplexity`, `val_loss`, `val_perplexity` (always present, `null` when unknown), plus extras |
| `compression` | `codec`, `lossy`, `raw_bytes`, `stored_bytes`, `ratio`; for `dct` also `quality`, `table`, `qtable` (64 steps, row-major), `rel_rmse`, `snr_db` |
| `source` | File name, format, size and sha256 of the checkpoint it was packed from |
| `notes` | Free text |
| `sections` | One JSON tree per section; a tensor is `{"$tensor": i}` |
| `tensors` | Entry `i`: `section`, `dtype`, `shape`, `codec`, `offset` (from the data region start), `nbytes`, `sha256` of the stored bytes; for `dct` also `scale`, `levels`, `rel_rmse` |

JSON cannot express everything a checkpoint holds, so trees use four markers:
`{"$tensor": i}`, `{"$tuple": [...]}`, `{"$float": "nan"|"inf"|"-inf"}` and
`{"$items": [[key, value], ...]}` for dicts whose keys are not plain strings
(an optimizer's integer parameter ids, for instance).

A `dct` payload is the deflate of 64 planes of quantized coefficients, one per
frequency in zigzag order, each plane listing the blocks row by row, as `int8`
or `int16` (`levels`). To decode: multiply by `qtable`, apply the inverse DCT,
crop the edge padding, multiply by `scale`.

## Limits

- `bfloat16` tensors are rejected; cast to `float32` first.
- A `.csmp` is written whole; there is no append or in-place update.
- `mmap=True` maps `raw` tensors read-only and skips their checksum.
- Corruption of the padding between tensors is not detected; corruption of the
  header or of any tensor is.

## Tests

```bash
python -m pytest -q
```

## License

Apache-2.0. The JPEG luminance quantization table is from ITU-T T.81, Annex K.
