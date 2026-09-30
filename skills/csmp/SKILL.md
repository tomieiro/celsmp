---
name: csmp
description: Use when a task touches a .csmp model file or the celsmp library - distributing a model for inference, inspecting a checkpoint's loss, perplexity, SFT flag or training stage, converting .pt checkpoints to .csmp or back, saving or loading weights with celsmp, choosing or evaluating dct (JPEG-style lossy) compression, or verifying file integrity.
---

# Working with `.csmp` files (celsmp)

CelSMP means **Cellular Storage Model Protocol**. It stores and distributes a
model's weights and state behind a JSON header that states
the training loss, validation loss, perplexities, whether the weights went
through SFT, the stage and step, and how the tensors are compressed.

The library is `celsmp` (source in `celsmp/src/celsmp/`, format spec in
`celsmp/README.md`). Check it imports with `python -c "import celsmp"`; if not,
`pip install -e celsmp`. In this workspace the interpreter is
`~/miniconda3/envs/celnn/bin/python`.

## Rule: the header answers most questions

Files are hundreds of MiB. Questions about what a file *is* need only the
header, which is read in milliseconds. Do not call `celsmp.load`, `torch.load`
or `cat` on a `.csmp` to answer them.

| Need | Call | Reads tensors |
| --- | --- | --- |
| Loss, perplexity, SFT, stage, step, compression | `celsmp.read_header(path)` | no |
| A summary to show the user | `celsmp.describe(path)` or `python scripts/csmp_info.py path...` | no |
| Tensor names, shapes, dtypes, sizes | `celsmp.describe(path, tensors=True)` | no |
| Model configuration (hyperparameters) | `celsmp.describe(path, config=True)` | no |
| Summary of a file already loaded | `csmp.describe()` or `print(csmp)` | no |
| Integrity | `celsmp.verify(path)` | hashes all |
| Weights only | `celsmp.load(path, sections=["weights"], as_torch=True).weights` | that section |
| One big tensor without copying | `celsmp.load(path, sections=["weights"], mmap=True)` (raw codec only) | on access |
| Everything, as the original checkpoint dict | `celsmp.to_checkpoint(celsmp.load(path, as_torch=True))` | all |

`scripts/csmp_info.py` (next to this file) prints one line per file, or the
full summary with `--full`; use it to compare several checkpoints at once.

Header fields: `header["metrics"]` always has `train_loss`, `train_perplexity`,
`val_loss`, `val_perplexity` (`None` when unknown) plus extras.
`header["training"]` always has `sft` (bool) and `regime`, and usually `stage`,
`step`, `tokens`. `header["compression"]` has `codec`, `lossy`, `ratio`, and
for `dct` also `quality`, `table`, `rel_rmse`. `header["model"]` has `name`,
`architecture`, `parameters`, `config`.

## Producing a file

From a checkpoint file:

```python
celsmp.pack("run/checkpoint.pt", run_dir="run", name="cellm-civ-100m")     # metrics from run/*.jsonl
celsmp.pack("ckpt.pt", sft=True, metrics={"train_loss": 3.9, "val_loss": 4.4})
```

From live objects, for example at the end of training:

```python
celsmp.save("model.csmp", {"weights": model.state_dict()},
            model={"name": ..., "architecture": ...},
            training={"sft": False, "stage": ..., "step": ...},
            metrics={"train_loss": ..., "val_loss": ...})
```

Before producing one:

- **Never invent metrics.** Pass the losses you measured or were given. A
  missing loss stays `None`; say so rather than filling it in.
- **Ask or confirm `sft`** when it is not evident from the run. It defaults to
  `False`, which is a claim.
- Perplexity is derived as `exp(loss)`; pass it only if measured differently.
- The optimizer is dropped unless `with_optimizer=True`. Keep it when the file
  must resume training.
- `pack` refuses checkpoints that need full unpickling. Use
  `unsafe_pickle=True` only for a file the user trusts.
- Writing is atomic and overwrites the target. Check the output path first.

## Compression

`codec="raw"` is the default and is exact. `zlib` is exact and about 10%
smaller. `dct` is **lossy**: use it only when the user asks for smaller files
and accepts a loss change.

- Start with `codec="dct", quality=90` and the default `flat` table. On the
  100M CelLM-Civ checkpoint that cut the weights from 379 to 81 MiB for +0.017
  validation loss. The `jpeg` table is much worse for weights (+0.35 at q90).
- The header metrics describe the weights *before* compression. After writing
  a lossy file, re-measure the validation loss on it and report both numbers.
- `header["compression"]["rel_rmse"]` is a cheap first check: above about 0.05
  expect a visible loss change.
- Only float tensors of the `weights` section with at least 8x8 elements are
  transformed. Civilization state and optimizer stay exact.
- Keep the lossless original until the lossy file has been evaluated.

## Errors

| Message | Cause |
| --- | --- |
| `CSMPError: not a .csmp file (bad magic)` | Wrong file, or a `.csm` written before the rename: re-pack from the `.pt`. |
| `CSMPError: ... failed its sha256 check` | Corrupted or truncated copy. Re-transfer; do not load with `check=False` to get past it. |
| `TypeError: bfloat16 tensors are not supported` | Cast to `float32` before saving. |
| `ValueError: no state_dict found` | `pack` did not recognise the checkpoint layout; build the sections yourself and call `save`. |

## Changing the library

Tests: `cd celsmp && python -m pytest -q`. A change to the file layout or to
required header keys is a format change: bump the major version in
`container.py`, and update the layout section of `celsmp/README.md`.
