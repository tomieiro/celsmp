"""Human-readable summary of a ``.csmp`` header."""

from __future__ import annotations

import os
from collections import Counter

from celsmp.container import read_header

_CORE = ("train_loss", "val_loss", "train_perplexity", "val_perplexity")


def _size(count: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if count < 1024 or unit == "GiB":
            return f"{count:.0f} {unit}" if unit == "B" else f"{count:.2f} {unit}"
        count /= 1024


def _show(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return f"{value:,}" if isinstance(value, int) and not isinstance(value, bool) else str(value)


def _flatten(tree, prefix=""):
    for key, value in tree.items():
        if isinstance(value, dict) and value:
            yield from _flatten(value, f"{prefix}{key}.")
        else:
            yield f"{prefix}{key}", value


def describe(source, tensors: bool = False, config: bool = False) -> str:
    """Summarize a file from its header alone.

    ``source`` is a path or an already loaded header. ``tensors`` adds one line
    per tensor; ``config`` adds the model configuration, one value per line.
    """
    path = None if isinstance(source, dict) else source
    header = source if path is None else read_header(path)
    model, training, metrics, compression = (header[k] for k in ("model", "training", "metrics", "compression"))
    where = f"{path}  ({_size(os.path.getsize(path))}, " if path is not None else "("
    lines = [("file", f"{where}CSMP {header['version']}, {header['created']})"),
             ("model", " ".join(filter(None, [model.get("name"), model.get("architecture"),
                                               f"{_show(model.get('parameters'))} parameters"]))),
             ("SFT", f"{'yes' if training['sft'] else 'no'}  (regime: {training['regime']})"),
             ("training", "  ".join(f"{k}={_show(training[k], 1)}"
                                    for k in ("stage", "step", "tokens", "train_seconds", "dataset")
                                    if training.get(k) is not None) or "n/a")]
    for split in ("train", "val"):
        lines.append((f"{split} loss", f"{_show(metrics[f'{split}_loss'])}   "
                                       f"perplexity {_show(metrics[f'{split}_perplexity'], 2)}"))
    extra = {k: v for k, v in metrics.items() if k not in _CORE}
    if extra:
        lines.append(("other metrics", "  ".join(f"{k}={_show(v)}" for k, v in extra.items())))
    kind = (f"dct quality {compression['quality']} ({compression['table']} table), LOSSY, "
            f"relative RMSE {compression['rel_rmse']:.4f}" if compression["lossy"]
            else f"{compression['codec']} (lossless)")
    lines.append(("compression", f"{kind}, {_size(compression['raw_bytes'])} -> "
                                 f"{_size(compression['stored_bytes'])} ({compression['ratio']:.2f}x)"))
    stored, count = Counter(), Counter()
    for entry in header["tensors"]:
        stored[entry["section"]] += entry["nbytes"]
        count[entry["section"]] += 1
    for name in header["sections"]:
        lines.append((f"[{name}]", f"{count[name]} tensors, {_size(stored[name])}"))
    if header.get("source"):
        lines.append(("source", f"{header['source']['file']} ({header['source']['format']})"))
    if header.get("notes"):
        lines.append(("notes", header["notes"]))
    text = [f"{label:<16}{value}" for label, value in lines]
    if config:
        text.append("[config]")
        text += [f"  {key:<44}{value}" for key, value in _flatten(model.get("config") or {})] or ["  n/a"]
    if tensors:
        for entry in header["tensors"]:
            error = f"  rel_rmse={entry['rel_rmse']:.4f}" if "rel_rmse" in entry else ""
            text.append(f"  {entry['section']:<13}{entry['codec']:<5}{entry['dtype']:<8}"
                        f"{str(tuple(entry['shape'])):<22}{_size(entry['nbytes'])}{error}")
    return "\n".join(text)
