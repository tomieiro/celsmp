"""Bridge between PyTorch checkpoints and ``.csmp`` files."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from celsmp.container import CSMP, load, save

_WEIGHT_KEYS = ("model", "state_dict", "model_state_dict")


def from_checkpoint(payload: dict, with_optimizer: bool = False) -> tuple[dict, dict, dict, str]:
    """Checkpoint dict -> (sections, model, training, source format)."""
    import torch

    if payload.get("format") == "cellm-civ/1":
        sections = {"weights": payload["model"], "civilization": payload["civilization"]}
        config, trainer = payload["config"], payload["trainer"]
        training = {key: trainer[key] for key in ("step", "tokens", "train_seconds") if key in trainer}
        stages = config.get("training", {}).get("stages", [])
        index = trainer.get("stage_index", -1)
        if 0 <= index < len(stages):
            training["stage"] = stages[index]["name"]
        training["state"] = trainer
        model = {"architecture": "cellm-civ", "config": config}
        source = "cellm-civ/1"
    else:
        key = next((k for k in _WEIGHT_KEYS if isinstance(payload.get(k), dict)), None)
        weights = payload[key] if key else payload
        if not weights or not all(torch.is_tensor(v) for v in weights.values()):
            raise ValueError("no state_dict found: expected a dict of tensors, or one under "
                             + "/".join(_WEIGHT_KEYS))
        sections = {"weights": weights}
        model, training, source = {"architecture": "unknown"}, {}, "torch-state-dict"
    if with_optimizer and payload.get("optimizer"):
        sections["optimizer"] = payload["optimizer"]
    model["parameters"] = sum(v.numel() for v in sections["weights"].values())
    return sections, model, training, source


def to_checkpoint(csmp: CSMP) -> dict:
    """Rebuild the checkpoint dict a ``.csmp`` was packed from (load it with ``as_torch=True``)."""
    sections = csmp.sections
    if (csmp.header.get("source") or {}).get("format") == "cellm-civ/1":
        return {"format": "cellm-civ/1", "model": sections["weights"],
                "optimizer": sections.get("optimizer"), "civilization": sections["civilization"],
                "trainer": csmp.training["state"], "config": csmp.model["config"]}
    return {"model": sections["weights"], **{k: v for k, v in sections.items() if k != "weights"}}


def run_metrics(directory) -> dict:
    """Last train and validation records of a CelLM-Civ run directory."""
    directory, result = Path(directory), {}
    for stream, fields in (("metrics", {"loss": "train_loss", "ppl": "train_perplexity", "step": "train_step"}),
                           ("eval", {"val_loss": "val_loss", "val_ppl": "val_perplexity", "step": "val_step",
                                     "label": "val_label",
                                     "val_loss_backbone_only": "val_loss_backbone_only"})):
        file = directory / f"{stream}.jsonl"
        if not file.is_file():
            continue
        lines = [line for line in file.read_text().splitlines() if line.strip()]
        if lines:
            record = json.loads(lines[-1])
            result.update({name: record[key] for key, name in fields.items() if key in record})
    return result


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


def pack(checkpoint, output=None, *, name: str | None = None, architecture: str | None = None,
         sft: bool = False, stage: str | None = None, dataset: str | None = None, run_dir=None,
         metrics: dict | None = None, with_optimizer: bool = False, codec: str = "raw",
         quality: int = 90, table: str = "flat", notes: str | None = None,
         source_hash: bool = True, unsafe_pickle: bool = False) -> Path:
    """Convert a PyTorch checkpoint file into a ``.csmp`` and return its path.

    Metrics come from ``run_dir`` (see :func:`run_metrics`) and are overridden
    by ``metrics``. The checkpoint is loaded with ``weights_only=True`` unless
    ``unsafe_pickle`` is set.
    """
    import torch

    checkpoint = Path(checkpoint)
    try:
        payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=not unsafe_pickle)
    except Exception as error:
        if unsafe_pickle or isinstance(error, FileNotFoundError):
            raise
        raise ValueError(f"could not load {checkpoint} without executing pickled code ({error}); "
                         "if you trust this file, retry with unsafe_pickle") from error
    sections, model, training, source_format = from_checkpoint(payload, with_optimizer)
    model.update({k: v for k, v in (("name", name), ("architecture", architecture)) if v})
    training.update({k: v for k, v in (("stage", stage), ("dataset", dataset)) if v}, sft=sft)
    output = Path(output or checkpoint.with_suffix(".csmp"))
    save(output, sections, model=model, training=training,
         metrics={**(run_metrics(run_dir) if run_dir else {}), **(metrics or {})},
         codec=codec, quality=quality, table=table, notes=notes,
         source={"file": checkpoint.name, "format": source_format, "bytes": checkpoint.stat().st_size,
                 "sha256": file_sha256(checkpoint) if source_hash else None})
    return output


def unpack(path, output=None) -> Path:
    """Write the checkpoint a ``.csmp`` was packed from back to a ``.pt`` file."""
    import torch

    output = Path(output or Path(path).with_suffix(".pt"))
    temporary = output.with_name(output.name + ".tmp")
    torch.save(to_checkpoint(load(path, as_torch=True)), temporary)
    os.replace(temporary, output)
    return output
