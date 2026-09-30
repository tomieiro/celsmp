"""The ``.csmp`` container: a readable JSON header followed by aligned tensors.

    0    8  magic  89 43 53 4D 50 0D 0A 1A  (\\x89 C S M P \\r \\n \\x1a)
    8    2  u16    major version
    10   2  u16    minor version
    12   4  u32    flags (bit 0: some tensor is stored lossily)
    16   8  u64    header length H
    24  32  sha256 of the header bytes
    56   H  header, UTF-8 JSON
    ...     zero padding to a 64-byte boundary = start of the data region
    data    tensor payloads, each starting on a 64-byte boundary

All integers are little-endian. Nothing in the file is executable: there is no
pickle, so reading an untrusted ``.csmp`` cannot run code.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import shutil
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from celsmp import codec as codecs

MAGIC = b"\x89CSMP\r\n\x1a"
VERSION = (1, 0)
ALIGN = 64
FLAG_LOSSY = 1
_PREAMBLE = struct.Struct("<8sHHIQ32s")
_DTYPES = {"bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
           "float16", "float32", "float64"}
_METRICS = ("train_loss", "train_perplexity", "val_loss", "val_perplexity")


class CSMPError(ValueError):
    """The file is not a valid ``.csmp`` or failed an integrity check."""


class CSMP:
    """A loaded file: ``header`` plus the decoded ``sections``."""

    def __init__(self, header: dict, sections: dict, path=None):
        self.header, self.sections, self.path = header, sections, path

    def describe(self, tensors: bool = False, config: bool = False) -> str:
        """Summary of the model, training, metrics and compression (see ``celsmp.describe``)."""
        from celsmp.report import describe
        return describe(self.header if self.path is None else self.path, tensors, config)

    __str__ = describe

    weights = property(lambda self: self.sections["weights"])
    model = property(lambda self: self.header["model"])
    training = property(lambda self: self.header["training"])
    metrics = property(lambda self: self.header["metrics"])
    compression = property(lambda self: self.header["compression"])


def _pad(size: int) -> int:
    return -size % ALIGN


def _is_torch(value) -> bool:
    return type(value).__module__.split(".")[0] == "torch" and hasattr(value, "detach")


def _array(value) -> np.ndarray:
    if _is_torch(value):
        if str(value.dtype) == "torch.bfloat16":
            raise TypeError("bfloat16 tensors are not supported; cast to float32 first")
        value = value.detach().cpu().numpy()
    if value.dtype.name not in _DTYPES:
        raise TypeError(f"unsupported tensor dtype {value.dtype}")
    return value


def _encode_tree(value, add):
    """Python structure -> strict JSON, tensors replaced by ``{"$tensor": i}``."""
    if _is_torch(value) or isinstance(value, np.ndarray):
        return {"$tensor": add(_array(value))}
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"$float": repr(value)}
    if isinstance(value, dict):
        if all(isinstance(k, str) and not k.startswith("$") for k in value):
            return {k: _encode_tree(v, add) for k, v in value.items()}
        return {"$items": [[_encode_tree(k, add), _encode_tree(v, add)] for k, v in value.items()]}
    if isinstance(value, tuple):
        return {"$tuple": [_encode_tree(v, add) for v in value]}
    if isinstance(value, list):
        return [_encode_tree(v, add) for v in value]
    raise TypeError(f"cannot store {type(value).__name__} in a .csmp file")


def _decode_tree(node, tensor):
    if isinstance(node, list):
        return [_decode_tree(v, tensor) for v in node]
    if not isinstance(node, dict):
        return node
    if "$tensor" in node:
        return tensor(node["$tensor"])
    if "$float" in node:
        return float(node["$float"])
    if "$tuple" in node:
        return tuple(_decode_tree(v, tensor) for v in node["$tuple"])
    if "$items" in node:
        return {_key(_decode_tree(k, tensor)): _decode_tree(v, tensor) for k, v in node["$items"]}
    return {k: _decode_tree(v, tensor) for k, v in node.items()}


def _key(key):
    return tuple(key) if isinstance(key, list) else key


def _metrics(metrics: dict | None) -> dict:
    result = {name: None for name in _METRICS}
    result.update(metrics or {})
    for split in ("train", "val"):
        loss = result[f"{split}_loss"]
        if result[f"{split}_perplexity"] is None and loss is not None and loss < 700:
            result[f"{split}_perplexity"] = math.exp(loss)
    return result


def save(path, sections: dict[str, Any], *, model: dict | None = None, training: dict | None = None,
         metrics: dict | None = None, codec: str = "raw", quality: int = 90, table: str = "flat",
         lossy_sections=("weights",), source: dict | None = None, notes: str | None = None) -> dict:
    """Write ``sections`` (``weights`` is mandatory) and return the header.

    ``codec`` is ``raw`` (default), ``zlib`` (lossless) or ``dct`` (lossy,
    JPEG-style). With ``dct`` only float tensors of ``lossy_sections`` with at
    least 8x8 elements are transformed; everything else is stored with zlib.
    """
    if codec not in codecs.CODECS:
        raise ValueError(f"codec must be one of {codecs.CODECS}")
    if "weights" not in sections:
        raise ValueError("a .csmp file needs a 'weights' section")
    training = {"sft": False, **(training or {})}
    if not isinstance(training["sft"], bool):
        raise TypeError("training['sft'] must be a bool")
    training.setdefault("regime", "sft" if training["sft"] else "pretrain")
    qtable = codecs.quant_table(quality, table) if codec == "dct" else None

    path = Path(path)
    data_path = path.with_name(path.name + ".data.tmp")
    temporary = path.with_name(path.name + ".tmp")
    tensors: list[dict] = []
    error = signal = 0.0
    try:
        with open(data_path, "wb") as data:
            trees = {}
            for name, tree in sections.items():
                def add(array: np.ndarray, name=name) -> int:
                    nonlocal error, signal
                    entry = {"section": name, "dtype": array.dtype.name, "shape": list(array.shape)}
                    raw = np.ascontiguousarray(array, dtype=array.dtype.newbyteorder("<"))
                    if codec == "dct" and name in lossy_sections and codecs.dct_eligible(array):
                        payload, params = codecs.encode_dct(array, qtable)
                        restored = codecs.decode_dct(payload, params, qtable, array.shape, array.dtype)
                        squared = float(np.square(restored.astype(np.float64) - array).sum())
                        energy = float(np.square(array.astype(np.float64)).sum())
                        error, signal = error + squared, signal + energy
                        entry.update(codec="dct", **params,
                                     rel_rmse=math.sqrt(squared / energy) if energy else 0.0)
                    elif codec == "raw":
                        payload = raw.tobytes()
                        entry["codec"] = "raw"
                    else:
                        payload = zlib.compress(raw.tobytes(), 6)
                        entry["codec"] = "zlib"
                    data.write(b"\0" * _pad(data.tell()))
                    entry.update(offset=data.tell(), nbytes=len(payload),
                                 sha256=hashlib.sha256(payload).hexdigest())
                    data.write(payload)
                    tensors.append(entry)
                    return len(tensors) - 1
                trees[name] = _encode_tree(tree, add)

        lossy = any(t["codec"] == "dct" for t in tensors)
        raw_bytes = sum(int(np.prod(t["shape"])) * np.dtype(t["dtype"]).itemsize for t in tensors)
        stored = sum(t["nbytes"] for t in tensors)
        compression = {"codec": codec, "lossy": lossy, "raw_bytes": raw_bytes, "stored_bytes": stored,
                       "ratio": raw_bytes / stored if stored else 1.0}
        if codec == "dct":
            compression.update(
                quality=quality, table=table, qtable=qtable, lossy_sections=list(lossy_sections),
                rel_rmse=math.sqrt(error / signal) if signal else 0.0,
                snr_db=10 * math.log10(signal / error) if error and signal else None,
                note="metrics were measured on the weights before lossy compression")
        header = {
            "format": "csmp", "version": "%d.%d" % VERSION,
            "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "model": _encode_tree(model or {}, None), "training": _encode_tree(training, None),
            "metrics": _encode_tree(_metrics(metrics), None), "compression": compression,
            "source": source, "notes": notes, "sections": trees, "tensors": tensors,
        }
        blob = json.dumps(header, ensure_ascii=False, allow_nan=False).encode()
        with open(temporary, "wb") as out, open(data_path, "rb") as data:
            out.write(_PREAMBLE.pack(MAGIC, *VERSION, FLAG_LOSSY if lossy else 0, len(blob),
                                     hashlib.sha256(blob).digest()))
            out.write(blob)
            out.write(b"\0" * _pad(out.tell()))
            shutil.copyfileobj(data, out, 1 << 24)
        os.replace(temporary, path)          # never leave a half-written file
    finally:
        data_path.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
    return header


def _read_header(file) -> tuple[dict, int]:
    preamble = file.read(_PREAMBLE.size)
    if len(preamble) < _PREAMBLE.size or not preamble.startswith(MAGIC):
        raise CSMPError("not a .csmp file (bad magic)")
    _, major, _, _, length, digest = _PREAMBLE.unpack(preamble)
    if major != VERSION[0]:
        raise CSMPError(f"unsupported .csmp major version {major}")
    blob = file.read(length)
    if len(blob) != length or hashlib.sha256(blob).digest() != digest:
        raise CSMPError("header is truncated or corrupted")
    header = json.loads(blob)
    for block in ("model", "training", "metrics"):
        header[block] = _decode_tree(header[block], None)
    start = _PREAMBLE.size + length
    return header, start + _pad(start)


def read_header(path) -> dict:
    """Metadata only; does not touch the tensor data."""
    with open(path, "rb") as file:
        return _read_header(file)[0]


def _payload(file, entry: dict, start: int, check: bool) -> bytes:
    file.seek(start + entry["offset"])
    payload = file.read(entry["nbytes"])
    if len(payload) != entry["nbytes"]:
        raise CSMPError("tensor data is truncated")
    if check and hashlib.sha256(payload).hexdigest() != entry["sha256"]:
        raise CSMPError(f"tensor at offset {entry['offset']} failed its sha256 check")
    return payload


def load(path, sections=None, *, as_torch: bool = False, mmap: bool = False, check: bool = True) -> CSMP:
    """Read a file. ``sections`` limits what is decoded (default: everything).

    ``mmap`` maps ``raw`` tensors read-only instead of copying them; their
    checksums are then not verified.
    """
    with open(path, "rb") as file:
        header, start = _read_header(file)
        qtable = header["compression"].get("qtable")

        def tensor(index: int):
            entry = header["tensors"][index]
            dtype, shape = np.dtype(entry["dtype"]).newbyteorder("<"), tuple(entry["shape"])
            if entry["codec"] == "raw" and mmap:
                array = np.memmap(path, dtype=dtype, mode="r", offset=start + entry["offset"], shape=shape)
            else:
                payload = _payload(file, entry, start, check)
                if entry["codec"] == "dct":
                    array = codecs.decode_dct(payload, entry, qtable, shape, dtype)
                else:
                    if entry["codec"] == "zlib":
                        payload = zlib.decompress(payload)
                    array = np.frombuffer(bytearray(payload), dtype=dtype).reshape(shape)
            if as_torch:
                import torch
                return torch.from_numpy(np.asarray(array).astype(dtype.newbyteorder("="), copy=False))
            return array

        wanted = header["sections"] if sections is None else sections
        missing = set(wanted) - set(header["sections"])
        if missing:
            raise KeyError(f"sections not in file: {sorted(missing)}")
        return CSMP(header, {name: _decode_tree(header["sections"][name], tensor) for name in wanted}, path)


def verify(path) -> dict:
    """Check the header and every tensor checksum; return the header."""
    size = os.path.getsize(path)
    with open(path, "rb") as file:
        header, start = _read_header(file)
        for entry in header["tensors"]:
            if start + entry["offset"] + entry["nbytes"] > size:
                raise CSMPError("tensor data is truncated")
            _payload(file, entry, start, True)
    return header
