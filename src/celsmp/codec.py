"""Tensor payload codecs: ``raw``, ``zlib`` (lossless) and ``dct`` (lossy).

``dct`` follows the JPEG pipeline on a tensor viewed as a 2-D plane:
8x8 blocks -> orthonormal DCT-II -> division by a quality-scaled quantization
table -> rounding -> zigzag order -> deflate. Weights are not images: the JPEG
luminance table (``jpeg``) spends its error on high frequencies that matter as
much as the low ones, so the default table is ``flat`` (one step for every
frequency). The table actually used is written to the file, so a decoder never
needs to know how it was chosen.
"""

from __future__ import annotations

import zlib

import numpy as np

CODECS = ("raw", "zlib", "dct")
TABLES = ("flat", "jpeg")

# Luminance quantization table from the JPEG standard (ITU-T T.81, Annex K).
_JPEG_LUMA = np.array([
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
], dtype=np.int64)

_ZIGZAG = np.array(sorted(range(64), key=lambda i: (
    i // 8 + i % 8, i // 8 if (i // 8 + i % 8) % 2 else -(i // 8))))

_n = np.arange(8)
_DCT = np.sqrt(2 / 8) * np.cos((2 * _n[None, :] + 1) * _n[:, None] * np.pi / 16)
_DCT[0] = np.sqrt(1 / 8)
_DCT = _DCT.astype(np.float32)

# A sample is mapped so that 4 standard deviations span the JPEG range 127,
# but never so tightly that an outlier could overflow an int16 coefficient.
_SIGMAS, _RANGE, _OUTLIER = 4.0, 127.0, 25.0


def quant_table(quality: int, table: str = "flat") -> list[int]:
    """Quality 1..100 -> 64 steps (row-major), using the IJG scaling rule."""
    if not 1 <= quality <= 100:
        raise ValueError("quality must be in 1..100")
    if table not in TABLES:
        raise ValueError(f"table must be one of {TABLES}")
    base = _JPEG_LUMA if table == "jpeg" else np.full(64, 16, dtype=np.int64)
    scale = 5000 // quality if quality < 50 else 200 - 2 * quality
    return np.clip((base * scale + 50) // 100, 1, 255).tolist()


def dct_eligible(array: np.ndarray) -> bool:
    return (array.dtype.kind == "f" and array.ndim >= 2 and array.shape[0] >= 8
            and array.size // array.shape[0] >= 8 and bool(np.isfinite(array).all()))


def _transform(plane: np.ndarray, inverse: bool) -> np.ndarray:
    height, width = plane.shape
    d = _DCT.T if inverse else _DCT
    rows = np.matmul(d, plane.reshape(height // 8, 8, width))
    return np.matmul(rows.reshape(height, width // 8, 8), d.T).reshape(height, width)


def encode_dct(array: np.ndarray, table: list[int]) -> tuple[bytes, dict]:
    plane = np.ascontiguousarray(array, dtype=np.float32).reshape(array.shape[0], -1)
    height, width = plane.shape
    scale = max(_SIGMAS * float(plane.std()), float(np.abs(plane).max()) / _OUTLIER) / _RANGE or 1.0
    plane = np.pad(plane / np.float32(scale), ((0, -height % 8), (0, -width % 8)), mode="edge")
    padded = plane.shape
    steps = np.asarray(table, dtype=np.float32).reshape(1, 8, 1, 8)
    blocks = _transform(plane, False).reshape(padded[0] // 8, 8, padded[1] // 8, 8)
    levels = np.rint(blocks / steps)
    dtype = "int8" if np.abs(levels).max() <= 127 else "int16"
    planes = levels.transpose(1, 3, 0, 2).reshape(64, -1)[_ZIGZAG]
    payload = zlib.compress(planes.astype("<i1" if dtype == "int8" else "<i2").tobytes(), 6)
    return payload, {"scale": scale, "levels": dtype}


def decode_dct(payload: bytes, params: dict, table: list[int], shape, dtype) -> np.ndarray:
    height = shape[0]
    width = int(np.prod(shape[1:]))
    padded = (height + -height % 8, width + -width % 8)
    code = "<i1" if params["levels"] == "int8" else "<i2"
    zigzag = np.frombuffer(zlib.decompress(payload), dtype=code).reshape(64, -1)
    planes = np.empty(zigzag.shape, dtype=np.float32)
    planes[_ZIGZAG] = zigzag
    blocks = planes.reshape(8, 8, padded[0] // 8, padded[1] // 8).transpose(2, 0, 3, 1)
    blocks = blocks * np.asarray(table, dtype=np.float32).reshape(1, 8, 1, 8)
    plane = _transform(np.ascontiguousarray(blocks).reshape(padded), True)
    plane = plane[:height, :width] * np.float32(params["scale"])
    return plane.reshape(shape).astype(dtype)
