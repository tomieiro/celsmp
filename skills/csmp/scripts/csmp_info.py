#!/usr/bin/env python3
"""Summarize .csmp files from their headers, without reading any tensor.

    python csmp_info.py a.csmp b.csmp     # one line per file
    python csmp_info.py --full a.csmp     # the complete summary
    python csmp_info.py --json a.csmp     # model, training, metrics, compression as JSON
"""

import argparse
import json
import os

import celsmp


def line(path: str) -> str:
    header = celsmp.read_header(path)
    training, metrics, compression = header["training"], header["metrics"], header["compression"]

    def show(value):
        return "n/a" if value is None else f"{value:.4f}"

    codec = compression["codec"]
    if compression["lossy"]:
        codec += f" q{compression['quality']} LOSSY rmse={compression['rel_rmse']:.3f}"
    return (f"{path}: {header['model'].get('name') or header['model'].get('architecture')}"
            f" | sft={'yes' if training['sft'] else 'no'} stage={training.get('stage', 'n/a')}"
            f" step={training.get('step', 'n/a')}"
            f" | train_loss={show(metrics['train_loss'])} val_loss={show(metrics['val_loss'])}"
            f" val_ppl={show(metrics['val_perplexity'])}"
            f" | {codec} {os.path.getsize(path) / 2 ** 20:.0f} MiB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    for path in args.files:
        if args.json:
            header = celsmp.read_header(path)
            header["model"].pop("config", None)
            print(json.dumps({"file": path, **{k: header[k] for k in ("model", "training", "metrics", "compression")}}))
        else:
            print(celsmp.describe(path) if args.full else line(path))


if __name__ == "__main__":
    main()
