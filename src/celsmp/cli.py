"""``python -m celsmp``: a thin command line over the library."""

from __future__ import annotations

import argparse
import json
import sys

import celsmp
from celsmp import codec


def _value(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def pack(args) -> None:
    metrics = {key: getattr(args, key) for key in ("train_loss", "val_loss", "train_perplexity", "val_perplexity")
               if getattr(args, key) is not None}
    for item in args.metric:
        key, _, value = item.partition("=")
        metrics[key] = _value(value)
    output = celsmp.pack(
        args.checkpoint, args.output, name=args.name, architecture=args.architecture, sft=args.sft,
        stage=args.stage, dataset=args.dataset, run_dir=args.run_dir, metrics=metrics,
        with_optimizer=args.with_optimizer, codec=args.codec, quality=args.quality, table=args.table,
        notes=args.notes, source_hash=not args.no_source_hash, unsafe_pickle=args.unsafe_pickle)
    print(celsmp.describe(output))


def info(args) -> None:
    if not args.json:
        print(celsmp.describe(args.file, tensors=args.tensors, config=args.config))
        return
    header = celsmp.read_header(args.file)
    header.pop("sections")
    if not args.tensors:
        header.pop("tensors")
    print(json.dumps(header, indent=2, ensure_ascii=False))


def verify(args) -> None:
    header = celsmp.verify(args.file)
    print(f"ok: header and {len(header['tensors'])} tensors match their sha256")


def unpack(args) -> None:
    print(f"wrote {celsmp.unpack(args.file, args.output)}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="celsmp", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("pack", help="produce a .csmp from a PyTorch checkpoint")
    p.add_argument("checkpoint")
    p.add_argument("-o", "--output", help="default: checkpoint path with .csmp suffix")
    p.add_argument("--name", help="model name")
    p.add_argument("--architecture")
    p.add_argument("--sft", action=argparse.BooleanOptionalAction, default=False,
                   help="the weights went through supervised fine-tuning (default: no)")
    p.add_argument("--stage", help="training stage (default: taken from the checkpoint)")
    p.add_argument("--dataset")
    p.add_argument("--run-dir", help="CelLM-Civ run directory; reads the last metrics.jsonl/eval.jsonl records")
    p.add_argument("--train-loss", type=float)
    p.add_argument("--val-loss", type=float)
    p.add_argument("--train-perplexity", type=float, help="default: exp(train loss)")
    p.add_argument("--val-perplexity", type=float, help="default: exp(val loss)")
    p.add_argument("--metric", action="append", default=[], metavar="KEY=VALUE", help="extra metric (repeatable)")
    p.add_argument("--with-optimizer", action="store_true", help="also store the optimizer state")
    p.add_argument("--codec", choices=codec.CODECS, default="raw",
                   help="raw (default), zlib (lossless) or dct (lossy, JPEG-style, weights only)")
    p.add_argument("--quality", type=int, default=90, help="dct quality 1..100 (default 90)")
    p.add_argument("--table", choices=codec.TABLES, default="flat",
                   help="dct quantization table: flat (default, same step for every frequency) or jpeg (luminance)")
    p.add_argument("--notes")
    p.add_argument("--no-source-hash", action="store_true", help="skip hashing the input checkpoint")
    p.add_argument("--unsafe-pickle", action="store_true", help="allow checkpoints that need full unpickling")
    p.set_defaults(run=pack)

    p = commands.add_parser("info", help="show what a .csmp contains")
    p.add_argument("file")
    p.add_argument("--json", action="store_true", help="print the header as JSON")
    p.add_argument("--tensors", action="store_true", help="list every tensor")
    p.add_argument("--config", action="store_true", help="list the model configuration")
    p.set_defaults(run=info)

    p = commands.add_parser("verify", help="check every checksum")
    p.add_argument("file")
    p.set_defaults(run=verify)

    p = commands.add_parser("unpack", help="convert a .csmp back to a .pt checkpoint")
    p.add_argument("file")
    p.add_argument("-o", "--output")
    p.set_defaults(run=unpack)

    args = parser.parse_args(argv)
    try:
        args.run(args)
    except (celsmp.CSMPError, FileNotFoundError, ValueError) as error:
        sys.exit(f"error: {error}")


if __name__ == "__main__":
    main()
