#!/usr/bin/env python3
import argparse
from pathlib import Path
from pprint import pprint

import torch

SERVE_RES = Path(__file__).parent / "SERVE_RES"


def main():
    parser = argparse.ArgumentParser(description="Print .pt files in SERVE_RES/")
    parser.add_argument(
        "file",
        nargs="?",
        help="Specific .pt file to print (name or path). If omitted, prints all.",
    )
    args = parser.parse_args()

    if args.file:
        p = Path(args.file)
        if not p.exists():
            p = SERVE_RES / args.file
        paths = [p]
    else:
        paths = sorted(SERVE_RES.glob("*.pt"))

    for path in paths:
        print(f"===== {path.name} =====")
        data = torch.load(path, map_location="cpu", weights_only=False)
        print(f"type: {type(data).__name__}")
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: Tensor shape={tuple(v.shape)} dtype={v.dtype}")
                elif isinstance(v, (list, tuple)):
                    print(f"  {k}: {type(v).__name__} len={len(v)}")
                    pprint(v, indent=4, depth=2, compact=True)
                else:
                    print(f"  {k}: {v!r}")
        elif isinstance(data, torch.Tensor):
            print(f"shape={tuple(data.shape)} dtype={data.dtype}")
            print(data)
        else:
            pprint(data, depth=3, compact=True)
        print()


if __name__ == "__main__":
    main()
