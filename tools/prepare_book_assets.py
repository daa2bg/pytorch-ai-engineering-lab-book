"""Prepare bounded, versioned book model assets; offline unless explicitly opted in."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ai_lab.projects.assets import MAX_BYTES, SPECS, asset_metadata, prepare_assets


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", action="append", choices=tuple(SPECS))
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--allow-downloads", action="store_true", default=None)
    parser.add_argument("--max-bytes", type=int, default=MAX_BYTES)
    args = parser.parse_args(argv)
    paths = prepare_assets(args.asset or tuple(SPECS), cache_dir=args.cache_dir,
                           allow_downloads=args.allow_downloads, max_bytes=args.max_bytes)
    print(json.dumps({kind: {"path": str(path), **asset_metadata(path)} for kind, path in paths.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
