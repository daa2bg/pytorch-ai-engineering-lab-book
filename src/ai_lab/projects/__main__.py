"""Bounded CLI: python -m ai_lab.projects {train,predict}."""
import argparse
import json
from pathlib import Path

from .sms import (evaluate_sms, load_sms_artifact, load_sms_records, make_sms_splits,
                  predict_sms, save_sms_artifact, train_sms_model)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train or restore the public SMS teaching model")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train 800 rows; validate 200; report test 200 once")
    train.add_argument("--output", required=True, type=Path, help="New version directory")
    train.add_argument("--sms-path", type=Path)
    train.add_argument("--allow-downloads", action="store_true", default=None)
    train.add_argument("--cache-dir", type=Path)
    train.add_argument("--epochs", type=int, default=8)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--threads", type=int, default=1)
    predict = commands.add_parser("predict", help="Fresh-process inference; no data or network needed")
    predict.add_argument("--artifact", required=True, type=Path)
    predict.add_argument("--text", action="append", required=True, help="Repeat for a batch")
    predict.add_argument("--review-threshold", type=float, default=None)
    args = parser.parse_args(argv)
    if args.command == "train":
        import torch
        if not 1 <= args.threads <= 16:
            parser.error("--threads must be 1..16")
        torch.set_num_threads(args.threads)
        dataset = load_sms_records(args.sms_path, allow_downloads=args.allow_downloads,
                                   cache_dir=args.cache_dir)
        splits = make_sms_splits(dataset, seed=args.seed)
        artifact = train_sms_model(splits, epochs=args.epochs, seed=args.seed)
        artifact.metadata["test"] = evaluate_sms(artifact, splits.test)
        save_sms_artifact(artifact, args.output)
        result = {"artifact": str(args.output), "artifact_id": artifact.metadata["artifact_id"],
                  "source": dataset.metadata, "split_counts": splits.metadata["class_counts"],
                  "baseline": artifact.metadata["baseline"],
                  "validation": artifact.metadata["validation"], "test": artifact.metadata["test"]}
    else:
        result = predict_sms(load_sms_artifact(args.artifact), args.text,
                             review_threshold=args.review_threshold)
    print(json.dumps(result, ensure_ascii=True, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
