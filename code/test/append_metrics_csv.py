"""Append one evaluator ``metrics.json`` as a row of a running results CSV.

Used by ``code/train/run_baselines.sh`` after every train+test cycle so all
CycleMix/DMSPS results across datasets land in a single, append-only,
schema-stable CSV (per-class Dice varies in width across datasets -- ACDC/
MSCMR have 3 foreground classes, WORD has 16 -- so it is kept as one JSON
string column rather than dynamic per-class columns).
"""

import argparse
import csv
import datetime
import json
from pathlib import Path

FIELDNAMES = [
    "timestamp",
    "method",
    "dataset",
    "stage",
    "checkpoint",
    "metrics_json",
    "num_cases",
    "mean_dice",
    "mean_hd95",
    "mean_assd",
    "per_class_dice",
    "per_class_hd95",
    "per_class_assd",
    "per_class_missed_cases",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics_json", required=True, help="metrics.json written by a test_*.py evaluator")
    parser.add_argument("--csv", required=True, help="results CSV to append a row to (created if missing)")
    parser.add_argument("--method", required=True, help="e.g. CycleMix, DMSPS")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--stage", default="", help="e.g. stage1/stage2 for DMSPS; empty for CycleMix")
    parser.add_argument("--checkpoint", required=True)
    return parser.parse_args()


def append_row(args):
    metrics_path = Path(args.metrics_json).resolve()
    with metrics_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    summary = payload["summary"]

    row = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "method": args.method,
        "dataset": args.dataset,
        "stage": args.stage,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "metrics_json": str(metrics_path),
        "num_cases": summary["num_cases"],
        "mean_dice": summary["scribblebench_mean_dice"],
        "mean_hd95": summary.get("scribblebench_mean_hd95"),
        "mean_assd": summary.get("scribblebench_mean_assd"),
        "per_class_dice": json.dumps(summary["per_class_dice"]),
        "per_class_hd95": json.dumps(summary.get("per_class_hd95", {})),
        "per_class_assd": json.dumps(summary.get("per_class_assd", {})),
        "per_class_missed_cases": json.dumps(summary.get("per_class_missed_cases", {})),
    }

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.is_file() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print("Appended {} / {} / {} -> {}".format(args.method, args.dataset, args.stage or "-", csv_path))
    return row


if __name__ == "__main__":
    append_row(parse_args())
