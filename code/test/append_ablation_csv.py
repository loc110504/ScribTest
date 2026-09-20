"""Append one test_voxtrust3d_ablation_2d.py ``metrics.json`` as a row of a
running Table 2 ("Ablating Trust Calibration", ``paper_icassp2027/main.tex``)
results CSV.

Used by ``code/train/run_voxtrust3d_dcc_full_ablation_2d.sh`` so every
configuration x dataset cell lands in one append-only, schema-stable CSV
with exactly the three columns Table 2 reports (Dice, PL-Acc, PL-Cov), plus
the dataset/configuration labels and enough provenance (checkpoint,
training_method, eval split) to audit a row later.
"""

import argparse
import csv
import datetime
import json
from pathlib import Path

FIELDNAMES = [
    "timestamp",
    "dataset",
    "configuration",
    "row_kind",
    "training_method",
    "eval_split",
    "dice",
    "pl_acc",
    "pl_cov",
    "num_cases",
    "checkpoint",
    "metrics_json",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics_json", required=True, help="metrics.json written by test_voxtrust3d_ablation_2d.py")
    parser.add_argument("--csv", required=True, help="results CSV to append a row to (created if missing)")
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--configuration", required=True,
        help="Table 2 row label, e.g. 'pCE only', 'DCC (full)', 'w/o Wilson bound (raw k/n)'",
    )
    parser.add_argument("--checkpoint", required=True)
    return parser.parse_args()


def append_row(args):
    metrics_path = Path(args.metrics_json).resolve()
    with metrics_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    row = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "dataset": args.dataset,
        "configuration": args.configuration,
        "row_kind": payload.get("row_kind"),
        "training_method": payload.get("training_method"),
        "eval_split": payload.get("eval_split"),
        "dice": payload.get("mean_dice"),
        "pl_acc": payload.get("pl_acc"),
        "pl_cov": payload.get("pl_cov"),
        "num_cases": payload.get("num_cases"),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "metrics_json": str(metrics_path),
    }

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.is_file() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print("Appended {} / {} -> {}".format(args.dataset, args.configuration, csv_path))
    return row


if __name__ == "__main__":
    append_row(parse_args())
