"""Append one hyperparameter-sensitivity row (test-set Dice/HD95/ASSD) to a
summary CSV.

``run_hyperparam_sensitivity.sh``'s own CSV
(``hparam_sensitivity_summary.csv``) reports mean Dice on the held-out
VALIDATION split, deliberately -- see that script's header comment and
``test_voxtrust3d_ablation_2d.py``'s module docstring: PL-Acc/PL-Cov need
real scribble geometry, which only ``imagesTr``'s held-out split carries.
Once a (rho, B) cell's checkpoint exists, its headline Dice/HD95/ASSD can
also be measured on the official, never-touched-during-training TEST split
(``imagesTs``/``labelsTs``) via ``test_pce_2d.py`` -- the same evaluator
every other ScribbleBench baseline reports Dice/HD95 from. This script
takes one such ``test_pce_2d.py`` metrics.json and appends one row --
run by ``code/train/run_hyperparam_sensitivity_test_eval.sh``. Kept as a
SEPARATE CSV from ``hparam_sensitivity_summary.csv`` (test-set numbers vs.
validation-set numbers must not be merged blindly).
"""

import argparse
import csv
import datetime
import json
from pathlib import Path

FIELDNAMES = [
    "timestamp",
    "dataset",
    "target_precision",
    "distance_strata",
    "eval_target",
    "dice_test_pct",
    "hd95_test",
    "assd_test",
    "num_test_cases",
    "checkpoint",
    "test_metrics_json",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test_metrics_json", required=True, help="metrics.json written by test_pce_2d.py (test split)")
    parser.add_argument("--csv", required=True, help="results CSV to append a row to (created if missing)")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--target_precision", required=True, help="rho used to train this checkpoint")
    parser.add_argument("--distance_strata", required=True, help="B used to train this checkpoint")
    parser.add_argument("--checkpoint", required=True)
    return parser.parse_args()


def append_row(args):
    test_metrics_path = Path(args.test_metrics_json).resolve()
    with test_metrics_path.open("r", encoding="utf-8") as handle:
        test_payload = json.load(handle)
    test_summary = test_payload["summary"]

    row = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "dataset": args.dataset,
        "target_precision": args.target_precision,
        "distance_strata": args.distance_strata,
        "eval_target": test_payload.get("eval_target"),
        "dice_test_pct": round(test_summary["scribblebench_mean_dice"] * 100, 4),
        "hd95_test": test_summary.get("scribblebench_mean_hd95"),
        "assd_test": test_summary.get("scribblebench_mean_assd"),
        "num_test_cases": len(test_payload.get("cases", [])),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "test_metrics_json": str(test_metrics_path),
    }

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.is_file() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print("Appended {}/rho={}/B={} -> {}".format(args.dataset, args.target_precision, args.distance_strata, csv_path))
    return row


if __name__ == "__main__":
    append_row(parse_args())
