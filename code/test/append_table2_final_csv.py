"""Append one Table 2 row combining test-set Dice/HD95 with held-out
validation-set PL-Acc/PL-Cov into a single CSV row.

Table 2's own evaluator, ``test_voxtrust3d_ablation_2d.py``, deliberately
scores Dice on the held-out VALIDATION split (see its module docstring):
that split is the only case pool carrying both real scribble geometry and
dense reference labels, which PL-Acc/PL-Cov's accept-rule replay needs.
Once a checkpoint is finalized, its Dice/HD95 headline numbers should
instead come from the official, never-touched-during-training TEST split
(``imagesTs``/``labelsTs``), via ``test_pce_2d.py`` -- the same evaluator
every other ScribbleBench baseline reports Dice/HD95 from. This script
takes one ``test_pce_2d.py`` metrics.json (test-set Dice/HD95/ASSD) and one
``test_voxtrust3d_ablation_2d.py`` metrics.json (val-set PL-Acc/PL-Cov) for
the SAME checkpoint and appends one combined row -- run by
``code/train/run_table2_final_metrics.sh``.
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
    "dice_test",
    "hd95_test",
    "assd_test",
    "num_test_cases",
    "pl_acc_val",
    "pl_cov_val",
    "num_val_cases",
    "checkpoint",
    "test_metrics_json",
    "val_metrics_json",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test_metrics_json", required=True, help="metrics.json written by test_pce_2d.py (test split)")
    parser.add_argument(
        "--val_metrics_json", required=True,
        help="metrics.json written by test_voxtrust3d_ablation_2d.py (held-out validation split)",
    )
    parser.add_argument("--csv", required=True, help="results CSV to append a row to (created if missing)")
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--configuration", required=True,
        help="Table 2 row label, e.g. 'ScribCal (full)', 'w/o Wilson bound (raw k/n)'",
    )
    parser.add_argument("--checkpoint", required=True)
    return parser.parse_args()


def append_row(args):
    test_metrics_path = Path(args.test_metrics_json).resolve()
    with test_metrics_path.open("r", encoding="utf-8") as handle:
        test_payload = json.load(handle)
    test_summary = test_payload["summary"]

    val_metrics_path = Path(args.val_metrics_json).resolve()
    with val_metrics_path.open("r", encoding="utf-8") as handle:
        val_payload = json.load(handle)

    row = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "dataset": args.dataset,
        "configuration": args.configuration,
        "row_kind": val_payload.get("row_kind"),
        "training_method": val_payload.get("training_method"),
        "dice_test": test_summary["scribblebench_mean_dice"],
        "hd95_test": test_summary.get("scribblebench_mean_hd95"),
        "assd_test": test_summary.get("scribblebench_mean_assd"),
        "num_test_cases": len(test_payload.get("cases", [])),
        "pl_acc_val": val_payload.get("pl_acc"),
        "pl_cov_val": val_payload.get("pl_cov"),
        "num_val_cases": val_payload.get("num_cases"),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "test_metrics_json": str(test_metrics_path),
        "val_metrics_json": str(val_metrics_path),
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
