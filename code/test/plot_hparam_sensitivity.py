"""Line-chart figure for the rho/B hyperparameter-sensitivity sweep
(``run_hyperparam_sensitivity.sh`` / ``run_hyperparam_sensitivity_test_eval.sh``).

Reads the sweep's summary CSV and draws a 1x2 figure: mean Dice (left) and
mean HD95 (right) on the y-axis, target precision rho on the x-axis, one
line per distance strata B -- the two panels ScribCal's hyperparameter-
sensitivity discussion in ``paper_icassp2027/main.tex`` needs side by side.

Two summary CSVs can feed this script:

- ``hparam_sensitivity_test_summary.csv`` (from
  ``run_hyperparam_sensitivity_test_eval.sh``, official TEST split):
  columns ``dice_test_pct``/``hd95_test`` -- has both metrics, so both
  panels are drawn. This is the default/expected input.
- ``hparam_sensitivity_summary.csv`` (from ``run_hyperparam_sensitivity.sh``
  itself, held-out VALIDATION split): only ``mean_dice_pct``, no HD95 --
  the right panel is skipped with a warning if this file is passed instead.

If the CSV contains rows for more than one dataset, one figure is saved
per dataset (``<output>_<dataset>.png``).

Usage:
    python code/test/plot_hparam_sensitivity.py \\
        --csv results/hparam_sensitivity_test_summary.csv \\
        --output figures/hparam_sensitivity
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DICE_COLUMN_CANDIDATES = ("dice_test_pct", "mean_dice_pct")
HD95_COLUMN_CANDIDATES = ("hd95_test",)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="summary CSV (see module docstring for expected columns)")
    parser.add_argument("--output", default=None, help="output path prefix (no extension); default: alongside --csv")
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def resolve_column(frame, candidates, required_for):
    for name in candidates:
        if name in frame.columns:
            return name
    raise KeyError(
        "CSV has none of {} (needed for {}); columns present: {}".format(
            candidates, required_for, list(frame.columns)
        )
    )


def plot_dataset(frame, dataset, dice_col, hd95_col, output_prefix, dpi):
    subset = frame[frame["dataset"] == dataset].copy()
    subset["target_precision"] = subset["target_precision"].astype(float)
    subset["distance_strata"] = subset["distance_strata"].astype(int)
    strata_values = sorted(subset["distance_strata"].unique())

    has_hd95 = hd95_col is not None and subset[hd95_col].notna().any()
    num_panels = 2 if has_hd95 else 1
    fig, axes = plt.subplots(1, num_panels, figsize=(5.2 * num_panels, 4.0))
    if num_panels == 1:
        axes = [axes]

    markers = ["o", "s", "^", "D", "v", "P"]
    for index, strata in enumerate(strata_values):
        line = subset[subset["distance_strata"] == strata].sort_values("target_precision")
        marker = markers[index % len(markers)]
        axes[0].plot(
            line["target_precision"], line[dice_col],
            marker=marker, label="B = {}".format(strata),
        )
        if has_hd95:
            axes[1].plot(
                line["target_precision"], line[hd95_col],
                marker=marker, label="B = {}".format(strata),
            )

    axes[0].set_xlabel(r"Target precision $\rho$")
    axes[0].set_ylabel("Mean Dice (%)")
    axes[0].set_title("{}: Dice vs. $\\rho$".format(dataset))
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(title="Distance strata")

    if has_hd95:
        axes[1].set_xlabel(r"Target precision $\rho$")
        axes[1].set_ylabel("Mean HD95 (mm)")
        axes[1].set_title("{}: HD95 vs. $\\rho$".format(dataset))
        axes[1].grid(True, linestyle="--", alpha=0.4)
        axes[1].legend(title="Distance strata")
    else:
        print(
            "Warning: no usable HD95 column for dataset={} -- skipping HD95 panel "
            "(pass the test-set CSV from run_hyperparam_sensitivity_test_eval.sh for both panels)".format(dataset)
        )

    fig.tight_layout()
    output_path = Path("{}_{}.png".format(output_prefix, dataset))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print("Saved {}".format(output_path))


def main():
    args = parse_args()
    csv_path = Path(args.csv).resolve()
    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise ValueError("CSV {} has no rows".format(csv_path))

    dice_col = resolve_column(frame, DICE_COLUMN_CANDIDATES, "Dice panel")
    try:
        hd95_col = resolve_column(frame, HD95_COLUMN_CANDIDATES, "HD95 panel")
    except KeyError:
        hd95_col = None

    output_prefix = args.output or str(csv_path.with_suffix(""))
    for dataset in sorted(frame["dataset"].unique()):
        plot_dataset(frame, dataset, dice_col, hd95_col, output_prefix, args.dpi)


if __name__ == "__main__":
    main()
