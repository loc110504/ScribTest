"""Test UNetCCT2D+DMPLS checkpoints on the official ScribbleBench ACDC/MSCMR
test split.

Mirrors ``test_dmsps_2d.py``/``test_pce_2d.py``; only the model class and its
main-decoder-only wrapper differ, since a DMPLS checkpoint stores a
dual-decoder DB-Net instead of a plain ``UNet2D``. The paper's own testing
protocol uses the primary (main) decoder's output only.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from dataloader.scribblebench_3d import ScribbleBench3DDataset  # noqa: E402
from networks.unet_2d import UNetCCT2D  # noqa: E402
from train.common_2d import predict_volume_2d  # noqa: E402
from train.legacy_splits import published_test_groups  # noqa: E402

# Python auto-adds this script's own directory (code/test/) to sys.path, so
# the sibling module resolves as a bare import (avoids clashing with the
# stdlib `test` package that `from test.metrics_3d import ...` would hit).
from metrics_3d import aggregate_summary, summarize_case  # noqa: E402


class MainDecoderOnly(nn.Module):
    """Drops the auxiliary decoder at inference time; same logits either way."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x, return_auxiliary=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a UNetCCT2D+DMPLS checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--root_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--case_limit", type=int, default=None)
    parser.add_argument("--save_predictions", action="store_true")
    return parser.parse_args()


def save_prediction(prediction_dhw, image_path, output_path):
    source = nib.load(str(image_path))
    prediction_xyz = np.ascontiguousarray(prediction_dhw.transpose(2, 1, 0))
    header = source.header.copy()
    header.set_data_dtype(prediction_xyz.dtype)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(prediction_xyz, source.affine, header), str(output_path))


def evaluate(args):
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("schema_version") != 1:
        raise ValueError("checkpoint is not a schema-v1 checkpoint")
    if checkpoint.get("model_name") != "unet_cct_2d":
        raise ValueError("checkpoint was not produced by train_dmpls_2d.py")
    model_config = checkpoint["model_config"]
    data_config = checkpoint["data_config"]
    dataset_name = data_config["dataset"]
    if dataset_name not in ("ACDC", "MSCMR"):
        raise ValueError("test_dmpls_2d.py only evaluates ACDC/MSCMR checkpoints")
    patch_size = tuple(data_config["patch_size_hw"])
    dataset = ScribbleBench3DDataset(dataset_name, base_dir=args.root_path, split="test", sup_type="dense")

    def group_name(case):
        if dataset_name == "ACDC":
            return case.split("_")[0]
        return case.removesuffix("_DE")

    observed_test_groups = {group_name(sample["case"]) for sample in dataset.samples}
    expected_test_groups = set(published_test_groups(dataset_name))
    if observed_test_groups != expected_test_groups:
        raise RuntimeError(
            "Test data does not match the published {} protocol; missing={}, unexpected={}".format(
                dataset_name,
                sorted(expected_test_groups - observed_test_groups),
                sorted(observed_test_groups - expected_test_groups),
            )
        )
    if dataset.num_classes != model_config["class_num"]:
        raise ValueError("checkpoint class count does not match the dataset")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = args.amp and device.type == "cuda"
    dual_model = UNetCCT2D(
        in_chns=model_config["in_chns"],
        class_num=model_config["class_num"],
        feature_chns=tuple(model_config["feature_chns"]),
        perturbations=tuple(model_config.get("perturbations", ["dropout"])),
        perturbation_dropout=model_config.get("perturbation_dropout", 0.5),
    ).to(device)
    dual_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    dual_model.eval()
    model = MainDecoderOnly(dual_model).to(device)
    model.eval()

    output_dir = Path(args.output_dir or REPO_ROOT / "results" / "ScribbleBench_DMPLS" / dataset_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    count = len(dataset) if args.case_limit is None else min(args.case_limit, len(dataset))
    cases = []
    for index in tqdm(range(count), desc="{} test".format(dataset_name)):
        sample = dataset[index]
        prediction = predict_volume_2d(model, sample["image"], patch_size, device, use_amp=use_amp)
        target = sample["label"]
        spacing = tuple(sample["spacing"].tolist())
        cases.append(summarize_case(sample["case"], prediction, target, dataset.num_classes, spacing))
        if args.save_predictions:
            save_prediction(
                prediction, sample["image_path"], output_dir / "predictions" / "{}.nii.gz".format(sample["case"])
            )

    summary = aggregate_summary(cases, dataset.num_classes)
    payload = {
        "checkpoint": str(checkpoint_path),
        "dataset": dataset_name,
        "patch_size_hw": list(patch_size),
        "summary": summary,
        "cases": cases,
    }

    def json_safe(value):
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [json_safe(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, allow_nan=False)
    print(
        "ScribbleBench mean Dice: {:.6f} | HD95: {:.4f} mm | ASSD: {:.4f} mm".format(
            summary["scribblebench_mean_dice"], summary["scribblebench_mean_hd95"], summary["scribblebench_mean_assd"]
        )
    )
    if any(summary["per_class_missed_cases"].values()):
        print("Missed cases (excluded from HD95/ASSD mean): {}".format(summary["per_class_missed_cases"]))
    print("Metrics: {}".format(output_dir / "metrics.json"))
    return summary


if __name__ == "__main__":
    evaluate(parse_args())
