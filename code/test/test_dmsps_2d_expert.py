"""Test UNetCCT2D+DMSPS checkpoints trained on the expert-scribble ACDC/
MSCMR archive, on that same archive's own held-out test patients/subjects.

Mirrors ``test_dmsps_2d.py``; only the dataset source differs -- see
``test_pce_2d_expert.py``'s module docstring for the full rationale
(pixel-unit HD95/ASSD, no ``--save_predictions``).
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from networks.unet_2d import UNetCCT2D  # noqa: E402
from train.common_2d import predict_volume_2d  # noqa: E402
from train.train_pce_2d_expert import build_test_dataset, resolve_test_indices  # noqa: E402

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
    parser = argparse.ArgumentParser(description="Evaluate a UNetCCT2D+DMSPS expert-scribble checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--root_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--case_limit", type=int, default=None)
    return parser.parse_args()


def evaluate(args):
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("schema_version") != 1:
        raise ValueError("checkpoint is not a schema-v1 checkpoint")
    if checkpoint.get("model_name") != "unet_cct_2d":
        raise ValueError("checkpoint was not produced by train_dmsps_2d_expert.py")
    model_config = checkpoint["model_config"]
    data_config = checkpoint["data_config"]
    dataset_name = data_config["dataset"]
    if dataset_name not in ("ACDC", "MSCMR"):
        raise ValueError("test_dmsps_2d_expert.py only evaluates ACDC/MSCMR checkpoints")
    if data_config.get("data_source") != "expert_scribble":
        raise ValueError(
            "checkpoint was not trained by train_dmsps_2d_expert.py (data_config.data_source != "
            "'expert_scribble'); use test_dmsps_2d.py for a ScribbleBench checkpoint"
        )
    patch_size = tuple(data_config["patch_size_hw"])
    dataset = build_test_dataset(dataset_name, root_path=args.root_path)
    test_indices = resolve_test_indices(dataset.cases, dataset_name)

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

    output_dir = Path(args.output_dir or REPO_ROOT / "results" / "ExpertScribble_DMSPS" / dataset_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    count = len(test_indices) if args.case_limit is None else min(args.case_limit, len(test_indices))
    cases = []
    for index in tqdm(test_indices[:count], desc="{} (expert) test".format(dataset_name)):
        sample = dataset[index]
        prediction = predict_volume_2d(model, sample["image"], patch_size, device, use_amp=use_amp)
        target = sample["gt_label"]
        spacing = tuple(sample["spacing"].tolist())
        cases.append(summarize_case(sample["case"], prediction, target, dataset.num_classes, spacing))

    summary = aggregate_summary(cases, dataset.num_classes)
    payload = {
        "checkpoint": str(checkpoint_path),
        "dataset": dataset_name,
        "data_source": "expert_scribble",
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
        "Expert-scribble mean Dice: {:.6f} | HD95: {:.4f} px | ASSD: {:.4f} px".format(
            summary["scribblebench_mean_dice"], summary["scribblebench_mean_hd95"], summary["scribblebench_mean_assd"]
        )
    )
    if any(summary["per_class_missed_cases"].values()):
        print("Missed cases (excluded from HD95/ASSD mean): {}".format(summary["per_class_missed_cases"]))
    print("Metrics: {}".format(output_dir / "metrics.json"))
    return summary


if __name__ == "__main__":
    evaluate(parse_args())
