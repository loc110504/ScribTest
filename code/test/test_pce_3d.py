"""Test UNet3D+pCE checkpoints on the official ScribbleBench test split."""

import argparse
import json
import math
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm


CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from dataloader.scribblebench_3d import ScribbleBench3DDataset  # noqa: E402
from networks.unet_3d import UNet3D  # noqa: E402
from train.legacy_splits import published_test_groups  # noqa: E402
from utils.sliding_window_3d import sliding_window_predict  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a UNet3D+pCE checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--root_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--sw_batch_size", type=int, default=1)
    parser.add_argument("--max_accumulator_mb", type=int, default=1024)
    parser.add_argument("--temp_dir", default=None)
    parser.add_argument("--case_limit", type=int, default=None)
    parser.add_argument("--save_predictions", action="store_true")
    return parser.parse_args()


def mean_finite(values):
    values = [value for value in values if math.isfinite(value)]
    return float(np.mean(values)) if values else math.nan


def dice(prediction, target, class_id):
    pred = prediction == class_id
    truth = target == class_id
    if not pred.any() and not truth.any():
        return math.nan
    if not pred.any() or not truth.any():
        return 0.0
    return float(2 * np.count_nonzero(pred & truth) / (pred.sum() + truth.sum()))


def save_prediction(prediction_dhw, image_path, output_path):
    source = nib.load(str(image_path))
    prediction_xyz = np.ascontiguousarray(prediction_dhw.transpose(2, 1, 0))
    header = source.header.copy()
    header.set_data_dtype(prediction_xyz.dtype)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(prediction_xyz, source.affine, header), str(output_path)
    )


def evaluate(args):
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("schema_version") != 1:
        raise ValueError("checkpoint is not a UNet3D+pCE schema-v1 checkpoint")
    model_config = checkpoint["model_config"]
    data_config = checkpoint["data_config"]
    dataset_name = data_config["dataset"]
    patch_size = tuple(data_config["patch_size_dhw"])
    dataset = ScribbleBench3DDataset(
        dataset_name,
        base_dir=args.root_path,
        split="test",
        sup_type="dense",
    )
    def group_name(case):
        if dataset_name == "ACDC":
            return case.split("_")[0]
        if dataset_name == "MSCMR":
            return case.removesuffix("_DE")
        return case

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
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    use_amp = args.amp and device.type == "cuda"
    model = UNet3D(
        in_chns=model_config["in_chns"],
        class_num=model_config["class_num"],
        feature_chns=tuple(model_config["feature_chns"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    output_dir = Path(
        args.output_dir
        or REPO_ROOT / "results" / "ScribbleBench_pCE" / dataset_name
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    count = len(dataset) if args.case_limit is None else min(args.case_limit, len(dataset))
    cases = []
    for index in tqdm(range(count), desc="{} test".format(dataset_name)):
        sample = dataset[index]
        image = torch.from_numpy(sample["image"]).unsqueeze(0).unsqueeze(0).float()
        prediction = sliding_window_predict(
            model,
            image,
            dataset.num_classes,
            patch_size,
            device,
            overlap=args.overlap,
            sw_batch_size=args.sw_batch_size,
            use_amp=use_amp,
            max_accumulator_mb=args.max_accumulator_mb,
            temp_dir=args.temp_dir,
        )
        target = sample["label"]
        per_class = {
            str(class_id): dice(prediction, target, class_id)
            for class_id in range(1, dataset.num_classes)
        }
        cases.append(
            {
                "case": sample["case"],
                "mean_foreground_dice": mean_finite(per_class.values()),
                "per_class_dice": per_class,
            }
        )
        if args.save_predictions:
            save_prediction(
                prediction,
                sample["image_path"],
                output_dir / "predictions" / "{}.nii.gz".format(sample["case"]),
            )

    summary = {
        "scribblebench_mean_dice": mean_finite(
            case["mean_foreground_dice"] for case in cases
        ),
        "per_class_dice": {
            str(class_id): mean_finite(
                case["per_class_dice"][str(class_id)] for case in cases
            )
            for class_id in range(1, dataset.num_classes)
        },
        "num_cases": len(cases),
    }
    payload = {
        "checkpoint": str(checkpoint_path),
        "dataset": dataset_name,
        "patch_size_dhw": list(patch_size),
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
    print("ScribbleBench mean Dice: {:.6f}".format(summary["scribblebench_mean_dice"]))
    print("Metrics: {}".format(output_dir / "metrics.json"))
    return summary


if __name__ == "__main__":
    evaluate(parse_args())
