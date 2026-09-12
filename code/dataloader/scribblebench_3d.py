"""Shared 3D data loading and augmentation for ScribbleBench datasets.

NIfTI arrays are stored as ``(X, Y, Z)``.  This module returns NumPy arrays as
``(D, H, W)`` and PyTorch images as ``(C, D, H, W)``, which is the layout
expected by ``torch.nn.Conv3d``.  Sparse and dense labels are always transformed
together so that they remain spatially aligned.
"""

import random
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset


DATASET_CONFIGS = {
    "ACDC": {
        "class_names": ("background", "RV", "MYO", "LV"),
        "ignore_index": 4,
        "normalization": "zscore_nonzero",
    },
    "MSCMR": {
        "class_names": ("background", "RV", "MYO", "LV"),
        "ignore_index": 4,
        "normalization": "zscore_nonzero",
    },
    # Restricted to the 7 organs DMSPS (Han et al., 2024, Sec. 4.1) reports on
    # for WORD -- liver, spleen, left/right kidney, stomach, gallbladder,
    # pancreas -- rather than all 16 organs in the raw WORD label space.
    "WORD": {
        "class_names": (
            "background",
            "liver",
            "spleen",
            "left_kidney",
            "right_kidney",
            "stomach",
            "gallbladder",
            "pancreas",
        ),
        "ignore_index": 8,
        "normalization": "ct_window",
    },
}

# WORD-v0.1.0's original fixed label ids (1-16) for the 7 organs kept above.
# Everything else on disk (esophagus=7, duodenum=9, colon=10, intestine=11,
# adrenal=12, rectum=13, bladder=14, both femur heads=15/16) is outside this
# benchmark's task and is collapsed away by `_remap_word_label` below.
_WORD_ORIGINAL_TO_REDUCED = {
    0: 0,  # background
    1: 1,  # liver
    2: 2,  # spleen
    3: 3,  # left_kidney
    4: 4,  # right_kidney
    5: 5,  # stomach
    6: 6,  # gallbladder
    8: 7,  # pancreas (original id 8; id 7/esophagus is dropped)
}


def _remap_word_label(label, is_scribble, ignore_index):
    """Collapse WORD's 16-organ label space to DMSPS's 7-organ subset.

    Dense masks (``is_scribble=False``) are exhaustive, so any voxel outside
    the 7 target organs is genuinely "not one of these organs" and becomes
    background. Scribble points (``is_scribble=True``) are sparse, hand-drawn
    annotations; remapping a real organ's scribble to background would
    inject false supervision (the voxel is not background, it is simply an
    organ this benchmark ignores), so those points are dropped to
    ``ignore_index`` instead.
    """
    fallback = ignore_index if is_scribble else 0
    remapped = np.full_like(label, fallback)
    for original_id, new_id in _WORD_ORIGINAL_TO_REDUCED.items():
        remapped[label == original_id] = new_id
    return remapped


def _strip_nii_suffix(path):
    name = Path(path).name
    if not name.endswith(".nii.gz"):
        raise ValueError("Expected a .nii.gz file, got: {}".format(path))
    return name[:-7]


def _as_dhw(array):
    if array.ndim != 3:
        raise ValueError("Expected a 3D NIfTI array, got shape {}".format(array.shape))
    return np.ascontiguousarray(array.transpose(2, 1, 0))


def _normalize_image(image, mode, ct_window=(-125.0, 275.0), eps=1e-8):
    image = image.astype(np.float32, copy=False)
    if mode in (None, "none"):
        return image
    if mode == "zscore":
        return (image - image.mean()) / max(float(image.std()), eps)
    if mode == "zscore_nonzero":
        mask = image != 0
        if not np.any(mask):
            return image
        output = np.zeros_like(image, dtype=np.float32)
        values = image[mask]
        output[mask] = (values - values.mean()) / max(float(values.std()), eps)
        return output
    if mode == "ct_window":
        lower, upper = ct_window
        if upper <= lower:
            raise ValueError("ct_window upper bound must be greater than lower bound")
        image = np.clip(image, lower, upper)
        return (image - lower) / (upper - lower)
    raise ValueError("Unsupported normalization mode: {}".format(mode))


class ScribbleBench3DDataset(Dataset):
    """Load complete 3D NIfTI volumes from ScribbleBench.

    Args:
        dataset_name: One of ``ACDC``, ``MSCMR`` or ``WORD``.
        base_dir: Either the ScribbleBench root or one dataset directory.  If
            omitted, ``<repo>/dataset/ScribbleBench/<dataset_name>`` is used.
        split: ``train`` loads ``imagesTr``; ``test`` loads ``imagesTs``.
            ``val`` is a compatibility alias for ``test`` because the provided
            ScribbleBench layout has no separate validation directory.
        sup_type: For training, ``scribble`` uses ``labelsTr`` and
            ``label``/``dense`` uses ``labelsTr_dense``.
        return_full_label: Also return the corresponding dense label in
            ``gt_label``.  This is useful for monitoring but it must not be used
            as supervision in a scribble-only experiment.
        transform: Callable receiving and returning a sample dictionary.
        normalization: ``zscore_nonzero``, ``zscore``, ``ct_window`` or
            ``none``.  Dataset-specific defaults are used when omitted.
        ct_window: Intensity window used by ``ct_window`` normalization.
        verify_pairs: Check all image/label pairs when the dataset is created.
    """

    def __init__(
        self,
        dataset_name,
        base_dir=None,
        split="train",
        sup_type="scribble",
        return_full_label=False,
        transform=None,
        normalization=None,
        ct_window=(-125.0, 275.0),
        verify_pairs=True,
    ):
        super().__init__()
        dataset_name = dataset_name.upper()
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError("Unsupported ScribbleBench dataset: {}".format(dataset_name))
        if split not in ("train", "val", "test"):
            raise ValueError("split must be one of: train, val, test")
        if sup_type not in ("scribble", "label", "dense"):
            raise ValueError("sup_type must be one of: scribble, label, dense")

        self.dataset_name = dataset_name
        self.split = split
        self.sup_type = sup_type
        self.return_full_label = return_full_label
        self.transform = transform
        self.config = DATASET_CONFIGS[dataset_name]
        self.class_names = self.config["class_names"]
        self.num_classes = len(self.class_names)
        self.ignore_index = self.config["ignore_index"]
        self.normalization = normalization or self.config["normalization"]
        self.ct_window = tuple(float(value) for value in ct_window)
        self.base_dir = self._resolve_base_dir(base_dir)

        image_folder = "imagesTr" if split == "train" else "imagesTs"
        self.image_paths = sorted((self.base_dir / image_folder).glob("*.nii.gz"))
        if not self.image_paths:
            raise FileNotFoundError("No NIfTI images found in {}".format(self.base_dir / image_folder))

        self.samples = [self._build_paths(path) for path in self.image_paths]
        if verify_pairs:
            self._verify_pairs()

    def _resolve_base_dir(self, base_dir):
        if base_dir is None:
            repo_root = Path(__file__).resolve().parents[2]
            path = repo_root / "dataset" / "ScribbleBench" / self.dataset_name
        else:
            path = Path(base_dir).expanduser()
            if not (path / "imagesTr").is_dir():
                path = path / self.dataset_name
        if not (path / "imagesTr").is_dir():
            raise FileNotFoundError(
                "Could not find ScribbleBench {} at {}".format(self.dataset_name, path)
            )
        return path.resolve()

    def _label_candidates(self, image_path, label_folder):
        image_id = _strip_nii_suffix(image_path)
        if not image_id.endswith("_0000"):
            raise ValueError("Image name must end in _0000.nii.gz: {}".format(image_path))
        case_id = image_id[:-5]
        candidates = [case_id + ".nii.gz"]
        if self.dataset_name == "MSCMR":
            if self.split == "train" and case_id.endswith("_DE"):
                candidates.insert(0, case_id[:-3] + ".nii.gz")
            if self.split != "train":
                candidates.insert(0, case_id + "_manual.nii.gz")
        return [self.base_dir / label_folder / name for name in candidates]

    def _find_label(self, image_path, label_folder):
        candidates = self._label_candidates(image_path, label_folder)
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return candidates[0]

    def _build_paths(self, image_path):
        case_id = _strip_nii_suffix(image_path)[:-5]
        if self.split == "train":
            target_folder = "labelsTr" if self.sup_type == "scribble" else "labelsTr_dense"
            label_path = self._find_label(image_path, target_folder)
            dense_path = self._find_label(image_path, "labelsTr_dense")
        else:
            label_path = self._find_label(image_path, "labelsTs")
            dense_path = label_path
        return {
            "case": case_id,
            "image_path": image_path,
            "label_path": label_path,
            "dense_path": dense_path,
        }

    def _verify_pairs(self):
        missing = []
        for sample in self.samples:
            for key in ("label_path", "dense_path"):
                if not sample[key].is_file():
                    missing.append(str(sample[key]))
        if missing:
            preview = "\n".join(missing[:10])
            raise FileNotFoundError("Missing {} paired labels:\n{}".format(len(missing), preview))

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _load_nifti(path):
        nii = nib.load(str(path))
        array = np.asanyarray(nii.dataobj)
        return nii, array

    def __getitem__(self, idx):
        paths = self.samples[idx]
        image_nii, image_xyz = self._load_nifti(paths["image_path"])
        label_nii, label_xyz = self._load_nifti(paths["label_path"])
        if image_xyz.shape != label_xyz.shape:
            raise ValueError(
                "Image/label shape mismatch for {}: {} versus {}".format(
                    paths["case"], image_xyz.shape, label_xyz.shape
                )
            )

        image = _normalize_image(
            _as_dhw(image_xyz), self.normalization, ct_window=self.ct_window
        )
        is_scribble = self.split == "train" and self.sup_type == "scribble"
        label = _as_dhw(label_xyz).astype(np.int64, copy=False)
        if self.dataset_name == "WORD":
            label = _remap_word_label(label, is_scribble, self.ignore_index)
        spacing = tuple(float(value) for value in image_nii.header.get_zooms()[:3][::-1])
        sample = {
            "image": image,
            "label": label,
            "idx": idx,
            "case": paths["case"],
            "spacing": np.asarray(spacing, dtype=np.float32),
            "affine": np.asarray(image_nii.affine, dtype=np.float64),
            "original_shape": np.asarray(image.shape, dtype=np.int64),
            "num_classes": self.num_classes,
            "ignore_index": self.ignore_index,
            "is_scribble": is_scribble,
            "image_path": str(paths["image_path"]),
            "label_path": str(paths["label_path"]),
        }

        if self.return_full_label:
            if paths["dense_path"] == paths["label_path"]:
                sample["gt_label"] = label.copy()
            else:
                _, dense_xyz = self._load_nifti(paths["dense_path"])
                if dense_xyz.shape != image_xyz.shape:
                    raise ValueError(
                        "Image/dense-label shape mismatch for {}".format(paths["case"])
                    )
                dense_label = _as_dhw(dense_xyz).astype(np.int64, copy=False)
                if self.dataset_name == "WORD":
                    dense_label = _remap_word_label(dense_label, is_scribble=False, ignore_index=self.ignore_index)
                sample["gt_label"] = dense_label

        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class Compose3D(object):
    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)
        return sample


def _pad_to_shape(array, output_size, value):
    padding = []
    for current, target in zip(array.shape, output_size):
        total = max(target - current, 0)
        padding.append((total // 2, total - total // 2))
    return np.pad(array, padding, mode="constant", constant_values=value), padding


class RandomCrop3D(object):
    """Crop an aligned 3D patch, preferentially around foreground scribbles."""

    def __init__(self, output_size, foreground_prob=0.5):
        if len(output_size) != 3:
            raise ValueError("output_size must be (D, H, W)")
        self.output_size = tuple(int(value) for value in output_size)
        self.foreground_prob = float(foreground_prob)

    def __call__(self, sample):
        image = sample["image"]
        label = sample["label"]
        label_pad_value = sample["ignore_index"] if sample.get("is_scribble") else 0
        image, padding = _pad_to_shape(image, self.output_size, 0.0)
        label, _ = _pad_to_shape(label, self.output_size, label_pad_value)
        gt_label = sample.get("gt_label")
        if gt_label is not None:
            gt_label, _ = _pad_to_shape(gt_label, self.output_size, 0)

        starts = None
        foreground = np.argwhere((label > 0) & (label < sample["num_classes"]))
        if len(foreground) and random.random() < self.foreground_prob:
            center = foreground[np.random.randint(len(foreground))]
            starts = []
            for axis, (current, target) in enumerate(zip(image.shape, self.output_size)):
                low = max(int(center[axis]) - target + 1, 0)
                high = min(int(center[axis]), current - target)
                starts.append(np.random.randint(low, high + 1) if high > low else low)
        if starts is None:
            starts = [
                np.random.randint(0, current - target + 1) if current > target else 0
                for current, target in zip(image.shape, self.output_size)
            ]

        slices = tuple(slice(start, start + size) for start, size in zip(starts, self.output_size))
        sample = dict(sample)
        sample["image"] = np.ascontiguousarray(image[slices])
        sample["label"] = np.ascontiguousarray(label[slices])
        if gt_label is not None:
            sample["gt_label"] = np.ascontiguousarray(gt_label[slices])
        sample["crop_start"] = np.asarray(starts, dtype=np.int64)
        sample["padding"] = np.asarray(padding, dtype=np.int64)
        return sample


class CenterCrop3D(object):
    def __init__(self, output_size):
        if len(output_size) != 3:
            raise ValueError("output_size must be (D, H, W)")
        self.output_size = tuple(int(value) for value in output_size)

    def __call__(self, sample):
        image = sample["image"]
        label_pad_value = sample["ignore_index"] if sample.get("is_scribble") else 0
        image, padding = _pad_to_shape(image, self.output_size, 0.0)
        label, _ = _pad_to_shape(sample["label"], self.output_size, label_pad_value)
        gt_label = sample.get("gt_label")
        if gt_label is not None:
            gt_label, _ = _pad_to_shape(gt_label, self.output_size, 0)
        starts = [(current - target) // 2 for current, target in zip(image.shape, self.output_size)]
        slices = tuple(slice(start, start + size) for start, size in zip(starts, self.output_size))
        sample = dict(sample)
        sample["image"] = np.ascontiguousarray(image[slices])
        sample["label"] = np.ascontiguousarray(label[slices])
        if gt_label is not None:
            sample["gt_label"] = np.ascontiguousarray(gt_label[slices])
        sample["crop_start"] = np.asarray(starts, dtype=np.int64)
        sample["padding"] = np.asarray(padding, dtype=np.int64)
        return sample


class RandomFlipRotate3D(object):
    """Apply label-safe flips and 90-degree in-plane rotations."""

    def __init__(self, flip_prob=0.5, rotate_prob=0.5):
        self.flip_prob = float(flip_prob)
        self.rotate_prob = float(rotate_prob)

    def __call__(self, sample):
        keys = [key for key in ("image", "label", "gt_label") if key in sample]
        arrays = {key: sample[key] for key in keys}
        for axis in range(3):
            if random.random() < self.flip_prob:
                arrays = {key: np.flip(value, axis=axis) for key, value in arrays.items()}
        if random.random() < self.rotate_prob:
            square_plane = arrays["image"].shape[1] == arrays["image"].shape[2]
            choices = (1, 2, 3) if square_plane else (2,)
            k = random.choice(choices)
            arrays = {key: np.rot90(value, k=k, axes=(1, 2)) for key, value in arrays.items()}
        sample = dict(sample)
        for key, value in arrays.items():
            sample[key] = np.ascontiguousarray(value)
        return sample


class Resample3D(object):
    """Optionally resample arrays to ``target_spacing=(D, H, W)``."""

    def __init__(self, target_spacing):
        if len(target_spacing) != 3:
            raise ValueError("target_spacing must be (D, H, W)")
        self.target_spacing = np.asarray(target_spacing, dtype=np.float32)

    def __call__(self, sample):
        spacing = np.asarray(sample["spacing"], dtype=np.float32)
        factors = spacing / self.target_spacing
        sample = dict(sample)
        sample["image"] = zoom(sample["image"], factors, order=3).astype(np.float32)
        sample["label"] = zoom(sample["label"], factors, order=0).astype(np.int64)
        if "gt_label" in sample:
            sample["gt_label"] = zoom(sample["gt_label"], factors, order=0).astype(np.int64)
        sample["spacing"] = self.target_spacing.copy()
        return sample


class ToTensor3D(object):
    def __call__(self, sample):
        sample = dict(sample)
        image = np.ascontiguousarray(sample["image"], dtype=np.float32)
        label = np.ascontiguousarray(sample["label"], dtype=np.int64)
        sample["image"] = torch.from_numpy(image).unsqueeze(0)
        sample["label"] = torch.from_numpy(label).long()
        if "gt_label" in sample:
            gt_label = np.ascontiguousarray(sample["gt_label"], dtype=np.int64)
            sample["gt_label"] = torch.from_numpy(gt_label).long()
        return sample


class RandomGenerator3D(object):
    """Compatibility transform: crop, augment and convert a 3D sample."""

    def __init__(self, output_size, foreground_prob=0.5):
        self.transform = Compose3D(
            [
                RandomCrop3D(output_size, foreground_prob=foreground_prob),
                RandomFlipRotate3D(),
                ToTensor3D(),
            ]
        )

    def __call__(self, sample):
        return self.transform(sample)
